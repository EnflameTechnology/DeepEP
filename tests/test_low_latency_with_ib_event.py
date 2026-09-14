"""Low-latency dispatch/combine benchmark with amortized per-phase GCU-event timing.

This script is a standalone alternative to test_low_latency_with_ib.py.
Instead of using bench_kineto (profiler-based), it uses bench_separate
(GCU events) to measure dispatch and combine latency independently with
whole-loop event timing (amortized by iteration count).

Supports multi-node: uses dist.all_gather to collect every rank's raw
timing data inside the distributed context, so the final aggregation
covers ALL ranks regardless of shared filesystem availability.
"""
import argparse
import random
import os
import json
import datetime
import numpy as np
import torch
import torch.distributed as dist

import deep_ep
from utils import (init_dist, bench, bench_separate, per_token_cast_back,
                   get_imbalance_token_counts, aggregate_bench_separate_results,
                   build_base_parser, add_fp8_dispatch_arg, add_ib_args)

# -----------------------------------------------------------------------
# Predefined sweep lists.
# Edit these values to define the (total_tokens, imbalance_ratio) matrix
# used when --run-mode=list is passed on the command line.
# The cartesian product of the two lists is executed in order.
# -----------------------------------------------------------------------
SWEEP_TOTAL_TOKENS_LIST = [2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 34, 36, 40, 48, 56, 60, 64, 72, 80, 84, 96, 98, 128, 144, 160, 192, 240, 288, 512]
# SWEEP_TOTAL_TOKENS_LIST = [2]

# SWEEP_TOTAL_TOKENS_LIST = [2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 34, 36, 40]
# SWEEP_IMBALANCE_RATIO_LIST = [0]

SWEEP_IMBALANCE_RATIO_LIST = [0, 40, 80, 120, 200]


def create_test_data(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                     rank: int, num_ranks: int, use_logfmt: bool = False, seed: int = 0):
    """Create test data for performance testing."""
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0, 'num_experts must be divisible by num_ranks'

    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * 0.1
    scores = torch.rand((num_tokens, num_experts), dtype=torch.float32, device='gcu').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu').abs()

    return x, topk_idx, topk_weights


def run_performance_test(current_num_tokens: int, max_tokens_per_rank: int, hidden: int,
                         num_experts: int, num_topk: int, rank: int, num_ranks: int,
                         group: dist.ProcessGroup, buffer: deep_ep.Buffer,
                         use_logfmt: bool = False, seed: int = 0, use_fp8: bool = False,
                         result_path_template: str = None):
    """Run performance tests for low-latency dispatch and combine.

    Benchmarks three items:
      1. dispatch + combine total  (bench, for reference)
      2. dispatch only             (bench)
      3. combine only              (bench)
      4. dispatch / combine split  (bench_separate, amortized phase latency)

    Uses dist.all_gather to collect all ranks' phase timing data,
    then global rank 0 writes per-rank JSON files for ALL ranks so that
    post-process aggregation covers every rank regardless of filesystem.
    """
    x, topk_idx, topk_weights = create_test_data(
        current_num_tokens, hidden, num_experts, num_topk, rank, num_ranks, use_logfmt, seed)

    num_local_experts = num_experts // num_ranks
    cumulative_local_expert_recv_stats = torch.zeros((num_local_experts,), dtype=torch.int, device='gcu')

    print(f'[rank {rank}] Running low-latency dispatch + combine performance test with\n'
          f'  max_tokens_per_rank={max_tokens_per_rank}, current_num_tokens={current_num_tokens},\n'
          f'  hidden={hidden}, num_experts={num_experts}, use_fp8={use_fp8}', flush=True)

    # Initial dispatch to obtain handle and simulated GEMM output
    packed_recv_x, packed_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(x, topk_idx, max_tokens_per_rank, num_experts,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    use_fp8=use_fp8, async_finish=True, return_recv_hook=False)
    event.current_stream_wait()

    if use_fp8:
        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous())
        simulated_gemm_x = per_token_cast_back(
            packed_recv_x[0].view(-1, hidden),
            packed_recv_x[1].view(-1, hidden // 128)
        ).view(packed_recv_x[0].shape)
    else:
        simulated_gemm_x = packed_recv_x.clone()

    # ---- bandwidth calculation ----
    num_fp8_bytes = hidden + hidden / 128 * 4 + 16
    num_bf16_bytes = hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(current_num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += (num_fp8_bytes if use_fp8 else num_bf16_bytes) * num_selections
        num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections

    # ---- closure: shared handle between dispatch_fn / combine_fn ----
    _handle_ref = [None]
    _dispatch_event_ref = [None]

    def dispatch_fn():
        recv_x, recv_count, h, event, hook = \
            buffer.low_latency_dispatch(x, topk_idx, max_tokens_per_rank, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=use_fp8, async_finish=True, return_recv_hook=False)
        _handle_ref[0] = h
        _dispatch_event_ref[0] = event

    def combine_fn():
        buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, _handle_ref[0],
                                   use_logfmt=use_logfmt, return_recv_hook=False)

    def dispatch_wait_fn():
        if _dispatch_event_ref[0] is not None:
            _dispatch_event_ref[0].current_stream_wait()

    def combine_setup_fn():
        # Prepare a fresh handle for combine but keep dispatch out of combine timing.
        dispatch_fn()
        dispatch_wait_fn()

    def dispatch_and_combine():
        dispatch_fn()
        combine_fn()

    # ============================================================
    # 1) bench: dispatch + combine total
    # ============================================================
    total_avg, total_min, total_max = bench(dispatch_and_combine)
    print(f'[rank {rank}] Dispatch+Combine total: '
          f'{(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / total_avg:.2f} GB/s, '
          f'avg={total_avg * 1e6:.2f} us, min={total_min * 1e6:.2f} us, max={total_max * 1e6:.2f} us',
          flush=True)

    # ============================================================
    # 2) bench: dispatch only
    # ============================================================
    d_only_avg, d_only_min, d_only_max = bench(dispatch_fn)
    print(f'[rank {rank}] Dispatch only: '
          f'{num_dispatch_comm_bytes / 1e9 / d_only_avg:.2f} GB/s, '
          f'avg={d_only_avg * 1e6:.2f} us, min={d_only_min * 1e6:.2f} us, max={d_only_max * 1e6:.2f} us',
          flush=True)

    # ============================================================
    # 3) bench: combine only (dispatch once to get a valid handle)
    # ============================================================
    dispatch_fn()
    c_only_avg, c_only_min, c_only_max = bench(combine_fn)
    print(f'[rank {rank}] Combine only: '
          f'{num_combine_comm_bytes / 1e9 / c_only_avg:.2f} GB/s, '
          f'avg={c_only_avg * 1e6:.2f} us, min={c_only_min * 1e6:.2f} us, max={c_only_max * 1e6:.2f} us',
          flush=True)

    # ============================================================
    # 4) bench_separate: dispatch / combine split (whole-loop timing, amortized)
    # ============================================================
    dist.barrier()
    phase_times = bench_separate(
        {'dispatch': dispatch_fn, 'combine': combine_fn},
        phase_setup_fns={'combine': combine_setup_fn},
        phase_post_fns={'dispatch': dispatch_wait_fn},
    )
    dispatch_times = phase_times['dispatch']   # numpy array([avg_seconds])
    combine_times = phase_times['combine']

    dispatch_avg = np.average(dispatch_times)
    dispatch_min_t = np.min(dispatch_times)
    dispatch_max_t = np.max(dispatch_times)
    combine_avg = np.average(combine_times)
    combine_min_t = np.min(combine_times)
    combine_max_t = np.max(combine_times)

    dispatch_bandwidth = num_dispatch_comm_bytes / 1e9 / dispatch_avg
    combine_bandwidth = num_combine_comm_bytes / 1e9 / combine_avg
    dispatch_latency_us = dispatch_avg * 1e6
    combine_latency_us = combine_avg * 1e6

    print(f'[rank {rank}] bench_separate Dispatch: {dispatch_bandwidth:.2f} GB/s, '
          f'avg={dispatch_latency_us:.2f} us, min={dispatch_min_t * 1e6:.2f} us, '
          f'max={dispatch_max_t * 1e6:.2f} us | '
          f'Combine: {combine_bandwidth:.2f} GB/s, '
          f'avg={combine_latency_us:.2f} us, min={combine_min_t * 1e6:.2f} us, '
          f'max={combine_max_t * 1e6:.2f} us', flush=True)

    # ---- cross-rank average via all_reduce ----
    stats_tensor = torch.tensor([
        dispatch_latency_us, dispatch_min_t * 1e6, dispatch_max_t * 1e6, dispatch_bandwidth,
        combine_latency_us, combine_min_t * 1e6, combine_max_t * 1e6, combine_bandwidth,
    ], dtype=torch.float64, device='gcu')
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM, group=group)
    stats_tensor /= num_ranks
    torch.gcu.synchronize()

    dispatch_latency_us_mean = stats_tensor[0].item()
    dispatch_min_us_mean = stats_tensor[1].item()
    dispatch_max_us_mean = stats_tensor[2].item()
    dispatch_bandwidth_mean = stats_tensor[3].item()
    combine_latency_us_mean = stats_tensor[4].item()
    combine_min_us_mean = stats_tensor[5].item()
    combine_max_us_mean = stats_tensor[6].item()
    combine_bandwidth_mean = stats_tensor[7].item()

    if rank == 0:
        print(f'[avg across {num_ranks} ranks] '
              f'Dispatch: {dispatch_bandwidth_mean:.2f} GB/s, avg={dispatch_latency_us_mean:.2f} us | '
              f'Combine: {combine_bandwidth_mean:.2f} GB/s, avg={combine_latency_us_mean:.2f} us',
              flush=True)

    results = {
        'dispatch': {
            'latency_us': dispatch_latency_us_mean,
            'latency_us_min': dispatch_min_us_mean,
            'latency_us_max': dispatch_max_us_mean,
            'bandwidth_gbps': dispatch_bandwidth_mean,
        },
        'combine': {
            'latency_us': combine_latency_us_mean,
            'latency_us_min': combine_min_us_mean,
            'latency_us_max': combine_max_us_mean,
            'bandwidth_gbps': combine_bandwidth_mean,
        },
    }

    # ============================================================
    # all_gather: collect every rank's phase timing data for cross-node aggregation
    # ============================================================
    if result_path_template is not None:
        num_iters = len(dispatch_times)
        local_data = torch.tensor(
            np.concatenate([dispatch_times * 1e6, combine_times * 1e6]),
            dtype=torch.float64, device='gcu')

        gathered = [torch.zeros_like(local_data) for _ in range(num_ranks)]
        dist.all_gather(gathered, local_data, group=group)
        torch.gcu.synchronize()

        # Global rank 0 writes per-rank JSONs for ALL ranks (cross-node safe)
        if rank == 0:
            for r in range(num_ranks):
                rpath = result_path_template.format(rank=r)
                rdata = {
                    'rank': r,
                    'num_tests': num_iters,
                    'dispatch_times_us': gathered[r][:num_iters].cpu().tolist(),
                    'combine_times_us': gathered[r][num_iters:].cpu().tolist(),
                }
                with open(rpath, 'w') as f:
                    json.dump(rdata, f, indent=4)
            print(f'[rank 0] Saved per-rank bench results for all {num_ranks} ranks', flush=True)

    return results


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace,
              counts: list[int], result_template: str, log_suffix: str = ''):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_file = f"{rank}_test_low_latency_with_ib_event{log_suffix}.log"

    with open(log_file, 'w') as f:
        import sys
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)

        try:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = f
            sys.stderr = f

            print(f'Initialized rank {rank}/{num_ranks} on local rank {local_rank}', flush=True)

            max_tokens_per_rank = max(counts)
            current_rank_num_tokens = counts[rank]
            hidden = args.hidden
            num_topk, num_experts = args.num_topk, args.num_experts

            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                max_tokens_per_rank, hidden, num_ranks, num_experts)
            if local_rank == 0:
                print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
            buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                                    num_qps_per_rank=num_experts // num_ranks,
                                    allow_nvlink_for_low_latency_mode=True, explicitly_destroy=True,
                                    allow_mnnvl=False)
            try:
                run_performance_test(current_rank_num_tokens, max_tokens_per_rank, hidden,
                                     num_experts, num_topk, rank, num_ranks, group, buffer,
                                     use_logfmt=False, seed=1,
                                     use_fp8=args.dispatch_use_fp8,
                                     result_path_template=result_template)
                print(f'Performance test successfully finished!', flush=True)
            finally:
                buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)


if __name__ == '__main__':
    usage_examples = """Examples:
  # Single run, single node
  python3 test_low_latency_with_ib_event.py \\
    --num-processes 16 --num-tokens 20 --hidden 4096 --num-topk 8 --num-experts 192 \\
    --imbalance-ratio 0 --result-dir ./results --emit-compact-shape-json

  # Single run with FP8 dispatch
  python3 test_low_latency_with_ib_event.py \\
    --num-processes 16 --num-tokens 32 --hidden 4096 --num-topk 8 --num-experts 192 \\
    --imbalance-ratio 80 --dispatch-use-fp8 --result-dir ./results --emit-compact-shape-json

  # List mode: iterate over SWEEP_TOTAL_TOKENS_LIST x SWEEP_IMBALANCE_RATIO_LIST
  # (edit those constants at the top of the script before running)
  python3 test_low_latency_with_ib_event.py \\
    --num-processes 16 --hidden 4096 --num-topk 8 --num-experts 192 \\
    --run-mode list --result-dir ./results --emit-compact-shape-json

  # Multi-node launch example (env vars set by launcher)
  python3 test_low_latency_with_ib_event.py \\
    --num-processes 16 --num-tokens 50 --hidden 4096 --num-topk 8 --num-experts 192 \\
    --imbalance-ratio 120 --result-dir ./results --emit-compact-shape-json
"""
    parser = build_base_parser(
        'Low-latency EP benchmark with separate dispatch/combine timing (GCU events)',
        defaults={'num_experts': 288},
        )
    parser.epilog = usage_examples
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    add_fp8_dispatch_arg(parser)
    add_ib_args(parser, imbalance_default=40, add_run_mode=True,
                total_tokens_help='Total tokens across all ranks. Defaults to num-tokens * num-ranks.')
    parser.add_argument('--result-dir', type=str, default='./',
                        help='Directory for per-rank result JSONs and final output (default: ./)')
    parser.add_argument('--emit-compact-shape-json', action='store_true',
                        help='Also emit compact shape JSON (dispatch/combine latency only)')
    args = parser.parse_args()

    num_local_ranks = args.num_processes
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    num_ranks = num_local_ranks * num_nodes

    print(f'num_nodes={num_nodes}, node_rank={node_rank}, num_local_ranks={num_local_ranks}, '
          f'num_ranks(world_size)={num_ranks}')

    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build the list of (total_tokens, imbalance_ratio) combinations.
    #
    # --run-mode list  : cartesian product of SWEEP_TOTAL_TOKENS_LIST and
    #                    SWEEP_IMBALANCE_RATIO_LIST (defined at top of file).
    # --run-mode single: one run using --total-tokens / --imbalance-ratio,
    #                    identical behaviour to the original script (default).
    # ------------------------------------------------------------------
    default_tt = args.total_tokens if args.total_tokens is not None else args.num_tokens * num_ranks
    default_ir = args.imbalance_ratio

    if args.run_mode == 'list':
        tt_list = [x * num_ranks for x in SWEEP_TOTAL_TOKENS_LIST]
        ir_list = SWEEP_IMBALANCE_RATIO_LIST
    else:
        tt_list = [default_tt]
        ir_list = [default_ir]

    combos = [(tt, ir) for tt in tt_list for ir in ir_list]
    print(f'Running {len(combos)} combination(s): '
          f'total_tokens={tt_list}, imbalance_ratio={ir_list}')

    for combo_idx, (combo_tt, combo_ir) in enumerate(combos):
        combo_counts = get_imbalance_token_counts(num_ranks, combo_tt, combo_ir / 100.0, 100)
        print(f'\n[combo {combo_idx + 1}/{len(combos)}] '
              f'total_tokens={combo_tt}, imbalance_ratio={combo_ir}%, '
              f'counts={combo_counts}', flush=True)

        combo_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
        combo_result_template = os.path.join(
            result_dir, f"bench_separate_{combo_ts}_rank{{rank}}.json")
        # Include combo params in log filenames only when there are multiple combos
        # so single-run output names stay unchanged.
        log_suffix = f'_tokens{combo_tt}_ibratio{combo_ir}' if len(combos) > 1 else ''

        torch.multiprocessing.spawn(
            test_loop,
            args=(num_local_ranks, args, combo_counts, combo_result_template, log_suffix),
            nprocs=num_local_ranks)

        # ================================================================
        # Post-process aggregation (node 0 only)
        #
        # Global rank 0 already wrote per-rank JSONs for ALL ranks via
        # all_gather inside the distributed context. When running on
        # node 0 these files are available locally; on other nodes the
        # aggregation is skipped.
        # ================================================================
        if node_rank == 0:
            all_global_ranks = list(range(num_ranks))

            # --- global aggregation (all ranks across all nodes) ---
            global_agg = aggregate_bench_separate_results(
                result_path_template=combo_result_template, ranks=all_global_ranks)

            # combine was measured as (dispatch + combine) pipeline because
            # combine_setup_fn runs inside the timed region.  Subtract dispatch_avg
            # from every combine value so that global.combine and summary.combine
            # both reflect true combine-only latency.
            d_avg_us = global_agg['dispatch']['summary_max_per_round']['avg_us']
            c_data = global_agg['combine']
            for rnd in c_data['per_round']:
                rnd['rank_durations_us'] = [v - d_avg_us for v in rnd['rank_durations_us']]
                rnd['avg_us'] -= d_avg_us
                rnd['min_us'] -= d_avg_us
                rnd['max_us'] -= d_avg_us
            for key in ('summary_max_per_round', 'summary_all'):
                for stat in ('avg_us', 'min_us', 'max_us'):
                    c_data[key][stat] -= d_avg_us

            d_smp = global_agg['dispatch']['summary_max_per_round']
            c_smp = global_agg['combine']['summary_max_per_round']

            output = {
                'metadata': {
                    'timestamp': combo_ts,
                    'num_nodes': num_nodes,
                    'num_local_ranks': num_local_ranks,
                    'num_ranks': num_ranks,
                    'total_tokens': combo_tt,
                    'hidden': args.hidden,
                    'num_experts': args.num_experts,
                    'num_topk': args.num_topk,
                    'imbalance_ratio': combo_ir,
                    'dispatch_use_fp8': args.dispatch_use_fp8,
                },
                'summary': {
                    'dispatch': {
                        'latency_us': d_smp['avg_us'],
                        'latency_us_min': d_smp['min_us'],
                        'latency_us_max': d_smp['max_us'],
                    },
                    'combine': {
                        'latency_us': c_smp['avg_us'],
                        'latency_us_min': c_smp['min_us'],
                        'latency_us_max': c_smp['max_us'],
                    },
                },
                'global': global_agg,
            }

            output_name = os.path.join(
                result_dir,
                f'bench_separate_{combo_ts}'
                f'_nodes{num_nodes}_ranks{num_ranks}_tokens{int(combo_tt/num_ranks)}'
                f'_hidden{args.hidden}_experts{args.num_experts}'
                f'_topk{args.num_topk}_ibratio{combo_ir}.json')
            with open(output_name, 'w') as json_file:
                json.dump(output, json_file, indent=4)
            print(f'Aggregated results saved to {output_name}')

            if args.emit_compact_shape_json:
                compact_output = {
                    'dispatch': {
                        'latency_us': d_smp['avg_us'],
                        'latency_us_min': d_smp['min_us'],
                        'latency_us_max': d_smp['max_us'],
                    },
                    'combine': {
                        'latency_us': c_smp['avg_us'],
                        'latency_us_min': c_smp['min_us'],
                        'latency_us_max': c_smp['max_us'],
                    },
                }

                compact_name = os.path.join(
                    result_dir,
                    f'ranks{num_ranks}_tokens{combo_tt}'
                    f'_hidden{args.hidden}_experts{args.num_experts}'
                    f'_topk{args.num_topk}_ibratio{combo_ir}.json')
                with open(compact_name, 'w') as compact_file:
                    json.dump(compact_output, compact_file, indent=4)
                print(f'Compact shape JSON saved to {compact_name}')

            # Cleanup per-rank raw iteration JSONs used for aggregation only.
            for r in all_global_ranks:
                rpath = combo_result_template.format(rank=r)
                if os.path.exists(rpath):
                    os.remove(rpath)
            print(f'Removed temporary per-rank bench JSONs for {num_ranks} ranks')
        else:
            print(f'[node {node_rank}] Skipping aggregation (handled by node 0)')