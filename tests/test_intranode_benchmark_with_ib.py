import argparse
import json
import time
import os
import sys
import datetime
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, inplace_unique, per_token_cast_to_fp8, per_token_cast_back, get_performance_data, get_imbalance_token_counts, analyse_kernel_durations, build_base_parser, add_ib_args

# Test compatibility with low latency functions
import test_low_latency

# -----------------------------------------------------------------------
# Predefined sweep lists.
# Edit these values to define the (total_tokens, imbalance_ratio) matrix
# used when --run-mode=list is passed on the command line.
# The cartesian product of the two lists is executed in order.
# -----------------------------------------------------------------------
# SWEEP_TOTAL_TOKENS_LIST = [4096 * 8, 8192 * 8]  # total across all ranks

SWEEP_TOTAL_TOKENS_LIST = [
    16, 32, 49, 64, 96, 98, 
    128, 147,
    192, 196,
    256, 294, 384, 392, 512, 768, 784, 1024, 1152, 1176,
    1536, 2048, 2304, 3072, 4096, 6144, 8192, 9216, 12288, 16384,
    24576, 32768, 36864, 49152, 65536, 73728, 98304, 131072,
    # 131072, 196608, 262144,
    # 294912, 393216, 524288, 786432, 1048576, 1572864,
]
# SWEEP_TOTAL_TOKENS_LIST = [16]

SWEEP_IMBALANCE_RATIO_LIST = [80]
# SWEEP_IMBALANCE_RATIO_LIST = [0]



# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup,
              actual_num_tokens: int = None, trace_name: str = None,
              trace_template: str = None, trace_ranks: list = None):
    # Settings: use per-rank actual token count when imbalance mode is active
    num_tokens = actual_num_tokens if actual_num_tokens is not None else args.num_tokens
    hidden = args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens} (configured={args.num_tokens}), hidden={hidden}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='gcu').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='gcu') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='gcu')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='gcu')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='gcu')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='gcu')
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    
    # Benchmark layout
    avg_t, min_t, max_t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))
    if local_rank == 0:
        print(f'[layout] performance: avg={avg_t * 1000:.3f} ms, min={min_t * 1000:.3f} ms, max={max_t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    # group.barrier()
    time.sleep(1)

    # Config
    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in filter(lambda elem: elem is not None, (x, x_pure_rand, x_e4m3)):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...', flush=True, end='')
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank,  'is_token_in_rank': is_token_in_rank,
                                     'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights})
                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()
                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    rank_prefix_matrix = handle[0]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if current_x is not x_pure_rand:
                        check_data(recv_x, rank_prefix_matrix)

                    recv_topk_weights_clone = None
                    if with_topk:
                        assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, rank_prefix_matrix)

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        if current_x is not x_pure_rand:
                            check_data(recv_x, rank_prefix_matrix)

                    # Test combine
                    combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
                    if num_tokens > 0:
                        check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
                        ref_x = x_pure_rand if current_x is x_pure_rand else x
                        # assert calc_diff(check_x, ref_x) < 5e-6
                        if with_topk:
                            check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                            ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                            # assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-7

                    # For later tuning
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    # Tune dispatch and combine performance for each data type (FP8 and BF16)
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_dispatch_time, best_dispatch_config = 1e10, None
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        data_type = "FP8" if isinstance(current_x, tuple) else "BF16"
        
        # Tune dispatch for current data type using bench() for low-overhead config search
        for nvl_chunk_size in tuple(range(1, 2, 1)) + (0, ):
            # if nvl_chunk_size > 0:
            #     config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            # else:
            #     # Test default config as well
            deep_ep.Buffer.set_num_sms(num_sms)
            config = deep_ep.Buffer.get_dispatch_config(num_ranks)
            tune_args = {'x': current_x, 'handle': handle, 'config': config}
            dispatch_avg_t, dispatch_min_t, dispatch_max_t = bench(lambda: buffer.dispatch(**tune_args))
            
            if dispatch_avg_t < best_dispatch_time and nvl_chunk_size > 0:
                best_dispatch_time, best_dispatch_config = dispatch_avg_t, (num_sms, nvl_chunk_size)
            
            # Calculate dispatch bandwidth and latency
            dispatch_bandwidth = nvl_recv_bytes / 1e9 / dispatch_avg_t
            dispatch_latency_us = dispatch_avg_t * 1e6
            dispatch_min_us = dispatch_min_t * 1e6
            dispatch_max_us = dispatch_max_t * 1e6

        # Gather the best config from rank 0 and the first test setting
        if best_dispatch_results is None:
            best_dispatch_results = torch.tensor([best_dispatch_config[0], best_dispatch_config[1]], dtype=torch.int32, device='gcu')
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()

        # Re-dispatch to get recv_x for combine testing
        dispatch_config_tmp = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size)
        dispatch_args = {'x': x, 'num_tokens_per_rank': num_tokens_per_rank,
                         'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
                         'config': dispatch_config_tmp}
        recv_x, _, _, _, handle_for_combine, _ = buffer.dispatch(**dispatch_args)

        # Tune combine for current data type using bench() for low-overhead config search
        best_combine_time, best_combine_config = 1e10, None
        for nvl_chunk_size in tuple(range(1, 2, 1)) + (0, ):
            if nvl_chunk_size > 0:
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            else:
                # Test default config as well
                deep_ep.Buffer.set_num_sms(num_sms)
                config = deep_ep.Buffer.get_combine_config(num_ranks)
            tune_args = {'x': recv_x, 'handle': handle_for_combine, 'config': config}
            combine_avg_t, combine_min_t, combine_max_t = bench(lambda: buffer.combine(**tune_args))

            # Include default config (nvl_chunk_size=0) in best selection;
            # store 0 as chunk marker so combine_config creation handles it correctly.
            if combine_avg_t < best_combine_time:
                best_combine_time, best_combine_config = combine_avg_t, (num_sms, nvl_chunk_size)
            
            # Calculate combine bandwidth and latency
            combine_bandwidth = combine_bf16_nvl_send_bytes / 1e9 / combine_avg_t
            combine_latency_us = combine_avg_t * 1e6
            combine_min_us = combine_min_t * 1e6
            combine_max_us = combine_max_t * 1e6

        # Print per-rank results for dispatch and combine together (with data type)
        print(f'[rank {rank}] ({data_type}) Dispatch bandwidth: {dispatch_bandwidth:.2f} GB/s, avg_t={dispatch_latency_us:.2f} us, '
              f'min_t={dispatch_min_us:.2f} us, max_t={dispatch_max_us:.2f} us | '
              f'Combine bandwidth: {combine_bandwidth:.2f} GB/s, avg_t={combine_latency_us:.2f} us, '
              f'min_t={combine_min_us:.2f} us, max_t={combine_max_us:.2f} us', flush=True)
        
        # Aggregate results across all ranks and compute average
        stats_tensor = torch.tensor([
            dispatch_latency_us, dispatch_min_us, dispatch_max_us, dispatch_bandwidth,
            combine_latency_us, combine_min_us, combine_max_us, combine_bandwidth
        ], dtype=torch.float64, device='gcu')
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM, group=group)
        stats_tensor /= num_ranks
        torch.gcu.synchronize()

        dispatch_latency_us_mean = round(stats_tensor[0].item(), 2)
        dispatch_min_us_mean = round(stats_tensor[1].item(), 2)
        dispatch_max_us_mean = round(stats_tensor[2].item(), 2)
        dispatch_bandwidth_mean = round(stats_tensor[3].item(), 2)
        combine_latency_us_mean = round(stats_tensor[4].item(), 2)
        combine_min_us_mean = round(stats_tensor[5].item(), 2)
        combine_max_us_mean = round(stats_tensor[6].item(), 2)
        combine_bandwidth_mean = round(stats_tensor[7].item(), 2)
        
        results = {}
        test_name = '_'.join(['intranode_ib', 'num_ranks', str(num_ranks), 'num_tokens', str(num_tokens),
                               'hidden', str(hidden), 'num_topk', str(num_topk), 'num_experts', str(num_experts)])
        results[test_name] = {
            'mode': 'intranode_ib',
            'num_tokens': num_tokens,
            'hidden': hidden,
            'num_topk': num_topk,
            'num_experts': num_experts,
            'num_ranks': num_ranks,
        }
        results[test_name]['dispatch_latency_us_mean'] = dispatch_latency_us_mean
        results[test_name]['dispatch_bandwidth_mean'] = dispatch_bandwidth_mean
        results[test_name]['combine_latency_us_mean'] = combine_latency_us_mean
        results[test_name]['combine_bandwidth_mean'] = combine_bandwidth_mean

        if rank == 0:
            print(f'[avg across {num_ranks} ranks] ({data_type}) Dispatch bandwidth: {dispatch_bandwidth_mean:.2f} GB/s, avg_t={dispatch_latency_us_mean:.2f} us | '
                  f'Combine bandwidth: {combine_bandwidth_mean:.2f} GB/s, avg_t={combine_latency_us_mean:.2f} us', flush=True)
            print(f'[tuning] ({data_type}) Best dispatch: SMs {best_dispatch_config[0]}, NVL chunk {best_dispatch_config[1]}, '
                  f'{nvl_recv_bytes / 1e9 / best_dispatch_time:.2f} GB/s, t: {best_dispatch_time * 1e6:.2f} us', flush=True)
            print(f'[tuning] ({data_type}) Best combine: SMs {best_combine_config[0]}, NVL chunk {best_combine_config[1]}, '
                  f'{combine_bf16_nvl_send_bytes / 1e9 / best_combine_time:.2f} GB/s, t: {best_combine_time * 1e6:.2f} us', flush=True)
            print('', flush=True)
            
            if args.output_dir is not None:
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                get_performance_data(args.output_dir, test_name, results)
    
    # Final dispatch config for subsequent operations
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size)
    # combine_config: chunk_size=0 means default config obtained via get_combine_config()
    if best_combine_config[1] > 0:
        combine_config = deep_ep.Config(best_combine_config[0], best_combine_config[1], nvl_buffer_size)
    else:
        deep_ep.Buffer.set_num_sms(best_combine_config[0])
        combine_config = deep_ep.Buffer.get_combine_config(num_ranks)

    # Capture kineto trace (dispatch + combine in one pass) for post-hoc analysis
    if trace_name is not None:
        trace_dispatch_args = {'x': x, 'num_tokens_per_rank': num_tokens_per_rank,
                               'is_token_in_rank': is_token_in_rank,
                               'num_tokens_per_expert': num_tokens_per_expert,
                               'config': dispatch_config}

        def combined_trace_fn():
            rv, _, _, _, h, _ = buffer.dispatch(**trace_dispatch_args)
            buffer.combine(x=rv, handle=h, config=combine_config)

        # Step 1: save combined trace for post-hoc analysis (no stats needed here)
        dist.barrier()
        # num_kernels_per_period=1: each buffer.dispatch() / buffer.combine() produces exactly
        # 1 matching kernel event, so treat every event independently (no component summing).
        kineto_trace_stats = bench_kineto(
            combined_trace_fn,
            kernel_names=('intranode::dispatch<', 'intranode::combine<'),
            barrier_comm_profiling=True, trace_path=trace_name,
            suppress_kineto_output=True, num_kernels_per_period=1,
            return_stats=True)
        if rank == 0:
            print(f'[trace] Saved kineto trace to {trace_name}', flush=True)

        # With num_kernels_per_period=1, bench_kineto returns a plain dict per kernel
        # (not a list of dicts), so index directly by key.
        dispatch_kineto_avg = kineto_trace_stats[0]['avg']
        dispatch_kineto_min = kineto_trace_stats[0]['min']
        dispatch_kineto_max = kineto_trace_stats[0]['max']
        combine_kineto_avg  = kineto_trace_stats[1]['avg']
        combine_kineto_min  = kineto_trace_stats[1]['min']
        combine_kineto_max  = kineto_trace_stats[1]['max']

        dispatch_kineto_bw = dispatch_bf16_nvl_recv_bytes / 1e9 / dispatch_kineto_avg
        combine_kineto_bw  = combine_bf16_nvl_send_bytes  / 1e9 / combine_kineto_avg
        if rank == 0:
            print(f'[kineto/rank0] Dispatch bandwidth: {dispatch_kineto_bw:.2f} GB/s, '
                  f'avg_t={dispatch_kineto_avg * 1e6:.2f} us, '
                  f'min_t={dispatch_kineto_min * 1e6:.2f} us, '
                  f'max_t={dispatch_kineto_max * 1e6:.2f} us', flush=True)
            print(f'[kineto/rank0] Combine  bandwidth: {combine_kineto_bw:.2f} GB/s, '
                  f'avg_t={combine_kineto_avg * 1e6:.2f} us, '
                  f'min_t={combine_kineto_min * 1e6:.2f} us, '
                  f'max_t={combine_kineto_max * 1e6:.2f} us', flush=True)

        # Barrier to ensure every rank has finished writing its trace file before rank 0 reads them.
        dist.barrier()

        # Re-parse all local rank traces for exact collective latency (avg of per-iter max across ranks).
        # This is identical to the methodology used by analyse_kernel_durations() in __main__ to
        # produce the JSON output, so [kineto/collective] and JSON latency_us will match exactly.
        if rank == 0 and trace_template is not None and trace_ranks is not None:
            coll_results = analyse_kernel_durations(
                trace_name_template=trace_template,
                ranks=trace_ranks,
                kernel_names=['dispatch', 'combine'],
                verbose=False)

            dispatch_coll_lat_us = coll_results['dispatch']['latency_us']
            dispatch_coll_min_us = coll_results['dispatch']['latency_us_min']
            dispatch_coll_max_us = coll_results['dispatch']['latency_us_max']
            combine_coll_lat_us  = coll_results['combine']['latency_us']
            combine_coll_min_us  = coll_results['combine']['latency_us_min']
            combine_coll_max_us  = coll_results['combine']['latency_us_max']

            dispatch_coll_bw = dispatch_bf16_nvl_recv_bytes / 1e9 / (dispatch_coll_lat_us / 1e6)
            combine_coll_bw  = combine_bf16_nvl_send_bytes  / 1e9 / (combine_coll_lat_us  / 1e6)
            print(f'[kineto/collective] Dispatch bandwidth: {dispatch_coll_bw:.2f} GB/s, '
                  f'avg_t={dispatch_coll_lat_us:.2f} us, '
                  f'min_t={dispatch_coll_min_us:.2f} us, '
                  f'max_t={dispatch_coll_max_us:.2f} us', flush=True)
            print(f'[kineto/collective] Combine  bandwidth: {combine_coll_bw:.2f} GB/s, '
                  f'avg_t={combine_coll_lat_us:.2f} us, '
                  f'min_t={combine_coll_min_us:.2f} us, '
                  f'max_t={combine_coll_max_us:.2f} us', flush=True)

        # Save actual NVLink bytes to sidecar so __main__ can compute accurate bandwidth_gb_s in JSON
        if rank == 0:
            bytes_meta_path = trace_name.replace('.json', '.bytes_meta.json')
            with open(bytes_meta_path, 'w') as _f:
                json.dump({
                    'dispatch_bytes': dispatch_bf16_nvl_recv_bytes,
                    'combine_bytes': combine_bf16_nvl_send_bytes,
                }, _f)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace,
              counts: list = None, log_suffix: str = '', trace_template: str = None):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = f"{rank}_test_intranode_benchmark_with_ib{log_suffix}_{log_timestamp}.log"
    trace_name = trace_template.format(rank=rank) if trace_template is not None else None

    # Redirect stdout and stderr to log file
    with open(log_file, 'w') as f:
        # Save original file descriptors
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)

        try:
            # Redirect file descriptors (for C++ prints and subprocess output)
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)

            # Redirect Python stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = f
            sys.stderr = f

            print(f'Initialized rank {rank}/{num_ranks} on local rank {local_rank}', flush=True)

            test_ll_compatibility, num_rdma_bytes = False, 0
            if test_ll_compatibility:
                ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9
                num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(ll_num_tokens, ll_hidden, num_ranks, ll_num_experts)

            buffer = deep_ep.Buffer(group, int(2e9), num_rdma_bytes, low_latency_mode=test_ll_compatibility,
                                    num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1), explicitly_destroy=True)

            print(f'Rank {rank} buffer.runtime.is_available(): {buffer.runtime.is_available()}', flush=True)
            torch.manual_seed(rank)

            # Determine per-rank token count: imbalance mode if counts provided, else uniform
            actual_num_tokens = counts[rank] if counts is not None else args.num_tokens
            if counts is not None:
                print(f'[rank {rank}] Imbalance mode: actual_num_tokens={actual_num_tokens} (max={max(counts)}, configured={args.num_tokens})', flush=True)

            # Compute the global rank IDs that reside on this node so that
            # analyse_kernel_durations() inside test_main can read all local traces.
            node_offset = rank - local_rank  # = node_rank * num_local_ranks
            trace_ranks = list(range(node_offset, node_offset + num_local_ranks))

            for i in (24, ):
                test_main(args, i, local_rank, num_ranks, rank, buffer, group,
                          actual_num_tokens=actual_num_tokens, trace_name=trace_name,
                          trace_template=trace_template, trace_ranks=trace_ranks)
                if local_rank == 0:
                    print('', flush=True)

            # Test compatibility with low latency functions
            if test_ll_compatibility:
                buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
                test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

            print(f'Test intranode benchmark (with imbalance) successfully finished!', flush=True)

            # Destroy the buffer runtime and communication group
            buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()

        finally:
            # Restore original file descriptors and Python stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)


if __name__ == '__main__':
    parser = build_base_parser('Test intranode EP kernels with imbalance (IB) support', defaults={'num_tokens': 4096})
    # num-tokens has a richer meaning here; override its help text.
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for performance JSON data. If not specified, no JSON file is saved.')
    add_ib_args(parser, imbalance_default=0, add_run_mode=True,
                total_tokens_help='Total tokens across all ranks (single mode). '
                                  'Defaults to --num-tokens * num_ranks when not specified.')
    args = parser.parse_args()

    num_processes = args.num_processes
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    num_ranks = num_processes * num_nodes

    print(f'num_nodes={num_nodes}, node_rank={node_rank}, num_processes={num_processes}, num_ranks={num_ranks}')

    # ------------------------------------------------------------------
    # Build the list of (total_tokens, imbalance_ratio) combinations.
    #
    # --run-mode list  : cartesian product of SWEEP_TOTAL_TOKENS_LIST and
    #                    SWEEP_IMBALANCE_RATIO_LIST (defined at top of file).
    # --run-mode single: one run using --total-tokens / --imbalance-ratio.
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
        # Deterministic seed ensures same distribution on every node
        counts = get_imbalance_token_counts(num_ranks, combo_tt, combo_ir / 100.0, 100)

        print(f'\n[combo {combo_idx + 1}/{len(combos)}] '
              f'total_tokens={combo_tt}, imbalance_ratio={combo_ir}%, '
              f'counts={counts}', flush=True)

        # Include combo params in log filenames only when there are multiple combos
        log_suffix = f'_tokens{combo_tt}_ibratio{combo_ir}' if len(combos) > 1 else ''

        # Unique trace path template for this combo (one file per rank)
        combo_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
        trace_path_template = os.path.join('./', f'kineto_trace_intranode_{combo_ts}_rank{{rank}}.json')

        torch.multiprocessing.spawn(
            test_loop,
            args=(num_processes, args, counts, log_suffix, trace_path_template),
            nprocs=num_processes)

        # Analyse traces for local ranks only (each node only has its own trace files)
        local_global_ranks = list(range(node_rank * num_processes, (node_rank + 1) * num_processes))
        results = analyse_kernel_durations(
            trace_name_template=trace_path_template,
            ranks=local_global_ranks,
            kernel_names=['dispatch', 'combine'])

        # Append bandwidth (GB/s) from kineto latency + actual NVLink bytes (from sidecar).
        # min latency → max bandwidth; max latency → min bandwidth (inverse relationship).
        # The sidecar (.bytes_meta.json) is written by rank 0 in test_main and contains the
        # actual recv/send bytes after topk routing, which matches the printed log bandwidth.
        # Fall back to per-rank input size (tokens * hidden * 2) if sidecar is unavailable.
        _fallback_bytes = int(combo_tt / num_ranks) * args.hidden * 2  # BF16 input per rank
        bytes_meta_path = trace_path_template.format(rank=0).replace('.json', '.bytes_meta.json')
        try:
            with open(bytes_meta_path) as _meta_f:
                _bytes_meta = json.load(_meta_f)
            dispatch_data_bytes = _bytes_meta['dispatch_bytes']
            combine_data_bytes  = _bytes_meta['combine_bytes']
        except (FileNotFoundError, KeyError):
            dispatch_data_bytes = _fallback_bytes
            combine_data_bytes  = _fallback_bytes

        def _bw(lat_us, data_bytes):
            return round(data_bytes / 1e9 / (lat_us / 1e6), 2) if lat_us > 0 else 0.0

        for kernel in ['dispatch', 'combine']:
            r = results[kernel]
            data_bytes = dispatch_data_bytes if kernel == 'dispatch' else combine_data_bytes
            r['bandwidth_gb_s']     = _bw(r['latency_us'],     data_bytes)
            r['bandwidth_gb_s_max'] = _bw(r['latency_us_min'], data_bytes)  # fastest → highest BW
            r['bandwidth_gb_s_min'] = _bw(r['latency_us_max'], data_bytes)  # slowest → lowest BW

        output_name = (f'node{node_rank}_ranks{num_ranks}_tokens{int(combo_tt/num_ranks)}_hidden{args.hidden}'
                       f'_experts{args.num_experts}_topk{args.num_topk}_ibratio{combo_ir}.json')
        with open(output_name, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        print(f'[combo {combo_idx + 1}/{len(combos)}] Analysis saved to {output_name}', flush=True)
