import argparse
import random
import os
from sys import stderr
import torch
import torch.distributed as dist
from functools import partial
import datetime
import subprocess

import deep_ep
from utils import init_dist, bench, bench_kineto, per_token_cast_back, draw_latency_chart, draw_bandwidth_chart, get_performance_data, get_imbalance_token_counts, build_base_parser, add_fp8_dispatch_arg, add_ib_args


def create_test_data(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                     rank: int, num_ranks: int, use_logfmt: bool = False, seed: int = 0):
    """Create test data for performance testing."""
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * 0.1

    scores = torch.rand((num_tokens, num_experts), dtype=torch.float32, device='gcu').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu').abs()

    return x, topk_idx, topk_weights


def run_performance_test(current_num_tokens: int, max_tokens_per_rank: int, hidden: int,
                         num_experts: int, num_topk: int,
                         rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
                         use_logfmt: bool = False, seed: int = 0, use_fp8: bool = False,
                         num_processes: int = 1, iters: int = 300):
    """Run performance tests for low-latency dispatch and combine."""
    x, topk_idx, topk_weights = create_test_data(current_num_tokens, hidden, num_experts, num_topk,
                                                  rank, num_ranks, use_logfmt, seed)

    num_local_experts = num_experts // num_ranks
    cumulative_local_expert_recv_stats = torch.zeros((num_local_experts,), dtype=torch.int, device='gcu')

    # Dispatch to get handle and simulated data
    packed_recv_x, packed_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(x, topk_idx, max_tokens_per_rank, num_experts,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    use_fp8=use_fp8, async_finish=True, return_recv_hook=False)
    event.current_stream_wait()

    # Handle FP8 case: packed_recv_x is a tuple (x_fp8, x_scales) when use_fp8=True
    if use_fp8:
        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous())
        simulated_gemm_x = per_token_cast_back(
            packed_recv_x[0].view(-1, hidden),
            packed_recv_x[1].view(-1, hidden // 128)
        ).view(packed_recv_x[0].shape)
    else:
        simulated_gemm_x = packed_recv_x.clone()

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool, iters: int = 100):
        for _ in range(iters):
            _, _, handle, _, hook = \
                buffer.low_latency_dispatch(x, topk_idx, max_tokens_per_rank, num_experts,
                                            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                            use_fp8=use_fp8, async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
            large_gemm_with_hook(hook) if return_recv_hook else None
            _, _, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                                use_logfmt=use_logfmt, return_recv_hook=return_recv_hook)
            large_gemm_with_hook(hook) if return_recv_hook else None
    try:
        test_func(return_recv_hook=False, iters=iters)
    except Exception as e:
        print(f'Error: {e}', flush=True)

# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace, counts: list):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = f"{rank}_test_low_latency_perf_{log_timestamp}.log"

    # Redirect stdout and stderr to log file
    with open(log_file, 'w') as f:
        import sys
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

            max_tokens_per_rank = max(counts)
            current_rank_num_tokens = counts[rank]
            hidden = args.hidden
            num_topk, num_experts = args.num_topk, args.num_experts

            print(f'[rank {rank}] current_num_tokens={current_rank_num_tokens}, '
                  f'max_tokens_per_rank={max_tokens_per_rank}, '
                  f'imbalance_ratio={args.imbalance_ratio}%', flush=True)

            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(max_tokens_per_rank, hidden, num_ranks, num_experts)
            if local_rank == 0:
                print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
            buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                                    num_qps_per_rank=num_experts // num_ranks,
                                    allow_nvlink_for_low_latency_mode=True, explicitly_destroy=True,
                                    allow_mnnvl=False)
            run_performance_test(current_rank_num_tokens, max_tokens_per_rank, hidden, num_experts, num_topk,
                                 rank, num_ranks, group, buffer,
                                 use_logfmt=False, seed=1, use_fp8=args.dispatch_use_fp8,
                                 num_processes=args.num_processes, iters=args.iters)
            print(f'Performance test successfully finished!', flush=True)
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
    parser = build_base_parser('Performance test for low-latency EP kernels (topsprof mode)')
    add_fp8_dispatch_arg(parser)
    add_ib_args(parser, imbalance_default=0,
                total_tokens_help='Total tokens across all ranks. Defaults to num-tokens * num-ranks if not specified.')
    parser.add_argument('--iters', type=int, default=300,
                       help='Number of iterations to run (default: 300)')
    args = parser.parse_args()

    num_local_ranks = args.num_processes
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    num_ranks = num_local_ranks * num_nodes

    total_tokens = args.total_tokens if args.total_tokens is not None else args.num_tokens * num_ranks
    imbalance_ratio = args.imbalance_ratio

    counts = get_imbalance_token_counts(num_ranks, total_tokens, imbalance_ratio / 100.0, 100)
    print(f'num_nodes={num_nodes}, node_rank={node_rank}, num_local_ranks={num_local_ranks}, '
          f'num_ranks(world_size)={num_ranks}')
    print(f'total_tokens={total_tokens}, imbalance_ratio={imbalance_ratio}%, '
          f'max_tokens_per_rank={max(counts)}')
    print(f'counts of tokens per rank: {counts}')

    torch.multiprocessing.spawn(test_loop, args=(num_local_ranks, args, counts), nprocs=num_local_ranks)
