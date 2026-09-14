import argparse
import datetime
import os
import sys

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench_kineto, create_grouped_scores, inplace_unique, build_base_parser, add_sms_arg, add_hardcode_local_ranks_arg


def _resolve_num_topk_groups(requested_num_topk_groups, num_nodes: int):
    num_topk_groups = min(num_nodes, 4) if requested_num_topk_groups is None else requested_num_topk_groups
    assert 1 <= num_topk_groups <= num_nodes
    return num_topk_groups


def _build_dispatch_inputs(args: argparse.Namespace,
                           rank: int,
                           num_ranks: int,
                           num_nodes: int,
                           num_local_ranks: int):
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk_groups = _resolve_num_topk_groups(args.num_topk_groups, num_nodes)
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0
    assert num_experts % num_nodes == 0

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='gcu').abs() + 1

    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False).indices
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu').abs()

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='gcu')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='gcu')
    num_tokens_per_rdma_rank = torch.empty((num_nodes,), dtype=torch.int, device='gcu')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='gcu')

    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='gcu')

    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()

    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0

    return x, topk_idx, topk_weights, num_tokens_per_expert, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank


def _build_config(num_sms: int, num_ranks: int):
    rdma_buffer_size = 128
    nvl_buffer_size = 720 if num_ranks in (48, 96, 144, 160) else 128
    return deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)


def _run_benchmark(args: argparse.Namespace,
                   rank: int,
                   local_rank: int,
                   num_ranks: int,
                   num_nodes: int,
                   num_local_ranks: int,
                   buffer: deep_ep.Buffer,
                   group: dist.ProcessGroup):
    effective_num_topk_groups = _resolve_num_topk_groups(args.num_topk_groups, num_nodes)
    x, topk_idx, topk_weights, manual_num_tokens_per_expert, manual_num_tokens_per_rank, manual_num_tokens_per_rdma_rank, manual_is_token_in_rank = \
        _build_dispatch_inputs(args, rank, num_ranks, num_nodes, num_local_ranks)
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, args.num_experts)
    if rank == 0 and not (
        torch.equal(num_tokens_per_rank, manual_num_tokens_per_rank) and
        torch.equal(num_tokens_per_rdma_rank, manual_num_tokens_per_rdma_rank) and
        torch.equal(num_tokens_per_expert, manual_num_tokens_per_expert) and
        torch.equal(is_token_in_rank, manual_is_token_in_rank)
    ):
        print(
            '[warning] manual dispatch layout differs from get_dispatch_layout; '
            'benchmark will use get_dispatch_layout results.',
            flush=True
        )
    config = _build_config(args.num_sms, num_ranks)

    # RDMA dispatch counts (matches original test_internode.py methodology)
    rdma_idx = topk_idx // (args.num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    if local_rank == 0:
        print(
            f'[config] num_tokens={args.num_tokens}, hidden={args.hidden}, num_topk_groups={effective_num_topk_groups}, '
            f'num_topk={args.num_topk}, num_sms={args.num_sms}, num_nodes={num_nodes}, num_local_ranks={num_local_ranks}',
            flush=True
        )

    dispatch_args = {
        'x': x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'topk_idx': topk_idx,
        'topk_weights': topk_weights,
        'config': config,
        'async_finish': False,
    }

    # Warmup once to materialize representative recv tensors and handle.
    recv_x, _, _, _, _, _ = buffer.dispatch(**dispatch_args)

    dispatch_rdma_send_bytes = num_rdma_token_sent * args.hidden * x.element_size()
    dispatch_nvl_recv_bytes = recv_x.numel() * recv_x.element_size()
    combine_nvl_send_bytes = dispatch_nvl_recv_bytes
    combine_rdma_recv_bytes = dispatch_rdma_send_bytes
    # NOTE: internode combine handle is tied to a concrete dispatch output.
    # Reusing a stale handle (especially after many dispatch calls) can hang.
    def run_combine_with_fresh_dispatch():
        fresh_recv_x, _, fresh_recv_topk_weights, _, fresh_handle, _ = buffer.dispatch(**dispatch_args)
        buffer.combine(
            x=fresh_recv_x,
            handle=fresh_handle,
            topk_weights=fresh_recv_topk_weights,
            config=config,
            async_finish=False,
        )

    dist.barrier(group=group)
    try:
        dispatch_stats = bench_kineto(
            lambda: buffer.dispatch(**dispatch_args),
            kernel_names=args.dispatch_kernel_name,
            num_tests=args.kineto_num_tests,
            barrier_comm_profiling=True,
            suppress_kineto_output=True,
            return_stats=True,
        )
    except AssertionError as exc:
        raise RuntimeError(
            f'Failed to profile dispatch kernel "{args.dispatch_kernel_name}". '
            f'Please verify kernel name with --dispatch-kernel-name.'
        ) from exc

    dist.barrier(group=group)
    try:
        combine_stats = bench_kineto(
            run_combine_with_fresh_dispatch,
            kernel_names=args.combine_kernel_name,
            num_tests=args.kineto_num_tests,
            barrier_comm_profiling=True,
            suppress_kineto_output=True,
            return_stats=True,
        )
    except AssertionError as exc:
        raise RuntimeError(
            f'Failed to profile combine kernel "{args.combine_kernel_name}". '
            f'Please verify kernel name with --combine-kernel-name.'
        ) from exc

    dispatch_avg_us = dispatch_stats['avg'] * 1e6
    combine_avg_us = combine_stats['avg'] * 1e6
    dispatch_rdma_bw = dispatch_rdma_send_bytes / 1e9 / dispatch_stats['avg']
    dispatch_nvl_bw = dispatch_nvl_recv_bytes / 1e9 / dispatch_stats['avg']
    combine_rdma_bw = combine_rdma_recv_bytes / 1e9 / combine_stats['avg']
    combine_nvl_bw = combine_nvl_send_bytes / 1e9 / combine_stats['avg']

    if args.print_per_rank:
        print(
            f'[rank {rank}] Dispatch: {dispatch_rdma_bw:.2f} GB/s (RDMA), {dispatch_nvl_bw:.2f} GB/s (NVL), '
            f'avg_t={dispatch_avg_us:.2f} us | '
            f'Combine: {combine_rdma_bw:.2f} GB/s (RDMA), {combine_nvl_bw:.2f} GB/s (NVL), '
            f'avg_t={combine_avg_us:.2f} us',
            flush=True
        )

    stats_tensor = torch.tensor(
        [dispatch_rdma_bw, dispatch_nvl_bw, dispatch_avg_us,
         combine_rdma_bw, combine_nvl_bw, combine_avg_us],
        dtype=torch.float64,
        device='gcu'
    )
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM, group=group)
    stats_tensor /= num_ranks
    torch.gcu.synchronize()

    if rank == 0:
        print(
            f'[avg across {num_ranks} ranks] '
            f'Dispatch: {stats_tensor[0].item():.2f} GB/s (RDMA), {stats_tensor[1].item():.2f} GB/s (NVL), '
            f'avg_t={stats_tensor[2].item():.2f} us | '
            f'Combine: {stats_tensor[3].item():.2f} GB/s (RDMA), {stats_tensor[4].item():.2f} GB/s (NVL), '
            f'avg_t={stats_tensor[5].item():.2f} us',
            flush=True
        )


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    effective_num_nodes = int(os.getenv('WORLD_SIZE', 1))
    effective_num_local_ranks = num_local_ranks

    if args.num_hardcode_local_ranks is not None:
        # Single-node internode simulation:
        # split spawned ranks into virtual nodes by EP_INTRA_RANKS.
        if num_local_ranks % args.num_hardcode_local_ranks != 0:
            raise ValueError(
                f'num_processes={num_local_ranks} must be divisible by '
                f'num_hardcode_local_ranks={args.num_hardcode_local_ranks}'
            )
        effective_num_nodes = num_local_ranks // args.num_hardcode_local_ranks
        effective_num_local_ranks = args.num_hardcode_local_ranks

    log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = f"{rank}_test_internode_benchmark.log"
    with open(log_file, 'w') as f:
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            sys.stdout = f
            sys.stderr = f

            print(
                f'Initialized rank {rank}/{num_ranks} on local rank {local_rank}, '
                f'num_nodes={effective_num_nodes}, num_local_ranks={effective_num_local_ranks}',
                flush=True
            )

            buffer = deep_ep.Buffer(
                group,
                int(2e9),
                int(2e9),
                low_latency_mode=False,
                num_qps_per_rank=max(args.num_sms, 1),
                explicitly_destroy=True,
                is_internode=True
            )

            dist.barrier(group=group)
            if rank == 0 and buffer.runtime.get_num_rdma_ranks() <= 1:
                print(
                    '[warning] runtime reports num_rdma_ranks <= 1, benchmark may not include real cross-node traffic.',
                    flush=True
                )

            torch.manual_seed(args.seed + rank)
            _run_benchmark(
                args,
                rank,
                local_rank,
                num_ranks,
                effective_num_nodes,
                effective_num_local_ranks,
                buffer,
                group
            )

            dist.barrier(group=group)
            buffer.destroy()
            dist.barrier(group=group)
            dist.destroy_process_group()
            print('Test internode benchmark successfully finished!', flush=True)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)

    # Workaround: exit without running full interpreter shutdown (atexit, gc, module
    # teardown). ECCL/backend teardown can trigger malloc_consolidate heap corruption
    # and SIGABRT on one rank after the benchmark has already completed successfully.
    os._exit(0)


if __name__ == '__main__':
    parser = build_base_parser('Benchmark internode EP kernels',
                               defaults={'num_processes': 16, 'num_tokens': 10240, 'num_experts': 256},
                               num_topk_groups=True)
    add_hardcode_local_ranks_arg(parser, default=None)
    add_sms_arg(parser, default=24)
    parser.add_argument('--kineto-num-tests', type=int, default=30,
                        help='Number of timed calls per kineto period (default: 30)')
    parser.add_argument('--dispatch-kernel-name', type=str, default='internode::dispatch<',
                        help='Dispatch kernel name fragment for bench_kineto (default: internode::dispatch<)')
    parser.add_argument('--combine-kernel-name', type=str, default='internode::combine<',
                        help='Combine kernel name fragment for bench_kineto (default: internode::combine<)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Base seed, each rank uses seed+rank (default: 1)')
    parser.add_argument('--print-per-rank', action='store_true',
                        help='Print per-rank dispatch/combine stats before all-rank average')
    args = parser.parse_args()

    world_size_env = int(os.getenv('WORLD_SIZE', 1))
    if world_size_env == 1 and args.num_processes > 1 and args.num_hardcode_local_ranks is None:
        # Choose a practical default split for single-node simulation.
        args.num_hardcode_local_ranks = args.num_processes // 2 if args.num_processes % 2 == 0 else 1

    if args.num_hardcode_local_ranks is not None:
        if args.num_hardcode_local_ranks <= 0:
            raise ValueError('num-hardcode-local-ranks must be > 0')
        if args.num_processes <= args.num_hardcode_local_ranks:
            raise ValueError(
                f'num_processes={args.num_processes} must be greater than '
                f'num_hardcode_local_ranks={args.num_hardcode_local_ranks} for internode mode'
            )
        if args.num_processes % args.num_hardcode_local_ranks != 0:
            raise ValueError(
                f'num_processes={args.num_processes} must be divisible by '
                f'num_hardcode_local_ranks={args.num_hardcode_local_ranks}'
            )
        os.environ['EP_INTRA_RANKS'] = str(args.num_hardcode_local_ranks)

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
