"""Intranode dispatch GCUGraph test with ``num_worst_tokens``.

``num_worst_tokens > 0`` skips the host poll of recv counts so dispatch is
graph-compatible. Capture / replay follows ``pcals/test_mega_moe_benchmark.py``.

Stress note: capture once then replay many times. Do NOT re-capture in a
tight loop — each ``torch.gcu.Stream()`` for capture can eventually collide
with DeepEP's ``comm_stream``, failing ``event.hpp`` ``s_0.id() != s_1.id()``
and yielding empty GCUGraph warnings.
"""
import argparse
import os
from typing import Callable

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, calc_diff, inplace_unique, per_token_cast_to_fp8, per_token_cast_back, build_base_parser, add_sms_arg


def _make_gcu_graph_replay_fn(
    fn: Callable[[], None],
    *,
    stream: torch.Stream,
    num_warmup: int = 3,
) -> Callable[[], None]:
    """Capture ``fn`` into a GCUGraph on a fixed ``stream`` and return ``replay``."""
    for _ in range(num_warmup):
        fn()
    torch.gcu.synchronize()
    if dist.is_initialized():
        dist.barrier()

    graph = torch.gcu.GCUGraph()
    with torch.gcu.graph(graph, stream=stream):
        fn()
    torch.gcu.synchronize()
    if dist.is_initialized():
        dist.barrier()

    def replay() -> None:
        graph.replay()

    return replay


def _check_graph_outs(outs, recv_exact, recv_eager, recv_eager_topk_idx, recv_eager_topk_weights,
                      actual_eager, num_worst_tokens):
    assert len(outs['empty_list']) == 0
    graph_recv = outs['recv_x']
    graph_recv_tokens = graph_recv[0] if isinstance(graph_recv, tuple) else graph_recv
    assert graph_recv_tokens.size(0) == num_worst_tokens
    assert len(outs['handle']) == 8
    actual_graph = outs['handle'][-1].item()
    assert actual_graph == recv_exact.size(0), f'graph actual {actual_graph} != exact {recv_exact.size(0)}'

    graph_x = per_token_cast_back(*graph_recv) if isinstance(graph_recv, tuple) else graph_recv
    assert torch.equal(graph_x[:actual_graph], recv_eager[:actual_eager])
    assert torch.equal(outs['recv_topk_idx'][:actual_graph], recv_eager_topk_idx[:actual_eager])
    assert torch.equal(outs['recv_topk_weights'][:actual_graph], recv_eager_topk_weights[:actual_eager])
    assert torch.equal(graph_x[:actual_graph], recv_exact)
    assert calc_diff(graph_x[:actual_graph].float(), recv_exact.float()) < 1e-6
    return actual_graph


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup, capture_stream: torch.Stream):
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    stress_iters = args.stress_iters
    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, '
              f'stress_iters={stress_iters}', flush=True)

    # Random / rank-tagged data (same setup as test_intranode.py)
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='gcu').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='gcu') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='gcu')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

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

    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)
    num_worst_tokens = num_tokens * num_ranks

    for current_x, with_rand_topk in (
        (x, False),
        (x_pure_rand, True),
        *(( (x_e4m3, False), ) if x_e4m3 is not None else ()),
    ):
        dtype_name = 'FP8' if isinstance(current_x, tuple) else 'BF16'
        tw = topk_weights_pure_rand if with_rand_topk else topk_weights
        if local_rank == 0:
            print(f'[testing] eager baseline {dtype_name} ...', flush=True, end='')

        recv_exact, recv_exact_topk_idx, recv_exact_topk_weights, recv_num_list, _, _ = buffer.dispatch(
            x=current_x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=tw,
            config=config,
        )
        recv_exact = per_token_cast_back(*recv_exact) if isinstance(recv_exact, tuple) else recv_exact
        assert gbl_num_tokens_per_rank[rank].item() == recv_exact.size(0)
        assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_list

        recv_eager, recv_eager_topk_idx, recv_eager_topk_weights, empty_list, handle_eager, _ = buffer.dispatch(
            x=current_x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=tw,
            num_worst_tokens=num_worst_tokens,
            config=config,
        )
        recv_eager = per_token_cast_back(*recv_eager) if isinstance(recv_eager, tuple) else recv_eager
        assert len(empty_list) == 0
        assert recv_eager.size(0) == num_worst_tokens
        assert len(handle_eager) == 8
        actual_eager = handle_eager[-1].item()
        assert actual_eager == recv_exact.size(0), f'{actual_eager} != {recv_exact.size(0)}'
        assert torch.equal(recv_exact, recv_eager[:actual_eager])
        assert torch.equal(recv_exact_topk_idx, recv_eager_topk_idx[:actual_eager])
        assert torch.equal(recv_exact_topk_weights, recv_eager_topk_weights[:actual_eager])
        if local_rank == 0:
            print(' passed', flush=True)

        # Capture once, then stress-replay (do not re-capture each iter).
        if local_rank == 0:
            print(f'[testing] GCUGraph capture + stress replay {dtype_name} '
                  f'(iters={stress_iters}) ...', flush=True, end='')

        outs = {}

        def dispatch_fn():
            recv_x, recv_topk_idx, recv_topk_weights, empty, handle, _event = buffer.dispatch(
                x=current_x,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx,
                topk_weights=tw,
                num_worst_tokens=num_worst_tokens,
                config=config,
                async_finish=False,
            )
            outs['recv_x'] = recv_x
            outs['recv_topk_idx'] = recv_topk_idx
            outs['recv_topk_weights'] = recv_topk_weights
            outs['empty_list'] = empty
            outs['handle'] = handle

        dist.barrier()
        replay_fn = _make_gcu_graph_replay_fn(dispatch_fn, stream=capture_stream)

        actual_graph = None
        for _ in range(stress_iters):
            replay_fn()
            torch.gcu.synchronize()
            dist.barrier()
            actual_graph = _check_graph_outs(
                outs, recv_exact, recv_eager, recv_eager_topk_idx, recv_eager_topk_weights,
                actual_eager, num_worst_tokens)

        if local_rank == 0:
            print(f' passed (actual_num_recv_tokens={actual_graph})', flush=True)

    if local_rank == 0:
        print('', flush=True)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(group, int(2e9), 0, low_latency_mode=False,
                            num_qps_per_rank=1, explicitly_destroy=True)
    print(f'Rank {rank} buffer.runtime.is_available(): {buffer.runtime.is_available()}', flush=True)
    torch.manual_seed(rank)

    # One capture stream for the whole process. Recreating streams per capture
    # eventually collides with DeepEP's pooled comm_stream.
    capture_stream = torch.gcu.Stream()
    comm_stream = buffer.get_comm_stream()
    assert capture_stream.stream_id != comm_stream.stream_id, \
        f'capture stream id {capture_stream.stream_id} collides with DeepEP comm_stream'

    env_num_sms = os.environ.get('EP_DEBUG_COMBINE_USE_SM')
    num_sms = int(env_num_sms) if env_num_sms is not None else args.num_sms
    test_main(args, num_sms, local_rank, num_ranks, rank, buffer, group, capture_stream)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = build_base_parser('Test intranode EP dispatch under GCUGraph (num_worst_tokens)',
                               defaults={'num_tokens': 4096})
    add_sms_arg(parser, default=24)
    parser.add_argument('--stress-iters', type=int, default=1000,
                        help='GCUGraph replay iterations after a single capture (default: 100)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
