"""Intranode FP8 dispatch-only: determinism + reference bitwise check.

Phase 1 — determinism (run0 vs run_i, i>=1):
  1. fp8 bytes in valid rows: torch.equal on view(uint8)
  2. scales in valid rows: torch.equal after reshape [M, ngroups]
  3. raw scales buffer: torch.equal on reshape(-1)

Phase 2 — after all repeats, dispatch vs source reference (test_intranode.py):
  recv_x / recv_x_scales uint8-equal to gathered source via recv_src_idx.

Env vars:
  DET_REPEATS  default 5
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, inplace_unique, per_token_cast_back, per_token_cast_to_fp8, build_base_parser, add_sms_arg


def _make_noisy_bf16_tokens(num_tokens, hidden, rank, seed):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed + rank + 98765)
    return torch.randn(
        (num_tokens, hidden), dtype=torch.float32, generator=g, device='cpu'
    ).to(torch.bfloat16).to('gcu')


def _make_inputs(num_tokens, hidden, num_experts, num_topk, seed, rank):
    tokens = _make_noisy_bf16_tokens(num_tokens, hidden, rank, seed)
    g_topk = torch.Generator(device='cpu')
    g_topk.manual_seed(seed + rank + 12345)
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, num_topk), dtype=torch.int64, generator=g_topk, device='cpu'
    ).to('gcu')
    g_w = torch.Generator(device='cpu')
    g_w.manual_seed(seed + rank + 54321)
    topk_weights = (
        torch.rand((num_tokens, num_topk), dtype=torch.float32, generator=g_w, device='cpu') + 0.5
    ).to('gcu')
    return tokens, topk_idx, topk_weights


def _valid_row_mask(recv_bf16: torch.Tensor) -> torch.Tensor:
    return recv_bf16.abs().sum(dim=1) > 0


def _build_layout_manual(topk_idx, num_experts, num_ranks):
    num_tokens = topk_idx.size(0)
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.masked_fill(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='gcu')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='gcu')
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
    return num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert


def _build_layout_kernel(buffer, topk_idx, num_experts):
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    return num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert


def _build_layout(buffer, topk_idx, num_experts, num_ranks, use_kernel_layout: bool):
    if use_kernel_layout:
        return _build_layout_kernel(buffer, topk_idx, num_experts)
    return _build_layout_manual(topk_idx, num_experts, num_ranks)


def _do_dispatch(buffer, x_fp8, x_scales, topk_idx, topk_weights,
                 num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, config):
    recv_x, recv_topk_idx, recv_topk_weights, _, handle, event = buffer.dispatch(
        x=(x_fp8, x_scales),
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
        async_finish=False,
    )
    return recv_x, handle


def _gather_dispatch_inputs(x_fp8, x_scales, group, num_ranks):
    x_fp8 = x_fp8.contiguous()
    x_scales = x_scales.contiguous()
    fp8_list = [torch.empty_like(x_fp8) for _ in range(num_ranks)]
    scales_list = [torch.empty_like(x_scales) for _ in range(num_ranks)]
    dist.all_gather(fp8_list, x_fp8, group=group)
    dist.all_gather(scales_list, x_scales, group=group)
    return fp8_list, scales_list


def _check_recv_x_bitwise(recv_x, recv_x_scales, rank_prefix_matrix, recv_src_idx, gathered, rank):
    """Return (ok, err_msg). Same logic as test_intranode.check_recv_x_bitwise."""
    fp8_list, scales_list = gathered
    check_start = 0
    num_ranks = len(fp8_list)
    for src_rank in range(num_ranks):
        check_end = rank_prefix_matrix[src_rank][rank].item()
        if check_end == check_start:
            continue
        rows = recv_x[check_start:check_end]
        src_tokens = recv_src_idx[check_start:check_end]
        ref_x = fp8_list[src_rank][src_tokens]
        ref_scales = scales_list[src_rank][src_tokens]
        if not torch.equal(rows.view(torch.uint8), ref_x.view(torch.uint8)):
            return False, (
                f'fp8 mismatch src_rank={src_rank} recv_rows[{check_start}:{check_end})'
            )
        if not torch.equal(
            recv_x_scales[check_start:check_end].view(torch.uint8),
            ref_scales.view(torch.uint8),
        ):
            return False, (
                f'scales mismatch src_rank={src_rank} recv_rows[{check_start}:{check_end})'
            )
        check_start = check_end
    return True, ''


def _run_reference_check(
    buffer, x_fp8, x_scales, topk_idx, topk_weights,
    num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
    config, group, num_ranks, rank,
):
    gathered = _gather_dispatch_inputs(x_fp8, x_scales, group, num_ranks)
    recv_x, handle = _do_dispatch(
        buffer, x_fp8, x_scales, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, config,
    )
    if not isinstance(recv_x, tuple):
        return False, f'unexpected recv dtype {recv_x.dtype} (expected FP8 tuple)'

    rank_prefix_matrix, _, _, recv_src_idx, _, _, _ = handle
    ref_ok, ref_err = _check_recv_x_bitwise(
        recv_x[0], recv_x[1], rank_prefix_matrix, recv_src_idx, gathered, rank,
    )
    dist.barrier(group=group)
    return ref_ok, ref_err


def run_case(
    args: argparse.Namespace,
    num_sms: int,
    rank: int,
    num_ranks: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
    num_tokens: int,
    num_repeats: int,
    seed: int,
):
    hidden = args.hidden
    num_experts = args.num_experts
    num_topk = args.num_topk

    tokens, topk_idx, topk_weights = _make_inputs(
        num_tokens, hidden, num_experts, num_topk, seed, rank
    )
    x_fp8, x_scales = per_token_cast_to_fp8(tokens)
    x_fp8 = x_fp8.contiguous()
    x_scales = x_scales.contiguous()

    num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert = _build_layout(
        buffer, topk_idx, num_experts, num_ranks, args.use_kernel_layout,
    )
    config = deep_ep.Config(num_sms, 8, 256)

    ref_bytes = ref_scales = ref_raw = None
    ref_mask = None
    raw_shape = None
    m = 0

    bytes_ok = True
    scales_ok = True
    raw_ok = True
    fail_run = -1
    s_ndiff = 0
    s_total = 0
    s_max = 0.0
    raw_ndiff = 0
    raw_max = 0.0
    sample_rows = []

    for i in range(num_repeats):
        recv_x, handle = _do_dispatch(
            buffer, x_fp8, x_scales, topk_idx, topk_weights,
            num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, config,
        )
        if not isinstance(recv_x, tuple):
            if rank == 0:
                print(f'[N={num_tokens}] SKIP: backend did not use FP8 dispatch (dtype={recv_x.dtype})')
            return None

        rx, rs = recv_x
        if rs is None or rx.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            if rank == 0:
                print(f'[N={num_tokens}] SKIP: unexpected recv dtype (rx={rx.dtype}, rs={rs})')
            return None

        rx = rx.contiguous()
        rs = rs.contiguous()
        deq = per_token_cast_back(rx, rs)
        if ref_mask is None:
            ref_mask = _valid_row_mask(deq).cpu()
            raw_shape = tuple(rs.shape)
            m = rx.shape[0]

        vb = rx.view(torch.uint8).cpu()[ref_mask]
        vs = rs.float().cpu()[ref_mask]
        raw_s = rs.detach().float().reshape(-1).cpu()

        if ref_bytes is None:
            ref_bytes = vb.clone()
            ref_scales = vs.clone()
            ref_raw = raw_s.clone()
            s_total = int(ref_mask.sum().item())
        else:
            if not (ref_bytes.shape == vb.shape and torch.equal(ref_bytes, vb)):
                bytes_ok = False

            s_same = ref_scales.shape == vs.shape and torch.equal(ref_scales, vs)
            if not s_same:
                scales_ok = False
                if fail_run < 0:
                    fail_run = i
                    sd = (ref_scales - vs).abs()
                    s_ndiff = int((sd.sum(dim=1) > 0).sum())
                    s_max = float(sd.max())
                    bad = (sd.sum(dim=1) > 0).nonzero(as_tuple=False).flatten()[:3]
                    for bi in bad.tolist():
                        sample_rows.append({
                            'row': bi,
                            'ref_scale_mean': float(ref_scales[bi].mean()),
                            'cur_scale_mean': float(vs[bi].mean()),
                            'max_delta': float(sd[bi].max()),
                            'ngroups_wrong': int((sd[bi] > 0).sum()),
                        })

            r_same = ref_raw.shape == raw_s.shape and torch.equal(ref_raw, raw_s)
            if not r_same:
                raw_ok = False
                if fail_run < 0:
                    fail_run = i
                rd = (ref_raw - raw_s).abs()
                raw_ndiff = int((rd > 0).sum())
                raw_max = float(rd.max())

        dist.barrier(group=group)

    ref_ok, ref_err = _run_reference_check(
        buffer, x_fp8, x_scales, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        config, group, num_ranks, rank,
    )

    passed = bytes_ok and scales_ok and raw_ok and ref_ok
    result = {
        'num_tokens': num_tokens,
        'hidden_size': hidden,
        'recv_rows': m,
        'scales_shape': raw_shape,
        'num_repeats': num_repeats,
        'bytes_ok': bytes_ok,
        'scales_ok': scales_ok,
        'raw_ok': raw_ok,
        'ref_ok': ref_ok,
        'ref_err': ref_err,
        'passed': passed,
        'fail_run': fail_run,
        'valid_rows': s_total,
        'scale_diff_rows': s_ndiff,
        'scale_diff_pct': 100.0 * s_ndiff / max(s_total, 1),
        'max_delta_scale': s_max,
        'raw_diff_elems': raw_ndiff,
        'raw_max_delta': raw_max,
        'sample_rows': sample_rows,
    }

    if rank == 0:
        status = 'PASS' if passed else f'FAIL@run{fail_run}'
        print(f'\n{"=" * 60}')
        print(f'  N={num_tokens}, H={hidden}, repeats={num_repeats} -> {status}')
        print(f'  recv_rows={m}, scales.shape={raw_shape}, valid_rows={s_total}')
        print(f'  fp8 bytes bitwise equal: {bytes_ok}')
        print(f'  scales valid rows bitwise equal: {scales_ok} '
              f'(diff_rows={s_ndiff}/{s_total}, {result["scale_diff_pct"]:.2f}%, '
              f'max|delta|={s_max:.2e})')
        print(f'  scales raw buffer bitwise equal: {raw_ok} '
              f'(diff_elems={raw_ndiff}, max|delta|={raw_max:.2e})')
        print(f'  reference bitwise (recv_x + recv_scales): {ref_ok}')
        if ref_err:
            print(f'    {ref_err}')
        if sample_rows:
            print('  sample diff rows:')
            for s in sample_rows:
                print(f'    row={s["row"]}: ref_mean={s["ref_scale_mean"]:.4e}, '
                      f'cur_mean={s["cur_scale_mean"]:.4e}, '
                      f'max|delta|={s["max_delta"]:.4e}, '
                      f'wrong_groups={s["ngroups_wrong"]}/{hidden // 128}')
        print(f'{"=" * 60}', flush=True)

    return result


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    buffer = deep_ep.Buffer(
        group, int(2e9), 0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )

    seed = 42
    num_repeats = int(os.environ.get('DET_REPEATS', str(args.det_repeats)))

    if rank == 0:
        print('=' * 60)
        print('  intranode FP8 dispatch-only: determinism + reference check')
        print(f'  world_size={num_ranks}, experts={args.num_experts}, topk={args.num_topk}, seed={seed}')
        print(f'  layout={"kernel(get_dispatch_layout)" if args.use_kernel_layout else "manual(python)"}')
        print(f'  num_tokens={args.num_tokens}, repeats={num_repeats}')
        print('  phase1: repeat dispatch determinism (run0 vs run_i)')
        print('  phase2: reference bitwise via recv_src_idx (test_intranode.py)')
        print('=' * 60, flush=True)

    results = []
    r = run_case(
        args, args.num_sms, rank, num_ranks, buffer, group,
        args.num_tokens, num_repeats, seed,
    )
    if r is not None:
        results.append(r)
    dist.barrier(group=group)

    if rank == 0 and results:
        print('\nsummary:')
        print(f'{"N":>6} {"recv":>8} {"fp8":>8} {"scales":>8} {"raw":>8} {"ref":>8} '
              f'{"diff_rows":>12} {"max|dscale|":>12} {"result":>8}')
        for r in results:
            print(
                f'{r["num_tokens"]:>6} {r["recv_rows"]:>8} '
                f'{"OK" if r["bytes_ok"] else "FAIL":>8} '
                f'{"OK" if r["scales_ok"] else "FAIL":>8} '
                f'{"OK" if r["raw_ok"] else "FAIL":>8} '
                f'{"OK" if r["ref_ok"] else "FAIL":>8} '
                f'{r["scale_diff_rows"]:>5}/{r["valid_rows"]:<6} '
                f'{r["max_delta_scale"]:>12.2e} '
                f'{"PASS" if r["passed"] else "FAIL":>8}'
            )

    buffer.destroy()
    dist.barrier(group=group)
    dist.destroy_process_group()

    if results and not all(r['passed'] for r in results):
        sys.exit(1)


if __name__ == '__main__':
    parser = build_base_parser(
        'Intranode FP8 dispatch-only determinism + reference bitwise check',
        defaults={'num_tokens': 8000, 'hidden': 4096})
    add_sms_arg(parser, default=24)
    parser.add_argument('--det-repeats', type=int, default=500,
                        help='Dispatch repeats per case (default: 500)')
    parser.add_argument('--use-kernel-layout', action='store_true',
                        help='Use buffer.get_dispatch_layout() instead of manual Python layout')
    args = parser.parse_args()
    if int(os.environ.get('USE_KERNEL_LAYOUT', '0')):
        args.use_kernel_layout = True

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
