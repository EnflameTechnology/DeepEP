import argparse
import datetime
import os
import random
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, calc_diff, inplace_unique, per_token_cast_to_fp8, per_token_cast_back, build_base_parser, add_sms_arg, add_imbalance_arg, add_iteration_seed_args, build_topk_idx
from common.complex_scenario_tests import test_nonuniform_tokens_normal

# Test compatibility with low latency functions
import test_low_latency

DEFAULT_NUM_TOKENS = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
    4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288,
    13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480,
    21504, 22528, 23552, 24576, 25600, 26624, 27648, 28672,
    29696, 30720, 31744, 32768, 33792, 34816, 35840, 36864,
    37888, 38912, 39936, 40960, 41984, 43008, 44032, 45056,
    46080, 47104, 48128, 49152, 50176, 51200,
)

# Between dispatch and combine, each expert rank adds its local rank id so
# missing/wrong-rank combine bugs are visible after /count.
def _combine_ref_after_rank_add(src_x: torch.Tensor, is_token_in_rank: torch.Tensor,
                                num_ranks: int):
    # Match expert-side cast: (src + rank).to(src_x.dtype) before combine.
    ref_sum = torch.zeros_like(src_x, dtype=torch.float32)
    for r in range(num_ranks):
        mask = is_token_in_rank[:, r]
        transformed = (src_x[mask].float() + r).to(src_x.dtype).float()
        ref_sum[mask] += transformed
    count = is_token_in_rank.sum(dim=1).unsqueeze(1).float()
    return ref_sum / count


def _fmt_abbrev(vals, show_all: bool = False):
    """Abbreviate a 1-D list/tensor: first 8 + last 8; print all if n < 16."""
    n = len(vals)
    if n == 0:
        return '[]'
    def _f(v):
        return f'{float(v):.4g}'
    if show_all or n < 16:
        return '[' + ', '.join(_f(v) for v in vals) + ']'
    head, tail = 8, 8
    body = ', '.join(_f(v) for v in vals[:head]) + ', ..., ' + ', '.join(_f(v) for v in vals[-tail:])
    return f'[{body}]'


def _assert_close_with_dump(name: str, rank: int, actual: torch.Tensor, ref: torch.Tensor,
                            rtol: float, atol: float, max_tokens: int = 5, max_elems: int = 16):
    """Dump token-level mismatches (same criterion as assert_close), then assert_close.

    torch.testing.assert_close considers an element close iff:
        |actual - expected| <= atol + rtol * |expected|
    """
    assert actual.dim() == 2 and ref.shape == actual.shape
    diff = calc_diff(actual, ref)
    a, r = actual.float(), ref.float()
    abs_err = (a - r).abs()
    # Match assert_close: close when abs_err <= atol + rtol * |expected|.
    tol = atol + rtol * r.abs()
    elem_bad = abs_err > tol
    rel_err = abs_err / r.abs().clamp(min=torch.finfo(torch.float32).tiny)

    per_tok_n = elem_bad.sum(dim=1)
    per_tok_abs_max = torch.where(elem_bad, abs_err, torch.zeros_like(abs_err)).max(dim=1).values
    per_tok_rel_max = torch.where(elem_bad, rel_err, torch.zeros_like(rel_err)).max(dim=1).values
    tok_sel = per_tok_n > 0
    n_tok, hidden = a.shape
    n_sel = int(tok_sel.sum().item())
    n_bad = int(elem_bad.sum().item())

    if n_sel > 0:
        score = per_tok_abs_max.clone()
        score[~tok_sel] = -1.0
        order = torch.argsort(score * 1e3 + per_tok_rel_max, descending=True)
        dump_ids = [int(i) for i in order[:max_tokens].tolist() if tok_sel[int(i)]]
        show_all = (hidden <= max_elems) or ('topk' in name.lower())

        lines = [
            f'[warn][rank {rank}] {name}: calc_diff={diff:.6e}, shape={tuple(a.shape)}, '
            f'mismatched_elems={n_bad}/{abs_err.numel()}, '
            f'tokens_over_thresh={n_sel}/{n_tok} (atol={atol:g}, rtol={rtol:g})',
            f'    dump {len(dump_ids)}/{n_sel} worst tokens:',
        ]
        for tid in dump_ids:
            bad = elem_bad[tid]
            bad_idx = bad.nonzero(as_tuple=False).flatten()
            n_mm = int(bad_idx.numel())
            abs_tok = abs_err[tid][bad]
            rel_tok = rel_err[tid][bad]
            act_mm = a[tid][bad_idx].detach().cpu().tolist()
            ref_mm = r[tid][bad_idx].detach().cpu().tolist()
            lines.append(
                f'    token[{tid}]: mismatch {n_mm}/{hidden}, '
                f'abs avg/max={abs_tok.mean().item():.6e}/{abs_tok.max().item():.6e}, '
                f'rel avg/max={rel_tok.mean().item():.6e}/{rel_tok.max().item():.6e}')
            lines.append(f'      actual {_fmt_abbrev(act_mm, show_all)}')
            lines.append(f'      ref    {_fmt_abbrev(ref_mm, show_all)}')
        print('\n'.join(lines), flush=True)

    torch.testing.assert_close(
        actual.float(), ref.float(), rtol=rtol, atol=atol,
        msg=lambda m: f'{name} mismatch, calc_diff={diff}\n{m}')

# noinspection PyShadowingNames
def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              num_sms: int, local_rank: int, num_ranks: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup, seed: int = 0,
              expert_imbalance_ratio: float = 0.0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
    topk_idx = build_topk_idx(num_tokens, num_experts, num_topk,
                              expert_imbalance_ratio=expert_imbalance_ratio,
                              sorted_=False)
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
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
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

                    # Test `num_worst_tokens != 0`
                    if with_topk:
                        num_worst_tokens = num_tokens * num_ranks
                        dispatch_args.update({'num_worst_tokens': num_worst_tokens})
                        recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, handle_worst, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_worst_x = per_token_cast_back(*recv_worst_x) if isinstance(recv_worst_x, tuple) else recv_worst_x
                        assert len(empty_list) == 0
                        assert num_worst_tokens == recv_worst_x.size(0)
                        assert num_worst_tokens == recv_worst_topk_idx.size(0)
                        assert num_worst_tokens == recv_worst_topk_weights.size(0)
                        assert len(handle_worst) == 8
                        actual_num_recv_tokens = handle_worst[-1].item()
                        assert actual_num_recv_tokens == recv_x.size(0), f'{actual_num_recv_tokens} != {recv_x.size(0)}'
                        assert torch.equal(recv_x, recv_worst_x[:actual_num_recv_tokens])
                        assert torch.equal(recv_topk_idx, recv_worst_topk_idx[:actual_num_recv_tokens])
                        assert torch.equal(recv_topk_weights_clone, recv_worst_topk_weights[:actual_num_recv_tokens])

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

                    # Simulate expert compute: add local rank so combine must
                    # gather distinct payloads from each dispatch destination rank.
                    recv_x = (recv_x.float() + rank).to(recv_x.dtype)

                    # Test combine
                    combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
                    check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    src_x = x_pure_rand if current_x is x_pure_rand else x
                    ref_x = _combine_ref_after_rank_add(src_x, is_token_in_rank, num_ranks)
                    _assert_close_with_dump('combine x', rank, check_x, ref_x, rtol=1e-2, atol=1e-3)

                    if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        _assert_close_with_dump('combine topk_weights', rank,
                                                check_topk_weights, ref_topk_weights,
                                                rtol=1e-6, atol=1e-6)


                    # Test combine with pre-allocated `out` tensor (zero-copy path)
                    out = torch.zeros((num_tokens, hidden), dtype=recv_x.dtype, device=recv_x.device)
                    combine_args_out = dict(combine_args)
                    # async+out is tested only without allocate_on_comm_stream (incompatible)
                    combine_args_out['async_finish'] = False
                    combine_args_out['out'] = out
                    if previous_mode:
                        combine_args_out['previous_event'] = buffer.capture()
                    combined_x_out, _, _ = buffer.combine(**combine_args_out)
                    assert combined_x_out.data_ptr() == out.data_ptr(), \
                        'combine with out= should return the same tensor'
                    torch.testing.assert_close(
                        combined_x_out, combined_x, rtol=0.0, atol=0.0,
                        msg=lambda m: 'combine with out= should produce the same result '
                                      f'as without out=\n{m}')

                    # For later tuning
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in tuple(range(1, 2, 1)) + (0, ):
            if nvl_chunk_size > 0:
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            else:
                # Test default config as well
                deep_ep.Buffer.set_num_sms(num_sms)
                config = deep_ep.Buffer.get_dispatch_config(num_ranks)
            tune_args = {'x': current_x, 'handle': handle, 'config': config}
            t = bench(lambda: buffer.dispatch(**tune_args))[0]
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)
            if local_rank == 0:
                print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                      f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us', flush=True)
        if local_rank == 0:
            print(f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us', flush=True)
            print('', flush=True)

        # Gather the best config from rank 0 and the first test setting
        if best_dispatch_results is None:
            best_dispatch_results = torch.tensor([best_results[0], best_results[1]], dtype=torch.int32, device='gcu')
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size)

    dispatch_args = {'x': x, 'num_tokens_per_rank': num_tokens_per_rank,
                     'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
                     'config': dispatch_config if dispatch_config is not None else config}
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in tuple(range(1, 2, 1)) + (0, ):
        if nvl_chunk_size > 0:
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        else:
            # Test default config as well
            deep_ep.Buffer.set_num_sms(num_sms)
            config = deep_ep.Buffer.get_combine_config(num_ranks)
        tune_args = {'x': recv_x, 'handle': handle, 'config': config}
        t = bench(lambda: buffer.combine(**tune_args))[0]
        if local_rank == 0:
            print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                  f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us', flush=True)
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)

    if local_rank == 0:
        print(f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}: {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us', flush=True)
        print('', flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = f"{rank}_test_intranode_{log_timestamp}.log"

    # Redirect stdout and stderr to log file (mirrors test_low_latency.py):
    # one log file per rank, capturing Python prints and C++/subprocess output.
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

            test_ll_compatibility, num_rdma_bytes = False, 0
            if test_ll_compatibility:
                ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9
                num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(ll_num_tokens, ll_hidden, num_ranks, ll_num_experts)

            buffer = deep_ep.Buffer(group, int(2e9), num_rdma_bytes, low_latency_mode=test_ll_compatibility,
                                    num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1), explicitly_destroy=True)

            print(f'Rank {rank} buffer.runtime.is_available(): {buffer.runtime.is_available()}', flush=True)
            torch.manual_seed(rank)

            env_num_sms = os.environ.get('EP_DEBUG_COMBINE_USE_SM')

            num_sms = int(env_num_sms) if env_num_sms is not None else args.num_sms
            # SM tuning config shared with test_main (num_sms, nvl_chunk_size, nvl_buffer_size).
            nvl_buffer_size = 256
            config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

            # Pressure/imbalance loop, mirroring test_low_latency.py (lines 365-409):
            # --num-seeds controls the seed count (a large value effectively acts
            # as a pressure test); --token-imbalance-ratio != 0 additionally runs
            # the non-uniform per-rank token test on the normal dispatch/combine
            # path, combined with --expert-imbalance-ratio if set. Within each seed
            # the non-uniform hash is computed once as reference, then verified
            # across NUM_ITERATIONS iterations (hash-consistency / determinism check).
            NUM_ITERATIONS = args.num_iterations if args.num_iterations is not None else 50
            NUM_SEEDS = args.num_seeds if args.num_seeds is not None else 2
            imbalance_ratio = args.token_imbalance_ratio / 100.0
            # Unpack config once here (mirroring test_low_latency.py); test_main and
            # the non-uniform path receive these as plain parameters.
            num_tokens, hidden = args.num_tokens, args.hidden
            num_topk, num_experts = args.num_topk, args.num_experts
            for seed in range(NUM_SEEDS):
                if args.num_tokens is None:
                    num_tokens = random.Random(seed).choice(DEFAULT_NUM_TOKENS)
                if local_rank == 0:
                    print(f'Testing with seed {seed} (num_tokens={num_tokens}) ...', flush=True)
                for i in (num_sms, ):
                    test_main(num_tokens, hidden, num_experts, num_topk, i, local_rank,
                              num_ranks, rank, buffer, group, seed=seed,
                              expert_imbalance_ratio=args.expert_imbalance_ratio)
                    if local_rank == 0:
                        print('', flush=True)
                if args.token_imbalance_ratio != 0:
                    if local_rank == 0:
                        print(f'[seed={seed}] Non-uniform token test '
                              f'(imbalance_ratio={args.token_imbalance_ratio}%) — '
                              f'computing reference hash ...', flush=True)
                    # Reference hash from the FIRST run of this seed. (Different
                    # seeds produce different counts/data, so hashes are only
                    # comparable within one seed.)
                    ref_hash_nonuniform = test_nonuniform_tokens_normal(
                        hidden, num_experts, num_topk, rank, num_ranks,
                        buffer, config, seed=seed, imbalance_ratio=imbalance_ratio,
                        total_tokens=num_tokens * num_ranks,
                        host_jitter=args.host_jitter, device_jitter=args.device_jitter,
                        expert_imbalance_ratio=args.expert_imbalance_ratio)
                    for i in range(NUM_ITERATIONS):
                        print(f'[seed={seed}] Testing with iteration {i} ...', flush=True)
                        cur_hash = test_nonuniform_tokens_normal(
                            hidden, num_experts, num_topk, rank, num_ranks,
                            buffer, config, seed=seed, imbalance_ratio=imbalance_ratio,
                            total_tokens=num_tokens * num_ranks,
                            host_jitter=args.host_jitter, device_jitter=args.device_jitter,
                            expert_imbalance_ratio=args.expert_imbalance_ratio)
                        assert cur_hash == ref_hash_nonuniform, f'Error: seed={seed}'
                        if local_rank == 0 and i == NUM_ITERATIONS - 1:
                            print(f'[seed={seed}] Non-uniform token test passed.', flush=True)

            # Test compatibility with low latency functions
            if test_ll_compatibility:
                buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
                test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

            if local_rank == 0:
                print('Test intranode successfully finished!', flush=True)

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
    parser = build_base_parser('Test intranode EP kernels', defaults={'num_tokens': 4096})
    # --num-tokens: use the given value when specified; otherwise a random value
    # from DEFAULT_NUM_TOKENS is picked per seed in test_loop (all ranks agree
    # because the choice is derived from the seed). Override the shared default.
    parser.set_defaults(num_tokens=None)
    for action in parser._actions:
        if action.dest == 'num_tokens':
            action.help = (f'Number of tokens. If omitted, a random value per seed is picked '
                           f'from DEFAULT_NUM_TOKENS={DEFAULT_NUM_TOKENS}')
    add_sms_arg(parser, default=24)
    add_imbalance_arg(parser)
    add_iteration_seed_args(parser)
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
