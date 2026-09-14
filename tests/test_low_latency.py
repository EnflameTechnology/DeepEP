import argparse
import random
import time
import os
import torch
import torch.distributed as dist
from functools import partial
import datetime

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back, build_base_parser, add_low_latency_flags, add_imbalance_arg, add_iteration_seed_args, build_topk_idx
from common.boundary_data import create_boundary_test_data, BoundaryTestType
from common.complex_scenario_tests import test_nonuniform_tokens


FP8_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
DEFAULT_NUM_TOKENS = tuple(range(1, 17))


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
              use_logfmt: bool = False, seed: int = 0,
              test_data_type: BoundaryTestType = BoundaryTestType.ORIGINAL,
              test_per_tensor_scale: bool = False,
              expert_imbalance_ratio: float = 0.0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = create_boundary_test_data(num_tokens, hidden, rank, rank_offset, test_type=test_data_type, device='gcu')
    if num_tokens > 0:
        x[:, -128:] = torch.arange(num_tokens, device='gcu').to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for i in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * 0.5 * random.random())
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * 0.1)

    topk_idx = build_topk_idx(num_tokens, num_experts, num_topk,
                              expert_imbalance_ratio=expert_imbalance_ratio)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu').abs()

    # Randomly mask some positions (skip when num_tokens=0)
    if num_tokens > 0:
        for i in range(10):
            topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0

    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False, ):
                    for use_ue8m0 in (False, True) if round_scale else (False, ):
                        num_times += 1
                        for i in range((num_times % 2) + 1):
                            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='gcu')
                            packed_recv_x, packed_recv_count, handle, event, hook = \
                                buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                                            use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                                            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                            async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                            hook() if return_recv_hook else event.current_stream_wait()
                        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                        all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='gcu')
                        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)
                        simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
                            if dispatch_use_fp8 else packed_recv_x.clone()
                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
                            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

                            # GCU platform: C++ returns int32 tensor
                            # For single expert i: recv_layout_range shape is [num_ranks*2]
                            # where each pair (count, begin_idx) represents data from one rank
                            # Reshape to [num_ranks, 2] where [:, 0]=count, [:, 1]=begin_idx
                            recv_layout_reshaped = recv_layout_range.view(num_ranks, 2)
                            recv_counts = recv_layout_reshaped[:, 0]      # [num_ranks]
                            recv_begin_indices = recv_layout_reshaped[:, 1]   # [num_ranks]

                            # Check expert indices
                            num_valid_tokens = recv_count.item()
                            # TODO: Implement cumulative_local_expert_recv_stats check
                            # assert cumulative_local_expert_recv_stats[i].item() == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'
                            assert num_valid_tokens == recv_counts.sum().item(), f'{num_valid_tokens} != {recv_counts.sum().item()}'
                            assert num_valid_tokens == (all_topk_idx == expert_id).sum().item(), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}'

                            if num_valid_tokens == 0:
                                continue

                            # Check received data
                            if current_x is x:
                                recv_x = recv_x[:num_valid_tokens]
                                # TODO: topsAten amax check failed, why? use max instead temporarily
                                # recv_x_amax = recv_x[:, :-128].amax(dim=-1)
                                recv_src_info = recv_src_info[:num_valid_tokens]

                                if round_scale:
                                    diff_val = calc_diff(recv_x[:, -1], recv_src_info.view(-1))
                                    assert diff_val < 0.007, f'round_scale check failed: diff={diff_val}'
                                else:
                                    token_idx_diff = (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item()
                                    assert token_idx_diff == 0, f'token_idx check failed: diff_sum={token_idx_diff}'

                                if test_data_type == BoundaryTestType.ORIGINAL and not round_scale:
                                    recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                    recv_x_amax = recv_x[:, :-128].max(dim=-1).values
                                    assert torch.equal(recv_x_amin, recv_x_amax)
                                    for j in range(num_ranks):
                                        count = recv_counts[j].item()
                                        begin_idx = recv_begin_indices[j].item()
                                        assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
                                        assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0

                            # Calculate hash in rank order for reproducibility and easier debugging
                            for j in range(num_ranks):
                                count = recv_counts[j].item()
                                if count == 0:
                                    continue
                                begin_idx = recv_begin_indices[j].item()

                                if dispatch_use_fp8:
                                    hash_value ^= hash_tensor(packed_recv_x[0][i, begin_idx:begin_idx + count])
                                    hash_value ^= hash_tensor(packed_recv_x[1][i, begin_idx:begin_idx + count])
                                else:
                                    hash_value ^= hash_tensor(packed_recv_x[i, begin_idx:begin_idx + count])

                        for zero_copy in (False, ) if use_logfmt else (False, ): # TODO: Enable zero_copy testing
                            if zero_copy:
                                buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
                            out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')
                            combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                                            use_logfmt=use_logfmt,
                                                                            async_finish=not return_recv_hook, zero_copy=zero_copy,
                                                                            return_recv_hook=return_recv_hook, out=out)
                            hook() if return_recv_hook else event.current_stream_wait()
                            if do_check:
                                # 对于极端边界测试类型，combine 阶段的加权求和会自然溢出
                                # 这些类型只验证 dispatch 阶段的正确性，跳过 combine 精度检查
                                # num_tokens=0 时张量为空，calc_diff 的 .abs().max() 会抛出异常，跳过精度检查
                                skip_combine_precision_check = num_tokens == 0 or test_data_type in (
                                    BoundaryTestType.BF16_MAX,
                                    BoundaryTestType.BF16_MIN,
                                    BoundaryTestType.MIXED_EXTREME,
                                    BoundaryTestType.ROW_MIXED,
                                )

                                if skip_combine_precision_check:
                                    # 只计算 hash，不做精度检查
                                    hash_value ^= hash_tensor(combined_x)
                                else:
                                    diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
                                    assert torch.isnan(combined_x).sum().item() == 0, f'combined_x contains NaN: {torch.isnan(combined_x).sum().item()}'
                                    if not round_scale:
                                        # 算子的余弦相似度精度为 1e-2.Deepep目前的相似度为精度为 1e-5
                                        assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'
                                    hash_value ^= hash_tensor(combined_x)

    # ------------------------------------------------------------------
    # Per-tensor external FP8 scale dispatch test
    # fp8_quant_scale is a **scalar** (one value for the entire token tensor).
    # combine is NOT tested here: fp8_quant_scale only affects dispatch-side
    # quantization; the combine path is identical for per-group/per-tensor
    # and is already covered by the dispatch_use_fp8=True loop above.
    # ------------------------------------------------------------------
    if test_per_tensor_scale:
        for current_x in x_list:
            for return_recv_hook in (False, True):
                num_times += 1
                # Per-tensor dequant scale: single amax over the whole token tensor / fp8_max.
                # Shape [1] so that numel()==1 as required by C++ validation.
                fp8_quant_scale = (
                    current_x.abs().float().amax().clamp(min=1e-12) / FP8_E4M3_MAX
                ).unsqueeze(0)  # [1]
                for _iter in range((num_times % 2) + 1):
                    cumulative_local_expert_recv_stats = torch.zeros((num_local_experts,), dtype=torch.int, device='gcu')
                    packed_recv_x, packed_recv_count, handle, event, hook = \
                        buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                                    use_fp8=True, fp8_quant_scale=fp8_quant_scale,
                                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                    async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                hook() if return_recv_hook else event.current_stream_wait()

                recv_x_fp8 = packed_recv_x[0]
                recv_x_scales = packed_recv_x[1].contiguous()

                # Gather for validation
                all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='gcu')
                dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

                # Gather the scalar scale from each rank: one float per rank.
                all_fp8_scales = torch.empty((num_ranks,), dtype=torch.float32, device='gcu')
                dist.all_gather_into_tensor(all_fp8_scales, fp8_quant_scale, group=group)

                # Gather original x from all ranks for dequant comparison
                all_x = torch.empty((num_ranks, num_tokens, hidden), dtype=current_x.dtype, device='gcu')
                dist.all_gather_into_tensor(all_x, current_x, group=group)

                if do_check:
                    for expert_i in range(num_local_experts):
                        expert_id = rank * num_local_experts + expert_i
                        num_valid_tokens = packed_recv_count[expert_i].item()
                        recv_layout = handle[1][expert_i].view(num_ranks, 2)
                        recv_counts = recv_layout[:, 0]
                        recv_begin_indices = recv_layout[:, 1]

                        assert num_valid_tokens == recv_counts.sum().item(), \
                            f'Per-tensor scale expert {expert_id}: count mismatch'
                        assert num_valid_tokens == (all_topk_idx == expert_id).sum().item(), \
                            f'Per-tensor scale expert {expert_id}: topk count mismatch'

                        if num_valid_tokens == 0:
                            continue

                        # 1. All scale slots within each token must be identical
                        scales = recv_x_scales[expert_i, :num_valid_tokens]  # [valid, hidden/128]
                        assert torch.allclose(scales, scales[:, :1].expand_as(scales), rtol=1e-4), \
                            f'Expert {expert_id}: scale slots not uniform (per-tensor violated)'

                        # 2. Scale values must match the scalar fp8_quant_scale of the source rank.
                        #    all_fp8_scales[r] is a single float for the whole rank-r tensor.
                        src_info = handle[0][expert_i, :num_valid_tokens]
                        for r in range(num_ranks):
                            count = recv_counts[r].item()
                            begin = recv_begin_indices[r].item()
                            if count == 0:
                                continue
                            expected_s = all_fp8_scales[r]  # scalar: same for all tokens from rank r
                            received_s = scales[begin:begin + count, 0]  # [count]
                            assert torch.allclose(received_s, expected_s.expand_as(received_s), rtol=1e-4), \
                                f'Expert {expert_id} rank {r}: per-tensor scale value mismatch'

                        # 3. Dequantized FP8 must approximate original x.
                        dequant_x = per_token_cast_back(
                            recv_x_fp8[expert_i, :num_valid_tokens], scales).float()
                        ref_x = torch.zeros(
                            (num_valid_tokens, hidden), dtype=torch.float32, device='gcu')
                        for r in range(num_ranks):
                            count = recv_counts[r].item()
                            begin = recv_begin_indices[r].item()
                            if count > 0:
                                ref_x[begin:begin + count] = \
                                    all_x[r][src_info[begin:begin + count]].float()
                        diff = calc_diff(dequant_x, ref_x)
                        assert diff < 0.05, \
                            f'Expert {expert_id}: dequant diff too large: {diff:.4f} (threshold 0.05)'

                # Hash (rank-ordered for reproducibility)
                for expert_i in range(num_local_experts):
                    recv_layout = handle[1][expert_i].view(num_ranks, 2)
                    for j in range(num_ranks):
                        count = recv_layout[j, 0].item()
                        if count == 0:
                            continue
                        begin = recv_layout[j, 1].item()
                        hash_value ^= hash_tensor(recv_x_fp8[expert_i, begin:begin + count])
                        hash_value ^= hash_tensor(recv_x_scales[expert_i, begin:begin + count])

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                             use_logfmt=use_logfmt, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    # num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    # num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    # num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    # for i in range(num_tokens):
    #     num_selections = (topk_idx[i] != -1).sum().item()
    #     num_dispatch_comm_bytes += num_fp8_bytes * num_selections
    #     num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections

    # # Dispatch + combine testing
    # avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    # print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
    #       f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)

    # # Separate profiling
    # for return_recv_hook in (False, True):
    #     dist.barrier()
    #     dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
    #                                          kernel_names=('dispatch', 'combine'), barrier_comm_profiling=True,
    #                                          suppress_kineto_output=True, num_kernels_per_period=2 if return_recv_hook else 1)
    #     if not return_recv_hook:
    #         print(f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
    #               f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us', flush=True)
    #     else:
    #         print(f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
    #               f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us', flush=True)
    return hash_value



# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = f"{rank}_test_low_latency_{log_timestamp}.log"

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

            num_tokens, hidden = args.num_tokens, args.hidden
            num_topk, num_experts = args.num_topk, args.num_experts

            if args.num_tokens is not None:
                buffer_num_tokens = num_tokens
            else:
                num_tokens = random.Random(1).choice(DEFAULT_NUM_TOKENS)  # warm-up value
                buffer_num_tokens = max(DEFAULT_NUM_TOKENS)

            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(buffer_num_tokens, hidden, num_ranks, num_experts)
            if local_rank == 0:
                print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
            buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                                    num_qps_per_rank=num_experts // num_ranks,
                                    allow_nvlink_for_low_latency_mode=not args.disable_nvlink, explicitly_destroy=True,
                                    allow_mnnvl=args.allow_mnnvl)

            test_data_type = BoundaryTestType(args.test_data_type.lower())
            print(f'[Rank {rank}] Using test data type: {test_data_type.name} (value: {test_data_type.value})', flush=True)
            # Warm-up / basic correctness check (always runs, independent of mode)
            test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                    use_logfmt=args.use_logfmt, seed=1, test_data_type=test_data_type,
                    test_per_tensor_scale=args.test_per_tensor_scale,
                    expert_imbalance_ratio=args.expert_imbalance_ratio)

            NUM_ITERATIONS = args.num_iterations if args.num_iterations is not None else 50
            NUM_SEEDS = args.num_seeds if args.num_seeds is not None else 2
            imbalance_ratio = args.token_imbalance_ratio / 100.0

            # Seed loop — always runs regardless of imbalance-ratio flags.
            # A large --num-seeds value effectively acts as a pressure test.
            for seed in range(NUM_SEEDS):
                if args.num_tokens is None:
                    num_tokens = random.Random(seed).choice(DEFAULT_NUM_TOKENS)
                if local_rank == 0:
                    print(f'Testing with seed {seed} (num_tokens={num_tokens}) ...', flush=True)
                # Standard test: hash-consistency check across NUM_ITERATIONS.
                ref_hash = test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                                     use_logfmt=args.use_logfmt, seed=seed, test_data_type=test_data_type,
                                     test_per_tensor_scale=args.test_per_tensor_scale,
                                     expert_imbalance_ratio=args.expert_imbalance_ratio)
                if args.token_imbalance_ratio != 0:
                    if local_rank == 0:
                        print(f'[seed={seed}] Non-uniform token test '
                              f'(imbalance_ratio={args.token_imbalance_ratio}%) — computing reference hash ...',
                              flush=True)
                    ref_hash_nonuniform = test_nonuniform_tokens(hidden, num_experts, num_topk, rank, num_ranks, group,
                                                                 seed=seed, imbalance_ratio=imbalance_ratio,
                                                                 total_tokens=num_tokens * num_ranks,
                                                                 host_jitter=args.host_jitter,
                                                                 device_jitter=args.device_jitter,
                                                                 expert_imbalance_ratio=args.expert_imbalance_ratio)
                for i in range(NUM_ITERATIONS):
                    if args.token_imbalance_ratio == 0:
                        print(f'Testing with seed {seed} and iteration {i} ...', flush=True)
                        assert test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                                        use_logfmt=args.use_logfmt, seed=seed, test_data_type=test_data_type,
                                        test_per_tensor_scale=args.test_per_tensor_scale,
                                        expert_imbalance_ratio=args.expert_imbalance_ratio) == ref_hash, \
                            f'Error: seed={seed}'
                    else:
                        print(f'[seed={seed}] Testing with iteration {i} ...', flush=True)
                        assert test_nonuniform_tokens(hidden, num_experts, num_topk, rank, num_ranks, group,
                                              seed=seed, imbalance_ratio=imbalance_ratio,
                                              total_tokens=num_tokens * num_ranks,
                                              host_jitter=args.host_jitter,
                                              device_jitter=args.device_jitter,
                                              expert_imbalance_ratio=args.expert_imbalance_ratio) == ref_hash_nonuniform, \
                            f'Error: seed={seed}'
                        if local_rank == 0 and i == NUM_ITERATIONS - 1:
                            print(f'[seed={seed}] Non-uniform token test passed.', flush=True)

            if local_rank == 0:
                print('Test low-latency successfully finished!', flush=True)

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
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = build_base_parser('Test low-latency EP kernels', defaults={'num_experts': 288},
                               num_topk_groups=True)
    # --num-tokens: use the given value when specified; otherwise a random value
    # from DEFAULT_NUM_TOKENS is picked per seed in test_loop (all ranks agree
    # because the choice is derived from the seed). Override the shared default.
    parser.set_defaults(num_tokens=None)
    for action in parser._actions:
        if action.dest == 'num_tokens':
            action.help = (f'Number of tokens. If omitted, a random value per seed is picked '
                           f'from DEFAULT_NUM_TOKENS={DEFAULT_NUM_TOKENS} '
                           f'(buffer is sized for the largest candidate)')
    add_low_latency_flags(parser)
    add_imbalance_arg(parser)
    add_iteration_seed_args(parser)
    parser.add_argument('--test-per-tensor-scale', action='store_true',
                    help='Whether to test FP8 per-tensor external scale (fp8_quant_scale)')
    parser.add_argument('--test-data-type', type=str, default='original',
                        choices=[
                            # BF16 边界
                            'bf16_max', 'bf16_min', 'zero', 'tiny_positive',
                            'mixed_extreme', 'row_mixed', 'original',
                            # FP8 E4M3 量化边界
                            'fp8_e4m3_max', 'fp8_e4m3_overflow', 'fp8_e4m3_underflow',
                            'fp8_mixed_range', 'fp8_scale_boundary',
                            # UE8M0 Scale 边界
                            'ue8m0_scale_exact_pow2', 'ue8m0_scale_round_up',
                            'ue8m0_scale_min', 'ue8m0_scale_mixed',
                        ],
                        help='Boundary test type (default: original)')
    args = parser.parse_args()

    # Set default `num_topk_groups` if not provided
    if args.num_topk_groups is None:
        num_nodes = int(os.getenv('WORLD_SIZE', 1))
        args.num_topk_groups = min(num_nodes, 4)

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
