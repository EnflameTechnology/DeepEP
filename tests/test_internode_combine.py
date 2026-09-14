import argparse
import os
import signal
import sys
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, calc_diff, hash_tensor, build_base_parser, add_sms_arg



def _build_uniform_all2all_meta(num_tokens: int,
                                num_ranks: int,
                                num_sms: int,
                                device: torch.device):
    """
    Build a *deterministic* uniform all-to-all layout under Convention A.

    Convention A:
      - The local input `x` is ordered by destination global rank (dst_rank = dst_rdma*8 + dst_nvl),
        and within each destination bucket by token index ascending.
      - We force chunked send token size = 1 at both NVL and RDMA levels by choosing config accordingly.

    Assumptions:
      - num_ranks is a multiple of 8 (NUM_MAX_NVL_PEERS).
      - num_tokens is divisible by num_ranks (uniform split).
      - num_sms is even

    Returns:
      - gbl_channel_prefix_matrix: int32 [num_ranks, num_channels] (segment start positions)
      - rdma_channel_prefix_matrix: int32 [num_rdma_ranks, num_channels] (segment start positions)
      - rdma_rank_prefix_sum: int32 [num_rdma_ranks] (cumulative token counts per dst_rdma_rank)
      - combined_nvl_head: int32 [num_tokens, 8]
      - combined_rdma_head: int32 [num_tokens, num_rdma_ranks]
    """
    assert num_ranks % 8 == 0
    num_channels = num_sms // 2
    num_rdma_ranks = num_ranks // 8

    assert num_tokens % num_ranks == 0, f'num_tokens({num_tokens}) must be divisible by num_ranks({num_ranks})'
    per_dst = num_tokens // num_ranks
    assert per_dst > 0

    # ---------------------------------------------------------------------
    # Prefix matrices (segment START positions for each (dst_rank, channel)).
    #
    # We partition each destination bucket across channels. We allow non-divisible splits:
    #   step = ceil(per_dst / num_channels)
    #   channel c owns [base + min(c*step, per_dst), base + min((c+1)*step, per_dst))
    # This makes the last channel process fewer tokens when per_dst is not divisible.
    # ---------------------------------------------------------------------
    per_dst_step = (per_dst + num_channels - 1) // num_channels
    assert per_dst_step > 0

    gbl_channel_prefix_matrix = torch.empty((num_ranks, num_channels), dtype=torch.int32, device=device)
    for dst_rank in range(num_ranks):
        base = dst_rank * per_dst
        for ch in range(num_channels):
            gbl_channel_prefix_matrix[dst_rank, ch] = base + min(ch * per_dst_step, per_dst)

    # RDMA-level prefix over destination RDMA ranks (each RDMA rank contains 8 dst_ranks).
    per_dst_rdma = per_dst  # 12 tokens destined to one dst_rdma_rank, 所有token都需要reduce。
    rdma_channel_prefix_matrix = torch.empty((num_rdma_ranks, num_channels), dtype=torch.int32, device=device)
    for dst_rdma in range(num_rdma_ranks):
        step = (per_dst_rdma + num_channels - 1) // num_channels
        for ch in range(num_channels):
            # cumulative token count up to channel ch (inclusive)
            rdma_channel_prefix_matrix[dst_rdma, ch] = min((ch + 1) * step, per_dst_rdma)

    # `rdma_channel_prefix_matrix[dst_rdma, ch]` is a cumulative count up to channel `ch` (inclusive),
    # so the total tokens for that dst_rdma is just the last channel entry.
    #
    # IMPORTANT: Do NOT do `cumsum(...).sum(...)` here — that "double integrates" and becomes huge when num_channels>1,
    # which will make the kernel advance `combined_nvl_head` by a wrong prefix and can hang in tail/head wait loops.
    rdma_rank_prefix_sum = rdma_channel_prefix_matrix[:, -1].contiguous()

    num_combined_token = per_dst_rdma
    combined_nvl_head = torch.full((num_combined_token * num_rdma_ranks, 32), -1, dtype=torch.int32, device=device)
    combined_rdma_head = torch.full((num_combined_token, num_rdma_ranks), -1, dtype=torch.int32, device=device)
    # IMPORTANT: the combine kernel uses per-channel ring buffers (addressing includes channel_id),
    # so head values must be **channel-local slot indices**, not "global index inside dst bucket".
    # For Convention A, each dst bucket is partitioned across channels with step=ceil(...), thus:
    #   local_slot = idx - channel_start(idx)
    for dst_rank in range(num_ranks):
        dst_rdma = dst_rank // 8
        dst_nvl = dst_rank % 8
        base = dst_rdma * num_combined_token
        idx = torch.arange(per_dst, dtype=torch.int32, device=device)
        # Determine per-token channel start by bucketing with step=ceil(per_dst/num_channels).
        # ch = min(idx // step, num_channels-1) (clamp protects when num_channels > per_dst)
        ch = (idx // per_dst_step).clamp(max=num_channels - 1)
        ch_start = ch * per_dst_step
        idx_local = idx - ch_start
        combined_nvl_head[base:base + per_dst, dst_nvl] = idx_local
        combined_rdma_head[0:0 + per_dst, dst_rdma] = idx_local

    return (gbl_channel_prefix_matrix,
            rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum,
            combined_nvl_head,
            combined_rdma_head)



# noinspection PyShadowingNames
def test_main(args: argparse.Namespace,
              num_sms: int,
              local_rank: int,
              num_local_ranks: int,
              num_ranks: int,
              num_nodes: int,
              rank: int,
              buffer: deep_ep.Buffer,
              group: dist.ProcessGroup):
    # Settings
    hidden = args.hidden
    num_topk = args.num_topk
    # assert num_local_ranks == 8

    # This test is intended to exercise the *internode* combine path specifically.
    # If the runtime is not initialized with multiple RDMA ranks, internode kernels are not available.
    if buffer.runtime.get_num_rdma_ranks() <= 1:
        if local_rank == 0:
            print('[skip] internode kernels are unavailable (num_rdma_ranks <= 1)', flush=True)
        return 0

    if local_rank == 0:
        print(f'[config] hidden={hidden}, num_topk={num_topk}', flush=True)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (24, 48, 96, 144, 160) else 512)
    # config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # -------------------------------------------------------------------------
    # Construct a SIMPLE, UNIFORM synthetic layout that actually processes tokens.
    #
    # Assumption (uniform distribution):
    # - Each GPU owns the same number of tokens (args.num_tokens).
    # - Each output token on a rank is contributed ONLY by itself (no cross-rank reduction),
    #   which still executes the full internode combine kernel pipeline with non-empty inputs.
    #
    # This avoids needing internode_dispatch while letting you "actually run" internode_combine
    # on real token data (non-empty x/topk_weights).
    # -------------------------------------------------------------------------
    num_tokens = args.num_tokens
    num_combined_tokens = num_tokens // num_ranks
    num_rdma_ranks = num_ranks // 8
    rdma_rank = rank // 8
    nvl_rank = rank % 8

    # Use a single channel to simplify prefix layout.
    # (The kernel supports any even num_sms; num_channels = num_sms // 2.)
    num_sms = args.num_sms
    num_channels = num_sms//2
    config = deep_ep.Config(num_sms, 4, nvl_buffer_size, 16, rdma_buffer_size)

    src_meta_bytes = 8  # sizeof(SourceMeta): {int src_rdma_rank, int is_token_in_nvl_rank_bits}
    device = torch.device('gcu')
    (gbl_channel_prefix_matrix,
     rdma_channel_prefix_matrix,
     rdma_rank_prefix_sum,
     send_nvl_head,
     send_rdma_head) = _build_uniform_all2all_meta(num_tokens, num_ranks, num_sms, device)

    # gbl_rank_prefix_sum is not used by the combine kernel hotpath; keep zeros.
    gbl_rank_prefix_sum = torch.zeros((num_ranks,), dtype=torch.int32, device='gcu')

    # Source meta for each input token: all tokens belong to THIS rank (src_rdma_rank==rdma_rank and bitmask selects nvl_rank).
    per_dst = num_tokens // num_ranks
    dst_rank_ids = (torch.arange(num_tokens, device=device, dtype=torch.int32) // per_dst).clamp(max=num_ranks - 1)
    src_meta_i32 = torch.empty((num_tokens, 2), dtype=torch.int32, device=device)
    src_meta_i32[:, 0] = dst_rank_ids // 8
    src_meta_i32[:, 1] = 1 << (dst_rank_ids % 8)
    src_meta = src_meta_i32.view(torch.uint8).reshape(num_tokens, src_meta_bytes).contiguous()

    # Heads: map token i -> head i for the single contributing lane; others set to -1.
    # NOTE: These heads are consumed by the combine pipeline to match sender->forwarder->receiver slots.
    is_combined_token_in_rank = torch.zeros((num_combined_tokens, num_ranks), dtype=torch.bool, device=device)
    is_combined_token_in_rank[torch.arange(num_combined_tokens, device=device),:] = True

    print("rdma_channel_prefix_matrix, shape: ", rdma_channel_prefix_matrix.shape, ", value: ", rdma_channel_prefix_matrix)
    print("gbl_channel_prefix_matrix, shape: ", gbl_channel_prefix_matrix.shape, ", value: ", gbl_channel_prefix_matrix)
    print("rdma_rank_prefix_sum, shape: ", rdma_rank_prefix_sum.shape, ", value: ", rdma_rank_prefix_sum)
    print("send_rdma_head, shape: ", send_rdma_head.shape, ", value: ", send_rdma_head)
    print("send_nvl_head, shape: ", send_nvl_head.shape, ", value: ", send_nvl_head)
    print("is_combined_token_in_rank, shape: ", is_combined_token_in_rank.shape, ", value: ", is_combined_token_in_rank)
    print("src_meta, shape: ", src_meta.shape, ", value: ", src_meta)

    # assert False
    handle = (
        is_combined_token_in_rank,
        rdma_channel_prefix_matrix,
        gbl_channel_prefix_matrix,
        rdma_channel_prefix_matrix,
        rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        gbl_rank_prefix_sum,
        src_meta,
        send_rdma_head,
        send_nvl_head,
    )

    # Real, non-empty inputs
    # x = (dst_rank_ids.to(torch.float32).unsqueeze(1).expand(num_tokens, hidden) + 1).to(torch.bfloat16)
    x = (torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')).contiguous()
    topk_weights = (dst_rank_ids.to(torch.float32).unsqueeze(1).expand(num_tokens, num_topk) + 1).contiguous()
    # No bias, so output should match input for this synthetic 1-to-1 mapping.
    bias_0 = None
    bias_1 = None

    dist.barrier()
    combined_x, combined_topk_weights, _ = buffer.combine(
        x=x,
        handle=handle,
        topk_weights=topk_weights,
        bias=(bias_0, bias_1),
        config=config,
        async_finish=False,
    )
    dist.barrier()

    assert combined_topk_weights is not None
    assert combined_x.shape == (num_combined_tokens, hidden)
    assert combined_topk_weights.shape == (num_combined_tokens, num_topk)
    
    # Check that all values in combined_x are 16
    expected_value = torch.tensor(16.0, dtype=torch.bfloat16, device=combined_x.device)
    expected_topk_weights = torch.tensor(16*(rank+1), dtype=torch.float32, device=combined_topk_weights.device)
    all_equal_topk_weights = torch.all(combined_topk_weights == expected_topk_weights)
    all_equal_16 = torch.all(combined_x == expected_value)
    if not all_equal_16:
        # Print some statistics for debugging
        unique_values = torch.unique(combined_x)
        min_val = combined_x.min().item()
        max_val = combined_x.max().item()
        mean_val = combined_x.float().mean().item()
        print(f"ERROR: combined_x values are not all 16!")
        print(f"  Min: {min_val}, Max: {max_val}, Mean: {mean_val}")
        print(f"  Unique values (first 20): {unique_values[:20]}")
        bad_mask = (combined_x != expected_value)
        bad_count = bad_mask.sum().item()
        print(f"  Number of values not equal to 16: {bad_count}")

        # Print up to 100 positions (token_idx, hidden_idx) and their values for debugging.
        bad_pos = torch.nonzero(bad_mask, as_tuple=False)
        n_show = min(200, bad_pos.shape[0])
        print(f"  First {n_show} bad positions (token_idx, hidden_idx) with values:")
        for i in range(n_show):
            ti = int(bad_pos[i, 0].item())
            hi = int(bad_pos[i, 1].item())
            v = combined_x[ti, hi].item()
            print(f"    [{i:03d}] token={ti}, hidden={hi}, value={v}")
        # Print all token indices that contain at least one bad element.
        bad_token_idx = torch.unique(bad_pos[:, 0]).cpu().tolist()
        print(f"  All bad token_idx (count={len(bad_token_idx)}): {bad_token_idx}")
    if not all_equal_topk_weights:
        # Print some statistics mirroring the combined_x block above.
        unique_values = torch.unique(combined_topk_weights)
        min_val = combined_topk_weights.min().item()
        max_val = combined_topk_weights.max().item()
        mean_val = combined_topk_weights.float().mean().item()
        print("ERROR: combined_topk_weights values do not match expected!")
        print(f"  Expected scalar: {expected_topk_weights.item()}")
        print(f"  Min: {min_val}, Max: {max_val}, Mean: {mean_val}")
        print(f"  Unique values (first 20): {unique_values[:20]}")

        bad_mask = (combined_topk_weights != expected_topk_weights)
        bad_count = bad_mask.sum().item()
        print(f"  Number of values not equal to expected: {bad_count}")

        bad_pos = torch.nonzero(bad_mask, as_tuple=False)
        n_show = min(200, bad_pos.shape[0])
        print(f"  First {n_show} bad positions (token_idx, topk_idx) with values:")
        for i in range(n_show):
            ti = int(bad_pos[i, 0].item())
            ki = int(bad_pos[i, 1].item())
            v = combined_topk_weights[ti, ki].item()
            print(f"    [{i:03d}] token={ti}, topk={ki}, value={v}")
        bad_token_idx = torch.unique(bad_pos[:, 0]).cpu().tolist()
        print(f"  All bad token_idx (count={len(bad_token_idx)}): {bad_token_idx}")
    # assert all_equal_16, f"combined_x should contain all 16s, but found values ranging from {combined_x.min().item()} to {combined_x.max().item()}"
    # assert all_equal_topk_weights, f"combined_topk_weights should contain all {expected_topk_weights.item()}s, but found values ranging from {combined_topk_weights.min().item()} to {combined_topk_weights.max().item()}"
    
    print("x, shape: ", x.shape, ", value: ", x)
    print("combined_x, shape: ", combined_x.shape, ", value: ", combined_x)
    print("topk_weights, shape: ", topk_weights.shape, ", value: ", topk_weights)
    print("combined_topk_weights, shape: ", combined_topk_weights.shape, ", value: ", combined_topk_weights)

    return hash_tensor(combined_x)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_file = f"{rank}_test_internode_combine.log"
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
            print(f'Initialized rank {rank}/{num_ranks} on local rank {local_rank}, device {torch.gcu.gcu_device()}', flush=True, file=f)
            num_sms = args.num_sms
            buffer = deep_ep.Buffer(group,
                                    int(2e9),
                                    int(2e9),
                                    low_latency_mode=False,
                                    num_qps_per_rank=num_sms,
                                    explicitly_destroy=True,
                                    is_internode=True)
            # assert num_local_ranks == 8 and num_ranks > 8

            # One-shot by default; keep the same structure as other tests.
            for seed in range(1):
                if local_rank == 0:
                    print(f'Testing combine-only with seed {seed} ...', flush=True)
                torch.manual_seed(rank + seed)
                _ = test_main(args, num_sms, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group)

            dist.barrier()
            buffer.destroy()
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
    parser = build_base_parser('Test internode EP combine kernel only (synthetic inputs)',
                               defaults={'num_processes': 16, 'num_tokens': 128})
    add_sms_arg(parser, default=24)
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)

