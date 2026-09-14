import argparse
import os
import signal
import sys
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, calc_diff, hash_tensor, build_base_parser



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
    # We partition each destination bucket evenly across channels:
    #   channel c owns [base + c*step, base + (c+1)*step)
    # where step = per_dst / num_channels (require divisibility).
    # ---------------------------------------------------------------------

    # assert per_dst % num_channels == 0, f'per-dst({per_dst}) must be divisible by num_channels({num_channels})'
    per_dst_per_ch = (per_dst + num_channels - 1) // num_channels

    gbl_channel_prefix_matrix = torch.empty((num_ranks, num_channels), dtype=torch.int32, device=device)
    for dst_rank in range(num_ranks):
        base = dst_rank * per_dst
        for ch in range(num_channels):
            gbl_channel_prefix_matrix[dst_rank, ch] = min(base + ch * per_dst_per_ch, per_dst)

    # RDMA-level prefix over destination RDMA ranks (each RDMA rank contains 8 dst_ranks).
    per_dst_rdma = per_dst  # tokens destined to one dst_rdma_rank, 所有token都需要reduce。
    # assert per_dst_rdma % num_channels == 0
    per_dst_rdma_per_ch = (per_dst_rdma + num_channels - 1) // num_channels # 4
    rdma_channel_prefix_matrix = torch.empty((num_rdma_ranks, num_channels), dtype=torch.int32, device=device)
    for dst_rdma in range(num_rdma_ranks):
        # base = per_dst_rdma
        base = per_dst_rdma_per_ch
        for ch in range(num_channels):
            rdma_channel_prefix_matrix[dst_rdma, ch] = min(base + ch * per_dst_rdma_per_ch, per_dst_rdma)

    rdma_rank_prefix_sum = torch.cumsum(rdma_channel_prefix_matrix, dim=1).sum(dim=1, dtype=torch.int32).contiguous()

    num_combined_token = per_dst_rdma
    # num_rdma_recv_token = per_dst_rdma + (num_channels - 1) * (per_dst_rdma // num_channels)
    combined_nvl_head = torch.full((num_combined_token, 8), -1, dtype=torch.int32, device=device)
    combined_rdma_head = torch.full((num_combined_token, num_rdma_ranks), -1, dtype=torch.int32, device=device)
    for dst_rank in range(num_ranks):
        dst_rdma = dst_rank // 8
        dst_nvl = dst_rank % 8
        base = 0
        # idx = torch.arange(per_dst, dtype=torch.int32, device=device)
        idx = torch.randint(low=-100, high=101, size=(per_dst,), dtype=torch.int32, device=device)
        combined_nvl_head[base:base + per_dst, dst_nvl] = idx
        combined_rdma_head[base:base + per_dst, dst_rdma] = idx

    return (gbl_channel_prefix_matrix,
            rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum,
            combined_nvl_head,
            combined_rdma_head)

def cached_notify_ref(num_channels, rdma_channel_prefix_matrix, rdma_rank_prefix_sum, send_rdma_head, send_nvl_head):
    num_combined_tokens = send_rdma_head.shape[0]
    num_rdma_ranks = send_rdma_head.shape[1]
    cpu_rdma_head = torch.zeros((num_combined_tokens, num_rdma_ranks), dtype = torch.int)
    cpu_nvl_head = torch.zeros(send_nvl_head.shape, dtype = torch.int)

    for warp_idx in range(num_channels):
        num_tokens_per_channel = (num_combined_tokens + num_channels - 1) // num_channels
        token_start_idx = min(num_tokens_per_channel * warp_idx, num_combined_tokens)
        token_end_idx = min(token_start_idx + num_tokens_per_channel, num_combined_tokens)

        for lane_id in range(num_rdma_ranks):
            last_head = 1 << 25
            
            # 从后往前遍历（反向迭代）
            for token_idx in range(token_end_idx - 1, token_start_idx - 1, -1):
                current_head = send_rdma_head[token_idx, lane_id]
                if current_head < 0:
                    # 如果当前值是负数，用 -last_head - 1 填充
                    cpu_rdma_head[token_idx, lane_id] = -last_head - 1
                else:
                    # 否则更新last_head为当前值
                    cpu_rdma_head[token_idx, lane_id] = current_head
                    last_head = current_head

    num_max_nvl_peers = 8       # 固定为8卡机
    # 外层循环：遍历dst_rdma_rank
    for dst_rdma_rank in range(num_rdma_ranks):
        for warp_id in range(num_channels):
            # 计算token范围
            if warp_id == 0:
                token_start_idx = 0
            else:
                token_start_idx = rdma_channel_prefix_matrix[dst_rdma_rank, warp_id - 1]
            
            token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank, warp_id]
            
            # 计算shift
            if dst_rdma_rank == 0:
                shift = 0
            else:
                shift = rdma_rank_prefix_sum[dst_rdma_rank - 1]
            
            token_start_idx += shift
            token_end_idx += shift
            
            # NOTES: `1 << 25` 是一个启发式的大数值
            # last_head = 1 << 25
            last_head_per_lane = torch.full((num_max_nvl_peers,), 1 << 25, dtype=torch.int32)
            
            for token_idx in range(token_end_idx - 1, token_start_idx - 1, -1):
                # 遍历每个lane_id (对应NUM_MAX_NVL_PEERS)
                for lane_id in range(num_max_nvl_peers):
                    # 直接读取当前值
                    current_head = send_nvl_head[token_idx, lane_id]
                    
                    if current_head < 0:
                        # 如果当前值是负数，用 -last_head - 1 填充
                        cpu_nvl_head[token_idx, lane_id] = -last_head_per_lane[lane_id] - 1
                    else:
                        # 否则更新last_head为当前值
                        # last_head = current_head
                        cpu_nvl_head[token_idx, lane_id] = current_head
                        last_head_per_lane[lane_id] = current_head

    return cpu_rdma_head, cpu_nvl_head


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
    num_sms = 2
    hidden = args.hidden
    num_topk = args.num_topk
    assert num_local_ranks == 8

    # This test is intended to exercise the *internode* combine path specifically.
    # If the runtime is not initialized with multiple RDMA ranks, internode kernels are not available.



    # if buffer.runtime.get_num_rdma_ranks() <= 1:
    #     if local_rank == 0:
    #         print('[skip] internode kernels are unavailable (num_rdma_ranks <= 1)', flush=True)
    #     return 0

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
    num_channels = num_sms // 2
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

    print(f'rdma_channel_prefix_matrix, shape={rdma_channel_prefix_matrix.shape}, value={rdma_channel_prefix_matrix}', flush=True)
    print(f'gbl_channel_prefix_matrix, shape={gbl_channel_prefix_matrix.shape}, value={gbl_channel_prefix_matrix}', flush=True)
    print(f'rdma_rank_prefix_sum, shape={rdma_rank_prefix_sum.shape}, value={rdma_rank_prefix_sum}', flush=True)
    print(f'send_rdma_head, shape={send_rdma_head.shape}, value={send_rdma_head}', flush=True)
    print(f'send_nvl_head, shape={send_nvl_head.shape}, value={send_nvl_head}', flush=True)
    print(f'is_combined_token_in_rank, shape={is_combined_token_in_rank.shape}, value={is_combined_token_in_rank}', flush=True)
    print(f'src_meta, shape={src_meta.shape}, value={src_meta}', flush=True)
    print(f'num_channels, value={num_channels}', flush=True)

    # run cpu golden before gcu
    cpu_rdma_head, cpu_nvl_head = cached_notify_ref(num_channels,
                                                    rdma_channel_prefix_matrix,
                                                    rdma_rank_prefix_sum,
                                                    send_rdma_head,
                                                    send_nvl_head)

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
    x = (dst_rank_ids.to(torch.float32).unsqueeze(1).expand(num_tokens, hidden) + 1).to(torch.bfloat16)
    topk_weights = (dst_rank_ids.to(torch.float32).unsqueeze(1).expand(num_tokens, num_topk) + 1).contiguous()
    # No bias, so output should match input for this synthetic 1-to-1 mapping.
    bias_0 = None
    bias_1 = None


    # combined_x, combined_topk_weights, _ = buffer.combine(
    gcu_rdma_head, gcu_nvl_head, _ = buffer.combine(
        x=x,
        handle=handle,
        topk_weights=topk_weights,
        bias=(bias_0, bias_1),
        config=config,
        async_finish=False,
    )
    
    print(gcu_rdma_head is cpu_rdma_head) # 输出 True 则表示是同一对象
    print(id(gcu_rdma_head), id(cpu_rdma_head)) # 查看内存地址是否相同

    print("gcu rdma head, value: ", gcu_rdma_head)
    print("gcu nvl head, value: ", gcu_nvl_head)
    print("cpu rdma head, value: ", cpu_rdma_head)
    print("cpu nv head, value: ", cpu_nvl_head)
    torch.testing.assert_close(gcu_rdma_head, cpu_rdma_head)
    torch.testing.assert_close(gcu_nvl_head, cpu_nvl_head)

    return 0


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    print("after init dist, rank: {}, num_ranks: {}, group: {}".format(rank, num_ranks, group))
    log_file = f"{rank}_test_internode_cached_notify.log"
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
            num_sms = 2
            buffer = deep_ep.Buffer(group,
                                    int(2e9),
                                    int(1e9),
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
                               defaults={'num_processes': 8, 'num_tokens': 64})
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)

