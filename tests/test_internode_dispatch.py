import argparse
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import sys

import deep_ep
from utils import init_dist, build_base_parser

torch.manual_seed(1)


def make_channel_prefix_matrix(is_token_mat: torch.Tensor, channel_num: int, token_per_channel: int):
    """
    is_token_mat: [num_tokens, num_peer] (bool / int)
    return: [num_peer, channel_num] int32, cumsum over channel dim
    """
    num_tokens, num_peer = is_token_mat.shape
    total = channel_num * token_per_channel  # 一定 >= num_tokens（因为 token_per_channel 是 ceil）

    if total > num_tokens:
        pad = torch.zeros((total - num_tokens, num_peer), dtype=is_token_mat.dtype, device=is_token_mat.device)
        is_token_mat = torch.cat([is_token_mat, pad], dim=0)

    # [channel_num, token_per_channel, num_peer]
    v = is_token_mat.view(channel_num, token_per_channel, num_peer)

    # 每个 channel 内对 token 求和 -> [channel_num, num_peer]
    per_ch = v.sum(dim=1)

    # 转成 [num_peer, channel_num]，并做前缀和
    return per_ch.transpose(0, 1).cumsum(dim=1).to(torch.int32)


def test_main(
    buffer: deep_ep.Buffer,
    dist,
    hidden_bytes,
    num_topk,
    num_experts,
    num_tokens,
    rank,
    rank_num,
    channel_num,
    rdma_rank_num=2,
    esl_rank_num=8,
):
    # 生成测试数据
    num_expert_per_rank = num_experts // rank_num
    token_per_channel = (num_tokens + channel_num - 1) // channel_num

    x_list = []
    topk_idx_list = []
    topk_weights_list = []
    is_token_in_rank_list = []
    rdma_channel_prefix_matrix_list = []
    global_channel_prefix_matrix_list = []
    recv_global_rank_prefix_sum_list = []
    recv_rdma_rank_prefix_sum_list = []

    for r in range(rank_num):
        # x
        # x = torch.randint(0, 256, (num_tokens, hidden_bytes), dtype=torch.uint8)
        x = torch.full((num_tokens, hidden_bytes), fill_value=r + 1, dtype=torch.uint8)
        x_list.append(x)

        # topk_idx
        topk_idx = torch.rand(num_tokens, num_experts).topk(num_topk, dim=1).indices.int()
        topk_idx_list.append(topk_idx)

        # topk_weights
        topk_weights = F.normalize(torch.rand((num_tokens, num_topk), dtype=torch.float32), p=1, dim=1)
        topk_weights_list.append(topk_weights)

        # is_token_in_rank
        expert_ranks = topk_idx // num_expert_per_rank
        is_token_in_rank = torch.zeros(num_tokens, rank_num, dtype=torch.bool)
        token_indices = torch.arange(num_tokens).view(-1, 1).expand(-1, num_topk)
        is_token_in_rank[token_indices, expert_ranks] = True
        is_token_in_rank_list.append(is_token_in_rank)

        # rdma_channel_prefix_matrix
        is_token_in_rdma_rank = is_token_in_rank.view(num_tokens, rdma_rank_num, -1).any(dim=-1)
        rdma_channel_prefix_matrix = make_channel_prefix_matrix(is_token_in_rdma_rank, channel_num, token_per_channel)
        rdma_channel_prefix_matrix_list.append(rdma_channel_prefix_matrix)

        # global_channel_prefix_matrix
        global_channel_prefix_matrix = make_channel_prefix_matrix(is_token_in_rank, channel_num, token_per_channel)
        global_channel_prefix_matrix_list.append(global_channel_prefix_matrix)

    for r in range(rank_num):
        # recv_global_rank_prefix_sum
        recv_from_rank = torch.empty(rank_num, dtype=torch.int32)
        for src_rank in range(rank_num):
            recv_from_rank[src_rank] = is_token_in_rank_list[src_rank][:, r].sum()
        recv_global_rank_prefix_sum = recv_from_rank.cumsum(0).int()
        recv_global_rank_prefix_sum_list.append(recv_global_rank_prefix_sum)

        # recv_rdma_rank_prefix_sum
        esl_rank = r % esl_rank_num
        rdma_rank = r // esl_rank_num
        recv_from_rdma_peer = torch.empty(rdma_rank_num, dtype=torch.int32)
        local_slice = slice(rdma_rank * esl_rank_num, (rdma_rank + 1) * esl_rank_num)
        for rdma_id in range(rdma_rank_num):
            src_rank = rdma_id * esl_rank_num + esl_rank
            recv_from_rdma_peer[rdma_id] = is_token_in_rank_list[src_rank][:, local_slice].any(dim=1).sum()
        recv_rdma_rank_prefix_sum = recv_from_rdma_peer.cumsum(0).int()
        recv_rdma_rank_prefix_sum_list.append(recv_rdma_rank_prefix_sum)

    # 输出数据
    recv_rdma_channel_prefix_matrix = torch.zeros(rdma_rank_num, channel_num, dtype=torch.int32)
    recv_global_channel_prefix_matrix = torch.zeros(rank_num, channel_num, dtype=torch.int32)
    num_recv_tokens = recv_global_rank_prefix_sum_list[rank][-1].item()
    recv_src_meta = torch.zeros(num_recv_tokens, 2, dtype=torch.int32)
    send_rdma_head = torch.zeros(num_tokens, rdma_rank_num, dtype=torch.int32)
    num_rdma_recv_tokens = recv_rdma_rank_prefix_sum_list[rank][-1].item()
    send_esl_head = torch.zeros(num_rdma_recv_tokens, esl_rank_num, dtype=torch.int32)


    handle = (
        is_token_in_rank_list[rank],
        rdma_channel_prefix_matrix_list[rank],
        global_channel_prefix_matrix_list[rank],
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum_list[rank],
        recv_global_channel_prefix_matrix,
        recv_global_rank_prefix_sum_list[rank],
        recv_src_meta,
        send_rdma_head,
        send_esl_head,
    )

    dist.barrier()
    recv_x = buffer.dispatch(
        x_list[rank],
        handle,
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        is_token_in_rank=is_token_in_rank_list[rank],
        num_tokens_per_expert=None,
        topk_idx=topk_idx_list[rank],
        topk_weights=topk_weights_list[rank],
    )[0]

    print(f"rdma_rank_num = {buffer.runtime.get_num_rdma_ranks()}")
    print(global_channel_prefix_matrix_list[rank])
    print(recv_x)

    # 期望：本rank从每个src_rank接收的token数量
    expected = torch.tensor(
        [0] + [is_token_in_rank_list[src_rank][:, rank].sum().item() for src_rank in range(rank_num)],
        dtype=torch.int64,
    )

    # 1) 校验：每个token内部所有byte是否都一样
    # recv_x: [num_recv_tokens, hidden_bytes] uint8
    same_with_first_byte = (recv_x == recv_x[:, :1]).all(dim=1)
    bad = (~same_with_first_byte).nonzero(as_tuple=False).view(-1)
    if bad.numel() > 0:
        i = bad[0].item()
        print(f"[rank {rank}] recv_x token bytes mismatch at token_idx={i}, " f"first16={recv_x[i, :16].tolist()}")
        print(bad)

    # 2) 校验：按源rank计数（因为源rank被写进了token的每个byte）
    src_rank_val = recv_x[:, 0].to(torch.int64)  # 每个recv token的“源rank”
    counts = torch.bincount(src_rank_val, minlength=rank_num + 1)

    if not torch.equal(counts.cpu(), expected.cpu()):
        diff = (counts - expected).cpu().tolist()
        print(
            f"[rank {rank}] rank-count mismatch.\n"
            f"expected={expected.tolist()}\n"
            f"got     ={counts.tolist()}\n"
            f"diff    ={diff}"
        )
    else:
        print(f"[rank {rank}] recv_x check PASS. total={recv_x.shape[0]}, counts={counts.tolist()}")


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, rank_num, group = init_dist(local_rank, num_local_ranks)
    log_file = f"{rank}_test_internode_dispatch.log"

    with open(log_file, 'w') as f:
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)

        try:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = f
            sys.stderr = f

            print(f'Initialized rank {rank}/{rank_num} on local rank {local_rank}', flush=True)

            buffer = deep_ep.Buffer(
                group,
                int(2e9),
                int(2e9),
                low_latency_mode=False,
                num_qps_per_rank=args.channel_num,
                explicitly_destroy=True,
                is_internode=True,
            )
            buffer.set_num_sms(args.channel_num * 2)

            test_main(
                buffer,
                dist,
                args.hidden * 2,
                args.num_topk,
                args.num_experts,
                args.num_tokens,
                rank,
                rank_num,
                args.channel_num,
            )
            dist.barrier()
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
    parser = build_base_parser('test internode dispatch kernel',
                               defaults={'num_processes': 16, 'num_tokens': 16})
    parser.add_argument('--channel-num', type=int, default=12, help='Number of channels (default: 12)')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)