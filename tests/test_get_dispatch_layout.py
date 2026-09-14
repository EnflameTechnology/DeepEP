import argparse
import torch
import deep_ep
import torch.distributed as dist
from utils import init_dist, build_base_parser
torch.manual_seed(0)

def get_dispatch_layout_ref(topk_idx, num_experts, num_ranks, num_local_ranks=8):
    num_tokens = topk_idx.shape[0]
    topk = topk_idx.shape[1]
    num_rdma_ranks = int(num_ranks / num_local_ranks)
    assert(num_ranks % num_local_ranks == 0)
    num_tokens_per_expert = torch.zeros(num_experts, dtype = torch.int)
    num_tokens_per_rank = torch.zeros(num_ranks, dtype = torch.int)
    num_tokens_per_rdma_rank = torch.zeros(num_rdma_ranks, dtype = torch.int)
    is_token_in_rank_t = torch.zeros(num_ranks, num_tokens, dtype = torch.bool)
    num_experts_per_rank = int(num_experts / num_ranks)
    assert(num_experts % num_ranks == 0)
    for e in range(num_experts):
        num_tokens_per_expert[e] = (topk_idx == e).sum()
    topk_idx_rank = topk_idx // num_experts_per_rank
    for r in range(num_ranks):
        num_tokens_per_rank[r] = ((topk_idx_rank == r).sum(dim=1) >= 1).sum()


    topk_idx_rdma = topk_idx_rank // num_local_ranks
    assert(num_ranks % num_local_ranks == 0)
    for r in range(num_rdma_ranks):
        num_tokens_per_rdma_rank[r] = ((topk_idx_rdma == r).sum(dim=1) >= 1).sum()

    for r in range(num_ranks):
      is_token_in_rank_t[r, :] =  ((topk_idx_rank == r).sum(dim=1)) >= 1

    is_token_in_rank = is_token_in_rank_t.T.contiguous()

    return num_tokens_per_expert, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    print("local_rank", local_rank)
    print("num_local_ranks", num_local_ranks)
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_rdma_bytes = 0

    buffer = deep_ep.Buffer(group, int(2e9), num_rdma_bytes, low_latency_mode=False, num_qps_per_rank=1, explicitly_destroy=True)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='gcu').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    ref_num_tokens_per_expert, ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, ref_is_token_in_rank = get_dispatch_layout_ref(topk_idx, num_experts, num_ranks, num_local_ranks)
    #print("ref_num_tokens_per_expert", ref_num_tokens_per_expert)
    #print("num_tokens_per_expert", num_tokens_per_expert)
    torch.testing.assert_close(num_tokens_per_expert, ref_num_tokens_per_expert)


    #print("ref_is_token_in_rank", ref_is_token_in_rank)
    #print("is_token_in_rank", is_token_in_rank)
    torch.testing.assert_close(is_token_in_rank, ref_is_token_in_rank)


    #print("topk_ids", topk_idx)
    #print("ref num_tokens_per_rank", ref_num_tokens_per_rank)
    #print("num_tokens_per_rank", num_tokens_per_rank)

    torch.testing.assert_close(num_tokens_per_rank, ref_num_tokens_per_rank)
    if (num_tokens_per_rdma_rank is not None):
        #print("ref num_tokens_per_rdma_rank", ref_num_tokens_per_rdma_rank)
        #print("num_tokens_per_rdma_rank", num_tokens_per_rdma_rank)
        torch.testing.assert_close(num_tokens_per_rdma_rank, ref_num_tokens_per_rdma_rank)



if __name__ == '__main__':
    parser = build_base_parser('Test intranode EP kernels', defaults={'num_tokens': 4096})
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_main, args=(num_processes, args), nprocs=num_processes)
