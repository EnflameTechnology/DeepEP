import argparse
import os
import sys
import time
from pprint import pprint
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back, hash_tensor, build_base_parser, add_sms_arg, add_hardcode_local_ranks_arg
# Test compatibility with low latency functions
# import test_low_latency


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int,
              local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup, skip_benchmark: bool = False):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk_groups, num_topk, num_experts = args.num_topk_groups, args.num_topk, args.num_experts
    print(f"local_rank: {local_rank}, num_nodes: {num_nodes}, num_ranks: {num_ranks}, num_local_ranks: {num_local_ranks}, num_experts: {num_experts}")

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')
    x_e4m3 = per_token_cast_to_fp8(x)
    x_pure_rand_e4m3 = per_token_cast_to_fp8(x_pure_rand)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='gcu').abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='gcu') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)
    hash_value = 0

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='gcu')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='gcu')
    num_tokens_per_rdma_rank = torch.empty((num_nodes, ), dtype=torch.int, device='gcu')
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
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    # group.barrier()
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (48, 96, 144, 160) else 128)
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, ):
        for async_mode in (False, ):
            # for current_x in (x_pure_rand, x, x_pure_rand_e4m3, x_e4m3):
            for current_x in ((x, x_pure_rand, x, x_pure_rand)*5):
                for with_topk in (True, ):
                    is_rand = current_x is x_pure_rand or current_x is x_pure_rand_e4m3
                    if local_rank == 0:
                        print(f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...', flush=True, end='')
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,  'is_token_in_rank': is_token_in_rank,
                                     'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if is_rand else topk_weights})
                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()

                    if current_x is x_pure_rand or current_x is x:
                        hash_value += hash_tensor(recv_x)
                    else:
                        hash_value += hash_tensor(recv_x[0])
                        hash_value += hash_tensor(recv_x[1])
                    # recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-4]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if not is_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        # max_expert_idx = num_experts // num_ranks
                        # is_invalid = recv_topk_idx.eq(-1)
                        # is_valid_range = (recv_topk_idx >= 0) & (recv_topk_idx < max_expert_idx)
                        # all_valid = (is_invalid | is_valid_range).all()
                        # assert all_valid.item(), f'Invalid topk_idx values found. Expected: -1 or [0, {max_expert_idx}), got: {recv_topk_idx.unique().tolist()}'
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        if not is_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)
                    
                    # Test combine
                    bias_0 = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')
                    bias_1 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')

                    combine_args = {'x': recv_x, 'bias': (bias_0, bias_1), 'handle': handle, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
                    check_x = (combined_x.float()) / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if is_rand else x
                    # Compare check_x and ref_x row by row, first element
                    for idx in range(min(check_x.shape[0], ref_x.shape[0])):
                        check_first = check_x[idx, 0].item()
                        ref_first = ref_x[idx, 0].item()
                        if abs(check_first - ref_first) >= 0.001:
                            combined_first = combined_x[idx, 0].item()
                            is_token_sum_first = is_token_in_rank.sum(dim=1).unsqueeze(1)[idx, 0].item()
                            # print(f"Row {idx}: check_x[0]={check_first}, ref_x[0]={ref_first}, combined_x[0]={combined_first}, is_token_in_rank.sum(dim=1).unsqueeze(1)[0]={is_token_sum_first}")
                    assert calc_diff(check_x, ref_x) < 5e-4 if current_x is x_pure_rand_e4m3 else 5e-6 


                    hash_value += hash_tensor(recv_x)
                    # dist.barrier()
                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    if skip_benchmark:
        return hash_value
    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if args.test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9


    ## hack
    num_hardcode_local_ranks = args.num_hardcode_local_ranks 
    num_nodes = num_local_ranks // num_hardcode_local_ranks
    assert num_local_ranks % num_hardcode_local_ranks == 0
    num_ranks = num_local_ranks 
    num_local_ranks = num_hardcode_local_ranks

    log_file = f"{rank}_test_internode_single_node_sim.log"
    with open(log_file, 'w') as f:
        import sys
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)

        try:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = f
            sys.stderr = f

            print(f'Initialized rank {rank}/{num_ranks} on local rank {local_rank}, num_nodes {num_nodes}', flush=True)

            num_sms = args.num_sms
            num_qps_per_rank = max(num_sms, ll_num_experts // num_ranks if args.test_ll_compatibility else 0)

            buffer = deep_ep.Buffer(group, int(2e9), int(2e9), low_latency_mode=False,
                                    num_qps_per_rank=num_qps_per_rank, explicitly_destroy=True, is_internode=True)
            dist.barrier()

            seed = 0
            if local_rank == 0:
                print(f'Testing with seed {seed} ...', flush=True)
            torch.manual_seed(rank + seed)
            ref_hash = 0
            for i in (num_sms, ):
                ref_hash += test_main(args, i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group, args.pressure_test_mode == 1)
            dist.barrier()
            buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()
            print(f'Test internode successfully finished!', flush=True)
        finally:
            # sys.stdout = original_stdout
            # sys.stderr = original_stderr
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)



if __name__ == '__main__':
    parser = build_base_parser('Test internode EP kernels',
                               defaults={'num_processes': 16, 'num_tokens': 4096, 'num_experts': 256},
                               num_topk_groups=True)
    add_hardcode_local_ranks_arg(parser, default=8)
    add_sms_arg(parser, default=24)
    parser.add_argument('--pressure-test-mode', type=int, default=0,
                        help='Pressure test mode. 0: don\'t do pressure test, 1: do pressure test without benchmarks, 2: do pressure test with benchmarks')
    parser.add_argument('--test-ll-compatibility', action='store_true',
                        help='whether to test compatibility with low-latency kernels')
    args = parser.parse_args()

    # Set default `num_topk_groups` if not provided
    if args.num_topk_groups is None:
        num_nodes = int(os.getenv('WORLD_SIZE', 1))
        args.num_topk_groups = min(num_nodes, 4)

    os.environ["EP_INTRA_RANKS"] = str(args.num_hardcode_local_ranks)
    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
