import argparse
import random
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Optional, List, Tuple, Dict, Any

# Constants
RANK_OFFSET = 128  # Offset used to ensure values fit within BF16 precision limits

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back, build_base_parser, add_low_latency_flags, add_iteration_seed_args

def create_test_data(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                     rank: int, num_ranks: int, use_logfmt: bool = False, seed: int = 0):
    """Create test data, only generate one set of data"""
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    assert num_ranks - RANK_OFFSET < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * (rank - RANK_OFFSET)
    x[:, -128:] = torch.arange(num_tokens, device='gcu').to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    # for i in range(4 if use_logfmt else 0):
    #     # NOTES: make more LogFMT casts and also with some BF16
    #     x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * 0.5 * random.random())
    # # NOTES: the last one is for performance testing
    # # Most of the values in the perf case is lower than the threshold, casting most channels
    # x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * 0.1)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='gcu').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu').abs()

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1
    
    return x_list, topk_idx, topk_weights

def run_combine_test(x_list: List[torch.Tensor], topk_idx: torch.Tensor, topk_weights: torch.Tensor, num_tokens: int, hidden: int, num_experts: int,
                      rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
                      return_recv_hook: bool = False, dispatch_use_fp8: bool = False,
                      round_scale: bool = False, use_ue8m0: bool = False,  zero_copy: bool = False, do_check: bool = True, use_logfmt: bool = False) -> int:

    hash_value = 0
    num_local_experts = num_experts // num_ranks
   
    # Iterate through all elements in x_list
    for current_x in x_list:
        # Reinitialize statistics for each input
        cumulative_local_expert_recv_stats = torch.zeros((num_local_experts,), dtype=torch.int, device='gcu')
        packed_recv_x, packed_recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
        hook() if return_recv_hook else event.current_stream_wait()
                        
        # Handle FP8 case
        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                        
        # Collect topk_idx from all ranks for validation
        all_topk_idx = torch.empty((num_ranks, num_tokens, topk_idx.shape[1]), dtype=topk_idx.dtype, device='gcu')
        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)
        simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
            if dispatch_use_fp8 else packed_recv_x.clone()                
        # Validate data for each local expert
        if do_check:
            for i in range(num_local_experts):
                expert_id = rank * num_local_experts + i
                recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
                recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]
                
                # GCU platform: C++ returns int32 tensor
                # For single expert i: recv_layout_range shape is [num_ranks*2]
                # where each pair (count, begin_idx) represents data from one rank
                # Reshape to [num_ranks, 2] where [:, 0]=count, [:, 1]=begin_idx
                recv_layout_reshaped = recv_layout_range.view(num_ranks, 2)
                recv_counts = recv_layout_reshaped[:, 0]
                recv_begin_indices = recv_layout_reshaped[:, 1]

                # Check expert index
                # TODO: Implement cumulative_local_expert_recv_stats check
                # assert cumulative_local_expert_recv_stats[i].item() == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'                
                num_valid_tokens = recv_count.item()
                assert num_valid_tokens == recv_counts.sum().item(), f'{num_valid_tokens} != {recv_counts.sum().item()}'
                assert num_valid_tokens == (all_topk_idx == expert_id).sum().item(), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}'

                if num_valid_tokens == 0:
                    continue
                                        
                # Check received data
                if current_x is x_list[0]:  # Only perform strict checks on the first (deterministic) tensor in x_list
                    recv_x = recv_x[:num_valid_tokens]
                    recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                    # TODO: topsAten amax check failed, why? use max instead temporarily
                    # recv_x_amax = recv_x[:, :-128].amax(dim=-1)                
                    recv_x_amax = recv_x[:, :-128].max(dim=-1).values
                    recv_src_info = recv_src_info[:num_valid_tokens]
                    assert torch.equal(recv_x_amin, recv_x_amax)
                                            
                    if round_scale:
                        assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
                    else:
                        assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                                                 
                    for j in range(num_ranks):
                        count = recv_counts[j].item()
                        begin_idx = recv_begin_indices[j].item()
                        if not round_scale:
                            assert (recv_x_amin == j - RANK_OFFSET).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
                            assert (recv_x[begin_idx:begin_idx + count, :-128] - j + RANK_OFFSET).sum().item() == 0
                                    
                # Calculate hash value for validation
                if dispatch_use_fp8:
                    hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                    hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
                else:
                    hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])
        if zero_copy:
            buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
        out = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device='gcu')
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                        use_logfmt=use_logfmt,
                                                        async_finish=not return_recv_hook, zero_copy=zero_copy,
                                                        return_recv_hook=return_recv_hook, out=out)
        hook() if return_recv_hook else event.current_stream_wait()
        if do_check:
            diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
            assert torch.isnan(combined_x).sum().item() == 0
            if not round_scale:
                assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'
            hash_value ^= hash_tensor(combined_x)

    return hash_value

def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
              use_logfmt: bool = False, seed: int = 0, test_fp8: bool = False) -> Dict[str, Any]:
    """Main test function, run tests using configuration list"""
    # Create test data
    x_list, topk_idx, topk_weights = create_test_data(
        num_tokens, hidden, num_experts, num_topk, rank, num_ranks, use_logfmt, seed
    )
    
    # Define test configurations, reference test_low_latency_perf.py
    # Configuration format: (config_name, return_recv_hook, dispatch_use_fp8, round_scale, use_ue8m0, zero_copy)
    # return_recv_hook: Whether to return receive hook instead of waiting for event
    # dispatch_use_fp8: Whether to use FP8 format for dispatch
    # round_scale: Whether to use rounding scaling
    # use_ue8m0: Whether to use UE8M0 format
    # zero_copy: Whether to use zero copy mode
    configurations = [
        ("BF16 (default, return_recv_hook=False, zero_copy=False)", False, False, False, False, False),  # Default config, no hook, no zero copy
        ("BF16 (return_recv_hook=True, zero_copy=False)", True, False, False, False, False),  # Use receive hook, no zero copy
        # Zero copy is not supported yet, temporarily commented out
        # ("BF16 (return_recv_hook=False, zero_copy=True)", False, False, False, False, True),  # No hook, use zero copy
        # ("BF16 (return_recv_hook=True, zero_copy=True)", True, False, False, False, True),  # Use receive hook, use zero copy
    ]
    
    # If FP8 test is enabled, add FP8 configurations
    if test_fp8:
        configurations.extend([
            ("FP8 (return_recv_hook=False, zero_copy=False)", False, True, False, False, False),
            ("FP8 (return_recv_hook=True, zero_copy=False)", True, True, False, False, False),
            ("FP8 with round_scale (return_recv_hook=False, zero_copy=False)", False, True, True, False, False),
            ("FP8 with round_scale (return_recv_hook=True, zero_copy=False)", True, True, True, False, False),
            ("FP8 with round_scale and use_ue8m0 (return_recv_hook=False, zero_copy=False)", False, True, True, True, False),
            ("FP8 with round_scale and use_ue8m0 (return_recv_hook=True, zero_copy=False)", True, True, True, True, False),
            ("FP8 (return_recv_hook=False, zero_copy=True)", False, True, False, False, True),
            ("FP8 (return_recv_hook=True, zero_copy=True)", True, True, False, False, True),
            ("FP8 with round_scale (return_recv_hook=False, zero_copy=True)", False, True, True, False, True),
            ("FP8 with round_scale (return_recv_hook=True, zero_copy=True)", True, True, True, False, True),
            ("FP8 with round_scale and use_ue8m0 (return_recv_hook=False, zero_copy=True)", False, True, True, True, True),
            ("FP8 with round_scale and use_ue8m0 (return_recv_hook=True, zero_copy=True)", True, True, True, True, True)
        ])
    
    # Run tests for all configurations
    results = {}
    combined_hash = 0  # Initialize combined hash value for XOR operation

    for config_name, return_recv_hook, dispatch_use_fp8, round_scale, use_ue8m0, zero_copy in configurations:
        # print(f"\nRunning configuration: {config_name}, return_recv_hook={return_recv_hook}, "
        #       f"dispatch_use_fp8={dispatch_use_fp8}, round_scale={round_scale}, use_ue8m0={use_ue8m0}", flush=True)
        
        # Run combine test instead of dispatch
        hash_value = run_combine_test(
            x_list, topk_idx, topk_weights, num_tokens, hidden, num_experts,
            rank, num_ranks, group, buffer,
            return_recv_hook=return_recv_hook, dispatch_use_fp8=dispatch_use_fp8,
            round_scale=round_scale, use_ue8m0=use_ue8m0, zero_copy=zero_copy, use_logfmt=use_logfmt
        )
        results[config_name] = hash_value
        combined_hash ^= hash_value  # XOR all hash values together
        # print(f"Configuration {config_name} completed with hash: {hash_value}", flush=True)
    
    # Return combined hash value (XOR of all configurations) to maintain consistency with test_low_latency.py
    return combined_hash

# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    """Main test loop, reference structure from test_low_latency_perf.py"""
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_file = f"{rank}_test_low_latency_combine.log"

    # Redirect output to log file
    with open(log_file, 'w') as f:
        import sys
        # Save original file descriptors
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)

        try:
            # Redirect file descriptors
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

            # Allocate buffer
            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
            if local_rank == 0:
                print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
            buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                                    num_qps_per_rank=num_experts // num_ranks,
                                    allow_nvlink_for_low_latency_mode=not args.disable_nvlink, explicitly_destroy=True,
                                    allow_mnnvl=args.allow_mnnvl)
            
            test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                    use_logfmt=args.use_logfmt, seed=1)

            NUM_ITERATIONS = args.num_iterations if args.num_iterations is not None else 300
            NUM_SEEDS = args.num_seeds if args.num_seeds is not None else 3
            for seed in range(NUM_SEEDS):
                if local_rank == 0:
                    print(f'Testing with seed {seed} ...', flush=True)
                ref_hash = test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                                    use_logfmt=args.use_logfmt, seed=seed)
                for i in range(NUM_ITERATIONS):
                    print(f'Testing with seed {seed} and iteration {i} ...', flush=True)
                    # get test_main return value
                    current_hash = test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                                           use_logfmt=args.use_logfmt, seed=seed)
                    print(f"test_main hash: {current_hash}, ref_hash: {ref_hash}")
                    assert current_hash == ref_hash, f'Error: seed={seed}'
            print(f'Test low-latency successfully finished!', flush=True)

            # Clean up resources
            buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()

        finally:
            # Restore original file descriptors and output streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)

if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = build_base_parser('Split test for low-latency EP kernels', defaults={'num_experts': 288})
    add_low_latency_flags(parser, test_fp8=True)
    add_iteration_seed_args(parser)
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
