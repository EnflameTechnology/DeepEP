import argparse
import random
import os
from sys import stderr
import torch
import torch.distributed as dist
from functools import partial
import datetime

import deep_ep
from utils import init_dist, bench, bench_kineto, per_token_cast_back, draw_latency_chart, draw_bandwidth_chart, get_performance_data, build_base_parser, add_fp8_dispatch_arg, add_kineto_args


def create_test_data(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                     rank: int, num_ranks: int, use_logfmt: bool = False, seed: int = 0):
    """Create test data for performance testing."""
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    # Create test data for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='gcu') * 0.1

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='gcu').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='gcu').abs()

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    return x, topk_idx, topk_weights


def run_performance_test(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                         rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
                         use_logfmt: bool = False, seed: int = 0, use_fp8: bool = False,
                         num_processes: int = 1, output_dir: str = None):
    """Run performance tests for low-latency dispatch and combine."""
    x, topk_idx, topk_weights = create_test_data(num_tokens, hidden, num_experts, num_topk,
                                                  rank, num_ranks, use_logfmt, seed)
    
    num_local_experts = num_experts // num_ranks
    cumulative_local_expert_recv_stats = torch.zeros((num_local_experts,), dtype=torch.int, device='gcu')
    
    # Dispatch to get handle and simulated data
    packed_recv_x, packed_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    use_fp8=use_fp8, async_finish=True, return_recv_hook=False)
    event.current_stream_wait()
    
    # Handle FP8 case: packed_recv_x is a tuple (x_fp8, x_scales) when use_fp8=True
    if use_fp8:
        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous())
        simulated_gemm_x = per_token_cast_back(
            packed_recv_x[0].view(-1, hidden), 
            packed_recv_x[1].view(-1, hidden // 128)
        ).view(packed_recv_x[0].shape)
    else:
        simulated_gemm_x = packed_recv_x.clone()

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=use_fp8, async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                             use_logfmt=use_logfmt, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes = hidden + hidden / 128 * 4 + 16  # FP8 data + scales
    num_bf16_bytes = hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += (num_fp8_bytes if use_fp8 else num_bf16_bytes) * num_selections
        num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    
    # Print per-rank results for verification
    print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)
    
    # Aggregate results across all ranks and compute average
    timing_tensor = torch.tensor([avg_t, min_t, max_t, num_dispatch_comm_bytes, num_combine_comm_bytes], 
                                  dtype=torch.float64, device='gcu')
    dist.all_reduce(timing_tensor, op=dist.ReduceOp.SUM, group=group)
    timing_tensor /= num_ranks
    
    # Synchronize to ensure all_reduce is complete before reading values
    torch.gcu.synchronize()
    
    avg_t_mean, min_t_mean, max_t_mean = timing_tensor[0].item(), timing_tensor[1].item(), timing_tensor[2].item()
    num_dispatch_comm_bytes_mean, num_combine_comm_bytes_mean = timing_tensor[3].item(), timing_tensor[4].item()
    
    if rank == 0:
        print(f'[avg across {num_ranks} ranks] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes_mean + num_combine_comm_bytes_mean) / 1e9 / avg_t_mean:.2f} GB/s, '
              f'avg_t={avg_t_mean * 1e6:.2f} us, min_t={min_t_mean * 1e6:.2f} us, max_t={max_t_mean * 1e6:.2f} us', flush=True)

    # Separate profiling
    results = {}
    for return_recv_hook in (False, ): #TODO: enable true?
        dist.barrier()
        dispatch_stats, combine_stats = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
                                                     kernel_names=('dispatch', 'combine'), barrier_comm_profiling=True,
                                                     suppress_kineto_output=True, num_kernels_per_period=2 if return_recv_hook else 1,
                                                     return_stats=True)
        if num_ranks != num_processes:
            test_name = '_'.join(['deepep', 'low_latency', 'perf', str(int(num_ranks/num_processes)), 'nodes', str(num_processes), 'nodes', str(num_ranks), 'cards', str(num_processes), 'processes', 'performance', str(num_tokens), 'tokens', str(hidden), 'hidden', str(num_topk), 'topk', str(num_experts), 'experts', 'fp8' if use_fp8 else 'bf16'])
        else:
            test_name = '_'.join(['deepep', 'low_latency', 'perf', str(num_ranks), 'cards', str(num_processes), 'processes', 'performance', str(num_tokens), 'tokens', str(hidden), 'hidden', str(num_topk), 'topk', str(num_experts), 'experts', 'fp8' if use_fp8 else 'bf16'])

        results[test_name] = {
            'mode': 'low_latency',
            'use_fp8': use_fp8,
            'num_tokens': num_tokens,
            'hidden': hidden,
            'num_topk': num_topk,
            'num_experts': num_experts,
            'num_ranks': num_ranks,
            'return_recv_hook': return_recv_hook
        }

        if not return_recv_hook:
            # Extract stats (single value for each kernel)
            dispatch_avg = dispatch_stats['avg']
            dispatch_min = dispatch_stats['min']
            dispatch_max = dispatch_stats['max']
            combine_avg = combine_stats['avg']
            combine_min = combine_stats['min']
            combine_max = combine_stats['max']
            
            dispatch_bandwidth = num_dispatch_comm_bytes / 1e9 / dispatch_avg
            combine_bandwidth = num_combine_comm_bytes / 1e9 / combine_avg
            dispatch_latency_us = dispatch_avg * 1e6
            combine_latency_us = combine_avg * 1e6
            
            # Print per-rank results for verification
            print(f'[rank {rank}] Dispatch bandwidth: {dispatch_bandwidth:.2f} GB/s, avg_t={dispatch_latency_us:.2f} us, '
                  f'min_t={dispatch_min * 1e6:.2f} us, max_t={dispatch_max * 1e6:.2f} us | '
                  f'Combine bandwidth: {combine_bandwidth:.2f} GB/s, avg_t={combine_latency_us:.2f} us, '
                  f'min_t={combine_min * 1e6:.2f} us, max_t={combine_max * 1e6:.2f} us', flush=True)
            
            # Aggregate stats across all ranks and compute average
            stats_tensor = torch.tensor([
                dispatch_latency_us, dispatch_min * 1e6, dispatch_max * 1e6, dispatch_bandwidth,
                combine_latency_us, combine_min * 1e6, combine_max * 1e6, combine_bandwidth
            ], dtype=torch.float64, device='gcu')
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM, group=group)
            stats_tensor /= num_ranks
            
            # Synchronize to ensure all_reduce is complete before reading values
            torch.gcu.synchronize()
            
            dispatch_latency_us_mean = round(stats_tensor[0].item(), 2)
            dispatch_min_us_mean = round(stats_tensor[1].item(), 2)
            dispatch_max_us_mean = round(stats_tensor[2].item(), 2)
            dispatch_bandwidth_mean = round(stats_tensor[3].item(), 2)
            combine_latency_us_mean = round(stats_tensor[4].item(), 2)
            combine_min_us_mean = round(stats_tensor[5].item(), 2)
            combine_max_us_mean = round(stats_tensor[6].item(), 2)
            combine_bandwidth_mean = round(stats_tensor[7].item(), 2)
            results[test_name]['dispatch_latency_us_mean'] = dispatch_latency_us_mean
            results[test_name]['dispatch_bandwidth_mean'] = dispatch_bandwidth_mean
            results[test_name]['combine_latency_us_mean'] = combine_latency_us_mean
            results[test_name]['combine_bandwidth_mean'] = combine_bandwidth_mean
            
            results['dispatch'] = {
                'latency_us': dispatch_latency_us_mean,
                'latency_us_min': dispatch_min_us_mean,
                'latency_us_max': dispatch_max_us_mean,
                'bandwidth_gbps': dispatch_bandwidth_mean
            }
            results['combine'] = {
                'latency_us': combine_latency_us_mean,
                'latency_us_min': combine_min_us_mean,
                'latency_us_max': combine_max_us_mean,
                'bandwidth_gbps': combine_bandwidth_mean
            }

            if rank == 0:
                print(f'[avg across {num_ranks} ranks] Dispatch bandwidth: {dispatch_bandwidth_mean:.2f} GB/s, avg_t={dispatch_latency_us_mean:.2f} us | '
                      f'Combine bandwidth: {combine_bandwidth_mean:.2f} GB/s, avg_t={combine_latency_us_mean:.2f} us', flush=True)

                if output_dir is not None:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    get_performance_data(output_dir, test_name, results)
        else:
            # For return_recv_hook=True, stats is a list of dicts (one for each period component)
            dispatch_send_avg = dispatch_stats[0]['avg']
            dispatch_recv_avg = dispatch_stats[1]['avg']
            combine_send_avg = combine_stats[0]['avg']
            combine_recv_avg = combine_stats[1]['avg']
            
            # Print per-rank results for verification
            print(f'[rank {rank}] Dispatch send/recv time: {dispatch_send_avg * 1e6:.2f} + {dispatch_recv_avg * 1e6:.2f} us | '
                  f'Combine send/recv time: {combine_send_avg * 1e6:.2f} + {combine_recv_avg * 1e6:.2f} us', flush=True)
            
            # Aggregate stats across all ranks and compute average
            hook_stats_tensor = torch.tensor([
                dispatch_send_avg, dispatch_recv_avg, combine_send_avg, combine_recv_avg
            ], dtype=torch.float64, device='gcu')
            dist.all_reduce(hook_stats_tensor, op=dist.ReduceOp.SUM, group=group)
            hook_stats_tensor /= num_ranks

            if rank == 0:
                print(f'[avg across {num_ranks} ranks] Dispatch send/recv time: {hook_stats_tensor[0].item() * 1e6:.2f} + {hook_stats_tensor[1].item() * 1e6:.2f} us | '
                      f'Combine send/recv time: {hook_stats_tensor[2].item() * 1e6:.2f} + {hook_stats_tensor[3].item() * 1e6:.2f} us', flush=True)
    
    return results


def run_benchmark_with_plotting(max_num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                                rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
                                use_logfmt: bool = False, seed: int = 0, disable_nvlink: bool = False,
                                allow_mnnvl: bool = False, use_log_scale: bool = False, dispatch_use_fp8: bool = False,
                                num_processes: int = 1):
    """Run benchmark with different num_tokens and hidden values, then plot results."""
    # Define test parameters
    num_tokens_list = list(range(4, max_num_tokens + 1, 4))  # 4, 8, 12, ..., 128
    hidden_list = [4096, 7168]
    
    # Collect data for plotting (only on rank 0)
    dispatch_latency_data = []
    combine_latency_data = []
    dispatch_bandwidth_data = []
    combine_bandwidth_data = []
    
    if rank == 0:
        print(f'[rank {rank}] Starting benchmark with plotting...', flush=True)
        print(f'[rank {rank}] Testing num_tokens: {num_tokens_list}', flush=True)
        print(f'[rank {rank}] Testing hidden: {hidden_list}', flush=True)
    
    for hidden_size in hidden_list:
        for tokens in num_tokens_list:
            if rank == 0:
                print(f'[rank {rank}] Testing hidden={hidden_size}, num_tokens={tokens}, use_fp8={dispatch_use_fp8}...', flush=True)
            
            # Run performance test (all ranks participate) - reuse the same buffer
            results = run_performance_test(tokens, hidden_size, num_experts, num_topk, rank, num_ranks, group, buffer,
                                         use_logfmt=use_logfmt, seed=seed, use_fp8=dispatch_use_fp8,
                                         num_processes=num_processes)
            
            # Collect data (only on rank 0)
            if rank == 0:
                if 'dispatch' in results:
                    dispatch_latency_data.append({
                        'hidden': hidden_size,
                        'num_tokens': tokens,
                        'avg': results['dispatch']['latency_us'],
                        'min': results['dispatch']['latency_us_min'],
                        'max': results['dispatch']['latency_us_max']
                    })
                    dispatch_bandwidth_data.append({
                        'hidden': hidden_size,
                        'num_tokens': tokens,
                        'bandwidth': results['dispatch']['bandwidth_gbps']
                    })
                
                if 'combine' in results:
                    combine_latency_data.append({
                        'hidden': hidden_size,
                        'num_tokens': tokens,
                        'avg': results['combine']['latency_us'],
                        'min': results['combine']['latency_us_min'],
                        'max': results['combine']['latency_us_max']
                    })
                    combine_bandwidth_data.append({
                        'hidden': hidden_size,
                        'num_tokens': tokens,
                        'bandwidth': results['combine']['bandwidth_gbps']
                    })
            
            dist.barrier()
    dist.barrier()
    
    # Plot results (only on rank 0)
    if rank == 0:
        format_suffix = '_fp8' if dispatch_use_fp8 else '_bf16'
        format_name = format_suffix[1:]  # Remove leading underscore
        
        if dispatch_latency_data:
            draw_latency_chart(dispatch_latency_data, 
                             output_path=f'dispatch_latency_chart{format_suffix}.png',
                             title=f'Dispatch Performance Analysis ({format_name.upper()})',
                             use_log_scale=use_log_scale)
        
        if combine_latency_data:
            draw_latency_chart(combine_latency_data,
                             output_path=f'combine_latency_chart{format_suffix}.png',
                             title=f'Combine Performance Analysis',
                             use_log_scale=use_log_scale)
        
        if dispatch_bandwidth_data:
            draw_bandwidth_chart(dispatch_bandwidth_data,
                               output_path=f'dispatch_bandwidth_chart{format_suffix}.png',
                               title=f'Dispatch Bandwidth Analysis ({format_name.upper()})')
        
        if combine_bandwidth_data:
            draw_bandwidth_chart(combine_bandwidth_data,
                               output_path=f'combine_bandwidth_chart{format_suffix}.png',
                               title=f'Combine Bandwidth Analysis')
        
        print(f'[rank {rank}] Charts saved successfully!', flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = f"{rank}_test_low_latency_perf_{log_timestamp}.log"

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

            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
            if local_rank == 0:
                print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
            buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                                    num_qps_per_rank=num_experts // num_ranks,
                                    allow_nvlink_for_low_latency_mode=True, explicitly_destroy=True,
                                    allow_mnnvl=False)
            try:
                # Run performance tests
                if args.enable_plotting:
                    # Run benchmark with plotting (all ranks participate)
                    run_benchmark_with_plotting(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                                            use_logfmt=False, seed=1,
                                            disable_nvlink=False, allow_mnnvl=False,
                                            use_log_scale=args.use_log_scale,
                                            dispatch_use_fp8=args.dispatch_use_fp8,
                                            num_processes=args.num_processes)
                else:
                    run_performance_test(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                                    use_logfmt=False, seed=1, use_fp8=args.dispatch_use_fp8,
                                    num_processes=args.num_processes, output_dir=args.output_dir)
                
                print(f'Performance test successfully finished!', flush=True)
            finally:
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
    parser = build_base_parser('Performance test for low-latency EP kernels', defaults={'num_experts': 288})
    parser.add_argument('--enable-plotting', action='store_true',
                       help='Enable plotting latency and bandwidth charts')
    parser.add_argument('--use-log-scale', action='store_true',
                       help='Use logarithmic scale for x-axis in latency charts (default: False)')
    add_fp8_dispatch_arg(parser)
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for performance JSON data. If not specified, no JSON file is saved.')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)