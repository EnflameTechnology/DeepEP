#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试 internode::notify_dispatch 函数的脚本

该脚本：
1. 参考 test_low_latency_dispatch.py 的实现结构
2. 使用 test_internode_notify_dispatch.py 生成输入数据和 golden 数据
3. 通过 Buffer 类获取必要的内部指针和参数
4. 直接调用 internode::notify_dispatch（通过 C++ 绑定）
5. 对比实际结果与 golden 数据

注意：此脚本假设已经在 C++ 中添加了 notify_dispatch 的 Python 绑定
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.distributed as dist

# 导入 deep_ep 模块
# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, hash_tensor, build_base_parser

# import warnings, traceback

# def _warn_with_stack(message, category, filename, lineno, file=None, line=None):
#     print("=== GCU Long->Int warning traceback ===", flush=True)
#     traceback.print_stack()
#     print("=== warning ===", message, flush=True)

# warnings.showwarning = _warn_with_stack
# warnings.filterwarnings("default", message=r"GCU not support Long use Int replace.*")

# 导入测试数据生成模块
try:
    from test_internode_notify_dispatch import (
        NotifyDispatchConfig,
        build_random_cluster_inputs,
        simulate_rdma_mixed_cluster,
        simulate_nvl_cluster,
        compute_prefix_matrices_for_rank,
        ClusterInputs,
        RDMAMixedState,
        NVLState,
    )
except ImportError:
    # 如果导入失败，尝试从当前目录导入
    sys.path.insert(0, os.path.dirname(__file__))
    from test_internode_notify_dispatch import (
        NotifyDispatchConfig,
        build_random_cluster_inputs,
        simulate_rdma_mixed_cluster,
        simulate_nvl_cluster,
        compute_prefix_matrices_for_rank,
        ClusterInputs,
        RDMAMixedState,
        NVLState,
    )


def prepare_test_data(cfg: NotifyDispatchConfig, rank: int, num_channels: int) -> dict:
    """
    为指定 rank 准备测试数据
    
    Args:
        cfg: NotifyDispatchConfig 配置
        rank: 当前 rank
        num_channels: 通道数，必须与后面实际测试时使用的 num_channels 一致
    
    Returns:
        包含输入数据和 golden 数据的字典
    """
    # 生成集群输入
    cluster = build_random_cluster_inputs(cfg)
    
    # 模拟 RDMA 和 NVL 通信，生成 golden 数据
    rdma_states = simulate_rdma_mixed_cluster(cfg, cluster)
    nvl_states = simulate_nvl_cluster(cfg, rdma_states)
    
    # 计算前缀矩阵
    rdma_prefix_mat, gbl_prefix_mat = compute_prefix_matrices_for_rank(
        cfg, cluster.per_rank[rank], num_channels
    )
    
    # 获取当前 rank 的输入
    per_rank_input = cluster.per_rank[rank]
    rdma_state = rdma_states[rank]
    nvl_state = nvl_states[rank]
    
    return {
        'per_rank_input': per_rank_input,
        'rdma_state': rdma_state,
        'nvl_state': nvl_state,
        'rdma_prefix_mat': rdma_prefix_mat,
        'gbl_prefix_mat': gbl_prefix_mat,
        'num_channels': num_channels,
    }


def test_notify_dispatch_direct(args: argparse.Namespace, rank: int, num_ranks: int, 
                                 buffer: deep_ep.Buffer, cfg: NotifyDispatchConfig, num_channels: int):
    """
    直接测试 notify_dispatch 函数
    
    使用新添加的 internode_notify_dispatch 方法直接调用
    """
    if rank == 0:
        print(f'[test] Testing notify_dispatch directly (num_channels={num_channels})...', flush=True)
    
    # 准备测试数据
    test_data = prepare_test_data(cfg, rank, num_channels)
    per_rank_input = test_data['per_rank_input']
    
    # 创建输入张量
    num_tokens = cfg.num_tokens
    num_experts = cfg.num_experts
    num_ranks = cfg.num_ranks
    num_rdma_ranks = cfg.num_rdma_ranks
    
    # 输入数据
    num_tokens_per_rank = torch.from_numpy(per_rank_input.num_tokens_per_rank.astype(np.int32)).to('gcu')
    num_tokens_per_rdma_rank = torch.from_numpy(per_rank_input.num_tokens_per_rdma_rank.astype(np.int32)).to('gcu')
    num_tokens_per_expert = torch.from_numpy(per_rank_input.num_tokens_per_expert.astype(np.int32)).to('gcu')
    is_token_in_rank = torch.from_numpy(per_rank_input.is_token_in_rank).to('gcu')
    
    # 创建 Config
    config_num_sms = num_channels * 2
    num_max_nvl_chunked_send_tokens = 8
    num_max_nvl_chunked_recv_tokens = 512
    num_max_rdma_chunked_send_tokens = args.expert_alignment
    num_max_rdma_chunked_recv_tokens = 128
    # config = deep_ep.Config(config_num_sms, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens,
    #                        num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens)
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (24, 48, 96, 144, 160) else 512)
    config = deep_ep.Config(24, 4, nvl_buffer_size, 16, rdma_buffer_size)
    
    # 计算 hidden_int4 和 num_scales
    hidden = args.hidden
    hidden_int4 = (hidden * 2) // 4  # bfloat16 is 2 bytes, int4 is 4 bytes
    num_scales = 0  # Not using FP8
    num_topk = cfg.topk
    
    try:
        # 直接调用 internode_notify_dispatch
        previous_event = None
        rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, event = \
            buffer.internode_notify_dispatch(
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                is_token_in_rank=is_token_in_rank,
                num_channels=num_channels,
                hidden_int4=hidden_int4,
                num_scales=num_scales,
                num_topk=num_topk,
                expert_alignment=args.expert_alignment,
                config=config,
                previous_event=previous_event,
                async_=False
            )
        
        # When async_=False, synchronization is already done in C++ code, so event.event will be None
        # Only wait if event is actually set (when async_=True)
        if event is not None and event.event is not None:
            event.current_stream_wait()
        
        # 构造结果字典
        # 注意：moe_recv_counter, moe_recv_rdma_counter, moe_recv_expert_counter 在 internode_notify_dispatch 内部使用
        # 但不会返回，因为它们是通过 host 内存映射的，需要额外的同步才能读取
        # 这里我们只验证前缀矩阵和前缀和
        # Read host-mapped MoE counters from Buffer for full validation.
        #
        # IMPORTANT: emulate the same "busy-wait until ready" logic used in `deep_ep.cpp`:
        # the counters are written by device into host-mapped memory, so reading immediately
        # after kernel launch can observe -1 (not-ready) transiently.
        num_local_experts = cfg.num_experts // cfg.num_ranks
        start_time = time.monotonic()
        spin = 0
        while True:
            moe_recv_counter, moe_recv_rdma_counter, moe_recv_expert_counter = buffer.runtime.get_moe_recv_info(num_local_experts)
            ready = (moe_recv_counter >= 0) and (moe_recv_rdma_counter >= 0)
            if ready:
                for v in moe_recv_expert_counter:
                    if v < 0:
                        ready = False
                        break
            if ready:
                break
            if time.monotonic() - start_time > 100.0:
                raise RuntimeError(
                    f"Timeout waiting MoE counters: moe_recv_counter={moe_recv_counter}, "
                    f"moe_recv_rdma_counter={moe_recv_rdma_counter}, first10={moe_recv_expert_counter[:10]}"
                )
            spin += 1
            # Keep it close to C++ busy-wait; just yield occasionally to reduce CPU starvation.
            if (spin & 0xFFFF) == 0:
                time.sleep(0)

        buffers = {
            'moe_recv_counter': moe_recv_counter,
            'moe_recv_rdma_counter': moe_recv_rdma_counter,
            'moe_recv_expert_counter': moe_recv_expert_counter,
            'recv_rdma_rank_prefix_sum': recv_rdma_rank_prefix_sum,
            'recv_gbl_rank_prefix_sum': recv_gbl_rank_prefix_sum,
            'rdma_channel_prefix_matrix': rdma_channel_prefix_matrix,
            'gbl_channel_prefix_matrix': gbl_channel_prefix_matrix,
        }
        
        return buffers, test_data
        
    except Exception as e:
        if rank == 0:
            print(f'[test] Error in internode_notify_dispatch: {e}', flush=True)
            import traceback
            traceback.print_exc()
        raise


def compare_results(buffers: dict, test_data: dict, rank: int, rtol: float = 1e-5, atol: float = 1e-8):
    """
    对比实际结果与 golden 数据
    """
    errors = []
    rdma_state = test_data['rdma_state']
    nvl_state = test_data['nvl_state']
    rdma_prefix_mat = test_data['rdma_prefix_mat']
    gbl_prefix_mat = test_data['gbl_prefix_mat']
    
    # 0. 检查 host-mapped MoE counters（total / rdma_total / per-expert）
    if buffers.get('moe_recv_rdma_counter', None) is not None:
        expected_rdma_total = int(rdma_state.moe_recv_rdma_counter)
        actual_rdma_total = int(buffers['moe_recv_rdma_counter'])
        if actual_rdma_total != expected_rdma_total:
            errors.append(
                f"moe_recv_rdma_counter mismatch:\n"
                f"  expected={expected_rdma_total}\n"
                f"  actual={actual_rdma_total}"
            )

    if buffers.get('moe_recv_counter', None) is not None:
        expected_total = int(nvl_state.moe_recv_counter)
        actual_total = int(buffers['moe_recv_counter'])
        if actual_total != expected_total:
            errors.append(
                f"moe_recv_counter mismatch:\n"
                f"  expected={expected_total}\n"
                f"  actual={actual_total}"
            )

    if buffers.get('moe_recv_expert_counter', None) is not None:
        expected_expert = [int(x) for x in nvl_state.moe_recv_expert_counter.tolist()]
        actual_expert = [int(x) for x in buffers['moe_recv_expert_counter']]
        if actual_expert != expected_expert:
            # Keep the message short: show only first few diffs.
            diffs = []
            for i, (a, e) in enumerate(zip(actual_expert, expected_expert)):
                if a != e:
                    diffs.append(f"idx={i}: expected={e} actual={a}")
                    if len(diffs) >= 10:
                        break
            errors.append(
                f"moe_recv_expert_counter mismatch (showing up to 10 diffs):\n"
                f"  " + "\n  ".join(diffs)
            )
    
    # 1. 检查前缀和矩阵
    if buffers['recv_gbl_rank_prefix_sum'] is not None:
        expected_gbl_prefix = torch.from_numpy(nvl_state.recv_gbl_rank_prefix_sum.astype(np.int32))
        actual_gbl_prefix = buffers['recv_gbl_rank_prefix_sum'].cpu()
        if not torch.allclose(actual_gbl_prefix, expected_gbl_prefix, rtol=rtol, atol=atol):
            errors.append(
                f"recv_gbl_rank_prefix_sum mismatch:\n"
                f"  expected={expected_gbl_prefix.tolist()}\n"
                f"  actual={actual_gbl_prefix.tolist()}"
            )
    
    if buffers['recv_rdma_rank_prefix_sum'] is not None:
        expected_rdma_prefix = torch.from_numpy(rdma_state.recv_rdma_rank_prefix_sum.astype(np.int32))
        actual_rdma_prefix = buffers['recv_rdma_rank_prefix_sum'].cpu()
        if not torch.allclose(actual_rdma_prefix, expected_rdma_prefix, rtol=rtol, atol=atol):
            errors.append(
                f"recv_rdma_rank_prefix_sum mismatch:\n"
                f"  expected={expected_rdma_prefix.tolist()}\n"
                f"  actual={actual_rdma_prefix.tolist()}"
            )
    
    # 4. 检查前缀矩阵
    if buffers['rdma_channel_prefix_matrix'] is not None:
        expected_rdma_prefix_mat = torch.from_numpy(rdma_prefix_mat.astype(np.int32))
        actual_rdma_prefix_mat = buffers['rdma_channel_prefix_matrix'].cpu()
        if not torch.allclose(actual_rdma_prefix_mat, expected_rdma_prefix_mat, rtol=rtol, atol=atol):
            errors.append(
                f"rdma_channel_prefix_matrix mismatch:\n"
                f"  expected={expected_rdma_prefix_mat.tolist()}\n"
                f"  actual={actual_rdma_prefix_mat.tolist()}"
            )
    
    if buffers['gbl_channel_prefix_matrix'] is not None:
        expected_gbl_prefix_mat = torch.from_numpy(gbl_prefix_mat.astype(np.int32))
        actual_gbl_prefix_mat = buffers['gbl_channel_prefix_matrix'].cpu()
        if not torch.allclose(actual_gbl_prefix_mat, expected_gbl_prefix_mat, rtol=rtol, atol=atol):
            errors.append(
                f"gbl_channel_prefix_matrix mismatch:\n"
                f"  expected={expected_gbl_prefix_mat.tolist()}\n"
                f"  actual={actual_gbl_prefix_mat.tolist()}"
            )
    
    return len(errors) == 0, errors


def test_main(args: argparse.Namespace, rank: int, num_ranks: int, buffer: deep_ep.Buffer, num_channels: int, skip_validation: bool = False):
    """
    执行 notify_dispatch 直接测试
    
    Args:
        skip_validation: 如果为 True，跳过结果校验（用于压测）
    """
    # 配置参数
    cfg = NotifyDispatchConfig(
        num_ranks=num_ranks,
        num_rdma_ranks=args.num_rdma_ranks,
        num_nvl_ranks=args.num_nvl_ranks,
        num_experts=args.num_experts,
        topk=args.topk,
        num_tokens=args.num_tokens,
        num_worst_tokens=0,
        expert_alignment=args.expert_alignment,
        seed=args.seed,
    )
    
    # 验证配置
    cfg.validate()
    
    if rank == 0:
        print(f'[test] Starting direct notify_dispatch test with config:', flush=True)
        print(f'  num_ranks={num_ranks}, num_rdma_ranks={cfg.num_rdma_ranks}, '
              f'num_nvl_ranks={cfg.num_nvl_ranks}, num_experts={cfg.num_experts}, '
              f'topk={cfg.topk}, num_tokens={cfg.num_tokens}, expert_alignment={cfg.expert_alignment}, '
              f'num_channels={num_channels}',
              flush=True)
    
    try:
        buffers, test_data = test_notify_dispatch_direct(args, rank, num_ranks, buffer, cfg, num_channels)
        
        # 对比结果（压测时跳过）
        if skip_validation:
            return True
        
        if rank == 0:
            print('[test] Comparing results with golden data...', flush=True)
        
        is_match, errors = compare_results(buffers, test_data, rank)
        
        if is_match:
            if rank == 0:
                print('[test] ✓ All results match golden data!', flush=True)
            return True
        else:
            # Always emit mismatch details into the per-rank log file.
            # Rank0 is still the most important, but non-zero ranks can fail independently.
            print(f'[test][rank {rank}] ✗ Found {len(errors)} mismatches:', flush=True)
            for error in errors[:20]:
                print(f'  - {error}', flush=True)
            if len(errors) > 20:
                print(f'  - ... {len(errors) - 20} more ...', flush=True)
            return False
            
    except Exception as e:
        print(f'[test][rank {rank}] Error during testing: {e}', flush=True)
        import traceback
        traceback.print_exc()
        return False


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    """
    每个进程的测试循环函数（参考 test_low_latency_dispatch.py）
    """
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_file = f"{rank}_test_notify_dispatch_direct.log"
    
    # Redirect stdout and stderr to log file
    with open(log_file, 'w') as f:
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
            
            # 计算 num_rdma_ranks 和 num_nvl_ranks
            NUM_MAX_NVL_PEERS = 2
            if args.num_nvl_ranks is None:
                args.num_nvl_ranks = NUM_MAX_NVL_PEERS
            
            num_channels = args.num_channels
            
            if args.num_rdma_ranks is None:
                if num_ranks % NUM_MAX_NVL_PEERS != 0:
                    raise ValueError(
                        f"num_ranks={num_ranks} must be divisible by NUM_MAX_NVL_PEERS={NUM_MAX_NVL_PEERS}"
                    )
                args.num_rdma_ranks = num_ranks // NUM_MAX_NVL_PEERS
            
            # 验证 num_rdma_ranks 是否足够
            # if args.num_rdma_ranks < num_channels:
            #     raise ValueError(
            #         f"num_rdma_ranks={args.num_rdma_ranks} must be >= num_channels={num_channels}"
            #     )
            
            # 验证配置
            if num_ranks != args.num_rdma_ranks * args.num_nvl_ranks:
                raise ValueError(
                    f"num_ranks={num_ranks} must equal num_rdma_ranks * num_nvl_ranks = "
                    f"{args.num_rdma_ranks} * {args.num_nvl_ranks}"
                )
            
            # if args.num_nvl_ranks != NUM_MAX_NVL_PEERS: #FIXME: check not suit for nvl_ranks
            #     raise ValueError(
            #         f"num_nvl_ranks={args.num_nvl_ranks} must equal NUM_MAX_NVL_PEERS={NUM_MAX_NVL_PEERS}"
            #     )
            
            if local_rank == 0:
                print(f'[config] num_ranks={num_ranks}, num_rdma_ranks={args.num_rdma_ranks}, '
                      f'num_nvl_ranks={args.num_nvl_ranks}, num_experts={args.num_experts}, '
                      f'topk={args.topk}, num_tokens={args.num_tokens}, expert_alignment={args.expert_alignment}, '
                      f'num_channels={num_channels}',
                      flush=True)
            
            # 创建 Buffer（参考 test_low_latency.py）
            # In low_latency_mode, we still need num_nvl_bytes for notify_dispatch (it uses buffer_ptrs for NVLink)
            # But we can use a smaller value since notify_dispatch only needs metadata buffers
            num_nvl_bytes = 64 * 1024 * 1024  # 64 MB (smaller than normal mode, just for metadata)
            num_rdma_bytes = 128 * 1024 * 1024  # 128 MB
            
            if local_rank == 0:
                print(f'Allocating buffer: nvl={num_nvl_bytes / 1e6:.1f} MB, rdma={num_rdma_bytes / 1e6:.1f} MB ...', flush=True)
            
            # CRITICAL: low_latency_mode must be True because internode.tops uses LowLatencyPrimitives
            # Match test_low_latency.py configuration exactly
            # buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, 
            #                        low_latency_mode=True, num_qps_per_rank=args.num_experts // args.num_processes, #num_qps_per_rank
            #                        allow_nvlink_for_low_latency_mode=False, explicitly_destroy=True,
            #                        allow_mnnvl=True)
            buffer = deep_ep.Buffer(group,
                                    int(2e9),
                                    int(1e9),
                                    low_latency_mode=False,
                                    num_qps_per_rank=24,
                                    explicitly_destroy=True,
                                    is_internode=True)
            
            if local_rank == 0:
                print(f'[config] Using num_channels={num_channels} for notify_dispatch test', flush=True)
                print(f'[config] num_rdma_ranks={args.num_rdma_ranks} (must be >= num_channels={num_channels})', flush=True)
            
            # 运行测试
            success = test_main(args, rank, num_ranks, buffer, num_channels)

            # IMPORTANT: test_main() does per-rank validation. Always aggregate a global verdict;
            # otherwise you can see "rank0 passed" while some other rank failed, which breaks pressure-test flow.
            success_tensor = torch.tensor([1 if success else 0], dtype=torch.int32, device='gcu')
            dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN, group=group)
            global_success = bool(int(success_tensor.item()) == 1)

            if global_success:
                print(f'[rank {rank}] Test passed!', flush=True)
            else:
                print(f'[rank {rank}] Test failed!', flush=True)
                # For non-pressure mode, fail-fast; for pressure mode, still fail-fast here because
                # the "baseline" run failing means the setup is already broken.
                dist.barrier()
                sys.exit(1)
            
            # 压测逻辑（如果启用）
            if args.pressure_test:
                if local_rank == 0:
                    print(f'[pressure-test] Starting pressure test (100 random iterations with validation)...', flush=True)
                
                import random
                NUM_PRESSURE_ITERATIONS = 100
                failed_iterations = []
                for i in range(NUM_PRESSURE_ITERATIONS):
                    # 只在 rank 0 生成随机 seed，然后同步到所有 rank
                    if rank == 0:
                        random_seed = random.randint(0, int(1e9))
                    else:
                        random_seed = 0
                    
                    # 同步 seed 到所有 rank
                    random_seed_tensor = torch.tensor([random_seed], dtype=torch.int64, device='gcu')
                    dist.broadcast(random_seed_tensor, src=0, group=group)
                    random_seed = int(random_seed_tensor.item())
                    if local_rank == 0:
                        print(f'[pressure-test] Iteration {i+1}/{NUM_PRESSURE_ITERATIONS} with seed {random_seed}...', flush=True)
                    
                    # 临时修改 seed 进行随机测试（保留结果校验）
                    original_seed = args.seed
                    args.seed = random_seed
                    try:
                        # 运行测试并校验结果
                        iteration_success = test_main(args, rank, num_ranks, buffer, num_channels, skip_validation=False)
                        iter_tensor = torch.tensor([1 if iteration_success else 0], dtype=torch.int32, device='gcu')
                        dist.all_reduce(iter_tensor, op=dist.ReduceOp.MIN, group=group)
                        global_iter_success = bool(int(iter_tensor.item()) == 1)
                        if not global_iter_success:
                            failed_iterations.append((i+1, random_seed))
                            if local_rank == 0:
                                print(f'[pressure-test] ✗ Iteration {i+1} with seed {random_seed} failed!', flush=True)
                    finally:
                        args.seed = original_seed
                
                if local_rank == 0:
                    if len(failed_iterations) == 0:
                        print(f'[pressure-test] ✓ Pressure test completed successfully! All {NUM_PRESSURE_ITERATIONS} iterations passed.', flush=True)
                    else:
                        print(f'[pressure-test] ✗ Pressure test completed with {len(failed_iterations)} failures:', flush=True)
                        for iter_num, seed in failed_iterations:
                            print(f'  - Iteration {iter_num} with seed {seed} failed', flush=True)
                        sys.exit(1)
            
            # 清理
            buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()
            
        except Exception as e:
            print(f'[rank {rank}] Fatal error: {e}', flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            # Restore original file descriptors and Python stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)


if __name__ == '__main__':
    parser = build_base_parser('Direct unit test for internode::notify_dispatch',
                               defaults={'num_processes': 16, 'num_tokens': 1024, 'hidden': 512},
                               num_topk=False)
    parser.add_argument('--topk', type=int, default=8, help='Top-k value (default: 8)')
    parser.add_argument('--expert-alignment', type=int, default=16, help='Expert alignment (default: 16)')
    parser.add_argument('--num-rdma-ranks', type=int, default=2,
                        help='Number of RDMA ranks (auto: num_ranks // 2 if not set)')
    parser.add_argument('--num-nvl-ranks', type=int, default=8,
                        help='Number of NVL ranks (auto: 2 if not set)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--num-channels', type=int, default=12,
                        help='Number of channels for notify_dispatch (default: 12)')
    parser.add_argument('--pressure-test', action='store_true',
                        help='Enable pressure test: run 100 random iterations after normal test (default: False)')

    args = parser.parse_args()
    
    # 验证至少需要2个进程
    if args.num_processes < 2:
        print('Error: --num-processes must be at least 2 for communication testing', flush=True)
        sys.exit(1)
    
    # 使用 torch.multiprocessing.spawn 启动多进程测试
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)

