#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


# ==================# ...省略其他代码... 


# ==== 修改后代码 ====
# 一、配置与基础工具函数
# ==================# ...省略其他代码... 


# ==== 修改后代码 ====

@dataclass
class NotifyDispatchConfig:
    num_ranks: int           # 全局 rank 数，必须 = num_rdma_ranks * num_nvl_ranks
    num_rdma_ranks: int      # RDMA rank 数（跨节点）
    num_nvl_ranks: int       # NVL rank 数（节点内 NVLink peer 数）
    num_experts: int         # 全局 expert 数
    topk: int                # gating 的 top-k
    num_tokens: int          # 每个 rank 本地 token 数
    num_worst_tokens: int    # 与 C++ 接口一致，这里只做占位（num_worst_tokens=0 情况）
    expert_alignment: int    # notify_dispatch 的 expert_alignment
    seed: int = 0            # 随机种子（用于复现）

    def validate(self):
        # 1. rank 拆分关系
        if self.num_ranks != self.num_rdma_ranks * self.num_nvl_ranks:
            raise ValueError(
                f"num_ranks={self.num_ranks} 必须等于 "
                f"num_rdma_ranks * num_nvl_ranks = "
                f"{self.num_rdma_ranks} * {self.num_nvl_ranks}"
            )

        # 2. expert 在 ranks / rdma / nvl 维度的均分关系
        if self.num_experts % self.num_ranks != 0:
            raise ValueError(
                f"num_experts={self.num_experts} 必须能被 num_ranks={self.num_ranks} 整除"
            )

        if self.num_experts % self.num_rdma_ranks != 0:
            raise ValueError(
                f"num_experts={self.num_experts} 必须能被 num_rdma_ranks={self.num_rdma_ranks} 整除"
            )

        num_rdma_experts = self.num_experts // self.num_rdma_ranks
        if num_rdma_experts % self.num_nvl_ranks != 0:
            raise ValueError(
                f"num_experts / num_rdma_ranks = {num_rdma_experts} "
                f"必须能被 num_nvl_ranks={self.num_nvl_ranks} 整除"
            )

        # 3. topk 合法性
        if self.topk <= 0 or self.topk > self.num_experts:
            raise ValueError(
                f"topk={self.topk} 必须在 1..num_experts({self.num_experts}) 范围内"
            )

        # 4. token / alignment 合法性
        if self.num_tokens <= 0:
            raise ValueError("num_tokens 必须 > 0")
        if self.expert_alignment <= 0:
            raise ValueError("expert_alignment 必须 > 0")


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_channel_task_range(num_tokens: int,
                           num_channels: int,
                           channel_id: int) -> Tuple[int, int]:
    """
    对应 C++ 中 get_channel_task_range 的语义（简化版）：
      num_tokens_per_sm = ceil_div(num_tokens, num_sms)
      token_start_idx   = min(num_tokens_per_sm * sm_id, num_tokens)
      token_end_idx     = min(token_start_idx + num_tokens_per_sm, num_tokens)

    在 notify_dispatch 里 num_sms 换成 num_channels，sm_id 换成 channel_id。
    """
    num_tokens_per_ch = ceil_div(num_tokens, num_channels)
    start = min(num_tokens_per_ch * channel_id, num_tokens)
    end = min(start + num_tokens_per_ch, num_tokens)
    return start, end


# ==================# ...省略其他代码... 


# ==== 修改后代码 ====
# 二、输入数据结构
# ==================# ...省略其他代码... 


# ==== 修改后代码 ====

@dataclass
class PerRankInputs:
    # 单个 rank 的本地输入（“局部视角”，但 index 是全局 rank / expert）
    num_tokens_per_rank: np.ndarray          # [num_ranks]
    num_tokens_per_rdma_rank: np.ndarray     # [num_rdma_ranks]
    num_tokens_per_expert: np.ndarray        # [num_experts]
    is_token_in_rank: np.ndarray             # [num_tokens, num_ranks] bool


@dataclass
class ClusterInputs:
    # 整个集群的输入（所有 rank）
    per_rank: List[PerRankInputs]
    global_num_tokens_per_rank: np.ndarray       # [num_ranks]，所有 rank 局部统计之和
    global_num_tokens_per_rdma_rank: np.ndarray  # [num_rdma_ranks]
    global_num_tokens_per_expert: np.ndarray     # [num_experts]
    global_is_token_in_rank: np.ndarray          # [num_ranks * num_tokens, num_ranks] bool


@dataclass
class RDMAMixedState:
    # 对应“某个 global_rank 的视角”下：
    #   - 本 rank 在 RDMA 发送前写入 mixed send buffer 的内容
    #   - RDMA 通信结束后，本 rank 在 mixed recv buffer 中看到的内容
    #   - 以及基于 recv buffer 做的本地加和（recv_rdma_rank_prefix_sum / moe_recv_rdma_counter）
    send: np.ndarray                      # [num_rdma_ranks, num_nvl_ranks + num_rdma_experts + 1]
    recv: np.ndarray                      # [num_rdma_ranks, num_nvl_ranks + num_rdma_experts + 1]
    recv_rdma_rank_prefix_sum: np.ndarray # [num_rdma_ranks]
    moe_recv_rdma_counter: int            # 标量：本 rank 基于 recv buffer 做的 RDMA 维度总和


@dataclass
class NVLState:
    # 对应“某个 global_rank 的视角”下 NVL 阶段的状态：
    #   - 本 rank 在进入 NVL 发送前，根据 RDMA recv 得到的 nvl_reduced_num_tokens_per_expert
    #   - NVL 通信前：本 rank 往 NVL send buffer 写了什么
    #   - NVL 通信结束后：本 rank 在 NVL recv buffer 中看到了什么
    #   - 基于 NVL recv buffer 做的本地加和结果（recv_gbl_rank_prefix_sum / moe_recv_counter / moe_recv_expert_counter）
    nvl_reduced_num_tokens_per_expert: np.ndarray      # [num_rdma_experts]
    send_num_tokens_per_rank: np.ndarray               # [dst_nvl, rdma_rank]
    recv_num_tokens_per_rank: np.ndarray               # [src_nvl, rdma_rank]
    send_num_tokens_per_expert: np.ndarray             # [dst_nvl, local_expert_idx]
    recv_num_tokens_per_expert: np.ndarray             # [src_nvl, local_expert_idx]
    recv_gbl_rank_prefix_sum: np.ndarray               # [num_ranks]
    moe_recv_counter: int                              # 标量：所有 global_rank 的总 token 数（通常等于本 rank 作为目标 rank 的全局 num_tokens_per_rank）
    moe_recv_expert_counter: np.ndarray                # [num_local_experts]，本 rank 本地 experts 对齐后的计数


# ==================# ...省略其他代码... 


# ==== 修改后代码 ====
# 三、构造合法的随机输入（每个 rank 一份）
# ==================# ...省略其他代码... 


# ==== 修改后代码 ====

def build_random_inputs_for_rank(cfg: NotifyDispatchConfig,
                                 rng: np.random.Generator,
                                 rank: int) -> PerRankInputs:
    """
    为单个 global rank 构造一份本地合法输入：
      - is_token_in_rank[t, r]：本 rank 上第 t 个 token 是否路由到 global rank r 上的某个 expert
      - num_tokens_per_expert[e]：本 rank 上所有 token 中，路由到 expert e 的次数
      - num_tokens_per_rank[r]：本 rank 上所有 token 中，至少有 1 个 expert 在 global rank r 的 token 数
      - num_tokens_per_rdma_rank[rd]：本 rank 上 token 中，至少有 1 个 expert 落在 rd 这个 RDMA 段的 token 数
    """
    num_ranks = cfg.num_ranks
    num_experts = cfg.num_experts
    num_tokens = cfg.num_tokens
    num_rdma_ranks = cfg.num_rdma_ranks
    num_nvl_ranks = cfg.num_nvl_ranks
    topk = cfg.topk

    experts_per_rank = num_experts // num_ranks

    is_token_in_rank = np.zeros((num_tokens, num_ranks), dtype=bool)
    num_tokens_per_expert = np.zeros((num_experts,), dtype=np.int64)

    # 1. 为本 rank 上的每个 token 随机选 topk 个 expert
    for t in range(num_tokens):
        chosen_experts = rng.choice(num_experts, size=topk, replace=False)
        for e in chosen_experts:
            num_tokens_per_expert[e] += 1
            dest_rank = e // experts_per_rank
            is_token_in_rank[t, dest_rank] = True

    # 2. num_tokens_per_rank：按 “是否落在该 rank” 聚合 token 粒度
    num_tokens_per_rank = is_token_in_rank.sum(axis=0).astype(np.int64)

    # 3. num_tokens_per_rdma_rank：按 RDMA 段聚合
    num_tokens_per_rdma_rank = np.zeros((num_rdma_ranks,), dtype=np.int64)
    for rdma in range(num_rdma_ranks):
        begin = rdma * num_nvl_ranks
        end = begin + num_nvl_ranks
        token_in_rdma = is_token_in_rank[:, begin:end].any(axis=1)
        num_tokens_per_rdma_rank[rdma] = int(token_in_rdma.sum())

    # -------- 合法性断言（per-rank） --------
    assert is_token_in_rank.shape == (num_tokens, num_ranks)
    assert num_tokens_per_rank.shape == (num_ranks,)
    assert num_tokens_per_rdma_rank.shape == (num_rdma_ranks,)
    assert num_tokens_per_expert.shape == (num_experts,)

    # 每个 token 至少路由到 1 个 rank
    assert is_token_in_rank.any(axis=1).all(), \
        f"rank={rank}: 存在没有任何目标 rank 的 token"

    # expert 侧统计：sum_e num_tokens_per_expert[e] 要等于 num_tokens * topk
    total_pairs = int(num_tokens_per_expert.sum())
    expected_pairs = num_tokens * topk
    assert total_pairs == expected_pairs, \
        f"rank={rank}: num_tokens_per_expert.sum()={total_pairs}, 期望={expected_pairs}"

    # rank 侧统计：sum_r num_tokens_per_rank[r] 在 [num_tokens, num_tokens*topk] 区间内
    sum_rank = int(num_tokens_per_rank.sum())
    assert num_tokens <= sum_rank <= num_tokens * topk, \
        f"rank={rank}: sum(num_tokens_per_rank)={sum_rank}, 超出合法范围 [{num_tokens}, {num_tokens * topk}]"

    return PerRankInputs(
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        is_token_in_rank=is_token_in_rank,
    )


def build_random_cluster_inputs(cfg: NotifyDispatchConfig) -> ClusterInputs:
    """
    为整个集群（num_ranks 个 rank）构造一份合法的输入：
      - per_rank：每个 rank 一份本地输入
      - global_num_tokens_per_*：所有 rank 局部统计的求和（只用于合法性检查和期望值）
      - global_is_token_in_rank：把每个 rank 的 is_token_in_rank 叠起来，方便校验
    """
    cfg.validate()

    per_rank: List[PerRankInputs] = []
    for rank in range(cfg.num_ranks):
        rng = np.random.default_rng(cfg.seed + rank)
        per_rank.append(build_random_inputs_for_rank(cfg, rng, rank))

    # 1. 直接按元素求和，得到全局统计
    global_num_tokens_per_rank = sum((p.num_tokens_per_rank for p in per_rank))
    global_num_tokens_per_rdma_rank = sum((p.num_tokens_per_rdma_rank for p in per_rank))
    global_num_tokens_per_expert = sum((p.num_tokens_per_expert for p in per_rank))

    # 2. 组合出全局的 is_token_in_rank（把每个 rank 的本地矩阵按 token 维度堆叠）
    total_tokens = cfg.num_ranks * cfg.num_tokens
    global_is_token_in_rank = np.zeros((total_tokens, cfg.num_ranks), dtype=bool)
    offset = 0
    for p in per_rank:
        global_is_token_in_rank[offset:offset + cfg.num_tokens, :] = p.is_token_in_rank
        offset += cfg.num_tokens

    # -------- 合法性断言（cluster 级别） --------
    # 2.1 按 global_is_token_in_rank 重算 num_tokens_per_rank
    chk_rank = global_is_token_in_rank.sum(axis=0).astype(np.int64)
    if not np.array_equal(global_num_tokens_per_rank, chk_rank):
        raise RuntimeError(
            "global_num_tokens_per_rank 与 global_is_token_in_rank 不一致：\n"
            f"  直接累加={global_num_tokens_per_rank}\n"
            f"  由 global_is_token_in_rank 重算={chk_rank}"
        )

    # 2.2 按 global_is_token_in_rank 重算 num_tokens_per_rdma_rank
    chk_rdma = np.zeros((cfg.num_rdma_ranks,), dtype=np.int64)
    for rdma in range(cfg.num_rdma_ranks):
        begin = rdma * cfg.num_nvl_ranks
        end = begin + cfg.num_nvl_ranks
        token_in_rdma = global_is_token_in_rank[:, begin:end].any(axis=1)
        chk_rdma[rdma] = int(token_in_rdma.sum())
    if not np.array_equal(global_num_tokens_per_rdma_rank, chk_rdma):
        raise RuntimeError(
            "global_num_tokens_per_rdma_rank 与 global_is_token_in_rank 不一致：\n"
            f"  直接累加={global_num_tokens_per_rdma_rank}\n"
            f"  由 global_is_token_in_rank 重算={chk_rdma}"
        )

    # 2.3 expert 侧统计：全局 sum_e num_tokens_per_expert[e] 必须等于 total_tokens * topk
    total_pairs_expected = total_tokens * cfg.topk
    total_pairs_actual = int(global_num_tokens_per_expert.sum())
    if total_pairs_actual != total_pairs_expected:
        raise RuntimeError(
            f"global_num_tokens_per_expert.sum()={total_pairs_actual}, "
            f"期望={total_pairs_expected}"
        )

    # 2.4 rank 侧统计的边界：sum_r num_tokens_per_rank[r] ∈ [total_tokens, total_tokens * topk]
    sum_rank = int(global_num_tokens_per_rank.sum())
    if not (total_tokens <= sum_rank <= total_tokens * cfg.topk):
        raise RuntimeError(
            f"sum(global_num_tokens_per_rank)={sum_rank}, "
            f"超出合法范围 [{total_tokens}, {total_tokens * cfg.topk}]"
        )

    return ClusterInputs(
        per_rank=per_rank,
        global_num_tokens_per_rank=global_num_tokens_per_rank,
        global_num_tokens_per_rdma_rank=global_num_tokens_per_rdma_rank,
        global_num_tokens_per_expert=global_num_tokens_per_expert,
        global_is_token_in_rank=global_is_token_in_rank,
    )


# ==================# ...省略其他代码... 


# ==== 修改后代码 ====
# 四、RDMA mixed buffers 的完整模拟（按 rank 展开）
# ==================# ...省略其他代码... 


# ==== 修改后代码 ====

def simulate_rdma_mixed_cluster(cfg: NotifyDispatchConfig,
                                cluster: ClusterInputs) -> List[RDMAMixedState]:
    """
    模拟 C++ internode::notify_dispatch 中 RDMA mixed 部分的核心逻辑：

    对于每一个 global_rank g（即一块 GPU）：
      1. 根据本 rank 的 num_tokens_per_* 构造 rdma_recv_num_tokens_mixed.send_buffer
      2. 按 low_latency_mode=true 的拓扑，把 send_buffer 在 RDMA 维度上做一次 all-to-all：
         - 固定 nvl_rank，跨 rdma_rank 互相收发
         - 得到每个 rank 自己看到的 rdma_recv_num_tokens_mixed.recv_buffer
      3. 基于本 rank 的 recv_buffer，做本地加和，得到：
         - recv_rdma_rank_prefix_sum
         - moe_recv_rdma_counter（注意：这里是“该 rdma_rank 内本 GPU 自己负责的那部分 token 数”）
    """
    num_ranks = cfg.num_ranks
    num_rdma_ranks = cfg.num_rdma_ranks
    num_nvl = cfg.num_nvl_ranks
    num_experts = cfg.num_experts
    num_rdma_experts = num_experts // num_rdma_ranks

    # mixed buffer 每行的长度：前 num_nvl 列是 per-NVL-rank 的 num_tokens_per_rank，
    # 后面 num_rdma_experts 列是 num_tokens_per_expert，最后 1 列是 num_tokens_per_rdma_rank
    row_len = num_nvl + num_rdma_experts + 1

    # 为每个 global_rank 分配 send / recv
    per_rank_send = [np.zeros((num_rdma_ranks, row_len), dtype=np.int64) for _ in range(num_ranks)]
    per_rank_recv = [np.zeros((num_rdma_ranks, row_len), dtype=np.int64) for _ in range(num_ranks)]

    # 1. 填充每个 rank 自己的 send_buffer（按“目标 rdma_rank / nvl_rank / expert”分桶）
    for g in range(num_ranks):
        inp = cluster.per_rank[g]
        send = per_rank_send[g]

        # 1.1 num_tokens_per_rank：dest_g 是“目标 global_rank”
        for dest_g in range(num_ranks):
            dst_rdma = dest_g // num_nvl      # 目标 RDMA 段
            dst_nvl = dest_g % num_nvl        # 目标 NVL 段
            send[dst_rdma, dst_nvl] = int(inp.num_tokens_per_rank[dest_g])

        # 1.2 num_tokens_per_expert：按 expert -> rdma 段拆分
        for e in range(num_experts):
            dst_rdma = e // num_rdma_experts
            local_e = e % num_rdma_experts
            send[dst_rdma, num_nvl + local_e] = int(inp.num_tokens_per_expert[e])

        # 1.3 num_tokens_per_rdma_rank：最后一列
        for rd in range(num_rdma_ranks):
            send[rd, num_nvl + num_rdma_experts] = int(inp.num_tokens_per_rdma_rank[rd])

    # 2. 模拟 RDMA all-to-all（low_latency_mode=true）：
    #    固定 nvl_rank，跨 rdma_rank 做互相发送。
    #
    # C++ 中的效果（简化概念）：
    #   - 源 GPU：global_rank = src_rdma * num_nvl + nvl
    #   - 目标 GPU：global_rank = dst_rdma * num_nvl + nvl
    #   - 源 GPU 往目标 GPU 的 recv_buffer[src_rdma] 写入 send_buffer[dst_rdma]
    for src_rdma in range(num_rdma_ranks):
        for nvl in range(num_nvl):
            src_g = src_rdma * num_nvl + nvl
            for dst_rdma in range(num_rdma_ranks):
                dst_g = dst_rdma * num_nvl + nvl
                # recv[dst_g][src_rdma] <- send[src_g][dst_rdma]
                per_rank_recv[dst_g][src_rdma, :] = per_rank_send[src_g][dst_rdma, :]

    # 3. 基于 recv_buffer 做本地 RDMA 维度加和（每个 GPU 自己负责的那一份）
    rdma_states: List[RDMAMixedState] = []
    for g in range(num_ranks):
        recv = per_rank_recv[g]
        # 最后一列保存的是 num_tokens_per_rdma_rank[dst_rdma] 的本地贡献
        per_rdma_counts = recv[:, num_nvl + num_rdma_experts]  # [num_rdma_ranks]
        recv_prefix = np.cumsum(per_rdma_counts, axis=0)
        moe_counter = int(recv_prefix[-1])
        rdma_states.append(
            RDMAMixedState(
                send=per_rank_send[g],
                recv=recv,
                recv_rdma_rank_prefix_sum=recv_prefix,
                moe_recv_rdma_counter=moe_counter,
            )
        )

    return rdma_states


# ==================# ...省略其他代码... 


# ==== 修改后代码 ====
# 五、NVL 通信的模拟（按 rank 展开）
# ==================# ...省略其他代码... 


# ==== 修改后代码 ====

def simulate_nvl_cluster(cfg: NotifyDispatchConfig,
                         rdma_states: List[RDMAMixedState]) -> List[NVLState]:
    """
    模拟 C++ internode::notify_dispatch 中 NVL 阶段的核心逻辑：

    对于每一个 global_rank g（即一块 GPU）：
      1. 使用本 rank 的 rdma_recv_num_tokens_mixed.recv_buffer 计算：
         - nvl_reduced_num_tokens_per_expert（RDMA 维度规约）
      2. 基于本地 nvl_reduced_num_tokens_per_expert 和 rdma_recv，构造：
         - NVL send buffer 中 num_tokens_per_rank / num_tokens_per_expert 部分
      3. 按 NVL 拓扑（同一个 rdma_rank 内的不同 nvl_rank 之间）模拟 NVL 发送，将 send 写入对端 recv
      4. 在每个 rank 上基于 NVL recv buffer 做本地加和，得到：
         - recv_gbl_rank_prefix_sum（按 global_rank 的前缀和）
         - moe_recv_counter
         - moe_recv_expert_counter（本地 experts 的对齐后计数）
    """
    num_ranks = cfg.num_ranks
    num_rdma_ranks = cfg.num_rdma_ranks
    num_nvl = cfg.num_nvl_ranks
    num_experts = cfg.num_experts

    num_rdma_experts = num_experts // num_rdma_ranks
    assert num_rdma_experts % num_nvl == 0
    num_nvl_experts = num_rdma_experts // num_nvl  # 每个 GPU 本地 expert 数 = num_experts / num_ranks

    # ---- 第一步：对每个 rank 先计算 nvl_reduced_num_tokens_per_expert 和 send buffer ----
    reduced_list: List[np.ndarray] = []
    send_rank_list: List[np.ndarray] = []    # [g][dst_nvl, rdma_rank]
    send_expert_list: List[np.ndarray] = []  # [g][dst_nvl, local_expert_idx]

    for g in range(num_ranks):
        rdma_recv = rdma_states[g].recv
        # 形状检查
        assert rdma_recv.shape[0] == num_rdma_ranks
        assert rdma_recv.shape[1] == num_nvl + num_rdma_experts + 1

        # 1.1 nvl_reduced_num_tokens_per_expert：沿着 RDMA 维度求和
        nvl_reduced = np.zeros((num_rdma_experts,), dtype=np.int64)
        for rdma_e in range(num_rdma_experts):
            col = num_nvl + rdma_e
            nvl_reduced[rdma_e] = int(rdma_recv[:, col].sum())

        # 1.2 NVL send buffer 中的 num_tokens_per_rank 部分
        send_rank = np.zeros((num_nvl, num_rdma_ranks), dtype=np.int64)
        for dst_nvl in range(num_nvl):
            for rdma_idx in range(num_rdma_ranks):
                send_rank[dst_nvl, rdma_idx] = int(rdma_recv[rdma_idx, dst_nvl])

        # 1.3 NVL send buffer 中的 num_tokens_per_expert 部分
        #      对于 dst_nvl，每个 local_expert_idx 对应 nvl_reduced[dst_nvl * num_nvl_experts + local_e]
        send_expert = np.zeros((num_nvl, num_nvl_experts), dtype=np.int64)
        for dst_nvl in range(num_nvl):
            for le in range(num_nvl_experts):
                rdma_e = dst_nvl * num_nvl_experts + le
                send_expert[dst_nvl, le] = int(nvl_reduced[rdma_e])

        reduced_list.append(nvl_reduced)
        send_rank_list.append(send_rank)
        send_expert_list.append(send_expert)

    # ---- 第二步：在 NVL 拓扑下做“发送”，写出每个 rank 的 recv buffer ----
    nvl_states: List[NVLState] = []

    for g_dest in range(num_ranks):
        rdma_d = g_dest // num_nvl
        nvl_d = g_dest % num_nvl

        # recv_num_tokens_per_rank[src_nvl, rdma_rank]
        recv_rank = np.zeros((num_nvl, num_rdma_ranks), dtype=np.int64)
        # recv_num_tokens_per_expert[src_nvl, local_expert_idx]
        recv_expert = np.zeros((num_nvl, num_nvl_experts), dtype=np.int64)

        # 对于同一 rdma_d 内所有 src_nvl，取其 send[dst_nvl=nvl_d] 作为本 rank 的 recv 源
        for src_nvl in range(num_nvl):
            g_src = rdma_d * num_nvl + src_nvl
            send_rank_src = send_rank_list[g_src]
            send_expert_src = send_expert_list[g_src]

            # 本 rank 是 dst_nvl = nvl_d
            recv_rank[src_nvl, :] = send_rank_src[nvl_d, :]
            recv_expert[src_nvl, :] = send_expert_src[nvl_d, :]

        # ---- 第三步：基于 NVL recv buffer 做本地加和 ----
        # 3.1 recv_gbl_rank_prefix_sum
        recv_gbl_rank_prefix_sum = np.zeros((num_ranks,), dtype=np.int64)
        s = 0
        for i in range(num_ranks):
            src_rdma = i // num_nvl
            src_nvl = i % num_nvl
            s += int(recv_rank[src_nvl, src_rdma])
            recv_gbl_rank_prefix_sum[i] = s
        moe_recv_counter = int(s)

        # 3.2 moe_recv_expert_counter（本 rank 本地 experts 的对齐后计数）
        moe_recv_expert_counter = np.zeros((num_nvl_experts,), dtype=np.int64)
        for le in range(num_nvl_experts):
            val = int(recv_expert[:, le].sum())
            moe_recv_expert_counter[le] = ceil_div(val, cfg.expert_alignment) * cfg.expert_alignment

        nvl_states.append(
            NVLState(
                nvl_reduced_num_tokens_per_expert=reduced_list[g_dest],
                send_num_tokens_per_rank=send_rank_list[g_dest],
                recv_num_tokens_per_rank=recv_rank,
                send_num_tokens_per_expert=send_expert_list[g_dest],
                recv_num_tokens_per_expert=recv_expert,
                recv_gbl_rank_prefix_sum=recv_gbl_rank_prefix_sum,
                moe_recv_counter=moe_recv_counter,
                moe_recv_expert_counter=moe_recv_expert_counter,
            )
        )

    return nvl_states


# ==================# ...省略其他代码... 


# ==== 修改后代码 ====
# 六、per-rank 的 channel 前缀矩阵
# ==================# ...省略其他代码... 


# ==== 修改后代码 ====

def compute_prefix_matrices_for_rank(cfg: NotifyDispatchConfig,
                                     inp: PerRankInputs,
                                     num_channels: int):
    """
    只针对单个 rank 的 is_token_in_rank，计算：
      - rdma_channel_prefix_matrix[dst_rdma_rank, channel]
      - gbl_channel_prefix_matrix[global_rank, channel]

    完全按照 C++ notify_dispatch 中 sm_id>0 分支的逻辑，只是去掉了 warp / lane 的细节。
    """
    num_ranks = cfg.num_ranks
    num_rdma_ranks = cfg.num_rdma_ranks
    num_nvl = cfg.num_nvl_ranks
    num_tokens = cfg.num_tokens

    rdma_channel_prefix_matrix = np.zeros((num_rdma_ranks, num_channels), dtype=np.int64)
    gbl_channel_prefix_matrix = np.zeros((num_ranks, num_channels), dtype=np.int64)

    # 遍历 dst_rdma_rank 和 channel
    for dst_rdma in range(num_rdma_ranks):
        for ch in range(num_channels):
            start, end = get_channel_task_range(num_tokens, num_channels, ch)
            total_count = 0
            per_nvl_count = np.zeros((num_nvl,), dtype=np.int64)

            for t in range(start, end):
                token_row = inp.is_token_in_rank[t]  # [num_ranks]
                belongs_any = False
                for nvl in range(num_nvl):
                    gbl = dst_rdma * num_nvl + nvl
                    if token_row[gbl]:
                        per_nvl_count[nvl] += 1
                        belongs_any = True
                if belongs_any:
                    total_count += 1

            # 写入“非 prefix”的计数
            for nvl in range(num_nvl):
                gbl = dst_rdma * num_nvl + nvl
                gbl_channel_prefix_matrix[gbl, ch] = per_nvl_count[nvl]
            rdma_channel_prefix_matrix[dst_rdma, ch] = total_count

    # 沿着 channel 维度做前缀和
    rdma_channel_prefix_matrix = np.cumsum(rdma_channel_prefix_matrix, axis=1)
    gbl_channel_prefix_matrix = np.cumsum(gbl_channel_prefix_matrix, axis=1)

    return rdma_channel_prefix_matrix, gbl_channel_prefix_matrix


# ==================# ...省略其他代码... 


# ==== 修改后代码 ====
# 七、打印工具（方便人眼检查）
# ==================# ...省略其他代码... 


# ==== 修改后代码 ====

def print_array_1d(name: str, arr: np.ndarray):
    print(f"{name} (shape={arr.shape}, dtype={arr.dtype}):")
    for i, v in enumerate(arr):
        print(f"  [{i}] = {int(v)}")
    print("")


def print_array_2d(name: str, arr: np.ndarray):
    print(f"{name} (shape={arr.shape}, dtype={arr.dtype}):")
    for i in range(arr.shape[0]):
        print(f"  row {i}: {arr[i].tolist()}")
    print("")


def print_is_token_in_rank(name: str, is_token_in_rank: np.ndarray):
    num_tokens, num_ranks = is_token_in_rank.shape
    print(f"{name} (shape={is_token_in_rank.shape}, dtype=bool):")
    for t in range(num_tokens):
        ranks = np.nonzero(is_token_in_rank[t])[0].tolist()
        print(f"  token {t}: ranks = {ranks}")
    print("")


def print_rank_inputs(cfg: NotifyDispatchConfig, cluster: ClusterInputs):
    print("==== 每个 rank 的本地输入（per-rank 局部视角） ====")
    for r, pr in enumerate(cluster.per_rank):
        rdma_rank = r // cfg.num_nvl_ranks
        nvl_rank = r % cfg.num_nvl_ranks
        print(f"---- global_rank {r} (rdma_rank={rdma_rank}, nvl_rank={nvl_rank}) ----")
        print_array_1d(f"rank{r}.num_tokens_per_rank", pr.num_tokens_per_rank)
        print_array_1d(f"rank{r}.num_tokens_per_rdma_rank", pr.num_tokens_per_rdma_rank)
        print_array_1d(f"rank{r}.num_tokens_per_expert", pr.num_tokens_per_expert)
        print_is_token_in_rank(f"rank{r}.is_token_in_rank", pr.is_token_in_rank)


def print_rdma_mixed_stage(cfg: NotifyDispatchConfig,
                           rdma_states: List[RDMAMixedState]):
    """
    RDMA mixed 部分拆成三步打印：

      1. “发送前”：每个 global_rank 在自己的 send buffer 里写了什么
      2. “RDMA 通信结束后”：每个 global_rank 在自己的 recv buffer 里看到了什么
      3. 本 rank 的 RDMA 维度本地加和（prefix 之前 vs 之后）
    """
    num_rdma_ranks = cfg.num_rdma_ranks
    num_nvl = cfg.num_nvl_ranks
    num_experts = cfg.num_experts
    num_rdma_experts = num_experts // num_rdma_ranks

    print("==== RDMA mixed buffers：每个 rank 的 send/recv 以及本地 RDMA 加和 ====")
    for g, state in enumerate(rdma_states):
        rdma_rank = g // num_nvl
        nvl_rank = g % num_nvl
        send = state.send
        recv = state.recv

        print(f"---- global_rank {g} (rdma_rank={rdma_rank}, nvl_rank={nvl_rank}) ----")

        # 1. 发送前：send buffer 的内容
        print("  [1] RDMA 发送前：本 rank 往 mixed send_buffer 写入的内容")
        for dst_rdma in range(num_rdma_ranks):
            row = send[dst_rdma]
            print(f"    - 发送到 rdma_rank={dst_rdma}:")
            # num_tokens_per_rank 部分
            print("      num_tokens_per_rank 部分 (列 0..num_nvl_ranks-1):")
            for nvl in range(num_nvl):
                dest_g = dst_rdma * num_nvl + nvl
                val = int(row[nvl])
                print(f"        dest_global_rank={dest_g} (rdma={dst_rdma}, nvl={nvl}): {val}")
            # num_tokens_per_expert 部分
            print("      num_tokens_per_expert 部分 (按 rdma_expert_idx 展开):")
            for le in range(num_rdma_experts):
                global_e = dst_rdma * num_rdma_experts + le
                val = int(row[num_nvl + le])
                print(f"        expert_global_id={global_e} (rdma_group={dst_rdma}, local_e={le}): {val}")
            # num_tokens_per_rdma_rank 部分
            val_last = int(row[num_nvl + num_rdma_experts])
            print("      num_tokens_per_rdma_rank 部分（最后一列）:")
            print(f"        num_tokens_per_rdma_rank[{dst_rdma}] 本地值: {val_last}")
        print("")

        # 2. RDMA 通信结束后：recv buffer 的内容
        print("  [2] RDMA 通信结束后：本 rank 在 mixed recv_buffer 中看到的内容")
        for src_rdma in range(num_rdma_ranks):
            row = recv[src_rdma]
            print(f"    - 来自 rdma_rank={src_rdma} 的一行（recv_buffer[{src_rdma}]）：")
            # num_tokens_per_rank 部分
            print("      num_tokens_per_rank 部分 (列 0..num_nvl_ranks-1):")
            for nvl in range(num_nvl):
                dest_g = rdma_rank * num_nvl + nvl  # 这里 dest 是“本 rank 所在的 rdma 段”
                val = int(row[nvl])
                print(f"        对 dest_global_rank={dest_g} (本 rdma={rdma_rank}, nvl={nvl}) 的贡献: {val}")
            # num_tokens_per_expert 部分
            print("      num_tokens_per_expert 部分 (按 rdma_expert_idx 展开):")
            for le in range(num_rdma_experts):
                global_e = rdma_rank * num_rdma_experts + le
                val = int(row[num_nvl + le])
                print(f"        expert_global_id={global_e} 来自 rdma_rank={src_rdma} 的贡献: {val}")
            # num_tokens_per_rdma_rank 部分
            val_last = int(row[num_nvl + num_rdma_experts])
            print("      num_tokens_per_rdma_rank 部分（最后一列）:")
            print(f"        对 rdma_rank={rdma_rank} 的贡献（来自 rdma_rank={src_rdma}）: {val_last}")
        print("")

        # 3. 本 rank 的 RDMA 维度本地加和（prefix 之前 vs 之后）
        print("  [3] 本 rank 基于 recv_buffer 做的 RDMA 维度本地加和")
        per_rdma_before_prefix = recv[:, num_nvl + num_rdma_experts]
        print_array_1d("    (a) prefix 之前的 per-rdma 计数（来自 recv_buffer 最后一列）",
                       per_rdma_before_prefix)
        print_array_1d("    (b) prefix 之后的 recv_rdma_rank_prefix_sum",
                       state.recv_rdma_rank_prefix_sum)
        print(f"    (c) moe_recv_rdma_counter（本 rank 的最终 RDMA 计数总和）: {state.moe_recv_rdma_counter}")
        print("")


def print_nvl_stage(cfg: NotifyDispatchConfig,
                    nvl_states: List[NVLState]):
    """
    NVL 阶段也拆成三步打印：

      1. “发送前”：每个 global_rank 在 NVL send buffer 里写了什么
      2. “NVL 通信结束后”：每个 global_rank 在 NVL recv buffer 里看到了什么
      3. 本 rank 的 NVL 维度本地加和（global_rank / expert 维度的前后对比）
    """
    num_nvl = cfg.num_nvl_ranks
    num_rdma_ranks = cfg.num_rdma_ranks
    num_experts = cfg.num_experts
    num_rdma_experts = num_experts // num_rdma_ranks
    num_nvl_experts = num_rdma_experts // num_nvl
    num_ranks = cfg.num_ranks

    print("==== NVL 通信：每个 rank 的 NVL send/recv 以及本地 NVL 加和 ====")
    for g, st in enumerate(nvl_states):
        rdma_rank = g // num_nvl
        nvl_rank = g % num_nvl

        print(f"---- global_rank {g} (rdma_rank={rdma_rank}, nvl_rank={nvl_rank}) ----")

        # 0. 本 rank 的 nvl_reduced_num_tokens_per_expert
        print("  [0] 进入 NVL 发送阶段前，本 rank 基于 RDMA recv 计算得到的 nvl_reduced_num_tokens_per_expert：")
        for idx, v in enumerate(st.nvl_reduced_num_tokens_per_expert):
            print(f"      rdma_expert_idx={idx}: {int(v)}")
        print("")

        # 1. 发送前：NVL send buffer 的内容
        print("  [1] NVL 发送前：本 rank 往 NVL send_buffer 写入的内容")
        for dst_nvl in range(num_nvl):
            print(f"    - 发送到 nvl_rank={dst_nvl}:")
            # num_tokens_per_rank
            print("      num_tokens_per_rank 相关（对每个 rdma_rank 的发送值）:")
            vals_rank = st.send_num_tokens_per_rank[dst_nvl]
            for rd in range(num_rdma_ranks):
                val = int(vals_rank[rd])
                print(
                    f"        send_num_tokens_per_rank[dst_nvl={dst_nvl}, rdma_rank={rd}] = {val}"
                )
            # num_tokens_per_expert
            print("      num_tokens_per_expert 相关（对每个 local_expert_idx 的发送值）:")
            vals_exp = st.send_num_tokens_per_expert[dst_nvl]
            for le in range(num_nvl_experts):
                val = int(vals_exp[le])
                global_e = rdma_rank * num_rdma_experts + dst_nvl * num_nvl_experts + le
                print(
                    f"        send_num_tokens_per_expert[dst_nvl={dst_nvl}, local_expert_idx={le}] "
                    f"(映射到 global_expert_id={global_e}) = {val}"
                )
        print("")

        # 2. NVL 通信结束后：recv buffer 的内容
        print("  [2] NVL 通信结束后：本 rank 在 NVL recv_buffer 中看到的内容")
        for src_nvl in range(num_nvl):
            print(
                f"    - 来自 nvl_rank={src_nvl} 的一行："
                f"recv_num_tokens_per_rank[src_nvl={src_nvl}, :], "
                f"recv_num_tokens_per_expert[src_nvl={src_nvl}, :]"
            )
            vals_rank = st.recv_num_tokens_per_rank[src_nvl]
            print("      num_tokens_per_rank 相关：")
            for rd in range(num_rdma_ranks):
                val = int(vals_rank[rd])
                print(
                    f"        recv_num_tokens_per_rank[src_nvl={src_nvl}, rdma_rank={rd}] = {val}"
                )
            vals_exp = st.recv_num_tokens_per_expert[src_nvl]
            print("      num_tokens_per_expert 相关：")
            for le in range(num_nvl_experts):
                val = int(vals_exp[le])
                global_e = rdma_rank * num_rdma_experts + src_nvl * num_nvl_experts + le
                print(
                    f"        recv_num_tokens_per_expert[src_nvl={src_nvl}, local_expert_idx={le}] "
                    f"(对应 global_expert_id={global_e}) = {val}"
                )
        print("")

        # 3. 本 rank 的 NVL 维度本地加和
        print("  [3] 本 rank 基于 NVL recv_buffer 做的本地加和")
        print_array_1d("    (a) recv_gbl_rank_prefix_sum（按 global_rank 做 prefix 的结果）",
                       st.recv_gbl_rank_prefix_sum)
        print(f"    (b) moe_recv_counter（所有 global_rank 的总 token 数）: {st.moe_recv_counter}")
        print_array_1d("    (c) moe_recv_expert_counter（本 rank 本地 experts 的对齐后计数）",
                       st.moe_recv_expert_counter)
        print("")


def print_prefix_matrices_for_rank(cfg: NotifyDispatchConfig,
                                   rdma_prefix_mat: np.ndarray,
                                   gbl_prefix_mat: np.ndarray,
                                   local_rank: int):
    print(f"==== notify_dispatch 中 per-channel 前缀矩阵（local_rank={local_rank} 的视角） ====")
    print_array_2d("  rdma_channel_prefix_matrix [rdma_rank, channel]", rdma_prefix_mat)
    print_array_2d("  gbl_channel_prefix_matrix [global_rank, channel]", gbl_prefix_mat)


# ==================# ...省略其他代码... 


# ==== 修改后代码 ====
# 八、示例：直接运行本脚本
# ==================# ...省略其他代码... 


# ==== 修改后代码 ====

def main():
    # 按你要求的配置：
    cfg = NotifyDispatchConfig(
        num_ranks=4,
        num_rdma_ranks=2,
        num_nvl_ranks=2,
        num_experts=8,
        topk=2,
        num_tokens=8,
        num_worst_tokens=0,
        expert_alignment=16,
        seed=42,
    )
    num_channels = 1    # 为了输出易读，这里先用 2
    local_rank = 0      # 从 rank0（rdma_rank=0, nvl_rank=0）的视角看 notify_dispatch 的部分结果

    cfg.validate()
    cluster = build_random_cluster_inputs(cfg)

    # RDMA mixed 部分：每个 rank 的 send/recv + 本地 RDMA 加和
    rdma_states = simulate_rdma_mixed_cluster(cfg, cluster)

    # NVL 通信部分：每个 rank 的 NVL send/recv + 本地 NVL 加和
    nvl_states = simulate_nvl_cluster(cfg, rdma_states)

    # per-rank 的 channel 前缀矩阵（只看 local_rank 的一份）
    rdma_prefix_mat, gbl_prefix_mat = compute_prefix_matrices_for_rank(
        cfg, cluster.per_rank[local_rank], num_channels
    )

    # ---- RDMA 级别一致性检查（修正版）----
    # 对于每个 rdma_rank，把同一 rdma_rank 下所有 nvl_rank 上的 moe_recv_rdma_counter 求和，
    # 再与聚合得到的 global_num_tokens_per_rdma_rank 对比。
    for rdma_rank in range(cfg.num_rdma_ranks):
        moe_sum = 0
        for nvl in range(cfg.num_nvl_ranks):
            g = rdma_rank * cfg.num_nvl_ranks + nvl
            moe_sum += rdma_states[g].moe_recv_rdma_counter
        expected_rdma = int(cluster.global_num_tokens_per_rdma_rank[rdma_rank])
        if moe_sum != expected_rdma:
            raise RuntimeError(
                f"RDMA 模拟结果不一致：rdma_rank={rdma_rank} 上所有 NVL peer 的 "
                f"moe_recv_rdma_counter 之和为 {moe_sum}, 但聚合得到的 "
                f"global_num_tokens_per_rdma_rank[{rdma_rank}]={expected_rdma}"
            )

    # ---- NVL 级别一致性检查：对于任意 global_rank g，其 moe_recv_counter
    #      应该等于 聚合得到的 global_num_tokens_per_rank[g] ----
    for g, st in enumerate(nvl_states):
        expected = int(cluster.global_num_tokens_per_rank[g])
        if st.moe_recv_counter != expected:
            raise RuntimeError(
                f"NVL 模拟结果不一致：global_rank={g} 的 moe_recv_counter={st.moe_recv_counter}, "
                f"但聚合得到的 global_num_tokens_per_rank[{g}]={expected}"
            )

    # === 开始打印 ===
    print("==== NotifyDispatchConfig ====")
    print(cfg)
    print("")

    # 1. 每个 rank 的本地输入
    print_rank_inputs(cfg, cluster)

    # 2. RDMA mixed：send/recv + 本地 RDMA 维度加和
    print_rdma_mixed_stage(cfg, rdma_states)

    # 3. NVL 通信：send/recv + 本地 NVL 维度加和
    print_nvl_stage(cfg, nvl_states)

    # 4. per-channel 前缀矩阵（local_rank 的视角）
    print_prefix_matrices_for_rank(cfg, rdma_prefix_mat, gbl_prefix_mat, local_rank)


if __name__ == "__main__":
    main()