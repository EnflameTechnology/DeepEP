import torch
from enum import Enum

class BoundaryTestType(Enum):
    """Enum for boundary test types"""
    # BF16 数据边界
    BF16_MAX = "bf16_max"
    BF16_MIN = "bf16_min"
    ZERO = "zero"
    TINY_POSITIVE = "tiny_positive"
    MIXED_EXTREME = "mixed_extreme"
    ROW_MIXED = "row_mixed"
    ORIGINAL = "original"

    # FP8 E4M3 量化边界
    # kernel 量化公式 (per-group, group_size=128):
    #   amax = max(kFP8Margin=1e-4, max(|group_128|))
    #   精确:      scale = 448/amax, scale_inv = amax/448
    #   round_scale: scale_inv = 2^ceil(log2(amax/448)), scale = 1/scale_inv
    #   fp8_val = cvt_to_fp8(value * scale)   // clamp 到 [-448, 448]
    #   dequant = fp8_val * scale_inv
    FP8_E4M3_MAX = "fp8_e4m3_max"
    FP8_E4M3_OVERFLOW = "fp8_e4m3_overflow"
    FP8_E4M3_UNDERFLOW = "fp8_e4m3_underflow"
    FP8_MIXED_RANGE = "fp8_mixed_range"
    FP8_SCALE_BOUNDARY = "fp8_scale_boundary"

    # UE8M0 Scale 边界（use_ue8m0=True 时，scale 编码为 uint8 指数字节）
    # kernel 使用 fast_log2_ceil 向上取整（ceil），不是四舍五入
    UE8M0_SCALE_EXACT_POW2 = "ue8m0_scale_exact_pow2"
    UE8M0_SCALE_ROUND_UP = "ue8m0_scale_round_up"
    UE8M0_SCALE_MIN = "ue8m0_scale_min"
    UE8M0_SCALE_MIXED = "ue8m0_scale_mixed"


def create_boundary_test_data(num_tokens: int, hidden: int, rank: int, rank_offset: int,
                              test_type: BoundaryTestType, device: str = 'gcu'):
    if test_type == BoundaryTestType.ORIGINAL:
        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device) * (rank - rank_offset)

    # ============ BF16 数据边界 ============

    # bf16 最大正值 3.39e38
    elif test_type == BoundaryTestType.BF16_MAX:
        x = torch.full((num_tokens, hidden), torch.finfo(torch.bfloat16).max,
                       dtype=torch.bfloat16, device=device)

    # bf16 最大负值 -3.39e38（与 BF16_MAX 符号对称，FP8 amax 相同）
    elif test_type == BoundaryTestType.BF16_MIN:
        x = torch.full((num_tokens, hidden), torch.finfo(torch.bfloat16).min,
                       dtype=torch.bfloat16, device=device)

    # 全零：FP8 中 amax 被 clamp 到 kFP8Margin=1e-4
    elif test_type == BoundaryTestType.ZERO:
        x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)

    # bf16 最小正规数 1.175e-38，远小于 kFP8Margin=1e-4
    # FP8 中 amax 被 clamp 到 1e-4，行为与 ZERO 相同（测试 kFP8Margin clamp 路径）
    elif test_type == BoundaryTestType.TINY_POSITIVE:
        x = torch.full((num_tokens, hidden), torch.finfo(torch.bfloat16).tiny,
                       dtype=torch.bfloat16, device=device)

    # 同一 batch 内不同 token 有极端差异的值
    # row 0,4,...: +bf16_max, row 1,5,...: -bf16_max, row 2,6,...: +tiny, row 3,7,...: -tiny
    elif test_type == BoundaryTestType.MIXED_EXTREME:
        x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        x[0::4, :] = torch.finfo(torch.bfloat16).max
        x[1::4, :] = torch.finfo(torch.bfloat16).min
        x[2::4, :] = torch.finfo(torch.bfloat16).tiny
        x[3::4, :] = -torch.finfo(torch.bfloat16).tiny

    # 行内列维度混合符号极端值：左半 +bf16_max，右半 -bf16_max
    # 注：|bf16_max| = |bf16_min|，所以所有 group 的 FP8 amax 相同
    # 主要测试 BF16 直传时 dispatch 能否正确处理行内混合符号
    elif test_type == BoundaryTestType.ROW_MIXED:
        x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        x[:, :hidden//2] = torch.finfo(torch.bfloat16).max
        x[:, hidden//2:] = torch.finfo(torch.bfloat16).min

    # ============ FP8 E4M3 量化边界 ============

    # 全 448（FP8 E4M3 最大值），scale=1
    # 精确 & round_scale: fp8=448, dequant=448（精确还原）
    elif test_type == BoundaryTestType.FP8_E4M3_MAX:
        x = torch.full((num_tokens, hidden), 448.0,
                       dtype=torch.bfloat16, device=device)

    # 不同 group 有不同量级的 amax（per-group 独立计算 scale）
    # 前半 groups: 全 1000, amax=1000
    #   精确:      scale=448/1000, fp8=round(1000*0.448)=448, dequant=448*(1000/448)=1000
    #   round_scale: ceil(log2(1000/448))=2, scale_inv=4
    #     fp8=round(1000*0.25)=250→fp8(256), dequant=256*4=1024
    # 后半 groups: 全 448, amax=448, scale=1
    #   fp8=448, dequant=448（所有模式精确）
    elif test_type == BoundaryTestType.FP8_E4M3_OVERFLOW:
        x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        x[:, :hidden//2] = 1000.0
        x[:, hidden//2:] = 448.0

    # 每个 group(128) 内：1 个大值(448)主导 scale，其余小值(0.001)下溢
    # amax=448, scale=1 → fp8(0.001*1)=fp8(0.001)
    # FP8 E4M3 最小次正规 ≈ 0.001953，0.001 < 0.001953 → 量化为 0
    # 测试 per-group 量化中小值被大值的 scale "淹没"的场景
    elif test_type == BoundaryTestType.FP8_E4M3_UNDERFLOW:
        x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        for i in range(0, hidden, 128):
            group_end = min(i + 128, hidden)
            x[:, i:i+1] = 448.0
            x[:, i+1:group_end] = 0.001

    # 每个 group(128) 内混合大小值（测试量化精度损失）
    # 前 64=448, 后 64=0.01, amax=448, scale=1
    # 448 → fp8=448（精确）; 0.01 → fp8≈0.009766（精度损失 2.3%）
    elif test_type == BoundaryTestType.FP8_MIXED_RANGE:
        x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        for i in range(0, hidden, 128):
            x[:, i:i+64] = 448.0
            x[:, i+64:min(i+128, hidden)] = 0.01

    # 测试 round_scale 的 ceil 舍入对量化结果的影响
    # amax=300, raw_scale_inv = 300/448 ≈ 0.6696
    # 精确:      scale=448/300, fp8=round(300*1.493)=448, dequant=448*(300/448)=300（精确还原）
    # round_scale: ceil(log2(0.6696))=0, scale_inv=1, scale=1
    #   fp8=round(300*1)→fp8(288), dequant=288（原值 300 变为 288，精度损失 4%）
    elif test_type == BoundaryTestType.FP8_SCALE_BOUNDARY:
        x = torch.full((num_tokens, hidden), 300.0,
                       dtype=torch.bfloat16, device=device)

    # ============ UE8M0 Scale 边界 ============

    # scale_inv 刚好是 2^n（无舍入误差）
    # 偶数 group: 全 448 → scale_inv=448/448=1=2^0
    # 奇数 group: 全 224 → scale_inv=224/448=0.5=2^(-1)
    elif test_type == BoundaryTestType.UE8M0_SCALE_EXACT_POW2:
        x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        for i in range(0, hidden, 128):
            group_end = min(i + 128, hidden)
            if (i // 128) % 2 == 0:
                x[:, i:group_end] = 448.0
            else:
                x[:, i:group_end] = 224.0

    # scale_inv 不是 2 的幂，ceil 向上舍入
    # amax=600 → raw_scale_inv=600/448≈1.339
    # ceil(log2(1.339))=ceil(0.421)=1 → scale_inv=2^1=2
    # fp8=round(600*0.5)=300→fp8(288), dequant=288*2=576（原值 600 → 576）
    elif test_type == BoundaryTestType.UE8M0_SCALE_ROUND_UP:
        x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        for i in range(0, hidden, 128):
            group_end = min(i + 128, hidden)
            x[:, i:group_end] = 600.0

    # scale_inv 接近 UE8M0 最小可表示值
    # amax=0.001 → raw_scale_inv=0.001/448≈2.23e-6
    # ceil(log2(2.23e-6))=-18 → scale_inv=2^(-18)≈3.815e-6
    # fp8=round(0.001/3.815e-6)≈262→fp8(256), dequant=256*3.815e-6≈0.000977
    elif test_type == BoundaryTestType.UE8M0_SCALE_MIN:
        x = torch.full((num_tokens, hidden), 0.001,
                       dtype=torch.bfloat16, device=device)

    # 不同 group 有截然不同量级的 scale
    # group%3==0: 448 → scale_inv=1=2^0
    # group%3==1: 10000 → scale_inv ceil→2^5=32
    # group%3==2: 0.01 → scale_inv ceil→2^(-15)≈3.05e-5
    elif test_type == BoundaryTestType.UE8M0_SCALE_MIXED:
        x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        for i in range(0, hidden, 128):
            group_end = min(i + 128, hidden)
            group_idx = i // 128
            if group_idx % 3 == 0:
                x[:, i:group_end] = 448.0
            elif group_idx % 3 == 1:
                x[:, i:group_end] = 10000.0
            else:
                x[:, i:group_end] = 0.01

    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return x
