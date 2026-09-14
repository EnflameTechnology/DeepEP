#pragma once

#include "exception.h"
#include "prims.h"

using T_U64   = unsigned long long;
using T_U64Vt = __vector unsigned long long;

// Async DTE engine: overlaps consecutive local memcpy operations.
// The previous async copy is waited only when the next one is issued,
// allowing computation between two copies to overlap with DTE transfer.
class dte_engine_async {
public:
    template<typename T>
    __forceinline__ __device__ void memcpyGtoGx(tops::private_dte& engine, T* dst, const T* src, ssize_t nelem) {
        tops::mdspan srcL3(tops::Global, const_cast<T*>(src), static_cast<int>(nelem));
        tops::mdspan dstL3(tops::Global, dst, static_cast<int>(nelem));

        // Wait for previous operation to complete before starting new one
        if (event_in_flight_) {
            event_.wait();
            event_in_flight_ = false;
        }

        // Launch new async memcpy
        event_ = tops::memcpy_async(engine, dstL3, srcL3);
        event_in_flight_ = true;
    }

    template<typename T>
    __forceinline__ __device__ void memcpyLtoGx(tops::private_dte& engine,
                                                 T* dst_remote,
                                                 const T* src_local,
                                                 ssize_t nelem) {
        tops::mdspan srcL(tops::Local, const_cast<T*>(src_local), static_cast<int>(nelem));
        tops::mdspan dstGx(tops::Global, dst_remote, static_cast<int>(nelem));
        if (event_in_flight_) {
            event_.wait();
            event_in_flight_ = false;
        }
        event_ = tops::memcpy_async(engine, dstGx, srcL);
        event_in_flight_ = true;
    }

    __forceinline__ __device__ void wait() {
        if (event_in_flight_) {
            event_.wait();
            event_in_flight_ = false;
        }
    }

private:
    bool event_in_flight_ = false;
    tops::event event_;
};

template<typename T>
__forceinline__ __device__ void memcpyGtoG(tops::private_dte& engine, T* dst, const T* src, ssize_t nelem) {
  tops::mdspan srcL3(tops::Global, const_cast<T*>(src), static_cast<int>(nelem));
  tops::mdspan dstL3(tops::Global, dst, static_cast<int>(nelem));
  tops::event event = tops::memcpy_async(engine, dstL3, srcL3);
  event.wait();
}

template <typename T>
__forceinline__ __device__ void transferGlobaltoStackUnaligned(T_U64* stack, T* global) {
  constexpr auto nelemPerGlobal = sizeof(T)/sizeof(T_U64);
  static_assert(0 == (sizeof(T) % sizeof(T_U64)), "transferGlobaltoStackUnaligned require sizeof(T) aligned with T_U64");
  tcle::set_vl(nelemPerGlobal);
  T_U64Vt v;
  T_U64Vt* glb = reinterpret_cast<T_U64Vt *>(global);
  v = *glb;
  for (size_t i=0;i<nelemPerGlobal;i++)
    stack[i] = v[i];
}


#if defined __cplusplus
#define __OP_ASSERT_NO_CAST static_cast<void>
#else
#define __OP_ASSERT_NO_CAST (void)
#endif

#define __OP_ASSERT_STR(x) #x

#define __op_assert_fail(assertion, file_name, line)                    \
  (__builtin_printf(file_name                                           \
      " " __OP_ASSERT_STR(line) " : "                                   \
      "op_assertion " __OP_ASSERT_STR(assertion) " failed.\r\n"), tops::abort())
#define op_assert(val, expr)                                            \
  ((val) ? (__OP_ASSERT_NO_CAST(0)) :                                   \
  (__op_assert_fail(#expr, __FILE_NAME__, __LINE__)));

namespace deep_ep {
template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id,
    int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
    token_start_idx = (num_tokens_per_sm * sm_id) > num_tokens ? num_tokens : (num_tokens_per_sm * sm_id);
    token_end_idx = (token_start_idx + num_tokens_per_sm) > num_tokens ? num_tokens : (token_start_idx + num_tokens_per_sm);
}

template <int kNumRanks>
__device__ __forceinline__ bool not_finished(int *task, int expected) {
    auto result = false;
    result = *((volatile int*)task) != expected;
    return result;
}

template <int kNumRanks>
__forceinline__ __device__ void
timeout_check(int **task_fifo_ptrs, int rank, int expected, int subthread_id) {
    auto start_time = tops::clock64();
    while (not_finished<kNumRanks>(task_fifo_ptrs[rank] + subthread_id, expected)) {
        if (tops::clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf("DeepEP timeout check failed: (rank = %d), value is %d, peer_rank is %d\n", rank, *(volatile int*)(task_fifo_ptrs[rank] + subthread_id), subthread_id);
            // tops::abort();
        }
    }
}

template <int kNumRanks>
__forceinline__ __device__ void
barrier_device(int* cached_value_ptrs, int **task_fifo_ptrs, int rank, primitives_simple* prim) {
    const auto sm_id = static_cast<int>(threadIdx.x) + (static_cast<int>(blockDim.x) * static_cast<int>(blockIdx.x));
    auto subthread_id = static_cast<int>(subThreadIdx.x);
    if ((subthread_id < kNumRanks) && sm_id == 0) {
        tcle::atomic_add(task_fifo_ptrs[rank] + subthread_id, 1);
        prim->atomic_add(cached_value_ptrs + subthread_id, task_fifo_ptrs[subthread_id] + rank, -1, subthread_id, rank);
        timeout_check<kNumRanks>(task_fifo_ptrs, rank, 0, subthread_id);
    }
    __syncsubthreads();
}

// Memory-semantic device-wide barrier over LARE-registered peer_direct_addrs.
//
template <int kNumRanks>
__forceinline__ __device__ void
barrier_device_peer_memory(uint64_t* volatile* peer_direct_addrs,
                           int rank,
                           int slot_offset) {
    const auto sm_id = static_cast<int>(threadIdx.x) +
                       (static_cast<int>(blockDim.x) * static_cast<int>(blockIdx.x));
    auto subthread_id = static_cast<int>(subThreadIdx.x);

    constexpr uint64_t kArrived = 1ULL;
    constexpr uint64_t kReset   = 0ULL;

    if ((subthread_id < kNumRanks) && sm_id == 0) {
        const int peer = subthread_id;
        volatile uint64_t* peer_slot = peer_direct_addrs[peer] + slot_offset + rank;
        volatile uint64_t* my_slot   = peer_direct_addrs[rank] + slot_offset + peer;

        // (1) Remote write: signal arrival to peer.
        *peer_slot = kArrived;
        tcle::fence<tcle::FenceType::L3_MEM_STORE>();

        // (2) Local spin until peer writes kArrived here.
        auto start_time = tops::clock64();
        while (*my_slot != kArrived) {
            if (tops::clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("DeepEP barrier_device_peer_memory timeout: rank=%d peer=%d "
                       "slot_offset=%d got=0x%llx\n",
                       rank, peer, slot_offset,
                       (unsigned long long)(*my_slot));
            }
        }

        // (3) Local reset: prepare slot for next (ping-pong) reuse.
        *my_slot = kReset;
    }
    tcle::fence<tcle::L3_MEM_STORE>();
    __syncsubthreads();
}

constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;
constexpr float kFP8ScaleEps = 1e-10f;
constexpr int kFP8CfuncUnrollSize = 8;

template <typename VecIntT>
__device__ __forceinline__ VecIntT make_linear_scale_ids() {
  return {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
      26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
      39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
      52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}


template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
__forceinline__ __device__ out_dtype_t extract_required_scale_format(float value) {
    if constexpr (kIsUE8M0) {
        return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
    } else {
        return value;
    }
}

__device__ __forceinline__ int fast_log2_ceil(float x) {
  uint32_t bits_x = *reinterpret_cast<uint32_t*>(&x);
  int exp_x = (int)((bits_x >> 23) & 0xff) - 127;
  uint32_t man_bits = bits_x & ((1u << 23) - 1u);
  return exp_x + (man_bits != 0);
}
__device__ __forceinline__ float fast_pow2(int e) {
  uint32_t bits_x = (uint32_t)(e + 127) << 23;
  return *reinterpret_cast<float*>(&bits_x);
}

__device__ __forceinline__ float calculate_fp8_scale_acc_cfunc(
    float amax, bool round_scale) {
  float scale_acc = tcle::max(amax, kFP8ScaleEps) * kFinfoAmaxInvE4M3;
  if (!round_scale)
    return scale_acc;

  const float clamped = tcle::max(tcle::abs(scale_acc), kFP8ScaleEps);
  const int exp_scale_acc = fast_log2_ceil(clamped);
  return fast_pow2(exp_scale_acc);
}

// 64-bit element indexing helper.
//
// Use this only for indices that may overflow int32, e.g. `token_idx * kHidden`
// where num_tokens or num_recv_tokens can reach ~1M and kHidden is on the order
// of 7168 (1M * 7168 ≈ 7.2G > 2^31). Indices that only multiply by num_ranks /
// num_topk (small constants) do NOT need this and should use plain `ptr[idx]`.
template <typename T>
__forceinline__ __device__ T* ptr_at_elem(T* base, int64_t elem_idx) {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(base) +
                                elem_idx * static_cast<int64_t>(sizeof(T)));
}

template <typename T>
__forceinline__ __device__ const T* ptr_at_elem(const T* base, int64_t elem_idx) {
    return reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) +
                                      elem_idx * static_cast<int64_t>(sizeof(T)));
}

template<int bytes>
class InstG2G {
public:
__forceinline__ __device__ void fetch(void* src) {
    auto in_leaptr = tcle::simple_leaptr<__vector char>(src);
    #pragma clang loop unroll(full)
    for (int i = 0; i < UNROLL - 1; i++) {
      tmp[i] = in_leaptr.load();
    }
    tcle::set_vl(std::min( bytes - (UNROLL - 1) * 512, 512));
    tmp[UNROLL - 1] = in_leaptr.load();
    tcle::set_vl(512);
}
__forceinline__ __device__ void store(void* dst) {
    auto out_leaptr = tcle::simple_leaptr<__vector char>(dst);
    #pragma clang loop unroll(full)
    for (int i = 0; i < UNROLL - 1; i++) {
      out_leaptr.store(tmp[i]);
    }
    tcle::set_vl(std::min( bytes - (UNROLL - 1) * 512, 512));
    out_leaptr.store(tmp[UNROLL - 1]);
    tcle::set_vl(512);
  }
private:
    static constexpr int UNROLL = (bytes + 512 - 1) / 512;
    __vector char tmp[UNROLL];
};



template<bool kUseUE8M0>
class ScaleStorer {
  public:
template <typename scale_t>
__device__ __forceinline__ void write_fp8_scales_strided(
    scale_t* dst_scales, int num_scales, int pack_stride) {
  if constexpr (kUseUE8M0) {
    using VF = typename tcle::altivector<float, 512>::VT;
    using VU = typename tcle::altivector<unsigned int, 512>::VT;
    using VU8 = typename tcle::altivector<unsigned char, 512>::VT;
    using VInt = typename tcle::altivector<int, 512>::VT;
    using M = typename tcle::altivector_to_mask<VF>::type;

    const VInt linear_ids = make_linear_scale_ids<VInt>();

    //tcle::set_vl(num_scales);
    M mask = tcle::vset_mb<M>(num_scales);
    //VF raw_scales = tcle::load<VF>(
    //    reinterpret_cast<void*>(const_cast<float*>(src_scales)), (VF)0, mask);
    VU bits = tcle::bit_cast<VU>(scalesx4);
    VU8 exps = tcle::cvt<VU8>((bits << 1) >> 24);
    VInt pack_idx = linear_ids >> 2;
    VInt elem_idx = linear_ids & 3;
    VInt offset = pack_idx * pack_stride + elem_idx;
    tcle::scatter(exps, dst_scales, offset, mask);
  } else {
    //using VF = tcle::simple_altivector<float>::VT;
    //using M = typename tcle::altivector_to_mask<VF>::type;
    tcle::set_vl(num_scales);
    //M mask = tcle::vset_mb<M>(num_scales);
    //VF scales = tcle::load<VF>(
    //    reinterpret_cast<void*>(const_cast<float*>(src_scales)), (VF)0, mask);
    tcle::store_stride(scales, dst_scales, pack_stride);
    tcle::set_vl(512);
  }
}

__device__ __forceinline__ void fetch_scale(const float* src_scales, int num_scales) {

  if constexpr (kUseUE8M0) {
    using VF = typename tcle::altivector<float, 512>::VT;
    using M = typename tcle::altivector_to_mask<VF>::type;

    M mask = tcle::vset_mb<M>(num_scales);
    scalesx4 = tcle::load<VF>(
        reinterpret_cast<void*>(const_cast<float*>(src_scales)), (VF)0, mask);
  } else {
    using VF = tcle::simple_altivector<float>::VT;
    using M = typename tcle::altivector_to_mask<VF>::type;

    M mask = tcle::vset_mb<M>(num_scales);
    scales = tcle::load<VF>(
        reinterpret_cast<void*>(const_cast<float*>(src_scales)), (VF)0, mask);
  }
}
  private:
  typename tcle::altivector<float, 512>::VT scalesx4;
  typename tcle::altivector<float>::VT scales;
};


template <int kNumPerChannels, int kUnrollSize = kFP8CfuncUnrollSize>
__device__ __forceinline__ void quantize_fp8_groups_cfunc_even(
    const tops::__ef_bfloat16* x_bf16, uint8_t* out_fp8_bytes, float* out_scales,
    bool round_scale, int start_group, int end_group) {
  if (start_group >= end_group)
    return;

  tcle::set_vl(kNumPerChannels);
  int group_idx = start_group;
  while (group_idx + kUnrollSize <= end_group) {
    __vector float fp32_vec[kUnrollSize];
    __vector float scale_acc_vec[kUnrollSize];
    __vector char fp8_vec[kUnrollSize];

#pragma unroll
    for (int i = 0; i < kUnrollSize; ++i) {
      const int elem_base = (group_idx + i) * kNumPerChannels;
      const __vector __bf16 bf16_vec =
          *reinterpret_cast<const __vector __bf16*>(&x_bf16[elem_base]);
      const __vector2 float fp32_buf = tcle::cvt<__vector2 float>(bf16_vec);
      fp32_vec[i] = *reinterpret_cast<const __vector float*>(&fp32_buf);
    }

#pragma unroll
    for (int i = 0; i < kUnrollSize; ++i) {
      const float amax = tcle::redmax<0>(tcle::abs(fp32_vec[i]))[0];
      scale_acc_vec[i] =
          (__vector float)(calculate_fp8_scale_acc_cfunc(amax, round_scale));
    }

#pragma unroll
    for (int i = 0; i < kUnrollSize; ++i)
      out_scales[group_idx + i] = scale_acc_vec[i][0];

#pragma unroll
    for (int i = 0; i < kUnrollSize; ++i) {
      const __vector float scaled_vec =
          fp32_vec[i] * tcle::msf<6>(scale_acc_vec[i]);
      __vector4 float scaled_x4 = (__vector4 float)(0.f);
      *reinterpret_cast<__vector float*>(&scaled_x4) = scaled_vec;
      fp8_vec[i] = tcle::cvt<__vector char, F8E4M3 | CLAMP>(scaled_x4);
    }

#pragma unroll
    for (int i = 0; i < kUnrollSize; ++i) {
      const int elem_base = (group_idx + i) * kNumPerChannels;
      *reinterpret_cast<__vector char*>(&out_fp8_bytes[elem_base]) = fp8_vec[i];
    }

    group_idx += kUnrollSize;
  }

  for (; group_idx < end_group; ++group_idx) {
    const int elem_base = group_idx * kNumPerChannels;
    const __vector __bf16 bf16_vec =
        *reinterpret_cast<const __vector __bf16*>(&x_bf16[elem_base]);
    const __vector2 float fp32_buf = tcle::cvt<__vector2 float>(bf16_vec);
    const __vector float fp32_vec =
        *reinterpret_cast<const __vector float*>(&fp32_buf);
    const float amax = tcle::redmax<0>(tcle::abs(fp32_vec))[0];
    const float scale_acc = calculate_fp8_scale_acc_cfunc(amax, round_scale);
    out_scales[group_idx] = scale_acc;

    const __vector float scaled_vec =
        fp32_vec * tcle::msf<6>((__vector float)(scale_acc));
    __vector4 float scaled_x4 = (__vector4 float)(0.f);
    *reinterpret_cast<__vector float*>(&scaled_x4) = scaled_vec;
    const __vector char fp8_vec =
        tcle::cvt<__vector char, F8E4M3 | CLAMP>(scaled_x4);
    *reinterpret_cast<__vector char*>(&out_fp8_bytes[elem_base]) = fp8_vec;
  }
}

__device__ __forceinline__ __vector unsigned int fast_log2_ceil(__vector float x)
{
  __vector unsigned int exp = tcle::gete(x) - 127;
  __vector unsigned int man = tcle::getm(x);
  auto vone = (__vector unsigned int)(1);
  auto vzero = (__vector unsigned int)(0);
  auto offset = tcle::vsel(man == 0, vzero, vone);
  return exp + offset;
}

__device__ __forceinline__ __vector float fast_pow2(__vector unsigned int e) {
  auto tmp = (e + 127) << 23;
  return (__vector float)(tmp);
}

// syncsubthreads is used, make sure all subthreads enter this func
template<int kHidden, int kNumPerChannels>
__forceinline__ __device__ void
token_quantize_t(bool round_scale, void* rdma_x_vec, float* rdma_x_scales,
    const tops::__ef_bfloat16* x_bf16, int subthread_id, int num_subthreads) {
  static_assert(kNumPerChannels==128);
  const int num_scales = kHidden / kNumPerChannels;
  __local__ __valigned__ float tmp_scales[128];
  __local__ __valigned__ int16_t tmp_token[kHidden];
  using FP32VT = vector float;
  using FP32VT2 = vector2 float;
  using FP32VT4 = vector4 float;
  using BF16VT = vector __bf16;
  using BF16VT2 = vector2 __bf16;
  using FP8VT = vector char;
  auto curr_rdma_x_vec =
      reinterpret_cast<tops::float_e4m3*>(rdma_x_vec);
  auto curr_rdma_x_scales = rdma_x_scales;

  if (subthread_id  == 0) {
    auto input_tmp_leaptr =
      tcle::simple_leaptr<BF16VT>((void*)(x_bf16));
    auto output_tmp_leaptr =
      tcle::simple_leaptr<BF16VT>((void*)(tmp_token));
    BF16VT vtoken[kHidden / 256];
    #pragma clang loop unroll(full)
    for (int h = 0; h < kHidden / 256; h++) {
      vtoken[h] = input_tmp_leaptr.load();
    }
    #pragma clang loop unroll(full)
    for (int h = 0; h < kHidden / 256; h++) {
      output_tmp_leaptr.store(vtoken[h]);
    }

    constexpr int UNROLL_Q = 16;
    static_assert(kNumPerChannels % UNROLL_Q == 0);
    BF16VT vin[UNROLL_Q];
    FP32VT vmax[UNROLL_Q];
    //FP32VT final_vmax;
    for (int ee = 0; ee < UNROLL_Q; ee++) {
      vmax[ee] = FP32VT(kFP8Margin);
    }

    tcle::set_vl(num_scales);
    for (int e = 0; e < kNumPerChannels; e+=UNROLL_Q) {
      #pragma clang loop unroll(full)
      for (int ee = 0; ee < UNROLL_Q; ee++) {
        vin[ee] = tcle::loadeoa<BF16VT>((void*)(tmp_token + e + ee), kNumPerChannels);
        //vin[ee] = tcle::gather<BF16VT>((void*)(curr_x_bf16 + e + ee), offset);
      }
      #pragma clang loop unroll(full)
      for (int ee = 0; ee < UNROLL_Q; ee++) {
        auto vfp32 = tcle::extract<FP32VT, 0>(tcle::cvt<FP32VT2>(vin[ee]));
        vmax[ee] = tcle::absmax(vmax[ee], vfp32);
      }
    }
    tcle::set_vl(512);
    //printf("main absmax time %lld \n", tops::clock64()-start_time);

    #pragma clang loop unroll(full)
    for (int ee = 1; ee < UNROLL_Q; ee++) {
      vmax[0] = tcle::max(vmax[0], vmax[ee]);
    }

    #pragma clang loop unroll(full)
    for (int ee = 0; ee < UNROLL_Q / 2; ee++) {
      vmax[ee] = tcle::max(vmax[ee], vmax[ee + UNROLL_Q / 2]);
    }
    #pragma clang loop unroll(full)
    for (int ee = 0; ee < UNROLL_Q / 4; ee++) {
      vmax[ee] = tcle::max(vmax[ee], vmax[ee + UNROLL_Q / 4]);
    }
    #pragma clang loop unroll(full)
    for (int ee = 0; ee < UNROLL_Q / 8; ee++) {
      vmax[ee] = tcle::max(vmax[ee], vmax[ee + UNROLL_Q / 8]);
    }
    #pragma clang loop unroll(full)
    for (int ee = 0; ee < UNROLL_Q / 16; ee++) {
      vmax[ee] = tcle::max(vmax[ee], vmax[ee + UNROLL_Q / 16]);
    }
    //printf("tail max time %lld \n", tops::clock64()-start1_time);
    auto final_vmax = vmax[0];
    static_assert(UNROLL_Q <= 16);
    //printf("vmax[0] %f vmax[1] %f vmax[2] %f vmax[3] %f \n", vmax[0], vmax[1], vmax[2], vmax[3]);
    FP32VT vscale_inv, vscale;

    if (round_scale) {
      auto exp_scale_inv =
          fast_log2_ceil(final_vmax * kFinfoAmaxInvE4M3);
      vscale_inv = fast_pow2(exp_scale_inv);
      vscale = fast_pow2(-exp_scale_inv);
      //auto tmp_inv = final_vmax * kFinfoAmaxInvE4M3;
      ////log2
      //__vector unsigned int exp = tcle::gete(tmp_inv) - 127;
      //__vector unsigned int man = tcle::getm(tmp_inv);
      //auto vone = (__vector unsigned int)(1);
      //auto vzero = (__vector unsigned int)(0);

      //auto offset = tcle::vsel(man == 0, vzero, vone);
      //auto logout = exp + offset;
      //// pow2
      //auto  tmp = (logout + 127) << 23;
      //vscale_inv = (__vector float)(tmp);
      //auto  tmp1 = (-logout + 127) << 23;
      //vscale = (__vector float)(tmp1);

    } else {
      vscale_inv = final_vmax * kFinfoAmaxInvE4M3;
      vscale = kFinfoAmaxE4M3 / final_vmax;
    }

    tcle::set_vl(num_scales);
    tcle::store(vscale_inv, curr_rdma_x_scales);
    tcle::store(vscale, tmp_scales);
    tcle::set_vl(512);

  }
  // fence l1 store
  tcle::fence<tcle::FenceType::L1_VDMEM_STORE>();
  tops::sync_subthreads();
  //__syncsubthreads();
  //tcle::set_vl(num_scales);
  //auto vscale_tmp = tcle::load<FP32VT>(tmp_scales);
  //tcle::set_vl(512);
  auto input_leaptr =
      tcle::leaptr<BF16VT2>((void*)(tmp_token + subthread_id * 4 * 128), num_subthreads * sizeof(BF16VT2));
  tcle::tar base =
      tcle::movsr2tar(reinterpret_cast<int64_t>(curr_rdma_x_vec + subthread_id * 4 * 128));
  tcle::tar stride = tcle::movsr2tar(128);
  tcle::tar stride1 = tcle::movsr2tar(128 + (num_subthreads - 1) * 4 * 128);
  for (int g = 4 * subthread_id; g < num_scales; g += (4 * num_subthreads)) {
    auto float_in = tcle::cvt<FP32VT4>(input_leaptr.load());
    auto float_in0 = tcle::extract<FP32VT, 0>(float_in);
    auto float_in1 = tcle::extract<FP32VT, 1>(float_in);
    auto float_in2 = tcle::extract<FP32VT, 2>(float_in);
    auto float_in3 = tcle::extract<FP32VT, 3>(float_in);
    float_in0 = float_in0 * tmp_scales[g];
    float_in1 = float_in1 * tmp_scales[g + 1];
    float_in2 = float_in2 * tmp_scales[g + 2];
    float_in3 = float_in3 * tmp_scales[g + 3];
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in0, base, stride);
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in1, base, stride);
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in2, base, stride);
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in3, base, stride1);
  }
}

// Per-tensor external-scale quantization.

// All subthreads must enter this function (uses sync_subthreads internally).
template<int kHidden, int kNumPerChannels>
__forceinline__ __device__ void
token_quantize_with_external_scale_t(void* rdma_x_vec, float* rdma_x_scales,
    const tops::__ef_bfloat16* x_bf16, float ext_scale,
    int subthread_id, int num_subthreads) {
  static_assert(kNumPerChannels == 128);
  const int num_scales = kHidden / kNumPerChannels;
  __local__ __valigned__ float tmp_scales[128];
  __local__ __valigned__ int16_t tmp_token[kHidden];
  using FP32VT = vector float;
  using FP32VT4 = vector4 float;
  using BF16VT = vector __bf16;
  using BF16VT2 = vector2 __bf16;
  using FP8VT = vector char;
  auto curr_rdma_x_vec = reinterpret_cast<tops::float_e4m3*>(rdma_x_vec);
  auto curr_rdma_x_scales = rdma_x_scales;

  if (subthread_id == 0) {
    // Phase 1: copy x_bf16 from global to L1 scratch
    auto input_tmp_leaptr = tcle::simple_leaptr<BF16VT>((void*)(x_bf16));
    auto output_tmp_leaptr = tcle::simple_leaptr<BF16VT>((void*)(tmp_token));
    BF16VT vtoken[kHidden / 256];
    #pragma clang loop unroll(full)
    for (int h = 0; h < kHidden / 256; h++) {
      vtoken[h] = input_tmp_leaptr.load();
    }
    #pragma clang loop unroll(full)
    for (int h = 0; h < kHidden / 256; h++) {
      output_tmp_leaptr.store(vtoken[h]);
    }
    // Broadcast external scale to all num_scales slots (no absmax computation).
    // rdma_x_scales: dequant scale (fp8 → bf16 multiplier), sent via RDMA.
    // tmp_scales: quant scale (bf16 → fp8 multiplier = 1/ext_scale), used in Phase 2.
    const float ext_quant_scale = 1.0f / ext_scale;
    tcle::set_vl(num_scales);
    tcle::store((__vector float)(ext_scale), curr_rdma_x_scales);
    tcle::store((__vector float)(ext_quant_scale), tmp_scales);
    tcle::set_vl(512);
  }
  // Fence + sync: ensure subthread 0's L1 writes are visible to all subthreads
  tcle::fence<tcle::FenceType::L1_VDMEM_STORE>();
  tops::sync_subthreads();

  // Phase 2: quantize bf16 → fp8 using ext_quant_scale (identical to token_quantize_t Phase 2)
  auto input_leaptr =
      tcle::leaptr<BF16VT2>((void*)(tmp_token + subthread_id * 4 * 128), num_subthreads * sizeof(BF16VT2));
  tcle::tar base =
      tcle::movsr2tar(reinterpret_cast<int64_t>(curr_rdma_x_vec + subthread_id * 4 * 128));
  tcle::tar stride = tcle::movsr2tar(128);
  tcle::tar stride1 = tcle::movsr2tar(128 + (num_subthreads - 1) * 4 * 128);
  for (int g = 4 * subthread_id; g < num_scales; g += (4 * num_subthreads)) {
    auto float_in = tcle::cvt<FP32VT4>(input_leaptr.load());
    auto float_in0 = tcle::extract<FP32VT, 0>(float_in);
    auto float_in1 = tcle::extract<FP32VT, 1>(float_in);
    auto float_in2 = tcle::extract<FP32VT, 2>(float_in);
    auto float_in3 = tcle::extract<FP32VT, 3>(float_in);
    float_in0 = float_in0 * tmp_scales[g];
    float_in1 = float_in1 * tmp_scales[g + 1];
    float_in2 = float_in2 * tmp_scales[g + 2];
    float_in3 = float_in3 * tmp_scales[g + 3];
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in0, base, stride);
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in1, base, stride);
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in2, base, stride);
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in3, base, stride1);
  }
}

// Per-tensor external-scale quantization from L1 buffer (internode prefetch path).
// x_l1 must already be in L1 (loaded by caller via memcpy_async prefetch).
// This variant skips the internal global→L1 copy, enabling overlapped prefetch.
// ext_scale is a single scalar for the whole token tensor (same semantics as
//   token_quantize_with_external_scale_t).
// All subthreads must enter this function (uses sync_subthreads internally).
template<int kHidden, int kNumPerChannels>
__forceinline__ __device__ void
token_quantize_with_external_scale_from_l1_t(void* rdma_x_vec, float* rdma_x_scales,
    const tops::__ef_bfloat16* x_l1, float ext_scale,
    int subthread_id, int num_subthreads) {
  static_assert(kNumPerChannels == 128);
  const int num_scales = kHidden / kNumPerChannels;
  __local__ __valigned__ float tmp_scales[128];
  using FP32VT = vector float;
  using FP32VT4 = vector4 float;
  using BF16VT2 = vector2 __bf16;
  using FP8VT = vector char;
  auto curr_rdma_x_vec = reinterpret_cast<tops::float_e4m3*>(rdma_x_vec);
  auto curr_rdma_x_scales = rdma_x_scales;

  if (subthread_id == 0) {
    // No copy: x_l1 already in L1 (prefetched by caller).
    // Broadcast scale values to all num_scales slots.
    const float ext_quant_scale = 1.0f / ext_scale;
    tcle::set_vl(num_scales);
    tcle::store((__vector float)(ext_scale), curr_rdma_x_scales);
    tcle::store((__vector float)(ext_quant_scale), tmp_scales);
    tcle::set_vl(512);
  }
  tcle::fence<tcle::FenceType::L1_VDMEM_STORE>();
  tops::sync_subthreads();

  // Phase 2: read from x_l1 (L1). Identical FP8 cast logic to token_quantize_with_external_scale_t.
  auto input_leaptr =
      tcle::leaptr<BF16VT2>((void*)(x_l1 + subthread_id * 4 * 128), num_subthreads * sizeof(BF16VT2));
  tcle::tar base =
      tcle::movsr2tar(reinterpret_cast<int64_t>(curr_rdma_x_vec + subthread_id * 4 * 128));
  tcle::tar stride = tcle::movsr2tar(128);
  tcle::tar stride1 = tcle::movsr2tar(128 + (num_subthreads - 1) * 4 * 128);
  for (int g = 4 * subthread_id; g < num_scales; g += (4 * num_subthreads)) {
    auto float_in = tcle::cvt<FP32VT4>(input_leaptr.load());
    auto float_in0 = tcle::extract<FP32VT, 0>(float_in);
    auto float_in1 = tcle::extract<FP32VT, 1>(float_in);
    auto float_in2 = tcle::extract<FP32VT, 2>(float_in);
    auto float_in3 = tcle::extract<FP32VT, 3>(float_in);
    float_in0 = float_in0 * tmp_scales[g];
    float_in1 = float_in1 * tmp_scales[g + 1];
    float_in2 = float_in2 * tmp_scales[g + 2];
    float_in3 = float_in3 * tmp_scales[g + 3];
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in0, base, stride);
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in1, base, stride);
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in2, base, stride);
    tcle::internal_tar_cvt_store<FP8VT, FP32VT, F8E4M3 | CLAMP>(
        float_in3, base, stride1);
  }
}


} // namespace deep_ep
