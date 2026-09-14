#pragma once

#include "devcomm.h"

#define MAX_IBGDA_EP_SIZE   (264) // size of MORI Endpoint entry (actual data size)
#define MAX_IBGDA_QP_NUMS   (32)  // number of QPs of MORI
#define MAX_IBGDA_MR_SIZE   (24)  // size of MORI Memory Regions
#define DEFAULT_VL_LEN      (512) // default Vector length

// Align up macro: round x up to the nearest multiple of align
#define ALIGN_UP(x, align) (((x) + (align) - 1) / (align) * (align))

// Aligned EP size: each endpoint is 128-byte aligned (264 -> 384 bytes)
#define MAX_IBGDA_EP_SIZE_ALIGNED ALIGN_UP(MAX_IBGDA_EP_SIZE, 128)

// Unified buffer access macros for LARE/IBGDA endpoint storage
//
// Unified Buffer Layout: [MR Region][Padding][Endpoint Region]
//   - MR Region:       rank0, rank1, ..., rank(EP_MAX_RANKS-1)  (24 bytes each)
//   - Endpoint Region: rank0[qp0-31], rank1[qp0-31], ..., rank(EP_MAX_RANKS-1)[qp0-31]
//                      Each endpoint is 384 bytes (aligned); EP_MAX_RANKS = 256
//
// Filling order matches the layout: rank0[qp0-31], rank1[qp0-31], ...
// This is used by both LARE (intra-node) and IBGDA (inter-node)

#define IBGDA_MR_OFFSET (0UL)
// EP offset is aligned to 128 bytes for better memory access
// = ALIGN_UP(EP_MAX_RANKS × 24 bytes, 128) = ALIGN_UP(256×24=6144, 128) = 6144 bytes
#define IBGDA_EP_OFFSET ALIGN_UP(MAX_IBGDA_MR_SIZE * EP_MAX_RANKS, 128)

typedef struct {
    int num_rc_per_pe;          // master ESL QP count (adaptive: kMasterChannelCount*2)
    int num_slave_qp_per_pe;    // slave-mode QP count (adaptive: kSlaveChannelCount*2)
} qp_msg_meta_t;

// Total unified buffer size
// = 6144 + (EP_MAX_RANKS × 32 qps × 384 bytes) + sizeof(meta)
// = 6144 + (256 × 32 × 384) + 8 = 6144 + 3,145,728 + 8 ≈ 3 MB
#define MAX_IBGDA_INFO_SIZE (IBGDA_EP_OFFSET + EP_MAX_RANKS * MAX_IBGDA_EP_SIZE_ALIGNED * MAX_IBGDA_QP_NUMS + sizeof(qp_msg_meta_t))

#define QP_MSG_META_OFFSET (IBGDA_EP_OFFSET + EP_MAX_RANKS * MAX_IBGDA_EP_SIZE_ALIGNED * MAX_IBGDA_QP_NUMS)

// Get MR offset for specified rank
// Layout: rank0, rank1, ..., rank31 (each 24 bytes)
#define GET_UNIFIED_MR_OFFSET(rank) \
    (IBGDA_MR_OFFSET + (rank) * MAX_IBGDA_MR_SIZE)

// Get endpoint offset for specified rank and sm_id
// Layout: [rank][sm_id] - rank0[sm0-31], rank1[sm0-31], ..., rank31[sm0-31]
// Formula: IBGDA_EP_OFFSET + rank×(32×384) + sm_id×384
//   - Each rank has 32 endpoints (sm_id: 0-31)
//   - Each endpoint is 384 bytes (aligned)
// Example: rank=1, sm_id=5: offset = 768 + 1×12,288 + 5×384 = 14,976
#define GET_UNIFIED_EP_OFFSET(rank, sm_id) \
    (IBGDA_EP_OFFSET + \
     (rank) * MAX_IBGDA_EP_SIZE_ALIGNED * MAX_IBGDA_QP_NUMS + \
     (sm_id) * MAX_IBGDA_EP_SIZE_ALIGNED)

// Type-safe endpoint pointer getter (optional, for convenience)
#define GET_UNIFIED_EP_PTR(base_ptr, rank, sm_id) \
    ((unified_endpoint*)((uintptr_t)(base_ptr) + GET_UNIFIED_EP_OFFSET(rank, sm_id)))

#define NUM_MAX_NVL_PEERS 256
#define NUM_MAX_RDMA_PEERS 20
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define LOW_LATENCY_MAX_EXPERTS 1024
#define LOW_LATENCY_MAX_TOKENS 4096

// Low-latency intranode dispatch: total subthreads per thread (launch threadDim.x).
#define LOW_LATENCY_DISPATCH_SUBTHREADS 8
#define LOW_LATENCY_DISPATCH_SUBTHREADS_SLAVE 4
// Each subthread owns 2 private_dte + dte_engine_async banks for ping-pong.
#define LOW_LATENCY_DISPATCH_DTE_PINGPONG 2

#define NUM_MAX_INTRA_CHANNELS 12
#define NUM_BUFFER_ALIGNMENT_BYTES 128

// Total bytes of the intranode `cached_value_ptrs` pool (allocated in
// deep_ep.cpp, consumed by intranode kernels). MUST stay in sync between the
// allocator and any kernel-side capacity guards (see intranode.tops dispatch
// host wrapper). Per-call need is
//   4 * num_ranks * num_tokens * (1 + 2 * num_topk) + 4 * num_channels_total.
#define INTRANODE_CACHED_VALUE_POOL_BYTES(num_ranks) \
    (static_cast<size_t>(1024) * 1024 * (num_ranks) * NUM_MAX_INTRA_CHANNELS)

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500
#define FINISHED_CLEAN_FLAG -1
#define COMPRESSION_DONE_FLAG 1

#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull // 200G cycles ~= 100s
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull // 20G cycles ~= 10s
#endif

#define NOTIFY_DISPATCH_OFFSET 4096
#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900 // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__ // NOLINT(*-reserved-identifier)
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include <cstdint>
#include <tops/tops_bf16.h>
#include <tops_runtime.h>

#ifndef DISABLE_SM90_FEATURES
#include <tops/tops_fp8.h>
#else
// Ampere does not support FP8 features
#define __NV_E4M3 0
#define __NV_E5M2 1
typedef int __nv_fp8_interpretation_t;
typedef int __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;
#endif

#ifndef DISABLE_NVSHMEM
//#include <nvshmem.h>
//#include <nvshmemx.h>
// #include <infiniband/mlx5dv.h>
// #include <non_abi/device/threadgroup/nvshmemi_common_device_defines.h>
// #include <device_host_transport/nvshmem_common_ibgda.h>
#endif

// Hidden-size threshold for intranode_ll dispatch kernel selection.
// hidden <= SLAVE_MODE_HIDDEN_THRESHOLD  → intranode_ll_slave (direct RDMA writes, lower latency)
// hidden >  SLAVE_MODE_HIDDEN_THRESHOLD  → intranode_ll        (ESL QP sends, higher throughput)
// Adjust this value to tune the crossover point.
#define SLAVE_MODE_HIDDEN_THRESHOLD 4096
