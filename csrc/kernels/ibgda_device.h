// Portions derived from NVSHMEM (https://developer.nvidia.com/nvshmem)
// Copyright (c) NVIDIA Corporation.
// Licensed under the NVSHMEM Software License Agreement (version: September 3, 2019).
// See full license at: https://docs.nvidia.com/nvshmem/api/sla.html
//
// Modified from original source:
//  - nvshmem/src/include/non_abi/device/pt-to-pt/ibgda_device.h
#pragma once

#include "configs.h"
#include "exception.h"
#include "utils.h"
#include "primitive_simple.hpp"

#include "tops.h"
#include <krt/topstx.h>
#include <krt/lare.h>

#include "devcomm.h"

#define ibgdaTRACE(...)

#ifdef ENABLE_MORI_GCU

#define IBGDA_MR_ACCESS_FLAG  (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC)
#define IBGDA_QP2ID(_rr, _qp) (((_rr)*IBGDA_NUM_QPS) + ((_qp) % IBGDA_NUM_QPS))
#define IBGDA_NUM_QPS         (MAX_IBGDA_QP_NUMS)
#define IBGDA_GID_IDX         (3)
#define IBGDA_WQE_NUM         (2048)
#define IBGDA_ALIGNMENT       (4096)
#define IBGDA_ON_GCU          (true)
#define IBGDA_BNXT_CQE_NUM    (1)
#define IBGDA_CQE_STEP        (IBGDA_WQE_NUM/2)
#define IBGDA_CQE_NUM         (8)
// Note: IBGDA_MR_OFFSET and IBGDA_EP_OFFSET are now defined in configs.h

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/udma_barrier.h"
#include "mori/core/core.hpp"


using namespace mori;
using namespace mori::application;
using namespace mori::core;

static_assert(MAX_IBGDA_EP_SIZE == sizeof(RdmaEndpoint), "MAX_IBGDA_EP_SIZE too small than RdmaEndpoint");
static_assert(MAX_IBGDA_MR_SIZE == sizeof(RdmaMemoryRegion), "MAX_IBGDA_MR_SIZE too small than RdmaMemoryRegion");

static_assert(4 == sizeof(RdmaDeviceVendorId) &&       // 4
              40 == sizeof(RdmaEndpointHandle) &&    // 40
              56 == sizeof(CompletionQueueHandle) && // 56
              32 == sizeof(IBVerbsHandle) &&         // 32
              32 == sizeof(IbufHandle) &&            // 32
              24 == sizeof(RdmaMemoryRegion),        // 24
              "Mismatched size of IBGDA");

#endif


namespace deep_ep {

#ifdef ENABLE_MORI_GCU

#if (IBGDA_CQE_STEP == 0 || (IBGDA_CQE_STEP & (IBGDA_CQE_STEP - 1)) != 0)
#error IBGDA_CQE_STEP:  NOT Power of Two !!
#endif

namespace deep_ep_ibgda {


static_assert((IBGDA_CQE_STEP * IBGDA_CQE_NUM) > (IBGDA_WQE_NUM), "Ensure CQE_NUM x STEP to cover WQE_NUM");

template <ProviderType P>
__forceinline__ __device__ void ibgdaWrite(RdmaEndpoint &endPoint, RdmaMemoryRegion &sendMr, RdmaMemoryRegion &recvMr, size_t sendOffset,
                                          size_t recvOffset, size_t szToWrite) {
  uint32_t curPostIdx = tcle::atomic_add(&endPoint.wqHandle.postIdx, 1);

  uintptr_t laddr = sendMr.addr + sendOffset;
  uintptr_t raddr = recvMr.addr + recvOffset;
  bool cqeSignal  = (0 == (curPostIdx & (IBGDA_CQE_STEP - 1)));

  uint64_t dbr_val = PostWrite<P>(endPoint.wqHandle, curPostIdx, curPostIdx, curPostIdx, cqeSignal,
                                  endPoint.handle.qpn, laddr, sendMr.lkey, raddr, recvMr.rkey, szToWrite);
  ibgdaTRACE("PostWrite is done\n");
  ibgdaTRACE("%s %d, local %lx %x, remote %lx %x, [%ld] B\n", __FUNCTION__, __LINE__, laddr, sendMr.lkey, raddr, recvMr.rkey,
              szToWrite);

  // Wait for our turn to write
  volatile uint32_t* volatile dbTouchPtr = reinterpret_cast<volatile uint32_t* volatile>(&endPoint.wqHandle.dbTouchIdx);
  uint32_t db_touched = *dbTouchPtr;
  ibgdaTRACE("Wait for our turn: db_touched: %u, curPostIdx: %u, qpn %d\n", db_touched, curPostIdx, endPoint.handle.qpn);
  while (db_touched != curPostIdx) db_touched = *dbTouchPtr;

  ++curPostIdx;

  UpdateSendDbrRecord<P>(endPoint.wqHandle.dbrRecAddr, curPostIdx);
  //UpdateSendDbrRecord<P>(endPoint.wqHandle.dbrRecAddr, endPoint.wqHandle.postIdx);
  ibgdaTRACE("UpdateSendDbrRecord is done qpn %d\n", endPoint.handle.qpn);

  // RingDoorbell write to L3 too
  RingDoorbell<P>(endPoint.wqHandle.dbrAddr, dbr_val);

  __gcu_get_spr_barrier(BARRIER_MASK_L3);
  //tcle::fence<0xD8>();
  ibgdaTRACE("RingDoorbell is done%s.\n", (cqeSignal)?", PollCq Required":"");

  if (false != cqeSignal) {
    uint16_t wqeIdx;
    int snd_opcode = PollCq<P>(endPoint.cqHandle.cqAddr, endPoint.cqHandle.cqeNum, &endPoint.cqHandle.consIdx, &wqeIdx);
    static_cast<void>(snd_opcode);
    tcle::atomic_add(&endPoint.cqHandle.consIdx, 1);
    ibgdaTRACE("send PollCq is done with snd_opcode %d, wqeIdx: %hu, consIdx %d, cqeNum %d\n", snd_opcode, wqeIdx,
            endPoint.cqHandle.consIdx, endPoint.cqHandle.cqeNum);
    UpdateCqDbrRecord<P>(endPoint.cqHandle.dbrRecAddr, endPoint.cqHandle.consIdx, endPoint.cqHandle.cqeNum);
    ibgdaTRACE("send UpdateCqDbrRecord is done\n");
  }

  ibgdaTRACE("Done for our turn dbTouchIdx %u, postIdx %u, curPostIdx %u, qpn %d\n", endPoint.wqHandle.dbTouchIdx, endPoint.wqHandle.postIdx, curPostIdx, endPoint.handle.qpn);
  tcle::atomic_add(reinterpret_cast<void *>(&endPoint.wqHandle.dbTouchIdx), 1);
}

template <ProviderType P>
__forceinline__ __device__ uint64_t ibgdaPrepTransData(RdmaEndpoint &endPoint, RdmaMemoryRegion &sendMr, RdmaMemoryRegion &recvMr, size_t sendOffset,
                                          size_t recvOffset, size_t szToWrite, uint32_t &curPostIdx) {
  curPostIdx = tcle::atomic_add(&endPoint.wqHandle.postIdx, 1);

  uintptr_t laddr = sendMr.addr + sendOffset;
  uintptr_t raddr = recvMr.addr + recvOffset;
  bool cqeSignal  = 0;

  uint64_t dbr_val = PostWrite<P>(endPoint.wqHandle, curPostIdx, curPostIdx, curPostIdx, cqeSignal,
                                  endPoint.handle.qpn, laddr, sendMr.lkey, raddr, recvMr.rkey, szToWrite);
  ibgdaTRACE("PostWrite is done\n");
  ibgdaTRACE("%s %d, local %lx %x, remote %lx %x, [%ld] B\n", __FUNCTION__, __LINE__, laddr, sendMr.lkey, raddr, recvMr.rkey,
              szToWrite);

  return dbr_val;
}

template <ProviderType P>
__forceinline__ __device__ void ibgdaRingDb(RdmaEndpoint &endPoint, uint32_t &curPostIdx, const uint64_t dbr_val) {
  // Wait for our turn to write
  volatile uint32_t* volatile dbTouchPtr = reinterpret_cast<volatile uint32_t* volatile>(&endPoint.wqHandle.dbTouchIdx);
  uint32_t db_touched = *dbTouchPtr;
  ibgdaTRACE("Wait for our turn: db_touched: %u, curPostIdx: %u, qpn %d\n", db_touched, curPostIdx, endPoint.handle.qpn);
  while (db_touched != curPostIdx) db_touched = *dbTouchPtr;

  ++curPostIdx;

  UpdateSendDbrRecord<P>(endPoint.wqHandle.dbrRecAddr, curPostIdx);
  tcle::fence<tcle::L3_MEM_STORE>();
  ibgdaTRACE("UpdateSendDbrRecord is done qpn %d\n", endPoint.handle.qpn);

  // RingDoorbell write to L3 too
  RingDoorbell<P>(endPoint.wqHandle.dbrAddr, dbr_val);

  ibgdaTRACE("Done for our turn dbTouchIdx %u, postIdx %u, curPostIdx %u, qpn %d\n", endPoint.wqHandle.dbTouchIdx, endPoint.wqHandle.postIdx, curPostIdx, endPoint.handle.qpn);
  tcle::atomic_add(reinterpret_cast<void *>(&endPoint.wqHandle.dbTouchIdx), 1);
}

template <ProviderType P>
__forceinline__ __device__ void ibgdaPollCq(RdmaEndpoint &endPoint, const uint32_t curPostIdx) {
  (void)endPoint, (void)curPostIdx;
}

template <ProviderType P, typename F>
__forceinline__ __device__ void ibgdaSendFlag(
  RdmaEndpoint &endPoint, RdmaMemoryRegion &sendMr, RdmaMemoryRegion &recvMr, const F *src_flag, F *dst_flag) {
  size_t srcOffs = (size_t)(src_flag) - sendMr.addr;
  size_t dstOffs = (size_t)(dst_flag) - recvMr.addr;
  size_t szToWrite = sizeof(F);

  if (srcOffs >= sendMr.length || srcOffs + szToWrite > sendMr.length)
    printf("%s %d, OOR: srcOffs %ld from %lx, OOR %ld, [%ld]\n", __FUNCTION__, __LINE__, srcOffs, (size_t)(src_flag), szToWrite, sendMr.length);

  if (dstOffs >= recvMr.length || dstOffs + szToWrite > recvMr.length)
    printf("%s %d, OOR: dstOffs %ld from %lx, OOR %ld, [%ld]\n", __FUNCTION__, __LINE__, dstOffs, (size_t)(dst_flag), szToWrite, recvMr.length);

  ibgdaWrite<P>(endPoint, sendMr, recvMr, srcOffs, dstOffs, szToWrite);
}

class ibgda_engine {
public:
  __device__ ibgda_engine() : endPoint_(nullptr) {}
  __device__ ~ibgda_engine() = default;

  __forceinline__ __device__ void load(const uintptr_t ibgda_info_ptr,
                                       const int qpId, const int nQps,
                                       const int lRank, const int rRank) {
    RdmaMemoryRegion *mrhandles = reinterpret_cast<RdmaMemoryRegion *>(ibgda_info_ptr + IBGDA_MR_OFFSET);
    // Note: The endpoints array is unified_endpoint array (384 bytes each, aligned)
    // For inter-node communication, ibgdaSetup has written full RdmaEndpoint (264B)
    // The ibgda_storage in unified_endpoint has the same memory layout as RdmaEndpoint
    // We use unified_endpoint* for correct pointer arithmetic (384-byte stride)
    unified_endpoint *endpoints = reinterpret_cast<unified_endpoint *>(ibgda_info_ptr + IBGDA_EP_OFFSET);

    assert(ibgda_info_ptr && "Ibgda Info Ptr OOR");
    assert(qpId < nQps && "qpId OOR");
    assert(rRank != lRank && "Rank mismtached");

    // for RdmaMemoryRegion, SEND using lRank, RECV using rRank;
    //mrHdls_[SEND] = mrhandles[lRank];
    //mrHdls_[RECV] = mrhandles[rRank];
    transferGlobaltoStackUnaligned((T_U64*)(mrHdls_ + SEND), mrhandles + lRank);
    transferGlobaltoStackUnaligned((T_U64*)(mrHdls_ + RECV), mrhandles + rRank);

    ibgdaTRACE("%s %d ibgda_info_ptr 0x%lX q[%d/%d] lR[%d] rR[%d], lMR: %lx key[%x/%x] len=%ld, rMR %lx key[%x/%x] len=%ld\n",
           __FUNCTION__, __LINE__,
           ibgda_info_ptr, qpId, nQps, lRank, rRank,
           mrHdls_[SEND].addr, mrHdls_[SEND].lkey, mrHdls_[SEND].rkey, mrHdls_[SEND].length,
           mrHdls_[RECV].addr, mrHdls_[RECV].lkey, mrHdls_[RECV].rkey, mrHdls_[RECV].length);

    // for RdmaEndpoint, local endPoint_ using rRank;
    // Index logic: endpoints[rRank * nQps + qpId] with 384-byte stride
    // This is now equivalent to GET_UNIFIED_EP_OFFSET(rRank, qpId)
    // Cast ibgda_storage to RdmaEndpoint* to access RDMA fields
    endPoint_ = reinterpret_cast<RdmaEndpoint*>(&endpoints[rRank * nQps + qpId].ibgda_storage);
    provider_ = endPoint_->GetProviderType();
    assert(provider_ != ProviderType::Unknown);
    if (provider_ == ProviderType::Unknown) {
        printf("%s %d OOR, Unknown ProviderType, ibgda_info_ptr 0x%lX q[%d/%d] lR[%d] rR[%d], lMR: %lx key[%x/%x] len=%ld, rMR %lx key[%x/%x] len=%ld\n",
            __FUNCTION__, __LINE__,
           ibgda_info_ptr, qpId, nQps, lRank, rRank,
           mrHdls_[SEND].addr, mrHdls_[SEND].lkey, mrHdls_[SEND].rkey, mrHdls_[SEND].length,
           mrHdls_[RECV].addr, mrHdls_[RECV].lkey, mrHdls_[RECV].rkey, mrHdls_[RECV].length);
    }
  }

  template <typename T>
  __forceinline__ __device__ bool send(T *dst, T *src, size_t nelem) {
    uint64_t srcOffs = (uint64_t)src - mrHdls_[SEND].addr;
    uint64_t dstOffs = (uint64_t)dst - mrHdls_[RECV].addr;
    size_t nBytes = nelem * sizeof(T);

    if (srcOffs + nBytes < mrHdls_[SEND].length && dstOffs + nBytes < mrHdls_[RECV].length) {
      if (provider_ == ProviderType::MLX5) {
        deep_ep_ibgda::ibgdaWrite<ProviderType::MLX5>(*endPoint_, mrHdls_[SEND], mrHdls_[RECV], srcOffs, dstOffs, nBytes);
      } if (provider_ == ProviderType::BNXT) {
        deep_ep_ibgda::ibgdaWrite<ProviderType::BNXT>(*endPoint_, mrHdls_[SEND], mrHdls_[RECV], srcOffs, dstOffs, nBytes);
      }
    } else {
      printf("%s %d OOR, SA 0x%lX DA 0x%lx nelem=%ld,"
             "lMR: %lx key[%x/%x] len=%ld offset=%ld, "
             "rMR: %lx key[%x/%x] len=%ld offset=%ld\n",
             __FUNCTION__, __LINE__, (uint64_t)src, (uint64_t)dst, nelem,
             mrHdls_[SEND].addr, mrHdls_[SEND].lkey, mrHdls_[SEND].rkey, mrHdls_[SEND].length, srcOffs,
             mrHdls_[RECV].addr, mrHdls_[RECV].lkey, mrHdls_[RECV].rkey, mrHdls_[RECV].length, dstOffs);
      return false;
    }

    return true;
  }

  template <typename T>
  __forceinline__ __device__ bool prepTransData(T *dst, T *src, size_t nelem) {
    uint64_t srcOffs = (uint64_t)src - mrHdls_[SEND].addr;
    uint64_t dstOffs = (uint64_t)dst - mrHdls_[RECV].addr;
    size_t nBytes = nelem * sizeof(T);

    if (srcOffs + nBytes >= mrHdls_[SEND].length || dstOffs + nBytes >= mrHdls_[RECV].length) {
      printf("%s %d OOR ProviderType %d, SA 0x%lX DA 0x%lx nelem=%ld,"
             "lMR: %lx key[%x/%x] len=%ld offset=%ld, "
             "rMR: %lx key[%x/%x] len=%ld offset=%ld\n",
             __FUNCTION__, __LINE__,
             provider_, (uint64_t)src, (uint64_t)dst, nelem,
             mrHdls_[SEND].addr, mrHdls_[SEND].lkey, mrHdls_[SEND].rkey, mrHdls_[SEND].length, srcOffs,
             mrHdls_[RECV].addr, mrHdls_[RECV].lkey, mrHdls_[RECV].rkey, mrHdls_[RECV].length, dstOffs);
      return false;
    }

    if (provider_ == ProviderType::MLX5) {
      dbr_val_ = deep_ep_ibgda::ibgdaPrepTransData<ProviderType::MLX5>(*endPoint_, mrHdls_[SEND], mrHdls_[RECV], srcOffs, dstOffs, nBytes, postIdx_);
    } else if (provider_ == ProviderType::BNXT) {
      dbr_val_ = deep_ep_ibgda::ibgdaPrepTransData<ProviderType::BNXT>(*endPoint_, mrHdls_[SEND], mrHdls_[RECV], srcOffs, dstOffs, nBytes, postIdx_);
    }

    return true;
  }

  __forceinline__ __device__ void ringDb() {
    if (provider_ == ProviderType::MLX5) {
      deep_ep_ibgda::ibgdaRingDb<ProviderType::MLX5>(*endPoint_, postIdx_, dbr_val_);
    } else if (provider_ == ProviderType::BNXT) {
      deep_ep_ibgda::ibgdaRingDb<ProviderType::BNXT>(*endPoint_, postIdx_, dbr_val_);
    }
  }

  __forceinline__ __device__ void waitTransDone() {
    if (provider_ == ProviderType::MLX5) {
      deep_ep_ibgda::ibgdaPollCq<ProviderType::MLX5>(*endPoint_, postIdx_);
    } else if (provider_ == ProviderType::BNXT) {
      deep_ep_ibgda::ibgdaPollCq<ProviderType::BNXT>(*endPoint_, postIdx_);
    }
  }

private:
  ProviderType provider_{ProviderType::Unknown};
  RdmaMemoryRegion mrHdls_[MAX_SIDES];
  RdmaEndpoint *endPoint_;
  uint64_t dbr_val_;
  uint32_t postIdx_;
};

} // namespace deep_ep_ibgda


#endif

#ifdef DEEP_EP_NV_IBGDA

EP_STATIC_ASSERT(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64, "Invalid QP minimum depth");

__device__ static __forceinline__
uint64_t HtoBE64(uint64_t x) {
    uint64_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 lo;\n\t"
        ".reg .b32 hi;\n\t"
        ".reg .b32 new_lo;\n\t"
        ".reg .b32 new_hi;\n\t"
        "mov.b64 {lo,hi}, %1;\n\t"
        "prmt.b32 new_hi, lo, ign, 0x0123;\n\t"
        "prmt.b32 new_lo, hi, ign, 0x0123;\n\t"
        "mov.b64 %0, {new_lo,new_hi};\n\t"
        "}" : "=l"(ret) : "l"(x));
    return ret;
}

__device__ static __forceinline__
uint32_t HtoBE32(uint32_t x) {
    uint32_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        "prmt.b32 %0, %1, ign, 0x0123;\n\t"
        "}" : "=r"(ret) : "r"(x));
    return ret;
}

__device__ static __forceinline__
uint16_t HtoBE16(uint16_t x) {
    // TODO: simplify PTX using 16-bit instructions
    auto a = static_cast<uint32_t>(x);
    uint32_t d;
    // asm volatile(
    //     "{\n\t"
    //     ".reg .b32 mask;\n\t"
    //     ".reg .b32 ign;\n\t"
    //     "mov.b32 mask, 0x4401;\n\t"
    //     "mov.b32 ign, 0x0;\n\t"
    //     "prmt.b32 %0, %1, ign, mask;\n\t"
    //     "}"
    //     : "=r"(d)
    //     : "r"(a));
    return static_cast<uint16_t>(d);
}

typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) ibgda_ctrl_seg_t;

typedef struct {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;

__device__ static __forceinline__
nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;
}

__device__ static __forceinline__
nvshmemi_ibgda_device_qp_t* ibgda_get_rc(int pe, int id) {
    auto state = ibgda_get_state();
    const auto num_rc_per_pe = ibgda_get_state()->num_rc_per_pe;
    return &state->globalmem.rcs[pe * num_rc_per_pe * state->num_devices_initialized + id % (num_rc_per_pe * state->num_devices_initialized)];
}

__device__ static __forceinline__
void ibgda_lock_acquire(int *lock) {
    while (atomicCAS(lock, 0, 1) == 1);

    // Prevent reordering before the lock is acquired
    memory_fence_cta();
}

__device__ static __forceinline__
void ibgda_lock_release(int *lock) {
    memory_fence_cta();

    // Prevent reordering before lock is released
    st_na_relaxed(lock, 0);
}

__device__ static __forceinline__
void ibgda_update_dbr(nvshmemi_ibgda_device_qp_t *qp, uint32_t dbrec_head) {
    // `DBREC` contains the index of the next empty `WQEBB`
    __be32 dbrec_val;
    __be32 *dbrec_ptr = qp->tx_wq.dbrec;

    // This is equivalent to `WRITE_ONCE(dbrec_ptr, HtoBE32(dbrec_head & 0xffff))`
    asm("{\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"
        ".reg .b32 ign;\n\t"
        "and.b32 dbrec_head_16b, %1, 0xffff;\n\t"
        "prmt.b32 %0, dbrec_head_16b, ign, 0x123;\n\t"
        "}"
        : "=r"(dbrec_val)
        : "r"(dbrec_head));
    st_na_release(dbrec_ptr, dbrec_val);
}

__device__ static __forceinline__
void ibgda_ring_db(nvshmemi_ibgda_device_qp_t *qp, uint16_t prod_idx) {
    auto bf_ptr = reinterpret_cast<uint64_t*>(qp->tx_wq.bf);
    ibgda_ctrl_seg_t ctrl_seg = {
        .opmod_idx_opcode = HtoBE32(prod_idx << 8),
        .qpn_ds = HtoBE32(qp->qpn << 8)
    };

    EP_STATIC_ASSERT(sizeof(decltype(&ctrl_seg)) == sizeof(uint64_t), "");
    st_na_release(bf_ptr, *(reinterpret_cast<uint64_t*>(&ctrl_seg)));
}

__device__ static __forceinline__
void ibgda_post_send(nvshmemi_ibgda_device_qp_t *qp, uint64_t new_prod_idx) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t old_prod_idx;

    // Update `prod_idx` before ringing the doorbell, so that we know which index is needed in quiet/fence
    ibgda_lock_acquire(&mvars->post_send_lock);

    old_prod_idx = atomicMax(reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.prod_idx), new_prod_idx);
    if (new_prod_idx > old_prod_idx) {
        ibgda_update_dbr(qp, new_prod_idx);
        ibgda_ring_db(qp, new_prod_idx);
    }
    ibgda_lock_release(&mvars->post_send_lock);
}

template <bool kAlwaysDoPostSend>
__device__ static __forceinline__
void ibgda_submit_requests(nvshmemi_ibgda_device_qp_t *qp, uint64_t base_wqe_idx,
                           uint32_t num_wqes, int message_idx = 0) {
    auto state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

    // WQE writes must be finished first
    __threadfence();

    unsigned long long int *ready_idx =
        (unsigned long long int *)(state->use_async_postsend ? qp->tx_wq.prod_idx
                                                             : &mvars->tx_wq.ready_head);

    // Wait for prior WQE slots to be filled first
    while (atomicCAS(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx);

    // Always post, not in batch
    if (!state->use_async_postsend) {
        constexpr int kNumRequestInBatch = 4;
        if (kAlwaysDoPostSend or (message_idx + 1) % kNumRequestInBatch == 0)
            ibgda_post_send(qp, new_wqe_idx);
    }
}

__device__ static __forceinline__ void
ibgda_write_rdma_write_inl_wqe(nvshmemi_ibgda_device_qp_t *qp, const uint32_t *val, uint64_t raddr,
                               __be32 rkey, uint16_t wqe_idx, void** out_wqes, uint32_t imm) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    auto *ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    auto *raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    auto *inl_seg_ptr = reinterpret_cast<mlx5_wqe_inl_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
    auto *wqe_data_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(inl_seg_ptr) + sizeof(*inl_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HtoBE32(4 | MLX5_INLINE_SEG);

    // `imm == std::numeric_limits<uint32_t>::max()` means no imm writes
    ctrl_seg = {0};
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | (imm != std::numeric_limits<uint32_t>::max() ? MLX5_OPCODE_RDMA_WRITE_IMM : MLX5_OPCODE_RDMA_WRITE));
    if (imm != std::numeric_limits<uint32_t>::max())
        ctrl_seg.imm = HtoBE32(imm);

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*inl_seg_ptr) == 4, "sizeof(*inl_seg_ptr) == 4");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<uint32_t*>(inl_seg_ptr), *reinterpret_cast<const uint32_t*>(&inl_seg));
    st_na_relaxed(reinterpret_cast<uint32_t*>(wqe_data_ptr), *reinterpret_cast<const uint32_t*>(val));
}

__device__ static __forceinline__
uint64_t ibgda_get_lkey_and_rkey(uint64_t laddr, __be32 *lkey,
                                 uint64_t raddr, int dst_pe, uint64_t *out_raddr, __be32 *out_rkey, uint32_t dev_idx) {
    auto state = ibgda_get_state();
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
    auto log2_cumem_granularity = state->log2_cumem_granularity;

    // Local key
    uint64_t idx = ((laddr - heap_start) >> log2_cumem_granularity) * state->num_devices_initialized + dev_idx;
    auto device_key = state->constmem.lkeys[idx];
    auto lchunk_size = device_key.next_addr - laddr;
    *lkey = device_key.key;

    // Remote key
    uint64_t roffset = raddr - heap_start;

    idx = ((roffset >> log2_cumem_granularity) * nvshmemi_device_state_d.npes) * state->num_devices_initialized
          + dst_pe * state->num_devices_initialized + dev_idx;
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {
        device_key = state->constmem.rkeys[idx];
    } else {
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    }
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;

    // Return the minimum of local and remote chunk sizes
    auto rchunk_size = device_key.next_addr - roffset;
    return min(lchunk_size, rchunk_size);
}

__device__ static __forceinline__ void
ibgda_get_rkey(uint64_t addr, int dst_pe, uint64_t *out_raddr, __be32 *out_rkey, uint32_t dev_idx) {
    auto state = ibgda_get_state();
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);

    uint64_t roffset = addr - heap_start;
    uint64_t idx = ((roffset >> state->log2_cumem_granularity) * nvshmemi_device_state_d.npes * state->num_devices_initialized)
                   + dst_pe * state->num_devices_initialized + dev_idx;
    nvshmemi_ibgda_device_key_t device_key;
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS)
        device_key = state->constmem.rkeys[idx];
    else
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;
}

__device__ static __forceinline__ uint64_t
ibgda_reserve_wqe_slots(nvshmemi_ibgda_device_qp_t *qp, uint32_t num_wqes) {
    auto mvars = &qp->mvars;
    return atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.resv_head), static_cast<unsigned long long>(num_wqes));
}

__device__ static __forceinline__ void*
ibgda_get_wqe_ptr(nvshmemi_ibgda_device_qp_t* qp, uint16_t wqe_idx) {
    uint16_t cnt = qp->tx_wq.nwqes;
    uint16_t idx = wqe_idx & (cnt - 1);
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(qp->tx_wq.wqe) + (idx << MLX5_SEND_WQE_SHIFT));
}

__device__ static __forceinline__ void
nvshmemi_ibgda_rma_p(int *rptr, const int value, int dst_pe, int qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max()) {
    // Get rkey
    // NOTES: the `p` operation will not cross multiple remote chunks
    __be32 rkey;
    uint64_t raddr;
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), dst_pe, &raddr, &rkey, qp->dev_idx);

    // Write WQEs
    uint64_t base_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
    void *wqe_ptrs;
    wqe_ptrs = ibgda_get_wqe_ptr(qp, base_wqe_idx);
    ibgda_write_rdma_write_inl_wqe(qp, reinterpret_cast<const uint32_t*>(&value), raddr, rkey, base_wqe_idx, &wqe_ptrs, imm);

    // Submit requests
    ibgda_submit_requests<true>(qp, base_wqe_idx, 1);
}

__device__ static __forceinline__ void
ibgda_write_rdma_write_wqe(nvshmemi_ibgda_device_qp_t *qp, uint64_t laddr, __be32 lkey,
                           uint64_t raddr, __be32 rkey, uint32_t bytes, uint16_t wqe_idx,
                           void** out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_data_seg data_seg;

    auto *ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    void *av_seg_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_data_seg *data_seg_ptr;

    raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(av_seg_ptr));
    data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    data_seg.byte_count = HtoBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HtoBE64(laddr);

    ctrl_seg = {0};
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == 16, "sizeof(*data_seg_ptr) == 16");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
}

__device__ static __forceinline__ void
ibgda_write_empty_recv_wqe(void *out_wqe) {
    auto *data_seg_ptr = reinterpret_cast<struct mlx5_wqe_data_seg*>(out_wqe);
    struct mlx5_wqe_data_seg data_seg;

    // Make the first segment in the WQE invalid, then the entire list will be invalid
    data_seg.byte_count = 0;
    data_seg.lkey = HtoBE64(MLX5_INVALID_LKEY);
    data_seg.addr = 0;

    EP_STATIC_ASSERT(sizeof(mlx5_wqe_data_seg) == sizeof(int4), "Invalid data type length");
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
}

template <bool kAlwaysDoPostSend = false>
__device__ static __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id, int lane_id, int message_idx) {
    // Get lkey and rkey, store them into lanes
    uint32_t num_wqes = 0;
    __be32 my_lkey = 0;
    uint64_t my_laddr = 0;
    __be32 my_rkey = 0;
    uint64_t my_raddr = 0;
    uint64_t my_chunk_size = 0;

    auto qp = ibgda_get_rc(dst_pe, qp_id);

    // Decide how many messages (theoretically 3 for maximum)
    auto remaining_bytes = bytes;
    while (remaining_bytes > 0) {
        if (lane_id == num_wqes) {
            my_chunk_size = min(remaining_bytes,
                                ibgda_get_lkey_and_rkey(my_laddr = req_lptr,
                                                        &my_lkey,
                                                        req_rptr,
                                                        dst_pe,
                                                        &my_raddr,
                                                        &my_rkey,
                                                        qp->dev_idx));
        }

        // Move one more message
        auto chunk_size = __shfl_sync(0xffffffff, my_chunk_size, static_cast<int>(num_wqes));
        remaining_bytes -= chunk_size;
        req_lptr += chunk_size;
        req_rptr += chunk_size;
        ++ num_wqes;
    }
    EP_DEVICE_ASSERT(num_wqes <= 32);

    // Process WQE
    uint64_t base_wqe_idx = 0;
    if (lane_id == 0)
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes);
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
    if (lane_id < num_wqes) {
        auto wqe_idx = base_wqe_idx + lane_id;
        auto wqe_ptr = ibgda_get_wqe_ptr(qp, wqe_idx);
        ibgda_write_rdma_write_wqe(qp, my_laddr, my_lkey, my_raddr, my_rkey, my_chunk_size,
                                   wqe_idx, &wqe_ptr);
    }
    __syncwarp();

    // Submit
    if (lane_id == 0)
        ibgda_submit_requests<kAlwaysDoPostSend>(qp, base_wqe_idx, num_wqes, message_idx);
    __syncwarp();
}

__device__ static __forceinline__ void ibgda_write_amo_add_wqe(
        nvshmemi_ibgda_device_qp_t *qp, const int &value,
        uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey,
        uint16_t wqe_idx, void** out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg = {0};
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_atomic_seg atomic_seg_1;
    struct mlx5_wqe_data_seg data_seg;

    auto ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    auto raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    auto atomic_seg_ptr = reinterpret_cast<mlx5_wqe_atomic_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
    auto data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(atomic_seg_ptr) + sizeof(*atomic_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    // NOTES: `0x08000000` means `IBGDA_4_BYTE_EXT_AMO_OPMOD`
    ctrl_seg.opmod_idx_opcode = HtoBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | 0x08000000);
    auto atomic_32_masked_fa_seg = reinterpret_cast<ibgda_atomic_32_masked_fa_seg_t*>(&atomic_seg_1);
    atomic_32_masked_fa_seg->add_data = HtoBE32(value);
    atomic_32_masked_fa_seg->field_boundary = 0;

    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 4);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

    data_seg.byte_count = HtoBE32(sizeof(int));
    data_seg.lkey = lkey;
    data_seg.addr = HtoBE64(laddr);

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == sizeof(int4), "Invalid vectorization");
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == sizeof(int4), "Invalid vectorization");
    EP_STATIC_ASSERT(sizeof(*atomic_seg_ptr) == sizeof(int4), "Invalid vectorization");
    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == sizeof(int4), "Invalid vectorization");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<int4*>(atomic_seg_ptr), *reinterpret_cast<int4*>(&atomic_seg_1));
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<int4*>(&data_seg));
}

__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    if (is_local_copy) {
        atomicAdd(static_cast<unsigned long long*>(rptr), value);
    } else {
        nvshmemi_ibgda_device_qp_t *qp = ibgda_get_rc(pe, qp_id);

        __be32 rkey;
        uint64_t raddr;
        ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), pe, &raddr, &rkey, qp->dev_idx);

        uint64_t my_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
        void *wqe_ptrs = ibgda_get_wqe_ptr(qp, my_wqe_idx);

        ibgda_write_amo_add_wqe(qp, value, reinterpret_cast<uint64_t>(qp->ibuf.buf),
                                qp->ibuf.lkey, raddr, rkey, my_wqe_idx, &wqe_ptrs);

        ibgda_submit_requests<true>(qp, my_wqe_idx, 1);
    }
}

__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // Local rank, no need for mapping
    if (rank == dst_rank)
        return ptr;
    auto peer_base = __ldg(reinterpret_cast<uint64_t*>(nvshmemi_device_state_d.peer_heap_base_p2p) + dst_rank);

    // RDMA connected
    if (peer_base == 0)
        return 0;

    // NVLink P2P is enabled
    return peer_base + (ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base));
}

// This is a simplified version of NVSHMEM's `ibgda_poll_cq`.
// Note that this implementation does not guarantee thread safety,
// so we must ensure that no other threads are concurrently using the same QP.
__device__ static __forceinline__ void
ibgda_poll_cq(nvshmemi_ibgda_device_cq_t *cq, uint64_t idx) {
    const auto cqe64 = static_cast<mlx5_cqe64*>(cq->cqe);
    const uint32_t ncqes = cq->ncqes;
    memory_fence_cta();
    if (*cq->cons_idx >= idx) return;
    // NOTES: this while loop is part of do-while below.
    // `wqe_counter` is the HW consumer index. However, we always maintain `index + 1`.
    // To be able to compare with the index, we need to use `wqe_counter + 1`.
    // Because `wqe_counter` is `uint16_t`, it may be overflow. Still, we know for
    // sure that if `idx - wqe_counter - 1 < ncqes`, `wqe_counter + 1 is less than
    // idx, and thus we need to wait. We don't need to wait when `idx == wqe_counter + 1`
    // That's why we use `- 2` here to make this case overflow.
    uint16_t wqe_counter;
    do {
        wqe_counter = HtoBE16(ld_na_relaxed(&cqe64->wqe_counter));
    } while ((static_cast<uint16_t>(static_cast<uint16_t>(idx) - wqe_counter - static_cast<uint16_t>(2)) < ncqes));
    *cq->cons_idx = idx;

    // Prevent reordering of this function and later instructions
    memory_fence_cta();
}

// Wait until wqe `idx - 1` is completed.
__device__ static __forceinline__ void
nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    auto state = ibgda_get_state();
    uint64_t prod_idx = state->use_async_postsend ? ld_na_relaxed(qp->tx_wq.prod_idx) : ld_na_relaxed(&qp->mvars.tx_wq.ready_head);
    ibgda_poll_cq(qp->tx_wq.cq, prod_idx);
}

#endif

} // namespace deep_ep
