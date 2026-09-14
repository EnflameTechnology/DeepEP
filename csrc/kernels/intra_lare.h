// Copyright 2025 Enflame. All Rights Reserved.
#pragma once
#include <tops.h>
#include <krt/topstx.h>
#include <krt/lare.h>
#include <type_traits>
#include "primitive_simple.hpp"
#include "utils.h"
#define MAX_INTRA_DISPATCH_SEQELEM_NUM 256
#define MAX_ROCE_SQELEM_NUM 256
#define CREDIT_DEFAULT_VALUE 0
#define TIMEOUT -1

#ifndef AWUSERTOP_LINEAR_COPY
#define AWUSERTOP_LINEAR_COPY 0
#endif
#ifndef SCF_REG_BASE
#define SCF_REG_BASE 0x100000000ULL
#endif
#ifndef AP_REGS_AP_ETA_PACKET_INFO7_ADDR
#define AP_REGS_AP_ETA_PACKET_INFO7_ADDR (SCF_REG_BASE + 0x2687e38ULL)
#endif
#ifndef OPCODE_WRITE
#define OPCODE_WRITE 2
#endif
#define USE_WQE_TEMPLATE
constexpr int INTRA_LARE_RING_DB_BATCH = 4;
constexpr int INTRA_LARE_FETCH_HEAD_BATCH = 16;
struct lare_wqe {
    __forceinline__ __device__ void init(unsigned int port,
                                         unsigned int qp,
                                         bool fence = true,
                                         bool enableMsi = true) {
        (void)enableMsi;
        op_code = OPCODE_WRITE;
        fence_end = 0;
        enable_msg_interrupt = false;
        enable_nnc = 1;
        fence_start = fence;
        rsv_flags = 0;
        data_type = U32_DTYPE;
        cas = 0;
        scalar_or_vector = 0;
        basic_or_extend = 0;
        awatop = AWUSERTOP_LINEAR_COPY;
        rsv_atomic_userbits = 0;
        msn = 0;
        dst_qp = 0;
        pkt_len = 0;
        src_addr_low = 0;
        src_addr_high = 0;
        dst_addr_low = 0;
        dst_addr_high = 0;
        inline_data0 = 0;
        inline_data1 = 0;
        msi_addr_low = (unsigned int)(AP_REGS_AP_ETA_PACKET_INFO7_ADDR & 0xffffffffULL);
        msi_addr_high = (unsigned int)((AP_REGS_AP_ETA_PACKET_INFO7_ADDR >> 32) & 0xffffffffULL);
        msi_addr_data = (port << 8) + (qp & 0xff);
        rd_resp_psn = 0;
        reserve[0] = 0;
        reserve[1] = 0;
        reserve[2] = 0;
    }

    __forceinline__ __device__ void set_src(void *src) {
        unsigned long long src_u64 = (unsigned long long)src;
        src_addr_low = (unsigned int)(src_u64 & 0xffffffffULL);
        src_addr_high = (unsigned int)((src_u64 >> 32) & 0xffffffffULL);
    }

    __forceinline__ __device__ void set_dst(void *dst) {
        unsigned long long dst_u64 = (unsigned long long)dst;
        dst_addr_low = (unsigned int)(dst_u64 & 0xffffffffULL);
        dst_addr_high = (unsigned int)((dst_u64 >> 32) & 0xffffffffULL);
    }

    union {
        struct {
            unsigned int op_code : 8;
            unsigned int fence_end : 1;
            unsigned int enable_msg_interrupt : 1;
            unsigned int enable_nnc : 1;
            unsigned int fence_start : 1;
            unsigned int rsv_flags : 4;
            unsigned int data_type : 4;
            unsigned int cas : 1;
            unsigned int scalar_or_vector : 1;
            unsigned int basic_or_extend : 1;
            unsigned int awatop : 5;
            unsigned int rsv_atomic_userbits : 4;
            unsigned int msn : 16;
            unsigned int dst_qp : 16;
            unsigned int pkt_len;
            unsigned int src_addr_low;
            unsigned int src_addr_high;
            unsigned int dst_addr_low;
            unsigned int dst_addr_high;
            unsigned int inline_data0;
            unsigned int inline_data1;
            unsigned int msi_addr_low;
            unsigned int msi_addr_high;
            unsigned int msi_addr_data;
            unsigned int rd_resp_psn;
            unsigned int reserve[3];
        };
        unsigned long long data[8];
    };
};

template <int Batch, int fetchBatch, bool WithDte = false>
class intra_lare_engine {
  public:
  __device__ intra_lare_engine() {}
  __device__ ~intra_lare_engine() = default;
  __forceinline__ __device__ void init(struct primitives* prims, int dst_rank, int src_rank, int sm_id, int max_seqelem_num = MAX_ROCE_SQELEM_NUM - 1) {
    max_seqelem_num_ = max_seqelem_num;
    if (dst_rank != src_rank) {
      auto info = prims[sm_id * 2].primitive[dst_rank].lare_dev_info;
      auto curr_slot = tops::lare_get_wqe_slot(info.port_id, info.qp_id);
      lare_dev_info_ = info;
      tailId_ = curr_slot;
      consumer_ptr = tops::lare_get_consumer_addr(info.port_id, info.qp_id);
      headId_ = tailId_;
      dbId_ = tailId_ - 1;
      volatile uint32_t fetchConId = *consumer_ptr;
      conId_ = fetchConId;
#ifdef USE_WQE_TEMPLATE
      wqe_template_.init(info.port_id, info.qp_id, true, true);
#endif
    }
    if constexpr (WithDte) {
      dte_state_.dte.init();
    }
    dstRank_ = dst_rank;
    srcRank_ = src_rank;
  }

  template <typename D>
  __forceinline__ __device__ void normalSend(D* dst, D* src, size_t nelem) {
      if constexpr (WithDte) {
        if (srcRank_ == dstRank_) {
          dteSendAsync(dst, src, nelem);
          return;
        }
      }
      // Cross-rank (always), or same-rank when WithDte=false (caller
      // must not use).
      int curr_idx = getEmptySlot(1);
      write<NON_ATOMIC_OPERATION>(dst, src, curr_idx, nelem);
      ringDb<false>();
      fetchHead<false>();
    }

  template <typename D>
  __forceinline__ __device__ void dteSendAsync(D* dst, D* src, size_t nelem) {
      static_assert(WithDte, "dteSendAsync requires WithDte=true");
      if constexpr (WithDte) {
        if (dte_state_.has_pending_dte_) {
          dte_state_.pending_dte_event_.wait();
          dte_state_.has_pending_dte_ = false;
        }
        tops::mdspan srcL3(tops::Global, src, static_cast<int>(nelem));
        tops::mdspan dstL3(tops::Global, dst, static_cast<int>(nelem));
        dte_state_.pending_dte_event_ =
            tops::memcpy_async(dte_state_.dte, dstL3, srcL3);
        dte_state_.has_pending_dte_ = true;
      }
  }

   __forceinline__ __device__ void dteMayWait() {
      static_assert(WithDte, "dteMayWait requires WithDte=true");
      if constexpr (WithDte) {
        if (dte_state_.has_pending_dte_) {
          dte_state_.pending_dte_event_.wait();
          dte_state_.has_pending_dte_ = false;
        }
      }
  }

  __forceinline__ __device__ int getEmptySlot(int slots) {
    auto start_time = tops::clock64();
    while (true) {
      // max_cap is max_seqelem_num_
      if (tailId_ - headId_ + slots <= max_seqelem_num_) {
        // Enough slot
        break;
      }
      // update headId
      fetchHead<true>();
      if (tops::clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf("DeepEP getEmptySlot timeout\n");
      }
    }
    auto oldTailId = tailId_;
    tailId_ += slots;
    return oldTailId;
  }


  template <atomic_op_t OPS, typename D>
  __forceinline__ __device__ void write(D* dst,
                                        D* src, int idx, size_t nelem) {
#ifdef USE_WQE_TEMPLATE
    wqe_template_.set_dst(reinterpret_cast<void*>(dst));
    wqe_template_.set_src(reinterpret_cast<void*>(src));
    wqe_template_.pkt_len = (unsigned int)(nelem * sizeof(D));
    uint32_t base_wqeId = (idx) & (MAX_ROCE_SQELEM_NUM - 1);
    wqe_template_.msn = base_wqeId;
    __vector long long v_wqe = tcle::load<__vector long long>(&wqe_template_);
    tcle::store(v_wqe, (char*)lare_dev_info_.base_addr + (base_wqeId << 6));
#else
    // Build WQE
    tops::lare_wrinfo wr;
    wr.port = lare_dev_info_.port_id;
    wr.qp = lare_dev_info_.qp_id;
    wr.sq_baseaddr = lare_dev_info_.base_addr;
    wr.sq_nelem = MAX_ROCE_SQELEM_NUM;
    wr.dstAddr = reinterpret_cast<void*>(dst);
    wr.srcAddr = const_cast<D*>(src);
    wr.ops = OPS;
    wr.data_type = getDataType<D>();
    wr.data_nelem = nelem;
    wr.fence = true;
    wr.enableMsi = true;

    // Write WQE to the specified slot (use lower bits for circular buffer index)
    uint32_t base_wqeId = (uint32_t)(idx & (MAX_ROCE_SQELEM_NUM - 1));
    tops::lare_write_wqe(base_wqeId, &wr);
#endif
  }


  template <bool force>
  __forceinline__ __device__ void ringDb() {
    auto readyId_ = tailId_ - 1;
    if constexpr (!force) {
      if (readyId_ % Batch != 0) {
        return;
      }
    }
    if (dbId_ >= readyId_) {
      return;
    }
    // wait for wqe L3 write 
    tcle::fence<tcle::FenceType::L3_MEM_STORE>();
    tops::lare_ring_db(lare_dev_info_.port_id, lare_dev_info_.qp_id, MAX_ROCE_SQELEM_NUM, readyId_, 1);
    dbId_ = readyId_;
  }

  template <bool force>
  __forceinline__ __device__ void fetchHead() {
    auto readyId_ = tailId_ - 1;
    if constexpr (!force) {
      if (readyId_ % (fetchBatch) != 0) {
        return;
      }
    }

    volatile uint32_t newConId = *consumer_ptr;
    //volatile uint32_t newConId = *(tops::lare_get_consumer_addr(lare_dev_info_.port_id, lare_dev_info_.qp_id));
    auto newConsumed = (newConId +  MAX_ROCE_SQELEM_NUM - conId_) & (MAX_ROCE_SQELEM_NUM - 1);
    headId_ = headId_ + newConsumed;
    conId_ = newConId;
  }

  __forceinline__ __device__ void waitTransDone(int idx) {
    uint32_t waitId = (idx) & (MAX_ROCE_SQELEM_NUM - 1);
    (void)tops::lare_wait_done(lare_dev_info_.port_id, lare_dev_info_.qp_id, MAX_ROCE_SQELEM_NUM, waitId, TIMEOUT);
  }
  __forceinline__ __device__ uint32_t tailId() {
    return tailId_;
  }

  __forceinline__ __device__ uint32_t dbId() {
    return dbId_;
  }

  __forceinline__ __device__ uint32_t headId() {
    return headId_;
  }
  private:
  // latest empty idx
  int32_t tailId_;
  // latest ready to fetch idx
  int32_t headId_; 
  // latest idx that already ring
  int32_t dbId_;

  //     head    tail
  //      |       |
  // | | |x|x|x|x| | |
  uint32_t conId_;

  int max_seqelem_num_;
  volatile unsigned int* consumer_ptr;
  struct lare_dev_info lare_dev_info_;
  __valigned__ lare_wqe wqe_template_;
  // private_dte is forbidden in redundant-SIP kernels
  // (privateDdteCount must be 0).
  // Only classic dispatch instantiates WithDte=true.
  struct DteState {
    tops::private_dte dte;
    tops::event pending_dte_event_;
    bool has_pending_dte_ = false;
  };
  struct DteStateEmpty {};
  using DteStorage = typename std::conditional<WithDte, DteState,
                                               DteStateEmpty>::type;
  DteStorage dte_state_;
  int dstRank_;
  int srcRank_;
};
