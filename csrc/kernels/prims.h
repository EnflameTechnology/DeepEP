// Copyright 2025 Enflame. All Rights Reserved.
#pragma once
#include "tops.h"
#include <krt/topstx.h>
#include <krt/lare.h>
#include "primitive_simple.hpp"

#define MAX_ROCE_SQELEM_NUM 256
#define CREDIT_DEFAULT_VALUE 0
#define TIMEOUT -1

// NOTE: Do NOT provide a default mapping here.
// If a new type is used without an explicit specialization, we want a compile-time error
// instead of silently falling back to an incorrect datatype (e.g. `char*` -> TF32_DTYPE).
template <typename D>
__forceinline__ __device__ constexpr enum datatype_t getDataType() = delete;
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<int8_t>() {
  return S8_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<uint8_t>() {
  return U8_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<int16_t>() {
  return S16_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<uint16_t>() {
  return U16_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<int32_t>() {
  return S32_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<uint32_t>() {
  return U32_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<int64_t>() {
  return S64_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<uint64_t>() {
  return U64_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<vhalf>() {
  return F8_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<__fp16>() {
  return F16_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<tops::__ef_bfloat16>() {
  return BF16_DTYPE;
}

template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<float>() {
  return F32_DTYPE;
}
template <>
__forceinline__ __device__ constexpr enum datatype_t getDataType<double>() {
  return F64_DTYPE;
}

class lare_engine {
  public:
  __device__ lare_engine():
      waitId_(0),
      credit_(0) {}
  __device__ ~lare_engine() = default;
  __forceinline__ __device__ void load(struct lare_dev_info const& info) {
    waitId_ = tops::lare_get_wqe_slot(info.port_id, info.qp_id);
    lare_dev_info_ = info;
  }

  template <atomic_op_t OPS, typename D>
  __forceinline__ __device__ void prep_trans_data(D* dst,
                                    D* src, size_t nelem) {
    tops::lare_wrinfo wr;
    wr.port = lare_dev_info_.port_id;
    wr.qp = lare_dev_info_.qp_id;
    wr.sq_baseaddr = lare_dev_info_.base_addr;
    wr.sq_nelem = MAX_ROCE_SQELEM_NUM;
    wr.dstAddr = reinterpret_cast<void *>(dst);
    wr.srcAddr = reinterpret_cast<void *>(src);
    wr.ops = OPS;
    wr.data_type = getDataType<D>();
    wr.data_nelem = nelem;
    wr.fence = true;
    wr.enableMsi = true;
    uint32_t base_wqeId = (waitId_ + credit_) & (MAX_ROCE_SQELEM_NUM - 1);
    tops::lare_write_wqe(base_wqeId, &wr);
    credit_++;
  }

  __forceinline__ __device__ void ringDb() {
      waitId_ = tops::lare_ring_db(lare_dev_info_.port_id, lare_dev_info_.qp_id, MAX_ROCE_SQELEM_NUM, waitId_, credit_);
      credit_ = CREDIT_DEFAULT_VALUE;
  }

  __forceinline__ __device__ void waitTransDone() {
    (void)tops::lare_wait_done(lare_dev_info_.port_id, lare_dev_info_.qp_id, MAX_ROCE_SQELEM_NUM, waitId_, TIMEOUT);
  }

  private:
  uint32_t waitId_;
  uint32_t credit_;
  struct lare_dev_info lare_dev_info_;
};

class primitives_simple {
    public:
    __forceinline__ __device__ primitives_simple(struct primitives* prims, int subthread_id, int rank)
    :prims_(prims) {
      engine_.init();
      // Todo.在OGX上，subthread_id 跟dts_rank id 绑定。在后续支持超节点上，一个subthread需要多个dst_rank
      int responsible_rank_idx_ = subthread_id;
      if (responsible_rank_idx_ != rank) {
        lare_engine_.load(prims_->primitive[responsible_rank_idx_].lare_dev_info);
      }
    }

    template<typename T>
    __forceinline__ __device__ void atomic_add(T* input_buff, T* output_buff,
                               int value, int dst_rank, int rank) {
      if (dst_rank == rank) {
        tcle::atomic_add(output_buff, value);
      } else {
        *input_buff = value;
        lare_engine_.prep_trans_data<ATOMIC_SCALAR_STORE_ADD>(output_buff, input_buff, 1);
        __gcu_get_spr_barrier(BARRIER_MASK_L3);
        lare_engine_.ringDb();
        lare_engine_.waitTransDone();
      }
    }

    private:
    tops::private_dte engine_;
    struct primitives* prims_;
    lare_engine lare_engine_;
    // void* cached_value_ptrs_;
  };
