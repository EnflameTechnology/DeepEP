#pragma once
#include "exception.h"
#include "primitive_simple.hpp"
#include "prims.h"
#include "utils.h"

#include "ibgda_device.h"
#include <cstdint>

// #define ibgdaTRACE(...)

namespace deep_ep {

template<int NVL_NUM, int RDMA_NUM>
class InternodePrimitives {
  public:
  // Default constructor to allow stack declaration then assignment in device code.
  __forceinline__ __device__ InternodePrimitives()
    : prims_(nullptr),
      sm_id_(0),
      subthread_id_(0),
      rank_(0),
      num_ranks_(0),
      nvl_rank_(0),
      rdma_rank_(0),
      rdma_buffer_base_(0),
      buffer_ptrs_(nullptr) {
        for (int i = 0; i < NVL_NUM; ++i) nvl_rank_to_peer_id_[i] = -1;
        for (int i = 0; i < RDMA_NUM; ++i) rdma_rank_to_peer_id_[i] = -1;
        for (int i = 0; i < RDMA_NUM; ++i) rdma_peer_base_addr_[i] = 0;
  }

  __forceinline__ __device__ InternodePrimitives(struct primitives* prims, int sm_id, void** buffer_ptrs, void* rdma_buffer_ptr,
      int subthread_id, int rank, int num_ranks)
    : prims_(prims),
      sm_id_(sm_id),
      subthread_id_(subthread_id),
      rank_(rank),
      num_ranks_(num_ranks),
      nvl_rank_(rank % NVL_NUM),
      rdma_rank_(rank / NVL_NUM),
      rdma_buffer_base_(reinterpret_cast<uint64_t>(rdma_buffer_ptr)),
      buffer_ptrs_(buffer_ptrs) {
        // Initialize rank_to_peer_id_ mapping table (default to -1 for invalid/self)
        for (int i = 0; i < NVL_NUM; i++) {
          nvl_rank_to_peer_id_[i] = -1;
        }
        for (int i = 0; i < RDMA_NUM; i++) {
          rdma_rank_to_peer_id_[i] = -1;
        }
        // 获取负责的nvl_rank 和 rdma_rank。
        int responsible_nvl_rank_idx_begin_ = rdma_rank_ * NVL_NUM;
        int responsible_nvl_rank_idx_end_ = (rdma_rank_ + 1) * NVL_NUM;
        int nvl_peer_num = 0;
        for (int peer_rank = responsible_nvl_rank_idx_begin_; peer_rank < responsible_nvl_rank_idx_end_; peer_rank++) {
          if (peer_rank == rank_) {
            continue;
          }
          int relative_rank_idx = peer_rank - responsible_nvl_rank_idx_begin_;
          nvl_rank_to_peer_id_[relative_rank_idx] = nvl_peer_num;
          lare_engine_[nvl_peer_num].load(prims_->primitive[peer_rank].lare_dev_info);
          nvl_peer_num++;
        }
        int rdma_peer_num = 0;
        for (int i=0; i<RDMA_NUM; i++) {
          int peer_rank = nvl_rank_ + i * NVL_NUM;
          if (peer_rank == rank_) continue;
          rdma_rank_to_peer_id_[i] = rdma_peer_num;
#ifdef ENABLE_MORI_GCU 
          assert(sm_id_ < num_ibgda_qps_);
          ibgda_engine_[rdma_peer_num].load(prims_->ibgda_info_ptr, sm_id_, num_ibgda_qps_, rank_, peer_rank);
#else
          ibgda_engine_esl_[rdma_peer_num].load(prims_->primitive[peer_rank].lare_dev_info);
#endif
          rdma_peer_base_addr_[rdma_peer_num] = prims_->rdma_peer_base[peer_rank];
          rdma_peer_num++;
        }
  }

  template<typename T>
  __forceinline__ __device__ void send_esl(const T* src, T* dst, ssize_t nelem, int dst_nvl_rank) {
    int peer_id = nvl_rank_to_peer_id_[dst_nvl_rank];
    if (peer_id < 0) return;  // Skip self or invalid rank
    lare_engine_[peer_id].template prep_trans_data<NON_ATOMIC_OPERATION>(dst, const_cast<T*>(src), nelem);
    __gcu_get_spr_barrier(BARRIER_MASK_L3);
    lare_engine_[peer_id].ringDb();
    lare_engine_[peer_id].waitTransDone();
  }

  template<typename T>
  __forceinline__ __device__ void atomic_add_esl(T* input_buff, T* output_buff,
                             int value, int dst_nvl_rank) {
    int peer_id = nvl_rank_to_peer_id_[dst_nvl_rank];
    if (dst_nvl_rank == nvl_rank_) {
      tcle::atomic_add(output_buff, value);
    } else {
      *input_buff = value;
      lare_engine_[peer_id].template prep_trans_data<ATOMIC_SCALAR_STORE_ADD>(output_buff, input_buff, 1);
      __gcu_get_spr_barrier(BARRIER_MASK_L3);
      lare_engine_[peer_id].ringDb();
      lare_engine_[peer_id].waitTransDone();
    }
  }

  template<typename T>
  __forceinline__ __device__ void send_rdma(const T* src, T* dst, ssize_t nelem, int dst_rdma_rank) {
      // EP_DEVICE_ASSERT(dst_rdma_rank>0 && dst_rdma_rank<RDMA_NUM);
      // EP_DEVICE_ASSERT(dst_rdma_rank != rdma_rank_);
      int peer_id = rdma_rank_to_peer_id_[dst_rdma_rank];
      if (peer_id < 0) return;  // Skip self or invalid rank
      uint64_t offset = (uint64_t)dst - rdma_buffer_base_;
      T* dst_ptr = reinterpret_cast<T*>(rdma_peer_base_addr_[peer_id] + offset);
#ifdef ENABLE_MORI_GCU
      ibgda_engine_[peer_id].template send(dst_ptr, const_cast<T*>(src), nelem);
#else
      ibgda_engine_esl_[peer_id].template prep_trans_data<NON_ATOMIC_OPERATION>(dst_ptr, const_cast<T*>(src), nelem);
      // Make sure WQE is visible before ringing DB to avoid occasional hang.
      __gcu_get_spr_barrier(BARRIER_MASK_L3);
      ibgda_engine_esl_[peer_id].ringDb();
      ibgda_engine_esl_[peer_id].waitTransDone();
#endif
  }

  private:
    struct primitives* prims_;
    int sm_id_;
    int subthread_id_;
    int rank_;
    int num_ranks_;
    int nvl_rank_;
    int rdma_rank_;
    uint64_t rdma_buffer_base_{0};
    void** buffer_ptrs_;
    uint64_t rdma_peer_base_addr_[RDMA_NUM] = {0};
    int nvl_rank_to_peer_id_[NVL_NUM];
    class lare_engine lare_engine_[NVL_NUM -1];

    int num_ibgda_qps_{MAX_IBGDA_QP_NUMS};
    int rdma_rank_to_peer_id_[RDMA_NUM];
#ifdef ENABLE_MORI_GCU
  class deep_ep_ibgda::ibgda_engine ibgda_engine_[RDMA_NUM - 1];
#else
class lare_engine ibgda_engine_esl_[RDMA_NUM - 1];
#endif
};

} // namespace deep_ep
