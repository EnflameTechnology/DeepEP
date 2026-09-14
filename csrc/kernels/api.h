#pragma once

#include <vector>
#include "primitive_simple.hpp"
#include "ep.h"

namespace deep_ep {

std::vector<uint8_t> get_unique_id();

int init(epComm_t* comm, const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks);

int destroy(epComm_t comm);

int allGather(epComm_t comm, void* data, size_t size);

namespace ibgda_gcu {
// setup RDMA for with rank / nRanks, buffer and size, output Cookie for free
int ibgdaSetup(epComm_t comm, int rank, int nRanks,
              void *rdma_buffer_ptr, size_t rdma_buffer_size, uintptr_t ibgda_info_ptr,
              void **ppCookie);

int ibgdaFree(epComm_t comm, void ** cookie);

} // namespace ibgda_gcu

// Intranode runtime
namespace intranode {

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, topsStream_t stream, struct primitives* prims, int* cached_value_ptrs);

} // namespace intranode

// Internode runtime
namespace internode {

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks, bool low_latency_mode);

void *alloc(size_t size, size_t alignment);

void free(void *ptr);

void barrier();

void finalize();

} // namespace internode

// Layout kernels
namespace layout {

void get_dispatch_layout(int64_t* topk_idx,
                         int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank,
                         int num_tokens, int num_topk, int num_ranks, int num_experts,
                         int num_rdma_ranks, int num_device_sms, int max_threads_per_sm,
                         topsStream_t stream);

} // namespace layout

// Intranode kernels
namespace intranode {

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     int num_tokens, const bool* is_token_in_rank, int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy, int num_memset_int, int expert_alignment,
                     void** buffer_ptrs, int** barrier_signal_ptrs, int rank, struct primitives* prims,
                     int* cached_value_ptrs, topsStream_t stream, int num_sms,
                     int num_device_sms, int max_threads_per_sm,
                     int* actual_num_recv_tokens = nullptr);

void cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int,
                            void** buffer_ptrs, int** barrier_signal_ptrs, int rank, int num_ranks,
                            topsStream_t stream, struct primitives* prims, int* cached_value_ptrs);

void dispatch(void* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,
              int* send_head, const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
              const bool* is_token_in_rank, const int* channel_prefix_matrix,
              int num_tokens, int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
              int scale_token_stride, int scale_hidden_stride,
              void** buffer_ptrs, int rank, int num_ranks,
              topsStream_t stream, int num_sms,
              int num_max_send_tokens, int num_recv_buffer_tokens, struct primitives* prims, void* cached_value_ptrs);

void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens, int num_memset_int,
                           int num_experts, int** barrier_signal_ptrs, int rank, int num_ranks, topsStream_t stream,
                           struct primitives* prims, int* cached_value_ptrs);

void combine(int type,
             void* recv_x, float* recv_topk_weights,
             const void* x, const float* topk_weights,
             const void* bias_0, const void* bias_1,
             const int* src_idx, const int* rank_prefix_matrix, const int* channel_prefix_matrix,
             int* send_head, int num_tokens, int num_recv_tokens, int hidden, int num_topk,
             void** buffer_ptrs, int rank, int num_ranks,
             topsStream_t stream, int num_sms,
             int num_max_send_tokens, int num_recv_buffer_tokens, struct primitives* prims, int* cached_value_ptrs);

} // namespace intranode

// Internode kernels
namespace internode {

int get_source_meta_bytes();

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     int num_nvl_ranks,
                     const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     const bool* is_token_in_rank, int num_tokens, int num_worst_tokens,
                     int num_channels, int num_device_sms, int max_threads_per_sm,
                     int hidden_int4, int num_scales, int num_topk, int expert_alignment,
                     int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs, int rank,
                     topsStream_t stream, int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                     bool low_latency_mode, struct primitives* prims);

void dispatch(void* recv_x, float* recv_x_scales, int* recv_topk_idx, float* recv_topk_weights, void* recv_src_meta,
              const void* x, const float* x_scales, const int* topk_idx, const float* topk_weights,
              int* send_rdma_head, int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
              int scale_token_stride, int scale_hidden_stride,
              void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs, int** barrier_signal_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
              int rank, int num_ranks, int num_nvl_ranks, bool is_cached_dispatch,
              topsStream_t stream, int num_channels, bool low_latency_mode, struct primitives* prims);

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
                   int num_ranks, int num_nvl_ranks, int num_channels, int num_device_sms, int max_threads_per_sm,
                   int num_combined_tokens, int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_nvl_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs, int rank, topsStream_t stream,
                   int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                   bool is_cached_dispatch, bool low_latency_mode);

void combine(int type,
             void* combined_x, float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x, const float* topk_weights,
             const void* bias_0, const void* bias_1,
             const int* combined_rdma_head, const int* combined_nvl_head,
             const void* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
             int num_tokens, int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
             int** barrier_signal_ptrs, int rank, int num_ranks, int num_nvl_ranks, topsStream_t stream, int num_channels, bool low_latency_mode, struct primitives* prims);

} // namespace internode

// Intranode low-latency kernels (single node)
namespace intranode_ll {

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x, void* rdma_buffer_ptr,
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0,
              const float* fp8_quant_scale,
              void* workspace, int* num_tokens_per_expert, int num_device_sms, int max_threads_per_sm,
              topsStream_t stream, int phases, struct primitives* prims);

void combine(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms, int max_threads_per_sm,
             topsStream_t stream, int phases, bool zero_copy, struct primitives* prims, int* cached_value_ptrs, void* rdma_buffer_ptr);

} // namespace intranode_ll

// Intranode low-latency slave-mode kernels (single node, direct RDMA writes)
// Selected when hidden <= SLAVE_MODE_HIDDEN_THRESHOLD
namespace intranode_ll_slave {

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x, void* rdma_buffer_ptr,
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0,
              const float* fp8_quant_scale,
              void* workspace, int* num_tokens_per_expert, int num_device_sms, int max_threads_per_sm,
              topsStream_t stream, int phases, struct primitives* prims);

} // namespace intranode_ll_slave

// Internode low-latency kernels (multi-node)
namespace internode_ll {

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              topsStream_t stream);

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x, void* rdma_buffer_ptr,
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0,
              const float* fp8_quant_scale,
              void* workspace, int* num_tokens_per_expert, int num_device_sms, int max_threads_per_sm,
              topsStream_t stream, int phases, struct primitives* prims);

void combine(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms, int max_threads_per_sm,
             topsStream_t stream, int phases, bool zero_copy, struct primitives* prims, int* cached_value_ptrs, void* rdma_buffer_ptr);

} // namespace internode_ll

} // namespace deep_ep
