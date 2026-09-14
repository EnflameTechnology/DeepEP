// Copyright 2025 Enflame. All Rights Reserved.
#pragma once

#include <tops.h>
#include <krt/topstx.h>
#include <krt/lare.h>
#include "primitive_simple.hpp"
#include "configs.h"
#include "prims.h"

#define MAX_ROCE_USABLE_NUM (MAX_ROCE_SQELEM_NUM/2) // half of the ring is usable to avoid overflow


/**
 * @brief ESL GDA (Global Direct Access) Engine
 *
 * Handles multi-QP contention scenarios where multiple SMs/subthreads
 * may compete for access to the same QP.
 *
 * Key features:
 * - Atomic WQE slot allocation
 * - Ordered doorbell triggering
 * - Lock-free space management
 */
class EslGdaEngine {
public:
    /**
     * @brief Constructor
     *
     * @param ibgda_info_ptr Pointer to unified_endpoint array in L3 buffer
     * @param qp_shared_states_ptr Pointer to qp_shared_state array in L3 buffer
     * @param rank Current rank ID
     * @param num_ranks Total number of ranks
     */
    __forceinline__ __device__ EslGdaEngine(
        uint64_t ibgda_info_ptr,
        uint64_t qp_shared_states_ptr,
        int rank,
        int num_ranks
    ) : ibgda_info_ptr_(ibgda_info_ptr) {
          // will be used later
          static_cast<void>(qp_shared_states_ptr);
          // Load number of RC connections per PE from unified buffer metadata
          qp_msg_meta_t* meta_ptr = (qp_msg_meta_t*)(ibgda_info_ptr_ + QP_MSG_META_OFFSET);
          num_rc_per_pe_ = meta_ptr->num_rc_per_pe;
        }

    /**
     * @brief Get number of RC connections per PE
     *
     * @return Number of RC connections per PE
     */
    __forceinline__ __device__ int get_num_rc_per_pe() const {
        return num_rc_per_pe_;
    }

    /**
     * @brief Get QP endpoint for specified destination rank and QP index
     *
     * @param dst_rank Destination rank ID
     * @param qp_idx QP index within the destination rank (0-31)
     * @param eslEndpoint Output parameter to receive the endpoint structure
     */
    __forceinline__ __device__ void get_qp(int dst_rank, int qp_idx, esl_endpoint& endpoint) {
        // Calculate offset in unified buffer: [dst_rank][qp_idx]
        uint64_t offset = GET_UNIFIED_EP_OFFSET(dst_rank, qp_idx) + offsetof(struct unified_endpoint, esl_endpoint);
        esl_endpoint* ep_ptr = (esl_endpoint*)(ibgda_info_ptr_ + offset);
        endpoint = *ep_ptr;
    }
    __forceinline__ __device__ void get_qp_fetch(int dst_rank, int qp_idx) {
        // Calculate offset in unified buffer: [dst_rank][qp_idx]
        uint64_t offset = GET_UNIFIED_EP_OFFSET(dst_rank, qp_idx) + offsetof(struct unified_endpoint, esl_endpoint);
        esl_endpoint* ep_ptr = (esl_endpoint*)(ibgda_info_ptr_ + offset);
        constexpr int ep_bytes = sizeof(struct esl_endpoint);
        static_assert(ep_bytes <= 512);
        tcle::set_vl(std::min(512, ep_bytes));
        qp_cache = tcle::load<__vector char>((char*)ep_ptr);
        tcle::set_vl(512);
    }

    __forceinline__ __device__ void get_qp(esl_endpoint* endpoint) {
        constexpr int ep_bytes = sizeof(struct esl_endpoint);
        static_assert(ep_bytes <= 512);
        tcle::set_vl(std::min(512, ep_bytes));
               tcle::store(qp_cache, (char*)endpoint);
        tcle::set_vl(512);
    }

    /**
     * @brief Atomically allocate WQE slots for a QP
     *
     * This function ensures thread-safe allocation of WQE indices when
     * multiple threads compete for the same QP.
     *
     * @param qp Pointer to esl_endpoint structure
     * @param num_wqes Number of WQE slots to allocate
     * @return Base WQE index allocated to this thread
     */
    __forceinline__ __device__ uint64_t get_wqe_slots(
        struct esl_endpoint* qp,
        int num_wqes
    ) {
        // Get qp_shared_state for this QP
        volatile struct qp_shared_state* qp_state = (volatile struct qp_shared_state*) qp->qp_shared_state_ptr;

        // 1. Atomically allocate WQE indices (64-bit to prevent overflow)
        uint64_t my_wqe_index = tcle::atomic_add((void*) &qp_state->postIdx, (unsigned long long)num_wqes);

        // 2. Space check (prevent queue overflow).
        // Check that the allocated slot of our allocation (my_wqe_index + num_wqes) fits
        // in the ring, i.e. the distance from done_idx is < MAX_ROCE_USABLE_NUM.
        while (true) {
            // Use volatile read to ensure memory visibility
            volatile uint64_t done_idx = qp_state->doneIdx;

            // Check if the last allocated slot fits within the ring
            if (my_wqe_index + (uint64_t)num_wqes - done_idx < MAX_ROCE_USABLE_NUM) {
                break;  // Enough space available
            }

            // Not enough space, need to wait for completed WQEs to free up space
            volatile uint64_t db_touched = qp_state->dbTouchIdx;

            // Check if there are WQEs that have been triggered but not completed
            if (db_touched != done_idx) {
                uint64_t waitId = (uint64_t)db_touched;
                // Wait for the doorbell-triggered WQEs to complete
                // Note: lare_wait_done uses uint32_t waitId, so we cast (lower 32 bits are sufficient for HW)
                if (false != tops::lare_wait_done(qp->port_id, qp->qp_id, MAX_ROCE_SQELEM_NUM, (uint32_t)waitId, 0)) {
                    // Update doneIdx using atomic max (ensures monotonic increase)
                    tcle::atomic_max((void*) &qp_state->doneIdx, (unsigned long long)waitId);
                }
            }
        }

        return my_wqe_index;
    }

    /**
     * @brief Send data using specified QP and WQE index
     *
     * @param dst Destination address (remote)
     * @param src Source address (local)
     * @param qp Pointer to esl_endpoint structure
     * @param wqe_idx WQE index to use (allocated by get_wqe_slots)
     * @param nelem Number of elements to send
     * @param ops Operation type (e.g., NON_ATOMIC_OPERATION)
     */
    template<atomic_op_t OPS, typename T>
    __forceinline__ __device__ void send(
        T* dst,
        const T* src,
        struct esl_endpoint* qp,
        uint64_t wqe_idx,
        size_t nelem,
        bool fence
    ) {
        // Build WQE
        tops::lare_wrinfo wr;
        wr.port = qp->port_id;
        wr.qp = qp->qp_id;
        wr.sq_baseaddr = qp->base_addr;
        wr.sq_nelem = MAX_ROCE_SQELEM_NUM;
        wr.dstAddr = reinterpret_cast<void*>(dst);
        wr.srcAddr = const_cast<T*>(src);
        wr.ops = OPS;
        wr.data_type = getDataType<T>();
        wr.data_nelem = nelem;
        wr.fence = fence;
        wr.enableMsi = true;

        // Write WQE to the specified slot (use lower bits for circular buffer index)
        uint32_t base_wqeId = (uint32_t)(wqe_idx & (MAX_ROCE_SQELEM_NUM - 1));
        tops::lare_write_wqe(base_wqeId, &wr);
    }

    /**
     * @brief Trigger doorbell for specified QP and WQE index
     *
     * This function ensures ordered doorbell triggering when multiple threads
     * share the same QP. It waits until it's this thread's turn to trigger.
     *
     * @param qp Pointer to esl_endpoint structure
     * @param wqe_idx WQE index to trigger (allocated by get_wqe_slots)
     * @param num_wqes Number of WQEs to trigger (credit)
     * @return New waitId (64-bit) for wait_done
     */
    __forceinline__ __device__ uint64_t ringDb(
        struct esl_endpoint* qp,
        uint64_t wqe_idx,
        int num_wqes
    ) {
        // Get qp_shared_state for this QP
        volatile struct qp_shared_state* qp_state = (volatile struct qp_shared_state*) qp->qp_shared_state_ptr;

        // Wait for our turn to trigger doorbell (ensures ordering)
        volatile uint64_t db_touched = qp_state->dbTouchIdx;

        // printf("+++ entter: ringDb: db_touched: %llu, wqe_idx: %llu\n", db_touched, wqe_idx);

        while (db_touched != wqe_idx) {
          tops::__nanosleep(2);
          db_touched = qp_state->dbTouchIdx;
        }

        // Now it's our turn, trigger doorbell
        // Note: lare_ring_db uses uint32_t internally, cast wqe_idx (lower 32 bits sufficient for HW)
        tops::lare_ring_db(
            qp->port_id,
            qp->qp_id,
            MAX_ROCE_SQELEM_NUM,
            (uint32_t)wqe_idx,
            num_wqes  // credit
        );

        // printf("+++ ringDb: db_touched: %llu, wqe_idx: %llu, hw_waitId: %u\n", db_touched, wqe_idx, hw_waitId);

        // Calculate the 64-bit waitId for tracking
        uint64_t new_waitId = wqe_idx + num_wqes;

        // Update dbTouchIdx to let next thread continue
        // Since wqe_idx is strictly increasing, wqe_idx + num_wqes is also strictly increasing
        tcle::atomic_max((void*) &qp_state->dbTouchIdx, (unsigned long long)new_waitId);

        return new_waitId;
    }

    /**
     * @brief Wait for transmission to complete and update doneIdx
     *
     * @param qp Pointer to esl_endpoint structure
     * @param waitId WaitId returned by ringDb (64-bit)
     */
    __forceinline__ __device__ void wait_done(
        struct esl_endpoint* qp,
        uint64_t waitId
    ) {
        // Get qp_shared_state for this QP
        volatile struct qp_shared_state* qp_state = (volatile struct qp_shared_state*) qp->qp_shared_state_ptr;

        // Wait for transmission to complete
        // Note: lare_wait_done uses uint32_t waitId, cast (lower 32 bits sufficient for HW)
        static_assert(MAX_ROCE_USABLE_NUM <= MAX_ROCE_SQELEM_NUM/2, "MAX_ROCE_USABLE_NUM must NOT greater than half of MAX_ROCE_SQELEM_NUM");
        while (true) {
            uint32_t lareDoneIdx = 0;
            bool lareDone = tops::lare_wait_done(qp->port_id, qp->qp_id, MAX_ROCE_SQELEM_NUM, (uint32_t)waitId, &lareDoneIdx, 0);
            if (lareDone) {
                tcle::atomic_max((void*) &qp_state->doneIdx, (unsigned long long)waitId);
                break;
            } else {
                int32_t delta = (int32_t)lareDoneIdx - (int32_t)(waitId & 0xFFULL);
                if (delta >= MAX_ROCE_USABLE_NUM) {
                    delta -= MAX_ROCE_SQELEM_NUM;
                } else if (delta <= -MAX_ROCE_USABLE_NUM) {
                    delta += MAX_ROCE_SQELEM_NUM;
                }

                if (delta >= 0) {
                    uint64_t doneIdx = (uint64_t)((int64_t)waitId + (int64_t)delta);
                    tcle::atomic_max((void*) &qp_state->doneIdx, (unsigned long long)doneIdx);
                    break;
                }
            }
        }
    }

    // =========================================================================
    // Non-atomic versions (use when QP is guaranteed not to be shared)
    // =========================================================================

    /**
     * @brief Allocate WQE slots for a QP (non-atomic version)
     *
     * Use this version when the QP is guaranteed not to be shared between
     * multiple threads/SMs. Avoids atomic operations for better performance.
     *
     * @param qp Pointer to esl_endpoint structure
     * @param num_wqes Number of WQE slots to allocate
     * @return Base WQE index allocated
     */
    __forceinline__ __device__ uint64_t get_wqe_slots_raw(
        struct esl_endpoint* qp,
        int num_wqes
    ) {
        // Get qp_shared_state for this QP
        volatile struct qp_shared_state* qp_state = (volatile struct qp_shared_state*) qp->qp_shared_state_ptr;

        // 1. Read and update WQE index (no atomic needed - exclusive access)
        uint64_t my_wqe_index = qp_state->postIdx;
        qp_state->postIdx = my_wqe_index + num_wqes;

        // 2. Space check (prevent queue overflow).
        // Check that the LAST slot of our allocation fits within the ring.
        while (true) {
            uint64_t done_idx = qp_state->doneIdx;

            if (my_wqe_index + (uint64_t)num_wqes - 1 - done_idx < MAX_ROCE_SQELEM_NUM) {
                break;  // Enough space available
            }

            // Not enough space, wait for completed WQEs
            uint64_t db_touched = qp_state->dbTouchIdx;

            if (db_touched != done_idx) {
                uint64_t waitId = db_touched;
                tops::lare_wait_done(qp->port_id, qp->qp_id, MAX_ROCE_SQELEM_NUM, (uint32_t)waitId, TIMEOUT);
                // Direct update (no atomic needed - exclusive access)
                qp_state->doneIdx = waitId;
            }
        }

        return my_wqe_index;
    }

    /**
     * @brief Trigger doorbell for specified QP (non-atomic version)
     *
     * Use this version when the QP is guaranteed not to be shared between
     * multiple threads/SMs. Skips ordering wait and atomic update.
     *
     * @param qp Pointer to esl_endpoint structure
     * @param wqe_idx WQE index to trigger
     * @param num_wqes Number of WQEs to trigger (credit)
     * @return New waitId (64-bit) for wait_done
     */
    __forceinline__ __device__ uint64_t ringDb_raw(
        struct esl_endpoint* qp,
        uint64_t wqe_idx,
        int num_wqes
    ) {
        // Get qp_shared_state for this QP
        volatile struct qp_shared_state* qp_state = (volatile struct qp_shared_state*) qp->qp_shared_state_ptr;

        // No need to wait for turn - exclusive access guarantees ordering

        // Trigger doorbell
        tops::lare_ring_db(
            qp->port_id,
            qp->qp_id,
            MAX_ROCE_SQELEM_NUM,
            (uint32_t)wqe_idx,
            num_wqes  // credit
        );

        // Calculate the 64-bit waitId for tracking
        uint64_t new_waitId = wqe_idx + num_wqes;

        // Direct update dbTouchIdx (no atomic needed - exclusive access)
        qp_state->dbTouchIdx = new_waitId;

        return new_waitId;
    }

    /**
     * @brief Wait for transmission to complete (non-atomic version)
     *
     * Use this version when the QP is guaranteed not to be shared between
     * multiple threads/SMs. Avoids atomic update of doneIdx.
     *
     * @param qp Pointer to esl_endpoint structure
     * @param waitId WaitId returned by ringDb (64-bit)
     */
    __forceinline__ __device__ void wait_done_raw(
        struct esl_endpoint* qp,
        uint64_t waitId
    ) {
        // Get qp_shared_state for this QP
        volatile struct qp_shared_state* qp_state = (volatile struct qp_shared_state*) qp->qp_shared_state_ptr;

        // Wait for transmission to complete
        tops::lare_wait_done(qp->port_id, qp->qp_id, MAX_ROCE_SQELEM_NUM, (uint32_t)waitId, TIMEOUT);

        // Direct update doneIdx (no atomic needed - exclusive access)
        qp_state->doneIdx = waitId;
    }

private:

    uint64_t ibgda_info_ptr_;          // Pointer to unified_endpoint array
    int num_rc_per_pe_;                 // Number of RC connections per PE
    __vector char qp_cache;           // to cache qp
};


