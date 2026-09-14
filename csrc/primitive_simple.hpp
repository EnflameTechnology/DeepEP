#define MAXSIDES 2

#define DEEP_EP_MULTINODES(_nNodes)  ((_nNodes) > 1)

#pragma once

#include "devcomm.h"

#define MAX_ESL_QP_NUMS 32

    struct lare_dev_info {
        uint32_t port_id;
        uint32_t qp_id;
        void* base_addr;
    };

    struct primitive {
        struct lare_dev_info lare_dev_info;
    };

struct esl_endpoint {
    uint32_t port_id; // the esl port id (0-15)
    uint32_t qp_id; // the qp_id from the create qp call
    void* base_addr; // the send queue base address

    int rank; // the rank of the current rank
    int peer_rank; // the rank of the peer
    uint32_t qp_index; // qp_index of the qp in the rank
    uint64_t qp_shared_state_ptr; // point this qp's shared state in L3 buffer
};

// Unified endpoint struct: stores either ESL or IBGDA endpoint
// - For intra-node peers: stores esl_endpoint info
// - For inter-node peers: stores IBGDA endpoint info
// - Memory layout: endpoints indexed by (peer_rank, qp_index)
// - Size: 384 bytes (aligned) to match GET_UNIFIED_EP_OFFSET layout

struct unified_endpoint {
    // Inter-node: IBGDA endpoint (264 bytes, placeholder for RdmaEndpoint)
    uint8_t ibgda_storage[264];

    // Intra-node: ESL endpoint
    struct esl_endpoint esl_endpoint;

    // Padding to align to 384 bytes (264 + 120 = 384)
    // This ensures pointer arithmetic matches GET_UNIFIED_EP_OFFSET calculation
    uint8_t padding[80];
};
static_assert(sizeof(unified_endpoint) == 384, "unified_endpoint size must be 384 bytes (aligned)");

struct primitives {
    // TODO(inter&intra) primitive and peerDirectAddrs should be indexed by local peer rank
    struct primitive primitive[EP_MAX_LOCAL_RANKS];  // Kept for backward compatibility
    uint64_t* peerDirectAddrs[EP_MAX_LOCAL_RANKS];
    uint64_t* rdma_peer_base{nullptr};
    uint64_t ibgda_info_ptr;  // Now points to unified_endpoint array
    uint64_t qp_shared_states_ptr{0};  // Points to qp_shared_state array in L3 buffer (for QP contention synchronization)
    uint64_t slave_rdma_base[EP_MAX_RANKS][MAX_ESL_QP_NUMS];
    int nNodes;
    int nLocalRanks;
};

// QP shared state structure (used for the scenario where multiple subthread compete for access to the same QP)
// Note: Using uint64_t to prevent overflow issues with atomic_max operations.
//       With uint32_t, atomic_max would fail when indices wrap around (e.g., max(0xFFFFFFF0, 0x00000002) = 0xFFFFFFF0).
//       With uint64_t, overflow is practically impossible (5800+ years at 100M WQE/sec).
struct qp_shared_state {
    // WQE index allocation state
    uint64_t postIdx{0};        // Allocated WQE index (atomic fetch_add)
    uint64_t dbTouchIdx{0};     // Maximum WQE index with doorbell triggered (updated with tcle::atomic_max after ringDb)
    uint64_t doneIdx{0};        // Maximum completed WQE index (updated with tcle::atomic_max after waitTransDone)
};
static_assert(sizeof(qp_shared_state) == 24, "qp_shared_state size must be 24 bytes");
