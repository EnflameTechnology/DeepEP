#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <tops_runtime.h>
#include <memory>
#include <pybind11/functional.h>
#include <torch/python.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>

#include "ep.h"
#include "deep_ep.hpp"
#include "deep_ep_trace.h"
#include "kernels/api.h"
#include "kernels/configs.h"
#include "comm.h"

namespace deep_ep {

namespace delay {
void launch_delay(unsigned int delay_ns, topsStream_t stream);
}

// Jitter injection via environment variables (non-invasive, no API changes)
// Set DEEP_EP_HOST_JITTER_MS / DEEP_EP_DEVICE_JITTER_MS to enable jitter.
static float get_jitter_ms(const char* env_name) {
    const char* val = std::getenv(env_name);
    if (val != nullptr) {
        try {
            return std::stof(val);
        } catch (...) {
            return 0.0f;
        }
    }
    return 0.0f;
}

// Only used for debug, simulate inter-node communication by setting INTRA_RANKS to the number of ranks in the node
EP_PARAM(IntraRanks, "INTRA_RANKS", -1);
EP_PARAM(DebugCombineUseSm, "DEBUG_COMBINE_USE_SM", -1);

namespace {

std::string format_int32_vector_json(const std::vector<int>& values, size_t max_chars = 4096) {
    std::string s = "[";
    bool truncated = false;
    for (size_t i = 0; i < values.size() && !truncated; ++i) {
        if (i > 0)
            s += ',';
        const std::string cell = std::to_string(values[i]);
        if (s.size() + cell.size() + 8 > max_chars) {
            truncated = true;
            break;
        }
        s += cell;
    }
    if (truncated)
        s += "...";
    s += ']';
    return s;
}

// rank_prefix_matrix[row][col] = cumulative tokens from ranks 0..row sent to col.
// send_tokens[dst] = tokens this rank sends to dst; recv_tokens[src] = tokens received from src.
std::pair<std::vector<int>, std::vector<int>>
compute_send_recv_tokens_from_prefix_matrix(const torch::Tensor& cpu_matrix, int rank) {
    EP_HOST_ASSERT(cpu_matrix.device().is_cpu());
    EP_HOST_ASSERT(cpu_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(cpu_matrix.dim() == 2);
    EP_HOST_ASSERT(cpu_matrix.size(0) == cpu_matrix.size(1));
    const int num_ranks = static_cast<int>(cpu_matrix.size(0));
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks);
    auto acc = cpu_matrix.accessor<int, 2>();
    std::vector<int> send_tokens(num_ranks), recv_tokens(num_ranks);
    for (int dst = 0; dst < num_ranks; ++dst)
        send_tokens[dst] = acc[rank][dst] - (rank > 0 ? acc[rank - 1][dst] : 0);
    for (int src = 0; src < num_ranks; ++src)
        recv_tokens[src] = acc[src][rank] - (src > 0 ? acc[src - 1][rank] : 0);
    return {send_tokens, recv_tokens};
}

} // namespace

Buffer::Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode, bool explicitly_destroy,
               bool is_internode, const std::vector<uint8_t>& root_unique_id):
        low_latency_mode(low_latency_mode),
        num_nvl_bytes(num_nvl_bytes),
        num_rdma_bytes(num_rdma_bytes),
        rank(rank), num_ranks(num_ranks),
        comm_stream(torch_gcu::getStreamFromPool(true)),
        explicitly_destroy(explicitly_destroy),
        is_internode(is_internode) {
    // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    TOPS_CHECK(topsGetDevice(&device_id));
    deep_ep::init(&epComm, root_unique_id, rank, num_ranks);
    local_ranks = epComm->localRanks;
    // Get the simulated intra-node ranks from user environment variable
    int intra_ranks = deep_ep::epParamIntraRanks();
    if (intra_ranks > 0) {
        EP_HOST_ASSERT(intra_ranks <= local_ranks and "INTRA_RANKS must be less than or equal to the number of ranks in the node");
        local_ranks = intra_ranks;
    }

    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % local_ranks == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > local_ranks or low_latency_mode);
        //EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode or is_internode);

    // Get ranks
    rdma_rank = rank / local_ranks, nvl_rank = rank % local_ranks;
    num_rdma_ranks = std::max(1, num_ranks / local_ranks), num_nvl_ranks = std::min(num_ranks, local_ranks);
    
#ifdef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");
#endif

    //  Get device info
    topsDeviceProp_t device_prop = {};
    TOPS_CHECK(topsGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;
    max_threads_per_sm = device_prop.maxThreadsPerBlock;

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handles
        TOPS_CHECK(topsMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes));
        TOPS_CHECK(topsPointerGetAttribute(&buffer_ptrs_edf[nvl_rank], TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER, buffer_ptrs[nvl_rank]));
        buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        // Set barrier signals
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu = reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        TOPS_CHECK(topsMemset(buffer_ptrs[nvl_rank], 0, num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes));
        TOPS_CHECK(topsMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
    }
    TOPS_CHECK(topsExtMallocWithFlags((void **)&cached_value_ptrs, INTRANODE_CACHED_VALUE_POOL_BYTES(num_ranks), topsMallocHostAccessable));
    *cached_value_ptrs = 1;

    // Create 32 MiB workspace
    TOPS_CHECK(topsMalloc(&workspace, NUM_WORKSPACE_BYTES));
    TOPS_CHECK(topsMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    // MoE counter
    TOPS_CHECK(topsHostMalloc(&moe_recv_counter, sizeof(int64_t)));
    TOPS_CHECK(topsHostGetDevicePointer(reinterpret_cast<void**>(&moe_recv_counter_mapped), const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    TOPS_CHECK(topsHostMalloc(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, topsHostMallocMapped));
    TOPS_CHECK(topsHostGetDevicePointer(reinterpret_cast<void**>(&moe_recv_expert_counter_mapped), const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++ i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        TOPS_CHECK(topsHostMalloc(&moe_recv_rdma_counter, sizeof(int), topsHostMallocMapped));
        TOPS_CHECK(topsHostGetDevicePointer(reinterpret_cast<void**>(&moe_recv_rdma_counter_mapped), const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }
}

Buffer::~Buffer() noexcept(false) {
    if (not explicitly_destroy) {
        destroy();
    } else if (not destroyed) {
        printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_internode_available() const {
    return is_available() and (is_internode or num_rdma_ranks > 0);
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    if (is_internode) {
        return global ? rank : 0;
    } else {
        return global ? nvl_rank : 0;
    }
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    return {ipc_handles[nvl_rank].reserved, TOPS_IPC_HANDLE_SIZE};
}

uint64_t Buffer::get_local_edf() const {
    return buffer_ptrs_edf[nvl_rank];
}

pybind11::bytearray Buffer::get_unique_id() const {
    auto unique_id = deep_ep::get_unique_id();
    return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
    return pybind11::bytearray();
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& /*dtype*/ , int64_t /* offset*/, bool /*use_rdma_buffer*/) const {
    return torch::Tensor();
}

torch::Stream Buffer::get_comm_stream() const {
    return comm_stream;
}

std::tuple<int, int, std::vector<int>> Buffer::get_moe_recv_info(int num_local_experts) const {
    EP_HOST_ASSERT(num_local_experts >= 0 && num_local_experts <= NUM_MAX_LOCAL_EXPERTS);
    EP_HOST_ASSERT(moe_recv_counter != nullptr);
    EP_HOST_ASSERT(moe_recv_expert_counter != nullptr);

    const int total = static_cast<int>(*moe_recv_counter);
    const int rdma_total = (moe_recv_rdma_counter != nullptr) ? static_cast<int>(*moe_recv_rdma_counter) : -1;

    std::vector<int> per_expert;
    per_expert.reserve(num_local_experts);
    for (int i = 0; i < num_local_experts; ++i) {
        per_expert.push_back(static_cast<int>(moe_recv_expert_counter[i]));
    }
    return {total, rdma_total, per_expert};
}

void Buffer::get_topo_info() {
  const int numChannels = epComm->meshnChannels;
  EP_HOST_ASSERT(numChannels > 0 && numChannels <= EP_MAX_MESHX_CHANNELS);

  size_t size = sizeof(struct primitives);
  struct primitives* hostPrims = (struct primitives*)malloc(size * numChannels * MAXSIDES);
  for (int channelId = 0; channelId < numChannels; channelId++) {
    struct epChannel channel = epComm->channels[channelId];
    for (int side = 0; side < MAXSIDES; side++) {
      for (int peerRank = 0; peerRank < epComm->nRanks; peerRank++) {
        if (peerRank == epComm->rank) continue;
        hostPrims[channelId * MAXSIDES + side].nNodes         = num_rdma_ranks;
        hostPrims[channelId * MAXSIDES + side].nLocalRanks    = local_ranks;
        hostPrims[channelId * MAXSIDES + side].ibgda_info_ptr = 0;
        hostPrims[channelId * MAXSIDES + side].ibgda_info_ptr = (uintptr_t)rdma_ibgda_info_addr_dev;
        hostPrims[channelId * MAXSIDES + side].qp_shared_states_ptr = (uintptr_t)qp_shared_states_addr_dev;
        hostPrims[channelId * MAXSIDES + side].rdma_peer_base = reinterpret_cast<uint64_t*>(rdma_peer_base_addr_dev);
        // Ensure peerRank is NOT OOR
        EP_HOST_ASSERT(peerRank < EP_MAX_LOCAL_RANKS);
        hostPrims[channelId * MAXSIDES + side].primitive[peerRank].lare_dev_info.base_addr = reinterpret_cast<uint64_t*>(-1UL);
        if (peerRank / local_ranks != rdma_rank) continue;
        EP_HOST_ASSERT(nullptr !=  channel.devPeers[peerRank].connInfo[side].devConnInfoV4Ptr);
        struct epSimpleConnInfoCtxt contxt = channel.devPeers[peerRank].connInfo[side].devConnInfoV4Ptr->simple.contxt;
        // Only use first LARE port info for now
        hostPrims[channelId * MAXSIDES + side].primitive[peerRank].lare_dev_info.port_id = contxt.localLareInfo[0].qpInfo.portId;
        hostPrims[channelId * MAXSIDES + side].primitive[peerRank].lare_dev_info.qp_id = contxt.localLareInfo[0].qpInfo.qpId;
        hostPrims[channelId * MAXSIDES + side].primitive[peerRank].lare_dev_info.base_addr = contxt.localLareInfo[0].sqInfo.baseAddr;
      }
    }
    memcpy(hostPrims[channelId].peerDirectAddrs, epComm->devPeerDirectBuffAddrs[channelId], EP_MAX_LOCAL_RANKS * sizeof(uint64_t));
  }

#ifdef ENABLE_MORI_GCU

  if (DEEP_EP_MULTINODES(num_rdma_ranks) || is_internode) {
    NEQCHECK(epSuccess != ibgda_gcu::ibgdaSetup(epComm, epComm->rank, epComm->nRanks, rdma_buffer_ptr, num_rdma_bytes,
                                                (uintptr_t)rdma_ibgda_info_addr_dev, &epComm->ibgdaCookie), false);
  }
#endif

  // Copy into devPrims — allocate only the channels ep_init actually created.
  auto topsext_err = topsExtMallocWithFlags((void **)&prims, numChannels * size * MAXSIDES, topsMallocHostAccessable);
  (void) topsext_err;
  memcpy(prims, hostPrims, numChannels * size * MAXSIDES);

  // Copy LARE QP info for intra-node peers to unified buffer
  // This overwrites IBGDA endpoint slots for intra-node peers
  //
  // Unified Buffer Layout (matching IBGDA) qp_num = 32 (default):
  //   [rank0][qp_index0-31]  [rank1][qp_index0-31]  ...  [rank31][qp_index0-31]
  //
  // Key details:
  //   - Each rank has numChannels * MAXSIDES QP endpoints (numChannels ≤ EP_MAX_MESHX_CHANNELS)
  //   - Each channel has MAXSIDES (2) sides, each side has its own QP
  //   - Each endpoint: 384 bytes (aligned from 264 bytes)
  //   - Only store SEND QP info (recv is not needed for current implementation)
  //   - qp_index field in lare struct records the QP index for each endpoint
  //
  // Filling order: rank0[qp_index0-31], rank1[qp_index0-31], ..., rank31[qp_index0-31]
  // This matches IBGDA's layout and GET_UNIFIED_EP_OFFSET calculation

  const int rank = epComm->rank;
  const int nRanks = epComm->nRanks;
  const int localRanks = local_ranks;
  const int node = rdma_rank;

  // Adaptive QP channel configuration based on cluster scale.
  // ESL QP has two working modes: master (WQE-based send) and slave (load/store RDMA write).
  // A single QP can only operate in one mode; both may be needed simultaneously, so separate
  // QPs are allocated per mode.  slave_rdma_base[peer][qp] is the slave-mode window;
  // kernel stores to it trigger RDMA writes to the peer's receive buffer.
  //
  // ep_init.cc computes maxChannels via formula: budget × 16 / (N × 2), so
  // epComm->meshnChannels reflects the actual QP-budget-constrained channel count.
  //
  // Representative channel/QP layout (budget=128 for N>64):
  //   N= 64: 14 ch → 12 master + 2 slave → (64/16)×14×2=112 QP/port ≤ 128 ✓
  //   N=128:  8 ch →  6 master + 2 slave → (128/16)× 8×2=128 QP/port ✓
  //   N=192:  5 ch →  3 master + 2 slave → (192/16)× 5×2=120 QP/port ✓
  //   N=256:  4 ch →  2 master + 2 slave → (256/16)× 4×2=128 QP/port ✓
  //   N=512:  2 ch →  1 master + 1 slave → (512/16)× 2×2=128 QP/port ✓ (future)
  //
  // kSlaveChannelCount is capped so that at least 1 master channel always remains.
  const int actualChannels         = epComm->meshnChannels;
  // Reserve up to 2 slave channels; ensure at least 1 master channel survives.
  const int kSlaveChannelCount     = std::min(2, std::max(0, actualChannels - 1));
  const int kMasterChannelCount    = std::min(12, actualChannels - kSlaveChannelCount);
  const int kSlaveModeChannelStart = kMasterChannelCount;
  const int kMasterQpCount         = kMasterChannelCount * MAXSIDES;
  const int kSlaveQpCount          = kSlaveChannelCount  * MAXSIDES;
  EP_HOST_ASSERT(kMasterChannelCount > 0 &&
                 "ep_init did not create enough channels; check meshGraph.maxChannels");
  EP_HOST_ASSERT(kSlaveModeChannelStart + kSlaveChannelCount <= actualChannels &&
                 "Not enough ep_init channels for master + slave QPs");
  EP_HOST_ASSERT(kMasterQpCount <= MAX_IBGDA_QP_NUMS &&
                 "Master QP count exceeds unified buffer capacity");
  const int master_qp_reg_count = kMasterQpCount;

  // Write metadata: num_rc_per_pe for master, num_slave_qp_per_pe for slave.
  {
    qp_msg_meta_t* qp_msg_meta = (qp_msg_meta_t*)((uintptr_t)rdma_ibgda_info_addr_dev + QP_MSG_META_OFFSET);
    int num_master_qp = kMasterQpCount;
    TOPS_CHECK(topsMemcpy(&qp_msg_meta->num_rc_per_pe, &num_master_qp, sizeof(int), topsMemcpyHostToDevice));
    int num_slave_qp = kSlaveQpCount;
    TOPS_CHECK(topsMemcpy(&qp_msg_meta->num_slave_qp_per_pe, &num_slave_qp, sizeof(int), topsMemcpyHostToDevice));
  }

  // Fill unified buffer ESL endpoints for master-mode QPs (channels 0 .. kMasterChannelCount-1).
  for (int peerRank = 0; peerRank < nRanks; peerRank++) {
    if (peerRank == rank) continue;  // Skip self
    // Only process intra-node peers (same node)
    if ((peerRank / localRanks) != node) continue;

    // For each peerRank, iterate over all master QP indices
    // Each channel has MAXSIDES sides, use all of them as separate QPs
    for (int qp_index = 0; qp_index < master_qp_reg_count; qp_index++) {
      // Decompose qp_index to channelId and side (master channels: 0 .. kMasterChannelCount-1)
      int channelId = qp_index / MAXSIDES;
      int side = qp_index % MAXSIDES;
      // Get the channel
      struct epChannel channel = epComm->channels[channelId];
      // Extract connection context from the corresponding side
      struct epSimpleConnInfoCtxt sendContxt =
            channel.devPeers[peerRank].connInfo[side].devConnInfoV4Ptr->simple.contxt;
      // Construct unified_endpoint with ESL endpoint info
      struct esl_endpoint ep;
      memset(&ep, 0, sizeof(ep));
      // Fill ESL endpoint info using localLareInfo[0] (only first LARE port)
      ep.port_id = sendContxt.localLareInfo[0].qpInfo.portId;
      ep.qp_id = sendContxt.localLareInfo[0].qpInfo.qpId;
      ep.base_addr = sendContxt.localLareInfo[0].sqInfo.baseAddr;
      ep.rank = rank;
      ep.peer_rank = peerRank;
      // Record QP index
      ep.qp_index = qp_index;  // QP index (0 to kMasterQpCount-1 = 23)
      // Calculate qp_shared_state pointer for this QP
      // Layout: qp_shared_states_addr_dev[peerRank * MAX_IBGDA_QP_NUMS + qp_index]
      int qp_state_index = peerRank * MAX_IBGDA_QP_NUMS + qp_index;
      ep.qp_shared_state_ptr = (uint64_t)qp_shared_states_addr_dev
                                           + qp_state_index * sizeof(struct qp_shared_state);
      // Calculate offset in unified buffer: [peerRank][qp_index]
      // This matches IBGDA's layout: rank0[0-31], rank1[0-31], ...
      uint64_t offset = GET_UNIFIED_EP_OFFSET(peerRank, qp_index) + offsetof(struct unified_endpoint, esl_endpoint);
      // Copy ESL endpoint to device
      // Note: sizeof(esl_endpoint) = 32 bytes (with compiler padding for alignment)
      TOPS_CHECK(topsMemcpy(
          (void*)((uintptr_t)rdma_ibgda_info_addr_dev + offset),
          &ep,
          sizeof(struct esl_endpoint),
          topsMemcpyHostToDevice
      ));
    }
  }

  // Register each intra-node peer's rdma_buffer_ptr for slave mode writes.
  // Uses slave-dedicated channels [kSlaveModeChannelStart, kSlaveModeChannelStart+kSlaveChannelCount)
  // (= channels 12,13) so these QPs are never shared with the master-mode ESL sends in combine.
  // For each (peerRank, qp_index), call topsRoceRegMem with the slave port+QP so that kernel
  // stores to (slave_rdma_base[peerRank][qp_index] + offset) trigger RDMA writes to the peer.
  if (num_rdma_bytes > 0) {
    const int slave_qp_reg_count = kSlaveQpCount;
    memset(hostPrims[0].slave_rdma_base, 0, sizeof(hostPrims[0].slave_rdma_base));
    for (int peerRank = 0; peerRank < epComm->nRanks; peerRank++) {
      if (peerRank == epComm->rank) continue;
      // Only register intra-node peers (same node = same rdma_rank bucket)
      if (peerRank / local_ranks != rdma_rank) continue;
      for (int qp_index = 0; qp_index < slave_qp_reg_count; qp_index++) {
        // Slave-dedicated channels: kSlaveModeChannelStart .. kSlaveModeChannelStart+kSlaveChannelCount-1
        int slaveChannelId = kSlaveModeChannelStart + qp_index / MAXSIDES;
        int slaveSide      = qp_index % MAXSIDES;
        struct epChannel slaveChannel = epComm->channels[slaveChannelId];
        EP_HOST_ASSERT(nullptr != slaveChannel.devPeers[peerRank].connInfo[slaveSide].devConnInfoV4Ptr);
        struct epSimpleConnInfoCtxt slaveContxt =
            slaveChannel.devPeers[peerRank].connInfo[slaveSide].devConnInfoV4Ptr->simple.contxt;
        uint32_t portId = slaveContxt.localLareInfo[0].qpInfo.portId;
        uint32_t qpId   = slaveContxt.localLareInfo[0].qpInfo.qpId;

        void* regMem = nullptr;
        topsError_t err = topsRoceRegMem(portId, qpId,
                                         (void*)rdma_peer_base_addr[peerRank],
                                         (size_t)num_rdma_bytes,
                                         &regMem);
        EP_HOST_ASSERT(err == topsSuccess && "topsRoceRegMem failed for slave mode");

        uint64_t slavePa = 0;
        TOPS_CHECK(topsPointerGetAttribute(&slavePa, TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER, regMem));

        hostPrims[0].slave_rdma_base[peerRank][qp_index] = slavePa;
        slave_reg_addrs_.push_back(regMem);
      }
    }
    // Replicate slave_rdma_base to all actual channel/side entries in hostPrims
    for (int i = 1; i < numChannels * MAXSIDES; i++) {
      memcpy(hostPrims[i].slave_rdma_base, hostPrims[0].slave_rdma_base,
             sizeof(hostPrims[0].slave_rdma_base));
    }
    for (int i = 0; i < numChannels * MAXSIDES; i++) {
      memcpy(&prims[i].slave_rdma_base, &hostPrims[i].slave_rdma_base,
             sizeof(hostPrims[i].slave_rdma_base));
    }
  }

  TOPS_CHECK(topsDeviceSynchronize());
  free((void *)hostPrims);
}

void Buffer::destroy() {
    // Synchronize
    TOPS_CHECK(topsDeviceSynchronize());

#ifdef ENABLE_MORI_GCU
    if (DEEP_EP_MULTINODES(num_rdma_ranks) || is_internode) {
      NEQCHECK(epSuccess != ibgda_gcu::ibgdaFree(epComm, &epComm->ibgdaCookie), false);
    }
#endif

    // Unregister slave mode memory BEFORE tearing down the QPs (epComm destroy).
    // topsRoceUnregMem requires the port+QP used at registration to still be alive.
    for (auto regMem : slave_reg_addrs_) {
        topsError_t err = topsRoceUnregMem(regMem);
        if (err != topsSuccess) {
            fprintf(stderr, "topsRoceUnregMem(%p) failed: %s\n", regMem, topsGetErrorString(err));
        }
    }
    slave_reg_addrs_.clear();


    if (num_nvl_bytes > 0 &&  !is_internode) {
        // Barrier
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream, prims, cached_value_ptrs);
        TOPS_CHECK(topsDeviceSynchronize());

        // Free local buffer and error flag
        TOPS_CHECK(topsFree(buffer_ptrs[nvl_rank]));
    }

    if (epComm != nullptr) {
        deep_ep::destroy(epComm);
        epComm = nullptr;
    }
    TOPS_CHECK(topsFree(reinterpret_cast<void *>(prims)));
    TOPS_CHECK(topsFree(reinterpret_cast<void *>(cached_value_ptrs)));

    // Free NVSHMEM
#ifndef DISABLE_NVSHMEM
    if (is_available() and num_rdma_bytes > 0) {
        TOPS_CHECK(topsDeviceSynchronize());
        internode::barrier();
        internode::free(rdma_buffer_ptr);
        internode::finalize();
        TOPS_CHECK(topsFree(rdma_peer_base_addr_dev));

        TOPS_CHECK(topsFree(rdma_buffer_ptr));
    }

    if(rdma_ibgda_info_addr_dev) TOPS_CHECK(topsFree(rdma_ibgda_info_addr_dev));

    // Free QP shared state array
    if (qp_shared_states_addr_dev != nullptr) {
        TOPS_CHECK(topsFree(qp_shared_states_addr_dev));
        qp_shared_states_addr_dev = nullptr;
    }
#endif

    // Free workspace and MoE counter
    TOPS_CHECK(topsFree(workspace));
    auto free_err = topsFree(const_cast<int*>(moe_recv_counter));
    (void) free_err;

    // Free chunked mode staffs
    TOPS_CHECK(topsFree(const_cast<int*>(moe_recv_expert_counter)));

    destroyed = true;
    available = false;
}

void Buffer::sync(const std::vector<int> &device_ids,
                  const std::vector<uint64_t>& local_edfs,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());
    EP_HOST_ASSERT(root_unique_id_opt.has_value());

    // Sync IPC handles
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(static_cast<long unsigned int>(num_ranks) == device_ids.size());
        EP_HOST_ASSERT(device_ids.size() == local_edfs.size());
        if (!is_internode) {
            std::memcpy(buffer_ptrs_edf, local_edfs.data(), sizeof(void*) * num_nvl_ranks);
            for (int i = 0; i < num_nvl_ranks; ++ i) {
                barrier_signal_ptrs_edf[i] = reinterpret_cast<uint64_t>(reinterpret_cast<uint8_t*>(buffer_ptrs_edf[i]) + num_nvl_bytes);
            }
            // Copy all buffer and barrier signal pointers to GPU
            TOPS_CHECK(topsMemcpy(buffer_ptrs_gpu, buffer_ptrs_edf, sizeof(void*) * NUM_MAX_NVL_PEERS, topsMemcpyHostToDevice));
            TOPS_CHECK(topsMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs_edf, sizeof(int*) * NUM_MAX_NVL_PEERS, topsMemcpyHostToDevice));
            TOPS_CHECK(topsDeviceSynchronize());
        }
        else {
            std::memcpy(buffer_ptrs_edf, local_edfs.data() + rdma_rank * num_nvl_ranks, sizeof(void*) * num_nvl_ranks);
            for (int i = 0; i < num_nvl_ranks; ++ i) {
                barrier_signal_ptrs_edf[i] = reinterpret_cast<uint64_t>(reinterpret_cast<uint8_t*>(buffer_ptrs_edf[i]) + num_nvl_bytes);
            }
            // Copy all buffer and barrier signal pointers to GPU
            TOPS_CHECK(topsMemcpy(buffer_ptrs_gpu, buffer_ptrs_edf, sizeof(void*) * local_ranks, topsMemcpyHostToDevice));
            TOPS_CHECK(topsMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs_edf, sizeof(int*) * local_ranks, topsMemcpyHostToDevice));
            TOPS_CHECK(topsDeviceSynchronize());
        }
    }

    // Sync NVSHMEM handles and allocate memory
#ifndef DISABLE_NVSHMEM
    if (num_rdma_bytes > 0) {
        // Allocate RDMA buffer and metadata
        TOPS_CHECK(topsExtMallocWithFlags((void **)&rdma_buffer_ptr, num_rdma_bytes, topsMallocHostAccessable));
        TOPS_CHECK(topsMalloc(&rdma_peer_base_addr_dev, sizeof(uint64_t) * EP_MAX_RANKS));

        TOPS_CHECK(topsPointerGetAttribute(&rdma_peer_base_addr[rank], TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER, rdma_buffer_ptr));
        deep_ep::allGather(epComm, rdma_peer_base_addr, sizeof(uint64_t));
        TOPS_CHECK(topsMemcpy(rdma_peer_base_addr_dev, rdma_peer_base_addr, sizeof(uint64_t) * EP_MAX_RANKS, topsMemcpyHostToDevice));

        // Clean buffer (mainly for low-latency mode)
        TOPS_CHECK(topsMemset(rdma_buffer_ptr, 0, num_rdma_bytes));
    }

    TOPS_CHECK(topsMalloc(&rdma_ibgda_info_addr_dev, MAX_IBGDA_INFO_SIZE));

    // Allocate QP shared state array (for QP contention synchronization)
    // Layout: [rank0][sm0-31], [rank1][sm0-31], ..., [rank31][sm0-31]
    // Total: 32 ranks × 32 sm_ids × 12 bytes = 12,288 bytes (~12 KB)
    constexpr size_t QP_SHARED_STATE_SIZE = sizeof(struct qp_shared_state);
    constexpr size_t TOTAL_QP_SHARED_STATE_SIZE = EP_MAX_RANKS * MAX_IBGDA_QP_NUMS * QP_SHARED_STATE_SIZE;
    TOPS_CHECK(topsExtMallocWithFlags((void **)&qp_shared_states_addr_dev, TOTAL_QP_SHARED_STATE_SIZE, topsMallocHostAccessable));
    // Initialize QP shared states to zero
    TOPS_CHECK(topsMemset(qp_shared_states_addr_dev, 0, TOTAL_QP_SHARED_STATE_SIZE));

    TOPS_CHECK(topsDeviceSynchronize());
#endif
    // TODO: refine get_topo_info position and structure
    get_topo_info();

    // Ready to use
    available = true;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts,
                            std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = torch_gcu::getCurrentGCUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        torch_gcu::setCurrentGCUStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kPrivateUse1));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    layout::get_dispatch_layout(topk_idx.data_ptr<int64_t>(),
                                num_tokens_per_rank.data_ptr<int>(),
                                num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                num_tokens_per_expert.data_ptr<int>(),
                                is_token_in_rank.data_ptr<bool>(),
                                num_tokens, num_topk, num_ranks, num_experts,
                                num_rdma_ranks, num_device_sms, max_threads_per_sm,
                                comm_stream);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {num_tokens_per_rdma_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        torch_gcu::setCurrentGCUStream(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}


std::tuple<torch::Tensor, std::optional<torch::Tensor>,
std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>,
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::intranode_dispatch(const torch::Tensor& x,
        const std::optional<torch::Tensor>& x_scales,
        const std::optional<torch::Tensor>& topk_idx,
        const std::optional<torch::Tensor>& topk_weights,
        const std::optional<torch::Tensor>& num_tokens_per_rank,
        const torch::Tensor& is_token_in_rank,
        const std::optional<torch::Tensor>& num_tokens_per_expert,
        int cached_num_recv_tokens,
        const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
        const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
        int expert_alignment, int num_worst_tokens, const Config& config,
        std::optional<EventHandle>& previous_event, bool async,
        bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    // num_sms == 3: redundant-SIP dispatch (fixed 12 logical channels).
    // Otherwise classic: num_channels = num_sms / 2 (must be even).
    const bool use_redundant_sip = (config.num_sms == 3);
    if (!use_redundant_sip) {
        EP_HOST_ASSERT(config.num_sms % 2 == 0);
    }
    int num_channels = use_redundant_sip ? 12 : (config.num_sms / 2);

    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    int64_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<int64_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = torch_gcu::getCurrentGCUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        torch_gcu::setCurrentGCUStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;
    std::optional<torch::Tensor> actual_num_recv_tokens = std::nullopt;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        rank_prefix_matrix = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();

        // Copy rank prefix matrix and clean flags
        intranode::cached_notify_dispatch(rank_prefix_matrix.data_ptr<int>(), num_memset_int,
                                          buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, num_ranks,
                                          comm_stream, prims, cached_value_ptrs);
    } else {
        rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kPrivateUse1));
        channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kPrivateUse1));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++ i)
            moe_recv_expert_counter[i] = -1;
        EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int)
                             <= static_cast<uint64_t>(num_nvl_bytes));

        // When num_worst_tokens > 0,
        // allocate before notify so the kernel can write the actual count.
        int* actual_num_recv_tokens_ptr = nullptr;
        if (num_worst_tokens > 0) {
            actual_num_recv_tokens = torch::empty(
                {1}, dtype(torch::kInt32).device(torch::kPrivateUse1));
            actual_num_recv_tokens_ptr =
                actual_num_recv_tokens->data_ptr<int>();
        }

        // Todo. will be removed later
        intranode::notify_dispatch(
                        num_tokens_per_rank->data_ptr<int>(),
                        moe_recv_counter_mapped, num_ranks,
                        num_tokens_per_expert->data_ptr<int>(),
                        moe_recv_expert_counter_mapped, num_experts,
                        num_tokens, is_token_in_rank.data_ptr<bool>(),
                        channel_prefix_matrix.data_ptr<int>(),
                        rank_prefix_matrix.data_ptr<int>(),
                        num_memset_int, expert_alignment,
                        buffer_ptrs_gpu, barrier_signal_ptrs_gpu,
                        rank, prims, cached_value_ptrs,
                        comm_stream, num_channels, num_device_sms,
                        max_threads_per_sm, actual_num_recv_tokens_ptr);
        if (num_worst_tokens > 0) {
            // No CPU sync / host poll: allocate the worst case.
            // Actual count is written by notify_dispatch.
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        } else {
            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS)
                    throw std::runtime_error("DeepEP error: CPU recv timeout");
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Skip host TRACE when num_worst_tokens > 0 to avoid forced CPU sync.
    if (num_worst_tokens == 0) {
        const auto rank_prefix_matrix_cpu = rank_prefix_matrix.contiguous().cpu();
        const auto send_recv_tokens = compute_send_recv_tokens_from_prefix_matrix(rank_prefix_matrix_cpu, rank);
        int total_send_tokens_to_other_ranks = 0;
        int total_recv_tokens_from_other_ranks = 0;
        for (int peer_rank = 0; peer_rank < num_ranks; ++peer_rank) {
            if (peer_rank == rank)
                continue;
            total_send_tokens_to_other_ranks += send_recv_tokens.first[peer_rank];
            total_recv_tokens_from_other_ranks += send_recv_tokens.second[peer_rank];
        }
        DEEP_EP_TRACE(intranode_dispatch,
                      deep_ep::TraceJson()
                              .add("tokens", num_tokens)
                              .add("hidden", hidden)
                              .add("num_ranks", num_ranks)
                              .addRaw("send_tokens_per_rank", format_int32_vector_json(send_recv_tokens.first))
                              .addRaw("recv_tokens_per_rank", format_int32_vector_json(send_recv_tokens.second))
                              .add("total_send_tokens_to_other_ranks", total_send_tokens_to_other_ranks)
                              .add("total_recv_tokens_from_other_ranks", total_recv_tokens_from_other_ranks)
                              .finish());
    }
    // Inject jitter between notify_dispatch (Phase 1) and data
    // dispatch (Phase 2)
    // Controlled via env vars DEEP_EP_HOST_JITTER_MS / DEEP_EP_DEVICE_JITTER_MS
    auto host_jitter_ms = get_jitter_ms("DEEP_EP_HOST_JITTER_MS");
    auto device_jitter_ms = get_jitter_ms("DEEP_EP_DEVICE_JITTER_MS");
    if (device_jitter_ms > 0.0f) {
        auto jitter_ns = static_cast<unsigned int>(device_jitter_ms * 1e6f);
        fprintf(stderr,
                "[jitter] Rank %d: device sleep %.2f ms "
                "(intranode dispatch)\n",
                rank, device_jitter_ms);
        delay::launch_delay(jitter_ns, comm_stream);
    }
    if (host_jitter_ms > 0.0f) {
        auto jitter_us = static_cast<int64_t>(host_jitter_ms * 1000.0f);
        fprintf(stderr,
                "[jitter] Rank %d: host sleep %.2f ms "
                "(intranode dispatch)\n",
                rank, host_jitter_ms);
        std::this_thread::sleep_for(std::chrono::microseconds(jitter_us));
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(), recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kPrivateUse1));

    // Assign pointers
    int64_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ?
                        torch::empty({num_recv_tokens}, x_scales->options()) :
                        torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Dispatch
    EP_HOST_ASSERT(num_ranks * num_ranks * sizeof(int) +                                                                    // Size prefix matrix
                   num_channels * num_ranks * sizeof(int) +                                                                 // Channel start offset
                   num_channels * num_ranks * sizeof(int) +                                                                 // Channel end offset
                   num_channels * num_ranks * sizeof(int) * 2 +                                                             // Queue head and tail
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +     // Data buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                        // Source index buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(int64_t) +         // Top-k index buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +           // Top-k weight buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales           // FP8 scale buffer
                   <= static_cast<long unsigned int>(num_nvl_bytes));
    intranode::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_src_idx.data_ptr<int>(), recv_topk_idx_ptr, recv_topk_weights_ptr, recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                        is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
                        num_tokens, num_worst_tokens, static_cast<int>(hidden* recv_x.element_size() / sizeof(tops::__ef_bfloat16)),
                        num_topk, num_experts, num_scales,
                        scale_token_stride, scale_hidden_stride,
                        buffer_ptrs_gpu, rank, num_ranks, comm_stream, config.num_sms,
                        config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens, prims, cached_value_ptrs);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, recv_x, recv_src_idx, recv_channel_prefix_matrix, send_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream) {
                t.record_stream(compute_stream);
            }
        }
        for (auto& to : {x_scales, topk_idx, topk_weights, num_tokens_per_rank,
                num_tokens_per_expert, cached_channel_prefix_matrix,
                cached_rank_prefix_matrix, recv_topk_idx, recv_topk_weights,
                recv_x_scales, actual_num_recv_tokens}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        torch_gcu::setCurrentGCUStream(compute_stream);

    // Return values
    return {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
        num_recv_tokens_per_expert_list, rank_prefix_matrix,
        channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx,
        send_head, actual_num_recv_tokens, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::intranode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                          const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
                          const torch::Tensor& src_idx, const torch::Tensor& rank_prefix_matrix, const torch::Tensor& channel_prefix_matrix,
                          const torch::Tensor& send_head, int num_experts, const Config& config,
                          std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream,
                          const std::optional<torch::Tensor>& out) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and channel_prefix_matrix.scalar_type() == torch::kInt32);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    // num_sms == 3: redundant-SIP dispatch (fixed 12 logical channels).
    // Otherwise classic: num_channels = num_sms / 2 (must be even).
    const bool use_redundant_sip = (config.num_sms == 3);
    if (!use_redundant_sip) {
        EP_HOST_ASSERT(config.num_sms % 2 == 0);
    }
    int num_channels = use_redundant_sip ? 12 : (config.num_sms / 2);


    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    DEEP_EP_TRACE(intranode_combine,
                  deep_ep::TraceInfo().add("tokens", num_tokens).add("hidden", hidden));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(num_experts > 0 && num_experts % num_ranks == 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = torch_gcu::getCurrentGCUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        torch_gcu::setCurrentGCUStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        EP_HOST_ASSERT(num_topk <= 32);
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Launch barrier and reset queue head / tail / barrier (SymBuffer doubles for send+recv)
    const int num_memset_int = num_channels * num_ranks * 2 * 3;
    EP_HOST_ASSERT(num_memset_int * sizeof(int) <= static_cast<long unsigned int>(num_nvl_bytes));
    intranode::cached_notify_combine(buffer_ptrs_gpu, send_head.data_ptr<int>(),
                                     num_channels, num_recv_tokens, num_memset_int, num_experts,
                                     barrier_signal_ptrs_gpu, rank, num_ranks,
                                     comm_stream, prims, cached_value_ptrs);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++ i) if (bias_opts[i].has_value()) {
        auto bias = bias_opts[i].value();
        EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
        EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);
        bias_ptrs[i] = bias.data_ptr();
    }

    // Combine data: ring-buffer flow control requires send_tokens < recv_buffer_tokens
    // so a chunk cannot overrun unconsumed slots in the receive ring.
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens < config.num_max_nvl_chunked_recv_tokens);

    torch::Tensor recv_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(out->size(0) == num_recv_tokens and out->size(1) == hidden);
        recv_x = out.value();
    } else {
        recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    }
    int raw_bytes_per_token = hidden * static_cast<int>(x.element_size()) + num_topk * static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(int));
    int aligned_bytes_per_token = (raw_bytes_per_token + 127) & ~127;
    long unsigned int combine_buffer_need =
        static_cast<long unsigned int>(num_channels) * num_ranks * 2 * 3 * sizeof(int) +       // head + tail + barrier (SymBuffer doubled)
        static_cast<long unsigned int>(num_channels) * num_ranks * config.num_max_nvl_chunked_recv_tokens *
            aligned_bytes_per_token * 2;                                                        // data (SymBuffer doubled)
    EP_HOST_ASSERT(combine_buffer_need <= static_cast<long unsigned int>(num_nvl_bytes));
    intranode::combine(torch_gcu::optionalScalarTypeToTopsatenDataType(x.scalar_type()),
                       recv_x.data_ptr(), recv_topk_weights_ptr,
                       x.data_ptr(), topk_weights_ptr, bias_ptrs[0], bias_ptrs[1],
                       src_idx.data_ptr<int>(), rank_prefix_matrix.data_ptr<int>(), channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(), num_tokens, num_recv_tokens, hidden, num_topk,
                       buffer_ptrs_gpu, rank, num_ranks,
                       comm_stream, config.num_sms,
                       config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens, prims, cached_value_ptrs);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {topk_weights, recv_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        torch_gcu::setCurrentGCUStream(compute_stream);

    return {recv_x, recv_topk_weights, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::internode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank, const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                           const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                           const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                           int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
#ifndef DISABLE_NVSHMEM
    // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
    // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
    // unless we release GIL here.
    pybind11::gil_scoped_release release;
    const int num_used_sms = config.num_sms;
    EP_HOST_ASSERT(num_device_sms * max_threads_per_sm >= config.num_sms);
    const int num_channels = num_used_sms / 2;
    EP_HOST_ASSERT(num_used_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    // printf("cached_mode: %d\n", cached_mode);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
        // EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        // EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
        // EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)), hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    DEEP_EP_TRACE(internode_dispatch,
                  deep_ep::TraceInfo().add("tokens", num_tokens).add("hidden", hidden));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    int* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        // EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<int>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = torch_gcu::getCurrentGCUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        torch_gcu::setCurrentGCUStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

        // Just a barrier and clean flags
        // internode::cached_notify(hidden_int4, num_scales, num_topk, num_topk,
        //                          num_ranks, num_channels, 0, nullptr,
        //                          nullptr, nullptr, nullptr,
        //                          rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
        //                          buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
        //                          barrier_signal_ptrs_gpu, rank, comm_stream,
        //                          config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks, local_ranks),
        //                          num_nvl_bytes, true, low_latency_mode);
    } else {
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kPrivateUse1));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kPrivateUse1));
        gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kPrivateUse1));
        recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kPrivateUse1));

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++ i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
                                   num_nvl_ranks,
                                   num_tokens_per_rdma_rank->data_ptr<int>(), moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
                                   is_token_in_rank.data_ptr<bool>(), num_tokens, 0, num_channels,
                                   num_device_sms, max_threads_per_sm,
                                   hidden_int4, num_scales, num_topk, expert_alignment,
                                   rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                   gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                   rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                                   barrier_signal_ptrs_gpu, this->rank, comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks, local_ranks),
                                   num_nvl_bytes, low_latency_mode, prims);
        

        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            // Read total count
            num_recv_tokens = static_cast<int>(*moe_recv_counter);
            num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

            // Read per-expert count
            bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++ i)
                ready &= moe_recv_expert_counter[i] >= 0;

            if (ready)
                break;

            // Timeout check
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS) {
                printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
                for (int i = 0; i < num_local_experts; ++ i)
                    printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
            }
        }
        num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    // Inject jitter between notify_dispatch (Phase 1) and data
    // dispatch (Phase 2)
    // Controlled via env vars DEEP_EP_HOST_JITTER_MS / DEEP_EP_DEVICE_JITTER_MS
    auto host_jitter_ms = get_jitter_ms("DEEP_EP_HOST_JITTER_MS");
    auto device_jitter_ms = get_jitter_ms("DEEP_EP_DEVICE_JITTER_MS");
    if (device_jitter_ms > 0.0f) {
        auto jitter_ns = static_cast<unsigned int>(device_jitter_ms * 1e6f);
        fprintf(stderr,
                "[jitter] Rank %d: device sleep %.2f ms "
                "(internode dispatch)\n",
                rank, device_jitter_ms);
        delay::launch_delay(jitter_ns, comm_stream);
    }
    if (host_jitter_ms > 0.0f) {
        auto jitter_us = static_cast<int64_t>(host_jitter_ms * 1000.0f);
        fprintf(stderr,
                "[jitter] Rank %d: host sleep %.2f ms "
                "(internode dispatch)\n",
                rank, host_jitter_ms);
        std::this_thread::sleep_for(std::chrono::microseconds(jitter_us));
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(), recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_nvl_head = std::optional<torch::Tensor>();
    if (not cached_mode) {
        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kPrivateUse1));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kPrivateUse1));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kPrivateUse1));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kPrivateUse1));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    }

    // Assign pointers
    int* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        // printf("before recv_topk_idx: ");
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<int>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ?
                        torch::empty({num_recv_tokens}, x_scales->options()) :
                        torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }
    // uint8_t* nvl_buffer_offset = reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + 4096;
    // uint8_t* rdma_buffer_offset = reinterpret_cast<uint8_t*>(rdma_buffer_ptr) + 4096;
    // TOPS_CHECK(topsMemsetAsync(reinterpret_cast<void*>(nvl_buffer_offset), 0, num_nvl_bytes - 4096, comm_stream));
    // TOPS_CHECK(topsMemsetAsync(reinterpret_cast<void*>(rdma_buffer_offset), 0, num_rdma_bytes - 4096, comm_stream));


    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    internode::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_topk_idx_ptr, recv_topk_weights_ptr,
                        cached_mode ? nullptr : recv_src_meta->data_ptr(),
                        x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                        cached_mode ? nullptr : send_rdma_head->data_ptr<int>(), cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
                        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                        rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
                        gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
                        is_token_in_rank.data_ptr<bool>(),
                        num_tokens, hidden_int4, num_scales, num_topk, num_experts,
                        scale_token_stride, scale_hidden_stride,
                        rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu, barrier_signal_ptrs_gpu, config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens,
                        rank, num_ranks, num_nvl_ranks, cached_mode,
                        comm_stream, num_channels, low_latency_mode, prims);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, is_token_in_rank, recv_x,
                       rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {x_scales, topk_idx, topk_weights,
                        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
                        cached_rdma_channel_prefix_matrix, cached_recv_rdma_rank_prefix_sum,
                        cached_gbl_channel_prefix_matrix, cached_recv_gbl_rank_prefix_sum,
                        recv_topk_idx, recv_topk_weights, recv_x_scales,
                        recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix, send_rdma_head, send_nvl_head,
                        recv_src_meta}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        torch_gcu::setCurrentGCUStream(compute_stream);
    // uint8_t* nvl_buffer_offset = reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + 4096;
    // uint8_t* rdma_buffer_offset = reinterpret_cast<uint8_t*>(rdma_buffer_ptr) + 4096;
    // TOPS_CHECK(topsMemsetAsync(reinterpret_cast<void*>(nvl_buffer_offset), 0, num_nvl_bytes - 4096, comm_stream));
    // TOPS_CHECK(topsMemsetAsync(reinterpret_cast<void*>(rdma_buffer_offset), 0, num_rdma_bytes - 4096, comm_stream));

    // Return values
    return {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
            recv_src_meta, send_rdma_head, send_nvl_head, event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::internode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                          const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
                          const torch::Tensor& src_meta, const torch::Tensor& is_combined_token_in_rank,
                          const torch::Tensor& rdma_channel_prefix_matrix, const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
                          const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_nvl_head,
                          const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream,
                          const std::optional<torch::Tensor>& out) {
#ifndef DISABLE_NVSHMEM
    const int num_used_sms = config.num_sms;
    EP_HOST_ASSERT(num_device_sms * max_threads_per_sm >= config.num_sms);
    const int num_channels = num_used_sms / 2;
    EP_HOST_ASSERT(num_used_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)), hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    DEEP_EP_TRACE(internode_combine,
                  deep_ep::TraceInfo().add("tokens", num_tokens).add("hidden", hidden));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = torch_gcu::getCurrentGCUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        torch_gcu::setCurrentGCUStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Top-k checks
    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    internode::cached_notify(hidden_int4, 0, 0, num_topk,
                             num_ranks, num_nvl_ranks, num_channels, num_device_sms, max_threads_per_sm,
                             num_combined_tokens, combined_rdma_head.data_ptr<int>(),
                             rdma_channel_prefix_matrix.data_ptr<int>(), rdma_rank_prefix_sum.data_ptr<int>(), combined_nvl_head.data_ptr<int>(),
                             rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                             barrier_signal_ptrs_gpu, rank, comm_stream,
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks, local_ranks),
                             num_nvl_bytes, false, low_latency_mode);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++ i) if (bias_opts[i].has_value()) {
        auto bias = bias_opts[i].value();
        EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
        EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
        bias_ptrs[i] = bias.data_ptr();
    }

    // Launch data combine
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }
    internode::combine(torch_gcu::optionalScalarTypeToTopsatenDataType(x.scalar_type()),
                       combined_x.data_ptr(), combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_ptr<bool>(),
                       x.data_ptr(), topk_weights_ptr, bias_ptrs[0], bias_ptrs[1],
                       combined_rdma_head.data_ptr<int>(), combined_nvl_head.data_ptr<int>(),
                       src_meta.data_ptr(), rdma_channel_prefix_matrix.data_ptr<int>(), rdma_rank_prefix_sum.data_ptr<int>(), gbl_channel_prefix_matrix.data_ptr<int>(),
                       num_tokens, num_combined_tokens, hidden, num_topk,
                       rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
                       buffer_ptrs_gpu, config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens,
                       barrier_signal_ptrs_gpu,
                       rank, num_ranks, num_nvl_ranks, comm_stream, num_channels, low_latency_mode, prims);

    // uint8_t* nvl_buffer_offset = reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + 4096;
    // uint8_t* rdma_buffer_offset = reinterpret_cast<uint8_t*>(rdma_buffer_ptr) + 4096;
    // TOPS_CHECK(topsMemsetAsync(reinterpret_cast<void*>(nvl_buffer_offset), 0, num_nvl_bytes - 4096, comm_stream));
    // TOPS_CHECK(topsMemsetAsync(reinterpret_cast<void*>(rdma_buffer_offset), 0, num_rdma_bytes - 4096, comm_stream));
    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, src_meta,
                       is_combined_token_in_rank, rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
                       combined_x, combined_rdma_head, combined_nvl_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {topk_weights, combined_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        torch_gcu::setCurrentGCUStream(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::internode_notify_dispatch(const torch::Tensor& num_tokens_per_rank,
                                   const torch::Tensor& num_tokens_per_rdma_rank,
                                   const torch::Tensor& num_tokens_per_expert,
                                   const torch::Tensor& is_token_in_rank,
                                   int num_channels, int hidden_int4, int num_scales, int num_topk, int expert_alignment,
                                   const Config& config, std::optional<EventHandle>& previous_event, bool async) {
#ifndef DISABLE_NVSHMEM
    int num_used_sms = num_channels * 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);
    
    // Input validation
    EP_HOST_ASSERT(num_tokens_per_rank.dim() == 1 and num_tokens_per_rank.is_contiguous() and num_tokens_per_rank.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(num_tokens_per_rdma_rank.dim() == 1 and num_tokens_per_rdma_rank.is_contiguous() and num_tokens_per_rdma_rank.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(num_tokens_per_expert.dim() == 1 and num_tokens_per_expert.is_contiguous() and num_tokens_per_expert.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous() and is_token_in_rank.scalar_type() == torch::kBool);
    
    auto num_tokens = static_cast<int>(is_token_in_rank.size(0));
    auto num_experts = static_cast<int>(num_tokens_per_expert.size(0));
    EP_HOST_ASSERT(num_tokens_per_rank.size(0) == num_ranks);

    // Software-topology for notify_dispatch is derived from the input tensors.
    // This test hook should NOT be constrained by Buffer's internal topo inference (which is based on NUM_MAX_NVL_PEERS=32).
    const int sw_num_ranks = static_cast<int>(num_tokens_per_rank.size(0));
    const int sw_num_rdma_ranks = static_cast<int>(num_tokens_per_rdma_rank.size(0));
    EP_HOST_ASSERT(sw_num_ranks == num_ranks);
    EP_HOST_ASSERT(is_token_in_rank.size(1) == sw_num_ranks);
    EP_HOST_ASSERT(sw_num_rdma_ranks > 0);
    EP_HOST_ASSERT(sw_num_ranks % sw_num_rdma_ranks == 0);
    const int sw_num_nvl_ranks = sw_num_ranks / sw_num_rdma_ranks;

    printf("[internode_notify_dispatch] num_tokens_per_rank.size(0)=%d, buffer.num_ranks=%d\n", sw_num_ranks, num_ranks);
    printf("[internode_notify_dispatch] num_tokens_per_rdma_rank.size(0)=%d, buffer.num_rdma_ranks=%d (ignored for this test hook)\n",
           sw_num_rdma_ranks, num_rdma_ranks);
    printf("[internode_notify_dispatch] derived sw_num_nvl_ranks=%d\n", sw_num_nvl_ranks);
    printf("[internode_notify_dispatch] is_token_in_rank.size(1)=%d\n", static_cast<int>(is_token_in_rank.size(1)));
    fflush(stdout);
    
    // Allocate output tensors
    auto rdma_channel_prefix_matrix = torch::empty({sw_num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    auto recv_rdma_rank_prefix_sum = torch::empty({sw_num_rdma_ranks}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    auto gbl_channel_prefix_matrix = torch::empty({sw_num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    auto recv_gbl_rank_prefix_sum = torch::empty({sw_num_ranks}, dtype(torch::kInt32).device(torch::kPrivateUse1));
    
    // Use the existing host-side counters from Buffer class
    // These are already allocated with topsHostMalloc and mapped via topsHostGetDevicePointer
    // Initialize counters to -1 (not ready)
    *moe_recv_counter = -1;
    if (sw_num_rdma_ranks > 0) {
        *moe_recv_rdma_counter = -1;
    }
    auto num_local_experts = num_experts / sw_num_ranks;
    for (int i = 0; i < num_local_experts; ++i) {
        moe_recv_expert_counter[i] = -1;
    }
    
    // Debug: Print rank before calling notify_dispatch
    printf("[internode_notify_dispatch] Before notify_dispatch: this->rank=%d (0x%x), this->num_ranks=%d, num_channels=%d\n",
           this->rank, this->rank, this->num_ranks, num_channels);
    fflush(stdout);
    
    // Allocate all tensors on comm stream if set
    auto compute_stream = torch_gcu::getCurrentGCUStream();
    if (async) {
        EP_HOST_ASSERT(previous_event.has_value());
        torch_gcu::setCurrentGCUStream(comm_stream);
    }
    
    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }
    
    // Call internode::notify_dispatch
    // Use the existing mapped pointers from Buffer class
    // Save rank to a local variable to prevent corruption
    const int saved_rank = this->rank;
    if (saved_rank < 0 || saved_rank >= this->num_ranks) {
        printf("[internode_notify_dispatch] ERROR: Invalid rank! this->rank=%d (0x%x) is out of range [0, %d)\n",
               saved_rank, saved_rank, this->num_ranks);
        printf("[internode_notify_dispatch] Buffer member vars: rank=%d, num_ranks=%d, rdma_rank=%d, nvl_rank=%d\n",
               this->rank, this->num_ranks, this->rdma_rank, this->nvl_rank);
        fflush(stdout);
        EP_HOST_ASSERT(false && "Invalid rank in internode_notify_dispatch");
    }
    
    printf("[internode_notify_dispatch] Calling notify_dispatch with rank=%d (0x%x), num_ranks=%d, num_channels=%d\n",
           saved_rank, saved_rank, this->num_ranks, num_channels);
    fflush(stdout);
    
    internode::notify_dispatch(
        num_tokens_per_rank.data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
        sw_num_nvl_ranks,
        num_tokens_per_rdma_rank.data_ptr<int>(), moe_recv_rdma_counter_mapped,
        num_tokens_per_expert.data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
        is_token_in_rank.data_ptr<bool>(), num_tokens, 0, num_channels, num_used_sms, max_threads_per_sm,
        hidden_int4, num_scales, num_topk, expert_alignment,
        rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
        gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
        rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
        buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
        barrier_signal_ptrs_gpu, saved_rank, comm_stream,
        config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks, local_ranks),
        num_nvl_bytes, low_latency_mode, prims);
    
    // Wait for completion
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }
    
    // Switch back compute stream
    if (async) {
        torch_gcu::setCurrentGCUStream(compute_stream);
    }
    
    return {rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= static_cast<long unsigned int>(num_rdma_bytes));
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
    check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

    internode_ll::clean_low_latency_buffer(clean_meta_0.first, clean_meta_0.second,
                                           clean_meta_1.first, clean_meta_1.second,
                                           torch_gcu::getCurrentGCUStream());
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                             int num_max_dispatch_tokens_per_rank, int num_experts,
                             bool use_fp8, bool round_scale, bool use_ue8m0,
                             bool async, bool return_recv_hook,
                             const std::optional<torch::Tensor>& fp8_quant_scale) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    // EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(num_max_dispatch_tokens_per_rank <= LOW_LATENCY_MAX_TOKENS);
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    // Check token count buffer capacity
    EP_HOST_ASSERT(num_experts <= LOW_LATENCY_MAX_EXPERTS and
                   "Number of experts exceeds the allocated buffer size. "
                   "Please increase LOW_LATENCY_MAX_EXPERTS in configs.h.");

    // Diagnosis tensors
    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    DEEP_EP_TRACE(low_latency_dispatch,
                  deep_ep::TraceInfo().add("tokens", num_tokens).add("hidden", hidden));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks;

    // fp8_quant_scale validation and pointer extraction
    const float* fp8_quant_scale_ptr = nullptr;
    if (fp8_quant_scale.has_value()) {
        EP_HOST_ASSERT(use_fp8 and "fp8_quant_scale requires use_fp8=True");
        EP_HOST_ASSERT(not use_ue8m0 and "fp8_quant_scale is incompatible with use_ue8m0=True");
        EP_HOST_ASSERT(fp8_quant_scale->scalar_type() == torch::kFloat32 and "fp8_quant_scale must be float32");
        EP_HOST_ASSERT(fp8_quant_scale->is_contiguous());
        EP_HOST_ASSERT(fp8_quant_scale->numel() == 1 and "fp8_quant_scale must be a scalar tensor (numel==1), one value shared across all tokens");
        fp8_quant_scale_ptr = fp8_quant_scale->data_ptr<float>();
    }

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= static_cast<long unsigned int>(num_rdma_bytes));
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = torch_gcu::getCurrentGCUStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate packed tensors
    auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn: torch::kBFloat16));
    auto packed_recv_src_info = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kPrivateUse1));
    // The int64_t is limited to 32 bits on torch_gcu, so we use 2 int32_t to pack the count and begin_idx
    // So the shape is [num_local_experts, num_ranks * 2]
    // auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kPrivateUse1));
    auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks * 2}, torch::dtype(torch::kInt32).device(torch::kPrivateUse1));
    auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kPrivateUse1));

    // Allocate column-majored scales
    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kFloat32).device(torch::kPrivateUse1));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kInt).device(torch::kPrivateUse1));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();

    auto current_num_tokens_per_expert = reinterpret_cast<int*>(GET_NUM_TOKENS_PER_EXPERT_BUFFER(buffer));
    auto launcher = [=](int phases) {
        if (num_rdma_ranks == 1) {
            // Single node: select dispatch kernel based on hidden size.
            // hidden <= SLAVE_MODE_HIDDEN_THRESHOLD → slave mode (direct RDMA writes, lower latency)
            // hidden >  SLAVE_MODE_HIDDEN_THRESHO → master mode (ESL QP sends, higher throughput)
            // TODO: maybe there is a bug on superpod, so we disable the slave mode for now
            if (false && hidden <= SLAVE_MODE_HIDDEN_THRESHOLD) {
                intranode_ll_slave::dispatch(packed_recv_x.data_ptr(), packed_recv_x_scales_ptr,
                                       packed_recv_src_info.data_ptr<int>(),
                                       reinterpret_cast<int64_t*>(packed_recv_layout_range.data_ptr()),
                                       packed_recv_count.data_ptr<int>(),
                                       cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                                       dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                                       buffer.dispatch_rdma_recv_data_buffer, buffer.dispatch_rdma_recv_count_buffer,
                                       buffer.dispatch_rdma_send_buffer,
                                       rdma_buffer_ptr,
                                       x.data_ptr(), topk_idx.data_ptr<int64_t>(),
                                       next_clean_meta.first, next_clean_meta.second,
                                       num_tokens, hidden, num_max_dispatch_tokens_per_rank,
                                       num_topk, num_experts, rank, num_ranks,
                                       use_fp8, round_scale, use_ue8m0,
                                       fp8_quant_scale_ptr,
                                       workspace, current_num_tokens_per_expert, num_device_sms,
                                       max_threads_per_sm,
                                       launch_stream, phases, prims);
            } else {
                intranode_ll::dispatch(packed_recv_x.data_ptr(), packed_recv_x_scales_ptr,
                                       packed_recv_src_info.data_ptr<int>(),
                                       reinterpret_cast<int64_t*>(packed_recv_layout_range.data_ptr()),
                                       packed_recv_count.data_ptr<int>(),
                                       cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                                       dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                                       buffer.dispatch_rdma_recv_data_buffer, buffer.dispatch_rdma_recv_count_buffer,
                                       buffer.dispatch_rdma_send_buffer,
                                       rdma_buffer_ptr,
                                       x.data_ptr(), topk_idx.data_ptr<int64_t>(),
                                       next_clean_meta.first, next_clean_meta.second,
                                       num_tokens, hidden, num_max_dispatch_tokens_per_rank,
                                       num_topk, num_experts, rank, num_ranks,
                                       use_fp8, round_scale, use_ue8m0,
                                       fp8_quant_scale_ptr,
                                       workspace, current_num_tokens_per_expert, num_device_sms,
                                       max_threads_per_sm,
                                       launch_stream, phases, prims);
            }
        } else {
            // Multi-node: use internode_ll kernel
            internode_ll::dispatch(packed_recv_x.data_ptr(), packed_recv_x_scales_ptr,
                                   packed_recv_src_info.data_ptr<int>(),
                                   reinterpret_cast<int64_t*>(packed_recv_layout_range.data_ptr()),
                                   packed_recv_count.data_ptr<int>(),
                                   cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                                   dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                                   buffer.dispatch_rdma_recv_data_buffer, buffer.dispatch_rdma_recv_count_buffer,
                                   buffer.dispatch_rdma_send_buffer,
                                   rdma_buffer_ptr,
                                   x.data_ptr(), topk_idx.data_ptr<int64_t>(),
                                   next_clean_meta.first, next_clean_meta.second,
                                   num_tokens, hidden, num_max_dispatch_tokens_per_rank,
                                   num_topk, num_experts, rank, num_ranks,
                                   use_fp8, round_scale, use_ue8m0,
                                   fp8_quant_scale_ptr,
                                   workspace, current_num_tokens_per_expert, num_device_sms,
                                   max_threads_per_sm,
                                   launch_stream, phases, prims);
        }
    };
    // Jitter controlled via env vars
    // DEEP_EP_HOST_JITTER_MS / DEEP_EP_DEVICE_JITTER_MS
    auto host_jitter_ms = get_jitter_ms("DEEP_EP_HOST_JITTER_MS");
    auto device_jitter_ms = get_jitter_ms("DEEP_EP_DEVICE_JITTER_MS");
    bool jitter_needed =
        (host_jitter_ms > 0.0f || device_jitter_ms > 0.0f) && !return_recv_hook;
    if (jitter_needed) {
        launcher(LOW_LATENCY_SEND_PHASE);
        if (device_jitter_ms > 0.0f) {
            auto jitter_ns = static_cast<unsigned int>(device_jitter_ms * 1e6f);
            fprintf(stderr,
                    "[jitter] Rank %d: device sleep %.2f ms "
                    "(low-latency dispatch)\n",
                    rank, device_jitter_ms);
            delay::launch_delay(jitter_ns, launch_stream);
        }
        if (host_jitter_ms > 0.0f) {
            TOPS_CHECK(topsStreamSynchronize(launch_stream));
            auto jitter_us = static_cast<int64_t>(host_jitter_ms * 1000.0f);
            fprintf(stderr,
                    "[jitter] Rank %d: host sleep %.2f ms "
                    "(low-latency dispatch)\n",
                    rank, host_jitter_ms);
            std::this_thread::sleep_for(std::chrono::microseconds(jitter_us));
        }
        launcher(LOW_LATENCY_RECV_PHASE);
    } else {
        if (return_recv_hook) {
            launcher(LOW_LATENCY_SEND_PHASE);
        } else {
            launcher(LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE);
        }
    }

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                            const torch::Tensor& src_info, const torch::Tensor& layout_range,
                            const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
                            int num_max_dispatch_tokens_per_rank, int num_experts,
                            bool use_logfmt, bool zero_copy, bool async, bool return_recv_hook,
                            const std::optional<torch::Tensor>& out) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
    EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    // EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == 2 * num_ranks);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));
    DEEP_EP_TRACE(low_latency_combine,
                  deep_ep::TraceInfo().add("tokens", num_combined_tokens).add("hidden", hidden));

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= static_cast<long unsigned int>(num_rdma_bytes));
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];
    auto cached_value_buffer = reinterpret_cast<int*>(GET_CACHED_VALUE_PTRS(buffer));
    *cached_value_buffer = CACHED_VALUE_PTRS_VALUE;

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = torch_gcu::getCurrentGCUStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    int num_used_sms = deep_ep::epParamDebugCombineUseSm();
    if (num_used_sms > 0) {
        num_device_sms = num_used_sms;
        max_threads_per_sm = 1;
    }
    auto launcher = [=](int phases) {
        if (num_rdma_ranks == 1) {
            // Single node: use intranode_ll kernel
            intranode_ll::combine(combined_x.data_ptr(),
                                  buffer.combine_rdma_recv_data_buffer, buffer.combine_rdma_recv_flag_buffer,
                                  buffer.combine_rdma_send_buffer,
                                  x.data_ptr(), topk_idx.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
                                  src_info.data_ptr<int>(),
                                  reinterpret_cast<int64_t*>(layout_range.data_ptr()),
                                  combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                                  next_clean_meta.first, next_clean_meta.second,
                                  num_combined_tokens, hidden, num_max_dispatch_tokens_per_rank,
                                  num_topk, num_experts, rank, num_ranks,
                                  use_logfmt,
                                  workspace, num_device_sms, max_threads_per_sm,
                                  launch_stream, phases, zero_copy, prims, cached_value_buffer, rdma_buffer_ptr);
        } else {
            // Multi-node: use internode_ll kernel
            internode_ll::combine(combined_x.data_ptr(),
                                  buffer.combine_rdma_recv_data_buffer, buffer.combine_rdma_recv_flag_buffer,
                                  buffer.combine_rdma_send_buffer,
                                  x.data_ptr(), topk_idx.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
                                  src_info.data_ptr<int>(),
                                  reinterpret_cast<int64_t*>(layout_range.data_ptr()),
                                  combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                                  next_clean_meta.first, next_clean_meta.second,
                                  num_combined_tokens, hidden, num_max_dispatch_tokens_per_rank,
                                  num_topk, num_experts, rank, num_ranks,
                                  use_logfmt,
                                  workspace, num_device_sms, max_threads_per_sm,
                                  launch_stream, phases, zero_copy, prims, cached_value_buffer, rdma_buffer_ptr);
        }
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {combined_x, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

torch::Tensor
Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
#ifndef DISABLE_NVSHMEM
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                            {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(torch::kPrivateUse1));
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

} // namespace deep_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6, py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6, py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint,
             py::arg("hidden_bytes"), py::arg("num_ranks"), py::arg("num_local_ranks"))
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint,
             py::arg("hidden_bytes"), py::arg("num_ranks"), py::arg("num_local_ranks"));
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, const std::vector<uint8_t>&>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_edf", &deep_ep::Buffer::get_local_edf)
        .def("get_unique_id", &deep_ep::Buffer::get_unique_id)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("get_moe_recv_info", &deep_ep::Buffer::get_moe_recv_info,
             py::arg("num_local_experts"),
             "Return (moe_recv_counter, moe_recv_rdma_counter, moe_recv_expert_counter[0:num_local_experts]).")
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine,
             py::arg("x"), py::arg("topk_weights"), py::arg("bias_0"), py::arg("bias_1"),
             py::arg("src_idx"), py::arg("rank_prefix_matrix"), py::arg("channel_prefix_matrix"),
             py::arg("send_head"), py::arg("num_experts"), py::arg("config"),
             py::arg("previous_event"), py::arg("async"), py::arg("allocate_on_comm_stream"),
             py::arg("out") = py::none())
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine,
             py::arg("x"), py::arg("topk_weights"), py::arg("bias_0"), py::arg("bias_1"),
             py::arg("src_meta"), py::arg("is_combined_token_in_rank"),
             py::arg("rdma_channel_prefix_matrix"), py::arg("rdma_rank_prefix_sum"), py::arg("gbl_channel_prefix_matrix"),
             py::arg("combined_rdma_head"), py::arg("combined_nvl_head"),
             py::arg("config"), py::arg("previous_event"), py::arg("async"), py::arg("allocate_on_comm_stream"),
             py::arg("out") = py::none())
        .def("internode_notify_dispatch", &deep_ep::Buffer::internode_notify_dispatch,
             py::arg("num_tokens_per_rank"), py::arg("num_tokens_per_rdma_rank"),
             py::arg("num_tokens_per_expert"), py::arg("is_token_in_rank"),
             py::arg("num_channels"), py::arg("hidden_int4"), py::arg("num_scales"),
             py::arg("num_topk"), py::arg("expert_alignment"), py::arg("config"),
             py::arg("previous_event") = py::none(), py::arg("async_") = false)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("get_next_low_latency_combine_buffer", &deep_ep::Buffer::get_next_low_latency_combine_buffer);

    m.def("is_sm90_compiled", deep_ep::is_sm90_compiled);

    // Global function to get unique_id (can be called before Buffer creation)
    m.def("get_unique_id", []() {
        auto unique_id = deep_ep::get_unique_id();
        return pybind11::bytearray(reinterpret_cast<const char*>(unique_id.data()), unique_id.size());
    });
}
