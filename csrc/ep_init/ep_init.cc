/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "net.h"
#include "bootstrap.h"
#include "argcheck.h"
#include "graph.h"
#include "transport.h"
#include "channel.h"
#include "efmlwrap.h"
#include "alloc.h"
#include "mnlare.h"
#include "exception.h"

#ifdef ENABLE_TRACE
std::chrono::high_resolution_clock::time_point epEpoch;
#endif

// Compute the maximum number of channels for a given rank count, based on the
// per-ESL-port QP budget.  This must match the meshGraph.maxChannels computation in
// initTransportsRank; being a standalone function allows reuse before epTopoCompute.
//   budget  = 128 for N>64  (larger clusters share ports with other components)
//             kMaxQpPerLare (248) for N≤64
//   formula = budget × kEslPorts / (nranks × MAXSIDES)  (floor)
static int epChannelCap(int nranks) {
  constexpr int kEslPorts           = 16;
  constexpr int kMaxSides           = 2;  // MAXSIDES
  constexpr int kSmallClusterCap    = 64;
  constexpr int kLargeClusterBudget = 128;
  constexpr int kMaxQpPerLare       = 248; // MAX_QP_PER_LARE from transport/lare_roce_ctxt.cc
  int budget = (nranks > kSmallClusterCap) ? kLargeClusterBudget : kMaxQpPerLare;
  int cap    = budget * kEslPorts / (nranks * kMaxSides);
  if (cap > EP_MAX_MESHX_CHANNELS) cap = EP_MAX_MESHX_CHANNELS;
  if (cap < 1)                     cap = 1;
  return cap;
}

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static epResult_t epInit() {
  if (initialized) return epSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    EP_CHECK(epNetInit());
    INFO(EP_INIT, "Using network %s", epNetName());
    initialized = true;
  }
  pthread_mutex_unlock(&initLock);
  return epSuccess;
}

static epResult_t commFree(epComm_t comm) {
  if (comm == NULL) return epSuccess;

  epCfree(comm->clique.ranks);
  TOPS_CHECK(topsFree(comm->hostDevComm->channelsHostPtr));

  free(comm->hostDevComm);
  free(comm->connectSend);
  free(comm->connectRecv);
  free(comm->peerInfo);
  epTopoGraphFree(&comm->meshGraph);
  if (comm->topo) epTopoFree(comm->topo);

  // EP_CHECK(epAllToAllvHostDestroy(comm));

  if (comm->nodeRanks) {
    for (int n=0; n<comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);
    free(comm->nodeRanks);
  }
  free(comm->rankToNode);
  free(comm->rankToLocalRank);
  free(comm->railRanks);

  if (comm->bootstrap) EP_CHECK(bootstrapClose(comm->bootstrap));
  free(comm->bootstrapIfAddresses);

  for (int channel = 0; channel < MAXCHANNELS; channel++)
    EP_CHECK(freeChannel(comm->efmlArch, comm->channels + channel, comm->nRanks));

  free(comm->topParentRanks);
  free(comm->topParentLocalRanks);
  EP_CHECK(epNetFinalize(comm));

  for (int c = 0; c < epChannelCap(comm->nRanks); c++) {
    EP_CHECK(comm->peerDirectBuffAddrs[c][comm->rank].dealloc());
  }

  EP_CHECK(epTopsFree(comm->abortFlag));

  if (comm->internalStream != NULL) {
    TOPS_CHECK(topsStreamDestroy(comm->internalStream));
    comm->internalStream = NULL;
  }

  comm->rank = comm->topsDev = comm->busId = comm->nRanks = -1;

  free(comm);
  return epSuccess;
}

static epResult_t commCleanup(epComm_t comm) {
  int savedDevice;
  TOPS_CHECK(topsGetDevice(&savedDevice));
  int commDevice = comm->topsDev;

  if (savedDevice != commDevice) {
    TOPS_CHECK(topsSetDevice(commDevice));
  }

  EP_CHECK(epTransportP2pTeardown(comm));
  EP_CHECK(commFree(comm));

  if (savedDevice != commDevice)
    TOPS_CHECK(topsSetDevice(savedDevice));
  return epSuccess;
}

static epResult_t commReclaim(epComm_t comm) {
  epResult_t ret = epSuccess;
  auto efmlArch = comm->efmlArch;
  if (comm->intraComm0 != NULL) {
    int curRankCnt;
    int curRank; /* Debug info */
    int intraRanks = comm->intraRanks;
    epComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = __atomic_add_fetch(finalizeRankCnt, 1, __ATOMIC_ACQ_REL);
    if (curRankCnt == intraRanks) {
      epComm_t curIntraComm;
      epComm_t nextIntraComm = intracomm0;

      /* free local resources. */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if ((ret = commCleanup(curIntraComm)) != epSuccess) {
          // We pass a freed pointer, but we don't dereference; we merely print its value, so it's OK.
          // coverity[pass_freed_arg]
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  } else {
    // intraComm0 is NULL: this comm was never linked into an intra-process chain
    // (single-comm-per-process scenario). Clean up directly.
    if ((ret = commCleanup(comm)) != epSuccess) {
      WARN("commReclaim: direct cleanup comm %p rank %d failed, error %d", comm, comm->rank, ret);
    }
  }
  if (efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    EP_CHECK(epEfmlShutdown());
  }

  return epSuccess;
}

static epResult_t commAlloc(epComm_t* comret, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return epInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return epInvalidArgument;
  }

  struct epComm* comm;
  EP_CHECK(epCalloc(&comm, 1));
  EP_CHECK(epCalloc(&comm->hostDevComm, 1));

  TOPS_CHECK(topsStreamCreate(&comm->internalStream));

  comm->rank = comm->hostDevComm->rank = rank;
  comm->nRanks = comm->hostDevComm->nRanks = ndev;
  TOPS_CHECK(topsGetDevice(&comm->topsDev));
  EP_CHECK(epCalloc(&comm->peerInfo, comm->nRanks));
  EP_CHECK(initGcuInfo(comm->topsDev, comm->peerInfo[rank].gcuInfo, comm->capability));
  EP_CHECK(busIdToInt64(comm->peerInfo[rank].gcuInfo.busIdStr, &comm->busId));
  comm->efmlArch = comm->peerInfo[rank].gcuInfo.efmlArch;
  TRACE(EP_INIT,"comm %p rank %d nranks %d topsDev %d busId %x efmlArch %u", comm, rank, ndev, comm->topsDev, comm->busId, comm->efmlArch);

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  EP_CHECK(epCalloc(&comm->connectSend, comm->nRanks));
  EP_CHECK(epCalloc(&comm->connectRecv, comm->nRanks));

  comm->epNet = epNet;

  if (comm->topParentRanks == NULL) {
    EP_CHECK(epCalloc(&comm->topParentRanks, comm->nRanks));
    for (int i = 0; i < comm->nRanks; ++i)
      comm->topParentRanks[i] = i;
  }

  // Mark channels as non initialized.
  for (int c = 0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;

  EP_CHECK(epTopsCallocWithFlags(&comm->abortFlag, 1, topsMallocHostAccessable));

  // Allocate only for channels that will actually be created by ep_init (epChannelCap
  // uses the same formula as meshGraph.maxChannels below, but is usable pre-topo).
  for (int c=0;c<epChannelCap(comm->nRanks);c++) {
      // Fixed EP_PEER_DIRECT_BUFF_SIZE uint64 slots per channel per rank.
      EP_CHECK(comm->peerDirectBuffAddrs[c][rank].alloc<E_topsMalloc>(
          EP_PEER_DIRECT_BUFF_SIZE,
          comm->internalStream, comm->capability.buffAlignment,
          topsDeviceMallocDefault, 0xFF));
  }

  EP_CHECK(epCalloc(&comm->clique.ranks, comm->nRanks));

  *comret = comm;
  return epSuccess;
}

static epResult_t fillInfo(struct epComm* comm, struct epPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  TOPS_CHECK(topsGetDevice(&info->topsDev));
  info->efmlDev = comm->efmlDev;
  info->hostHash = getHostHash() + commHash;
  info->pidHash = getPidHash() + commHash;
  info->busId = comm->busId;
  info->comm = comm;
#ifdef ENABLE_MORI_GCU  // TBD
  EP_CHECK(epGcuGdrSupport(&info->gdrSupport));
#endif
  info->gcuInfo = comm->peerInfo[comm->rank].gcuInfo;
  return epSuccess;
}

// DeepEP no need any fifo buff for now
// EP_PARAM(BuffSize, "BUFFSIZE", -2);
// EP_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);
// static epResult_t computeBuffSizes(struct epComm* comm) {
//   int64_t envs[EP_NUM_PROTOCOLS] = { epParamLl128BuffSize(), epParamBuffSize() };
//   for (int p = 0; p < EP_NUM_PROTOCOLS; p++) {
//     comm->buffSizes[p] = comm->hostDevComm->buffSizes[p] = envs[p] != -2 ? envs[p] : comm->capability.defaultBuffSize[p];
//   }
//   return epSuccess;
// }

// MNLARE: Flag to indicate whether to enable Multi-Node LARE
EP_PARAM(MNLareEnable, "MNLARE_ENABLE", 2);
EP_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
static epResult_t initTransportsRank(struct epComm* comm, epUniqueId* commId) {
  // EP_CHECK(epTransportP2pInit(comm));
  // We use 2 AllGathers
  // 1. { peerInfo }
  // 2. { nChannels, graphInfo, topoRanks }

  epResult_t ret = epSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int *nodesFirstRank = NULL;
  uint64_t commHash = getHash(commId->internal, EP_UNIQUE_ID_BYTES);
  TRACE(EP_INIT, "comm %p, commHash %lx, rank %d nranks %d - BEGIN", comm, commHash, rank, nranks);
  EP_CHECK(bootstrapInit((struct epBootstrapHandle*)commId, comm));

  // ################################### AllGather1 - begin ###################################

  struct epPeerInfo* myInfo = comm->peerInfo + rank;
  EP_CHECK(fillInfo(comm, myInfo, commHash));
  EP_CHECK(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct epPeerInfo)));

  int nHosts = 1;

  // Do some sanity checks
  do {
    for (int i = 0; i < nranks; i++) {
      if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) nHosts++;
      if ((i != rank) && (comm->peerInfo[i].hostHash == myInfo->hostHash) &&
          (comm->peerInfo[i].busId == myInfo->busId)) {
        WARN("Duplicate GCU detected : rank %d and rank %d both on gcu device %x", rank, i, myInfo->busId);
        return epInvalidUsage;
      }
    }
  } while(0);

  if (comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    // Check for MNLARE support
    if ((nHosts > 1 && epParamMNLareEnable() != 0) || epParamMNLareEnable() == 1) {
      EP_CHECK(epMnlareCheck(comm));
    }

    if (nranks <= EP_MAX_LOCAL_RANKS) {
      // Allgather only for channels that will be created; epChannelCap pre-computes
      // the same cap that meshGraph.maxChannels (below) will use.
      for (int c=0;c<epChannelCap(nranks);c++) {
        EP_CHECK(bootstrapAllGather(comm->bootstrap, comm->peerDirectBuffAddrs[c],
                                    sizeof(comm->peerDirectBuffAddrs[c][0])));
      }
    }
  }

  // ################################### AllGather1 - end ###################################

  epTopoSystem* topo = nullptr;
  // Topo detection / System graph creation
  EP_CHECK(epTopoGetSystem(comm, &topo));
  // Compute paths between GCUs and NICs
  EP_CHECK(epTopoComputePaths(topo, comm));
  // Remove inaccessible GCUs and unused NICs
  EP_CHECK(epTopoTrimSystem(topo, comm));
  // Recompute paths after trimming
  EP_CHECK(epTopoComputePaths(topo, comm));
  // Init search
  EP_CHECK(epTopoSearchInit(topo));
  // Print final topology
  epTopoPrint(topo);

  struct epTopoGraph& meshGraph = comm->meshGraph;
  struct epTopoGraph* graphs[EP_NUM_ALGORITHMS] = {&meshGraph};

  meshGraph.nChannels = 0;
  meshGraph.pattern = EP_TOPO_PATTERN_MESH;
  meshGraph.minChannels = 1;
  // Adaptive channel limit derived from the per-ESL-port QP budget.
  //   Hardware: 16 ESL ports/GCU, MAXSIDES=2
  //   Formula:  QPs/port = (N/16) × channels × 2 ≤ budget
  //             → max_channels = budget × 16 / (N × 2)  (floor)
  //
  // Budget tiers (shared with other components on the same ports):
  //   N ≤ 64 : 248 QPs/port  (kMaxQpPerLare=248; small cluster, full budget)
  //   N > 64 : 128 QPs/port  (larger cluster; reserve headroom for other users)
  //
  // Representative points (budget=128 for N>64):
  //   N= 64: ⌊1024/ 64⌋=16→14(cap) → (64/16)×14×2=112 ≤ 128 ✓
  //   N=128: ⌊1024/128⌋= 8         → (128/16)× 8×2=128 ✓
  //   N=192: ⌊1024/192⌋= 5         → (192/16)× 5×2=120 ✓
  //   N=256: ⌊1024/256⌋= 4         → (256/16)× 4×2=128 ✓
  //   N=512: ⌊1024/512⌋= 2         → (512/16)× 2×2=128 ✓  (reserved)
  // epChannelCap() encapsulates the QP-budget formula; see its definition near top of file.
  meshGraph.maxChannels = epChannelCap(nranks);
  EP_CHECK(epTopoCompute(topo, &meshGraph));
  comm->meshnChannels = meshGraph.nChannels;
  epTopoPrintGraph(topo, &meshGraph);

  if (comm->rank == epParamGraphDumpFileRank()) {
    EP_CHECK(epTopoDumpGraphs(topo, EP_NUM_ALGORITHMS, graphs));
  }

  // ################################### AllGather2 - begin ###################################

  struct epGraphInfo {
    int nChannels;
    int pattern;
    float bwIntra;
    float bwInter;
    int typeIntra;
    int typeInter;
  };

  struct {
    int nChannels;
    struct epGraphInfo graphInfo[EP_NUM_ALGORITHMS];
    struct epTopoRanks topoRanks;
  } *allGather2Data;

  EP_CHECK(epCalloc(&allGather2Data, nranks));
  for (int a = 0; a < EP_NUM_ALGORITHMS; a++) {
    allGather2Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
    allGather2Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
    allGather2Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
    allGather2Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
    allGather2Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
    allGather2Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
  }

  comm->nChannels = comm->meshnChannels;

  EP_CHECK(epTopoPreset(comm, graphs, &allGather2Data[rank].topoRanks));
  EP_CHECK(bootstrapAllGather(comm->bootstrap, allGather2Data, sizeof(*allGather2Data)));

  // Determine nNodes, firstRanks, ...
  EP_CHECK(epCalloc(&nodesFirstRank, nranks));
  EP_CHECK(epCalloc(&comm->rankToNode, comm->nRanks));
  for (int r = 0; r < nranks; r++) {
    int node;
    int firstRank = allGather2Data[r].topoRanks.meshRecv[0];
    for (node = 0; node < comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
    }
    comm->rankToNode[r] = node;
  }

  struct epTopoRanks** allTopoRanks;
  EP_CHECK(epCalloc(&allTopoRanks, comm->nRanks));
  for (int i = 0; i < nranks; i++) {
    allTopoRanks[i] = &allGather2Data[i].topoRanks;
    // Make sure we align all ranks so that the tuning is consistent across ranks
    for (int a = 0; a < EP_NUM_ALGORITHMS; a++) {
      graphs[a]->nChannels = std::min(allGather2Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
      graphs[a]->bwIntra = std::min(allGather2Data[i].graphInfo[a].bwIntra, graphs[a]->bwIntra);
      graphs[a]->bwInter = std::min(allGather2Data[i].graphInfo[a].bwInter, graphs[a]->bwInter);
      graphs[a]->typeIntra = std::min(allGather2Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
      graphs[a]->typeInter = std::min(allGather2Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
      graphs[a]->nChannels = std::min(allGather2Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
    }
  }

  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  EP_CHECK(epCalloc(&comm->nodeRanks, comm->nNodes));
  EP_CHECK(epCalloc(&comm->rankToLocalRank, comm->nRanks));
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    EP_CHECK(epCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks));
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  TRACE(EP_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    ret = epInternalError;
    goto affinity_restore;
  }

  // fill rail rank array; rail ranks meanings the same local rank id in the node
  EP_CHECK(epCalloc(&comm->railRanks, comm->nNodes));
  for (int r = 0; r < comm->nRanks; r++) {
    int peerLocalRank = comm->rankToLocalRank[r];
    if (peerLocalRank == comm->localRank) {
      comm->railRanks[comm->rankToNode[r]] = r;
    }
  }

  /* Shift by node position
    * for example: assume 32 ranks on 4 nodes, rail 0: [0, 8, 16, 24]
    * rank 0:  [0, 8, 16, 24] -> [0, 8, 16, 24]
    * rank 8:  [0, 8, 16, 24] -> [8, 16, 24, 0]
    * rank 16: [0, 8, 16, 24] -> [16, 24, 0, 8]
    * rank 24: [0, 8, 16, 24] -> [24, 0, 8, 16]
  */
  leftRotate(comm->railRanks, comm->nNodes, comm->rankToNode[comm->rank]);

  INFO(EP_INIT, "comm %p rank %d nRanks %d nNodes %d localRanks %d localRank %d MNLARE %d",
        comm, rank, comm->nRanks, comm->nNodes, comm->localRanks, comm->localRank, comm->MNLARE);

  comm->hostDevComm->nChannels = comm->nChannels;

  free(allTopoRanks);
  free(nodesFirstRank);
  free(allGather2Data);

  // ################################### AllGather2 - end ###################################

  // Set Affinity to a CPU local the our GCU, so that all memory we allocate on the host is local.
  cpu_set_t affinitySave;
  EP_CHECKGOTO(epTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), ret, affinity_restore);
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }

  // EP_CHECK(computeBuffSizes(comm)); DeepEP no need any fifo buffer for now

  for (int c = 0; c < comm->meshnChannels; c++) {
    struct epChannel* channel = comm->channels + c;
    if (comm->nRanks == 1) continue;
    epChannelPeer* peers = epGetChanPeer(comm->efmlArch, false, channel);
    if (peers == nullptr) {
      // EP_CHECKGOTO(setupChannel(comm, c, rank, nranks, rings), ret, affinity_restore);
      EP_CHECKGOTO(initChannel(comm, c), ret, affinity_restore);
    }
    EP_CHECKGOTO(epTransportP2pConnect(comm, &meshGraph, channel,  comm->localRanks, channel->mesh.peerRanks,  comm->localRanks, channel->mesh.peerRanks), ret, affinity_restore);
  }
  EP_CHECKGOTO(epTransportP2pSetup(comm, &meshGraph), ret, affinity_restore);
  INFO(EP_INIT, "Connected mesh");
  TRACE(EP_INIT, "rank %d nranks %d - CONNECTED %d mesh", rank, nranks, comm->meshnChannels);

  // We should have allocated all buffers, collective fifos, ... we can restore the affinity.
affinity_restore:
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);

  if (ret != epSuccess) return ret;

  return epSuccess;
}

static epResult_t devCommSetup(epComm_t comm) {

  EP_CHECK(epTopsCallocWithFlags(&comm->hostDevComm->channelsHostPtr, comm->nChannels, topsMallocHostAccessable));

  uint64_t devChannels;
  TOPS_CHECK(topsPointerGetAttribute(&devChannels, TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER, comm->hostDevComm->channelsHostPtr));
  comm->hostDevComm->channels = (struct epDevChannel*)devChannels;

  if (comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    comm->isAllPeersLinkLare = true;
    for (int c = 0; c < comm->meshnChannels && comm->isAllPeersLinkLare; c++) {
      for (int peerRank = 0; peerRank < comm->localRanks; peerRank++) {
        if (peerRank == comm->rank) continue;
        if (comm->channels[c].peers[peerRank].send.conn.hostConnInfoV4Ptr->simple.contxt.transType != TRANSPORT_TYPE_LARE) {
          comm->isAllPeersLinkLare = false;
          break;
        }
        if (comm->channels[c].peers[peerRank].recv.conn.hostConnInfoV4Ptr->simple.contxt.transType != TRANSPORT_TYPE_LARE) {
          comm->isAllPeersLinkLare = false;
          break;
        }
      }
    }
  }

  for (int c = 0; c < comm->nChannels; c++) {
    epDevChannelPeer* peers = comm->hostDevComm->channelsHostPtr[c].peers;
    if (comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
      comm->hostDevComm->channelsHostPtr[c].peersV4DevPtr = comm->channels[c].peersV4.alignedDevPtr;
      peers = comm->channels[c].peersV4.alignedPtr;
    }
    memcpy(peers, comm->channels[c].devPeers, comm->nRanks * sizeof(struct epDevChannelPeer));
    memcpy(comm->hostDevComm->channelsHostPtr[c].mesh.devPeerRanks, comm->channels[c].mesh.peerRanks, EP_MAX_LOCAL_RANKS * sizeof(int));
    if (comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
      memcpy(comm->channels[c].peersP2PV4.alignedPtr, comm->channels[c].devPeersP2P, comm->nRanks * sizeof(struct epDevChannelPeer));
      comm->hostDevComm->channelsHostPtr[c].peersP2PV4DevPtr = comm->channels[c].peersP2PV4.alignedDevPtr;
      comm->hostDevComm->channelsHostPtr[c].stepPerSlice = comm->channels[c].stepPerSlice.alignedDevPtr;
    }
  }

  comm->hostDevComm->nLocalRanks = comm->localRanks;
  if (comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    TOPS_CHECK(topsPointerGetAttribute(&comm->hostDevComm->abortFlag,
                                                              TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER, comm->abortFlag));
    // meshnChannels is set after epTopoCompute; only initialize entries for actual channels.
    for (int c = 0; c < comm->meshnChannels; c++) {
      comm->devPeerDirectBuffAddrs[c][comm->rank] = comm->peerDirectBuffAddrs[c][comm->rank].alignedDevPtr;
    }
  }

  return epSuccess;
}

static epResult_t epCommInitRankSync(epComm_t* newcomm, int nranks, epUniqueId commId, int myrank, int topsDev) {
  epResult_t res;

  TOPS_CHECK(topsSetDevice(topsDev));
  EP_CHECKGOTO(commAlloc(newcomm, nranks, myrank), res, cleanup);
  EP_CHECKGOTO(initTransportsRank(*newcomm, &commId), res, cleanup);
  EP_CHECKGOTO(devCommSetup(*newcomm), res, cleanup);

  (*newcomm)->commHash = getHash(commId.internal, EP_UNIQUE_ID_BYTES);

  INFO(EP_INIT,"comm %p rank %d nranks %d topsDev %d busId %x - setup success!", *newcomm, myrank, nranks, (*newcomm)->topsDev, (*newcomm)->busId);

  return epSuccess;
cleanup:
  if ((*newcomm) && (*newcomm)->bootstrap) bootstrapAbort((*newcomm)->bootstrap);
  *newcomm = NULL;
  return res;
}


static epResult_t epCommInitRankDev(epComm_t* newcomm, int nranks, epUniqueId commId, int myrank, int topsDev) {
  epResult_t res;
  const char* env = epGetEnv("EP_COMM_ID");
  if (env && myrank == 0) {
    if(strlen(env) > 0) INFO(EP_INIT|EP_ENV, ENV_FORMAT_STR, "EP_COMM_ID", env);
    EP_CHECKGOTO(bootstrapCreateRoot((struct epBootstrapHandle*)&commId, true), res, end);
  }

  EP_CHECKGOTO(epInit(), res, end);

  EP_CHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, end);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = epInvalidArgument;
    goto end;
  }

  EP_CHECKGOTO(epCommInitRankSync(newcomm, nranks, commId, myrank, topsDev), res, end);

end:
  return res;
}

epResult_t epGetUniqueId(epUniqueId* out) {
  EP_CHECK(epInit());
  if (out == nullptr) {
    WARN("%s : %s argument is nullptr", "epGetUniqueId", "out");
    return epInvalidArgument;
  }
  epResult_t res = bootstrapGetUniqueId((struct epBootstrapHandle*)out);
  return res;
}

epResult_t epCommInitRank(epComm_t* newcomm, int nranks, epUniqueId commId, int myrank) {
  int topsDev;
  TOPS_CHECK(topsGetDevice(&topsDev));
  EP_CHECK(epCommInitRankDev(newcomm, nranks, commId, myrank, topsDev));
  INFO(EP_INIT,"comm %p rank %d dev: %d, nranks %d - Init COMPLETE", *newcomm, myrank, topsDev, nranks);
  return epSuccess;
}

epResult_t epCommDestroy(epComm_t comm) {
  if (comm == NULL) return epSuccess;

  TRACE(EP_INIT, "comm %p rank %d nRanks %d topsDev %d busId %x", comm, comm->rank, comm->nRanks, comm->topsDev, comm->busId);

  // Try and prevent a double free of the comm struct
  if (comm->rank == -1 || comm->nRanks <= 0 || comm->topsDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return epInvalidArgument;
  }

  EP_CHECK(commReclaim(comm));
  return epSuccess;
}

epResult_t epAllGather(epComm_t comm, void* data, size_t size) {
  EP_CHECK(PtrCheck(comm, "epAllGather", "comm"));
  EP_CHECK(PtrCheck(data, "epAllGather", "data"));
  return bootstrapAllGather(comm->bootstrap, data, size);
}

const char* epGetErrorString(epResult_t code) {
  switch (code) {
    case epSuccess                  : return "no error";
    case epUnhandledTopsError       : return "unhandled tops error (run with EP_DEBUG=INFO and EP_DEBUG_SUBSYS=ALL for details)";
    case epSystemError              : return "unhandled system error (run with EP_DEBUG=INFO and EP_DEBUG_SUBSYS=ALL for details)";
    case epInternalError            : return "EP process error";
    case epInvalidArgument          : return "input argument invalid (run with EP_DEBUG=INFO and EP_DEBUG_SUBSYS=ALL for details)";
    case epInvalidUsage             : return "configuration invalid (run with EP_DEBUG=INFO and EP_DEBUG_SUBSYS=ALL for details)";
    case epRemoteError              : return "remote process exited or there was a network error";
    case epInProgress               : return "EP operation in progress";
    default                           : return "unknown result code";
  }
}
