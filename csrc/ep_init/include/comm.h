/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_COMM_H_
#define EP_COMM_H_

#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>

#include "graph.h"
#include "devcomm.h"
#include "checks.h"
#include "p2p.h"
#include "transport.h"
#include "alloc.h"
#include "socket.h"
#include "ep_net.h"

struct cliqueInfo {
  int id;
  int size;
  int *ranks;
};

struct epChannel {
  struct epMesh mesh;

  // index of this channel
  int id;

  // Communication structures
  struct epChannelPeer* peers;
  struct epDevChannelPeer* devPeers;
  // For P2P only
  struct epChannelPeer* peersP2P;
  struct epDevChannelPeer* devPeersP2P;

  struct epDevMemDesc<uint64_t> stepPerSlice;

  // alloc device mem for peers[r].send/recv.conn and peersP2P[r].send/recv.conn
  struct epDevMemDesc<uint8_t> connInfoV4s[MAXCHANNELS];
  struct epDevMemDesc<epChannelConfig> configV4; //EP_MAX_OPS*2 entries

  struct epDevMemDesc<epDevChannelPeer> peersV4; //nranks entries
  struct epDevMemDesc<epDevChannelPeer> peersP2PV4; //nranks entries
  struct epDevMemDesc<int> userRanksV4; //nranks entries
};

struct epNodeRanks {
  int localRanks;
  int* localRankToRank;
};

struct epComm {
  struct epChannel channels[MAXCHANNELS];
  bool isAllPeersLinkLare; //TODO: used to check if use direct mode, it will be renamed in future

  struct epPeerInfo* peerInfo;
  struct epTopoSystem* topo;

  epTopoGraph meshGraph;

  void* bootstrap;
  // Bitmasks for epTransportP2pSetup
  int connect;
  uint64_t* connectSend;
  uint64_t* connectRecv;

  uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.

  uint64_t commHash;
  // My rank in the communicator
  int rank;
  // Number of GCUs in communicator
  int nRanks;
  // My tops device index
  int topsDev;
  int efmlDev; // my efml device index
  // My PCI bus ID in int format
  int64_t busId;
  cpu_set_t cpuAffinity; // CPU affinity of the GCU

  // struct epSharedResources* sharedRes;
  /* map to top parent ranks. */
  int* topParentRanks;
  int* topParentLocalRanks;

  int node;
  int nNodes;
  int localRank;
  int localRanks;
  int maxLocalRanks;
  int* rankToNode;
  int* rankToLocalRank;
  int* localRankToRank;
  int* railRanks; // ranks in the same rail; rail ranks meanings the same local rank id in the node
  // localRanks and localRanktoRank for all nodes
  struct epNodeRanks* nodeRanks;

  topsStream_t userStream;
  topsStream_t internalStream = nullptr; // EP-internal stream for asynchronous memory management, preserves graph mode execution
  bool userStreamSet;

  // Channels for collectives
  int nChannels;
  int meshnChannels;
  int p2pnChannels;
  int p2pnChannelsPerPeer;
  int p2pChannels[MAXCHANNELS];

  // Buffer sizes, fifo size actually
  // int buffSizes[EP_NUM_PROTOCOLS]; DeepEP no need any buff for now

  // Whether there has been a fatal error in this communicator.
  epResult_t fatalError;

  // Flag to ask EP kernels to abort
  /* volatile */ uint32_t *abortFlag;

  // Host copy of the devComm
  struct epDevComm *hostDevComm;

  // Intra-process sync
  struct epComm* intraComm0; // leader of intra-process comms (self possible)
  struct epComm* intraNext; // next of intra-process comms, intraComm0 is head
  int intraRank;
  int intraRanks;

  // shared structures for finalization
  int finalizeRankCnt;

  epNet_t* epNet;
  struct epProxyState* proxyState;
  union epSocketAddress *peerIfAddresses;
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
  union epSocketAddress *bootstrapIfAddresses;

  efmlDeviceArchitecture_t efmlArch;
  struct epGcuCapabity capability;

  struct epDevMemDesc<uint64_t> peerDirectBuffAddrs[EP_MAX_MESHX_CHANNELS][EP_MAX_LOCAL_RANKS];
  uint64_t* devPeerDirectBuffAddrs[EP_MAX_MESHX_CHANNELS][EP_MAX_LOCAL_RANKS];

  // MNLARE: Multi-Node LARE
  int MNLARE; // true when MNLARE is available
  struct cliqueInfo clique; // Our MNLARE clique information
  int cliqueRank; // Our rank within the MNLARE clique

  // for alltollv
  int agentInitFlag = 0;
  struct epAgentState* agentState = nullptr;

  void* ibgdaCookie = nullptr;
};

static inline epChannelPeer* epGetChanPeer(efmlDeviceArchitecture_t efmlArch, bool usingP2P,
                                                                          struct epChannel* channel, int peerOff=0) {
  if (efmlArch >= EFML_DEVICE_ARCH_GCU400 && usingP2P) {
    return channel->peersP2P + peerOff;
  } else {
    return channel->peers + peerOff;
  }
}

static inline epDevChannelPeer* epGetChanDevPeer(efmlDeviceArchitecture_t efmlArch, bool usingP2P,
                                                                          struct epChannel* channel, int peerOff=0) {
  if (efmlArch >= EFML_DEVICE_ARCH_GCU400 && usingP2P) {
    return channel->devPeersP2P + peerOff;
  } else {
    return channel->devPeers + peerOff;
  }
}

#endif
