/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
// #include "collectives.h"
#include "transport.h"

epResult_t initChannel(struct epComm* comm, int channelId) {
  struct epChannel* channel = comm->channels + channelId;
  if (channel->id != -1) return epSuccess;
  channel->id = channelId;

  // Communication structures with peers.
  EP_CHECK(epCalloc(&channel->peers, comm->nRanks + 1));
  EP_CHECK(epCalloc(&channel->devPeers, comm->nRanks + 1));

  // Save the peers information for p2p communication.
  if (comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    EP_CHECK(epCalloc(&channel->peersP2P, comm->nRanks + 1));
    EP_CHECK(epCalloc(&channel->devPeersP2P, comm->nRanks + 1));
    size_t connInfoSize = sizeof(struct epConnInfoV4);
    ALIGN_SIZE(connInfoSize, comm->capability.buffAlignment);
    // 2 represents two variables: channel->peers and channel->peersP2P
    size_t curChannelConnInfoSize = 2 * (comm->nRanks+1) * MAX_SIDES * connInfoSize;
    // peersP2POffset is the offset of channel->connInfoV4s, represents channel->peersP2P's base address
    size_t peersP2POffset = (comm->nRanks+1) * MAX_SIDES * connInfoSize;
    EP_CHECK(channel->connInfoV4s[channelId].alloc<E_topsMallocWithFlags>(curChannelConnInfoSize, comm->internalStream,
                                                        comm->capability.buffAlignment, topsMallocHostAccessable));
    for (int peerRank = 0;peerRank<comm->nRanks+1;peerRank++) {
      channel->peers[peerRank].send.conn.hostConnInfoV4Ptr = (struct epConnInfoV4*)(channel->connInfoV4s[channelId].alignedPtr + (peerRank * MAX_SIDES + SEND) * connInfoSize);
      channel->peers[peerRank].send.conn.devConnInfoV4Ptr = (struct epConnInfoV4*)(channel->connInfoV4s[channelId].alignedDevPtr + (peerRank * MAX_SIDES + SEND) * connInfoSize);
      channel->peers[peerRank].recv.conn.hostConnInfoV4Ptr = (struct epConnInfoV4*)(channel->connInfoV4s[channelId].alignedPtr + (peerRank * MAX_SIDES + RECV) * connInfoSize);
      channel->peers[peerRank].recv.conn.devConnInfoV4Ptr = (struct epConnInfoV4*)(channel->connInfoV4s[channelId].alignedDevPtr + (peerRank * MAX_SIDES + RECV) * connInfoSize);
      channel->peersP2P[peerRank].send.conn.hostConnInfoV4Ptr = (struct epConnInfoV4*)(channel->connInfoV4s[channelId].alignedPtr + peersP2POffset + (peerRank * MAX_SIDES + SEND) * connInfoSize);
      channel->peersP2P[peerRank].send.conn.devConnInfoV4Ptr = (struct epConnInfoV4*)(channel->connInfoV4s[channelId].alignedDevPtr + peersP2POffset + (peerRank * MAX_SIDES + SEND) * connInfoSize);
      channel->peersP2P[peerRank].recv.conn.hostConnInfoV4Ptr = (struct epConnInfoV4*)(channel->connInfoV4s[channelId].alignedPtr + peersP2POffset + (peerRank * MAX_SIDES + RECV) * connInfoSize);
      channel->peersP2P[peerRank].recv.conn.devConnInfoV4Ptr = (struct epConnInfoV4*)(channel->connInfoV4s[channelId].alignedDevPtr + peersP2POffset + (peerRank * MAX_SIDES + RECV) * connInfoSize);
    }
  }
  for (int i = 0; i < comm->nRanks + 1; ++i) {
    channel->peers[i].send.comm = comm;
    channel->peers[i].recv.comm = comm;
    if (comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
      channel->peersP2P[i].send.comm = comm;
      channel->peersP2P[i].recv.comm = comm;
    }
  }

  if (comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    EP_CHECK(channel->stepPerSlice.alloc<E_topsMalloc>(1, comm->internalStream, comm->capability.buffAlignment));
    EP_CHECK(channel->configV4.alloc<E_topsMallocWithFlags>( std::max(EP_MAX_WORK_ELEMENTS, EP_MAX_WORK_ELEMENTS_P2P) * EP_MAX_OPS,
                                                              comm->internalStream,
                                                              comm->capability.buffAlignment, topsMallocHostAccessable,
                                                              0xff));
    EP_CHECK(channel->peersV4.alloc<E_topsMallocWithFlags>(comm->nRanks, comm->internalStream,
                                                            comm->capability.buffAlignment, topsMallocHostAccessable));
    EP_CHECK(channel->peersP2PV4.alloc<E_topsMallocWithFlags>(comm->nRanks, comm->internalStream,
                                                            comm->capability.buffAlignment, topsMallocHostAccessable));
    EP_CHECK(channel->userRanksV4.alloc<E_topsMallocWithFlags>(comm->nRanks, comm->internalStream,
                                                            comm->capability.buffAlignment, topsMallocHostAccessable));
  } else {
    // EP_CHECK(epTopsHostCalloc(&channel->workFifo, EP_MAX_OPS));
  }
  return epSuccess;
}

epResult_t freeChannel(efmlDeviceArchitecture_t efmlArch, struct epChannel* channel, int nRanks) {
  if (channel->id == -1) return epSuccess;

  //plz see comments note for gcu400 above
  if (efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    EP_CHECK(channel->userRanksV4.dealloc());
    EP_CHECK(channel->peersP2PV4.dealloc());
    EP_CHECK(channel->peersV4.dealloc());
    EP_CHECK(channel->configV4.dealloc());
    EP_CHECK(channel->stepPerSlice.dealloc());
  } else {
    // EP_CHECK(epTopsHostFree(channel->workFifo));
  }

  // Free transport proxy resources
  for (int r = nRanks; r >= 0; r--) {
    struct epChannelPeer* peer = channel->peers + r;
    if (peer->send.transportResources) EP_CHECK(peer->send.transportComm->free(&peer->send));
    if (peer->recv.transportResources) EP_CHECK(peer->recv.transportComm->free(&peer->recv));
    if (efmlArch >= EFML_DEVICE_ARCH_GCU400) {
      struct epChannelPeer* peerP2P = channel->peersP2P + r;
      if (peerP2P->send.transportResources) EP_CHECK(peerP2P->send.transportComm->free(&peerP2P->send));
      if (peerP2P->recv.transportResources) EP_CHECK(peerP2P->recv.transportComm->free(&peerP2P->recv));
      for (int c = 0; c < MAXCHANNELS; c++) {
        EP_CHECK(channel->connInfoV4s[c].dealloc());
      }
      channel->peers[r].send.conn.hostConnInfoV4Ptr = nullptr;
      channel->peers[r].send.conn.devConnInfoV4Ptr = nullptr;
      channel->peers[r].recv.conn.hostConnInfoV4Ptr = nullptr;
      channel->peers[r].recv.conn.devConnInfoV4Ptr = nullptr;
      channel->peersP2P[r].send.conn.hostConnInfoV4Ptr = nullptr;
      channel->peersP2P[r].send.conn.devConnInfoV4Ptr = nullptr;
      channel->peersP2P[r].recv.conn.hostConnInfoV4Ptr = nullptr;
      channel->peersP2P[r].recv.conn.devConnInfoV4Ptr = nullptr;
    }
  }

  // Free the peer structures.
  free(channel->peers);
  free(channel->devPeers);
  if (efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    free(channel->peersP2P);
    free(channel->devPeersP2P);
  }
  return epSuccess;
}
