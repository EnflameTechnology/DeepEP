/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <mutex>
#include "transport.h"

#include "comm.h"
#include "bootstrap.h"
#include "timer.h"

extern struct epTransport p2pTransport;

struct epTransport* epTransports[NTRANSPORTS] = {
  &p2pTransport,
};

template <int type>
static epResult_t selectTransport(struct epComm* comm, struct epTopoGraph* graph, struct epConnect* connect, int channelId, int peer, int* transportType) {
  struct epPeerInfo* myInfo = comm->peerInfo+comm->rank;
  struct epPeerInfo* peerInfo = comm->peerInfo+peer;
  struct epChannelPeer* peers = epGetChanPeer(comm->efmlArch, graph->pattern == EP_TOPO_PATTERN_RING, &comm->channels[channelId]);
  struct epConnector* connector = (type == 1) ? &peers[peer].send : &peers[peer].recv;
  for (int t=0; t<NTRANSPORTS; t++) {
    struct epTransport *transport = epTransports[t];
    struct epTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    EP_CHECK(transport->canConnect(&ret, comm, comm->topo, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      EP_CHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId));
      if (transportType) *transportType = t;
      return epSuccess;
    }
  }
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return epSystemError;
}

epResult_t epTransportP2pConnect(struct epComm* comm, struct epTopoGraph* graph, struct epChannel* channel,
                                      int nrecv, int* peerRecv, int nsend, int* peerSend) {
  TRACE(EP_INIT, "nsend %d nrecv %d", nsend, nrecv);
  uint64_t mask = 1UL << channel->id;
  epChannelPeer* peers = epGetChanPeer(comm->efmlArch, graph->pattern == EP_TOPO_PATTERN_RING, channel);
  CheckPrintAndDo(peers != nullptr, return epSystemError, "channelId %d epGetChanPeer(%u, %u) failed\n",
                                                                    channel->id, comm->efmlArch, graph->pattern);
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || peers[peer].recv.connected) continue;
    comm->connectRecv[peer] |= mask;
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || peers[peer].send.connected) continue;
    comm->connectSend[peer] |= mask;
  }
  return epSuccess;
}

void dumpData(struct epConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i=0; i<(int)sizeof(struct epConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

EP_PARAM(ConnectRoundMaxPeers, "CONNECT_ROUND_MAX_PEERS", 128);
EP_PARAM(ReportConnectProgress, "REPORT_CONNECT_PROGRESS", 0);
#include <sys/time.h>
epResult_t epTransportP2pSetup(struct epComm* comm, struct epTopoGraph* graph) {
  epResult_t ret = epSuccess;
  struct epConnect** data; // Store intermediate send/recvData structs for connect
  struct epConnect** recvData = NULL; // Points to entries inside data for given recv connection within a channel
  struct epConnect** sendData = NULL; // Points to entries inside data for given send connection within a channel
  int done = 0;
  int maxPeers = epParamConnectRoundMaxPeers();

  struct timeval timeStart, timeLast;
  gettimeofday(&timeStart, NULL);
  timeLast = timeStart; // struct copy
  bool timeReported = false;

  EP_CHECK(epCalloc(&data, maxPeers));
  EP_CHECKGOTO(epCalloc(&recvData, maxPeers), ret, fail);
  EP_CHECKGOTO(epCalloc(&sendData, maxPeers), ret, fail);

  // First time initialization
  for (int i=1; i<comm->nRanks; i++) {
    int bootstrapTag = (i<<8) + (graph ? graph->pattern+1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint64_t recvMask = comm->connectRecv[recvPeer];
    uint64_t sendMask = comm->connectSend[sendPeer];

    // Data[i] contains all epConnect information for all send and receive connections with a given send and recv peer
    // This data is packed in the array based on the number of sendChannels and recvChannels connected with these peers
    // The first N entries contain recvData, connection information for recv connections
    // The next M entries contain sendData, connection information for send connections
    // It's not guaranteed that each entry of data has the same number of total or send/recv specific connections
    int p = i-(done+1);
    if (recvMask || sendMask) EP_CHECKGOTO(epCalloc(data+p, 2*MAXCHANNELS), ret, fail);
    recvData[p] = data[p];
    int sendChannels = 0, recvChannels = 0;
    int type;
    TIME_START(0);
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1UL<<c)) {
        EP_CHECKGOTO(selectTransport<0>(comm, graph, recvData[p]+recvChannels++, c, recvPeer, &type), ret, fail);
      }
    }
    TIME_STOP(0);
    TIME_START(1);
    sendData[p] = recvData[p]+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1UL<<c)) {
        EP_CHECKGOTO(selectTransport<1>(comm, graph, sendData[p]+sendChannels++, c, sendPeer, &type), ret, fail);
      }
    }
    TIME_STOP(1);

    TIME_START(2);
    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
        EP_CHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct epConnect)*(recvChannels+sendChannels)), ret, fail);
        EP_CHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct epConnect)*(recvChannels+sendChannels)), ret, fail);
        sendData[p] = data[p];
        recvData[p] = data[p]+sendChannels;
      }
    } else {
      if (recvChannels) EP_CHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct epConnect)*recvChannels), ret, fail);
      if (sendChannels) EP_CHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct epConnect)*sendChannels), ret, fail);
      if (sendChannels) EP_CHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct epConnect)*sendChannels), ret, fail);
      if (recvChannels) EP_CHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct epConnect)*recvChannels), ret, fail);
    }
    TIME_STOP(2);

    if (i-done == maxPeers || i == comm->nRanks-1) {
      // Loop until all channels with all ranks have been connected
      bool allChannelsConnected;
      allChannelsConnected = false;
      while (!allChannelsConnected) {
        allChannelsConnected = true;
        for (int j=done+1; j<=i; j++) {
          int recvPeer = (comm->rank - j + comm->nRanks) % comm->nRanks;
          int sendPeer = (comm->rank + j) % comm->nRanks;
          uint64_t recvMask = comm->connectRecv[recvPeer];
          uint64_t sendMask = comm->connectSend[sendPeer];

          int p = j-(done+1);
          int sendDataOffset = 0;
          int recvDataOffset = 0;
          for (int c=0; c<MAXCHANNELS; c++) {
            TIME_START(3);
            struct epChannelPeer* peers = epGetChanPeer(comm->efmlArch, graph->pattern == EP_TOPO_PATTERN_RING,
                                                                                                    &comm->channels[c]);
            struct epDevChannelPeer* devPeers = epGetChanDevPeer(comm->efmlArch, graph->pattern == EP_TOPO_PATTERN_RING,
                                                                                                    &comm->channels[c]);
            if (sendMask & (1UL<<c)) {
              struct epConnector* conn = &peers[sendPeer].send;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                EP_CHECKGOTO(conn->transportComm->connect(comm, graph, sendData[p] + sendDataOffset, 1, comm->rank, conn, c), ret, fail);
                if (ret == epSuccess) {
                  conn->connected = 1;
                  memcpy(&devPeers[sendPeer].connInfo[SEND], &conn->conn, sizeof(struct epConnInfo));
                } else if (ret == epInProgress) {
                  allChannelsConnected = false;
                }
              }
              sendDataOffset++;
            }
            TIME_STOP(3);

            // Start with recv channels
            TIME_START(4);
            if (recvMask & (1UL<<c)) {
              struct epConnector* conn = &peers[recvPeer].recv;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                EP_CHECKGOTO(conn->transportComm->connect(comm, graph, recvData[p] + recvDataOffset, 1, comm->rank, conn, c), ret, fail);
                if (ret == epSuccess) {
                  conn->connected = 1;
                  memcpy(&devPeers[recvPeer].connInfo[RECV], &conn->conn, sizeof(struct epConnInfo));
                } else if (ret == epInProgress) {
                  allChannelsConnected = false;
                }
              }
              recvDataOffset++;
            }
            TIME_STOP(4);
          }
          if (sendMask || recvMask) {
            free(data[p]);
            data[p] = NULL;
          }
        }
        if (epParamReportConnectProgress() && comm->rank == 0 && done > 0) {
          struct timeval now;
          gettimeofday(&now, NULL);
          if (((now.tv_sec - timeLast.tv_sec)*1.0 + (now.tv_usec-timeLast.tv_usec)*1e-6) > 1) {
            float elapsed = (now.tv_sec - timeStart.tv_sec)*1.0 + (now.tv_usec-timeStart.tv_usec)*1e-6;
            float remaining = elapsed*(comm->nRanks-done)/done;
            printf("%sP2p connect: %g%% Elapsed %d:%02d Remaining %d:%02d                                       ",
                timeReported ? "\r" : "", done*100.0/comm->nRanks, ((int)elapsed)/60, ((int)elapsed)%60, ((int)remaining)/60, ((int)remaining)%60);
            fflush(stdout);
            timeReported = true;
            timeLast = now; // struct copy;
          }
        }
      }
      done = i;
    }
  }

  /* We need to sync ranks here since some ranks might run too fast after connection setup
   * and start to destroy the connection after returning from this function; however, the
   * others might still be trying to connect and import the buffer. No sync can lead to invalid
   * shmem/tops buffer. In addition, we also clear all connect masks and free each connectInfo array */
  for (int i = 1; i < comm->nRanks; i++) {
    int bootstrapTag = (i << 8) + (1 << 7) + (graph ? graph->pattern + 1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    if (recvPeer != sendPeer) {
      if (comm->connectSend[sendPeer] != 0UL) EP_CHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL) EP_CHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectSend[sendPeer] != 0UL) EP_CHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL) EP_CHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
    } else {
      if (comm->connectSend[sendPeer] != 0UL || comm->connectRecv[recvPeer] != 0UL) {
        EP_CHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
        EP_CHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0UL;
  }

  TIME_PRINT("P2P Setup/Connect");
exit:
  for(int i=0; i<maxPeers; ++i){
    if(data[i]) free(data[i]);
  }
  free(data);
  if (sendData) free(sendData);
  if (recvData) free(recvData);

  return ret;
fail:
  goto exit;
}

epResult_t epTransportP2pTeardown(struct epComm* comm) {
  /*      TODO: workaround here to sleep 5 seconds for gcu400.
   * backgroud1: in communicating phase, for send side, it must be waiting for post recv from recv side.
   *            otherwise, the post recv packet may be dropped while lare send queues are deleted by commDestroy
   *            so that for send side, it need to sync here that indicate there is no packet from recv side any more.
   * background2: for broadcast root side, it will be sending a large data while commDestroy will delete the qp which 
   *              is not as we expected.
   */
  if (comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    sleep(5);
  }
  return epSuccess;
}
