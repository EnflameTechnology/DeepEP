/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_TRANSPORT_H_
#define EP_TRANSPORT_H_

#include "devcomm.h"
#include "core.h"
#include "gcu_info.h"

#define NTRANSPORTS 1
#define TRANSPORT_P2P 0

extern struct epTransport* epTransports[NTRANSPORTS];

// Forward declarations
struct epComm;

struct epPeerInfo {
  int rank;
  int topsDev;
  int efmlDev;
  int gdrSupport;
  uint64_t hostHash;
  uint64_t pidHash;
  dev_t shmDev;
  int64_t busId;
  struct epComm* comm;
  struct epGcuInfo gcuInfo;
};

#define CONNECT_SIZE 512
struct epConnect {
  char data[CONNECT_SIZE];
};

struct epTransportComm {
  epResult_t (*setup)(struct epComm* comm, struct epTopoGraph* graph, struct epPeerInfo*, struct epPeerInfo*, struct epConnect*, struct epConnector*, int channelId);
  epResult_t (*connect)(struct epComm* comm, struct epTopoGraph* graph, struct epConnect*, int nranks, int rank, struct epConnector*, int channelId);
  epResult_t (*free)(struct epConnector*);
  epResult_t (*proxySharedInit)(struct epProxyConnection* connection, struct epProxyState* proxyState, int nChannels);
  epResult_t (*proxySetup)(struct epProxyConnection* connection, struct epProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  epResult_t (*proxyConnect)(struct epProxyConnection* connection, struct epProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  epResult_t (*proxyFree)(struct epProxyConnection* connection, struct epProxyState* proxyState);
  epResult_t (*proxyProgress)(struct epProxyState* proxyState, struct epProxyArgs*);
};

struct epTransport {
  const char name[4];
  epResult_t (*canConnect)(int*, struct epComm* comm, struct epTopoSystem* topo, struct epTopoGraph* graph, struct epPeerInfo*, struct epPeerInfo*);
  struct epTransportComm send;
  struct epTransportComm recv;
};

epResult_t epTransportP2pConnect(struct epComm* comm, struct epTopoGraph* graph, struct epChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend);
epResult_t epTransportP2pSetup(struct epComm* comm, struct epTopoGraph* graph);
epResult_t epTransportP2pTeardown(struct epComm* comm);
epResult_t epTransportP2pInit(struct epComm* comm);
#endif
