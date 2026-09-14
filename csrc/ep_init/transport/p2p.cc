/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "transport_lare.h"
#include <string>
// #include "ras.h"


static_assert(sizeof(struct TransportConnInfo) <= sizeof(struct epConnect), "TransportConnInfo Connect Info is too big");
static int busIdToTopsDev(int64_t busId) {
  int ndev;
  if (topsGetDeviceCount(&ndev) != topsSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[EFML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (topsDeviceGetPCIBusId(devBusIdStr, EFML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != topsSuccess)
      return -1;
    int64_t devBusId;
    EP_CHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible tops devices
  return -1;
}

epResult_t p2pConnect(int* ret, struct epComm* comm, struct epTopoSystem* topo, struct epTopoGraph* graph,
                                                           struct epPeerInfo* myInfo, struct epPeerInfo* peerInfo) {
  // Rule out different nodes
  CheckPrintAndDo(ret != nullptr, return epInvalidArgument, "ret is nullptr\n");
  CheckPrintAndDo(topo != nullptr, return epInvalidArgument, "topo is nullptr\n");
  CheckPrintAndDo(myInfo != nullptr, return epInvalidArgument, "info1 is nullptr\n");
  CheckPrintAndDo(peerInfo != nullptr, return epInvalidArgument, "info2 is nullptr\n");

  EP_CHECK(epTopoCheckP2p(comm, topo, myInfo->rank, peerInfo->rank, ret, nullptr));
  if (*ret == 0) {
    return epSuccess;
  }

  // Convert the peer's busId into a local topsDev index (cf. TOPS_VISIBLE_DEVICES)
  int topsDev1 = busIdToTopsDev(myInfo->busId);
  int topsDev2 = busIdToTopsDev(peerInfo->busId);
  if (topsDev1 == -1 || topsDev2 == -1) {
    // Peer's tops device is not visible in this process : we can't communicate with it.
    return epSuccess;
  }

  // Check that TOPS can do P2P
  if (*ret == EP_P2P_PCIE) {
    INFO(EP_P2P, "PCIE transport not support");
    *ret = 0;
  } else {
    *ret = 1;
  }

  (void)graph;
  return epSuccess;
}

template <int SIDE>
static epResult_t p2pSetup(struct epComm* comm, struct epTopoGraph* graph,
                          struct epPeerInfo* myInfo, struct epPeerInfo* peerInfo,
                          struct epConnect* connectInfo, struct epConnector* connector, int channelId) {
  CheckPrintAndDo(comm != nullptr,
        return epInvalidArgument, "comm is nullptr\n");
  CheckPrintAndDo(graph != nullptr,
        return epInvalidArgument, "graph is nullptr\n");
  CheckPrintAndDo(myInfo != nullptr,
        return epInvalidArgument, "myInfo is nullptr\n");
  CheckPrintAndDo(peerInfo != nullptr,
        return epInvalidArgument, "peerInfo is nullptr\n");
  CheckPrintAndDo(connectInfo != nullptr,
        return epInvalidArgument, "connectInfo is nullptr\n");
  CheckPrintAndDo(connector != nullptr,
        return epInvalidArgument, "connector is nullptr\n");
  CheckPrintAndDo(channelId < MAXCHANNELS,
        return epInvalidArgument,
        "channelId(%d) is out of range(%d)\n", channelId, MAXCHANNELS);

  epTopoPort portInfo;
  EP_CHECK(epTopoComputePort(
        comm, graph, comm->rank, peerInfo->rank, channelId, &portInfo));
  TransportResource<SIDE>* resources = nullptr;
  switch(portInfo.linkType) {
    case EP_TOPO_PORT_LINK_LARE:
      resources = new TransportLare<SIDE>(comm, graph, channelId, portInfo);
      break;
    case EP_TOPO_PORT_LINK_PCIE:
    case EP_TOPO_PORT_LINK_NET:
    default:
      WARN("not support linktype[%u] for p2p setup", portInfo.linkType);
      return epInternalError;
  }
  EP_CHECK(resources->setup());

  TransportConnInfo * info = (TransportConnInfo *)connectInfo;
  EP_CHECK(resources->generate(*info));
  connector->transportResources = resources;
  return epSuccess;
}

template <int SIDE>
static epResult_t p2pConnect(struct epComm* comm, struct epTopoGraph* graph, struct epConnect* connectInfo,
                               int nranks, int rank, struct epConnector* connector, int channelId) {
  (void)nranks;
  CheckPrintAndDo(comm != nullptr, return epInvalidArgument, "comm is nullptr\n");
  CheckPrintAndDo(graph != nullptr, return epInvalidArgument, "graph is nullptr\n");
  CheckPrintAndDo(connectInfo != nullptr, return epInvalidArgument, "connectInfo is nullptr\n");
  CheckPrintAndDo(rank >= 0, return epInvalidArgument, "rank(%d) is invalid\n", rank);
  //CheckPrintAndDo(rank < nranks, return epInvalidArgument, "rank(%d) is out of range(%d)\n", rank, nranks);
  CheckPrintAndDo(connector != nullptr, return epInvalidArgument, "connector is nullptr\n");
  CheckPrintAndDo(channelId < MAXCHANNELS, return epInvalidArgument, "channelId(%d) is out of range(%d)\n", channelId, MAXCHANNELS);
  TransportResource<SIDE>* resources = (TransportResource<SIDE>*)connector->transportResources;
  TransportConnInfo* peerConnInfo = (TransportConnInfo*)connectInfo;

  EP_CHECK(resources->connect(*peerConnInfo, *connector));

  return epSuccess;
}

template <int SIDE>
epResult_t p2pFree(struct epConnector* conn) {
  CheckPrintAndDo(conn != nullptr, return epInvalidArgument, "conn is nullptr\n");
  CheckPrintAndDo(conn->transportResources != nullptr, return epInvalidArgument, "transportResources is nullptr\n");
  TransportResource<SIDE>* resource = (TransportResource<SIDE>*)conn->transportResources;
  epResult_t ret = resource->teardown();
  delete resource;
  return ret;
}

template epResult_t p2pSetup<SEND>(struct epComm* comm, struct epTopoGraph* graph,
                          struct epPeerInfo* myInfo, struct epPeerInfo* peerInfo,
                          struct epConnect* connectInfo, struct epConnector* connector, int channelId);
template epResult_t p2pSetup<RECV>(struct epComm* comm, struct epTopoGraph* graph,
                          struct epPeerInfo* myInfo, struct epPeerInfo* peerInfo,
                          struct epConnect* connectInfo, struct epConnector* connector, int channelId);
template epResult_t p2pConnect<SEND>(struct epComm* comm, struct epTopoGraph* graph, struct epConnect* connectInfo,
                                   int nranks, int rank, struct epConnector* connector, int channelId);
template epResult_t p2pConnect<RECV>(struct epComm* comm, struct epTopoGraph* graph, struct epConnect* connectInfo,
                                   int nranks, int rank, struct epConnector* connector, int channelId);
template epResult_t p2pFree<SEND>(struct epConnector* conn);
template epResult_t p2pFree<RECV>(struct epConnector* conn);

struct epTransport p2pTransport = {
  "P2P",
  p2pConnect,
  { p2pSetup<SEND>, p2pConnect<SEND>, p2pFree<SEND>, nullptr, nullptr, nullptr, nullptr, nullptr },
  { p2pSetup<RECV>, p2pConnect<RECV>, p2pFree<RECV>, nullptr, nullptr, nullptr, nullptr, nullptr }
};
