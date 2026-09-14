#include <ctype.h>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include "graph.h"
#include "topo.h"
#include "search.h"
#include "xml.h"
#include "alloc.h"
#include "detector.h"
#include "transport.h"
#include "checks.h"
#include "net.h"
#include "debug.h"
#include "channel.h"
#include "bootstrap.h"


EP_PARAM(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);
epResult_t epTopoGetSystem(epComm* comm, epTopoSystem** system) {
  epXml* xml;
  EP_CHECK(xmlAlloc(&xml, EP_TOPO_XML_MAX_NODES));
  const char* xmlTopoFile = epGetEnv("EP_TOPO_FILE");
  if (xmlTopoFile && strlen(xmlTopoFile) > 0) {
    INFO(EP_GRAPH|EP_ENV, ENV_FORMAT_STR, "EP_TOPO_FILE", xmlTopoFile);
    EP_CHECK(xml->loadFromFile(xmlTopoFile));
  } else {
    // TODO : Try default XML topology location
    //INFO(EP_GRAPH, "Try default XML topology location");
  }

  // XML file have only one "system" tag, create it if not exist.
  if (xml->size() == 0) {
    epXmlNode* sys;
    EP_CHECK(xml->createNode("system", &sys))
    EP_CHECK(sys->attributes()->create("version", EP_TOPO_XML_VERSION));
  }

  // Detect system according to communicator. If user have read from a xml topology file,
  // gcu/net node will be add incremental, which means if one gcu exist in xml, this gcu will
  // not be detected.

  // Detect only the GCU managed by this process.  We'll get any others through XML fusion.
  struct epPeerInfo* myInfo = &comm->peerInfo[comm->rank];
  epTopoDetector detector(xml, &myInfo->gcuInfo);
  //gcu400 need to bring systemid
  epXmlNode* gcu = xml->findNode("gcu", "dev", std::to_string(myInfo->gcuInfo.efmlDevId).c_str());
  if (!gcu) {
    EP_CHECK(detector.fillGcu(&gcu));
  }
  if (gcu) {
    EP_CHECK(gcu->setAttr("rank", comm->rank));
    EP_CHECK(gcu->setAttr("gdr", comm->peerInfo[comm->rank].gdrSupport));
  }

  // Auto-detect NICs if needed.
  int netDevCount = 0;
  EP_CHECK(epNetDevices(&netDevCount));
  for (int n = 0; n < netDevCount; n ++) {
    epNetProperties_t props;
    EP_CHECK(epNetGetProperties(n, &props));
    char guid[MAX_STR_LEN + 1] = {0};
    int gdrSupport = props.ptrSupport & EP_PTR_TOPS;
    sprintf(guid, "%lu", props.guid);
    //gcu400 need to bring systemid
    epXmlNode* net = xml->findNode("net", "guid", guid);
    if (net == nullptr) {
      char* gcuBusId = nullptr;
      int r;
      for (r = 0; r < comm->nRanks; r ++) {
        if (comm->peerInfo[comm->rank].hostHash == comm->peerInfo[r].hostHash) break;
      }
      gcuBusId = comm->peerInfo[r].gcuInfo.busIdStr;

      EP_CHECK(detector.fillNet(props.pciPath, gcuBusId, &net));
      EP_CHECK(net->attributes()->create("name", props.name));
      EP_CHECK(net->attributes()->create("dev", n));
      EP_CHECK(net->attributes()->create("speed", props.speed));
      EP_CHECK(net->attributes()->create("port", props.port));
      EP_CHECK(net->attributes()->create("latency", props.latency));
      EP_CHECK(net->attributes()->create("guid", props.guid));
      EP_CHECK(net->attributes()->create("maxconn", props.maxComms));
      EP_CHECK(net->attributes()->create("gdr", gdrSupport));
    }
    INFO(EP_GRAPH, "NET/%s : GCU Direct RDMA %s for HCA %d '%s'",
         epNet->name, gdrSupport ? "Enabled" : "Disabled", n, props.name);
  }

  // Remove xml branches which have a node with keep="0",
  // typically when importing a topology.
  //EP_CHECK(epTopoTrimXml(xml));

  // XML topo fusion.
  int* localRanks = nullptr;
  int localRank = -1, nLocalRanks = 0;
  if (comm->MNLARE) {
    // MNLARE clique support
    nLocalRanks = comm->clique.size;
    localRank = comm->cliqueRank;
    localRanks = comm->clique.ranks;
  } else {
    // Intra-node fusion.  Much of the comm is not initialized yet at this point so we need to do our own calculations.
    EP_CHECK(epCalloc(&localRanks, comm->nRanks));
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank)
          localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
  }
  char* mem = nullptr;
  EP_CHECK(epCalloc(&mem, nLocalRanks * xmlMemSize(EP_TOPO_XML_MAX_NODES)));
  epXml*rankXml = (epXml*)(mem+xmlMemSize(EP_TOPO_XML_MAX_NODES)*localRank);
  *rankXml = *xml;
  rankXml->convertXml(xml->getNodeBase(), 1);

  // nLocalRanks can't actually be 0, or we wouldn't be running at all...
  // coverity[divide_by_zero]
  EP_CHECK(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, localRank, nLocalRanks, mem, xmlMemSize(EP_TOPO_XML_MAX_NODES)));
  if (comm->MNLARE) {
    // Ensure that we have enough room when fusing topos from multiple nodes.
    free(xml);
    EP_CHECK(xmlAlloc(&xml, nLocalRanks*EP_TOPO_XML_MAX_NODES));
  } else {
    // In the intra-node case there's no need to enlarge the topo xml.
    *xml->getNNodePtr() = 0;
  }

  for (int i = 0; i < nLocalRanks; i++) {
    epXml* peerXml = (epXml*)(mem+xmlMemSize(EP_TOPO_XML_MAX_NODES)*i);
    peerXml->convertXml(peerXml->getNodeBase(), 0);
    EP_CHECK(xml->fuseXml(peerXml));
  }

  // TODO: When inter-node, some node don't have gcu with rank=0
  xmlTopoFile = epGetEnv("EP_TOPO_DUMP_FILE");
  if (xmlTopoFile && comm->rank == epParamTopoDumpFileRank()) {
    INFO(EP_GRAPH, "comm->rank=%d, EP_TOPO_DUMP_FILE set by environment to %s",
         comm->rank, xmlTopoFile);
    EP_CHECK(xml->dumpToFile(xmlTopoFile));
  }

  // Transfer xml tree structure to topology graph structure.
  EP_CHECK(epCalloc(system, 1));
  EP_CHECK((*system)->getSystemFromXml(xml, comm->peerInfo[comm->rank].hostHash));
  if (!comm->MNLARE && localRanks) free(localRanks);
  if (mem) free(mem);
  free(xml);
  comm->topo = *system;
  return epSuccess;
}

epResult_t epTopoSortSystem(epTopoSystem* system);

void epTopoPrint(epTopoSystem* system) {
  INFO(EP_GRAPH, "=== System : maxBw %2.1f totalBw %2.1f ===", system->maxBw(),
                   system->totalBw());
  system->printLinks();
  system->printPathMatrix();
}

epResult_t epTopoComputePaths(struct epTopoSystem* system, struct epComm* comm) {
  int cpus = system->cpuCount();
  int gcus = system->gcuCount();
  int nets = system->netCount();
  int lareSwitches = system->getCount(LARESWT);

  // Remove everything in case we're re-computing.
  for (int t = 0; t < EP_TOPO_NODE_TYPES; t ++) {
    system->removePaths(static_cast<epTopoNodeType>(t));
  }

  // Set direct paths to CPUs. We need them in many cases.
  for (int c = 0; c < cpus; c ++) {
    EP_CHECK(system->setPaths(CPU, c));
  }

  // Set direct paths to GCUs.
  for (int g = 0; g < gcus; g ++) {
    EP_CHECK(system->setPaths(GCU, g));
  }

  // Set direct paths to NICs.
  for (int n = 0; n < nets; n ++) {
    EP_CHECK(system->setPaths(NET, n));
  }

  // Set direct paths to LareSwitches.
  for (int n = 0; n < lareSwitches; n ++) {
    EP_CHECK(system->setPaths(LARESWT, n));
  }

  // Update path for GCUs when we don't want to / can't use GCU Direct P2P
  for (int g=0; g<gcus; g++) {
    epTopoNode* gcu1;
    EP_CHECK(system->getNode(GCU, g, &gcu1));
    for (int p=0; p<gcus; p++) {
      int p2p;
      epTopoNode* gcu2;
      EP_CHECK(system->getNode(GCU, p, &gcu2));
      EP_CHECK(epTopoCheckP2p(comm, system, gcu1->gcu.rank, gcu2->gcu.rank, &p2p, nullptr));
      if (p2p == 0) {
        // Divert all traffic through the CPU
        int c;
        EP_CHECK(system->getLocalCpu(g, &c));
        system->addInterStep(CPU, c, GCU, p, GCU, g);
      }
    }

    if (comm == nullptr) continue;
    // Remove GCUs we can't (or don't want to) communicate with through P2P or SHM
    struct epPeerInfo* dstInfo = comm->peerInfo+gcu1->gcu.rank;
    for (int p=0; p<gcus; p++) {
      if (p == g) continue;
      epTopoNode* gcu2;
      EP_CHECK(system->getNode(GCU, p, &gcu2));
      struct epPeerInfo* srcInfo = comm->peerInfo+gcu2->gcu.rank;
      int p2p;
      EP_CHECK(epTransports[TRANSPORT_P2P]->canConnect(&p2p, comm, system, nullptr, srcInfo, dstInfo));
      // if (p2p == 0) {
      //   // Mark this peer as inaccessible. We'll trim it later.
      //   epTopoLinkList* path;
      //   EP_CHECK(system->getPath(GCU, p, GCU, g, &path));
      //   path->type = PATH_NET;
      // } else if (shm == 1 && (dstInfo->rank == comm->rank || srcInfo->rank ==  comm->rank)){
      //   comm->hMemStack.shmemUsed = true;
      //   }
      // }
    }
  }

  // Update paths for NICs (no GCU Direct)
  for (int n=0; n<nets; n++) {
    struct epTopoNode* net;
    EP_CHECK(system->getNode(NET, n, &net));

    for (int g=0; g<gcus; g++) {
      struct epTopoNode* gcu;
      EP_CHECK(system->getNode(GCU, g, &gcu));
      // Update path when we dont want to / can't use GCU Direct RDMA.
      int gdr;
      EP_CHECK(epTopoCheckGdr(system, gcu->gcu.rank, net->id, 0, &gdr));
      if (gdr == 0) {
        // We cannot use GCU Direct RDMA, divert all traffic through the CPU local to the GCU
        int localCpu;
        EP_CHECK(system->getLocalCpu(g, &localCpu));
        system->addInterStep(CPU, localCpu, NET, n, GCU, g);
        system->addInterStep(CPU, localCpu, GCU, g, NET, n);
      }
    }
  }

  return epSuccess;
}

void epTopoFree(epTopoSystem* system) {
  for (int t = 0; t < EP_TOPO_NODE_TYPES; t ++) {
    system->removePaths(static_cast<epTopoNodeType>(t));
  }
  free(system);
}

epResult_t epTopoTrimSystem(epTopoSystem* system, epComm* comm) {
  // the root of domain is min GCU index in the domain
  int *domains;
  int64_t *ids;
  int gcus = system->gcuCount();
  EP_CHECK(epCalloc(&domains, gcus));
  EP_CHECK(epCalloc(&ids, gcus));
  int myDomain = 0;
  for (int g = 0; g < gcus; g ++) {
    epTopoNode* gcu;
    EP_CHECK(system->getNode(GCU, g, &gcu));
    domains[g] = g;
    ids[g] = gcu->id;
    for (int p = 0; p < g; p ++) {
      if (gcu->paths[GCU][p].type < PATH_NET) {
        domains[g] = std::min(domains[g], domains[p]);
      }
    }
    if (gcu->gcu.rank == comm->rank) myDomain = domains[g];
  }

  // GCU count would decrease during loop
  for (int i = 0; i < gcus; i ++) {
    if (domains[i] == myDomain) continue;
    epTopoNode* gcu;
    int g;
    for (g = 0; g < system->gcuCount() /* This one varies over the loops */; g ++) {
      EP_CHECK(system->getNode(GCU, g, &gcu));
      if (gcu->id == ids[i]) break;
      else gcu = nullptr;
    }
    if (gcu == nullptr) {
      WARN("Could not find id GCU/%lx", ids[i]);
      free(domains);
      free(ids);
      return epInternalError;
    }
    EP_CHECK(system->removeNode(GCU, g));
  }

#ifndef ENABLE_MORI_GCU // TBD: mori NOT TRIM_UNUSD_NET
  // trim unusable NIC/NET
  if (system->gcuCount() == comm->nRanks) {
    for (int n = system->netCount() - 1; n >= 0; n --)
      EP_CHECK(system->removeNode(NET, n));
  }
#endif
  free(domains);
  free(ids);
  comm->localRanks = system->gcuCount();
  return epSuccess;
}

epResult_t epTopoSearchInit(struct epTopoSystem* system) {
  system->graphSearchInit();
  return epSuccess;
}

epResult_t epTopoCompute(struct epTopoSystem* system, struct epTopoGraph* graph) {
  switch(graph->pattern){
    case EP_TOPO_PATTERN_MESH:
      graph->hdl = new epMeshGraphHandler(system, graph);
      break;
    case EP_TOPO_PATTERN_RING:
    case EP_TOPO_PATTERN_BALANCED_TREE:
    default:
      graph->hdl = new epTopoGraphHdl(system, graph);
  }

  // First read from graph xml file.
  const char* str = epGetEnv("EP_GRAPH_FILE");
  if (str) {
    if (strlen(str) > 0) INFO(EP_GRAPH|EP_ENV, ENV_FORMAT_STR, "EP_GRAPH_FILE", str);
    epXml* xml;
    EP_CHECK(xmlAlloc(&xml, EP_GRAPH_XML_MAX_NODES));
    EP_CHECK(xml->loadFromFile(str));
    int channels;
    EP_CHECK(graph->hdl->getGraphFromXml(xml, &channels));
    INFO(EP_GRAPH, "Search %s : %d channels loaded from XML graph", graph->hdl->getPatternDesc(), channels);
    free(xml);
    graph->hdl->copyRaw(graph);
    if (channels > 0) return epSuccess;
  }

  // If don't load graph from xml, compute optimal graph.
  EP_CHECK(graph->hdl->compute());
  graph->hdl->copyRaw(graph);
  return epSuccess;
}

void epTopoGraphFree(epTopoGraph* graph) {
  if (graph->hdl) {
    delete(graph->hdl);
    graph->hdl = nullptr;
  }
}

void epTopoPrintGraph(struct epTopoSystem* /*system*/, struct epTopoGraph* graph) {
  graph->hdl->print();
}

// /** LareSwitch indices that have ports to both GCUs (forward and reverse link). */
// static size_t epTopoComputeLareSwitch(int forwardGcuIdx, int reverseGcuIdx, epTopoLink* forwardLink,
//                                       epTopoLink* reverseLink, std::vector<int>& vtLareSwitch) {
//   std::vector<int>().swap(vtLareSwitch);
//   for (int swId = 0; swId < EP_TOPO_MAX_LARESWTS; swId++) {
//     if (forwardLink->portSizeToLareSwitch[reverseGcuIdx][swId]
//         && reverseLink->portSizeToLareSwitch[forwardGcuIdx][swId]) {
//       vtLareSwitch.push_back(swId);
//     }
//   }
//   return vtLareSwitch.size();
// }

/**
 * Returns the sorted list of efml port IDs that are simultaneously active on
 * both the local GCU and the peer GCU (flat across all lare switches).
 *
 * Hardware guarantee: same-numbered ports on any two GCUs connect to the same
 * switch.  The intersection of both sides' flat active port sets therefore gives
 * exactly the ports that can be used symmetrically (same physical port number
 * on both ends).
 *
 * @param lareLink     Link info from local GCU's perspective.
 * @param reverseLink  Link info from peer GCU's perspective.
 * @param gcuIdx       Local GCU index (used to index into reverseLink).
 * @param peerGcuIdx   Peer GCU index (used to index into lareLink).
 * @return             Sorted vector of common active efml port IDs.
 */
static std::vector<int> epTopoGetCommonActivePorts(const epTopoLink* lareLink,
                                                   const epTopoLink* reverseLink,
                                                   int gcuIdx, int peerGcuIdx) {
  // Build set of peer GCU's active ports (flat list across all switches).
  const int peerCount = reverseLink->portCount[gcuIdx];
  std::unordered_set<int> peerPortSet;
  peerPortSet.reserve(peerCount);
  for (int i = 0; i < peerCount; i++) {
    int port = reverseLink->localEfmlPortId[gcuIdx][i];
    if (port >= 0) peerPortSet.insert(port);
  }

  // Keep only local ports that also appear in peer's set (same-numbered constraint).
  const int localCount = lareLink->portCount[peerGcuIdx];
  std::vector<int> common;
  for (int i = 0; i < localCount; i++) {
    int port = lareLink->localEfmlPortId[peerGcuIdx][i];
    if (port >= 0 && peerPortSet.count(port)) {
      common.push_back(port);
    }
  }
  std::sort(common.begin(), common.end());
  return common;
}

epResult_t epTopoComputePort(const epComm* comm, const epTopoGraph* graph, const int rank,
                                                     const int peerRank, int channelId, struct epTopoPort* portInfo) {
  int gcuIdx, peerGcuIdx;
  EP_CHECK(comm->topo->rankToIndex(rank, &gcuIdx));
  epTopoLinkList *path = nullptr;

  EP_CHECK(comm->topo->rankToIndex(peerRank, &peerGcuIdx));
  EP_CHECK(comm->topo->getPath(GCU, gcuIdx, GCU, peerGcuIdx, &path));
  epTopoLinkList *peerPath;
  EP_CHECK(comm->topo->getPath(GCU, peerGcuIdx, GCU, gcuIdx, &peerPath));
  CheckPrintAndDo(path->type == peerPath->type, return epInternalError,
                                        "local host srcGcu[%d] pathType[%d] <==> dstGcu[%d] pathType[%d] not equal\n", 
                                            gcuIdx, path->type, peerGcuIdx, peerPath->type);
  CheckPrintAndDo(path->type >= PATH_LARE || path->type < PATH_SYS, return epInternalError,
                                        "local host srcGcu[%d] pathType[%d] not support\n", 
                                            gcuIdx, path->type);

  if (path->type == PATH_LARE) {
    portInfo->count = 0;
    epTopoLink* lareLink = path->list[0];
    if (lareLink->portCount[peerGcuIdx] < 1) {
      WARN("Failed to alloc LarePort for rank[%d] channel[%d] => rank %d for ALGO : %d, lareLink->count[%d]",
           rank, channelId, peerRank, graph->pattern, lareLink->portCount[peerGcuIdx]);
      return epInternalError;
    }
    int maxPortsPerTrunk = MAX_PORTS_PER_TRUNK;
    epTopoNode* gcuNode = nullptr;
    EP_CHECK(comm->topo->getNode(GCU, gcuIdx, &gcuNode));
    if (SWITCH_CONNECTED == gcuNode->gcu.connectType) {
      epTopoLink* reverseLink = peerPath->list[0];

      // Build a flat sorted list of efml ports active on BOTH local and peer GCU
      // (same-numbered ports across all switches).
      std::vector<int> commonPorts =
          epTopoGetCommonActivePorts(lareLink, reverseLink, gcuIdx, peerGcuIdx);
      CheckPrintAndDo(!commonPorts.empty(), return epInternalError,
                      "No common active ports: local r[%d] gcu[%d] <-> remote r[%d] gcu[%d]\n",
                      rank, gcuIdx, peerRank, peerGcuIdx);

      // Determine portsPerPlane; fall back if we have fewer common ports than one plane.
      int portsPerPlane = comm->capability.portsPerPlane;
      if ((int)commonPorts.size() < portsPerPlane) {
        WARN("commonPorts.size()=%zu < portsPerPlane=%d for r[%d]<->r[%d], fallback portsPerPlane to %zu",
             commonPorts.size(), portsPerPlane, rank, peerRank, commonPorts.size());
        portsPerPlane = (int)commonPorts.size();
      }

      // Layer 1: plane selection by (rank + peerRank).
      // nPlanes = floor(commonPorts.size() / portsPerPlane); trailing ports discarded.
      int nPlanes = (int)commonPorts.size() / portsPerPlane;
      CheckPrintAndDo(nPlanes > 0, return epInternalError,
                      "nPlanes=0: local r[%d] gcu[%d] <-> remote r[%d] gcu[%d]: "
                      "%zu common ports, portsPerPlane=%d\n",
                      rank, gcuIdx, peerRank, peerGcuIdx, commonPorts.size(), portsPerPlane);
      int planeId = (rank + peerRank) % nPlanes;

      // Layer 2: trunk selection by channelId within the chosen plane.
      // nTrunksPerPlane = floor(portsPerPlane / maxPortsPerTrunk); trailing ports discarded.
      int nTrunksPerPlane = portsPerPlane / maxPortsPerTrunk;
      CheckPrintAndDo(nTrunksPerPlane > 0, return epInternalError,
                      "nTrunksPerPlane=0: local r[%d] gcu[%d] <-> remote r[%d] gcu[%d]: "
                      "portsPerPlane=%d, maxPortsPerTrunk=%d\n",
                      rank, gcuIdx, peerRank, peerGcuIdx, portsPerPlane, maxPortsPerTrunk);
      int trunkId = channelId % nTrunksPerPlane;

      // Assign the maxPortsPerTrunk consecutive ports belonging to this trunk within the plane.
      int planeBase = planeId * portsPerPlane;
      int trunkBase = planeBase + trunkId * maxPortsPerTrunk;
      for (int p = 0; p < maxPortsPerTrunk; p++) {
        int efmlPort = commonPorts[trunkBase + p];
#ifdef ENABLE_TRACE
        std::ostringstream oss;
        oss << "C[" << channelId << "][" << p << "]: plane[" << planeId << "/" << nPlanes
            << "] trunk[" << trunkId << "/" << nTrunksPerPlane
            << "] efmlPort[" << efmlPort << "] commonPorts:";
        for (int port : commonPorts) oss << " " << port;
        TRACE(EP_GRAPH, "Compute Port local<r[%2d] g[%2d]> peer<r[%2d] g[%2d]> T%d%s\n", rank, gcuIdx, peerRank,
               peerGcuIdx, graph->pattern, oss.str().c_str());
#endif
        portInfo->efmlPorts[p] = efmlPort;
        portInfo->count++;
      }
    } else {
      // Workaround for compatibility with supported legacy code that expects channelId to be divided by 2 for SWITCH_CONNECTED links.
      // will be removed after all code is updated to use the new plane/trunk logic.
      channelId = channelId / 2;
      if (lareLink->portCount[peerGcuIdx] < maxPortsPerTrunk) {
        WARN("Alloc LarePort for rank[%d] channel[%d] => rank %d for ALGO : %d, lareLink->count[%d] < maxPortsPerTrunk[%d], adjust maxPortsPerTrunk to lareLink->count",
             rank, channelId, peerRank, graph->pattern, lareLink->portCount[peerGcuIdx], maxPortsPerTrunk);
        maxPortsPerTrunk = lareLink->portCount[peerGcuIdx];
      }
      int port = 0;
      for (int i = 0; i < maxPortsPerTrunk; ++i) {
        if (graph->pattern == EP_TOPO_PATTERN_MESH) {
          int nTrunks = lareLink->portCount[peerGcuIdx] / maxPortsPerTrunk;
          int trunkId = channelId % nTrunks;
          port = trunkId * maxPortsPerTrunk + i;
        }

        portInfo->efmlPorts[i] = lareLink->localEfmlPortId[peerGcuIdx][port];
        portInfo->count++;
        TRACE(EP_INIT, "Alloc LarePort[%d] for rank[%d] channel[%d] => rank %d for ALGO : %d", portInfo->efmlPorts[i],
              rank, channelId, peerRank, graph->pattern);
      }
    }
    portInfo->linkType = EP_TOPO_PORT_LINK_LARE;
  } else {
    // only consider pcie, net may need re-construct
    portInfo->linkType = EP_TOPO_PORT_LINK_PCIE;
    portInfo->efmlPorts[0] = 0;
    portInfo->count = 1;
  }
  return epSuccess;
}

epResult_t epTopoGetNTrunks(const struct epComm* comm, int rank1, int rank2, int* nTrunks) {
  int gcuIdx1, gcuIdx2;
  EP_CHECK(comm->topo->rankToIndex(rank1, &gcuIdx1));
  EP_CHECK(comm->topo->rankToIndex(rank2, &gcuIdx2));
  epTopoLinkList *path, *reversePath;
  EP_CHECK(comm->topo->getPath(GCU, gcuIdx1, GCU, gcuIdx2, &path));
  EP_CHECK(comm->topo->getPath(GCU, gcuIdx2, GCU, gcuIdx1, &reversePath));
  CheckPrintAndDo(path->type == reversePath->type, return epInternalError,
                                        "local host srcGcu[%d] pathType[%d] <==> dstGcu[%d] pathType[%d] not equal\n",
                                            gcuIdx1, path->type, gcuIdx2, reversePath->type);
  CheckPrintAndDo(path->type >= PATH_LARE || path->type < PATH_SYS, return epInternalError,
                                        "local host srcGcu[%d] pathType[%d] not support\n",
                                            gcuIdx1, path->type);
  if (path->type == PATH_LARE) {
    epTopoNode* gcuNode1 = nullptr, *gcuNode2 = nullptr;
    EP_CHECK(comm->topo->getNode(GCU, gcuIdx1, &gcuNode1));
    EP_CHECK(comm->topo->getNode(GCU, gcuIdx2, &gcuNode2));
    CheckPrintAndDo(gcuNode1->gcu.connectType == gcuNode2->gcu.connectType, return epInternalError,
                    "Mismatched connectType Gcu[%d] connectType[%d] vs Gcu[%d] connectType[%d]\n", gcuIdx1,
                    gcuNode1->gcu.connectType, gcuIdx2, gcuNode2->gcu.connectType);

    TRACE(EP_GRAPH, "%s %d local r[%d] gcu[%d] type [%d] <-> remote r[%d] gcu[%d] type [%d] <SWITCH_CONNECTED=%d>\n",
          __FUNCTION__, __LINE__, rank1, gcuIdx1, gcuNode1->gcu.connectType, rank2, gcuIdx2, gcuNode2->gcu.connectType,
          SWITCH_CONNECTED);

    epTopoLink* lareLink = path->list[0];
    if (lareLink->portCount[gcuIdx2] < 1) {
      WARN("Failed to get the number of trunks for rank[%d] => rank %d, lareLink->count[%d]",
           rank1, rank2, lareLink->portCount[gcuIdx2]);
      return epInternalError;
    }
    int maxPortsPerTrunk = MAX_PORTS_PER_TRUNK;
    if (SWITCH_CONNECTED == gcuNode1->gcu.connectType) {
      epTopoLink* reverseLink = reversePath->list[0];
      std::vector<int> commonPorts =
          epTopoGetCommonActivePorts(lareLink, reverseLink, gcuIdx1, gcuIdx2);

      // Mirror the plane/trunk logic in epTopoComputePort exactly.
      int portsPerPlane = comm->capability.portsPerPlane;
      if ((int)commonPorts.size() < portsPerPlane) {
        portsPerPlane = (int)commonPorts.size();
      }
      // nTrunks reported here is nTrunksPerPlane — callers use this to
      // determine how many distinct channels can share a peer pair.
      *nTrunks = portsPerPlane / maxPortsPerTrunk;
      TRACE(EP_GRAPH, "GetNTrunks local r[%d] gcu[%d] <-> remote r[%d] gcu[%d], "
            "commonPorts=%zu portsPerPlane=%d nTrunksPerPlane=%d\n",
            rank1, gcuIdx1, rank2, gcuIdx2, commonPorts.size(), portsPerPlane, *nTrunks);
    } else {
      if (lareLink->portCount[gcuIdx2] < maxPortsPerTrunk) {
        WARN("Get the number of trunks for rank[%d] => rank %d, lareLink->count[%d] < maxPortsPerTrunk[%d], adjust maxPortsPerTrunk to lareLink->count",
             rank1, rank2, lareLink->portCount[gcuIdx2], maxPortsPerTrunk);
        maxPortsPerTrunk = lareLink->portCount[gcuIdx2];
      }
      *nTrunks = lareLink->portCount[gcuIdx2] / maxPortsPerTrunk;
    }
    if (*nTrunks < 1) {
      WARN("Get the number of trunks for rank[%d] => rank %d, lareLink->count[%d] / maxPortsPerTrunk[%d] < 1",
           rank1, rank2, lareLink->portCount[gcuIdx2], maxPortsPerTrunk);
      return epInternalError;
    }
  } else {
    *nTrunks = 1;
  }

  return epSuccess;
}

epResult_t epTopoDumpGraphs(struct epTopoSystem* /*system*/, int ngraphs,
                                struct epTopoGraph** graphs) {
  const char* str = epGetEnv("EP_GRAPH_DUMP_FILE");
  if (str) {
    if (strlen(str) > 0) INFO(EP_GRAPH|EP_ENV, ENV_FORMAT_STR, "EP_GRAPH_DUMP_FILE", str);
    epXml* xml;
    EP_CHECK(xmlAlloc(&xml, EP_GRAPH_XML_MAX_NODES));
    // Create "graphs" node first.
    epXmlNode* xmlGraphs;
    EP_CHECK(xml->createNode("graphs", &xmlGraphs));
    EP_CHECK(xmlGraphs->attributes()->create("version", EP_GRAPH_XML_VERSION));
    // Add All graphs.
    for(int i = 0; i < ngraphs; ++i) {
      graphs[i]->hdl->updateGraphInfo(graphs[i]);
      EP_CHECK(graphs[i]->hdl->getXmlFromGraph(xml));
    }
    EP_CHECK(xml->dumpToFile(str));
    free(xml);
  }
  return epSuccess;
}

epResult_t epTopoGetNetDev(epTopoSystem* system, int rank, struct epTopoGraph* graph, int channelId,
                                                                               int peerRank, int64_t* netId, int* dev) {
  (void)peerRank;
  int netDev = -1;
  if (graph) {
    // Get the net device in the graph
    EP_CHECK(graph->hdl->getNetDev(rank, channelId, netId, &netDev));
  }
  else {
    EP_CHECK(system->getLocalNet(rank, channelId, netId, &netDev));
  }

  if (dev) *dev = netDev;
  return epSuccess;
}
EP_PARAM(P2pDisable, "P2P_DISABLE", 0);
static const int levelsOldToNew[] = { PATH_LOC, PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS };
static epResult_t epGetLevel(int* level, const char* disableEnv, const char* levelEnv) {
  if (*level == -1) {
    int l = -1;
    if (disableEnv) {
      if (epParamP2pDisable()) l = 0;
    }
    if (l == -1) {
      const char* str = epGetEnv(levelEnv);
      if (str) {
        for (int i=0; i<=PATH_SYS; i++) {
          if (strcmp(str, topoPathTypeStr[i]) == 0) {
            l = i;
            break;
          }
        }
        // Old style numbering
        if (l == -1 && str[0] >= '0' && str[0] <= '9') {
          int oldLevel = strtol(str, nullptr, 0);
          const int maxOldLevel = sizeof(levelsOldToNew)/sizeof(int) - 1;
          if (oldLevel > maxOldLevel) oldLevel = maxOldLevel;
          l = levelsOldToNew[oldLevel];
        }
      }
    }
    if (l >= 0) INFO(EP_ALL, "%s set by environment to %s", levelEnv, topoPathTypeStr[l]);
    *level = l >= 0 ? l : -2;
  }
  return epSuccess;
}

int epTopoUserP2pLevel = -1;
epResult_t epTopoCheckP2p(struct epComm* comm, struct epTopoSystem* system, int rank1, int rank2, int* p2p, int *read) {
  *p2p = 0;
  // Read is Not Supported/Used on gcu300&gcu400
  if (read) *read = 0;

  // Rule out different nodes / isolated containers
  if (comm && comm->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    struct epPeerInfo* info1 = comm->peerInfo+rank1;
    struct epPeerInfo* info2 = comm->peerInfo+rank2;
    if (info1->hostHash != info2->hostHash) {
      if (comm->MNLARE) {
        int mnlare = 0;
        EP_CHECK(epTopoCheckMNLARE(comm->topo, info1, info2, &mnlare));
        if (!mnlare) return epSuccess;
      } else {
        return epSuccess;
      }
    } else if (info1->shmDev != info2->shmDev) {
      return epSuccess;
    }
  }

  // Get GCUs from topology
  int g1, g2;
  epResult_t ret1 = system->rankToIndex(rank1, &g1);
  epResult_t ret2 = system->rankToIndex(rank2, &g2);
  if (ret1 != epSuccess || ret2 != epSuccess) {
    // GCU not found, we can't use p2p.
    return epSuccess;
  }

  // In general, use P2P whenever we can.
  int p2pLevel = PATH_SYS;

  // User override
  if (epTopoUserP2pLevel == -1)
    EP_CHECK(epGetLevel(&epTopoUserP2pLevel, "EP_P2P_DISABLE", "EP_P2P_LEVEL"));
  if (epTopoUserP2pLevel != -2) {
    p2pLevel = epTopoUserP2pLevel;
    goto compare;
  }

  // Don't use P2P through ARM CPUs
  // int arch, vendor, model;
  // EP_CHECK(epTopoCpuType(system, &arch, &vendor, &model));
  // if (arch == EP_TOPO_CPU_ARCH_ARM) p2pLevel = PATH_PXB;
  // if (arch == EP_TOPO_CPU_ARCH_X86 && vendor == EP_TOPO_CPU_VENDOR_INTEL) {
  //   p2pLevel = PATH_PXB;
  // }

compare:
  // Compute the PCI distance and compare with the p2pLevel.
  epTopoLinkList* path;
  EP_CHECK(system->getPath(GCU, g1, GCU, g2, &path));
  if (path->type <= p2pLevel) {
    if (path->type == PATH_LARE)
      *p2p = EP_P2P_LARE;
    else if (path->type == PATH_PIX)
      *p2p = EP_P2P_PCIE;
    else
      *p2p = 1;
  }
  return epSuccess;
}

EP_PARAM(NetDisableIntra, "NET_DISABLE_INTRA", 0);

// Check whether going through the network would be faster than going through P2P/SHM.
epResult_t epTopoCheckNet(struct epTopoSystem* system, int rank1, int rank2, int* net) {
  if (epParamNetDisableIntra() == 1) {
    *net = 0;
    return epSuccess;
  }
  *net = 1;
  // First check the current GCU-to-GCU speed.
  int g1, g2;
  if (system->rankToIndex(rank1, &g1) != epSuccess ||
      system->rankToIndex(rank2, &g2) != epSuccess) {
    return epSuccess;
  }

  struct epTopoLinkList* path;
  EP_CHECK(system->getPath(GCU, g1, GCU, g1, &path));

  float speed = path->bw;

  // Now check the speed each GCU can access the network through PXB or better
  float netSpeed1 = 0, netSpeed2 = 0;
  for (int n=0; n<system->netCount(); n++) {
    EP_CHECK(system->getPath(GCU, g1, NET, n, &path));
    if (path->type <= PATH_PXB && path->bw > netSpeed1) netSpeed1 = path->bw;
    EP_CHECK(system->getPath(GCU, g2, NET, n, &path));
    if (path->type <= PATH_PXB && path->bw > netSpeed2) netSpeed2 = path->bw;
  }

  if (netSpeed1 > speed && netSpeed2 > speed) return epSuccess;
  *net = 0;
  return epSuccess;
}

int epTopoUserGdrLevel = -1;
EP_PARAM(NetGdrRead, "NET_GDR_READ", -2);

epResult_t epTopoCheckGdr(struct epTopoSystem* system, int rank, int64_t netId, int read, int* useGdr){
  *useGdr = 0;

  // Get GCU and NET
  int n, g;
  EP_CHECK(system->topoIdToIndex(NET, netId, &n));
  struct epTopoNode* net;
  EP_CHECK(system->getNode(NET, n, &net));
  EP_CHECK(system->rankToIndex(rank, &g));
  struct epTopoNode* gcu;
  EP_CHECK(system->getNode(GCU, g, &gcu));

  // Check that both the NIC and GCUs support it
  if (net->net.gdrSupport == 0) return epSuccess;
  if (gcu->gcu.gdrSupport == 0) return epSuccess;

  if (read) { // For reads (sends) only enable under certain conditions
    int gdrReadParam = epParamNetGdrRead();
    if (gdrReadParam == 0) return epSuccess;
    if (gdrReadParam < 0) {
      int isLarelink = 0;
      // Since we don't know whether there are other communicators,
      // it's better to keep things local if we have a single GCU.
      if (system->gcuCount() == 1) isLarelink = 1;
      for (int i=0; i<system->gcuCount(); i++) {
        if (i == g) continue;
        epTopoLinkList* path;
        EP_CHECK(system->getPath(GCU, i, GCU, g, &path));
        if (path->type == PATH_LARE) {
          isLarelink = 1;
          break;
        }
      }
      if (!isLarelink) return epSuccess;
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  int netGdrLevel = PATH_PXB;
  EP_CHECK(epGetLevel(&epTopoUserGdrLevel, nullptr, "EP_NET_GDR_LEVEL"));
  if (epTopoUserGdrLevel != -2) netGdrLevel = epTopoUserGdrLevel;
  int distance = gcu->paths[NET][n].type;
  if (distance > netGdrLevel) {
   INFO(EP_NET,"GCU Direct RDMA Disabled for GCU %d / HCA %d (distance %d > %d)", rank, netId, distance, netGdrLevel);
   return epSuccess;
  }

  *useGdr = 1;
  INFO(EP_NET,"GCU Direct RDMA Enabled for GCU %d / HCA %d, read %d", rank, netId, read);
  return epSuccess;
}

EP_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

epResult_t epTopoGetCpuAffinity(struct epTopoSystem* system, int rank, cpu_set_t* affinity) {
  struct epTopoNode* cpu = nullptr, *gcu = nullptr;

  for (int g=0; g<system->gcuCount(); g++) {
    struct epTopoNode* gcuNode;
    EP_CHECK(system->getNode(GCU, g, &gcuNode));
    if (gcuNode->gcu.rank == rank) {
      gcu = gcuNode;
      // Find closer CPU
      int cpuIndex = -1, minHops = 0;
      for (int c=0; c<system->cpuCount(); c++) {
        epTopoLinkList* path;
        EP_CHECK(system->getPath(GCU, g, CPU, c, &path));
        int nHops = path->count;
        if (cpuIndex == -1 || nHops < minHops) {
          cpuIndex = c;
          minHops = nHops;
        }
      }
      EP_CHECK(system->getNode(CPU, cpuIndex, &cpu));
    }
  }

  if (cpu == nullptr) {
    WARN("Set CPU affinity : unable to find GCU/CPU for rank %d", rank);
    return epInternalError;
  }

  // Query the CPU affinity set we were provided
  cpu_set_t mask;
  SYSCHECK(sched_getaffinity(0, sizeof(cpu_set_t), &mask), "sched_getaffinity");

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    EP_CHECK(epCpusetToStr(&mask, affinityStr));
    TRACE(EP_INIT, "Current affinity for GCU %d is %s", gcu->gcu.dev, affinityStr);
  }
#endif

  // Get the affinity of the CPU close to our GCU.
  cpu_set_t cpuMask = cpu->cpu.affinity;

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    EP_CHECK(epCpusetToStr(&cpuMask, affinityStr));
    TRACE(EP_INIT, "CPU GCU affinity for GCU %d is %s", gcu->gcu.dev, affinityStr);
  }
#endif

  cpu_set_t finalMask;
  if (epParamIgnoreCpuAffinity())
    // Ignore the CPU affinity set and use the GCU one instead
    finalMask = cpuMask;
  else
    // Use a subset of the GCU affinity set
    CPU_AND(&finalMask, &mask, &cpuMask);

  memcpy(affinity, &finalMask, sizeof(cpu_set_t));

  // If there is a non empty set, use it to set affinity
  if (CPU_COUNT(&finalMask)) {
    char affinityStr[sizeof(cpu_set_t)*2];
    EP_CHECK(epCpusetToStr(&finalMask, affinityStr));
    INFO(EP_INIT, "Setting affinity for GCU %d to %s", gcu->gcu.dev, affinityStr);
  }

  return epSuccess;
}

epResult_t epTopoCheckMNLARE(struct epTopoSystem* system, struct epPeerInfo* info1,
                                                                                 struct epPeerInfo* info2, int* ret) {
  efmlGcuFabricInfoV_t *fabricInfo1 = &info1->gcuInfo.fabricInfo;
  efmlGcuFabricInfoV_t *fabricInfo2 = &info2->gcuInfo.fabricInfo;
  return system->checkMNLARE(fabricInfo1, fabricInfo2, ret);
}

