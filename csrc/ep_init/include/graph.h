/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_GRAPH_H_
#define EP_GRAPH_H_

#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

#include "ep.h"
#include "devcomm.h"
#include "cpuset.h"

#define EP_TOPO_MAX_LARESWTS 8
#define EP_TOPO_MAX_GCUS 256
#define EP_TOPO_MAX_NETS 64
#define EP_TOPO_MAX_NODES 256
#define EP_P2P_LARE 0x2    //0x1 is default value which indicate other type
#define EP_P2P_PCIE 0x4

enum epTopoNodeType : uint8_t { GCU = 0, PCI = 1, LARESWT = 2/* lare switch */,
                                  CPU = 3, NIC = 4, NET = 5, EP_TOPO_NODE_TYPES};

struct epTopoSystem;
// Build the topology.
epResult_t epTopoGetSystem(struct epComm* comm, struct epTopoSystem** system);
// Sort topology for speed up paths compute.
epResult_t epTopoSortSystem(struct epTopoSystem* system);
// Print topo system with path if computed.
void epTopoPrint(struct epTopoSystem* system);

// Create a topoloy system according to comm's content.
epResult_t epTopoComputePaths(struct epTopoSystem* system, struct epComm* comm);
void epTopoFree(struct epTopoSystem* system);
// Trim unused topo node.
epResult_t epTopoTrimSystem(struct epTopoSystem* system, epComm* comm);
epResult_t epTopoComputeP2pChannels(epComm* comm);

// Query topology
epResult_t epTopoGetNetDev(epTopoSystem* system, int rank, struct epTopoGraph* graph, int channelId,
                                                                               int peerRank, int64_t* netId, int* dev);
epResult_t epTopoCheckP2p(struct epComm* comm, struct epTopoSystem* system, int rank1, int rank2, int* p2p, int *read);

epResult_t epTopoCheckGdr(struct epTopoSystem* system, int rank, int64_t netId, int read, int* useGdr);
epResult_t epTopoCheckNet(struct epTopoSystem* system, int rank1, int rank2, int* net);

// Find CPU affinity
epResult_t epTopoGetCpuAffinity(struct epTopoSystem* system, int rank, cpu_set_t* affinity);


typedef enum epTopoCpuArchType {
    EP_TOPO_CPU_ARCH_X86,
    EP_TOPO_CPU_ARCH_ARM
} epTopoCpuArchType;

typedef enum epTopoCpuVendorType {
    EP_TOPO_CPU_VENDOR_INTEL,
    EP_TOPO_CPU_VENDOR_AMD
} epTopoCpuVendorType;

typedef enum epTopoCpuModelType {
  EP_TOPO_CPU_MODEL_BDW,
  EP_TOPO_CPU_MODEL_SKL
} epTopoCpuModelType;

epResult_t epTopoCpuType(struct epTopoSystem* system, int* arch, int* vendor, int* model);
epResult_t epTopoGetGcuCount(struct epTopoSystem* system, int* count);
epResult_t epTopoGetGcuArch(struct epTopoSystem* system, int* arch);
epResult_t epTopoGetNetCount(struct epTopoSystem* system, int* count);
epResult_t epTopoGetLocalNet(struct epTopoSystem* system, int rank, int* id);

// Init search. Needs to be done before calling epTopoCompute
epResult_t epTopoSearchInit(struct epTopoSystem* system);


class epTopoGraphHdl;
typedef enum epTopoPatternType {
  EP_TOPO_PATTERN_RING,
  EP_TOPO_PATTERN_MESH,
  EP_TOPO_PATTERN_BALANCED_TREE,
  EP_TOPO_PATTERN_NUM
} epTopoGraphPatternType;

struct epTopoGraph {
  // Input / output
  int pattern; // ring : 0, mesh : 1
  int crossNic;
  int minChannels;
  int maxChannels;
  // Output
  int nChannels;
  float bwIntra;
  float bwInter;
  int typeIntra;
  int typeInter;
  int nHops;
  int intra[MAXCHANNELS*EP_TOPO_MAX_NODES];
  int64_t inter[MAXCHANNELS*2];
  epTopoGraphHdl* hdl;
};
// Compute Topo Graph by given topo pattern.
epResult_t epTopoCompute(struct epTopoSystem* system, struct epTopoGraph* graph);
// Free topo graph handler.
void epTopoGraphFree(epTopoGraph* graph);

typedef enum epTopoPortLinkType {
  EP_TOPO_PORT_LINK_PCIE,
  EP_TOPO_PORT_LINK_LARE,
  EP_TOPO_PORT_LINK_NET // use PCIe actually
} epTopoPortLinkType;

struct epTopoPort {
  epTopoPortLinkType linkType;
  int efmlPorts[MAX_PORTS_PER_TRUNK];
  int count;
};

// compute Port connection of one channel specified by channelId in graph.
epResult_t epTopoComputePort(const epComm* comm, const epTopoGraph* graph, const int rank,
                                                      const int peerRank, int channelId, struct epTopoPort* portInfo);

// Compute the number of trunks between two ranks.
epResult_t epTopoGetNTrunks(const epComm* comm, int rank1, int rank2, int* nTrunks);

void epTopoPrintGraph(struct epTopoSystem* system, struct epTopoGraph* graph);
// Dump n graph.
epResult_t epTopoDumpGraphs(struct epTopoSystem* system, int ngraphs, struct epTopoGraph** graphs);

struct epTopoRanks {
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];

  int meshSend[MAXCHANNELS * EP_MAX_LOCAL_RANKS];
  int meshRecv[MAXCHANNELS * EP_MAX_LOCAL_RANKS];
};

epResult_t epTopoPreset(epComm* comm, struct epTopoGraph** graphs, epTopoRanks* topoRanks);

epResult_t epTopoPostset(epComm* comm, int* firstRanks, epTopoRanks** allTopoRanks, int* rings);
epResult_t epTopoCheckMNLARE(struct epTopoSystem* system, struct epPeerInfo* info1,
                                                                                 struct epPeerInfo* info2, int* ret);

#define EP_NUM_DATAFILEDS 3
#define EP_HIGHEST_PRIORITY 1
#define EP_LOWEST_PRIORITY INT_MAX
// EP_UNSUPPORTED_PRIORITY == EP_ALGO_PROTO_IGNORE
#define EP_UNSUPPORTED_PRIORITY -1

epResult_t epTopoTuneModel(struct epComm* comm, int minCompCap, int maxCompCap, struct epTopoGraph** graphs);
#define MEM_OUTPLACE 0
#define MEM_INPLACE 1
#define MAX_MEM_PLACE_TYPE 2
epResult_t epTopoGetAlgoTime(struct epComm* comm, int coll, int algorithm, int protocol, size_t nBytes,
                                                            int numPipeOps, float* time, int memPlaceType=MEM_OUTPLACE);
epResult_t epTopoGetDefaultAlgoProto(struct epComm* comm, int coll, int* algorithm, int* protocol, size_t nBytes);
static constexpr uint32_t treeAlgoMask = (1U<<EP_ALGO_TREE);
static constexpr uint32_t meshDirectAlgoMask = (1U<<EP_ALGO_MESH_DIRECT);
static constexpr uint32_t meshAlgoMask = (1U<<EP_ALGO_MESH);
static constexpr uint32_t hybridTreeAlgoMask = (1U<<EP_ALGO_HYBRID_TREE);

void epTopoInitAlgo();

#endif
