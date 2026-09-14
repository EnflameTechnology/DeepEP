#ifndef EP_TOPO_H_
#define EP_TOPO_H_

#include <stdint.h>
#include <sched.h>

#include "ep.h"
#include "comm.h"
#include "graph.h"
#include "detector.h"

enum epTopoLinkType : uint8_t {
    LINK_LOC  = 0,
    LINK_LARE = 1,
    LINK_PCI  = 2,
    // Skipping 3 for PATH_PXB
    // Skipping 4 for PATH_PHB
    LINK_SYS  = 5,
    LINK_NET  = 6,
    MAX_LINK_TYPE
};

enum epTopoPathType : uint8_t {
    PATH_LOC  = 0,  // Local (myself)
    PATH_LARE = 1,  // Connection traversing LARE
    PATH_PIX  = 2,  // Connection traversing at most a single PCIe bridge
    PATH_PXB  = 3,  // Connection traversing multiple PCIe bridges (without
                    // traversing the PCIe Host Bridge)
    PATH_PHB  = 4,  // Connection traversing PCIe as well as a PCIe Host Bridge
                    // (typically the CPU)
    PATH_SYS  = 5,  // Connection traversing PCIe as well as the SMP interconnect
                    // between NUMA nodes (e.g., QPI/UPI)
    PATH_NET  = 6,  // Connection through the network
    PATH_DIS  = 7,   // Disconnected
    MAX_PATH_TYPE
};
enum epGcuConncetType : uint8_t {
  DIRECT_CONNECTED = 0,  //for OGX, 0-13 port are direct connected to intra GCU, port 14-15 may connect to switch
  SWITCH_CONNECTED       //for POD, 0-15 port connecting to switch
};

extern const char* topoNodeTypeStr[];
extern const char* topoLinkTypeStr[];
extern const char* topoPathTypeStr[];

// BandWidth
constexpr float LOC_BW  = 5000.0;
constexpr float GCU_400_LARE_BW = 23.0;
constexpr float QPI_BW = 9.0;

constexpr int EP_TOPO_MAX_LINKS = 128;
constexpr int EP_TOPO_MAX_HOPS = EP_TOPO_MAX_NODES * EP_TOPO_NODE_TYPES;

#define EP_TOPO_ID_LOCAL_ID_MASK 0x00ffffffffffffff
#define EP_TOPO_ID_SYSTEM_ID(id) (id >> 56)
#define EP_TOPO_ID_LOCAL_ID(id) (id & EP_TOPO_ID_LOCAL_ID_MASK)
#define EP_TOPO_LOCAL_NIC_ID(numaid, busid) (((int64_t)numaid << 56) + busid)
#define EP_TOPO_ID(systemid, localid) (((int64_t)systemid << 56) + (localid & EP_TOPO_ID_LOCAL_ID_MASK))

#define EP_TOPO_NTRUNK_PER_LARESWITCH (2)

struct epTopoNode;
struct epTopoLink {
  int type;
  float bw;
  epTopoNode* remNode;

  int localEfmlPortId[EP_TOPO_MAX_GCUS][MAX_LARES_PER_GCU] = {{-1}}; // local efml port IDs to peer gcus
  int portCount[EP_TOPO_MAX_GCUS] = {0};                    // local efml port num to peer gcus

  int efmlPortToLareSwitch[EP_TOPO_MAX_GCUS][EP_TOPO_MAX_LARESWTS][MAX_LARES_PER_GCU] =
      {{{-1}}}; // local efml port IDs to peer LareSwitch
  int portSizeToLareSwitch[EP_TOPO_MAX_GCUS][EP_TOPO_MAX_LARESWTS] = {{0}}; // ports count
  int portSelectionOfLareSwitch[EP_TOPO_MAX_LARESWTS] = {0};              // ports selection
};

struct epTopoLinkList {
  epTopoLink* list[EP_TOPO_MAX_HOPS];
  int count;
  float bw;
  int type;
};

struct epTopoNode {
  epTopoNodeType type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // EFML dev number
      int rank;
      efmlDeviceArchitecture_t efmlArch;
      int gdrSupport;
      enum epGcuConncetType connectType; //it's a total conclusion only for all lare links between GCU400
    }gcu;
    struct {
      int dev; // Plugin dev number
      uint64_t asic;
      int port;
      float bw;
      float latency;
      int gdrSupport;
      int maxChannels;
    }net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    }cpu;
    struct {
      uint64_t device;
    }pci;
    struct {
      int dev;
    }lareswitch;
  };
  int nlinks;
  epTopoLink links[EP_TOPO_MAX_LINKS];
  // Pre-computed paths to GCUs and NICs
  epTopoLinkList* paths[EP_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
};

struct epTopoSystem {
public:
  epTopoSystem() : nodes_{}, maxBw_(0.0), totalBw_(0.0)  { }
  ~epTopoSystem() { }

  // Build topology system from xml, which is detected or load from a file.
  epResult_t getSystemFromXml(epXml* xml, const uint64_t localHostHash);

  // Get path of from node type1/index1 to node type2/index2. Path will be
  epResult_t getPath(epTopoNodeType type1, int index1, epTopoNodeType type2,
  // created if path not exist.
                       int index2, epTopoLinkList** path);
  // Getting all node in topology to the given node path. We use breadth-first
  // search algorithm to solove it.
  epResult_t setPaths(epTopoNodeType type, int index);

  // We define local cpu of gcu by the hops, which means the path's
  // count from gcu to cpu. Local cpu have min hops to gcu given by
  // the index of g.
  epResult_t getLocalCpu(int gcuIdx, int* retCpu);

  epResult_t getLocalNet(int rank, int channelId, int64_t* netId, int* netDev);

  // We can use CPU as intermediate node, but we may need decrease
  // bandwidth of path from src to dst, as well as path type.
  //
  void addInterStep(epTopoNodeType tx, int ix, epTopoNodeType t1, int i1, epTopoNodeType t2, int i2);

  // Remove node of type/index, including path from the given node,
  // link to the given node. We know that not all information about
  // node type/index is cleared, saying other node's path to the
  // given node. We will always re-compute path after this function call.
  epResult_t removeNode(epTopoNodeType type, int index);

  // Remove/free all paths for a given type of node.
  void removePaths(epTopoNodeType type);

  epResult_t getNode(epTopoNodeType type, int index, epTopoNode** node);

  epResult_t topoIdToIndex(epTopoNodeType type, int64_t id, int* index);
  epResult_t rankToIndex(int rank, int* idx, epTopoNodeType type=GCU);
  epResult_t topoIdToNetDev(int64_t id, int* netDev);

  int netCount() const { return nodes_[NET].count; }
  int gcuCount() const { return nodes_[GCU].count; }
  int cpuCount() const { return nodes_[CPU].count; }
  int getCount(epTopoNodeType type) const { return nodes_[type].count; }
  float maxBw() const { return maxBw_; }
  float totalBw() const { return totalBw_; }

  epResult_t getLocalGcus(epTopoNodeType type, int idx, float bw, int* retGcusIdx, int* retGcuCount);
  epResult_t getDeviceType(efmlDeviceArchitecture_t *arch);
  int gcuPciBw(int gcuIdx);

  // maxBw from GCU to GCU or from GCU to NIC/NET
  float getMaxBw(int gcuIdx, epTopoNodeType type);
  // GCU's totalBw, Pci/Lare
  float getTotalBw(int gcuIdx);

  epResult_t graphSearchInit();

  epResult_t addChannelToNode(epTopoNodeType type, int g, int channel);
  bool gcuInChannel(int index, int channel);
  epResult_t removeChannelFromNode(epTopoNodeType type, int n, int channel);

  bool allocBw(epTopoNodeType type1, int index1, epTopoNodeType type2,
               int index2, epTopoPathType type, float bw);
  epResult_t freeBw(epTopoNodeType type1, int index1, epTopoNodeType type2,
                      int index2, float bw);

  epResult_t getLocalNet(int g, int* netDev, int seed);
  // Prints
  void printPathMatrix();
  void printLinks();
  epResult_t checkMNLARE(efmlGcuFabricInfoV_t* fabricInfo1, efmlGcuFabricInfoV_t* fabricInfo2, int* ret);

private:
  // The topology node will be created during transfer process from xml
  // to topology system. The following two function will be used only at
  // the transfer process, so make it private.
  epTopoNode* getNode(epTopoNodeType type, int64_t id);
  epResult_t createNode(epTopoNodeType type, uint64_t id,
                          epTopoNode** node);

  // Internal APIs for adding topology node from xml node, as well as
  // some of Links, including LINK_PCI, LINK_NET.
  epResult_t addCpu(epXmlNode* xmlCpu);
  epResult_t addPci(epXmlNode* xmlPci, epTopoNode* upNode, int systemId, int numaId);
  epResult_t addNic(epXmlNode* xmlNet, epTopoNode* nic, int systemId);
  epResult_t addNet(epXmlNode* xmlNet, epTopoNode* nic, int systemId);
  epResult_t addGcu(epXmlNode* xmlGcu, epTopoNode* gcu);

  // Internal APIs for adding other links including LINK_SYS, LINK_LARE.
  epResult_t addLares(epXmlNode* node, const char* parentBusId, int systemId);

  // For multi-CPUs connections, just connect them witch each other directly.
  epResult_t addCpusLink();

  // Add link between node and remNode, it is one direction link.
  // When creating links, this function will be called.
  epResult_t connectNodes(epTopoNode* node, epTopoNode* remNode, int type,
                            float bw);
  epResult_t calcEfmlLarePorts(int systemId, epTopoNode* srcGcuNode, epTopoNode* dstNode, const char *portConnStrPtr);
  // Sort the links in one topology node in order to accelerate toplogy system
  // traversal
  void sortSystem();

  // Bandwidth.
  float getInterCpuBw(epTopoNode* cpu);
  float getLareBw(epTopoNode* gcu);

  void getLevel(epTopoPathType* level,
                const char* disableEnv, const char* levelEnv);

  epResult_t followPath(epTopoLinkList* path, epTopoNode* start,
                          int maxSteps, float bw, int* retFollowedSteps);

  epResult_t findRevLink(epTopoNode* node1, epTopoNode* node2,
                           epTopoLink** revLink);
  epResult_t getSystemId(epXmlNode* xmlCpu, int* systemIdPtr);
  inline epResult_t getGcuNode(int systemId, const char* busIdStr, epTopoNode* & gcuNode) {
    int64_t busId;
    EP_CHECK(busIdToInt64(busIdStr, &busId));
    int64_t topoId = EP_TOPO_ID(systemId, busId);
    gcuNode = getNode(GCU, topoId);
    return epSuccess;
  }
  // Internal calls for print.
  #ifdef ENABLE_TRACE
  void printNodePaths(epTopoNode* node);
  void printPaths();
  #endif
  void printLinkRec(epTopoNode* node, epTopoNode* prevNode, char* line,
                    int offset);
  void printGcuNetPathMatrix();
  void printGcuSwitchPathMatrix();

  struct epTopoNodeSet {
    int count;
    struct epTopoNode nodes[EP_TOPO_MAX_NODES];
  };
  int mySystemId_;
  uint64_t hostHashes_[EP_TOPO_MAX_NODES];
  int nHosts_;
  struct epTopoNodeSet nodes_[EP_TOPO_NODE_TYPES];
  float maxBw_;
  float totalBw_;
};

#endif
