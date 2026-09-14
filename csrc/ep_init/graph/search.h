#ifndef EP_SEARCH_H__
#define EP_SEARCH_H__

#include <utility>
#include "graph.h"
#include "topo.h"

class epTopoGraphHdl {
public:
  epTopoGraphHdl(epTopoSystem* system, struct epTopoGraph* graph) {
    system_ = system;
    pattern_ = static_cast<epTopoGraphPatternType>(graph->pattern);
    minChannels_ = 1;
    maxChannels_ = graph->maxChannels;
  }
  virtual ~epTopoGraphHdl() {}
  epTopoGraphHdl(const epTopoGraphHdl& graph) = default;

  bool operator<(const epTopoGraphHdl& graph) const;
  epTopoGraphHdl& operator=(const epTopoGraphHdl& graph);

  // Compute graph for the pattern according to toplogy system.
  virtual epResult_t compute();
  // Dump graph to xml.
  epResult_t getXmlFromGraph(epXml *xml);
  // Load graph from xml.
  epResult_t getGraphFromXml(epXml* xmlGraphs, int* channels);
  virtual void print();
  // Update
  virtual void copyRaw(epTopoGraph* graph);
  epResult_t getNetDev(int rank, int channelId, int64_t* netId, int* netDev);
  void updateGraphInfo(epTopoGraph* graph);
  virtual const char* getPatternDesc();

private:
  // This call give us a optimal graph for current condition,
  // including intra path type, inter path type, etc.
  epResult_t searchGraphRec(epTopoGraphHdl* optimalGraph, int* time);
  // We search channel begin from different node for
  // different connection. In intra-node connection,
  // we always search begin from net; As for inter-node,
  // we should from gcu, as there is no net for outside
  // communication.
  epResult_t searchChannelFromRgcu(epTopoGraphHdl* optimalGraph, int* time);
  epResult_t searchChannelFrom(epTopoNodeType fromType, epTopoGraphHdl* optimalGraph, int* time);
  epResult_t searchChannelFrom(epTopoNodeType type, int index, epTopoNodeType dstType,
                                        int dstIdx, epTopoGraphHdl* optimalGraph, int step, int gcuOrder, int* time);
  epResult_t searchRestore(epTopoNodeType type, int index, epTopoNodeType type2, int index2);
  // before add a gcu to channel, we should try this gcu.
  // If try(check condition) success, return a boolean true,
  // and than add this gcu to channel. After we finish use
  // this gcu in the channel, we should call restore gcu
  // so that the resources of using this gcu would be
  // recovered.
  bool searchTry(epTopoNodeType type, int index, epTopoNodeType type2, int index2);
  // Search one channel.
  epResult_t searchChannelRec(epTopoGraphHdl* optimalGraph, epTopoNodeType srcType,
                                                                        int srcIdx, int step, int gcuOrder, int *time);
  epResult_t searchSelectGcuOrderReplay(int step, epTopoNodeType fromType, int* retNextGcu, int* retCount);
  epResult_t searchSelectGcuOrderScore(int step, int* retNextGcu, int* retCount, int sortInter, 
                                                                                                epTopoNodeType type);
  epResult_t searchSelectDstNodes(epTopoNodeType srcType, int srcIdx, epTopoNodeType dstType,
                                                                                      int* dstIdxs, int* retNodeCount);
  epResult_t addChannel(epXmlNode *xmlChannel, int channelIdx);

  epResult_t getXmlFromChannel(int channel, epXml *xml, epXmlNode* graph);
protected:
  epTopoSystem* system_;
  // Input / output
  epTopoGraphPatternType pattern_;
  int crossNic_;
  int minChannels_;
  int maxChannels_;
  // Output
  int nChannels_;
  float bwIntra_;
  float bwInter_;
  float latencyInter_;
  epTopoPathType typeIntra_;
  epTopoPathType typeInter_;
  int sameChannels_;
  int nHops_;
  int intra_[MAXCHANNELS*EP_TOPO_MAX_NODES];
  int64_t inter_[MAXCHANNELS*2];
};

class epMeshGraphHandler: public epTopoGraphHdl {
public:
  epMeshGraphHandler(epTopoSystem* system, struct epTopoGraph* graph);
  virtual ~epMeshGraphHandler();
  virtual epResult_t compute();
  virtual const char* getPatternDesc();
  virtual void print();
  virtual void copyRaw(epTopoGraph* graph);
};

#endif
