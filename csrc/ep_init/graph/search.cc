#include "search.h"
#include "checks.h"
#include "alloc.h"
#include "xml.h"
#include "align.h"

#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <vector>

#define GCU_ORDER_SCORE 1
#define GCU_ORDER_REPLAY 2

const char* graphPatternStr[] = { "ring", "mesh", "db_tree"};
static_assert((sizeof(graphPatternStr)/sizeof(graphPatternStr[0])) == EP_TOPO_PATTERN_NUM,
                                                                              "graphPatternStr length is not enough");
float speedArrayIntra[] = {46., 23./* Lare */, 12. /* PCI Gen3 x16 */, 23./2, 9./*QPI*/, 23./3};
float speedArrayInter[] = {24./* IB */, 23./* Lare */, 12. /* PCI Gen3 x16 */, 23./2, 10.7 /* RoCE*/, 9. /* QPI */, 23./3, 23./4};
#define NSPEEDSINTRA (sizeof(speedArrayIntra)/sizeof(float))
#define NSPEEDSINTER (sizeof(speedArrayInter)/sizeof(float))

#define EP_SEARCH_GLOBAL_TIMEOUT (1ULL<<18) // 262144
#define EP_SEARCH_TIMEOUT (1<<16) // 65536
#define EP_SEARCH_TREE_GLOBAL_TIMEOUT (1ULL<<22) // 4194304
#define EP_SEARCH_TREE_TIMEOUT (1<<20) // 1048576

EP_PARAM(CrossNic, "CROSS_NIC", 2);
EP_PARAM(AlgoForceOrderEnable, "ALGO_FORCE_ORDER_ENABLE", 0);

/*
 * Choose the order in which we try next GCUs. This is
 * critical for the search to quickly converge to the best
 * solution even if it eventually times out.
 */
struct epGcuScore {
  int g;             // Retain the index
  int startIndex;    // Least important
  int intraNhops;
  int intraBw;
  int interNhops;
  int interPciBw;
  int interBw;    // Most important
};
static bool cmpIntraScores(epGcuScore* scores, int count) {
  int intraBw = scores[0].intraBw;
  int intraNhops = scores[0].intraNhops;
  for (int i = 1; i < count; i ++) {
    if (scores[i].intraBw != intraBw || scores[i].intraNhops != intraNhops)
      return true;
  }
  return false;
}
epTopoPathType strToPathType(const char* str) {
  int size = -1;
  while (topoPathTypeStr[++ size]) ;
  for (int i = 0; i < size; ++ i) {
    if (strcmp(str, topoPathTypeStr[i]) == 0)
      return static_cast<epTopoPathType>(i);
  }
  return PATH_LOC;
}

// Try add GCU/g to current channel
// Previous node of GCU/g is node type/index
// When type/index == */-1, GCU/g don't have pre node
bool epTopoGraphHdl::searchTry(epTopoNodeType type, int index, epTopoNodeType type2, int index2) {
  // GCU/g is first node
  if (index == -1) {
    if (type2 == NET) {
      epTopoNode* net;
      if (system_->getNode(NET, index2, &net) != epSuccess) {
        WARN("not found the net node we going to try, ignore...");
        return false;
      }
      if (net->net.bw < bwInter_) return false;
      if (net->net.maxChannels == 0) return false;
      net->net.bw -= bwInter_;
    }
    return true;
  }
  // Now check link type
  int intra = (type == GCU);
  float bw = intra ? bwIntra_ : bwInter_;
  epTopoPathType pathType = intra ? typeIntra_ : typeInter_;

  // INFO("tryGcu %s/%d->GCU/%d, pathType(%s) bw(%f)",
  // topoNodeTypeStr[type], index, g, topoPathTypeStr[pathType], bw);
  // Try to allocate bandwidth from path type/index->GCU/g
  if (system_->allocBw(type, index, type2, index2, pathType, bw)) {
    epTopoLinkList* path;
    system_->getPath(type, index, type2, index2, &path);
    nHops_ += path->count;
    return true;
  }
  // TODO : why
  /*
      for (int i=0; i<system->nodes[NET].count; i++) {
        if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
            (system->nodes[NET].nodes[i].net.port == net->net.port)) {
          system->nodes[NET].nodes[i].net.bw -= bwInter_;
        }
      }
  */

  return false;
}

epResult_t epTopoGraphHdl::searchRestore(epTopoNodeType type, int index, epTopoNodeType type2, int index2) {
  if (index == -1) {
    if (type2 == NET) {
      epTopoNode* net;
      EP_CHECK(system_->getNode(NET, index2, &net));
      net->net.bw += bwInter_;
      net->net.maxChannels ++;
    }
    return epSuccess;
  }
  int intra = (type == GCU);
  float bw = intra ? bwIntra_ : bwInter_;
  EP_CHECK(system_->freeBw(type, index, type2, index2, bw));
  epTopoLinkList* path;
  EP_CHECK(system_->getPath(type, index, type2, index2, &path));
  nHops_ -= path->count;
  return epSuccess;
}

epResult_t epTopoGraphHdl::searchSelectGcuOrderReplay(int step, epTopoNodeType fromType, int* retNextGcu, int* retCount) {
  if (nChannels_ == 0) return epInternalError;
  int count = system_->getCount(fromType);
  int nextRank = intra_[(nChannels_-1) * count + step + 1];
  EP_CHECK(system_->rankToIndex(nextRank, retNextGcu));
  *retCount = 1;
  return epSuccess;
}

epResult_t epTopoGraphHdl::searchSelectGcuOrderScore(int step, int* retNextGcu, int* retCount,
                                                                            int sortInter, epTopoNodeType interType) {
  int gcus = system_->gcuCount();

  epGcuScore scores[EP_TOPO_MAX_NODES];
  memset(scores, 0, sizeof(scores));

  int start;
  EP_CHECK(system_->rankToIndex(intra_[(nChannels_) * gcus + step], &start));
  epTopoLinkList* paths;
  EP_CHECK(system_->getPath(GCU, start, GCU, 0, &paths));

  epTopoLinkList* interPaths = nullptr;
  if (sortInter) {
    int n;
    EP_CHECK(system_->topoIdToIndex(interType, inter_[nChannels_*2], &n));
    EP_CHECK(system_->getPath(interType, n, GCU, 0, &interPaths));
  }

  // Get the starting GCU for current channel
  int channelStartGcu;
  EP_CHECK(system_->rankToIndex(intra_[nChannels_ * gcus], &channelStartGcu));

  int count = 0;
  // Calculate rotation offset based on current step
  int rotationOffset = nChannels_ + step + 1;
  // First try the rotated position
  if (pattern_ == EP_TOPO_PATTERN_BALANCED_TREE) {
    for (int i = 0; i < gcus; i++) {
      // Calculate target GCU index with rotation
      int targetGcu = (channelStartGcu + i * rotationOffset) % gcus;

      // Skip if no path exists
      if (paths[targetGcu].count == 0) continue;

      // Skip if GCU is already used in this channel
      if (system_->gcuInChannel(targetGcu, nChannels_)) continue;

      // Add to scores array
      scores[count].g = targetGcu;
      scores[count].startIndex = i;  // Keep track of original position
      scores[count].intraNhops = paths[targetGcu].count;
      scores[count].intraBw = paths[targetGcu].bw;

      if (interPaths) {
        scores[count].interNhops = interPaths[targetGcu].count;
        scores[count].interPciBw = system_->gcuPciBw(targetGcu);
        scores[count].interBw = interPaths[targetGcu].bw;
      }
      count++;
    }
  }

  // If no GCUs found with rotation pattern, fall back to original scoring method
  if (count == 0) {
    for (int i = 0; i < gcus; i++) {
      if (paths[i].count == 0) continue;
      if (system_->gcuInChannel(i, nChannels_)) continue;

      scores[count].g = i;
      scores[count].startIndex = i;
      scores[count].intraNhops = paths[i].count;
      scores[count].intraBw = paths[i].bw;

      if (interPaths) {
        scores[count].interNhops = interPaths[i].count;
        scores[count].interPciBw = system_->gcuPciBw(i);
        scores[count].interBw = interPaths[i].bw;
      }
      count++;
    }
  }

  // Sort GCUs by score if we have multiple options
  std::sort(scores, scores + count,
      [](const epGcuScore& s1, const epGcuScore& s2) -> bool {
        if (s2.interBw < s1.interBw) return true;
        if (s2.interPciBw < s1.interPciBw) return true;
        if (s2.interNhops > s1.interNhops) return true;
        if (s2.intraBw < s1.intraBw) return true;
        if (s2.intraNhops > s1.intraNhops) return true;
        return (s2.startIndex > s1.startIndex);
      });
#if 0
  char line[2048] = "\nscore: \n";
  int length = 0;
  for (int i = 0; i < count; ++ i) {
    epTopoNode* gcu;
    EP_CHECK(system_->getNode(GCU, scores[i].g, &gcu));
    int rank = gcu->gcu.rank;

    length = strlen(line);
    sprintf(line + length, "(%d)=interBw(%d), interPciBw(%d), interNhops(%d),"
           "intraBw(%d), intraNhops(%d), startIndex(%d)\n",
           rank, scores[i].interBw, scores[i].interPciBw, scores[i].interNhops,
           scores[i].intraBw, scores[i].intraNhops, scores[i].startIndex);
  }
  INFO(EP_GRAPH, line);
#endif

  // Check if all have the same intra-node score in which case we go reverse
  if (pattern_ != EP_TOPO_PATTERN_BALANCED_TREE && cmpIntraScores(scores, count) == 0) {
    for (int i=0; i<count; i++) retNextGcu[i] = scores[count-1-i].g;
  } else {
    for (int i = 0; i < count; i ++) {
      retNextGcu[i] = scores[i].g;
    }
  }
  *retCount = count;
  return epSuccess;
}

bool epTopoGraphHdl::operator<(const epTopoGraphHdl& refGraph) const {
  // 1. Constraint to get the same nChannels between Rings and Trees
  if (refGraph.nChannels_ < refGraph.minChannels_) return false;

  // 2. Try to get better bandwidth
  if (nChannels_ * bwIntra_ > refGraph.nChannels_ * refGraph.bwIntra_)
      return false;
  else if (nChannels_ * bwIntra_ < refGraph.nChannels_ * refGraph.bwIntra_)
      return true;
  else {
    if (bwIntra_ < refGraph.bwIntra_) return false;
  }

  // 3. Less hops (but not at the price of going cross NICs)
  if (pattern_ == refGraph.pattern_ &&
      crossNic_ == refGraph.crossNic_ &&
      nHops_ > refGraph.nHops_)
    return true;

  if (nChannels_ > 0 && nChannels_ < refGraph.nChannels_) return false;

  return false;
}

epTopoGraphHdl& epTopoGraphHdl::operator=(const epTopoGraphHdl& graph) {
  system_= graph.system_;
  pattern_ = graph.pattern_;
  crossNic_ = graph.crossNic_;
  minChannels_ = graph.minChannels_;
  maxChannels_ = graph.maxChannels_;
  nChannels_ = graph.nChannels_;
  bwIntra_ = graph.bwIntra_;
  bwInter_ = graph.bwInter_;
  latencyInter_ = graph.latencyInter_;
  typeIntra_ = graph.typeIntra_;
  typeInter_ = graph.typeInter_;
  sameChannels_ = graph.sameChannels_;
  nHops_ = graph.nHops_;
  std::memcpy(intra_, graph.intra_, sizeof(intra_));
  std::memcpy(inter_, graph.inter_, sizeof(inter_));
  return *this;
}

/* One channel is searched by this function.
 *
 * Assume we have n gcus in topology system, there are two kind of
 * connection. What we fill every step is showed following.
 * 1. intra-node connection :
 *   <                        gcu                        >
 *   +------------+------------+------------+------------+
 *   |  intra[0]  |  intra[1]  |     ...    |  intra[n-1]|
 *   +------------+------------+------------+------------+
 *          ^            ^            ^            ^           ^
 *          |            |            |            |           |
 *       step 0      step 1          ...        step n-1     step n
 *
 * 2. inter-node connection :
 *   <    net    ><                           gcu                    ><    net     >
 *   +------------+------------+------------+------------+------------+------------+
 *   |  inter[0]  |  intra[0]  |  intra[1]  |     ...    |  intra[n-1]|  inter[0]  |
 *   +------------+------------+------------+------------+------------+------------+
 *                       ^            ^            ^            ^            ^           ^
 *                       |            |            |            |            |           |
 *                    step 0      step 1          ...        step n-1     step n-1     step n
**/
epResult_t epTopoGraphHdl::searchChannelRec(epTopoGraphHdl* optimalGraph, epTopoNodeType srcType,
                                                                        int srcIdx, int step, int gcuOrder, int *time) {
  if ((*time) <= 0) return epSuccess;
   (*time)--;

  int gcus = system_->gcuCount();
  int nets = system_->netCount();
  if (step == gcus) {
    nChannels_ ++;
#if 0
    char channelStr[512];
    int length = 0;
    for (int c = 0; c < nChannels_; c ++) {
      length += snprintf(channelStr + length, 512 - length,
                        " %2d:", c);
      if (nets > 0)
        length += snprintf(channelStr + length, 512 - length,
                          "[%d->", inter_[c*2+0]);
      for (int g=0; g<gcus; g++) {
        length += snprintf(channelStr + length, 512 - length,
                          "%d", intra_[c*gcus+g]);
      }
      if (nets > 0)
        length += snprintf(channelStr + length, 512 - length,
                           "->%d]", inter_[c*2+1]);
    }
    INFO(EP_GRAPH, channelStr);
#endif
    // Here we already searched a channel and have stored to graph.
    // First we compare optimalGraph, as the name said, it store
    // current most optimal graph. If we find a better graph, store
    // it to graph. And search next channel for current graph.
    if (*optimalGraph < *this) {
      *optimalGraph = *this;
      // If find maxChannels channel, end search directly.
      if (nChannels_ == maxChannels_) *time = 0;
    }
    if (nChannels_ < maxChannels_) {
      EP_CHECK(searchGraphRec(optimalGraph, time));
    }

    nChannels_ --;
    return epSuccess;
  }

  // Here use gcu we choose to fill intra structure.
  epTopoNode* gcu;
  EP_CHECK(system_->getNode(srcType, srcIdx, &gcu));
  intra_[nChannels_ * gcus + step] = gcu->gcu.rank;

  // For step 0 ~ n-1
  if (nets > 0 && step == gcus - 1) {
    // Now we are at inter-node connection and the last step for
    // searching channel, The last gcu is filled by the code before,
    // what we should  do next is find a optimal net connect to last gcu.

    // First net in channel.
    int firstNet = -1;
    EP_CHECK(system_->topoIdToIndex(NET, inter_[nChannels_ * 2], &firstNet));

    int netCount = 0;
    int lastNets[EP_TOPO_MAX_NETS];
    memset(lastNets, 0, sizeof(lastNets));
    EP_CHECK(searchSelectDstNodes(srcType, srcIdx, NET, lastNets, &netCount));
    for (int i = 0; i < netCount; i ++) {
      epTopoNode* net;
      EP_CHECK(system_->getNode(NET, lastNets[i], &net));

      // Parameter crossNic take effect here.
      // If crossNic = 0, the channel we searched must use same
      // net for first and last net.
      // If crossNic = 1, the first net and the last net may/may
      // not same.
      if (!crossNic_  && lastNets[i] != firstNet) continue;

      // try use the net we chose.
      if (searchTry(srcType, srcIdx, NET, lastNets[i])) {
        inter_[nChannels_*2 + 1] = net->id;
        EP_CHECK(searchChannelRec(optimalGraph, srcType, srcIdx, step + 1, gcuOrder, time));
        EP_CHECK(searchRestore(srcType, srcIdx, NET, lastNets[i]));
      }
    }
  } else if (step < gcus - 1) {
    // This condition works for both inter-node and intra-node
    // connection, which search middle gcu for channel. The gcu
    // order we are going to try is decided by GCU_ORDER* flag.
    // We currently give three type of ORDER, including
    // GCU_ORDER_SCORE, GCU_ORDER_REPLAY.
    // These two order strategy return a gcu list we are going
    // to try.
    int next[EP_TOPO_MAX_GCUS] = {0};
    int gcuCount = 0;
    if (gcuOrder == GCU_ORDER_SCORE) {
      if (nets > 0) {
        EP_CHECK(searchSelectGcuOrderScore(step, next, &gcuCount, 1, NET));
      } else {
        EP_CHECK(searchSelectGcuOrderScore(step, next, &gcuCount, 0, GCU));
      }
    } else if (gcuOrder == GCU_ORDER_REPLAY) {
      EP_CHECK(searchSelectGcuOrderReplay(step, GCU, next, &gcuCount));
    }
    for (int i = 0; i < gcuCount; i ++) {
      // Try every gcu in list for channel.
      if (searchTry(srcType, srcIdx, GCU, next[i])) {
        EP_CHECK(system_->addChannelToNode(GCU, next[i], nChannels_));
        EP_CHECK(searchChannelRec(optimalGraph, GCU, next[i], step + 1, gcuOrder, time));
        EP_CHECK(system_->removeChannelFromNode(GCU, next[i], nChannels_));
        EP_CHECK(searchRestore(srcType, srcIdx, GCU, next[i]));
       }
    }
  } else if (step == gcus - 1) {
    // Now we are at intra-node connection and the last step for
    // searching channel. The last gcu will loop back to first
    // gcu, we should try to check out the path from g to first
    // gcu is OK for channel.
    int firstGcuIdx = 0;
    EP_CHECK(system_->rankToIndex(intra_[nChannels_ * gcus],
                                  &firstGcuIdx));
    if (pattern_ == EP_TOPO_PATTERN_BALANCED_TREE) {
      EP_CHECK(searchChannelRec(optimalGraph, GCU, srcIdx, step + 1, gcuOrder, time));
    } else {
      if (searchTry(srcType, srcIdx, GCU, firstGcuIdx)) {
        // We know that this gcu already in channel.
        EP_CHECK(searchChannelRec(optimalGraph, GCU, firstGcuIdx, step + 1, gcuOrder, time));
        EP_CHECK(searchRestore(srcType, srcIdx, GCU, firstGcuIdx));
      }
    }
  }
  return epSuccess;
}

// Start channel search from gcu node, this is for intra-node
// process. Current step is always zero, the searching order
// of gcu is specified by gcuOrder.
epResult_t epTopoGraphHdl::searchChannelFrom(epTopoNodeType type, int index, epTopoNodeType dstType,
                                        int dstIdx, epTopoGraphHdl* optimalGraph, int step, int gcuOrder, int* time) {
  // If try gcu success, this gcu would be added to channel,
  // and than search next gcu for currently channel.
  if (searchTry(type, index, dstType, dstIdx)) {
    EP_CHECK(system_->addChannelToNode(dstType, dstIdx, nChannels_));
    EP_CHECK(searchChannelRec(optimalGraph, dstType, dstIdx, step, gcuOrder, time));
    EP_CHECK(system_->removeChannelFromNode(dstType, dstIdx, nChannels_));
    EP_CHECK(searchRestore(type, index, dstType, dstIdx));
  }
  return epSuccess;
}

epResult_t epTopoGraphHdl::searchSelectDstNodes(epTopoNodeType srcType, int srcIdx, epTopoNodeType dstType,
                                                                                      int* dstIdxs, int* retNodeCount) {
  *retNodeCount = 0;
  int nodeCount = 0;
  int localNodes[EP_TOPO_MAX_NODES];
  memset(localNodes, 0, sizeof(localNodes));
  for (int t = 0; t <= typeInter_; t ++) {
    for (int g = 0; g < system_->getCount(srcType); g ++) {
      if (srcIdx != -1 && srcIdx != g) continue;
      epTopoLinkList* paths=nullptr;
      EP_CHECK(system_->getPath(srcType, g, dstType, 0, &paths));
      int validDstNodeCount = 0;
      for (int n = 0; n < system_->getCount(dstType); n ++) {
        if (paths[n].type == t) {
          localNodes[validDstNodeCount++] = n;
        }
      }
      if (validDstNodeCount == 0) continue;
      // Shuffle by gcu EFML device number so that GCUs
      // on the same PCI switch with multiple NICs don't
      // use the same one as first choice. For 1-net on
      // the same PCI switch, every GCU dev use same net.
      // For x-net on the same PCI switch, we give
      // following example to illustrate:
      // dev(0) [n1, n2, n3] << 0 => [n1, n2, n3]
      // dev(1) [n1, n2, n3] << 1 => [n2, n3, n1]
      // dev(2) [n1, n2, n3] << 2 => [n3, n1, n2]
      // dev(3) [n1, n2, n3] << 3 => [n1, n2, n3]
      epTopoNode* gcu;
      EP_CHECK(system_->getNode(srcType, g, &gcu));
      for (int r = 0; r < gcu->gcu.dev % validDstNodeCount; r ++) {
        int net0 = localNodes[0];
        for (int i = 0; i < validDstNodeCount-1; i ++)
          localNodes[i] = localNodes[i+1];
        localNodes[validDstNodeCount-1] = net0;
      }
      // Append NICs to list
      for (int i = 0; i < validDstNodeCount; i ++) {
        int n = localNodes[i];
        int found = 0;
        while (dstIdxs[found] != n && found < nodeCount)
          found ++;
        if (found == nodeCount) dstIdxs[nodeCount++] = n;
      }
    }
  }

  *retNodeCount = nodeCount;
  return epSuccess;
}

// Start channel search from net, this is for inter-node process.
epResult_t epTopoGraphHdl::searchChannelFrom(epTopoNodeType fromType, epTopoGraphHdl* optimalGraph, int* time) {
  int step = 0;
  // We first choose a net node as traffic inbound, and than choose a local cpu for this net
  // as firstgcu in the channel, which process the traffic from net we add before.
  int dstIdx[EP_TOPO_MAX_NODES];
  int count=0;
  EP_CHECK(searchSelectDstNodes(GCU, -1, fromType, dstIdx, &count));
  for (int i = 0; i < count; i++) {
    if (searchTry(fromType/* nonsense */, -1, fromType, dstIdx[i])) {
      EP_CHECK(system_->addChannelToNode(fromType, dstIdx[i], nChannels_));
      epTopoNode* node;
      EP_CHECK(system_->getNode(fromType, dstIdx[i], &node));
      if (fromType == NET) {
        inter_[nChannels_ * 2] = node->id;
        latencyInter_ = node->net.latency;
      }

      // If graph have one or more channels, always try to replay the last channel.
      if (nChannels_ > 0) {
        int count = 0;
        int firstGcuIdx = -1;
        EP_CHECK(searchSelectGcuOrderReplay(-1, GCU, &firstGcuIdx, &count));
        EP_CHECK(searchChannelFrom(fromType, dstIdx[i], GCU, firstGcuIdx, optimalGraph, step, GCU_ORDER_REPLAY, time));
      }
      // Use the most local gcu of net.
      int firstGcus[EP_TOPO_MAX_GCUS] = {-1};
      int gcuCount = 0;
      EP_CHECK(system_->getLocalGcus(fromType, dstIdx[i], bwInter_, firstGcus, &gcuCount));
      if (gcuCount) {
        // In the first loop, avoid using gcu in both directions between channels (one channel
        // sending from that GCU and one channel receiving to that GCU), since that usually leads
        // to lower BW.
        for (int tryGcuBidir = 0; tryGcuBidir < 2; tryGcuBidir ++) {
          for (int j = 0; j < gcuCount; ++ j) {
            int gcuUsed = system_->gcuPciBw(firstGcus[j]) > 0 ? 0 : 1;
            if (gcuUsed == tryGcuBidir)
              EP_CHECK(searchChannelFrom(fromType, dstIdx[i], GCU, firstGcus[j], optimalGraph,
                  step, GCU_ORDER_SCORE, time));
          }
        }
      }
      EP_CHECK(system_->removeChannelFromNode(fromType, dstIdx[i], nChannels_));
      EP_CHECK(searchRestore(fromType/* nonsense */, -1, fromType, dstIdx[i]));
    }
  }
  return epSuccess;
}

/* Search Patterns
 *
 *     Intra-node
 * Ring            : GCU a -> GCU b -> .. -> GCU x -> GCU a
 *
 *     Inter-node
 * Ring            : NET n -> GCU a -> GCU b -> .. -> GCU x -> NET n (or m if crossNic)
 *
 */
epResult_t epTopoGraphHdl::searchGraphRec(epTopoGraphHdl* optimalGraph, int* time) {
  // For inter-node and intra-node, searching graph is different, we use different workflow
  // to process it.
  int nets = system_->netCount();
  if (nets > 0) {
    // In inter-node, we begin search channel from net.
    EP_CHECK(searchChannelFrom(NET, optimalGraph, time));
  } else {
    // In intra-node, we begin search channel from gcu.
    int step = 0;
    int firstGcuIdx = 0;
    int count = 0;
    if (nChannels_ && pattern_ != EP_TOPO_PATTERN_BALANCED_TREE) {
      // If we already find one or more channel in graph, firstly we try to replay last channel
      // as much as we can.
      EP_CHECK(searchSelectGcuOrderReplay(-1, GCU, &firstGcuIdx, &count));
      EP_CHECK(searchChannelFrom(GCU/* nonsense */, -1, GCU, firstGcuIdx, optimalGraph,
          step, GCU_ORDER_REPLAY, time));
    }

    int gcus = system_->gcuCount();
    // Track which GCUs have been used as starting points
    bool usedAsStart[EP_TOPO_MAX_NODES] = {false};
    for(int c = 0; c < nChannels_; c++) {
      int firstGcuIdx;
      EP_CHECK(system_->rankToIndex(intra_[c * gcus], &firstGcuIdx));
      usedAsStart[firstGcuIdx] = true;
    }

    // Always first try to search a channel from gcu 0. If search channel failed, try gcu 1
    // as first gcu in channel, and so on. The next gcu will try will be ordered by score.
    for(int firstGcuIdx = 0; firstGcuIdx < gcus; firstGcuIdx++) {
      if (pattern_ == EP_TOPO_PATTERN_BALANCED_TREE && usedAsStart[firstGcuIdx]) {
        continue;
      }
      EP_CHECK(searchChannelFrom(GCU /* nonsense */, -1, GCU, firstGcuIdx, optimalGraph,
          step, GCU_ORDER_SCORE, time));
    }
  }
  return epSuccess;
}

epResult_t epTopoGraphHdl::compute() {
  int gcus = system_->gcuCount();
  // Value nets here have two meanings. The first meaning is the number of net in the system
  // as we can see. The second meaning is current compute graph process is inter-node or
  // intra-node. Here nets is equal to 0 means intra-node, search channel process will begin
  // from gcu. Meanwhile, nets exist in topology system indicate that channel search process
  // will begin from net.
  int nets = system_->netCount();

  // Parameter crossNic may come into effect for inter-node connection. If crossNic enabled,
  // we always try to use crossNic graph firstly. If disabled, using same nic would be first
  // choice.
  crossNic_ = epParamCrossNic();
  int crossNic = (nets > 1) && crossNic_ ? 1 : 0;
  if (crossNic_ == 2) crossNic_ = 0;

  bwIntra_ = bwInter_ = 0;
  latencyInter_ = 0;

  typeIntra_ = gcus == 1 ? PATH_LOC : PATH_LARE;
  typeInter_ = PATH_PIX;

  nChannels_ = 0;

  // TODO : need to optimization the code here
  // Current searching Graph is stored in optimalGraph
  // which is initialized by this
  epTopoGraphHdl optimalGraph = *this;

  // We know bwIntra and bwInter always the same. But inter-node and intra-node use different
  // speed array cause there are extra bandwidth(IB/RoCE/Socket) for inter-node connection.
  int nspeeds = 0;
  float* speedArray = nullptr;
  if (nets > 0) {
    // Speed array for inter-node.
    nspeeds = NSPEEDSINTER;
    speedArray = speedArrayInter;
  } else {
    // Speed array for intra-node.
    nspeeds = NSPEEDSINTRA;
    speedArray = speedArrayIntra;
  }
  // We always try to find the floor value of system max bandwidth as the first shot for parameter
  // bwIntra and bwInter.
  int speedIndex = 0;
  while (speedArray[speedIndex] > system_->maxBw() && speedIndex < nspeeds-1) speedIndex++;
  bwIntra_ = bwInter_ = speedArray[speedIndex];

  // TODO : need to optimization the time system.
  int64_t globalTimeout = pattern_ == EP_TOPO_PATTERN_BALANCED_TREE ? EP_SEARCH_TREE_GLOBAL_TIMEOUT : EP_SEARCH_GLOBAL_TIMEOUT;
  epTopoPathType maxTypeIntra = PATH_SYS;

  int time;

search:
  time = pattern_ == EP_TOPO_PATTERN_BALANCED_TREE ? EP_SEARCH_TREE_TIMEOUT : EP_SEARCH_TIMEOUT;
  globalTimeout -= time;

  // Every time we begin search graph, parameter nChannels will be reset to 0, which means zero
  // channel find in graph.
  nChannels_ = 0;
  EP_CHECK(searchGraphRec(&optimalGraph, &time));

#if 0
  char printLine[512];
  int lineIdx = snprintf(printLine, 512, "Pattern %d, crossNic %d, Bw %g/%g, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%g/%g %s", optimalGraph.pattern, optimalGraph.crossNic, optimalGraph.bwInter, optimalGraph.bwIntra, optimalGraph.typeInter, optimalGraph.typeIntra, optimalGraph.minChannels, optimalGraph.maxChannels, optimalGraph.sameChannels, graph->nChannels, graph->bwInter, graph->bwIntra, time == 0 ? "TIMEOUT" : time == -1 ? "PERFECT" : "");
  for (int c=0; c<graph->nChannels; c++) {
    lineIdx += snprintf(printLine + lineIdx, 512 - lineIdx, "%2d : ", c);
    for (int g=0; g<gcus; g++) {
      lineIdx += snprintf(printLine + lineIdx, 512 - lineIdx, "%d ", graph->intra[c*gcus+g]);
    }
    lineIdx += snprintf(printLine + lineIdx, 512 - lineIdx, "[%d %d]", graph->inter[c*2+0], graph->inter[c*2+1]);
    //lineIdx += snprintf(printLine + lineIdx, 512 - lineIdx, "\n");
  }
  INFO(EP_INIT, printLine);

#endif
  // Optimal solution, stop here.
  if (time == -1) {
    goto done;
  }

  // All channel used inter
  if (optimalGraph.nChannels_ * optimalGraph.bwInter_ >= system_->totalBw()) {
    goto done;
  }

  // Timeout.
  if (time != -1) globalTimeout += time;
  else globalTimeout = EP_SEARCH_GLOBAL_TIMEOUT;
  if (globalTimeout < 0 && nChannels_) {
    goto done;
  }

  // Type intra iterating.
  maxTypeIntra = nets > 0 ? typeInter_ : PATH_SYS;
  if (typeIntra_ < maxTypeIntra && (nChannels_ == 0 || typeIntra_ < optimalGraph.typeIntra_)) {
    typeIntra_ = static_cast<epTopoPathType>(static_cast<int>(typeIntra_) + 1);
    goto search;
  }
  typeIntra_ = gcus == 1 ? PATH_LOC : PATH_LARE;

  // Type inter iterating.
  if (nets > 0 && typeInter_ < PATH_SYS && (optimalGraph.nChannels_ == 0 || typeInter_ < optimalGraph.typeInter_)) {
    typeInter_ = static_cast<epTopoPathType>(static_cast<int>(typeInter_) + 1);
    goto search;
  }
  typeInter_ = PATH_PIX;

  // Try with crossNic if permitted.
  if (crossNic && crossNic_ == 0) {
    crossNic_ = crossNic;
    goto search;
  }
  crossNic_ = 0;

  // Decrease bandwidth until we find a solution.
  if ((speedIndex < nspeeds-1) && (nChannels_ == 0 || (speedArray[speedIndex+1]/bwInter_ > .49))) {
    bwInter_ = bwIntra_ = speedArray[++speedIndex];
    goto search;
  }
  speedIndex = 0;
  while (speedArray[speedIndex] > system_->maxBw() && speedIndex < nspeeds-1) speedIndex++;
  bwIntra_ = bwInter_ = speedArray[speedIndex];

done:
  *this = optimalGraph;
  //use default order when force order enabled.
  int algoForceOrderEnable = epParamAlgoForceOrderEnable();
  if (algoForceOrderEnable) nChannels_ = 0;

  // Exception: can't find a graph with one channel, falling back to simple
  // order: 0->1->2->3
  if (nChannels_ == 0) {
    INFO(EP_INIT, "Could not find a path for pattern %s, falling back to simple order", graphPatternStr[pattern_]);
    for (int g = 0; g < gcus; g ++) {
      epTopoNode* gcu;
      EP_CHECK(system_->getNode(GCU, g, &gcu));
      intra_[g] = gcu->gcu.rank;
    }
    //  EP_CHECK(system_->getGcuRankByIndex(i, intra_ + i));
    inter_[0] = inter_[1] = 0;
    bwIntra_ = bwInter_ = 0.1;
    typeIntra_ = typeInter_ = PATH_SYS;
    nChannels_ = 1;
  }

  // Situation we need to duplicate channel.
  efmlDeviceArchitecture_t arch;
  EP_CHECK(system_->getDeviceType(&arch));
  if (arch == EFML_DEVICE_ARCH_GCU300) {
    if (bwIntra_ >= 46.0) {
      int dupChannels  = std::min(nChannels_ * 2, maxChannels_);
      memcpy(intra_ + nChannels_ * gcus, intra_, (dupChannels - nChannels_) * gcus * sizeof(intra_[0]));
      memcpy(inter_ + nChannels_ * 2, inter_, (dupChannels - nChannels_) * 2 * sizeof(inter_[0]));
      bwIntra_ /= DIVUP(dupChannels, nChannels_);
      bwInter_ /= DIVUP(dupChannels, nChannels_);
      nChannels_ = dupChannels;
    } else {
      int dupChannels = 8;
      for (int i=nChannels_; i<dupChannels; i++) {
        memcpy(intra_ + i * gcus, intra_, gcus * sizeof(intra_[0]));
        memcpy(inter_ + i * 2, inter_, 2 * sizeof(inter_[0]));
      }
      bwIntra_ /= DIVUP(dupChannels, nChannels_);
      bwInter_ /= DIVUP(dupChannels, nChannels_);
      nChannels_ = dupChannels;
    }
  }

  if (arch == EFML_DEVICE_ARCH_GCU400 && pattern_ == EP_TOPO_PATTERN_BALANCED_TREE && nChannels_ < maxChannels_) {
    int dupChannels = std::min(nChannels_ * 2, maxChannels_);
    memcpy(intra_ + nChannels_ * gcus, intra_, (dupChannels - nChannels_) * gcus * sizeof(intra_[0]));
    memcpy(inter_ + nChannels_ * 2, inter_, (dupChannels - nChannels_) * 2 * sizeof(inter_[0]));
    bwIntra_ /= DIVUP(dupChannels, nChannels_);
    bwInter_ /= DIVUP(dupChannels, nChannels_);
    nChannels_ = dupChannels;
  }

  return epSuccess;
}


void epTopoGraphHdl::print() {
  INFO(EP_GRAPH, "Pattern %s, crossNic %d, nChannels %d, bw %.2f/%.2f, type %s/%s ",
      graphPatternStr[pattern_], crossNic_, nChannels_, bwIntra_, bwInter_, topoPathTypeStr[typeIntra_],
      topoPathTypeStr[typeInter_]);
  int gcus = system_->gcuCount();

  char line[1024];
  for (int c = 0; c < nChannels_; c ++) {
    sprintf(line, "channel %2d :", c);
    int offset = strlen(line);
    if (system_->netCount() > 0) {
      // NET/(dev)
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], EP_TOPO_ID_SYSTEM_ID(inter_[2*c]), EP_TOPO_ID_LOCAL_ID(inter_[2*c]));
      offset = strlen(line);
    }
    for (int i=0; i<gcus; i++) {
      // GCU/(rank)
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[GCU], intra_[gcus*c+i]);
      offset = strlen(line);
    }
    if (system_->netCount() > 0) {
      // NET/(dev)
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], EP_TOPO_ID_SYSTEM_ID(inter_[2*c+1]), EP_TOPO_ID_LOCAL_ID(inter_[2*c+1]));
      offset = strlen(line);
    }
    INFO(EP_GRAPH, "%s", line);
  }
}

// Add xml channel
epResult_t epTopoGraphHdl::addChannel(epXmlNode *xmlChannel, int c) {
  int gcus = system_->gcuCount();
  int64_t* inter = inter_ + 2 * c;
  int* intra = intra_ + gcus * c;
  int n = 0, g = 0;
  for (size_t i = 0; i < xmlChannel->childSize(); i ++) {
    epXmlNode* node = xmlChannel->childAt(i);
    int dev;
    EP_CHECK(xmlNodeGetAttr(node, "dev", &dev));
    if (strcmp(node->name(), "net") == 0) {
      inter[n ++] = dev;
    } else if (strcmp(node->name(), "gcu") == 0) {
      int rank = -1;
      EP_CHECK(xmlNodeGetAttr(node, "rank", &rank));
      if (rank == -1) {
        WARN("XML Import Channel : rank %ld not found.", rank);
        return epSystemError;
      }
      intra[g++] = rank;
    }
  }
  return epSuccess;
}

epResult_t epTopoGraphHdl::getGraphFromXml(epXml* xml, int* retChannels) {
  epXmlNode* xmlGraphs = xml->findNode("graphs");
  // Find Graph in xml of given pattern, than add channel to Graph.
  for (size_t c = 0; c < xmlGraphs->childSize(); c ++) {
    epXmlNode* xmlGraph = xmlGraphs->childAt(c);
    const char* patternStr;
    EP_CHECK(xmlNodeGetAttr(xmlGraph, "pattern", &patternStr));
    if (strcmp(graphPatternStr[pattern_], patternStr) != 0) continue;

    int crossNic;
    EP_CHECK(xmlNodeGetAttr(xmlGraph, "crossnic", &crossNic));
    if (crossNic_ == 0 && crossNic == 1) continue;
    crossNic_ = crossNic;

    EP_CHECK(xmlNodeGetAttr(xmlGraph, "nchannels", &nChannels_));
    EP_CHECK(xmlNodeGetAttr(xmlGraph, "speedintra", &bwIntra_));
    EP_CHECK(xmlNodeGetAttr(xmlGraph, "speedinter", &bwInter_));
    xmlNodeGetAttrDefault(xmlGraph, "latencyinter", &latencyInter_, 0.0);

    const char* str;
    EP_CHECK(xmlNodeGetAttr(xmlGraph, "typeintra", &str));
    typeIntra_ = strToPathType(str);
    EP_CHECK(xmlNodeGetAttr(xmlGraph, "typeinter", &str));
    typeInter_ = strToPathType(str);

    for (size_t c = 0; c < xmlGraph->childSize(); c ++) {
      EP_CHECK(addChannel(xmlGraph->childAt(c), c));
    }
    nChannels_ = xmlGraph->childSize();
    *retChannels = nChannels_;
  }
  return epSuccess;
}

epResult_t epTopoGraphHdl::getXmlFromChannel(int channel, epXml *xml, epXmlNode* graph) {
  int gcus = system_->gcuCount();
  int nets = system_->netCount();

  int64_t* inter = inter_ + 2 * channel;
  int* intra = intra_ + gcus * channel;

  epXmlNode* xmlChannel;
  EP_CHECK(xml->createNode("channel", &xmlChannel));
  xmlChannel->setParent(graph);
  EP_CHECK(graph->addChild(xmlChannel));

  // Xml net.
  epXmlNode* node;
  if (nets > 0) {
    EP_CHECK(xml->createNode("net", &node));
    node->setParent(xmlChannel);
    EP_CHECK(xmlChannel->addChild(node));
    EP_CHECK(node->attributes()->create("dev", inter[0]));
  }
  // Xml gcu.
  for (int g = 0; g < gcus; g ++) {
    EP_CHECK(xml->createNode("gcu", &node));
    node->setParent(xmlChannel);
    EP_CHECK(xmlChannel->addChild(node));

    int index;
    if (system_->rankToIndex(intra[g], &index) != epSuccess) {
      WARN("XML Channel : rank %d not found in topology system.", intra[g]);
      return epInternalError;
    }
    
    epTopoNode* gcu;
    EP_CHECK(system_->getNode(GCU, index, &gcu));
    int systemId = EP_TOPO_ID_SYSTEM_ID(gcu->id);
    int dev = EP_TOPO_ID(systemId, gcu->gcu.dev);
    EP_CHECK(node->attributes()->create("dev", dev));
    EP_CHECK(node->attributes()->create("rank", gcu->gcu.rank));
  }
  // Xml net.
  if (nets > 0) {
    EP_CHECK(xml->createNode("net", &node));
    node->setParent(xmlChannel);
    EP_CHECK(xmlChannel->addChild(node));
    EP_CHECK(node->attributes()->create("dev", inter[1]));
  }
  return epSuccess;
}

epResult_t epTopoGraphHdl::getXmlFromGraph(epXml *xml) {
  epXmlNode* xmlGraph;
  EP_CHECK(xml->createNode("graph", &xmlGraph));
  epXmlNode* xmlGraphs = xml->findNode("graphs");
  xmlGraph->setParent(xmlGraphs);
  EP_CHECK(xmlGraphs->addChild(xmlGraph));

  EP_CHECK(xmlGraph->attributes()->create("pattern", graphPatternStr[pattern_]));
  EP_CHECK(xmlGraph->attributes()->create("crossnic", crossNic_));
  EP_CHECK(xmlGraph->attributes()->create("nchannels", nChannels_));
  EP_CHECK(xmlGraph->attributes()->create("speedintra", bwIntra_));
  EP_CHECK(xmlGraph->attributes()->create("speedinter", bwInter_));
  EP_CHECK(xmlGraph->attributes()->create("latencyinter", latencyInter_));
  EP_CHECK(xmlGraph->attributes()->create("typeintra", topoPathTypeStr[typeIntra_]));
  EP_CHECK(xmlGraph->attributes()->create("typeinter", topoPathTypeStr[typeInter_]));

  for (int c = 0; c < nChannels_; c ++) {
    EP_CHECK(getXmlFromChannel(c, xml, xmlGraph));
  }
  return epSuccess;
}

void epTopoGraphHdl::copyRaw(epTopoGraph* graph) {
  graph->nChannels = nChannels_;
  graph->pattern = pattern_;
  graph->bwIntra = bwIntra_;
  graph->bwInter = bwInter_;
  graph->typeIntra = typeIntra_;
  graph->typeInter = typeInter_;

  int gcus = system_->gcuCount();
  memcpy(graph->intra, intra_, gcus * sizeof(intra_[0]) * nChannels_);
  memcpy(graph->inter, inter_, 2 * sizeof(inter_[0]) * nChannels_);
}

epResult_t epTopoGraphHdl::getNetDev(int rank, int channelId, int64_t* netId, int* netDev) {
  // TODO : when rank isn't connect to net
  int c = channelId % nChannels_;
  int gcus = system_->gcuCount();
  int index = intra_[c * gcus] == rank ? 0 : 1;
  *netId = inter_[c * 2 + index];
  EP_CHECK(system_->topoIdToNetDev(*netId, netDev));
  return epSuccess;
}

void epTopoGraphHdl::updateGraphInfo(epTopoGraph* graph) {
  nChannels_ = graph->nChannels;
  pattern_ = static_cast<epTopoGraphPatternType>(graph->pattern);
  bwIntra_ = graph->bwIntra;
  bwInter_ = graph->bwInter;
  typeIntra_ = static_cast<epTopoPathType>(graph->typeIntra);
  typeInter_ = static_cast<epTopoPathType>(graph->typeInter);
}
const char* epTopoGraphHdl::getPatternDesc() {
  return graphPatternStr[pattern_];
}
epMeshGraphHandler::epMeshGraphHandler(epTopoSystem* system, struct epTopoGraph* graph):
  epTopoGraphHdl(system, graph){
}
epMeshGraphHandler::~epMeshGraphHandler() {
}

epResult_t epMeshGraphHandler::compute() {
  int totalGcuNum = system_->gcuCount();
  if (totalGcuNum == 0) {
    WARN("there is no gcu for full mesh search");
    return epSystemError;
  }
  crossNic_ = 0;
  latencyInter_ = 0;
  typeIntra_ = totalGcuNum == 1 ? PATH_LOC : PATH_LARE;
  typeInter_ = PATH_LARE;
  bwIntra_ = LOC_BW;
  bwInter_ = 0.0;
  epTopoNode* srcNode = nullptr;
  epTopoNode* dstNode = nullptr;
  nChannels_ = MAXCHANNELS;
  bool hasOtherPeer = false;
  for (int srcGcu = 0; srcGcu < totalGcuNum; srcGcu++) {
    EP_CHECK(system_->getNode(GCU, srcGcu, &srcNode));
    intra_[srcGcu] = srcNode->gcu.rank;
    std::vector<int> linkPortCount(totalGcuNum, 0);
    linkPortCount[srcGcu] = MAXCHANNELS;
    for (int dstGcu = 0; dstGcu < totalGcuNum; dstGcu++) {
      EP_CHECK(system_->getNode(GCU, dstGcu, &dstNode));
      epTopoLinkList* gcuLinkList = nullptr;
      EP_CHECK(system_->getPath(GCU, srcGcu, GCU, dstGcu, &gcuLinkList));
      for (int i = 0; i < gcuLinkList->count; i++) {
        epTopoLink* link = gcuLinkList->list[i];
        epTopoNode* rmtNode = link->remNode;
        if (rmtNode->type != GCU) continue;
        if (rmtNode->gcu.rank != dstNode->gcu.rank) continue;

        bwIntra_ = std::min(gcuLinkList->bw, bwIntra_);
        if (link->type == LINK_LARE) {
          if (link->portCount[dstGcu] == 0) continue;
          linkPortCount[dstGcu] = link->portCount[dstGcu];
          for (int c=0;c<linkPortCount[dstGcu];c++) {
            intra_[c*totalGcuNum + srcGcu] = srcNode->gcu.rank;
            intra_[c*totalGcuNum + dstGcu] = rmtNode->gcu.rank;
          }
          typeIntra_ = std::max(typeIntra_, PATH_LARE);
        } else if (link->type == LINK_PCI) {
          linkPortCount[dstGcu] = 1;
          for (int c=0;c<linkPortCount[dstGcu];c++) {
            intra_[c*totalGcuNum + srcGcu] = srcNode->gcu.rank;
            intra_[c*totalGcuNum + dstGcu] = rmtNode->gcu.rank;
          }
          typeIntra_ = std::max(typeIntra_, PATH_PIX);
        }
      }
    }

    for (size_t i=0;i<linkPortCount.size();i++) {
      if ((size_t)srcGcu == i) continue;
      hasOtherPeer = true;
      nChannels_ = std::min(nChannels_, linkPortCount[i]);
    }
  }
  nChannels_ = hasOtherPeer? nChannels_ : 1;

  if (nChannels_ < minChannels_) {
    WARN("search mesh channels %d less than : %d", nChannels_, minChannels_);
    return epSystemError;
  }

  if (nChannels_ < maxChannels_) {
    // duplicate channel to maxChannels_
    int originalChannels = nChannels_;
    for (int i = nChannels_; i < maxChannels_; i++) {
      int srcChannel = i % originalChannels;
      memcpy(intra_ + i * totalGcuNum, intra_ + srcChannel * totalGcuNum, totalGcuNum * sizeof(intra_[0]));
    }
    bwIntra_ /= DIVUP(maxChannels_, originalChannels);
    nChannels_ = maxChannels_;
  }
  nChannels_ = std::min(nChannels_, maxChannels_);

  //use default order when force order enabled.
  int algoForceOrderEnable = epParamAlgoForceOrderEnable();
  if (algoForceOrderEnable){
    for (int g = 0; g < totalGcuNum; g ++) {
      epTopoNode* gcu;
      EP_CHECK(system_->getNode(GCU, g, &gcu));
      intra_[g] = gcu->gcu.rank;
    }
    for (int i=1; i<nChannels_; i++) {
      memcpy(intra_ + i * totalGcuNum, intra_, totalGcuNum * sizeof(intra_[0]));
    }
  }
  return epSuccess;
}
const char* epMeshGraphHandler::getPatternDesc() {
  return "mesh";
}
void epMeshGraphHandler::print() {
  INFO(EP_GRAPH, "Pattern mesh, crossNic %d, nChannels %d, bw %.2f/%.2f, type %s/%s ",
                   crossNic_, nChannels_, bwIntra_, bwInter_, topoPathTypeStr[typeIntra_], topoPathTypeStr[typeInter_]);
  int totalGcus = system_->getCount(GCU);

  char line[1024];
  for (int c = 0; c < nChannels_; c ++) {
    sprintf(line, "channel %2d : {", c);
    int offset = strlen(line);
    for (int i=0; i<totalGcus; i++) {
      // GCU/(rank)
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[GCU], intra_[totalGcus*c+i]);
      offset = strlen(line);
    }
    INFO(EP_GRAPH, "%s }", line);
  }
}
void epMeshGraphHandler::copyRaw(epTopoGraph* graph) {
  graph->nChannels = nChannels_;
  graph->pattern = pattern_;
  graph->bwIntra = bwIntra_;
  graph->bwInter = bwInter_;
  graph->typeIntra = typeIntra_;
  graph->typeInter = typeInter_;

  int gcus = system_->gcuCount();
  memcpy(graph->intra, intra_, gcus * sizeof(intra_[0]) * nChannels_);
  memcpy(graph->inter, inter_, 2 * sizeof(inter_[0]) * nChannels_);
}


