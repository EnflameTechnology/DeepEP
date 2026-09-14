#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "checks.h"
#include "graph.h"
#include "topo.h"
#include "alloc.h"
#include "param.h"

epResult_t epTopoSystem::topoIdToIndex(epTopoNodeType type, int64_t id, int* index) {
  *index = -1;
  for (int i = 0; i < nodes_[type].count; i ++) {
    if (nodes_[type].nodes[i].id == id) {
      *index = i;
      return epSuccess;
    }
  }
  return epInternalError;
}

epResult_t epTopoSystem::getPath(epTopoNodeType type1, int index1, epTopoNodeType type2, int index2, epTopoLinkList** retPath) {
  epTopoNode* srcNode;
  EP_CHECK(getNode(type1, index1, &srcNode));
  if (srcNode->paths[type2] == nullptr) {
    EP_CHECK(epCalloc(&srcNode->paths[type2], nodes_[type2].count));
  }

  if (index2 < nodes_[type2].count) {
    *retPath = srcNode->paths[type2] + index2;
    return epSuccess;
  }

  WARN("Topo : Could not find node of type %d index %lx", type2, index2);
  return epInternalError;
}

epResult_t epTopoSystem::setPaths(epTopoNodeType type, int index) {
  epTopoNode* dstNode;
  EP_CHECK(getNode(type, index, &dstNode));

  // In breadth-first search, we use nodeQueue to store the node that already
  // calculated path to dstNode before. Meanwhile, using  nextNodeQueue store the
  // node we are working on, which will than copied to nodeQueue.
  std::vector<std::pair<int, epTopoNode*>> nodeQueue;
  std::vector<std::pair<int, epTopoNode*>> nextNodeQueue;
  nodeQueue.reserve(EP_TOPO_MAX_NODES);
  nextNodeQueue.reserve(EP_TOPO_MAX_NODES);

  nodeQueue.push_back(std::make_pair(index, dstNode));
  epTopoLinkList* basePath;
  EP_CHECK(getPath(type, index, type, index, &basePath));
  basePath->count = 0;
  basePath->bw = LOC_BW;
  basePath->type = PATH_LOC;

  while (!nodeQueue.empty()) {
    nextNodeQueue.clear();
    for (size_t n = 0; n < nodeQueue.size(); n ++) {
      int idx = nodeQueue[n].first;
      epTopoNode* node = nodeQueue[n].second;

      epTopoLinkList* path;
      EP_CHECK(getPath(node->type, idx, type, index, &path));

      // Traverse the node connected to the node we already have path.
      // In function getSystem, we have sorted all links in one node,
      // It worked here.
      for (int l = 0; l < node->nlinks; l ++) {
        epTopoLink* link = node->links + l;
        epTopoNode* curNode = link->remNode;
        int curIdx;
        EP_CHECK(topoIdToIndex(curNode->type, curNode->id, &curIdx));
        epTopoLinkList* curPath;
        EP_CHECK(getPath(curNode->type, curIdx, type, index, &curPath));

        // Current bandwidth of path from curNode to dstNode.
        float bw = std::min(path->bw, link->bw);

        // When node is dstNode, this is the first time we go through the code,
        // Just goto update path, including bandwidth, path link list, path type.
        // When node is GCU, for our device connection, link from GCU can be
        // LINK_LOC, LINK_LARE, LINK_PCI. Here if link's type not equal to LINK_LARE,
        // we ignore the node self and node pci ( actually it is reverse link from pci
        // to gcu we have used).

        // Due to device constraints, gcu can't show up at the middle of path. So we add
        // an extra condition to control it.
        // Indeed, we can classify setPath function to two type of call, one is GCU type,
        // the other is non-GCU type. ... "add more explanation here"
        if ( node != dstNode && (
            ((node->type == GCU) && link->type != LINK_LARE) ||
            ((node->type == GCU) && curNode->type == PCI) ||
            ((node->type == GCU) && path->count >= 1) // can't routing through a GCU
            )) continue;

        // We don't update current path until we found a better path solution.
        // So what is the better path solution? the first one is always the time
        // of first to calculate path; and the second one is when current path have
        // larger path count as well as lower bandwidth(why is that).
        if ((curPath->bw == 0 || curPath->count > path->count) && curPath->bw < bw) {
          // We find reverse link which is from curNode to node firstly, in order to
          // connect with path.
          for (int l = 0; l < curNode->nlinks; l ++) {
            if (curNode->links[l].remNode == node) {
              curPath->list[0] = curNode->links + l;
              break;
            }
          }
          if (curPath->list[0] == nullptr) {
            WARN("Failed to find reverse path from curNode %d/%lx nlinks %d to node %d/%lx",
                 curNode->type, curNode->id, curNode->nlinks, node->type, node->id);
            return epInternalError;
          }
          // copy the rest of the path next.
          for (int i = 0; i < path->count; i++) curPath->list[i+1] = path->list[i];
          curPath->count = path->count + 1;
          curPath->bw = bw;

          // Don't consider LINK_NET as we only care about the path of NICs.
          int type_l = link->type == LINK_NET ? LINK_LOC : link->type;
          // Now consider about the path's type. You can just list all possible type and
          // than figure out the following condition, detail information see :
          // "add link here"
          // Differentiate between one and multiple PCI switches.
          if (node->type == PCI && curNode->type == PCI) type_l = PATH_PXB;
          // Consider a path going through the CPU as PATH_PHB.
          if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) type_l = PATH_PHB;
          curPath->type = std::max(path->type, type_l);

          // We already obtain all path information for curNode, now we can add it to nextNodeQueue
          // for temporary storage.
          if (std::find(nextNodeQueue.begin(), nextNodeQueue.end(), std::make_pair(curIdx, curNode)) == nextNodeQueue.end()) {
            nextNodeQueue.push_back(std::make_pair(curIdx, curNode));
           }
        }
      }
    }
    nodeQueue.assign(nextNodeQueue.begin(), nextNodeQueue.end());
  }
  return epSuccess;
}

epResult_t epTopoSystem::getLocalCpu(int g, int* retCpu) {
  int minHops = 0;
  int localCpu = -1;
  epTopoLinkList* paths = nodes_[GCU].nodes[g].paths[CPU];
  for (int c = 0; c < nodes_[CPU].count; c ++) {
    int hops = paths[c].count;
    if (minHops == 0 || hops < minHops) {
      localCpu = c;
      minHops = hops;
    }
  }
  if (localCpu == -1) {
    WARN("Error : could not find CPU close to GCU %d", g);
    return epInternalError;
  }
  *retCpu = localCpu;
  return epSuccess;
}

void epTopoSystem::addInterStep(epTopoNodeType tx, int ix, epTopoNodeType t1, int i1, epTopoNodeType t2, int i2) {
  epTopoNode* cpuNode = nodes_[tx].nodes + ix;
  epTopoNode* srcNode = nodes_[t1].nodes + i1;

  int l=0;
  // Connect links from srcNode to CPU.
  for (int i = 0; i < srcNode->paths[tx][ix].count; i++)
    srcNode->paths[t2][i2].list[l++] = srcNode->paths[tx][ix].list[i];
  // Connect links from CPU to dstNode.
  for (int i = 0; i < cpuNode->paths[t2][i2].count; i++)
    srcNode->paths[t2][i2].list[l++] = cpuNode->paths[t2][i2].list[i];

  // Update path.
  srcNode->paths[t2][i2].count = l;
  srcNode->paths[t2][i2].type =
      std::max(srcNode->paths[tx][ix].type, cpuNode->paths[t2][i2].type);
  srcNode->paths[t2][i2].bw =
      std::min(srcNode->paths[tx][ix].bw, cpuNode->paths[t2][i2].bw);
}

epResult_t epTopoSystem::removeNode(epTopoNodeType type, int idx) {
  epTopoNode* delNode = nodes_[type].nodes + idx;
  for (int t = 0; t < EP_TOPO_NODE_TYPES; t ++) {
    // Remove path
    free(delNode->paths[t]);
    // Remove link to delNode
    for (int n = 0; n < nodes_[t].count; n ++) {
      epTopoNode* node = nodes_[t].nodes + n;
      if (node == delNode) continue;
      for (int l = 0; l < node->nlinks; l ++) {
        while (l < node->nlinks && node->links[l].remNode == delNode) {
          memmove(node->links + l, node->links + l + 1,
                  (node->nlinks-(l+1))*sizeof(epTopoLink));
          node->nlinks--;
        }
        // Offset for other node after delNode
        if (l < node->nlinks &&
            node->links[l].remNode->type == type &&
            node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  memmove(delNode, delNode + 1, (nodes_[type].count-(idx+1))*sizeof(epTopoNode));
  nodes_[type].count--;
  return epSuccess;
}

void epTopoSystem::removePaths(epTopoNodeType type) {
  for (int t = 0; t < EP_TOPO_NODE_TYPES; t ++) {
    // Remove links _to_ the given type.
    for (int n = 0; n < nodes_[t].count; n ++) {
      epTopoNode* node = nodes_[t].nodes + n;
      free(node->paths[type]);
      node->paths[type] = nullptr;
    }
    // Remove links _from_ the given type.
    for (int n = 0; n < nodes_[type].count; n ++) {
      epTopoNode* node = nodes_[type].nodes + n;
      free(node->paths[t]);
      node->paths[t] = nullptr;
    }
  }
}

#ifdef ENABLE_TRACE
void epTopoSystem::printNodePaths(epTopoNode* node) {
  char line[1024];
  INFO(EP_GRAPH, "Paths from %s/%lx-%lx :", topoNodeTypeStr[node->type],
                                                     EP_TOPO_ID_SYSTEM_ID(node->id), EP_TOPO_ID_LOCAL_ID(node->id));
  for (int t = 0; t < EP_TOPO_NODE_TYPES; t ++) {
    if (node->paths[t] == nullptr) continue;
    for (int n = 0; n < nodes_[t].count; n ++) {
      line[0] = 0;
      int offset = 0;
      for (int i = 0; i < node->paths[t][n].count; i ++) {
        epTopoLink* link = node->paths[t][n].list[i];
        epTopoNode* remNode = link->remNode;
        sprintf(line+offset, "--%s->%s/%lx-%lx", topoLinkTypeStr[link->type], topoNodeTypeStr[remNode->type],
                                               EP_TOPO_ID_SYSTEM_ID(remNode->id), EP_TOPO_ID_LOCAL_ID(remNode->id));
        offset = strlen(line);
      }
      INFO(EP_GRAPH, "%s (%lx-%lx %d/%f/%s)", line, EP_TOPO_ID_SYSTEM_ID(node->id), EP_TOPO_ID_LOCAL_ID(node->id),
                                node->paths[t][n].count, node->paths[t][n].bw, topoPathTypeStr[node->paths[t][n].type]);
    }
  }
}

void epTopoSystem::printPaths() {
  for (int i = 0; i < nodes_[CPU].count; i ++) {
    printNodePaths(nodes_[CPU].nodes + i);
  }
  for (int i = 0; i < nodes_[GCU].count; i ++) {
    printNodePaths(nodes_[GCU].nodes+i);
  }
  for (int i = 0; i < nodes_[NET].count; i ++) {
    printNodePaths(nodes_[NET].nodes+i);
  }
}
#endif

void epTopoSystem::printGcuNetPathMatrix() {
  std::ostringstream oss;
  oss << "\t      ";
  int nNets = netCount();
  for (int i = 0; i < nNets; ++ i) {
    oss << " NET(" << std::hex << EP_TOPO_ID_SYSTEM_ID(nodes_[NET].nodes[i].id) << "-"
                                                << EP_TOPO_ID_LOCAL_ID(nodes_[NET].nodes[i].id) << std::dec << ") ";
  }
  INFO(EP_GRAPH, oss.str().c_str());
  // Print paths from GCU.
  if (nNets > 0) {
    int ngcus = gcuCount();
    for (int i = 0; i < ngcus; ++ i) {
      oss.str("");
      oss.clear();
      epTopoNode* node = nodes_[GCU].nodes + i;
      oss << "\tGCU(" << node->gcu.rank <<")  ";
      for (int j = 0; j < nNets; ++ j) {
        oss << topoPathTypeStr[node->paths[NET][j].type] <<"    ";
      }
      INFO(EP_GRAPH, oss.str().c_str());
    }
  }
}

void epTopoSystem::printGcuSwitchPathMatrix() {
  constexpr int widthPerItem = 6;
  std::ostringstream oss;
  oss << "\t      ";
  int ngcus = gcuCount();
  for (int i = 0; i < ngcus; ++i) {
    oss << " "<< std::setw(widthPerItem) << ("GCU" + std::to_string(nodes_[GCU].nodes[i].gcu.rank));
  }
  int nLareSwitches = getCount(LARESWT);
  for (int i = 0; i < nLareSwitches; ++ i) {
    oss << " "<< std::setw(widthPerItem) << ("SWT" + std::to_string(nodes_[LARESWT].nodes[i].lareswitch.dev));
  }
  INFO(EP_GRAPH, oss.str().c_str());

  // Print paths from GCU.
  for (int i = 0; i < ngcus; ++ i) {
    oss.str("");
    oss.clear();
    epTopoNode* node = nodes_[GCU].nodes + i;
    oss << "\t" << std::setw(widthPerItem) << ("GCU" + std::to_string(node->gcu.rank));
    for (int j = 0; j < ngcus; ++ j) {
      std::string member;
      if (node->paths[GCU][j].type == PATH_LARE) {
        member = std::string(topoPathTypeStr[PATH_LARE]) + std::to_string(static_cast<int>(node->paths[GCU][j].bw / 23.0));
      } else {
        member = topoPathTypeStr[node->paths[GCU][j].type];
      }
      oss << " " << std::setw(widthPerItem) << member;
    }
    for (int j = 0; j < nLareSwitches; ++ j) {
      std::string member;
      if (node->paths[LARESWT][j].type == PATH_LARE) {
        member = std::string(topoPathTypeStr[PATH_LARE]) +
                   std::to_string(static_cast<int>(node->paths[LARESWT][j].bw / 23.0));
      } else {
        member = topoPathTypeStr[node->paths[LARESWT][j].type];
      }
      oss << " " << std::setw(widthPerItem) << member;
    }
    INFO(EP_GRAPH, oss.str().c_str());
  }

  for (int i = 0; i < nLareSwitches; ++ i) {
    oss.str("");
    oss.clear();
    epTopoNode* node = nodes_[LARESWT].nodes + i;
    oss << "\t" << std::setw(widthPerItem) << ("SWT" + std::to_string(node->lareswitch.dev));
    for (int j = 0; j < ngcus; ++ j) {
      std::string member;
      if (node->paths[GCU][j].type == PATH_LARE) {
        member = std::string(topoPathTypeStr[PATH_LARE]) + std::to_string(static_cast<int>(node->paths[GCU][j].bw / 23.0));
      } else {
        member = topoPathTypeStr[node->paths[GCU][j].type];
      }
      oss << " " << std::setw(widthPerItem) << member;
    }
    for (int j = 0; j < nLareSwitches; ++ j) {
      if (i == j)
        oss << " " << std::setw(widthPerItem) << std::string(topoPathTypeStr[PATH_LOC]);
      else
        oss << " " << std::setw(widthPerItem) << std::string("DIS");
    }
    INFO(EP_GRAPH, oss.str().c_str());
  }
}

void epTopoSystem::printPathMatrix() {
  printGcuSwitchPathMatrix();
  printGcuNetPathMatrix();
}

epResult_t epTopoSystem::checkMNLARE(efmlGcuFabricInfoV_t* fabricInfo1, efmlGcuFabricInfoV_t* fabricInfo2, int* ret) {
  *ret = 0;

  // A zero UUID means we don't have MNLARE fabric info
  //if ((((long *)&fabricInfo2->clusterUuid)[0]|((long *)fabricInfo2->clusterUuid)[1]) == 0) return epSuccess;
  if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, EFML_GCU_FABRIC_UUID_LEN) == 0) &&
      (fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {
    TRACE(EP_GRAPH, "MNLARE matching peer UUID %lx.%lx cliqueId 0x%x",
                  ((long *)fabricInfo2->clusterUuid)[0], ((long *)fabricInfo2->clusterUuid)[1], fabricInfo2->cliqueId);
    *ret = 1;
  }
  return epSuccess;
}

