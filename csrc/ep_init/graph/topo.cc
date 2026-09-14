#include <math.h>
#include <ctype.h>

#include "topo.h"
#include "checks.h"
#include "net.h"
#include "cpuset.h"
#include "alloc.h"
#include "debug.h"
#include "transport.h"
#include <iostream>
#include <sstream>

const char* topoNodeTypeStr[] = { "GCU", "PCI", "LARESWT", "CPU", "NIC", "NET"};
const char* topoLinkTypeStr[] = { "LOC", "LARE", "PCI", "PXB", "PHB", "SYS", "NET"};
const char* topoPathTypeStr[] = { "LOC", "LARE", "PIX", "PXB", "PHB", "SYS", "NET", "DIS"};

static_assert((sizeof(topoNodeTypeStr)/sizeof(topoNodeTypeStr[0])) == EP_TOPO_NODE_TYPES,
                                                                              "topoNodeTypeStr length is not enough");
static_assert((sizeof(topoLinkTypeStr)/sizeof(topoLinkTypeStr[0])) == MAX_LINK_TYPE,
                                                                              "topoLinkTypeStr length is not enough");
static_assert((sizeof(topoPathTypeStr)/sizeof(topoPathTypeStr[0])) == MAX_PATH_TYPE,
                                                                              "topoPathTypeStr length is not enough");

/* SuperNode connections topology (gcu2switch): map each efml LARE port to a Lare switch index.
 * See eccl PC-1661 / commit a8d62e0b. */
static int defaultLareSwtConnectedPort[MAX_LARES_PER_GCU] = {3, 3, 2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 0, 0, 1, 1};
static int getPortConnectedLareSwtId(epTopoNode* gcuNode, const int efmlPort) {
  (void)gcuNode;
  return defaultLareSwtConnectedPort[efmlPort];
}

// set to 1 means not use rev bw in lare
#define USE_RECV_BW 0
#define MAX_DIRECT_CONNECTED_PORTID 13

epResult_t epTopoSystem::getNode(epTopoNodeType type, int idx, epTopoNode** retNode) {
  if (idx < nodes_[type].count) {
    *retNode = nodes_[type].nodes + idx;
    return epSuccess;
  }
  WARN("%s node with index %d (max %d) not found", topoNodeTypeStr[type], idx, nodes_[type].count - 1);
  return epInternalError;
}

epTopoNode* epTopoSystem::getNode(epTopoNodeType type, int64_t id) {
  for (int i = 0; i < nodes_[type].count; i ++) {
    if (nodes_[type].nodes[i].id == id) {
      return nodes_[type].nodes+i;
    }
  }
  return nullptr;
}

epResult_t epTopoSystem::createNode(epTopoNodeType type, uint64_t id, epTopoNode** retNode) {
  if (nodes_[type].count == EP_TOPO_MAX_NODES) {
    WARN("Topo : Try to create too many nodes of type %d", type);
    return epInternalError;
  }
  epTopoNode* n = nodes_[type].nodes + (nodes_[type].count ++);
  n->type = type;
  n->id = id;
  if (type == GCU) {
    // Create link to itself (used in some corner cases)
    // TODO : what's the corner cases
    n->nlinks=1;
    n->links[0].type = LINK_LOC;
    n->links[0].remNode = n;
    n->links[0].bw = LOC_BW;
    n->gcu.dev = -1;
    n->gcu.rank = -1;
    n->gcu.efmlArch = 0;
  } else if (type == CPU) {
    n->cpu.arch = -1;
    n->cpu.vendor = -1;
    n->cpu.model = -1;
  } else if (type == NET) {
    n->net.asic = 0ULL;
    n->net.port = -1;
    n->net.bw = 0.0;
    n->net.latency = 0.0;
  } else { // type == PCI
    n->pci.device = 0ULL;
  }
  *retNode = n;
  return epSuccess;
}

epResult_t epTopoSystem::connectNodes(epTopoNode* node, epTopoNode* remNode, int type, float bw) {
  epTopoLink* link;
  // Link node->remNode exist in node's link list.
  for (link = node->links; link->remNode && node->nlinks < EP_TOPO_MAX_LINKS; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  // Link node->remoNode not exist in node's link list.
  if (node->nlinks >= EP_TOPO_MAX_LINKS) {
    WARN("Topo : Too many links for node %s/%ld", topoNodeTypeStr[node->type], node->id);
    return epInternalError;
  }

  if (link->remNode == nullptr) node->nlinks++;

  // Aggregate links into higher bw for lare link.
  link->type = type;
  link->remNode = remNode;
  link->bw += bw;

  // Sort links in bandwidth descending order.
  epTopoLink linkSave;
  memcpy(&linkSave, link, sizeof(struct epTopoLink));
  while (link != node->links) {
    if ((link-1)->bw >= linkSave.bw) break;
    memcpy(link, link-1, sizeof(struct epTopoLink));
    link--;
  }
  memcpy(link, &linkSave, sizeof(struct epTopoLink));
  return epSuccess;
}

struct kvDict {
  const char* str;
  int value;
};

static int find(kvDict* dict, const char* str) {
  kvDict* d = dict;
  while (d->str) {
    // only compare string length in dict
    if (strncmp(str, d->str, strlen(d->str)) == 0) {
      return d->value;
    }
    d++;
  }
  INFO(EP_GRAPH, "KV Convert to int : could not find value of '%s' in dictionary, falling back to %d", str, d->value);
  return d->value;
}

kvDict kvDictCpuArch[] = {
    { "x86_64", EP_TOPO_CPU_ARCH_X86 },
    { "arm64", EP_TOPO_CPU_ARCH_ARM },
    /* Default fallback */
    { nullptr, 0 }
};

kvDict kvDictCpuVendor[] = {
    { "GenuineIntel", EP_TOPO_CPU_VENDOR_INTEL },
    { "AuthenticAMD", EP_TOPO_CPU_VENDOR_AMD },
    /* Default fallback */
    { nullptr, 0 }
};

epResult_t epTopoSystem::getSystemId(epXmlNode* xmlCpu, int* systemIdPtr) {
  const char* hostHashStr;
  EP_CHECK(xmlNodeGetAttr(xmlCpu, "host_hash", &hostHashStr));
  uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, nullptr, 10) : 0;
  int systemId;
  for (systemId=0; systemId<nHosts_; systemId++) if (hostHashes_[systemId] == hostHash) break;
  if (systemId == nHosts_) hostHashes_[nHosts_++] = hostHash;
  *systemIdPtr = systemId;
  return epSuccess;
}

epResult_t epTopoSystem::addCpu(epXmlNode* xmlCpu) {
  epTopoNode* cpu = nullptr;
  epXmlAttribute* attrs = xmlCpu->attributes();
  epXmlAttributeElement* numaid = attrs->find("numaid");
  int systemId;
  EP_CHECK(getSystemId(xmlCpu, &systemId));
  if(numaid) {
    EP_CHECK(createNode(CPU, EP_TOPO_ID(systemId, numaid->asInt()), &cpu));
  } else {
    WARN("Topo : cpu node have't numaid attribute.");
    return epInternalError;
  }

  if(auto affinity = attrs->find("affinity")) {
    EP_CHECK(epStrToCpuset(affinity->value(), &cpu->cpu.affinity));
  }
  // TODO : ARCH/VENDOR/MODEL add here

  for (size_t c = 0; c < xmlCpu->childSize(); c ++) {
    epXmlNode* xmlNode = xmlCpu->childAt(c);
    if (strcmp(xmlNode->name(), "pci") == 0) EP_CHECK(addPci(xmlNode, cpu, systemId, numaid->asInt()));
    if (strcmp(xmlNode->name(), "nic") == 0) {
      int64_t localNicId = EP_TOPO_LOCAL_NIC_ID(numaid->asInt(), 0);
      int64_t id = EP_TOPO_ID(systemId, localNicId);
      epTopoNode* nic = getNode(NIC, id);
      if (nic == nullptr) {
        EP_CHECK(createNode(NIC, id, &nic));
        EP_CHECK(connectNodes(cpu, nic, LINK_PCI, LOC_BW));
        EP_CHECK(connectNodes(nic, cpu, LINK_PCI, LOC_BW));
      }
      EP_CHECK(addNic(xmlNode, nic, systemId));
    }
  }
  return epSuccess;
}

epResult_t epTopoSystem::addNet(epXmlNode* xmlNet, epTopoNode* nic, int systemId) {
  int dev;
  EP_CHECK(xmlNodeGetAttr(xmlNet, "dev", &dev));

  epTopoNode* net;
  EP_CHECK(createNode(NET, EP_TOPO_ID(systemId, dev), &net));
  net->net.dev = dev;

  const char* str;
  EP_CHECK(xmlNodeGetAttr(xmlNet, "guid", &str));
  sscanf(str, "%ld", &net->net.asic);

  uint64_t mbps;
  xmlNodeGetAttrDefault(xmlNet, "speed", &mbps, 0);
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0; // GB/s
  xmlNodeGetAttrDefault(xmlNet, "latency", &net->net.latency, 0);
  xmlNodeGetAttrDefault(xmlNet, "port", &net->net.port, 0);
  xmlNodeGetAttrDefault(xmlNet, "gdr", &net->net.gdrSupport, 0);
  xmlNodeGetAttrDefault(xmlNet, "maxconn", &net->net.maxChannels, MAXCHANNELS);

  EP_CHECK(connectNodes(nic, net, LINK_NET, net->net.bw));
  EP_CHECK(connectNodes(net, nic, LINK_NET, net->net.bw));
  return epSuccess;
}

epResult_t epTopoSystem::addNic(epXmlNode* xmlNic, epTopoNode* nic, int systemId) {
  for (size_t c = 0; c < xmlNic->childSize(); c ++) {
    epXmlNode* xmlNet = xmlNic->childAt(c);
    if (strcmp(xmlNet->name(), "net") != 0) continue;
    EP_CHECK(addNet(xmlNet, nic, systemId));
  }
  return epSuccess;
}

epResult_t epTopoSystem::addGcu(epXmlNode* xmlGcuNode, epTopoNode* gcu) {
  int efmlArch;
  EP_CHECK(xmlNodeGetAttr(xmlGcuNode, "arch", &efmlArch));
  gcu->gcu.efmlArch = static_cast<efmlDeviceArchitecture_t>(efmlArch);
  EP_CHECK(xmlNodeGetAttr(xmlGcuNode, "rank", &gcu->gcu.rank));
  EP_CHECK(xmlNodeGetAttr(xmlGcuNode, "dev", &gcu->gcu.dev));
  EP_CHECK(xmlNodeGetAttr(xmlGcuNode, "gdr", &gcu->gcu.gdrSupport));
  // Do not go any further, lares will be added in a second pass
  return epSuccess;
}

kvDict kvDictPciGen[] = {
     /* Kernel 5.6 and earlier */
    { "2.5 GT/s", 15 },
    { "5 GT/s", 30 },
    { "8 GT/s", 60 },   // Gen 3
    { "16 GT/s", 120 }, // Gen 4
    { "32 GT/s", 240 }, // Gen 5
     /* Kernel later */
    { "2.5 GT/s PCIe", 15 },
    { "5.0 GT/s PCIe", 30 },
    { "8.0 GT/s PCIe", 60 },
    { "16.0 GT/s PCIe", 120 },
    { "32.0 GT/s PCIe", 240 },
    { "64.0 GT/s PCIe", 480 },
    /* Default fallback */
    { nullptr, 60  }
}; // x100 Mbps per lane

epResult_t epTopoSystem::addPci(epXmlNode* xmlPci, epTopoNode* upNode, int systemId, int numaId) {
  const char* busIdStr;
  EP_CHECK(xmlNodeGetAttr(xmlPci, "busid", &busIdStr));
  int64_t busId;
  EP_CHECK(busIdToInt64(busIdStr, &busId));

  epTopoNode* node = nullptr;
  epXmlNode* xmlGcu = xmlPci->findChild("gcu");
  epXmlNode* xmlNic = xmlPci->findChild("nic");
  if (xmlGcu && xmlNic) {
    WARN("PCI %s connect Gcu as well as Nic node", busIdStr);
    return epInternalError;
  }

  // We know that a pci node can connect to another pci, or connect to a gcu, or connect to
  // a nci. For these three situation, process differently.
  if (xmlGcu) {
    // pci connect to gcu
    int rank;
    EP_CHECK(xmlNodeGetAttr(xmlGcu, "rank", &rank));
    EP_CHECK(createNode(GCU, EP_TOPO_ID(systemId, busId), &node));
    EP_CHECK(addGcu(xmlGcu, node));
  } else if (xmlNic) {
    // pci connect to nic
    // Ignore sub device ID and merge multi-port NICs into one PCI device.
    int64_t localNicId = EP_TOPO_LOCAL_NIC_ID(numaId, busId);
    int64_t id = EP_TOPO_ID(systemId, localNicId);
    epTopoNode* nic = getNode(NIC, id);
    if (nic == nullptr) {
      EP_CHECK(createNode(NIC, id, &nic));
      node = nic; // Connect it to upNode later
    }
    EP_CHECK(addNic(xmlNic, nic, systemId));
  } else {
    // pci connect to pci
    EP_CHECK(createNode(PCI, EP_TOPO_ID(systemId, busId), &node));

    // Topology pci node's device field is assembled by vendor + device +
    // subsystem_vendor + subsystem_device.
    uint64_t value;
    xmlNodeGetAttrDefault(xmlPci, "vendor", &value, 0);
    node->pci.device += value << 48;
    xmlNodeGetAttrDefault(xmlPci, "device", &value, 0);
    node->pci.device += value << 32;
    xmlNodeGetAttrDefault(xmlPci, "subsystem_vendor", &value, 0);
    node->pci.device += value << 16;
    xmlNodeGetAttrDefault(xmlPci, "subsystem_device", &value, 0);
    node->pci.device += value << 0;

    for (size_t c = 0; c < xmlPci->childSize(); c ++)
      EP_CHECK(addPci(xmlPci->childAt(c), node, systemId, numaId));
  }

  if (node) {
    int width = 0;
    const char* speedStr = nullptr;
    EP_CHECK(xmlNodeGetAttr(xmlPci, "link_width", &width));
    EP_CHECK(xmlNodeGetAttr(xmlPci, "link_speed", &speedStr));

    // Manage cases where speed was not indicated in /sys .
    if (width == 0) width = 16;
    // Values in 100Mbps, per lane (we want GB/s in the end).
    int speed = find(kvDictPciGen, speedStr);

    EP_CHECK(connectNodes(node, upNode, LINK_PCI, width * speed / 80.0));
    EP_CHECK(connectNodes(upNode, node, LINK_PCI, width * speed / 80.0));
  }
  return epSuccess;
}

float epTopoSystem::getLareBw(epTopoNode* gcu) {
  (void)gcu;
  return GCU_400_LARE_BW;
}

kvDict kvDictPciClass[] = {
  { "0x060400", PCI },
  { "0x068000", LARESWT}, //lare switch's class.
  { "0x068001", CPU },
  { "0x12", GCU },
  { "0x02", NIC },
  /* Default fallback value */
  { nullptr, PCI }
};

epResult_t epTopoSystem::calcEfmlLarePorts(int systemId, epTopoNode* srcGcuNode, epTopoNode* dstNode, const char *portConnStrPtr) {
  // Find link.
  int l = 0;
  while (l < srcGcuNode->nlinks && srcGcuNode->links[l].remNode != dstNode) ++ l;
  epTopoLink* link = &srcGcuNode->links[l];
  if (link->type != LINK_LARE) {
    if (dstNode->type == GCU) {
      WARN("TOPO : something wrong for link type GCU/%d -> GCU/%d.", srcGcuNode->gcu.rank, dstNode->gcu.rank);
    } else {
      WARN("TOPO : something wrong for link type GCU/%d -> LARESWT/%d.", srcGcuNode->gcu.rank, dstNode->lareswitch.dev);
    }
    return epInternalError;
  }

  // Add extra information to link lare, including conn_port ...
  std::istringstream portConnIss(portConnStrPtr);
  std::vector<int> efmlPortsConnectedSwitch;
  int efmlPort = -1;
  if (dstNode->type == GCU) {
    int peerGcuId;
    EP_CHECK(topoIdToIndex(GCU, dstNode->id, &peerGcuId));
    while (portConnIss >> efmlPort) {
      link->localEfmlPortId[peerGcuId][link->portCount[peerGcuId]++] = efmlPort;
    }
    if (srcGcuNode->gcu.connectType == SWITCH_CONNECTED) srcGcuNode->gcu.connectType = DIRECT_CONNECTED;
  } else {
    int intraGcuPortCount = 0;
    int interGcuPortCount = 0;
    while (portConnIss >> efmlPort) {
      efmlPortsConnectedSwitch.push_back(efmlPort);
      if (efmlPort > MAX_DIRECT_CONNECTED_PORTID)
        interGcuPortCount++;
      else
        intraGcuPortCount++;
    }
    if (srcGcuNode->gcu.connectType == DIRECT_CONNECTED && intraGcuPortCount > 0 && interGcuPortCount > 0)
      srcGcuNode->gcu.connectType = SWITCH_CONNECTED;
    for (int peerGcuId=0; peerGcuId < nodes_[GCU].count; peerGcuId++) {
      epTopoNode* peerNode = nullptr;
      EP_CHECK(getNode(GCU, peerGcuId, &peerNode));
      if (srcGcuNode->gcu.rank == peerNode->gcu.rank) continue;
      //for OGX boards, only lare port 14/15 may connect with switch for inter-OS communication.
      // port 0-13 for intra-OS communication. so for port 14/15 we will ignore it for intra-OS communication.
      for (size_t p = 0; p < efmlPortsConnectedSwitch.size(); p++) {
        int efmlPortPeer = efmlPortsConnectedSwitch[p];
        if (srcGcuNode->gcu.connectType != DIRECT_CONNECTED || systemId != EP_TOPO_ID_SYSTEM_ID(peerNode->id)
            || efmlPortPeer <= MAX_DIRECT_CONNECTED_PORTID)
          link->localEfmlPortId[peerGcuId][link->portCount[peerGcuId]++] = efmlPortsConnectedSwitch[p];
        if (SWITCH_CONNECTED == srcGcuNode->gcu.connectType && efmlPortPeer >= 0
            && efmlPortPeer < (int)MAX_LARES_PER_GCU) {
          int lareSwtId = getPortConnectedLareSwtId(srcGcuNode, efmlPortPeer);
          TRACE(EP_GRAPH, "link->efmlPortToLareSwitch[gcu%d][lareSwtId%d][%d]=efp%d\n", peerGcuId, lareSwtId,
                link->portSizeToLareSwitch[peerGcuId][lareSwtId], efmlPortPeer);
          link->efmlPortToLareSwitch[peerGcuId][lareSwtId][link->portSizeToLareSwitch[peerGcuId][lareSwtId]++] =
              efmlPortPeer;
        }
      }
    }
  }

  if (dstNode->type != GCU) {
    l = 0;
    while (l < dstNode->nlinks && dstNode->links[l].remNode != srcGcuNode)
      ++ l;
    link = &dstNode->links[l];
    if (link->type != LINK_LARE) {
      WARN("TOPO : something wrong for link type LARESWT/%d -> GCU/%d.", dstNode->lareswitch.dev, srcGcuNode->gcu.rank);
      return epInternalError;
    }
    int srcGcuId;
    EP_CHECK(topoIdToIndex(GCU, srcGcuNode->id, &srcGcuId));
    link->portCount[srcGcuId] = (int)efmlPortsConnectedSwitch.size();
  }
  return epSuccess;
}

epResult_t epTopoSystem::addLares(epXmlNode* node, const char* parentBusId, int systemId) {
  // We need first find lare xml node. If we find it,
  // connect nodes with lare link.
  if (strcmp(node->name(), "lare") == 0) {
    epTopoNode* srcGcuNode = nullptr;
    EP_CHECK(getGcuNode(systemId, parentBusId, srcGcuNode));
    if (srcGcuNode == nullptr) {
      WARN("Add Lare error : could not find GCU %x-%s", systemId, parentBusId);
      return epInternalError;
    }

    const char* targetClass = nullptr;
    EP_CHECK(xmlNodeGetAttr(node, "tclass", &targetClass));
    int targetType = find(kvDictPciClass, targetClass);
    epTopoNode* dstNode = nullptr;
    if (targetType == GCU) {      
      // lare connection to another GCU
      const char* target;
      EP_CHECK(xmlNodeGetAttr(node, "target", &target));
      EP_CHECK(getGcuNode(systemId, target, dstNode));
    } else {
      if (nodes_[LARESWT].count == 0) {
        EP_CHECK(createNode(LARESWT, 0, &dstNode));
      } else {
        dstNode = nodes_[LARESWT].nodes;
      }
    }

    if (dstNode) {
      int count;
      EP_CHECK(xmlNodeGetAttr(node, "count", &count));
      EP_CHECK(connectNodes(srcGcuNode, dstNode, LINK_LARE, count * getLareBw(srcGcuNode)));
      if (dstNode->type != GCU) {
        EP_CHECK(connectNodes(dstNode, srcGcuNode, LINK_LARE, count*getLareBw(srcGcuNode)));
      }

      const char *portConnStrPtr;
      EP_CHECK(xmlNodeGetAttr(node, "port_conn", &portConnStrPtr));
      EP_CHECK(calcEfmlLarePorts(systemId, srcGcuNode, dstNode, portConnStrPtr));
    }
  } else {
    // We have not find lare xml node, triverse the xml tree
    // to find lare node.

    // We know that not all xml node have busid attribute,
    // including "system", "gcu", "nic", "net" node.
    // For these situation, use last node's busid as gcuBusId.
    // Cause lare xml always connect to gcu node and than connect
    // to pci node, which is gcuBusId we use here.
    int systemId_l = systemId;
    if (strcmp(node->name(), "cpu") == 0) {
      EP_CHECK(getSystemId(node, &systemId_l));
    }
    const char* busId;
    xmlNodeGetAttrDefault(node, "busid", &busId, nullptr);
    for (size_t c = 0; c < node->childSize(); c ++) {
      EP_CHECK(addLares(node->childAt(c), busId ? busId : parentBusId, systemId_l));
    }
  }
  return epSuccess;
}

float epTopoSystem::getInterCpuBw(epTopoNode* /*cpu*/) {
  float bw = QPI_BW;
  // TODO : support more CPU architecture
  return bw;
}

/*
 * Just connect all cpus directly. Notice that connection
 * between cpus may not support, it will be determined later.
 */
epResult_t epTopoSystem::addCpusLink() {
  for (int n=0; n<nodes_[CPU].count; n++) {
    struct epTopoNode* cpu1 = nodes_[CPU].nodes+n;
    for (int p=0; p< nodes_[CPU].count; p++) {
      struct epTopoNode* cpu2 = nodes_[CPU].nodes+p;
      if (n == p || (EP_TOPO_ID_SYSTEM_ID(cpu1->id) != EP_TOPO_ID_SYSTEM_ID(cpu2->id))) continue;
      EP_CHECK(connectNodes(
          nodes_[CPU].nodes + n, nodes_[CPU].nodes + p,
          LINK_SYS, getInterCpuBw(nodes_[CPU].nodes + n)));
    }
  }
  return epSuccess;
}

static void sortRec(epTopoNode* node, epTopoNode* upNode) {
  // Shift all links to have upLink as last link
  if (upNode) {
    // find upLink
    int l=0;
    while (node->links[l].remNode != upNode) l++;
    epTopoLink upLink;
    memcpy(&upLink, node->links+l, sizeof(epTopoLink));
    // shift upLink to end
    while (node->links[l+1].remNode) {
      memcpy(node->links+l, node->links+l+1, sizeof(epTopoLink));
      l++;
    }
    memcpy(node->links+l, &upLink, sizeof(epTopoLink));
  }

  // Recursively sort the PCI tree
  for (int l = 0; l < node->nlinks; l ++) {
    epTopoLink* link = node->links+l;
    if (link->type == LINK_PCI && link->remNode != upNode)
      sortRec(link->remNode, node);
  }
}

/*
 * We want the links in a topology node to be organized in
 * bandwidth descend order. The reason why we do it is that
 * traversing a node will always begin from the link with
 * biggest bandwidth.
 * Link in bandwidth descend order is :
 * LARE : only exits at GCU-GCU link(already the case)
 * PCI down
 * PCI up
 * SYS : only exist at CPU-CPU link (already the case)
 */
void epTopoSystem::sortSystem() {
  for (int n = 0; n < nodes_[CPU].count; n++)
    sortRec(nodes_[CPU].nodes+n, nullptr);
}

epResult_t epTopoSystem::getSystemFromXml(epXml* xml, const uint64_t localHostHash) {
  epXmlNode* system = xml->findNode("system");
  if (system == nullptr) return epInternalError;

  // Transfer xml to topology system.
  // firstly, adding all xml node to topology system
  // begin from cpu xml node
  for (size_t c = 0; c < system->childSize(); ++ c) {
    epXmlNode* cpu = system->childAt(c);
    if (strcmp(cpu->name(), "cpu") == 0)
        EP_CHECK(addCpu(cpu));
  }
  for (int systemId=0; systemId<nHosts_; systemId++)
    if (hostHashes_[systemId] == localHostHash)
      mySystemId_ = systemId;

  // Secondly, adding connection between GCU and
  // adding connection between CPUs to topology system.
  EP_CHECK(addLares(system, nullptr, 0));
  EP_CHECK(addCpusLink());
  // Lastly, sorting the topology system for computing
  // paths process.
  sortSystem();
  return epSuccess;
}

void epTopoSystem::printLinkRec(epTopoNode* node,
    epTopoNode* prevNode, char* line, int offset) {
  if (node->type == GCU) {
    // GCU print format : GCU/id (rank)
    sprintf(line+offset, "%s/%lx-%lx (%d)",
        topoNodeTypeStr[node->type], EP_TOPO_ID_SYSTEM_ID(node->id), EP_TOPO_ID_LOCAL_ID(node->id), node->gcu.rank);
  } else if (node->type == CPU) {
    // CPU print format : CPU/id (arch/vendor/model)
    sprintf(line+offset, "%s/%lx-%lx (%d/%d/%d)",
        topoNodeTypeStr[node->type], EP_TOPO_ID_SYSTEM_ID(node->id), EP_TOPO_ID_LOCAL_ID(node->id), node->cpu.arch,
        node->cpu.vendor, node->cpu.model);
  } else if (node->type == PCI) {
    // PCI print format : PCI/id (device)
    sprintf(line+offset, "%s/%lx-%lx (%lx)",
        topoNodeTypeStr[node->type], EP_TOPO_ID_SYSTEM_ID(node->id), EP_TOPO_ID_LOCAL_ID(node->id), node->pci.device);
  } else {
    sprintf(line+offset, "%s/%lx-%lx",
        topoNodeTypeStr[node->type], EP_TOPO_ID_SYSTEM_ID(node->id), EP_TOPO_ID_LOCAL_ID(node->id));
  }
  INFO(EP_GRAPH, "%s", line);
  for (int i=0; i<offset; i++) line[i] = ' ';

  for (int l=0; l<node->nlinks; l++) {
    struct epTopoLink* link = node->links+l;
    if (link->type == LINK_LOC) continue;
    if (link->type != LINK_PCI || link->remNode != prevNode) {
      sprintf(line+offset, "+ LINK_%s[%2.1f] - ",
          topoLinkTypeStr[link->type], link->bw);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        printLinkRec(link->remNode, node, line, nextOffset);
      } else {
        if (link->remNode->type == NET) {
          // NET print format : NET/id (asic/port/bw)
          sprintf(line+nextOffset, "%s/%lx-%lx (%lx/%d/%2.1f)",
              topoNodeTypeStr[link->remNode->type], EP_TOPO_ID_SYSTEM_ID(link->remNode->id), EP_TOPO_ID_LOCAL_ID(link->remNode->id),
              link->remNode->net.asic, link->remNode->net.port,
              link->remNode->net.bw);
        } else {
          sprintf(line+nextOffset, "%s/%lx-%lx",
          topoNodeTypeStr[link->remNode->type], EP_TOPO_ID_SYSTEM_ID(link->remNode->id), EP_TOPO_ID_LOCAL_ID(link->remNode->id));
        }
        INFO(EP_GRAPH, "%s", line);
      }
    }
  }
}

void epTopoSystem::printLinks() {
  char line[1024];
  for (int n = 0; n < nodes_[CPU].count; n ++)
    printLinkRec(nodes_[CPU].nodes+n, nullptr, line, 0);
}

// maxBw from GCU to GCU or from GCU to NIC/NET
float epTopoSystem::getMaxBw(int g, epTopoNodeType type) {
  epTopoNode* gcu = nodes_[GCU].nodes + g;
  float maxBw = 0.0;
  for (int i=0; i < nodes_[type].count; i++) {
    epTopoLinkList* path = gcu->paths[type] + i;
    float bw = path->bw;
    if (path->count == 0) continue;
    maxBw = std::max(maxBw, bw);
  }
  return maxBw;
}

// totalBw is total bandwidth of all Lare or pciBw
// if have Lare connection, max bandwidth is 6(port) * bwLarePort
float epTopoSystem::getTotalBw(int g) {
  epTopoNode* gcu = nodes_[GCU].nodes + g;
  float lareBw = 0.0, pciBw = 0.0;
  for (int l = 0; l < gcu->nlinks; l ++) {
    epTopoLink* link = gcu->links + l;
    if (link->type == LINK_LARE) lareBw += link->bw;
    if (link->type == LINK_PCI) pciBw = link->bw;
  }
  return std::max(pciBw, lareBw);
}

epResult_t epTopoSystem::graphSearchInit() {
  int nets = netCount();
  int gcus = gcuCount();
  if (nets == 0 && gcus == 1) {
    maxBw_ = LOC_BW;
    return epSuccess;
  }
  for (int g = 0; g < gcus; g ++) {
    // All GCU->GCU or GCU->NET path's max bandwidth
    maxBw_ = std::max(maxBw_, getMaxBw(g, (nets > 0) ? NET : GCU));
    // All GCU's max total totalbandwith
    totalBw_ = std::max(totalBw_, getTotalBw(g));
  }
  return epSuccess;
}

epResult_t epTopoSystem::rankToIndex(int rank, int* retIdx, epTopoNodeType type) {
  *retIdx = -1;
  int gcus = gcuCount();
  for (int i = 0; i < gcus; i ++) {
    if (nodes_[type].nodes[i].gcu.rank == rank) {
      *retIdx = i;
      return epSuccess;
    }
  }

  return epInternalError;
}

epResult_t epTopoSystem::topoIdToNetDev(int64_t id, int* netDev) {
  *netDev = -1;
  for (int i=0; i<nodes_[NET].count; i++) {
    if (nodes_[NET].nodes[i].id == id) {
      *netDev = nodes_[NET].nodes[i].net.dev;
      return epSuccess;
    }
  }
  WARN("Could not find NET with id %lx", id);
  return epInternalError;
}

bool epTopoSystem::gcuInChannel(int gcuIdx, int channel) {
  const uint64_t flag = 1ULL<<(channel);
  return nodes_[GCU].nodes[gcuIdx].used & flag;
}

epResult_t epTopoSystem::addChannelToNode(epTopoNodeType type, int index, int channel) {
  epTopoNode* node;
  if (getNode(type, index, &node) != epSuccess) {
    WARN("Can't add gcu node with index(%d) to channel(%d)", index, channel);
    return epInternalError;
  }
  if (type == GCU) {
    const uint64_t flag = 1ULL << channel;
    uint64_t used = node->used;
    if (used & flag) {
      WARN("Gcu node with rank %d reuse in one channel", node->gcu.rank);
      return epInternalError;
    }
    node->used ^= flag;
  } else if (type == NET) {
    if (node->net.maxChannels == 0) {
      WARN("no more net channel for graph channel");
      return epInternalError;
    }
    node->net.maxChannels--;
  }
  return epSuccess;
}

epResult_t epTopoSystem::removeChannelFromNode(epTopoNodeType type, int n, int channel) {
  epTopoNode* node;
  if (getNode(type, n, &node) != epSuccess) {
    WARN("Can't remove net node with index %d to channel %d", n, channel);
    return epInternalError;
  }
  if (type == GCU) {
    const uint64_t flag = 1ULL << channel;
    uint64_t used = node->used;
    if (!(used & flag)) {
      WARN("remove gcu node with rank %d which is non't used in one channel", node->gcu.rank);
      return epInternalError;
    }
    node->used ^= flag;
  } else if (type == NET)
    node->net.maxChannels ++;

  return epSuccess;
}

#if USE_RECV_BW
epResult_t epTopoSystem::findRevLink(
    epTopoNode* node1, epTopoNode* node2,
    epTopoLink** revLink) {
  for (int l = 0; l < node2->nlinks; ++ l) {
    struct epTopoLink* link = node2->links+l;
    if (link->remNode == node1) {
      *revLink = link;
      return epSuccess;
    }
  }
  WARN("Could not find rev link for %d/%ld -> %d/%ld",
      node1->type, node1->id, node2->type, node2->id);
  return epInternalError;
}
#endif

// This is unfortunately needed since manipulating floats
// often results in rounding errors.
#define SUB_ROUND(a, b) (a = roundf((a-b)*1000)/1000)
epResult_t epTopoSystem::followPath(
    epTopoLinkList* path, epTopoNode* /*start*/, int maxSteps,
    float bw, int* retFollowedSteps) {
  float pciBw = bw;
  /*
  for (int step=0; step<path->count; step++) {
    struct epTopoNode* node = path->list[step]->remNode;
    if (node->type == CPU) {
      // Account for P2P inefficiency through Intel CPU RC
      if (path->type == PATH_PHB && start->type == GCU &&
          node->cpu.arch == EP_TOPO_CPU_ARCH_X86 &&
          node->cpu.vendor == EP_TOPO_CPU_VENDOR_INTEL) {
        pciBw = INTEL_P2P_OVERHEAD(bw);
      }
    }
  }
  */
  for (int step = 0; step < maxSteps; step ++) {
    epTopoLink* link = path->list[step];
    float fwBw = link->type == LINK_PCI ? pciBw : bw;

    epTopoLink* revLink = nullptr;
    #if USE_RECV_BW
    if (link->type == LINK_LARE) {
      EP_CHECK(findRevLink(start, link->remNode, &revLink));
    }
    #endif


    // Not enough bandwidth.
    if (link->bw < fwBw || (revLink && revLink->bw < fwBw)) {
      *retFollowedSteps = step;
      return epSuccess;
    }
    // Have enough bandwidth.
    SUB_ROUND(link->bw, fwBw);
    #if USE_RECV_BW
    if (revLink) {
      SUB_ROUND(revLink->bw, fwBw);
    }
    #endif
  }
  *retFollowedSteps = maxSteps;
  return epSuccess;
}

// Alloc bandwidth from type1/index1 to type2/index2,
// which pathType < type and have enough bw
bool epTopoSystem::allocBw(epTopoNodeType type1, int index1, epTopoNodeType type2,
                                                                          int index2, epTopoPathType type, float bw) {
  // First handle easy cases
  epTopoLinkList* path;
  if (getPath(type1, index1, type2, index2, &path) != epSuccess) {
    WARN("can't get path we are going to alloc, ignore...");
    return false;
  }
  if (path->count == 0 ) return false;

  // Check path type
  if (path->type > type) return false;

  // Check there is enough bandwidth on paths.
  int followedStep = 0;
  epTopoNode* node;
  if (getNode(type1, index1, &node) != epSuccess) {
    WARN("ignore...");
    return false;
  }
  if (followPath(path, node, path->count, bw, &followedStep)
      != epSuccess) {
    WARN("ignore...");
    return false;
  }
  if (followedStep < path->count) goto rewind;

  return true;

rewind:
  // Not enough bandwidth : rewind and return false.
  if (followPath(path, node, followedStep, -bw, &followedStep)
      != epSuccess) {
    WARN("ignore...");
    return false;
  }
  return false;
}

// Free bandwidth from type1/index1 to type2/index2
epResult_t epTopoSystem::freeBw(epTopoNodeType type1, int index1,
                                    epTopoNodeType type2, int index2, float bw) {
  epTopoLinkList* path;
  if (getPath(type1, index1, type2, index2, &path) != epSuccess) {
    WARN("can't get path we are going to free, ignore...");
    return epInternalError;
  }
  int followedStep = 0;
  epTopoNode* node;
  EP_CHECK(getNode(type1, index1, &node));
  EP_CHECK(followPath(path, node, path->count, -bw, &followedStep));
  return epSuccess;
}

epResult_t epTopoSystem::getLocalNet(int rank, int channelId, int64_t* netId, int* netDev) {
  int minType = PATH_SYS;
  float maxBw = 0;
  int count = 0;
  int64_t choseNets[EP_TOPO_MAX_NETS];
  int nets = netCount();

  int g;
  EP_CHECK(rankToIndex(rank, &g));

  for (int n = 0; n < nets; ++n) {
    epTopoLinkList* path;
    EP_CHECK(getPath(NET, n, GCU, g, &path));
    if (path->bw > maxBw || (path->bw == maxBw && path->type < minType)) {
      maxBw = path->bw;
      minType = path->type;
      count = 0;
    }
    if (path->bw == maxBw && path->type == minType) {
      epTopoNode* net;
      EP_CHECK(getNode(NET, n, &net));
      choseNets[count++] = net->id;
    }
  }
  if (netId)
    *netId = choseNets[channelId % count];

  if (netDev) {
    EP_CHECK(topoIdToNetDev(choseNets[channelId % count], netDev));
  }
  return epSuccess;
}

epResult_t epTopoSystem::getDeviceType(efmlDeviceArchitecture_t *arch) {
  epTopoNode* gcu;
  EP_CHECK(getNode(GCU, 0, &gcu));
  *arch = gcu->gcu.efmlArch;
  return epSuccess;
}
epResult_t epTopoSystem::getLocalGcus(epTopoNodeType type, int idx, float bw, int* retGcusIdx, int* retGcuCount) {
  epTopoNode* node;
  EP_CHECK(getNode(type, idx, &node));
  epTopoLinkList* paths = node->paths[GCU];
  float maxBw = 0;
  int minHops = 0xfffffff;
  for (int g = 0; g < gcuCount(); g ++) {
    if (paths[g].bw > maxBw) {
      maxBw = paths[g].bw;
      minHops = paths[g].count;
    } else if (abs(paths[g].bw - maxBw) < 1e-4 && paths[g].count < minHops) {
      minHops = paths[g].count;
    }
  }
  //INFO("maxBw(%2.1f) minHops(%d), bw(%2.1f)", maxBw, minHops, bw);
  if (maxBw >= bw) {
    int count = 0;
    for (int g = 0; g < gcuCount(); g ++) {
      if (abs(paths[g].bw - maxBw) < 1e-4 && minHops == paths[g].count)
        retGcusIdx[count ++] = g;
    }
    *retGcuCount = count;
    return epSuccess;
  }
  return epSuccess;
}

int epTopoSystem::gcuPciBw(int gcuIdx) {
  epTopoNode* gcu;
  EP_CHECK(getNode(GCU, gcuIdx, &gcu));
  for (int l = 0; l < gcu->nlinks; l ++) {
    epTopoLink* gcuLink = gcu->links + l;
    if (gcuLink->type != LINK_PCI) continue;
    epTopoNode* pci = gcuLink->remNode;
    for (int link = 0; link < pci->nlinks; link ++) {
      epTopoLink* pciLink = pci->links + link;
      if (pciLink->remNode != gcu) continue;
      return std::min(gcuLink->bw, pciLink->bw);
    }
  }
  return -1;
}

