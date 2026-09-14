#include <string>
#include <algorithm>
#include <fstream>

#include "ep.h"
#include "detector.h"
#include "utils.h"
#include "net.h"
#include "checks.h"
#include "graph.h"

#if defined(__x86_64__)
#include <cpuid.h>
#endif

#define BUSID_SIZE (sizeof("0000:00:00.0"))

static void memcpylower(char* dst, const char* src, const size_t size) {
  for (size_t i = 0; i < size; i ++) dst[i] = tolower(src[i]);
}

static epResult_t getPciPath(const char* busId, char** path) {
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpylower(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpylower(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
  *path = realpath(busPath, nullptr);
  if (*path == nullptr) {
    WARN("Could not find real path of %s", busPath);
    return epInternalError;
  }
  return epSuccess;
}

void getStrFromSys(const char* path, const char* fileName, char* retStrValue) {
  char filePath[PATH_MAX];
  sprintf(filePath, "%s/%s", path, fileName);
  int offset = 0;
  FILE* file;
  if ((file = fopen(filePath, "r")) != nullptr) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN) {
      int len = fread(retStrValue+offset, 1, MAX_STR_LEN-offset, file);
      offset += len;
    }
    fclose(file);
  }
  if (offset == 0) {
    retStrValue[0] = '\0';
    INFO(EP_GRAPH, "Topology Detection, could not read %s, ignoring", filePath);
  } else {
    retStrValue[offset-1] = '\0';
  }
}

static epResult_t setAttrFromSys(epXmlNode* pci, const char* path, const char* fileName, const char* attrName) {
  char strValue[MAX_STR_LEN] = {0};
  getStrFromSys(path, fileName, strValue);
  if (strValue[0] != '\0') { EP_CHECK(pci->attributes()->create(attrName, strValue)); }
  return epSuccess;
}

// Check whether a string is in BDF format or not.
// BDF (Bus-Device-Function) is "BBBB:BB:DD.F" where B, D and F are hex digits.
// There can be trailing chars.
static inline bool isHex(char c) { return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')); }
static bool isBDFFormat(char* bdf) {
  if (bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.') return false;
  if (isHex(bdf[0]) == 0 || isHex(bdf[1] == 0) || isHex(bdf[2] == 0) || isHex(bdf[3] == 0) ||
      isHex(bdf[5] == 0) || isHex(bdf[6] == 0) || isHex(bdf[8] == 0) || isHex(bdf[9] == 0) ||
      isHex(bdf[11] == 0)) return false;
  return true;
}

epTopoDetector::epTopoDetector(epXml* xml, struct epGcuInfo* myGcuInfo) :
  xml_(xml),
  myGcuInfo_(myGcuInfo){
}

epTopoDetector::~epTopoDetector() {
}

epResult_t epTopoDetector::fillCpu(char* numaIdStr, epXmlNode* child) {
  // When cpu is in xml.
  if (auto node = xml_->findNode("cpu", "numaid", numaIdStr)) {
    for (size_t i = 0; i < node->childSize(); ++ i) {
      if (node->childAt(i) == child) return epSuccess;
    }

    EP_CHECK(node->addChild(child));
    child->setParent(node);
    return epSuccess;
  }

  // Cpu isn't in xml, create one.
  epXmlNode* cpu = nullptr;
  EP_CHECK(xml_->createNode("cpu", &cpu));
  EP_CHECK(cpu->attributes()->create("numaid", numaIdStr));
  EP_CHECK(cpu->attributes()->create("host_hash", getHostHash()));

  char cpumaskPath[MAX_STR_LEN + sizeof("/sys/devices/system/node/node") + 1] = "/sys/devices/system/node/node0000";
  sprintf(cpumaskPath, "/sys/devices/system/node/node%s", numaIdStr);
  EP_CHECK(setAttrFromSys(cpu, cpumaskPath, "cpumap", "affinity"));

#if defined(__PPC__)
  EP_CHECK(cpu->attributes()->create("arch", "ppc64"));
#elif defined(__aarch64__)
  EP_CHECK(cpu->attributes()->create("arch", "arm64"));
#elif defined(__x86_64__)
  EP_CHECK(cpu->attributes()->create("arch", "x86_64"));
#endif

#if defined(__x86_64__)
  union {
    struct {
      // CPUID 0 String register order
      uint32_t ebx;
      uint32_t edx;
      uint32_t ecx;
    };
    char vendor[12];
  } cpuid0;

  unsigned unused;
  __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
  char vendor[13];
  strncpy(vendor, cpuid0.vendor, 12);
  vendor[12] = '\0';
  EP_CHECK(cpu->attributes()->create("vendor", vendor));

  union {
    struct {
      unsigned steppingId:4;
      unsigned modelId:4;
      unsigned familyId:4;
      unsigned processorType:2;
      unsigned resv0:2;
      unsigned extModelId:4;
      unsigned extFamilyId:8;
      unsigned resv1:4;
    };
    uint32_t val;
  } cpuid1;
  __cpuid(1, cpuid1.val, unused, unused, unused);
  int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
  int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
  EP_CHECK(cpu->attributes()->create("familyid", familyId));
  EP_CHECK(cpu->attributes()->create("modelid", modelId));
#endif

  // Connect with child node and connect with system node.
  EP_CHECK(cpu->addChild(child));
  child->setParent(cpu);
  auto system = xml_->findNode("system");
  EP_CHECK(system->addChild(cpu));
  cpu->setParent(system);
  return epSuccess;
}

epResult_t epTopoDetector::fillPciRec(const char* busId, char* path, epXmlNode* child) {
  if (busId == nullptr) {
    path[strlen(path)] = '/';
    // Detect numa_id by restored path.
    char numaIdStr[MAX_STR_LEN];
    getStrFromSys(path, "numa_node", numaIdStr);
    EP_CHECK(fillCpu(numaIdStr, child));
    return epSuccess;
  } else if (auto node = xml_->findNode("pci", "busid", busId)) {
    EP_CHECK(node->addChild(child));
    child->setParent(node);
    return epSuccess;
  }
  // Begin pci detection process.
  epXmlNode* pci = nullptr;
  EP_CHECK(xml_->createNode("pci", &pci));
  EP_CHECK(pci->attributes()->create("busid", busId));
  EP_CHECK(setAttrFromSys(pci, path, "class", "class"));
  EP_CHECK(setAttrFromSys(pci, path, "vendor", "vendor"));
  EP_CHECK(setAttrFromSys(pci, path, "device", "device"));
  EP_CHECK(setAttrFromSys(pci, path, "subsystem_vendor", "subsystem_vendor"));
  EP_CHECK(setAttrFromSys(pci, path, "subsystem_device", "subsystem_device"));

  // Detect link_speed by comparing the speed of local port
  // and remote port.
  char deviceSpeedStr[MAX_STR_LEN];
  float deviceSpeed;
  getStrFromSys(path, "max_link_speed", deviceSpeedStr);
  sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
  char portSpeedStr[MAX_STR_LEN];
  float portSpeed;
  getStrFromSys(path, "../max_link_speed", portSpeedStr);
  sscanf(portSpeedStr, "%f GT/s", &portSpeed);
  EP_CHECK(pci->attributes()->create("link_speed",
            portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr));

  // Detect link_width by comparing the width of local port
  // and remote port.
  char strValue[MAX_STR_LEN];
  getStrFromSys(path, "max_link_width", strValue);
  int deviceWidth = strtol(strValue, nullptr, 0);
  getStrFromSys(path, "../max_link_width", strValue);
  int portWidth = strtol(strValue, nullptr, 0);
  EP_CHECK(pci->attributes()->create("link_width", std::min(deviceWidth,portWidth)));

  EP_CHECK(pci->addChild(child));
  child->setParent(pci);

  // Go up one level in the PCI tree. Rewind two "/" and follow the upper PCI
  // switch, or stop if we reach a CPU root complex.
  int slashCount = 0;
  for (int parentOffset = strlen(path)-1; parentOffset>0; parentOffset--) {
    if (path[parentOffset] == '/') {
      slashCount++;
      path[parentOffset] = '\0';
      int start = parentOffset - 1;
      while (start>0 && path[start] != '/') start--;
      // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
      if (isBDFFormat(path+start+1) == 0) {
        // This a CPU root complex
        EP_CHECK(fillPciRec(nullptr, path, pci));
        return epSuccess;
      } else if (slashCount == 2) {
        // Continue on the upper PCI switch
        char busId[BUSID_SIZE + 1];
        strncpy(busId, path+start+1, BUSID_SIZE);
        busId[BUSID_SIZE] = '\0';
        EP_CHECK(fillPciRec(busId, path, pci));
        return epSuccess;
      }
    }
  }
  return epSuccess;
}

// Detect links begin from pci and end with cpu.
epResult_t epTopoDetector::fillPci(const char* busId, epXmlNode* child) {
  char* path = nullptr;
  EP_CHECK(getPciPath(busId, &path));
  EP_CHECK(fillPciRec(busId, path, child));
  free(path);
  return epSuccess;
}

epResult_t epTopoDetector::setLareNode(int port, char * remoteBusId, epXmlNode* gcu) {
  epXmlNode* lareNode = gcu->findChild("lare", "target", remoteBusId);
  std::string portDesc = std::to_string(port);
  if (lareNode == nullptr) {
    EP_CHECK(xml_->createNode("lare", &lareNode));
    EP_CHECK(lareNode->attributes()->create("target", remoteBusId));
    EP_CHECK(lareNode->attributes()->create("port_conn", portDesc.c_str()));
    EP_CHECK(lareNode->attributes()->create("count", 1));
    EP_CHECK(gcu->addChild(lareNode));
    lareNode->setParent(gcu);
  } else {
    char* portValue = lareNode->attributes()->find("port_conn")->value();
    sprintf(portValue+strlen(portValue), " %s", portDesc.c_str());
    epXmlAttributeElement* attr = lareNode->attributes()->find("count");
    attr->setValue(lareNode->attributes()->find("count")->asInt() + 1);
  }
  return epSuccess;
}

epResult_t epTopoDetector::fillTargetClass(epXmlNode* lare) {
  epXmlAttributeElement* tclassAttr = lare->attributes()->find("tclass");
  if (tclassAttr == nullptr) {
    const char* busId;
    EP_CHECK(xmlNodeGetAttr(lare, "target", &busId));
    char* path = nullptr;
    if (strcmp(busId, SWITCH_BUSID) != 0)
      getPciPath(busId, &path);

    if (path == nullptr) {
      // Remote Lare device is not visible inside this VM. Assume Switch.
      lare->attributes()->create("tclass", "0x068000");
    } else {
      EP_CHECK(setAttrFromSys(lare, path, "class", "tclass"));
      free(path);
    }
  }
  return epSuccess;
}
EP_PARAM(SimLareDisable, "SIM_LARE_DISABLE", 0);
epResult_t epTopoDetector::fillLare(epXmlNode* gcu) {
  int lareDisable = epParamSimLareDisable();
  if (lareDisable) return epSuccess;

  efmlDevice_t efmlDev;
  EP_CHECK(epEfmlDeviceGetHandleByPciBusId(myGcuInfo_->busIdStr, &efmlDev));

  epXmlNode* lareNode = gcu->findChild("lare");
  do {
    if (lareNode) break;
    if (efmlDev == nullptr) {
      WARN("No EFML device handle. Skipping lare detection for efmlDevId[%d]", myGcuInfo_->efmlDevId);
      break;
    }
    efmlFieldValue_t fv;
    fv.fieldId = EFML_FI_DEV_GCULARE_PORT_COUNT;
    EP_CHECK(epEfmlDeviceGetFieldValues(efmlDev, 1, &fv));
    if (fv.efmlReturn != EFML_SUCCESS_V2) {
      WARN("efml device %d epEfmlDeviceGetFieldValues failed, fv.efmlReturn %d", myGcuInfo_->efmlDevId, fv.efmlReturn);
      return epInternalError;
    }
    unsigned int maxLarePorts = fv.value.uiVal;
    if (maxLarePorts == 0) {
      WARN("efml device %d has not lare ports by GetFieldValues", myGcuInfo_->efmlDevId);
      return epInternalError;
    }
    unsigned int validPorts = 0;
    for (unsigned int p=0; p<maxLarePorts; ++p) {
      // Check whether we can use this Lare for P2P
      unsigned int canP2P;
      if ((epEfmlDeviceGetGcuLareCapability(efmlDev, p, EFML_GCULARE_CAP_P2P_SUPPORTED, &canP2P) != epSuccess) || !canP2P) {
        INFO(EP_GRAPH, "efml device %d lare port %d skipped. because canP2P is %u", myGcuInfo_->efmlDevId, p, canP2P);
        continue;
      }

      // Make sure the Lare is up. The previous call should have trained the link.
      efmlEnableState_t isActive = EFML_FEATURE_DISABLED;
      EP_CHECK(epEfmlDeviceGetGcuLareState(efmlDev, p, &isActive));
      if (isActive != EFML_FEATURE_ENABLED) {
        INFO(EP_GRAPH, "efml device %d lare port %d skipped. because link status is down", myGcuInfo_->efmlDevId, p);
        continue;
      }

      // Try to figure out what's on the other side of the Lare
      efmlPciInfo_t remoteProc;
      efmlIntGcuLareDeviceType_t rmtDeviceType;
      EP_CHECK(epEfmlDeviceGetGcuLareRemoteDeviceType(efmlDev, p, &rmtDeviceType));
      if (rmtDeviceType == EFML_GCULARE_DEVICE_TYPE_SWITCH) {
        strcpy(remoteProc.busId, SWITCH_BUSID);
      } else {
        EP_CHECK(epEfmlDeviceGetGcuLareRemotePciInfo(efmlDev, p, &remoteProc));
      }

      setLareNode(p, remoteProc.busId, gcu);
      validPorts++;
    }
    if (validPorts == 0) {
      WARN("Critical issue: No lare ports valid for efml device %d", myGcuInfo_->efmlDevId);
    }
  } while(0);

  // Fill target classes
  for (size_t s=0; s<gcu->childSize(); s++) {
    epXmlNode* lare = gcu->childAt(s);
    if (strcmp(lare->name(), "lare") != 0) continue;
    EP_CHECK(fillTargetClass(lare));
  }
  return epSuccess;
}

epResult_t epTopoDetector::fillGcu(epXmlNode** retGcuNode) {
  epXmlNode* gcu = nullptr;
  EP_CHECK(xml_->createNode("gcu", &gcu));
  EP_CHECK(gcu->setAttr("dev", myGcuInfo_->efmlDevId));
  EP_CHECK(gcu->setAttr("arch", myGcuInfo_->efmlArch));
  if (myGcuInfo_->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
    EP_CHECK(fillLare(gcu));
  }

  // Process links: CPU<-..-PCI<-..-GCU_PCI--GCU
  EP_CHECK(fillPci(myGcuInfo_->busIdStr, gcu));
  *retGcuNode = gcu;
  return epSuccess;
}

static bool isPciDevice(const char* pciSysPath) {
// Returns the subsystem name of a path, i.e. the end of the path
// where sysPath/subsystem points to.
  char subSysPath[PATH_MAX];
  char subSys[PATH_MAX];
  sprintf(subSysPath, "%s/subsystem", pciSysPath);
  char* path = realpath(subSysPath, nullptr);
  if (path == nullptr) {
    subSys[0] = '\0';
  } else {
    int offset;
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--);
    strcpy(subSys, path+offset+1);
    free(path);
  }
  if (strcmp(subSys, "pci") != 0) {
    INFO(EP_GRAPH, "Topology detection, network path %s is not a PCI device (%s). Attaching to first CPU", pciSysPath, subSys);
    return false;
  }
  return true;
}

static void getNumaId(char* gcuPciPath, char* numaIdStr) {
  std::string currentNumafile(gcuPciPath);
  currentNumafile = currentNumafile + "/numa_node";
  std::ifstream ifs(currentNumafile);
  if (ifs.good()) {
    getStrFromSys(gcuPciPath, "numa_node", numaIdStr);
    return;
  }
  for (int parentOffset = strlen(gcuPciPath)-1; parentOffset>0; parentOffset--) {
    if (gcuPciPath[parentOffset] != '/') continue;
    gcuPciPath[parentOffset] = '\0';
    std::string numafile(gcuPciPath);
    numafile = numafile + "/numa_node";
    std::ifstream f(numafile);
    if (!f.good()) continue;
    // Detect numa_id by restored path.
    getStrFromSys(gcuPciPath, "numa_node", numaIdStr);
    return;
  }

  //illegal branch, numaId is default empty
  numaIdStr[0] = '\0';
  WARN("there is no numa_node file found in %s and its parent path, so numaID will be empty", gcuPciPath);
}

epResult_t epTopoDetector::fillNet(const char* netPciPath, char* gcuBusId, epXmlNode** retNetNode) {
  // At last, detect net node and connect to nic we create before.
  epXmlNode* net;
  EP_CHECK(xml_->createNode("net", &net));
  // In ep xml, net node always connected by a nic node. So
  // create a nic node firstly.
  epXmlNode* nic = nullptr;

  // Detect Net's device type : pci device, not pci device(virtual, usb, ...).
  // For pci device, nic node connect to a pci node.
  // For not pci device, nic node connect to a cpu node.
  if (isPciDevice(netPciPath)) {
    int offset;
    for (offset = strlen(netPciPath) - 1; netPciPath[offset] != '/'; offset --);
    char busId[BUSID_SIZE + 1] = {0};
    strncpy(busId, netPciPath+offset+1, BUSID_SIZE);
    busId[BUSID_SIZE] = '\0';
    auto pciNode = xml_->findNode("pci", "busid", busId);
    if (pciNode == nullptr) {
      EP_CHECK(xml_->createNode("nic", &nic));
    } else {
      nic = pciNode->findChild("nic");
      if (nic == nullptr) EP_CHECK(xml_->createNode("nic", &nic));
    }
    EP_CHECK(fillPci(busId, nic));
    EP_CHECK(nic->addChild(net));
    net->setParent(nic);
  } else {
    // Find cpu which first rank gcu in same host belong to and connect this nic node to it.
    if (gcuBusId) {
      char* gcuPciPath = nullptr;
      EP_CHECK(getPciPath(gcuBusId, &gcuPciPath));
      if (gcuPciPath) {
        char numaIdStr[MAX_STR_LEN] = "";
        getNumaId(gcuPciPath, numaIdStr);
        auto cpu = xml_->findNode("cpu", "numaid", numaIdStr);
        if (cpu == nullptr) {
          EP_CHECK(xml_->createNode("nic", &nic));
        } else {
          nic = cpu->findChild("nic");
          if (nic == nullptr) EP_CHECK(xml_->createNode("nic", &nic));
        }
        EP_CHECK(fillCpu(numaIdStr, nic));
        EP_CHECK(nic->addChild(net));
        net->setParent(nic);
        free(gcuPciPath);
        gcuPciPath = nullptr;
      }
    }
  }

  *retNetNode = net;
  return epSuccess;
}

