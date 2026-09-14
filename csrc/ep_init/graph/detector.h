#ifndef EP_DETECTOR_H
#define EP_DETECTOR_H

#include "xml.h"
#include "ep_net.h"
#include "gcu_info.h"
#include "comm.h"
#include "efmlwrap.h"

/* Topology information to be detected :
 * [CPU]
 *        numaid
 *        affinity
 *        arch
 *        vendor
 *        familyid
 *        modelid
 *
 * [PCI]
 *        busid
 *        class
 *        vendor
 *        device
 *        subsystem_vendor
 *        subsystem_device
 *        link_speed
 *        link_width
 *
 * [GCU]
 *        dev  : gcu's device id.
 *        arch : gcu architecture, GCU300, GCU400, etc.
 *        rank : the rank to take control of this gcu.
 *        gdr : gcu direct rdma.
 *
 * [LARE]
 *        target : target gcu this lare connect to, which is a busid string.
 *        tclass : target class, current support GCU, in the future switch will be supported.
 *        port_conn : port connection, is one pair port(eg: "1:1") or multi-pair ports(eg: "1:1 2:2").
 *        count : lare connection have same
 *
 * [NIC]
 *        <none>
 *
 * [NET]
 *         name
 *         dev
 *         speed
 *         port
 *         latency
 *         guid
 *         maxconn
 *         gdr
 *
 */
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

class epTopoDetector {
public:
  epTopoDetector(epXml* xml, struct epGcuInfo* myGcuInfo);
  ~epTopoDetector();

  // Detector begin detection at endpoint of the topology tree,
  // which can be gcu or net.
  epResult_t fillGcu(epXmlNode** retGcuNode);
  epResult_t fillNet(const char* netPciPath, char* gcuBusId, epXmlNode** retNetNode);

private:
  epResult_t fillCpu(char* numaIdStr, epXmlNode* child);
  epResult_t fillPciRec(const char* busId, char* path, epXmlNode* child);
  epResult_t fillPci(const char* busId, epXmlNode* child);
  epResult_t fillLare(epXmlNode* gcu);
  epResult_t setLareNode(int port, char * remoteBusId, epXmlNode* gcu);
  epResult_t fillTargetClass(epXmlNode* lare);

private:
  epXml* xml_;
  struct epGcuInfo* myGcuInfo_;
};

#endif
