#include <string.h>
#include <errno.h>
#include <dlfcn.h>

#include "net.h"
#include "bootstrap.h"
#include "checks.h"


epNet_t *epNet;

epResult_t epNetInit() {
  // Always initialize bootstrap network
  EP_CHECK(bootstrapNetInit());

  // Initialize main communication network.
  // MORI (GCU-side RDMA) is the only data path, so the TCP socket backend
  // (net_socket) has been removed. IB enumeration is still needed for topology
  // detection and GDR device selection.
  epNet_t* nets[] = { &epNetIb };
  const char* netName = epGetEnv("EP_NET");
  if (netName && strlen(netName) > 0) INFO(EP_NET|EP_ENV, ENV_FORMAT_STR, "EP_NET", netName);
  bool ok = false;

  for (size_t i = 0; i < sizeof(nets) / sizeof(nets[0]); i++) {
    if (nets[i] == nullptr) continue;
    if (netName && strcmp(netName, nets[i]->name) != 0) continue;

    int ndev;
    if (nets[i]->init(epDebugLog) != epSuccess) continue;
    if (nets[i]->devices(&ndev) != epSuccess) continue;
    if (ndev <= 0) continue;
    epNet = nets[i];
    ok = true;

    break;
  }

  if (!ok) {
    WARN("Error: network %s not found.", netName ? netName : "");
    return epInvalidUsage;
  }
  return epSuccess;
}

epResult_t epNetFinalize(struct epComm* comm) {
  comm->epNet = nullptr;
  return epSuccess;
}

epResult_t epGcuGdrSupport(int* gdrSupport) {
  int netDevs;
  EP_CHECK(epNetDevices(&netDevs));
  *gdrSupport = 0;

  // MORI (GCU-side RDMA) handles the actual data path. GDR capability of a
  // device is already reflected in its properties (EP_PTR_TOPS), so no
  // host-side connection/registration test is required here.
  for (int dev = 0; dev < netDevs; dev++) {
    epNetProperties_t props;
    EP_CHECK(epNetGetProperties(dev, &props));
    if (props.ptrSupport & EP_PTR_TOPS) {
      *gdrSupport = 1;
      break;
    }
  }
  return epSuccess;
}

int epNetGetGdrDevice(epComm_t comm, int rank, char *devName) {
#define EP_NET_INVALID_DEV (-1)
  int netDevs           = EP_NET_INVALID_DEV;
  int selectedDevice    = EP_NET_INVALID_DEV;
  int gdrCapableDevices = 0;

  epResult_t result = epNetDevices(&netDevs);
  INFO(EP_NET, "[Net GDR] Evaluating %d network devices for GDR capability", netDevs);
  if (result != epSuccess || 0 >= netDevs) return EP_NET_INVALID_DEV;

  for (int netId = 0; netId < netDevs; netId++) {
    int useGdr = 0, gdrRead = 0;
    result = epTopoCheckGdr(comm->topo, rank, netId, gdrRead, &useGdr);

    if (result != epSuccess) continue;

    if (useGdr) {
      gdrCapableDevices++;
      if (selectedDevice == EP_NET_INVALID_DEV) {
        selectedDevice = netId;
      }
      TRACE(EP_NET, "[Net GDR] Device %d: GDR supported", netId);
    } else {
      TRACE(EP_NET, "[Net GDR] Device %d: No GDR support", netId);
    }
  }

  if (selectedDevice != EP_NET_INVALID_DEV) {
    INFO(EP_NET, "[Net GDR] Selected device %d (%d/%d devices support GDR)", selectedDevice, gdrCapableDevices, netDevs);
  } else {
    selectedDevice = rank % netDevs;
    INFO(EP_NET, "[Net GDR] Not found, using fallback device %d (%d devices checked)", selectedDevice, netDevs);
  }

  if (selectedDevice != EP_NET_INVALID_DEV && devName) {
    epNetProperties_t props;
    result = epNetGetProperties(selectedDevice, &props);
    if (result != epSuccess) return EP_NET_INVALID_DEV;
    strcpy(devName, props.name);
  }

  INFO(EP_NET, "[Net GDR] r[%d] netDevs[%d] selectedDevice[%d] of %s\n", rank, netDevs, selectedDevice,
    devName ? devName : "N/A");

  return selectedDevice;
}