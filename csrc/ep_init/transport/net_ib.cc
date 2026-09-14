/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ep.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#define ENABLE_TIMER 0
#include "timer.h"

#include <string>

#include "ibvwrap.h"
#include "mlxwrap.h"


using namespace std;
#define MAXNAMESIZE 64
static char epIbIfName[MAX_IF_NAME_SIZE+1];
static union epSocketAddress epIbIfAddr;

static int epNMergedIbDevs = -1;
#define EP_IB_MAX_DEVS_PER_NIC 2
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*EP_IB_MAX_DEVS_PER_NIC)+EP_IB_MAX_DEVS_PER_NIC
struct alignas(64) epIbMergedDev {
  int ndevs;
  int devs[EP_IB_MAX_DEVS_PER_NIC]; // Points to an index in epIbDevs
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to EP_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
  int dmaBufSupported;               //  0 = uninit, 1 = yes, -1 = no
};

struct epIbStats {
  int fatalErrorCount;
};

static int epNIbDevs = -1;
struct alignas(64) epIbDev {
  pthread_mutex_t lock;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char pciPath[PATH_MAX];
  int realPort;
  int maxQp;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
  struct epIbStats stats;
};

#define MAX_IB_DEVS 32
struct epIbMergedDev epIbMergedDevs[MAX_IB_DEVS];
struct epIbDev epIbDevs[MAX_IB_DEVS];
pthread_mutex_t epIbLock = PTHREAD_MUTEX_INITIALIZER;
static int epIbRelaxedOrderingEnabled = 0;

EP_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
EP_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);

static epResult_t epIbStatsInit(struct epIbStats* stat) {
  __atomic_store_n(&stat->fatalErrorCount, 0, __ATOMIC_RELAXED);
  return epSuccess;
}
static void epIbStatsFatalError(struct epIbStats* stat){
  __atomic_fetch_add(&stat->fatalErrorCount, 1, __ATOMIC_RELAXED);
}
static void epIbQpFatalError(struct ibv_qp* qp) {
  epIbStatsFatalError((struct epIbStats*)qp->qp_context);
}
static void epIbCqFatalError(struct ibv_cq* cq) {
  epIbStatsFatalError((struct epIbStats*)cq->cq_context);
}
static void epIbDevFatalError(struct epIbDev* dev) {
  epIbStatsFatalError(&dev->stats);
}

pthread_t epIbAsyncThread;
static void* epIbAsyncThreadMain(void* args) {
  struct epIbDev* dev = (struct epIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (epSuccess != wrap_ibv_get_async_event(__atomic_load_n(&dev->context, __ATOMIC_RELAXED), &event)) { break; }
    char *str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (epSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // the above is device fatal error
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      epIbDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      // the above is a CQ fatal error
      WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->portNum, cq, str);
      epIbCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      // the above are QP fatal errors
      WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->portNum, qp, str);
      epIbQpFatalError(qp);
      break;
    case IBV_EVENT_SRQ_ERR:
      // SRQ are not used in EP
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // the above are non-fatal
      WARN("NET/IB : %s:%d Got async error event: %s", dev->devName, dev->portNum, str);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (epSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

EP_PARAM(IbDisable, "IB_DISABLE", 0);
EP_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
EP_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);

static epResult_t epIbGetPciPath(char* devName, char* path, int* realPort) {
  char devicePath[PATH_MAX];
  char resolved_devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, resolved_devicePath);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (epParamIbMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d=0; d<epNIbDevs; d++) {
      if (strcmp(p, epIbDevs[d].pciPath) == 0) (*realPort)++;
    }
  }
  strncpy(path, p, PATH_MAX);
  return epSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
  10000, /* QDR */
  14000, /* FDR */
  25000, /* EDR */
  50000, /* HDR */
  100000 /* NDR */ };

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}
static int epIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int epIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int epIbRelaxedOrderingCapable(void) {
  int roMode = epParamIbPciRelaxedOrdering();
  epResult_t r = epInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == epInternalError ? 0 : 1;
}

// Compare epIbDev[dev] to all stored mergedIbDevs
int epIbFindMatchingDev(int dev) {
  for (int i = 0; i < epNMergedIbDevs; i++) {
    if (epIbMergedDevs[i].ndevs < EP_IB_MAX_DEVS_PER_NIC) {
      int compareDev = epIbMergedDevs[i].devs[0];
      if (strcmp(epIbDevs[dev].pciPath, epIbDevs[compareDev].pciPath) == 0 &&
          (epIbDevs[dev].guid == epIbDevs[compareDev].guid) &&
          (epIbDevs[dev].link == epIbDevs[compareDev].link)) {
          TRACE(EP_NET, "NET/IB: Matched name1=%s pciPath1=%s guid1=0x%lx link1=%u name2=%s pciPath2=%s guid2=0x%lx link2=%u",
            epIbDevs[dev].devName, epIbDevs[dev].pciPath, epIbDevs[dev].guid, epIbDevs[dev].link,
            epIbDevs[compareDev].devName, epIbDevs[compareDev].pciPath, epIbDevs[compareDev].guid, epIbDevs[compareDev].link);
          return i;
      }
    }
  }

  return epNMergedIbDevs;
}

epResult_t epIbInit(epDebugLogger_t logFunction) {
  epResult_t ret = epSuccess;
  if (epParamIbDisable()) return epInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != epSuccess) { return epInternalError; }

  if(wrap_mlx5_symbols() == epSuccess) INFO(EP_NET, "NET/IB : load mlx5 symbols success.");
  else INFO(EP_NET, "NET/IB : mlx5 symbols unload, skip.");

  if (epNIbDevs == -1) {
    pthread_mutex_lock(&epIbLock);
    wrap_ibv_fork_init();
    if (epNIbDevs == -1) {
      epNIbDevs = 0;
      epNMergedIbDevs = 0;
      if (epFindInterfaces(epIbIfName, &epIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = epInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      const char* userIbEnv = epGetEnv("EP_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(EP_NET|EP_ENV, ENV_FORMAT_STR, "EP_IB_HCA", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (epSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) { ret = epInternalError; goto fail; }

      // Should EP merge multi-port devices into one?
      int mergeNics;
      mergeNics = epParamIbMergeNics();
build_ib_list:
      for (int d=0; d<nIbDevs && epNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context;
        if (epSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (epSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (epSuccess != wrap_ibv_close_device(context)) { ret = epInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          struct ibv_port_attr portAttr;
          if (epSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
            WARN("NET/IB : Unable to query port_num %d", port_num);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
            continue;
          }
          pthread_mutex_init(&epIbDevs[epNIbDevs].lock, NULL);
          epIbDevs[epNIbDevs].device = d;
          epIbDevs[epNIbDevs].guid = devAttr.sys_image_guid;
          epIbDevs[epNIbDevs].portAttr = portAttr;
          epIbDevs[epNIbDevs].portNum = port_num;
          epIbDevs[epNIbDevs].link = portAttr.link_layer;
          epIbDevs[epNIbDevs].speed = epIbSpeed(portAttr.active_speed) * epIbWidth(portAttr.active_width);
          __atomic_store_n(&epIbDevs[epNIbDevs].context, context, __ATOMIC_RELAXED);
          epIbDevs[epNIbDevs].pdRefs = 0;
          epIbDevs[epNIbDevs].pd = NULL;
          strncpy(epIbDevs[epNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          EP_CHECKGOTO(epIbGetPciPath(epIbDevs[epNIbDevs].devName, epIbDevs[epNIbDevs].pciPath, &epIbDevs[epNIbDevs].realPort), ret, fail);
          epIbDevs[epNIbDevs].maxQp = devAttr.max_qp;
          EP_CHECK(epIbStatsInit(&epIbDevs[epNIbDevs].stats));

          // Enable ADAPTIVE_ROUTING by default on IB networks
          // But allow it to be overloaded by an env parameter
          epIbDevs[epNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
          if (epParamIbAdaptiveRouting() != -2) epIbDevs[epNIbDevs].ar = epParamIbAdaptiveRouting();

          TRACE(EP_NET,"NET/IB: [%d] %s:%s:%d/%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name, epIbDevs[epNIbDevs].portNum,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE",
              epIbDevs[epNIbDevs].speed, context,
              epIbDevs[epNIbDevs].pciPath, epIbDevs[epNIbDevs].ar);

          PTHREADCHECKGOTO(pthread_create(&epIbAsyncThread, NULL, epIbAsyncThreadMain, epIbDevs + epNIbDevs), "pthread_create", ret, fail);
          epSetThreadName(epIbAsyncThread, "EP IbAsync %2d", epNIbDevs);
          PTHREADCHECKGOTO(pthread_detach(epIbAsyncThread), "pthread_detach", ret, fail); // will not be pthread_join()'d

          int mergedDev = epNMergedIbDevs;
          if (mergeNics) {
            mergedDev = epIbFindMatchingDev(epNIbDevs);
          }

          // No matching dev found, create new mergedDev entry (it's okay if there's only one dev inside)
          if (mergedDev == epNMergedIbDevs) {
            // Set ndevs to 1, assign first ibDevN to the current IB device
            epIbMergedDevs[mergedDev].ndevs = 1;
            epIbMergedDevs[mergedDev].devs[0] = epNIbDevs;
            epNMergedIbDevs++;
            strncpy(epIbMergedDevs[mergedDev].devName, epIbDevs[epNIbDevs].devName, MAXNAMESIZE);
          // Matching dev found, edit name
          } else {
            // Set next device in this array to the current IB device
            int ndevs = epIbMergedDevs[mergedDev].ndevs;
            epIbMergedDevs[mergedDev].devs[ndevs] = epNIbDevs;
            epIbMergedDevs[mergedDev].ndevs++;
            snprintf(epIbMergedDevs[mergedDev].devName + strlen(epIbMergedDevs[mergedDev].devName), MAXNAMESIZE+1, "+%s", epIbDevs[epNIbDevs].devName);
          }

          // Aggregate speed
          epIbMergedDevs[mergedDev].speed += epIbDevs[epNIbDevs].speed;
          epNIbDevs++;
          nPorts++;
        }
        if (nPorts == 0 && epSuccess != wrap_ibv_close_device(context)) { ret = epInternalError; goto fail; }
      }

      // Detect if there are both multi-port and single-port NICs in the system. If so, disable port merging and build the list again
      if (mergeNics) {
        for (int d = 0; d < epNMergedIbDevs; d++) {
          if (epIbMergedDevs[d].ndevs != epIbMergedDevs[0].ndevs) {
            INFO(EP_NET, "Detected a mix of single and multiple-port NICs. Force-disabling EP_IB_MERGE_NICS");
            mergeNics = 0;
            epNIbDevs = 0;
            epNMergedIbDevs = 0;
            memset(epIbMergedDevs, 0, sizeof(epIbMergedDevs));
            goto build_ib_list;
          }
        }
      }

      if (nIbDevs && (epSuccess != wrap_ibv_free_device_list(devices))) { ret = epInternalError; goto fail; };
    }
    if (epNIbDevs == 0) {
      INFO(EP_INIT|EP_NET, "NET/IB : No device found.");
    } else {
      char line[2048];
      line[0] = '\0';
      // Determine whether RELAXED_ORDERING is enabled and possible
      epIbRelaxedOrderingEnabled = epIbRelaxedOrderingCapable();
      for (int d = 0; d < epNMergedIbDevs; d++) {
        struct epIbMergedDev* mergedDev = epIbMergedDevs + d;
        if (mergedDev->ndevs > 1) {
          // Print out merged dev info
          snprintf(line+strlen(line), 2047-strlen(line), " [%d]={", d);
          for (int i = 0; i < mergedDev->ndevs; i++) {
            int ibDev = mergedDev->devs[i];
            snprintf(line+strlen(line), 2047-strlen(line), "[%d] %s:%d/%s%s", ibDev, epIbDevs[ibDev].devName,
              epIbDevs[ibDev].portNum, epIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE",
              // Insert comma to delineate
              i == (mergedDev->ndevs - 1) ? "" : ", ");
          }
          snprintf(line+strlen(line), 2047-strlen(line), "}");
        } else {
          int ibDev = mergedDev->devs[0];
          snprintf(line+strlen(line), 2047-strlen(line), " [%d]%s:%d/%s", ibDev, epIbDevs[ibDev].devName,
            epIbDevs[ibDev].portNum, epIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
        }
      }
      line[2047] = '\0';
      char addrline[SOCKET_NAME_MAXLEN+1];
      INFO(EP_INIT|EP_NET, "NET/IB : Using%s %s; OOB %s:%s", line, epIbRelaxedOrderingEnabled ? "[RO]" : "",
           epIbIfName, epSocketToString(&epIbIfAddr, addrline));
    }
    pthread_mutex_unlock(&epIbLock);
  }
exit:
  (void)logFunction;
  return ret;
fail:
  pthread_mutex_unlock(&epIbLock);
  goto exit;
}

epResult_t epIbDevices(int* ndev) {
  *ndev = epNMergedIbDevs;
  return epSuccess;
}

// Detect whether GDR can work on a given NIC with the current TOPS device
// Returns :
// epSuccess : GDR works
// epSystemError : no module or module loaded but not supported by GCU
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static int epIbGdrModuleLoaded = 0; // 1 = true, 0 = false
static void ibGdrSupportInitOnce() {
  // Check for the nv_peer_mem module being loaded
  epIbGdrModuleLoaded = KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/enflame_peer_mem/version") ||
                          KNL_MODULE_LOADED("/sys/module/enflame_peer_mem/version") ||
                          KNL_MODULE_LOADED("/sys/bus/pci/drivers/enflame/peermem_client_version");
}
epResult_t epIbGdrSupport() {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, ibGdrSupportInitOnce);
  if (!epIbGdrModuleLoaded)
    return epSystemError;
  return epSuccess;
}

static __thread int ibDmaSupportInitDev; // which device to init, must be thread local
static void ibDmaBufSupportInitOnce(){
//not support
#if 0
  epResult_t res;
  // select the appropriate
  struct epIbMergedDev* mergedDev = epIbMergedDevs + ibDmaSupportInitDev;
  // Test each real devices
  int dev_fail = 0;
  for (int i = 0; i < mergedDev->ndevs; i++) {
    int ibDev = mergedDev->devs[i];
    struct ibv_pd* pd;
    struct ibv_context* ctx = epIbDevs[ibDev].context;
    EP_CHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
    // Test kernel DMA-BUF support with a dummy call (fd=-1)
    (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
    // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
    dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
    EP_CHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
    // stop the search and goto failure
    if (dev_fail) goto failure;
  }
  mergedDev->dmaBufSupported = 1;
  return;
failure:
#endif
  struct epIbMergedDev* mergedDev = epIbMergedDevs + ibDmaSupportInitDev;
  mergedDev->dmaBufSupported = -1;
  return;
}

// Detect whether DMA-BUF support is present in the kernel
// Returns :
// epSuccess : DMA-BUF support is available
// epSystemError : DMA-BUF is not supported by the kernel
epResult_t epIbDmaBufSupport(int dev) {
  struct oncewrap {
    pthread_once_t once = PTHREAD_ONCE_INIT;
  };
  static oncewrap once[MAX_IB_DEVS];
  // init the device only once
  ibDmaSupportInitDev = dev;
  pthread_once(&once[dev].once, ibDmaBufSupportInitOnce);

  int dmaBufSupported = epIbMergedDevs[dev].dmaBufSupported;
  if (dmaBufSupported == 1) return epSuccess;
  return epSystemError;
}

#define EP_NET_IB_MAX_RECVS 8

epResult_t epIbGetProperties(int dev, epNetProperties_t* props) {
  struct epIbMergedDev* mergedDev = epIbMergedDevs+dev;
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;

  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  struct epIbDev* ibDev = epIbDevs + mergedDev->devs[0];
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = EP_PTR_HOST;
  if (epIbGdrSupport() == epSuccess) {
    props->ptrSupport |= EP_PTR_TOPS; // GDR support via nv_peermem
  }
  props->regIsGlobal = 1;
  if (epIbDmaBufSupport(dev) == epSuccess) {
    props->ptrSupport |= EP_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = EP_NET_IB_MAX_RECVS;
  props->netDeviceType    = EP_NET_DEVICE_INVALID;
  props->netDeviceVersion = EP_NET_DEVICE_INVALID_VERSION;
  return epSuccess;
}

// Note: only device enumeration/init is implemented here. Actual data transfer
// (QP creation, send/recv) is handled by MORI on the GCU device side, so the
// following entry points are intentionally left as NULL.
epNet_t epNetIb = {
  "IB",
  epIbInit,
  epIbDevices,
  epIbGetProperties,
  NULL /* listen */,
  NULL /* connect */,
  NULL /* accept */,
  NULL /* regMr */,
  NULL /* regMrDmaBuf */,
  NULL /* deregMr */,
  NULL /* isend */,
  NULL /* irecv */,
  NULL /* iflush */,
  NULL /* test */,
  NULL /* closeSend */,
  NULL /* closeRecv */,
  NULL /* closeListen */,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */
};

