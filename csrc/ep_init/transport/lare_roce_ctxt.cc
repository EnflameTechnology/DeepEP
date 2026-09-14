#include "lare_roce_ctxt.h"
#include "checks.h"
#include "align.h"

#define MAX_QP_PER_LARE 248  //qp[0-247] for commlib, qp[249-255] for other user
#define SQ_USER_COMMLIB 1    //it indicates driver will allocate qpid from fixed pool for communication lib
#define SEND_WQE_BB 64
#define GCU_PAGE_SHIFT   12
#define GCU_PAGE_SIZE    (1ULL << GCU_PAGE_SHIFT)

LareRoceCtxt::LareRoceCtxt(struct epComm* comm, uint32_t efmlPortId):
  comm_(comm) {
  connInfo_.qpInfo.portId = efmlPortId;
  connInfo_.qpInfo.qpId = 0xff;
  connInfo_.qpConfig.mac = 0xdeadbeaf;
  connInfo_.qpConfig.ip = 0xdeadbeaf;
}
LareRoceCtxt::~LareRoceCtxt(){
}

epResult_t LareRoceCtxt::setup() {
  const size_t sqSize = EP_MAX_ROCE_SQELEM_NUM* SEND_WQE_BB;
  EP_CHECK(sq_.alloc<E_topsMalloc>(sqSize, comm_->internalStream, GCU_PAGE_SIZE));

  /*sq baseaddr which is page-aligned is needed for Qps of lare ports.
   *althrough topsMalloc always returns page-aligned address at present, but we don't make this a prerequisite.
   */
  struct epRoceQpInfo * localQp = &connInfo_.qpInfo;
  topsError_t topsErr = topsRoceCreateQueue(localQp->portId, sq_.alignedPtr, sqSize, SQ_USER_COMMLIB, &localQp->qpId);
  CheckPrintAndDo(topsErr == topsSuccess, return epUnhandledTopsError,
                                    "topsRoceCreateQueue(port[%u], sq[%p], sq_size[%lu], sq_user[1], qpId_o[%u]) = %s\n",
                               localQp->portId, sq_.alignedPtr, sqSize, localQp->qpId, topsGetErrorString(topsErr));
  topsErr = topsRoceQueryQueue(localQp->portId, localQp->qpId, &connInfo_.qpConfig.mac, &connInfo_.qpConfig.ip);
  CheckPrintAndDo(topsErr == topsSuccess, return epUnhandledTopsError,
                                                          "topsRoceQueryQueue(%u, %u, mac, ip) = %s\n",
                                                          localQp->portId, localQp->qpId, topsGetErrorString(topsErr));
  return epSuccess;
}
epResult_t LareRoceCtxt::get(struct epRoceConnInfo & roceConnInfo) {
  roceConnInfo = connInfo_;
  return epSuccess;
}

epResult_t LareRoceCtxt::connect(struct epRoceConnInfo const& peerQpInfo, struct epLareDevInfo & localLareInfo) {
  topsError_t topsErr = topsSuccess;
  localLareInfo.sqInfo.baseAddr = (void *)sq_.alignedDevPtr;
  localLareInfo.qpInfo = connInfo_.qpInfo;
  topsErr = topsRoceBindQueuePair(connInfo_.qpInfo.portId, connInfo_.qpInfo.qpId,
                                               peerQpInfo.qpInfo.qpId, peerQpInfo.qpConfig.mac, peerQpInfo.qpConfig.ip);
  CheckPrintAndDo(topsErr == topsSuccess, return epUnhandledTopsError,
                                    "topsRoceBindQueuePair(%u, %u, %u, 0x%lx, 0x%x) = %s\n",
                                    connInfo_.qpInfo.portId, connInfo_.qpInfo.qpId, peerQpInfo.qpInfo.qpId,
                                    peerQpInfo.qpConfig.mac, peerQpInfo.qpConfig.ip, topsGetErrorString(topsErr));
  return epSuccess;
}
epResult_t LareRoceCtxt::registerMem(void *inputPeerDevAddr, size_t size, uint64_t* outputPA) {
  void* regMem = nullptr;
  topsError_t topsErr = topsRoceRegMem(connInfo_.qpInfo.portId, connInfo_.qpInfo.qpId, inputPeerDevAddr, size, &regMem);
  CheckPrintAndDo(topsErr == topsSuccess, return epUnhandledTopsError, "topsRoceRegMem(%u, %u, %p, %lu, %p) = %s\n",
           connInfo_.qpInfo.portId, connInfo_.qpInfo.qpId, inputPeerDevAddr, size, regMem, topsGetErrorString(topsErr));
  regAddrs_.push(regMem);
  TOPS_CHECK(topsPointerGetAttribute(outputPA, TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER, regMem));
  return epSuccess;
}

epResult_t LareRoceCtxt::teardown() {
  topsError_t topsErr = topsSuccess;
  if (connInfo_.qpInfo.qpId < MAX_QP_PER_LARE) {
    topsErr = topsRoceDeleteQueue(connInfo_.qpInfo.portId, connInfo_.qpInfo.qpId);
    CheckPrintAndDo(topsErr == topsSuccess, return epUnhandledTopsError,
                                    "topsRoceDeleteQueue(%u, %u) = %s\n",
                                    connInfo_.qpInfo.portId, connInfo_.qpInfo.qpId, topsGetErrorString(topsErr));
    connInfo_.qpInfo.qpId = 0xff;
  }
  EP_CHECK(sq_.dealloc());

  connInfo_.qpConfig.mac = 0xdeadbeaf;
  connInfo_.qpConfig.ip = 0xdeadbeaf;

  while (!regAddrs_.empty()) {
    auto regMem = regAddrs_.top();
    topsErr = topsRoceUnregMem(regMem);
    CheckPrintAndDo(topsErr == topsSuccess, continue, "topsRoceUnregMem(%p) = %s\n", regMem, topsGetErrorString(topsErr));
    regAddrs_.pop();
  }
  return epSuccess;
}

