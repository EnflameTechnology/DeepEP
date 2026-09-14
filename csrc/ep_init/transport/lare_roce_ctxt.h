#ifndef EP_LARE_ROCE_CTXT_H_
#define EP_LARE_ROCE_CTXT_H_
#include "comm.h"
#include <stack>

struct epRoceQpConfig {
  uint64_t mac;
  uint32_t ip;
};
struct epRoceConnInfo {
  struct epRoceQpInfo qpInfo;
  struct epRoceQpConfig qpConfig;
};

class LareRoceCtxt {
  public:
    LareRoceCtxt(struct epComm* comm, uint32_t efmlPortId);
    ~LareRoceCtxt();
    epResult_t setup();
    epResult_t get(struct epRoceConnInfo & roceConnInfo);
    epResult_t connect(struct epRoceConnInfo const& peerQpInfo, struct epLareDevInfo & localLareInfo);
    epResult_t teardown();
    epResult_t registerMem(void *inputPeerDevAddr, size_t size, uint64_t* outputPA);
    struct epComm* comm_;
    struct epRoceConnInfo connInfo_;
    struct epDevMemDesc<uint8_t> sq_;
    std::stack<void*> regAddrs_;
};
#endif
