#ifndef EP_TRANSPORT_RESOURCE_H_
#define EP_TRANSPORT_RESOURCE_H_

#include "lare_roce_ctxt.h"
#include "graph.h"
#include "transport.h"

// DeepEP no need any buff for now
#define DIRECT_BUFF_SIZE  0
#define DEFAULT_FIFO_SIZE 0

struct BuffConnInfo {
  struct epDevMemDesc<uint8_t> fifo;

  // Used for direct mode
  struct epDevMemDesc<uint8_t> directBuff;

  struct epDevMemDesc<uint64_t> controlBuff;
};
struct PcieConnInfo {
  topsIpcMemHandle_t fifo ;
  topsIpcMemHandle_t flag;
};

struct TransportConnInfo {
  int dev;
  int rank;
  struct BuffConnInfo buffConnInfo;
  int linkCount;
  union {
    struct epRoceConnInfo roce[MAX_PORTS_PER_TRUNK];
    struct PcieConnInfo pcie;
  };
};

struct localResourceInfo {
  template <E_epMallocType RegularMallocType = E_topsMalloc>
  epResult_t alloc(epComm* comm, size_t fifoSize, unsigned int flag=topsDeviceMallocDefault) {
    auto alignment = comm->capability.buffAlignment;
    if (fifoSize) {
      EP_CHECK(buffConnInfo.fifo.alloc<RegularMallocType>(fifoSize, comm->internalStream, alignment, flag));
    }
    if constexpr(DIRECT_BUFF_SIZE > 0) {
      EP_CHECK(buffConnInfo.directBuff.alloc<RegularMallocType>(DIRECT_BUFF_SIZE, comm->internalStream, alignment, flag));
    }

    // control buffers for flag/step on both direct/non-direct mode over all trunk ports
    size_t count = CONTROL_BUFF_COUNT * MAX_PORTS_PER_TRUNK + 1;     // one more Buffering
    count = count * (CONTROL_BUFF_ALIGNMENT / sizeof(uint64_t)) + 1; // one more Buffering

    EP_CHECK(buffConnInfo.controlBuff.alloc<RegularMallocType>(count, comm->internalStream, alignment, flag));
    return epSuccess;
  }

  epResult_t dealloc() {
    EP_CHECK(buffConnInfo.directBuff.dealloc());
    EP_CHECK(buffConnInfo.fifo.dealloc());
    EP_CHECK(buffConnInfo.controlBuff.dealloc());
    return epSuccess;
  }
  struct BuffConnInfo buffConnInfo;
  int linkCount = 0;
};

template <int SIDE>
class TransportResource {
  public:
    TransportResource(struct epComm* comm, struct epTopoGraph* graph, int channelId, epTopoPort const& portInfo):
      comm_(comm),
      graph_(graph),
      channelId_(channelId),
      usedPortsCount_(portInfo.count){
      localResource_.linkCount = portInfo.count;
    }
    virtual ~TransportResource()=default;
    virtual epResult_t setup() {
      size_t fifoSize = DEFAULT_FIFO_SIZE;
      // DeepEP no need any buff for now
      // if CONSTEXPR (SIDE == RECV) {
      //   for (int p = 0; p < EP_NUM_PROTOCOLS; p++) fifoSize += comm_->buffSizes[p];
      // }
      return localResource_.alloc(comm_, fifoSize);
    }
    virtual epResult_t generate(struct TransportConnInfo & connInfo) {
      TOPS_CHECK(topsGetDevice(&connInfo.dev));
      connInfo.rank = comm_->rank;
      connInfo.buffConnInfo = localResource_.buffConnInfo;
      return epSuccess;
    }
    virtual epResult_t connect(struct TransportConnInfo const& peerConnInfo, struct epConnector & connector)=0;
    virtual epResult_t connect(struct TransportConnInfo const& peerConnInfo) {
      peerResource_ = peerConnInfo.buffConnInfo;
      return epSuccess;
    }
    virtual epResult_t teardown() {
      return localResource_.dealloc();
    }
  protected:
    struct epComm* comm_;
    struct epTopoGraph* graph_;
    int channelId_;
    struct BuffConnInfo peerResource_;   //used to recv and process connectInfo that from peer rank
    struct localResourceInfo localResource_;
    int usedPortsCount_;
    const char* sideDesc_[MAX_SIDES] = {"SEND", "RECV"};
};
#endif

