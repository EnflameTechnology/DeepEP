/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_DEVICE_H_
#define EP_DEVICE_H_

#include "ep.h"

#include "align.h"

#define EP_NUM_FUNCTIONS 5 // SendRecv not included for now
typedef enum { epFuncBroadcast, epFuncReduce, epFuncAllGather, epFuncReduceScatter, epFuncAllReduce, epFuncSendRecv, epFuncSend, epFuncRecv, epNumFuncs } epFunc_t;

#define EP_MAX_ALGORITHMS_NUM 1
#define EP_NUM_ALGORITHMS 1
#define EP_ALGO_UNDEF -1
#define EP_ALGO_RING 0
#define EP_ALGO_TREE 0
#define EP_ALGO_MESH 0
#define EP_ALGO_MESH_DIRECT 0
#define EP_ALGO_HYBRID_TREE 0

#define EP_NUM_PROTOCOLS 2 // Simple
#define EP_PROTO_UNDEF -1
#define EP_PROTO_LL128 0
#define EP_PROTO_SIMPLE 1

#define EP_ALGO_PROTO_IGNORE -1.0

// Support up to 256-rank single superpod; 512 reserved as future expansion
#define EP_MAX_LOCAL_RANKS 256

#define MAXTHREADS 6
#define MAXCHANNELS 16
#define MAX_BLOCK_PER_TASK 2
#define EP_MAX_MESHX_CHANNELS 14
#define EP_MAX_RANKS 256
#define EP_MAX_OPS 1024
#define EP_MAX_WORK_ELEMENTS 1
#define EP_MAX_WORK_ELEMENTS_P2P 2
#define EP_MAX_STEPS 32


// Fixed per-channel-per-rank peerDirectBuffAddrs size in uint64 slots.
// 1024 * 8 = 8 KB
#define EP_PEER_DIRECT_BUFF_SIZE 1024

#define MAX_SHM_SIZE (12*1024*1024)
#define TRANSPORT_TYPE_P2P 0x01
#define TRANSPORT_TYPE_SHM 0x02
#define TRANSPORT_TYPE_NET 0x04
#define TRANSPORT_TYPE_LARE 0x08

#define EP_LL128_LINESIZE 128
#define EP_LL128_FLAGSIZE (sizeof(uint32_t))
#define EP_LL128_LINEELEMS (EP_LL128_LINESIZE/sizeof(uint32_t))
#define EP_LL128_DATAELEMS (EP_LL128_LINEELEMS-1)

// TODO: may need better value
#define EP_LL128_MAX_NTHREADS 160
#define EP_LL128_ELEMS_PER_THREAD 120 // ?

#if __GCU_ARCH__ >= 400
#define EP_LL128_VECTOR_ELEMS_PER_THREAD 4
#else
#define EP_LL128_VECTOR_ELEMS_PER_THREAD 8
#endif
#define EP_LL128_VECTOR_SIZE (EP_LL128_VECTOR_ELEMS_PER_THREAD*EP_LL128_MAX_NTHREADS)

#define OFFSET_OF(s, m) (reinterpret_cast<size_t>(&reinterpret_cast<const volatile char&>(static_cast<s*>(nullptr)->m)))
#define MAPPING_WORKERID_TO_CHANNELID(_workId, _workersPerChn)                 \
  ((_workId) / (_workersPerChn))
#define MAPPING_WORKERID_TO_MY_WORKERID(_workId, _workersPerChn)               \
  ((_workId) % (_workersPerChn))
#define COUNTING_MY_WORKERID_W_CHANNEL(_myWorkId) (0 == (_myWorkId))

#define MEM_ALIGN 1024 // 4K ?

#define SEND 0
#define RECV 1
#define MAX_SIDES 2
#define EP_MAX_ROCE_SQELEM_NUM 256

#define MAX_PORTS_PER_TRUNK 1
#define MASTER_PORTID 0

struct epSendMem {
  union {
    struct {
      uint64_t head;
      uint64_t step;
      void* ptrExchange;
      void* buffs; // Buffs on device
    };
    char pad[MEM_ALIGN];
  };

};

struct epRecvMem {
  union {
    struct {
      uint64_t tail;
      uint64_t step;
      int sizesFifo[EP_MAX_STEPS];
      //host ptr for shared buffers
      void* ptrsFifo[EP_MAX_STEPS];
      //edf addr for shared buffers, kernel use
      uint64_t devPtrsFifo[EP_MAX_STEPS];
      void* buffs; // Buffs on device
    };
    char pad[MEM_ALIGN];
  };
};

struct epRoceQpInfo {
  uint32_t portId; //efml port
  uint32_t qpId;
};

struct epRoceSqInfo {
  void* baseAddr;
};

struct epLareDevInfo {
  struct epRoceQpInfo qpInfo;
  struct epRoceSqInfo sqInfo;
};

// For gcu400:
//  ALIGNMENT: must be same with capability.buffAlignment
//  OFFSET: cover all trunk ports and address aligned with ALIGNMENT
// Control                                                                                *
//  Buff                                                                                  *
//   | Aligned                                                                            *
//   |---|                                                                                *
//      step     portN   directStep  portN      flag     portN   directFlag  .........MAX *
//       +- ..... -+- ..... -+- ..... -+- ..... -+- ..... -+- ..... -+- ..... -+- .....|  *
//       |<-  OFFSET_TRUNK ->|<-  OFFSET_TRUNK ->|<-  OFFSET_TRUNK ->|  ...............|  *
//       |    N    |                                                                      *
//       |    x    |                                                                      *
//       |  OFFSET |                                                                      *
//       |  PORTS  |                                                                      *
#define CONTROL_BUFF_STEP                           (0)     // Index for step
#define CONTROL_BUFF_DIRECT_STEP                    (1)     // Index for direct-step
#define CONTROL_BUFF_FLAG                           (2)     // Index for flag
#define CONTROL_BUFF_DIRECT_FLAG                    (3)     // Index for direct-flag
#define CONTROL_BUFF_COUNT                          (4 + 1) // total count
#define CONTROL_BUFF_ALIGNMENT                      (128)
#define PTR_OFFSET_PER_PORTS                        (CONTROL_BUFF_ALIGNMENT / sizeof(uint64_t *))
#define PTR_OFFSET_PER_TRUNK                        (PTR_OFFSET_PER_PORTS * MAX_PORTS_PER_TRUNK)
#define CONTROL_BUFF_BASE_GET(_port)                (PTR_OFFSET_PER_PORTS * (_port))
#define CONTROL_BUFF_OFFSET_GET(_field)             (PTR_OFFSET_PER_TRUNK * (CONTROL_BUFF_ ## _field))

// buffInfo from struct epSimpleConnInfo => contxt => buffInfo
#define CONN_BUFF_BUF_ADDR(_conn)                   ((_conn).contxt.buffInfo.bufAddr)
#define CONN_BUFF_DIRECT_ADDR(_conn)                ((_conn).contxt.buffInfo.directAddr)
#define CONN_BUFF_SIZES_FIFO(_conn, _p)             ((_conn).contxt.buffInfo.sizesFifo[_p])
#define CONN_BUFF_SIZES_DEVPTRS_FIFO(_conn, _p)     ((_conn).contxt.buffInfo.devPtrsFifo[_p])
#define CONN_BUFF_STEP_CACHE(_conn, _p)             ((_conn).contxt.buffInfo.stepCache[_p])
#define CONN_BUFF_STEP_PTR(_conn, _p)               ((_conn).contxt.buffInfo.step[_p])
#define CONN_BUFF_PEER_FLAG_PTR(_conn, _p)          ((_conn).contxt.buffInfo.peerFlag[_p])
#define CONN_BUFF_LOCAL_FLAG_PTR(_conn, _p)         ((_conn).contxt.buffInfo.localFlag[_p])
#define CONN_BUFF_DIRECT_STEP_PTR(_conn, _p)        ((_conn).contxt.buffInfo.step[_p] + (PTR_OFFSET_PER_TRUNK))
#define CONN_BUFF_DIRECT_PEER_FLAG_PTR(_conn, _p)   ((_conn).contxt.buffInfo.peerFlag[_p] + (PTR_OFFSET_PER_TRUNK))
#define CONN_BUFF_DIRECT_LOCAL_FLAG_PTR(_conn, _p)  ((_conn).contxt.buffInfo.localFlag[_p] + (PTR_OFFSET_PER_TRUNK))

static_assert(((CONTROL_BUFF_DIRECT_STEP - CONTROL_BUFF_STEP) == 1), "Delta mismatch to get CONN_BUFF_DIRECT_STEP_PTR");
static_assert(((CONTROL_BUFF_DIRECT_FLAG - CONTROL_BUFF_FLAG) == 1), "Delta mismatch to get DIRECT_PEER_FLAG_PTR & CONN_BUFF_DIRECT_LOCAL_FLAG_PTR");

struct epBuffInfo {
  uint64_t *step[MAX_PORTS_PER_TRUNK];
  uint64_t bufAddr;
  uint64_t stepCache[MAX_PORTS_PER_TRUNK];
  uint64_t * peerFlag[MAX_PORTS_PER_TRUNK];
  uint64_t * localFlag[MAX_PORTS_PER_TRUNK];

  // Used for direct mode
  uint64_t directAddr;
  // uint64_t *directStep[MAX_PORTS_PER_TRUNK];
  // uint64_t *directPeerFlag[MAX_PORTS_PER_TRUNK];
  // uint64_t *directLocalFlag[MAX_PORTS_PER_TRUNK];

  // Sizes fifo from GCU to proxy
  int *sizesFifo[MAX_PORTS_PER_TRUNK];
  // Buffer fifo edf addr from proxy to GCU when using share buffer for send/recv
  uint64_t *devPtrsFifo[MAX_PORTS_PER_TRUNK];
};
static_assert((CONTROL_BUFF_ALIGNMENT % sizeof(uint64_t*)) == 0, "BUFF_ALIGNMENT is not best for read");

struct epSimpleConnInfoCtxt {
  uint32_t usedPortsCount;
  uint32_t transType;
  struct epLareDevInfo localLareInfo[MAX_PORTS_PER_TRUNK];
  struct epBuffInfo buffInfo;
};
using T_U64=unsigned long long;
constexpr auto nelemPerConnInfo = sizeof(struct epSimpleConnInfoCtxt)/sizeof(T_U64);
static_assert((sizeof(struct epSimpleConnInfoCtxt) <= 152), "Be care of stack size impact to 4.0");
static_assert((sizeof(struct epSimpleConnInfoCtxt) % sizeof(T_U64)) == 0,
                                                                  "epSimpleConnInfo length is mismatch for read");

struct epSimpleConnInfo {
  union {
    struct epSimpleConnInfoCtxt contxt;
    T_U64 data[nelemPerConnInfo];
  };
};

//for gcu400
struct epConnInfoV4 {
  struct epSimpleConnInfo simple;
  struct epBuffInfo ll128;
};

struct epConnInfo {
  union {
    //TODO: need merge here. common member lists below for gcu300 and gcu400:
    //uint64_t buffs[EP_NUM_PROTOCOLS]
    //int transType
    //uint64_t *tail[MAX_PORTS_PER_TRUNK];   GCU300 user master port
    //uint64_t *head[MAX_PORTS_PER_TRUNK];   GCU300 user master port
    //int *sizesFifo;      for proxy
    //uint64_t *devPtrsFifo; for proxy
    //uint64_t *step[MAX_PORTS_PER_TRUNK];   GCU300 user master port

    struct {
      uint64_t devPtrBase; // rem res base. rem tail+fifo base when send
      uint64_t buffs[EP_NUM_PROTOCOLS];  // Local for recv, remote for send
      char *buffers[EP_NUM_PROTOCOLS];
      int transType;
      int shared;         // Buffers are shared
      uint64_t *tail;
      uint64_t *head;

      // Sizes fifo from GCU to proxy
      int *sizesFifo;
      // Buffer fifo edf addr from proxy to GCU when using share buffer for send/recv
      uint64_t *devPtrsFifo;

      // Keep where we are
      uint64_t *step;
      uint8_t reserved[192]; //total 280B is best performance for gcu300
    };
    struct {
      struct epConnInfoV4* hostConnInfoV4Ptr;
      struct epConnInfoV4* devConnInfoV4Ptr;
    };
  };
};

struct epProxyConnector {
  bool initialized;
  int rank;
  int tpRank;
  int tpLocalRank;
  int sameProcess;
  struct epProxyConnection* connection;
  epResult_t (*proxyProgress)(struct epProxyState* proxyState, struct epProxyArgs*); // Copied from transport if necessary
};

#define NET_HANDLE_MAXSIZE 128
struct epConnector {
  int connected;
  struct epProxyConnector proxyConn;
  struct epTransportComm* transportComm;
  void* transportResources; // Host-side resources
  struct epConnInfo conn;
  struct epComm *comm;
};

struct epMesh {
  /* record peer ranks, the arrayhas been left shift by localRank
    * for example: assume nranks is 4
  * rank 0: [1 2 3] -> [1 2 3]
  * rank 1: [0 2 3] -> [2 3 0]
  * rank 2: [0 1 3] -> [3 0 1]
  * rank 3: [0 1 2] -> [0 1 2]
  */
  int peerRanks[EP_MAX_LOCAL_RANKS];
};

struct epDevMesh {
  int devPeerRanks[EP_MAX_LOCAL_RANKS];
};

struct epChannelPeer {
  struct epConnector send;
  struct epConnector recv;
};

struct epDevChannelPeer {
  // Stripped version of epChannelPeer where we only keep the epConnInfo
  // instead of the full epConnector.
  struct epConnInfo connInfo[MAX_SIDES];
};

//sizeof(struct epDevChannel)= 43432B
struct epDevChannel {
  // Used for gcu300
  struct epDevChannelPeer peers[EP_MAX_RANKS];

  // Dynamic allocation for gcu400 to support large scale of ranks
  struct epDevChannelPeer* peersP2PV4;
  // Access from device
  struct epDevChannelPeer* peersV4DevPtr;
  struct epDevChannelPeer* peersP2PV4DevPtr;

  struct epDevMesh mesh;

  /* stepPerSlice is used for gcu400 lare port master mode. primitive will use it to update remote's head/tail by atomic
   * operation instead of linear copy
   */
  uint64_t* stepPerSlice;
};

struct epChannelConfig {
  uint64_t* peerDirectAddrs[EP_MAX_LOCAL_RANKS];
  uintptr_t rasInfoPtr;
  // PeersInfo pointer points to the peersInfo array according to the algorithm.
  // For example, in ring algorithm, it points to the peersP2P array,
  // in mesh/tree algorithm, it points to the peers array.
  struct epDevChannelPeer *peersInfo;
};

struct epDevComm {
  int nChannels;
  int rank;
  int nRanks;
  int nLocalRanks;
  // int buffSizes[EP_NUM_PROTOCOLS]; DeepEP no need any buff for now

  // Flag to ask EP kernels to abort
  /* volatile */ uint32_t *abortFlag;

  // Channels, device side, but only access from host
  struct epDevChannel* channelsHostPtr;

  // Channels, device side, access from device
  struct epDevChannel* channels;

  // struct epDevChannel* currentChannel;
  uint8_t reserved[112];
};

//this is a empirical value that has best performance for every lare port sending.
#define BEST_DATASIZE_PER_LARE (16*1024)
struct epCommonConfigs {
  //global
  uint8_t useDirectMode;
  uint8_t workersPerChn;
  uint8_t peerNumPerWorker;
  uint8_t usedPortCount;    //real lare ports used for sending according to data size
  uint32_t isInplace : 1;
  uint32_t reserved : 31;
  int myRank;
  int algoRanks;            //nRanks for ring, nLocalRanks for mesh
  ssize_t stepCount;         //used for legacy fifo transferring
  uint64_t stepsPerSlice;    //L3 memory used for lare flag sending
};

#endif
