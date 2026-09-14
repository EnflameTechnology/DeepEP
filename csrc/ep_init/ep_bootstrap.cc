/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ep.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include <unistd.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include "socket.h"
// #include "proxy.h"
#include "param.h"

struct bootstrapRootArgs {
  struct epSocket* listenSock;
  uint64_t magic;
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];
static union epSocketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

epResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetInitDone == 0) {
      const char* env = epGetEnv("EP_COMM_ID");
      if (env) {
        union epSocketAddress remoteAddr;
        if (epSocketGetAddrFromString(&remoteAddr, env) != epSuccess) {
          WARN("Invalid EP_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          pthread_mutex_unlock(&bootstrapNetLock);
          return epInvalidArgument;
        }
        if (epFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          pthread_mutex_unlock(&bootstrapNetLock);
          return epSystemError;
        }
      } else {
        int nIfs = epFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          pthread_mutex_unlock(&bootstrapNetLock);
          return epInternalError;
        }
      }
      char line[SOCKET_NAME_MAXLEN+MAX_IF_NAME_SIZE+2];
      sprintf(line, " %s:", bootstrapNetIfName);
      epSocketToString(&bootstrapNetIfAddr, line+strlen(line));
      INFO(EP_INIT, "Bootstrap : Using%s", line);
      bootstrapNetInitDone = 1;
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return epSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

// Additional sync functions
static epResult_t bootstrapNetSend(struct epSocket* sock, void* data, int size) {
  EP_CHECK(epSocketSend(sock, &size, sizeof(int)));
  EP_CHECK(epSocketSend(sock, data, size));
  return epSuccess;
}
static epResult_t bootstrapNetRecv(struct epSocket* sock, void* data, int size) {
  int recvSize;
  EP_CHECK(epSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return epInternalError;
  }
  EP_CHECK(epSocketRecv(sock, data, std::min(recvSize, size)));
  return epSuccess;
}
static epResult_t bootstrapNetSendRecv(struct epSocket* sendSock, void* sendData, int sendSize, struct epSocket* recvSock, void* recvData, int recvSize) {
  int senderRecvSize;
  EP_CHECK(epSocketSendRecv(sendSock, &sendSize, sizeof(int), recvSock, &senderRecvSize, sizeof(int)));
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
    return epInternalError;
  }
  EP_CHECK(epSocketSendRecv(sendSock, sendData, sendSize, recvSock, recvData, recvSize));
  return epSuccess;
}

struct extInfo {
  int rank;
  int nranks;
  union epSocketAddress extAddressListenRoot;
  union epSocketAddress extAddressListen;
};

#include <sys/resource.h>

static epResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return epSuccess;
}

static void *bootstrapRoot(void* rargs) {
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  struct epSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  epResult_t res = epSuccess;
  int nranks = 0, c = 0;
  struct extInfo info;
  union epSocketAddress *rankAddresses = NULL;
  union epSocketAddress *rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  union epSocketAddress *zero = NULL;
  EP_CHECKGOTO(epCalloc(&zero, 1), res, out);
  setFilesLimit();

  TRACE(EP_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    struct epSocket sock;
    EP_CHECKGOTO(epSocketInit(&sock), res, out);
    EP_CHECKGOTO(epSocketAccept(&sock, listenSock), res, out);
    EP_CHECKGOTO(bootstrapNetRecv(&sock, &info, sizeof(info)), res, out);
    EP_CHECKGOTO(epSocketClose(&sock), res, out);

    if (c == 0) {
      nranks = info.nranks;
      EP_CHECKGOTO(epCalloc(&rankAddresses, nranks), res, out);
      EP_CHECKGOTO(epCalloc(&rankAddressesRoot, nranks), res, out);
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(zero, &rankAddressesRoot[info.rank], sizeof(union epSocketAddress)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for that rank
    memcpy(rankAddressesRoot+info.rank, &info.extAddressListenRoot, sizeof(union epSocketAddress));
    memcpy(rankAddresses+info.rank, &info.extAddressListen, sizeof(union epSocketAddress));

    ++c;
    TRACE(EP_INIT, "Received connect from rank %d total %d/%d",  info.rank, c, nranks);
  } while (c < nranks);
  TRACE(EP_INIT, "COLLECTED ALL %d HANDLES", nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r=0; r<nranks; ++r) {
    int next = (r+1) % nranks;
    struct epSocket sock;
    EP_CHECKGOTO(epSocketInit(&sock, rankAddressesRoot+r, magic, epSocketTypeBootstrap), res, out);
    EP_CHECKGOTO(epSocketConnect(&sock), res, out);
    EP_CHECKGOTO(bootstrapNetSend(&sock, rankAddresses+next, sizeof(union epSocketAddress)), res, out);
    EP_CHECKGOTO(epSocketClose(&sock), res, out);
  }
  TRACE(EP_INIT, "SENT OUT ALL %d HANDLES", nranks);

out:
  if (listenSock != NULL) {
    epSocketClose(listenSock);
    free(listenSock);
  }
  if (rankAddresses) free(rankAddresses);
  if (rankAddressesRoot) free(rankAddressesRoot);
  if (zero) free(zero);
  free(rargs);

  TRACE(EP_INIT, "DONE");
  return NULL;
}

epResult_t bootstrapCreateRoot(struct epBootstrapHandle* handle, bool /*idFromEnv*/) {
  struct epSocket* listenSock;
  struct bootstrapRootArgs* args;
  pthread_t thread;

  EP_CHECK(epCalloc(&listenSock, 1));
  EP_CHECK(epSocketInit(listenSock, &handle->addr, handle->magic, epSocketTypeBootstrap, NULL, 0));
  EP_CHECK(epSocketListen(listenSock));
  EP_CHECK(epSocketGetAddr(listenSock, &handle->addr));

  EP_CHECK(epCalloc(&args, 1));
  args->listenSock = listenSock;
  args->magic = handle->magic;
  NEQCHECK(pthread_create(&thread, NULL, bootstrapRoot, (void*)args), 0);
  epSetThreadName(thread, "EP BootstrapR");
  NEQCHECK(pthread_detach(thread), 0); // will not be pthread_join()'d
  return epSuccess;
}

epResult_t bootstrapGetUniqueId(struct epBootstrapHandle* handle) {
  memset(handle, 0, sizeof(epBootstrapHandle));
  EP_CHECK(getRandomData(&handle->magic, sizeof(handle->magic)));

  const char* env = epGetEnv("EP_COMM_ID");
  if (env) {
    INFO(EP_ENV, "EP_COMM_ID set by environment to %s", env);
    if (epSocketGetAddrFromString(&handle->addr, env) != epSuccess) {
      WARN("Invalid EP_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return epInvalidArgument;
    }
  } else {
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union epSocketAddress));
    EP_CHECK(bootstrapCreateRoot(handle, false));
  }

  return epSuccess;
}

struct unexConn {
  int peer;
  int tag;
  struct epSocket sock;
  struct unexConn* next;
};

struct bootstrapState {
  struct epSocket listenSock;
  struct epSocket ringRecvSocket;
  struct epSocket ringSendSocket;
  union epSocketAddress* peerCommAddresses;
  union epSocketAddress* peerProxyAddresses;
  union epSocketAddress* peerIfAddresses;
  uint64_t* peerProxyAddressesUDS;
  struct unexConn* unexpectedConnections;
  int topsDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t *abortFlag;
};

epResult_t bootstrapInit(struct epBootstrapHandle* handle, struct epComm* comm) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  struct bootstrapState* state;
  struct epSocket* proxySocket;
  epSocketAddress nextAddr;
  struct epSocket sock, listenSockRoot;
  struct extInfo info;
  memset(&info, 0, sizeof(struct extInfo));

  EP_CHECK(epCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->abortFlag = comm->abortFlag;
  comm->bootstrap = state;
  comm->magic = state->magic = handle->magic;

  TRACE(EP_INIT, "rank %d nranks %d", rank, nranks);

  info.rank = rank;
  info.nranks = nranks;
  // Create socket for other ranks to contact me
  EP_CHECK(epSocketInit(&state->listenSock, &bootstrapNetIfAddr, comm->magic, epSocketTypeBootstrap, comm->abortFlag));
  EP_CHECK(epSocketListen(&state->listenSock));
  EP_CHECK(epSocketGetAddr(&state->listenSock, &info.extAddressListen));

  // Create socket for root to contact me
  EP_CHECK(epSocketInit(&listenSockRoot, &bootstrapNetIfAddr, comm->magic, epSocketTypeBootstrap, comm->abortFlag));
  EP_CHECK(epSocketListen(&listenSockRoot));
  EP_CHECK(epSocketGetAddr(&listenSockRoot, &info.extAddressListenRoot));

  // stagger connection times to avoid an overload of the root
  if (nranks > 128) {
    long msec = rank;
    struct timespec tv;
    tv.tv_sec = msec / 1000;
    tv.tv_nsec = 1000000 * (msec % 1000);
    TRACE(EP_INIT, "rank %d delaying connection to root by %ld msec", rank, msec);
    (void) nanosleep(&tv, NULL);
  }

  // send info on my listening socket to root
  EP_CHECK(epSocketInit(&sock, &handle->addr, comm->magic, epSocketTypeBootstrap, comm->abortFlag));
  EP_CHECK(epSocketConnect(&sock));
  EP_CHECK(bootstrapNetSend(&sock, &info, sizeof(info)));
  EP_CHECK(epSocketClose(&sock));

  // get info on my "next" rank in the bootstrap ring from root
  EP_CHECK(epSocketInit(&sock));
  EP_CHECK(epSocketAccept(&sock, &listenSockRoot));
  EP_CHECK(bootstrapNetRecv(&sock, &nextAddr, sizeof(union epSocketAddress)));
  EP_CHECK(epSocketClose(&sock));
  EP_CHECK(epSocketClose(&listenSockRoot));

  EP_CHECK(epSocketInit(&state->ringSendSocket, &nextAddr, comm->magic, epSocketTypeBootstrap, comm->abortFlag));
  EP_CHECK(epSocketConnect(&state->ringSendSocket));
  // Accept the connect request from the previous rank in the AllGather ring
  EP_CHECK(epSocketInit(&state->ringRecvSocket));
  EP_CHECK(epSocketAccept(&state->ringRecvSocket, &state->listenSock));

  // AllGather all listen handlers
  EP_CHECK(epCalloc(&state->peerCommAddresses, nranks));
  EP_CHECK(epSocketGetAddr(&state->listenSock, state->peerCommAddresses+rank));
  EP_CHECK(bootstrapAllGather(state, state->peerCommAddresses, sizeof(union epSocketAddress)));

  // Create the service proxy
  EP_CHECK(epCalloc(&state->peerProxyAddresses, nranks));
  EP_CHECK(epCalloc(&state->peerProxyAddressesUDS, nranks));

  EP_CHECK(epCalloc(&comm->bootstrapIfAddresses, nranks));
  memcpy(comm->bootstrapIfAddresses, state->peerCommAddresses, nranks * sizeof(union epSocketAddress));

  // proxy is aborted through a message; don't set abortFlag
  EP_CHECK(epCalloc(&proxySocket, 1));
  EP_CHECK(epSocketInit(proxySocket, &bootstrapNetIfAddr, comm->magic, epSocketTypeProxy, comm->abortFlag));
  EP_CHECK(epSocketListen(proxySocket));
  EP_CHECK(epSocketGetAddr(proxySocket, state->peerProxyAddresses+rank));
  EP_CHECK(bootstrapAllGather(state, state->peerProxyAddresses, sizeof(union epSocketAddress)));
  // cuMem UDS support
  // Make sure we create a unique UDS socket name
  uint64_t randId;
  EP_CHECK(getRandomData(&randId, sizeof(randId)));
  state->peerProxyAddressesUDS[rank] = getPidHash()+randId;
  EP_CHECK(bootstrapAllGather(state, state->peerProxyAddressesUDS, sizeof(*state->peerProxyAddressesUDS)));
  // EP_CHECK(epProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS));

  // AllGather all interfaces address
  EP_CHECK(epCalloc(&state->peerIfAddresses, nranks));
  struct ifaddrs *ifList;
  struct ifaddrs *currentIf;
  getifaddrs(&ifList);
  for (currentIf = ifList; currentIf != NULL; currentIf = currentIf->ifa_next) {
    if (currentIf->ifa_addr == NULL) continue;
    int family = currentIf->ifa_addr->sa_family;
    if (family != AF_INET) continue;

    const char *affinityIfNameEnv = epGetEnv("EP_TOPO_AFFINITY_IFNAME");
    if (affinityIfNameEnv && strlen(affinityIfNameEnv) > 0 ) INFO(EP_BOOTSTRAP|EP_ENV, ENV_FORMAT_STR, "EP_TOPO_AFFINITY_IFNAME", affinityIfNameEnv);
    if (!affinityIfNameEnv || strlen(affinityIfNameEnv) == 0) affinityIfNameEnv = "bond2";
    if (strcmp(currentIf->ifa_name, affinityIfNameEnv) == 0) {
      memcpy(state->peerIfAddresses + rank, (union epSocketAddress *) currentIf->ifa_addr, sizeof(union epSocketAddress));
    }
  }
  freeifaddrs(ifList);
  // Perform AllGather to collect interface addresses from all ranks
  EP_CHECK(bootstrapAllGather(state, state->peerIfAddresses, sizeof(union epSocketAddress)));
  comm->peerIfAddresses = state->peerIfAddresses;

  TRACE(EP_INIT, "rank %d nranks %d - DONE", rank, nranks);

  return epSuccess;
}

// Bootstrap send/receive functions
//
// We do not keep connections opened with all ranks at all times, and we have no guarantee
// that connections to our unique listen socket will arrive in the same order as we need
// them. Therefore, when establishing a connection, the sender sends a (peer, tag) tuple to
// allow the receiver to identify the flow, and keep it in an unexpected queue if needed.

epResult_t bootstrapConnect(void* commState, int peer, int tag, struct epSocket* sock) {
  epResult_t ret = epSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  EP_CHECKGOTO(epSocketInit(sock, state->peerCommAddresses+peer, state->magic, epSocketTypeBootstrap), ret, fail);
  EP_CHECKGOTO(epSocketConnect(sock), ret, fail);
  EP_CHECKGOTO(bootstrapNetSend(sock, &state->rank, sizeof(int)), ret, fail);
  EP_CHECKGOTO(bootstrapNetSend(sock, &tag, sizeof(int)), ret, fail);
  return epSuccess;
fail:
  EP_CHECK(epSocketClose(sock));
  return ret;
}

epResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  epResult_t ret = epSuccess;
  struct epSocket sock;

  TRACE(EP_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  EP_CHECK(bootstrapConnect(commState, peer, tag, &sock));
  EP_CHECKGOTO(bootstrapNetSend(&sock, data, size), ret, exit);

  TRACE(EP_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);

exit:
  EP_CHECK(epSocketClose(&sock));
  return ret;
}

epResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct epSocket* sock) {
  // New unex
  struct unexConn* unex;
  EP_CHECK(epCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct epSocket));

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return epSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return epSuccess;
}

epResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct epSocket* sock, int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct epSocket));
      free(elem);
      *found = 1;
      return epSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return epSuccess;
}

static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at once
epResult_t bootstrapAccept(void* commState, int peer, int tag, struct epSocket* sock) {
  epResult_t ret = epSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int newPeer, newTag;

  // Search unexpected connections first
  int found;
  EP_CHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found) return epSuccess;

  // Then look for new connections
  while (1) {
    EP_CHECKGOTO(epSocketInit(sock), ret, fail);
    EP_CHECKGOTO(epSocketAccept(sock, &state->listenSock), ret, fail);
    EP_CHECKGOTO(bootstrapNetRecv(sock, &newPeer, sizeof(int)), ret, fail);
    EP_CHECKGOTO(bootstrapNetRecv(sock, &newTag, sizeof(int)), ret, fail);
    if (newPeer == peer && newTag == tag) return epSuccess;
    EP_CHECKGOTO(unexpectedEnqueue(state, newPeer, newTag, sock), ret, fail);
  }
  return epSuccess;
fail:
  EP_CHECK(epSocketClose(sock));
  return ret;
}

// We can't know who we'll receive from, so we need to receive everything at once
epResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  epResult_t ret;
  struct epSocket sock;
  EP_CHECK(bootstrapAccept(commState, peer, tag, &sock));
  TRACE(EP_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  EP_CHECKGOTO(bootstrapNetRecv(&sock, ((char*)data), size), ret, exit);
exit:
  EP_CHECK(epSocketClose(&sock));
  return ret;
}

// Collective algorithms, based on bootstrapSend/Recv, and sometimes bootstrapConnect/Accept

epResult_t bootstrapRingAllGather(struct epSocket* prevSocket, struct epSocket* nextSocket, int rank, int nranks, char* data, int size) {
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  for (int i=0; i<nranks-1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right, recv slice from the left
    EP_CHECK(bootstrapNetSendRecv(nextSocket, data+sslice*size, size, prevSocket, data+rslice*size, size));
  }
  return epSuccess;
}
epResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(EP_INIT, "rank %d nranks %d size %d", rank, nranks, size);

  EP_CHECK(bootstrapRingAllGather(&state->ringRecvSocket, &state->ringSendSocket, rank, nranks, (char*)allData, size));

  TRACE(EP_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return epSuccess;
}

epResult_t bootstrapTryAllGather(void* commState, void* allData, int size, int *closed) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(EP_INIT, "rank %d nranks %d size %d", rank, nranks, size);

  //bootstrapRingAllGather(&state->ringRecvSocket, &state->ringSendSocket, rank, nranks, (char*)allData, size));
  //bootstrapRingAllGather(struct epSocket* prevSocket, struct epSocket* nextSocket, int rank, int nranks, char* data, int size)
  char* data = (char*)allData;
  int sendSize = size;
  int recvSize = size;
  *closed = 0;
  for (int i=0; i<nranks-1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right, recv slice from the left
    //bootstrapNetSendRecv(nextSocket, data+sslice*size, size, prevSocket, data+rslice*size, size));
    //static epResult_t bootstrapNetSendRecv(struct epSocket* sendSock, void* sendData, int sendSize, struct epSocket* recvSock, void* recvData, int recvSize)
    struct epSocket* recvSock = &state->ringRecvSocket; // prevSocket
    struct epSocket* sendSock = &state->ringSendSocket; // nextSocket

    void* sendData = data+sslice*size;
    void* recvData = data+rslice*size;

    int senderRecvSize;
    EP_CHECK(epSocketTrySendRecv(sendSock, &sendSize, sizeof(int), recvSock, &senderRecvSize, sizeof(int), closed));
    if (*closed) return epSuccess;
    if (senderRecvSize > recvSize) {
      WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
      return epInternalError;
    }
    EP_CHECK(epSocketTrySendRecv(sendSock, sendData, sendSize, recvSock, recvData, recvSize, closed));
    if (*closed) return epSuccess;
  }

  TRACE(EP_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);

  return epSuccess;
}

epResult_t bootstrapIntraNodeBarrier(void* commState, int *ranks, int rank, int nranks, int tag) {
  if (nranks == 1) return epSuccess;
  TRACE(EP_INIT, "rank %d nranks %d tag %x - ENTER", rank, nranks, tag);

  /* Simple [intra] process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1];
  for (int mask=1; mask<nranks; mask<<=1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    EP_CHECK(bootstrapSend(commState, ranks ? ranks[dst] : dst, tag, data, sizeof(data)));
    EP_CHECK(bootstrapRecv(commState, ranks ? ranks[src] : src, tag, data, sizeof(data)));
  }

  TRACE(EP_INIT, "rank %d nranks %d tag %x - DONE", rank, nranks, tag);
  return epSuccess;
}

// epResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag) {
//   return bootstrapIntraNodeBarrier(commState, NULL, rank, nranks, tag);
// }

epResult_t bootstrapBarrier(void* commState, uint64_t barrier_value) {
  struct bootstrapState* state = (struct bootstrapState*)commState;

  int rank = state->rank;
  int nranks = state->nranks;

  uint64_t recv_data;

  if (nranks == 1) return epSuccess;

  if (rank == 0) { // root
    // recv form the others ranks
    for (int i=1; i<nranks; i++) {
      EP_CHECK(bootstrapRecv(commState, i, 0xdf, &recv_data, sizeof(uint64_t)));
      if (recv_data != barrier_value) {
        WARN("bootstrapBarrier: root recv barrier value(%lu) is wrong", recv_data);
        return epSystemError;
      }
    }

    // send to the other ranks
    for (int i=1; i<nranks; i++) {
      EP_CHECK(bootstrapSend(commState, i, 0xdf, &barrier_value, sizeof(uint64_t)));
    }

  } else {
    // Send to the root rank
    EP_CHECK(bootstrapSend(commState, 0, 0xdf, &barrier_value, sizeof(uint64_t)));

    // Recv from root rank
    EP_CHECK(bootstrapRecv(commState, 0, 0xdf, &recv_data, sizeof(uint64_t)));
    if (recv_data != barrier_value) {
      WARN("bootstrapBarrier: non-root rank recv ack barrier value(%lu) is wrong", recv_data);
      return epSystemError;
    }
  }

  return epSuccess;
}

epResult_t bootstrapIntraNodeAllGather(void* commState, int *ranks, int rank, int nranks, void* allData, int size) {
  if (nranks == 1) return epSuccess;
  TRACE(EP_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  int prevRank = ranks[(rank - 1 + nranks)%nranks];
  int nextRank = ranks[(rank + 1) % nranks];
  struct epSocket prevSocket, nextSocket;
  EP_CHECK(bootstrapConnect(commState, nextRank, 0, &nextSocket));
  EP_CHECK(bootstrapAccept(commState, prevRank, 0, &prevSocket));

  EP_CHECK(bootstrapRingAllGather(&prevSocket, &nextSocket, rank, nranks, (char*)allData, size));

  EP_CHECK(epSocketClose(&nextSocket));
  EP_CHECK(epSocketClose(&prevSocket));

  TRACE(EP_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return epSuccess;
}

// [IntraNode] in-place Broadcast
epResult_t bootstrapIntraNodeBroadcast(void* commState, int *ranks, int rank, int nranks, int root, void* bcastData, int size) {
  if (nranks == 1) return epSuccess;
  TRACE(EP_INIT, "rank %d nranks %d root %d size %d - ENTER", rank, nranks, root, size);

  if (rank == root) {
    for (int i=0; i<nranks; i++) {
      if (i != root) EP_CHECK(bootstrapSend(commState, ranks ? ranks[i] : i, /*tag=*/ranks ? ranks[i] : i, bcastData, size));
    }
  }
  else {
    EP_CHECK(bootstrapRecv(commState, ranks ? ranks[root] : root, /*tag=*/ranks ? ranks[rank] : rank, bcastData, size));
  }

  TRACE(EP_INIT, "rank %d nranks %d root %d size %d - DONE", rank, nranks, root, size);
  return epSuccess;
}

epResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size) {
  return bootstrapIntraNodeBroadcast(commState, NULL, rank, nranks, root, bcastData, size);
}

epResult_t bootstrapClose(void* commState) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (__atomic_load_n(state->abortFlag, __ATOMIC_RELAXED) == 0) {
      WARN("Unexpected connections are not empty");
      return epInternalError;
    }
  }

  EP_CHECK(epSocketClose(&state->listenSock));
  EP_CHECK(epSocketClose(&state->ringSendSocket));
  EP_CHECK(epSocketClose(&state->ringRecvSocket));

  free(state->peerCommAddresses);
  free(state->peerIfAddresses);
  free(state);

  return epSuccess;
}

epResult_t bootstrapAbort(void* commState) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  if (commState == NULL) return epSuccess;
  EP_CHECK(epSocketClose(&state->listenSock));
  EP_CHECK(epSocketClose(&state->ringSendSocket));
  EP_CHECK(epSocketClose(&state->ringRecvSocket));
  free(state->peerCommAddresses);
  free(state->peerIfAddresses);
//  for proxy, not support now
//  free(state->peerProxyAddresses);
//  free(state->peerProxyAddressesUDS);
  free(state);
  return epSuccess;
}
