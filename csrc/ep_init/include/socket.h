/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_SOCKET_H_
#define EP_SOCKET_H_

#include "ep.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT            1000 // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES   2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES    3 // connection timed out retry times (each one can take 20s)
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)
#define EP_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union epSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum epSocketState {
  epSocketStateNone = 0,
  epSocketStateInitialized = 1,
  epSocketStateAccepting = 2,
  epSocketStateAccepted = 3,
  epSocketStateConnecting = 4,
  epSocketStateConnectPolling = 5,
  epSocketStateConnected = 6,
  epSocketStateReady = 7,
  epSocketStateClosed = 8,
  epSocketStateError = 9,
  epSocketStateNum = 10
};

enum epSocketType {
  epSocketTypeUnknown = 0,
  epSocketTypeBootstrap = 1,
  epSocketTypeProxy = 2,
  epSocketTypeNetSocket = 3,
  epSocketTypeNetIb = 4
};

struct epSocket {
  int fd;
  int acceptFd;
  int timedOutRetries;
  int refusedRetries;
  union epSocketAddress addr;
  volatile uint32_t* abortFlag;
  int asyncFlag;
  enum epSocketState state;
  int salen;
  uint64_t magic;
  enum epSocketType type;
};

const char *epSocketToString(union epSocketAddress *addr, char *buf, const int numericHostForm = 1);
epResult_t epSocketGetAddrFromString(union epSocketAddress* ua, const char* ip_port_pair);
int epFindInterfaceMatchSubnet(char* ifNames, union epSocketAddress* localAddrs, union epSocketAddress* remoteAddr, int ifNameMaxSize, int maxIfs);
int epFindInterfaces(char* ifNames, union epSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs);

// Initialize a socket
epResult_t epSocketInit(struct epSocket* sock, union epSocketAddress* addr = NULL, uint64_t magic = EP_SOCKET_MAGIC, enum epSocketType type = epSocketTypeUnknown, volatile uint32_t* abortFlag = NULL, int asyncFlag = 0);
// Create a listening socket. sock->addr can be pre-filled with IP & port info. sock->fd is set after a successful call
epResult_t epSocketListen(struct epSocket* sock);
epResult_t epSocketGetAddr(struct epSocket* sock, union epSocketAddress* addr);
// Connect to sock->addr. sock->fd is set after a successful call.
epResult_t epSocketConnect(struct epSocket* sock);
// Return socket connection state.
epResult_t epSocketReady(struct epSocket* sock, int *running);
// Accept an incoming connection from listenSock->fd and keep the file descriptor in sock->fd, with the remote side IP/port in sock->addr.
epResult_t epSocketAccept(struct epSocket* sock, struct epSocket* ulistenSock);
epResult_t epSocketGetFd(struct epSocket* sock, int* fd);
epResult_t epSocketSetFd(int fd, struct epSocket* sock);

#define EP_SOCKET_SEND 0
#define EP_SOCKET_RECV 1

epResult_t epSocketProgress(int op, struct epSocket* sock, void* ptr, int size, int* offset);
epResult_t epSocketWait(int op, struct epSocket* sock, void* ptr, int size, int* offset);
epResult_t epSocketSend(struct epSocket* sock, void* ptr, int size);
epResult_t epSocketRecv(struct epSocket* sock, void* ptr, int size);
epResult_t epSocketSendRecv(struct epSocket* sendSock, void* sendPtr, int sendSize, struct epSocket* recvSock, void* recvPtr, int recvSize);
epResult_t epSocketTrySendRecv(struct epSocket* sendSock, void* sendPtr, int sendSize, struct epSocket* recvSock, void* recvPtr, int recvSize, int *closed);
epResult_t epSocketTryRecv(struct epSocket* sock, void* ptr, int size, int* closed, bool blocking);
epResult_t epSocketClose(struct epSocket* sock);
#endif
