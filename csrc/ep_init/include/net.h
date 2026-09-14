/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_INT_NET_H_
#define EP_INT_NET_H_

#include "ep.h"
#include "ep_net.h"
#include "checks.h"

extern epNet_t* epNet;
typedef char epNetHandle_t[EP_NET_HANDLE_MAXSIZE];

epResult_t epNetInit();
epResult_t epNetFinalize(struct epComm* comm);
int epNetVersion();

// Translation to external API
__attribute__((unused)) static const char* epNetName() { return epNet->name; }
__attribute__((unused)) static epResult_t epNetDevices(int* ndev) { EP_CHECK(epNet->devices(ndev)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetGetProperties(int dev, epNetProperties_t* props) { EP_CHECK(epNet->getProperties(dev, props)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetListen(int dev, void* handle, void** listenComm) { EP_CHECK(epNet->listen(dev, handle, listenComm)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetConnect(int dev, void* handle, void** sendComm) { EP_CHECK(epNet->connect(dev, handle, sendComm, NULL)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetAccept(void* listenComm, void** recvComm) { EP_CHECK(epNet->accept(listenComm, recvComm, NULL)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetRegMr(void* comm, void* data, int size, int type, void** mhandle) { EP_CHECK(epNet->regMr(comm, data, size, type, mhandle)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetDeregMr(void* comm, void* mhandle) { EP_CHECK(epNet->deregMr(comm, mhandle)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) { EP_CHECK(epNet->isend(sendComm, data, size, tag, mhandle, request)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) { EP_CHECK(epNet->irecv(recvComm, n, data, sizes, tags, mhandles, request)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { EP_CHECK(epNet->iflush(recvComm, n, data, sizes, mhandles, request)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetTest(void* request, int* done, int* sizes) { EP_CHECK(epNet->test(request, done, sizes)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetCloseSend(void* sendComm) { EP_CHECK(epNet->closeSend(sendComm)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetCloseRecv(void* recvComm) { EP_CHECK(epNet->closeRecv(recvComm)); return epSuccess; }
__attribute__((unused)) static epResult_t epNetCloseListen(void* listenComm) { EP_CHECK(epNet->closeListen(listenComm)); return epSuccess; }

// Test whether the current GCU support GCU Direct RDMA.
epResult_t epGcuGdrSupport(int* gdrSupport);

extern epNet_t epNetIb;

#endif
