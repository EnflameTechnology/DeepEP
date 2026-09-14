/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_BOOTSTRAP_H_
#define EP_BOOTSTRAP_H_

#include "ep.h"
#include "comm.h"

struct epBootstrapHandle {
  uint64_t magic;
  union epSocketAddress addr;
};
static_assert(sizeof(struct epBootstrapHandle) <= sizeof(epUniqueId), "Bootstrap handle is too large to fit inside EP unique ID");

epResult_t bootstrapNetInit();
epResult_t bootstrapCreateRoot(struct epBootstrapHandle* handle, bool idFromEnv);
epResult_t bootstrapGetUniqueId(struct epBootstrapHandle* handle);
epResult_t bootstrapInit(struct epBootstrapHandle* handle, struct epComm* comm);
epResult_t bootstrapSplit(struct epBootstrapHandle* handle, struct epComm* comm, struct epComm* parent, int color, int key, int* parentRanks);
epResult_t bootstrapAllGather(void* commState, void* allData, int size);
epResult_t bootstrapTryAllGather(void* commState, void* allData, int size, int *closed);
epResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);
epResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);
// epResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);
epResult_t bootstrapBarrier(void* commState, uint64_t barrier_value);
epResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);
epResult_t bootstrapIntraNodeBarrier(void* commState, int *ranks, int rank, int nranks, int tag);
epResult_t bootstrapIntraNodeAllGather(void* commState, int *ranks, int rank, int nranks, void* allData, int size);
epResult_t bootstrapIntraNodeBroadcast(void* commState, int *ranks, int rank, int nranks, int root, void* bcastData, int size);
epResult_t bootstrapClose(void* commState);
epResult_t bootstrapAbort(void* commState);
#endif
