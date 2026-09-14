/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_P2P_H_
#define EP_P2P_H_

typedef union {
  topsIpcMemHandle_t devIpc;
} epIpcDesc;

epResult_t epP2pAllocateShareableBuffer(size_t size, epIpcDesc *ipcDesc, void **ptr);
epResult_t epP2pFreeShareableBuffer(epIpcDesc *ipcDesc);
epResult_t epP2pImportShareableBuffer(struct epComm *comm, int peer, size_t size, epIpcDesc *ipcDesc, void **devMemPtr);

#endif
