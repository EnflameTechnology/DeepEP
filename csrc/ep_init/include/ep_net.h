/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_NET_H_
#define EP_NET_H_

#include "ep.h"
#include <stdint.h>
#include "debug.h"
#include "net_device.h"

#define EP_NET_HANDLE_MAXSIZE 128

#define EP_PTR_HOST 0x1
#define EP_PTR_TOPS 0x2
//DMABUF not support
#define EP_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define EP_NET_MAX_REQUESTS 32

typedef void (*epDebugLogger_t)(epDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

typedef struct {
  char* name;                      // Used mostly for logging.
  char* pciPath;                   // Path to the PCI device in /sys.
  uint64_t guid;                   // Unique identifier for the NIC chip. Important for
                                   // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport;                  // [EP_PTR_HOST|EP_PTR_TOPS|EP_PTR_DMABUF]
  int regIsGlobal;                 // regMr is not tied to a particular comm
  int speed;                       // Port speed in Mbps.
  int port;                        // Port number.
  float latency;                   // Network latency
  int maxComms;                    // Maximum number of comms we can create
  int maxRecvs;                    // Maximum number of grouped receives.

  //Network offload not support
  epNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload

} epNetProperties_t;


typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  epResult_t (*init)(epDebugLogger_t logFunction);
  // Return the number of adapters.
  epResult_t (*devices)(int* ndev);
  // Get various device properties.
  epResult_t (*getProperties)(int dev, epNetProperties_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to EP_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  epResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  // If *sendDevComm points to a valid object, then EP is requesting device offload for this connection
  epResult_t (*connect)(int dev, void* handle, void** sendComm, epNetDeviceHandle_t** sendDevComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  // If *recvDevComm points to a valid object, then EP is requesting device offload for this connection
  epResult_t (*accept)(void* listenComm, void** recvComm, epNetDeviceHandle_t** recvDevComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either EP_PTR_HOST or EP_PTR_TOPS.
  epResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  /* DMA-BUF reg, not support */
  epResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  epResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  epResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  epResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with EP_PTR_TOPS is
  // visible to the GCU
  epResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  epResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  epResult_t (*closeSend)(void* sendComm);
  epResult_t (*closeRecv)(void* recvComm);
  epResult_t (*closeListen)(void* listenComm);

  // Copy the given mhandle to a dptr in a format usable by this plugin's device code
  epResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);

  // Notify the plugin that a recv has completed by the device
  epResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
} epNet_t;


#endif // end include guard
