/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_NET_DEVICE_H_
#define EP_NET_DEVICE_H_

#define EP_NET_DEVICE_INVALID_VERSION      0x0
#define EP_NET_MTU_SIZE                    4096

#define EP_NET_DEVICE_UNPACK_VERSION 0x0

typedef enum {EP_NET_DEVICE_INVALID=0} epNetDeviceType;

typedef struct {
  epNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  void* handle;
  size_t size;
  int needsProxyProgress;
} epNetDeviceHandle_t;

#endif
