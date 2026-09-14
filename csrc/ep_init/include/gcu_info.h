/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_GCU_INFO_H_
#define EP_GCU_INFO_H_

#include "ep.h"
#include "efmlwrap.h"
#include "devcomm.h"

struct epGcuCapabity {
  bool supportLaunchKernelEx;
  int needThdsPerSubblk;
  int needSubthdsPerThd;
  int maxThdsPerBlk;
  int maxSubthdsPerThd;
  int maxSubblksPerBlk;
  int maxSearchRingChannels;
  int maxSearchTreeChannels;
  int meshThreshold;
  int maxBlks;
  int maxSteps;
  int allreduceSliceSteps;
  int allreduceChunkSteps;
  // int defaultBuffSize[EP_NUM_PROTOCOLS]; DeepEP no need any buff for now
  int platform;
  size_t buffAlignment;
  int portsPerPlane;
};

struct epGcuInfo {
  int efmlDevId = -1;
  efmlDeviceArchitecture_t efmlArch = EFML_DEVICE_ARCH_GCU300;
  char busIdStr[EFML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = "";
  // MNLARE support
  efmlGcuFabricInfoV_t fabricInfo;
};

epResult_t initGcuInfo(int topsDev, struct epGcuInfo & gcuInfo, struct epGcuCapabity & cap);
#endif

