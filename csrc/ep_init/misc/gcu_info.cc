/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gcu_info.h"
#include "checks.h"
#include "alloc.h"
#include "ep.h"
// #include "collectives.h"
#include <string>

EP_PARAM(MNLareUUID, "MNLARE_UUID", -1);
EP_PARAM(MNLareCliqueId, "MNLARE_CLIQUE_ID", -1);
EP_PARAM(PortsPerPlane, "PORTS_PER_PLANE", 4);

static epResult_t initCapability(efmlDeviceArchitecture_t efmlArch,
    topsDeviceProp_t const& prop, struct epGcuCapabity & capability) {
  char* platformEnv = nullptr;
  capability.platform = PLATFORM_ASIC;
  if (efmlArch >= EFML_DEVICE_ARCH_GCU400) {
      capability.supportLaunchKernelEx = true;
      capability.needThdsPerSubblk = 1;
      capability.needSubthdsPerThd = 1;
      capability.maxThdsPerBlk = 1;
      capability.maxSubthdsPerThd = 8;
      /*For gcu 400 ring topo,
        expected channels to search will up to max 6 withnot duplicating
       *TODO: for all scale-up topo, there will be max 16 channel for ring.
        try later then
       */
      capability.maxSearchRingChannels = 8;
      capability.maxSearchTreeChannels = 8;
      capability.meshThreshold = 128*1024; //max mesh direct data size(Bytes)
      capability.maxBlks = 8; //assigned to default value but it's not used
      capability.maxSteps = 8;           // this value must equal with EP_STEPS defined in ep/src/collectives/device/gcu400/common_kernel.h
      capability.allreduceSliceSteps = 1; // this value must equal with ALLREDUCE_SLICESTEPS defined in ep/src/collectives/device/gcu400/common_kernel.h
      capability.allreduceChunkSteps = 2; // this value must equal with ALLREDUCE_CHUNKSTEPS defined in ep/src/collectives/device/gcu400/common_kernel.h
      // 1024 is a debug parameter, a better value may needed according to performance
      // DeepEP no need any buff for now
      // capability.defaultBuffSize[EP_PROTO_LL128] = 1024*capability.maxSteps*sizeof(uint32_t);
      // capability.defaultBuffSize[EP_PROTO_SIMPLE] = (1 << 23); /* 8MiB */
      // to avoid performance issue between lare and L3 RMW, buffer address has better be 128 aligned
      capability.buffAlignment = 128;
      capability.portsPerPlane = epParamPortsPerPlane();
      if (capability.portsPerPlane <= 0) {
        WARN("portsPerPlane %d should be greater than 0, set to default value 4", capability.portsPerPlane);
        capability.portsPerPlane = 4;
      }
      platformEnv = std::getenv("GTEST_PLATFORM");
      if (platformEnv != nullptr) {
        std::string name(platformEnv, platformEnv + strlen(platformEnv));
        capability.platform = std::stoul(name, nullptr, 16);
      }
    } else {
      WARN("Unrecognized  EFML_DEVICE_ARCH %d", efmlArch);
      return epInvalidArgument;
    }
  capability.maxSubblksPerBlk = (capability.maxThdsPerBlk*capability.maxSubthdsPerThd) /
                                            (capability.needThdsPerSubblk*capability.needSubthdsPerThd);

  (void) prop;

  return epSuccess;
}

epResult_t initGcuInfo(int topsDev, struct epGcuInfo & gcuInfo, struct epGcuCapabity & cap) {
  TOPS_CHECK(topsDeviceGetPCIBusId(gcuInfo.busIdStr, EFML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, topsDev));
  efmlDeviceArchitecture_t efmlArch = EFML_DEVICE_ARCH_GCU300;
  int index;
  topsDeviceProp_t prop;
  TOPS_CHECK(topsGetDeviceProperties(&prop, topsDev));
  int arch;
  sscanf(prop.gcuArchName, "%*[^0-9]%d", &arch);
  //gcu300 does not support efml at present.
  if (arch >= 400) {
    efmlDevice_t efmlDev;
    EP_CHECK(epEfmlDeviceGetHandleByPciBusId(gcuInfo.busIdStr, &efmlDev));
    unsigned efmlId;
    EP_CHECK(epEfmlDeviceGetIndex(efmlDev, &efmlId));
    EP_CHECK(epEfmlDeviceGetArchitecture(efmlDev, &efmlArch));
    index = efmlId;
    if (efmlArch >= EFML_DEVICE_ARCH_GCU400) {
      // MNLARE support
      gcuInfo.fabricInfo.state = EFML_GCU_FABRIC_STATE_NOT_SUPPORTED;
      EP_CHECK(epEfmlDeviceGetGcuFabricInfoV(efmlDev, &gcuInfo.fabricInfo));
      if (gcuInfo.fabricInfo.state != EFML_GCU_FABRIC_STATE_NOT_SUPPORTED) {
        if (epParamMNLareUUID() != -1) {
          union {
            long uuid;
            char uuid_bytes[sizeof(long)];
          } converter;
          converter.uuid = epParamMNLareUUID();
          memcpy(&gcuInfo.fabricInfo.clusterUuid[0], converter.uuid_bytes, sizeof(long));
          memcpy(&gcuInfo.fabricInfo.clusterUuid[1], converter.uuid_bytes, sizeof(long));
        }
        if (epParamMNLareCliqueId() != -1) gcuInfo.fabricInfo.cliqueId = epParamMNLareCliqueId();
        long uuid_part1, uuid_part2;
        memcpy(&uuid_part1, &gcuInfo.fabricInfo.clusterUuid[0], sizeof(long));
        memcpy(&uuid_part2, &gcuInfo.fabricInfo.clusterUuid[sizeof(long)], sizeof(long));
        INFO(EP_INIT, "MNLARE busIdStr %s fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x",
             gcuInfo.busIdStr, uuid_part1, uuid_part2,
             gcuInfo.fabricInfo.cliqueId, gcuInfo.fabricInfo.state, gcuInfo.fabricInfo.healthMask);
      }
    }
  } else {
    index = topsDev;
  }
  gcuInfo.efmlDevId = index;
  gcuInfo.efmlArch = efmlArch;
  EP_CHECK(initCapability(efmlArch, prop, cap));
  return epSuccess;
}

