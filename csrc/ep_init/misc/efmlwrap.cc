/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <initializer_list>
#include <memory>
#include <mutex>
#include <sys/ioctl.h>
#include <unistd.h>
#include "efmlwrap.h"
#include "checks.h"
#include "debug.h"
#include <ctype.h>

int epEfmlDeviceCount = 0;
epEfmlDeviceInfo epEfmlDevices[epEfmlMaxDevices];

#if EP_EFML_DIRECT
  #define EP_EFML_FN(name, rettype, arglist) constexpr rettype(*pfn_##name)arglist = name;
#else
  #include <dlfcn.h>
  #define EP_EFML_FN(name, rettype, arglist) rettype(*pfn_##name)arglist = nullptr;
#endif
#define EP_EFML_CB_ADDR(name) ((void**)&pfn_##name)
#define EP_EFML_API_CB(name) (pfn_##name)
#define EFML_API_DESC(name) ((char *)#name)

#define EFMLCHECK(name, ...) do { \
  efmlReturn_v1_t efmlErrStatus = EP_EFML_API_CB(name)(__VA_ARGS__); \
  if (efmlErrStatus != EFML_SUCCESS_V2) { \
    WARN(#name "() failed: %s", EP_EFML_API_CB(efmlErrorString)(efmlErrStatus)); \
    return epSystemError; \
  } \
} while(0)

#define EFMLTRY(name, ...) do { \
  if (!EP_EFML_DIRECT && EP_EFML_API_CB(name) == nullptr) {\
    WARN("pfn_%s() is nullptr", EFML_API_DESC(name));\
    return epInternalError; /* missing symbol is not a warned error */ \
  }\
  efmlReturn_v1_t efmlErrStatus = EP_EFML_API_CB(name)(__VA_ARGS__); \
  if (efmlErrStatus != EFML_SUCCESS_V2) { \
    if (efmlErrStatus != EFML_ERROR_NOT_SUPPORTED_V2) \
      INFO(EP_INIT, #name "() failed: %s", EP_EFML_API_CB(efmlErrorString)(efmlErrStatus)); \
    return epSystemError; \
  } \
} while(0)

namespace {
  //for gcu400
  EP_EFML_FN(efmlInit_v2, efmlReturn_v1_t, ())
  EP_EFML_FN(efmlShutdown, efmlReturn_v1_t, ())
  EP_EFML_FN(efmlDeviceGetCount_v2, efmlReturn_v1_t, (uint32_t*))
  EP_EFML_FN(efmlDeviceGetHandleByPciBusId_v2, efmlReturn_v1_t, (const char* pciBusId, efmlDevice_t* device))
  EP_EFML_FN(efmlDeviceGetHandleByIndex_v2, efmlReturn_v1_t, (unsigned int index, efmlDevice_t *device))
  EP_EFML_FN(efmlErrorString, char const*, (efmlReturn_v1_t r))
  EP_EFML_FN(efmlDeviceGetGcuLareRemotePciInfo_v2, efmlReturn_v1_t, (efmlDevice_t device, unsigned int link, efmlPciInfo_t *pci))
  EP_EFML_FN(efmlDeviceGetArchitecture, efmlReturn_v1_t, (efmlDevice_t device, efmlDeviceArchitecture_t* arch))
  EP_EFML_FN(efmlDeviceGetGcuLareState, efmlReturn_v1_t, (efmlDevice_t device, unsigned int link, efmlEnableState_t *isActive))
  EP_EFML_FN(efmlDeviceGetGcuLareCapability, efmlReturn_v1_t, (efmlDevice_t device, unsigned int port, efmlGcuLareCapability_t capability, unsigned int *capResult))
  EP_EFML_FN(efmlDeviceGetGcuLareRemoteDeviceType, efmlReturn_v1_t, (efmlDevice_t device, unsigned int port, efmlIntGcuLareDeviceType_t* deviceType))
  EP_EFML_FN(efmlDeviceGetFieldValues, efmlReturn_v1_t, (efmlDevice_t device, int valuesCount, efmlFieldValue_t *values))
  // MNLARE support
  EP_EFML_FN(efmlDeviceGetGcuFabricInfoV, efmlReturn_v1_t, (efmlDevice_t device, efmlGcuFabricInfoV_t *gcuFabricInfo))

  std::mutex lock; // EFML has had some thread safety bugs
  bool initialized = false;
  thread_local bool threadInitialized = false;
  epResult_t initResult = epSuccess;
}

epResult_t epEfmlEnsureInitialized() {
  // Optimization to avoid repeatedly grabbing the lock when we only want to
  // read from the global tables.
  if (threadInitialized) return initResult;
  threadInitialized = true;

  std::lock_guard<std::mutex> locked(lock);

  if (initialized) return initResult;
  initialized = true;

  #if !EP_EFML_DIRECT
  if (EP_EFML_API_CB(efmlInit_v2) == nullptr) {
    void *libhandle = dlopen("libefml.so", RTLD_NOW);
    if (libhandle == nullptr) {
      WARN("Failed to open libefml.so");
      threadInitialized = false;
      initialized = false;
      initResult = epSystemError;
      return initResult;
    }

    struct Symbol { void **ppfn; char const *name; char const *old_name;};
    std::initializer_list<Symbol> symbols = {
      //for gcu400
      {EP_EFML_CB_ADDR(efmlInit_v2), EFML_API_DESC(efmlInit_v2), EFML_API_DESC(efmlInit)},
      {EP_EFML_CB_ADDR(efmlShutdown), EFML_API_DESC(efmlShutdown), EFML_API_DESC(efmlShutdown)},
      {EP_EFML_CB_ADDR(efmlDeviceGetCount_v2), EFML_API_DESC(efmlDeviceGetCount_v2), EFML_API_DESC(efmlDeviceGetCount)},
      {EP_EFML_CB_ADDR(efmlDeviceGetHandleByPciBusId_v2), EFML_API_DESC(efmlDeviceGetHandleByPciBusId_v2), EFML_API_DESC(efmlDeviceGetHandleByPciBusId)},
      {EP_EFML_CB_ADDR(efmlDeviceGetHandleByIndex_v2), EFML_API_DESC(efmlDeviceGetHandleByIndex_v2), EFML_API_DESC(efmlDeviceGetHandleByIndex)},
      {EP_EFML_CB_ADDR(efmlErrorString), EFML_API_DESC(efmlErrorString), EFML_API_DESC(efmlErrorString)},
      {EP_EFML_CB_ADDR(efmlDeviceGetGcuLareRemotePciInfo_v2), EFML_API_DESC(efmlDeviceGetGcuLareRemotePciInfo_v2), EFML_API_DESC(efmlDeviceGetEslRemotePciInfo)},
      {EP_EFML_CB_ADDR(efmlDeviceGetArchitecture), EFML_API_DESC(efmlDeviceGetArchitecture), EFML_API_DESC(efmlDeviceGetArchitecture)},
      {EP_EFML_CB_ADDR(efmlDeviceGetGcuLareState), EFML_API_DESC(efmlDeviceGetGcuLareState), EFML_API_DESC(efmlDeviceGetGcuLareState)},
      {EP_EFML_CB_ADDR(efmlDeviceGetGcuLareCapability), EFML_API_DESC(efmlDeviceGetGcuLareCapability), EFML_API_DESC(efmlDeviceGetGcuLareCapability)},
      {EP_EFML_CB_ADDR(efmlDeviceGetGcuLareRemoteDeviceType), EFML_API_DESC(efmlDeviceGetGcuLareRemoteDeviceType), EFML_API_DESC(efmlDeviceGetGcuLareRemoteDeviceType)},
      {EP_EFML_CB_ADDR(efmlDeviceGetFieldValues), EFML_API_DESC(efmlDeviceGetFieldValues), EFML_API_DESC(efmlDeviceGetFieldValues)},
      // MNLARE support
      {EP_EFML_CB_ADDR(efmlDeviceGetGcuFabricInfoV), EFML_API_DESC(efmlDeviceGetGcuFabricInfoV), EFML_API_DESC(efmlDeviceGetGcuFabricInfoV)}
    };
    for(Symbol sym: symbols) {
      *sym.ppfn = dlsym(libhandle, sym.name);
      //if we run on old efml so, need to update by old symbol name
      if (*sym.ppfn == nullptr) {
        *sym.ppfn = dlsym(libhandle, sym.old_name);
      }
    }

    // Coverity complains that we never dlclose this object, but that's
    // deliberate, since we want the loaded object to remain in memory until
    // the process terminates, so that we can use its code.
    // coverity[leaked_storage]
  }
  #endif

  efmlReturn_v1_t res1 = EP_EFML_API_CB(efmlInit_v2)();
  if (res1 != EFML_SUCCESS_V2) {
    WARN("efmlInit_v2() failed: %s", EP_EFML_API_CB(efmlErrorString)(res1));
    initResult = epSystemError;
    return initResult;
  }

  unsigned int ndev;
  res1 = EP_EFML_API_CB(efmlDeviceGetCount_v2)(&ndev);
  if (res1 != EFML_SUCCESS_V2) {
    WARN("efmlDeviceGetCount_v2() failed: %s", EP_EFML_API_CB(efmlErrorString)(res1));
    initResult = epSystemError;
    return initResult;
  }

  epEfmlDeviceCount = int(ndev);
  if (epEfmlMaxDevices < epEfmlDeviceCount) {
    WARN("efmlDeviceGetCount() reported more devices (%d) than the internal maximum (epEfmlMaxDevices=%d)", epEfmlDeviceCount, epEfmlMaxDevices);
    initResult = epInternalError;
    return initResult;
  }

  for(int a=0; a < epEfmlDeviceCount; a++) {
    res1 = EP_EFML_API_CB(efmlDeviceGetHandleByIndex_v2)(a, &epEfmlDevices[a].handle);
    if (res1 != EFML_SUCCESS_V2) {
      WARN("efmlDeviceGetHandleByIndex_v2(%d) failed: %s", int(a), EP_EFML_API_CB(efmlErrorString)(res1));
      initResult = epSystemError;
      return initResult;
    }
  }
  initResult = epSuccess;
  return initResult;
}

epResult_t epEfmlShutdown() {
  if (!threadInitialized) return initResult;
  threadInitialized = false;
  std::lock_guard<std::mutex> locked(lock);
  if (!initialized) return initResult;
  initialized = false;
  EFMLTRY(efmlShutdown);
  EP_EFML_API_CB(efmlInit_v2) = nullptr;
  return epSuccess;
}

epResult_t epEfmlDeviceGetHandleByPciBusId(const char* pciBusId, efmlDevice_t* device) {
  EP_CHECK(epEfmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  EFMLCHECK(efmlDeviceGetHandleByPciBusId_v2, pciBusId, device);
  return epSuccess;
}

epResult_t epEfmlDeviceGetHandleByIndex(unsigned int index, efmlDevice_t *device) {
  EP_CHECK(epEfmlEnsureInitialized());
  *device = epEfmlDevices[index].handle;
  return epSuccess;
}

epResult_t epEfmlDeviceGetIndex(efmlDevice_t device, unsigned* index) {
  EP_CHECK(epEfmlEnsureInitialized());
  for (int d=0; d < epEfmlDeviceCount; d++) {
    if (epEfmlDevices[d].handle == device) {
      *index = d;
      return epSuccess;
    }
  }
  return epInvalidArgument;
}

epResult_t epEfmlDeviceGetGcuLareState(efmlDevice_t device, unsigned int port, efmlEnableState_t *isActive) {
  EP_CHECK(epEfmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  EFMLTRY(efmlDeviceGetGcuLareState, device, port, isActive);
  return epSuccess;
}

epResult_t epEfmlDeviceGetGcuLareRemotePciInfo(efmlDevice_t device, unsigned int port, efmlPciInfo_t *pci) {
  EP_CHECK(epEfmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  EFMLTRY(efmlDeviceGetGcuLareRemotePciInfo_v2, device, port, pci);
  return epSuccess;
}

epResult_t epEfmlDeviceGetArchitecture(efmlDevice_t device, efmlDeviceArchitecture_t* arch) {
  EP_CHECK(epEfmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  EFMLTRY(efmlDeviceGetArchitecture, device, arch);
  return epSuccess;
}

epResult_t epEfmlDeviceGetGcuLareCapability(efmlDevice_t device, unsigned int port,
                                                             efmlGcuLareCapability_t capability, unsigned int *capResult) {
  EP_CHECK(epEfmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  unsigned efmlId;
  EP_CHECK(epEfmlDeviceGetIndex(device, &efmlId));
  EFMLTRY(efmlDeviceGetGcuLareCapability, device, port, capability, capResult);
  return epSuccess;
}

epResult_t epEfmlDeviceGetGcuLareRemoteDeviceType(efmlDevice_t device, unsigned int port,
                                                                               efmlIntGcuLareDeviceType_t* deviceType) {
  EP_CHECK(epEfmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  unsigned efmlId;
  EP_CHECK(epEfmlDeviceGetIndex(device, &efmlId));
  EFMLTRY(efmlDeviceGetGcuLareRemoteDeviceType, device, port, deviceType);
  return epSuccess;
}

epResult_t epEfmlDeviceGetFieldValues(efmlDevice_t device, int valuesCount, efmlFieldValue_t *values) {
  EP_CHECK(epEfmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  EFMLTRY(efmlDeviceGetFieldValues, device, valuesCount, values);
  return epSuccess;
}

// MNLARE support
epResult_t epEfmlDeviceGetGcuFabricInfoV(efmlDevice_t device, efmlGcuFabricInfoV_t *gcuFabricInfo) {
  EP_CHECK(epEfmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  gcuFabricInfo->version = efmlGcuFabricInfo_v2;
  EFMLTRY(efmlDeviceGetGcuFabricInfoV, device, gcuFabricInfo);
  return epSuccess;
}

