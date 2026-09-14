#ifndef EP_EFML_WRAP_H_
#define EP_EFML_WRAP_H_
#include "ep.h"
#include <efml/efml.h>
#include <vector>

//#define EP_EFML_DIRECT 1
#ifndef EP_EFML_DIRECT
#define EP_EFML_DIRECT 0
#endif

#define SWITCH_BUSID "fffffff:ffff:ff"
#define MAX_LARES_PER_GCU 16U

#define UUID_SIZE 16
typedef struct {
  unsigned int efmlPort;
  char remoteUuidId[UUID_SIZE];  ///< uuid of remote devices
} epEfmlGcuLareInfo;

constexpr int epEfmlMaxDevices = 16;
struct epEfmlDeviceInfo {
  efmlDevice_t handle = nullptr;
};

extern int epEfmlDeviceCount;
extern epEfmlDeviceInfo epEfmlDevices[epEfmlMaxDevices];

// All epEfmlFoo() functions call epEfmlEnsureInitialized() implicitly.
// Outsiders need only call it if they want to inspect the epEfml global
// tables above.
epResult_t epEfmlEnsureInitialized();
epResult_t epEfmlShutdown();
epResult_t epEfmlDeviceGetHandleByPciBusId(const char* pciBusId, efmlDevice_t* device);
epResult_t epEfmlDeviceGetIndex(efmlDevice_t device, unsigned* index);
epResult_t epEfmlDeviceGetHandleByIndex(unsigned int index, efmlDevice_t *device);
epResult_t epEfmlDeviceGetGcuLareState(efmlDevice_t device, unsigned int port, efmlEnableState_t *isActive);
epResult_t epEfmlDeviceGetGcuLareRemotePciInfo(efmlDevice_t device, unsigned int port, efmlPciInfo_t *pci);
epResult_t epEfmlDeviceGetArchitecture(efmlDevice_t device, efmlDeviceArchitecture_t* arch);
epResult_t epEfmlDeviceGetGcuLareCapability(efmlDevice_t device, unsigned int port, efmlGcuLareCapability_t capability, unsigned int *capResult);
epResult_t epEfmlDeviceGetGcuLareRemoteDeviceType(efmlDevice_t device, unsigned int port,
                                                                               efmlIntGcuLareDeviceType_t* deviceType);
epResult_t epEfmlDeviceGetFieldValues(efmlDevice_t device, int valuesCount, efmlFieldValue_t *values);
epResult_t epEfmlDeviceGetGcuFabricInfoV(efmlDevice_t device, efmlGcuFabricInfoV_t *gcuFabricInfo);
#endif
