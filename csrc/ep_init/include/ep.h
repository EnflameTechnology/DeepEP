#pragma once

#include <tops/tops_runtime_api.h>

/* Opaque handle to communicator */
typedef struct epComm* epComm_t;

#define EP_UNIQUE_ID_BYTES 128
typedef struct { char internal[EP_UNIQUE_ID_BYTES]; } epUniqueId;

/* Error type */
typedef enum { epSuccess                 =  0,
               epUnhandledTopsError      =  1,
               epSystemError             =  2,
               epInternalError           =  3,
               epInvalidArgument         =  4,
               epInvalidUsage            =  5,
               epRemoteError             =  6,
               epInProgress              =  7,
               epNumResults              =  8 } epResult_t;

typedef enum { epInt8       = 0, epChar       = 0,
               epUint8      = 1,
               epInt32      = 2, epInt        = 2,
               epUint32     = 3,
               epInt64      = 4,
               epUint64     = 5,
               epFloat16    = 6, epHalf       = 6,
               epFloat32    = 7, epFloat      = 7,
               epFloat64    = 8, epDouble     = 8,
               epBfloat16   = 9,
               epNumTypes   = 10 } epDataType_t;

epResult_t epGetUniqueId(epUniqueId* out);
epResult_t epCommInitRank(epComm_t* newcomm, int nranks, epUniqueId commId, int myrank);
epResult_t epCommDestroy(epComm_t comm);
epResult_t epAllGather(epComm_t comm, void* data, size_t size);
int epNetGetGdrDevice(epComm_t comm, int rank, char *devName); // invalid w/ return -1.
const char*  epGetErrorString(epResult_t result);