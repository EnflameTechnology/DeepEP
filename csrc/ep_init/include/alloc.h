/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_ALLOC_H_
#define EP_ALLOC_H_

#include <sys/mman.h>
#include <tops/tops_ext.h>

#include "ep.h"
#include "align.h"
#include "checks.h"
#include "core.h"

#define PLATFORM_ASIC 0x0

template <typename T>
static epResult_t epTopsHostCalloc(T** ptr, size_t nelem, topsStream_t stream = nullptr) {
  if(stream == nullptr){
    TOPS_CHECK(topsHostMalloc(ptr, nelem*sizeof(T), topsHostMallocMapped));
    TOPS_CHECK(topsMemset(*ptr, 0, nelem*sizeof(T)));
  }
  else{
    topsStreamCaptureMode mode = topsStreamCaptureModeRelaxed;
    TOPS_CHECK(topsThreadExchangeStreamCaptureMode(&mode));
    TOPS_CHECK(topsHostMalloc(ptr, nelem*sizeof(T), topsHostMallocMapped));
    TOPS_CHECK(topsMemsetAsync(*ptr, 0, nelem*sizeof(T), stream));
    TOPS_CHECK(topsStreamSynchronize(stream));
    TOPS_CHECK(topsThreadExchangeStreamCaptureMode(&mode));
  }
  return epSuccess;
}

template <typename T>
static inline epResult_t epTopsHostFree(T* &ptr) {
  if (ptr) {
    TOPS_CHECK(topsHostFree((void*)ptr));
    ptr = nullptr;
  }
  return epSuccess;
}

template <typename T>
static epResult_t epCalloc(T** ptr, size_t nelem) {
  void* p = malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return epSystemError;
  }
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  return epSuccess;
}
template <typename T>
static inline void epCfree(T* & ptr) {
  if (ptr) {
    free((void*)ptr);
    ptr = nullptr;
  }
}

template <typename T>
static epResult_t epTopsCalloc(T** ptr, size_t nelem) {
  TOPS_CHECK(topsMalloc(ptr, nelem*sizeof(T)));
  TOPS_CHECK(topsMemset(*ptr, 0, nelem*sizeof(T)));
  return epSuccess;
}

template <typename T>
static epResult_t epTopsCallocWithFlags(T** ptr, size_t nelem, unsigned int flags, topsStream_t stream = nullptr) {
  if(stream == nullptr){
    TOPS_CHECK(topsExtMallocWithFlags((void**)ptr, nelem*sizeof(T), flags));
    TOPS_CHECK(topsMemset(*ptr, 0, nelem*sizeof(T)));
  }
  else{
    topsStreamCaptureMode mode = topsStreamCaptureModeRelaxed;
    TOPS_CHECK(topsThreadExchangeStreamCaptureMode(&mode));
    TOPS_CHECK(topsExtMallocWithFlags((void**)ptr, nelem*sizeof(T), flags));
    TOPS_CHECK(topsMemsetAsync(*ptr, 0, nelem*sizeof(T), stream));
    TOPS_CHECK(topsStreamSynchronize(stream));
    TOPS_CHECK(topsThreadExchangeStreamCaptureMode(&mode));
  }

  return epSuccess;
}

template <typename T>
static inline epResult_t epTopsFree(T* & ptr) {
  if (ptr) {
    TOPS_CHECK(topsFree((void*)ptr));
    ptr = nullptr;
  }
  return epSuccess;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
__attribute__((unused))
static epResult_t epIbMalloc(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return epSystemError;
  memset(p, 0, size);
  *ptr = p;
  return epSuccess;
}
#define LARE_SEND_BANK_ID 3

typedef enum {
  E_topsMalloc,
  E_topsMallocWithFlags,
  E_topsMallocWithBank
}E_epMallocType;

template <typename T>
struct epDevMemDesc {
  void* unalignedPtr = nullptr;
  T* alignedPtr = nullptr;
  size_t userSize = 0;
  T* alignedDevPtr = 0;

  template <E_epMallocType MallocType>
  epResult_t alloc(size_t count, topsStream_t stream, size_t alignment=0, unsigned int flags=topsDeviceMallocDefault, int value=0) {
    userSize = count*sizeof(T);
    size_t allocatedSize = alignment? userSize + alignment - 1 : userSize;
    topsStreamCaptureMode mode = topsStreamCaptureModeRelaxed;
    TOPS_CHECK(topsThreadExchangeStreamCaptureMode(&mode));
    if (MallocType == E_topsMalloc) {
      TOPS_CHECK(topsMalloc((T**)&unalignedPtr, allocatedSize));
    } else if (MallocType == E_topsMallocWithFlags) {
      TOPS_CHECK(topsExtMallocWithFlags(&unalignedPtr, allocatedSize, flags));
    }
    TOPS_CHECK(topsMemsetAsync(unalignedPtr, value, allocatedSize, stream));
    TOPS_CHECK(topsStreamSynchronize(stream));
    TOPS_CHECK(topsThreadExchangeStreamCaptureMode(&mode));
    if (alignment)
      alignedPtr = (T*)ROUNDUP((uint64_t)unalignedPtr, alignment);
    else
      alignedPtr = (T*)unalignedPtr;

    uint64_t alignedAddr;
    TOPS_CHECK(topsPointerGetAttribute(&alignedAddr, TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER, (void*)alignedPtr));
    alignedDevPtr = (T*)alignedAddr;
    return epSuccess;
  }
  epResult_t dealloc() {
    epResult_t res = epSuccess;
    if (unalignedPtr && userSize) {
      res = epTopsFree(unalignedPtr);
      unalignedPtr = nullptr;
      alignedPtr = nullptr;
      alignedDevPtr = nullptr;
      userSize = 0;
    }
    return res;
  }
};

template <typename T>
static epResult_t epRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  if (nelem < oldNelem) return epInternalError;
  if (nelem == oldNelem) return epSuccess;

  T* oldp = *ptr;
  T* p = (T*)malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return epSystemError;
  }
  memcpy(p, oldp, oldNelem*sizeof(T));
  free(oldp);
  memset(p+oldNelem, 0, (nelem-oldNelem)*sizeof(T));
  *ptr = (T*)p;
  INFO(EP_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*sizeof(T), nelem*sizeof(T), *ptr);
  return epSuccess;
}

#endif
