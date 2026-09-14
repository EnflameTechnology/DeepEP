/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_UTILS_H_
#define EP_UTILS_H_

#include <stdint.h>
#include "ep.h"
#include "checks.h"
#include "alloc.h"

#ifdef __cplusplus
  #if __cplusplus >= 201703L
    #define CONSTEXPR constexpr
  #else
    #define CONSTEXPR
  #endif
#else
  #define CONSTEXPR
#endif

// PCI Bus ID <-> int64 conversion functions
epResult_t int64ToBusId(int64_t id, char* busId);
epResult_t busIdToInt64(const char* busId, int64_t* id);

epResult_t getBusId(int topsDev, int64_t *busId);

epResult_t getHostName(char* hostname, int maxlen, const char delim);
uint64_t getHash(const char* string, int n);
uint64_t getHostHash();
uint64_t getPidHash();
epResult_t getRandomData(void* buffer, size_t bytes);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

void leftRotate(int arr[], int size, int shift);

inline static long log2i(long n) {
 long l = 0;
 while (n>>=1) l++;
 return l;
}

/* get any bytes of random data from /dev/urandom, return 0 if it succeeds; else
 * return -1 */
inline epResult_t getRandomData(void* buffer, size_t bytes) {
  epResult_t ret = epSuccess;
  if (bytes > 0) {
    const size_t one = 1UL;
    FILE* fp = fopen("/dev/urandom", "r");
    if (buffer == NULL || fp == NULL || fread(buffer, bytes, one, fp) != one) ret = epSystemError;
    if (fp) fclose(fp);
  }
  return ret;
}

template<typename Int>
inline void epAtomicRefCountIncrement(Int* refs) {
  __atomic_fetch_add(refs, 1, __ATOMIC_RELAXED);
}

template<typename Int>
inline Int epAtomicRefCountDecrement(Int* refs) {
  return __atomic_sub_fetch(refs, 1, __ATOMIC_ACQ_REL);
}

// Recyclable list that avoids frequent malloc/free
template<typename T>
struct epListElem {
  T data;
  struct epListElem* next;
};

template<typename T>
class epRecyclableList {
 private:
  struct epListElem<T>* head;
  struct epListElem<T>* tail;
  struct epListElem<T>* cursor;
  int n;

 public:
  epRecyclableList() {
    tail = cursor = head = NULL;
    n = 0;
  }

  int count() const { return n; }

  // Get a new element from the list and return pointer
  epResult_t getNewElem(T** dataOut) {
    if (tail != NULL) {
      *dataOut = &tail->data;
      memset(*dataOut, 0, sizeof(T));
    } else {
      EP_CHECK(epCalloc(&tail, 1));
      *dataOut = &tail->data;
      cursor = head = tail;
    }
    if (tail->next == NULL) {
      EP_CHECK(epCalloc(&tail->next, 1));
    }
    tail = tail->next;
    n += 1;
    return epSuccess;
  }

  T* begin() {
    if (head == NULL || head == tail) return NULL;
    cursor = head->next;
    return &head->data;
  }

  // Get next element from the list during an iteration
  T* getNext() {
    // tail always points to the next element to be enqueued
    // hence does not contain valid data
    if (cursor == NULL || cursor == tail) return NULL;
    T* rv = &cursor->data;
    cursor = cursor->next;
    return rv;
  }

  T* peakNext() {
    if (cursor == NULL || cursor == tail) return NULL;
    return &cursor->data;
  }

  // Recycle the list without freeing the space
  void recycle() {
    tail = cursor = head;
    n = 0;
  }

  ~epRecyclableList() {
    while (head != NULL) {
      struct epListElem<T>* temp = head;
      head = head->next;
      free(temp);
    }
  }
};

#endif
