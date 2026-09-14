/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_DEBUG_H_
#define EP_DEBUG_H_

#include <stdio.h>
#include <chrono>
#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include <pthread.h>

#include "ep.h"

typedef enum {
  EP_LOG_NONE = 0,
  EP_LOG_VERSION = 1,
  EP_LOG_WARN = 2,
  EP_LOG_INFO = 3,
  EP_LOG_ABORT = 4,
  EP_LOG_TRACE = 5
} epDebugLogLevel;

typedef enum {
  EP_INIT = 0x1,
  EP_COLL = 0x2,
  EP_P2P = 0x4,
  EP_SHM = 0x8,
  EP_NET = 0x10,
  EP_GRAPH = 0x20,
  EP_TUNING = 0x40,
  EP_ENV = 0x80,
  EP_ALLOC = 0x100,
  EP_CALL = 0x200,
  EP_PROXY = 0x400,
  EP_BOOTSTRAP = 0x800,
  EP_PROFILE = 0x1000,
  EP_ALL = ~0
} epDebugLogSubSys;

#define EP_MAX_DEBUG_CNT 5

#define EP_THREAD_NAMELEN 16

extern int epDebugLevel;
extern uint64_t epDebugMask;
extern pthread_mutex_t epDebugOutputLock;
extern FILE *epDebugFile;
extern epResult_t getHostName(char* hostname, int maxlen, const char delim);
extern char epLastError[1024];

void epDebugLog(epDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...);

// Let code temporarily downgrade WARN into INFO
__attribute__((unused))
static thread_local int epDebugNoWarn;

#define WARN(...)        epDebugLog(EP_LOG_WARN, EP_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) epDebugLog(EP_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) epDebugLog(EP_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::high_resolution_clock::time_point epEpoch;
#else
#define TRACE(...)
#endif

void epSetThreadName(pthread_t thread, const char *fmt, ...);

#endif
