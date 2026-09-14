/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_PARAM_H_
#define EP_PARAM_H_

#include <unistd.h>
#include <sys/types.h>

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();
const char *epGetEnv(const char *name);
void epLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache);

#define ENV_FORMAT_INT "%s set by environment to %lld."
#define ENV_FORMAT_STR "%s set by environment to %s."

#define EP_PARAM(name, env, deftVal) \
  int64_t epParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false)) { \
      epLoadParam("EP_" env, deftVal, uninitialized, &cache); \
    } \
    return cache; \
  }

#endif
