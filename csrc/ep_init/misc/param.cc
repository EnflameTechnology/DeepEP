/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"
#include <sys/types.h>
#include <pwd.h>
#include <cerrno>
#include <algorithm>

void epLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  if (__atomic_load_n(cache, __ATOMIC_ACQUIRE) == uninitialized) {
    pthread_mutex_lock(&mutex);
    if (__atomic_load_n(cache, __ATOMIC_ACQUIRE) == uninitialized) {
      const char* str = (env) ? epGetEnv(env) : nullptr;
      int64_t value = deftVal;

      if (str && strlen(str) > 0) {
        errno = 0;
        char* endptr;
        value = strtoll(str, &endptr, 0);
        // Check for errors in conversion
        if (endptr == str || *endptr != '\0' || errno == ERANGE) {
          value = deftVal;
          INFO(EP_ALL, "Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
        } else {
          INFO(EP_ENV, ENV_FORMAT_INT, env, (long long)value);
        }
      }
      __atomic_store_n(cache, value, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&mutex);
  }
}

const char *epGetEnv(const char *name) {
  return getenv(name);
}