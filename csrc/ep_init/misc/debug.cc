#include <stdlib.h>
#include <stdarg.h>

#include "core.h"

int epDebugLevel = -1;
// Default debug sub-system mask is INIT
uint64_t epDebugMask = EP_INIT;
FILE *epDebugFile = stdout;
pthread_mutex_t epDebugLock = PTHREAD_MUTEX_INITIALIZER;
char epLastError[1024] = ""; // Global string for the last error in human readable form

void epDebugInit() {
  pthread_mutex_lock(&epDebugLock);
  if (epDebugLevel != -1) { pthread_mutex_unlock(&epDebugLock); return; }
  const char* ep_debug = getenv("EP_DEBUG");
  if (ep_debug == NULL) {
    epDebugLevel = EP_LOG_WARN; // default to WARN
  } else if (strcasecmp(ep_debug, "VERSION") == 0) {
    epDebugLevel = EP_LOG_VERSION;
  } else if (strcasecmp(ep_debug, "WARN") == 0) {
    epDebugLevel = EP_LOG_WARN;
  } else if (strcasecmp(ep_debug, "INFO") == 0) {
    epDebugLevel = EP_LOG_INFO;
  } else if (strcasecmp(ep_debug, "ABORT") == 0) {
    epDebugLevel = EP_LOG_ABORT;
  } else if (strcasecmp(ep_debug, "TRACE") == 0) {
    epDebugLevel = EP_LOG_TRACE;
  }

  /* Parse the EP_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  char* epDebugSubsysEnv = getenv("EP_DEBUG_SUBSYS");
  if (epDebugSubsysEnv != NULL) {
    int invert = 0;
    if (epDebugSubsysEnv[0] == '^') { invert = 1; epDebugSubsysEnv++; }
    epDebugMask = invert ? ~0ULL : 0ULL;
    char *epDebugSubsys = strdup(epDebugSubsysEnv);
    char *subsys = strtok(epDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = EP_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = EP_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = EP_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = EP_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = EP_NET;
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = EP_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = EP_TUNING;
      } else if (strcasecmp(subsys, "ENV") == 0) {
        mask = EP_ENV;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = EP_ALL;
      }
      if (mask) {
        if (invert) epDebugMask &= ~mask; else epDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(epDebugSubsys);
  }

  /* Parse and expand the EP_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * EP_DEBUG level is > VERSION
   */
  const char* epDebugFileEnv = getenv("EP_DEBUG_FILE");
  if (epDebugLevel > EP_LOG_VERSION && epDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX+1] = "";
    char *dfn = debugFn;
    while (epDebugFileEnv[c] != '\0' && c < PATH_MAX) {
      if (epDebugFileEnv[c++] != '%') {
        *dfn++ = epDebugFileEnv[c-1];
        continue;
      }
      switch (epDebugFileEnv[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          char hostname[1024];
          getHostName(hostname, 1024, '.');
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = epDebugFileEnv[c-1];
          break;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE *file = fopen(debugFn, "w");
      if (file != NULL) {
        epDebugFile = file;
      }
    }
  }

#ifdef ENABLE_TRACE
  epEpoch = std::chrono::high_resolution_clock::now();
#endif
  pthread_mutex_unlock(&epDebugLock);
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void epDebugLog(epDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) {
  if (epDebugLevel == -1) epDebugInit();
  if (epDebugNoWarn != 0 && level == EP_LOG_WARN) { level = EP_LOG_INFO; flags = epDebugNoWarn; }
  // Save the last error (WARN) as a human readable string
  if (level == EP_LOG_WARN) {
    pthread_mutex_lock(&epDebugLock);
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(epLastError, sizeof(epLastError), fmt, vargs);
    va_end(vargs);
    pthread_mutex_unlock(&epDebugLock);
  }
  if (epDebugLevel < level || ((flags & epDebugMask) == 0)) return;

  // Gather the rank information. This can take > 1us so we want to make sure
  // we only do it when needed.
  char hostname[1024];
  getHostName(hostname, 1024, '.');
  int topsDev;
  (void)topsGetDevice(&topsDev);
  int pid = getpid();

  // Get the current date and time, put it at the beginning of the log
  char datetime[64];
  time_t now = time(NULL);
  struct tm tm_now;
  // localtime_r will automatically use the system timezone, same as date
  localtime_r(&now, &tm_now);
  strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", &tm_now);

  char buffer[1024];
  size_t len = 0;
  pthread_mutex_lock(&epDebugLock);
  if (level == EP_LOG_WARN)
    len = snprintf(buffer, sizeof(buffer),
        "\n EP WARN %s %s:%d [%d] %s:%d", datetime, hostname, pid, topsDev, filefunc, line);
  else if (level == EP_LOG_INFO)
    len = snprintf(buffer, sizeof(buffer),
        "EP INFO %s %s:%d [%d]", datetime, hostname, pid, topsDev);
#ifdef ENABLE_TRACE
  else if (level == EP_LOG_TRACE) {
    auto delta = std::chrono::high_resolution_clock::now() - epEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;
    len = snprintf(buffer, sizeof(buffer),
       "%s:%d:%lu [%d] %f %s:%d EP TRACE ", hostname, pid, topsDev, timestamp, filefunc, line);
  }
#endif
  if (len) {
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(buffer+len, sizeof(buffer)-len, fmt, vargs);
    va_end(vargs);
    fprintf(epDebugFile,"%s\n", buffer);
    fflush(epDebugFile);
  }
  pthread_mutex_unlock(&epDebugLock);
}

EP_PARAM(SetThreadName, "SET_THREAD_NAME", 0);

void epSetThreadName(pthread_t thread, const char *fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (epParamSetThreadName() != 1) return;
  char threadName[EP_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, EP_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread, threadName);
#endif
  (void) thread;
  (void) fmt;
}

