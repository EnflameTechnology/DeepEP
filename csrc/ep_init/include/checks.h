/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_CHECKS_H_
#define EP_CHECKS_H_

#include <string>
#include <exception>

#include "debug.h"
#include "configs.h"
#include "ep.h"

class EPException: public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char *what() const noexcept override { return message.c_str(); }
};

#define EP_CHECK(call) do { \
  epResult_t RES = call; \
  if (RES != epSuccess && RES != epInProgress) { \
    throw EPException("EP", __FILE__, __LINE__, epGetErrorString(RES)); \
  } \
} while (0);

#define EP_CHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != epSuccess && res != epInProgress) { \
    throw EPException("EP", __FILE__, __LINE__, epGetErrorString(res)); \
    goto label; \
  } \
} while (0);

#ifndef TOPS_CHECK
#define TOPS_CHECK(cmd) \
do { \
    topsError_t e = (cmd); \
    if (e != topsSuccess) { \
        throw EPException("TOPS", __FILE__, __LINE__, topsGetErrorString(e)); \
    } \
} while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif


#define __FILENAME__                                                           \
  ((strrchr(__FILE__, '/')) ? (strrchr(__FILE__, '/') + 1) : __FILE__)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return epSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    WARN("Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define SYSCHECKGOTO(statement, name, RES, label) do { \
    int retval; \
    SYSCHECKSYNC((statement), name, retval); \
    if (retval == -1) { \
      WARN("Call to " name " failed: %s", strerror(errno)); \
      RES = epSystemError; \
      goto label; \
    } \
  } while (0)

// Pthread calls don't set errno and never return EINTR.
#define PTHREADCHECK(statement, name) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to " name " failed: %s", strerror(retval)); \
    return epSystemError; \
  } \
} while (0)

#define PTHREADCHECKGOTO(statement, name, RES, label) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to " name " failed: %s", strerror(retval)); \
    RES = epSystemError; \
    goto label; \
  } \
} while (0)

#define NEQCHECK(statement, value) do {   \
    if ((statement) != value) {             \
      throw EPException("EP", __FILE__, __LINE__, epGetErrorString(epSystemError)); \
    }                             \
  } while (0);

#define PLAIN "\033[0m"
#define RED    "\033[0;32;31m"
#define LIGHT_RED    "\033[1;31m"
#define GREEN  "\033[0;32;32m"
#define LIGHT_GREEN  "\033[1;32m"
#define YELLOW "\033[0;32;33m"
#define BLUE   "\033[0;32;34m"
#define LIGHT_BLUE   "\033[1;34m"
#define BOLD   "\033[1m"
#define DARY_GRAY    "\033[1;30m"
#define CYAN         "\033[0;36m"
#define LIGHT_CYAN   "\033[1;36m"
#define PURPLE       "\033[0;35m"
#define LIGHT_PURPLE "\033[1;35m"
#define BROWN        "\033[0;33m"
#define LIGHT_GRAY   "\033[0;37m"
#define WHITE        "\033[1;37m"

#define VOID
#define CheckPrintAndDo(exp, ops, fmt, args...)                                \
  do {                                                                         \
    if (!(exp)) {                                                              \
      INFO(EP_ALL, RED "[%s:%d %s()] " fmt PLAIN, __FILENAME__, __LINE__, __FUNCTION__, ##args);    \
      ops;                                                                     \
    }                                                                          \
  } while (0)


#endif
