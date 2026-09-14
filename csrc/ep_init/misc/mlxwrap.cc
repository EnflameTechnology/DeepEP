/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>
#include "core.h"
#include "mlxwrap.h"

static pthread_once_t mlx5InitOnce = PTHREAD_ONCE_INIT;
static epResult_t mlx5InitResult;
struct epMlx5Symbols mlx5Symbols;

/* helper macro to check for NULL symbol */
#define CHECK_MLX5_NOT_NULL(container, internal_name) \
  if (container.internal_name == NULL) { \
     WARN("mlx5dr wrapper not initialized."); \
     return epInternalError; \
  }

#define MLX5_INT_CHECK_RET_ERRNO(container, internal_name, call, success_retval, name) \
  CHECK_MLX5_NOT_NULL(container, internal_name); \
  int ret = container.call; \
  if (ret != success_retval) { \
    WARN("Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    return epSystemError; \
  } \
  return epSuccess;

// return 0 represent success, otherwise failure
epResult_t wrap_mlx5dv_modify_qp_udp_sport(ibv_qp* qp, uint16_t src_port) {
  if(mlx5InitResult != epSuccess) return epSuccess;
  MLX5_INT_CHECK_RET_ERRNO(mlx5Symbols, mlx5dv_modify_qp_udp_sport, mlx5dv_modify_qp_udp_sport(qp, src_port),
                       epSuccess, "mlx5dv_modify_qp_udp_sport");
}

// return 0 represent success, otherwise failure
epResult_t wrap_mlx5dv_query_qp_lag_port(ibv_qp* qp, uint8_t* lag_num, uint8_t* active_port_num) {
  // not support, return directly
  return epSuccess;
  MLX5_INT_CHECK_RET_ERRNO(mlx5Symbols, mlx5dv_query_qp_lag_port, mlx5dv_query_qp_lag_port(qp, lag_num, active_port_num),
                       epSuccess, "mlx5dv_query_qp_lag_port");
}

// return 0 represent success, otherwise failure
epResult_t wrap_mlx5dv_modify_qp_lag_port(ibv_qp* qp, int qp_num) {
  // not support, return directly
  return epSuccess;
  MLX5_INT_CHECK_RET_ERRNO(mlx5Symbols, mlx5dv_modify_qp_lag_port, mlx5dv_modify_qp_lag_port(qp, qp_num),
                       epSuccess, "mlx5dv_modify_qp_lag_port");
}

epResult_t buildMlx5Symbols(struct epMlx5Symbols* mlx5Symbols) {
  static void* mlx5Handle = NULL;
  void* symbolPtr;
  void** usedFuncPtr;

  // Attempt to load libmlx5.so or libmlx5.so.1
  mlx5Handle=dlopen("libmlx5.so", RTLD_NOW);
  if (!mlx5Handle) {
    mlx5Handle=dlopen("libmlx5.so.1", RTLD_NOW);
    if (!mlx5Handle) {
      INFO(EP_INIT, "Failed to open libmlx5.so[.1]");
      return epSystemError;
    }
  }

  // Define a macro to load symbols from the shared library
  #define LOAD_SYM(handle, symbol, funcptr) do {           \
    usedFuncPtr = (void**)&funcptr;                             \
    symbolPtr = dlsym(handle, symbol);                         \
    if (symbolPtr == NULL) {                                   \
      WARN("dlvsym failed on %s - %s", symbol, dlerror());  \
      goto teardown;                                     \
    }                                                    \
    *usedFuncPtr = symbolPtr;                                         \
  } while (0)

  // Load the required symbols
  LOAD_SYM(mlx5Handle, "mlx5dv_modify_qp_udp_sport", mlx5Symbols->mlx5dv_modify_qp_udp_sport);
  LOAD_SYM(mlx5Handle, "mlx5dv_query_qp_lag_port", mlx5Symbols->mlx5dv_query_qp_lag_port);
  LOAD_SYM(mlx5Handle, "mlx5dv_modify_qp_lag_port", mlx5Symbols->mlx5dv_modify_qp_lag_port);

  return epSuccess;

teardown:
  // Clear the symbol pointers
  mlx5Symbols->mlx5dv_modify_qp_udp_sport = NULL;
  mlx5Symbols->mlx5dv_query_qp_lag_port = NULL;
  mlx5Symbols->mlx5dv_modify_qp_lag_port = NULL;

  // Close the shared library handle
  if (mlx5Handle != NULL) {
    dlclose(mlx5Handle);
    mlx5Handle = NULL;
  }
  return epSystemError;
}

epResult_t wrap_mlx5_symbols(void) {
  pthread_once(&mlx5InitOnce,
               [](){ mlx5InitResult = buildMlx5Symbols(&mlx5Symbols); });
  return mlx5InitResult;
}
