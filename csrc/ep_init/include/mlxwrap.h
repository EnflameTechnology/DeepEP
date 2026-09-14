/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_MLX5WRAP_H_
#define EP_MLX5WRAP_H_

#include "core.h"
#include <sys/types.h>
#include <unistd.h>
#include "ibvwrap.h"

epResult_t wrap_mlx5_symbols(void);

epResult_t wrap_mlx5dv_modify_qp_udp_sport(ibv_qp* qp, uint16_t src_port);
epResult_t wrap_mlx5dv_query_qp_lag_port(ibv_qp* qp, uint8_t* lag_num, uint8_t* active_port_num);
epResult_t wrap_mlx5dv_modify_qp_lag_port(ibv_qp* qp, int qp_num);


/* mlx5dv Helper APIs Function Pointers*/
struct epMlx5Symbols {
  int (*mlx5dv_modify_qp_udp_sport)(ibv_qp* qp, uint16_t src_port);
  int (*mlx5dv_query_qp_lag_port)(ibv_qp* qp, uint8_t* lag_num, uint8_t* active_port_num);
  int (*mlx5dv_modify_qp_lag_port)(ibv_qp* qp, int qp_num);
};

#endif //End include guard
