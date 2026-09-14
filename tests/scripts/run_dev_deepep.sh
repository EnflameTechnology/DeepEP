#!/bin/bash

LOG_TIME=`date '+%Y%m%d%H%M%S'`

# ============= 分布式训练环境变量配置 =============
# 本地 rank (节点内的进程编号)
export DEVICE_ID=$OMPI_COMM_WORLD_LOCAL_RANK

# 全局 rank (所有节点中的进程编号)
export RANK=${OMPI_COMM_WORLD_RANK:-0}

# 总进程数
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}

# 主节点地址和端口 (可通过命令行参数或环境变量设置)
# 默认值：MASTER_ADDR=localhost, MASTER_PORT=29500
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}

# export EP_DEBUG=INFO
# export EP_DEBUG_SUBSYS=ALL

echo "=== Distributed Training Info ==="
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "LOCAL_RANK: $DEVICE_ID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "================================="
# ============= 分布式训练环境变量配置结束 =============

CUR_DIR=$(pwd)
LOG_DIR=$CUR_DIR/logs

if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi

echo "this is rank $DEVICE_ID"
echo "================================="
echo "Executing command: $*"
echo "Log file: $LOG_DIR/${LOG_TIME}_rank${RANK}_local${DEVICE_ID}.log"
echo "================================="
#$@
"$@" > $LOG_DIR/${LOG_TIME}_rank${RANK}_local${DEVICE_ID}.log 2>&1
