#!/bin/bash
PORT=${1:-3720}
MOREH_BACKEND_PATH=/home/tran/workspace/archive/moreh-24.10.3001-Linux

MOREH_LIB_PATH+=:/opt/rocm/lib
MOREH_LIB_PATH+=:/opt/rocm/opencl
MOREH_LIB_PATH+=:/opt/openmpi/lib
MOREH_LIB_PATH+=:/opt/moreh/lib
[[ ! -z $CONDA_PREFIX ]] && MOREH_LIB_PATH+=:${CONDA_PREFIX}/lib
MOREH_LIB_PATH+=:${MOREH_BACKEND_PATH}/lib
MOREH_LIB_PATH+=:${MOREH_BACKEND_PATH}/bin
MOREH_LIB_PATH+=:${LD_LIBRARY_PATH}

mpirun -np 1 -H localhost --bind-to none \
    --mca btl_openib_allow_ib 1 \
    -x MOREH_BACKEND_TYPE=ucx \
    -x HSA_DISABLE_CACHE=0 \
    -x HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    -x MODNN_SEND_RECV_PIPE=1 \
    -x MODNN_KERNEL_DIR=$MOREH_BACKEND_PATH/kernels \
    -x LD_LIBRARY_PATH=$MOREH_LIB_PATH \
    $MOREH_BACKEND_PATH/bin/moreh_worker $PORT $@
