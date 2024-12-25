#!/bin/bash
MODIFUSSION_DIR="/home/tran/workspace/modiffusion"
OPENSORA_DIR="/home/tran/workspace/Open-Sora"

docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri --group-add video \
    --ipc=host --shm-size 8G \
    -v $MODIFUSSION_DIR:/workspace/modiffusion \
    -v $OPENSORA_DIR:/workspace/Open-Sora \
    -w /workspace/Open-Sora \
    --name anhtt-opensora \
    modiffusion:latest
