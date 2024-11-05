#!/bin/bash
MOREH_BACKEND_TYPE=ucx
MOREH_BACKEND_PORT=${1:-3720}
MOREH_BACKEND_ADDRESS=localhost
MOREH_BACKEND_NODES=localhost:0,1
MOREH_USE_OP_DEF=0

ENABLE_XFORMERS=0 CUDA_VISIBLE_DEVICES=2 \
python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 2s --resolution 144p --aspect-ratio 9:16 \
    --num-sampling-steps 30 --flow 5 --aes 6.5 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg"
