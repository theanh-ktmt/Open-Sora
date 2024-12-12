#!/bin/bash
MIOPEN_DISABLE_CACHE=1 HIP_VISIBLE_DEVICES=1 \
ENABLE_XFORMERS=1 ENABLE_TORCHCOMPILE=0 \
python3 scripts/benchmark_performance.py \
    configs/anhtt/benchmark_performance.py \
    --flash-attn "true" --layernorm-kernel "false" --dtype "fp16"
