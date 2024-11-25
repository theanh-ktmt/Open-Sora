#!/bin/bash
ENABLE_TENSORRT=1 CUDA_VISIBLE_DEVICES=5 \
python3 scripts/benchmark_performance.py \
	configs/anhtt/benchmark_performance.py \
	--dtype "fp32" --flash-attn "false" --layernorm-kernel "false" \
	--trt-engine-path "save/tensorrt/720p-4s/stdit3_fp32_mp.engine"
