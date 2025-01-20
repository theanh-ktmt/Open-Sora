#!/bin/bash
ENABLE_TENSORRT=1 CUDA_VISIBLE_DEVICES=5 \
python3 scripts/benchmark_performance.py configs/opensora-v1-2/inference/sample.py \
	--flash-attn "false" --layernorm-kernel "false" \
	--dtype "fp32" --save-dir "save/benchmark" \
	--trt-engine-path "save/tensorrt/720p-4s/stdit3_fp32_mp_o0.engine"
