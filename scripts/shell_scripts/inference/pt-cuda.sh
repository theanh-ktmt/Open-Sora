#!/bin/bash
CUDA_VISIBLE_DEVICES=0 ENABLE_TORCHCOMPILE=1 \
python3 scripts/benchmark_performance.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --num-sampling-steps 30 \
  --prompt "a beautiful vocanic mountain" \
  --reference-path "save/references/sample.jpg" \
	--flash-attn "true" --layernorm-kernel "false" \
	--dtype "fp16" --save-dir "save/inference/test"
