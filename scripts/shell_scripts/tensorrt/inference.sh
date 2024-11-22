#!/bin/bash
ENABLE_TENSORRT=1 CUDA_VISIBLE_DEVICES=0 \
python3 scripts/inference.py configs/anhtt/inference.py \
  --num-frames "4s" --resolution "720p" \
  --num-sampling-steps 30 --dtype "fp32" \
  --prompt "a beautiful vocanic mountain" \
  --reference-path "save/references/sample.jpg" \
  --flash-attn "false" --layernorm-kernel "false" \
  --trt-engine-path "save/tensorrt/720p-4s/stdit3.engine"
