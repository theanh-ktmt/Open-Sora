#!/bin/bash
ENABLE_XFORMERS=0 CUDA_VISIBLE_DEVICES=0 \
ENABLE_XFORMERS=1 ENABLE_TORCHCOMPILE=1 \
TORCH_TRACE="./" python3 scripts/inference.py configs/anhtt/inference.py \
  --num-frames "4s" --resolution "720p" \
  --num-sampling-steps 30 --dtype "fp16" \
  --prompt "a beautiful vocanic mountain" \
  --reference-path "save/references/sample.jpg" \
  --flash-attn "true" --layernorm-kernel "true"
