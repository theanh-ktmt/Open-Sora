#!/bin/bash
ENABLE_XFORMERS=0 CUDA_VISIBLE_DEVICES=0 \
python3 scripts/inference.py configs/anhtt/inference.py \
  --num-frames "2s" --resolution "144p" \
  --num-sampling-steps 30 --dtype "fp16" \
  --prompt "a beautiful vocanic mountain" \
  --reference-path "save/references/sample.jpg" \
  --flash-attn "true" --layernorm-kernel "true"
