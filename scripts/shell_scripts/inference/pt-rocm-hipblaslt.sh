#!/bin/bash
MIOPEN_DISABLE_CACHE=1 HIP_VISIBLE_DEVICES=0 \
ENABLE_XFORMERS=0 CUSTOM_BACKEND="hipblaslt" ENABLE_TORCHCOMPILE=0 \
python3 scripts/inference.py configs/anhtt/inference.py \
  --num-frames 4s --resolution 720p --num-sampling-steps 30 \
  --prompt "a beautiful vocanic mountain" \
  --reference-path "save/references/sample.jpg" \
  --flash-attn "true" --layernorm-kernel "false" --dtype "fp16"
