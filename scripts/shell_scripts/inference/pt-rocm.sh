#!/bin/bash
MIOPEN_DISABLE_CACHE=1 HIP_VISIBLE_DEVICES=7 \
ENABLE_XFORMERS=0 ENABLE_TORCHCOMPILE=1 \
python3 scripts/inference.py configs/anhtt/inference.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --prompt "a beautiful vocanic mountain" \
  # --reference-path "save/references/sample.jpg"
