#!/bin/bash
ENABLE_XFORMERS=0 CUDA_VISIBLE_DEVICES=0 ENABLE_TORCHCOMPILE=1 \
python3 scripts/inference.py configs/anhtt/inference.py \
  --num-frames 2s --resolution 144p --aspect-ratio 9:16 \
  --num-sampling-steps 30 --flow 5 --aes 6.5 \
  --prompt "a beautiful vocanic mountain" \
  --reference-path "save/references/sample.jpg"
