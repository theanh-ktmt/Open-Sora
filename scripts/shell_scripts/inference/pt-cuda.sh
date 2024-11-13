#!/bin/bash
ENABLE_XFORMERS=1 CUDA_VISIBLE_DEVICES=3 \
python3 scripts/inference.py configs/anhtt/inference.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --num-sampling-steps 30 --flow 5 --aes 6.5 \
  --prompt "A person is playing piano" \
  # --reference-path "save/references/sample.jpg"
