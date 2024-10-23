#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python3 scripts/benchmark_performance.py configs/anhtt/benchmark_performance.py \
  --aspect-ratio 9:16 \
  --num-sampling-steps 30 \
  --flow 5 \
  --aes 6.5
