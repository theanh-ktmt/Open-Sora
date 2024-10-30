#!/bin/bash
ENABLE_XFORMERS=0 ENABLE_TENSORRT=1 CUDA_VISIBLE_DEVICES=2 \
python3 scripts/benchmark_performance.py configs/anhtt/benchmark_performance.py \
  --num-sampling-steps 30 --flow 5 --aes 6.5 \
  --trt-onnx-path save/onnx/ckpts/144p-2s/stdit3_inferred.onnx \
  --trt-cache-dir save/onnx/cache/144p-2s-fp16/
