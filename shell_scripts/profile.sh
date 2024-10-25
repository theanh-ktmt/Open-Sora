#!/bin/bash
GPU=${1:-0}

IS_PROFILING=1 IGNORE_STEPS=1 PROFILE_OUTDIR=save/profile/tensorrt/fp32 \
ENABLE_TENSORRT=1 \
CUDA_VISIBLE_DEVICES=$GPU \
python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 2s --resolution 144p --aspect-ratio 9:16 \
    --num-sampling-steps 3 --flow 5 --aes 6.5 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg" \
    --trt-onnx-path save/onnx/ckpts/144p-2s/stdit3_inferred.onnx \
    --trt-cache-dir save/onnx/cache/144p-2s
