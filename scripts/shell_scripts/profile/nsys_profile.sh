#!/bin/bash
GPU=${1:-0}
export ENABLE_TENSORRT=1
export CUDA_VISIBLE_DEVICES=$GPU

nsys profile -t cuda --output=opensora_profile --sample=none \
python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 2s --resolution 144p --aspect-ratio 9:16 \
    --num-sampling-steps 3 --flow 5 --aes 6.5 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg" \
    --trt-onnx-path save/onnx/ckpts/144p-2s/stdit3_inferred.onnx \
    --trt-cache-dir save/onnx/cache/144p-2s-fp32
