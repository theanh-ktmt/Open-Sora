#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

nsys profile -t cuda --output=opensora_profile --sample=none \
python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 3 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg" \
	--flash-attn "true" --layernorm-kernel "false" \
	--dtype "fp16" --save-dir "save/inference/test"

python3 scripts/benchmark_performance.py configs/opensora-v1-2/inference/sample.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 30 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg" \
	--flash-attn "true" --layernorm-kernel "false" \
	--dtype "fp16" --save-dir "save/inference/test"
