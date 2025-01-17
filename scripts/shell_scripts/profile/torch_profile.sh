#!/bin/bash
# profile args
export ENABLE_PROFILER=1
export TARGET_SAMPLE=2
export PROFILE_OUTDIR="save/profile/pt-rocm-tc/mi300-fp16/720p-4s/torch_profile/replaced-attn"
export TORCH_TRACE=$PROFILE_OUTDIR

# gpus
export HIP_VISIBLE_DEVICES=7

# modules
export ENABLE_TORCHCOMPILE=1
# export CUSTOM_BACKEND="ck" # or "hipblaslt"

python3 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 3 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg" \
	--flash-attn "true" --layernorm-kernel "false" \
	--dtype "fp16" --save-dir "save/inference/test"
