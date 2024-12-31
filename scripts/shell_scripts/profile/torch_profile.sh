#!/bin/bash
# profile args
export IS_PROFILING=1
export TARGET_SAMPLE=2
export PROFILE_OUTDIR="save/profile/pt-rocm-tc/mi300-fp16/720p-4s/torch_profile/replaced-attn"
export TORCH_TRACE="tmp"

# gpus
export MIOPEN_DISABLE_CACHE=1
export HIP_VISIBLE_DEVICES=7

# modules
export ENABLE_XFORMERS=1
export ENABLE_TORCHCOMPILE=1
# export CUSTOM_BACKEND="ck" # or "hipblaslt"
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1

python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 30 \
    --prompt "a beautiful vocanic mountain" --num-sample 3 \
    --reference-path "save/references/sample.jpg" \
    --flash-attn "true" --layernorm-kernel "false" --dtype "fp16"
