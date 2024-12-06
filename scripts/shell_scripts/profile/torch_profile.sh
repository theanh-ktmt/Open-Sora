#!/bin/bash
IS_PROFILING=1 TARGET_SAMPLE=2 PROFILE_OUTDIR=save/profile/pt-rocm-tc/mi300-fp16/720p-4s/torch_profile \
MIOPEN_DISABLE_CACHE=1 HIP_VISIBLE_DEVICES=6 \
ENABLE_XFORMERS=0 ENABLE_TORCHCOMPILE=1 \
python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 30 \
    --prompt "a beautiful vocanic mountain" --num-sample 3 \
    --reference-path "save/references/sample.jpg" \
    --flash-attn "true" --layernorm-kernel "false" --dtype "fp16"
