#!/bin/bash
# profile args
export IS_PROFILING=1
export TARGET_SAMPLE=2
export PROFILE_OUTDIR="save/profile/custom/end2end/torch.compile_wo_ck"
export TORCH_TRACE="./"

# gpus
export MIOPEN_DISABLE_CACHE=1
export HIP_VISIBLE_DEVICES=0

# modules
export ENABLE_XFORMERS=0
export ENABLE_TORCHCOMPILE=1
export ENABLE_CK=0

python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 30 \
    --prompt "a beautiful vocanic mountain" --num-sample 3 \
    --reference-path "save/references/sample.jpg" \
    --flash-attn "true" --layernorm-kernel "false" --dtype "fp16"
