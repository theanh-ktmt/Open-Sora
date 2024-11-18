#!/bin/bash
export MIOPEN_DISABLE_CACHE=1
export HIP_VISIBLE_DEVICES=5
export ENABLE_XFORMERS=0
export ENABLE_TORCHCOMPILE=1

rocprofv2 --sys-trace -o profile_v2_output  \
python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 3 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg"
