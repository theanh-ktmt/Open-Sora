#!/bin/bash
export MIOPEN_DISABLE_CACHE=1
export HIP_VISIBLE_DEVICES=6
export ENABLE_XFORMERS=0
export ENABLE_TORCHCOMPILE=1

rocprof --sys-trace -o profile_output.csv \
python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 3 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg"
