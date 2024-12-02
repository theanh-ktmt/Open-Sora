#!/bin/bash
GPU=${1:-0}

IS_PROFILING=1 PROFILE_OUTDIR=tmp \
MIOPEN_DISABLE_CACHE=1 \
ENABLE_XFORMERS=1 CUDA_VISIBLE_DEVICES=$GPU \
python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 2s --resolution 144p --aspect-ratio 9:16 \
    --num-sampling-steps 30 --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg"
