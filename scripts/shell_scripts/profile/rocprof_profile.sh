#!/bin/bash
export HIP_VISIBLE_DEVICES=7
export ENABLE_TORCHCOMPILE=1

rocprof --sys-trace -o profile_output.csv \
python3 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 3 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg" \
	--flash-attn "true" --layernorm-kernel "false" \
	--dtype "fp16" --save-dir "save/inference/test"
