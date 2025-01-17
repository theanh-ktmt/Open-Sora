#!/bin/bash
# NUM_PROMPTS=5 PICK_RANDOM=0 \
HIP_VISIBLE_DEVICES=7 ENABLE_TORCHCOMPILE=1 \
python3 scripts/benchmark_performance.py configs/opensora-v1-2/inference/sample.py \
	--flash-attn "true" --layernorm-kernel "false" \
	--dtype "fp16" --save-dir "save/benchmark"
