#!/bin/bash
IS_PROFILING=1 PROFILE_OUTDIR=save/profile/tensorrt/a100-fp16-mp/720p-4s/torch_profile \
CUDA_VISIBLE_DEVICES=5 ENABLE_XFORMERS=0 ENABLE_TENSORRT=1 \
python3 scripts/inference.py configs/anhtt/inference.py \
    --num-frames 4s --resolution 720p --num-sampling-steps 3 \
    --prompt "a beautiful vocanic mountain" \
    --reference-path "save/references/sample.jpg" \
    --dtype "fp32" --flash-attn "false" --layernorm-kernel "false" \
	--trt-engine-path "save/tensorrt/720p-4s/stdit3_fp32_mp.engine"
