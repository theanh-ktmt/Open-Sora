#!/bin/bash
export CKPT_PATH=hpcai-tech/OpenSora-STDiT-v3
export MODEL_NAME=backbone_fp16_text_fp32
export START_INDEX=0
export END_INDEX=2000

CUDA_VISIBLE_DEVICES=2 \
bash eval/sample.sh \
	$CKPT_PATH 51 $MODEL_NAME -4 $START_INDEX $END_INDEX
