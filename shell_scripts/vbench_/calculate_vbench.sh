#!/bin/bash
VIDEO_DIR=save/eval/backbone_fp16_text_fp32/generated_videos_vbench/

CUDA_VISIBLE_DEVICES=2 \
python eval/vbench/calc_vbench.py $VIDEO_DIR
