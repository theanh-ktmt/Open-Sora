#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python3 scripts/benchmark_performance.py \
	configs/anhtt/benchmark_performance.py
