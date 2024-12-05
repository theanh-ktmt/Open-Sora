#!/bin/bash
MIOPEN_DISABLE_CACHE=1 ENABLE_XFORMERS=1 HIP_VISIBLE_DEVICES=6 \
python3 scripts/benchmark_performance.py \
    configs/anhtt/benchmark_performance.py
