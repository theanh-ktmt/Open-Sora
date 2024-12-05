#!/bin/bash
MIOPEN_DISABLE_CACHE=1 HIP_VISIBLE_DEVICES=7 \
ENABLE_XFORMERS=1 ENABLE_CK=1 \
python3 scripts/benchmark_performance.py \
    configs/anhtt/benchmark_performance.py
