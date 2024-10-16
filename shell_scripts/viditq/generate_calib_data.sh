#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python3 scripts/viditq/generate_calib_data.py configs/anhtt/generate_calib_data.py \
  --num-sampling-steps 30 --flow 5 --aes 6.5 \
  --prompt "a beautiful vocanic mountain" \
  # --reference-path "save/references/sample.jpg"
