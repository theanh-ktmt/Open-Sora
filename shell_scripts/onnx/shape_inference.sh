#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python -m onnxruntime.tools.symbolic_shape_infer \
    --input save/onnx/ckpt/stdit3_inferred.onnx \
    --output save/onnx/ckpt/stdit3_symbolic_inferred.onnx \
    --auto_merge --verbose 3
