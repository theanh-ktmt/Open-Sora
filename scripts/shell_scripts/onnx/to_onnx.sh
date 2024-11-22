#!/bin/bash

# Set default values if not provided
GPU=${1:-0}
RESOLUTION=${2:-"720p"}
DURATION=${3:-"4s"}

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU
export ENABLE_XFORMERS=0
export DATA_DIR="save/onnx/data/${RESOLUTION}-${DURATION}"
export ONNX_PATH="save/onnx/ckpts/${RESOLUTION}-${DURATION}/stdit3.onnx"
export INFERRED_PATH="save/onnx/ckpts/${RESOLUTION}-${DURATION}/stdit3_inferred.onnx"
export SIMPLIFIED_PATH="save/onnx/ckpts/${RESOLUTION}-${DURATION}/stdit3_simplified.onnx"

# Function to measure time taken for a command
measure_time() {
    local start=$(date +%s)
    "$@"
    local end=$(date +%s)
    echo "Time taken: $((end - start)) seconds"
}

# Run ONNX on FP32
# echo "Preparing input..."
# measure_time python scripts/onnx/prepare_input.py --data-dir "$DATA_DIR" --resolution $RESOLUTION --duration $DURATION

# echo "Exporting to ONNX..."
# measure_time python scripts/onnx/export_onnx.py --data-dir "$DATA_DIR" --onnx-path "$ONNX_PATH"

# echo "Shape Inference ONNX architecture..."
# measure_time python scripts/onnx/shape_inference.py --input "$ONNX_PATH" --output "$INFERRED_PATH"

# echo "Simplify ONNX graph..."
# measure_time onnxsim $INFERRED_PATH $SIMPLIFIED_PATH

# Run TensorRT with desired precision
echo "Comparing ONNX outputs..."
measure_time python scripts/onnx/check_onnx.py \
    --data-dir "$DATA_DIR" \
    --onnx-path "$SIMPLIFIED_PATH"


echo "Done! ONNX saved at $SIMPLIFIED_PATH."
