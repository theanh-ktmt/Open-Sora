#!/bin/bash

# Set default values if not provided
RESOLUTION=${1:-"144p"}
DURATION=${2:-"2s"}

# Set environment variables
export CUDA_VISIBLE_DEVICES=2
export ENABLE_XFORMERS=0
export DATA_DIR="save/onnx/data/${RESOLUTION}-${DURATION}"
export ONNX_PATH="save/onnx/ckpts/${RESOLUTION}-${DURATION}/stdit3.onnx"
export INFERRED_PATH="save/onnx/ckpts/${RESOLUTION}-${DURATION}/stdit3_inferred.onnx"

# Function to measure time taken for a command
measure_time() {
    local start=$(date +%s)
    "$@"
    local end=$(date +%s)
    echo "Time taken: $((end - start)) seconds"
}

# Main script execution
echo "Preparing input..."
measure_time python scripts/onnx/prepare_input.py --data-dir "$DATA_DIR"

echo "Exporting to ONNX..."
measure_time python scripts/onnx/export_onnx.py --data-dir "$DATA_DIR" --onnx-path "$ONNX_PATH"

echo "Shape Inference ONNX architecture..."
measure_time python scripts/onnx/shape_inference.py --input "$ONNX_PATH" --output "$INFERRED_PATH"

echo "Comparing ONNX outputs..."
measure_time python scripts/onnx/check_onnx.py --data-dir "$DATA_DIR" --onnx-path "$INFERRED_PATH"

echo "Done!"
