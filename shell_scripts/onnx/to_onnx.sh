#!/bin/bash

# Set default values if not provided
GPU=${1:-0}
RESOLUTION=${2:-"144p"}
DURATION=${3:-"2s"}
DTYPE=${4:-"fp32"}

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU
export ENABLE_XFORMERS=0
export DATA_DIR="save/onnx/data/${RESOLUTION}-${DURATION}"
export ONNX_PATH="save/onnx/ckpts/${RESOLUTION}-${DURATION}/stdit3.onnx"
export INFERRED_PATH="save/onnx/ckpts/${RESOLUTION}-${DURATION}/stdit3_inferred.onnx"
export CACHE_DIR="save/onnx/cache/${RESOLUTION}-${DURATION}-${DTYPE}"

# Function to measure time taken for a command
measure_time() {
    local start=$(date +%s)
    "$@"
    local end=$(date +%s)
    echo "Time taken: $((end - start)) seconds"
}

# Run ONNX on FP32
echo "Preparing input..."
measure_time python scripts/onnx/prepare_input.py --data-dir "$DATA_DIR"

echo "Exporting to ONNX..."
measure_time python scripts/onnx/export_onnx.py --data-dir "$DATA_DIR" --onnx-path "$ONNX_PATH"

echo "Shape Inference ONNX architecture..."
measure_time python scripts/onnx/shape_inference.py --input "$ONNX_PATH" --output "$INFERRED_PATH"

# Run TensorRT with desired precision
echo "Comparing ONNX outputs..."
CMD="measure_time python scripts/onnx/check_onnx.py \
    --data-dir \"$DATA_DIR\" \
    --onnx-path \"$INFERRED_PATH\" \
    --cache-dir \"$CACHE_DIR\""
if [ "$DTYPE" == "fp16" ]; then
    CMD+=" --fp16"
fi
eval $CMD

echo "Done!"
