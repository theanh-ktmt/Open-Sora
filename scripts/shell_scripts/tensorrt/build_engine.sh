#!/bin/bash
GPU=${1:-0}
RESOLUTION=${2:-"720p"}
DURATION=${3:-"4s"}
DTYPE=${4:-"fp16"}

export CUDA_VISIBLE_DEVICES=$GPU
export INFERRED_PATH="save/onnx/ckpts/${RESOLUTION}-${DURATION}/stdit3_inferred.onnx"
export SIMPLIFIED_PATH="save/onnx/ckpts/${RESOLUTION}-${DURATION}/stdit3_simplified.onnx"
export ENGINE_PATH="save/tensorrt/${RESOLUTION}-${DURATION}/stdit3.engine"

# Function to measure time taken for a command
measure_time() {
    local start=$(date +%s)
    "$@"
    local end=$(date +%s)
    echo "Time taken: $((end - start)) seconds"
}

echo "Building TensorRT engine..."
cp "$SIMPLIFIED_PATH.data" "stdit3_simplified.onnx.data" # for TensorRT reference
CMD="measure_time python scripts/tensorrt/build_engine.py \
    --onnx-path $SIMPLIFIED_PATH \
    --engine-path $ENGINE_PATH \
    --workspace-size 80"
if [ "$DTYPE" == "fp16" ]; then
    CMD+=" --fp16"
fi
eval $CMD
rm "stdit3_simplified.onnx.data"

echo "Done! TensorRT engine saved at $ENGINE_PATH."
