#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export ENABLE_XFORMERS=0
export DATA_DIR=save/onnx/data
export ONNX_PATH=save/onnx/ckpt/stdit3.onnx
export INFERRED_PATH=save/onnx/ckpt/stdit3_inferred.onnx

echo "Preparing input..."
start=$(date +%s)
python scripts/onnx/prepare_input.py --data-dir $DATA_DIR
end=$(date +%s)
echo "Time taken: $(($end - $start)) seconds"

echo "Exporting to ONNX..."
start=$(date +%s)
python scripts/onnx/export_onnx.py --data-dir $DATA_DIR --onnx-path $ONNX_PATH
end=$(date +%s)
echo "Time taken: $(($end - $start)) seconds"

echo "Shape Inference ONNX architecture..."
start=$(date +%s)
python scripts/onnx/shape_inference.py --input $ONNX_PATH --output $INFERRED_PATH
end=$(date +%s)
echo "Time taken: $(($end - $start)) seconds"

echo "Comparing ONNX outputs..."
start=$(date +%s)
python scripts/onnx/check_onnx.py --data-dir $DATA_DIR --onnx-path $INFERRED_PATH
end=$(date +%s)
echo "Time taken: $(($end - $start)) seconds"



echo "Done!"
