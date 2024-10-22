import onnx

# Load the ONNX model
model_path = "save/onnx/ckpt/stdit3.onnx"
model = onnx.load(model_path)

# Print the opset version(s) used by the model
if len(model.opset_import) > 0:
    for opset in model.opset_import:
        print(f"Opset domain: {opset.domain}, version: {opset.version}")
else:
    print("No opset information found in the model.")
