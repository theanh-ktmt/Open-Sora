import onnx

# Load the ONNX model
path = "stdit3_simplified.onnx"
model = onnx.load(path)
onnx.checker.check_model(path)

# Print model information
print(f"IR Version: {model.ir_version}")
print(f"Producer Name: {model.producer_name}")
print(f"Producer Version: {model.producer_version}")
print(f"Model Version: {model.model_version}")
print(f"Graph Name: {model.graph.name}")

# Check the nodes and initializers
for node in model.graph.node:
    print(f"Node: {node.name}, OpType: {node.op_type}")

for initializer in model.graph.initializer:
    print(f"Initializer: {initializer.name}, Shape: {initializer.dims}")
