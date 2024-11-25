import pprint

import onnx


def get_int64_weight_layers(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    int64_layers = []

    # Iterate through the model's initializers
    for initializer in model.graph.initializer:
        # Check if the data type is int64
        if initializer.data_type == onnx.TensorProto.INT64:
            int64_layers.append(initializer.name)

    return int64_layers


# Example usage
model_path = "save/onnx/ckpts/720p-4s/stdit3_simplified.onnx"
int64_layers = get_int64_weight_layers(model_path)
print("Layers with int64 weights:", pprint.pformat(int64_layers))
