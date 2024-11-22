import argparse

import tensorrt as trt
from loguru import logger

from opensora.utils.custom.tensorrt import build_engine

# All layer types in OpenSora
ALL_LAYER_TYPES = [
    trt.LayerType.ACTIVATION,
    trt.LayerType.CAST,
    trt.LayerType.CONCATENATION,
    trt.LayerType.CONSTANT,
    trt.LayerType.CONVOLUTION,
    trt.LayerType.ELEMENTWISE,
    trt.LayerType.MATRIX_MULTIPLY,
    trt.LayerType.NORMALIZATION,
    trt.LayerType.REDUCE,
    trt.LayerType.SELECT,
    trt.LayerType.SHUFFLE,
    trt.LayerType.SLICE,
    trt.LayerType.SOFTMAX,
    trt.LayerType.UNARY,
]


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx-path", type=str, required=True, help="Path to ONNX file")
    parser.add_argument("--engine-path", type=str, required=True, help="Path to save engine")
    parser.add_argument("--workspace-size", type=int, default=80, help="Size in GB use for temporary storing")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Build engine
    logger.info("Bulding TensorRT engine...")
    build_engine(
        args.onnx_path,  # onnx path
        args.engine_path,  # path to save engine
        workspace_size=args.workspace_size,  # available work space
        fp16_layers=[
            trt.LayerType.CONVOLUTION,
            trt.LayerType.ELEMENTWISE,
            trt.LayerType.MATRIX_MULTIPLY,
        ],
        strict=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
