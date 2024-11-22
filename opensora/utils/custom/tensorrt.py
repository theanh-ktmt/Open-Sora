import os
import time
from typing import List, Optional

import tensorrt as trt
from loguru import logger

ENABLE_TENSORRT: Optional[bool] = None


def is_tensorrt_enabled() -> bool:
    """Check if TensorRT is enabled or not."""
    global ENABLE_TENSORRT
    if ENABLE_TENSORRT is not None:
        return ENABLE_TENSORRT

    ENABLE_TENSORRT = os.environ.get("ENABLE_TENSORRT", "0") == "1"
    logger.info("Enable TensorRT: {}".format(ENABLE_TENSORRT))

    return ENABLE_TENSORRT


def build_engine(
    onnx_file_path: str,
    save_path: str,
    workspace_size: int = 1,
    use_fp16: bool = False,
    layers_to_keep_fp32: List[str] = [],
    verbose: bool = True,
):
    """Build/Serialize TensorRT engine."""
    # Initialize Logger
    trt_logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()

    start = time.time()
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # Initialize TensorRT engine and parse ONNX model
    with trt.Builder(trt_logger) as builder, builder.create_network(
        explicit_batch_flag
    ) as network, builder.create_builder_config() as config:
        # Parse ONNX
        parser = trt.OnnxParser(network, trt_logger)
        with open(onnx_file_path, "rb") as model:
            logger.info("Begin parsing ONNX file...")
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.info(parser.get_error(error))
                return None
        logger.info("Completed parsing ONNX model")

        # Config builder
        logger.info("Prepare builder config...")
        # Allow TensorRT to use up to workspace_size GB of GPU memory for tactic selection
        if workspace_size != 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1024 * 1024 * 1024))
        # Use FP16 mode if possible
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Enforce certain layers to remain in FP32
        for layer_name in layers_to_keep_fp32:
            layer = network.get_layer(network.get_layer_index(layer_name))
            layer.precision = trt.DataType.FLOAT
            layer.set_output_type(0, trt.DataType.FLOAT)

        # Generate TensorRT engine optimized for the target platform
        logger.info("Building engine...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.info("Failed to create the engine.")
            return None
        logger.info("Completed creating Engine after {:.2f}s".format(time.time() - start))

        with open(save_path, "wb") as f:
            f.write(serialized_engine)
        logger.info("Write serialized engine to {}!".format(save_path))
