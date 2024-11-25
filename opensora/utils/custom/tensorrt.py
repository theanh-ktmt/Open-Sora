import os
import time
from typing import List, Optional

import onnx
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
    workspace_size: int = 80,
    fp16_layers: List[trt.LayerType] = [],
    strict: bool = False,
    verbose: bool = False,
):
    """
    Build and Serialize a TensorRT engine from an ONNX file.

    Args:
        onnx_file_path (str): Path to the ONNX model file.
        save_path (str): Path to save the serialized TensorRT engine.
        workspace_size (int): Maximum GPU memory (in GB) that TensorRT can use for tactic selection.
        fp16_layers (List[trt.LayerType]): List of layers to be kept at FP32.
        strict (bool): If True, enforce strict precision constraints. If False, prefer precision constraints.
        verbose (bool): If True, enable verbose logging.

    Returns:
        None: If the engine creation fails, the function returns None.
    """
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
            logger.info("- Workspace size = {}GB".format(workspace_size))
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1024 * 1024 * 1024))

        # Enable FP16 if specified
        if len(fp16_layers) > 0:
            logger.info("- Enable FP16".format(workspace_size))
            config.set_flag(trt.BuilderFlag.FP16)

        # TensorRT behavior towards the precision constraints
        constraint_flag = (
            trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS if strict else trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS
        )
        logger.info("- Turn on '{}'".format(constraint_flag))
        config.set_flag(constraint_flag)

        # Precision constraints
        int_or_bool_layers = get_int_or_bool_layers(onnx_file_path)
        for i in range(network.num_layers):
            layer = network.get_layer(i)

            if (
                layer.name in int_or_bool_layers
                or "ONNXTRT_ShapeTensorFromDims" in layer.name
                or layer.type == trt.LayerType.CAST
            ):
                # if verbose:
                logger.info(f"Skip layer '{layer.name}' ({layer.type})")
                continue

            if layer.type in fp16_layers:
                layer.precision = trt.DataType.HALF
                for j in range(layer.num_outputs):
                    # element-wise output has to be FLOAT
                    layer.set_output_type(
                        j, trt.DataType.HALF if "ONNXTRT_ShapeElementWise" not in layer.name else trt.DataType.FLOAT
                    )
                if verbose:
                    logger.info(f"Configure layer '{layer.name}' ({layer.type}) at {trt.DataType.HALF}")
            else:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)
                if verbose:
                    logger.info(f"Configure layer '{layer.name}' ({layer.type}) at {trt.DataType.FLOAT}")

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


def get_int_or_bool_layers(onnx_path):
    """Return names of layers that have any integer data type or BOOL data type."""
    int_or_bool_layers = []

    int_types = [onnx.TensorProto.INT8, onnx.TensorProto.INT16, onnx.TensorProto.INT32, onnx.TensorProto.INT64]

    model = onnx.load(onnx_path)
    for initializer in model.graph.initializer:
        if initializer.data_type in int_types or initializer.data_type == onnx.TensorProto.BOOL:
            int_or_bool_layers.append(initializer.name)

    return int_or_bool_layers
