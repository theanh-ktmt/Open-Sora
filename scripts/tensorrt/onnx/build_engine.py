import time

from loguru import logger

import tensorrt as trt

TRT_LOGGER = trt.Logger()


def build_serialized_engine(onnx_file_path, save_path, workspace_size=1):
    start = time.time()
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        explicit_batch_flag
    ) as network, builder.create_builder_config() as config:
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_file_path, "rb") as model:
            logger.info("Parsing ONNX file...")
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return

        # Find and mark the correct output layer
        last_layer = network.get_layer(network.num_layers - 1)
        if last_layer is not None:
            network.mark_output(last_layer.get_output(0))
        else:
            logger.error("Failed to retrieve the last layer.")
            return

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1024 * 1024))

        if builder.platform_has_fast_fp16:
            logger.info("Enable FP16 inference.")
            config.set_flag(trt.BuilderFlag.FP16)

        logger.info("Building an engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("Failed to build the serialized engine.")
            return

        logger.info("Completed creating Engine after {:.2f}s".format(time.time() - start))

        with open(save_path, "wb") as f:
            f.write(serialized_engine)
        logger.info("Write serialize engine to {}!".format(save_path))


ONNX_FILE_PATH = "save/onnx/ckpt/stdit3.onnx"
TENSORRT_PATH = "save/tensorrt/144p_2s/stdit3.engine"
WORKSPACE_SIZE = 10

build_serialized_engine(ONNX_FILE_PATH, TENSORRT_PATH)
logger.success("Saved TensorRT engine to {}!".format(TENSORRT_PATH))
