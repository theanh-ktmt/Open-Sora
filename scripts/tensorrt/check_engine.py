import argparse

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from loguru import logger

from opensora.utils.custom.tensorrt import load_engine


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-path", type=str, required=True, help="Path to built engine file")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to inputs and configs.")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load engine
    logger.info("Loading pre-built engine from {}...".format(args.engine_path))
    engine = load_engine(args.engine_path)

    for idx, _tensor in enumerate(engine):  # inputs and outputs
        name = engine.get_tensor_name(idx)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # in case it is input
            input_shape = engine.get_tensor_shape(_tensor)
            input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize  # in bytes
            cuda.mem_alloc(input_size)
        else:  # output
            output_shape = engine.get_tensor_shape(_tensor)

            # Create page-locked memory buffer (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            cuda.mem_alloc(host_output.nbytes)


if __name__ == "__main__":
    main()
