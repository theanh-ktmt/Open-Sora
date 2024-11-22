import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
from loguru import logger

from opensora.utils.custom.common import to_numpy, to_tensor


class STDiT3TRT:
    def __init__(self, engine_path: str, verbose: bool = True):
        # prepare trt_logger
        self.trt_logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()

        # load engine from path
        logger.info("Loading TensorRT engine from {}.".format(engine_path))
        self.runtime = trt.Runtime(self.trt_logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # allocate memory buffers
        logger.info("Allocating CPU and GPU memory for inference...")
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path: str):
        """Load/Deserialize TensorRT engine."""
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(self, engine):
        """Allocalte memory for engine inference."""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            # get tensor info
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            # allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # classify inputs and outputs
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(IOElement(name, shape, host_mem, device_mem))
            else:
                outputs.append(IOElement(name, shape, host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, **kwargs) -> torch.Tensor:
        logger.info("Start inference...")

        # for multiple inputs
        # copy input data to GPU
        logger.info("Copy input from CPU to GPU...")
        for input in self.inputs:
            input_data = to_numpy(kwargs[input.name])
            np.copyto(input.host, input_data.ravel())
            cuda.memcpy_htod_async(input.device, input.host, self.stream)

        # map context with corresponding GPU memory buffer
        logger.info("Map context with GPU address...")
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # run inference
        logger.info("Run inference...")
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # for single output
        # copy output to CPU
        logger.info("Copy output back to CPU")
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)

        # synchronize stream
        self.stream.synchronize()

        # post-process output
        output_shape = self.outputs[0].shape
        output_data = np.reshape(self.outputs[0].host, output_shape)

        device = kwargs["x"].device  # HACK: Hard coded
        return to_tensor(output_data, device)


class IOElement:
    """Class of input/output element information.

    Information includes:
    - Tensor name
    - Tensor shape
    - Host memory (CPU)
    - Device memory (GPU)"""

    def __init__(self, name, shape, host_mem, device_mem):
        self.name = name
        self.shape = shape
        self.host = host_mem
        self.device = device_mem
