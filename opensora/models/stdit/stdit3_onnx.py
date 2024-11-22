import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import onnxruntime as ort
import torch
from loguru import logger

from opensora.utils.custom.common import to_numpy, to_tensor


class STDiT3ONNX:
    def __init__(
        self,
        onnx_path: str,
        device: str = "cpu",
        cache_dir: str = "./",
        fp16: bool = False,
        max_workspace_size: int = 10,
    ):
        # Init data
        self.onnx_path = Path(onnx_path)
        logger.info("Onnx path: {}".format(self.onnx_path))

        self.device = device
        self.fp16 = fp16
        self.cache_dir = Path(cache_dir)
        self.max_workspace_size = max_workspace_size
        if self.device == "tensorrt":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("TensorRT cache dir: {}".format(self.cache_dir))

            if self.fp16:
                logger.info("TensorRT FP16 is enabled!")

        # Create session
        logger.info("Initializing ONNX Session...")
        start = time.time()
        self.session = self.create_session()
        logger.success("Done after {:.2f}s!".format(time.time() - start))

    def create_session(self) -> ort.InferenceSession:
        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 1
        # Disable GraphOptimization and leave that for TensorRT
        # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        providers = self.get_providers()
        session = ort.InferenceSession(self.onnx_path, sess_options=sess_opt, providers=providers)
        return session

    def get_providers(self) -> Optional[List[Tuple[Any]]]:
        trt_opts = {
            "trt_fp16_enable": self.fp16,
            "trt_layer_norm_fp32_fallback": True,  # force Pow + Reduce ops in layer norm to FP32
            "trt_max_workspace_size": self.max_workspace_size * 1024 * 1024 * 1024,  # in bytes
            "trt_detailed_build_log": True,
            # Engine cache
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./",
            # For embedding context
            "trt_dump_ep_context_model": True,
            "trt_ep_context_file_path": self.cache_dir,
            # Timing cache
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": self.cache_dir,
            "trt_force_timing_cache": True,
        }

        cuda_opts = {
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": self.max_workspace_size * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }

        DEVICE_MAPPING = {
            "cpu": ("CPUExecutionProvider", {}),
            "cuda": ("CUDAExecutionProvider", cuda_opts),
            "tensorrt": ("TensorrtExecutionProvider", trt_opts),
        }

        if self.device.lower() in DEVICE_MAPPING:
            return DEVICE_MAPPING[self.device.lower()]
        return None

    def __call__(self, **kwargs) -> torch.Tensor:
        # Check inputs before input to session
        logger.info("Start inference...")
        start = time.time()

        inputs = {}
        for input_meta in self.session.get_inputs():
            name = input_meta.name
            assert name in kwargs, "Input '{}' not found!".format(name)
            value = kwargs[name]
            shape = input_meta.shape
            assert list(shape) == list(value.shape), "Shape not matched! {} != {}".format(shape, value.shape)
            inputs[name] = to_numpy(value)

        # run inference
        outputs = self.session.run(None, inputs)
        logger.success("Done after {:.2f}s!".format(time.time() - start))

        device = kwargs["x"].device  # HACK: Hard coded
        output = to_tensor(outputs[0], device)
        return output
