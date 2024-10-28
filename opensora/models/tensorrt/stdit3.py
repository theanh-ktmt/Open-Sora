import time
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch

from opensora.utils.misc import get_logger

logger = get_logger()


class STDiT3TRT:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, onnx_path: str, cache_dir: str, fp16: bool = False, max_workspace_size: int = 10):
        if not hasattr(self, "_initialized"):
            self._initialized = True

            # Init data
            self.onnx_path = Path(onnx_path)
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Onnx path: {}".format(self.onnx_path))
            logger.info("Cache dir: {}".format(self.cache_dir))

            self.fp16 = fp16
            if self.fp16:
                logger.info("TensorRT FP16 is enabled!")
            self.max_workspace_size = max_workspace_size

            # Create session
            logger.info("Initializing TensorRT Session...")
            start = time.time()
            self.session = self.create_session()
            logger.info("Done after {:.2f}s!".format(time.time() - start))

    def create_session(self) -> ort.InferenceSession:
        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 1
        # Disable GraphOptimization and leave that for TensorRT
        # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        providers = self.get_providers()
        session = ort.InferenceSession(self.onnx_path, sess_options=sess_opt, providers=providers)
        return session

    def get_providers(self) -> List[Tuple[Any]]:
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

        providers = [
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", cuda_opts),
        ]

        return providers

    def __call__(
        self,
        z_in: torch.Tensor,
        t_in: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        height: Optional[torch.Tensor] = None,
        width: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if mask is not None and x_mask is not None:
            warnings.warn(
                "Masking isn't supported in TensorRT at the moment. mask and x_mask will be ignored.", UserWarning
            )

            mask = None
            x_mask = None

        device = z_in.device
        inputs = {
            "z_in": to_numpy(z_in),
            "t_in": to_numpy(t_in),
            "y": to_numpy(y),
            # "mask": to_numpy(mask),
            # "x_mask": to_numpy(x_mask),
            "fps": to_numpy(fps),
            "height": to_numpy(height),
            "width": to_numpy(width),
        }

        # Check inputs before input to session
        for input_meta in self.session.get_inputs():
            name = input_meta.name
            assert name in inputs, "Input '{}' not found!".format(name)
            shape = input_meta.shape
            assert list(shape) == list(inputs[name].shape), "Shape not matched! {} != {}".format(
                inputs[name].shape, shape
            )
            # logger.info("Input '{}' shape: {}".format(name, shape))

        outputs = self.session.run(None, inputs)
        output = to_tensor(outputs[0], device)
        return output


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch.Tensor to numpy array."""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def to_tensor(array: np.ndarray, device: torch.device = torch.device("cpu")):
    """Convert numpy array to torch.Tensor."""
    return torch.from_numpy(array).to(device)
