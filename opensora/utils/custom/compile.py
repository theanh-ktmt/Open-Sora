import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger

ENABLE_TORCHCOMPILE = os.environ.get("ENABLE_TORCHCOMPILE", "0") == "1"
CUSTOM_BACKEND = os.environ.get("CUSTOM_BACKEND", None)

logger.info("Enable torch.compile: {}".format(ENABLE_TORCHCOMPILE))
logger.info("Customed backend: {}".format(CUSTOM_BACKEND))


def is_torch_compile_enabled() -> bool:
    """Check if torch.compile is enabled or not."""
    return ENABLE_TORCHCOMPILE


def get_custom_backend() -> str:
    """Get customed backend used for our customed ops."""
    return CUSTOM_BACKEND


def hook_after_compiled():
    """Replace original torch ops with our customed ops."""
    from torch._inductor.select_algorithm import extern_kernels

    assert extern_kernels.mm is torch.mm
    assert extern_kernels.addmm is torch.addmm

    if CUSTOM_BACKEND is None:

        def custom_func(fn):
            def wrapper(*args, **kwargs):
                # for check hooking
                # print(f"{fn.__name__=}")

                # for shape inference
                # for tensor in filter(lambda x: isinstance(x, torch.Tensor), args):
                #     print(f"{tensor.shape=}, {tensor.stride()=} {tensor.dtype=}")
                return fn(*args, **kwargs)

            return wrapper

        extern_kernels.addmm = custom_func(extern_kernels.addmm)
        extern_kernels.mm = custom_func(extern_kernels.mm)
        logger.info("Skipped hooking after compiled because no backend provided.")

    elif CUSTOM_BACKEND == "hipblaslt":
        from modiffusion.ops.hipblaslt_gemm import hipblaslt_addmm_out, hipblaslt_mm_out

        extern_kernels.addmm = hipblaslt_addmm_out
        extern_kernels.mm = hipblaslt_mm_out
        logger.info("Done hooked after compile for backend '{}'.".format(CUSTOM_BACKEND))
    else:
        raise NotImplementedError(
            "Backend '{}' is currently not supported for hooking after 'torch.compile'!".format(CUSTOM_BACKEND)
        )


def compile_module(
    module: nn.Module, configs: Optional[Dict[str, Any]] = None, cache_path: str = "save/cache/model.compiled.pth"
):
    """Implement torch.compile for an nn.Module."""
    # hook custom kernels to after 'torch.compile'
    hook_after_compiled()

    # prepare configs
    DEFAULT_CONFIGS = {
        "mode": "default",
        # "mode": "max-autotune",   # not working
        # "fullgraph": True,        # not working
    }
    configs = DEFAULT_CONFIGS if configs is None else configs

    # start compile
    logger.info("Start compiling...")
    return torch.compile(module, **configs)
