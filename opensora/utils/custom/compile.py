import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger

ENABLE_TORCHCOMPILE: Optional[bool] = None


def is_torch_compile_enabled() -> bool:
    """Check if torch.compile is enabled or not."""
    global ENABLE_TORCHCOMPILE
    if ENABLE_TORCHCOMPILE is not None:
        return ENABLE_TORCHCOMPILE

    ENABLE_TORCHCOMPILE = os.environ.get("ENABLE_TORCHCOMPILE", "0") == "1"
    logger.info("Enable torch.compile: {}".format(ENABLE_TORCHCOMPILE))

    return ENABLE_TORCHCOMPILE


def compile_module(
    module: nn.Module, configs: Optional[Dict[str, Any]] = None, cache_path: str = "save/cache/model.compiled.pth"
):
    """Implement torch.compile for an nn.Module."""
    # Hook custom kernels to after 'torch.compile'
    # from torch._inductor.select_algorithm import extern_kernels

    # def trace_fn_name(fn):
    #     def wrapper(*args, **kwargs):
    #         # do some side effect to see that the hook is working
    #         print(f"Patched: {fn.__name__=}")
    #         return fn(*args, **kwargs)

    #     return wrapper

    # assert extern_kernels.mm is torch.mm
    # assert extern_kernels.addmm is torch.addmm
    # extern_kernels.addmm = trace_fn_name(extern_kernels.addmm)
    # extern_kernels.mm = trace_fn_name(extern_kernels.mm)

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
