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
