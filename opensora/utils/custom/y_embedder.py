from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

Y_EMBEDDER: Optional[nn.Module] = None


def load_y_embedder(path: str, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
    """Load y_embedder, a module of STDiT3."""
    global Y_EMBEDDER

    if Y_EMBEDDER is None:
        Y_EMBEDDER = torch.load(path, map_location=device).to(dtype)
        logger.success("Loaded y_embedder from {}!".format(path))


def get_y_embedder() -> Optional[nn.Module]:
    """Get loaded y_embedder module."""
    return Y_EMBEDDER
