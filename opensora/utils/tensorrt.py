import os
from typing import Optional

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
