from typing import Optional

import torch
from loguru import logger

from opensora.utils.custom.common import get_dynamic_size

POS_EMB: Optional[torch.tensor] = None


def prepare_pos_emb(
    x: torch.tensor, height: torch.tensor, width: torch.tensor, hidden_size: int = 1152, input_sq_size: int = 512
) -> None:
    """Prepare position embedding for STDiT3 model."""
    global POS_EMB

    from opensora.models.layers.blocks import PositionEmbedding2D

    pos_embed = PositionEmbedding2D(hidden_size).to(x.device, x.dtype)
    _, H, W = get_dynamic_size(x)

    S = H * W
    base_size = round(S**0.5)
    resolution_sq = (height[0].item() * width[0].item()) ** 0.5
    scale = resolution_sq / input_sq_size

    # handle device, dtype in pos_emb forward
    POS_EMB = pos_embed(x, H, W, scale=scale, base_size=base_size)
    logger.success("Done preparing position embedding for STDiT3 model!")
    logger.info("Position Embedding: {} {}".format(POS_EMB.shape, POS_EMB.device))


def get_pos_emb() -> Optional[torch.tensor]:
    """Get prepared position embedding for STDiT3 model."""
    return POS_EMB
