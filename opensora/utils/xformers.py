import os
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger

ENABLE_XFORMERS: Optional[bool] = None


def is_xformers_enabled() -> bool:
    """Check if xformers is enabled or not."""
    global ENABLE_XFORMERS
    if ENABLE_XFORMERS is not None:
        return ENABLE_XFORMERS

    ENABLE_XFORMERS = os.environ.get("ENABLE_XFORMERS", "1") == "1"
    logger.info("Enable xformers: {}".format(ENABLE_XFORMERS))

    return ENABLE_XFORMERS


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    p: float = 0.0,
    attn_bias: Optional[torch.Tensor] = None,
):
    """Equivalent torch implementation for xformers.ops.memory_efficient_attentions.

    Input shape (QKV) should [B, M, H, K] where:
    - B is the batch size
    - M is the sequence length
    - H is the number of heads
    - K the embeding size per head
    Attention bias shape should be [B, H, M, M] where M should be divided by 8.

    Args:
        query (torch.Tensor): Query tensor of shape [B, M, H, K]
        key (torch.Tensor): Key tensor of shape [B, M, H, K]
        value (torch.Tensor): Value tensor of shape [B, M, H, K]
        p (float): Dropout probability. Default is 0.0.
        attn_bias (Optional[torch.Tensor]): Optional attention bias tensor of shape [B, H, M, M]

    Returns:
        torch.Tensor: The output tensor after applying memory-efficient attention, of shape [B, M, H, K]
    """
    # Scale query tensor
    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale

    # Transpose to [B, H, M, K]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Calculate attention
    attn = query @ key.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    attn = F.dropout(attn, p)
    attn = attn @ value

    # Transpose back to [B, M, H, K]
    return attn.transpose(1, 2)
