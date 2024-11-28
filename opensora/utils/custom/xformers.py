import os
from typing import List, Optional

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
    attn = torch.matmul(query, key.transpose(-2, -1))
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    attn = F.dropout(attn, p)
    attn = torch.matmul(attn, value)

    # Transpose back to [B, M, H, K]
    return attn.transpose(1, 2)


def block_diagonal_mask(
    q_seqlen: List[int],
    kv_seqlen: List[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Creates a block diagonal mask tensor for attention mechanisms.

    Parameters:
    -----------
    q_seqlen : List[int]
        A list containing the lengths of the query sequences for each batch.
    kv_seqlen : List[int]
        A list containing the lengths of the key/value sequences for each batch.
    dtype : torch.dtype, optional
        The data type of the mask tensor. Default is torch.float32.
    device : torch.device, optional
        The device on which the tensor is allocated. Default is CPU.

    Returns:
    --------
    torch.Tensor
        A tensor of shape (total_q_seqlen, total_kv_seqlen) where each block is filled with 0
        and the rest is filled with -inf.

    Raises:
    -------
    AssertionError
        If the lengths of q_seqlen and kv_seqlen are not equal.
    """

    assert len(q_seqlen) == len(kv_seqlen), "Length of 'q_seqlen' and 'kv_seqlen' must be equal!"

    total_q_seqlen = sum(q_seqlen)
    total_kv_seqlen = sum(kv_seqlen)

    # Initialize mask with -inf
    mask = torch.full((total_q_seqlen, total_kv_seqlen), -float("inf"), dtype=dtype, device=device)

    # Fill diagonal with 0
    q_offset = 0
    kv_offset = 0
    for q_len, kv_len in zip(q_seqlen, kv_seqlen):
        mask[q_offset : q_len + q_offset, kv_offset : kv_len + kv_offset] = 0
        q_offset += q_len
        kv_offset += kv_len

    return mask
