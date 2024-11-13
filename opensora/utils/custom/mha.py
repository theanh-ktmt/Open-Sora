import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from opensora.utils.custom.common import add_padding, get_dynamic_size
from opensora.utils.custom.xformers import block_diagonal_mask

MHA_KVS: Optional[Dict[str, Any]] = None
MHA_BIAS: Optional[torch.tensor] = None


# For KV
def prepare_mha_kv(
    y, mask, hidden_size=1152, num_heads=16, dtype=torch.float32, device=torch.device("cpu"), ckpt_dir="save/weights"
):
    """Prepare KV for Multi-head Attention."""
    global MHA_KVS
    MHA_KVS = {"spatial": [None] * 28, "temporal": [None] * 28}

    _, max_len = mask.shape
    y = y.detach().to(device)

    logger.info("Loading kv_linear weights from {}...".format(ckpt_dir))
    for ckpt in os.listdir(ckpt_dir):
        ckpt_path = os.path.join(ckpt_dir, ckpt)
        state_dict = torch.load(ckpt_path)

        kv_linear = nn.Linear(hidden_size, hidden_size * 2, dtype=dtype, device=device)
        kv_linear.load_state_dict(state_dict)
        kv_linear.eval()

        with torch.no_grad():
            kv = kv_linear(y).view(1, -1, 2, num_heads, hidden_size // num_heads)
            k, v = kv.unbind(2)

            # add padding
            k = add_padding(k, dim=1, len=max_len * 2)
            v = add_padding(v, dim=1, len=max_len * 2)

        type_ = "spatial" if "spatial" in ckpt else "temporal"
        index = int(ckpt.split(".")[-2])
        MHA_KVS[type_][index] = [k, v]
        del kv, k, v, kv_linear, state_dict
        torch.cuda.empty_cache()

    logger.success("Done preparing KV for Multi-head Attention!")


def get_mha_kv(index, temporal=False):
    if temporal:
        return MHA_KVS["temporal"][index]
    else:
        return MHA_KVS["spatial"][index]


# For bias
def prepare_mha_bias(
    input: torch.Tensor, mask: torch.Tensor, y_lens: List[int], dtype: torch.dtype, device: torch.device
) -> None:
    """Pre-compute the bias for cross attention module."""
    global MHA_BIAS

    T, H, W = get_dynamic_size(input)
    B = len(y_lens)
    max_len = mask.shape[1]

    # Prepare block diagonal mask
    config_size = T * H * W  # Example value: 2160

    # HACK: Uncomment to skip padding
    # return block_diagonal_mask([config_size] * B, y_lens, dtype, device)

    # Calculate the number of tokens per batch
    num_tokens = mask.sum(dim=1).tolist()[0]  # Example value: 21
    attn_bias = block_diagonal_mask([config_size] * B, [num_tokens] * B, dtype, device)

    # Calculate the padded length
    padded_len = B * max_len - B * num_tokens
    MHA_BIAS = F.pad(attn_bias, (0, padded_len), mode="constant", value=-float("inf"))
    logger.success("Done preparing bias for Multi-head Attention!")


def get_mha_bias() -> Optional[torch.tensor]:
    """Get bias for Multi-head CrossAttention module."""
    return MHA_BIAS
