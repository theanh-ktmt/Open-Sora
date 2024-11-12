import os

import torch
import torch.nn as nn
from loguru import logger

MHA_KVS = None


def add_padding(tensor, dim, len=600):
    """Add padding to a tensor in a specific dimension."""
    if tensor.shape[dim] >= len:
        return tensor

    padded_shape = list(tensor.shape)
    padded_shape[dim] = len - tensor.shape[dim]
    padding = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
    padded_tensor = torch.cat((tensor, padding), dim=dim)
    return padded_tensor


def prepare_mha_kv(
    y, mask, hidden_size=1152, num_heads=16, dtype=torch.float32, device=torch.device("cpu"), ckpt_dir="save/weights"
):
    """Prepare KV for Multi-head Attention."""
    global MHA_KVS
    MHA_KVS = {"spatial": [None] * 28, "temporal": [None] * 28}

    _, max_len = mask.shape
    y = y.detach().to(device)

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
