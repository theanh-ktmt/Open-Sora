from typing import Tuple

import torch


def add_padding(tensor, dim, len=600):
    """Add padding to a tensor in a specific dimension."""
    if tensor.shape[dim] >= len:
        return tensor

    padded_shape = list(tensor.shape)
    padded_shape[dim] = len - tensor.shape[dim]
    padding = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
    padded_tensor = torch.cat((tensor, padding), dim=dim)
    return padded_tensor


def get_dynamic_size(x, patch_size=(1, 2, 2)) -> Tuple[int, int, int]:
    _, _, T, H, W = x.size()
    if T % patch_size[0] != 0:
        T += patch_size[0] - T % patch_size[0]
    if H % patch_size[1] != 0:
        H += patch_size[1] - H % patch_size[1]
    if W % patch_size[2] != 0:
        W += patch_size[2] - W % patch_size[2]
    T = T // patch_size[0]
    H = H // patch_size[1]
    W = W // patch_size[2]
    return (T, H, W)
