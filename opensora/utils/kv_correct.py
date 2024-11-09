import os

import torch
from loguru import logger

KV_CORRECT = None


def prepare_kv_correct(mask, hidden_size=1152, dtype=torch.float, device=torch.device("cpu"), ckpt_dir="save/weights"):
    """Correctness for kv_linear bias."""
    global KV_CORRECT

    KV_CORRECT = {"spatial": [None] * 28, "temporal": [None] * 28}

    B, max_len = mask.shape

    for ckpt in os.listdir(ckpt_dir):
        ckpt_path = os.path.join(ckpt_dir, ckpt)
        kv_linear = torch.load(ckpt_path, map_location=device)
        kv_bias = kv_linear.bias.detach().to(dtype)

        out = torch.zeros(B, 2 * max_len, 2 * hidden_size, dtype=dtype, device=device)

        num_tokens = mask.sum(dim=1).tolist()[0]
        out[:, num_tokens:max_len, :] = -kv_bias
        out[:, max_len + num_tokens :, :] = -kv_bias

        type_ = "spatial" if "spatial" in ckpt else "temporal"
        index = int(ckpt.split(".")[-2])
        KV_CORRECT[type_][index] = out

    logger.info("Done preparing KV correctness!")


def get_kv_correct(index, temporal=False):
    if temporal:
        return KV_CORRECT["temporal"][index]
    else:
        return KV_CORRECT["spatial"][index]
