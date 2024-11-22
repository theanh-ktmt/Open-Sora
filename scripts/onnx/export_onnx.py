import argparse
import os
import pprint

import torch
from loguru import logger

from opensora.registry import MODELS, build_module
from opensora.utils.custom.pos_emb import prepare_pos_emb

# Load argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx-path", type=str, default="save/onnx/ckpts/144p-2s/stdit3.onnx", help="Path to save ONNX file."
)
parser.add_argument("--data-dir", type=str, default="save/onnx/data/144p-2s", help="Path to inputs and configs.")
args = parser.parse_args()

# Create save dir
os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)


# Settings
device = torch.device("cpu")
dtype = torch.float32


# Build models
config_path = os.path.join(args.data_dir, "configs.pth")
logger.info("Loading configs from {}...".format(config_path))
configs = torch.load(config_path)
logger.info("Configs:\n{}".format(pprint.pformat(configs)))

logger.info("Building diffusion model...")
model = (
    build_module(
        configs["model_cfg"],
        MODELS,
        input_size=configs["input_size"],
        in_channels=configs["in_channels"],
        caption_channels=configs["caption_channels"],
        model_max_length=configs["model_max_length"],
        enable_sequence_parallelism=False,
    )
    .to(device, dtype)
    .eval()
)
logger.info("Model:\n{}".format(model))

# Unpack inputs
input_path = os.path.join(args.data_dir, "inputs.pth")
logger.info("Loading inputs from {}...".format(input_path))
inputs = torch.load(input_path, map_location=device)

z_in = inputs["z_in"]
t_in = inputs["t_in"]
fps = inputs["fps"]
mha_kvs = inputs["mha_kvs"]
mha_bias = inputs["mha_bias"]
height = inputs["height"]
width = inputs["width"]
logger.info(
    f"""Inputs:
    z_in: {z_in.shape} {z_in.dtype} {z_in.device}
    t_in: {t_in.shape} {t_in.dtype} {t_in.device}
    attn_kv: {mha_kvs["mha_s00_k"].shape} {mha_kvs["mha_s00_k"].dtype} {mha_kvs["mha_s00_k"].device}
    attn_bias: {mha_bias.shape} {mha_bias.dtype} {mha_bias.device}
    fps: {fps.shape} {fps.dtype} {fps.device}
    height: {height.shape} {height.dtype} {height.device}
    width: {width.shape} {width.dtype} {width.device}"""
)

# Prepare additional informations
logger.info("Preparing positional embedding...")
hidden_size = 1152
num_heads = 16
input_sq_size = 512

prepare_pos_emb(
    z_in[0].unsqueeze(0),
    height,
    width,
    hidden_size=hidden_size,
    input_sq_size=input_sq_size,
)

# Convert to ONNX
logger.info("Exporting ONNX models...")
# NOTE: Skip dynamic shapes
# dynamic_axes = {
#     "z_in": {
#         0: "2batchsize"
#     },
#     "t_in": {
#         0: "2batchsize",
#     }
# }

# Unpack mha_kvs
k_list = sorted(mha_kvs.keys())
v_list = [mha_kvs[k] for k in k_list]

input_names = ["x", "timestep", "fps", "mha_bias", *k_list]
inputs = (z_in, t_in, fps, mha_bias, *v_list)
output_names = ["output"]

with torch.no_grad():
    torch.onnx.export(
        model,
        inputs,
        args.onnx_path,
        export_params=True,
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        verbose=True,
    )
logger.success("ONNX model saved at {}!".format(args.onnx_path))
