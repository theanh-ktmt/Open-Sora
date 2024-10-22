import argparse
import os
import pprint

import torch
from loguru import logger

from opensora.registry import MODELS, build_module

# Load argument
parser = argparse.ArgumentParser()
parser.add_argument("--onnx-path", type=str, default="save/onnx/ckpt/stdit3.onnx", help="Path to save ONNX file.")
parser.add_argument("--data-dir", type=str, default="save/onnx/data", help="Path to inputs and configs.")
args = parser.parse_args()

# Settings
device = torch.device("cuda")
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
y = inputs["y"]
mask = inputs["mask"]
x_mask = inputs["x_mask"]
fps = inputs["fps"]
height = inputs["height"]
width = inputs["width"]
logger.info(
    f"""Inputs:
z_in: {z_in.shape} {z_in.dtype}
t_in: {t_in.shape} {t_in.dtype}
y: {y.shape} {y.dtype}
mask: {mask.shape} {mask.dtype}
x_mask: {x_mask.shape} {x_mask.dtype}
fps: {fps.shape} {fps.dtype}
height: {height.shape} {height.dtype}
width: {width.shape} {width.dtype}"""
)

# Convert to ONNX
logger.info("Exporting ONNX models...")
dynamic_axes = {
    "z_in": {
        0: "2batchsize",
        # 2: "frames",
        # 3: "height",
        # 4: "width",
    },
    "t_in": {
        0: "2batchsize",
    },
    "y": {
        0: "2batchsize",
    },
    "mask": {
        0: "2batchsize",
    },
    # "x_mask": {
    #     0: "2batchsize",
    #     # 1: "frames",
    # },
    "output": {
        0: "2batchsize",
    },
}

# ignore x_mask
x_mask = None

# input_names = ["z_in", "t_in", "y", "mask", "x_mask", "fps", "height", "width"]
input_names = ["z_in", "t_in", "y", "mask", "fps", "height", "width"]
inputs = (z_in, t_in, y, mask, x_mask, fps, height, width)
output_names = ["output"]

torch.onnx.export(
    model,
    inputs,
    args.onnx_path,
    export_params=True,
    opset_version=13,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
)
logger.success("ONNX model saved at {}!".format(args.onnx_path))
