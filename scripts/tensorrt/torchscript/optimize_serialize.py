###
### NOT WORKING
### Dynamo not support conditioning
### has to wrap all condition with torchfunctorch.experimental.control_flow.cond
###

import argparse
import os
import pprint
import time

import torch
import torch_tensorrt
from loguru import logger

from opensora.registry import MODELS, build_module

# Load argument
parser = argparse.ArgumentParser()
parser.add_argument("--trt-path", type=str, default="save/tensorrt/stdit3.onnx", help="Path to save TensorRT file.")
parser.add_argument("--data-dir", type=str, default="save/onnx/data", help="Path to inputs and configs.")
args = parser.parse_args()


# Settings
device = torch.device("cuda")
dtype = torch.float16
os.makedirs(os.path.dirname(args.trt_path), exist_ok=True)


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

z_in = inputs["z_in"].to(dtype)
t_in = inputs["t_in"].to(dtype)
y = inputs["y"].to(dtype)
mask = inputs["mask"].to(dtype)
x_mask = inputs["x_mask"].to(dtype)
fps = inputs["fps"].to(dtype)
height = inputs["height"].to(dtype)
width = inputs["width"].to(dtype)
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

logger.info("Building TensorRT engine...")
start = time.time()
inputs = [z_in, t_in, y, mask, x_mask, fps, height, width]

enabled_precisions = {torch.float, torch.half}  # Run with fp16
trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions)
inputs = inputs
result = trt_ts_module(inputs)
torch.jit.save(trt_ts_module, args.trt_path)
logger.sucess("Saved TensorRT file to {}!".format(args.trt_gm))
logger.info("Done after {:.2f}s!".format(time.time() - start))
