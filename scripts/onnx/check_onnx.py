import argparse
import os
import pprint
import time

import numpy as np
import onnxruntime as ort
import torch
from loguru import logger

from opensora.registry import MODELS, build_module


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# Load argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx-path", type=str, default="save/onnx/ckpts/144p-2s/stdit3.onnx", help="Path to save ONNX file."
)
parser.add_argument("--data-dir", type=str, default="save/onnx/data/144p-2s", help="Path to inputs and configs.")
parser.add_argument("--cache-dir", type=str, default="save/onnx/cache/144p-2s", help="Path to cache directory")
args = parser.parse_args()

# Create save dir
os.makedirs(os.path.dirname(args.cache_dir), exist_ok=True)

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
# for name, module in model.named_modules():
#     logger.info(f"Layer {name}: {'Training' if module.training else 'Evaluation'}")

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

# ignore mask and x_mask
mask = None
x_mask = None

# TRUE OUTPUT
logger.info("Running true inference...")
start = time.time()
true_output = model(z_in, t_in, y, mask=mask, x_mask=x_mask, fps=fps, height=height, width=width)
true_output = to_numpy(true_output)
logger.success("Done after {:.2f}s!".format(time.time() - start))


# ONNX OUTPUT
logger.info("Running ONNX inference...")
start = time.time()

# Prepare for model execution
max_workspace_size = 10  # GB
providers = [
    (
        "TensorrtExecutionProvider",
        {  # Select GPU to execute
            "trt_max_workspace_size": max_workspace_size * 1024 * 1024 * 1024,  # Set GPU memory usage limit
            # "trt_fp16_enable": True,  # Enable FP16 precision for faster inference
            # Engine cache
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./",
            # For embedding context
            "trt_dump_ep_context_model": True,
            "trt_ep_context_file_path": args.cache_dir,
            # Timing cache
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": args.cache_dir,
            "trt_force_timing_cache": True,
        },
    ),
    (
        "CUDAExecutionProvider",
        {
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": max_workspace_size * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        },
    ),
]

# Load the ONNX model
sess_opt = ort.SessionOptions()
sess_opt.log_severity_level = 1

ort_session = ort.InferenceSession(args.onnx_path, sess_options=sess_opt, providers=providers)

# Prepare the input dictionary
inputs = {
    "z_in": to_numpy(z_in),
    "t_in": to_numpy(t_in),
    "y": to_numpy(y),
    # "mask": to_numpy(mask),
    # "x_mask": to_numpy(x_mask),
    "fps": to_numpy(fps),
    "height": to_numpy(height),
    "width": to_numpy(width),
}

# Run inference
ort_outs = ort_session.run(None, inputs)

# Process the results
onnx_output = ort_outs[0]
logger.success("Done after {:.2f}s!".format(time.time() - start))

# VERIFY RESULT RELIABILITY
logger.info("Comparing results...")
# Check shape
if onnx_output.shape == true_output.shape:
    logger.success("Shape matched! {}".format(onnx_output.shape))
else:
    logger.error("Shape not matched! {} != {}".format(onnx_output.shape, true_output.shape))

# Check correction
np.testing.assert_allclose(onnx_output, true_output, rtol=1e-03, atol=1e-03, verbose=True)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
