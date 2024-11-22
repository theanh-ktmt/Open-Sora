import argparse
import os
import time

import numpy as np
import torch
from loguru import logger

from opensora.models.stdit.stdit3_onnx import STDiT3ONNX
from opensora.utils.custom.common import to_numpy

# Load argument
parser = argparse.ArgumentParser()
parser.add_argument("--onnx-path", type=str, required=True, help="Path to save ONNX file.")
parser.add_argument("--data-dir", type=str, required=True, help="Path to inputs and configs.")
args = parser.parse_args()

# Settings
device = torch.device("cpu")
dtype = torch.float32


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

# TRUE OUTPUT
output_path = os.path.join(args.data_dir, "true_output.pth")
true_output = torch.load(output_path)
true_output = to_numpy(true_output)
logger.info("Loaded true output from {}.".format(output_path))

# ONNX OUTPUT
logger.info("Running ONNX inference...")
start = time.time()

# Prepare for model execution
# NOTE: Running on CPU only
model = STDiT3ONNX(args.onnx_path, device=str(device))

# Prepare the input dictionary
inputs = {
    "x": z_in,
    "timestep": t_in,
    "fps": fps,
    "mha_bias": mha_bias,
    **mha_kvs,
}

# Run inference
onnx_output = model(**inputs)
onnx_output = to_numpy(onnx_output)
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
logger.success("Exported model has been tested with ONNXRuntime, and the result looks good!")
