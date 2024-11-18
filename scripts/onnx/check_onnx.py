import argparse
import os
import pprint
import time

import numpy as np
import onnxruntime as ort
import torch
from loguru import logger

from opensora.registry import MODELS, build_module
from opensora.utils.custom.pos_emb import prepare_pos_emb


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# Load argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx-path", type=str, default="save/onnx/ckpts/144p-2s/stdit3.onnx", help="Path to save ONNX file."
)
parser.add_argument("--data-dir", type=str, default="save/onnx/data/144p-2s", help="Path to inputs and configs.")
parser.add_argument("--cache-dir", type=str, default="save/onnx/cache/144p-2s", help="Path to cache directory")
parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
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
fps = inputs["fps"]
mha_kvs = inputs["mha_kvs"]
mha_bias = inputs["mha_bias"]
height = inputs["height"]
width = inputs["width"]
logger.info(
    f"""Inputs:
    z_in: {z_in.shape} {z_in.dtype}
    t_in: {t_in.shape} {t_in.dtype}
    attn_kv: {mha_kvs["mha_s00_k"].shape} {mha_kvs["mha_s00_k"].dtype}
    attn_bias: {mha_bias.shape} {mha_bias.dtype}
    fps: {fps.shape} {fps.dtype}
    height: {height.shape} {height.dtype}
    width: {width.shape} {width.dtype}"""
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

# TRUE OUTPUT
logger.info("Running true inference...")
start = time.time()
true_output = model(z_in, t_in, fps, mha_bias, **mha_kvs)
true_output = to_numpy(true_output)
logger.success("Done after {:.2f}s!".format(time.time() - start))

del model
torch.cuda.empty_cache()

# ONNX OUTPUT
logger.info("Running ONNX inference...")
start = time.time()

# Prepare for model execution
max_workspace_size = 40  # GB
providers = [
    (
        "TensorrtExecutionProvider",
        {
            "trt_detailed_build_log": True,
            "trt_max_workspace_size": max_workspace_size * 1024 * 1024 * 1024,
            "trt_layer_norm_fp32_fallback": True,  # force Pow + Reduce ops in layer norm to FP32
            "trt_fp16_enable": args.fp16,
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
    "fps": to_numpy(fps),
    "mha_bias": to_numpy(mha_bias),
}

for k, v in mha_kvs.items():
    inputs[k] = to_numpy(v)

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
