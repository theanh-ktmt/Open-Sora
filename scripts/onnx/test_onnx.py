import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
from loguru import logger


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


device = "cuda"

# Load argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx-path", type=str, default="save/onnx/ckpts/144p-2s/stdit3_inferred.onnx", help="Path to save ONNX file."
)
parser.add_argument("--data-dir", type=str, default="save/onnx/data/144p-2s", help="Path to inputs and configs.")
parser.add_argument("--cache-dir", type=str, default="save/onnx/cache/144p-2s", help="Path to cache directory")
args = parser.parse_args()

input_path = os.path.join(args.data_dir, "inputs.pth")
logger.info("Loading inputs from {}...".format(input_path))
inputs = torch.load(input_path, map_location=device)

z_in = to_numpy(inputs["z_in"])
t_in = to_numpy(inputs["t_in"])
y = to_numpy(inputs["y"])
mask = to_numpy(inputs["mask"])
x_mask = to_numpy(inputs["x_mask"])
fps = to_numpy(inputs["fps"])
height = to_numpy(inputs["height"])
width = to_numpy(inputs["width"])

z_in = np.random.randn(2, 4, 15, 18, 32).astype(np.float32)
t_in = np.random.randn(2).astype(np.float32)
y = np.random.randn(2, 1, 300, 4096).astype(np.float32)
# mask = np.random.randint(0, 2, size=(2, 300)).astype(np.int64)
# x_mask = np.random.choice([False, True], size=(2, 15))
fps = np.array([24]).astype(np.float32)
height = np.array([144]).astype(np.float32)
width = np.array([256]).astype(np.float32)

inputs = {
    "z_in": z_in,
    "t_in": t_in,
    "y": y,
    # "mask": mask,
    # "x_mask": x_mask,
    "fps": fps,
    "height": height,
    "width": width,
}

max_workspace_size = 20  # GB
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
    # (
    #     "CUDAExecutionProvider",
    #     {
    #         "arena_extend_strategy": "kNextPowerOfTwo",
    #         "gpu_mem_limit": max_workspace_size * 1024 * 1024 * 1024,
    #         "cudnn_conv_algo_search": "EXHAUSTIVE",
    #         "do_copy_in_default_stream": True,
    #     },
    # ),
]

sess_opt = ort.SessionOptions()
sess_opt.log_severity_level = 1
# sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL # Disable GraphOptimization and leave that for TensorRT

ort_session = ort.InferenceSession(args.onnx_path, sess_options=sess_opt, providers=providers)

# Run inference
ort_outs = ort_session.run(None, inputs)
onnx_output = ort_outs[0]
print("Outputs: {}".format(onnx_output))
print("Shape: {}".format(onnx_output.shape))
