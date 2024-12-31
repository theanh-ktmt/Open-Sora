#!/bin/bash

# Create a conda environment (Python 3.10 is required)
conda create -n opensora python=3.10
conda activate opensora

# Install OpenSora
git clone https://github.com/hpcaitech/Open-Sora.git
cd OpenSora
pip install -v -e .

# Install torch-rocm
pip install --force-reinstall torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/rocm6.1

# Install bits and bytes
pip install --force-reinstall --no-deps https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl

# Edit diffusers utils to remove cached_download
# (Example file path: /home/username/miniconda3/envs/opensora/lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py)
#   from huggingface_hub import cached_download, hf_hub_download, model_info
# -> Remove the cached_download import

# Install additional dependencies
pip install loguru onnxruntime av==13.1.0

# Install Flash Attention
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention/
git checkout c1d146cbd5becd9e33634b1310c2d27a49c7e862
GPU_ARCHS=gfx942 python setup.py install # For MI300 series GPUs
# GPU_ARCHS=gfx90a python setup.py install # For MI250 series GPUs

# Install Xformers using pip
pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/rocm6.1

# Alternatively, install Xformers (currently failed)
git clone https://github.com/ROCm/xformers.git
cd xformers/
git submodule update --init --recursive
PYTORCH_ROCM_ARCH=gfx942 python setup.py install # For Instinct MI300 series GPUs
# PYTORCH_ROCM_ARCH=gfx90a python setup.py install # For Instinct MI250 series GPUs
