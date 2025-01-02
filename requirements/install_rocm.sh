#!/bin/bash

# 1. Create a conda environment (Python 3.10 is required)
conda create -n opensora python=3.10
conda activate opensora

# 2. Install torch, torchvision, xformers
pip install -r requirements/requirements-rocm62.txt

# 3. Install colossalai (avoid override torch)
pip install colossalai --no-deps
pip install bitsandbytes click contexttimer diffusers einops fabric fastapi galore_torch google ninja numpy packaging peft pre-commit protobuf psutil pydantic ray rich rpyc safetensors sentencepiece tqdm transformers uvicorn

# 4. Install OpenSora
git clone git@github.com:theanh-ktmt/Open-Sora.git
cp -r /home/share-mv/mv-930/opensora/save . # copy weights to opensora working directory
cd Open-Sora
git checkout tensorrt
pip install -v -e .
pip install loguru onnxruntime

# 5. Install Flash Attention
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention/
git checkout c1d146cbd5becd9e33634b1310c2d27a49c7e862
GPU_ARCHS=gfx942 python setup.py install # For MI300 series GPUs
# GPU_ARCHS=gfx90a python setup.py install # For MI250 series GPUs
