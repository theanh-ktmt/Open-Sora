#!/bin/bash

# 1. Create a conda environment (Python 3.10 is required)
conda create -n opensora python=3.10
conda activate opensora

# 2. Install torch, torchvision, xformers
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 xformers==0.0.29 --index-url https://download.pytorch.org/whl/cu124
# pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 xformers==0.0.29 --index-url https://download.pytorch.org/whl/cu121 # for cuda 12.1

# 3. Install colossalai (avoid override torch)
pip install colossalai --no-deps
pip install bitsandbytes click contexttimer diffusers einops fabric fastapi galore_torch google ninja numpy packaging peft pre-commit protobuf psutil pydantic ray rich rpyc safetensors sentencepiece tqdm transformers uvicorn

# 4. Install OpenSora
git clone git@github.com:theanh-ktmt/Open-Sora.git
cp -r /home/share-mv/mv-930/opensora/save . # copy weights to opensora working directory
cd Open-Sora
git checkout tensorrt
pip install -v -e .

# 5. Install Flash Attention
pip install packaging ninja
pip install flash-attn==2.6.3 --no-build-isolation

# 6. Install Apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# 7. (Optional) Install TensorRT
pip install tensorrt==10.5.0 pycuda==2024.1.2
