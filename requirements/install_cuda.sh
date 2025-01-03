#!/bin/bash

# 1. Create a conda environment (Python 3.10 is required)
conda create -n opensora python=3.10
conda activate opensora

# 2. Install torch, torchvision, xformers (can replace with requirements-cuda121.txt)
pip install -r requirements/requirements-cuda124.txt

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
