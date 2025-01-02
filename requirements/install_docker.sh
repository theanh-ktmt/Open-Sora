# 1. Clone modiffusion
git clone git@github.com:loctxmoreh/modiffusion.git

# 2. Clone Open-Sora
git clone git@github.com:theanh-ktmt/Open-Sora.git
cd Open-Sora
git checkout tensorrt
cp -r /home/share-mv/mv-930/opensora/save . # copy weights to opensora working directory

# 3. Start Docker container (contains torch, torchvision)
bash scripts/custom/run_docker.sh

# 4. (Inside Docker) Install modiffusion
cd ../modiffusion
pip install --no-build-isolation -e .[dev]

# 5. (Inside Docker) Install colossalai (avoid override torch)
pip install colossalai --no-deps
pip install bitsandbytes click contexttimer diffusers einops fabric fastapi galore_torch google ninja numpy packaging peft pre-commit protobuf psutil pydantic ray rich rpyc safetensors sentencepiece tqdm transformers uvicorn

# 6. (Inside Docker) Install Open-Sora
cd ../Open-Sora
pip install -v -e .
pip install loguru onnxruntime

# 6. (Inside Docker) Install flash-attn
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention/
git checkout c1d146cbd5becd9e33634b1310c2d27a49c7e862
GPU_ARCHS=gfx942 python setup.py install # For MI300 series GPUs
# GPU_ARCHS=gfx90a python setup.py install # For MI250 series GPUs

# 7. (Inside Docker) Install xformers (only rocm 6.1 is available and build from source not runnable for rocm 6.2)
pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/rocm6.1
