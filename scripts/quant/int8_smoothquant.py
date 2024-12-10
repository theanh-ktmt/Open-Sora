import pprint

import torch
from datasets import load_from_disk
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
from loguru import logger

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, build_module

NUM_CALIBRATION_SAMPLES = 128
SMOOTHQUANT_ALPHA = 0.5

# Config
resolution = "720p"
aspect_ratio = "9:16"
num_frames = "4s"
seed = 42

model_cfg = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae_cfg = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder_cfg = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
logger.info("Model Configs: \n{}".format(pprint.pformat(model_cfg)))
logger.info("VAE Configs: \n{}".format(pprint.pformat(vae_cfg)))
logger.info("T5 Configs: \n{}".format(pprint.pformat(text_encoder_cfg)))


# Settings
torch.manual_seed(seed)
logger.info("Setting background...")
dtype = torch.float16
device = torch.device("cuda")
image_size = get_image_size(resolution, aspect_ratio)
num_frames = get_num_frames(num_frames)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Build text and image encoder
logger.info("Building text and image encoder...")
text_encoder = build_module(text_encoder_cfg, MODELS, device=device)
vae = build_module(vae_cfg, MODELS).to(device, dtype).eval()

# Build diffusion model
logger.info("Building diffusion model...")
input_size = (num_frames, *image_size)
latent_size = vae.get_latent_size(input_size)

model = (
    build_module(
        model_cfg,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=False,
    )
    .to(device, dtype)
    .eval()
)

# Load calib data
logger.info("Load calibration data...")
data_path = "save/checkpoints/OpenSora-STDiT-v3-INT8-SmoothQuant/calib_data"
dataset = load_from_disk(data_path)

# Quantize model
logger.info("Quantizing model...")
recipe = [
    SmoothQuantModifier(smoothing_strength=SMOOTHQUANT_ALPHA),
    QuantizationModifier(targets="Linear", scheme="W8A16"),
]
oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save model
SAVE_DIR = "save/checkpoints/OpenSora-STDiT-v3-INT8-SmoothQuant"
model.save_pretrained(SAVE_DIR)
logger.success("Saved quantized model to {}.".format(SAVE_DIR))
