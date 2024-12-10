import os
import pprint
import random

import torch
from datasets import Dataset
from loguru import logger

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.schedulers.rf.rectified_flow import timestep_transform
from opensora.utils.inference_utils import (
    apply_mask_strategy,
    collect_references_batch,
    extract_json_from_prompts,
    prepare_multi_resolution_info,
)

NUM_CALIBRATION_SAMPLES = 128

# Configs
resolution = "720p"
aspect_ratio = "9:16"
num_frames = "4s"
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "/home/tran/workspace/Open-Sora/save/inference/test"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
condition_frame_length = 5
align = 5

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
text_encoder_cfg = dict(type="t5", from_pretrained="DeepFloyd/t5-v1_1-xxl", model_max_length=300)

num_sampling_steps = 30
scheduler_cfg = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=num_sampling_steps,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
logger.info("Used configuration: {} {} {}".format(resolution, aspect_ratio, num_frames))


# Settings
torch.manual_seed(seed)
logger.info("Setting background...")
device = torch.device("cuda")
dtype = torch.float16
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
text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
logger.info("Model:\n{}".format(model))

# Prepare prompt list
prompts_path = "assets/texts/VBench/all_dimension.txt"
with open(prompts_path, "r") as f:
    all_prompts = [line.strip() for line in f.readlines()]
    all_prompts = random.sample(all_prompts, k=NUM_CALIBRATION_SAMPLES)
logger.info("Prompts: {}".format(pprint.pformat(all_prompts)))

# Prepare calibration data
data = []
for i, prompt in enumerate(all_prompts):
    logger.info(f"[{i+1:03}/{NUM_CALIBRATION_SAMPLES:03}] Processing prompt '{prompt}'...")
    prompts = [prompt]
    n = len(prompts)
    num_timesteps = 1000

    # Build scheduler
    scheduler = build_module(scheduler_cfg, SCHEDULERS)

    # Prepare additional arguments
    additional_args = prepare_multi_resolution_info(multi_resolution, n, image_size, num_frames, fps, device, dtype)

    # Timesteps
    logger.info("Preparing noise and timestep input...")
    timesteps = [(1.0 - i / num_sampling_steps) * num_timesteps for i in range(num_sampling_steps)]
    timesteps = [timestep_transform(t, additional_args, num_timesteps=num_timesteps) for t in timesteps]
    z = torch.randn(n, vae.out_channels, *latent_size, device=device, dtype=dtype)

    # Text conditioning
    logger.info("Embedding texts...")
    model_args = text_encoder.encode(prompts)

    # Classifier-free guidance
    y_null = text_encoder.null(n)
    model_args["y"] = torch.cat([model_args["y"], y_null], 0).to(dtype)
    model_args["mask"] = model_args["mask"].repeat(2, 1)
    model_args.update(additional_args)

    # Image conditioning
    logger.info("Embedding images...")
    refs = ["save/references/sample.jpg"]
    mask_strategy = [""]

    prompts, refs, ms = extract_json_from_prompts(prompts, refs, mask_strategy)
    refs = collect_references_batch(refs, vae, image_size)
    mask = apply_mask_strategy(z, refs, ms, 0, align=align)

    # Init noise added
    noise_added = torch.zeros_like(mask, dtype=torch.bool)
    noise_added = noise_added | (mask == 1)

    # Init scheduler
    t = timesteps[0]

    # Prepare mask
    logger.info("Convert image embedding to mask...")
    mask_t = mask * num_timesteps
    x0 = z.clone()
    x_noise = scheduler.scheduler.add_noise(x0, torch.randn_like(x0), t)

    mask_t_upper = mask_t >= t.unsqueeze(1)
    model_args["x_mask"] = mask_t_upper.repeat(2, 1)
    mask_add_noise = mask_t_upper & ~noise_added

    z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
    noise_added = mask_t_upper

    # Prepare input data
    z_in = torch.cat([z, z], 0).to(dtype)
    t_in = torch.cat([t, t], 0).to(dtype)

    # KV and bias for MHA
    hidden_size = 1152
    num_heads = 16
    input_sq_size = 512

    input = {
        "x": z_in.tolist(),
        "timestep": t_in.tolist(),
        "y": model_args["y"].tolist(),
        "mask": model_args["mask"].tolist(),
        "x_mask": model_args["x_mask"].tolist(),
        "fps": model_args["fps"].tolist(),
        "height": model_args["height"].tolist(),
        "width": model_args["width"].tolist(),
    }
    data.append(input)

# Building dataset
dataset = Dataset.from_list(data)

# Save dataset
path = "save/checkpoints/OpenSora-STDiT-v3-INT8-SmoothQuant/calib_data"
os.makedirs(path, exist_ok=True)
dataset.save_to_disk(path)
logger.success("Saved calibration data to {}.".format(path))
