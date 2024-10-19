import torch
from loguru import logger
from mmengine.runner import set_random_seed

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, build_module
from opensora.schedulers.rf.rectified_flow import RFlowScheduler, timestep_transform
from opensora.utils.inference_utils import (
    apply_mask_strategy,
    collect_references_batch,
    extract_json_from_prompts,
    prepare_multi_resolution_info,
)
from opensora.utils.misc import to_torch_dtype

# Configs
resolution = "144p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "/home/tran/workspace/Open-Sora/save/inference/test"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "fp16"
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
text_encoder_cfg = dict(type="t5", from_pretrained="DeepFloyd/t5-v1_1-xxl", model_max_length=300, dtype=dtype)

num_sampling_steps = 30
scheduler_cfg = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=num_sampling_steps,
    cfg_scale=7.0,
)

aes = 6.5
flow = None


# Settings
device = torch.device("cuda")
dtype = to_torch_dtype(dtype)

image_size = get_image_size(resolution, aspect_ratio)
num_frames = get_num_frames(num_frames)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

set_random_seed(seed)

# Init text and image encoder
text_encoder = build_module(text_encoder_cfg, MODELS, device=device)
vae = build_module(vae_cfg, MODELS).to(device, dtype).eval()

# Build diffusion model
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


# Text conditioning
prompts = ["A bear climbing a tree"]
model_args = text_encoder.encode(prompts)
model_args["y"].shape, model_args["mask"].shape

# Classifier-free guidance
n = len(prompts)
y_null = text_encoder.null(n)
model_args["y"] = torch.cat([model_args["y"], y_null], 0)

# Prepare additional arguments
additional_args = prepare_multi_resolution_info(multi_resolution, n, image_size, num_frames, fps, device, dtype)
model_args.update(additional_args)

# Timesteps
num_timesteps = 1000

timesteps = [(1.0 - i / num_sampling_steps) * num_timesteps for i in range(num_sampling_steps)]
timesteps = [timestep_transform(t, additional_args, num_timesteps=num_timesteps) for t in timesteps]

# Noise input
z = torch.randn(n, vae.out_channels, *latent_size, device=device, dtype=dtype)

# Image conditioning
refs = ["/remote/vast0/tran/workspace/Open-Sora/save/references/sample.jpg"]
mask_strategy = [""]

prompts, refs, ms = extract_json_from_prompts(prompts, refs, mask_strategy)
refs = collect_references_batch(refs, vae, image_size)
mask = apply_mask_strategy(z, refs, ms, 0, align=align)

# Init noise added
noise_added = torch.zeros_like(mask, dtype=torch.bool)
noise_added = noise_added | (mask == 1)

# Init scheduler
scheduler = RFlowScheduler(
    num_timesteps=num_timesteps,
    num_sampling_steps=num_sampling_steps,
    use_discrete_timesteps=False,
    use_timestep_transform=True,
)


# Prepare mask
t = timesteps[0]

mask_t = mask * num_timesteps
x0 = z.clone()
x_noise = scheduler.add_noise(x0, torch.randn_like(x0), t)

mask_t_upper = mask_t >= t.unsqueeze(1)
model_args["x_mask"] = mask_t_upper.repeat(2, 1)
mask_add_noise = mask_t_upper & ~noise_added

z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
noise_added = mask_t_upper


# Prepare data
z_in = torch.cat([z, z], 0)
t_in = torch.cat([t, t], 0)


# Unpack model args
y = model_args["y"]
mask = model_args["mask"]
x_mask = model_args["x_mask"]
fps = model_args["fps"]
height = model_args["height"]
width = model_args["width"]


# First real iteration
true_pred = model(z_in, t_in, y, mask=mask, x_mask=x_mask, fps=fps, height=height, width=width)


# Convert to ONNX
ONNX_FILEPATH = "save/onnx/stdit3.onnx"

dynamic_axes = {
    "z_in": {
        0: "2batchsize",
        2: "frames",
        3: "height",
        4: "width",
    },
    "t_in": {
        0: "2batchsize",
    },
    "y": {
        0: "2batchsize",
    },
    "mask": {
        0: "batchsize",
    },
    "x_mask": {
        0: "2batchsize",
        1: "frames",
    },
}

input_names = ["z_in", "t_in", "y", "mask", "x_mask", "fps", "height", "width"]

inputs = (z_in, t_in, y, mask, x_mask, fps, height, width)

output_names = ["output"]

torch.onnx.export(
    model,
    inputs,
    ONNX_FILEPATH,
    export_params=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
)
