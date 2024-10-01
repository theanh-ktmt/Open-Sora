import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from opensora.registry import SCHEDULERS
from opensora.utils.misc import create_logger

from .rectified_flow import RFlowScheduler, timestep_transform

logger = create_logger()


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
        return_latencies=False,
        do_profiling=False,
    ):
        latencies = {}

        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        start = time.time()
        if do_profiling:
            with profile(
                activities=[ProfilerActivity.CUDA],
                # profile_memory=True,
                record_shapes=True,
            ) as prof:
                with record_function("text_embedding"):
                    model_args = text_encoder.encode(prompts)

            with open("save/profile/text_embedding.profile", "w") as f:
                table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=100)
                f.write(str(table))
            prof.export_chrome_trace("save/profile/text_embedding.json")
        else:
            model_args = text_encoder.encode(prompts)
        latencies["text_encoder"] = time.time() - start

        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)

        latencies["backbone"] = 0
        if do_profiling:
            with profile(
                activities=[ProfilerActivity.CUDA],
                # profile_memory=True,
                record_shapes=True,
            ) as prof:
                for i, t in progress_wrap(enumerate(timesteps)):
                    # mask for adding noise
                    if mask is not None:
                        mask_t = mask * self.num_timesteps
                        x0 = z.clone()
                        x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                        mask_t_upper = mask_t >= t.unsqueeze(1)
                        model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                        mask_add_noise = mask_t_upper & ~noise_added

                        z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                        noise_added = mask_t_upper

                    # classifier-free guidance
                    z_in = torch.cat([z, z], 0).to(text_encoder.dtype)
                    t = torch.cat([t, t], 0).to(text_encoder.dtype)

                    start = time.time()

                    with record_function("video_generation"):
                        pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
                    latencies["backbone"] += time.time() - start

                    pred_cond, pred_uncond = pred.chunk(2, dim=0)
                    v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                    # update z
                    dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                    dt = dt / self.num_timesteps
                    z = z + v_pred * dt[:, None, None, None, None]

                    if mask is not None:
                        z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

            with open("save/profile/backbone.profile", "w") as f:
                table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=100)
                f.write(str(table))
            prof.export_chrome_trace("save/profile/backbone.json")

        else:
            for i, t in progress_wrap(enumerate(timesteps)):
                # mask for adding noise
                if mask is not None:
                    mask_t = mask * self.num_timesteps
                    x0 = z.clone()
                    x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                    mask_t_upper = mask_t >= t.unsqueeze(1)
                    model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                    mask_add_noise = mask_t_upper & ~noise_added

                    z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                    noise_added = mask_t_upper

                # classifier-free guidance
                z_in = torch.cat([z, z], 0).to(text_encoder.dtype)
                t = torch.cat([t, t], 0).to(text_encoder.dtype)

                start = time.time()
                pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
                latencies["backbone"] += time.time() - start

                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # update z
                dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                dt = dt / self.num_timesteps
                z = z + v_pred * dt[:, None, None, None, None]

                if mask is not None:
                    z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        if return_latencies:
            return z, latencies
        return z

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)
