import json
import pprint
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    log_model_arch,
    log_model_dtype,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import create_logger, is_distributed, is_main_process, to_torch_dtype

VIDEO_GENERATION_PROMPTS = [
    "A day in the life of a busy city street from dawn to dusk.",
    "A timelapse of a flower blooming in a garden.",
    "An animation of a spaceship traveling through a colorful galaxy.",
    "A scenic drone flyover of a mountain range during sunset.",
    "A futuristic cityscape with flying cars and holographic advertisements.",
    # "A short story about a robot exploring an abandoned warehouse.",
    # "A fantasy world where dragons soar over castles and forests.",
    # "A cooking tutorial showing how to make a delicious dessert step-by-step.",
    # "A virtual tour of a famous historical landmark.",
    # "A wildlife documentary featuring animals in their natural habitats."
]

VIDEO_REFERENCES = ["save/references/sample.jpg"] * len(VIDEO_GENERATION_PROMPTS)

VIDEO_RESOLUTIONS = ["144p", "240p", "360p", "480p", "720p"]
VIDEO_LENGTHS = ["2s", "4s", "8s", "16s"]
ASPECT_RATIO = "9:16"
BATCH_SIZE = 1


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # Save dir
    now = datetime.now()
    save_dir = Path(cfg.save_dir) / str(now.strftime("%Y%m%d.%H%M%S"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building text and image encoder...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    log_model_arch(text_encoder.t5.model, save_dir / "text_encoder" / "arch.log")
    log_model_dtype(text_encoder.t5.model, save_dir / "text_encoder" / "dtype.log")

    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    log_model_arch(vae, save_dir / "vae" / "arch.log")
    log_model_dtype(vae, save_dir / "vae" / "dtype.log")

    logger.info("Start benchmarking...")
    results = {}
    for video_resolution in VIDEO_RESOLUTIONS:
        results[video_resolution] = {}

    for video_resolution, video_length in product(VIDEO_RESOLUTIONS, VIDEO_LENGTHS):
        logger.info("=== Benchmark resolution {} with length {} ===".format(video_resolution, video_length))
        video_dir = save_dir / "videos" / "{}.{}".format(video_resolution, video_length)
        video_dir.mkdir(parents=True, exist_ok=True)

        results[video_resolution][video_length] = {}

        end2end_latencies = []
        text_encoder_latencies = []
        image_encoder_latencies = []
        backbone_latencies = []

        # == prepare video size ==
        resolution = video_resolution
        aspect_ratio = ASPECT_RATIO
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
        num_frames = get_num_frames(video_length)

        # == build diffusion model ==
        input_size = (num_frames, *image_size)
        latent_size = vae.get_latent_size(input_size)
        model = (
            build_module(
                cfg.model,
                MODELS,
                input_size=latent_size,
                in_channels=vae.out_channels,
                caption_channels=text_encoder.output_dim,
                model_max_length=text_encoder.model_max_length,
                enable_sequence_parallelism=enable_sequence_parallelism,
            )
            .to(device, dtype)
            .eval()
        )
        text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
        log_model_arch(model, save_dir / "stdit" / "arch.log")
        log_model_dtype(model, save_dir / "stdit" / "dtype.log")

        # == build scheduler ==
        scheduler = build_module(cfg.scheduler, SCHEDULERS)

        # ======================================================
        # inference
        # ======================================================
        # == load prompts ==
        # prompts = cfg.get("prompt", None)
        prompts = VIDEO_GENERATION_PROMPTS
        start_idx = cfg.get("start_index", 0)

        # == prepare reference ==
        # reference_path = cfg.get("reference_path", [""] * len(prompts))
        reference_path = VIDEO_REFERENCES
        mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
        assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

        # == prepare arguments ==
        fps = cfg.fps
        save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
        multi_resolution = cfg.get("multi_resolution", None)
        batch_size = cfg.get("batch_size", 1)
        num_sample = 1
        loop = 1
        condition_frame_length = cfg.get("condition_frame_length", 5)
        condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
        align = cfg.get("align", None)

        sample_name = cfg.get("sample_name", None)
        prompt_as_path = True

        # == Iter over all samples ==
        for i in progress_wrap(range(0, len(prompts), batch_size)):
            begin = time.time()

            # == prepare batch prompts ==
            batch_prompts = prompts[i : i + batch_size]
            ms = mask_strategy[i : i + batch_size]
            refs = reference_path[i : i + batch_size]

            logger.info(
                "Inferencing prompts: \n{}".format("\n".join(["- {}".format(prompt) for prompt in batch_prompts]))
            )

            # == get json from prompts ==
            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
            original_batch_prompts = batch_prompts

            # == get reference for condition ==
            refs, batched_image_encoder_latencies = collect_references_batch(
                refs, vae, image_size, return_latencies=True
            )
            image_encoder_latencies.append(batched_image_encoder_latencies[0])  # batch 1
            logger.info("Image Encoder Latency: {}s.".format(image_encoder_latencies[-1]))

            # == multi-resolution info ==
            model_args = prepare_multi_resolution_info(
                multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
            )

            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    video_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                )
                for idx in range(len(batch_prompts))
            ]

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 1. refine prompt by openai
            if cfg.get("llm_refine", False):
                # only call openai API when
                # 1. seq parallel is not enabled
                # 2. seq parallel is enabled and the process is rank 0
                if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                # sync the prompt if using seq parallel
                if enable_sequence_parallelism:
                    coordinator.block_all()
                    prompt_segment_length = [
                        len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                    ]

                    # flatten the prompt segment list
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]

                    # create a list of size equal to world size
                    broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)

                    # recover the prompt list
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(
                            all_prompts[segment_start_idx : segment_start_idx + num_segment]
                        )
                        segment_start_idx += num_segment

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                torch.manual_seed(1024)
                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                samples, backbone_latency = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                    return_latencies=True,
                )
                samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                video_clips.append(samples)

            # Done a prompt
            end2end_latencies.append(time.time() - begin)
            backbone_latencies.append(backbone_latency["backbone"])
            text_encoder_latencies.append(backbone_latency["text_encoder"])

            # == save samples ==
            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 2:
                        logger.info("Prompt: %s", batch_prompt)
                    save_path = save_paths[idx]
                    video = [video_clips[i][idx] for i in range(loop)]
                    for i in range(1, loop):
                        video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                    video = torch.cat(video, dim=1)
                    save_path = save_sample(
                        video,
                        fps=save_fps,
                        save_path=save_path,
                        verbose=verbose >= 2,
                    )
                    if save_path.endswith(".mp4") and cfg.get("watermark", False):
                        time.sleep(1)  # prevent loading previous generated video
                        add_watermark(save_path)
            start_idx += len(batch_prompts)

        # Done a combination (Remove first sample for warmup)
        results[video_resolution][video_length]["end2end"] = sum(end2end_latencies[1:]) / len(end2end_latencies[1:])
        results[video_resolution][video_length]["backbone"] = sum(backbone_latencies[1:]) / len(backbone_latencies[1:])
        results[video_resolution][video_length]["text_encoder"] = sum(text_encoder_latencies[1:]) / len(
            text_encoder_latencies[1:]
        )
        results[video_resolution][video_length]["image_encoder"] = sum(image_encoder_latencies[1:]) / len(
            image_encoder_latencies[1:]
        )
        logger.info(
            "Done {} - {}.\n{}\n".format(
                video_resolution, video_length, pprint.pformat(results[video_resolution][video_length])
            )
        )

        # Aggressive save
        with open(save_dir / "latencies.json", "w") as f:
            json.dump(results, f)

    logger.info("Latency information:\n {}".format(pprint.pformat(results)))
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
