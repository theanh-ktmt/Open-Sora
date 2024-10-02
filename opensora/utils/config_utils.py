import argparse
import json
import os
from glob import glob

from mmengine.config import Config


def parse_args(training=False, mode=None):
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("config", help="model config file path")

    # ======================================================
    # General
    # ======================================================
    parser.add_argument("--seed", default=None, type=int, help="seed for reproducibility")
    parser.add_argument(
        "--ckpt-path",
        default=None,
        type=str,
        help="path to model ckpt; will overwrite cfg.model.from_pretrained if specified",
    )
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")
    parser.add_argument("--outputs", default=None, type=str, help="the dir to save model weights")
    parser.add_argument("--flash-attn", default=None, type=str2bool, help="enable flash attention")
    parser.add_argument("--layernorm-kernel", default=None, type=str2bool, help="enable layernorm kernel")
    parser.add_argument("--resolution", default=None, type=str, help="multi resolution")
    parser.add_argument("--data-path", default=None, type=str, help="path to data csv")
    parser.add_argument("--dtype", default=None, type=str, help="data type")

    # ======================================================
    # Inference
    # ======================================================
    if not training:
        # output
        parser.add_argument("--save-dir", default=None, type=str, help="path to save generated samples")
        parser.add_argument("--sample-name", default=None, type=str, help="sample name, default is sample_idx")
        parser.add_argument("--start-index", default=None, type=int, help="start index for sample name")
        parser.add_argument("--end-index", default=None, type=int, help="end index for sample name")
        parser.add_argument("--num-sample", default=None, type=int, help="number of samples to generate for one prompt")
        parser.add_argument("--prompt-as-path", action="store_true", help="use prompt as path to save samples")
        parser.add_argument("--verbose", default=None, type=int, help="verbose level")

        # prompt
        parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
        parser.add_argument("--prompt", default=None, type=str, nargs="+", help="prompt list")
        parser.add_argument("--llm-refine", default=None, type=str2bool, help="enable LLM refine")
        parser.add_argument("--prompt-generator", default=None, type=str, help="prompt generator")

        # image/video
        parser.add_argument("--num-frames", default=None, type=str, help="number of frames")
        parser.add_argument("--fps", default=None, type=int, help="fps")
        parser.add_argument("--save-fps", default=None, type=int, help="save fps")
        parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")
        parser.add_argument("--frame-interval", default=None, type=int, help="frame interval")
        parser.add_argument("--aspect-ratio", default=None, type=str, help="aspect ratio (h:w)")
        parser.add_argument("--watermark", default=None, type=str2bool, help="watermark video")

        # hyperparameters
        parser.add_argument("--num-sampling-steps", default=None, type=int, help="sampling steps")
        parser.add_argument("--cfg-scale", default=None, type=float, help="balance between cond & uncond")

        # reference
        parser.add_argument("--loop", default=None, type=int, help="loop")
        parser.add_argument("--condition-frame-length", default=None, type=int, help="condition frame length")
        parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
        parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")
        parser.add_argument("--aes", default=None, type=float, help="aesthetic score")
        parser.add_argument("--flow", default=None, type=float, help="flow score")
        parser.add_argument("--camera-motion", default=None, type=str, help="camera motion")
    # ======================================================
    # Training
    # ======================================================
    else:
        parser.add_argument("--lr", default=None, type=float, help="learning rate")
        parser.add_argument("--wandb", default=None, type=bool, help="enable wandb")
        parser.add_argument("--load", default=None, type=str, help="path to continue training")
        parser.add_argument("--start-from-scratch", action="store_true", help="start training from scratch")
        parser.add_argument("--warmup-steps", default=None, type=int, help="warmup steps")
        parser.add_argument("--record-time", default=False, action="store_true", help="record time of each part")

    # ======================================================
    # Additional Arguments
    # ======================================================
    if mode == "get_calib":
        parser.add_argument("--data_num", default=100, type=int)
        parser.add_argument("--save_inp_oup", action="store_true")
    elif mode == "ptq":
        parser.add_argument("--calib_data", default=None, type=str, help="path to quantization calib data")
    elif mode == "quant_inference":
        parser.add_argument("--dataset_type", default="opensora", type=str)
        parser.add_argument(
            "--quant_ckpt",
            type=str,
            default=None,
            help="path to config which constructs model",
        )
    elif mode == "qat":
        parser.add_argument("--quant_dir", default=None, type=str, help="the dir with quant ckpt and quant config")
    elif mode is None:
        pass
    else:
        raise NotImplementedError

    if mode == "ptq" or mode == "quant_inference":
        parser.add_argument(
            "--ptq_config",
            type=str,
            default=None,
            help="path to config which constructs model",
        )
        parser.add_argument(
            "--part_quant",
            action="store_true",
            help="whether only quant a part",
        )
        parser.add_argument(
            "--skip_quant_weight",
            action="store_true",
            help="whether to skip weight quantization",
        )
        parser.add_argument(
            "--skip_quant_act",
            action="store_true",
            help="whether to skip activation quantization",
        )
        parser.add_argument(
            "--num_videos",
            type=int,
            default=1000,
            help="number of generated videos",
        )
        parser.add_argument(
            "--layer_wise_quant",
            action="store_true",
            help="whether only quant a part of layers",
        )
        parser.add_argument(
            "--group_wise_quant",
            action="store_true",
            help="whether only quant a group",
        )
        parser.add_argument(
            "--timestep_wise_quant",
            action="store_true",
            help="whether only quant a part of timesteps",
        )
        parser.add_argument(
            "--block_group_wise_quant",
            action="store_true",
            help="whether only quant a part of timesteps",
        )
        parser.add_argument(
            "--quant_ratio",
            type=float,
            default=1.0,
            help="the ratio of quant layer",
        )
        parser.add_argument(
            "--part_fp",
            action="store_true",
            help="whether only fp a part of layer",
        )
        parser.add_argument(
            "--fp_ratio",
            type=float,
            default=1.0,
            help="the ratio of fp layer",
        )
        parser.add_argument(
            "--timestep_wise_mp",
            action="store_true",
            help="timestep wise mixed precision",
        )
        parser.add_argument(
            "--weight_mp",
            action="store_true",
            help="mixed precision for weight",
        )
        parser.add_argument(
            "--act_mp",
            action="store_true",
            help="mixed precision for act",
        )
        parser.add_argument(
            "--time_mp_config_weight",
            type=str,
            default=None,
            help="path to config of mixed precision for weight",
        )
        parser.add_argument(
            "--time_mp_config_act",
            type=str,
            default=None,
            help="path to config of mixed precision for act",
        )
        parser.add_argument(
            "--group_quant",
            type=str,
            default=None,
            help="path to config of mixed precision for act",
        )
        parser.add_argument(
            "--smooth_quant_alpha",
            nargs="+",
            type=float,
            default=None,
            help="path to config of mixed precision for act",
        )
        parser.add_argument(
            "--block_wise_quant_progressively",
            action="store_true",
            help="mixed precision for act",
        )
        parser.add_argument(
            "--block_wise_quant",
            action="store_true",
            help="mixed precision for act",
        )

    return parser.parse_args()


def merge_args(cfg, args, training=False):
    if args.ckpt_path is not None:
        cfg.model["from_pretrained"] = args.ckpt_path
        if cfg.get("discriminator") is not None:
            cfg.discriminator["from_pretrained"] = args.ckpt_path
        args.ckpt_path = None
    if args.flash_attn is not None:
        cfg.model["enable_flash_attn"] = args.flash_attn
        args.enable_flash_attn = None
    if args.layernorm_kernel is not None:
        cfg.model["enable_layernorm_kernel"] = args.layernorm_kernel
        args.enable_layernorm_kernel = None
    if args.data_path is not None:
        cfg.dataset["data_path"] = args.data_path
        args.data_path = None
    # NOTE: for vae inference (reconstruction)
    if not training and "dataset" in cfg:
        if args.image_size is not None:
            cfg.dataset["image_size"] = args.image_size
        if args.num_frames is not None:
            cfg.dataset["num_frames"] = args.num_frames
    if not training:
        if args.cfg_scale is not None:
            cfg.scheduler["cfg_scale"] = args.cfg_scale
            args.cfg_scale = None
        if args.num_sampling_steps is not None:
            cfg.scheduler["num_sampling_steps"] = args.num_sampling_steps
            args.num_sampling_steps = None

    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    return cfg


def read_config(config_path):
    cfg = Config.fromfile(config_path)
    return cfg


def parse_configs(training=False, mode=None):
    args = parse_args(training, mode)
    cfg = read_config(args.config)
    cfg = merge_args(cfg, args, training)
    return cfg


def define_experiment_workspace(cfg, get_last_workspace=False):
    """
    This function creates a folder for experiment tracking.

    Args:
        args: The parsed arguments.

    Returns:
        exp_dir: The path to the experiment folder.
    """
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(cfg.outputs, exist_ok=True)
    experiment_index = len(glob(f"{cfg.outputs}/*"))
    if get_last_workspace:
        experiment_index -= 1

    # Create an experiment folder
    model_name = cfg.model["type"].replace("/", "-")
    exp_name = f"{experiment_index:03d}-{model_name}"
    exp_dir = f"{cfg.outputs}/{exp_name}"
    return exp_name, exp_dir


def save_training_config(cfg, experiment_dir):
    with open(f"{experiment_dir}/config.txt", "w") as f:
        json.dump(cfg, f, indent=4)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
