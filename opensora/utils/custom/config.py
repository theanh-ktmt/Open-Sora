import os
import pprint
import random
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


class ConfigurationManager:
    CONFIGS = {}
    ENV_VARS = {
        "MIOPEN_DISABLE_CACHE": "1",
        "TORCHINDUCTOR_UNIQUE_KERNEL_NAMES": "1",
    }

    @classmethod
    def initialize(cls) -> None:
        # mlflow: default to "0" and tracking uri is None ("./mlruns")
        cls.CONFIGS["ENABLE_MLFLOW"] = os.environ.get("ENABLE_MLFLOW", "0") == "1"
        cls.CONFIGS["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", None)

        # torch.compile: default to "0"
        cls.CONFIGS["ENABLE_TORCHCOMPILE"] = os.environ.get("ENABLE_TORCHCOMPILE", "0") == "1"
        cls.CONFIGS["CUSTOM_BACKEND"] = os.environ.get("CUSTOM_BACKEND", None)
        cls.CONFIGS["TORCHCOMPILE_CONFIG"] = {
            "fullgraph": True,
            "mode": "default",
            # "mode": "max-autotune", # not working
        }

        # attention impls: default to best
        cls.CONFIGS["ATTN_IMPLS"] = {
            "self_attn.spatial_blocks": "flash_attn",  # only option
            "self_attn.temporal_blocks": "triton_bhsd_attn",  # others: torch_impl_attn, triton_bshd_attn
            "multihead_attn": "padded_xformers_default_attn",  # others: torch_impl_attn, xformers_default_attn, torch_default_attn
        }

        # profile: default to "0" (off), target sample index 2, and output to save/profile
        cls.CONFIGS["ENABLE_PROFILER"] = os.environ.get("ENABLE_PROFILER", "0") == "1"
        cls.CONFIGS["TARGET_SAMPLE"] = int(os.environ.get("TARGET_SAMPLE", "1"))
        cls.CONFIGS["PROFILE_OUTDIR"] = Path(os.environ.get("PROFILE_OUTDIR", "save/profile"))
        cls.CONFIGS["PROFILE_OUTDIR"].mkdir(parents=True, exist_ok=True)

        # tensorrt: default to "0" (off)
        cls.CONFIGS["ENABLE_TENSORRT"] = os.environ.get("ENABLE_TENSORRT", "0") == "1"
        if cls.CONFIGS["ENABLE_TORCHCOMPILE"] and cls.CONFIGS["ENABLE_TENSORRT"]:
            logger.error("TensorRT and torch.compile are not working along! Shutting down.")
            exit(0)

        # benchmark: default to 5 fixed prompts
        num_prompts = int(os.environ.get("NUM_PROMPTS", "5"))
        pick_random = os.environ.get("PICK_RANDOM", "0") == "1"
        prompt_set = get_prompt_set(num_prompts, pick_random)
        cls.CONFIGS["BENCHMARK"] = {
            "prompts": prompt_set,
            "references": ["save/references/sample.jpg"] * num_prompts,
            "num_prompts": num_prompts,
            "pick_random": pick_random,
            "resolutions": [
                # "144p",
                # "240p",
                # "360p",
                # "480p",
                "720p",
            ],
            "lengths": [
                # "2s",
                "4s",
                # "8s",
                # "16s",
            ],
            "aspect_ratio": "9:16",
            "batch_size": 1,
        }
        logger.info("Loaded configurations:\n{}".format(pprint.pformat(cls.CONFIGS)))

        # env: setup env vars
        setup_env_vars(cls.ENV_VARS)
        logger.info("Environment variables is set: {}".format(pprint.pformat(cls.ENV_VARS)))

    @classmethod
    def get(cls, key: str) -> Any:
        if key in cls.CONFIGS:
            return cls.CONFIGS[key]
        else:
            raise NameError("Configuration '{}' not found!".format(key))


def setup_env_vars(env_vars: Dict[str, Any]) -> None:
    """Set values to all environment variables"""
    for key, value in env_vars.items():
        os.environ[key] = value


def get_prompt_set(num_prompts: int = 5, pick_random: bool = False) -> List[str]:
    """A selected list of maximum 20 prompts. Refer to VBench prompt set.

    Origin can be found at 'assets/texts/VBench/all_dimension.txt'."""
    PROMPT_SET = [
        "A day in the life of a busy city street from dawn to dusk.",
        "A timelapse of a flower blooming in a garden.",
        "An animation of a spaceship traveling through a colorful galaxy.",
        "A scenic drone flyover of a mountain range during sunset.",
        "A futuristic cityscape with flying cars and holographic advertisements.",
        "A short story about a robot exploring an abandoned warehouse.",
        "A fantasy world where dragons soar over castles and forests.",
        "A cooking tutorial showing how to make a delicious dessert step-by-step.",
        "A virtual tour of a famous historical landmark.",
        "A wildlife documentary featuring animals in their natural habitats.",
        "A modern art museum, with colorful paintings.",
        "A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo.",
        "A panda drinking coffee in a cafe in Paris.",
        "A corgi's head depicted as an explosion of a nebula.",
        "Campfire at night in a snowy forest with starry sky in the background.",
        "A shark is swimming in the ocean.",
        "Snow rocky mountains peaks canyon. snow blanketed rocky mountains surround and shadow deep canyons. the canyons twist and bend through the high elevated mountain peaks, pan left.",
        "Turtle swimming in ocean.",
        "An astronaut flying in space.",
        "A boat sailing leisurely along the Seine River with the Eiffel Tower in background, pan right.",
    ]
    assert num_prompts <= len(PROMPT_SET), "Number of prompts ({}) exceeded prompt set size ({}).".format(
        num_prompts, len(PROMPT_SET)
    )
    if not pick_random:
        return PROMPT_SET[:num_prompts]
    else:
        return random.sample(PROMPT_SET, k=num_prompts)


ConfigurationManager.initialize()
