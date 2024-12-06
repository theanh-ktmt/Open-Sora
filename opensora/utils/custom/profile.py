import os
from pathlib import Path
from typing import Tuple

from loguru import logger

IS_PROFILING: bool = os.environ.get("IS_PROFILING", "0") == "1"
TARGET_SAMPLE: int = int(os.environ.get("TARGET_SAMPLE", "1"))
PROFILE_OUTDIR: Path = Path(os.environ.get("PROFILE_OUTDIR", "save/profile"))
PROFILE_OUTDIR.mkdir(parents=True, exist_ok=True)
logger.info(
    """Profiling status:
- IS_PROFILING: {}
- TARGET_SAMPLE: {}
- PROFILE_OUTDIR: {}""".format(
        IS_PROFILING, TARGET_SAMPLE, PROFILE_OUTDIR
    )
)


def get_profiling_status() -> Tuple[bool, int, Path]:
    """Get information about current profiling status.
    Returns:
        bool: If profiling is turned on or off.
        int: Target sample to be profile.
        Path: Path to output directory.
    """
    return IS_PROFILING, TARGET_SAMPLE, PROFILE_OUTDIR


def is_profiling_sample(sample_index: int) -> bool:
    """Return if current sample is profiling or not."""
    if not IS_PROFILING:
        return False
    else:
        return sample_index == TARGET_SAMPLE


def trace_handler_wrapper(target, row_limit=10000):
    """Function to create a custom trace handler."""
    global PROFILE_OUTDIR

    def trace_handler(prof):
        with open(PROFILE_OUTDIR / f"{target}.profile", "w") as f:
            table = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=row_limit)
            f.write(str(table))
        prof.export_chrome_trace(str(PROFILE_OUTDIR / f"{target}.json"))

    return trace_handler
