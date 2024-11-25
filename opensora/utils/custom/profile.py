import os
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

IS_PROFILING: Optional[bool] = None
IGNORE_STEPS: Optional[int] = None
PROFILE_OUTDIR: Optional[str] = None


def get_profiling_status() -> Tuple[bool, int, Path]:
    """Get information about current profiling status.

    Returns:
        bool: If profiling is turned on or off.
        int: Number samples to be ignored when warming up.
        Path: Path to output directory.
    """
    global IS_PROFILING
    global IGNORE_SAMPLES
    global PROFILE_OUTDIR

    if IS_PROFILING is None or IGNORE_SAMPLES is None or PROFILE_OUTDIR is None:
        IS_PROFILING = os.environ.get("IS_PROFILING", "0") == "1"
        IGNORE_SAMPLES = int(os.environ.get("IGNORE_SAMPLES", "0"))
        PROFILE_OUTDIR = Path(os.environ.get("PROFILE_OUTDIR", "save/profile"))
        PROFILE_OUTDIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            """Profiling status:
        - IS_PROFILING: {}
        - IGNORE_SAMPLES: {} samples
        - PROFILE_OUTDIR: {}""".format(
                IS_PROFILING, IGNORE_SAMPLES, PROFILE_OUTDIR
            )
        )

    return IS_PROFILING, IGNORE_SAMPLES, PROFILE_OUTDIR
