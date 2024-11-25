import os
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

IS_PROFILING: Optional[bool] = None
PROFILE_OUTDIR: Optional[str] = None


def get_profiling_status() -> Tuple[bool, Path]:
    """Get information about current profiling status.

    Returns:
        bool: If profiling is turned on or off.
        int: Number samples to be ignored when warming up.
        Path: Path to output directory.
    """
    global IS_PROFILING
    global PROFILE_OUTDIR

    if IS_PROFILING is None or PROFILE_OUTDIR is None:
        IS_PROFILING = os.environ.get("IS_PROFILING", "0") == "1"
        PROFILE_OUTDIR = Path(os.environ.get("PROFILE_OUTDIR", "save/profile"))
        PROFILE_OUTDIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            """Profiling status:
        - IS_PROFILING: {}
        - PROFILE_OUTDIR: {}""".format(
                IS_PROFILING, PROFILE_OUTDIR
            )
        )

    return IS_PROFILING, PROFILE_OUTDIR


def trace_handler_wrapper(target, row_limit=10000):
    """Function to create a custom trace handler."""
    global PROFILE_OUTDIR

    def trace_handler(prof):
        with open(PROFILE_OUTDIR / f"{target}.profile", "w") as f:
            table = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=row_limit)
            f.write(str(table))
        prof.export_chrome_trace(str(PROFILE_OUTDIR / f"{target}.json"))

    return trace_handler
