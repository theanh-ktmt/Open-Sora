from opensora.utils.custom.config import ConfigurationManager


def is_profiling_sample(sample_index: int) -> bool:
    """Return if current sample is profiling or not."""
    if not ConfigurationManager.get("ENABLE_PROFILER"):
        return False
    else:
        return sample_index == ConfigurationManager.get("TARGET_SAMPLE")


def trace_handler_wrapper(target, row_limit=10000):
    """Function to create a custom trace handler."""

    outdir = ConfigurationManager.get("PROFILE_OUTDIR")

    def trace_handler(prof):
        with open(outdir / f"{target}.profile", "w") as f:
            table = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=row_limit)
            f.write(str(table))
        prof.export_chrome_trace(str(outdir / f"{target}.json"))

    return trace_handler
