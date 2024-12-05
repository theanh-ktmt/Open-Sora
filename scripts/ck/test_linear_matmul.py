import time

import torch
import torch.nn as nn
from loguru import logger
from torch.profiler import ProfilerActivity, profile, schedule
from tqdm import tqdm

from opensora.utils.custom.layers.linear import CustomedCKLinear
from opensora.utils.custom.profile import get_profiling_status, trace_handler_wrapper

torch.manual_seed(2024)
_, _ = get_profiling_status()

n_runs = 10000
warm_ups = 1000
all_durs = {}

for in_channels, out_channels in [
    (1152, 4608),
    (4608, 1152),
    (1152, 1152),
    (1152, 3456),
]:
    logger.info("In: {} - Out: {}".format(in_channels, out_channels))
    linear = nn.Linear(in_channels, out_channels, dtype=torch.float16, device="cuda")
    ck_linear = CustomedCKLinear(linear)

    # Correctness
    input = torch.rand(2, 108000, in_channels, dtype=torch.float16, device="cuda")
    out1 = linear(input)
    out2 = ck_linear(input)
    logger.info("Max absolute difference: {}".format(torch.max(torch.abs(out1 - out2)).item()))
    logger.info("All close: {}".format(torch.allclose(out1, out2, rtol=1e-3, atol=1e-3)))

    # Speed
    logger.info("Benchmarking original linear...")
    dur1 = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=warm_ups,  # Number of steps to skip
            warmup=0,  # Number of steps to include in the warm-up phase
            active=5,  # Number of steps to include in the active phase (profiling)
            repeat=1,  # Number of times to repeat the above schedule
        ),
        record_shapes=True,
        with_stack=True,
        on_trace_ready=trace_handler_wrapper(f"in{in_channels}.out{out_channels}.pt-rocm"),
    ) as prof:
        for i in tqdm(range(n_runs)):
            start = time.perf_counter()
            out1 = linear(input)
            dur1.append(time.perf_counter() - start)
            prof.step()

    logger.info("Benchmarking CK linear...")
    dur2 = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=warm_ups,  # Number of steps to skip
            warmup=0,  # Number of steps to include in the warm-up phase
            active=5,  # Number of steps to include in the active phase (profiling)
            repeat=1,  # Number of times to repeat the above schedule
        ),
        record_shapes=True,
        with_stack=True,
        on_trace_ready=trace_handler_wrapper(f"in{in_channels}.out{out_channels}.tunned-ck"),
    ) as prof:
        for i in tqdm(range(n_runs)):
            start = time.perf_counter()
            out2 = ck_linear(input)
            dur2.append(time.perf_counter() - start)
            prof.step()

    all_durs[f"in{in_channels}.out{out_channels}"] = {
        "pt-rocm": dur1,
        "tunned-ck": dur2,
    }

    avg_dur1 = sum(dur1[warm_ups:]) / (n_runs - warm_ups)
    avg_dur2 = sum(dur2[warm_ups:]) / (n_runs - warm_ups)
    logger.info("Baseline: {:.10f}s".format(avg_dur1))
    logger.info("CK Linear: {:.10f}s ~ {:.2f}%\n".format(avg_dur2, avg_dur2 / avg_dur1 * 100))

torch.save(all_durs, "save/all_durations.pth")
