import time

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from opensora.utils.custom.layers.linear import CustomedCKLinear

torch.manual_seed(42)
# _, _ = get_profiling_status()

for in_channels, out_channels in [
    (1152, 4608),
    (4608, 1152),
    (1152, 1152),
    (1152, 3456),
]:
    logger.info("In: {} - Out: {}".format(in_channels, out_channels))
    linear = nn.Linear(in_channels, out_channels, dtype=torch.float16, device="cuda")
    ck_linear = CustomedCKLinear(linear)

    input = torch.rand(2, 108000, in_channels, dtype=torch.float16, device="cuda")

    # Correctness
    out1 = linear(input)
    out2 = ck_linear(input)
    logger.info("Max absolute difference: {}".format(torch.max(torch.abs(out1 - out2)).item()))
    logger.info("All close: {}".format(torch.allclose(out1, out2, rtol=1e-3, atol=1e-3)))

    # Speed
    n_runs = 10000
    warm_ups = 1000

    logger.info("Benchmarking original linear...")
    dur1 = 0
    # with profile(
    #     activities=[
    #         ProfilerActivity.CPU,
    #         ProfilerActivity.CUDA
    #     ],
    #     schedule=schedule(
    #         wait=1000,  # Number of steps to skip
    #         warmup=0,  # Number of steps to include in the warm-up phase
    #         active=5,  # Number of steps to include in the active phase (profiling)
    #         repeat=1,  # Number of times to repeat the above schedule
    #     ),
    #     record_shapes=True,
    #     with_stack=True,
    #     on_trace_ready=trace_handler_wrapper("pt-rocm"),
    # ) as prof:
    for i in tqdm(range(n_runs)):
        start = time.perf_counter()
        out1 = linear(input)
        if i >= warm_ups:
            dur1 += time.perf_counter() - start
        # prof.step()

    logger.info("Benchmarking CK linear...")
    dur2 = 0
    # with profile(
    #     activities=[
    #         ProfilerActivity.CPU,
    #         ProfilerActivity.CUDA
    #     ],
    #     schedule=schedule(
    #         wait=1000,  # Number of steps to skip
    #         warmup=0,  # Number of steps to include in the warm-up phase
    #         active=5,  # Number of steps to include in the active phase (profiling)
    #         repeat=1,  # Number of times to repeat the above schedule
    #     ),
    #     record_shapes=True,
    #     with_stack=True,
    #     on_trace_ready=trace_handler_wrapper("tunned_ck"),
    # ) as prof:
    for i in tqdm(range(n_runs)):
        start = time.perf_counter()
        out2 = ck_linear(input)
        if i >= warm_ups:
            dur2 += time.perf_counter() - start
        # prof.step()

    logger.info("Baseline: {:.10f}s".format(dur1 / (n_runs - warm_ups)))
    logger.info("CK Linear: {:.10f}s ~ {:.2f}%\n".format(dur2 / (n_runs - warm_ups), dur2 / dur1 * 100))
