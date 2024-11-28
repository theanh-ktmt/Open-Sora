import time

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from opensora.models.layers.custom import CustomedCKLinear

torch.manual_seed(42)

linear = nn.Linear(1152, 4608, dtype=torch.float16, device="cuda")
ck_linear = CustomedCKLinear(linear)

input = torch.rand(844, 256, 1152, dtype=torch.float16, device="cuda")

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
for i in tqdm(range(n_runs)):
    start = time.perf_counter()
    out1 = linear(input)
    if i >= warm_ups:
        dur1 += time.perf_counter() - start

logger.info("Benchmarking CK linear...")
dur2 = 0

for i in tqdm(range(n_runs)):
    start = time.perf_counter()
    out2 = ck_linear(input)
    if i >= warm_ups:
        dur2 += time.perf_counter() - start

logger.info("Baseline: {:.10f}s".format(dur1))
logger.info("CK Linear: {:.10f}s ~ {:.2f}%".format(dur2, dur2 / dur1 * 100))
