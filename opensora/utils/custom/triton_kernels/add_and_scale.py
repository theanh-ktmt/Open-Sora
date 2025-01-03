import os

import torch
import triton
import triton.language as tl

from opensora.utils.custom.triton_kernels.utils import round

configs = [triton.Config({"BLOCK_SIZE": bs}, num_stages=ns, num_warps=nw) for bs in [2048] for ns in [4] for nw in [4]]
if os.environ.get("AUTOTUNE", "0") == "1":
    print("Enable autotuning for add_and_scale kernels")
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=ns, num_warps=nw)
        for bs in [2048]
        for ns in range(3, 16)
        for nw in [2, 4, 8]
    ]

###############################################################################################################################################
##################################################################### INT8 ####################################################################
###############################################################################################################################################


@triton.autotune(
    configs=configs,
    key=["N"],
)
@triton.jit
def _add_and_scale(
    X,  # pointer to the input
    Y,  # pointer to the output
    X_STRIDE,  # how much to increase the pointer when moving by 1 row of X
    Y_STRIDE,  # how much to increase the pointer when moving by 1 row of Y
    N,  # number of columns in X
    QSUM,  # pointer to the quantized sum
    ORG_SUM,  # pointer to the original sum
    SCALE,  # pointer to the scaling factor
    SUM_STRIDE,  # how much to increase the pointer when moving by 1 row of SUM
    ORG_SUM_STRIDE,  # how much to increase the pointer when moving by 1 row of ORG_SUM
    SCALE_STRIDE,  # how much to increase the pointer when moving by 1 row of SCALE
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * X_STRIDE
    Y += row * Y_STRIDE
    QSUM += row * SUM_STRIDE
    ORG_SUM += row * SUM_STRIDE
    SCALE += row * SCALE_STRIDE

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=mask, other=0.0).to(tl.float32)
    sum = x + y
    tl.store(ORG_SUM + cols, sum, mask=mask)

    quant_scale = tl.max(tl.abs(sum)) / 127

    sum /= quant_scale
    sum = round(sum)

    tl.store(QSUM + cols, sum, mask=mask)
    tl.store(SCALE, quant_scale)  # .to(tl.float32))


@triton.autotune(
    configs=configs,
    key=["N"],
)
@triton.jit
def _add_and_scale_with_smooth_scale(
    X,  # pointer to the input
    Y,  # pointer to the output
    SMOOTH_SCALE,  # pointer to the smooth scale
    X_STRIDE,  # how much to increase the pointer when moving by 1 row of X
    Y_STRIDE,  # how much to increase the pointer when moving by 1 row of Y
    N,  # number of columns in X
    QSUM,  # pointer to the quantized sum
    ORG_SUM,  # pointer to the original sum
    SCALE,  # pointer to the scaling factor
    SUM_STRIDE,  # how much to increase the pointer when moving by 1 row of SUM
    ORG_SUM_STRIDE,  # how much to increase the pointer when moving by 1 row of ORG_SUM
    SCALE_STRIDE,  # how much to increase the pointer when moving by 1 row of SCALE
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * X_STRIDE
    Y += row * Y_STRIDE
    QSUM += row * SUM_STRIDE
    ORG_SUM += row * SUM_STRIDE
    SCALE += row * SCALE_STRIDE

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0)
    y = tl.load(Y + cols, mask=mask, other=0.0)
    smooth_scale = tl.load(SMOOTH_SCALE + cols, mask=mask)
    sum = x + y
    tl.store(ORG_SUM + cols, sum, mask=mask)

    sum *= smooth_scale
    quant_scale = tl.max(tl.abs(sum)) / 127
    sum /= quant_scale
    sum = round(sum)

    tl.store(QSUM + cols, sum, mask=mask)
    tl.store(SCALE, quant_scale.to(tl.float32))


@torch.library.custom_op("triton_add_and_scale::triton_op", mutates_args=(), device_types="cuda")
def add_and_scale(
    x: torch.Tensor, y: torch.Tensor, smooth_scale: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    org_shape = x.shape

    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.shape[-1] <= 2048, "Kernel for add_and_scale only supports up to 2048 columns"

    x_arg = x.reshape(-1, x.shape[-1])
    y_arg = y.reshape(-1, y.shape[-1])

    M, N = x_arg.shape
    qsum = torch.empty(x_arg.shape, dtype=torch.int8, device=x.device)
    quant_scale = torch.empty((M,), dtype=torch.float32, device=x.device)
    org_sum = torch.empty(x_arg.shape, dtype=x.dtype, device=x.device)

    if smooth_scale is None:
        _add_and_scale[(M,)](
            x_arg,
            y_arg,
            x_arg.stride(0),
            y_arg.stride(0),
            N,
            qsum,
            org_sum,
            quant_scale,
            qsum.stride(0),
            org_sum.stride(0),
            quant_scale.stride(0),
        )
    else:
        _add_and_scale_with_smooth_scale[(M,)](
            x_arg,
            y_arg,
            smooth_scale,
            x_arg.stride(0),
            y_arg.stride(0),
            N,
            qsum,
            org_sum,
            quant_scale,
            qsum.stride(0),
            org_sum.stride(0),
            quant_scale.stride(0),
        )
    return qsum.view(org_shape), quant_scale, org_sum.view(org_shape)


@add_and_scale.register_fake
def _(
    x: torch.Tensor, y: torch.Tensor, smooth_scale: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    org_shape = x.shape
    x_arg = x.reshape(-1, x.shape[-1])
    M, _ = x_arg.shape

    qsum = torch.empty(x_arg.shape, dtype=torch.int8, device=x.device)
    quant_scale = torch.empty((M,), dtype=torch.float32, device=x.device)
    org_sum = torch.empty(x_arg.shape, dtype=x.dtype, device=x.device)

    return qsum.view(org_shape), quant_scale, org_sum.view(org_shape)
