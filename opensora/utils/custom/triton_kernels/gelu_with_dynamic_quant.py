import os

import torch
import triton
import triton.language as tl

from opensora.utils.custom.triton_kernels.utils import round

configs = [triton.Config({"BLOCK_SIZE": bs}, num_stages=ns, num_warps=nw) for bs in [2048] for ns in [3] for nw in [4]]
if os.environ.get("AUTOTUNE", "0") == "1":
    print("Enable autotuning for triton gelu kernels")
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_stages=ns, num_warps=nw)
        for bs in [2048]
        for ns in range(3, 16)
        for nw in [2, 4, 8]
    ]


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


###############################################################################################################################################
##################################################################### INT8 ####################################################################
###############################################################################################################################################


@triton.autotune(
    configs=configs,
    key=["stride", "N"],
)
@triton.jit
def _gelu_fused_with_int8_dynamic_quant(
    X,  # pointer to the input
    Y,  # pointer to the output
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    x1 = tl.load(X + cols).to(tl.float32)
    x2 = tl.load(X + cols + BLOCK_SIZE).to(tl.float32)
    x3 = tl.load(X + cols + BLOCK_SIZE + BLOCK_SIZE).to(tl.float32)

    y1 = gelu(x1)
    y2 = gelu(x2)
    y3 = gelu(x3)

    max_val = tl.max(tl.abs(y1))
    max_val = tl.maximum(max_val, tl.max(tl.abs(y2)))
    max_val = tl.maximum(max_val, tl.max(tl.abs(y3)))
    max_val /= 127

    y1 = y1 / max_val
    y2 = y2 / max_val
    y3 = y3 / max_val

    y1 = round(y1)
    y2 = round(y2)
    y3 = round(y3)

    tl.store(S + row, max_val)
    tl.store(Y + cols, y1)
    tl.store(Y + cols + BLOCK_SIZE, y2)
    tl.store(Y + cols + BLOCK_SIZE + BLOCK_SIZE, y3)


@torch.library.custom_op("triton_gelu_with_int8_dynamic_quant::triton_op", mutates_args=(), device_types="cuda")
def gelu_with_int8_dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # allocate output
    # reshape input data into 2D tensor
    org_shape = x.shape
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    y = torch.empty_like(x_arg, dtype=torch.int8, device=x.device)
    # temp_y = torch.empty_like(x_arg, dtype=torch.float32, device=x.device)
    quant_scale = torch.empty((M,), dtype=torch.float32, device=x.device)

    _gelu_fused_with_int8_dynamic_quant[(M,)](x_arg, y, quant_scale, x_arg.stride(0), N)  #  #

    return y.view(org_shape), quant_scale


@gelu_with_int8_dynamic_quant.register_fake
def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_arg = x.reshape(-1, x.shape[-1])
    M, _ = x_arg.shape
    return torch.empty_like(x, dtype=torch.int8, device=x.device), torch.empty(
        (M,), dtype=torch.float32, device=x.device
    )


###############################################################################################################################################
##################################################################### FP8 #####################################################################
###############################################################################################################################################


@triton.autotune(
    configs=configs,
    key=["stride", "N"],
)
@triton.jit
def _gelu_fused_with_fp8_dynamic_quant(
    X,  # pointer to the input
    Y,  # pointer to the output
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    x1 = tl.load(X + cols).to(tl.float32)
    x2 = tl.load(X + cols + BLOCK_SIZE).to(tl.float32)
    x3 = tl.load(X + cols + BLOCK_SIZE + BLOCK_SIZE).to(tl.float32)

    y1 = gelu(x1)
    y2 = gelu(x2)
    y3 = gelu(x3)

    max_val = tl.max(tl.abs(y1))
    max_val = tl.maximum(max_val, tl.max(tl.abs(y2)))
    max_val = tl.maximum(max_val, tl.max(tl.abs(y3)))

    # Write output
    max_val /= 240
    tl.store(S + row, max_val)

    y1 = y1 / max_val
    y2 = y2 / max_val
    y3 = y3 / max_val

    y1 = tl.cast(y1, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")
    y2 = tl.cast(y2, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")
    y3 = tl.cast(y3, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")

    tl.store(Y + cols, y1)
    tl.store(Y + cols + BLOCK_SIZE, y2)
    tl.store(Y + cols + BLOCK_SIZE + BLOCK_SIZE, y3)


@torch.library.custom_op("triton_gelu_with_fp8_dynamic_quant::triton_op", mutates_args=(), device_types="cuda")
def gelu_with_fp8_dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # allocate output
    # reshape input data into 2D tensor
    org_shape = x.shape
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    y = torch.empty_like(x_arg, dtype=torch.float8_e4m3fnuz, device=x.device)
    # temp_y = torch.empty_like(x_arg, dtype=torch.float32, device=x.device)
    quant_scale = torch.empty((M,), dtype=torch.float32, device=x.device)

    _gelu_fused_with_fp8_dynamic_quant[(M,)](x_arg, y, quant_scale, x_arg.stride(0), N)  #  #

    return y.view(org_shape), quant_scale


@gelu_with_fp8_dynamic_quant.register_fake
def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_arg = x.reshape(-1, x.shape[-1])
    M, _ = x_arg.shape
    return torch.empty_like(x, dtype=torch.float8_e4m3fnuz, device=x.device), torch.empty(
        (M,), dtype=torch.float32, device=x.device
    )
