import os
from typing import List

import torch
import triton
import triton.language as tl

from opensora.utils.custom.triton_kernels.utils import round
from opensora.utils.misc import create_logger

logger = create_logger()

configs = [triton.Config({"BLOCK_SIZE": bs}, num_stages=ns, num_warps=nw) for bs in [2048] for ns in [4] for nw in [4]]
if os.environ.get("AUTOTUNE", "0") == "1":
    logger.info("Enable autotuning for triton layer norm kernels")
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
    key=["stride", "N"],
)
@triton.jit
def _layer_norm_fused_with_int8_dynamic_quant(
    X,  # pointer to the input
    Y,  # pointer to the output
    SCALE,  # pointer to the scale vector
    SHIFT,  # pointer to the shift vector
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    stride_scale,  # how much to increase the pointer when moving by 1 row of SCALE
    stride_shift,  # how much to increase the pointer when moving by 1 row of SHIFT
    SD,  # Second dimension of X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    X += row * stride
    SCALE = SCALE + (row // SD) * stride_scale
    SHIFT = SHIFT + (row // SD) * stride_shift

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    scale = tl.load(SCALE + cols, mask=mask, other=-1.0)
    shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
    y = x * rstd * (1 + scale) + shift
    max_val = tl.max(tl.abs(y))
    max_val /= 127

    y = y / max_val
    y = round(y)

    tl.store(S + row, max_val)
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=configs,
    key=["stride", "N"],
)
@triton.jit
def _weighted_layer_norm_fused_with_int8_dynamic_quant(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weight vector
    SCALE,  # pointer to the scale vector
    SHIFT,  # pointer to the shift vector
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    stride_scale,  # how much to increase the pointer when moving by 1 row of SCALE
    stride_shift,  # how much to increase the pointer when moving by 1 row of SHIFT
    SD,  # Second dimension of X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    X += row * stride
    SCALE = SCALE + (row // SD) * stride_scale
    SHIFT = SHIFT + (row // SD) * stride_shift

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N
    x = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    scale = tl.load(SCALE + cols, mask=mask, other=-1.0)
    shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
    y = (x * rstd * (1 + scale) + shift) * w
    max_val = tl.max(tl.abs(y))
    max_val /= 127

    y = y / max_val
    y = round(y)

    tl.store(S + row, max_val)
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=configs,
    key=["stride", "N"],
)
@triton.jit
def _layer_norm_fused_with_int8_dynamic_quant_and_shift_scale_zero(
    X,  # pointer to the input
    Y,  # pointer to the output
    Y_ZERO,  # pointer to the output
    SCALE,  # pointer to the scale vector
    SHIFT,  # pointer to the shift vector
    SCALE_ZERO,  # pointer to the scale vector
    SHIFT_ZERO,  # pointer to the shift vector
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    stride_scale,  # how much to increase the pointer when moving by 1 row of SCALE
    stride_shift,  # how much to increase the pointer when moving by 1 row of SHIFT
    stride_scale_zero,  # how much to increase the pointer when moving by 1 row of SCALE_ZERO
    stride_shift_zero,  # how much to increase the pointer when moving by 1 row of SHIFT_ZERO
    SD,  # Second dimension of X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    Y_ZERO += row * N
    X += row * stride
    SCALE = SCALE + (row // SD) * stride_scale
    SHIFT = SHIFT + (row // SD) * stride_shift
    SCALE_ZERO = SCALE_ZERO + (row // SD) * stride_scale_zero
    SHIFT_ZERO = SHIFT_ZERO + (row // SD) * stride_shift_zero

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    scale = tl.load(SCALE + cols, mask=mask, other=-1.0)
    shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
    scale_zero = tl.load(SCALE_ZERO + cols, mask=mask, other=-1.0)
    shift_zero = tl.load(SHIFT_ZERO + cols, mask=mask, other=0.0)

    norm_x = x * rstd
    y = norm_x * (1 + scale) + shift
    y_zero = norm_x * (1 + scale_zero) + shift_zero

    max_val = tl.max(tl.abs(y))
    max_val_zero = tl.max(tl.abs(y_zero))
    max_val = tl.where(max_val > max_val_zero, max_val, max_val_zero)
    max_val /= 127

    y = y / max_val
    y_zero = y_zero / max_val
    y = round(y)
    y_zero = round(y_zero)

    tl.store(S + row, max_val)
    tl.store(Y + cols, y, mask=mask)
    tl.store(Y_ZERO + cols, y_zero, mask=mask)


@triton.autotune(
    configs=configs,
    key=["stride", "N"],
)
@triton.jit
def _weighted_layer_norm_fused_with_int8_dynamic_quant_and_shift_scale_zero(
    X,  # pointer to the input
    Y,  # pointer to the output
    Y_ZERO,  # pointer to the output
    W,  # pointer to the weight vector
    SCALE,  # pointer to the scale vector
    SHIFT,  # pointer to the shift vector
    SCALE_ZERO,  # pointer to the scale vector
    SHIFT_ZERO,  # pointer to the shift vector
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    stride_scale,  # how much to increase the pointer when moving by 1 row of SCALE
    stride_shift,  # how much to increase the pointer when moving by 1 row of SHIFT
    stride_scale_zero,  # how much to increase the pointer when moving by 1 row of SCALE_ZERO
    stride_shift_zero,  # how much to increase the pointer when moving by 1 row of SHIFT_ZERO
    SD,  # Second dimension of X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    Y_ZERO += row * N
    X += row * stride
    SCALE = SCALE + (row // SD) * stride_scale
    SHIFT = SHIFT + (row // SD) * stride_shift
    SCALE_ZERO = SCALE_ZERO + (row // SD) * stride_scale_zero
    SHIFT_ZERO = SHIFT_ZERO + (row // SD) * stride_shift_zero

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N
    x = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    scale = tl.load(SCALE + cols, mask=mask, other=-1.0)
    shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
    scale_zero = tl.load(SCALE_ZERO + cols, mask=mask, other=-1.0)
    shift_zero = tl.load(SHIFT_ZERO + cols, mask=mask, other=0.0)

    norm_x = x * rstd
    y = (norm_x * (1 + scale) + shift) * w
    y_zero = (norm_x * (1 + scale_zero) + shift_zero) * w

    max_val = tl.max(tl.abs(y))
    max_val_zero = tl.max(tl.abs(y_zero))
    max_val = tl.where(max_val > max_val_zero, max_val, max_val_zero)
    max_val /= 127

    y = y / max_val
    y_zero = y_zero / max_val
    y = round(y)
    y_zero = round(y_zero)

    tl.store(S + row, max_val)
    tl.store(Y + cols, y, mask=mask)
    tl.store(Y_ZERO + cols, y_zero, mask=mask)


@torch.library.custom_op("triton_layer_norm_with_int8_dynamic_quant::triton_op", mutates_args=(), device_types="cuda")
def layer_norm_with_int8_dynamic_quant(
    x: torch.Tensor,
    normalized_shape: List[float],
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_zero: torch.Tensor = None,
    shift_zero: torch.Tensor = None,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    org_shape = x.shape

    assert len(x.shape) == 3
    assert scale.shape[0] == x.shape[0] and shift.shape[0] == x.shape[0]
    assert scale.shape[-1] == x.shape[-1] and shift.shape[-1] == x.shape[-1]
    if len(scale.shape) > 2:
        for i in range(1, len(scale.shape) - 1):
            assert scale.shape[i] == 1 and shift.shape[i] == 1

    SD = x.shape[1]  # Second dimension of x
    x_arg = x.reshape(-1, x.shape[-1])
    y = torch.empty(x_arg.shape, dtype=torch.int8, device=x.device)
    y_zero = torch.empty(x_arg.shape, dtype=torch.int8, device=x.device)

    M, N = x_arg.shape
    quant_scale = torch.empty((M,), dtype=torch.float32, device=x.device)

    if weight is not None and bias is not None:
        raise NotImplementedError
    if scale_zero is None and shift_zero is None:
        if weight is not None:
            _weighted_layer_norm_fused_with_int8_dynamic_quant[(M,)](  #
                x_arg,
                y,
                weight,
                scale,
                shift,
                quant_scale,  #
                x_arg.stride(0),
                scale.stride(0),
                shift.stride(0),
                SD,
                N,
                eps,
            )
        else:
            _layer_norm_fused_with_int8_dynamic_quant[(M,)](  #
                x_arg,
                y,
                scale,
                shift,
                quant_scale,  #
                x_arg.stride(0),
                scale.stride(0),
                shift.stride(0),
                SD,
                N,
                eps,
            )
    elif scale_zero is not None and shift_zero is not None:
        assert scale_zero.shape == scale.shape and shift_zero.shape == shift.shape
        if weight is not None:
            _weighted_layer_norm_fused_with_int8_dynamic_quant_and_shift_scale_zero[(M,)](  #
                x_arg,
                y,
                y_zero,
                weight,
                scale,
                shift,
                scale_zero,
                shift_zero,
                quant_scale,  #
                x_arg.stride(0),
                scale.stride(0),
                shift.stride(0),
                scale_zero.stride(0),
                shift_zero.stride(0),
                SD,
                N,
                eps,
            )
        else:
            _layer_norm_fused_with_int8_dynamic_quant_and_shift_scale_zero[(M,)](  #
                x_arg,
                y,
                y_zero,
                scale,
                shift,
                scale_zero,
                shift_zero,
                quant_scale,  #
                x_arg.stride(0),
                scale.stride(0),
                shift.stride(0),
                scale_zero.stride(0),
                shift_zero.stride(0),
                SD,
                N,
                eps,
            )
    else:
        raise NotImplementedError
    return y.view(org_shape), y_zero.view(org_shape), quant_scale


@layer_norm_with_int8_dynamic_quant.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: List[float],
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_zero: torch.Tensor = None,
    shift_zero: torch.Tensor = None,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
    prev_gate_mlp: torch.Tensor = None,
    prev_context_output: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    org_shape = x.shape
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    return (
        torch.empty(org_shape, dtype=torch.int8, device=x.device),
        torch.empty(org_shape, dtype=torch.int8, device=x.device),
        torch.empty((M,), dtype=torch.float32, device=x.device),
    )


###############################################################################################################################################
##################################################################### FP8 #####################################################################
###############################################################################################################################################


@triton.autotune(
    configs=configs,
    key=["stride", "N"],
)
@triton.jit
def _layer_norm_fused_with_fp8_dynamic_quant(
    X,  # pointer to the input
    Y,  # pointer to the output
    SCALE,  # pointer to the scale vector
    SHIFT,  # pointer to the shift vector
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    stride_scale,  # how much to increase the pointer when moving by 1 row of SCALE
    stride_shift,  # how much to increase the pointer when moving by 1 row of SHIFT
    SD,  # Second dimension of X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    X += row * stride
    SCALE = SCALE + (row // SD) * stride_scale
    SHIFT = SHIFT + (row // SD) * stride_shift

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    scale = tl.load(SCALE + cols, mask=mask, other=-1.0)
    shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
    y = x * rstd * (1 + scale) + shift
    max_val = tl.max(tl.abs(y))
    max_val /= 240

    y = y / max_val
    y = tl.cast(y, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")

    tl.store(S + row, max_val)
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=configs,
    key=["stride", "N"],
)
@triton.jit
def _weighted_layer_norm_fused_with_fp8_dynamic_quant(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weight vector
    SCALE,  # pointer to the scale vector
    SHIFT,  # pointer to the shift vector
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    stride_scale,  # how much to increase the pointer when moving by 1 row of SCALE
    stride_shift,  # how much to increase the pointer when moving by 1 row of SHIFT
    SD,  # Second dimension of X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    X += row * stride
    SCALE = SCALE + (row // SD) * stride_scale
    SHIFT = SHIFT + (row // SD) * stride_shift

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N
    x = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    scale = tl.load(SCALE + cols, mask=mask, other=-1.0)
    shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
    y = (x * rstd * (1 + scale) + shift) * w
    max_val = tl.max(tl.abs(y))
    max_val /= 240

    y = y / max_val
    y = tl.cast(y, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")

    tl.store(S + row, max_val)
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=configs,
    key=["stride", "N"],
)
@triton.jit
def _layer_norm_fused_with_fp8_dynamic_quant_and_shift_scale_zero(
    X,  # pointer to the input
    Y,  # pointer to the output
    Y_ZERO,  # pointer to the output
    SCALE,  # pointer to the scale vector
    SHIFT,  # pointer to the shift vector
    SCALE_ZERO,  # pointer to the scale vector
    SHIFT_ZERO,  # pointer to the shift vector
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    stride_scale,  # how much to increase the pointer when moving by 1 row of SCALE
    stride_shift,  # how much to increase the pointer when moving by 1 row of SHIFT
    stride_scale_zero,  # how much to increase the pointer when moving by 1 row of SCALE_ZERO
    stride_shift_zero,  # how much to increase the pointer when moving by 1 row of SHIFT_ZERO
    SD,  # Second dimension of X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    Y_ZERO += row * N
    X += row * stride
    SCALE = SCALE + (row // SD) * stride_scale
    SHIFT = SHIFT + (row // SD) * stride_shift
    SCALE_ZERO = SCALE_ZERO + (row // SD) * stride_scale_zero
    SHIFT_ZERO = SHIFT_ZERO + (row // SD) * stride_shift_zero

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    scale = tl.load(SCALE + cols, mask=mask, other=-1.0)
    shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
    scale_zero = tl.load(SCALE_ZERO + cols, mask=mask, other=-1.0)
    shift_zero = tl.load(SHIFT_ZERO + cols, mask=mask, other=0.0)

    norm_x = x * rstd
    y = norm_x * (1 + scale) + shift
    y_zero = norm_x * (1 + scale_zero) + shift_zero

    max_val = tl.max(tl.abs(y))
    max_val_zero = tl.max(tl.abs(y_zero))
    max_val = tl.where(max_val > max_val_zero, max_val, max_val_zero)
    max_val /= 240

    y = y / max_val
    y_zero = y_zero / max_val
    y = tl.cast(y, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")
    y_zero = tl.cast(y_zero, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")

    tl.store(S + row, max_val)
    tl.store(Y + cols, y, mask=mask)
    tl.store(Y_ZERO + cols, y_zero, mask=mask)


@triton.autotune(
    configs=configs,
    key=["stride", "N"],
)
@triton.jit
def _weighted_layer_norm_fused_with_fp8_dynamic_quant_and_shift_scale_zero(
    X,  # pointer to the input
    Y,  # pointer to the output
    Y_ZERO,  # pointer to the output
    W,  # pointer to the weight vector
    SCALE,  # pointer to the scale vector
    SHIFT,  # pointer to the shift vector
    SCALE_ZERO,  # pointer to the scale vector
    SHIFT_ZERO,  # pointer to the shift vector
    S,  # pointer to the scaling factor
    stride,  # how much to increase the pointer when moving by 1 row of X
    stride_scale,  # how much to increase the pointer when moving by 1 row of SCALE
    stride_shift,  # how much to increase the pointer when moving by 1 row of SHIFT
    stride_scale_zero,  # how much to increase the pointer when moving by 1 row of SCALE_ZERO
    stride_shift_zero,  # how much to increase the pointer when moving by 1 row of SHIFT_ZERO
    SD,  # Second dimension of X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    Y_ZERO += row * N
    X += row * stride
    SCALE = SCALE + (row // SD) * stride_scale
    SHIFT = SHIFT + (row // SD) * stride_shift
    SCALE_ZERO = SCALE_ZERO + (row // SD) * stride_scale_zero
    SHIFT_ZERO = SHIFT_ZERO + (row // SD) * stride_shift_zero

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N
    x = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    scale = tl.load(SCALE + cols, mask=mask, other=-1.0)
    shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
    scale_zero = tl.load(SCALE_ZERO + cols, mask=mask, other=-1.0)
    shift_zero = tl.load(SHIFT_ZERO + cols, mask=mask, other=0.0)

    norm_x = x * rstd
    y = (norm_x * (1 + scale) + shift) * w
    y_zero = (norm_x * (1 + scale_zero) + shift_zero) * w

    max_val = tl.max(tl.abs(y))
    max_val_zero = tl.max(tl.abs(y_zero))
    max_val = tl.where(max_val > max_val_zero, max_val, max_val_zero)
    max_val /= 240

    y = y / max_val
    y_zero = y_zero / max_val
    y = tl.cast(y, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")
    y_zero = tl.cast(y_zero, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")

    tl.store(S + row, max_val)
    tl.store(Y + cols, y, mask=mask)
    tl.store(Y_ZERO + cols, y_zero, mask=mask)


@torch.library.custom_op("triton_layer_norm_with_fp8_dynamic_quant::triton_op", mutates_args=(), device_types="cuda")
def layer_norm_with_fp8_dynamic_quant(
    x: torch.Tensor,
    normalized_shape: List[float],
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_zero: torch.Tensor = None,
    shift_zero: torch.Tensor = None,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    org_shape = x.shape

    assert len(x.shape) == 3
    assert scale.shape[0] == x.shape[0] and shift.shape[0] == x.shape[0]
    assert scale.shape[-1] == x.shape[-1] and shift.shape[-1] == x.shape[-1]
    if len(scale.shape) > 2:
        for i in range(1, len(scale.shape) - 1):
            assert scale.shape[i] == 1 and shift.shape[i] == 1

    SD = x.shape[1]  # Second dimension of x
    x_arg = x.reshape(-1, x.shape[-1])
    y = torch.empty(x_arg.shape, dtype=torch.float8_e4m3fnuz, device=x.device)
    y_zero = torch.empty(x_arg.shape, dtype=torch.float8_e4m3fnuz, device=x.device)

    M, N = x_arg.shape
    quant_scale = torch.empty((M,), dtype=torch.float32, device=x.device)

    if weight is not None and bias is not None:
        raise NotImplementedError
    if scale_zero is None and shift_zero is None:
        if weight is None:
            _layer_norm_fused_with_fp8_dynamic_quant[(M,)](  #
                x_arg,
                y,
                scale,
                shift,
                quant_scale,  #
                x_arg.stride(0),
                scale.stride(0),
                shift.stride(0),
                SD,
                N,
                eps,
            )
        else:
            _weighted_layer_norm_fused_with_fp8_dynamic_quant(
                x_arg,
                y,
                weight,
                scale,
                shift,
                quant_scale,  #
                x_arg.stride(0),
                scale.stride(0),
                shift.stride(0),
                SD,
                N,
                eps,
            )
    elif scale_zero is not None and shift_zero is not None:
        assert scale_zero.shape == scale.shape and shift_zero.shape == shift.shape
        if weight is None:
            _layer_norm_fused_with_fp8_dynamic_quant_and_shift_scale_zero[(M,)](  #
                x_arg,
                y,
                y_zero,
                scale,
                shift,
                scale_zero,
                shift_zero,
                quant_scale,  #
                x_arg.stride(0),
                scale.stride(0),
                shift.stride(0),
                scale_zero.stride(0),
                shift_zero.stride(0),
                SD,
                N,
                eps,
            )
        else:
            _weighted_layer_norm_fused_with_fp8_dynamic_quant_and_shift_scale_zero[(M,)](  #
                x_arg,
                y,
                y_zero,
                weight,
                scale,
                shift,
                scale_zero,
                shift_zero,
                quant_scale,  #
                x_arg.stride(0),
                scale.stride(0),
                shift.stride(0),
                scale_zero.stride(0),
                shift_zero.stride(0),
                SD,
                N,
                eps,
            )
    else:
        raise NotImplementedError
    return y.view(org_shape), y_zero.view(org_shape), quant_scale


@layer_norm_with_fp8_dynamic_quant.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: List[float],
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_zero: torch.Tensor = None,
    shift_zero: torch.Tensor = None,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
    prev_gate_mlp: torch.Tensor = None,
    prev_context_output: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    org_shape = x.shape
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    return (
        torch.empty(org_shape, dtype=torch.float8_e4m3fnuz, device=x.device),
        torch.empty(org_shape, dtype=torch.float8_e4m3fnuz, device=x.device),
        torch.empty((M,), dtype=torch.float32, device=x.device),
    )
