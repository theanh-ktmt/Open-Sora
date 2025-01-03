import os
from typing import Optional

import torch
import triton
import triton.language as tl

from opensora.utils.misc import create_logger

logger = create_logger()


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


configs = [
    triton.Config({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk})
    for bm in [128]  # , 1024]
    for bn in [128]  # , 1024]
    for bk in [128]  # , 1024]
]
if os.environ.get("AUTOTUNE", "0") == "1":
    logger.info("Enable autotuning for scaled_mm")
    configs = [
        triton.Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk},
            num_stages=ns,
            num_warps=nw,
        )
        for bm in [32, 64, 128, 256]  # , 1024]
        for bn in [32, 64, 128, 256]  # , 1024]
        for bk in [32, 64, 128, 256]  # , 1024]
        for ns in [2, 4, 6]
        for nw in [2, 4, 8]
    ]


@triton.autotune(
    configs=configs,
    key=[
        "M",
        "N",
        "K",
        "stride_am",
        "stride_ak",
        "stride_bk",
        "stride_bn",
        "stride_cm",
        "stride_cn",
    ],  # This triggers the tuning when input dimensions change
)
@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_input_ptr,
    scale_weight_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # .to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # .to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K)  # .to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_scale_input could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_scale_weight.
    BLOCK_SIZE_scale_input: tl.constexpr = BLOCK_SIZE_M
    BLOCK_SIZE_scale_weight: tl.constexpr = BLOCK_SIZE_N
    offsets_scale_inputm = tl.arange(0, BLOCK_SIZE_scale_input) + (BLOCK_SIZE_scale_input > 1) * pid_m * BLOCK_SIZE_M
    masks_scale_inputm = offsets_scale_inputm < M

    offsets_scale_weightn = tl.arange(0, BLOCK_SIZE_scale_weight) + (BLOCK_SIZE_scale_weight > 1) * pid_n * BLOCK_SIZE_N
    masks_scale_weightn = offsets_scale_weightn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_input_ptrs = scale_input_ptr + offsets_scale_inputm
    scale_weight_ptrs = scale_weight_ptr + offsets_scale_weightn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_input = masks_scale_inputm[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_input = tl.load(scale_input_ptrs[:, None], masks_scale_input)
    # Need to broadcast to the appropriate size, if scale_input is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_weight below.
    scale_input = scale_input.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_input * accumulator.to(tl.float32)

    masks_scale_weight = masks_scale_weightn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_weight = tl.load(scale_weight_ptrs[:, None], masks_scale_weight)
    scale_weight = scale_weight.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_weight.T * accumulator  # .to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # .to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # .to(tl.int64)
    offs_cm = offs_cm  # .to(tl.int64)
    offs_cn = offs_cn  # .to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


# input   - [M, K]
# weight - [K, N]
@torch.library.custom_op("triton_scaled_mm::triton_op", mutates_args=(), device_types="cuda")
def triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_input: torch.Tensor,
    scale_weight: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.shape[0] == weight.shape[1] and bias.dtype == out_dtype

    if len(scale_weight.shape) == 1:
        scale_input = scale_input.view(-1, 1)
        scale_weight = scale_weight.view(-1, 1)

    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype
    assert scale_input.dtype == scale_weight.dtype and scale_input.is_floating_point()
    assert scale_input.shape == torch.Size([1, 1]) or scale_input.shape == torch.Size([M, 1])
    assert scale_weight.shape == torch.Size([1, 1]) or scale_weight.shape == torch.Size([N, 1])
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel[grid](
        input,
        weight,
        scale_input,
        scale_weight,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
    )

    return result.to(out_dtype)


@triton_scaled_mm.register_fake
def _(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_input: torch.Tensor,
    scale_weight: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]
    return torch.empty((M, N), device=input.device, dtype=out_dtype)
