import torch
import triton
import triton.language as tl


@triton.jit
def round(y):
    y_floor = tl.floor(y)
    fractional_part = y - y_floor
    mask_round_up = fractional_part >= 0.5
    ret_y = tl.where(mask_round_up, y_floor + 1, y_floor)
    return tl.clamp(ret_y, -128, 127).to(tl.int8)


@triton.jit
def _cast_to_fp8_nearest(
    X,  # pointer to the input
    Y,  # pointer to the output
    stride,  # how much to increase the pointer when moving by 1 row of X
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr = 2048,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(axis=0)
    X += row * stride
    Y += row * stride

    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    for i in range(num_pid_n):
        cols = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0)
        y = tl.cast(x, dtype=tl.float8e4b8, fp_downcast_rounding="rtne")
        tl.store(Y + cols, y, mask=mask)


def cast_to_fp8_nearest(x: torch.Tensor) -> torch.Tensor:
    # allocate output
    # reshape input data into 2D tensor
    org_shape = x.shape
    x_arg = x.reshape(-1, org_shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x_arg, dtype=torch.float8_e4m3fnuz, device=x.device)
    _cast_to_fp8_nearest[(M,)](x_arg, y, x_arg.stride(0), N)  #  #
    return y.view(org_shape)
