import re
from typing import Any, List

import torch
import torch.nn.functional as F

from . import gemm_ops


def get_ck_matmul_ops(in_channels: int, out_channels: int, dtype: torch.dtype):
    """
    Get CK Matrix Multiplication operation specific for Linear(in_channels, out_channels, dtype).

    Parameters:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        dtype (torch.dtype): The data type of the matrices.

    Returns:
        list: A list of pairs (op_M, op_matmul).

    Raises:
        NotImplementedError: If the given dtype is not supported or no suitable GEMM operation is found.
    """
    SUPPORTED_DTYPE_PREFIXES = {"torch.float16": "gemm_Afp16_Bfp16_Cfp16"}

    dtype_str = str(dtype)
    if dtype_str not in SUPPORTED_DTYPE_PREFIXES:
        raise NotImplementedError(f"Matrix multiplication for data type {dtype} is not supported.")

    # List all available ops for specified dtype
    prefix = SUPPORTED_DTYPE_PREFIXES[dtype_str]
    available_ops = [op for op in dir(gemm_ops) if op.startswith(prefix)]

    # Filter ops that match the in_channels and out_channels
    matched_ops = []
    for op in available_ops:
        numbers = re.findall(r"(\d+)x(\d+)x(\d+)", op)
        op_M, op_N, op_K = list(map(int, numbers[0]))
        if op_N == out_channels and op_K == in_channels:
            matched_ops.append((op_M, getattr(gemm_ops, op)))

    if not matched_ops:
        raise NotImplementedError(f"No suitable Gemm operation found for Linear({in_channels}, {out_channels}).")
    return matched_ops


def get_matmul_op(matmul_ops, M: int):
    """Get suitable matmul op with padding for M."""
    for op_M, op in matmul_ops:
        if op_M >= M:
            return op, op_M - M
    raise NotImplementedError(f"No GEMM operators suitable for M={M}.")


def ck_matmul_linear(
    matmul_ops: List[Any],
    input: torch.Tensor,
    weight_T: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cuda"),
):
    """CK Matrix Multiplication for Linear layer."""
    assert input.is_contiguous() and weight_T.is_contiguous(), "Both tensors used for CK matmul must be contiguous."
    assert input.shape[-1] == weight_T.shape[0], f"Invalid shape {input.shape[-1]} != {weight_T.shape[0]}."

    # Prepare in-out shape
    in_channels, out_channels = weight_T.shape
    output_shape = list(input.shape)
    output_shape[-1] = out_channels

    # Preprocess input
    input = input.reshape(-1, in_channels)  # reshape to 2 dims
    matmul_op, padding = get_matmul_op(matmul_ops, input.shape[0])
    if padding > 0:
        input = F.pad(input, (0, 0, 0, padding), "constant", 0)  # pad M dim to op_M

    # Forward through gemm
    output = torch.zeros(input.shape[0], out_channels, dtype=dtype, device=device)
    _ = matmul_op(input, weight_T, output)

    # Postprocess output
    if padding > 0:
        output = output[:-padding]
    output = output.reshape(output_shape)
    return output
