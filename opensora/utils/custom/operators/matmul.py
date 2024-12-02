import re

import torch
import torch.nn.functional as F

from . import gemm_ops


def get_ck_matmul_op(M: int, N: int, K: int, dtype: torch.dtype):
    """
    Get CK Matrix Multiplication operation specific for dimensions M, N, K, and dtype.

    Parameters:
        M (int): The number of rows in the output matrix.
        N (int): The number of columns in the output matrix.
        K (int): The shared dimension of the input matrices.
        dtype (torch.dtype): The data type of the matrices.

    Returns:
        tuple: A tuple containing the selected GEMM operation and the M padding value.

    Raises:
        NotImplementedError: If the given dtype is not supported or no suitable GEMM operation is found.
    """
    SUPPORTED_DTYPE_PREFIXES = {"torch.float16": "gemm_Afp16_Bfp16_Cfp16"}

    dtype_str = str(dtype)
    if dtype_str not in SUPPORTED_DTYPE_PREFIXES:
        raise NotImplementedError(f"Matrix multiplication for data type {dtype} is not supported.")

    # List all available ops that start with the prefix
    prefix = SUPPORTED_DTYPE_PREFIXES[dtype_str]
    available_ops = [op for op in dir(gemm_ops) if op.startswith(prefix)]

    # Filter ops that match the given N and K
    matched_ops = []
    for op in available_ops:
        numbers = re.findall(r"(\d+)x(\d+)x(\d+)", op)
        op_M, op_N, op_K = list(map(int, numbers[0]))
        if op_N == N and op_K == K:
            matched_ops.append((op_M, op))

    if not matched_ops:
        raise NotImplementedError(f"No suitable Gemm operation found for N={N}, K={K}.")

    # Find the closest op with M >= given M (for padding)
    closest_op = None
    min_pad = float("inf")
    for op_M, op_name in matched_ops:
        if op_M >= M:
            pad = op_M - M
            if pad < min_pad:
                closest_op = op_name
                min_pad = pad

    if closest_op:
        return getattr(gemm_ops, closest_op), min_pad
    else:
        raise NotImplementedError(f"No suitable Gemm operation found for N={N}, K={K}, and M>={M}.")


def ck_matmul_linear(
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
    matmul_op, pad = get_ck_matmul_op(input.shape[0], out_channels, in_channels, dtype)
    if pad > 0:
        input = F.pad(input, (0, 0, 0, pad), "constant", 0)  # pad M dim to op_M

    # Forward through gemm
    output = torch.zeros(input.shape[0], out_channels, dtype=dtype, device=device)
    _ = matmul_op(input, weight_T, output)

    # Postprocess output
    if pad > 0:
        output = output[:-pad]
    output = output.reshape(output_shape)
    return output
