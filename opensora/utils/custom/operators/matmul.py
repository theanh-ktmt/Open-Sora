from typing import List, Tuple

import torch
import torch.nn.functional as F
from modiffusion.ops.ck_mm import SUPPORTED_SHAPES, ck_fp16_mm_op


def get_available_shapes(in_channels: int, out_channels: int) -> List[Tuple[int, int, int]]:
    """
    Get available shapes specific for Linear(in_channels, out_channels, dtype).

    Parameters:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.

    Returns:
        list: A list of shapes (op_M, op_N, op_K).

    Raises:
        NotImplementedError: If the given linear are not supported or not implemented.
    """
    # Filter shapes that match the in_channels and out_channels
    matched_shapes = []
    for shape in SUPPORTED_SHAPES:
        if shape[1] == out_channels and shape[2] == in_channels:
            matched_shapes.append(shape)

    if not matched_shapes:
        raise NotImplementedError(f"No suitable shape found for Linear({in_channels}, {out_channels}).")
    return matched_shapes


def get_shape(available_shapes, M: int) -> Tuple[int, int]:
    """Get suitable shape with padding for M."""
    for shape in available_shapes:
        if shape[0] >= M:
            return shape, shape[0] - M
    raise NotImplementedError(f"No GEMM operators suitable for M={M}.")


def ck_matmul_linear(
    input: torch.Tensor,
    weight_T: torch.Tensor,
    available_shapes: List[Tuple[int]] = [],
) -> torch.Tensor:
    """CK Matrix Multiplication for Linear layer."""
    assert input.is_contiguous() and weight_T.is_contiguous(), "Both tensors used for CK matmul must be contiguous."
    assert input.shape[-1] == weight_T.shape[0], f"Invalid shape {input.shape[-1]} != {weight_T.shape[0]}."

    # Prepare in-out shape
    in_channels, out_channels = weight_T.shape
    output_shape = list(input.shape)
    output_shape[-1] = out_channels

    # Preprocess input
    input = input.reshape(-1, in_channels)  # reshape to 2 dims
    _, padding = get_shape(available_shapes, input.shape[0])
    if padding > 0:
        input = F.pad(input, (0, 0, 0, padding), "constant", 0)  # pad M dim to op_M

    # Forward through gemm
    output = ck_fp16_mm_op(input, weight_T)

    # Postprocess output
    if padding > 0:
        output = output[:-padding]
    output = output.reshape(output_shape)
    return output
