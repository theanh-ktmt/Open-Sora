from typing import Optional

import torch
from modiffusion.ops.hipblaslt_addmm import hipblaslt_fp16_addmm_op


def hipblaslt_addmm_linear(
    input: torch.Tensor, weight_T: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """HipBLASLt Add Matrix Multiplication for Linear layer."""
    assert input.shape[-1] == weight_T.shape[0], f"Invalid shape {input.shape[-1]} != {weight_T.shape[0]}."

    # Prepare in-out shape
    in_channels, out_channels = weight_T.shape
    output_shape = list(input.shape)
    output_shape[-1] = out_channels

    # Reshape input
    input = input.reshape(-1, in_channels)  # reshape to 2 dims

    # Forward
    if bias is None:
        _, N = weight_T.shape
        bias = torch.zeros(N, dtype=input.dtype, device=input.device)
    output = hipblaslt_fp16_addmm_op(input, weight_T, bias)

    output = output.reshape(output_shape)
    return output