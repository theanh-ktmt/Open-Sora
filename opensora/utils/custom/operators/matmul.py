import torch

FP16_GEMM_PREFIX = "gemm_Afp16_Bfp16_Cfp16"
FP16_MATMULS = {name: locals()[name] for name in locals() if name.startswith(FP16_GEMM_PREFIX)}


def get_ck_matmul_op(M: int, N: int, K: int, dtype: torch.dtype):
    """Get CK Matrix Multiplication specific for M, N, K and dtype."""
    if dtype == torch.float16:
        op_name = f"{FP16_GEMM_PREFIX}_{M}x{N}x{K}"
        if op_name in FP16_MATMULS:
            return FP16_MATMULS[op_name]
        raise NotImplementedError(f"Gemm {op_name} not found!")

    raise NotImplementedError(f"Matrix multiplication for data type {dtype} is not supported.")


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

    # Reshape input
    input = input.reshape(-1, in_channels)
    batch_size = input.shape[0]

    # Forward through gemm
    matmul_op = get_ck_matmul_op(batch_size, out_channels, in_channels, dtype)
    output = torch.zeros(batch_size, out_channels, dtype=dtype, device=device)
    _ = matmul_op(input, weight_T, output)
    output = output.reshape(output_shape)
    return output
