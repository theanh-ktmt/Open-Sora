import torch
from gemm_ck import (
    gemm_Afp16_Bfp16_Cfp16_8192x1536x1536,
    gemm_Afp16_Bfp16_Cfp16_8192x1536x6144,
    gemm_Afp16_Bfp16_Cfp16_8192x6144x1536,
    gemm_Afp16_Bfp16_Cfp16_216064x1152x1152,
    gemm_Afp16_Bfp16_Cfp16_216064x1152x4608,
    gemm_Afp16_Bfp16_Cfp16_216064x3456x1152,
    gemm_Afp16_Bfp16_Cfp16_216064x4608x1152,
)

__all__ = [
    "gemm_Afp16_Bfp16_Cfp16_8192x6144x1536_op",
    "gemm_Afp16_Bfp16_Cfp16_8192x1536x6144_op",
    "gemm_Afp16_Bfp16_Cfp16_8192x1536x1536_op",
    "gemm_Afp16_Bfp16_Cfp16_216064x4608x1152_op",
    "gemm_Afp16_Bfp16_Cfp16_216064x1152x4608_op",
    "gemm_Afp16_Bfp16_Cfp16_216064x3456x1152_op",
    "gemm_Afp16_Bfp16_Cfp16_216064x1152x1152_op",
]


def _register_gemm(gemm_instance, M, N, K):
    def _fwd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        gemm_instance(A, B, C)
        return C

    def _fake_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.empty((M, N), device=A.device, dtype=A.dtype)

    op_name = f"gemm_ck::{gemm_instance.__name__}_op"
    op = torch.library.custom_op(op_name, _fwd, mutates_args=(), device_types=("cuda",))
    torch.library.register_fake(op_name, _fake_impl)

    return op


gemm_Afp16_Bfp16_Cfp16_8192x6144x1536_op = _register_gemm(gemm_Afp16_Bfp16_Cfp16_8192x6144x1536, 8192, 6144, 1536)
gemm_Afp16_Bfp16_Cfp16_8192x1536x6144_op = _register_gemm(gemm_Afp16_Bfp16_Cfp16_8192x1536x6144, 8192, 1536, 6144)
gemm_Afp16_Bfp16_Cfp16_8192x1536x1536_op = _register_gemm(gemm_Afp16_Bfp16_Cfp16_8192x1536x1536, 8192, 1536, 1536)
gemm_Afp16_Bfp16_Cfp16_216064x4608x1152_op = _register_gemm(gemm_Afp16_Bfp16_Cfp16_216064x4608x1152, 216064, 4608, 1152)
gemm_Afp16_Bfp16_Cfp16_216064x1152x4608_op = _register_gemm(gemm_Afp16_Bfp16_Cfp16_216064x1152x4608, 216064, 1152, 4608)
gemm_Afp16_Bfp16_Cfp16_216064x3456x1152_op = _register_gemm(gemm_Afp16_Bfp16_Cfp16_216064x3456x1152, 216064, 3456, 1152)
gemm_Afp16_Bfp16_Cfp16_216064x1152x1152_op = _register_gemm(gemm_Afp16_Bfp16_Cfp16_216064x1152x1152, 216064, 1152, 1152)
