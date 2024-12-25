from typing import Optional

import torch

from opensora.utils.custom.common import is_cuda, is_rocm

# Register ops
if is_rocm():
    from opensora.utils.custom.operators.flash_attn_triton.rocm_impl import MetaData
    from opensora.utils.custom.operators.flash_attn_triton.rocm_impl import attention as attn_triton_impl

    @torch.library.custom_op("custom_attn::triton_flash_attn_bshd", mutates_args=(), device_types="cuda")
    def triton_flash_attn_bshd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5) if scale is None else scale
        metadata = MetaData(sm_scale)
        metadata.layout = "bshd"
        metadata.max_seqlens_q = q.shape[1]
        metadata.max_seqlens_k = k.shape[1]
        out, _ = attn_triton_impl(q, k, v, None, metadata)
        return out

    @torch.library.custom_op("custom_attn::triton_flash_attn_bhsd", mutates_args=(), device_types="cuda")
    def triton_flash_attn_bhsd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5) if scale is None else scale
        metadata = MetaData(sm_scale)
        metadata.layout = "bhsd"
        metadata.max_seqlens_q = q.shape[2]
        metadata.max_seqlens_k = k.shape[2]
        out, _ = attn_triton_impl(q, k, v, None, metadata)
        return out

elif is_cuda():
    from opensora.utils.custom.operators.flash_attn_triton.cuda_impl import attention as attn_triton_impl

    @torch.library.custom_op("custom_attn::triton_flash_attn_bshd", mutates_args=(), device_types="cuda")
    def triton_flash_attn_bshd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        raise NotImplementedError("Layout 'bshd' is not supported for CUDA.")

    @torch.library.custom_op("custom_attn::triton_flash_attn_bhsd", mutates_args=(), device_types="cuda")
    def triton_flash_attn_bhsd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5) if scale is None else scale
        causal = False
        out, *_ = attn_triton_impl(q, k, v, causal, sm_scale)
        return out.half()

else:
    raise NotImplementedError("Flash-Attention Triton Support ROCm and CUDA only.")


# Register fake
@triton_flash_attn_bshd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    return torch.empty_like(q)


@triton_flash_attn_bhsd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    return torch.empty_like(q)
