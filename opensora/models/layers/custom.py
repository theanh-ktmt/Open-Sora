import torch
import torch.nn as nn
from loguru import logger

from opensora.utils.custom.operators.matmul import ck_matmul_linear


class CustomedCKLinear(nn.Module):
    """Customed linear layer using CK operators."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight_T = linear.weight.transpose(0, 1).contiguous()  # Transpose and make contiguous
        self.bias = linear.bias

        self.dtype = linear.weight.dtype
        self.device = linear.weight.device
        assert self.dtype == torch.float16, f"Data type '{self.dtype}' is not supported!"
        assert "cuda" in str(self.device), f"Device '{self.device}' is not supported!"

    def forward(self, x: torch.Tensor):
        try:
            out = ck_matmul_linear(x.contiguous(), self.weight_T, self.dtype, self.device)
        except NotImplementedError as e:
            logger.info(e)
            out = x @ self.weight_T

        if self.bias is not None:
            out = out + self.bias
        return out
