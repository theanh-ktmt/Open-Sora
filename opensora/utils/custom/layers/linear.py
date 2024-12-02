import torch
import torch.nn as nn

from opensora.utils.custom.operators.matmul import ck_matmul_linear, get_ck_matmul_ops


class CustomedCKLinear(nn.Module):
    """Customed linear layer using CK operators."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight_T = linear.weight.transpose(0, 1).contiguous()  # Transpose and make contiguous
        self.in_channels, self.out_channels = self.weight_T.shape
        self.bias = linear.bias

        self.dtype = linear.weight.dtype
        self.device = linear.weight.device
        assert self.dtype == torch.float16, f"Data type '{self.dtype}' is not supported!"
        assert "cuda" in str(self.device), f"Device '{self.device}' is not supported!"

        self.matmul_ops = sorted(get_ck_matmul_ops(self.in_channels, self.out_channels, self.dtype), key=lambda x: x[0])

    def forward(self, x: torch.Tensor):
        out = ck_matmul_linear(self.matmul_ops, x.contiguous(), self.weight_T, dtype=self.dtype, device=self.device)

        if self.bias is not None:
            out = out + self.bias
        return out
