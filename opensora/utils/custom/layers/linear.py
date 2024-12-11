import torch
import torch.nn as nn

from opensora.utils.custom.operators.addmm import hipblaslt_addmm_linear
from opensora.utils.custom.operators.matmul import ck_matmul_linear, get_available_shapes


class CustomedCKLinear(nn.Module):
    """Customed linear layer using CK operators."""

    SUPPORTED_DTYPES = [torch.float16]
    SUPPORTED_DEVICES = ["cuda"]

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight_T = linear.weight.transpose(0, 1).contiguous()  # Transpose and make contiguous
        self.in_channels, self.out_channels = self.weight_T.shape
        self.bias = linear.bias

        self.dtype = linear.weight.dtype
        self.device = linear.weight.device
        assert (
            self.dtype in self.SUPPORTED_DTYPES
        ), f"Data type '{self.dtype}' is not supported. Supported dtypes: {self.SUPPORTED_DTYPES}"
        assert (
            self.device.type in self.SUPPORTED_DEVICES
        ), f"Device '{self.device}' is not supported. Supported devices: {self.SUPPORTED_DEVICES}"

        self.available_shapes = sorted(get_available_shapes(self.in_channels, self.out_channels))

    def __repr__(self):
        return (
            f"CKLinear(in_features={self.in_channels}, out_features={self.out_channels}, bias={self.bias is not None})"
        )

    def forward(self, x: torch.Tensor):
        out = ck_matmul_linear(x.contiguous(), self.weight_T, self.available_shapes)

        if self.bias is not None:
            out = out + self.bias
        return out


class CustomHipblasltLinear(nn.Module):
    """Customed linear layer using hipBLASLt operators."""

    SUPPORTED_DTYPES = [torch.float16]
    SUPPORTED_DEVICES = ["cuda"]

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight_T = linear.weight.transpose(0, 1)  # Transpose, not require contiguous
        self.in_channels, self.out_channels = self.weight_T.shape
        self.bias = linear.bias

        self.dtype = linear.weight.dtype
        self.device = linear.weight.device
        assert (
            self.dtype in self.SUPPORTED_DTYPES
        ), f"Data type '{self.dtype}' is not supported. Supported dtypes: {self.SUPPORTED_DTYPES}"
        assert (
            self.device.type in self.SUPPORTED_DEVICES
        ), f"Device '{self.device}' is not supported. Supported devices: {self.SUPPORTED_DEVICES}"

    def __repr__(self):
        return f"HipblasltLinear(in_features={self.in_channels}, out_features={self.out_channels}, bias={self.bias is not None})"

    def forward(self, x: torch.Tensor):
        return hipblaslt_addmm_linear(x, self.weight_T, self.bias)
