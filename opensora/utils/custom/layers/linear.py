import torch
import torch.nn as nn

from opensora.utils.custom.operators.matmul import ck_matmul_linear, get_available_shapes

SUPPORTED_DTYPES = [torch.float16]
SUPPORTED_DEVICES = ["cuda"]


class CustomedCKLinear(nn.Module):
    """Customed linear layer using CK operators."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight_T = linear.weight.transpose(0, 1).contiguous()  # Transpose and make contiguous
        self.in_channels, self.out_channels = self.weight_T.shape
        self.bias = linear.bias

        self.dtype = linear.weight.dtype
        self.device = linear.weight.device
        assert (
            self.dtype in SUPPORTED_DTYPES
        ), f"Data type '{self.dtype}' is not supported. Supported dtypes: {SUPPORTED_DTYPES}"
        assert (
            self.device.type in SUPPORTED_DEVICES
        ), f"Device '{self.device}' is not supported. Supported devices: {SUPPORTED_DEVICES}"

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
