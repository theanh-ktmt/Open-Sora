import torch
from torch import nn

activations_dict = {}


class CollectLinearInputs(nn.Linear):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    @classmethod
    def from_original_module(cls, module: nn.Linear, name: str = None):
        new_module = cls(
            in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None, name=name
        )
        new_module.weight = module.weight
        new_module.bias = module.bias
        return new_module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.name not in activations_dict:
            activations_dict[self.name] = input.abs().reshape(-1, input.shape[-1]).max(dim=0).values
        else:
            activations_dict[self.name] = torch.max(
                activations_dict[self.name], input.abs().reshape(-1, input.shape[-1]).max(dim=0).values
            )

        return super().forward(input)
