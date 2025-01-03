import torch

from opensora.models.stdit.quant_stdit3 import QuantSTDiT3Block
from opensora.models.stdit.stdit3 import STDiT3Block


def quant(model: torch.nn.Module, quant_mode: str = "int8", use_smoothquant: bool = True):
    cur_device = model.device
    all_modules = dict(model.named_modules())
    torch.set_grad_enabled(False)

    if quant_mode != "fp16":
        print("Quantizing")
        for name, module in all_modules.items():
            if isinstance(module, STDiT3Block):
                parent_module = all_modules[".".join(name.split(".")[:-1])]
                setattr(
                    parent_module,
                    name.split(".")[-1],
                    QuantSTDiT3Block.from_original_module(
                        module, name=name, quant_mode=quant_mode, use_smoothquant=use_smoothquant
                    ),
                )
        print("Finish quant!")

    model.to(cur_device)
    model.to(torch.float16)
    print(model)
