import os

import torch.nn as nn
from loguru import logger

from .linear import CustomedCKLinear

ENABLE_CK = os.environ.get("ENABLE_CK", "0") == "1"


def replace_with_custom_layers(module: nn.Module) -> nn.Module:
    """Replace all module layers with custom layers if ENABLE_CK is set to True."""
    if not ENABLE_CK:
        return module

    def replace_child(child_module: nn.Module) -> nn.Module:
        """Recursively replace layers in the given module."""
        for name, child in child_module.named_children():
            if isinstance(child, nn.Linear):
                try:
                    custom_linear = CustomedCKLinear(child)
                    setattr(child_module, name, custom_linear)
                    logger.info(f"Replaced layer '{name}' with '{[op[1].__name__ for op in custom_linear.matmul_ops]}'")
                except Exception as e:
                    logger.warning(f"Could not replace layer '{name}': {e}")
            else:
                replace_child(child)

    replace_child(module)
    logger.success(f"Model after replacements:\n{module}")
    return module
