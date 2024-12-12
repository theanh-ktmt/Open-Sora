import os

import torch.nn as nn
from loguru import logger

CUSTOM_BACKEND = os.environ.get("CUSTOM_BACKEND", None)


def replace_with_custom_layers(module: nn.Module) -> nn.Module:
    """Replace all module layers with custom layers if CUSTOM_BACKEND is set."""
    if CUSTOM_BACKEND is None or not isinstance(module, nn.Module):
        return module

    if CUSTOM_BACKEND == "ck":
        module = replace_with_ck_layers(module)
    elif CUSTOM_BACKEND == "hipblaslt":
        module = replace_with_hipblaslt_layers(module)
    else:
        raise NotImplementedError("Backend '{}' is currently not supported!".format(CUSTOM_BACKEND))

    logger.info(f"Model after replacements:\n{module}")
    return module


def replace_with_ck_layers(module: nn.Module) -> nn.Module:
    """Replace all layers inside module with customed CK module."""
    from .linear import CustomedCKLinear

    replace_all_linears(module, CustomedCKLinear)
    return module


def replace_with_hipblaslt_layers(module: nn.Module) -> nn.Module:
    """Replace all layers inside module with customed hipBLASLt module."""
    from .linear import CustomHipblasltLinear

    replace_all_linears(module, CustomHipblasltLinear)
    return module


def replace_all_linears(child_module: nn.Module, customed_linear: nn.Module) -> nn.Module:
    """Recursively replace all linear layers in the given module."""
    SKIPPED_MODULES = [
        "pos_embed",
        "rope",
        "x_embedder",
        "t_embedder",
        "t_block",
        "fps_embedder",
    ]

    for name, child in child_module.named_children():
        if isinstance(child, nn.Linear):
            try:
                custom_linear = customed_linear(child)
                setattr(child_module, name, custom_linear)
                logger.info(f"Replaced layer '{name}' with '{custom_linear}'")
            except Exception as e:
                logger.warning(f"Could not replace layer '{name}': {e}")
        else:
            if all(x not in name for x in SKIPPED_MODULES):
                replace_all_linears(child, customed_linear)
