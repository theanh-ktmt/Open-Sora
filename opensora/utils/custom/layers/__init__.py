"""Replace Open-Sora layers with our customed layers.
This equals to hook before compiled."""

import torch.nn as nn
from loguru import logger

from opensora.utils.custom.compile import get_custom_backend


def replace_with_custom_layers(module: nn.Module) -> nn.Module:
    """Replace all module layers with custom layers if custom_backend is not None."""
    custom_backend = get_custom_backend()

    if custom_backend is None or not isinstance(module, nn.Module):
        logger.info("Customed backend is None or module is not nn.Module. Keep the origin module.")
        return module

    if custom_backend == "ck":
        module = replace_with_ck_layers(module)
    elif custom_backend == "hipblaslt":
        module = replace_with_hipblaslt_layers(module)
    else:
        raise NotImplementedError(
            "Backend '{}' is currently not supported for hooking before 'torch.compile'!".format(custom_backend)
        )

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
