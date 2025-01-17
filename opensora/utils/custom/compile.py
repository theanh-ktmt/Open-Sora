import torch
import torch.nn as nn
from loguru import logger

from opensora.utils.custom.config import ConfigurationManager
from opensora.utils.custom.mlflow import MLFlowManager


def hook_after_compiled():
    """Replace original torch ops with our customed ops."""
    from torch._inductor.select_algorithm import extern_kernels

    assert extern_kernels.mm is torch.mm
    assert extern_kernels.addmm is torch.addmm

    custom_backend = ConfigurationManager.get("CUSTOM_BACKEND")
    if custom_backend is None:

        def custom_func(fn):
            def wrapper(*args, **kwargs):
                # for check hooking
                # print(f"{fn.__name__=}")

                # for shape inference
                # for tensor in filter(lambda x: isinstance(x, torch.Tensor), args):
                #     print(f"{tensor.shape=}, {tensor.stride()=} {tensor.dtype=}")
                return fn(*args, **kwargs)

            return wrapper

        extern_kernels.addmm = custom_func(extern_kernels.addmm)
        extern_kernels.mm = custom_func(extern_kernels.mm)
        logger.info("Skipped hooking after compiled because no backend provided.")

    elif custom_backend == "hipblaslt":
        from modiffusion.ops.hipblaslt_gemm import hipblaslt_addmm_out, hipblaslt_mm_out

        MLFlowManager.set_tag("hook_after", custom_backend)
        extern_kernels.addmm = hipblaslt_addmm_out
        extern_kernels.mm = hipblaslt_mm_out
        logger.info("Done hooked after compile for backend '{}'.".format(custom_backend))
    else:
        raise NotImplementedError(
            "Backend '{}' is currently not supported for hooking after 'torch.compile'!".format(custom_backend)
        )


def compile_module(module: nn.Module):
    """Implement torch.compile for an nn.Module."""
    if ConfigurationManager.get("ENABLE_TORCHCOMPILE"):
        logger.info("Start compiling...")
        MLFlowManager.set_tag("torch.compile")

        # hook custom kernels to after 'torch.compile'
        hook_after_compiled()

        # prepare configs
        configs = ConfigurationManager.get("TORCHCOMPILE_CONFIG")
        MLFlowManager.log_params({f"compile_{k}": v for k, v in configs.items()})

        # start compile
        logger.info("Start compiling...")
        return torch.compile(module, **configs)
    else:
        logger.info("Skip compiling...")
        return module
