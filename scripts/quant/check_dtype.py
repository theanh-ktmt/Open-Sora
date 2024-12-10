import torch
from loguru import logger
from safetensors.torch import load_file


def check_safetensor_checkpoint_dtype(checkpoint_path):
    def log_if_dtype_present(dtype, dtype_name):
        if dtype in dtypes:
            logger.info(f"The checkpoint contains {dtype_name} parameters.")

    tensors = load_file(checkpoint_path)
    dtypes = {tensor.dtype for tensor in tensors.values()}
    logger.info(f"All dtypes: {dtypes}")

    dtype_checks = [
        (torch.int8, "INT8"),
        (torch.float8_e4m3fn, "FP8 E4M3FN"),
        (torch.float8_e4m3fnuz, "FP8 E4M3FNUZ"),
        (torch.float8_e5m2, "FP8 E5M2"),
        (torch.float16, "FP16"),
        (torch.float32, "FP32"),
    ]

    for dtype, dtype_name in dtype_checks:
        log_if_dtype_present(dtype, dtype_name)


if __name__ == "__main__":
    logger.info("SafeTensor Checkpoint:")
    model_path = "save/checkpoints/OpenSora-STDiT-v3-FP8-Naive/model.safetensors"
    check_safetensor_checkpoint_dtype(model_path)
