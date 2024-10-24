import argparse
import time

import onnx
from loguru import logger

# Load argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=str, default="save/onnx/ckpts/144p-2s/stdit3.onnx", help="Path to original ONNX file."
)
parser.add_argument(
    "--output",
    type=str,
    default="save/onnx/ckpts/144p-2s/stdit3_inferred.onnx",
    help="Path to shape-inferred ONNX file.",
)
args = parser.parse_args()

# Infer shape
logger.info("Shape inferencing ONNX from {}...".format(args.input))
start = time.time()
onnx.shape_inference.infer_shapes_path(
    args.input,
    args.output,
    # check_type=True,  # Check types during shape inference
    # strict_mode=True,  # Enforce stricter rules
    # data_prop=True,  # Perform data propagation
)
logger.success("Done after {:.2f}s!".format(time.time() - start))
logger.info("Saved new ONNX checkpoint to {}!".format(args.output))
