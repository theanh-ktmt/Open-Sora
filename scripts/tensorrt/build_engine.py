import argparse

from opensora.utils.custom.tensorrt import build_engine


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx-path", type=str, required=True, help="Path to ONNX file")
    parser.add_argument("--engine-path", type=str, required=True, help="Path to save engine")
    parser.add_argument("--fp16", action="store_true", help="Whether FP16 mode is enabled or not")
    parser.add_argument("--workspace-size", type=int, default=80, help="Size in GB use for temporary storing")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Build engine
    build_engine(
        args.onnx_path,  # onnx path
        args.engine_path,  # path to save engine
        workspace_size=args.workspace_size,  # available work space
        use_fp16=args.fp16,  # whether to turn on or off FP16
        layers_to_keep_fp32=[],  # layer to keeps in FP32
    )


if __name__ == "__main__":
    main()
