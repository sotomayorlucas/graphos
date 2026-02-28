"""Export ComposedRouterModel to ONNX with static shapes for NPU."""

import argparse
import os

import torch
import onnx

from core.constants import NUM_ROUTES, DEFAULT_BATCH_SIZE
from kernel.programs.composed_router import ComposedRouterModel, COMPOSED_INPUT_DIM


def export_composed_router(
    onnx_path: str = "models/composed_router_b1.onnx",
    batch_size: int = 1,
    opset_version: int = 18,
):
    """Export composed router model to ONNX."""
    model = ComposedRouterModel()
    model.eval()

    dummy_input = torch.randn(batch_size, COMPOSED_INPUT_DIM)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["composed_input"],
        output_names=["route_scores"],
        dynamic_axes=None,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print(f"Composed router ONNX exported to: {onnx_path}")
    print(f"  Input:  composed_input  shape=({batch_size}, {COMPOSED_INPUT_DIM})")
    print(f"  Output: route_scores    shape=({batch_size}, {NUM_ROUTES})")
    print(f"  Opset:  {opset_version}")
    print("  Validation: PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ComposedRouter to ONNX")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    onnx_path = f"models/composed_router_b{args.batch_size}.onnx"
    export_composed_router(onnx_path=onnx_path, batch_size=args.batch_size)
