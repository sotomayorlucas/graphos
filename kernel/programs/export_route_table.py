"""Export RouteTableModel to ONNX with static shapes for NPU."""

import argparse
import os

import torch
import onnx

from core.constants import TENSOR_DIM, NUM_ROUTES, DEFAULT_BATCH_SIZE
from kernel.programs.route_table import RouteTableModel


def export_route_table(
    onnx_path: str = "models/route_table_b1.onnx",
    batch_size: int = 1,
    opset_version: int = 18,
):
    """Export route table model to ONNX."""
    model = RouteTableModel()
    model.eval()

    dummy_input = torch.randn(batch_size, TENSOR_DIM)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["packet_tensor"],
        output_names=["route_scores"],
        dynamic_axes=None,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print(f"Route table ONNX exported to: {onnx_path}")
    print(f"  Input:  packet_tensor  shape=({batch_size}, {TENSOR_DIM})")
    print(f"  Output: route_scores   shape=({batch_size}, {NUM_ROUTES})")
    print(f"  Opset:  {opset_version}")
    print("  Validation: PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RouteTable to ONNX")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    onnx_path = f"models/route_table_b{args.batch_size}.onnx"
    export_route_table(onnx_path=onnx_path, batch_size=args.batch_size)
