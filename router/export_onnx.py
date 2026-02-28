"""Export trained RouterGraph to ONNX with static shapes for NPU."""

import argparse
import os
import sys

import torch
import onnx

from core.constants import TENSOR_DIM, NUM_CLASSES, DEFAULT_BATCH_SIZE
from router.model import RouterGraph


def export(
    checkpoint_path="models/router_graph.pth",
    onnx_path="models/router_graph.onnx",
    opset_version=18,
    batch_size=1,
):
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Run training first: python -m router.train")
        sys.exit(1)

    model = RouterGraph()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    dummy_input = torch.randn(batch_size, TENSOR_DIM)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["packet_tensor"],
        output_names=["class_logits"],
        dynamic_axes=None,  # Static shapes required for NPU
        opset_version=opset_version,
    )

    # Validate the exported model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print(f"ONNX model exported to: {onnx_path}")
    print(f"  Input:  packet_tensor  shape=({batch_size}, {TENSOR_DIM})")
    print(f"  Output: class_logits   shape=({batch_size}, {NUM_CLASSES})")
    print(f"  Opset:  {opset_version}")
    print("  Validation: PASSED")


def export_batched(
    batch_size=DEFAULT_BATCH_SIZE,
    checkpoint_path="models/router_graph.pth",
    opset_version=18,
):
    """Export a batched ONNX model to models/router_graph_b{B}.onnx."""
    onnx_path = f"models/router_graph_b{batch_size}.onnx"
    export(
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        opset_version=opset_version,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RouterGraph to ONNX")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for the exported model (default: 1)",
    )
    args = parser.parse_args()

    if args.batch_size == 1:
        export()
    else:
        export_batched(batch_size=args.batch_size)
