"""Classifier program spec — wraps existing RouterGraph ONNX model."""

import os

from core.constants import TENSOR_DIM, NUM_CLASSES, DEFAULT_BATCH_SIZE
from kernel.program import ProgramSpec


def classifier_spec(
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_dir: str = "models",
) -> ProgramSpec:
    """Create a ProgramSpec for the packet classifier.

    Uses the existing router_graph ONNX model exported by router.export_onnx.
    """
    if batch_size == 1:
        filename = "router_graph.onnx"
    else:
        filename = f"router_graph_b{batch_size}.onnx"

    return ProgramSpec(
        name="classifier",
        onnx_path=os.path.join(model_dir, filename),
        input_shape=(batch_size, TENSOR_DIM),
        output_shape=(batch_size, NUM_CLASSES),
        description="Packet classifier: HTTP/DNS/OTHER",
    )
