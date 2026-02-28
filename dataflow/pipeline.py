"""Helper to build the standard classifier pipeline."""

import os

from core.constants import DEFAULT_BATCH_SIZE
from dataflow.graph import Graph
from dataflow.nodes.source import SourceNode
from dataflow.nodes.batch import BatchNode
from dataflow.nodes.tensor import TensorNode
from dataflow.nodes.infer import InferNode
from dataflow.nodes.sink import SinkNode


def build_classifier_pipeline(
    packets: list[bytes],
    model_path: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    channel_capacity: int | None = None,
    device: str = "NPU",
) -> tuple[Graph, SinkNode]:
    """Wire Source -> Batch -> Tensor -> Infer -> Sink.

    Returns (graph, sink_node) so the caller can read sink_node.results.
    """
    if model_path is None:
        model_path = os.path.join("models", f"router_graph_b{batch_size}.onnx")
    if channel_capacity is None:
        channel_capacity = 2 * batch_size

    source = SourceNode("source", packets)
    batch = BatchNode("batch", batch_size)
    tensor = TensorNode("tensor", batch_size)
    infer = InferNode("infer", model_path, device=device)
    sink = SinkNode("sink")

    graph = Graph()
    graph.add_node(source)
    graph.add_node(batch)
    graph.add_node(tensor)
    graph.add_node(infer)
    graph.add_node(sink)

    graph.connect("source", "out", "batch", "in", capacity=channel_capacity)
    graph.connect("batch", "out", "tensor", "in", capacity=channel_capacity)
    graph.connect("tensor", "out", "infer", "in", capacity=channel_capacity)
    graph.connect("infer", "out", "sink", "in", capacity=channel_capacity)

    return graph, sink
