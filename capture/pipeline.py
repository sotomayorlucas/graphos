"""Helper to build a capture-based classifier pipeline."""

import os

from core.constants import DEFAULT_BATCH_SIZE
from dataflow.graph import Graph
from dataflow.node import Node
from dataflow.nodes.batch import BatchNode
from dataflow.nodes.tensor import TensorNode
from dataflow.nodes.infer import InferNode
from dataflow.nodes.sink import SinkNode
from dataflow.nodes.tee import TeeNode
from dataflow.nodes.router import RouterSink


def build_capture_pipeline(
    source_node: Node,
    model_path: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    channel_capacity: int | None = None,
    device: str = "NPU",
) -> tuple[Graph, SinkNode]:
    """Wire source_node -> Batch -> Tensor -> Infer -> Sink.

    Accepts any source node (PcapSourceNode, LiveCaptureNode, etc.)
    that has an "out" output port emitting raw packet bytes.

    Returns (graph, sink_node) so the caller can read sink_node.results.
    """
    if model_path is None:
        model_path = os.path.join("models", f"router_graph_b{batch_size}.onnx")
    if channel_capacity is None:
        channel_capacity = 2 * batch_size

    batch = BatchNode("batch", batch_size)
    tensor = TensorNode("tensor", batch_size)
    infer = InferNode("infer", model_path, device=device)
    sink = SinkNode("sink")

    graph = Graph()
    graph.add_node(source_node)
    graph.add_node(batch)
    graph.add_node(tensor)
    graph.add_node(infer)
    graph.add_node(sink)

    graph.connect(source_node.name, "out", "batch", "in", capacity=channel_capacity)
    graph.connect("batch", "out", "tensor", "in", capacity=channel_capacity)
    graph.connect("tensor", "out", "infer", "in", capacity=channel_capacity)
    graph.connect("infer", "out", "sink", "in", capacity=channel_capacity)

    return graph, sink


def build_routing_pipeline(
    source_node: Node,
    actions: dict,
    model_path: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    channel_capacity: int | None = None,
    device: str = "NPU",
) -> tuple[Graph, RouterSink]:
    """Wire source -> tee -> batch -> tensor -> infer -> router_sink.

    The tee node duplicates the packet stream:
    - 'out' feeds the classification path (batch -> tensor -> infer)
    - 'copy' feeds raw packets directly to the router sink

    Returns (graph, router_sink) for result inspection.
    """
    if model_path is None:
        model_path = os.path.join("models", f"router_graph_b{batch_size}.onnx")
    if channel_capacity is None:
        channel_capacity = 2 * batch_size

    copy_capacity = batch_size * 4

    tee = TeeNode("tee")
    batch = BatchNode("batch", batch_size)
    tensor = TensorNode("tensor", batch_size)
    infer = InferNode("infer", model_path, device=device)
    router_sink = RouterSink("router_sink", actions)

    graph = Graph()
    graph.add_node(source_node)
    graph.add_node(tee)
    graph.add_node(batch)
    graph.add_node(tensor)
    graph.add_node(infer)
    graph.add_node(router_sink)

    # Source -> Tee
    graph.connect(source_node.name, "out", "tee", "in", capacity=channel_capacity)
    # Tee out -> classification path
    graph.connect("tee", "out", "batch", "in", capacity=channel_capacity)
    graph.connect("batch", "out", "tensor", "in", capacity=channel_capacity)
    graph.connect("tensor", "out", "infer", "in", capacity=channel_capacity)
    # Infer -> RouterSink classes
    graph.connect("infer", "out", "router_sink", "classes", capacity=channel_capacity)
    # Tee copy -> RouterSink packets
    graph.connect("tee", "copy", "router_sink", "packets", capacity=copy_capacity)

    return graph, router_sink
