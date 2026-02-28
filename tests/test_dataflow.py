"""Tests for the dataflow graph scheduler framework."""

import os
import queue
import threading
import time

import pytest

from dataflow.primitives import _STOP, Channel, InputPort, OutputPort
from dataflow.node import Node
from dataflow.graph import Graph, GraphError
from dataflow.scheduler import Scheduler, NodeMetrics
from dataflow.nodes.source import SourceNode
from dataflow.nodes.batch import BatchNode
from dataflow.nodes.sink import SinkNode


# ---- Helpers ----

class PassthroughNode(Node):
    """Reads items from 'in', writes them to 'out'."""

    def __init__(self, name: str):
        super().__init__(name)
        self.add_input("in")
        self.add_output("out")

    def process(self):
        inp = self.inputs["in"]
        out = self.outputs["out"]
        while True:
            item = inp.get()
            if item is _STOP:
                out.put(_STOP)
                return
            out.put(item)
            self.items_processed += 1


class DoubleNode(Node):
    """Reads ints from 'in', writes item*2 to 'out'."""

    def __init__(self, name: str):
        super().__init__(name)
        self.add_input("in")
        self.add_output("out")

    def process(self):
        inp = self.inputs["in"]
        out = self.outputs["out"]
        while True:
            item = inp.get()
            if item is _STOP:
                out.put(_STOP)
                return
            out.put(item * 2)
            self.items_processed += 1


# ---- TestChannel ----

class TestChannel:
    def test_put_get_fifo(self):
        ch = Channel(capacity=10)
        ch.put(1)
        ch.put(2)
        ch.put(3)
        assert ch.get() == 1
        assert ch.get() == 2
        assert ch.get() == 3

    def test_capacity_blocking(self):
        ch = Channel(capacity=2)
        ch.put("a")
        ch.put("b")
        assert ch.full
        # put should block/timeout on a full channel
        with pytest.raises(queue.Full):
            ch.put("c", timeout=0.01)

    def test_sentinel_passthrough(self):
        ch = Channel()
        ch.put(_STOP)
        assert ch.get() is _STOP

    def test_items_passed_counter(self):
        ch = Channel()
        assert ch.items_passed == 0
        ch.put(1)
        ch.put(2)
        assert ch.items_passed == 2

    def test_empty_qsize(self):
        ch = Channel()
        assert ch.empty
        assert ch.qsize == 0
        ch.put(42)
        assert not ch.empty
        assert ch.qsize == 1


# ---- TestPorts ----

class TestPorts:
    def test_unconnected_input_raises(self):
        port = InputPort("test")
        assert not port.connected
        with pytest.raises(RuntimeError, match="not connected"):
            port.get()

    def test_unconnected_output_raises(self):
        port = OutputPort("test")
        assert not port.connected
        with pytest.raises(RuntimeError, match="not connected"):
            port.put(42)

    def test_connected_roundtrip(self):
        ch = Channel()
        inp = InputPort("in")
        out = OutputPort("out")
        inp.channel = ch
        out.channel = ch
        assert inp.connected
        assert out.connected
        out.put("hello")
        assert inp.get() == "hello"


# ---- TestGraph ----

class TestGraph:
    def test_add_and_connect(self):
        g = Graph()
        src = SourceNode("src", [])
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(sink)
        g.connect("src", "out", "sink", "in")

    def test_duplicate_node_raises(self):
        g = Graph()
        g.add_node(SourceNode("a", []))
        with pytest.raises(GraphError, match="Duplicate"):
            g.add_node(SourceNode("a", []))

    def test_missing_node_raises(self):
        g = Graph()
        g.add_node(SourceNode("src", []))
        with pytest.raises(GraphError, match="Unknown"):
            g.connect("src", "out", "nonexistent", "in")

    def test_missing_port_raises(self):
        g = Graph()
        g.add_node(SourceNode("src", []))
        g.add_node(SinkNode("sink"))
        with pytest.raises(GraphError, match="no output port"):
            g.connect("src", "bad_port", "sink", "in")

    def test_self_loop_raises(self):
        g = Graph()
        p = PassthroughNode("p")
        g.add_node(p)
        with pytest.raises(GraphError, match="Self-loop"):
            g.connect("p", "out", "p", "in")

    def test_cycle_detection(self):
        g = Graph()
        a = PassthroughNode("a")
        b = PassthroughNode("b")
        # Need extra ports to form a cycle attempt
        a.add_input("in2")
        b.add_output("out2")
        g.add_node(a)
        g.add_node(b)
        g.connect("a", "out", "b", "in")
        with pytest.raises(GraphError, match="cycle"):
            g.connect("b", "out2", "a", "in2")

    def test_topological_order(self):
        g = Graph()
        src = SourceNode("src", [])
        mid = PassthroughNode("mid")
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(mid)
        g.add_node(sink)
        g.connect("src", "out", "mid", "in")
        g.connect("mid", "out", "sink", "in")
        order = g.topological_order()
        names = [n.name for n in order]
        assert names.index("src") < names.index("mid") < names.index("sink")

    def test_source_and_sink_nodes(self):
        g = Graph()
        src = SourceNode("src", [])
        mid = PassthroughNode("mid")
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(mid)
        g.add_node(sink)
        g.connect("src", "out", "mid", "in")
        g.connect("mid", "out", "sink", "in")
        assert [n.name for n in g.source_nodes()] == ["src"]
        assert [n.name for n in g.sink_nodes()] == ["sink"]

    def test_validate_unconnected(self):
        g = Graph()
        g.add_node(PassthroughNode("p"))  # has 'in' and 'out' unconnected
        with pytest.raises(GraphError, match="unconnected"):
            g.validate()


# ---- TestScheduler ----

class TestScheduler:
    def test_two_node_pipeline(self):
        g = Graph()
        src = SourceNode("src", [1, 2, 3])
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(sink)
        g.connect("src", "out", "sink", "in")

        s = Scheduler(g)
        metrics = s.run()

        assert sink.results == [1, 2, 3]
        assert metrics["src"].items_processed == 3
        assert metrics["sink"].items_processed == 3

    def test_three_node_with_double(self):
        g = Graph()
        src = SourceNode("src", [10, 20, 30])
        dbl = DoubleNode("dbl")
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(dbl)
        g.add_node(sink)
        g.connect("src", "out", "dbl", "in")
        g.connect("dbl", "out", "sink", "in")

        s = Scheduler(g)
        metrics = s.run()

        assert sink.results == [20, 40, 60]
        assert metrics["dbl"].items_processed == 3

    def test_empty_source_shutdown(self):
        g = Graph()
        src = SourceNode("src", [])
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(sink)
        g.connect("src", "out", "sink", "in")

        s = Scheduler(g)
        metrics = s.run()

        assert sink.results == []
        assert metrics["src"].items_processed == 0

    def test_wall_time(self):
        g = Graph()
        src = SourceNode("src", range(100))
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(sink)
        g.connect("src", "out", "sink", "in")

        s = Scheduler(g)
        s.run()
        assert s.wall_time > 0

    def test_metrics_throughput(self):
        g = Graph()
        src = SourceNode("src", range(50))
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(sink)
        g.connect("src", "out", "sink", "in")

        s = Scheduler(g)
        metrics = s.run()
        # Source processed 50 items in some positive time
        assert metrics["src"].items_processed == 50
        assert metrics["src"].elapsed > 0


# ---- TestBatchNode ----

class TestBatchNode:
    def test_exact_batches(self):
        """6 items with batch_size=3 -> 2 exact batches."""
        g = Graph()
        src = SourceNode("src", list(range(6)))
        batch = BatchNode("batch", batch_size=3)
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(batch)
        g.add_node(sink)
        g.connect("src", "out", "batch", "in")
        g.connect("batch", "out", "sink", "in")

        s = Scheduler(g)
        s.run()

        # Sink receives tuples (list, count)
        assert len(sink.results) == 2
        assert sink.results[0] == ([0, 1, 2], 3)
        assert sink.results[1] == ([3, 4, 5], 3)

    def test_partial_flush(self):
        """5 items with batch_size=3 -> 1 full + 1 partial."""
        g = Graph()
        src = SourceNode("src", list(range(5)))
        batch = BatchNode("batch", batch_size=3)
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(batch)
        g.add_node(sink)
        g.connect("src", "out", "batch", "in")
        g.connect("batch", "out", "sink", "in")

        s = Scheduler(g)
        s.run()

        assert len(sink.results) == 2
        assert sink.results[0] == ([0, 1, 2], 3)
        assert sink.results[1] == ([3, 4], 2)

    def test_empty_source(self):
        """No items -> no batches emitted."""
        g = Graph()
        src = SourceNode("src", [])
        batch = BatchNode("batch", batch_size=4)
        sink = SinkNode("sink")
        g.add_node(src)
        g.add_node(batch)
        g.add_node(sink)
        g.connect("src", "out", "batch", "in")
        g.connect("batch", "out", "sink", "in")

        s = Scheduler(g)
        s.run()

        assert sink.results == []


# ---- TestFullPipeline (requires model + NPU/CPU) ----

MODEL_B64 = os.path.join("models", "router_graph_b64.onnx")
SKIP_FULL = not os.path.exists(MODEL_B64)


@pytest.mark.skipif(SKIP_FULL, reason=f"Model not found: {MODEL_B64}")
class TestFullPipeline:
    def test_pipeline_classifies_all_packets(self):
        from runtime.benchmark import generate_test_packets, benchmark_cpu
        from dataflow.pipeline import build_classifier_pipeline
        from dataflow.scheduler import Scheduler

        packets = generate_test_packets(n=200)

        graph, sink = build_classifier_pipeline(
            packets, model_path=MODEL_B64, batch_size=64
        )
        scheduler = Scheduler(graph)
        metrics = scheduler.run()

        # All packets classified
        assert len(sink.results) == 200

        # Compare with CPU baseline
        _, cpu_results = benchmark_cpu(packets)
        agree = sum(1 for a, b in zip(sink.results, cpu_results) if a == b)
        accuracy = agree / len(cpu_results) * 100
        assert accuracy > 90, f"Pipeline vs CPU agreement too low: {accuracy:.1f}%"

        # No errors in any node
        for m in metrics.values():
            assert m.error is None, f"Node '{m.name}' had error: {m.error}"
