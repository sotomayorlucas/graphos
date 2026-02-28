"""Tests for Phase 5: Action Routing — all offline, no Npcap or model required."""

import os
import tempfile
import threading

import numpy as np
import pytest

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.dns import DNS, DNSQR
from scapy.layers.l2 import Ether
from scapy.packet import Raw
from scapy.utils import wrpcap, rdpcap

from dataflow.primitives import _STOP, Channel
from dataflow.graph import Graph
from dataflow.scheduler import Scheduler
from dataflow.nodes.source import SourceNode
from dataflow.nodes.sink import SinkNode
from dataflow.nodes.tee import TeeNode
from dataflow.nodes.router import RouterSink

from actions.base import Action
from actions.counter import CountAction
from actions.log_action import LogAction
from actions.pcap_writer import PcapWriteAction


# ---- Helpers ----

def _build_http_packet():
    return (
        Ether(dst="ff:ff:ff:ff:ff:ff", src="00:11:22:33:44:55")
        / IP(src="192.168.1.1", dst="10.0.0.1", ttl=64)
        / TCP(sport=12345, dport=80, flags="PA")
        / Raw(load=b"GET / HTTP/1.1\r\n")
    )


def _build_dns_packet():
    return (
        Ether(dst="ff:ff:ff:ff:ff:ff", src="00:11:22:33:44:55")
        / IP(src="192.168.1.1", dst="8.8.8.8", ttl=64)
        / UDP(sport=54321, dport=53)
        / DNS(rd=1, qd=DNSQR(qname="example.com"))
    )


def _build_other_packet():
    return (
        Ether(dst="ff:ff:ff:ff:ff:ff", src="00:11:22:33:44:55")
        / IP(src="192.168.1.1", dst="10.0.0.1", proto=1, ttl=64)
        / Raw(load=b"\x08\x00" + b"\x00" * 20)
    )


class RecordingAction(Action):
    """Test action that records all calls."""

    def __init__(self):
        self.calls: list[tuple[bytes, int]] = []
        self.closed = False

    def execute(self, packet: bytes, class_id: int) -> None:
        self.calls.append((packet, class_id))

    def close(self) -> None:
        self.closed = True


# ---- TestTeeNode ----

class TestTeeNode:
    def test_fan_out(self):
        """Both outputs receive identical items and clean _STOP."""
        items = [b"pkt1", b"pkt2", b"pkt3"]
        source = SourceNode("source", items)
        tee = TeeNode("tee")
        sink_out = SinkNode("sink_out")
        sink_copy = SinkNode("sink_copy")

        graph = Graph()
        graph.add_node(source)
        graph.add_node(tee)
        graph.add_node(sink_out)
        graph.add_node(sink_copy)

        graph.connect("source", "out", "tee", "in")
        graph.connect("tee", "out", "sink_out", "in")
        graph.connect("tee", "copy", "sink_copy", "in")

        scheduler = Scheduler(graph)
        scheduler.run()

        assert sink_out.results == items
        assert sink_copy.results == items
        assert tee.items_processed == 3

    def test_empty_source(self):
        """Empty source → both sinks get nothing, clean shutdown."""
        source = SourceNode("source", [])
        tee = TeeNode("tee")
        sink_out = SinkNode("sink_out")
        sink_copy = SinkNode("sink_copy")

        graph = Graph()
        graph.add_node(source)
        graph.add_node(tee)
        graph.add_node(sink_out)
        graph.add_node(sink_copy)

        graph.connect("source", "out", "tee", "in")
        graph.connect("tee", "out", "sink_out", "in")
        graph.connect("tee", "copy", "sink_copy", "in")

        scheduler = Scheduler(graph)
        scheduler.run()

        assert sink_out.results == []
        assert sink_copy.results == []
        assert tee.items_processed == 0


# ---- TestCountAction ----

class TestCountAction:
    def test_counts(self):
        counter = CountAction()
        counter.execute(b"pkt", 0)
        counter.execute(b"pkt", 0)
        counter.execute(b"pkt", 1)
        counter.execute(b"pkt", 2)
        counter.execute(b"pkt", 1)

        assert counter.summary() == {0: 2, 1: 2, 2: 1}

    def test_empty(self):
        counter = CountAction()
        assert counter.summary() == {}

    def test_thread_safety(self):
        counter = CountAction()
        n = 1000

        def writer(class_id):
            for _ in range(n):
                counter.execute(b"pkt", class_id)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.summary() == {0: n, 1: n, 2: n}


# ---- TestLogAction ----

class TestLogAction:
    def test_no_crash(self):
        """LogAction handles raw packet bytes without crashing."""
        pkt = bytes(_build_http_packet())
        action = LogAction(verbose=True)
        action.execute(pkt, 0)
        action.close()

    def test_short_packet(self):
        """LogAction handles very short packets gracefully."""
        action = LogAction(verbose=True)
        action.execute(b"\x00" * 10, 2)
        action.close()

    def test_default_limits_output(self):
        """Non-verbose mode only logs first 10 packets."""
        action = LogAction(verbose=False)
        pkt = bytes(_build_http_packet())
        for _ in range(20):
            action.execute(pkt, 0)
        assert action._count == 20
        action.close()


# ---- TestPcapWriteAction ----

class TestPcapWriteAction:
    def test_write_and_read_back(self):
        """Write packets to per-class pcap files and verify they read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            action = PcapWriteAction(output_dir=tmpdir)

            http_pkt = bytes(_build_http_packet())
            dns_pkt = bytes(_build_dns_packet())

            action.execute(http_pkt, 0)  # TCP_HTTP
            action.execute(dns_pkt, 1)   # UDP_DNS
            action.execute(http_pkt, 0)  # TCP_HTTP again
            action.close()

            # Verify TCP_HTTP.pcap
            http_path = os.path.join(tmpdir, "TCP_HTTP.pcap")
            assert os.path.exists(http_path)
            pkts = rdpcap(http_path)
            assert len(pkts) == 2

            # Verify UDP_DNS.pcap
            dns_path = os.path.join(tmpdir, "UDP_DNS.pcap")
            assert os.path.exists(dns_path)
            pkts = rdpcap(dns_path)
            assert len(pkts) == 1

            # OTHER.pcap should not exist (no class 2 packets written)
            assert not os.path.exists(os.path.join(tmpdir, "OTHER.pcap"))


# ---- TestRouterSink ----

class TestRouterSink:
    def test_dispatch(self):
        """RouterSink dispatches packets to correct actions based on class IDs."""
        recorder = RecordingAction()
        counter = CountAction()
        actions_map = {
            0: [recorder, counter],
            1: [recorder, counter],
            2: [recorder, counter],
        }
        router = RouterSink("router", actions_map)

        # Manually wire channels
        classes_ch = Channel(capacity=64)
        packets_ch = Channel(capacity=64)
        router.inputs["classes"].channel = classes_ch
        router.inputs["packets"].channel = packets_ch

        # Simulate a batch of 3 class IDs
        classes_ch.put([0, 1, 2])
        # Corresponding 3 packets
        packets_ch.put(b"http_pkt")
        packets_ch.put(b"dns_pkt")
        packets_ch.put(b"other_pkt")

        # Send _STOP
        classes_ch.put(_STOP)
        packets_ch.put(_STOP)

        router.run()

        assert recorder.calls == [
            (b"http_pkt", 0),
            (b"dns_pkt", 1),
            (b"other_pkt", 2),
        ]
        assert counter.summary() == {0: 1, 1: 1, 2: 1}
        assert router.results == [0, 1, 2]
        assert recorder.closed

    def test_default_action(self):
        """Unmapped class IDs go to the default action."""
        default = RecordingAction()
        mapped = RecordingAction()
        actions_map = {0: [mapped]}
        router = RouterSink("router", actions_map, default_action=default)

        classes_ch = Channel(capacity=64)
        packets_ch = Channel(capacity=64)
        router.inputs["classes"].channel = classes_ch
        router.inputs["packets"].channel = packets_ch

        classes_ch.put([0, 5])  # class 5 is unmapped
        packets_ch.put(b"pkt0")
        packets_ch.put(b"pkt5")
        classes_ch.put(_STOP)
        packets_ch.put(_STOP)

        router.run()

        assert mapped.calls == [(b"pkt0", 0)]
        assert default.calls == [(b"pkt5", 5)]

    def test_empty_pipeline(self):
        """RouterSink handles immediate _STOP on both channels."""
        recorder = RecordingAction()
        router = RouterSink("router", {0: [recorder]})

        classes_ch = Channel(capacity=64)
        packets_ch = Channel(capacity=64)
        router.inputs["classes"].channel = classes_ch
        router.inputs["packets"].channel = packets_ch

        classes_ch.put(_STOP)
        packets_ch.put(_STOP)

        router.run()

        assert recorder.calls == []
        assert router.results == []
        assert recorder.closed


# ---- TestBuildRoutingPipeline ----

class TestBuildRoutingPipeline:
    def test_graph_validates(self):
        """build_routing_pipeline creates a valid 6-node graph."""
        from capture.pipeline import build_routing_pipeline

        source = SourceNode("source", [b"pkt1"])
        counter = CountAction()
        actions_map = {0: [counter], 1: [counter], 2: [counter]}

        graph, router_sink = build_routing_pipeline(
            source, actions=actions_map,
            model_path="dummy.onnx",  # won't actually load
            batch_size=4,
        )

        graph.validate()
        order = graph.topological_order()
        assert len(order) == 6
        # Source should be first, router_sink should be last
        assert order[0].name == "source"
        assert order[-1].name == "router_sink"


# ---- TestFullRoutingPipeline ----

class TestFullRoutingPipeline:
    @pytest.fixture
    def sample_pcap(self, tmp_path):
        packets = [
            _build_http_packet(),
            _build_dns_packet(),
            _build_other_packet(),
            _build_http_packet(),
            _build_dns_packet(),
        ]
        pcap_path = str(tmp_path / "routing_test.pcap")
        wrpcap(pcap_path, packets)
        return pcap_path

    def _model_exists(self, batch_size):
        path = os.path.join("models", f"router_graph_b{batch_size}.onnx")
        return os.path.exists(path)

    def test_full_pipeline(self, sample_pcap):
        """Full routing pipeline: pcap → classify → route → count."""
        batch_size = 4
        if not self._model_exists(batch_size):
            pytest.skip(f"Model not found for batch_size={batch_size}")

        from capture.nodes import PcapSourceNode
        from capture.pipeline import build_routing_pipeline

        source = PcapSourceNode("source", sample_pcap)
        counter = CountAction()
        recorder = RecordingAction()
        actions_map = {0: [counter, recorder], 1: [counter, recorder], 2: [counter, recorder]}

        graph, router_sink = build_routing_pipeline(
            source, actions=actions_map, batch_size=batch_size,
        )

        scheduler = Scheduler(graph)
        scheduler.run()

        total = sum(counter.summary().values())
        assert total == 5  # 5 packets in sample pcap
        assert len(recorder.calls) == 5
        assert len(router_sink.results) == 5
