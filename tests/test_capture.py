"""Tests for the capture module — all offline, no Npcap required."""

import os
import tempfile
import queue
import threading

import pytest

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.dns import DNS, DNSQR
from scapy.layers.l2 import Ether
from scapy.packet import Raw
from scapy.utils import wrpcap, rdpcap

from capture.source import PcapSource, CaptureSource
from capture.nodes import PcapSourceNode, LiveCaptureNode
from capture.pipeline import build_capture_pipeline
from dataflow.primitives import _STOP
from dataflow.graph import Graph
from dataflow.scheduler import Scheduler
from dataflow.nodes.sink import SinkNode


# ---- Fixtures ----

def _build_http_packet():
    """Build a realistic HTTP packet (TCP port 80)."""
    return (
        Ether(dst="ff:ff:ff:ff:ff:ff", src="00:11:22:33:44:55")
        / IP(src="192.168.1.1", dst="10.0.0.1", ttl=64)
        / TCP(sport=12345, dport=80, flags="PA")
        / Raw(load=b"GET / HTTP/1.1\r\n")
    )


def _build_dns_packet():
    """Build a realistic DNS query packet (UDP port 53)."""
    return (
        Ether(dst="ff:ff:ff:ff:ff:ff", src="00:11:22:33:44:55")
        / IP(src="192.168.1.1", dst="8.8.8.8", ttl=64)
        / UDP(sport=54321, dport=53)
        / DNS(rd=1, qd=DNSQR(qname="example.com"))
    )


def _build_other_packet():
    """Build an ICMP-like packet (neither HTTP nor DNS)."""
    return (
        Ether(dst="ff:ff:ff:ff:ff:ff", src="00:11:22:33:44:55")
        / IP(src="192.168.1.1", dst="10.0.0.1", proto=1, ttl=64)
        / Raw(load=b"\x08\x00" + b"\x00" * 20)
    )


@pytest.fixture
def sample_pcap(tmp_path):
    """Create a temp pcap file with HTTP, DNS, and OTHER packets."""
    packets = [_build_http_packet(), _build_dns_packet(), _build_other_packet()]
    pcap_path = str(tmp_path / "sample.pcap")
    wrpcap(pcap_path, packets)
    return pcap_path, packets


@pytest.fixture
def large_pcap(tmp_path):
    """Create a temp pcap with many packets for pipeline tests."""
    packets = []
    for i in range(30):
        if i % 3 == 0:
            packets.append(_build_http_packet())
        elif i % 3 == 1:
            packets.append(_build_dns_packet())
        else:
            packets.append(_build_other_packet())
    pcap_path = str(tmp_path / "large.pcap")
    wrpcap(pcap_path, packets)
    return pcap_path, packets


# ---- TestPcapSource ----

class TestPcapSource:
    def test_reads_all_packets(self, sample_pcap):
        pcap_path, original = sample_pcap
        source = PcapSource(pcap_path)
        result = list(source)
        assert len(result) == len(original)

    def test_yields_bytes(self, sample_pcap):
        pcap_path, _ = sample_pcap
        source = PcapSource(pcap_path)
        for pkt in source:
            assert isinstance(pkt, bytes)
            assert len(pkt) > 0

    def test_bytes_match_original(self, sample_pcap):
        pcap_path, original = sample_pcap
        source = PcapSource(pcap_path)
        for got, expected in zip(source, original):
            assert got == bytes(expected)

    def test_empty_pcap(self, tmp_path):
        pcap_path = str(tmp_path / "empty.pcap")
        wrpcap(pcap_path, [])
        source = PcapSource(pcap_path)
        assert list(source) == []

    def test_file_not_found(self):
        with pytest.raises(Exception):
            list(PcapSource("nonexistent.pcap"))


# ---- TestCaptureSource ----

class TestCaptureSource:
    def test_stop_event(self):
        """CaptureSource.stop() sets the event."""
        src = CaptureSource()
        assert not src._stop_event.is_set()
        src.stop()
        assert src._stop_event.is_set()

    def test_queue_based_iteration(self):
        """Manually feed queue and verify iteration."""
        src = CaptureSource()
        # Manually put packets and stop
        src._queue.put(b"packet1")
        src._queue.put(b"packet2")
        src._stop_event.set()  # Signal stop immediately

        results = []
        for pkt in src:
            results.append(pkt)
        assert results == [b"packet1", b"packet2"]


# ---- TestPcapSourceNode ----

class TestPcapSourceNode:
    def test_node_outputs_all_packets(self, sample_pcap):
        pcap_path, original = sample_pcap

        source = PcapSourceNode("source", pcap_path)
        sink = SinkNode("sink")

        g = Graph()
        g.add_node(source)
        g.add_node(sink)
        g.connect("source", "out", "sink", "in")

        s = Scheduler(g)
        s.run()

        assert len(sink.results) == len(original)
        for pkt in sink.results:
            assert isinstance(pkt, bytes)

    def test_node_sends_stop(self, sample_pcap):
        """Verify sink completes (meaning _STOP was received)."""
        pcap_path, _ = sample_pcap

        source = PcapSourceNode("source", pcap_path)
        sink = SinkNode("sink")

        g = Graph()
        g.add_node(source)
        g.add_node(sink)
        g.connect("source", "out", "sink", "in")

        s = Scheduler(g)
        metrics = s.run()

        # No errors means clean _STOP propagation
        for m in metrics.values():
            assert m.error is None

    def test_empty_pcap_node(self, tmp_path):
        pcap_path = str(tmp_path / "empty.pcap")
        wrpcap(pcap_path, [])

        source = PcapSourceNode("source", pcap_path)
        sink = SinkNode("sink")

        g = Graph()
        g.add_node(source)
        g.add_node(sink)
        g.connect("source", "out", "sink", "in")

        s = Scheduler(g)
        s.run()

        assert sink.results == []


# ---- TestBuildCapturePipeline ----

class TestBuildCapturePipeline:
    def test_graph_validates(self, sample_pcap):
        pcap_path, _ = sample_pcap
        source = PcapSourceNode("source", pcap_path)
        graph, sink = build_capture_pipeline(
            source, model_path="fake_model.onnx", batch_size=4
        )
        # Should validate without errors (all ports connected)
        graph.validate()

    def test_graph_topology(self, sample_pcap):
        pcap_path, _ = sample_pcap
        source = PcapSourceNode("source", pcap_path)
        graph, sink = build_capture_pipeline(
            source, model_path="fake_model.onnx", batch_size=4
        )
        order = graph.topological_order()
        names = [n.name for n in order]
        assert names == ["source", "batch", "tensor", "infer", "sink"]

    def test_returns_sink_node(self, sample_pcap):
        pcap_path, _ = sample_pcap
        source = PcapSourceNode("source", pcap_path)
        graph, sink = build_capture_pipeline(
            source, model_path="fake_model.onnx", batch_size=4
        )
        assert isinstance(sink, SinkNode)
        assert sink.name == "sink"


# ---- TestFullCapturePipeline (requires model) ----

MODEL_B64 = os.path.join("models", "router_graph_b64.onnx")
SKIP_FULL = not os.path.exists(MODEL_B64)


@pytest.mark.skipif(SKIP_FULL, reason=f"Model not found: {MODEL_B64}")
class TestFullCapturePipeline:
    def test_pcap_through_full_pipeline(self, large_pcap):
        """Classify packets from a pcap file through the full pipeline."""
        pcap_path, original = large_pcap

        source = PcapSourceNode("source", pcap_path)
        graph, sink = build_capture_pipeline(
            source, model_path=MODEL_B64, batch_size=64
        )
        scheduler = Scheduler(graph)
        metrics = scheduler.run()

        # All packets classified
        assert len(sink.results) == len(original)

        # Results are valid class IDs (0, 1, or 2)
        for cls_id in sink.results:
            assert cls_id in (0, 1, 2), f"Invalid class ID: {cls_id}"

        # No errors
        for m in metrics.values():
            assert m.error is None, f"Node '{m.name}' had error: {m.error}"
