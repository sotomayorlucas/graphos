"""Tests for runtime modules."""

import pytest

from runtime.cpu_baseline import classify_packet_switch
from router.dataset import generate_http_packet, generate_dns_packet, generate_other_packet
from core.constants import CLASS_HTTP, CLASS_DNS, CLASS_OTHER, PROTO_TCP, PROTO_UDP, HTTP_PORTS
from runtime.benchmark import generate_test_packets, percentile


class TestCPUBaseline:
    def test_http_classification(self):
        for _ in range(50):
            pkt = generate_http_packet()
            assert classify_packet_switch(pkt) == CLASS_HTTP

    def test_dns_classification(self):
        for _ in range(50):
            pkt = generate_dns_packet()
            assert classify_packet_switch(pkt) == CLASS_DNS

    def test_short_packet_is_other(self):
        assert classify_packet_switch(b'\x00' * 10) == CLASS_OTHER

    def test_other_packet(self):
        # TCP with non-HTTP port
        import struct
        pkt = bytearray(64)
        pkt[23] = PROTO_TCP
        struct.pack_into('!H', pkt, 36, 12345)
        assert classify_packet_switch(bytes(pkt)) == CLASS_OTHER

    def test_udp_non_dns(self):
        import struct
        pkt = bytearray(64)
        pkt[23] = PROTO_UDP
        struct.pack_into('!H', pkt, 36, 8888)
        assert classify_packet_switch(bytes(pkt)) == CLASS_OTHER


class TestBenchmarkUtils:
    def test_generate_test_packets(self):
        packets = generate_test_packets(n=30)
        assert len(packets) == 30
        assert all(isinstance(p, bytes) for p in packets)

    def test_percentile(self):
        values = list(range(100))
        assert percentile(values, 50) == pytest.approx(49.5)
        assert percentile(values, 0) == 0
        assert percentile(values, 100) == 99
