"""Tests for router.model and router.dataset modules."""

import torch
import numpy as np
import pytest

from router.model import RouterGraph
from router.dataset import (
    PacketDataset,
    generate_http_packet,
    generate_dns_packet,
    generate_other_packet,
)
from core.constants import TENSOR_DIM, NUM_CLASSES, PROTO_TCP, PROTO_UDP, HTTP_PORTS, DNS_PORT


class TestRouterGraph:
    def test_output_shape(self):
        model = RouterGraph()
        x = torch.randn(1, TENSOR_DIM)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_batch_output_shape(self):
        model = RouterGraph()
        x = torch.randn(32, TENSOR_DIM)
        out = model(x)
        assert out.shape == (32, NUM_CLASSES)

    def test_parameter_count(self):
        model = RouterGraph()
        count = sum(p.numel() for p in model.parameters())
        # 64*32 + 32 + 32*32 + 32 + 32*3 + 3 = 3267
        assert count < 5000, "Model should be tiny"

    def test_deterministic_forward(self):
        model = RouterGraph()
        model.eval()
        x = torch.randn(1, TENSOR_DIM)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


class TestPacketGeneration:
    def test_http_packet_structure(self):
        pkt = generate_http_packet()
        assert len(pkt) == 64
        assert pkt[23] == PROTO_TCP
        dst_port = (pkt[36] << 8) | pkt[37]
        assert dst_port in HTTP_PORTS

    def test_dns_packet_structure(self):
        pkt = generate_dns_packet()
        assert len(pkt) == 64
        assert pkt[23] == PROTO_UDP
        dst_port = (pkt[36] << 8) | pkt[37]
        assert dst_port == DNS_PORT

    def test_other_packet_generation(self):
        # Generate several to cover different branches
        for _ in range(20):
            pkt = generate_other_packet()
            assert len(pkt) >= 14  # At least ethernet header


class TestPacketDataset:
    def test_dataset_size(self):
        ds = PacketDataset(samples_per_class=100)
        assert len(ds) == 300

    def test_dataset_shapes(self):
        ds = PacketDataset(samples_per_class=100)
        x, y = ds[0]
        assert x.shape == (TENSOR_DIM,)
        assert y.shape == ()

    def test_class_balance(self):
        ds = PacketDataset(samples_per_class=100)
        labels = [ds[i][1].item() for i in range(len(ds))]
        for cls in range(NUM_CLASSES):
            assert labels.count(cls) == 100
