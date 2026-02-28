"""Tests for route table model, export, and routing decisions."""

import os
import tempfile

import numpy as np
import torch
import pytest

from core.constants import (
    TENSOR_DIM, NUM_ROUTES, DEFAULT_BATCH_SIZE,
    ROUTE_LOCAL, ROUTE_FORWARD, ROUTE_DROP, ROUTE_MONITOR,
    OFFSET_PROTOCOL, OFFSET_DST_PORT, OFFSET_TTL,
)
from kernel.programs.route_table import RouteTableModel, route_table_spec
from kernel.program import ProgramSpec


# --- TestRouteTableModel ---

class TestRouteTableModel:
    def test_forward_shape(self):
        model = RouteTableModel()
        x = torch.randn(8, TENSOR_DIM)
        out = model(x)
        assert out.shape == (8, NUM_ROUTES)

    def test_single_input(self):
        model = RouteTableModel()
        x = torch.randn(1, TENSOR_DIM)
        out = model(x)
        assert out.shape == (1, NUM_ROUTES)

    def test_batch_size_64(self):
        model = RouteTableModel()
        x = torch.randn(DEFAULT_BATCH_SIZE, TENSOR_DIM)
        out = model(x)
        assert out.shape == (DEFAULT_BATCH_SIZE, NUM_ROUTES)

    def test_deterministic(self):
        model = RouteTableModel()
        model.eval()
        x = torch.randn(4, TENSOR_DIM)
        out1 = model(x).detach().numpy()
        out2 = model(x).detach().numpy()
        np.testing.assert_array_equal(out1, out2)


# --- TestRouteTableExport ---

class TestRouteTableExport:
    def test_export_and_validate(self):
        import onnx
        from kernel.programs.export_route_table import export_route_table

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "route_table_b1.onnx")
            export_route_table(onnx_path=onnx_path, batch_size=1)
            assert os.path.exists(onnx_path)

            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)

    def test_export_batched(self):
        import onnx
        from kernel.programs.export_route_table import export_route_table

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "route_table_b8.onnx")
            export_route_table(onnx_path=onnx_path, batch_size=8)

            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)

            # Check input shape
            inp = model.graph.input[0]
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            assert dims == [8, TENSOR_DIM]


# --- TestRouteTableDecisions ---

class TestRouteTableDecisions:
    def _make_packet(self, **overrides) -> torch.Tensor:
        """Create a normalized packet tensor with specific byte overrides."""
        pkt = torch.zeros(1, TENSOR_DIM)
        for offset, value in overrides.items():
            pkt[0, int(offset)] = value / 255.0
        return pkt

    def test_dns_routes_local(self):
        """UDP + DNS port should score high for LOCAL."""
        model = RouteTableModel()
        model.eval()
        # UDP=17, DNS dst port low byte=53 (0x35)
        pkt = self._make_packet(**{
            str(OFFSET_PROTOCOL): 17,
            str(OFFSET_DST_PORT + 1): 53,
        })
        with torch.no_grad():
            scores = model(pkt)
        route = torch.argmax(scores, dim=1).item()
        assert route == ROUTE_LOCAL

    def test_high_ttl_not_dropped(self):
        """Packet with high TTL should not be routed to DROP."""
        model = RouteTableModel()
        model.eval()
        pkt = self._make_packet(**{
            str(OFFSET_TTL): 128,
            str(OFFSET_PROTOCOL): 6,
            str(OFFSET_DST_PORT): 0,
            str(OFFSET_DST_PORT + 1): 80,
        })
        with torch.no_grad():
            scores = model(pkt)
        route = torch.argmax(scores, dim=1).item()
        assert route != ROUTE_DROP


# --- TestRouteTableSpec ---

class TestRouteTableSpec:
    def test_spec_fields(self):
        spec = route_table_spec(batch_size=64)
        assert spec.name == "route_table"
        assert spec.input_shape == (64, TENSOR_DIM)
        assert spec.output_shape == (64, NUM_ROUTES)
        assert "route_table_b64.onnx" in spec.onnx_path
        assert isinstance(spec, ProgramSpec)

    def test_spec_default_batch(self):
        spec = route_table_spec()
        assert spec.input_shape[0] == DEFAULT_BATCH_SIZE
