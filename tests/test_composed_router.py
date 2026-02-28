"""Tests for composed router model, export, and routing decisions."""

import os
import tempfile

import numpy as np
import torch
import pytest

from core.constants import (
    TENSOR_DIM, NUM_CLASSES, NUM_ROUTES, DEFAULT_BATCH_SIZE,
    ROUTE_LOCAL, ROUTE_FORWARD, ROUTE_DROP, ROUTE_MONITOR,
    OFFSET_PROTOCOL, OFFSET_DST_PORT, OFFSET_TTL,
)
from kernel.programs.composed_router import (
    ComposedRouterModel, composed_router_spec, COMPOSED_INPUT_DIM,
)
from kernel.program import ProgramSpec


# --- TestComposedRouterModel ---

class TestComposedRouterModel:
    def test_forward_shape(self):
        model = ComposedRouterModel()
        x = torch.randn(8, COMPOSED_INPUT_DIM)
        out = model(x)
        assert out.shape == (8, NUM_ROUTES)

    def test_deterministic(self):
        model = ComposedRouterModel()
        model.eval()
        x = torch.randn(4, COMPOSED_INPUT_DIM)
        out1 = model(x).detach().numpy()
        out2 = model(x).detach().numpy()
        np.testing.assert_array_equal(out1, out2)

    def test_composed_input_dim(self):
        assert COMPOSED_INPUT_DIM == TENSOR_DIM + NUM_CLASSES
        assert COMPOSED_INPUT_DIM == 67

    def test_classifier_logits_influence_routing(self):
        """Key proof: classifier logits actually change routing decisions."""
        model = ComposedRouterModel()
        model.eval()

        # Create a neutral packet (all zeros for packet bytes)
        packet_bytes = torch.zeros(1, TENSOR_DIM)

        # Case 1: Strong HTTP logit -> should favor FORWARD
        http_logits = torch.tensor([[5.0, -1.0, -1.0]])  # strong HTTP
        input_http = torch.cat([packet_bytes, http_logits], dim=1)

        # Case 2: Strong DNS logit -> should favor LOCAL
        dns_logits = torch.tensor([[-1.0, 5.0, -1.0]])  # strong DNS
        input_dns = torch.cat([packet_bytes, dns_logits], dim=1)

        with torch.no_grad():
            scores_http = model(input_http)
            scores_dns = model(input_dns)

        route_http = torch.argmax(scores_http, dim=1).item()
        route_dns = torch.argmax(scores_dns, dim=1).item()

        # HTTP logit should boost FORWARD, DNS logit should boost LOCAL
        assert route_http == ROUTE_FORWARD, f"Expected FORWARD for HTTP logits, got {route_http}"
        assert route_dns == ROUTE_LOCAL, f"Expected LOCAL for DNS logits, got {route_dns}"

        # They must be different — proving logits influence the decision
        assert route_http != route_dns

    def test_uncertain_logits_favor_monitor(self):
        """When all logits are negative (uncertain), MONITOR should be favored."""
        model = ComposedRouterModel()
        model.eval()

        # Neutral packet + very negative logits (classifier unsure)
        packet_bytes = torch.zeros(1, TENSOR_DIM)
        uncertain_logits = torch.tensor([[-3.0, -3.0, -3.0]])
        x = torch.cat([packet_bytes, uncertain_logits], dim=1)

        with torch.no_grad():
            scores = model(x)

        route = torch.argmax(scores, dim=1).item()
        # MONITOR penalizes logits, so negative logits become positive boost
        assert route == ROUTE_MONITOR


# --- TestComposedRouterSpec ---

class TestComposedRouterSpec:
    def test_spec_fields(self):
        spec = composed_router_spec(batch_size=64)
        assert spec.name == "composed_router"
        assert spec.input_shape == (64, COMPOSED_INPUT_DIM)
        assert spec.output_shape == (64, NUM_ROUTES)
        assert "composed_router_b64.onnx" in spec.onnx_path
        assert isinstance(spec, ProgramSpec)

    def test_spec_default_batch(self):
        spec = composed_router_spec()
        assert spec.input_shape[0] == DEFAULT_BATCH_SIZE


# --- TestComposedRouterExport ---

class TestComposedRouterExport:
    def test_export_and_validate(self):
        import onnx
        from kernel.programs.export_composed import export_composed_router

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "composed_router_b1.onnx")
            export_composed_router(onnx_path=onnx_path, batch_size=1)
            assert os.path.exists(onnx_path)

            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)

    def test_export_batched(self):
        import onnx
        from kernel.programs.export_composed import export_composed_router

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "composed_router_b8.onnx")
            export_composed_router(onnx_path=onnx_path, batch_size=8)

            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)

            inp = model.graph.input[0]
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            assert dims == [8, COMPOSED_INPUT_DIM]
