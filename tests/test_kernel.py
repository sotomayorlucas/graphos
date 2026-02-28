"""Tests for kernel runtime, program abstraction, and KernelInferNode."""

import os

import numpy as np
import pytest

from core.constants import TENSOR_DIM, NUM_CLASSES, NUM_ROUTES, DEFAULT_BATCH_SIZE
from kernel.program import ProgramSpec, Program
from kernel.runtime import KernelRuntime, KernelError
from kernel.programs.classifier import classifier_spec
from kernel.programs.route_table import route_table_spec


# --- Helpers ---

MODEL_B64 = os.path.join("models", "router_graph_b64.onnx")
ROUTE_B64 = os.path.join("models", "route_table_b64.onnx")
SKIP_CLASSIFIER = not os.path.exists(MODEL_B64)
SKIP_ROUTE = not os.path.exists(ROUTE_B64)


def _make_runtime():
    """Create a CPU-fallback runtime for testing."""
    return KernelRuntime(device="CPU")


# --- TestProgramSpec ---

class TestProgramSpec:
    def test_create(self):
        spec = ProgramSpec(
            name="test",
            onnx_path="models/test.onnx",
            input_shape=(1, 64),
            output_shape=(1, 3),
            description="Test program",
        )
        assert spec.name == "test"
        assert spec.onnx_path == "models/test.onnx"
        assert spec.input_shape == (1, 64)
        assert spec.output_shape == (1, 3)
        assert spec.description == "Test program"

    def test_frozen(self):
        spec = ProgramSpec(name="x", onnx_path="x.onnx",
                           input_shape=(1,), output_shape=(1,))
        with pytest.raises(AttributeError):
            spec.name = "y"


# --- TestKernelRuntime ---

@pytest.mark.skipif(SKIP_CLASSIFIER, reason=f"Model not found: {MODEL_B64}")
class TestKernelRuntime:
    def test_load_and_get(self):
        rt = _make_runtime()
        spec = classifier_spec(DEFAULT_BATCH_SIZE)
        program = rt.load(spec)
        assert program.name == "classifier"
        assert rt.get("classifier") is program
        assert "classifier" in rt.programs

    def test_execute(self):
        rt = _make_runtime()
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))
        tensor = np.random.rand(DEFAULT_BATCH_SIZE, TENSOR_DIM).astype(np.float32)
        output = rt.execute("classifier", tensor)
        assert output.shape == (DEFAULT_BATCH_SIZE, NUM_CLASSES)

    def test_unload(self):
        rt = _make_runtime()
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))
        rt.unload("classifier")
        assert "classifier" not in rt.programs

    def test_device_info(self):
        rt = _make_runtime()
        info = rt.device_info()
        assert "device" in info
        assert "available_devices" in info


# --- TestKernelRuntime_errors ---

@pytest.mark.skipif(SKIP_CLASSIFIER, reason=f"Model not found: {MODEL_B64}")
class TestKernelRuntime_errors:
    def test_duplicate_load(self):
        rt = _make_runtime()
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))
        with pytest.raises(KernelError, match="already loaded"):
            rt.load(classifier_spec(DEFAULT_BATCH_SIZE))

    def test_get_nonexistent(self):
        rt = _make_runtime()
        with pytest.raises(KernelError, match="not loaded"):
            rt.get("nonexistent")

    def test_unload_nonexistent(self):
        rt = _make_runtime()
        with pytest.raises(KernelError, match="not loaded"):
            rt.unload("nonexistent")

    def test_execute_nonexistent(self):
        rt = _make_runtime()
        with pytest.raises(KernelError, match="not loaded"):
            rt.execute("nonexistent", np.zeros((1, 64), dtype=np.float32))


# --- TestKernelRuntime_health ---

@pytest.mark.skipif(SKIP_CLASSIFIER, reason=f"Model not found: {MODEL_B64}")
class TestKernelRuntime_health:
    def test_health_after_execution(self):
        rt = _make_runtime()
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))
        tensor = np.random.rand(DEFAULT_BATCH_SIZE, TENSOR_DIM).astype(np.float32)
        rt.execute("classifier", tensor)
        rt.execute("classifier", tensor)

        health = rt.health()
        assert health["exec_count"] == 2
        assert health["mean_latency_us"] > 0
        assert health["last_latency_us"] > 0
        assert health["errors"] == 0
        assert health["healthy"] is True

    def test_health_initial(self):
        rt = _make_runtime()
        health = rt.health()
        assert health["exec_count"] == 0
        assert health["mean_latency_us"] == 0.0
        assert health["healthy"] is True


# --- TestKernelRuntime_multi_program ---

@pytest.mark.skipif(
    SKIP_CLASSIFIER or SKIP_ROUTE,
    reason="Need both classifier and route_table ONNX models",
)
class TestKernelRuntime_multi_program:
    def test_load_and_execute_both(self):
        rt = _make_runtime()
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))
        rt.load(route_table_spec(DEFAULT_BATCH_SIZE))
        assert len(rt.programs) == 2

        tensor = np.random.rand(DEFAULT_BATCH_SIZE, TENSOR_DIM).astype(np.float32)

        cls_out = rt.execute("classifier", tensor)
        assert cls_out.shape == (DEFAULT_BATCH_SIZE, NUM_CLASSES)

        rt_out = rt.execute("route_table", tensor)
        assert rt_out.shape == (DEFAULT_BATCH_SIZE, NUM_ROUTES)

        assert rt.health()["exec_count"] == 2


# --- TestKernelInferNode ---

@pytest.mark.skipif(SKIP_CLASSIFIER, reason=f"Model not found: {MODEL_B64}")
class TestKernelInferNode:
    def test_dataflow_pipeline(self):
        from dataflow.primitives import _STOP, Channel
        from dataflow.nodes.infer import KernelInferNode

        rt = _make_runtime()
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))

        node = KernelInferNode("kinfer", rt, "classifier")

        # Wire channels
        in_ch = Channel()
        out_ch = Channel()
        node.inputs["in"].channel = in_ch
        node.outputs["out"].channel = out_ch

        # Feed a batch
        tensor = np.random.rand(DEFAULT_BATCH_SIZE, TENSOR_DIM).astype(np.float32)
        in_ch.put((tensor, 10))
        in_ch.put(_STOP)

        node.run()

        results = out_ch.get()
        assert isinstance(results, list)
        assert len(results) == 10
        assert all(r in (0, 1, 2) for r in results)

        assert out_ch.get() is _STOP


# --- TestKernelNPU (only runs when NPU available) ---

@pytest.mark.skipif(SKIP_CLASSIFIER, reason=f"Model not found: {MODEL_B64}")
class TestKernelNPU:
    def _npu_available(self):
        import openvino as ov
        return "NPU" in ov.Core().available_devices

    def test_npu_execution(self):
        if not self._npu_available():
            pytest.skip("NPU not available")
        rt = KernelRuntime(device="NPU")
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))
        tensor = np.random.rand(DEFAULT_BATCH_SIZE, TENSOR_DIM).astype(np.float32)
        output = rt.execute("classifier", tensor)
        assert output.shape == (DEFAULT_BATCH_SIZE, NUM_CLASSES)
        assert rt.device == "NPU"
