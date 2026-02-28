"""Tests for KernelLoop — persistent processing daemon."""

import os
import threading

import numpy as np
import pytest

from core.constants import TENSOR_DIM, DEFAULT_BATCH_SIZE, NUM_CLASSES, NUM_ROUTES
from kernel.runtime import KernelRuntime
from kernel.loop import KernelLoop
from kernel.programs.classifier import classifier_spec
from kernel.programs.route_table import route_table_spec


MODEL_B64 = os.path.join("models", "router_graph_b64.onnx")
ROUTE_B64 = os.path.join("models", "route_table_b64.onnx")
SKIP_CLASSIFIER = not os.path.exists(MODEL_B64)
SKIP_ROUTE = not os.path.exists(ROUTE_B64)


def _random_packets(n: int) -> list[bytes]:
    rng = np.random.default_rng(123)
    return [bytes(rng.integers(0, 256, size=TENSOR_DIM, dtype=np.uint8)) for _ in range(n)]


@pytest.mark.skipif(SKIP_CLASSIFIER, reason=f"Model not found: {MODEL_B64}")
class TestKernelLoop:
    def test_run_with_source(self):
        rt = KernelRuntime(device="CPU")
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))

        loop = KernelLoop(rt, batch_size=DEFAULT_BATCH_SIZE, health_interval=60.0)
        packets = _random_packets(200)

        loop.run(iter(packets))

        stats = loop.stats
        assert stats["packets_processed"] == 200
        assert stats["batches_processed"] >= 1
        assert stats["elapsed"] > 0
        assert stats["throughput"] > 0

    def test_process_batch(self):
        rt = KernelRuntime(device="CPU")
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))

        loop = KernelLoop(rt, batch_size=DEFAULT_BATCH_SIZE)
        packets = _random_packets(10)
        results = loop.process_batch(packets)

        assert "classifier" in results
        assert results["classifier"].shape == (DEFAULT_BATCH_SIZE, NUM_CLASSES)


@pytest.mark.skipif(SKIP_CLASSIFIER, reason=f"Model not found: {MODEL_B64}")
class TestKernelLoop_stop:
    def test_clean_shutdown(self):
        rt = KernelRuntime(device="CPU")
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))
        loop = KernelLoop(rt, batch_size=DEFAULT_BATCH_SIZE, health_interval=60.0)

        # Generate a large source
        packets = _random_packets(10000)

        def _run():
            loop.run(iter(packets))

        t = threading.Thread(target=_run)
        t.start()
        # Request stop after a short while
        loop.stop()
        t.join(timeout=5.0)
        assert not t.is_alive(), "KernelLoop did not stop cleanly"


@pytest.mark.skipif(
    SKIP_CLASSIFIER or SKIP_ROUTE,
    reason="Need both classifier and route_table ONNX models",
)
class TestKernelLoop_multi_program:
    def test_both_programs_executed(self):
        rt = KernelRuntime(device="CPU")
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))
        rt.load(route_table_spec(DEFAULT_BATCH_SIZE))

        loop = KernelLoop(rt, batch_size=DEFAULT_BATCH_SIZE, health_interval=60.0)
        packets = _random_packets(128)

        loop.run(iter(packets))

        stats = loop.stats
        assert stats["packets_processed"] == 128
        # Both programs should have been executed (exec_count >= 2 per batch)
        health = rt.health()
        assert health["exec_count"] >= 4  # at least 2 batches x 2 programs

    def test_process_batch_both(self):
        rt = KernelRuntime(device="CPU")
        rt.load(classifier_spec(DEFAULT_BATCH_SIZE))
        rt.load(route_table_spec(DEFAULT_BATCH_SIZE))

        loop = KernelLoop(rt, batch_size=DEFAULT_BATCH_SIZE)
        packets = _random_packets(10)
        results = loop.process_batch(packets)

        assert "classifier" in results
        assert "route_table" in results
        assert results["classifier"].shape == (DEFAULT_BATCH_SIZE, NUM_CLASSES)
        assert results["route_table"].shape == (DEFAULT_BATCH_SIZE, NUM_ROUTES)
