"""Tests for batch inference: tensor functions and PacketBatcher."""

import os

import numpy as np
import pytest

from core.constants import TENSOR_DIM, NUM_CLASSES, DEFAULT_BATCH_SIZE
from core.tensor_layout import packets_to_batch_tensor, batch_tensor_to_classes


class TestPacketsToBatchTensor:
    def test_full_batch(self):
        packets = [bytes([i] * TENSOR_DIM) for i in range(4)]
        tensor = packets_to_batch_tensor(packets, batch_size=4)
        assert tensor.shape == (4, TENSOR_DIM)
        assert tensor.dtype == np.float32

    def test_partial_batch_padded(self):
        packets = [bytes([0xFF] * TENSOR_DIM), bytes([0x80] * TENSOR_DIM)]
        tensor = packets_to_batch_tensor(packets, batch_size=4)
        assert tensor.shape == (4, TENSOR_DIM)
        # First two rows should have data
        assert tensor[0, 0] == pytest.approx(1.0)
        assert tensor[1, 0] == pytest.approx(0x80 / 255.0)
        # Last two rows should be zero-padded
        assert np.all(tensor[2] == 0.0)
        assert np.all(tensor[3] == 0.0)

    def test_overflow_raises(self):
        packets = [b'\x00' * TENSOR_DIM] * 5
        with pytest.raises(ValueError, match="batch_size is 4"):
            packets_to_batch_tensor(packets, batch_size=4)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            packets_to_batch_tensor([], batch_size=4)

    def test_normalization(self):
        packets = [bytes(range(TENSOR_DIM))]
        tensor = packets_to_batch_tensor(packets, batch_size=1)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        assert tensor[0, 0] == pytest.approx(0.0)
        assert tensor[0, 63] == pytest.approx(63.0 / 255.0)

    def test_short_packets_padded(self):
        packets = [bytes([0xFF] * 10)]
        tensor = packets_to_batch_tensor(packets, batch_size=1)
        assert tensor.shape == (1, TENSOR_DIM)
        assert all(tensor[0, i] == pytest.approx(1.0) for i in range(10))
        assert all(tensor[0, i] == pytest.approx(0.0) for i in range(10, TENSOR_DIM))

    def test_default_batch_size(self):
        packets = [b'\x00' * TENSOR_DIM]
        tensor = packets_to_batch_tensor(packets)
        assert tensor.shape == (DEFAULT_BATCH_SIZE, TENSOR_DIM)


class TestBatchTensorToClasses:
    def test_basic(self):
        output = np.array([
            [10.0, 1.0, 2.0],
            [1.0, 10.0, 2.0],
            [1.0, 2.0, 10.0],
        ])
        classes = batch_tensor_to_classes(output)
        assert classes == [0, 1, 2]

    def test_single_row(self):
        output = np.array([[1.0, 5.0, 3.0]])
        assert batch_tensor_to_classes(output) == [1]

    def test_returns_list_of_int(self):
        output = np.array([[1.0, 2.0, 3.0]])
        result = batch_tensor_to_classes(output)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)


BATCHED_MODEL = f"models/router_graph_b{DEFAULT_BATCH_SIZE}.onnx"


@pytest.mark.skipif(
    not os.path.exists(BATCHED_MODEL),
    reason=f"Batched model not exported: {BATCHED_MODEL}",
)
class TestPacketBatcher:
    def test_fills_and_returns(self):
        from runtime.npu_engine import NPUEngine
        from runtime.packet_batcher import PacketBatcher

        engine = NPUEngine(BATCHED_MODEL)
        batcher = PacketBatcher(engine, batch_size=DEFAULT_BATCH_SIZE)

        result = None
        for i in range(DEFAULT_BATCH_SIZE):
            pkt = bytes([i % 256] * TENSOR_DIM)
            result = batcher.add(pkt)
            if i < DEFAULT_BATCH_SIZE - 1:
                assert result is None
                assert batcher.pending == i + 1

        assert result is not None
        assert len(result) == DEFAULT_BATCH_SIZE
        assert batcher.pending == 0

    def test_flush_partial(self):
        from runtime.npu_engine import NPUEngine
        from runtime.packet_batcher import PacketBatcher

        engine = NPUEngine(BATCHED_MODEL)
        batcher = PacketBatcher(engine, batch_size=DEFAULT_BATCH_SIZE)

        for i in range(10):
            batcher.add(bytes([i] * TENSOR_DIM))

        assert batcher.pending == 10
        flushed = batcher.flush()
        assert flushed is not None
        results, actual = flushed
        assert actual == 10
        assert len(results) == DEFAULT_BATCH_SIZE  # padded
        assert batcher.pending == 0

    def test_flush_empty(self):
        from runtime.npu_engine import NPUEngine
        from runtime.packet_batcher import PacketBatcher

        engine = NPUEngine(BATCHED_MODEL)
        batcher = PacketBatcher(engine, batch_size=DEFAULT_BATCH_SIZE)
        assert batcher.flush() is None
