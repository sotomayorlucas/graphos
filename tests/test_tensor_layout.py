"""Tests for core.tensor_layout module."""

import numpy as np
import pytest

from core.tensor_layout import packet_to_tensor, tensor_to_class
from core.constants import TENSOR_DIM, NUM_CLASSES


class TestPacketToTensor:
    def test_exact_size_packet(self):
        raw = bytes(range(64))
        tensor = packet_to_tensor(raw)
        assert tensor.shape == (1, TENSOR_DIM)
        assert tensor.dtype == np.float32
        assert tensor[0, 0] == pytest.approx(0.0 / 255.0)
        assert tensor[0, 63] == pytest.approx(63.0 / 255.0)

    def test_short_packet_padded(self):
        raw = bytes([0xFF] * 10)
        tensor = packet_to_tensor(raw)
        assert tensor.shape == (1, TENSOR_DIM)
        # First 10 bytes should be 1.0
        assert all(tensor[0, i] == pytest.approx(1.0) for i in range(10))
        # Remaining should be 0.0 (zero-padded)
        assert all(tensor[0, i] == pytest.approx(0.0) for i in range(10, TENSOR_DIM))

    def test_long_packet_truncated(self):
        raw = bytes([0x80] * 128)
        tensor = packet_to_tensor(raw)
        assert tensor.shape == (1, TENSOR_DIM)
        assert tensor[0, 0] == pytest.approx(0x80 / 255.0)

    def test_empty_packet(self):
        tensor = packet_to_tensor(b'')
        assert tensor.shape == (1, TENSOR_DIM)
        assert np.all(tensor == 0.0)

    def test_normalization_range(self):
        raw = bytes(range(256))[:64]
        tensor = packet_to_tensor(raw)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0


class TestTensorToClass:
    def test_class_0(self):
        output = np.array([[10.0, 1.0, 2.0]])
        assert tensor_to_class(output) == 0

    def test_class_1(self):
        output = np.array([[1.0, 10.0, 2.0]])
        assert tensor_to_class(output) == 1

    def test_class_2(self):
        output = np.array([[1.0, 2.0, 10.0]])
        assert tensor_to_class(output) == 2

    def test_flat_array(self):
        output = np.array([1.0, 2.0, 10.0])
        assert tensor_to_class(output) == 2
