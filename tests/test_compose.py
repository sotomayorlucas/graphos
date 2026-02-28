"""Tests for adapter and pipeline composition framework."""

import numpy as np
import pytest

from kernel.compose import (
    AdapterSpec, TensorAdapter, ProgramPipeline,
    concat_adapter, pad_adapter,
)
from kernel.program import ProgramSpec
from dataflow.primitives import _STOP, Channel
from dataflow.nodes.adapter import AdapterNode


# --- TestAdapterSpec ---

class TestAdapterSpec:
    def test_create(self):
        spec = AdapterSpec(
            name="test",
            input_shapes={"in": (4, 64)},
            output_shape=(4, 67),
            description="Test adapter",
        )
        assert spec.name == "test"
        assert spec.input_shapes == {"in": (4, 64)}
        assert spec.output_shape == (4, 67)
        assert spec.description == "Test adapter"

    def test_frozen(self):
        spec = AdapterSpec(name="x", input_shapes={}, output_shape=(1,))
        with pytest.raises(AttributeError):
            spec.name = "y"


# --- TestConcatAdapter ---

class TestConcatAdapter:
    def test_shape(self):
        adapter = concat_adapter(batch_size=4, left_dim=64, right_dim=3)
        assert adapter.spec.output_shape == (4, 67)
        assert adapter.spec.input_shapes["left"] == (4, 64)
        assert adapter.spec.input_shapes["right"] == (4, 3)

    def test_values(self):
        adapter = concat_adapter(batch_size=2, left_dim=3, right_dim=2)
        left = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        right = np.array([[7, 8], [9, 10]], dtype=np.float32)
        result = adapter.execute({"left": left, "right": right})
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 7, 8])
        np.testing.assert_array_equal(result[1], [4, 5, 6, 9, 10])

    def test_name(self):
        adapter = concat_adapter(batch_size=1, left_dim=10, right_dim=5)
        assert adapter.name == "concat"


# --- TestPadAdapter ---

class TestPadAdapter:
    def test_padding(self):
        adapter = pad_adapter(batch_size=2, input_dim=3, output_dim=5)
        tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = adapter.execute({"in": tensor})
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 0, 0])
        np.testing.assert_array_equal(result[1], [4, 5, 6, 0, 0])

    def test_spec(self):
        adapter = pad_adapter(batch_size=4, input_dim=64, output_dim=128)
        assert adapter.spec.output_shape == (4, 128)
        assert adapter.spec.input_shapes["in"] == (4, 64)
        assert adapter.name == "pad"


# --- TestProgramPipeline ---

class TestProgramPipeline:
    def test_stages(self):
        spec_a = ProgramSpec(name="a", onnx_path="a.onnx",
                             input_shape=(4, 64), output_shape=(4, 3))
        spec_b = ProgramSpec(name="b", onnx_path="b.onnx",
                             input_shape=(4, 67), output_shape=(4, 4))
        adapter = concat_adapter(4, 64, 3)

        pipe = ProgramPipeline()
        pipe.add_program("a", spec_a)
        pipe.add_adapter(adapter)
        pipe.add_program("b", spec_b)

        stages = pipe.stages
        assert len(stages) == 3
        assert stages[0] == ("program", "a")
        assert stages[1] == ("adapter", "concat")
        assert stages[2] == ("program", "b")

    def test_describe(self):
        spec_a = ProgramSpec(name="a", onnx_path="a.onnx",
                             input_shape=(4, 64), output_shape=(4, 3))
        pipe = ProgramPipeline()
        pipe.add_program("a", spec_a)
        desc = pipe.describe()
        assert "a" in desc
        assert "(4, 64)" in desc

    def test_describe_empty(self):
        pipe = ProgramPipeline()
        assert "empty" in pipe.describe()

    def test_validate_valid(self):
        spec_a = ProgramSpec(name="a", onnx_path="a.onnx",
                             input_shape=(4, 64), output_shape=(4, 3))
        spec_b = ProgramSpec(name="b", onnx_path="b.onnx",
                             input_shape=(4, 67), output_shape=(4, 4))
        adapter = concat_adapter(4, 64, 3)

        pipe = ProgramPipeline()
        pipe.add_program("a", spec_a)
        pipe.add_adapter(adapter)
        pipe.add_program("b", spec_b)
        pipe.with_raw_passthrough("raw")

        errors = pipe.validate()
        assert errors == []

    def test_validate_shape_mismatch(self):
        spec_a = ProgramSpec(name="a", onnx_path="a.onnx",
                             input_shape=(4, 64), output_shape=(4, 3))
        # Expects (4, 100) but concat produces (4, 67)
        spec_b = ProgramSpec(name="b", onnx_path="b.onnx",
                             input_shape=(4, 100), output_shape=(4, 4))
        adapter = concat_adapter(4, 64, 3)

        pipe = ProgramPipeline()
        pipe.add_program("a", spec_a)
        pipe.add_adapter(adapter)
        pipe.add_program("b", spec_b)

        errors = pipe.validate()
        assert len(errors) > 0
        assert "100" in errors[0] or "67" in errors[0]

    def test_validate_empty(self):
        pipe = ProgramPipeline()
        errors = pipe.validate()
        assert len(errors) > 0


# --- TestAdapterNode ---

class TestAdapterNode:
    def test_single_input(self):
        adapter = pad_adapter(batch_size=2, input_dim=3, output_dim=5)
        node = AdapterNode("pad_node", adapter, needs_raw=False)

        in_ch = Channel()
        out_ch = Channel()
        node.inputs["in"].channel = in_ch
        node.outputs["out"].channel = out_ch

        tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        in_ch.put((tensor, 2))
        in_ch.put(_STOP)

        node.run()

        result, count = out_ch.get()
        assert count == 2
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 0, 0])
        assert out_ch.get() is _STOP

    def test_dual_input(self):
        adapter = concat_adapter(batch_size=2, left_dim=3, right_dim=2)
        node = AdapterNode("concat_node", adapter, needs_raw=True)

        in_ch = Channel()
        raw_ch = Channel()
        out_ch = Channel()
        node.inputs["in"].channel = in_ch
        node.inputs["raw"].channel = raw_ch
        node.outputs["out"].channel = out_ch

        right = np.array([[7, 8], [9, 10]], dtype=np.float32)
        left = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        in_ch.put((right, 2))
        raw_ch.put((left, 2))
        in_ch.put(_STOP)
        raw_ch.put(_STOP)

        node.run()

        result, count = out_ch.get()
        assert count == 2
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 7, 8])
        assert out_ch.get() is _STOP
