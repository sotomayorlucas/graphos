"""Program composition — adapters and pipelines for chaining ONNX programs."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class AdapterSpec:
    """Metadata for a tensor adapter (shape transform between programs)."""

    name: str
    input_shapes: dict[str, tuple]
    output_shape: tuple
    description: str = ""


class TensorAdapter:
    """Wraps a pure function that transforms tensors between programs."""

    def __init__(self, spec: AdapterSpec, fn: Callable[[dict[str, np.ndarray]], np.ndarray]):
        self._spec = spec
        self._fn = fn

    @property
    def spec(self) -> AdapterSpec:
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.name

    def execute(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Apply the adapter function to named input tensors."""
        return self._fn(inputs)


def concat_adapter(batch_size: int, left_dim: int, right_dim: int) -> TensorAdapter:
    """Create an adapter that concatenates two tensors along axis=1.

    (B, left_dim) + (B, right_dim) -> (B, left_dim + right_dim)
    """
    spec = AdapterSpec(
        name="concat",
        input_shapes={"left": (batch_size, left_dim), "right": (batch_size, right_dim)},
        output_shape=(batch_size, left_dim + right_dim),
        description=f"Concat (B,{left_dim}) + (B,{right_dim}) -> (B,{left_dim + right_dim})",
    )

    def _concat(inputs: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([inputs["left"], inputs["right"]], axis=1)

    return TensorAdapter(spec, _concat)


def pad_adapter(batch_size: int, input_dim: int, output_dim: int) -> TensorAdapter:
    """Create an adapter that zero-pads a tensor along axis=1.

    (B, input_dim) -> (B, output_dim) where output_dim >= input_dim
    """
    spec = AdapterSpec(
        name="pad",
        input_shapes={"in": (batch_size, input_dim)},
        output_shape=(batch_size, output_dim),
        description=f"Pad (B,{input_dim}) -> (B,{output_dim})",
    )

    def _pad(inputs: dict[str, np.ndarray]) -> np.ndarray:
        tensor = inputs["in"]
        pad_width = output_dim - tensor.shape[1]
        if pad_width <= 0:
            return tensor[:, :output_dim]
        return np.pad(tensor, ((0, 0), (0, pad_width)), constant_values=0.0)

    return TensorAdapter(spec, _pad)


class ProgramPipeline:
    """Linear chain of programs and adapters for composed execution.

    Stages are either ('program', name) or ('adapter', name).
    Programs execute on the kernel runtime; adapters transform tensors in between.
    """

    def __init__(self):
        self._stages: list[tuple[str, str, object]] = []  # (kind, name, spec_or_adapter)
        self._raw_passthrough: str | None = None

    def add_program(self, name: str, spec) -> "ProgramPipeline":
        """Add a program stage. The spec is a ProgramSpec."""
        self._stages.append(("program", name, spec))
        return self

    def add_adapter(self, adapter: TensorAdapter) -> "ProgramPipeline":
        """Add an adapter stage between programs."""
        self._stages.append(("adapter", adapter.name, adapter))
        return self

    def with_raw_passthrough(self, input_name: str = "raw") -> "ProgramPipeline":
        """Thread the original input tensor through to adapters.

        When set, the original input to the pipeline is available to adapters
        under the given name (default "raw"). This enables concat adapters
        that need both the raw input and intermediate outputs.
        """
        self._raw_passthrough = input_name
        return self

    @property
    def stages(self) -> list[tuple[str, str]]:
        """Return [(kind, name), ...] for each stage."""
        return [(kind, name) for kind, name, _ in self._stages]

    def validate(self) -> list[str]:
        """Check shape compatibility between consecutive stages.

        Returns a list of error strings (empty = valid).
        """
        errors = []
        if len(self._stages) == 0:
            errors.append("Pipeline has no stages")
            return errors

        prev_output_shape = None
        for i, (kind, name, obj) in enumerate(self._stages):
            if kind == "program":
                if prev_output_shape is not None:
                    expected_input = obj.input_shape
                    if prev_output_shape != expected_input:
                        errors.append(
                            f"Stage {i} '{name}': expected input {expected_input}, "
                            f"got {prev_output_shape} from previous stage"
                        )
                prev_output_shape = obj.output_shape
            elif kind == "adapter":
                adapter = obj
                # For single-input adapters, check shape from previous stage
                if "in" in adapter.spec.input_shapes and prev_output_shape is not None:
                    expected = adapter.spec.input_shapes["in"]
                    if prev_output_shape != expected:
                        errors.append(
                            f"Stage {i} '{name}': expected input {expected}, "
                            f"got {prev_output_shape} from previous stage"
                        )
                prev_output_shape = adapter.spec.output_shape
        return errors

    def execute(self, runtime, input_tensor: np.ndarray) -> np.ndarray:
        """Run the full pipeline: programs on runtime, adapters inline.

        Args:
            runtime: KernelRuntime with programs loaded.
            input_tensor: Input to the first stage.

        Returns:
            Output tensor from the last stage.
        """
        current = input_tensor
        raw = input_tensor if self._raw_passthrough else None

        for kind, name, obj in self._stages:
            if kind == "program":
                current = runtime.execute(name, current)
            elif kind == "adapter":
                adapter = obj
                inputs = {}
                # Build adapter inputs based on its spec
                input_keys = list(adapter.spec.input_shapes.keys())
                if len(input_keys) == 1 and input_keys[0] == "in":
                    inputs["in"] = current
                elif "left" in adapter.spec.input_shapes and "right" in adapter.spec.input_shapes:
                    # Concat adapter: left=raw passthrough, right=current (program output)
                    if raw is not None:
                        inputs["left"] = raw
                        inputs["right"] = current
                    else:
                        raise ValueError(
                            f"Adapter '{name}' needs 'left'+'right' but no raw passthrough set"
                        )
                else:
                    inputs[input_keys[0]] = current
                current = adapter.execute(inputs)

        return current

    def describe(self) -> str:
        """Human-readable pipeline description."""
        if not self._stages:
            return "(empty pipeline)"
        parts = []
        for kind, name, obj in self._stages:
            if kind == "program":
                spec = obj
                parts.append(f"[{name}] {spec.input_shape} -> {spec.output_shape}")
            elif kind == "adapter":
                adapter = obj
                parts.append(f"<{name}> {adapter.spec.description}")
        desc = " -> ".join(parts)
        if self._raw_passthrough:
            desc += f"  (raw passthrough: '{self._raw_passthrough}')"
        return desc
