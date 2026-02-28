"""Program abstraction — ONNX compiled model as a kernel program."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProgramSpec:
    """Metadata for a kernel program (ONNX model)."""

    name: str
    onnx_path: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    description: str = ""


class Program:
    """Compiled ONNX program ready for execution on a device."""

    def __init__(self, spec: ProgramSpec, compiled_model, ov_module):
        self._spec = spec
        self._compiled = compiled_model
        self._infer_req = compiled_model.create_infer_request()
        self._ov = ov_module

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def spec(self) -> ProgramSpec:
        return self._spec

    def execute(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run synchronous inference. Returns raw output (no interpretation)."""
        self._infer_req.set_input_tensor(self._ov.Tensor(input_tensor))
        self._infer_req.infer()
        return self._infer_req.get_output_tensor().data.copy()
