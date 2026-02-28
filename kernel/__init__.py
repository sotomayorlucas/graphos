"""GraphOS Kernel — NPU runtime for ONNX programs."""

from kernel.program import ProgramSpec, Program
from kernel.runtime import KernelRuntime, KernelError
from kernel.health import HealthMonitor
from kernel.loop import KernelLoop
from kernel.compose import (
    AdapterSpec, TensorAdapter, ProgramPipeline,
    concat_adapter, pad_adapter,
)

__all__ = [
    "ProgramSpec", "Program",
    "KernelRuntime", "KernelError",
    "HealthMonitor", "KernelLoop",
    "AdapterSpec", "TensorAdapter", "ProgramPipeline",
    "concat_adapter", "pad_adapter",
]
