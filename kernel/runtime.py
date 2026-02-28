"""KernelRuntime — multi-program NPU/CPU execution environment."""

import threading
import time

import numpy as np

from kernel.program import ProgramSpec, Program


class KernelError(Exception):
    """Error raised by the kernel runtime."""


class KernelRuntime:
    """Manages device, loads and executes ONNX programs.

    Thread-safe. Falls back from NPU to CPU if NPU is unavailable.
    """

    def __init__(self, device: str = "NPU"):
        import openvino as ov

        self._ov = ov
        self._core = ov.Core()
        self._lock = threading.RLock()
        self._programs: dict[str, Program] = {}

        available = self._core.available_devices
        if device not in available:
            fallback = "CPU"
            print(f"WARNING: {device} not available (found: {available}). "
                  f"Falling back to {fallback}.")
            device = fallback
        self._device = device

        # Metrics
        self._exec_count = 0
        self._total_exec_us = 0.0
        self._last_exec_us = 0.0
        self._errors = 0

    @property
    def device(self) -> str:
        return self._device

    @property
    def programs(self) -> list[str]:
        """Names of loaded programs."""
        with self._lock:
            return list(self._programs.keys())

    def load(self, spec: ProgramSpec) -> Program:
        """Compile and register a program from its spec."""
        with self._lock:
            if spec.name in self._programs:
                raise KernelError(f"Program '{spec.name}' already loaded")
            model = self._core.read_model(spec.onnx_path)
            compiled = self._core.compile_model(model, self._device)
            program = Program(spec, compiled, self._ov)
            self._programs[spec.name] = program
            return program

    def unload(self, name: str) -> None:
        """Remove a program from the registry."""
        with self._lock:
            if name not in self._programs:
                raise KernelError(f"Program '{name}' not loaded")
            del self._programs[name]

    def get(self, name: str) -> Program:
        """Look up a loaded program by name."""
        with self._lock:
            if name not in self._programs:
                raise KernelError(f"Program '{name}' not loaded")
            return self._programs[name]

    def execute(self, program_name: str, input_tensor: np.ndarray) -> np.ndarray:
        """Execute a loaded program with timing."""
        program = self.get(program_name)
        t0 = time.perf_counter()
        try:
            result = program.execute(input_tensor)
        except Exception:
            self._errors += 1
            raise
        elapsed_us = (time.perf_counter() - t0) * 1e6
        with self._lock:
            self._exec_count += 1
            self._total_exec_us += elapsed_us
            self._last_exec_us = elapsed_us
        return result

    def device_info(self) -> dict:
        """Return info about the active device."""
        return {
            "device": self._device,
            "available_devices": self._core.available_devices,
        }

    def health(self) -> dict:
        """Return runtime health metrics."""
        with self._lock:
            mean_us = (self._total_exec_us / self._exec_count
                       if self._exec_count > 0 else 0.0)
            return {
                "device": self._device,
                "programs": list(self._programs.keys()),
                "exec_count": self._exec_count,
                "mean_latency_us": mean_us,
                "last_latency_us": self._last_exec_us,
                "errors": self._errors,
                "healthy": self._errors == 0,
            }
