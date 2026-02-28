"""HealthMonitor — periodic health checks for the kernel runtime."""

import threading

import numpy as np


class HealthMonitor:
    """Daemon thread that periodically polls runtime health."""

    def __init__(self, runtime, interval: float = 5.0):
        self._runtime = runtime
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_health: dict = {}

    @property
    def last_health(self) -> dict:
        return self._last_health

    def start(self):
        """Start the health monitoring daemon."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the health monitor."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1)
            self._thread = None

    def _loop(self):
        while not self._stop_event.is_set():
            self._last_health = self._runtime.health()
            self._stop_event.wait(self._interval)

    def check_latency(self, program_name: str, threshold_us: float = 5000.0) -> bool:
        """Run a dummy inference and check if latency is under threshold."""
        program = self._runtime.get(program_name)
        dummy = np.zeros(program.spec.input_shape, dtype=np.float32)
        self._runtime.execute(program_name, dummy)
        health = self._runtime.health()
        return health["last_latency_us"] < threshold_us
