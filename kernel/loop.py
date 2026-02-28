"""KernelLoop — persistent processing daemon for the NPU kernel."""

import signal
import time

import numpy as np

from core.constants import DEFAULT_BATCH_SIZE
from core.tensor_layout import packets_to_batch_tensor
from kernel.health import HealthMonitor


class KernelLoop:
    """Main kernel loop: reads packets, batches, executes all loaded programs."""

    def __init__(
        self,
        runtime,
        batch_size: int = DEFAULT_BATCH_SIZE,
        health_interval: float = 5.0,
    ):
        self._runtime = runtime
        self._batch_size = batch_size
        self._health_monitor = HealthMonitor(runtime, interval=health_interval)
        self._running = False
        self._packets_processed = 0
        self._batches_processed = 0
        self._start_time = 0.0

    @property
    def stats(self) -> dict:
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0.0
        throughput = self._packets_processed / elapsed if elapsed > 0 else 0.0
        return {
            "packets_processed": self._packets_processed,
            "batches_processed": self._batches_processed,
            "elapsed": elapsed,
            "throughput": throughput,
            "health": self._health_monitor.last_health,
        }

    def process_batch(self, raw_packets: list[bytes]) -> dict[str, np.ndarray]:
        """Execute all loaded programs on a batch of raw packets."""
        tensor = packets_to_batch_tensor(raw_packets, self._batch_size)
        results = {}
        for name in self._runtime.programs:
            results[name] = self._runtime.execute(name, tensor)
        return results

    def run(self, packet_source, programs: list[str] | None = None):
        """Main loop: iterate packet_source, batch, execute programs.

        Args:
            packet_source: Iterable yielding raw packet bytes.
            programs: Optional list of program names to execute.
                      If None, executes all loaded programs.
        """
        self._running = True
        self._start_time = time.perf_counter()
        self._health_monitor.start()

        # Install SIGINT handler for clean shutdown (main thread only)
        original_handler = None
        try:
            original_handler = signal.getsignal(signal.SIGINT)

            def _stop_handler(signum, frame):
                self._running = False

            signal.signal(signal.SIGINT, _stop_handler)
        except ValueError:
            pass  # Not in main thread — skip signal handling

        try:
            batch: list[bytes] = []
            for pkt in packet_source:
                if not self._running:
                    break
                batch.append(pkt)
                if len(batch) >= self._batch_size:
                    self._execute_batch(batch, programs)
                    batch = []

            # Flush remaining
            if batch and self._running:
                self._execute_batch(batch, programs)
        finally:
            self._running = False
            self._health_monitor.stop()
            if original_handler is not None:
                try:
                    signal.signal(signal.SIGINT, original_handler)
                except ValueError:
                    pass

    def stop(self):
        """Request clean shutdown."""
        self._running = False

    def _execute_batch(self, raw_packets: list[bytes], programs: list[str] | None):
        tensor = packets_to_batch_tensor(raw_packets, self._batch_size)
        names = programs if programs is not None else self._runtime.programs
        for name in names:
            self._runtime.execute(name, tensor)
        self._packets_processed += len(raw_packets)
        self._batches_processed += 1
