"""Packet batching layer that accumulates packets and auto-dispatches."""

from core.constants import DEFAULT_BATCH_SIZE
from core.tensor_layout import packets_to_batch_tensor


class PacketBatcher:
    """Accumulates raw packets and dispatches to NPUEngine when a batch fills.

    Args:
        engine: An NPUEngine instance with classify_batch().
        batch_size: Number of packets per batch.
    """

    def __init__(self, engine, batch_size: int = DEFAULT_BATCH_SIZE):
        self.engine = engine
        self.batch_size = batch_size
        self._buffer: list[bytes] = []

    @property
    def pending(self) -> int:
        """Number of packets waiting in the buffer."""
        return len(self._buffer)

    def add(self, raw_bytes: bytes) -> list[int] | None:
        """Add a packet to the buffer. Returns results when batch fills.

        Args:
            raw_bytes: Raw packet bytes.

        Returns:
            List of class indices for the full batch, or None if batch not yet full.
        """
        self._buffer.append(raw_bytes)
        if len(self._buffer) >= self.batch_size:
            return self._dispatch(self._buffer[:self.batch_size])
        return None

    def flush(self) -> tuple[list[int], int] | None:
        """Dispatch any remaining packets as a padded partial batch.

        Returns:
            Tuple of (class indices for full padded batch, actual packet count),
            or None if buffer is empty.
        """
        if not self._buffer:
            return None
        actual = len(self._buffer)
        results = self._dispatch(self._buffer)
        return results, actual

    def _dispatch(self, packets: list[bytes]) -> list[int]:
        """Build batch tensor and run inference."""
        tensor = packets_to_batch_tensor(packets, self.batch_size)
        results = self.engine.classify_batch(tensor)
        self._buffer = self._buffer[len(packets):]
        return results
