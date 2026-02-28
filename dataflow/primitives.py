"""Dataflow primitives: sentinel, channels, and ports."""

import queue

# Sentinel singleton — identity-compared to signal shutdown.
_STOP = object()


class Channel:
    """Bounded FIFO channel with backpressure, wrapping queue.Queue."""

    def __init__(self, capacity: int = 64):
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._capacity = capacity
        self._items_passed = 0

    def put(self, item, timeout=None):
        self._queue.put(item, timeout=timeout)
        self._items_passed += 1

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)

    @property
    def items_passed(self) -> int:
        return self._items_passed

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def full(self) -> bool:
        return self._queue.full()


class InputPort:
    """Named input port that reads from an attached Channel."""

    def __init__(self, name: str):
        self.name = name
        self.channel: Channel | None = None

    def get(self, timeout=None):
        if self.channel is None:
            raise RuntimeError(f"InputPort '{self.name}' is not connected")
        return self.channel.get(timeout=timeout)

    @property
    def connected(self) -> bool:
        return self.channel is not None


class OutputPort:
    """Named output port that writes to an attached Channel."""

    def __init__(self, name: str):
        self.name = name
        self.channel: Channel | None = None

    def put(self, item, timeout=None):
        if self.channel is None:
            raise RuntimeError(f"OutputPort '{self.name}' is not connected")
        self.channel.put(item, timeout=timeout)

    @property
    def connected(self) -> bool:
        return self.channel is not None
