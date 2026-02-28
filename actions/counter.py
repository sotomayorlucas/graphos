"""CountAction — counts packets per class."""

import threading

from actions.base import Action


class CountAction(Action):
    """Thread-safe per-class packet counter."""

    def __init__(self):
        self.counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def execute(self, packet: bytes, class_id: int) -> None:
        with self._lock:
            self.counts[class_id] = self.counts.get(class_id, 0) + 1

    def summary(self) -> dict[int, int]:
        """Return a copy of the current counts."""
        with self._lock:
            return dict(self.counts)
