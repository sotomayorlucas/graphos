"""SinkNode — collects results into a thread-safe list."""

import threading

from dataflow.primitives import _STOP
from dataflow.node import Node


class SinkNode(Node):
    """Terminal node that accumulates all received items."""

    def __init__(self, name: str):
        super().__init__(name)
        self.add_input("in")
        self.results: list = []
        self._lock = threading.Lock()

    def process(self):
        inp = self.inputs["in"]

        while True:
            item = inp.get()
            if item is _STOP:
                return
            with self._lock:
                if isinstance(item, list):
                    self.results.extend(item)
                    self.items_processed += len(item)
                else:
                    self.results.append(item)
                    self.items_processed += 1
