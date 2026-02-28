"""SourceNode — pushes items from an iterable to its output port."""

from dataflow.primitives import _STOP
from dataflow.node import Node


class SourceNode(Node):
    """Emits each item from an iterable, then sends _STOP."""

    def __init__(self, name: str, iterable):
        super().__init__(name)
        self._iterable = iterable
        self.add_output("out")

    def process(self):
        out = self.outputs["out"]
        for item in self._iterable:
            out.put(item)
            self.items_processed += 1
        out.put(_STOP)
