"""TeeNode — fan-out node that duplicates items to two outputs."""

from dataflow.node import Node
from dataflow.primitives import _STOP


class TeeNode(Node):
    """Duplicates every item from 'in' to both 'out' and 'copy' outputs."""

    def __init__(self, name: str):
        super().__init__(name)
        self.add_input("in")
        self.add_output("out")
        self.add_output("copy")

    def process(self):
        inp = self.inputs["in"]
        out = self.outputs["out"]
        copy = self.outputs["copy"]

        while True:
            item = inp.get()
            if item is _STOP:
                out.put(_STOP)
                copy.put(_STOP)
                return
            out.put(item)
            copy.put(item)
            self.items_processed += 1
