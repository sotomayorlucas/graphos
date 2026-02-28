"""BatchNode — accumulates items into fixed-size batches."""

from dataflow.primitives import _STOP
from dataflow.node import Node


class BatchNode(Node):
    """Collects items into (list, count) tuples of up to batch_size."""

    def __init__(self, name: str, batch_size: int):
        super().__init__(name)
        self.batch_size = batch_size
        self.add_input("in")
        self.add_output("out")

    def process(self):
        inp = self.inputs["in"]
        out = self.outputs["out"]
        buffer = []

        while True:
            item = inp.get()
            if item is _STOP:
                # Flush partial batch
                if buffer:
                    out.put((list(buffer), len(buffer)))
                out.put(_STOP)
                return
            buffer.append(item)
            self.items_processed += 1
            if len(buffer) >= self.batch_size:
                out.put((list(buffer), len(buffer)))
                buffer.clear()
