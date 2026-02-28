"""TensorNode — converts packet batches to tensor batches."""

from dataflow.primitives import _STOP
from dataflow.node import Node
from core.tensor_layout import packets_to_batch_tensor


class TensorNode(Node):
    """Receives (list[bytes], count), outputs (ndarray, count)."""

    def __init__(self, name: str, batch_size: int):
        super().__init__(name)
        self.batch_size = batch_size
        self.add_input("in")
        self.add_output("out")

    def process(self):
        inp = self.inputs["in"]
        out = self.outputs["out"]

        while True:
            item = inp.get()
            if item is _STOP:
                out.put(_STOP)
                return
            packets, count = item
            tensor = packets_to_batch_tensor(packets, batch_size=self.batch_size)
            out.put((tensor, count))
            self.items_processed += count
