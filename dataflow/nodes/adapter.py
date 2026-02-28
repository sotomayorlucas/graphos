"""AdapterNode — shape transform between inference nodes in dataflow."""

from dataflow.primitives import _STOP
from dataflow.node import Node


class AdapterNode(Node):
    """Dataflow node wrapping a TensorAdapter.

    Single-input mode: reads (ndarray, count) from "in", applies adapter, emits to "out".
    Dual-input mode (needs_raw=True): also reads from "raw" input for concat adapters.
    """

    def __init__(self, name: str, adapter, needs_raw: bool = False):
        super().__init__(name)
        self._adapter = adapter
        self._needs_raw = needs_raw
        self.add_input("in")
        if needs_raw:
            self.add_input("raw")
        self.add_output("out")

    def process(self):
        inp = self.inputs["in"]
        out = self.outputs["out"]
        raw_port = self.inputs.get("raw") if self._needs_raw else None

        while True:
            item = inp.get()
            if item is _STOP:
                # Drain raw port if needed
                if raw_port is not None:
                    raw_port.get()  # consume matching _STOP
                out.put(_STOP)
                return

            tensor, count = item
            if self._needs_raw:
                raw_item = raw_port.get()
                if raw_item is _STOP:
                    out.put(_STOP)
                    return
                raw_tensor, _ = raw_item
                inputs = {"left": raw_tensor, "right": tensor}
            else:
                inputs = {"in": tensor}

            result = self._adapter.execute(inputs)
            out.put((result, count))
            self.items_processed += count
