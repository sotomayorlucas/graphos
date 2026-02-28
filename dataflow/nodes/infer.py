"""InferNode — runs NPU/CPU inference on tensor batches."""

import numpy as np

from dataflow.primitives import _STOP
from dataflow.node import Node


class InferNode(Node):
    """Receives (ndarray, count), runs classify_batch(), outputs results[:count]."""

    def __init__(self, name: str, model_path: str, device: str = "NPU"):
        super().__init__(name)
        self._model_path = model_path
        self._device = device
        self._engine = None
        self.add_input("in")
        self.add_output("out")

    def setup(self):
        # Lazy import — avoids OpenVINO dependency at graph construction time.
        from runtime.npu_engine import NPUEngine
        self._engine = NPUEngine(self._model_path, device=self._device)

    def process(self):
        inp = self.inputs["in"]
        out = self.outputs["out"]

        while True:
            item = inp.get()
            if item is _STOP:
                out.put(_STOP)
                return
            tensor, count = item
            results = self._engine.classify_batch(tensor)
            out.put(results[:count])
            self.items_processed += count

    def teardown(self):
        self._engine = None


class KernelInferNode(Node):
    """Inference node backed by a shared KernelRuntime.

    Same contract as InferNode: reads (ndarray, count) from "in",
    emits list[int] to "out". Multiple KernelInferNodes can share
    one runtime executing different programs.
    """

    def __init__(self, name: str, runtime, program_name: str):
        super().__init__(name)
        self._runtime = runtime
        self._program_name = program_name
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
            tensor, count = item
            raw_output = self._runtime.execute(self._program_name, tensor)
            results = np.argmax(raw_output, axis=1).tolist()
            out.put(results[:count])
            self.items_processed += count


class RawInferNode(Node):
    """Inference node that outputs raw tensors (no argmax).

    Same input contract as KernelInferNode: reads (ndarray, count) from "in".
    Outputs (ndarray, count) to "out" — raw logits for pipeline chaining.
    """

    def __init__(self, name: str, runtime, program_name: str):
        super().__init__(name)
        self._runtime = runtime
        self._program_name = program_name
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
            tensor, count = item
            raw_output = self._runtime.execute(self._program_name, tensor)
            out.put((raw_output, count))
            self.items_processed += count
