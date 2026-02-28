"""OpenVINO NPU inference wrapper for RouterGraph."""

import threading

import numpy as np

from core.constants import CLASS_NAMES
from core.tensor_layout import tensor_to_class, batch_tensor_to_classes


class NPUEngine:
    """Runs ONNX model inference on Intel NPU via OpenVINO.

    Falls back to CPU if NPU is not available.
    """

    def __init__(self, model_path: str, device: str = "NPU"):
        import openvino as ov

        self.core = ov.Core()
        available = self.core.available_devices

        if device not in available:
            fallback = "CPU"
            print(f"WARNING: {device} not available (found: {available}). "
                  f"Falling back to {fallback}.")
            device = fallback

        self.device = device
        model = self.core.read_model(model_path)
        self.compiled = self.core.compile_model(model, device)
        self.infer_req = self.compiled.create_infer_request()
        self._ov = ov

    def classify(self, tensor: np.ndarray) -> int:
        """Classify a single packet tensor.

        Args:
            tensor: np.ndarray of shape (1, 64) float32.

        Returns:
            Integer class index.
        """
        self.infer_req.set_input_tensor(self._ov.Tensor(tensor))
        self.infer_req.infer()
        output = self.infer_req.get_output_tensor().data
        return tensor_to_class(output)

    def classify_label(self, tensor: np.ndarray) -> str:
        """Classify and return human-readable label."""
        return CLASS_NAMES[self.classify(tensor)]

    def classify_batch(self, batch_tensor: np.ndarray) -> list[int]:
        """Classify a batch of packet tensors synchronously.

        Args:
            batch_tensor: np.ndarray of shape (B, 64) float32.

        Returns:
            List of integer class indices, one per row.
        """
        self.infer_req.set_input_tensor(self._ov.Tensor(batch_tensor))
        self.infer_req.infer()
        output = self.infer_req.get_output_tensor().data
        return batch_tensor_to_classes(output)

    def classify_async_pipeline(
        self,
        batch_tensors: list[np.ndarray],
        nreq: int = 4,
    ) -> list[list[int]]:
        """Classify multiple batches concurrently using AsyncInferQueue.

        Args:
            batch_tensors: List of np.ndarray, each of shape (B, 64) float32.
            nreq: Number of concurrent inference requests.

        Returns:
            List of class index lists, one per input batch, in order.
        """
        ov = self._ov
        n = len(batch_tensors)
        results = [None] * n
        done = threading.Event()
        counter = [0]  # mutable counter for callback
        lock = threading.Lock()

        infer_queue = ov.AsyncInferQueue(self.compiled, nreq)

        def callback(request, userdata):
            idx = userdata
            output = request.get_output_tensor().data.copy()
            results[idx] = batch_tensor_to_classes(output)
            with lock:
                counter[0] += 1
                if counter[0] == n:
                    done.set()

        infer_queue.set_callback(callback)

        for i, batch in enumerate(batch_tensors):
            infer_queue.start_async({0: ov.Tensor(batch)}, userdata=i)

        infer_queue.wait_all()
        return results
