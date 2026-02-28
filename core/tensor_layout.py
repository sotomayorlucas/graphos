"""Packet <-> tensor serialization for the RouterGraph."""

import numpy as np

from core.constants import TENSOR_DIM, NUM_CLASSES, DEFAULT_BATCH_SIZE


def packet_to_tensor(raw_bytes: bytes) -> np.ndarray:
    """Convert raw packet bytes to a normalized FP32 tensor.

    Pads or truncates to TENSOR_DIM bytes, then normalizes each byte to [0, 1].

    Args:
        raw_bytes: Raw packet bytes (any length).

    Returns:
        np.ndarray of shape (1, TENSOR_DIM) with dtype float32.
    """
    data = bytearray(raw_bytes[:TENSOR_DIM])
    # Pad with zeros if shorter than TENSOR_DIM
    if len(data) < TENSOR_DIM:
        data.extend(b'\x00' * (TENSOR_DIM - len(data)))
    arr = np.frombuffer(bytes(data), dtype=np.uint8).astype(np.float32)
    arr = arr / 255.0
    return arr.reshape(1, TENSOR_DIM)


def packets_to_batch_tensor(
    raw_packets: list[bytes],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Stack raw packets into a batched tensor, padding short batches with zeros.

    Args:
        raw_packets: List of raw packet byte strings (1 to batch_size).
        batch_size: Expected batch dimension B.

    Returns:
        np.ndarray of shape (batch_size, TENSOR_DIM) with dtype float32.

    Raises:
        ValueError: If raw_packets is empty or exceeds batch_size.
    """
    if len(raw_packets) == 0:
        raise ValueError("raw_packets must not be empty")
    if len(raw_packets) > batch_size:
        raise ValueError(
            f"Got {len(raw_packets)} packets but batch_size is {batch_size}"
        )

    rows = []
    for pkt in raw_packets:
        data = bytearray(pkt[:TENSOR_DIM])
        if len(data) < TENSOR_DIM:
            data.extend(b'\x00' * (TENSOR_DIM - len(data)))
        arr = np.frombuffer(bytes(data), dtype=np.uint8).astype(np.float32) / 255.0
        rows.append(arr)

    # Pad remaining rows with zeros
    for _ in range(batch_size - len(raw_packets)):
        rows.append(np.zeros(TENSOR_DIM, dtype=np.float32))

    return np.stack(rows)  # (batch_size, TENSOR_DIM)


def batch_tensor_to_classes(output: np.ndarray) -> list[int]:
    """Convert batched model output logits to class indices via argmax per row.

    Args:
        output: np.ndarray of shape (B, NUM_CLASSES).

    Returns:
        List of integer class indices, one per row.
    """
    return np.argmax(output, axis=1).tolist()


def tensor_to_class(output: np.ndarray) -> int:
    """Convert model output logits to a class index via argmax.

    Args:
        output: np.ndarray of shape (1, NUM_CLASSES) or (NUM_CLASSES,).

    Returns:
        Integer class index.
    """
    return int(np.argmax(output))
