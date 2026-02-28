"""Kernel demo — loads classifier + route_table, processes test packets."""

import os
import sys

import numpy as np

from core.constants import (
    TENSOR_DIM, DEFAULT_BATCH_SIZE,
    CLASS_NAMES, ROUTE_NAMES,
)
from core.tensor_layout import packets_to_batch_tensor, batch_tensor_to_classes
from kernel.runtime import KernelRuntime
from kernel.programs.classifier import classifier_spec
from kernel.programs.route_table import route_table_spec
from router.dataset import generate_http_packet, generate_dns_packet, generate_other_packet


def generate_demo_packets() -> list[tuple[bytes, str]]:
    """Generate structured packets with known types for demo."""
    packets = []
    for _ in range(6):
        packets.append((generate_http_packet(), "HTTP"))
    for _ in range(6):
        packets.append((generate_dns_packet(), "DNS"))
    for _ in range(6):
        packets.append((generate_other_packet(), "OTHER"))
    return packets


def main():
    batch_size = DEFAULT_BATCH_SIZE

    # Check models exist
    cls_spec = classifier_spec(batch_size)
    rt_spec = route_table_spec(batch_size)

    for spec in [cls_spec, rt_spec]:
        if not os.path.exists(spec.onnx_path):
            print(f"ERROR: Model not found: {spec.onnx_path}")
            if spec.name == "classifier":
                print("Run: python -m router.export_onnx --batch-size 64")
            else:
                print("Run: PYTHONIOENCODING=utf-8 python -m kernel.programs.export_route_table --batch-size 64")
            sys.exit(1)

    # Boot kernel
    print("=== GraphOS Kernel Demo ===\n")
    runtime = KernelRuntime()
    print(f"Device: {runtime.device}")
    print(f"Available: {runtime.device_info()['available_devices']}\n")

    # Load programs
    runtime.load(cls_spec)
    runtime.load(rt_spec)
    print(f"Loaded programs: {runtime.programs}\n")

    # Generate structured packets
    demo_packets = generate_demo_packets()
    raw_packets = [pkt for pkt, _ in demo_packets]
    labels = [lbl for _, lbl in demo_packets]
    tensor = packets_to_batch_tensor(raw_packets, batch_size)

    # Run classifier
    cls_output = runtime.execute("classifier", tensor)
    cls_results = batch_tensor_to_classes(cls_output)[:len(raw_packets)]

    # Run route table
    rt_output = runtime.execute("route_table", tensor)
    rt_results = np.argmax(rt_output, axis=1).tolist()[:len(raw_packets)]

    # Display results
    print(f"{'#':<4} {'Expected':<10} {'Class':<12} {'Route':<10}")
    print("-" * 36)
    for i in range(len(raw_packets)):
        cls_name = CLASS_NAMES.get(cls_results[i], "?")
        rt_name = ROUTE_NAMES.get(rt_results[i], "?")
        print(f"  {i:<3} {labels[i]:<10} {cls_name:<12} {rt_name:<10}")

    # Summary
    print(f"\n--- Summary ---")
    cls_correct = sum(1 for i, lbl in enumerate(labels)
                      if (lbl == "HTTP" and cls_results[i] == 0)
                      or (lbl == "DNS" and cls_results[i] == 1)
                      or (lbl == "OTHER" and cls_results[i] == 2))
    print(f"Classifier accuracy: {cls_correct}/{len(labels)}")

    from collections import Counter
    route_counts = Counter(ROUTE_NAMES[r] for r in rt_results)
    print(f"Route distribution: {dict(route_counts)}")

    # Health
    health = runtime.health()
    print(f"\nDevice: {health['device']}")
    print(f"Executions: {health['exec_count']}")
    print(f"Mean latency: {health['mean_latency_us']:.0f} us")
    print(f"Healthy: {health['healthy']}")


if __name__ == "__main__":
    main()
