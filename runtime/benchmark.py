"""Side-by-side latency benchmark: NPU inference vs CPU switch."""

import argparse
import time
import sys
import os

import numpy as np

from core.constants import CLASS_NAMES, TENSOR_DIM, DEFAULT_BATCH_SIZE
from core.tensor_layout import packet_to_tensor, packets_to_batch_tensor
from router.dataset import generate_http_packet, generate_dns_packet, generate_other_packet
from runtime.cpu_baseline import classify_packet_switch


def generate_test_packets(n=10000, seed=123):
    """Generate n random test packets with balanced classes."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    packets = []
    generators = [generate_http_packet, generate_dns_packet, generate_other_packet]
    per_class = n // 3
    remainder = n - per_class * 3

    for gen in generators:
        for _ in range(per_class):
            packets.append(gen())
    for _ in range(remainder):
        packets.append(generate_other_packet())

    random.shuffle(packets)
    return packets


def percentile(values, p):
    """Compute percentile of a sorted list."""
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(values):
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def benchmark_cpu(packets):
    """Benchmark CPU switch classification."""
    latencies = []
    results = []
    for pkt in packets:
        t0 = time.perf_counter_ns()
        cls = classify_packet_switch(pkt)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)  # Convert to microseconds
        results.append(cls)
    return latencies, results


def benchmark_npu(packets, model_path="models/router_graph.onnx", warmup=100):
    """Benchmark NPU (or fallback) single inference."""
    from runtime.npu_engine import NPUEngine

    engine = NPUEngine(model_path)
    tensors = [packet_to_tensor(pkt) for pkt in packets]

    # Warmup
    for i in range(min(warmup, len(tensors))):
        engine.classify(tensors[i])

    latencies = []
    results = []
    for t in tensors:
        t0 = time.perf_counter_ns()
        cls = engine.classify(t)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)  # Convert to microseconds
        results.append(cls)

    return latencies, results, engine.device


def benchmark_npu_batched(packets, batch_size=DEFAULT_BATCH_SIZE,
                          model_path=None, warmup=10):
    """Benchmark NPU batched inference throughput."""
    from runtime.npu_engine import NPUEngine

    if model_path is None:
        model_path = f"models/router_graph_b{batch_size}.onnx"

    engine = NPUEngine(model_path)

    # Build batch tensors
    batches = []
    for i in range(0, len(packets), batch_size):
        chunk = packets[i:i + batch_size]
        tensor = packets_to_batch_tensor(chunk, batch_size)
        batches.append((tensor, len(chunk)))

    # Warmup
    for i in range(min(warmup, len(batches))):
        engine.classify_batch(batches[i][0])

    # Timed run
    results = []
    t0 = time.perf_counter_ns()
    for tensor, actual in batches:
        cls_list = engine.classify_batch(tensor)
        results.extend(cls_list[:actual])
    t1 = time.perf_counter_ns()

    total_us = (t1 - t0) / 1000.0
    return total_us, results, engine.device


def benchmark_npu_async(packets, batch_size=DEFAULT_BATCH_SIZE,
                        model_path=None, nreq=4, warmup=10):
    """Benchmark NPU async pipeline throughput."""
    from runtime.npu_engine import NPUEngine

    if model_path is None:
        model_path = f"models/router_graph_b{batch_size}.onnx"

    engine = NPUEngine(model_path)

    # Build batch tensors
    batch_tensors = []
    actual_counts = []
    for i in range(0, len(packets), batch_size):
        chunk = packets[i:i + batch_size]
        tensor = packets_to_batch_tensor(chunk, batch_size)
        batch_tensors.append(tensor)
        actual_counts.append(len(chunk))

    # Warmup
    if warmup > 0 and batch_tensors:
        warmup_batches = batch_tensors[:min(warmup, len(batch_tensors))]
        engine.classify_async_pipeline(warmup_batches, nreq=nreq)

    # Timed run
    t0 = time.perf_counter_ns()
    all_results = engine.classify_async_pipeline(batch_tensors, nreq=nreq)
    t1 = time.perf_counter_ns()

    total_us = (t1 - t0) / 1000.0
    results = []
    for cls_list, actual in zip(all_results, actual_counts):
        results.extend(cls_list[:actual])

    return total_us, results, engine.device


def print_stats(name, latencies_us):
    """Print latency statistics."""
    sorted_lat = sorted(latencies_us)
    mean = sum(sorted_lat) / len(sorted_lat)
    p50 = percentile(sorted_lat, 50)
    p95 = percentile(sorted_lat, 95)
    p99 = percentile(sorted_lat, 99)

    print(f"  {name}:")
    print(f"    Mean:  {mean:>10.2f} us")
    print(f"    P50:   {p50:>10.2f} us")
    print(f"    P95:   {p95:>10.2f} us")
    print(f"    P99:   {p99:>10.2f} us")
    print(f"    Total: {sum(sorted_lat)/1000:>10.2f} ms ({len(sorted_lat)} packets)")


def print_throughput_stats(name, total_us, n_packets, device):
    """Print throughput statistics for batched/async modes."""
    total_ms = total_us / 1000.0
    total_s = total_us / 1_000_000.0
    pps = n_packets / total_s if total_s > 0 else 0
    per_pkt_us = total_us / n_packets if n_packets > 0 else 0

    print(f"  {name} ({device}):")
    print(f"    Total:       {total_ms:>10.2f} ms ({n_packets} packets)")
    print(f"    Per-packet:  {per_pkt_us:>10.2f} us")
    print(f"    Throughput:  {pps:>10.0f} packets/sec")


def run_benchmark(n_packets=10000, batch_size=DEFAULT_BATCH_SIZE,
                  model_path="models/router_graph.onnx"):
    """Run full benchmark comparison."""
    if not os.path.exists(model_path):
        print(f"ERROR: ONNX model not found at {model_path}")
        print("Run training and export first:")
        print("  python -m router.train")
        print("  python -m router.export_onnx")
        sys.exit(1)

    batched_model_path = f"models/router_graph_b{batch_size}.onnx"
    has_batched = os.path.exists(batched_model_path)

    print(f"Generating {n_packets} test packets...")
    packets = generate_test_packets(n_packets)

    # --- CPU Switch ---
    print("\n--- CPU Switch Baseline ---")
    cpu_latencies, cpu_results = benchmark_cpu(packets)
    print_stats("CPU Switch", cpu_latencies)
    cpu_total_us = sum(cpu_latencies)
    cpu_pps = n_packets / (cpu_total_us / 1_000_000.0) if cpu_total_us > 0 else 0

    # --- NPU Single ---
    print("\n--- NPU Single Inference ---")
    npu_latencies, npu_results, device = benchmark_npu(packets, model_path)
    print_stats(f"Inference ({device})", npu_latencies)
    npu_total_us = sum(npu_latencies)
    npu_pps = n_packets / (npu_total_us / 1_000_000.0) if npu_total_us > 0 else 0

    # Accuracy check
    matches = sum(1 for a, b in zip(cpu_results, npu_results) if a == b)
    accuracy = matches / len(cpu_results) * 100
    print(f"\n  NPU vs CPU agreement: {matches}/{len(cpu_results)} ({accuracy:.2f}%)")

    # --- NPU Batched ---
    batched_pps = 0
    async_pps = 0
    if has_batched:
        print(f"\n--- NPU Batched (B={batch_size}) ---")
        b_total_us, b_results, b_device = benchmark_npu_batched(
            packets, batch_size, batched_model_path)
        print_throughput_stats(f"Batched B={batch_size}", b_total_us, n_packets, b_device)
        batched_pps = n_packets / (b_total_us / 1_000_000.0) if b_total_us > 0 else 0

        b_matches = sum(1 for a, b in zip(cpu_results, b_results) if a == b)
        b_acc = b_matches / len(cpu_results) * 100
        print(f"  Batched vs CPU agreement: {b_matches}/{len(cpu_results)} ({b_acc:.2f}%)")

        # --- NPU Async ---
        print(f"\n--- NPU Async Pipeline (B={batch_size}, nreq=4) ---")
        a_total_us, a_results, a_device = benchmark_npu_async(
            packets, batch_size, batched_model_path)
        print_throughput_stats(f"Async B={batch_size}", a_total_us, n_packets, a_device)
        async_pps = n_packets / (a_total_us / 1_000_000.0) if a_total_us > 0 else 0

        a_matches = sum(1 for a, b in zip(cpu_results, a_results) if a == b)
        a_acc = a_matches / len(cpu_results) * 100
        print(f"  Async vs CPU agreement: {a_matches}/{len(cpu_results)} ({a_acc:.2f}%)")
    else:
        print(f"\n--- Batched/Async benchmarks skipped ---")
        print(f"  Export batched model first:")
        print(f"  PYTHONIOENCODING=utf-8 python -m router.export_onnx --batch-size {batch_size}")

    # --- Comparison Table ---
    print(f"\n{'='*60}")
    print(f"  THROUGHPUT COMPARISON ({n_packets} packets)")
    print(f"{'='*60}")
    print(f"  {'Mode':<30} {'Packets/sec':>12}")
    print(f"  {'-'*42}")
    print(f"  {'CPU Switch':<30} {cpu_pps:>12,.0f}")
    print(f"  {'NPU Single':<30} {npu_pps:>12,.0f}")
    if has_batched:
        print(f"  {f'NPU Batched (B={batch_size})':<30} {batched_pps:>12,.0f}")
        print(f"  {f'NPU Async (B={batch_size}, nreq=4)':<30} {async_pps:>12,.0f}")
    print(f"  {'-'*42}")

    if has_batched and batched_pps > 0:
        print(f"  Batched vs Single speedup: {batched_pps/npu_pps:.1f}x")
        if async_pps > 0:
            print(f"  Async vs Single speedup:   {async_pps/npu_pps:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NPU vs CPU")
    parser.add_argument("--n-packets", type=int, default=10000,
                        help="Number of test packets (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for batched modes (default: {DEFAULT_BATCH_SIZE})")
    args = parser.parse_args()

    run_benchmark(n_packets=args.n_packets, batch_size=args.batch_size)
