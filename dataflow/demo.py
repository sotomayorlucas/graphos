"""End-to-end dataflow pipeline demo with benchmark comparison."""

import argparse
import sys
import os

from core.constants import DEFAULT_BATCH_SIZE


def run_demo(n_packets: int = 1000, batch_size: int = DEFAULT_BATCH_SIZE):
    from runtime.benchmark import generate_test_packets, benchmark_cpu
    from dataflow.pipeline import build_classifier_pipeline
    from dataflow.scheduler import Scheduler

    print(f"Dataflow Pipeline Demo")
    print(f"  Packets: {n_packets}, Batch size: {batch_size}")
    print()

    # Generate packets
    packets = generate_test_packets(n=n_packets)

    # Check model exists
    model_path = os.path.join("models", f"router_graph_b{batch_size}.onnx")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print(f"Run: PYTHONIOENCODING=utf-8 python -m router.export_onnx --batch-size {batch_size}")
        sys.exit(1)

    # Build and run pipeline
    print("Running dataflow pipeline...")
    graph, sink = build_classifier_pipeline(
        packets, model_path=model_path, batch_size=batch_size
    )
    scheduler = Scheduler(graph)
    metrics = scheduler.run()

    pipeline_results = sink.results
    pipeline_time = scheduler.wall_time

    Scheduler.print_metrics(metrics, wall_time=pipeline_time)

    # CPU baseline comparison
    print("\nRunning CPU baseline...")
    cpu_latencies, cpu_results = benchmark_cpu(packets)
    cpu_total = sum(cpu_latencies)  # microseconds

    # Agreement
    n_classified = min(len(pipeline_results), len(cpu_results))
    agree = sum(
        1 for a, b in zip(pipeline_results[:n_classified], cpu_results[:n_classified])
        if a == b
    )
    agreement = (agree / n_classified * 100) if n_classified > 0 else 0

    pipeline_pps = n_packets / pipeline_time if pipeline_time > 0 else 0
    cpu_pps = n_packets / (cpu_total / 1e6) if cpu_total > 0 else 0

    print(f"\n{'Metric':<30} {'Pipeline':>14} {'CPU Switch':>14}")
    print("-" * 60)
    print(f"{'Packets classified':<30} {len(pipeline_results):>14} {len(cpu_results):>14}")
    print(f"{'Total time':<30} {pipeline_time:>13.4f}s {cpu_total/1e6:>13.4f}s")
    print(f"{'Throughput':<30} {pipeline_pps:>12.0f}/s {cpu_pps:>12.0f}/s")
    print(f"{'Agreement':<30} {agreement:>13.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataflow pipeline demo")
    parser.add_argument("--n-packets", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    run_demo(n_packets=args.n_packets, batch_size=args.batch_size)
