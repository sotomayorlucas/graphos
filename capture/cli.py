"""CLI entry point for real packet capture and classification."""

import argparse
import os
import sys

from core.constants import DEFAULT_BATCH_SIZE, CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="Classify network packets using the NPU dataflow pipeline"
    )
    parser.add_argument(
        "--pcap", metavar="FILE",
        help="Read packets from a .pcap file (offline mode)",
    )
    parser.add_argument(
        "--iface", metavar="NAME",
        help="Network interface for live capture (needs Npcap)",
    )
    parser.add_argument(
        "--count", type=int, default=0,
        help="Stop after N packets (default: 0 = unlimited)",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Stop after N seconds (default: no timeout)",
    )
    parser.add_argument(
        "--filter", dest="bpf_filter", metavar="BPF",
        help="BPF filter string (e.g., 'tcp port 80 or udp port 53')",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--device", default="NPU",
        help="Inference device (default: NPU)",
    )
    parser.add_argument(
        "--model", metavar="PATH", default=None,
        help="ONNX model path (default: models/router_graph_b{batch_size}.onnx)",
    )
    parser.add_argument(
        "--route", action="store_true",
        help="Enable action routing (dispatch packets to actions by class)",
    )
    parser.add_argument(
        "--action", action="append", choices=["count", "log", "pcap"],
        default=None,
        help="Action to apply (repeatable; default: count). Requires --route",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory for pcap writer (default: output/)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose logging for LogAction",
    )
    args = parser.parse_args()

    model_path = args.model or os.path.join(
        "models", f"router_graph_b{args.batch_size}.onnx"
    )
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print(
            f"Run: PYTHONIOENCODING=utf-8 python -m router.export_onnx"
            f" --batch-size {args.batch_size}"
        )
        sys.exit(1)

    # Build source node
    if args.pcap:
        if not os.path.exists(args.pcap):
            print(f"Pcap file not found: {args.pcap}")
            sys.exit(1)
        from capture.nodes import PcapSourceNode
        source = PcapSourceNode("source", args.pcap)
        mode = f"pcap: {args.pcap}"
    else:
        from capture.nodes import LiveCaptureNode
        source = LiveCaptureNode(
            "source",
            iface=args.iface,
            count=args.count,
            timeout=args.timeout,
            bpf_filter=args.bpf_filter,
        )
        mode = f"live: {args.iface or 'default interface'}"

    print(f"{'Routing' if args.route else 'Capture'} Pipeline")
    print(f"  Mode: {mode}")
    print(f"  Batch size: {args.batch_size}, Device: {args.device}")
    if args.count > 0:
        print(f"  Count limit: {args.count}")
    if args.timeout is not None:
        print(f"  Timeout: {args.timeout}s")
    if args.bpf_filter:
        print(f"  BPF filter: {args.bpf_filter}")
    print()

    if not args.pcap:
        print("Capturing... (Ctrl+C to stop)")

    if args.route:
        _run_routing_pipeline(args, source, model_path)
    else:
        _run_capture_pipeline(args, source, model_path)


def _run_capture_pipeline(args, source, model_path):
    from capture.pipeline import build_capture_pipeline
    from dataflow.scheduler import Scheduler

    graph, sink = build_capture_pipeline(
        source, model_path=model_path,
        batch_size=args.batch_size, device=args.device,
    )
    scheduler = Scheduler(graph)
    metrics = scheduler.run()

    results = sink.results
    total = len(results)

    print(f"\n{'='*50}")
    print(f"Classification Summary")
    print(f"{'='*50}")
    print(f"Total packets classified: {total}")

    if total > 0:
        class_counts = {}
        for cls_id in results:
            name = CLASS_NAMES.get(cls_id, f"UNKNOWN({cls_id})")
            class_counts[name] = class_counts.get(name, 0) + 1

        print(f"\n{'Class':<20} {'Count':>10} {'Percent':>10}")
        print("-" * 42)
        for name, count in sorted(class_counts.items()):
            pct = count / total * 100
            print(f"{name:<20} {count:>10} {pct:>9.1f}%")

        wall = scheduler.wall_time
        if wall > 0:
            pps = total / wall
            print(f"\nThroughput: {pps:,.0f} packets/s ({wall:.4f}s wall time)")

    Scheduler.print_metrics(metrics, wall_time=scheduler.wall_time)


def _run_routing_pipeline(args, source, model_path):
    from capture.pipeline import build_routing_pipeline
    from dataflow.scheduler import Scheduler
    from actions.counter import CountAction
    from actions.log_action import LogAction
    from actions.pcap_writer import PcapWriteAction
    from core.constants import NUM_CLASSES

    action_names = args.action or ["count"]

    # Build action list — CountAction always included for summary
    action_list = []
    counter = CountAction()
    action_list.append(counter)

    if "log" in action_names:
        action_list.append(LogAction(verbose=args.verbose))
    if "pcap" in action_names:
        action_list.append(PcapWriteAction(output_dir=args.output_dir))

    # Map every class to the same action list
    actions_map = {cid: action_list for cid in range(NUM_CLASSES)}

    print(f"  Actions: {', '.join(action_names)}")
    print()

    graph, router_sink = build_routing_pipeline(
        source, actions=actions_map, model_path=model_path,
        batch_size=args.batch_size, device=args.device,
    )
    scheduler = Scheduler(graph)
    metrics = scheduler.run()

    # Summary from CountAction
    counts = counter.summary()
    total = sum(counts.values())

    print(f"\n{'='*50}")
    print(f"Routing Summary")
    print(f"{'='*50}")
    print(f"Total packets routed: {total}")

    if total > 0:
        print(f"\n{'Class':<20} {'Count':>10} {'Percent':>10}")
        print("-" * 42)
        for cid, count in sorted(counts.items()):
            name = CLASS_NAMES.get(cid, f"UNKNOWN({cid})")
            pct = count / total * 100
            print(f"{name:<20} {count:>10} {pct:>9.1f}%")

        wall = scheduler.wall_time
        if wall > 0:
            pps = total / wall
            print(f"\nThroughput: {pps:,.0f} packets/s ({wall:.4f}s wall time)")

    Scheduler.print_metrics(metrics, wall_time=scheduler.wall_time)


if __name__ == "__main__":
    main()
