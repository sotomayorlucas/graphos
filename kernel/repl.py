"""GraphOS Shell — interactive REPL for exploring program composition."""

import cmd
import os
import random
import sys

import numpy as np

from core.constants import (
    TENSOR_DIM, NUM_CLASSES, NUM_ROUTES, DEFAULT_BATCH_SIZE,
    CLASS_NAMES, ROUTE_NAMES,
)
from core.tensor_layout import packets_to_batch_tensor
from kernel.runtime import KernelRuntime, KernelError
from kernel.programs.classifier import classifier_spec
from kernel.programs.route_table import route_table_spec
from kernel.programs.composed_router import composed_router_spec, COMPOSED_INPUT_DIM
from kernel.compose import concat_adapter, ProgramPipeline


class GraphOSShell(cmd.Cmd):
    """Interactive shell for GraphOS kernel exploration."""

    intro = (
        "=== GraphOS Shell ===\n"
        "Type 'help' for commands. Tab-completion available.\n"
    )
    prompt = "graphos> "

    def __init__(self, runtime=None, batch_size=DEFAULT_BATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = batch_size
        self._runtime = runtime or KernelRuntime()
        self._packets: list[bytes] = []
        self._labels: list[str] = []
        self._pipeline: ProgramPipeline | None = None
        self._last_results: dict[str, object] = {}
        self._exec_history: list[dict] = []

    # --- Spec helpers ---

    def _spec_for(self, name: str):
        """Return the ProgramSpec for a known program name."""
        if name == "classifier":
            return classifier_spec(self._batch_size)
        elif name == "route_table":
            return route_table_spec(self._batch_size)
        elif name == "composed_router":
            return composed_router_spec(self._batch_size)
        else:
            return None

    # --- Commands ---

    def do_programs(self, arg):
        """List loaded programs with shapes."""
        programs = self._runtime.programs
        if not programs:
            print("No programs loaded.")
            return
        for name in programs:
            prog = self._runtime.get(name)
            spec = prog.spec
            print(f"  {name}: {spec.input_shape} -> {spec.output_shape}")
            if spec.description:
                print(f"    {spec.description}")

    def do_load(self, arg):
        """Load a program: load <classifier|route_table|composed_router>"""
        name = arg.strip()
        if not name:
            print("Usage: load <classifier|route_table|composed_router>")
            return
        spec = self._spec_for(name)
        if spec is None:
            print(f"Unknown program: {name}")
            return
        if not os.path.exists(spec.onnx_path):
            print(f"Model not found: {spec.onnx_path}")
            return
        try:
            self._runtime.load(spec)
            print(f"Loaded '{name}': {spec.input_shape} -> {spec.output_shape}")
        except KernelError as e:
            print(f"Error: {e}")

    def do_unload(self, arg):
        """Unload a program: unload <name>"""
        name = arg.strip()
        if not name:
            print("Usage: unload <name>")
            return
        try:
            self._runtime.unload(name)
            print(f"Unloaded '{name}'")
        except KernelError as e:
            print(f"Error: {e}")

    def do_send(self, arg):
        """Generate test packets: send <http|dns|other|random> [count]"""
        parts = arg.strip().split()
        if not parts:
            print("Usage: send <http|dns|other|random> [count]")
            return
        pkt_type = parts[0].lower()
        count = int(parts[1]) if len(parts) > 1 else 8

        from router.dataset import (
            generate_http_packet, generate_dns_packet, generate_other_packet,
        )

        generators = {
            "http": (generate_http_packet, "HTTP"),
            "dns": (generate_dns_packet, "DNS"),
            "other": (generate_other_packet, "OTHER"),
        }

        self._packets = []
        self._labels = []
        for _ in range(count):
            if pkt_type == "random":
                gen_name = random.choice(["http", "dns", "other"])
                gen_fn, label = generators[gen_name]
            elif pkt_type in generators:
                gen_fn, label = generators[pkt_type]
            else:
                print(f"Unknown packet type: {pkt_type}")
                return
            self._packets.append(gen_fn())
            self._labels.append(label)

        print(f"Generated {count} {pkt_type} packet(s).")

    def do_send_pcap(self, arg):
        """Load packets from pcap file: send_pcap <file>"""
        path = arg.strip()
        if not path:
            print("Usage: send_pcap <file>")
            return
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        try:
            from scapy.all import rdpcap
            pkts = rdpcap(path)
            self._packets = [bytes(p) for p in pkts]
            self._labels = ["pcap"] * len(self._packets)
            print(f"Loaded {len(self._packets)} packet(s) from {path}")
        except ImportError:
            print("scapy not installed (pip install scapy)")
        except Exception as e:
            print(f"Error reading pcap: {e}")

    def do_run(self, arg):
        """Execute a single program: run <program_name>"""
        name = arg.strip()
        if not name:
            print("Usage: run <program_name>")
            return
        if not self._packets:
            print("No packets loaded. Use 'send' first.")
            return
        if name not in self._runtime.programs:
            print(f"Program '{name}' not loaded. Use 'load' first.")
            return

        prog = self._runtime.get(name)
        spec = prog.spec
        input_dim = spec.input_shape[1]

        # Build tensor
        if input_dim == TENSOR_DIM:
            tensor = packets_to_batch_tensor(self._packets[:self._batch_size], self._batch_size)
        else:
            print(f"Program '{name}' requires input dim {input_dim}, use 'run_pipe' for composed programs.")
            return

        count = min(len(self._packets), self._batch_size)
        output = self._runtime.execute(name, tensor)
        results = np.argmax(output, axis=1).tolist()[:count]

        # Store for later
        self._last_results[name] = {"output": output, "results": results, "count": count}
        self._exec_history.append({"program": name, "count": count})

        # Display
        if spec.output_shape[1] == NUM_CLASSES:
            label_map = CLASS_NAMES
        elif spec.output_shape[1] == NUM_ROUTES:
            label_map = ROUTE_NAMES
        else:
            label_map = {}

        print(f"\n{'#':<4} {'Label':<8} {'Result':<12} {'Logits'}")
        print("-" * 50)
        for i in range(count):
            label = self._labels[i] if i < len(self._labels) else "?"
            result_name = label_map.get(results[i], str(results[i]))
            logits = output[i, :spec.output_shape[1]]
            logits_str = ", ".join(f"{v:+.2f}" for v in logits)
            print(f"  {i:<3} {label:<8} {result_name:<12} [{logits_str}]")

    def do_pipe(self, arg):
        """Define a pipeline: pipe <classifier_prog> <composed_prog>"""
        parts = arg.strip().split()
        if len(parts) != 2:
            print("Usage: pipe <classifier_prog> <composed_prog>")
            return

        first, second = parts
        first_spec = self._spec_for(first)
        second_spec = self._spec_for(second)
        if first_spec is None or second_spec is None:
            print(f"Unknown program(s): {first}, {second}")
            return

        adapter = concat_adapter(
            self._batch_size,
            left_dim=TENSOR_DIM,
            right_dim=first_spec.output_shape[1],
        )

        self._pipeline = ProgramPipeline()
        self._pipeline.add_program(first, first_spec)
        self._pipeline.add_adapter(adapter)
        self._pipeline.add_program(second, second_spec)
        self._pipeline.with_raw_passthrough("raw")

        errors = self._pipeline.validate()
        if errors:
            print("Pipeline validation errors:")
            for e in errors:
                print(f"  - {e}")
            self._pipeline = None
            return

        print(f"Pipeline defined: {first} -> concat -> {second}")

    def do_run_pipe(self, arg):
        """Execute the defined pipeline."""
        if self._pipeline is None:
            print("No pipeline defined. Use 'pipe' first.")
            return
        if not self._packets:
            print("No packets loaded. Use 'send' first.")
            return

        tensor = packets_to_batch_tensor(self._packets[:self._batch_size], self._batch_size)
        count = min(len(self._packets), self._batch_size)

        output = self._pipeline.execute(self._runtime, tensor)
        results = np.argmax(output, axis=1).tolist()[:count]

        self._last_results["pipeline"] = {"output": output, "results": results, "count": count}
        self._exec_history.append({"program": "pipeline", "count": count})

        print(f"\n{'#':<4} {'Label':<8} {'Route':<12} {'Scores'}")
        print("-" * 50)
        for i in range(count):
            label = self._labels[i] if i < len(self._labels) else "?"
            route_name = ROUTE_NAMES.get(results[i], str(results[i]))
            scores = output[i, :NUM_ROUTES]
            scores_str = ", ".join(f"{v:+.2f}" for v in scores)
            print(f"  {i:<3} {label:<8} {route_name:<12} [{scores_str}]")

    def do_compare(self, arg):
        """Side-by-side comparison: standalone route_table vs composed pipeline."""
        if not self._packets:
            print("No packets loaded. Use 'send' first.")
            return

        # Need classifier + route_table loaded, pipeline defined
        needed = []
        if "classifier" not in self._runtime.programs:
            needed.append("classifier")
        if "route_table" not in self._runtime.programs:
            needed.append("route_table")
        if needed:
            print(f"Load programs first: {', '.join(needed)}")
            return
        if self._pipeline is None:
            print("Define a pipeline first with 'pipe'.")
            return

        tensor = packets_to_batch_tensor(self._packets[:self._batch_size], self._batch_size)
        count = min(len(self._packets), self._batch_size)

        # Run standalone
        cls_out = self._runtime.execute("classifier", tensor)
        cls_results = np.argmax(cls_out, axis=1).tolist()[:count]
        rt_out = self._runtime.execute("route_table", tensor)
        rt_results = np.argmax(rt_out, axis=1).tolist()[:count]

        # Run pipeline
        pipe_out = self._pipeline.execute(self._runtime, tensor)
        pipe_results = np.argmax(pipe_out, axis=1).tolist()[:count]

        print(f"\n{'#':<4} {'Label':<8} {'Class':<12} {'Standalone':<12} {'Composed':<12} {'Match'}")
        print("-" * 64)
        matches = 0
        for i in range(count):
            label = self._labels[i] if i < len(self._labels) else "?"
            cls_name = CLASS_NAMES.get(cls_results[i], "?")
            rt_name = ROUTE_NAMES.get(rt_results[i], "?")
            pipe_name = ROUTE_NAMES.get(pipe_results[i], "?")
            match = "=" if rt_results[i] == pipe_results[i] else "*"
            if rt_results[i] == pipe_results[i]:
                matches += 1
            print(f"  {i:<3} {label:<8} {cls_name:<12} {rt_name:<12} {pipe_name:<12} {match}")

        diff = count - matches
        print(f"\nAgreement: {matches}/{count} ({100*matches/count:.0f}%), Differences: {diff}")

    def do_inspect(self, arg):
        """Show pipeline topology and shapes."""
        if self._pipeline is None:
            print("No pipeline defined. Use 'pipe' first.")
            return
        print(f"Pipeline: {self._pipeline.describe()}")
        print(f"Stages:")
        for i, (kind, name) in enumerate(self._pipeline.stages):
            print(f"  {i}: [{kind}] {name}")

    def do_health(self, arg):
        """Runtime health metrics."""
        health = self._runtime.health()
        print(f"Device: {health['device']}")
        print(f"Programs: {health['programs']}")
        print(f"Executions: {health['exec_count']}")
        print(f"Mean latency: {health['mean_latency_us']:.0f} us")
        print(f"Last latency: {health['last_latency_us']:.0f} us")
        print(f"Errors: {health['errors']}")
        print(f"Healthy: {health['healthy']}")

    def do_stats(self, arg):
        """Show execution history."""
        if not self._exec_history:
            print("No executions yet.")
            return
        print(f"Execution history ({len(self._exec_history)} runs):")
        for i, entry in enumerate(self._exec_history):
            print(f"  {i}: {entry['program']} ({entry['count']} packets)")

    def do_quit(self, arg):
        """Exit the shell."""
        print("Goodbye.")
        return True

    do_exit = do_quit
    do_EOF = do_quit

    def emptyline(self):
        """Do nothing on empty input."""
        pass

    def default(self, line):
        print(f"Unknown command: {line}. Type 'help' for available commands.")


def main():
    """Entry point for python -m kernel.repl"""
    shell = GraphOSShell()
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
