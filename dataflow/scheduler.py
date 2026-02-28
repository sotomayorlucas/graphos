"""Thread-per-node Scheduler with metrics collection."""

import threading
import time
from dataclasses import dataclass, field

from dataflow.primitives import _STOP
from dataflow.node import Node
from dataflow.graph import Graph


@dataclass
class NodeMetrics:
    """Per-node execution metrics."""
    name: str
    items_processed: int = 0
    elapsed: float = 0.0
    throughput: float = 0.0
    error: str | None = None


class Scheduler:
    """Runs each node in its own thread, in topological order."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.wall_time: float = 0.0
        self._metrics: dict[str, NodeMetrics] = {}

    def run(self) -> dict[str, NodeMetrics]:
        """Start all node threads, join them, and return metrics."""
        self.graph.validate()
        order = self.graph.topological_order()

        threads: list[threading.Thread] = []
        for node in order:
            t = threading.Thread(
                target=self._run_node,
                args=(node,),
                name=f"node-{node.name}",
                daemon=True,
            )
            threads.append(t)

        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.wall_time = time.perf_counter() - t0

        return dict(self._metrics)

    def _run_node(self, node: Node):
        """Thread target: run node, catch exceptions, propagate _STOP on error."""
        metrics = NodeMetrics(name=node.name)
        try:
            node.run()
            metrics.items_processed = node.items_processed
            metrics.elapsed = node.elapsed
            if metrics.elapsed > 0:
                metrics.throughput = metrics.items_processed / metrics.elapsed
        except Exception as e:
            metrics.error = str(e)
            # Propagate _STOP downstream so dependent nodes shut down
            for port in node.outputs.values():
                if port.connected:
                    try:
                        port.put(_STOP)
                    except Exception:
                        pass
        finally:
            self._metrics[node.name] = metrics

    @staticmethod
    def print_metrics(metrics: dict[str, NodeMetrics], wall_time: float = 0.0):
        """Print a formatted metrics table."""
        print(f"\n{'Node':<20} {'Items':>10} {'Time (s)':>10} {'Throughput':>14} {'Error'}")
        print("-" * 70)
        for m in metrics.values():
            thr = f"{m.throughput:>12.0f}/s" if m.throughput > 0 else "           -"
            err = m.error or ""
            print(f"{m.name:<20} {m.items_processed:>10} {m.elapsed:>10.4f} {thr} {err}")
        if wall_time > 0:
            print(f"\nWall time: {wall_time:.4f}s")
