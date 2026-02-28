"""Graph registry with DAG validation and cycle detection."""

from collections import deque

from dataflow.primitives import Channel
from dataflow.node import Node


class GraphError(Exception):
    """Raised for graph construction errors (duplicates, cycles, etc.)."""


class Graph:
    """Directed acyclic graph of Nodes connected by Channels."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        # Adjacency: src_name -> set of dst_names
        self._edges: dict[str, set[str]] = {}

    def add_node(self, node: Node):
        if node.name in self._nodes:
            raise GraphError(f"Duplicate node name: '{node.name}'")
        self._nodes[node.name] = node
        self._edges.setdefault(node.name, set())

    def connect(
        self,
        src: str,
        src_port: str,
        dst: str,
        dst_port: str,
        capacity: int = 64,
    ):
        """Create a Channel wiring src's output port to dst's input port."""
        if src not in self._nodes:
            raise GraphError(f"Unknown source node: '{src}'")
        if dst not in self._nodes:
            raise GraphError(f"Unknown destination node: '{dst}'")

        src_node = self._nodes[src]
        dst_node = self._nodes[dst]

        if src_port not in src_node.outputs:
            raise GraphError(f"Node '{src}' has no output port '{src_port}'")
        if dst_port not in dst_node.inputs:
            raise GraphError(f"Node '{dst}' has no input port '{dst_port}'")

        # Check for cycles: would adding src->dst create a cycle?
        if src == dst:
            raise GraphError(f"Self-loop on node '{src}'")
        if self._would_create_cycle(src, dst):
            raise GraphError(
                f"Adding edge {src}->{dst} would create a cycle"
            )

        channel = Channel(capacity)
        src_node.outputs[src_port].channel = channel
        dst_node.inputs[dst_port].channel = channel
        self._edges[src].add(dst)

    def _would_create_cycle(self, src: str, dst: str) -> bool:
        """Check if adding dst->...->src path exists (which means src->dst creates cycle)."""
        # If there's a path from dst to src, adding src->dst creates a cycle.
        visited = set()
        stack = [dst]
        while stack:
            node = stack.pop()
            if node == src:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._edges.get(node, set()))
        return False

    def validate(self):
        """Check that all ports on all nodes are connected."""
        errors = []
        for name, node in self._nodes.items():
            for port_name, port in node.inputs.items():
                if not port.connected:
                    errors.append(f"Node '{name}' input '{port_name}' unconnected")
            for port_name, port in node.outputs.items():
                if not port.connected:
                    errors.append(f"Node '{name}' output '{port_name}' unconnected")
        if errors:
            raise GraphError("Validation failed:\n  " + "\n  ".join(errors))

    def source_nodes(self) -> list[Node]:
        """Nodes with no input ports."""
        return [n for n in self._nodes.values() if not n.inputs]

    def sink_nodes(self) -> list[Node]:
        """Nodes with no output ports."""
        return [n for n in self._nodes.values() if not n.outputs]

    def topological_order(self) -> list[Node]:
        """Kahn's algorithm for topological sort."""
        # Build in-degree map
        in_degree: dict[str, int] = {name: 0 for name in self._nodes}
        for src, dsts in self._edges.items():
            for dst in dsts:
                in_degree[dst] += 1

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        order = []

        while queue:
            name = queue.popleft()
            order.append(self._nodes[name])
            for dst in self._edges.get(name, set()):
                in_degree[dst] -= 1
                if in_degree[dst] == 0:
                    queue.append(dst)

        if len(order) != len(self._nodes):
            raise GraphError("Graph contains a cycle")

        return order

    @property
    def nodes(self) -> dict[str, Node]:
        return dict(self._nodes)
