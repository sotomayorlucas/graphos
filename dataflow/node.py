"""Abstract Node base class with lifecycle hooks."""

import time
from abc import ABC, abstractmethod

from dataflow.primitives import InputPort, OutputPort


class Node(ABC):
    """Abstract dataflow processing node.

    Subclasses implement process() to read from inputs, compute, and write
    to outputs. The run() method manages the setup/process/teardown lifecycle.
    """

    def __init__(self, name: str):
        self.name = name
        self.inputs: dict[str, InputPort] = {}
        self.outputs: dict[str, OutputPort] = {}
        self.items_processed: int = 0
        self.elapsed: float = 0.0

    def add_input(self, port_name: str) -> InputPort:
        port = InputPort(port_name)
        self.inputs[port_name] = port
        return port

    def add_output(self, port_name: str) -> OutputPort:
        port = OutputPort(port_name)
        self.outputs[port_name] = port
        return port

    def setup(self):
        """Called before process(). Override for initialization."""

    def teardown(self):
        """Called after process(). Override for cleanup."""

    @abstractmethod
    def process(self):
        """Main processing loop. Must handle _STOP sentinel."""

    def run(self):
        """Execute the full lifecycle: setup -> process -> teardown."""
        self.setup()
        t0 = time.perf_counter()
        try:
            self.process()
        finally:
            self.elapsed = time.perf_counter() - t0
            self.teardown()
