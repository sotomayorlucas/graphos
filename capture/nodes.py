"""Dataflow source nodes for packet capture."""

import signal
import threading

from dataflow.primitives import _STOP
from dataflow.node import Node
from capture.source import PcapSource, CaptureSource


class PcapSourceNode(Node):
    """Source node that reads packets from a .pcap file."""

    def __init__(self, name: str, pcap_path: str):
        super().__init__(name)
        self._pcap_path = pcap_path
        self.add_output("out")

    def process(self):
        out = self.outputs["out"]
        source = PcapSource(self._pcap_path)
        for pkt_bytes in source:
            out.put(pkt_bytes)
            self.items_processed += 1
        out.put(_STOP)


class LiveCaptureNode(Node):
    """Source node that captures live packets via scapy sniff."""

    def __init__(
        self,
        name: str,
        iface: str | None = None,
        count: int = 0,
        timeout: float | None = None,
        bpf_filter: str | None = None,
    ):
        super().__init__(name)
        self._iface = iface
        self._count = count
        self._timeout = timeout
        self._bpf_filter = bpf_filter
        self._source: CaptureSource | None = None
        self._prev_sigint = None
        self.add_output("out")

    def setup(self):
        self._source = CaptureSource(
            iface=self._iface,
            count=self._count,
            timeout=self._timeout,
            bpf_filter=self._bpf_filter,
        )
        # Register Ctrl+C handler for graceful shutdown (main thread only)
        try:
            self._prev_sigint = signal.signal(
                signal.SIGINT, self._sigint_handler
            )
        except ValueError:
            # Not in main thread — skip signal registration
            self._prev_sigint = None

    def _sigint_handler(self, signum, frame):
        """Graceful Ctrl+C: stop capture, don't crash."""
        if self._source is not None:
            self._source.stop()

    def process(self):
        out = self.outputs["out"]
        for pkt_bytes in self._source:
            out.put(pkt_bytes)
            self.items_processed += 1
        out.put(_STOP)

    def teardown(self):
        if self._source is not None:
            self._source.stop()
        # Restore previous SIGINT handler
        if self._prev_sigint is not None:
            try:
                signal.signal(signal.SIGINT, self._prev_sigint)
            except ValueError:
                pass
