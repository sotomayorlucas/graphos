"""Packet sources: live capture (scapy sniff) and offline pcap file reader."""

import queue
import threading


class PcapSource:
    """Iterable that yields raw bytes from a .pcap file."""

    def __init__(self, pcap_path: str):
        self._path = pcap_path

    def __iter__(self):
        from scapy.utils import rdpcap

        packets = rdpcap(self._path)
        for pkt in packets:
            yield bytes(pkt)


class CaptureSource:
    """Iterable live packet capture using scapy.sniff().

    Bridges scapy's callback-based sniff to a synchronous iterator via a queue.
    Thread-safe stop via threading.Event for graceful Ctrl+C shutdown.
    """

    def __init__(
        self,
        iface: str | None = None,
        count: int = 0,
        timeout: float | None = None,
        bpf_filter: str | None = None,
    ):
        self._iface = iface
        self._count = count
        self._timeout = timeout
        self._bpf_filter = bpf_filter
        self._queue: queue.Queue = queue.Queue(maxsize=4096)
        self._stop_event = threading.Event()
        self._sniff_thread: threading.Thread | None = None

    def _packet_callback(self, pkt):
        """Called by scapy for each captured packet."""
        if self._stop_event.is_set():
            return
        try:
            self._queue.put(bytes(pkt), timeout=1.0)
        except queue.Full:
            pass  # Drop packet if queue is full

    def _sniff_worker(self):
        """Run scapy.sniff in a background thread."""
        from scapy.sendrecv import sniff

        kwargs = {
            "prn": self._packet_callback,
            "store": False,
            "stop_filter": lambda _: self._stop_event.is_set(),
        }
        if self._iface is not None:
            kwargs["iface"] = self._iface
        if self._count > 0:
            kwargs["count"] = self._count
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        if self._bpf_filter is not None:
            kwargs["filter"] = self._bpf_filter

        try:
            sniff(**kwargs)
        except Exception:
            pass  # Interface not available, permission error, etc.
        finally:
            self._stop_event.set()

    def __iter__(self):
        self._sniff_thread = threading.Thread(
            target=self._sniff_worker, daemon=True
        )
        self._sniff_thread.start()

        while True:
            if self._stop_event.is_set() and self._queue.empty():
                break
            try:
                pkt = self._queue.get(timeout=0.5)
                yield pkt
            except queue.Empty:
                continue

    def stop(self):
        """Signal the capture to stop."""
        self._stop_event.set()
