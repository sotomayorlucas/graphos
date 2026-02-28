"""PcapWriteAction — writes packets to per-class pcap files."""

import os

from actions.base import Action
from core.constants import CLASS_NAMES


class PcapWriteAction(Action):
    """Appends each packet to a per-class pcap file using scapy."""

    def __init__(self, output_dir: str = "output"):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def execute(self, packet: bytes, class_id: int) -> None:
        from scapy.layers.l2 import Ether
        from scapy.utils import wrpcap

        label = CLASS_NAMES.get(class_id, f"CLASS_{class_id}")
        path = os.path.join(self._output_dir, f"{label}.pcap")
        wrpcap(path, Ether(packet), append=True)
