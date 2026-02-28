"""LogAction — parses and prints one-line packet summaries."""

import struct

from actions.base import Action
from core.constants import (
    CLASS_NAMES,
    OFFSET_PROTOCOL,
    OFFSET_SRC_PORT,
    OFFSET_DST_PORT,
)


# IP header: src IP at bytes 26-29, dst IP at bytes 30-33 (standard Ethernet+IP)
_OFFSET_SRC_IP = 26
_OFFSET_DST_IP = 30


def _parse_ip(raw: bytes, offset: int) -> str:
    """Extract dotted-quad IP from raw packet bytes."""
    if len(raw) < offset + 4:
        return "?.?.?.?"
    return ".".join(str(b) for b in raw[offset : offset + 4])


def _parse_port(raw: bytes, offset: int) -> int:
    """Extract big-endian 16-bit port from raw packet bytes."""
    if len(raw) < offset + 2:
        return 0
    return struct.unpack("!H", raw[offset : offset + 2])[0]


class LogAction(Action):
    """Logs one-line packet summaries to stdout."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._count = 0

    def execute(self, packet: bytes, class_id: int) -> None:
        self._count += 1
        if not self.verbose and self._count > 10:
            return

        label = CLASS_NAMES.get(class_id, f"CLASS_{class_id}")
        src_ip = _parse_ip(packet, _OFFSET_SRC_IP)
        dst_ip = _parse_ip(packet, _OFFSET_DST_IP)
        src_port = _parse_port(packet, OFFSET_SRC_PORT)
        dst_port = _parse_port(packet, OFFSET_DST_PORT)

        print(
            f"[{label}] {src_ip}:{src_port} -> {dst_ip}:{dst_port} "
            f"({len(packet)} bytes)"
        )

    def close(self) -> None:
        if not self.verbose and self._count > 10:
            print(f"... ({self._count - 10} more packets not shown)")
        print(f"LogAction: {self._count} packets total")
