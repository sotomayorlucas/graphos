"""C-style switch baseline for packet classification."""

from core.constants import (
    CLASS_HTTP, CLASS_DNS, CLASS_OTHER,
    PROTO_TCP, PROTO_UDP, HTTP_PORTS, DNS_PORT,
    CLASS_NAMES,
)


def classify_packet_switch(raw_bytes: bytes) -> int:
    """Traditional C-style switch classification using byte inspection.

    Args:
        raw_bytes: Raw packet bytes.

    Returns:
        Integer class index (0=TCP_HTTP, 1=UDP_DNS, 2=OTHER).
    """
    if len(raw_bytes) < 34:
        return CLASS_OTHER

    protocol = raw_bytes[23]

    if protocol == PROTO_TCP and len(raw_bytes) >= 38:
        dst_port = (raw_bytes[36] << 8) | raw_bytes[37]
        if dst_port in HTTP_PORTS:
            return CLASS_HTTP
    elif protocol == PROTO_UDP and len(raw_bytes) >= 38:
        dst_port = (raw_bytes[36] << 8) | raw_bytes[37]
        if dst_port == DNS_PORT:
            return CLASS_DNS

    return CLASS_OTHER


def classify_packet_switch_label(raw_bytes: bytes) -> str:
    """Classify and return human-readable label."""
    return CLASS_NAMES[classify_packet_switch(raw_bytes)]
