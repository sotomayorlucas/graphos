"""Synthetic packet dataset generator for RouterGraph training."""

import struct
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from core.constants import (
    TENSOR_DIM, CLASS_HTTP, CLASS_DNS, CLASS_OTHER,
    PROTO_TCP, PROTO_UDP, PROTO_ICMP, HTTP_PORTS, DNS_PORT,
)
from core.tensor_layout import packet_to_tensor


def _random_mac():
    return bytes(random.randint(0, 255) for _ in range(6))


def _random_ip():
    return bytes(random.randint(1, 254) for _ in range(4))


def _random_port():
    return random.randint(1024, 65535)


def _build_packet(protocol, src_port, dst_port, payload=None):
    """Build a raw 64-byte packet with the given transport parameters."""
    pkt = bytearray(64)

    # L2: MACs + EtherType (IPv4 = 0x0800)
    pkt[0:6] = _random_mac()
    pkt[6:12] = _random_mac()
    struct.pack_into('!H', pkt, 12, 0x0800)

    # IP header
    pkt[14] = 0x45  # Version 4, IHL 5
    pkt[15] = 0x00  # DSCP/ECN
    struct.pack_into('!H', pkt, 16, 64)  # Total length
    struct.pack_into('!H', pkt, 18, random.randint(0, 65535))  # ID
    struct.pack_into('!H', pkt, 20, 0x4000)  # Don't Fragment
    pkt[22] = random.randint(32, 128)  # TTL
    pkt[23] = protocol
    struct.pack_into('!H', pkt, 24, random.randint(0, 65535))  # Checksum (fake)
    pkt[26:30] = _random_ip()  # Src IP
    pkt[30:34] = _random_ip()  # Dst IP

    # Transport header
    if protocol in (PROTO_TCP, PROTO_UDP):
        struct.pack_into('!H', pkt, 34, src_port)
        struct.pack_into('!H', pkt, 36, dst_port)

    # TCP-specific: seq/ack/flags randomized
    if protocol == PROTO_TCP:
        for i in range(38, 54):
            pkt[i] = random.randint(0, 255)
    elif protocol == PROTO_UDP:
        struct.pack_into('!H', pkt, 38, 20)  # UDP length
        struct.pack_into('!H', pkt, 40, 0)   # UDP checksum
        for i in range(42, 54):
            pkt[i] = random.randint(0, 255)

    # Payload / padding
    if payload:
        pkt[54:54 + len(payload)] = payload[:10]
    else:
        for i in range(54, 64):
            pkt[i] = random.randint(0, 255)

    return bytes(pkt)


def generate_http_packet():
    """Generate a TCP packet with an HTTP destination port."""
    dst_port = random.choice(list(HTTP_PORTS))
    return _build_packet(PROTO_TCP, _random_port(), dst_port)


def generate_dns_packet():
    """Generate a UDP packet with destination port 53."""
    return _build_packet(PROTO_UDP, _random_port(), DNS_PORT)


def generate_other_packet():
    """Generate a non-HTTP, non-DNS packet."""
    kind = random.choice(['icmp', 'tcp_random', 'udp_random', 'short'])
    if kind == 'icmp':
        return _build_packet(PROTO_ICMP, 0, 0)
    elif kind == 'tcp_random':
        # TCP with a non-HTTP port
        dst_port = _random_port()
        while dst_port in HTTP_PORTS:
            dst_port = _random_port()
        return _build_packet(PROTO_TCP, _random_port(), dst_port)
    elif kind == 'udp_random':
        # UDP with a non-DNS port
        dst_port = _random_port()
        while dst_port == DNS_PORT:
            dst_port = _random_port()
        return _build_packet(PROTO_UDP, _random_port(), dst_port)
    else:  # short / malformed
        length = random.randint(14, 40)
        return bytes(random.randint(0, 255) for _ in range(length))


class PacketDataset(Dataset):
    """Synthetic packet dataset for training the RouterGraph."""

    def __init__(self, samples_per_class=50000, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        generators = [
            (generate_http_packet, CLASS_HTTP),
            (generate_dns_packet, CLASS_DNS),
            (generate_other_packet, CLASS_OTHER),
        ]

        tensors = []
        labels = []
        for gen_fn, label in generators:
            for _ in range(samples_per_class):
                raw = gen_fn()
                t = packet_to_tensor(raw)
                tensors.append(t)
                labels.append(label)

        self.data = torch.from_numpy(np.concatenate(tensors, axis=0))
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Shuffle
        perm = torch.randperm(len(self.labels))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
