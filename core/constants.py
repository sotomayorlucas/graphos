"""GraphOS tensor layout constants."""

# Tensor dimensions
TENSOR_DIM = 64
NUM_CLASSES = 3

# Class labels
CLASS_HTTP = 0
CLASS_DNS = 1
CLASS_OTHER = 2

CLASS_NAMES = {
    CLASS_HTTP: "TCP_HTTP",
    CLASS_DNS: "UDP_DNS",
    CLASS_OTHER: "OTHER",
}

# Key byte offsets in the 64-byte packet tensor
OFFSET_DST_MAC = 0       # 6 bytes
OFFSET_SRC_MAC = 6       # 6 bytes
OFFSET_ETHERTYPE = 12    # 2 bytes
OFFSET_IP_VERSION = 14   # 1 byte
OFFSET_IP_DSCP = 15      # 1 byte
OFFSET_IP_TOTAL_LEN = 16 # 2 bytes
OFFSET_IP_ID = 18        # 2 bytes
OFFSET_IP_FLAGS = 20     # 2 bytes
OFFSET_TTL = 22          # 1 byte
OFFSET_PROTOCOL = 23     # 1 byte (6=TCP, 17=UDP)
OFFSET_IP_CHECKSUM = 24  # 2 bytes
OFFSET_SRC_IP = 26       # 4 bytes
OFFSET_DST_IP = 30       # 4 bytes
OFFSET_SRC_PORT = 34     # 2 bytes
OFFSET_DST_PORT = 36     # 2 bytes (TCP) or 34 (UDP — same as src_port offset for TCP)
OFFSET_TCP_SEQ = 38      # 16 bytes (TCP: Seq/Ack/Flags) or UDP: Len/Checksum+pad
OFFSET_PAYLOAD = 54      # 10 bytes

# Protocol numbers
PROTO_TCP = 6
PROTO_UDP = 17
PROTO_ICMP = 1

# HTTP ports
HTTP_PORTS = {80, 443, 8080, 8443}
DNS_PORT = 53

# Batching
DEFAULT_BATCH_SIZE = 64

# Model architecture
HIDDEN_DIM = 32
