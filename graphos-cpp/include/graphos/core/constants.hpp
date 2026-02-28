#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace graphos {

// Tensor dimensions
inline constexpr int TENSOR_DIM = 64;
inline constexpr int NUM_CLASSES = 3;
inline constexpr int NUM_ROUTES = 4;
inline constexpr int DEFAULT_BATCH_SIZE = 64;
inline constexpr int HIDDEN_DIM = 32;

// Class IDs
inline constexpr int CLASS_HTTP = 0;
inline constexpr int CLASS_DNS = 1;
inline constexpr int CLASS_OTHER = 2;

// Route IDs
inline constexpr int ROUTE_LOCAL = 0;
inline constexpr int ROUTE_FORWARD = 1;
inline constexpr int ROUTE_DROP = 2;
inline constexpr int ROUTE_MONITOR = 3;

// Class and route name lookups
inline const std::unordered_map<int, std::string> CLASS_NAMES = {
    {CLASS_HTTP, "TCP_HTTP"},
    {CLASS_DNS, "UDP_DNS"},
    {CLASS_OTHER, "OTHER"}
};

inline const std::unordered_map<int, std::string> ROUTE_NAMES = {
    {ROUTE_LOCAL, "LOCAL"},
    {ROUTE_FORWARD, "FORWARD"},
    {ROUTE_DROP, "DROP"},
    {ROUTE_MONITOR, "MONITOR"}
};

// Ethernet/IP/TCP byte offsets into the 64-byte window
inline constexpr int OFFSET_DST_MAC = 0;
inline constexpr int OFFSET_SRC_MAC = 6;
inline constexpr int OFFSET_ETHERTYPE = 12;
inline constexpr int OFFSET_IP_VERSION = 14;
inline constexpr int OFFSET_IP_DSCP = 15;
inline constexpr int OFFSET_IP_TOTAL_LEN = 16;
inline constexpr int OFFSET_IP_ID = 18;
inline constexpr int OFFSET_IP_FLAGS = 20;
inline constexpr int OFFSET_TTL = 22;
inline constexpr int OFFSET_PROTOCOL = 23;
inline constexpr int OFFSET_IP_CHECKSUM = 24;
inline constexpr int OFFSET_SRC_IP = 26;
inline constexpr int OFFSET_DST_IP = 30;
inline constexpr int OFFSET_SRC_PORT = 34;
inline constexpr int OFFSET_DST_PORT = 36;
inline constexpr int OFFSET_TCP_SEQ = 38;
inline constexpr int OFFSET_PAYLOAD = 54;

// Protocol numbers
inline constexpr uint8_t PROTO_TCP = 6;
inline constexpr uint8_t PROTO_UDP = 17;
inline constexpr uint8_t PROTO_ICMP = 1;

// Well-known ports
inline const std::unordered_set<uint16_t> HTTP_PORTS = {80, 443, 8080, 8443};
inline constexpr uint16_t DNS_PORT = 53;

// Composed router input dimension (raw packet + classifier logits)
inline constexpr int COMPOSED_INPUT_DIM = TENSOR_DIM + NUM_CLASSES; // 67

} // namespace graphos
