#include "graphos/actions/log_action.hpp"
#include "graphos/core/constants.hpp"
#include <cstdio>

namespace graphos {

std::string LogAction::parse_ip(const uint8_t* data, size_t len, int offset) {
    if (static_cast<size_t>(offset + 4) > len)
        return "?.?.?.?";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u",
                  data[offset], data[offset+1],
                  data[offset+2], data[offset+3]);
    return buf;
}

uint16_t LogAction::parse_port(const uint8_t* data, size_t len, int offset) {
    if (static_cast<size_t>(offset + 2) > len)
        return 0;
    return static_cast<uint16_t>((data[offset] << 8) | data[offset+1]);
}

void LogAction::execute(const OwnedPacket& packet, int class_id) {
    count_++;
    if (!verbose_ && count_ > 10) return;

    auto it = CLASS_NAMES.find(class_id);
    const char* label = (it != CLASS_NAMES.end()) ? it->second.c_str() : "UNKNOWN";

    auto src_ip = parse_ip(packet.bytes.data(), packet.length, OFFSET_SRC_IP);
    auto dst_ip = parse_ip(packet.bytes.data(), packet.length, OFFSET_DST_IP);
    auto src_port = parse_port(packet.bytes.data(), packet.length, OFFSET_SRC_PORT);
    auto dst_port = parse_port(packet.bytes.data(), packet.length, OFFSET_DST_PORT);

    std::printf("[%s] %s:%u -> %s:%u (%zu bytes)\n",
                label, src_ip.c_str(), src_port,
                dst_ip.c_str(), dst_port, packet.length);
}

void LogAction::close() noexcept {
    if (!verbose_ && count_ > 10) {
        std::printf("... (%zu more packets not shown)\n", count_ - 10);
    }
    std::printf("LogAction: %zu packets total\n", count_);
}

} // namespace graphos
