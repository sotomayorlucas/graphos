#pragma once
#include "graphos/actions/action.hpp"
#include <cstdio>
#include <string>

namespace graphos {

// ── LogAction — prints packet summary to stdout ──
class LogAction : public Action {
    bool verbose_;
    size_t count_ = 0;

    static std::string parse_ip(const uint8_t* data, size_t len, int offset);
    static uint16_t parse_port(const uint8_t* data, size_t len, int offset);

public:
    explicit LogAction(bool verbose = false) : verbose_(verbose) {}

    void execute(const OwnedPacket& packet, int class_id) override;
    void close() noexcept override;
};

} // namespace graphos
