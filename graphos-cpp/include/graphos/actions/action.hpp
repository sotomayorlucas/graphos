#pragma once
#include "graphos/core/types.hpp"

namespace graphos {

// ── Action ABC — executed per-packet by RouterSink ──
class Action {
public:
    virtual ~Action() = default;
    virtual void execute(const OwnedPacket& packet, int class_id) = 0;
    virtual void close() noexcept {}
};

} // namespace graphos
