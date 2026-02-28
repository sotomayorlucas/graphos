#pragma once
#include <optional>
#include "graphos/core/types.hpp"

namespace graphos {

// ── CaptureSource interface — zero-overhead via CRTP ──
// Implementations: PcapSource, DpdkSource
// Used as template param in SourceNode<Source> (no vtable on hot path).
class CaptureSource {
public:
    virtual ~CaptureSource() = default;
    virtual std::optional<OwnedPacket> next() = 0;
    virtual void stop() = 0;
};

} // namespace graphos
