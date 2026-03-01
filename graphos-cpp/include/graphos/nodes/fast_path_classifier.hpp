#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/decision.hpp"
#include <atomic>

namespace graphos {

// CPU fast-path: classifies "obvious" packets via byte-level rules.
// Packets that don't match any rule go to hard_out for NPU inference.
class FastPathClassifier : public Node {
    std::atomic<uint64_t> fast_count_{0};
    std::atomic<uint64_t> hard_count_{0};
    std::atomic<uint64_t> sequence_{0};

    static constexpr bool is_http_port(uint16_t port) noexcept {
        return port == 80 || port == 443 || port == 8080 || port == 8443;
    }

public:
    InputPort<OwnedPacket> in{"in"};
    OutputPort<Decision> fast_out{"fast_out"};
    OutputPort<HardPathItem> hard_out{"hard_out"};

    explicit FastPathClassifier(std::string name)
        : Node(std::move(name)) {}

    void process(std::stop_token st) override;

    uint64_t fast_count() const noexcept {
        return fast_count_.load(std::memory_order_relaxed);
    }
    uint64_t hard_count() const noexcept {
        return hard_count_.load(std::memory_order_relaxed);
    }
};

} // namespace graphos
