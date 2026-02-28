#pragma once
#include "graphos/actions/action.hpp"
#include <array>
#include <atomic>
#include <unordered_map>

namespace graphos {

// ── CountAction — lock-free atomic per-class counters ──
class CountAction : public Action {
    // Fast path: array for known class IDs (0..NUM_CLASSES-1)
    std::array<std::atomic<uint64_t>, NUM_CLASSES + NUM_ROUTES> counts_{};

public:
    void execute(const OwnedPacket& packet, int class_id) override {
        if (class_id >= 0 &&
            class_id < static_cast<int>(counts_.size())) {
            counts_[class_id].fetch_add(1, std::memory_order_relaxed);
        }
    }

    uint64_t count(int class_id) const noexcept {
        if (class_id >= 0 &&
            class_id < static_cast<int>(counts_.size())) {
            return counts_[class_id].load(std::memory_order_relaxed);
        }
        return 0;
    }

    std::unordered_map<int, uint64_t> summary() const {
        std::unordered_map<int, uint64_t> result;
        for (int i = 0; i < static_cast<int>(counts_.size()); ++i) {
            auto c = counts_[i].load(std::memory_order_relaxed);
            if (c > 0) result[i] = c;
        }
        return result;
    }
};

} // namespace graphos
