#pragma once
#include "graphos/core/constants.hpp"
#include "graphos/core/types.hpp"
#include <array>
#include <cstdint>

namespace graphos {

enum class PathKind : uint8_t { FAST_PATH = 0, HARD_PATH = 1 };

struct Decision {
    OwnedPacket packet;
    int class_id = -1;
    int route_id = -1;
    PathKind path = PathKind::FAST_PATH;
    float confidence = 0.0f;
    uint64_t sequence = 0;
    uint64_t ingress_ns = 0;
};

struct HardPathItem {
    OwnedPacket packet;
    uint64_t sequence = 0;
    uint64_t ingress_ns = 0;
};

struct NpuBatchItem {
    std::array<HardPathItem, DEFAULT_BATCH_SIZE> items{};
    size_t count = 0;
    size_t padded_to = 0;
};

} // namespace graphos
