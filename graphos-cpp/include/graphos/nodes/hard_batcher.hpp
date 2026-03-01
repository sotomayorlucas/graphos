#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/decision.hpp"
#include <chrono>

namespace graphos {

// Accumulates HardPathItems into NpuBatchItems with deadline-based flushing.
// Flushes when: (a) batch full, or (b) deadline expired and count >= min_fill.
class HardBatcher : public Node {
    size_t batch_size_;
    size_t min_fill_;
    std::chrono::microseconds deadline_;

public:
    InputPort<HardPathItem> in{"in"};
    OutputPort<NpuBatchItem> out{"out"};

    HardBatcher(std::string name,
                size_t batch_size = DEFAULT_BATCH_SIZE,
                size_t min_fill = 12,
                std::chrono::microseconds deadline = std::chrono::microseconds{150})
        : Node(std::move(name)),
          batch_size_(batch_size),
          min_fill_(min_fill),
          deadline_(deadline) {}

    void process(std::stop_token st) override;
};

} // namespace graphos
