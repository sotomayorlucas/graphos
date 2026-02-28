#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/types.hpp"

namespace graphos {

// ── BatchNode — accumulates OwnedPackets into fixed-size BatchItems ──
// Zero heap allocation: BatchItem uses std::array<OwnedPacket, 64>.
class BatchNode : public Node {
    size_t batch_size_;

public:
    InputPort<OwnedPacket> in{"in"};
    OutputPort<BatchItem> out{"out"};

    BatchNode(std::string name, size_t batch_size = DEFAULT_BATCH_SIZE)
        : Node(std::move(name)), batch_size_(batch_size) {}

    void process(std::stop_token st) override;
};

} // namespace graphos
