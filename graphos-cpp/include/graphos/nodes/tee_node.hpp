#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/types.hpp"

namespace graphos {

// ── TeeNode — fan-out: duplicates OwnedPackets to two outputs ──
class TeeNode : public Node {
public:
    InputPort<OwnedPacket> in{"in"};
    OutputPort<OwnedPacket> out{"out"};
    OutputPort<OwnedPacket> copy{"copy"};

    explicit TeeNode(std::string name) : Node(std::move(name)) {}

    void process(std::stop_token st) override;
};

} // namespace graphos
