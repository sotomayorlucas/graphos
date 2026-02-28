#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/types.hpp"
#include "graphos/kernel/compose.hpp"

namespace graphos {

// ── AdapterNode — shape transform between programs ──
// Single-input or dual-input (needs_raw) for concat adapters.
class AdapterNode : public Node {
    TensorAdapter adapter_;
    bool needs_raw_;

public:
    InputPort<RawTensorResult> in{"in"};
    InputPort<TensorItem> raw{"raw"};  // only used if needs_raw
    OutputPort<TensorItem> out{"out"};

    AdapterNode(std::string name, TensorAdapter adapter,
                bool needs_raw = false)
        : Node(std::move(name)),
          adapter_(std::move(adapter)),
          needs_raw_(needs_raw) {}

    bool needs_raw() const noexcept { return needs_raw_; }

    void process(std::stop_token st) override;
};

} // namespace graphos
