#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/types.hpp"

namespace graphos {

// ── TensorNode — converts BatchItem → TensorItem ──
// Uses AVX2 SIMD when available for ~4x normalization speedup.
class TensorNode : public Node {
    size_t batch_size_;

public:
    InputPort<BatchItem> in{"in"};
    OutputPort<TensorItem> out{"out"};

    TensorNode(std::string name, size_t batch_size = DEFAULT_BATCH_SIZE)
        : Node(std::move(name)), batch_size_(batch_size) {}

    void process(std::stop_token st) override;
};

} // namespace graphos
