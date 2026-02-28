#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/types.hpp"
#include <string>

namespace graphos {

// Forward declare — avoids including OpenVINO headers in public API
class KernelRuntime;

// ── InferNode — TensorItem → ResultItem via KernelRuntime (argmax applied) ──
class InferNode : public Node {
    KernelRuntime& runtime_;
    std::string program_name_;
    size_t batch_size_;

public:
    InputPort<TensorItem> in{"in"};
    OutputPort<ResultItem> out{"out"};

    InferNode(std::string name, KernelRuntime& runtime,
              std::string program_name,
              size_t batch_size = DEFAULT_BATCH_SIZE)
        : Node(std::move(name)), runtime_(runtime),
          program_name_(std::move(program_name)),
          batch_size_(batch_size) {}

    void process(std::stop_token st) override;
};

// ── RawInferNode — TensorItem → RawTensorResult (no argmax, for chaining) ──
class RawInferNode : public Node {
    KernelRuntime& runtime_;
    std::string program_name_;
    size_t batch_size_;
    size_t output_dim_;

public:
    InputPort<TensorItem> in{"in"};
    OutputPort<RawTensorResult> out{"out"};

    RawInferNode(std::string name, KernelRuntime& runtime,
                 std::string program_name, size_t output_dim,
                 size_t batch_size = DEFAULT_BATCH_SIZE)
        : Node(std::move(name)), runtime_(runtime),
          program_name_(std::move(program_name)),
          batch_size_(batch_size), output_dim_(output_dim) {}

    void process(std::stop_token st) override;
};

} // namespace graphos
