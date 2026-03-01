#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/decision.hpp"
#include <memory>
#include <string>

namespace graphos {

// Async NPU inference with triple-buffering (K pre-allocated slots).
// Pimpl hides OpenVINO headers — compiles its own model for lock-free async.
class NpuExecutorAsync : public Node {
public:
    InputPort<NpuBatchItem> in{"in"};
    OutputPort<Decision> out{"out"};

    NpuExecutorAsync(std::string name,
                     const std::string& onnx_path,
                     const std::string& device = "NPU",
                     size_t batch_size = DEFAULT_BATCH_SIZE,
                     size_t num_inflight = 3);
    ~NpuExecutorAsync();

    void process(std::stop_token st) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    size_t batch_size_;
    size_t num_inflight_;
};

} // namespace graphos
