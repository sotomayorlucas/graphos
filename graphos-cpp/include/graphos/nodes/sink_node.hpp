#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/types.hpp"
#include <mutex>
#include <vector>

namespace graphos {

// ── SinkNode — terminal node collecting ResultItems ──
class SinkNode : public Node {
    std::vector<int> results_;
    mutable std::mutex mu_;

public:
    InputPort<ResultItem> in{"in"};

    explicit SinkNode(std::string name) : Node(std::move(name)) {}

    void process(std::stop_token st) override;

    // Thread-safe access to accumulated results
    std::vector<int> results() const {
        std::lock_guard lock(mu_);
        return results_;
    }

    size_t result_count() const {
        std::lock_guard lock(mu_);
        return results_.size();
    }
};

} // namespace graphos
