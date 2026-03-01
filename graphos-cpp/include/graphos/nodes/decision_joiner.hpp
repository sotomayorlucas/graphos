#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/decision.hpp"
#include <map>

namespace graphos {

// Joins fast-path and hard-path Decisions into a single output stream.
// Polls both inputs with try_get(). Optional ordered mode via reorder buffer.
class DecisionJoiner : public Node {
    bool ordered_;
    uint64_t next_seq_ = 0;
    std::map<uint64_t, Decision> reorder_buf_;

public:
    InputPort<Decision> fast_in{"fast_in"};
    InputPort<Decision> hard_in{"hard_in"};
    OutputPort<Decision> out{"out"};

    explicit DecisionJoiner(std::string name, bool ordered = false)
        : Node(std::move(name)), ordered_(ordered) {}

    void process(std::stop_token st) override;
};

} // namespace graphos
