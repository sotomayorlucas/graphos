#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/types.hpp"
#include "graphos/actions/action.hpp"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace graphos {

// ── RouterSink — multi-input terminal: dispatches packets to Actions by class ──
class RouterSink : public Node {
    std::unordered_map<int, std::vector<std::shared_ptr<Action>>> actions_;
    std::shared_ptr<Action> default_action_;
    std::vector<int> results_;
    mutable std::mutex mu_;

    void close_actions() noexcept;

public:
    InputPort<ResultItem> classes{"classes"};
    InputPort<OwnedPacket> packets{"packets"};

    RouterSink(std::string name,
               std::unordered_map<int, std::vector<std::shared_ptr<Action>>> actions,
               std::shared_ptr<Action> default_action = nullptr)
        : Node(std::move(name)),
          actions_(std::move(actions)),
          default_action_(std::move(default_action)) {}

    void process(std::stop_token st) override;

    void teardown() noexcept override { close_actions(); }

    std::vector<int> results() const {
        std::lock_guard lock(mu_);
        return results_;
    }
};

} // namespace graphos
