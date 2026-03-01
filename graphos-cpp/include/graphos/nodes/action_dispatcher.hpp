#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/decision.hpp"
#include "graphos/core/latency_histogram.hpp"
#include "graphos/actions/action.hpp"
#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>

namespace graphos {

// Terminal node: dispatches Decisions to Actions by class_id and route_id.
class ActionDispatcher : public Node {
    std::unordered_map<int, std::vector<std::shared_ptr<Action>>> class_actions_;
    std::unordered_map<int, std::vector<std::shared_ptr<Action>>> route_actions_;
    std::shared_ptr<Action> default_action_;
    std::atomic<uint64_t> fast_path_count_{0};
    std::atomic<uint64_t> hard_path_count_{0};
    LatencyHistogram overall_latency_;
    LatencyHistogram fast_latency_;
    LatencyHistogram hard_latency_;

    void close_actions() noexcept;

public:
    InputPort<Decision> in{"in"};

    ActionDispatcher(
        std::string name,
        std::unordered_map<int, std::vector<std::shared_ptr<Action>>> class_actions = {},
        std::unordered_map<int, std::vector<std::shared_ptr<Action>>> route_actions = {},
        std::shared_ptr<Action> default_action = nullptr)
        : Node(std::move(name)),
          class_actions_(std::move(class_actions)),
          route_actions_(std::move(route_actions)),
          default_action_(std::move(default_action)) {}

    void process(std::stop_token st) override;
    void teardown() noexcept override { close_actions(); }

    uint64_t fast_path_count() const noexcept {
        return fast_path_count_.load(std::memory_order_relaxed);
    }
    uint64_t hard_path_count() const noexcept {
        return hard_path_count_.load(std::memory_order_relaxed);
    }
    const LatencyHistogram& overall_latency() const noexcept { return overall_latency_; }
    const LatencyHistogram& fast_latency() const noexcept { return fast_latency_; }
    const LatencyHistogram& hard_latency() const noexcept { return hard_latency_; }
};

} // namespace graphos
