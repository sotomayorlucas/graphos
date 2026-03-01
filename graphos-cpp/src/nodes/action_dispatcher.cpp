#include "graphos/nodes/action_dispatcher.hpp"
#include <chrono>
#include <unordered_set>

namespace graphos {

void ActionDispatcher::process(std::stop_token st) {
    while (!st.stop_requested()) {
        auto decision = in.get();
        if (!decision.has_value()) {
            close_actions();
            return;
        }

        auto& d = *decision;

        // Track path counts
        if (d.path == PathKind::FAST_PATH) {
            fast_path_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            hard_path_count_.fetch_add(1, std::memory_order_relaxed);
        }

        // Record latency
        if (d.ingress_ns > 0) {
            auto now = static_cast<uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            double latency_us = static_cast<double>(now - d.ingress_ns) / 1000.0;
            overall_latency_.record(latency_us);
            if (d.path == PathKind::FAST_PATH)
                fast_latency_.record(latency_us);
            else
                hard_latency_.record(latency_us);
        }

        bool dispatched = false;

        // Dispatch by class_id
        if (d.class_id >= 0) {
            auto it = class_actions_.find(d.class_id);
            if (it != class_actions_.end()) {
                for (auto& action : it->second) {
                    action->execute(d.packet, d.class_id);
                }
                dispatched = true;
            }
        }

        // Dispatch by route_id
        if (d.route_id >= 0) {
            auto it = route_actions_.find(d.route_id);
            if (it != route_actions_.end()) {
                for (auto& action : it->second) {
                    action->execute(d.packet, d.route_id);
                }
                dispatched = true;
            }
        }

        if (!dispatched && default_action_) {
            default_action_->execute(d.packet, d.class_id);
        }

        items_processed_.fetch_add(1, std::memory_order_relaxed);
    }
    close_actions();
}

void ActionDispatcher::close_actions() noexcept {
    std::unordered_set<Action*> seen;
    for (auto& [_, handlers] : class_actions_) {
        for (auto& action : handlers) {
            if (seen.insert(action.get()).second) {
                try { action->close(); } catch (...) {}
            }
        }
    }
    for (auto& [_, handlers] : route_actions_) {
        for (auto& action : handlers) {
            if (seen.insert(action.get()).second) {
                try { action->close(); } catch (...) {}
            }
        }
    }
    if (default_action_ && seen.insert(default_action_.get()).second) {
        try { default_action_->close(); } catch (...) {}
    }
}

} // namespace graphos
