#include "graphos/nodes/router_sink.hpp"
#include <unordered_set>

namespace graphos {

void RouterSink::process(std::stop_token st) {
    while (!st.stop_requested()) {
        auto batch = classes.get();
        if (!batch.has_value()) {
            // Drain packets channel
            while (auto pkt = packets.get()) { /* discard */ }
            close_actions();
            return;
        }

        auto& result = *batch;
        for (size_t i = 0; i < result.count; ++i) {
            auto pkt = packets.get();
            if (!pkt.has_value()) {
                close_actions();
                return;
            }

            int class_id = result.class_ids[i];
            auto it = actions_.find(class_id);
            if (it != actions_.end()) {
                for (auto& action : it->second) {
                    action->execute(*pkt, class_id);
                }
            } else if (default_action_) {
                default_action_->execute(*pkt, class_id);
            }

            {
                std::lock_guard lock(mu_);
                results_.push_back(class_id);
            }
        }

        items_processed_.fetch_add(result.count, std::memory_order_relaxed);
    }
    close_actions();
}

void RouterSink::close_actions() noexcept {
    // Deduplicate by pointer address (same action may appear in multiple classes)
    std::unordered_set<Action*> seen;
    for (auto& [_, handlers] : actions_) {
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
