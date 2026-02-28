#include "graphos/nodes/sink_node.hpp"

namespace graphos {

void SinkNode::process(std::stop_token st) {
    while (!st.stop_requested()) {
        auto item = in.get();
        if (!item.has_value()) return;

        auto& result = *item;
        {
            std::lock_guard lock(mu_);
            results_.insert(results_.end(),
                           result.class_ids.begin(),
                           result.class_ids.begin() + result.count);
        }
        items_processed_.fetch_add(result.count, std::memory_order_relaxed);
    }
}

} // namespace graphos
