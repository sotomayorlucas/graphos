#include "graphos/nodes/tee_node.hpp"

namespace graphos {

void TeeNode::process(std::stop_token st) {
    while (!st.stop_requested()) {
        auto item = in.get();
        if (!item.has_value()) {
            out.close();
            copy.close();
            return;
        }

        // Duplicate: copy the packet for the second output
        OwnedPacket dup = *item;
        out.put(std::move(*item));
        copy.put(std::move(dup));
        items_processed_.fetch_add(1, std::memory_order_relaxed);
    }
    out.close();
    copy.close();
}

} // namespace graphos
