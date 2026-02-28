#include "graphos/nodes/batch_node.hpp"

namespace graphos {

void BatchNode::process(std::stop_token st) {
    BatchItem batch;
    batch.count = 0;

    while (!st.stop_requested()) {
        auto item = in.get();
        if (!item.has_value()) {
            // Channel closed — flush partial batch
            if (batch.count > 0) {
                out.put(std::move(batch));
            }
            out.close();
            return;
        }

        batch.packets[batch.count] = std::move(*item);
        batch.count++;
        items_processed_.fetch_add(1, std::memory_order_relaxed);

        if (batch.count >= batch_size_) {
            out.put(std::move(batch));
            batch = BatchItem{};
            batch.count = 0;
        }
    }

    // Stop requested — flush and close
    if (batch.count > 0) out.put(std::move(batch));
    out.close();
}

} // namespace graphos
