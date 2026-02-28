#include "graphos/nodes/tensor_node.hpp"

namespace graphos {

void TensorNode::process(std::stop_token st) {
    while (!st.stop_requested()) {
        auto item = in.get();
        if (!item.has_value()) {
            out.close();
            return;
        }

        auto& batch = *item;
        TensorItem tensor(batch_size_);
        tensor.count = batch.count;

        // Convert raw bytes → normalized floats (uses AVX2 if available)
        packets_to_batch_tensor(batch.packets.data(), batch.count,
                                batch_size_, tensor.data.data());

        items_processed_.fetch_add(batch.count, std::memory_order_relaxed);
        out.put(std::move(tensor));
    }
    out.close();
}

} // namespace graphos
