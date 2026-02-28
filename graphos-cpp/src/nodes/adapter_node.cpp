#include "graphos/nodes/adapter_node.hpp"

namespace graphos {

void AdapterNode::process(std::stop_token st) {
    while (!st.stop_requested()) {
        auto item = in.get();
        if (!item.has_value()) {
            if (needs_raw_) raw.get(); // consume matching close
            out.close();
            return;
        }

        auto& tensor_result = *item;

        if (needs_raw_) {
            auto raw_item = raw.get();
            if (!raw_item.has_value()) {
                out.close();
                return;
            }

            // Dual-input: concat raw packet tensor with program logits
            auto result = adapter_.execute_concat(
                raw_item->data.data(), raw_item->data.size(),
                tensor_result.data.data(), tensor_result.data.size());

            TensorItem output(result.size() / COMPOSED_INPUT_DIM);
            output.data = std::move(result);
            output.count = tensor_result.count;

            items_processed_.fetch_add(tensor_result.count,
                                       std::memory_order_relaxed);
            out.put(std::move(output));
        } else {
            // Single-input: pass-through or pad
            auto result = adapter_.execute_single(
                tensor_result.data.data(), tensor_result.data.size());

            TensorItem output(result.size() / TENSOR_DIM);
            output.data = std::move(result);
            output.count = tensor_result.count;

            items_processed_.fetch_add(tensor_result.count,
                                       std::memory_order_relaxed);
            out.put(std::move(output));
        }
    }
    out.close();
}

} // namespace graphos
