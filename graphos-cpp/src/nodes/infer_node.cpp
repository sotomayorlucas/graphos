#include "graphos/nodes/infer_node.hpp"
#include "graphos/kernel/runtime.hpp"

namespace graphos {

void InferNode::process(std::stop_token st) {
    // Pre-allocate output buffer (reused across batches — zero allocation in loop)
    std::vector<float> output_buf(batch_size_ * NUM_CLASSES);

    while (!st.stop_requested()) {
        auto item = in.get();
        if (!item.has_value()) {
            out.close();
            return;
        }

        auto& tensor = *item;

        // Zero-copy execute: passes raw float* to OpenVINO
        runtime_.execute(program_name_,
                         tensor.data.data(), output_buf.data(),
                         batch_size_);

        // Argmax into ResultItem
        ResultItem result;
        result.count = tensor.count;
        batch_tensor_to_classes(output_buf.data(), tensor.count,
                                NUM_CLASSES, result.class_ids.data());

        items_processed_.fetch_add(tensor.count, std::memory_order_relaxed);
        out.put(std::move(result));
    }
    out.close();
}

void RawInferNode::process(std::stop_token st) {
    // Pre-allocate output buffer
    std::vector<float> output_buf(batch_size_ * output_dim_);

    while (!st.stop_requested()) {
        auto item = in.get();
        if (!item.has_value()) {
            out.close();
            return;
        }

        auto& tensor = *item;

        runtime_.execute(program_name_,
                         tensor.data.data(), output_buf.data(),
                         batch_size_);

        // Pass raw logits through — no argmax
        RawTensorResult result(batch_size_, output_dim_);
        result.count = tensor.count;
        result.data = output_buf; // copy (output_buf is reused)

        items_processed_.fetch_add(tensor.count, std::memory_order_relaxed);
        out.put(std::move(result));
    }
    out.close();
}

} // namespace graphos
