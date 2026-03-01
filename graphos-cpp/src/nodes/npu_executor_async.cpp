#include "graphos/nodes/npu_executor_async.hpp"
#include "graphos/core/constants.hpp"
#include "graphos/core/types.hpp"
#include <openvino/openvino.hpp>
#include <vector>

namespace graphos {

struct NpuExecutorAsync::Impl {
    struct Slot {
        std::vector<float> input_buf;
        std::vector<float> output_buf;
        ov::InferRequest request;
        NpuBatchItem batch_context;
        bool pending = false;

        Slot(ov::CompiledModel& model, size_t batch_size)
            : input_buf(batch_size * TENSOR_DIM, 0.0f),
              output_buf(batch_size * NUM_CLASSES, 0.0f),
              request(model.create_infer_request()) {}
    };

    ov::Core core;
    ov::CompiledModel compiled;
    std::vector<Slot> slots;
    size_t submit_idx = 0;
    size_t collect_idx = 0;

    Impl(const std::string& onnx_path, const std::string& device,
         size_t batch_size, size_t num_inflight)
        : compiled(core.compile_model(
              core.read_model(onnx_path), device)) {
        slots.reserve(num_inflight);
        for (size_t i = 0; i < num_inflight; ++i) {
            slots.emplace_back(compiled, batch_size);
        }
    }
};

NpuExecutorAsync::NpuExecutorAsync(
    std::string name, const std::string& onnx_path,
    const std::string& device, size_t batch_size, size_t num_inflight)
    : Node(std::move(name)),
      impl_(std::make_unique<Impl>(onnx_path, device, batch_size, num_inflight)),
      batch_size_(batch_size),
      num_inflight_(num_inflight) {}

NpuExecutorAsync::~NpuExecutorAsync() = default;

void NpuExecutorAsync::process(std::stop_token st) {
    size_t inflight = 0;

    auto submit_batch = [&](NpuBatchItem&& batch) {
        auto& slot = impl_->slots[impl_->submit_idx % num_inflight_];

        // Normalize packets into slot's input buffer
        for (size_t i = 0; i < batch.count; ++i) {
            packet_to_tensor(batch.items[i].packet,
                             slot.input_buf.data() + i * TENSOR_DIM);
        }
        // Zero-pad remaining slots for static ONNX shape
        for (size_t i = batch.count; i < batch_size_; ++i) {
            std::fill_n(slot.input_buf.data() + i * TENSOR_DIM, TENSOR_DIM, 0.0f);
        }

        // Zero-copy wrap
        ov::Shape in_shape = {batch_size_, static_cast<size_t>(TENSOR_DIM)};
        ov::Shape out_shape = {batch_size_, static_cast<size_t>(NUM_CLASSES)};
        ov::Tensor in_tensor(ov::element::f32, in_shape, slot.input_buf.data());
        ov::Tensor out_tensor(ov::element::f32, out_shape, slot.output_buf.data());

        slot.request.set_input_tensor(in_tensor);
        slot.request.set_output_tensor(out_tensor);
        slot.batch_context = std::move(batch);
        slot.pending = true;
        slot.request.start_async();

        impl_->submit_idx++;
        inflight++;
    };

    auto collect_oldest = [&]() {
        auto& slot = impl_->slots[impl_->collect_idx % num_inflight_];
        slot.request.wait();

        auto& batch = slot.batch_context;
        for (size_t i = 0; i < batch.count; ++i) {
            const float* logits = slot.output_buf.data() + i * NUM_CLASSES;
            int class_id = tensor_to_class(logits, NUM_CLASSES);

            // Find max logit for confidence
            float max_logit = logits[0];
            for (int c = 1; c < NUM_CLASSES; ++c) {
                if (logits[c] > max_logit) max_logit = logits[c];
            }

            Decision d;
            d.packet = std::move(batch.items[i].packet);
            d.class_id = class_id;
            d.route_id = -1; // NPU doesn't produce route decisions
            d.path = PathKind::HARD_PATH;
            d.confidence = max_logit;
            d.sequence = batch.items[i].sequence;
            d.ingress_ns = batch.items[i].ingress_ns;
            out.put(std::move(d));
        }

        items_processed_.fetch_add(batch.count, std::memory_order_relaxed);
        slot.pending = false;
        impl_->collect_idx++;
        inflight--;
    };

    while (!st.stop_requested()) {
        // Collect if all slots full
        if (inflight >= num_inflight_) {
            collect_oldest();
        }

        auto batch = in.get();
        if (!batch.has_value()) break;

        submit_batch(std::move(*batch));
    }

    // Drain all pending slots
    while (inflight > 0) {
        collect_oldest();
    }

    out.close();
}

} // namespace graphos
