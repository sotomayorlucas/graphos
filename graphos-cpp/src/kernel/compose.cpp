#include "graphos/kernel/compose.hpp"
#include <cstring>
#include <sstream>

namespace graphos {

TensorAdapter make_concat_adapter(size_t batch_size,
                                   size_t left_dim, size_t right_dim) {
    AdapterSpec spec;
    spec.name = "concat";
    spec.input_shapes["left"] = {static_cast<int>(batch_size),
                                  static_cast<int>(left_dim)};
    spec.input_shapes["right"] = {static_cast<int>(batch_size),
                                   static_cast<int>(right_dim)};
    spec.output_shape = {static_cast<int>(batch_size),
                         static_cast<int>(left_dim + right_dim)};
    spec.description = "Concatenate left + right along dim=1";

    size_t out_dim = left_dim + right_dim;

    return TensorAdapter(std::move(spec),
        [batch_size, left_dim, right_dim, out_dim](
                const float* left, size_t /*left_size*/,
                const float* right, size_t /*right_size*/) -> std::vector<float> {
            std::vector<float> result(batch_size * out_dim);
            for (size_t row = 0; row < batch_size; ++row) {
                float* dst = result.data() + row * out_dim;
                // Copy left (raw packet)
                std::memcpy(dst, left + row * left_dim,
                           left_dim * sizeof(float));
                // Copy right (logits)
                std::memcpy(dst + left_dim, right + row * right_dim,
                           right_dim * sizeof(float));
            }
            return result;
        });
}

TensorAdapter make_pad_adapter(size_t batch_size,
                                size_t input_dim, size_t output_dim) {
    AdapterSpec spec;
    spec.name = "pad";
    spec.input_shapes["in"] = {static_cast<int>(batch_size),
                                static_cast<int>(input_dim)};
    spec.output_shape = {static_cast<int>(batch_size),
                         static_cast<int>(output_dim)};
    spec.description = output_dim > input_dim ? "Zero-pad" : "Truncate";

    return TensorAdapter(std::move(spec),
        [batch_size, input_dim, output_dim](
                const float* data, size_t /*size*/) -> std::vector<float> {
            std::vector<float> result(batch_size * output_dim, 0.0f);
            size_t copy_dim = std::min(input_dim, output_dim);
            for (size_t row = 0; row < batch_size; ++row) {
                std::memcpy(result.data() + row * output_dim,
                           data + row * input_dim,
                           copy_dim * sizeof(float));
            }
            return result;
        });
}

ProgramPipeline& ProgramPipeline::add_program(const std::string& name,
                                                const ProgramSpec& spec) {
    stages_.push_back({StageKind::Program, name, spec, {}});
    return *this;
}

ProgramPipeline& ProgramPipeline::add_adapter(TensorAdapter adapter) {
    stages_.push_back({StageKind::Adapter, adapter.name(), {}, std::move(adapter)});
    return *this;
}

ProgramPipeline& ProgramPipeline::with_raw_passthrough(const std::string& input_name) {
    raw_passthrough_ = input_name;
    return *this;
}

std::vector<std::string> ProgramPipeline::validate() const {
    std::vector<std::string> errors;
    if (stages_.empty()) {
        errors.push_back("Pipeline has no stages");
    }
    return errors;
}

std::vector<float> ProgramPipeline::execute(KernelRuntime& runtime,
                                             const float* input,
                                             size_t input_size,
                                             size_t batch_size) {
    std::vector<float> current(input, input + input_size);
    std::vector<float> raw_copy;
    if (!raw_passthrough_.empty()) {
        raw_copy.assign(input, input + input_size);
    }

    // Pre-allocate output buffer for program stages
    std::vector<float> output_buf;

    for (auto& stage : stages_) {
        if (stage.kind == StageKind::Program) {
            auto& spec = stage.program_spec;
            size_t out_size = 1;
            for (auto d : spec.output_shape) out_size *= d;
            // Adjust for actual batch size
            out_size = (out_size / spec.output_shape[0]) * batch_size;

            output_buf.resize(out_size);
            runtime.execute(stage.name, current.data(), output_buf.data(),
                           batch_size);
            current = std::move(output_buf);
            output_buf = {};
        } else {
            // Adapter stage
            auto& adapter = stage.adapter;
            auto& keys = adapter.spec().input_shapes;

            if (keys.count("left") && keys.count("right")) {
                // Dual-input concat
                current = adapter.execute_concat(
                    raw_copy.data(), raw_copy.size(),
                    current.data(), current.size());
            } else {
                // Single-input
                current = adapter.execute_single(
                    current.data(), current.size());
            }
        }
    }

    return current;
}

std::string ProgramPipeline::describe() const {
    std::ostringstream ss;
    for (size_t i = 0; i < stages_.size(); ++i) {
        if (i > 0) ss << " -> ";
        auto& s = stages_[i];
        if (s.kind == StageKind::Program) {
            ss << "[prog:" << s.name << "]";
        } else {
            ss << "<" << s.name << ">";
        }
    }
    if (!raw_passthrough_.empty()) {
        ss << " (raw passthrough: '" << raw_passthrough_ << "')";
    }
    return ss.str();
}

} // namespace graphos
