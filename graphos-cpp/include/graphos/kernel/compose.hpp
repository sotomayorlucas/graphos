#pragma once
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "graphos/kernel/runtime.hpp"

namespace graphos {

// ── AdapterSpec — describes a shape transform ──
struct AdapterSpec {
    std::string name;
    std::unordered_map<std::string, std::vector<int>> input_shapes;
    std::vector<int> output_shape;
    std::string description;
};

// ── TensorAdapter — pure function wrapper for shape transforms ──
class TensorAdapter {
    AdapterSpec spec_;
    // Single-input fn: (in_data, in_size) → output
    std::function<std::vector<float>(const float*, size_t)> single_fn_;
    // Dual-input fn: (left, left_size, right, right_size) → output
    std::function<std::vector<float>(const float*, size_t,
                                     const float*, size_t)> concat_fn_;

public:
    TensorAdapter() = default;

    TensorAdapter(AdapterSpec spec,
                  std::function<std::vector<float>(const float*, size_t)> fn)
        : spec_(std::move(spec)), single_fn_(std::move(fn)) {}

    TensorAdapter(AdapterSpec spec,
                  std::function<std::vector<float>(const float*, size_t,
                                                   const float*, size_t)> fn)
        : spec_(std::move(spec)), concat_fn_(std::move(fn)) {}

    const std::string& name() const noexcept { return spec_.name; }
    const AdapterSpec& spec() const noexcept { return spec_; }

    std::vector<float> execute_single(const float* data, size_t size) const {
        return single_fn_(data, size);
    }

    std::vector<float> execute_concat(const float* left, size_t left_size,
                                       const float* right, size_t right_size) const {
        return concat_fn_(left, left_size, right, right_size);
    }
};

// Factory functions
TensorAdapter make_concat_adapter(size_t batch_size,
                                   size_t left_dim, size_t right_dim);
TensorAdapter make_pad_adapter(size_t batch_size,
                                size_t input_dim, size_t output_dim);

// ── ProgramPipeline — linear chain of programs + adapters ──
class ProgramPipeline {
public:
    enum class StageKind { Program, Adapter };

    struct Stage {
        StageKind kind;
        std::string name;
        ProgramSpec program_spec;    // only valid for Program stages
        TensorAdapter adapter;       // only valid for Adapter stages
    };

    ProgramPipeline& add_program(const std::string& name,
                                  const ProgramSpec& spec);
    ProgramPipeline& add_adapter(TensorAdapter adapter);
    ProgramPipeline& with_raw_passthrough(const std::string& input_name = "raw");

    const std::vector<Stage>& stages() const noexcept { return stages_; }
    std::vector<std::string> validate() const;

    // Execute the full pipeline
    std::vector<float> execute(KernelRuntime& runtime,
                                const float* input, size_t input_size,
                                size_t batch_size);

    std::string describe() const;

private:
    std::vector<Stage> stages_;
    std::string raw_passthrough_;
};

} // namespace graphos
