#pragma once
#include <memory>
#include <string>
#include <vector>

namespace graphos {

// ── ProgramSpec — immutable descriptor for an ONNX program ──
struct ProgramSpec {
    std::string name;
    std::string onnx_path;
    std::vector<int> input_shape;  // e.g., {64, 64} for batch=64, dim=64
    std::vector<int> output_shape; // e.g., {64, 3} for batch=64, classes=3
    std::string description;
};

// ── Program — compiled ONNX model (pimpl hides OpenVINO headers) ──
class Program {
public:
    Program(const ProgramSpec& spec, void* compiled_model_ptr);
    ~Program();

    Program(Program&&) noexcept;
    Program& operator=(Program&&) noexcept;

    const std::string& name() const noexcept;
    const ProgramSpec& spec() const noexcept;

    // Execute inference: input/output are raw float buffers
    // Zero-copy: wraps pointers as ov::Tensor (no memcpy)
    void execute(const float* input, float* output, size_t batch_size);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace graphos
