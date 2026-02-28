#include "graphos/kernel/program.hpp"
#include <openvino/openvino.hpp>

namespace graphos {

struct Program::Impl {
    ProgramSpec spec;
    ov::CompiledModel compiled;
    ov::InferRequest infer_req;

    Impl(const ProgramSpec& s, ov::CompiledModel&& model)
        : spec(s), compiled(std::move(model)),
          infer_req(compiled.create_infer_request()) {}
};

Program::Program(const ProgramSpec& spec, void* compiled_model_ptr)
    : impl_(std::make_unique<Impl>(
          spec,
          std::move(*static_cast<ov::CompiledModel*>(compiled_model_ptr)))) {}

Program::~Program() = default;
Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;

const std::string& Program::name() const noexcept { return impl_->spec.name; }
const ProgramSpec& Program::spec() const noexcept { return impl_->spec; }

void Program::execute(const float* input, float* output, size_t batch_size) {
    auto& spec = impl_->spec;

    // Zero-copy input: wrap caller's buffer as ov::Tensor
    ov::Shape in_shape;
    for (auto d : spec.input_shape) in_shape.push_back(static_cast<size_t>(d));
    in_shape[0] = batch_size;
    ov::Tensor in_tensor(ov::element::f32, in_shape,
                         const_cast<float*>(input));

    // Zero-copy output: wrap caller's buffer as ov::Tensor
    ov::Shape out_shape;
    for (auto d : spec.output_shape) out_shape.push_back(static_cast<size_t>(d));
    out_shape[0] = batch_size;
    ov::Tensor out_tensor(ov::element::f32, out_shape, output);

    impl_->infer_req.set_input_tensor(in_tensor);
    impl_->infer_req.set_output_tensor(out_tensor);
    impl_->infer_req.infer(); // synchronous
}

} // namespace graphos
