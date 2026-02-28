#include "graphos/kernel/runtime.hpp"
#include <openvino/openvino.hpp>
#include <chrono>
#include <mutex>
#include <unordered_map>

namespace graphos {

struct KernelRuntime::Impl {
    ov::Core core;
    std::string device;
    std::unordered_map<std::string, Program> programs;
    mutable std::recursive_mutex mu; // reentrant (execute calls get)

    // Metrics
    size_t exec_count = 0;
    double total_exec_us = 0.0;
    double last_exec_us = 0.0;
    size_t errors = 0;

    explicit Impl(const std::string& dev) {
        auto available = core.get_available_devices();
        bool found = false;
        for (const auto& d : available) {
            if (d == dev) { found = true; break; }
        }
        if (found) {
            device = dev;
        } else {
            device = "CPU";
        }
    }
};

KernelRuntime::KernelRuntime(const std::string& device)
    : impl_(std::make_unique<Impl>(device)) {}

KernelRuntime::~KernelRuntime() = default;
KernelRuntime::KernelRuntime(KernelRuntime&&) noexcept = default;
KernelRuntime& KernelRuntime::operator=(KernelRuntime&&) noexcept = default;

void KernelRuntime::load(const ProgramSpec& spec) {
    std::lock_guard lock(impl_->mu);
    if (impl_->programs.count(spec.name))
        throw KernelError("Program '" + spec.name + "' already loaded");

    auto model = impl_->core.read_model(spec.onnx_path);
    auto compiled = impl_->core.compile_model(model, impl_->device);

    // Program takes ownership via void* (pimpl boundary)
    impl_->programs.emplace(spec.name,
                            Program(spec, &compiled));
}

void KernelRuntime::unload(const std::string& name) {
    std::lock_guard lock(impl_->mu);
    auto it = impl_->programs.find(name);
    if (it == impl_->programs.end())
        throw KernelError("Program '" + name + "' not found");
    impl_->programs.erase(it);
}

bool KernelRuntime::has(const std::string& name) const {
    std::lock_guard lock(impl_->mu);
    return impl_->programs.count(name) > 0;
}

std::vector<std::string> KernelRuntime::programs() const {
    std::lock_guard lock(impl_->mu);
    std::vector<std::string> names;
    names.reserve(impl_->programs.size());
    for (const auto& [name, _] : impl_->programs)
        names.push_back(name);
    return names;
}

void KernelRuntime::execute(const std::string& name,
                            const float* input, float* output,
                            size_t batch_size) {
    Program* prog;
    {
        std::lock_guard lock(impl_->mu);
        auto it = impl_->programs.find(name);
        if (it == impl_->programs.end())
            throw KernelError("Program '" + name + "' not found");
        prog = &it->second;
    }

    auto t0 = std::chrono::steady_clock::now();
    try {
        prog->execute(input, output, batch_size);
    } catch (...) {
        std::lock_guard lock(impl_->mu);
        impl_->errors++;
        throw;
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    std::lock_guard lock(impl_->mu);
    impl_->exec_count++;
    impl_->total_exec_us += us;
    impl_->last_exec_us = us;
}

const std::string& KernelRuntime::device() const noexcept {
    return impl_->device;
}

std::vector<std::string> KernelRuntime::available_devices() const {
    return impl_->core.get_available_devices();
}

HealthStatus KernelRuntime::health() const {
    std::lock_guard lock(impl_->mu);
    HealthStatus h;
    h.device = impl_->device;
    for (const auto& [name, _] : impl_->programs)
        h.programs.push_back(name);
    h.exec_count = impl_->exec_count;
    h.mean_latency_us = impl_->exec_count > 0
        ? impl_->total_exec_us / impl_->exec_count : 0.0;
    h.last_latency_us = impl_->last_exec_us;
    h.errors = impl_->errors;
    h.healthy = (impl_->errors == 0);
    return h;
}

} // namespace graphos
