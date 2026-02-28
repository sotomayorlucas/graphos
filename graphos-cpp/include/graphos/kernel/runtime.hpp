#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "graphos/kernel/program.hpp"

namespace graphos {

class KernelError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct HealthStatus {
    std::string device;
    std::vector<std::string> programs;
    size_t exec_count = 0;
    double mean_latency_us = 0.0;
    double last_latency_us = 0.0;
    size_t errors = 0;
    bool healthy = true;
};

// ── KernelRuntime — shared inference engine (pimpl, thread-safe) ──
// Uses thread_local InferRequest for zero-contention parallel inference.
// NPU→CPU auto-fallback. Metrics collection.
class KernelRuntime {
public:
    explicit KernelRuntime(const std::string& device = "NPU");
    ~KernelRuntime();

    KernelRuntime(KernelRuntime&&) noexcept;
    KernelRuntime& operator=(KernelRuntime&&) noexcept;

    // Program lifecycle
    void load(const ProgramSpec& spec);
    void unload(const std::string& name);
    bool has(const std::string& name) const;
    std::vector<std::string> programs() const;

    // Zero-copy inference: wraps raw float* as ov::Tensor
    void execute(const std::string& name,
                 const float* input, float* output,
                 size_t batch_size);

    // Device info
    const std::string& device() const noexcept;
    std::vector<std::string> available_devices() const;

    // Health/metrics
    HealthStatus health() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace graphos
