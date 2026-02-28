#pragma once
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include "graphos/core/constants.hpp"
#include "graphos/kernel/runtime.hpp"

namespace graphos {

// ── HealthMonitor — daemon thread polling runtime health ──
class HealthMonitor {
    KernelRuntime& runtime_;
    std::chrono::milliseconds interval_;
    std::jthread thread_;
    HealthStatus last_health_;
    mutable std::mutex mu_;

    void loop(std::stop_token st);

public:
    explicit HealthMonitor(KernelRuntime& runtime,
                           double interval_s = 5.0)
        : runtime_(runtime),
          interval_(static_cast<int>(interval_s * 1000)) {}

    void start();
    void stop();

    HealthStatus last_health() const {
        std::lock_guard lock(mu_);
        return last_health_;
    }

    // Latency probe: runs a dummy inference, checks if below threshold
    bool check_latency(const std::string& program_name,
                       double threshold_us = 5000.0);
};

} // namespace graphos
