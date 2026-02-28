#include "graphos/kernel/health.hpp"
#include <condition_variable>

namespace graphos {

void HealthMonitor::start() {
    thread_ = std::jthread([this](std::stop_token st) { loop(st); });
}

void HealthMonitor::stop() {
    if (thread_.joinable()) {
        thread_.request_stop();
        thread_.join();
    }
}

void HealthMonitor::loop(std::stop_token st) {
    std::mutex sleep_mu;
    std::condition_variable_any cv;

    // Register callback: when stop is requested, notify the CV immediately
    std::stop_callback on_stop(st, [&cv]() { cv.notify_all(); });

    while (!st.stop_requested()) {
        auto h = runtime_.health();
        {
            std::lock_guard lock(mu_);
            last_health_ = std::move(h);
        }
        // Interruptible sleep — wakes immediately on stop_requested
        std::unique_lock sleep_lock(sleep_mu);
        cv.wait_for(sleep_lock, interval_, [&st]{ return st.stop_requested(); });
    }
}

bool HealthMonitor::check_latency(const std::string& program_name,
                                   double threshold_us) {
    std::vector<float> dummy(DEFAULT_BATCH_SIZE * TENSOR_DIM, 0.0f);
    std::vector<float> output(DEFAULT_BATCH_SIZE * NUM_ROUTES, 0.0f);

    runtime_.execute(program_name, dummy.data(), output.data(),
                     DEFAULT_BATCH_SIZE);

    auto h = runtime_.health();
    return h.last_latency_us < threshold_us;
}

} // namespace graphos
