#pragma once
#include <atomic>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include "graphos/kernel/runtime.hpp"
#include "graphos/kernel/health.hpp"
#include "graphos/core/types.hpp"

namespace graphos {

struct LoopStats {
    size_t packets_processed = 0;
    size_t batches_processed = 0;
    double elapsed_s = 0.0;
    double throughput = 0.0; // packets/sec
    HealthStatus health;
};

// ── KernelLoop — batch processor with signal handling ──
class KernelLoop {
    KernelRuntime& runtime_;
    size_t batch_size_;
    HealthMonitor health_monitor_;
    std::atomic<bool> running_{false};
    std::atomic<size_t> packets_processed_{0};
    std::atomic<size_t> batches_processed_{0};
    std::chrono::steady_clock::time_point start_time_;

    void execute_batch(const std::vector<OwnedPacket>& packets,
                       const std::vector<std::string>& programs,
                       std::vector<float>& tensor_buf,
                       std::vector<float>& output_buf);

public:
    KernelLoop(KernelRuntime& runtime,
               size_t batch_size = DEFAULT_BATCH_SIZE,
               double health_interval = 5.0)
        : runtime_(runtime), batch_size_(batch_size),
          health_monitor_(runtime, health_interval) {}

    LoopStats stats() const;

    // Process a single batch (for external use / testing)
    std::unordered_map<std::string, std::vector<float>>
    process_batch(const std::vector<OwnedPacket>& packets);

    // Main loop: iterates a packet source
    template <typename Source>
    void run(Source& source,
             const std::vector<std::string>& programs = {}) {
        running_.store(true, std::memory_order_release);
        start_time_ = std::chrono::steady_clock::now();
        health_monitor_.start();

        // Pre-allocate buffers (reused across batches — zero allocation in loop)
        std::vector<float> tensor_buf(batch_size_ * TENSOR_DIM);
        std::vector<float> output_buf(batch_size_ * NUM_ROUTES); // max output dim

        auto prog_list = programs.empty() ? runtime_.programs() : programs;
        std::vector<OwnedPacket> batch;
        batch.reserve(batch_size_);

        try {
            while (running_.load(std::memory_order_acquire)) {
                auto pkt = source.next();
                if (!pkt.has_value()) break;

                batch.push_back(std::move(*pkt));

                if (batch.size() >= batch_size_) {
                    execute_batch(batch, prog_list, tensor_buf, output_buf);
                    batch.clear();
                }
            }
            // Flush partial batch
            if (!batch.empty()) {
                execute_batch(batch, prog_list, tensor_buf, output_buf);
            }
        } catch (...) {
            running_.store(false, std::memory_order_release);
            health_monitor_.stop();
            throw;
        }

        running_.store(false, std::memory_order_release);
        health_monitor_.stop();
    }

    void stop() noexcept {
        running_.store(false, std::memory_order_release);
    }

    bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }
};

} // namespace graphos
