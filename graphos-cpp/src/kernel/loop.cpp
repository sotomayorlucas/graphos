#include "graphos/kernel/loop.hpp"
#include <cstring>

namespace graphos {

LoopStats KernelLoop::stats() const {
    LoopStats s;
    s.packets_processed = packets_processed_.load(std::memory_order_relaxed);
    s.batches_processed = batches_processed_.load(std::memory_order_relaxed);
    auto now = std::chrono::steady_clock::now();
    s.elapsed_s = std::chrono::duration<double>(now - start_time_).count();
    s.throughput = s.elapsed_s > 0 ?
        static_cast<double>(s.packets_processed) / s.elapsed_s : 0.0;
    s.health = health_monitor_.last_health();
    return s;
}

std::unordered_map<std::string, std::vector<float>>
KernelLoop::process_batch(const std::vector<OwnedPacket>& packets) {
    std::vector<float> tensor_buf(batch_size_ * TENSOR_DIM, 0.0f);
    packets_to_batch_tensor(packets.data(), packets.size(),
                            batch_size_, tensor_buf.data());

    auto prog_list = runtime_.programs();
    std::unordered_map<std::string, std::vector<float>> results;

    for (const auto& name : prog_list) {
        std::vector<float> output(batch_size_ * NUM_ROUTES, 0.0f);
        runtime_.execute(name, tensor_buf.data(), output.data(), batch_size_);
        results[name] = std::move(output);
    }
    return results;
}

void KernelLoop::execute_batch(const std::vector<OwnedPacket>& packets,
                               const std::vector<std::string>& programs,
                               std::vector<float>& tensor_buf,
                               std::vector<float>& output_buf) {
    packets_to_batch_tensor(packets.data(), packets.size(),
                            batch_size_, tensor_buf.data());

    for (const auto& name : programs) {
        runtime_.execute(name, tensor_buf.data(), output_buf.data(),
                        batch_size_);
    }

    packets_processed_.fetch_add(packets.size(), std::memory_order_relaxed);
    batches_processed_.fetch_add(1, std::memory_order_relaxed);
}

} // namespace graphos
