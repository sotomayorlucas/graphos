#include "graphos/nodes/hard_batcher.hpp"
#include <thread>

namespace graphos {

void HardBatcher::process(std::stop_token st) {
    NpuBatchItem batch;
    batch.count = 0;
    auto batch_start = std::chrono::steady_clock::now();
    int spin_count = 0;

    while (!st.stop_requested()) {
        auto item = in.try_get();

        if (item.has_value()) {
            batch.items[batch.count] = std::move(*item);
            batch.count++;
            spin_count = 0;

            if (batch.count == 0) {
                // First item in batch — reset deadline
                batch_start = std::chrono::steady_clock::now();
            }

            // Flush if full
            if (batch.count >= batch_size_) {
                batch.padded_to = batch_size_;
                out.put(std::move(batch));
                items_processed_.fetch_add(batch_size_, std::memory_order_relaxed);
                batch = NpuBatchItem{};
                batch.count = 0;
                batch_start = std::chrono::steady_clock::now();
            }
            continue;
        }

        // No item available — check if channel closed
        if (in.channel()->is_closed() && in.channel()->empty()) {
            // Flush remaining
            if (batch.count > 0) {
                batch.padded_to = batch_size_;
                out.put(std::move(batch));
                items_processed_.fetch_add(batch.count, std::memory_order_relaxed);
            }
            out.close();
            return;
        }

        // Check deadline flush
        if (batch.count >= min_fill_) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - batch_start);
            if (elapsed >= deadline_) {
                batch.padded_to = batch_size_;
                out.put(std::move(batch));
                items_processed_.fetch_add(batch.count, std::memory_order_relaxed);
                batch = NpuBatchItem{};
                batch.count = 0;
                batch_start = std::chrono::steady_clock::now();
                spin_count = 0;
                continue;
            }
        }

        // Adaptive backoff
        ++spin_count;
        if (spin_count < 32) {
#if defined(_MSC_VER)
            _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        } else if (spin_count < 64) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }

    // Stop requested — flush remaining
    if (batch.count > 0) {
        batch.padded_to = batch_size_;
        out.put(std::move(batch));
        items_processed_.fetch_add(batch.count, std::memory_order_relaxed);
    }
    out.close();
}

} // namespace graphos
