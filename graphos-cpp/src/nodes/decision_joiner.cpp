#include "graphos/nodes/decision_joiner.hpp"
#include <thread>

namespace graphos {

void DecisionJoiner::process(std::stop_token st) {
    bool fast_closed = false;
    bool hard_closed = false;
    int spin_count = 0;

    auto emit = [&](Decision&& d) {
        if (!ordered_) {
            out.put(std::move(d));
            items_processed_.fetch_add(1, std::memory_order_relaxed);
        } else {
            reorder_buf_.emplace(d.sequence, std::move(d));
            // Flush contiguous prefix
            while (!reorder_buf_.empty() &&
                   reorder_buf_.begin()->first == next_seq_) {
                out.put(std::move(reorder_buf_.begin()->second));
                reorder_buf_.erase(reorder_buf_.begin());
                items_processed_.fetch_add(1, std::memory_order_relaxed);
                next_seq_++;
            }
        }
    };

    while (!st.stop_requested()) {
        bool got_something = false;

        if (!fast_closed) {
            auto d = fast_in.try_get();
            if (d.has_value()) {
                emit(std::move(*d));
                got_something = true;
            } else if (fast_in.channel()->is_closed() &&
                       fast_in.channel()->empty()) {
                fast_closed = true;
            }
        }

        if (!hard_closed) {
            auto d = hard_in.try_get();
            if (d.has_value()) {
                emit(std::move(*d));
                got_something = true;
            } else if (hard_in.channel()->is_closed() &&
                       hard_in.channel()->empty()) {
                hard_closed = true;
            }
        }

        if (fast_closed && hard_closed) break;

        if (got_something) {
            spin_count = 0;
        } else {
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
    }

    // Flush remaining reorder buffer
    if (ordered_) {
        for (auto& [seq, d] : reorder_buf_) {
            out.put(std::move(d));
            items_processed_.fetch_add(1, std::memory_order_relaxed);
        }
        reorder_buf_.clear();
    }

    out.close();
}

} // namespace graphos
