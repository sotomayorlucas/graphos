#pragma once
#include <atomic>
#include <cstddef>
#include <optional>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace graphos {

// ── Lock-free SPSC ring buffer ──
// Lamport-style with cache-line padding to prevent false sharing.
// Capacity is rounded up to power-of-2 for branchless modulo (bitwise AND).
// Shutdown via close() + nullopt return — no sentinel objects.

template <typename T>
class SpscChannel {
    static_assert(std::is_move_constructible_v<T>,
                  "SpscChannel requires movable types");

    // Cache-line aligned atomic counters — prevent false sharing
    struct alignas(64) PaddedAtomic {
        std::atomic<size_t> value{0};
        char pad[64 - sizeof(std::atomic<size_t>)];
    };

    PaddedAtomic head_; // written by producer, read by consumer
    PaddedAtomic tail_; // written by consumer, read by producer

    // Slot storage — aligned allocation
    struct Slot {
        typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
        std::atomic<bool> occupied{false};
    };

    // We use a simpler approach: direct array indexed by (head/tail & mask)
    std::vector<std::optional<T>> buffer_;
    size_t mask_; // capacity - 1 (for branchless modulo)

    alignas(64) std::atomic<bool> closed_{false};

    // Stats (cache-line separated from hot atomics)
    alignas(64) std::atomic<size_t> items_passed_{0};

    // Round up to next power of 2
    static constexpr size_t next_pow2(size_t n) noexcept {
        n--;
        n |= n >> 1; n |= n >> 2; n |= n >> 4;
        n |= n >> 8; n |= n >> 16; n |= n >> 32;
        return n + 1;
    }

    // Adaptive backoff: spin → pause → yield
    struct Backoff {
        int spins = 0;
        static constexpr int SPIN_LIMIT = 32;
        static constexpr int PAUSE_LIMIT = 64;

        void wait() noexcept {
            if (spins < SPIN_LIMIT) {
                ++spins;
                // Spin with CPU hint
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                _mm_pause();
#elif defined(__aarch64__)
                __asm__ volatile("yield");
#endif
            } else if (spins < PAUSE_LIMIT) {
                ++spins;
                std::this_thread::yield();
            } else {
                // Brief sleep for long waits (prevents 100% CPU spin)
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }

        void reset() noexcept { spins = 0; }
    };

public:
    explicit SpscChannel(size_t capacity)
        : buffer_(next_pow2(std::max(capacity, size_t(2)))),
          mask_(buffer_.size() - 1) {}

    // Non-copyable, non-movable (shared between threads)
    SpscChannel(const SpscChannel&) = delete;
    SpscChannel& operator=(const SpscChannel&) = delete;

    size_t capacity() const noexcept { return mask_ + 1; }
    size_t items_passed() const noexcept {
        return items_passed_.load(std::memory_order_relaxed);
    }

    // ── Producer API ──

    bool try_push(T&& item) noexcept {
        const size_t head = head_.value.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & mask_;
        if (next == tail_.value.load(std::memory_order_acquire)) {
            return false; // full
        }
        buffer_[head].emplace(std::move(item));
        head_.value.store(next, std::memory_order_release);
        items_passed_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool try_push(const T& item) noexcept {
        T copy = item;
        return try_push(std::move(copy));
    }

    // Blocking push with adaptive backoff
    void push(T&& item) noexcept {
        Backoff backoff;
        while (!try_push(std::move(item))) {
            if (closed_.load(std::memory_order_acquire)) return;
            backoff.wait();
        }
    }

    void push(const T& item) {
        T copy = item;
        push(std::move(copy));
    }

    // ── Consumer API ──

    std::optional<T> try_pop() noexcept {
        const size_t tail = tail_.value.load(std::memory_order_relaxed);
        if (tail == head_.value.load(std::memory_order_acquire)) {
            return std::nullopt; // empty
        }
        T item = std::move(*buffer_[tail]);
        buffer_[tail].reset();
        tail_.value.store((tail + 1) & mask_, std::memory_order_release);
        return item;
    }

    // Blocking pop — returns nullopt only when closed AND drained
    std::optional<T> pop() noexcept {
        Backoff backoff;
        for (;;) {
            auto item = try_pop();
            if (item.has_value()) return item;
            if (closed_.load(std::memory_order_acquire)) {
                // Drain: one final check after seeing closed
                return try_pop();
            }
            backoff.wait();
        }
    }

    // ── Lifecycle ──

    void close() noexcept {
        closed_.store(true, std::memory_order_release);
    }

    bool is_closed() const noexcept {
        return closed_.load(std::memory_order_acquire);
    }

    bool empty() const noexcept {
        return tail_.value.load(std::memory_order_acquire) ==
               head_.value.load(std::memory_order_acquire);
    }

    size_t size() const noexcept {
        const size_t head = head_.value.load(std::memory_order_acquire);
        const size_t tail = tail_.value.load(std::memory_order_acquire);
        return (head - tail) & mask_;
    }
};

} // namespace graphos
