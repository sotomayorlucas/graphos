#pragma once
#include <atomic>
#include <chrono>
#include <string>
#include <stop_token>

namespace graphos {

// ── Node ABC — dataflow processing unit ──
// Each node runs in its own jthread. Communicates via typed ports.
// Shutdown signaled via stop_token + channel close.
class Node {
    std::string name_;

protected:
    std::atomic<size_t> items_processed_{0};
    double elapsed_s_ = 0.0;

public:
    explicit Node(std::string name) : name_(std::move(name)) {}
    virtual ~Node() = default;

    // Non-copyable, non-movable (registered in graph by pointer)
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    const std::string& name() const noexcept { return name_; }
    size_t items_processed() const noexcept {
        return items_processed_.load(std::memory_order_relaxed);
    }
    double elapsed() const noexcept { return elapsed_s_; }

    // Lifecycle — override in subclasses
    virtual void setup() {}
    virtual void process(std::stop_token st) = 0;
    virtual void teardown() noexcept {}

    // Called by Scheduler — measures elapsed time
    void run(std::stop_token st) {
        setup();
        auto t0 = std::chrono::steady_clock::now();
        try {
            process(st);
        } catch (...) {
            auto t1 = std::chrono::steady_clock::now();
            elapsed_s_ = std::chrono::duration<double>(t1 - t0).count();
            teardown();
            throw;
        }
        auto t1 = std::chrono::steady_clock::now();
        elapsed_s_ = std::chrono::duration<double>(t1 - t0).count();
        teardown();
    }
};

} // namespace graphos
