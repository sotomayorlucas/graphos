#include "graphos/dataflow/scheduler.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <mutex>

namespace graphos {

std::unordered_map<std::string, NodeMetrics> Scheduler::run() {
    graph_.validate();
    auto order = graph_.topological_order();

    std::unordered_map<std::string, NodeMetrics> metrics;
    std::mutex metrics_mu;

    auto t0 = std::chrono::steady_clock::now();

    // Launch one jthread per node in topological order
    threads_.reserve(order.size());
    for (auto& node_ptr : order) {
        threads_.emplace_back([&node_ptr, &metrics, &metrics_mu, this](
                                  std::stop_token st) {
            NodeMetrics m;
            m.name = node_ptr->name();
            try {
                node_ptr->run(st);
                m.items_processed = node_ptr->items_processed();
                m.elapsed_s = node_ptr->elapsed();
                if (m.elapsed_s > 0.0)
                    m.throughput = static_cast<double>(m.items_processed) /
                                  m.elapsed_s;
            } catch (const std::exception& e) {
                m.error = e.what();
                m.items_processed = node_ptr->items_processed();
                m.elapsed_s = node_ptr->elapsed();
            } catch (...) {
                m.error = "unknown exception";
            }
            std::lock_guard lock(metrics_mu);
            metrics[m.name] = std::move(m);
        });
    }

    // Wait for all threads to finish naturally (via channel close propagation).
    // Do NOT use threads_.clear() which calls ~jthread() → request_stop() prematurely.
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();

    auto t1 = std::chrono::steady_clock::now();
    wall_time_ = std::chrono::duration<double>(t1 - t0).count();

    return metrics;
}

void Scheduler::stop() noexcept {
    for (auto& t : threads_) {
        t.request_stop();
    }
    graph_.close_all_channels();
}

void Scheduler::print_metrics(
    const std::unordered_map<std::string, NodeMetrics>& metrics,
    double wall_time) {
    std::cout << std::left
              << std::setw(24) << "Node"
              << std::setw(12) << "Items"
              << std::setw(12) << "Time(s)"
              << std::setw(16) << "Throughput"
              << "Error" << '\n';
    std::cout << std::string(76, '-') << '\n';

    for (const auto& [name, m] : metrics) {
        std::cout << std::left
                  << std::setw(24) << m.name
                  << std::setw(12) << m.items_processed
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << m.elapsed_s
                  << std::setw(16) << std::fixed << std::setprecision(1)
                  << m.throughput;
        if (!m.error.empty()) std::cout << m.error;
        std::cout << '\n';
    }
    std::cout << std::string(76, '-') << '\n';
    std::cout << "Wall time: " << std::fixed << std::setprecision(4)
              << wall_time << "s\n";
}

} // namespace graphos
