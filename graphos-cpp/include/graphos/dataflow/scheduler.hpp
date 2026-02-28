#pragma once
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "graphos/dataflow/graph.hpp"

namespace graphos {

struct NodeMetrics {
    std::string name;
    size_t items_processed = 0;
    double elapsed_s = 0.0;
    double throughput = 0.0; // items/sec
    std::string error;       // empty if no error
};

// ── Scheduler — one jthread per node in topological order ──
class Scheduler {
public:
    explicit Scheduler(Graph& graph) : graph_(graph) {}

    // Run all nodes, block until complete. Returns metrics per node.
    std::unordered_map<std::string, NodeMetrics> run();

    // Request graceful stop
    void stop() noexcept;

    double wall_time() const noexcept { return wall_time_; }

    // Print metrics table to stdout
    static void print_metrics(
        const std::unordered_map<std::string, NodeMetrics>& metrics,
        double wall_time);

private:
    Graph& graph_;
    std::vector<std::jthread> threads_;
    double wall_time_ = 0.0;
};

} // namespace graphos
