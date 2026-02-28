#pragma once
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "graphos/core/constants.hpp"
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"

namespace graphos {

class GraphError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ── Dataflow Graph — DAG of nodes connected by typed channels ──
class Graph {
public:
    // Add a node (graph takes ownership via shared_ptr for scheduler lifetime)
    void add_node(std::shared_ptr<Node> node);

    // Type-erased edge record (for cycle detection / topo sort)
    void add_edge(const std::string& src, const std::string& dst);

    // Connect typed ports — creates SpscChannel, registers edge
    template <typename T>
    std::shared_ptr<SpscChannel<T>> connect(
            OutputPort<T>& src_port, const std::string& src_name,
            InputPort<T>& dst_port, const std::string& dst_name,
            size_t capacity = DEFAULT_BATCH_SIZE) {
        // Validate nodes exist
        if (nodes_.find(src_name) == nodes_.end())
            throw GraphError("Source node '" + src_name + "' not in graph");
        if (nodes_.find(dst_name) == nodes_.end())
            throw GraphError("Dest node '" + dst_name + "' not in graph");
        if (src_name == dst_name)
            throw GraphError("Self-loop on '" + src_name + "'");
        if (would_create_cycle(src_name, dst_name))
            throw GraphError("Edge " + src_name + "->" + dst_name +
                             " would create cycle");

        auto ch = graphos::connect(src_port, dst_port, capacity);
        add_edge(src_name, dst_name);
        channels_.push_back([ch]() { ch->close(); });
        return ch;
    }

    // Validate all ports connected
    void validate() const;

    // Kahn's topological sort
    std::vector<std::shared_ptr<Node>> topological_order() const;

    // Node queries
    const std::unordered_map<std::string, std::shared_ptr<Node>>& nodes() const noexcept {
        return nodes_;
    }

    // Close all channels (for shutdown)
    void close_all_channels() noexcept;

private:
    bool would_create_cycle(const std::string& src,
                            const std::string& dst) const;

    std::unordered_map<std::string, std::shared_ptr<Node>> nodes_;
    std::unordered_map<std::string, std::unordered_set<std::string>> edges_;
    // Type-erased channel closers
    std::vector<std::function<void()>> channels_;
};

} // namespace graphos
