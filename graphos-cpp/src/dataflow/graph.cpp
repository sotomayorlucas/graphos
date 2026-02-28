#include "graphos/dataflow/graph.hpp"
#include <deque>
#include <sstream>

namespace graphos {

void Graph::add_node(std::shared_ptr<Node> node) {
    const auto& name = node->name();
    if (nodes_.count(name))
        throw GraphError("Duplicate node name: '" + name + "'");
    edges_[name]; // init empty adjacency set
    nodes_.emplace(name, std::move(node));
}

void Graph::add_edge(const std::string& src, const std::string& dst) {
    edges_[src].insert(dst);
    if (edges_.find(dst) == edges_.end())
        edges_[dst]; // ensure dst has an entry
}

bool Graph::would_create_cycle(const std::string& src,
                               const std::string& dst) const {
    // DFS from dst — if we can reach src, adding src→dst creates a cycle
    std::unordered_set<std::string> visited;
    std::vector<std::string> stack = {dst};

    while (!stack.empty()) {
        auto node = std::move(stack.back());
        stack.pop_back();
        if (node == src) return true;
        if (visited.count(node)) continue;
        visited.insert(node);
        auto it = edges_.find(node);
        if (it != edges_.end()) {
            for (const auto& neighbor : it->second) {
                stack.push_back(neighbor);
            }
        }
    }
    return false;
}

void Graph::validate() const {
    // For now, just ensure graph is non-empty and acyclic (topo sort will catch cycles)
    if (nodes_.empty())
        throw GraphError("Graph has no nodes");
    // Try topo sort — throws on cycle
    topological_order();
}

std::vector<std::shared_ptr<Node>> Graph::topological_order() const {
    // Kahn's algorithm
    std::unordered_map<std::string, int> in_degree;
    for (const auto& [name, _] : nodes_) {
        in_degree[name] = 0;
    }
    for (const auto& [src, dsts] : edges_) {
        for (const auto& dst : dsts) {
            if (in_degree.count(dst))
                in_degree[dst]++;
        }
    }

    std::deque<std::string> queue;
    for (const auto& [name, deg] : in_degree) {
        if (deg == 0) queue.push_back(name);
    }

    std::vector<std::shared_ptr<Node>> order;
    order.reserve(nodes_.size());

    while (!queue.empty()) {
        auto name = std::move(queue.front());
        queue.pop_front();
        order.push_back(nodes_.at(name));
        auto it = edges_.find(name);
        if (it != edges_.end()) {
            for (const auto& dst : it->second) {
                if (--in_degree[dst] == 0) {
                    queue.push_back(dst);
                }
            }
        }
    }

    if (order.size() != nodes_.size())
        throw GraphError("Graph contains a cycle");

    return order;
}

void Graph::close_all_channels() noexcept {
    for (auto& closer : channels_) {
        try { closer(); } catch (...) {}
    }
}

} // namespace graphos
