#pragma once
#include "graphos/core/decision.hpp"
#include "graphos/actions/action.hpp"
#include "graphos/dataflow/graph.hpp"
#include "graphos/dataflow/scheduler.hpp"
#include "graphos/nodes/source_node.hpp"
#include "graphos/nodes/fast_path_classifier.hpp"
#include "graphos/nodes/hard_batcher.hpp"
#include "graphos/nodes/npu_executor_async.hpp"
#include "graphos/nodes/decision_joiner.hpp"
#include "graphos/nodes/action_dispatcher.hpp"
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace graphos {

struct GpnpuConfig {
    std::string onnx_path;
    std::string device = "NPU";
    size_t batch_size = DEFAULT_BATCH_SIZE;
    size_t min_fill = 12;
    std::chrono::microseconds deadline{150};
    size_t num_inflight = 3;
    bool ordered = false;
};

struct GpnpuPipeline {
    Graph graph;
    std::shared_ptr<FastPathClassifier> classifier;
    std::shared_ptr<HardBatcher> batcher;
    std::shared_ptr<NpuExecutorAsync> executor;
    std::shared_ptr<DecisionJoiner> joiner;
    std::shared_ptr<ActionDispatcher> dispatcher;
};

// Builds the GPNPU diamond-DAG pipeline:
//   Source → FastPathClassifier ──fast_out──→ DecisionJoiner → ActionDispatcher
//                               └─hard_out──→ HardBatcher → NpuExecutorAsync ─→ DecisionJoiner
template <typename SrcNode>
GpnpuPipeline build_gpnpu_pipeline(
    std::shared_ptr<SrcNode> source,
    const GpnpuConfig& config,
    std::unordered_map<int, std::vector<std::shared_ptr<Action>>> class_actions = {},
    std::unordered_map<int, std::vector<std::shared_ptr<Action>>> route_actions = {},
    std::shared_ptr<Action> default_action = nullptr)
{
    GpnpuPipeline p;

    // Create nodes
    auto classifier = std::make_shared<FastPathClassifier>("fast_path");
    auto batcher = std::make_shared<HardBatcher>(
        "hard_batcher", config.batch_size, config.min_fill, config.deadline);
    auto executor = std::make_shared<NpuExecutorAsync>(
        "npu_executor", config.onnx_path, config.device,
        config.batch_size, config.num_inflight);
    auto joiner = std::make_shared<DecisionJoiner>("joiner", config.ordered);
    auto dispatcher = std::make_shared<ActionDispatcher>(
        "dispatcher", std::move(class_actions), std::move(route_actions),
        std::move(default_action));

    // Register nodes
    p.graph.add_node(source);
    p.graph.add_node(classifier);
    p.graph.add_node(batcher);
    p.graph.add_node(executor);
    p.graph.add_node(joiner);
    p.graph.add_node(dispatcher);

    // Wire: Source → FastPathClassifier
    p.graph.connect(source->out, source->name(),
                    classifier->in, classifier->name(),
                    config.batch_size * 4);

    // Wire: FastPathClassifier fast_out → DecisionJoiner fast_in
    p.graph.connect(classifier->fast_out, classifier->name(),
                    joiner->fast_in, joiner->name(),
                    config.batch_size * 4);

    // Wire: FastPathClassifier hard_out → HardBatcher
    p.graph.connect(classifier->hard_out, classifier->name(),
                    batcher->in, batcher->name(),
                    config.batch_size * 4);

    // Wire: HardBatcher → NpuExecutorAsync
    p.graph.connect(batcher->out, batcher->name(),
                    executor->in, executor->name(),
                    config.num_inflight + 1);

    // Wire: NpuExecutorAsync → DecisionJoiner hard_in
    p.graph.connect(executor->out, executor->name(),
                    joiner->hard_in, joiner->name(),
                    config.batch_size * 4);

    // Wire: DecisionJoiner → ActionDispatcher
    p.graph.connect(joiner->out, joiner->name(),
                    dispatcher->in, dispatcher->name(),
                    config.batch_size * 4);

    // Store shared_ptrs
    p.classifier = classifier;
    p.batcher = batcher;
    p.executor = executor;
    p.joiner = joiner;
    p.dispatcher = dispatcher;

    return p;
}

} // namespace graphos
