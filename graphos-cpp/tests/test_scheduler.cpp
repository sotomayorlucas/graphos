#include <gtest/gtest.h>
#include "graphos/dataflow/graph.hpp"
#include "graphos/dataflow/scheduler.hpp"
#include "graphos/dataflow/port.hpp"

using namespace graphos;

// Source node: emits N integers then closes
class IntSourceNode : public Node {
    size_t count_;
public:
    OutputPort<int> out{"out"};

    IntSourceNode(std::string name, size_t count)
        : Node(std::move(name)), count_(count) {}

    void process(std::stop_token st) override {
        for (size_t i = 0; i < count_ && !st.stop_requested(); ++i) {
            out.put(static_cast<int>(i));
            items_processed_.fetch_add(1, std::memory_order_relaxed);
        }
        out.close();
    }
};

// Sink node: collects ints
class IntSinkNode : public Node {
    std::vector<int> results_;
    mutable std::mutex mu_;
public:
    InputPort<int> in{"in"};

    explicit IntSinkNode(std::string name) : Node(std::move(name)) {}

    void process(std::stop_token st) override {
        while (!st.stop_requested()) {
            auto v = in.get();
            if (!v.has_value()) return;
            std::lock_guard lock(mu_);
            results_.push_back(*v);
            items_processed_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::vector<int> results() const {
        std::lock_guard lock(mu_);
        return results_;
    }
};

TEST(Scheduler, SourceToSink) {
    Graph g;
    auto src = std::make_shared<IntSourceNode>("src", 100);
    auto sink = std::make_shared<IntSinkNode>("sink");
    g.add_node(src);
    g.add_node(sink);
    g.connect<int>(src->out, "src", sink->in, "sink");

    Scheduler sched(g);
    auto metrics = sched.run();

    EXPECT_EQ(metrics.size(), 2u);
    EXPECT_EQ(metrics.at("src").items_processed, 100u);
    EXPECT_EQ(metrics.at("sink").items_processed, 100u);
    EXPECT_TRUE(metrics.at("src").error.empty());
    EXPECT_TRUE(metrics.at("sink").error.empty());
    EXPECT_GT(sched.wall_time(), 0.0);

    auto results = sink->results();
    ASSERT_EQ(results.size(), 100u);
    for (size_t i = 0; i < 100; ++i)
        EXPECT_EQ(results[i], static_cast<int>(i));
}

// Pass-through node: reads int, writes int+1
class IncrementNode : public Node {
public:
    InputPort<int> in{"in"};
    OutputPort<int> out{"out"};

    explicit IncrementNode(std::string name) : Node(std::move(name)) {}

    void process(std::stop_token st) override {
        while (!st.stop_requested()) {
            auto v = in.get();
            if (!v.has_value()) { out.close(); return; }
            out.put(*v + 1);
            items_processed_.fetch_add(1, std::memory_order_relaxed);
        }
        out.close();
    }
};

TEST(Scheduler, ThreeNodeChain) {
    Graph g;
    auto src = std::make_shared<IntSourceNode>("src", 50);
    auto inc = std::make_shared<IncrementNode>("inc");
    auto sink = std::make_shared<IntSinkNode>("sink");

    g.add_node(src);
    g.add_node(inc);
    g.add_node(sink);
    g.connect<int>(src->out, "src", inc->in, "inc");
    g.connect<int>(inc->out, "inc", sink->in, "sink");

    Scheduler sched(g);
    auto metrics = sched.run();

    auto results = sink->results();
    ASSERT_EQ(results.size(), 50u);
    for (size_t i = 0; i < 50; ++i)
        EXPECT_EQ(results[i], static_cast<int>(i + 1));
}

TEST(Scheduler, LargeVolume) {
    constexpr size_t N = 10000;
    Graph g;
    auto src = std::make_shared<IntSourceNode>("src", N);
    auto sink = std::make_shared<IntSinkNode>("sink");
    g.add_node(src);
    g.add_node(sink);
    g.connect<int>(src->out, "src", sink->in, "sink", 256);

    Scheduler sched(g);
    auto metrics = sched.run();

    EXPECT_EQ(metrics.at("src").items_processed, N);
    EXPECT_EQ(metrics.at("sink").items_processed, N);
    EXPECT_EQ(sink->results().size(), N);
}
