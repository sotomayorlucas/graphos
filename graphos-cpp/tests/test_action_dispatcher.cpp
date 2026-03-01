#include <gtest/gtest.h>
#include "graphos/nodes/action_dispatcher.hpp"
#include "graphos/actions/counter.hpp"
#include "graphos/dataflow/channel.hpp"
#include <chrono>
#include <stop_token>

using namespace graphos;

using ActionMap = std::unordered_map<int, std::vector<std::shared_ptr<Action>>>;

namespace {

Decision make_decision(int class_id, int route_id, PathKind path, uint64_t seq) {
    Decision d;
    d.packet.bytes.fill(static_cast<uint8_t>(seq % 256));
    d.packet.length = TENSOR_DIM;
    d.class_id = class_id;
    d.route_id = route_id;
    d.path = path;
    d.confidence = 1.0f;
    d.sequence = seq;
    return d;
}

} // namespace

TEST(ActionDispatcher, DispatchesByClass) {
    auto counter = std::make_shared<CountAction>();
    ActionMap class_actions;
    class_actions[CLASS_HTTP] = {counter};
    class_actions[CLASS_DNS] = {counter};

    auto node = std::make_shared<ActionDispatcher>("dispatcher",
        std::move(class_actions));

    auto in_ch = std::make_shared<SpscChannel<Decision>>(16);
    node->in.set_channel(in_ch);

    in_ch->push(make_decision(CLASS_HTTP, -1, PathKind::FAST_PATH, 0));
    in_ch->push(make_decision(CLASS_HTTP, -1, PathKind::FAST_PATH, 1));
    in_ch->push(make_decision(CLASS_DNS, -1, PathKind::FAST_PATH, 2));
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_EQ(counter->count(CLASS_HTTP), 2u);
    EXPECT_EQ(counter->count(CLASS_DNS), 1u);
    EXPECT_EQ(node->items_processed(), 3u);
}

TEST(ActionDispatcher, DispatchesByRoute) {
    auto counter = std::make_shared<CountAction>();
    ActionMap route_actions;
    route_actions[ROUTE_DROP] = {counter};
    route_actions[ROUTE_FORWARD] = {counter};

    ActionMap empty_class_actions;
    auto node = std::make_shared<ActionDispatcher>("dispatcher",
        std::move(empty_class_actions), std::move(route_actions));

    auto in_ch = std::make_shared<SpscChannel<Decision>>(16);
    node->in.set_channel(in_ch);

    in_ch->push(make_decision(-1, ROUTE_DROP, PathKind::FAST_PATH, 0));
    in_ch->push(make_decision(-1, ROUTE_FORWARD, PathKind::FAST_PATH, 1));
    in_ch->push(make_decision(-1, ROUTE_FORWARD, PathKind::HARD_PATH, 2));
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_EQ(counter->count(ROUTE_DROP), 1u);
    EXPECT_EQ(counter->count(ROUTE_FORWARD), 2u);
}

TEST(ActionDispatcher, DefaultAction) {
    auto default_counter = std::make_shared<CountAction>();

    ActionMap empty1, empty2;
    auto node = std::make_shared<ActionDispatcher>("dispatcher",
        std::move(empty1), std::move(empty2), default_counter);

    auto in_ch = std::make_shared<SpscChannel<Decision>>(16);
    node->in.set_channel(in_ch);

    in_ch->push(make_decision(CLASS_HTTP, -1, PathKind::FAST_PATH, 0));
    in_ch->push(make_decision(CLASS_OTHER, -1, PathKind::HARD_PATH, 1));
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_EQ(default_counter->count(CLASS_HTTP), 1u);
    EXPECT_EQ(default_counter->count(CLASS_OTHER), 1u);
}

TEST(ActionDispatcher, TracksPathCounts) {
    auto node = std::make_shared<ActionDispatcher>("dispatcher");

    auto in_ch = std::make_shared<SpscChannel<Decision>>(16);
    node->in.set_channel(in_ch);

    in_ch->push(make_decision(CLASS_HTTP, ROUTE_FORWARD, PathKind::FAST_PATH, 0));
    in_ch->push(make_decision(CLASS_DNS, ROUTE_LOCAL, PathKind::FAST_PATH, 1));
    in_ch->push(make_decision(CLASS_OTHER, -1, PathKind::HARD_PATH, 2));
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_EQ(node->fast_path_count(), 2u);
    EXPECT_EQ(node->hard_path_count(), 1u);
}

TEST(ActionDispatcher, EmptyInput) {
    auto node = std::make_shared<ActionDispatcher>("dispatcher");

    auto in_ch = std::make_shared<SpscChannel<Decision>>(16);
    node->in.set_channel(in_ch);
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_EQ(node->items_processed(), 0u);
    EXPECT_EQ(node->fast_path_count(), 0u);
    EXPECT_EQ(node->hard_path_count(), 0u);
}

TEST(ActionDispatcher, BothClassAndRouteActions) {
    auto class_counter = std::make_shared<CountAction>();
    auto route_counter = std::make_shared<CountAction>();

    ActionMap class_actions;
    class_actions[CLASS_HTTP] = {class_counter};

    ActionMap route_actions;
    route_actions[ROUTE_FORWARD] = {route_counter};

    auto node = std::make_shared<ActionDispatcher>("dispatcher",
        std::move(class_actions), std::move(route_actions));

    auto in_ch = std::make_shared<SpscChannel<Decision>>(16);
    node->in.set_channel(in_ch);

    in_ch->push(make_decision(CLASS_HTTP, ROUTE_FORWARD, PathKind::FAST_PATH, 0));
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_EQ(class_counter->count(CLASS_HTTP), 1u);
    EXPECT_EQ(route_counter->count(ROUTE_FORWARD), 1u);
}

TEST(ActionDispatcher, RecordsLatency) {
    auto node = std::make_shared<ActionDispatcher>("dispatcher");

    auto in_ch = std::make_shared<SpscChannel<Decision>>(16);
    node->in.set_channel(in_ch);

    // Create decisions with ingress_ns set to a past timestamp
    auto past = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) - 5000; // 5us ago

    Decision d1;
    d1.packet.bytes.fill(0);
    d1.packet.length = TENSOR_DIM;
    d1.class_id = CLASS_HTTP;
    d1.path = PathKind::FAST_PATH;
    d1.sequence = 0;
    d1.ingress_ns = past;

    Decision d2;
    d2.packet.bytes.fill(1);
    d2.packet.length = TENSOR_DIM;
    d2.class_id = CLASS_OTHER;
    d2.path = PathKind::HARD_PATH;
    d2.sequence = 1;
    d2.ingress_ns = past;

    // Also test that ingress_ns=0 does NOT record
    Decision d3;
    d3.packet.bytes.fill(2);
    d3.packet.length = TENSOR_DIM;
    d3.class_id = CLASS_DNS;
    d3.path = PathKind::FAST_PATH;
    d3.sequence = 2;
    d3.ingress_ns = 0;  // should be skipped

    in_ch->push(std::move(d1));
    in_ch->push(std::move(d2));
    in_ch->push(std::move(d3));
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_EQ(node->overall_latency().count(), 2u);
    EXPECT_EQ(node->fast_latency().count(), 1u);
    EXPECT_EQ(node->hard_latency().count(), 1u);
    // Latencies should be positive (ingress was in the past)
    EXPECT_GT(node->overall_latency().mean(), 0.0);
    EXPECT_GT(node->fast_latency().mean(), 0.0);
    EXPECT_GT(node->hard_latency().mean(), 0.0);
}
