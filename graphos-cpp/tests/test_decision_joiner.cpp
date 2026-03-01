#include <gtest/gtest.h>
#include "graphos/nodes/decision_joiner.hpp"
#include "graphos/dataflow/channel.hpp"
#include <stop_token>
#include <thread>

using namespace graphos;

namespace {

Decision make_decision(int class_id, PathKind path, uint64_t seq) {
    Decision d;
    d.packet.bytes.fill(static_cast<uint8_t>(seq % 256));
    d.packet.length = TENSOR_DIM;
    d.class_id = class_id;
    d.path = path;
    d.sequence = seq;
    d.confidence = (path == PathKind::FAST_PATH) ? 1.0f : 0.8f;
    return d;
}

} // namespace

TEST(DecisionJoiner, FastPathOnly) {
    auto node = std::make_shared<DecisionJoiner>("joiner");
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto out_ch = std::make_shared<SpscChannel<Decision>>(16);

    node->fast_in.set_channel(fast_ch);
    node->hard_in.set_channel(hard_ch);
    node->out.set_channel(out_ch);

    fast_ch->push(make_decision(CLASS_HTTP, PathKind::FAST_PATH, 0));
    fast_ch->push(make_decision(CLASS_DNS, PathKind::FAST_PATH, 1));
    fast_ch->close();
    hard_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    size_t count = 0;
    while (auto d = out_ch->pop()) {
        EXPECT_EQ(d->path, PathKind::FAST_PATH);
        ++count;
    }
    EXPECT_EQ(count, 2u);
}

TEST(DecisionJoiner, HardPathOnly) {
    auto node = std::make_shared<DecisionJoiner>("joiner");
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto out_ch = std::make_shared<SpscChannel<Decision>>(16);

    node->fast_in.set_channel(fast_ch);
    node->hard_in.set_channel(hard_ch);
    node->out.set_channel(out_ch);

    hard_ch->push(make_decision(CLASS_OTHER, PathKind::HARD_PATH, 0));
    hard_ch->push(make_decision(CLASS_HTTP, PathKind::HARD_PATH, 1));
    fast_ch->close();
    hard_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    size_t count = 0;
    while (auto d = out_ch->pop()) {
        EXPECT_EQ(d->path, PathKind::HARD_PATH);
        ++count;
    }
    EXPECT_EQ(count, 2u);
}

TEST(DecisionJoiner, MixedPaths) {
    auto node = std::make_shared<DecisionJoiner>("joiner");
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto out_ch = std::make_shared<SpscChannel<Decision>>(64);

    node->fast_in.set_channel(fast_ch);
    node->hard_in.set_channel(hard_ch);
    node->out.set_channel(out_ch);

    fast_ch->push(make_decision(CLASS_HTTP, PathKind::FAST_PATH, 0));
    hard_ch->push(make_decision(CLASS_OTHER, PathKind::HARD_PATH, 1));
    fast_ch->push(make_decision(CLASS_DNS, PathKind::FAST_PATH, 2));
    hard_ch->push(make_decision(CLASS_HTTP, PathKind::HARD_PATH, 3));
    fast_ch->close();
    hard_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    size_t count = 0;
    while (auto d = out_ch->pop()) ++count;
    EXPECT_EQ(count, 4u);
}

TEST(DecisionJoiner, OrderedMode) {
    auto node = std::make_shared<DecisionJoiner>("joiner", /*ordered=*/true);
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto out_ch = std::make_shared<SpscChannel<Decision>>(64);

    node->fast_in.set_channel(fast_ch);
    node->hard_in.set_channel(hard_ch);
    node->out.set_channel(out_ch);

    // Push out of order: seq 0, 2, 1, 3
    fast_ch->push(make_decision(CLASS_HTTP, PathKind::FAST_PATH, 0));
    fast_ch->push(make_decision(CLASS_DNS, PathKind::FAST_PATH, 2));
    hard_ch->push(make_decision(CLASS_OTHER, PathKind::HARD_PATH, 1));
    hard_ch->push(make_decision(CLASS_HTTP, PathKind::HARD_PATH, 3));
    fast_ch->close();
    hard_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    // Output should be in sequence order: 0, 1, 2, 3
    uint64_t expected_seq = 0;
    while (auto d = out_ch->pop()) {
        EXPECT_EQ(d->sequence, expected_seq);
        expected_seq++;
    }
    EXPECT_EQ(expected_seq, 4u);
}

TEST(DecisionJoiner, EmptyInputs) {
    auto node = std::make_shared<DecisionJoiner>("joiner");
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto out_ch = std::make_shared<SpscChannel<Decision>>(16);

    node->fast_in.set_channel(fast_ch);
    node->hard_in.set_channel(hard_ch);
    node->out.set_channel(out_ch);

    fast_ch->close();
    hard_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_FALSE(out_ch->pop().has_value());
}

TEST(DecisionJoiner, ConcurrentPaths) {
    auto node = std::make_shared<DecisionJoiner>("joiner");
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(256);
    auto hard_ch = std::make_shared<SpscChannel<Decision>>(256);
    auto out_ch = std::make_shared<SpscChannel<Decision>>(512);

    node->fast_in.set_channel(fast_ch);
    node->hard_in.set_channel(hard_ch);
    node->out.set_channel(out_ch);

    constexpr size_t N = 100;

    // Producer threads
    std::jthread fast_producer([&](std::stop_token) {
        for (size_t i = 0; i < N; ++i) {
            fast_ch->push(make_decision(CLASS_HTTP, PathKind::FAST_PATH, i * 2));
        }
        fast_ch->close();
    });

    std::jthread hard_producer([&](std::stop_token) {
        for (size_t i = 0; i < N; ++i) {
            hard_ch->push(make_decision(CLASS_OTHER, PathKind::HARD_PATH, i * 2 + 1));
        }
        hard_ch->close();
    });

    std::stop_source ss;
    node->run(ss.get_token());

    fast_producer.join();
    hard_producer.join();

    size_t count = 0;
    while (auto d = out_ch->pop()) ++count;
    EXPECT_EQ(count, 2u * N);
}
