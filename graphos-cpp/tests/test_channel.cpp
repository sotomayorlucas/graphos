#include <gtest/gtest.h>
#include "graphos/dataflow/channel.hpp"
#include <thread>
#include <vector>
#include <numeric>

using namespace graphos;

TEST(SpscChannel, BasicPushPop) {
    SpscChannel<int> ch(4);
    EXPECT_TRUE(ch.empty());
    EXPECT_EQ(ch.size(), 0u);

    EXPECT_TRUE(ch.try_push(42));
    EXPECT_EQ(ch.size(), 1u);
    EXPECT_FALSE(ch.empty());

    auto val = ch.try_pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
    EXPECT_TRUE(ch.empty());
}

TEST(SpscChannel, CapacityRoundsUpToPow2) {
    SpscChannel<int> ch(5);
    // Rounded to 8, but usable slots = capacity - 1 = 7
    EXPECT_GE(ch.capacity(), 8u);
}

TEST(SpscChannel, FullChannel) {
    SpscChannel<int> ch(4); // pow2 = 4, usable = 3
    EXPECT_TRUE(ch.try_push(1));
    EXPECT_TRUE(ch.try_push(2));
    EXPECT_TRUE(ch.try_push(3));
    EXPECT_FALSE(ch.try_push(4)); // full

    auto v = ch.try_pop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 1);

    EXPECT_TRUE(ch.try_push(4)); // now there's space
}

TEST(SpscChannel, CloseSignal) {
    SpscChannel<int> ch(4);
    ch.try_push(1);
    ch.close();

    EXPECT_TRUE(ch.is_closed());

    // Can still drain
    auto v = ch.pop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 1);

    // After drain, pop returns nullopt
    v = ch.pop();
    EXPECT_FALSE(v.has_value());
}

TEST(SpscChannel, BlockingPopReturnsOnClose) {
    SpscChannel<int> ch(4);

    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ch.try_push(99);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ch.close();
    });

    auto v = ch.pop(); // blocks until item or close
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 99);

    v = ch.pop(); // blocks until close
    EXPECT_FALSE(v.has_value());

    producer.join();
}

TEST(SpscChannel, ProducerConsumerStress) {
    constexpr size_t N = 100000;
    SpscChannel<size_t> ch(256);

    std::thread producer([&]() {
        for (size_t i = 0; i < N; ++i) {
            ch.push(std::move(i));
        }
        ch.close();
    });

    std::vector<size_t> received;
    received.reserve(N);

    std::thread consumer([&]() {
        while (auto v = ch.pop()) {
            received.push_back(*v);
        }
    });

    producer.join();
    consumer.join();

    // Verify all items received in order
    ASSERT_EQ(received.size(), N);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(received[i], i) << "Mismatch at index " << i;
    }
}

TEST(SpscChannel, MoveOnlyTypes) {
    SpscChannel<std::unique_ptr<int>> ch(4);

    auto p = std::make_unique<int>(42);
    ch.push(std::move(p));
    EXPECT_EQ(p, nullptr); // moved

    auto v = ch.pop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(**v, 42);
}

TEST(SpscChannel, ItemsPassedCounter) {
    SpscChannel<int> ch(8);
    EXPECT_EQ(ch.items_passed(), 0u);

    ch.push(1);
    ch.push(2);
    ch.push(3);
    EXPECT_EQ(ch.items_passed(), 3u);

    ch.pop();
    EXPECT_EQ(ch.items_passed(), 3u); // only incremented on push
}
