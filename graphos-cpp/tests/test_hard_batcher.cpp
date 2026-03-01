#include <gtest/gtest.h>
#include "graphos/nodes/hard_batcher.hpp"
#include "graphos/dataflow/channel.hpp"
#include <stop_token>
#include <thread>

using namespace graphos;

namespace {

HardPathItem make_hard_item(uint8_t val, uint64_t seq) {
    HardPathItem item;
    item.packet.bytes.fill(val);
    item.packet.length = TENSOR_DIM;
    item.sequence = seq;
    return item;
}

} // namespace

TEST(HardBatcher, FlushOnFull) {
    auto node = std::make_shared<HardBatcher>("batcher", 4, 1,
                                               std::chrono::microseconds{1000000});

    auto in_ch = std::make_shared<SpscChannel<HardPathItem>>(64);
    auto out_ch = std::make_shared<SpscChannel<NpuBatchItem>>(16);
    node->in.set_channel(in_ch);
    node->out.set_channel(out_ch);

    // Push exactly 4 items (batch_size=4)
    for (int i = 0; i < 4; ++i) {
        in_ch->push(make_hard_item(static_cast<uint8_t>(i), i));
    }
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    auto batch = out_ch->pop();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->count, 4u);
    EXPECT_EQ(batch->padded_to, 4u);
    EXPECT_EQ(batch->items[0].sequence, 0u);
    EXPECT_EQ(batch->items[3].sequence, 3u);

    EXPECT_FALSE(out_ch->pop().has_value());
}

TEST(HardBatcher, FlushPartialOnClose) {
    auto node = std::make_shared<HardBatcher>("batcher", 4, 1,
                                               std::chrono::microseconds{1000000});

    auto in_ch = std::make_shared<SpscChannel<HardPathItem>>(64);
    auto out_ch = std::make_shared<SpscChannel<NpuBatchItem>>(16);
    node->in.set_channel(in_ch);
    node->out.set_channel(out_ch);

    // Push 2 items (less than batch_size=4)
    in_ch->push(make_hard_item(10, 0));
    in_ch->push(make_hard_item(20, 1));
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    auto batch = out_ch->pop();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->count, 2u);
    EXPECT_EQ(batch->padded_to, 4u); // padded to batch_size
    EXPECT_EQ(batch->items[0].packet.bytes[0], 10);
    EXPECT_EQ(batch->items[1].packet.bytes[0], 20);

    EXPECT_FALSE(out_ch->pop().has_value());
}

TEST(HardBatcher, MultipleBatches) {
    auto node = std::make_shared<HardBatcher>("batcher", 4, 1,
                                               std::chrono::microseconds{1000000});

    auto in_ch = std::make_shared<SpscChannel<HardPathItem>>(64);
    auto out_ch = std::make_shared<SpscChannel<NpuBatchItem>>(16);
    node->in.set_channel(in_ch);
    node->out.set_channel(out_ch);

    // Push 6 items → 1 full batch of 4 + 1 partial of 2
    for (int i = 0; i < 6; ++i) {
        in_ch->push(make_hard_item(static_cast<uint8_t>(i), i));
    }
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    auto b1 = out_ch->pop();
    ASSERT_TRUE(b1.has_value());
    EXPECT_EQ(b1->count, 4u);

    auto b2 = out_ch->pop();
    ASSERT_TRUE(b2.has_value());
    EXPECT_EQ(b2->count, 2u);

    EXPECT_FALSE(out_ch->pop().has_value());
}

TEST(HardBatcher, EmptyInputClosesOutput) {
    auto node = std::make_shared<HardBatcher>("batcher", 4);

    auto in_ch = std::make_shared<SpscChannel<HardPathItem>>(16);
    auto out_ch = std::make_shared<SpscChannel<NpuBatchItem>>(16);
    node->in.set_channel(in_ch);
    node->out.set_channel(out_ch);

    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_FALSE(out_ch->pop().has_value());
}

TEST(HardBatcher, DeadlineFlush) {
    // batch_size=64, min_fill=2, deadline=1ms
    auto node = std::make_shared<HardBatcher>("batcher", 64, 2,
                                               std::chrono::microseconds{1000});

    auto in_ch = std::make_shared<SpscChannel<HardPathItem>>(64);
    auto out_ch = std::make_shared<SpscChannel<NpuBatchItem>>(16);
    node->in.set_channel(in_ch);
    node->out.set_channel(out_ch);

    // Push 3 items (well below batch_size=64, but above min_fill=2)
    in_ch->push(make_hard_item(1, 0));
    in_ch->push(make_hard_item(2, 1));
    in_ch->push(make_hard_item(3, 2));

    // Run node in background thread, let deadline trigger
    std::jthread t([&](std::stop_token st) {
        node->run(st);
    });

    // Wait for deadline to trigger (>1ms)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Should have flushed a partial batch
    auto batch = out_ch->try_pop();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->count, 3u);
    EXPECT_EQ(batch->padded_to, 64u);

    // Close and cleanup
    in_ch->close();
    t.request_stop();
    t.join();
}
