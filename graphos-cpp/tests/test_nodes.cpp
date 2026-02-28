#include <gtest/gtest.h>
#include "graphos/dataflow/graph.hpp"
#include "graphos/dataflow/scheduler.hpp"
#include "graphos/nodes/source_node.hpp"
#include "graphos/nodes/batch_node.hpp"
#include "graphos/nodes/tensor_node.hpp"
#include "graphos/nodes/sink_node.hpp"
#include "graphos/nodes/tee_node.hpp"
#include "graphos/core/types.hpp"
#include <vector>

using namespace graphos;

// Helper: generate N test packets
std::vector<std::vector<uint8_t>> make_test_packets(size_t n) {
    std::vector<std::vector<uint8_t>> pkts;
    for (size_t i = 0; i < n; ++i) {
        std::vector<uint8_t> raw(TENSOR_DIM, static_cast<uint8_t>(i % 256));
        pkts.push_back(std::move(raw));
    }
    return pkts;
}

TEST(Types, OwnedPacketFromRaw) {
    std::vector<uint8_t> raw = {1, 2, 3, 4, 5};
    auto pkt = OwnedPacket::from_raw(raw);
    EXPECT_EQ(pkt.length, 5u);
    EXPECT_EQ(pkt.bytes[0], 1);
    EXPECT_EQ(pkt.bytes[4], 5);
    EXPECT_EQ(pkt.bytes[5], 0); // zero-padded
}

TEST(Types, PacketToTensor) {
    OwnedPacket pkt{};
    pkt.bytes[0] = 255;
    pkt.bytes[1] = 128;
    pkt.length = TENSOR_DIM;

    float tensor[TENSOR_DIM];
    packet_to_tensor(pkt, tensor);

    EXPECT_FLOAT_EQ(tensor[0], 1.0f);
    EXPECT_NEAR(tensor[1], 128.0f / 255.0f, 1e-6f);
    EXPECT_FLOAT_EQ(tensor[2], 0.0f);
}

TEST(Types, BatchTensorConversion) {
    std::array<OwnedPacket, 2> pkts;
    pkts[0].bytes.fill(255);
    pkts[0].length = TENSOR_DIM;
    pkts[1].bytes.fill(0);
    pkts[1].length = TENSOR_DIM;

    float output[4 * TENSOR_DIM]; // batch_size=4
    packets_to_batch_tensor(pkts.data(), 2, 4, output);

    // First row: all 1.0
    EXPECT_FLOAT_EQ(output[0], 1.0f);
    // Second row: all 0.0
    EXPECT_FLOAT_EQ(output[TENSOR_DIM], 0.0f);
    // Third row (padding): all 0.0
    EXPECT_FLOAT_EQ(output[2 * TENSOR_DIM], 0.0f);
}

TEST(Types, Argmax) {
    // 2 rows, 3 classes
    float logits[] = {0.1f, 0.9f, 0.0f,   // class 1
                      0.5f, 0.3f, 0.2f};   // class 0
    int classes[2];
    batch_tensor_to_classes(logits, 2, 3, classes);
    EXPECT_EQ(classes[0], 1);
    EXPECT_EQ(classes[1], 0);
}

TEST(Types, TensorToClass) {
    float logits[] = {0.1f, 0.2f, 0.7f};
    EXPECT_EQ(tensor_to_class(logits, 3), 2);
}

TEST(BatchNode, AccumulatesPackets) {
    auto batch = std::make_shared<BatchNode>("batch", 4);

    // Create channel pair manually for unit test
    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto out_ch = std::make_shared<SpscChannel<BatchItem>>(16);
    batch->in.set_channel(in_ch);
    batch->out.set_channel(out_ch);

    // Push 6 packets (should get 1 full batch of 4 + 1 partial of 2)
    for (int i = 0; i < 6; ++i) {
        OwnedPacket pkt{};
        pkt.bytes[0] = static_cast<uint8_t>(i);
        pkt.length = TENSOR_DIM;
        in_ch->push(std::move(pkt));
    }
    in_ch->close();

    std::stop_source ss;
    batch->run(ss.get_token());

    // Pop results
    auto b1 = out_ch->pop();
    ASSERT_TRUE(b1.has_value());
    EXPECT_EQ(b1->count, 4u);
    EXPECT_EQ(b1->packets[0].bytes[0], 0);
    EXPECT_EQ(b1->packets[3].bytes[0], 3);

    auto b2 = out_ch->pop();
    ASSERT_TRUE(b2.has_value());
    EXPECT_EQ(b2->count, 2u);
    EXPECT_EQ(b2->packets[0].bytes[0], 4);
    EXPECT_EQ(b2->packets[1].bytes[0], 5);

    // Channel closed
    auto b3 = out_ch->pop();
    EXPECT_FALSE(b3.has_value());
}

TEST(TensorNode, ConvertsBatchToTensor) {
    auto tensor_node = std::make_shared<TensorNode>("tensor", 4);

    auto in_ch = std::make_shared<SpscChannel<BatchItem>>(16);
    auto out_ch = std::make_shared<SpscChannel<TensorItem>>(16);
    tensor_node->in.set_channel(in_ch);
    tensor_node->out.set_channel(out_ch);

    // Create a batch
    BatchItem batch;
    batch.count = 2;
    batch.packets[0].bytes.fill(255);
    batch.packets[0].length = TENSOR_DIM;
    batch.packets[1].bytes.fill(128);
    batch.packets[1].length = TENSOR_DIM;

    in_ch->push(std::move(batch));
    in_ch->close();

    std::stop_source ss;
    tensor_node->run(ss.get_token());

    auto result = out_ch->pop();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->count, 2u);
    EXPECT_EQ(result->data.size(), 4u * TENSOR_DIM); // batch_size=4

    // First row: 255/255 = 1.0
    EXPECT_FLOAT_EQ(result->data[0], 1.0f);
    // Second row: 128/255
    EXPECT_NEAR(result->data[TENSOR_DIM], 128.0f / 255.0f, 1e-6f);
    // Third row (padding): 0.0
    EXPECT_FLOAT_EQ(result->data[2 * TENSOR_DIM], 0.0f);
}

TEST(TeeNode, DuplicatesOutput) {
    auto tee = std::make_shared<TeeNode>("tee");

    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto out_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto copy_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    tee->in.set_channel(in_ch);
    tee->out.set_channel(out_ch);
    tee->copy.set_channel(copy_ch);

    OwnedPacket pkt{};
    pkt.bytes[0] = 42;
    pkt.length = 1;
    in_ch->push(std::move(pkt));
    in_ch->close();

    std::stop_source ss;
    tee->run(ss.get_token());

    auto v1 = out_ch->pop();
    auto v2 = copy_ch->pop();
    ASSERT_TRUE(v1.has_value());
    ASSERT_TRUE(v2.has_value());
    EXPECT_EQ(v1->bytes[0], 42);
    EXPECT_EQ(v2->bytes[0], 42);

    // Both channels closed
    EXPECT_FALSE(out_ch->pop().has_value());
    EXPECT_FALSE(copy_ch->pop().has_value());
}

TEST(SourceNode, EmitsPackets) {
    std::vector<std::vector<uint8_t>> pkts = {{1, 2, 3}, {4, 5, 6}};
    using Src = IterableSource<std::vector<std::vector<uint8_t>>::iterator>;
    auto source = std::make_shared<SourceNode<Src>>(
        "src", Src(pkts.begin(), pkts.end()));

    auto out_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    source->out.set_channel(out_ch);

    std::stop_source ss;
    source->run(ss.get_token());

    auto p1 = out_ch->pop();
    ASSERT_TRUE(p1.has_value());
    EXPECT_EQ(p1->bytes[0], 1);
    EXPECT_EQ(p1->length, 3u);

    auto p2 = out_ch->pop();
    ASSERT_TRUE(p2.has_value());
    EXPECT_EQ(p2->bytes[0], 4);

    EXPECT_FALSE(out_ch->pop().has_value());
}
