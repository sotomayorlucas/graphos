#include <gtest/gtest.h>
#include "graphos/nodes/fast_path_classifier.hpp"
#include "graphos/core/constants.hpp"
#include "graphos/dataflow/channel.hpp"
#include <stop_token>

using namespace graphos;

namespace {

OwnedPacket make_packet_ttl0() {
    OwnedPacket pkt{};
    pkt.bytes.fill(0);
    pkt.bytes[OFFSET_TTL] = 0;
    pkt.bytes[OFFSET_PROTOCOL] = PROTO_TCP;
    pkt.length = TENSOR_DIM;
    return pkt;
}

OwnedPacket make_packet_dns() {
    OwnedPacket pkt{};
    pkt.bytes.fill(0);
    pkt.bytes[OFFSET_TTL] = 64;
    pkt.bytes[OFFSET_PROTOCOL] = PROTO_UDP;
    pkt.bytes[OFFSET_DST_PORT] = 0;
    pkt.bytes[OFFSET_DST_PORT + 1] = 53; // port 53
    pkt.length = TENSOR_DIM;
    return pkt;
}

OwnedPacket make_packet_http80() {
    OwnedPacket pkt{};
    pkt.bytes.fill(0);
    pkt.bytes[OFFSET_TTL] = 64;
    pkt.bytes[OFFSET_PROTOCOL] = PROTO_TCP;
    pkt.bytes[OFFSET_DST_PORT] = 0;
    pkt.bytes[OFFSET_DST_PORT + 1] = 80; // port 80
    pkt.length = TENSOR_DIM;
    return pkt;
}

OwnedPacket make_packet_https443() {
    OwnedPacket pkt{};
    pkt.bytes.fill(0);
    pkt.bytes[OFFSET_TTL] = 64;
    pkt.bytes[OFFSET_PROTOCOL] = PROTO_TCP;
    pkt.bytes[OFFSET_DST_PORT] = 1;
    pkt.bytes[OFFSET_DST_PORT + 1] = 187; // 443 = 0x01BB
    pkt.length = TENSOR_DIM;
    return pkt;
}

OwnedPacket make_packet_http8080() {
    OwnedPacket pkt{};
    pkt.bytes.fill(0);
    pkt.bytes[OFFSET_TTL] = 64;
    pkt.bytes[OFFSET_PROTOCOL] = PROTO_TCP;
    pkt.bytes[OFFSET_DST_PORT] = 0x1F;
    pkt.bytes[OFFSET_DST_PORT + 1] = 0x90; // 8080 = 0x1F90
    pkt.length = TENSOR_DIM;
    return pkt;
}

OwnedPacket make_packet_random() {
    OwnedPacket pkt{};
    pkt.bytes.fill(42);
    pkt.bytes[OFFSET_TTL] = 64;
    pkt.bytes[OFFSET_PROTOCOL] = PROTO_ICMP;
    pkt.length = TENSOR_DIM;
    return pkt;
}

} // namespace

TEST(FastPathClassifier, TTL0GoesToDrop) {
    auto node = std::make_shared<FastPathClassifier>("fpc");
    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<HardPathItem>>(16);

    node->in.set_channel(in_ch);
    node->fast_out.set_channel(fast_ch);
    node->hard_out.set_channel(hard_ch);

    in_ch->push(make_packet_ttl0());
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    auto d = fast_ch->pop();
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->class_id, CLASS_OTHER);
    EXPECT_EQ(d->route_id, ROUTE_DROP);
    EXPECT_EQ(d->path, PathKind::FAST_PATH);
    EXPECT_FLOAT_EQ(d->confidence, 1.0f);

    EXPECT_FALSE(hard_ch->pop().has_value());
    EXPECT_EQ(node->fast_count(), 1u);
    EXPECT_EQ(node->hard_count(), 0u);
}

TEST(FastPathClassifier, DNSGoesToLocal) {
    auto node = std::make_shared<FastPathClassifier>("fpc");
    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<HardPathItem>>(16);

    node->in.set_channel(in_ch);
    node->fast_out.set_channel(fast_ch);
    node->hard_out.set_channel(hard_ch);

    in_ch->push(make_packet_dns());
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    auto d = fast_ch->pop();
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->class_id, CLASS_DNS);
    EXPECT_EQ(d->route_id, ROUTE_LOCAL);
    EXPECT_EQ(d->path, PathKind::FAST_PATH);
}

TEST(FastPathClassifier, HTTPGoesToForward) {
    auto node = std::make_shared<FastPathClassifier>("fpc");
    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<HardPathItem>>(16);

    node->in.set_channel(in_ch);
    node->fast_out.set_channel(fast_ch);
    node->hard_out.set_channel(hard_ch);

    in_ch->push(make_packet_http80());
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    auto d = fast_ch->pop();
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->class_id, CLASS_HTTP);
    EXPECT_EQ(d->route_id, ROUTE_FORWARD);
}

TEST(FastPathClassifier, HTTPS443GoesToForward) {
    auto node = std::make_shared<FastPathClassifier>("fpc");
    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<HardPathItem>>(16);

    node->in.set_channel(in_ch);
    node->fast_out.set_channel(fast_ch);
    node->hard_out.set_channel(hard_ch);

    in_ch->push(make_packet_https443());
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    auto d = fast_ch->pop();
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->class_id, CLASS_HTTP);
    EXPECT_EQ(d->route_id, ROUTE_FORWARD);
}

TEST(FastPathClassifier, HTTP8080GoesToForward) {
    auto node = std::make_shared<FastPathClassifier>("fpc");
    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<HardPathItem>>(16);

    node->in.set_channel(in_ch);
    node->fast_out.set_channel(fast_ch);
    node->hard_out.set_channel(hard_ch);

    in_ch->push(make_packet_http8080());
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    auto d = fast_ch->pop();
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->class_id, CLASS_HTTP);
    EXPECT_EQ(d->route_id, ROUTE_FORWARD);
}

TEST(FastPathClassifier, RandomGoesToHardPath) {
    auto node = std::make_shared<FastPathClassifier>("fpc");
    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<HardPathItem>>(16);

    node->in.set_channel(in_ch);
    node->fast_out.set_channel(fast_ch);
    node->hard_out.set_channel(hard_ch);

    in_ch->push(make_packet_random());
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_FALSE(fast_ch->pop().has_value());

    auto h = hard_ch->pop();
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->packet.bytes[OFFSET_TTL], 64);
    EXPECT_EQ(node->fast_count(), 0u);
    EXPECT_EQ(node->hard_count(), 1u);
}

TEST(FastPathClassifier, MixedTraffic) {
    auto node = std::make_shared<FastPathClassifier>("fpc");
    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(64);
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(64);
    auto hard_ch = std::make_shared<SpscChannel<HardPathItem>>(64);

    node->in.set_channel(in_ch);
    node->fast_out.set_channel(fast_ch);
    node->hard_out.set_channel(hard_ch);

    // 4 fast-path + 1 hard-path
    in_ch->push(make_packet_ttl0());
    in_ch->push(make_packet_dns());
    in_ch->push(make_packet_http80());
    in_ch->push(make_packet_https443());
    in_ch->push(make_packet_random());
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_EQ(node->fast_count(), 4u);
    EXPECT_EQ(node->hard_count(), 1u);
    EXPECT_EQ(node->items_processed(), 5u);
}

TEST(FastPathClassifier, SequenceNumbers) {
    auto node = std::make_shared<FastPathClassifier>("fpc");
    auto in_ch = std::make_shared<SpscChannel<OwnedPacket>>(16);
    auto fast_ch = std::make_shared<SpscChannel<Decision>>(16);
    auto hard_ch = std::make_shared<SpscChannel<HardPathItem>>(16);

    node->in.set_channel(in_ch);
    node->fast_out.set_channel(fast_ch);
    node->hard_out.set_channel(hard_ch);

    in_ch->push(make_packet_ttl0());
    in_ch->push(make_packet_random());
    in_ch->push(make_packet_dns());
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    auto d1 = fast_ch->pop(); // TTL=0
    ASSERT_TRUE(d1.has_value());
    EXPECT_EQ(d1->sequence, 0u);

    auto h1 = hard_ch->pop(); // random
    ASSERT_TRUE(h1.has_value());
    EXPECT_EQ(h1->sequence, 1u);

    auto d2 = fast_ch->pop(); // DNS
    ASSERT_TRUE(d2.has_value());
    EXPECT_EQ(d2->sequence, 2u);
}
