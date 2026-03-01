#include <gtest/gtest.h>
#include "graphos/gpnpu/pipeline.hpp"
#include "graphos/actions/counter.hpp"
#include "graphos/nodes/source_node.hpp"
#include "graphos/dataflow/scheduler.hpp"
#include <filesystem>

using namespace graphos;

namespace {

bool model_available() {
    for (const auto& path : {
        "models/router_graph_b64.onnx",
        "../models/router_graph_b64.onnx",
        "../../models/router_graph_b64.onnx"}) {
        if (std::filesystem::exists(path)) return true;
    }
    return false;
}

std::string find_model() {
    for (const auto& path : {
        "models/router_graph_b64.onnx",
        "../models/router_graph_b64.onnx",
        "../../models/router_graph_b64.onnx"}) {
        if (std::filesystem::exists(path)) return path;
    }
    return "";
}

std::vector<OwnedPacket> make_mixed_packets(size_t n) {
    std::vector<OwnedPacket> packets;
    packets.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        OwnedPacket pkt{};
        pkt.bytes.fill(static_cast<uint8_t>(i % 256));
        pkt.length = TENSOR_DIM;

        switch (i % 5) {
        case 0: // TTL=0 → DROP
            pkt.bytes[OFFSET_TTL] = 0;
            break;
        case 1: // UDP:53 → DNS
            pkt.bytes[OFFSET_TTL] = 64;
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_UDP;
            pkt.bytes[OFFSET_DST_PORT] = 0;
            pkt.bytes[OFFSET_DST_PORT + 1] = 53;
            break;
        case 2: // TCP:80 → HTTP
            pkt.bytes[OFFSET_TTL] = 64;
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_TCP;
            pkt.bytes[OFFSET_DST_PORT] = 0;
            pkt.bytes[OFFSET_DST_PORT + 1] = 80;
            break;
        case 3: // TCP:443 → HTTPS
            pkt.bytes[OFFSET_TTL] = 64;
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_TCP;
            pkt.bytes[OFFSET_DST_PORT] = 1;
            pkt.bytes[OFFSET_DST_PORT + 1] = 187; // 443 = 0x01BB
            break;
        case 4: // Random (hard path)
            pkt.bytes[OFFSET_TTL] = 64;
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_ICMP;
            break;
        }
        packets.push_back(std::move(pkt));
    }
    return packets;
}

} // namespace

TEST(GpnpuPipeline, FullPipeline) {
    if (!model_available()) GTEST_SKIP() << "ONNX model not found";

    auto model_path = find_model();
    auto packets = make_mixed_packets(100);

    auto counter = std::make_shared<CountAction>();
    std::unordered_map<int, std::vector<std::shared_ptr<Action>>> class_actions;
    class_actions[CLASS_HTTP] = {counter};
    class_actions[CLASS_DNS] = {counter};
    class_actions[CLASS_OTHER] = {counter};

    GpnpuConfig config;
    config.onnx_path = model_path;
    config.device = "CPU";
    config.batch_size = DEFAULT_BATCH_SIZE;
    config.min_fill = 4;
    config.deadline = std::chrono::microseconds{5000};
    config.num_inflight = 2;
    config.ordered = false;

    auto source = std::make_shared<SourceNode<OwnedPacketVectorSource>>(
        "source", OwnedPacketVectorSource(packets));

    auto pipeline = build_gpnpu_pipeline(
        source, config, std::move(class_actions));

    Scheduler scheduler(pipeline.graph);
    auto metrics = scheduler.run();

    // All packets accounted for
    uint64_t fast = pipeline.classifier->fast_count();
    uint64_t hard = pipeline.classifier->hard_count();
    EXPECT_EQ(fast + hard, 100u);

    // With our 5-pattern traffic: 4/5 are fast-path (TTL0, DNS, HTTP, HTTPS)
    EXPECT_EQ(fast, 80u);
    EXPECT_EQ(hard, 20u);

    // Dispatcher received all packets
    uint64_t disp_total = pipeline.dispatcher->fast_path_count() +
                          pipeline.dispatcher->hard_path_count();
    EXPECT_EQ(disp_total, 100u);
}

TEST(GpnpuPipeline, OrderedMode) {
    if (!model_available()) GTEST_SKIP() << "ONNX model not found";

    auto model_path = find_model();
    auto packets = make_mixed_packets(50);

    GpnpuConfig config;
    config.onnx_path = model_path;
    config.device = "CPU";
    config.batch_size = DEFAULT_BATCH_SIZE;
    config.min_fill = 4;
    config.deadline = std::chrono::microseconds{5000};
    config.num_inflight = 2;
    config.ordered = true;

    auto counter = std::make_shared<CountAction>();
    std::unordered_map<int, std::vector<std::shared_ptr<Action>>> class_actions;
    class_actions[CLASS_HTTP] = {counter};
    class_actions[CLASS_DNS] = {counter};
    class_actions[CLASS_OTHER] = {counter};

    auto source = std::make_shared<SourceNode<OwnedPacketVectorSource>>(
        "source", OwnedPacketVectorSource(packets));

    auto pipeline = build_gpnpu_pipeline(
        source, config, std::move(class_actions));

    Scheduler scheduler(pipeline.graph);
    auto metrics = scheduler.run();

    uint64_t total = pipeline.classifier->fast_count() +
                     pipeline.classifier->hard_count();
    EXPECT_EQ(total, 50u);
}

TEST(GpnpuPipeline, FastPathRatio) {
    if (!model_available()) GTEST_SKIP() << "ONNX model not found";

    auto model_path = find_model();

    // All fast-path packets (TTL=0)
    std::vector<OwnedPacket> packets;
    for (int i = 0; i < 50; ++i) {
        OwnedPacket pkt{};
        pkt.bytes.fill(0);
        pkt.bytes[OFFSET_TTL] = 0;
        pkt.length = TENSOR_DIM;
        packets.push_back(std::move(pkt));
    }

    GpnpuConfig config;
    config.onnx_path = model_path;
    config.device = "CPU";
    config.batch_size = DEFAULT_BATCH_SIZE;
    config.min_fill = 4;
    config.deadline = std::chrono::microseconds{5000};
    config.num_inflight = 2;

    auto source = std::make_shared<SourceNode<OwnedPacketVectorSource>>(
        "source", OwnedPacketVectorSource(packets));

    auto pipeline = build_gpnpu_pipeline(source, config);

    Scheduler scheduler(pipeline.graph);
    scheduler.run();

    // All should be fast-path
    EXPECT_EQ(pipeline.classifier->fast_count(), 50u);
    EXPECT_EQ(pipeline.classifier->hard_count(), 0u);
}
