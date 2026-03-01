#include <gtest/gtest.h>
#include "graphos/nodes/npu_executor_async.hpp"
#include "graphos/core/constants.hpp"
#include "graphos/dataflow/channel.hpp"
#include <stop_token>
#include <filesystem>

using namespace graphos;

namespace {

// Check if model file exists — skip test if not available
bool model_available() {
    // Try common paths
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

NpuBatchItem make_test_batch(size_t count) {
    NpuBatchItem batch;
    batch.count = count;
    batch.padded_to = DEFAULT_BATCH_SIZE;
    for (size_t i = 0; i < count; ++i) {
        batch.items[i].packet.bytes.fill(static_cast<uint8_t>(i % 256));
        batch.items[i].packet.length = TENSOR_DIM;
        batch.items[i].sequence = i;
    }
    return batch;
}

} // namespace

TEST(NpuExecutorAsync, ProcessesBatch) {
    if (!model_available()) GTEST_SKIP() << "ONNX model not found";

    auto model_path = find_model();
    // Use CPU device for testing (NPU may not be available in CI)
    auto node = std::make_shared<NpuExecutorAsync>(
        "npu_exec", model_path, "CPU", DEFAULT_BATCH_SIZE, 2);

    auto in_ch = std::make_shared<SpscChannel<NpuBatchItem>>(8);
    auto out_ch = std::make_shared<SpscChannel<Decision>>(256);
    node->in.set_channel(in_ch);
    node->out.set_channel(out_ch);

    // Send one batch with 4 packets
    in_ch->push(make_test_batch(4));
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    // Should get 4 individual Decisions
    for (size_t i = 0; i < 4; ++i) {
        auto d = out_ch->pop();
        ASSERT_TRUE(d.has_value()) << "Missing decision " << i;
        EXPECT_EQ(d->path, PathKind::HARD_PATH);
        EXPECT_GE(d->class_id, 0);
        EXPECT_LT(d->class_id, NUM_CLASSES);
        EXPECT_EQ(d->sequence, i);
    }

    EXPECT_FALSE(out_ch->pop().has_value());
}

TEST(NpuExecutorAsync, MultipleBatches) {
    if (!model_available()) GTEST_SKIP() << "ONNX model not found";

    auto model_path = find_model();
    auto node = std::make_shared<NpuExecutorAsync>(
        "npu_exec", model_path, "CPU", DEFAULT_BATCH_SIZE, 3);

    auto in_ch = std::make_shared<SpscChannel<NpuBatchItem>>(8);
    auto out_ch = std::make_shared<SpscChannel<Decision>>(512);
    node->in.set_channel(in_ch);
    node->out.set_channel(out_ch);

    // Send 3 batches
    in_ch->push(make_test_batch(DEFAULT_BATCH_SIZE));
    in_ch->push(make_test_batch(DEFAULT_BATCH_SIZE));
    in_ch->push(make_test_batch(10));
    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    // Should get 64 + 64 + 10 = 138 decisions
    size_t count = 0;
    while (auto d = out_ch->pop()) {
        EXPECT_EQ(d->path, PathKind::HARD_PATH);
        EXPECT_GE(d->class_id, 0);
        EXPECT_LT(d->class_id, NUM_CLASSES);
        ++count;
    }
    EXPECT_EQ(count, 2u * DEFAULT_BATCH_SIZE + 10u);
}

TEST(NpuExecutorAsync, EmptyInput) {
    if (!model_available()) GTEST_SKIP() << "ONNX model not found";

    auto model_path = find_model();
    auto node = std::make_shared<NpuExecutorAsync>(
        "npu_exec", model_path, "CPU", DEFAULT_BATCH_SIZE, 2);

    auto in_ch = std::make_shared<SpscChannel<NpuBatchItem>>(8);
    auto out_ch = std::make_shared<SpscChannel<Decision>>(16);
    node->in.set_channel(in_ch);
    node->out.set_channel(out_ch);

    in_ch->close();

    std::stop_source ss;
    node->run(ss.get_token());

    EXPECT_FALSE(out_ch->pop().has_value());
}
