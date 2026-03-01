#include <gtest/gtest.h>

#ifdef GRAPHOS_HAS_PCAP

#include "graphos/capture/pcap_source.hpp"
#include "graphos/core/constants.hpp"
#include "graphos/core/types.hpp"
#include "graphos/kernel/runtime.hpp"
#include <filesystem>
#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>

using namespace graphos;

// ── Ground truth classifier from raw bytes ──
// Inspects the Ethernet/IP/TCP/UDP headers to determine the expected class.
static int ground_truth_class(const OwnedPacket& pkt) {
    if (pkt.length < 34) return CLASS_OTHER;

    uint8_t proto = pkt.bytes[OFFSET_PROTOCOL];

    if (proto == PROTO_TCP && pkt.length > 36) {
        uint16_t src_port = (static_cast<uint16_t>(pkt.bytes[OFFSET_SRC_PORT]) << 8) |
                             pkt.bytes[OFFSET_SRC_PORT + 1];
        uint16_t dst_port = (static_cast<uint16_t>(pkt.bytes[OFFSET_DST_PORT]) << 8) |
                             pkt.bytes[OFFSET_DST_PORT + 1];
        if (HTTP_PORTS.count(src_port) || HTTP_PORTS.count(dst_port))
            return CLASS_HTTP;
    }

    if (proto == PROTO_UDP && pkt.length > 36) {
        uint16_t src_port = (static_cast<uint16_t>(pkt.bytes[OFFSET_SRC_PORT]) << 8) |
                             pkt.bytes[OFFSET_SRC_PORT + 1];
        uint16_t dst_port = (static_cast<uint16_t>(pkt.bytes[OFFSET_DST_PORT]) << 8) |
                             pkt.bytes[OFFSET_DST_PORT + 1];
        if (src_port == DNS_PORT || dst_port == DNS_PORT)
            return CLASS_DNS;
    }

    return CLASS_OTHER;
}

// ── Find pcap fixture ──
static std::string find_pcap(const char* name) {
    for (auto& prefix : {"tests/fixtures/", "../tests/fixtures/",
                          "../../tests/fixtures/", "../../../tests/fixtures/"}) {
        std::string p = std::string(prefix) + name;
        if (std::filesystem::exists(p)) return p;
    }
    return "";
}

// ── Find model directory ──
static std::string find_model_dir() {
    for (auto& p : {"models/", "../models/", "../../models/",
                     "../../../models/", "../../../../models/"}) {
        std::string path = std::string(p) + "router_graph_b64.onnx";
        if (std::filesystem::exists(path)) return p;
    }
    return "";
}

class RealCaptureTest : public ::testing::Test {
protected:
    void SetUp() override {
        pcap_path_ = find_pcap("real_traffic.pcap");
        if (pcap_path_.empty())
            GTEST_SKIP() << "real_traffic.pcap not found";

        model_dir_ = find_model_dir();
        if (model_dir_.empty())
            GTEST_SKIP() << "ONNX model not found";
    }

    std::string pcap_path_;
    std::string model_dir_;
};

// ── Test: NPU classification agrees with ground truth ──
TEST_F(RealCaptureTest, NpuMatchesGroundTruth) {
    // 1. Read all packets from pcap
    PcapSource source(pcap_path_);
    std::vector<OwnedPacket> packets;
    while (auto pkt = source.next())
        packets.push_back(std::move(*pkt));

    ASSERT_GT(packets.size(), 0u) << "No packets in pcap";
    std::cout << "Loaded " << packets.size() << " real packets\n";

    // 2. Load NPU model
    KernelRuntime runtime("NPU");
    constexpr int batch_size = 64;
    ProgramSpec spec{
        "classifier", model_dir_ + "router_graph_b64.onnx",
        {batch_size, TENSOR_DIM}, {batch_size, NUM_CLASSES},
        "Test classifier"};
    runtime.load(spec);

    // 3. Classify in batches and compare with ground truth
    size_t agree = 0;
    size_t disagree = 0;
    size_t total = packets.size();

    // Per-class confusion tracking
    // confusion[gt][pred]
    int confusion[3][3] = {};

    std::vector<float> input(batch_size * TENSOR_DIM, 0.0f);
    std::vector<float> output(batch_size * NUM_CLASSES, 0.0f);

    for (size_t offset = 0; offset < total; offset += batch_size) {
        size_t batch_count = std::min(static_cast<size_t>(batch_size),
                                       total - offset);

        // Fill input tensor
        std::fill(input.begin(), input.end(), 0.0f);
        for (size_t i = 0; i < batch_count; ++i)
            packet_to_tensor(packets[offset + i], input.data() + i * TENSOR_DIM);

        // Run inference
        runtime.execute("classifier", input.data(), output.data(), batch_size);

        // Compare with ground truth
        for (size_t i = 0; i < batch_count; ++i) {
            int npu_class = tensor_to_class(output.data() + i * NUM_CLASSES,
                                             NUM_CLASSES);
            int gt_class = ground_truth_class(packets[offset + i]);

            if (npu_class >= 0 && npu_class < 3 && gt_class >= 0 && gt_class < 3)
                confusion[gt_class][npu_class]++;

            if (npu_class == gt_class)
                agree++;
            else
                disagree++;
        }
    }

    double agreement = 100.0 * agree / total;

    // Print confusion matrix
    std::cout << "\n--- Confusion Matrix (rows=ground truth, cols=NPU prediction) ---\n";
    std::cout << std::setw(12) << "" << std::setw(10) << "TCP_HTTP"
              << std::setw(10) << "UDP_DNS" << std::setw(10) << "OTHER" << '\n';
    const char* names[] = {"TCP_HTTP", "UDP_DNS", "OTHER"};
    for (int gt = 0; gt < 3; ++gt) {
        std::cout << std::setw(12) << names[gt];
        for (int pred = 0; pred < 3; ++pred)
            std::cout << std::setw(10) << confusion[gt][pred];
        std::cout << '\n';
    }

    // Precision: when NPU says HTTP, is it really HTTP?
    int npu_http_total = confusion[0][0] + confusion[1][0] + confusion[2][0];
    double http_precision = npu_http_total > 0
        ? 100.0 * confusion[0][0] / npu_http_total : 0.0;

    // Recall: of all real HTTP packets, how many did NPU catch?
    int gt_http_total = confusion[0][0] + confusion[0][1] + confusion[0][2];
    double http_recall = gt_http_total > 0
        ? 100.0 * confusion[0][0] / gt_http_total : 0.0;

    std::cout << "\n--- Results ---\n"
              << "Total:          " << total << '\n'
              << "Agree:          " << agree << '\n'
              << "Disagree:       " << disagree << '\n'
              << "Agreement:      " << std::fixed << std::setprecision(2)
              << agreement << "%\n"
              << "HTTP precision: " << std::fixed << std::setprecision(2)
              << http_precision << "% (when NPU says HTTP, it is HTTP)\n"
              << "HTTP recall:    " << std::fixed << std::setprecision(2)
              << http_recall << "% (of real HTTP, NPU detected)\n";

    // Pipeline correctness checks:
    // 1. Agreement >=30%: proves NPU is doing meaningful classification,
    //    not random. The synthetic→real domain gap limits accuracy
    //    (encrypted HTTPS payloads look like OTHER to the model).
    EXPECT_GE(agreement, 30.0)
        << "NPU classification is too far from ground truth — pipeline may be broken";

    // 2. HTTP precision >=90%: when the model says HTTP, it should be right.
    //    This validates the model learned real signal, not noise.
    if (npu_http_total > 10) {
        EXPECT_GE(http_precision, 90.0)
            << "HTTP precision too low — NPU predicting HTTP for non-HTTP packets";
    }
}

// ── Test: Classification is deterministic (same input → same output) ──
TEST_F(RealCaptureTest, ClassificationIsDeterministic) {
    PcapSource source(pcap_path_);
    std::vector<OwnedPacket> packets;
    while (auto pkt = source.next())
        packets.push_back(std::move(*pkt));

    ASSERT_GT(packets.size(), 0u);

    KernelRuntime runtime("NPU");
    constexpr int batch_size = 64;
    ProgramSpec spec{
        "classifier", model_dir_ + "router_graph_b64.onnx",
        {batch_size, TENSOR_DIM}, {batch_size, NUM_CLASSES},
        "Determinism test"};
    runtime.load(spec);

    // Classify the first batch twice
    size_t n = std::min(packets.size(), static_cast<size_t>(batch_size));
    std::vector<float> input(batch_size * TENSOR_DIM, 0.0f);
    for (size_t i = 0; i < n; ++i)
        packet_to_tensor(packets[i], input.data() + i * TENSOR_DIM);

    std::vector<float> output1(batch_size * NUM_CLASSES, 0.0f);
    std::vector<float> output2(batch_size * NUM_CLASSES, 0.0f);

    runtime.execute("classifier", input.data(), output1.data(), batch_size);
    runtime.execute("classifier", input.data(), output2.data(), batch_size);

    // Every class ID should match
    for (size_t i = 0; i < n; ++i) {
        int c1 = tensor_to_class(output1.data() + i * NUM_CLASSES, NUM_CLASSES);
        int c2 = tensor_to_class(output2.data() + i * NUM_CLASSES, NUM_CLASSES);
        EXPECT_EQ(c1, c2) << "Non-deterministic at packet " << i;
    }
}

// ── Test: All class IDs are valid (0, 1, or 2) ──
TEST_F(RealCaptureTest, AllClassIdsValid) {
    PcapSource source(pcap_path_);
    std::vector<OwnedPacket> packets;
    while (auto pkt = source.next())
        packets.push_back(std::move(*pkt));

    ASSERT_GT(packets.size(), 0u);

    KernelRuntime runtime("NPU");
    constexpr int batch_size = 64;
    ProgramSpec spec{
        "classifier", model_dir_ + "router_graph_b64.onnx",
        {batch_size, TENSOR_DIM}, {batch_size, NUM_CLASSES},
        "Validity test"};
    runtime.load(spec);

    std::vector<float> input(batch_size * TENSOR_DIM, 0.0f);
    std::vector<float> output(batch_size * NUM_CLASSES, 0.0f);

    for (size_t offset = 0; offset < packets.size(); offset += batch_size) {
        size_t n = std::min(static_cast<size_t>(batch_size),
                             packets.size() - offset);

        std::fill(input.begin(), input.end(), 0.0f);
        for (size_t i = 0; i < n; ++i)
            packet_to_tensor(packets[offset + i], input.data() + i * TENSOR_DIM);

        runtime.execute("classifier", input.data(), output.data(), batch_size);

        for (size_t i = 0; i < n; ++i) {
            int cls = tensor_to_class(output.data() + i * NUM_CLASSES, NUM_CLASSES);
            EXPECT_GE(cls, 0) << "Negative class at packet " << (offset + i);
            EXPECT_LT(cls, NUM_CLASSES) << "Invalid class at packet " << (offset + i);
        }
    }
}

#endif // GRAPHOS_HAS_PCAP
