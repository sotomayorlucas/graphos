#include <gtest/gtest.h>
#include "graphos/capture/pcap_source.hpp"
#include <filesystem>

using namespace graphos;

// PcapSource tests require a .pcap file. Skip if not found.
class PcapSourceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Try to find test pcap
        if (!std::filesystem::exists(pcap_path_)) {
            GTEST_SKIP() << "Test pcap not found: " << pcap_path_;
        }
    }
    std::string pcap_path_ = []() -> std::string {
        for (auto& p : {"tests/fixtures/test.pcap",
                        "../tests/fixtures/test.pcap",
                        "../../tests/fixtures/test.pcap",
                        "../../../tests/fixtures/test.pcap"}) {
            if (std::filesystem::exists(p)) return p;
        }
        return "tests/fixtures/test.pcap";
    }();
};

TEST_F(PcapSourceTest, ReadsPackets) {
    PcapSource source(pcap_path_);
    size_t count = 0;
    while (auto pkt = source.next()) {
        EXPECT_GT(pkt->length, 0u);
        EXPECT_LE(pkt->length, static_cast<size_t>(TENSOR_DIM));
        count++;
    }
    EXPECT_GT(count, 0u);
}

TEST_F(PcapSourceTest, StopEarly) {
    PcapSource source(pcap_path_);
    source.next(); // read one
    source.stop();
    auto pkt = source.next();
    EXPECT_FALSE(pkt.has_value());
}

TEST(PcapSource, NonexistentFile) {
    EXPECT_THROW(PcapSource("nonexistent.pcap"), std::runtime_error);
}
