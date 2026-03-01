#ifdef GRAPHOS_ENABLE_DPDK

#include <gtest/gtest.h>
#include "graphos/capture/dpdk_source.hpp"
#include "graphos/nodes/source_node.hpp"

using namespace graphos;

// Compile-time check: SourceNode<DpdkSource> must instantiate
TEST(DpdkCapture, SourceNodeCompiles) {
    // static_assert that the type is well-formed
    static_assert(std::is_constructible_v<
        SourceNode<DpdkSource>, std::string, DpdkSource>);
    SUCCEED();
}

// Integration test: only runs on DPDK hardware with --gtest_also_run_disabled_tests
TEST(DpdkCapture, DISABLED_LiveCapture) {
    // Requires EAL init + bound NIC — run manually:
    //   ./graphos_tests --gtest_also_run_disabled_tests --gtest_filter=DpdkCapture*
    DpdkSource source(0, 32);
    size_t count = 0;
    for (int i = 0; i < 100; ++i) {
        auto pkt = source.next();
        if (pkt.has_value()) ++count;
    }
    // Just verify it doesn't crash — packet count depends on traffic
    EXPECT_GE(count, 0u);
    source.stop();
}

#endif // GRAPHOS_ENABLE_DPDK
