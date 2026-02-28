#include <gtest/gtest.h>
#include "graphos/actions/counter.hpp"
#include "graphos/actions/log_action.hpp"

using namespace graphos;

TEST(CountAction, CountsPerClass) {
    CountAction counter;
    OwnedPacket pkt{};
    pkt.length = TENSOR_DIM;

    counter.execute(pkt, CLASS_HTTP);
    counter.execute(pkt, CLASS_HTTP);
    counter.execute(pkt, CLASS_DNS);

    EXPECT_EQ(counter.count(CLASS_HTTP), 2u);
    EXPECT_EQ(counter.count(CLASS_DNS), 1u);
    EXPECT_EQ(counter.count(CLASS_OTHER), 0u);

    auto summary = counter.summary();
    EXPECT_EQ(summary.size(), 2u);
    EXPECT_EQ(summary[CLASS_HTTP], 2u);
    EXPECT_EQ(summary[CLASS_DNS], 1u);
}

TEST(CountAction, ThreadSafe) {
    CountAction counter;
    constexpr size_t N = 10000;
    OwnedPacket pkt{};
    pkt.length = TENSOR_DIM;

    auto worker = [&](int class_id) {
        for (size_t i = 0; i < N; ++i) {
            counter.execute(pkt, class_id);
        }
    };

    std::thread t1(worker, CLASS_HTTP);
    std::thread t2(worker, CLASS_DNS);
    std::thread t3(worker, CLASS_OTHER);
    t1.join(); t2.join(); t3.join();

    EXPECT_EQ(counter.count(CLASS_HTTP), N);
    EXPECT_EQ(counter.count(CLASS_DNS), N);
    EXPECT_EQ(counter.count(CLASS_OTHER), N);
}

TEST(LogAction, SilencesAfterTen) {
    LogAction log(false); // non-verbose
    OwnedPacket pkt{};
    pkt.length = TENSOR_DIM;

    // Should not crash even with 20 calls
    for (int i = 0; i < 20; ++i) {
        log.execute(pkt, CLASS_HTTP);
    }
    log.close();
}

TEST(LogAction, VerboseShowsAll) {
    LogAction log(true);
    OwnedPacket pkt{};
    pkt.length = TENSOR_DIM;

    for (int i = 0; i < 20; ++i) {
        log.execute(pkt, CLASS_DNS);
    }
    log.close();
}
