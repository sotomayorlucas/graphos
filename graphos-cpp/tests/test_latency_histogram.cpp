#include <gtest/gtest.h>
#include "graphos/core/latency_histogram.hpp"
#include <cmath>

using namespace graphos;

TEST(LatencyHistogram, EmptyReturnsZero) {
    LatencyHistogram h;
    EXPECT_EQ(h.count(), 0u);
    EXPECT_DOUBLE_EQ(h.p50(), 0.0);
    EXPECT_DOUBLE_EQ(h.p95(), 0.0);
    EXPECT_DOUBLE_EQ(h.p99(), 0.0);
    EXPECT_DOUBLE_EQ(h.mean(), 0.0);
    EXPECT_DOUBLE_EQ(h.min(), 0.0);
    EXPECT_DOUBLE_EQ(h.max(), 0.0);
}

TEST(LatencyHistogram, SingleSample) {
    LatencyHistogram h;
    h.record(42.0);
    EXPECT_EQ(h.count(), 1u);
    EXPECT_DOUBLE_EQ(h.p50(), 42.0);
    EXPECT_DOUBLE_EQ(h.p95(), 42.0);
    EXPECT_DOUBLE_EQ(h.p99(), 42.0);
    EXPECT_DOUBLE_EQ(h.mean(), 42.0);
    EXPECT_DOUBLE_EQ(h.min(), 42.0);
    EXPECT_DOUBLE_EQ(h.max(), 42.0);
}

TEST(LatencyHistogram, TwoSamples) {
    LatencyHistogram h;
    h.record(10.0);
    h.record(20.0);
    EXPECT_EQ(h.count(), 2u);
    EXPECT_DOUBLE_EQ(h.mean(), 15.0);
    EXPECT_DOUBLE_EQ(h.min(), 10.0);
    EXPECT_DOUBLE_EQ(h.max(), 20.0);
}

TEST(LatencyHistogram, PercentilesOnUniformDistribution) {
    LatencyHistogram h(100);
    // Insert values 0..99
    for (int i = 0; i < 100; ++i) {
        h.record(static_cast<double>(i));
    }
    EXPECT_EQ(h.count(), 100u);

    // p50 should be around index 49 (value ~49)
    double p50 = h.p50();
    EXPECT_GE(p50, 45.0);
    EXPECT_LE(p50, 55.0);

    // p95 should be around index 94 (value ~94)
    double p95 = h.p95();
    EXPECT_GE(p95, 90.0);
    EXPECT_LE(p95, 99.0);

    // p99 should be around index 98 (value ~98)
    double p99 = h.p99();
    EXPECT_GE(p99, 95.0);
    EXPECT_LE(p99, 99.0);

    EXPECT_DOUBLE_EQ(h.min(), 0.0);
    EXPECT_DOUBLE_EQ(h.max(), 99.0);
    EXPECT_NEAR(h.mean(), 49.5, 0.01);
}

TEST(LatencyHistogram, LargeHistogram) {
    LatencyHistogram h(10000);
    for (int i = 0; i < 10000; ++i) {
        h.record(static_cast<double>(i));
    }
    EXPECT_EQ(h.count(), 10000u);
    EXPECT_DOUBLE_EQ(h.min(), 0.0);
    EXPECT_DOUBLE_EQ(h.max(), 9999.0);
    EXPECT_NEAR(h.mean(), 4999.5, 0.01);

    // p50 ~ 4999
    double p50 = h.p50();
    EXPECT_GE(p50, 4900.0);
    EXPECT_LE(p50, 5100.0);
}

TEST(LatencyHistogram, ReserveDoesNotAffectCount) {
    LatencyHistogram h(1000);
    EXPECT_EQ(h.count(), 0u);
    h.record(1.0);
    EXPECT_EQ(h.count(), 1u);
}

TEST(LatencyHistogram, NonDestructivePercentile) {
    // Calling percentile multiple times should give same results
    LatencyHistogram h;
    for (int i = 0; i < 50; ++i) {
        h.record(static_cast<double>(i));
    }
    double first = h.p50();
    double second = h.p50();
    EXPECT_DOUBLE_EQ(first, second);
}
