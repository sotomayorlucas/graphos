#pragma once
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace graphos {

/// Collects latency samples (in microseconds) and computes percentiles via nth_element (O(n)).
class LatencyHistogram {
    std::vector<double> samples_;

public:
    explicit LatencyHistogram(size_t reserve = 0) { samples_.reserve(reserve); }

    void record(double value_us) { samples_.push_back(value_us); }
    size_t count() const noexcept { return samples_.size(); }

    /// Compute the p-th percentile (0-100). Returns 0 if empty.
    double percentile(double p) const {
        if (samples_.empty()) return 0.0;
        auto copy = samples_;  // non-destructive
        if (copy.size() == 1) return copy[0];
        double rank = (p / 100.0) * static_cast<double>(copy.size() - 1);
        size_t idx = static_cast<size_t>(rank);
        if (idx >= copy.size() - 1) idx = copy.size() - 1;
        std::nth_element(copy.begin(), copy.begin() + static_cast<ptrdiff_t>(idx), copy.end());
        return copy[idx];
    }

    double p50() const { return percentile(50); }
    double p95() const { return percentile(95); }
    double p99() const { return percentile(99); }

    double mean() const {
        if (samples_.empty()) return 0.0;
        return std::accumulate(samples_.begin(), samples_.end(), 0.0) /
               static_cast<double>(samples_.size());
    }

    double min() const {
        if (samples_.empty()) return 0.0;
        return *std::min_element(samples_.begin(), samples_.end());
    }

    double max() const {
        if (samples_.empty()) return 0.0;
        return *std::max_element(samples_.begin(), samples_.end());
    }
};

} // namespace graphos
