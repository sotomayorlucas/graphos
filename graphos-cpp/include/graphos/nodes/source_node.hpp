#pragma once
#include "graphos/dataflow/node.hpp"
#include "graphos/dataflow/port.hpp"
#include "graphos/core/types.hpp"
#include <functional>
#include <stop_token>

namespace graphos {

// ── SourceNode — emits OwnedPackets from a CaptureSource or generator ──
// Templated on source type for zero-overhead: no virtual dispatch in hot loop.
template <typename Source>
class SourceNode : public Node {
    Source source_;

public:
    OutputPort<OwnedPacket> out{"out"};

    SourceNode(std::string name, Source source)
        : Node(std::move(name)), source_(std::move(source)) {}

    void process(std::stop_token st) override {
        // stop_callback calls source_.stop() when scheduler requests stop,
        // breaking streaming sources' infinite poll loops (e.g. DpdkSource).
        std::stop_callback on_stop(st, [this]{ source_.stop(); });
        while (!st.stop_requested()) {
            auto pkt = source_.next();
            if (!pkt.has_value()) break;
            out.put(std::move(*pkt));
            items_processed_.fetch_add(1, std::memory_order_relaxed);
        }
        out.close();
    }
};

// ── Iterable source adapter — wraps any container/range into CaptureSource API ──
template <typename Iter>
class IterableSource {
    Iter cur_, end_;
public:
    IterableSource(Iter begin, Iter end) : cur_(begin), end_(end) {}

    std::optional<OwnedPacket> next() {
        if (cur_ == end_) return std::nullopt;
        auto pkt = OwnedPacket::from_raw(cur_->data(), cur_->size());
        ++cur_;
        return pkt;
    }

    void stop() {} // no-op for iterables
};

// ── Function source — calls a callable to get packets ──
class FunctionSource {
    std::function<std::optional<OwnedPacket>()> fn_;
public:
    explicit FunctionSource(std::function<std::optional<OwnedPacket>()> fn)
        : fn_(std::move(fn)) {}

    std::optional<OwnedPacket> next() { return fn_(); }
    void stop() {}
};

// ── OwnedPacket vector source — wraps a vector<OwnedPacket> (copies on next()) ──
class OwnedPacketVectorSource {
    const std::vector<OwnedPacket>& pkts_;
    size_t idx_ = 0;
public:
    explicit OwnedPacketVectorSource(const std::vector<OwnedPacket>& pkts)
        : pkts_(pkts) {}

    std::optional<OwnedPacket> next() {
        if (idx_ >= pkts_.size()) return std::nullopt;
        return pkts_[idx_++];
    }

    void stop() { idx_ = pkts_.size(); }
};

} // namespace graphos
