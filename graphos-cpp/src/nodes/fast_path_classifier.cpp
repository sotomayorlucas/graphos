#include "graphos/nodes/fast_path_classifier.hpp"
#include "graphos/core/constants.hpp"
#include <chrono>

namespace graphos {

void FastPathClassifier::process(std::stop_token st) {
    while (!st.stop_requested()) {
        auto pkt = in.get();
        if (!pkt.has_value()) {
            fast_out.close();
            hard_out.close();
            return;
        }

        uint64_t seq = sequence_.fetch_add(1, std::memory_order_relaxed);
        auto now = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        const auto& bytes = pkt->bytes;

        // Rule 1: TTL == 0 → DROP
        if (bytes[OFFSET_TTL] == 0) {
            Decision d;
            d.packet = std::move(*pkt);
            d.class_id = CLASS_OTHER;
            d.route_id = ROUTE_DROP;
            d.path = PathKind::FAST_PATH;
            d.confidence = 1.0f;
            d.sequence = seq;
            d.ingress_ns = now;
            fast_out.put(std::move(d));
            fast_count_.fetch_add(1, std::memory_order_relaxed);
            items_processed_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        uint8_t proto = bytes[OFFSET_PROTOCOL];
        uint16_t dst_port = (static_cast<uint16_t>(bytes[OFFSET_DST_PORT]) << 8) |
                             static_cast<uint16_t>(bytes[OFFSET_DST_PORT + 1]);

        // Rule 2: UDP:53 → DNS, LOCAL
        if (proto == PROTO_UDP && dst_port == DNS_PORT) {
            Decision d;
            d.packet = std::move(*pkt);
            d.class_id = CLASS_DNS;
            d.route_id = ROUTE_LOCAL;
            d.path = PathKind::FAST_PATH;
            d.confidence = 1.0f;
            d.sequence = seq;
            d.ingress_ns = now;
            fast_out.put(std::move(d));
            fast_count_.fetch_add(1, std::memory_order_relaxed);
            items_processed_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Rule 3: TCP:{80,443,8080,8443} → HTTP, FORWARD
        if (proto == PROTO_TCP && is_http_port(dst_port)) {
            Decision d;
            d.packet = std::move(*pkt);
            d.class_id = CLASS_HTTP;
            d.route_id = ROUTE_FORWARD;
            d.path = PathKind::FAST_PATH;
            d.confidence = 1.0f;
            d.sequence = seq;
            d.ingress_ns = now;
            fast_out.put(std::move(d));
            fast_count_.fetch_add(1, std::memory_order_relaxed);
            items_processed_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // No rule matched → hard path (NPU)
        HardPathItem item;
        item.packet = std::move(*pkt);
        item.sequence = seq;
        item.ingress_ns = now;
        hard_out.put(std::move(item));
        hard_count_.fetch_add(1, std::memory_order_relaxed);
        items_processed_.fetch_add(1, std::memory_order_relaxed);
    }

    fast_out.close();
    hard_out.close();
}

} // namespace graphos
