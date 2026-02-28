#ifdef GRAPHOS_ENABLE_DPDK

#include "graphos/capture/dpdk_source.hpp"
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <atomic>
#include <stdexcept>

namespace graphos {

struct DpdkSource::Impl {
    uint16_t port_id;
    uint16_t burst_size;
    struct rte_mempool* mbuf_pool = nullptr;
    std::atomic<bool> stopped{false};

    // Pre-allocated burst buffer
    std::vector<struct rte_mbuf*> rx_bufs;
    size_t buf_idx = 0;   // current index in rx_bufs
    size_t buf_count = 0; // packets available in current burst

    Impl(uint16_t port, uint16_t burst)
        : port_id(port), burst_size(burst),
          rx_bufs(burst) {
        // Create mempool
        mbuf_pool = rte_pktmbuf_pool_create("GRAPHOS_POOL", 8192,
                                             256, 0,
                                             RTE_MBUF_DEFAULT_BUF_SIZE,
                                             rte_socket_id());
        if (!mbuf_pool)
            throw std::runtime_error("rte_pktmbuf_pool_create failed");

        // Configure port
        struct rte_eth_conf port_conf{};
        if (rte_eth_dev_configure(port_id, 1, 0, &port_conf) < 0)
            throw std::runtime_error("rte_eth_dev_configure failed");

        if (rte_eth_rx_queue_setup(port_id, 0, 1024,
                                    rte_eth_dev_socket_id(port_id),
                                    nullptr, mbuf_pool) < 0)
            throw std::runtime_error("rte_eth_rx_queue_setup failed");

        if (rte_eth_dev_start(port_id) < 0)
            throw std::runtime_error("rte_eth_dev_start failed");

        rte_eth_promiscuous_enable(port_id);
    }

    ~Impl() {
        rte_eth_dev_stop(port_id);
        // Free any remaining mbufs
        for (size_t i = buf_idx; i < buf_count; ++i) {
            rte_pktmbuf_free(rx_bufs[i]);
        }
    }
};

DpdkSource::DpdkSource(uint16_t port_id, uint16_t burst_size)
    : impl_(std::make_unique<Impl>(port_id, burst_size)) {}

DpdkSource::~DpdkSource() = default;

void DpdkSource::init_eal(int argc, char** argv) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        throw std::runtime_error("rte_eal_init failed");
}

std::optional<OwnedPacket> DpdkSource::next() {
    if (impl_->stopped.load(std::memory_order_acquire))
        return std::nullopt;

    // If we've consumed all packets from current burst, do another rx_burst
    while (impl_->buf_idx >= impl_->buf_count) {
        if (impl_->stopped.load(std::memory_order_relaxed))
            return std::nullopt;

        impl_->buf_count = rte_eth_rx_burst(
            impl_->port_id, 0,
            impl_->rx_bufs.data(), impl_->burst_size);
        impl_->buf_idx = 0;

        // If no packets, poll again (busy-wait — this is the DPDK model)
    }

    // Extract packet from mbuf — zero-copy read
    struct rte_mbuf* mbuf = impl_->rx_bufs[impl_->buf_idx++];
    auto* data = rte_pktmbuf_mtod(mbuf, const uint8_t*);
    size_t len = rte_pktmbuf_pkt_len(mbuf);

    auto pkt = OwnedPacket::from_raw(data, len);
    rte_pktmbuf_free(mbuf);
    return pkt;
}

void DpdkSource::stop() {
    impl_->stopped.store(true, std::memory_order_release);
}

} // namespace graphos

#endif // GRAPHOS_ENABLE_DPDK
