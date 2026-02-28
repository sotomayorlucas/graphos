#pragma once
#include "graphos/capture/source.hpp"
#include <atomic>
#include <cstdint>
#include <memory>

#ifdef GRAPHOS_ENABLE_DPDK

namespace graphos {

// ── DpdkSource — kernel-bypass poll-mode driver (Linux only) ──
// Zero-copy from NIC DMA ring. Requires hugepages + EAL init.
class DpdkSource : public CaptureSource {
public:
    // port_id: DPDK port index, burst_size: packets per rx_burst call
    DpdkSource(uint16_t port_id, uint16_t burst_size = 32);
    ~DpdkSource() override;

    std::optional<OwnedPacket> next() override;
    void stop() override;

    // Must be called once before creating DpdkSource instances
    static void init_eal(int argc, char** argv);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace graphos

#endif // GRAPHOS_ENABLE_DPDK
