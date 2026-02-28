#pragma once
#include "graphos/capture/source.hpp"
#include <memory>
#include <string>

namespace graphos {

// ── PcapSource — offline .pcap file reader via libpcap ──
class PcapSource : public CaptureSource {
public:
    explicit PcapSource(const std::string& path);
    ~PcapSource() override;

    PcapSource(PcapSource&&) noexcept;
    PcapSource& operator=(PcapSource&&) noexcept;

    std::optional<OwnedPacket> next() override;
    void stop() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ── LiveCaptureSource — live capture via libpcap ──
class LiveCaptureSource : public CaptureSource {
public:
    explicit LiveCaptureSource(const std::string& iface,
                                const std::string& bpf_filter = "",
                                int count = 0);
    ~LiveCaptureSource() override;

    std::optional<OwnedPacket> next() override;
    void stop() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace graphos
