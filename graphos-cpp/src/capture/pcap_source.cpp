#include "graphos/capture/pcap_source.hpp"
#include <pcap/pcap.h>
#include <atomic>
#include <stdexcept>

namespace graphos {

// ── PcapSource (offline) ──

struct PcapSource::Impl {
    pcap_t* handle = nullptr;
    bool done = false;

    explicit Impl(const std::string& path) {
        char errbuf[PCAP_ERRBUF_SIZE];
        handle = pcap_open_offline(path.c_str(), errbuf);
        if (!handle)
            throw std::runtime_error("pcap_open_offline failed: " +
                                     std::string(errbuf));
    }

    ~Impl() {
        if (handle) pcap_close(handle);
    }
};

PcapSource::PcapSource(const std::string& path)
    : impl_(std::make_unique<Impl>(path)) {}

PcapSource::~PcapSource() = default;
PcapSource::PcapSource(PcapSource&&) noexcept = default;
PcapSource& PcapSource::operator=(PcapSource&&) noexcept = default;

std::optional<OwnedPacket> PcapSource::next() {
    if (impl_->done) return std::nullopt;

    struct pcap_pkthdr* hdr;
    const u_char* data;
    int rc = pcap_next_ex(impl_->handle, &hdr, &data);

    if (rc == 1) {
        return OwnedPacket::from_raw(data, hdr->caplen);
    }
    // rc == -2 (EOF) or error
    impl_->done = true;
    return std::nullopt;
}

void PcapSource::stop() {
    impl_->done = true;
}

// ── LiveCaptureSource ──

struct LiveCaptureSource::Impl {
    pcap_t* handle = nullptr;
    std::atomic<bool> stopped{false};

    Impl(const std::string& iface, const std::string& bpf_filter) {
        char errbuf[PCAP_ERRBUF_SIZE];
        handle = pcap_open_live(iface.c_str(), TENSOR_DIM, 0, 100, errbuf);
        if (!handle)
            throw std::runtime_error("pcap_open_live failed: " +
                                     std::string(errbuf));

        if (!bpf_filter.empty()) {
            struct bpf_program fp;
            if (pcap_compile(handle, &fp, bpf_filter.c_str(), 1,
                            PCAP_NETMASK_UNKNOWN) == -1 ||
                pcap_setfilter(handle, &fp) == -1) {
                pcap_freecode(&fp);
                throw std::runtime_error("BPF filter failed");
            }
            pcap_freecode(&fp);
        }
    }

    ~Impl() {
        if (handle) pcap_close(handle);
    }
};

LiveCaptureSource::LiveCaptureSource(const std::string& iface,
                                      const std::string& bpf_filter,
                                      int /*count*/)
    : impl_(std::make_unique<Impl>(iface, bpf_filter)) {}

LiveCaptureSource::~LiveCaptureSource() = default;

std::optional<OwnedPacket> LiveCaptureSource::next() {
    if (impl_->stopped.load(std::memory_order_acquire))
        return std::nullopt;

    struct pcap_pkthdr* hdr;
    const u_char* data;
    int rc = pcap_next_ex(impl_->handle, &hdr, &data);

    if (rc == 1) {
        return OwnedPacket::from_raw(data, hdr->caplen);
    }
    if (rc == 0) {
        // Timeout — try again (non-blocking behavior)
        return next();
    }
    return std::nullopt;
}

void LiveCaptureSource::stop() {
    impl_->stopped.store(true, std::memory_order_release);
    if (impl_->handle) pcap_breakloop(impl_->handle);
}

} // namespace graphos
