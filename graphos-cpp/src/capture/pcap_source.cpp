#include "graphos/capture/pcap_source.hpp"
#include <pcap/pcap.h>
#include <atomic>
#include <stdexcept>
#include <string>

namespace graphos {

// ── resolve_iface_name — map friendly name (e.g. "Wi-Fi") to NPF device ──
static std::string resolve_iface_name(const std::string& name) {
    // Already an NPF device path — return as-is
    if (name.find("\\Device\\") != std::string::npos ||
        name.find("npf") != std::string::npos) {
        return name;
    }

    // Enumerate pcap devices and match by description
    pcap_if_t* alldevs = nullptr;
    char errbuf[PCAP_ERRBUF_SIZE];
    if (pcap_findalldevs(&alldevs, errbuf) == -1 || !alldevs)
        return name; // fallback to original

    std::string result = name;
    for (pcap_if_t* d = alldevs; d; d = d->next) {
        std::string dev_name = d->name ? d->name : "";
        std::string dev_desc = d->description ? d->description : "";

        // Match by description containing the friendly name, or device name containing it
        if ((!dev_desc.empty() && dev_desc.find(name) != std::string::npos) ||
            (!dev_name.empty() && dev_name.find(name) != std::string::npos)) {
            result = dev_name;
            break;
        }
    }

    pcap_freealldevs(alldevs);
    return result;
}

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
        std::string resolved = resolve_iface_name(iface);
        handle = pcap_open_live(resolved.c_str(), TENSOR_DIM, 0, 100, errbuf);
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

// ── list_capture_devices ──

std::vector<CaptureDevice> list_capture_devices() {
    std::vector<CaptureDevice> result;
    pcap_if_t* alldevs = nullptr;
    char errbuf[PCAP_ERRBUF_SIZE];
    if (pcap_findalldevs(&alldevs, errbuf) == -1 || !alldevs)
        return result;

    for (pcap_if_t* d = alldevs; d; d = d->next) {
        result.push_back({
            d->name ? d->name : "",
            d->description ? d->description : ""
        });
    }
    pcap_freealldevs(alldevs);
    return result;
}

} // namespace graphos
