#include "graphos/actions/pcap_writer.hpp"
#include "graphos/core/constants.hpp"
#include <pcap/pcap.h>
#include <filesystem>
#include <cstring>

namespace graphos {

struct PcapWriteAction::PcapHandle {
    pcap_t* dead = nullptr;
    pcap_dumper_t* dumper = nullptr;

    PcapHandle(const std::string& path) {
        dead = pcap_open_dead(DLT_EN10MB, 65535);
        if (!dead) throw std::runtime_error("pcap_open_dead failed");
        dumper = pcap_dump_open(dead, path.c_str());
        if (!dumper) {
            pcap_close(dead);
            throw std::runtime_error("pcap_dump_open failed: " + path);
        }
    }

    ~PcapHandle() {
        if (dumper) pcap_dump_close(dumper);
        if (dead) pcap_close(dead);
    }

    void write(const uint8_t* data, size_t len) {
        struct pcap_pkthdr hdr{};
        hdr.caplen = static_cast<bpf_u_int32>(len);
        hdr.len = static_cast<bpf_u_int32>(len);
        // Timestamp left as zero (offline captures)
        pcap_dump(reinterpret_cast<u_char*>(dumper), &hdr, data);
    }
};

PcapWriteAction::PcapWriteAction(const std::string& output_dir)
    : output_dir_(output_dir) {
    std::filesystem::create_directories(output_dir);
}

PcapWriteAction::~PcapWriteAction() = default;

PcapWriteAction::PcapHandle& PcapWriteAction::get_handle(int class_id) {
    auto it = handles_.find(class_id);
    if (it != handles_.end()) return *it->second;

    auto name_it = CLASS_NAMES.find(class_id);
    std::string label = (name_it != CLASS_NAMES.end())
        ? name_it->second
        : "CLASS_" + std::to_string(class_id);
    std::string path = output_dir_ + "/" + label + ".pcap";

    auto handle = std::make_unique<PcapHandle>(path);
    auto* ptr = handle.get();
    handles_.emplace(class_id, std::move(handle));
    return *ptr;
}

void PcapWriteAction::execute(const OwnedPacket& packet, int class_id) {
    auto& handle = get_handle(class_id);
    handle.write(packet.bytes.data(), packet.length);
}

void PcapWriteAction::close() noexcept {
    handles_.clear(); // destructors flush and close
}

} // namespace graphos
