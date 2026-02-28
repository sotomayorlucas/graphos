#pragma once
#include "graphos/actions/action.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace graphos {

// ── PcapWriteAction — writes packets to per-class .pcap files ──
class PcapWriteAction : public Action {
    std::string output_dir_;

    struct PcapHandle;
    std::unordered_map<int, std::unique_ptr<PcapHandle>> handles_;

    PcapHandle& get_handle(int class_id);

public:
    explicit PcapWriteAction(const std::string& output_dir = "output");
    ~PcapWriteAction() override;

    void execute(const OwnedPacket& packet, int class_id) override;
    void close() noexcept override;
};

} // namespace graphos
