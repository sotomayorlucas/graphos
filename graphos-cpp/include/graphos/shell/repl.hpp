#pragma once
#include "graphos/kernel/runtime.hpp"
#include "graphos/kernel/compose.hpp"
#include "graphos/core/types.hpp"
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace graphos {

// ── GraphOSShell — interactive REPL ──
class GraphOSShell {
    KernelRuntime& runtime_;
    size_t batch_size_;

    // State
    std::vector<OwnedPacket> packets_;
    std::vector<std::string> labels_;
    std::unique_ptr<ProgramPipeline> pipeline_;
    std::unordered_map<std::string, std::vector<float>> last_results_;
    std::vector<std::pair<std::string, size_t>> exec_history_;

    // Command dispatch
    using CmdFn = std::function<void(const std::string& args)>;
    std::unordered_map<std::string, CmdFn> commands_;

    void register_commands();

    // Command implementations
    void cmd_programs(const std::string& args);
    void cmd_load(const std::string& args);
    void cmd_unload(const std::string& args);
    void cmd_send(const std::string& args);
    void cmd_send_pcap(const std::string& args);
    void cmd_run(const std::string& args);
    void cmd_pipe(const std::string& args);
    void cmd_run_pipe(const std::string& args);
    void cmd_compare(const std::string& args);
    void cmd_inspect(const std::string& args);
    void cmd_health(const std::string& args);
    void cmd_stats(const std::string& args);
    void cmd_help(const std::string& args);

    // Helpers
    ProgramSpec spec_for(const std::string& name) const;
    void generate_packets(const std::string& type, size_t count);

public:
    GraphOSShell(KernelRuntime& runtime,
                 size_t batch_size = DEFAULT_BATCH_SIZE);

    // Run the REPL loop (blocks until quit)
    void cmdloop();
};

} // namespace graphos
