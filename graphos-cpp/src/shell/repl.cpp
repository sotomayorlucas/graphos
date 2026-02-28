#include "graphos/shell/repl.hpp"
#include "graphos/core/constants.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <filesystem>

namespace graphos {

GraphOSShell::GraphOSShell(KernelRuntime& runtime, size_t batch_size)
    : runtime_(runtime), batch_size_(batch_size) {
    register_commands();
}

void GraphOSShell::register_commands() {
    commands_["programs"] = [this](auto& a) { cmd_programs(a); };
    commands_["load"]     = [this](auto& a) { cmd_load(a); };
    commands_["unload"]   = [this](auto& a) { cmd_unload(a); };
    commands_["send"]     = [this](auto& a) { cmd_send(a); };
    commands_["send_pcap"]= [this](auto& a) { cmd_send_pcap(a); };
    commands_["run"]      = [this](auto& a) { cmd_run(a); };
    commands_["pipe"]     = [this](auto& a) { cmd_pipe(a); };
    commands_["run_pipe"] = [this](auto& a) { cmd_run_pipe(a); };
    commands_["compare"]  = [this](auto& a) { cmd_compare(a); };
    commands_["inspect"]  = [this](auto& a) { cmd_inspect(a); };
    commands_["health"]   = [this](auto& a) { cmd_health(a); };
    commands_["stats"]    = [this](auto& a) { cmd_stats(a); };
    commands_["help"]     = [this](auto& a) { cmd_help(a); };
    commands_["quit"]     = [](auto&) {};
    commands_["exit"]     = [](auto&) {};
}

void GraphOSShell::cmdloop() {
    std::cout << "=== GraphOS Shell ===\n"
              << "Type 'help' for commands.\n\n";

    std::string line;
    while (true) {
        std::cout << "graphos> " << std::flush;
        if (!std::getline(std::cin, line)) break;

        // Trim
        auto start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // Split command and args
        auto space = line.find(' ');
        std::string cmd = line.substr(0, space);
        std::string args = (space != std::string::npos)
            ? line.substr(space + 1) : "";

        if (cmd == "quit" || cmd == "exit") {
            std::cout << "Goodbye.\n";
            return;
        }

        auto it = commands_.find(cmd);
        if (it != commands_.end()) {
            try {
                it->second(args);
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << '\n';
            }
        } else {
            std::cout << "Unknown command: " << cmd
                      << ". Type 'help' for available commands.\n";
        }
    }
}

ProgramSpec GraphOSShell::spec_for(const std::string& name) const {
    int b = static_cast<int>(batch_size_);
    if (name == "classifier") {
        return {"classifier", "models/router_graph_b64.onnx",
                {b, TENSOR_DIM}, {b, NUM_CLASSES},
                "Packet classifier (HTTP/DNS/OTHER)"};
    }
    if (name == "route_table") {
        return {"route_table", "models/route_table_b64.onnx",
                {b, TENSOR_DIM}, {b, NUM_ROUTES},
                "Route table (LOCAL/FORWARD/DROP/MONITOR)"};
    }
    if (name == "composed_router") {
        return {"composed_router", "models/composed_router_b64.onnx",
                {b, COMPOSED_INPUT_DIM}, {b, NUM_ROUTES},
                "Composed router (classifier+route_table)"};
    }
    throw std::runtime_error("Unknown program: " + name);
}

void GraphOSShell::generate_packets(const std::string& type, size_t count) {
    packets_.clear();
    labels_.clear();
    std::mt19937 rng(42);

    for (size_t i = 0; i < count; ++i) {
        OwnedPacket pkt{};
        std::string label;

        if (type == "http") {
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_TCP;
            pkt.bytes[OFFSET_DST_PORT] = 0;
            pkt.bytes[OFFSET_DST_PORT + 1] = 80;
            pkt.length = TENSOR_DIM;
            label = "http";
        } else if (type == "dns") {
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_UDP;
            pkt.bytes[OFFSET_DST_PORT] = 0;
            pkt.bytes[OFFSET_DST_PORT + 1] = 53;
            pkt.length = TENSOR_DIM;
            label = "dns";
        } else if (type == "other") {
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_ICMP;
            pkt.length = TENSOR_DIM;
            label = "other";
        } else { // random
            std::uniform_int_distribution<int> dist(0, 255);
            for (int j = 0; j < TENSOR_DIM; ++j)
                pkt.bytes[j] = static_cast<uint8_t>(dist(rng));
            pkt.length = TENSOR_DIM;
            label = "random";
        }

        packets_.push_back(std::move(pkt));
        labels_.push_back(label);
    }
}

void GraphOSShell::cmd_programs(const std::string&) {
    auto progs = runtime_.programs();
    if (progs.empty()) {
        std::cout << "No programs loaded.\n";
        return;
    }
    for (const auto& name : progs) {
        std::cout << "  " << name << '\n';
    }
}

void GraphOSShell::cmd_load(const std::string& args) {
    if (args.empty()) { std::cout << "Usage: load <name>\n"; return; }
    auto spec = spec_for(args);
    if (!std::filesystem::exists(spec.onnx_path)) {
        std::cout << "ONNX file not found: " << spec.onnx_path << '\n';
        return;
    }
    runtime_.load(spec);
    std::cout << "Loaded: " << args << '\n';
}

void GraphOSShell::cmd_unload(const std::string& args) {
    if (args.empty()) { std::cout << "Usage: unload <name>\n"; return; }
    runtime_.unload(args);
    std::cout << "Unloaded: " << args << '\n';
}

void GraphOSShell::cmd_send(const std::string& args) {
    std::istringstream ss(args);
    std::string type;
    size_t count = 8;
    ss >> type;
    if (type.empty()) {
        std::cout << "Usage: send <http|dns|other|random> [count]\n";
        return;
    }
    ss >> count;

    generate_packets(type, count);
    std::cout << "Generated " << packets_.size() << " " << type
              << " packets\n";
}

void GraphOSShell::cmd_send_pcap(const std::string& args) {
    if (args.empty()) { std::cout << "Usage: send_pcap <file>\n"; return; }
    // Use PcapSource to load packets
    std::cout << "Loading pcap: " << args << '\n';
    // Simplified: direct pcap loading would be here
    std::cout << "(pcap loading requires libpcap linkage)\n";
}

void GraphOSShell::cmd_run(const std::string& args) {
    if (args.empty()) { std::cout << "Usage: run <program>\n"; return; }
    if (packets_.empty()) { std::cout << "No packets. Use 'send' first.\n"; return; }
    if (!runtime_.has(args)) { std::cout << "Program not loaded: " << args << '\n'; return; }

    size_t count = std::min(packets_.size(), batch_size_);

    // Build tensor
    std::vector<float> tensor(batch_size_ * TENSOR_DIM, 0.0f);
    packets_to_batch_tensor(packets_.data(), count, batch_size_, tensor.data());

    // Determine output dim
    auto spec = spec_for(args);
    size_t out_dim = spec.output_shape.back();
    std::vector<float> output(batch_size_ * out_dim, 0.0f);

    runtime_.execute(args, tensor.data(), output.data(), batch_size_);

    // Print results
    const auto& name_map = (out_dim == NUM_CLASSES) ? CLASS_NAMES : ROUTE_NAMES;
    std::vector<int> results(count);
    batch_tensor_to_classes(output.data(), count, out_dim, results.data());

    std::cout << std::left << std::setw(6) << "#"
              << std::setw(10) << "Label"
              << std::setw(12) << "Result"
              << "Logits\n";
    std::cout << std::string(50, '-') << '\n';

    for (size_t i = 0; i < count; ++i) {
        auto it = name_map.find(results[i]);
        std::string result_name = (it != name_map.end()) ? it->second : "?";

        std::cout << std::setw(6) << i
                  << std::setw(10) << labels_[i]
                  << std::setw(12) << result_name;

        // Print logits
        const float* logits = output.data() + i * out_dim;
        for (size_t j = 0; j < out_dim; ++j) {
            std::cout << std::fixed << std::setprecision(3) << logits[j];
            if (j + 1 < out_dim) std::cout << " ";
        }
        std::cout << '\n';
    }

    last_results_[args] = std::move(output);
    exec_history_.emplace_back(args, count);
}

void GraphOSShell::cmd_pipe(const std::string& args) {
    std::istringstream ss(args);
    std::string first, second;
    ss >> first >> second;
    if (first.empty() || second.empty()) {
        std::cout << "Usage: pipe <first_program> <second_program>\n";
        return;
    }

    auto first_spec = spec_for(first);
    auto second_spec = spec_for(second);

    auto concat = make_concat_adapter(
        batch_size_, TENSOR_DIM, first_spec.output_shape.back());

    pipeline_ = std::make_unique<ProgramPipeline>();
    pipeline_->add_program(first, first_spec)
              .add_adapter(std::move(concat))
              .add_program(second, second_spec)
              .with_raw_passthrough("raw");

    auto errors = pipeline_->validate();
    if (!errors.empty()) {
        for (const auto& e : errors) std::cout << "  " << e << '\n';
        pipeline_.reset();
        return;
    }
    std::cout << "Pipeline created: " << pipeline_->describe() << '\n';
}

void GraphOSShell::cmd_run_pipe(const std::string&) {
    if (!pipeline_) { std::cout << "No pipeline. Use 'pipe' first.\n"; return; }
    if (packets_.empty()) { std::cout << "No packets. Use 'send' first.\n"; return; }

    size_t count = std::min(packets_.size(), batch_size_);
    std::vector<float> tensor(batch_size_ * TENSOR_DIM, 0.0f);
    packets_to_batch_tensor(packets_.data(), count, batch_size_, tensor.data());

    auto output = pipeline_->execute(runtime_, tensor.data(),
                                      tensor.size(), batch_size_);

    size_t out_dim = NUM_ROUTES;
    std::vector<int> results(count);
    batch_tensor_to_classes(output.data(), count, out_dim, results.data());

    std::cout << std::left << std::setw(6) << "#"
              << std::setw(10) << "Label"
              << std::setw(12) << "Route"
              << "Scores\n";
    std::cout << std::string(50, '-') << '\n';

    for (size_t i = 0; i < count; ++i) {
        auto it = ROUTE_NAMES.find(results[i]);
        std::cout << std::setw(6) << i
                  << std::setw(10) << labels_[i]
                  << std::setw(12) << (it != ROUTE_NAMES.end() ? it->second : "?");
        const float* scores = output.data() + i * out_dim;
        for (size_t j = 0; j < out_dim; ++j) {
            std::cout << std::fixed << std::setprecision(3) << scores[j];
            if (j + 1 < out_dim) std::cout << " ";
        }
        std::cout << '\n';
    }
}

void GraphOSShell::cmd_compare(const std::string&) {
    if (!pipeline_) { std::cout << "No pipeline.\n"; return; }
    if (packets_.empty()) { std::cout << "No packets.\n"; return; }

    size_t count = std::min(packets_.size(), batch_size_);
    std::vector<float> tensor(batch_size_ * TENSOR_DIM, 0.0f);
    packets_to_batch_tensor(packets_.data(), count, batch_size_, tensor.data());

    // Run classifier
    std::vector<float> class_out(batch_size_ * NUM_CLASSES, 0.0f);
    runtime_.execute("classifier", tensor.data(), class_out.data(), batch_size_);

    // Run route_table standalone
    std::vector<float> route_out(batch_size_ * NUM_ROUTES, 0.0f);
    runtime_.execute("route_table", tensor.data(), route_out.data(), batch_size_);

    // Run pipeline
    auto pipe_out = pipeline_->execute(runtime_, tensor.data(),
                                        tensor.size(), batch_size_);

    std::vector<int> classes(count), routes(count), composed(count);
    batch_tensor_to_classes(class_out.data(), count, NUM_CLASSES, classes.data());
    batch_tensor_to_classes(route_out.data(), count, NUM_ROUTES, routes.data());
    batch_tensor_to_classes(pipe_out.data(), count, NUM_ROUTES, composed.data());

    size_t matches = 0;
    std::cout << std::left << std::setw(6) << "#"
              << std::setw(12) << "Class"
              << std::setw(12) << "Route(solo)"
              << std::setw(12) << "Route(pipe)"
              << "Match\n";
    std::cout << std::string(50, '-') << '\n';

    for (size_t i = 0; i < count; ++i) {
        auto cn = CLASS_NAMES.find(classes[i]);
        auto rn = ROUTE_NAMES.find(routes[i]);
        auto pn = ROUTE_NAMES.find(composed[i]);
        bool match = routes[i] == composed[i];
        if (match) matches++;

        std::cout << std::setw(6) << i
                  << std::setw(12) << (cn != CLASS_NAMES.end() ? cn->second : "?")
                  << std::setw(12) << (rn != ROUTE_NAMES.end() ? rn->second : "?")
                  << std::setw(12) << (pn != ROUTE_NAMES.end() ? pn->second : "?")
                  << (match ? "=" : "*") << '\n';
    }
    std::cout << "Agreement: " << std::fixed << std::setprecision(1)
              << (100.0 * matches / count) << "%\n";
}

void GraphOSShell::cmd_inspect(const std::string&) {
    if (!pipeline_) { std::cout << "No pipeline.\n"; return; }
    std::cout << pipeline_->describe() << '\n';
}

void GraphOSShell::cmd_health(const std::string&) {
    auto h = runtime_.health();
    std::cout << "Device: " << h.device << '\n'
              << "Programs: " << h.programs.size() << '\n'
              << "Executions: " << h.exec_count << '\n'
              << "Mean latency: " << std::fixed << std::setprecision(1)
              << h.mean_latency_us << " us\n"
              << "Last latency: " << h.last_latency_us << " us\n"
              << "Errors: " << h.errors << '\n'
              << "Healthy: " << (h.healthy ? "yes" : "no") << '\n';
}

void GraphOSShell::cmd_stats(const std::string&) {
    if (exec_history_.empty()) {
        std::cout << "No executions yet.\n";
        return;
    }
    for (size_t i = 0; i < exec_history_.size(); ++i) {
        std::cout << i << ": " << exec_history_[i].first
                  << " (" << exec_history_[i].second << " packets)\n";
    }
}

void GraphOSShell::cmd_help(const std::string&) {
    std::cout <<
        "Commands:\n"
        "  programs            List loaded programs\n"
        "  load <name>         Load program (classifier/route_table/composed_router)\n"
        "  unload <name>       Unload program\n"
        "  send <type> [n]     Generate packets (http/dns/other/random)\n"
        "  send_pcap <file>    Load packets from pcap\n"
        "  run <program>       Run program on loaded packets\n"
        "  pipe <p1> <p2>      Create pipeline\n"
        "  run_pipe            Run pipeline\n"
        "  compare             Compare standalone vs pipeline\n"
        "  inspect             Show pipeline\n"
        "  health              Runtime health\n"
        "  stats               Execution history\n"
        "  help                This message\n"
        "  quit                Exit\n";
}

} // namespace graphos
