#include "graphos/kernel/runtime.hpp"
#include "graphos/kernel/loop.hpp"
#include "graphos/shell/repl.hpp"
#include "graphos/core/constants.hpp"
#include "graphos/core/types.hpp"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <random>

namespace {

void print_usage() {
    std::cout <<
        "GraphOS — NPU Packet Classifier\n\n"
        "Usage: graphos <command> [options]\n\n"
        "Commands:\n"
        "  demo      Run classification demo\n"
        "  shell     Interactive REPL\n"
        "  bench     Benchmark inference\n\n"
        "Options:\n"
        "  --model-dir <dir>    Model directory (default: models/)\n"
        "  --batch-size <n>     Batch size (default: 64)\n"
        "  --device <dev>       Device: NPU|CPU|GPU (default: NPU)\n"
        "  --n-packets <n>      Packets for demo/bench (default: 1000)\n"
#ifdef GRAPHOS_ENABLE_DPDK
        "  --dpdk               Use DPDK capture source\n"
        "  --port <n>           DPDK port ID\n"
#endif
        ;
}

struct Options {
    std::string command;
    std::string model_dir = "models/";
    size_t batch_size = graphos::DEFAULT_BATCH_SIZE;
    std::string device = "NPU";
    size_t n_packets = 1000;
    bool dpdk = false;
    uint16_t dpdk_port = 0;
};

Options parse_args(int argc, char** argv) {
    Options opts;
    if (argc < 2) return opts;

    opts.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc)
            opts.model_dir = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc)
            opts.batch_size = std::stoull(argv[++i]);
        else if (arg == "--device" && i + 1 < argc)
            opts.device = argv[++i];
        else if (arg == "--n-packets" && i + 1 < argc)
            opts.n_packets = std::stoull(argv[++i]);
        else if (arg == "--dpdk")
            opts.dpdk = true;
        else if (arg == "--port" && i + 1 < argc)
            opts.dpdk_port = static_cast<uint16_t>(std::stoi(argv[++i]));
    }
    return opts;
}

void run_demo(const Options& opts) {
    using namespace graphos;

    std::cout << "=== GraphOS Demo ===\n";
    KernelRuntime runtime(opts.device);
    std::cout << "Device: " << runtime.device() << '\n';

    // Load classifier
    int b = static_cast<int>(opts.batch_size);
    ProgramSpec classifier_spec{
        "classifier", opts.model_dir + "/router_graph_b64.onnx",
        {b, TENSOR_DIM}, {b, NUM_CLASSES}, "Packet classifier"};
    ProgramSpec route_spec{
        "route_table", opts.model_dir + "/route_table_b64.onnx",
        {b, TENSOR_DIM}, {b, NUM_ROUTES}, "Route table"};

    runtime.load(classifier_spec);
    std::cout << "Loaded: classifier\n";
    runtime.load(route_spec);
    std::cout << "Loaded: route_table\n";

    // Generate synthetic packets
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<OwnedPacket> packets(opts.n_packets);
    for (auto& pkt : packets) {
        for (int j = 0; j < TENSOR_DIM; ++j)
            pkt.bytes[j] = static_cast<uint8_t>(dist(rng));
        pkt.length = TENSOR_DIM;
    }

    // Run kernel loop
    KernelLoop loop(runtime, opts.batch_size);

    struct VectorSource {
        const std::vector<OwnedPacket>& pkts;
        size_t idx = 0;
        std::optional<OwnedPacket> next() {
            if (idx >= pkts.size()) return std::nullopt;
            return pkts[idx++];
        }
        void stop() { idx = pkts.size(); }
    };

    VectorSource source{packets};
    loop.run(source);

    auto stats = loop.stats();
    std::cout << "\n--- Results ---\n"
              << "Packets: " << stats.packets_processed << '\n'
              << "Batches: " << stats.batches_processed << '\n'
              << "Elapsed: " << std::fixed << std::setprecision(4)
              << stats.elapsed_s << "s\n"
              << "Throughput: " << std::fixed << std::setprecision(0)
              << stats.throughput << " pkt/s\n";
}

void run_shell(const Options& opts) {
    graphos::KernelRuntime runtime(opts.device);
    graphos::GraphOSShell shell(runtime, opts.batch_size);
    shell.cmdloop();
}

void run_bench(const Options& opts) {
    using namespace graphos;

    std::cout << "=== GraphOS Benchmark ===\n";
    KernelRuntime runtime(opts.device);
    std::cout << "Device: " << runtime.device() << '\n';

    int b = static_cast<int>(opts.batch_size);
    ProgramSpec spec{"classifier", opts.model_dir + "/router_graph_b64.onnx",
                     {b, TENSOR_DIM}, {b, NUM_CLASSES}, "Benchmark"};
    runtime.load(spec);

    // Pre-allocate
    std::vector<float> input(opts.batch_size * TENSOR_DIM, 0.5f);
    std::vector<float> output(opts.batch_size * NUM_CLASSES, 0.0f);

    // Warmup
    for (int i = 0; i < 10; ++i)
        runtime.execute("classifier", input.data(), output.data(),
                       opts.batch_size);

    // Benchmark
    size_t iters = opts.n_packets / opts.batch_size;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        runtime.execute("classifier", input.data(), output.data(),
                       opts.batch_size);
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double total_pkts = static_cast<double>(iters * opts.batch_size);

    std::cout << "Iterations: " << iters << '\n'
              << "Batch size: " << opts.batch_size << '\n'
              << "Total packets: " << static_cast<size_t>(total_pkts) << '\n'
              << "Elapsed: " << std::fixed << std::setprecision(4)
              << elapsed << "s\n"
              << "Throughput: " << std::fixed << std::setprecision(0)
              << (total_pkts / elapsed) << " pkt/s\n"
              << "Latency/batch: " << std::fixed << std::setprecision(1)
              << (elapsed / iters * 1e6) << " us\n";

    auto health = runtime.health();
    std::cout << "Mean latency: " << std::fixed << std::setprecision(1)
              << health.mean_latency_us << " us\n";
}

} // anonymous namespace

int main(int argc, char** argv) {
    auto opts = parse_args(argc, argv);

    if (opts.command.empty() || opts.command == "--help" ||
        opts.command == "-h") {
        print_usage();
        return 0;
    }

    try {
        if (opts.command == "demo") run_demo(opts);
        else if (opts.command == "shell") run_shell(opts);
        else if (opts.command == "bench") run_bench(opts);
        else {
            std::cerr << "Unknown command: " << opts.command << '\n';
            print_usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
