#include "graphos/kernel/runtime.hpp"
#include "graphos/kernel/loop.hpp"
#include "graphos/shell/repl.hpp"
#include "graphos/core/constants.hpp"
#include "graphos/core/latency_histogram.hpp"
#include "graphos/core/types.hpp"
#include "graphos/gpnpu/pipeline.hpp"
#include "graphos/actions/counter.hpp"
#include "graphos/actions/log_action.hpp"
#ifdef GRAPHOS_HAS_PCAP
#include "graphos/capture/pcap_source.hpp"
#endif
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <csignal>

namespace {

void print_usage() {
    std::cout <<
        "GraphOS — NPU Packet Classifier\n\n"
        "Usage: graphos <command> [options]\n\n"
        "Commands:\n"
        "  demo      Run classification demo\n"
        "  shell     Interactive REPL\n"
        "  bench     Benchmark inference\n"
        "  gpnpu     GPNPU fast-path + hard-path demo\n"
#ifdef GRAPHOS_HAS_PCAP
        "  pcap      Classify packets from .pcap file or live capture\n"
        "  ifaces    List available capture interfaces\n"
#endif
        "\nOptions:\n"
        "  --model-dir <dir>    Model directory (default: models/)\n"
        "  --batch-size <n>     Batch size (default: 64)\n"
        "  --device <dev>       Device: NPU|CPU|GPU (default: NPU)\n"
        "  --n-packets <n>      Packets for demo/bench (default: 1000)\n"
        "  --deadline <us>      GPNPU: hard-batcher deadline in us (default: 150)\n"
        "  --min-fill <n>       GPNPU: min packets before deadline flush (default: 12)\n"
        "  --inflight <n>       GPNPU: async inference slots (default: 3)\n"
        "  --warmup <n>         GPNPU: warmup packets (default: 100)\n"
        "  --ordered            GPNPU: reorder output by sequence\n"
#ifdef GRAPHOS_HAS_PCAP
        "  --pcap <file>        Read packets from .pcap file\n"
        "  --iface <name>       Live capture from interface (requires Npcap)\n"
        "  --filter <bpf>       BPF filter for live capture\n"
        "  --log                Log each classified packet\n"
#endif
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
    // pcap options
    std::string pcap_file;
    std::string iface;
    std::string bpf_filter;
    bool log_packets = false;
    // GPNPU options
    size_t deadline_us = 150;
    size_t min_fill = 12;
    size_t inflight = 3;
    size_t warmup = 100;
    bool ordered = false;
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
        else if (arg == "--deadline" && i + 1 < argc)
            opts.deadline_us = std::stoull(argv[++i]);
        else if (arg == "--min-fill" && i + 1 < argc)
            opts.min_fill = std::stoull(argv[++i]);
        else if (arg == "--inflight" && i + 1 < argc)
            opts.inflight = std::stoull(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc)
            opts.warmup = std::stoull(argv[++i]);
        else if (arg == "--ordered")
            opts.ordered = true;
        else if (arg == "--pcap" && i + 1 < argc)
            opts.pcap_file = argv[++i];
        else if (arg == "--iface" && i + 1 < argc)
            opts.iface = argv[++i];
        else if (arg == "--filter" && i + 1 < argc)
            opts.bpf_filter = argv[++i];
        else if (arg == "--log")
            opts.log_packets = true;
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

void run_gpnpu(const Options& opts) {
    using namespace graphos;

    std::cout << "=== GraphOS GPNPU: CPU Fast-Path + NPU Hard-Path ===\n";
    std::cout << "Device: " << opts.device << '\n'
              << "Batch size: " << opts.batch_size << '\n'
              << "Deadline: " << opts.deadline_us << " us\n"
              << "Min fill: " << opts.min_fill << '\n'
              << "Inflight slots: " << opts.inflight << '\n'
              << "Warmup: " << opts.warmup << '\n'
              << "Ordered: " << (opts.ordered ? "yes" : "no") << '\n'
              << "Packets: " << opts.n_packets << "\n\n";

    // Generate mixed synthetic traffic:
    // 1) TTL=0 → DROP (fast path)
    // 2) UDP:53 → DNS/LOCAL (fast path)
    // 3) TCP:80 → HTTP/FORWARD (fast path)
    // 4) TCP:443 → HTTP/FORWARD (fast path)
    // 5) Random → hard path (NPU)
    size_t total_gen = opts.warmup + opts.n_packets;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<OwnedPacket> packets;
    packets.reserve(total_gen);

    for (size_t i = 0; i < total_gen; ++i) {
        OwnedPacket pkt{};
        for (int j = 0; j < TENSOR_DIM; ++j)
            pkt.bytes[j] = static_cast<uint8_t>(dist(rng));
        pkt.length = TENSOR_DIM;

        switch (i % 5) {
        case 0: // TTL=0 → DROP
            pkt.bytes[OFFSET_TTL] = 0;
            break;
        case 1: // UDP:53 → DNS
            pkt.bytes[OFFSET_TTL] = 64;
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_UDP;
            pkt.bytes[OFFSET_DST_PORT] = 0;
            pkt.bytes[OFFSET_DST_PORT + 1] = 53;
            break;
        case 2: // TCP:80 → HTTP
            pkt.bytes[OFFSET_TTL] = 64;
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_TCP;
            pkt.bytes[OFFSET_DST_PORT] = 0;
            pkt.bytes[OFFSET_DST_PORT + 1] = 80;
            break;
        case 3: // TCP:443 → HTTPS
            pkt.bytes[OFFSET_TTL] = 64;
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_TCP;
            pkt.bytes[OFFSET_DST_PORT] = 1;
            pkt.bytes[OFFSET_DST_PORT + 1] = 187; // 443 = 0x01BB
            break;
        case 4: // Random (hard path)
            pkt.bytes[OFFSET_TTL] = 64;
            pkt.bytes[OFFSET_PROTOCOL] = PROTO_ICMP; // Not TCP/UDP
            break;
        }

        packets.push_back(std::move(pkt));
    }

    // Build GPNPU pipeline
    std::string onnx_path = opts.model_dir + "/router_graph_b64.onnx";

    GpnpuConfig config;
    config.onnx_path = onnx_path;
    config.device = opts.device;
    config.batch_size = opts.batch_size;
    config.min_fill = opts.min_fill;
    config.deadline = std::chrono::microseconds(opts.deadline_us);
    config.num_inflight = opts.inflight;
    config.ordered = opts.ordered;

    // Warmup phase: run warmup packets to prime caches/NPU, discard results
    if (opts.warmup > 0) {
        std::cout << "Warming up with " << opts.warmup << " packets...\n";
        std::vector<OwnedPacket> warmup_pkts(
            packets.begin(), packets.begin() + static_cast<ptrdiff_t>(opts.warmup));

        auto warmup_counter = std::make_shared<CountAction>();
        std::unordered_map<int, std::vector<std::shared_ptr<Action>>> warmup_actions;
        warmup_actions[CLASS_HTTP] = {warmup_counter};
        warmup_actions[CLASS_DNS] = {warmup_counter};
        warmup_actions[CLASS_OTHER] = {warmup_counter};

        auto warmup_source = std::make_shared<SourceNode<OwnedPacketVectorSource>>(
            "warmup_source", OwnedPacketVectorSource(warmup_pkts));
        auto warmup_pipeline = build_gpnpu_pipeline(
            warmup_source, config, std::move(warmup_actions));
        Scheduler warmup_sched(warmup_pipeline.graph);
        warmup_sched.run();
        std::cout << "Warmup done.\n\n";
    }

    // Measurement phase: only measure the non-warmup packets
    std::vector<OwnedPacket> measure_pkts(
        packets.begin() + static_cast<ptrdiff_t>(opts.warmup), packets.end());

    auto counter = std::make_shared<CountAction>();
    std::unordered_map<int, std::vector<std::shared_ptr<Action>>> class_actions;
    class_actions[CLASS_HTTP] = {counter};
    class_actions[CLASS_DNS] = {counter};
    class_actions[CLASS_OTHER] = {counter};

    auto source = std::make_shared<SourceNode<OwnedPacketVectorSource>>(
        "source", OwnedPacketVectorSource(measure_pkts));

    auto pipeline = build_gpnpu_pipeline(
        source, config, std::move(class_actions));

    // Run
    Scheduler scheduler(pipeline.graph);
    auto metrics = scheduler.run();

    // Report
    double wall = scheduler.wall_time();
    std::cout << "\n--- GPNPU Results ---\n";
    uint64_t fast = pipeline.classifier->fast_count();
    uint64_t hard = pipeline.classifier->hard_count();
    uint64_t total = fast + hard;

    std::cout << "Total packets: " << total << '\n'
              << "Fast path:     " << fast << " ("
              << std::fixed << std::setprecision(1)
              << (total > 0 ? 100.0 * fast / total : 0.0) << "%)\n"
              << "Hard path:     " << hard << " ("
              << std::fixed << std::setprecision(1)
              << (total > 0 ? 100.0 * hard / total : 0.0) << "%)\n\n";

    std::cout << "Dispatcher: fast=" << pipeline.dispatcher->fast_path_count()
              << " hard=" << pipeline.dispatcher->hard_path_count() << '\n';

    std::cout << "\nPer-class counts:\n";
    auto summary = counter->summary();
    for (auto& [cid, cnt] : summary) {
        auto it = CLASS_NAMES.find(cid);
        std::string name = it != CLASS_NAMES.end() ? it->second : "?";
        std::cout << "  " << name << ": " << cnt << '\n';
    }

    // Throughput
    std::cout << "\n--- Throughput ---\n"
              << "Wall time:      " << std::fixed << std::setprecision(4) << wall << " s\n"
              << "Total:          " << std::fixed << std::setprecision(0)
              << (wall > 0 ? static_cast<double>(total) / wall : 0.0) << " pkt/s\n"
              << "Fast path:      " << std::fixed << std::setprecision(0)
              << (wall > 0 ? static_cast<double>(fast) / wall : 0.0) << " pkt/s\n"
              << "Hard path:      " << std::fixed << std::setprecision(0)
              << (wall > 0 ? static_cast<double>(hard) / wall : 0.0) << " pkt/s\n";

    // Latency percentiles
    auto print_latency = [](const char* label, const LatencyHistogram& h) {
        if (h.count() == 0) {
            std::cout << label << " (no samples)\n";
            return;
        }
        std::cout << label
                  << "p50=" << std::fixed << std::setprecision(1) << h.p50()
                  << "  p95=" << h.p95()
                  << "  p99=" << h.p99()
                  << "  mean=" << h.mean()
                  << "  min=" << h.min()
                  << "  max=" << h.max()
                  << "  n=" << h.count() << '\n';
    };

    std::cout << "\n--- Latency (us) ---\n";
    print_latency("Overall: ", pipeline.dispatcher->overall_latency());
    print_latency("Fast:    ", pipeline.dispatcher->fast_latency());
    print_latency("Hard:    ", pipeline.dispatcher->hard_latency());

    std::cout << '\n';
    Scheduler::print_metrics(metrics, wall);
}

#ifdef GRAPHOS_HAS_PCAP
// Global flag for Ctrl+C in live capture
std::atomic<bool> g_stop_capture{false};

void run_pcap(const Options& opts) {
    using namespace graphos;

    bool live = !opts.iface.empty();
    bool offline = !opts.pcap_file.empty();

    if (!live && !offline) {
        std::cerr << "pcap command requires --pcap <file> or --iface <name>\n";
        return;
    }

    std::cout << "=== GraphOS Pcap Classifier ===\n";

    if (live) {
        std::cout << "Mode: Live capture on '" << opts.iface << "'\n";
        if (!opts.bpf_filter.empty())
            std::cout << "Filter: " << opts.bpf_filter << '\n';
        std::cout << "Press Ctrl+C to stop.\n\n";
    } else {
        std::cout << "Mode: Offline .pcap '" << opts.pcap_file << "'\n\n";
    }

    // Load ONNX model
    KernelRuntime runtime(opts.device);
    std::cout << "Device: " << runtime.device() << '\n';

    int b = static_cast<int>(opts.batch_size);
    ProgramSpec spec{
        "classifier", opts.model_dir + "/router_graph_b64.onnx",
        {b, TENSOR_DIM}, {b, NUM_CLASSES}, "Packet classifier"};
    runtime.load(spec);
    std::cout << "Model loaded.\n\n";

    // Read all packets from source
    std::vector<OwnedPacket> packets;
    packets.reserve(1024);

    if (offline) {
        PcapSource source(opts.pcap_file);
        while (auto pkt = source.next()) {
            packets.push_back(std::move(*pkt));
        }
        std::cout << "Read " << packets.size() << " packets from file.\n";
    } else {
        // Live capture with Ctrl+C handling
        g_stop_capture.store(false, std::memory_order_relaxed);
        auto prev_handler = std::signal(SIGINT, [](int) {
            g_stop_capture.store(true, std::memory_order_release);
        });

        LiveCaptureSource source(opts.iface, opts.bpf_filter);
        size_t max_pkts = opts.n_packets > 0 ? opts.n_packets : 10000;

        while (!g_stop_capture.load(std::memory_order_acquire) &&
               packets.size() < max_pkts) {
            auto pkt = source.next();
            if (!pkt.has_value()) break;
            packets.push_back(std::move(*pkt));
            if (packets.size() % 100 == 0)
                std::cout << "\rCaptured: " << packets.size() << std::flush;
        }
        source.stop();
        std::signal(SIGINT, prev_handler);
        std::cout << "\rCaptured " << packets.size() << " packets.\n";
    }

    if (packets.empty()) {
        std::cout << "No packets to classify.\n";
        return;
    }

    // Build GPNPU pipeline
    GpnpuConfig config;
    config.onnx_path = opts.model_dir + "/router_graph_b64.onnx";
    config.device = opts.device;
    config.batch_size = opts.batch_size;
    config.min_fill = 4;
    config.deadline = std::chrono::microseconds(opts.deadline_us);
    config.num_inflight = opts.inflight;
    config.ordered = opts.ordered;

    auto counter = std::make_shared<CountAction>();
    std::shared_ptr<LogAction> logger;

    std::unordered_map<int, std::vector<std::shared_ptr<Action>>> class_actions;
    class_actions[CLASS_HTTP] = {counter};
    class_actions[CLASS_DNS] = {counter};
    class_actions[CLASS_OTHER] = {counter};

    if (opts.log_packets) {
        logger = std::make_shared<LogAction>(true);
        class_actions[CLASS_HTTP].push_back(logger);
        class_actions[CLASS_DNS].push_back(logger);
        class_actions[CLASS_OTHER].push_back(logger);
    }

    auto source = std::make_shared<SourceNode<OwnedPacketVectorSource>>(
        "pcap_source", OwnedPacketVectorSource(packets));

    auto pipeline = build_gpnpu_pipeline(
        source, config, std::move(class_actions));

    // Run
    std::cout << "\nClassifying " << packets.size() << " packets...\n";
    Scheduler scheduler(pipeline.graph);
    auto metrics = scheduler.run();

    if (logger) logger->close();

    // Report
    double wall = scheduler.wall_time();
    uint64_t fast = pipeline.classifier->fast_count();
    uint64_t hard = pipeline.classifier->hard_count();
    uint64_t total = fast + hard;

    std::cout << "\n--- Classification Results ---\n"
              << "Total packets:  " << total << '\n'
              << "Fast path:      " << fast << " ("
              << std::fixed << std::setprecision(1)
              << (total > 0 ? 100.0 * fast / total : 0.0) << "%)\n"
              << "Hard path:      " << hard << " ("
              << std::fixed << std::setprecision(1)
              << (total > 0 ? 100.0 * hard / total : 0.0) << "%)\n\n";

    std::cout << "Per-class breakdown:\n";
    auto summary = counter->summary();
    uint64_t total_counted = 0;
    for (auto& [cid, cnt] : summary) {
        auto it = CLASS_NAMES.find(cid);
        std::string name = it != CLASS_NAMES.end() ? it->second : "class_" + std::to_string(cid);
        std::cout << "  " << std::setw(10) << std::left << name
                  << ": " << cnt << " ("
                  << std::fixed << std::setprecision(1)
                  << (total > 0 ? 100.0 * cnt / total : 0.0) << "%)\n";
        total_counted += cnt;
    }

    std::cout << "\n--- Performance ---\n"
              << "Wall time:   " << std::fixed << std::setprecision(4) << wall << " s\n"
              << "Throughput:  " << std::fixed << std::setprecision(0)
              << (wall > 0 ? static_cast<double>(total) / wall : 0.0) << " pkt/s\n";

    // Latency
    auto print_latency = [](const char* label, const LatencyHistogram& h) {
        if (h.count() == 0) return;
        std::cout << label
                  << "p50=" << std::fixed << std::setprecision(1) << h.p50()
                  << "  p95=" << h.p95()
                  << "  p99=" << h.p99()
                  << "  mean=" << h.mean() << " us"
                  << "  (n=" << h.count() << ")\n";
    };

    std::cout << "\n--- Latency ---\n";
    print_latency("Overall: ", pipeline.dispatcher->overall_latency());
    print_latency("Fast:    ", pipeline.dispatcher->fast_latency());
    print_latency("Hard:    ", pipeline.dispatcher->hard_latency());

    std::cout << '\n';
    Scheduler::print_metrics(metrics, wall);
}
#endif

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
        else if (opts.command == "gpnpu") run_gpnpu(opts);
#ifdef GRAPHOS_HAS_PCAP
        else if (opts.command == "pcap") run_pcap(opts);
        else if (opts.command == "ifaces") {
            auto devs = graphos::list_capture_devices();
            std::cout << "Available capture interfaces:\n\n";
            for (auto& d : devs) {
                std::cout << "  " << d.name << '\n';
                if (!d.description.empty())
                    std::cout << "    " << d.description << '\n';
            }
        }
#endif
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
