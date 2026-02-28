#include <benchmark/benchmark.h>
#include "graphos/dataflow/channel.hpp"
#include "graphos/core/types.hpp"
#include <thread>

using namespace graphos;

// ── SPSC throughput: push+pop in tight loop ──
static void BM_SpscThroughput(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    SpscChannel<int> ch(1024);

    for (auto _ : state) {
        std::thread producer([&]() {
            for (size_t i = 0; i < n; ++i)
                ch.push(static_cast<int>(i));
            ch.close();
        });

        size_t count = 0;
        while (auto v = ch.pop()) {
            benchmark::DoNotOptimize(*v);
            count++;
        }

        producer.join();
        benchmark::DoNotOptimize(count);

        // Reset channel for next iteration
        ch = SpscChannel<int>(1024);
    }

    state.SetItemsProcessed(static_cast<int64_t>(n) * state.iterations());
}
BENCHMARK(BM_SpscThroughput)->Arg(10000)->Arg(100000)->Arg(1000000);

// ── OwnedPacket transfer through channel ──
static void BM_SpscPacketTransfer(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    SpscChannel<OwnedPacket> ch(1024);

    for (auto _ : state) {
        std::thread producer([&]() {
            for (size_t i = 0; i < n; ++i) {
                OwnedPacket pkt{};
                pkt.bytes[0] = static_cast<uint8_t>(i);
                pkt.length = TENSOR_DIM;
                ch.push(std::move(pkt));
            }
            ch.close();
        });

        size_t count = 0;
        while (auto pkt = ch.pop()) {
            benchmark::DoNotOptimize(pkt->bytes[0]);
            count++;
        }

        producer.join();
        ch = SpscChannel<OwnedPacket>(1024);
    }

    state.SetItemsProcessed(static_cast<int64_t>(n) * state.iterations());
    state.SetBytesProcessed(static_cast<int64_t>(n) * TENSOR_DIM *
                            state.iterations());
}
BENCHMARK(BM_SpscPacketTransfer)->Arg(10000)->Arg(100000);

// ── Tensor normalization (scalar vs AVX2) ──
static void BM_PacketToTensor_Scalar(benchmark::State& state) {
    const size_t batch_size = static_cast<size_t>(state.range(0));
    std::vector<OwnedPacket> pkts(batch_size);
    for (auto& p : pkts) {
        p.bytes.fill(128);
        p.length = TENSOR_DIM;
    }
    std::vector<float> output(batch_size * TENSOR_DIM);

    for (auto _ : state) {
        packets_to_batch_tensor(pkts.data(), batch_size, batch_size,
                                output.data());
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(batch_size) *
                            state.iterations());
    state.SetBytesProcessed(static_cast<int64_t>(batch_size) * TENSOR_DIM *
                            state.iterations());
}
BENCHMARK(BM_PacketToTensor_Scalar)->Arg(1)->Arg(16)->Arg(64)->Arg(256);

// ── Argmax throughput ──
static void BM_Argmax(benchmark::State& state) {
    const size_t batch_size = static_cast<size_t>(state.range(0));
    std::vector<float> logits(batch_size * NUM_CLASSES, 0.5f);
    std::vector<int> classes(batch_size);

    for (auto _ : state) {
        batch_tensor_to_classes(logits.data(), batch_size, NUM_CLASSES,
                                classes.data());
        benchmark::DoNotOptimize(classes.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(batch_size) *
                            state.iterations());
}
BENCHMARK(BM_Argmax)->Arg(1)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK_MAIN();
