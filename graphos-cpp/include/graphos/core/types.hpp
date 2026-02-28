#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "graphos/core/constants.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define GRAPHOS_LIKELY(x) __builtin_expect(!!(x), 1)
#define GRAPHOS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define GRAPHOS_FORCE_INLINE __attribute__((always_inline)) inline
#define GRAPHOS_RESTRICT __restrict__
#else
#define GRAPHOS_LIKELY(x) (x)
#define GRAPHOS_UNLIKELY(x) (x)
#define GRAPHOS_FORCE_INLINE __forceinline
#define GRAPHOS_RESTRICT __restrict
#endif

namespace graphos {

// ── Zero-copy view into raw packet memory (no ownership) ──
struct PacketView {
    const uint8_t* data = nullptr;
    size_t length = 0;
};

// ── Owning packet buffer — stack-allocated, no heap ──
struct alignas(64) OwnedPacket {
    std::array<uint8_t, TENSOR_DIM> bytes{};
    size_t length = 0;

    GRAPHOS_FORCE_INLINE
    static OwnedPacket from_raw(const uint8_t* GRAPHOS_RESTRICT data, size_t len) noexcept {
        OwnedPacket pkt;
        pkt.length = std::min(len, static_cast<size_t>(TENSOR_DIM));
        std::memcpy(pkt.bytes.data(), data, pkt.length);
        // bytes already zero-initialized, so padding is implicit
        return pkt;
    }

    static OwnedPacket from_raw(const std::vector<uint8_t>& data) noexcept {
        return from_raw(data.data(), data.size());
    }
};

// ── Fixed-capacity batch — avoids heap allocation on hot path ──
struct BatchItem {
    // Fixed buffer: up to DEFAULT_BATCH_SIZE packets, no vector
    std::array<OwnedPacket, DEFAULT_BATCH_SIZE> packets{};
    size_t count = 0;

    GRAPHOS_FORCE_INLINE void push(OwnedPacket&& pkt) noexcept {
        packets[count++] = std::move(pkt);
    }

    GRAPHOS_FORCE_INLINE void clear() noexcept { count = 0; }
    GRAPHOS_FORCE_INLINE bool full() const noexcept {
        return count >= DEFAULT_BATCH_SIZE;
    }
};

// ── Tensor buffer — contiguous floats for zero-copy OV wrapping ──
struct TensorItem {
    // Pre-allocated: batch_size * TENSOR_DIM floats
    std::vector<float> data;
    size_t count = 0;

    TensorItem() = default;

    // Pre-allocate for a given batch size
    explicit TensorItem(size_t batch_size)
        : data(batch_size * TENSOR_DIM, 0.0f), count(0) {}

    GRAPHOS_FORCE_INLINE float* row(size_t i) noexcept {
        return data.data() + i * TENSOR_DIM;
    }
    GRAPHOS_FORCE_INLINE const float* row(size_t i) const noexcept {
        return data.data() + i * TENSOR_DIM;
    }
};

// ── Inference result — class IDs after argmax ──
struct ResultItem {
    std::array<int, DEFAULT_BATCH_SIZE> class_ids{};
    size_t count = 0;
};

// ── Raw logits for pipeline chaining (no argmax) ──
struct RawTensorResult {
    std::vector<float> data;
    size_t count = 0;
    size_t output_dim = 0; // per-row dimension

    RawTensorResult() = default;
    RawTensorResult(size_t batch_size, size_t dim)
        : data(batch_size * dim, 0.0f), count(0), output_dim(dim) {}
};

// ── Conversion functions ──

// Single packet → normalized float[TENSOR_DIM], branchless
GRAPHOS_FORCE_INLINE
void packet_to_tensor(const OwnedPacket& pkt,
                      float* GRAPHOS_RESTRICT output) noexcept {
    constexpr float inv255 = 1.0f / 255.0f;
    for (int i = 0; i < TENSOR_DIM; ++i) {
        output[i] = static_cast<float>(pkt.bytes[i]) * inv255;
    }
}

// Batch: packets → flat float[batch_size * TENSOR_DIM], zero-padded
void packets_to_batch_tensor(const OwnedPacket* GRAPHOS_RESTRICT packets,
                             size_t count, size_t batch_size,
                             float* GRAPHOS_RESTRICT output) noexcept;

// Convenience wrapper
TensorItem make_tensor_item(const OwnedPacket* packets, size_t count,
                            size_t batch_size);

// Argmax: float[batch_size * num_classes] → int[batch_size]
void batch_tensor_to_classes(const float* GRAPHOS_RESTRICT output,
                             size_t batch_size, size_t num_classes,
                             int* GRAPHOS_RESTRICT class_ids) noexcept;

// Single-row argmax
GRAPHOS_FORCE_INLINE
int tensor_to_class(const float* output, size_t num_classes) noexcept {
    int best = 0;
    float best_val = output[0];
    for (size_t i = 1; i < num_classes; ++i) {
        if (output[i] > best_val) {
            best_val = output[i];
            best = static_cast<int>(i);
        }
    }
    return best;
}

#ifdef GRAPHOS_HAS_AVX2
// AVX2: 8 bytes → 8 floats simultaneously, ~4x faster than scalar
void packets_to_tensor_avx2(const OwnedPacket* GRAPHOS_RESTRICT packets,
                            size_t count,
                            float* GRAPHOS_RESTRICT output,
                            size_t tensor_dim) noexcept;
#endif

} // namespace graphos
