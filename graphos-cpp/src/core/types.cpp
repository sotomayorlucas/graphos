#include "graphos/core/types.hpp"
#include <cstring>

#ifdef GRAPHOS_HAS_AVX2
#include <immintrin.h>
#endif

namespace graphos {

void packets_to_batch_tensor(const OwnedPacket* GRAPHOS_RESTRICT packets,
                             size_t count, size_t batch_size,
                             float* GRAPHOS_RESTRICT output) noexcept {
#ifdef GRAPHOS_HAS_AVX2
    packets_to_tensor_avx2(packets, count, output, TENSOR_DIM);
    // Zero-pad remaining rows
    if (count < batch_size) {
        std::memset(output + count * TENSOR_DIM, 0,
                    (batch_size - count) * TENSOR_DIM * sizeof(float));
    }
#else
    constexpr float inv255 = 1.0f / 255.0f;
    for (size_t row = 0; row < count; ++row) {
        float* dst = output + row * TENSOR_DIM;
        const auto& bytes = packets[row].bytes;
        for (int col = 0; col < TENSOR_DIM; ++col) {
            dst[col] = static_cast<float>(bytes[col]) * inv255;
        }
    }
    // Zero-pad remaining rows
    if (count < batch_size) {
        std::memset(output + count * TENSOR_DIM, 0,
                    (batch_size - count) * TENSOR_DIM * sizeof(float));
    }
#endif
}

TensorItem make_tensor_item(const OwnedPacket* packets, size_t count,
                            size_t batch_size) {
    TensorItem item(batch_size);
    item.count = count;
    packets_to_batch_tensor(packets, count, batch_size, item.data.data());
    return item;
}

void batch_tensor_to_classes(const float* GRAPHOS_RESTRICT output,
                             size_t batch_size, size_t num_classes,
                             int* GRAPHOS_RESTRICT class_ids) noexcept {
    for (size_t row = 0; row < batch_size; ++row) {
        const float* logits = output + row * num_classes;
        int best = 0;
        float best_val = logits[0];
        for (size_t c = 1; c < num_classes; ++c) {
            if (logits[c] > best_val) {
                best_val = logits[c];
                best = static_cast<int>(c);
            }
        }
        class_ids[row] = best;
    }
}

#ifdef GRAPHOS_HAS_AVX2
void packets_to_tensor_avx2(const OwnedPacket* GRAPHOS_RESTRICT packets,
                            size_t count,
                            float* GRAPHOS_RESTRICT output,
                            size_t tensor_dim) noexcept {
    const __m256 inv255_vec = _mm256_set1_ps(1.0f / 255.0f);
    const __m256i zero = _mm256_setzero_si256();

    for (size_t row = 0; row < count; ++row) {
        const uint8_t* src = packets[row].bytes.data();
        float* dst = output + row * tensor_dim;

        // Process 8 bytes at a time: uint8 → int32 → float → float/255
        size_t col = 0;
        for (; col + 8 <= tensor_dim; col += 8) {
            // Load 8 bytes into low 64 bits of __m128i
            __m128i bytes8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(src + col));

            // Zero-extend uint8 → uint16 (128-bit)
            __m128i words = _mm_unpacklo_epi8(bytes8, _mm_setzero_si128());

            // Zero-extend uint16 → uint32 (256-bit)
            __m256i dwords = _mm256_cvtepu16_epi32(words);

            // Convert int32 → float32
            __m256 floats = _mm256_cvtepi32_ps(dwords);

            // Multiply by 1/255
            __m256 normalized = _mm256_mul_ps(floats, inv255_vec);

            // Store 8 floats
            _mm256_storeu_ps(dst + col, normalized);
        }

        // Scalar tail (tensor_dim=64 is perfectly divisible by 8, but safety)
        for (; col < tensor_dim; ++col) {
            dst[col] = static_cast<float>(src[col]) * (1.0f / 255.0f);
        }
    }
}
#endif

} // namespace graphos
