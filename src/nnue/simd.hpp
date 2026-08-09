#pragma once

#include <cstddef>
#include <cstdint>

// The four kernels NNUE inference spends all its time in, in a scalar version
// that is always available and a vector version for AVX2 and NEON.
//
// The scalar versions are not dead code kept for exotic hardware: they are the
// definition of what the vector versions have to compute, and the tests check
// the two against each other. A SIMD bug in evaluation does not crash, it just
// makes the engine quietly play worse, which is the hardest kind of bug to
// find in a chess or shogi program.

#if defined(__AVX2__)
#define LUNA_NNUE_AVX2 1
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#define LUNA_NNUE_NEON 1
#include <arm_neon.h>
#endif

namespace luna::nnue {

// AVX2 wants 32; giving NEON the same costs nothing and keeps one number.
constexpr size_t kSimdAlignment = 32;

// Which implementation was compiled in. Reported by "isready" so a build that
// silently fell back to scalar is visible rather than merely slow.
inline constexpr const char* SimdName() {
#if defined(LUNA_NNUE_AVX2)
  return "avx2";
#elif defined(LUNA_NNUE_NEON)
  return "neon";
#else
  return "scalar";
#endif
}

namespace reference {

// acc += row
inline void AddRow(int16_t* acc, const int16_t* row, int dim) {
  for (int i = 0; i < dim; ++i) acc[i] = static_cast<int16_t>(acc[i] + row[i]);
}

// acc -= row
inline void SubRow(int16_t* acc, const int16_t* row, int dim) {
  for (int i = 0; i < dim; ++i) acc[i] = static_cast<int16_t>(acc[i] - row[i]);
}

// The activation, clipped ReLU: clamp to [0, 127], which is the range the
// int8 layers below expect their inputs in.
inline void ClippedRelu(const int16_t* in, uint8_t* out, int dim) {
  for (int i = 0; i < dim; ++i) {
    const int v = in[i];
    out[i] = static_cast<uint8_t>(v < 0 ? 0 : (v > 127 ? 127 : v));
  }
}

// output[o] = bias[o] + sum_i weight[o][i] * input[i], with the weight rows
// laid out contiguously.
inline void Affine(const int8_t* weights,
                   const int32_t* biases,
                   const uint8_t* input,
                   int32_t* output,
                   int in_dim,
                   int out_dim) {
  for (int o = 0; o < out_dim; ++o) {
    const int8_t* row = weights + static_cast<size_t>(o) * in_dim;
    int32_t sum = biases[o];
    for (int i = 0; i < in_dim; ++i) sum += row[i] * input[i];
    output[o] = sum;
  }
}

}  // namespace reference

#if defined(LUNA_NNUE_AVX2)

inline int32_t HorizontalSum(__m256i v) {
  __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
  return _mm_cvtsi128_si32(sum);
}

inline void AddRow(int16_t* acc, const int16_t* row, int dim) {
  int i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_add_epi16(a, b));
  }
  for (; i < dim; ++i) acc[i] = static_cast<int16_t>(acc[i] + row[i]);
}

inline void SubRow(int16_t* acc, const int16_t* row, int dim) {
  int i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_sub_epi16(a, b));
  }
  for (; i < dim; ++i) acc[i] = static_cast<int16_t>(acc[i] - row[i]);
}

inline void ClippedRelu(const int16_t* in, uint8_t* out, int dim) {
  const __m256i zero = _mm256_setzero_si256();
  int i = 0;
  for (; i + 32 <= dim; i += 32) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 16));
    // packs_epi16 saturates to [-128, 127], which does the upper clamp for
    // free, but it packs each 128-bit lane separately; the permute puts the
    // 32 bytes back into the order the weights were trained in.
    __m256i packed = _mm256_packs_epi16(lo, hi);
    packed = _mm256_permute4x64_epi64(packed, 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_max_epi8(packed, zero));
  }
  for (; i < dim; ++i) {
    const int v = in[i];
    out[i] = static_cast<uint8_t>(v < 0 ? 0 : (v > 127 ? 127 : v));
  }
}

inline void Affine(const int8_t* weights,
                   const int32_t* biases,
                   const uint8_t* input,
                   int32_t* output,
                   int in_dim,
                   int out_dim) {
  const __m256i ones = _mm256_set1_epi16(1);
  for (int o = 0; o < out_dim; ++o) {
    const int8_t* row = weights + static_cast<size_t>(o) * in_dim;
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= in_dim; i += 32) {
      const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
      const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
      // Inputs are at most 127 and weights at most 127 in magnitude, so a
      // pair of products cannot leave the int16 range maddubs saturates at.
      acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones));
    }
    int32_t sum = biases[o] + HorizontalSum(acc);
    for (; i < in_dim; ++i) sum += row[i] * input[i];
    output[o] = sum;
  }
}

#elif defined(LUNA_NNUE_NEON)

inline void AddRow(int16_t* acc, const int16_t* row, int dim) {
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    vst1q_s16(acc + i, vaddq_s16(vld1q_s16(acc + i), vld1q_s16(row + i)));
  }
  for (; i < dim; ++i) acc[i] = static_cast<int16_t>(acc[i] + row[i]);
}

inline void SubRow(int16_t* acc, const int16_t* row, int dim) {
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    vst1q_s16(acc + i, vsubq_s16(vld1q_s16(acc + i), vld1q_s16(row + i)));
  }
  for (; i < dim; ++i) acc[i] = static_cast<int16_t>(acc[i] - row[i]);
}

inline void ClippedRelu(const int16_t* in, uint8_t* out, int dim) {
  const int8x16_t zero = vdupq_n_s8(0);
  int i = 0;
  for (; i + 16 <= dim; i += 16) {
    // vqmovn saturates to [-128, 127]; the max then takes the negatives out.
    const int8x16_t packed =
        vcombine_s8(vqmovn_s16(vld1q_s16(in + i)), vqmovn_s16(vld1q_s16(in + i + 8)));
    vst1q_u8(out + i, vreinterpretq_u8_s8(vmaxq_s8(packed, zero)));
  }
  for (; i < dim; ++i) {
    const int v = in[i];
    out[i] = static_cast<uint8_t>(v < 0 ? 0 : (v > 127 ? 127 : v));
  }
}

inline void Affine(const int8_t* weights,
                   const int32_t* biases,
                   const uint8_t* input,
                   int32_t* output,
                   int in_dim,
                   int out_dim) {
  for (int o = 0; o < out_dim; ++o) {
    const int8_t* row = weights + static_cast<size_t>(o) * in_dim;
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 16 <= in_dim; i += 16) {
      // Clipped ReLU leaves every input in [0, 127], so reading the unsigned
      // activations as signed bytes is exact and lets the signed kernels do
      // the work. NEON has no unsigned-times-signed multiply below ARMv8.6.
      const int8x16_t a = vld1q_s8(reinterpret_cast<const int8_t*>(input + i));
      const int8x16_t b = vld1q_s8(row + i);
#if defined(__ARM_FEATURE_DOTPROD)
      acc = vdotq_s32(acc, a, b);
#else
      acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(a), vget_low_s8(b)));
      acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(a), vget_high_s8(b)));
#endif
    }
    int32_t sum = biases[o] + vaddvq_s32(acc);
    for (; i < in_dim; ++i) sum += row[i] * input[i];
    output[o] = sum;
  }
}

#else

using reference::AddRow;
using reference::Affine;
using reference::ClippedRelu;
using reference::SubRow;

#endif

}  // namespace luna::nnue
