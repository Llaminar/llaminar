/**
 * @file Test__AVX2VNNIParity.cpp
 * @brief Parity tests proving AVX2 emulated VNNI produces identical results
 *        to native AVX512-VNNI for all GEMV/GEMM kernel paths.
 *
 * Test levels:
 *  1. Intrinsic-level:  avx2_dpbusd_epi32 vs _mm512_dpbusd_epi32
 *  2. Single-chunk GEMV: 64-column chunk output comparison
 *  3. Full GEMV (M=1):  gemv_native_vnni_preq with ISAPath::AVX512 vs AVX2
 *  4. Full GEMM (M>1):  gemm_native_vnni_preq with ISAPath::AVX512 vs AVX2
 *
 * All tests run on AVX512-VNNI hardware, calling both paths explicitly
 * via the ISAPath runtime dispatch enum.
 */

#include <gtest/gtest.h>
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <cmath>

#include "kernels/cpu/native_vnni/VNNIEmulation.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemv.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIWeightPacker.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "utils/CPUFeatures.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::cpu::native_vnni::isa;
using namespace llaminar2;
using llaminar2::test::TestTensorFactory;

// ============================================================================
// Helper functions (free functions for macro accessibility)
// ============================================================================

namespace avx2_parity_helpers
{
    // Create random Q8_1 blocks for activation vectors
    inline std::vector<Q8_1Block> createRandomQ8_1(int K, uint32_t seed = 42)
    {
        int K_blocks = (K + 31) / 32;
        std::vector<Q8_1Block> blocks(K_blocks);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.5f);

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            float scale = scale_dist(rng);
            blocks[kb].d = simd::fp32_to_fp16(scale);
            int32_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                blocks[kb].qs[i] = static_cast<int8_t>(dist(rng));
                sum += blocks[kb].qs[i];
            }
            blocks[kb].sum_qs = static_cast<int16_t>(std::clamp(sum, -32768, 32767));
        }
        return blocks;
    }

    // Pack weights and assert success
    inline CPUNativeVNNIPackedWeights packWeights(const TensorBase *tensor)
    {
        CPUNativeVNNIPackedWeights packed;
        bool ok = packWeightsCPUNativeVNNI(tensor, packed);
        if (!ok)
            throw std::runtime_error("Weight packing failed");
        return packed;
    }

    // Compare two float arrays for exact equality
    inline void assertExactEqual(const float *a, const float *b, int n,
                                 const std::string &label)
    {
        float max_diff = 0.0f;
        int max_idx = -1;
        int mismatches = 0;
        for (int i = 0; i < n; ++i)
        {
            float diff = std::fabs(a[i] - b[i]);
            if (diff > 0.0f)
            {
                mismatches++;
                if (diff > max_diff)
                {
                    max_diff = diff;
                    max_idx = i;
                }
            }
        }
        EXPECT_EQ(mismatches, 0)
            << label << ": " << mismatches << "/" << n
            << " mismatches, max diff=" << max_diff
            << " at index " << max_idx
            << (max_idx >= 0 ? (" (avx512=" + std::to_string(a[max_idx]) +
                                " avx2=" + std::to_string(b[max_idx]) + ")")
                             : "");
    }
} // namespace avx2_parity_helpers

using namespace avx2_parity_helpers;

// ============================================================================
// Test fixture
// ============================================================================

class AVX2VNNIParity : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!cpu_supports_avx512_vnni())
        {
            GTEST_SKIP() << "AVX512-VNNI not available; cannot run parity tests";
        }
    }
};

// ============================================================================
// Level 1: Intrinsic-level dpbusd parity
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

TEST_F(AVX2VNNIParity, DpbusdIntrinsic_ZeroAccumulator)
{
    // Test with zero accumulator — pure dot product
    alignas(64) uint8_t a_data[64];
    alignas(64) int8_t b_data[64];
    std::mt19937 rng(123);

    for (int i = 0; i < 64; ++i)
    {
        a_data[i] = static_cast<uint8_t>(rng() % 256);
        b_data[i] = static_cast<int8_t>((rng() % 256) - 128);
    }

    // AVX512: process full 64 bytes as one ZMM
    __m512i acc512 = _mm512_setzero_si512();
    __m512i a512 = _mm512_loadu_si512(a_data);
    __m512i b512 = _mm512_loadu_si512(b_data);
    acc512 = _mm512_dpbusd_epi32(acc512, a512, b512);

    alignas(64) int32_t result_512[16];
    _mm512_store_si512(result_512, acc512);

    // AVX2: process as two 32-byte YMM halves
    __m256i acc256_lo = _mm256_setzero_si256();
    __m256i acc256_hi = _mm256_setzero_si256();
    __m256i a256_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data));
    __m256i a256_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data + 32));
    __m256i b256_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data));
    __m256i b256_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data + 32));

    acc256_lo = avx2_dpbusd_epi32(acc256_lo, a256_lo, b256_lo);
    acc256_hi = avx2_dpbusd_epi32(acc256_hi, a256_hi, b256_hi);

    alignas(32) int32_t result_256[16];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256), acc256_lo);
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256 + 8), acc256_hi);

    for (int i = 0; i < 16; ++i)
    {
        EXPECT_EQ(result_512[i], result_256[i])
            << "Lane " << i << ": AVX512=" << result_512[i]
            << " AVX2=" << result_256[i];
    }
}

TEST_F(AVX2VNNIParity, DpbusdIntrinsic_WithAccumulator)
{
    // Test with non-zero accumulator
    alignas(64) uint8_t a_data[64];
    alignas(64) int8_t b_data[64];
    alignas(64) int32_t acc_init[16];
    std::mt19937 rng(456);

    for (int i = 0; i < 64; ++i)
    {
        a_data[i] = static_cast<uint8_t>(rng() % 256);
        b_data[i] = static_cast<int8_t>((rng() % 256) - 128);
    }
    for (int i = 0; i < 16; ++i)
    {
        acc_init[i] = static_cast<int32_t>(rng() % 100000) - 50000;
    }

    // AVX512
    __m512i acc512 = _mm512_load_si512(acc_init);
    acc512 = _mm512_dpbusd_epi32(acc512,
                                  _mm512_loadu_si512(a_data),
                                  _mm512_loadu_si512(b_data));
    alignas(64) int32_t result_512[16];
    _mm512_store_si512(result_512, acc512);

    // AVX2
    __m256i acc_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(acc_init));
    __m256i acc_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(acc_init + 8));
    acc_lo = avx2_dpbusd_epi32(acc_lo,
                                _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data)),
                                _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data)));
    acc_hi = avx2_dpbusd_epi32(acc_hi,
                                _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data + 32)),
                                _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data + 32)));
    alignas(32) int32_t result_256[16];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256), acc_lo);
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256 + 8), acc_hi);

    for (int i = 0; i < 16; ++i)
    {
        EXPECT_EQ(result_512[i], result_256[i])
            << "Lane " << i;
    }
}

TEST_F(AVX2VNNIParity, DpbusdIntrinsic_MultipleAccumulations)
{
    // Simulate K-block accumulation: multiple dpbusd calls on the same accumulator
    constexpr int K_ITERS = 16;
    alignas(64) uint8_t a_data[K_ITERS][64];
    alignas(64) int8_t b_data[K_ITERS][64];
    std::mt19937 rng(789);

    for (int k = 0; k < K_ITERS; ++k)
    {
        for (int i = 0; i < 64; ++i)
        {
            a_data[k][i] = static_cast<uint8_t>(rng() % 256);
            b_data[k][i] = static_cast<int8_t>((rng() % 256) - 128);
        }
    }

    // AVX512
    __m512i acc512 = _mm512_setzero_si512();
    for (int k = 0; k < K_ITERS; ++k)
    {
        acc512 = _mm512_dpbusd_epi32(acc512,
                                      _mm512_loadu_si512(a_data[k]),
                                      _mm512_loadu_si512(b_data[k]));
    }
    alignas(64) int32_t result_512[16];
    _mm512_store_si512(result_512, acc512);

    // AVX2
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();
    for (int k = 0; k < K_ITERS; ++k)
    {
        acc_lo = avx2_dpbusd_epi32(acc_lo,
                                    _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data[k])),
                                    _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data[k])));
        acc_hi = avx2_dpbusd_epi32(acc_hi,
                                    _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data[k] + 32)),
                                    _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data[k] + 32)));
    }
    alignas(32) int32_t result_256[16];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256), acc_lo);
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256 + 8), acc_hi);

    for (int i = 0; i < 16; ++i)
    {
        EXPECT_EQ(result_512[i], result_256[i])
            << "Lane " << i << " after " << K_ITERS << " accumulations";
    }
}

// ============================================================================
// Level 2: Single-chunk GEMV parity (64 columns)
// ============================================================================

// Macro to generate chunk-level parity tests for each quant format
#define CHUNK_PARITY_TEST(FORMAT, CREATE_FN, N, K)                                       \
    TEST_F(AVX2VNNIParity, ChunkGEMV_##FORMAT##_##N##x##K)                               \
    {                                                                                      \
        auto weights = TestTensorFactory::CREATE_FN({N, K});                               \
        auto packed = packWeights(weights.get());                                          \
        auto A_q8 = createRandomQ8_1(K, 100);                                             \
                                                                                           \
        alignas(64) float result_512[64] = {};                                             \
        alignas(64) float result_256[64] = {};                                             \
                                                                                           \
        /* AVX512 chunk */                                                                 \
        __m512i lut512 = packed.is_nibble_lut                                              \
                             ? build_decode_lut(packed.codebook_id)                        \
                             : _mm512_setzero_si512();                                     \
        if (packed.is_nibble_lut)                                                          \
            gemv_native_vnni_avx512_chunk_native(packed, A_q8.data(), result_512,          \
                                                 0, 0, packed.blocks_per_row, lut512);     \
        else                                                                               \
            gemv_native_vnni_avx512_chunk_int8(packed, A_q8.data(), result_512,            \
                                               0, 0, packed.blocks_per_row);               \
                                                                                           \
        /* AVX2 chunk */                                                                   \
        __m256i lut256 = packed.is_nibble_lut                                              \
                             ? build_decode_lut_avx2_for_codebook(packed.codebook_id)      \
                             : _mm256_setzero_si256();                                     \
        if (packed.is_nibble_lut)                                                          \
            gemv_avx2_chunk_native(packed, A_q8.data(), result_256,                        \
                                   0, 0, packed.blocks_per_row, lut256);                   \
        else                                                                               \
            gemv_avx2_chunk_int8(packed, A_q8.data(), result_256,                          \
                                 0, 0, packed.blocks_per_row);                             \
                                                                                           \
        assertExactEqual(result_512, result_256, 64,                                       \
                         #FORMAT " chunk GEMV " #N "x" #K);                                \
    }

// Nibble-LUT formats
CHUNK_PARITY_TEST(Q4_0, createQ4_0Random, 64, 256)
CHUNK_PARITY_TEST(Q4_0, createQ4_0Random, 64, 512)
CHUNK_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 64, 256)
CHUNK_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 64, 512)

// INT8 pre-decoded formats
CHUNK_PARITY_TEST(Q5_0, createQ5_0Random, 64, 256)
CHUNK_PARITY_TEST(Q5_0, createQ5_0Random, 64, 512)
CHUNK_PARITY_TEST(Q5_1, createQ5_1Random, 64, 256)

#undef CHUNK_PARITY_TEST

// ============================================================================
// Level 3: Full GEMV (M=1) parity via ISAPath dispatch
// ============================================================================

#define FULL_GEMV_PARITY_TEST(FORMAT, CREATE_FN, N, K, SEED)                              \
    TEST_F(AVX2VNNIParity, FullGEMV_##FORMAT##_##N##x##K)                                 \
    {                                                                                      \
        auto weights = TestTensorFactory::CREATE_FN({N, K}, SEED);                         \
        auto packed = packWeights(weights.get());                                          \
        auto A_q8 = createRandomQ8_1(K, SEED + 1);                                        \
                                                                                           \
        std::vector<float> result_512(N, 0.0f);                                            \
        std::vector<float> result_256(N, 0.0f);                                            \
                                                                                           \
        gemv_native_vnni_preq(packed, A_q8.data(), result_512.data(),                      \
                              ISAPath::AVX512);                                            \
        gemv_native_vnni_preq(packed, A_q8.data(), result_256.data(),                      \
                              ISAPath::AVX2);                                              \
                                                                                           \
        assertExactEqual(result_512.data(), result_256.data(), N,                           \
                         #FORMAT " full GEMV " #N "x" #K);                                 \
    }

// Small N (single chunk, no tiling needed)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 64, 256, 42)
FULL_GEMV_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 64, 256, 43)

// Medium N (multiple chunks, exercises N-parallel path)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 512, 512, 44)
FULL_GEMV_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 512, 512, 45)
FULL_GEMV_PARITY_TEST(Q5_0, createQ5_0Random, 512, 512, 46)
FULL_GEMV_PARITY_TEST(Q5_1, createQ5_1Random, 512, 512, 47)

// Large N (exercises tiling, may trigger K-parallel path)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 4096, 896, 48)
FULL_GEMV_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 4096, 896, 49)
FULL_GEMV_PARITY_TEST(Q5_0, createQ5_0Random, 4096, 896, 50)

// Non-64-aligned N (exercises partial chunk handling)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 100, 256, 51)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 200, 512, 52)
FULL_GEMV_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 200, 512, 53)

#undef FULL_GEMV_PARITY_TEST

// ============================================================================
// Level 4: Full GEMM (M>1) parity via ISAPath dispatch
// ============================================================================

#define FULL_GEMM_PARITY_TEST(FORMAT, CREATE_FN, M, N, K, SEED)                           \
    TEST_F(AVX2VNNIParity, FullGEMM_##FORMAT##_M##M##_##N##x##K)                          \
    {                                                                                      \
        auto weights = TestTensorFactory::CREATE_FN({N, K}, SEED);                         \
        auto packed = packWeights(weights.get());                                          \
        int K_blocks = packed.blocks_per_row;                                              \
                                                                                           \
        /* Create M rows of Q8_1 activations */                                            \
        std::vector<Q8_1Block> A_q8_all(static_cast<size_t>(M) * K_blocks);                \
        std::mt19937 rng(SEED + 100);                                                      \
        std::uniform_int_distribution<int> dist(-127, 127);                                \
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.5f);                    \
        for (int m = 0; m < M; ++m)                                                        \
        {                                                                                  \
            for (int kb = 0; kb < K_blocks; ++kb)                                          \
            {                                                                              \
                auto &blk = A_q8_all[m * K_blocks + kb];                                   \
                float scale = scale_dist(rng);                                             \
                blk.d = simd::fp32_to_fp16(scale);                                         \
                int32_t sum = 0;                                                           \
                for (int i = 0; i < 32; ++i)                                               \
                {                                                                          \
                    blk.qs[i] = static_cast<int8_t>(dist(rng));                            \
                    sum += blk.qs[i];                                                      \
                }                                                                          \
                blk.sum_qs = static_cast<int16_t>(std::clamp(sum, -32768, 32767));         \
            }                                                                              \
        }                                                                                  \
                                                                                           \
        int ldc = N;                                                                       \
        std::vector<float> result_512(static_cast<size_t>(M) * N, 0.0f);                   \
        std::vector<float> result_256(static_cast<size_t>(M) * N, 0.0f);                   \
                                                                                           \
        gemm_native_vnni_preq(packed, A_q8_all.data(), result_512.data(),                  \
                              M, ldc, ISAPath::AVX512);                                    \
        gemm_native_vnni_preq(packed, A_q8_all.data(), result_256.data(),                  \
                              M, ldc, ISAPath::AVX2);                                      \
                                                                                           \
        assertExactEqual(result_512.data(), result_256.data(),                              \
                         static_cast<int>(M) * N,                                          \
                         #FORMAT " GEMM M=" #M " " #N "x" #K);                            \
    }

// M=2 (exercises 2-row microkernel)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 2, 512, 512, 60)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 2, 512, 512, 61)
FULL_GEMM_PARITY_TEST(Q5_0, createQ5_0Random, 2, 512, 512, 62)
FULL_GEMM_PARITY_TEST(Q5_1, createQ5_1Random, 2, 512, 512, 63)

// M=3 (exercises 2-row + 1-row tail)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 3, 512, 512, 64)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 3, 512, 512, 65)

// M=8 (exercises multiple 2-row pairs)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 8, 512, 512, 66)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 8, 512, 512, 67)
FULL_GEMM_PARITY_TEST(Q5_0, createQ5_0Random, 8, 512, 512, 68)

// Large GEMM (realistic LLM dimensions)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 16, 4096, 896, 70)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 16, 4096, 896, 71)

// Non-64-aligned N
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 4, 200, 256, 72)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 4, 200, 256, 73)

// M=1 via GEMM path (should match GEMV)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 1, 512, 512, 74)

#undef FULL_GEMM_PARITY_TEST

// ============================================================================
// Level 5: Decode LUT builder parity
// ============================================================================

TEST_F(AVX2VNNIParity, DecodeLUT_Q4_0)
{
    // Verify build_decode_lut_avx2 produces the same per-lane mapping
    alignas(16) static constexpr int8_t expected[16] = {
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

    __m256i lut = build_decode_lut_avx2(expected);
    alignas(32) int8_t result[32];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result), lut);

    // AVX2 vpshufb within 128-bit lanes: both halves should match
    for (int lane = 0; lane < 2; ++lane)
    {
        for (int i = 0; i < 16; ++i)
        {
            EXPECT_EQ(result[lane * 16 + i], expected[i])
                << "Lane " << lane << " index " << i;
        }
    }
}

TEST_F(AVX2VNNIParity, DecodeLUT_IQ4_NL)
{
    alignas(16) static constexpr int8_t expected[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

    __m256i lut = build_decode_lut_avx2(expected);
    alignas(32) int8_t result[32];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result), lut);

    for (int lane = 0; lane < 2; ++lane)
    {
        for (int i = 0; i < 16; ++i)
        {
            EXPECT_EQ(result[lane * 16 + i], expected[i])
                << "Lane " << lane << " index " << i;
        }
    }
}

// ============================================================================
// Level 6: Horizontal reduction parity
// ============================================================================

TEST_F(AVX2VNNIParity, HsumPS)
{
    alignas(32) float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    __m256 v = _mm256_load_ps(data);
    float result = hsum_ps_avx2(v);
    float expected = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8;
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(AVX2VNNIParity, HsumEpi32)
{
    alignas(32) int32_t data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i *>(data));
    int32_t result = hsum_epi32_avx2(v);
    int32_t expected = 10 + 20 + 30 + 40 + 50 + 60 + 70 + 80;
    EXPECT_EQ(result, expected);
}

#endif // AVX512F && AVX512VNNI && AVX512BW
