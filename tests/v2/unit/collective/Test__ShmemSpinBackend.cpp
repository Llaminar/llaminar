/**
 * @file Test__ShmemSpinBackend.cpp
 * @brief Unit tests for ShmemSpinBackend shared-memory spin-wait allreduce
 *
 * Single-process tests (MPI_PROCS 1):
 *   - Arena layout / alignment assertions
 *   - Fast-path detection
 *   - AVX-512 reduce correctness
 *
 * Two-process tests (MPI_PROCS 2):
 *   - Basic allreduce correctness
 *   - Repeated allreduce (epoch counter correctness)
 *   - Boundary count (MAX_COUNT)
 *   - Small count (1 element)
 *   - Fallback delegation (allgather)
 *   - Stress test (1000 consecutive allreduces)
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <numeric>
#include <vector>

#include "collective/backends/ShmemSpinBackend.h"
#include "collective/backends/UPIBackend.h"
#include "collective/DeviceGroup.h"

using namespace llaminar2;

// ============================================================================
// Helper: MPI rank info
// ============================================================================

static int mpiRank()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

static int mpiSize()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

// ============================================================================
// Single-Process Tests (always run, even with MPI_PROCS 1)
// ============================================================================

TEST(Test__ShmemSpinBackend, ArenaLayoutAlignment)
{
    // ShmemSpinArena header must be exactly one cache line
    EXPECT_EQ(sizeof(ShmemSpinArena), 64u);

    // EpochSlots must be cache-line-sized
    EXPECT_EQ(sizeof(ShmemSpinArena::EpochSlot), 64u);

    // Verify dynamic layout correctness for various rank counts
    for (int n : {1, 2, 4, 8, 16})
    {
        size_t expected_size = 64 // header
                               + static_cast<size_t>(n) * 64 // epoch slots
                               + static_cast<size_t>(n) * 8192 * sizeof(float); // buffers
        EXPECT_EQ(ShmemSpinArena::compute_size(n), expected_size)
            << "Mismatched arena size for " << n << " ranks";
    }
}

TEST(Test__ShmemSpinBackend, ArenaMaxCount)
{
    EXPECT_EQ(ShmemSpinArena::MAX_COUNT, 8192u);
    // Each rank buffer: 8192 * 4 = 32KB
    EXPECT_EQ(ShmemSpinArena::MAX_COUNT * sizeof(float), 32768u);
}

TEST(Test__ShmemSpinBackend, FastPathDetection)
{
    // Create a backend with a null fallback (won't call MPI in this test)
    // We need MPI_COMM_WORLD for the fallback but we won't actually use it
    auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
    ShmemSpinBackend backend(999, mpiRank(), std::move(upi));

    // Fast path: FLOAT32 + SUM + count ≤ MAX
    EXPECT_TRUE(backend.isFastPath(2048, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.isFastPath(1, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.isFastPath(ShmemSpinArena::MAX_COUNT, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

    // Not fast path: count too large
    EXPECT_FALSE(backend.isFastPath(ShmemSpinArena::MAX_COUNT + 1, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

    // Not fast path: wrong dtype
    EXPECT_FALSE(backend.isFastPath(2048, CollectiveDataType::INT32, CollectiveOp::ALLREDUCE_SUM));

    // Fast path: FP16 and BF16 are also supported
    EXPECT_TRUE(backend.isFastPath(2048, CollectiveDataType::FLOAT16, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.isFastPath(2048, CollectiveDataType::BFLOAT16, CollectiveOp::ALLREDUCE_SUM));

    // Not fast path: wrong op
    EXPECT_FALSE(backend.isFastPath(2048, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_MAX));
    EXPECT_FALSE(backend.isFastPath(2048, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_MIN));
}

// ============================================================================
// Reduce ISA Parity Tests — all three implementations must agree
// ============================================================================

// Helper: fill vectors with a deterministic non-trivial pattern
static void fillTestVectors(std::vector<float> &a, std::vector<float> &b, size_t count)
{
    a.resize(count);
    b.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        a[i] = static_cast<float>(i) * 0.1f - 100.0f;       // [-100, ...]
        b[i] = static_cast<float>(count - i) * 0.3f + 50.0f; // [50+, ...]
    }
}

// Verify two float buffers are bitwise identical
static void expectExactMatch(const float *ref, const float *test, size_t count,
                             const std::string &label)
{
    for (size_t i = 0; i < count; ++i)
    {
        ASSERT_EQ(ref[i], test[i])
            << label << ": mismatch at index " << i
            << " (expected " << ref[i] << ", got " << test[i] << ")";
    }
}

TEST(Test__ShmemSpinBackend, ReduceScalarCorrectness)
{
    const size_t count = 137; // Odd count to exercise tails
    std::vector<float> a, b;
    fillTestVectors(a, b, count);
    std::vector<float> out(count, 0.0f);

    ShmemSpinBackend::reduce_scalar(out.data(), a.data(), b.data(), count);

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(out[i], a[i] + b[i]) << "Scalar mismatch at index " << i;
    }
}

TEST(Test__ShmemSpinBackend, ReduceAVX2MatchesScalar)
{
    // Test multiple sizes: aligned, unaligned, small, large, power-of-two, odd
    for (size_t count : {1u, 7u, 8u, 9u, 15u, 16u, 31u, 32u, 137u, 256u, 1024u, 2048u, 8192u})
    {
        std::vector<float> a, b;
        fillTestVectors(a, b, count);
        std::vector<float> ref(count), avx2_out(count);

        ShmemSpinBackend::reduce_scalar(ref.data(), a.data(), b.data(), count);
        ShmemSpinBackend::reduce_avx2(avx2_out.data(), a.data(), b.data(), count);

        expectExactMatch(ref.data(), avx2_out.data(), count,
                         "AVX2 vs Scalar (count=" + std::to_string(count) + ")");
    }
}

TEST(Test__ShmemSpinBackend, ReduceAVX512MatchesScalar)
{
    for (size_t count : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 137u, 256u, 1024u, 2048u, 8192u})
    {
        std::vector<float> a, b;
        fillTestVectors(a, b, count);
        std::vector<float> ref(count), avx512_out(count);

        ShmemSpinBackend::reduce_scalar(ref.data(), a.data(), b.data(), count);
        ShmemSpinBackend::reduce_avx512(avx512_out.data(), a.data(), b.data(), count);

        expectExactMatch(ref.data(), avx512_out.data(), count,
                         "AVX512 vs Scalar (count=" + std::to_string(count) + ")");
    }
}

TEST(Test__ShmemSpinBackend, ReduceDispatchMatchesScalar)
{
    // The runtime dispatcher should produce identical results to scalar
    for (size_t count : {1u, 15u, 16u, 17u, 137u, 2048u, 8192u})
    {
        std::vector<float> a, b;
        fillTestVectors(a, b, count);
        std::vector<float> ref(count), dispatched(count);

        ShmemSpinBackend::reduce_scalar(ref.data(), a.data(), b.data(), count);
        ShmemSpinBackend::reduce(dispatched.data(), a.data(), b.data(), count);

        expectExactMatch(ref.data(), dispatched.data(), count,
                         "Dispatch vs Scalar (count=" + std::to_string(count) + ")");
    }
}

TEST(Test__ShmemSpinBackend, ReduceAllThreeAgree)
{
    // Single comprehensive test: all three ISA paths must be bitwise identical
    const size_t count = 4096;
    std::vector<float> a, b;
    fillTestVectors(a, b, count);

    std::vector<float> scalar_out(count), avx2_out(count), avx512_out(count);

    ShmemSpinBackend::reduce_scalar(scalar_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_avx2(avx2_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_avx512(avx512_out.data(), a.data(), b.data(), count);

    expectExactMatch(scalar_out.data(), avx2_out.data(), count, "AVX2 vs Scalar (4096)");
    expectExactMatch(scalar_out.data(), avx512_out.data(), count, "AVX512 vs Scalar (4096)");
}

TEST(Test__ShmemSpinBackend, ReduceZeroCount)
{
    float a = 1.0f, b = 2.0f, out = -1.0f;
    // count=0 should be a no-op — out must be untouched
    ShmemSpinBackend::reduce_scalar(&out, &a, &b, 0);
    EXPECT_EQ(out, -1.0f);
    ShmemSpinBackend::reduce_avx2(&out, &a, &b, 0);
    EXPECT_EQ(out, -1.0f);
    ShmemSpinBackend::reduce_avx512(&out, &a, &b, 0);
    EXPECT_EQ(out, -1.0f);
}

TEST(Test__ShmemSpinBackend, ReduceNegativesAndSpecialValues)
{
    // Negative values, zeros, large magnitudes
    std::vector<float> a = {-1.0f, 0.0f, 1e30f, -1e30f, 0.5f};
    std::vector<float> b = {1.0f, 0.0f, -1e30f, 1e30f, -0.5f};
    const size_t count = a.size();

    std::vector<float> scalar_out(count), avx2_out(count), avx512_out(count);

    ShmemSpinBackend::reduce_scalar(scalar_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_avx2(avx2_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_avx512(avx512_out.data(), a.data(), b.data(), count);

    expectExactMatch(scalar_out.data(), avx2_out.data(), count, "AVX2 special values");
    expectExactMatch(scalar_out.data(), avx512_out.data(), count, "AVX512 special values");

    // Verify expected results
    EXPECT_EQ(scalar_out[0], 0.0f);  // -1 + 1
    EXPECT_EQ(scalar_out[1], 0.0f);  //  0 + 0
    EXPECT_EQ(scalar_out[2], 0.0f);  // 1e30 + -1e30
    EXPECT_EQ(scalar_out[3], 0.0f);  // -1e30 + 1e30
    EXPECT_EQ(scalar_out[4], 0.0f);  // 0.5 + -0.5
}

// ============================================================================
// FP16 Reduce ISA Parity Tests
// ============================================================================

// Helper: convert float array to FP16 (uint16_t) using F16C
static void floatsToFP16(const std::vector<float> &src, std::vector<uint16_t> &dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = static_cast<uint16_t>(
            _cvtss_sh(src[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
}

// Helper: convert FP16 back to float for comparison
static float fp16ToFloat(uint16_t h) { return _cvtsh_ss(h); }

static void expectExactMatchU16(const uint16_t *ref, const uint16_t *test, size_t count,
                                const std::string &label)
{
    for (size_t i = 0; i < count; ++i)
    {
        ASSERT_EQ(ref[i], test[i])
            << label << ": mismatch at index " << i
            << " (ref=0x" << std::hex << ref[i]
            << " [" << std::dec << fp16ToFloat(ref[i]) << "]"
            << ", got=0x" << std::hex << test[i]
            << " [" << std::dec << fp16ToFloat(test[i]) << "])";
    }
}

TEST(Test__ShmemSpinBackend, ReduceFP16ScalarCorrectness)
{
    // Known FP16-representable values
    std::vector<float> fa = {1.0f, -2.5f, 0.0f, 100.0f, -0.125f};
    std::vector<float> fb = {3.0f, 2.5f, 0.0f, -50.0f, 0.125f};
    std::vector<uint16_t> a, b;
    floatsToFP16(fa, a);
    floatsToFP16(fb, b);

    std::vector<uint16_t> out(a.size());
    ShmemSpinBackend::reduce_fp16_scalar(out.data(), a.data(), b.data(), a.size());

    // Verify results against expected float sums converted back to FP16
    for (size_t i = 0; i < fa.size(); ++i)
    {
        float expected = fa[i] + fb[i];
        float actual = fp16ToFloat(out[i]);
        EXPECT_FLOAT_EQ(actual, expected) << "FP16 scalar mismatch at index " << i;
    }
}

TEST(Test__ShmemSpinBackend, ReduceFP16AVX2MatchesScalar)
{
    // Generate FP16-safe test data from floats in [-100, 100]
    for (size_t count : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 137u, 256u, 1024u, 2048u})
    {
        std::vector<float> fa(count), fb(count);
        for (size_t i = 0; i < count; ++i)
        {
            fa[i] = static_cast<float>(i) * 0.05f - 25.0f;
            fb[i] = static_cast<float>(count - i) * 0.03f + 10.0f;
        }
        std::vector<uint16_t> a, b;
        floatsToFP16(fa, a);
        floatsToFP16(fb, b);

        std::vector<uint16_t> ref(count), avx2_out(count);
        ShmemSpinBackend::reduce_fp16_scalar(ref.data(), a.data(), b.data(), count);
        ShmemSpinBackend::reduce_fp16_avx2(avx2_out.data(), a.data(), b.data(), count);

        expectExactMatchU16(ref.data(), avx2_out.data(), count,
                            "FP16 AVX2 vs Scalar (count=" + std::to_string(count) + ")");
    }
}

TEST(Test__ShmemSpinBackend, ReduceFP16AVX512MatchesScalar)
{
    for (size_t count : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 137u, 256u, 1024u, 2048u})
    {
        std::vector<float> fa(count), fb(count);
        for (size_t i = 0; i < count; ++i)
        {
            fa[i] = static_cast<float>(i) * 0.05f - 25.0f;
            fb[i] = static_cast<float>(count - i) * 0.03f + 10.0f;
        }
        std::vector<uint16_t> a, b;
        floatsToFP16(fa, a);
        floatsToFP16(fb, b);

        std::vector<uint16_t> ref(count), avx512_out(count);
        ShmemSpinBackend::reduce_fp16_scalar(ref.data(), a.data(), b.data(), count);
        ShmemSpinBackend::reduce_fp16_avx512(avx512_out.data(), a.data(), b.data(), count);

        expectExactMatchU16(ref.data(), avx512_out.data(), count,
                            "FP16 AVX512 vs Scalar (count=" + std::to_string(count) + ")");
    }
}

TEST(Test__ShmemSpinBackend, ReduceFP16DispatchMatchesScalar)
{
    for (size_t count : {1u, 15u, 16u, 17u, 137u, 2048u})
    {
        std::vector<float> fa(count), fb(count);
        for (size_t i = 0; i < count; ++i)
        {
            fa[i] = static_cast<float>(i) * 0.1f - 50.0f;
            fb[i] = static_cast<float>(count - i) * 0.2f;
        }
        std::vector<uint16_t> a, b;
        floatsToFP16(fa, a);
        floatsToFP16(fb, b);

        std::vector<uint16_t> ref(count), dispatched(count);
        ShmemSpinBackend::reduce_fp16_scalar(ref.data(), a.data(), b.data(), count);
        ShmemSpinBackend::reduce_fp16(dispatched.data(), a.data(), b.data(), count);

        expectExactMatchU16(ref.data(), dispatched.data(), count,
                            "FP16 Dispatch vs Scalar (count=" + std::to_string(count) + ")");
    }
}

TEST(Test__ShmemSpinBackend, ReduceFP16AllThreeAgree)
{
    const size_t count = 4096;
    std::vector<float> fa(count), fb(count);
    for (size_t i = 0; i < count; ++i)
    {
        fa[i] = static_cast<float>(i) * 0.01f - 20.0f;
        fb[i] = static_cast<float>(count - i) * 0.01f;
    }
    std::vector<uint16_t> a, b;
    floatsToFP16(fa, a);
    floatsToFP16(fb, b);

    std::vector<uint16_t> scalar_out(count), avx2_out(count), avx512_out(count);
    ShmemSpinBackend::reduce_fp16_scalar(scalar_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_fp16_avx2(avx2_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_fp16_avx512(avx512_out.data(), a.data(), b.data(), count);

    expectExactMatchU16(scalar_out.data(), avx2_out.data(), count, "FP16 AVX2 vs Scalar (4096)");
    expectExactMatchU16(scalar_out.data(), avx512_out.data(), count, "FP16 AVX512 vs Scalar (4096)");
}

TEST(Test__ShmemSpinBackend, ReduceFP16ZeroCount)
{
    uint16_t a = 0x3C00, b = 0x4000, out = 0xFFFF; // 1.0, 2.0
    ShmemSpinBackend::reduce_fp16_scalar(&out, &a, &b, 0);
    EXPECT_EQ(out, 0xFFFFu);
    ShmemSpinBackend::reduce_fp16_avx2(&out, &a, &b, 0);
    EXPECT_EQ(out, 0xFFFFu);
    ShmemSpinBackend::reduce_fp16_avx512(&out, &a, &b, 0);
    EXPECT_EQ(out, 0xFFFFu);
}

// ============================================================================
// BF16 Reduce ISA Parity Tests
// ============================================================================

// Helper: convert float → BF16 via truncation (matches reduce implementation)
static uint16_t floatToBF16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

static float bf16ToFloat(uint16_t bf)
{
    uint32_t f = static_cast<uint32_t>(bf) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

static void floatsToBF16(const std::vector<float> &src, std::vector<uint16_t> &dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = floatToBF16(src[i]);
    }
}

static void expectExactMatchBF16(const uint16_t *ref, const uint16_t *test, size_t count,
                                 const std::string &label)
{
    for (size_t i = 0; i < count; ++i)
    {
        ASSERT_EQ(ref[i], test[i])
            << label << ": mismatch at index " << i
            << " (ref=0x" << std::hex << ref[i]
            << " [" << std::dec << bf16ToFloat(ref[i]) << "]"
            << ", got=0x" << std::hex << test[i]
            << " [" << std::dec << bf16ToFloat(test[i]) << "])";
    }
}

TEST(Test__ShmemSpinBackend, ReduceBF16ScalarCorrectness)
{
    std::vector<float> fa = {1.0f, -2.0f, 0.0f, 128.0f, -0.125f};
    std::vector<float> fb = {3.0f, 2.0f, 0.0f, -64.0f, 0.125f};
    std::vector<uint16_t> a, b;
    floatsToBF16(fa, a);
    floatsToBF16(fb, b);

    std::vector<uint16_t> out(a.size());
    ShmemSpinBackend::reduce_bf16_scalar(out.data(), a.data(), b.data(), a.size());

    for (size_t i = 0; i < fa.size(); ++i)
    {
        // Expected: convert inputs to BF16, add in FP32, truncate to BF16
        float sum_f32 = bf16ToFloat(a[i]) + bf16ToFloat(b[i]);
        uint16_t expected_bf16 = floatToBF16(sum_f32);
        EXPECT_EQ(out[i], expected_bf16) << "BF16 scalar mismatch at index " << i
                                         << " (expected " << bf16ToFloat(expected_bf16)
                                         << ", got " << bf16ToFloat(out[i]) << ")";
    }
}

TEST(Test__ShmemSpinBackend, ReduceBF16AVX2MatchesScalar)
{
    for (size_t count : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 137u, 256u, 1024u, 2048u})
    {
        std::vector<float> fa(count), fb(count);
        for (size_t i = 0; i < count; ++i)
        {
            fa[i] = static_cast<float>(i) * 0.5f - 100.0f;
            fb[i] = static_cast<float>(count - i) * 0.3f + 50.0f;
        }
        std::vector<uint16_t> a, b;
        floatsToBF16(fa, a);
        floatsToBF16(fb, b);

        std::vector<uint16_t> ref(count), avx2_out(count);
        ShmemSpinBackend::reduce_bf16_scalar(ref.data(), a.data(), b.data(), count);
        ShmemSpinBackend::reduce_bf16_avx2(avx2_out.data(), a.data(), b.data(), count);

        expectExactMatchBF16(ref.data(), avx2_out.data(), count,
                             "BF16 AVX2 vs Scalar (count=" + std::to_string(count) + ")");
    }
}

TEST(Test__ShmemSpinBackend, ReduceBF16AVX512MatchesScalar)
{
    for (size_t count : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 137u, 256u, 1024u, 2048u})
    {
        std::vector<float> fa(count), fb(count);
        for (size_t i = 0; i < count; ++i)
        {
            fa[i] = static_cast<float>(i) * 0.5f - 100.0f;
            fb[i] = static_cast<float>(count - i) * 0.3f + 50.0f;
        }
        std::vector<uint16_t> a, b;
        floatsToBF16(fa, a);
        floatsToBF16(fb, b);

        std::vector<uint16_t> ref(count), avx512_out(count);
        ShmemSpinBackend::reduce_bf16_scalar(ref.data(), a.data(), b.data(), count);
        ShmemSpinBackend::reduce_bf16_avx512(avx512_out.data(), a.data(), b.data(), count);

        expectExactMatchBF16(ref.data(), avx512_out.data(), count,
                             "BF16 AVX512 vs Scalar (count=" + std::to_string(count) + ")");
    }
}

TEST(Test__ShmemSpinBackend, ReduceBF16DispatchMatchesScalar)
{
    for (size_t count : {1u, 15u, 16u, 17u, 137u, 2048u})
    {
        std::vector<float> fa(count), fb(count);
        for (size_t i = 0; i < count; ++i)
        {
            fa[i] = static_cast<float>(i) * 1.0f - 500.0f;
            fb[i] = static_cast<float>(count - i) * 0.5f;
        }
        std::vector<uint16_t> a, b;
        floatsToBF16(fa, a);
        floatsToBF16(fb, b);

        std::vector<uint16_t> ref(count), dispatched(count);
        ShmemSpinBackend::reduce_bf16_scalar(ref.data(), a.data(), b.data(), count);
        ShmemSpinBackend::reduce_bf16(dispatched.data(), a.data(), b.data(), count);

        expectExactMatchBF16(ref.data(), dispatched.data(), count,
                             "BF16 Dispatch vs Scalar (count=" + std::to_string(count) + ")");
    }
}

TEST(Test__ShmemSpinBackend, ReduceBF16AllThreeAgree)
{
    const size_t count = 4096;
    std::vector<float> fa(count), fb(count);
    for (size_t i = 0; i < count; ++i)
    {
        fa[i] = static_cast<float>(i) * 0.1f - 200.0f;
        fb[i] = static_cast<float>(count - i) * 0.1f;
    }
    std::vector<uint16_t> a, b;
    floatsToBF16(fa, a);
    floatsToBF16(fb, b);

    std::vector<uint16_t> scalar_out(count), avx2_out(count), avx512_out(count);
    ShmemSpinBackend::reduce_bf16_scalar(scalar_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_bf16_avx2(avx2_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_bf16_avx512(avx512_out.data(), a.data(), b.data(), count);

    expectExactMatchBF16(scalar_out.data(), avx2_out.data(), count, "BF16 AVX2 vs Scalar (4096)");
    expectExactMatchBF16(scalar_out.data(), avx512_out.data(), count, "BF16 AVX512 vs Scalar (4096)");
}

TEST(Test__ShmemSpinBackend, ReduceBF16ZeroCount)
{
    uint16_t a = 0x3F80, b = 0x4000, out = 0xFFFF; // 1.0, 2.0 in BF16
    ShmemSpinBackend::reduce_bf16_scalar(&out, &a, &b, 0);
    EXPECT_EQ(out, 0xFFFFu);
    ShmemSpinBackend::reduce_bf16_avx2(&out, &a, &b, 0);
    EXPECT_EQ(out, 0xFFFFu);
    ShmemSpinBackend::reduce_bf16_avx512(&out, &a, &b, 0);
    EXPECT_EQ(out, 0xFFFFu);
}

TEST(Test__ShmemSpinBackend, ReduceFP16NegativeAndSpecialValues)
{
    // Negative values, zero cancellation, large magnitudes in FP16 range
    std::vector<float> fa = {-1.0f, 0.0f, 100.0f, -100.0f, 0.5f, -0.5f};
    std::vector<float> fb = {1.0f, 0.0f, -100.0f, 100.0f, -0.5f, 0.5f};
    std::vector<uint16_t> a, b;
    floatsToFP16(fa, a);
    floatsToFP16(fb, b);

    std::vector<uint16_t> scalar_out(a.size()), avx2_out(a.size()), avx512_out(a.size());
    ShmemSpinBackend::reduce_fp16_scalar(scalar_out.data(), a.data(), b.data(), a.size());
    ShmemSpinBackend::reduce_fp16_avx2(avx2_out.data(), a.data(), b.data(), a.size());
    ShmemSpinBackend::reduce_fp16_avx512(avx512_out.data(), a.data(), b.data(), a.size());

    expectExactMatchU16(scalar_out.data(), avx2_out.data(), a.size(), "FP16 AVX2 special");
    expectExactMatchU16(scalar_out.data(), avx512_out.data(), a.size(), "FP16 AVX512 special");

    // Verify cancellation produces zero
    for (size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_FLOAT_EQ(fp16ToFloat(scalar_out[i]), 0.0f)
            << "FP16 special value should cancel at index " << i;
    }
}

TEST(Test__ShmemSpinBackend, ReduceBF16NegativeAndSpecialValues)
{
    // Negative, zero cancellation, large BF16-representable magnitudes
    std::vector<float> fa = {-1.0f, 0.0f, 128.0f, -128.0f, 0.5f, -0.5f};
    std::vector<float> fb = {1.0f, 0.0f, -128.0f, 128.0f, -0.5f, 0.5f};
    std::vector<uint16_t> a, b;
    floatsToBF16(fa, a);
    floatsToBF16(fb, b);

    std::vector<uint16_t> scalar_out(a.size()), avx2_out(a.size()), avx512_out(a.size());
    ShmemSpinBackend::reduce_bf16_scalar(scalar_out.data(), a.data(), b.data(), a.size());
    ShmemSpinBackend::reduce_bf16_avx2(avx2_out.data(), a.data(), b.data(), a.size());
    ShmemSpinBackend::reduce_bf16_avx512(avx512_out.data(), a.data(), b.data(), a.size());

    expectExactMatchBF16(scalar_out.data(), avx2_out.data(), a.size(), "BF16 AVX2 special");
    expectExactMatchBF16(scalar_out.data(), avx512_out.data(), a.size(), "BF16 AVX512 special");

    for (size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_FLOAT_EQ(bf16ToFloat(scalar_out[i]), 0.0f)
            << "BF16 special value should cancel at index " << i;
    }
}

TEST(Test__ShmemSpinBackend, ReduceFP16MaxCountBoundary)
{
    // Test at MAX_COUNT — verifies vectorized tail handling at exact buffer limit
    const size_t count = ShmemSpinArena::MAX_COUNT;
    std::vector<float> fa(count), fb(count);
    for (size_t i = 0; i < count; ++i)
    {
        fa[i] = static_cast<float>(i % 100) * 0.01f;
        fb[i] = static_cast<float>((count - i) % 100) * 0.01f;
    }
    std::vector<uint16_t> a, b;
    floatsToFP16(fa, a);
    floatsToFP16(fb, b);

    std::vector<uint16_t> scalar_out(count), avx512_out(count);
    ShmemSpinBackend::reduce_fp16_scalar(scalar_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_fp16_avx512(avx512_out.data(), a.data(), b.data(), count);
    expectExactMatchU16(scalar_out.data(), avx512_out.data(), count, "FP16 MAX_COUNT boundary");
}

TEST(Test__ShmemSpinBackend, ReduceBF16MaxCountBoundary)
{
    const size_t count = ShmemSpinArena::MAX_COUNT;
    std::vector<float> fa(count), fb(count);
    for (size_t i = 0; i < count; ++i)
    {
        fa[i] = static_cast<float>(i % 200) * 0.5f - 50.0f;
        fb[i] = static_cast<float>((count - i) % 200) * 0.5f;
    }
    std::vector<uint16_t> a, b;
    floatsToBF16(fa, a);
    floatsToBF16(fb, b);

    std::vector<uint16_t> scalar_out(count), avx512_out(count);
    ShmemSpinBackend::reduce_bf16_scalar(scalar_out.data(), a.data(), b.data(), count);
    ShmemSpinBackend::reduce_bf16_avx512(avx512_out.data(), a.data(), b.data(), count);
    expectExactMatchBF16(scalar_out.data(), avx512_out.data(), count, "BF16 MAX_COUNT boundary");
}

// ============================================================================
// Fast-path detection for FP16 / BF16
// ============================================================================

TEST(Test__ShmemSpinBackend, FastPathDetectionFP16BF16)
{
    auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
    ShmemSpinBackend backend(994, mpiRank(), std::move(upi));

    // FP16 fast path
    EXPECT_TRUE(backend.isFastPath(2048, CollectiveDataType::FLOAT16, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.isFastPath(ShmemSpinArena::MAX_COUNT, CollectiveDataType::FLOAT16, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_FALSE(backend.isFastPath(ShmemSpinArena::MAX_COUNT + 1, CollectiveDataType::FLOAT16, CollectiveOp::ALLREDUCE_SUM));

    // BF16 fast path
    EXPECT_TRUE(backend.isFastPath(2048, CollectiveDataType::BFLOAT16, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.isFastPath(ShmemSpinArena::MAX_COUNT, CollectiveDataType::BFLOAT16, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_FALSE(backend.isFastPath(ShmemSpinArena::MAX_COUNT + 1, CollectiveDataType::BFLOAT16, CollectiveOp::ALLREDUCE_SUM));

    // Still not fast path for wrong op
    EXPECT_FALSE(backend.isFastPath(2048, CollectiveDataType::FLOAT16, CollectiveOp::ALLREDUCE_MAX));
    EXPECT_FALSE(backend.isFastPath(2048, CollectiveDataType::BFLOAT16, CollectiveOp::ALLREDUCE_MAX));
}

TEST(Test__ShmemSpinBackend, SupportsOnlyCPU)
{
    auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
    ShmemSpinBackend backend(998, 0, std::move(upi));

    EXPECT_TRUE(backend.supportsDevice(DeviceType::CPU));
    EXPECT_FALSE(backend.supportsDevice(DeviceType::CUDA));
    EXPECT_FALSE(backend.supportsDevice(DeviceType::ROCm));
}

TEST(Test__ShmemSpinBackend, DirectTransferOnlyCPU)
{
    auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
    ShmemSpinBackend backend(997, 0, std::move(upi));

    EXPECT_TRUE(backend.supportsDirectTransfer(DeviceId::cpu(), DeviceId::cpu()));
    EXPECT_FALSE(backend.supportsDirectTransfer(DeviceId::cpu(), DeviceId::cuda(0)));
    EXPECT_FALSE(backend.supportsDirectTransfer(DeviceId::cuda(0), DeviceId::cpu()));
}

TEST(Test__ShmemSpinBackend, IdentityAndType)
{
    auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
    ShmemSpinBackend backend(996, 0, std::move(upi));

    EXPECT_EQ(backend.type(), CollectiveBackendType::UPI);
    EXPECT_EQ(backend.name(), "ShmemSpin");
}

TEST(Test__ShmemSpinBackend, NotInitializedBeforeInit)
{
    auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
    ShmemSpinBackend backend(995, 0, std::move(upi));

    EXPECT_FALSE(backend.isInitialized());
    EXPECT_EQ(backend.arena(), nullptr);
    EXPECT_EQ(backend.currentEpoch(), 0u);
}

// ============================================================================
// Two-Process Tests (only run with MPI_PROCS >= 2)
// ============================================================================

class ShmemSpinMPITest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rank_ = mpiRank();
        size_ = mpiSize();
        if (size_ < 2)
        {
            GTEST_SKIP() << "Requires MPI_PROCS >= 2";
        }

        // Create backend with unique domain ID per test (use test info hash)
        domain_id_ = 42;
        auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
        backend_ = std::make_unique<ShmemSpinBackend>(domain_id_, rank_, std::move(upi));

        DeviceGroup group;
        group.name = "test_shmem_spin";
        group.scope = CollectiveScope::LOCAL;
        for (int r = 0; r < size_; ++r)
            group.devices.push_back(DeviceId::cpu());

        ASSERT_TRUE(backend_->initialize(group));
        ASSERT_TRUE(backend_->isInitialized());
        ASSERT_NE(backend_->arena(), nullptr);

        // Precompute rank sum: 1 + 2 + ... + size_ = size_ * (size_ + 1) / 2
        rank_sum_ = static_cast<float>(size_ * (size_ + 1)) / 2.0f;
    }

    void TearDown() override
    {
        if (backend_)
        {
            backend_->shutdown();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int rank_ = 0;
    int size_ = 0;
    int domain_id_ = 0;
    float rank_sum_ = 0.0f; ///< Sum of (rank+1) for all ranks: 1+2+...+N
    std::unique_ptr<ShmemSpinBackend> backend_;
};

TEST_F(ShmemSpinMPITest, BasicAllreduce2048)
{
    // Each rank has its own values: rank r → [(r+1), (r+1), ...]
    const size_t count = 2048;
    std::vector<float> data(count, static_cast<float>(rank_ + 1));

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_SUM));

    // All ranks should have sum: 1 + 2 + ... + N
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], rank_sum_) << "Mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, AllreduceVaryingValues)
{
    // Rank r: [r*100, r*100+1, r*100+2, ...]
    const size_t count = 512;
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        data[i] = static_cast<float>(rank_ * 100 + i);
    }

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_SUM));

    // Expected: sum over all ranks of (r*100 + i)
    // = i*N + 100*(0+1+...+(N-1)) = i*N + 100*N*(N-1)/2
    for (size_t i = 0; i < count; ++i)
    {
        float expected = static_cast<float>(i) * size_
                         + 100.0f * size_ * (size_ - 1) / 2.0f;
        EXPECT_FLOAT_EQ(data[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, RepeatedAllreduce)
{
    // Verify epoch counter works correctly across multiple allreduces
    const size_t count = 256;
    std::vector<float> data(count);

    for (int iter = 0; iter < 100; ++iter)
    {
        // Reset data each iteration
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = static_cast<float>(rank_ + 1) * (iter + 1);
        }

        ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                        CollectiveDataType::FLOAT32,
                                        CollectiveOp::ALLREDUCE_SUM));

        float expected = rank_sum_ * (iter + 1);
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(data[i], expected) << "Mismatch at iter=" << iter << " index=" << i;
        }
    }

    // Epoch should match iteration count (2 increments per allreduce: write + read-completion)
    EXPECT_EQ(backend_->currentEpoch(), 200u);
}

TEST_F(ShmemSpinMPITest, MaxCountBoundary)
{
    // Test at exactly MAX_COUNT
    const size_t count = ShmemSpinArena::MAX_COUNT;
    std::vector<float> data(count, static_cast<float>(rank_ + 1));

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_SUM));

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], rank_sum_) << "Mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, SmallCount)
{
    // Single element
    float data = static_cast<float>(rank_ + 1);

    ASSERT_TRUE(backend_->allreduce(&data, 1,
                                    CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_SUM));

    EXPECT_FLOAT_EQ(data, rank_sum_);
}

TEST_F(ShmemSpinMPITest, OddCount)
{
    // Non-power-of-2, non-multiple-of-16 count to test tail handling
    const size_t count = 137;
    std::vector<float> data(count, static_cast<float>(rank_ + 1));

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_SUM));

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], rank_sum_) << "Mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, OverflowFallsBackToMPI)
{
    // Count > MAX_COUNT should fall back to MPI (still correct)
    const size_t count = ShmemSpinArena::MAX_COUNT + 100;
    std::vector<float> data(count, static_cast<float>(rank_ + 1));

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_SUM));

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], rank_sum_) << "Mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, NonSumFallsBackToMPI)
{
    // MAX op should fall back to MPI
    const size_t count = 64;
    std::vector<float> data(count, static_cast<float>(rank_ + 1));

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_MAX));

    // MAX of 1.0, 2.0, ..., N = N
    float expected_max = static_cast<float>(size_);
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], expected_max) << "Mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, AllgatherDelegates)
{
    // Allgather always delegates to fallback
    const size_t count = 32;
    std::vector<float> send(count, static_cast<float>(rank_ + 1));
    std::vector<float> recv(count * size_);

    ASSERT_TRUE(backend_->allgather(send.data(), recv.data(), count,
                                    CollectiveDataType::FLOAT32));

    // recv = [rank0 data, rank1 data, ...]
    for (int r = 0; r < size_; ++r)
    {
        float expected = static_cast<float>(r + 1);
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(recv[r * count + i], expected)
                << "rank " << r << " slice mismatch at " << i;
        }
    }
}

TEST_F(ShmemSpinMPITest, StressTest1000)
{
    // 1000 consecutive allreduces to verify protocol robustness
    const size_t count = 2048;
    std::vector<float> data(count);

    for (int iter = 0; iter < 1000; ++iter)
    {
        float val = static_cast<float>(rank_ + 1);
        std::fill(data.begin(), data.end(), val);

        ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                        CollectiveDataType::FLOAT32,
                                        CollectiveOp::ALLREDUCE_SUM))
            << "Failed at iteration " << iter;

        // Spot-check first and last elements
        EXPECT_FLOAT_EQ(data[0], rank_sum_) << "First element wrong at iter " << iter;
        EXPECT_FLOAT_EQ(data[count - 1], rank_sum_) << "Last element wrong at iter " << iter;
    }

    EXPECT_EQ(backend_->currentEpoch(), 2000u);
}

TEST_F(ShmemSpinMPITest, NegativeAndZeroValues)
{
    const size_t count = 128;
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        // Rank 0: [-64, -63, ..., 63]  Rank 1: [0, 0, ..., 0]
        data[i] = (rank_ == 0) ? static_cast<float>(i) - 64.0f : 0.0f;
    }

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_SUM));

    for (size_t i = 0; i < count; ++i)
    {
        float expected = static_cast<float>(i) - 64.0f;
        EXPECT_FLOAT_EQ(data[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, AllreduceFP16)
{
    const size_t count = 2048;
    uint16_t val = static_cast<uint16_t>(
        _cvtss_sh(static_cast<float>(rank_ + 1),
                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    std::vector<uint16_t> data(count, val);

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    // Compute expected via exact pairwise FP16 reduce path (matches implementation)
    // Start with rank 0 + rank 1, then accumulate remaining ranks
    uint16_t expected;
    {
        uint16_t h0 = static_cast<uint16_t>(_cvtss_sh(1.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        uint16_t h1 = static_cast<uint16_t>(_cvtss_sh(2.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        float acc = _cvtsh_ss(h0) + _cvtsh_ss(h1);
        expected = static_cast<uint16_t>(_cvtss_sh(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        for (int r = 2; r < size_; ++r)
        {
            uint16_t hr = static_cast<uint16_t>(_cvtss_sh(static_cast<float>(r + 1),
                                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            acc = _cvtsh_ss(expected) + _cvtsh_ss(hr);
            expected = static_cast<uint16_t>(_cvtss_sh(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(data[i], expected) << "FP16 allreduce mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, AllreduceBF16)
{
    const size_t count = 2048;
    // Each rank: rank r → (r+1) in BF16
    float rank_val = static_cast<float>(rank_ + 1);
    uint32_t bits;
    std::memcpy(&bits, &rank_val, sizeof(float));
    uint16_t bf16_val = static_cast<uint16_t>(bits >> 16);

    std::vector<uint16_t> data(count, bf16_val);

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::BFLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    // Compute expected via exact pairwise BF16 reduce path
    uint16_t expected_bf16;
    {
        float f0 = 1.0f, f1 = 2.0f;
        uint32_t b0, b1;
        std::memcpy(&b0, &f0, sizeof(float));
        std::memcpy(&b1, &f1, sizeof(float));
        uint16_t h0 = static_cast<uint16_t>(b0 >> 16);
        uint16_t h1 = static_cast<uint16_t>(b1 >> 16);
        uint32_t a0 = static_cast<uint32_t>(h0) << 16;
        uint32_t a1 = static_cast<uint32_t>(h1) << 16;
        float fa, fb;
        std::memcpy(&fa, &a0, sizeof(float));
        std::memcpy(&fb, &a1, sizeof(float));
        float acc = fa + fb;
        uint32_t acc_bits;
        std::memcpy(&acc_bits, &acc, sizeof(float));
        expected_bf16 = static_cast<uint16_t>(acc_bits >> 16);

        for (int r = 2; r < size_; ++r)
        {
            float fr = static_cast<float>(r + 1);
            uint32_t br;
            std::memcpy(&br, &fr, sizeof(float));
            uint16_t hr = static_cast<uint16_t>(br >> 16);

            uint32_t ae = static_cast<uint32_t>(expected_bf16) << 16;
            uint32_t ar = static_cast<uint32_t>(hr) << 16;
            float fe, fv;
            std::memcpy(&fe, &ae, sizeof(float));
            std::memcpy(&fv, &ar, sizeof(float));
            float sum = fe + fv;
            uint32_t sb;
            std::memcpy(&sb, &sum, sizeof(float));
            expected_bf16 = static_cast<uint16_t>(sb >> 16);
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(data[i], expected_bf16)
            << "BF16 allreduce mismatch at index " << i
            << " (got 0x" << std::hex << data[i] << ")";
    }
}

TEST_F(ShmemSpinMPITest, AllreduceFP16OddCount)
{
    // Non-power-of-2 count to test tail handling
    const size_t count = 137;
    float rank_val = static_cast<float>(rank_ + 1);
    uint16_t val = static_cast<uint16_t>(
        _cvtss_sh(rank_val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    std::vector<uint16_t> data(count, val);

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    // Compute expected via pairwise FP16 path
    uint16_t expected = static_cast<uint16_t>(_cvtss_sh(1.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    for (int r = 1; r < size_; ++r)
    {
        uint16_t hr = static_cast<uint16_t>(_cvtss_sh(static_cast<float>(r + 1), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        float acc = _cvtsh_ss(expected) + _cvtsh_ss(hr);
        expected = static_cast<uint16_t>(_cvtss_sh(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(data[i], expected) << "FP16 odd-count mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, AllreduceBF16OddCount)
{
    const size_t count = 137;
    float rank_val = static_cast<float>(rank_ + 1);
    uint32_t bits;
    std::memcpy(&bits, &rank_val, sizeof(float));
    uint16_t bf16_val = static_cast<uint16_t>(bits >> 16);
    std::vector<uint16_t> data(count, bf16_val);

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::BFLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    // Compute expected via pairwise BF16 path
    uint16_t expected;
    {
        float f0 = 1.0f;
        uint32_t b0;
        std::memcpy(&b0, &f0, sizeof(float));
        expected = static_cast<uint16_t>(b0 >> 16);
        for (int r = 1; r < size_; ++r)
        {
            float fr = static_cast<float>(r + 1);
            uint32_t br;
            std::memcpy(&br, &fr, sizeof(float));
            uint16_t hr = static_cast<uint16_t>(br >> 16);
            uint32_t ae = static_cast<uint32_t>(expected) << 16;
            uint32_t ar = static_cast<uint32_t>(hr) << 16;
            float fe, fv;
            std::memcpy(&fe, &ae, sizeof(float));
            std::memcpy(&fv, &ar, sizeof(float));
            float sum = fe + fv;
            uint32_t sb;
            std::memcpy(&sb, &sum, sizeof(float));
            expected = static_cast<uint16_t>(sb >> 16);
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(data[i], expected)
            << "BF16 odd-count mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, AllreduceFP16VaryingValues)
{
    // Position-dependent values catch element swap/reorder bugs
    const size_t count = 512;
    std::vector<uint16_t> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        float val = static_cast<float>(rank_ * 100 + i) * 0.1f;
        data[i] = static_cast<uint16_t>(
            _cvtss_sh(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    for (size_t i = 0; i < count; ++i)
    {
        // Model exact pairwise FP16 arithmetic: FP16→FP32→add→FP16
        uint16_t h0 = static_cast<uint16_t>(
            _cvtss_sh(static_cast<float>(i) * 0.1f,
                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        uint16_t expected = h0;
        for (int r = 1; r < size_; ++r)
        {
            uint16_t hr = static_cast<uint16_t>(
                _cvtss_sh(static_cast<float>(r * 100 + i) * 0.1f,
                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            float sum = _cvtsh_ss(expected) + _cvtsh_ss(hr);
            expected = static_cast<uint16_t>(
                _cvtss_sh(sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
        EXPECT_EQ(data[i], expected) << "FP16 varying mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, AllreduceBF16VaryingValues)
{
    // Position-dependent values catch element swap/reorder bugs
    const size_t count = 512;
    std::vector<uint16_t> data(count);
    for (size_t i = 0; i < count; ++i)
    {
        float val = static_cast<float>(rank_ * 100 + i) * 0.5f;
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(float));
        data[i] = static_cast<uint16_t>(bits >> 16);
    }

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::BFLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    for (size_t i = 0; i < count; ++i)
    {
        // Model exact pairwise BF16 arithmetic: BF16→FP32→add→BF16 (truncation)
        float f0 = static_cast<float>(i) * 0.5f;
        uint32_t b0;
        std::memcpy(&b0, &f0, sizeof(float));
        uint16_t expected = static_cast<uint16_t>(b0 >> 16);

        for (int r = 1; r < size_; ++r)
        {
            float fr = static_cast<float>(r * 100 + i) * 0.5f;
            uint32_t br;
            std::memcpy(&br, &fr, sizeof(float));
            uint16_t hr = static_cast<uint16_t>(br >> 16);

            uint32_t ae = static_cast<uint32_t>(expected) << 16;
            uint32_t ar = static_cast<uint32_t>(hr) << 16;
            float fe, fv;
            std::memcpy(&fe, &ae, sizeof(float));
            std::memcpy(&fv, &ar, sizeof(float));
            float sum = fe + fv;
            uint32_t sb;
            std::memcpy(&sb, &sum, sizeof(float));
            expected = static_cast<uint16_t>(sb >> 16);
        }

        EXPECT_EQ(data[i], expected) << "BF16 varying mismatch at index " << i;
    }
}

TEST_F(ShmemSpinMPITest, AllreduceFP16SingleElement)
{
    const size_t count = 1;
    uint16_t val = static_cast<uint16_t>(
        _cvtss_sh(static_cast<float>(rank_ + 1),
                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    std::vector<uint16_t> data(count, val);

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    EXPECT_FLOAT_EQ(_cvtsh_ss(data[0]), rank_sum_);
}

TEST_F(ShmemSpinMPITest, AllreduceBF16SingleElement)
{
    const size_t count = 1;
    float rank_val = static_cast<float>(rank_ + 1);
    uint32_t bits;
    std::memcpy(&bits, &rank_val, sizeof(float));
    uint16_t bf16_val = static_cast<uint16_t>(bits >> 16);
    std::vector<uint16_t> data(count, bf16_val);

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::BFLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    // rank_sum_ is exact in BF16 for small integers
    uint32_t expected_bits;
    float rs = rank_sum_;
    std::memcpy(&expected_bits, &rs, sizeof(float));
    uint16_t expected_bf16 = static_cast<uint16_t>(expected_bits >> 16);
    EXPECT_EQ(data[0], expected_bf16);
}

TEST_F(ShmemSpinMPITest, AllreduceFP16MaxCount)
{
    // Boundary: exactly MAX_COUNT elements — verifies no buffer overflow
    const size_t count = ShmemSpinArena::MAX_COUNT;
    uint16_t val = static_cast<uint16_t>(
        _cvtss_sh(static_cast<float>(rank_ + 1),
                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    std::vector<uint16_t> data(count, val);

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    // rank_sum_ is exact in FP16 for small integers
    // Spot-check first, middle, last
    EXPECT_FLOAT_EQ(_cvtsh_ss(data[0]), rank_sum_);
    EXPECT_FLOAT_EQ(_cvtsh_ss(data[count / 2]), rank_sum_);
    EXPECT_FLOAT_EQ(_cvtsh_ss(data[count - 1]), rank_sum_);
}

TEST_F(ShmemSpinMPITest, AllreduceBF16MaxCount)
{
    const size_t count = ShmemSpinArena::MAX_COUNT;
    float rank_val = static_cast<float>(rank_ + 1);
    uint32_t bits;
    std::memcpy(&bits, &rank_val, sizeof(float));
    uint16_t bf16_val = static_cast<uint16_t>(bits >> 16);
    std::vector<uint16_t> data(count, bf16_val);

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::BFLOAT16,
                                    CollectiveOp::ALLREDUCE_SUM));

    uint32_t expected_bits;
    float rs = rank_sum_;
    std::memcpy(&expected_bits, &rs, sizeof(float));
    uint16_t expected_bf16 = static_cast<uint16_t>(expected_bits >> 16);
    EXPECT_EQ(data[0], expected_bf16);
    EXPECT_EQ(data[count / 2], expected_bf16);
    EXPECT_EQ(data[count - 1], expected_bf16);
}

TEST_F(ShmemSpinMPITest, AllreduceFP16Repeated)
{
    // Verify epoch protocol works for repeated FP16 allreduces.
    // Uses FP16-exact values (small integers) to avoid FMA rounding mismatches.
    const size_t count = 256;
    for (int iter = 0; iter < 100; ++iter)
    {
        float val = static_cast<float>(rank_ + 1);
        uint16_t fp16_val = static_cast<uint16_t>(
            _cvtss_sh(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        std::vector<uint16_t> data(count, fp16_val);

        ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                        CollectiveDataType::FLOAT16,
                                        CollectiveOp::ALLREDUCE_SUM))
            << "FP16 repeated failed at iteration " << iter;

        // rank_sum_ is exact in FP16 for small integers
        EXPECT_FLOAT_EQ(_cvtsh_ss(data[0]), rank_sum_)
            << "FP16 repeated mismatch at iteration " << iter;
        EXPECT_FLOAT_EQ(_cvtsh_ss(data[count - 1]), rank_sum_)
            << "FP16 repeated last element at iter " << iter;
    }
}

TEST_F(ShmemSpinMPITest, AllreduceBF16Repeated)
{
    const size_t count = 256;
    for (int iter = 0; iter < 100; ++iter)
    {
        float val = static_cast<float>(rank_ + 1) * 4.0f;
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(float));
        uint16_t bf16_val = static_cast<uint16_t>(bits >> 16);
        std::vector<uint16_t> data(count, bf16_val);

        ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                        CollectiveDataType::BFLOAT16,
                                        CollectiveOp::ALLREDUCE_SUM))
            << "BF16 repeated failed at iteration " << iter;

        // Each rank contributes (rank+1)*4.0, sum = 4.0 * rank_sum_
        float expected_f = 4.0f * rank_sum_;
        uint32_t expected_bits;
        std::memcpy(&expected_bits, &expected_f, sizeof(float));
        uint16_t expected_bf16 = static_cast<uint16_t>(expected_bits >> 16);
        EXPECT_EQ(data[0], expected_bf16) << "BF16 repeated mismatch at iteration " << iter;
    }
}

TEST_F(ShmemSpinMPITest, SynchronizeWorks)
{
    ASSERT_TRUE(backend_->synchronize());
    ASSERT_TRUE(backend_->synchronize());
}

// ============================================================================
// Main: Custom MPI-aware GTest main
// ============================================================================

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
