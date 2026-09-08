/**
 * @file Test__ShmemSpinBackend.cpp
 * @brief Unit tests for ShmemSpinBackend shared-memory spin-wait allreduce
 *
 * Single-process tests (MPI_PROCS 1):
 *   - Arena layout / alignment assertions
 *   - Native shared-memory route detection
 *   - AVX-512 reduce correctness
 *
 * Two-process tests (MPI_PROCS 2):
 *   - Basic allreduce correctness
 *   - Repeated allreduce (epoch counter correctness)
 *   - Chunk-boundary and multi-chunk payloads
 *   - Small count (1 element)
 *   - Native rooted packed-record gather and all-dtype broadcast
 *   - Epoch composition across allreduce, gather, and broadcast
 *   - Delegation of remaining collective semantics to the general backend
 *   - Stress test (1000 consecutive allreduces)
 *   - Full Qwen3.6-35B prefill-sequence value and epoch correctness
 *
 * The separately selected ShmemSpinEconomyMPITest suite measures the latency
 * distribution against MPI. CTest registers it as Perf, never as Unit or
 * production preflight: competing compilation or tests invalidate its timing.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <numeric>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "collective/backends/ShmemSpinBackend.h"
#include "collective/backends/UPIBackend.h"
#include "collective/DeviceGroup.h"
#include "collective/GlobalTPContext.h"

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

static DeviceGroup makeCpuDeviceGroup(int size)
{
    DeviceGroup group;
    group.name = "test_shmem_spin";
    group.scope = CollectiveScope::LOCAL;
    for (int r = 0; r < size; ++r)
        group.devices.push_back(DeviceId::cpu());
    return group;
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
                               + static_cast<size_t>(n) *
                                     ShmemSpinArena::CHUNK_CAPACITY * sizeof(float); // buffers
        EXPECT_EQ(ShmemSpinArena::compute_size(n), expected_size)
            << "Mismatched arena size for " << n << " ranks";
    }
}

TEST(Test__ShmemSpinBackend, ArenaChunkCapacity)
{
    EXPECT_EQ(ShmemSpinArena::CHUNK_CAPACITY, 1u << 20);
    EXPECT_EQ(
        ShmemSpinArena::CHUNK_CAPACITY * sizeof(float),
        4u * 1024u * 1024u);
}

TEST(Test__ShmemSpinBackend, RootedPublicationTransportPolicyHasExactBoundaries)
{
    using Operation = CPURootedPublicationOperation;
    using Transport = CPURootedPublicationTransport;
    using Policy = CPURootedPublicationTransportPolicy;

    EXPECT_EQ(
        Policy::select(
            Operation::PackedRecordGather,
            Policy::kPackedGatherSharedMemoryMaxBytes),
        Transport::SharedMemoryLatency);
    EXPECT_EQ(
        Policy::select(
            Operation::PackedRecordGather,
            Policy::kPackedGatherSharedMemoryMaxBytes + 1u),
        Transport::MPIBandwidth);
    EXPECT_EQ(
        Policy::select(
            Operation::CompactOutputBroadcast,
            Policy::kBroadcastSharedMemoryMaxBytes),
        Transport::SharedMemoryLatency);
    EXPECT_EQ(
        Policy::select(
            Operation::CompactOutputBroadcast,
            Policy::kBroadcastSharedMemoryMaxBytes + 1u),
        Transport::MPIBandwidth);

    constexpr size_t qwen36_d_model = 2048u;
    constexpr size_t qwen36_top_k = 8u;
    constexpr size_t serial_decode_gather_bytes =
        qwen36_top_k * (qwen36_d_model + 1u) * sizeof(float);
    constexpr size_t two_row_gather_bytes =
        2u * serial_decode_gather_bytes;
    EXPECT_EQ(
        Policy::select(
            Operation::PackedRecordGather,
            serial_decode_gather_bytes),
        Transport::SharedMemoryLatency);
    EXPECT_EQ(
        Policy::select(
            Operation::PackedRecordGather,
            two_row_gather_bytes),
        Transport::MPIBandwidth);

    constexpr size_t four_row_broadcast_bytes =
        4u * qwen36_d_model * sizeof(float);
    constexpr size_t five_row_broadcast_bytes =
        5u * qwen36_d_model * sizeof(float);
    EXPECT_EQ(
        Policy::select(
            Operation::CompactOutputBroadcast,
            four_row_broadcast_bytes),
        Transport::SharedMemoryLatency);
    EXPECT_EQ(
        Policy::select(
            Operation::CompactOutputBroadcast,
            five_row_broadcast_bytes),
        Transport::MPIBandwidth);
}

TEST(Test__ShmemSpinBackend, SharedMemoryPathDetectionIsSizeTotal)
{
    // Create a backend with a null fallback (won't call MPI in this test)
    // We need MPI_COMM_WORLD for the fallback but we won't actually use it
    auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
    ShmemSpinBackend backend(999, mpiRank(), std::move(upi));

    EXPECT_TRUE(backend.usesSharedMemoryAllreduce(
        1, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.usesSharedMemoryAllreduce(
        ShmemSpinArena::CHUNK_CAPACITY,
        CollectiveDataType::FLOAT32,
        CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.usesSharedMemoryAllreduce(
        ShmemSpinArena::CHUNK_CAPACITY * 17u + 3u,
        CollectiveDataType::FLOAT32,
        CollectiveOp::ALLREDUCE_SUM));

    EXPECT_FALSE(backend.usesSharedMemoryAllreduce(
        2048, CollectiveDataType::INT32, CollectiveOp::ALLREDUCE_SUM));

    EXPECT_TRUE(backend.usesSharedMemoryAllreduce(
        2048, CollectiveDataType::FLOAT16, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.usesSharedMemoryAllreduce(
        2048, CollectiveDataType::BFLOAT16, CollectiveOp::ALLREDUCE_SUM));

    EXPECT_FALSE(backend.usesSharedMemoryAllreduce(
        2048, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_MAX));
    EXPECT_FALSE(backend.usesSharedMemoryAllreduce(
        2048, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_MIN));
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
    // Exercise the vectorized reduction at the exact protocol chunk boundary.
    const size_t count = ShmemSpinArena::CHUNK_CAPACITY;
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
    expectExactMatchU16(scalar_out.data(), avx512_out.data(), count, "FP16 chunk boundary");
}

TEST(Test__ShmemSpinBackend, ReduceBF16MaxCountBoundary)
{
    const size_t count = ShmemSpinArena::CHUNK_CAPACITY;
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
    expectExactMatchBF16(scalar_out.data(), avx512_out.data(), count, "BF16 chunk boundary");
}

// ============================================================================
// Native shared-memory path detection for FP16 / BF16
// ============================================================================

TEST(Test__ShmemSpinBackend, SharedMemoryPathDetectionFP16BF16IsSizeTotal)
{
    auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
    ShmemSpinBackend backend(994, mpiRank(), std::move(upi));

    EXPECT_TRUE(backend.usesSharedMemoryAllreduce(
        2048, CollectiveDataType::FLOAT16, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.usesSharedMemoryAllreduce(
        ShmemSpinArena::CHUNK_CAPACITY + 1,
        CollectiveDataType::FLOAT16,
        CollectiveOp::ALLREDUCE_SUM));

    EXPECT_TRUE(backend.usesSharedMemoryAllreduce(
        2048, CollectiveDataType::BFLOAT16, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_TRUE(backend.usesSharedMemoryAllreduce(
        ShmemSpinArena::CHUNK_CAPACITY + 1,
        CollectiveDataType::BFLOAT16,
        CollectiveOp::ALLREDUCE_SUM));

    EXPECT_FALSE(backend.usesSharedMemoryAllreduce(
        2048, CollectiveDataType::FLOAT16, CollectiveOp::ALLREDUCE_MAX));
    EXPECT_FALSE(backend.usesSharedMemoryAllreduce(
        2048, CollectiveDataType::BFLOAT16, CollectiveOp::ALLREDUCE_MAX));
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

        DeviceGroup group = makeCpuDeviceGroup(size_);

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

TEST_F(ShmemSpinMPITest, SharedMemoryNameUnlinkedAfterInitialize)
{
    const std::string name = backend_->shmName();
    ASSERT_FALSE(name.empty());

    errno = 0;
    int fd = shm_open(name.c_str(), O_RDWR, 0);
    EXPECT_EQ(fd, -1) << "ShmemSpinBackend should unlink its POSIX shm name after all ranks map it";
    EXPECT_EQ(errno, ENOENT) << "Unexpected shm_open errno for unlinked name " << name;
    if (fd >= 0)
        close(fd);
}

TEST(Test__ShmemSpinBackend, ReinitializeUsesFreshSharedMemoryName)
{
    const int rank = mpiRank();
    const int size = mpiSize();
    if (size < 2)
    {
        GTEST_SKIP() << "Requires MPI_PROCS >= 2";
    }

    const int domain_id = 424200;
    const auto create_backend = [&]() {
        auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
        return std::make_unique<ShmemSpinBackend>(domain_id, rank, std::move(upi));
    };

    DeviceGroup group = makeCpuDeviceGroup(size);

    auto first = create_backend();
    ASSERT_TRUE(first->initialize(group)) << first->lastError();
    const std::string first_name = first->shmName();
    ASSERT_FALSE(first_name.empty());
    first->shutdown();
    MPI_Barrier(MPI_COMM_WORLD);

    auto second = create_backend();
    ASSERT_TRUE(second->initialize(group)) << second->lastError();
    const std::string second_name = second->shmName();
    ASSERT_FALSE(second_name.empty());
    EXPECT_NE(second_name, first_name)
        << "Reinitializing the same domain must not reuse a deterministic shm name";
    second->shutdown();
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Test__ShmemSpinBackend, LegacyDeterministicNameDoesNotBlockInitialization)
{
    const int rank = mpiRank();
    const int size = mpiSize();
    if (size < 2)
    {
        GTEST_SKIP() << "Requires MPI_PROCS >= 2";
    }

    const int domain_id = 424201;
    const std::string legacy_name = "/llaminar_shmem_ar_" + std::to_string(domain_id);

    if (rank == 0)
    {
        shm_unlink(legacy_name.c_str());
        int fd = shm_open(legacy_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        ASSERT_GE(fd, 0) << "Failed to create stale legacy shm object: " << strerror(errno);
        ASSERT_EQ(ftruncate(fd, 4096), 0) << "Failed to size stale legacy shm object: " << strerror(errno);
        close(fd);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
    ShmemSpinBackend backend(domain_id, rank, std::move(upi));
    DeviceGroup group = makeCpuDeviceGroup(size);
    ASSERT_TRUE(backend.initialize(group)) << backend.lastError();
    EXPECT_NE(backend.shmName(), legacy_name)
        << "Backend should use a fresh per-run name instead of the legacy deterministic name";

    float value = static_cast<float>(rank + 1);
    ASSERT_TRUE(backend.allreduce(&value, 1, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_FLOAT_EQ(value, static_cast<float>(size * (size + 1)) / 2.0f);

    backend.shutdown();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        errno = 0;
        int unlink_result = shm_unlink(legacy_name.c_str());
        EXPECT_TRUE(unlink_result == 0 || errno == ENOENT)
            << "Failed to cleanup stale legacy shm object: " << strerror(errno);
    }
    MPI_Barrier(MPI_COMM_WORLD);
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

    // Each allreduce publishes both ready-to-read and read-complete epochs.
    EXPECT_EQ(backend_->currentEpoch(), 200u);
}

/**
 * @test NativeBroadcastIsByteExactForEveryTypeAndMultipleChunks
 *
 * Broadcast does not perform arithmetic, so its contract is byte identity for
 * every collective element representation. Small cases cover all dtypes and
 * rotate the root; the FP32 case exceeds one arena slot to prove bounded
 * chunking and source-buffer reuse.
 */
TEST_F(
    ShmemSpinMPITest,
    NativeBroadcastIsByteExactForEveryTypeAndMultipleChunks)
{
    struct BroadcastCase
    {
        CollectiveDataType dtype;
        size_t element_size;
    };
    const std::array<BroadcastCase, 5> cases = {{
        {CollectiveDataType::FLOAT32, 4u},
        {CollectiveDataType::FLOAT16, 2u},
        {CollectiveDataType::BFLOAT16, 2u},
        {CollectiveDataType::INT32, 4u},
        {CollectiveDataType::INT8, 1u},
    }};

    for (size_t case_index = 0u; case_index < cases.size(); ++case_index)
    {
        const int root = static_cast<int>(case_index %
                                          static_cast<size_t>(size_));
        constexpr size_t count = 257u;
        const size_t byte_count = count * cases[case_index].element_size;
        std::vector<uint8_t> payload(byte_count, 0u);
        std::vector<uint8_t> expected(byte_count);
        for (size_t byte = 0u; byte < byte_count; ++byte)
        {
            expected[byte] = static_cast<uint8_t>(
                (byte * 29u + case_index * 17u + 3u) & 0xffu);
        }
        if (rank_ == root)
            payload = expected;

        ASSERT_TRUE(backend_->broadcast(
            payload.data(),
            count,
            cases[case_index].dtype,
            root));
        EXPECT_EQ(payload, expected)
            << "native broadcast changed bytes for dtype case " << case_index;
    }

    if (size_ != 2)
        return;

    const size_t large_count = ShmemSpinArena::CHUNK_CAPACITY + 257u;
    std::vector<float> large_payload(large_count, -1.0f);
    if (rank_ == 0)
    {
        for (size_t element = 0u; element < large_count; ++element)
            large_payload[element] = static_cast<float>(element % 8191u);
    }
    ASSERT_TRUE(backend_->broadcast(
        large_payload.data(),
        large_count,
        CollectiveDataType::FLOAT32,
        0));
    EXPECT_FLOAT_EQ(large_payload.front(), 0.0f);
    EXPECT_FLOAT_EQ(
        large_payload[ShmemSpinArena::CHUNK_CAPACITY - 1u],
        static_cast<float>(
            (ShmemSpinArena::CHUNK_CAPACITY - 1u) % 8191u));
    EXPECT_FLOAT_EQ(
        large_payload[ShmemSpinArena::CHUNK_CAPACITY],
        static_cast<float>(ShmemSpinArena::CHUNK_CAPACITY % 8191u));
    EXPECT_FLOAT_EQ(
        large_payload.back(),
        static_cast<float>((large_count - 1u) % 8191u));
}

/**
 * @test NativePackedGatherIsSizeTotalAndComposesWithOtherProtocols
 *
 * A production packed route block can exceed the four-megabyte participant
 * slot. This test transfers one multi-chunk block per rank, verifies root-first
 * participant layout, then alternates allreduce, empty/non-empty gather, and
 * broadcast without barriers. It catches epoch divergence between protocol
 * implementations and stale payload reuse at operation boundaries.
 */
TEST_F(
    ShmemSpinMPITest,
    NativePackedGatherIsSizeTotalAndComposesWithOtherProtocols)
{
    if (size_ != 2)
        GTEST_SKIP() << "Multi-chunk rooted publication economy targets two sockets";

    constexpr size_t record_width = 17u;
    const size_t base_records =
        ShmemSpinArena::CHUNK_CAPACITY / record_width + 3u;
    const size_t local_records = base_records + static_cast<size_t>(rank_);
    const size_t root_capacity = 2u * base_records + 1u;
    std::vector<float> local(local_records * record_width);
    for (size_t element = 0u; element < local.size(); ++element)
    {
        local[element] = static_cast<float>(
            static_cast<size_t>(rank_) * 4u *
                ShmemSpinArena::CHUNK_CAPACITY +
            element);
    }
    std::vector<float> gathered;
    if (rank_ == 0)
        gathered.assign(root_capacity * record_width, -1.0f);

    size_t gathered_records = 0u;
    ASSERT_TRUE(backend_->gatherVariableFloatRecordsToRoot(
        local.data(),
        local_records,
        rank_ == 0 ? gathered.data() : nullptr,
        root_capacity,
        record_width,
        0,
        gathered_records));
    if (rank_ == 0)
    {
        ASSERT_EQ(gathered_records, root_capacity);
        EXPECT_TRUE(std::equal(
            local.begin(),
            local.end(),
            gathered.begin()));
        const size_t peer_offset = local.size();
        EXPECT_FLOAT_EQ(
            gathered[peer_offset],
            static_cast<float>(4u * ShmemSpinArena::CHUNK_CAPACITY));
        EXPECT_FLOAT_EQ(
            gathered[peer_offset +
                     ShmemSpinArena::CHUNK_CAPACITY - 1u],
            static_cast<float>(
                5u * ShmemSpinArena::CHUNK_CAPACITY - 1u));
        EXPECT_FLOAT_EQ(
            gathered.back(),
            static_cast<float>(
                4u * ShmemSpinArena::CHUNK_CAPACITY +
                (base_records + 1u) * record_width - 1u));
    }

    constexpr size_t small_record_width = 3u;
    for (int iteration = 0; iteration < 100; ++iteration)
    {
        std::array<float, 13> reduced{};
        reduced.fill(static_cast<float>(rank_ + 1));
        ASSERT_TRUE(backend_->allreduce(
            reduced.data(),
            reduced.size(),
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
        EXPECT_FLOAT_EQ(reduced.front(), rank_sum_);
        EXPECT_FLOAT_EQ(reduced.back(), rank_sum_);

        /*
         * Exercise every two-rank count combination, including repeated
         * all-empty transactions. The asymmetric periods force a participant
         * to move quickly into the next transaction while its peer may still
         * be reading the reusable count slot, which directly regresses the
         * metadata-lifetime race found in the GlobalTP production wrapper.
         */
        const bool contributes = rank_ == 0
                                     ? iteration % 3 != 0
                                     : iteration % 2 != 0;
        const size_t record_count = contributes ? 1u : 0u;
        const std::array<float, small_record_width> record = {
            static_cast<float>(1000 * iteration + 100 * rank_ + 1),
            static_cast<float>(1000 * iteration + 100 * rank_ + 2),
            static_cast<float>(1000 * iteration + 100 * rank_ + 3),
        };
        std::array<float, 2u * small_record_width> root_records{};
        root_records.fill(-9.0f);
        gathered_records = 0u;
        ASSERT_TRUE(backend_->gatherVariableFloatRecordsToRoot(
            contributes ? record.data() : nullptr,
            record_count,
            rank_ == 0 ? root_records.data() : nullptr,
            2u,
            small_record_width,
            0,
            gathered_records));
        if (rank_ == 0)
        {
            const size_t expected_records =
                static_cast<size_t>(iteration % 3 != 0) +
                static_cast<size_t>(iteration % 2 != 0);
            EXPECT_EQ(gathered_records, expected_records);
        }

        std::array<uint32_t, 19> broadcast_payload{};
        if (rank_ == 1)
        {
            for (size_t element = 0u;
                 element < broadcast_payload.size();
                 ++element)
            {
                broadcast_payload[element] =
                    static_cast<uint32_t>(iteration * 100 + element);
            }
        }
        ASSERT_TRUE(backend_->broadcast(
            broadcast_payload.data(),
            broadcast_payload.size(),
            CollectiveDataType::INT32,
            1));
        EXPECT_EQ(
            broadcast_payload.front(),
            static_cast<uint32_t>(iteration * 100));
        EXPECT_EQ(
            broadcast_payload.back(),
            static_cast<uint32_t>(
                iteration * 100 + broadcast_payload.size() - 1u));
    }
}

TEST_F(ShmemSpinMPITest, MaxCountBoundary)
{
    // Test at exactly one complete chunk.
    const size_t count = ShmemSpinArena::CHUNK_CAPACITY;
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

TEST_F(ShmemSpinMPITest, MultiChunkAllreduceRemainsOnSharedMemoryPath)
{
    const size_t count = ShmemSpinArena::CHUNK_CAPACITY + 100;
    std::vector<float> data(count, static_cast<float>(rank_ + 1));

    ASSERT_TRUE(backend_->allreduce(data.data(), count,
                                    CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_SUM));

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], rank_sum_) << "Mismatch at index " << i;
    }
}

/** @brief Preserve full-sequence arithmetic coverage independently of timing. */
TEST_F(ShmemSpinMPITest, Qwen36PrefillSequencePreservesValues)
{
    if (size_ != 2)
        GTEST_SKIP() << "The production sequence models two CPU participants";

    constexpr size_t count = 434u * 2048u;
    constexpr int repetitions = 121;
    std::vector<float> data(count);
    for (int repetition = 0; repetition < repetitions; ++repetition)
    {
        // Vary successive publications so an old epoch cannot satisfy this
        // check merely by returning the previous reduction's identical bytes.
        const float offset = static_cast<float>(repetition);
        std::fill(data.begin(), data.end(), static_cast<float>(rank_ + 1) + offset);
        ASSERT_TRUE(backend_->allreduce(
            data.data(), count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
        const float expected = rank_sum_ + static_cast<float>(size_) * offset;
        EXPECT_TRUE(std::all_of(data.begin(), data.end(),
                                [expected](float value) { return value == expected; }))
            << "Full-buffer mismatch at prefill reduction " << repetition;
    }
}

/**
 * @brief Isolate competitive measurements from the functional Unit inventory.
 *
 * Reuses the same backend/rank lifecycle as the functional tests. The CMake
 * Perf registration selects this suite exclusively and runs it serially.
 */
class ShmemSpinEconomyMPITest : public ShmemSpinMPITest {};

/**
 * @brief Guard the complete Qwen3.6-35B prefill collective sequence against
 *        latency tails and MPI delegation.
 *
 * The production graph performs 121 SUM reductions of 434-by-2048 FP32
 * activations. A size threshold once routed all of them through MPI, consuming
 * roughly half a second per prefill. A median-only microbenchmark can still
 * miss the multi-millisecond tails that dominate a 121-call inference graph,
 * so this regression records one uninterrupted production-sized sequence and
 * evaluates p50, p95, p99, maximum, and aggregate latency.
 *
 * Each backend operation is itself the rank rendezvous. Per-sample MPI barriers
 * or timing reductions would perturb the protocol being measured, so rank-local
 * samples are combined with one vector MPI reduction only after the sequence
 * completes. Buffer initialization and the one initial alignment barrier remain
 * outside every timed interval.
 */
TEST_F(ShmemSpinEconomyMPITest, Qwen36PrefillSequenceHasBoundedTailAndBeatsMPI)
{
    if (size_ != 2)
        GTEST_SKIP() << "Dual-socket economy contract requires exactly two ranks";

    constexpr size_t count = 434u * 2048u;
    constexpr int repetitions = 121;
    std::vector<float> data(count);

    const auto measure_slowest_rank_us = [&](bool native_shared_memory) {
        std::vector<double> local_samples(static_cast<size_t>(repetitions));
        std::vector<double> slowest_rank_samples(static_cast<size_t>(repetitions));
        MPI_Barrier(MPI_COMM_WORLD);

        for (int repetition = 0; repetition < repetitions; ++repetition)
        {
            std::fill(data.begin(), data.end(), static_cast<float>(rank_ + 1));

            const auto start = std::chrono::steady_clock::now();
            if (native_shared_memory)
            {
                EXPECT_TRUE(backend_->allreduce(
                    data.data(),
                    count,
                    CollectiveDataType::FLOAT32,
                    CollectiveOp::ALLREDUCE_SUM));
            }
            else
            {
                EXPECT_EQ(
                    MPI_Allreduce(
                        MPI_IN_PLACE,
                        data.data(),
                        static_cast<int>(count),
                        MPI_FLOAT,
                        MPI_SUM,
                        MPI_COMM_WORLD),
                    MPI_SUCCESS);
            }
            const auto stop = std::chrono::steady_clock::now();
            local_samples[static_cast<size_t>(repetition)] =
                std::chrono::duration<double, std::micro>(stop - start).count();

            EXPECT_FLOAT_EQ(data.front(), rank_sum_);
            EXPECT_FLOAT_EQ(data[count / 2], rank_sum_);
            EXPECT_FLOAT_EQ(data.back(), rank_sum_);
        }

        EXPECT_EQ(
            MPI_Allreduce(
                local_samples.data(),
                slowest_rank_samples.data(),
                repetitions,
                MPI_DOUBLE,
                MPI_MAX,
                MPI_COMM_WORLD),
            MPI_SUCCESS);
        std::sort(slowest_rank_samples.begin(), slowest_rank_samples.end());
        return slowest_rank_samples;
    };

    // Prime both implementations before collecting their medians.
    std::fill(data.begin(), data.end(), static_cast<float>(rank_ + 1));
    ASSERT_TRUE(backend_->allreduce(
        data.data(), count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    std::fill(data.begin(), data.end(), static_cast<float>(rank_ + 1));
    ASSERT_EQ(
        MPI_Allreduce(
            MPI_IN_PLACE,
            data.data(),
            static_cast<int>(count),
            MPI_FLOAT,
            MPI_SUM,
            MPI_COMM_WORLD),
        MPI_SUCCESS);

    const std::vector<double> native_us = measure_slowest_rank_us(true);
    const std::vector<double> blocking_mpi_us = measure_slowest_rank_us(false);
    const auto percentile = [](const std::vector<double> &samples, size_t numerator) {
        const size_t rank =
            (numerator * samples.size() + 99u) / 100u;
        return samples[std::max<size_t>(1u, rank) - 1u];
    };
    const double native_p50_us = percentile(native_us, 50u);
    const double native_p95_us = percentile(native_us, 95u);
    const double native_p99_us = percentile(native_us, 99u);
    const double native_max_us = native_us.back();
    const double native_total_us =
        std::accumulate(native_us.begin(), native_us.end(), 0.0);
    const double blocking_mpi_p50_us = percentile(blocking_mpi_us, 50u);
    const double blocking_mpi_p95_us = percentile(blocking_mpi_us, 95u);
    if (rank_ == 0)
    {
        std::cout << "Qwen3.6-35B prefill allreduce sequence: shared_p50="
                  << native_p50_us << " us shared_p95=" << native_p95_us
                  << " us shared_p99=" << native_p99_us
                  << " us shared_max=" << native_max_us
                  << " us shared_total=" << native_total_us
                  << " us blocking_mpi_p50=" << blocking_mpi_p50_us
                  << " us blocking_mpi_p95=" << blocking_mpi_p95_us << '\n';
    }

    EXPECT_LT(native_p50_us, blocking_mpi_p50_us)
        << "The ordinary production-sized shared-memory reduction must beat MPI";
    EXPECT_LT(native_p95_us, blocking_mpi_p95_us)
        << "The shared-memory p95 must remain economical across the complete "
           "121-call production sequence; a good median cannot hide tail stalls";
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
    // Boundary: exactly one chunk verifies no buffer overrun.
    const size_t count = ShmemSpinArena::CHUNK_CAPACITY;
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
    const size_t count = ShmemSpinArena::CHUNK_CAPACITY;
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
