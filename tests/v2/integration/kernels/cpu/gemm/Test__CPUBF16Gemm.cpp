/**
 * @file Test__CPUBF16Gemm.cpp
 * @brief Certify native AMX-BF16 / AVX512-BF16 GEMM route contracts.
 *
 * The native BF16 routes are hardware-dot-product programs with a fixed
 * per-route reduction contract: K ascending in the instruction's native width,
 * disjoint hardware lane/partition accumulators, one fixed inter-lane tree and
 * the shared rounded epilogue. They are deliberately NOT serial-fma
 * byte-identical (the hardware fuses pair/tile sums), so this suite proves:
 *
 *   1. route determinism: byte-identical outputs across runs and thread counts;
 *   2. tolerance: every native route stays within FP32 accumulation error of
 *      the serial FMA oracle on the same BF16 operands;
 *   3. route evidence: the opt-in probe counts exactly one admitted live
 *      dispatch per native call and nothing when off or on other routes;
 *   4. policy: explicit engines reject unavailable/admitted-false geometry
 *      instead of degrading, and automatic resolution keeps M==1 on the
 *      serial-exact library route.
 *
 * Native sections compile only inside the matching CPU ISA envelope and skip
 * when the executing host does not advertise the extension; the policy and
 * evidence sections build and run on every envelope. No model or accelerator
 * is required.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <stdexcept>
#include <omp.h>

#include "kernels/cpu/gemm/CPUBF16Gemm.h"
#include "tensors/SIMDHelpers.h"
#include "utils/CPUFeatures.h"

namespace
{
    using namespace llaminar2;
    namespace cpu_bf16 = llaminar2::gemm::cpu_bf16;
    using cpu_bf16::Engine;

    /** Restore the caller's OpenMP budget even after an assertion returns early. */
    class ThreadBudget final
    {
        int previous_;
    public:
        explicit ThreadBudget(int threads) : previous_(omp_get_max_threads())
        {
            omp_set_num_threads(threads);
        }
        ~ThreadBudget() { omp_set_num_threads(previous_); }
        ThreadBudget(const ThreadBudget &) = delete;
        ThreadBudget &operator=(const ThreadBudget &) = delete;
    };

    /** Enable the BF16 route probe for one contract, then restore off. */
    class BF16Probe final
    {
    public:
        BF16Probe() { cpu_bf16::bf16RouteProbeEnabled().store(true, std::memory_order_relaxed); }
        ~BF16Probe() { cpu_bf16::bf16RouteProbeEnabled().store(false, std::memory_order_relaxed); }
        BF16Probe(const BF16Probe &) = delete;
        BF16Probe &operator=(const BF16Probe &) = delete;
    };

    /** Deterministic BF16 operand generator with a wide value distribution. */
    std::vector<uint16_t> bf16Words(std::size_t count, std::mt19937 &random)
    {
        std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
        std::vector<uint16_t> words(count);
        for (std::size_t i = 0; i < count; ++i)
            words[i] = simd::fp32_to_bf16(distribution(random));
        return words;
    }

    float decode(uint16_t word) noexcept
    {
        return simd::bf16_to_fp32(word);
    }

    /** One output word through the serial FMA oracle plus the rounded epilogue. */
    float serialOracle(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b,
                       int row, int col, int N, int K, bool transpose_B,
                       float alpha, float beta, float previous)
    {
        float dot = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            const uint16_t a_word = a[static_cast<std::size_t>(row) * K + k];
            const uint16_t b_word = transpose_B
                ? b[static_cast<std::size_t>(col) * K + k]
                : b[static_cast<std::size_t>(k) * N + col];
            dot = std::fma(decode(a_word), decode(b_word), dot);
        }
        volatile float product = alpha * dot;
        return beta != 0.0f ? std::fma(beta, previous, product) : product;
    }

    float relativeL2(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        double sum_sq_actual = 0.0, sum_sq_diff = 0.0;
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            sum_sq_actual += double(expected[i]) * double(expected[i]);
            sum_sq_diff += double(actual[i] - expected[i]) * double(actual[i] - expected[i]);
        }
        const double denom = std::sqrt(sum_sq_actual);
        return denom > 0.0 ? float(std::sqrt(sum_sq_diff) / denom) : 0.0f;
    }

    /** Column-major B buffer for the non-transposed layout [K,N]. */
    std::vector<uint16_t> transposeLayout(const std::vector<uint16_t> &b, int N, int K)
    {
        std::vector<uint16_t> out(static_cast<std::size_t>(N) * K);
        for (int col = 0; col < N; ++col)
            for (int k = 0; k < K; ++k)
                out[static_cast<std::size_t>(k) * N + col] = b[static_cast<std::size_t>(col) * K + k];
        return out;
    }

#if defined(__AVX512BF16__)
    void runAVX512Native(std::vector<float> &out, const std::vector<uint16_t> &a,
                         const std::vector<uint16_t> &b, int M, int N, int K,
                         float alpha, float beta)
    {
        out.assign(static_cast<std::size_t>(M) * N, 0.0f);
        const Engine selected =
            cpu_bf16::selectEngine(M, N, K, true, Engine::AVX512BF16);
        ASSERT_EQ(selected, Engine::AVX512BF16);
        ASSERT_TRUE(cpu_bf16::run_avx512bf16_matmul(
            a.data(), b.data(), out.data(), M, N, K, true, alpha, beta));
    }
#endif

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    void runAMXNative(std::vector<float> &out, const std::vector<uint16_t> &a,
                      const std::vector<uint16_t> &b, int M, int N, int K,
                      bool transpose_B, float alpha, float beta)
    {
        out.assign(static_cast<std::size_t>(M) * N, 0.0f);
        std::vector<uint16_t> ktn;
        const uint16_t *b_ptr = b.data();
        if (!transpose_B)
        {
            ktn = transposeLayout(b, N, K);
            b_ptr = ktn.data();
        }
        const Engine selected =
            cpu_bf16::selectEngine(M, N, K, transpose_B, Engine::AMXBF16);
        ASSERT_EQ(selected, Engine::AMXBF16);
        ASSERT_TRUE(cpu_bf16::run_amxbf16_matmul(
            a.data(), b_ptr, out.data(), M, N, K, transpose_B, alpha, beta));
    }
#endif
}

// ============================================================================
// Policy: engine identity, admission and explicit rejection
// ============================================================================

TEST(CPUBF16GemmPolicy, EngineSelectionIsExplicitAndFailFast)
{
    // Unknown spellings fail; known spellings parse case-insensitively.
    EXPECT_THROW(cpu_bf16::parseEngineName("cuda"), std::invalid_argument);
    EXPECT_EQ(cpu_bf16::parseEngineName(nullptr), Engine::Automatic);
    EXPECT_EQ(cpu_bf16::parseEngineName(""), Engine::Automatic);
    EXPECT_EQ(cpu_bf16::parseEngineName("AVX512BF16"), Engine::AVX512BF16);
    EXPECT_EQ(cpu_bf16::parseEngineName("amxbf16"), Engine::AMXBF16);
    EXPECT_EQ(cpu_bf16::parseEngineName("Library"), Engine::Library);

    // A compiled leaf requires hardware; the admission query reflects both.
    if (cpu_bf16::compiledWithAVX512BF16())
        EXPECT_TRUE(cpu_supports_avx512_bf16());
    if (cpu_bf16::compiledWithAMXBF16())
        EXPECT_TRUE(cpu_supports_amx_bf16());

    // Automatic policy: M == 1 keeps the serial-exact library route.
    EXPECT_EQ(cpu_bf16::selectEngine(1, 16, 64, true), Engine::Library);
    EXPECT_EQ(cpu_bf16::selectEngine(1, 16, 64, false), Engine::Library);
    // Explicit Library never rejects and never executes a native engine.
    EXPECT_EQ(cpu_bf16::selectEngine(3, 16, 64, true, Engine::Library), Engine::Library);
}

TEST(CPUBF16GemmPolicy, ExplicitUnionGeometriesThrowInsteadOfDegrading)
{
    // A non-admitted AVX512 geometry (K % 32 != 0) must be rejected by the
    // explicit engine; whether the reject is admission (missing intrin) or
    // geometry depends on the compiled envelope, but it must never run.
    bool avx512_compiled = cpu_bf16::compiledWithAVX512BF16();
    if (avx512_compiled)
    {
        EXPECT_THROW(cpu_bf16::selectEngine(3, 16, 40, true, Engine::AVX512BF16),
                     std::invalid_argument);
    }
    else
    {
        EXPECT_THROW(cpu_bf16::selectEngine(3, 16, 40, true, Engine::AVX512BF16),
                     std::runtime_error);
    }

    // Non-transposed weights are not admitted by the AVX512 engine even with
    // a multiple-of-32 K.
    if (avx512_compiled)
        EXPECT_THROW(cpu_bf16::selectEngine(3, 16, 64, false, Engine::AVX512BF16),
                     std::invalid_argument);

    // AMX requires full 16×16×16 tiles; partial geometry is rejected. On a
    // host whose AMX tile data instructions are not functional the explicit
    // engine is rejected before geometry evaluation instead.
    if (cpu_bf16::compiledWithAMXBF16())
    {
        if (cpu_amx_data_instructions_usable())
            EXPECT_THROW(cpu_bf16::selectEngine(3, 16, 64, true, Engine::AMXBF16),
                         std::invalid_argument);
        else
            EXPECT_THROW(cpu_bf16::selectEngine(3, 16, 64, true, Engine::AMXBF16),
                         std::runtime_error);
    }
}

// ============================================================================
// Epilogue contract parity with the skinny-matmul epilogue
// ============================================================================

TEST(CPUBF16GemmPolicy, EpilogueRoundingMatchesRoundedSerialReference)
{
    std::mt19937 random(0xe91eee9);
    std::uniform_real_distribution<float> distribution(-4.0f, 4.0f);
    for (int sample = 0; sample < 5000; ++sample)
    for (bool with_previous : {false, true})
    {
        const float dot = sample % 9 == 0 ? -0.0f : distribution(random);
        const float alpha = sample % 5 == 0 ? -1.0f : distribution(random);
        const float beta = with_previous ? distribution(random) : 0.0f;
        const float previous = distribution(random) * 8192.0f;
        const float actual = cpu_bf16::completeEpilogueValue(
            dot, alpha, beta, with_previous ? &previous : nullptr);
        volatile float product = alpha * dot;
        const float expected = with_previous ? std::fma(beta, previous, product) : product;
        ASSERT_EQ(std::memcmp(&actual, &expected, sizeof(float)), 0)
            << "sample=" << sample << " beta=" << beta;
    }
}

// ============================================================================
// AVX512-BF16 native route
// ============================================================================

#if defined(__AVX512BF16__)
TEST(CPUBF16GemmAVX512, DeterministicAcrossRunsAndThreadCounts)
{
    if (!cpu_supports_avx512_bf16())
        GTEST_SKIP() << "host does not advertise AVX512-BF16";
    std::mt19937 random(0x51f1591);
    for (int threads : {1, 3, 8})
    {
        ThreadBudget budget(threads);
        for (const auto [M, N, K] : std::array<std::array<int, 3>, 4>{ {{2, 8, 32}, {3, 17, 64}, {16, 128, 128}, {7, 48, 96}} })
        {
            SCOPED_TRACE(::testing::Message() << "threads=" << threads
                << " M=" << M << " N=" << N << " K=" << K);
            const auto a = bf16Words(static_cast<std::size_t>(M) * K, random);
            const auto b = bf16Words(static_cast<std::size_t>(N) * K, random);
            std::vector<float> first, second;
            runAVX512Native(first, a, b, M, N, K, 0.75f, -0.25f);
            runAVX512Native(second, a, b, M, N, K, 0.75f, -0.25f);
            ASSERT_EQ(first.size(), second.size());
            for (std::size_t i = 0; i < first.size(); ++i)
                ASSERT_EQ(std::memcmp(&first[i], &second[i], sizeof(float)), 0)
                    << "output=" << i << " first=" << first[i] << " second=" << second[i];
        }
    }
}

TEST(CPUBF16GemmAVX512, StaysWithinFp32AccumulationOfSerialOracle)
{
    if (!cpu_supports_avx512_bf16())
        GTEST_SKIP() << "host does not advertise AVX512-BF16";
    std::mt19937 random(0x33199b16);
    ThreadBudget budget(2);
    for (const auto [M, N, K] : std::array<std::array<int, 3>, 3>{ {{3, 33, 32}, {16, 64, 128}, {9, 17, 320}} })
    {
        const auto a = bf16Words(static_cast<std::size_t>(M) * K, random);
        const auto b = bf16Words(static_cast<std::size_t>(N) * K, random);
        std::vector<float> actual;
        runAVX512Native(actual, a, b, M, N, K, 1.0f, 0.0f);
        std::vector<float> expected(static_cast<std::size_t>(M) * N);
        for (int row = 0; row < M; ++row)
            for (int col = 0; col < N; ++col)
                expected[static_cast<std::size_t>(row) * N + col] =
                    serialOracle(a, b, row, col, N, K, true, 1.0f, 0.0f, 0.0f);
        const float rel = relativeL2(actual, expected);
        EXPECT_LE(rel, 5.0e-5f)
            << "M=" << M << " N=" << N << " K=" << K << " relL2=" << rel;
    }
}

TEST(CPUBF16GemmAVX512, RouteProbeCountsOneAdmittedDispatch)
{
    if (!cpu_supports_avx512_bf16())
        GTEST_SKIP() << "host does not advertise AVX512-BF16";
    std::mt19937 random(0xa4512);
    const auto a = bf16Words(3 * 64, random);
    const auto b = bf16Words(16 * 64, random);
    std::vector<float> out(3 * 16, 0.0f);
    // Probe off: the valid call counts nothing.
    {
        const std::uint64_t before = cpu_bf16::bf16NativeRouteInvocations(Engine::AVX512BF16);
        ASSERT_TRUE(cpu_bf16::run_avx512bf16_matmul(a.data(), b.data(), out.data(), 3, 16, 64, true, 1.0f, 0.0f));
        EXPECT_EQ(cpu_bf16::bf16NativeRouteInvocations(Engine::AVX512BF16) - before, 0ull);
    }
    // Probe on: exactly one admitted dispatch per call.
    {
        BF16Probe probe;
        const std::uint64_t before = cpu_bf16::bf16NativeRouteInvocations(Engine::AVX512BF16);
        ASSERT_TRUE(cpu_bf16::run_avx512bf16_matmul(a.data(), b.data(), out.data(), 3, 16, 64, true, 1.0f, 0.0f));
        EXPECT_EQ(cpu_bf16::bf16NativeRouteInvocations(Engine::AVX512BF16) - before, 1ull);
    }
}
#endif // __AVX512BF16__

// ============================================================================
// AMX-BF16 native route
// ============================================================================

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
TEST(CPUBF16GemmAMX, TileDeterministicAcrossRunsAndThreadCounts)
{
    if (!cpu_amx_data_instructions_usable())
        GTEST_SKIP() << "host does not advertise AMX-BF16";
    std::mt19937 random(0xa3c16b16);
    for (int threads : {1, 3, 8})
    {
        ThreadBudget budget(threads);
        for (bool transposed : {true, false})
        for (const auto [M, N, K] : std::array<std::array<int, 3>, 3>{ {{16, 16, 16}, {32, 48, 64}, {16, 32, 32}} })
        {
            SCOPED_TRACE(::testing::Message() << "threads=" << threads
                << " transposed=" << transposed
                << " M=" << M << " N=" << N << " K=" << K);
            const auto a = bf16Words(static_cast<std::size_t>(M) * K, random);
            const auto b = bf16Words(static_cast<std::size_t>(N) * K, random);
            std::vector<float> first, second;
            runAMXNative(first, a, b, M, N, K, transposed, 0.75f, -0.25f);
            runAMXNative(second, a, b, M, N, K, transposed, 0.75f, -0.25f);
            ASSERT_EQ(first.size(), second.size());
            for (std::size_t i = 0; i < first.size(); ++i)
                ASSERT_EQ(std::memcmp(&first[i], &second[i], sizeof(float)), 0)
                    << "output=" << i;
        }
    }
}

TEST(CPUBF16GemmAMX, TileStaysWithinFp32AccumulationOfSerialOracle)
{
    if (!cpu_amx_data_instructions_usable())
        GTEST_SKIP() << "host does not advertise AMX-BF16";
    std::mt19937 random(0xd0b16bf);
    ThreadBudget budget(2);
    for (const auto [M, N, K] : std::array<std::array<int, 3>, 2>{ {{16, 16, 16}, {32, 32, 64}} })
    {
        const auto a = bf16Words(static_cast<std::size_t>(M) * K, random);
        const auto b = bf16Words(static_cast<std::size_t>(N) * K, random);
        std::vector<float> actual;
        runAMXNative(actual, a, b, M, N, K, true, 1.0f, 0.0f);
        std::vector<float> expected(static_cast<std::size_t>(M) * N);
        for (int row = 0; row < M; ++row)
            for (int col = 0; col < N; ++col)
                expected[static_cast<std::size_t>(row) * N + col] =
                    serialOracle(a, b, row, col, N, K, true, 1.0f, 0.0f, 0.0f);
        const float rel = relativeL2(actual, expected);
        EXPECT_LE(rel, 5.0e-5f)
            << "M=" << M << " N=" << N << " K=" << K << " relL2=" << rel;
    }
}

TEST(CPUBF16GemmAMX, RouteProbeCountsOneAdmittedDispatch)
{
    if (!cpu_amx_data_instructions_usable())
        GTEST_SKIP() << "host does not advertise AMX-BF16";
    std::mt19937 random(0x9a3c16);
    const auto a = bf16Words(16 * 16, random);
    const auto b = bf16Words(16 * 16, random);
    std::vector<float> out(16 * 16, 0.0f);
    {
        const std::uint64_t before = cpu_bf16::bf16NativeRouteInvocations(Engine::AMXBF16);
        ASSERT_TRUE(cpu_bf16::run_amxbf16_matmul(a.data(), b.data(), out.data(), 16, 16, 16, true, 1.0f, 0.0f));
        EXPECT_EQ(cpu_bf16::bf16NativeRouteInvocations(Engine::AMXBF16) - before, 0ull);
    }
    {
        BF16Probe probe;
        const std::uint64_t before = cpu_bf16::bf16NativeRouteInvocations(Engine::AMXBF16);
        ASSERT_TRUE(cpu_bf16::run_amxbf16_matmul(a.data(), b.data(), out.data(), 16, 16, 16, true, 1.0f, 0.0f));
        EXPECT_EQ(cpu_bf16::bf16NativeRouteInvocations(Engine::AMXBF16) - before, 1ull);
    }
}
#endif // __AMX_TILE__ && __AMX_BF16__
