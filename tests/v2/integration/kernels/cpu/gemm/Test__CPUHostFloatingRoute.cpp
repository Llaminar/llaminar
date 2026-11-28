/**
 * @file Test__CPUHostFloatingRoute.cpp
 * @brief Certify the native FP32 floating GEMM route against the serial-FMA
 *        oracle at canonical serving shapes on the host-supported instruction
 *        envelope.
 *
 * The CPU floating GEMM route is a serial-FMA program by design: every output
 * word visits K in ascending order through independent fma() calls and the
 * shared rounded epilogue. Therefore the route must be byte-identical to the
 * serial-FMA oracle (a stronger contract than the 5e-5 relative-L2 gate used
 * for the fused AMX-BF16 / AVX512-BF16 hardware routes) and byte-identical
 * across thread counts, because the output words are partitioned, not
 * reduced. This suite proves both contracts at the four canonical serving
 * shapes used by the native-vs-oneDNN route comparison, and additionally
 * covers the transposed weight contract and the alpha/beta epilogue at
 * serving scale.
 *
 * No model or accelerator is required; the test is device-free and exercises
 * only the native CPU route against independently computed reference words.
 */

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <omp.h>

#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"

namespace
{
    using namespace llaminar2;

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

    /** Canonical serving shapes of the native-vs-oneDNN route comparison. */
    constexpr int kServingShapes[][3] =
        {{16, 4096, 4096}, {32, 4096, 4096}, {96, 4096, 4096}, {32, 8192, 8192}};

    /** Deterministic wide-value operand generator (mirrors the sibling suites). */
    std::vector<float> fp32Values(std::size_t count, std::uint32_t seed)
    {
        std::mt19937 random(seed);
        std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
        std::vector<float> values(count);
        for (float &v : values)
            v = distribution(random);
        return values;
    }

    /** One output word through the serial-FMA oracle plus the rounded epilogue. */
    float serialOracle(const std::vector<float> &a, const std::vector<float> &b,
                       int row, int col, int N, int K, bool transpose_B,
                       float alpha, float beta, float previous)
    {
        float dot = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            const float a_value = a[static_cast<std::size_t>(row) * K + k];
            const float b_value = transpose_B
                ? b[static_cast<std::size_t>(col) * K + k]
                : b[static_cast<std::size_t>(k) * N + col];
            dot = std::fma(a_value, b_value, dot);
        }
        // Volatile belongs only in the independent oracle; production's rounded
        // product is register-resident, without a store/reload.
        volatile float product = alpha * dot;
        return beta != 0.0f ? std::fma(beta, previous, product) : product;
    }

    double relativeL2(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        double sum_sq_actual = 0.0, sum_sq_diff = 0.0;
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            sum_sq_actual += double(expected[i]) * double(expected[i]);
            sum_sq_diff += double(actual[i] - expected[i]) * double(actual[i] - expected[i]);
        }
        const double denom = std::sqrt(sum_sq_actual);
        return denom > 0.0 ? std::sqrt(sum_sq_diff) / denom : 0.0;
    }

    /** Every output word of `actual` equals every byte of `expected`. */
    void expectByteIdentical(const std::vector<float> &actual,
                             const std::vector<float> &expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0)
            {
                if (mismatches < 8)
                    ADD_FAILURE() << "word " << i << ": route=" << actual[i]
                                  << " oracle=" << expected[i]
                                  << " bytes=" << std::memcmp(&actual[i], &expected[i], sizeof(float));
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0u);
    }
}

// ============================================================================
// Serial-oracle contract at the canonical serving shapes
// ============================================================================

TEST(CPUFloatingRouteSerialOracle, ServingShapesMatchSerialFmaByteExact)
{
    for (const auto &shape : kServingShapes)
    {
        const int M = shape[0], N = shape[1], K = shape[2];
        SCOPED_TRACE(::testing::Message() << "M=" << M << " N=" << N << " K=" << K);
        const std::vector<float> a = fp32Values(static_cast<std::size_t>(M) * K, 0xc0ffee);
        const std::vector<float> b = fp32Values(static_cast<std::size_t>(N) * K, 0x5ec0ffee);
        std::vector<float> actual(static_cast<std::size_t>(M) * N, 0.0f);

        ThreadBudget threads(8);
        ASSERT_TRUE(gemm::run_fp32_skinny_matmul(
            a.data(), b.data(), actual.data(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr));

        // Independent single-threaded serial-FMA reference (route is serial by design).
        std::vector<float> oracle(static_cast<std::size_t>(M) * N, 0.0f);
        for (int row = 0; row < M; ++row)
            for (int col = 0; col < N; ++col)
                oracle[static_cast<std::size_t>(row) * N + col] =
                    serialOracle(a, b, row, col, N, K, true, 1.0f, 0.0f, 0.0f);

        expectByteIdentical(actual, oracle);
        EXPECT_LE(relativeL2(actual, oracle), 1e-5);
    }
}

// ============================================================================
// Determinism contract across thread counts
// ============================================================================

TEST(CPUFloatingRouteDeterminism, ByteIdenticalAcrossThreadCounts)
{
    for (const auto &shape : kServingShapes)
    {
        const int M = shape[0], N = shape[1], K = shape[2];
        SCOPED_TRACE(::testing::Message() << "M=" << M << " N=" << N << " K=" << K);
        const std::vector<float> a = fp32Values(static_cast<std::size_t>(M) * K, 0x5eed);
        const std::vector<float> b = fp32Values(static_cast<std::size_t>(N) * K, 0x5eed);

        std::vector<float> one(static_cast<std::size_t>(M) * N, 0.0f);
        std::vector<float> eight(static_cast<std::size_t>(M) * N, 0.0f);

        {
            ThreadBudget threads(1);
            ASSERT_TRUE(gemm::run_fp32_skinny_matmul(
                a.data(), b.data(), one.data(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr));
        }
        {
            ThreadBudget threads(8);
            ASSERT_TRUE(gemm::run_fp32_skinny_matmul(
                a.data(), b.data(), eight.data(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr));
        }

        expectByteIdentical(eight, one);
        EXPECT_LE(relativeL2(eight, one), 1e-5);
    }
}

// ============================================================================
// Transposed weights and the alpha/beta epilogue at serving scale
// ============================================================================

TEST(CPUFloatingRouteSerialOracle, TransposedWeightsAndEpilogueAtServingScale)
{
    const int M = 16, N = 4096, K = 4096;
    const float alpha = 1.5f, beta = -0.25f;
    const std::vector<float> a = fp32Values(static_cast<std::size_t>(M) * K, 0xabcd);
    // Non-transposed weights: [K,N] row-major.
    const std::vector<float> b = fp32Values(static_cast<std::size_t>(K) * N, 0xbcda);
    const std::vector<float> previous = fp32Values(static_cast<std::size_t>(M) * N, 0xdead);
    std::vector<float> actual = previous;

    ThreadBudget threads(4);
    ASSERT_TRUE(gemm::run_fp32_skinny_matmul(
        a.data(), b.data(), actual.data(), M, N, K, false, alpha, beta, nullptr, nullptr));

    std::vector<float> oracle(static_cast<std::size_t>(M) * N, 0.0f);
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col)
            oracle[static_cast<std::size_t>(row) * N + col] =
                serialOracle(a, b, row, col, N, K, false, alpha, beta,
                             previous[static_cast<std::size_t>(row) * N + col]);

    expectByteIdentical(actual, oracle);
    EXPECT_LE(relativeL2(actual, oracle), 1e-5);
}
