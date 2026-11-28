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
 * The same shapes are then run against the oneDNN FP32 reference, the way
 * the AMX-BF16 route was compared on its host: an interleaved same-process
 * min-of-20 measurement with the GF/s / t=oneDNN/native table, and a
 * numerical agreement gate (the two implementations use different reduction
 * orders, so agreement is bounded, not byte-identical — measured here at
 * relL2 1.1-1.3e-6 with a 1e-4 contract for other envelopes).
 *
 * No model or accelerator is required; the test is device-free and exercises
 * only the native CPU route against independently computed reference words.
 */

#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

#include <oneapi/dnnl/dnnl.hpp>

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

    /** 64-byte-aligned buffer, as oneDNN expects for direct user storage. */
    class AlignedBuffer final
    {
        void *ptr_;
    public:
        explicit AlignedBuffer(std::size_t elements)
        {
            const std::size_t bytes = elements * sizeof(float);
            if (bytes == 0)
            {
                ptr_ = nullptr;
                return;
            }
            const std::size_t aligned = (bytes + 63) & ~std::size_t(63);
            ptr_ = std::aligned_alloc(64, aligned);
            if (!ptr_)
                throw std::runtime_error("AlignedBuffer allocation failed");
        }
        ~AlignedBuffer() { std::free(ptr_); }
        AlignedBuffer(const AlignedBuffer &) = delete;
        AlignedBuffer &operator=(const AlignedBuffer &) = delete;
        float *data() { return static_cast<float *>(ptr_); }
        const float *data() const { return static_cast<const float *>(ptr_); }
    };

    /** oneDNN FP32 matmul with the same row-major src / transposed weights. */
    class OneDNNFP32Matmul final
    {
        dnnl::engine eng_{dnnl::engine::kind::cpu, 0};
        dnnl::stream str_{eng_};
        dnnl::matmul prim_;
        dnnl::memory src_, wei_, dst_;
    public:
        OneDNNFP32Matmul(const float *A, const float *B, float *C, int M, int N, int K)
        {
            dnnl::memory::desc s({M, K}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
            dnnl::memory::desc w({K, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ba);
            dnnl::memory::desc d({M, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
            dnnl::matmul::primitive_desc pd(eng_, s, w, d);
            prim_ = dnnl::matmul(pd);
            src_ = dnnl::memory(s, eng_, const_cast<float *>(A));
            wei_ = dnnl::memory(w, eng_, const_cast<float *>(B));
            dst_ = dnnl::memory(d, eng_, C);
        }
        void run()
        {
            prim_.execute(str_, {{DNNL_ARG_SRC, src_}, {DNNL_ARG_WEIGHTS, wei_}, {DNNL_ARG_DST, dst_}});
            str_.wait();
        }
    };

    using Clock = std::chrono::steady_clock;
    double elapsedMs(Clock::time_point a, Clock::time_point b)
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    /** Max absolute deviation between two same-shape FP32 outputs. */
    double maxAbsoluteDiff(const std::vector<float> &a, const std::vector<float> &b)
    {
        double maximum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            maximum = std::max(maximum, std::fabs(double(a[i]) - double(b[i])));
        return maximum;
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

// ============================================================================
// Native-vs-oneDNN comparison at the canonical serving shapes
// ============================================================================

TEST(CPUFloatingRouteOneDNN, ServingShapesAgreeWithOneDNNReference)
{
    for (const auto &shape : kServingShapes)
    {
        const int M = shape[0], N = shape[1], K = shape[2];
        SCOPED_TRACE(::testing::Message() << "M=" << M << " N=" << N << " K=" << K);
        AlignedBuffer a(static_cast<std::size_t>(M) * K), b(static_cast<std::size_t>(N) * K);
        AlignedBuffer cdnn(static_cast<std::size_t>(M) * N);
        std::vector<float> cnat(static_cast<std::size_t>(M) * N, 0.0f);
        const std::vector<float> aValues = fp32Values(static_cast<std::size_t>(M) * K, 0xd011);
        const std::vector<float> bValues = fp32Values(static_cast<std::size_t>(N) * K, 0xd022);
        std::memcpy(a.data(), aValues.data(), aValues.size() * sizeof(float));
        std::memcpy(b.data(), bValues.data(), bValues.size() * sizeof(float));

        ThreadBudget threads(8);
        ASSERT_TRUE(gemm::run_fp32_skinny_matmul(
            a.data(), b.data(), cnat.data(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr));

        OneDNNFP32Matmul reference(a.data(), b.data(), cdnn.data(), M, N, K);
        reference.run();

        const std::vector<float> actual(cnat.begin(), cnat.end());
        const std::vector<float> expected(cdnn.data(), cdnn.data() + cnat.size());
        // Different reduction orders: agreement is bounded, not byte-identical.
        // Measured on the AVX2 host at relL2 1.06-1.34e-6; 1e-4 covers other
        // envelopes and library build geometry.
        EXPECT_LE(relativeL2(actual, expected), 1e-4)
            << "native FP32 and the oneDNN FP32 reference disagree beyond the gate";
        std::printf("    %dx%dx%d: relL2=%.3e maxAbs=%.3e\n",
                    M, N, K, relativeL2(actual, expected), maxAbsoluteDiff(actual, expected));
    }
}

TEST(CPUFloatingRouteOneDNN, ServingShapesInterleavedMin20ComparisonTable)
{
    // Same-process interleaved min-of-20 comparison, exactly the methodology
    // used for the AMX-BF16 route on its host. Timings are informational:
    // absolute GF/s drift under host load, the interleaved ratios are the
    // meaningful numbers. No performance gate is asserted (no flaky CI).
    printf("    native FP32 vs oneDNN FP32 (interleaved, min of 20, 8 threads)\n");
    printf("    %-15s | %10s %10s %9s\n", "shape", "FP32nat", "oneDNN32", "t=dnn/nat");
    ThreadBudget threads(8);
    for (const auto &shape : kServingShapes)
    {
        const int M = shape[0], N = shape[1], K = shape[2];
        SCOPED_TRACE(::testing::Message() << "M=" << M << " N=" << N << " K=" << K);
        AlignedBuffer a(static_cast<std::size_t>(M) * K), b(static_cast<std::size_t>(N) * K);
        AlignedBuffer c(static_cast<std::size_t>(M) * N);
        const std::vector<float> aValues = fp32Values(static_cast<std::size_t>(M) * K, 0x715e);
        const std::vector<float> bValues = fp32Values(static_cast<std::size_t>(N) * K, 0x715e);
        std::memcpy(a.data(), aValues.data(), aValues.size() * sizeof(float));
        std::memcpy(b.data(), bValues.data(), bValues.size() * sizeof(float));

        OneDNNFP32Matmul reference(a.data(), b.data(), c.data(), M, N, K);

        double bestNative = 1e9, bestOneDNN = 1e9;
        for (int warm = 0; warm < 3; ++warm)
        {
            ASSERT_TRUE(gemm::run_fp32_skinny_matmul(
                a.data(), b.data(), c.data(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr));
            reference.run();
        }
        for (int round = 0; round < 20; ++round)
        {
            auto t = Clock::now();
            ASSERT_TRUE(gemm::run_fp32_skinny_matmul(
                a.data(), b.data(), c.data(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr));
            bestNative = std::min(bestNative, elapsedMs(t, Clock::now()));
            t = Clock::now();
            reference.run();
            bestOneDNN = std::min(bestOneDNN, elapsedMs(t, Clock::now()));
        }
        const auto gfs = [&](double millis) { return 2.0 * M * N * K / 1e9 / millis * 1e3; };
        printf("    %-15s | %10.0f %10.0f %8.2fx\n",
               (std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K)).c_str(),
               gfs(bestNative), gfs(bestOneDNN), bestOneDNN / bestNative);
        EXPECT_GT(bestNative, 0.0);
        EXPECT_GT(bestOneDNN, 0.0);
    }
    printf("    t=dnn/nat = oneDNN_time / native_time; < 1.0 = oneDNN FASTER, > 1.0 = native faster.\n");
}
