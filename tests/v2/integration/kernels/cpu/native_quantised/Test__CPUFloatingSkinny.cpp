/**
 * @file Test__CPUFloatingSkinny.cpp
 * @brief Certify narrow floating verifier projections against serial decode.
 *
 * The suite crosses the historical N=128 dispatch boundary, odd row tails,
 * odd thread counts, contiguous/transposed weights and alpha/beta epilogues.
 * All floating weight formats keep FP32 activations. Outputs are compared as
 * bytes, including signed zero: a numerically harmless extra +0 is not an
 * equivalent MTP publication program. No model or accelerator is required.
 */

#include <gtest/gtest.h>
#include <array>
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
        /** Select a positive physical workshare width for this case. */
        explicit ThreadBudget(int threads) : previous_(omp_get_max_threads())
        { omp_set_num_threads(threads); }
        /** Return ownership of the original thread budget to the caller. */
        ~ThreadBudget() { omp_set_num_threads(previous_); }
        ThreadBudget(const ThreadBudget &) = delete;
        ThreadBudget &operator=(const ThreadBudget &) = delete;
    };

    /** One supported floating weight storage type; activations remain FP32. */
    class CPUFloatingSkinny : public ::testing::TestWithParam<TensorType> {};

    /** A scalar rounding oracle detects epilogue defects shared by both row paths. */
    TEST(CPUSkinnyMatmulEpilogue, PreservesSerialRoundingAndDisabledReads)
    {
        std::mt19937 random(0xe91109);
        std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
        for (int sample = 0; sample < 10000; ++sample)
        for (bool with_bias : {false, true})
        for (bool with_previous : {false, true})
        {
            const float dot = sample % 11 == 0 ? -0.0f : distribution(random);
            const float alpha = sample % 7 == 0 ? -1.0f : distribution(random);
            const float beta = with_previous ? distribution(random) : 0.0f;
            const float previous = distribution(random) * 8192.0f;
            const float bias = sample % 13 == 0 ? 0.0f : distribution(random);
            // Volatile belongs only in the independent oracle. Production's
            // rounded product is register-resident, without a store/reload.
            volatile float product = alpha * dot;
            const float biased = with_bias ? product + bias : product;
            const float expected = with_previous ? std::fma(beta, previous, biased) : biased;
            const float actual = gemm::completeSkinnyMatmulValue(dot, alpha, beta,
                with_previous ? &previous : nullptr, with_bias ? &bias : nullptr);
            ASSERT_EQ(std::memcmp(&actual, &expected, sizeof(float)), 0)
                << "sample=" << sample << " bias=" << with_bias << " beta=" << beta;
        }
    }

    /** Exercise scheduling boundaries without copying a second dispatch formula. */
    TEST_P(CPUFloatingSkinny, NarrowAndWideRowsAreSerialByteExact)
    {
        std::mt19937 random(0x534b494e);
        std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
        for (int threads : {1, 3, 8})
        {
            ThreadBudget budget(threads);
            for (int m : {2, 3, 4, 7, 15, 16, 31})
            for (int n : {1, 7, 48, 127, 128, 129})
            for (int k : {0, 17, 128})
            for (bool transposed : {false, true})
            for (bool with_bias : {false, true})
            {
                // Only the FP32 primitive owns a fused bias epilogue; mixed
                // FP16/BF16 storage entry points expose alpha and beta only.
                if (with_bias && GetParam() != TensorType::FP32)
                    continue;
                SCOPED_TRACE(::testing::Message() << "threads=" << threads
                    << " M=" << m << " N=" << n << " K=" << k << " transposed=" << transposed
                    << " bias=" << with_bias);
                // Non-null storage also admits K=0, whose epilogue must retain
                // the sign of alpha*0 when no bias or beta contribution exists.
                std::vector<float> a(std::max(1, m * k)), b(std::max(1, n * k));
                for (float &v : a) v = distribution(random);
                for (float &v : b) v = distribution(random);
                std::vector<float> bias(n);
                for (float &v : bias) v = distribution(random);
                const float *bias_values = with_bias ? bias.data() : nullptr;
                std::vector<uint16_t> stored(b.size());
                for (size_t i = 0; i < b.size(); ++i)
                    stored[i] = GetParam() == TensorType::BF16
                        ? simd::fp32_to_bf16(b[i]) : fp32_to_fp16(b[i]);
                for (const auto [alpha, beta] :
                     std::array<std::pair<float, float>, 2>{{{-1.0f, 0.0f}, {0.75f, -0.25f}}})
                {
                    SCOPED_TRACE(::testing::Message() << "alpha=" << alpha << " beta=" << beta);
                    constexpr float guard = 9182.5f;
                    std::vector<float> serial(static_cast<size_t>(m) * n + 2, guard);
                    auto grouped = serial;
                    const auto run = [&](const float *input, float *output, int rows)
                    {
                        switch (GetParam())
                        {
                        case TensorType::FP32:
                            return gemm::run_fp32_skinny_matmul(input, b.data(), output,
                                rows, n, k, transposed, alpha, beta, bias_values);
                        case TensorType::FP16:
                            return gemm::run_fp32xfp16_skinny_matmul(input, stored.data(), output,
                                rows, n, k, transposed, alpha, beta);
                        case TensorType::BF16:
                            return gemm::run_fp32xbf16_skinny_matmul(input, stored.data(), output,
                                rows, n, k, transposed, alpha, beta);
                        default:
                            return false;
                        }
                    };
                    for (int row = 0; row < m; ++row)
                        ASSERT_TRUE(run(a.data() + row * k, serial.data() + 1 + row * n, 1));
                    ASSERT_TRUE(run(a.data(), grouped.data() + 1, m));
                    for (size_t i = 0; i < serial.size(); ++i)
                    {
                        if (i > 0 && i + 1 < serial.size() && GetParam() == TensorType::FP32 &&
                            std::memcmp(&serial[i], &grouped[i], sizeof(float)) != 0)
                        {
                            std::vector<float> serial_dot(n), grouped_dot(static_cast<size_t>(m) * n);
                            const size_t row = (i - 1) / n, col = (i - 1) % n;
                            ASSERT_TRUE(gemm::run_fp32_skinny_matmul(a.data() + row * k, b.data(),
                                serial_dot.data(), 1, n, k, transposed));
                            ASSERT_TRUE(gemm::run_fp32_skinny_matmul(a.data(), b.data(),
                                grouped_dot.data(), m, n, k, transposed));
                            volatile float product = alpha * serial_dot[col];
                            const float separate_bias = product + (with_bias ? bias[col] : 0.0f);
                            const float fused_bias = std::fma(alpha, serial_dot[col], with_bias ? bias[col] : 0.0f);
                            ADD_FAILURE() << "unscaled serial=" << serial_dot[col]
                                << " grouped=" << grouped_dot[i - 1]
                                << " separate-bias=" << std::fma(beta, guard, separate_bias)
                                << " fused-bias=" << std::fma(beta, guard, fused_bias);
                        }
                        ASSERT_EQ(std::memcmp(&serial[i], &grouped[i], sizeof(float)), 0)
                            << "output=" << i << " serial=" << serial[i]
                            << " grouped=" << grouped[i];
                    }
                    EXPECT_EQ(grouped.front(), guard);
                    EXPECT_EQ(grouped.back(), guard);
                }
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(AllFloatingWeights, CPUFloatingSkinny,
        ::testing::Values(TensorType::FP32, TensorType::FP16, TensorType::BF16),
        [](const ::testing::TestParamInfo<TensorType> &info)
        { return info.param == TensorType::FP32 ? "FP32"
            : info.param == TensorType::FP16 ? "FP16" : "BF16"; });
}
