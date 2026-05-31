/**
 * @file Test__CPUQuantizedGemmParity.cpp
 * @brief Comprehensive GEMM/GEMV parity tests for all quantized formats across thread counts.
 *
 * For each of the 20 supported quantized formats, verifies the VNNI GEMM kernel at:
 *   - M=1 (GEMV decode path), M=9 (small prefill, odd M), M=64 (large prefill)
 *   - N=512 (standard, multiple VNNI chunks), N=32 (ldc<64), N=16 (TP-sharded, ldc<64)
 *   - Every OMP thread count from 1 to 56 (both even and odd)
 *
 * Validates:
 *   - Relative L2 error vs FP32 reference (format-specific threshold)
 *   - Cosine similarity vs FP32 reference (format-specific threshold)
 *   - No NaN/Inf values in output
 *   - No all-zero output rows (catches buffer overflow corruption)
 *
 * The small-N shapes (N=16, N=32) specifically regression-test the 2-row GEMM
 * microkernel overflow bug where _mm512_storeu_ps wrote 64 floats per row,
 * corrupting adjacent rows when ldc < 64. The fix forces the safe small-N
 * dispatch path (tmp[64] + memcpy) whenever ldc < 64.
 *
 * @note Run with:  ctest -R V2_Unit_CPUQuantizedGemmParity
 */

#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include "tensors/Tensors.h"
#include "../../utils/TestTensorFactory.h"
#include "v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{

    // =========================================================================
    // Format configuration: name, L2 threshold, cosine threshold
    // =========================================================================

    struct FormatConfig
    {
        const char *name;
        float l2_threshold;
        float cosine_threshold;
    };

    static const std::vector<FormatConfig> FORMAT_TABLE = {
        // Per-block formats (32-element blocks, deferred VNNI packing)
        {"Q8_0", 0.01f, 0.999f},
        {"Q8_1", 0.01f, 0.999f},
        {"Q4_0", 0.01f, 0.999f},
        {"IQ4_NL", 0.01f, 0.999f},
        {"Q4_1", 0.01f, 0.999f},
        {"Q5_0", 0.01f, 0.999f},
        {"Q5_1", 0.01f, 0.999f},

        // Superblock formats (256-element blocks, permanent interleaved)
        {"Q6_K", 0.01f, 0.999f},
        {"Q5_K", 0.01f, 0.999f},
        {"Q4_K", 0.01f, 0.999f},
        {"Q3_K", 0.02f, 0.998f},
        {"Q2_K", 0.05f, 0.990f},

        // IQ formats (importance quantized, superblock path)
        {"IQ4_XS", 0.01f, 0.999f},
        {"IQ3_S", 0.05f, 0.990f},
        {"IQ3_XXS", 0.05f, 0.990f},
        {"IQ2_S", 0.10f, 0.970f},
        {"IQ2_XS", 0.10f, 0.970f},
        {"IQ2_XXS", 0.10f, 0.970f},
        {"IQ1_S", 0.15f, 0.940f},
        {"IQ1_M", 0.15f, 0.940f},
    };

    // =========================================================================
    // Shape configurations
    // =========================================================================

    struct ShapeConfig
    {
        int M, N, K;
        const char *label;
    };

    // K=512 is divisible by both 32 (per-block) and 256 (superblock).
    // N=16,32 exercise the ldc<64 overflow protection path.
    // M=1 (GEMV), M=9 (odd GEMM, GDN bug shape), M=64 (large GEMM).
    static const std::vector<ShapeConfig> SHAPE_TABLE = {
        {1, 512, 512, "GEMV_N512"},
        {1, 32, 512, "GEMV_N32"},
        {9, 512, 512, "GEMM_M9_N512"},
        {9, 32, 512, "GEMM_M9_N32"},
        {9, 16, 512, "GEMM_M9_N16"},
        {64, 512, 512, "GEMM_M64_N512"},
    };

    // =========================================================================
    // Thread count range
    // =========================================================================

    static constexpr int MIN_THREADS = 1;
    static constexpr int MAX_THREADS = 56;

    // =========================================================================
    // Weight factory: creates quantized weights by format name
    // =========================================================================

    std::unique_ptr<TensorBase> createWeights(const std::string &fmt, size_t N, size_t K)
    {
        if (fmt == "Q8_0")
            return TestTensorFactory::createQ8_0Random({N, K});
        if (fmt == "Q8_1")
            return TestTensorFactory::createQ8_1Random({N, K});
        if (fmt == "Q4_0")
            return TestTensorFactory::createQ4_0Random({N, K});
        if (fmt == "IQ4_NL")
            return TestTensorFactory::createIQ4_NLRandom({N, K});
        if (fmt == "Q4_1")
            return TestTensorFactory::createQ4_1Random({N, K});
        if (fmt == "Q5_0")
            return TestTensorFactory::createQ5_0Random({N, K});
        if (fmt == "Q5_1")
            return TestTensorFactory::createQ5_1Random({N, K});
        if (fmt == "Q6_K")
            return TestTensorFactory::createQ6_KRandom({N, K});
        if (fmt == "Q5_K")
            return TestTensorFactory::createQ5_KRandom({N, K});
        if (fmt == "Q4_K")
            return TestTensorFactory::createQ4_KRandom({N, K});
        if (fmt == "Q3_K")
            return TestTensorFactory::createQ3_KRandom({N, K});
        if (fmt == "Q2_K")
            return TestTensorFactory::createQ2_KRandom({N, K});
        if (fmt == "IQ4_XS")
            return TestTensorFactory::createIQ4_XSRandom({N, K});
        if (fmt == "IQ3_S")
            return TestTensorFactory::createIQ3_SRandom({N, K});
        if (fmt == "IQ3_XXS")
            return TestTensorFactory::createIQ3_XXSRandom({N, K});
        if (fmt == "IQ2_S")
            return TestTensorFactory::createIQ2_SRandom({N, K});
        if (fmt == "IQ2_XS")
            return TestTensorFactory::createIQ2_XSRandom({N, K});
        if (fmt == "IQ2_XXS")
            return TestTensorFactory::createIQ2_XXSRandom({N, K});
        if (fmt == "IQ1_S")
            return TestTensorFactory::createIQ1_SRandom({N, K});
        if (fmt == "IQ1_M")
            return TestTensorFactory::createIQ1_MRandom({N, K});
        return nullptr;
    }

    // =========================================================================
    // Metrics
    // =========================================================================

    float computeRelativeL2(const float *a, const float *b, size_t n)
    {
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum_sq_diff += diff * diff;
            sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;
    }

    float computeCosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0, norm_a = 0, norm_b = 0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += (double)a[i] * (double)b[i];
            norm_a += (double)a[i] * (double)a[i];
            norm_b += (double)b[i] * (double)b[i];
        }
        if (norm_a < 1e-15 || norm_b < 1e-15)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    // =========================================================================
    // Parameterized test fixture (one instance per format)
    // =========================================================================

    class CPUQuantizedGemmParity : public ::testing::TestWithParam<std::string>
    {
    protected:
        int saved_threads_ = 0;

        void SetUp() override
        {
            saved_threads_ = omp_get_max_threads();
        }

        void TearDown() override
        {
            omp_set_num_threads(saved_threads_);
        }

        const FormatConfig &getConfig() const
        {
            const std::string &name = GetParam();
            for (const auto &cfg : FORMAT_TABLE)
                if (cfg.name == name)
                    return cfg;
            // Should never happen — INSTANTIATE_TEST_SUITE_P uses FORMAT_TABLE names
            static FormatConfig fallback{"UNKNOWN", 0.01f, 0.999f};
            return fallback;
        }
    };

    // =========================================================================
    // Main test: sweep shapes × thread counts for one format
    // =========================================================================

    TEST_P(CPUQuantizedGemmParity, ThreadSweep)
    {
        const auto &cfg = getConfig();
        const std::string &fmt = GetParam();
        constexpr int K = 512;

        int total_checks = 0;

        // Group shapes by N so we create/pack weight tensors only once per N.
        // The VNNI kernel always operates on packed.N columns, so weights must
        // be created with exactly the N dimension needed for each shape group.
        for (int N : {512, 32, 16})
        {
            // Create and pack weights once per N
            auto weights = createWeights(fmt, N, K);
            ASSERT_NE(weights, nullptr) << fmt << " createWeights [" << N << "x" << K << "]";

            auto fp32_weights = TestTensorFactory::createFP32({(size_t)N, (size_t)K});
            weights->to_fp32(fp32_weights->mutable_data());

            auto quantized_gemm = weights->createGemm();
            ASSERT_NE(quantized_gemm, nullptr) << fmt << " createGemm [" << N << "x" << K << "]";

            gemm::FloatingPointGemmKernel fp32_gemm(fp32_weights.get());

            // Process each shape with this N
            for (const auto &shape : SHAPE_TABLE)
            {
                if (shape.N != N)
                    continue;

                SCOPED_TRACE(shape.label);

                const int M = shape.M;

                // Create input once per shape
                auto input = TestTensorFactory::createFP32Random(
                    {(size_t)M, (size_t)K}, -1.0f, 1.0f, 42);

                // Compute FP32 reference at original thread count (thread-independent)
                omp_set_num_threads(saved_threads_);
                auto ref_output = TestTensorFactory::createFP32Zeros({(size_t)M, (size_t)N});
                bool ref_ok = fp32_gemm.multiply_tensor(
                    input.get(), ref_output.get(), M, N, K);
                ASSERT_TRUE(ref_ok) << "FP32 reference failed for " << fmt << " " << shape.label;

                const float *C_ref = ref_output->data();

                // Pre-allocate quantized output (reused across thread counts)
                auto quant_output = TestTensorFactory::createFP32Zeros({(size_t)M, (size_t)N});

                // Track min/max metrics across all thread counts for summary
                float worst_l2 = 0.0f;
                float worst_cosine = 1.0f;

                // Sweep every thread count from 1 to 56
                for (int t = MIN_THREADS; t <= MAX_THREADS; ++t)
                {
                    omp_set_num_threads(t);
                    ++total_checks;

                    // Zero output buffer for this run
                    std::memset(quant_output->mutable_data(), 0, (size_t)M * N * sizeof(float));

                    bool ok = quantized_gemm->multiply_tensor(
                        input.get(), quant_output.get(), M, N, K);
                    EXPECT_TRUE(ok) << fmt << " " << shape.label << " t=" << t
                                    << " multiply_tensor returned false";
                    if (!ok)
                        continue;

                    const float *C_q = quant_output->data();
                    const size_t total = (size_t)M * N;

                    // --- NaN/Inf check ---
                    bool has_anomaly = false;
                    for (size_t i = 0; i < total; ++i)
                    {
                        if (std::isnan(C_q[i]) || std::isinf(C_q[i]))
                        {
                            EXPECT_TRUE(false)
                                << fmt << " " << shape.label << " t=" << t
                                << " NaN/Inf at [" << i / N << "," << i % N << "]";
                            has_anomaly = true;
                            break;
                        }
                    }
                    if (has_anomaly)
                        continue;

                    // --- All-zero row check (catches buffer overflow corruption) ---
                    if (M > 1)
                    {
                        for (int m = 0; m < M; ++m)
                        {
                            bool all_zero = true;
                            for (int n = 0; n < N; ++n)
                            {
                                if (C_q[m * N + n] != 0.0f)
                                {
                                    all_zero = false;
                                    break;
                                }
                            }
                            EXPECT_FALSE(all_zero)
                                << fmt << " " << shape.label << " t=" << t
                                << " row " << m << " all zeros (buffer overflow?)";
                        }
                    }

                    // --- Relative L2 error ---
                    float l2 = computeRelativeL2(C_q, C_ref, total);
                    EXPECT_LT(l2, cfg.l2_threshold)
                        << fmt << " " << shape.label << " t=" << t
                        << " L2=" << (l2 * 100.0f) << "% exceeds "
                        << (cfg.l2_threshold * 100.0f) << "% threshold";

                    // --- Cosine similarity ---
                    float cos_sim = computeCosineSimilarity(C_q, C_ref, total);
                    EXPECT_GT(cos_sim, cfg.cosine_threshold)
                        << fmt << " " << shape.label << " t=" << t
                        << " cosine=" << cos_sim << " below "
                        << cfg.cosine_threshold << " threshold";

                    worst_l2 = std::max(worst_l2, l2);
                    worst_cosine = std::min(worst_cosine, cos_sim);
                }

                // Per-shape summary line
                std::cout << "  " << std::left << std::setw(18) << shape.label
                          << " L2_max=" << std::fixed << std::setprecision(4)
                          << (worst_l2 * 100.0f) << "%"
                          << "  cos_min=" << std::setprecision(6) << worst_cosine
                          << "  (threads 1-" << MAX_THREADS << ")" << std::endl;
            }
        }

        std::cout << "[" << fmt << "] " << total_checks << " checks PASSED"
                  << " (L2<" << (cfg.l2_threshold * 100.0f) << "%"
                  << " cosine>" << cfg.cosine_threshold << ")" << std::endl;
    }

    // =========================================================================
    // Patterned input tests: corruption sentinels and sign patterns
    // =========================================================================

    // Input patterns to test. Each creates an FP32 activation tensor with a
    // specific fill strategy designed to expose different classes of bugs.
    struct InputPattern
    {
        const char *name;
        // Fill the activation tensor (M × K) according to this pattern
        std::function<void(FP32Tensor *, size_t M, size_t K)> fill;
        // If true, odd-numbered rows of A are zero and C's odd rows must be
        // exactly zero (cross-row contamination sentinel)
        bool has_zero_sentinel_rows;
        // Minimum M required for this pattern
        int min_M;
    };

    static const std::vector<InputPattern> PATTERN_TABLE = {
        // --- Patterns valid for any M (including GEMV M=1) ---
        {"all_ones",
         [](FP32Tensor *t, size_t, size_t)
         { TestTensorFactory::fillValue(t, 1.0f); },
         false, 1},
        {"all_neg_ones",
         [](FP32Tensor *t, size_t, size_t)
         { TestTensorFactory::fillValue(t, -1.0f); },
         false, 1},
        {"checkerboard",
         [](FP32Tensor *t, size_t, size_t)
         { TestTensorFactory::fillCheckerboard(t, 0.5f); },
         false, 1},

        // --- Patterns requiring M≥2 (row-alternating with zero sentinels) ---
        {"alternating_1_0",
         [](FP32Tensor *t, size_t, size_t)
         { TestTensorFactory::fillAlternatingRows(t, 1.0f, 0.0f); },
         true, 2},
        {"alternating_neg1_0",
         [](FP32Tensor *t, size_t, size_t)
         { TestTensorFactory::fillAlternatingRows(t, -1.0f, 0.0f); },
         true, 2},
        {"alternating_0_1",
         [](FP32Tensor *t, size_t, size_t)
         { TestTensorFactory::fillAlternatingRows(t, 0.0f, 1.0f); },
         false, 2}, // even rows are the zero sentinels here
    };

    TEST_P(CPUQuantizedGemmParity, PatternedInputs)
    {
        const auto &cfg = getConfig();
        const std::string &fmt = GetParam();
        constexpr int K = 512;

        int total_checks = 0;

        for (int N : {512, 32, 16})
        {
            auto weights = createWeights(fmt, N, K);
            ASSERT_NE(weights, nullptr);

            auto fp32_weights = TestTensorFactory::createFP32({(size_t)N, (size_t)K});
            weights->to_fp32(fp32_weights->mutable_data());

            auto quantized_gemm = weights->createGemm();
            ASSERT_NE(quantized_gemm, nullptr);

            gemm::FloatingPointGemmKernel fp32_gemm(fp32_weights.get());

            for (const auto &shape : SHAPE_TABLE)
            {
                if (shape.N != N)
                    continue;

                for (const auto &pattern : PATTERN_TABLE)
                {
                    if (shape.M < pattern.min_M)
                        continue;
                    SCOPED_TRACE(std::string(shape.label) + "/" + pattern.name);
                    const int M = shape.M;

                    auto input = TestTensorFactory::createFP32({(size_t)M, (size_t)K});
                    pattern.fill(input.get(), M, K);

                    // FP32 reference
                    omp_set_num_threads(saved_threads_);
                    auto ref_output = TestTensorFactory::createFP32Zeros({(size_t)M, (size_t)N});
                    ASSERT_TRUE(fp32_gemm.multiply_tensor(input.get(), ref_output.get(), M, N, K));
                    const float *C_ref = ref_output->data();

                    auto quant_output = TestTensorFactory::createFP32Zeros({(size_t)M, (size_t)N});

                    float worst_l2 = 0.0f;
                    float worst_cosine = 1.0f;
                    int sentinel_violations = 0;

                    for (int t = MIN_THREADS; t <= MAX_THREADS; ++t)
                    {
                        omp_set_num_threads(t);
                        ++total_checks;

                        std::memset(quant_output->mutable_data(), 0, (size_t)M * N * sizeof(float));

                        bool ok = quantized_gemm->multiply_tensor(
                            input.get(), quant_output.get(), M, N, K);
                        EXPECT_TRUE(ok) << fmt << " " << pattern.name
                                        << " " << shape.label << " t=" << t;
                        if (!ok)
                            continue;

                        const float *C_q = quant_output->data();
                        const size_t total = (size_t)M * N;

                        // NaN/Inf check
                        for (size_t i = 0; i < total; ++i)
                        {
                            if (std::isnan(C_q[i]) || std::isinf(C_q[i]))
                            {
                                EXPECT_TRUE(false)
                                    << fmt << " " << pattern.name << " " << shape.label
                                    << " t=" << t << " NaN/Inf at [" << i / N << "," << i % N << "]";
                                break;
                            }
                        }

                        // Zero-row sentinel check: zero input rows must yield zero output rows.
                        // This is an EXACT check — any non-zero value is cross-row contamination.
                        for (int m = 0; m < M; ++m)
                        {
                            bool input_row_is_zero;
                            if (pattern.has_zero_sentinel_rows)
                                input_row_is_zero = (m % 2 == 1); // odd rows are zero
                            else if (std::string(pattern.name) == "alternating_0_1")
                                input_row_is_zero = (m % 2 == 0); // even rows are zero
                            else
                                continue; // no sentinel rows for this pattern

                            if (!input_row_is_zero)
                                continue;

                            for (int n = 0; n < N; ++n)
                            {
                                if (C_q[m * N + n] != 0.0f)
                                {
                                    ++sentinel_violations;
                                    EXPECT_EQ(C_q[m * N + n], 0.0f)
                                        << fmt << " " << pattern.name << " " << shape.label
                                        << " t=" << t << " sentinel row " << m
                                        << " col " << n << " is non-zero (CORRUPTION)";
                                    break; // one failure per row is enough
                                }
                            }
                        }

                        // L2 + cosine on non-zero rows
                        // Patterned (constant) inputs have inherently higher L2 than random
                        // because quantization errors are correlated, not statistically canceling.
                        // Use 2x relaxed L2 threshold; the primary value of this test is the
                        // zero-row sentinel check, not tight numerical bounds.
                        float l2 = computeRelativeL2(C_q, C_ref, total);
                        float cos_sim = computeCosineSimilarity(C_q, C_ref, total);

                        EXPECT_LT(l2, cfg.l2_threshold * 2.0f)
                            << fmt << " " << pattern.name << " " << shape.label << " t=" << t;
                        EXPECT_GT(cos_sim, cfg.cosine_threshold)
                            << fmt << " " << pattern.name << " " << shape.label << " t=" << t;

                        worst_l2 = std::max(worst_l2, l2);
                        worst_cosine = std::min(worst_cosine, cos_sim);
                    }

                    std::cout << "  " << std::left << std::setw(12) << shape.label
                              << "/" << std::setw(20) << pattern.name
                              << " L2_max=" << std::fixed << std::setprecision(4)
                              << (worst_l2 * 100.0f) << "%"
                              << "  cos_min=" << std::setprecision(6) << worst_cosine;
                    if (sentinel_violations > 0)
                        std::cout << "  SENTINEL_VIOLATIONS=" << sentinel_violations;
                    std::cout << std::endl;
                }
            }
        }

        std::cout << "[" << fmt << "/patterned] " << total_checks << " checks PASSED" << std::endl;
    }

    TEST(CPUQuantizedGemmParityRegression, Q5KQwen36NodeLocalTPGDNPrefillShape)
    {
        const int M = 9;
        const int N = 7168;
        const int K = 5120;

        auto weights = TestTensorFactory::createQ5_KRandom(
            {static_cast<size_t>(N), static_cast<size_t>(K)});
        ASSERT_NE(weights, nullptr);

        auto gemm = weights->createGemm();
        ASSERT_NE(gemm, nullptr);

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)}, -1.0f, 1.0f, 42);
        auto output = TestTensorFactory::createFP32Zeros(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        const int saved_threads = omp_get_max_threads();
        for (int threads : {1, std::max(1, saved_threads)})
        {
            SCOPED_TRACE(::testing::Message() << "threads=" << threads);
            omp_set_num_threads(threads);
            std::memset(output->mutable_data(), 0,
                        static_cast<size_t>(M) * N * sizeof(float));

            ASSERT_TRUE(gemm->multiply_tensor(input.get(), output.get(), M, N, K));

            const float *out = output->data();
            for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i)
            {
                ASSERT_TRUE(std::isfinite(out[i]))
                    << "non-finite output at flat index " << i;
            }

            for (int row = 0; row < M; ++row)
            {
                bool all_zero = true;
                for (int col = 0; col < N; ++col)
                {
                    if (out[static_cast<size_t>(row) * N + col] != 0.0f)
                    {
                        all_zero = false;
                        break;
                    }
                }
                EXPECT_FALSE(all_zero) << "row " << row << " is all zero";
            }
        }
        omp_set_num_threads(saved_threads);
    }

    // =========================================================================
    // Instantiate for all 20 quantized formats
    // =========================================================================

    INSTANTIATE_TEST_SUITE_P(
        AllFormats,
        CPUQuantizedGemmParity,
        ::testing::Values(
            "Q8_0", "Q8_1", "Q4_0", "IQ4_NL", "Q4_1", "Q5_0", "Q5_1",
            "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K",
            "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS",
            "IQ1_S", "IQ1_M"),
        [](const ::testing::TestParamInfo<std::string> &info)
        { return info.param; });

} // anonymous namespace
