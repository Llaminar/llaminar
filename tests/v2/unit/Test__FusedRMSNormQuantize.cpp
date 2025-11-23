/**
 * @file Test__FusedRMSNormQuantize.cpp
 * @brief Unit tests for FusedRMSNormQuantize kernel (RMSNorm + INT8 quantization)
 * @author David Sanftenberg
 *
 * Tests the fused RMSNorm+INT8Quantize kernel against separate operations
 * to validate correctness and measure performance improvement.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#include "kernels/cpu/fused/FusedRMSNormQuantize.h"
#include "utils/MPIContext.h"

namespace llaminar2
{
    namespace
    {
        // Helper: Generate random FP32 data in range [-bound, bound]
        void fill_random(float *data, size_t count, float bound = 1.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        // Helper: Compute reference RMSNorm (unfused)
        void compute_reference_rmsnorm(
            const float *input, const float *gamma, float *output,
            int seq_len, int d_model, float epsilon)
        {
            for (int row = 0; row < seq_len; ++row)
            {
                const float *in_row = input + row * d_model;
                float *out_row = output + row * d_model;

                // Compute RMS
                double sum_sq = 0.0;
                for (int i = 0; i < d_model; ++i)
                {
                    sum_sq += static_cast<double>(in_row[i]) * static_cast<double>(in_row[i]);
                }
                float rms = std::sqrt(static_cast<float>(sum_sq / d_model) + epsilon);

                // Normalize and apply gamma
                for (int i = 0; i < d_model; ++i)
                {
                    out_row[i] = (in_row[i] / rms) * gamma[i];
                }
            }
        }

        // Helper: Quantize FP32 to INT8 with per-row symmetric quantization
        void quantize_symmetric_per_row(
            const float *input, int8_t *output, float *scales,
            int seq_len, int d_model)
        {
            for (int row = 0; row < seq_len; ++row)
            {
                const float *in_row = input + row * d_model;
                int8_t *out_row = output + row * d_model;

                // Find max absolute value
                float max_abs = 0.0f;
                for (int i = 0; i < d_model; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(in_row[i]));
                }

                // Compute scale (map max_abs to 127)
                float scale = (max_abs > 1e-9f) ? (max_abs / 127.0f) : 1.0f;
                scales[row] = scale;

                // Quantize
                float inv_scale = 1.0f / scale;
                for (int i = 0; i < d_model; ++i)
                {
                    float scaled = in_row[i] * inv_scale;
                    int32_t quant = static_cast<int32_t>(std::round(scaled));
                    quant = std::max(-127, std::min(127, quant));
                    out_row[i] = static_cast<int8_t>(quant);
                }
            }
        }

        // Helper: Dequantize INT8 to FP32 for comparison
        void dequantize_symmetric_per_row(
            const int8_t *input, const float *scales, float *output,
            int seq_len, int d_model)
        {
            for (int row = 0; row < seq_len; ++row)
            {
                const int8_t *in_row = input + row * d_model;
                float *out_row = output + row * d_model;
                float scale = scales[row];

                for (int i = 0; i < d_model; ++i)
                {
                    out_row[i] = static_cast<float>(in_row[i]) * scale;
                }
            }
        }

        // Helper: Compute relative L2 error between two FP32 buffers
        float compute_relative_l2_error(const float *a, const float *b, size_t count)
        {
            double sum_sq_diff = 0.0;
            double sum_sq_ref = 0.0;

            for (size_t i = 0; i < count; ++i)
            {
                double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                sum_sq_diff += diff * diff;
                sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }

            return std::sqrt(sum_sq_diff / (sum_sq_ref + 1e-12));
        }
    }

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__FusedRMSNormQuantize : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Initialize MPI context for kernel creation
            mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
        }

        std::shared_ptr<MPIContext> mpi_ctx;
    };

    // =============================================================================
    // Correctness Tests
    // =============================================================================

    /**
     * @test Single token (seq_len=1) with small d_model (896)
     * Validates fused kernel output matches separate RMSNorm + quantize
     */
    TEST_F(Test__FusedRMSNormQuantize, SingleToken_SmallModel)
    {
        const int seq_len = 1;
        const int d_model = 896; // Qwen 0.5B
        const float epsilon = 1e-6f;

        // Allocate buffers
        std::vector<float> input(seq_len * d_model);
        std::vector<float> gamma(d_model, 1.0f); // Gamma weights (all 1s for simplicity)
        std::vector<int8_t> output_fused(seq_len * d_model);
        std::vector<float> scales_fused(seq_len);

        // Reference path buffers
        std::vector<float> rmsnorm_output(seq_len * d_model);
        std::vector<int8_t> output_ref(seq_len * d_model);
        std::vector<float> scales_ref(seq_len);

        // Initialize input with random data
        fill_random(input.data(), input.size(), 0.5f);

        // Execute fused kernel
        FusedRMSNormQuantize fused_kernel;
        bool success = fused_kernel.execute(
            input.data(), gamma.data(),
            output_fused.data(), scales_fused.data(),
            seq_len, d_model, epsilon);
        ASSERT_TRUE(success) << "Fused kernel execution failed";

        // Reference path: RMSNorm + Quantize separately
        compute_reference_rmsnorm(input.data(), gamma.data(), rmsnorm_output.data(),
                                  seq_len, d_model, epsilon);
        quantize_symmetric_per_row(rmsnorm_output.data(), output_ref.data(), scales_ref.data(),
                                   seq_len, d_model);

        // Dequantize both for FP32 comparison
        std::vector<float> dequant_fused(seq_len * d_model);
        std::vector<float> dequant_ref(seq_len * d_model);
        dequantize_symmetric_per_row(output_fused.data(), scales_fused.data(), dequant_fused.data(),
                                     seq_len, d_model);
        dequantize_symmetric_per_row(output_ref.data(), scales_ref.data(), dequant_ref.data(),
                                     seq_len, d_model);

        // Validate: dequantized outputs should match within quantization error
        float rel_l2_error = compute_relative_l2_error(dequant_fused.data(), dequant_ref.data(),
                                                       dequant_fused.size());
        EXPECT_LT(rel_l2_error, 0.005f) << "Dequantized outputs deviate by > 0.5%";

        // Validate scales (should be very close, but INT8 rounding may cause small differences)
        float scale_diff = std::abs(scales_fused[0] - scales_ref[0]);
        EXPECT_LT(scale_diff, scales_ref[0] * 0.01f) << "Scales differ by > 1%";
    }

    /**
     * @test Small batch (seq_len=8) with medium d_model (4864)
     * Tests parallelization correctness (OpenMP)
     */
    TEST_F(Test__FusedRMSNormQuantize, SmallBatch_LargeModel)
    {
        const int seq_len = 8;
        const int d_model = 4864; // Qwen 7B
        const float epsilon = 1e-6f;

        std::vector<float> input(seq_len * d_model);
        std::vector<float> gamma(d_model);
        std::vector<int8_t> output_fused(seq_len * d_model);
        std::vector<float> scales_fused(seq_len);

        std::vector<float> rmsnorm_output(seq_len * d_model);
        std::vector<int8_t> output_ref(seq_len * d_model);
        std::vector<float> scales_ref(seq_len);

        fill_random(input.data(), input.size(), 1.0f, 123);
        fill_random(gamma.data(), gamma.size(), 0.8f, 456);

        // Execute fused kernel
        FusedRMSNormQuantize fused_kernel;
        bool success = fused_kernel.execute(
            input.data(), gamma.data(),
            output_fused.data(), scales_fused.data(),
            seq_len, d_model, epsilon);
        ASSERT_TRUE(success);

        // Reference path
        compute_reference_rmsnorm(input.data(), gamma.data(), rmsnorm_output.data(),
                                  seq_len, d_model, epsilon);
        quantize_symmetric_per_row(rmsnorm_output.data(), output_ref.data(), scales_ref.data(),
                                   seq_len, d_model);

        // Dequantize and compare
        std::vector<float> dequant_fused(seq_len * d_model);
        std::vector<float> dequant_ref(seq_len * d_model);
        dequantize_symmetric_per_row(output_fused.data(), scales_fused.data(), dequant_fused.data(),
                                     seq_len, d_model);
        dequantize_symmetric_per_row(output_ref.data(), scales_ref.data(), dequant_ref.data(),
                                     seq_len, d_model);

        float rel_l2_error = compute_relative_l2_error(dequant_fused.data(), dequant_ref.data(),
                                                       dequant_fused.size());
        EXPECT_LT(rel_l2_error, 0.005f) << "Batch parity deviation > 0.5%";
    }

    /**
     * @test Large batch (seq_len=64) with various d_model sizes
     * Tests scalability and correctness across different workloads
     */
    TEST_F(Test__FusedRMSNormQuantize, LargeBatch_VariousModels)
    {
        const int seq_len = 64;
        const float epsilon = 1e-6f;

        // Test multiple model sizes
        std::vector<int> d_model_sizes = {896, 2048, 4864, 8192};

        for (int d_model : d_model_sizes)
        {
            SCOPED_TRACE("d_model=" + std::to_string(d_model));

            std::vector<float> input(seq_len * d_model);
            std::vector<float> gamma(d_model);
            std::vector<int8_t> output_fused(seq_len * d_model);
            std::vector<float> scales_fused(seq_len);

            std::vector<float> rmsnorm_output(seq_len * d_model);
            std::vector<int8_t> output_ref(seq_len * d_model);
            std::vector<float> scales_ref(seq_len);

            fill_random(input.data(), input.size(), 2.0f, 789 + d_model);
            fill_random(gamma.data(), gamma.size(), 1.2f, 101112 + d_model);

            FusedRMSNormQuantize fused_kernel;
            bool success = fused_kernel.execute(
                input.data(), gamma.data(),
                output_fused.data(), scales_fused.data(),
                seq_len, d_model, epsilon);
            ASSERT_TRUE(success);

            compute_reference_rmsnorm(input.data(), gamma.data(), rmsnorm_output.data(),
                                      seq_len, d_model, epsilon);
            quantize_symmetric_per_row(rmsnorm_output.data(), output_ref.data(), scales_ref.data(),
                                       seq_len, d_model);

            std::vector<float> dequant_fused(seq_len * d_model);
            std::vector<float> dequant_ref(seq_len * d_model);
            dequantize_symmetric_per_row(output_fused.data(), scales_fused.data(), dequant_fused.data(),
                                         seq_len, d_model);
            dequantize_symmetric_per_row(output_ref.data(), scales_ref.data(), dequant_ref.data(),
                                         seq_len, d_model);

            float rel_l2_error = compute_relative_l2_error(dequant_fused.data(), dequant_ref.data(),
                                                           dequant_fused.size());
            EXPECT_LT(rel_l2_error, 0.005f) << "Error for d_model=" << d_model;
        }
    }

    // =============================================================================
    // SIMD Variant Tests (when compiled with AVX512/AVX2)
    // =============================================================================

#if defined(__AVX512F__) || defined(__AVX2__)
    /**
     * @test Verify SIMD variants produce identical results to scalar
     * This test only runs when SIMD is available (march=native with AVX2/AVX512)
     */
    TEST_F(Test__FusedRMSNormQuantize, SIMD_Parity)
    {
        const int seq_len = 16;
        const int d_model = 2048;
        const float epsilon = 1e-6f;

        std::vector<float> input(seq_len * d_model);
        std::vector<float> gamma(d_model);
        fill_random(input.data(), input.size(), 1.5f, 999);
        fill_random(gamma.data(), gamma.size(), 0.9f, 888);

        std::vector<int8_t> output_simd(seq_len * d_model);
        std::vector<float> scales_simd(seq_len);

        FusedRMSNormQuantize kernel;
        bool success = kernel.execute(
            input.data(), gamma.data(),
            output_simd.data(), scales_simd.data(),
            seq_len, d_model, epsilon);
        ASSERT_TRUE(success);

        // Dequantize SIMD result
        std::vector<float> dequant_simd(seq_len * d_model);
        dequantize_symmetric_per_row(output_simd.data(), scales_simd.data(), dequant_simd.data(),
                                     seq_len, d_model);

        // Reference (scalar path)
        std::vector<float> rmsnorm_ref(seq_len * d_model);
        compute_reference_rmsnorm(input.data(), gamma.data(), rmsnorm_ref.data(),
                                  seq_len, d_model, epsilon);

        // SIMD-computed RMSNorm (before quantization) should match reference within FP32 precision
        // We can't directly compare pre-quantization since kernel is fused, but dequantized
        // output should be very close to normalized reference
        float rel_l2_error = compute_relative_l2_error(dequant_simd.data(), rmsnorm_ref.data(),
                                                       dequant_simd.size());
        EXPECT_LT(rel_l2_error, 0.01f) << "SIMD result deviates > 1% from reference";
    }
#endif

    // =============================================================================
    // Performance Benchmarks (informational, not strict pass/fail)
    // =============================================================================

    /**
     * @test Benchmark: Fused vs Separate (single token decode)
     * Expected: Fused should be 1.5-2x faster (saves 1 FP32 buffer + 1 pass)
     */
    TEST_F(Test__FusedRMSNormQuantize, DISABLED_Benchmark_SingleToken)
    {
        const int seq_len = 1;
        const int d_model = 4864; // Qwen 7B
        const float epsilon = 1e-6f;
        const int num_iterations = 10000;

        std::vector<float> input(seq_len * d_model);
        std::vector<float> gamma(d_model);
        std::vector<int8_t> output_fused(seq_len * d_model);
        std::vector<float> scales_fused(seq_len);

        std::vector<float> rmsnorm_temp(seq_len * d_model);
        std::vector<int8_t> output_sep(seq_len * d_model);
        std::vector<float> scales_sep(seq_len);

        fill_random(input.data(), input.size());
        fill_random(gamma.data(), gamma.size());

        FusedRMSNormQuantize fused_kernel;

        // Warmup
        for (int i = 0; i < 100; ++i)
        {
            fused_kernel.execute(input.data(), gamma.data(), output_fused.data(),
                                 scales_fused.data(), seq_len, d_model, epsilon);
        }

        // Benchmark fused
        auto t0_fused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i)
        {
            fused_kernel.execute(input.data(), gamma.data(), output_fused.data(),
                                 scales_fused.data(), seq_len, d_model, epsilon);
        }
        auto t1_fused = std::chrono::high_resolution_clock::now();
        double ms_fused = std::chrono::duration<double, std::milli>(t1_fused - t0_fused).count();

        // Benchmark separate
        auto t0_sep = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i)
        {
            compute_reference_rmsnorm(input.data(), gamma.data(), rmsnorm_temp.data(),
                                      seq_len, d_model, epsilon);
            quantize_symmetric_per_row(rmsnorm_temp.data(), output_sep.data(), scales_sep.data(),
                                       seq_len, d_model);
        }
        auto t1_sep = std::chrono::high_resolution_clock::now();
        double ms_separate = std::chrono::duration<double, std::milli>(t1_sep - t0_sep).count();

        double speedup = ms_separate / ms_fused;

        std::cout << "\n=== FusedRMSNormQuantize Benchmark (Single Token) ===\n";
        std::cout << "Iterations: " << num_iterations << "\n";
        std::cout << "Fused:      " << ms_fused << " ms (" << (ms_fused / num_iterations) << " ms/iter)\n";
        std::cout << "Separate:   " << ms_separate << " ms (" << (ms_separate / num_iterations) << " ms/iter)\n";
        std::cout << "Speedup:    " << speedup << "x\n";
        std::cout << "=========================================\n";

        // Informational: expect 1.5-2x speedup, but don't fail if slightly lower
        EXPECT_GT(speedup, 1.2) << "Fused kernel should be faster than separate ops";
    }

} // namespace llaminar2
