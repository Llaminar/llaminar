/**
 * @file test_dequant_vs_llamacpp.cpp
 * @brief Comprehensive dequantization accuracy and performance tests vs llama.cpp
 *
 * This test suite validates Llaminar's quantized tensor implementations by:
 * 1. Creating synthetic quantized data via llama.cpp quantization functions
 * 2. Dequantizing with both Llaminar and llama.cpp
 * 3. Comparing outputs for accuracy
 * 4. Benchmarking performance
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>

// Llaminar quantized tensor headers
#include "tensors/Q2_KTensor.h"
#include "tensors/Q3_KTensor.h"
#include "tensors/Q4_0Tensor.h"
#include "tensors/Q4_1Tensor.h"
#include "tensors/Q5_KTensor.h"
#include "tensors/Q6_KTensor.h"
#include "tensors/Q8_0Tensor.h"
#include "tensors/IQ1_STensor.h"
#include "tensors/IQ1_MTensor.h"
#include "tensors/IQ2_XXSTensor.h"
#include "tensors/IQ2_XSTensor.h"
#include "tensors/IQ2_STensor.h"
#include "tensors/IQ3_XXSTensor.h"
#include "tensors/IQ3_STensor.h"
#include "tensors/IQ4_NLTensor.h"
#include "tensors/IQ4_XSTensor.h"

// llama.cpp headers
extern "C"
{
#include "ggml-quants.h"
}

namespace
{

    using namespace llaminar;

    /**
     * @brief Generate random FP32 data for testing
     */
    std::vector<float> generateRandomData(size_t n, float min_val = -1.0f, float max_val = 1.0f, uint32_t seed = 42)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(min_val, max_val);

        std::vector<float> data(n);
        for (size_t i = 0; i < n; ++i)
        {
            data[i] = dist(gen);
        }
        return data;
    }

    /**
     * @brief Compare two float arrays
     */
    struct ComparisonResult
    {
        bool passed;
        double max_abs_diff;
        double mean_abs_diff;
        double rel_l2_error;
        size_t num_mismatches;
    };

    ComparisonResult compareArrays(const float *a, const float *b, size_t n,
                                   double abs_tol = 1e-4, double rel_tol = 1e-3)
    {
        ComparisonResult result = {true, 0.0, 0.0, 0.0, 0};

        double sum_abs_diff = 0.0;
        double sum_sq_a = 0.0;
        double sum_sq_diff = 0.0;

        for (size_t i = 0; i < n; ++i)
        {
            double diff = std::abs(a[i] - b[i]);
            sum_abs_diff += diff;
            sum_sq_diff += diff * diff;
            sum_sq_a += a[i] * a[i];

            result.max_abs_diff = std::max(result.max_abs_diff, diff);

            double threshold = std::max(abs_tol, rel_tol * std::max(std::abs(a[i]), std::abs(b[i])));
            if (diff > threshold)
            {
                result.num_mismatches++;
            }
        }

        result.mean_abs_diff = sum_abs_diff / n;
        result.rel_l2_error = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_a) + 1e-10);
        result.passed = (result.num_mismatches == 0);

        return result;
    }

    /**
     * @brief High-resolution timer
     */
    class Timer
    {
    public:
        void start() { start_ = std::chrono::high_resolution_clock::now(); }
        double elapsed_ms() const
        {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start_).count();
        }

    private:
        std::chrono::high_resolution_clock::time_point start_;
    };

    // ============================================================================
    // ACCURACY TESTS
    // ============================================================================

    /**
     * @brief Test Q4_0 dequantization accuracy
     */
    TEST(DequantAccuracy, Q4_0_Accuracy)
    {
        constexpr size_t ROWS = 100;
        constexpr size_t COLS = 896; // Must be multiple of QK4_0 (32)

        // Generate random FP32 data
        auto fp32_data = generateRandomData(ROWS * COLS);

        // Quantize with llama.cpp
        size_t quantized_size = (COLS / QK4_0) * sizeof(block_q4_0);
        std::vector<uint8_t> quantized(ROWS * quantized_size);

        for (size_t row = 0; row < ROWS; ++row)
        {
            quantize_row_q4_0_ref(
                fp32_data.data() + row * COLS,
                reinterpret_cast<block_q4_0 *>(quantized.data() + row * quantized_size),
                COLS);
        }

        // Create Llaminar tensor (requires std::vector<int> shape, std::vector<uint8_t> data)
        std::vector<int> shape = {static_cast<int>(ROWS), static_cast<int>(COLS)};
        Q4_0Tensor llaminar_tensor(shape, quantized);

        // Dequantize with both
        std::vector<float> llaminar_output(ROWS * COLS);
        std::vector<float> llamacpp_output(ROWS * COLS);

        llaminar_tensor.decode_to_fp32(llaminar_output.data());

        for (size_t row = 0; row < ROWS; ++row)
        {
            dequantize_row_q4_0(
                reinterpret_cast<const block_q4_0 *>(quantized.data() + row * quantized_size),
                llamacpp_output.data() + row * COLS,
                COLS);
        }

        // Compare
        auto result = compareArrays(llaminar_output.data(), llamacpp_output.data(), ROWS * COLS);

        std::cout << "Q4_0 Accuracy Test:\n";
        std::cout << "  Tensor: " << ROWS << " × " << COLS << "\n";
        std::cout << "  Max abs diff: " << result.max_abs_diff << "\n";
        std::cout << "  Mean abs diff: " << result.mean_abs_diff << "\n";
        std::cout << "  Rel L2 error: " << result.rel_l2_error << "\n";
        std::cout << "  Mismatches: " << result.num_mismatches << " / " << (ROWS * COLS) << "\n";

        EXPECT_TRUE(result.passed) << "Q4_0 dequantization differs from llama.cpp";
    }

    /**
     * @brief Test Q8_0 dequantization accuracy
     */
    TEST(DequantAccuracy, Q8_0_Accuracy)
    {
        constexpr size_t ROWS = 100;
        constexpr size_t COLS = 896;

        auto fp32_data = generateRandomData(ROWS * COLS);

        size_t quantized_size = (COLS / QK8_0) * sizeof(block_q8_0);
        std::vector<uint8_t> quantized(ROWS * quantized_size);

        for (size_t row = 0; row < ROWS; ++row)
        {
            quantize_row_q8_0_ref(
                fp32_data.data() + row * COLS,
                reinterpret_cast<block_q8_0 *>(quantized.data() + row * quantized_size),
                COLS);
        }

        std::vector<int> shape = {static_cast<int>(ROWS), static_cast<int>(COLS)};
        Q8_0Tensor llaminar_tensor(shape, quantized);

        std::vector<float> llaminar_output(ROWS * COLS);
        std::vector<float> llamacpp_output(ROWS * COLS);

        llaminar_tensor.decode_to_fp32(llaminar_output.data());

        for (size_t row = 0; row < ROWS; ++row)
        {
            dequantize_row_q8_0(
                reinterpret_cast<const block_q8_0 *>(quantized.data() + row * quantized_size),
                llamacpp_output.data() + row * COLS,
                COLS);
        }

        auto result = compareArrays(llaminar_output.data(), llamacpp_output.data(), ROWS * COLS);

        std::cout << "Q8_0 Accuracy Test:\n";
        std::cout << "  Max abs diff: " << result.max_abs_diff << "\n";
        std::cout << "  Mismatches: " << result.num_mismatches << " / " << (ROWS * COLS) << "\n";

        EXPECT_TRUE(result.passed) << "Q8_0 dequantization differs from llama.cpp";
    }

    /**
     * @brief Test IQ4_NL dequantization accuracy
     */
    TEST(DequantAccuracy, IQ4_NL_Accuracy)
    {
        constexpr size_t ROWS = 100;
        constexpr size_t COLS = 896; // Must be multiple of QK4_NL (32)

        auto fp32_data = generateRandomData(ROWS * COLS);

        size_t quantized_size = (COLS / QK4_NL) * sizeof(block_iq4_nl);
        std::vector<uint8_t> quantized(ROWS * quantized_size);

        for (size_t row = 0; row < ROWS; ++row)
        {
            quantize_row_iq4_nl_ref(
                fp32_data.data() + row * COLS,
                reinterpret_cast<block_iq4_nl *>(quantized.data() + row * quantized_size),
                COLS);
        }

        std::vector<int> shape = {static_cast<int>(ROWS), static_cast<int>(COLS)};
        IQ4_NLTensor llaminar_tensor(shape, quantized);

        std::vector<float> llaminar_output(ROWS * COLS);
        std::vector<float> llamacpp_output(ROWS * COLS);

        llaminar_tensor.decode_to_fp32(llaminar_output.data());

        for (size_t row = 0; row < ROWS; ++row)
        {
            dequantize_row_iq4_nl(
                reinterpret_cast<const block_iq4_nl *>(quantized.data() + row * quantized_size),
                llamacpp_output.data() + row * COLS,
                COLS);
        }

        auto result = compareArrays(llaminar_output.data(), llamacpp_output.data(), ROWS * COLS);

        std::cout << "IQ4_NL Accuracy Test:\n";
        std::cout << "  Max abs diff: " << result.max_abs_diff << "\n";
        std::cout << "  Mismatches: " << result.num_mismatches << " / " << (ROWS * COLS) << "\n";

        EXPECT_TRUE(result.passed) << "IQ4_NL dequantization differs from llama.cpp";
    }

    /**
     * @brief Test IQ4_XS dequantization accuracy
     */
    TEST(DequantAccuracy, IQ4_XS_Accuracy)
    {
        constexpr size_t ROWS = 100;
        constexpr size_t COLS = 256; // Must be multiple of QK_K (256)

        auto fp32_data = generateRandomData(ROWS * COLS);

        size_t quantized_size = (COLS / QK_K) * sizeof(block_iq4_xs);
        std::vector<uint8_t> quantized(ROWS * quantized_size);

        for (size_t row = 0; row < ROWS; ++row)
        {
            quantize_row_iq4_xs_ref(
                fp32_data.data() + row * COLS,
                reinterpret_cast<block_iq4_xs *>(quantized.data() + row * quantized_size),
                COLS);
        }

        std::vector<int> shape = {static_cast<int>(ROWS), static_cast<int>(COLS)};
        IQ4_XSTensor llaminar_tensor(shape, quantized);

        std::vector<float> llaminar_output(ROWS * COLS);
        std::vector<float> llamacpp_output(ROWS * COLS);

        llaminar_tensor.decode_to_fp32(llaminar_output.data());

        for (size_t row = 0; row < ROWS; ++row)
        {
            dequantize_row_iq4_xs(
                reinterpret_cast<const block_iq4_xs *>(quantized.data() + row * quantized_size),
                llamacpp_output.data() + row * COLS,
                COLS);
        }

        auto result = compareArrays(llaminar_output.data(), llamacpp_output.data(), ROWS * COLS);

        std::cout << "IQ4_XS Accuracy Test:\n";
        std::cout << "  Max abs diff: " << result.max_abs_diff << "\n";
        std::cout << "  Mismatches: " << result.num_mismatches << " / " << (ROWS * COLS) << "\n";

        EXPECT_TRUE(result.passed) << "IQ4_XS dequantization differs from llama.cpp";
    }

    // ============================================================================
    // PERFORMANCE TESTS
    // ============================================================================

    /**
     * @brief Benchmark Q4_0 performance
     */
    TEST(DequantPerformance, Q4_0_Performance)
    {
        constexpr size_t ROWS = 500;
        constexpr size_t COLS = 3584; // Larger tensor for meaningful benchmark
        constexpr size_t WARMUP = 3;
        constexpr size_t ITERS = 20;

        auto fp32_data = generateRandomData(ROWS * COLS);

        size_t quantized_size = (COLS / QK4_0) * sizeof(block_q4_0);
        std::vector<uint8_t> quantized(ROWS * quantized_size);

        for (size_t row = 0; row < ROWS; ++row)
        {
            quantize_row_q4_0_ref(
                fp32_data.data() + row * COLS,
                reinterpret_cast<block_q4_0 *>(quantized.data() + row * quantized_size),
                COLS);
        }

        Q4_0Tensor llaminar_tensor({ROWS, COLS}, quantized.data());

        std::vector<float> llaminar_output(ROWS * COLS);
        std::vector<float> llamacpp_output(ROWS * COLS);

        Timer timer;

        // Warmup
        for (size_t i = 0; i < WARMUP; ++i)
        {
            llaminar_tensor.decode_to_fp32(llaminar_output.data());
#pragma omp parallel for
            for (int64_t row = 0; row < static_cast<int64_t>(ROWS); ++row)
            {
                dequantize_row_q4_0(
                    reinterpret_cast<const block_q4_0 *>(quantized.data() + row * quantized_size),
                    llamacpp_output.data() + row * COLS,
                    COLS);
            }
        }

        // Benchmark Llaminar
        timer.start();
        for (size_t i = 0; i < ITERS; ++i)
        {
            llaminar_tensor.decode_to_fp32(llaminar_output.data());
        }
        double llaminar_ms = timer.elapsed_ms() / ITERS;

        // Benchmark llama.cpp
        timer.start();
        for (size_t i = 0; i < ITERS; ++i)
        {
#pragma omp parallel for
            for (int64_t row = 0; row < static_cast<int64_t>(ROWS); ++row)
            {
                dequantize_row_q4_0(
                    reinterpret_cast<const block_q4_0 *>(quantized.data() + row * quantized_size),
                    llamacpp_output.data() + row * COLS,
                    COLS);
            }
        }
        double llamacpp_ms = timer.elapsed_ms() / ITERS;

        size_t total_bytes = ROWS * COLS * sizeof(float);
        double llaminar_gbps = (total_bytes / 1e9) / (llaminar_ms / 1000.0);
        double llamacpp_gbps = (total_bytes / 1e9) / (llamacpp_ms / 1000.0);
        double speedup = llamacpp_ms / llaminar_ms;

        std::cout << "\nQ4_0 Performance Benchmark (" << ROWS << " × " << COLS << "):\n";
        std::cout << "  Llaminar:  " << std::fixed << std::setprecision(3) << llaminar_ms << " ms ("
                  << std::setprecision(2) << llaminar_gbps << " GB/s)\n";
        std::cout << "  llama.cpp: " << std::fixed << std::setprecision(3) << llamacpp_ms << " ms ("
                  << std::setprecision(2) << llamacpp_gbps << " GB/s)\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "×\n";
        std::cout << "  OMP threads: " << omp_get_max_threads() << "\n";
    }

    /**
     * @brief Benchmark IQ4_NL performance with SIMD optimizations
     */
    TEST(DequantPerformance, IQ4_NL_Performance)
    {
        constexpr size_t ROWS = 500;
        constexpr size_t COLS = 3584;
        constexpr size_t WARMUP = 3;
        constexpr size_t ITERS = 20;

        auto fp32_data = generateRandomData(ROWS * COLS);

        size_t quantized_size = (COLS / QK4_NL) * sizeof(block_iq4_nl);
        std::vector<uint8_t> quantized(ROWS * quantized_size);

        for (size_t row = 0; row < ROWS; ++row)
        {
            quantize_row_iq4_nl_ref(
                fp32_data.data() + row * COLS,
                reinterpret_cast<block_iq4_nl *>(quantized.data() + row * quantized_size),
                COLS);
        }

        std::vector<int> shape = {static_cast<int>(ROWS), static_cast<int>(COLS)};
        IQ4_NLTensor llaminar_tensor(shape, quantized);

        std::vector<float> llaminar_output(ROWS * COLS);
        std::vector<float> llamacpp_output(ROWS * COLS);

        Timer timer;

        // Warmup
        for (size_t i = 0; i < WARMUP; ++i)
        {
            llaminar_tensor.decode_to_fp32(llaminar_output.data());
#pragma omp parallel for
            for (int64_t row = 0; row < static_cast<int64_t>(ROWS); ++row)
            {
                dequantize_row_iq4_nl(
                    reinterpret_cast<const block_iq4_nl *>(quantized.data() + row * quantized_size),
                    llamacpp_output.data() + row * COLS,
                    COLS);
            }
        }

        // Benchmark Llaminar
        timer.start();
        for (size_t i = 0; i < ITERS; ++i)
        {
            llaminar_tensor.decode_to_fp32(llaminar_output.data());
        }
        double llaminar_ms = timer.elapsed_ms() / ITERS;

        // Benchmark llama.cpp
        timer.start();
        for (size_t i = 0; i < ITERS; ++i)
        {
#pragma omp parallel for
            for (int64_t row = 0; row < static_cast<int64_t>(ROWS); ++row)
            {
                dequantize_row_iq4_nl(
                    reinterpret_cast<const block_iq4_nl *>(quantized.data() + row * quantized_size),
                    llamacpp_output.data() + row * COLS,
                    COLS);
            }
        }
        double llamacpp_ms = timer.elapsed_ms() / ITERS;

        size_t total_bytes = ROWS * COLS * sizeof(float);
        double llaminar_gbps = (total_bytes / 1e9) / (llaminar_ms / 1000.0);
        double llamacpp_gbps = (total_bytes / 1e9) / (llamacpp_ms / 1000.0);
        double speedup = llamacpp_ms / llaminar_ms;

        std::cout << "\nIQ4_NL Performance Benchmark (SIMD-optimized) (" << ROWS << " × " << COLS << "):\n";
        std::cout << "  Llaminar:  " << std::fixed << std::setprecision(3) << llaminar_ms << " ms ("
                  << std::setprecision(2) << llaminar_gbps << " GB/s)\n";
        std::cout << "  llama.cpp: " << std::fixed << std::setprecision(3) << llamacpp_ms << " ms ("
                  << std::setprecision(2) << llamacpp_gbps << " GB/s)\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "×\n";
    }

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0)
    {
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    if (rank == 0)
    {
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║   LLAMINAR vs LLAMA.CPP DEQUANTIZATION TESTS                   ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ OMP threads: " << std::setw(46) << std::left << omp_get_max_threads() << " ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    }

    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
