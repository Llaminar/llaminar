/**
 * @file test_iq4_vs_llamacpp.cpp
 * @brief Quick IQ4 accuracy and performance test vs llama.cpp
 *
 * Simple focused test for IQ4_NL and IQ4_XS formats.
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include <gtest/gtest.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include "tensors/IQ4_NLTensor.h"
#include "tensors/IQ4_XSTensor.h"

extern "C"
{
#include "ggml-quants.h"
}

namespace
{

    using namespace llaminar;

    std::vector<float> genRandom(size_t n, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        std::vector<float> data(n);
        for (size_t i = 0; i < n; ++i)
            data[i] = dist(gen);
        return data;
    }

    struct CompareResult
    {
        double max_diff, mean_diff, rel_l2;
        size_t mismatches;
    };

    CompareResult compare(const float *a, const float *b, size_t n, double tol = 1e-4)
    {
        CompareResult r = {0, 0, 0, 0};
        double sum_diff = 0, sum_sq_a = 0, sum_sq_diff = 0;

        for (size_t i = 0; i < n; ++i)
        {
            double diff = std::abs(a[i] - b[i]);
            sum_diff += diff;
            sum_sq_diff += diff * diff;
            sum_sq_a += a[i] * a[i];
            r.max_diff = std::max(r.max_diff, diff);
            if (diff > tol)
                r.mismatches++;
        }

        r.mean_diff = sum_diff / n;
        r.rel_l2 = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_a) + 1e-10);
        return r;
    }

    TEST(IQ4Accuracy, IQ4_NL_vs_LlamaCpp)
    {
        constexpr size_t ROWS = 100, COLS = 896;
        auto fp32 = genRandom(ROWS * COLS);

        size_t qsize = (COLS / QK4_NL) * sizeof(block_iq4_nl);
        std::vector<uint8_t> quant(ROWS * qsize);

        for (size_t r = 0; r < ROWS; ++r)
        {
            quantize_row_iq4_nl_ref(fp32.data() + r * COLS,
                                    reinterpret_cast<block_iq4_nl *>(quant.data() + r * qsize), COLS);
        }

        std::vector<int> shape = {(int)ROWS, (int)COLS};
        IQ4_NLTensor llam_tensor(shape, quant);

        std::vector<float> llam_out(ROWS * COLS), cpp_out(ROWS * COLS);

        llam_tensor.decode_to_fp32(llam_out.data());
        for (size_t r = 0; r < ROWS; ++r)
        {
            dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(quant.data() + r * qsize),
                                  cpp_out.data() + r * COLS, COLS);
        }

        auto res = compare(llam_out.data(), cpp_out.data(), ROWS * COLS);

        std::cout << "IQ4_NL Accuracy (" << ROWS << "×" << COLS << "):\n";
        std::cout << "  Max diff: " << res.max_diff << "\n";
        std::cout << "  Mean diff: " << res.mean_diff << "\n";
        std::cout << "  Rel L2: " << res.rel_l2 << "\n";
        std::cout << "  Mismatches: " << res.mismatches << " / " << (ROWS * COLS) << "\n";

        EXPECT_EQ(res.mismatches, 0) << "IQ4_NL differs from llama.cpp";
    }

    TEST(IQ4Performance, IQ4_NL_Benchmark)
    {
        constexpr size_t ROWS = 500, COLS = 3584, WARMUP = 3, ITERS = 20;
        auto fp32 = genRandom(ROWS * COLS);

        size_t qsize = (COLS / QK4_NL) * sizeof(block_iq4_nl);
        std::vector<uint8_t> quant(ROWS * qsize);

        for (size_t r = 0; r < ROWS; ++r)
        {
            quantize_row_iq4_nl_ref(fp32.data() + r * COLS,
                                    reinterpret_cast<block_iq4_nl *>(quant.data() + r * qsize), COLS);
        }

        std::vector<int> shape = {(int)ROWS, (int)COLS};
        IQ4_NLTensor llam_tensor(shape, quant);

        std::vector<float> llam_out(ROWS * COLS), cpp_out(ROWS * COLS);

        // Warmup
        for (size_t i = 0; i < WARMUP; ++i)
        {
            llam_tensor.decode_to_fp32(llam_out.data());
#pragma omp parallel for
            for (int64_t r = 0; r < (int64_t)ROWS; ++r)
            {
                dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(quant.data() + r * qsize),
                                      cpp_out.data() + r * COLS, COLS);
            }
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ITERS; ++i)
            llam_tensor.decode_to_fp32(llam_out.data());
        auto t1 = std::chrono::high_resolution_clock::now();

        auto t2 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ITERS; ++i)
        {
#pragma omp parallel for
            for (int64_t r = 0; r < (int64_t)ROWS; ++r)
            {
                dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(quant.data() + r * qsize),
                                      cpp_out.data() + r * COLS, COLS);
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();

        double llam_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        double cpp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / ITERS;

        double bytes = ROWS * COLS * sizeof(float);
        double llam_gbps = (bytes / 1e9) / (llam_ms / 1000.0);
        double cpp_gbps = (bytes / 1e9) / (cpp_ms / 1000.0);

        std::cout << "\nIQ4_NL Performance (" << ROWS << "×" << COLS << ", " << omp_get_max_threads() << " threads):\n";
        std::cout << "  Llaminar:  " << std::fixed << std::setprecision(3) << llam_ms << " ms ("
                  << std::setprecision(2) << llam_gbps << " GB/s)\n";
        std::cout << "  llama.cpp: " << std::fixed << std::setprecision(3) << cpp_ms << " ms ("
                  << std::setprecision(2) << cpp_gbps << " GB/s)\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << (cpp_ms / llam_ms) << "×\n\n";
    }

} // namespace

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   IQ4 vs LLAMA.CPP DEQUANTIZATION TESTS                        ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ OMP threads: " << std::setw(48) << std::left << omp_get_max_threads() << " ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    return RUN_ALL_TESTS();
}
