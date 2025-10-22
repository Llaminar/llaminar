/**
 * @file test_iq4_nl_microkernel.cpp
 * @brief Correctness regression tests for IQ4_NL microkernel path.
 *
 * Ensures the environment-enabled microkernel (LLAMINAR_IQ4_MICROKERNEL=1) produces
 * identical results to llama.cpp reference dequantization across multiple shapes,
 * random seeds, and edge cases (non-multiple-of-block columns).
 *
 * Usage:
 *   OMP_NUM_THREADS=14 OMP_PLACES=cores OMP_PROC_BIND=close \
 *   LLAMINAR_IQ4_MICROKERNEL=1 ./test_iq4_nl_microkernel --gtest_filter=IQ4NLMicrokernel.*
 *
 * The test suite explicitly forces the microkernel path by setting the env var from code
 * when absent to avoid accidental omission. It validates:
 *   1. Exact match (0 mismatches) for full-block shapes.
 *   2. Exact match with tail-zeroing behavior for non-divisible column counts.
 *   3. Deterministic output across multiple seeds.
 *   4. Multi-row decoding parity under row-level parallelization.
 *
 * NOTE: The microkernel currently activates only when columns are a multiple of BLOCK_SIZE.
 *       For tail shapes we expect fallback path to run; test still asserts correctness.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <iostream>

#include "tensors/IQ4_NLTensor.h"

extern "C"
{
#include "ggml-quants.h"
}

using namespace llaminar;

namespace
{

    static std::vector<float> genRandom(size_t n, uint32_t seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i)
            out[i] = dist(gen);
        return out;
    }

    struct Metrics
    {
        double max_diff;
        double mean_diff;
        double rel_l2;
        size_t mismatches;
    };

    static Metrics compare(const float *a, const float *b, size_t n, double tol = 1e-4)
    {
        Metrics m{0, 0, 0, 0};
        double sum_diff = 0, sum_sq_diff = 0, sum_sq_a = 0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = std::abs(a[i] - b[i]);
            sum_diff += diff;
            sum_sq_diff += diff * diff;
            sum_sq_a += a[i] * a[i];
            if (diff > tol)
                m.mismatches++;
            if (diff > m.max_diff)
                m.max_diff = diff;
        }
        m.mean_diff = sum_diff / n;
        m.rel_l2 = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_a) + 1e-10);
        return m;
    }

    static void ensureMicrokernelEnv()
    {
        const char *val = std::getenv("LLAMINAR_IQ4_MICROKERNEL");
        if (!val)
        {
            // Set for this process (best-effort); subsequent debugEnv() snapshot should pick it up
            setenv("LLAMINAR_IQ4_MICROKERNEL", "1", 1);
            std::cerr << "[TestInit] Set LLAMINAR_IQ4_MICROKERNEL=1 (was unset)\n";
        }
    }

    TEST(IQ4NLMicrokernel, FullBlockShapesMultipleSeeds)
    {
        ensureMicrokernelEnv();
        const std::vector<std::pair<int, int>> shapes = {
            {1, 32}, {4, 256}, {8, 896}, {16, 3584}}; // 3584 matches performance benchmark size
        const std::vector<uint32_t> seeds = {1, 42, 1337, 2025};

        for (auto [rows, cols] : shapes)
        {
            for (auto seed : seeds)
            {
                auto fp32 = genRandom(static_cast<size_t>(rows) * cols, seed);
                size_t qsize_per_row = (cols / QK4_NL) * sizeof(block_iq4_nl);
                std::vector<uint8_t> quant(static_cast<size_t>(rows) * qsize_per_row);
                for (int r = 0; r < rows; ++r)
                {
                    quantize_row_iq4_nl_ref(fp32.data() + r * cols,
                                            reinterpret_cast<block_iq4_nl *>(quant.data() + static_cast<size_t>(r) * qsize_per_row), cols);
                }
                IQ4_NLTensor tensor({rows, cols}, quant);
                std::vector<float> out_micro(rows * cols), out_ref(rows * cols);
                tensor.decode_to_fp32(out_micro.data());
                for (int r = 0; r < rows; ++r)
                {
                    dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(quant.data() + static_cast<size_t>(r) * qsize_per_row),
                                          out_ref.data() + static_cast<size_t>(r) * cols, cols);
                }
                auto metrics = compare(out_micro.data(), out_ref.data(), static_cast<size_t>(rows) * cols);
                EXPECT_EQ(metrics.mismatches, 0) << "Mismatch detected for shape " << rows << "x" << cols << " seed=" << seed
                                                 << " max_diff=" << metrics.max_diff << " rel_l2=" << metrics.rel_l2;
            }
        }
    }

    TEST(IQ4NLMicrokernel, NonMultipleOfBlockColumnsTailHandling)
    {
        ensureMicrokernelEnv();

        // Test various non-multiple-of-32 column sizes
        std::vector<int> test_cols = {10, 50, 63, 100, 127, 200};

        for (int cols : test_cols)
        {
            int rows = 5;

            // Calculate blocks_per_row for per-row layout
            size_t blocks_per_row = (cols + QK4_NL - 1) / QK4_NL;
            size_t padded_cols = blocks_per_row * QK4_NL;

            // Generate random data for full padded size (to avoid llama.cpp asserts)
            auto fp32_padded = genRandom(static_cast<size_t>(rows) * padded_cols, 777 + cols);

            // Quantize using padded size (llama.cpp requires multiple of QK4_NL)
            size_t qsize_per_row = blocks_per_row * sizeof(block_iq4_nl);
            std::vector<uint8_t> quant(static_cast<size_t>(rows) * qsize_per_row);

            for (int r = 0; r < rows; ++r)
            {
                quantize_row_iq4_nl_ref(
                    fp32_padded.data() + static_cast<size_t>(r) * padded_cols,
                    reinterpret_cast<block_iq4_nl *>(quant.data() + static_cast<size_t>(r) * qsize_per_row),
                    padded_cols // Must be multiple of 32
                );
            }

            // Create tensor with actual (non-padded) shape
            IQ4_NLTensor tensor({rows, cols}, quant);

            // Verify logical_k and padded_k
            EXPECT_EQ(tensor.logical_k(), static_cast<size_t>(cols)) << "logical_k mismatch for cols=" << cols;
            EXPECT_EQ(tensor.padded_k(), padded_cols) << "padded_k mismatch for cols=" << cols;

            // Decode using tensor (should handle tail properly)
            std::vector<float> decoded(rows * cols);
            tensor.decode_to_fp32(decoded.data());

            // Build reference: decode full padded row, then extract logical_k elements
            std::vector<float> ref(rows * cols);
            for (int r = 0; r < rows; ++r)
            {
                std::vector<float> padded_ref(padded_cols);
                dequantize_row_iq4_nl(
                    reinterpret_cast<const block_iq4_nl *>(quant.data() + static_cast<size_t>(r) * qsize_per_row),
                    padded_ref.data(),
                    padded_cols);
                // Copy only logical_k elements
                std::memcpy(ref.data() + r * cols, padded_ref.data(), cols * sizeof(float));
            }

            // Compare (should match exactly within quantization error)
            auto metrics = compare(decoded.data(), ref.data(), static_cast<size_t>(rows) * cols);
            EXPECT_EQ(metrics.mismatches, 0) << "Tail handling failed for cols=" << cols
                                             << " (padded_cols=" << padded_cols << ", blocks_per_row=" << blocks_per_row << ")";
        }
    }

    TEST(IQ4NLMicrokernel, DeterminismSingleShapeMultipleRuns)
    {
        ensureMicrokernelEnv();
        int rows = 10, cols = 256;
        auto fp32 = genRandom(static_cast<size_t>(rows) * cols, 999);
        size_t qsize_per_row = (cols / QK4_NL) * sizeof(block_iq4_nl);
        std::vector<uint8_t> quant(static_cast<size_t>(rows) * qsize_per_row);
        for (int r = 0; r < rows; ++r)
        {
            quantize_row_iq4_nl_ref(fp32.data() + r * cols,
                                    reinterpret_cast<block_iq4_nl *>(quant.data() + static_cast<size_t>(r) * qsize_per_row), cols);
        }
        IQ4_NLTensor tensor({rows, cols}, quant);
        std::vector<float> first(rows * cols), second(rows * cols), ref(rows * cols);
        tensor.decode_to_fp32(first.data());
        tensor.decode_to_fp32(second.data());
        for (int r = 0; r < rows; ++r)
        {
            dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(quant.data() + static_cast<size_t>(r) * qsize_per_row),
                                  ref.data() + static_cast<size_t>(r) * cols, cols);
        }
        auto m1 = compare(first.data(), ref.data(), static_cast<size_t>(rows) * cols);
        auto m2 = compare(second.data(), ref.data(), static_cast<size_t>(rows) * cols);
        EXPECT_EQ(m1.mismatches, 0) << "First run mismatch";
        EXPECT_EQ(m2.mismatches, 0) << "Second run mismatch";
    }

} // namespace

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ IQ4_NL MICROKERNEL CORRECTNESS TESTS                        ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Env LLAMINAR_IQ4_MICROKERNEL=" << (std::getenv("LLAMINAR_IQ4_MICROKERNEL") ? std::getenv("LLAMINAR_IQ4_MICROKERNEL") : "(unset)") << "\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    return RUN_ALL_TESTS();
}
