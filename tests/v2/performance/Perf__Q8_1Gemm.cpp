/**
 * @file Perf__Q8_1Gemm.cpp
 * @brief Performance tests for Q8_1 × Q8_0 GEMM kernel
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>

#include "kernels/cpu/gemm_v2/Q8_1GemmKernel.h"
#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"

using namespace llaminar2;

class Q8_1GemmPerformance : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Load Q8_0 model for weights (path relative to workspace root)
        model_path_ = "models/qwen2.5-0.5b-instruct-q8_0.gguf";
        loader_ = std::make_unique<ModelLoader>();

        if (!loader_->loadModel(model_path_))
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }
    }

    std::string model_path_;
    std::unique_ptr<ModelLoader> loader_;
};

/**
 * @brief Verify Q8_1GemmKernel compiles and constants are correct
 */
TEST_F(Q8_1GemmPerformance, CompilationTest)
{
    std::cout << "Q8_1GemmKernel header compiled successfully" << std::endl;
    std::cout << "Microkernel size: " << Q8_1GemmKernel::MR << "×" << Q8_1GemmKernel::NR << std::endl;
    std::cout << "Block size: " << Q8_1GemmKernel::BLOCK_SIZE << std::endl;
    std::cout << "VECTOR_WIDTH: " << Q8_1GemmKernel::VECTOR_WIDTH << std::endl;

    EXPECT_EQ(Q8_1GemmKernel::MR, 32);
    EXPECT_EQ(Q8_1GemmKernel::NR, 128); // Updated to optimal configuration (from parameter sweep)
    EXPECT_EQ(Q8_1GemmKernel::BLOCK_SIZE, 32);
    EXPECT_EQ(Q8_1GemmKernel::VECTOR_WIDTH, 16);
}

/**
 * @brief Performance benchmark: Large batched prefill (4096 tokens)
 *
 * This represents the high-throughput scenario for multi-user serving.
 * Tests Q8_1 (activations) × Q8_0 (weights) with dpbusd + compensation.
 *
 * Target: Match or exceed Q8_0×Q8_0 performance (~1500+ GFLOPS)
 */
TEST_F(Q8_1GemmPerformance, LargeBatchedPrefill)
{
    // Load Q8_0 weight tensor (B matrix)
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_template, nullptr);
    ASSERT_EQ(wq_template->native_type(), TensorType::Q8_0);

    auto q8_0_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
    ASSERT_NE(q8_0_template, nullptr);

    // Large prefill: M=4096 (tokens), N=896 (d_model), K=896
    const int M = 4096; // Large batch of tokens
    const int N = 896;  // Output features (d_model for Qwen 0.5B)
    const int K = 896;  // Input features

    // Verify K is multiple of 32
    ASSERT_EQ(K % 32, 0) << "K must be multiple of block size (32)";

    std::cout << "\n=== Q8_1 × Q8_0 GEMM Performance: Large Batched Prefill ===" << std::endl;
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Scenario: 4096-token prefill (high throughput)" << std::endl;
    std::cout << "Format: Q8_1 activations × Q8_0 weights (dpbusd + compensation)" << std::endl;

    // Create Q8_1 activation tensor (A matrix) by tiling the Q8_0 template
    const size_t template_rows = q8_0_template->shape()[0]; // 896
    const size_t rows_per_tile = template_rows;
    const size_t num_tiles = (M + rows_per_tile - 1) / rows_per_tile; // 5 tiles

    // Get raw Q8_0 block data from template
    const void *q8_0_template_data = q8_0_template->get_raw_block_at(0, 0);
    const size_t K_blocks = K / 32;
    const size_t blocks_per_row = K_blocks;

    // Allocate buffer for Q8_0 data (will convert to Q8_1)
    std::vector<uint8_t> A_q8_0_data(M * blocks_per_row * sizeof(Q8_0Block));

    // Tile the template data to fill M rows
    for (size_t tile = 0; tile < num_tiles; ++tile)
    {
        const size_t dst_row_start = tile * rows_per_tile;
        const size_t rows_to_copy = std::min(rows_per_tile, M - dst_row_start);
        const size_t bytes_to_copy = rows_to_copy * blocks_per_row * sizeof(Q8_0Block);

        std::memcpy(A_q8_0_data.data() + dst_row_start * blocks_per_row * sizeof(Q8_0Block),
                    q8_0_template_data,
                    bytes_to_copy);
    }

    // Create temporary Q8_0 tensor
    auto q8_0_A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{M, K}, A_q8_0_data);
    ASSERT_NE(q8_0_A, nullptr);

    // Convert Q8_0 → FP32 → Q8_1 (simulates activation quantization)
    std::vector<float> A_fp32(M * K);

    // Dequantize Q8_0 to FP32
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t kb = 0; kb < K_blocks; ++kb)
        {
            const Q8_0Block *block = reinterpret_cast<const Q8_0Block *>(
                q8_0_A->get_raw_block_at(i, kb));
            float scale = fp16_to_fp32(block->d);

            for (size_t k_in = 0; k_in < 32; ++k_in)
            {
                A_fp32[i * K + kb * 32 + k_in] = static_cast<float>(block->qs[k_in]) * scale;
            }
        }
    }

    // Quantize FP32 → Q8_1 (with pre-computed sums)
    auto q8_1_A = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
    ASSERT_NE(q8_1_A, nullptr);

    // B tensor is the Q8_0 weight (already correct size)
    auto q8_0_B = q8_0_template;

    // Allocate output matrix
    std::vector<float> C(M * N, 0.0f);

    // Warmup iterations
    constexpr int WARMUP = 10;
    for (int i = 0; i < WARMUP; ++i)
    {
        std::fill(C.begin(), C.end(), 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *q8_1_A, *q8_0_B, C.data(), N);
    }

    // Timed iterations
    constexpr int ITERATIONS = 50;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; ++i)
    {
        std::fill(C.begin(), C.end(), 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *q8_1_A, *q8_0_B, C.data(), N);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / ITERATIONS;

    // Calculate GFLOPS
    // GEMM: 2*M*N*K FLOPs (multiply + add for each element)
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (avg_ms / 1000.0);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Average time:  " << std::fixed << std::setprecision(3) << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput:    " << std::fixed << std::setprecision(1) << gflops << " GFLOPS" << std::endl;
    std::cout << "  FLOPs:         " << std::scientific << std::setprecision(2) << flops << std::endl;

    // Performance expectations
    std::cout << "\nPerformance Analysis:" << std::endl;
    std::cout << "  Q8_0×Q8_0 target: ~1500 GFLOPS" << std::endl;

    if (gflops >= 1500)
    {
        std::cout << "  Status: ✅ EXCELLENT (≥1500 GFLOPS, matches Q8_0×Q8_0)" << std::endl;
    }
    else if (gflops >= 1400)
    {
        std::cout << "  Status: ✅ GOOD (≥1400 GFLOPS, within 7% of target)" << std::endl;
    }
    else if (gflops >= 1200)
    {
        std::cout << "  Status: ⚠️  ACCEPTABLE (≥1200 GFLOPS, needs optimization)" << std::endl;
    }
    else
    {
        std::cout << "  Status: ❌ NEEDS WORK (<1200 GFLOPS, investigate bottlenecks)" << std::endl;
    }

    // Sanity check: verify some outputs are non-zero
    int non_zero_count = 0;
    for (const auto &val : C)
    {
        if (std::abs(val) > 1e-6)
        {
            non_zero_count++;
        }
    }
    double non_zero_pct = 100.0 * non_zero_count / C.size();
    std::cout << "\nSanity check: " << std::fixed << std::setprecision(1)
              << non_zero_pct << "% non-zero values" << std::endl;
    EXPECT_GT(non_zero_pct, 10.0) << "Too few non-zero values, computation may be broken";

    // Performance expectation (should be competitive with Q8_0×Q8_0)
    EXPECT_GT(gflops, 1000.0) << "Performance significantly below expectations";
}

/**
 * @brief Performance scaling test: GFLOPS vs M (batch size)
 *
 * Tests how performance scales with increasing M (number of rows/tokens).
 * Sweeps M from 512 to 16384 in powers of 2.
 *
 * Expected behavior:
 * - Small M (512-1024): Lower GFLOPS due to startup overhead
 * - Medium M (2048-4096): Good GFLOPS, approaching peak
 * - Large M (8192-16384): Peak GFLOPS, amortized overhead
 */
TEST_F(Q8_1GemmPerformance, ThroughputScalingWithM)
{
    // Load Q8_0 weight tensor (B matrix)
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_template, nullptr);
    ASSERT_EQ(wq_template->native_type(), TensorType::Q8_0);

    auto q8_0_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
    ASSERT_NE(q8_0_template, nullptr);

    // Fixed dimensions
    const int N = 896; // Output features (d_model for Qwen 0.5B)
    const int K = 896; // Input features
    const size_t K_blocks = K / 32;

    std::cout << "\n=== Q8_1 × Q8_0 GEMM Performance Scaling Test ===" << std::endl;
    std::cout << "Fixed: N=" << N << ", K=" << K << std::endl;
    std::cout << "Variable: M (batch size) from 512 to 16384" << std::endl;
    std::cout << "Format: Q8_1 activations × Q8_0 weights" << std::endl;
    std::cout << "\n"
              << std::setw(8) << "M"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(15) << "Speedup"
              << std::setw(20) << "Status" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    // Get template data for tiling
    const void *q8_0_template_data = q8_0_template->get_raw_block_at(0, 0);
    const size_t template_rows = q8_0_template->shape()[0]; // 896
    const size_t blocks_per_row = K_blocks;

    double baseline_gflops = 0.0;

    // Sweep M from 512 to 16384 in powers of 2
    for (int M = 512; M <= 16384; M *= 2)
    {
        // Create Q8_0 A tensor by tiling template
        const size_t num_tiles = (M + template_rows - 1) / template_rows;
        std::vector<uint8_t> A_q8_0_data(M * blocks_per_row * sizeof(Q8_0Block));

        for (size_t tile = 0; tile < num_tiles; ++tile)
        {
            const size_t dst_row_start = tile * template_rows;
            const size_t rows_to_copy = std::min(template_rows, static_cast<size_t>(M - dst_row_start));
            const size_t bytes_to_copy = rows_to_copy * blocks_per_row * sizeof(Q8_0Block);

            std::memcpy(A_q8_0_data.data() + dst_row_start * blocks_per_row * sizeof(Q8_0Block),
                        q8_0_template_data,
                        bytes_to_copy);
        }

        auto q8_0_A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)}, A_q8_0_data);

        // Convert Q8_0 → FP32 → Q8_1
        std::vector<float> A_fp32(M * K);

        for (int i = 0; i < M; ++i)
        {
            for (size_t kb = 0; kb < K_blocks; ++kb)
            {
                const Q8_0Block *block = reinterpret_cast<const Q8_0Block *>(
                    q8_0_A->get_raw_block_at(i, kb));
                float scale = fp16_to_fp32(block->d);

                for (size_t k_in = 0; k_in < 32; ++k_in)
                {
                    A_fp32[i * K + kb * 32 + k_in] = static_cast<float>(block->qs[k_in]) * scale;
                }
            }
        }

        auto q8_1_A = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});
        ASSERT_NE(q8_1_A, nullptr);

        // Allocate output
        std::vector<float> C(M * N, 0.0f);

        // Warmup (fewer iterations for larger M to save time)
        const int warmup = (M >= 8192) ? 3 : 5;
        for (int i = 0; i < warmup; ++i)
        {
            std::fill(C.begin(), C.end(), 0.0f);
            Q8_1GemmKernel::gemm(M, N, K, *q8_1_A, *q8_0_template, C.data(), N);
        }

        // Timed iterations (fewer for larger M)
        const int iterations = (M >= 8192) ? 10 : 20;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            std::fill(C.begin(), C.end(), 0.0f);
            Q8_1GemmKernel::gemm(M, N, K, *q8_1_A, *q8_0_template, C.data(), N);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avg_ms = total_ms / iterations;

        // Calculate GFLOPS
        double flops = 2.0 * M * N * K;
        double gflops = (flops / 1e9) / (avg_ms / 1000.0);

        // Calculate speedup (relative to M=512)
        if (M == 512)
        {
            baseline_gflops = gflops;
        }
        double speedup = baseline_gflops > 0 ? gflops / baseline_gflops : 1.0;

        // Status based on GFLOPS
        std::string status;
        if (gflops >= 450)
        {
            status = "✅ Excellent";
        }
        else if (gflops >= 400)
        {
            status = "✅ Good";
        }
        else if (gflops >= 350)
        {
            status = "⚠️  Acceptable";
        }
        else
        {
            status = "❌ Needs work";
        }

        std::cout << std::setw(8) << M
                  << std::setw(12) << std::fixed << std::setprecision(2) << avg_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << gflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "×"
                  << std::setw(20) << status << std::endl;

        // Sanity check
        int non_zero_count = 0;
        for (const auto &val : C)
        {
            if (std::abs(val) > 1e-6)
                non_zero_count++;
        }
        EXPECT_GT(non_zero_count, M * N / 10) << "Too few non-zero values at M=" << M;
    }

    std::cout << std::string(70, '-') << std::endl;
    std::cout << "\nExpected behavior:" << std::endl;
    std::cout << "  • Small M (512-1024): Lower GFLOPS (startup overhead dominates)" << std::endl;
    std::cout << "  • Medium M (2048-4096): Good GFLOPS (amortized overhead)" << std::endl;
    std::cout << "  • Large M (8192-16384): Peak GFLOPS (compute-bound)" << std::endl;
    std::cout << "  • Speedup should plateau as M increases (approaching peak)" << std::endl;
}
