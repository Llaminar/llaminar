/**
 * @file Perf__Q8_0Gemm.cpp
 * @brief Performance tests for Q8_0 × Q8_0 GEMM kernel
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>

#include "kernels/cpu/gemm_v2/Q8_0GemmKernel.h"
#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"

using namespace llaminar2;

class Q8_0GemmPerformance : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Load Q8_0 model (path relative to workspace root)
        model_path_ = "models/qwen2.5-0.5b-instruct-q8_0.gguf";
        loader_ = std::make_unique<ModelLoader>();

        if (!loader_->loadModel(model_path_))
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }
    }

    /**
     * @brief Naive reference Q8_0 GEMM for correctness validation
     */
    void reference_q8_0_gemm(const Q8_0Tensor &A, const Q8_0Tensor &B,
                             float *C, int M, int N, int K)
    {
        const int K_blocks = K / 32;

        // Get block pointers
        const Q8_0Block *A_blocks = reinterpret_cast<const Q8_0Block *>(A.get_raw_block_at(0, 0));
        const Q8_0Block *B_blocks = reinterpret_cast<const Q8_0Block *>(B.get_raw_block_at(0, 0));

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float accum = 0.0f;

                // Sum over K blocks
                for (int kb = 0; kb < K_blocks; ++kb)
                {
                    const Q8_0Block &a_block = A_blocks[i * K_blocks + kb];
                    const Q8_0Block &b_block = B_blocks[j * K_blocks + kb];

                    // Compute int8×int8 dot product
                    int32_t dot = 0;
                    for (int k = 0; k < 32; ++k)
                    {
                        dot += static_cast<int32_t>(a_block.qs[k]) * static_cast<int32_t>(b_block.qs[k]);
                    }

                    // Apply per-block scales
                    float a_scale = fp16_to_fp32(a_block.d);
                    float b_scale = fp16_to_fp32(b_block.d);
                    accum += static_cast<float>(dot) * a_scale * b_scale;
                }

                C[i * N + j] = accum;
            }
        }
    }

    std::string model_path_;
    std::unique_ptr<ModelLoader> loader_;
};

/**
 * @brief Verify Q8_0GemmKernel compiles and constants are correct
 */
TEST_F(Q8_0GemmPerformance, CompilationTest)
{
    std::cout << "Q8_0GemmKernel header compiled successfully" << std::endl;
    std::cout << "Microkernel size: " << Q8_0GemmKernel::MR << "×" << Q8_0GemmKernel::NR << std::endl;
    std::cout << "Block size: " << Q8_0GemmKernel::BLOCK_SIZE << std::endl;

    EXPECT_EQ(Q8_0GemmKernel::MR, 8);
    EXPECT_EQ(Q8_0GemmKernel::NR, 8);
    EXPECT_EQ(Q8_0GemmKernel::BLOCK_SIZE, 32);
}

/**
 * @brief Load real Q8_0 weight tensor and verify we can access it
 */
TEST_F(Q8_0GemmPerformance, LoadQ8_0Tensor)
{
    // Load a Q8_0 weight tensor (first layer Q projection)
    auto wq_tensor = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_tensor, nullptr) << "Failed to load tensor";
    ASSERT_EQ(wq_tensor->native_type(), TensorType::Q8_0) << "Wrong tensor type";

    const auto &shape = wq_tensor->shape();
    ASSERT_EQ(shape.size(), 2) << "Expected 2D tensor";

    std::cout << "Loaded Q8_0 tensor shape: [" << shape[0] << ", " << shape[1] << "]" << std::endl;

    // Verify we can cast to Q8_0Tensor
    auto q8_tensor = std::dynamic_pointer_cast<Q8_0Tensor>(wq_tensor);
    ASSERT_NE(q8_tensor, nullptr) << "Failed to cast to Q8_0Tensor";

    // Verify we can access blocks
    const void *block_ptr = q8_tensor->get_raw_block_at(0, 0);
    ASSERT_NE(block_ptr, nullptr) << "Failed to get block pointer";
}

/**
 * @brief Test Q8_0 GEMM correctness against naive reference implementation
 */
TEST_F(Q8_0GemmPerformance, CorrectnessWithRealWeights)
{
    // Load Q8_0 tensor from model
    auto wq = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq, nullptr);
    ASSERT_EQ(wq->native_type(), TensorType::Q8_0);

    auto q8_wq = std::dynamic_pointer_cast<Q8_0Tensor>(wq);
    ASSERT_NE(q8_wq, nullptr);

    // Test with small matrices (wq × wq^T for simplicity)
    const int M = 16;
    const int N = 16;
    const int K = static_cast<int>(q8_wq->shape()[1]); // 896 for Qwen 0.5B

    // Verify K is multiple of 32
    ASSERT_EQ(K % 32, 0) << "K must be multiple of block size (32)";

    std::cout << "\n=== Q8_0 GEMM Correctness Test ===" << std::endl;
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Allocate output matrices
    std::vector<float> C_kernel(M * N, 0.0f);
    std::vector<float> C_reference(M * N, 0.0f);

    // Run optimized kernel
    auto t0_kernel = std::chrono::high_resolution_clock::now();
    Q8_0GemmKernel::gemm(M, N, K, *q8_wq, *q8_wq, C_kernel.data(), N);
    auto t1_kernel = std::chrono::high_resolution_clock::now();
    double time_kernel = std::chrono::duration<double, std::milli>(t1_kernel - t0_kernel).count();

    // Run naive reference
    auto t0_ref = std::chrono::high_resolution_clock::now();
    reference_q8_0_gemm(*q8_wq, *q8_wq, C_reference.data(), M, N, K);
    auto t1_ref = std::chrono::high_resolution_clock::now();
    double time_ref = std::chrono::duration<double, std::milli>(t1_ref - t0_ref).count();

    std::cout << "Kernel time:     " << std::fixed << std::setprecision(3) << time_kernel << " ms" << std::endl;
    std::cout << "Reference time:  " << std::fixed << std::setprecision(3) << time_ref << " ms" << std::endl;
    std::cout << "Speedup:         " << std::fixed << std::setprecision(2) << (time_ref / time_kernel) << "x" << std::endl;

    // Compare results
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    int num_mismatches = 0;
    constexpr double rel_tol = 1e-3; // 0.1% tolerance for Q8_0
    constexpr double abs_tol = 1e-2; // Absolute tolerance for near-zero values

    for (int i = 0; i < M * N; ++i)
    {
        double kernel_val = static_cast<double>(C_kernel[i]);
        double ref_val = static_cast<double>(C_reference[i]);
        double abs_diff = std::abs(kernel_val - ref_val);
        double denom = std::max(std::abs(ref_val), 1e-6);
        double rel_diff = abs_diff / denom;

        max_abs_diff = std::max(max_abs_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        if (rel_diff > rel_tol && abs_diff > abs_tol)
        {
            num_mismatches++;
        }
    }

    std::cout << "Max abs diff:    " << std::scientific << std::setprecision(2) << max_abs_diff << std::endl;
    std::cout << "Max rel diff:    " << std::scientific << std::setprecision(2) << max_rel_diff << std::endl;
    std::cout << "Mismatches:      " << num_mismatches << " / " << (M * N)
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * num_mismatches / (M * N)) << "%)" << std::endl;

    // Print sample values for inspection
    std::cout << "\nSample values (first 4×4 block):" << std::endl;
    std::cout << "Kernel:" << std::endl;
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "  ";
        for (int j = 0; j < 4; ++j)
        {
            std::cout << std::setw(12) << std::fixed << std::setprecision(4) << C_kernel[i * N + j];
        }
        std::cout << std::endl;
    }
    std::cout << "Reference:" << std::endl;
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "  ";
        for (int j = 0; j < 4; ++j)
        {
            std::cout << std::setw(12) << std::fixed << std::setprecision(4) << C_reference[i * N + j];
        }
        std::cout << std::endl;
    }

    // Validate correctness
    EXPECT_LT(max_rel_diff, 1e-2) << "Maximum relative difference too large";
    EXPECT_LT(num_mismatches, M * N / 100) << "Too many mismatches (>1%)";
}

/**
 * @brief Performance benchmark: Large batched prefill (4096 tokens)
 *
 * This represents the high-throughput scenario for multi-user serving.
 * We measure GFLOPS to compare against dense INT8 baseline (1273 GOPS from Phase 2).
 *
 * Target: 1500+ GFLOPS (15-20% improvement over Phase 2 baseline)
 */
TEST_F(Q8_0GemmPerformance, LargeBatchedPrefill)
{
    // Load Q8_0 weight tensor to get real quantized data
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_template, nullptr);
    ASSERT_EQ(wq_template->native_type(), TensorType::Q8_0);

    auto q8_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
    ASSERT_NE(q8_template, nullptr);

    // Large prefill: M=4096 (tokens), N=896 (d_model), K=896
    // Create properly sized test tensors by tiling the loaded weight
    const int M = 4096; // Large batch of tokens
    const int N = 896;  // Output features (d_model for Qwen 0.5B)
    const int K = 896;  // Input features

    // Verify K is multiple of 32
    ASSERT_EQ(K % 32, 0) << "K must be multiple of block size (32)";

    std::cout << "\n=== Q8_0 GEMM Performance: Large Batched Prefill ===" << std::endl;
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Scenario: 4096-token prefill (high throughput)" << std::endl;

    // Create test tensors by repeating the loaded weight data
    // A: [M=4096, K=896] - simulate activation tensor
    // B: [N=896, K=896] - use loaded weight (already correct size)
    const size_t template_rows = q8_template->shape()[0]; // 896
    const size_t rows_per_tile = template_rows;
    const size_t num_tiles = (M + rows_per_tile - 1) / rows_per_tile; // 5 tiles

    // Get raw block data from template
    const void *template_data = q8_template->get_raw_block_at(0, 0);
    const size_t block_size = sizeof(Q8_0Block);
    const size_t K_blocks = K / 32;
    const size_t blocks_per_row = K_blocks;
    const size_t template_size_bytes = template_rows * blocks_per_row * block_size;

    // Allocate buffer for tiled A tensor
    std::vector<uint8_t> A_data(M * blocks_per_row * block_size);

    // Tile the template data to fill M rows
    for (size_t tile = 0; tile < num_tiles; ++tile)
    {
        const size_t dst_row_start = tile * rows_per_tile;
        const size_t rows_to_copy = std::min(rows_per_tile, M - dst_row_start);
        const size_t bytes_to_copy = rows_to_copy * blocks_per_row * block_size;

        std::memcpy(A_data.data() + dst_row_start * blocks_per_row * block_size,
                    template_data,
                    bytes_to_copy);
    }

    // Create Q8_0 tensors directly
    auto q8_A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{M, K}, A_data);
    ASSERT_NE(q8_A, nullptr);

    // B tensor is the loaded weight (already correct size)
    auto q8_B = q8_template;

    // Allocate output matrix
    std::vector<float> C(M * N, 0.0f);

    // Warmup iterations - increased to ensure stable CPU frequency and cache state
    constexpr int WARMUP = 10;
    for (int i = 0; i < WARMUP; ++i)
    {
        std::fill(C.begin(), C.end(), 0.0f);
        Q8_0GemmKernel::gemm(M, N, K, *q8_A, *q8_B, C.data(), N);
    }

    // Timed iterations - increased to reduce variance and get stable measurements
    // Large GEMM operations need more iterations for accurate timing (reduces chrono overhead)
    constexpr int ITERATIONS = 50;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; ++i)
    {
        std::fill(C.begin(), C.end(), 0.0f);
        Q8_0GemmKernel::gemm(M, N, K, *q8_A, *q8_B, C.data(), N);
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
    std::cout << "  Phase 2 INT8 baseline: 1273 GFLOPS" << std::endl;

    if (gflops >= 1500)
    {
        std::cout << "  Status: ✅ EXCELLENT (>1500 GFLOPS, 18%+ improvement)" << std::endl;
    }
    else if (gflops >= 1400)
    {
        std::cout << "  Status: ✅ GOOD (>1400 GFLOPS, 10%+ improvement)" << std::endl;
    }
    else if (gflops >= 1273)
    {
        std::cout << "  Status: ⚠️  ACCEPTABLE (>baseline, needs optimization)" << std::endl;
    }
    else
    {
        std::cout << "  Status: ❌ NEEDS WORK (<baseline, investigate bottlenecks)" << std::endl;
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
    std::cout << "\nSanity check: " << non_zero_pct << "% non-zero values" << std::endl;
    EXPECT_GT(non_zero_pct, 10.0) << "Too few non-zero values, computation may be broken";
}

/**
 * @brief TODO: Add correctness test with properly constructed tensors
 */
TEST_F(Q8_0GemmPerformance, DISABLED_CorrectnessTest)
{
    // Will implement after understanding Q8_0Tensor construction from scratch
    GTEST_SKIP() << "Not yet implemented - needs Q8_0Tensor test data construction";
}

/**
 * @brief TODO: Add performance benchmark
 */
TEST_F(Q8_0GemmPerformance, DISABLED_PerformanceTest)
{
    // Will implement after correctness test passes
    GTEST_SKIP() << "Not yet implemented - needs correctness validation first";
}
