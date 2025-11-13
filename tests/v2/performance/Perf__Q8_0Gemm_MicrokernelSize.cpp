/**
 * @file Perf__Q8_0Gemm_MicrokernelSize.cpp
 * @brief Benchmark different Q8_0 GEMM microkernel sizes (8×8, 12×12, 16×16, etc.)
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

#include "kernels/cpu/gemm_v2/Q8_0GemmKernel.h"
#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

class Q8_0MicrokernelSizeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Load Q8_0 model
        model_path_ = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q8_0.gguf";
        loader_ = std::make_unique<ModelLoader>();

        if (!loader_->loadModel(model_path_))
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }
    }

    template <typename KernelType>
    double benchmark_kernel(const Q8_0Tensor &A, const Q8_0Tensor &B,
                            int M, int N, int K, int warmup, int iterations)
    {
        std::vector<float> C(M * N, 0.0f);

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            std::fill(C.begin(), C.end(), 0.0f);
            KernelType::gemm(M, N, K, A, B, C.data(), N);
        }

        // Timed iterations
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            std::fill(C.begin(), C.end(), 0.0f);
            KernelType::gemm(M, N, K, A, B, C.data(), N);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return total_ms / iterations; // Average ms per iteration
    }

    std::string model_path_;
    std::unique_ptr<ModelLoader> loader_;
};

/**
 * @brief Compare microkernel sizes on large prefill workload
 */
TEST_F(Q8_0MicrokernelSizeTest, LargePrefillComparison)
{
    // Load Q8_0 weight tensor
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_template, nullptr);
    ASSERT_EQ(wq_template->native_type(), TensorType::Q8_0);

    auto q8_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
    ASSERT_NE(q8_template, nullptr);

    // Large prefill: M=4096 (tokens), N=896, K=896
    const int M = 4096;
    const int N = 896;
    const int K = 896;

    // Create test tensors by tiling
    const size_t template_rows = q8_template->shape()[0];
    const size_t K_blocks = K / 32;
    const size_t blocks_per_row = K_blocks;
    const size_t block_size = sizeof(Q8_0Block);

    std::vector<uint8_t> A_data(M * blocks_per_row * block_size);
    const void *template_data = q8_template->get_raw_block_at(0, 0);

    for (size_t tile = 0; tile < (M + template_rows - 1) / template_rows; ++tile)
    {
        const size_t dst_row_start = tile * template_rows;
        const size_t rows_to_copy = std::min(template_rows, M - dst_row_start);
        const size_t bytes_to_copy = rows_to_copy * blocks_per_row * block_size;

        std::memcpy(A_data.data() + dst_row_start * blocks_per_row * block_size,
                    template_data, bytes_to_copy);
    }

    auto q8_A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{M, K}, A_data);
    auto q8_B = q8_template;

    std::cout << "\n=== Q8_0 GEMM Microkernel Size Sweep ===" << std::endl;
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Warmup: 10 iterations, Timed: 50 iterations\n"
              << std::endl;

    const int warmup = 10;
    const int iterations = 50;

    // Helper to print results
    auto print_result = [&](const std::string &name, double time_ms, double baseline_time)
    {
        double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
        double speedup = baseline_time / time_ms;
        std::cout << std::setw(12) << name << ": "
                  << std::fixed << std::setprecision(2) << std::setw(8) << time_ms << " ms, "
                  << std::setw(7) << std::setprecision(1) << gflops << " GFLOPS, "
                  << std::setprecision(2) << speedup << "× speedup" << std::endl;
        return time_ms;
    };

    // Baseline 8×8
    double time_8x8 = benchmark_kernel<Q8_0GemmKernelTemplate<8, 8>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    std::cout << "\n--- Square Microkernels ---" << std::endl;
    print_result("8×8", time_8x8, time_8x8);

    double time_12x12 = benchmark_kernel<Q8_0GemmKernelTemplate<12, 12>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("12×12", time_12x12, time_8x8);

    double time_16x16 = benchmark_kernel<Q8_0GemmKernelTemplate<16, 16>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("16×16", time_16x16, time_8x8);

    double time_24x24 = benchmark_kernel<Q8_0GemmKernelTemplate<24, 24>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("24×24", time_24x24, time_8x8);

    double time_32x32 = benchmark_kernel<Q8_0GemmKernelTemplate<32, 32>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("32×32", time_32x32, time_8x8);

    double time_48x48 = benchmark_kernel<Q8_0GemmKernelTemplate<48, 48>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("48×48", time_48x48, time_8x8);

    double time_64x64 = benchmark_kernel<Q8_0GemmKernelTemplate<64, 64>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("64×64", time_64x64, time_8x8);

    // Note: Larger sizes (96×96, 128×128) exceed thread-local storage limits
    // Would require dynamic allocation (96×96 = 4.7MB, 128×128 = 8.4MB per thread)

    // Asymmetric variants
    std::cout << "\n--- Asymmetric Microkernels (MR < NR) ---" << std::endl;

    double time_8x16 = benchmark_kernel<Q8_0GemmKernelTemplate<8, 16>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("8×16", time_8x16, time_8x8);

    double time_8x24 = benchmark_kernel<Q8_0GemmKernelTemplate<8, 24>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("8×24", time_8x24, time_8x8);

    double time_8x32 = benchmark_kernel<Q8_0GemmKernelTemplate<8, 32>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("8×32", time_8x32, time_8x8);

    double time_16x32 = benchmark_kernel<Q8_0GemmKernelTemplate<16, 32>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("16×32", time_16x32, time_8x8);

    double time_16x64 = benchmark_kernel<Q8_0GemmKernelTemplate<16, 64>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("16×64", time_16x64, time_8x8);

    double time_32x64 = benchmark_kernel<Q8_0GemmKernelTemplate<32, 64>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("32×64", time_32x64, time_8x8);

    // Note: Larger asymmetric sizes (32×128, etc.) exceed storage limits

    std::cout << "\n--- Asymmetric Microkernels (MR > NR) ---" << std::endl;

    double time_16x8 = benchmark_kernel<Q8_0GemmKernelTemplate<16, 8>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("16×8", time_16x8, time_8x8);

    double time_24x8 = benchmark_kernel<Q8_0GemmKernelTemplate<24, 8>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("24×8", time_24x8, time_8x8);

    double time_32x8 = benchmark_kernel<Q8_0GemmKernelTemplate<32, 8>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("32×8", time_32x8, time_8x8);

    double time_32x16 = benchmark_kernel<Q8_0GemmKernelTemplate<32, 16>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("32×16", time_32x16, time_8x8);

    double time_64x16 = benchmark_kernel<Q8_0GemmKernelTemplate<64, 16>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("64×16", time_64x16, time_8x8);

    double time_64x32 = benchmark_kernel<Q8_0GemmKernelTemplate<64, 32>>(
        *q8_A, *q8_B, M, N, K, warmup, iterations);
    print_result("64×32", time_64x32, time_8x8);

    // Note: Larger asymmetric sizes (128×32, etc.) exceed storage limits

    // Find overall best
    std::vector<std::pair<std::string, double>> all_results = {
        {"8×8", time_8x8}, {"12×12", time_12x12}, {"16×16", time_16x16}, {"24×24", time_24x24}, {"32×32", time_32x32}, {"48×48", time_48x48}, {"64×64", time_64x64}, {"8×16", time_8x16}, {"8×24", time_8x24}, {"8×32", time_8x32}, {"16×32", time_16x32}, {"16×64", time_16x64}, {"32×64", time_32x64}, {"16×8", time_16x8}, {"24×8", time_24x8}, {"32×8", time_32x8}, {"32×16", time_32x16}, {"64×16", time_64x16}, {"64×32", time_64x32}};

    auto best = std::min_element(all_results.begin(), all_results.end(),
                                 [](const auto &a, const auto &b)
                                 { return a.second < b.second; });

    std::cout << "\n=== BEST CONFIGURATION ===" << std::endl;
    double best_gflops = (2.0 * M * N * K) / (best->second * 1e6);
    std::cout << best->first << ": " << std::fixed << std::setprecision(2)
              << best->second << " ms, " << std::setprecision(1) << best_gflops
              << " GFLOPS (" << std::setprecision(2) << (time_8x8 / best->second)
              << "× vs 8×8)" << std::endl;
}

/**
 * @brief Test A Prefetch Distance with optimal 32×64 microkernel
 */
TEST_F(Q8_0MicrokernelSizeTest, PrefetchDistanceComparison)
{
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_template, nullptr);
    auto q8_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
    ASSERT_NE(q8_template, nullptr);

    const int M = 4096;
    const int N = 896;
    const int K = 896;

    const size_t template_rows = q8_template->shape()[0];
    const size_t K_blocks = K / 32;
    const size_t blocks_per_row = K_blocks;
    const size_t block_size = sizeof(Q8_0Block);

    std::vector<uint8_t> A_data(M * K_blocks * block_size);
    std::vector<uint8_t> B_data(N * K_blocks * block_size);

    const uint8_t *template_data = reinterpret_cast<const uint8_t *>(q8_template->data());

    for (int i = 0; i < M; ++i)
    {
        const uint8_t *src_row = template_data + (i % template_rows) * blocks_per_row * block_size;
        uint8_t *dst_row = A_data.data() + i * K_blocks * block_size;
        std::memcpy(dst_row, src_row, blocks_per_row * block_size);
    }
    for (int j = 0; j < N; ++j)
    {
        const uint8_t *src_row = template_data + (j % template_rows) * blocks_per_row * block_size;
        uint8_t *dst_row = B_data.data() + j * K_blocks * block_size;
        std::memcpy(dst_row, src_row, blocks_per_row * block_size);
    }

    auto A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)}, A_data);
    auto B = std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(K), static_cast<size_t>(N)}, B_data);

    std::cout << "\n=== A Prefetch Distance Sweep (32×64 microkernel) ===" << std::endl;
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    auto print_result = [&](const std::string &name, double time_ms, double baseline_time)
    {
        double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
        double speedup = baseline_time / time_ms;
        std::cout << name << ": " << std::fixed << std::setprecision(2) << time_ms
                  << " ms, " << std::setprecision(1) << gflops << " GFLOPS ("
                  << std::setprecision(2) << speedup << "×)" << std::endl;
    };

    using Kernel_Prefetch0 = Q8_0GemmKernelTemplate<32, 64, 0, 16>;
    using Kernel_Prefetch1 = Q8_0GemmKernelTemplate<32, 64, 1, 16>;
    using Kernel_Prefetch2 = Q8_0GemmKernelTemplate<32, 64, 2, 16>; // Current default
    using Kernel_Prefetch3 = Q8_0GemmKernelTemplate<32, 64, 3, 16>;
    using Kernel_Prefetch4 = Q8_0GemmKernelTemplate<32, 64, 4, 16>;

    double time_prefetch0 = benchmark_kernel<Kernel_Prefetch0>(*A, *B, M, N, K, 10, 50);
    double time_prefetch1 = benchmark_kernel<Kernel_Prefetch1>(*A, *B, M, N, K, 10, 50);
    double time_prefetch2 = benchmark_kernel<Kernel_Prefetch2>(*A, *B, M, N, K, 10, 50);
    double time_prefetch3 = benchmark_kernel<Kernel_Prefetch3>(*A, *B, M, N, K, 10, 50);
    double time_prefetch4 = benchmark_kernel<Kernel_Prefetch4>(*A, *B, M, N, K, 10, 50);

    std::cout << "\nResults (baseline = prefetch distance 2):\n"
              << std::endl;
    print_result("Prefetch 0 (disabled)", time_prefetch0, time_prefetch2);
    print_result("Prefetch 1", time_prefetch1, time_prefetch2);
    print_result("Prefetch 2 (current)", time_prefetch2, time_prefetch2);
    print_result("Prefetch 3", time_prefetch3, time_prefetch2);
    print_result("Prefetch 4", time_prefetch4, time_prefetch2);

    std::vector<std::pair<std::string, double>> results = {
        {"Prefetch 0", time_prefetch0}, {"Prefetch 1", time_prefetch1}, {"Prefetch 2", time_prefetch2}, {"Prefetch 3", time_prefetch3}, {"Prefetch 4", time_prefetch4}};

    auto best = std::min_element(results.begin(), results.end(),
                                 [](const auto &a, const auto &b)
                                 { return a.second < b.second; });

    std::cout << "\n=== BEST PREFETCH DISTANCE ===" << std::endl;
    double best_gflops = (2.0 * M * N * K) / (best->second * 1e6);
    std::cout << best->first << ": " << std::fixed << std::setprecision(2)
              << best->second << " ms, " << std::setprecision(1) << best_gflops
              << " GFLOPS (" << std::setprecision(2) << (time_prefetch2 / best->second)
              << "× vs current default)" << std::endl;
}

/**
 * @brief Test Vector Width with optimal 32×64 microkernel
 */
TEST_F(Q8_0MicrokernelSizeTest, VectorWidthComparison)
{
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_template, nullptr);
    auto q8_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
    ASSERT_NE(q8_template, nullptr);

    const int M = 4096;
    const int N = 896;
    const int K = 896;

    const size_t template_rows = q8_template->shape()[0];
    const size_t K_blocks = K / 32;
    const size_t blocks_per_row = K_blocks;
    const size_t block_size = sizeof(Q8_0Block);

    std::vector<uint8_t> A_data(M * K_blocks * block_size);
    std::vector<uint8_t> B_data(N * K_blocks * block_size);

    const uint8_t *template_data = reinterpret_cast<const uint8_t *>(q8_template->data());

    for (int i = 0; i < M; ++i)
    {
        const uint8_t *src_row = template_data + (i % template_rows) * blocks_per_row * block_size;
        uint8_t *dst_row = A_data.data() + i * K_blocks * block_size;
        std::memcpy(dst_row, src_row, blocks_per_row * block_size);
    }
    for (int j = 0; j < N; ++j)
    {
        const uint8_t *src_row = template_data + (j % template_rows) * blocks_per_row * block_size;
        uint8_t *dst_row = B_data.data() + j * K_blocks * block_size;
        std::memcpy(dst_row, src_row, blocks_per_row * block_size);
    }

    auto A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)}, A_data);
    auto B = std::make_unique<Q8_0Tensor>(std::vector<size_t>{static_cast<size_t>(K), static_cast<size_t>(N)}, B_data);

    std::cout << "\n=== Vector Width Sweep (32×64 microkernel) ===" << std::endl;
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << " (K_blocks=" << K_blocks << ")" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    auto print_result = [&](const std::string &name, double time_ms, double baseline_time)
    {
        double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
        double speedup = baseline_time / time_ms;
        std::cout << name << ": " << std::fixed << std::setprecision(2) << time_ms
                  << " ms, " << std::setprecision(1) << gflops << " GFLOPS ("
                  << std::setprecision(2) << speedup << "×)" << std::endl;
    };

    using Kernel_Width8 = Q8_0GemmKernelTemplate<32, 64, 2, 8>;
    using Kernel_Width16 = Q8_0GemmKernelTemplate<32, 64, 2, 16>; // Current default

    double time_width8 = benchmark_kernel<Kernel_Width8>(*A, *B, M, N, K, 10, 50);
    double time_width16 = benchmark_kernel<Kernel_Width16>(*A, *B, M, N, K, 10, 50);

    std::cout << "\nResults (baseline = vector width 16):\n"
              << std::endl;
    print_result("Vector Width 8", time_width8, time_width16);
    print_result("Vector Width 16 (current)", time_width16, time_width16);

    // Note: Width 32 would require more complex loop structure for partial chunks
    std::cout << "\nNote: Vector width 32 requires additional implementation complexity" << std::endl;
    std::cout << "      for handling partial 16-element chunks (K_blocks=" << K_blocks << ")." << std::endl;
    std::cout << "      Only testing 8 and 16 for now." << std::endl;

    std::vector<std::pair<std::string, double>> results = {
        {"Width 8", time_width8}, {"Width 16", time_width16}};

    auto best = std::min_element(results.begin(), results.end(),
                                 [](const auto &a, const auto &b)
                                 { return a.second < b.second; });

    std::cout << "\n=== BEST VECTOR WIDTH ===" << std::endl;
    double best_gflops = (2.0 * M * N * K) / (best->second * 1e6);
    std::cout << best->first << ": " << std::fixed << std::setprecision(2)
              << best->second << " ms, " << std::setprecision(1) << best_gflops
              << " GFLOPS (" << std::setprecision(2) << (time_width16 / best->second)
              << "× vs current default)" << std::endl;
}

/**
 * @brief NC Blocking Sweep - Test different N-dimension cache blocking sizes
 *
 * NC blocking improves L2/L3 cache reuse for large N dimensions.
 * Target: Keep NC × KC block of B in cache (~512KB optimal)
 */
TEST_F(Q8_0MicrokernelSizeTest, NCBlockingSweep)
{
    // Load template weights
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    auto q8_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
    ASSERT_NE(q8_template, nullptr);

    const int M = 4096, N = 896, K = 896;
    const size_t K_blocks = K / 32;
    const size_t block_size = sizeof(Q8_0Block);

    // Create test tensors by tiling template data
    std::vector<uint8_t> A_data(M * K_blocks * block_size);
    std::vector<uint8_t> B_data(N * K_blocks * block_size);

    const uint8_t *template_data = reinterpret_cast<const uint8_t *>(q8_template->data());
    const size_t template_rows = q8_template->shape()[0];
    const size_t template_kb = (q8_template->shape()[1] / 32);

    for (int i = 0; i < M; ++i)
    {
        const uint8_t *src = template_data + (i % template_rows) * template_kb * block_size;
        uint8_t *dst = A_data.data() + i * K_blocks * block_size;
        for (size_t kb = 0; kb < K_blocks; ++kb)
        {
            std::memcpy(dst + kb * block_size, src + (kb % template_kb) * block_size, block_size);
        }
    }

    for (int j = 0; j < N; ++j)
    {
        const uint8_t *src = template_data + (j % template_rows) * template_kb * block_size;
        uint8_t *dst = B_data.data() + j * K_blocks * block_size;
        for (size_t kb = 0; kb < K_blocks; ++kb)
        {
            std::memcpy(dst + kb * block_size, src + (kb % template_kb) * block_size, block_size);
        }
    }

    auto A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{M, K}, A_data);
    auto B = std::make_unique<Q8_0Tensor>(std::vector<size_t>{K, N}, B_data);

    std::cout << "\n=== NC Blocking Sweep (32×64 microkernel) ===" << std::endl;
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    // Test NC values: 0 (auto), 128, 256, 512, 896 (full N), 1024, 2048
    // NC must be multiple of NR=64
    using Kernel_NC0 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 0, 0>;       // Auto (default)
    using Kernel_NC128 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 128, 0>;   // Small block
    using Kernel_NC256 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 256, 0>;   // Medium block
    using Kernel_NC512 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 512, 0>;   // Target size
    using Kernel_NC896 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 896, 0>;   // Full N (no blocking)
    using Kernel_NC1024 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 1024, 0>; // Large block

    double time_nc0 = benchmark_kernel<Kernel_NC0>(*A, *B, M, N, K, 10, 50);
    double time_nc128 = benchmark_kernel<Kernel_NC128>(*A, *B, M, N, K, 10, 50);
    double time_nc256 = benchmark_kernel<Kernel_NC256>(*A, *B, M, N, K, 10, 50);
    double time_nc512 = benchmark_kernel<Kernel_NC512>(*A, *B, M, N, K, 10, 50);
    double time_nc896 = benchmark_kernel<Kernel_NC896>(*A, *B, M, N, K, 10, 50);
    double time_nc1024 = benchmark_kernel<Kernel_NC1024>(*A, *B, M, N, K, 10, 50);

    std::cout << "\nResults (baseline = NC=896, no blocking):" << std::endl;
    std::cout << std::endl;

    auto print_result = [&](const std::string &label, double time_ms)
    {
        double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
        double speedup = time_nc896 / time_ms;
        std::cout << label << ": " << std::fixed << std::setprecision(2) << time_ms
                  << " ms, " << std::setprecision(1) << gflops << " GFLOPS ("
                  << std::setprecision(2) << speedup << "×)" << std::endl;
    };

    print_result("NC=0 (auto)   ", time_nc0);
    print_result("NC=128        ", time_nc128);
    print_result("NC=256        ", time_nc256);
    print_result("NC=512        ", time_nc512);
    print_result("NC=896 (none) ", time_nc896);
    print_result("NC=1024       ", time_nc1024);

    std::vector<std::pair<std::string, double>> results = {
        {"NC=0 (auto)", time_nc0}, {"NC=128", time_nc128}, {"NC=256", time_nc256}, {"NC=512", time_nc512}, {"NC=896 (none)", time_nc896}, {"NC=1024", time_nc1024}};

    auto best = std::min_element(results.begin(), results.end(),
                                 [](const auto &a, const auto &b)
                                 { return a.second < b.second; });

    std::cout << "\n=== BEST NC BLOCKING ===" << std::endl;
    double best_gflops = (2.0 * M * N * K) / (best->second * 1e6);
    std::cout << best->first << ": " << std::fixed << std::setprecision(2)
              << best->second << " ms, " << std::setprecision(1) << best_gflops
              << " GFLOPS (" << std::setprecision(2) << (time_nc896 / best->second)
              << "× vs no blocking)" << std::endl;
}

/**
 * @brief KC Blocking Sweep - Test different K-dimension cache blocking sizes
 *
 * KC blocking enables K > MAX_K_BLOCKS (128) and improves cache reuse.
 * For K=896 (28 blocks), KC has minimal impact. Test with larger K for real benefit.
 */
TEST_F(Q8_0MicrokernelSizeTest, KCBlockingSweep)
{
    // Load template weights
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    auto q8_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
    ASSERT_NE(q8_template, nullptr);

    const int M = 4096, N = 896, K = 896;
    const size_t K_blocks = K / 32;
    const size_t block_size = sizeof(Q8_0Block);

    // Create test tensors by tiling template data
    std::vector<uint8_t> A_data(M * K_blocks * block_size);
    std::vector<uint8_t> B_data(N * K_blocks * block_size);

    const uint8_t *template_data = reinterpret_cast<const uint8_t *>(q8_template->data());
    const size_t template_rows = q8_template->shape()[0];
    const size_t template_kb = (q8_template->shape()[1] / 32);

    for (int i = 0; i < M; ++i)
    {
        const uint8_t *src = template_data + (i % template_rows) * template_kb * block_size;
        uint8_t *dst = A_data.data() + i * K_blocks * block_size;
        for (size_t kb = 0; kb < K_blocks; ++kb)
        {
            std::memcpy(dst + kb * block_size, src + (kb % template_kb) * block_size, block_size);
        }
    }

    for (int j = 0; j < N; ++j)
    {
        const uint8_t *src = template_data + (j % template_rows) * template_kb * block_size;
        uint8_t *dst = B_data.data() + j * K_blocks * block_size;
        for (size_t kb = 0; kb < K_blocks; ++kb)
        {
            std::memcpy(dst + kb * block_size, src + (kb % template_kb) * block_size, block_size);
        }
    }

    auto A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{M, K}, A_data);
    auto B = std::make_unique<Q8_0Tensor>(std::vector<size_t>{K, N}, B_data);

    std::cout << "\n=== KC Blocking Sweep (32×64 microkernel) ===" << std::endl;
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << " (K_blocks=" << K_blocks << ")" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << "Note: K=896 is small (28 blocks), KC blocking most beneficial for K > 4096" << std::endl;
    std::cout << std::endl;

    // Test KC values: 0 (auto=28), 8, 14, 28 (full), 64, 128 (max storage)
    using Kernel_KC0 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 0, 0>;     // Auto (default)
    using Kernel_KC8 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 0, 8>;     // Small block
    using Kernel_KC14 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 0, 14>;   // Half
    using Kernel_KC28 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 0, 28>;   // Full (no blocking)
    using Kernel_KC64 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 0, 64>;   // Large block
    using Kernel_KC128 = Q8_0GemmKernelTemplate<32, 64, 4, 8, 0, 128>; // Max storage

    double time_kc0 = benchmark_kernel<Kernel_KC0>(*A, *B, M, N, K, 10, 50);
    double time_kc8 = benchmark_kernel<Kernel_KC8>(*A, *B, M, N, K, 10, 50);
    double time_kc14 = benchmark_kernel<Kernel_KC14>(*A, *B, M, N, K, 10, 50);
    double time_kc28 = benchmark_kernel<Kernel_KC28>(*A, *B, M, N, K, 10, 50);
    double time_kc64 = benchmark_kernel<Kernel_KC64>(*A, *B, M, N, K, 10, 50);
    double time_kc128 = benchmark_kernel<Kernel_KC128>(*A, *B, M, N, K, 10, 50);

    std::cout << "Results (baseline = KC=28, no blocking):" << std::endl;
    std::cout << std::endl;

    auto print_result = [&](const std::string &label, double time_ms)
    {
        double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
        double speedup = time_kc28 / time_ms;
        std::cout << label << ": " << std::fixed << std::setprecision(2) << time_ms
                  << " ms, " << std::setprecision(1) << gflops << " GFLOPS ("
                  << std::setprecision(2) << speedup << "×)" << std::endl;
    };

    print_result("KC=0 (auto)   ", time_kc0);
    print_result("KC=8          ", time_kc8);
    print_result("KC=14         ", time_kc14);
    print_result("KC=28 (none)  ", time_kc28);
    print_result("KC=64         ", time_kc64);
    print_result("KC=128 (max)  ", time_kc128);

    std::vector<std::pair<std::string, double>> results = {
        {"KC=0 (auto)", time_kc0}, {"KC=8", time_kc8}, {"KC=14", time_kc14}, {"KC=28 (none)", time_kc28}, {"KC=64", time_kc64}, {"KC=128 (max)", time_kc128}};

    auto best = std::min_element(results.begin(), results.end(),
                                 [](const auto &a, const auto &b)
                                 { return a.second < b.second; });

    std::cout << "\n=== BEST KC BLOCKING ===" << std::endl;
    double best_gflops = (2.0 * M * N * K) / (best->second * 1e6);
    std::cout << best->first << ": " << std::fixed << std::setprecision(2)
              << best->second << " ms, " << std::setprecision(1) << best_gflops
              << " GFLOPS (" << std::setprecision(2) << (time_kc28 / best->second)
              << "× vs no blocking)" << std::endl;

    std::cout << "\nNote: For K > 4096, KC blocking becomes essential (enables K > MAX_K_BLOCKS)" << std::endl;
}
