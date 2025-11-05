/**
 * @file Perf__Phase7_CUTLASS.cu
 * @brief Performance benchmark for Phase 7 CUTLASS int8 GEMM with pre-converted INT8 weights
 *
 * Measures throughput (TFLOPS) for various matrix sizes with pre-converted INT8 weights:
 * - Small: 64×64×64 (single token decode)
 * - Medium: 512×512×512 (small batch)
 * - Large: 2048×2048×2048 (large batch/prefill)
 * - Huge: 4096×4096×4096 (stress test)
 *
 * This version simulates the production flow where IQ4_NL → INT8 conversion
 * happens once at model load time, not per-iteration.
 */

#include <gtest/gtest.h>
#include "kernels/cuda/CudaGemmKernelPhase7_CUTLASS.h"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace llaminar::v2;

namespace
{

    constexpr int8_t kvalues_iq4nl[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113};

    struct IQ4_NLBlock
    {
        uint8_t quants[16];
        uint16_t scale;
    } __attribute__((packed));

    float fp16_to_fp32(uint16_t h)
    {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exponent = (h & 0x7C00) >> 10;
        uint32_t mantissa = (h & 0x03FF) << 13;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                uint32_t result = sign;
                return *reinterpret_cast<float *>(&result);
            }
            exponent = 1;
            while ((mantissa & 0x00800000) == 0)
            {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x007FFFFF;
        }
        else if (exponent == 31)
        {
            exponent = 255;
        }
        else
        {
            exponent += 127 - 15;
        }

        uint32_t result = sign | (exponent << 23) | mantissa;
        return *reinterpret_cast<float *>(&result);
    }

    uint16_t fp32_to_fp16(float f)
    {
        uint32_t bits = *reinterpret_cast<uint32_t *>(&f);
        uint32_t sign = (bits & 0x80000000) >> 16;
        int32_t exponent = ((bits & 0x7F800000) >> 23) - 127 + 15;
        uint32_t mantissa = (bits & 0x007FFFFF) >> 13;

        if (exponent <= 0)
        {
            return sign;
        }
        else if (exponent >= 31)
        {
            return sign | 0x7C00;
        }

        return sign | (exponent << 10) | mantissa;
    }

    /**
     * @brief Convert IQ4_NL blocks to INT8 + per-column scales
     *
     * This simulates the one-time conversion that happens at model load time.
     * For each IQ4_NL block (32 elements):
     * 1. Extract 4-bit indices
     * 2. Lookup INT8 values using kvalues_iq4nl table
     * 3. Store FP16 scale as FP32 per column
     *
     * Output layout:
     * - B_int8: [K×N] row-major INT8 matrix
     * - scales_B: [N] per-column FP32 scales (averaged across K blocks)
     */
    void iq4nl_to_int8_with_scales(
        const IQ4_NLBlock *blocks,
        int8_t *B_int8,
        float *scales_B,
        int K,
        int N)
    {
        constexpr int BLOCK_SIZE = 32;
        int K_blocks = K / BLOCK_SIZE;
        int N_blocks = N / BLOCK_SIZE;

        // Initialize scales
        std::fill(scales_B, scales_B + N, 0.0f);
        std::vector<int> scale_counts(N, 0);

        for (int k_block = 0; k_block < K_blocks; ++k_block)
        {
            for (int n_block = 0; n_block < N_blocks; ++n_block)
            {
                const IQ4_NLBlock &block = blocks[k_block * N_blocks + n_block];
                float block_scale = fp16_to_fp32(block.scale);

                // Decode 16 bytes → 32 int8 values
                for (int i = 0; i < 16; ++i)
                {
                    uint8_t byte_val = block.quants[i];
                    uint8_t nibble0 = byte_val & 0x0F;
                    uint8_t nibble1 = byte_val >> 4;

                    int8_t val0 = kvalues_iq4nl[nibble0];
                    int8_t val1 = kvalues_iq4nl[nibble1];

                    int k_pos = k_block * BLOCK_SIZE + (i * 2);
                    int n_pos = n_block * BLOCK_SIZE;

                    B_int8[k_pos * N + n_pos + (i * 2)] = val0;
                    B_int8[k_pos * N + n_pos + (i * 2 + 1)] = val1;
                }

                // Accumulate scales for this column block
                for (int i = 0; i < BLOCK_SIZE; ++i)
                {
                    int col = n_block * BLOCK_SIZE + i;
                    scales_B[col] += block_scale;
                    scale_counts[col]++;
                }
            }
        }

        // Average scales across K blocks
        for (int col = 0; col < N; ++col)
        {
            if (scale_counts[col] > 0)
            {
                scales_B[col] /= scale_counts[col];
            }
        }
    }

    void quantize_to_iq4nl(const float *data, IQ4_NLBlock *blocks, int rows, int cols)
    {
        constexpr int BLOCK_SIZE = 32;
        int num_blocks_per_row = cols / BLOCK_SIZE;

        for (int row = 0; row < rows; ++row)
        {
            for (int block_idx = 0; block_idx < num_blocks_per_row; ++block_idx)
            {
                const float *block_data = data + row * cols + block_idx * BLOCK_SIZE;
                IQ4_NLBlock &block = blocks[row * num_blocks_per_row + block_idx];

                float max_abs = 0.0f;
                for (int i = 0; i < BLOCK_SIZE; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(block_data[i]));
                }

                float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                block.scale = fp32_to_fp16(scale);

                for (int i = 0; i < BLOCK_SIZE; i += 2)
                {
                    float val0 = block_data[i] / scale;
                    float val1 = block_data[i + 1] / scale;

                    auto find_nearest = [](float val) -> uint8_t
                    {
                        int best_idx = 0;
                        float best_dist = std::abs(val - kvalues_iq4nl[0]);
                        for (int j = 1; j < 16; ++j)
                        {
                            float dist = std::abs(val - kvalues_iq4nl[j]);
                            if (dist < best_dist)
                            {
                                best_dist = dist;
                                best_idx = j;
                            }
                        }
                        return best_idx;
                    };

                    uint8_t nibble0 = find_nearest(val0);
                    uint8_t nibble1 = find_nearest(val1);
                    block.quants[i / 2] = (nibble1 << 4) | nibble0;
                }
            }
        }
    }

    struct BenchmarkResult
    {
        int M, N, K;
        double time_ms;
        double gflops;

        void print() const
        {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  M=" << std::setw(4) << M
                      << " N=" << std::setw(4) << N
                      << " K=" << std::setw(4) << K
                      << " | Time: " << std::setw(8) << time_ms << " ms"
                      << " | Throughput: " << std::setw(7) << gflops << " GFLOPS\n";
        }
    };

    BenchmarkResult benchmark_gemm(int M, int N, int K, int warmup_iters = 3, int bench_iters = 10)
    {
        // Generate random data
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> A_fp32(M * K);
        for (auto &val : A_fp32)
            val = dist(gen);

        std::vector<float> B_fp32(K * N);
        for (auto &val : B_fp32)
            val = dist(gen);

        // Step 1: Quantize B to IQ4_NL (simulates GGUF format)
        int num_blocks = K * (N / 32);
        std::vector<IQ4_NLBlock> B_iq4nl(num_blocks);
        quantize_to_iq4nl(B_fp32.data(), B_iq4nl.data(), K, N);

        // Step 2: Convert IQ4_NL → INT8 + scales (ONE-TIME MODEL LOAD OPERATION)
        std::vector<int8_t> B_int8(K * N);
        std::vector<float> scales_B(N);
        iq4nl_to_int8_with_scales(B_iq4nl.data(), B_int8.data(), scales_B.data(), K, N);

        // Prepare output
        std::vector<float> C_gpu(M * N);

        // Create kernel
        CudaGemmKernelPhase7_CUTLASS kernel;

        // Warmup
        for (int i = 0; i < warmup_iters; ++i)
        {
            kernel.execute(A_fp32.data(), B_int8.data(), scales_B.data(), C_gpu.data(), M, N, K);
        }

        // Benchmark (measures only GEMM, not conversion!)
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i)
        {
            kernel.execute(A_fp32.data(), B_int8.data(), scales_B.data(), C_gpu.data(), M, N, K);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double time_per_iter = total_ms / bench_iters;

        // Calculate GFLOPS: 2*M*N*K operations (multiply-add)
        double flops = 2.0 * M * N * K;
        double gflops = (flops / time_per_iter) / 1e6; // GFLOPS

        return BenchmarkResult{M, N, K, time_per_iter, gflops};
    }

} // namespace

TEST(Phase7CUTLASSPerf, SmallMatrix_64x64)
{
    std::cout << "\n=== Small Matrix (Single Token Decode) ===\n";
    auto result = benchmark_gemm(64, 64, 64);
    result.print();

    // Should be faster than Phase 6's 0.34 TFLOPS
    EXPECT_GT(result.gflops, 0.5) << "Expected >0.5 GFLOPS for small matrix";
}

TEST(Phase7CUTLASSPerf, MediumMatrix_512x512)
{
    std::cout << "\n=== Medium Matrix (Small Batch) ===\n";
    auto result = benchmark_gemm(512, 512, 512);
    result.print();

    // Should be much faster than Phase 5's baseline
    EXPECT_GT(result.gflops, 10.0) << "Expected >10 GFLOPS for medium matrix";
}

TEST(Phase7CUTLASSPerf, LargeMatrix_2048x2048)
{
    std::cout << "\n=== Large Matrix (Large Batch/Prefill) ===\n";
    auto result = benchmark_gemm(2048, 2048, 2048);
    result.print();

    // Target: 50-90 TFLOPS range (pure CUTLASS, no conversion overhead!)
    EXPECT_GT(result.gflops, 1000.0) << "Expected >1000 GFLOPS (>1 TFLOPS) for large matrix";
}

TEST(Phase7CUTLASSPerf, HugeMatrix_4096x4096)
{
    std::cout << "\n=== Huge Matrix (Stress Test) ===\n";
    auto result = benchmark_gemm(4096, 4096, 4096);
    result.print();

    // Should scale well to large sizes and hit 50-90 TFLOPS
    EXPECT_GT(result.gflops, 50000.0) << "Expected >50 TFLOPS for huge matrix with pre-converted weights";
}

TEST(Phase7CUTLASSPerf, DISABLED_ComprehensiveSweep)
{
    std::cout << "\n=== Comprehensive Matrix Size Sweep ===\n";

    std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};

    for (int size : sizes)
    {
        auto result = benchmark_gemm(size, size, size, 2, 5);
        result.print();
    }
}
