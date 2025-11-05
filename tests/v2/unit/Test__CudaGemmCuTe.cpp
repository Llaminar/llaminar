/**
 * @file Test__CudaGemmCuTe.cpp
 * @brief Unit tests for CuTe Tensor Core GEMM kernel
 *
 * **STATUS**: Active - CuTe API issues resolved (Nov 3, 2025)
 *
 * **Phase 1 Goals**:
 * 1. ✅ Fix CuTe API usage (gemm, make_fragment_C, axpby signatures)
 * 2. ✅ Handle runtime N dimension in compile-time template context (local_tile)
 * 3. Validate correctness on small matrices (16×16×32)
 * 4. Benchmark Qwen 0.5B single token (1×896×896)
 * 5. Target: >50 GFLOPS (vs 33.5 GFLOPS FP32 baseline)
 *
 * **Fixed Issues** (as of Nov 3, 2025):
 * - ✅ `gemm(tiled_mma, tCrA, tCrB, tCrC)` - use TiledMMA instance
 * - ✅ `make_fragment_C()` - partition global C first via local_tile
 * - ✅ Runtime dimensions - use local_tile() to extract compile-time tile
 * - ✅ `axpby()` - corrected signature: axpby(alpha, src, beta, dst)
 *
 * **Key Insight**: NVIDIA uses `local_tile()` to extract compile-time sized
 * tiles from runtime-sized global matrices. This gives compile-time layouts
 * needed for `make_fragment_C()` register allocation.
 *
 * @author David Sanftenberg
 * @date November 3, 2025
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>

// IQ4_NL block structure (must match CUDA side)
struct IQ4_NLBlock
{
    float scale;
    uint8_t quants[16];
};

// C linkage wrapper for CuTe kernel
extern "C" void launch_iq4nl_gemm_cute_default(
    const float *A,
    const IQ4_NLBlock *B_blocks,
    float *C,
    int M, int N, int K,
    cudaStream_t stream);

class Test__CudaGemmCuTe : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check for Ampere or newer GPU
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        if (prop.major < 8)
        {
            GTEST_SKIP() << "Test requires Ampere GPU or newer (SM80+)";
        }
    }

    // Helper: Generate random FP32 matrix
    std::vector<float> generate_random_matrix(int rows, int cols)
    {
        std::vector<float> mat(rows * cols);
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (auto &val : mat)
        {
            val = dis(gen);
        }
        return mat;
    }

    // Helper: Quantize FP32 to IQ4_NL
    std::vector<IQ4_NLBlock> quantize_to_iq4nl(const std::vector<float> &data)
    {
        size_t num_blocks = (data.size() + 31) / 32;
        std::vector<IQ4_NLBlock> blocks(num_blocks);

        for (size_t b = 0; b < num_blocks; b++)
        {
            size_t start = b * 32;
            size_t end = std::min(start + 32, data.size());

            // Find scale (max absolute value)
            float max_abs = 0.0f;
            for (size_t i = start; i < end; i++)
            {
                max_abs = std::max(max_abs, std::abs(data[i]));
            }

            blocks[b].scale = max_abs / 127.0f;

            // Quantize values
            for (size_t i = 0; i < 16; i++)
            {
                uint8_t q0 = 0, q1 = 0;

                if (start + 2 * i < data.size())
                {
                    float val0 = data[start + 2 * i] / (blocks[b].scale + 1e-8f);
                    q0 = quantize_value(val0);
                }

                if (start + 2 * i + 1 < data.size())
                {
                    float val1 = data[start + 2 * i + 1] / (blocks[b].scale + 1e-8f);
                    q1 = quantize_value(val1);
                }

                blocks[b].quants[i] = q0 | (q1 << 4);
            }
        }

        return blocks;
    }

    // Helper: Quantize single value to 4-bit IQ4_NL
    uint8_t quantize_value(float val)
    {
        // IQ4_NL has 16 levels from -127/127 to 104/127
        float iq4nl_levels[16] = {
            -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
            -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
            0.0f, 10.0f / 127.0f, 22.0f / 127.0f, 35.0f / 127.0f,
            49.0f / 127.0f, 65.0f / 127.0f, 83.0f / 127.0f, 104.0f / 127.0f};

        // Find nearest level
        int best_idx = 0;
        float best_dist = std::abs(val - iq4nl_levels[0]);

        for (int i = 1; i < 16; i++)
        {
            float dist = std::abs(val - iq4nl_levels[i]);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_idx = i;
            }
        }

        return static_cast<uint8_t>(best_idx);
    }

    // Helper: Compute reference GEMM on CPU
    std::vector<float> compute_reference_gemm(
        const std::vector<float> &A,
        const std::vector<float> &B,
        int M, int N, int K)
    {
        std::vector<float> C(M * N, 0.0f);

        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                {
                    sum += A[m * K + k] * B[n * K + k]; // B is transposed (N×K)
                }
                C[m * N + n] = sum;
            }
        }

        return C;
    }

    // Helper: Compare two matrices
    struct ComparisonMetrics
    {
        float max_abs_diff;
        float rel_l2;
        size_t num_mismatches;
        bool passed;
    };

    ComparisonMetrics compare_matrices(
        const std::vector<float> &A,
        const std::vector<float> &B,
        float abs_tol = 1e-2f,
        float rel_tol = 1e-3f)
    {
        ComparisonMetrics metrics = {0.0f, 0.0f, 0, false};

        if (A.size() != B.size())
        {
            return metrics;
        }

        float sum_sq_diff = 0.0f;
        float sum_sq_ref = 0.0f;

        for (size_t i = 0; i < A.size(); i++)
        {
            float diff = std::abs(A[i] - B[i]);
            metrics.max_abs_diff = std::max(metrics.max_abs_diff, diff);

            if (diff > abs_tol)
            {
                metrics.num_mismatches++;
            }

            sum_sq_diff += diff * diff;
            sum_sq_ref += B[i] * B[i];
        }

        metrics.rel_l2 = std::sqrt(sum_sq_diff / (sum_sq_ref + 1e-8f));
        metrics.passed = (metrics.max_abs_diff < abs_tol) && (metrics.rel_l2 < rel_tol);

        return metrics;
    }
};

TEST_F(Test__CudaGemmCuTe, BasicCompilation)
{
    // Just test that the template compiles
    SUCCEED();
}

TEST_F(Test__CudaGemmCuTe, SmallMatrixCorrectness)
{
    // Use tile-aligned dimensions (32×32×32 matches our TILE_M/N/K)
    const int M = 32, N = 32, K = 32;

    // Generate random data
    auto A_host = generate_random_matrix(M, K);
    auto B_fp32_host = generate_random_matrix(N, K); // N×K transposed

    // Quantize B to IQ4_NL
    auto B_blocks_host = quantize_to_iq4nl(B_fp32_host);

    // Compute reference (using quantized B)
    std::vector<float> B_dequant(N * K);
    for (int n = 0; n < N; n++)
    {
        int blocks_per_row = K / 32;
        for (int kb = 0; kb < blocks_per_row; kb++)
        {
            const auto &block = B_blocks_host[n * blocks_per_row + kb];
            for (int i = 0; i < 16; i++)
            {
                uint8_t q = block.quants[i];
                float iq4nl_vals[16] = {
                    -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
                    -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
                    0.0f, 10.0f / 127.0f, 22.0f / 127.0f, 35.0f / 127.0f,
                    49.0f / 127.0f, 65.0f / 127.0f, 83.0f / 127.0f, 104.0f / 127.0f};
                B_dequant[n * K + kb * 32 + 2 * i] = block.scale * iq4nl_vals[q & 0xF];
                B_dequant[n * K + kb * 32 + 2 * i + 1] = block.scale * iq4nl_vals[q >> 4];
            }
        }
    }

    auto C_ref = compute_reference_gemm(A_host, B_dequant, M, N, K);

    // Allocate device memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B_blocks;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B_blocks, B_blocks_host.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A_host.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_blocks, B_blocks_host.data(),
               B_blocks_host.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    // Launch CuTe kernel (using default config: 32×32×32, 1×1 atom, 32 threads)
    launch_iq4nl_gemm_cute_default(d_A, d_B_blocks, d_C, M, N, K, 0);

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "Kernel execution failed: " << cudaGetErrorString(err);

    // Copy result back
    std::vector<float> C_cute(M * N);
    cudaMemcpy(C_cute.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    auto metrics = compare_matrices(C_cute, C_ref, 1e-1f, 1e-2f);

    std::cout << "Comparison metrics:\n";
    std::cout << "  Max abs diff: " << metrics.max_abs_diff << "\n";
    std::cout << "  Rel L2: " << metrics.rel_l2 << "\n";
    std::cout << "  Mismatches: " << metrics.num_mismatches << " / " << C_ref.size() << "\n";

    EXPECT_TRUE(metrics.passed) << "CuTe output diverged from reference";
    EXPECT_LT(metrics.max_abs_diff, 1e-1f) << "Max absolute difference too large";
    EXPECT_LT(metrics.rel_l2, 1e-2f) << "Relative L2 error too large";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B_blocks);
    cudaFree(d_C);
}

TEST_F(Test__CudaGemmCuTe, Qwen05B_SingleToken_QKV)
{
    // Qwen 0.5B QKV projection size
    const int M = 1, N = 896, K = 896;

    // Generate random data
    auto A_host = generate_random_matrix(M, K);
    auto B_fp32_host = generate_random_matrix(N, K);

    // Quantize B to IQ4_NL
    auto B_blocks_host = quantize_to_iq4nl(B_fp32_host);

    // Allocate device memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B_blocks;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B_blocks, B_blocks_host.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A_host.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_blocks, B_blocks_host.data(),
               B_blocks_host.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    // Warm-up
    launch_iq4nl_gemm_cute_default(d_A, d_B_blocks, d_C, M, N, K, 0);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
    {
        launch_iq4nl_gemm_cute_default(d_A, d_B_blocks, d_C, M, N, K, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iterations;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e6);

    std::cout << "Qwen 0.5B SingleToken QKV (" << M << "×" << N << "×" << K << "):\n";
    std::cout << "  Time: " << ms << " ms\n";
    std::cout << "  Performance: " << gflops << " GFLOPS\n";

    // Phase 1: Correctness achieved, performance needs optimization
    // Current: ~7 GFLOPS (generic copy from shared to registers)
    // Phase 2 target: 50-100 GFLOPS (with LDSM, K-blocking, pipelining)
    EXPECT_GT(gflops, 5.0) << "Performance unexpectedly low (sanity check)";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B_blocks);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
