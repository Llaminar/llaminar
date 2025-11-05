/**
 * @file Test__Phase6_Int8_Functional.cpp
 * @brief Functional correctness tests for Phase 6 Int8 DP4A GEMM kernel
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "kernels/cuda/CudaGemmJITPhase6.h"
#include "kernels/cuda/CudaGemmConfigPhase6.h"

using namespace llaminar2::cuda;

// CPU reference implementation for IQ4_NL lookup
static const int8_t kvalues_iq4nl_cpu[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113};

struct IQ4_NLBlock
{
    uint8_t quants[16]; // 16 bytes = 32×4-bit indices
    __half scale;       // FP16 scale (use CUDA's __half type)
};

// Helper to convert __half to float on CPU
float half_to_float_cpu(const __half &h)
{
    // Use CUDA's conversion
    return __half2float(h);
}

// CPU FP32→int8 quantization
float quantize_fp32_to_int8_cpu(const float *input, int8_t *output, int count)
{
    float max_val = 0.0f;
    for (int i = 0; i < count; i++)
    {
        max_val = std::max(max_val, std::fabs(input[i]));
    }

    if (max_val == 0.0f)
    {
        for (int i = 0; i < count; i++)
            output[i] = 0;
        return 0.0f;
    }

    float scale = max_val / 127.0f;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < count; i++)
    {
        output[i] = static_cast<int8_t>(std::round(input[i] * inv_scale));
    }

    return scale;
}

// CPU reference GEMM with int8 DP4A simulation
void gemm_reference_int8(
    const float *A, const IQ4_NLBlock *B, float *C,
    int M, int N, int K)
{
    constexpr int BLOCK_SIZE = 32;
    const int K_BLOCKS = K / BLOCK_SIZE;

    for (int m = 0; m < M; m++)
    {
        // Quantize A row to int8
        std::vector<int8_t> a_q(K);
        float scale_a = quantize_fp32_to_int8_cpu(&A[m * K], a_q.data(), K);

        for (int n = 0; n < N; n++)
        {
            float acc = 0.0f;

            // Process in blocks of 32 (IQ4_NL block size)
            for (int kb = 0; kb < K_BLOCKS; kb++)
            {
                const IQ4_NLBlock &b_block = B[n * K_BLOCKS + kb];

                // FP16→FP32 scale conversion
                float scale_b = half_to_float_cpu(b_block.scale);

                int32_t block_acc = 0;

                // Process 32 elements in this block
                for (int k_in_block = 0; k_in_block < BLOCK_SIZE; k_in_block++)
                {
                    int k = kb * BLOCK_SIZE + k_in_block;

                    // Decode 4-bit index to int8 value
                    int byte_idx = k_in_block / 2;
                    int nibble_idx = k_in_block % 2;
                    uint8_t byte_val = b_block.quants[byte_idx];
                    uint8_t nibble = (nibble_idx == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
                    int8_t b_val = kvalues_iq4nl_cpu[nibble];

                    // Accumulate (int8 multiply)
                    int8_t a_val = a_q[k];
                    block_acc += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
                }

                // Scale contribution from this block
                // kvalues are normalized to [-127, 113], so divide by 127 to get [-1, 1] range
                acc += scale_a * (scale_b / 127.0f) * static_cast<float>(block_acc);
            }

            C[m * N + n] = acc;
        }
    }
}

// Test fixture
class Phase6Int8Functional : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Set up CUDA device
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "Testing on: " << prop.name << std::endl;
    }
};

TEST_F(Phase6Int8Functional, SmallMatrix4x4)
{
    const int M = 4, N = 4, K = 32; // K must be multiple of 32 (IQ4_NL block size)
    constexpr int BLOCK_SIZE = 32;
    const int K_BLOCKS = K / BLOCK_SIZE;

    // Initialize random A matrix (FP32)
    std::vector<float> A_host(M * K);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &val : A_host)
        val = dist(rng);

    // Initialize B matrix with IQ4_NL blocks
    std::vector<IQ4_NLBlock> B_host(N * K_BLOCKS);
    for (int n = 0; n < N; n++)
    {
        for (int kb = 0; kb < K_BLOCKS; kb++)
        {
            IQ4_NLBlock &block = B_host[n * K_BLOCKS + kb];

            // Random scale (FP32→FP16)
            float scale_fp32 = std::fabs(dist(rng));
            block.scale = __float2half(scale_fp32);

            // Random 4-bit indices
            for (int i = 0; i < 16; i++)
            {
                uint8_t idx0 = rng() % 16;
                uint8_t idx1 = rng() % 16;
                block.quants[i] = (idx1 << 4) | idx0;
            }
        }
    }

    // CPU reference
    std::vector<float> C_ref(M * N, 0.0f);
    gemm_reference_int8(A_host.data(), B_host.data(), C_ref.data(), M, N, K);

    // GPU computation
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K_BLOCKS * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A_host.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_host.data(), N * K_BLOCKS * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    // Compile and launch kernel
    auto config = get_default_phase6_config();
    auto kernel = CudaGemmJITPhase6::compile(config);

    dim3 grid((M + config.tile_m - 1) / config.tile_m,
              (N + config.tile_n - 1) / config.tile_n);
    dim3 block(config.threads_per_block);

    void *args[] = {&d_A, &d_B, &d_C, (void *)&M, (void *)&N, (void *)&K};
    cuLaunchKernel(kernel.function, grid.x, grid.y, 1, block.x, block.y, block.z, 0, 0, args, nullptr);

    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel error: " << cudaGetErrorString(err);

    // Copy result back
    std::vector<float> C_gpu(M * N);
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    float max_diff = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float diff = std::fabs(C_gpu[i] - C_ref[i]);
        float rel_err = (std::fabs(C_ref[i]) > 1e-6f) ? diff / std::fabs(C_ref[i]) : diff;
        max_diff = std::max(max_diff, diff);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Max relative error: " << max_rel_err << std::endl;

    // Tolerance for int8 quantization (expect ~1% error)
    EXPECT_LT(max_rel_err, 0.05f) << "Relative error too high";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Phase6Int8Functional, MediumMatrix64x64)
{
    const int M = 64, N = 64, K = 64; // K must be multiple of 32
    constexpr int BLOCK_SIZE = 32;
    const int K_BLOCKS = K / BLOCK_SIZE;

    // Initialize random matrices
    std::vector<float> A_host(M * K);
    std::vector<IQ4_NLBlock> B_host(N * K_BLOCKS);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &val : A_host)
        val = dist(rng);

    for (auto &block : B_host)
    {
        float scale = std::fabs(dist(rng));
        block.scale = __float2half(scale);

        for (int i = 0; i < 16; i++)
        {
            block.quants[i] = static_cast<uint8_t>(rng() % 256);
        }
    }

    // CPU reference
    std::vector<float> C_ref(M * N, 0.0f);
    gemm_reference_int8(A_host.data(), B_host.data(), C_ref.data(), M, N, K);

    // GPU computation
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K_BLOCKS * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A_host.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_host.data(), N * K_BLOCKS * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    auto config = get_default_phase6_config();
    auto kernel = CudaGemmJITPhase6::compile(config);

    dim3 grid((M + config.tile_m - 1) / config.tile_m,
              (N + config.tile_n - 1) / config.tile_n);
    dim3 block(config.threads_per_block);

    void *args[] = {&d_A, &d_B, &d_C, (void *)&M, (void *)&N, (void *)&K};
    cuLaunchKernel(kernel.function, grid.x, grid.y, 1, block.x, block.y, block.z, 0, 0, args, nullptr);

    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<float> C_gpu(M * N);
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float diff = std::fabs(C_gpu[i] - C_ref[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= (M * N);

    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Average absolute difference: " << avg_diff << std::endl;

    EXPECT_LT(avg_diff, 0.1f) << "Average error too high";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Phase6Int8Functional, IdentityPattern)
{
    // Test with simple identity-like pattern for easy verification
    const int M = 32, N = 32, K = 32;
    constexpr int BLOCK_SIZE = 32;
    const int K_BLOCKS = K / BLOCK_SIZE;

    // A = all ones
    std::vector<float> A_host(M * K, 1.0f);

    // B = simple pattern
    std::vector<IQ4_NLBlock> B_host(N * K_BLOCKS);
    for (auto &block : B_host)
    {
        block.scale = __float2half(1.0f); // FP16 value of 1.0

        // All quants point to index 8 (value = 1 in kvalues_iq4nl)
        for (int i = 0; i < 16; i++)
        {
            block.quants[i] = 0x88; // Both nibbles = 8
        }
    }

    // Expected output: each element should be ~32 (32 multiplies of 1×1)
    // With quantization, expect slight variation

    // CPU reference
    std::vector<float> C_ref(M * N, 0.0f);
    gemm_reference_int8(A_host.data(), B_host.data(), C_ref.data(), M, N, K);

    std::cout << "CPU reference C[0,0] = " << C_ref[0] << std::endl;
    std::cout << "Expected: ~32 (with quantization error)" << std::endl;

    // GPU computation
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K_BLOCKS * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A_host.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_host.data(), N * K_BLOCKS * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    auto config = get_default_phase6_config();
    auto kernel = CudaGemmJITPhase6::compile(config);

    dim3 grid(1, 1); // Single tile covers entire matrix
    dim3 block(config.threads_per_block);

    void *args[] = {&d_A, &d_B, &d_C, (void *)&M, (void *)&N, (void *)&K};
    cuLaunchKernel(kernel.function, grid.x, grid.y, 1, block.x, block.y, block.z, 0, 0, args, nullptr);

    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<float> C_gpu(M * N);
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "GPU result C[0,0] = " << C_gpu[0] << std::endl;

    // Compare
    for (int i = 0; i < M * N; i++)
    {
        float diff = std::fabs(C_gpu[i] - C_ref[i]);
        EXPECT_LT(diff, 1.0f) << "Mismatch at index " << i
                              << ": GPU=" << C_gpu[i]
                              << ", CPU=" << C_ref[i];
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
