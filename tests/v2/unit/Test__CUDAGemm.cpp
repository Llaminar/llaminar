/**
 * @file Test__CUDAGemm.cpp
 * @brief Test CUDA IQ4_NL quantized GEMM kernel correctness
 *
 * **Purpose**: Validate CUDA IQ4_NL GEMM implementation against CPU reference.
 *
 * **Tests**:
 * - Basic correctness: Small matrices with known values
 * - Size variations: Different m, n, k combinations
 * - Edge cases: Single row/column, invalid k (not multiple of 32)
 * - Numerical accuracy: Compare CUDA vs CPU with tolerance
 *
 * **Requirements**: Run with `-DHAVE_CUDA=ON`
 *
 * @author David Sanftenberg
 * @date October 31, 2025
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include "kernels/cuda/IQ4_NL_BlockDecoder.h"
#include "tensors/FP16Utils.h"
#include "utils/Logger.h"
#endif

using namespace llaminar2;

#ifdef HAVE_CUDA

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Create IQ4_NL quantized blocks from FP32 weights
 *
 * Simple quantization: Find min/max per block, map to 4-bit indices.
 * This is a reference implementation for testing, not optimized.
 */
std::vector<cuda::IQ4_NLBlock> quantizeToIQ4NL(const std::vector<float> &weights, int n, int k)
{
    const int num_blocks_per_row = k / 32;
    const int total_blocks = n * num_blocks_per_row;
    std::vector<cuda::IQ4_NLBlock> blocks(total_blocks);

    // IQ4_NL lookup table (from IQ4_NLTensor.h)
    const int8_t kvalues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113};

    for (int row = 0; row < n; ++row)
    {
        for (int kb = 0; kb < num_blocks_per_row; ++kb)
        {
            int block_idx = row * num_blocks_per_row + kb;
            int start = row * k + kb * 32;

            // Find min/max in this block
            float min_val = weights[start];
            float max_val = weights[start];
            for (int i = 1; i < 32; ++i)
            {
                min_val = std::min(min_val, weights[start + i]);
                max_val = std::max(max_val, weights[start + i]);
            }

            // Compute scale factor
            float scale = (max_val - min_val) / 240.0f; // Range of kvalues: 113 - (-127) = 240
            if (scale < 1e-8f)
                scale = 1e-8f; // Avoid division by zero

            // Convert to FP16 using proper IEEE 754 conversion
            uint16_t d_fp16 = fp32_to_fp16(scale);
            blocks[block_idx].d = d_fp16;

            // Quantize values to 4-bit indices
            for (int i = 0; i < 16; ++i)
            {
                float val_low = weights[start + i];
                float val_high = weights[start + i + 16];

                // Find closest kvalues index
                int idx_low = 0;
                int idx_high = 0;
                float best_err_low = std::abs(val_low - scale * kvalues[0]);
                float best_err_high = std::abs(val_high - scale * kvalues[0]);

                for (int q = 1; q < 16; ++q)
                {
                    float err_low = std::abs(val_low - scale * kvalues[q]);
                    float err_high = std::abs(val_high - scale * kvalues[q]);
                    if (err_low < best_err_low)
                    {
                        best_err_low = err_low;
                        idx_low = q;
                    }
                    if (err_high < best_err_high)
                    {
                        best_err_high = err_high;
                        idx_high = q;
                    }
                }

                // Pack into byte (low nibble = idx_low, high nibble = idx_high)
                blocks[block_idx].qs[i] = static_cast<uint8_t>((idx_high << 4) | idx_low);
            }
        }
    }

    return blocks;
}

/**
 * @brief CPU reference GEMM implementation: C = A * B
 *
 * Simple nested loop for ground truth comparison.
 */
void cpuGemm(const float *A, const float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
            {
                sum += A[i * k + p] * B[j * k + p]; // B in row-major [n × k]
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * @brief Dequantize IQ4_NL blocks to FP32 for CPU reference
 */
std::vector<float> dequantizeIQ4NL(const std::vector<cuda::IQ4_NLBlock> &blocks, int n, int k)
{
    const int8_t kvalues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113};

    std::vector<float> weights(n * k);
    const int num_blocks_per_row = k / 32;

    for (int row = 0; row < n; ++row)
    {
        for (int kb = 0; kb < num_blocks_per_row; ++kb)
        {
            int block_idx = row * num_blocks_per_row + kb;
            const auto &block = blocks[block_idx];

            // Convert FP16 scale to FP32 using proper IEEE 754 conversion
            float scale = fp16_to_fp32(block.d);

            // Decode 32 elements
            int out_idx = row * k + kb * 32;
            for (int i = 0; i < 16; ++i)
            {
                uint8_t qbyte = block.qs[i];
                uint8_t idx_low = qbyte & 0x0F;
                uint8_t idx_high = qbyte >> 4;

                weights[out_idx + i] = scale * static_cast<float>(kvalues[idx_low]);
                weights[out_idx + i + 16] = scale * static_cast<float>(kvalues[idx_high]);
            }
        }
    }

    return weights;
}

/**
 * @brief Compare two float arrays with tolerance
 */
bool compareFloatArrays(const float *a, const float *b, size_t count,
                        float abs_tol = 1e-3f, float rel_tol = 1e-2f)
{
    bool all_match = true;
    size_t mismatches = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;

    for (size_t i = 0; i < count; ++i)
    {
        float abs_diff = std::abs(a[i] - b[i]);
        float rel_diff = std::abs(a[i] - b[i]) / (std::abs(b[i]) + 1e-8f);

        max_abs_diff = std::max(max_abs_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        if (abs_diff > abs_tol && rel_diff > rel_tol)
        {
            if (mismatches < 5)
            { // Print first 5 mismatches
                std::cout << "  Mismatch at [" << i << "]: CUDA=" << a[i]
                          << " CPU=" << b[i] << " (abs_diff=" << abs_diff
                          << ", rel_diff=" << rel_diff << ")" << std::endl;
            }
            mismatches++;
            all_match = false;
        }
    }

    std::cout << "  Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "  Max relative difference: " << max_rel_diff << std::endl;
    if (mismatches > 0)
    {
        std::cout << "  Total mismatches: " << mismatches << "/" << count
                  << " (" << (100.0f * mismatches / count) << "%)" << std::endl;
    }

    return all_match;
}

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAGemm : public ::testing::Test
{
protected:
    void SetUp() override
    {
        backend = std::make_unique<CUDABackend>();
        device_count = backend->deviceCount();

        if (device_count == 0)
        {
            GTEST_SKIP() << "No CUDA devices available, skipping CUDA GEMM tests";
        }

        device_id = 0;
        std::cout << "Using CUDA device 0: " << backend->deviceName(0) << std::endl;
    }

    void TearDown() override
    {
        backend.reset();
    }

    std::unique_ptr<CUDABackend> backend;
    int device_count = 0;
    int device_id = 0;
};

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(Test__CUDAGemm, BasicCorrectness)
{
    // Small test: m=4, n=8, k=64
    const int m = 4;
    const int n = 8;
    const int k = 64;

    std::cout << "Testing GEMM: C[" << m << "×" << n << "] = A[" << m << "×" << k
              << "] * B[" << n << "×" << k << "]" << std::endl;

    // Create random input matrices
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(m * k);
    std::vector<float> B_fp32(n * k);

    for (auto &val : A)
        val = dist(gen);
    for (auto &val : B_fp32)
        val = dist(gen);

    // Quantize B to IQ4_NL
    auto B_blocks = quantizeToIQ4NL(B_fp32, n, k);

    // Dequantize for CPU reference
    auto B_dequant = dequantizeIQ4NL(B_blocks, n, k);

    // CPU reference GEMM
    std::vector<float> C_cpu(m * n, 0.0f);
    cpuGemm(A.data(), B_dequant.data(), C_cpu.data(), m, n, k);

    // Allocate GPU memory
    float *A_device = static_cast<float *>(backend->allocate(m * k * sizeof(float), device_id));
    cuda::IQ4_NLBlock *B_device = static_cast<cuda::IQ4_NLBlock *>(
        backend->allocate(B_blocks.size() * sizeof(cuda::IQ4_NLBlock), device_id));
    float *C_device = static_cast<float *>(backend->allocate(m * n * sizeof(float), device_id));

    ASSERT_NE(A_device, nullptr);
    ASSERT_NE(B_device, nullptr);
    ASSERT_NE(C_device, nullptr);

    // Transfer to GPU
    ASSERT_TRUE(backend->hostToDevice(A_device, A.data(), m * k * sizeof(float), device_id));
    ASSERT_TRUE(backend->hostToDevice(B_device, B_blocks.data(),
                                      B_blocks.size() * sizeof(cuda::IQ4_NLBlock), device_id));

    // Run CUDA GEMM
    bool success = backend->gemmIQ4NL(A_device, B_device, C_device, m, n, k, device_id);
    ASSERT_TRUE(success) << "CUDA GEMM failed";

    // Transfer result back
    std::vector<float> C_cuda(m * n);
    ASSERT_TRUE(backend->deviceToHost(C_cuda.data(), C_device, m * n * sizeof(float), device_id));

    // Compare results
    std::cout << "Comparing CUDA vs CPU results..." << std::endl;
    EXPECT_TRUE(compareFloatArrays(C_cuda.data(), C_cpu.data(), m * n, 1e-2f, 5e-2f));

    // Cleanup
    backend->free(A_device, device_id);
    backend->free(B_device, device_id);
    backend->free(C_device, device_id);
}

TEST_F(Test__CUDAGemm, SingleRow)
{
    // Edge case: m=1 (single row output)
    const int m = 1;
    const int n = 16;
    const int k = 128;

    std::cout << "Testing single row: C[1×" << n << "] = A[1×" << k
              << "] * B[" << n << "×" << k << "]" << std::endl;

    std::mt19937 gen(67890);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> A(m * k);
    std::vector<float> B_fp32(n * k);
    for (auto &val : A)
        val = dist(gen);
    for (auto &val : B_fp32)
        val = dist(gen);

    auto B_blocks = quantizeToIQ4NL(B_fp32, n, k);
    auto B_dequant = dequantizeIQ4NL(B_blocks, n, k);

    std::vector<float> C_cpu(m * n, 0.0f);
    cpuGemm(A.data(), B_dequant.data(), C_cpu.data(), m, n, k);

    // GPU execution
    float *A_device = static_cast<float *>(backend->allocate(m * k * sizeof(float), device_id));
    cuda::IQ4_NLBlock *B_device = static_cast<cuda::IQ4_NLBlock *>(
        backend->allocate(B_blocks.size() * sizeof(cuda::IQ4_NLBlock), device_id));
    float *C_device = static_cast<float *>(backend->allocate(m * n * sizeof(float), device_id));

    ASSERT_TRUE(backend->hostToDevice(A_device, A.data(), m * k * sizeof(float), device_id));
    ASSERT_TRUE(backend->hostToDevice(B_device, B_blocks.data(),
                                      B_blocks.size() * sizeof(cuda::IQ4_NLBlock), device_id));

    ASSERT_TRUE(backend->gemmIQ4NL(A_device, B_device, C_device, m, n, k, device_id));

    std::vector<float> C_cuda(m * n);
    ASSERT_TRUE(backend->deviceToHost(C_cuda.data(), C_device, m * n * sizeof(float), device_id));

    EXPECT_TRUE(compareFloatArrays(C_cuda.data(), C_cpu.data(), m * n, 1e-2f, 5e-2f));

    backend->free(A_device, device_id);
    backend->free(B_device, device_id);
    backend->free(C_device, device_id);
}

TEST_F(Test__CUDAGemm, SingleColumn)
{
    // Edge case: n=1 (single column output)
    const int m = 16;
    const int n = 1;
    const int k = 64;

    std::cout << "Testing single column: C[" << m << "×1] = A[" << m << "×" << k
              << "] * B[1×" << k << "]" << std::endl;

    std::mt19937 gen(11111);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> A(m * k);
    std::vector<float> B_fp32(n * k);
    for (auto &val : A)
        val = dist(gen);
    for (auto &val : B_fp32)
        val = dist(gen);

    auto B_blocks = quantizeToIQ4NL(B_fp32, n, k);
    auto B_dequant = dequantizeIQ4NL(B_blocks, n, k);

    std::vector<float> C_cpu(m * n, 0.0f);
    cpuGemm(A.data(), B_dequant.data(), C_cpu.data(), m, n, k);

    // GPU execution
    float *A_device = static_cast<float *>(backend->allocate(m * k * sizeof(float), device_id));
    cuda::IQ4_NLBlock *B_device = static_cast<cuda::IQ4_NLBlock *>(
        backend->allocate(B_blocks.size() * sizeof(cuda::IQ4_NLBlock), device_id));
    float *C_device = static_cast<float *>(backend->allocate(m * n * sizeof(float), device_id));

    ASSERT_TRUE(backend->hostToDevice(A_device, A.data(), m * k * sizeof(float), device_id));
    ASSERT_TRUE(backend->hostToDevice(B_device, B_blocks.data(),
                                      B_blocks.size() * sizeof(cuda::IQ4_NLBlock), device_id));

    ASSERT_TRUE(backend->gemmIQ4NL(A_device, B_device, C_device, m, n, k, device_id));

    std::vector<float> C_cuda(m * n);
    ASSERT_TRUE(backend->deviceToHost(C_cuda.data(), C_device, m * n * sizeof(float), device_id));

    EXPECT_TRUE(compareFloatArrays(C_cuda.data(), C_cpu.data(), m * n, 1e-2f, 5e-2f));

    backend->free(A_device, device_id);
    backend->free(B_device, device_id);
    backend->free(C_device, device_id);
}

TEST_F(Test__CUDAGemm, MediumMatrix)
{
    // Medium size: m=128, n=256, k=512
    const int m = 128;
    const int n = 256;
    const int k = 512;

    std::cout << "Testing medium matrix: C[" << m << "×" << n << "] = A[" << m << "×" << k
              << "] * B[" << n << "×" << k << "]" << std::endl;

    std::mt19937 gen(22222);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> A(m * k);
    std::vector<float> B_fp32(n * k);
    for (auto &val : A)
        val = dist(gen);
    for (auto &val : B_fp32)
        val = dist(gen);

    auto B_blocks = quantizeToIQ4NL(B_fp32, n, k);
    auto B_dequant = dequantizeIQ4NL(B_blocks, n, k);

    std::vector<float> C_cpu(m * n, 0.0f);
    cpuGemm(A.data(), B_dequant.data(), C_cpu.data(), m, n, k);

    // GPU execution
    float *A_device = static_cast<float *>(backend->allocate(m * k * sizeof(float), device_id));
    cuda::IQ4_NLBlock *B_device = static_cast<cuda::IQ4_NLBlock *>(
        backend->allocate(B_blocks.size() * sizeof(cuda::IQ4_NLBlock), device_id));
    float *C_device = static_cast<float *>(backend->allocate(m * n * sizeof(float), device_id));

    ASSERT_TRUE(backend->hostToDevice(A_device, A.data(), m * k * sizeof(float), device_id));
    ASSERT_TRUE(backend->hostToDevice(B_device, B_blocks.data(),
                                      B_blocks.size() * sizeof(cuda::IQ4_NLBlock), device_id));

    ASSERT_TRUE(backend->gemmIQ4NL(A_device, B_device, C_device, m, n, k, device_id));

    std::vector<float> C_cuda(m * n);
    ASSERT_TRUE(backend->deviceToHost(C_cuda.data(), C_device, m * n * sizeof(float), device_id));

    std::cout << "Comparing results (sampling first 100 elements)..." << std::endl;
    EXPECT_TRUE(compareFloatArrays(C_cuda.data(), C_cpu.data(), std::min(100, m * n), 1e-2f, 5e-2f));

    backend->free(A_device, device_id);
    backend->free(B_device, device_id);
    backend->free(C_device, device_id);
}

TEST_F(Test__CUDAGemm, InvalidKDimension)
{
    // k not multiple of 32 should fail gracefully
    const int m = 16;
    const int n = 16;
    const int k = 63; // Invalid! Not multiple of 32

    std::cout << "Testing invalid k=" << k << " (not multiple of 32)" << std::endl;

    std::vector<float> A(m * k, 1.0f);

    // Allocate minimal GPU memory (kernel should reject before computation)
    float *A_device = static_cast<float *>(backend->allocate(m * k * sizeof(float), device_id));
    void *B_device = backend->allocate(1024, device_id); // Dummy allocation
    float *C_device = static_cast<float *>(backend->allocate(m * n * sizeof(float), device_id));

    ASSERT_TRUE(backend->hostToDevice(A_device, A.data(), m * k * sizeof(float), device_id));

    // This should fail (k not multiple of 32)
    bool success = backend->gemmIQ4NL(A_device, B_device, C_device, m, n, k, device_id);
    EXPECT_FALSE(success) << "GEMM should reject k=" << k << " (not multiple of 32)";

    backend->free(A_device, device_id);
    backend->free(B_device, device_id);
    backend->free(C_device, device_id);
}

#endif // HAVE_CUDA

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
