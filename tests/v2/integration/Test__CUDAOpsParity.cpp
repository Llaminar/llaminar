/**
 * @file Test__CUDAOpsParity.cpp
 * @brief Parity tests for CUDA RMSNorm, SwiGLU, RoPE kernels vs CPU reference
 *
 * **Purpose**: Validate that CUDA kernels produce numerically equivalent
 * results to CPU kernels with high cosine similarity (>= 0.999).
 *
 * **Tests**:
 * - CUDARMSNormKernelT vs CPURMSNormKernelT (FP32, BF16, FP16)
 * - CUDASwiGLUKernelT vs CPUSwiGLUKernelT (FP32, BF16, FP16)
 * - CUDARoPEKernelT vs CPURoPEKernelT (FP32, BF16, FP16)
 *
 * **Pass Criteria**:
 * - Cosine similarity >= 0.999 (very high correlation)
 * - No NaN/Inf in outputs
 * - Relative error < 1% for FP32, < 2% for BF16/FP16
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "execution/RuntimeConfig.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include "kernels/cuda/ops/CUDARMSNormKernelT.h"
#include "kernels/cuda/ops/CUDASwiGLUKernelT.h"
#include "kernels/cuda/ops/CUDARoPEKernelT.h"
#include "kernels/cpu/ops/CPURMSNormKernelT.h"
#include "kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "kernels/cpu/ops/CPURoPEKernelT.h"
#include <cuda_runtime.h>
#endif

// Now include test utils
#include "../utils/CUDATestUtils.h"
#include "../utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <random>

using namespace llaminar2;
using namespace llaminar2::test::cuda;
using namespace llaminar2::test;

namespace
{

    // ============================================================================
    // Similarity Utilities (same as CUDAGemmParity)
    // ============================================================================

    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-12)
            return 0.0;
        return dot / denom;
    }

    double relativeL2Error(const float *actual, const float *expected, size_t count)
    {
        double diff_norm = 0.0, expected_norm = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double diff = actual[i] - expected[i];
            diff_norm += diff * diff;
            expected_norm += static_cast<double>(expected[i]) * expected[i];
        }
        if (expected_norm < 1e-12)
            return diff_norm > 1e-12 ? 1e9 : 0.0;
        return std::sqrt(diff_norm / expected_norm);
    }

    // hasNaNOrInf is provided by CUDATestUtils.h

    // BF16 conversion helpers (host-side)
    inline uint16_t floatToBF16(float f)
    {
        uint32_t bits;
        memcpy(&bits, &f, sizeof(float));
        return static_cast<uint16_t>(bits >> 16);
    }

    inline float bf16ToFloat(uint16_t bf16)
    {
        uint32_t bits = static_cast<uint32_t>(bf16) << 16;
        float result;
        memcpy(&result, &bits, sizeof(float));
        return result;
    }

    // FP16 conversion helpers (simple implementation)
    inline uint16_t floatToFP16(float f)
    {
        uint32_t bits;
        memcpy(&bits, &f, sizeof(float));
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (bits >> 13) & 0x3FF;

        if (exp <= 0)
            return sign;
        if (exp >= 31)
            return sign | 0x7C00;
        return static_cast<uint16_t>(sign | (exp << 10) | mant);
    }

    inline float fp16ToFloat(uint16_t fp16)
    {
        uint32_t sign = (fp16 & 0x8000) << 16;
        int32_t exp = (fp16 >> 10) & 0x1F;
        uint32_t mant = fp16 & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0)
            {
                uint32_t bits = sign;
                float result;
                memcpy(&result, &bits, sizeof(float));
                return result;
            }
            // Denormalized
            exp = 1;
            while (!(mant & 0x400))
            {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
        }
        else if (exp == 31)
        {
            uint32_t bits = sign | 0x7F800000 | (mant << 13);
            float result;
            memcpy(&result, &bits, sizeof(float));
            return result;
        }

        uint32_t bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        float result;
        memcpy(&result, &bits, sizeof(float));
        return result;
    }

    // Dequantize BF16 array to FP32
    void dequantizeBF16(const uint16_t *src, float *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = bf16ToFloat(src[i]);
        }
    }

    // Dequantize FP16 array to FP32
    void dequantizeFP16(const uint16_t *src, float *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp16ToFloat(src[i]);
        }
    }

    // Quantize FP32 array to BF16
    void quantizeToBF16(const float *src, uint16_t *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = floatToBF16(src[i]);
        }
    }

    // Quantize FP32 array to FP16
    void quantizeToFP16(const float *src, uint16_t *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = floatToFP16(src[i]);
        }
    }

} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAOpsParity : public CUDATestBase
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

    std::vector<float> randomFP32(size_t count)
    {
        std::vector<float> data(count);
        for (auto &val : data)
        {
            val = dist_(rng_);
        }
        return data;
    }

    // Create gamma weights (always positive for RMSNorm)
    std::vector<float> randomGamma(size_t count)
    {
        std::uniform_real_distribution<float> gamma_dist{0.5f, 1.5f};
        std::vector<float> data(count);
        for (auto &val : data)
        {
            val = gamma_dist(rng_);
        }
        return data;
    }
};

#ifdef HAVE_CUDA

// ============================================================================
// RMSNorm Parity Tests
// ============================================================================

TEST_F(Test__CUDAOpsParity, RMSNorm_FP32_Small)
{
    SKIP_IF_NO_CUDA();

    constexpr int rows = 4;
    constexpr int cols = 64;
    constexpr float epsilon = 1e-6f;
    const size_t total = rows * cols;

    // Create test data
    auto input_data = randomFP32(total);
    auto gamma_data = randomGamma(cols);
    std::vector<float> cpu_output(total, 0.0f);
    std::vector<float> cuda_output(total, 0.0f);

    // CPU reference
    CPURMSNormKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply(input_data.data(), gamma_data.data(), cpu_output.data(),
                     rows, cols, epsilon, false, nullptr, -1);

    // CUDA kernel
    llaminar2::cuda::CUDARMSNormKernelT<ActivationPrecision::FP32> cuda_kernel;

    // Allocate device memory
    float *d_input, *d_gamma, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, input_data.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma_data.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    // Execute
    ASSERT_TRUE(cuda_kernel.apply_typed(d_input, d_gamma, d_output, rows, cols, epsilon, 0));
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(cuda_output.data(), d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_output);

    // Validate
    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), total)) << "CUDA output contains NaN/Inf";

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), total);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), total);

    std::cout << "  RMSNorm FP32 Small: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.9999) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.01) << "L2 error too high";
}

TEST_F(Test__CUDAOpsParity, RMSNorm_FP32_Large)
{
    SKIP_IF_NO_CUDA();

    constexpr int rows = 32;
    constexpr int cols = 896; // Qwen2-0.5B hidden_dim
    constexpr float epsilon = 1e-6f;
    const size_t total = rows * cols;

    auto input_data = randomFP32(total);
    auto gamma_data = randomGamma(cols);
    std::vector<float> cpu_output(total, 0.0f);
    std::vector<float> cuda_output(total, 0.0f);

    CPURMSNormKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply(input_data.data(), gamma_data.data(), cpu_output.data(),
                     rows, cols, epsilon, false, nullptr, -1);

    llaminar2::cuda::CUDARMSNormKernelT<ActivationPrecision::FP32> cuda_kernel;

    float *d_input, *d_gamma, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, input_data.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma_data.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel.apply_typed(d_input, d_gamma, d_output, rows, cols, epsilon, 0));
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_output.data(), d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), total));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), total);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), total);

    std::cout << "  RMSNorm FP32 Large: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.9999);
    EXPECT_LE(l2_error, 0.01);
}

// ============================================================================
// SwiGLU Parity Tests
// ============================================================================

TEST_F(Test__CUDAOpsParity, SwiGLU_FP32_Small)
{
    SKIP_IF_NO_CUDA();

    constexpr int rows = 4;
    constexpr int cols = 64;
    const size_t total = rows * cols;

    auto gate_data = randomFP32(total);
    auto up_data = randomFP32(total);
    std::vector<float> cpu_output(total, 0.0f);
    std::vector<float> cuda_output(total, 0.0f);

    // CPU reference
    CPUSwiGLUKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply(gate_data.data(), up_data.data(), cpu_output.data(),
                     rows, cols, false, nullptr, -1);

    // CUDA kernel
    llaminar2::cuda::CUDASwiGLUKernelT<ActivationPrecision::FP32> cuda_kernel;

    float *d_gate, *d_up, *d_output;
    cudaMalloc(&d_gate, total * sizeof(float));
    cudaMalloc(&d_up, total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_gate, gate_data.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up_data.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_output.data(), d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), total));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), total);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), total);

    std::cout << "  SwiGLU FP32 Small: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.9999);
    EXPECT_LE(l2_error, 0.01);
}

TEST_F(Test__CUDAOpsParity, SwiGLU_FP32_Large)
{
    SKIP_IF_NO_CUDA();

    constexpr int rows = 32;
    constexpr int cols = 4864; // Qwen2-0.5B intermediate_dim
    const size_t total = rows * cols;

    auto gate_data = randomFP32(total);
    auto up_data = randomFP32(total);
    std::vector<float> cpu_output(total, 0.0f);
    std::vector<float> cuda_output(total, 0.0f);

    CPUSwiGLUKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply(gate_data.data(), up_data.data(), cpu_output.data(),
                     rows, cols, false, nullptr, -1);

    llaminar2::cuda::CUDASwiGLUKernelT<ActivationPrecision::FP32> cuda_kernel;

    float *d_gate, *d_up, *d_output;
    cudaMalloc(&d_gate, total * sizeof(float));
    cudaMalloc(&d_up, total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_gate, gate_data.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up_data.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_output.data(), d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), total));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), total);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), total);

    std::cout << "  SwiGLU FP32 Large: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.9999);
    EXPECT_LE(l2_error, 0.01);
}

TEST_F(Test__CUDAOpsParity, SwiGLU_BF16_Small)
{
    SKIP_IF_NO_CUDA();

    constexpr int rows = 4;
    constexpr int cols = 64;
    const size_t total = rows * cols;

    // Generate FP32 data and convert to BF16
    auto gate_data_fp32 = randomFP32(total);
    auto up_data_fp32 = randomFP32(total);

    std::vector<uint16_t> gate_bf16(total);
    std::vector<uint16_t> up_bf16(total);
    quantizeToBF16(gate_data_fp32.data(), gate_bf16.data(), total);
    quantizeToBF16(up_data_fp32.data(), up_bf16.data(), total);

    // CPU reference (BF16)
    std::vector<uint16_t> cpu_output_bf16(total, 0);
    CPUSwiGLUKernelT<ActivationPrecision::BF16> cpu_kernel;
    cpu_kernel.apply_bf16(gate_bf16.data(), up_bf16.data(), cpu_output_bf16.data(),
                          rows, cols, false, nullptr, -1);

    // CUDA kernel
    llaminar2::cuda::CUDASwiGLUKernelT<ActivationPrecision::BF16> cuda_kernel;

    uint16_t *d_gate, *d_up, *d_output;
    cudaMalloc(&d_gate, total * sizeof(uint16_t));
    cudaMalloc(&d_up, total * sizeof(uint16_t));
    cudaMalloc(&d_output, total * sizeof(uint16_t));

    cudaMemcpy(d_gate, gate_bf16.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up_bf16.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    cudaDeviceSynchronize();

    std::vector<uint16_t> cuda_output_bf16(total);
    cudaMemcpy(cuda_output_bf16.data(), d_output, total * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_output);

    // Convert outputs to FP32 for comparison
    std::vector<float> cpu_output_fp32(total);
    std::vector<float> cuda_output_fp32(total);
    dequantizeBF16(cpu_output_bf16.data(), cpu_output_fp32.data(), total);
    dequantizeBF16(cuda_output_bf16.data(), cuda_output_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(cuda_output_fp32.data(), total));

    double cosine = cosineSimilarity(cuda_output_fp32.data(), cpu_output_fp32.data(), total);
    double l2_error = relativeL2Error(cuda_output_fp32.data(), cpu_output_fp32.data(), total);

    std::cout << "  SwiGLU BF16 Small: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999);  // Slightly relaxed for BF16
    EXPECT_LE(l2_error, 0.02); // 2% tolerance for BF16
}

TEST_F(Test__CUDAOpsParity, SwiGLU_BF16_Large)
{
    SKIP_IF_NO_CUDA();

    constexpr int rows = 32;
    constexpr int cols = 4864;
    const size_t total = rows * cols;

    auto gate_data_fp32 = randomFP32(total);
    auto up_data_fp32 = randomFP32(total);

    std::vector<uint16_t> gate_bf16(total);
    std::vector<uint16_t> up_bf16(total);
    quantizeToBF16(gate_data_fp32.data(), gate_bf16.data(), total);
    quantizeToBF16(up_data_fp32.data(), up_bf16.data(), total);

    std::vector<uint16_t> cpu_output_bf16(total, 0);
    CPUSwiGLUKernelT<ActivationPrecision::BF16> cpu_kernel;
    cpu_kernel.apply_bf16(gate_bf16.data(), up_bf16.data(), cpu_output_bf16.data(),
                          rows, cols, false, nullptr, -1);

    llaminar2::cuda::CUDASwiGLUKernelT<ActivationPrecision::BF16> cuda_kernel;

    uint16_t *d_gate, *d_up, *d_output;
    cudaMalloc(&d_gate, total * sizeof(uint16_t));
    cudaMalloc(&d_up, total * sizeof(uint16_t));
    cudaMalloc(&d_output, total * sizeof(uint16_t));

    cudaMemcpy(d_gate, gate_bf16.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up_bf16.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    cudaDeviceSynchronize();

    std::vector<uint16_t> cuda_output_bf16(total);
    cudaMemcpy(cuda_output_bf16.data(), d_output, total * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_output);

    std::vector<float> cpu_output_fp32(total);
    std::vector<float> cuda_output_fp32(total);
    dequantizeBF16(cpu_output_bf16.data(), cpu_output_fp32.data(), total);
    dequantizeBF16(cuda_output_bf16.data(), cuda_output_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(cuda_output_fp32.data(), total));

    double cosine = cosineSimilarity(cuda_output_fp32.data(), cpu_output_fp32.data(), total);
    double l2_error = relativeL2Error(cuda_output_fp32.data(), cpu_output_fp32.data(), total);

    std::cout << "  SwiGLU BF16 Large: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999);
    EXPECT_LE(l2_error, 0.02);
}

TEST_F(Test__CUDAOpsParity, SwiGLU_FP16_Small)
{
    SKIP_IF_NO_CUDA();

    constexpr int rows = 4;
    constexpr int cols = 64;
    const size_t total = rows * cols;

    // Generate FP32 data and convert to FP16
    auto gate_data_fp32 = randomFP32(total);
    auto up_data_fp32 = randomFP32(total);

    std::vector<uint16_t> gate_fp16(total);
    std::vector<uint16_t> up_fp16(total);
    quantizeToFP16(gate_data_fp32.data(), gate_fp16.data(), total);
    quantizeToFP16(up_data_fp32.data(), up_fp16.data(), total);

    // CPU reference (FP16)
    std::vector<uint16_t> cpu_output_fp16(total, 0);
    CPUSwiGLUKernelT<ActivationPrecision::FP16> cpu_kernel;
    cpu_kernel.apply_fp16(gate_fp16.data(), up_fp16.data(), cpu_output_fp16.data(),
                          rows, cols, false, nullptr, -1);

    // CUDA kernel
    llaminar2::cuda::CUDASwiGLUKernelT<ActivationPrecision::FP16> cuda_kernel;

    uint16_t *d_gate, *d_up, *d_output;
    cudaMalloc(&d_gate, total * sizeof(uint16_t));
    cudaMalloc(&d_up, total * sizeof(uint16_t));
    cudaMalloc(&d_output, total * sizeof(uint16_t));

    cudaMemcpy(d_gate, gate_fp16.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up_fp16.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    cudaDeviceSynchronize();

    std::vector<uint16_t> cuda_output_fp16(total);
    cudaMemcpy(cuda_output_fp16.data(), d_output, total * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_output);

    // Convert outputs to FP32 for comparison
    std::vector<float> cpu_output_fp32(total);
    std::vector<float> cuda_output_fp32(total);
    dequantizeFP16(cpu_output_fp16.data(), cpu_output_fp32.data(), total);
    dequantizeFP16(cuda_output_fp16.data(), cuda_output_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(cuda_output_fp32.data(), total));

    double cosine = cosineSimilarity(cuda_output_fp32.data(), cpu_output_fp32.data(), total);
    double l2_error = relativeL2Error(cuda_output_fp32.data(), cpu_output_fp32.data(), total);

    std::cout << "  SwiGLU FP16 Small: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999);  // Slightly relaxed for FP16
    EXPECT_LE(l2_error, 0.02); // 2% tolerance for FP16
}

TEST_F(Test__CUDAOpsParity, SwiGLU_FP16_Large)
{
    SKIP_IF_NO_CUDA();

    constexpr int rows = 32;
    constexpr int cols = 4864;
    const size_t total = rows * cols;

    auto gate_data_fp32 = randomFP32(total);
    auto up_data_fp32 = randomFP32(total);

    std::vector<uint16_t> gate_fp16(total);
    std::vector<uint16_t> up_fp16(total);
    quantizeToFP16(gate_data_fp32.data(), gate_fp16.data(), total);
    quantizeToFP16(up_data_fp32.data(), up_fp16.data(), total);

    std::vector<uint16_t> cpu_output_fp16(total, 0);
    CPUSwiGLUKernelT<ActivationPrecision::FP16> cpu_kernel;
    cpu_kernel.apply_fp16(gate_fp16.data(), up_fp16.data(), cpu_output_fp16.data(),
                          rows, cols, false, nullptr, -1);

    llaminar2::cuda::CUDASwiGLUKernelT<ActivationPrecision::FP16> cuda_kernel;

    uint16_t *d_gate, *d_up, *d_output;
    cudaMalloc(&d_gate, total * sizeof(uint16_t));
    cudaMalloc(&d_up, total * sizeof(uint16_t));
    cudaMalloc(&d_output, total * sizeof(uint16_t));

    cudaMemcpy(d_gate, gate_fp16.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up_fp16.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    cudaDeviceSynchronize();

    std::vector<uint16_t> cuda_output_fp16(total);
    cudaMemcpy(cuda_output_fp16.data(), d_output, total * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_output);

    std::vector<float> cpu_output_fp32(total);
    std::vector<float> cuda_output_fp32(total);
    dequantizeFP16(cpu_output_fp16.data(), cpu_output_fp32.data(), total);
    dequantizeFP16(cuda_output_fp16.data(), cuda_output_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(cuda_output_fp32.data(), total));

    double cosine = cosineSimilarity(cuda_output_fp32.data(), cpu_output_fp32.data(), total);
    double l2_error = relativeL2Error(cuda_output_fp32.data(), cpu_output_fp32.data(), total);

    std::cout << "  SwiGLU FP16 Large: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999);
    EXPECT_LE(l2_error, 0.02);
}

// ============================================================================
// RoPE Parity Tests
// ============================================================================

TEST_F(Test__CUDAOpsParity, RoPE_FP32_Small)
{
    SKIP_IF_NO_CUDA();

    constexpr int seq_len = 4;
    constexpr int n_heads = 14;
    constexpr int head_dim = 64;
    constexpr float rope_theta = 10000.0f;
    const size_t total = seq_len * n_heads * head_dim;

    auto q_data = randomFP32(total);
    auto k_data = randomFP32(total);

    // Position IDs: [0, 1, 2, 3]
    std::vector<int> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
        position_ids[i] = i;

    // Make copies for CPU and CUDA
    std::vector<float> cpu_q = q_data;
    std::vector<float> cpu_k = k_data;
    std::vector<float> cuda_q = q_data;
    std::vector<float> cuda_k = k_data;

    // CPU reference (in-place)
    CPURoPEKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply_typed(cpu_q.data(), cpu_k.data(), position_ids.data(),
                           seq_len, n_heads, n_heads, head_dim, rope_theta, -1);

    // CUDA kernel
    llaminar2::cuda::CUDARoPEKernelT<ActivationPrecision::FP32> cuda_kernel;

    float *d_q, *d_k;
    int *d_pos_ids;
    cudaMalloc(&d_q, total * sizeof(float));
    cudaMalloc(&d_k, total * sizeof(float));
    cudaMalloc(&d_pos_ids, seq_len * sizeof(int));

    cudaMemcpy(d_q, cuda_q.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, cuda_k.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_ids, position_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel.apply_typed(d_q, d_k, d_pos_ids, seq_len, n_heads, n_heads,
                                        head_dim, rope_theta, 0));
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_q.data(), d_q, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_k.data(), d_k, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_pos_ids);

    // Validate Q
    ASSERT_FALSE(hasNaNOrInf(cuda_q.data(), total)) << "CUDA Q output contains NaN/Inf";
    double cosine_q = cosineSimilarity(cuda_q.data(), cpu_q.data(), total);
    double l2_error_q = relativeL2Error(cuda_q.data(), cpu_q.data(), total);

    // Validate K
    ASSERT_FALSE(hasNaNOrInf(cuda_k.data(), total)) << "CUDA K output contains NaN/Inf";
    double cosine_k = cosineSimilarity(cuda_k.data(), cpu_k.data(), total);
    double l2_error_k = relativeL2Error(cuda_k.data(), cpu_k.data(), total);

    std::cout << "  RoPE FP32 Small Q: cosine=" << cosine_q << ", L2_error=" << l2_error_q << std::endl;
    std::cout << "  RoPE FP32 Small K: cosine=" << cosine_k << ", L2_error=" << l2_error_k << std::endl;

    EXPECT_GE(cosine_q, 0.9999);
    EXPECT_GE(cosine_k, 0.9999);
    EXPECT_LE(l2_error_q, 0.01);
    EXPECT_LE(l2_error_k, 0.01);
}

TEST_F(Test__CUDAOpsParity, RoPE_FP32_Large)
{
    SKIP_IF_NO_CUDA();

    constexpr int seq_len = 128;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr float rope_theta = 10000.0f;

    const size_t q_total = seq_len * n_heads * head_dim;
    const size_t k_total = seq_len * n_kv_heads * head_dim;

    auto q_data = randomFP32(q_total);
    auto k_data = randomFP32(k_total);

    std::vector<int> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
        position_ids[i] = i;

    std::vector<float> cpu_q = q_data;
    std::vector<float> cpu_k = k_data;
    std::vector<float> cuda_q = q_data;
    std::vector<float> cuda_k = k_data;

    CPURoPEKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply_typed(cpu_q.data(), cpu_k.data(), position_ids.data(),
                           seq_len, n_heads, n_kv_heads, head_dim, rope_theta, -1);

    llaminar2::cuda::CUDARoPEKernelT<ActivationPrecision::FP32> cuda_kernel;

    float *d_q, *d_k;
    int *d_pos_ids;
    cudaMalloc(&d_q, q_total * sizeof(float));
    cudaMalloc(&d_k, k_total * sizeof(float));
    cudaMalloc(&d_pos_ids, seq_len * sizeof(int));

    cudaMemcpy(d_q, cuda_q.data(), q_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, cuda_k.data(), k_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_ids, position_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel.apply_typed(d_q, d_k, d_pos_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, 0));
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_q.data(), d_q, q_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_k.data(), d_k, k_total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_pos_ids);

    ASSERT_FALSE(hasNaNOrInf(cuda_q.data(), q_total));
    ASSERT_FALSE(hasNaNOrInf(cuda_k.data(), k_total));

    double cosine_q = cosineSimilarity(cuda_q.data(), cpu_q.data(), q_total);
    double l2_error_q = relativeL2Error(cuda_q.data(), cpu_q.data(), q_total);
    double cosine_k = cosineSimilarity(cuda_k.data(), cpu_k.data(), k_total);
    double l2_error_k = relativeL2Error(cuda_k.data(), cpu_k.data(), k_total);

    std::cout << "  RoPE FP32 Large Q: cosine=" << cosine_q << ", L2_error=" << l2_error_q << std::endl;
    std::cout << "  RoPE FP32 Large K: cosine=" << cosine_k << ", L2_error=" << l2_error_k << std::endl;

    EXPECT_GE(cosine_q, 0.9999);
    EXPECT_GE(cosine_k, 0.9999);
    EXPECT_LE(l2_error_q, 0.01);
    EXPECT_LE(l2_error_k, 0.01);
}

// ============================================================================
// Residual Add Parity Tests
// ============================================================================

// Forward declarations for CUDA kernels
extern "C"
{
    bool cudaOps_residual_add_fp32(const float *input, const float *residual, float *output, int size, int device_idx);
    bool cudaOps_residual_add_bf16(const uint16_t *input, const uint16_t *residual, uint16_t *output, int size, int device_idx);
    bool cudaOps_residual_add_fp16(const uint16_t *input, const uint16_t *residual, uint16_t *output, int size, int device_idx);
}

TEST_F(Test__CUDAOpsParity, ResidualAdd_FP32_Small)
{
    SKIP_IF_NO_CUDA();

    constexpr int size = 256;

    // Create test data
    auto input_data = randomFP32(size);
    auto residual_data = randomFP32(size);
    std::vector<float> cpu_output(size, 0.0f);
    std::vector<float> cuda_output(size, 0.0f);

    // CPU reference: simple element-wise addition
    for (int i = 0; i < size; ++i)
    {
        cpu_output[i] = input_data[i] + residual_data[i];
    }

    // CUDA: Allocate device memory and run kernel
    float *d_input = nullptr, *d_residual = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_residual, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, residual_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cudaOps_residual_add_fp32(d_input, d_residual, d_output, size, 0));
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_output);

    // Verify results
    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), size);

    std::cout << "  Residual Add FP32 Small: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.99999); // Should be nearly exact for simple addition
    EXPECT_LE(l2_error, 0.0001);
}

TEST_F(Test__CUDAOpsParity, ResidualAdd_FP32_Large)
{
    SKIP_IF_NO_CUDA();

    constexpr int size = 896 * 128; // Typical hidden_dim * seq_len

    // Create test data
    auto input_data = randomFP32(size);
    auto residual_data = randomFP32(size);
    std::vector<float> cpu_output(size, 0.0f);
    std::vector<float> cuda_output(size, 0.0f);

    // CPU reference
    for (int i = 0; i < size; ++i)
    {
        cpu_output[i] = input_data[i] + residual_data[i];
    }

    // CUDA
    float *d_input = nullptr, *d_residual = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_residual, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, residual_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cudaOps_residual_add_fp32(d_input, d_residual, d_output, size, 0));
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), size);

    std::cout << "  Residual Add FP32 Large: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.99999);
    EXPECT_LE(l2_error, 0.0001);
}

TEST_F(Test__CUDAOpsParity, ResidualAdd_BF16)
{
    SKIP_IF_NO_CUDA();

    constexpr int size = 896 * 32;

    // Create FP32 test data
    auto input_fp32 = randomFP32(size);
    auto residual_fp32 = randomFP32(size);

    // Convert to BF16
    std::vector<uint16_t> input_bf16(size);
    std::vector<uint16_t> residual_bf16(size);
    quantizeToBF16(input_fp32.data(), input_bf16.data(), size);
    quantizeToBF16(residual_fp32.data(), residual_bf16.data(), size);

    // CPU reference: BF16 -> FP32 -> add -> BF16 -> FP32 for comparison
    std::vector<float> cpu_output_fp32(size, 0.0f);
    for (int i = 0; i < size; ++i)
    {
        float in_f = bf16ToFloat(input_bf16[i]);
        float res_f = bf16ToFloat(residual_bf16[i]);
        uint16_t result_bf16 = floatToBF16(in_f + res_f);
        cpu_output_fp32[i] = bf16ToFloat(result_bf16);
    }

    // CUDA
    uint16_t *d_input = nullptr, *d_residual = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, size * sizeof(uint16_t));
    cudaMalloc(&d_residual, size * sizeof(uint16_t));
    cudaMalloc(&d_output, size * sizeof(uint16_t));

    cudaMemcpy(d_input, input_bf16.data(), size * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, residual_bf16.data(), size * sizeof(uint16_t), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cudaOps_residual_add_bf16(d_input, d_residual, d_output, size, 0));
    cudaDeviceSynchronize();

    std::vector<uint16_t> cuda_output_bf16(size);
    cudaMemcpy(cuda_output_bf16.data(), d_output, size * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_output);

    // Convert CUDA output to FP32 for comparison
    std::vector<float> cuda_output_fp32(size);
    dequantizeBF16(cuda_output_bf16.data(), cuda_output_fp32.data(), size);

    ASSERT_FALSE(hasNaNOrInf(cuda_output_fp32.data(), size));

    double cosine = cosineSimilarity(cuda_output_fp32.data(), cpu_output_fp32.data(), size);
    double l2_error = relativeL2Error(cuda_output_fp32.data(), cpu_output_fp32.data(), size);

    std::cout << "  Residual Add BF16: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999); // BF16 has lower precision
    EXPECT_LE(l2_error, 0.02);
}

TEST_F(Test__CUDAOpsParity, ResidualAdd_FP16)
{
    SKIP_IF_NO_CUDA();

    constexpr int size = 896 * 32;

    // Create FP32 test data
    auto input_fp32 = randomFP32(size);
    auto residual_fp32 = randomFP32(size);

    // Convert to FP16
    std::vector<uint16_t> input_fp16(size);
    std::vector<uint16_t> residual_fp16(size);
    quantizeToFP16(input_fp32.data(), input_fp16.data(), size);
    quantizeToFP16(residual_fp32.data(), residual_fp16.data(), size);

    // CPU reference: FP16 -> FP32 -> add -> FP16 -> FP32 for comparison
    std::vector<float> cpu_output_fp32(size, 0.0f);
    for (int i = 0; i < size; ++i)
    {
        float in_f = fp16ToFloat(input_fp16[i]);
        float res_f = fp16ToFloat(residual_fp16[i]);
        uint16_t result_fp16 = floatToFP16(in_f + res_f);
        cpu_output_fp32[i] = fp16ToFloat(result_fp16);
    }

    // CUDA
    uint16_t *d_input = nullptr, *d_residual = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, size * sizeof(uint16_t));
    cudaMalloc(&d_residual, size * sizeof(uint16_t));
    cudaMalloc(&d_output, size * sizeof(uint16_t));

    cudaMemcpy(d_input, input_fp16.data(), size * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, residual_fp16.data(), size * sizeof(uint16_t), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cudaOps_residual_add_fp16(d_input, d_residual, d_output, size, 0));
    cudaDeviceSynchronize();

    std::vector<uint16_t> cuda_output_fp16(size);
    cudaMemcpy(cuda_output_fp16.data(), d_output, size * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_output);

    // Convert CUDA output to FP32 for comparison
    std::vector<float> cuda_output_fp32(size);
    dequantizeFP16(cuda_output_fp16.data(), cuda_output_fp32.data(), size);

    ASSERT_FALSE(hasNaNOrInf(cuda_output_fp32.data(), size));

    double cosine = cosineSimilarity(cuda_output_fp32.data(), cpu_output_fp32.data(), size);
    double l2_error = relativeL2Error(cuda_output_fp32.data(), cpu_output_fp32.data(), size);

    std::cout << "  Residual Add FP16: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999); // FP16 has lower precision
    EXPECT_LE(l2_error, 0.02);
}

#else // !HAVE_CUDA

// Stub tests when CUDA is not available
TEST(Test__CUDAOpsParity, NoCUDAAvailable)
{
    GTEST_SKIP() << "CUDA not available, skipping CUDA ops parity tests";
}

#endif // HAVE_CUDA
