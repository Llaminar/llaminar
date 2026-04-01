/**
 * @file Test__ROCmSwiGLUParity.cpp
 * @brief Parity tests for ROCm SwiGLU kernel vs CPU reference
 *
 * **Purpose**: Validate that ROCm SwiGLU kernels produce numerically equivalent
 * results to CPU kernels with high cosine similarity (>= 0.9999).
 *
 * **Tests**:
 * - ROCmSwiGLUKernelT vs CPUSwiGLUKernelT (FP32, BF16, FP16)
 * - Small (4x64) and large (32x4864) tensor sizes
 *
 * **Pass Criteria**:
 * - Cosine similarity >= 0.9999 (very high correlation)
 * - No NaN/Inf in outputs
 * - Relative L2 error < 1% for FP32
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 *
 * @author Llaminar Team
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers
#include "tensors/Tensors.h"
#include "execution/config/RuntimeConfig.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/ops/ROCmSwiGLUKernelT.h"
#include "kernels/cpu/ops/CPUSwiGLUKernelT.h"
#endif

#include "../../../utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{

    // ============================================================================
    // ROCm Availability Check
    // ============================================================================

    bool hasROCm()
    {
#ifdef HAVE_ROCM
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
#else
        return false;
#endif
    }

#define SKIP_IF_NO_ROCM()                                           \
    do                                                              \
    {                                                               \
        if (!hasROCm())                                             \
        {                                                           \
            GTEST_SKIP() << "No ROCm GPU available, skipping test"; \
        }                                                           \
    } while (0)

    // ============================================================================
    // Similarity Utilities
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

    bool hasNaNOrInf(const float *data, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::isnan(data[i]) || std::isinf(data[i]))
                return true;
        }
        return false;
    }

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

class Test__ROCmSwiGLUParity : public ::testing::Test
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
};

#ifdef HAVE_ROCM

// ============================================================================
// SwiGLU FP32 Parity Tests
// ============================================================================

TEST_F(Test__ROCmSwiGLUParity, SwiGLU_FP32_Small)
{
    SKIP_IF_NO_ROCM();

    constexpr int rows = 4;
    constexpr int cols = 64;
    const size_t total = rows * cols;

    auto gate_data = randomFP32(total);
    auto up_data = randomFP32(total);
    std::vector<float> cpu_output(total, 0.0f);
    std::vector<float> rocm_output(total, 0.0f);

    // CPU reference
    CPUSwiGLUKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply(gate_data.data(), up_data.data(), cpu_output.data(),
                     rows, cols, false, nullptr, -1);

    // ROCm kernel
    llaminar2::rocm::ROCmSwiGLUKernelT<ActivationPrecision::FP32> rocm_kernel;

    float *d_gate, *d_up, *d_output;
    hipMalloc(&d_gate, total * sizeof(float));
    hipMalloc(&d_up, total * sizeof(float));
    hipMalloc(&d_output, total * sizeof(float));

    hipMemcpy(d_gate, gate_data.data(), total * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_up, up_data.data(), total * sizeof(float), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    hipDeviceSynchronize();

    hipMemcpy(rocm_output.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_gate);
    hipFree(d_up);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), total));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), total);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), total);

    std::cout << "  SwiGLU FP32 Small: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.9999);
    EXPECT_LE(l2_error, 0.01);
}

TEST_F(Test__ROCmSwiGLUParity, SwiGLU_FP32_Large)
{
    SKIP_IF_NO_ROCM();

    constexpr int rows = 32;
    constexpr int cols = 4864; // Qwen2-0.5B intermediate_dim
    const size_t total = rows * cols;

    auto gate_data = randomFP32(total);
    auto up_data = randomFP32(total);
    std::vector<float> cpu_output(total, 0.0f);
    std::vector<float> rocm_output(total, 0.0f);

    CPUSwiGLUKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply(gate_data.data(), up_data.data(), cpu_output.data(),
                     rows, cols, false, nullptr, -1);

    llaminar2::rocm::ROCmSwiGLUKernelT<ActivationPrecision::FP32> rocm_kernel;

    float *d_gate, *d_up, *d_output;
    hipMalloc(&d_gate, total * sizeof(float));
    hipMalloc(&d_up, total * sizeof(float));
    hipMalloc(&d_output, total * sizeof(float));

    hipMemcpy(d_gate, gate_data.data(), total * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_up, up_data.data(), total * sizeof(float), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    hipDeviceSynchronize();

    hipMemcpy(rocm_output.data(), d_output, total * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_gate);
    hipFree(d_up);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), total));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), total);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), total);

    std::cout << "  SwiGLU FP32 Large: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.9999);
    EXPECT_LE(l2_error, 0.01);
}

// ============================================================================
// SwiGLU BF16 Parity Tests
// ============================================================================

TEST_F(Test__ROCmSwiGLUParity, SwiGLU_BF16_Small)
{
    SKIP_IF_NO_ROCM();

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

    // ROCm kernel
    llaminar2::rocm::ROCmSwiGLUKernelT<ActivationPrecision::BF16> rocm_kernel;

    uint16_t *d_gate, *d_up, *d_output;
    hipMalloc(&d_gate, total * sizeof(uint16_t));
    hipMalloc(&d_up, total * sizeof(uint16_t));
    hipMalloc(&d_output, total * sizeof(uint16_t));

    hipMemcpy(d_gate, gate_bf16.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_up, up_bf16.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    hipDeviceSynchronize();

    std::vector<uint16_t> rocm_output_bf16(total);
    hipMemcpy(rocm_output_bf16.data(), d_output, total * sizeof(uint16_t), hipMemcpyDeviceToHost);

    hipFree(d_gate);
    hipFree(d_up);
    hipFree(d_output);

    // Convert outputs to FP32 for comparison
    std::vector<float> cpu_output_fp32(total);
    std::vector<float> rocm_output_fp32(total);
    dequantizeBF16(cpu_output_bf16.data(), cpu_output_fp32.data(), total);
    dequantizeBF16(rocm_output_bf16.data(), rocm_output_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(rocm_output_fp32.data(), total));

    double cosine = cosineSimilarity(rocm_output_fp32.data(), cpu_output_fp32.data(), total);
    double l2_error = relativeL2Error(rocm_output_fp32.data(), cpu_output_fp32.data(), total);

    std::cout << "  SwiGLU BF16 Small: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999);  // Slightly relaxed for BF16
    EXPECT_LE(l2_error, 0.02); // 2% tolerance for BF16
}

TEST_F(Test__ROCmSwiGLUParity, SwiGLU_BF16_Large)
{
    SKIP_IF_NO_ROCM();

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

    llaminar2::rocm::ROCmSwiGLUKernelT<ActivationPrecision::BF16> rocm_kernel;

    uint16_t *d_gate, *d_up, *d_output;
    hipMalloc(&d_gate, total * sizeof(uint16_t));
    hipMalloc(&d_up, total * sizeof(uint16_t));
    hipMalloc(&d_output, total * sizeof(uint16_t));

    hipMemcpy(d_gate, gate_bf16.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_up, up_bf16.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    hipDeviceSynchronize();

    std::vector<uint16_t> rocm_output_bf16(total);
    hipMemcpy(rocm_output_bf16.data(), d_output, total * sizeof(uint16_t), hipMemcpyDeviceToHost);

    hipFree(d_gate);
    hipFree(d_up);
    hipFree(d_output);

    std::vector<float> cpu_output_fp32(total);
    std::vector<float> rocm_output_fp32(total);
    dequantizeBF16(cpu_output_bf16.data(), cpu_output_fp32.data(), total);
    dequantizeBF16(rocm_output_bf16.data(), rocm_output_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(rocm_output_fp32.data(), total));

    double cosine = cosineSimilarity(rocm_output_fp32.data(), cpu_output_fp32.data(), total);
    double l2_error = relativeL2Error(rocm_output_fp32.data(), cpu_output_fp32.data(), total);

    std::cout << "  SwiGLU BF16 Large: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999);
    EXPECT_LE(l2_error, 0.02);
}

// ============================================================================
// SwiGLU FP16 Parity Tests
// ============================================================================

TEST_F(Test__ROCmSwiGLUParity, SwiGLU_FP16_Small)
{
    SKIP_IF_NO_ROCM();

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

    // ROCm kernel
    llaminar2::rocm::ROCmSwiGLUKernelT<ActivationPrecision::FP16> rocm_kernel;

    uint16_t *d_gate, *d_up, *d_output;
    hipMalloc(&d_gate, total * sizeof(uint16_t));
    hipMalloc(&d_up, total * sizeof(uint16_t));
    hipMalloc(&d_output, total * sizeof(uint16_t));

    hipMemcpy(d_gate, gate_fp16.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_up, up_fp16.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    hipDeviceSynchronize();

    std::vector<uint16_t> rocm_output_fp16(total);
    hipMemcpy(rocm_output_fp16.data(), d_output, total * sizeof(uint16_t), hipMemcpyDeviceToHost);

    hipFree(d_gate);
    hipFree(d_up);
    hipFree(d_output);

    // Convert outputs to FP32 for comparison
    std::vector<float> cpu_output_fp32(total);
    std::vector<float> rocm_output_fp32(total);
    dequantizeFP16(cpu_output_fp16.data(), cpu_output_fp32.data(), total);
    dequantizeFP16(rocm_output_fp16.data(), rocm_output_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(rocm_output_fp32.data(), total));

    double cosine = cosineSimilarity(rocm_output_fp32.data(), cpu_output_fp32.data(), total);
    double l2_error = relativeL2Error(rocm_output_fp32.data(), cpu_output_fp32.data(), total);

    std::cout << "  SwiGLU FP16 Small: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999);  // Slightly relaxed for FP16
    EXPECT_LE(l2_error, 0.02); // 2% tolerance for FP16
}

TEST_F(Test__ROCmSwiGLUParity, SwiGLU_FP16_Large)
{
    SKIP_IF_NO_ROCM();

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

    llaminar2::rocm::ROCmSwiGLUKernelT<ActivationPrecision::FP16> rocm_kernel;

    uint16_t *d_gate, *d_up, *d_output;
    hipMalloc(&d_gate, total * sizeof(uint16_t));
    hipMalloc(&d_up, total * sizeof(uint16_t));
    hipMalloc(&d_output, total * sizeof(uint16_t));

    hipMemcpy(d_gate, gate_fp16.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_up, up_fp16.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_gate, d_up, d_output, total, 0));
    hipDeviceSynchronize();

    std::vector<uint16_t> rocm_output_fp16(total);
    hipMemcpy(rocm_output_fp16.data(), d_output, total * sizeof(uint16_t), hipMemcpyDeviceToHost);

    hipFree(d_gate);
    hipFree(d_up);
    hipFree(d_output);

    std::vector<float> cpu_output_fp32(total);
    std::vector<float> rocm_output_fp32(total);
    dequantizeFP16(cpu_output_fp16.data(), cpu_output_fp32.data(), total);
    dequantizeFP16(rocm_output_fp16.data(), rocm_output_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(rocm_output_fp32.data(), total));

    double cosine = cosineSimilarity(rocm_output_fp32.data(), cpu_output_fp32.data(), total);
    double l2_error = relativeL2Error(rocm_output_fp32.data(), cpu_output_fp32.data(), total);

    std::cout << "  SwiGLU FP16 Large: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.999);
    EXPECT_LE(l2_error, 0.02);
}

// ============================================================================
// SwiGLU Tensor Interface Test
// Note: Tensor interface tests require proper GPU coherence setup which is
// handled by DeviceGraphExecutor in the pipeline. The low-level apply_typed tests
// above verify kernel correctness. The tensor interface is tested implicitly
// through full pipeline integration tests.
// ============================================================================

// ============================================================================
// apply_tensor() API Tests
// ============================================================================
// These tests exercise the TensorBase* API (apply_tensor) which is what the
// actual pipeline uses.

TEST_F(Test__ROCmSwiGLUParity, SwiGLU_FP32_ApplyTensor)
{
    SKIP_IF_NO_ROCM();

    constexpr int rows = 4;
    constexpr int cols = 64;
    const size_t total = rows * cols;

    auto gate = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
    auto up = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
    auto cpu_output = TestTensorFactory::createFP32({rows, cols});
    auto rocm_output = TestTensorFactory::createFP32({rows, cols});

    CPUSwiGLUKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply(gate->data(), up->data(), cpu_output->mutable_data(),
                     rows, cols, false, nullptr, -1);

    DeviceId rocm_device = DeviceId::rocm(0);
    ASSERT_TRUE(gate->ensureOnDevice(rocm_device));
    ASSERT_TRUE(up->ensureOnDevice(rocm_device));
    ASSERT_TRUE(rocm_output->ensureOnDevice(rocm_device));

    llaminar2::rocm::ROCmSwiGLUKernelT<ActivationPrecision::FP32> rocm_kernel;
    ASSERT_TRUE(rocm_kernel.apply_tensor(
        gate.get(), up.get(), rocm_output.get(),
        rows, cols, false, nullptr, 0));

    hipDeviceSynchronize();
    rocm_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *result = rocm_output->data();

    ASSERT_FALSE(hasNaNOrInf(result, total));

    double cosine = cosineSimilarity(result, cpu_output->data(), total);
    double l2_error = relativeL2Error(result, cpu_output->data(), total);

    std::cout << "  SwiGLU FP32 ApplyTensor: cosine=" << cosine << ", L2_error=" << l2_error << std::endl;

    EXPECT_GE(cosine, 0.9999) << "Cosine similarity too low - apply_tensor() may have pointer issues";
    EXPECT_LE(l2_error, 0.01);
}

#else // !HAVE_ROCM

// Stub test when HAVE_ROCM is not defined
TEST(Test__ROCmSwiGLUParity, NotAvailable)
{
    GTEST_SKIP() << "ROCm support not compiled (HAVE_ROCM=OFF)";
}

#endif // HAVE_ROCM
