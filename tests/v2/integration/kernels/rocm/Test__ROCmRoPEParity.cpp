/**
 * @file Test__ROCmRoPEParity.cpp
 * @brief Parity tests for ROCm RoPE kernel vs CPU reference
 *
 * **Purpose**: Validate that ROCm RoPE kernels produce numerically equivalent
 * results to CPU kernels with high cosine similarity (>= 0.9999).
 *
 * **Tests**:
 * - ROCmRoPEKernelT vs CPURoPEKernelT (FP32, BF16, FP16)
 * - Small (4 seq_len × 14 heads × 64 head_dim) and large (128 seq_len) tensor sizes
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
#include "kernels/rocm/ops/ROCmRoPEKernelT.h"
#include "kernels/cpu/ops/CPURoPEKernelT.h"
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

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__ROCmRoPEParity : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

    std::vector<float> randomFP32(size_t count)
    {
        std::vector<float> vec(count);
        for (size_t i = 0; i < count; ++i)
        {
            vec[i] = dist_(rng_);
        }
        return vec;
    }
};

// ============================================================================
// FP32 Parity Tests
// ============================================================================

#ifdef HAVE_ROCM

TEST_F(Test__ROCmRoPEParity, RoPE_FP32_Small)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 4;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 14;
    constexpr int head_dim = 64;
    constexpr float rope_theta = 10000.0f;
    const size_t total = seq_len * n_heads * head_dim;
    const size_t total_k = seq_len * n_kv_heads * head_dim;

    // Generate random Q and K data
    auto q_data = randomFP32(total);
    auto k_data = randomFP32(total_k);

    // Position IDs: [0, 1, 2, 3]
    std::vector<int> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
        position_ids[i] = i;

    // Make copies for CPU and ROCm
    std::vector<float> cpu_q = q_data;
    std::vector<float> cpu_k = k_data;
    std::vector<float> rocm_q = q_data;
    std::vector<float> rocm_k = k_data;

    // CPU reference
    CPURoPEKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply_typed(cpu_q.data(), cpu_k.data(), position_ids.data(),
                           seq_len, n_heads, n_kv_heads, head_dim, rope_theta, -1);

    // ROCm kernel
    rocm::ROCmRoPEKernelT<ActivationPrecision::FP32> rocm_kernel;

    float *d_q, *d_k;
    int *d_pos_ids;
    hipMalloc(&d_q, total * sizeof(float));
    hipMalloc(&d_k, total_k * sizeof(float));
    hipMalloc(&d_pos_ids, seq_len * sizeof(int));

    hipMemcpy(d_q, rocm_q.data(), total * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_k, rocm_k.data(), total_k * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_pos_ids, position_ids.data(), seq_len * sizeof(int), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_q, d_k, d_pos_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, 0));
    hipDeviceSynchronize();

    hipMemcpy(rocm_q.data(), d_q, total * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(rocm_k.data(), d_k, total_k * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_pos_ids);

    // Validate Q
    ASSERT_FALSE(hasNaNOrInf(rocm_q.data(), total)) << "ROCm Q output contains NaN/Inf";
    double cosine_q = cosineSimilarity(rocm_q.data(), cpu_q.data(), total);
    double l2_error_q = relativeL2Error(rocm_q.data(), cpu_q.data(), total);

    // Validate K
    ASSERT_FALSE(hasNaNOrInf(rocm_k.data(), total_k)) << "ROCm K output contains NaN/Inf";
    double cosine_k = cosineSimilarity(rocm_k.data(), cpu_k.data(), total_k);
    double l2_error_k = relativeL2Error(rocm_k.data(), cpu_k.data(), total_k);

    std::cout << "  RoPE FP32 Small Q: cosine=" << cosine_q << ", L2_error=" << l2_error_q << std::endl;
    std::cout << "  RoPE FP32 Small K: cosine=" << cosine_k << ", L2_error=" << l2_error_k << std::endl;

    EXPECT_GE(cosine_q, 0.9999);
    EXPECT_GE(cosine_k, 0.9999);
    EXPECT_LE(l2_error_q, 0.01);
    EXPECT_LE(l2_error_k, 0.01);
}

TEST_F(Test__ROCmRoPEParity, RoPE_FP32_Large)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 128;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2; // GQA: different heads for Q and K
    constexpr int head_dim = 64;
    constexpr float rope_theta = 10000.0f;
    const size_t total_q = seq_len * n_heads * head_dim;
    const size_t total_k = seq_len * n_kv_heads * head_dim;

    auto q_data = randomFP32(total_q);
    auto k_data = randomFP32(total_k);

    std::vector<int> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
        position_ids[i] = i;

    std::vector<float> cpu_q = q_data;
    std::vector<float> cpu_k = k_data;
    std::vector<float> rocm_q = q_data;
    std::vector<float> rocm_k = k_data;

    CPURoPEKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.apply_typed(cpu_q.data(), cpu_k.data(), position_ids.data(),
                           seq_len, n_heads, n_kv_heads, head_dim, rope_theta, -1);

    rocm::ROCmRoPEKernelT<ActivationPrecision::FP32> rocm_kernel;

    float *d_q, *d_k;
    int *d_pos_ids;
    hipMalloc(&d_q, total_q * sizeof(float));
    hipMalloc(&d_k, total_k * sizeof(float));
    hipMalloc(&d_pos_ids, seq_len * sizeof(int));

    hipMemcpy(d_q, rocm_q.data(), total_q * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_k, rocm_k.data(), total_k * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_pos_ids, position_ids.data(), seq_len * sizeof(int), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_q, d_k, d_pos_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, 0));
    hipDeviceSynchronize();

    hipMemcpy(rocm_q.data(), d_q, total_q * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(rocm_k.data(), d_k, total_k * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_pos_ids);

    ASSERT_FALSE(hasNaNOrInf(rocm_q.data(), total_q)) << "ROCm Q output contains NaN/Inf";
    double cosine_q = cosineSimilarity(rocm_q.data(), cpu_q.data(), total_q);
    double l2_error_q = relativeL2Error(rocm_q.data(), cpu_q.data(), total_q);

    ASSERT_FALSE(hasNaNOrInf(rocm_k.data(), total_k)) << "ROCm K output contains NaN/Inf";
    double cosine_k = cosineSimilarity(rocm_k.data(), cpu_k.data(), total_k);
    double l2_error_k = relativeL2Error(rocm_k.data(), cpu_k.data(), total_k);

    std::cout << "  RoPE FP32 Large Q: cosine=" << cosine_q << ", L2_error=" << l2_error_q << std::endl;
    std::cout << "  RoPE FP32 Large K: cosine=" << cosine_k << ", L2_error=" << l2_error_k << std::endl;

    EXPECT_GE(cosine_q, 0.9999);
    EXPECT_GE(cosine_k, 0.9999);
    EXPECT_LE(l2_error_q, 0.01);
    EXPECT_LE(l2_error_k, 0.01);
}

// ============================================================================
// BF16 Parity Tests
// ============================================================================

TEST_F(Test__ROCmRoPEParity, RoPE_BF16_Small)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 4;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 14;
    constexpr int head_dim = 64;
    constexpr float rope_theta = 10000.0f;
    const size_t total = seq_len * n_heads * head_dim;
    const size_t total_k = seq_len * n_kv_heads * head_dim;

    // Generate FP32 data, then convert to BF16
    auto q_fp32 = randomFP32(total);
    auto k_fp32 = randomFP32(total_k);

    std::vector<uint16_t> q_bf16(total);
    std::vector<uint16_t> k_bf16(total_k);
    quantizeToBF16(q_fp32.data(), q_bf16.data(), total);
    quantizeToBF16(k_fp32.data(), k_bf16.data(), total_k);

    std::vector<int> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
        position_ids[i] = i;

    // CPU reference in BF16
    std::vector<uint16_t> cpu_q = q_bf16;
    std::vector<uint16_t> cpu_k = k_bf16;
    CPURoPEKernelT<ActivationPrecision::BF16> cpu_kernel;
    cpu_kernel.apply_typed(cpu_q.data(), cpu_k.data(), position_ids.data(),
                           seq_len, n_heads, n_kv_heads, head_dim, rope_theta, -1);

    // ROCm kernel
    std::vector<uint16_t> rocm_q = q_bf16;
    std::vector<uint16_t> rocm_k = k_bf16;
    rocm::ROCmRoPEKernelT<ActivationPrecision::BF16> rocm_kernel;

    uint16_t *d_q, *d_k;
    int *d_pos_ids;
    hipMalloc(&d_q, total * sizeof(uint16_t));
    hipMalloc(&d_k, total_k * sizeof(uint16_t));
    hipMalloc(&d_pos_ids, seq_len * sizeof(int));

    hipMemcpy(d_q, rocm_q.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_k, rocm_k.data(), total_k * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_pos_ids, position_ids.data(), seq_len * sizeof(int), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_q, d_k, d_pos_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, 0));
    hipDeviceSynchronize();

    hipMemcpy(rocm_q.data(), d_q, total * sizeof(uint16_t), hipMemcpyDeviceToHost);
    hipMemcpy(rocm_k.data(), d_k, total_k * sizeof(uint16_t), hipMemcpyDeviceToHost);

    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_pos_ids);

    // Dequantize for comparison
    std::vector<float> cpu_q_fp32(total), rocm_q_fp32(total);
    std::vector<float> cpu_k_fp32(total_k), rocm_k_fp32(total_k);
    dequantizeBF16(cpu_q.data(), cpu_q_fp32.data(), total);
    dequantizeBF16(rocm_q.data(), rocm_q_fp32.data(), total);
    dequantizeBF16(cpu_k.data(), cpu_k_fp32.data(), total_k);
    dequantizeBF16(rocm_k.data(), rocm_k_fp32.data(), total_k);

    ASSERT_FALSE(hasNaNOrInf(rocm_q_fp32.data(), total)) << "ROCm Q output contains NaN/Inf";
    double cosine_q = cosineSimilarity(rocm_q_fp32.data(), cpu_q_fp32.data(), total);
    double l2_error_q = relativeL2Error(rocm_q_fp32.data(), cpu_q_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(rocm_k_fp32.data(), total_k)) << "ROCm K output contains NaN/Inf";
    double cosine_k = cosineSimilarity(rocm_k_fp32.data(), cpu_k_fp32.data(), total_k);
    double l2_error_k = relativeL2Error(rocm_k_fp32.data(), cpu_k_fp32.data(), total_k);

    std::cout << "  RoPE BF16 Small Q: cosine=" << cosine_q << ", L2_error=" << l2_error_q << std::endl;
    std::cout << "  RoPE BF16 Small K: cosine=" << cosine_k << ", L2_error=" << l2_error_k << std::endl;

    // BF16 has lower precision, so relax thresholds slightly
    EXPECT_GE(cosine_q, 0.999);
    EXPECT_GE(cosine_k, 0.999);
    EXPECT_LE(l2_error_q, 0.02);
    EXPECT_LE(l2_error_k, 0.02);
}

TEST_F(Test__ROCmRoPEParity, RoPE_BF16_Large)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 128;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr float rope_theta = 10000.0f;
    const size_t total_q = seq_len * n_heads * head_dim;
    const size_t total_k = seq_len * n_kv_heads * head_dim;

    auto q_fp32 = randomFP32(total_q);
    auto k_fp32 = randomFP32(total_k);

    std::vector<uint16_t> q_bf16(total_q);
    std::vector<uint16_t> k_bf16(total_k);
    quantizeToBF16(q_fp32.data(), q_bf16.data(), total_q);
    quantizeToBF16(k_fp32.data(), k_bf16.data(), total_k);

    std::vector<int> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
        position_ids[i] = i;

    std::vector<uint16_t> cpu_q = q_bf16;
    std::vector<uint16_t> cpu_k = k_bf16;
    CPURoPEKernelT<ActivationPrecision::BF16> cpu_kernel;
    cpu_kernel.apply_typed(cpu_q.data(), cpu_k.data(), position_ids.data(),
                           seq_len, n_heads, n_kv_heads, head_dim, rope_theta, -1);

    std::vector<uint16_t> rocm_q = q_bf16;
    std::vector<uint16_t> rocm_k = k_bf16;
    rocm::ROCmRoPEKernelT<ActivationPrecision::BF16> rocm_kernel;

    uint16_t *d_q, *d_k;
    int *d_pos_ids;
    hipMalloc(&d_q, total_q * sizeof(uint16_t));
    hipMalloc(&d_k, total_k * sizeof(uint16_t));
    hipMalloc(&d_pos_ids, seq_len * sizeof(int));

    hipMemcpy(d_q, rocm_q.data(), total_q * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_k, rocm_k.data(), total_k * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_pos_ids, position_ids.data(), seq_len * sizeof(int), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_q, d_k, d_pos_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, 0));
    hipDeviceSynchronize();

    hipMemcpy(rocm_q.data(), d_q, total_q * sizeof(uint16_t), hipMemcpyDeviceToHost);
    hipMemcpy(rocm_k.data(), d_k, total_k * sizeof(uint16_t), hipMemcpyDeviceToHost);

    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_pos_ids);

    std::vector<float> cpu_q_fp32(total_q), rocm_q_fp32(total_q);
    std::vector<float> cpu_k_fp32(total_k), rocm_k_fp32(total_k);
    dequantizeBF16(cpu_q.data(), cpu_q_fp32.data(), total_q);
    dequantizeBF16(rocm_q.data(), rocm_q_fp32.data(), total_q);
    dequantizeBF16(cpu_k.data(), cpu_k_fp32.data(), total_k);
    dequantizeBF16(rocm_k.data(), rocm_k_fp32.data(), total_k);

    ASSERT_FALSE(hasNaNOrInf(rocm_q_fp32.data(), total_q)) << "ROCm Q output contains NaN/Inf";
    double cosine_q = cosineSimilarity(rocm_q_fp32.data(), cpu_q_fp32.data(), total_q);
    double l2_error_q = relativeL2Error(rocm_q_fp32.data(), cpu_q_fp32.data(), total_q);

    ASSERT_FALSE(hasNaNOrInf(rocm_k_fp32.data(), total_k)) << "ROCm K output contains NaN/Inf";
    double cosine_k = cosineSimilarity(rocm_k_fp32.data(), cpu_k_fp32.data(), total_k);
    double l2_error_k = relativeL2Error(rocm_k_fp32.data(), cpu_k_fp32.data(), total_k);

    std::cout << "  RoPE BF16 Large Q: cosine=" << cosine_q << ", L2_error=" << l2_error_q << std::endl;
    std::cout << "  RoPE BF16 Large K: cosine=" << cosine_k << ", L2_error=" << l2_error_k << std::endl;

    EXPECT_GE(cosine_q, 0.999);
    EXPECT_GE(cosine_k, 0.999);
    EXPECT_LE(l2_error_q, 0.02);
    EXPECT_LE(l2_error_k, 0.02);
}

// ============================================================================
// FP16 Parity Tests
// ============================================================================

TEST_F(Test__ROCmRoPEParity, RoPE_FP16_Small)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 4;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 14;
    constexpr int head_dim = 64;
    constexpr float rope_theta = 10000.0f;
    const size_t total = seq_len * n_heads * head_dim;
    const size_t total_k = seq_len * n_kv_heads * head_dim;

    auto q_fp32 = randomFP32(total);
    auto k_fp32 = randomFP32(total_k);

    std::vector<uint16_t> q_fp16(total);
    std::vector<uint16_t> k_fp16(total_k);
    quantizeToFP16(q_fp32.data(), q_fp16.data(), total);
    quantizeToFP16(k_fp32.data(), k_fp16.data(), total_k);

    std::vector<int> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
        position_ids[i] = i;

    std::vector<uint16_t> cpu_q = q_fp16;
    std::vector<uint16_t> cpu_k = k_fp16;
    CPURoPEKernelT<ActivationPrecision::FP16> cpu_kernel;
    cpu_kernel.apply_typed(cpu_q.data(), cpu_k.data(), position_ids.data(),
                           seq_len, n_heads, n_kv_heads, head_dim, rope_theta, -1);

    std::vector<uint16_t> rocm_q = q_fp16;
    std::vector<uint16_t> rocm_k = k_fp16;
    rocm::ROCmRoPEKernelT<ActivationPrecision::FP16> rocm_kernel;

    uint16_t *d_q, *d_k;
    int *d_pos_ids;
    hipMalloc(&d_q, total * sizeof(uint16_t));
    hipMalloc(&d_k, total_k * sizeof(uint16_t));
    hipMalloc(&d_pos_ids, seq_len * sizeof(int));

    hipMemcpy(d_q, rocm_q.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_k, rocm_k.data(), total_k * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_pos_ids, position_ids.data(), seq_len * sizeof(int), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_q, d_k, d_pos_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, 0));
    hipDeviceSynchronize();

    hipMemcpy(rocm_q.data(), d_q, total * sizeof(uint16_t), hipMemcpyDeviceToHost);
    hipMemcpy(rocm_k.data(), d_k, total_k * sizeof(uint16_t), hipMemcpyDeviceToHost);

    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_pos_ids);

    std::vector<float> cpu_q_fp32(total), rocm_q_fp32(total);
    std::vector<float> cpu_k_fp32(total_k), rocm_k_fp32(total_k);
    dequantizeFP16(cpu_q.data(), cpu_q_fp32.data(), total);
    dequantizeFP16(rocm_q.data(), rocm_q_fp32.data(), total);
    dequantizeFP16(cpu_k.data(), cpu_k_fp32.data(), total_k);
    dequantizeFP16(rocm_k.data(), rocm_k_fp32.data(), total_k);

    ASSERT_FALSE(hasNaNOrInf(rocm_q_fp32.data(), total)) << "ROCm Q output contains NaN/Inf";
    double cosine_q = cosineSimilarity(rocm_q_fp32.data(), cpu_q_fp32.data(), total);
    double l2_error_q = relativeL2Error(rocm_q_fp32.data(), cpu_q_fp32.data(), total);

    ASSERT_FALSE(hasNaNOrInf(rocm_k_fp32.data(), total_k)) << "ROCm K output contains NaN/Inf";
    double cosine_k = cosineSimilarity(rocm_k_fp32.data(), cpu_k_fp32.data(), total_k);
    double l2_error_k = relativeL2Error(rocm_k_fp32.data(), cpu_k_fp32.data(), total_k);

    std::cout << "  RoPE FP16 Small Q: cosine=" << cosine_q << ", L2_error=" << l2_error_q << std::endl;
    std::cout << "  RoPE FP16 Small K: cosine=" << cosine_k << ", L2_error=" << l2_error_k << std::endl;

    EXPECT_GE(cosine_q, 0.999);
    EXPECT_GE(cosine_k, 0.999);
    EXPECT_LE(l2_error_q, 0.02);
    EXPECT_LE(l2_error_k, 0.02);
}

TEST_F(Test__ROCmRoPEParity, RoPE_FP16_Large)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 128;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr float rope_theta = 10000.0f;
    const size_t total_q = seq_len * n_heads * head_dim;
    const size_t total_k = seq_len * n_kv_heads * head_dim;

    auto q_fp32 = randomFP32(total_q);
    auto k_fp32 = randomFP32(total_k);

    std::vector<uint16_t> q_fp16(total_q);
    std::vector<uint16_t> k_fp16(total_k);
    quantizeToFP16(q_fp32.data(), q_fp16.data(), total_q);
    quantizeToFP16(k_fp32.data(), k_fp16.data(), total_k);

    std::vector<int> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
        position_ids[i] = i;

    std::vector<uint16_t> cpu_q = q_fp16;
    std::vector<uint16_t> cpu_k = k_fp16;
    CPURoPEKernelT<ActivationPrecision::FP16> cpu_kernel;
    cpu_kernel.apply_typed(cpu_q.data(), cpu_k.data(), position_ids.data(),
                           seq_len, n_heads, n_kv_heads, head_dim, rope_theta, -1);

    std::vector<uint16_t> rocm_q = q_fp16;
    std::vector<uint16_t> rocm_k = k_fp16;
    rocm::ROCmRoPEKernelT<ActivationPrecision::FP16> rocm_kernel;

    uint16_t *d_q, *d_k;
    int *d_pos_ids;
    hipMalloc(&d_q, total_q * sizeof(uint16_t));
    hipMalloc(&d_k, total_k * sizeof(uint16_t));
    hipMalloc(&d_pos_ids, seq_len * sizeof(int));

    hipMemcpy(d_q, rocm_q.data(), total_q * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_k, rocm_k.data(), total_k * sizeof(uint16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_pos_ids, position_ids.data(), seq_len * sizeof(int), hipMemcpyHostToDevice);

    ASSERT_TRUE(rocm_kernel.apply_typed(d_q, d_k, d_pos_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, 0));
    hipDeviceSynchronize();

    hipMemcpy(rocm_q.data(), d_q, total_q * sizeof(uint16_t), hipMemcpyDeviceToHost);
    hipMemcpy(rocm_k.data(), d_k, total_k * sizeof(uint16_t), hipMemcpyDeviceToHost);

    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_pos_ids);

    std::vector<float> cpu_q_fp32(total_q), rocm_q_fp32(total_q);
    std::vector<float> cpu_k_fp32(total_k), rocm_k_fp32(total_k);
    dequantizeFP16(cpu_q.data(), cpu_q_fp32.data(), total_q);
    dequantizeFP16(rocm_q.data(), rocm_q_fp32.data(), total_q);
    dequantizeFP16(cpu_k.data(), cpu_k_fp32.data(), total_k);
    dequantizeFP16(rocm_k.data(), rocm_k_fp32.data(), total_k);

    ASSERT_FALSE(hasNaNOrInf(rocm_q_fp32.data(), total_q)) << "ROCm Q output contains NaN/Inf";
    double cosine_q = cosineSimilarity(rocm_q_fp32.data(), cpu_q_fp32.data(), total_q);
    double l2_error_q = relativeL2Error(rocm_q_fp32.data(), cpu_q_fp32.data(), total_q);

    ASSERT_FALSE(hasNaNOrInf(rocm_k_fp32.data(), total_k)) << "ROCm K output contains NaN/Inf";
    double cosine_k = cosineSimilarity(rocm_k_fp32.data(), cpu_k_fp32.data(), total_k);
    double l2_error_k = relativeL2Error(rocm_k_fp32.data(), cpu_k_fp32.data(), total_k);

    std::cout << "  RoPE FP16 Large Q: cosine=" << cosine_q << ", L2_error=" << l2_error_q << std::endl;
    std::cout << "  RoPE FP16 Large K: cosine=" << cosine_k << ", L2_error=" << l2_error_k << std::endl;

    EXPECT_GE(cosine_q, 0.999);
    EXPECT_GE(cosine_k, 0.999);
    EXPECT_LE(l2_error_q, 0.02);
    EXPECT_LE(l2_error_k, 0.02);
}

#else // !HAVE_ROCM

TEST(Test__ROCmRoPEParity, NotAvailable)
{
    GTEST_SKIP() << "ROCm support not compiled (HAVE_ROCM=OFF)";
}

#endif // HAVE_ROCM
