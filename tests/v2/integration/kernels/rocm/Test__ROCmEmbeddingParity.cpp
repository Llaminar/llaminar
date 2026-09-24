/**
 * @file Test__ROCmEmbeddingParity.cpp
 * @brief Integration tests for ROCm Embedding kernel vs CPU reference
 *
 * Tests embedding lookup operations on ROCm GPUs comparing against CPU reference.
 * Covers FP32, BF16, FP16, and Q8_1 output formats.
 */

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

// Include the kernels
#include "kernels/rocm/ops/ROCmEmbeddingKernelT.h"
#include "kernels/cpu/ops/CPUEmbeddingKernelT.h"
#include "tensors/TensorClasses.h"
#include "tensors/BlockStructures.h"
// Note: FP16Utils.h is already included via CPUTensors.h, provides llaminar2::fp16_to_fp32

using namespace llaminar2;

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * @brief Check if ROCm is available
 */
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

/**
 * @brief Compute cosine similarity between two vectors
 */
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

/**
 * @brief Compute relative L2 error between two vectors
 */
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

/**
 * @brief Check for NaN or Inf values
 */
bool hasNaNOrInf(const float *data, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (std::isnan(data[i]) || std::isinf(data[i]))
            return true;
    }
    return false;
}

// Use fp16_to_fp32 and bf16_to_fp32 from tensors/FP16Utils.h (included via CPUTensors.h)

// ============================================================================
// Test Fixture
// ============================================================================

#ifdef HAVE_ROCM

class Test__ROCmEmbeddingParity : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

    std::vector<float> randomFP32(size_t count)
    {
        std::vector<float> data(count);
        for (auto &val : data)
            val = dist_(rng_);
        return data;
    }

    std::vector<int> randomTokenIds(int num_tokens, int vocab_size)
    {
        std::vector<int> ids(num_tokens);
        std::uniform_int_distribution<int> id_dist(0, vocab_size - 1);
        for (auto &id : ids)
            id = id_dist(rng_);
        return ids;
    }
};

// ============================================================================
// FP32 Embedding Tests
// ============================================================================

TEST_F(Test__ROCmEmbeddingParity, Embedding_FP32_Small)
{
    SKIP_IF_NO_ROCM();

    constexpr int num_tokens = 8;
    constexpr int d_model = 64; // Small dimension for quick test
    constexpr int vocab_size = 1000;

    // Generate random embedding table and token IDs
    auto embed_data = randomFP32(vocab_size * d_model);
    auto token_ids = randomTokenIds(num_tokens, vocab_size);

    std::vector<float> cpu_output(num_tokens * d_model, 0.0f);
    std::vector<float> rocm_output(num_tokens * d_model, 0.0f);

    // CPU reference
    CPUEmbeddingKernelT<FP32Tensor> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.apply(embed_data.data(), token_ids.data(),
                                  num_tokens, d_model, cpu_output.data(),
                                  nullptr, -1));

    // Allocate GPU memory
    float *d_embed = nullptr;
    float *d_output = nullptr;
    int *d_token_ids = nullptr;

    ASSERT_EQ(hipMalloc(&d_embed, vocab_size * d_model * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_token_ids, num_tokens * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, num_tokens * d_model * sizeof(float)), hipSuccess);

    // Copy data to GPU
    ASSERT_EQ(hipMemcpy(d_embed, embed_data.data(),
                        vocab_size * d_model * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_token_ids, token_ids.data(),
                        num_tokens * sizeof(int), hipMemcpyHostToDevice), hipSuccess);

    // Execute ROCm kernel
    ROCmEmbeddingKernelT rocm_kernel;
    ASSERT_TRUE(rocm_kernel.apply(d_embed, d_token_ids, num_tokens, d_model,
                                   d_output, nullptr, 0));

    // Copy result back
    ASSERT_EQ(hipMemcpy(rocm_output.data(), d_output,
                        num_tokens * d_model * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

    // Cleanup
    (void)hipFree(d_embed);
    (void)hipFree(d_token_ids);
    (void)hipFree(d_output);

    // Validate
    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), num_tokens * d_model));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), num_tokens * d_model);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), num_tokens * d_model);

    std::cout << "  Embedding FP32 Small: cosine=" << cosine << ", L2_error=" << l2_error * 100 << "%" << std::endl;

    EXPECT_GE(cosine, 0.9999);
    EXPECT_LE(l2_error, 0.01);
}

TEST_F(Test__ROCmEmbeddingParity, Embedding_FP32_Large)
{
    SKIP_IF_NO_ROCM();

    constexpr int num_tokens = 128;
    constexpr int d_model = 896; // Qwen2.5-0.5B dimension
    constexpr int vocab_size = 10000;

    auto embed_data = randomFP32(vocab_size * d_model);
    auto token_ids = randomTokenIds(num_tokens, vocab_size);

    std::vector<float> cpu_output(num_tokens * d_model, 0.0f);
    std::vector<float> rocm_output(num_tokens * d_model, 0.0f);

    // CPU reference
    CPUEmbeddingKernelT<FP32Tensor> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.apply(embed_data.data(), token_ids.data(),
                                  num_tokens, d_model, cpu_output.data(),
                                  nullptr, -1));

    // GPU execution
    float *d_embed = nullptr;
    float *d_output = nullptr;
    int *d_token_ids = nullptr;

    ASSERT_EQ(hipMalloc(&d_embed, vocab_size * d_model * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_token_ids, num_tokens * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, num_tokens * d_model * sizeof(float)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_embed, embed_data.data(),
                        vocab_size * d_model * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_token_ids, token_ids.data(),
                        num_tokens * sizeof(int), hipMemcpyHostToDevice), hipSuccess);

    ROCmEmbeddingKernelT rocm_kernel;
    ASSERT_TRUE(rocm_kernel.apply(d_embed, d_token_ids, num_tokens, d_model,
                                   d_output, nullptr, 0));

    ASSERT_EQ(hipMemcpy(rocm_output.data(), d_output,
                        num_tokens * d_model * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

    (void)hipFree(d_embed);
    (void)hipFree(d_token_ids);
    (void)hipFree(d_output);

    // Validate
    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), num_tokens * d_model));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), num_tokens * d_model);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), num_tokens * d_model);

    std::cout << "  Embedding FP32 Large: cosine=" << cosine << ", L2_error=" << l2_error * 100 << "%" << std::endl;

    EXPECT_GE(cosine, 0.9999);
    EXPECT_LE(l2_error, 0.01);
}

// ============================================================================
// BF16 Embedding Tests
// ============================================================================

TEST_F(Test__ROCmEmbeddingParity, Embedding_BF16_Small)
{
    SKIP_IF_NO_ROCM();

    constexpr int num_tokens = 8;
    constexpr int d_model = 64;
    constexpr int vocab_size = 1000;

    auto embed_data = randomFP32(vocab_size * d_model);
    auto token_ids = randomTokenIds(num_tokens, vocab_size);

    // CPU reference (compute expected FP32 then we'll compare converted BF16)
    std::vector<float> cpu_output(num_tokens * d_model, 0.0f);
    CPUEmbeddingKernelT<FP32Tensor> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.apply(embed_data.data(), token_ids.data(),
                                  num_tokens, d_model, cpu_output.data(),
                                  nullptr, -1));

    // GPU execution with BF16 output
    float *d_embed = nullptr;
    uint16_t *d_output = nullptr;
    int *d_token_ids = nullptr;
    std::vector<uint16_t> rocm_bf16_output(num_tokens * d_model);

    ASSERT_EQ(hipMalloc(&d_embed, vocab_size * d_model * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_token_ids, num_tokens * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, num_tokens * d_model * sizeof(uint16_t)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_embed, embed_data.data(),
                        vocab_size * d_model * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_token_ids, token_ids.data(),
                        num_tokens * sizeof(int), hipMemcpyHostToDevice), hipSuccess);

    ROCmEmbeddingKernelT rocm_kernel;
    ASSERT_TRUE(rocm_kernel.apply_bf16(d_embed, d_token_ids, num_tokens, d_model,
                                        d_output, nullptr, 0));

    ASSERT_EQ(hipMemcpy(rocm_bf16_output.data(), d_output,
                        num_tokens * d_model * sizeof(uint16_t), hipMemcpyDeviceToHost), hipSuccess);

    (void)hipFree(d_embed);
    (void)hipFree(d_token_ids);
    (void)hipFree(d_output);

    // Convert BF16 to FP32 for comparison
    std::vector<float> rocm_output(num_tokens * d_model);
    for (size_t i = 0; i < rocm_bf16_output.size(); ++i)
    {
        rocm_output[i] = simd::bf16_to_fp32(rocm_bf16_output[i]);
    }

    // Validate
    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), num_tokens * d_model));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), num_tokens * d_model);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), num_tokens * d_model);

    std::cout << "  Embedding BF16 Small: cosine=" << cosine << ", L2_error=" << l2_error * 100 << "%" << std::endl;

    EXPECT_GE(cosine, 0.999);  // BF16 has lower precision
    EXPECT_LE(l2_error, 0.02);
}

TEST_F(Test__ROCmEmbeddingParity, Embedding_BF16_Large)
{
    SKIP_IF_NO_ROCM();

    constexpr int num_tokens = 128;
    constexpr int d_model = 896;
    constexpr int vocab_size = 10000;

    auto embed_data = randomFP32(vocab_size * d_model);
    auto token_ids = randomTokenIds(num_tokens, vocab_size);

    std::vector<float> cpu_output(num_tokens * d_model, 0.0f);
    CPUEmbeddingKernelT<FP32Tensor> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.apply(embed_data.data(), token_ids.data(),
                                  num_tokens, d_model, cpu_output.data(),
                                  nullptr, -1));

    float *d_embed = nullptr;
    uint16_t *d_output = nullptr;
    int *d_token_ids = nullptr;
    std::vector<uint16_t> rocm_bf16_output(num_tokens * d_model);

    ASSERT_EQ(hipMalloc(&d_embed, vocab_size * d_model * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_token_ids, num_tokens * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, num_tokens * d_model * sizeof(uint16_t)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_embed, embed_data.data(),
                        vocab_size * d_model * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_token_ids, token_ids.data(),
                        num_tokens * sizeof(int), hipMemcpyHostToDevice), hipSuccess);

    ROCmEmbeddingKernelT rocm_kernel;
    ASSERT_TRUE(rocm_kernel.apply_bf16(d_embed, d_token_ids, num_tokens, d_model,
                                        d_output, nullptr, 0));

    ASSERT_EQ(hipMemcpy(rocm_bf16_output.data(), d_output,
                        num_tokens * d_model * sizeof(uint16_t), hipMemcpyDeviceToHost), hipSuccess);

    (void)hipFree(d_embed);
    (void)hipFree(d_token_ids);
    (void)hipFree(d_output);

    std::vector<float> rocm_output(num_tokens * d_model);
    for (size_t i = 0; i < rocm_bf16_output.size(); ++i)
    {
        rocm_output[i] = simd::bf16_to_fp32(rocm_bf16_output[i]);
    }

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), num_tokens * d_model));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), num_tokens * d_model);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), num_tokens * d_model);

    std::cout << "  Embedding BF16 Large: cosine=" << cosine << ", L2_error=" << l2_error * 100 << "%" << std::endl;

    EXPECT_GE(cosine, 0.999);
    EXPECT_LE(l2_error, 0.02);
}

// ============================================================================
// FP16 Embedding Tests
// ============================================================================

TEST_F(Test__ROCmEmbeddingParity, Embedding_FP16_Small)
{
    SKIP_IF_NO_ROCM();

    constexpr int num_tokens = 8;
    constexpr int d_model = 64;
    constexpr int vocab_size = 1000;

    auto embed_data = randomFP32(vocab_size * d_model);
    auto token_ids = randomTokenIds(num_tokens, vocab_size);

    std::vector<float> cpu_output(num_tokens * d_model, 0.0f);
    CPUEmbeddingKernelT<FP32Tensor> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.apply(embed_data.data(), token_ids.data(),
                                  num_tokens, d_model, cpu_output.data(),
                                  nullptr, -1));

    float *d_embed = nullptr;
    uint16_t *d_output = nullptr;
    int *d_token_ids = nullptr;
    std::vector<uint16_t> rocm_fp16_output(num_tokens * d_model);

    ASSERT_EQ(hipMalloc(&d_embed, vocab_size * d_model * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_token_ids, num_tokens * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, num_tokens * d_model * sizeof(uint16_t)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_embed, embed_data.data(),
                        vocab_size * d_model * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_token_ids, token_ids.data(),
                        num_tokens * sizeof(int), hipMemcpyHostToDevice), hipSuccess);

    ROCmEmbeddingKernelT rocm_kernel;
    ASSERT_TRUE(rocm_kernel.apply_fp16(d_embed, d_token_ids, num_tokens, d_model,
                                        d_output, nullptr, 0));

    ASSERT_EQ(hipMemcpy(rocm_fp16_output.data(), d_output,
                        num_tokens * d_model * sizeof(uint16_t), hipMemcpyDeviceToHost), hipSuccess);

    (void)hipFree(d_embed);
    (void)hipFree(d_token_ids);
    (void)hipFree(d_output);

    std::vector<float> rocm_output(num_tokens * d_model);
    for (size_t i = 0; i < rocm_fp16_output.size(); ++i)
    {
        rocm_output[i] = fp16_to_fp32(rocm_fp16_output[i]);
    }

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), num_tokens * d_model));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), num_tokens * d_model);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), num_tokens * d_model);

    std::cout << "  Embedding FP16 Small: cosine=" << cosine << ", L2_error=" << l2_error * 100 << "%" << std::endl;

    EXPECT_GE(cosine, 0.999);
    EXPECT_LE(l2_error, 0.02);
}

TEST_F(Test__ROCmEmbeddingParity, Embedding_FP16_Large)
{
    SKIP_IF_NO_ROCM();

    constexpr int num_tokens = 128;
    constexpr int d_model = 896;
    constexpr int vocab_size = 10000;

    auto embed_data = randomFP32(vocab_size * d_model);
    auto token_ids = randomTokenIds(num_tokens, vocab_size);

    std::vector<float> cpu_output(num_tokens * d_model, 0.0f);
    CPUEmbeddingKernelT<FP32Tensor> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.apply(embed_data.data(), token_ids.data(),
                                  num_tokens, d_model, cpu_output.data(),
                                  nullptr, -1));

    float *d_embed = nullptr;
    uint16_t *d_output = nullptr;
    int *d_token_ids = nullptr;
    std::vector<uint16_t> rocm_fp16_output(num_tokens * d_model);

    ASSERT_EQ(hipMalloc(&d_embed, vocab_size * d_model * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_token_ids, num_tokens * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, num_tokens * d_model * sizeof(uint16_t)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_embed, embed_data.data(),
                        vocab_size * d_model * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_token_ids, token_ids.data(),
                        num_tokens * sizeof(int), hipMemcpyHostToDevice), hipSuccess);

    ROCmEmbeddingKernelT rocm_kernel;
    ASSERT_TRUE(rocm_kernel.apply_fp16(d_embed, d_token_ids, num_tokens, d_model,
                                        d_output, nullptr, 0));

    ASSERT_EQ(hipMemcpy(rocm_fp16_output.data(), d_output,
                        num_tokens * d_model * sizeof(uint16_t), hipMemcpyDeviceToHost), hipSuccess);

    (void)hipFree(d_embed);
    (void)hipFree(d_token_ids);
    (void)hipFree(d_output);

    std::vector<float> rocm_output(num_tokens * d_model);
    for (size_t i = 0; i < rocm_fp16_output.size(); ++i)
    {
        rocm_output[i] = fp16_to_fp32(rocm_fp16_output[i]);
    }

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), num_tokens * d_model));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), num_tokens * d_model);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), num_tokens * d_model);

    std::cout << "  Embedding FP16 Large: cosine=" << cosine << ", L2_error=" << l2_error * 100 << "%" << std::endl;

    EXPECT_GE(cosine, 0.999);
    EXPECT_LE(l2_error, 0.02);
}

// ============================================================================
// Q8_1 Embedding Tests
// ============================================================================

// Use the actual Q8_1Block from BlockStructures.h via CPUTensors.h
// Q8_1Block has: uint16_t d (FP16 scale), int16_t sum_qs, int8_t qs[32]

/**
 * @brief Dequantize Q8_1 blocks to FP32
 * Q8_1Block has: uint16_t d (FP16 scale), int16_t sum_qs, int8_t qs[32]
 * Uses fp16_to_fp32 from FP16Utils.h
 */
void dequantize_q8_1(const Q8_1Block *blocks, float *output, int num_blocks)
{
    for (int b = 0; b < num_blocks; ++b)
    {
        const Q8_1Block &block = blocks[b];
        float scale = fp16_to_fp32(block.d);
        for (int i = 0; i < 32; ++i)
        {
            output[b * 32 + i] = static_cast<float>(block.qs[i]) * scale;
        }
    }
}

TEST_F(Test__ROCmEmbeddingParity, Embedding_Q8_1_Small)
{
    SKIP_IF_NO_ROCM();

    constexpr int num_tokens = 8;
    constexpr int d_model = 64; // Must be multiple of 32 for Q8_1
    constexpr int vocab_size = 1000;
    constexpr int blocks_per_row = d_model / 32;

    auto embed_data = randomFP32(vocab_size * d_model);
    auto token_ids = randomTokenIds(num_tokens, vocab_size);

    // CPU reference (FP32)
    std::vector<float> cpu_output(num_tokens * d_model, 0.0f);
    CPUEmbeddingKernelT<FP32Tensor> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.apply(embed_data.data(), token_ids.data(),
                                  num_tokens, d_model, cpu_output.data(),
                                  nullptr, -1));

    // GPU execution with Q8_1 output
    float *d_embed = nullptr;
    void *d_output = nullptr;
    int *d_token_ids = nullptr;
    std::vector<Q8_1Block> rocm_q8_1_output(num_tokens * blocks_per_row);

    ASSERT_EQ(hipMalloc(&d_embed, vocab_size * d_model * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_token_ids, num_tokens * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, num_tokens * blocks_per_row * sizeof(Q8_1Block)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_embed, embed_data.data(),
                        vocab_size * d_model * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_token_ids, token_ids.data(),
                        num_tokens * sizeof(int), hipMemcpyHostToDevice), hipSuccess);

    ROCmEmbeddingKernelT rocm_kernel;
    ASSERT_TRUE(rocm_kernel.apply_q8_1(d_embed, d_token_ids, num_tokens, d_model,
                                        d_output, nullptr, 0));

    ASSERT_EQ(hipMemcpy(rocm_q8_1_output.data(), d_output,
                        num_tokens * blocks_per_row * sizeof(Q8_1Block), hipMemcpyDeviceToHost), hipSuccess);

    (void)hipFree(d_embed);
    (void)hipFree(d_token_ids);
    (void)hipFree(d_output);

    // Dequantize Q8_1 to FP32 for comparison
    std::vector<float> rocm_output(num_tokens * d_model);
    dequantize_q8_1(rocm_q8_1_output.data(), rocm_output.data(), num_tokens * blocks_per_row);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), num_tokens * d_model));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), num_tokens * d_model);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), num_tokens * d_model);

    std::cout << "  Embedding Q8_1 Small: cosine=" << cosine << ", L2_error=" << l2_error * 100 << "%" << std::endl;

    // Q8_1 has lower precision due to quantization
    EXPECT_GE(cosine, 0.995);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__ROCmEmbeddingParity, Embedding_Q8_1_Large)
{
    SKIP_IF_NO_ROCM();

    constexpr int num_tokens = 128;
    constexpr int d_model = 896; // Must be multiple of 32 for Q8_1
    constexpr int vocab_size = 10000;
    constexpr int blocks_per_row = d_model / 32;

    auto embed_data = randomFP32(vocab_size * d_model);
    auto token_ids = randomTokenIds(num_tokens, vocab_size);

    std::vector<float> cpu_output(num_tokens * d_model, 0.0f);
    CPUEmbeddingKernelT<FP32Tensor> cpu_kernel;
    ASSERT_TRUE(cpu_kernel.apply(embed_data.data(), token_ids.data(),
                                  num_tokens, d_model, cpu_output.data(),
                                  nullptr, -1));

    float *d_embed = nullptr;
    void *d_output = nullptr;
    int *d_token_ids = nullptr;
    std::vector<Q8_1Block> rocm_q8_1_output(num_tokens * blocks_per_row);

    ASSERT_EQ(hipMalloc(&d_embed, vocab_size * d_model * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_token_ids, num_tokens * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_output, num_tokens * blocks_per_row * sizeof(Q8_1Block)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_embed, embed_data.data(),
                        vocab_size * d_model * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_token_ids, token_ids.data(),
                        num_tokens * sizeof(int), hipMemcpyHostToDevice), hipSuccess);

    ROCmEmbeddingKernelT rocm_kernel;
    ASSERT_TRUE(rocm_kernel.apply_q8_1(d_embed, d_token_ids, num_tokens, d_model,
                                        d_output, nullptr, 0));

    ASSERT_EQ(hipMemcpy(rocm_q8_1_output.data(), d_output,
                        num_tokens * blocks_per_row * sizeof(Q8_1Block), hipMemcpyDeviceToHost), hipSuccess);

    (void)hipFree(d_embed);
    (void)hipFree(d_token_ids);
    (void)hipFree(d_output);

    std::vector<float> rocm_output(num_tokens * d_model);
    dequantize_q8_1(rocm_q8_1_output.data(), rocm_output.data(), num_tokens * blocks_per_row);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), num_tokens * d_model));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), num_tokens * d_model);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), num_tokens * d_model);

    std::cout << "  Embedding Q8_1 Large: cosine=" << cosine << ", L2_error=" << l2_error * 100 << "%" << std::endl;

    EXPECT_GE(cosine, 0.995);
    EXPECT_LE(l2_error, 0.05);
}

#endif // HAVE_ROCM

// ============================================================================
// Fallback test when ROCm is not available
// ============================================================================

#ifndef HAVE_ROCM
TEST(Test__ROCmEmbeddingParity, NotAvailable)
{
    GTEST_SKIP() << "ROCm support not compiled in";
}
#endif
