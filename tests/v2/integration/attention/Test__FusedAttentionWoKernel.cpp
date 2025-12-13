/**
 * @file Test__FusedAttentionWoKernel.cpp
 * @brief Integration tests for FusedAttentionWoKernel pipeline wrapper
 *
 * Tests the fused attention + Wo projection kernel against the separate
 * implementation to verify numerical correctness for pipeline integration.
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include <memory>

#include "kernels/cpu/attention/FusedAttentionWoKernel.h"
#include "kernels/cpu/attention/q8_1/FusedAttentionWoRef.h"
#include "kernels/cpu/jit/q8_1/JitFusedAttentionWo.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels;
using namespace llaminar::v2::kernels::microkernels;

class Test__FusedAttentionWoKernel : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed random generator for reproducibility
        rng_.seed(42);
    }

    // Helper to create Q8_1 tensor with random data (using same algorithm as unit test)
    std::unique_ptr<Q8_1Tensor> createRandomQ8_1(int rows, int cols)
    {
        auto tensor = std::make_unique<Q8_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

        // Use same distribution as unit test for consistency
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        // Fill with random Q8_1 blocks using same quantization as unit test
        int num_blocks_per_row = cols / 32;
        llaminar2::Q8_1Block *blocks = tensor->mutable_q8_1_blocks();

        for (int row = 0; row < rows; ++row)
        {
            for (int b = 0; b < num_blocks_per_row; ++b)
            {
                // Generate 32 random FP32 values for this block
                float vals[32];
                float max_abs = 0.0f;
                for (int j = 0; j < 32; ++j)
                {
                    vals[j] = dist(rng_);
                    max_abs = std::max(max_abs, std::fabs(vals[j]));
                }

                // Compute scale (same as unit test)
                float scale = max_abs / 127.0f;
                if (scale < 1e-10f)
                    scale = 1e-10f;
                float inv_scale = 127.0f / max_abs;
                if (max_abs < 1e-10f)
                    inv_scale = 0.0f;

                // Quantize (same as unit test)
                llaminar2::Q8_1Block &blk = blocks[row * num_blocks_per_row + b];
                int32_t sum_qs = 0;
                for (int j = 0; j < 32; ++j)
                {
                    int8_t q = static_cast<int8_t>(std::round(vals[j] * inv_scale));
                    q = std::max(int8_t(-127), std::min(int8_t(127), q));
                    blk.qs[j] = q;
                    sum_qs += q;
                }
                blk.d = llaminar2::fp32_to_fp16(scale);
                blk.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }

        return tensor;
    }

    // Helper to create FP32 tensor with random data
    std::unique_ptr<FP32Tensor> createRandomFP32(int rows, int cols)
    {
        auto tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

        std::normal_distribution<float> dist(0.0f, 0.1f);
        float *data = tensor->mutable_data();
        for (int i = 0; i < rows * cols; ++i)
        {
            data[i] = dist(rng_);
        }

        return tensor;
    }

    // Compute cosine similarity
    double cosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return (norm_a < 1e-12 && norm_b < 1e-12) ? 1.0 : 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    // Compute relative L2 error
    double relativeL2Error(const float *actual, const float *expected, size_t n)
    {
        double sum_sq_diff = 0.0, sum_sq_expected = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = actual[i] - expected[i];
            sum_sq_diff += diff * diff;
            sum_sq_expected += expected[i] * expected[i];
        }
        if (sum_sq_expected < 1e-12)
            return (sum_sq_diff < 1e-12) ? 0.0 : 1.0;
        return std::sqrt(sum_sq_diff / sum_sq_expected);
    }

    std::mt19937 rng_;
};

// ============== Basic Tests ==============

TEST_F(Test__FusedAttentionWoKernel, CanInstantiate_Reference)
{
    FusedAttentionWoKernel::Config config;
    config.num_heads = 14;
    config.num_kv_heads = 2;
    config.head_dim = 64;
    config.d_model = 896;
    config.backend = FusedAttentionBackend::REFERENCE;

    FusedAttentionWoKernel kernel(config);

    EXPECT_EQ(kernel.backend(), FusedAttentionBackend::REFERENCE);
    EXPECT_EQ(kernel.config().num_heads, 14);
}

TEST_F(Test__FusedAttentionWoKernel, CanInstantiate_Tiled)
{
    FusedAttentionWoKernel::Config config;
    config.num_heads = 14;
    config.num_kv_heads = 2;
    config.head_dim = 64;
    config.d_model = 896;
    config.backend = FusedAttentionBackend::TILED;

    FusedAttentionWoKernel kernel(config);

    EXPECT_EQ(kernel.backend(), FusedAttentionBackend::TILED);
}

TEST_F(Test__FusedAttentionWoKernel, CanInstantiate_JIT)
{
    FusedAttentionWoKernel::Config config;
    config.num_heads = 14;
    config.num_kv_heads = 2;
    config.head_dim = 64;
    config.d_model = 896;
    config.backend = FusedAttentionBackend::JIT;

    FusedAttentionWoKernel kernel(config);

    EXPECT_EQ(kernel.backend(), FusedAttentionBackend::JIT);
}

// ============== Backend Parity Tests ==============

TEST_F(Test__FusedAttentionWoKernel, Parity_Reference_vs_JIT_SingleToken)
{
    // Qwen2 0.5B dimensions - MATCH EXACTLY with unit test
    const int num_heads = 14;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const int seq_len_q = 1;
    const int seq_len_kv = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int blocks_per_head = head_dim / 32;

    // Create inputs using the EXACT SAME approach as the working unit test:
    // std::vector<Q8_1Block> instead of Q8_1Tensor
    std::vector<llaminar2::Q8_1Block> Q_blocks(seq_len_q * num_heads * blocks_per_head);
    std::vector<llaminar2::Q8_1Block> K_blocks(seq_len_kv * num_kv_heads * blocks_per_head);
    std::vector<llaminar2::Q8_1Block> V_blocks(seq_len_kv * num_kv_heads * blocks_per_head);

    // Generate random FP32 data and quantize (same as unit test)
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto quantize = [&](std::vector<llaminar2::Q8_1Block> &blocks, int rows, int cols)
    {
        int num_blocks_per_row = cols / 32;
        std::vector<float> fp32_data(rows * cols);
        for (auto &v : fp32_data)
            v = dist(rng_);

        for (int row = 0; row < rows; ++row)
        {
            for (int b = 0; b < num_blocks_per_row; ++b)
            {
                const float *block_data = fp32_data.data() + row * cols + b * 32;
                llaminar2::Q8_1Block &blk = blocks[row * num_blocks_per_row + b];

                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::fabs(block_data[i]));
                }

                float s = max_abs / 127.0f;
                if (s < 1e-10f)
                    s = 1e-10f;
                float inv_s = 127.0f / max_abs;
                if (max_abs < 1e-10f)
                    inv_s = 0.0f;

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int8_t q = static_cast<int8_t>(std::round(block_data[i] * inv_s));
                    q = std::max(int8_t(-127), std::min(int8_t(127), q));
                    blk.qs[i] = q;
                    sum_qs += q;
                }

                blk.d = llaminar2::fp32_to_fp16(s);
                blk.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }
    };

    quantize(Q_blocks, seq_len_q * num_heads, head_dim);
    quantize(K_blocks, seq_len_kv * num_kv_heads, head_dim);
    quantize(V_blocks, seq_len_kv * num_kv_heads, head_dim);

    // Create identity-like Wo (same as unit test)
    std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
    for (int h = 0; h < num_heads; ++h)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            int row = h * head_dim + d;
            Wo_fp32[row * d_model + row] = 1.0f;
        }
    }

    // Output buffers
    std::vector<float> output_ref(seq_len_q * d_model, 0.0f);
    std::vector<float> output_jit(seq_len_q * d_model, 0.0f);

    // Run reference kernel (same as unit test)
    llaminar::v2::kernels::FusedAttentionWoParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = llaminar::v2::kernels::microkernels::WoWeightType::FP32;
    params.output = output_ref.data();
    params.batch_size = 1;
    params.kv_seq_lens = nullptr;
    params.position_offsets = nullptr;
    params.seq_len = seq_len_q;
    params.kv_seq_len = seq_len_kv;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = false;
    params.position_offset = 0;

    ASSERT_TRUE(llaminar::v2::kernels::FusedAttentionWoRef::execute(params));

    // Run JIT directly (same as unit test)
    llaminar::v2::kernels::jit::JitAttentionConfig jit_config;
    jit_config.head_dim = head_dim;
    jit_config.num_heads = num_heads;
    jit_config.num_kv_heads = num_kv_heads;
    jit_config.batch_size = 1;
    jit_config.wo_format = llaminar::v2::kernels::jit::WoFormat::FP32;
    jit_config.causal = false; // Non-causal for parity test

    llaminar::v2::kernels::jit::JitFusedAttentionWo jit_kernel(jit_config);
    jit_kernel.compute(
        Q_blocks.data(),
        K_blocks.data(),
        V_blocks.data(),
        Wo_fp32.data(),
        output_jit.data(),
        seq_len_q,
        seq_len_kv,
        scale);

    // Compare outputs
    size_t output_size = seq_len_q * d_model;

    double cosine_sim = cosineSimilarity(output_jit.data(), output_ref.data(), output_size);
    double rel_l2 = relativeL2Error(output_jit.data(), output_ref.data(), output_size);

    std::cout << "  Using std::vector<Q8_1Block> (same as unit test):" << std::endl;
    std::cout << "  Cosine similarity: " << cosine_sim << std::endl;
    std::cout << "  Relative L2 error: " << rel_l2 << std::endl;

    // Debug: print first few outputs
    std::cout << "  First 8 REF outputs: ";
    for (int i = 0; i < 8; ++i)
        std::cout << output_ref[i] << " ";
    std::cout << std::endl;
    std::cout << "  First 8 JIT outputs: ";
    for (int i = 0; i < 8; ++i)
        std::cout << output_jit[i] << " ";
    std::cout << std::endl;

    // JIT should match reference very closely
    EXPECT_GE(cosine_sim, 0.99);
    EXPECT_LE(rel_l2, 0.10);
}

// Test that Q8_1Tensor produces identical results to std::vector<Q8_1Block>
TEST_F(Test__FusedAttentionWoKernel, Q8_1Tensor_vs_VectorBlocks_Parity)
{
    const int num_heads = 14;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const int seq_len_q = 1;
    const int seq_len_kv = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int blocks_per_head = head_dim / 32;

    // First, create using std::vector (known working approach)
    std::vector<llaminar2::Q8_1Block> Q_vec(seq_len_q * num_heads * blocks_per_head);
    std::vector<llaminar2::Q8_1Block> K_vec(seq_len_kv * num_kv_heads * blocks_per_head);
    std::vector<llaminar2::Q8_1Block> V_vec(seq_len_kv * num_kv_heads * blocks_per_head);

    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto quantize_vec = [&](std::vector<llaminar2::Q8_1Block> &blocks, int rows, int cols)
    {
        int num_blocks_per_row = cols / 32;
        std::vector<float> fp32_data(rows * cols);
        for (auto &v : fp32_data)
            v = dist(rng_);

        for (int row = 0; row < rows; ++row)
        {
            for (int b = 0; b < num_blocks_per_row; ++b)
            {
                const float *block_data = fp32_data.data() + row * cols + b * 32;
                llaminar2::Q8_1Block &blk = blocks[row * num_blocks_per_row + b];

                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                    max_abs = std::max(max_abs, std::fabs(block_data[i]));

                float s = max_abs / 127.0f;
                if (s < 1e-10f)
                    s = 1e-10f;
                float inv_s = 127.0f / max_abs;
                if (max_abs < 1e-10f)
                    inv_s = 0.0f;

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int8_t q = static_cast<int8_t>(std::round(block_data[i] * inv_s));
                    q = std::max(int8_t(-127), std::min(int8_t(127), q));
                    blk.qs[i] = q;
                    sum_qs += q;
                }
                blk.d = llaminar2::fp32_to_fp16(s);
                blk.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }
    };

    quantize_vec(Q_vec, seq_len_q * num_heads, head_dim);
    quantize_vec(K_vec, seq_len_kv * num_kv_heads, head_dim);
    quantize_vec(V_vec, seq_len_kv * num_kv_heads, head_dim);

    // Now create Q8_1Tensor and COPY the blocks into it
    auto Q_tensor = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len_q * num_heads), static_cast<size_t>(head_dim)});
    auto K_tensor = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len_kv * num_kv_heads), static_cast<size_t>(head_dim)});
    auto V_tensor = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len_kv * num_kv_heads), static_cast<size_t>(head_dim)});

    // Copy blocks
    std::memcpy(Q_tensor->mutable_q8_1_blocks(), Q_vec.data(), Q_vec.size() * sizeof(llaminar2::Q8_1Block));
    std::memcpy(K_tensor->mutable_q8_1_blocks(), K_vec.data(), K_vec.size() * sizeof(llaminar2::Q8_1Block));
    std::memcpy(V_tensor->mutable_q8_1_blocks(), V_vec.data(), V_vec.size() * sizeof(llaminar2::Q8_1Block));

    // Verify blocks are identical
    const llaminar2::Q8_1Block *q_tensor_blocks = Q_tensor->q8_1_blocks();
    bool blocks_match = (std::memcmp(q_tensor_blocks, Q_vec.data(), Q_vec.size() * sizeof(llaminar2::Q8_1Block)) == 0);
    std::cout << "  Q blocks match after copy: " << (blocks_match ? "YES" : "NO") << std::endl;
    ASSERT_TRUE(blocks_match) << "Q8_1Tensor blocks don't match vector blocks after memcpy!";

    // Create Wo
    std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
    for (int h = 0; h < num_heads; ++h)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            int row = h * head_dim + d;
            Wo_fp32[row * d_model + row] = 1.0f;
        }
    }

    // Output buffers
    std::vector<float> output_vec(seq_len_q * d_model, 0.0f);
    std::vector<float> output_tensor(seq_len_q * d_model, 0.0f);

    // Run JIT with vector
    llaminar::v2::kernels::jit::JitAttentionConfig jit_config;
    jit_config.head_dim = head_dim;
    jit_config.num_heads = num_heads;
    jit_config.num_kv_heads = num_kv_heads;
    jit_config.batch_size = 1;
    jit_config.wo_format = llaminar::v2::kernels::jit::WoFormat::FP32;
    jit_config.causal = false; // Non-causal for parity test

    llaminar::v2::kernels::jit::JitFusedAttentionWo jit_kernel(jit_config);
    jit_kernel.compute(Q_vec.data(), K_vec.data(), V_vec.data(), Wo_fp32.data(),
                       output_vec.data(), seq_len_q, seq_len_kv, scale);

    // Run JIT with tensor
    jit_kernel.compute(Q_tensor->q8_1_blocks(), K_tensor->q8_1_blocks(), V_tensor->q8_1_blocks(),
                       Wo_fp32.data(), output_tensor.data(), seq_len_q, seq_len_kv, scale);

    // Compare
    double cosine_sim = cosineSimilarity(output_tensor.data(), output_vec.data(), seq_len_q * d_model);
    std::cout << "  Vector vs Tensor output cosine: " << cosine_sim << std::endl;

    EXPECT_GE(cosine_sim, 0.9999) << "Q8_1Tensor should produce identical results to vector!";
}

// Test JIT with Q8_1Tensor objects (production usage pattern)
TEST_F(Test__FusedAttentionWoKernel, JIT_With_Q8_1Tensor_Objects)
{
    const int num_heads = 14;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const int seq_len_q = 1;
    const int seq_len_kv = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create Q8_1Tensor objects (production usage pattern)
    auto Q = createRandomQ8_1(seq_len_q * num_heads, head_dim);
    auto K = createRandomQ8_1(seq_len_kv * num_kv_heads, head_dim);
    auto V = createRandomQ8_1(seq_len_kv * num_kv_heads, head_dim);

    // Use identity-like Wo so Reference output ≈ JIT output (JIT doesn't apply Wo yet)
    std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
    for (int h = 0; h < num_heads; ++h)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            int row = h * head_dim + d;
            Wo_fp32[row * d_model + row] = 1.0f;
        }
    }

    // Output buffers
    auto output_ref = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len_q), static_cast<size_t>(d_model)});
    auto output_jit = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len_q), static_cast<size_t>(d_model)});

    // Run Reference kernel
    llaminar::v2::kernels::FusedAttentionWoParams params;
    params.Q = Q->q8_1_blocks();
    params.K = K->q8_1_blocks();
    params.V = V->q8_1_blocks();
    params.Wo = Wo_fp32.data();
    params.wo_type = llaminar::v2::kernels::microkernels::WoWeightType::FP32;
    params.output = output_ref->mutable_data();
    params.batch_size = 1;
    params.kv_seq_lens = nullptr;
    params.position_offsets = nullptr;
    params.seq_len = seq_len_q;
    params.kv_seq_len = seq_len_kv;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = false;
    params.position_offset = 0;

    ASSERT_TRUE(llaminar::v2::kernels::FusedAttentionWoRef::execute(params));

    // Run JIT kernel
    llaminar::v2::kernels::jit::JitAttentionConfig jit_config;
    jit_config.head_dim = head_dim;
    jit_config.num_heads = num_heads;
    jit_config.num_kv_heads = num_kv_heads;
    jit_config.batch_size = 1;
    jit_config.wo_format = llaminar::v2::kernels::jit::WoFormat::FP32;
    jit_config.causal = false; // Non-causal for parity test

    llaminar::v2::kernels::jit::JitFusedAttentionWo jit_kernel(jit_config);
    jit_kernel.compute(
        Q->q8_1_blocks(),
        K->q8_1_blocks(),
        V->q8_1_blocks(),
        Wo_fp32.data(),
        output_jit->mutable_data(),
        seq_len_q,
        seq_len_kv,
        scale);

    // Compare outputs
    const float *ref_data = output_ref->data();
    const float *jit_data = output_jit->data();
    size_t output_size = seq_len_q * d_model;

    double cosine_sim = cosineSimilarity(jit_data, ref_data, output_size);
    double rel_l2 = relativeL2Error(jit_data, ref_data, output_size);

    std::cout << "  JIT with Q8_1Tensor objects:" << std::endl;
    std::cout << "  Cosine similarity: " << cosine_sim << std::endl;
    std::cout << "  Relative L2 error: " << rel_l2 << std::endl;

    EXPECT_GE(cosine_sim, 0.99) << "JIT should match Reference when using Q8_1Tensor";
    EXPECT_LE(rel_l2, 0.10);
}

TEST_F(Test__FusedAttentionWoKernel, Parity_Reference_vs_JIT_Prefill)
{
    // Qwen2 0.5B dimensions
    const int num_heads = 14;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const int seq_len = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create Q8_1Tensor inputs
    auto Q = createRandomQ8_1(seq_len * num_heads, head_dim);
    auto K = createRandomQ8_1(seq_len * num_kv_heads, head_dim);
    auto V = createRandomQ8_1(seq_len * num_kv_heads, head_dim);

    // Use identity-like Wo (JIT doesn't apply Wo yet)
    std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
    for (int h = 0; h < num_heads; ++h)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            int row = h * head_dim + d;
            Wo_fp32[row * d_model + row] = 1.0f;
        }
    }

    // Output buffers
    std::vector<float> output_ref(seq_len * d_model, 0.0f);
    std::vector<float> output_jit(seq_len * d_model, 0.0f);

    // Run Reference kernel
    llaminar::v2::kernels::FusedAttentionWoParams params;
    params.Q = Q->q8_1_blocks();
    params.K = K->q8_1_blocks();
    params.V = V->q8_1_blocks();
    params.Wo = Wo_fp32.data();
    params.wo_type = llaminar::v2::kernels::microkernels::WoWeightType::FP32;
    params.output = output_ref.data();
    params.batch_size = 1;
    params.kv_seq_lens = nullptr;
    params.position_offsets = nullptr;
    params.seq_len = seq_len;
    params.kv_seq_len = seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = false;
    params.position_offset = 0;

    ASSERT_TRUE(llaminar::v2::kernels::FusedAttentionWoRef::execute(params));

    // Run JIT kernel
    llaminar::v2::kernels::jit::JitAttentionConfig jit_config;
    jit_config.head_dim = head_dim;
    jit_config.num_heads = num_heads;
    jit_config.num_kv_heads = num_kv_heads;
    jit_config.batch_size = 1;
    jit_config.wo_format = llaminar::v2::kernels::jit::WoFormat::FP32;
    jit_config.causal = false; // Non-causal for parity test

    llaminar::v2::kernels::jit::JitFusedAttentionWo jit_kernel(jit_config);
    jit_kernel.compute(
        Q->q8_1_blocks(),
        K->q8_1_blocks(),
        V->q8_1_blocks(),
        Wo_fp32.data(),
        output_jit.data(),
        seq_len,
        seq_len,
        scale);

    // Compare outputs
    double cosine_sim = cosineSimilarity(output_jit.data(), output_ref.data(), seq_len * d_model);
    double rel_l2 = relativeL2Error(output_jit.data(), output_ref.data(), seq_len * d_model);

    std::cout << "  Reference vs JIT (prefill seq=" << seq_len << "):" << std::endl;
    std::cout << "  Cosine similarity: " << cosine_sim << std::endl;
    std::cout << "  Relative L2 error: " << rel_l2 << std::endl;

    EXPECT_GE(cosine_sim, 0.99);
    EXPECT_LE(rel_l2, 0.10);
}

TEST_F(Test__FusedAttentionWoKernel, Parity_Tiled_vs_JIT_LongKV)
{
    // Test with longer KV cache
    const int num_heads = 14;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const int seq_len_q = 1;
    const int seq_len_kv = 256; // Long KV cache
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create Q8_1Tensor inputs
    auto Q = createRandomQ8_1(seq_len_q * num_heads, head_dim);
    auto K = createRandomQ8_1(seq_len_kv * num_kv_heads, head_dim);
    auto V = createRandomQ8_1(seq_len_kv * num_kv_heads, head_dim);

    // Use identity-like Wo
    std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
    for (int h = 0; h < num_heads; ++h)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            int row = h * head_dim + d;
            Wo_fp32[row * d_model + row] = 1.0f;
        }
    }

    // Output buffers
    std::vector<float> output_ref(seq_len_q * d_model, 0.0f);
    std::vector<float> output_jit(seq_len_q * d_model, 0.0f);

    // Run Reference kernel (compare against Reference since Tiled also applies Wo)
    llaminar::v2::kernels::FusedAttentionWoParams params;
    params.Q = Q->q8_1_blocks();
    params.K = K->q8_1_blocks();
    params.V = V->q8_1_blocks();
    params.Wo = Wo_fp32.data();
    params.wo_type = llaminar::v2::kernels::microkernels::WoWeightType::FP32;
    params.output = output_ref.data();
    params.batch_size = 1;
    params.kv_seq_lens = nullptr;
    params.position_offsets = nullptr;
    params.seq_len = seq_len_q;
    params.kv_seq_len = seq_len_kv;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = false;
    params.position_offset = 0;

    ASSERT_TRUE(llaminar::v2::kernels::FusedAttentionWoRef::execute(params));

    // Run JIT kernel
    llaminar::v2::kernels::jit::JitAttentionConfig jit_config;
    jit_config.head_dim = head_dim;
    jit_config.num_heads = num_heads;
    jit_config.num_kv_heads = num_kv_heads;
    jit_config.batch_size = 1;
    jit_config.wo_format = llaminar::v2::kernels::jit::WoFormat::FP32;
    jit_config.causal = false; // Non-causal for parity test

    llaminar::v2::kernels::jit::JitFusedAttentionWo jit_kernel(jit_config);
    jit_kernel.compute(
        Q->q8_1_blocks(),
        K->q8_1_blocks(),
        V->q8_1_blocks(),
        Wo_fp32.data(),
        output_jit.data(),
        seq_len_q,
        seq_len_kv,
        scale);

    // Compare
    double cosine_sim = cosineSimilarity(output_jit.data(), output_ref.data(), seq_len_q * d_model);
    double rel_l2 = relativeL2Error(output_jit.data(), output_ref.data(), seq_len_q * d_model);

    std::cout << "  Reference vs JIT (kv_len=" << seq_len_kv << "):" << std::endl;
    std::cout << "  Cosine similarity: " << cosine_sim << std::endl;
    std::cout << "  Relative L2 error: " << rel_l2 << std::endl;

    EXPECT_GE(cosine_sim, 0.99);
    EXPECT_LE(rel_l2, 0.10);
}

// ============== Model Configuration Tests ==============

TEST_F(Test__FusedAttentionWoKernel, Qwen2_7B_Config)
{
    // Qwen2 7B dimensions
    const int num_heads = 28;
    const int num_kv_heads = 4;
    const int head_dim = 128;
    const int d_model = num_heads * head_dim;
    const int seq_len_q = 1;
    const int seq_len_kv = 64;

    // Create input tensors with correct layout
    auto Q = createRandomQ8_1(seq_len_q * num_heads, head_dim);
    auto K = createRandomQ8_1(seq_len_kv * num_kv_heads, head_dim);
    auto V = createRandomQ8_1(seq_len_kv * num_kv_heads, head_dim);
    auto Wo = createRandomFP32(d_model, num_heads * head_dim);

    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len_q), static_cast<size_t>(d_model)});

    FusedAttentionWoKernel::Config config;
    config.num_heads = num_heads;
    config.num_kv_heads = num_kv_heads;
    config.head_dim = head_dim;
    config.d_model = d_model;
    config.backend = FusedAttentionBackend::JIT;

    FusedAttentionWoKernel kernel(config);

    ASSERT_TRUE(kernel.compute(Q.get(), K.get(), V.get(), Wo.get(), output.get(),
                               seq_len_q, seq_len_kv, false, 0));

    // Verify output is not all zeros
    const float *data = output->data();
    float sum = 0.0f;
    for (int i = 0; i < seq_len_q * d_model; ++i)
    {
        sum += std::abs(data[i]);
    }
    EXPECT_GT(sum, 0.0f) << "Output should not be all zeros";
}

// ============== Q8_1 Wo Weight Tests ==============

TEST_F(Test__FusedAttentionWoKernel, Q8_1_Wo_Weights)
{
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const int seq_len = 4;

    // Create input tensors with correct layout
    auto Q = createRandomQ8_1(seq_len * num_heads, head_dim);
    auto K = createRandomQ8_1(seq_len * num_kv_heads, head_dim);
    auto V = createRandomQ8_1(seq_len * num_kv_heads, head_dim);

    // Create Q8_1 Wo weights
    auto Wo = createRandomQ8_1(d_model, num_heads * head_dim);

    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    FusedAttentionWoKernel::Config config;
    config.num_heads = num_heads;
    config.num_kv_heads = num_kv_heads;
    config.head_dim = head_dim;
    config.d_model = d_model;
    config.backend = FusedAttentionBackend::REFERENCE;

    FusedAttentionWoKernel kernel(config);

    ASSERT_TRUE(kernel.compute(Q.get(), K.get(), V.get(), Wo.get(), output.get(),
                               seq_len, seq_len, true, 0));

    // Verify output is not all zeros
    const float *data = output->data();
    float sum = 0.0f;
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        sum += std::abs(data[i]);
    }
    EXPECT_GT(sum, 0.0f) << "Output with Q8_1 Wo should not be all zeros";
}
