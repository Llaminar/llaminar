/**
 * @file Test__CPUAttentionT.cpp
 * @brief Unit tests for CPUAttentionT template-based attention kernel
 * @author David Sanftenberg
 *
 * Tests:
 * 1. FP32Tensor instantiation works
 * 2. Basic attention computation (small test case)
 * 3. Interface compatibility (supports_device, workspace allocation)
 * 4. GQA broadcasting
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>

#include "v2/kernels/cpu/CPUAttentionT.h"
#include "v2/tensors/Tensors.h"

using namespace llaminar2;

namespace
{
    constexpr float FP32_TOLERANCE = 1e-5f;
    constexpr float BF16_TOLERANCE = 5e-3f; // BF16: 7-bit mantissa precision
    constexpr float FP16_TOLERANCE = 5e-4f; // FP16: 10-bit mantissa precision

    // Helper: Initialize FP32 tensor with sequential values
    void init_sequential(float *data, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            data[i] = static_cast<float>(i) / 10.0f;
        }
    }

    // Helper: Initialize BF16 tensor with sequential values
    void init_sequential_bf16(uint16_t *data, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            float val = static_cast<float>(i) / 10.0f;
            // Convert float to BF16 (truncate lower 16 bits)
            uint32_t tmp;
            std::memcpy(&tmp, &val, sizeof(float));
            data[i] = static_cast<uint16_t>(tmp >> 16);
        }
    }

    // Helper: Initialize FP16 tensor with sequential values
    void init_sequential_fp16(uint16_t *data, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            float val = static_cast<float>(i) / 10.0f;
            // Simple FP16 conversion (note: not IEEE 754 compliant, for testing only)
            // For production, use proper FP16 conversion
            uint32_t tmp;
            std::memcpy(&tmp, &val, sizeof(float));
            // Simplified: just truncate (real FP16 has different exponent bias)
            data[i] = static_cast<uint16_t>(tmp >> 16);
        }
    }

    // Helper: Convert BF16 to FP32
    float bf16_to_fp32(uint16_t bf16_val)
    {
        uint32_t tmp = static_cast<uint32_t>(bf16_val) << 16;
        float result;
        std::memcpy(&result, &tmp, sizeof(float));
        return result;
    }

    // Helper: Check FP32 output is not all zeros
    bool has_nonzero(const float *data, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            if (std::abs(data[i]) > 1e-9f)
            {
                return true;
            }
        }
        return false;
    }

    // Helper: Check BF16 output is not all zeros (via FP32 conversion)
    bool has_nonzero_bf16(const uint16_t *data, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            float val = bf16_to_fp32(data[i]);
            if (std::abs(val) > 1e-9f)
            {
                return true;
            }
        }
        return false;
    }

    // Helper: Check FP16 output is not all zeros (via FP32 conversion)
    bool has_nonzero_fp16(const uint16_t *data, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            float val = bf16_to_fp32(data[i]); // Simplified - reuse BF16 conversion
            if (std::abs(val) > 1e-9f)
            {
                return true;
            }
        }
        return false;
    }

} // anonymous namespace

// ============================================================================
// FP32Tensor Tests
// ============================================================================

TEST(CPUAttentionT_FP32, InstantiationWorks)
{
    CPUAttentionT<FP32Tensor> attention;
    EXPECT_TRUE(attention.supports_device(-1)) << "Should support CPU (device_idx=-1)";
    EXPECT_FALSE(attention.supports_device(0)) << "Should NOT support GPU";
}

TEST(CPUAttentionT_FP32, BasicAttentionComputation)
{
    // Small test: 2 tokens, 1 head, 4 dims
    const int seq_len = 2;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    // Allocate tensors
    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    // Initialize with sequential values
    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size());
    init_sequential(V.data(), V.size());

    // Create attention kernel
    CPUAttentionT<FP32Tensor> attention;

    // Compute attention
    bool success = attention.compute(
        Q.data(), K.data(), V.data(), output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false,                              // causal
        -1,                                 // window_size
        nullptr, nullptr, nullptr, nullptr, // workspaces (auto-allocate)
        false,                              // use_bf16
        nullptr, -1);

    EXPECT_TRUE(success) << "Attention computation should succeed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size())) << "Output should be non-zero";
}

TEST(CPUAttentionT_FP32, CausalMasking)
{
    // Test causal masking: future tokens should not affect past tokens
    const int seq_len = 4;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 8;

    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size());
    init_sequential(V.data(), V.size());

    CPUAttentionT<FP32Tensor> attention;

    bool success = attention.compute(
        Q.data(), K.data(), V.data(), output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal=true
        -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);

    EXPECT_TRUE(success) << "Causal attention computation should succeed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size())) << "Output should be non-zero";

    // With causal masking, output should differ from non-causal
    // (Not testing exact values, just that it runs without crashing)
}

TEST(CPUAttentionT_FP32, MultiHeadAttention)
{
    // Test with multiple heads (MHA)
    const int seq_len = 3;
    const int n_heads = 2;
    const int n_kv_heads = 2; // MHA: n_heads == n_kv_heads
    const int head_dim = 4;

    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size());
    init_sequential(V.data(), V.size());

    CPUAttentionT<FP32Tensor> attention;

    bool success = attention.compute(
        Q.data(), K.data(), V.data(), output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);

    EXPECT_TRUE(success) << "Multi-head attention should succeed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size()));
}

TEST(CPUAttentionT_FP32, GroupedQueryAttention)
{
    // Test GQA: n_heads > n_kv_heads (KV broadcasting)
    const int seq_len = 2;
    const int n_heads = 4;
    const int n_kv_heads = 2; // GQA: 4 query heads, 2 KV heads
    const int head_dim = 4;

    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size());
    init_sequential(V.data(), V.size());

    CPUAttentionT<FP32Tensor> attention;

    bool success = attention.compute(
        Q.data(), K.data(), V.data(), output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);

    EXPECT_TRUE(success) << "GQA should succeed (with KV broadcasting)";
    EXPECT_TRUE(has_nonzero(output.data(), output.size()));
}

TEST(CPUAttentionT_FP32, WorkspaceProvided)
{
    // Test with pre-allocated workspaces
    const int seq_len = 2;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size());
    init_sequential(V.data(), V.size());

    // Allocate workspaces manually
    FP32Tensor scores_workspace({static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)});
    FP32Tensor buffer_workspace({static_cast<size_t>(seq_len * head_dim * 2)});
    FP32Tensor context_workspace({static_cast<size_t>(seq_len * head_dim)});

    CPUAttentionT<FP32Tensor> attention;

    bool success = attention.compute(
        Q.data(), K.data(), V.data(), output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        &scores_workspace, &buffer_workspace, &context_workspace, nullptr,
        false, nullptr, -1);

    EXPECT_TRUE(success) << "Attention with provided workspaces should succeed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size()));
}

TEST(CPUAttentionT_FP32, InvalidDevice)
{
    const int seq_len = 2;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size());
    init_sequential(V.data(), V.size());

    CPUAttentionT<FP32Tensor> attention;

    // Try with device_idx=0 (GPU) - should fail
    bool success = attention.compute(
        Q.data(), K.data(), V.data(), output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, 0); // device_idx=0 (GPU)

    EXPECT_FALSE(success) << "Should fail on GPU device (CPU-only kernel)";
}

TEST(CPUAttentionT_FP32, NullPointerInputs)
{
    const int seq_len = 2;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    CPUAttentionT<FP32Tensor> attention;

    // Null Q
    bool success = attention.compute(
        nullptr, nullptr, nullptr, output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);

    EXPECT_FALSE(success) << "Should fail with null input pointers";
}

TEST(CPUAttentionT_FP32, InvalidDimensions)
{
    const int seq_len = 2;
    const int n_heads = 3;    // 3 query heads
    const int n_kv_heads = 2; // 2 KV heads - NOT divisible!
    const int head_dim = 4;

    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    CPUAttentionT<FP32Tensor> attention;

    // n_heads (3) not divisible by n_kv_heads (2) - should fail
    bool success = attention.compute(
        Q.data(), K.data(), V.data(), output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);

    EXPECT_FALSE(success) << "Should fail when n_heads not divisible by n_kv_heads";
}

// ============================================================================
// BF16Tensor Tests
// ============================================================================
// CRITICAL NOTES:
// 1. GEMM kernels ALWAYS output FP32 (not BF16!)
// 2. Output buffer MUST be std::vector<float>, not std::vector<uint16_t>
// 3. Input conversion: FP32 → BF16 for storage, passed as reinterpret_cast<float*>
// 4. Workspaces are ALWAYS FP32Tensor (even for BF16 inputs)

TEST(CPUAttentionT_BF16, InstantiationWorks)
{
    CPUAttentionT<BF16Tensor> attention;
    EXPECT_TRUE(attention.supports_device(-1));
    EXPECT_FALSE(attention.supports_device(0)); // CPU only
}

TEST(CPUAttentionT_BF16, BasicAttentionComputation)
{
    // Test dimensions
    const int seq_len = 2;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    // 1. Create FP32 reference inputs
    std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
    std::vector<float> K_fp32(seq_len * n_kv_heads * head_dim);
    std::vector<float> V_fp32(seq_len * n_kv_heads * head_dim);
    init_sequential(Q_fp32.data(), Q_fp32.size());
    init_sequential(K_fp32.data(), K_fp32.size());
    init_sequential(V_fp32.data(), V_fp32.size());

    // 2. Convert to BF16 (storage format)
    std::vector<uint16_t> Q_bf16(Q_fp32.size());
    std::vector<uint16_t> K_bf16(K_fp32.size());
    std::vector<uint16_t> V_bf16(V_fp32.size());
    init_sequential_bf16(Q_bf16.data(), Q_bf16.size());
    init_sequential_bf16(K_bf16.data(), K_bf16.size());
    init_sequential_bf16(V_bf16.data(), V_bf16.size());

    // 3. CRITICAL: Output buffer MUST be FP32 (GEMM outputs FP32!)
    std::vector<float> output(seq_len * n_heads * head_dim);

    // 4. Compute attention
    CPUAttentionT<BF16Tensor> attention;
    bool success = attention.compute(
        reinterpret_cast<float *>(Q_bf16.data()), // Type-erased interface
        reinterpret_cast<float *>(K_bf16.data()),
        reinterpret_cast<float *>(V_bf16.data()),
        output.data(), // FP32 output
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        false, // causal
        -1,    // device_idx
        nullptr,
        nullptr);

    ASSERT_TRUE(success) << "BF16 attention computation failed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size())) << "BF16 output is all zeros";
}

TEST(CPUAttentionT_BF16, CausalMasking)
{
    const int seq_len = 4;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    // FP32 reference
    std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
    std::vector<float> K_fp32(seq_len * n_kv_heads * head_dim);
    std::vector<float> V_fp32(seq_len * n_kv_heads * head_dim);
    init_sequential(Q_fp32.data(), Q_fp32.size());
    init_sequential(K_fp32.data(), K_fp32.size());
    init_sequential(V_fp32.data(), V_fp32.size());

    // Convert to BF16
    std::vector<uint16_t> Q_bf16(Q_fp32.size());
    std::vector<uint16_t> K_bf16(K_fp32.size());
    std::vector<uint16_t> V_bf16(V_fp32.size());
    init_sequential_bf16(Q_bf16.data(), Q_bf16.size());
    init_sequential_bf16(K_bf16.data(), K_bf16.size());
    init_sequential_bf16(V_bf16.data(), V_bf16.size());

    // FP32 output
    std::vector<float> output(seq_len * n_heads * head_dim);

    CPUAttentionT<BF16Tensor> attention;
    bool success = attention.compute(
        reinterpret_cast<float *>(Q_bf16.data()),
        reinterpret_cast<float *>(K_bf16.data()),
        reinterpret_cast<float *>(V_bf16.data()),
        output.data(),
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        true, // causal=true
        -1,
        nullptr,
        nullptr);

    ASSERT_TRUE(success) << "BF16 causal attention failed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size())) << "BF16 causal output is all zeros";
}

TEST(CPUAttentionT_BF16, MultiHeadAttention)
{
    const int seq_len = 3;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;

    std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
    std::vector<float> K_fp32(seq_len * n_kv_heads * head_dim);
    std::vector<float> V_fp32(seq_len * n_kv_heads * head_dim);
    init_sequential(Q_fp32.data(), Q_fp32.size());
    init_sequential(K_fp32.data(), K_fp32.size());
    init_sequential(V_fp32.data(), V_fp32.size());

    std::vector<uint16_t> Q_bf16(Q_fp32.size());
    std::vector<uint16_t> K_bf16(K_fp32.size());
    std::vector<uint16_t> V_bf16(V_fp32.size());
    init_sequential_bf16(Q_bf16.data(), Q_bf16.size());
    init_sequential_bf16(K_bf16.data(), K_bf16.size());
    init_sequential_bf16(V_bf16.data(), V_bf16.size());

    std::vector<float> output(seq_len * n_heads * head_dim);

    CPUAttentionT<BF16Tensor> attention;
    bool success = attention.compute(
        reinterpret_cast<float *>(Q_bf16.data()),
        reinterpret_cast<float *>(K_bf16.data()),
        reinterpret_cast<float *>(V_bf16.data()),
        output.data(),
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        false,
        -1,
        nullptr,
        nullptr);

    ASSERT_TRUE(success) << "BF16 MHA failed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size())) << "BF16 MHA output is all zeros";
}

TEST(CPUAttentionT_BF16, GroupedQueryAttention)
{
    const int seq_len = 2;
    const int n_heads = 4;    // Query heads
    const int n_kv_heads = 2; // KV heads (GQA)
    const int head_dim = 4;

    std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
    std::vector<float> K_fp32(seq_len * n_kv_heads * head_dim);
    std::vector<float> V_fp32(seq_len * n_kv_heads * head_dim);
    init_sequential(Q_fp32.data(), Q_fp32.size());
    init_sequential(K_fp32.data(), K_fp32.size());
    init_sequential(V_fp32.data(), V_fp32.size());

    std::vector<uint16_t> Q_bf16(Q_fp32.size());
    std::vector<uint16_t> K_bf16(K_fp32.size());
    std::vector<uint16_t> V_bf16(V_fp32.size());
    init_sequential_bf16(Q_bf16.data(), Q_bf16.size());
    init_sequential_bf16(K_bf16.data(), K_bf16.size());
    init_sequential_bf16(V_bf16.data(), V_bf16.size());

    std::vector<float> output(seq_len * n_heads * head_dim);

    CPUAttentionT<BF16Tensor> attention;
    bool success = attention.compute(
        reinterpret_cast<float *>(Q_bf16.data()),
        reinterpret_cast<float *>(K_bf16.data()),
        reinterpret_cast<float *>(V_bf16.data()),
        output.data(),
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        false,
        -1,
        nullptr,
        nullptr);

    ASSERT_TRUE(success) << "BF16 GQA failed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size())) << "BF16 GQA output is all zeros";
}

TEST(CPUAttentionT_BF16, WorkspaceProvided)
{
    const int seq_len = 2;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
    std::vector<float> K_fp32(seq_len * n_kv_heads * head_dim);
    std::vector<float> V_fp32(seq_len * n_kv_heads * head_dim);
    init_sequential(Q_fp32.data(), Q_fp32.size());
    init_sequential(K_fp32.data(), K_fp32.size());
    init_sequential(V_fp32.data(), V_fp32.size());

    std::vector<uint16_t> Q_bf16(Q_fp32.size());
    std::vector<uint16_t> K_bf16(K_fp32.size());
    std::vector<uint16_t> V_bf16(V_fp32.size());
    init_sequential_bf16(Q_bf16.data(), Q_bf16.size());
    init_sequential_bf16(K_bf16.data(), K_bf16.size());
    init_sequential_bf16(V_bf16.data(), V_bf16.size());

    std::vector<float> output(seq_len * n_heads * head_dim);

    // CRITICAL: Workspaces MUST be FP32Tensor (even for BF16 inputs!)
    auto workspace_scores = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)});
    auto workspace_weights = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)});

    CPUAttentionT<BF16Tensor> attention;
    bool success = attention.compute(
        reinterpret_cast<float *>(Q_bf16.data()),
        reinterpret_cast<float *>(K_bf16.data()),
        reinterpret_cast<float *>(V_bf16.data()),
        output.data(),
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        false,
        -1,
        workspace_scores.get(),
        workspace_weights.get());

    ASSERT_TRUE(success) << "BF16 workspace test failed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size())) << "BF16 workspace output is all zeros";
}

// ============================================================================
// FP16Tensor Tests
// ============================================================================
// NOTE: FP16 GEMM kernel not yet implemented - only test instantiation

TEST(CPUAttentionT_FP16, InstantiationWorks)
{
    CPUAttentionT<FP16Tensor> attention;
    EXPECT_TRUE(attention.supports_device(-1));
    EXPECT_FALSE(attention.supports_device(0)); // CPU only
}

// TODO: FP16 computation tests disabled until FP16GemmKernel is implemented
// Will add: BasicAttentionComputation, CausalMasking, MultiHeadAttention, GroupedQueryAttention

// ============================================================================
// INT32Tensor Tests
// ============================================================================
// NOTE: INT32 has limited support (createGemm returns nullptr)
// Only test instantiation and error handling

TEST(CPUAttentionT_INT32, InstantiationWorks)
{
    CPUAttentionT<INT32Tensor> attention;
    EXPECT_TRUE(attention.supports_device(-1));
    EXPECT_FALSE(attention.supports_device(0)); // CPU only
}

// ============================================================================
// Q8_0Tensor Tests
// ============================================================================

TEST(CPUAttentionT_Q8_0, InstantiationWorks)
{
    CPUAttentionT<Q8_0Tensor> attention;
    EXPECT_TRUE(attention.supports_device(-1)) << "Should support CPU (device_idx=-1)";
    EXPECT_FALSE(attention.supports_device(0)) << "Should NOT support GPU";
}

TEST(CPUAttentionT_Q8_0, BasicAttentionComputation)
{
    // Small test: 2 tokens, 1 head, 4 dims (head_dim must be multiple of 32 for Q8_0 block alignment)
    const int seq_len = 2;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 32; // Changed from 4 to 32 for Q8_0 block alignment

    // Create FP32 data for quantization
    std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
    std::vector<float> K_fp32(seq_len * n_kv_heads * head_dim);
    std::vector<float> V_fp32(seq_len * n_kv_heads * head_dim);
    init_sequential(Q_fp32.data(), Q_fp32.size());
    init_sequential(K_fp32.data(), K_fp32.size());
    init_sequential(V_fp32.data(), V_fp32.size());

    // Quantize to Q8_0 using FP32Tensor conversion (with 2D shape for to_fp32_via_blocks)
    FP32Tensor Q_fp32_tensor({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    FP32Tensor K_fp32_tensor({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
    FP32Tensor V_fp32_tensor({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});

    std::memcpy(Q_fp32_tensor.mutable_data(), Q_fp32.data(), Q_fp32.size() * sizeof(float));
    std::memcpy(K_fp32_tensor.mutable_data(), K_fp32.data(), K_fp32.size() * sizeof(float));
    std::memcpy(V_fp32_tensor.mutable_data(), V_fp32.data(), V_fp32.size() * sizeof(float));

    // Convert to Q8_0
    size_t Q_n_blocks = (Q_fp32.size() + 31) / 32;
    size_t K_n_blocks = (K_fp32.size() + 31) / 32;
    size_t V_n_blocks = (V_fp32.size() + 31) / 32;

    std::vector<Q8_0Block> Q_blocks(Q_n_blocks);
    std::vector<Q8_0Block> K_blocks(K_n_blocks);
    std::vector<Q8_0Block> V_blocks(V_n_blocks);

    Q_fp32_tensor.to_q8_0(Q_blocks.data());
    K_fp32_tensor.to_q8_0(K_blocks.data());
    V_fp32_tensor.to_q8_0(V_blocks.data());

    // Convert blocks to raw bytes for Q8_0Tensor construction
    std::vector<uint8_t> Q_raw(reinterpret_cast<uint8_t *>(Q_blocks.data()),
                               reinterpret_cast<uint8_t *>(Q_blocks.data()) + Q_n_blocks * 34);
    std::vector<uint8_t> K_raw(reinterpret_cast<uint8_t *>(K_blocks.data()),
                               reinterpret_cast<uint8_t *>(K_blocks.data()) + K_n_blocks * 34);
    std::vector<uint8_t> V_raw(reinterpret_cast<uint8_t *>(V_blocks.data()),
                               reinterpret_cast<uint8_t *>(V_blocks.data()) + V_n_blocks * 34);

    // Create Q8_0Tensors with 2D shape (required by to_fp32_via_blocks)
    Q8_0Tensor Q_q8({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)}, Q_raw);
    Q8_0Tensor K_q8({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, K_raw);
    Q8_0Tensor V_q8({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, V_raw);

    // Allocate output (Q8_0 attention should write FP32 output, but we'll test with raw data pointer)
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    // NOTE: CPUAttentionT::compute expects float* interface (reinterprets to ElementType* internally)
    // For Q8_0: ElementType=int8_t, but we must pass float* pointers
    // Cast the raw Q8_0 block data to float* for the interface
    const float *Q_ptr = reinterpret_cast<const float *>(Q_raw.data());
    const float *K_ptr = reinterpret_cast<const float *>(K_raw.data());
    const float *V_ptr = reinterpret_cast<const float *>(V_raw.data());

    // Create attention kernel
    CPUAttentionT<Q8_0Tensor> attention;

    // Compute attention
    bool success = attention.compute(
        Q_ptr, K_ptr, V_ptr, output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false,                              // causal
        -1,                                 // window_size
        nullptr, nullptr, nullptr, nullptr, // workspaces (auto-allocate)
        false,                              // use_bf16
        nullptr, -1);

    EXPECT_TRUE(success) << "Q8_0 attention computation should succeed";
    EXPECT_TRUE(has_nonzero(output.data(), output.size())) << "Output should be non-zero";

    // NOTE: We don't validate exact values due to quantization error, just that it runs
}
