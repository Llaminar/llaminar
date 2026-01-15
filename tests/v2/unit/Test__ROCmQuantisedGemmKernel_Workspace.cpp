/**
 * @file Test__ROCmQuantisedGemmKernel_Workspace.cpp
 * @brief Unit tests for ROCmQuantisedGemmKernel IWorkspaceConsumer implementation
 *
 * These tests verify the workspace integration (Phase 2) for ROCm GEMM kernels.
 * Tests run on CPU backend without requiring actual ROCm hardware.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <cstring>

#include "execution/DeviceWorkspaceManager.h"
#include "execution/WorkspaceDescriptor.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "backends/BackendManager.h"

using namespace llaminar2;

/**
 * @brief Test fixture for ROCmQuantisedGemmKernel workspace tests
 *
 * Uses CPU device so tests run without ROCm hardware.
 */
class Test__ROCmQuantisedGemmKernel_Workspace : public ::testing::Test
{
protected:
    DeviceId device = DeviceId::cpu();
    size_t budget = 256 * 1024 * 1024; // 256MB

    void SetUp() override
    {
        if (!hasCPUBackend())
        {
            initCPUBackend(-1);
        }
    }
};

// ============================================================================
// GemmWorkspaceBuffers Namespace Tests
// ============================================================================

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, StandardBufferNamesAreDefined)
{
    // Verify all standard buffer names are non-null
    EXPECT_NE(GemmWorkspaceBuffers::QUANT_A, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::SCALES_A, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::ACC_INT32, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::TEMP_A_FP32, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::TEMP_C_FP32, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::SLAB_A_FP16, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::SLAB_B_FP16, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::SLAB_C_FP16, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::FULL_A_FP16, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::FULL_B_FP16, nullptr);
    EXPECT_NE(GemmWorkspaceBuffers::FULL_C_FP16, nullptr);
}

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, StandardBufferNamesAreUnique)
{
    // Verify all buffer names are unique strings
    std::vector<const char *> names = {
        GemmWorkspaceBuffers::QUANT_A,
        GemmWorkspaceBuffers::SCALES_A,
        GemmWorkspaceBuffers::ACC_INT32,
        GemmWorkspaceBuffers::TEMP_A_FP32,
        GemmWorkspaceBuffers::TEMP_C_FP32,
        GemmWorkspaceBuffers::SLAB_A_FP16,
        GemmWorkspaceBuffers::SLAB_B_FP16,
        GemmWorkspaceBuffers::SLAB_C_FP16,
        GemmWorkspaceBuffers::FULL_A_FP16,
        GemmWorkspaceBuffers::FULL_B_FP16,
        GemmWorkspaceBuffers::FULL_C_FP16,
    };

    // Check all pairs are different
    for (size_t i = 0; i < names.size(); ++i)
    {
        for (size_t j = i + 1; j < names.size(); ++j)
        {
            EXPECT_STRNE(names[i], names[j])
                << "Buffer names at indices " << i << " and " << j << " are not unique";
        }
    }
}

// ============================================================================
// Workspace Requirements Tests (without actual kernel)
// ============================================================================

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, INT8PathRequirementsSizesCorrect)
{
    // Manually compute INT8 path requirements (as kernel would)
    int m = 128, n = 4096, k = 896;

    size_t quant_a_bytes = static_cast<size_t>(m) * k * sizeof(int8_t);
    size_t scales_a_bytes = static_cast<size_t>(m) * sizeof(float);
    size_t acc_int32_bytes = static_cast<size_t>(m) * n * sizeof(int32_t);
    size_t temp_a_fp32_bytes = static_cast<size_t>(m) * k * sizeof(float);
    size_t temp_c_fp32_bytes = static_cast<size_t>(m) * n * sizeof(float);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({GemmWorkspaceBuffers::QUANT_A, quant_a_bytes, 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A, scales_a_bytes, 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::ACC_INT32, acc_int32_bytes, 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::TEMP_A_FP32, temp_a_fp32_bytes, 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::TEMP_C_FP32, temp_c_fp32_bytes, 256, true});

    // Verify sizes
    EXPECT_EQ(quant_a_bytes, 128 * 896);        // 114,688 bytes
    EXPECT_EQ(scales_a_bytes, 128 * 4);         // 512 bytes
    EXPECT_EQ(acc_int32_bytes, 128 * 4096 * 4); // 2,097,152 bytes

    // Verify requirements structure
    EXPECT_EQ(reqs.buffers.size(), 5);
    EXPECT_GT(reqs.total_bytes(), 0);

    // Can find each buffer
    EXPECT_NE(reqs.find(GemmWorkspaceBuffers::QUANT_A), nullptr);
    EXPECT_NE(reqs.find(GemmWorkspaceBuffers::SCALES_A), nullptr);
    EXPECT_NE(reqs.find(GemmWorkspaceBuffers::ACC_INT32), nullptr);
}

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, FP16PathRequirementsSizesCorrect)
{
    // Manually compute FP16 path requirements (as kernel would for M > 128)
    int m = 256, n = 4096, k = 896;

    size_t a_fp16_bytes = static_cast<size_t>(m) * k * sizeof(uint16_t);
    size_t b_fp16_bytes = static_cast<size_t>(k) * n * sizeof(uint16_t);
    size_t c_fp16_bytes = static_cast<size_t>(m) * n * sizeof(uint16_t);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({GemmWorkspaceBuffers::FULL_A_FP16, a_fp16_bytes, 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::FULL_B_FP16, b_fp16_bytes, 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::FULL_C_FP16, c_fp16_bytes, 256, true});

    // Verify sizes
    EXPECT_EQ(a_fp16_bytes, 256 * 896 * 2);  // 458,752 bytes
    EXPECT_EQ(b_fp16_bytes, 896 * 4096 * 2); // 7,340,032 bytes (~7MB)
    EXPECT_EQ(c_fp16_bytes, 256 * 4096 * 2); // 2,097,152 bytes

    // Verify structure
    EXPECT_EQ(reqs.buffers.size(), 3);
}

// ============================================================================
// DeviceWorkspaceManager Integration Tests
// ============================================================================

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, WorkspaceManagerCanAllocateINT8Buffers)
{
    DeviceWorkspaceManager mgr(device, budget);

    // Create INT8 GEMM workspace requirements
    int m = 128, n = 4096, k = 896;

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({GemmWorkspaceBuffers::QUANT_A,
                            static_cast<size_t>(m) * k * sizeof(int8_t), 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A,
                            static_cast<size_t>(m) * sizeof(float), 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::ACC_INT32,
                            static_cast<size_t>(m) * n * sizeof(int32_t), 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    // Verify buffers exist
    EXPECT_TRUE(mgr.hasBuffer(GemmWorkspaceBuffers::QUANT_A));
    EXPECT_TRUE(mgr.hasBuffer(GemmWorkspaceBuffers::SCALES_A));
    EXPECT_TRUE(mgr.hasBuffer(GemmWorkspaceBuffers::ACC_INT32));

    // Verify pointers are valid
    EXPECT_NE(mgr.getBuffer(GemmWorkspaceBuffers::QUANT_A), nullptr);
    EXPECT_NE(mgr.getBuffer(GemmWorkspaceBuffers::SCALES_A), nullptr);
    EXPECT_NE(mgr.getBuffer(GemmWorkspaceBuffers::ACC_INT32), nullptr);

    // Verify sizes match
    EXPECT_EQ(mgr.getBufferSize(GemmWorkspaceBuffers::QUANT_A),
              static_cast<size_t>(m) * k * sizeof(int8_t));
    EXPECT_EQ(mgr.getBufferSize(GemmWorkspaceBuffers::SCALES_A),
              static_cast<size_t>(m) * sizeof(float));
    EXPECT_EQ(mgr.getBufferSize(GemmWorkspaceBuffers::ACC_INT32),
              static_cast<size_t>(m) * n * sizeof(int32_t));
}

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, WorkspaceManagerCanAllocateFP16Buffers)
{
    DeviceWorkspaceManager mgr(device, budget);

    // Create FP16 GEMM workspace requirements
    int m = 256, n = 4096, k = 896;

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({GemmWorkspaceBuffers::FULL_A_FP16,
                            static_cast<size_t>(m) * k * sizeof(uint16_t), 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::FULL_B_FP16,
                            static_cast<size_t>(k) * n * sizeof(uint16_t), 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::FULL_C_FP16,
                            static_cast<size_t>(m) * n * sizeof(uint16_t), 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    // Verify buffers exist and are usable
    void *a_ptr = mgr.getBuffer(GemmWorkspaceBuffers::FULL_A_FP16);
    void *b_ptr = mgr.getBuffer(GemmWorkspaceBuffers::FULL_B_FP16);
    void *c_ptr = mgr.getBuffer(GemmWorkspaceBuffers::FULL_C_FP16);

    EXPECT_NE(a_ptr, nullptr);
    EXPECT_NE(b_ptr, nullptr);
    EXPECT_NE(c_ptr, nullptr);

    // Verify buffers don't overlap
    EXPECT_NE(a_ptr, b_ptr);
    EXPECT_NE(b_ptr, c_ptr);
    EXPECT_NE(a_ptr, c_ptr);
}

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, WorkspaceBuffersAreWritable)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({GemmWorkspaceBuffers::QUANT_A, 4096, 256, true});
    reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A, 1024, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    // Write pattern to buffer (verify it doesn't crash)
    void *quant_ptr = mgr.getBuffer(GemmWorkspaceBuffers::QUANT_A);
    void *scales_ptr = mgr.getBuffer(GemmWorkspaceBuffers::SCALES_A);

    std::memset(quant_ptr, 0x55, 4096);
    std::memset(scales_ptr, 0xAA, 1024);

    // Read back and verify
    auto *quant_bytes = static_cast<uint8_t *>(quant_ptr);
    auto *scales_bytes = static_cast<uint8_t *>(scales_ptr);

    EXPECT_EQ(quant_bytes[0], 0x55);
    EXPECT_EQ(quant_bytes[4095], 0x55);
    EXPECT_EQ(scales_bytes[0], 0xAA);
    EXPECT_EQ(scales_bytes[1023], 0xAA);
}

// ============================================================================
// WorkspaceRequirements Factory Method Tests
// ============================================================================

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, ForQuantizedGemmCreateCorrectBuffers)
{
    int m = 128, n = 4096, k = 896;
    auto reqs = WorkspaceRequirements::forQuantizedGemm(m, n, k);

    // Should have 3 buffers
    EXPECT_EQ(reqs.buffers.size(), 3);

    // Verify buffer names match constants
    EXPECT_NE(reqs.find("gemm_quant_a"), nullptr);
    EXPECT_NE(reqs.find("gemm_scales_a"), nullptr);
    EXPECT_NE(reqs.find("gemm_acc_int32"), nullptr);

    // Verify all are required
    for (const auto &buf : reqs.buffers)
    {
        EXPECT_TRUE(buf.required);
        EXPECT_EQ(buf.alignment, 256);
    }
}

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, ForFP16GemmCreateCorrectBuffers)
{
    int m = 256, n = 4096, k = 896;
    auto reqs = WorkspaceRequirements::forFP16Gemm(m, n, k);

    // Should have 3 buffers
    EXPECT_EQ(reqs.buffers.size(), 3);

    // Verify buffer names match constants
    EXPECT_NE(reqs.find("gemm_full_a_fp16"), nullptr);
    EXPECT_NE(reqs.find("gemm_full_b_fp16"), nullptr);
    EXPECT_NE(reqs.find("gemm_full_c_fp16"), nullptr);
}

// ============================================================================
// Workspace Requirements Merge Tests
// ============================================================================

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, MergeRequirementsKeepsLargerSize)
{
    WorkspaceRequirements reqs1;
    reqs1.buffers.push_back({"shared_buffer", 1024, 256, true});

    WorkspaceRequirements reqs2;
    reqs2.buffers.push_back({"shared_buffer", 4096, 256, true});

    reqs1.merge(reqs2);

    // Should keep the larger size
    EXPECT_EQ(reqs1.buffers.size(), 1);
    EXPECT_EQ(reqs1.buffers[0].size_bytes, 4096);
}

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, MergeRequirementsAddsNewBuffers)
{
    WorkspaceRequirements reqs1;
    reqs1.buffers.push_back({GemmWorkspaceBuffers::QUANT_A, 1024, 256, true});

    WorkspaceRequirements reqs2;
    reqs2.buffers.push_back({GemmWorkspaceBuffers::SCALES_A, 512, 256, true});

    reqs1.merge(reqs2);

    // Should have both buffers
    EXPECT_EQ(reqs1.buffers.size(), 2);
    EXPECT_NE(reqs1.find(GemmWorkspaceBuffers::QUANT_A), nullptr);
    EXPECT_NE(reqs1.find(GemmWorkspaceBuffers::SCALES_A), nullptr);
}

// ============================================================================
// Realistic Model Dimension Tests
// ============================================================================

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, Qwen2_0_5BFFNDimensionsFitInBudget)
{
    // Qwen2.5-0.5B: d_model=896, intermediate=4864
    // Test FFN projection only (not LM head which requires vocab-sized buffers)
    int d_model = 896;
    int intermediate = 4864;
    int max_seq_len = 2048;

    // FFN down projection: [seq_len × intermediate] × [intermediate × d_model]
    auto ffn_reqs = WorkspaceRequirements::forQuantizedGemm(max_seq_len, d_model, intermediate);

    // FFN should fit in 256MB budget
    size_t total = ffn_reqs.total_bytes_with_alignment();
    EXPECT_LT(total, budget);
}

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, Qwen2_7BDimensionsFitInBudget)
{
    // Qwen2.5-7B: d_model=3584, intermediate=18944, vocab=152064
    int d_model = 3584;
    int intermediate = 18944;
    int max_seq_len = 512; // Reduced for budget fit

    // FFN projection (INT8 path for small seq_len)
    auto ffn_reqs = WorkspaceRequirements::forQuantizedGemm(max_seq_len, d_model, intermediate);

    // Should fit in 256MB budget for INT8 path
    size_t total = ffn_reqs.total_bytes_with_alignment();
    EXPECT_LT(total, budget);
}

TEST_F(Test__ROCmQuantisedGemmKernel_Workspace, LMHeadRequiresLargeBudget)
{
    // LM head requires very large buffers due to vocab size
    // This test documents the memory requirement, not a pass/fail expectation
    int vocab = 151936; // Qwen2.5-0.5B vocab size
    int d_model = 896;
    int seq_len = 1; // Decode mode (single token)

    auto lm_head_reqs = WorkspaceRequirements::forQuantizedGemm(seq_len, vocab, d_model);

    // For decode (seq_len=1), even LM head fits easily
    size_t total = lm_head_reqs.total_bytes_with_alignment();
    EXPECT_LT(total, 256 * 1024 * 1024); // Should fit in 256MB for decode
}
