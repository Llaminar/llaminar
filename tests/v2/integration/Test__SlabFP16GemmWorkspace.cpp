/**
 * @file Test__SlabFP16GemmWorkspace.cpp
 * @brief Integration tests for slab-based FP16 GEMM with workspace buffer management
 *
 * Tests the Phase 3 integration of SlabGemmConfig with DeviceWorkspaceManager,
 * verifying that slab buffers can be pre-allocated and reused without
 * hot-path allocations.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "execution/DeviceWorkspaceManager.h"
#include "execution/WorkspaceDescriptor.h"
#include "kernels/SlabGemmConfig.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "backends/DeviceId.h"
#include "utils/Logger.h"

#include <cstring>
#include <memory>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__SlabFP16GemmWorkspace : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use CPU device for workspace allocation tests (doesn't require ROCm)
        device_ = DeviceId::cpu();
        budget_ = 10 * 1024 * 1024; // 10MB test budget
    }

    DeviceId device_;
    size_t budget_;
};

// =============================================================================
// SlabGemmConfig Tests
// =============================================================================

TEST_F(Test__SlabFP16GemmWorkspace, SlabConfigFitsInBudget)
{
    // 7B model FFN dimensions (typical stress test)
    int m = 512, n = 18944, k = 3584;

    SlabGemmConfig config = SlabGemmConfig::fromBudget(budget_, m, n, k, SlabDataType::FP16);

    // Verify slab dimensions are positive
    EXPECT_GT(config.slab_m, 0) << "slab_m should be positive";
    EXPECT_GT(config.slab_n, 0) << "slab_n should be positive";
    EXPECT_GT(config.slab_k, 0) << "slab_k should be positive";

    // Verify total workspace fits in budget
    size_t workspace_bytes = config.totalWorkspaceBytes(SlabDataType::FP16);
    EXPECT_LE(workspace_bytes, budget_)
        << "Total workspace " << (workspace_bytes / 1024 / 1024) << "MB exceeds budget "
        << (budget_ / 1024 / 1024) << "MB";

    LOG_INFO("[Test] SlabConfig: " << config.describe());
}

TEST_F(Test__SlabFP16GemmWorkspace, SlabConfigReturnsValidRequirements)
{
    int m = 512, n = 18944, k = 3584;

    SlabGemmConfig config = SlabGemmConfig::fromBudget(budget_, m, n, k, SlabDataType::FP16);
    auto reqs = config.workspaceRequirements(SlabDataType::FP16);

    // Should have 3 buffers: slab_a, slab_b, slab_c
    EXPECT_EQ(reqs.buffers.size(), 3u);

    // Check that all buffers have valid sizes
    for (const auto &buf : reqs.buffers)
    {
        EXPECT_GT(buf.size_bytes, 0u) << "Buffer " << buf.name << " has zero size";
        EXPECT_GE(buf.alignment, 1u) << "Buffer " << buf.name << " has invalid alignment";
        LOG_INFO("[Test] Buffer: " << buf.name << " = " << (buf.size_bytes / 1024) << "KB");
    }
}

TEST_F(Test__SlabFP16GemmWorkspace, WorkspaceCanAllocateSlabBuffers)
{
    int m = 512, n = 18944, k = 3584;

    SlabGemmConfig config = SlabGemmConfig::fromBudget(budget_, m, n, k, SlabDataType::FP16);
    auto reqs = config.workspaceRequirements(SlabDataType::FP16);

    DeviceWorkspaceManager mgr(device_, budget_);
    ASSERT_TRUE(mgr.allocate(reqs)) << "Failed to allocate workspace buffers";

    // Verify buffers exist
    EXPECT_TRUE(mgr.hasBuffer("slab_a_fp16"));
    EXPECT_TRUE(mgr.hasBuffer("slab_b_fp16"));
    EXPECT_TRUE(mgr.hasBuffer("slab_c_fp16"));

    // Verify buffer count
    EXPECT_EQ(mgr.bufferCount(), 3u);

    LOG_INFO("[Test] Allocated " << mgr.bufferCount() << " buffers, "
                                 << "used " << (mgr.used() / 1024) << "KB of "
                                 << (mgr.budget() / 1024) << "KB");
}

TEST_F(Test__SlabFP16GemmWorkspace, SlabIterationsEstimate)
{
    int m = 512, n = 18944, k = 3584;

    SlabGemmConfig config = SlabGemmConfig::fromBudget(budget_, m, n, k, SlabDataType::FP16);
    int iters = config.estimateIterations(m, n, k);

    // With 10MB budget and large dimensions, we expect multiple iterations
    EXPECT_GT(iters, 1) << "Expected multiple slab iterations for large GEMM";

    LOG_INFO("[Test] " << config.describe() << ", iterations: " << iters);
}

TEST_F(Test__SlabFP16GemmWorkspace, SlabBuffersAreWritable)
{
    SlabGemmConfig config;
    config.slab_m = 256;
    config.slab_n = 1024;
    config.slab_k = 512;

    auto reqs = config.workspaceRequirements(SlabDataType::FP16);

    DeviceWorkspaceManager mgr(device_, budget_);
    ASSERT_TRUE(mgr.allocate(reqs));

    void *slab_a = mgr.getBuffer("slab_a_fp16");
    void *slab_b = mgr.getBuffer("slab_b_fp16");
    void *slab_c = mgr.getBuffer("slab_c_fp16");

    ASSERT_NE(slab_a, nullptr);
    ASSERT_NE(slab_b, nullptr);
    ASSERT_NE(slab_c, nullptr);

    // Verify buffers are writable (memset should not crash)
    memset(slab_a, 0, config.slabABytes(SlabDataType::FP16));
    memset(slab_b, 0, config.slabBBytes(SlabDataType::FP16));
    memset(slab_c, 0, config.slabCBytes(SlabDataType::FP16));

    LOG_INFO("[Test] Slab buffers are writable");
}

TEST_F(Test__SlabFP16GemmWorkspace, DecodeUsesSmallSlabs)
{
    // Decode has M=1
    int m = 1, n = 18944, k = 3584;

    SlabGemmConfig config = SlabGemmConfig::forDecode(n, k, budget_);

    // For decode, slab_m should be small
    EXPECT_LE(config.slab_m, 32) << "Decode should use small slab_m";
    EXPECT_LE(config.totalWorkspaceBytes(SlabDataType::FP16), budget_);

    LOG_INFO("[Test] Decode config: " << config.describe());
}

TEST_F(Test__SlabFP16GemmWorkspace, PrefillUsesLargerSlabs)
{
    // Prefill has larger M
    int m = 512, n = 18944, k = 3584;

    SlabGemmConfig config = SlabGemmConfig::forPrefill(m, n, k, budget_);

    // For prefill, slab_m should be larger than decode's minimum
    EXPECT_GE(config.slab_m, 32) << "Prefill should use larger slab_m";
    EXPECT_LE(config.totalWorkspaceBytes(SlabDataType::FP16), budget_);

    LOG_INFO("[Test] Prefill config: " << config.describe());
}

TEST_F(Test__SlabFP16GemmWorkspace, SmallGemmDoesNotNeedSlabs)
{
    // Small GEMM that fits entirely in one slab
    int m = 32, n = 256, k = 128;

    SlabGemmConfig config = SlabGemmConfig::fromBudget(budget_, m, n, k, SlabDataType::FP16);

    // Should cover entire GEMM in one slab
    bool covers_all = config.coversEntireGemm(m, n, k);
    int iters = config.estimateIterations(m, n, k);

    EXPECT_TRUE(covers_all) << "Small GEMM should fit in single slab";
    EXPECT_EQ(iters, 1) << "Should need only 1 iteration";

    LOG_INFO("[Test] Small GEMM: " << config.describe() << ", single slab=" << covers_all);
}

TEST_F(Test__SlabFP16GemmWorkspace, ActualSlabDimsAtEdges)
{
    SlabGemmConfig config;
    config.slab_m = 256;
    config.slab_n = 1024;
    config.slab_k = 512;

    int full_m = 300, full_n = 1500, full_k = 700;

    // At the end of M dimension
    int actual_m, actual_n, actual_k;
    config.actualSlabDims(256, 0, 0, full_m, full_n, full_k, actual_m, actual_n, actual_k);

    EXPECT_EQ(actual_m, 44);   // 300 - 256 = 44
    EXPECT_EQ(actual_n, 1024); // Full slab_n
    EXPECT_EQ(actual_k, 512);  // Full slab_k

    // At the end of N dimension
    config.actualSlabDims(0, 1024, 0, full_m, full_n, full_k, actual_m, actual_n, actual_k);
    EXPECT_EQ(actual_n, 476); // 1500 - 1024 = 476

    LOG_INFO("[Test] Edge slab dims calculated correctly");
}

TEST_F(Test__SlabFP16GemmWorkspace, WorkspaceReleaseFreesBuffers)
{
    SlabGemmConfig config;
    config.slab_m = 256;
    config.slab_n = 1024;
    config.slab_k = 512;

    auto reqs = config.workspaceRequirements(SlabDataType::FP16);

    DeviceWorkspaceManager mgr(device_, budget_);
    ASSERT_TRUE(mgr.allocate(reqs));
    EXPECT_TRUE(mgr.isAllocated());
    EXPECT_EQ(mgr.bufferCount(), 3u);

    mgr.release();

    EXPECT_FALSE(mgr.isAllocated());
    EXPECT_EQ(mgr.bufferCount(), 0u);
    EXPECT_EQ(mgr.used(), 0u);
    EXPECT_FALSE(mgr.hasBuffer("slab_a_fp16"));

    LOG_INFO("[Test] Workspace release freed all buffers");
}

TEST_F(Test__SlabFP16GemmWorkspace, StandardBufferNamesUsed)
{
    // Verify that SlabGemmConfig uses standard GEMM buffer names
    SlabGemmConfig config;
    config.slab_m = 128;
    config.slab_n = 256;
    config.slab_k = 64;

    auto reqs = config.workspaceRequirements(SlabDataType::FP16);

    // Check for standard buffer names (fp16 suffix)
    bool has_slab_a = false, has_slab_b = false, has_slab_c = false;
    for (const auto &buf : reqs.buffers)
    {
        if (buf.name.find("slab_a") != std::string::npos)
            has_slab_a = true;
        if (buf.name.find("slab_b") != std::string::npos)
            has_slab_b = true;
        if (buf.name.find("slab_c") != std::string::npos)
            has_slab_c = true;
    }

    EXPECT_TRUE(has_slab_a) << "Missing slab_a buffer in requirements";
    EXPECT_TRUE(has_slab_b) << "Missing slab_b buffer in requirements";
    EXPECT_TRUE(has_slab_c) << "Missing slab_c buffer in requirements";
}

TEST_F(Test__SlabFP16GemmWorkspace, MultipleAllocationsWithSameManager)
{
    // Test that a workspace manager can be reused after release
    SlabGemmConfig config1;
    config1.slab_m = 128;
    config1.slab_n = 256;
    config1.slab_k = 64;

    SlabGemmConfig config2;
    config2.slab_m = 256;
    config2.slab_n = 512;
    config2.slab_k = 128;

    DeviceWorkspaceManager mgr(device_, budget_);

    // First allocation
    auto reqs1 = config1.workspaceRequirements(SlabDataType::FP16);
    ASSERT_TRUE(mgr.allocate(reqs1));
    size_t used1 = mgr.used();

    // Release and reallocate with different config
    mgr.release();
    auto reqs2 = config2.workspaceRequirements(SlabDataType::FP16);
    ASSERT_TRUE(mgr.allocate(reqs2));
    size_t used2 = mgr.used();

    // Second config should use more memory
    EXPECT_GT(used2, used1) << "Larger slab config should use more memory";

    LOG_INFO("[Test] First allocation: " << (used1 / 1024) << "KB, "
                                         << "Second allocation: " << (used2 / 1024) << "KB");
}

TEST_F(Test__SlabFP16GemmWorkspace, BudgetExceededGracefully)
{
    // Try to allocate more than budget allows
    size_t tiny_budget = 1024; // 1KB - way too small
    DeviceWorkspaceManager mgr(device_, tiny_budget);

    SlabGemmConfig config;
    config.slab_m = 1024;
    config.slab_n = 1024;
    config.slab_k = 1024;

    auto reqs = config.workspaceRequirements(SlabDataType::FP16);

    // Should fail gracefully (not crash)
    bool allocated = mgr.allocate(reqs);
    EXPECT_FALSE(allocated) << "Should fail when requirements exceed budget";

    LOG_INFO("[Test] Allocation correctly failed when budget exceeded");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
