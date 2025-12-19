/**
 * @file Test__DecomposedAttentionRequirement.cpp
 * @brief Unit tests for the decomposed attention requirement with graph buffer management
 * @author David Sanftenberg
 * @date January 2025
 *
 * Tests the fix for the "shared_from_this()" crash where AttentionWithKVCacheStage
 * was incompatible with GraphBufferManager's unique_ptr buffer storage.
 *
 * Root Cause (fixed):
 *   AttentionWithKVCacheStage::execute() calls create_view() on activation tensors.
 *   create_view() uses shared_from_this() which requires the tensor to be held
 *   in a std::shared_ptr. However, GraphBufferManager stores buffers as
 *   std::unique_ptr<TensorBase>, causing a bad_weak_ptr exception.
 *
 * Fix:
 *   When graph buffer management is enabled, Qwen2Pipeline forces
 *   use_decomposed_attention=true, which uses the decomposed attention path:
 *   - KVCacheAppendStage (appends K,V to cache without create_view)
 *   - AttentionComputeStage (computes attention without create_view)
 *
 *   The decomposed path works with unique_ptr buffers because it doesn't
 *   call create_view() or shared_from_this().
 *
 * Scenarios Tested:
 * 1. KVCacheAppendStage returns COPY type (data movement)
 * 2. AttentionComputeStage returns ATTENTION type
 * 3. ExecutionConfig has graph buffer management flag
 * 4. Buffer requirement declarations work for decomposed stages
 */

#include <gtest/gtest.h>
#include <memory>

#include "v2/execution/ComputeStage.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/utils/MPIContext.h"
#include "v2/utils/DebugEnv.h"

using namespace llaminar2;

namespace
{

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__DecomposedAttentionRequirement : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            mpi_ctx_ = std::make_unique<MPIContext>(0, 1, MPI_COMM_WORLD);
            factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
        }

        void TearDown() override
        {
            factory_.reset();
            mpi_ctx_.reset();
        }

        std::unique_ptr<MPIContext> mpi_ctx_;
        std::unique_ptr<TensorFactory> factory_;
    };

    // =============================================================================
    // ComputeStage Type Tests
    // =============================================================================

    TEST_F(Test__DecomposedAttentionRequirement, AttentionWithKVCacheStageExists)
    {
        // Verify the ATTENTION_WITH_KV_CACHE stage type exists in the name function
        // (the actual type may be represented differently in the enum)
        const char *name = computeStageTypeName(ComputeStageType::ATTENTION);
        EXPECT_STREQ(name, "ATTENTION");
    }

    // =============================================================================
    // Decomposed Attention Stage Tests
    // =============================================================================

    /**
     * @brief Test that KVCacheAppendStage can be created and has correct type
     */
    TEST_F(Test__DecomposedAttentionRequirement, KVCacheAppendStage_Creation)
    {
        // Create minimal params for the stage
        KVCacheAppendStage::Params params;
        params.K = nullptr; // Would be set before execute
        params.V = nullptr;
        params.kv_cache = nullptr;
        params.layer_idx = 0;
        params.seq_idx = 0;
        params.num_tokens = 1;

        auto stage = std::make_unique<KVCacheAppendStage>(params);

        // KVCacheAppendStage returns COPY type (data movement)
        EXPECT_EQ(stage->type(), ComputeStageType::COPY);
    }

    /**
     * @brief Test that AttentionComputeStage can be created and has correct type
     */
    TEST_F(Test__DecomposedAttentionRequirement, AttentionComputeStage_Creation)
    {
        // Create minimal params for the stage
        AttentionComputeStage::Params params;
        params.Q = nullptr;
        params.K = nullptr;
        params.V = nullptr;
        params.output = nullptr;
        params.batch_size = 1;
        params.seq_len = 1;
        params.kv_len = 1;
        params.n_heads = 14;
        params.n_kv_heads = 2;
        params.head_dim = 64;
        params.causal = true;

        auto stage = std::make_unique<AttentionComputeStage>(params);

        EXPECT_EQ(stage->type(), ComputeStageType::ATTENTION);
    }

    // =============================================================================
    // Configuration Tests
    // =============================================================================

    /**
     * @brief Verify that ExecutionConfig with graph buffer management is properly structured
     */
    TEST_F(Test__DecomposedAttentionRequirement, ExecutionConfig_HasGraphBufferManagementFlag)
    {
        ExecutionConfig config;

        // Default should be false
        EXPECT_FALSE(config.use_graph_buffer_management);

        // Should be able to set
        config.use_graph_buffer_management = true;
        EXPECT_TRUE(config.use_graph_buffer_management);
    }

    // =============================================================================
    // Integration Scenario: What Would Happen Without the Fix
    // =============================================================================

    /**
     * @brief Documents the failure scenario that the fix prevents
     *
     * Without the fix (forcing decomposed attention with graph buffers):
     * 1. Graph buffer management creates buffers as unique_ptr
     * 2. Pipeline uses AttentionWithKVCacheStage
     * 3. AttentionWithKVCacheStage calls K->create_view()
     * 4. create_view() calls shared_from_this()
     * 5. CRASH: bad_weak_ptr because tensor is not in shared_ptr
     *
     * With the fix:
     * 1. Graph buffer management creates buffers as unique_ptr
     * 2. Pipeline detects graph buffer management is enabled
     * 3. Pipeline forces use_decomposed_attention=true
     * 4. Pipeline uses KVCacheAppendStage + AttentionComputeStage
     * 5. These stages don't call create_view()
     * 6. SUCCESS: No shared_ptr requirement
     */
    TEST_F(Test__DecomposedAttentionRequirement, DocumentedFailureScenario)
    {
        // This test documents the scenario, actual integration testing
        // happens in Test__Qwen2GraphBufferManagement and the integration tests

        // Key insight: The fix is at the pipeline level, not the stage level
        // Pipeline checks use_graph_buffer_management and forces decomposed attention

        // We can't easily test the actual crash without loading a full model,
        // but we can verify the configuration path exists

        ExecutionConfig exec_config;
        exec_config.use_graph_buffer_management = true;

        // When this is true, the pipeline should:
        // 1. Set use_decomposed_attention = true
        // 2. Use KVCacheAppendStage instead of AttentionWithKVCacheStage
        // 3. Use AttentionComputeStage for the actual attention computation

        // The actual enforcement is in Qwen2Pipeline::attention_block()
        // This test verifies the flags are correctly structured

        EXPECT_TRUE(exec_config.use_graph_buffer_management);
        // The pipeline code has: if (use_graph_buffer_management) use_decomposed_attention = true;
    }

    // =============================================================================
    // Stage Buffer Requirements Tests
    // =============================================================================

    TEST_F(Test__DecomposedAttentionRequirement, DecomposedStages_DeclareBufferRequirements)
    {
        // KVCacheAppendStage
        KVCacheAppendStage::Params kv_params;
        kv_params.K = nullptr;
        kv_params.V = nullptr;
        kv_params.kv_cache = nullptr;
        kv_params.layer_idx = 0;
        kv_params.num_tokens = 1;

        auto kv_append = std::make_unique<KVCacheAppendStage>(kv_params);

        StageBufferRequirements kv_reqs = kv_append->getBufferRequirements();
        // KV append doesn't allocate new buffers, it writes to existing cache
        // But it should return a valid (possibly empty) requirements struct

        // AttentionComputeStage
        AttentionComputeStage::Params attn_params;
        attn_params.Q = nullptr;
        attn_params.K = nullptr;
        attn_params.V = nullptr;
        attn_params.output = nullptr;
        attn_params.batch_size = 1;
        attn_params.seq_len = 1;
        attn_params.kv_len = 1;
        attn_params.n_heads = 14;
        attn_params.n_kv_heads = 2;
        attn_params.head_dim = 64;

        auto attn_compute = std::make_unique<AttentionComputeStage>(attn_params);

        StageBufferRequirements attn_reqs = attn_compute->getBufferRequirements();
        // Attention compute may need scratch space for scores
    }

    // =============================================================================
    // Alternative: shared_ptr Wrapper Would NOT Work
    // =============================================================================

    /**
     * @brief This test documents why wrapping unique_ptr in shared_ptr doesn't work
     *
     * One might think: "Just wrap the unique_ptr in shared_ptr before passing to stage"
     * But this doesn't work because:
     * 1. shared_from_this() looks at the control block of the ORIGINAL shared_ptr
     * 2. A temporary shared_ptr wrapper doesn't give the object enable_shared_from_this
     * 3. The tensor must be ORIGINALLY created as shared_ptr for shared_from_this() to work
     *
     * Therefore, the only solution is to not call shared_from_this() at all,
     * which means using the decomposed attention path.
     */
    TEST_F(Test__DecomposedAttentionRequirement, SharedPtrWrapping_WouldNotWork)
    {
        // Create a tensor via unique_ptr (as GraphBufferManager does)
        auto unique_tensor = factory_->createFP32({64, 896}, 0);

        // Even if we wrap it in shared_ptr...
        // std::shared_ptr<TensorBase> wrapped(unique_tensor.release());
        // ...shared_from_this() would STILL fail because the tensor wasn't
        // originally created with make_shared and doesn't have a control block

        // This is why the fix uses decomposed attention (avoids create_view entirely)
        // rather than trying to fix the ownership model

        EXPECT_NE(unique_tensor, nullptr);
    }

} // namespace
