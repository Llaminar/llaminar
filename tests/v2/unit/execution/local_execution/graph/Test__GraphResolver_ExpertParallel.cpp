/**
 * @file Test__GraphResolver_ExpertParallel.cpp
 * @brief Unit tests for GraphResolver ExpertParallel TP mode
 *
 * Verifies that when a StageSpec has tp_mode = TPMode::ExpertParallel,
 * the GraphResolver inserts an AllReduce collective stage (for world_size > 1).
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/GraphResolver.h"
#include "execution/local_execution/graph/GraphSchema.h"

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__GraphResolver_ExpertParallel : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Minimal schema with a single MoE FFN layer stage using ExpertParallel
        schema_.name = "test_moe";
        schema_.version = "1.0";

        schema_.layer_template.ffn_stages = {
            StageSpec{
                .name = "moe_ffn",
                .type = StageType::MoEFFN,
                .inputs = {{"normalized", BufferSemantic::Input}},
                .outputs = {{"moe_combined_output", BufferSemantic::Output}},
                .tp_mode = TPMode::ExpertParallel,
                .is_optional = true,
                .exec_policy_key = "exec_moe_ffn"}};

        config_.batch_size = 1;
        config_.seq_len = 10;
        config_.n_layers = 1;
        config_.d_model = 896;
        config_.n_heads = 14;
        config_.n_kv_heads = 2;
        config_.head_dim = 64;
        config_.d_ff = 4864;
        config_.vocab_size = 151936;
        config_.exec_policy = ExecutionPolicyFlags::fromDebugEnv();
    }

    GraphSchema schema_;
    GraphResolverConfig config_;
    TensorContext empty_tensors_;
    GraphResolver resolver_;
};

// ============================================================================
// ExpertParallel TP Tests
// ============================================================================

TEST_F(Test__GraphResolver_ExpertParallel, SingleRank_NoAllreduce)
{
    config_.world_size = 1;

    ResolvedGraphSpec resolved = resolver_.resolve(schema_, config_, empty_tensors_);

    // Single rank: no allreduce should be inserted
    for (const auto &stage : resolved.stages)
    {
        EXPECT_NE(stage.type, StageType::Allreduce)
            << "No allreduce expected for single rank, but found: " << stage.name;
    }
    EXPECT_EQ(resolved.stats.allreduce_inserted, 0);
}

TEST_F(Test__GraphResolver_ExpertParallel, MultiRank_InsertsAllreduce)
{
    config_.world_size = 2;
    config_.rank = 0;

    ResolvedGraphSpec resolved = resolver_.resolve(schema_, config_, empty_tensors_);

    // Multi-rank with ExpertParallel: should insert an allreduce
    bool found_allreduce = false;
    std::string allreduce_name;
    for (const auto &stage : resolved.stages)
    {
        if (stage.type == StageType::Allreduce)
        {
            found_allreduce = true;
            allreduce_name = stage.name;
            break;
        }
    }

    EXPECT_TRUE(found_allreduce)
        << "Expected allreduce stage for ExpertParallel moe_ffn with world_size=2";
}

TEST_F(Test__GraphResolver_ExpertParallel, AllreduceNameFollowsConvention)
{
    config_.world_size = 2;
    config_.rank = 0;

    ResolvedGraphSpec resolved = resolver_.resolve(schema_, config_, empty_tensors_);

    // The allreduce stage name should be "<stage_name>_allreduce"
    bool found_correct_name = false;
    for (const auto &stage : resolved.stages)
    {
        if (stage.type == StageType::Allreduce &&
            stage.name == "layer0_moe_ffn_allreduce")
        {
            found_correct_name = true;

            // Verify dependency chain
            ASSERT_EQ(stage.dependencies.size(), 1u);
            EXPECT_EQ(stage.dependencies[0], "layer0_moe_ffn");
            break;
        }
    }

    EXPECT_TRUE(found_correct_name)
        << "Expected allreduce named 'layer0_moe_ffn_allreduce'";
}

TEST_F(Test__GraphResolver_ExpertParallel, AllreduceHasCorrectCount)
{
    config_.world_size = 2;
    config_.rank = 0;
    config_.batch_size = 1;
    config_.seq_len = 10;
    config_.d_model = 896;

    ResolvedGraphSpec resolved = resolver_.resolve(schema_, config_, empty_tensors_);

    for (const auto &stage : resolved.stages)
    {
        if (stage.type == StageType::Allreduce)
        {
            auto it = stage.int_params.find("count");
            ASSERT_NE(it, stage.int_params.end()) << "Allreduce missing 'count' param";

            // count = batch_size * seq_len * d_model = 1 * 10 * 896 = 8960
            EXPECT_EQ(it->second, 8960);
            break;
        }
    }
}

TEST_F(Test__GraphResolver_ExpertParallel, TPModeNone_NoAllreduce)
{
    // A stage with TPMode::None should never get an allreduce
    schema_.layer_template.ffn_stages = {
        StageSpec{
            .name = "moe_router",
            .type = StageType::MoERouter,
            .inputs = {{"normalized", BufferSemantic::Input}},
            .outputs = {{"moe_expert_indices", BufferSemantic::Output}},
            .tp_mode = TPMode::None,
            .is_optional = true,
            .exec_policy_key = "exec_moe_router"}};

    config_.world_size = 2;

    ResolvedGraphSpec resolved = resolver_.resolve(schema_, config_, empty_tensors_);

    for (const auto &stage : resolved.stages)
    {
        EXPECT_NE(stage.type, StageType::Allreduce)
            << "No allreduce expected for TPMode::None, but found: " << stage.name;
    }
}

TEST_F(Test__GraphResolver_ExpertParallel, SharedExpert_RowParallel_InsertsAllreduce)
{
    // Shared expert with RowParallel should also get an allreduce
    schema_.layer_template.ffn_stages = {
        StageSpec{
            .name = "shared_expert",
            .type = StageType::MoESharedExpert,
            .inputs = {{"normalized", BufferSemantic::Input}},
            .outputs = {{"shared_expert_output", BufferSemantic::Output}},
            .tp_mode = TPMode::RowParallel,
            .is_optional = true,
            .exec_policy_key = "exec_moe_shared_expert"}};

    config_.world_size = 2;

    ResolvedGraphSpec resolved = resolver_.resolve(schema_, config_, empty_tensors_);

    bool found_allreduce = false;
    for (const auto &stage : resolved.stages)
    {
        if (stage.type == StageType::Allreduce &&
            stage.name == "layer0_shared_expert_allreduce")
        {
            found_allreduce = true;
            break;
        }
    }

    EXPECT_TRUE(found_allreduce)
        << "Expected allreduce for RowParallel shared_expert with world_size=2";
}
