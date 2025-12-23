/**
 * @file Test__GraphResolver.cpp
 * @brief Unit tests for GraphResolver schema resolution
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "../../../src/v2/execution/GraphResolver.h"
#include "../../../src/v2/execution/GraphSchema.h"
#include "../../../src/v2/execution/GraphExecutor.h"
#include "../../../src/v2/models/qwen/Qwen2Schema.h"
#include "../../../src/v2/tensors/Tensors.h"

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__GraphResolver : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a minimal schema for testing
        test_schema_.name = "test_model";
        test_schema_.version = "1.0";

        // Simple embedding stage
        test_schema_.embedding = StageSpec{
            .name = "embedding",
            .type = StageType::Embedding,
            .inputs = {{"token_ids"}, {"weights.embedding_table"}},
            .outputs = {{"hidden"}},
            .is_optional = true,
            .exec_policy_key = "exec_embedding"};

        // Simple layer template
        test_schema_.layer_template.attention_stages = {
            StageSpec{
                .name = "attn_norm",
                .type = StageType::RMSNorm,
                .inputs = {{"hidden"}, {"weights.attn_norm"}},
                .outputs = {{"normalized"}},
                .is_optional = true,
                .exec_policy_key = "exec_rmsnorm"},
            StageSpec{
                .name = "wo_proj",
                .type = StageType::GEMM,
                .inputs = {{"attn_output"}, {"weights.wo"}},
                .outputs = {{"attn_proj"}},
                .dependencies = {"attn_norm"},
                .tp_mode = TPMode::RowParallel,
                .is_optional = true,
                .exec_policy_key = "exec_gemm"}};

        test_schema_.layer_template.ffn_stages = {
            StageSpec{
                .name = "ffn_norm",
                .type = StageType::RMSNorm,
                .inputs = {{"hidden"}, {"weights.ffn_norm"}},
                .outputs = {{"normalized"}},
                .is_optional = true,
                .exec_policy_key = "exec_rmsnorm"}};

        // LM head
        test_schema_.lm_head_stages = {
            StageSpec{
                .name = "final_norm",
                .type = StageType::RMSNorm,
                .inputs = {{"hidden"}, {"weights.final_norm"}},
                .outputs = {{"hidden"}},
                .is_optional = true,
                .exec_policy_key = "exec_rmsnorm"},
            StageSpec{
                .name = "lm_head",
                .type = StageType::LMHead,
                .inputs = {{"hidden"}, {"weights.lm_head"}},
                .outputs = {{"logits"}},
                .dependencies = {"final_norm"},
                .tp_mode = TPMode::ColumnParallel,
                .is_optional = true,
                .exec_policy_key = "exec_lm_head"}};

        // Default runtime config
        default_config_.world_size = 1;
        default_config_.rank = 0;
        default_config_.batch_size = 1;
        default_config_.seq_len = 10;
        default_config_.n_layers = 2;
        default_config_.d_model = 64;
        default_config_.n_heads = 4;
        default_config_.n_kv_heads = 4;
        default_config_.head_dim = 16;
        default_config_.d_ff = 256;
        default_config_.vocab_size = 1000;
        default_config_.exec_policy = ExecutionPolicyFlags::fromDebugEnv();
    }

    GraphSchema test_schema_;
    GraphResolverConfig default_config_;
    TensorContext empty_tensors_;
    GraphResolver resolver_;
};

// ============================================================================
// ExecutionPolicyFlags Tests
// ============================================================================

TEST_F(Test__GraphResolver, ExecutionPolicyFlags_DefaultsAllEnabled)
{
    ExecutionPolicyFlags flags;

    EXPECT_TRUE(flags.exec_embedding);
    EXPECT_TRUE(flags.exec_rmsnorm);
    EXPECT_TRUE(flags.exec_gemm);
    EXPECT_TRUE(flags.exec_rope);
    EXPECT_TRUE(flags.exec_attention);
    EXPECT_TRUE(flags.exec_swiglu);
    EXPECT_TRUE(flags.exec_residual);
    EXPECT_TRUE(flags.exec_lm_head);
}

TEST_F(Test__GraphResolver, ExecutionPolicyFlags_ShouldExecute)
{
    ExecutionPolicyFlags flags;
    flags.exec_gemm = false;

    EXPECT_TRUE(flags.shouldExecute("exec_rmsnorm"));
    EXPECT_FALSE(flags.shouldExecute("exec_gemm"));
    EXPECT_TRUE(flags.shouldExecute(""));            // Empty key = enabled
    EXPECT_TRUE(flags.shouldExecute("unknown_key")); // Unknown = enabled
}

TEST_F(Test__GraphResolver, ExecutionPolicyFlags_FromDebugEnv)
{
    // fromDebugEnv() reads the actual debugEnv() singleton
    // Just verify it doesn't crash and returns valid flags
    ExecutionPolicyFlags flags = ExecutionPolicyFlags::fromDebugEnv();

    // All should be true by default (unless env vars set)
    EXPECT_TRUE(flags.exec_embedding);
    EXPECT_TRUE(flags.exec_rmsnorm);
}

// ============================================================================
// Schema Resolution Tests
// ============================================================================

TEST_F(Test__GraphResolver, Resolve_EmptySchema)
{
    GraphSchema empty_schema;
    empty_schema.name = "empty";
    empty_schema.version = "1.0";

    GraphResolverConfig config = default_config_;
    config.n_layers = 0;

    ResolvedGraphSpec resolved = resolver_.resolve(empty_schema, config, empty_tensors_);

    EXPECT_EQ(resolved.name, "empty");
    // Empty schema: no embedding, no layers, no lm_head
    EXPECT_EQ(resolved.stages.size(), 0);
}

TEST_F(Test__GraphResolver, Resolve_SingleLayer)
{
    GraphResolverConfig config = default_config_;
    config.n_layers = 1;

    ResolvedGraphSpec resolved = resolver_.resolve(test_schema_, config, empty_tensors_);

    EXPECT_EQ(resolved.name, "test_model");
    // Should have: embedding + attn_norm + wo_proj + ffn_norm + final_norm + lm_head
    // = 6 stages minimum (depending on policy)
    EXPECT_GE(resolved.stages.size(), 5);
}

TEST_F(Test__GraphResolver, Resolve_StageNames_HaveLayerPrefix)
{
    GraphResolverConfig config = default_config_;
    config.n_layers = 2;

    ResolvedGraphSpec resolved = resolver_.resolve(test_schema_, config, empty_tensors_);

    // Find layer stages and verify naming
    bool found_layer0_attn_norm = false;
    bool found_layer1_attn_norm = false;

    for (const auto &stage : resolved.stages)
    {
        if (stage.name == "layer0_attn_norm")
            found_layer0_attn_norm = true;
        if (stage.name == "layer1_attn_norm")
            found_layer1_attn_norm = true;
    }

    EXPECT_TRUE(found_layer0_attn_norm) << "Layer 0 attn_norm not found";
    EXPECT_TRUE(found_layer1_attn_norm) << "Layer 1 attn_norm not found";
}

TEST_F(Test__GraphResolver, Resolve_DisabledStages_AreSkipped)
{
    GraphResolverConfig config = default_config_;
    config.n_layers = 1;
    config.exec_policy.exec_rmsnorm = false;

    ResolvedGraphSpec resolved = resolver_.resolve(test_schema_, config, empty_tensors_);

    // RMSNorm stages should be skipped
    for (const auto &stage : resolved.stages)
    {
        EXPECT_NE(stage.type, StageType::RMSNorm)
            << "RMSNorm stage " << stage.name << " should have been skipped";
    }

    EXPECT_GT(resolved.stats.stages_skipped, 0);
}

// ============================================================================
// Tensor Parallelism Tests
// ============================================================================

TEST_F(Test__GraphResolver, Resolve_SingleRank_NoCollectives)
{
    GraphResolverConfig config = default_config_;
    config.world_size = 1;
    config.n_layers = 1;

    ResolvedGraphSpec resolved = resolver_.resolve(test_schema_, config, empty_tensors_);

    // No allreduce/allgather for single rank
    EXPECT_EQ(resolved.stats.allreduce_inserted, 0);
    EXPECT_EQ(resolved.stats.allgather_inserted, 0);
}

TEST_F(Test__GraphResolver, Resolve_MultiRank_InsertsAllreduce)
{
    GraphResolverConfig config = default_config_;
    config.world_size = 2;
    config.rank = 0;
    config.n_layers = 1;

    ResolvedGraphSpec resolved = resolver_.resolve(test_schema_, config, empty_tensors_);

    // wo_proj has tp_mode = RowParallel, should insert allreduce
    // Search the stages for an allreduce stage
    bool found_allreduce = false;
    for (const auto &stage : resolved.stages)
    {
        if (stage.type == StageType::Allreduce)
        {
            found_allreduce = true;
            break;
        }
    }
    EXPECT_TRUE(found_allreduce)
        << "Expected allreduce stage for row-parallel wo_proj with world_size=2";
}

TEST_F(Test__GraphResolver, Resolve_MultiRank_LMHead_InsertsAllgather)
{
    GraphResolverConfig config = default_config_;
    config.world_size = 2;
    config.rank = 0;
    config.n_layers = 1;

    ResolvedGraphSpec resolved = resolver_.resolve(test_schema_, config, empty_tensors_);

    // lm_head has tp_mode = ColumnParallel, should insert allgather
    EXPECT_GT(resolved.stats.allgather_inserted, 0)
        << "Expected allgather for column-parallel lm_head with world_size=2";
}

// ============================================================================
// Attention Mode Detection Tests
// ============================================================================

TEST_F(Test__GraphResolver, DetectAttentionMode_Prefill)
{
    GraphResolverConfig config = default_config_;
    config.batch_size = 1;
    config.seq_len = 100;
    config.cached_tokens = 0; // No cache = prefill

    // Access private method via friend or just test indirectly
    // For now, test via resolve() and check params
    // (Full test would require exposing detectAttentionMode or making it protected)
}

// ============================================================================
// TensorContext Tests
// ============================================================================

TEST_F(Test__GraphResolver, TensorContext_ResolveBuffer)
{
    TensorContext ctx;

    // Create a test tensor
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{10, 64});
    TensorBase *raw_ptr = tensor.get();
    ctx.buffers["hidden"] = raw_ptr;

    TensorRef ref{"hidden", BufferSemantic::Input};
    TensorBase *resolved = ctx.resolve(ref, -1);

    EXPECT_EQ(resolved, raw_ptr);
}

TEST_F(Test__GraphResolver, TensorContext_ResolveModelWeight)
{
    TensorContext ctx;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1000, 64});
    TensorBase *raw_ptr = tensor.get();
    ctx.model_weights["embedding_table"] = raw_ptr;

    TensorRef ref{"weights.embedding_table", BufferSemantic::Input};
    TensorBase *resolved = ctx.resolve(ref, -1);

    EXPECT_EQ(resolved, raw_ptr);
}

TEST_F(Test__GraphResolver, TensorContext_ResolveLayerWeight)
{
    TensorContext ctx;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    TensorBase *raw_ptr = tensor.get();

    // Set up layer weight accessor
    ctx.get_layer_weight = [raw_ptr](int layer_idx, const std::string &name) -> TensorBase *
    {
        if (layer_idx == 0 && name == "attn_norm")
            return raw_ptr;
        return nullptr;
    };

    TensorRef ref{"weights.attn_norm", BufferSemantic::Input};
    TensorBase *resolved = ctx.resolve(ref, 0);

    EXPECT_EQ(resolved, raw_ptr);
}

TEST_F(Test__GraphResolver, TensorContext_ResolveUnknown_ReturnsNull)
{
    TensorContext ctx;

    TensorRef ref{"nonexistent", BufferSemantic::Input};
    TensorBase *resolved = ctx.resolve(ref, -1);

    EXPECT_EQ(resolved, nullptr);
}

// ============================================================================
// Qwen2Schema Factory Tests
// ============================================================================

TEST_F(Test__GraphResolver, Qwen2SchemaFactory_CreatesValidSchema)
{
    Qwen2SchemaFactory factory;

    EXPECT_EQ(factory.architectureName(), "qwen2");

    GraphSchema schema = factory.createSchema();

    EXPECT_EQ(schema.name, "qwen2");
    EXPECT_EQ(schema.version, "1.0");
    EXPECT_FALSE(schema.required_params.empty());

    // Check layer template has stages
    EXPECT_FALSE(schema.layer_template.attention_stages.empty());
    EXPECT_FALSE(schema.layer_template.ffn_stages.empty());
    EXPECT_FALSE(schema.lm_head_stages.empty());
}

TEST_F(Test__GraphResolver, Qwen2Schema_AttentionStages_HaveCorrectTPModes)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Find QKV proj and verify column parallel
    bool found_qkv = false;
    for (const auto &stage : schema.layer_template.attention_stages)
    {
        if (stage.name == "qkv_proj")
        {
            found_qkv = true;
            EXPECT_EQ(stage.tp_mode, TPMode::ColumnParallel);
        }
        if (stage.name == "wo_proj")
        {
            EXPECT_EQ(stage.tp_mode, TPMode::RowParallel);
        }
    }

    EXPECT_TRUE(found_qkv) << "qkv_proj stage not found in Qwen2 schema";
}

// ============================================================================
// GraphBuilder Tests
// ============================================================================

TEST_F(Test__GraphResolver, GraphBuilder_BuildsFromResolvedSpec)
{
    GraphResolverConfig config = default_config_;
    config.n_layers = 1;

    ResolvedGraphSpec resolved = resolver_.resolve(test_schema_, config, empty_tensors_);

    // Build the graph
    ComputeGraph graph = GraphBuilder::build(resolved);

    // Graph should have nodes
    EXPECT_GT(graph.size(), 0);
}
