/**
 * @file Test__Qwen2GraphSchema.cpp
 * @brief Unit tests for schema-based Qwen2 graph building (Phase 4d)
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the declarative three-layer architecture:
 *   Qwen2Schema -> GraphResolver -> GraphBuilder
 *
 * This validates that:
 * 1. Qwen2SchemaFactory creates valid schemas
 * 2. GraphResolver resolves schemas correctly
 * 3. Schema-based graphs work for inference
 */

#include <gtest/gtest.h>
#include "models/qwen/Qwen2Schema.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// ============================================================================
// Schema Factory Tests
// ============================================================================

/**
 * Test that Qwen2SchemaFactory creates a complete schema
 */
TEST(Test__Qwen2GraphSchema, SchemaFactory_CreatesValidSchema)
{
    Qwen2SchemaFactory factory;

    // Check architecture name
    EXPECT_EQ(factory.architectureName(), "qwen2");

    // Create schema
    GraphSchema schema = factory.createSchema();

    // Verify basic structure
    EXPECT_EQ(schema.name, "qwen2");
    EXPECT_EQ(schema.version, "1.0");

    // Verify required params are specified
    EXPECT_FALSE(schema.required_params.empty());
    EXPECT_NE(std::find(schema.required_params.begin(), schema.required_params.end(), "n_layers"),
              schema.required_params.end());
    EXPECT_NE(std::find(schema.required_params.begin(), schema.required_params.end(), "d_model"),
              schema.required_params.end());
}

/**
 * Test that schema has embedding stage
 */
TEST(Test__Qwen2GraphSchema, SchemaFactory_HasEmbeddingStage)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    EXPECT_EQ(schema.embedding.name, "embedding");
    EXPECT_EQ(schema.embedding.type, StageType::Embedding);

    // Should have token_ids input
    bool has_token_ids = false;
    for (const auto &input : schema.embedding.inputs)
    {
        if (input.name == "token_ids")
        {
            has_token_ids = true;
            break;
        }
    }
    EXPECT_TRUE(has_token_ids) << "Embedding should have token_ids input";

    // Should have hidden output
    bool has_hidden = false;
    for (const auto &output : schema.embedding.outputs)
    {
        if (output.name == "hidden")
        {
            has_hidden = true;
            break;
        }
    }
    EXPECT_TRUE(has_hidden) << "Embedding should have hidden output";
}

/**
 * Test that schema has layer template with attention stages
 */
TEST(Test__Qwen2GraphSchema, SchemaFactory_HasAttentionStages)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Should have attention stages
    EXPECT_GE(schema.layer_template.attention_stages.size(), 5u)
        << "Layer template should have at least 5 attention stages (norm, qkv, rope, attention, wo)";

    // Find attn_norm stage
    auto attn_norm_it = std::find_if(
        schema.layer_template.attention_stages.begin(),
        schema.layer_template.attention_stages.end(),
        [](const StageSpec &s)
        { return s.name == "attn_norm"; });
    EXPECT_NE(attn_norm_it, schema.layer_template.attention_stages.end())
        << "Should have attn_norm stage";

    // Find qkv_proj stage
    auto qkv_it = std::find_if(
        schema.layer_template.attention_stages.begin(),
        schema.layer_template.attention_stages.end(),
        [](const StageSpec &s)
        { return s.name == "qkv_proj"; });
    EXPECT_NE(qkv_it, schema.layer_template.attention_stages.end())
        << "Should have qkv_proj stage";

    // Find attention stage
    auto attn_it = std::find_if(
        schema.layer_template.attention_stages.begin(),
        schema.layer_template.attention_stages.end(),
        [](const StageSpec &s)
        { return s.name == "attention"; });
    EXPECT_NE(attn_it, schema.layer_template.attention_stages.end())
        << "Should have attention stage";
}

/**
 * Test that schema has layer template with FFN stages
 */
TEST(Test__Qwen2GraphSchema, SchemaFactory_HasFFNStages)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Should have FFN stages
    EXPECT_GE(schema.layer_template.ffn_stages.size(), 4u)
        << "Layer template should have at least 4 FFN stages (norm, gate_up, swiglu, down)";

    // Find ffn_norm stage
    auto ffn_norm_it = std::find_if(
        schema.layer_template.ffn_stages.begin(),
        schema.layer_template.ffn_stages.end(),
        [](const StageSpec &s)
        { return s.name == "ffn_norm"; });
    EXPECT_NE(ffn_norm_it, schema.layer_template.ffn_stages.end())
        << "Should have ffn_norm stage";

    // Find gate_up_proj stage
    auto gate_up_it = std::find_if(
        schema.layer_template.ffn_stages.begin(),
        schema.layer_template.ffn_stages.end(),
        [](const StageSpec &s)
        { return s.name == "gate_up_proj"; });
    EXPECT_NE(gate_up_it, schema.layer_template.ffn_stages.end())
        << "Should have gate_up_proj stage";

    // Find down_proj stage
    auto down_it = std::find_if(
        schema.layer_template.ffn_stages.begin(),
        schema.layer_template.ffn_stages.end(),
        [](const StageSpec &s)
        { return s.name == "down_proj"; });
    EXPECT_NE(down_it, schema.layer_template.ffn_stages.end())
        << "Should have down_proj stage";
}

/**
 * Test that schema has LM head stages
 */
TEST(Test__Qwen2GraphSchema, SchemaFactory_HasLMHeadStages)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Should have LM head stages
    EXPECT_GE(schema.lm_head_stages.size(), 2u)
        << "Should have at least 2 LM head stages (final_norm, lm_head)";

    // Find final_norm stage
    auto final_norm_it = std::find_if(
        schema.lm_head_stages.begin(),
        schema.lm_head_stages.end(),
        [](const StageSpec &s)
        { return s.name == "final_norm"; });
    EXPECT_NE(final_norm_it, schema.lm_head_stages.end())
        << "Should have final_norm stage";

    // Find lm_head stage
    auto lm_head_it = std::find_if(
        schema.lm_head_stages.begin(),
        schema.lm_head_stages.end(),
        [](const StageSpec &s)
        { return s.name == "lm_head"; });
    EXPECT_NE(lm_head_it, schema.lm_head_stages.end())
        << "Should have lm_head stage";
}

// ============================================================================
// TensorRef Optional Weight Tests
// ============================================================================

/**
 * Test TensorRef default construction is required (not optional)
 */
TEST(Test__Qwen2GraphSchema, TensorRef_DefaultIsRequired)
{
    TensorRef ref;
    EXPECT_FALSE(ref.is_optional) << "Default TensorRef should be required";
}

/**
 * Test TensorRef construction from string is required
 */
TEST(Test__Qwen2GraphSchema, TensorRef_StringConstructorIsRequired)
{
    TensorRef ref("weights.wq");
    EXPECT_EQ(ref.name, "weights.wq");
    EXPECT_FALSE(ref.is_optional) << "String-constructed TensorRef should be required";
}

/**
 * Test TensorRef named constructor creates required reference
 */
TEST(Test__Qwen2GraphSchema, TensorRef_NamedConstructorIsRequired)
{
    TensorRef ref("weights.wq", BufferSemantic::Input);
    EXPECT_EQ(ref.name, "weights.wq");
    EXPECT_EQ(ref.semantic, BufferSemantic::Input);
    EXPECT_FALSE(ref.is_optional) << "Named constructor TensorRef should be required";
}

/**
 * Test TensorRef::optional() factory creates optional reference
 */
TEST(Test__Qwen2GraphSchema, TensorRef_OptionalFactoryCreatesOptional)
{
    TensorRef ref = TensorRef::optional("weights.q_bias", BufferSemantic::Input);
    EXPECT_EQ(ref.name, "weights.q_bias");
    EXPECT_EQ(ref.semantic, BufferSemantic::Input);
    EXPECT_TRUE(ref.is_optional) << "TensorRef::optional() should create optional reference";
}

/**
 * Test TensorRef full constructor with explicit optional flag
 */
TEST(Test__Qwen2GraphSchema, TensorRef_ExplicitOptionalFlag)
{
    TensorRef required_ref("weights.wq", BufferSemantic::Input, false);
    EXPECT_FALSE(required_ref.is_optional);

    TensorRef optional_ref("weights.q_bias", BufferSemantic::Input, true);
    EXPECT_TRUE(optional_ref.is_optional);
}

// ============================================================================
// isWeightOptional Tests - Qwen2 Architecture
// ============================================================================

/**
 * Test that QKV weights are required
 */
TEST(Test__Qwen2GraphSchema, IsWeightOptional_QKVWeightsAreRequired)
{
    Qwen2SchemaFactory factory;

    // Q/K/V projection weights are required
    EXPECT_FALSE(factory.isWeightOptional("blk.0.attn_q.weight"))
        << "attn_q.weight should be required";
    EXPECT_FALSE(factory.isWeightOptional("blk.0.attn_k.weight"))
        << "attn_k.weight should be required";
    EXPECT_FALSE(factory.isWeightOptional("blk.0.attn_v.weight"))
        << "attn_v.weight should be required";

    // Test different layer indices
    EXPECT_FALSE(factory.isWeightOptional("blk.15.attn_q.weight"));
    EXPECT_FALSE(factory.isWeightOptional("blk.23.attn_k.weight"));
}

/**
 * Test that QKV biases are optional (not all Qwen2 models have them)
 */
TEST(Test__Qwen2GraphSchema, IsWeightOptional_QKVBiasesAreOptional)
{
    Qwen2SchemaFactory factory;

    // Q/K/V biases are optional (some Qwen2 variants don't have them)
    EXPECT_TRUE(factory.isWeightOptional("blk.0.attn_q.bias"))
        << "attn_q.bias should be optional";
    EXPECT_TRUE(factory.isWeightOptional("blk.0.attn_k.bias"))
        << "attn_k.bias should be optional";
    EXPECT_TRUE(factory.isWeightOptional("blk.0.attn_v.bias"))
        << "attn_v.bias should be optional";

    // Test different layer indices
    EXPECT_TRUE(factory.isWeightOptional("blk.15.attn_q.bias"));
    EXPECT_TRUE(factory.isWeightOptional("blk.23.attn_k.bias"));
}

/**
 * Test that Wo (attention output) weight is required
 */
TEST(Test__Qwen2GraphSchema, IsWeightOptional_WoWeightIsRequired)
{
    Qwen2SchemaFactory factory;

    EXPECT_FALSE(factory.isWeightOptional("blk.0.attn_output.weight"))
        << "attn_output.weight should be required";
    EXPECT_FALSE(factory.isWeightOptional("blk.15.attn_output.weight"));
}

/**
 * Test that FFN weights are required
 */
TEST(Test__Qwen2GraphSchema, IsWeightOptional_FFNWeightsAreRequired)
{
    Qwen2SchemaFactory factory;

    EXPECT_FALSE(factory.isWeightOptional("blk.0.ffn_gate.weight"))
        << "ffn_gate.weight should be required";
    EXPECT_FALSE(factory.isWeightOptional("blk.0.ffn_up.weight"))
        << "ffn_up.weight should be required";
    EXPECT_FALSE(factory.isWeightOptional("blk.0.ffn_down.weight"))
        << "ffn_down.weight should be required";

    // Test different layer indices
    EXPECT_FALSE(factory.isWeightOptional("blk.23.ffn_gate.weight"));
    EXPECT_FALSE(factory.isWeightOptional("blk.23.ffn_up.weight"));
    EXPECT_FALSE(factory.isWeightOptional("blk.23.ffn_down.weight"));
}

/**
 * Test that norm weights are required
 */
TEST(Test__Qwen2GraphSchema, IsWeightOptional_NormWeightsAreRequired)
{
    Qwen2SchemaFactory factory;

    // Layer norms are required
    EXPECT_FALSE(factory.isWeightOptional("blk.0.attn_norm.weight"))
        << "attn_norm.weight should be required";
    EXPECT_FALSE(factory.isWeightOptional("blk.0.ffn_norm.weight"))
        << "ffn_norm.weight should be required";

    // Model-level norms are required
    EXPECT_FALSE(factory.isWeightOptional("output_norm.weight"))
        << "output_norm.weight should be required";
}

/**
 * Test that model-level weights are required
 */
TEST(Test__Qwen2GraphSchema, IsWeightOptional_ModelLevelWeightsAreRequired)
{
    Qwen2SchemaFactory factory;

    EXPECT_FALSE(factory.isWeightOptional("token_embd.weight"))
        << "token_embd.weight should be required";
    EXPECT_FALSE(factory.isWeightOptional("output.weight"))
        << "output.weight (LM head) should be required";
}

/**
 * Test that schema QKV projection has optional bias inputs marked correctly
 */
TEST(Test__Qwen2GraphSchema, Schema_QKVBiasInputsMarkedOptional)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Find qkv_proj stage
    auto qkv_it = std::find_if(
        schema.layer_template.attention_stages.begin(),
        schema.layer_template.attention_stages.end(),
        [](const StageSpec &s)
        { return s.name == "qkv_proj"; });

    ASSERT_NE(qkv_it, schema.layer_template.attention_stages.end())
        << "Should have qkv_proj stage";

    // Count required and optional inputs
    int required_count = 0;
    int optional_count = 0;

    for (const auto &input : qkv_it->inputs)
    {
        if (input.is_optional)
        {
            optional_count++;
            // Bias inputs should be optional
            EXPECT_TRUE(input.name.find("bias") != std::string::npos)
                << "Optional input " << input.name << " should contain 'bias'";
        }
        else
        {
            required_count++;
        }
    }

    // QKV should have 4 required inputs (normalized, wq, wk, wv) + 3 optional (q_bias, k_bias, v_bias)
    EXPECT_EQ(required_count, 4) << "Should have 4 required inputs (normalized + 3 weights)";
    EXPECT_EQ(optional_count, 3) << "Should have 3 optional inputs (Q/K/V biases)";
}

// ============================================================================
// TP Annotation Tests
// ============================================================================

/**
 * Test that QKV projection has column-parallel annotation
 */
TEST(Test__Qwen2GraphSchema, Schema_QKVIsColumnParallel)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    auto qkv_it = std::find_if(
        schema.layer_template.attention_stages.begin(),
        schema.layer_template.attention_stages.end(),
        [](const StageSpec &s)
        { return s.name == "qkv_proj"; });

    ASSERT_NE(qkv_it, schema.layer_template.attention_stages.end());
    EXPECT_EQ(qkv_it->tp_mode, TPMode::ColumnParallel)
        << "QKV projection should be column-parallel";
}

/**
 * Test that Wo projection has row-parallel annotation
 */
TEST(Test__Qwen2GraphSchema, Schema_WoIsRowParallel)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    auto wo_it = std::find_if(
        schema.layer_template.attention_stages.begin(),
        schema.layer_template.attention_stages.end(),
        [](const StageSpec &s)
        { return s.name == "wo_proj"; });

    ASSERT_NE(wo_it, schema.layer_template.attention_stages.end());
    EXPECT_EQ(wo_it->tp_mode, TPMode::RowParallel)
        << "Wo projection should be row-parallel";
}

/**
 * Test that FFN down projection has row-parallel annotation
 */
TEST(Test__Qwen2GraphSchema, Schema_FFNDownIsRowParallel)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    auto down_it = std::find_if(
        schema.layer_template.ffn_stages.begin(),
        schema.layer_template.ffn_stages.end(),
        [](const StageSpec &s)
        { return s.name == "down_proj"; });

    ASSERT_NE(down_it, schema.layer_template.ffn_stages.end());
    EXPECT_EQ(down_it->tp_mode, TPMode::RowParallel)
        << "FFN down projection should be row-parallel";
}

/**
 * Test that LM head has column-parallel annotation
 */
TEST(Test__Qwen2GraphSchema, Schema_LMHeadIsColumnParallel)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    auto lm_head_it = std::find_if(
        schema.lm_head_stages.begin(),
        schema.lm_head_stages.end(),
        [](const StageSpec &s)
        { return s.name == "lm_head"; });

    ASSERT_NE(lm_head_it, schema.lm_head_stages.end());
    EXPECT_EQ(lm_head_it->tp_mode, TPMode::ColumnParallel)
        << "LM head should be column-parallel";
}

// ============================================================================
// Execution Policy Key Tests
// ============================================================================

/**
 * Test that stages have correct execution policy keys
 */
TEST(Test__Qwen2GraphSchema, Schema_HasExecutionPolicyKeys)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Embedding should have exec_embedding
    EXPECT_EQ(schema.embedding.exec_policy_key, "exec_embedding");

    // Find attn_norm - should have exec_rmsnorm
    auto attn_norm_it = std::find_if(
        schema.layer_template.attention_stages.begin(),
        schema.layer_template.attention_stages.end(),
        [](const StageSpec &s)
        { return s.name == "attn_norm"; });
    if (attn_norm_it != schema.layer_template.attention_stages.end())
    {
        EXPECT_EQ(attn_norm_it->exec_policy_key, "exec_rmsnorm");
    }

    // Find attention - should have exec_attention
    auto attn_it = std::find_if(
        schema.layer_template.attention_stages.begin(),
        schema.layer_template.attention_stages.end(),
        [](const StageSpec &s)
        { return s.name == "attention"; });
    if (attn_it != schema.layer_template.attention_stages.end())
    {
        EXPECT_EQ(attn_it->exec_policy_key, "exec_attention");
    }
}

// ============================================================================
// Execution Policy Tests
// ============================================================================

/**
 * Test that ExecutionPolicyFlags defaults enable all stages
 */
TEST(Test__Qwen2GraphSchema, ExecutionPolicyFlags_DefaultsEnable)
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

/**
 * Test shouldExecute() method
 */
TEST(Test__Qwen2GraphSchema, ExecutionPolicyFlags_ShouldExecute)
{
    ExecutionPolicyFlags flags;

    // All enabled by default
    EXPECT_TRUE(flags.shouldExecute("exec_embedding"));
    EXPECT_TRUE(flags.shouldExecute("exec_rmsnorm"));
    EXPECT_TRUE(flags.shouldExecute("exec_attention"));
    EXPECT_TRUE(flags.shouldExecute("exec_lm_head"));

    // Disable one
    flags.exec_attention = false;
    EXPECT_FALSE(flags.shouldExecute("exec_attention"));
    EXPECT_TRUE(flags.shouldExecute("exec_embedding")); // Others still enabled

    // Unknown keys should return true (don't skip)
    EXPECT_TRUE(flags.shouldExecute("unknown_key"));
}

// ============================================================================
// GraphResolverConfig Tests
// ============================================================================

/**
 * Test GraphResolverConfig defaults
 */
TEST(Test__Qwen2GraphSchema, ResolverConfig_Defaults)
{
    GraphResolverConfig config;

    EXPECT_EQ(config.world_size, 1);
    EXPECT_EQ(config.rank, 0);
    EXPECT_EQ(config.batch_size, 1);
    EXPECT_EQ(config.seq_len, 0);
    EXPECT_FALSE(config.has_kv_cache);
    EXPECT_EQ(config.cached_tokens, 0);
}

// ============================================================================
// TensorContext Tests
// ============================================================================

/**
 * Test TensorContext basic buffer resolution
 */
TEST(Test__Qwen2GraphSchema, TensorContext_ResolvesBuffers)
{
    TensorContext ctx;

    // Add a mock buffer
    int dummy_tensor = 42; // Just a placeholder
    TensorBase *mock_ptr = reinterpret_cast<TensorBase *>(&dummy_tensor);
    ctx.buffers["hidden"] = mock_ptr;

    // Resolve reference - use string name directly
    TensorRef ref("hidden", BufferSemantic::Input);

    TensorBase *resolved = ctx.resolve(ref);
    EXPECT_EQ(resolved, mock_ptr);
}

/**
 * Test TensorContext weight resolution
 */
TEST(Test__Qwen2GraphSchema, TensorContext_ResolvesWeights)
{
    TensorContext ctx;

    // Add a mock weight
    int dummy_tensor = 42;
    TensorBase *mock_ptr = reinterpret_cast<TensorBase *>(&dummy_tensor);
    ctx.model_weights["final_norm"] = mock_ptr;

    // Resolve reference using weights prefix
    TensorRef ref("weights.final_norm", BufferSemantic::Input);

    TensorBase *resolved = ctx.resolve(ref);
    EXPECT_EQ(resolved, mock_ptr);
}

/**
 * Test TensorContext layer weight resolution
 */
TEST(Test__Qwen2GraphSchema, TensorContext_ResolvesLayerWeights)
{
    TensorContext ctx;

    // Set up layer weight accessor
    int dummy_tensor = 42;
    TensorBase *mock_ptr = reinterpret_cast<TensorBase *>(&dummy_tensor);
    ctx.get_layer_weight = [mock_ptr](int layer_idx, const std::string &name) -> TensorBase *
    {
        if (layer_idx == 0 && name == "wq")
        {
            return mock_ptr;
        }
        return nullptr;
    };

    // Resolve reference with weights prefix - layer_idx determines which layer
    TensorRef ref("weights.wq", BufferSemantic::Input);

    TensorBase *resolved = ctx.resolve(ref, 0); // layer 0
    EXPECT_EQ(resolved, mock_ptr);

    // Different layer should return null (our mock only returns for layer 0)
    TensorBase *resolved_other = ctx.resolve(ref, 1);
    EXPECT_EQ(resolved_other, nullptr);
}

// ============================================================================
// Buffer Specification Tests (Schema-Based)
// ============================================================================

/**
 * Test that schema has layer buffers with aliasing info
 */
TEST(Test__Qwen2GraphSchema, Schema_HasLayerBuffersWithAliasing)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Should have layer buffers
    EXPECT_GE(schema.layer_buffers.size(), 7u)
        << "Schema should have at least 7 layer buffers (normalized, Q, K, V, attn_output, attn_proj, gate, up)";

    // Find Q buffer - should have attn_scratch alias group
    auto q_it = std::find_if(
        schema.layer_buffers.begin(),
        schema.layer_buffers.end(),
        [](const BufferSpec &b)
        { return b.name == "Q"; });
    ASSERT_NE(q_it, schema.layer_buffers.end()) << "Should have Q buffer";
    EXPECT_EQ(q_it->alias_group, "attn_scratch") << "Q should be in attn_scratch alias group";
    EXPECT_EQ(q_it->semantic, BufferSemantic::Scratch) << "Q should be Scratch semantic";

    // Find gate buffer - should have ffn_scratch alias group
    auto gate_it = std::find_if(
        schema.layer_buffers.begin(),
        schema.layer_buffers.end(),
        [](const BufferSpec &b)
        { return b.name == "gate"; });
    ASSERT_NE(gate_it, schema.layer_buffers.end()) << "Should have gate buffer";
    EXPECT_EQ(gate_it->alias_group, "ffn_scratch") << "gate should be in ffn_scratch alias group";
}

/**
 * Test that schema has model buffers
 */
TEST(Test__Qwen2GraphSchema, Schema_HasModelBuffers)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Should have model buffers
    EXPECT_GE(schema.model_buffers.size(), 2u)
        << "Schema should have at least 2 model buffers (hidden, logits)";

    // Find hidden buffer
    auto hidden_it = std::find_if(
        schema.model_buffers.begin(),
        schema.model_buffers.end(),
        [](const BufferSpec &b)
        { return b.name == "hidden"; });
    ASSERT_NE(hidden_it, schema.model_buffers.end()) << "Should have hidden buffer";
    EXPECT_EQ(hidden_it->semantic, BufferSemantic::InOut) << "hidden should be InOut semantic";
    EXPECT_TRUE(hidden_it->alias_group.empty()) << "hidden should not be aliased";

    // Find logits buffer
    auto logits_it = std::find_if(
        schema.model_buffers.begin(),
        schema.model_buffers.end(),
        [](const BufferSpec &b)
        { return b.name == "logits"; });
    ASSERT_NE(logits_it, schema.model_buffers.end()) << "Should have logits buffer";
    EXPECT_EQ(logits_it->semantic, BufferSemantic::Output) << "logits should be Output semantic";
}

/**
 * Test that schema has alias group definitions
 */
TEST(Test__Qwen2GraphSchema, Schema_HasAliasGroups)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Should have alias groups
    EXPECT_GE(schema.alias_groups.size(), 2u)
        << "Schema should have at least 2 alias groups (attn_scratch, ffn_scratch)";

    // Find attn_scratch group
    auto attn_group_it = std::find_if(
        schema.alias_groups.begin(),
        schema.alias_groups.end(),
        [](const AliasGroupSpec &g)
        { return g.name == "attn_scratch"; });
    ASSERT_NE(attn_group_it, schema.alias_groups.end()) << "Should have attn_scratch alias group";
    EXPECT_GE(attn_group_it->buffer_names.size(), 4u)
        << "attn_scratch should have at least 4 buffers (Q, K, V, attn_output)";
    EXPECT_GT(attn_group_it->estimated_savings_percent, 0.0f)
        << "Should have non-zero estimated savings";

    // Find ffn_scratch group
    auto ffn_group_it = std::find_if(
        schema.alias_groups.begin(),
        schema.alias_groups.end(),
        [](const AliasGroupSpec &g)
        { return g.name == "ffn_scratch"; });
    ASSERT_NE(ffn_group_it, schema.alias_groups.end()) << "Should have ffn_scratch alias group";
    EXPECT_GE(ffn_group_it->buffer_names.size(), 2u)
        << "ffn_scratch should have at least 2 buffers (gate, up)";
}

/**
 * Test that buffer alias priorities are set correctly
 */
TEST(Test__Qwen2GraphSchema, Schema_BufferAliasPriorities)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Find Q and K buffers to verify priorities
    const BufferSpec *q_spec = nullptr;
    const BufferSpec *k_spec = nullptr;

    for (const auto &buf : schema.layer_buffers)
    {
        if (buf.name == "Q")
            q_spec = &buf;
        if (buf.name == "K")
            k_spec = &buf;
    }

    ASSERT_NE(q_spec, nullptr);
    ASSERT_NE(k_spec, nullptr);

    // Q should have higher priority than K (Q is larger due to more heads)
    EXPECT_GT(q_spec->alias_priority, k_spec->alias_priority)
        << "Q should have higher alias priority than K (Q is larger in GQA)";
}

/**
 * Test that BufferSpec has description field
 */
TEST(Test__Qwen2GraphSchema, Schema_BufferHasDescription)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Find Q buffer and check description
    auto q_it = std::find_if(
        schema.layer_buffers.begin(),
        schema.layer_buffers.end(),
        [](const BufferSpec &b)
        { return b.name == "Q"; });

    ASSERT_NE(q_it, schema.layer_buffers.end());
    EXPECT_FALSE(q_it->description.empty())
        << "Buffer should have a description for documentation";
}
// ============================================================================
// BufferAllocator Tests
// ============================================================================

/**
 * Test that BufferAllocator evaluates simple formulas correctly
 */
TEST(Test__Qwen2GraphSchema, BufferAllocator_EvaluatesSimpleFormulas)
{
    GraphResolverConfig config{};
    config.d_model = 896;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.n_layers = 24;
    config.d_ff = 4864;
    config.vocab_size = 151936;
    config.seq_len = 512;

    // Direct dimensions
    EXPECT_EQ(BufferAllocator::evaluateFormula("d_model", config), 896u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("seq_len", config), 512u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("n_heads", config), 14u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("n_kv_heads", config), 2u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("head_dim", config), 64u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("d_ff", config), 4864u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("vocab_size", config), 151936u);
}

/**
 * Test that BufferAllocator evaluates tensor-parallel formulas
 */
TEST(Test__Qwen2GraphSchema, BufferAllocator_EvaluatesTPFormulas)
{
    GraphResolverConfig config{};
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.d_ff = 4864;
    config.vocab_size = 151936;
    config.local_n_heads = 7; // TP=2
    config.local_n_kv_heads = 1;
    config.local_d_ff = 2432;
    config.local_vocab = 75968;

    // Local dimensions
    EXPECT_EQ(BufferAllocator::evaluateFormula("local_n_heads", config), 7u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("local_n_kv_heads", config), 1u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("local_d_ff", config), 2432u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("local_vocab", config), 75968u);

    // Computed dimensions
    EXPECT_EQ(BufferAllocator::evaluateFormula("qkv_dim", config), 14u * 64u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("local_qkv_dim", config), 7u * 64u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("kv_dim", config), 2u * 64u);
    EXPECT_EQ(BufferAllocator::evaluateFormula("local_kv_dim", config), 1u * 64u);
}

/**
 * Test that BufferAllocator resolves a single BufferSpec
 */
TEST(Test__Qwen2GraphSchema, BufferAllocator_ResolvesSingleBuffer)
{
    GraphResolverConfig config{};
    config.seq_len = 512;
    config.d_model = 896;
    config.local_n_heads = 14;
    config.head_dim = 64;

    // Create a BufferSpec with shape formulas
    BufferSpec spec("Q", {"seq_len", "local_qkv_dim"}, "fp32",
                    BufferSemantic::Scratch, "attn_scratch", 100, "Query projection");

    // Resolve it
    ResolvedBufferSpec resolved = BufferAllocator::resolve(spec, config);

    EXPECT_EQ(resolved.name, "Q");
    EXPECT_EQ(resolved.shape.size(), 2u);
    EXPECT_EQ(resolved.shape[0], 512u);      // seq_len
    EXPECT_EQ(resolved.shape[1], 14u * 64u); // local_n_heads * head_dim = 896
    EXPECT_EQ(resolved.dtype, "fp32");
    EXPECT_EQ(resolved.semantic, BufferSemantic::Scratch);
    EXPECT_EQ(resolved.alias_group, "attn_scratch");
    EXPECT_EQ(resolved.alias_priority, 100);
}

/**
 * Test that BufferAllocator resolves all buffers from schema
 */
TEST(Test__Qwen2GraphSchema, BufferAllocator_ResolvesAllBuffers)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    GraphResolverConfig config{};
    config.d_model = 896;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.n_layers = 24;
    config.d_ff = 4864;
    config.vocab_size = 151936;
    config.seq_len = 512;
    config.local_n_heads = 14;
    config.local_n_kv_heads = 2;
    config.local_d_ff = 4864;
    config.local_vocab = 151936;

    // Resolve all buffers
    auto resolved = BufferAllocator::resolveAll(schema, config);

    // Should have same count as schema buffers
    size_t total_schema_buffers = schema.layer_buffers.size() + schema.model_buffers.size();
    EXPECT_EQ(resolved.size(), total_schema_buffers);

    // All shapes should be non-empty
    for (const auto &buf : resolved)
    {
        EXPECT_FALSE(buf.shape.empty())
            << "Buffer " << buf.name << " should have non-empty shape";
        for (size_t dim : buf.shape)
        {
            EXPECT_GT(dim, 0u)
                << "Buffer " << buf.name << " has zero dimension";
        }
    }
}

/**
 * Test that BufferAllocator estimates memory savings correctly
 */
TEST(Test__Qwen2GraphSchema, BufferAllocator_EstimatesMemorySavings)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    GraphResolverConfig config{};
    config.d_model = 896;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.n_layers = 24;
    config.d_ff = 4864;
    config.vocab_size = 151936;
    config.seq_len = 512;
    config.local_n_heads = 14;
    config.local_n_kv_heads = 2;
    config.local_d_ff = 4864;
    config.local_vocab = 151936;

    auto [original, optimized] = BufferAllocator::estimateMemorySavings(schema, config);

    // Optimized should be less than original (aliasing saves memory)
    EXPECT_LT(optimized, original)
        << "Aliasing should reduce memory from " << original << " to " << optimized;

    // Calculate savings percent
    float savings_percent = 100.0f * (1.0f - static_cast<float>(optimized) / static_cast<float>(original));
    EXPECT_GT(savings_percent, 0.0f)
        << "Memory savings should be positive";

    // Log the savings for visibility
    std::cout << "[BufferAllocator] Memory: original=" << (original / 1024 / 1024) << "MB"
              << ", optimized=" << (optimized / 1024 / 1024) << "MB"
              << ", savings=" << savings_percent << "%" << std::endl;
}

/**
 * Test ResolvedBufferSpec totalBytes calculation
 */
TEST(Test__Qwen2GraphSchema, ResolvedBufferSpec_CalculatesTotalBytes)
{
    ResolvedBufferSpec spec;
    spec.shape = {512, 896}; // seq_len x d_model
    spec.dtype = "fp32";

    // 512 * 896 * 4 bytes = 1,835,008 bytes
    EXPECT_EQ(spec.totalBytes(), 512u * 896u * 4u);

    // Test bf16
    spec.dtype = "bf16";
    EXPECT_EQ(spec.totalBytes(), 512u * 896u * 2u);

    // Test q8_0
    spec.dtype = "q8_0";
    EXPECT_EQ(spec.totalBytes(), 512u * 896u * 1u);
}
/**
 * Test BufferAllocator resolveLayerBuffers returns StageBufferRequirements
 */
TEST(Test__Qwen2GraphSchema, BufferAllocator_ResolveLayerBuffers)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    GraphResolverConfig config{};
    config.d_model = 896;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.seq_len = 512;
    config.local_n_heads = 14;
    config.local_n_kv_heads = 2;
    config.local_d_ff = 4864;

    // Resolve layer buffers
    auto reqs = BufferAllocator::resolveLayerBuffers(schema, config);

    // Should have buffers
    EXPECT_FALSE(reqs.buffers.empty()) << "Should have layer buffers";

    // All buffers should have non-empty shapes
    for (const auto &buf : reqs.buffers)
    {
        EXPECT_FALSE(buf.shape.empty())
            << "Buffer " << buf.name << " should have non-empty shape";
        for (size_t dim : buf.shape)
        {
            EXPECT_GT(dim, 0u)
                << "Buffer " << buf.name << " has zero dimension";
        }
    }
}

/**
 * Test BufferAllocator resolveModelBuffers returns StageBufferRequirements
 */
TEST(Test__Qwen2GraphSchema, BufferAllocator_ResolveModelBuffers)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    GraphResolverConfig config{};
    config.d_model = 896;
    config.vocab_size = 151936;
    config.seq_len = 512;
    config.local_vocab = 151936;

    // Resolve model buffers
    auto reqs = BufferAllocator::resolveModelBuffers(schema, config);

    // Should have model buffers
    EXPECT_FALSE(reqs.buffers.empty()) << "Should have model buffers";

    // Find hidden buffer
    auto hidden_it = std::find_if(
        reqs.buffers.begin(),
        reqs.buffers.end(),
        [](const BufferDescriptor &b)
        { return b.name == "hidden"; });
    ASSERT_NE(hidden_it, reqs.buffers.end()) << "Should have hidden buffer";
    EXPECT_EQ(hidden_it->shape.size(), 2u);
    EXPECT_EQ(hidden_it->shape[0], 512u); // seq_len
    EXPECT_EQ(hidden_it->shape[1], 896u); // d_model

    // Find logits buffer
    auto logits_it = std::find_if(
        reqs.buffers.begin(),
        reqs.buffers.end(),
        [](const BufferDescriptor &b)
        { return b.name == "logits"; });
    ASSERT_NE(logits_it, reqs.buffers.end()) << "Should have logits buffer";
    EXPECT_EQ(logits_it->shape.size(), 2u);
    EXPECT_EQ(logits_it->shape[0], 512u);    // seq_len
    EXPECT_EQ(logits_it->shape[1], 151936u); // vocab_size
}
