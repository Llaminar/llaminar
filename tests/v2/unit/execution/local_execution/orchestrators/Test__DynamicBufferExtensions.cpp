/**
 * @file Test__DynamicBufferExtensions.cpp
 * @brief Unit tests for the dynamic buffer extension system
 * @author David Sanftenberg
 * @date June 2026
 *
 * Tests the complete buffer wiring chain for model-specific (extension)
 * buffers that flow through infrastructure without hardcoding:
 *
 *   BufferArena::forEachRegistered()
 *     → initializeInferenceStateFromArena() auto-discovery
 *       → InferenceState::extension_buffers
 *         → toModelBuffers()
 *           → ActivationBuffers::extensions / get()
 *
 * Uses MockGraphBuilder (IGraphBuilder interface mock) with configurable
 * schema/resolver to prove end-to-end wiring without depending on any
 * real model (Qwen2, Qwen3.5, etc.).
 */

#include <gtest/gtest.h>
#include <unordered_set>

#include "backends/DeviceId.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "memory/BufferArena.h"
#include "memory/BufferId.h"
#include "models/GraphTypes.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// ============================================================================
// Part 1: BufferArena::forEachRegistered() Tests
// ============================================================================

TEST(Test__DynamicBufferExtensions, ForEachRegistered_EmptyArena)
{
    BufferArena arena;
    int count = 0;
    arena.forEachRegistered([&](BufferId)
                            { ++count; });
    EXPECT_EQ(count, 0) << "Empty arena should iterate zero buffers";
}

TEST(Test__DynamicBufferExtensions, ForEachRegistered_CoreBuffersOnly)
{
    BufferArena arena;
    arena.registerBuffer(BufferId::HIDDEN_STATE, 64, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::LOGITS, 64, 151936, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::NORMALIZED, 64, 896, "FP32", DeviceId::cpu());

    std::vector<BufferId> visited;
    arena.forEachRegistered([&](BufferId id)
                            { visited.push_back(id); });

    ASSERT_EQ(visited.size(), 3u);
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::HIDDEN_STATE), visited.end());
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::LOGITS), visited.end());
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::NORMALIZED), visited.end());
}

TEST(Test__DynamicBufferExtensions, ForEachRegistered_IncludesExtensionBufferIds)
{
    BufferArena arena;
    // Core
    arena.registerBuffer(BufferId::HIDDEN_STATE, 64, 896, "FP32", DeviceId::cpu());
    // Extensions (GDN model-specific)
    arena.registerBuffer(BufferId::GDN_QKV, 64, 3072, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GDN_Z, 64, 1024, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GDN_ALPHA, 64, 16, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GDN_BETA, 64, 16, "FP32", DeviceId::cpu());

    std::vector<BufferId> visited;
    arena.forEachRegistered([&](BufferId id)
                            { visited.push_back(id); });

    ASSERT_EQ(visited.size(), 5u);
    // Check all extension IDs are discovered
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::GDN_QKV), visited.end());
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::GDN_Z), visited.end());
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::GDN_ALPHA), visited.end());
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::GDN_BETA), visited.end());
}

TEST(Test__DynamicBufferExtensions, ForEachRegistered_SkipsUnregistered)
{
    BufferArena arena;
    arena.registerBuffer(BufferId::HIDDEN_STATE, 64, 896, "FP32", DeviceId::cpu());
    // Register one GDN buffer but NOT all of them
    arena.registerBuffer(BufferId::GDN_QKV, 64, 3072, "FP32", DeviceId::cpu());

    std::vector<BufferId> visited;
    arena.forEachRegistered([&](BufferId id)
                            { visited.push_back(id); });

    ASSERT_EQ(visited.size(), 2u);
    // These should NOT appear (never registered)
    EXPECT_EQ(std::find(visited.begin(), visited.end(), BufferId::GDN_Z), visited.end());
    EXPECT_EQ(std::find(visited.begin(), visited.end(), BufferId::LOGITS), visited.end());
    EXPECT_EQ(std::find(visited.begin(), visited.end(), BufferId::MOE_ROUTER_LOGITS), visited.end());
}

TEST(Test__DynamicBufferExtensions, ForEachRegistered_MoEBufferIds)
{
    BufferArena arena;
    arena.registerBuffer(BufferId::HIDDEN_STATE, 64, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::MOE_ROUTER_LOGITS, 64, 256, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::MOE_EXPERT_WEIGHTS, 64, 8, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::MOE_COMBINED_OUTPUT, 64, 896, "FP32", DeviceId::cpu());

    std::vector<BufferId> visited;
    arena.forEachRegistered([&](BufferId id)
                            { visited.push_back(id); });

    ASSERT_EQ(visited.size(), 4u);
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::MOE_ROUTER_LOGITS), visited.end());
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::MOE_EXPERT_WEIGHTS), visited.end());
    EXPECT_NE(std::find(visited.begin(), visited.end(), BufferId::MOE_COMBINED_OUTPUT), visited.end());
}

// ============================================================================
// Part 2: ActivationBuffers::extensions and get() Tests
// ============================================================================

TEST(Test__DynamicBufferExtensions, ActivationBuffers_GetReturnsNullForEmpty)
{
    ActivationBuffers buffers;
    EXPECT_EQ(buffers.get(BufferId::GDN_QKV), nullptr);
    EXPECT_EQ(buffers.get(BufferId::GDN_Z), nullptr);
    EXPECT_EQ(buffers.get(BufferId::MOE_ROUTER_LOGITS), nullptr);
    EXPECT_EQ(buffers.get(BufferId::HIDDEN_STATE), nullptr);
}

TEST(Test__DynamicBufferExtensions, ActivationBuffers_SetAndGetExtension)
{
    ActivationBuffers buffers;

    // Use stack-allocated dummy pointers (only testing wiring, not data)
    float dummy_qkv = 1.0f;
    float dummy_z = 2.0f;
    auto *ptr_qkv = reinterpret_cast<TensorBase *>(&dummy_qkv);
    auto *ptr_z = reinterpret_cast<TensorBase *>(&dummy_z);

    buffers.extensions[BufferId::GDN_QKV] = ptr_qkv;
    buffers.extensions[BufferId::GDN_Z] = ptr_z;

    EXPECT_EQ(buffers.get(BufferId::GDN_QKV), ptr_qkv);
    EXPECT_EQ(buffers.get(BufferId::GDN_Z), ptr_z);
    // Unset IDs still return nullptr
    EXPECT_EQ(buffers.get(BufferId::GDN_ALPHA), nullptr);
}

TEST(Test__DynamicBufferExtensions, ActivationBuffers_ExtensionsMapSupportsMultipleIds)
{
    ActivationBuffers buffers;

    // Simulate a model with many extension buffers
    float dummies[8];
    const BufferId ids[] = {
        BufferId::GDN_QKV, BufferId::GDN_Z,
        BufferId::GDN_ALPHA, BufferId::GDN_BETA,
        BufferId::ATTN_OUTPUT_GATE, BufferId::GDN_CONV_STATE,
        BufferId::GDN_RECURRENCE_IN, BufferId::GDN_RECURRENCE_OUT};

    for (int i = 0; i < 8; ++i)
    {
        dummies[i] = static_cast<float>(i + 1);
        buffers.extensions[ids[i]] = reinterpret_cast<TensorBase *>(&dummies[i]);
    }

    EXPECT_EQ(buffers.extensions.size(), 8u);
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_EQ(buffers.get(ids[i]), reinterpret_cast<TensorBase *>(&dummies[i]))
            << "Extension buffer " << bufferIdName(ids[i]) << " should be accessible via get()";
    }
}

TEST(Test__DynamicBufferExtensions, ActivationBuffers_NullExtensionIsRetrievable)
{
    ActivationBuffers buffers;
    // Explicitly store nullptr for a BufferId
    buffers.extensions[BufferId::GDN_QKV] = nullptr;

    // get() should return nullptr (not crash)
    EXPECT_EQ(buffers.get(BufferId::GDN_QKV), nullptr);
    // The key exists though
    EXPECT_NE(buffers.extensions.find(BufferId::GDN_QKV), buffers.extensions.end());
}

// ============================================================================
// Part 3: InferenceState::toModelBuffers() Extension Propagation
// ============================================================================

// Helper to build a minimal schema suitable for MockGraphBuilder that
// registers the required core buffers + optional extension buffers.
// Shape formulas use literal numbers so no complex dimension resolution
// is needed.
struct ExtensionBufferSpec
{
    std::string name;
    BufferId id;
    size_t rows;
    size_t cols;
};

static GraphSchema buildMinimalSchema(
    const std::vector<ExtensionBufferSpec> &extensions = {})
{
    GraphSchema schema;
    schema.name = "test_minimal";

    // Core layer buffers (the minimum set required by initializeInferenceStateFromArena)
    schema.layer_buffers = {
        BufferSpec("normalized", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("residual", {"64", "32"}, "fp32", BufferSemantic::InOut),
        BufferSpec("Q", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("K", {"64", "16"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("V", {"64", "16"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("attn_output", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("attn_proj", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("gate", {"64", "64"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("up", {"64", "64"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("ffn_output", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("workspace_scores", {"64", "64"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("workspace_context", {"64", "8"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("workspace_mask", {"64", "64"}, "fp32", BufferSemantic::Scratch),
    };

    // Extension layer buffers
    for (const auto &ext : extensions)
    {
        schema.layer_buffers.push_back(BufferSpec(
            ext.name,
            {std::to_string(ext.rows), std::to_string(ext.cols)},
            "fp32",
            BufferSemantic::Scratch));
    }

    // Core model buffers
    schema.model_buffers = {
        BufferSpec("hidden", {"64", "32"}, "fp32", BufferSemantic::InOut),
        BufferSpec("logits", {"64", "128"}, "fp32", BufferSemantic::Output),
    };

    return schema;
}

static GraphResolverConfig buildMinimalResolverConfig(
    int seq_len,
    const std::vector<ExtensionBufferSpec> &extensions = {})
{
    GraphResolverConfig config;
    config.seq_len = seq_len;
    config.batch_size = 1;
    config.d_model = 32;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.head_dim = 8;
    config.d_ff = 64;
    config.vocab_size = 128;
    config.n_layers = 2;
    config.local_n_heads = 4;
    config.local_n_kv_heads = 2;
    config.local_d_ff = 64;
    config.local_vocab = 128;

    // Map extension buffer names to BufferIds
    for (const auto &ext : extensions)
    {
        config.buffer_name_to_id[ext.name] = ext.id;
    }

    return config;
}

static std::shared_ptr<MockGraphBuilder> createMockGraphBuilder(
    const std::vector<ExtensionBufferSpec> &extensions = {})
{
    GraphConfig gc;
    gc.d_model = 32;
    gc.n_heads = 4;
    gc.n_kv_heads = 2;
    gc.head_dim = 8;
    gc.d_ff = 64;
    gc.vocab_size = 128;
    gc.n_layers = 2;
    gc.use_graph_buffer_management = true;
    gc.max_seq_len = 64;
    gc.default_device = DeviceId::cpu();

    auto mock = std::make_shared<MockGraphBuilder>();
    mock->setConfig(gc);
    mock->setSchema(buildMinimalSchema(extensions));
    mock->setResolverConfigFactory(
        [extensions](int seq_len)
        {
            return buildMinimalResolverConfig(seq_len, extensions);
        });

    return mock;
}

class Test__DynamicBufferExtensions_Orchestrator : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
        tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> tensor_factory_;
};

TEST_F(Test__DynamicBufferExtensions_Orchestrator, ToModelBuffers_EmptyExtensions)
{
    // Orchestrator with no extension buffers
    auto mock = createMockGraphBuilder();
    DeviceGraphOrchestrator orchestrator(mock, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        1, 64, DeviceId::cpu()));

    const auto &state = orchestrator.inferenceState();
    ModelBuffers mb = state.toModelBuffers();

    // Core buffers should be set
    EXPECT_NE(mb.current_hidden, nullptr);
    EXPECT_NE(mb.logits, nullptr);
    EXPECT_NE(mb.layer_buffers.Q, nullptr);

    // Extension map should be empty
    EXPECT_TRUE(mb.layer_buffers.extensions.empty())
        << "No extension buffers in schema; extensions should be empty";
}

TEST_F(Test__DynamicBufferExtensions_Orchestrator, ToModelBuffers_ExtensionsPropagate)
{
    // Build a standalone InferenceState and verify toModelBuffers() propagation
    InferenceState state;
    state.hidden = std::shared_ptr<TensorBase>(tensor_factory_->createFP32({64, 32}, DeviceId::cpu()));
    state.logits = std::shared_ptr<TensorBase>(tensor_factory_->createFP32({64, 128}, DeviceId::cpu()));
    state.batch_size = 1;

    std::shared_ptr<TensorBase> fake_gdn_qkv = tensor_factory_->createFP32({64, 3072}, DeviceId::cpu());
    std::shared_ptr<TensorBase> fake_gdn_z = tensor_factory_->createFP32({64, 1024}, DeviceId::cpu());

    state.extension_buffers[BufferId::GDN_QKV] = fake_gdn_qkv;
    state.extension_buffers[BufferId::GDN_Z] = fake_gdn_z;

    ModelBuffers mb = state.toModelBuffers();

    EXPECT_EQ(mb.layer_buffers.extensions.size(), 2u);
    EXPECT_EQ(mb.layer_buffers.get(BufferId::GDN_QKV), fake_gdn_qkv.get());
    EXPECT_EQ(mb.layer_buffers.get(BufferId::GDN_Z), fake_gdn_z.get());
    // Non-existent extensions still return nullptr via get()
    EXPECT_EQ(mb.layer_buffers.get(BufferId::GDN_ALPHA), nullptr);
}

TEST_F(Test__DynamicBufferExtensions_Orchestrator, ToModelBuffers_NullExtensionSkipped)
{
    InferenceState state;
    state.hidden = std::shared_ptr<TensorBase>(tensor_factory_->createFP32({64, 32}, DeviceId::cpu()));
    state.logits = std::shared_ptr<TensorBase>(tensor_factory_->createFP32({64, 128}, DeviceId::cpu()));
    state.batch_size = 1;

    std::shared_ptr<TensorBase> real_tensor = tensor_factory_->createFP32({64, 1024}, DeviceId::cpu());
    state.extension_buffers[BufferId::GDN_QKV] = real_tensor;
    state.extension_buffers[BufferId::GDN_Z] = nullptr;

    ModelBuffers mb = state.toModelBuffers();

    EXPECT_EQ(mb.layer_buffers.extensions.count(BufferId::GDN_QKV), 1u);
    EXPECT_EQ(mb.layer_buffers.get(BufferId::GDN_QKV), real_tensor.get());
    // The nullptr one is skipped by the "if (tensor)" guard
    EXPECT_EQ(mb.layer_buffers.extensions.count(BufferId::GDN_Z), 0u);
}

// ============================================================================
// Part 4: End-to-End Auto-Discovery via initializeInferenceStateFromArena()
//
// Uses MockGraphBuilder (IGraphBuilder interface mock) to test the full chain:
//   Schema declares extension buffers
//   → Resolver maps names to BufferIds
//   → BufferArena registers + allocates
//   → initializeInferenceStateFromArena() auto-discovers non-core buffers
//   → InferenceState::extension_buffers populated
//   → toModelBuffers() propagates to ActivationBuffers::extensions
// ============================================================================

TEST_F(Test__DynamicBufferExtensions_Orchestrator, EndToEnd_CoreIdsFilterCorrectly)
{
    // Use a mock model with no extensions — verify all core buffers go
    // into named fields, NOT extension_buffers
    auto mock = createMockGraphBuilder();
    DeviceGraphOrchestrator orchestrator(mock, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        1, 64, DeviceId::cpu()));

    const auto &state = orchestrator.inferenceState();

    // Core buffers should be in named fields
    EXPECT_NE(state.hidden, nullptr);
    EXPECT_NE(state.logits, nullptr);
    EXPECT_NE(state.normalized, nullptr);
    EXPECT_NE(state.Q, nullptr);
    EXPECT_NE(state.K, nullptr);
    EXPECT_NE(state.V, nullptr);
    EXPECT_NE(state.attn_output, nullptr);
    EXPECT_NE(state.attn_proj, nullptr);
    EXPECT_NE(state.gate, nullptr);
    EXPECT_NE(state.up, nullptr);
    EXPECT_NE(state.ffn_output, nullptr);

    // Extension map should be empty (only core buffers in schema)
    EXPECT_TRUE(state.extension_buffers.empty())
        << "Schema has only core BufferIds; extension_buffers should be empty. "
        << "Got " << state.extension_buffers.size() << " extension(s)";
}

TEST_F(Test__DynamicBufferExtensions_Orchestrator,
       EndToEnd_ExtensionsAutoDiscovered)
{
    // Add extension buffers to the schema via MockGraphBuilder and verify
    // they flow through the complete wiring chain.
    std::vector<ExtensionBufferSpec> extensions = {
        {"mock_gdn_qkv", BufferId::GDN_QKV, 64, 3072},
        {"mock_gdn_z", BufferId::GDN_Z, 64, 1024},
        {"mock_moe_router", BufferId::MOE_ROUTER_LOGITS, 64, 256},
    };

    auto mock = createMockGraphBuilder(extensions);
    DeviceGraphOrchestrator orchestrator(mock, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        1, 64, DeviceId::cpu()));

    const auto &state = orchestrator.inferenceState();

    // Verify extension buffers were auto-discovered
    EXPECT_EQ(state.extension_buffers.count(BufferId::GDN_QKV), 1u)
        << "GDN_QKV should be auto-discovered as extension";
    EXPECT_EQ(state.extension_buffers.count(BufferId::GDN_Z), 1u)
        << "GDN_Z should be auto-discovered as extension";
    EXPECT_EQ(state.extension_buffers.count(BufferId::MOE_ROUTER_LOGITS), 1u)
        << "MOE_ROUTER_LOGITS should be auto-discovered as extension";

    // Verify they're real tensors (not nullptr)
    EXPECT_NE(state.extension_buffers.at(BufferId::GDN_QKV), nullptr);
    EXPECT_NE(state.extension_buffers.at(BufferId::GDN_Z), nullptr);
    EXPECT_NE(state.extension_buffers.at(BufferId::MOE_ROUTER_LOGITS), nullptr);

    // Verify full propagation through toModelBuffers()
    ModelBuffers mb = state.toModelBuffers();

    EXPECT_EQ(mb.layer_buffers.extensions.size(), 3u);
    EXPECT_NE(mb.layer_buffers.get(BufferId::GDN_QKV), nullptr);
    EXPECT_NE(mb.layer_buffers.get(BufferId::GDN_Z), nullptr);
    EXPECT_NE(mb.layer_buffers.get(BufferId::MOE_ROUTER_LOGITS), nullptr);

    // Core buffers should NOT be in extensions
    EXPECT_EQ(mb.layer_buffers.extensions.count(BufferId::HIDDEN_STATE), 0u);
    EXPECT_EQ(mb.layer_buffers.extensions.count(BufferId::Q_PROJ), 0u);
    EXPECT_EQ(mb.layer_buffers.extensions.count(BufferId::LOGITS), 0u);
}

TEST_F(Test__DynamicBufferExtensions_Orchestrator,
       EndToEnd_ExtensionsPreserveIdentity)
{
    // Verify pointer identity: extension_buffers shared_ptr → toModelBuffers() raw ptr
    std::vector<ExtensionBufferSpec> extensions = {
        {"mock_gdn_qkv", BufferId::GDN_QKV, 64, 2048},
    };

    auto mock = createMockGraphBuilder(extensions);
    DeviceGraphOrchestrator orchestrator(mock, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        1, 64, DeviceId::cpu()));

    const auto &state = orchestrator.inferenceState();

    // Get the tensor from extension_buffers
    auto it = state.extension_buffers.find(BufferId::GDN_QKV);
    ASSERT_NE(it, state.extension_buffers.end());
    auto ext_tensor = it->second;
    ASSERT_NE(ext_tensor, nullptr);

    // Verify through toModelBuffers — same pointer
    ModelBuffers mb = state.toModelBuffers();
    EXPECT_EQ(mb.layer_buffers.get(BufferId::GDN_QKV), ext_tensor.get())
        << "ModelBuffers extension must point to the same tensor object";
}

// ============================================================================
// Part 5: Mock Model End-to-End Test
//
// Creates a mock model with diverse fictitious buffer names (GDN, MoE)
// and proves they flow through the complete wiring chain using only
// the IGraphBuilder interface — no real model, no friend access.
// ============================================================================

TEST_F(Test__DynamicBufferExtensions_Orchestrator,
       MockModel_FictitiousBufferIdsEndToEnd)
{
    // Register a diverse set of extension buffers representing
    // what various future models might need
    const std::vector<ExtensionBufferSpec> mock_extensions = {
        // GDN buffers (Qwen 3.5-like)
        {"mock_gdn_qkv", BufferId::GDN_QKV, 64, 3072},
        {"mock_gdn_z", BufferId::GDN_Z, 64, 1024},
        {"mock_gdn_alpha", BufferId::GDN_ALPHA, 64, 16},
        {"mock_gdn_beta", BufferId::GDN_BETA, 64, 16},
        {"mock_attn_output_gate", BufferId::ATTN_OUTPUT_GATE, 64, 32},
        {"mock_gdn_conv_state", BufferId::GDN_CONV_STATE, 16, 192},
        {"mock_gdn_recurrence_in", BufferId::GDN_RECURRENCE_IN, 64, 32},
        {"mock_gdn_recurrence_out", BufferId::GDN_RECURRENCE_OUT, 64, 32},
        // MoE buffers (DeepSeek-like)
        {"mock_moe_router_logits", BufferId::MOE_ROUTER_LOGITS, 64, 256},
        {"mock_moe_expert_indices", BufferId::MOE_EXPERT_INDICES, 64, 8},
        {"mock_moe_expert_weights", BufferId::MOE_EXPERT_WEIGHTS, 64, 8},
        {"mock_moe_expert_output", BufferId::MOE_EXPERT_OUTPUT, 64, 4096},
        {"mock_moe_combined_output", BufferId::MOE_COMBINED_OUTPUT, 64, 32},
        {"mock_moe_shared_expert_output", BufferId::MOE_SHARED_EXPERT_OUTPUT, 64, 32},
    };

    auto mock = createMockGraphBuilder(mock_extensions);
    DeviceGraphOrchestrator orchestrator(mock, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        1, 64, DeviceId::cpu()));

    const auto &state = orchestrator.inferenceState();

    // Verify all mock extensions were discovered
    for (const auto &ext : mock_extensions)
    {
        EXPECT_EQ(state.extension_buffers.count(ext.id), 1u)
            << bufferIdName(ext.id) << " should be in extension_buffers";
        EXPECT_NE(state.extension_buffers.at(ext.id), nullptr)
            << bufferIdName(ext.id) << " should not be null";
    }

    // Verify NONE of the core buffers leaked into extensions
    const std::vector<BufferId> core_ids = {
        BufferId::HIDDEN_STATE,
        BufferId::LOGITS,
        BufferId::LOGITS_LOCAL,
        BufferId::NORMALIZED,
        BufferId::RESIDUAL,
        BufferId::Q_PROJ,
        BufferId::K_PROJ,
        BufferId::V_PROJ,
        BufferId::ATTN_OUTPUT,
        BufferId::ATTN_PROJ,
        BufferId::GATE_PROJ,
        BufferId::UP_PROJ,
        BufferId::FFN_OUTPUT,
        BufferId::ATTN_SCORES_WORKSPACE,
        BufferId::ATTN_CONTEXT_WORKSPACE,
        BufferId::GEMM_WORKSPACE,
        BufferId::Q_ROPE,
        BufferId::K_ROPE,
        BufferId::V_DEQUANT,
    };
    for (BufferId core : core_ids)
    {
        EXPECT_EQ(state.extension_buffers.count(core), 0u)
            << bufferIdName(core) << " is a core buffer and must NOT appear in extension_buffers";
    }

    // Verify full propagation to ModelBuffers
    ModelBuffers mb = state.toModelBuffers();

    EXPECT_EQ(mb.layer_buffers.extensions.size(), mock_extensions.size())
        << "All " << mock_extensions.size() << " extensions should propagate to ModelBuffers";

    for (const auto &ext : mock_extensions)
    {
        auto *tensor = mb.layer_buffers.get(ext.id);
        EXPECT_NE(tensor, nullptr)
            << bufferIdName(ext.id) << " should be accessible via get() in ActivationBuffers";
    }

    // Verify core buffers accessible via named fields (not extensions)
    EXPECT_NE(mb.current_hidden, nullptr);
    EXPECT_NE(mb.logits, nullptr);
    EXPECT_NE(mb.layer_buffers.Q, nullptr);
    EXPECT_NE(mb.layer_buffers.gate, nullptr);
}

TEST_F(Test__DynamicBufferExtensions_Orchestrator,
       MockModel_ExtensionCountIsExact)
{
    // Verify the exact count: only extension buffers appear in extension_buffers
    std::vector<ExtensionBufferSpec> extensions = {
        {"mock_gdn_qkv", BufferId::GDN_QKV, 64, 2048},
        {"mock_gdn_z", BufferId::GDN_Z, 64, 1024},
        {"mock_moe_router", BufferId::MOE_ROUTER_LOGITS, 64, 256},
    };

    auto mock = createMockGraphBuilder(extensions);
    DeviceGraphOrchestrator orchestrator(mock, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        1, 64, DeviceId::cpu()));

    const auto &state = orchestrator.inferenceState();
    EXPECT_EQ(state.extension_buffers.size(), 3u)
        << "Should have exactly 3 extension buffers (the ones we added)";
}

// ============================================================================
// Part 6: core_ids Sync Validation
//
// Validates that the hardcoded core_ids set in
// initializeInferenceStateFromArena() stays in sync with the named
// InferenceState fields. If a core BufferId is missing from core_ids,
// it would leak into extension_buffers.
// ============================================================================

TEST_F(Test__DynamicBufferExtensions_Orchestrator,
       CoreIdsSync_NoCoreBufferLeaksToExtensions)
{
    // Register ALL possible core BufferIds in the schema (including conditional ones)
    // and verify that NONE of them appear in extension_buffers.
    // This test catches core_ids sync failures — if a new core buffer is added
    // to the arena pull block but not to core_ids, it will leak here.

    // Build a schema that has ALL core buffers including conditional ones
    GraphSchema schema;
    schema.name = "test_core_ids_sync";

    schema.layer_buffers = {
        BufferSpec("normalized", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("residual", {"64", "32"}, "fp32", BufferSemantic::InOut),
        BufferSpec("Q", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("K", {"64", "16"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("V", {"64", "16"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("attn_output", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("attn_proj", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("gate", {"64", "64"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("up", {"64", "64"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("ffn_output", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("workspace_scores", {"64", "64"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("workspace_context", {"64", "8"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("workspace_mask", {"64", "64"}, "fp32", BufferSemantic::Scratch),
        // Conditional buffers (normally only present in Hybrid/HybridQ16 mode)
        BufferSpec("Q_rope", {"64", "32"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("K_rope", {"64", "16"}, "fp32", BufferSemantic::Scratch),
        BufferSpec("V_dequant", {"64", "16"}, "fp32", BufferSemantic::Scratch),
    };

    schema.model_buffers = {
        BufferSpec("hidden", {"64", "32"}, "fp32", BufferSemantic::InOut),
        BufferSpec("logits", {"64", "128"}, "fp32", BufferSemantic::Output),
        BufferSpec("logits_local", {"64", "128"}, "fp32", BufferSemantic::Scratch),
    };

    GraphConfig gc;
    gc.d_model = 32;
    gc.n_heads = 4;
    gc.n_kv_heads = 2;
    gc.head_dim = 8;
    gc.d_ff = 64;
    gc.vocab_size = 128;
    gc.n_layers = 2;
    gc.use_graph_buffer_management = true;
    gc.max_seq_len = 64;
    gc.default_device = DeviceId::cpu();

    auto mock = std::make_shared<MockGraphBuilder>();
    mock->setConfig(gc);
    mock->setSchema(std::move(schema));
    mock->setResolverConfigFactory(
        [](int seq_len)
        {
            return buildMinimalResolverConfig(seq_len);
        });

    DeviceGraphOrchestrator orchestrator(mock, mpi_ctx_);
    orchestrator.setTensorFactory(tensor_factory_.get());

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        1, 64, DeviceId::cpu()));

    const auto &state = orchestrator.inferenceState();

    // ALL buffers in the schema are core buffers — extension_buffers must be empty
    EXPECT_TRUE(state.extension_buffers.empty())
        << "All schema buffers are core; extension_buffers should be empty. "
        << "Got " << state.extension_buffers.size() << " extension(s). "
        << "This likely means a new core BufferId was added to "
        << "initializeInferenceStateFromArena() but NOT to core_ids.";

    if (!state.extension_buffers.empty())
    {
        for (const auto &[id, tensor] : state.extension_buffers)
        {
            ADD_FAILURE() << "Core BufferId " << bufferIdName(id)
                          << " leaked into extension_buffers — "
                          << "add it to the core_ids set in "
                          << "initializeInferenceStateFromArena()";
        }
    }
}
