/**
 * @file Test__StageRunnerFactory.cpp
 * @brief Unit tests for StageRunnerFactory (Phase 3 of Multi-Domain Pipeline Plan)
 *
 * Tests that StageRunnerFactory builds the correct runner type for each stage
 * variant: single-device, local TP, and global TP (via injected context).
 *
 * Acceptance criteria (Phase 3):
 * - Single-device stage: non-null runner, correct stage_id/domain, pp_stage_config,
 *   no local/global tp ctx.
 * - Local TP stage:      non-null runner, correct local_tp_ctx, no global_tp_ctx.
 * - Global TP stage:     non-null runner using injected GlobalTPContext,
 *                        global_tp_ctx != nullptr, correct tp_rank_in_domain.
 * - IDLE action throws std::invalid_argument.
 * - Null model_ctx throws std::runtime_error.
 * - Global TP with null domain_registry throws std::runtime_error.
 * - Global TP with no context in registry throws std::runtime_error.
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "execution/global/StageRunnerFactory.h"
#include "execution/global/DomainCommunicatorRegistry.h"
#include "collective/GlobalTPContext.h"
#include "tensors/TensorClasses.h"
#include "mocks/MockModelContext.h"

namespace llaminar2::test
{

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * Create a MockModelContext with the minimal weights required by
     * QwenStandardGraphConfigBuilder::buildWeights():
    *   - token_embd.weight  [vocab_size x d_model]
     *   - output_norm.weight [d_model]
    *   (output.weight falls back to token_embd.weight: tied embeddings)
     */
    static std::shared_ptr<MockModelContext> makeMinimalModelCtx()
    {
        auto ctx = MockModelContext::createMinimal();
        // Minimal preset: vocab=1000, d_model=128
        auto embedding = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1000, 128}, DeviceId::cpu());
        auto norm = std::make_shared<FP32Tensor>(
            std::vector<size_t>{128}, DeviceId::cpu());
        ctx->mockWeightManager().addWeight("token_embd.weight", embedding);
        ctx->mockWeightManager().addWeight("output_norm.weight", norm);
        return ctx;
    }

    /**
     * Build a minimal StageBuildContext using makeMinimalModelCtx().
     */
    static StageBuildContext makeCtx(DomainCommunicatorRegistry *registry = nullptr)
    {
        StageBuildContext ctx;
        ctx.model_ctx = makeMinimalModelCtx();
        ctx.mpi_ctx = nullptr;
        ctx.runner_config.max_seq_len = 128;
        ctx.runner_config.batch_size = 1;
        ctx.domain_registry = registry;
        return ctx;
    }

    /**
     * Build a single-device stage spec/action for layer range [first, last] (inclusive).
     */
    static std::pair<GlobalPPStageSpec, RankStageAction> makeSingleDeviceStage(
        int stage_id,
        int first_layer,
        int last_layer,
        bool has_embedding,
        bool has_lm_head)
    {
        GlobalPPStageSpec spec;
        spec.stage_id = stage_id;
        spec.domain_name = "stage_" + std::to_string(stage_id);
        spec.first_layer = first_layer;
        spec.last_layer = last_layer;
        spec.has_embedding = has_embedding;
        spec.has_lm_head = has_lm_head;
        spec.is_global_tp = false;
        spec.owning_rank = 0;
        spec.inner_mode = InnerParallelism::SINGLE_DEVICE;
        spec.devices = {GlobalDeviceAddress::cpu()};

        RankStageAction action;
        action.stage_id = stage_id;
        action.domain_name = spec.domain_name;
        action.role = RankStageAction::Role::EXECUTE;
        action.first_layer = first_layer;
        action.last_layer = last_layer;
        action.has_embedding = has_embedding;
        action.has_lm_head = has_lm_head;
        action.inner_mode = InnerParallelism::SINGLE_DEVICE;
        action.devices = {GlobalDeviceAddress::cpu()};
        action.is_global_tp = false;

        return {spec, action};
    }

    /**
     * Build a local TP stage spec/action with 2 CPU devices.
     */
    static std::pair<GlobalPPStageSpec, RankStageAction> makeLocalTPStage(
        int stage_id,
        int first_layer,
        int last_layer)
    {
        GlobalPPStageSpec spec;
        spec.stage_id = stage_id;
        spec.domain_name = "local_tp_" + std::to_string(stage_id);
        spec.first_layer = first_layer;
        spec.last_layer = last_layer;
        spec.has_embedding = (first_layer == 0);
        spec.has_lm_head = false;
        spec.is_global_tp = false;
        spec.owning_rank = 0;
        spec.inner_mode = InnerParallelism::LOCAL_TP;
        spec.devices = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};

        RankStageAction action;
        action.stage_id = stage_id;
        action.domain_name = spec.domain_name;
        action.role = RankStageAction::Role::EXECUTE;
        action.first_layer = first_layer;
        action.last_layer = last_layer;
        action.has_embedding = spec.has_embedding;
        action.has_lm_head = false;
        action.inner_mode = InnerParallelism::LOCAL_TP;
        action.devices = spec.devices;
        action.is_global_tp = false;

        return {spec, action};
    }

    /**
     * Build a global TP stage spec/action for a single rank (simulated via MPI_COMM_SELF).
     */
    static std::pair<GlobalPPStageSpec, RankStageAction> makeGlobalTPStage(
        int stage_id,
        int first_layer,
        int last_layer,
        bool has_lm_head,
        int tp_rank_in_domain = 0,
        int tp_domain_size = 1)
    {
        GlobalPPStageSpec spec;
        spec.stage_id = stage_id;
        spec.domain_name = "global_tp_" + std::to_string(stage_id);
        spec.first_layer = first_layer;
        spec.last_layer = last_layer;
        spec.has_embedding = false;
        spec.has_lm_head = has_lm_head;
        spec.is_global_tp = true;
        spec.per_rank_device = GlobalDeviceAddress::cpu();
        spec.participating_ranks = {0};

        RankStageAction action;
        action.stage_id = stage_id;
        action.domain_name = spec.domain_name;
        action.role = RankStageAction::Role::EXECUTE;
        action.first_layer = first_layer;
        action.last_layer = last_layer;
        action.has_embedding = false;
        action.has_lm_head = has_lm_head;
        action.inner_mode = InnerParallelism::SINGLE_DEVICE;
        action.is_global_tp = true;
        action.tp_rank_in_domain = tp_rank_in_domain;
        action.tp_domain_size = tp_domain_size;
        action.device = GlobalDeviceAddress::cpu();

        return {spec, action};
    }

    // =========================================================================
    // Fixture
    // =========================================================================

    class Test__StageRunnerFactory : public ::testing::Test
    {
    };

    // =========================================================================
    // Single-device stage
    // =========================================================================

    TEST(Test__StageRunnerFactory, SingleDevice_HeadStage)
    {
        auto [spec, action] = makeSingleDeviceStage(0, 0, 0, /*emb=*/true, /*lm=*/false);
        auto ctx = makeCtx();

        StageRunnerEntry entry = StageRunnerFactory::create(spec, action, ctx);

        EXPECT_NE(entry.runner, nullptr);
        EXPECT_EQ(entry.stage_id, 0);
        EXPECT_EQ(entry.domain_name, "stage_0");
        ASSERT_TRUE(entry.pp_stage_config.has_value());
        // last_layer in FactoryPPStageConfig is exclusive: 0+1 = 1
        EXPECT_EQ(entry.pp_stage_config->first_layer, 0);
        EXPECT_EQ(entry.pp_stage_config->last_layer, 1);
        EXPECT_TRUE(entry.pp_stage_config->has_embedding);
        EXPECT_FALSE(entry.pp_stage_config->has_lm_head);
        EXPECT_EQ(entry.local_tp_ctx, nullptr);
        EXPECT_EQ(entry.global_tp_ctx, nullptr);
        ASSERT_NE(entry.weight_context, nullptr);
        EXPECT_NE(entry.weight_context->prepared_store, nullptr);
        EXPECT_EQ(entry.weight_context->stage_id, entry.stage_id);
    }

    TEST(Test__StageRunnerFactory, SingleDevice_TailStage)
    {
        auto [spec, action] = makeSingleDeviceStage(1, 1, 0, /*emb=*/false, /*lm=*/true);
        // Override layer range to something valid: layer 1..0 would be invalid.
        // Use layer 0 only (single layer for minimal model).
        action.first_layer = 0;
        action.last_layer = 0;
        action.has_embedding = false;
        action.has_lm_head = true;
        spec.first_layer = 0;
        spec.last_layer = 0;
        spec.has_embedding = false;
        spec.has_lm_head = true;
        spec.stage_id = 1;
        action.stage_id = 1;

        auto ctx = makeCtx();

        StageRunnerEntry entry = StageRunnerFactory::create(spec, action, ctx);

        EXPECT_NE(entry.runner, nullptr);
        EXPECT_EQ(entry.stage_id, 1);
        ASSERT_TRUE(entry.pp_stage_config.has_value());
        EXPECT_FALSE(entry.pp_stage_config->has_embedding);
        EXPECT_TRUE(entry.pp_stage_config->has_lm_head);
        EXPECT_EQ(entry.local_tp_ctx, nullptr);
        EXPECT_EQ(entry.global_tp_ctx, nullptr);
    }

    TEST(Test__StageRunnerFactory, SingleDevice_PPStageConfig_ExclusiveLastLayer)
    {
        // Verify inclusive-to-exclusive conversion: action.last_layer=4 -> pp_cfg.last_layer=5
        auto [spec, action] = makeSingleDeviceStage(0, 2, 4, /*emb=*/false, /*lm=*/false);
        auto ctx = makeCtx();

        StageRunnerEntry entry = StageRunnerFactory::create(spec, action, ctx);

        EXPECT_NE(entry.runner, nullptr);
        ASSERT_TRUE(entry.pp_stage_config.has_value());
        EXPECT_EQ(entry.pp_stage_config->first_layer, 2);
        EXPECT_EQ(entry.pp_stage_config->last_layer, 5); // exclusive
    }

    // =========================================================================
    // Local TP stage
    // =========================================================================

    TEST(Test__StageRunnerFactory, LocalTP_TwoDevices)
    {
        auto [spec, action] = makeLocalTPStage(0, 0, 0);
        spec.backend = CollectiveBackendType::HOST;
        action.backend = CollectiveBackendType::HOST;
        auto ctx = makeCtx();

        StageRunnerEntry entry = StageRunnerFactory::create(spec, action, ctx);

        EXPECT_NE(entry.runner, nullptr);
        EXPECT_EQ(entry.stage_id, 0);
        EXPECT_EQ(entry.domain_name, "local_tp_0");
        ASSERT_TRUE(entry.pp_stage_config.has_value());
        EXPECT_EQ(entry.pp_stage_config->first_layer, 0);
        EXPECT_EQ(entry.pp_stage_config->last_layer, 1);
        // local_tp_ctx is non-null (lifetime holder)
        EXPECT_NE(entry.local_tp_ctx, nullptr);
        EXPECT_EQ(entry.local_tp_ctx->backend(), CollectiveBackendType::HOST);
        EXPECT_EQ(entry.global_tp_ctx, nullptr);
        ASSERT_NE(entry.weight_context, nullptr);
        EXPECT_NE(entry.weight_context->prepared_store, nullptr);
    }

    TEST(Test__StageRunnerFactory, LocalTP_DomainName)
    {
        auto [spec, action] = makeLocalTPStage(2, 0, 0);
        spec.domain_name = "rocm_socket0";
        action.domain_name = "rocm_socket0";
        auto ctx = makeCtx();

        StageRunnerEntry entry = StageRunnerFactory::create(spec, action, ctx);

        EXPECT_EQ(entry.domain_name, "rocm_socket0");
        EXPECT_NE(entry.runner, nullptr);
    }

    // =========================================================================
    // Global TP stage (injected via DomainCommunicatorRegistry test helper)
    // =========================================================================

    TEST(Test__StageRunnerFactory, GlobalTP_InjectedContext)
    {
        auto [spec, action] = makeGlobalTPStage(1, 0, 0, /*has_lm_head=*/true,
                                                 /*tp_rank_in_domain=*/0, /*tp_domain_size=*/1);

        // Create a 1-rank GlobalTPContext using MPI_COMM_SELF (available in MPI_PROCS 1)
        auto global_ctx = GlobalTPContext::createForTest(
            MPI_COMM_SELF, /*domain_id=*/1, /*world_ranks=*/{0});
        ASSERT_NE(global_ctx, nullptr);

        DomainCommunicatorRegistry registry;
        registry.addContextForTest(1, std::move(global_ctx));

        auto ctx = makeCtx(&registry);

        StageRunnerEntry entry = StageRunnerFactory::create(spec, action, ctx);

        EXPECT_NE(entry.runner, nullptr);
        EXPECT_EQ(entry.stage_id, 1);
        EXPECT_NE(entry.global_tp_ctx, nullptr);
        EXPECT_EQ(entry.global_tp_ctx->domainId(), 1);
        EXPECT_EQ(entry.local_tp_ctx, nullptr);
        ASSERT_TRUE(entry.pp_stage_config.has_value());
        EXPECT_EQ(entry.pp_stage_config->first_layer, 0);
        EXPECT_EQ(entry.pp_stage_config->last_layer, 1);
        EXPECT_TRUE(entry.pp_stage_config->has_lm_head);
        ASSERT_NE(entry.weight_context, nullptr);
        EXPECT_NE(entry.weight_context->prepared_store, nullptr);
    }

    TEST(Test__StageRunnerFactory, GlobalTP_UsesContextIndexInsteadOfActionIndex)
    {
        auto [spec, action] = makeGlobalTPStage(1, 0, 0, /*has_lm_head=*/true,
                                                 /*tp_rank_in_domain=*/999,
                                                 /*tp_domain_size=*/999);

        auto global_ctx = GlobalTPContext::createForTest(
            MPI_COMM_SELF, /*domain_id=*/1, /*world_ranks=*/{0});
        ASSERT_NE(global_ctx, nullptr);

        DomainCommunicatorRegistry registry;
        registry.addContextForTest(1, std::move(global_ctx));

        auto ctx = makeCtx(&registry);

        StageRunnerEntry entry = StageRunnerFactory::create(spec, action, ctx);

        EXPECT_NE(entry.runner, nullptr);
        ASSERT_NE(entry.global_tp_ctx, nullptr);
        EXPECT_EQ(entry.action.tp_rank_in_domain, entry.global_tp_ctx->myIndex());
        EXPECT_EQ(entry.action.tp_domain_size, entry.global_tp_ctx->degree());
    }

    TEST(Test__StageRunnerFactory, GlobalTP_MultipleLayers)
    {
        auto [spec, action] = makeGlobalTPStage(2, 10, 19, /*has_lm_head=*/false,
                                                 /*tp_rank_in_domain=*/0, /*tp_domain_size=*/1);

        auto global_ctx = GlobalTPContext::createForTest(
            MPI_COMM_SELF, /*domain_id=*/2, /*world_ranks=*/{0});
        ASSERT_NE(global_ctx, nullptr);

        DomainCommunicatorRegistry registry;
        registry.addContextForTest(2, std::move(global_ctx));

        auto ctx = makeCtx(&registry);

        StageRunnerEntry entry = StageRunnerFactory::create(spec, action, ctx);

        EXPECT_NE(entry.runner, nullptr);
        EXPECT_EQ(entry.stage_id, 2);
        ASSERT_TRUE(entry.pp_stage_config.has_value());
        EXPECT_EQ(entry.pp_stage_config->first_layer, 10);
        EXPECT_EQ(entry.pp_stage_config->last_layer, 20); // exclusive
        EXPECT_NE(entry.global_tp_ctx, nullptr);
    }

    // =========================================================================
    // Error cases
    // =========================================================================

    TEST(Test__StageRunnerFactory, Error_IdleAction)
    {
        auto [spec, action] = makeSingleDeviceStage(0, 0, 0, true, false);
        action.role = RankStageAction::Role::IDLE;
        auto ctx = makeCtx();

        EXPECT_THROW(StageRunnerFactory::create(spec, action, ctx),
                     std::invalid_argument);
    }

    TEST(Test__StageRunnerFactory, Error_NullModelCtx)
    {
        auto [spec, action] = makeSingleDeviceStage(0, 0, 0, true, false);
        StageBuildContext ctx;
        ctx.model_ctx = nullptr;

        EXPECT_THROW(StageRunnerFactory::create(spec, action, ctx),
                     std::runtime_error);
    }

    TEST(Test__StageRunnerFactory, Error_GlobalTP_NullRegistry)
    {
        auto [spec, action] = makeGlobalTPStage(1, 0, 0, false, 0, 1);
        auto ctx = makeCtx(nullptr); // no registry

        EXPECT_THROW(StageRunnerFactory::create(spec, action, ctx),
                     std::runtime_error);
    }

    TEST(Test__StageRunnerFactory, Error_GlobalTP_MissingContextInRegistry)
    {
        auto [spec, action] = makeGlobalTPStage(1, 0, 0, false, 0, 1);
        DomainCommunicatorRegistry registry; // empty: no context for stage 1
        auto ctx = makeCtx(&registry);

        EXPECT_THROW(StageRunnerFactory::create(spec, action, ctx),
                     std::runtime_error);
    }

    // =========================================================================
    // Both orientations: two-stage PP rank participation
    // =========================================================================

    /**
     * Orientation A: Rank 0 builds local TP stage (0) then global TP shard stage (1).
     *
     * Stage 0: local TP, 2 CPU devices, layers 0..0
     * Stage 1: global TP, rank 0 is participant (idx=0), layers 0..0
     */
    TEST(Test__StageRunnerFactory, Orientation_A_LocalTP_then_GlobalTP)
    {
        // --- Stage 0: local TP ---
        auto [spec0, action0] = makeLocalTPStage(0, 0, 0);

        auto ctx0 = makeCtx();
        StageRunnerEntry entry0 = StageRunnerFactory::create(spec0, action0, ctx0);

        EXPECT_NE(entry0.runner, nullptr);
        EXPECT_EQ(entry0.stage_id, 0);
        EXPECT_NE(entry0.local_tp_ctx, nullptr);
        EXPECT_EQ(entry0.global_tp_ctx, nullptr);
        EXPECT_EQ(entry0.local_tp_ctx->degree(), 2);
        EXPECT_EQ(entry0.local_tp_ctx->devices().size(), 2u);
        ASSERT_TRUE(entry0.pp_stage_config.has_value());
        EXPECT_EQ(entry0.pp_stage_config->first_layer, 0);
        EXPECT_EQ(entry0.pp_stage_config->last_layer, 1);
        ASSERT_NE(entry0.weight_context, nullptr);
        ASSERT_NE(entry0.weight_context->prepared_store, nullptr);

        // --- Stage 1: global TP shard ---
        auto [spec1, action1] = makeGlobalTPStage(1, 0, 0, /*has_lm_head=*/true, 0, 1);

        auto global_ctx = GlobalTPContext::createForTest(
            MPI_COMM_SELF, /*domain_id=*/1, /*world_ranks=*/{0});
        ASSERT_NE(global_ctx, nullptr);

        DomainCommunicatorRegistry registry;
        registry.addContextForTest(1, std::move(global_ctx));

        auto ctx1 = makeCtx(&registry);
        StageRunnerEntry entry1 = StageRunnerFactory::create(spec1, action1, ctx1);

        EXPECT_NE(entry1.runner, nullptr);
        EXPECT_EQ(entry1.stage_id, 1);
        EXPECT_EQ(entry1.local_tp_ctx, nullptr);
        EXPECT_NE(entry1.global_tp_ctx, nullptr);
        EXPECT_EQ(entry1.global_tp_ctx->domainId(), 1);
        EXPECT_EQ(entry1.global_tp_ctx->degree(), 1);
        ASSERT_TRUE(entry1.pp_stage_config.has_value());
        EXPECT_TRUE(entry1.pp_stage_config->has_lm_head);
        ASSERT_NE(entry1.weight_context, nullptr);
        ASSERT_NE(entry1.weight_context->prepared_store, nullptr);
        EXPECT_NE(entry0.weight_context, entry1.weight_context);
        EXPECT_NE(entry0.weight_context->prepared_store, entry1.weight_context->prepared_store);
    }

    /**
     * Orientation B: Rank 1 builds global TP shard stage (0) then local TP stage (1).
     *
     * Stage 0: global TP, this rank is index 1 in domain
     * Stage 1: local TP, 2 CPU devices
     */
    TEST(Test__StageRunnerFactory, Orientation_B_GlobalTP_then_LocalTP)
    {
        // --- Stage 0: global TP shard (rank 1 perspective) ---
        // Simulated as a 1-rank domain using MPI_COMM_SELF
        auto [spec0, action0] = makeGlobalTPStage(0, 0, 0, false, /*tp_rank=*/0, 1);
        action0.domain_name = "node_local_tp";
        spec0.domain_name = "node_local_tp";

        auto global_ctx = GlobalTPContext::createForTest(
            MPI_COMM_SELF, /*domain_id=*/0, /*world_ranks=*/{1}); // world rank 1
        ASSERT_NE(global_ctx, nullptr);

        DomainCommunicatorRegistry registry;
        registry.addContextForTest(0, std::move(global_ctx));

        auto ctx0 = makeCtx(&registry);
        StageRunnerEntry entry0 = StageRunnerFactory::create(spec0, action0, ctx0);

        EXPECT_NE(entry0.runner, nullptr);
        EXPECT_EQ(entry0.stage_id, 0);
        EXPECT_EQ(entry0.domain_name, "node_local_tp");
        EXPECT_NE(entry0.global_tp_ctx, nullptr);
        EXPECT_EQ(entry0.local_tp_ctx, nullptr);
        EXPECT_EQ(entry0.global_tp_ctx->degree(), 1);
        ASSERT_NE(entry0.weight_context, nullptr);
        ASSERT_NE(entry0.weight_context->prepared_store, nullptr);

        // --- Stage 1: local TP ---
        auto [spec1, action1] = makeLocalTPStage(1, 0, 0);

        auto ctx1 = makeCtx();
        StageRunnerEntry entry1 = StageRunnerFactory::create(spec1, action1, ctx1);

        EXPECT_NE(entry1.runner, nullptr);
        EXPECT_EQ(entry1.stage_id, 1);
        EXPECT_NE(entry1.local_tp_ctx, nullptr);
        EXPECT_EQ(entry1.global_tp_ctx, nullptr);
        EXPECT_EQ(entry1.local_tp_ctx->degree(), 2);
        EXPECT_EQ(entry1.local_tp_ctx->devices().size(), 2u);
        ASSERT_NE(entry1.weight_context, nullptr);
        ASSERT_NE(entry1.weight_context->prepared_store, nullptr);
        EXPECT_NE(entry0.weight_context, entry1.weight_context);
        EXPECT_NE(entry0.weight_context->prepared_store, entry1.weight_context->prepared_store);
    }

} // namespace llaminar2::test
