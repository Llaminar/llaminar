/**
 * @file PreparedSourceNonemptyBankContract.inc
 * @brief Authenticate nonempty real CPU banks against their original tiny model source.
 *
 * One genuine local actor prepares an E2 GGUF expert through the common admitted
 * loader. The second owner in the existing fixture remains metadata only. No
 * mocked engine or fabricated GPU/socket preparation certifies source binding.
 */
#include "loaders/ModelContext.h"
#include "loaders/WeightManager.h"
#include "loaders/PreparedWeightStore.h"
#include "utils/PlanningGGUFFixture.h"
#include "utils/CPUWeightPlanTestAdmission.h"
#include "utils/CPUProjectionTestWorkspace.h"
#include "backends/BackendManager.h"
namespace
{
    /** @brief Real source, manager and scoped engine triplet for the original first owner. */
    struct NonemptyPreparedSource
    {
        test::PlanningGGUFFixture file{true,
                                       false,
                                       GGUFTensorType::F32,
                                       32,
                                       std::nullopt,
                                       8,
                                       test::PlanningGGUFFixture::Geometry::FullAttentionGraph,
                                       1,
                                       2};
        std::shared_ptr<ModelContext> context;
        /** @brief Prepare actual G/U/D native FP32 consumers with exact participant keys. */
        explicit NonemptyPreparedSource(float value = 1.f)
        {
            if (!hasCPUBackend())
                initCPUBackend(-1);
            std::vector<float> values(2u * 256u * 32u, value);
            for (const auto role : {"gate", "up", "down"})
                file.writePayload("blk.0.ffn_" + std::string(role) + "_exps.weight", 0,
                                  {reinterpret_cast<const uint8_t *>(values.data()),
                                   values.size() * sizeof(float)});
            ModelContextConfig config;
            config.mpi_ctx = MPIContextFactory::self();
            config.use_mmap = false;
            context = ModelContext::create(file.path(), config);
            if (!context)
                throw std::runtime_error("Nonempty source tiny context failed");
            auto manager = context->concreteWeightManager();
            InferenceStrategy strategy;
            strategy.model_id = {reinterpret_cast<uint64_t>(context.get())};
            strategy.devices = {DeviceId::cpu()};
            WeightPlan plan(strategy, PhysicalMemoryOwner::RoutedExpertWeights);
            for (int role = 0; role < 3; ++role)
            {
                WeightRequirement request;
                request.canonical_name = "blk.0.ffn_" +
                                         std::string(role == 0   ? "gate"
                                                     : role == 1 ? "up"
                                                                 : "down") +
                                         "_exps.weight";
                request.role = role == 0   ? WeightRole::MoEExpertGate
                               : role == 1 ? WeightRole::MoEExpertUp
                                           : WeightRole::MoEExpertDown;
                request.derivation = WeightDerivationKind::ExpertSlice;
                request.layer = 0;
                request.expert = 0;
                request.target_device = DeviceId::cpu();
                request.overlay_domain = "first";
                request.overlay_participant_index = 0;
                request.overlay_participant_world_rank = 0;
                request.residency_category = WeightResidencyCategory::CpuFallbackExpert;
                request.expected_prepared_kind = PreparedWeightKind::CpuPackedGemm;
                const size_t k = role == 2 ? 32u : 256u, n = role == 2 ? 256u : 32u;
                request.slice = {.source_rows = k,
                                 .source_cols = n,
                                 .row_count = k,
                                 .col_count = n,
                                 .expert_count = 1,
                                 .expert_ids = {0},
                                 .inner_is_presliced = true};
                plan.add(request);
            }
            if (!manager->preparedWeightStore()->bindModelIdIfUnset(strategy.model_id))
                throw std::runtime_error("Nonempty source model/store differs");
            test::admitCPUWeightPlansForTest(*manager, context->concreteLoader(), {&plan});
            const auto frozen = manager->materialize(plan);
            if (frozen.bindings().size() != 3)
                throw std::runtime_error("Nonempty source triplet incomplete");
        }
        /** @return Ordinary source bank using the existing two-owner whole placement fixture. */
        std::shared_ptr<MoEOverlayParticipantResidencyRegistry> banks() const
        {
            return std::make_shared<MoEOverlayParticipantResidencyRegistry>(
                MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map = twoParticipantOwnerMap(),
                    .local_participant_ids = {0},
                    .num_layers = 1,
                    .num_experts = 2,
                    .initial_epoch = 1,
                    .original_model = ModelContextId{reinterpret_cast<uint64_t>(context.get())}});
        }
        /** @return Exact actual first-owner triplet from the real original manager registry. */
        std::vector<MoEOverlayPreparedExpertPayload> engines() const
        {
            const auto owners = twoParticipantOwnerMap();
            std::vector<MoEOverlayPreparedExpertPayload> result;
            std::string error;
            if (!resolveMoEOverlayPreparedExpertPayloads(
                    context->concreteWeightManager()->expertGemmRegistry(),
                    *owners.participantForId(0), 0, 2, {true, false}, result, &error))
                throw std::runtime_error(error);
            return result;
        }
    };
} // namespace
/** @test Binding after real nonempty publication rejects foreign data and independent engine
 * owners. */
TEST(PreparedSourceBindingContract, NonemptyOrdinaryBanksAuthenticateActualOriginalEngines)
{
    NonemptyPreparedSource original, foreign(3.f);
    auto expected = original.engines(), other = foreign.engines();
    std::string error;
    auto good = original.banks();
    ASSERT_TRUE(good->registerInitialLayer(0, 0, {true, false}, expected, &error)) << error;
    ASSERT_TRUE(good->bindOriginalPreparedContext(original.context, {}, &error)) << error;
    ASSERT_TRUE(good->registerInitialLayer(0, 0, {true, false}, expected, &error)) << error;
    EXPECT_FALSE(good->registerInitialLayer(0, 0, {true, false}, other, &error));
    auto independent = expected;
    independent[0] = MoEOverlayPreparedExpertPayload(
        std::shared_ptr<ITensorGemm>(expected[0].gate().get(), [](auto *) {}), expected[0].up(),
        expected[0].down());
    EXPECT_FALSE(good->registerInitialLayer(0, 0, {true, false}, independent, &error));
    auto bad = original.banks();
    ASSERT_TRUE(bad->registerInitialLayer(0, 0, {true, false}, other, &error)) << error;
    EXPECT_FALSE(bad->bindOriginalPreparedContext(original.context, {}, &error));
    EXPECT_FALSE(bad->acquireOriginalPreparedSource(&error));
    auto alias = original.banks();
    ASSERT_TRUE(alias->registerInitialLayer(0, 0, {true, false}, independent, &error)) << error;
    EXPECT_FALSE(alias->bindOriginalPreparedContext(original.context, {}, &error));
    EXPECT_NE(error.find("independent engine owner"), std::string::npos);
    auto retained = good->endpoint(0)->acquire(1);
    ASSERT_TRUE(retained);
    EXPECT_TRUE(retained->layerForModelLayer(0).experts[0].sameIdentity(expected[0]));
}
/** @test Weak source binding expires without invalidating a retained real prepared bank. */
TEST(PreparedSourceBindingContract, RetainedNonemptyBankSurvivesOriginalContextExpiry)
{
    NonemptyPreparedSource original;
    auto banks = original.banks();
    std::string error;
    auto expected = original.engines();
    ASSERT_TRUE(banks->registerInitialLayer(0, 0, {true, false}, expected, &error)) << error;
    ASSERT_TRUE(banks->bindOriginalPreparedContext(original.context, {}, &error)) << error;
    auto retained = banks->endpoint(0)->acquire(1);
    ASSERT_TRUE(retained);
    expected.clear();
    auto source = banks->acquireOriginalPreparedSource(&error);
    ASSERT_TRUE(source) << error;
    std::weak_ptr<ModelContext> weak = original.context;
    original.context.reset();
    EXPECT_FALSE(weak.expired());
    source.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_FALSE(banks->acquireOriginalPreparedSource(&error));
    const auto &engine = retained->layerForModelLayer(0).experts[0].gate();
    ASSERT_TRUE(engine);
    test::CPUProjectionTestWorkspace workspace(
        1, 256, test::cpuProjectionTestRequirements(1, {engine.get()}));
    FP32Tensor input({1, 256}), output({1, 32});
    std::fill_n(input.mutable_data(), 256, .25f);
    ASSERT_TRUE(engine->multiply_tensor(&input, &output, 1, 32, 256, true, 1.f, 0.f, nullptr,
                                        nullptr, -1, workspace.get()));
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(output.data()[i], 64.f);
}
/** @test Public migration preparation has no zero-model source escape. */
TEST(PreparedSourceBindingContract, PublicFactoryRejectsUnboundZeroModelBeforeTransferPreparation)
{
    auto fixture = participantMigrationFixture();
    auto provider = std::make_shared<IdentityTransferProvider>();
    MoEOverlayParticipantPreparedWaveFactory factory(
        {.registry = fixture.registry, .transfer_provider = provider});
    const auto result = factory.prepare(fixture.transaction);
    EXPECT_EQ(result.status, MoEOverlayResidencyStageStartStatus::Failed);
    EXPECT_NE(result.error.find("original ModelContext binding"), std::string::npos);
    EXPECT_EQ(provider->prepare_calls, 0);
}
