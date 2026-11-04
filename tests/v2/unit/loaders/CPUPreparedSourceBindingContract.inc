/**
 * @file CPUPreparedSourceBindingContract.inc
 * @brief Real tiny-GGUF context/manager binding without device or socket fabrication.
 *
 * The original source helper supplies genuine GGUF bytes and closed preparation.
 * Context identity and weak-owner lifecycle are production code. The lifetime
 * case reads one actual expert through the admitted common CPU loader and keeps
 * only its genuine prepared engine after context retirement.
 */
#include "loaders/ModelContext.h"
#include "loaders/WeightManager.h"
#include "utils/CPUWeightPlanTestAdmission.h"
#include "utils/CPUProjectionTestWorkspace.h"
#include "backends/BackendManager.h"
namespace
{
    /** @return Genuine metadata/parser/manager ownership over the existing tiny source. */
    std::shared_ptr<llaminar2::ModelContext> coldSourceContext(const StaticJointOriginal &original)
    {
        llaminar2::ModelContextConfig config;
        config.mpi_ctx = original.mpi;
        config.factory = original.factory.get();
        config.use_mmap = false;
        auto result = llaminar2::ModelContext::create(original.file->path.string(), config);
        if (!result)
            throw std::runtime_error("Tiny original prepared context failed to load");
        return result;
    }
    /** @return Relay-only registry retains all original metadata but allocates no invented
     * endpoint. */
    std::unique_ptr<llaminar2::MoEOverlayParticipantResidencyRegistry>
    coldSourceRegistry(const StaticJointOriginal &original,
                       const std::shared_ptr<llaminar2::ModelContext> &context)
    {
        using namespace llaminar2;
        return std::make_unique<MoEOverlayParticipantResidencyRegistry>(
            MoEOverlayParticipantResidencyRegistry::Config{
                .num_layers = 1,
                .num_experts = 512,
                .initial_epoch = 1,
                .projection_preparation = original.preparation,
                .first_model_layer = 7,
                .original_model = ModelContextId{reinterpret_cast<uint64_t>(context.get())}});
    }
} // namespace
/** @test Expected metadata cannot grant source use; only the exact real model/manager association
 * can. */
TEST(CPUPreparedSourceBindingContract, ActualContextBindsExactRegistryAndPreparationOnce)
{
    using namespace llaminar2;
    const auto original = staticJointOriginal();
    auto context = coldSourceContext(original);
    auto banks = coldSourceRegistry(original, context);
    std::string error;
    EXPECT_FALSE(banks->acquireOriginalPreparedSource(&error));
    ASSERT_TRUE(banks->bindOriginalPreparedContext(context, original.preparation, &error)) << error;
    ASSERT_TRUE(banks->bindOriginalPreparedContext(context, original.preparation, &error)) << error;
    auto use = banks->acquireOriginalPreparedSource(&error);
    ASSERT_TRUE(use) << error;
    EXPECT_EQ(&use->registry(), &context->concreteWeightManager()->expertGemmRegistry());
    EXPECT_EQ(use->preparation(), original.preparation);
    ExpertGemmRegistry foreign;
    EXPECT_FALSE(banks->registerInitialLayerFromPreparedRegistry(foreign, 7, &error));
    EXPECT_NE(error.find("original producer"), std::string::npos);
    EXPECT_TRUE(banks->registerInitialLayerFromPreparedRegistry(use->registry(), 7, &error))
        << error;
}
/** @test Foreign or independently owned aliases cannot replace a cold producer or preparation. */
TEST(CPUPreparedSourceBindingContract,
     ForeignModelRegistryPreparationAndSharedOwnersRejectWithoutRebinding)
{
    using namespace llaminar2;
    const auto original = staticJointOriginal(), other = staticJointOriginal();
    auto context = coldSourceContext(original), foreign = coldSourceContext(original);
    auto banks = coldSourceRegistry(original, context);
    std::string error;
    EXPECT_FALSE(banks->bindOriginalPreparedContext(foreign, original.preparation, &error));
    EXPECT_FALSE(banks->bindOriginalPreparedContext(context, other.preparation, &error));
    auto wrongPreparation = std::shared_ptr<const MoEExpertOverlayPreparationPlan>(
        original.preparation.get(), [](auto *) {});
    EXPECT_FALSE(banks->bindOriginalPreparedContext(context, wrongPreparation, &error));
    EXPECT_FALSE(banks->acquireOriginalPreparedSource(&error));
    ASSERT_TRUE(banks->bindOriginalPreparedContext(context, original.preparation, &error)) << error;
    auto wrongContextOwner = std::shared_ptr<ModelContext>(context.get(), [](auto *) {});
    EXPECT_FALSE(
        banks->bindOriginalPreparedContext(wrongContextOwner, original.preparation, &error));
    ASSERT_TRUE(banks->acquireOriginalPreparedSource(&error)) << error;
    // A generic component producer is not silently adopted as a different model.
    auto unbound = coldSourceRegistry(original, context);
    ExpertGemmRegistry component;
    ASSERT_TRUE(unbound->registerInitialLayerFromPreparedRegistry(component, 7, &error)) << error;
    EXPECT_FALSE(unbound->bindOriginalPreparedContext(context, original.preparation, &error));
    EXPECT_FALSE(unbound->acquireOriginalPreparedSource(&error));
}
/** @test Scoped source use is temporary; genuine prepared storage survives without retaining model
 * cycles. */
TEST(CPUPreparedSourceBindingContract, ExpiredOwnerCannotRebindAndActualPreparedEngineSurvives)
{
    using namespace llaminar2;
    if (!hasCPUBackend())
        initCPUBackend(-1);
    const auto original = staticJointOriginal();
    auto context = coldSourceContext(original);
    auto banks = coldSourceRegistry(original, context);
    std::string error;
    ASSERT_TRUE(banks->bindOriginalPreparedContext(context, original.preparation, &error)) << error;
    auto manager = context->concreteWeightManager();
    const auto id = banks->originalModel();
    std::shared_ptr<ITensorGemm> engine;
    {
        InferenceStrategy strategy;
        strategy.model_id = id;
        strategy.devices = {DeviceId::cpu()};
        WeightPlan plan(strategy);
        WeightRequirement requirement;
        requirement.canonical_name = "blk.7.ffn_gate_exps.weight";
        requirement.role = WeightRole::MoEExpertGate;
        requirement.layer = 7;
        requirement.expert = 71;
        requirement.derivation = WeightDerivationKind::ExpertSlice;
        requirement.target_device = DeviceId::cpu();
        requirement.expected_prepared_kind = PreparedWeightKind::CpuPackedGemm;
        requirement.slice = {.source_rows = 8,
                             .source_cols = 20,
                             .row_count = 8,
                             .col_count = 20,
                             .expert_count = 1,
                             .expert_ids = {71},
                             .inner_is_presliced = true};
        plan.add(requirement);
        ASSERT_TRUE(manager->preparedWeightStore()->bindModelIdIfUnset(id));
        test::admitCPUWeightPlansForTest(*manager, context->concreteLoader(), {&plan});
        const auto frozen = manager->materialize(plan);
        ASSERT_EQ(frozen.bindings().size(), 1u);
        const auto &binding = frozen.bindings().front();
        ASSERT_TRUE(binding.prepared);
        engine = manager->preparedWeightStore()->gemmKernelLifetime(*binding.prepared);
        ASSERT_TRUE(engine);
        EXPECT_TRUE(engine->weights_converted());
    }
    auto use = banks->acquireOriginalPreparedSource(&error);
    ASSERT_TRUE(use) << error;
    std::weak_ptr<ModelContext> contextWeak = context;
    std::weak_ptr<WeightManager> managerWeak = manager;
    context.reset();
    manager.reset();
    EXPECT_FALSE(contextWeak.expired());
    use.reset();
    EXPECT_TRUE(contextWeak.expired());
    EXPECT_TRUE(managerWeak.expired());
    EXPECT_FALSE(banks->acquireOriginalPreparedSource(&error));
    EXPECT_NE(error.find("expired"), std::string::npos);
    auto replacement = coldSourceContext(original);
    EXPECT_FALSE(banks->bindOriginalPreparedContext(replacement, original.preparation, &error));
    test::CPUProjectionTestWorkspace workspace(
        1, 8, test::cpuProjectionTestRequirements(1, {engine.get()}));
    FP32Tensor input({1, 8}), output({1, 20});
    std::fill_n(input.mutable_data(), 8, 0.25f);
    ASSERT_TRUE(engine->multiply_tensor(&input, &output, 1, 20, 8, true, 1.f, 0.f, nullptr, nullptr,
                                        -1, workspace.get()));
    for (size_t i = 0; i < 20; ++i)
        EXPECT_EQ(output.data()[i], 142.f); // eight original71-valued weights ×.25
}
