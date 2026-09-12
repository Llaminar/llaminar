/**
 * @file Test__PreparedOverlaySourceRetirement.cpp
 * @brief Device-free proofs of frozen expert-source retirement and retention.
 *
 * GPU addresses below are registry keys only: no accelerator is initialized.
 * All native codebooks and FP32/FP16/BF16 own real host storage. Adversarial
 * registry and policy cases prove that incomplete or shared consumers cannot
 * make another participant's upload source disappear.
 */
#include <gtest/gtest.h>

#include "loaders/PreparedOverlaySourceRetirement.h"
#include "loaders/WeightMetadataRegistry.h"
#include "loaders/WeightManager.h"
#include "loaders/WeightPlan.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/EmbeddingVerifierFormats.h"

namespace llaminar2::test
{
namespace
{
    /** @brief Non-executable engine owning an independent prepared lifetime. */
    class RetirementEngine final : public ITensorGemm
    {
    public:
        /** @return False: these ownership tests must never dispatch a kernel. */
        bool supports_device(int) const override { return false; }
        /** @brief Reject execution; the object certifies ownership only. */
        bool multiply_tensor(const TensorBase *, TensorBase *, int, int, int,
                             bool, float, float, const TensorBase *,
                             const IMPIContext *, int, DeviceWorkspaceManager *, int) override
        {
            throw std::logic_error("Ownership-only test engine cannot execute");
        }
    };

    /** @brief Build the same prepared-request identities for either GPU vendor. */
    std::shared_ptr<MoEExpertOverlayRuntimePlan> runtime(DeviceId device)
    {
        auto source = std::make_shared<MoERoutedExpertPlacementPlan>();
        source->enabled = true;
        source->topology = RoutedExpertPlacementTopology::TieredOverlay;
        source->continuation_domain = "accelerator";
        source->shared_expert_domain = "accelerator";
        source->residency_policy = RoutedExpertResidencyPolicy::ExplicitMasks;
        RoutedExpertDomain domain;
        domain.name = "accelerator";
        domain.scope = ExecutionDomainScope::SINGLE;
        domain.backend = device.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        domain.participants = {device.is_cuda() ? GlobalDeviceAddress::cuda(0) : GlobalDeviceAddress::rocm(0)};
        domain.owner_rank = 0;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        source->domains = {domain};
        RoutedExpertTier tier;
        tier.name = "priority_0";
        tier.domain = domain.name;
        tier.priority = 0;
        tier.fallback = true;
        source->routed_tiers = {tier};
        source->placements = {RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 0, 0}}};
        return resolveMoEExpertOverlayRuntimePlan(source);
    }

    /** @brief Resolve requests without initializing any backend. */
    MoEExpertOverlayPreparationPlan preparation(DeviceId device)
    {
        return MoEExpertOverlayPreparationPlan::build(*runtime(device));
    }

    /** @brief Reuse canonical fixtures, reshaping their bytes into two expert slots. */
    std::shared_ptr<TensorBase> sourceTensor(const EmbeddingVerifierFormatCase &format)
    {
        const auto flat = format.create({512, 256}, 71);
        const std::vector<size_t> shape{256, 256, 2};
        switch (flat->native_type())
        {
        case TensorType::FP32:
            return TestTensorFactory::createFP32Random(shape, -1.0f, 1.0f, 71);
        case TensorType::FP16:
        case TensorType::BF16:
        {
            const auto *data = static_cast<const uint16_t *>(flat->raw_data());
            const std::vector<uint16_t> bytes(data, data + 512 * 256);
            if (flat->native_type() == TensorType::FP16)
                return std::make_shared<FP16Tensor>(shape, bytes);
            return std::make_shared<BF16Tensor>(shape, bytes);
        }
        default:
        {
            const auto *data = static_cast<const uint8_t *>(flat->raw_data());
            const std::vector<uint8_t> bytes(data, data + flat->size_bytes());
            if (flat->native_type() == TensorType::Q8_1)
                // Q8_1 is a 2-D activation format, not a loadable 3-D GGUF
                // expert parent. Preserve its shape to test rejection below.
                return std::make_shared<Q8_1Tensor>(flat->shape(), bytes);
            MockMPIContext mpi;
            TensorFactory factory(mpi);
            return factory.createQuantized(flat->native_type(), shape, bytes);
        }
        }
    }

    /** @brief Bind an owned source using the plan's complete participant identity. */
    WeightBinding bindingFor(std::shared_ptr<TensorBase> tensor,
                             const MoEExpertOverlayPreparationPlan &plan)
    {
        const auto &request = plan.requests().front();
        WeightBinding binding;
        binding.binding_id = 1;
        binding.identity = makeSourceWeightIdentity("blk.0.ffn_gate_exps.weight", ModelContextId{91}, 1);
        binding.identity.role = WeightRole::MoEExpertGate;
        binding.identity.derivation = WeightDerivationKind::ExpertSlice;
        binding.identity.layer = 0;
        binding.identity.overlay_domain = request.domain_name;
        binding.identity.overlay_participant_index = request.participant_index;
        binding.identity.overlay_participant_world_rank = request.participant_world_rank;
        binding.slice.expert_ids = {0, 2};
        binding.slice.expert_count = 2;
        binding.slice.inner_is_presliced = true;
        binding.residency.home_device = request.device;
        binding.residency.resident_device = request.device;
        binding.residency.host_policy = WeightHostPolicy::RequiredUntilPreparedOrTransferred;
        binding.tensor_owner = std::move(tensor);
        binding.tensor = binding.tensor_owner.get();
        binding.immutable = true;
        return binding;
    }

    /** @brief Install exact lifetime aliases, optionally omitting one expert. */
    void publish(ExpertGemmRegistry &registry, const WeightBinding &binding,
                 const std::vector<int> &experts = {0, 2})
    {
        const auto &id = binding.identity;
        const auto role = id.role == WeightRole::MoEExpertGate ? ExpertGemmRegistry::WeightRole::GATE
                        : id.role == WeightRole::MoEExpertUp ? ExpertGemmRegistry::WeightRole::UP
                                                           : ExpertGemmRegistry::WeightRole::DOWN;
        for (const int expert : experts)
        {
            auto engine = std::make_shared<RetirementEngine>();
            registry.registerEngineForParticipant(id.overlay_domain, binding.residency.home_device,
                id.overlay_participant_world_rank, id.overlay_participant_index, 0, expert,
                role, engine.get(), engine);
            registry.registerEngineForDomain(id.overlay_domain, binding.residency.home_device,
                0, expert, role, engine.get(), engine);
        }
    }

    /** @brief Register the actual source consumer policy rather than a test-side flag. */
    void registerPolicy(WeightMetadataRegistry &metadata, const WeightBinding &binding)
    {
        ASSERT_TRUE(metadata.registerWeight(binding.tensor, binding.identity, binding.slice, binding.residency));
    }
}

TEST(PreparedOverlaySourceRetirement, WeightManagerPreparationRetiresSourcesOutsideLegacyCaches)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        SCOPED_TRACE(device.to_string());
        auto model = MockModelLoader::createMinimal();
        WeightManager manager(*model);
        auto runtime_plan = runtime(device);
        auto plan = MoEExpertOverlayPreparationPlan::build(*runtime_plan);
        std::vector<WeightBinding> bindings;
        for (const auto role : {WeightRole::MoEExpertGate, WeightRole::MoEExpertUp, WeightRole::MoEExpertDown})
        {
            auto binding = bindingFor(sourceTensor(embeddingVerifierFormats().front()), plan);
            binding.binding_id = bindings.size() + 1;
            binding.identity.role = role;
            binding.identity.canonical_name = role == WeightRole::MoEExpertGate ? "blk.0.ffn_gate_exps.weight"
                                              : role == WeightRole::MoEExpertUp ? "blk.0.ffn_up_exps.weight"
                                                                              : "blk.0.ffn_down_exps.weight";
            registerPolicy(*manager.weightMetadataRegistry(), binding);
            publish(manager.expertGemmRegistry(), binding);
            bindings.push_back(std::move(binding));
        }
        FrozenModelWeightSet weights({}, bindings);

        // The complete prepared bank makes this a device-free setup replay.
        // Crucially, none of its frozen source tensors were inserted in cache_.
        ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(*runtime_plan, device, &weights));
        for (const auto &binding : bindings)
            EXPECT_TRUE(binding.tensor->is_raw_data_released()) << binding.identity.canonical_name;
        EXPECT_TRUE(manager.prepareMoEExpertOverlayWeights(*runtime_plan, device, &weights));
    }
}

TEST(PreparedOverlaySourceRetirement, AllFormatsBothGPUIdentitiesReleaseOwnedFrozenSources)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const auto &format : embeddingVerifierFormats())
    {
        SCOPED_TRACE(device.to_string() + "/" + format.label);
        auto plan = preparation(device);
        auto binding = bindingFor(sourceTensor(format), plan);
        WeightMetadataRegistry metadata;
        registerPolicy(metadata, binding);
        ExpertGemmRegistry registry;
        publish(registry, binding);
        FrozenModelWeightSet weights({}, {binding});
        const auto bytes = binding.tensor->size_bytes();
        ASSERT_GT(bytes, 0u);
        if (binding.tensor->shape().size() != 3u)
        {
            EXPECT_THROW(retirePreparedOverlaySources(weights, plan, registry, metadata), std::logic_error);
            EXPECT_FALSE(binding.tensor->is_raw_data_released());
            continue;
        }
        EXPECT_EQ(retirePreparedOverlaySources(weights, plan, registry, metadata), bytes);
        EXPECT_TRUE(binding.tensor->is_raw_data_released());
        EXPECT_EQ(retirePreparedOverlaySources(weights, plan, registry, metadata), 0u);
        EXPECT_NE(registry.getEngineLifetimeForDomain("accelerator", device, 0, 2,
            ExpertGemmRegistry::WeightRole::GATE), nullptr);
    }
}

TEST(PreparedOverlaySourceRetirement, MissingExpertPreservesEverySourceUntilComplete)
{
    auto plan = preparation(DeviceId::rocm(0));
    auto first = bindingFor(sourceTensor(embeddingVerifierFormats().front()), plan);
    auto second = first;
    second.binding_id = 2;
    second.tensor_owner = sourceTensor(embeddingVerifierFormats().front());
    second.tensor = second.tensor_owner.get();
    second.slice.expert_ids = {0, 1};
    WeightMetadataRegistry metadata;
    registerPolicy(metadata, first);
    registerPolicy(metadata, second);
    ExpertGemmRegistry registry;
    publish(registry, first);
    FrozenModelWeightSet weights({}, {first, second});
    EXPECT_THROW(retirePreparedOverlaySources(weights, plan, registry, metadata), std::logic_error);
    EXPECT_FALSE(first.tensor->is_raw_data_released());
    EXPECT_FALSE(second.tensor->is_raw_data_released());
    publish(registry, second, {1});
    EXPECT_GT(retirePreparedOverlaySources(weights, plan, registry, metadata), 0u);
}

TEST(PreparedOverlaySourceRetirement, RetainsCPUAndGraphConsumersAndRejectsTornRegistry)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        auto plan = preparation(device);
        auto binding = bindingFor(sourceTensor(embeddingVerifierFormats().front()), plan);
        WeightMetadataRegistry metadata;
        registerPolicy(metadata, binding);
        ExpertGemmRegistry registry;
        publish(registry, binding);
        FrozenModelWeightSet weights({}, {binding});
        auto other = std::make_shared<RetirementEngine>();
        registry.registerEngineForDomain("accelerator", device, 0, 2,
            ExpertGemmRegistry::WeightRole::GATE, other.get(), other);
        EXPECT_THROW(retirePreparedOverlaySources(weights, plan, registry, metadata), std::logic_error);
        EXPECT_FALSE(binding.tensor->is_raw_data_released());
        publish(registry, binding);
        ASSERT_TRUE(metadata.mergeHostPolicy(binding.tensor, WeightHostPolicy::RequiredForCPUExecution));
        EXPECT_EQ(retirePreparedOverlaySources(weights, plan, registry, metadata), 0u);
        EXPECT_FALSE(binding.tensor->is_raw_data_released());
        binding.residency.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
        registerPolicy(metadata, binding);
        FrozenModelWeightSet graph_owned({}, {binding});
        EXPECT_EQ(retirePreparedOverlaySources(graph_owned, plan, registry, metadata), 0u);
        EXPECT_FALSE(binding.tensor->is_raw_data_released());
    }
}
}
