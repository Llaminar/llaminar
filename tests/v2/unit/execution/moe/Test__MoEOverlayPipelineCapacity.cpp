/**
 * @file Test__MoEOverlayPipelineCapacity.cpp
 * @brief Device-free joint admission and materialization proof for MoE pipelines.
 *
 * Distinct pipeline stages retain compact expert quotas while charging shared
 * CPU and GPU allocators together. These tests expose independently fitting
 * stages that overcommit their shared allocator, prove mandatory later-stage
 * owners precede optional placement, and exercise the resulting single ledger.
 * No model, GPU context, prepared weight or cache payload is materialized.
 */
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace
    {
        constexpr std::size_t available = std::size_t{1} << 40;

        /** @return Stable rank-bound allocator identity without parsing device text. */
        PhysicalMemoryAllocatorIdentity identity(DeviceId device)
        {
            return {.world_rank = 0, .device = device};
        }

        /** @return A diagnostic label; typed identity remains the allocator authority. */
        std::string name(DeviceId device)
        {
            return "rank0/" + device.toString();
        }

        /**
         * @brief Contribute one stage's distinct fixed bytes on a physical resource.
         * @param device Typed process-local allocator.
         * @param capacity Shared allocatable-memory observation.
         * @param prefix_bytes This stage's persistent host-cache backing.
         * @return Certified contribution used by the production quota resolver.
         */
        MoEOverlayPhysicalMemoryBudget budget(
            DeviceId device, std::size_t capacity = available,
            std::size_t prefix_bytes = 0)
        {
            PhysicalMemoryBOMBuilder builder({.world_rank = 0, .device = device,
                .total_bytes = capacity, .admission_available_bytes = capacity});
            builder.add(PhysicalMemoryOwner::PrefixHostTier, prefix_bytes);
            return {name(device), PhysicalMemoryAdmissionCertificate(builder.build())};
        }

        /**
         * @brief Declare a compact stage with one complete expert-coverage tier.
         * @param first First global routed layer owned by this stage.
         * @param count Number of contiguous main/retained-head routed layers.
         * @param devices Exact continuation/routed participants.
         * @param format Source codebook or floating type; no payload is allocated.
         * @param copies Physical projection-copy policy of this stage.
         * @return Production capacity input with unique per-stage fixed contributions.
         */
        MoEOverlayCapacityResolverInput stage(
            int first, int count, std::vector<DeviceId> devices,
            ExpertWeightFormat format = ExpertWeightFormat::floating(TensorType::FP32),
            MoEOverlayTierCopyPolicy copies = MoEOverlayTierCopyPolicy::Apportioned)
        {
            MoEOverlayCapacityResolverInput input;
            input.num_experts = 4;
            for (int layer = first; layer < first + count; ++layer)
                input.layer_weight_manifest.push_back({.layer_idx = layer, .projections = {{
                    {.projection = ExpertTierWeightProjection::Gate, .N = 64, .K = 64, .format = format},
                    {.projection = ExpertTierWeightProjection::Up, .N = 64, .K = 64, .format = format},
                    {.projection = ExpertTierWeightProjection::Down, .N = 64, .K = 64, .format = format},
                }}});
            MoEOverlayTierCapacityRequest tier{
                .tier_index = 0, .tier_name = "experts", .priority = 0, .fallback = true,
                .copy_policy = copies};
            const std::size_t shadows = copies == MoEOverlayTierCopyPolicy::GateUpOwnedDownColumns ? 0u : 1u;
            for (std::size_t participant = 0; participant < devices.size(); ++participant)
            {
                const auto device = devices[participant];
                tier.participants.push_back({.participant_id = static_cast<int>(participant),
                    .resource_id = name(device), .shadow_slots_per_layer = shadows,
                    .maximum_concurrent_shadow_slots = shadows * static_cast<std::size_t>(count)});
                input.physical_budgets.push_back(budget(device));
            }
            // GPU stages share the same host allocator, but each retains its
            // own prefix backing. The aggregate must add these contributions.
            if (std::none_of(devices.begin(), devices.end(), [](auto device) { return device.is_cpu(); }))
                input.physical_budgets.push_back(budget(DeviceId::cpu(), available, 4096u * devices.size()));
            input.tiers.push_back(std::move(tier));
            return input;
        }

        /**
         * @brief Replace one allocator observation without changing stage geometry.
         * @param input Stage that already contributes to the allocator.
         * @param device Allocator whose observation is replaced.
         * @param capacity New shared physical capacity.
         * @param prefix_bytes Exact fixed contribution of this stage.
         */
        void setBudget(MoEOverlayCapacityResolverInput &input, DeviceId device,
            std::size_t capacity, std::size_t prefix_bytes = 0)
        {
            const auto found = std::find_if(input.physical_budgets.begin(), input.physical_budgets.end(),
                [&](const auto &value) { return value.device() == device; });
            if (found == input.physical_budgets.end()) throw std::logic_error("Missing fixture allocator");
            *found = budget(device, capacity, prefix_bytes);
        }

        /** @return Every catalogued quantized source and all three floating types. */
        std::vector<ExpertWeightFormat> sourceFormats()
        {
            std::vector<ExpertWeightFormat> formats;
            for (const auto &source : native_vnni_formats::kAllSourceFormats)
                formats.push_back(ExpertWeightFormat::nativeVnni({
                    .codebook_id = source.metadata->codebook_id,
                    .is_superblock = source.metadata->is_superblock, .present = true}));
            for (auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
                formats.push_back(ExpertWeightFormat::floating(type));
            return formats;
        }

        /** @brief Own immutable request referents for a metadata-only native TP stage. */
        struct NativeStage
        {
            MoEOverlayCapacityResolverInput geometry;
            MoERoutedExpertPlacementPlan plan;
            MoEOverlayCapacityAdmissionPolicy policy;
            std::vector<MoEOverlayBoundPhysicalMemoryBudget> budgets;

            /**
             * @brief Bind a homogeneous GPU stage to the production capacity adapter.
             * @param input Owned interval and exact device identities.
             * @param replicas Requested upper bound for each layer/participant.
             */
            NativeStage(MoEOverlayCapacityResolverInput input, std::uint32_t replicas)
                : geometry(std::move(input))
            {
                plan.enabled = true;
                plan.first_model_layer = geometry.layer_weight_manifest.front().layer_idx;
                plan.authority_execution = MoEOverlayAuthorityExecutionKind::DeviceResident;
                plan.continuation_domain = plan.base_model_domain = plan.shared_expert_domain = "compute";
                RoutedExpertDomain domain;
                domain.name = "compute"; domain.scope = ExecutionDomainScope::RANK_LOCAL; domain.owner_rank = 0;
                domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
                for (const auto &physical : geometry.physical_budgets)
                {
                    budgets.emplace_back(physical.resourceId(), physical.certificate());
                    const auto device = physical.device();
                    if (!device.is_gpu()) continue;
                    domain.participants.push_back(device.type == DeviceType::CUDA
                        ? GlobalDeviceAddress::cuda(device.ordinal) : GlobalDeviceAddress::rocm(device.ordinal));
                    domain.backend = device.type == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                }
                plan.domains = {domain};
                plan.routed_tiers = {{.name = "experts", .domain = "compute", .priority = 0, .fallback = true}};
                setReplicaPolicy(replicas);
            }

            /** @brief Set exact directory/workspace geometry for @p replicas. */
            void setReplicaPolicy(std::uint32_t replicas)
            {
                const auto layers = static_cast<std::uint32_t>(geometry.layer_weight_manifest.size());
                const auto directory = DeviceMoETransferSlotDirectory::planBufferedCapacity(
                    layers * std::max(1u, replicas), 2u, 2u);
                policy = {.migration_storage = MoEOverlayMigrationStorageKind::DeviceTransferDirectory,
                    .device_transfer_directory_capacity = directory,
                    .device_rebalance_workspace_capacity = DeviceMoERebalanceWorkspaceCapacity{
                        .num_layers = layers, .num_experts = static_cast<std::uint32_t>(geometry.num_experts),
                        .participant_count = 2, .layer_window_count = 1,
                        .max_hot_replicas_per_participant = replicas,
                        .flags = replicas > 0 ? static_cast<std::uint32_t>(DeviceMoERebalanceFlags::HotReplicaCache) : 0u,
                        .local_transfer_slot_count = directory.total_slots, .collective_payload_slot_capacity = 1,
                        .phase = DeviceMoERebalanceStagePhase::PlanCopyApply,
                        .transfer_mode = DeviceMoERebalanceTransferMode::CompactTransferSlots},
                    .overlay_world_size = 1};
            }

            /** @return Borrowed production request; this fixture must outlive admission. */
            MoEOverlayStageCapacityRequest request() const
            {
                return {plan, geometry.num_experts, geometry.layer_weight_manifest, budgets, policy};
            }
        };
    }

    TEST(MoEOverlayPipelineCapacity, AllFormatsAndBackendsRetainOneAggregateAuthority)
    {
        for (const auto &format : sourceFormats())
        for (int order = 0; order < 3; ++order)
        for (int width : {1, 2, 4})
        for (auto copies : {MoEOverlayTierCopyPolicy::Apportioned,
                            MoEOverlayTierCopyPolicy::Replicated,
                            MoEOverlayTierCopyPolicy::GateUpOwnedDownColumns})
        {
            if (order == 2 && (width != 1 || copies == MoEOverlayTierCopyPolicy::GateUpOwnedDownColumns)) continue;
            SCOPED_TRACE(::testing::Message() << "order=" << order << " width=" << width
                << " policy=" << static_cast<int>(copies));
            std::vector<DeviceId> first, second;
            for (int device = 0; device < width; ++device)
            {
                first.push_back(order == 2 ? DeviceId::cpu() : order == 0 ? DeviceId::rocm(device) : DeviceId::cuda(device));
                second.push_back(order == 2 ? DeviceId::cpu() : order == 0 ? DeviceId::cuda(device) : DeviceId::rocm(device));
            }
            const std::array inputs{stage(0, 2, first, format, copies), stage(32, 3, second, format, copies)};
            const auto left = MoEOverlayCapacityResolver::resolve(inputs[0]);
            const auto right = MoEOverlayCapacityResolver::resolve(inputs[1]);
            const auto result = MoEOverlayCapacityResolver::resolvePipeline(inputs);
            ASSERT_EQ(result.size(), 2u);
            EXPECT_EQ(result[0].physical_memory_admission, result[1].physical_memory_admission);
            EXPECT_EQ(result[0].physical_memory_admission->plan().totalBytes(),
                left.physical_memory_admission->plan().totalBytes() + right.physical_memory_admission->plan().totalBytes());
            for (std::size_t index = 0; index < result.size(); ++index)
            {
                const auto &isolated = index == 0 ? left : right;
                EXPECT_EQ(result[index].tier(0)->live_experts_per_layer, isolated.tier(0)->live_experts_per_layer);
                EXPECT_EQ(result[index].tier(0)->participant_live_copies, isolated.tier(0)->participant_live_copies);
                EXPECT_EQ(result[index].layer_footprints.front().layer_idx, index == 0 ? 0 : 32);
                for (const auto &resource : result[index].physical_resources)
                {
                    EXPECT_EQ(resource.authority(), result[0].physical_memory_admission);
                    EXPECT_EQ(resource.live_copies_per_layer, isolated.resource(resource.resourceId())->live_copies_per_layer);
                    EXPECT_EQ(resource.shadow_arrival_capacity_per_layer,
                        isolated.resource(resource.resourceId())->shadow_arrival_capacity_per_layer);
                }
            }
            if (order != 2)
                EXPECT_EQ(result[0].resource(name(DeviceId::cpu()))->bom().bytes(PhysicalMemoryOwner::PrefixHostTier),
                    2u * 4096u * width);
        }
    }

    TEST(MoEOverlayPipelineCapacity, SharedAllocatorRejectsIndividuallyFittingExpertBanks)
    {
        std::array inputs{stage(0, 2, {DeviceId::cpu()}), stage(32, 2, {DeviceId::cpu()})};
        const auto isolated = MoEOverlayCapacityResolver::resolve(inputs[0]);
        const auto one_stage_bytes = isolated.resource(name(DeviceId::cpu()))->usedBytes();
        for (auto &input : inputs) setBudget(input, DeviceId::cpu(), one_stage_bytes + one_stage_bytes / 2);
        for (const auto &input : inputs) EXPECT_NO_THROW((void)MoEOverlayCapacityResolver::resolve(input));
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), MoEOverlayCapacityExhausted);
        for (auto &input : inputs) setBudget(input, DeviceId::cpu(), 2 * one_stage_bytes);
        const auto admitted = MoEOverlayCapacityResolver::resolvePipeline(inputs);
        EXPECT_EQ(admitted[0].resource(name(DeviceId::cpu()))->remainingBytes(), 0u);
    }

    TEST(MoEOverlayPipelineCapacity, SharedRamPrefixCapacityIsNotGrantedOncePerStage)
    {
        std::array inputs{stage(0, 2, {DeviceId::rocm(0), DeviceId::rocm(1)}),
                          stage(32, 2, {DeviceId::cuda(0), DeviceId::cuda(1)})};
        for (auto &input : inputs) setBudget(input, DeviceId::cpu(), 12'000, 8192);
        for (const auto &input : inputs) EXPECT_NO_THROW((void)MoEOverlayCapacityResolver::resolve(input));
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), MoEOverlayCapacityExhausted);
    }

    TEST(MoEOverlayPipelineCapacity, LaterExactQuotaPrecedesEarlierAutomaticPlacement)
    {
        const auto gpu = DeviceId::cuda(0);
        const auto cpu = DeviceId::cpu();
        auto early = stage(0, 1, {gpu});
        auto late = stage(32, 1, {gpu});
        early.tiers[0].fallback = false;
        auto cold = early.tiers[0];
        cold.tier_index = 1; cold.tier_name = "coverage"; cold.priority = 1; cold.fallback = true;
        cold.participants[0].participant_id = 1; cold.participants[0].resource_id = name(cpu);
        early.tiers.push_back(std::move(cold));
        late.tiers[0].quota_mode = MoEOverlayLiveQuotaMode::FixedPerLayer;
        late.tiers[0].fixed_live_experts_per_layer = {4};
        const auto footprint = MoEOverlayCapacityResolver::preparedFootprints(late.layer_weight_manifest).front();
        const auto gpu_bytes = 4u * footprint.gpu_live_bytes + 2u * footprint.gpu_shadow_bytes;
        setBudget(early, gpu, gpu_bytes); setBudget(late, gpu, gpu_bytes);
        const std::array inputs{early, late};
        const auto result = MoEOverlayCapacityResolver::resolvePipeline(inputs);
        EXPECT_EQ(result[0].tier(0)->live_experts_per_layer, std::vector<int>{0});
        EXPECT_EQ(result[0].tier(1)->live_experts_per_layer, std::vector<int>{4});
        EXPECT_EQ(result[1].tier(0)->live_experts_per_layer, std::vector<int>{4});
        EXPECT_EQ(result[0].resource(name(gpu))->remainingBytes(), 0u);
    }

    TEST(MoEOverlayPipelineCapacity, StageViewsMaterializeThroughOneLedgerAndRetireIndependently)
    {
        const auto cpu = DeviceId::cpu();
        const std::array inputs{stage(0, 2, {cpu}), stage(32, 3, {cpu})};
        const auto result = MoEOverlayCapacityResolver::resolvePipeline(inputs);
        const auto one = MoEOverlayCapacityResolver::resolve(inputs[0]).resource(name(cpu))->liveExpertBytes();
        const auto two = MoEOverlayCapacityResolver::resolve(inputs[1]).resource(name(cpu))->liveExpertBytes();
        PhysicalMemoryMaterializationLedger ledger(result[0].physical_memory_admission);
        auto first = ledger.claimNewAllocation(identity(cpu), PhysicalMemoryOwner::RoutedExpertWeights, one);
        auto second = ledger.claimNewAllocation(identity(cpu), PhysicalMemoryOwner::RoutedExpertWeights, two);
        EXPECT_THROW((void)ledger.claimNewAllocation(identity(cpu), PhysicalMemoryOwner::RoutedExpertWeights, 1u), std::logic_error);
        first = {};
        auto replacement = ledger.claimNewAllocation(identity(cpu), PhysicalMemoryOwner::RoutedExpertWeights, one);
        EXPECT_TRUE(second.valid()); EXPECT_TRUE(replacement.valid());
        EXPECT_EQ(result[0].resource(name(cpu))->live_copies_per_layer, (std::vector<int>{4, 4}));
        EXPECT_EQ(result[1].resource(name(cpu))->live_copies_per_layer, (std::vector<int>{4, 4, 4}));
    }

    TEST(MoEOverlayPipelineCapacity, RejectsAliasesOfOnePhysicalAllocatorInSingleStage)
    {
        auto input = stage(0, 1, {DeviceId::cuda(0)});
        input.physical_budgets.emplace_back("another-name", input.physical_budgets.front().certificate());
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolve(input), std::invalid_argument);
    }

    TEST(MoEOverlayPipelineCapacity, RejectsAliasesOfOnePhysicalAllocatorAcrossStages)
    {
        std::array inputs{stage(0, 1, {DeviceId::cpu()}), stage(32, 1, {DeviceId::cpu()})};
        inputs[1].physical_budgets[0] = MoEOverlayPhysicalMemoryBudget("another-name", inputs[1].physical_budgets[0].certificate());
        inputs[1].tiers[0].participants[0].resource_id = "another-name";
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), std::invalid_argument);
    }

    TEST(MoEOverlayPipelineCapacity, RejectsConflictingObservationsAndForeignStageBindings)
    {
        const std::array valid{stage(0, 1, {DeviceId::rocm(0)}), stage(32, 1, {DeviceId::cuda(0)})};
        auto inputs = valid;
        setBudget(inputs[1], DeviceId::cpu(), available - 1);
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), std::invalid_argument);
        inputs = valid;
        inputs[1].tiers[0].participants[0].resource_id = name(DeviceId::rocm(0));
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), std::invalid_argument);
        inputs = valid;
        inputs[1].physical_budgets[0] = MoEOverlayPhysicalMemoryBudget(name(DeviceId::rocm(0)),
            inputs[1].physical_budgets[0].certificate());
        inputs[1].tiers[0].participants[0].resource_id = name(DeviceId::rocm(0));
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), std::invalid_argument);
    }

    TEST(MoEOverlayPipelineCapacity, RejectsMissingOverlappingAndUnorderedStages)
    {
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline({}), std::invalid_argument);
        const std::array valid{stage(0, 2, {DeviceId::cpu()}), stage(32, 2, {DeviceId::cpu()})};
        auto inputs = valid;
        inputs[1].layer_weight_manifest.clear();
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), std::invalid_argument);
        inputs = valid; inputs[1].physical_budgets.clear();
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), std::invalid_argument);
        inputs = valid; inputs[1].layer_weight_manifest = inputs[0].layer_weight_manifest;
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), std::invalid_argument);
        inputs = valid; std::swap(inputs[0], inputs[1]);
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolvePipeline(inputs), std::invalid_argument);
    }

    TEST(MoEOverlayPipelineCapacity, JointReplicaSearchPreservesEveryStageAndRetainedGrant)
    {
        for (const auto &format : sourceFormats())
        for (bool reverse : {false, true})
        {
            const auto devices = [&](bool rocm)
            { return rocm ? std::vector{DeviceId::rocm(0), DeviceId::rocm(1)}
                          : std::vector{DeviceId::cuda(0), DeviceId::cuda(1)}; };
            std::array stages{NativeStage(stage(0, 2, devices(reverse), format), 3),
                              NativeStage(stage(32, 3, devices(!reverse), format), 2)};
            const std::array requests{stages[0].request(), stages[1].request()};
            const auto exact = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
            // Set real byte boundaries from exact smaller policies, then request
            // larger caches. Each stage must retain its own largest fitting grant.
            for (std::size_t index = 0; index < stages.size(); ++index)
            {
                auto &value = stages[index];
                for (auto &physical : value.budgets)
                    if (physical.device().is_gpu())
                        physical = MoEOverlayBoundPhysicalMemoryBudget(physical.resourceId(),
                            budget(physical.device(), exact[index].resource(physical.resourceId())->usedBytes()).certificate());
                value.setReplicaPolicy(4);
            }
            const auto bounded = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
            ASSERT_EQ(bounded.size(), 2u);
            EXPECT_EQ(bounded[0].physical_memory_admission, bounded[1].physical_memory_admission);
            for (std::size_t index = 0; index < stages.size(); ++index)
            {
                ASSERT_TRUE(bounded[index].replica_cache_capacity);
                EXPECT_EQ(bounded[index].replica_cache_capacity->requested(), 4);
                EXPECT_EQ(bounded[index].replica_cache_capacity->admitted(), index == 0 ? 3 : 2);
                EXPECT_EQ(bounded[index].tier(0)->live_experts_per_layer,
                    std::vector<int>(index == 0 ? 2 : 3, 4));
                stages[index].plan = MoEOverlayCapacityResolver::installResolvedQuotas(stages[index].plan, bounded[index]);
                stages[index].setReplicaPolicy(index == 0 ? 3 : 2);
            }
            const auto retained = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
            EXPECT_EQ(retained[0].replica_cache_capacity, bounded[0].replica_cache_capacity);
            EXPECT_EQ(retained[1].replica_cache_capacity, bounded[1].replica_cache_capacity);
            stages[1].setReplicaPolicy(4);
            EXPECT_THROW((void)MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests), std::logic_error);
        }
    }

    TEST(MoEOverlayPipelineCapacity, MinimumCacheFailureDoesNotDisableEitherStage)
    {
        std::array stages{NativeStage(stage(0, 2, {DeviceId::rocm(0), DeviceId::rocm(1)}), 1),
                          NativeStage(stage(32, 2, {DeviceId::cuda(0), DeviceId::cuda(1)}), 1)};
        const std::array requests{stages[0].request(), stages[1].request()};
        const auto exact = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
        auto &limiting = stages[1].budgets.front();
        limiting = MoEOverlayBoundPhysicalMemoryBudget(limiting.resourceId(),
            budget(limiting.device(), exact[1].resource(limiting.resourceId())->usedBytes() - 1).certificate());
        for (auto &value : stages) value.setReplicaPolicy(4);
        EXPECT_THROW((void)MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests), MoEOverlayCapacityExhausted);
        stages[1].policy.device_rebalance_workspace_capacity.reset();
        EXPECT_THROW((void)MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests), std::invalid_argument);
    }

    TEST(MoEOverlayPipelineCapacity, SharedGpuReplicaBudgetPreservesLaterMinimumAndRetainedGrant)
    {
        for (const auto &format : sourceFormats())
        for (bool rocm : {false, true})
        {
            const auto devices = rocm ? std::vector{DeviceId::rocm(0), DeviceId::rocm(1)}
                                      : std::vector{DeviceId::cuda(0), DeviceId::cuda(1)};
            std::array stages{NativeStage(stage(0, 2, devices, format), 3),
                              NativeStage(stage(32, 2, devices, format), 1)};
            const std::array requests{stages[0].request(), stages[1].request()};
            const auto exact = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
            for (auto &value : stages)
            {
                for (auto &physical : value.budgets)
                    if (physical.device().is_gpu())
                        physical = MoEOverlayBoundPhysicalMemoryBudget(physical.resourceId(),
                            budget(physical.device(), exact[0].resource(physical.resourceId())->usedBytes()).certificate());
                value.setReplicaPolicy(4);
                const auto isolated = MoEOverlayCapacityAdmission::resolveCapacity(value.plan, value.geometry.num_experts,
                    value.geometry.layer_weight_manifest, value.budgets, value.policy);
                EXPECT_EQ(isolated.replica_cache_capacity->admitted(), 4);
            }
            const auto jointly = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
            EXPECT_EQ(jointly[0].replica_cache_capacity->admitted(), 3);
            EXPECT_EQ(jointly[1].replica_cache_capacity->admitted(), 1);
            EXPECT_EQ(jointly[0].resource(name(devices[0]))->remainingBytes(), 0u);
            // A retained later-stage grant has priority over new optional
            // replicas in the earlier stage, regardless of declaration order.
            stages[1].plan.replica_cache_capacity.emplace(4, 3);
            stages[1].setReplicaPolicy(3);
            const auto retained = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
            EXPECT_EQ(retained[0].replica_cache_capacity->admitted(), 1);
            EXPECT_EQ(retained[1].replica_cache_capacity->admitted(), 3);
        }
    }

    TEST(MoEOverlayPipelineCapacity, ExplicitDisabledStorageAndDisabledReplicaCacheRemainDistinct)
    {
        std::array stages{NativeStage(stage(0, 2, {DeviceId::rocm(0), DeviceId::rocm(1)}), 0),
                          NativeStage(stage(32, 2, {DeviceId::cuda(0), DeviceId::cuda(1)}), 4)};
        const std::array requests{stages[0].request(), stages[1].request()};
        const auto cache_off = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
        EXPECT_EQ(cache_off[0].replica_cache_capacity->admitted(), 0);
        EXPECT_EQ(cache_off[1].replica_cache_capacity->admitted(), 4);
        EXPECT_GT(cache_off[0].resource(name(DeviceId::rocm(0)))->transferStagingBytes(), 0u);
        stages[0].policy = {};
        const auto movement_off = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
        EXPECT_FALSE(movement_off[0].replica_cache_capacity);
        EXPECT_EQ(movement_off[0].resource(name(DeviceId::rocm(0)))->transferStagingBytes(), 0u);
        EXPECT_EQ(movement_off[1].replica_cache_capacity->admitted(), 4);
        EXPECT_EQ(movement_off[0].physical_memory_admission, movement_off[1].physical_memory_admission);
    }
} // namespace llaminar2
