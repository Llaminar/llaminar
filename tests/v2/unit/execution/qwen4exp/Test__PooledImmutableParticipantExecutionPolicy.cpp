/**
 * @file Test__PooledImmutableParticipantExecutionPolicy.cpp
 * @brief Original participant admission separates storage ownership from readers and bounds retained MTP policies.
 *
 * Checked geometry exercises both GPU backend identities without opening a
 * runtime. Native source-admission cases separately prove physical publication.
 */
#include "execution/qwen4exp/PooledImmutableParticipantExecutionPolicy.h"
#include <gtest/gtest.h>
#include "execution/mtp/MTPDeviceGenerationPolicy.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "models/GraphTypes.h"
#include "kernels/qwen4exp/PooledImmutableSourceGeometry.h"
#include "loaders/ImmutableRowTable.h"
#include "../../../utils/CPUJointSourceTestPlan.h"
#include <functional>
#include <algorithm>
#include <array>
#include <vector>
namespace llaminar2::test {
/** @test Query-only readers require transport even when the coordinator is the sole storage owner. */
TEST(PooledImmutableParticipantExecutionPolicy, SharedReadersRequireTransportWithoutRemoteStorage)
{
    namespace q = qwen4exp;
    using Policy = q::PooledImmutableParticipantExecutionPolicy;
    const ImmutableRowTableSource source{.path="declaration_only", .tensor_name="per_layer_token_embd.weight",
        .rows=1024, .row_elements=160, .format=GGUFTensorType::IQ4_NL};
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const unsigned owners : {1u, 2u, 4u, 8u})
    for (const unsigned readers : {1u, 2u, 4u, 8u})
    for (const unsigned slots : {0u, 64u})
    for (const bool coordinatorOwnsRows : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "backend=" << int(backend) << " owners=" << owners
            << " readers=" << readers << " slots=" << slots << " local=" << coordinatorOwnsRows);
        const PhysicalMemoryAllocatorIdentity coordinator{0, DeviceId(backend, 0)};
        std::vector<PhysicalMemoryAllocatorIdentity> order;
        std::vector<ImmutableRowTableLayoutOwner> storage;
        for (unsigned i = 0; i < owners; ++i)
        {
            order.push_back({0, DeviceId(backend, int(i + (coordinatorOwnsRows ? 0 : 1)))});
            storage.push_back({{order.back(), ImmutableRowStorage::VRAM}, 0, slots, {}});
        }
        const std::array tiers{ImmutableRowStorage::VRAM, ImmutableRowStorage::MemoryMapped};
        const auto layout = ImmutableRowTableLayoutGeometry::checked(
            ImmutableRowTableExtent::fromNative(source.format, source.rows, 90),
            ImmutableRowTableTiers::of(tiers), ImmutableRowTableResidency::Adaptive, storage,
            {coordinator}, {}, {{{0, DeviceId::cpu()},
                ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},
            {}, slots ? std::optional(coordinator) : std::nullopt, 64);
        const auto physical = q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout, order, coordinator);
        const auto geometry = q::PooledImmutableCoordinatorGeometry::fromOwners(physical, {65536, 3, 4}, 64);
        const auto coordinatorPlan = q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical, geometry, 701, 17);
        std::vector<q::PooledImmutableOwnerNativePlan> plans;
        for (unsigned i = 0; i < owners; ++i)
            plans.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(physical, geometry, i, 701, 17));
        const auto declared = q::PooledImmutableSourceGeometry::fromPlans(plans, coordinatorPlan, source, readers);
        const bool needsParticipants = readers > 1 || owners > 1 || !coordinatorOwnsRows;
        EXPECT_EQ(Policy::requiresRetainedParticipants(declared), needsParticipants);
        if (coordinatorOwnsRows && owners == 1 && readers > 1)
        {
            // Causal control: the former owner-only predicate omits every
            // query-only reader in this real source-admission configuration.
            EXPECT_FALSE(std::any_of(plans.begin(), plans.end(), [&](const auto& plan) {
                return plan.allocator() != coordinator;
            }));
            EXPECT_TRUE(Policy::requiresRetainedParticipants(declared));
        }
    }
}

TEST(PooledImmutableParticipantExecutionPolicy, OffRetainsCapacityAndBoundsRequestSwitches){
    using Policy=qwen4exp::PooledImmutableParticipantExecutionPolicy;
    for(const int capacity:{0,1,3,sampling_math::DeviceGenerationPolicy::kMaximumSupportedDraftDepth}){
        MTPRuntimeConfig off;off.enabled=false;off.graph_capacity_draft_tokens=capacity;
        const auto before=off.graph_capacity_draft_tokens;
        EXPECT_NO_THROW(Policy::requireConfiguration(off));
        EXPECT_EQ(off.graph_capacity_draft_tokens,before);
        EXPECT_EQ(retainsMTPGraphCapacity(off),capacity>0);
        EXPECT_NO_THROW(Policy::requireAdmission(sampling_math::DeviceGenerationPolicy::ordinary(),off));
        EXPECT_NO_THROW(Policy::requireAdmission(sampling_math::DeviceGenerationPolicy::forwardOnly(),off));
        for(const int depth:{1,3}){
            const auto request=sampling_math::DeviceGenerationPolicy::fixed(depth);
            if(depth<=capacity)EXPECT_NO_THROW(Policy::requireAdmission(request,off));
            else EXPECT_THROW(Policy::requireAdmission(request,off),std::invalid_argument);
        }
        auto dynamic=sampling_math::DeviceGenerationPolicy::fixed(1);
        dynamic.mode=sampling_math::DeviceGenerationPolicyMode::Dynamic;dynamic.maximum_depth=3;
        ASSERT_TRUE(dynamic.valid());
        if(capacity>=3)EXPECT_NO_THROW(Policy::requireAdmission(dynamic,off));
        else EXPECT_THROW(Policy::requireAdmission(dynamic,off),std::invalid_argument);

    }
    EXPECT_THROW(Policy::requireAdmission(sampling_math::DeviceGenerationPolicy::fixed(0),MTPRuntimeConfig{}),std::invalid_argument);
}
/** @test Real host-to-controller conversion admits fixed and dynamic requests within the same retained capacity. */
TEST(PooledImmutableParticipantExecutionPolicy, ActiveCanonicalPoliciesRequireOriginalRetainedCapacity){
 using Policy=qwen4exp::PooledImmutableParticipantExecutionPolicy;
 for(const int depth:{1,3}){
  MTPRuntimeConfig active;active.enabled=true;active.draft_tokens=depth;active.graph_capacity_draft_tokens=15;
  const auto before=active.graph_capacity_draft_tokens;
  const auto request=resolveMTPDeviceGenerationDepthPolicy(active);ASSERT_TRUE(request.valid());
  EXPECT_NO_THROW(Policy::requireConfiguration(active));
  EXPECT_NO_THROW(Policy::requireAdmission(request,active));
  EXPECT_EQ(active.graph_capacity_draft_tokens,before);EXPECT_TRUE(active.enabled);
  auto tooSmall=active;tooSmall.graph_capacity_draft_tokens=depth-1;
  if(depth>1)EXPECT_THROW(Policy::requireConfiguration(tooSmall),std::invalid_argument);
 }
 MTPRuntimeConfig dynamic;dynamic.enabled=true;dynamic.graph_capacity_draft_tokens=15;
 dynamic.depth_policy.mode=MTPDepthPolicyMode::Dynamic;dynamic.depth_policy.min_depth=1;
 dynamic.depth_policy.max_depth=15;dynamic.depth_policy.initial_depth=1;
 const auto request=resolveMTPDeviceGenerationDepthPolicy(dynamic);ASSERT_TRUE(request.valid());
 EXPECT_NO_THROW(Policy::requireConfiguration(dynamic));
 EXPECT_NO_THROW(Policy::requireAdmission(request,dynamic));
 auto insufficient=dynamic;insufficient.depth_policy.max_depth=3;insufficient.graph_capacity_draft_tokens=3;
 EXPECT_THROW(Policy::requireAdmission(request,insufficient),std::invalid_argument);
 auto batched=dynamic;batched.max_request_batch=2;EXPECT_THROW(Policy::requireConfiguration(batched),std::invalid_argument);
 auto negative=dynamic;negative.graph_capacity_draft_tokens=-1;EXPECT_THROW(Policy::requireConfiguration(negative),std::invalid_argument);
 auto malformed=request;malformed.minimum_depth=0;EXPECT_THROW(Policy::requireAdmission(malformed,dynamic),std::invalid_argument);
}

/** @test Independent real descriptor resolution preserves policy while original ownership remains separate. */
TEST(PooledImmutableParticipantExecutionPolicy, SharedReaderUsesCompleteDescriptorNotAllocationIdentity)
{
    using Policy = qwen4exp::PooledImmutableParticipantExecutionPolicy;
    for (const bool cuda : {true, false})
    {
        SCOPED_TRACE(cuda ? "CUDA metadata" : "ROCm metadata");
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(cpuJointSourceTestPlan());
        auto& compute = plan->domains[0];
        compute.backend = cuda ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        compute.participants = cuda
            ? std::vector{GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}
            : std::vector{GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        auto& cpu = plan->domains[1];
        cpu.scope = ExecutionDomainScope::SINGLE;
        cpu.participants = {GlobalDeviceAddress::cpu(0)};
        cpu.world_ranks = {0}; cpu.owner_rank = 0;
        cpu.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        // This is exactly the per-child factory resolution sequence. No native
        // context, model ownership or ready storage is fabricated by this test.
        const auto resolve = [&] {
            const auto runtime = resolveMoEExpertOverlayRuntimePlan(plan, {.current_world_rank = 0});
            return std::make_shared<const MoEExpertOverlayExecutionPlan>(
                buildMoEExpertOverlayExecutionPlan(*runtime, 1));
        };
        const auto original = resolve();
        GraphConfig reader; reader.n_layers = reader.total_n_layers = 1;
        reader.moe.expert_overlay_execution_plan = resolve();
        ASSERT_NE(reader.moe.expert_overlay_execution_plan.get(), original.get());
        // The former pointer-only predicate rejected this authentic sibling.
        EXPECT_TRUE(reader.moe.expert_overlay_execution_plan != original);
        EXPECT_NO_THROW(Policy::requireGraphDescriptor(original, reader, Policy::GraphRole::ReplicatedReader));
        EXPECT_THROW(Policy::requireGraphDescriptor(original, reader, Policy::GraphRole::Source), std::invalid_argument);
        auto source = reader; source.moe.expert_overlay_execution_plan = original;
        EXPECT_NO_THROW(Policy::requireGraphDescriptor(original, source, Policy::GraphRole::Source));
        EXPECT_THROW(Policy::requireGraphDescriptor({}, source, Policy::GraphRole::Source), std::invalid_argument);
        auto missing = reader; missing.moe.expert_overlay_execution_plan.reset();
        EXPECT_THROW(Policy::requireGraphDescriptor(original, missing, Policy::GraphRole::ReplicatedReader), std::invalid_argument);
        for (const auto change : std::vector<std::function<void(MoEExpertOverlayExecutionPlan&)>>{
            [](auto& p) { ++p.world_size; },
            [](auto& p) { p.continuation_domain += "foreign"; },
            [](auto& p) { p.base_model_domain += "foreign"; },
            [](auto& p) { p.shared_expert_domain += "foreign"; },
            [](auto& p) { ++p.continuation_root_rank; },
            [](auto& p) { std::swap(p.domains[0].participants[0], p.domains[0].participants[1]); },
            [](auto& p) { ++p.domains[0].participants[0].world_rank; },
            [](auto& p) { p.domains[0].participants[0].locally_addressable = false; },
            [](auto& p) { p.domains[0].participants[0].local_device = DeviceId::cpu(); },
            [](auto& p) { p.domains[0].routed_compute_policy = RoutedExpertComputePolicy::TensorSharded; },
            [](auto& p) { p.domains[0].rebalance_domain_id += "foreign"; },
            [](auto& p) { p.current_rank.loads_root_weights = !p.current_rank.loads_root_weights; },
            [](auto& p) { p.rank_plans[0].owned_domains.push_back("foreign"); }})
        {
            auto changed = std::make_shared<MoEExpertOverlayExecutionPlan>(*original);
            change(*changed);
            auto foreign = reader; foreign.moe.expert_overlay_execution_plan = std::move(changed);
            EXPECT_THROW(Policy::requireGraphDescriptor(original, foreign, Policy::GraphRole::ReplicatedReader),
                std::invalid_argument);
        }
        auto partial = reader; partial.pp_layer_offset = 1;
        EXPECT_THROW(Policy::requireGraphDescriptor(original, partial, Policy::GraphRole::ReplicatedReader), std::invalid_argument);
        partial = reader; partial.total_n_layers = 2;
        EXPECT_THROW(Policy::requireGraphDescriptor(original, partial, Policy::GraphRole::ReplicatedReader), std::invalid_argument);
    }
}

}
