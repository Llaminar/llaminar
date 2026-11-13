/**
 * @file Test__NativeStoragePreparingPlacement.cpp
 * @brief Mandatory native preparation floor precedes real optional table fill.
 * Real SDK-shaped root declarations feed canonical PMA and the production
 * selector. No model/device allocation occurs while source state is Preparing.
 */
#include "collective/NativeStoragePreparationPmaBinding.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include <gtest/gtest.h>
#include <array>
using namespace llaminar2;
#include "fixtures/planning/NativeStoragePreparingReplayAccess.h"
TEST(NativeStoragePreparingPlacement,
     SourceFloorDisplacesOptionalCpuAndGpuTablesBeforePhysicalModelAllocation)
{
    constexpr std::size_t GiB = std::size_t{1} << 30;
    const auto extent =
        ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0), 131072u, 4096u);
    for (bool gpuConsumer : {false, true})
    {
        const PhysicalMemoryResource gpu{0, DeviceId::cuda(0), GiB, GiB},
            host{0, DeviceId::cpu(), GiB, GiB};
        const std::array observed{gpu, host};
        auto preparing = llaminar2::test::NativeStoragePreparationReplayAccess::begin(observed);
        auto memory = preparing->memory();
        auto authority = preparing->preparationAuthority();
        llaminarNativeRootRequestV1 container{1,
                                              sizeof(llaminarNativeRootRequestV1),
                                              0x100,
                                              0x100,
                                              4096,
                                              4096,
                                              4096,
                                              0,
                                              authority.setup_epoch,
                                              llaminarNativeHostCPU0V1,
                                              0,
                                              -1,
                                              llaminarNativeCommMetadataV1,
                                              llaminarNativeCommLifetimeV1,
                                              llaminarNativeHostAnonymousPagesV1,
                                              llaminarNativeRetainedSetupV1,
                                              0};
        void *token = nullptr;
        ASSERT_EQ(authority.claim(authority.context, &container, &token),
                  llaminarNativeStorageSuccessV1);
        ASSERT_EQ(authority.materialize(token), llaminarNativeStorageSuccessV1);
        auto snapshot = preparing->closeForPlanning();
        auto early = container;
        early.owner_identity = early.setup_epoch = 0;
        std::vector<NativeStorageCreatorDeclaration> creators{
            {NativeStorageCreatorUse::PreparationOwner, early}};
        if (gpuConsumer)
        {
            auto native = early;
            native.shared_anchor_identity = 0x200;
            native.resource_kind = llaminarNativeCUDADeviceV1;
            native.owner_family = llaminarNativeP2pBackingV1;
            native.allocator_kind = llaminarNativeCUDADeviceVmmV1;
            creators.push_back({NativeStorageCreatorUse::RetainedOwner, native});
        }
        auto mandatory = NativeMandatoryStorageDeclarations::fromCreators(snapshot, creators);
        const std::array consumers{gpuConsumer ? gpu : host};
        auto input = ImmutableRowTablePlacementInput::forConsumers(extent, host, consumers);
        const ImmutableRowTablePolicy policy{ImmutableRowStoragePreference::Automatic, 65536u, 1u};
        const ImmutableRowTableModeContributions contributions{.vram = PhysicalMemoryPlan{}};
        PhysicalMemoryPlanBuilder old;
        old.add(consumers[0], PhysicalMemoryOwner::PrimaryModelWeights, GiB - extent.nativeBytes());
        const auto without =
            ImmutableRowTablePlacementPlanner::select(policy, input, old.build(), contributions);
        ASSERT_EQ(without.placement->storage(),
                  gpuConsumer ? ImmutableRowStorage::VRAM : ImmutableRowStorage::RAM);
        PhysicalMemoryPlanBuilder fixed(mandatory.mandatoryBOM());
        fixed.add(consumers[0], PhysicalMemoryOwner::PrimaryModelWeights,
                  GiB - extent.nativeBytes());
        auto selected =
            ImmutableRowTablePlacementPlanner::select(policy, input, fixed.build(), contributions);
        ASSERT_EQ(selected.placement->storage(),
                  gpuConsumer ? ImmutableRowStorage::RAM : ImmutableRowStorage::MemoryMapped);
        EXPECT_EQ(memory->phase(), PhysicalMemoryAuthorityPhase::Preparing);
        EXPECT_THROW(memory->claimNewAllocation(consumers[0].device,
                                                PhysicalMemoryOwner::PrimaryModelWeights, 1),
                     std::logic_error);
        auto final = preparing->admit(std::move(snapshot), std::move(mandatory),
                                      std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(
                                          std::move(selected.physical_plan)));
        EXPECT_EQ(memory->phase(), PhysicalMemoryAuthorityPhase::Admitted);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::LocalCollective,
                                       PhysicalMemoryMaterializationKind::NewAllocation),
                  4096);
        EXPECT_EQ(memory->committedBytes(DeviceId::cpu(), PhysicalMemoryOwner::LocalCollective,
                                         PhysicalMemoryMaterializationKind::NewAllocation),
                  4096);
        auto a = final->authority();
        void *deviceToken = nullptr;
        if (gpuConsumer)
        {
            auto request = creators.back().geometry;
            request.owner_identity = 0x300;
            request.setup_epoch = a.setup_epoch;
            ASSERT_EQ(a.claim(a.context, &request, &deviceToken), llaminarNativeStorageSuccessV1);
            ASSERT_EQ(a.materialize(deviceToken), llaminarNativeStorageSuccessV1);
        }
        final->sealForExecution();
        EXPECT_EQ(authority.release(&token), llaminarNativeStorageSuccessV1);
        if (deviceToken)
            EXPECT_EQ(a.release(&deviceToken), llaminarNativeStorageSuccessV1);
        final->retirementGate().requireRetired();
    }
}
