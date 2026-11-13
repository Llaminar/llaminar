/**
 * @file Test__NativeStoragePreparing.cpp
 * @brief Actual same-state PMA preparation/publication/retirement replay.
 * No device/runtime/fresh observation/copy of canonical accounting is involved.
 */
#include "collective/NativeStoragePreparationPmaBinding.h"
#include <gtest/gtest.h>
#include <array>
#include <optional>
#include <barrier>
#include <thread>
using namespace llaminar2;
#include "fixtures/planning/NativeStoragePreparingReplayAccess.h"
using llaminar2::test::NativeStoragePreparationReplayAccess;
namespace
{
    std::array<PhysicalMemoryResource, 2> observations(std::size_t available = 100)
    {
        return {{{0, DeviceId::cuda(0), 1000, available}, {0, DeviceId::cpu(), 1000, available}}};
    }
    llaminarNativeRootRequestV1 root(llaminarNativeStorageAuthorityV1 a, bool abort = false)
    {
        return {1,
                sizeof(llaminarNativeRootRequestV1),
                abort ? 0x200u : 0x100u,
                abort ? 0x200u : 0x100u,
                abort ? 64u : 20u,
                abort ? 64u : 20u,
                abort ? 64u : 4u,
                0,
                a.setup_epoch,
                llaminarNativeHostCPU0V1,
                0,
                -1,
                uint32_t(abort ? llaminarNativeAbortFlagV1 : llaminarNativeCommMetadataV1),
                llaminarNativeCommLifetimeV1,
                uint32_t(abort ? llaminarNativeCUDAHostVmmV1 : llaminarNativeHostAnonymousPagesV1),
                llaminarNativeRetainedSetupV1,
                0};
    }
    std::array<NativeStorageCreatorDeclaration, 2> creators(llaminarNativeStorageAuthorityV1 a)
    {
        auto container = root(a), abort = root(a, true);
        container.owner_identity = abort.owner_identity = 0;
        container.setup_epoch = abort.setup_epoch = 0;
        return {{{NativeStorageCreatorUse::PreparationOwner, container},
                 {NativeStorageCreatorUse::RetainedOwner, abort}}};
    }
    PhysicalMemoryOwnerAttestation host(const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        for (auto const &r : memory->rankAttestation())
            if (r.identity.device == DeviceId::cpu() &&
                r.owner == PhysicalMemoryOwner::LocalCollective)
                return r;
        throw std::logic_error("missing canonical CPU0 row");
    }
} // namespace
TEST(NativeStoragePreparing, SameStateSourceColumnsAndOneEpochSurviveFinalPublication)
{
    auto initial = observations();
    auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
    auto memory = preparing->memory();
    auto before = preparing->preparationAuthority();
    auto container = root(before);
    void *containerToken = nullptr;
    EXPECT_EQ(memory->phase(), PhysicalMemoryAuthorityPhase::Preparing);
    EXPECT_THROW(memory->admission(), std::logic_error);
    EXPECT_THROW(
        memory->claimNewAllocation(DeviceId::cuda(0), PhysicalMemoryOwner::ExecutionWorkspace, 1),
        std::logic_error);
    EXPECT_THROW(
        memory->reserveNewAllocations(DeviceId::cpu(), PhysicalMemoryOwner::LocalCollective, 1),
        std::logic_error);
    ASSERT_EQ(before.claim(before.context, &container, &containerToken),
              llaminarNativeStorageSuccessV1);
    EXPECT_EQ(host(memory).preparation_committed_bytes, 20);
    EXPECT_FALSE(host(memory).complete());
    ASSERT_EQ(before.materialize(containerToken), llaminarNativeStorageSuccessV1);
    auto snapshot = preparing->closeForPlanning();
    auto declared = NativeMandatoryStorageDeclarations::fromCreators(snapshot, creators(before));
    auto finalCertificate =
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(declared.mandatoryBOM());
    auto final = preparing->admit(std::move(snapshot), std::move(declared), finalCertificate);
    EXPECT_FALSE(snapshot.valid());
    EXPECT_EQ(memory->phase(), PhysicalMemoryAuthorityPhase::Admitted);
    EXPECT_EQ(memory->admission(), finalCertificate);
    EXPECT_EQ(host(memory).preparation_committed_bytes, 20);
    EXPECT_EQ(host(memory).preparation_materialized_bytes, 20);
    EXPECT_EQ(host(memory).committed_new_bytes, 84);
    EXPECT_EQ(host(memory).materialized_new_bytes, 20);
    auto admitted = final->authority();
    EXPECT_EQ(admitted.setup_epoch, before.setup_epoch);
    void *invalid = nullptr;
    EXPECT_EQ(before.claim(before.context, &container, &invalid), llaminarNativeStorageInvalidV1);
    EXPECT_EQ(invalid, nullptr);
    preparing.reset(); // Successful preparation destruction must not seal/cancel final setup.
    auto abort = root(admitted, true);
    void *abortToken = nullptr;
    ASSERT_EQ(admitted.claim(admitted.context, &abort, &abortToken),
              llaminarNativeStorageSuccessV1);
    ASSERT_EQ(admitted.materialize(abortToken), llaminarNativeStorageSuccessV1);
    EXPECT_TRUE(memory->rankComplete());
    EXPECT_EQ(host(memory).materialized_new_bytes, 84);
    final->sealForExecution();
    EXPECT_FALSE(final->retirementGate().retired());
    ASSERT_EQ(before.release(&containerToken), llaminarNativeStorageSuccessV1);
    EXPECT_EQ(host(memory).preparation_committed_bytes, 0);
    EXPECT_EQ(host(memory).materialized_new_bytes, 64);
    EXPECT_FALSE(final->retirementGate().retired());
    ASSERT_EQ(admitted.release(&abortToken), llaminarNativeStorageSuccessV1);
    final->retirementGate().requireRetired();
}
TEST(NativeStoragePreparing, AllocationFailureBeforeCommitPreservesEveryIdentityAndCounter)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        auto initial = observations();
        auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
        auto a = preparing->preparationAuthority();
        auto r = root(a);
        void *token = nullptr;
        ASSERT_EQ(a.claim(a.context, &r, &token), llaminarNativeStorageSuccessV1);
        ASSERT_EQ(a.materialize(token), llaminarNativeStorageSuccessV1);
        auto snapshot = preparing->closeForPlanning();
        auto declared = NativeMandatoryStorageDeclarations::fromCreators(snapshot, creators(a));
        auto certificate =
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(declared.mandatoryBOM());
        auto memory = preparing->memory();
        const auto before = memory->rankAttestationSummary(true);
        void *identity = token;
        const auto epoch = snapshot.epochIdentity();
        NativeStoragePreparationReplayAccess::failBeforeCommit(*preparing, true);
        EXPECT_THROW(preparing->admit(std::move(snapshot), std::move(declared), certificate),
                     std::bad_alloc);
        EXPECT_TRUE(snapshot.valid());
        EXPECT_EQ(snapshot.epochIdentity(), epoch);
        EXPECT_EQ(token, identity);
        EXPECT_EQ(a.setup_epoch, preparing->preparationAuthority().setup_epoch);
        EXPECT_EQ(memory->phase(), PhysicalMemoryAuthorityPhase::Preparing);
        EXPECT_EQ(memory->rankAttestationSummary(true), before);
        EXPECT_THROW(memory->admission(), std::logic_error);
        auto gate = preparing->retirementGate();
        preparing->cancel();
        EXPECT_EQ(memory->phase(), PhysicalMemoryAuthorityPhase::Cancelled);
        EXPECT_FALSE(gate.retired());
        EXPECT_EQ(a.release(&token), llaminarNativeStorageSuccessV1);
        gate.requireRetired();
    }
}
TEST(NativeStoragePreparing, ExactRootAnchorsAndGeometryAreRequiredInCreatorProjection)
{
    auto initial = observations();
    auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
    auto a = preparing->preparationAuthority();
    auto r = root(a);
    void *token = nullptr;
    ASSERT_EQ(a.claim(a.context, &r, &token), llaminarNativeStorageSuccessV1);
    ASSERT_EQ(a.materialize(token), llaminarNativeStorageSuccessV1);
    auto snapshot = preparing->closeForPlanning();
    const auto before = preparing->memory()->rankAttestationSummary(true);
    for (int fault = 0; fault < 5; ++fault)
    {
        auto declarations = creators(a);
        switch (fault)
        {
        case 0:
            declarations[0].geometry.shared_anchor_identity++;
            break;
        case 1:
            declarations[0].geometry.allocator_flags++;
            break;
        case 2:
            declarations[0].geometry.requested_bytes = 16;
            declarations[0].geometry.physical_bytes = 16;
            break;
        case 3:
            declarations[1].geometry.shared_anchor_identity =
                declarations[0].geometry.shared_anchor_identity;
            break;
        case 4:
            declarations[0].use = NativeStorageCreatorUse::RetainedOwner;
            break;
        }
        EXPECT_THROW(NativeMandatoryStorageDeclarations::fromCreators(snapshot, declarations),
                     std::logic_error);
        EXPECT_EQ(preparing->memory()->rankAttestationSummary(true), before);
    }
    preparing->cancel();
    EXPECT_EQ(a.release(&token), llaminarNativeStorageSuccessV1);
}
TEST(NativeStoragePreparing, FinalSealRejectsChangedObservationMissingFloorAndResidencyRelabel)
{
    for (int fault = 0; fault < 5; ++fault)
    {
        auto initial = observations();
        auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
        auto a = preparing->preparationAuthority();
        auto r = root(a);
        void *token = nullptr;
        ASSERT_EQ(a.claim(a.context, &r, &token), llaminarNativeStorageSuccessV1);
        ASSERT_EQ(a.materialize(token), llaminarNativeStorageSuccessV1);
        auto snapshot = preparing->closeForPlanning();
        auto declared = NativeMandatoryStorageDeclarations::fromCreators(snapshot, creators(a));
        PhysicalMemoryPlanBuilder wrong;
        for (auto observed : snapshot.initialObservations())
        {
            if (fault == 0 && observed.device.is_cuda())
                continue;
            if (fault == 1 && observed.device.is_cpu())
                observed.admission_available_bytes++;
            if (fault == 2 && observed.device.is_cpu())
                observed.total_bytes++;
            wrong.add(observed, PhysicalMemoryOwner::LocalCollective,
                      observed.device.is_cpu() ? (fault == 3 ? 64u : 84u) : 0u,
                      fault == 4 && observed.device.is_cpu() ? 20u : 0u);
        }
        const auto before = preparing->memory()->rankAttestationSummary(true);
        auto certificate = std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(wrong.build());
        EXPECT_THROW(preparing->admit(std::move(snapshot), std::move(declared), certificate),
                     std::logic_error);
        EXPECT_TRUE(snapshot.valid());
        EXPECT_EQ(preparing->memory()->rankAttestationSummary(true), before);
        preparing->cancel();
        EXPECT_EQ(a.release(&token), llaminarNativeStorageSuccessV1);
    }
}
TEST(NativeStoragePreparing, ConcurrentCpuPreparationClaimsAndCloseShareCanonicalMutex)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        auto initial = observations();
        auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
        auto a = preparing->preparationAuthority();
        std::array<void *, 2> tokens{};
        std::array<int, 2> statuses{};
        std::barrier start(3);
        std::array<std::thread, 2> workers;
        for (int i = 0; i < 2; ++i)
            workers[i] = std::thread(
                [&, i]
                {
                    auto r = root(a);
                    r.owner_identity += i;
                    r.shared_anchor_identity += i;
                    r.requested_bytes = r.physical_bytes = 60;
                    start.arrive_and_wait();
                    statuses[i] = a.claim(a.context, &r, &tokens[i]);
                });
        start.arrive_and_wait();
        for (auto &t : workers)
            t.join();
        EXPECT_EQ(int(statuses[0] == llaminarNativeStorageSuccessV1) +
                      int(statuses[1] == llaminarNativeStorageSuccessV1),
                  1);
        EXPECT_EQ(host(preparing->memory()).preparation_committed_bytes, 60);
        for (auto &t : tokens)
            if (t)
                ASSERT_EQ(a.materialize(t), llaminarNativeStorageSuccessV1);
        auto snapshot = preparing->closeForPlanning();
        auto r = root(a);
        void *late = nullptr;
        EXPECT_EQ(a.claim(a.context, &r, &late), llaminarNativeStorageInvalidV1);
        auto gate = preparing->retirementGate();
        preparing->cancel();
        for (auto &t : tokens)
            if (t)
                EXPECT_EQ(a.release(&t), llaminarNativeStorageSuccessV1);
        gate.requireRetired();
    }
}
TEST(NativeStoragePreparing, PendingPublicationAndCancelledSourceNeverAdmitOrRevive)
{
    auto initial = observations();
    auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
    auto memory = preparing->memory();
    auto a = preparing->preparationAuthority();
    auto r = root(a);
    void *token = nullptr;
    ASSERT_EQ(a.claim(a.context, &r, &token), llaminarNativeStorageSuccessV1);
    EXPECT_THROW(preparing->closeForPlanning(), std::logic_error);
    auto gate = preparing->retirementGate();
    preparing->cancel();
    EXPECT_EQ(a.materialize(token), llaminarNativeStorageInvalidV1);
    void *other = nullptr;
    EXPECT_EQ(a.claim(a.context, &r, &other), llaminarNativeStorageInvalidV1);
    EXPECT_THROW(preparing->closeForPlanning(), std::logic_error);
    preparing.reset();
    EXPECT_FALSE(gate.retired());
    EXPECT_EQ(a.release(&token), llaminarNativeStorageSuccessV1);
    gate.requireRetired();
    EXPECT_EQ(host(memory).preparation_committed_bytes, 0);
}
TEST(NativeStoragePreparing, EveryOrdinaryAllocatorAndCertificateEntryIsPhaseGuarded)
{
    auto initial = observations();
    auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
    auto memory = preparing->memory();
    EXPECT_THROW(memory->admission(), std::logic_error);
    EXPECT_THROW(memory->contains(DeviceId::cuda(0)), std::logic_error);
    EXPECT_THROW(
        memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights, 1),
        std::logic_error);
    EXPECT_THROW(memory->adoptResidentAllocation(DeviceId::cpu(),
                                                 PhysicalMemoryOwner::PrimaryModelWeights, 1),
                 std::logic_error);
    EXPECT_THROW(
        memory->reserveNewAllocations(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights, 1),
        std::logic_error);
    EXPECT_THROW(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::LocalCollective,
                                      PhysicalMemoryMaterializationKind::NewAllocation),
                 std::logic_error);
    EXPECT_THROW(memory->committedBytes(DeviceId::cpu(), PhysicalMemoryOwner::LocalCollective,
                                        PhysicalMemoryMaterializationKind::NewAllocation),
                 std::logic_error);
    EXPECT_THROW(memory->reservedBytes(DeviceId::cpu(), PhysicalMemoryOwner::LocalCollective),
                 std::logic_error);
    EXPECT_THROW(memory->remainingAdmittedNewAllocationBytes(DeviceId::cpu(),
                                                             PhysicalMemoryOwner::LocalCollective),
                 std::logic_error);
    EXPECT_THROW(memory->rankCommitted(), std::logic_error);
    EXPECT_THROW(memory->requireRankCommitted(), std::logic_error);
    EXPECT_FALSE(memory->rankComplete());
    EXPECT_THROW(memory->requireRankComplete(), std::logic_error);
    EXPECT_NE(memory->rankAttestationSummary().find("phase=Preparing source_plan_admitted=false"),
              std::string::npos);
    auto bad = initial;
    bad[1] = bad[0];
    EXPECT_THROW(NativeStoragePreparationReplayAccess::begin(bad), std::invalid_argument);
    preparing->cancel();
    EXPECT_THROW(memory->admission(), std::logic_error);
    EXPECT_FALSE(memory->rankComplete());
}
TEST(NativeStoragePreparing, ConcurrentCloseAndClaimHaveOneCanonicalOrdering)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        auto initial = observations();
        auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
        auto a = preparing->preparationAuthority();
        std::barrier start(3);
        void *token = nullptr;
        int result = -1;
        bool closeRejected = false;
        std::optional<NativePreparationPlanningSnapshot> snapshot;
        std::thread claimant(
            [&]
            {
                auto r = root(a);
                start.arrive_and_wait();
                result = a.claim(a.context, &r, &token);
            });
        std::thread closer(
            [&]
            {
                start.arrive_and_wait();
                try
                {
                    snapshot.emplace(preparing->closeForPlanning());
                }
                catch (std::logic_error const &)
                {
                    closeRejected = true;
                }
            });
        start.arrive_and_wait();
        claimant.join();
        closer.join();
        EXPECT_NE(result == llaminarNativeStorageSuccessV1, bool(snapshot));
        EXPECT_EQ(closeRejected, result == llaminarNativeStorageSuccessV1);
        preparing->cancel();
        if (token)
            EXPECT_EQ(a.release(&token), llaminarNativeStorageSuccessV1);
        preparing->retirementGate().requireRetired();
    }
}
TEST(NativeStoragePreparing, TwoConcurrentSealAttemptsPublishOnceWithoutSnapshotOrEpochCorruption)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        auto initial = observations();
        auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
        auto a = preparing->preparationAuthority();
        auto r = root(a);
        void *token = nullptr;
        ASSERT_EQ(a.claim(a.context, &r, &token), llaminarNativeStorageSuccessV1);
        ASSERT_EQ(a.materialize(token), llaminarNativeStorageSuccessV1);
        auto first = preparing->closeForPlanning();
        auto second = NativeStoragePreparationReplayAccess::duplicate(first);
        auto d1 = NativeMandatoryStorageDeclarations::fromCreators(first, creators(a)),
             d2 = NativeMandatoryStorageDeclarations::fromCreators(second, creators(a));
        auto certificate =
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(d1.mandatoryBOM());
        std::barrier start(3);
        std::array<std::shared_ptr<NativeStoragePmaBinding>, 2> final{};
        std::array<bool, 2> rejected{};
        std::thread one(
            [&]
            {
                start.arrive_and_wait();
                try
                {
                    final[0] = preparing->admit(std::move(first), std::move(d1), certificate);
                }
                catch (std::logic_error const &)
                {
                    rejected[0] = true;
                }
            });
        std::thread two(
            [&]
            {
                start.arrive_and_wait();
                try
                {
                    final[1] = preparing->admit(std::move(second), std::move(d2), certificate);
                }
                catch (std::logic_error const &)
                {
                    rejected[1] = true;
                }
            });
        start.arrive_and_wait();
        one.join();
        two.join();
        EXPECT_EQ(int(bool(final[0])) + int(bool(final[1])), 1);
        EXPECT_EQ(int(rejected[0]) + int(rejected[1]), 1);
        EXPECT_EQ(host(preparing->memory()).committed_new_bytes, 84);
        EXPECT_EQ(host(preparing->memory()).preparation_committed_bytes, 20);
        auto winner = final[0] ? final[0] : final[1];
        auto finalAuthority = winner->authority();
        EXPECT_EQ(finalAuthority.setup_epoch, a.setup_epoch);
        auto selected = root(finalAuthority, true);
        void *selectedToken = nullptr;
        ASSERT_EQ(finalAuthority.claim(finalAuthority.context, &selected, &selectedToken),
                  llaminarNativeStorageSuccessV1);
        ASSERT_EQ(finalAuthority.materialize(selectedToken), llaminarNativeStorageSuccessV1);
        winner->sealForExecution();
        EXPECT_EQ(a.release(&token), llaminarNativeStorageSuccessV1);
        EXPECT_EQ(finalAuthority.release(&selectedToken), llaminarNativeStorageSuccessV1);
        winner->retirementGate().requireRetired();
    }
}
TEST(NativeStoragePreparing, LateContainerAliasRetainsFinalAuthorityUntilLastRetirement)
{
    auto initial = observations();
    auto preparing = NativeStoragePreparationReplayAccess::begin(initial);
    auto a = preparing->preparationAuthority();
    auto r = root(a);
    void *token = nullptr;
    ASSERT_EQ(a.claim(a.context, &r, &token), llaminarNativeStorageSuccessV1);
    ASSERT_EQ(a.materialize(token), llaminarNativeStorageSuccessV1);
    auto snapshot = preparing->closeForPlanning();
    auto declarations = creators(a);
    auto mandatory = NativeMandatoryStorageDeclarations::fromCreators(
        snapshot, std::span(declarations).first(1));
    auto certificate =
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mandatory.mandatoryBOM());
    std::weak_ptr<const PhysicalMemoryPlanAdmissionCertificate> weak = certificate;
    auto final = preparing->admit(std::move(snapshot), std::move(mandatory), certificate);
    std::optional<NativeStorageRetirementGate> gate(final->retirementGate());
    final->sealForExecution();
    preparing.reset();
    certificate.reset();
    final.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_THROW(gate->requireRetired(), std::logic_error);
    EXPECT_EQ(a.release(&token), llaminarNativeStorageSuccessV1);
    gate->requireRetired();
    gate.reset();
    EXPECT_TRUE(weak.expired());
}
