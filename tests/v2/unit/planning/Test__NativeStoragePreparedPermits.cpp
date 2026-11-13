/**
 * @file Test__NativeStoragePreparedPermits.cpp
 * @brief Actual canonical per-descriptor Create admission, cancellation and execution seal.
 * Exact source slots share one aggregate owner quota; changed/duplicate requests
 * must not consume another selected root's bytes. No device API is involved.
 */
#include "fixtures/planning/NativeStoragePreparingReplayAccess.h"
#include <gtest/gtest.h>
#include <array>
#include <barrier>
#include <thread>
using namespace llaminar2;
namespace
{
    struct PermitSession
    {
        std::unique_ptr<NativeStoragePreparationPmaBinding> preparing;
        std::shared_ptr<NativeStoragePmaBinding> final;
        std::shared_ptr<PhysicalMemoryAuthority> memory;
        llaminarNativeStorageAuthorityV1 early{}, authority{};
        std::array<llaminarNativeRootRequestV1, 2> requests{};
        void *container = nullptr;
        std::array<void *, 2> roots{};
        explicit PermitSession(bool recording = false)
        {
            const std::array observations{PhysicalMemoryResource{0, DeviceId::cuda(0), 1000, 200},
                                          PhysicalMemoryResource{0, DeviceId::cpu(), 1000, 200}};
            preparing = llaminar2::test::NativeStoragePreparationReplayAccess::begin(observations);
            memory = preparing->memory();
            early = preparing->preparationAuthority();
            llaminarNativeRootRequestV1 metadata{1,
                                                 sizeof(llaminarNativeRootRequestV1),
                                                 0x100,
                                                 0x100,
                                                 20,
                                                 20,
                                                 4,
                                                 0,
                                                 early.setup_epoch,
                                                 llaminarNativeHostCPU0V1,
                                                 0,
                                                 -1,
                                                 llaminarNativeCommMetadataV1,
                                                 llaminarNativeCommLifetimeV1,
                                                 llaminarNativeHostAnonymousPagesV1,
                                                 llaminarNativeRetainedSetupV1,
                                                 0};
            if (early.claim(early.context, &metadata, &container) || early.materialize(container))
                throw std::logic_error("actual preparation fixture failed");
            auto snapshot = preparing->closeForPlanning();
            metadata.owner_identity = metadata.setup_epoch = 0;
            std::vector<NativeStorageCreatorDeclaration> creators{
                {NativeStorageCreatorUse::PreparationOwner, metadata}};
            for (int i = 0; i < 2; ++i)
            {
                auto request = metadata;
                request.shared_anchor_identity = 0x200 + i * 0x100;
                request.requested_bytes = request.physical_bytes = request.allocation_granularity =
                    64;
                request.owner_family = i ? llaminarNativeProfilerV1 : llaminarNativeAbortFlagV1;
                request.allocator_kind = llaminarNativeCUDAHostVmmV1;
                if (recording)
                {
                    request.allocator_kind = llaminarNativeHostAnonymousPagesV1;
                    request.owner_family = llaminarNativeCommMetadataV1;
                    request.phase = llaminarNativeGraphRecordingHostMetadataV1;
                }
                creators.push_back({NativeStorageCreatorUse::RetainedOwner, request});
                requests[i] = request;
            }
            auto declarations =
                NativeMandatoryStorageDeclarations::fromCreators(snapshot, creators);
            final = preparing->admit(std::move(snapshot), std::move(declarations),
                                     std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(
                                         declarations.mandatoryBOM()));
            authority = final->authority();
            for (int i = 0; i < 2; ++i)
            {
                requests[i].owner_identity = requests[i].shared_anchor_identity + 8;
                requests[i].setup_epoch = authority.setup_epoch;
            }
        }
        void abandon()
        {
#ifdef ORIGINAL_AGGREGATE_SOURCE
            final->sealForExecution();
#else
            final->cancelSetup();
#endif
        }
        ~PermitSession()
        {
            if (final)
                abandon();
            for (auto &root : roots)
                if (root)
                    authority.release(&root);
            if (container)
                early.release(&container);
        }
        PhysicalMemoryOwnerAttestation row() const
        {
            for (auto const &r : memory->rankAttestation())
                if (r.identity.device == DeviceId::cpu() &&
                    r.owner == PhysicalMemoryOwner::LocalCollective)
                    return r;
            throw std::logic_error("missing canonical source owner");
        }
        void publish(int i)
        {
            ASSERT_EQ(authority.claim(authority.context, &requests[i], &roots[i]),
                      llaminarNativeStorageSuccessV1);
            ASSERT_EQ(authority.materialize(roots[i]), llaminarNativeStorageSuccessV1);
        }
    };
} // namespace
TEST(NativeStoragePreparedPermits, ChangedRootWithSpareCapacityCannotSpendAnotherDeclaration)
{
    PermitSession session;
    auto changed = session.requests[0];
    changed.requested_bytes = changed.physical_bytes = 128;
    void *unexpected = nullptr;
    const auto status = session.authority.claim(session.authority.context, &changed, &unexpected);
    if (unexpected)
        session.authority.release(&unexpected);
    EXPECT_EQ(status, llaminarNativeStorageInvalidV1);
    EXPECT_EQ(session.row().native_declared_preclaimed_bytes, 0);
    session.publish(0);
    session.publish(1);
    EXPECT_EQ(session.row().materialized_new_bytes, 148);
    EXPECT_TRUE(session.memory->rankComplete());
    session.final->sealForExecution();
}
TEST(NativeStoragePreparedPermits, DuplicateRootRejectsAlthoughAggregateCapacityRemains)
{
    PermitSession session;
    ASSERT_EQ(
        session.authority.claim(session.authority.context, &session.requests[0], &session.roots[0]),
        llaminarNativeStorageSuccessV1);
    void *duplicate = nullptr;
    const auto status =
        session.authority.claim(session.authority.context, &session.requests[0], &duplicate);
    if (duplicate)
        session.authority.release(&duplicate);
    EXPECT_EQ(status, llaminarNativeStorageInvalidV1);
    ASSERT_EQ(session.authority.materialize(session.roots[0]), llaminarNativeStorageSuccessV1);
    session.publish(1);
    session.final->sealForExecution();
}
TEST(NativeStoragePreparedPermits, EveryFrozenAllocatorOwnerAndPhaseFieldIsAuthenticated)
{
    PermitSession session;
    for (int fault = 0; fault < 8; ++fault)
    {
        auto changed = session.requests[0];
        switch (fault)
        {
        case 0:
            changed.shared_anchor_identity = session.requests[1].shared_anchor_identity;
            break;
        case 1:
            changed.allocator_flags = 1;
            break;
        case 2:
            changed.owner_family = llaminarNativeP2pBackingV1;
            break;
        case 3:
            changed.lifetime = llaminarNativeSharedLifetimeV1;
            break;
        case 4:
            changed.host_numa = 2;
            break;
        case 5:
            changed.allocator_kind = llaminarNativeRegisteredHostFileV1;
            break;
        case 6:
            changed.phase = llaminarNativeColdProbeV1;
            break;
        case 7:
            changed.requested_bytes = 32;
            changed.physical_bytes = 32;
            changed.allocation_granularity = 32;
            break;
        }
        void *token = nullptr;
        EXPECT_EQ(session.authority.claim(session.authority.context, &changed, &token),
                  llaminarNativeStorageInvalidV1);
        EXPECT_EQ(token, nullptr);
        EXPECT_EQ(session.row().native_declared_preclaimed_bytes, 0);
    }
    session.publish(0);
    session.publish(1);
    session.final->sealForExecution();
}
TEST(NativeStoragePreparedPermits, FailedCreateReturnsBytesWithoutRecyclingConsumedSourceSlot)
{
    PermitSession session;
    ASSERT_EQ(
        session.authority.claim(session.authority.context, &session.requests[0], &session.roots[0]),
        llaminarNativeStorageSuccessV1);
    EXPECT_EQ(session.row().native_declared_preclaimed_bytes, 64);
    ASSERT_EQ(session.authority.release(&session.roots[0]), llaminarNativeStorageSuccessV1);
    EXPECT_EQ(session.row().native_declared_preclaimed_bytes, 0);
    void *retry = nullptr;
    EXPECT_EQ(session.authority.claim(session.authority.context, &session.requests[0], &retry),
              llaminarNativeStorageInvalidV1);
    EXPECT_EQ(retry, nullptr);
    session.publish(1);
    EXPECT_THROW(session.final->sealForExecution(), std::logic_error);
    session.abandon();
}
TEST(NativeStoragePreparedPermits, PhysicalCleanupFailureKeepsOriginalPermitAndClaim)
{
    PermitSession session;
    session.publish(0);
    session.publish(1);
    session.final->sealForExecution();
    auto gate = session.final->retirementGate();
    EXPECT_EQ(session.row().materialized_new_bytes, 148);
    EXPECT_FALSE(gate.retired());
    // A real failed native Free retains its owning token. Canonical materialized
    // bytes and descriptor consumption must remain untouched until cleanup succeeds.
    void *retry = nullptr;
    EXPECT_EQ(session.authority.claim(session.authority.context, &session.requests[0], &retry),
              llaminarNativeStorageInvalidV1);
    EXPECT_EQ(session.row().materialized_new_bytes, 148);
    ASSERT_EQ(session.authority.release(&session.roots[0]), llaminarNativeStorageSuccessV1);
    EXPECT_EQ(session.row().materialized_new_bytes, 84);
    EXPECT_FALSE(gate.retired());
}
TEST(NativeStoragePreparedPermits, MissingSelectedRootCannotPublishExecutable)
{
    PermitSession session;
    EXPECT_THROW(session.final->sealForExecution(), std::logic_error);
    session.publish(0);
    EXPECT_THROW(session.final->sealForExecution(), std::logic_error);
    session.publish(1);
    EXPECT_NO_THROW(session.final->sealForExecution());
}
TEST(NativeStoragePreparedPermits, HostRecordingPermitsRejectEarlyGrowthAndCapturedReplayGrowth)
{
    PermitSession session(true);
    void *early = nullptr;
    EXPECT_EQ(session.authority.claim(session.authority.context, &session.requests[0], &early),
              llaminarNativeStorageInvalidV1);
    session.final->beginHostRecording();
    session.final->beginHostRecording();
    session.publish(0);
    session.publish(1);
    session.final->sealForExecution();
    void *late = nullptr;
    EXPECT_EQ(session.authority.claim(session.authority.context, &session.requests[0], &late),
              llaminarNativeStorageInvalidV1);
    EXPECT_THROW(session.final->beginHostRecording(), std::logic_error);
}
TEST(NativeStoragePreparedPermits, SameAnchorContentionHasOneAtomicSourceConsumption)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        PermitSession session;
        std::barrier start(3);
        std::array<void *, 2> tokens{};
        std::array<int, 2> status{};
        std::array<std::thread, 2> workers;
        for (int i = 0; i < 2; ++i)
            workers[i] = std::thread(
                [&, i]
                {
                    start.arrive_and_wait();
                    status[i] = session.authority.claim(session.authority.context,
                                                        &session.requests[0], &tokens[i]);
                });
        start.arrive_and_wait();
        for (auto &t : workers)
            t.join();
        EXPECT_EQ(int(status[0] == llaminarNativeStorageSuccessV1) +
                      int(status[1] == llaminarNativeStorageSuccessV1),
                  1);
        EXPECT_EQ(session.row().native_declared_preclaimed_bytes, 64);
        session.roots[0] = tokens[0] ? tokens[0] : tokens[1];
        ASSERT_EQ(session.authority.materialize(session.roots[0]), llaminarNativeStorageSuccessV1);
        session.publish(1);
        session.final->sealForExecution();
    }
}

TEST(NativeStoragePreparedPermits, CancelledPublishedSessionCannotBecomeExecutable)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        PermitSession session;
        session.publish(0);
        session.publish(1);
        auto gate = session.final->retirementGate();
        session.final->cancelSetup();
        session.final->cancelSetup();
        EXPECT_THROW(session.final->sealForExecution(), std::logic_error);
        EXPECT_THROW(session.final->beginHostRecording(), std::logic_error);
        EXPECT_EQ(session.row().materialized_new_bytes, 148);
        EXPECT_FALSE(gate.retired());
        for (auto &root : session.roots)
            ASSERT_EQ(session.authority.release(&root), llaminarNativeStorageSuccessV1);
        ASSERT_EQ(session.early.release(&session.container), llaminarNativeStorageSuccessV1);
        EXPECT_TRUE(gate.retired());
        EXPECT_THROW(session.final->sealForExecution(), std::logic_error);
    }
}
TEST(NativeStoragePreparedPermits, SuccessfulExecutableSealIsIdempotentWhileRootsRemainPublished)
{
    PermitSession session;
    session.publish(0);
    session.publish(1);
    session.final->sealForExecution();
    EXPECT_NO_THROW(session.final->sealForExecution());
    EXPECT_EQ(session.row().materialized_new_bytes, 148);
    auto gate = session.final->retirementGate();
    EXPECT_FALSE(gate.retired());
    for (auto &root : session.roots)
        ASSERT_EQ(session.authority.release(&root), llaminarNativeStorageSuccessV1);
    ASSERT_EQ(session.early.release(&session.container), llaminarNativeStorageSuccessV1);
    EXPECT_TRUE(gate.retired());
    EXPECT_THROW(session.final->sealForExecution(), std::logic_error);
}
