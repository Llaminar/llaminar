/**
 * @file Test__PhysicalMemoryAuthority.cpp
 * @brief Adversarial tests for topology-wide memory admission/materialization.
 */

#include "planning/PhysicalMemoryAuthority.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Build one resource observation with small readable byte counts. */
    PhysicalMemoryResource resource(
        int rank,
        DeviceId device,
        std::size_t total,
        std::size_t available)
    {
        return {
            .world_rank = rank,
            .device = device,
            .total_bytes = total,
            .admission_available_bytes = available,
        };
    }
}

TEST(PhysicalMemoryAuthority, CoalescesCpuAndGpuOwnersExactlyOnce)
{
    PhysicalMemoryBOMBuilder first_gpu(resource(
        0, DeviceId::cuda(0), 1000u, 700u));
    first_gpu.add(PhysicalMemoryOwner::PrimaryModelWeights, 200u);
    PhysicalMemoryBOMBuilder second_gpu(resource(
        0, DeviceId::cuda(0), 1000u, 700u));
    second_gpu
        .add(PhysicalMemoryOwner::KVCache, 100u)
        .add(PhysicalMemoryOwner::PrimaryModelWeights, 50u, 50u);
    PhysicalMemoryBOMBuilder cpu(resource(
        0, DeviceId::cpu(), 4000u, 3000u));
    cpu.add(PhysicalMemoryOwner::PrefixHostTier, 512u);

    PhysicalMemoryPlanBuilder builder;
    const auto plan = builder.add(first_gpu.build())
                          .add(cpu.build())
                          .add(second_gpu.build())
                          .build();

    ASSERT_EQ(plan.resources().size(), 2u);
    const auto *gpu = plan.find({0, DeviceId::cuda(0)});
    const auto *host = plan.find({0, DeviceId::cpu()});
    ASSERT_NE(gpu, nullptr);
    ASSERT_NE(host, nullptr);
    EXPECT_EQ(
        gpu->bytes(PhysicalMemoryOwner::PrimaryModelWeights), 250u);
    EXPECT_EQ(
        gpu->alreadyResidentBytes(
            PhysicalMemoryOwner::PrimaryModelWeights),
        50u);
    EXPECT_EQ(gpu->incrementalBytes(), 300u);
    EXPECT_EQ(host->totalBytes(), 512u);
    EXPECT_EQ(plan.totalBytes(), 862u);
    EXPECT_EQ(plan.incrementalBytes(), 812u);
    EXPECT_TRUE(plan.fits());
}

TEST(PhysicalMemoryAuthority, RejectsConflictingAllocatorObservations)
{
    PhysicalMemoryBOMBuilder original(resource(
        3, DeviceId::rocm(2), 1000u, 800u));
    original.add(PhysicalMemoryOwner::ActivationArena, 10u);
    PhysicalMemoryBOMBuilder stale(resource(
        3, DeviceId::rocm(2), 1000u, 799u));
    stale.add(PhysicalMemoryOwner::ExecutionWorkspace, 10u);

    PhysicalMemoryPlanBuilder builder;
    builder.add(original.build());
    EXPECT_THROW((void)builder.add(stale.build()), std::invalid_argument);
}

TEST(PhysicalMemoryAuthority,
     RetainedEnvelopeIgnoresVolatileAvailabilityAndResidencyClassification)
{
    PhysicalMemoryBOMBuilder admitted_gpu(resource(
        0, DeviceId::cuda(0), 1000u, 700u));
    admitted_gpu
        .add(PhysicalMemoryOwner::PrimaryModelWeights, 300u)
        .add(PhysicalMemoryOwner::ExecutionWorkspace, 100u)
        .add(PhysicalMemoryOwner::WeightLoadStaging, 50u);
    PhysicalMemoryPlanBuilder admitted_builder;
    const auto admitted =
        admitted_builder.add(admitted_gpu.build()).build();

    // A subsequent runner sees less free memory because the model and primary
    // workspace remain live. They are now classified as resident, while the
    // setup-only upload ring is no longer required.
    PhysicalMemoryBOMBuilder required_gpu(resource(
        0, DeviceId::cuda(0), 1000u, 300u));
    required_gpu
        .add(PhysicalMemoryOwner::PrimaryModelWeights, 300u, 300u)
        .add(PhysicalMemoryOwner::ExecutionWorkspace, 100u, 100u);
    PhysicalMemoryPlanBuilder required_builder;
    const auto required =
        required_builder.add(required_gpu.build()).build();

    EXPECT_FALSE(admitted.requiredFootprintMismatch(required).has_value());
}

TEST(PhysicalMemoryAuthority,
     RetainedEnvelopeRejectsNewResourceCapacityAndOwnerDemand)
{
    PhysicalMemoryBOMBuilder admitted_gpu(resource(
        0, DeviceId::rocm(1), 1000u, 800u));
    admitted_gpu.add(PhysicalMemoryOwner::KVCache, 100u);
    PhysicalMemoryPlanBuilder admitted_builder;
    const auto admitted =
        admitted_builder.add(admitted_gpu.build()).build();

    PhysicalMemoryBOMBuilder oversized_owner(resource(
        0, DeviceId::rocm(1), 1000u, 700u));
    oversized_owner.add(PhysicalMemoryOwner::KVCache, 101u);
    PhysicalMemoryPlanBuilder oversized_builder;
    const auto oversized =
        oversized_builder.add(oversized_owner.build()).build();
    const auto owner_mismatch =
        admitted.requiredFootprintMismatch(oversized);
    ASSERT_TRUE(owner_mismatch.has_value());
    EXPECT_NE(owner_mismatch->find("kv_cache"), std::string::npos);

    PhysicalMemoryBOMBuilder changed_capacity(resource(
        0, DeviceId::rocm(1), 999u, 700u));
    changed_capacity.add(PhysicalMemoryOwner::KVCache, 100u);
    PhysicalMemoryPlanBuilder changed_builder;
    const auto changed =
        changed_builder.add(changed_capacity.build()).build();
    const auto capacity_mismatch =
        admitted.requiredFootprintMismatch(changed);
    ASSERT_TRUE(capacity_mismatch.has_value());
    EXPECT_NE(capacity_mismatch->find("capacity changed"), std::string::npos);

    PhysicalMemoryBOMBuilder missing_cpu(resource(
        0, DeviceId::cpu(), 4000u, 3000u));
    missing_cpu.add(PhysicalMemoryOwner::PrefixHostTier, 128u);
    PhysicalMemoryPlanBuilder missing_builder;
    const auto missing =
        missing_builder.add(missing_cpu.build()).build();
    const auto resource_mismatch =
        admitted.requiredFootprintMismatch(missing);
    ASSERT_TRUE(resource_mismatch.has_value());
    EXPECT_NE(resource_mismatch->find("no admitted resource"),
              std::string::npos);
}

TEST(PhysicalMemoryAuthority, AggregateCertificateFailsOnAnyResource)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::cpu(), 1000u, 900u),
        PhysicalMemoryOwner::ModelSourcePayload,
        100u);
    builder.add(
        resource(1, DeviceId::rocm(0), 1000u, 50u),
        PhysicalMemoryOwner::RoutedExpertWeights,
        51u);

    EXPECT_THROW(
        PhysicalMemoryPlanAdmissionCertificate(builder.build()),
        std::invalid_argument);
}

TEST(PhysicalMemoryAuthority, LiveLedgerSeparatesNewAndRetainedBytes)
{
    PhysicalMemoryBOMBuilder gpu(resource(
        0, DeviceId::cuda(1), 1000u, 600u));
    gpu.add(PhysicalMemoryOwner::PrimaryModelWeights, 300u, 100u)
        .add(PhysicalMemoryOwner::KVCache, 200u);
    PhysicalMemoryPlanBuilder builder;
    builder.add(gpu.build());
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryMaterializationLedger ledger(admission);
    EXPECT_EQ(ledger.admission().get(), admission.get());
    const PhysicalMemoryAllocatorIdentity identity{0, DeviceId::cuda(1)};

    auto retained = ledger.adoptResidentAllocation(
        identity, PhysicalMemoryOwner::PrimaryModelWeights, 100u);
    auto weights_a = ledger.claimNewAllocation(
        identity, PhysicalMemoryOwner::PrimaryModelWeights, 75u);
    auto weights_b = ledger.claimNewAllocation(
        identity, PhysicalMemoryOwner::PrimaryModelWeights, 125u);
    auto cache = ledger.claimNewAllocation(
        identity, PhysicalMemoryOwner::KVCache, 200u);

    EXPECT_TRUE(ledger.complete());
    EXPECT_NO_THROW(ledger.requireComplete());
    EXPECT_THROW(
        (void)ledger.claimNewAllocation(
            identity, PhysicalMemoryOwner::KVCache, 1u),
        std::logic_error);
    EXPECT_THROW(
        (void)ledger.adoptResidentAllocation(
            identity, PhysicalMemoryOwner::PrimaryModelWeights, 1u),
        std::logic_error);

    cache = {};
    EXPECT_FALSE(ledger.complete());
    EXPECT_THROW(ledger.requireComplete(), std::logic_error);
    EXPECT_EQ(
        ledger.claimedBytes(
            identity,
            PhysicalMemoryOwner::KVCache,
            PhysicalMemoryMaterializationKind::NewAllocation),
        0u);
}

TEST(PhysicalMemoryAuthority, MovedLeaseReleasesItsClaimExactlyOnce)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::cpu(), 1024u, 1024u),
        PhysicalMemoryOwner::ActivationTransportStaging,
        128u);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryMaterializationLedger ledger(admission);
    const PhysicalMemoryAllocatorIdentity identity{0, DeviceId::cpu()};

    auto first = ledger.claimNewAllocation(
        identity, PhysicalMemoryOwner::ActivationTransportStaging, 128u);
    auto moved = std::move(first);
    EXPECT_FALSE(first.valid());
    EXPECT_TRUE(moved.valid());
    EXPECT_TRUE(ledger.complete());
    moved = {};
    EXPECT_EQ(
        ledger.claimedBytes(
            identity,
            PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),
        0u);
}

TEST(PhysicalMemoryAuthority, LazyPoolSeparatesCapacityCommitmentFromMaterialization)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::cuda(0), 1024u, 1024u),
        PhysicalMemoryOwner::PrefixDeviceTier,
        128u);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryMaterializationLedger ledger(admission);
    const PhysicalMemoryAllocatorIdentity identity{0, DeviceId::cuda(0)};

    auto pool = ledger.reserveNewAllocations(
        identity, PhysicalMemoryOwner::PrefixDeviceTier, 128u);
    EXPECT_TRUE(pool.valid());
    EXPECT_EQ(pool.capacityBytes(), 128u);
    EXPECT_EQ(pool.materializedBytes(), 0u);
    EXPECT_EQ(pool.remainingBytes(), 128u);
    EXPECT_EQ(
        ledger.reservedBytes(
            identity, PhysicalMemoryOwner::PrefixDeviceTier),
        128u);
    EXPECT_EQ(
        ledger.committedBytes(
            identity,
            PhysicalMemoryOwner::PrefixDeviceTier,
            PhysicalMemoryMaterializationKind::NewAllocation),
        128u);
    EXPECT_EQ(
        ledger.remainingAdmittedNewAllocationBytes(
            identity, PhysicalMemoryOwner::PrefixDeviceTier),
        0u);
    EXPECT_TRUE(ledger.committed());
    EXPECT_NO_THROW(ledger.requireCommitted());
    EXPECT_FALSE(ledger.complete())
        << "Reserved capacity is not evidence that its backing bytes exist";

    auto first = pool.claimAllocation(48u);
    auto second = pool.claimAllocation(80u);
    EXPECT_EQ(pool.materializedBytes(), 128u);
    EXPECT_EQ(pool.remainingBytes(), 0u);
    EXPECT_EQ(
        ledger.claimedBytes(
            identity,
            PhysicalMemoryOwner::PrefixDeviceTier,
            PhysicalMemoryMaterializationKind::NewAllocation),
        128u);
    EXPECT_TRUE(ledger.complete());
    EXPECT_THROW((void)pool.claimAllocation(1u), std::logic_error);
    EXPECT_THROW(
        (void)ledger.claimNewAllocation(
            identity, PhysicalMemoryOwner::PrefixDeviceTier, 1u),
        std::logic_error)
        << "A direct allocation cannot consume capacity owned by a pool";

    second = {};
    EXPECT_FALSE(ledger.complete());
    EXPECT_TRUE(ledger.committed());
    EXPECT_EQ(pool.materializedBytes(), 48u);
}

TEST(PhysicalMemoryAuthority,
     RemainingCapacityTracksDirectClaimsAndReservationsWithoutTelemetry)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::cuda(0), 1024u, 1024u),
        PhysicalMemoryOwner::ExecutionWorkspace,
        256u);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryAuthority authority(admission, 0);

    EXPECT_EQ(
        authority.remainingAdmittedNewAllocationBytes(
            DeviceId::cuda(0),
            PhysicalMemoryOwner::ExecutionWorkspace),
        256u);
    auto direct = authority.claimNewAllocation(
        DeviceId::cuda(0),
        PhysicalMemoryOwner::ExecutionWorkspace,
        64u);
    auto pool = authority.reserveNewAllocations(
        DeviceId::cuda(0),
        PhysicalMemoryOwner::ExecutionWorkspace,
        96u);
    EXPECT_EQ(
        authority.remainingAdmittedNewAllocationBytes(
            DeviceId::cuda(0),
            PhysicalMemoryOwner::ExecutionWorkspace),
        96u);

    direct = {};
    EXPECT_EQ(
        authority.remainingAdmittedNewAllocationBytes(
            DeviceId::cuda(0),
            PhysicalMemoryOwner::ExecutionWorkspace),
        160u);
    pool = {};
    EXPECT_EQ(
        authority.remainingAdmittedNewAllocationBytes(
            DeviceId::cuda(0),
            PhysicalMemoryOwner::ExecutionWorkspace),
        256u);
}

TEST(PhysicalMemoryAuthority, ChildLeaseRetainsReservationAfterPoolHandleCloses)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::cpu(), 1024u, 1024u),
        PhysicalMemoryOwner::PrefixHostTier,
        96u);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryMaterializationLedger ledger(admission);
    const PhysicalMemoryAllocatorIdentity identity{0, DeviceId::cpu()};
    PhysicalMemorySuballocationLease surviving_child;

    {
        auto pool = ledger.reserveNewAllocations(
            identity, PhysicalMemoryOwner::PrefixHostTier, 96u);
        surviving_child = pool.claimAllocation(32u);
        EXPECT_EQ(
            ledger.reservedBytes(
                identity, PhysicalMemoryOwner::PrefixHostTier),
            96u);
    }

    EXPECT_EQ(
        ledger.reservedBytes(
            identity, PhysicalMemoryOwner::PrefixHostTier),
        96u)
        << "A live allocation must retain its parent capacity authority";
    EXPECT_EQ(
        ledger.claimedBytes(
            identity,
            PhysicalMemoryOwner::PrefixHostTier,
            PhysicalMemoryMaterializationKind::NewAllocation),
        32u);

    surviving_child = {};
    EXPECT_EQ(
        ledger.reservedBytes(
            identity, PhysicalMemoryOwner::PrefixHostTier),
        0u);
    EXPECT_EQ(
        ledger.claimedBytes(
            identity,
            PhysicalMemoryOwner::PrefixHostTier,
            PhysicalMemoryMaterializationKind::NewAllocation),
        0u);
}

TEST(PhysicalMemoryAuthority, IndependentReservationsShareOneOwnerLineSafely)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::rocm(0), 2048u, 2048u),
        PhysicalMemoryOwner::ExecutionWorkspace,
        128u);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryMaterializationLedger ledger(admission);
    const PhysicalMemoryAllocatorIdentity identity{0, DeviceId::rocm(0)};

    auto first_pool = ledger.reserveNewAllocations(
        identity, PhysicalMemoryOwner::ExecutionWorkspace, 64u);
    auto second_pool = ledger.reserveNewAllocations(
        identity, PhysicalMemoryOwner::ExecutionWorkspace, 64u);
    auto first_child = first_pool.claimAllocation(32u);
    auto second_child = second_pool.claimAllocation(32u);
    EXPECT_EQ(
        ledger.reservedBytes(
            identity, PhysicalMemoryOwner::ExecutionWorkspace),
        128u);
    EXPECT_EQ(
        ledger.claimedBytes(
            identity,
            PhysicalMemoryOwner::ExecutionWorkspace,
            PhysicalMemoryMaterializationKind::NewAllocation),
        64u);

    first_pool = {};
    EXPECT_EQ(
        ledger.reservedBytes(
            identity, PhysicalMemoryOwner::ExecutionWorkspace),
        128u);
    first_child = {};
    EXPECT_EQ(
        ledger.reservedBytes(
            identity, PhysicalMemoryOwner::ExecutionWorkspace),
        64u)
        << "Only the closed reservation may retire when its last child dies";
    EXPECT_EQ(second_pool.materializedBytes(), 32u);
}

TEST(PhysicalMemoryAuthority, ReservationSupportsConcurrentSetupWorkers)
{
    constexpr std::size_t kWorkers = 16u;
    constexpr std::size_t kBytesPerWorker = 16u;
    constexpr std::size_t kCapacity = kWorkers * kBytesPerWorker;
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::cuda(1), 4096u, 4096u),
        PhysicalMemoryOwner::ActivationArena,
        kCapacity);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryMaterializationLedger ledger(admission);
    const PhysicalMemoryAllocatorIdentity identity{0, DeviceId::cuda(1)};
    auto pool = ledger.reserveNewAllocations(
        identity, PhysicalMemoryOwner::ActivationArena, kCapacity);

    std::vector<PhysicalMemorySuballocationLease> leases(kWorkers);
    std::vector<std::thread> workers;
    workers.reserve(kWorkers);
    std::atomic<std::size_t> failures{0u};
    for (std::size_t index = 0; index < kWorkers; ++index)
    {
        workers.emplace_back(
            [&, index]
            {
                try
                {
                    leases[index] =
                        pool.claimAllocation(kBytesPerWorker);
                }
                catch (...)
                {
                    failures.fetch_add(1u, std::memory_order_relaxed);
                }
            });
    }
    for (auto &worker : workers)
        worker.join();

    EXPECT_EQ(failures.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(pool.materializedBytes(), kCapacity);
    EXPECT_EQ(
        ledger.claimedBytes(
            identity,
            PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation),
        kCapacity);
    EXPECT_TRUE(ledger.complete());
}

TEST(PhysicalMemoryAuthority, RankBindingRejectsRemoteClaimsAndTracksLocalCompletion)
{
    PhysicalMemoryPlanBuilder builder;
    builder
        .add(
            resource(0, DeviceId::cuda(0), 2048u, 2048u),
            PhysicalMemoryOwner::PrimaryModelWeights,
            256u)
        .add(
            resource(0, DeviceId::cpu(), 4096u, 4096u),
            PhysicalMemoryOwner::WeightLoadStaging,
            64u)
        .add(
            resource(1, DeviceId::rocm(0), 2048u, 2048u),
            PhysicalMemoryOwner::RoutedExpertWeights,
            512u);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryAuthority authority(admission, 0);

    EXPECT_EQ(authority.admission().get(), admission.get());
    EXPECT_TRUE(authority.contains(DeviceId::cuda(0)));
    EXPECT_TRUE(authority.contains(DeviceId::cpu()));
    EXPECT_FALSE(authority.contains(DeviceId::rocm(0)))
        << "A device ordinal on another rank is not a local allocator";
    EXPECT_EQ(
        authority.plannedBytes(
            DeviceId::cuda(0),
            PhysicalMemoryOwner::PrimaryModelWeights),
        256u);
    EXPECT_THROW(
        (void)authority.claimNewAllocation(
            DeviceId::rocm(0),
            PhysicalMemoryOwner::RoutedExpertWeights,
            512u),
        std::out_of_range);

    auto weights = authority.claimNewAllocation(
        DeviceId::cuda(0),
        PhysicalMemoryOwner::PrimaryModelWeights,
        256u);
    EXPECT_FALSE(authority.rankComplete());
    auto host_staging = authority.claimNewAllocation(
        DeviceId::cpu(),
        PhysicalMemoryOwner::WeightLoadStaging,
        64u);
    EXPECT_TRUE(authority.rankComplete())
        << "Remote rank-one owners are outside this process's completion edge";
    EXPECT_NO_THROW(authority.requireRankComplete());
}

TEST(PhysicalMemoryAuthority, RankBindingRequiresAnOwnedAllocator)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::cuda(0), 1024u, 1024u),
        PhysicalMemoryOwner::ActivationArena,
        32u);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());

    EXPECT_THROW(
        (void)PhysicalMemoryAuthority(admission, 1),
        std::invalid_argument);
    EXPECT_THROW(
        (void)PhysicalMemoryAuthority(admission, -1),
        std::invalid_argument);
}

TEST(PhysicalMemoryAuthority,
     RankAttestationIsAtomicTypedAndExcludesRemoteResources)
{
    PhysicalMemoryPlanBuilder builder;
    builder
        .add(
            resource(0, DeviceId::rocm(0), 4096u, 4096u),
            PhysicalMemoryOwner::ExecutionWorkspace,
            128u)
        .add(
            resource(0, DeviceId::rocm(0), 4096u, 4096u),
            PhysicalMemoryOwner::PrimaryModelWeights,
            64u,
            16u)
        .add(
            resource(1, DeviceId::cuda(0), 4096u, 4096u),
            PhysicalMemoryOwner::RoutedExpertWeights,
            256u);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryAuthority authority(admission, 0);

    auto workspace = authority.reserveNewAllocations(
        DeviceId::rocm(0),
        PhysicalMemoryOwner::ExecutionWorkspace,
        128u);
    auto workspace_child = workspace.claimAllocation(32u);
    auto retained = authority.adoptResidentAllocation(
        DeviceId::rocm(0),
        PhysicalMemoryOwner::PrimaryModelWeights,
        16u);
    auto new_weights = authority.claimNewAllocation(
        DeviceId::rocm(0),
        PhysicalMemoryOwner::PrimaryModelWeights,
        48u);

    const auto rows = authority.rankAttestation();
    EXPECT_EQ(
        rows.size(),
        PhysicalMemoryBOM::ownerCount())
        << "Only rank-zero's one physical resource belongs in this view";
    const auto find_owner = [&](PhysicalMemoryOwner owner)
        -> const PhysicalMemoryOwnerAttestation &
    {
        const auto found = std::find_if(
            rows.begin(),
            rows.end(),
            [owner](const auto &row) { return row.owner == owner; });
        EXPECT_NE(found, rows.end());
        return *found;
    };

    const auto &workspace_row = find_owner(
        PhysicalMemoryOwner::ExecutionWorkspace);
    EXPECT_EQ(workspace_row.planned_new_bytes, 128u);
    EXPECT_EQ(workspace_row.committed_new_bytes, 128u);
    EXPECT_EQ(workspace_row.materialized_new_bytes, 32u);
    EXPECT_TRUE(workspace_row.committed());
    EXPECT_FALSE(workspace_row.complete());

    const auto &weight_row = find_owner(
        PhysicalMemoryOwner::PrimaryModelWeights);
    EXPECT_EQ(weight_row.planned_new_bytes, 48u);
    EXPECT_EQ(weight_row.materialized_new_bytes, 48u);
    EXPECT_EQ(weight_row.planned_resident_bytes, 16u);
    EXPECT_EQ(weight_row.adopted_resident_bytes, 16u);
    EXPECT_TRUE(weight_row.complete());

    const std::string summary = authority.rankAttestationSummary();
    EXPECT_NE(summary.find("owner=execution_workspace"),
              std::string::npos);
    EXPECT_NE(summary.find("committed_new=128"), std::string::npos);
    EXPECT_EQ(summary.find("routed_expert_weights"), std::string::npos)
        << "A remote resource must never leak into rank-local attestation";
}

/**
 * @brief Opaque driver pools commit family capacity without fake allocations.
 *
 * CUDA/HIP expose no independently releasable pointer for native graph slabs.
 * A model runner therefore retains one owner reservation for the complete
 * graph family. Treating each observed pool-growth event as a standalone
 * allocation would both double-account the bytes and make sibling graphs with
 * zero observed growth appear unmaterialized.
 */
TEST(PhysicalMemoryAuthority,
     NativeGraphDriverPoolUsesOneCompleteFamilyReservation)
{
    constexpr std::size_t kFamilyBytes = 96u;
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::cuda(0), 1024u, 1024u),
        PhysicalMemoryOwner::NativeGraphExecutable,
        kFamilyBytes);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryAuthority authority(admission, 0);

    auto graph_pool = authority.reserveNewAllocations(
        DeviceId::cuda(0),
        PhysicalMemoryOwner::NativeGraphExecutable,
        kFamilyBytes);

    EXPECT_TRUE(graph_pool.valid());
    EXPECT_EQ(graph_pool.capacityBytes(), kFamilyBytes);
    EXPECT_EQ(graph_pool.materializedBytes(), 0u);
    EXPECT_EQ(
        authority.committedBytes(
            DeviceId::cuda(0),
            PhysicalMemoryOwner::NativeGraphExecutable,
            PhysicalMemoryMaterializationKind::NewAllocation),
        kFamilyBytes);
    EXPECT_EQ(
        authority.claimedBytes(
            DeviceId::cuda(0),
            PhysicalMemoryOwner::NativeGraphExecutable,
            PhysicalMemoryMaterializationKind::NewAllocation),
        0u);
    EXPECT_TRUE(authority.rankCommitted());
    EXPECT_FALSE(authority.rankComplete());
}

/**
 * @brief Snapshot arenas materialize only inside their admitted owner pool.
 *
 * The failed production sequence historically reached hipMalloc with no free
 * bytes because snapshot storage was absent from admission. This test proves
 * both sides of the replacement contract: several retained graph families may
 * safely suballocate one committed envelope, and an oversized family is
 * rejected synchronously by the canonical ledger.
 */
TEST(PhysicalMemoryAuthority,
     GraphSnapshotArenaReservationBoundsEveryRetainedFamily)
{
    constexpr std::size_t kEnvelopeBytes = 128u;
    PhysicalMemoryPlanBuilder builder;
    builder.add(
        resource(0, DeviceId::rocm(0), 1024u, 1024u),
        PhysicalMemoryOwner::GraphSnapshotArena,
        kEnvelopeBytes);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
    PhysicalMemoryAuthority authority(admission, 0);

    auto snapshot_pool = authority.reserveNewAllocations(
        DeviceId::rocm(0),
        PhysicalMemoryOwner::GraphSnapshotArena,
        kEnvelopeBytes);
    EXPECT_EQ(snapshot_pool.capacityBytes(), kEnvelopeBytes);
    EXPECT_TRUE(authority.rankCommitted());

    auto prefill_arena = snapshot_pool.claimAllocation(96u);
    auto decode_arena = snapshot_pool.claimAllocation(32u);
    EXPECT_TRUE(prefill_arena.valid());
    EXPECT_TRUE(decode_arena.valid());
    EXPECT_EQ(snapshot_pool.materializedBytes(), kEnvelopeBytes);
    EXPECT_EQ(snapshot_pool.remainingBytes(), 0u);
    EXPECT_THROW(
        (void)snapshot_pool.claimAllocation(1u),
        std::logic_error);

    const auto rows = authority.rankAttestation();
    const auto found = std::find_if(
        rows.begin(),
        rows.end(),
        [](const PhysicalMemoryOwnerAttestation &row)
        {
            return row.owner ==
                PhysicalMemoryOwner::GraphSnapshotArena;
        });
    ASSERT_NE(found, rows.end());
    EXPECT_EQ(found->planned_new_bytes, kEnvelopeBytes);
    EXPECT_EQ(found->committed_new_bytes, kEnvelopeBytes);
    EXPECT_EQ(found->materialized_new_bytes, kEnvelopeBytes);
    EXPECT_TRUE(found->complete());
}
