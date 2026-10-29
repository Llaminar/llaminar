/**
 * @file Test__CPUPreparedStorageAccounting.cpp
 * @brief Device-free replay of final CPU allocation admission and ownership.
 *
 * The real placement allocator proves byte rounding, preclaim rejection,
 * rollback and retained lifetime through copy, move, sharing and retirement.
 */
#include "memory/CPUWeightStoragePlacement.h"
#include <gtest/gtest.h>
using namespace llaminar2;
namespace
{
/** @brief Admit one bounded CPU prepared owner without sampling host memory. */
std::shared_ptr<PhysicalMemoryAuthority> admitted(size_t bytes, size_t staging = 0)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
        .total_bytes = bytes + staging, .admission_available_bytes = bytes + staging},
        PhysicalMemoryOwner::PrimaryModelWeights, bytes);
    if (staging)
        builder.add(PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = bytes + staging, .admission_available_bytes = bytes + staging},
            PhysicalMemoryOwner::ExpertMigrationStaging, staging);
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
}
/** @return Live exact physical bytes, with no estimator used as observation. */
size_t live(const std::shared_ptr<PhysicalMemoryAuthority> &memory)
{
    return memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights, PhysicalMemoryMaterializationKind::NewAllocation);
}
}
TEST(CPUPreparedStorageAccounting, NativeElementsUseAllocatorGeometry)
{
    auto memory = admitted(65536);
    const auto placement = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    for (size_t count : {1u, 63u, 64u, 65u, 4095u, 4096u, 4097u})
    {
        auto bytes = placement.allocate<uint8_t>(count);
        auto halves = placement.allocate<uint16_t>(count);
        auto floats = placement.allocate<float>(count);
        EXPECT_EQ(live(memory), bytes.allocationBytes() + halves.allocationBytes() + floats.allocationBytes());
        EXPECT_EQ(bytes.allocationBytes(), placement.requiredAllocationBytes<uint8_t>(count));
        EXPECT_EQ(halves.allocationBytes(), placement.requiredAllocationBytes<uint16_t>(count));
        EXPECT_EQ(floats.allocationBytes(), placement.requiredAllocationBytes<float>(count));
    }
    EXPECT_EQ(live(memory), 0u);
    EXPECT_EQ(CPUWeightStoragePlacement::local().requiredAllocationBytes<uint8_t>(1), 64u);
    EXPECT_GT(CPUWeightStoragePlacement::onNode(0).requiredAllocationBytes<uint8_t>(1), 64u);
}
TEST(CPUPreparedStorageAccounting, PreclaimAndFailedPreparationRollback)
{
    auto memory = admitted(64);
    const auto placement = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    EXPECT_THROW(placement.allocate<uint8_t>(65), std::exception);
    EXPECT_EQ(live(memory), 0u);
    try
    {
        auto unpublished = placement.allocate<uint8_t>(64);
        EXPECT_EQ(live(memory), 64u);
        throw std::runtime_error("producer failed before publication");
    }
    catch (const std::runtime_error &) {}
    EXPECT_EQ(live(memory), 0u);
}
TEST(CPUPreparedStorageAccounting, ReplaySharingCopyAndMoveRetainOneClaimPerAllocation)
{
    auto memory = admitted(128);
    const auto placement = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    for (int replay = 0; replay < 3; ++replay)
    {
        auto source = std::make_shared<AlignedVector<uint8_t>>(placement.allocate<uint8_t>(1));
        (*source)[0] = 7;
        auto alias = source;
        EXPECT_EQ(live(memory), 64u);
        {
            AlignedVector<uint8_t> copy(*source);
            EXPECT_EQ(live(memory), 128u);
            AlignedVector<uint8_t> detached(std::move(copy));
            AlignedVector<uint8_t> attached;
            attached = std::move(detached);
            EXPECT_EQ(live(memory), 128u);
            attached = {};
            EXPECT_EQ(live(memory), 64u);
        }
        source.reset();
        EXPECT_EQ(live(memory), 64u);
        alias.reset();
        EXPECT_EQ(live(memory), 0u);
    }
}

#include "tensors/Tensors.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
TEST(CPUPreparedStorageAccounting, PackedKernelDetachAttachReleaseReplay)
{
    auto memory = admitted(65536);
    const auto placement = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    std::vector<uint8_t> raw(64 * sizeof(Q8_0Block));
    for (size_t i = 0; i < 64; ++i)
    {
        auto *block = reinterpret_cast<Q8_0Block *>(raw.data() + i * sizeof(Q8_0Block));
        block->d = 0.5f;
        for (int k = 0; k < 32; ++k) block->qs[k] = static_cast<int8_t>(k);
    }
    Q8_0Tensor tensor({64, 32}, raw);
    for (int replay = 0; replay < 3; ++replay)
    {
        cpu::native_vnni::CPUNativeVNNIGemmKernel kernel(&tensor, 0, -1,
            CPUProjectionNumericalPolicy::BackendNative, placement);
        ASSERT_TRUE(kernel.isValid());
        const auto bytes = kernel.packedWeights().native_interleaved.allocationBytes();
        ASSERT_GT(bytes, 0u);
        EXPECT_EQ(live(memory), bytes);
        auto detached = kernel.detachWeights();
        EXPECT_FALSE(kernel.hasWeights());
        EXPECT_EQ(live(memory), bytes);
        ASSERT_TRUE(kernel.attachWeights(std::move(detached)));
        EXPECT_EQ(live(memory), bytes);
        kernel.releaseWeights();
        EXPECT_EQ(live(memory), 0u);
    }
}

#include "../../utils/QuantizedVerifierFormats.h"
TEST(CPUPreparedStorageAccounting, CanonicalQuantizedFormatsPublishExactPhysicalStorage)
{
    auto memory = admitted(1024 * 1024);
    const auto placement = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    uint32_t seed = 7300;
    for (const auto &format : test::quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto tensor = format.create({65u, 256u}, seed++);
        ASSERT_NE(tensor, nullptr);
        {
            cpu::native_vnni::CPUNativeVNNIGemmKernel kernel(tensor.get(), 0, -1,
                CPUProjectionNumericalPolicy::BackendNative, placement);
            ASSERT_TRUE(kernel.isValid());
            const auto &storage = kernel.packedWeights();
            EXPECT_EQ(storage.payload.allocationBytes(), 0u);
            EXPECT_EQ(storage.int8_flat.allocationBytes(), 0u);
            EXPECT_EQ(live(memory), storage.native_interleaved.allocationBytes());
            EXPECT_EQ(live(memory), placement.requiredAllocationBytes<uint8_t>(storage.native_interleaved.size()));
            auto detached = kernel.detachWeights();
            ASSERT_NE(detached, nullptr);
            EXPECT_GT(live(memory), 0u);
            ASSERT_TRUE(kernel.attachWeights(std::move(detached)));
            kernel.releaseWeights();
            EXPECT_EQ(live(memory), 0u);
        }
        EXPECT_EQ(live(memory), 0u);
    }
}

/** @test A clone cannot silently allocate outside a fully occupied owner.
 * This is a fail-closed boundary regression, not certification of migration:
 * migration must supply an explicit destination/staging owner before cloning.
 */
TEST(CPUPreparedStorageAccounting, FullyOccupiedOwnerRejectsImplicitPackedClone)
{
    auto tensor = test::quantizedVerifierFormats().front().create({64u, 256u}, 7450);
    cpu::native_vnni::CPUNativeVNNIGemmKernel reference(tensor.get());
    ASSERT_TRUE(reference.isValid());
    const auto bytes = reference.packedWeights().native_interleaved.allocationBytes();
    reference.releaseWeights();
    auto memory = admitted(bytes);
    const auto placement = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    cpu::native_vnni::CPUNativeVNNIGemmKernel kernel(tensor.get(), 0, -1,
        CPUProjectionNumericalPolicy::BackendNative, placement);
    ASSERT_TRUE(kernel.isValid());
    ASSERT_EQ(live(memory), bytes);
    EXPECT_THROW(kernel.cloneWeights(), std::exception);
    EXPECT_EQ(live(memory), bytes);
    auto detached = kernel.detachWeights();
    EXPECT_THROW(detached->clone(), std::exception);
    EXPECT_EQ(live(memory), bytes);
    ASSERT_TRUE(kernel.attachWeights(std::move(detached)));
    kernel.releaseWeights();
    EXPECT_EQ(live(memory), 0u);
}

/** @test Explicit migration destination remains valid when source owner is full. */
TEST(CPUPreparedStorageAccounting, FullPersistentOwnerClonesThroughAdmittedStaging)
{
    auto tensor = test::quantizedVerifierFormats().front().create({64u, 256u}, 7600);
    cpu::native_vnni::CPUNativeVNNIGemmKernel reference(tensor.get());
    ASSERT_TRUE(reference.isValid());
    const auto bytes = reference.packedWeights().native_interleaved.allocationBytes();
    reference.releaseWeights();
    auto memory = admitted(bytes, bytes);
    const auto persistent = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    const auto staging = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::ExpertMigrationStaging, CPUWeightStoragePlacement::local());
    for (int replay = 0; replay < 3; ++replay)
    {
        cpu::native_vnni::CPUNativeVNNIGemmKernel kernel(tensor.get(), 0, -1,
            CPUProjectionNumericalPolicy::BackendNative, persistent);
        ASSERT_TRUE(kernel.isValid());
        ASSERT_EQ(live(memory), bytes);
        auto migration = kernel.cloneWeightsTo(staging);
        ASSERT_NE(migration, nullptr);
        EXPECT_EQ(live(memory), bytes);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExpertMigrationStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), bytes);
        // Departure retires its physical bytes; the destination retains its
        // own staging claim through attachment and eventual release.
        kernel.releaseWeights();
        EXPECT_EQ(live(memory), 0u);
        ASSERT_TRUE(kernel.attachWeights(std::move(migration)));
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExpertMigrationStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), bytes);
        kernel.releaseWeights();
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExpertMigrationStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
}
TEST(CPUPreparedStorageAccounting, EmptyShrinkRetiresCapacityAndLeaseImmediately)
{
    auto memory = admitted(64);
    const auto placement = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    auto storage = placement.allocate<uint8_t>(1);
    storage.clear();
    EXPECT_EQ(live(memory), 64u);
    storage.shrink_to_fit();
    EXPECT_EQ(storage.allocationBytes(), 0u);
    EXPECT_EQ(live(memory), 0u);
}

/** @test CPU allocator ordinals belong to admission identity, never NUMA nodes. */
TEST(CPUPreparedStorageAccounting, ExactCpuAllocatorIdentitySurvivesCopies)
{
    const DeviceId cpu1{DeviceType::CPU, 1};
    PhysicalMemoryPlanBuilder builder;
    builder.add(PhysicalMemoryResource{.world_rank = 0, .device = cpu1,
        .total_bytes = 128, .admission_available_bytes = 128},
        PhysicalMemoryOwner::PrimaryModelWeights, 128);
    auto memory = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    const auto placement = CPUWeightStoragePlacement::accounted(memory, cpu1,
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    {
        auto storage = placement.allocate<uint8_t>(1);
        storage[0] = 42;
        AlignedVector<uint8_t> copy(storage);
        EXPECT_EQ(memory->claimedBytes(cpu1, PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation), 128u);
    }
    EXPECT_EQ(memory->claimedBytes(cpu1, PhysicalMemoryOwner::PrimaryModelWeights,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    EXPECT_THROW(CPUWeightStoragePlacement::accounted(memory, DeviceId::cuda(0),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local()), std::invalid_argument);
    EXPECT_THROW(CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local()), std::invalid_argument);
    EXPECT_EQ(memory->claimedBytes(cpu1, PhysicalMemoryOwner::PrimaryModelWeights,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test Empty final storage and copied cleared storage cannot escape their issuer. */
TEST(CPUPreparedStorageAccounting, EmptyFinalStorageRetainsAdmissionPolicy)
{
    auto memory = admitted(64);
    const auto placement = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
    auto empty = placement.allocate<uint8_t>(0);
    EXPECT_THROW(empty.resize_uninitialized(1), std::logic_error);
    auto storage = placement.allocate<uint8_t>(1);
    storage.clear();
    AlignedVector<uint8_t> copy(storage);
    EXPECT_EQ(live(memory), 64u);
    EXPECT_THROW(copy.resize_uninitialized(1), std::logic_error);
    storage.shrink_to_fit();
    EXPECT_EQ(live(memory), 0u);
    EXPECT_THROW(storage.resize_uninitialized(1), std::logic_error);
    EXPECT_EQ(live(memory), 0u);
}

#include "../../utils/PreparedWeightTestHarness.h"
/** @test Derived physical slices retain the exact primary/additional owner and reuse one claim. */
TEST(CPUPreparedStorageAccounting, StoreSlicePreservesExplicitOwnerAndReusesAllocation)
{
    for (const auto owner : {PhysicalMemoryOwner::PrimaryModelWeights, PhysicalMemoryOwner::AdditionalModelWeights})
    {
        auto tensor = test::quantizedVerifierFormats().front().create({128u, 256u}, 7800);
        cpu::native_vnni::CPUNativeVNNIGemmKernel sourceReference(tensor.get());
        cpu::native_vnni::CPUNativeVNNIGemmKernel sliceReference(tensor.get(), 0, 64);
        const size_t full = sourceReference.packedWeights().native_interleaved.allocationBytes();
        const size_t slice = sliceReference.packedWeights().native_interleaved.allocationBytes();
        sourceReference.releaseWeights(); sliceReference.releaseWeights();
        const DeviceId cpu1{DeviceType::CPU, 1};
        PhysicalMemoryPlanBuilder builder;
        builder.add(PhysicalMemoryResource{.world_rank = 0, .device = cpu1,
            .total_bytes = full + slice, .admission_available_bytes = full + slice}, owner, full + slice);
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        PreparedWeightStore store(ModelContextId{7820});
        store.installPhysicalMemoryAuthority(memory);
        auto binding = test::makePreparedWeightTestBinding(tensor.get(), cpu1, "blk.0.attn_q.weight", store.modelId());
        const auto ref = store.prepareGemm(binding, owner);
        ASSERT_EQ(memory->claimedBytes(cpu1, owner, PhysicalMemoryMaterializationKind::NewAllocation), full);
        auto *first = store.slicedGemmKernel(ref, 0, 64);
        ASSERT_NE(first, nullptr);
        EXPECT_EQ(memory->claimedBytes(cpu1, owner, PhysicalMemoryMaterializationKind::NewAllocation), full + slice);
        EXPECT_EQ(store.slicedGemmKernel(ref, 0, 64), first);
        EXPECT_EQ(memory->claimedBytes(cpu1, owner, PhysicalMemoryMaterializationKind::NewAllocation), full + slice);
        store.clear();
        EXPECT_EQ(memory->claimedBytes(cpu1, owner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
}
