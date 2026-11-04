/**
 * @file Test__CPUPreparedStorageAccounting.cpp
 * @brief Device-free lifecycle checks for final CPU allocation admission and ownership.
 *
 * The real placement allocator proves byte rounding, preclaim rejection,
 * rollback and retained lifetime through copy, move, sharing and retirement.
 * Floating preparation must share an already-final tensor without doubling its
 * storage or permitting source cleanup to discard live execution bytes.
 */
#include "memory/CPUWeightStoragePlacement.h"
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "tensors/TensorSlice.h"
#include <gtest/gtest.h>
using namespace llaminar2;
namespace
{
/**
 * @brief Admit exact final, explicit migration and serialized preparation owners without sampling host memory.
 * @param bytes Persistent final-root capacity, kept unchanged by preparation scratch.
 * @param staging Explicit migration destination capacity; not interchangeable with source preparation.
 * @param source_staging Code-owned bounded immediate packer workspace, retired synchronously.
 * @return Original sole PMA carrying each independent typed owner.
 */
std::shared_ptr<PhysicalMemoryAuthority> admitted(size_t bytes, size_t staging = 0, size_t source_staging = 0)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
        .total_bytes = bytes + staging + source_staging, .admission_available_bytes = bytes + staging + source_staging},
        PhysicalMemoryOwner::PrimaryModelWeights, bytes);
    if (staging)
        builder.add(PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = bytes + staging + source_staging, .admission_available_bytes = bytes + staging + source_staging},
            PhysicalMemoryOwner::ExpertMigrationStaging, staging);
    if (source_staging)
        builder.add(PhysicalMemoryResource{.world_rank=0,.device=DeviceId::cpu(),
            .total_bytes=bytes+staging+source_staging,.admission_available_bytes=bytes+staging+source_staging},
            PhysicalMemoryOwner::WeightLoadStaging,source_staging);
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
    auto memory = admitted(65536,0,CPUWeightPreparationMemory::sourceNative("Q8_0",64,32).temporary_bytes);
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
    size_t source_staging=0;
    for(const auto& format:test::quantizedVerifierFormats())
        source_staging=std::max(source_staging,CPUWeightPreparationMemory::sourceNative(
            tensorTypeName(format.tensor_type),65,256).temporary_bytes);
    auto memory = admitted(1024 * 1024,0,source_staging);
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
    auto memory = admitted(bytes,0,CPUWeightPreparationMemory::sourceNative(tensorTypeName(tensor->native_type()),64,256).temporary_bytes);
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
    auto memory = admitted(bytes,bytes,CPUWeightPreparationMemory::sourceNative(tensorTypeName(tensor->native_type()),64,256).temporary_bytes);
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
        const auto source_staging=CPUWeightPreparationMemory::sourceNative(
            tensorTypeName(tensor->native_type()),128,256).temporary_bytes;
        PhysicalMemoryPlanBuilder builder;
        builder.add(PhysicalMemoryResource{.world_rank = 0, .device = cpu1,
            .total_bytes = full + slice + source_staging, .admission_available_bytes = full + slice + source_staging}, owner, full + slice);
        builder.add(PhysicalMemoryResource{.world_rank=0,.device=cpu1,
            .total_bytes=full+slice+source_staging,.admission_available_bytes=full+slice+source_staging},
            PhysicalMemoryOwner::WeightLoadStaging,source_staging);
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

/** @test Every floating format keeps one original root through engine and binding retirement. */
TEST(CPUPreparedStorageAccounting, FinalFloatingTensorIsRetainedWithoutCopyOrSecondClaim)
{
    using Engine = gemm::FloatingPointGemmKernel;
    using Operand = FloatingActivationOperandPrecision;
    constexpr int N = 3, K = 65, M = 3;
    for (const auto format : {TensorType::FP32, TensorType::FP16, TensorType::BF16})
    for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative,
                              CPUProjectionNumericalPolicy::GPUAlignedExpert})
    {
        SCOPED_TRACE(tensorTypeName(format));
        const auto geometry = CPUWeightStoragePlacement::local();
        const size_t bytes = format == TensorType::FP32
            ? geometry.requiredAllocationBytes<float>(N * K)
            : geometry.requiredAllocationBytes<uint16_t>(N * K);
        auto memory = admitted(bytes); // Any second prepared allocation must fail.
        const auto placement = CPUWeightStoragePlacement::accounted(memory, DeviceId::cpu(),
            PhysicalMemoryOwner::PrimaryModelWeights, geometry);
        std::shared_ptr<const TensorBase> final_tensor;
        if (format == TensorType::FP32)
        {
            auto storage = placement.allocate<float>(N * K);
            for (size_t i = 0; i < N * K; ++i) storage[i] = float(int(i % 13) - 6) / 16.f;
            final_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{N, K}, std::move(storage));
        }
        else
        {
            auto storage = placement.allocate<uint16_t>(N * K);
            for (size_t i = 0; i < N * K; ++i)
            {
                const float value = float(int(i % 13) - 6) / 16.f;
                storage[i] = format == TensorType::FP16 ? fp32_to_fp16(value) : simd::fp32_to_bf16(value);
            }
            if (format == TensorType::FP16)
                final_tensor = std::make_shared<FP16Tensor>(std::vector<size_t>{N, K}, std::move(storage));
            else
                final_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{N, K}, std::move(storage));
        }
        const auto operand = format == TensorType::FP32 ? Operand::FP32
            : format == TensorType::FP16 ? Operand::FP16RNE : Operand::Native;
        ASSERT_EQ(live(memory), bytes);
        const void *const original_bytes = final_tensor->raw_data();
        std::weak_ptr<const TensorBase> original_owner = final_tensor;
        FP32Tensor input({M, K}), expected({M, N}), actual({M, N});
        for (size_t i = 0; i < M * K; ++i) input.mutable_data()[i] = float(int(i % 17) - 8) / 32.f;
        {
            Engine reference(final_tensor.get(), policy, operand);
            ASSERT_TRUE(reference.multiply_tensor(&input, &expected, M, N, K));
        }
        auto engine = Engine::fromFinalCPUStorage(final_tensor, policy, operand);
        ASSERT_EQ(engine->retainedFinalWeightTensor().get(), final_tensor.get());
        EXPECT_FALSE(engine->retainedFinalWeightTensor().owner_before(final_tensor));
        EXPECT_FALSE(final_tensor.owner_before(engine->retainedFinalWeightTensor()));
        EXPECT_FALSE(engine->canReleaseSourceWeightTensor());
        EXPECT_EQ(engine->packedWeightBytes(), 0u);
        EXPECT_EQ(engine->operandPrecision(), operand);
        EXPECT_EQ(live(memory), bytes);
        auto binding = engine->retainedFinalWeightTensor();
        auto service = std::make_shared<Engine>(std::shared_ptr<const Engine>(engine));
        final_tensor.reset();
        engine.reset();
        binding.reset();
        ASSERT_FALSE(original_owner.expired());
        EXPECT_FALSE(service->canReleaseSourceWeightTensor());
        EXPECT_EQ(service->packedWeightBytes(), 0u);
        EXPECT_EQ(live(memory), bytes);
        ContiguousFloatingPointWeightDescriptor descriptor;
        ASSERT_TRUE(service->exportContiguousFloatingPointWeights(descriptor));
        EXPECT_EQ(descriptor.data, original_bytes);
        // The original loader handle is gone. Reusing the final prepared root
        // must still execute the same serial/grouped arithmetic on dirty output.
        for (int rows : {1, 3, 1})
        {
            std::fill_n(actual.mutable_data(), M * N, -93.f);
            ASSERT_TRUE(service->multiply_tensor(&input, &actual, rows, N, K));
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), rows * N * sizeof(float)), 0);
            for (size_t i = rows * N; i < M * N; ++i) EXPECT_EQ(actual.data()[i], -93.f);
        }
        service.reset();
        EXPECT_TRUE(original_owner.expired());
        EXPECT_EQ(live(memory), 0u);
    }
}

/** @test Source views, missing storage and incompatible precision never become final roots. */
TEST(CPUPreparedStorageAccounting, FinalFloatingTensorRejectsInvalidStorageAndPolicy)
{
    using Engine = gemm::FloatingPointGemmKernel;
    using Operand = FloatingActivationOperandPrecision;
    EXPECT_THROW(Engine::fromFinalCPUStorage({}), std::invalid_argument);
    EXPECT_THROW(Engine::fromFinalCPUStorage(std::make_shared<FP32Tensor>(
        std::vector<size_t>{12})), std::invalid_argument);
    EXPECT_THROW(Engine::fromFinalCPUStorage(std::make_shared<FP32Tensor>(
        std::vector<size_t>{1, 3, 4})), std::invalid_argument);
    auto source = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 4});
    auto view = source->create_view({3, 4});
    ASSERT_NE(view, nullptr);
    EXPECT_THROW(Engine::fromFinalCPUStorage(view), std::invalid_argument);
    EXPECT_THROW(Engine::fromFinalCPUStorage(source, CPUProjectionNumericalPolicy::BackendNative,
        Operand::FP16RNE), std::invalid_argument);
    EXPECT_THROW(Engine::fromFinalCPUStorage(source, CPUProjectionNumericalPolicy::BackendNative,
        static_cast<Operand>(999)), std::invalid_argument);
    source->release_host_weight_data();
    EXPECT_THROW(Engine::fromFinalCPUStorage(source), std::invalid_argument);
    auto quantized = test::quantizedVerifierFormats().front().create({3u, 256u}, 7920);
    EXPECT_THROW(Engine::fromFinalCPUStorage(std::shared_ptr<TensorBase>(std::move(quantized))),
        std::invalid_argument);
}
