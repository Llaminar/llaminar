/**
 * @file Test__PreparedEmbeddingWeightsLifecycle.cpp
 * @brief Unit tests for model-owned PreparedEmbeddingWeights lifecycle
 *
 * Tests verify:
 * 1. PreparedWeightStore owns embedding handles and can release them
 * 2. FP32/FP16/BF16 tensors and nested slices reject packed preparation
 * 3. Q8_0 and Q4_0 tensors get proper native source copy + handle metadata
 * 4. Prepared refs and tensor lookups resolve existing handles without creating entries
 * 5. Different bindings get separate embedding entries
 * 6. Null tensor input throws
 * 7. Move semantics are safe (no double-free)
 * 8. Final CPU storage is shared with lookup without a second allocation, and
 *    its original physical-memory owner survives source/window retirement.
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <string>
#include <utility>

#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "execution/compute_stages/stages/EmbeddingStage.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/NativeEmbeddingLayout.h"
#include "kernels/common/PreparedEmbeddingWeights.h"
#include "kernels/cpu/ops/CPUEmbeddingKernelT.h"
#include "loaders/PreparedWeightStore.h"
#include "memory/CPUStorageRoot.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "utils/DebugEnv.h"
#include "../../utils/TestTensorFactory.h"
#include "../../utils/EmbeddingVerifierFormats.h"
#include "../../mocks/MockMPIContext.h"

using namespace llaminar::v2::kernels;
using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__PreparedEmbeddingWeightsLifecycle : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
            dm.initialize(-1);

        if (!getCPUBackend())
            initCPUBackend(0);

        KernelFactory::clearCache();
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
    }

    WeightBinding makeEmbeddingBinding(uint64_t binding_id, TensorBase *tensor,
                                       DeviceId device = DeviceId::cpu(),
                                       const std::string &name = "token_embd.weight") const
    {
        WeightBinding binding;
        binding.binding_id = binding_id;
        binding.identity.model_id = kModelId;
        binding.identity.logical_id = binding_id;
        binding.identity.instance_id = binding_id;
        binding.identity.canonical_name = name;
        binding.identity.role = WeightRole::Embedding;
        binding.identity.derivation = WeightDerivationKind::Source;
        binding.residency.home_device = device;
        binding.residency.resident_device = device;
        binding.tensor = tensor;
        return binding;
    }

    PreparedWeightRef prepareEmbedding(PreparedWeightStore &store, uint64_t binding_id,
                                       TensorBase *tensor, size_t vocab_offset = 0,
                                       size_t total_vocab = 0) const
    {
        auto binding = makeEmbeddingBinding(binding_id, tensor);
        return store.prepareEmbedding(binding, kDModelInt, vocab_offset, total_vocab);
    }

    static constexpr ModelContextId kModelId{77};
    static constexpr size_t kVocabSize = 128;
    static constexpr size_t kDModel = 64;
    static constexpr int kDModelInt = 64;
};

// ============================================================================
// Store State Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, StoreStartsWithoutEmbeddingEntries)
{
    PreparedWeightStore store(kModelId);
    PreparedWeightRef missing{kModelId, 1, PreparedWeightKind::PreparedEmbedding, DeviceId::cpu()};

    EXPECT_EQ(store.embeddingHandle(missing), nullptr);
    EXPECT_FALSE(store.preparedRefForBinding(missing.binding_id, missing.device).has_value());
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, ClearReleasesEmbeddingEntries)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto ref = prepareEmbedding(store, 1, tensor.get());

    ASSERT_TRUE(store.contains(ref));
    ASSERT_NE(store.embeddingHandle(ref), nullptr);

    store.clear();

    EXPECT_FALSE(store.contains(ref));
    EXPECT_EQ(store.embeddingHandle(ref), nullptr);
}

// ============================================================================
// FP32 Rejection Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, FP32TensorRejectsPackedPreparation)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createFP32Random({kVocabSize, kDModel});
    auto binding = makeEmbeddingBinding(1, tensor.get());

    EXPECT_THROW(
        store.prepareEmbedding(binding, kDModelInt),
        std::runtime_error);
    PreparedWeightRef rejected{kModelId, binding.binding_id, PreparedWeightKind::PreparedEmbedding, DeviceId::cpu()};
    EXPECT_EQ(store.embeddingHandle(rejected), nullptr);
}

/** @test Wrapping storage cannot change its native embedding representation. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, NestedSlicesPreserveEveryFormatCapability)
{
    for (const auto &format : embeddingVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        // K=256 covers every superblock codebook, including Q8_K, which has
        // no NativeQuantised metadata but does have complete native unpack support.
        std::unique_ptr<TensorBase> tensor = format.create({3, 256}, 481);
        for (int nesting = 0; nesting != 3; ++nesting)
        {
            SCOPED_TRACE(nesting);
            EXPECT_EQ(IINT8Unpackable::fromTensor(tensor.get()) != nullptr,
                      format.prepared_native_quantized);
            if (!format.prepared_native_quantized)
            {
                for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
                {
                    // Validation is device-free: no GPU is initialized. It
                    // must not demand an native quantized handle for native storage.
                    EmbeddingStage::Params params;
                    params.device_id = device;
                    params.embed_table = tensor.get();
                    EmbeddingStage stage(params);
                    std::string error;
                    EXPECT_TRUE(stage.validatePreparedWeights(&error)) << error;
                }
                EXPECT_EQ(KernelFactory::prepareEmbeddingHandleLocal(
                              tensor.get(), 256, DeviceId::cpu(), nullptr), nullptr);
                if (const auto *slice = dynamic_cast<const TensorSlice *>(tensor.get()))
                {
                    std::array<int8_t, 32> output;
                    output.fill(91);
                    EXPECT_THROW(slice->unpack_block_to_int8(0, 0, output.data()),
                                 std::logic_error);
                    EXPECT_TRUE(std::all_of(output.begin(), output.end(),
                                            [](int8_t value) { return value == 91; }));
                }
            }
            else
            {
                const auto layout = NativeEmbeddingLayout::forSource(tensor->native_type(), tensor->rows(), 256);
                EXPECT_EQ(layout.byte_size, tensor->size_bytes());
            }
            tensor = std::make_unique<TensorSlice>(std::move(tensor), SliceMetadata{});
        }
    }
}

// ============================================================================
// Quantized Tensor Preparation Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, Q8_0TensorGetsPreparation)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto ref = prepareEmbedding(store, 1, tensor.get());
    auto *handle = store.embeddingHandle(ref);

    ASSERT_NE(handle, nullptr);
    EXPECT_TRUE(store.contains(ref));
    EXPECT_EQ(handle->tensor, tensor.get());
    EXPECT_EQ(handle->device_id, DeviceId::cpu());
    ASSERT_NE(handle->weights, nullptr);
    EXPECT_NE(handle->weights->device_data, nullptr);
    EXPECT_GT(handle->weights->layout.byte_size, 0u);
    EXPECT_EQ(handle->weights->vocab_size, kVocabSize);
    EXPECT_EQ(handle->weights->d_model, kDModelInt);
    EXPECT_GT(handle->weights->layout.blocks_per_row, 0u);
}

TEST_F(
    Test__PreparedEmbeddingWeightsLifecycle,
    AdditionalEmbeddingClaimsAndReleasesItsExactOwnerLine)
{
    const size_t bytes = NativeEmbeddingLayout::forSource(
        TensorType::Q8_0, kVocabSize, kDModelInt).byte_size;
    PhysicalMemoryBOMBuilder bom({
        .world_rank = 0,
        .device = DeviceId::cpu(),
        .total_bytes = 1024u * 1024u,
        .admission_available_bytes = 1024u * 1024u,
    });
    bom.add(PhysicalMemoryOwner::AdditionalModelWeights, bytes);
    PhysicalMemoryPlanBuilder plan;
    plan.add(bom.build());
    auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(plan.build());
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::move(admission),
        0);

    PreparedWeightStore store(kModelId);
    store.installPhysicalMemoryAuthority(authority);
    auto tensor =
        TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    const auto binding = makeEmbeddingBinding(1, tensor.get());
    const auto ref = store.prepareEmbedding(
        binding,
        kDModelInt,
        /*vocab_offset=*/0,
        /*total_vocab=*/0,
        PhysicalMemoryOwner::AdditionalModelWeights);

    ASSERT_NE(store.embeddingHandle(ref), nullptr);
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::AdditionalModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation),
        bytes);
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation),
        0u);

    store.clear();
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::AdditionalModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation),
        0u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, Q4_0TensorGetsPreparation)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ4_0Random({kVocabSize, kDModel});
    auto ref = prepareEmbedding(store, 1, tensor.get());
    auto *handle = store.embeddingHandle(ref);

    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->weights, nullptr);
    EXPECT_NE(handle->weights->device_data, nullptr);
    EXPECT_EQ(handle->weights->vocab_size, kVocabSize);
    EXPECT_EQ(handle->weights->d_model, kDModelInt);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, HandleMetadataCorrect)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto ref = prepareEmbedding(store, 1, tensor.get());
    auto *handle = store.embeddingHandle(ref);

    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->weights, nullptr);

    const size_t expected_blocks_per_row = (kDModel + 31) / 32;
    EXPECT_EQ(handle->weights->layout.blocks_per_row, expected_blocks_per_row);

    const size_t expected_byte_size = kVocabSize * expected_blocks_per_row * sizeof(Q8_0Block);
    EXPECT_EQ(handle->weights->layout.byte_size, expected_byte_size);
}

/** @test CPU source copies preserve bytes independently of the GPU setup cap. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, CPUEmbeddingSourceBytesIgnoreGPUStagingPolicy)
{
    struct ConfigRestore
    {
        int &value;
        int original;
        ~ConfigRestore() { value = original; }
    } restore{mutableDebugEnv().rocm.repack_budget_mb,
              mutableDebugEnv().rocm.repack_budget_mb};

    mutableDebugEnv().rocm.repack_budget_mb = 1;
    constexpr size_t large_vocab = 65536;
    auto tensor = TestTensorFactory::createQ4_0Random({large_vocab, kDModel});
    const size_t expected_bytes = tensor->size_bytes();

    PreparedWeightStore store(kModelId);
    auto ref = prepareEmbedding(store, 1, tensor.get());
    auto *handle = store.embeddingHandle(ref);

    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->weights, nullptr);
    ASSERT_NE(handle->weights->device_data, nullptr);
    ASSERT_GT(expected_bytes, 1024u * 1024u);
    ASSERT_EQ(handle->weights->layout.byte_size, expected_bytes);
    EXPECT_EQ(std::memcmp(handle->weights->device_data,
                          tensor->raw_data(), expected_bytes),
              0);
}

// ============================================================================
// Lookup Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, RefLookupReturnsSameHandle)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto ref = prepareEmbedding(store, 1, tensor.get());

    auto *handle1 = store.embeddingHandle(ref);
    auto *handle2 = store.embeddingHandle(ref);

    ASSERT_NE(handle1, nullptr);
    EXPECT_EQ(handle1, handle2);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, DifferentBindingsGetSeparateEntries)
{
    PreparedWeightStore store(kModelId);
    auto tensor_a = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel}, 42);
    auto tensor_b = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel}, 99);

    auto ref_a = prepareEmbedding(store, 1, tensor_a.get());
    auto ref_b = prepareEmbedding(store, 2, tensor_b.get());
    auto *handle_a = store.embeddingHandle(ref_a);
    auto *handle_b = store.embeddingHandle(ref_b);

    ASSERT_NE(handle_a, nullptr);
    ASSERT_NE(handle_b, nullptr);
    EXPECT_NE(handle_a, handle_b);
    EXPECT_NE(ref_a.binding_id, ref_b.binding_id);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, LookupOnlyDoesNotCreate)
{
    PreparedWeightStore store(kModelId);
    PreparedWeightRef missing{kModelId, 1, PreparedWeightKind::PreparedEmbedding, DeviceId::cpu()};

    EXPECT_EQ(store.embeddingHandle(missing), nullptr);
    EXPECT_FALSE(store.preparedRefForBinding(missing.binding_id, missing.device).has_value());
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, BindingLookupFindsCreatedEntry)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto ref = prepareEmbedding(store, 1, tensor.get());

    auto found_ref = store.preparedRefForBinding(ref.binding_id, DeviceId::cpu());
    ASSERT_TRUE(found_ref.has_value());
    EXPECT_EQ(found_ref->binding_id, ref.binding_id);
    EXPECT_EQ(store.embeddingHandle(*found_ref), store.embeddingHandle(ref));
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, LookupMissesWrongDevice)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto ref = prepareEmbedding(store, 1, tensor.get());

    auto wrong_device = ref;
    wrong_device.device = DeviceId(DeviceType::CPU, 1);
    EXPECT_EQ(store.embeddingHandle(wrong_device), nullptr);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, NullTensorThrows)
{
    PreparedWeightStore store(kModelId);
    auto binding = makeEmbeddingBinding(1, nullptr);

    EXPECT_THROW(
        store.prepareEmbedding(binding, kDModelInt),
        std::runtime_error);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, MoveConstructorTransfersOwnership)
{
    PreparedEmbeddingWeights src;
    src.device_data = malloc(4 * 2 * sizeof(Q8_0Block));
    src.layout = NativeEmbeddingLayout::forSource(TensorType::Q8_0, 4, 64);
    src.vocab_size = 4;
    src.d_model = 64;
    src.total_vocab = 4;
    src.device_id = DeviceId::cpu();

    void *original_ptr = src.device_data;

    PreparedEmbeddingWeights dst(std::move(src));

    EXPECT_EQ(dst.device_data, original_ptr);
    EXPECT_EQ(dst.layout.byte_size, 4u * 2u * sizeof(Q8_0Block));
    EXPECT_EQ(dst.layout.blocks_per_row, 2u);
    EXPECT_EQ(dst.vocab_size, 4u);
    EXPECT_EQ(dst.d_model, 64);

    EXPECT_EQ(src.device_data, nullptr);
    EXPECT_EQ(src.layout.byte_size, 0u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, MoveAssignmentTransfersOwnership)
{
    PreparedEmbeddingWeights src;
    src.device_data = malloc(4 * sizeof(Q8_0Block));
    src.layout = NativeEmbeddingLayout::forSource(TensorType::Q8_0, 4, 32);
    src.vocab_size = 4;
    src.total_vocab = 4;
    src.d_model = 32;
    src.device_id = DeviceId::cpu();

    void *original_ptr = src.device_data;

    PreparedEmbeddingWeights dst;
    dst = std::move(src);

    EXPECT_EQ(dst.device_data, original_ptr);
    EXPECT_EQ(dst.layout.byte_size, 4u * sizeof(Q8_0Block));
    EXPECT_EQ(src.device_data, nullptr);
    EXPECT_EQ(src.layout.byte_size, 0u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, DestructorSafeOnNullData)
{
    {
        PreparedEmbeddingWeights w;
    }
    SUCCEED();
}

/** @test Every prepared codebook keeps original payload and original scale bytes. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, AllNativePreparedBytesPreserveOriginalSourceMetadata)
{
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto source = format.create({7, 256}, 139);
        PreparedWeightStore store(kModelId);
        const auto binding = makeEmbeddingBinding(71, source.get());
        const auto ref = store.prepareEmbedding(binding, 256);
        const auto *handle = store.embeddingHandle(ref);
        ASSERT_NE(handle, nullptr);
        ASSERT_NE(handle->weights, nullptr);
        EXPECT_EQ(handle->weights->layout,
            NativeEmbeddingLayout::forSource(format.tensor_type, 7, 256));
        ASSERT_EQ(handle->weights->layout.byte_size, source->size_bytes());
        EXPECT_EQ(std::memcmp(handle->weights->device_data, source->raw_data(), source->size_bytes()), 0);
    }
}

/** @test Native GPU upload rejects an absent/foreign/unbound source before backend lookup. */
TEST(PreparedEmbeddingUploadContract, RejectsMissingUnboundAndWrongDeviceBeforeAllocation)
{
    auto tensor = TestTensorFactory::createQ4_0Random({67, 256}, 193701);
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        SCOPED_TRACE(device.toString());
        EXPECT_THROW(KernelFactory::prepareEmbeddingHandleLocal(
            tensor.get(), 256, device, nullptr), std::invalid_argument);
        auto unbound = TransferEngine::instance().declarePinnedHostBuffer(4096, device);
        EXPECT_THROW(KernelFactory::prepareEmbeddingHandleLocal(
            tensor.get(), 256, device, unbound.get()), std::invalid_argument);
        const DeviceId foreign(device.type, 1);
        auto wrong_device = TransferEngine::instance().declarePinnedHostBuffer(4096, foreign);
        EXPECT_THROW(KernelFactory::prepareEmbeddingHandleLocal(
            tensor.get(), 256, device, wrong_device.get()), std::invalid_argument);
    }
}

namespace
{
    /**
     * @brief Admit only the final CPU table; a duplicate prepared root cannot fit.
     * @param bytes Physical allocator extent of the final native table.
     * @param device Exact CPU allocator, independent of NUMA placement.
     * @return Original authority retained by the actual final storage root.
     */
    std::shared_ptr<PhysicalMemoryAuthority> admitFinalEmbedding(size_t bytes, DeviceId device)
    {
        PhysicalMemoryPlanBuilder builder;
        builder.add(PhysicalMemoryResource{.world_rank = 0, .device = device,
            .total_bytes = bytes, .admission_available_bytes = bytes},
            PhysicalMemoryOwner::PrimaryModelWeights, bytes);
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    }

    /** @return Actual live final-weight bytes in the original CPU allocator. */
    size_t embeddingClaim(const std::shared_ptr<PhysicalMemoryAuthority> &memory,
                          DeviceId device = DeviceId::cpu())
    {
        return memory->claimedBytes(device, PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation);
    }

    /**
     * @brief Populate a final admitted root with unchanged native fixture bytes.
     * @param source Original native table whose format and payload are preserved.
     * @param memory Authority admitting the final root, not an extra prepared copy.
     * @return Owning tensor that adopts the same allocated pointer and lease.
     */
    std::shared_ptr<const TensorBase> finalNativeEmbedding(
        const TensorBase &source, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
        DeviceId device = DeviceId::cpu())
    {
        const CPUStorageRoot<uint8_t> root(source.size_bytes(),
            PhysicalMemoryOwner::PrimaryModelWeights, device,
            CPUStoragePlacement::local(), CPUStorageInitialization::Uninitialized);
        auto bytes = root.allocate(memory);
        std::memcpy(bytes.data(), source.raw_data(), source.size_bytes());
        return TensorFactory::adoptQuantizedFinalStorage(
            source.native_type(), source.shape(), std::move(bytes), device);
    }
}

/** @test All native formats retain one final root through dirty partial lookups and handle retirement. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, FinalCPUStorageAllFormatsRetainOneOwnerAndExecute)
{
    constexpr size_t rows = 7, width = 256;
    const std::array<int, rows> tokens{6, 0, 3, 6, 2, 0, 1};
    for (const auto &format : quantizedVerifierFormats())
    for (const DeviceId device : {DeviceId::cpu(), DeviceId(DeviceType::CPU, 1)})
    {
        SCOPED_TRACE(format.label);
        SCOPED_TRACE(device.to_string());
        auto source = format.create({rows, width}, 73491);
        const size_t bytes = CPUStoragePlacement::local().physicalBytes<uint8_t>(source->size_bytes());
        auto memory = admitFinalEmbedding(bytes, device);
        auto final_tensor = finalNativeEmbedding(*source, memory, device);
        ASSERT_EQ(final_tensor->home_device(), device);
        const auto *const original_tensor = final_tensor.get();
        const void *const original_bytes = final_tensor->raw_data();
        std::weak_ptr<const TensorBase> owner = final_tensor;
        auto handle = KernelFactory::prepareEmbeddingHandleFromFinalCPUStorage(final_tensor, width);
        ASSERT_NE(handle, nullptr);
        ASSERT_NE(handle->weights, nullptr);
        EXPECT_EQ(handle->tensor, original_tensor);
        EXPECT_EQ(handle->weights->device_data, original_bytes);
        EXPECT_EQ(handle->weights->retainedFinalCPUTensor().get(), original_tensor);
        EXPECT_FALSE(final_tensor.owner_before(handle->weights->retainedFinalCPUTensor()));
        EXPECT_FALSE(handle->weights->retainedFinalCPUTensor().owner_before(final_tensor));
        ASSERT_EQ(embeddingClaim(memory, device), bytes);
        EXPECT_EQ(handle->device_id, device);
        {
            auto view = std::const_pointer_cast<TensorBase>(final_tensor)->create_view({3, width}, width);
            ASSERT_NE(view, nullptr);
            EXPECT_TRUE(view->is_view());
            EXPECT_EQ(view->home_device(), device);
            auto nested = view->create_view({1, width}, width);
            EXPECT_EQ(nested->home_device(), device);
        }
        EXPECT_EQ(std::memcmp(original_bytes, source->raw_data(), source->size_bytes()), 0);

        FP32Tensor expected({rows, width}), actual({rows, width});
        for (size_t row = 0; row < rows; ++row)
            source->to_fp32_row(tokens[row], expected.mutable_data() + row * width);
        source.reset();
        final_tensor.reset();
        ASSERT_FALSE(owner.expired());
        CPUEmbeddingKernelT<FP32Tensor> kernel;
        for (int live_rows : {7, 3, 1, 7, 2})
        {
            std::fill_n(actual.mutable_data(), rows * width, -93.f);
            ASSERT_TRUE(kernel.apply_tensor(handle->tensor, tokens.data(), live_rows, width, &actual));
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), live_rows * width * sizeof(float)), 0);
            for (size_t i = live_rows * width; i < rows * width; ++i)
                EXPECT_EQ(actual.data()[i], -93.f);
            EXPECT_EQ(embeddingClaim(memory, device), bytes);
        }
        // Store/graph aliases can outlive the handle: the original final tensor
        // and its sole PMA lease must retire only with the last actual consumer.
        auto retained = handle->weights;
        handle.reset();
        EXPECT_FALSE(owner.expired());
        EXPECT_EQ(embeddingClaim(memory, device), bytes);
        retained.reset();
        EXPECT_TRUE(owner.expired());
        EXPECT_EQ(embeddingClaim(memory, device), 0u);
    }
}

/** @test Final storage moves preserve its exact CPU identity and never backend-free an alias. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, FinalCPUStorageMovesPreserveExactAllocator)
{
    const DeviceId cpu1(DeviceType::CPU, 1);
    auto source = TestTensorFactory::createQ8_1Random({7, 256}, -1.f, 1.f, 1901);
    const CPUStorageRoot<uint8_t> root(source->size_bytes(),
        PhysicalMemoryOwner::PrimaryModelWeights, cpu1,
        CPUStoragePlacement::local(), CPUStorageInitialization::Uninitialized);
    auto memory = admitFinalEmbedding(root.physicalBytes(), cpu1);
    auto bytes = root.allocate(memory);
    std::memcpy(bytes.data(), source->raw_data(), source->size_bytes());
    std::shared_ptr<const TensorBase> final_tensor = std::make_shared<Q8_1Tensor>(
        source->shape(), std::move(bytes), cpu1);
    std::weak_ptr<const TensorBase> original_owner = final_tensor;
    auto handle = KernelFactory::prepareEmbeddingHandleFromFinalCPUStorage(final_tensor, 256, 11, 31);
    ASSERT_EQ(handle->device_id, cpu1);
    EXPECT_EQ(handle->weights->device_id, cpu1);
    EXPECT_EQ(handle->weights->vocab_offset, 11u);
    EXPECT_EQ(handle->weights->total_vocab, 31u);
    const void *const original_bytes = final_tensor->raw_data();
    final_tensor.reset();
    {
        PreparedEmbeddingWeights moved(std::move(*handle->weights));
        EXPECT_EQ(handle->weights->device_data, nullptr);
        EXPECT_EQ(handle->weights->retainedFinalCPUTensor(), nullptr);
        EXPECT_EQ(moved.device_data, original_bytes);
        EXPECT_EQ(moved.device_id, cpu1);
        PreparedEmbeddingWeights assigned;
        assigned = std::move(moved);
        EXPECT_EQ(moved.device_data, nullptr);
        EXPECT_EQ(moved.retainedFinalCPUTensor(), nullptr);
        EXPECT_EQ(assigned.device_data, original_bytes);
        EXPECT_EQ(assigned.retainedFinalCPUTensor()->home_device(), cpu1);
        EXPECT_EQ(embeddingClaim(memory, cpu1), root.physicalBytes());
        handle.reset();
        EXPECT_FALSE(original_owner.expired());
    }
    EXPECT_TRUE(original_owner.expired());
    EXPECT_EQ(embeddingClaim(memory, cpu1), 0u);
}

/** @test Borrowed/mapped storage and incompatible geometry cannot masquerade as final native roots. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, FinalCPUStorageRejectsInvalidOwnershipAndGeometry)
{
    EXPECT_THROW(KernelFactory::prepareEmbeddingHandleFromFinalCPUStorage({}, 256), std::invalid_argument);
    auto source = std::shared_ptr<TensorBase>(TestTensorFactory::createQ4_0Random({7, 256}, 1349));
    const auto prepare = [](std::shared_ptr<const TensorBase> tensor, int width = 256,
                            size_t offset = 0, size_t total = 0)
    { return KernelFactory::prepareEmbeddingHandleFromFinalCPUStorage(std::move(tensor), width, offset, total); };
    EXPECT_THROW(prepare(std::make_shared<FP32Tensor>(std::vector<size_t>{7, 256})), std::invalid_argument);
    EXPECT_THROW(prepare(source, 0), std::invalid_argument);
    EXPECT_THROW(prepare(source, 128), std::invalid_argument);
    EXPECT_THROW(prepare(source, 256, 8, 14), std::invalid_argument);
    EXPECT_THROW(prepare(source, 256, 21, 14), std::invalid_argument);
    auto view = source->create_view({7, 256});
    ASSERT_NE(view, nullptr);
    EXPECT_THROW(prepare(view), std::invalid_argument);
    MockMPIContext context;
    auto borrowed = TensorFactory(context).createQuantizedZeroCopy(source->native_type(), source->shape(),
        static_cast<const uint8_t *>(source->raw_data()), source->size_bytes(), source);
    EXPECT_THROW(prepare(std::move(borrowed)), std::invalid_argument);
    source->release_host_weight_data();
    EXPECT_THROW(prepare(source), std::invalid_argument);
}

/** @test Every native row view reports borrowing and is rejected by final-root adoption. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, FinalCPUStorageAllFormatViewsReportBorrowedOwnership)
{
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto source = std::shared_ptr<TensorBase>(format.create({7, 256}, 2301));
        ASSERT_FALSE(source->is_view());
        auto view = source->create_view({3, 256}, 256);
        ASSERT_NE(view, nullptr);
        EXPECT_TRUE(view->is_view());
        EXPECT_THROW(KernelFactory::prepareEmbeddingHandleFromFinalCPUStorage(view, 256), std::invalid_argument);
        auto nested = view->create_view({1, 256}, 256);
        ASSERT_NE(nested, nullptr);
        EXPECT_TRUE(nested->is_view());
        EXPECT_THROW(KernelFactory::prepareEmbeddingHandleFromFinalCPUStorage(nested, 256), std::invalid_argument);
    }
}

/** @test Rebinding or charging an adopted alias fails without retiring the original live tensor. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, FinalCPUStorageRejectsDuplicateBackingAndClaim)
{
    auto source = TestTensorFactory::createQ4_0Random({7, 256}, 5341);
    const size_t bytes = CPUStoragePlacement::local().physicalBytes<uint8_t>(source->size_bytes());
    auto memory = admitFinalEmbedding(bytes, DeviceId::cpu());
    auto final_tensor = finalNativeEmbedding(*source, memory);
    auto handle = KernelFactory::prepareEmbeddingHandleFromFinalCPUStorage(final_tensor, 256);
    EXPECT_THROW(handle->weights->bindFinalCPUStorage(final_tensor), std::logic_error);
    auto other_memory = admitFinalEmbedding(source->size_bytes(), DeviceId::cpu());
    auto extra_claim = other_memory->claimNewAllocation(DeviceId::cpu(),
        PhysicalMemoryOwner::PrimaryModelWeights, source->size_bytes());
    EXPECT_THROW(handle->weights->bindPhysicalMemoryLease(std::move(extra_claim)), std::logic_error);
    EXPECT_EQ(embeddingClaim(other_memory), 0u);
    EXPECT_EQ(embeddingClaim(memory), bytes);
    EXPECT_EQ(handle->weights->device_data, final_tensor->raw_data());
    EXPECT_EQ(handle->weights->retainedFinalCPUTensor().get(), final_tensor.get());
}

/** @test Native final adoption rejects a foreign allocator before exposing a tensor. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, FinalCPUStorageAllFormatsRejectForeignAllocators)
{
    for (const auto &format : quantizedVerifierFormats())
    for (const auto device : {DeviceId::invalid(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        SCOPED_TRACE(format.label);
        EXPECT_THROW(TensorFactory::adoptQuantizedFinalStorage(
            format.tensor_type, {7, 256}, AlignedVector<uint8_t>{}, device), std::invalid_argument);
    }
}

/** @test The runtime Q8_1 native-window decoder borrows exact bytes and retains a supplied real owner. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, BorrowedQ81SourceRetainsOwnerAndRevokesViews)
{
    auto source = std::shared_ptr<TensorBase>(TestTensorFactory::createQ8_1Random({7, 256}, -1.f, 1.f, 9137));
    std::weak_ptr<TensorBase> owner = source;
    const auto *raw = static_cast<const uint8_t *>(source->raw_data());
    auto borrowed = std::make_shared<Q8_1Tensor>(source->shape(), raw, source->size_bytes(), std::shared_ptr<void>(source));
    EXPECT_EQ(borrowed->raw_data(), raw);
    EXPECT_TRUE(borrowed->is_view());
    EXPECT_FALSE(borrowed->is_mutable());
    EXPECT_THROW(borrowed->mutable_q8_1_blocks(), std::logic_error);
    EXPECT_THROW(KernelFactory::prepareEmbeddingHandleFromFinalCPUStorage(borrowed, 256), std::invalid_argument);
    auto child = borrowed->create_view({3, 256}, 256);
    auto nested = child->create_view({1, 256}, 256);
    std::array<float, 256> expected{}, actual{};
    source->to_fp32_row(2, expected.data());
    source.reset();
    child.reset();
    EXPECT_FALSE(owner.expired());
    nested->to_fp32_row(0, actual.data());
    EXPECT_EQ(std::memcmp(expected.data(), actual.data(), sizeof(expected)), 0);
    borrowed->release_host_weight_data();
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(nested->raw_data(), nullptr);
    EXPECT_TRUE(nested->is_raw_data_released());

    auto scoped_source = TestTensorFactory::createQ8_1Random({1, 256}, -1.f, 1.f, 9781);
    const auto *scoped_bytes = static_cast<const uint8_t *>(scoped_source->raw_data());
    // The real source visitor owns this input span throughout its synchronous
    // callback. No fake lifetime owner or second native allocation is needed.
    Q8_1Tensor scoped(scoped_source->shape(), scoped_bytes, scoped_source->size_bytes(), std::shared_ptr<void>{});
    EXPECT_EQ(scoped.raw_data(), scoped_bytes);
    scoped.to_fp32_row(0, actual.data());
    scoped_source->to_fp32_row(0, expected.data());
    EXPECT_EQ(std::memcmp(expected.data(), actual.data(), sizeof(expected)), 0);
    EXPECT_THROW((Q8_1Tensor({1, 256}, nullptr, scoped_source->size_bytes(), std::shared_ptr<void>{})), std::invalid_argument);
    EXPECT_THROW((Q8_1Tensor({1, 256}, scoped_bytes, scoped_source->size_bytes() - 1, std::shared_ptr<void>{})), std::invalid_argument);
}

/** @test The final weight factory preserves the root and seals activation writes before publication. */
TEST_F(Test__PreparedEmbeddingWeightsLifecycle, FinalCPUStorageQ81IsImmutable)
{
    auto source = TestTensorFactory::createQ8_1Random({7, 256}, -1.f, 1.f, 1781);
    const auto cpu1 = DeviceId{DeviceType::CPU, 1};
    const auto physical_bytes = CPUStoragePlacement::local().physicalBytes<uint8_t>(source->size_bytes());
    auto memory = admitFinalEmbedding(physical_bytes, cpu1);
    auto published = finalNativeEmbedding(*source, memory, cpu1);
    auto final = std::dynamic_pointer_cast<const Q8_1Tensor>(published);
    ASSERT_TRUE(final);
    EXPECT_FALSE(final->is_mutable());
    EXPECT_EQ(final->home_device(), cpu1);
    EXPECT_EQ(std::memcmp(final->raw_data(), source->raw_data(), source->size_bytes()), 0);
    // Factory publication returns const ownership, but a later service retaining
    // the concrete owner still must not obtain mutable activation blocks.
    auto retained = std::const_pointer_cast<Q8_1Tensor>(final);
    EXPECT_THROW(retained->mutable_q8_1_blocks(), std::logic_error);
    auto view = std::dynamic_pointer_cast<Q8_1Tensor>(retained->create_view({1, 256}));
    ASSERT_TRUE(view);
    EXPECT_FALSE(view->is_mutable());
    EXPECT_THROW(view->mutable_q8_1_blocks(), std::logic_error);
    Q8_1Tensor activation({1, 256});
    EXPECT_TRUE(activation.is_mutable());
    EXPECT_NE(activation.mutable_q8_1_blocks(), nullptr);
}
