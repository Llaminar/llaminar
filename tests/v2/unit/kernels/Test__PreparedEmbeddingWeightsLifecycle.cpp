/**
 * @file Test__PreparedEmbeddingWeightsLifecycle.cpp
 * @brief Unit tests for model-owned PreparedEmbeddingWeights lifecycle
 *
 * Tests verify:
 * 1. PreparedWeightStore owns embedding handles and can release them
 * 2. FP32 tensors are rejected for packed embedding preparation
 * 3. Q8_0 and Q4_0 tensors get proper EmbedQ8 repack + handle metadata
 * 4. Prepared refs and tensor lookups resolve existing handles without creating entries
 * 5. Different bindings get separate embedding entries
 * 6. Null tensor input throws
 * 7. Move semantics are safe (no double-free)
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <utility>

#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/PreparedEmbeddingWeights.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "../../utils/TestTensorFactory.h"

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
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});

    EXPECT_EQ(store.embeddingHandleForTensor(tensor.get(), DeviceId::cpu()), nullptr);
    EXPECT_FALSE(store.preparedRefForTensor(tensor.get(), DeviceId::cpu()).has_value());
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, ClearReleasesEmbeddingEntries)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto ref = prepareEmbedding(store, 1, tensor.get());

    ASSERT_TRUE(store.contains(ref));
    ASSERT_NE(store.embeddingHandle(ref), nullptr);
    ASSERT_NE(store.embeddingHandleForTensor(tensor.get(), DeviceId::cpu()), nullptr);

    store.clear();

    EXPECT_FALSE(store.contains(ref));
    EXPECT_EQ(store.embeddingHandle(ref), nullptr);
    EXPECT_EQ(store.embeddingHandleForTensor(tensor.get(), DeviceId::cpu()), nullptr);
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
    EXPECT_EQ(store.embeddingHandleForTensor(tensor.get(), DeviceId::cpu()), nullptr);
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
    EXPECT_GT(handle->weights->byte_size, 0u);
    EXPECT_EQ(handle->weights->vocab_size, kVocabSize);
    EXPECT_EQ(handle->weights->d_model, kDModelInt);
    EXPECT_GT(handle->weights->blocks_per_row, 0u);
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
    EXPECT_EQ(handle->weights->blocks_per_row, expected_blocks_per_row);

    const size_t expected_byte_size = kVocabSize * expected_blocks_per_row * 36;
    EXPECT_EQ(handle->weights->byte_size, expected_byte_size);
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
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});

    EXPECT_EQ(store.embeddingHandleForTensor(tensor.get(), DeviceId::cpu()), nullptr);
    EXPECT_FALSE(store.preparedRefForTensor(tensor.get(), DeviceId::cpu()).has_value());
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, LookupFindsCreatedEntry)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    auto ref = prepareEmbedding(store, 1, tensor.get());

    auto *found = store.embeddingHandleForTensor(tensor.get(), DeviceId::cpu());
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found, store.embeddingHandle(ref));
    EXPECT_EQ(found->tensor, tensor.get());

    auto found_ref = store.preparedRefForTensor(tensor.get(), DeviceId::cpu());
    ASSERT_TRUE(found_ref.has_value());
    EXPECT_EQ(found_ref->binding_id, ref.binding_id);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, LookupMissesWrongDevice)
{
    PreparedWeightStore store(kModelId);
    auto tensor = TestTensorFactory::createQ8_0Random({kVocabSize, kDModel});
    prepareEmbedding(store, 1, tensor.get());

    auto *found = store.embeddingHandleForTensor(tensor.get(), DeviceId(DeviceType::CPU, 1));
    EXPECT_EQ(found, nullptr);
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
    src.device_data = malloc(256);
    src.byte_size = 256;
    src.blocks_per_row = 2;
    src.vocab_size = 4;
    src.d_model = 32;
    src.device_id = DeviceId::cpu();

    void *original_ptr = src.device_data;

    PreparedEmbeddingWeights dst(std::move(src));

    EXPECT_EQ(dst.device_data, original_ptr);
    EXPECT_EQ(dst.byte_size, 256u);
    EXPECT_EQ(dst.blocks_per_row, 2u);
    EXPECT_EQ(dst.vocab_size, 4u);
    EXPECT_EQ(dst.d_model, 32);

    EXPECT_EQ(src.device_data, nullptr);
    EXPECT_EQ(src.byte_size, 0u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, MoveAssignmentTransfersOwnership)
{
    PreparedEmbeddingWeights src;
    src.device_data = malloc(128);
    src.byte_size = 128;
    src.device_id = DeviceId::cpu();

    void *original_ptr = src.device_data;

    PreparedEmbeddingWeights dst;
    dst = std::move(src);

    EXPECT_EQ(dst.device_data, original_ptr);
    EXPECT_EQ(dst.byte_size, 128u);
    EXPECT_EQ(src.device_data, nullptr);
    EXPECT_EQ(src.byte_size, 0u);
}

TEST_F(Test__PreparedEmbeddingWeightsLifecycle, DestructorSafeOnNullData)
{
    {
        PreparedEmbeddingWeights w;
    }
    SUCCEED();
}