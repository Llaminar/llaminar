/**
 * @file Test__PreparedWeightStore.cpp
 * @brief Prepared-weight identity, lifetime, adoption, and backend isolation tests.
 */

#include <gtest/gtest.h>

#include "loaders/PreparedWeightStore.h"
#include "tensors/TensorSlice.h"
#include "tensors/Tensors.h"
#include "../../utils/PreparedWeightTestHarness.h"

#include <cstring>
#include <memory>
#include <random>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    WeightBinding makeStoreBinding(uint64_t binding_id, const std::string &name, DeviceId device)
    {
        WeightBinding binding;
        binding.binding_id = binding_id;
        binding.identity = makeSourceWeightIdentity(name, ModelContextId{99}, binding_id);
        binding.residency.home_device = device;
        binding.residency.resident_device = device;
        binding.immutable = true;
        return binding;
    }

    std::unique_ptr<Q8_0Tensor> makeQ8_0Tensor(size_t rows, size_t cols)
    {
        constexpr size_t block_size = 32;
        constexpr size_t bytes_per_block = 34;
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
            byte = static_cast<uint8_t>(dist(gen));
        for (size_t block = 0; block < num_blocks; ++block)
        {
            uint16_t scale_bits = 0x3C00;
            std::memcpy(&raw_data[block * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q8_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    PreparedEmbeddingHandle makeEmbeddingHandle(TensorBase *tensor, DeviceId device)
    {
        PreparedEmbeddingHandle handle;
        handle.tensor = tensor;
        handle.device_id = device;
        handle.weights = std::make_shared<PreparedEmbeddingWeights>();
        handle.weights->device_id = device;
        return handle;
    }
}

TEST(Test__PreparedWeightStore, RegistersAndFindsMockPreparedWeights)
{
    PreparedWeightStore store(ModelContextId{99});
    auto binding = makeStoreBinding(42, "blk.0.ffn_gate.weight", DeviceId::rocm(0));

    auto ref = store.registerPreparedForTest(
        binding, PreparedWeightKind::RocmInt8PackedGemm, DeviceId::rocm(0));

    EXPECT_EQ(ref.model_id.value, 99u);
    EXPECT_EQ(ref.binding_id, 42u);
    EXPECT_TRUE(store.contains(ref));
    EXPECT_EQ(store.size(), 1u);

    auto stored = store.binding(ref);
    ASSERT_TRUE(stored.has_value());
    EXPECT_EQ(stored->identity.canonical_name, "blk.0.ffn_gate.weight");
    ASSERT_TRUE(stored->prepared.has_value());
    EXPECT_EQ(stored->prepared->kind, PreparedWeightKind::RocmInt8PackedGemm);
}

TEST(Test__PreparedWeightStore, BindModelIdIfUnsetBindsOnce)
{
    PreparedWeightStore store;

    EXPECT_EQ(store.modelId().value, 0u);
    EXPECT_TRUE(store.bindModelIdIfUnset(ModelContextId{77}));
    EXPECT_EQ(store.modelId().value, 77u);
    EXPECT_TRUE(store.bindModelIdIfUnset(ModelContextId{77}));
    EXPECT_FALSE(store.bindModelIdIfUnset(ModelContextId{78}));
    EXPECT_EQ(store.modelId().value, 77u);
}

TEST(Test__PreparedWeightStore, BindModelIdIfUnsetRejectsZeroAgainstBoundStore)
{
    PreparedWeightStore store(ModelContextId{88});

    EXPECT_FALSE(store.bindModelIdIfUnset(ModelContextId{}));
    EXPECT_EQ(store.modelId().value, 88u);
}

TEST(Test__PreparedWeightStore, RejectsWrongModelOrDevice)
{
    PreparedWeightStore store(ModelContextId{99});
    auto binding = makeStoreBinding(7, "blk.0.attn_q.weight", DeviceId::cuda(0));
    auto ref = store.registerPreparedForTest(
        binding, PreparedWeightKind::CudaInt8PackedGemm, DeviceId::cuda(0));

    auto wrong_model = ref;
    wrong_model.model_id = ModelContextId{100};
    EXPECT_FALSE(store.contains(wrong_model));

    auto wrong_device = ref;
    wrong_device.device = DeviceId::cuda(1);
    EXPECT_FALSE(store.contains(wrong_device));
    EXPECT_FALSE(store.binding(wrong_device).has_value());
}

TEST(Test__PreparedWeightStore, RejectsStageLocalBindingModelIds)
{
    PreparedWeightStore store(ModelContextId{99});
    auto binding = makeStoreBinding(17, "blk.0.attn_q.weight", DeviceId::cpu());
    binding.identity.model_id = ModelContextId{1234};

    EXPECT_THROW(store.registerPreparedForTest(
                     binding, PreparedWeightKind::CpuPackedGemm, DeviceId::cpu()),
                 std::runtime_error);
}

TEST(Test__PreparedWeightStore, MockEntriesHaveNoExecutableKernel)
{
    PreparedWeightStore store(ModelContextId{99});
    auto binding = makeStoreBinding(8, "blk.0.ffn_down.weight", DeviceId::cpu());
    auto ref = store.registerPreparedForTest(
        binding, PreparedWeightKind::CpuPackedGemm, DeviceId::cpu());

    EXPECT_EQ(store.gemmKernel(ref), nullptr);
    store.clear();
    EXPECT_EQ(store.size(), 0u);
    EXPECT_FALSE(store.contains(ref));
}

TEST(Test__PreparedWeightStore, ResolvesPreparedRefByBindingAndDevice)
{
    PreparedWeightStore store(ModelContextId{99});
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 8});
    auto binding = makeStoreBinding(12, "blk.0.ffn_gate.weight", DeviceId::rocm(0));
    binding.tensor = tensor.get();

    auto ref = store.registerPreparedForTest(
        binding, PreparedWeightKind::RocmInt8PackedGemm, DeviceId::rocm(0));

    auto resolved = store.preparedRefForBinding(binding.binding_id, DeviceId::rocm(0));
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved->binding_id, ref.binding_id);
    EXPECT_EQ(resolved->kind, PreparedWeightKind::RocmInt8PackedGemm);
    EXPECT_FALSE(store.preparedRefForBinding(binding.binding_id, DeviceId::rocm(1)).has_value());
}

TEST(Test__PreparedWeightStore, SameBindingIdRetainsDistinctGemmEntriesPerDevice)
{
    PreparedWeightStore store(ModelContextId{99});
    auto cuda0_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 8});
    auto cuda1_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 8});

    auto cuda0_binding = makeStoreBinding(12, "blk.0.ffn_gate.weight", DeviceId::cuda(0));
    cuda0_binding.tensor = cuda0_tensor.get();
    auto cuda1_binding = makeStoreBinding(12, "blk.0.ffn_gate.weight", DeviceId::cuda(1));
    cuda1_binding.tensor = cuda1_tensor.get();

    const auto cuda0_ref = store.registerPreparedForTest(
        cuda0_binding, PreparedWeightKind::CudaInt8PackedGemm, DeviceId::cuda(0));
    const auto cuda1_ref = store.registerPreparedForTest(
        cuda1_binding, PreparedWeightKind::CudaInt8PackedGemm, DeviceId::cuda(1));

    EXPECT_EQ(store.size(), 2u);
    EXPECT_TRUE(store.contains(cuda0_ref));
    EXPECT_TRUE(store.contains(cuda1_ref));
    ASSERT_TRUE(store.binding(cuda0_ref).has_value());
    ASSERT_TRUE(store.binding(cuda1_ref).has_value());
    EXPECT_EQ(store.binding(cuda0_ref)->tensor, cuda0_tensor.get());
    EXPECT_EQ(store.binding(cuda1_ref)->tensor, cuda1_tensor.get());
}

TEST(Test__PreparedWeightStore, RegistersOwnedPipelineHandleByBinding)
{
    PreparedWeightStore store(ModelContextId{99});
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 8});
    auto binding = makeStoreBinding(13, "blk.0.ffn_up.weight", DeviceId::cuda(0));
    binding.tensor = tensor.get();

    auto handle = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>();
    handle->tensor = tensor.get();
    handle->device_id = DeviceId::cuda(0);
    handle->kind = llaminar::v2::kernels::KernelFactory::GemmPreparationKind::FLOATING_POINT;
    handle->prepared_weights = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmWeights>();

    auto ref = store.registerPreparedGemmHandle(
        binding,
        PreparedWeightKind::CudaInt8PackedGemm,
        DeviceId::cuda(0),
        std::move(handle));

    EXPECT_TRUE(store.contains(ref));
    auto resolved = store.preparedRefForBinding(binding.binding_id, DeviceId::cuda(0));
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved->binding_id, ref.binding_id);
    EXPECT_TRUE(tensor->hasPreparedDeviceState());
}

TEST(Test__PreparedWeightStore, AdoptsPreparedGemmByCanonicalNameAfterHostPayloadRelease)
{
    PreparedWeightStore store(ModelContextId{99});
    auto prepared_tensor = makeQ8_0Tensor(64, 64);
    auto prepared_binding = makeStoreBinding(31, "output.weight", DeviceId::rocm(0));
    prepared_binding.tensor = prepared_tensor.get();

    auto handle = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>();
    handle->tensor = prepared_tensor.get();
    handle->device_id = DeviceId::rocm(0);
    handle->kind = llaminar::v2::kernels::KernelFactory::GemmPreparationKind::ROCM_INT8_PACKED;
    handle->prepared_weights = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmWeights>();

    auto prepared_ref = store.registerPreparedGemmHandle(
        prepared_binding,
        PreparedWeightKind::RocmInt8PackedGemm,
        DeviceId::rocm(0),
        std::move(handle));
    ASSERT_TRUE(store.contains(prepared_ref));

    auto later_tensor = makeQ8_0Tensor(64, 64);
    later_tensor->release_host_weight_data();
    ASSERT_TRUE(later_tensor->is_raw_data_released());
    ASSERT_EQ(later_tensor->size_bytes(), 0u);

    auto later_binding = makeStoreBinding(32, "output.weight", DeviceId::rocm(0));
    later_binding.tensor = later_tensor.get();
    later_binding.prepared = PreparedWeightRef{
        ModelContextId{99},
        later_binding.binding_id,
        PreparedWeightKind::RocmInt8PackedGemm,
        DeviceId::rocm(0)};

    ASSERT_TRUE(store.adoptPreparedGemmForBinding(later_binding, DeviceId::rocm(0)));
    auto adopted_ref = store.preparedRefForBinding(later_binding.binding_id, DeviceId::rocm(0));
    ASSERT_TRUE(adopted_ref.has_value());
    EXPECT_EQ(adopted_ref->binding_id, later_binding.binding_id);
    EXPECT_EQ(adopted_ref->kind, PreparedWeightKind::RocmInt8PackedGemm);
    EXPECT_TRUE(later_tensor->hasPreparedDeviceState());
}

TEST(Test__PreparedWeightStore, ExactBindingReuseRestoresStoreOwnedTensorAuthority)
{
    PreparedWeightStore store(ModelContextId{99});
    auto authoritative_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 8});
    auto authoritative_binding = makeStoreBinding(
        33, "blk.0.ssm_alpha.weight", DeviceId::cpu());
    authoritative_binding.tensor_owner = authoritative_tensor;
    authoritative_binding.tensor = authoritative_tensor.get();

    store.registerPreparedForTest(
        authoritative_binding,
        PreparedWeightKind::CpuPackedGemm,
        DeviceId::cpu());

    auto rematerialized_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 8});
    auto rematerialized_binding = makeStoreBinding(
        33, "blk.0.ssm_alpha.weight", DeviceId::cpu());
    rematerialized_binding.tensor_owner = rematerialized_tensor;
    rematerialized_binding.tensor = rematerialized_tensor.get();
    rematerialized_binding.prepared = PreparedWeightRef{
        ModelContextId{99},
        rematerialized_binding.binding_id,
        PreparedWeightKind::CpuPackedGemm,
        DeviceId::cpu()};

    ASSERT_TRUE(store.adoptPreparedGemmForBinding(
        rematerialized_binding,
        DeviceId::cpu()));
    EXPECT_EQ(rematerialized_binding.tensor, authoritative_tensor.get());
    EXPECT_EQ(rematerialized_binding.tensor_owner, authoritative_tensor);
    EXPECT_TRUE(authoritative_tensor->hasPreparedDeviceState());
}

TEST(Test__PreparedWeightStore, DoesNotAdoptPreparedGemmAcrossDifferentSlicesWithSameCanonicalName)
{
    PreparedWeightStore store(ModelContextId{99});

    auto sharded_tensor = makeQ8_0Tensor(2048, 4096);
    auto sharded_binding = makeStoreBinding(41, "blk.19.attn_output.weight", DeviceId::cuda(1));
    sharded_binding.tensor = sharded_tensor.get();
    sharded_binding.slice.source_rows = 4096;
    sharded_binding.slice.source_cols = 4096;
    sharded_binding.slice.row_start = 2048;
    sharded_binding.slice.row_count = 2048;
    sharded_binding.slice.col_start = 0;
    sharded_binding.slice.col_count = 4096;
    sharded_binding.slice.inner_is_presliced = true;

    auto handle = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>();
    handle->tensor = sharded_tensor.get();
    handle->device_id = DeviceId::cuda(1);
    handle->kind = llaminar::v2::kernels::KernelFactory::GemmPreparationKind::CUDA_INT8_PACKED;
    handle->prepared_weights = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmWeights>();

    auto sharded_ref = store.registerPreparedGemmHandle(
        sharded_binding,
        PreparedWeightKind::CudaInt8PackedGemm,
        DeviceId::cuda(1),
        std::move(handle));
    ASSERT_TRUE(store.contains(sharded_ref));

    auto full_tensor = makeQ8_0Tensor(4096, 4096);
    auto full_binding = makeStoreBinding(42, "blk.19.attn_output.weight", DeviceId::cuda(1));
    full_binding.tensor = full_tensor.get();
    full_binding.slice.source_rows = 4096;
    full_binding.slice.source_cols = 4096;
    full_binding.slice.row_start = 0;
    full_binding.slice.row_count = 4096;
    full_binding.slice.col_start = 0;
    full_binding.slice.col_count = 4096;
    full_binding.prepared = PreparedWeightRef{
        ModelContextId{99},
        full_binding.binding_id,
        PreparedWeightKind::CudaInt8PackedGemm,
        DeviceId::cuda(1)};

    EXPECT_FALSE(store.adoptPreparedGemmForBinding(full_binding, DeviceId::cuda(1)))
        << "Dense decode replicated full weights must not inherit a TP shard prepared handle by canonical name";
    EXPECT_FALSE(store.preparedRefForBinding(full_binding.binding_id, DeviceId::cuda(1)).has_value());
}

TEST(Test__PreparedWeightStore, AdoptsPreparedGemmAcrossEquivalentSlicesWithSameCanonicalName)
{
    PreparedWeightStore store(ModelContextId{99});

    auto prepared_tensor = makeQ8_0Tensor(2048, 2048);
    auto prepared_binding = makeStoreBinding(43, "output.weight", DeviceId::cuda(0));
    prepared_binding.tensor = prepared_tensor.get();
    prepared_binding.slice.source_rows = 4096;
    prepared_binding.slice.source_cols = 2048;
    prepared_binding.slice.row_start = 0;
    prepared_binding.slice.row_count = 2048;
    prepared_binding.slice.col_start = 0;
    prepared_binding.slice.col_count = 2048;
    prepared_binding.slice.inner_is_presliced = true;

    auto handle = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>();
    handle->tensor = prepared_tensor.get();
    handle->device_id = DeviceId::cuda(0);
    handle->kind = llaminar::v2::kernels::KernelFactory::GemmPreparationKind::CUDA_INT8_PACKED;
    handle->prepared_weights = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmWeights>();

    auto prepared_ref = store.registerPreparedGemmHandle(
        prepared_binding,
        PreparedWeightKind::CudaInt8PackedGemm,
        DeviceId::cuda(0),
        std::move(handle));
    ASSERT_TRUE(store.contains(prepared_ref));

    auto later_tensor = makeQ8_0Tensor(2048, 2048);
    later_tensor->release_host_weight_data();
    auto later_binding = makeStoreBinding(44, "output.weight", DeviceId::cuda(0));
    later_binding.tensor = later_tensor.get();
    later_binding.slice = prepared_binding.slice;
    later_binding.prepared = PreparedWeightRef{
        ModelContextId{99},
        later_binding.binding_id,
        PreparedWeightKind::CudaInt8PackedGemm,
        DeviceId::cuda(0)};

    EXPECT_TRUE(store.adoptPreparedGemmForBinding(later_binding, DeviceId::cuda(0)))
        << "Frozen sharded bindings should be able to adopt equivalent generic pipeline handles";
    auto adopted_ref = store.preparedRefForBinding(later_binding.binding_id, DeviceId::cuda(0));
    ASSERT_TRUE(adopted_ref.has_value());
    EXPECT_EQ(adopted_ref->binding_id, later_binding.binding_id);
    EXPECT_TRUE(later_tensor->hasPreparedDeviceState());
}

TEST(Test__PreparedWeightStore, SameTensorDifferentBindingDoesNotResolveAccidentally)
{
    PreparedWeightStore store(ModelContextId{99});
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 8});

    auto prepared_binding = makeStoreBinding(21, "blk.0.ffn_gate.weight", DeviceId::cuda(0));
    prepared_binding.tensor = tensor.get();
    auto unprepared_binding = makeStoreBinding(22, "blk.0.ffn_up.weight", DeviceId::cuda(0));
    unprepared_binding.tensor = tensor.get();

    auto ref = store.registerPreparedForTest(
        prepared_binding,
        PreparedWeightKind::CudaInt8PackedGemm,
        DeviceId::cuda(0));

    ASSERT_TRUE(store.contains(ref));
    EXPECT_TRUE(store.preparedRefForBinding(prepared_binding.binding_id, DeviceId::cuda(0)).has_value());
    EXPECT_FALSE(store.preparedRefForBinding(unprepared_binding.binding_id, DeviceId::cuda(0)).has_value());

    PreparedWeightRef wrong_ref{
        ModelContextId{99},
        unprepared_binding.binding_id,
        PreparedWeightKind::CudaInt8PackedGemm,
        DeviceId::cuda(0)};
    EXPECT_FALSE(store.contains(wrong_ref));
}

TEST(Test__PreparedWeightStore, ResolvesPreparedEmbeddingRefsByBinding)
{
    PreparedWeightStore store(ModelContextId{99});
    auto tensor = makeQ8_0Tensor(64, 96);
    auto binding = makeStoreBinding(15, "token_embd.weight", DeviceId::cuda(0));
    binding.identity.role = WeightRole::Embedding;
    binding.tensor = tensor.get();

    auto handle = makeEmbeddingHandle(tensor.get(), DeviceId::cuda(0));
    auto ref = store.registerPreparedEmbeddingFromPipeline(
        binding, DeviceId::cuda(0), &handle);

    EXPECT_EQ(ref.kind, PreparedWeightKind::PreparedEmbedding);
    EXPECT_TRUE(store.contains(ref));

    auto by_binding = store.preparedRefForBinding(binding.binding_id, DeviceId::cuda(0));
    ASSERT_TRUE(by_binding.has_value());
    EXPECT_EQ(by_binding->kind, PreparedWeightKind::PreparedEmbedding);

    auto stored = store.binding(ref);
    ASSERT_TRUE(stored.has_value());
    EXPECT_EQ(stored->identity.role, WeightRole::Embedding);
}

TEST(Test__PreparedWeightStore, TypedEmbeddingAdoptionSurvivesGemmBindingIdCollision)
{
    PreparedWeightStore store(ModelContextId{99});

    auto gemm_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{64, 96});
    auto gemm_binding = makeStoreBinding(
        27,
        "blk.0.ffn_gate.weight",
        DeviceId::cuda(0));
    gemm_binding.tensor = gemm_tensor.get();
    store.registerPreparedForTest(
        gemm_binding,
        PreparedWeightKind::CudaInt8PackedGemm,
        DeviceId::cuda(0));

    auto embedding_tensor = makeQ8_0Tensor(64, 96);
    auto original_embedding = makeStoreBinding(
        91,
        "token_embd.weight",
        DeviceId::cuda(0));
    original_embedding.identity.role = WeightRole::Embedding;
    original_embedding.tensor = embedding_tensor.get();
    auto handle = makeEmbeddingHandle(
        embedding_tensor.get(),
        DeviceId::cuda(0));
    store.registerPreparedEmbeddingFromPipeline(
        original_embedding,
        DeviceId::cuda(0),
        &handle);

    auto rematerialized_embedding = makeStoreBinding(
        27,
        "token_embd.weight",
        DeviceId::cuda(0));
    rematerialized_embedding.identity.role = WeightRole::Embedding;
    rematerialized_embedding.tensor = embedding_tensor.get();
    rematerialized_embedding.prepared = PreparedWeightRef{
        ModelContextId{99},
        rematerialized_embedding.binding_id,
        PreparedWeightKind::PreparedEmbedding,
        DeviceId::cuda(0)};

    EXPECT_TRUE(store.adoptPreparedForBinding(
        rematerialized_embedding,
        DeviceId::cuda(0)));
    EXPECT_FALSE(store.preparedRefForBinding(
        rematerialized_embedding.binding_id,
        DeviceId::cuda(0)).has_value())
        << "Untyped lookup must reject a cross-kind binding-id collision";

    const auto embedding_ref = store.preparedRefForBinding(
        rematerialized_embedding.binding_id,
        DeviceId::cuda(0),
        PreparedWeightKind::PreparedEmbedding);
    ASSERT_TRUE(embedding_ref.has_value());
    EXPECT_EQ(embedding_ref->kind, PreparedWeightKind::PreparedEmbedding);
    EXPECT_NE(store.embeddingHandle(*embedding_ref), nullptr);

    const auto gemm_ref = store.preparedRefForBinding(
        rematerialized_embedding.binding_id,
        DeviceId::cuda(0),
        PreparedWeightKind::CudaInt8PackedGemm);
    ASSERT_TRUE(gemm_ref.has_value());
    EXPECT_EQ(gemm_ref->kind, PreparedWeightKind::CudaInt8PackedGemm);
}

TEST(Test__PreparedWeightStore, SameBindingIdRetainsDistinctEmbeddingEntriesPerDevice)
{
    PreparedWeightStore store(ModelContextId{99});
    auto cuda0_tensor = makeQ8_0Tensor(64, 96);
    auto cuda1_tensor = makeQ8_0Tensor(64, 96);

    auto cuda0_binding = makeStoreBinding(15, "token_embd.weight", DeviceId::cuda(0));
    cuda0_binding.identity.role = WeightRole::Embedding;
    cuda0_binding.tensor = cuda0_tensor.get();
    auto cuda1_binding = makeStoreBinding(15, "token_embd.weight", DeviceId::cuda(1));
    cuda1_binding.identity.role = WeightRole::Embedding;
    cuda1_binding.tensor = cuda1_tensor.get();

    auto cuda0_handle = makeEmbeddingHandle(cuda0_tensor.get(), DeviceId::cuda(0));
    auto cuda1_handle = makeEmbeddingHandle(cuda1_tensor.get(), DeviceId::cuda(1));
    const auto cuda0_ref = store.registerPreparedEmbeddingFromPipeline(
        cuda0_binding, DeviceId::cuda(0), &cuda0_handle);
    const auto cuda1_ref = store.registerPreparedEmbeddingFromPipeline(
        cuda1_binding, DeviceId::cuda(1), &cuda1_handle);

    EXPECT_TRUE(store.contains(cuda0_ref));
    EXPECT_TRUE(store.contains(cuda1_ref));
    ASSERT_NE(store.embeddingHandle(cuda0_ref), nullptr);
    ASSERT_NE(store.embeddingHandle(cuda1_ref), nullptr);
    EXPECT_NE(store.embeddingHandle(cuda0_ref), store.embeddingHandle(cuda1_ref));
    EXPECT_EQ(store.embeddingHandle(cuda0_ref)->device_id, DeviceId::cuda(0));
    EXPECT_EQ(store.embeddingHandle(cuda1_ref)->device_id, DeviceId::cuda(1));
}

TEST(Test__PreparedWeightStore, PreparedEmbeddingPublicationDelegatesThroughTensorSlice)
{
    PreparedWeightStore store(ModelContextId{99});
    auto inner = std::shared_ptr<TensorBase>(makeQ8_0Tensor(64, 96).release());
    TensorSlice slice(
        inner,
        SliceMetadata::forRowParallel(64, 96, 0, 1, true));
    auto binding = makeStoreBinding(16, "token_embd.weight", DeviceId::cuda(0));
    binding.identity.role = WeightRole::Embedding;
    binding.tensor = &slice;
    auto handle = makeEmbeddingHandle(&slice, DeviceId::cuda(0));

    ASSERT_FALSE(inner->hasPreparedDeviceState());
    ASSERT_FALSE(slice.hasPreparedDeviceState());
    store.registerPreparedEmbeddingFromPipeline(binding, DeviceId::cuda(0), &handle);

    EXPECT_TRUE(inner->hasPreparedDeviceState());
    EXPECT_TRUE(slice.hasPreparedDeviceState());
}

TEST(Test__PreparedWeightStore, RejectsInvalidPreparedEmbeddingIdentity)
{
    PreparedWeightStore store(ModelContextId{99});
    auto tensor = makeQ8_0Tensor(64, 96);
    auto other = makeQ8_0Tensor(64, 96);
    auto binding = makeStoreBinding(17, "token_embd.weight", DeviceId::cuda(0));
    binding.identity.role = WeightRole::Embedding;
    binding.tensor = tensor.get();

    EXPECT_THROW(
        store.registerPreparedEmbeddingFromPipeline(binding, DeviceId::cuda(0), nullptr),
        std::runtime_error);

    auto wrong_tensor = makeEmbeddingHandle(other.get(), DeviceId::cuda(0));
    EXPECT_THROW(
        store.registerPreparedEmbeddingFromPipeline(binding, DeviceId::cuda(0), &wrong_tensor),
        std::runtime_error);

    auto wrong_device = makeEmbeddingHandle(tensor.get(), DeviceId::cuda(1));
    EXPECT_THROW(
        store.registerPreparedEmbeddingFromPipeline(binding, DeviceId::cuda(0), &wrong_device),
        std::runtime_error);
}

TEST(Test__PreparedWeightStore, ResolvesSlicedGemmKernelByPreparedRef)
{
    PreparedWeightStore store(ModelContextId{99});
    auto tensor = makeQ8_0Tensor(1024, 896);
    auto binding = makeStoreBinding(13, "blk.0.ffn_down.weight", DeviceId::cpu());
    binding.tensor = tensor.get();

    auto ref = store.registerPreparedForTest(
        binding, PreparedWeightKind::CpuPackedGemm, DeviceId::cpu());

    auto *first = store.slicedGemmKernel(ref, 0, 512);
    ASSERT_NE(first, nullptr);
    auto *again = store.slicedGemmKernel(ref, 0, 512);
    EXPECT_EQ(again, first);

    auto wrong_ref = ref;
    wrong_ref.binding_id = 999;
    EXPECT_EQ(store.slicedGemmKernel(wrong_ref, 0, 512), nullptr);
}

TEST(Test__PreparedWeightStore, TestHarnessBuildsExecutablePreparedGemmFixture)
{
    auto tensor = makeQ8_0Tensor(128, 96);
    auto fixture = makePreparedGemmFixture(
        tensor.get(),
        DeviceId::cpu(),
        "blk.0.ffn_gate.weight");

    EXPECT_TRUE(fixture.store->contains(fixture.ref));
    EXPECT_EQ(fixture.ref.binding_id, fixture.binding.binding_id);
    EXPECT_EQ(fixture.binding.tensor, tensor.get());
    EXPECT_NE(fixture.store->gemmKernel(fixture.ref), nullptr);
}

TEST(Test__PreparedWeightStore, TestHarnessBuildsRegisteredGateUpFixture)
{
    auto gate = makeQ8_0Tensor(128, 96);
    auto up = makeQ8_0Tensor(128, 96);
    auto fixture = makePreparedGateUpFixture(
        gate.get(),
        up.get(),
        DeviceId::cpu(),
        3);

    EXPECT_TRUE(fixture.store->contains(fixture.gate_ref));
    EXPECT_TRUE(fixture.store->contains(fixture.up_ref));
    EXPECT_EQ(fixture.gate_binding.identity.canonical_name, "blk.3.ffn_gate.weight");
    EXPECT_EQ(fixture.up_binding.identity.canonical_name, "blk.3.ffn_up.weight");
    EXPECT_NE(fixture.store->gemmKernel(fixture.gate_ref), nullptr);
    EXPECT_NE(fixture.store->gemmKernel(fixture.up_ref), nullptr);
}

TEST(Test__PreparedWeightStore, RejectsZeroBindingId)
{
    PreparedWeightStore store(ModelContextId{99});
    auto binding = makeStoreBinding(0, "blk.0.ffn_up.weight", DeviceId::cpu());
    EXPECT_THROW(store.registerPreparedForTest(
                     binding, PreparedWeightKind::CpuPackedGemm, DeviceId::cpu()),
                 std::runtime_error);
}

TEST(Test__PreparedWeightStore, RejectsMismatchedBindingModelId)
{
    PreparedWeightStore store(ModelContextId{99});
    auto binding = makeStoreBinding(9, "blk.0.ffn_up.weight", DeviceId::cpu());
    binding.identity.model_id = ModelContextId{100};

    EXPECT_THROW(store.registerPreparedForTest(
                     binding, PreparedWeightKind::CpuPackedGemm, DeviceId::cpu()),
                 std::runtime_error);
}

TEST(Test__PreparedWeightStore, RejectsEmptyPreparedKind)
{
    PreparedWeightStore store(ModelContextId{99});
    auto binding = makeStoreBinding(10, "blk.0.ffn_up.weight", DeviceId::cpu());

    EXPECT_THROW(store.registerPreparedForTest(
                     binding, PreparedWeightKind::None, DeviceId::cpu()),
                 std::runtime_error);
}
