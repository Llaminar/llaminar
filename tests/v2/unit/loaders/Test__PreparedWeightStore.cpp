#include <gtest/gtest.h>

#include "loaders/PreparedWeightStore.h"
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

    auto ref = store.registerPreparedEmbeddingFromPipeline(
        binding, DeviceId::cuda(0), nullptr);

    EXPECT_EQ(ref.kind, PreparedWeightKind::PreparedEmbedding);
    EXPECT_TRUE(store.contains(ref));

    auto by_binding = store.preparedRefForBinding(binding.binding_id, DeviceId::cuda(0));
    ASSERT_TRUE(by_binding.has_value());
    EXPECT_EQ(by_binding->kind, PreparedWeightKind::PreparedEmbedding);

    auto stored = store.binding(ref);
    ASSERT_TRUE(stored.has_value());
    EXPECT_EQ(stored->identity.role, WeightRole::Embedding);
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
