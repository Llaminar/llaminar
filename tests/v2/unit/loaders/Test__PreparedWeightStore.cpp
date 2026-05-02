#include <gtest/gtest.h>

#include "loaders/PreparedWeightStore.h"

using namespace llaminar2;

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
