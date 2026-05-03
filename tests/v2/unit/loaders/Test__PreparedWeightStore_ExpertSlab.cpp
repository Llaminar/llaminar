#include <gtest/gtest.h>

#include "loaders/PreparedWeightStore.h"

using namespace llaminar2;

namespace
{
    ExpertSlabDescriptor makeDescriptor(
        int layer_idx, WeightRole role, int num_experts, DeviceId device = DeviceId::cpu())
    {
        ExpertSlabDescriptor desc;
        desc.layer_idx = layer_idx;
        desc.role = role;
        desc.device = device;
        desc.num_experts = num_experts;
        desc.local_expert_start = 0;
        desc.local_expert_count = num_experts;
        desc.rows_per_expert = 128;
        desc.cols_per_expert = 64;
        return desc;
    }

    ExpertArrival makeArrival(int expert_id, ITensorGemm *engine,
                              WeightDerivationKind derivation = WeightDerivationKind::ExpertSlice,
                              std::optional<DeviceId> source_device = std::nullopt)
    {
        ExpertArrival arrival;
        arrival.expert_id = expert_id;
        arrival.engine = engine;
        arrival.engine_lifetime = nullptr;
        arrival.view_lifetime = nullptr;
        arrival.derivation = derivation;
        arrival.source_device = source_device;
        return arrival;
    }
}

// ---------------------------------------------------------------------------
// 1. Registration basics
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, RegisterSlab_ValidDescriptor)
{
    PreparedWeightStore store(ModelContextId{1});
    auto desc = makeDescriptor(3, WeightRole::MoEExpertGate, 8);

    auto ref = store.registerExpertSlab(desc);

    EXPECT_EQ(ref.model_id.value, 1u);
    EXPECT_NE(ref.slab_id, 0u);
    EXPECT_EQ(ref.layer_idx, 3);
    EXPECT_EQ(ref.role, WeightRole::MoEExpertGate);
    EXPECT_EQ(ref.device, DeviceId::cpu());
    EXPECT_EQ(store.expertSlabCount(), 1u);
}

TEST(Test__PreparedWeightStore_ExpertSlab, RegisterSlab_InvalidNumExperts_Throws)
{
    PreparedWeightStore store(ModelContextId{1});
    auto desc = makeDescriptor(0, WeightRole::MoEExpertGate, /*num_experts=*/0);

    EXPECT_THROW(store.registerExpertSlab(desc), std::runtime_error);
    EXPECT_EQ(store.expertSlabCount(), 0u);
}

TEST(Test__PreparedWeightStore_ExpertSlab, RegisterSlab_InvalidLayerIdx_Throws)
{
    PreparedWeightStore store(ModelContextId{1});
    auto desc = makeDescriptor(/*layer_idx=*/-1, WeightRole::MoEExpertGate, 4);

    EXPECT_THROW(store.registerExpertSlab(desc), std::runtime_error);
    EXPECT_EQ(store.expertSlabCount(), 0u);
}

// ---------------------------------------------------------------------------
// 2. Lookup before population
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, ExpertGemmKernel_BeforePopulation_ReturnsNull)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(store.expertGemmKernel(ref, i), nullptr);
}

// ---------------------------------------------------------------------------
// 3. Arrival registration
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, RegisterArrivedExperts_PopulatesSlots)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 8));

    auto *eng0 = reinterpret_cast<ITensorGemm *>(0x1000);
    auto *eng3 = reinterpret_cast<ITensorGemm *>(0x2000);
    auto *eng7 = reinterpret_cast<ITensorGemm *>(0x3000);

    std::vector<ExpertArrival> arrivals = {
        makeArrival(0, eng0),
        makeArrival(3, eng3),
        makeArrival(7, eng7),
    };

    auto actually_new = store.registerArrivedExperts(ref, arrivals);

    ASSERT_EQ(actually_new.size(), 3u);
    EXPECT_EQ(store.expertGemmKernel(ref, 0), eng0);
    EXPECT_EQ(store.expertGemmKernel(ref, 3), eng3);
    EXPECT_EQ(store.expertGemmKernel(ref, 7), eng7);
    EXPECT_EQ(store.expertGemmKernel(ref, 1), nullptr); // Not populated
}

TEST(Test__PreparedWeightStore_ExpertSlab, RegisterArrivedExperts_Idempotent)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    auto *eng_first = reinterpret_cast<ITensorGemm *>(0x1000);
    auto *eng_second = reinterpret_cast<ITensorGemm *>(0x2000);

    auto new1 = store.registerArrivedExperts(ref, {makeArrival(2, eng_first)});
    ASSERT_EQ(new1.size(), 1u);
    EXPECT_EQ(new1[0], 2);

    // Second arrival for same expert_id should be skipped
    auto new2 = store.registerArrivedExperts(ref, {makeArrival(2, eng_second)});
    EXPECT_TRUE(new2.empty());

    // Engine should still be the first one
    EXPECT_EQ(store.expertGemmKernel(ref, 2), eng_first);
}

TEST(Test__PreparedWeightStore_ExpertSlab, RegisterArrivedExperts_OutOfRange_Ignored)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    auto *eng = reinterpret_cast<ITensorGemm *>(0x1000);

    std::vector<ExpertArrival> arrivals = {
        makeArrival(-1, eng),  // Negative
        makeArrival(4, eng),   // Equal to num_experts
        makeArrival(99, eng),  // Way out of range
        makeArrival(1, eng),   // Valid
    };

    auto actually_new = store.registerArrivedExperts(ref, arrivals);
    ASSERT_EQ(actually_new.size(), 1u);
    EXPECT_EQ(actually_new[0], 1);
    EXPECT_EQ(store.totalPopulatedExperts(), 1u);
}

TEST(Test__PreparedWeightStore_ExpertSlab, RegisterArrivedExperts_InvalidSlab_Throws)
{
    PreparedWeightStore store(ModelContextId{1});

    ExpertSlabRef bogus;
    bogus.model_id = ModelContextId{1};
    bogus.slab_id = 999;

    auto *eng = reinterpret_cast<ITensorGemm *>(0x1000);
    EXPECT_THROW(
        store.registerArrivedExperts(bogus, {makeArrival(0, eng)}),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// 4. Release departed experts
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, ReleaseDepartedExperts_ClearsSlots)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    auto *eng0 = reinterpret_cast<ITensorGemm *>(0x1000);
    auto *eng1 = reinterpret_cast<ITensorGemm *>(0x2000);
    auto *eng2 = reinterpret_cast<ITensorGemm *>(0x3000);

    store.registerArrivedExperts(ref, {
        makeArrival(0, eng0),
        makeArrival(1, eng1),
        makeArrival(2, eng2),
    });

    store.releaseDepartedExperts(ref, {0, 2});

    EXPECT_EQ(store.expertGemmKernel(ref, 0), nullptr);
    EXPECT_EQ(store.expertGemmKernel(ref, 1), eng1);
    EXPECT_EQ(store.expertGemmKernel(ref, 2), nullptr);
    EXPECT_EQ(store.totalPopulatedExperts(), 1u);
}

TEST(Test__PreparedWeightStore_ExpertSlab, ReleaseDepartedExperts_OutOfRange_Ignored)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    auto *eng = reinterpret_cast<ITensorGemm *>(0x1000);
    store.registerArrivedExperts(ref, {makeArrival(0, eng)});

    // Should not throw or affect valid entries
    store.releaseDepartedExperts(ref, {-1, 4, 100});

    EXPECT_EQ(store.expertGemmKernel(ref, 0), eng);
    EXPECT_EQ(store.totalPopulatedExperts(), 1u);
}

TEST(Test__PreparedWeightStore_ExpertSlab, ReleaseDepartedExperts_InvalidSlab_NoThrow)
{
    PreparedWeightStore store(ModelContextId{1});

    ExpertSlabRef bogus;
    bogus.model_id = ModelContextId{1};
    bogus.slab_id = 999;

    EXPECT_NO_THROW(store.releaseDepartedExperts(bogus, {0, 1}));
}

// ---------------------------------------------------------------------------
// 5. Availability mask
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, ExpertAvailabilityMask_ReflectsState)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    // Initially all false
    auto mask0 = store.expertAvailabilityMask(ref);
    ASSERT_EQ(mask0.size(), 4u);
    for (bool b : mask0)
        EXPECT_FALSE(b);

    // Populate experts 0 and 2
    auto *eng0 = reinterpret_cast<ITensorGemm *>(0x1000);
    auto *eng2 = reinterpret_cast<ITensorGemm *>(0x2000);
    store.registerArrivedExperts(ref, {makeArrival(0, eng0), makeArrival(2, eng2)});

    auto mask1 = store.expertAvailabilityMask(ref);
    ASSERT_EQ(mask1.size(), 4u);
    EXPECT_TRUE(mask1[0]);
    EXPECT_FALSE(mask1[1]);
    EXPECT_TRUE(mask1[2]);
    EXPECT_FALSE(mask1[3]);

    // Release expert 0
    store.releaseDepartedExperts(ref, {0});

    auto mask2 = store.expertAvailabilityMask(ref);
    EXPECT_FALSE(mask2[0]);
    EXPECT_TRUE(mask2[2]);
}

TEST(Test__PreparedWeightStore_ExpertSlab, ExpertAvailabilityMask_InvalidSlab_ReturnsEmpty)
{
    PreparedWeightStore store(ModelContextId{1});

    ExpertSlabRef bogus;
    bogus.model_id = ModelContextId{1};
    bogus.slab_id = 999;

    auto mask = store.expertAvailabilityMask(bogus);
    EXPECT_TRUE(mask.empty());
}

// ---------------------------------------------------------------------------
// 6. Release entire slab
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, ReleaseExpertSlab_RemovesSlab)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    auto *eng = reinterpret_cast<ITensorGemm *>(0x1000);
    store.registerArrivedExperts(ref, {makeArrival(0, eng)});
    EXPECT_EQ(store.expertSlabCount(), 1u);
    EXPECT_EQ(store.totalPopulatedExperts(), 1u);

    store.releaseExpertSlab(ref);

    EXPECT_EQ(store.expertSlabCount(), 0u);
    EXPECT_EQ(store.totalPopulatedExperts(), 0u);
    EXPECT_EQ(store.expertGemmKernel(ref, 0), nullptr);
    EXPECT_TRUE(store.expertAvailabilityMask(ref).empty());
}

// ---------------------------------------------------------------------------
// 7. Multiple slabs
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, MultipleSlabs_Independent)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref_layer0 = store.registerExpertSlab(
        makeDescriptor(0, WeightRole::MoEExpertGate, 4));
    auto ref_layer1 = store.registerExpertSlab(
        makeDescriptor(1, WeightRole::MoEExpertDown, 4));

    EXPECT_EQ(store.expertSlabCount(), 2u);
    EXPECT_NE(ref_layer0.slab_id, ref_layer1.slab_id);

    auto *eng_a = reinterpret_cast<ITensorGemm *>(0x1000);
    auto *eng_b = reinterpret_cast<ITensorGemm *>(0x2000);

    store.registerArrivedExperts(ref_layer0, {makeArrival(0, eng_a)});
    store.registerArrivedExperts(ref_layer1, {makeArrival(0, eng_b)});

    EXPECT_EQ(store.expertGemmKernel(ref_layer0, 0), eng_a);
    EXPECT_EQ(store.expertGemmKernel(ref_layer1, 0), eng_b);

    // Releasing one slab doesn't affect the other
    store.releaseExpertSlab(ref_layer0);
    EXPECT_EQ(store.expertSlabCount(), 1u);
    EXPECT_EQ(store.expertGemmKernel(ref_layer0, 0), nullptr);
    EXPECT_EQ(store.expertGemmKernel(ref_layer1, 0), eng_b);
}

// ---------------------------------------------------------------------------
// 8. Cross-slab counting
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, TotalPopulatedExperts_CountsAcrossSlabs)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref1 = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 8));
    auto ref2 = store.registerExpertSlab(makeDescriptor(1, WeightRole::MoEExpertUp, 8));

    auto *eng = reinterpret_cast<ITensorGemm *>(0x1000);

    // 3 experts in slab 1
    store.registerArrivedExperts(ref1, {
        makeArrival(0, eng), makeArrival(1, eng), makeArrival(2, eng)});
    // 5 experts in slab 2
    store.registerArrivedExperts(ref2, {
        makeArrival(0, eng), makeArrival(1, eng), makeArrival(2, eng),
        makeArrival(3, eng), makeArrival(4, eng)});

    EXPECT_EQ(store.totalPopulatedExperts(), 8u);

    // Release 1 from slab 1
    store.releaseDepartedExperts(ref1, {0});
    EXPECT_EQ(store.totalPopulatedExperts(), 7u);
}

// ---------------------------------------------------------------------------
// 9. Derivation and source device
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, ReplicaDerivation_Preserved)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    auto *eng = reinterpret_cast<ITensorGemm *>(0x1000);
    auto arrival = makeArrival(
        1, eng, WeightDerivationKind::RebalancedExpertReplica);

    auto actually_new = store.registerArrivedExperts(ref, {arrival});
    ASSERT_EQ(actually_new.size(), 1u);

    // Engine is available
    EXPECT_EQ(store.expertGemmKernel(ref, 1), eng);

    // Availability mask reflects it
    auto mask = store.expertAvailabilityMask(ref);
    EXPECT_FALSE(mask[0]);
    EXPECT_TRUE(mask[1]);
    EXPECT_FALSE(mask[2]);
    EXPECT_FALSE(mask[3]);
}

TEST(Test__PreparedWeightStore_ExpertSlab, SourceDevice_Preserved)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    auto *eng = reinterpret_cast<ITensorGemm *>(0x1000);
    auto arrival = makeArrival(
        2, eng, WeightDerivationKind::RebalancedExpertReplica, DeviceId::cuda(1));

    auto actually_new = store.registerArrivedExperts(ref, {arrival});
    ASSERT_EQ(actually_new.size(), 1u);

    // Expert is queryable
    EXPECT_EQ(store.expertGemmKernel(ref, 2), eng);
    EXPECT_EQ(store.totalPopulatedExperts(), 1u);
}

// ---------------------------------------------------------------------------
// 10. Out-of-range lookups on valid slab
// ---------------------------------------------------------------------------

TEST(Test__PreparedWeightStore_ExpertSlab, ExpertGemmKernel_OutOfRange_ReturnsNull)
{
    PreparedWeightStore store(ModelContextId{1});
    auto ref = store.registerExpertSlab(makeDescriptor(0, WeightRole::MoEExpertGate, 4));

    EXPECT_EQ(store.expertGemmKernel(ref, -1), nullptr);
    EXPECT_EQ(store.expertGemmKernel(ref, 4), nullptr);
    EXPECT_EQ(store.expertGemmKernel(ref, 100), nullptr);
}
