#include <gtest/gtest.h>

#include "loaders/PreparedWeightStore.h"
#include "loaders/ExpertSlabTypes.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

namespace
{
    /// Sentinel addresses as fake engine pointers (never dereferenced).
    ITensorGemm *fakeEngine(int id)
    {
        return reinterpret_cast<ITensorGemm *>(static_cast<uintptr_t>(0x1000 + id * 0x100));
    }

    ExpertSlabDescriptor makeDesc(int layer, WeightRole role, int num_experts = 64)
    {
        ExpertSlabDescriptor desc;
        desc.layer_idx = layer;
        desc.role = role;
        desc.device = DeviceId::cpu();
        desc.num_experts = num_experts;
        desc.local_expert_start = 0;
        desc.local_expert_count = num_experts;
        desc.rows_per_expert = 2048;
        desc.cols_per_expert = 896;
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

    /// Populate a slab with sequential experts [0, count).
    void populateAll(PreparedWeightStore &store, const ExpertSlabRef &ref, int count)
    {
        std::vector<ExpertArrival> arrivals;
        arrivals.reserve(count);
        for (int i = 0; i < count; ++i)
            arrivals.push_back(makeArrival(i, fakeEngine(i)));
        store.registerArrivedExperts(ref, arrivals);
    }
}

// ---------------------------------------------------------------------------
// 1. Simulates what prepareGemmEngines does: 3 slabs (gate/up/down) per layer
// ---------------------------------------------------------------------------

TEST(Test__MoEWeightService_DualRegistration, DualRegistration_ThreeSlabsPerLayer)
{
    PreparedWeightStore store(ModelContextId{42});

    constexpr int layer = 5;
    constexpr int num_experts = 8;

    auto gate_ref = store.registerExpertSlab(makeDesc(layer, WeightRole::MoEExpertGate, num_experts));
    auto up_ref = store.registerExpertSlab(makeDesc(layer, WeightRole::MoEExpertUp, num_experts));
    auto down_ref = store.registerExpertSlab(makeDesc(layer, WeightRole::MoEExpertDown, num_experts));

    EXPECT_EQ(store.expertSlabCount(), 3u);

    // Populate all experts in each slab
    populateAll(store, gate_ref, num_experts);
    populateAll(store, up_ref, num_experts);
    populateAll(store, down_ref, num_experts);

    EXPECT_EQ(store.totalPopulatedExperts(), static_cast<size_t>(num_experts * 3));

    // Verify each slab returns the correct engine pointer
    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(store.expertGemmKernel(gate_ref, e), fakeEngine(e));
        EXPECT_EQ(store.expertGemmKernel(up_ref, e), fakeEngine(e));
        EXPECT_EQ(store.expertGemmKernel(down_ref, e), fakeEngine(e));
    }
}

// ---------------------------------------------------------------------------
// 2. Multi-layer: 3 layers × 3 roles = 9 slabs coexist
// ---------------------------------------------------------------------------

TEST(Test__MoEWeightService_DualRegistration, DualRegistration_MultiLayerAccumulates)
{
    PreparedWeightStore store(ModelContextId{42});

    constexpr int num_layers = 3;
    constexpr int num_experts = 4;
    constexpr WeightRole roles[] = {
        WeightRole::MoEExpertGate,
        WeightRole::MoEExpertUp,
        WeightRole::MoEExpertDown,
    };

    std::vector<ExpertSlabRef> refs;
    for (int layer = 0; layer < num_layers; ++layer)
    {
        for (auto role : roles)
        {
            auto ref = store.registerExpertSlab(makeDesc(layer, role, num_experts));
            refs.push_back(ref);
        }
    }

    EXPECT_EQ(store.expertSlabCount(), 9u);

    // Populate all
    for (auto &ref : refs)
        populateAll(store, ref, num_experts);

    EXPECT_EQ(store.totalPopulatedExperts(), static_cast<size_t>(9 * num_experts));

    // Verify each slab is independently addressable
    for (size_t i = 0; i < refs.size(); ++i)
    {
        for (int e = 0; e < num_experts; ++e)
            EXPECT_NE(store.expertGemmKernel(refs[i], e), nullptr);
    }

    // Verify slab IDs are all unique
    for (size_t i = 0; i < refs.size(); ++i)
        for (size_t j = i + 1; j < refs.size(); ++j)
            EXPECT_NE(refs[i].slab_id, refs[j].slab_id);
}

// ---------------------------------------------------------------------------
// 3. Both-paths-agree: ctx.prepared_gate_gemm[e] == store.expertGemmKernel()
// ---------------------------------------------------------------------------

TEST(Test__MoEWeightService_DualRegistration, DualRegistration_AgreementValidation)
{
    PreparedWeightStore store(ModelContextId{42});

    constexpr int num_experts = 8;

    // Simulate ctx.prepared_gate_gemm / up / down vectors
    std::vector<ITensorGemm *> ctx_gate(num_experts);
    std::vector<ITensorGemm *> ctx_up(num_experts);
    std::vector<ITensorGemm *> ctx_down(num_experts);

    for (int e = 0; e < num_experts; ++e)
    {
        ctx_gate[e] = fakeEngine(e);
        ctx_up[e] = fakeEngine(100 + e);
        ctx_down[e] = fakeEngine(200 + e);
    }

    // Register slabs (mirrors what prepareGemmEngines does)
    auto gate_ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertGate, num_experts));
    auto up_ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertUp, num_experts));
    auto down_ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertDown, num_experts));

    // Populate from same engine pointers that live in ctx vectors
    std::vector<ExpertArrival> gate_arrivals, up_arrivals, down_arrivals;
    for (int e = 0; e < num_experts; ++e)
    {
        gate_arrivals.push_back(makeArrival(e, ctx_gate[e]));
        up_arrivals.push_back(makeArrival(e, ctx_up[e]));
        down_arrivals.push_back(makeArrival(e, ctx_down[e]));
    }

    store.registerArrivedExperts(gate_ref, gate_arrivals);
    store.registerArrivedExperts(up_ref, up_arrivals);
    store.registerArrivedExperts(down_ref, down_arrivals);

    // *** The dual-path agreement assertion ***
    // store.expertGemmKernel() must return the exact same pointer as ctx vectors
    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(store.expertGemmKernel(gate_ref, e), ctx_gate[e])
            << "Gate engine mismatch at expert " << e;
        EXPECT_EQ(store.expertGemmKernel(up_ref, e), ctx_up[e])
            << "Up engine mismatch at expert " << e;
        EXPECT_EQ(store.expertGemmKernel(down_ref, e), ctx_down[e])
            << "Down engine mismatch at expert " << e;
    }
}

// ---------------------------------------------------------------------------
// 4. Rebalance arrivals with ReplicaDerivation
// ---------------------------------------------------------------------------

TEST(Test__MoEWeightService_DualRegistration, Rebalance_ArrivalsWithReplicaDerivation)
{
    PreparedWeightStore store(ModelContextId{42});

    constexpr int num_experts = 8;
    auto ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertGate, num_experts));

    // Initial population: experts 0-3
    std::vector<ExpertArrival> initial;
    for (int e = 0; e < 4; ++e)
        initial.push_back(makeArrival(e, fakeEngine(e)));
    store.registerArrivedExperts(ref, initial);
    EXPECT_EQ(store.totalPopulatedExperts(), 4u);

    // Rebalance: experts 4-5 arrive as replicas from cuda:1
    std::vector<ExpertArrival> rebalanced;
    for (int e = 4; e < 6; ++e)
    {
        rebalanced.push_back(makeArrival(
            e, fakeEngine(e),
            WeightDerivationKind::RebalancedExpertReplica,
            DeviceId::cuda(1)));
    }

    auto actually_new = store.registerArrivedExperts(ref, rebalanced);
    ASSERT_EQ(actually_new.size(), 2u);
    EXPECT_EQ(store.totalPopulatedExperts(), 6u);

    // All six experts are accessible
    for (int e = 0; e < 6; ++e)
        EXPECT_EQ(store.expertGemmKernel(ref, e), fakeEngine(e));

    // Experts 6-7 are still empty
    EXPECT_EQ(store.expertGemmKernel(ref, 6), nullptr);
    EXPECT_EQ(store.expertGemmKernel(ref, 7), nullptr);

    // Availability mask reflects all six
    auto mask = store.expertAvailabilityMask(ref);
    ASSERT_EQ(mask.size(), static_cast<size_t>(num_experts));
    for (int e = 0; e < 6; ++e)
        EXPECT_TRUE(mask[e]) << "Expert " << e << " should be available";
    for (int e = 6; e < num_experts; ++e)
        EXPECT_FALSE(mask[e]) << "Expert " << e << " should NOT be available";
}

// ---------------------------------------------------------------------------
// 5. Departed experts cleared
// ---------------------------------------------------------------------------

TEST(Test__MoEWeightService_DualRegistration, Rebalance_DepartedExpertsCleared)
{
    PreparedWeightStore store(ModelContextId{42});

    constexpr int num_experts = 8;
    auto ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertGate, num_experts));

    // Populate all 8
    populateAll(store, ref, num_experts);
    EXPECT_EQ(store.totalPopulatedExperts(), 8u);

    // Depart experts 2 and 5
    store.releaseDepartedExperts(ref, {2, 5});

    EXPECT_EQ(store.totalPopulatedExperts(), 6u);
    EXPECT_EQ(store.expertGemmKernel(ref, 2), nullptr);
    EXPECT_EQ(store.expertGemmKernel(ref, 5), nullptr);

    // Others still alive
    EXPECT_EQ(store.expertGemmKernel(ref, 0), fakeEngine(0));
    EXPECT_EQ(store.expertGemmKernel(ref, 1), fakeEngine(1));
    EXPECT_EQ(store.expertGemmKernel(ref, 3), fakeEngine(3));
    EXPECT_EQ(store.expertGemmKernel(ref, 4), fakeEngine(4));
    EXPECT_EQ(store.expertGemmKernel(ref, 6), fakeEngine(6));
    EXPECT_EQ(store.expertGemmKernel(ref, 7), fakeEngine(7));

    // Availability mask
    auto mask = store.expertAvailabilityMask(ref);
    EXPECT_FALSE(mask[2]);
    EXPECT_FALSE(mask[5]);
    for (int e : {0, 1, 3, 4, 6, 7})
        EXPECT_TRUE(mask[e]);
}

// ---------------------------------------------------------------------------
// 6. Null store safety (trivial — validates the pattern is safe)
// ---------------------------------------------------------------------------

TEST(Test__MoEWeightService_DualRegistration, NullStore_NoRegistration)
{
    // When prepareGemmEngines receives ctx.prepared_store == nullptr,
    // it skips store registration. This test just confirms the code
    // pattern: a null pointer guard is safe.
    PreparedWeightStore *store = nullptr;

    // This is the guard pattern used in prepareGemmEngines:
    if (store != nullptr)
    {
        // Would register slabs here — should NOT execute
        FAIL() << "Should not reach store registration with null store";
    }

    SUCCEED();
}

// ---------------------------------------------------------------------------
// 7. Partial population — mask reflects only populated experts
// ---------------------------------------------------------------------------

TEST(Test__MoEWeightService_DualRegistration, PartialPopulation_MaskReflectsAvailability)
{
    PreparedWeightStore store(ModelContextId{42});

    constexpr int num_experts = 64;
    auto ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertGate, num_experts));

    // Populate only even experts: 0, 2, 4, 6
    std::vector<ExpertArrival> arrivals;
    for (int e = 0; e < 8; e += 2)
        arrivals.push_back(makeArrival(e, fakeEngine(e)));
    store.registerArrivedExperts(ref, arrivals);

    EXPECT_EQ(store.totalPopulatedExperts(), 4u);

    auto mask = store.expertAvailabilityMask(ref);
    ASSERT_EQ(mask.size(), static_cast<size_t>(num_experts));

    // Experts 0, 2, 4, 6 are true
    EXPECT_TRUE(mask[0]);
    EXPECT_TRUE(mask[2]);
    EXPECT_TRUE(mask[4]);
    EXPECT_TRUE(mask[6]);

    // Odd experts and remaining are false
    EXPECT_FALSE(mask[1]);
    EXPECT_FALSE(mask[3]);
    EXPECT_FALSE(mask[5]);
    EXPECT_FALSE(mask[7]);

    // All experts from 8-63 should be false
    for (int e = 8; e < num_experts; ++e)
        EXPECT_FALSE(mask[e]) << "Expert " << e << " should NOT be available";
}
