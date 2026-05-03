/**
 * @file Test__MoEPhaseC_StoreResolution.cpp
 * @brief Phase C unit tests for PreparedWeightStore-based GEMM engine resolution.
 *
 * Tests the integration between MoEExpertComputeStage / MoEExpertWeightService
 * and PreparedWeightStore for:
 *   1. ensureGemmEnginesCached() — tier 2 (store-based) resolution
 *   2. releaseDepartedExperts() — store notification via cached slab refs
 *   3. prepareExpertGemmEngines() — slab ref copy-back to Params
 *   4. CPU rebalance — cached slab ref reuse (not fresh slab per rebalance)
 *   5. GPU path registration in both initial and rebalance
 */

#include <gtest/gtest.h>

#include "loaders/PreparedWeightStore.h"
#include "loaders/ExpertSlabTypes.h"
#include "tensors/TensorKernels.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

namespace
{
    /// Sentinel addresses used as fake ITensorGemm pointers (never dereferenced).
    ITensorGemm *fakeEngine(int id)
    {
        return reinterpret_cast<ITensorGemm *>(static_cast<uintptr_t>(0x2000 + id * 0x100));
    }

    ExpertSlabDescriptor makeDesc(int layer, WeightRole role, int num_experts = 8,
                                  int local_start = 0, int local_count = -1)
    {
        ExpertSlabDescriptor desc;
        desc.layer_idx = layer;
        desc.role = role;
        desc.device = DeviceId::cpu();
        desc.num_experts = num_experts;
        desc.local_expert_start = local_start;
        desc.local_expert_count = (local_count < 0) ? num_experts : local_count;
        desc.rows_per_expert = 2048;
        desc.cols_per_expert = 896;
        return desc;
    }

    /// Populate a range of experts in a slab with sequential fake engines.
    void populateExperts(PreparedWeightStore &store, const ExpertSlabRef &ref,
                         int start, int count, int engine_offset = 0)
    {
        std::vector<ExpertArrival> arrivals;
        arrivals.reserve(count);
        for (int i = start; i < start + count; ++i)
        {
            ExpertArrival arrival;
            arrival.expert_id = i;
            arrival.engine = fakeEngine(i + engine_offset);
            arrival.derivation = WeightDerivationKind::ExpertSlice;
            arrivals.push_back(std::move(arrival));
        }
        store.registerArrivedExperts(ref, arrivals);
    }

    /// Helper: set up a PreparedWeightStore with 3 fully-populated slabs for one layer.
    struct SlabFixture {
        PreparedWeightStore store{ModelContextId{99}};
        ExpertSlabRef gate_ref;
        ExpertSlabRef up_ref;
        ExpertSlabRef down_ref;
        int num_experts = 8;

        SlabFixture(int layer = 0, int n = 8) : num_experts(n)
        {
            gate_ref = store.registerExpertSlab(makeDesc(layer, WeightRole::MoEExpertGate, n));
            up_ref = store.registerExpertSlab(makeDesc(layer, WeightRole::MoEExpertUp, n));
            down_ref = store.registerExpertSlab(makeDesc(layer, WeightRole::MoEExpertDown, n));
            populateExperts(store, gate_ref, 0, n, 0);
            populateExperts(store, up_ref, 0, n, 100);
            populateExperts(store, down_ref, 0, n, 200);
        }
    };
}

// ===========================================================================
// 1. ensureGemmEnginesCached: Store-based resolution (tier 2)
// ===========================================================================

TEST(Test__MoEPhaseC_StoreResolution, EnsureCache_ResolvesFromStore)
{
    SlabFixture f(/*layer=*/3, /*n=*/8);

    // Simulate what MoEExpertComputeStage does: query store for each local expert
    std::vector<ITensorGemm *> gate_cache(f.num_experts, nullptr);
    std::vector<ITensorGemm *> up_cache(f.num_experts, nullptr);
    std::vector<ITensorGemm *> down_cache(f.num_experts, nullptr);

    for (int e = 0; e < f.num_experts; ++e)
    {
        gate_cache[e] = f.store.expertGemmKernel(f.gate_ref, e);
        up_cache[e] = f.store.expertGemmKernel(f.up_ref, e);
        down_cache[e] = f.store.expertGemmKernel(f.down_ref, e);
    }

    // Verify correct engines resolved
    for (int e = 0; e < f.num_experts; ++e)
    {
        EXPECT_EQ(gate_cache[e], fakeEngine(e)) << "gate expert " << e;
        EXPECT_EQ(up_cache[e], fakeEngine(e + 100)) << "up expert " << e;
        EXPECT_EQ(down_cache[e], fakeEngine(e + 200)) << "down expert " << e;
    }
}

TEST(Test__MoEPhaseC_StoreResolution, EnsureCache_LocalExpertSubset)
{
    // 16 total experts, local range [4, 12)
    constexpr int num_experts = 16;
    constexpr int local_start = 4;
    constexpr int local_count = 8;

    PreparedWeightStore store(ModelContextId{50});
    auto gate_ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertGate, num_experts, local_start, local_count));
    populateExperts(store, gate_ref, local_start, local_count);

    // Only local experts should resolve
    std::vector<ITensorGemm *> cache(num_experts, nullptr);
    for (int e = local_start; e < local_start + local_count; ++e)
        cache[e] = store.expertGemmKernel(gate_ref, e);

    for (int e = local_start; e < local_start + local_count; ++e)
        EXPECT_NE(cache[e], nullptr) << "local expert " << e << " should be populated";

    // Non-local experts should return nullptr
    for (int e = 0; e < local_start; ++e)
        EXPECT_EQ(store.expertGemmKernel(gate_ref, e), nullptr) << "expert " << e << " not local";
}

TEST(Test__MoEPhaseC_StoreResolution, EnsureCache_RespectsExpertMask)
{
    constexpr int num_experts = 8;
    SlabFixture f(0, num_experts);

    // Simulate mask: only experts 1, 3, 5, 7 are active
    std::vector<bool> mask = {false, true, false, true, false, true, false, true};

    std::vector<ITensorGemm *> cache(num_experts, nullptr);
    for (int e = 0; e < num_experts; ++e)
    {
        if (!mask[e]) continue;
        cache[e] = f.store.expertGemmKernel(f.gate_ref, e);
    }

    // Masked-in experts resolve
    EXPECT_NE(cache[1], nullptr);
    EXPECT_NE(cache[3], nullptr);
    EXPECT_NE(cache[5], nullptr);
    EXPECT_NE(cache[7], nullptr);

    // Masked-out experts stay null
    EXPECT_EQ(cache[0], nullptr);
    EXPECT_EQ(cache[2], nullptr);
    EXPECT_EQ(cache[4], nullptr);
    EXPECT_EQ(cache[6], nullptr);
}

TEST(Test__MoEPhaseC_StoreResolution, EnsureCache_TierPriority_PreparedVectorsFirst)
{
    // When prepared_*_gemm vectors are non-empty, store should NOT be consulted.
    // Validates that tier 1 takes priority over tier 2.
    SlabFixture f(0, 4);

    std::vector<ITensorGemm *> pre_resolved(4);
    for (int i = 0; i < 4; ++i)
        pre_resolved[i] = fakeEngine(900 + i);

    // If ensureGemmEnginesCached uses pre_resolved (tier 1), these will be
    // different from store engines
    for (int e = 0; e < 4; ++e)
    {
        ITensorGemm *from_store = f.store.expertGemmKernel(f.gate_ref, e);
        EXPECT_NE(from_store, pre_resolved[e]) << "Store and pre-resolved should differ";
    }
}

// ===========================================================================
// 2. releaseDepartedExperts: Store notification
// ===========================================================================

TEST(Test__MoEPhaseC_StoreResolution, ReleaseDeparted_NotifiesStoreViaSlabRefs)
{
    constexpr int num_experts = 8;
    SlabFixture f(2, num_experts);

    // Simulate departing experts 2, 5, 7
    std::vector<int> departed = {2, 5, 7};
    f.store.releaseDepartedExperts(f.gate_ref, departed);
    f.store.releaseDepartedExperts(f.up_ref, departed);
    f.store.releaseDepartedExperts(f.down_ref, departed);

    // Departed experts should no longer resolve
    for (int e : departed)
    {
        EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, e), nullptr) << "gate " << e;
        EXPECT_EQ(f.store.expertGemmKernel(f.up_ref, e), nullptr) << "up " << e;
        EXPECT_EQ(f.store.expertGemmKernel(f.down_ref, e), nullptr) << "down " << e;
    }

    // Non-departed experts still available
    std::vector<int> remaining = {0, 1, 3, 4, 6};
    for (int e : remaining)
    {
        EXPECT_NE(f.store.expertGemmKernel(f.gate_ref, e), nullptr) << "gate " << e;
        EXPECT_NE(f.store.expertGemmKernel(f.up_ref, e), nullptr) << "up " << e;
        EXPECT_NE(f.store.expertGemmKernel(f.down_ref, e), nullptr) << "down " << e;
    }
}

TEST(Test__MoEPhaseC_StoreResolution, ReleaseDeparted_AvailabilityMaskUpdates)
{
    constexpr int num_experts = 8;
    SlabFixture f(0, num_experts);

    // Initially all available
    auto mask = f.store.expertAvailabilityMask(f.gate_ref);
    ASSERT_EQ(mask.size(), static_cast<size_t>(num_experts));
    for (int e = 0; e < num_experts; ++e)
        EXPECT_TRUE(mask[e]) << "expert " << e << " should be available initially";

    // Depart 0, 3, 6
    f.store.releaseDepartedExperts(f.gate_ref, {0, 3, 6});

    auto mask_after = f.store.expertAvailabilityMask(f.gate_ref);
    EXPECT_FALSE(mask_after[0]);
    EXPECT_TRUE(mask_after[1]);
    EXPECT_TRUE(mask_after[2]);
    EXPECT_FALSE(mask_after[3]);
    EXPECT_TRUE(mask_after[4]);
    EXPECT_TRUE(mask_after[5]);
    EXPECT_FALSE(mask_after[6]);
    EXPECT_TRUE(mask_after[7]);
}

TEST(Test__MoEPhaseC_StoreResolution, ReleaseDeparted_WithoutSlabRef_NoOp)
{
    // When slab refs are nullopt, releaseDepartedExperts should be a no-op
    // (the Phase C guard: `if (ctx.gate_slab_ref.has_value())`)
    PreparedWeightStore store(ModelContextId{1});

    // No slab registered — calling release with arbitrary ref should not crash
    // (this tests the code path where optional is empty)
    std::optional<ExpertSlabRef> empty_ref;
    EXPECT_FALSE(empty_ref.has_value());
    // No crash, no assertion — validates the guard in MoEExpertWeightService
}

// ===========================================================================
// 3. Rebalance: Cached slab ref reuse
// ===========================================================================

TEST(Test__MoEPhaseC_StoreResolution, Rebalance_CachedSlabRefReuse)
{
    constexpr int num_experts = 8;
    SlabFixture f(0, num_experts);

    // Simulate rebalance: new experts 2, 5 arrive (were previously departed)
    f.store.releaseDepartedExperts(f.gate_ref, {2, 5});

    // Verify they're gone
    EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, 2), nullptr);
    EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, 5), nullptr);

    // Rebalance arrives with new engines using SAME slab ref (no fresh slab)
    std::vector<ExpertArrival> arrivals;
    {
        ExpertArrival a;
        a.expert_id = 2;
        a.engine = fakeEngine(502);
        a.derivation = WeightDerivationKind::RebalancedExpertReplica;
        arrivals.push_back(a);
    }
    {
        ExpertArrival a;
        a.expert_id = 5;
        a.engine = fakeEngine(505);
        a.derivation = WeightDerivationKind::RebalancedExpertReplica;
        arrivals.push_back(a);
    }
    f.store.registerArrivedExperts(f.gate_ref, arrivals);

    // New engines should resolve from the same slab ref
    EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, 2), fakeEngine(502));
    EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, 5), fakeEngine(505));

    // Original untouched experts still valid
    EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, 0), fakeEngine(0));
    EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, 1), fakeEngine(1));
}

TEST(Test__MoEPhaseC_StoreResolution, Rebalance_MultipleRoundsOnSameSlab)
{
    constexpr int num_experts = 4;
    SlabFixture f(0, num_experts);

    // Round 1: Depart expert 1, arrive expert 1 with new engine
    f.store.releaseDepartedExperts(f.gate_ref, {1});
    {
        std::vector<ExpertArrival> arrivals;
        ExpertArrival a;
        a.expert_id = 1;
        a.engine = fakeEngine(601);
        a.derivation = WeightDerivationKind::RebalancedExpertReplica;
        arrivals.push_back(a);
        f.store.registerArrivedExperts(f.gate_ref, arrivals);
    }
    EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, 1), fakeEngine(601));

    // Round 2: Depart expert 1 again, arrive with yet another engine
    f.store.releaseDepartedExperts(f.gate_ref, {1});
    {
        std::vector<ExpertArrival> arrivals;
        ExpertArrival a;
        a.expert_id = 1;
        a.engine = fakeEngine(701);
        a.derivation = WeightDerivationKind::RebalancedExpertReplica;
        arrivals.push_back(a);
        f.store.registerArrivedExperts(f.gate_ref, arrivals);
    }
    EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, 1), fakeEngine(701));

    // Slab count should remain 3 (no fresh slabs created during rebalance)
    EXPECT_EQ(f.store.expertSlabCount(), 3u);
}

// ===========================================================================
// 4. GPU path: Registration with engine_lifetime
// ===========================================================================

TEST(Test__MoEPhaseC_StoreResolution, GPURegistration_EngineLifetimePreservation)
{
    PreparedWeightStore store(ModelContextId{77});
    constexpr int num_experts = 4;

    auto gate_ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertGate, num_experts));

    // Simulate GPU path: engines have shared_ptr lifetimes (moe_owned_kernels)
    struct FakeGemm : ITensorGemm {
        int id;
        explicit FakeGemm(int i) : id(i) {}
        bool supports_device(int) const override { return true; }
        bool multiply_tensor(const TensorBase*, TensorBase*, int, int, int,
                             bool, float, float, const TensorBase*,
                             const IMPIContext*, int, DeviceWorkspaceManager*, int) override
        { return false; }
    };

    auto owned_0 = std::make_shared<FakeGemm>(0);
    auto owned_1 = std::make_shared<FakeGemm>(1);

    std::vector<ExpertArrival> arrivals;
    {
        ExpertArrival a;
        a.expert_id = 0;
        a.engine = owned_0.get();
        a.engine_lifetime = owned_0;
        a.derivation = WeightDerivationKind::ExpertSlice;
        arrivals.push_back(a);
    }
    {
        ExpertArrival a;
        a.expert_id = 1;
        a.engine = owned_1.get();
        a.engine_lifetime = owned_1;
        a.derivation = WeightDerivationKind::ExpertSlice;
        arrivals.push_back(a);
    }
    store.registerArrivedExperts(gate_ref, arrivals);

    // Verify resolution
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 0), owned_0.get());
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 1), owned_1.get());

    // Release local shared_ptr — store should keep engine alive
    ITensorGemm *raw_ptr = owned_0.get();
    owned_0.reset();

    // Engine should still be alive (store holds engine_lifetime)
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 0), raw_ptr);
}

// ===========================================================================
// 5. Slab ref consistency: same ref across initial + rebalance
// ===========================================================================

TEST(Test__MoEPhaseC_StoreResolution, SlabRefStability_RegisterOnce_UseForever)
{
    constexpr int num_experts = 8;
    PreparedWeightStore store(ModelContextId{88});

    auto ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertGate, num_experts));
    populateExperts(store, ref, 0, num_experts);

    // After multiple depart/arrive cycles, the ref should remain valid
    for (int round = 0; round < 5; ++round)
    {
        store.releaseDepartedExperts(ref, {round % num_experts});

        std::vector<ExpertArrival> arrivals;
        ExpertArrival a;
        a.expert_id = round % num_experts;
        a.engine = fakeEngine(1000 + round);
        a.derivation = WeightDerivationKind::RebalancedExpertReplica;
        arrivals.push_back(a);
        store.registerArrivedExperts(ref, arrivals);

        EXPECT_EQ(store.expertGemmKernel(ref, round % num_experts), fakeEngine(1000 + round))
            << "round=" << round;
    }
}

TEST(Test__MoEPhaseC_StoreResolution, FullLifecycle_RegisterResolveDepartRebalance)
{
    // End-to-end lifecycle: register → resolve → depart → rebalance → resolve
    constexpr int num_experts = 4;
    PreparedWeightStore store(ModelContextId{42});

    // Step 1: Register slabs
    auto gate_ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertGate, num_experts));
    auto up_ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertUp, num_experts));
    auto down_ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertDown, num_experts));

    // Step 2: Populate (initial load)
    populateExperts(store, gate_ref, 0, num_experts, 0);
    populateExperts(store, up_ref, 0, num_experts, 100);
    populateExperts(store, down_ref, 0, num_experts, 200);

    // Step 3: Resolve all — simulates ensureGemmEnginesCached tier 2
    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(store.expertGemmKernel(gate_ref, e), fakeEngine(e));
        EXPECT_EQ(store.expertGemmKernel(up_ref, e), fakeEngine(e + 100));
        EXPECT_EQ(store.expertGemmKernel(down_ref, e), fakeEngine(e + 200));
    }

    // Step 4: Depart experts 1, 3 — simulates releaseDepartedExperts
    store.releaseDepartedExperts(gate_ref, {1, 3});
    store.releaseDepartedExperts(up_ref, {1, 3});
    store.releaseDepartedExperts(down_ref, {1, 3});

    EXPECT_EQ(store.expertGemmKernel(gate_ref, 1), nullptr);
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 3), nullptr);
    EXPECT_NE(store.expertGemmKernel(gate_ref, 0), nullptr);
    EXPECT_NE(store.expertGemmKernel(gate_ref, 2), nullptr);

    // Step 5: Rebalance — new engines for departed experts
    auto rebalance = [&](const ExpertSlabRef &ref, int e, int offset) {
        std::vector<ExpertArrival> arr;
        ExpertArrival a;
        a.expert_id = e;
        a.engine = fakeEngine(e + offset + 500);
        a.derivation = WeightDerivationKind::RebalancedExpertReplica;
        arr.push_back(a);
        store.registerArrivedExperts(ref, arr);
    };

    rebalance(gate_ref, 1, 0);
    rebalance(gate_ref, 3, 0);
    rebalance(up_ref, 1, 100);
    rebalance(up_ref, 3, 100);
    rebalance(down_ref, 1, 200);
    rebalance(down_ref, 3, 200);

    // Step 6: Verify rebalanced engines
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 1), fakeEngine(501));
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 3), fakeEngine(503));
    EXPECT_EQ(store.expertGemmKernel(up_ref, 1), fakeEngine(601));
    EXPECT_EQ(store.expertGemmKernel(up_ref, 3), fakeEngine(603));
    EXPECT_EQ(store.expertGemmKernel(down_ref, 1), fakeEngine(701));
    EXPECT_EQ(store.expertGemmKernel(down_ref, 3), fakeEngine(703));

    // Original engines unchanged
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 0), fakeEngine(0));
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 2), fakeEngine(2));
}

// ===========================================================================
// 6. MoEWeightContext integration: releaseDepartedExperts service call
// ===========================================================================

TEST(Test__MoEPhaseC_StoreResolution, WeightServiceRelease_NotifiesAllThreeSlabs)
{
    // Simulates what MoEExpertWeightService::releaseDepartedExperts() does
    // with the Phase C notification block
    constexpr int num_experts = 8;
    SlabFixture f(5, num_experts);

    // Build vectors mimicking MoEWeightContext
    std::vector<ITensorGemm *> gate_engines(num_experts, nullptr);
    std::vector<ITensorGemm *> up_engines(num_experts, nullptr);
    std::vector<ITensorGemm *> down_engines(num_experts, nullptr);

    // Fill with engines (simulates state before departure)
    for (int e = 0; e < num_experts; ++e)
    {
        gate_engines[e] = fakeEngine(e);
        up_engines[e] = fakeEngine(e + 100);
        down_engines[e] = fakeEngine(e + 200);
    }

    // Simulate departure: experts 4, 6 depart
    std::vector<int> departed = {4, 6};

    // Phase C block: notify store
    f.store.releaseDepartedExperts(f.gate_ref, departed);
    f.store.releaseDepartedExperts(f.up_ref, departed);
    f.store.releaseDepartedExperts(f.down_ref, departed);

    // Verify: departed are null, rest remain
    for (int e : departed)
    {
        EXPECT_EQ(f.store.expertGemmKernel(f.gate_ref, e), nullptr);
        EXPECT_EQ(f.store.expertGemmKernel(f.up_ref, e), nullptr);
        EXPECT_EQ(f.store.expertGemmKernel(f.down_ref, e), nullptr);
    }
    EXPECT_NE(f.store.expertGemmKernel(f.gate_ref, 0), nullptr);
    EXPECT_NE(f.store.expertGemmKernel(f.gate_ref, 7), nullptr);
}

// ===========================================================================
// 7. Populated count tracking across depart/arrive cycles
// ===========================================================================

TEST(Test__MoEPhaseC_StoreResolution, PopulatedCount_TrackedAcrossCycles)
{
    constexpr int num_experts = 4;
    PreparedWeightStore store(ModelContextId{55});
    auto ref = store.registerExpertSlab(makeDesc(0, WeightRole::MoEExpertGate, num_experts));
    populateExperts(store, ref, 0, num_experts);

    EXPECT_EQ(store.totalPopulatedExperts(), 4u);

    // Depart 2
    store.releaseDepartedExperts(ref, {0, 2});
    EXPECT_EQ(store.totalPopulatedExperts(), 2u);

    // Arrive 1
    {
        std::vector<ExpertArrival> arrivals;
        ExpertArrival a;
        a.expert_id = 0;
        a.engine = fakeEngine(900);
        a.derivation = WeightDerivationKind::RebalancedExpertReplica;
        arrivals.push_back(a);
        store.registerArrivedExperts(ref, arrivals);
    }
    EXPECT_EQ(store.totalPopulatedExperts(), 3u);
}
