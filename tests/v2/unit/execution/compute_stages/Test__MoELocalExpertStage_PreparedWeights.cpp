/**
 * @file Test__MoELocalExpertStage_PreparedWeights.cpp
 * @brief Phase 12 unit tests — MoELocalExpertStage prepared-weight validation.
 *
 * Verifies that:
 *   1. validatePreparedWeights() succeeds with complete prepared engine vectors.
 *   2. validatePreparedWeights() fails when an active expert is missing an engine.
 *   3. validatePreparedWeights() succeeds with slab-refs pointing to a registered store.
 *   4. validatePreparedWeights() fails when a slab is not in the store (empty mask).
 *   5. Params still lacks forbidden runtime/runner/peer fields (reuse type-trait guard).
 */

#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/ExpertSlabTypes.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

using namespace llaminar2;

namespace
{
    // -----------------------------------------------------------------------
    // Sentinel engine pointers — not null, never dereferenced in validation.
    // -----------------------------------------------------------------------
    ITensorGemm *eng(int n)
    {
        // n must be in [1..63] to keep the pointer obviously non-null and distinct.
        return reinterpret_cast<ITensorGemm *>(static_cast<uintptr_t>(n) * 0x1000u);
    }

    // -----------------------------------------------------------------------
    // Helpers for building minimal PreparedWeightStore slabs
    // -----------------------------------------------------------------------
    ExpertSlabDescriptor makeSlabDesc(int layer_idx, WeightRole role, int num_experts)
    {
        ExpertSlabDescriptor desc;
        desc.layer_idx = layer_idx;
        desc.role = role;
        desc.device = DeviceId::cpu();
        desc.num_experts = num_experts;
        desc.local_expert_start = 0;
        desc.local_expert_count = num_experts;
        desc.rows_per_expert = 64;
        desc.cols_per_expert = 32;
        return desc;
    }

    ExpertArrival makeArrival(int expert_id, ITensorGemm *engine)
    {
        ExpertArrival arrival;
        arrival.expert_id = expert_id;
        arrival.engine = engine;
        arrival.engine_lifetime = nullptr;
        arrival.view_lifetime = nullptr;
        arrival.derivation = WeightDerivationKind::ExpertSlice;
        return arrival;
    }

    // -----------------------------------------------------------------------
    // Build a Params with fully-populated prepared engine vectors
    // (num_experts=4, all active).
    // -----------------------------------------------------------------------
    MoELocalExpertStage::Params makePreparedVectorParams(int num_experts = 4)
    {
        MoELocalExpertStage::Params p;
        p.num_experts = num_experts;
        p.top_k = 2;
        p.d_model = 32;
        p.expert_intermediate = 64;
        p.layer_idx = 0;
        p.prepared_gate_gemm.resize(static_cast<size_t>(num_experts));
        p.prepared_up_gemm.resize(static_cast<size_t>(num_experts));
        p.prepared_down_gemm.resize(static_cast<size_t>(num_experts));
        for (int e = 0; e < num_experts; ++e)
        {
            p.prepared_gate_gemm[static_cast<size_t>(e)] = eng(1 + e);
            p.prepared_up_gemm[static_cast<size_t>(e)] = eng(10 + e);
            p.prepared_down_gemm[static_cast<size_t>(e)] = eng(20 + e);
        }
        return p;
    }

    // -----------------------------------------------------------------------
    // Type-trait guards (reused from Test__MoELocalExpertStage_Params)
    // -----------------------------------------------------------------------
    template <typename, typename = void>
    struct has_runtime : std::false_type
    {
    };
    template <typename T>
    struct has_runtime<T, std::void_t<decltype(std::declval<T &>().runtime)>> : std::true_type
    {
    };

    template <typename, typename = void>
    struct has_peer_participants : std::false_type
    {
    };
    template <typename T>
    struct has_peer_participants<T, std::void_t<decltype(std::declval<T &>().peer_participants)>>
        : std::true_type
    {
    };

    template <typename, typename = void>
    struct has_prepared_participants : std::false_type
    {
    };
    template <typename T>
    struct has_prepared_participants<T, std::void_t<decltype(std::declval<T &>().prepared_participants)>>
        : std::true_type
    {
    };

} // namespace

// ===========================================================================
// Test suite
// ===========================================================================

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ValidatePreparedWeights_SucceedsWithCompleteEngineVectors)
{
    auto p = makePreparedVectorParams(4);
    MoELocalExpertStage stage(p);

    std::string err;
    EXPECT_TRUE(stage.validatePreparedWeights(&err));
    EXPECT_TRUE(err.empty());
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ValidatePreparedWeights_SucceedsWithActiveExpertMaskAndCompleteOwned)
{
    // Only experts 0 and 2 are "owned" by this participant.
    auto p = makePreparedVectorParams(4);
    p.expert_mask = {true, false, true, false};
    // Experts 1 and 3 are null — but they are masked out, so OK.
    p.prepared_gate_gemm[1] = nullptr;
    p.prepared_up_gemm[1] = nullptr;
    p.prepared_down_gemm[1] = nullptr;
    p.prepared_gate_gemm[3] = nullptr;
    p.prepared_up_gemm[3] = nullptr;
    p.prepared_down_gemm[3] = nullptr;

    MoELocalExpertStage stage(p);
    std::string err;
    EXPECT_TRUE(stage.validatePreparedWeights(&err));
    EXPECT_TRUE(err.empty());
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ValidatePreparedWeights_FailsWhenActiveExpertMissingEngine)
{
    auto p = makePreparedVectorParams(4);
    // Expert 2 is active but its gate engine is null.
    p.prepared_gate_gemm[2] = nullptr;

    MoELocalExpertStage stage(p);
    std::string err;
    EXPECT_FALSE(stage.validatePreparedWeights(&err));
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("expert 2"), std::string::npos);
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ValidatePreparedWeights_FailsWithMismatchedVectorSizes)
{
    auto p = makePreparedVectorParams(4);
    // Deliberately mismatch prepared_up_gemm size.
    p.prepared_up_gemm.resize(3);

    MoELocalExpertStage stage(p);
    std::string err;
    EXPECT_FALSE(stage.validatePreparedWeights(&err));
    EXPECT_FALSE(err.empty());
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ValidatePreparedWeights_SucceedsWithSlabRefsAndRegisteredStore)
{
    constexpr int kNumExperts = 4;
    PreparedWeightStore store(ModelContextId{42});

    auto gate_ref = store.registerExpertSlab(makeSlabDesc(0, WeightRole::MoEExpertGate, kNumExperts));
    auto up_ref = store.registerExpertSlab(makeSlabDesc(0, WeightRole::MoEExpertUp, kNumExperts));
    auto down_ref = store.registerExpertSlab(makeSlabDesc(0, WeightRole::MoEExpertDown, kNumExperts));

    // Populate at least one expert engine per slab so the availability mask is non-empty.
    for (int e = 0; e < kNumExperts; ++e)
    {
        store.registerArrivedExperts(gate_ref, {makeArrival(e, eng(1 + e))});
        store.registerArrivedExperts(up_ref, {makeArrival(e, eng(10 + e))});
        store.registerArrivedExperts(down_ref, {makeArrival(e, eng(20 + e))});
    }

    MoELocalExpertStage::Params p;
    p.num_experts = kNumExperts;
    p.top_k = 2;
    p.d_model = 32;
    p.expert_intermediate = 64;
    p.layer_idx = 0;
    p.prepared_store = &store;
    p.gate_slab_ref = gate_ref;
    p.up_slab_ref = up_ref;
    p.down_slab_ref = down_ref;

    MoELocalExpertStage stage(p);
    std::string err;
    EXPECT_TRUE(stage.validatePreparedWeights(&err));
    EXPECT_TRUE(err.empty());
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ValidatePreparedWeights_FailsWhenSlabNotRegisteredInStore)
{
    constexpr int kNumExperts = 4;
    PreparedWeightStore store_a(ModelContextId{10});
    PreparedWeightStore store_b(ModelContextId{11});

    // Register slabs in store_b.
    auto gate_ref = store_b.registerExpertSlab(makeSlabDesc(0, WeightRole::MoEExpertGate, kNumExperts));
    auto up_ref = store_b.registerExpertSlab(makeSlabDesc(0, WeightRole::MoEExpertUp, kNumExperts));
    auto down_ref = store_b.registerExpertSlab(makeSlabDesc(0, WeightRole::MoEExpertDown, kNumExperts));

    for (int e = 0; e < kNumExperts; ++e)
    {
        store_b.registerArrivedExperts(gate_ref, {makeArrival(e, eng(1 + e))});
        store_b.registerArrivedExperts(up_ref, {makeArrival(e, eng(10 + e))});
        store_b.registerArrivedExperts(down_ref, {makeArrival(e, eng(20 + e))});
    }

    // But params point to store_a — the slab_ids from store_b won't be found there.
    MoELocalExpertStage::Params p;
    p.num_experts = kNumExperts;
    p.top_k = 2;
    p.d_model = 32;
    p.expert_intermediate = 64;
    p.layer_idx = 0;
    p.prepared_store = &store_a; // wrong store
    p.gate_slab_ref = gate_ref;
    p.up_slab_ref = up_ref;
    p.down_slab_ref = down_ref;

    MoELocalExpertStage stage(p);
    std::string err;
    EXPECT_FALSE(stage.validatePreparedWeights(&err));
    EXPECT_FALSE(err.empty());
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ValidatePreparedWeights_FailsWhenActiveExpertMissingFromSlab)
{
    constexpr int kNumExperts = 4;
    PreparedWeightStore store(ModelContextId{12});

    auto gate_ref = store.registerExpertSlab(makeSlabDesc(0, WeightRole::MoEExpertGate, kNumExperts));
    auto up_ref = store.registerExpertSlab(makeSlabDesc(0, WeightRole::MoEExpertUp, kNumExperts));
    auto down_ref = store.registerExpertSlab(makeSlabDesc(0, WeightRole::MoEExpertDown, kNumExperts));

    for (int e = 0; e < kNumExperts; ++e)
    {
        if (e != 2)
            store.registerArrivedExperts(gate_ref, {makeArrival(e, eng(1 + e))});
        store.registerArrivedExperts(up_ref, {makeArrival(e, eng(10 + e))});
        store.registerArrivedExperts(down_ref, {makeArrival(e, eng(20 + e))});
    }

    MoELocalExpertStage::Params p;
    p.num_experts = kNumExperts;
    p.top_k = 2;
    p.d_model = 32;
    p.expert_intermediate = 64;
    p.layer_idx = 0;
    p.expert_mask = {true, false, true, false};
    p.prepared_store = &store;
    p.gate_slab_ref = gate_ref;
    p.up_slab_ref = up_ref;
    p.down_slab_ref = down_ref;

    MoELocalExpertStage stage(p);
    std::string err;
    EXPECT_FALSE(stage.validatePreparedWeights(&err));
    EXPECT_NE(err.find("expert 2"), std::string::npos);
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ValidatePreparedWeights_FallsBackToLegacyWhenNoPreparedStateAndRawTensorsPresent)
{
    // When no prepared state and raw tensors are present → legacy path, returns true.
    MoELocalExpertStage::Params p;
    p.num_experts = 4;
    p.top_k = 2;
    p.d_model = 32;
    p.expert_intermediate = 64;
    p.layer_idx = 0;
    // Provide sentinel non-null raw tensors.
    p.gate_exps = reinterpret_cast<TensorBase *>(uintptr_t{0x4000});
    p.up_exps = reinterpret_cast<TensorBase *>(uintptr_t{0x5000});
    p.down_exps = reinterpret_cast<TensorBase *>(uintptr_t{0x6000});

    MoELocalExpertStage stage(p);
    std::string err;
    EXPECT_TRUE(stage.validatePreparedWeights(&err));
    EXPECT_TRUE(err.empty());
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ValidatePreparedWeights_FailsWithNoPreparedStateAndNoRawTensors)
{
    MoELocalExpertStage::Params p;
    p.num_experts = 4;
    p.top_k = 2;
    p.d_model = 32;
    p.expert_intermediate = 64;
    p.layer_idx = 0;
    // No prepared vectors, no slab refs, no raw tensors.

    MoELocalExpertStage stage(p);
    std::string err;
    EXPECT_FALSE(stage.validatePreparedWeights(&err));
    EXPECT_FALSE(err.empty());
}

// ---------------------------------------------------------------------------
// Structural: Params must not have forbidden runtime/peer fields
// ---------------------------------------------------------------------------

TEST(Test__MoELocalExpertStage_PreparedWeights, ParamsHasNoPreparedParticipantsOrRuntimeFields)
{
    using P = MoELocalExpertStage::Params;
    EXPECT_FALSE(has_runtime<P>::value);
    EXPECT_FALSE(has_peer_participants<P>::value);
    EXPECT_FALSE(has_prepared_participants<P>::value);
}
