/**
 * @file Test__MoEExpertWeightService.cpp
 * @brief Unit tests for MoEExpertWeightService — weight lifecycle service
 *        extracted from MoEExpertComputeStage.
 *
 * Tests: extractExpertViews, prepareGemmEngines, releaseRawWeights,
 *        releaseDepartedExperts, registerAndPrepareNewExperts.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEExpertWeightService.h"
#include "loaders/ExpertGemmRegistry.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/KernelFactory.h"
#include "backends/DeviceId.h"
#include "backends/BackendManager.h"
#include "utils/TestTensorFactory.h"

#include <memory>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;
using KernelFactory = llaminar::v2::kernels::KernelFactory;

namespace {

/// Number of experts and dimensions for test fixtures.
constexpr int kNumExperts = 4;
constexpr int kExpertIntermediate = 64;  // small for fast tests
constexpr int kDModel = 32;
constexpr size_t kBlockSize = 32;  // Q4_0 block size

/// Create a 3D Q4_0 tensor [cols, rows, num_experts] (GGUF convention).
/// cols = fastest-varying dimension (ne[0]).
/// Returns shared_ptr (required for create_view/shared_from_this).
std::shared_ptr<Q4_0Tensor> createQ4_0_3D(size_t cols, size_t rows_per_expert, size_t num_experts, uint32_t seed = 42) {
    size_t blocks_per_row = (cols + kBlockSize - 1) / kBlockSize;
    size_t total_blocks = rows_per_expert * num_experts * blocks_per_row;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_0Block));
    auto* blocks = reinterpret_cast<Q4_0Block*>(raw_data.data());

    for (size_t i = 0; i < total_blocks; ++i) {
        float max_abs = 0.0f;
        float values[kBlockSize];
        for (size_t j = 0; j < kBlockSize; ++j) {
            values[j] = dist(rng);
            max_abs = std::max(max_abs, std::abs(values[j]));
        }
        float scale = max_abs / 7.0f;
        blocks[i].d = fp32_to_fp16(scale);
        float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        for (size_t j = 0; j < kBlockSize / 2; ++j) {
            int32_t q0 = static_cast<int32_t>(std::round(values[2*j]     * inv_scale)) + 8;
            int32_t q1 = static_cast<int32_t>(std::round(values[2*j + 1] * inv_scale)) + 8;
            q0 = std::clamp(q0, 0, 15);
            q1 = std::clamp(q1, 0, 15);
            blocks[i].qs[j] = static_cast<uint8_t>((q1 << 4) | q0);
        }
    }

    // GGUF 3D: shape = [cols, rows_per_expert, num_experts]
    std::vector<size_t> shape = {cols, rows_per_expert, num_experts};
    return std::make_shared<Q4_0Tensor>(shape, raw_data);
}

/// Helper to build a MoEWeightContext from local vectors.
struct TestWeightContextOwner {
    DeviceId device_id = DeviceId::cpu();
    std::vector<bool> expert_mask;
    std::vector<std::shared_ptr<TensorBase>> expert_gate_views;
    std::vector<std::shared_ptr<TensorBase>> expert_up_views;
    std::vector<std::shared_ptr<TensorBase>> expert_down_views;
    std::vector<ITensorGemm*> prepared_gate_gemm;
    std::vector<ITensorGemm*> prepared_up_gemm;
    std::vector<ITensorGemm*> prepared_down_gemm;
    std::vector<std::shared_ptr<ITensorGemm>> moe_owned_kernels;
    std::shared_ptr<void> moe_packed_gate_lifetime;
    std::shared_ptr<void> moe_packed_up_lifetime;
    std::shared_ptr<void> moe_packed_down_lifetime;
    ExpertGemmRegistry* expert_registry = nullptr;

    // 3D parent tensors (owned — must be shared_ptr for create_view/shared_from_this)
    std::shared_ptr<Q4_0Tensor> gate_3d;
    std::shared_ptr<Q4_0Tensor> up_3d;
    std::shared_ptr<Q4_0Tensor> down_3d;

    TestWeightContextOwner() {
        // gate/up: [d_model, intermediate, num_experts]  (K=d_model cols, N=intermediate rows)
        gate_3d = createQ4_0_3D(kDModel, kExpertIntermediate, kNumExperts, 42);
        up_3d   = createQ4_0_3D(kDModel, kExpertIntermediate, kNumExperts, 43);
        // down: [intermediate, d_model, num_experts]  (K=intermediate cols, N=d_model rows)
        down_3d = createQ4_0_3D(kExpertIntermediate, kDModel, kNumExperts, 44);
    }

    MoEWeightContext buildContext() {
        return MoEWeightContext{
            device_id,
            kNumExperts,
            kExpertIntermediate,
            kDModel,
            0,   // local_expert_start
            -1,  // local_expert_count (-1 = all)
            0,   // layer_idx
            expert_mask,
            gate_3d.get(),
            up_3d.get(),
            down_3d.get(),
            expert_gate_views,
            expert_up_views,
            expert_down_views,
            prepared_gate_gemm,
            prepared_up_gemm,
            prepared_down_gemm,
            moe_owned_kernels,
            moe_packed_gate_lifetime,
            moe_packed_up_lifetime,
            moe_packed_down_lifetime,
            nullptr,
            nullptr,
            expert_registry
        };
    }
};

bool hasGPU() {
    return hasROCmBackend() || hasCUDABackend();
}

DeviceId firstGPU() {
    if (hasROCmBackend()) return DeviceId(DeviceType::ROCm, 0);
    if (hasCUDABackend()) return DeviceId(DeviceType::CUDA, 0);
    return DeviceId::cpu();
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// extractExpertViews
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, ExtractExpertViews_AllExperts) {
    TestWeightContextOwner owner;
    auto ctx = owner.buildContext();

    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));

    // Should have created views for all experts
    ASSERT_EQ(owner.expert_gate_views.size(), static_cast<size_t>(kNumExperts));
    ASSERT_EQ(owner.expert_up_views.size(), static_cast<size_t>(kNumExperts));
    ASSERT_EQ(owner.expert_down_views.size(), static_cast<size_t>(kNumExperts));

    // Each view should be non-null and 2D
    for (int e = 0; e < kNumExperts; ++e) {
        ASSERT_NE(owner.expert_gate_views[e], nullptr) << "gate view null at expert " << e;
        ASSERT_NE(owner.expert_up_views[e], nullptr) << "up view null at expert " << e;
        ASSERT_NE(owner.expert_down_views[e], nullptr) << "down view null at expert " << e;

        EXPECT_EQ(owner.expert_gate_views[e]->shape().size(), 2u);
        EXPECT_EQ(owner.expert_up_views[e]->shape().size(), 2u);
        EXPECT_EQ(owner.expert_down_views[e]->shape().size(), 2u);

        // gate/up views: [intermediate, d_model]
        EXPECT_EQ(owner.expert_gate_views[e]->rows(), kExpertIntermediate);
        EXPECT_EQ(owner.expert_gate_views[e]->cols(), kDModel);
        // down views: [d_model, intermediate]
        EXPECT_EQ(owner.expert_down_views[e]->rows(), kDModel);
        EXPECT_EQ(owner.expert_down_views[e]->cols(), kExpertIntermediate);
    }
}

TEST(Test__MoEExpertWeightService, ExtractExpertViews_EPRange) {
    TestWeightContextOwner owner;
    auto ctx = owner.buildContext();
    // Set EP range: only experts 1..2
    ctx.local_expert_start = 1;
    ctx.local_expert_count = 2;

    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));

    ASSERT_EQ(owner.expert_gate_views.size(), static_cast<size_t>(kNumExperts));
    // Only experts 1 and 2 should have non-null views
    EXPECT_EQ(owner.expert_gate_views[0], nullptr);
    EXPECT_NE(owner.expert_gate_views[1], nullptr);
    EXPECT_NE(owner.expert_gate_views[2], nullptr);
    EXPECT_EQ(owner.expert_gate_views[3], nullptr);
}

TEST(Test__MoEExpertWeightService, ExtractExpertViews_NullTensors) {
    TestWeightContextOwner owner;
    owner.gate_3d.reset();
    auto ctx = owner.buildContext();

    EXPECT_FALSE(MoEExpertWeightService::extractExpertViews(ctx));
}

TEST(Test__MoEExpertWeightService, ExtractExpertViews_WithMask_ExtractsAll) {
    TestWeightContextOwner owner;
    // When expert_mask is non-empty, all views are extracted (for dynamic rebalancing)
    owner.expert_mask = {true, false, true, false};
    auto ctx = owner.buildContext();

    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));

    // All experts get views even though mask is partial
    for (int e = 0; e < kNumExperts; ++e) {
        EXPECT_NE(owner.expert_gate_views[e], nullptr) << "expert " << e;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// prepareGemmEngines (CPU path)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_CPU_AllExperts) {
    TestWeightContextOwner owner;

    // Step 1: Extract views
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx)); }

    // Step 2: Prepare engines
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx)); }

    // All experts should have prepared GEMM engines
    ASSERT_EQ(owner.prepared_gate_gemm.size(), static_cast<size_t>(kNumExperts));
    for (int e = 0; e < kNumExperts; ++e) {
        EXPECT_NE(owner.prepared_gate_gemm[e], nullptr) << "gate engine null at expert " << e;
        EXPECT_NE(owner.prepared_up_gemm[e], nullptr) << "up engine null at expert " << e;
        EXPECT_NE(owner.prepared_down_gemm[e], nullptr) << "down engine null at expert " << e;
    }
}

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_WithMask_OnlyMaskedExperts) {
    TestWeightContextOwner owner;
    owner.expert_mask = {true, false, true, false};

    // Extract views (all experts get views with mask)
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx)); }

    // Prepare engines (only mask-active experts get engines)
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx)); }

    ASSERT_EQ(owner.prepared_gate_gemm.size(), static_cast<size_t>(kNumExperts));
    EXPECT_NE(owner.prepared_gate_gemm[0], nullptr);
    EXPECT_EQ(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_NE(owner.prepared_gate_gemm[2], nullptr);
    EXPECT_EQ(owner.prepared_gate_gemm[3], nullptr);
}

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_NoViews_Fails) {
    TestWeightContextOwner owner;
    auto ctx = owner.buildContext();

    // Skip extractExpertViews — views are empty
    EXPECT_FALSE(MoEExpertWeightService::prepareGemmEngines(ctx));
}

// ─────────────────────────────────────────────────────────────────────────────
// releaseRawWeights
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, ReleaseRawWeights_NullsParentPointers) {
    TestWeightContextOwner owner;

    // Extract and prepare first
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx)); }
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx)); }

    // Now release
    auto ctx = owner.buildContext();
    size_t freed = MoEExpertWeightService::releaseRawWeights(ctx);

    // Parent 3D pointers should be nulled in the context
    EXPECT_EQ(ctx.gate_exps, nullptr);
    EXPECT_EQ(ctx.up_exps, nullptr);
    EXPECT_EQ(ctx.down_exps, nullptr);

    // Should have freed some bytes (heap-allocated test tensors)
    EXPECT_GT(freed, 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// releaseDepartedExperts
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, ReleaseDepartedExperts_ClearsEngines) {
    TestWeightContextOwner owner;

    // Full setup: extract + prepare all experts
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx)); }
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx)); }

    // Verify all engines exist
    for (int e = 0; e < kNumExperts; ++e) {
        ASSERT_NE(owner.prepared_gate_gemm[e], nullptr);
    }

    // New mask: keep experts 0 and 2, depart 1 and 3
    std::vector<bool> new_mask = {true, false, true, false};
    { auto ctx = owner.buildContext(); auto evicted = MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);

    // Departed experts (1,3) should have null engines
    EXPECT_EQ(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_up_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_down_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_gate_gemm[3], nullptr);

    // Retained experts (0,2) should still have engines
    EXPECT_NE(owner.prepared_gate_gemm[0], nullptr);
    EXPECT_NE(owner.prepared_gate_gemm[2], nullptr);

    // Evicted tensor list should be non-empty (3 views per departed expert × 2 experts)
    EXPECT_EQ(evicted.size(), 6u);
    }
}

TEST(Test__MoEExpertWeightService, ReleaseDepartedExperts_RemovesRegistryEntries) {
    TestWeightContextOwner owner;
    ExpertGemmRegistry registry;
    owner.expert_registry = &registry;

    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx)); }
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx)); }

    for (int e = 0; e < kNumExperts; ++e) {
        registry.registerEngine(owner.device_id, 0, e, ExpertGemmRegistry::WeightRole::GATE,
                                owner.prepared_gate_gemm[e], nullptr);
        registry.registerEngine(owner.device_id, 0, e, ExpertGemmRegistry::WeightRole::UP,
                                owner.prepared_up_gemm[e], nullptr);
        registry.registerEngine(owner.device_id, 0, e, ExpertGemmRegistry::WeightRole::DOWN,
                                owner.prepared_down_gemm[e], nullptr);
    }
    ASSERT_TRUE(registry.hasCompleteLayer(owner.device_id, 0, kNumExperts));

    std::vector<bool> new_mask = {true, false, true, false};
    { auto ctx = owner.buildContext(); (void)MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask); }

    EXPECT_NE(registry.getEngine(owner.device_id, 0, 0, ExpertGemmRegistry::WeightRole::GATE), nullptr);
    EXPECT_EQ(registry.getEngine(owner.device_id, 0, 1, ExpertGemmRegistry::WeightRole::GATE), nullptr);
    EXPECT_EQ(registry.getEngine(owner.device_id, 0, 1, ExpertGemmRegistry::WeightRole::UP), nullptr);
    EXPECT_EQ(registry.getEngine(owner.device_id, 0, 1, ExpertGemmRegistry::WeightRole::DOWN), nullptr);
    EXPECT_NE(registry.getEngine(owner.device_id, 0, 2, ExpertGemmRegistry::WeightRole::GATE), nullptr);
    EXPECT_EQ(registry.getEngine(owner.device_id, 0, 3, ExpertGemmRegistry::WeightRole::GATE), nullptr);
    EXPECT_FALSE(registry.hasCompleteLayer(owner.device_id, 0, kNumExperts));
}

// ─────────────────────────────────────────────────────────────────────────────
// registerAndPrepareNewExperts
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, RegisterAndPrepareNewExperts_NullReceived) {
    TestWeightContextOwner owner;
    owner.expert_mask = {true, false, true, false};

    // Setup with mask for experts 0 and 2
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx)); }
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx)); }

    // Now "acquire" expert 1 (no transferred weights — must repack from views)
    std::vector<bool> new_mask = {true, true, true, false};
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, nullptr)); }

    // Expert 1 should now have engines
    EXPECT_NE(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_NE(owner.prepared_up_gemm[1], nullptr);
    EXPECT_NE(owner.prepared_down_gemm[1], nullptr);
}

TEST(Test__MoEExpertWeightService, RegisterAndPrepareNewExperts_NoNewExperts) {
    TestWeightContextOwner owner;

    // Setup all experts
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx)); }
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx)); }

    // Same mask — no new experts
    std::vector<bool> same_mask = {true, true, true, true};
    { auto ctx = owner.buildContext(); EXPECT_TRUE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, same_mask, nullptr)); }
}

TEST(Test__MoEExpertWeightService, GPURebalanceRequiresPayloadWhenCacheMissing) {
        if (!hasGPU()) {
                GTEST_SKIP() << "No GPU backend available";
        }

        TestWeightContextOwner owner;
        owner.device_id = firstGPU();
        owner.expert_mask = {true, false, false, false};
        owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
        owner.prepared_up_gemm.assign(kNumExperts, nullptr);
        owner.prepared_down_gemm.assign(kNumExperts, nullptr);

        { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx)); }

        std::vector<bool> new_mask = {true, true, false, false};
        { auto ctx = owner.buildContext();
            EXPECT_FALSE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, nullptr)); }

        EXPECT_EQ(owner.prepared_gate_gemm[1], nullptr);
        EXPECT_EQ(owner.prepared_up_gemm[1], nullptr);
        EXPECT_EQ(owner.prepared_down_gemm[1], nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Full lifecycle: extract → prepare → release → rebalance
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, FullLifecycle) {
    TestWeightContextOwner owner;

    // 1. Extract views
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx)); }

    // 2. Prepare engines
    { auto ctx = owner.buildContext(); ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx)); }

    // 3. Release raw weights
    { auto ctx = owner.buildContext();
      size_t freed = MoEExpertWeightService::releaseRawWeights(ctx);
      EXPECT_GT(freed, 0u); }

    // 4. Depart experts 2, 3
    std::vector<bool> new_mask = {true, true, false, false};
    { auto ctx = owner.buildContext();
      auto evicted = MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
      EXPECT_EQ(evicted.size(), 6u); }  // 3 views × 2 departed

    // 5. Re-acquire expert 3 (from views, not transferred)
    std::vector<bool> final_mask = {true, true, false, true};
    { auto ctx = owner.buildContext();
      ASSERT_TRUE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, final_mask, nullptr)); }

    // Final state: experts 0, 1, 3 have engines; expert 2 does not
    EXPECT_NE(owner.prepared_gate_gemm[0], nullptr);
    EXPECT_NE(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_gate_gemm[2], nullptr);
    EXPECT_NE(owner.prepared_gate_gemm[3], nullptr);
}
