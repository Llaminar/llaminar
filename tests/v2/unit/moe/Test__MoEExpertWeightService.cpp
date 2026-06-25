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
#include "execution/moe/GpuExpertSlotPool.h"
#include "loaders/ExpertGemmRegistry.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/KernelFactory.h"
#include "backends/DeviceId.h"
#include "backends/BackendManager.h"
#include "utils/TestTensorFactory.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;
using KernelFactory = llaminar::v2::kernels::KernelFactory;

namespace
{

    /// Number of experts and dimensions for test fixtures.
    constexpr int kNumExperts = 4;
    constexpr int kExpertIntermediate = 64; // small for fast tests
    constexpr int kDModel = 32;
    constexpr size_t kBlockSize = 32; // Q4_0 block size

    /// Create a 3D Q4_0 tensor [cols, rows, num_experts] (GGUF convention).
    /// cols = fastest-varying dimension (ne[0]).
    /// Returns shared_ptr (required for create_view/shared_from_this).
    std::shared_ptr<Q4_0Tensor> createQ4_0_3D(size_t cols, size_t rows_per_expert, size_t num_experts, uint32_t seed = 42)
    {
        size_t blocks_per_row = (cols + kBlockSize - 1) / kBlockSize;
        size_t total_blocks = rows_per_expert * num_experts * blocks_per_row;

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_0Block));
        auto *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());

        for (size_t i = 0; i < total_blocks; ++i)
        {
            float max_abs = 0.0f;
            float values[kBlockSize];
            for (size_t j = 0; j < kBlockSize; ++j)
            {
                values[j] = dist(rng);
                max_abs = std::max(max_abs, std::abs(values[j]));
            }
            float scale = max_abs / 7.0f;
            blocks[i].d = fp32_to_fp16(scale);
            float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
            for (size_t j = 0; j < kBlockSize / 2; ++j)
            {
                int32_t q0 = static_cast<int32_t>(std::round(values[2 * j] * inv_scale)) + 8;
                int32_t q1 = static_cast<int32_t>(std::round(values[2 * j + 1] * inv_scale)) + 8;
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
    struct TestWeightContextOwner
    {
        DeviceId device_id = DeviceId::cpu();
        std::vector<bool> expert_mask;
        std::vector<std::shared_ptr<TensorBase>> expert_gate_views;
        std::vector<std::shared_ptr<TensorBase>> expert_up_views;
        std::vector<std::shared_ptr<TensorBase>> expert_down_views;
        std::vector<ITensorGemm *> prepared_gate_gemm;
        std::vector<ITensorGemm *> prepared_up_gemm;
        std::vector<ITensorGemm *> prepared_down_gemm;
        std::vector<std::shared_ptr<ITensorGemm>> moe_owned_kernels;
        std::shared_ptr<void> moe_packed_gate_lifetime;
        std::shared_ptr<void> moe_packed_up_lifetime;
        std::shared_ptr<void> moe_packed_down_lifetime;
        PreparedWeightStore *prepared_store = nullptr;
        ExpertGemmRegistry *expert_registry = nullptr;
        std::optional<ExpertSlabRef> gate_slab_ref;
        std::optional<ExpertSlabRef> up_slab_ref;
        std::optional<ExpertSlabRef> down_slab_ref;

        // 3D parent tensors (owned — must be shared_ptr for create_view/shared_from_this)
        std::shared_ptr<Q4_0Tensor> gate_3d;
        std::shared_ptr<Q4_0Tensor> up_3d;
        std::shared_ptr<Q4_0Tensor> down_3d;

        TestWeightContextOwner()
        {
            // gate/up: [d_model, intermediate, num_experts]  (K=d_model cols, N=intermediate rows)
            gate_3d = createQ4_0_3D(kDModel, kExpertIntermediate, kNumExperts, 42);
            up_3d = createQ4_0_3D(kDModel, kExpertIntermediate, kNumExperts, 43);
            // down: [intermediate, d_model, num_experts]  (K=intermediate cols, N=d_model rows)
            down_3d = createQ4_0_3D(kExpertIntermediate, kDModel, kNumExperts, 44);
        }

        MoEWeightContext buildContext()
        {
            return MoEWeightContext{
                device_id,
                kNumExperts,
                kExpertIntermediate,
                kDModel,
                0,  // local_expert_start
                -1, // local_expert_count (-1 = all)
                0,  // layer_idx
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
                prepared_store,
                expert_registry,
                gate_slab_ref,
                up_slab_ref,
                down_slab_ref};
        }
    };

    bool hasGPU()
    {
        return hasROCmBackend() || hasCUDABackend();
    }

    DeviceId firstGPU()
    {
        if (hasROCmBackend())
            return DeviceId(DeviceType::ROCm, 0);
        if (hasCUDABackend())
            return DeviceId(DeviceType::CUDA, 0);
        return DeviceId::cpu();
    }

    ITensorGemm *fakeGemm(int id)
    {
        return reinterpret_cast<ITensorGemm *>(static_cast<uintptr_t>(0x7000 + id * 0x100));
    }

    class OwnedFakeGemm final : public ITensorGemm
    {
    public:
        explicit OwnedFakeGemm(std::shared_ptr<int> release_count)
            : release_count_(std::move(release_count)) {}

        bool supports_device(int) const override { return true; }

        bool multiply_tensor(const TensorBase *, TensorBase *,
                             int, int, int,
                             bool, float, float,
                             const TensorBase *,
                             const IMPIContext *,
                             int,
                             DeviceWorkspaceManager *,
                             int) override
        {
            return false;
        }

        void releaseWeights() override
        {
            if (release_count_)
                ++(*release_count_);
        }

    private:
        std::shared_ptr<int> release_count_;
    };

    class OwnedFakeGemmWithAllocation final : public ITensorGemm
    {
    public:
        OwnedFakeGemmWithAllocation(std::shared_ptr<int> release_count,
                                    std::shared_ptr<void> allocation_owner)
            : release_count_(std::move(release_count)),
              allocation_owner_(std::move(allocation_owner)) {}

        bool supports_device(int) const override { return true; }

        bool multiply_tensor(const TensorBase *, TensorBase *,
                             int, int, int,
                             bool, float, float,
                             const TensorBase *,
                             const IMPIContext *,
                             int,
                             DeviceWorkspaceManager *,
                             int) override
        {
            return false;
        }

        void releaseWeights() override
        {
            if (release_count_)
                ++(*release_count_);
        }

    private:
        std::shared_ptr<int> release_count_;
        std::shared_ptr<void> allocation_owner_;
    };

    ExpertSlabDescriptor makeGpuStoreDesc(DeviceId device, WeightRole role)
    {
        ExpertSlabDescriptor desc;
        desc.layer_idx = 0;
        desc.role = role;
        desc.device = device;
        desc.num_experts = kNumExperts;
        desc.local_expert_start = 0;
        desc.local_expert_count = kNumExperts;
        desc.rows_per_expert = kExpertIntermediate;
        desc.cols_per_expert = kDModel;
        return desc;
    }

    void populateOneExpert(PreparedWeightStore &store, const ExpertSlabRef &ref, int expert_id, ITensorGemm *engine)
    {
        ExpertArrival arrival;
        arrival.expert_id = expert_id;
        arrival.engine = engine;
        arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
        store.registerArrivedExperts(ref, {arrival});
    }

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// extractExpertViews
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, ExtractExpertViews_AllExperts)
{
    TestWeightContextOwner owner;
    auto ctx = owner.buildContext();

    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));

    // Should have created views for all experts
    ASSERT_EQ(owner.expert_gate_views.size(), static_cast<size_t>(kNumExperts));
    ASSERT_EQ(owner.expert_up_views.size(), static_cast<size_t>(kNumExperts));
    ASSERT_EQ(owner.expert_down_views.size(), static_cast<size_t>(kNumExperts));

    // Each view should be non-null and 2D
    for (int e = 0; e < kNumExperts; ++e)
    {
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

TEST(Test__MoEExpertWeightService, ExtractExpertViews_EPRange)
{
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

TEST(Test__MoEExpertWeightService, ExtractExpertViews_NullTensors)
{
    TestWeightContextOwner owner;
    owner.gate_3d.reset();
    auto ctx = owner.buildContext();

    EXPECT_FALSE(MoEExpertWeightService::extractExpertViews(ctx));
}

TEST(Test__MoEExpertWeightService, ExtractExpertViews_WithMask_ExtractsAll)
{
    TestWeightContextOwner owner;
    // When expert_mask is non-empty, all views are extracted (for dynamic rebalancing)
    owner.expert_mask = {true, false, true, false};
    auto ctx = owner.buildContext();

    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));

    // All experts get views even though mask is partial
    for (int e = 0; e < kNumExperts; ++e)
    {
        EXPECT_NE(owner.expert_gate_views[e], nullptr) << "expert " << e;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// prepareGemmEngines (CPU path)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_CPU_AllExperts)
{
    TestWeightContextOwner owner;

    // Step 1: Extract views
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }

    // Step 2: Prepare engines
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    // All experts should have prepared GEMM engines
    ASSERT_EQ(owner.prepared_gate_gemm.size(), static_cast<size_t>(kNumExperts));
    for (int e = 0; e < kNumExperts; ++e)
    {
        EXPECT_NE(owner.prepared_gate_gemm[e], nullptr) << "gate engine null at expert " << e;
        EXPECT_NE(owner.prepared_up_gemm[e], nullptr) << "up engine null at expert " << e;
        EXPECT_NE(owner.prepared_down_gemm[e], nullptr) << "down engine null at expert " << e;
    }
}

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_CPU_StoreOwnsPreparedEngines)
{
    TestWeightContextOwner owner;
    PreparedWeightStore store(ModelContextId{7});
    owner.prepared_store = &store;

    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }

    ExpertSlabRef gate_ref;
    ExpertSlabRef up_ref;
    ExpertSlabRef down_ref;
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
        ASSERT_TRUE(ctx.gate_slab_ref.has_value());
        ASSERT_TRUE(ctx.up_slab_ref.has_value());
        ASSERT_TRUE(ctx.down_slab_ref.has_value());
        gate_ref = *ctx.gate_slab_ref;
        up_ref = *ctx.up_slab_ref;
        down_ref = *ctx.down_slab_ref;
    }

    EXPECT_TRUE(owner.moe_owned_kernels.empty());
    EXPECT_EQ(store.expertSlabCount(), 3u);
    EXPECT_EQ(store.totalPopulatedExperts(), static_cast<size_t>(kNumExperts * 3));

    for (int e = 0; e < kNumExperts; ++e)
    {
        EXPECT_EQ(owner.prepared_gate_gemm[e], store.expertGemmKernel(gate_ref, e));
        EXPECT_EQ(owner.prepared_up_gemm[e], store.expertGemmKernel(up_ref, e));
        EXPECT_EQ(owner.prepared_down_gemm[e], store.expertGemmKernel(down_ref, e));
        EXPECT_NE(owner.prepared_gate_gemm[e], nullptr);
        EXPECT_NE(owner.prepared_up_gemm[e], nullptr);
        EXPECT_NE(owner.prepared_down_gemm[e], nullptr);
    }
}

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_GPU_ReusesCompletePreparedStoreSlabs)
{
    TestWeightContextOwner owner;
    owner.device_id = DeviceId::cuda(0);
    PreparedWeightStore store(ModelContextId{81});
    owner.prepared_store = &store;

    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }

    auto gate_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertGate));
    auto up_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertUp));
    auto down_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertDown));
    for (int e = 0; e < kNumExperts; ++e)
    {
        populateOneExpert(store, gate_ref, e, fakeGemm(10 + e));
        populateOneExpert(store, up_ref, e, fakeGemm(110 + e));
        populateOneExpert(store, down_ref, e, fakeGemm(210 + e));
    }

    auto ctx = owner.buildContext();
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));

    EXPECT_TRUE(owner.moe_owned_kernels.empty())
        << "GPU prepared-store reuse must not allocate graph-local duplicate expert pools.";
    ASSERT_TRUE(ctx.gate_slab_ref.has_value());
    ASSERT_TRUE(ctx.up_slab_ref.has_value());
    ASSERT_TRUE(ctx.down_slab_ref.has_value());
    EXPECT_EQ(*ctx.gate_slab_ref, gate_ref);
    EXPECT_EQ(*ctx.up_slab_ref, up_ref);
    EXPECT_EQ(*ctx.down_slab_ref, down_ref);
    for (int e = 0; e < kNumExperts; ++e)
    {
        EXPECT_EQ(owner.prepared_gate_gemm[e], fakeGemm(10 + e));
        EXPECT_EQ(owner.prepared_up_gemm[e], fakeGemm(110 + e));
        EXPECT_EQ(owner.prepared_down_gemm[e], fakeGemm(210 + e));
    }
}

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_GPU_IncompletePreparedStoreSlabsDoNotRepack)
{
    TestWeightContextOwner owner;
    owner.device_id = DeviceId::cuda(0);
    PreparedWeightStore store(ModelContextId{82});
    owner.prepared_store = &store;

    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }

    auto gate_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertGate));
    auto up_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertUp));
    auto down_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertDown));
    populateOneExpert(store, gate_ref, 0, fakeGemm(10));
    populateOneExpert(store, up_ref, 0, fakeGemm(110));
    populateOneExpert(store, down_ref, 0, fakeGemm(210));

    auto ctx = owner.buildContext();
    EXPECT_FALSE(MoEExpertWeightService::prepareGemmEngines(ctx));
    EXPECT_TRUE(owner.moe_owned_kernels.empty())
        << "Incomplete GPU slabs should fail instead of allocating a second full expert pool.";
}

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_CPU_IncrementallyFillsExistingStoreSlabs)
{
    TestWeightContextOwner owner;
    PreparedWeightStore store(ModelContextId{8});
    owner.prepared_store = &store;

    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }

    owner.expert_mask = {true, false, true, false};
    ExpertSlabRef gate_ref;
    ExpertSlabRef up_ref;
    ExpertSlabRef down_ref;
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
        ASSERT_TRUE(ctx.gate_slab_ref.has_value());
        ASSERT_TRUE(ctx.up_slab_ref.has_value());
        ASSERT_TRUE(ctx.down_slab_ref.has_value());
        gate_ref = *ctx.gate_slab_ref;
        up_ref = *ctx.up_slab_ref;
        down_ref = *ctx.down_slab_ref;
    }

    EXPECT_EQ(store.expertSlabCount(), 3u);
    EXPECT_EQ(store.totalPopulatedExperts(), 6u);
    EXPECT_NE(store.expertGemmKernel(gate_ref, 0), nullptr);
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 1), nullptr);
    EXPECT_NE(store.expertGemmKernel(gate_ref, 2), nullptr);
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 3), nullptr);

    owner.expert_mask = {false, true, false, true};
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
        ASSERT_TRUE(ctx.gate_slab_ref.has_value());
        ASSERT_TRUE(ctx.up_slab_ref.has_value());
        ASSERT_TRUE(ctx.down_slab_ref.has_value());
        EXPECT_EQ(*ctx.gate_slab_ref, gate_ref);
        EXPECT_EQ(*ctx.up_slab_ref, up_ref);
        EXPECT_EQ(*ctx.down_slab_ref, down_ref);
    }

    EXPECT_TRUE(owner.moe_owned_kernels.empty());
    EXPECT_EQ(store.expertSlabCount(), 3u);
    EXPECT_EQ(store.totalPopulatedExperts(), static_cast<size_t>(kNumExperts * 3));
    for (int e = 0; e < kNumExperts; ++e)
    {
        EXPECT_NE(store.expertGemmKernel(gate_ref, e), nullptr) << "gate expert " << e;
        EXPECT_NE(store.expertGemmKernel(up_ref, e), nullptr) << "up expert " << e;
        EXPECT_NE(store.expertGemmKernel(down_ref, e), nullptr) << "down expert " << e;
    }
}

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_WithMask_OnlyMaskedExperts)
{
    TestWeightContextOwner owner;
    owner.expert_mask = {true, false, true, false};

    // Extract views (all experts get views with mask)
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }

    // Prepare engines (only mask-active experts get engines)
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    ASSERT_EQ(owner.prepared_gate_gemm.size(), static_cast<size_t>(kNumExperts));
    EXPECT_NE(owner.prepared_gate_gemm[0], nullptr);
    EXPECT_EQ(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_NE(owner.prepared_gate_gemm[2], nullptr);
    EXPECT_EQ(owner.prepared_gate_gemm[3], nullptr);
}

TEST(Test__MoEExpertWeightService, PrepareGemmEngines_NoViews_Fails)
{
    TestWeightContextOwner owner;
    auto ctx = owner.buildContext();

    // Skip extractExpertViews — views are empty
    EXPECT_FALSE(MoEExpertWeightService::prepareGemmEngines(ctx));
}

// ─────────────────────────────────────────────────────────────────────────────
// releaseRawWeights
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, ReleaseRawWeights_NullsParentPointers)
{
    TestWeightContextOwner owner;

    // Extract and prepare first
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

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

TEST(Test__MoEExpertWeightService, ReleaseDepartedExperts_ClearsEngines)
{
    TestWeightContextOwner owner;

    // Full setup: extract + prepare all experts
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    // Verify all engines exist
    for (int e = 0; e < kNumExperts; ++e)
    {
        ASSERT_NE(owner.prepared_gate_gemm[e], nullptr);
    }

    // New mask: keep experts 0 and 2, depart 1 and 3
    std::vector<bool> new_mask = {true, false, true, false};
    {
        auto ctx = owner.buildContext();
        auto evicted = MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);

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

TEST(Test__MoEExpertWeightService, ReleaseDepartedExperts_RemovesRegistryEntries)
{
    TestWeightContextOwner owner;
    ExpertGemmRegistry registry;
    owner.expert_registry = &registry;

    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    for (int e = 0; e < kNumExperts; ++e)
    {
        registry.registerEngine(owner.device_id, 0, e, ExpertGemmRegistry::WeightRole::GATE,
                                owner.prepared_gate_gemm[e], nullptr);
        registry.registerEngine(owner.device_id, 0, e, ExpertGemmRegistry::WeightRole::UP,
                                owner.prepared_up_gemm[e], nullptr);
        registry.registerEngine(owner.device_id, 0, e, ExpertGemmRegistry::WeightRole::DOWN,
                                owner.prepared_down_gemm[e], nullptr);
    }
    ASSERT_TRUE(registry.hasCompleteLayer(owner.device_id, 0, kNumExperts));

    std::vector<bool> new_mask = {true, false, true, false};
    {
        auto ctx = owner.buildContext();
        (void)MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
    }

    EXPECT_NE(registry.getEngine(owner.device_id, 0, 0, ExpertGemmRegistry::WeightRole::GATE), nullptr);
    EXPECT_EQ(registry.getEngine(owner.device_id, 0, 1, ExpertGemmRegistry::WeightRole::GATE), nullptr);
    EXPECT_EQ(registry.getEngine(owner.device_id, 0, 1, ExpertGemmRegistry::WeightRole::UP), nullptr);
    EXPECT_EQ(registry.getEngine(owner.device_id, 0, 1, ExpertGemmRegistry::WeightRole::DOWN), nullptr);
    EXPECT_NE(registry.getEngine(owner.device_id, 0, 2, ExpertGemmRegistry::WeightRole::GATE), nullptr);
    EXPECT_EQ(registry.getEngine(owner.device_id, 0, 3, ExpertGemmRegistry::WeightRole::GATE), nullptr);
    EXPECT_FALSE(registry.hasCompleteLayer(owner.device_id, 0, kNumExperts));
}

TEST(Test__MoEExpertWeightService, ReleaseDepartedExperts_DropsGraphLocalOwnersForDirectArrivals)
{
    TestWeightContextOwner owner;
    owner.device_id = DeviceId::cuda(0);
    owner.expert_mask.assign(kNumExperts, true);
    owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
    owner.prepared_up_gemm.assign(kNumExperts, nullptr);
    owner.prepared_down_gemm.assign(kNumExperts, nullptr);

    PreparedWeightStore store(ModelContextId{1234});
    ExpertGemmRegistry registry;
    owner.prepared_store = &store;
    owner.expert_registry = &registry;
    owner.gate_slab_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertGate));
    owner.up_slab_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertUp));
    owner.down_slab_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertDown));

    struct RegisteredEngine
    {
        std::weak_ptr<ITensorGemm> weak;
        std::shared_ptr<int> release_count;
    };

    auto register_engine = [&](int expert_id,
                               WeightRole store_role,
                               ExpertGemmRegistry::WeightRole registry_role,
                               const ExpertSlabRef &slab_ref,
                               std::vector<ITensorGemm *> &prepared) -> RegisteredEngine
    {
        auto release_count = std::make_shared<int>(0);
        auto engine = std::make_shared<OwnedFakeGemm>(release_count);
        std::weak_ptr<ITensorGemm> weak = engine;
        prepared[expert_id] = engine.get();
        owner.moe_owned_kernels.push_back(engine);

        ExpertArrival arrival;
        arrival.expert_id = expert_id;
        arrival.engine = engine.get();
        arrival.engine_lifetime = engine;
        arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
        store.registerArrivedExperts(slab_ref, {arrival});
        registry.registerEngine(owner.device_id, 0, expert_id, registry_role, engine.get(), engine);

        (void)store_role;
        return RegisteredEngine{weak, release_count};
    };

    auto retained_gate = register_engine(0, WeightRole::MoEExpertGate, ExpertGemmRegistry::WeightRole::GATE,
                                         *owner.gate_slab_ref, owner.prepared_gate_gemm);
    auto retained_up = register_engine(0, WeightRole::MoEExpertUp, ExpertGemmRegistry::WeightRole::UP,
                                       *owner.up_slab_ref, owner.prepared_up_gemm);
    auto retained_down = register_engine(0, WeightRole::MoEExpertDown, ExpertGemmRegistry::WeightRole::DOWN,
                                         *owner.down_slab_ref, owner.prepared_down_gemm);
    auto departed_gate = register_engine(1, WeightRole::MoEExpertGate, ExpertGemmRegistry::WeightRole::GATE,
                                         *owner.gate_slab_ref, owner.prepared_gate_gemm);
    auto departed_up = register_engine(1, WeightRole::MoEExpertUp, ExpertGemmRegistry::WeightRole::UP,
                                       *owner.up_slab_ref, owner.prepared_up_gemm);
    auto departed_down = register_engine(1, WeightRole::MoEExpertDown, ExpertGemmRegistry::WeightRole::DOWN,
                                         *owner.down_slab_ref, owner.prepared_down_gemm);

    ASSERT_EQ(owner.moe_owned_kernels.size(), 6u);
    ASSERT_EQ(store.expertGemmKernel(*owner.gate_slab_ref, 1), owner.prepared_gate_gemm[1]);
    ASSERT_NE(registry.getEngine(owner.device_id, 0, 1, ExpertGemmRegistry::WeightRole::GATE), nullptr);

    std::vector<bool> new_mask = {true, false, false, false};
    {
        auto ctx = owner.buildContext();
        (void)MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
    }

    EXPECT_EQ(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_up_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_down_gemm[1], nullptr);
    EXPECT_EQ(store.expertGemmKernel(*owner.gate_slab_ref, 1), nullptr);
    EXPECT_EQ(registry.getEngine(owner.device_id, 0, 1, ExpertGemmRegistry::WeightRole::GATE), nullptr);
    EXPECT_EQ(owner.moe_owned_kernels.size(), 3u);

    EXPECT_TRUE(departed_gate.weak.expired());
    EXPECT_TRUE(departed_up.weak.expired());
    EXPECT_TRUE(departed_down.weak.expired());
    EXPECT_EQ(*departed_gate.release_count, 1);
    EXPECT_EQ(*departed_up.release_count, 1);
    EXPECT_EQ(*departed_down.release_count, 1);

    EXPECT_FALSE(retained_gate.weak.expired());
    EXPECT_FALSE(retained_up.weak.expired());
    EXPECT_FALSE(retained_down.weak.expired());
    EXPECT_EQ(*retained_gate.release_count, 0);
    EXPECT_EQ(*retained_up.release_count, 0);
    EXPECT_EQ(*retained_down.release_count, 0);
}

TEST(Test__MoEExpertWeightService, ReleaseDepartedExperts_FreesDepartedDirectArrivalAllocationOwner)
{
    TestWeightContextOwner owner;
    owner.device_id = DeviceId::cuda(0);
    owner.expert_mask.assign(kNumExperts, true);
    owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
    owner.prepared_up_gemm.assign(kNumExperts, nullptr);
    owner.prepared_down_gemm.assign(kNumExperts, nullptr);

    PreparedWeightStore store(ModelContextId{2345});
    ExpertGemmRegistry registry;
    owner.prepared_store = &store;
    owner.expert_registry = &registry;
    owner.gate_slab_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertGate));
    owner.up_slab_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertUp));
    owner.down_slab_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertDown));

    struct AllocationRecord
    {
        std::weak_ptr<void> weak_allocation;
        std::shared_ptr<int> free_count;
    };

    auto register_role = [&](int expert_id,
                             WeightRole store_role,
                             ExpertGemmRegistry::WeightRole registry_role,
                             const ExpertSlabRef &slab_ref,
                             std::vector<ITensorGemm *> &prepared,
                             const std::shared_ptr<void> &allocation_owner)
    {
        auto release_count = std::make_shared<int>(0);
        auto engine = std::make_shared<OwnedFakeGemmWithAllocation>(
            release_count,
            allocation_owner);
        prepared[expert_id] = engine.get();
        owner.moe_owned_kernels.push_back(engine);

        ExpertArrival arrival;
        arrival.expert_id = expert_id;
        arrival.engine = engine.get();
        arrival.engine_lifetime = engine;
        arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
        store.registerArrivedExperts(slab_ref, {arrival});
        registry.registerEngine(owner.device_id, 0, expert_id, registry_role, engine.get(), engine);
        (void)store_role;
    };

    auto register_expert = [&](int expert_id) -> AllocationRecord
    {
        auto free_count = std::make_shared<int>(0);
        std::weak_ptr<void> weak_allocation;
        {
            auto allocation_owner = std::shared_ptr<void>(
                new int(1),
                [free_count](void *ptr)
                {
                    ++(*free_count);
                    delete static_cast<int *>(ptr);
                });
            weak_allocation = allocation_owner;

            register_role(expert_id, WeightRole::MoEExpertGate, ExpertGemmRegistry::WeightRole::GATE,
                          *owner.gate_slab_ref, owner.prepared_gate_gemm, allocation_owner);
            register_role(expert_id, WeightRole::MoEExpertUp, ExpertGemmRegistry::WeightRole::UP,
                          *owner.up_slab_ref, owner.prepared_up_gemm, allocation_owner);
            register_role(expert_id, WeightRole::MoEExpertDown, ExpertGemmRegistry::WeightRole::DOWN,
                          *owner.down_slab_ref, owner.prepared_down_gemm, allocation_owner);
        }
        return AllocationRecord{weak_allocation, free_count};
    };

    auto retained_allocation = register_expert(0);
    auto departed_allocation = register_expert(1);

    ASSERT_FALSE(retained_allocation.weak_allocation.expired());
    ASSERT_FALSE(departed_allocation.weak_allocation.expired());
    ASSERT_EQ(owner.moe_owned_kernels.size(), 6u);

    std::vector<bool> new_mask = {true, false, false, false};
    {
        auto ctx = owner.buildContext();
        (void)MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
    }

    EXPECT_FALSE(retained_allocation.weak_allocation.expired())
        << "retained expert allocation should stay resident";
    EXPECT_TRUE(departed_allocation.weak_allocation.expired())
        << "departed expert allocation should not be pinned by unrelated arrivals";
    EXPECT_EQ(*retained_allocation.free_count, 0);
    EXPECT_EQ(*departed_allocation.free_count, 1);
}

TEST(Test__MoEExpertWeightService, ReleaseDepartedExperts_KeepsSiblingStageOwnerAlive)
{
    TestWeightContextOwner first_stage;
    TestWeightContextOwner second_stage;
    first_stage.device_id = DeviceId::cuda(0);
    second_stage.device_id = DeviceId::cuda(0);
    first_stage.prepared_gate_gemm.assign(kNumExperts, nullptr);
    first_stage.prepared_up_gemm.assign(kNumExperts, nullptr);
    first_stage.prepared_down_gemm.assign(kNumExperts, nullptr);
    second_stage.prepared_gate_gemm.assign(kNumExperts, nullptr);
    second_stage.prepared_up_gemm.assign(kNumExperts, nullptr);
    second_stage.prepared_down_gemm.assign(kNumExperts, nullptr);

    PreparedWeightStore store(ModelContextId{5678});
    auto gate_ref = store.registerExpertSlab(makeGpuStoreDesc(first_stage.device_id, WeightRole::MoEExpertGate));
    first_stage.prepared_store = &store;
    second_stage.prepared_store = &store;
    first_stage.gate_slab_ref = gate_ref;
    second_stage.gate_slab_ref = gate_ref;

    auto release_count = std::make_shared<int>(0);
    std::weak_ptr<ITensorGemm> weak;
    ITensorGemm *raw = nullptr;
    {
        auto engine = std::make_shared<OwnedFakeGemm>(release_count);
        weak = engine;
        raw = engine.get();
        first_stage.prepared_gate_gemm[1] = raw;
        second_stage.prepared_gate_gemm[1] = raw;
        first_stage.moe_owned_kernels.push_back(engine);
        second_stage.moe_owned_kernels.push_back(engine);

        ExpertArrival arrival;
        arrival.expert_id = 1;
        arrival.engine = raw;
        arrival.engine_lifetime = engine;
        arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
        store.registerArrivedExperts(gate_ref, {arrival});
    }

    std::vector<bool> new_mask = {true, false, true, true};
    {
        auto ctx = first_stage.buildContext();
        (void)MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
    }

    EXPECT_EQ(first_stage.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(first_stage.moe_owned_kernels.size(), 0u);
    EXPECT_EQ(store.expertGemmKernel(gate_ref, 1), nullptr);
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(second_stage.prepared_gate_gemm[1], raw);
    EXPECT_EQ(*release_count, 1);

    {
        auto ctx = second_stage.buildContext();
        (void)MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
    }

    EXPECT_EQ(second_stage.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(second_stage.moe_owned_kernels.size(), 0u);
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(*release_count, 2);
}

TEST(Test__MoEExpertWeightService, GpuDirectSlotPool_ReusesReleasedPhysicalSlot)
{
    std::vector<GpuExpertSlotPool::ProjectionSpec> specs;
    for (const char *label : {"gate", "up", "down"})
    {
        GpuExpertSlotPool::ProjectionSpec spec;
        spec.label = label;
        spec.N = 4;
        spec.K = 32;
        spec.payload_bytes_per_block = 16;
        spec.is_asymmetric = true;
        spec.has_emins = false;
        spec.codebook_id = 7;
        specs.push_back(std::move(spec));
    }

    auto pool = GpuExpertSlotPool::create(
        nullptr,
        DeviceId::cuda(0),
        /*device_ordinal=*/0,
        /*layer_idx=*/3,
        /*capacity=*/2,
        std::move(specs),
        /*vram_safety_margin_bytes=*/0);

    auto first = pool->acquire(11);
    auto second = pool->acquire(12);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(pool->usedSlots(), 2u);
    EXPECT_FALSE(pool->acquire(13).has_value());

    const int released_slot = first->slot_index;
    EXPECT_EQ(pool->slotForExpert(11), released_slot);
    first->lifetime.reset();

    EXPECT_FALSE(pool->slotForExpert(11).has_value());
    EXPECT_EQ(pool->usedSlots(), 1u);

    auto reused = pool->acquire(13);
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(reused->slot_index, released_slot);
    EXPECT_EQ(pool->slotForExpert(13), released_slot);
}

TEST(Test__MoEExpertWeightService, GpuDirectSlotPool_RecommendedCapacityCoversBatchAndTenPercent)
{
    EXPECT_EQ(GpuExpertSlotPool::recommendedCapacity(64, 2), 7);
    EXPECT_EQ(GpuExpertSlotPool::recommendedCapacity(64, 9), 9);
    EXPECT_EQ(GpuExpertSlotPool::recommendedCapacity(1, 0), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// registerAndPrepareNewExperts
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, RegisterAndPrepareNewExperts_NullReceived_FailsWithoutStoreEngine)
{
    TestWeightContextOwner owner;
    owner.expert_mask = {true, false, true, false};

    // Setup with mask for experts 0 and 2
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    // Now "acquire" expert 1 without transferred/store-owned packed weights.
    // Raw repacking from expert views is forbidden after eager materialization.
    std::vector<bool> new_mask = {true, true, true, false};
    {
        auto ctx = owner.buildContext();
        EXPECT_FALSE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, nullptr));
    }

    EXPECT_EQ(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_up_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_down_gemm[1], nullptr);
}

TEST(Test__MoEExpertWeightService, RegisterAndPrepareNewExperts_StoreEngineReuse)
{
    TestWeightContextOwner owner;
    PreparedWeightStore store(ModelContextId{9});
    owner.prepared_store = &store;

    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }

    ExpertSlabRef gate_ref;
    ExpertSlabRef up_ref;
    ExpertSlabRef down_ref;
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
        ASSERT_TRUE(ctx.gate_slab_ref.has_value());
        ASSERT_TRUE(ctx.up_slab_ref.has_value());
        ASSERT_TRUE(ctx.down_slab_ref.has_value());
        gate_ref = *ctx.gate_slab_ref;
        up_ref = *ctx.up_slab_ref;
        down_ref = *ctx.down_slab_ref;
    }

    owner.prepared_gate_gemm[1] = nullptr;
    owner.prepared_up_gemm[1] = nullptr;
    owner.prepared_down_gemm[1] = nullptr;

    std::vector<bool> new_mask = {true, true, true, false};
    {
        auto ctx = owner.buildContext();
        ctx.gate_slab_ref = gate_ref;
        ctx.up_slab_ref = up_ref;
        ctx.down_slab_ref = down_ref;
        ASSERT_TRUE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, nullptr));
    }

    EXPECT_EQ(owner.prepared_gate_gemm[1], store.expertGemmKernel(gate_ref, 1));
    EXPECT_EQ(owner.prepared_up_gemm[1], store.expertGemmKernel(up_ref, 1));
    EXPECT_EQ(owner.prepared_down_gemm[1], store.expertGemmKernel(down_ref, 1));
}

TEST(Test__MoEExpertWeightService, RegisterAndPrepareNewExperts_InvalidBlobDoesNotRepack)
{
    TestWeightContextOwner owner;
    owner.expert_mask = {true, false, true, false};

    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    std::unordered_map<int, ExpertWeightBlobs> received;
    received[1].gate = {0x01, 0x02, 0x03};
    received[1].up = {0x04, 0x05};
    received[1].down = {0x06};

    std::vector<bool> new_mask = {true, true, true, false};
    {
        auto ctx = owner.buildContext();
        EXPECT_FALSE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, &received));
    }

    EXPECT_EQ(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_up_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_down_gemm[1], nullptr);
}

TEST(Test__MoEExpertWeightService, RegisterAndPrepareNewExperts_NoNewExperts)
{
    TestWeightContextOwner owner;

    // Setup all experts
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    // Same mask — no new experts
    std::vector<bool> same_mask = {true, true, true, true};
    {
        auto ctx = owner.buildContext();
        EXPECT_TRUE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, same_mask, nullptr));
    }
}

TEST(Test__MoEExpertWeightService, GPURebalanceRequiresPayloadWhenCacheMissing)
{
    if (!hasGPU())
    {
        GTEST_SKIP() << "No GPU backend available";
    }

    TestWeightContextOwner owner;
    owner.device_id = firstGPU();
    owner.expert_mask = {true, false, false, false};
    owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
    owner.prepared_up_gemm.assign(kNumExperts, nullptr);
    owner.prepared_down_gemm.assign(kNumExperts, nullptr);

    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }

    std::vector<bool> new_mask = {true, true, false, false};
    {
        auto ctx = owner.buildContext();
        EXPECT_FALSE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, nullptr));
    }

    EXPECT_EQ(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_up_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_down_gemm[1], nullptr);
}

TEST(Test__MoEExpertWeightService, GPURebalanceResolvesStoreSlabsWhenCachedRefsMissing)
{
    if (!hasGPU())
    {
        GTEST_SKIP() << "No GPU backend available";
    }

    TestWeightContextOwner owner;
    owner.device_id = firstGPU();
    owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
    owner.prepared_up_gemm.assign(kNumExperts, nullptr);
    owner.prepared_down_gemm.assign(kNumExperts, nullptr);

    PreparedWeightStore store(ModelContextId{10});
    owner.prepared_store = &store;

    auto gate_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertGate));
    auto up_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertUp));
    auto down_ref = store.registerExpertSlab(makeGpuStoreDesc(owner.device_id, WeightRole::MoEExpertDown));
    populateOneExpert(store, gate_ref, 1, fakeGemm(1));
    populateOneExpert(store, up_ref, 1, fakeGemm(101));
    populateOneExpert(store, down_ref, 1, fakeGemm(201));

    std::vector<bool> new_mask = {false, true, false, false};
    auto ctx = owner.buildContext();
    ASSERT_FALSE(ctx.gate_slab_ref.has_value());
    ASSERT_FALSE(ctx.up_slab_ref.has_value());
    ASSERT_FALSE(ctx.down_slab_ref.has_value());

    ASSERT_TRUE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, nullptr));

    EXPECT_EQ(owner.prepared_gate_gemm[1], fakeGemm(1));
    EXPECT_EQ(owner.prepared_up_gemm[1], fakeGemm(101));
    EXPECT_EQ(owner.prepared_down_gemm[1], fakeGemm(201));
    ASSERT_TRUE(ctx.gate_slab_ref.has_value());
    ASSERT_TRUE(ctx.up_slab_ref.has_value());
    ASSERT_TRUE(ctx.down_slab_ref.has_value());
    EXPECT_EQ(ctx.gate_slab_ref->slab_id, gate_ref.slab_id);
    EXPECT_EQ(ctx.up_slab_ref->slab_id, up_ref.slab_id);
    EXPECT_EQ(ctx.down_slab_ref->slab_id, down_ref.slab_id);
}

// ─────────────────────────────────────────────────────────────────────────────
// Full lifecycle: extract → prepare → release → rebalance
// ─────────────────────────────────────────────────────────────────────────────

TEST(Test__MoEExpertWeightService, FullLifecycle)
{
    TestWeightContextOwner owner;

    // 1. Extract views
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    }

    // 2. Prepare engines
    {
        auto ctx = owner.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    // 3. Release raw weights
    {
        auto ctx = owner.buildContext();
        size_t freed = MoEExpertWeightService::releaseRawWeights(ctx);
        EXPECT_GT(freed, 0u);
    }

    // 4. Depart experts 2, 3
    std::vector<bool> new_mask = {true, true, false, false};
    {
        auto ctx = owner.buildContext();
        auto evicted = MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
        EXPECT_EQ(evicted.size(), 6u);
    } // 3 views × 2 departed

    // 5. Re-acquire expert 3 without transferred/store-owned packed weights.
    // This must fail instead of repacking from raw views.
    std::vector<bool> final_mask = {true, true, false, true};
    {
        auto ctx = owner.buildContext();
        EXPECT_FALSE(MoEExpertWeightService::registerAndPrepareNewExperts(ctx, final_mask, nullptr));
    }

    // Final state: experts 0 and 1 have engines; departed/reacquired experts do not.
    EXPECT_NE(owner.prepared_gate_gemm[0], nullptr);
    EXPECT_NE(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_gate_gemm[2], nullptr);
    EXPECT_EQ(owner.prepared_gate_gemm[3], nullptr);
}
