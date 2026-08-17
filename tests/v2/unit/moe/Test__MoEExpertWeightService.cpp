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
#include "execution/moe/GpuExpertTransferStagingPool.h"
#include "loaders/ExpertGemmRegistry.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/KernelFactory.h"
#include "backends/DeviceId.h"
#include "backends/BackendManager.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
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

    std::string readRepoFile(const std::filesystem::path &relative_path)
    {
        for (auto dir = std::filesystem::current_path(); !dir.empty(); dir = dir.parent_path())
        {
            const auto candidate = dir / relative_path;
            if (std::filesystem::exists(candidate))
            {
                std::ifstream in(candidate);
                std::ostringstream buffer;
                buffer << in.rdbuf();
                return buffer.str();
            }
            if (dir == dir.root_path())
                break;
        }
        return {};
    }

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
        std::shared_ptr<GpuExpertSlotPool> gpu_direct_slot_pool;

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
                down_slab_ref,
                true,
                &gpu_direct_slot_pool,
                CPUExpertNUMAPlacement::aggregateDomain()};
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

    std::string readMoEWeightServiceSource()
    {
        std::ifstream in("/workspaces/llaminar/src/v2/execution/moe/MoEExpertWeightService.cpp");
        if (!in)
            return {};
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    GpuExpertPackedDescriptor fakePackedDesc(uintptr_t base)
    {
        GpuExpertPackedDescriptor desc;
        desc.ptrs.d_vnni = reinterpret_cast<uint8_t *>(base);
        desc.ptrs.d_scales = reinterpret_cast<void *>(base + 0x1000);
        desc.ptrs.d_mins = reinterpret_cast<void *>(base + 0x2000);
        desc.ptrs.d_emins = reinterpret_cast<void *>(base + 0x3000);
        desc.n = kExpertIntermediate;
        desc.k = kDModel;
        desc.blocks_per_row = 1;
        desc.codebook_id = 4;
        desc.payload_bytes_per_block = 16;
        desc.is_asymmetric = true;
        desc.has_emins = true;
        desc.vnni_bytes = static_cast<size_t>(desc.n * desc.blocks_per_row * desc.payload_bytes_per_block);
        desc.scales_bytes = static_cast<size_t>(desc.n * desc.blocks_per_row * sizeof(uint16_t));
        desc.mins_bytes = desc.scales_bytes;
        desc.emins_bytes = static_cast<size_t>(desc.n * desc.blocks_per_row * sizeof(uint32_t));
        return desc;
    }

    GpuDirectTransferSlotProjection fakeTransferSlotProjection(
        int expert_id,
        WeightRole role,
        uintptr_t base,
        const std::shared_ptr<void> &lifetime)
    {
        GpuDirectTransferSlotProjection projection;
        projection.expert_id = expert_id;
        projection.role = role;
        projection.staged = fakePackedDesc(base);
        projection.N = projection.staged.n;
        projection.K = projection.staged.k;
        projection.blocks_per_row = projection.staged.blocks_per_row;
        projection.payload_bytes_per_block = projection.staged.payload_bytes_per_block;
        projection.is_asymmetric = projection.staged.is_asymmetric;
        projection.has_emins = projection.staged.has_emins;
        projection.codebook_id = projection.staged.codebook_id;
        projection.source_identity = NativeVnniSourceIdentity{
            .codebook_id = native_vnni_formats::IQ4_NL.codebook_id,
            .is_superblock = native_vnni_formats::IQ4_NL.is_superblock,
            .present = true,
        };
        projection.transfer_slot_lifetime = lifetime;
        return projection;
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
        if (role == WeightRole::MoEExpertDown)
        {
            desc.rows_per_expert = kDModel;
            desc.cols_per_expert = kExpertIntermediate;
        }
        else
        {
            desc.rows_per_expert = kExpertIntermediate;
            desc.cols_per_expert = kDModel;
        }
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
    // Set the local expert-ID range to experts 1 and 2.
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

TEST(Test__MoEExpertWeightService, ExtractExpertViews_PackedNonContiguousMaskMapsGlobalIdsToPackedSlots)
{
    TestWeightContextOwner owner;
    owner.gate_3d = createQ4_0_3D(kDModel, kExpertIntermediate, 2, 142);
    owner.up_3d = createQ4_0_3D(kDModel, kExpertIntermediate, 2, 143);
    owner.down_3d = createQ4_0_3D(kExpertIntermediate, kDModel, 2, 144);
    owner.expert_mask = {true, false, true, false};

    auto ctx = owner.buildContext();
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));

    ASSERT_NE(owner.expert_gate_views[0], nullptr);
    EXPECT_EQ(owner.expert_gate_views[1], nullptr);
    ASSERT_NE(owner.expert_gate_views[2], nullptr);
    EXPECT_EQ(owner.expert_gate_views[3], nullptr);

    const size_t gate_bytes_per_expert =
        static_cast<size_t>(kExpertIntermediate) *
        ((static_cast<size_t>(kDModel) + kBlockSize - 1u) / kBlockSize) *
        sizeof(Q4_0Block);
    const auto *gate_base = static_cast<const uint8_t *>(owner.gate_3d->raw_data());
    EXPECT_EQ(owner.expert_gate_views[0]->raw_data(), gate_base);
    EXPECT_EQ(
        owner.expert_gate_views[2]->raw_data(),
        gate_base + gate_bytes_per_expert);

    const size_t down_bytes_per_expert =
        static_cast<size_t>(kDModel) *
        ((static_cast<size_t>(kExpertIntermediate) + kBlockSize - 1u) / kBlockSize) *
        sizeof(Q4_0Block);
    const auto *down_base = static_cast<const uint8_t *>(owner.down_3d->raw_data());
    EXPECT_EQ(owner.expert_down_views[0]->raw_data(), down_base);
    EXPECT_EQ(
        owner.expert_down_views[2]->raw_data(),
        down_base + down_bytes_per_expert);
}

TEST(Test__MoEExpertWeightService, ExtractExpertViews_PackedMaskCardinalityMismatchIsFatal)
{
    TestWeightContextOwner owner;
    owner.gate_3d = createQ4_0_3D(kDModel, kExpertIntermediate, 2, 242);
    owner.up_3d = createQ4_0_3D(kDModel, kExpertIntermediate, 2, 243);
    owner.down_3d = createQ4_0_3D(kExpertIntermediate, kDModel, 2, 244);
    owner.expert_mask = {true, false, false, false};

    auto ctx = owner.buildContext();
    EXPECT_THROW(
        (void)MoEExpertWeightService::extractExpertViews(ctx),
        std::runtime_error);
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

TEST(Test__MoEExpertWeightService, ReleaseDepartedExperts_DoesNotCallThroughRawSlotBackedEngines)
{
    TestWeightContextOwner owner;
    owner.expert_gate_views.clear();
    owner.expert_up_views.clear();
    owner.expert_down_views.clear();
    owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
    owner.prepared_up_gemm.assign(kNumExperts, nullptr);
    owner.prepared_down_gemm.assign(kNumExperts, nullptr);
    owner.prepared_gate_gemm[1] = fakeGemm(1);
    owner.prepared_up_gemm[1] = fakeGemm(101);
    owner.prepared_down_gemm[1] = fakeGemm(201);

    std::vector<bool> new_mask = {true, false, true, true};
    auto ctx = owner.buildContext();
    auto evicted = MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);

    EXPECT_TRUE(evicted.empty());
    EXPECT_EQ(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_up_gemm[1], nullptr);
    EXPECT_EQ(owner.prepared_down_gemm[1], nullptr);
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
    ASSERT_NE(first->lifetime, nullptr);
    ASSERT_NE(second->lifetime, nullptr);
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

TEST(Test__MoEExpertWeightService, GpuDirectSlotPool_RetainsSameExpertAcrossRcuEpochs)
{
    std::vector<GpuExpertSlotPool::ProjectionSpec> specs;
    for (const char *label : {"gate", "up", "down"})
    {
        specs.push_back({
            .label = label,
            .N = 4,
            .K = 32,
            .payload_bytes_per_block = 16,
            .is_asymmetric = true,
            .has_emins = false,
            .codebook_id = 7,
        });
    }

    auto pool = GpuExpertSlotPool::create(
        nullptr,
        DeviceId::cuda(0),
        /*device_ordinal=*/0,
        /*layer_idx=*/3,
        /*active_capacity=*/2,
        std::move(specs),
        /*vram_safety_margin_bytes=*/0);

    auto old_epoch = pool->acquire(11, /*residency_epoch=*/40);
    auto candidate_epoch = pool->acquire(11, /*residency_epoch=*/41);
    ASSERT_TRUE(old_epoch.has_value());
    ASSERT_TRUE(candidate_epoch.has_value());
    EXPECT_NE(old_epoch->slot_index, candidate_epoch->slot_index);
    EXPECT_EQ(old_epoch->residency_epoch, 40u);
    EXPECT_EQ(candidate_epoch->residency_epoch, 41u);
    EXPECT_EQ(pool->slotForExpert(11, 40), old_epoch->slot_index);
    EXPECT_EQ(pool->slotForExpert(11, 41), candidate_epoch->slot_index);
    EXPECT_FALSE(pool->slotForExpert(11).has_value());
    EXPECT_FALSE(pool->acquire(11, 41).has_value());
    EXPECT_FALSE(pool->acquire(12, 41).has_value());

    const int retired_slot = old_epoch->slot_index;
    old_epoch->lifetime.reset();
    EXPECT_FALSE(pool->slotForExpert(11, 40).has_value());
    EXPECT_EQ(pool->slotForExpert(11, 41), candidate_epoch->slot_index);

    auto reused = pool->acquire(12, /*residency_epoch=*/41);
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(reused->slot_index, retired_slot);
}

TEST(Test__MoEExpertWeightService, GpuDirectSlotPool_TransferSlotsAreSurplusAndReleaseIndependently)
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
        /*layer_idx=*/4,
        /*active_capacity=*/1,
        std::move(specs),
        /*vram_safety_margin_bytes=*/0,
        /*transfer_capacity=*/1);

    EXPECT_EQ(pool->capacity(), 1u);
    EXPECT_EQ(pool->activeCapacity(), 1u);
    EXPECT_EQ(pool->transferCapacity(), 1u);
    EXPECT_EQ(pool->availableSlots(), 1u);
    EXPECT_EQ(pool->availableTransferSlots(), 1u);

    auto active = pool->acquire(21);
    auto transfer = pool->acquireTransferSlot(22);
    ASSERT_TRUE(active.has_value());
    ASSERT_TRUE(transfer.has_value());
    ASSERT_NE(active->lifetime, nullptr);
    ASSERT_NE(transfer->lifetime, nullptr);
    EXPECT_EQ(pool->usedSlots(), 1u);
    EXPECT_EQ(pool->usedTransferSlots(), 1u);
    EXPECT_EQ(pool->availableSlots(), 0u);
    EXPECT_EQ(pool->availableTransferSlots(), 0u);
    EXPECT_EQ(pool->slotForExpert(21), active->slot_index);
    EXPECT_EQ(pool->transferSlotForExpert(22), transfer->slot_index);

    EXPECT_FALSE(pool->acquire(23).has_value());
    EXPECT_FALSE(pool->acquireTransferSlot(24).has_value());

    const int active_slot = active->slot_index;
    active->lifetime.reset();
    EXPECT_EQ(pool->usedSlots(), 0u);
    EXPECT_EQ(pool->usedTransferSlots(), 1u);
    EXPECT_EQ(pool->availableSlots(), 1u);
    EXPECT_EQ(pool->availableTransferSlots(), 0u);
    EXPECT_FALSE(pool->slotForExpert(21).has_value());
    EXPECT_EQ(pool->transferSlotForExpert(22), transfer->slot_index);

    auto reused_active = pool->acquire(23);
    ASSERT_TRUE(reused_active.has_value());
    EXPECT_EQ(reused_active->slot_index, active_slot);

    const int transfer_slot = transfer->slot_index;
    transfer->lifetime.reset();
    EXPECT_EQ(pool->usedTransferSlots(), 0u);
    EXPECT_EQ(pool->availableTransferSlots(), 1u);
    EXPECT_FALSE(pool->transferSlotForExpert(22).has_value());

    auto reused_transfer = pool->acquireTransferSlot(24);
    ASSERT_TRUE(reused_transfer.has_value());
    EXPECT_EQ(reused_transfer->slot_index, transfer_slot);
}

TEST(Test__MoEExpertWeightService, GpuDirectSlotPool_RecommendedCapacityCoversHotCacheChurnAndBatch)
{
    EXPECT_EQ(GpuExpertSlotPool::recommendedCapacity(64, 2), 11);
    EXPECT_EQ(GpuExpertSlotPool::recommendedCapacity(64, 9), 11);
    EXPECT_EQ(GpuExpertSlotPool::recommendedCapacity(64, 20), 20);
    EXPECT_EQ(GpuExpertSlotPool::recommendedCapacity(64, 100), 64);
    EXPECT_EQ(GpuExpertSlotPool::recommendedCapacity(256, 35), 39);
    EXPECT_EQ(GpuExpertSlotPool::recommendedCapacity(1, 0), 1);
}

TEST(Test__MoEExpertWeightService, GpuDirectSlotPool_RecommendedTransferCapacityCoversArrivalBatch)
{
    EXPECT_EQ(GpuExpertSlotPool::recommendedTransferCapacity(64, 0), 11);
    EXPECT_EQ(GpuExpertSlotPool::recommendedTransferCapacity(64, 5), 11);
    EXPECT_EQ(GpuExpertSlotPool::recommendedTransferCapacity(64, 20), 20);
    EXPECT_EQ(GpuExpertSlotPool::recommendedTransferCapacity(64, 100), 64);
    EXPECT_EQ(GpuExpertSlotPool::recommendedTransferCapacity(1, 0), 1);
    EXPECT_EQ(GpuExpertSlotPool::recommendedTransferCapacity(0, 5), 0);
    EXPECT_EQ(GpuExpertSlotPool::recommendedTransferCapacity(256, 25), 39);
}

TEST(Test__MoEExpertWeightService, GpuDirectTransferStagingPool_RecommendedCapacityAndLeaseReuse)
{
    EXPECT_EQ(GpuExpertTransferStagingPool::recommendedCapacity(64, 32), 32);
    EXPECT_EQ(GpuExpertTransferStagingPool::recommendedCapacity(64, 100), 64);
    EXPECT_EQ(GpuExpertTransferStagingPool::recommendedCapacity(1, 0), 1);
    EXPECT_EQ(GpuExpertTransferStagingPool::recommendedCapacity(0, 8), 0);

    std::vector<GpuExpertTransferStagingPool::ProjectionSpec> specs;
    for (const char *label : {"gate", "up", "down"})
    {
        GpuExpertTransferStagingPool::ProjectionSpec spec;
        spec.label = label;
        spec.N = 4;
        spec.K = 32;
        spec.payload_bytes_per_block = 16;
        spec.is_asymmetric = true;
        spec.has_emins = false;
        spec.codebook_id = 7;
        specs.push_back(std::move(spec));
    }

    auto pool = GpuExpertTransferStagingPool::create(
        nullptr,
        DeviceId::cuda(0),
        /*device_ordinal=*/0,
        /*capacity=*/2,
        specs,
        /*vram_safety_margin_bytes=*/0);

    EXPECT_TRUE(pool->compatibleWith(specs));
    auto different_specs = specs;
    different_specs.front().N *= 2;
    EXPECT_FALSE(pool->compatibleWith(different_specs));
    EXPECT_EQ(pool->capacity(), 2u);
    EXPECT_EQ(pool->availableSlots(), 2u);

    auto first = pool->acquire(11);
    auto second = pool->acquire(11);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(first->slot_index, second->slot_index)
        << "the same expert id may be staged concurrently for different layers";
    EXPECT_EQ(pool->availableSlots(), 0u);
    EXPECT_FALSE(pool->acquire(13).has_value());

    const int reusable_slot = first->slot_index;
    first->lifetime.reset();
    EXPECT_EQ(pool->availableSlots(), 1u);
    EXPECT_TRUE(pool->slotForExpert(11).has_value());
    second->lifetime.reset();
    EXPECT_EQ(pool->availableSlots(), 2u);
    EXPECT_FALSE(pool->slotForExpert(11).has_value());

    auto reused = pool->acquire(13);
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(reused->slot_index, reusable_slot);
}

TEST(Test__MoEExpertWeightService, GpuDirectStagedArrivalsExposeStableExpertAndActivationMetadata)
{
    GpuDirectStagedExpertArrivals arrivals;
    arrivals.device_id = DeviceId::cuda(0);
    arrivals.source_device = DeviceId::cuda(1);
    arrivals.layer_idx = 7;

    GpuDirectStagedExpertProjection gate;
    gate.expert_id = 3;
    gate.role = WeightRole::MoEExpertGate;
    gate.engine = fakeGemm(3);
    gate.activation = GpuExpertStagedActivation{};
    arrivals.projections.push_back(gate);

    GpuDirectStagedExpertProjection up;
    up.expert_id = 1;
    up.role = WeightRole::MoEExpertUp;
    up.engine = fakeGemm(11);
    arrivals.projections.push_back(up);

    GpuDirectStagedExpertProjection down;
    down.expert_id = 3;
    down.role = WeightRole::MoEExpertDown;
    down.engine = fakeGemm(33);
    down.activation = GpuExpertStagedActivation{};
    arrivals.projections.push_back(down);

    EXPECT_FALSE(arrivals.empty());
    EXPECT_EQ(arrivals.projectionCount(), 3u);
    EXPECT_EQ(arrivals.activationCount(), 2u);

    const auto ids = arrivals.expertIds();
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 1);
    EXPECT_EQ(ids[1], 3);

    auto activation_batch =
        MoEExpertWeightService::activationBatchForStagedArrivals(arrivals);
    EXPECT_EQ(activation_batch.size(), 2u);
}

TEST(Test__MoEExpertWeightService, GpuDirectTransferSlotArrivalsExposeScratchOnlyMetadata)
{
    auto transfer_lifetime_a = std::shared_ptr<void>(
        new int(1),
        [](void *ptr)
        {
            delete static_cast<int *>(ptr);
        });
    auto transfer_lifetime_b = std::shared_ptr<void>(
        new int(2),
        [](void *ptr)
        {
            delete static_cast<int *>(ptr);
        });
    auto completion_lifetime = std::shared_ptr<void>(
        new int(3),
        [](void *ptr)
        {
            delete static_cast<int *>(ptr);
        });

    GpuDirectTransferSlotArrivals arrivals;
    arrivals.device_id = DeviceId::cuda(0);
    arrivals.source_device = DeviceId::cuda(1);
    arrivals.layer_idx = 9;
    arrivals.projections.push_back(fakeTransferSlotProjection(
        3, WeightRole::MoEExpertGate, 0x100000, transfer_lifetime_a));
    arrivals.projections.push_back(fakeTransferSlotProjection(
        1, WeightRole::MoEExpertUp, 0x200000, transfer_lifetime_b));
    arrivals.projections.push_back(fakeTransferSlotProjection(
        3, WeightRole::MoEExpertDown, 0x300000, transfer_lifetime_a));
    arrivals.completion.transient_lifetimes.push_back(transfer_lifetime_a);
    arrivals.completion.transient_lifetimes.push_back(completion_lifetime);

    ASSERT_FALSE(arrivals.empty());
    EXPECT_EQ(arrivals.projectionCount(), 3u);
    EXPECT_EQ(arrivals.totalBytes(),
              arrivals.projections[0].bytes() +
                  arrivals.projections[1].bytes() +
                  arrivals.projections[2].bytes());

    for (const auto &projection : arrivals.projections)
        EXPECT_TRUE(projection.valid());

    const auto ids = arrivals.expertIds();
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 1);
    EXPECT_EQ(ids[1], 3);

    const auto lifetimes = arrivals.transferSlotLifetimes();
    ASSERT_EQ(lifetimes.size(), 3u);
    EXPECT_NE(std::find(lifetimes.begin(), lifetimes.end(), transfer_lifetime_a), lifetimes.end());
    EXPECT_NE(std::find(lifetimes.begin(), lifetimes.end(), transfer_lifetime_b), lifetimes.end());
    EXPECT_NE(std::find(lifetimes.begin(), lifetimes.end(), completion_lifetime), lifetimes.end());
}

TEST(Test__MoEExpertWeightService, GpuDirectTransferSlotProjectionRequiresScratchLifetime)
{
    auto lifetime = std::shared_ptr<void>(
        new int(1),
        [](void *ptr)
        {
            delete static_cast<int *>(ptr);
        });

    auto projection = fakeTransferSlotProjection(
        0, WeightRole::MoEExpertGate, 0x400000, lifetime);
    EXPECT_TRUE(projection.valid());

    projection.transfer_slot_lifetime.reset();
    EXPECT_FALSE(projection.valid())
        << "transfer-slot arrivals must pin scratch VRAM until activation completion";
}

TEST(Test__MoEExpertWeightService, GpuDirectTransferSlotActivationRejectsNullStreamBeforePublish)
{
    TestWeightContextOwner owner;
    owner.device_id = DeviceId::cuda(0);
    owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
    owner.prepared_up_gemm.assign(kNumExperts, nullptr);
    owner.prepared_down_gemm.assign(kNumExperts, nullptr);
    auto ctx = owner.buildContext();

    auto lifetime = std::shared_ptr<void>(
        new int(1),
        [](void *ptr)
        {
            delete static_cast<int *>(ptr);
        });
    GpuDirectTransferSlotArrivals arrivals;
    arrivals.device_id = owner.device_id;
    arrivals.layer_idx = ctx.layer_idx;
    arrivals.projections.push_back(fakeTransferSlotProjection(
        0, WeightRole::MoEExpertGate, 0x500000, lifetime));

    GpuDirectStagedExpertArrivals activated;
    EXPECT_FALSE(MoEExpertWeightService::activateGpuDirectTransferSlotArrivals(
        ctx,
        arrivals,
        nullptr,
        &activated));
    EXPECT_TRUE(activated.empty());
    EXPECT_EQ(owner.prepared_gate_gemm[0], nullptr);
    EXPECT_TRUE(owner.moe_owned_kernels.empty());
}

TEST(Test__MoEExpertWeightService, GpuDirectTransferSlotActivationRejectsMissingActivePoolBeforePublish)
{
    TestWeightContextOwner owner;
    owner.device_id = DeviceId::cuda(0);
    owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
    owner.prepared_up_gemm.assign(kNumExperts, nullptr);
    owner.prepared_down_gemm.assign(kNumExperts, nullptr);
    auto ctx = owner.buildContext();

    auto lifetime = std::shared_ptr<void>(
        new int(1),
        [](void *ptr)
        {
            delete static_cast<int *>(ptr);
        });
    GpuDirectTransferSlotArrivals arrivals;
    arrivals.device_id = owner.device_id;
    arrivals.layer_idx = ctx.layer_idx;
    arrivals.projections.push_back(fakeTransferSlotProjection(
        0, WeightRole::MoEExpertGate, 0x600000, lifetime));

    GpuDirectStagedExpertArrivals activated;
    void *fake_explicit_stream = reinterpret_cast<void *>(0x1234);
    EXPECT_FALSE(MoEExpertWeightService::activateGpuDirectTransferSlotArrivals(
        ctx,
        arrivals,
        fake_explicit_stream,
        &activated));
    EXPECT_TRUE(activated.empty());
    EXPECT_EQ(owner.prepared_gate_gemm[0], nullptr);
    EXPECT_TRUE(owner.moe_owned_kernels.empty());
}

TEST(Test__MoEExpertWeightService, GpuDirectTransferSlotStagingIsTransactionalBeforePublish)
{
    const std::string source = readMoEWeightServiceSource();
    ASSERT_FALSE(source.empty());

    const auto fn_pos =
        source.find("bool MoEExpertWeightService::stageExpertsGPUDirectToTransferSlots");
    ASSERT_NE(fn_pos, std::string::npos);

    const auto capacity_reject_pos =
        source.find("gpu_direct_transfer_staging_pool_capacity_rejects", fn_pos);
    const auto transfer_begin_pos =
        source.find("transfer_batch.begin", fn_pos);
    const auto staged_success_pos =
        source.find("staged_satisfied.push_back", fn_pos);
    const auto requested_active_capacity_pos =
        source.find("requested_active_capacity =\n            std::max(experts_to_stage.size(), active_arrival_capacity)", fn_pos);
    const auto active_capacity_uses_requested_pos =
        source.find("recommendedCapacity(\n                dst_ctx.num_experts,\n                requested_active_capacity)", fn_pos);
    const auto requested_staging_capacity_pos =
        source.find("requested_staging_capacity =\n            std::max(experts_to_stage.size(), staging_pool_capacity)", fn_pos);
    const auto staging_capacity_uses_requested_pos =
        source.find("GpuExpertTransferStagingPool::recommendedCapacity(\n                dst_ctx.num_experts,\n                requested_staging_capacity)", fn_pos);
    const auto staging_capacity_downshift_pos =
        source.find("gpu_direct_transfer_staging_pool_capacity_downshifts", fn_pos);
    const auto staging_wave_trim_pos =
        source.find("experts_to_copy.resize(available_staging_slots)", fn_pos);
    const auto compatible_pool_scan_pos =
        source.find("for (auto &pool : *transfer_staging_pools)", fn_pos);
    const auto heterogeneous_shape_abort_pos =
        source.find("heterogeneous GPU expert staging pools are not supported", fn_pos);
    const auto record_failure_pos =
        source.find("GPU-direct transfer-slot staging completion event record failed", fn_pos);
    const auto publish_already_after_record_failure =
        source.find("publish_already_satisfied();", record_failure_pos);
    const auto carrier_commit_pos =
        source.find("*staged_arrivals = std::move(out)", fn_pos);
    const auto publish_success_pos =
        source.find("publish_success();", carrier_commit_pos);

    ASSERT_NE(capacity_reject_pos, std::string::npos);
    ASSERT_NE(transfer_begin_pos, std::string::npos);
    ASSERT_NE(staged_success_pos, std::string::npos);
    ASSERT_NE(requested_active_capacity_pos, std::string::npos);
    ASSERT_NE(active_capacity_uses_requested_pos, std::string::npos);
    ASSERT_NE(requested_staging_capacity_pos, std::string::npos);
    ASSERT_NE(staging_capacity_uses_requested_pos, std::string::npos);
    ASSERT_NE(staging_capacity_downshift_pos, std::string::npos);
    ASSERT_NE(staging_wave_trim_pos, std::string::npos);
    ASSERT_NE(compatible_pool_scan_pos, std::string::npos);
    ASSERT_EQ(heterogeneous_shape_abort_pos, std::string::npos);
    ASSERT_NE(record_failure_pos, std::string::npos);
    ASSERT_NE(publish_already_after_record_failure, std::string::npos);
    ASSERT_NE(carrier_commit_pos, std::string::npos);
    ASSERT_NE(publish_success_pos, std::string::npos);

    EXPECT_LT(capacity_reject_pos, transfer_begin_pos)
        << "oversized staging-pool batches should fail before stream/event/copy work";
    EXPECT_LT(requested_active_capacity_pos, active_capacity_uses_requested_pos)
        << "active pools must be sized independently from the rolling staging wave";
    EXPECT_LT(requested_staging_capacity_pos, staging_capacity_uses_requested_pos)
        << "staging pools must be sized from the explicit rolling-wave budget";
    EXPECT_LT(staging_capacity_downshift_pos, transfer_begin_pos)
        << "tight VRAM should downshift staging capacity before stream/event/copy work";
    EXPECT_LT(staging_wave_trim_pos, transfer_begin_pos)
        << "reduced-capacity staging pools should trim the wave before enqueue";
    EXPECT_LT(staged_success_pos, carrier_commit_pos)
        << "staged experts are only publishable after the arrival carrier is committed";
    EXPECT_LT(publish_already_after_record_failure, carrier_commit_pos)
        << "pre-commit failures may report already-resident experts, not partial staged copies";
    EXPECT_LT(carrier_commit_pos, publish_success_pos)
        << "new staged experts become satisfied only after the completion carrier is published";
}

TEST(Test__MoEExpertWeightService, GpuRebalanceResolvesPreparedStoreSlabBeforeRawFallback)
{
    const std::string source = readMoEWeightServiceSource();
    ASSERT_FALSE(source.empty());

    const auto fn_pos =
        source.find("bool MoEExpertWeightService::registerAndPrepareNewExpertsGPU");
    ASSERT_NE(fn_pos, std::string::npos);

    const auto slab_fallback_pos =
        source.find("findExpertSlab(makeExpertSlabDescriptor(ctx, role))", fn_pos);
    const auto raw_fallback_error_pos =
        source.find("requires transferred/provider blobs for all gate/up/down weights", fn_pos);

    ASSERT_NE(slab_fallback_pos, std::string::npos)
        << "GPU rebalance must re-resolve prepared-store slabs when duplicate graph stages have stale refs";
    ASSERT_NE(raw_fallback_error_pos, std::string::npos);
    EXPECT_LT(slab_fallback_pos, raw_fallback_error_pos)
        << "prepared-store arrivals must be checked before hard-failing into the no-raw-fallback path";
}

TEST(Test__MoEExpertWeightService, ExpertSlabDescriptorsUseRoleSpecificProjectionShape)
{
    const std::string source = readMoEWeightServiceSource();
    ASSERT_FALSE(source.empty());

    const auto helper_pos =
        source.find("static ExpertSlabDescriptor makeExpertSlabDescriptor");
    ASSERT_NE(helper_pos, std::string::npos);

    const auto down_branch_pos =
        source.find("role == WeightRole::MoEExpertDown", helper_pos);
    const auto down_rows_pos =
        source.find("desc.rows_per_expert = ctx.d_model", down_branch_pos);
    const auto down_cols_pos =
        source.find("desc.cols_per_expert = ctx.expert_intermediate", down_branch_pos);
    const auto gateup_rows_pos =
        source.find("desc.rows_per_expert = ctx.expert_intermediate", down_cols_pos);
    const auto gateup_cols_pos =
        source.find("desc.cols_per_expert = ctx.d_model", gateup_rows_pos);
    const auto prepare_lookup_pos =
        source.find("return makeExpertSlabDescriptor(ctx, role)", helper_pos);
    const auto initial_register_pos =
        source.find("ExpertSlabDescriptor desc = makeExpertSlabDescriptor(ctx, role)", prepare_lookup_pos);

    ASSERT_NE(down_branch_pos, std::string::npos)
        << "down expert slabs must not reuse gate/up descriptor shape";
    ASSERT_NE(down_rows_pos, std::string::npos);
    ASSERT_NE(down_cols_pos, std::string::npos);
    ASSERT_NE(gateup_rows_pos, std::string::npos);
    ASSERT_NE(gateup_cols_pos, std::string::npos);
    ASSERT_NE(prepare_lookup_pos, std::string::npos)
        << "initial store lookup must use the shared descriptor helper";
    ASSERT_NE(initial_register_pos, std::string::npos)
        << "initial CPU/GPU slab registration must use the shared descriptor helper";

    EXPECT_LT(down_branch_pos, down_rows_pos);
    EXPECT_LT(down_rows_pos, down_cols_pos);
    EXPECT_LT(down_cols_pos, gateup_rows_pos);
    EXPECT_LT(gateup_rows_pos, gateup_cols_pos);
}

TEST(Test__MoEExpertWeightService, GpuDirectStagedArrivalsInstallOnlyWhenActivated)
{
    TestWeightContextOwner owner;
    owner.device_id = DeviceId::cuda(0);
    owner.expert_mask.assign(kNumExperts, false);
    owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
    owner.prepared_up_gemm.assign(kNumExperts, nullptr);
    owner.prepared_down_gemm.assign(kNumExperts, nullptr);

    PreparedWeightStore store;
    ExpertGemmRegistry registry;
    owner.prepared_store = &store;
    owner.expert_registry = &registry;
    auto ctx = owner.buildContext();

    const int expert_id = 2;
    auto gate_owner = std::make_shared<OwnedFakeGemm>(std::make_shared<int>(0));
    auto up_owner = std::make_shared<OwnedFakeGemm>(std::make_shared<int>(0));
    auto down_owner = std::make_shared<OwnedFakeGemm>(std::make_shared<int>(0));

    GpuDirectStagedExpertArrivals arrivals;
    arrivals.device_id = owner.device_id;
    arrivals.source_device = DeviceId::cuda(1);
    arrivals.layer_idx = ctx.layer_idx;
    arrivals.completion.device_id = owner.device_id;
    arrivals.completion.device_ordinal = 0;
    arrivals.completion.ready_event = std::shared_ptr<void>(
        new int(1),
        [](void *ptr)
        {
            delete static_cast<int *>(ptr);
        });
    arrivals.completion.transient_lifetimes.push_back(std::shared_ptr<void>(
        new int(2),
        [](void *ptr)
        {
            delete static_cast<int *>(ptr);
        }));

    auto add_projection = [&](WeightRole role, const std::shared_ptr<ITensorGemm> &engine)
    {
        GpuDirectStagedExpertProjection projection;
        projection.expert_id = expert_id;
        projection.role = role;
        projection.engine = engine.get();
        projection.engine_lifetime = engine;
        arrivals.projections.push_back(std::move(projection));
    };
    add_projection(WeightRole::MoEExpertGate, gate_owner);
    add_projection(WeightRole::MoEExpertUp, up_owner);
    add_projection(WeightRole::MoEExpertDown, down_owner);

    EXPECT_EQ(owner.prepared_gate_gemm[expert_id], nullptr)
        << "staging itself must not publish a live expert engine";

    ASSERT_TRUE(MoEExpertWeightService::installActivatedGpuDirectArrivals(ctx, arrivals));

    EXPECT_EQ(owner.prepared_gate_gemm[expert_id], gate_owner.get());
    EXPECT_EQ(owner.prepared_up_gemm[expert_id], up_owner.get());
    EXPECT_EQ(owner.prepared_down_gemm[expert_id], down_owner.get());
    EXPECT_EQ(owner.moe_owned_kernels.size(), 3u);

    EXPECT_EQ(registry.getEngine(owner.device_id, ctx.layer_idx, expert_id,
                                 ExpertGemmRegistry::WeightRole::GATE),
              gate_owner.get());
    EXPECT_EQ(registry.getEngine(owner.device_id, ctx.layer_idx, expert_id,
                                 ExpertGemmRegistry::WeightRole::UP),
              up_owner.get());
    EXPECT_EQ(registry.getEngine(owner.device_id, ctx.layer_idx, expert_id,
                                 ExpertGemmRegistry::WeightRole::DOWN),
              down_owner.get());

    ASSERT_TRUE(ctx.gate_slab_ref.has_value());
    auto stored_gate = store.expertGemmKernel(*ctx.gate_slab_ref, expert_id);
    EXPECT_EQ(stored_gate, gate_owner.get());
    auto stored_completion = store.expertGpuDirectCompletion(*ctx.gate_slab_ref, expert_id);
    ASSERT_TRUE(stored_completion.has_value());
    EXPECT_TRUE(stored_completion->valid());
    EXPECT_TRUE(stored_completion->transient_lifetimes.empty())
        << "PreparedWeightStore must not pin short-lived transfer slots for whole expert residency";
}

TEST(Test__MoEExpertWeightService, GpuDirectStagedArrivalsRejectMismatchedDeviceBeforeMutation)
{
    TestWeightContextOwner owner;
    owner.device_id = DeviceId::cuda(0);
    owner.expert_mask.assign(kNumExperts, false);
    owner.prepared_gate_gemm.assign(kNumExperts, nullptr);
    owner.prepared_up_gemm.assign(kNumExperts, nullptr);
    owner.prepared_down_gemm.assign(kNumExperts, nullptr);
    auto ctx = owner.buildContext();

    auto gate_owner = std::make_shared<OwnedFakeGemm>(std::make_shared<int>(0));
    GpuDirectStagedExpertArrivals arrivals;
    arrivals.device_id = DeviceId::cuda(1);
    arrivals.layer_idx = ctx.layer_idx;

    GpuDirectStagedExpertProjection projection;
    projection.expert_id = 0;
    projection.role = WeightRole::MoEExpertGate;
    projection.engine = gate_owner.get();
    projection.engine_lifetime = gate_owner;
    arrivals.projections.push_back(std::move(projection));

    EXPECT_FALSE(MoEExpertWeightService::installActivatedGpuDirectArrivals(ctx, arrivals));
    EXPECT_EQ(owner.prepared_gate_gemm[0], nullptr);
    EXPECT_TRUE(owner.moe_owned_kernels.empty());
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

TEST(Test__MoEExpertWeightService, RegisterAndPrepareNewExperts_DirectPreparedArrivalIsShared)
{
    TestWeightContextOwner source;
    source.expert_mask = {true, true, true, true};
    {
        auto ctx = source.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    ExpertPackedWeights packed;
    {
        auto ctx = source.buildContext();
        packed = MoEExpertWeightService::clonePreparedExpert(ctx, 1);
    }
    ASSERT_TRUE(packed.complete());

    PreparedExpertEngines arrival;
    arrival.packed_bytes = packed.totalBytes();
    arrival.gate = KernelFactory::createExpertGemmFromPackedWeights(
        std::move(packed.gate));
    arrival.up = KernelFactory::createExpertGemmFromPackedWeights(
        std::move(packed.up));
    arrival.down = KernelFactory::createExpertGemmFromPackedWeights(
        std::move(packed.down));
    ASSERT_TRUE(arrival.complete());
    std::unordered_map<int, PreparedExpertEngines> received_prepared;
    received_prepared.emplace(1, arrival);

    TestWeightContextOwner destination;
    destination.expert_mask = {true, false, true, false};
    {
        auto ctx = destination.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
    }

    std::vector<bool> new_mask = {true, true, true, false};
    {
        auto ctx = destination.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::registerAndPrepareNewExperts(
            ctx, new_mask, nullptr, &received_prepared));
    }

    EXPECT_EQ(destination.prepared_gate_gemm[1], arrival.gate.get());
    EXPECT_EQ(destination.prepared_up_gemm[1], arrival.up.get());
    EXPECT_EQ(destination.prepared_down_gemm[1], arrival.down.get());
}

TEST(Test__MoEExpertWeightService, RegisterAndPrepareNewExperts_SerializedCPUArrivalIsAccepted)
{
    TestWeightContextOwner source;
    source.expert_mask = {true, true, true, true};
    ExpertWeightBlobs serialized;
    {
        auto ctx = source.buildContext();
        ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
        ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));
        serialized = MoEExpertWeightService::serializeExpert(ctx, 1);
    }
    ASSERT_FALSE(serialized.empty());

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
    received.emplace(1, std::move(serialized));

    std::vector<bool> new_mask = {true, true, true, false};
    {
        auto ctx = owner.buildContext();
        EXPECT_TRUE(MoEExpertWeightService::registerAndPrepareNewExperts(
            ctx, new_mask, &received, nullptr));
    }

    EXPECT_NE(owner.prepared_gate_gemm[1], nullptr);
    EXPECT_NE(owner.prepared_up_gemm[1], nullptr);
    EXPECT_NE(owner.prepared_down_gemm[1], nullptr);
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

TEST(Test__MoEExpertWeightService, GpuDirectTransferSlotStagingUsesSourceDescriptorFormat)
{
    const std::string source =
        readRepoFile("src/v2/execution/moe/MoEExpertWeightService.cpp");
    ASSERT_FALSE(source.empty());

    const size_t staging_fn =
        source.find("bool MoEExpertWeightService::stageExpertsGPUDirectToTransferSlots");
    ASSERT_NE(staging_fn, std::string::npos);

    const size_t experts_to_copy =
        source.find("std::vector<int> experts_to_copy", staging_fn);
    ASSERT_NE(experts_to_copy, std::string::npos);
    const size_t make_specs =
        source.find("auto make_slot_pool_specs", experts_to_copy);
    ASSERT_NE(make_specs, std::string::npos);

    const std::string admission_block =
        source.substr(experts_to_copy, make_specs - experts_to_copy);
    EXPECT_NE(admission_block.find("deviceMoEProjectionFormat"), std::string::npos)
        << "inactive destination expert views may not carry NativeVNNI metadata";
    EXPECT_EQ(admission_block.find("has no NativeVNNI format info"), std::string::npos)
        << "GPU-direct arrivals must not require destination VNNI format metadata";
    EXPECT_EQ(admission_block.find("grp.dst_views[expert_id]->rows()"), std::string::npos)
        << "inactive destination expert views may be absent during transfer-slot staging";

    const size_t allocation_ns =
        source.find("uint64_t allocation_ns", make_specs);
    ASSERT_NE(allocation_ns, std::string::npos);
    const std::string spec_block =
        source.substr(make_specs, allocation_ns - make_specs);
    const size_t source_descriptor_call =
        spec_block.find("source_descriptor_for(");
    ASSERT_NE(source_descriptor_call, std::string::npos);
    EXPECT_NE(
        spec_block.find("sample_expert", source_descriptor_call),
        std::string::npos)
        << "slot specs must resolve the sample expert through the live source descriptor";
    EXPECT_NE(spec_block.find("deviceMoEProjectionFormat"), std::string::npos);
    EXPECT_EQ(spec_block.find("vnni_info_for(grp, sample_expert"), std::string::npos)
        << "slot pool specs must come from the source descriptor for inactive arrivals";
    EXPECT_EQ(spec_block.find("grp.dst_views[static_cast<size_t>(sample_expert)]->rows()"), std::string::npos)
        << "slot pool specs must not require an already-materialized destination view";

    const size_t copy_loop =
        source.find("for (int expert_id : experts_to_copy)", allocation_ns);
    ASSERT_NE(copy_loop, std::string::npos);
    const size_t publish_stats =
        source.find("PerfStatsCollector::Tags tags", copy_loop);
    ASSERT_NE(publish_stats, std::string::npos);
    const std::string copy_block =
        source.substr(copy_loop, publish_stats - copy_loop);
    EXPECT_EQ(copy_block.find("vnni_info_for(grp, expert_id"), std::string::npos)
        << "async transfer-slot enqueue must use source descriptor metadata";
    EXPECT_EQ(copy_block.find("grp.dst_views[expert_id]->rows()"), std::string::npos)
        << "async transfer-slot enqueue must not dereference absent destination views";
}
