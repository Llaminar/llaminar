/**
 * @file Test__MoELocalExpertStage_PreparedWeights.cpp
 * @brief Phase 12 unit tests — MoELocalExpertStage prepared-weight validation.
 *
 * Verifies that:
 *   1. validatePreparedWeights() succeeds with complete prepared engine vectors.
 *   2. validatePreparedWeights() fails when an active expert is missing an engine.
 *   3. validatePreparedWeights() succeeds with slab-refs pointing to a registered store.
 *   4. validatePreparedWeights() fails when a slab is not in the store (empty mask).
 *   5. Params has only the allowed MoE runtime-table hook, not runner/peer fields.
 */

#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/moe/MoERuntimeTable.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/ExpertSlabTypes.h"
#include "mocks/MockComputeStage.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

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

    class FakePreparedGemm final : public ITensorGemm, public IWorkspaceConsumer
    {
    public:
        explicit FakePreparedGemm(DeviceNativeVNNIMatrixDesc desc)
            : desc_(desc)
        {
        }

        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0;
        }

        bool multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const TensorBase *bias = nullptr,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr,
            int activation_row_offset = 0) override
        {
            (void)A;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)bias;
            (void)mpi_ctx;
            (void)device_idx;
            (void)workspace;
            (void)activation_row_offset;
            return false;
        }

        bool weights_converted() const override { return true; }

        bool exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out) override
        {
            out = desc_;
            return out.valid();
        }

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override
        {
            WorkspaceRequirements reqs;
            reqs.buffers.push_back({
                "fake_prepared_gemm_workspace",
                static_cast<size_t>(std::max(1, m)) *
                    static_cast<size_t>(std::max(1, n)) *
                    static_cast<size_t>(std::max(1, k)),
                256,
                true});
            return reqs;
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override { workspace_ = workspace; }
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

    private:
        DeviceNativeVNNIMatrixDesc desc_;
        DeviceWorkspaceManager *workspace_ = nullptr;
    };

    DeviceNativeVNNIMatrixDesc nativeDesc(int expert_id, int role, int n, int k)
    {
        const uintptr_t base = 0x10000000u + static_cast<uintptr_t>(expert_id) * 0x10000u +
                               static_cast<uintptr_t>(role) * 0x1000u;
        DeviceNativeVNNIMatrixDesc desc;
        desc.payload = reinterpret_cast<const uint8_t *>(base + 0x0100u);
        desc.scales = reinterpret_cast<const void *>(base + 0x0200u);
        desc.mins = reinterpret_cast<const void *>(base + 0x0300u);
        desc.n = n;
        desc.k = k;
        desc.blocks_per_row = static_cast<uint32_t>(std::max(k / 32, 1));
        desc.codebook_id = 4;
        return desc;
    }

    void attachFakePreparedEngines(
        MoELocalExpertStage::Params &p,
        std::vector<std::unique_ptr<FakePreparedGemm>> &owned)
    {
        p.prepared_gate_gemm.assign(static_cast<size_t>(p.num_experts), nullptr);
        p.prepared_up_gemm.assign(static_cast<size_t>(p.num_experts), nullptr);
        p.prepared_down_gemm.assign(static_cast<size_t>(p.num_experts), nullptr);
        owned.clear();
        owned.reserve(static_cast<size_t>(p.num_experts) * 3u);

        for (int expert_id = 0; expert_id < p.num_experts; ++expert_id)
        {
            owned.push_back(std::make_unique<FakePreparedGemm>(
                nativeDesc(expert_id, 0, p.expert_intermediate, p.d_model)));
            p.prepared_gate_gemm[static_cast<size_t>(expert_id)] = owned.back().get();

            owned.push_back(std::make_unique<FakePreparedGemm>(
                nativeDesc(expert_id, 1, p.expert_intermediate, p.d_model)));
            p.prepared_up_gemm[static_cast<size_t>(expert_id)] = owned.back().get();

            owned.push_back(std::make_unique<FakePreparedGemm>(
                nativeDesc(expert_id, 2, p.d_model, p.expert_intermediate)));
            p.prepared_down_gemm[static_cast<size_t>(expert_id)] = owned.back().get();
        }
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

TEST(Test__MoELocalExpertStage_PreparedWeights,
     RuntimeTableInitializesOverlayPlacementBankForLocalMask)
{
    constexpr int kNumExperts = 4;
    MoERuntimeTable table(DeviceId::cpu(), 1, kNumExperts, 2);

    MoELocalExpertStage::Params p;
    p.device_id = DeviceId::cpu();
    p.num_experts = kNumExperts;
    p.top_k = 2;
    p.d_model = 32;
    p.expert_intermediate = 64;
    p.layer_idx = 0;
    p.expert_mask = {true, false, true, false};
    p.runtime_participant_index = 3;
    p.moe_runtime_table = &table;

    std::vector<std::unique_ptr<FakePreparedGemm>> owned;
    attachFakePreparedEngines(p, owned);

    MoELocalExpertStage stage(p);
    EXPECT_FALSE(stage.isGraphCapturable());
    EXPECT_TRUE(stage.refreshRuntimePlacement());

    const auto &state = table.hostLayerState(0);
    ASSERT_EQ(state.active_epoch, 1u);
    ASSERT_LE(state.active_bank, 1u);
    const auto &bank = state.banks[state.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_EQ(bank.local_compute_mask[1], 0u);
    EXPECT_EQ(bank.local_compute_mask[2], 1u);
    EXPECT_EQ(bank.local_compute_mask[3], 0u);
    EXPECT_EQ(bank.experts[0].logical_expert_id, 0);
    EXPECT_EQ(bank.experts[0].owner_participant, 3);
    EXPECT_TRUE(bank.experts[0].gate.valid());
    EXPECT_TRUE(bank.experts[2].down.valid());
    EXPECT_TRUE(hasMoEExpertFlag(bank.experts[2].flags, DeviceMoEExpertFlags::LocalCompute));
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     RuntimeTableEmptyMaskNoOpsWithoutPreparedWeights)
{
    constexpr int kNumExperts = 4;
    constexpr int kDModel = 8;
    constexpr int kTopK = 2;

    MoERuntimeTable table(DeviceId::cpu(), 1, kNumExperts, kTopK);
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(/*max_rows=*/2, /*max_entries=*/4, kDModel, kTopK, DeviceId::cpu());

    auto input = workspace.localExpertInput(0, 0);
    input.live_row_count = 1;
    input.live_entry_count = 1;
    input.row_ids_host[0] = 0;
    input.entry_offsets_host[0] = 0;
    input.entry_offsets_host[1] = 1;
    input.expert_ids_host[0] = 2;
    input.route_weights_host[0] = 1.0f;
    for (int col = 0; col < kDModel; ++col)
        input.hidden_rows_fp32[col] = static_cast<float>(col + 1);

    auto output = workspace.localExpertOutput(0, 0);
    MoELocalExpertStage::Params p;
    p.device_id = DeviceId::cpu();
    p.input_rows = &input;
    p.output_rows = &output;
    p.num_experts = kNumExperts;
    p.top_k = kTopK;
    p.d_model = kDModel;
    p.expert_intermediate = 32;
    p.layer_idx = 0;
    p.expert_mask = {false, false, false, false};
    p.moe_runtime_table = &table;

    MoELocalExpertStage stage(p);
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
    ASSERT_TRUE(stage.execute(&ctx));
    EXPECT_EQ(output.live_row_count, 0u);

    const auto &state = table.hostLayerState(0);
    const auto &bank = state.banks[state.active_bank];
    for (int expert = 0; expert < kNumExperts; ++expert)
        EXPECT_EQ(bank.local_compute_mask[static_cast<size_t>(expert)], 0u);
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ExecuteFailsOnNonFiniteSparseHiddenBeforeExpertDispatch)
{
    constexpr int kNumExperts = 4;
    constexpr int kDModel = 8;
    constexpr int kTopK = 2;

    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(/*max_rows=*/1, /*max_entries=*/2, kDModel, kTopK, DeviceId::cpu());

    auto input = workspace.localExpertInput(0, 0);
    input.live_row_count = 1;
    input.live_entry_count = 1;
    input.row_ids_host[0] = 0;
    input.entry_offsets_host[0] = 0;
    input.entry_offsets_host[1] = 1;
    input.expert_ids_host[0] = 0;
    input.route_weights_host[0] = 1.0f;
    for (int col = 0; col < kDModel; ++col)
        input.hidden_rows_fp32[col] = static_cast<float>(col + 1);
    input.hidden_rows_fp32[3] = std::numeric_limits<float>::quiet_NaN();

    auto output = workspace.localExpertOutput(0, 0);
    MoELocalExpertStage::Params p = makePreparedVectorParams(kNumExperts);
    p.device_id = DeviceId::cpu();
    p.input_rows = &input;
    p.output_rows = &output;
    p.d_model = kDModel;
    p.top_k = kTopK;
    p.expert_intermediate = 64;

    MoELocalExpertStage stage(p);
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
    EXPECT_FALSE(stage.execute(&ctx));
    EXPECT_EQ(output.live_row_count, 0u);
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ExecuteFailsWhenGpuStageDeviceDoesNotMatchExecutionContext)
{
    MoELocalExpertStage::Params p = makePreparedVectorParams(4);
    p.device_id = DeviceId::rocm(1);

    MoELocalExpertStage stage(p);
    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);
    EXPECT_FALSE(stage.execute(&ctx));
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ROCmWorkspaceRequirementsDeclareNestedMoEAndProjectedGemmScratch)
{
    constexpr int kNumExperts = 4;
    constexpr int kRows = 4;
    constexpr int kTopK = 3;
    constexpr int kDModel = 32;
    constexpr int kIntermediate = 64;

    MoEOverlayCollectiveWorkspace overlay_workspace;
    overlay_workspace.ensureCapacity(
        /*max_rows=*/kRows,
        /*max_entries=*/kRows * kTopK,
        kDModel,
        kTopK,
        DeviceId::cpu());

    auto input = overlay_workspace.localExpertInput(0, 0);
    auto output = overlay_workspace.localExpertOutput(0, 0);

    MoELocalExpertStage::Params p;
    p.device_id = DeviceId::rocm(0);
    p.input_rows = &input;
    p.output_rows = &output;
    p.num_experts = kNumExperts;
    p.top_k = kTopK;
    p.d_model = kDModel;
    p.expert_intermediate = kIntermediate;
    p.layer_idx = 0;

    std::vector<std::unique_ptr<FakePreparedGemm>> owned;
    attachFakePreparedEngines(p, owned);

    MoELocalExpertStage stage(p);
    const WorkspaceRequirements reqs = stage.getWorkspaceRequirements(kRows, 0, 0);

    const auto *group_indices = reqs.find(MoEWorkspaceBuffers::GROUP_INT_INDICES);
    ASSERT_NE(group_indices, nullptr)
        << "ROCm graph-native local experts delegate to MoEExpertComputeStage at "
           "execute time, so they must declare grouped-MoE scratch during graph "
           "workspace allocation.";
    EXPECT_GE(group_indices->size_bytes,
              static_cast<size_t>(kRows * kTopK) * sizeof(int));

    const auto *shared_gate = reqs.find(MoEWorkspaceBuffers::ROCM_SHARED_GATE);
    ASSERT_NE(shared_gate, nullptr);
    EXPECT_GE(shared_gate->size_bytes,
              static_cast<size_t>(kRows * kTopK) * sizeof(float));

    const auto *gemm_workspace = reqs.find("fake_prepared_gemm_workspace");
    ASSERT_NE(gemm_workspace, nullptr);
    EXPECT_GE(gemm_workspace->size_bytes,
              static_cast<size_t>(kRows * kTopK) *
                  static_cast<size_t>(kDModel) *
                  static_cast<size_t>(kIntermediate));
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     RuntimeTableRejectsActiveBankOutsideStaticExpertMask)
{
    constexpr int kNumExperts = 4;
    MoERuntimeTable table(DeviceId::cpu(), 1, kNumExperts, 2);

    MoELocalExpertStage::Params all_local;
    all_local.device_id = DeviceId::cpu();
    all_local.num_experts = kNumExperts;
    all_local.top_k = 2;
    all_local.d_model = 32;
    all_local.expert_intermediate = 64;
    all_local.layer_idx = 0;
    all_local.expert_mask = {true, true, true, true};
    all_local.moe_runtime_table = &table;

    std::vector<std::unique_ptr<FakePreparedGemm>> owned;
    attachFakePreparedEngines(all_local, owned);

    MoELocalExpertStage all_local_stage(all_local);
    ASSERT_TRUE(all_local_stage.refreshRuntimePlacement());

    MoELocalExpertStage::Params subset = all_local;
    subset.expert_mask = {true, false, true, false};
    MoELocalExpertStage subset_stage(subset);
    EXPECT_FALSE(subset_stage.refreshRuntimePlacement());
}

// ---------------------------------------------------------------------------
// Structural: Params must not have forbidden runtime/peer fields
// ---------------------------------------------------------------------------

TEST(Test__MoELocalExpertStage_PreparedWeights, ParamsHasOnlyMoERuntimeTableAndNoRunnerFields)
{
    using P = MoELocalExpertStage::Params;
    EXPECT_FALSE(has_runtime<P>::value);
    EXPECT_TRUE((std::is_member_object_pointer_v<decltype(&P::moe_runtime_table)>));
    EXPECT_FALSE(has_peer_participants<P>::value);
    EXPECT_FALSE(has_prepared_participants<P>::value);
}
