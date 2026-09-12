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
#include "tensors/Tensors.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
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

    /**
     * @brief Deterministic prepared expert used to prove canonical route wiring.
     *
     * Gate/up projections merely satisfy the grouped CPU pipeline. The down
     * projection publishes an expert-specific FP32 row, making the expected
     * original-slot and router-weight mapping exact and easy to inspect.
     */
    class CanonicalRouteTestGemm final : public ITensorGemm
    {
    public:
        CanonicalRouteTestGemm(int expert_id, bool down_projection)
            : expert_id_(expert_id), down_projection_(down_projection)
        {
        }

        bool supports_device(int) const override { return true; }

        bool multiply_tensor(
            const TensorBase *, TensorBase *output,
            int m, int n, int,
            bool = true,
            float = 1.0f, float = 0.0f,
            const TensorBase * = nullptr,
            const IMPIContext * = nullptr,
            int = -1,
            DeviceWorkspaceManager * = nullptr,
            int = 0) override
        {
            if (!output || m <= 0 || n <= 0)
                return false;
            float *const values = output->mutable_data();
            if (!values)
                return false;
            std::fill_n(
                values,
                static_cast<size_t>(m) * static_cast<size_t>(n),
                1.0f);
            return true;
        }

        bool multiply_tensor_with_fused_swiglu(
            const TensorBase *, const TensorBase *, TensorBase *output,
            int m, int n, int,
            float = 1.0f, float = 0.0f,
            DeviceWorkspaceManager * = nullptr) override
        {
            if (!down_projection_ || !output || m <= 0 || n <= 0)
                return false;
            float *const values = output->mutable_data();
            if (!values)
                return false;
            for (int row = 0; row < m; ++row)
            {
                for (int col = 0; col < n; ++col)
                {
                    values[static_cast<size_t>(row) *
                               static_cast<size_t>(n) +
                           static_cast<size_t>(col)] =
                        static_cast<float>((expert_id_ + 1) * 16 + col);
                }
            }
            return true;
        }

        bool multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_context = nullptr,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            for (const auto &projection : projections)
            {
                if (!projection.kernel || !projection.output ||
                    !projection.kernel->multiply_tensor(
                        input,
                        projection.output,
                        m,
                        projection.n,
                        k,
                        true,
                        1.0f,
                        0.0f,
                        projection.bias,
                        mpi_context,
                        -1,
                        workspace))
                {
                    return false;
                }
            }
            return true;
        }

        bool weights_converted() const override { return true; }

    private:
        int expert_id_ = -1;
        bool down_projection_ = false;
    };

    /**
     * @brief Floating test GEMM that preserves its borrowed activation row.
     *
     * Gate and up roles copy the gathered input; the down role copies the gate
     * row through the fused SwiGLU/down interface. The intentionally simple
     * arithmetic isolates compact-to-physical row mapping from expert math.
     */
    class BorrowedHiddenEchoGemm final : public ITensorGemm
    {
    public:
        explicit BorrowedHiddenEchoGemm(bool down_projection)
            : down_projection_(down_projection)
        {
        }

        bool supports_device(int) const override { return true; }

        bool multiply_tensor(
            const TensorBase *input,
            TensorBase *output,
            int m,
            int n,
            int k,
            bool = true,
            float = 1.0f,
            float = 0.0f,
            const TensorBase * = nullptr,
            const IMPIContext * = nullptr,
            int = -1,
            DeviceWorkspaceManager * = nullptr,
            int = 0) override
        {
            if (down_projection_ || !input || !output || m <= 0 ||
                n <= 0 || k != n || !input->data() ||
                !output->mutable_data())
            {
                return false;
            }
            std::copy_n(
                input->data(),
                static_cast<size_t>(m) * static_cast<size_t>(n),
                output->mutable_data());
            return true;
        }

        bool multiply_tensor_with_fused_swiglu(
            const TensorBase *gate,
            const TensorBase *,
            TensorBase *output,
            int m,
            int n,
            int intermediate,
            float = 1.0f,
            float = 0.0f,
            DeviceWorkspaceManager * = nullptr) override
        {
            if (!down_projection_ || !gate || !output || m <= 0 ||
                n <= 0 || intermediate != n || !gate->data() ||
                !output->mutable_data())
            {
                return false;
            }
            std::copy_n(
                gate->data(),
                static_cast<size_t>(m) * static_cast<size_t>(n),
                output->mutable_data());
            return true;
        }

        bool multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m,
            int k,
            const IMPIContext *mpi_context = nullptr,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            for (const auto &projection : projections)
            {
                if (!projection.kernel || !projection.output ||
                    !projection.kernel->multiply_tensor(
                        input,
                        projection.output,
                        m,
                        projection.n,
                        k,
                        true,
                        1.0f,
                        0.0f,
                        projection.bias,
                        mpi_context,
                        -1,
                        workspace))
                {
                    return false;
                }
            }
            return true;
        }

        bool weights_converted() const override { return true; }

    private:
        bool down_projection_ = false; ///< Select the fused down role.
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

    /** @brief Exercise immutable preparation before fixed-slot publication. */
    MoEOverlayParticipantBankInstallStatus prepareAndInstall(
        MoEOverlayParticipantResidency &residency,
        const MoEOverlayParticipantResidencyBank &bank,
        std::string *error)
    {
        auto prepared = residency.prepareReadyBank(bank, error);
        if (!prepared)
            return MoEOverlayParticipantBankInstallStatus::Invalid;
        return residency.installReadyBank(std::move(*prepared), error);
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
     CPUCanonicalRouteArenaRejectsGPUAndMissingStorageContracts)
{
    constexpr int kDModel = 8;
    constexpr int kTopK = 2;
    EXPECT_THROW(
        ([&]
         {
             (void)MoELocalExpertSerialBufferArena{
                 MoELocalExpertSerialBufferArena::Config{
                     .device_id = DeviceId::cuda(0),
                     .row_capacity = 2,
                     .d_model = kDModel,
                     .routing_top_k = kTopK,
                     .cpu_canonical_route_storage =
                         MoELocalExpertSerialBufferArena::
                             CPUCanonicalRouteStoragePolicy::
                                 RetainSerialMaximum,
                 }};
         }()),
        std::invalid_argument);

    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(2, 4, kDModel, kTopK, DeviceId::cpu());
    auto input = workspace.localExpertInput(0, 0);
    auto output = workspace.localExpertOutput(0, 0);
    std::vector<std::int32_t> original_slots(4, 0);
    std::vector<std::int32_t> compact_slots(4, 0);
    std::vector<float> canonical_rows(4 * kDModel, 0.0f);

    auto params = makePreparedVectorParams(2);
    params.device_id = DeviceId::cpu();
    params.input_rows = &input;
    params.output_rows = &output;
    params.d_model = kDModel;
    params.top_k = kTopK;
    params.cpu_canonical_route_return =
        MoELocalExpertStage::CPUCanonicalRouteReturnBinding{
            .original_route_slots = original_slots.data(),
            .compact_route_slots = compact_slots.data(),
            .preweighted_route_contributions = canonical_rows.data(),
            .route_slot_capacity = original_slots.size(),
        };
    EXPECT_THROW(
        (void)MoELocalExpertStage{params},
        std::invalid_argument)
        << "A mapped route destination without its serial raw-route arena is an invalid lifecycle state";
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     CPUCanonicalReturnPublishesPreweightedOriginalRouteSlots)
{
    constexpr int kExperts = 2;
    constexpr int kRows = 2;
    constexpr int kDModel = 8;
    constexpr int kTopK = 2;
    constexpr size_t kMappedRouteCapacity = 6;

    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(
        kRows, kRows * kTopK, kDModel, kTopK, DeviceId::cpu());
    auto input = workspace.localExpertInput(0, 0);
    auto output = workspace.localExpertOutput(0, 0);
    input.residency_epoch = 1;
    input.source_participant = 0;
    input.target_participant = 3;
    input.live_row_count = kRows;
    input.live_entry_count = 3;
    input.row_ids_host[0] = 2;
    input.row_ids_host[1] = 5;
    input.entry_offsets_host[0] = 0;
    input.entry_offsets_host[1] = 2;
    input.entry_offsets_host[2] = 3;
    input.expert_ids_host[0] = 0;
    input.expert_ids_host[1] = 1;
    input.expert_ids_host[2] = 1;
    input.route_weights_host[0] = 0.25f;
    input.route_weights_host[1] = 0.75f;
    input.route_weights_host[2] = 0.5f;
    for (int row = 0; row < kRows; ++row)
    {
        for (int col = 0; col < kDModel; ++col)
        {
            input.hidden_rows_fp32[
                static_cast<size_t>(row) * kDModel + col] =
                static_cast<float>(row * kDModel + col + 1);
        }
    }

    std::vector<std::int32_t> original_slots(
        kMappedRouteCapacity, -1);
    std::vector<std::int32_t> compact_slots(
        kMappedRouteCapacity, -1);
    original_slots[0] = 1;
    original_slots[1] = 3;
    original_slots[2] = 4;
    compact_slots[0] = 0;
    compact_slots[1] = 1;
    compact_slots[2] = 2;
    constexpr float kUntouched = -777.0f;
    std::vector<float> canonical_rows(
        kMappedRouteCapacity * kDModel, kUntouched);

    auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
        MoELocalExpertSerialBufferArena::Config{
            .device_id = DeviceId::cpu(),
            .row_capacity = kRows,
            .row_capacity_buckets = {kRows},
            .d_model = kDModel,
            .routing_top_k = kTopK,
            .cpu_canonical_route_storage =
                MoELocalExpertSerialBufferArena::
                    CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
            .logical_participant_id = 3,
            .debug_name = "unit.cpu_canonical_route_return",
        });

    std::vector<std::unique_ptr<CanonicalRouteTestGemm>> engines;
    std::vector<ITensorGemm *> gate(kExperts, nullptr);
    std::vector<ITensorGemm *> up(kExperts, nullptr);
    std::vector<ITensorGemm *> down(kExperts, nullptr);
    for (int expert = 0; expert < kExperts; ++expert)
    {
        engines.push_back(std::make_unique<CanonicalRouteTestGemm>(
            expert, false));
        gate[expert] = engines.back().get();
        engines.push_back(std::make_unique<CanonicalRouteTestGemm>(
            expert, false));
        up[expert] = engines.back().get();
        engines.push_back(std::make_unique<CanonicalRouteTestGemm>(
            expert, true));
        down[expert] = engines.back().get();
    }

    MoELocalExpertStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input_rows = &input;
    params.output_rows = &output;
    params.serial_compact_buffer_arena = arena;
    params.graph_row_capacity = kRows;
    params.cpu_canonical_route_return =
        MoELocalExpertStage::CPUCanonicalRouteReturnBinding{
            .original_route_slots = original_slots.data(),
            .compact_route_slots = compact_slots.data(),
            .preweighted_route_contributions = canonical_rows.data(),
            .route_slot_capacity = kMappedRouteCapacity,
        };
    params.num_experts = kExperts;
    params.top_k = kTopK;
    params.d_model = kDModel;
    params.expert_intermediate = 16;
    params.layer_idx = 0;
    params.runtime_participant_index = 3;
    params.expert_mask.assign(kExperts, true);
    params.prepared_gate_gemm = gate;
    params.prepared_up_gemm = up;
    params.prepared_down_gemm = down;

    MoELocalExpertStage stage(params);
    llaminar2::testing::MockDeviceContext context(
        DeviceId::cpu(), ComputeBackendType::CPU);
    ASSERT_TRUE(stage.execute(&context));
    ASSERT_EQ(output.live_row_count, static_cast<size_t>(kRows));
    EXPECT_EQ(output.row_ids_host[0], input.row_ids_host[0]);
    EXPECT_EQ(output.row_ids_host[1], input.row_ids_host[1]);

    const auto expect_route = [&](size_t slot, int expert, float weight)
    {
        for (int col = 0; col < kDModel; ++col)
        {
            EXPECT_FLOAT_EQ(
                canonical_rows[slot * kDModel + col],
                weight * static_cast<float>((expert + 1) * 16 + col));
        }
    };
    expect_route(1, 0, 0.25f);
    expect_route(3, 1, 0.75f);
    expect_route(4, 1, 0.5f);
    for (const size_t untouched_slot : {size_t{0}, size_t{2}, size_t{5}})
    {
        for (int col = 0; col < kDModel; ++col)
        {
            EXPECT_FLOAT_EQ(
                canonical_rows[untouched_slot * kDModel + col],
                kUntouched);
        }
    }
}

/** @brief Host packets retain raw rows and original CSR identities after filtering. */
TEST(Test__MoELocalExpertStage_PreparedWeights,
     CPUCanonicalHostPacketPreservesRawRowsAndFilteredOriginalSlots)
{
    constexpr int width = 8;
    constexpr int experts = 4;
    MoEOverlayCollectiveWorkspace workspace({
        .max_rows = 1, .max_entries = 4, .d_model = width, .top_k = 4,
        .device = DeviceId::cpu(),
        .return_layout = MoEOverlayReturnLayout::CanonicalExpertRoutes,
    });
    auto input = workspace.localExpertInput(0, 0);
    auto output = workspace.localExpertOutput(0, 0);
    input.residency_epoch = 1;
    input.live_row_count = 1;
    input.live_entry_count = 4;
    input.row_ids_host[0] = 1;
    input.entry_offsets_host[0] = 0;
    input.entry_offsets_host[1] = 4;
    const float weights[4]{0.25f, 0.0f, 0.75f, 0.5f};
    for (int entry = 0; entry < experts; ++entry)
    {
        input.expert_ids_host[entry] = entry;
        input.original_route_slots_host[entry] = 4 + entry;
        input.compact_route_slots_host[entry] = entry;
        input.route_weights_host[entry] = weights[entry];
    }
    std::fill_n(input.hidden_rows_fp32, width, 1.0f);
    auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
        MoELocalExpertSerialBufferArena::Config{
            .device_id = DeviceId::cpu(), .row_capacity = 1,
            .d_model = width, .routing_top_k = 4,
            .cpu_canonical_route_storage = MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
        });
    std::vector<std::unique_ptr<CanonicalRouteTestGemm>> owned;
    MoELocalExpertStage::Params params;
    params.device_id = DeviceId::cpu();
    params.return_layout = MoEOverlayReturnLayout::CanonicalExpertRoutes;
    params.input_rows = &input;
    params.output_rows = &output;
    params.serial_compact_buffer_arena = arena;
    params.graph_row_capacity = 1;
    params.num_experts = experts;
    params.top_k = 4;
    params.d_model = width;
    params.expert_intermediate = 16;
    params.layer_idx = 0;
    params.expert_mask = {true, true, false, true};
    for (int expert = 0; expert < experts; ++expert)
    {
        owned.push_back(std::make_unique<CanonicalRouteTestGemm>(expert, false));
        params.prepared_gate_gemm.push_back(owned.back().get());
        owned.push_back(std::make_unique<CanonicalRouteTestGemm>(expert, false));
        params.prepared_up_gemm.push_back(owned.back().get());
        owned.push_back(std::make_unique<CanonicalRouteTestGemm>(expert, true));
        params.prepared_down_gemm.push_back(owned.back().get());
    }
    MoELocalExpertStage stage(params);
    llaminar2::testing::MockDeviceContext context(DeviceId::cpu(), ComputeBackendType::CPU);
    ASSERT_TRUE(stage.execute(&context));
    ASSERT_EQ(output.layout, MoEOverlayReturnLayout::CanonicalExpertRoutes);
    ASSERT_EQ(output.live_row_count, 2u);
    EXPECT_EQ(output.row_ids_host[0], 4);
    EXPECT_EQ(output.row_ids_host[1], 7);
    for (int col = 0; col < width; ++col)
    {
        EXPECT_EQ(output.output_rows_fp32[col], 16.0f + col);
        EXPECT_EQ(output.output_rows_fp32[width + col], 64.0f + col);
    }
}

/**
 * @brief Stable single-pass admission preserves row and route order after filtering.
 *
 * Each source row deliberately mixes resident, non-resident, and zero-weight
 * routes.  The compact packet must omit empty row 11, retain the other logical
 * row ids in source order, and accumulate each surviving route exactly once.
 */
TEST(Test__MoELocalExpertStage_PreparedWeights,
     CPUCompactAdmissionPreservesFilteredRouteOrder)
{
    constexpr int kExperts = 4;
    constexpr int kRows = 4;
    constexpr int kDModel = 8;
    constexpr int kTopK = 3;

    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(
        kRows, kRows * kTopK, kDModel, kTopK, DeviceId::cpu());
    auto input = workspace.localExpertInput(0, 0);
    auto output = workspace.localExpertOutput(0, 0);
    input.residency_epoch = 1;
    input.live_row_count = kRows;
    input.live_entry_count = 10;
    const std::int32_t row_ids[kRows] = {10, 11, 12, 13};
    const std::int32_t offsets[kRows + 1] = {0, 3, 5, 8, 10};
    std::copy_n(row_ids, kRows, input.row_ids_host);
    std::copy_n(offsets, kRows + 1, input.entry_offsets_host);
    const std::int32_t experts[10] = {1, 0, 2, 1, 3, 2, 0, 1, 3, 0};
    const float weights[10] = {
        1.0f, 0.25f, 0.5f,
        0.5f, 0.5f,
        0.75f, 0.0f, 1.0f,
        0.25f, 1.0f};
    std::copy_n(experts, 10, input.expert_ids_host);
    std::copy_n(weights, 10, input.route_weights_host);
    for (int row = 0; row < kRows; ++row)
    {
        for (int col = 0; col < kDModel; ++col)
        {
            input.hidden_rows_fp32[
                static_cast<size_t>(row) * kDModel + col] =
                static_cast<float>(row * kDModel + col + 1);
        }
    }

    auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
        MoELocalExpertSerialBufferArena::Config{
            .device_id = DeviceId::cpu(),
            .row_capacity = kRows,
            .row_capacity_buckets = {1, 2, kRows},
            .d_model = kDModel,
            .routing_top_k = kTopK,
            .logical_participant_id = 2,
            .debug_name = "unit.cpu_single_pass_compaction",
        });

    std::vector<std::unique_ptr<CanonicalRouteTestGemm>> engines;
    std::vector<ITensorGemm *> gate(kExperts, nullptr);
    std::vector<ITensorGemm *> up(kExperts, nullptr);
    std::vector<ITensorGemm *> down(kExperts, nullptr);
    for (int expert = 0; expert < kExperts; ++expert)
    {
        engines.push_back(std::make_unique<CanonicalRouteTestGemm>(
            expert, false));
        gate[expert] = engines.back().get();
        engines.push_back(std::make_unique<CanonicalRouteTestGemm>(
            expert, false));
        up[expert] = engines.back().get();
        engines.push_back(std::make_unique<CanonicalRouteTestGemm>(
            expert, true));
        down[expert] = engines.back().get();
    }

    MoELocalExpertStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input_rows = &input;
    params.output_rows = &output;
    params.serial_compact_buffer_arena = arena;
    params.graph_row_capacity = kRows;
    params.num_experts = kExperts;
    params.top_k = kTopK;
    params.d_model = kDModel;
    params.expert_intermediate = 16;
    params.layer_idx = 0;
    params.runtime_participant_index = 2;
    params.expert_mask = {true, false, true, false};
    params.prepared_gate_gemm = gate;
    params.prepared_up_gemm = up;
    params.prepared_down_gemm = down;

    MoELocalExpertStage stage(params);
    llaminar2::testing::MockDeviceContext context(
        DeviceId::cpu(), ComputeBackendType::CPU);
    ASSERT_TRUE(stage.execute(&context));
    ASSERT_EQ(output.live_row_count, 3u);
    EXPECT_EQ(output.row_ids_host[0], 10);
    EXPECT_EQ(output.row_ids_host[1], 12);
    EXPECT_EQ(output.row_ids_host[2], 13);

    const auto expect_row = [&](size_t row, float expert_zero_weight,
                                float expert_two_weight)
    {
        for (int col = 0; col < kDModel; ++col)
        {
            const float expected =
                expert_zero_weight * static_cast<float>(16 + col) +
                expert_two_weight * static_cast<float>(48 + col);
            EXPECT_FLOAT_EQ(
                output.output_rows_fp32[row * kDModel + col], expected);
        }
    };
    expect_row(0, 0.25f, 0.5f);
    expect_row(1, 0.0f, 0.75f);
    expect_row(2, 1.0f, 0.0f);

    /*
     * Reuse the same stage and durable residency epoch with a narrower live
     * packet. This proves the retained CPU executor changes only its live
     * dimensions: tensor identity and prepared engines remain setup-owned,
     * while stale rows and the previous two-route stride cannot leak forward.
     */
    input.live_row_count = 2;
    input.live_entry_count = 3;
    input.row_ids_host[0] = 20;
    input.row_ids_host[1] = 21;
    input.entry_offsets_host[0] = 0;
    input.entry_offsets_host[1] = 1;
    input.entry_offsets_host[2] = 3;
    input.expert_ids_host[0] = 2;
    input.expert_ids_host[1] = 1;
    input.expert_ids_host[2] = 0;
    input.route_weights_host[0] = 0.5f;
    input.route_weights_host[1] = 1.0f;
    input.route_weights_host[2] = 0.75f;
    std::fill_n(
        output.output_rows_fp32,
        static_cast<size_t>(kRows) * kDModel,
        -99.0f);

    ASSERT_TRUE(stage.execute(&context));
    ASSERT_EQ(output.live_row_count, 2u);
    EXPECT_EQ(output.row_ids_host[0], 20);
    EXPECT_EQ(output.row_ids_host[1], 21);
    expect_row(0, 0.0f, 0.5f);
    expect_row(1, 0.75f, 0.0f);
}

/**
 * @brief A CPU endpoint consumes shared physical transport rows without a copy.
 *
 * Row ids intentionally reverse and sparsify the physical activation order.
 * The floating expert path must gather those authenticated rows directly and
 * return them in compact packet order; reading the stale compact tensor would
 * make this regression fail deterministically.
 */
TEST(Test__MoELocalExpertStage_PreparedWeights,
     CPUTransportedHiddenBindingPreservesSharedPhysicalRowIdentity)
{
    constexpr int kRows = 2;
    constexpr int kPhysicalRows = 5;
    constexpr int kDModel = 8;
    constexpr int kExperts = 1;
    constexpr int kTopK = 1;

    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(
        kRows,
        kRows * kTopK,
        kDModel,
        kTopK,
        DeviceId::cpu());
    auto input = workspace.localExpertInput(0, 0);
    auto output = workspace.localExpertOutput(0, 0);
    std::vector<float> physical_hidden(
        static_cast<size_t>(kPhysicalRows) * kDModel);
    for (int row = 0; row < kPhysicalRows; ++row)
    {
        for (int column = 0; column < kDModel; ++column)
        {
            physical_hidden[
                static_cast<size_t>(row) * kDModel + column] =
                static_cast<float>(row * 100 + column);
        }
    }

    input.residency_epoch = 1;
    input.live_row_count = kRows;
    input.live_entry_count = kRows;
    input.row_ids_host[0] = 4;
    input.row_ids_host[1] = 1;
    input.entry_offsets_host[0] = 0;
    input.entry_offsets_host[1] = 1;
    input.entry_offsets_host[2] = 2;
    input.expert_ids_host[0] = 0;
    input.expert_ids_host[1] = 0;
    input.route_weights_host[0] = 1.0f;
    input.route_weights_host[1] = 1.0f;
    input.hidden_rows_fp32 = physical_hidden.data();
    input.hidden_row_capacity = kPhysicalRows;
    input.hidden_payload_layout =
        MoEOverlayActivationHiddenPayloadLayout::SharedPhysicalRows;

    auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
        MoELocalExpertSerialBufferArena::Config{
            .device_id = DeviceId::cpu(),
            .row_capacity = kRows,
            .d_model = kDModel,
            .routing_top_k = kTopK,
            .logical_participant_id = 2,
            .debug_name = "unit.cpu_borrowed_physical_hidden",
        });
    BorrowedHiddenEchoGemm gate(/*down_projection=*/false);
    BorrowedHiddenEchoGemm up(/*down_projection=*/false);
    BorrowedHiddenEchoGemm down(/*down_projection=*/true);

    MoELocalExpertStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input_rows = &input;
    params.output_rows = &output;
    params.serial_compact_buffer_arena = arena;
    params.graph_row_capacity = kRows;
    params.num_experts = kExperts;
    params.top_k = kTopK;
    params.d_model = kDModel;
    params.expert_intermediate = kDModel;
    params.layer_idx = 0;
    params.runtime_participant_index = 2;
    params.expert_mask = {true};
    params.prepared_gate_gemm = {&gate};
    params.prepared_up_gemm = {&up};
    params.prepared_down_gemm = {&down};

    MoELocalExpertStage stage(std::move(params));
    llaminar2::testing::MockDeviceContext context(
        DeviceId::cpu(), ComputeBackendType::CPU);
    ASSERT_TRUE(stage.execute(&context));
    ASSERT_EQ(output.live_row_count, static_cast<size_t>(kRows));
    for (int compact_row = 0; compact_row < kRows; ++compact_row)
    {
        const int physical_row = input.row_ids_host[compact_row];
        EXPECT_EQ(output.row_ids_host[compact_row], physical_row);
        for (int column = 0; column < kDModel; ++column)
        {
            EXPECT_FLOAT_EQ(
                output.output_rows_fp32[
                    static_cast<size_t>(compact_row) * kDModel + column],
                physical_hidden[
                    static_cast<size_t>(physical_row) * kDModel + column]);
        }
    }

    /*
     * Rebind the retained stage to a serial decode packet. M=1 deliberately
     * uses the specialized serial FFN contract, while M>1 uses the grouped
     * executor above; both contracts must preserve the same authenticated
     * physical-row identity at the transport boundary.
     */
    input.live_row_count = 1;
    input.live_entry_count = 1;
    input.row_ids_host[0] = 3;
    input.entry_offsets_host[0] = 0;
    input.entry_offsets_host[1] = 1;
    input.expert_ids_host[0] = 0;
    input.route_weights_host[0] = 1.0f;
    std::fill_n(
        output.output_rows_fp32,
        static_cast<size_t>(kRows) * kDModel,
        -99.0f);

    ASSERT_TRUE(stage.execute(&context));
    ASSERT_EQ(output.live_row_count, 1u);
    EXPECT_EQ(output.row_ids_host[0], 3);
    for (int column = 0; column < kDModel; ++column)
    {
        EXPECT_FLOAT_EQ(
            output.output_rows_fp32[column],
            physical_hidden[static_cast<size_t>(3) * kDModel + column]);
    }
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
     RegistryOnlyPreparationRejectsRawParentFallback)
{
    auto params = makePreparedVectorParams(2);
    params.expert_mask = {true, true};
    params.prepared_gate_gemm[1] = nullptr;
    params.expert_weight_resolution_policy =
        MoELocalExpertStage::ExpertWeightResolutionPolicy::RegistryOnly;

    /*
     * An ExpertOverlay graph must fail during construction when its exact
     * registry slice is incomplete. It may not recover by reading a raw 3-D
     * parent that could belong to a different tier.
     */
    EXPECT_FALSE(MoELocalExpertStage::prepareExpertGemmEngines(params));
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     CudaWorkspaceReservesFullLogicalExpertDescriptorTables)
{
    constexpr int kLogicalExperts = 256;
    MoELocalExpertStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.num_experts = kLogicalExperts;
    params.top_k = 2;
    params.d_model = 256;
    params.expert_intermediate = 512;
    params.layer_idx = 0;

    MoELocalExpertStage stage(params);
    const WorkspaceRequirements requirements =
        stage.getWorkspaceRequirements(/*m=*/9);
    const auto descriptor_bytes = [](const WorkspaceRequirements &candidate,
                                     const char *name) -> size_t
    {
        const auto it = std::find_if(
            candidate.buffers.begin(), candidate.buffers.end(),
            [name](const WorkspaceDescriptor &buffer)
            {
                return buffer.name == name;
            });
        return it == candidate.buffers.end() ? 0u : it->size_bytes;
    };
    const size_t expected_bytes =
        static_cast<size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots) *
        static_cast<size_t>(kLogicalExperts) *
        sizeof(DeviceNativeVNNIMatrixDesc);

    EXPECT_EQ(
        descriptor_bytes(
            requirements,
            MoEWorkspaceBuffers::CUDA_GROUPED_GATE_DESC_TABLES),
        expected_bytes);
    EXPECT_EQ(
        descriptor_bytes(
            requirements,
            MoEWorkspaceBuffers::CUDA_GROUPED_UP_DESC_TABLES),
        expected_bytes);
    EXPECT_EQ(
        descriptor_bytes(
            requirements,
            MoEWorkspaceBuffers::CUDA_GROUPED_DOWN_DESC_TABLES),
        expected_bytes);
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
    input.residency_epoch = 1;
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
    input.residency_epoch = 1;
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

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ResidencyPublicationFlipsOneCompletePreparedMaskAtANewerEpoch)
{
    constexpr int kNumExperts = 4;
    MoERuntimeTable table(DeviceId::cpu(), 1, kNumExperts, 2);

    MoELocalExpertStage::Params params;
    params.device_id = DeviceId::cpu();
    params.num_experts = kNumExperts;
    params.top_k = 2;
    params.d_model = 32;
    params.expert_intermediate = 64;
    params.layer_idx = 0;
    params.expert_mask = {true, false, true, false};
    params.runtime_participant_index = 1;
    params.moe_runtime_table = &table;

    std::vector<std::unique_ptr<FakePreparedGemm>> owned;
    attachFakePreparedEngines(params, owned);

    MoELocalExpertStage stage(params);
    ASSERT_TRUE(stage.refreshRuntimePlacement());
    ASSERT_EQ(table.hostLayerState(0).active_epoch, 1u);

    const std::vector<bool> migrated_mask{false, true, false, true};
    EXPECT_TRUE(stage.missingPreparedExpertIds({1, 3}).empty());
    ASSERT_NO_THROW(stage.applyExpertMask(migrated_mask));

    const auto &state = table.hostLayerState(0);
    ASSERT_EQ(state.active_epoch, 2u);
    ASSERT_LE(state.active_bank, 1u);
    const auto &bank = state.banks[state.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 0u);
    EXPECT_EQ(bank.local_compute_mask[1], 1u);
    EXPECT_EQ(bank.local_compute_mask[2], 0u);
    EXPECT_EQ(bank.local_compute_mask[3], 1u);
    EXPECT_EQ(bank.experts[1].owner_participant, 1);
    EXPECT_EQ(stage.expertMask(), migrated_mask);
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ResidencyPublicationRejectsMissingArrivalWithoutChangingLiveEpoch)
{
    constexpr int kNumExperts = 4;
    MoERuntimeTable table(DeviceId::cpu(), 1, kNumExperts, 2);

    MoELocalExpertStage::Params params;
    params.device_id = DeviceId::cpu();
    params.num_experts = kNumExperts;
    params.top_k = 2;
    params.d_model = 32;
    params.expert_intermediate = 64;
    params.layer_idx = 0;
    params.expert_mask = {true, false, true, false};
    params.moe_runtime_table = &table;

    std::vector<std::unique_ptr<FakePreparedGemm>> owned;
    attachFakePreparedEngines(params, owned);
    params.prepared_gate_gemm[1] = nullptr;
    params.prepared_up_gemm[1] = nullptr;
    params.prepared_down_gemm[1] = nullptr;

    MoELocalExpertStage stage(params);
    ASSERT_TRUE(stage.refreshRuntimePlacement());
    ASSERT_EQ(stage.missingPreparedExpertIds({1}), (std::vector<int>{1}));

    EXPECT_THROW(
        stage.applyExpertMask({false, true, true, false}),
        std::runtime_error);
    EXPECT_EQ(table.hostLayerState(0).active_epoch, 1u);
    EXPECT_EQ(stage.expertMask(),
              (std::vector<bool>{true, false, true, false}));
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     SparseExecutionAcquiresExactParticipantResidencyEpoch)
{
    constexpr int kNumExperts = 2;
    constexpr int kDModel = 8;
    constexpr int kTopK = 1;

    auto residency = std::make_shared<MoEOverlayParticipantResidency>(
        MoEOverlayParticipantResidency::Config{
            .participant_id = 3,
            .device = DeviceId::cpu(),
            .num_layers = 1,
            .num_experts = kNumExperts,
            .retained_epoch_capacity = 2,
        });
    MoEOverlayParticipantResidencyBank epoch_one;
    epoch_one.epoch = 1;
    epoch_one.participant_id = 3;
    epoch_one.device = DeviceId::cpu();
    epoch_one.layers.resize(1);
    epoch_one.layers[0].resident_mask.assign(kNumExperts, false);
    epoch_one.layers[0].experts.resize(kNumExperts);
    epoch_one.layers[0].setResidentExpert(
        0,
        {
            .gate = std::make_shared<FakePreparedGemm>(
                nativeDesc(0, 0, 32, kDModel)),
            .up = std::make_shared<FakePreparedGemm>(
                nativeDesc(0, 1, 32, kDModel)),
            .down = std::make_shared<FakePreparedGemm>(
                nativeDesc(0, 2, kDModel, 32)),
        });
    std::string error;
    ASSERT_EQ(
        prepareAndInstall(*residency, epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;
    const auto epoch_two = residency->cloneCandidate(1, 2);
    ASSERT_EQ(
        prepareAndInstall(*residency, epoch_two, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;

    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(1, 1, kDModel, kTopK, DeviceId::cpu());
    auto input = workspace.localExpertInput(0, 0);
    input.residency_epoch = 1;
    auto output = workspace.localExpertOutput(0, 0);

    MoELocalExpertStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input_rows = &input;
    params.output_rows = &output;
    params.num_experts = kNumExperts;
    params.top_k = kTopK;
    params.d_model = kDModel;
    params.expert_intermediate = 32;
    params.layer_idx = 0;
    params.runtime_participant_index = 3;
    params.overlay_participant_residency = residency;
    MoELocalExpertStage stage(params);
    llaminar2::testing::MockDeviceContext ctx(
        DeviceId::cpu(), ComputeBackendType::CPU);

    /* Epoch one remains executable even after epoch two is ready. */
    EXPECT_TRUE(stage.execute(&ctx));
    EXPECT_EQ(output.residency_epoch, 1u);

    /* Retirement makes a stale packet fail instead of using epoch two. */
    ASSERT_TRUE(residency->retire(1));
    EXPECT_FALSE(stage.execute(&ctx));

    input.residency_epoch = 2;
    EXPECT_TRUE(stage.execute(&ctx));
    EXPECT_EQ(output.residency_epoch, 2u);
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     ParticipantResidencyRejectsCompetingMutableRuntimeTable)
{
    constexpr int kNumExperts = 2;
    auto residency = std::make_shared<MoEOverlayParticipantResidency>(
        MoEOverlayParticipantResidency::Config{
            .participant_id = 0,
            .device = DeviceId::cpu(),
            .num_layers = 1,
            .num_experts = kNumExperts,
        });
    MoEOverlayParticipantResidencyBank bank;
    bank.epoch = 1;
    bank.participant_id = 0;
    bank.device = DeviceId::cpu();
    bank.layers.resize(1);
    bank.layers[0].resident_mask.assign(kNumExperts, false);
    bank.layers[0].experts.resize(kNumExperts);
    std::string error;
    ASSERT_EQ(
        prepareAndInstall(*residency, bank, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, kNumExperts, 1);
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(1, 1, 8, 1, DeviceId::cpu());
    auto input = workspace.localExpertInput(0, 0);
    input.residency_epoch = 1;
    auto output = workspace.localExpertOutput(0, 0);

    MoELocalExpertStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input_rows = &input;
    params.output_rows = &output;
    params.num_experts = kNumExperts;
    params.top_k = 1;
    params.d_model = 8;
    params.expert_intermediate = 16;
    params.layer_idx = 0;
    params.runtime_participant_index = 0;
    params.overlay_participant_residency = residency;
    params.moe_runtime_table = &runtime_table;

    MoELocalExpertStage stage(params);
    llaminar2::testing::MockDeviceContext ctx(
        DeviceId::cpu(), ComputeBackendType::CPU);
    EXPECT_FALSE(stage.execute(&ctx));
}

TEST(Test__MoELocalExpertStage_PreparedWeights,
     DeferredCompletionIsADeviceFreeTypedGraphOwnershipContract)
{
    constexpr int kNumExperts = 2;
    constexpr int kDModel = 8;
    constexpr int kTopK = 2;
    const DeviceId device = DeviceId::cuda(0);

    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(
        /*max_rows=*/1,
        /*max_entries=*/kTopK,
        kDModel,
        kTopK,
        device);
    auto input = workspace.localExpertInput(0, 0);
    auto output = workspace.localExpertOutput(0, 0);
    auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
        MoELocalExpertSerialBufferArena::Config{
            .device_id = device,
            .row_capacity = 1,
            .row_capacity_buckets = {1},
            .d_model = kDModel,
            .routing_top_k = kTopK,
            .logical_participant_id = 7,
            .debug_name = "unit.deferred_local_expert"});
    auto residency = std::make_shared<MoEOverlayParticipantResidency>(
        MoEOverlayParticipantResidency::Config{
            .participant_id = 7,
            .device = device,
            .num_layers = 1,
            .num_experts = kNumExperts,
        });
    DeviceMoELayerRuntime telemetry_runtime{};
    DeviceMoEOverlayServiceTelemetryCell telemetry_cells
        [kDeviceMoEOverlayServicePhaseCount]{};
    DeviceMoEOverlayServiceTelemetrySample telemetry_sample{};

    MoELocalExpertStage::Params params;
    params.device_id = device;
    params.input_rows = &input;
    params.output_rows = &output;
    params.serial_compact_buffer_arena = arena;
    params.graph_row_capacity = 1;
    params.completion_policy =
        MoELocalExpertStage::CompletionPolicy::DeferredExplicitStage;
    params.num_experts = kNumExperts;
    params.top_k = kTopK;
    params.d_model = kDModel;
    params.expert_intermediate = 16;
    params.layer_idx = 0;
    params.runtime_participant_index = 7;
    params.expert_mask.assign(kNumExperts, true);
    params.overlay_participant_residency = residency;
    params.overlay_service_telemetry = {
        .runtime_layer = &telemetry_runtime,
        .layer_telemetry = telemetry_cells,
        .sample = &telemetry_sample,
    };
    std::vector<std::unique_ptr<FakePreparedGemm>> owned;
    attachFakePreparedEngines(params, owned);

    MoELocalExpertStage producer(params);
    EXPECT_TRUE(producer.usesDeferredCompletion());
    EXPECT_FALSE(producer.hasPendingDeferredOutput());
    EXPECT_TRUE(
        producer.allDeferredReplayFamiliesUseServiceTelemetryForTesting())
        << "Every retained sparse shape/route-width family must preserve the "
           "canonical observation capability without borrowing placement authority.";

    MoELocalExpertCompletionStage completion(
        MoELocalExpertCompletionStage::Params{
            .device_id = device,
            .producer = &producer,
        });
    EXPECT_EQ(
        completion.type(),
        ComputeStageType::MOE_LOCAL_EXPERT_COMPLETION);
    EXPECT_TRUE(completion.supportsBackend(
        ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(completion.supportsBackend(
        ComputeBackendType::GPU_ROCM));
    EXPECT_FALSE(completion.supportsBackend(
        ComputeBackendType::CPU));

    EXPECT_THROW(
        ([&]
         {
             (void)MoELocalExpertCompletionStage{
                 MoELocalExpertCompletionStage::Params{
                     .device_id = device,
                     .producer = nullptr,
                 }};
         }()),
        std::invalid_argument);

    auto cpu_params = params;
    cpu_params.device_id = DeviceId::cpu();
    cpu_params.serial_compact_buffer_arena.reset();
    EXPECT_THROW(
        (void)MoELocalExpertStage{cpu_params},
        std::invalid_argument)
        << "Only GPU endpoints may defer their host-visible completion";
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
