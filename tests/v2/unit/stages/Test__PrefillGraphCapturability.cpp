/**
 * @file Test__PrefillGraphCapturability.cpp
 * @brief Acceptance gates for grouped prefill and routed decode GPU capture.
 *
 * Tests that isGraphCapturable() returns true for prefill (seq_len > 1) on GPU
 * when all readiness conditions are met, and false when any condition is violated.
 *
 * Covers:
 * - MoERoutingStage
 * - MoEExpertComputeStage (fixed-topology grouped prefill)
 * - SharedExpertFFNStage
 * - SharedExpertGateStage
 * - GDNRecurrenceStage
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/compute_stages/stages/GDNLiveStateAllGatherStage.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "kernels/IMoEKernel.h"
#include "mocks/MockComputeStage.h"
#include "../../mocks/MockBackend.h"
#include "mocks/MockLocalTPContext.h"
#include "utils/TestTensorFactory.h"
#include "utils/DebugEnv.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar2::testing;

namespace
{

    // =========================================================================
    // Scoped DebugEnv flag helpers
    // =========================================================================

    class ScopedMoEGraphCaptureFlags
    {
    public:
        ScopedMoEGraphCaptureFlags(bool grouped_decode, bool device_routed_decode)
            : old_grouped_(mutableDebugEnv().rocm.moe_grouped_decode),
              old_device_routed_(mutableDebugEnv().rocm.moe_device_routed_decode)
        {
            mutableDebugEnv().rocm.moe_grouped_decode = grouped_decode;
            mutableDebugEnv().rocm.moe_device_routed_decode = device_routed_decode;
        }

        ~ScopedMoEGraphCaptureFlags()
        {
            mutableDebugEnv().rocm.moe_grouped_decode = old_grouped_;
            mutableDebugEnv().rocm.moe_device_routed_decode = old_device_routed_;
        }

    private:
        bool old_grouped_;
        bool old_device_routed_;
    };

    DeviceNativeVNNIMatrixDesc runtimeDesc(uintptr_t base, int n, int k)
    {
        DeviceNativeVNNIMatrixDesc desc;
        desc.payload = reinterpret_cast<const uint8_t *>(base);
        desc.scales = reinterpret_cast<const void *>(base + 0x1000u);
        desc.mins = reinterpret_cast<const void *>(base + 0x2000u);
        desc.n = n;
        desc.k = k;
        desc.blocks_per_row = static_cast<uint32_t>(k / 32);
        desc.codebook_id = 4;
        return desc;
    }

    MoEPlacementUpdate routingRuntimeUpdate(uint32_t epoch, int num_experts, int d_model)
    {
        MoEPlacementUpdate update;
        update.epoch = epoch;
        update.expert_count = static_cast<uint32_t>(num_experts);
        update.experts.resize(static_cast<size_t>(num_experts));
        update.local_compute_mask.assign(static_cast<size_t>(num_experts), 1u);
        update.replica_role.assign(static_cast<size_t>(num_experts),
                                   static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));

        for (int expert = 0; expert < num_experts; ++expert)
        {
            const uintptr_t base = 0x70000000u + static_cast<uintptr_t>(expert) * 0x10000u;
            auto &desc = update.experts[static_cast<size_t>(expert)];
            desc.gate = runtimeDesc(base + 0x0100u, d_model, d_model);
            desc.up = runtimeDesc(base + 0x0200u, d_model, d_model);
            desc.down = runtimeDesc(base + 0x0300u, d_model, d_model);
            desc.logical_expert_id = expert;
            desc.owner_participant = 0;
            desc.local_slot = expert;
            desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                          DeviceMoEExpertFlags::Resident |
                                          DeviceMoEExpertFlags::LocalCompute);
        }

        return update;
    }

    MoEPlacementUpdate apportionedRuntimeUpdate(
        uint32_t epoch,
        int num_experts,
        int d_model,
        int local_participant,
        int participant_count,
        const std::vector<int> &owners,
        const std::vector<bool> &local_mask)
    {
        MoEPlacementUpdate update;
        update.epoch = epoch;
        update.expert_count = static_cast<uint32_t>(num_experts);
        update.participant_id = static_cast<uint32_t>(local_participant);
        update.participant_count = static_cast<uint32_t>(participant_count);
        update.experts.resize(static_cast<size_t>(num_experts));
        update.local_compute_mask.assign(static_cast<size_t>(num_experts), 0u);
        update.replica_role.assign(static_cast<size_t>(num_experts),
                                   static_cast<uint8_t>(DeviceMoEReplicaRole::None));
        update.resident_participant_mask.assign(static_cast<size_t>(num_experts), 0u);

        for (int expert = 0; expert < num_experts; ++expert)
        {
            const int owner = owners.at(static_cast<size_t>(expert));
            const bool local = local_mask.at(static_cast<size_t>(expert));
            auto &desc = update.experts[static_cast<size_t>(expert)];
            desc.logical_expert_id = expert;
            desc.owner_participant = owner;
            desc.local_slot = local ? expert : -1;
            if (owner >= 0 && owner < participant_count)
                update.resident_participant_mask[static_cast<size_t>(expert)] =
                    1u << static_cast<uint32_t>(owner);

            if (!local)
                continue;

            const uintptr_t base = 0x71000000u + static_cast<uintptr_t>(expert) * 0x10000u;
            desc.gate = runtimeDesc(base + 0x0100u, d_model, d_model);
            desc.up = runtimeDesc(base + 0x0200u, d_model, d_model);
            desc.down = runtimeDesc(base + 0x0300u, d_model, d_model);
            desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                          DeviceMoEExpertFlags::Resident |
                                          DeviceMoEExpertFlags::LocalCompute);
            update.local_compute_mask[static_cast<size_t>(expert)] = 1u;
            update.replica_role[static_cast<size_t>(expert)] =
                owner == local_participant
                    ? static_cast<uint8_t>(DeviceMoEReplicaRole::Primary)
                    : static_cast<uint8_t>(DeviceMoEReplicaRole::Replica);
        }

        return update;
    }

    // =========================================================================
    // Minimal stub IMoEKernel (no actual compute, just satisfies the interface)
    // =========================================================================

    class DeviceResidentFP32Tensor final : public FP32Tensor
    {
    public:
        explicit DeviceResidentFP32Tensor(std::vector<size_t> shape)
            : FP32Tensor(std::move(shape))
        {
        }

        ~DeviceResidentFP32Tensor() override
        {
            // The pointer below is a predicate-test sentinel, not backend-owned
            // memory. Clear it before TensorBase destruction can try to free it.
            gpu_data_ptr_ = nullptr;
            gpu_device_.reset();
            setCoherenceState_(TensorCoherenceState::HOST_ONLY);
        }

        /**
         * @brief Mark the tensor as already resident on a device without
         * allocating GPU memory.
         *
         * SharedExpertGateStage::isGraphCapturable() only needs to verify the
         * stage's readiness predicate. The unit must not perform real H2D work
         * or use a default stream, so this helper supplies a non-null sentinel
         * pointer and matching coherence state.
         */
        void markResidentForGraphCaptureTest(DeviceId device)
        {
            gpu_data_ptr_ = reinterpret_cast<void *>(uintptr_t{0x1000});
            gpu_device_ = device;
            setCoherenceState_(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        }
    };

    class StubMoEKernel : public IMoEKernel
    {
    public:
        mutable int gateup_table_uploads = 0;
        mutable int down_table_uploads = 0;
        mutable int route_launch_preparations = 0;
        mutable int verifier_route_calls = 0;
        mutable int verifier_histogram_commit_calls = 0;
        mutable int runtime_decode_preparations = 0;
        mutable int fused_runtime_decode_calls = 0;
        mutable int expert_mask_publications = 0;
        mutable ITensor *last_prepared_route_gate = nullptr;
        mutable MoERouteLaunchPlan last_route_launch_plan{};
        mutable MoEDecodeDescriptorSource last_fused_descriptor_source =
            MoEDecodeDescriptorSource::RuntimePlacementTable;
        mutable DeviceMoELayerRuntime *last_deferred_selected_route_ledger =
            nullptr;
        mutable DeviceMoELayerRuntime *last_committed_route_ledger = nullptr;
        mutable void *last_histogram_commit_stream = nullptr;

        bool supports_device(int) const override { return true; }
        bool routeWithTensors(
            ITensor *,
            ITensor *,
            int,
            int,
            int,
            int,
            bool,
            ITensor *,
            ITensor *,
            MoERoutingResult &) override
        {
            return false;
        }
        bool prepareRouteLaunch(
            ITensor *gate_weights,
            const MoERouteLaunchPlan &plan) override
        {
            ++route_launch_preparations;
            last_prepared_route_gate = gate_weights;
            last_route_launch_plan = plan;
            return true;
        }
        bool routeVerifierRowsDecodeEquivalent(
            ITensor *,
            ITensor *,
            int,
            int,
            int,
            int,
            bool,
            ITensor *,
            ITensor *,
            const int *,
            DeviceMoELayerRuntime *deferred_selected_route_ledger) override
        {
            ++verifier_route_calls;
            last_deferred_selected_route_ledger =
                deferred_selected_route_ledger;
            return true;
        }
        bool commitGroupedVerifierHistograms(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            const int32_t *,
            const int32_t *,
            int,
            int,
            int,
            int,
            int) override
        {
            ++verifier_histogram_commit_calls;
            last_committed_route_ledger = runtime_layer;
            last_histogram_commit_stream = launch.stream;
            return launch.hasExplicitStream() && runtime_layer != nullptr;
        }
        void gatherTokenBatch(const float *, float *, const int *, int, int) override {}
        void scatterAddWeighted(float *, const float *, const int *, const float *,
                                int, int) override {}
        void sharedExpertGate(const float *, const float *, float *, int, int) override {}
        void swiGLU(float *, const float *, int) override {}
        int uploadGroupedExpertGateUpDescriptorTables(
            const DeviceNativeVNNIMatrixDesc *,
            const DeviceNativeVNNIMatrixDesc *,
            int,
            int,
            int,
            MoEDecodeDescriptorSource =
                MoEDecodeDescriptorSource::StaticDescriptorTable) override
        {
            return gateup_table_uploads++;
        }
        int uploadGroupedExpertDownDescriptorTable(
            const DeviceNativeVNNIMatrixDesc *,
            int,
            int,
            int,
            MoEDecodeDescriptorSource =
                MoEDecodeDescriptorSource::StaticDescriptorTable) override
        {
            return down_table_uploads++;
        }
        bool prepareGroupedRuntimeDecodeLaunchState(
            int,
            int,
            int,
            int,
            int,
            MoEDecodeDescriptorSource descriptor_source) override
        {
            ++runtime_decode_preparations;
            last_fused_descriptor_source = descriptor_source;
            return true;
        }
        bool updateGroupedPrefillExpertMask(
            const uint8_t *expert_mask,
            int num_experts) override
        {
            ++expert_mask_publications;
            return expert_mask != nullptr && num_experts > 0;
        }
        bool groupedExpertDecodeFromRuntime(
            DeviceMoELayerRuntime *,
            const TensorBase *,
            int,
            int,
            int,
            ITensor *,
            int,
            int,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::RuntimePlacementTable,
            ITensor * = nullptr) override
        {
            ++fused_runtime_decode_calls;
            last_fused_descriptor_source = descriptor_source;
            return true;
        }
    };

    // =========================================================================
    // Minimal stub ITensorGatedDeltaNet for GDN tests
    // =========================================================================

    class StubGDNKernel : public ITensorGatedDeltaNet
    {
    public:
        explicit StubGDNKernel(bool state_ready, int state_size = 0, bool supports_padded_real_length = true)
            : state_ready_(state_ready),
              state_size_(state_size),
              supports_padded_real_length_(supports_padded_real_length) {}

        bool isGPUStateReady(int required_state_size) const override
        {
            return state_ready_ && (state_size_ == required_state_size);
        }

        bool supportsPaddedPrefillRealLength() const override
        {
            return supports_padded_real_length_;
        }

        bool supportsRequestLiveStateBank(
            int request_count,
            int state_size) const override
        {
            return supports_padded_real_length_ &&
                   request_count > 0 &&
                   state_size > 0;
        }

        bool chunk_forward(
            const float *, const float *, const float *,
            const float *, const float *,
            const float *, const float *,
            float *, float *,
            int, int, int, int, int, bool) override
        {
            return true;
        }

        bool chunkForwardMergedQKV(
            const float *, int,
            const float *, const float *,
            const float *, const float *,
            float *, float *,
            int, int, int, int, int,
            int, int, bool) override
        {
            return true;
        }

        bool recurrent_step(
            const float *, const float *, const float *,
            const float *, const float *,
            const float *, const float *,
            float *, float *,
            int, int, int, bool) override
        {
            return true;
        }

    private:
        bool state_ready_;
        int state_size_;
        bool supports_padded_real_length_;
    };

    // =========================================================================
    // Minimal stub ITensorGemm for expert GEMM engine checks
    // =========================================================================

    class StubGemmEngine : public ITensorGemm
    {
    public:
        void setNativeDescShape(int n, int k)
        {
            desc_n_ = n;
            desc_k_ = k;
        }

        bool supports_device(int) const override { return true; }
        bool multiply_tensor(
            const TensorBase *, TensorBase *,
            int, int, int,
            bool, float, float,
            const TensorBase *,
            const IMPIContext *,
            int,
            DeviceWorkspaceManager *,
            int) override
        {
            return true;
        }
        bool exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out) override
        {
            out = runtimeDesc(0x71000000u, desc_n_, desc_k_);
            return true;
        }
        bool canReleaseSourceWeightTensor() const override
        {
            /*
             * This descriptor-only test engine owns no TensorBase view. Marking
             * that fact explicitly exercises the same lifetime contract used by
             * independently packed production GEMM engines.
             */
            return true;
        }

    private:
        int desc_n_ = 128;
        int desc_k_ = 64;
    };

} // anonymous namespace

// =========================================================================
// MoERoutingStage — Prefill Graph Capturability
// =========================================================================

class MoERoutingPrefillGraphCapture : public ::testing::Test
{
protected:
    static constexpr int D_MODEL = 64;
    static constexpr int NUM_EXPERTS = 4;
    static constexpr int TOP_K = 2;
    static constexpr int SEQ_LEN = 16; // prefill

    std::unique_ptr<FP32Tensor> input_;
    std::unique_ptr<FP32Tensor> gate_weights_;
    std::unique_ptr<FP32Tensor> output_indices_;
    std::unique_ptr<FP32Tensor> output_weights_;
    StubMoEKernel stub_kernel_;
    int32_t active_rows_ = SEQ_LEN;

    void SetUp() override
    {
        input_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
        gate_weights_ = TestTensorFactory::createFP32({NUM_EXPERTS, D_MODEL});
        output_indices_ = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
        output_weights_ = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
    }

    MoERoutingStage::Params makeValidPrefillParams() const
    {
        MoERoutingStage::Params p;
        p.device_id = DeviceId::rocm(0);
        p.seq_len = SEQ_LEN;
        p.d_model = D_MODEL;
        p.num_experts = NUM_EXPERTS;
        p.top_k = TOP_K;
        p.input = input_.get();
        p.gate_weights = gate_weights_.get();
        p.output_indices = output_indices_.get();
        p.output_weights = output_weights_.get();
        p.active_row_count_device = &active_rows_;
        return p;
    }
};

TEST_F(MoERoutingPrefillGraphCapture, PrefillCapturableWhenAllConditionsMet)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Prefill routing should be capturable on ROCm with valid buffers and kernel";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Prefill routing should not be capturable without ROCm";
#endif
}

/**
 * @brief Multi-row verifier routing prepares capture without eager arithmetic.
 *
 * A fixed grouped verifier row count is decode geometry backed by a stable
 * device scalar. Launch preparation binds the persistent backend launcher once;
 * changing the scalar's value never changes the host launch policy or asks the
 * stage to upload replay metadata.
 */
TEST_F(MoERoutingPrefillGraphCapture, GroupedVerifierUsesCaptureOnlyLaunchPreparation)
{
    auto params = makeValidPrefillParams();
    params.force_decode_equivalent_verifier_prefill = true;
    MoERoutingStage stage(params);

    EXPECT_EQ(
        stage.graphLaunchPreparationPolicy(),
        GraphLaunchPreparationPolicy::CaptureOnly);
    EXPECT_EQ(stage.activeRowCountDeviceForTesting(), &active_rows_);
}

/**
 * @brief Prove capture preparation binds every grouped-verifier route depth.
 *
 * This is the stage-level regression for the Q8 router cache miss discovered
 * by the CUDAx2 long-context LLEP lane.  Preparation must describe the actual
 * grouped decode-equivalent route without executing a warmup inference row.
 */
TEST_F(MoERoutingPrefillGraphCapture,
       GroupedVerifierMTotalPreparesTypedRouterResources)
{
    void *const producer_stream =
        reinterpret_cast<void *>(uintptr_t{0x4400});

    for (int verifier_rows = 1; verifier_rows <= 15; ++verifier_rows)
    {
        auto params = makeValidPrefillParams();
        params.seq_len = verifier_rows;
        params.force_decode_equivalent_verifier_prefill = true;
        MoERoutingStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);

        ASSERT_TRUE(stage.prepareGraphLaunch(nullptr, producer_stream))
            << "grouped verifier M=" << verifier_rows;
        EXPECT_EQ(stub_kernel_.last_prepared_route_gate, gate_weights_.get());
        EXPECT_EQ(
            stub_kernel_.last_route_launch_plan.kind,
            MoERouteLaunchKind::DecodeEquivalentVerifier);
        EXPECT_EQ(
            stub_kernel_.last_route_launch_plan.physical_rows,
            verifier_rows);
        EXPECT_EQ(stub_kernel_.last_route_launch_plan.d_model, D_MODEL);
        EXPECT_EQ(stub_kernel_.last_route_launch_plan.num_experts, NUM_EXPERTS);
        EXPECT_EQ(stub_kernel_.last_route_launch_plan.top_k, TOP_K);
    }

    EXPECT_EQ(stub_kernel_.route_launch_preparations, 15);
    EXPECT_EQ(stub_kernel_.verifier_route_calls, 0)
        << "capture preparation must not warm up by executing router arithmetic";
}

TEST_F(MoERoutingPrefillGraphCapture,
       PaddedPrefillRequiresDeviceOwnedRowCount)
{
    auto params = makeValidPrefillParams();
    params.active_row_count_device = nullptr;
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsLazyPrefillGraphCapturePreflight())
        << "An exact-shape grouped launch remains capturable.";
#endif
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "A reusable padded graph has no legal host-side row-count source.";
    EXPECT_FALSE(stage.supportsPaddedPrefillRealLengthContract());
}

/**
 * @brief A one-row prefix suffix remains a prefill graph, not decode policy.
 *
 * Prefix restore deliberately admits a one-token uncached suffix. The
 * physical M=1 router uses the same runtime-table kernel as serial decode, but
 * its device-owned active-row scalar identifies the surrounding transaction as
 * prefill. Cold serving-family materialization must therefore admit it on both
 * GPU backends without an eager warmup or a parity-only retry.
 */
TEST_F(MoERoutingPrefillGraphCapture,
       OneRowPrefixSuffixAdvertisesDeviceOwnedPrefillCapture)
{
    const auto expect_backend =
        [&](DeviceId device, bool backend_supported, const char *backend_name)
    {
        auto input = TestTensorFactory::createFP32({1, D_MODEL});
        auto output_indices = TestTensorFactory::createFP32({TOP_K, 1});
        auto output_weights = TestTensorFactory::createFP32({TOP_K, 1});
        int32_t active_rows = 1;

        MoERuntimeTable runtime_table(
            DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(
            0, routingRuntimeUpdate(1, NUM_EXPERTS, D_MODEL)));
        ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

        MoERoutingStage::Params params;
        params.device_id = device;
        params.seq_len = 1;
        params.d_model = D_MODEL;
        params.num_experts = NUM_EXPERTS;
        params.top_k = TOP_K;
        params.input = input.get();
        params.gate_weights = gate_weights_.get();
        params.output_indices = output_indices.get();
        params.output_weights = output_weights.get();
        params.layer_idx = 0;
        params.moe_runtime_table = &runtime_table;
        params.active_row_count_device = &active_rows;

        MoERoutingStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);
        EXPECT_EQ(
            stage.supportsLazyPrefillGraphCapturePreflight(),
            backend_supported)
            << backend_name;
        EXPECT_EQ(
            stage.supportsPaddedPrefillGraphCapturePreflight(),
            backend_supported)
            << backend_name;
        EXPECT_EQ(stage.isGraphCapturable(), backend_supported)
            << backend_name;
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif
#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

/**
 * @brief Lock down the heterogeneous overlay verifier publication protocol.
 *
 * The continuation GPU router must retain selected expert IDs in its own
 * per-layer ledger while routing, then expose that exact ledger to the later
 * accepted-state transaction.  The unit uses pointer sentinels only: no device
 * allocation or GPU work is permitted in this protocol-level suite.
 */
TEST_F(MoERoutingPrefillGraphCapture,
       OverlayTicketVerifierRetainsThenCommitsOneTypedLedger)
{
#if defined(HAVE_ROCM)
    const DeviceId device = DeviceId::rocm(0);
    std::vector<int32_t> retained_experts(
        static_cast<size_t>(SEQ_LEN * TOP_K), -1);
    std::vector<int32_t> retained_participants(
        static_cast<size_t>(SEQ_LEN * TOP_K), -1);
    MoERuntimeTable runtime_table(
        DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
    auto &runtime_host = runtime_table.hostLayerState(0);
    runtime_host.deferred_verifier_route_expert_ids =
        retained_experts.data();
    runtime_host.deferred_verifier_route_participant_ids =
        retained_participants.data();
    runtime_host.deferred_verifier_route_capacity =
        static_cast<uint32_t>(retained_experts.size());

    DeviceResidentFP32Tensor input({SEQ_LEN, D_MODEL});
    DeviceResidentFP32Tensor gate({NUM_EXPERTS, D_MODEL});
    DeviceResidentFP32Tensor indices({SEQ_LEN * TOP_K, 1});
    DeviceResidentFP32Tensor weights({SEQ_LEN * TOP_K, 1});
    input.markResidentForGraphCaptureTest(device);
    gate.markResidentForGraphCaptureTest(device);
    indices.markResidentForGraphCaptureTest(device);
    weights.markResidentForGraphCaptureTest(device);

    auto params = makeValidPrefillParams();
    params.device_id = device;
    params.input = &input;
    params.gate_weights = &gate;
    params.output_indices = &indices;
    params.output_weights = &weights;
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;
    params.force_decode_equivalent_verifier_prefill = true;
    params.defer_overlay_grouped_verifier_histogram_publication = true;

    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);
    void *const route_stream =
        reinterpret_cast<void *>(uintptr_t{0x5500});
    void *const publication_stream =
        reinterpret_cast<void *>(uintptr_t{0x6600});
    stage.setGPUStream(route_stream);
    MockDeviceContext ctx(device, ComputeBackendType::GPU_ROCM);

    ASSERT_TRUE(stage.requiresCommittedGroupedVerifierHistogramPublication());
    ASSERT_TRUE(stage.execute(&ctx));
    EXPECT_EQ(stub_kernel_.verifier_route_calls, 1);
    EXPECT_EQ(
        stub_kernel_.last_deferred_selected_route_ledger,
        runtime_table.deviceLayerState(0));

    int32_t accepted_rows = SEQ_LEN - 1;
    int32_t publication_ok = 1;
    ASSERT_TRUE(stage.enqueueCommittedGroupedVerifierHistograms(
        &accepted_rows,
        &publication_ok,
        /*request_count=*/1,
        /*rows_per_request=*/SEQ_LEN,
        publication_stream));
    EXPECT_EQ(stub_kernel_.verifier_histogram_commit_calls, 1);
    EXPECT_EQ(
        stub_kernel_.last_committed_route_ledger,
        runtime_table.deviceLayerState(0));
    EXPECT_EQ(
        stub_kernel_.last_histogram_commit_stream,
        publication_stream);
#else
    GTEST_SKIP() << "ROCm graph-capture path is not compiled in this build";
#endif
}

TEST_F(MoERoutingPrefillGraphCapture,
       OverlayTicketVerifierRejectsIncompleteDeferredLedger)
{
    MoERuntimeTable runtime_table(
        DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
    auto params = makeValidPrefillParams();
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;
    params.force_decode_equivalent_verifier_prefill = true;
    params.defer_overlay_grouped_verifier_histogram_publication = true;

    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);
    EXPECT_FALSE(stage.supportsGraphCaptureAfterLaunchPreparation());
    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithoutKernel)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    MoERoutingStage stage(params);
    // moe_kernel_ left as nullptr (default)

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Cold padded preflight should allow routing before kernel warmup";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Prefill routing should not be capturable without cached kernel";
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation());
#else
    EXPECT_FALSE(stage.supportsGraphCaptureAfterLaunchPreparation());
#endif
}

TEST_F(MoERoutingPrefillGraphCapture, CudaForcedVerifierReplaySeqLenOneUsesPrefillCaptureContract)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto output_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({TOP_K, 1});

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, routingRuntimeUpdate(1, NUM_EXPERTS, D_MODEL)));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

    MoERoutingStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.input = input.get();
    params.gate_weights = gate_weights_.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;
    params.force_grouped_verifier_prefill_for_decode = true;

    MoERoutingStage cold_stage(params);
#if defined(HAVE_CUDA)
    EXPECT_TRUE(cold_stage.supportsLazyPrefillGraphCapturePreflight());
    EXPECT_TRUE(cold_stage.supportsGraphCaptureAfterLaunchPreparation());
    EXPECT_FALSE(cold_stage.isGraphCapturable())
        << "Forced seq_len=1 verifier replay must wait for the prefill router kernel warmup";

    cold_stage.setMoEKernelForTesting(&stub_kernel_);
    EXPECT_TRUE(cold_stage.isGraphCapturable())
        << "Forced seq_len=1 verifier replay should use the prefill routing capture contract";
#else
    EXPECT_FALSE(cold_stage.supportsLazyPrefillGraphCapturePreflight());
    EXPECT_FALSE(cold_stage.supportsGraphCaptureAfterLaunchPreparation());
    EXPECT_FALSE(cold_stage.isGraphCapturable());
#endif
}

TEST_F(MoERoutingPrefillGraphCapture, CudaForcedVerifierReplayDoesNotFallBackToDecodeCapture)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto output_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({TOP_K, 1});

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, routingRuntimeUpdate(1, NUM_EXPERTS, D_MODEL)));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

    MoERoutingStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.input = input.get();
    params.gate_weights = gate_weights_.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;
    params.force_grouped_verifier_prefill_for_decode = true;

    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);
#if defined(HAVE_CUDA)
    EXPECT_TRUE(stage.supportsLazyPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation());
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Forced CUDA verifier replay must keep the grouped-prefill route even "
           "without any runtime capability-advertisement switch";
#else
    EXPECT_FALSE(stage.supportsLazyPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.supportsGraphCaptureAfterLaunchPreparation());
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Forced verifier replay must hard-require grouped prefill instead of using decode capture";
#endif
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsOnCPU)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.device_id = DeviceId::cpu();
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillUsesBackendGroupedCapability)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Grouped prefill is now a hard GPU capability requirement, not an "
           "optional runtime advertisement";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithNullInput)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.input = nullptr;
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithNullGateWeights)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.gate_weights = nullptr;
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithNullOutputIndices)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.output_indices = nullptr;
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithNullOutputWeights)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.output_weights = nullptr;
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsInvalidTopK)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    // top_k = 0
    {
        auto params = makeValidPrefillParams();
        params.top_k = 0;
        MoERoutingStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);
        EXPECT_FALSE(stage.isGraphCapturable());
    }

    // top_k > num_experts
    {
        auto params = makeValidPrefillParams();
        params.top_k = NUM_EXPERTS + 1;
        MoERoutingStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);
        EXPECT_FALSE(stage.isGraphCapturable());
    }
}

TEST_F(MoERoutingPrefillGraphCapture, DeviceRuntimeHistogramNeedsNoHostReplayCallback)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 1;
    histogram_config.num_experts = NUM_EXPERTS;
    histogram_config.top_k = TOP_K;
    histogram_config.sockets = {params.device_id};
    histogram_config.ownership = MoELayeredExpertOwnership::uniform(
        /*num_layers=*/1,
        /*participant_count=*/1,
        std::vector<int>(NUM_EXPERTS, 0));
    DecodeExpertHistogram histogram(histogram_config);
    params.decode_histogram = &histogram;

    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.isGraphCapturable());
#else
    (void)stage;
#endif
}

// =========================================================================
// MoEExpertComputeStage — Fixed-Topology Prefill Graph Capturability
// =========================================================================

class MoEExpertPrefillGraphCapture : public ::testing::Test
{
protected:
    static constexpr int D_MODEL = 64;
    static constexpr int NUM_EXPERTS = 4;
    static constexpr int TOP_K = 2;
    static constexpr int SEQ_LEN = 16;
    static constexpr int INTERMEDIATE = 128;

    // Declared before the tensors so the simulated backend outlives every
    // allocation that refers to it during fixture teardown.
    test::MockBackend backend_{DeviceType::ROCm};
    std::unique_ptr<FP32Tensor> input_;
    std::unique_ptr<FP32Tensor> output_;
    std::unique_ptr<FP32Tensor> routing_indices_;
    std::unique_ptr<FP32Tensor> routing_weights_;
    StubMoEKernel stub_kernel_;
    std::vector<StubGemmEngine> stub_gemms_;
    std::vector<ITensorGemm *> gate_gemm_ptrs_;
    std::vector<ITensorGemm *> up_gemm_ptrs_;
    std::vector<ITensorGemm *> down_gemm_ptrs_;
    int32_t active_rows_ = SEQ_LEN;
    void *execution_stream_ = nullptr;

    void SetUp() override
    {
        execution_stream_ = backend_.createStream(0);
        ASSERT_NE(execution_stream_, nullptr);

        input_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
        output_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
        routing_indices_ = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
        routing_weights_ = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

        // The execute-based warmup tests publish output completion. Give the
        // tensor physically consistent, host-backed simulated ROCm storage so
        // publication remains event-backed without performing real GPU work.
        output_->setBackendForTesting(&backend_);
        ASSERT_TRUE(output_->allocateOnDevice(DeviceId::rocm(0)));

        // Create stub GEMM engines for all experts (gate, up, down × num_experts)
        stub_gemms_.resize(static_cast<size_t>(NUM_EXPERTS * 3));
        gate_gemm_ptrs_.resize(static_cast<size_t>(NUM_EXPERTS));
        up_gemm_ptrs_.resize(static_cast<size_t>(NUM_EXPERTS));
        down_gemm_ptrs_.resize(static_cast<size_t>(NUM_EXPERTS));

        for (int e = 0; e < NUM_EXPERTS; ++e)
        {
            stub_gemms_[static_cast<size_t>(e * 3 + 0)].setNativeDescShape(INTERMEDIATE, D_MODEL);
            stub_gemms_[static_cast<size_t>(e * 3 + 1)].setNativeDescShape(INTERMEDIATE, D_MODEL);
            stub_gemms_[static_cast<size_t>(e * 3 + 2)].setNativeDescShape(D_MODEL, INTERMEDIATE);
            gate_gemm_ptrs_[static_cast<size_t>(e)] = &stub_gemms_[static_cast<size_t>(e * 3 + 0)];
            up_gemm_ptrs_[static_cast<size_t>(e)] = &stub_gemms_[static_cast<size_t>(e * 3 + 1)];
            down_gemm_ptrs_[static_cast<size_t>(e)] = &stub_gemms_[static_cast<size_t>(e * 3 + 2)];
        }
    }

    void TearDown() override
    {
        if (execution_stream_)
        {
            backend_.destroyStream(execution_stream_, 0);
            execution_stream_ = nullptr;
        }
    }

    MoEExpertComputeStage::Params makeValidPrefillParams() const
    {
        MoEExpertComputeStage::Params p;
        p.device_id = DeviceId::rocm(0);
        p.seq_len = SEQ_LEN;
        p.d_model = D_MODEL;
        p.num_experts = NUM_EXPERTS;
        p.top_k = TOP_K;
        p.expert_intermediate = INTERMEDIATE;
        p.local_expert_start = 0;
        p.local_expert_count = NUM_EXPERTS; // full local ownership
        p.input = input_.get();
        p.output = output_.get();
        p.routing_indices = routing_indices_.get();
        p.routing_weights = routing_weights_.get();
        p.active_row_count_device = &active_rows_;
        p.prepared_gate_gemm = gate_gemm_ptrs_;
        p.prepared_up_gemm = up_gemm_ptrs_;
        p.prepared_down_gemm = down_gemm_ptrs_;
        // No expert_mask → all enabled
        // No replicas
        return p;
    }
};

TEST_F(MoEExpertPrefillGraphCapture,
       FullyReplicatedLocalRowsBypassParticipantAssignment)
{
    auto params = makeValidPrefillParams();
    params.routed_assignment_policy =
        RoutedExpertAssignmentPolicy::LeastLoadedResident;
    params.routed_row_execution_policy =
        RoutedExpertRowExecutionPolicy::FullyReplicatedLocal;

    MoEExpertComputeStage stage(params);
    EXPECT_EQ(
        stage.routedExpertRowExecutionPolicyForTesting(),
        RoutedExpertRowExecutionPolicy::FullyReplicatedLocal);
    EXPECT_TRUE(stage.supportsRequestedRoutedAssignmentPolicyForTesting())
        << "A complete local replica does not need a LocalTP assignment context.";
}

TEST_F(MoEExpertPrefillGraphCapture,
       FullyReplicatedLocalRowsRejectPartialExpertOwnership)
{
    auto params = makeValidPrefillParams();
    params.routed_row_execution_policy =
        RoutedExpertRowExecutionPolicy::FullyReplicatedLocal;
    params.expert_mask = {true, true, true, false};

    EXPECT_THROW(
        MoEExpertComputeStage stage(params),
        std::invalid_argument);
}

TEST_F(MoEExpertPrefillGraphCapture,
       FullyReplicatedLocalRowsRejectCanonicalRoutePublication)
{
    auto params = makeValidPrefillParams();
    params.routed_row_execution_policy =
        RoutedExpertRowExecutionPolicy::FullyReplicatedLocal;
    params.canonical_route_contributions = output_.get();

    EXPECT_THROW(
        MoEExpertComputeStage stage(params),
        std::invalid_argument);
}

TEST_F(MoEExpertPrefillGraphCapture, FixedTopologyCapturableWhenReady)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Fixed-topology grouped prefill should be capturable with all engines ready";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

/**
 * @brief Reject reusable physical-width expert graphs without device geometry.
 *
 * Routing can invalidate a padded suffix only if every downstream routed stage
 * agrees on the same logical prefix. A physically capturable expert kernel is
 * therefore insufficient for padded replay when the graph omitted that owner.
 */
TEST_F(MoEExpertPrefillGraphCapture,
       PaddedPrefillRequiresDeviceOwnedRowCount)
{
    auto params = makeValidPrefillParams();
    params.active_row_count_device = nullptr;
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsLazyPrefillGraphCapturePreflight())
        << "An exact physical-shape grouped launch remains capturable.";
#endif
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "A reusable physical width requires one device-owned logical count.";
}

/**
 * @brief Prefix restore may reuse the production one-row runtime-table kernel.
 *
 * The stable device row-count binding authenticates the prefill transaction;
 * the stage must not infer phase from physical M or require a separate eager
 * arithmetic path for the one uncached suffix token.
 */
TEST_F(MoEExpertPrefillGraphCapture,
       OneRowPrefixSuffixUsesDeviceRoutedGraphPreflight)
{
    const auto expect_backend =
        [&](DeviceId device, bool backend_supported, const char *backend_name)
    {
        MoERuntimeTable runtime_table(
            DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        auto params = makeValidPrefillParams();
        params.device_id = device;
        params.seq_len = 1;
        params.layer_idx = 0;
        params.moe_runtime_table = &runtime_table;

        MoEExpertComputeStage stage(params);
        EXPECT_EQ(
            stage.supportsPaddedPrefillGraphCapturePreflight(),
            backend_supported)
            << backend_name << " must admit the exact one-row prefix suffix";

        params.active_row_count_device = nullptr;
        MoEExpertComputeStage unauthenticated_stage(params);
        EXPECT_FALSE(
            unauthenticated_stage.supportsLazyPrefillGraphCapturePreflight())
            << backend_name << " one-row prefill requires device-owned geometry";
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif
#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWithoutKernel)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    MoEExpertComputeStage stage(params);
    // moe_kernel_ left nullptr

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Cold padded preflight should allow fixed-topology MoE before kernel warmup";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Should not be capturable without MoE kernel";
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation());
#else
    EXPECT_FALSE(stage.supportsGraphCaptureAfterLaunchPreparation());
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsOnCPU)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.device_id = DeviceId::cpu();
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoEExpertPrefillGraphCapture, UsesBackendGroupedCapability)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Fixed-topology grouped prefill must stay available on compiled GPU "
           "backends without any runtime capability-advertisement switch";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWithPartialExpertOwnership)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.local_expert_count = NUM_EXPERTS - 1; // not full
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable())
        << "Should reject when not all experts are locally owned";
}

TEST_F(MoEExpertPrefillGraphCapture, ForcedDecodeReplaySupportsPublishedPartialOwnershipMask)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto expect_backend = [&](DeviceId device, bool backend_supported, const char *backend_name)
    {
        auto full_params = makeValidPrefillParams();
        full_params.device_id = device;
        full_params.seq_len = 1;
        full_params.force_grouped_verifier_prefill_for_decode = true;

        MoEExpertComputeStage full_stage(full_params);
        full_stage.setMoEKernelForTesting(&stub_kernel_);
        EXPECT_EQ(full_stage.usesFixedTopologyGroupedVerifierReplayForTesting(),
                  backend_supported)
            << backend_name << " full-ownership verifier correction replay should "
            << "advertise the fixed descriptor-table route only when the backend supports it";

        auto partial_params = full_params;
        partial_params.local_expert_count = NUM_EXPERTS - 1;
        partial_params.expert_mask = {true, true, true, false};
        partial_params.prepared_gate_gemm[3] = nullptr;
        partial_params.prepared_up_gemm[3] = nullptr;
        partial_params.prepared_down_gemm[3] = nullptr;
        MoEExpertComputeStage partial_stage(partial_params);
        partial_stage.setMoEKernelForTesting(&stub_kernel_);
        EXPECT_EQ(
            partial_stage.usesFixedTopologyGroupedVerifierReplayForTesting(),
            backend_supported)
            << backend_name
            << " partial ownership must use the same economical grouped "
               "descriptor route with a graph-published local expert mask";
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

/**
 * @brief Prove that every supported verifier depth enters grouped-prefill setup.
 *
 * Grouped verifier stages intentionally carry the forced-verifier marker at
 * every M.  That marker selects decode-equivalent arithmetic; it must not
 * exclude the stage from the pre-capture operation that binds the persistent
 * MoE kernel, GEMM descriptor tables, expert mask, and runtime grouping arena.
 * M=8 is especially important because it reproduces the long-context LLEP
 * capture failure that first exposed the contradictory route predicates.
 */
TEST_F(MoEExpertPrefillGraphCapture,
       GroupedVerifierMTotalUsesGraphStablePrefillPlacement)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    const auto expect_backend =
        [&](DeviceId device, bool backend_supported, const char *backend_name)
    {
        MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        for (int verifier_rows = 1; verifier_rows <= 15; ++verifier_rows)
        {
            auto params = makeValidPrefillParams();
            params.device_id = device;
            params.seq_len = verifier_rows;
            params.force_grouped_verifier_prefill_for_decode = true;
            params.moe_runtime_table = &runtime_table;
            params.layer_idx = 0;
            params.use_runtime_row_grouping = verifier_rows > 1;

            MoEExpertComputeStage stage(params);
            EXPECT_EQ(
                stage.usesGraphStableFixedTopologyPrefillPlacement(),
                backend_supported)
                << backend_name << " grouped verifier M=" << verifier_rows
                << " must enter graph-stable grouped-prefill preparation";
        }
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, MaskedFixedTopologyRequiresPublishedDeviceMask)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.expert_mask = {true, false, true, false};
    params.prepared_gate_gemm[1] = nullptr;
    params.prepared_up_gemm[1] = nullptr;
    params.prepared_down_gemm[1] = nullptr;
    params.prepared_gate_gemm[3] = nullptr;
    params.prepared_up_gemm[3] = nullptr;
    params.prepared_down_gemm[3] = nullptr;
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.hasPublishedFixedTopologyMaskForTesting());
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Static masked LocalTP prefill must warm and publish its owner mask "
           "before graph capture can consume the backend-owned device slot";
    EXPECT_NE(stage.graphCaptureReadinessDebugString().find(
                  "fixed_mask_publication=needs_publication"),
              std::string::npos);
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, RuntimeGroupedLLEPDoesNotRequireUnusedFixedMaskPublication)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    const auto expect_runtime_table_is_authoritative =
        [&](DeviceId device, const char *backend_name)
    {
        MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        auto params = makeValidPrefillParams();
        params.device_id = device;
        params.layer_idx = 0;
        params.expert_mask = {true, false, true, false};
        params.prepared_gate_gemm[1] = nullptr;
        params.prepared_up_gemm[1] = nullptr;
        params.prepared_down_gemm[1] = nullptr;
        params.prepared_gate_gemm[3] = nullptr;
        params.prepared_up_gemm[3] = nullptr;
        params.prepared_down_gemm[3] = nullptr;
        params.moe_runtime_table = &runtime_table;
        params.use_runtime_row_grouping = true;
        params.routed_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        MoEExpertComputeStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);

        /*
         * Model the already allocated device-resident prefill workspace without
         * allocating GPU memory in a unit test.  The production predicate checks
         * every stable pointer and capacity below on each capture boundary.
         */
        int32_t route_expert_ids = 0;
        float route_weights = 0.0f;
        int32_t route_participant_ids = 0;
        int32_t expert_counts = 0;
        int32_t expert_offsets = 0;
        int32_t grouped_token_ids = 0;
        float grouped_route_weights = 0.0f;
        int32_t llep_split_ends = 0;
        least_loaded_ep::LeastLoadedExpertAssignmentSpan assignment_span{};
        least_loaded_ep::LeastLoadedExpertWeightTransfer weight_transfer{};
        auto &runtime_state = runtime_table.hostLayerState(0);
        runtime_state.prefill_token_capacity =
            static_cast<uint32_t>(params.seq_len);
        runtime_state.prefill_route_capacity =
            static_cast<uint32_t>(params.seq_len * params.top_k);
        runtime_state.route_expert_ids = &route_expert_ids;
        runtime_state.route_weights = &route_weights;
        runtime_state.route_participant_ids = &route_participant_ids;
        runtime_state.expert_counts = &expert_counts;
        runtime_state.expert_offsets = &expert_offsets;
        runtime_state.grouped_token_ids = &grouped_token_ids;
        runtime_state.grouped_route_weights = &grouped_route_weights;
        runtime_state.reserved_ptrs[0] = &llep_split_ends;
        runtime_state.reserved_ptrs[1] = &assignment_span;
        runtime_state.reserved_ptrs[2] = &weight_transfer;
        runtime_state.reserved_u64[0] =
            static_cast<uint64_t>(NUM_EXPERTS) *
            kDeviceMoEMaxParticipants;
        runtime_state.reserved_u64[1] = runtime_state.reserved_u64[0];
        stage.setRuntimePrefillGroupingAvailableForTesting(true);

        ASSERT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
            << backend_name;
        EXPECT_FALSE(stage.hasPublishedFixedTopologyMaskForTesting())
            << backend_name
            << " runtime-table grouping must not publish the unused static owner mask";
        EXPECT_TRUE(stage.isGraphCapturable())
            << backend_name
            << " warmed LLEP runtime table is the sole placement source and must be "
               "capture-ready without an unrelated fixed-mask publication";
        const std::string readiness = stage.graphCaptureReadinessDebugString();
        EXPECT_NE(readiness.find("route=runtime_table_grouped_rows"), std::string::npos)
            << backend_name;
        EXPECT_NE(readiness.find("fixed_mask_consumed=false"), std::string::npos)
            << backend_name;
        EXPECT_NE(readiness.find("runtime_row_grouping_ready=true"), std::string::npos)
            << backend_name;
    };

#if defined(HAVE_CUDA)
    expect_runtime_table_is_authoritative(DeviceId::cuda(0), "CUDA");
#endif
#if defined(HAVE_ROCM)
    expect_runtime_table_is_authoritative(DeviceId::rocm(0), "ROCm");
#endif
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
    GTEST_SKIP() << "GPU graph capture is not compiled in this build";
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, MaskedFixedTopologyRejectsMissingLocalEngine)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.expert_mask = {true, false, true, false};
    params.prepared_gate_gemm[2] = nullptr;
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Masked LocalTP prefill still needs every locally computed expert engine";
}

TEST_F(MoEExpertPrefillGraphCapture, AllowsFixedTopologyPrefillWithReplicas)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.replica_set.num_replicated = 1;
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Regular fixed-topology prefill can capture with a static replica topology";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWithMissingGemmEngines)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    // One expert has null gate GEMM
    params.prepared_gate_gemm[1] = nullptr;
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Should reject when any expert GEMM engine is null";
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsDecodeSeqLen)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();
    params.seq_len = 1; // decode — should not hit fixed-topology prefill path
    // (but might hit decode path if conditions match)
    params.moe_runtime_table = nullptr; // ensure decode path also fails
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    // Without runtime table, decode path also fails
    EXPECT_FALSE(stage.isGraphCapturable())
        << "seq_len=1 without runtime table should not be capturable via either path";
}

TEST_F(MoEExpertPrefillGraphCapture,
       ExplicitRoutingDecodePreparesFusedCaptureWithoutRuntimeTable)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    const auto expect_backend =
        [&](DeviceId device, bool backend_supported, const char *backend_name)
    {
        auto params = makeValidPrefillParams();
        params.device_id = device;
        params.seq_len = 1;
        params.layer_idx = 0;
        params.moe_runtime_table = nullptr;
        params.require_device_routing_tensor_decode = true;
        params.local_expert_count = NUM_EXPERTS / 2;
        params.expert_mask = {true, true, false, false};
        params.prepared_gate_gemm[2] = nullptr;
        params.prepared_up_gemm[2] = nullptr;
        params.prepared_down_gemm[2] = nullptr;
        params.prepared_gate_gemm[3] = nullptr;
        params.prepared_up_gemm[3] = nullptr;
        params.prepared_down_gemm[3] = nullptr;

        MoEExpertComputeStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);

        EXPECT_EQ(
            stage.supportsGraphCaptureAfterLaunchPreparation(),
            backend_supported)
            << backend_name
            << " explicit device routing must be a first-class cold capture route";
        EXPECT_FALSE(stage.isGraphCapturable())
            << backend_name
            << " explicit routing must not capture before descriptors, mask, and "
               "pointer arrays are published";

        if (!backend_supported)
            return;

        const int preparations_before =
            stub_kernel_.runtime_decode_preparations;
        const int masks_before = stub_kernel_.expert_mask_publications;
        ASSERT_TRUE(stage.prepareGraphLaunch(nullptr, execution_stream_))
            << backend_name;
        EXPECT_TRUE(stage.isGraphCapturable()) << backend_name;
        EXPECT_EQ(
            stub_kernel_.runtime_decode_preparations,
            preparations_before + 1);
        EXPECT_EQ(stub_kernel_.expert_mask_publications, masks_before + 1);
        EXPECT_EQ(
            stub_kernel_.last_fused_descriptor_source,
            MoEDecodeDescriptorSource::StaticDescriptorTable);

        const std::string readiness =
            stage.graphCaptureReadinessDebugString();
        EXPECT_NE(
            readiness.find("route=explicit_routing_decode"),
            std::string::npos)
            << readiness;
        EXPECT_NE(
            readiness.find("explicit_decode_ready=true"),
            std::string::npos)
            << readiness;
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, DeviceRoutedDecodeRequiresFusedRuntimeWarmupBeforeCapture)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, routingRuntimeUpdate(1, NUM_EXPERTS, D_MODEL)));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

    auto expectWarmupContract = [&](DeviceId device)
    {
        auto params = makeValidPrefillParams();
        params.device_id = device;
        params.seq_len = 1;
        params.layer_idx = 0;
        params.moe_runtime_table = &runtime_table;

        MoEExpertComputeStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);

        EXPECT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation())
            << "Cold routed MoE decode should advertise that warmup can make it capturable on "
            << device.to_string();
        EXPECT_FALSE(stage.isGraphCapturable())
            << "Routed MoE decode must not capture before fused runtime pointer arrays are staged on "
            << device.to_string();

        stage.setRuntimeGroupedDecodeLaunchStatePreparedForTesting(true);
        EXPECT_TRUE(stage.isGraphCapturable())
            << "After fused runtime warmup, routed MoE decode is graph-capturable on "
            << device.to_string();

        stage.resetSessionState();
        EXPECT_FALSE(stage.isGraphCapturable())
            << "Session reset clears backend pointer-table readiness; routed MoE decode must warm again on "
            << device.to_string();
    };

#if defined(HAVE_ROCM)
    expectWarmupContract(DeviceId::rocm(0));
#endif
#if defined(HAVE_CUDA)
    expectWarmupContract(DeviceId::cuda(0));
#endif
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
    GTEST_SKIP() << "GPU routed MoE decode graph capture is not compiled in this build";
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, RuntimeDecodeDescriptorSourceDefaultsToStaticTables)
{
    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, routingRuntimeUpdate(1, NUM_EXPERTS, D_MODEL)));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

    auto params = makeValidPrefillParams();
    params.seq_len = 1;
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;

    MoEExpertComputeStage static_stage(params);
    EXPECT_EQ(static_stage.runtimeDecodeDescriptorSourceForTesting(),
              MoEDecodeDescriptorSource::StaticDescriptorTable)
        << "Static/off GPU decode should keep using immutable descriptor tables; "
           "only runtime top-k ids/weights need the runtime table.";

    params.weight_descriptor_source =
        MoEDecodeDescriptorSource::RuntimePlacementTable;
    MoEExpertComputeStage mutable_stage(params);
    EXPECT_EQ(mutable_stage.runtimeDecodeDescriptorSourceForTesting(),
              MoEDecodeDescriptorSource::RuntimePlacementTable)
        << "Device-side graph-stable rebalance must opt into mutable runtime descriptors.";
}

TEST_F(MoEExpertPrefillGraphCapture, MutableDecodePreservesApportionedRuntimeOwners)
{
    auto expect_backend = [&](DeviceId device, bool backend_supported)
    {
        MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(
            0,
            apportionedRuntimeUpdate(
                1,
                NUM_EXPERTS,
                D_MODEL,
                /*local_participant=*/0,
                /*participant_count=*/2,
                {0, 0, 1, 1},
                {true, true, false, false})));
        ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

        auto params = makeValidPrefillParams();
        params.device_id = device;
        params.seq_len = 1;
        params.layer_idx = 0;
        params.moe_runtime_table = &runtime_table;
        params.my_socket_id = 0;
        params.participant_count = 2;
        params.expert_mask = {true, true, false, false};
        params.weight_descriptor_source =
            MoEDecodeDescriptorSource::RuntimePlacementTable;

        MoEExpertComputeStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);
        stage.setRuntimeGroupedDecodeLaunchStatePreparedForTesting(true);

        const auto &state = runtime_table.hostLayerState(0);
        const auto &bank = state.banks[state.active_bank];
        EXPECT_EQ(bank.experts[2].owner_participant, 1)
            << "Mutable graph decode must not synthesize local ownership for remote apportioned experts";
        EXPECT_EQ(bank.resident_participant_mask[2], 2u)
            << "Remote apportioned expert residency must remain on participant 1";
        EXPECT_EQ(bank.local_compute_mask[2], 0u);
        EXPECT_EQ(stage.isGraphCapturable(), backend_supported)
            << "A coherent mutable runtime table should be accepted for graph-captured decode";
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true);
#else
    expect_backend(DeviceId::cuda(0), false);
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true);
#else
    expect_backend(DeviceId::rocm(0), false);
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, StaticDescriptorMaskedDecodeAcceptsExplicitRuntimeOwners)
{
    auto expect_backend = [&](DeviceId device, bool backend_supported)
    {
        MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(
            0,
            apportionedRuntimeUpdate(
                1,
                NUM_EXPERTS,
                D_MODEL,
                /*local_participant=*/0,
                /*participant_count=*/2,
                {0, 0, 1, 1},
                {true, true, false, false})));
        ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

        auto params = makeValidPrefillParams();
        params.device_id = device;
        params.seq_len = 1;
        params.layer_idx = 0;
        params.moe_runtime_table = &runtime_table;
        params.my_socket_id = 0;
        params.participant_count = 2;
        params.expert_mask = {true, true, false, false};
        params.weight_descriptor_source =
            MoEDecodeDescriptorSource::StaticDescriptorTable;
        params.runtime_decode_has_explicit_owner_metadata = true;

        MoEExpertComputeStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);
        stage.setRuntimeGroupedDecodeLaunchStatePreparedForTesting(true);

        const auto &state = runtime_table.hostLayerState(0);
        const auto &bank = state.banks[state.active_bank];
        EXPECT_EQ(stage.runtimeDecodeDescriptorSourceForTesting(),
                  MoEDecodeDescriptorSource::StaticDescriptorTable)
            << "Explicit owner metadata must not force transfer-slot mutable descriptors.";
        EXPECT_EQ(bank.experts[2].owner_participant, 1)
            << "Masked static-descriptor decode must preserve remote owner metadata.";
        EXPECT_EQ(bank.resident_participant_mask[2], 2u);
        EXPECT_EQ(bank.local_compute_mask[2], 0u);
        EXPECT_EQ(stage.isGraphCapturable(), backend_supported)
            << "A masked LocalTP decode bank with explicit owners should be graph-capturable "
               "without mutable transfer-slot descriptors.";
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true);
#else
    expect_backend(DeviceId::cuda(0), false);
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true);
#else
    expect_backend(DeviceId::rocm(0), false);
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, FullyReplicatedDecodePreservesExplicitRuntimeOwners)
{
    auto expect_backend = [&](DeviceId device, bool backend_supported)
    {
        constexpr int kParticipantCount = 2;
        const std::vector<int> owners{0, 1, 0, 1};

        MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        auto update = routingRuntimeUpdate(
            /*epoch=*/1,
            NUM_EXPERTS,
            D_MODEL);
        declareFullyReplicatedPlacementTopology(
            update,
            /*local_participant=*/0,
            kParticipantCount,
            owners);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
        ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, nullptr));

        auto params = makeValidPrefillParams();
        params.device_id = device;
        params.seq_len = 1;
        params.layer_idx = 0;
        params.moe_runtime_table = &runtime_table;
        params.my_socket_id = 0;
        params.participant_count = kParticipantCount;
        params.expert_mask.assign(NUM_EXPERTS, true);
        params.weight_descriptor_source =
            MoEDecodeDescriptorSource::StaticDescriptorTable;
        params.runtime_decode_has_explicit_owner_metadata = true;

        MoEExpertComputeStage stage(params);
        stage.setMoEKernelForTesting(&stub_kernel_);
        stage.setRuntimeGroupedDecodeLaunchStatePreparedForTesting(true);

        const auto &state = runtime_table.hostLayerState(0);
        const auto &bank = state.banks[state.active_bank];
        EXPECT_EQ(state.active_epoch, update.epoch)
            << "A complete replicated graph bank must not be replaced by stage-local topology synthesis";
        EXPECT_EQ(state.participant_count, kParticipantCount);
        EXPECT_EQ(bank.resident_participant_mask[1], 0b11u);
        EXPECT_EQ(bank.experts[1].owner_participant, 1);
        EXPECT_EQ(
            bank.replica_role[1],
            static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
        EXPECT_EQ(stage.runtimeDecodeDescriptorSourceForTesting(),
                  MoEDecodeDescriptorSource::StaticDescriptorTable);
        EXPECT_EQ(stage.isGraphCapturable(), backend_supported)
            << "A fully replicated LocalTP bank with explicit owner metadata should remain graph-capturable";
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true);
#else
    expect_backend(DeviceId::cuda(0), false);
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true);
#else
    expect_backend(DeviceId::rocm(0), false);
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, LaunchPreparationInitializesRuntimeBankWithoutDecode)
{
#if defined(HAVE_ROCM)
    ScopedMoEGraphCaptureFlags flags(true, true);

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);

    auto params = makeValidPrefillParams();
    params.device_id = DeviceId::rocm(0);
    params.seq_len = 1;
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;
    params.output_registered_in_arena = true;

    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);
    stage.setGPUStream(execution_stream_);
    stage.releaseRawExpertWeights();
    MockDeviceContext ctx(params.device_id, ComputeBackendType::GPU_ROCM);

    ASSERT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation());
    ASSERT_FALSE(stage.isGraphCapturable())
        << "The cold stage should require explicit launch preparation.";

    ASSERT_TRUE(stage.prepareGraphLaunch(&ctx, execution_stream_))
        << "Launch preparation must initialize the runtime placement bank and "
           "bind persistent grouped-decode descriptors.";
    EXPECT_EQ(stub_kernel_.runtime_decode_preparations, 1);
    EXPECT_EQ(stub_kernel_.fused_runtime_decode_calls, 0)
        << "Launch preparation must not execute transaction-zero arithmetic.";
    EXPECT_EQ(stub_kernel_.last_fused_descriptor_source,
              MoEDecodeDescriptorSource::StaticDescriptorTable)
        << "Ordinary runtime-routed decode should use the fast immutable descriptor tables.";
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Prepared immutable launch state must make transaction zero capturable.";

    ASSERT_TRUE(stage.execute(&ctx));
    EXPECT_EQ(stub_kernel_.fused_runtime_decode_calls, 1)
        << "Only graph execution may perform the first fused grouped decode.";
    EXPECT_FALSE(backend_.getEventRecordsForStream(execution_stream_).empty())
        << "GPU output publication must record completion on the executor-provided stream.";
#else
    GTEST_SKIP() << "ROCm graph-capture path is not compiled in this build";
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, LaunchPreparationInitializesReplicaRuntimeBankWithoutDecode)
{
#if defined(HAVE_ROCM)
    ScopedMoEGraphCaptureFlags flags(true, true);

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);

    auto params = makeValidPrefillParams();
    params.device_id = DeviceId::rocm(0);
    params.seq_len = 1;
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;
    params.output_registered_in_arena = true;
    params.expert_mask = {true, false, true, true};

    ExpertReplicaSet replicas;
    replicas.base_ownership = MoELayeredExpertOwnership::uniform(
        1, 2, {0, 1, 0, 1});
    replicas.replica_participants_by_layer.assign(
        1,
        std::vector<std::vector<bool>>(4, std::vector<bool>(2, false)));
    replicas.setReplicaOnParticipant(0, 2, 1);
    replicas.setReplicaOnParticipant(0, 3, 0);
    replicas.rebuildAggregateReplicaFlags();

    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);
    stage.setGPUStream(execution_stream_);
    stage.releaseRawExpertWeights();
    stage.setReplicaSet(replicas, /*socket_id=*/0);
    MockDeviceContext ctx(params.device_id, ComputeBackendType::GPU_ROCM);

    ASSERT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation());
    ASSERT_TRUE(stage.prepareGraphLaunch(&ctx, execution_stream_));
    EXPECT_EQ(stub_kernel_.runtime_decode_preparations, 1);
    EXPECT_EQ(stub_kernel_.fused_runtime_decode_calls, 0)
        << "Replica-aware launch preparation must not execute decode arithmetic.";
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Replica-aware runtime-table metadata should make the prepared decode "
           "stage graph-capturable.";

    ASSERT_TRUE(stage.execute(&ctx))
        << "Replicated GPU participants must use the prepared device runtime path.";
    EXPECT_EQ(stub_kernel_.fused_runtime_decode_calls, 1);
    EXPECT_FALSE(backend_.getEventRecordsForStream(execution_stream_).empty())
        << "Replica-aware output publication must retain the exact producer stream.";

    const auto &state = runtime_table.hostLayerState(0);
    ASSERT_LT(state.active_bank, 2u);
    EXPECT_EQ(state.participant_id, 0u);
    EXPECT_EQ(state.participant_count, 2u);

    const auto &bank = state.banks[state.active_bank];
    EXPECT_EQ(bank.local_compute_mask[0], 1u);
    EXPECT_EQ(bank.local_compute_mask[1], 0u);
    EXPECT_EQ(bank.local_compute_mask[2], 1u);
    EXPECT_EQ(bank.local_compute_mask[3], 1u);
    EXPECT_EQ(bank.experts[3].owner_participant, 1);
    EXPECT_TRUE(hasMoEExpertFlag(bank.experts[3].flags, DeviceMoEExpertFlags::Replicated));
    EXPECT_EQ(bank.replica_role[3], static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
#else
    GTEST_SKIP() << "ROCm graph-capture path is not compiled in this build";
#endif
}

// =========================================================================
// SharedExpertFFNStage — Prefill Graph Capturability
// =========================================================================

class SharedExpertFFNPrefillGraphCapture : public ::testing::Test
{
protected:
    static constexpr int D_MODEL = 64;
    static constexpr int SEQ_LEN = 16;
    static constexpr int INTERMEDIATE = 128;

    std::unique_ptr<FP32Tensor> input_;
    std::unique_ptr<FP32Tensor> output_;
    StubMoEKernel stub_kernel_;

    void SetUp() override
    {
        input_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
        output_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    }
};

TEST_F(SharedExpertFFNPrefillGraphCapture, PrefillPreflightSupportDoesNotRequireWarmScratch)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.input = input_.get();
    params.gate_w = gate_w.get();
    params.up_w = up_w.get();
    params.down_w = down_w.get();
    params.output = output_.get();

    SharedExpertFFNStage stage(params);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable());
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation());
#else
    EXPECT_FALSE(stage.supportsGraphCaptureAfterLaunchPreparation());
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, PrefillPreflightUsesBackendGroupedCapability)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.input = input_.get();
    params.gate_w = gate_w.get();
    params.up_w = up_w.get();
    params.down_w = down_w.get();
    params.output = output_.get();

    SharedExpertFFNStage stage(params);
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Shared-expert grouped prefill is a backend capability, not an "
           "optional runtime advertisement";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
}

/**
 * @brief One uncached prefix token admits the exact one-row shared FFN graph.
 *
 * Physical row count does not define transaction phase. The parent serving
 * graph owns the prefill identity, while this row-local stage may reuse its
 * already-proven M=1 production launch contract.
 */
TEST_F(SharedExpertFFNPrefillGraphCapture,
       OneRowPrefixSuffixSupportsExactShapePrefillCapture)
{
    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    const auto expect_backend =
        [&](DeviceId device, bool backend_supported, const char *backend_name)
    {
        SharedExpertFFNStage::Params params;
        params.device_id = device;
        params.seq_len = 1;
        params.d_model = D_MODEL;
        params.intermediate = INTERMEDIATE;
        params.input = input_.get();
        params.gate_w = gate_w.get();
        params.up_w = up_w.get();
        params.down_w = down_w.get();
        params.output = output_.get();

        SharedExpertFFNStage stage(params);
        EXPECT_EQ(
            stage.supportsLazyPrefillGraphCapturePreflight(),
            backend_supported)
            << backend_name << " must admit a one-row prefix suffix without "
                               "eager replay";
        EXPECT_EQ(
            stage.supportsPaddedPrefillGraphCapturePreflight(),
            backend_supported);
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif
#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, PrefillCapturableWhenScratchReady)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.input = input_.get();
    params.output = output_.get();

    SharedExpertFFNStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);
    stage.setScratchSeqLenForTesting(SEQ_LEN);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.isGraphCapturable())
        << "SharedExpertFFN should be capturable with sufficient scratch";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, PrefillRejectsInsufficientScratch)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.input = input_.get();
    params.output = output_.get();

    SharedExpertFFNStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);
    stage.setScratchSeqLenForTesting(SEQ_LEN - 1); // undersized

    EXPECT_FALSE(stage.isGraphCapturable())
        << "SharedExpertFFN should reject when scratch is undersized for prefill";
}

TEST_F(SharedExpertFFNPrefillGraphCapture, PrefillRejectsWithoutKernel)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.input = input_.get();
    params.output = output_.get();

    SharedExpertFFNStage stage(params);
    stage.setScratchSeqLenForTesting(SEQ_LEN);
    // moe_kernel_ left nullptr

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(SharedExpertFFNPrefillGraphCapture, PrefillRejectsOnCPU)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.input = input_.get();
    params.output = output_.get();

    SharedExpertFFNStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);
    stage.setScratchSeqLenForTesting(SEQ_LEN);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(SharedExpertFFNPrefillGraphCapture, GpuForcedVerifierSmallMUsesGroupedPrefillRoute)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    auto expect_backend = [&](DeviceId device, bool supported, const char *backend_name)
    {
        for (int seq_len : {2, 3, 4})
        {
            SharedExpertFFNStage::Params params;
            params.device_id = device;
            params.seq_len = seq_len;
            params.d_model = D_MODEL;
            params.intermediate = INTERMEDIATE;
            params.input = input_.get();
            params.gate_w = gate_w.get();
            params.up_w = up_w.get();
            params.down_w = down_w.get();
            params.output = output_.get();
            params.force_grouped_verifier_prefill_for_decode = true;
            params.required_router_q8_publication =
                std::make_shared<MoERouterQ8HiddenPublication>();

            SharedExpertFFNStage stage(params);
            EXPECT_EQ(stage.usesGroupedVerifierPrefillRouteForTesting(), supported)
                << backend_name << " all-position verifier shared expert M=" << seq_len
                << " grouped prefill route support mismatch";
        }
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

/**
 * @brief Standalone grouped verifier quantization remains graph-capturable.
 *
 * Deterministic execution disables router-Q8 reuse on CUDA and ROCm. The
 * shared verifier must keep the same grouped production route while owning its
 * input quantization instead of requiring a publication that cannot exist.
 */
TEST_F(SharedExpertFFNPrefillGraphCapture,
       GpuForcedVerifierDoesNotRequireRouterQ8ReuseForCapturePreflight)
{
    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    auto expect_backend = [&](DeviceId device, bool supported, const char *backend_name)
    {
        SharedExpertFFNStage::Params params;
        params.device_id = device;
        params.seq_len = 2;
        params.d_model = D_MODEL;
        params.intermediate = INTERMEDIATE;
        params.input = input_.get();
        params.gate_w = gate_w.get();
        params.up_w = up_w.get();
        params.down_w = down_w.get();
        params.output = output_.get();
        params.force_grouped_verifier_prefill_for_decode = true;

        SharedExpertFFNStage stage(params);
        EXPECT_EQ(stage.supportsPaddedPrefillGraphCapturePreflight(), supported)
            << backend_name << " standalone grouped-verifier quantization must "
                            "remain a capturable production path";
        EXPECT_FALSE(stage.requiresRouterQ8PublicationForTesting());
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, GpuNormalSmallMPrefillDoesNotForceGroupedVerifierRoute)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto expect_backend = [&](DeviceId device, const char *backend_name)
    {
        for (int seq_len : {2, 3, 4})
        {
            SharedExpertFFNStage::Params params;
            params.device_id = device;
            params.seq_len = seq_len;
            params.d_model = D_MODEL;
            params.intermediate = INTERMEDIATE;
            params.input = input_.get();
            params.output = output_.get();

            SharedExpertFFNStage stage(params);
            EXPECT_FALSE(stage.usesGroupedVerifierPrefillRouteForTesting())
                << "Normal " << backend_name << " shared-expert prefill M=" << seq_len
                << " should not enter the verifier-only grouped route without the explicit verifier flag";
        }
    };

    expect_backend(DeviceId::cuda(0), "CUDA");
    expect_backend(DeviceId::rocm(0), "ROCm");
}

TEST_F(SharedExpertFFNPrefillGraphCapture, GpuForcedDecodeReplayKeepsGroupedPrefillRoute)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    auto expect_backend = [&](DeviceId device, bool supported, const char *backend_name)
    {
        SharedExpertFFNStage::Params params;
        params.device_id = device;
        params.seq_len = 1;
        params.d_model = D_MODEL;
        params.intermediate = INTERMEDIATE;
        params.input = input_.get();
        params.gate_w = gate_w.get();
        params.up_w = up_w.get();
        params.down_w = down_w.get();
        params.output = output_.get();
        params.force_grouped_verifier_prefill_for_decode = true;
        params.required_router_q8_publication =
            std::make_shared<MoERouterQ8HiddenPublication>();

        SharedExpertFFNStage stage(params);
        EXPECT_EQ(stage.usesGroupedVerifierPrefillRouteForTesting(), supported)
            << backend_name << " forced single-row verifier replay should stay on grouped prefill when supported";
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, ForcedDecodeReplayCapturesAfterGroupedPrefillWarmup)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    auto expect_backend = [&](DeviceId device, bool supported, const char *backend_name)
    {
        SharedExpertFFNStage::Params params;
        params.device_id = device;
        params.seq_len = 1;
        params.d_model = D_MODEL;
        params.intermediate = INTERMEDIATE;
        params.input = input_.get();
        params.gate_w = gate_w.get();
        params.up_w = up_w.get();
        params.down_w = down_w.get();
        params.output = output_.get();
        params.force_grouped_verifier_prefill_for_decode = true;
        params.required_router_q8_publication =
            std::make_shared<MoERouterQ8HiddenPublication>();

        SharedExpertFFNStage stage(params);
        EXPECT_EQ(stage.supportsPaddedPrefillGraphCapturePreflight(), supported)
            << backend_name << " forced verifier replay should preflight as grouped prefill";
        EXPECT_FALSE(stage.isGraphCapturable())
            << backend_name << " cold forced verifier replay still needs warmup resources";

        stage.setMoEKernelForTesting(&stub_kernel_);
        stage.setScratchSeqLenForTesting(1);
        stage.setGroupedDecodeLaunchStatePreparedForTesting(false);
        EXPECT_EQ(stage.isGraphCapturable(), supported)
            << backend_name
            << " forced verifier replay must not depend on the normal grouped-decode warmed flag";
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, SessionResetPreservesForcedVerifierPrefillReadiness)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    auto expect_backend = [&](DeviceId device,
                              bool route_supported,
                              bool capture_supported,
                              const char *backend_name)
    {
        SharedExpertFFNStage::Params params;
        params.device_id = device;
        params.seq_len = 4;
        params.d_model = D_MODEL;
        params.intermediate = INTERMEDIATE;
        params.input = input_.get();
        params.gate_w = gate_w.get();
        params.up_w = up_w.get();
        params.down_w = down_w.get();
        params.output = output_.get();
        params.force_grouped_verifier_prefill_for_decode = true;
        params.required_router_q8_publication =
            std::make_shared<MoERouterQ8HiddenPublication>();

        SharedExpertFFNStage stage(params);
        EXPECT_EQ(stage.usesGroupedVerifierPrefillRouteForTesting(), route_supported)
            << backend_name << " forced verifier replay route support mismatch";

        stage.setMoEKernelForTesting(&stub_kernel_);
        stage.setScratchSeqLenForTesting(params.seq_len);
        stage.setGroupedDecodeLaunchStatePreparedForTesting(true);
        EXPECT_EQ(stage.isGraphCapturable(), capture_supported)
            << backend_name << " forced verifier replay should capture after grouped-prefill warmup";

        /**
         * Session reset clears grouped-decode pointer-table readiness, but the
         * forced verifier path is a grouped-prefill route. Its safety predicate
         * is the warmed scratch capacity plus MoE kernel binding, so resetting
         * decode state must not silently push verifier rows back to serial work.
         */
        stage.resetSessionState();
        EXPECT_EQ(stage.usesGroupedVerifierPrefillRouteForTesting(), route_supported)
            << backend_name << " reset must not change verifier route selection";
        EXPECT_EQ(stage.isGraphCapturable(), capture_supported)
            << backend_name << " reset must preserve warmed verifier-prefill capture readiness";
    };

    constexpr bool kGraphCaptureSupportedInThisBuild = true;

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, kGraphCaptureSupportedInThisBuild, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, false, "CUDA");
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, kGraphCaptureSupportedInThisBuild, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, false, "ROCm");
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, GpuNormalDecodeUsesWorkspaceBackedGroupedTableRoute)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    auto expect_backend = [&](DeviceId device, bool supported, const char *backend_name)
    {
        SharedExpertFFNStage::Params params;
        params.device_id = device;
        params.seq_len = 1;
        params.d_model = D_MODEL;
        params.intermediate = INTERMEDIATE;
        params.input = input_.get();
        params.gate_w = gate_w.get();
        params.up_w = up_w.get();
        params.down_w = down_w.get();
        params.output = output_.get();

        SharedExpertFFNStage stage(params);
        EXPECT_FALSE(stage.usesGroupedVerifierPrefillRouteForTesting())
            << backend_name << " normal shared-expert decode must not borrow the "
               "verifier-only grouped prefill route";
        EXPECT_EQ(stage.usesGroupedDecodeForTesting(), supported)
            << backend_name << " shared-expert normal decode grouped-table route mismatch";
        EXPECT_EQ(stage.supportsGraphCaptureAfterLaunchPreparation(), supported)
            << backend_name << " grouped decode should advertise one warmup pass before capture";
        EXPECT_FALSE(stage.isGraphCapturable())
            << backend_name << " grouped decode is not capturable until warmup has "
               "populated pointer arrays";

        stage.setMoEKernelForTesting(&stub_kernel_);
        stage.setScratchSeqLenForTesting(1);
        stage.setGroupedDecodeLaunchStatePreparedForTesting(true);
        EXPECT_EQ(stage.isGraphCapturable(), supported)
            << backend_name << " grouped decode should be capturable after successful warmup";
        stage.resetSessionStatePreservingCapturedReplay();
        EXPECT_EQ(stage.isGraphCapturable(), supported)
            << backend_name << " preserving captured replay must keep grouped decode tables warm";
        stage.resetSessionState();
        EXPECT_FALSE(stage.isGraphCapturable())
            << backend_name << " session reset clears backend pointer-table readiness";
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif

#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif

    const auto reqs = MoEWorkspaceBuffers::cudaMoE(4, D_MODEL, INTERMEDIATE, 256, 8);
    EXPECT_NE(reqs.find(MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_GATE_PTRS), nullptr);
    EXPECT_NE(reqs.find(MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_UP_PTRS), nullptr);
    EXPECT_NE(reqs.find(MoEWorkspaceBuffers::CUDA_DECODE_DOWN_GATE_PTRS), nullptr);
    EXPECT_NE(reqs.find(MoEWorkspaceBuffers::CUDA_DECODE_DOWN_UP_PTRS), nullptr);

    const auto rocm_reqs = MoEWorkspaceBuffers::rocmMoE(4, D_MODEL, INTERMEDIATE, 256, 8);
    EXPECT_NE(rocm_reqs.find(MoEWorkspaceBuffers::ROCM_DECODE_GATE_PTRS), nullptr);
    EXPECT_NE(rocm_reqs.find(MoEWorkspaceBuffers::ROCM_DECODE_UP_PTRS), nullptr);
    EXPECT_NE(rocm_reqs.find(MoEWorkspaceBuffers::ROCM_DECODE_GATE_OUTPUT_PTRS), nullptr);
    EXPECT_NE(rocm_reqs.find(MoEWorkspaceBuffers::ROCM_DECODE_UP_OUTPUT_PTRS), nullptr);
    EXPECT_NE(rocm_reqs.find(MoEWorkspaceBuffers::ROCM_DECODE_DOWN_DESCS), nullptr);
}

TEST_F(SharedExpertFFNPrefillGraphCapture, CudaMTPVerifierRowsCanBypassGroupedDecodeShortcut)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.input = input_.get();
    params.gate_w = gate_w.get();
    params.up_w = up_w.get();
    params.down_w = down_w.get();
    params.output = output_.get();
    /*
     * MTP phase-split sidecars verify one row at a time from the replicated
     * dense-decode weight set.  That row is allowed to publish into live state,
     * so the graph builder must be able to request the canonical
     * decode-equivalent verifier route even on CUDA builds where normal decode
     * uses the grouped shared-expert pointer-table shortcut.
     */
    params.disable_grouped_decode_shortcut = true;

    SharedExpertFFNStage stage(params);
    EXPECT_FALSE(stage.usesGroupedDecodeForTesting())
        << "MTP verifier rows must be able to bypass the normal CUDA grouped "
           "shared-expert decode shortcut and run through the serial-decode oracle.";
}

TEST_F(SharedExpertFFNPrefillGraphCapture, GpuForcedDecodeReplayUsesBackendGroupedCapability)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto gate_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto up_w = TestTensorFactory::createFP32({INTERMEDIATE, D_MODEL});
    auto down_w = TestTensorFactory::createFP32({D_MODEL, INTERMEDIATE});

    auto expect_backend = [&](DeviceId device, const char *backend_name)
    {
        SharedExpertFFNStage::Params params;
        params.device_id = device;
        params.seq_len = 1;
        params.d_model = D_MODEL;
        params.intermediate = INTERMEDIATE;
        params.input = input_.get();
        params.gate_w = gate_w.get();
        params.up_w = up_w.get();
        params.down_w = down_w.get();
        params.output = output_.get();
        params.force_grouped_verifier_prefill_for_decode = true;
        params.required_router_q8_publication =
            std::make_shared<MoERouterQ8HiddenPublication>();

        SharedExpertFFNStage stage(params);
        const bool backend_compiled =
            (device.is_cuda()
#if defined(HAVE_CUDA)
             && true
#else
             && false
#endif
            ) ||
            (device.is_rocm()
#if defined(HAVE_ROCM)
             && true
#else
             && false
#endif
            );
        EXPECT_EQ(stage.usesGroupedVerifierPrefillRouteForTesting(), backend_compiled)
            << backend_name
            << " verifier replay must use grouped prefill whenever the backend is compiled, "
               "without consulting a runtime capability-advertisement switch";
    };

    expect_backend(DeviceId::cuda(0), "CUDA");
    expect_backend(DeviceId::rocm(0), "ROCm");
}

// =========================================================================
// SharedExpertGateStage — Prefill Graph Capturability
// =========================================================================

class SharedExpertGatePrefillGraphCapture : public ::testing::Test
{
protected:
    static constexpr int D_MODEL = 64;
    static constexpr int SEQ_LEN = 16;

    std::unique_ptr<FP32Tensor> input_;
    std::unique_ptr<DeviceResidentFP32Tensor> gate_inp_;
    std::unique_ptr<FP32Tensor> shared_output_;
    StubMoEKernel stub_kernel_;
    int32_t active_rows_ = SEQ_LEN;

    void SetUp() override
    {
        input_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
        gate_inp_ = std::make_unique<DeviceResidentFP32Tensor>(
            std::vector<size_t>{1, D_MODEL});
        shared_output_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    }

    SharedExpertGateStage::Params makeValidPrefillParams() const
    {
        SharedExpertGateStage::Params params;
        params.device_id = DeviceId::rocm(0);
        params.seq_len = SEQ_LEN;
        params.d_model = D_MODEL;
        params.input = input_.get();
        params.gate_inp = gate_inp_.get();
        params.shared_output = shared_output_.get();
        params.active_row_count_device = &active_rows_;
        return params;
    }
};

TEST_F(SharedExpertGatePrefillGraphCapture, PrefillRequiresDeviceResidentGateWithKernel)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();

    SharedExpertGateStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_FALSE(stage.isGraphCapturable())
        << "A host-only shared-expert gate vector would force an H2D during "
        << "graph capture; warmup must make the effective gate tensor resident first.";

    gate_inp_->markResidentForGraphCaptureTest(params.device_id);
    EXPECT_TRUE(stage.isGraphCapturable())
        << "SharedExpertGate should be capturable after warmup has made the "
        << "gate vector resident on the stage device.";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

/**
 * @brief Grouped verifier shared gating prepares capture without replay state.
 *
 * The fixed-M verifier binds its persistent kernel and validates model-weight
 * residency before capture. Its active row count remains device-owned, so no
 * host replay metadata can upgrade or otherwise mutate the launch policy.
 */
TEST_F(SharedExpertGatePrefillGraphCapture, GroupedVerifierUsesCaptureOnlyLaunchPreparation)
{
    auto params = makeValidPrefillParams();
    SharedExpertGateStage stage(params);

    EXPECT_EQ(
        stage.graphLaunchPreparationPolicy(),
        GraphLaunchPreparationPolicy::CaptureOnly);
    EXPECT_EQ(stage.activeRowCountDeviceForTesting(), &active_rows_);
}

TEST_F(SharedExpertGatePrefillGraphCapture,
       PaddedPrefillRequiresDeviceOwnedRowCount)
{
    auto params = makeValidPrefillParams();
    params.active_row_count_device = nullptr;
    SharedExpertGateStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsLazyPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.supportsPaddedPrefillRealLengthContract());
}

TEST_F(SharedExpertGatePrefillGraphCapture, PrefillRejectsWithoutKernel)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();

    SharedExpertGateStage stage(params);
    // moe_kernel_ left nullptr

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Cold padded preflight should allow shared gate before kernel warmup";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable());
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation());
#else
    EXPECT_FALSE(stage.supportsGraphCaptureAfterLaunchPreparation());
#endif
}

TEST_F(SharedExpertGatePrefillGraphCapture, DecodePlansWarmupDependentCaptureWithoutKernel)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.input = input_.get();
    params.gate_inp = gate_inp_.get();
    params.shared_output = shared_output_.get();

    SharedExpertGateStage stage(params);
    // moe_kernel_ left nullptr until warmup execution.

    EXPECT_FALSE(stage.isGraphCapturable());
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation())
        << "Decode shared gate should be planned capturable before warmup";
#else
    EXPECT_FALSE(stage.supportsGraphCaptureAfterLaunchPreparation());
#endif
}

/**
 * @brief One-row prefix suffixes retain the device-owned prefill contract.
 */
TEST_F(SharedExpertGatePrefillGraphCapture,
       OneRowPrefixSuffixSupportsDeviceOwnedPrefillCapture)
{
    const auto expect_backend =
        [&](DeviceId device, bool backend_supported, const char *backend_name)
    {
        int32_t active_rows = 1;
        SharedExpertGateStage::Params params;
        params.device_id = device;
        params.seq_len = 1;
        params.d_model = D_MODEL;
        params.input = input_.get();
        params.gate_inp = gate_inp_.get();
        params.shared_output = shared_output_.get();
        params.active_row_count_device = &active_rows;

        SharedExpertGateStage stage(params);
        EXPECT_EQ(
            stage.supportsPaddedPrefillGraphCapturePreflight(),
            backend_supported)
            << backend_name << " must admit a one-row restored-prefix suffix";
    };

#if defined(HAVE_CUDA)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif
#if defined(HAVE_ROCM)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

TEST_F(SharedExpertGatePrefillGraphCapture, PrefillPreflightUsesBackendGroupedCapability)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    auto params = makeValidPrefillParams();

    SharedExpertGateStage stage(params);
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
}

TEST_F(SharedExpertGatePrefillGraphCapture, RejectsOnCPU)
{
    ScopedMoEGraphCaptureFlags flags(true, true);

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.input = input_.get();
    params.gate_inp = gate_inp_.get();
    params.shared_output = shared_output_.get();

    SharedExpertGateStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

// =========================================================================
// GDNRecurrenceStage — Prefill Graph Capturability
// =========================================================================

class GDNPrefillGraphCapture : public ::testing::Test
{
protected:
    static constexpr int SEQ_LEN = 16;
    static constexpr int N_HEADS = 4;
    static constexpr int D_K = 64;
    static constexpr int D_V = 128;

    std::unique_ptr<FP32Tensor> q_;
    std::unique_ptr<FP32Tensor> k_;
    std::unique_ptr<FP32Tensor> v_;
    std::unique_ptr<FP32Tensor> alpha_;
    std::unique_ptr<FP32Tensor> beta_;
    std::unique_ptr<FP32Tensor> a_log_;
    std::unique_ptr<FP32Tensor> dt_bias_;
    std::unique_ptr<FP32Tensor> output_;
    std::vector<float> state_;
    int32_t request_seq_len_device_ = SEQ_LEN;

    void SetUp() override
    {
        q_ = TestTensorFactory::createFP32({SEQ_LEN, N_HEADS * D_K});
        k_ = TestTensorFactory::createFP32({SEQ_LEN, N_HEADS * D_K});
        v_ = TestTensorFactory::createFP32({SEQ_LEN, N_HEADS * D_V});
        alpha_ = TestTensorFactory::createFP32({SEQ_LEN, N_HEADS});
        beta_ = TestTensorFactory::createFP32({SEQ_LEN, N_HEADS});
        a_log_ = TestTensorFactory::createFP32({1, N_HEADS});
        dt_bias_ = TestTensorFactory::createFP32({1, N_HEADS});
        output_ = TestTensorFactory::createFP32({SEQ_LEN, N_HEADS * D_V});
        state_.resize(static_cast<size_t>(N_HEADS * D_K * D_V), 0.0f);
    }

    GDNRecurrenceStage::Params makeValidPrefillParams(ITensorGatedDeltaNet *kernel)
    {
        GDNRecurrenceStage::Params p;
        p.device_id = DeviceId::rocm(0);
        p.seq_len = SEQ_LEN;
        p.request_count = 1;
        p.request_seq_len = SEQ_LEN;
        p.request_seq_lens_device = &request_seq_len_device_;
        p.n_heads = N_HEADS;
        p.d_k = D_K;
        p.d_v = D_V;
        p.Q = q_.get();
        p.K = k_.get();
        p.V = v_.get();
        p.alpha = alpha_.get();
        p.beta = beta_.get();
        p.A_log = a_log_.get();
        p.dt_bias = dt_bias_.get();
        p.output = output_.get();
        p.recurrence_state = state_.data();
        p.kernel = kernel;
        return p;
    }
};

TEST_F(GDNPrefillGraphCapture, PrefillCapturableWhenGPUStateReady)
{
    const int required_state_size = N_HEADS * D_K * D_V;
    StubGDNKernel ready_kernel(true, required_state_size);
    auto params = makeValidPrefillParams(&ready_kernel);
    GDNRecurrenceStage stage(params);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.isGraphCapturable())
        << "GDN prefill should be capturable when GPU state is ready";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(GDNPrefillGraphCapture, ColdPaddedPrefillPreflightAllowsCompiledGPUWhenRealLengthKernelReadyButStateIsNot)
{
    StubGDNKernel not_ready_kernel(false, 0, true);
    auto params = makeValidPrefillParams(&not_ready_kernel);
    GDNRecurrenceStage stage(params);

#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsLazyPrefillGraphCapturePreflight())
        << "Exact-shape cold preflight should validate ROCm GDN support without requiring warmed GPU state";
    EXPECT_TRUE(stage.supportsPaddedPrefillRealLengthContract());
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Cold padded preflight should validate ROCm GDN support without requiring warmed GPU state";
#else
    EXPECT_FALSE(stage.supportsLazyPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Actual graph capture must still wait for warmup to allocate GPU recurrence state";

#ifdef HAVE_CUDA
    auto cuda_params = makeValidPrefillParams(&not_ready_kernel);
    cuda_params.device_id = DeviceId::cuda(0);
    GDNRecurrenceStage cuda_stage(cuda_params);
    EXPECT_TRUE(cuda_stage.supportsLazyPrefillGraphCapturePreflight())
        << "Exact-shape cold preflight should validate CUDA GDN support without requiring warmed GPU state";
    EXPECT_TRUE(cuda_stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Cold padded preflight should validate CUDA GDN support without requiring warmed GPU state";
    EXPECT_FALSE(cuda_stage.isGraphCapturable())
        << "Actual CUDA graph capture must still wait for warmup to allocate GPU recurrence state";
#endif
}

TEST_F(GDNPrefillGraphCapture, ColdPaddedPrefillPreflightRejectsKernelWithoutRealLengthContract)
{
    const int required_state_size = N_HEADS * D_K * D_V;
    StubGDNKernel unsupported_kernel(true, required_state_size, false);
    auto params = makeValidPrefillParams(&unsupported_kernel);
    GDNRecurrenceStage stage(params);

    EXPECT_FALSE(stage.supportsPaddedPrefillRealLengthContract());
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Padded bucket GDN preflight must reject kernels that cannot commit only the real prompt prefix";
}

TEST_F(GDNPrefillGraphCapture, ColdPaddedPrefillPreflightRejectsNullKernel)
{
    auto params = makeValidPrefillParams(nullptr);
    GDNRecurrenceStage stage(params);

    EXPECT_FALSE(stage.supportsPaddedPrefillRealLengthContract());
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "GDN padded preflight needs an explicit backend real-length contract";
    EXPECT_FALSE(stage.isGraphCapturable())
        << "GDN prefill should reject when kernel is null";
}

TEST_F(GDNPrefillGraphCapture, ColdPaddedPrefillPreflightRejectsCPU)
{
    const int required_state_size = N_HEADS * D_K * D_V;
    StubGDNKernel ready_kernel(true, required_state_size, true);

    auto cpu_params = makeValidPrefillParams(&ready_kernel);
    cpu_params.device_id = DeviceId::cpu();
    GDNRecurrenceStage cpu_stage(cpu_params);
    EXPECT_FALSE(cpu_stage.supportsPaddedPrefillGraphCapturePreflight())
        << "CPU GDN real-length execution does not imply GPU graph prefill support";
}

TEST_F(GDNPrefillGraphCapture, PrefillRejectsWhenGPUStateNotReady)
{
    StubGDNKernel not_ready_kernel(false, 0);
    auto params = makeValidPrefillParams(&not_ready_kernel);
    GDNRecurrenceStage stage(params);

    EXPECT_FALSE(stage.isGraphCapturable())
        << "GDN prefill should not be capturable before GPU state warmup";
}

TEST_F(GDNPrefillGraphCapture, PrefillRejectsWhenGPUStateSizeMismatch)
{
    // State allocated with wrong size
    StubGDNKernel wrong_size_kernel(true, N_HEADS * D_K * D_V - 1);
    auto params = makeValidPrefillParams(&wrong_size_kernel);
    GDNRecurrenceStage stage(params);

    EXPECT_FALSE(stage.isGraphCapturable())
        << "GDN prefill should reject when GPU state size doesn't match";
}

TEST_F(GDNPrefillGraphCapture, PrefillRejectsWithNullKernel)
{
    auto params = makeValidPrefillParams(nullptr);
    GDNRecurrenceStage stage(params);

    EXPECT_FALSE(stage.isGraphCapturable())
        << "GDN prefill should reject when kernel is null";
}

TEST_F(GDNPrefillGraphCapture, PrefillRejectsOnCPU)
{
    const int required_state_size = N_HEADS * D_K * D_V;
    StubGDNKernel ready_kernel(true, required_state_size);
    auto params = makeValidPrefillParams(&ready_kernel);
    params.device_id = DeviceId::cpu();
    GDNRecurrenceStage stage(params);

    EXPECT_FALSE(stage.isGraphCapturable())
        << "GDN prefill should never be capturable on CPU";
}

TEST_F(GDNPrefillGraphCapture, CUDAPrefillCapturableWhenGPUStateReady)
{
#ifdef HAVE_CUDA
    const int required_state_size = N_HEADS * D_K * D_V;
    StubGDNKernel ready_kernel(true, required_state_size);
    auto params = makeValidPrefillParams(&ready_kernel);
    params.device_id = DeviceId::cuda(0);
    GDNRecurrenceStage stage(params);

    EXPECT_TRUE(stage.isGraphCapturable())
        << "CUDA GDN prefill graph capture should be enabled once recurrence state is ready.";
#else
    GTEST_SKIP() << "CUDA backend not compiled";
#endif
}

TEST_F(GDNPrefillGraphCapture, DecodeAlwaysCapturableRegardlessOfState)
{
    // Decode (seq_len=1) has always been capturable on ROCm (and in snapshot builds)
    StubGDNKernel not_ready_kernel(false, 0);
    auto params = makeValidPrefillParams(&not_ready_kernel);
    params.seq_len = 1;
    GDNRecurrenceStage stage(params);

#if defined(HAVE_ROCM)
    // Decode is capturable in both snapshot and non-snapshot ROCm builds
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Decode GDN should be capturable regardless of GPU state";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(GDNPrefillGraphCapture, LiveStateAllGatherCapturableRequiresRawAllgatherGraphCaptureSupport)
{
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    MockLocalTPContext tp_ctx;
#if defined(HAVE_CUDA)
    const DeviceId device = DeviceId::cuda(0);
    tp_ctx.setDevices({GlobalDeviceAddress::cuda(0, 0), GlobalDeviceAddress::cuda(1, 0)});
    tp_ctx.setBackend(CollectiveBackendType::NCCL);
#else
    const DeviceId device = DeviceId::rocm(0);
    tp_ctx.setDevices({GlobalDeviceAddress::rocm(0, 0), GlobalDeviceAddress::rocm(1, 0)});
    tp_ctx.setBackend(CollectiveBackendType::RCCL);
#endif
    tp_ctx.setRawAllgatherGraphCaptureSupported(true);

    StubGDNKernel recurrence_kernel(true, 8);
    GDNLiveStateAllGatherStage::Params params;
    params.device_id = device;
    params.tp_ctx = &tp_ctx;
    params.recurrence_kernel = &recurrence_kernel;
    params.layer_idx = 0;
    params.tp_device_idx = 0;
    params.local_conv_state_floats = 0;
    params.full_conv_state_floats = 0;
    params.local_recurrence_state_floats = 4;
    params.full_recurrence_state_floats = 8;
    params.stage_name = "gdn_live_state_allgather";

    GDNLiveStateAllGatherStage supported(params);
    EXPECT_TRUE(supported.isGraphCapturable())
        << "GDN live-state handoff should be capturable when LocalTP can record raw allgather on the capture stream.";

    tp_ctx.setRawAllgatherGraphCaptureSupported(false);
    GDNLiveStateAllGatherStage unsupported(params);
    EXPECT_FALSE(unsupported.isGraphCapturable())
        << "GDN live-state handoff must not use prefill graph capture without a graph-capturable raw allgather.";
#else
    GTEST_SKIP() << "GPU backend not compiled";
#endif
}
