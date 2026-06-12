/**
 * @file Test__PrefillGraphCapturability.cpp
 * @brief Phase 1 acceptance gate tests for prefill GPU graph capturability predicates.
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
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "kernels/IMoEKernel.h"
#include "mocks/MockComputeStage.h"
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

    class ScopedRocmMoEFlags
    {
    public:
        ScopedRocmMoEFlags(bool grouped_decode, bool device_routed_decode, bool grouped_prefill)
            : old_grouped_(mutableDebugEnv().rocm.moe_grouped_decode),
              old_device_routed_(mutableDebugEnv().rocm.moe_device_routed_decode),
              old_prefill_(mutableDebugEnv().rocm.moe_grouped_prefill)
        {
            mutableDebugEnv().rocm.moe_grouped_decode = grouped_decode;
            mutableDebugEnv().rocm.moe_device_routed_decode = device_routed_decode;
            mutableDebugEnv().rocm.moe_grouped_prefill = grouped_prefill;
        }

        ~ScopedRocmMoEFlags()
        {
            mutableDebugEnv().rocm.moe_grouped_decode = old_grouped_;
            mutableDebugEnv().rocm.moe_device_routed_decode = old_device_routed_;
            mutableDebugEnv().rocm.moe_grouped_prefill = old_prefill_;
        }

    private:
        bool old_grouped_;
        bool old_device_routed_;
        bool old_prefill_;
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

    // =========================================================================
    // Minimal stub IMoEKernel (no actual compute, just satisfies the interface)
    // =========================================================================

    class StubMoEKernel : public IMoEKernel
    {
    public:
        bool supports_device(int) const override { return true; }
        bool route(const float *, const float *, int, int, int, int, bool,
                   MoERoutingResult &) override
        {
            return false;
        }
        void gatherTokenBatch(const float *, float *, const int *, int, int) override {}
        void scatterAddWeighted(float *, const float *, const int *, const float *,
                                int, int) override {}
        void sharedExpertGate(const float *, const float *, float *, int, int) override {}
        void swiGLU(float *, const float *, int) override {}
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

        bool chunk_forward(
            const float *, const float *, const float *,
            const float *, const float *,
            const float *, const float *,
            float *, float *,
            int, int, int, int, int, bool) override
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
        return p;
    }
};

TEST_F(MoERoutingPrefillGraphCapture, PrefillCapturableWhenAllConditionsMet)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Prefill routing should be capturable on ROCm with valid buffers and kernel";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Prefill routing should not be capturable without ROCm or in snapshot builds";
#endif
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithoutKernel)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    MoERoutingStage stage(params);
    // moe_kernel_ left as nullptr (default)

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Cold padded preflight should allow routing before kernel warmup";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Prefill routing should not be capturable without cached kernel";
#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsWarmupDependentGraphCapture());
#else
    EXPECT_FALSE(stage.supportsWarmupDependentGraphCapture());
#endif
}

TEST_F(MoERoutingPrefillGraphCapture, CudaForcedVerifierReplaySeqLenOneUsesPrefillCaptureContract)
{
    ScopedRocmMoEFlags flags(true, true, true);

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
#if defined(HAVE_CUDA) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(cold_stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(cold_stage.supportsWarmupDependentGraphCapture());
    EXPECT_FALSE(cold_stage.isGraphCapturable())
        << "Forced seq_len=1 verifier replay must wait for the prefill router kernel warmup";

    cold_stage.setMoEKernelForTesting(&stub_kernel_);
    EXPECT_TRUE(cold_stage.isGraphCapturable())
        << "Forced seq_len=1 verifier replay should use the prefill routing capture contract";
#else
    EXPECT_FALSE(cold_stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(cold_stage.supportsWarmupDependentGraphCapture());
    EXPECT_FALSE(cold_stage.isGraphCapturable());
#endif
}

TEST_F(MoERoutingPrefillGraphCapture, CudaForcedVerifierReplayDoesNotFallBackToDecodeCapture)
{
    ScopedRocmMoEFlags flags(true, true, false);

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
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.supportsWarmupDependentGraphCapture());
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Forced verifier replay must hard-require grouped prefill instead of using decode capture";
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsOnCPU)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    params.device_id = DeviceId::cpu();
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWhenGroupedPrefillDisabled)
{
    ScopedRocmMoEFlags flags(true, true, false); // grouped_prefill = false

    auto params = makeValidPrefillParams();
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Prefill routing should not be capturable when moe_grouped_prefill is disabled";
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithNullInput)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    params.input = nullptr;
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithNullGateWeights)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    params.gate_weights = nullptr;
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithNullOutputIndices)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    params.output_indices = nullptr;
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsWithNullOutputWeights)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    params.output_weights = nullptr;
    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingPrefillGraphCapture, PrefillRejectsInvalidTopK)
{
    ScopedRocmMoEFlags flags(true, true, true);

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

TEST_F(MoERoutingPrefillGraphCapture, NeedsOnGraphReplayedForPrefill)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 1;
    histogram_config.num_experts = NUM_EXPERTS;
    histogram_config.top_k = TOP_K;
    DecodeExpertHistogram histogram(histogram_config);
    params.decode_histogram = &histogram;

    MoERoutingStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.isGraphCapturable());
    EXPECT_TRUE(stage.needsOnGraphReplayed());
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

    std::unique_ptr<FP32Tensor> input_;
    std::unique_ptr<FP32Tensor> output_;
    std::unique_ptr<FP32Tensor> routing_indices_;
    std::unique_ptr<FP32Tensor> routing_weights_;
    StubMoEKernel stub_kernel_;
    std::vector<StubGemmEngine> stub_gemms_;
    std::vector<ITensorGemm *> gate_gemm_ptrs_;
    std::vector<ITensorGemm *> up_gemm_ptrs_;
    std::vector<ITensorGemm *> down_gemm_ptrs_;

    void SetUp() override
    {
        input_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
        output_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
        routing_indices_ = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
        routing_weights_ = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

        // Create stub GEMM engines for all experts (gate, up, down × num_experts)
        stub_gemms_.resize(static_cast<size_t>(NUM_EXPERTS * 3));
        gate_gemm_ptrs_.resize(static_cast<size_t>(NUM_EXPERTS));
        up_gemm_ptrs_.resize(static_cast<size_t>(NUM_EXPERTS));
        down_gemm_ptrs_.resize(static_cast<size_t>(NUM_EXPERTS));

        for (int e = 0; e < NUM_EXPERTS; ++e)
        {
            gate_gemm_ptrs_[static_cast<size_t>(e)] = &stub_gemms_[static_cast<size_t>(e * 3 + 0)];
            up_gemm_ptrs_[static_cast<size_t>(e)] = &stub_gemms_[static_cast<size_t>(e * 3 + 1)];
            down_gemm_ptrs_[static_cast<size_t>(e)] = &stub_gemms_[static_cast<size_t>(e * 3 + 2)];
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
        p.prepared_gate_gemm = gate_gemm_ptrs_;
        p.prepared_up_gemm = up_gemm_ptrs_;
        p.prepared_down_gemm = down_gemm_ptrs_;
        // No expert_mask → all enabled
        // No replicas
        return p;
    }
};

TEST_F(MoEExpertPrefillGraphCapture, FixedTopologyCapturableWhenReady)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Fixed-topology grouped prefill should be capturable with all engines ready";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWithoutKernel)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    MoEExpertComputeStage stage(params);
    // moe_kernel_ left nullptr

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Cold padded preflight should allow fixed-topology MoE before kernel warmup";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Should not be capturable without MoE kernel";
#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsWarmupDependentGraphCapture());
#else
    EXPECT_FALSE(stage.supportsWarmupDependentGraphCapture());
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsOnCPU)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    params.device_id = DeviceId::cpu();
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWhenGroupedPrefillDisabled)
{
    ScopedRocmMoEFlags flags(true, true, false); // grouped_prefill = false

    auto params = makeValidPrefillParams();
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWithPartialExpertOwnership)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    params.local_expert_count = NUM_EXPERTS - 1; // not full
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable())
        << "Should reject when not all experts are locally owned";
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWithExpertMaskHavingDisabled)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    params.expert_mask = {true, true, false, true}; // one disabled
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable())
        << "Should reject when expert mask has disabled entries";
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWithReplicas)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    params.replica_set.num_replicated = 1;
    MoEExpertComputeStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

    EXPECT_FALSE(stage.isGraphCapturable())
        << "Should reject when replicas are active";
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWithMissingGemmEngines)
{
    ScopedRocmMoEFlags flags(true, true, true);

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
    ScopedRocmMoEFlags flags(true, true, true);

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
    ScopedRocmMoEFlags flags(true, true, true);

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

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable());
#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsWarmupDependentGraphCapture());
#else
    EXPECT_FALSE(stage.supportsWarmupDependentGraphCapture());
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, PrefillPreflightSupportRejectsDisabledGroupedPrefill)
{
    ScopedRocmMoEFlags flags(true, true, false);

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
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
}

TEST_F(SharedExpertFFNPrefillGraphCapture, PrefillCapturableWhenScratchReady)
{
    ScopedRocmMoEFlags flags(true, true, true);

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

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.isGraphCapturable())
        << "SharedExpertFFN should be capturable with sufficient scratch";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, PrefillRejectsInsufficientScratch)
{
    ScopedRocmMoEFlags flags(true, true, true);

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
    ScopedRocmMoEFlags flags(true, true, true);

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
    ScopedRocmMoEFlags flags(true, true, true);

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

TEST_F(SharedExpertFFNPrefillGraphCapture, RocmDecodeCapturableAfterWarmupWhenGroupedRouteIsDisabled)
{
    ScopedRocmMoEFlags flags(true, true, true);
    const bool old_shared_grouped = mutableDebugEnv().rocm.shared_expert_grouped_decode;
    mutableDebugEnv().rocm.shared_expert_grouped_decode = false;

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = 1; // decode
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.input = input_.get();
    params.output = output_.get();

    SharedExpertFFNStage stage(params);

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Cold decode still waits for the normal warmup pass to allocate scratch.";
    stage.setScratchSeqLenForTesting(1);
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Non-grouped decode is capturable once warmup has allocated scratch.";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif

    mutableDebugEnv().rocm.shared_expert_grouped_decode = old_shared_grouped;
}

TEST_F(SharedExpertFFNPrefillGraphCapture, GpuForcedVerifierSmallMUsesGroupedPrefillRoute)
{
    ScopedRocmMoEFlags flags(true, true, true);

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

TEST_F(SharedExpertFFNPrefillGraphCapture, GpuNormalSmallMPrefillDoesNotForceGroupedVerifierRoute)
{
    ScopedRocmMoEFlags flags(true, true, true);

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
    ScopedRocmMoEFlags flags(true, true, true);

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
    ScopedRocmMoEFlags flags(true, true, true);

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

        SharedExpertFFNStage stage(params);
        EXPECT_EQ(stage.supportsPaddedPrefillGraphCapturePreflight(), supported)
            << backend_name << " forced verifier replay should preflight as grouped prefill";
        EXPECT_FALSE(stage.isGraphCapturable())
            << backend_name << " cold forced verifier replay still needs warmup resources";

        stage.setMoEKernelForTesting(&stub_kernel_);
        stage.setScratchSeqLenForTesting(1);
        stage.setGroupedDecodeWarmedForTesting(false);
        EXPECT_EQ(stage.isGraphCapturable(), supported)
            << backend_name
            << " forced verifier replay must not depend on the normal grouped-decode warmed flag";
    };

#if defined(HAVE_CUDA) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    expect_backend(DeviceId::cuda(0), true, "CUDA");
#else
    expect_backend(DeviceId::cuda(0), false, "CUDA");
#endif

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    expect_backend(DeviceId::rocm(0), true, "ROCm");
#else
    expect_backend(DeviceId::rocm(0), false, "ROCm");
#endif
}

TEST_F(SharedExpertFFNPrefillGraphCapture, CudaNormalDecodeUsesWorkspaceBackedGroupedTableRoute)
{
    ScopedRocmMoEFlags flags(true, true, true);

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

    SharedExpertFFNStage stage(params);
    EXPECT_FALSE(stage.usesGroupedVerifierPrefillRouteForTesting())
        << "Normal CUDA shared-expert decode must not borrow the verifier-only grouped prefill route";
#if defined(HAVE_CUDA) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.usesGroupedDecodeForTesting())
        << "CUDA shared-expert normal decode should use the grouped table route once its "
           "pointer arrays are declared in graph-owned workspace.";
    EXPECT_TRUE(stage.supportsWarmupDependentGraphCapture())
        << "Cold grouped decode must execute one warmup pass to upload runtime pointer arrays";
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Grouped decode is not capturable until the warmup pass has populated pointer arrays.";
    stage.setMoEKernelForTesting(&stub_kernel_);
    stage.setScratchSeqLenForTesting(1);
    stage.setGroupedDecodeWarmedForTesting(true);
    EXPECT_TRUE(stage.isGraphCapturable())
        << "After successful warmup, grouped decode is safe for graph capture.";
#else
    (void)stage;
#endif

    const auto reqs = MoEWorkspaceBuffers::cudaMoE(4, D_MODEL, INTERMEDIATE, 256, 8);
    EXPECT_NE(reqs.find(MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_GATE_PTRS), nullptr);
    EXPECT_NE(reqs.find(MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_UP_PTRS), nullptr);
    EXPECT_NE(reqs.find(MoEWorkspaceBuffers::CUDA_DECODE_DOWN_GATE_PTRS), nullptr);
    EXPECT_NE(reqs.find(MoEWorkspaceBuffers::CUDA_DECODE_DOWN_UP_PTRS), nullptr);
}

TEST_F(SharedExpertFFNPrefillGraphCapture, GpuForcedDecodeReplayRequiresGroupedPrefillEnabled)
{
    ScopedRocmMoEFlags flags(true, true, false);

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

        SharedExpertFFNStage stage(params);
        EXPECT_FALSE(stage.usesGroupedVerifierPrefillRouteForTesting())
            << "No " << backend_name
            << " verifier replay should silently enter grouped prefill when grouped prefill is disabled";
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
    std::unique_ptr<FP32Tensor> gate_inp_;
    std::unique_ptr<FP32Tensor> shared_output_;
    StubMoEKernel stub_kernel_;

    void SetUp() override
    {
        input_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
        gate_inp_ = TestTensorFactory::createFP32({1, D_MODEL});
        shared_output_ = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    }
};

TEST_F(SharedExpertGatePrefillGraphCapture, PrefillCapturableWithKernel)
{
    ScopedRocmMoEFlags flags(true, true, true);

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.input = input_.get();
    params.gate_inp = gate_inp_.get();
    params.shared_output = shared_output_.get();

    SharedExpertGateStage stage(params);
    stage.setMoEKernelForTesting(&stub_kernel_);

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.isGraphCapturable())
        << "SharedExpertGate should be capturable for prefill on ROCm with kernel";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(SharedExpertGatePrefillGraphCapture, PrefillRejectsWithoutKernel)
{
    ScopedRocmMoEFlags flags(true, true, true);

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.input = input_.get();
    params.gate_inp = gate_inp_.get();
    params.shared_output = shared_output_.get();

    SharedExpertGateStage stage(params);
    // moe_kernel_ left nullptr

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Cold padded preflight should allow shared gate before kernel warmup";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable());
#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsWarmupDependentGraphCapture());
#else
    EXPECT_FALSE(stage.supportsWarmupDependentGraphCapture());
#endif
}

TEST_F(SharedExpertGatePrefillGraphCapture, DecodePlansWarmupDependentCaptureWithoutKernel)
{
    ScopedRocmMoEFlags flags(true, true, true);

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
#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.supportsWarmupDependentGraphCapture())
        << "Decode shared gate should be planned capturable before warmup";
#else
    EXPECT_FALSE(stage.supportsWarmupDependentGraphCapture());
#endif
}

TEST_F(SharedExpertGatePrefillGraphCapture, PrefillPreflightSupportRejectsDisabledGroupedPrefill)
{
    ScopedRocmMoEFlags flags(true, true, false);

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.input = input_.get();
    params.gate_inp = gate_inp_.get();
    params.shared_output = shared_output_.get();

    SharedExpertGateStage stage(params);
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
}

TEST_F(SharedExpertGatePrefillGraphCapture, RejectsOnCPU)
{
    ScopedRocmMoEFlags flags(true, true, true);

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
    EXPECT_TRUE(stage.supportsPaddedPrefillRealLengthContract());
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight())
        << "Cold padded preflight should validate ROCm GDN support without requiring warmed GPU state";
#else
    EXPECT_FALSE(stage.supportsPaddedPrefillGraphCapturePreflight());
#endif
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Actual graph capture must still wait for warmup to allocate GPU recurrence state";

#ifdef HAVE_CUDA
    auto cuda_params = makeValidPrefillParams(&not_ready_kernel);
    cuda_params.device_id = DeviceId::cuda(0);
    GDNRecurrenceStage cuda_stage(cuda_params);
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
