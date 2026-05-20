/**
 * @file Test__PrefillGraphCapturability.cpp
 * @brief Phase 1 acceptance gate tests for prefill HIP graph capturability predicates.
 *
 * Tests that isGraphCapturable() returns true for prefill (seq_len > 1) on ROCm
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
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "kernels/IMoEKernel.h"
#include "mocks/MockComputeStage.h"
#include "utils/TestTensorFactory.h"
#include "utils/DebugEnv.h"

#include <cmath>
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
    explicit StubGDNKernel(bool state_ready, int state_size = 0)
        : state_ready_(state_ready), state_size_(state_size) {}

    bool isGPUStateReady(int required_state_size) const override
    {
        return state_ready_ && (state_size_ == required_state_size);
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
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Prefill routing should be capturable on ROCm with valid buffers and kernel";
#else
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

    EXPECT_FALSE(stage.isGraphCapturable())
        << "Prefill routing should not be capturable without cached kernel";
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
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Fixed-topology grouped prefill should be capturable with all engines ready";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(MoEExpertPrefillGraphCapture, RejectsWithoutKernel)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto params = makeValidPrefillParams();
    MoEExpertComputeStage stage(params);
    // moe_kernel_ left nullptr

    EXPECT_FALSE(stage.isGraphCapturable())
        << "Should not be capturable without MoE kernel";
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

TEST_F(SharedExpertFFNPrefillGraphCapture, DecodeAlwaysCapturableOnRocmWithoutScratch)
{
    ScopedRocmMoEFlags flags(true, true, true);

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = 1; // decode
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.input = input_.get();
    params.output = output_.get();

    SharedExpertFFNStage stage(params);
    // No scratch needed for decode, no kernel needed for decode path

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.isGraphCapturable())
        << "Decode SharedExpertFFN should always be capturable on ROCm";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
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

    EXPECT_FALSE(stage.isGraphCapturable());
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

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.isGraphCapturable())
        << "GDN prefill should be capturable when GPU state is ready";
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
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
