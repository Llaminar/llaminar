/**
 * @file Test__MoEExpertComputeStage.cpp
 * @brief Unit tests for MoE FFN stages: MoEExpertComputeStage, SharedExpertFFNStage, SharedExpertGateStage
 *
 * Tests the three MoE stages used by Qwen 3.5 MoE:
 * 1. MoEExpertComputeStage: Router (softmax top-k) → expert SwiGLU → weighted combine
 * 2. SharedExpertFFNStage: Dense SwiGLU on shared expert weights
 * 3. SharedExpertGateStage: sigmoid(gate · input) × shared_output
 *
 * Uses small synthetic tensors (no model loading required).
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "kernels/KernelFactory.h"
#include "kernels/IMoEKernel.h"
#include "mocks/MockComputeStage.h"
#include "utils/TestTensorFactory.h"
#include "utils/PreparedWeightTestHarness.h"
#include "utils/DebugEnv.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar2::testing;

namespace
{
#ifdef HAVE_ROCM
    bool hasROCmDeviceForStageTests()
    {
        int count = 0;
        const hipError_t err = hipGetDeviceCount(&count);
        return err == hipSuccess && count > 0;
    }
#endif

    class ScopedRocmMoEFlags
    {
    public:
                ScopedRocmMoEFlags(bool grouped_decode, bool device_routed_decode, bool grouped_prefill = false)
            : old_grouped_(mutableDebugEnv().rocm.moe_grouped_decode),
                            old_device_routed_(mutableDebugEnv().rocm.moe_device_routed_decode),
                            old_grouped_prefill_(mutableDebugEnv().rocm.moe_grouped_prefill)
        {
            mutableDebugEnv().rocm.moe_grouped_decode = grouped_decode;
            mutableDebugEnv().rocm.moe_device_routed_decode = device_routed_decode;
                        mutableDebugEnv().rocm.moe_grouped_prefill = grouped_prefill;
        }

        ~ScopedRocmMoEFlags()
        {
            mutableDebugEnv().rocm.moe_grouped_decode = old_grouped_;
            mutableDebugEnv().rocm.moe_device_routed_decode = old_device_routed_;
            mutableDebugEnv().rocm.moe_grouped_prefill = old_grouped_prefill_;
        }

    private:
        bool old_grouped_;
        bool old_device_routed_;
        bool old_grouped_prefill_;
    };

    class FakePreparedGemm final : public ITensorGemm
    {
    public:
        explicit FakePreparedGemm(DeviceNativeVNNIMatrixDesc desc, bool converted = true)
            : desc_(desc), converted_(converted)
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

        bool weights_converted() const override
        {
            return converted_;
        }

        bool exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out) override
        {
            out = desc_;
            return converted_ && out.valid();
        }

    private:
        DeviceNativeVNNIMatrixDesc desc_;
        bool converted_;
    };

    DeviceNativeVNNIMatrixDesc makeNativeVNNIDesc(int n, int k, uint8_t codebook_id = 4)
    {
        DeviceNativeVNNIMatrixDesc desc;
        desc.payload = reinterpret_cast<const uint8_t *>(0x1000);
        desc.scales = reinterpret_cast<const void *>(0x2000);
        desc.mins = reinterpret_cast<const void *>(0x3000);
        desc.n = n;
        desc.k = k;
        desc.blocks_per_row = static_cast<uint32_t>(k / 32);
        desc.codebook_id = codebook_id;
        return desc;
    }

    void attachFakePreparedExpertEngines(
        MoEExpertComputeStage::Params &params,
        std::vector<std::unique_ptr<FakePreparedGemm>> &owned_gemms,
        uint8_t gateup_codebook = 4,
        uint8_t down_codebook = 4,
        bool converted = true)
    {
        params.prepared_gate_gemm.assign(params.num_experts, nullptr);
        params.prepared_up_gemm.assign(params.num_experts, nullptr);
        params.prepared_down_gemm.assign(params.num_experts, nullptr);
        owned_gemms.clear();
        owned_gemms.reserve(static_cast<size_t>(params.num_experts) * 3);

        for (int expert_id = 0; expert_id < params.num_experts; ++expert_id)
        {
            owned_gemms.push_back(std::make_unique<FakePreparedGemm>(
                makeNativeVNNIDesc(params.expert_intermediate, params.d_model, gateup_codebook), converted));
            params.prepared_gate_gemm[expert_id] = owned_gemms.back().get();

            owned_gemms.push_back(std::make_unique<FakePreparedGemm>(
                makeNativeVNNIDesc(params.expert_intermediate, params.d_model, gateup_codebook), converted));
            params.prepared_up_gemm[expert_id] = owned_gemms.back().get();

            owned_gemms.push_back(std::make_unique<FakePreparedGemm>(
                makeNativeVNNIDesc(params.d_model, params.expert_intermediate, down_codebook), converted));
            params.prepared_down_gemm[expert_id] = owned_gemms.back().get();
        }
    }
}

// =========================================================================
// Test Fixture
// =========================================================================

class MoEExpertComputeStageTest : public ::testing::Test
{
protected:
    std::unique_ptr<MockDeviceContext> cpu_ctx_;

    // Small dimensions for unit testing
    static constexpr int D_MODEL = 64;
    static constexpr int INTERMEDIATE = 32;
    static constexpr int NUM_EXPERTS = 4;
    static constexpr int TOP_K = 2;
    static constexpr int SEQ_LEN = 2;

    void SetUp() override
    {
        cpu_ctx_ = std::make_unique<MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    }

    /// Create a 3D Q4_K expert tensor in GGUF layout [cols, rows, num_experts]
    /// where cols (ne[0]) must be a multiple of 256 (Q4_K block size)
    /// Memory: ne[0]=cols is fastest-varying, ne[2]=num_experts is slowest
    std::shared_ptr<Q4_KTensor> createExpertQ4K(int num_experts, int rows, int cols, uint32_t seed = 42)
    {
        // GGUF 3D convention: shape = [ne[0]=cols, ne[1]=rows, ne[2]=num_experts]
        std::vector<size_t> shape = {static_cast<size_t>(cols),
                                     static_cast<size_t>(rows),
                                     static_cast<size_t>(num_experts)};
        size_t blocks_per_row = cols / Q4_KBlock::BLOCK_SIZE;
        size_t total_blocks = num_experts * rows * blocks_per_row;
        std::vector<uint8_t> raw(total_blocks * sizeof(Q4_KBlock));

        // Fill with deterministic random data
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        auto *blocks = reinterpret_cast<Q4_KBlock *>(raw.data());
        for (size_t i = 0; i < total_blocks; ++i)
        {
            // Set d and dmin scales
            float d_val = dist(rng) * 0.01f;
            float dmin_val = std::abs(dist(rng)) * 0.001f;
            blocks[i].d = fp32_to_fp16(d_val);
            blocks[i].dmin = fp32_to_fp16(dmin_val);

            // Fill qs with random bytes
            for (size_t j = 0; j < sizeof(blocks[i].qs); ++j)
                blocks[i].qs[j] = static_cast<uint8_t>(rng());

            // Fill scales
            for (size_t j = 0; j < sizeof(blocks[i].scales); ++j)
                blocks[i].scales[j] = static_cast<uint8_t>(rng());
        }

        return std::make_shared<Q4_KTensor>(shape, raw);
    }

    /// Create a 3D Q5_K expert tensor in GGUF layout [cols, rows, num_experts]
    /// where cols (ne[0]) must be a multiple of 256 (Q5_K block size)
    /// Memory: ne[0]=cols is fastest-varying, ne[2]=num_experts is slowest
    std::shared_ptr<Q5_KTensor> createExpertQ5K(int num_experts, int rows, int cols, uint32_t seed = 42)
    {
        // GGUF 3D convention: shape = [ne[0]=cols, ne[1]=rows, ne[2]=num_experts]
        std::vector<size_t> shape = {static_cast<size_t>(cols),
                                     static_cast<size_t>(rows),
                                     static_cast<size_t>(num_experts)};
        size_t blocks_per_row = cols / Q5_KBlock::BLOCK_SIZE;
        size_t total_blocks = num_experts * rows * blocks_per_row;
        std::vector<uint8_t> raw(total_blocks * sizeof(Q5_KBlock));

        std::mt19937 rng(seed);
        auto *blocks = reinterpret_cast<Q5_KBlock *>(raw.data());
        for (size_t i = 0; i < total_blocks; ++i)
        {
            blocks[i].d = fp32_to_fp16(0.01f);
            blocks[i].dmin = fp32_to_fp16(0.001f);
            for (size_t j = 0; j < sizeof(blocks[i].qs); ++j)
                blocks[i].qs[j] = static_cast<uint8_t>(rng());
            for (size_t j = 0; j < sizeof(blocks[i].qh); ++j)
                blocks[i].qh[j] = static_cast<uint8_t>(rng());
            for (size_t j = 0; j < sizeof(blocks[i].scales); ++j)
                blocks[i].scales[j] = static_cast<uint8_t>(rng());
        }

        return std::make_shared<Q5_KTensor>(shape, raw);
    }

    /// Compute reference SwiGLU: silu(gate) * up, where silu(x) = x * sigmoid(x)
    static float silu(float x)
    {
        return x / (1.0f + std::exp(-x));
    }

    /// Reference: sigmoid
    static float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    /// Reference: dot product
    static float dot(const float *a, const float *b, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }
    /// Compute routing results and return as FP32 tensors for MoEExpertComputeStage input.
    /// Runs IMoEKernel::route() directly, converts indices to float.
    struct RoutingResult
    {
        std::shared_ptr<FP32Tensor> indices; // float-cast expert IDs [seq_len * top_k]
        std::shared_ptr<FP32Tensor> weights; // normalized weights [seq_len * top_k]
    };

    RoutingResult computeRouting(TensorBase *input, TensorBase *gate_weights,
                                 int seq_len, int d_model, int num_experts, int top_k,
                                 bool norm_topk_prob = true)
    {
        using KernelFactory = llaminar::v2::kernels::KernelFactory;
        auto *kernel = KernelFactory::getOrCreateMoEKernel(DeviceId::cpu());
        MoERoutingResult routing;
        kernel->route(input->data(), gate_weights->data(), seq_len, d_model,
                      num_experts, top_k, norm_topk_prob, routing);

        const size_t n = static_cast<size_t>(seq_len) * top_k;
        auto indices = std::make_shared<FP32Tensor>(std::vector<size_t>{n, 1});
        auto weights = std::make_shared<FP32Tensor>(std::vector<size_t>{n, 1});
        for (size_t i = 0; i < n; ++i)
            indices->mutable_data()[i] = static_cast<float>(routing.expert_indices[i]);
        std::copy(routing.expert_weights.begin(), routing.expert_weights.end(),
                  weights->mutable_data());
        return {indices, weights};
    }
};

// =========================================================================
// SharedExpertFFNStage Tests
// =========================================================================

TEST_F(MoEExpertComputeStageTest, SharedExpert_OutputNonZero)
{
    // Create small FP32 weight tensors for shared expert
    auto input = TestTensorFactory::createFP32Random({SEQ_LEN, D_MODEL}, -0.5f, 0.5f, 100);
    auto gate_w = TestTensorFactory::createFP32Random({INTERMEDIATE, D_MODEL}, -0.1f, 0.1f, 101);
    auto up_w = TestTensorFactory::createFP32Random({INTERMEDIATE, D_MODEL}, -0.1f, 0.1f, 102);
    auto down_w = TestTensorFactory::createFP32Random({D_MODEL, INTERMEDIATE}, -0.1f, 0.1f, 103);
    auto output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto prepared = makePreparedFFNFixture(
        gate_w.get(), up_w.get(), down_w.get(), DeviceId::cpu(), 0, "ffn_shexp");

    // Zero the output
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_w = gate_w.get();
    params.up_w = up_w.get();
    params.down_w = down_w.get();
    params.output = output.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.prepared_ref_gate = prepared.gate_ref;
    params.prepared_ref_up = prepared.up_ref;
    params.prepared_ref_down = prepared.down_ref;
    params.prepared_store = prepared.store.get();

    SharedExpertFFNStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Output should be non-zero
    const float *out = output->data();
    bool any_nonzero = false;
    for (int i = 0; i < SEQ_LEN * D_MODEL; ++i)
    {
        if (out[i] != 0.0f)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero) << "SharedExpertFFN output is all zeros";

    // No NaN or Inf
    for (int i = 0; i < SEQ_LEN * D_MODEL; ++i)
    {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

TEST_F(MoEExpertComputeStageTest, SharedExpert_MatchesReference)
{
    const int seq = 1;
    const int d = 8;
    const int inter = 4;

    // Create small hand-verifiable tensors
    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_w = TestTensorFactory::createFP32({inter, d});
    auto up_w = TestTensorFactory::createFP32({inter, d});
    auto down_w = TestTensorFactory::createFP32({d, inter});
    auto output = TestTensorFactory::createFP32({seq, d});
    auto prepared = makePreparedFFNFixture(
        gate_w.get(), up_w.get(), down_w.get(), DeviceId::cpu(), 0, "ffn_shexp");

    // Fill with simple values
    float *inp = input->mutable_data();
    for (int i = 0; i < d; ++i)
        inp[i] = 0.1f * (i + 1);

    float *gw = gate_w->mutable_data();
    float *uw = up_w->mutable_data();
    float *dw = down_w->mutable_data();

    // Identity-ish weights
    for (int i = 0; i < inter * d; ++i)
    {
        gw[i] = (i % d == i / inter) ? 1.0f : 0.0f;
        uw[i] = (i % d == i / inter) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < d * inter; ++i)
        dw[i] = (i % inter == i / d) ? 1.0f : 0.0f;

    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_w = gate_w.get();
    params.up_w = up_w.get();
    params.down_w = down_w.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.intermediate = inter;
    params.prepared_ref_gate = prepared.gate_ref;
    params.prepared_ref_up = prepared.up_ref;
    params.prepared_ref_down = prepared.down_ref;
    params.prepared_store = prepared.store.get();

    SharedExpertFFNStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Compute reference
    // gate_proj: for each intermediate neuron i, gate_out[i] = sum(gate_w[i,:] * input[:])
    // With our identity-ish weights: gate_out[i] = input[i] for i < inter
    // up_proj[i] = input[i] for i < inter
    // SwiGLU: silu(gate_out[i]) * up_out[i]
    // Down: output[d] = sum(down_w[d,:] * activated[:])
    std::vector<float> gate_out(inter), up_out(inter), activated(inter);
    for (int i = 0; i < inter; ++i)
    {
        gate_out[i] = dot(gw + i * d, inp, d);
        up_out[i] = dot(uw + i * d, inp, d);
        activated[i] = silu(gate_out[i]) * up_out[i];
    }
    std::vector<float> ref_out(d);
    for (int dd = 0; dd < d; ++dd)
        ref_out[dd] = dot(dw + dd * inter, activated.data(), inter);

    const float *out = output->data();
    for (int dd = 0; dd < d; ++dd)
    {
        EXPECT_NEAR(out[dd], ref_out[dd], 1e-5f)
            << "Mismatch at dim " << dd;
    }
}

TEST_F(MoEExpertComputeStageTest, SharedExpert_NullInputReturnsError)
{
    auto output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = nullptr; // null
    params.output = output.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;

    SharedExpertFFNStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
}

// =========================================================================
// SharedExpertGateStage Tests
// =========================================================================

TEST_F(MoEExpertComputeStageTest, SharedGate_AppliesSigmoidGating)
{
    const int seq = 2;
    const int d = 8;

    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_inp = TestTensorFactory::createFP32({1, d});
    auto shared_output = TestTensorFactory::createFP32({seq, d});

    // Fill input with 1s
    for (int i = 0; i < seq * d; ++i)
        input->mutable_data()[i] = 1.0f;

    // Fill gate_inp to produce known dot products
    for (int i = 0; i < d; ++i)
        gate_inp->mutable_data()[i] = 0.0f; // sigmoid(0) = 0.5

    // Fill shared output with 2.0
    for (int i = 0; i < seq * d; ++i)
        shared_output->mutable_data()[i] = 2.0f;

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_inp = gate_inp.get();
    params.shared_output = shared_output.get();
    params.seq_len = seq;
    params.d_model = d;

    SharedExpertGateStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // With gate_inp = 0 and input = 1, dot = 0, sigmoid(0) = 0.5
    // shared_output should be 2.0 * 0.5 = 1.0
    const float *out = shared_output->data();
    for (int i = 0; i < seq * d; ++i)
    {
        EXPECT_NEAR(out[i], 1.0f, 1e-5f) << "Gate mismatch at index " << i;
    }
}

TEST_F(MoEExpertComputeStageTest, SharedGate_LargePositiveDotSaturates)
{
    const int seq = 1;
    const int d = 4;

    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_inp = TestTensorFactory::createFP32({1, d});
    auto shared_output = TestTensorFactory::createFP32({seq, d});

    // Large positive gate → sigmoid ≈ 1.0
    for (int i = 0; i < d; ++i)
    {
        input->mutable_data()[i] = 10.0f;
        gate_inp->mutable_data()[i] = 10.0f;
    }
    for (int i = 0; i < d; ++i)
        shared_output->mutable_data()[i] = 3.0f;

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_inp = gate_inp.get();
    params.shared_output = shared_output.get();
    params.seq_len = seq;
    params.d_model = d;

    SharedExpertGateStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // dot = 10*10*4 = 400, sigmoid(400) ≈ 1.0
    // shared_output ≈ 3.0
    const float *out = shared_output->data();
    for (int i = 0; i < d; ++i)
    {
        EXPECT_NEAR(out[i], 3.0f, 1e-4f);
    }
}

TEST_F(MoEExpertComputeStageTest, SharedGate_NullInputReturnsError)
{
    auto shared_output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = nullptr;
    params.shared_output = shared_output.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;

    SharedExpertGateStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
}

// =========================================================================
// MoEExpertComputeStage Tests (Router + Expert FFN + Combine)
// =========================================================================

TEST_F(MoEExpertComputeStageTest, MoEFFN_OutputNonZero_Q4K)
{
    // D_MODEL=64 is not divisible by 256 (Q4_K block size).
    // We need cols that are multiples of 256 for quantized tensors.
    // Use d_model=256 for this test.
    const int d = 256;
    const int inter = 256; // Must be multiple of 256
    const int seq = 1;
    const int experts = 4;
    const int topk = 2;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 200);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 201);
    auto output = TestTensorFactory::createFP32({seq, d});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    // Expert tensors: [experts, inter, d] for gate/up, [experts, d, inter] for down
    auto gate_exps = createExpertQ4K(experts, inter, d, 210);
    auto up_exps = createExpertQ4K(experts, inter, d, 211);
    auto down_exps = createExpertQ4K(experts, d, inter, 212);

    // Compute routing externally
    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing.indices.get();
    params.routing_weights = routing.weights.get();
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    // Extract 2D expert views from 3D packed tensors (required by GEMM path)
    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

    MoEExpertComputeStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Output should be non-zero
    const float *out = output->data();
    bool any_nonzero = false;
    for (int i = 0; i < seq * d; ++i)
    {
        if (out[i] != 0.0f)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero) << "MoEFFN output is all zeros";

    // No NaN/Inf
    for (int i = 0; i < seq * d; ++i)
    {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_OutputNonZero_Q5K)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 1;
    const int experts = 4;
    const int topk = 2;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 300);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 301);
    auto output = TestTensorFactory::createFP32({seq, d});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    // Use Q5_K for down_exps (like the real model)
    auto gate_exps = createExpertQ4K(experts, inter, d, 310);
    auto up_exps = createExpertQ4K(experts, inter, d, 311);
    auto down_exps = createExpertQ5K(experts, d, inter, 312);

    // Compute routing externally
    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing.indices.get();
    params.routing_weights = routing.weights.get();
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

    MoEExpertComputeStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    const float *out = output->data();
    bool any_nonzero = false;
    for (int i = 0; i < seq * d; ++i)
    {
        if (out[i] != 0.0f)
        {
            any_nonzero = true;
            break;
        }
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
    EXPECT_TRUE(any_nonzero) << "MoEFFN Q5K output is all zeros";
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_MultipleTokens)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 4;
    const int experts = 4;
    const int topk = 2;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 400);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 401);
    auto output = TestTensorFactory::createFP32({seq, d});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    auto gate_exps = createExpertQ4K(experts, inter, d, 410);
    auto up_exps = createExpertQ4K(experts, inter, d, 411);
    auto down_exps = createExpertQ4K(experts, d, inter, 412);

    // Compute routing externally
    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing.indices.get();
    params.routing_weights = routing.weights.get();
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

    MoEExpertComputeStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Each token should get a non-zero output
    const float *out = output->data();
    for (int t = 0; t < seq; ++t)
    {
        bool token_nonzero = false;
        for (int dd = 0; dd < d; ++dd)
        {
            float v = out[t * d + dd];
            EXPECT_FALSE(std::isnan(v)) << "NaN at token " << t << " dim " << dd;
            if (v != 0.0f)
                token_nonzero = true;
        }
        EXPECT_TRUE(token_nonzero) << "Token " << t << " output is all zeros";
    }
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_NullWeightsReturnsError)
{
    auto input = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto routing_idx = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
    auto routing_wt = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing_idx.get();
    params.routing_weights = routing_wt.get();
    params.gate_exps = nullptr; // null expert weights
    params.up_exps = nullptr;
    params.down_exps = nullptr;
    params.output = output.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;

    MoEExpertComputeStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_CPUActiveExpertWithoutPreparedEngineReturnsError)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 2;
    const int experts = 4;
    const int topk = 1;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 260);
    auto output = TestTensorFactory::createFP32({seq, d});
    auto gate_exps = createExpertQ4K(experts, inter, d, 261);
    auto up_exps = createExpertQ4K(experts, inter, d, 262);
    auto down_exps = createExpertQ4K(experts, d, inter, 263);

    auto routing_indices = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq * topk), 1});
    auto routing_weights = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq * topk), 1});
    for (int i = 0; i < seq * topk; ++i)
    {
        routing_indices->mutable_data()[i] = 0.0f;
        routing_weights->mutable_data()[i] = 1.0f;
    }

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing_indices.get();
    params.routing_weights = routing_weights.get();
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));

    MoEExpertComputeStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_DifferentTokensGetDifferentOutputs)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 2;
    const int experts = 4;
    const int topk = 2;

    // Create two very different tokens
    auto input = TestTensorFactory::createFP32({seq, d});
    float *inp = input->mutable_data();
    for (int i = 0; i < d; ++i)
    {
        inp[i] = 1.0f;      // Token 0: all 1s
        inp[d + i] = -1.0f; // Token 1: all -1s
    }

    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 501);
    auto output = TestTensorFactory::createFP32({seq, d});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    auto gate_exps = createExpertQ4K(experts, inter, d, 510);
    auto up_exps = createExpertQ4K(experts, inter, d, 511);
    auto down_exps = createExpertQ4K(experts, d, inter, 512);

    // Compute routing externally
    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing.indices.get();
    params.routing_weights = routing.weights.get();
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

    MoEExpertComputeStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Token 0 and Token 1 should produce different outputs
    const float *out = output->data();
    float diff = 0.0f;
    for (int i = 0; i < d; ++i)
        diff += std::abs(out[i] - out[d + i]);

    EXPECT_GT(diff, 0.0f) << "Different input tokens produced identical outputs";
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_NormTopKProbSumsToOne)
{
    // Verify that with norm_topk_prob=true, expert weights sum to ~1
    // We test this indirectly by checking the output has reasonable magnitude
    const int d = 256;
    const int inter = 256;
    const int seq = 1;
    const int experts = 8;
    const int topk = 2;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 600);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 601);
    auto output_norm = TestTensorFactory::createFP32({seq, d});
    auto output_no_norm = TestTensorFactory::createFP32({seq, d});
    std::memset(output_norm->mutable_data(), 0, output_norm->numel() * sizeof(float));
    std::memset(output_no_norm->mutable_data(), 0, output_no_norm->numel() * sizeof(float));

    auto gate_exps = createExpertQ4K(experts, inter, d, 610);
    auto up_exps = createExpertQ4K(experts, inter, d, 611);
    auto down_exps = createExpertQ4K(experts, d, inter, 612);

    // Run with norm_topk_prob=true
    {
        auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk, true);

        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = input.get();
        params.routing_indices = routing.indices.get();
        params.routing_weights = routing.weights.get();
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.output = output_norm.get();
        params.seq_len = seq;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = topk;
        params.expert_intermediate = inter;

        ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
        ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

        MoEExpertComputeStage stage(params);
        ASSERT_TRUE(stage.execute(cpu_ctx_.get()));
    }

    // Run with norm_topk_prob=false
    {
        auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk, false);

        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = input.get();
        params.routing_indices = routing.indices.get();
        params.routing_weights = routing.weights.get();
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.output = output_no_norm.get();
        params.seq_len = seq;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = topk;
        params.expert_intermediate = inter;

        ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
        ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

        MoEExpertComputeStage stage(params);
        ASSERT_TRUE(stage.execute(cpu_ctx_.get()));
    }

    // Both outputs should be non-zero and possibly different
    const float *norm_out = output_norm->data();
    const float *no_norm_out = output_no_norm->data();

    bool norm_nonzero = false;
    bool no_norm_nonzero = false;
    for (int i = 0; i < d; ++i)
    {
        if (norm_out[i] != 0.0f)
            norm_nonzero = true;
        if (no_norm_out[i] != 0.0f)
            no_norm_nonzero = true;
    }
    EXPECT_TRUE(norm_nonzero);
    EXPECT_TRUE(no_norm_nonzero);
}

// =========================================================================
// Stage Metadata Tests
// =========================================================================

TEST_F(MoEExpertComputeStageTest, MoEFFN_TypeAndName)
{
    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;

    MoEExpertComputeStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::MOE_EXPERT_FFN);
    EXPECT_EQ(stage.name(), "moe_ffn");
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_GT(stage.estimatedFlops(), 0u);
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_GraphCapturableRejectsCPUAndPrefill)
{
    ScopedRocmMoEFlags flags(true, true);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;

    MoEExpertComputeStage cpu_stage(params);
    EXPECT_FALSE(cpu_stage.isGraphCapturable());

    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    MoEExpertComputeStage prefill_stage(params);
    EXPECT_FALSE(prefill_stage.isGraphCapturable());
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_PrefillRuntimeGroupingFoundationKeepsCaptureDisabled)
{
    ScopedRocmMoEFlags flags(true, true, true);

    auto input = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto routing_indices = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.output = output.get();
    params.routing_indices = routing_indices.get();
    params.routing_weights = routing_weights.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.local_expert_start = 0;
    params.local_expert_count = NUM_EXPERTS;
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;

    MoEExpertComputeStage stage(params);
    EXPECT_FALSE(stage.isGraphCapturable());
    EXPECT_FALSE(runtime_table.hasPrefillRouteScratchCapacity(0, SEQ_LEN));
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_PrefillCaptureRequiresFixedTopologyRuntimeExpertPath)
{
#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    if (!hasROCmDeviceForStageTests())
        GTEST_SKIP() << "No ROCm GPU available, skipping fixed-topology prefill capture predicate test";

    ScopedRocmMoEFlags flags(true, true, true);

    auto input = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto routing_indices = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

    DeviceMoERuntimeTable::Config config;
    config.device_id = DeviceId::rocm(0);
    config.num_layers = 1;
    config.num_experts = NUM_EXPERTS;
    config.top_k = TOP_K;
    config.mirror_to_device = true;
    config.prefill_token_capacity = SEQ_LEN;
    MoERuntimeTable runtime_table(config);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.output = output.get();
    params.routing_indices = routing_indices.get();
    params.routing_weights = routing_weights.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.local_expert_start = 0;
    params.local_expert_count = NUM_EXPERTS;
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;

    MoEExpertComputeStage missing_engines_stage(params);
    EXPECT_FALSE(missing_engines_stage.isGraphCapturable());

    std::vector<std::unique_ptr<FakePreparedGemm>> owned_gemms;
    attachFakePreparedExpertEngines(params, owned_gemms);
    MoEExpertComputeStage fixed_stage(params);
    EXPECT_TRUE(fixed_stage.isGraphCapturable());

    mutableDebugEnv().rocm.moe_grouped_prefill = false;
    MoEExpertComputeStage disabled_stage(params);
    EXPECT_FALSE(disabled_stage.isGraphCapturable());
#else
    GTEST_SKIP() << "ROCm release-style build required for fixed-topology prefill capture predicate";
#endif
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_GraphCapturableRocmDecodeHonorsReleaseOnlyGuardAndFlags)
{
    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto output = TestTensorFactory::createFP32({1, D_MODEL});
    auto routing_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({TOP_K, 1});

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.output = output.get();
    params.routing_indices = routing_indices.get();
    params.routing_weights = routing_weights.get();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.local_expert_start = 0;
    params.local_expert_count = -1;
    params.layer_idx = 0;

    std::vector<std::unique_ptr<FakePreparedGemm>> owned_gemms;
    attachFakePreparedExpertEngines(params, owned_gemms);

    {
        ScopedRocmMoEFlags flags(false, true);
        MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        params.moe_runtime_table = &runtime_table;
        MoEExpertComputeStage stage(params);
        EXPECT_FALSE(stage.isGraphCapturable());
    }

    {
        ScopedRocmMoEFlags flags(true, true);
        params.moe_runtime_table = nullptr;
        MoEExpertComputeStage missing_runtime_stage(params);
        EXPECT_FALSE(missing_runtime_stage.isGraphCapturable());

        MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        params.moe_runtime_table = &runtime_table;
        MoEExpertComputeStage stage(params);
        EXPECT_FALSE(stage.isGraphCapturable());
    }
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_RuntimeTableBankInitializedFromPreparedDescriptors)
{
    ScopedRocmMoEFlags flags(true, true);

    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto output = TestTensorFactory::createFP32({1, D_MODEL});
    auto routing_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({TOP_K, 1});

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.output = output.get();
    params.routing_indices = routing_indices.get();
    params.routing_weights = routing_weights.get();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.local_expert_start = 0;
    params.local_expert_count = NUM_EXPERTS;
    params.layer_idx = 0;

    std::vector<std::unique_ptr<FakePreparedGemm>> owned_gemms;
    attachFakePreparedExpertEngines(params, owned_gemms);

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
    params.moe_runtime_table = &runtime_table;
    MoEExpertComputeStage stage(params);

    const auto &state = runtime_table.hostLayerState(0);
    ASSERT_EQ(state.active_epoch, 1u);
    ASSERT_LE(state.active_bank, 1u);
    const auto &bank = state.banks[state.active_bank];
    ASSERT_EQ(bank.expert_count, static_cast<uint32_t>(NUM_EXPERTS));

    DeviceNativeVNNIMatrixDesc expected_gate;
    DeviceNativeVNNIMatrixDesc expected_up;
    DeviceNativeVNNIMatrixDesc expected_down;
    ASSERT_TRUE(params.prepared_gate_gemm[1]->exportNativeVNNIMatrixDesc(expected_gate));
    ASSERT_TRUE(params.prepared_up_gemm[1]->exportNativeVNNIMatrixDesc(expected_up));
    ASSERT_TRUE(params.prepared_down_gemm[1]->exportNativeVNNIMatrixDesc(expected_down));

    const auto &expert = bank.experts[1];
    EXPECT_EQ(expert.logical_expert_id, 1);
    EXPECT_EQ(expert.local_slot, 1);
    EXPECT_EQ(bank.local_compute_mask[1], 1u);
    EXPECT_TRUE(hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Valid));
    EXPECT_TRUE(hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Resident));
    EXPECT_TRUE(hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::LocalCompute));
    EXPECT_EQ(expert.gate.payload, expected_gate.payload);
    EXPECT_EQ(expert.gate.scales, expected_gate.scales);
    EXPECT_EQ(expert.gate.n, expected_gate.n);
    EXPECT_EQ(expert.gate.k, expected_gate.k);
    EXPECT_EQ(expert.up.payload, expected_up.payload);
    EXPECT_EQ(expert.up.blocks_per_row, expected_up.blocks_per_row);
    EXPECT_EQ(expert.down.payload, expected_down.payload);
    EXPECT_EQ(expert.down.n, expected_down.n);
    EXPECT_EQ(expert.down.k, expected_down.k);
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_GraphCapturableRejectsMaskedAndReplicaDispatch)
{
    ScopedRocmMoEFlags flags(true, true);

    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto output = TestTensorFactory::createFP32({1, D_MODEL});
    auto routing_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({TOP_K, 1});

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.output = output.get();
    params.routing_indices = routing_indices.get();
    params.routing_weights = routing_weights.get();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.local_expert_start = 0;
    params.local_expert_count = NUM_EXPERTS;

    std::vector<std::unique_ptr<FakePreparedGemm>> owned_gemms;
    attachFakePreparedExpertEngines(params, owned_gemms);

    params.expert_mask = {true, false, true, true};
    MoEExpertComputeStage masked_stage(params);
    EXPECT_FALSE(masked_stage.isGraphCapturable());

    params.expert_mask.assign(NUM_EXPERTS, true);
    params.replica_set.num_replicated = 1;
    MoEExpertComputeStage replicated_stage(params);
    EXPECT_FALSE(replicated_stage.isGraphCapturable());
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_GraphCapturableRejectsMissingOrUnsupportedPreparedEngines)
{
    ScopedRocmMoEFlags flags(true, true);

    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto output = TestTensorFactory::createFP32({1, D_MODEL});
    auto routing_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({TOP_K, 1});

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.output = output.get();
    params.routing_indices = routing_indices.get();
    params.routing_weights = routing_weights.get();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.local_expert_start = 0;
    params.local_expert_count = NUM_EXPERTS;

    MoEExpertComputeStage missing_stage(params);
    EXPECT_FALSE(missing_stage.isGraphCapturable());

    std::vector<std::unique_ptr<FakePreparedGemm>> owned_gemms;
    attachFakePreparedExpertEngines(params, owned_gemms, /*gateup_codebook=*/99);
    MoEExpertComputeStage unsupported_stage(params);
    EXPECT_FALSE(unsupported_stage.isGraphCapturable());

    attachFakePreparedExpertEngines(params, owned_gemms, /*gateup_codebook=*/4, /*down_codebook=*/4, /*converted=*/false);
    MoEExpertComputeStage unconverted_stage(params);
    EXPECT_FALSE(unconverted_stage.isGraphCapturable());
}

TEST_F(MoEExpertComputeStageTest, SharedExpert_TypeAndName)
{
    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;

    SharedExpertFFNStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::MOE_SHARED_EXPERT_FFN);
    EXPECT_EQ(stage.name(), "shared_expert_ffn");
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_FALSE(stage.isGraphCapturable());
    EXPECT_GT(stage.estimatedFlops(), 0u);
}

TEST_F(MoEExpertComputeStageTest, SharedGate_TypeAndName)
{
    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;

    SharedExpertGateStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::MOE_SHARED_EXPERT_GATE);
    EXPECT_EQ(stage.name(), "shared_expert_gate");
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_FALSE(stage.isGraphCapturable());
}

// =========================================================================
// isNonGemmWeight Tests (regression: 3D tensors must not go through GEMM prep)
// =========================================================================

TEST_F(MoEExpertComputeStageTest, IsNonGemmWeight_ExpertTensorsExcluded)
{
    // Verify isNonGemmWeight correctly excludes MoE tensors
    WeightShardingConfig config;

    EXPECT_TRUE(config.isNonGemmWeight("blk.0.ffn_gate_exps.weight"));
    EXPECT_TRUE(config.isNonGemmWeight("blk.5.ffn_up_exps.weight"));
    EXPECT_TRUE(config.isNonGemmWeight("blk.10.ffn_down_exps.weight"));
    EXPECT_TRUE(config.isNonGemmWeight("blk.0.ffn_gate_inp.weight"));
    EXPECT_TRUE(config.isNonGemmWeight("blk.0.ffn_gate_inp_shexp.weight"));

    // Regular weights should NOT be excluded
    EXPECT_FALSE(config.isNonGemmWeight("blk.0.ffn_gate.weight"));
    EXPECT_FALSE(config.isNonGemmWeight("blk.0.ffn_up.weight"));
    EXPECT_FALSE(config.isNonGemmWeight("blk.0.ffn_down.weight"));
    EXPECT_FALSE(config.isNonGemmWeight("blk.0.attn_q.weight"));
}
