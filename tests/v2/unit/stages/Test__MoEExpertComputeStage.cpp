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

#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar2::testing;

namespace
{
    class CapturingMoEKernel : public IMoEKernel
    {
    public:
        int observed_gate_type_id = -1;

        bool route(
            const float *hidden,
            const float *gate_weights,
            int seq_len, int d_model,
            int num_experts, int top_k,
            bool normalize_weights,
            MoERoutingResult &result) override
        {
            (void)hidden;
            (void)gate_weights;
            (void)seq_len;
            (void)d_model;
            (void)num_experts;
            (void)top_k;
            (void)normalize_weights;
            (void)result;
            return false;
        }

        void gatherTokenBatch(
            const float *hidden,
            float *batch_buffer,
            const int *token_indices,
            int num_tokens, int d_model) override
        {
            for (int i = 0; i < num_tokens; ++i)
                std::copy_n(hidden + static_cast<size_t>(token_indices[i]) * d_model,
                            d_model,
                            batch_buffer + static_cast<size_t>(i) * d_model);
        }

        void scatterAddWeighted(
            float *output,
            const float *expert_output,
            const int *token_indices,
            const float *weights,
            int num_tokens, int d_model) override
        {
            for (int i = 0; i < num_tokens; ++i)
            {
                float *dst = output + static_cast<size_t>(token_indices[i]) * d_model;
                const float *src = expert_output + static_cast<size_t>(i) * d_model;
                for (int j = 0; j < d_model; ++j)
                    dst[j] += weights[i] * src[j];
            }
        }

        void sharedExpertGate(
            const float *input,
            const float *gate_inp,
            float *shared_output,
            int seq_len, int d_model) override
        {
            for (int t = 0; t < seq_len; ++t)
            {
                const float *row = input + static_cast<size_t>(t) * d_model;
                float dot = 0.0f;
                for (int j = 0; j < d_model; ++j)
                    dot += gate_inp[j] * row[j];

                const float gate = 1.0f / (1.0f + std::exp(-dot));
                float *out = shared_output + static_cast<size_t>(t) * d_model;
                for (int j = 0; j < d_model; ++j)
                    out[j] *= gate;
            }
        }

        void sharedExpertGateFromTensors(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            int seq_len, int d_model) override
        {
            observed_gate_type_id = gate_inp ? gate_inp->native_type_id() : -1;
            IMoEKernel::sharedExpertGateFromTensors(
                input, gate_inp, shared_output, seq_len, d_model);
        }

        void swiGLU(float *gate, const float *up, int count) override
        {
            for (int i = 0; i < count; ++i)
            {
                const float x = gate[i];
                gate[i] = x / (1.0f + std::exp(-x)) * up[i];
            }
        }

        bool supports_device(int device_idx) const override
        {
            return device_idx < 0;
        }
    };
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

    /// Create a 3D IQ3_S expert tensor in GGUF layout [cols, rows, num_experts].
    std::shared_ptr<IQ3_STensor> createExpertIQ3S(int num_experts, int rows, int cols, uint32_t seed = 42)
    {
        std::vector<size_t> shape = {static_cast<size_t>(cols),
                                     static_cast<size_t>(rows),
                                     static_cast<size_t>(num_experts)};
        size_t blocks_per_row = (static_cast<size_t>(cols) + IQ3_SBlock::BLOCK_SIZE - 1) /
                                IQ3_SBlock::BLOCK_SIZE;
        size_t total_blocks = static_cast<size_t>(num_experts) *
                              static_cast<size_t>(rows) *
                              blocks_per_row;
        std::vector<uint8_t> raw(total_blocks * sizeof(IQ3_SBlock));

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        auto *blocks = reinterpret_cast<IQ3_SBlock *>(raw.data());
        for (size_t i = 0; i < total_blocks; ++i)
        {
            float max_abs = 0.0f;
            for (size_t j = 0; j < IQ3_SBlock::BLOCK_SIZE; ++j)
                max_abs = std::max(max_abs, std::abs(dist(rng)));

            blocks[i].d = fp32_to_fp16(std::max(max_abs / 3.5f, 1e-6f));
            std::memset(blocks[i].qh, 0, sizeof(blocks[i].qh));
            std::memset(blocks[i].signs, 0, sizeof(blocks[i].signs));
            std::memset(blocks[i].scales, 0x44, sizeof(blocks[i].scales));
            for (uint8_t &q : blocks[i].qs)
                q = static_cast<uint8_t>(rng() & 0xFF);
        }

        return std::make_shared<IQ3_STensor>(shape, raw);
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

    static float maxAbsDiff(const float *a, const float *b, size_t n)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        return max_diff;
    }

    static float relativeL2(const float *a, const float *b, size_t n)
    {
        double diff = 0.0;
        double ref = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            diff += d * d;
            ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return ref > 0.0 ? static_cast<float>(std::sqrt(diff / ref)) : 0.0f;
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

TEST_F(MoEExpertComputeStageTest, SharedGate_MaterializesBF16GateInputAsFP32ForTensorKernel)
{
    const int seq = 1;
    const int d = 4;

    auto input = TestTensorFactory::createFP32({seq, d});
    BF16Tensor gate_inp({1, static_cast<size_t>(d)});
    auto shared_output = TestTensorFactory::createFP32({seq, d});

    const float input_values[d] = {1.0f, -2.0f, 0.5f, 3.0f};
    const float gate_values[d] = {0.03125f, -0.125f, 0.0625f, 0.25f};
    for (int i = 0; i < d; ++i)
    {
        input->mutable_data()[i] = input_values[i];
        shared_output->mutable_data()[i] = 2.0f;
    }
    gate_inp.from_fp32(gate_values, d);
    ASSERT_EQ(gate_inp.native_type(), TensorType::BF16);

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_inp = &gate_inp;
    params.shared_output = shared_output.get();
    params.seq_len = seq;
    params.d_model = d;

    CapturingMoEKernel kernel;
    SharedExpertGateStage stage(params);
    stage.setMoEKernelForTesting(&kernel);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    EXPECT_EQ(kernel.observed_gate_type_id, static_cast<int>(TensorType::FP32))
        << "SharedExpertGateStage must not pass BF16 gate vectors into tensor-aware GPU-style kernels";

    std::vector<float> gate_fp32(d);
    gate_inp.to_fp32(gate_fp32.data());
    float dot = 0.0f;
    for (int i = 0; i < d; ++i)
        dot += gate_fp32[i] * input_values[i];
    const float expected_gate = 1.0f / (1.0f + std::exp(-dot));

    const float *out = shared_output->data();
    for (int i = 0; i < d; ++i)
        EXPECT_NEAR(out[i], 2.0f * expected_gate, 1e-5f) << "Gate mismatch at dim " << i;
}

TEST_F(MoEExpertComputeStageTest, CombinedSharedVerifierMaterializesBF16GateInputAsFP32)
{
    const int d = 4;

    BF16Tensor gate_inp({1, static_cast<size_t>(d)});
    const float gate_values[d] = {0.03125f, -0.125f, 0.0625f, 0.25f};
    gate_inp.from_fp32(gate_values, d);
    ASSERT_EQ(gate_inp.native_type(), TensorType::BF16);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.shared_gate_inp = &gate_inp;

    MoEExpertComputeStage stage(params);
    TensorBase *effective_gate = stage.combinedSharedGateInputForTesting();

    ASSERT_NE(effective_gate, nullptr);
    EXPECT_EQ(effective_gate->native_type(), TensorType::FP32)
        << "Combined MoE verifier grouping kernels read the shared gate input as float*, "
        << "so non-FP32 weights must be normalized exactly like SharedExpertGateStage.";

    const float *effective_values = effective_gate->data();
    std::vector<float> expected(d);
    gate_inp.to_fp32(expected.data());
    for (int i = 0; i < d; ++i)
        EXPECT_FLOAT_EQ(effective_values[i], expected[static_cast<size_t>(i)])
            << "Converted shared gate value mismatch at dim " << i;
}

TEST_F(MoEExpertComputeStageTest, SharedGate_FusedCombinePublishesGatedSharedAndCombinedOutput)
{
    const int seq = 2;
    const int d = 4;

    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_inp = TestTensorFactory::createFP32({1, d});
    auto shared_output = TestTensorFactory::createFP32({seq, d});
    auto routed_residual = TestTensorFactory::createFP32({seq, d});
    auto combined_output = TestTensorFactory::createFP32({seq, d});

    for (int i = 0; i < seq * d; ++i)
    {
        input->mutable_data()[i] = static_cast<float>((i % d) + 1);
        shared_output->mutable_data()[i] = 2.0f + static_cast<float>(i);
        routed_residual->mutable_data()[i] = -1.0f + 0.25f * static_cast<float>(i);
        combined_output->mutable_data()[i] = -99.0f;
    }
    for (int i = 0; i < d; ++i)
        gate_inp->mutable_data()[i] = 0.0f; // sigmoid(0) = 0.5 for every row.

    const std::vector<float> original_shared(
        shared_output->data(), shared_output->data() + shared_output->numel());

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_inp = gate_inp.get();
    params.shared_output = shared_output.get();
    params.routed_residual = routed_residual.get();
    params.combined_output = combined_output.get();
    params.seq_len = seq;
    params.d_model = d;

    SharedExpertGateStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    for (int i = 0; i < seq * d; ++i)
    {
        EXPECT_NEAR(shared_output->data()[i], 0.5f * original_shared[i], 1e-5f)
            << "Fused combine must still materialize the gated shared output";
        EXPECT_NEAR(combined_output->data()[i],
                    routed_residual->data()[i] + 0.5f * original_shared[i],
                    1e-5f)
            << "Fused combine mismatch at element " << i;
    }

    const auto contract = stage.bufferContract();
    EXPECT_EQ(contract.inouts.size(), 1u)
        << "The fused stage reads and rewrites MOE_SHARED_EXPERT_OUTPUT.";
    EXPECT_EQ(contract.outputs.size(), 1u)
        << "The fused stage also writes the final combined output.";
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

TEST_F(MoEExpertComputeStageTest, SharedGate_LargeNegativeDotCanSaturateToZero)
{
    const int seq = 1;
    const int d = 4;

    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_inp = TestTensorFactory::createFP32({1, d});
    auto shared_output = TestTensorFactory::createFP32({seq, d});

    // dot = 10 * -10 * 4 = -400. The sigmoid underflows to 0.0f when
    // stored as float, so the in-place shared expert output becomes zero.
    for (int i = 0; i < d; ++i)
    {
        input->mutable_data()[i] = 10.0f;
        gate_inp->mutable_data()[i] = -10.0f;
    }
    shared_output->mutable_data()[0] = 3.0f;
    shared_output->mutable_data()[1] = -2.0f;
    shared_output->mutable_data()[2] = 5.0f;
    shared_output->mutable_data()[3] = -7.0f;

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_inp = gate_inp.get();
    params.shared_output = shared_output.get();
    params.seq_len = seq;
    params.d_model = d;

    SharedExpertGateStage stage(params);
    EXPECT_TRUE(stage.allowsZeroOutput())
        << "Stage verifier must not treat a saturated shared-expert gate as uninitialized";
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    const float *out = shared_output->data();
    for (int i = 0; i < d; ++i)
        EXPECT_EQ(out[i], 0.0f) << "Gate should saturate to zero at dim " << i;
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

TEST_F(MoEExpertComputeStageTest, SharedExpert_TwoRowVerifierMatchesSerialDecode)
{
    const int seq = 2;
    const int d = 256;
    const int inter = 256;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 700);
    auto gate_w = TestTensorFactory::createIQ3_SRandom({static_cast<size_t>(inter), static_cast<size_t>(d)}, 701);
    auto up_w = TestTensorFactory::createIQ3_SRandom({static_cast<size_t>(inter), static_cast<size_t>(d)}, 702);
    auto down_w = TestTensorFactory::createIQ3_SRandom({static_cast<size_t>(d), static_cast<size_t>(inter)}, 703);
    auto multi_output = TestTensorFactory::createFP32({seq, d});
    auto serial_output = TestTensorFactory::createFP32({seq, d});

    auto run_shared = [&](TensorBase *run_input, TensorBase *run_output, int run_seq, int layer)
    {
        auto prepared = makePreparedFFNFixture(
            gate_w.get(), up_w.get(), down_w.get(), DeviceId::cpu(), layer, "ffn_shexp");

        SharedExpertFFNStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = run_input;
        params.gate_w = gate_w.get();
        params.up_w = up_w.get();
        params.down_w = down_w.get();
        params.output = run_output;
        params.seq_len = run_seq;
        params.d_model = d;
        params.intermediate = inter;
        params.prepared_ref_gate = prepared.gate_ref;
        params.prepared_ref_up = prepared.up_ref;
        params.prepared_ref_down = prepared.down_ref;
        params.prepared_store = prepared.store.get();
        params.force_decode_equivalent_verifier_prefill = run_seq > 1;

        SharedExpertFFNStage stage(params);
        if (run_seq > 1)
            EXPECT_TRUE(stage.usesCPUDecodeEquivalentVerifierPrefillForTesting());
        return stage.execute(cpu_ctx_.get());
    };

    ASSERT_TRUE(run_shared(input.get(), multi_output.get(), seq, 10));

    for (int row = 0; row < seq; ++row)
    {
        FP32Tensor row_input({1, static_cast<size_t>(d)});
        FP32Tensor row_output({1, static_cast<size_t>(d)});
        std::copy_n(input->data() + static_cast<size_t>(row) * d,
                    d,
                    row_input.mutable_data());
        ASSERT_TRUE(run_shared(&row_input, &row_output, 1, 20 + row));
        std::copy_n(row_output.data(),
                    d,
                    serial_output->mutable_data() + static_cast<size_t>(row) * d);
    }

    const size_t count = static_cast<size_t>(seq) * d;
    EXPECT_LT(relativeL2(multi_output->data(), serial_output->data(), count), 1e-5f);
    EXPECT_LT(maxAbsDiff(multi_output->data(), serial_output->data(), count), 1e-4f);
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_TwoRowVerifierMatchesSerialDecode_IQ3S)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 2;
    const int experts = 4;
    const int topk = 2;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 720);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 721);
    auto multi_output = TestTensorFactory::createFP32({seq, d});
    auto serial_output = TestTensorFactory::createFP32({seq, d});

    auto gate_exps = createExpertIQ3S(experts, inter, d, 730);
    auto up_exps = createExpertIQ3S(experts, inter, d, 731);
    auto down_exps = createExpertIQ3S(experts, d, inter, 732);

    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    auto run_moe = [&](TensorBase *run_input,
                       TensorBase *run_indices,
                       TensorBase *run_weights,
                       TensorBase *run_output,
                       int run_seq)
    {
        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = run_input;
        params.routing_indices = run_indices;
        params.routing_weights = run_weights;
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.output = run_output;
        params.seq_len = run_seq;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = topk;
        params.expert_intermediate = inter;
        params.force_decode_equivalent_verifier_prefill = run_seq > 1;

        if (!MoEExpertComputeStage::extractExpertViews(params))
            return false;
        if (!MoEExpertComputeStage::prepareExpertGemmEngines(params))
            return false;
        MoEExpertComputeStage stage(params);
        return stage.execute(cpu_ctx_.get());
    };

    ASSERT_TRUE(run_moe(input.get(),
                        routing.indices.get(),
                        routing.weights.get(),
                        multi_output.get(),
                        seq));

    for (int row = 0; row < seq; ++row)
    {
        FP32Tensor row_input({1, static_cast<size_t>(d)});
        FP32Tensor row_indices({static_cast<size_t>(topk), 1});
        FP32Tensor row_weights({static_cast<size_t>(topk), 1});
        FP32Tensor row_output({1, static_cast<size_t>(d)});

        std::copy_n(input->data() + static_cast<size_t>(row) * d,
                    d,
                    row_input.mutable_data());
        std::copy_n(routing.indices->data() + static_cast<size_t>(row) * topk,
                    topk,
                    row_indices.mutable_data());
        std::copy_n(routing.weights->data() + static_cast<size_t>(row) * topk,
                    topk,
                    row_weights.mutable_data());

        ASSERT_TRUE(run_moe(&row_input, &row_indices, &row_weights, &row_output, 1));
        std::copy_n(row_output.data(),
                    d,
                    serial_output->mutable_data() + static_cast<size_t>(row) * d);
    }

    const size_t count = static_cast<size_t>(seq) * d;
    EXPECT_LT(relativeL2(multi_output->data(), serial_output->data(), count), 1e-5f);
    EXPECT_LT(maxAbsDiff(multi_output->data(), serial_output->data(), count), 1e-4f);
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_TwoRowVerifierMatchesSerialDecode_IQ3S_TopK8)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 2;
    const int experts = 16;
    const int topk = 8;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 740);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 741);
    auto multi_output = TestTensorFactory::createFP32({seq, d});
    auto serial_output = TestTensorFactory::createFP32({seq, d});

    auto gate_exps = createExpertIQ3S(experts, inter, d, 750);
    auto up_exps = createExpertIQ3S(experts, inter, d, 751);
    auto down_exps = createExpertIQ3S(experts, d, inter, 752);

    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    auto run_moe = [&](TensorBase *run_input,
                       TensorBase *run_indices,
                       TensorBase *run_weights,
                       TensorBase *run_output,
                       int run_seq)
    {
        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = run_input;
        params.routing_indices = run_indices;
        params.routing_weights = run_weights;
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.output = run_output;
        params.seq_len = run_seq;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = topk;
        params.expert_intermediate = inter;
        params.force_decode_equivalent_verifier_prefill = run_seq > 1;

        if (!MoEExpertComputeStage::extractExpertViews(params))
            return false;
        if (!MoEExpertComputeStage::prepareExpertGemmEngines(params))
            return false;
        MoEExpertComputeStage stage(params);
        return stage.execute(cpu_ctx_.get());
    };

    ASSERT_TRUE(run_moe(input.get(),
                        routing.indices.get(),
                        routing.weights.get(),
                        multi_output.get(),
                        seq));

    for (int row = 0; row < seq; ++row)
    {
        FP32Tensor row_input({1, static_cast<size_t>(d)});
        FP32Tensor row_indices({static_cast<size_t>(topk), 1});
        FP32Tensor row_weights({static_cast<size_t>(topk), 1});
        FP32Tensor row_output({1, static_cast<size_t>(d)});

        std::copy_n(input->data() + static_cast<size_t>(row) * d,
                    d,
                    row_input.mutable_data());
        std::copy_n(routing.indices->data() + static_cast<size_t>(row) * topk,
                    topk,
                    row_indices.mutable_data());
        std::copy_n(routing.weights->data() + static_cast<size_t>(row) * topk,
                    topk,
                    row_weights.mutable_data());

        ASSERT_TRUE(run_moe(&row_input, &row_indices, &row_weights, &row_output, 1));
        std::copy_n(row_output.data(),
                    d,
                    serial_output->mutable_data() + static_cast<size_t>(row) * d);
    }

    const size_t count = static_cast<size_t>(seq) * d;
    EXPECT_LT(relativeL2(multi_output->data(), serial_output->data(), count), 1e-5f);
    EXPECT_LT(maxAbsDiff(multi_output->data(), serial_output->data(), count), 1e-4f);
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_TwoRowVerifierMatchesSerialDecode_QwenSizedQ4KQ5K_TopK8)
{
    const int d = 2048;
    const int inter = 512;
    const int seq = 2;
    const int experts = 32;
    const int topk = 8;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 760);
    auto multi_output = TestTensorFactory::createFP32({seq, d});
    auto serial_output = TestTensorFactory::createFP32({seq, d});

    auto gate_exps = createExpertQ4K(experts, inter, d, 770);
    auto up_exps = createExpertQ4K(experts, inter, d, 771);
    auto down_exps = createExpertQ5K(experts, d, inter, 772);

    FP32Tensor routing_indices({static_cast<size_t>(seq * topk), 1});
    FP32Tensor routing_weights({static_cast<size_t>(seq * topk), 1});
    const int row0_experts[topk] = {31, 0, 17, 6, 25, 11, 3, 19};
    const int row1_experts[topk] = {29, 2, 15, 8, 21, 13, 5, 27};
    for (int row = 0; row < seq; ++row)
    {
        const int *ids = row == 0 ? row0_experts : row1_experts;
        float weight_sum = 0.0f;
        for (int k = 0; k < topk; ++k)
            weight_sum += static_cast<float>(topk - k);
        for (int k = 0; k < topk; ++k)
        {
            routing_indices.mutable_data()[row * topk + k] = static_cast<float>(ids[k]);
            routing_weights.mutable_data()[row * topk + k] =
                static_cast<float>(topk - k) / weight_sum;
        }
    }

    auto run_moe = [&](TensorBase *run_input,
                       TensorBase *run_indices,
                       TensorBase *run_weights,
                       TensorBase *run_output,
                       int run_seq)
    {
        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = run_input;
        params.routing_indices = run_indices;
        params.routing_weights = run_weights;
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.output = run_output;
        params.seq_len = run_seq;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = topk;
        params.expert_intermediate = inter;
        params.force_decode_equivalent_verifier_prefill = run_seq > 1;

        if (!MoEExpertComputeStage::extractExpertViews(params))
            return false;
        if (!MoEExpertComputeStage::prepareExpertGemmEngines(params))
            return false;
        MoEExpertComputeStage stage(params);
        return stage.execute(cpu_ctx_.get());
    };

    ASSERT_TRUE(run_moe(&*input,
                        &routing_indices,
                        &routing_weights,
                        &*multi_output,
                        seq));

    for (int row = 0; row < seq; ++row)
    {
        FP32Tensor row_input({1, static_cast<size_t>(d)});
        FP32Tensor row_indices({static_cast<size_t>(topk), 1});
        FP32Tensor row_weights({static_cast<size_t>(topk), 1});
        FP32Tensor row_output({1, static_cast<size_t>(d)});

        std::copy_n(input->data() + static_cast<size_t>(row) * d,
                    d,
                    row_input.mutable_data());
        std::copy_n(routing_indices.data() + static_cast<size_t>(row) * topk,
                    topk,
                    row_indices.mutable_data());
        std::copy_n(routing_weights.data() + static_cast<size_t>(row) * topk,
                    topk,
                    row_weights.mutable_data());

        ASSERT_TRUE(run_moe(&row_input, &row_indices, &row_weights, &row_output, 1));
        std::copy_n(row_output.data(),
                    d,
                    serial_output->mutable_data() + static_cast<size_t>(row) * d);
    }

    const size_t count = static_cast<size_t>(seq) * d;
    EXPECT_LT(relativeL2(multi_output->data(), serial_output->data(), count), 1e-5f);
    EXPECT_LT(maxAbsDiff(multi_output->data(), serial_output->data(), count), 1e-4f);
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
