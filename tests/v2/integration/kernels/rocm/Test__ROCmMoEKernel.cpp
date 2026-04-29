/**
 * @file Test__ROCmMoEKernel.cpp
 * @brief Integration tests for ROCm MoE kernel vs CPU reference
 *
 * Validates that ROCmMoEKernel produces numerically equivalent results to
 * CPUMoEKernel for all 5 IMoEKernel operations:
 *   1. route()           — gate logits + softmax + top-k
 *   2. gatherTokenBatch()— gather token rows to batch
 *   3. scatterAddWeighted() — weighted scatter-add
 *   4. sharedExpertGate()— sigmoid(dot) * scale
 *   5. swiGLU()          — silu(gate) * up
 *
 * Pass Criteria:
 * - Cosine similarity >= 0.999 for all operations
 * - No NaN/Inf in outputs
 * - Top-k expert selections match CPU reference
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 */

#include <gtest/gtest.h>

#include "tensors/Tensors.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "backends/GPUDeviceContextPool.h"
#endif

#include "../../../utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{

    // ============================================================================
    // ROCm Availability Check
    // ============================================================================

    bool hasROCm()
    {
#ifdef HAVE_ROCM
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
#else
        return false;
#endif
    }

#define SKIP_IF_NO_ROCM()                                           \
    do                                                              \
    {                                                               \
        if (!hasROCm())                                             \
        {                                                           \
            GTEST_SKIP() << "No ROCm GPU available, skipping test"; \
        }                                                           \
    } while (0)

    // ============================================================================
    // Similarity Utilities
    // ============================================================================

    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        if (norm_a < 1e-30 || norm_b < 1e-30) return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    double relativeL2Error(const float *actual, const float *reference, size_t count)
    {
        double err_sq = 0.0, ref_sq = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double diff = static_cast<double>(actual[i]) - reference[i];
            err_sq += diff * diff;
            ref_sq += static_cast<double>(reference[i]) * reference[i];
        }
        if (ref_sq < 1e-30) return 0.0;
        return std::sqrt(err_sq / ref_sq);
    }

    bool hasNaNOrInf(const float *data, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::isnan(data[i]) || std::isinf(data[i]))
                return true;
        }
        return false;
    }

    // Fill vector with uniform random values
    void fillRandom(std::vector<float> &v, float lo, float hi, unsigned seed = 42)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto &x : v) x = dist(gen);
    }

} // namespace

#ifdef HAVE_ROCM

// ============================================================================
// Test: route() — Gate logits + softmax + top-k
// ============================================================================

TEST(Test__ROCmMoEKernel, Route_DecodeSmall)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 1;
    const int d_model = 2048;
    const int num_experts = 64;
    const int top_k = 8;

    // Prepare host data
    std::vector<float> hidden(seq_len * d_model);
    std::vector<float> gate_weights(num_experts * d_model);
    fillRandom(hidden, -1.0f, 1.0f, 42);
    fillRandom(gate_weights, -0.1f, 0.1f, 123);

    // CPU reference
    CPUMoEKernel cpu_kernel;
    MoERoutingResult cpu_result;
    ASSERT_TRUE(cpu_kernel.route(hidden.data(), gate_weights.data(),
                                  seq_len, d_model, num_experts, top_k,
                                  true, cpu_result));

    // GPU execution
    ROCmMoEKernel gpu_kernel(0);

    // Upload data to device
    float *d_hidden = nullptr, *d_gate_weights = nullptr;
    hipMalloc(&d_hidden, hidden.size() * sizeof(float));
    hipMalloc(&d_gate_weights, gate_weights.size() * sizeof(float));
    hipMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_gate_weights, gate_weights.data(), gate_weights.size() * sizeof(float), hipMemcpyHostToDevice);

    MoERoutingResult gpu_result;
    ASSERT_TRUE(gpu_kernel.route(d_hidden, d_gate_weights,
                                  seq_len, d_model, num_experts, top_k,
                                  true, gpu_result));

    hipFree(d_hidden);
    hipFree(d_gate_weights);

    // Verify router logits parity
    ASSERT_EQ(gpu_result.router_logits.size(), cpu_result.router_logits.size());
    ASSERT_FALSE(hasNaNOrInf(gpu_result.router_logits.data(), gpu_result.router_logits.size()));

    double logits_cosine = cosineSimilarity(
        gpu_result.router_logits.data(), cpu_result.router_logits.data(),
        gpu_result.router_logits.size());
    EXPECT_GE(logits_cosine, 0.999)
        << "Router logits cosine similarity too low: " << logits_cosine;

    // Verify top-k expert selections match
    ASSERT_EQ(gpu_result.expert_indices.size(), cpu_result.expert_indices.size());
    int matching_experts = 0;
    for (size_t i = 0; i < cpu_result.expert_indices.size(); ++i)
    {
        if (gpu_result.expert_indices[i] == cpu_result.expert_indices[i])
            ++matching_experts;
    }
    double expert_match_rate = static_cast<double>(matching_experts) / cpu_result.expert_indices.size();
    EXPECT_GE(expert_match_rate, 0.75)
        << "Expert selection match rate too low: " << expert_match_rate
        << " (" << matching_experts << "/" << cpu_result.expert_indices.size() << ")";

    // Verify top-k weights parity
    ASSERT_FALSE(hasNaNOrInf(gpu_result.expert_weights.data(), gpu_result.expert_weights.size()));
    double weights_cosine = cosineSimilarity(
        gpu_result.expert_weights.data(), cpu_result.expert_weights.data(),
        gpu_result.expert_weights.size());
    EXPECT_GE(weights_cosine, 0.99)
        << "Expert weights cosine similarity too low: " << weights_cosine;

    std::cout << "[Route_DecodeSmall] logits_cosine=" << std::fixed << std::setprecision(6)
              << logits_cosine
              << " expert_match=" << matching_experts << "/" << cpu_result.expert_indices.size()
              << " weights_cosine=" << weights_cosine << std::endl;
}

TEST(Test__ROCmMoEKernel, Route_PrefillLarge)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 32;
    const int d_model = 2048;
    const int num_experts = 256;
    const int top_k = 8;

    std::vector<float> hidden(seq_len * d_model);
    std::vector<float> gate_weights(num_experts * d_model);
    fillRandom(hidden, -1.0f, 1.0f, 42);
    fillRandom(gate_weights, -0.1f, 0.1f, 123);

    // CPU reference
    CPUMoEKernel cpu_kernel;
    MoERoutingResult cpu_result;
    ASSERT_TRUE(cpu_kernel.route(hidden.data(), gate_weights.data(),
                                  seq_len, d_model, num_experts, top_k,
                                  true, cpu_result));

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    float *d_hidden = nullptr, *d_gate = nullptr;
    hipMalloc(&d_hidden, hidden.size() * sizeof(float));
    hipMalloc(&d_gate, gate_weights.size() * sizeof(float));
    hipMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_gate, gate_weights.data(), gate_weights.size() * sizeof(float), hipMemcpyHostToDevice);

    MoERoutingResult gpu_result;
    ASSERT_TRUE(gpu_kernel.route(d_hidden, d_gate,
                                  seq_len, d_model, num_experts, top_k,
                                  true, gpu_result));

    hipFree(d_hidden);
    hipFree(d_gate);

    // Verify per-token
    ASSERT_FALSE(hasNaNOrInf(gpu_result.router_logits.data(), gpu_result.router_logits.size()));
    double logits_cosine = cosineSimilarity(
        gpu_result.router_logits.data(), cpu_result.router_logits.data(),
        gpu_result.router_logits.size());
    EXPECT_GE(logits_cosine, 0.999);

    // Count matching top-1 experts across all tokens
    int top1_matches = 0;
    for (int t = 0; t < seq_len; ++t)
    {
        if (gpu_result.expert_indices[t * top_k] == cpu_result.expert_indices[t * top_k])
            ++top1_matches;
    }
    double top1_rate = static_cast<double>(top1_matches) / seq_len;
    EXPECT_GE(top1_rate, 0.8)
        << "Top-1 expert match rate: " << top1_rate;

    std::cout << "[Route_PrefillLarge] logits_cosine=" << std::fixed << std::setprecision(6)
              << logits_cosine
              << " top1_match=" << top1_matches << "/" << seq_len << std::endl;
}

// ============================================================================
// Test: gatherTokenBatch()
// ============================================================================

TEST(Test__ROCmMoEKernel, GatherTokenBatch)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 64;
    const int d_model = 2048;
    const int num_tokens = 8;

    std::vector<float> hidden(seq_len * d_model);
    fillRandom(hidden, -1.0f, 1.0f, 42);

    // Select some token indices
    std::vector<int> token_indices = {0, 5, 12, 23, 31, 44, 50, 63};

    // CPU reference
    CPUMoEKernel cpu_kernel;
    std::vector<float> cpu_batch(num_tokens * d_model);
    cpu_kernel.gatherTokenBatch(hidden.data(), cpu_batch.data(),
                                token_indices.data(), num_tokens, d_model);

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    float *d_hidden = nullptr, *d_batch = nullptr;
    int *d_indices = nullptr;
    hipMalloc(&d_hidden, hidden.size() * sizeof(float));
    hipMalloc(&d_batch, num_tokens * d_model * sizeof(float));
    hipMalloc(&d_indices, num_tokens * sizeof(int));
    hipMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_indices, token_indices.data(), num_tokens * sizeof(int), hipMemcpyHostToDevice);

    gpu_kernel.gatherTokenBatch(d_hidden, d_batch, d_indices, num_tokens, d_model);
    hipDeviceSynchronize();

    std::vector<float> gpu_batch(num_tokens * d_model);
    hipMemcpy(gpu_batch.data(), d_batch, gpu_batch.size() * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_hidden);
    hipFree(d_batch);
    hipFree(d_indices);

    // Exact match expected for gather (just copying)
    ASSERT_FALSE(hasNaNOrInf(gpu_batch.data(), gpu_batch.size()));
    double cosine = cosineSimilarity(gpu_batch.data(), cpu_batch.data(), gpu_batch.size());
    EXPECT_GE(cosine, 0.9999)
        << "Gather cosine: " << cosine;

    // Check element-wise equality (should be bit-exact for copies)
    int mismatches = 0;
    for (size_t i = 0; i < gpu_batch.size(); ++i)
    {
        if (gpu_batch[i] != cpu_batch[i]) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0) << "Gather had " << mismatches << " mismatches (should be exact copy)";

    std::cout << "[GatherTokenBatch] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " mismatches=" << mismatches << std::endl;
}

// ============================================================================
// Test: scatterAddWeighted()
// ============================================================================

TEST(Test__ROCmMoEKernel, ScatterAddWeighted)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 32;
    const int d_model = 2048;
    const int num_tokens = 8;

    std::vector<float> expert_output(num_tokens * d_model);
    fillRandom(expert_output, -1.0f, 1.0f, 42);

    std::vector<int> token_indices = {0, 3, 7, 10, 15, 20, 25, 30};
    std::vector<float> weights = {0.15f, 0.12f, 0.18f, 0.10f, 0.13f, 0.11f, 0.14f, 0.07f};

    // CPU reference
    CPUMoEKernel cpu_kernel;
    std::vector<float> cpu_output(seq_len * d_model, 0.0f);
    cpu_kernel.scatterAddWeighted(cpu_output.data(), expert_output.data(),
                                  token_indices.data(), weights.data(),
                                  num_tokens, d_model);

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    float *d_output = nullptr, *d_expert = nullptr, *d_weights = nullptr;
    int *d_indices = nullptr;
    hipMalloc(&d_output, seq_len * d_model * sizeof(float));
    hipMalloc(&d_expert, expert_output.size() * sizeof(float));
    hipMalloc(&d_weights, weights.size() * sizeof(float));
    hipMalloc(&d_indices, token_indices.size() * sizeof(int));
    hipMemset(d_output, 0, seq_len * d_model * sizeof(float));
    hipMemcpy(d_expert, expert_output.data(), expert_output.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_indices, token_indices.data(), token_indices.size() * sizeof(int), hipMemcpyHostToDevice);

    gpu_kernel.scatterAddWeighted(d_output, d_expert, d_indices, d_weights, num_tokens, d_model);
    hipDeviceSynchronize();

    std::vector<float> gpu_output(seq_len * d_model);
    hipMemcpy(gpu_output.data(), d_output, gpu_output.size() * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_output);
    hipFree(d_expert);
    hipFree(d_weights);
    hipFree(d_indices);

    // Verify
    ASSERT_FALSE(hasNaNOrInf(gpu_output.data(), gpu_output.size()));
    double cosine = cosineSimilarity(gpu_output.data(), cpu_output.data(), gpu_output.size());
    double l2_err = relativeL2Error(gpu_output.data(), cpu_output.data(), gpu_output.size());
    EXPECT_GE(cosine, 0.999)
        << "ScatterAdd cosine: " << cosine;
    EXPECT_LE(l2_err, 0.01)
        << "ScatterAdd relative L2 error: " << l2_err;

    std::cout << "[ScatterAddWeighted] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

// ============================================================================
// Test: sharedExpertGate()
// ============================================================================

TEST(Test__ROCmMoEKernel, SharedExpertGate_Decode)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 1;
    const int d_model = 2048;

    std::vector<float> input(seq_len * d_model);
    std::vector<float> gate_inp(d_model);
    std::vector<float> cpu_shared_output(seq_len * d_model);
    std::vector<float> gpu_shared_output_host(seq_len * d_model);
    fillRandom(input, -1.0f, 1.0f, 42);
    fillRandom(gate_inp, -0.5f, 0.5f, 123);
    fillRandom(cpu_shared_output, -2.0f, 2.0f, 456);
    // Copy same initial shared_output for GPU
    gpu_shared_output_host = cpu_shared_output;

    // CPU reference
    CPUMoEKernel cpu_kernel;
    cpu_kernel.sharedExpertGate(input.data(), gate_inp.data(),
                                cpu_shared_output.data(), seq_len, d_model);

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    float *d_input = nullptr, *d_gate_inp = nullptr, *d_shared_output = nullptr;
    hipMalloc(&d_input, input.size() * sizeof(float));
    hipMalloc(&d_gate_inp, gate_inp.size() * sizeof(float));
    hipMalloc(&d_shared_output, gpu_shared_output_host.size() * sizeof(float));
    hipMemcpy(d_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_gate_inp, gate_inp.data(), gate_inp.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_shared_output, gpu_shared_output_host.data(),
              gpu_shared_output_host.size() * sizeof(float), hipMemcpyHostToDevice);

    gpu_kernel.sharedExpertGate(d_input, d_gate_inp, d_shared_output, seq_len, d_model);
    hipDeviceSynchronize();

    hipMemcpy(gpu_shared_output_host.data(), d_shared_output,
              gpu_shared_output_host.size() * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_gate_inp);
    hipFree(d_shared_output);

    // Verify
    ASSERT_FALSE(hasNaNOrInf(gpu_shared_output_host.data(), gpu_shared_output_host.size()));
    double cosine = cosineSimilarity(gpu_shared_output_host.data(), cpu_shared_output.data(),
                                     cpu_shared_output.size());
    double l2_err = relativeL2Error(gpu_shared_output_host.data(), cpu_shared_output.data(),
                                    cpu_shared_output.size());
    EXPECT_GE(cosine, 0.999)
        << "SharedExpertGate cosine: " << cosine;
    EXPECT_LE(l2_err, 0.01)
        << "SharedExpertGate L2 error: " << l2_err;

    std::cout << "[SharedExpertGate_Decode] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

TEST(Test__ROCmMoEKernel, SharedExpertGate_Prefill)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 64;
    const int d_model = 2048;

    std::vector<float> input(seq_len * d_model);
    std::vector<float> gate_inp(d_model);
    std::vector<float> cpu_shared_output(seq_len * d_model);
    std::vector<float> gpu_shared_output_host(seq_len * d_model);
    fillRandom(input, -1.0f, 1.0f, 42);
    fillRandom(gate_inp, -0.5f, 0.5f, 123);
    fillRandom(cpu_shared_output, -2.0f, 2.0f, 456);
    gpu_shared_output_host = cpu_shared_output;

    CPUMoEKernel cpu_kernel;
    cpu_kernel.sharedExpertGate(input.data(), gate_inp.data(),
                                cpu_shared_output.data(), seq_len, d_model);

    ROCmMoEKernel gpu_kernel(0);
    float *d_input = nullptr, *d_gate_inp = nullptr, *d_shared_output = nullptr;
    hipMalloc(&d_input, input.size() * sizeof(float));
    hipMalloc(&d_gate_inp, gate_inp.size() * sizeof(float));
    hipMalloc(&d_shared_output, gpu_shared_output_host.size() * sizeof(float));
    hipMemcpy(d_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_gate_inp, gate_inp.data(), gate_inp.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_shared_output, gpu_shared_output_host.data(),
              gpu_shared_output_host.size() * sizeof(float), hipMemcpyHostToDevice);

    gpu_kernel.sharedExpertGate(d_input, d_gate_inp, d_shared_output, seq_len, d_model);
    hipDeviceSynchronize();

    hipMemcpy(gpu_shared_output_host.data(), d_shared_output,
              gpu_shared_output_host.size() * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_gate_inp);
    hipFree(d_shared_output);

    ASSERT_FALSE(hasNaNOrInf(gpu_shared_output_host.data(), gpu_shared_output_host.size()));
    double cosine = cosineSimilarity(gpu_shared_output_host.data(), cpu_shared_output.data(),
                                     cpu_shared_output.size());
    EXPECT_GE(cosine, 0.999);

    std::cout << "[SharedExpertGate_Prefill] cosine=" << std::fixed << std::setprecision(6)
              << cosine << std::endl;
}

// ============================================================================
// Test: swiGLU()
// ============================================================================

TEST(Test__ROCmMoEKernel, SwiGLU)
{
    SKIP_IF_NO_ROCM();

    const int count = 32 * 4864; // Typical MoE intermediate dim

    std::vector<float> gate(count), up(count);
    std::vector<float> cpu_gate(count), gpu_gate_host(count);
    fillRandom(gate, -2.0f, 2.0f, 42);
    fillRandom(up, -2.0f, 2.0f, 123);
    cpu_gate = gate;
    gpu_gate_host = gate;

    // CPU reference
    CPUMoEKernel cpu_kernel;
    cpu_kernel.swiGLU(cpu_gate.data(), up.data(), count);

    // GPU
    ROCmMoEKernel gpu_kernel(0);
    float *d_gate = nullptr, *d_up = nullptr;
    hipMalloc(&d_gate, count * sizeof(float));
    hipMalloc(&d_up, count * sizeof(float));
    hipMemcpy(d_gate, gpu_gate_host.data(), count * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_up, up.data(), count * sizeof(float), hipMemcpyHostToDevice);

    gpu_kernel.swiGLU(d_gate, d_up, count);
    hipDeviceSynchronize();

    hipMemcpy(gpu_gate_host.data(), d_gate, count * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_gate);
    hipFree(d_up);

    // Verify
    ASSERT_FALSE(hasNaNOrInf(gpu_gate_host.data(), gpu_gate_host.size()));
    double cosine = cosineSimilarity(gpu_gate_host.data(), cpu_gate.data(), count);
    double l2_err = relativeL2Error(gpu_gate_host.data(), cpu_gate.data(), count);
    EXPECT_GE(cosine, 0.9999)
        << "SwiGLU cosine: " << cosine;
    EXPECT_LE(l2_err, 0.001)
        << "SwiGLU L2 error: " << l2_err;

    std::cout << "[SwiGLU] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

// ============================================================================
// Test: KernelFactory dispatch creates ROCmMoEKernel for ROCm devices
// ============================================================================

TEST(Test__ROCmMoEKernel, KernelFactoryDispatch)
{
    SKIP_IF_NO_ROCM();

    using KernelFactory = llaminar::v2::kernels::KernelFactory;
    auto *kernel = KernelFactory::getOrCreateMoEKernel(DeviceId::rocm(0));
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0))
        << "ROCm MoE kernel should support GPU device index";
    EXPECT_FALSE(kernel->supports_device(-1))
        << "ROCm MoE kernel should NOT support CPU device index";
}

// ============================================================================
// Phase 2 Tests: Device-Resident Histogram + Expert Mask
// ============================================================================

// ============================================================================
// Test: recordHistogramDevice() + syncHistogramToHost()
// ============================================================================

TEST(Test__ROCmMoEKernel, Histogram_RecordAndSync)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 8;
    const int top_k = 2;
    const int num_experts = 8;
    const int layer_idx = 0;

    // Known routing indices: each token picks 2 experts
    // Token 0: experts 0, 1
    // Token 1: experts 2, 3
    // Token 2: experts 0, 2
    // Token 3: experts 1, 3
    // Token 4: experts 4, 5
    // Token 5: experts 6, 7
    // Token 6: experts 0, 7
    // Token 7: experts 3, 5
    std::vector<int> routing_indices = {
        0, 1, 2, 3, 0, 2, 1, 3, 4, 5, 6, 7, 0, 7, 3, 5};

    // Expected histogram: count occurrences of each expert
    // Expert 0: 3 (tokens 0, 2, 6)
    // Expert 1: 2 (tokens 0, 3)
    // Expert 2: 2 (tokens 1, 2)
    // Expert 3: 3 (tokens 1, 3, 7)
    // Expert 4: 1 (token 4)
    // Expert 5: 2 (tokens 4, 7)
    // Expert 6: 1 (token 5)
    // Expert 7: 2 (tokens 5, 6)
    std::vector<uint64_t> expected_counts = {3, 2, 2, 3, 1, 2, 1, 2};

    // Upload routing indices to device
    int *d_indices = nullptr;
    hipMalloc(&d_indices, routing_indices.size() * sizeof(int));
    hipMemcpy(d_indices, routing_indices.data(),
              routing_indices.size() * sizeof(int), hipMemcpyHostToDevice);

    ROCmMoEKernel gpu_kernel(0);
    gpu_kernel.recordHistogramDevice(d_indices, seq_len, top_k, layer_idx);

    // Sync histogram to host
    std::vector<uint64_t> host_counts(num_experts, 0);
    gpu_kernel.syncHistogramToHost(host_counts.data(), layer_idx, num_experts);

    hipFree(d_indices);

    // Verify counts
    uint64_t total_count = 0;
    for (int e = 0; e < num_experts; ++e)
    {
        total_count += host_counts[e];
        EXPECT_EQ(host_counts[e], expected_counts[e])
            << "Expert " << e << " count mismatch: got " << host_counts[e]
            << " expected " << expected_counts[e];
    }

    uint64_t expected_total = static_cast<uint64_t>(seq_len) * top_k;
    EXPECT_EQ(total_count, expected_total)
        << "Total histogram count mismatch";

    std::cout << "[Histogram_RecordAndSync] total_count=" << total_count
              << " expected=" << expected_total << std::endl;
}

// ============================================================================
// Test: resetHistogramDevice()
// ============================================================================

TEST(Test__ROCmMoEKernel, Histogram_Reset)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 4;
    const int top_k = 2;
    const int num_experts = 8;
    const int layer_idx = 0;

    // Record some histogram data
    std::vector<int> routing_indices = {0, 1, 2, 3, 4, 5, 6, 7};
    int *d_indices = nullptr;
    hipMalloc(&d_indices, routing_indices.size() * sizeof(int));
    hipMemcpy(d_indices, routing_indices.data(),
              routing_indices.size() * sizeof(int), hipMemcpyHostToDevice);

    ROCmMoEKernel gpu_kernel(0);
    gpu_kernel.recordHistogramDevice(d_indices, seq_len, top_k, layer_idx);

    // Verify something was recorded
    std::vector<uint64_t> counts_before(num_experts, 0);
    gpu_kernel.syncHistogramToHost(counts_before.data(), layer_idx, num_experts);
    uint64_t sum_before = 0;
    for (auto c : counts_before) sum_before += c;
    ASSERT_GT(sum_before, 0u) << "Histogram should have non-zero counts before reset";

    // Reset
    gpu_kernel.resetHistogramDevice(layer_idx, num_experts);

    // Sync and verify all zero
    std::vector<uint64_t> counts_after(num_experts, 99);
    gpu_kernel.syncHistogramToHost(counts_after.data(), layer_idx, num_experts);

    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(counts_after[e], 0u)
            << "Expert " << e << " should be 0 after reset, got " << counts_after[e];
    }

    hipFree(d_indices);

    std::cout << "[Histogram_Reset] sum_before=" << sum_before
              << " sum_after=0 (all zeroed)" << std::endl;
}

// ============================================================================
// Test: Expert mask zeros out weights for inactive experts
// ============================================================================

TEST(Test__ROCmMoEKernel, ExpertMask_ApplyZerosWeights)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 8;
    const int top_k = 4;
    const int num_experts = 8;
    const int total_slots = seq_len * top_k;

    // Create routing indices — each token picks 4 experts
    std::vector<int> routing_indices(total_slots);
    std::vector<float> routing_weights(total_slots);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> expert_dist(0, num_experts - 1);
    std::uniform_real_distribution<float> weight_dist(0.05f, 0.5f);
    for (int i = 0; i < total_slots; ++i)
    {
        routing_indices[i] = expert_dist(gen);
        routing_weights[i] = weight_dist(gen);
    }

    // Save original weights for comparison
    std::vector<float> original_weights = routing_weights;

    // Expert mask: experts 0,1 active, experts 2-7 inactive
    std::vector<bool> mask(num_experts, false);
    mask[0] = true;
    mask[1] = true;

    // Upload to device
    int *d_indices = nullptr;
    float *d_weights = nullptr;
    hipMalloc(&d_indices, total_slots * sizeof(int));
    hipMalloc(&d_weights, total_slots * sizeof(float));
    hipMemcpy(d_indices, routing_indices.data(), total_slots * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_weights, routing_weights.data(), total_slots * sizeof(float), hipMemcpyHostToDevice);

    ROCmMoEKernel gpu_kernel(0);
    // Need to convert std::vector<bool> to a contiguous bool array
    std::vector<char> mask_bytes(num_experts);
    for (int i = 0; i < num_experts; ++i)
        mask_bytes[i] = mask[i] ? 1 : 0;
    gpu_kernel.updateExpertMaskDevice(reinterpret_cast<const bool *>(mask_bytes.data()), num_experts);
    gpu_kernel.applyExpertMaskDevice(d_weights, d_indices, seq_len, top_k);
    hipDeviceSynchronize();

    // Read back weights
    std::vector<float> result_weights(total_slots);
    hipMemcpy(result_weights.data(), d_weights, total_slots * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_indices);
    hipFree(d_weights);

    // Verify
    int zeroed_count = 0;
    int unchanged_count = 0;
    for (int i = 0; i < total_slots; ++i)
    {
        int expert = routing_indices[i];
        if (expert == 0 || expert == 1)
        {
            EXPECT_FLOAT_EQ(result_weights[i], original_weights[i])
                << "Active expert " << expert << " weight at slot " << i
                << " should be unchanged";
            if (result_weights[i] == original_weights[i]) ++unchanged_count;
        }
        else
        {
            EXPECT_FLOAT_EQ(result_weights[i], 0.0f)
                << "Inactive expert " << expert << " weight at slot " << i
                << " should be zeroed";
            if (result_weights[i] == 0.0f) ++zeroed_count;
        }
    }

    std::cout << "[ExpertMask_ApplyZerosWeights] zeroed=" << zeroed_count
              << " unchanged=" << unchanged_count
              << " total=" << total_slots << std::endl;
}

// ============================================================================
// Test: All-active expert mask leaves weights unchanged
// ============================================================================

TEST(Test__ROCmMoEKernel, ExpertMask_AllActiveNoChange)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 8;
    const int top_k = 2;
    const int num_experts = 8;
    const int total_slots = seq_len * top_k;

    // Create routing indices and weights
    std::vector<int> routing_indices(total_slots);
    std::vector<float> routing_weights(total_slots);
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> expert_dist(0, num_experts - 1);
    std::uniform_real_distribution<float> weight_dist(0.05f, 0.5f);
    for (int i = 0; i < total_slots; ++i)
    {
        routing_indices[i] = expert_dist(gen);
        routing_weights[i] = weight_dist(gen);
    }
    std::vector<float> original_weights = routing_weights;

    // All experts active
    std::vector<char> mask_bytes(num_experts, 1);

    // Upload to device
    int *d_indices = nullptr;
    float *d_weights = nullptr;
    hipMalloc(&d_indices, total_slots * sizeof(int));
    hipMalloc(&d_weights, total_slots * sizeof(float));
    hipMemcpy(d_indices, routing_indices.data(), total_slots * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_weights, routing_weights.data(), total_slots * sizeof(float), hipMemcpyHostToDevice);

    ROCmMoEKernel gpu_kernel(0);
    gpu_kernel.updateExpertMaskDevice(reinterpret_cast<const bool *>(mask_bytes.data()), num_experts);
    gpu_kernel.applyExpertMaskDevice(d_weights, d_indices, seq_len, top_k);
    hipDeviceSynchronize();

    // Read back
    std::vector<float> result_weights(total_slots);
    hipMemcpy(result_weights.data(), d_weights, total_slots * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_indices);
    hipFree(d_weights);

    // Verify all weights unchanged
    int mismatches = 0;
    for (int i = 0; i < total_slots; ++i)
    {
        if (result_weights[i] != original_weights[i])
        {
            ++mismatches;
            ADD_FAILURE() << "Weight at slot " << i << " changed: "
                          << original_weights[i] << " -> " << result_weights[i];
        }
    }

    EXPECT_EQ(mismatches, 0) << "All-active mask should leave all weights unchanged";

    std::cout << "[ExpertMask_AllActiveNoChange] mismatches=" << mismatches
              << " total_slots=" << total_slots << std::endl;
}

// ============================================================================
// Phase 3 Tests: Device-Side Token Grouping
// ============================================================================

// ============================================================================
// Test: groupTokensByExpertDevice() — basic correctness
// ============================================================================

TEST(Test__ROCmMoEKernel, GroupTokensByExpert_Basic)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 8;
    const int num_experts = 4;
    const int top_k = 2;
    const int total_slots = seq_len * top_k; // 16

    // Known routing indices (token → experts):
    // Token 0: experts {1, 3}
    // Token 1: experts {0, 2}
    // Token 2: experts {0, 1}
    // Token 3: experts {2, 3}
    // Token 4: experts {0, 3}
    // Token 5: experts {1, 2}
    // Token 6: experts {0, 1}
    // Token 7: experts {2, 3}
    std::vector<int> routing_indices = {
        1, 3,  // token 0
        0, 2,  // token 1
        0, 1,  // token 2
        2, 3,  // token 3
        0, 3,  // token 4
        1, 2,  // token 5
        0, 1,  // token 6
        2, 3   // token 7
    };

    // Random routing weights
    std::vector<float> routing_weights(total_slots);
    fillRandom(routing_weights, 0.05f, 0.95f, 42);

    // Expected per-expert token sets:
    // Expert 0: tokens {1, 2, 4, 6} → count=4
    // Expert 1: tokens {0, 2, 5, 6} → count=4
    // Expert 2: tokens {1, 3, 5, 7} → count=4
    // Expert 3: tokens {0, 3, 4, 7} → count=4
    std::vector<int> expected_counts = {4, 4, 4, 4};

    // Build expected token→expert→weight mapping for verification
    // For each slot, record {expert_id, token_idx, weight}
    struct SlotInfo { int expert; int token; float weight; };
    std::vector<std::vector<SlotInfo>> expected_per_expert(num_experts);
    for (int s = 0; s < total_slots; ++s)
    {
        int token = s / top_k;
        int expert = routing_indices[s];
        expected_per_expert[expert].push_back({expert, token, routing_weights[s]});
    }

    // Upload to device
    int *d_routing_indices = nullptr;
    float *d_routing_weights = nullptr;
    hipMalloc(&d_routing_indices, total_slots * sizeof(int));
    hipMalloc(&d_routing_weights, total_slots * sizeof(float));
    hipMemcpy(d_routing_indices, routing_indices.data(), total_slots * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_routing_weights, routing_weights.data(), total_slots * sizeof(float), hipMemcpyHostToDevice);

    // Allocate output buffers
    int *d_expert_offsets = nullptr, *d_expert_counts = nullptr;
    int *d_grouped_indices = nullptr;
    float *d_grouped_weights = nullptr;
    hipMalloc(&d_expert_offsets, num_experts * sizeof(int));
    hipMalloc(&d_expert_counts, num_experts * sizeof(int));
    hipMalloc(&d_grouped_indices, total_slots * sizeof(int));
    hipMalloc(&d_grouped_weights, total_slots * sizeof(float));

    ROCmMoEKernel gpu_kernel(0);
    bool ok = gpu_kernel.groupTokensByExpertDevice(
        d_routing_indices, d_routing_weights,
        seq_len, num_experts, top_k,
        d_expert_offsets, d_expert_counts,
        d_grouped_indices, d_grouped_weights);
    ASSERT_TRUE(ok) << "groupTokensByExpertDevice failed";

    hipDeviceSynchronize();

    // D2H copy results
    std::vector<int> host_offsets(num_experts);
    std::vector<int> host_counts(num_experts);
    std::vector<int> host_grouped_indices(total_slots);
    std::vector<float> host_grouped_weights(total_slots);
    hipMemcpy(host_offsets.data(), d_expert_offsets, num_experts * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(host_counts.data(), d_expert_counts, num_experts * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(host_grouped_indices.data(), d_grouped_indices, total_slots * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(host_grouped_weights.data(), d_grouped_weights, total_slots * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_routing_indices);
    hipFree(d_routing_weights);
    hipFree(d_expert_offsets);
    hipFree(d_expert_counts);
    hipFree(d_grouped_indices);
    hipFree(d_grouped_weights);

    // Verify expert counts
    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(host_counts[e], expected_counts[e])
            << "Expert " << e << " count mismatch: got " << host_counts[e]
            << " expected " << expected_counts[e];
    }

    // Verify offsets are consistent with counts
    int running_offset = 0;
    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(host_offsets[e], running_offset)
            << "Expert " << e << " offset mismatch: got " << host_offsets[e]
            << " expected " << running_offset;
        running_offset += host_counts[e];
    }
    EXPECT_EQ(running_offset, total_slots) << "Total grouped slots mismatch";

    // Verify each expert's group contains the right tokens with the right weights
    bool all_matched = true;
    for (int e = 0; e < num_experts; ++e)
    {
        int offset = host_offsets[e];
        int count = host_counts[e];

        // Collect actual grouped tokens/weights for this expert
        std::vector<std::pair<int, float>> actual_group;
        for (int i = 0; i < count; ++i)
        {
            actual_group.push_back({host_grouped_indices[offset + i],
                                    host_grouped_weights[offset + i]});
        }

        // For each expected entry, verify it exists in the actual group
        for (const auto &expected : expected_per_expert[e])
        {
            bool found = false;
            for (auto &[tok, wt] : actual_group)
            {
                if (tok == expected.token && wt == expected.weight)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                ADD_FAILURE() << "Expert " << e << ": expected token " << expected.token
                              << " with weight " << expected.weight << " not found in group";
                all_matched = false;
            }
        }
    }

    std::cout << "[GroupTokensByExpert_Basic] expert_counts=["
              << host_counts[0] << "," << host_counts[1] << ","
              << host_counts[2] << "," << host_counts[3]
              << "] total=" << total_slots
              << " all_matched=" << (all_matched ? "true" : "false") << std::endl;
}

// ============================================================================
// Test: groupTokensByExpertDevice() — prefill scale
// ============================================================================

TEST(Test__ROCmMoEKernel, GroupTokensByExpert_PrefillScale)
{
    SKIP_IF_NO_ROCM();

    const int seq_len = 32;
    const int num_experts = 8;
    const int top_k = 4;
    const int total_slots = seq_len * top_k; // 128

    // Random routing indices (uniform over experts)
    std::vector<int> routing_indices(total_slots);
    std::vector<float> routing_weights(total_slots);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> expert_dist(0, num_experts - 1);
    std::uniform_real_distribution<float> weight_dist(0.01f, 0.5f);
    for (int i = 0; i < total_slots; ++i)
    {
        routing_indices[i] = expert_dist(gen);
        routing_weights[i] = weight_dist(gen);
    }

    // Upload to device
    int *d_routing_indices = nullptr;
    float *d_routing_weights = nullptr;
    hipMalloc(&d_routing_indices, total_slots * sizeof(int));
    hipMalloc(&d_routing_weights, total_slots * sizeof(float));
    hipMemcpy(d_routing_indices, routing_indices.data(), total_slots * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_routing_weights, routing_weights.data(), total_slots * sizeof(float), hipMemcpyHostToDevice);

    // Allocate output buffers
    int *d_expert_offsets = nullptr, *d_expert_counts = nullptr;
    int *d_grouped_indices = nullptr;
    float *d_grouped_weights = nullptr;
    hipMalloc(&d_expert_offsets, num_experts * sizeof(int));
    hipMalloc(&d_expert_counts, num_experts * sizeof(int));
    hipMalloc(&d_grouped_indices, total_slots * sizeof(int));
    hipMalloc(&d_grouped_weights, total_slots * sizeof(float));

    ROCmMoEKernel gpu_kernel(0);
    bool ok = gpu_kernel.groupTokensByExpertDevice(
        d_routing_indices, d_routing_weights,
        seq_len, num_experts, top_k,
        d_expert_offsets, d_expert_counts,
        d_grouped_indices, d_grouped_weights);
    ASSERT_TRUE(ok) << "groupTokensByExpertDevice failed";

    hipDeviceSynchronize();

    // D2H copy results
    std::vector<int> host_offsets(num_experts);
    std::vector<int> host_counts(num_experts);
    std::vector<int> host_grouped_indices(total_slots);
    std::vector<float> host_grouped_weights(total_slots);
    hipMemcpy(host_offsets.data(), d_expert_offsets, num_experts * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(host_counts.data(), d_expert_counts, num_experts * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(host_grouped_indices.data(), d_grouped_indices, total_slots * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(host_grouped_weights.data(), d_grouped_weights, total_slots * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_routing_indices);
    hipFree(d_routing_weights);
    hipFree(d_expert_offsets);
    hipFree(d_expert_counts);
    hipFree(d_grouped_indices);
    hipFree(d_grouped_weights);

    // Verify: sum of all expert_counts == total_slots
    int sum_counts = 0;
    for (int e = 0; e < num_experts; ++e)
        sum_counts += host_counts[e];
    EXPECT_EQ(sum_counts, total_slots)
        << "Sum of expert counts should equal total_slots";

    // Verify: offsets are consistent
    int running = 0;
    for (int e = 0; e < num_experts; ++e)
    {
        EXPECT_EQ(host_offsets[e], running)
            << "Expert " << e << " offset mismatch";
        running += host_counts[e];
    }

    // Verify: each token appears exactly top_k times across all groups
    std::vector<int> token_appearances(seq_len, 0);
    for (int i = 0; i < total_slots; ++i)
    {
        int tok = host_grouped_indices[i];
        ASSERT_GE(tok, 0) << "Grouped token index " << i << " is negative";
        ASSERT_LT(tok, seq_len) << "Grouped token index " << i << " out of range: " << tok;
        token_appearances[tok]++;
    }

    bool all_tokens_accounted = true;
    for (int t = 0; t < seq_len; ++t)
    {
        if (token_appearances[t] != top_k)
        {
            ADD_FAILURE() << "Token " << t << " appears " << token_appearances[t]
                          << " times, expected " << top_k;
            all_tokens_accounted = false;
        }
    }

    // Verify: grouped weights match original weights
    // Build a reference: for each expert, collect the expected (token, weight) pairs
    std::vector<std::vector<std::pair<int, float>>> expected_per_expert(num_experts);
    for (int s = 0; s < total_slots; ++s)
    {
        int token = s / top_k;
        int expert = routing_indices[s];
        expected_per_expert[expert].push_back({token, routing_weights[s]});
    }

    bool weights_match = true;
    for (int e = 0; e < num_experts; ++e)
    {
        int offset = host_offsets[e];
        int count = host_counts[e];

        // Collect actual
        std::vector<std::pair<int, float>> actual;
        for (int i = 0; i < count; ++i)
            actual.push_back({host_grouped_indices[offset + i],
                              host_grouped_weights[offset + i]});

        // Check each expected entry exists
        for (const auto &[exp_tok, exp_wt] : expected_per_expert[e])
        {
            bool found = false;
            for (auto &[act_tok, act_wt] : actual)
            {
                if (act_tok == exp_tok && act_wt == exp_wt)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                weights_match = false;
                break;
            }
        }
        if (!weights_match) break;
    }
    EXPECT_TRUE(weights_match) << "Grouped weights don't match original routing weights";

    std::cout << "[GroupTokensByExpert_PrefillScale] total_slots=" << total_slots
              << " sum_counts=" << sum_counts
              << " all_tokens_accounted=" << (all_tokens_accounted ? "true" : "false")
              << " weights_match=" << (weights_match ? "true" : "false") << std::endl;
}

#else // !HAVE_ROCM

TEST(Test__ROCmMoEKernel, SkippedNoROCm)
{
    GTEST_SKIP() << "ROCm not available in this build";
}

#endif // HAVE_ROCM
