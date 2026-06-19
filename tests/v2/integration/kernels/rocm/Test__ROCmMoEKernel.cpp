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
#include "utils/Assertions.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/ROCmWeightPacker.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/moe/MoEExpertWeightService.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#endif

#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/GpuPreparedGemmHarness.h"

#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#ifdef HAVE_ROCM
extern "C" bool hipMoE_group_tokens_small_float(
    const float *routing_indices, const float *routing_weights,
    int *expert_counts, int *expert_offsets,
    int *grouped_token_indices, int *original_to_grouped,
    float *grouped_weights,
    int *active_expert_ids,
    int total_slots, int num_experts, int top_k,
    int max_active_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_gate_logits_small_m(
    const float *hidden, const float *gate_weights, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_gate_logits_small_m_bf16_weights(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_gate_logits_single_token(
    const float *hidden, const float *gate_weights, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_gate_logits_single_token_bf16_weights(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream);

extern "C" bool hipMoE_softmax_topk(
    float *logits,
    int *expert_indices, float *expert_weights,
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    int device_idx, void *stream);
#endif

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{

    // ============================================================================
    // ROCm Availability Check
    // ============================================================================

    std::unique_ptr<DeviceWorkspaceManager> bindDefaultMoEWorkspace(
        ROCmMoEKernel &kernel,
        int max_seq_len = 64,
        int d_model = 2048,
        int intermediate = 512,
        int num_experts = 256,
        int top_k = 16)
    {
        auto reqs = MoEWorkspaceBuffers::rocmMoE(
            max_seq_len,
            d_model,
            intermediate,
            num_experts,
            top_k);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            reqs.total_bytes_with_alignment() + 4 * 1024 * 1024);
        EXPECT_TRUE(workspace->allocate(reqs));
        kernel.bindWorkspace(workspace.get());
        return workspace;
    }

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
        if (norm_a < 1e-30 && norm_b < 1e-30)
            return 1.0;
        if (norm_a < 1e-30 || norm_b < 1e-30)
            return 0.0;
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
        if (ref_sq < 1e-30)
            return err_sq < 1e-30 ? 0.0 : std::numeric_limits<double>::infinity();
        return std::sqrt(err_sq / ref_sq);
    }

    double maxAbsDiff(const float *actual, const float *reference, size_t count)
    {
        double max_diff = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double diff = std::fabs(static_cast<double>(actual[i]) - reference[i]);
            max_diff = std::max(max_diff, diff);
        }
        return max_diff;
    }

    struct StrictVerifierSimilarityMetrics
    {
        double cosine = 0.0;
        double relative_l2 = 0.0;
        double max_abs = 0.0;
        double min_row_cosine = 1.0;
        double max_row_relative_l2 = 0.0;
        double max_row_kl = 0.0;
        size_t worst_row = 0;
    };

    /**
     * @brief KL(reference || actual) after a stable row-wise softmax.
     *
     * Hidden-state vectors are not probabilities, but row-softmax KL is a useful
     * verifier guard: it catches shape changes in the largest coordinates that a
     * raw cosine can hide.  MTP verifier rows are short and near token decisions,
     * so a path that cannot satisfy this check is not decode-equivalent enough to
     * justify publishing accepted speculative state.
     */
    double rowSoftmaxKLDivergence(const float *actual, const float *reference, size_t row_width)
    {
        if (row_width == 0)
            return 0.0;

        double max_actual = -std::numeric_limits<double>::infinity();
        double max_reference = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < row_width; ++i)
        {
            max_actual = std::max(max_actual, static_cast<double>(actual[i]));
            max_reference = std::max(max_reference, static_cast<double>(reference[i]));
        }

        double sum_actual = 0.0;
        double sum_reference = 0.0;
        for (size_t i = 0; i < row_width; ++i)
        {
            sum_actual += std::exp(static_cast<double>(actual[i]) - max_actual);
            sum_reference += std::exp(static_cast<double>(reference[i]) - max_reference);
        }

        constexpr double kEps = 1.0e-30;
        double kl = 0.0;
        for (size_t i = 0; i < row_width; ++i)
        {
            const double p = std::exp(static_cast<double>(reference[i]) - max_reference) /
                             std::max(sum_reference, kEps);
            const double q = std::exp(static_cast<double>(actual[i]) - max_actual) /
                             std::max(sum_actual, kEps);
            kl += p * (std::log(std::max(p, kEps)) - std::log(std::max(q, kEps)));
        }
        return kl;
    }

    StrictVerifierSimilarityMetrics computeStrictVerifierSimilarity(
        const float *actual,
        const float *reference,
        size_t count,
        size_t row_width)
    {
        StrictVerifierSimilarityMetrics metrics;
        metrics.cosine = cosineSimilarity(actual, reference, count);
        metrics.relative_l2 = relativeL2Error(actual, reference, count);
        metrics.max_abs = maxAbsDiff(actual, reference, count);

        if (row_width == 0 || count % row_width != 0)
            row_width = count;
        const size_t rows = (row_width == 0) ? 0 : count / row_width;

        for (size_t row = 0; row < rows; ++row)
        {
            const float *row_actual = actual + row * row_width;
            const float *row_reference = reference + row * row_width;
            const double row_cosine = cosineSimilarity(row_actual, row_reference, row_width);
            const double row_rel_l2 = relativeL2Error(row_actual, row_reference, row_width);
            const double row_kl = rowSoftmaxKLDivergence(row_actual, row_reference, row_width);

            if (row_rel_l2 > metrics.max_row_relative_l2 ||
                row_kl > metrics.max_row_kl ||
                row_cosine < metrics.min_row_cosine)
            {
                metrics.worst_row = row;
            }
            metrics.min_row_cosine = std::min(metrics.min_row_cosine, row_cosine);
            metrics.max_row_relative_l2 = std::max(metrics.max_row_relative_l2, row_rel_l2);
            metrics.max_row_kl = std::max(metrics.max_row_kl, row_kl);
        }

        return metrics;
    }

    void expectStrictVerifierSimilarity(
        const char *label,
        const float *actual,
        const float *reference,
        size_t count,
        size_t row_width,
        double min_cosine = 0.9999,
        double max_relative_l2 = 0.006,
        double min_row_cosine = 0.9998,
        double max_row_relative_l2 = 0.008,
        double max_row_kl = 1.0e-4)
    {
        SCOPED_TRACE(label);
        ASSERT_NE(actual, nullptr);
        ASSERT_NE(reference, nullptr);
        ASSERT_GT(count, 0u);
        const auto metrics = computeStrictVerifierSimilarity(actual, reference, count, row_width);
        EXPECT_GE(metrics.cosine, min_cosine)
            << "relative_l2=" << metrics.relative_l2
            << " max_abs=" << metrics.max_abs
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.relative_l2, max_relative_l2)
            << "cosine=" << metrics.cosine
            << " max_abs=" << metrics.max_abs
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_GE(metrics.min_row_cosine, min_row_cosine)
            << "cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.max_row_relative_l2, max_row_relative_l2)
            << "cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_kl=" << metrics.max_row_kl
            << " worst_row=" << metrics.worst_row;
        EXPECT_LE(metrics.max_row_kl, max_row_kl)
            << "cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " min_row_cosine=" << metrics.min_row_cosine
            << " max_row_relative_l2=" << metrics.max_row_relative_l2
            << " worst_row=" << metrics.worst_row;
    }

    double klDivergenceNormalized(const float *actual, const float *reference, size_t count)
    {
        constexpr double kEps = 1.0e-30;
        double kl = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double p = std::max(static_cast<double>(reference[i]), kEps);
            const double q = std::max(static_cast<double>(actual[i]), kEps);
            kl += p * (std::log(p) - std::log(q));
        }
        return kl;
    }

    /**
     * @brief Enforce decode-equivalent top-k routing for verifier-sized M=2..4 batches.
     */
    void expectStrictTopKRowsEquivalent(
        const std::vector<float> &batched_indices,
        const std::vector<float> &batched_weights,
        const std::vector<float> &rowwise_indices,
        const std::vector<float> &rowwise_weights,
        int seq_len,
        int top_k)
    {
        ASSERT_EQ(batched_indices.size(), rowwise_indices.size());
        ASSERT_EQ(batched_weights.size(), rowwise_weights.size());
        for (size_t i = 0; i < batched_indices.size(); ++i)
            ASSERT_EQ(static_cast<int>(batched_indices[i]), static_cast<int>(rowwise_indices[i]))
                << "top-k index mismatch at flattened slot " << i;

        for (int row = 0; row < seq_len; ++row)
        {
            const float *actual = batched_weights.data() + static_cast<size_t>(row) * top_k;
            const float *reference = rowwise_weights.data() + static_cast<size_t>(row) * top_k;
            for (int k = 0; k < top_k; ++k)
                ASSERT_NEAR(actual[k], reference[k], 1.0e-4f)
                    << "row=" << row << " slot=" << k;
            EXPECT_GT(cosineSimilarity(actual, reference, static_cast<size_t>(top_k)), 0.999999)
                << "row=" << row;
            EXPECT_LT(relativeL2Error(actual, reference, static_cast<size_t>(top_k)), 1.0e-3)
                << "row=" << row;
            EXPECT_LT(klDivergenceNormalized(actual, reference, static_cast<size_t>(top_k)), 1.0e-5)
                << "row=" << row;
        }
    }

    class ScopedEnvOverride
    {
    public:
        ScopedEnvOverride(const char *name, const char *value)
            : name_(name)
        {
            const char *old_value = std::getenv(name);
            if (old_value)
            {
                had_old_value_ = true;
                old_value_ = old_value;
            }
            ::setenv(name_.c_str(), value, 1);
        }

        ~ScopedEnvOverride()
        {
            if (had_old_value_)
                ::setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                ::unsetenv(name_.c_str());
        }

        ScopedEnvOverride(const ScopedEnvOverride &) = delete;
        ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

    private:
        std::string name_;
        bool had_old_value_ = false;
        std::string old_value_;
    };

    /// @brief Give synthetic K-quant blocks nonzero min terms so tests cover asymmetric correction.
    void injectNonZeroKQuantMins(TensorBase *tensor)
    {
        ASSERT_NE(tensor, nullptr);
        if (tensor->native_type() == TensorType::Q4_K)
        {
            auto *blocks = reinterpret_cast<Q4_KBlock *>(tensor->raw_mutable_data());
            const size_t block_count = tensor->size_bytes() / sizeof(Q4_KBlock);
            for (size_t block = 0; block < block_count; ++block)
                blocks[block].dmin = fp32_to_fp16(0.006f + 0.001f * static_cast<float>(block % 7));
            return;
        }
        if (tensor->native_type() == TensorType::Q5_K)
        {
            auto *blocks = reinterpret_cast<Q5_KBlock *>(tensor->raw_mutable_data());
            const size_t block_count = tensor->size_bytes() / sizeof(Q5_KBlock);
            for (size_t block = 0; block < block_count; ++block)
                blocks[block].dmin = fp32_to_fp16(0.005f + 0.001f * static_cast<float>(block % 5));
        }
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
        for (auto &x : v)
            x = dist(gen);
    }

} // namespace

#ifdef HAVE_ROCM

namespace
{
    class ScopedROCmEnvOverride
    {
    public:
        ScopedROCmEnvOverride(const char *name, const char *value)
            : name_(name)
        {
            const char *existing = std::getenv(name_);
            if (existing)
            {
                had_original_ = true;
                original_ = existing;
            }
            ::setenv(name_, value, 1);
            mutableDebugEnv().rocm.reload();
        }

        ~ScopedROCmEnvOverride()
        {
            if (had_original_)
                ::setenv(name_, original_.c_str(), 1);
            else
                ::unsetenv(name_);
            mutableDebugEnv().rocm.reload();
        }

        ScopedROCmEnvOverride(const ScopedROCmEnvOverride &) = delete;
        ScopedROCmEnvOverride &operator=(const ScopedROCmEnvOverride &) = delete;

    private:
        const char *name_ = nullptr;
        bool had_original_ = false;
        std::string original_;
    };

    /// @brief Concatenates per-expert 2D tensors into the 3D GGUF-style parent tensor used by MoE views.
    template <typename TensorT>
    std::shared_ptr<TensorT> makeExpertParentTensor(
        const std::vector<std::unique_ptr<TensorBase>> &experts,
        const std::vector<size_t> &parent_shape)
    {
        std::vector<uint8_t> raw;
        for (const auto &expert : experts)
        {
            const auto *bytes = static_cast<const uint8_t *>(expert->raw_data());
            raw.insert(raw.end(), bytes, bytes + expert->size_bytes());
        }
        return std::make_shared<TensorT>(parent_shape, raw);
    }

    /// @brief Computes the routed MoE FFN output with dequantized FP32 weights as an independent reference.
    std::vector<float> computeCpuDequantMoEPrefillReference(
        const float *input,
        int seq_len,
        int d_model,
        int intermediate,
        int num_experts,
        int top_k,
        const std::vector<std::unique_ptr<TensorBase>> &gate_weight_tensors,
        const std::vector<std::unique_ptr<TensorBase>> &up_weight_tensors,
        const std::vector<std::unique_ptr<TensorBase>> &down_weight_tensors,
        const std::vector<int> &routing_indices,
        const std::vector<float> &routing_weights)
    {
        std::vector<std::vector<std::pair<int, float>>> routes_by_expert(static_cast<size_t>(num_experts));
        for (int token_idx = 0; token_idx < seq_len; ++token_idx)
        {
            for (int route_idx = 0; route_idx < top_k; ++route_idx)
            {
                const int slot_idx = token_idx * top_k + route_idx;
                routes_by_expert[static_cast<size_t>(routing_indices[static_cast<size_t>(slot_idx)])].push_back(
                    {token_idx, routing_weights[static_cast<size_t>(slot_idx)]});
            }
        }

        std::vector<float> output(static_cast<size_t>(seq_len) * d_model, 0.0f);
        std::vector<float> gate(static_cast<size_t>(intermediate));
        std::vector<float> up(static_cast<size_t>(intermediate));
        std::vector<float> swiglu(static_cast<size_t>(intermediate));

        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &routes = routes_by_expert[static_cast<size_t>(expert_id)];
            if (routes.empty())
                continue;

            const float *gate_weights = gate_weight_tensors[static_cast<size_t>(expert_id)]->data();
            const float *up_weights = up_weight_tensors[static_cast<size_t>(expert_id)]->data();
            const float *down_weights = down_weight_tensors[static_cast<size_t>(expert_id)]->data();

            for (const auto &[token_idx, route_weight] : routes)
            {
                const float *token_input = input + static_cast<size_t>(token_idx) * d_model;

                for (int hidden_idx = 0; hidden_idx < intermediate; ++hidden_idx)
                {
                    const float *gate_row = gate_weights + static_cast<size_t>(hidden_idx) * d_model;
                    const float *up_row = up_weights + static_cast<size_t>(hidden_idx) * d_model;
                    double gate_accum = 0.0;
                    double up_accum = 0.0;
                    for (int model_idx = 0; model_idx < d_model; ++model_idx)
                    {
                        const double activation = static_cast<double>(token_input[model_idx]);
                        gate_accum += activation * gate_row[model_idx];
                        up_accum += activation * up_row[model_idx];
                    }
                    gate[static_cast<size_t>(hidden_idx)] = static_cast<float>(gate_accum);
                    up[static_cast<size_t>(hidden_idx)] = static_cast<float>(up_accum);
                }

                for (int hidden_idx = 0; hidden_idx < intermediate; ++hidden_idx)
                {
                    const float gate_value = gate[static_cast<size_t>(hidden_idx)];
                    const float silu = gate_value / (1.0f + std::exp(-gate_value));
                    swiglu[static_cast<size_t>(hidden_idx)] = silu * up[static_cast<size_t>(hidden_idx)];
                }

                float *token_output = output.data() + static_cast<size_t>(token_idx) * d_model;
                for (int model_idx = 0; model_idx < d_model; ++model_idx)
                {
                    const float *down_row = down_weights + static_cast<size_t>(model_idx) * intermediate;
                    double down_accum = 0.0;
                    for (int hidden_idx = 0; hidden_idx < intermediate; ++hidden_idx)
                        down_accum += static_cast<double>(swiglu[static_cast<size_t>(hidden_idx)]) * down_row[hidden_idx];
                    token_output[model_idx] += route_weight * static_cast<float>(down_accum);
                }
            }
        }

        return output;
    }
}

TEST(Test__ROCmMoEKernel, UploadGroupedDescriptorTablesRejectInvalidDescriptors)
{
    SKIP_IF_NO_ROCM();

    const int d_model = 128;
    const int intermediate = 128;
    const int num_experts = 4;

    auto make_down_desc = [](int rows, int cols)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = reinterpret_cast<const uint8_t *>(0x1000);
        desc.scales = reinterpret_cast<const void *>(0x2000);
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = 0;
        return desc;
    };

    auto make_gateup_desc = [](int rows, int cols)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = reinterpret_cast<const uint8_t *>(0x3000);
        desc.scales = reinterpret_cast<const void *>(0x4000);
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = 0;
        return desc;
    };

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<DeviceNativeVNNIMatrixDesc> down_descs;
    down_descs.reserve(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        down_descs.push_back(make_down_desc(d_model, intermediate));
    down_descs[2].payload = nullptr;
    EXPECT_EQ(moe_kernel.uploadGroupedExpertDownDescriptorTable(
                  down_descs.data(), num_experts, d_model, intermediate),
              -1);

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs;
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_descs.push_back(make_gateup_desc(intermediate, d_model));
        up_descs.push_back(make_gateup_desc(intermediate, d_model));
    }
    up_descs[1].scales = nullptr;
    EXPECT_EQ(moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
                  gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate),
              -1);
}

TEST(Test__ROCmMoEKernel, UploadGroupedDescriptorTablesAcceptAllNativeVNNICodebooks)
{
    SKIP_IF_NO_ROCM();

    struct CodebookCase
    {
        uint8_t codebook_id;
        bool requires_mins;
        bool requires_emins;
    };

    const std::vector<CodebookCase> codebooks = {
        {0, false, false},  {4, false, false},  {5, true, false},
        {6, false, false},  {7, true, false},   {8, true, false},
        {9, true, false},   {10, true, true},   {11, false, false},
        {12, false, false}, {13, true, false},  {14, true, false},
        {15, false, false}, {16, true, false},  {17, true, false},
        {19, false, false},
    };

    const int d_model = 128;
    const int intermediate = 128;
    const int num_experts = 2;

    auto make_desc = [](int rows, int cols, const CodebookCase &codebook, std::uintptr_t base)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = reinterpret_cast<const uint8_t *>(base + 0x1000);
        desc.scales = reinterpret_cast<const void *>(base + 0x2000);
        desc.mins = codebook.requires_mins ? reinterpret_cast<const void *>(base + 0x3000) : nullptr;
        desc.emins = codebook.requires_emins ? reinterpret_cast<const void *>(base + 0x4000) : nullptr;
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = codebook.codebook_id;
        return desc;
    };

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    for (size_t case_idx = 0; case_idx < codebooks.size(); ++case_idx)
    {
        const auto &codebook = codebooks[case_idx];
        const std::uintptr_t base = 0x100000 + case_idx * 0x10000;

        std::vector<DeviceNativeVNNIMatrixDesc> down_descs;
        std::vector<DeviceNativeVNNIMatrixDesc> gate_descs;
        std::vector<DeviceNativeVNNIMatrixDesc> up_descs;
        down_descs.reserve(num_experts);
        gate_descs.reserve(num_experts);
        up_descs.reserve(num_experts);

        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const std::uintptr_t expert_base = base + static_cast<std::uintptr_t>(expert_id) * 0x1000;
            down_descs.push_back(make_desc(d_model, intermediate, codebook, expert_base));
            gate_descs.push_back(make_desc(intermediate, d_model, codebook, expert_base + 0x5000));
            up_descs.push_back(make_desc(intermediate, d_model, codebook, expert_base + 0xA000));
        }

        EXPECT_GE(moe_kernel.uploadGroupedExpertDownDescriptorTable(
                      down_descs.data(), num_experts, d_model, intermediate),
                  0)
            << "down descriptor rejected codebook " << static_cast<int>(codebook.codebook_id);
        EXPECT_GE(moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
                      gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate),
                  0)
            << "gate/up descriptor rejected codebook " << static_cast<int>(codebook.codebook_id);
    }

    CodebookCase q2_k_missing_emins{10, true, false};
    std::vector<DeviceNativeVNNIMatrixDesc> missing_emins;
    missing_emins.reserve(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        const std::uintptr_t expert_base = 0x500000 + static_cast<std::uintptr_t>(expert_id) * 0x1000;
        missing_emins.push_back(make_desc(d_model, intermediate, q2_k_missing_emins, expert_base));
    }
    EXPECT_EQ(moe_kernel.uploadGroupedExpertDownDescriptorTable(
                  missing_emins.data(), num_experts, d_model, intermediate),
              -1);

    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);

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
    expectStrictVerifierSimilarity(
        "decode router logits should match CPU routing row-by-row",
        gpu_result.router_logits.data(),
        cpu_result.router_logits.data(),
        gpu_result.router_logits.size(),
        static_cast<size_t>(num_experts));

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
    EXPECT_GE(weights_cosine, 0.999)
        << "Expert weights cosine similarity too low: " << weights_cosine;

    std::cout << "[Route_DecodeSmall] logits_cosine=" << std::fixed << std::setprecision(6)
              << logits_cosine
              << " expert_match=" << matching_experts << "/" << cpu_result.expert_indices.size()
              << " weights_cosine=" << weights_cosine << std::endl;
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectRuntimeStateUpdatesTopKAndHistogram)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 256;
    const int num_experts = 16;
    const int top_k = 4;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    fillRandom(hidden_host, -1.0f, 1.0f, 7201);
    fillRandom(gate_host, -0.1f, 0.1f, 7202);
    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices->ensureOnDevice(device));
    ASSERT_TRUE(output_weights->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime{};
    host_runtime.active_bank = 0;
    host_runtime.active_epoch = 1;
    host_runtime.expert_count = static_cast<uint32_t>(num_experts);
    host_runtime.top_k = static_cast<uint32_t>(top_k);
    host_runtime.banks[0].epoch = 1;
    host_runtime.banks[0].expert_count = static_cast<uint32_t>(num_experts);

    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(device_runtime, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    EXPECT_FALSE(gpu_kernel.decodeRouteSelect(
        nullptr,
        hidden.get(), gate_weights.get(),
        d_model, num_experts, top_k,
        true,
        output_indices.get(), output_weights.get(),
        true, true));

    // Runtime-table semantic validation belongs to the graph-build/stage setup
    // boundary.  decodeRouteSelect() is part of the captured hot path, so this
    // test deliberately avoids expecting a D2H validation pass here.

    ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
        device_runtime,
        hidden.get(), gate_weights.get(),
        d_model, num_experts, top_k,
        true,
        output_indices.get(), output_weights.get(),
        true, true));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after{};
    ASSERT_EQ(hipMemcpy(&after, device_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(device_runtime), hipSuccess);

    output_indices->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices = output_indices->data();
    const float *legacy_weights = output_weights->data();

    uint64_t histogram_sum = 0;
    for (int expert = 0; expert < num_experts; ++expert)
        histogram_sum += after.decode_histogram[expert];
    EXPECT_EQ(histogram_sum, static_cast<uint64_t>(top_k));

    for (int k = 0; k < top_k; ++k)
    {
        const int expert_id = after.topk_expert_ids[k];
        EXPECT_GE(expert_id, 0);
        EXPECT_LT(expert_id, num_experts);
        EXPECT_FLOAT_EQ(legacy_indices[k], static_cast<float>(expert_id));
        EXPECT_NEAR(legacy_weights[k], after.topk_weights[k], 1e-6f);
        EXPECT_GT(after.topk_weights[k], 0.0f);
        ASSERT_GE(expert_id, 0);
        ASSERT_LT(expert_id, num_experts);
        EXPECT_GE(after.decode_histogram[expert_id], 1u);
    }

    float weight_sum = 0.0f;
    for (int k = 0; k < top_k; ++k)
        weight_sum += after.topk_weights[k];
    EXPECT_NEAR(weight_sum, 1.0f, 1e-4f);
}

TEST(Test__ROCmMoEKernel, TokenRowPublicationTopK2SurvivesSnapshotSync)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int kRows = 2;
    constexpr int kTopK = 2;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    static_cast<ITensorKernel &>(gpu_kernel).setGPUStream(stream);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);

    auto full_routes = TestTensorFactory::createFP32({kRows, kTopK});
    auto row0 = TestTensorFactory::createFP32({kTopK});
    auto row1 = TestTensorFactory::createFP32({kTopK});
    auto copied = TestTensorFactory::createFP32({kTopK});

    std::fill(full_routes->mutable_data(),
              full_routes->mutable_data() + kRows * kTopK,
              -99.0f);
    row0->mutable_data()[0] = 7.0f;
    row0->mutable_data()[1] = 11.0f;
    row1->mutable_data()[0] = 13.0f;
    row1->mutable_data()[1] = 17.0f;
    std::fill(copied->mutable_data(),
              copied->mutable_data() + kTopK,
              -1.0f);

    ASSERT_TRUE(full_routes->ensureOnDevice(device));
    ASSERT_TRUE(row0->ensureOnDevice(device));
    ASSERT_TRUE(row1->ensureOnDevice(device));
    ASSERT_TRUE(copied->ensureOnDevice(device));

    /*
     * MoERoutingStage publishes verifier routing one row at a time, then the
     * snapshot build refreshes the host mirror before MoEExpertComputeStage
     * gathers rows back on device. This regression locks that ownership handoff
     * down for top_k=2, the Qwen3.6 MoE shape that exposed stale/corrupt route
     * rows during ROCm verifier replay.
     */
    ASSERT_TRUE(gpu_kernel.writeTokenRowToTensor(full_routes.get(), row0.get(), 0, kTopK));
    ASSERT_TRUE(gpu_kernel.writeTokenRowToTensor(full_routes.get(), row1.get(), 1, kTopK));

    ASSERT_TRUE(full_routes->ensureOnHost(stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    const float *published = full_routes->data();
    ASSERT_NE(published, nullptr);
    EXPECT_FLOAT_EQ(published[0], 7.0f);
    EXPECT_FLOAT_EQ(published[1], 11.0f);
    EXPECT_FLOAT_EQ(published[2], 13.0f);
    EXPECT_FLOAT_EQ(published[3], 17.0f);

    ASSERT_TRUE(gpu_kernel.copyTokenRowFromTensor(full_routes.get(), copied.get(), 0, kTopK));
    ASSERT_TRUE(copied->ensureOnHost(stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    const float *copied_row = copied->data();
    ASSERT_NE(copied_row, nullptr);
    EXPECT_FLOAT_EQ(copied_row[0], 7.0f);
    EXPECT_FLOAT_EQ(copied_row[1], 11.0f);

    ASSERT_TRUE(gpu_kernel.copyTokenRowFromTensor(full_routes.get(), copied.get(), 1, kTopK));
    ASSERT_TRUE(copied->ensureOnHost(stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    copied_row = copied->data();
    ASSERT_NE(copied_row, nullptr);
    EXPECT_FLOAT_EQ(copied_row[0], 13.0f);
    EXPECT_FLOAT_EQ(copied_row[1], 17.0f);

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, DecodeRuntimeHistogramSyncMatchesHostRecordAcrossTokensAndLayers)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int num_layers = 2;
    const int d_model = 128;
    const int num_experts = 16;
    const int top_k = 4;
    const int tokens = 5;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    fillRandom(gate_host, -0.1f, 0.1f, 8102);
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices->ensureOnDevice(device));
    ASSERT_TRUE(output_weights->ensureOnDevice(device));

    DeviceMoERuntimeTable::Config table_config;
    table_config.device_id = device;
    table_config.num_layers = num_layers;
    table_config.num_experts = num_experts;
    table_config.top_k = top_k;
    table_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(table_config);

    auto routing_only_update = [&](uint32_t epoch)
    {
        MoEPlacementUpdate update;
        update.epoch = epoch;
        update.expert_count = static_cast<uint32_t>(num_experts);
        update.experts.resize(static_cast<size_t>(num_experts));
        update.local_compute_mask.assign(static_cast<size_t>(num_experts), 0u);
        update.replica_role.assign(static_cast<size_t>(num_experts), 0u);
        return update;
    };

    for (int layer = 0; layer < num_layers; ++layer)
    {
        ASSERT_TRUE(runtime_table.prepareInactiveBank(layer, routing_only_update(1)));
        ASSERT_TRUE(runtime_table.flipActiveBank(layer, 1, nullptr));
    }

    DecodeExpertHistogramConfig hist_config;
    hist_config.num_layers = num_layers;
    hist_config.num_experts = num_experts;
    hist_config.top_k = top_k;
    hist_config.window_size = 32;
    hist_config.sockets = {DeviceId(DeviceType::CPU, 0), DeviceId(DeviceType::CPU, 1)};
    hist_config.expert_to_socket.resize(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
        hist_config.expert_to_socket[expert] = expert % 2;
    DecodeExpertHistogram host_record(hist_config);
    DecodeExpertHistogram runtime_merged(hist_config);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    for (int token = 0; token < tokens; ++token)
    {
        for (int layer = 0; layer < num_layers; ++layer)
        {
            fillRandom(hidden_host, -1.0f, 1.0f, 8200 + token * 17 + layer);
            std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
            ASSERT_TRUE(hidden->ensureOnDevice(device));

            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_table.deviceLayerState(layer),
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices.get(), output_weights.get(),
                true, true));

            output_indices->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            output_weights->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            const float *legacy_indices = output_indices->data();
            const float *legacy_weights = output_weights->data();
            int host_indices[top_k];
            float host_weights[top_k];
            for (int k = 0; k < top_k; ++k)
            {
                host_indices[k] = static_cast<int>(legacy_indices[k]);
                host_weights[k] = legacy_weights[k];
            }
            host_record.record(layer, host_indices, host_weights, top_k);
            runtime_merged.recordTokenBoundary(layer);
        }
    }

    ASSERT_TRUE(runtime_table.syncDecodeHistogramToHost(runtime_merged));

    for (int layer = 0; layer < num_layers; ++layer)
        EXPECT_EQ(runtime_merged.layerHistogram(layer), host_record.layerHistogram(layer));
    EXPECT_EQ(runtime_merged.windowTokenCount(), host_record.windowTokenCount());

    for (int layer = 0; layer < num_layers; ++layer)
    {
        const auto &state = runtime_table.hostLayerState(layer);
        for (int expert = 0; expert < num_experts; ++expert)
            EXPECT_EQ(state.decode_histogram[expert], 0u);
    }
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectWaveTopKMatchesDefaultRuntimeTopK)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 512;
    const int num_experts = 64;
    const int top_k = 8;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_default = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_default = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices_wave = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_wave = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    fillRandom(hidden_host, -1.0f, 1.0f, 9401);
    fillRandom(gate_host, -0.2f, 0.2f, 9402);
    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_default->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_default->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_wave->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_wave->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime{};
    host_runtime.active_bank = 0;
    host_runtime.active_epoch = 1;
    host_runtime.expert_count = static_cast<uint32_t>(num_experts);
    host_runtime.top_k = static_cast<uint32_t>(top_k);
    host_runtime.banks[0].epoch = 1;
    host_runtime.banks[0].expert_count = static_cast<uint32_t>(num_experts);

    DeviceMoELayerRuntime *runtime_default = nullptr;
    DeviceMoELayerRuntime *runtime_wave = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_default), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_wave), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_default, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_wave, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "0");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_default,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_default.get(), output_weights_default.get(),
            true, true));
    }
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "1");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_wave,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_wave.get(), output_weights_wave.get(),
            true, true));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_default{};
    DeviceMoELayerRuntime after_wave{};
    ASSERT_EQ(hipMemcpy(&after_default, runtime_default, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&after_wave, runtime_wave, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_default), hipSuccess);
    ASSERT_EQ(hipFree(runtime_wave), hipSuccess);

    output_indices_default->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_default->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices_wave->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_wave->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_default = output_indices_default->data();
    const float *legacy_weights_default = output_weights_default->data();
    const float *legacy_indices_wave = output_indices_wave->data();
    const float *legacy_weights_wave = output_weights_wave->data();

    double max_weight_diff = 0.0;
    for (int slot = 0; slot < top_k; ++slot)
    {
        EXPECT_EQ(after_wave.topk_expert_ids[slot], after_default.topk_expert_ids[slot]);
        EXPECT_FLOAT_EQ(legacy_indices_wave[slot], legacy_indices_default[slot]);
        EXPECT_FLOAT_EQ(legacy_indices_wave[slot], static_cast<float>(after_wave.topk_expert_ids[slot]));
        EXPECT_NEAR(after_wave.topk_weights[slot], after_default.topk_weights[slot], 2e-6f);
        EXPECT_NEAR(legacy_weights_wave[slot], legacy_weights_default[slot], 2e-6f);
        max_weight_diff = std::max(
            max_weight_diff,
            std::fabs(static_cast<double>(after_wave.topk_weights[slot]) -
                      static_cast<double>(after_default.topk_weights[slot])));
    }

    uint64_t histogram_sum_default = 0;
    uint64_t histogram_sum_wave = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        histogram_sum_default += after_default.decode_histogram[expert];
        histogram_sum_wave += after_wave.decode_histogram[expert];
    }
    EXPECT_EQ(histogram_sum_default, static_cast<uint64_t>(top_k));
    EXPECT_EQ(histogram_sum_wave, static_cast<uint64_t>(top_k));

    std::cout << "[DecodeRouteSelectWaveTopKMatchesDefaultRuntimeTopK] max_weight_diff="
              << max_weight_diff << "\n";
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectWaveTopKMatchesDefaultRuntimeTopKQwen35Shape)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 512;
    const int num_experts = 256;
    const int top_k = 8;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_default = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_default = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices_wave = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_wave = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    fillRandom(hidden_host, -1.0f, 1.0f, 9501);
    fillRandom(gate_host, -0.2f, 0.2f, 9502);
    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_default->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_default->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_wave->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_wave->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime{};
    host_runtime.active_bank = 0;
    host_runtime.active_epoch = 1;
    host_runtime.expert_count = static_cast<uint32_t>(num_experts);
    host_runtime.top_k = static_cast<uint32_t>(top_k);
    host_runtime.banks[0].epoch = 1;
    host_runtime.banks[0].expert_count = static_cast<uint32_t>(num_experts);

    DeviceMoELayerRuntime *runtime_default = nullptr;
    DeviceMoELayerRuntime *runtime_wave = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_default), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_wave), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_default, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_wave, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "0");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_default,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_default.get(), output_weights_default.get(),
            true, true));
    }
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "1");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_wave,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_wave.get(), output_weights_wave.get(),
            true, true));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_default{};
    DeviceMoELayerRuntime after_wave{};
    ASSERT_EQ(hipMemcpy(&after_default, runtime_default, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&after_wave, runtime_wave, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_default), hipSuccess);
    ASSERT_EQ(hipFree(runtime_wave), hipSuccess);

    output_indices_default->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_default->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices_wave->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_wave->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_default = output_indices_default->data();
    const float *legacy_weights_default = output_weights_default->data();
    const float *legacy_indices_wave = output_indices_wave->data();
    const float *legacy_weights_wave = output_weights_wave->data();

    double max_weight_diff = 0.0;
    for (int slot = 0; slot < top_k; ++slot)
    {
        EXPECT_EQ(after_wave.topk_expert_ids[slot], after_default.topk_expert_ids[slot]);
        EXPECT_FLOAT_EQ(legacy_indices_wave[slot], legacy_indices_default[slot]);
        EXPECT_FLOAT_EQ(legacy_indices_wave[slot], static_cast<float>(after_wave.topk_expert_ids[slot]));
        EXPECT_NEAR(after_wave.topk_weights[slot], after_default.topk_weights[slot], 2e-6f);
        EXPECT_NEAR(legacy_weights_wave[slot], legacy_weights_default[slot], 2e-6f);
        max_weight_diff = std::max(
            max_weight_diff,
            std::fabs(static_cast<double>(after_wave.topk_weights[slot]) -
                      static_cast<double>(after_default.topk_weights[slot])));
    }

    uint64_t histogram_sum_default = 0;
    uint64_t histogram_sum_wave = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        histogram_sum_default += after_default.decode_histogram[expert];
        histogram_sum_wave += after_wave.decode_histogram[expert];
    }
    EXPECT_EQ(histogram_sum_default, static_cast<uint64_t>(top_k));
    EXPECT_EQ(histogram_sum_wave, static_cast<uint64_t>(top_k));

    std::cout << "[DecodeRouteSelectWaveTopKMatchesDefaultRuntimeTopKQwen35Shape] max_weight_diff="
              << max_weight_diff << "\n";
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectFP16RouterMatchesFP32TopK)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 1024;
    const int num_experts = 32;
    const int top_k = 6;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices_fp16 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_fp16 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    fillRandom(hidden_host, -1.0f, 1.0f, 9301);
    const float hidden_norm_sq = std::inner_product(
        hidden_host.begin(), hidden_host.end(), hidden_host.begin(), 0.0f);

    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    std::mt19937 gen(9302);
    std::uniform_real_distribution<float> noise(-1.0e-5f, 1.0e-5f);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const float expert_score = 2.0f - 0.05f * static_cast<float>(expert);
        for (int i = 0; i < d_model; ++i)
        {
            gate_host[static_cast<size_t>(expert) * d_model + i] =
                expert_score * hidden_host[i] / hidden_norm_sq + noise(gen);
        }
    }

    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_fp16->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_fp16->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime{};
    host_runtime.active_bank = 0;
    host_runtime.active_epoch = 1;
    host_runtime.expert_count = static_cast<uint32_t>(num_experts);
    host_runtime.top_k = static_cast<uint32_t>(top_k);
    host_runtime.banks[0].epoch = 1;
    host_runtime.banks[0].expert_count = static_cast<uint32_t>(num_experts);

    DeviceMoELayerRuntime *runtime_fp32 = nullptr;
    DeviceMoELayerRuntime *runtime_fp16 = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp32), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp16), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_fp32, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_fp16, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        {
            ScopedROCmEnvOverride router_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_fp32,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_fp32.get(), output_weights_fp32.get(),
                true, false));
        }
        {
            ScopedROCmEnvOverride router_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "1");
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_fp16,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_fp16.get(), output_weights_fp16.get(),
                true, false));
        }
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_fp32{};
    DeviceMoELayerRuntime after_fp16{};
    ASSERT_EQ(hipMemcpy(&after_fp32, runtime_fp32, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&after_fp16, runtime_fp16, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_fp32), hipSuccess);
    ASSERT_EQ(hipFree(runtime_fp16), hipSuccess);

    output_indices_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices_fp16->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_fp16->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_fp32 = output_indices_fp32->data();
    const float *legacy_weights_fp32 = output_weights_fp32->data();
    const float *legacy_indices_fp16 = output_indices_fp16->data();
    const float *legacy_weights_fp16 = output_weights_fp16->data();

    float max_weight_diff = 0.0f;
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_EQ(after_fp16.topk_expert_ids[k], after_fp32.topk_expert_ids[k]);
        EXPECT_FLOAT_EQ(legacy_indices_fp32[k], static_cast<float>(after_fp32.topk_expert_ids[k]));
        EXPECT_FLOAT_EQ(legacy_indices_fp16[k], static_cast<float>(after_fp16.topk_expert_ids[k]));
        const float diff = std::fabs(after_fp16.topk_weights[k] - after_fp32.topk_weights[k]);
        max_weight_diff = std::max(max_weight_diff, diff);
        EXPECT_NEAR(after_fp16.topk_weights[k], after_fp32.topk_weights[k], 2.0e-3f);
        EXPECT_NEAR(legacy_weights_fp16[k], after_fp16.topk_weights[k], 1.0e-6f);
        EXPECT_NEAR(legacy_weights_fp32[k], after_fp32.topk_weights[k], 1.0e-6f);
    }

    std::cout << "[DecodeRouteSelectFP16RouterMatchesFP32TopK] max_weight_diff="
              << std::fixed << std::setprecision(8) << max_weight_diff << std::endl;
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectBF16RouterMatchesFP32TopK)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 1024;
    const int num_experts = 32;
    const int top_k = 6;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto gate_weights_bf16 = TestTensorFactory::createBF16({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_indices_bf16 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_bf16 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    fillRandom(hidden_host, -1.0f, 1.0f, 9361);
    const float hidden_norm_sq = std::inner_product(
        hidden_host.begin(), hidden_host.end(), hidden_host.begin(), 0.0f);

    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    std::mt19937 gen(9362);
    std::uniform_real_distribution<float> noise(-1.0e-5f, 1.0e-5f);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const float expert_score = 2.0f - 0.05f * static_cast<float>(expert);
        for (int i = 0; i < d_model; ++i)
        {
            gate_host[static_cast<size_t>(expert) * d_model + i] =
                expert_score * hidden_host[i] / hidden_norm_sq + noise(gen);
        }
    }

    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights_fp32->mutable_data());
    gate_weights_bf16->from_fp32(gate_host.data(), gate_host.size());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights_fp32->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights_bf16->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_bf16->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_bf16->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime{};
    host_runtime.active_bank = 0;
    host_runtime.active_epoch = 1;
    host_runtime.expert_count = static_cast<uint32_t>(num_experts);
    host_runtime.top_k = static_cast<uint32_t>(top_k);
    host_runtime.banks[0].epoch = 1;
    host_runtime.banks[0].expert_count = static_cast<uint32_t>(num_experts);

    DeviceMoELayerRuntime *runtime_fp32 = nullptr;
    DeviceMoELayerRuntime *runtime_bf16 = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp32), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_bf16), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_fp32, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_bf16, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ScopedROCmEnvOverride wave_env("LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "0");

        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_fp32,
            hidden.get(), gate_weights_fp32.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_fp32.get(), output_weights_fp32.get(),
            true, false));
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_bf16,
            hidden.get(), gate_weights_bf16.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_bf16.get(), output_weights_bf16.get(),
            true, false));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_fp32{};
    DeviceMoELayerRuntime after_bf16{};
    ASSERT_EQ(hipMemcpy(&after_fp32, runtime_fp32, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&after_bf16, runtime_bf16, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_fp32), hipSuccess);
    ASSERT_EQ(hipFree(runtime_bf16), hipSuccess);

    output_indices_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_indices_bf16->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_bf16->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_fp32 = output_indices_fp32->data();
    const float *legacy_weights_fp32 = output_weights_fp32->data();
    const float *legacy_indices_bf16 = output_indices_bf16->data();
    const float *legacy_weights_bf16 = output_weights_bf16->data();

    float max_weight_diff = 0.0f;
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_EQ(after_bf16.topk_expert_ids[k], after_fp32.topk_expert_ids[k]);
        EXPECT_FLOAT_EQ(legacy_indices_fp32[k], static_cast<float>(after_fp32.topk_expert_ids[k]));
        EXPECT_FLOAT_EQ(legacy_indices_bf16[k], static_cast<float>(after_bf16.topk_expert_ids[k]));
        const float diff = std::fabs(after_bf16.topk_weights[k] - after_fp32.topk_weights[k]);
        max_weight_diff = std::max(max_weight_diff, diff);
        EXPECT_NEAR(after_bf16.topk_weights[k], after_fp32.topk_weights[k], 5.0e-3f);
        EXPECT_NEAR(legacy_weights_bf16[k], after_bf16.topk_weights[k], 1.0e-6f);
        EXPECT_NEAR(legacy_weights_fp32[k], after_fp32.topk_weights[k], 1.0e-6f);
    }

    std::cout << "[DecodeRouteSelectBF16RouterMatchesFP32TopK] max_weight_diff="
              << std::fixed << std::setprecision(8) << max_weight_diff << std::endl;
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectKPartRouterMatchesFP32TopK)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 2048;
    const int num_experts = 64;
    const int top_k = 8;

    auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    auto output_indices_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
    auto output_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

    std::vector<float> hidden_host(static_cast<size_t>(d_model));
    fillRandom(hidden_host, -1.0f, 1.0f, 9401);
    const float hidden_norm_sq = std::inner_product(
        hidden_host.begin(), hidden_host.end(), hidden_host.begin(), 0.0f);

    std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
    std::mt19937 gen(9402);
    std::uniform_real_distribution<float> noise(-1.0e-6f, 1.0e-6f);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const float expert_score = 3.0f - 0.05f * static_cast<float>(expert);
        for (int i = 0; i < d_model; ++i)
        {
            gate_host[static_cast<size_t>(expert) * d_model + i] =
                expert_score * hidden_host[i] / hidden_norm_sq + noise(gen);
        }
    }

    std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
    std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(output_indices_fp32->ensureOnDevice(device));
    ASSERT_TRUE(output_weights_fp32->ensureOnDevice(device));

    DeviceMoELayerRuntime host_runtime{};
    host_runtime.active_bank = 0;
    host_runtime.active_epoch = 1;
    host_runtime.expert_count = static_cast<uint32_t>(num_experts);
    host_runtime.top_k = static_cast<uint32_t>(top_k);
    host_runtime.banks[0].epoch = 1;
    host_runtime.banks[0].expert_count = static_cast<uint32_t>(num_experts);

    DeviceMoELayerRuntime *runtime_fp32 = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp32), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(runtime_fp32, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    auto expectHistogramMatchesTopK = [&](const DeviceMoELayerRuntime &runtime)
    {
        std::vector<uint64_t> expected(static_cast<size_t>(num_experts), 0);
        for (int k = 0; k < top_k; ++k)
        {
            const int expert_id = runtime.topk_expert_ids[k];
            EXPECT_GE(expert_id, 0);
            EXPECT_LT(expert_id, num_experts);
            if (expert_id >= 0 && expert_id < num_experts)
            {
                ++expected[static_cast<size_t>(expert_id)];
            }
        }
        for (int expert = 0; expert < num_experts; ++expert)
        {
            EXPECT_EQ(runtime.decode_histogram[expert], expected[static_cast<size_t>(expert)])
                << "expert=" << expert;
        }
    };

    {
        ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
        ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
        ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
        ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
        ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
            runtime_fp32,
            hidden.get(), gate_weights.get(),
            d_model, num_experts, top_k,
            true,
            output_indices_fp32.get(), output_weights_fp32.get(),
            true, true));
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoELayerRuntime after_fp32{};
    ASSERT_EQ(hipMemcpy(&after_fp32, runtime_fp32, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipFree(runtime_fp32), hipSuccess);

    output_indices_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_weights_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *legacy_indices_fp32 = output_indices_fp32->data();
    const float *legacy_weights_fp32 = output_weights_fp32->data();
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_FLOAT_EQ(legacy_indices_fp32[k], static_cast<float>(after_fp32.topk_expert_ids[k]));
        EXPECT_NEAR(legacy_weights_fp32[k], after_fp32.topk_weights[k], 1.0e-6f);
    }
    expectHistogramMatchesTopK(after_fp32);

    for (int k_partitions : {2, 4, 8, 16})
    {
        auto output_indices_kpart = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        auto output_weights_kpart = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        ASSERT_TRUE(output_indices_kpart->ensureOnDevice(device));
        ASSERT_TRUE(output_weights_kpart->ensureOnDevice(device));

        DeviceMoELayerRuntime *runtime_kpart = nullptr;
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_kpart), sizeof(DeviceMoELayerRuntime)), hipSuccess);
        ASSERT_EQ(hipMemcpy(runtime_kpart, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

        const std::string kparts_value = std::to_string(k_partitions);
        {
            ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
            ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
            ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
            ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
            ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "1");
            ScopedROCmEnvOverride kparts_env("LLAMINAR_ROCM_MOE_ROUTER_KPARTS", kparts_value.c_str());
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_kpart,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_kpart.get(), output_weights_kpart.get(),
                true, true));
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        DeviceMoELayerRuntime after_kpart{};
        ASSERT_EQ(hipMemcpy(&after_kpart, runtime_kpart, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipFree(runtime_kpart), hipSuccess);

        output_indices_kpart->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_weights_kpart->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *legacy_indices_kpart = output_indices_kpart->data();
        const float *legacy_weights_kpart = output_weights_kpart->data();

        float max_weight_diff = 0.0f;
        for (int k = 0; k < top_k; ++k)
        {
            EXPECT_EQ(after_kpart.topk_expert_ids[k], after_fp32.topk_expert_ids[k])
                << "k_partitions=" << k_partitions << " slot=" << k;
            EXPECT_FLOAT_EQ(legacy_indices_kpart[k], static_cast<float>(after_kpart.topk_expert_ids[k]));
            const float diff = std::fabs(after_kpart.topk_weights[k] - after_fp32.topk_weights[k]);
            max_weight_diff = std::max(max_weight_diff, diff);
            EXPECT_NEAR(after_kpart.topk_weights[k], after_fp32.topk_weights[k], 1.0e-4f)
                << "k_partitions=" << k_partitions << " slot=" << k;
            EXPECT_NEAR(legacy_weights_kpart[k], after_kpart.topk_weights[k], 1.0e-6f);
        }
        expectHistogramMatchesTopK(after_kpart);

        std::cout << "[DecodeRouteSelectKPartRouterMatchesFP32TopK] kparts="
                  << k_partitions << " max_weight_diff="
                  << std::fixed << std::setprecision(8) << max_weight_diff << std::endl;
    }
}

TEST(Test__ROCmMoEKernel, DecodeRouteSelectQ8RouterMatchesFP32TopKAcrossSeeds)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int d_model = 2048;
    const int num_experts = 64;
    const int top_k = 8;
    const int case_count = 8;

    int drift_cases = 0;
    int drift_slots = 0;
    float max_weight_diff = 0.0f;

    for (int case_idx = 0; case_idx < case_count; ++case_idx)
    {
        auto hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
        auto gate_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
        auto output_indices_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        auto output_weights_fp32 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        auto output_indices_q8 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});
        auto output_weights_q8 = TestTensorFactory::createFP32({static_cast<size_t>(top_k), 1});

        std::vector<float> hidden_host(static_cast<size_t>(d_model));
        fillRandom(hidden_host, -1.0f, 1.0f, static_cast<unsigned>(9501 + case_idx));
        const float hidden_norm_sq = std::inner_product(
            hidden_host.begin(), hidden_host.end(), hidden_host.begin(), 0.0f);

        std::vector<float> gate_host(static_cast<size_t>(num_experts) * d_model);
        std::mt19937 gen(static_cast<unsigned>(9601 + case_idx));
        std::uniform_real_distribution<float> noise(-2.0e-4f, 2.0e-4f);
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const float expert_score = 5.0f - 0.16f * static_cast<float>(expert);
            for (int i = 0; i < d_model; ++i)
            {
                gate_host[static_cast<size_t>(expert) * d_model + i] =
                    expert_score * hidden_host[i] / hidden_norm_sq + noise(gen);
            }
        }

        std::copy(hidden_host.begin(), hidden_host.end(), hidden->mutable_data());
        std::copy(gate_host.begin(), gate_host.end(), gate_weights->mutable_data());

        ASSERT_TRUE(hidden->ensureOnDevice(device));
        ASSERT_TRUE(gate_weights->ensureOnDevice(device));
        ASSERT_TRUE(output_indices_fp32->ensureOnDevice(device));
        ASSERT_TRUE(output_weights_fp32->ensureOnDevice(device));
        ASSERT_TRUE(output_indices_q8->ensureOnDevice(device));
        ASSERT_TRUE(output_weights_q8->ensureOnDevice(device));

        DeviceMoELayerRuntime host_runtime{};
        host_runtime.active_bank = 0;
        host_runtime.active_epoch = 1;
        host_runtime.expert_count = static_cast<uint32_t>(num_experts);
        host_runtime.top_k = static_cast<uint32_t>(top_k);
        host_runtime.banks[0].epoch = 1;
        host_runtime.banks[0].expert_count = static_cast<uint32_t>(num_experts);

        DeviceMoELayerRuntime *runtime_fp32 = nullptr;
        DeviceMoELayerRuntime *runtime_q8 = nullptr;
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_fp32), sizeof(DeviceMoELayerRuntime)), hipSuccess);
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&runtime_q8), sizeof(DeviceMoELayerRuntime)), hipSuccess);
        ASSERT_EQ(hipMemcpy(runtime_fp32, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(runtime_q8, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

        ROCmMoEKernel gpu_kernel(0);
        auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
        {
            ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
            ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
            ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "0");
            ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
            ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_fp32,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_fp32.get(), output_weights_fp32.get(),
                true, false));
        }
        {
            ScopedROCmEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
            ScopedROCmEnvOverride grouped_env("LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER", "0");
            ScopedROCmEnvOverride q8_env("LLAMINAR_ROCM_MOE_ROUTER_Q8", "1");
            ScopedROCmEnvOverride fp16_env("LLAMINAR_ROCM_MOE_ROUTER_FP16", "0");
            ScopedROCmEnvOverride kpart_env("LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "0");
            ASSERT_TRUE(gpu_kernel.decodeRouteSelect(
                runtime_q8,
                hidden.get(), gate_weights.get(),
                d_model, num_experts, top_k,
                true,
                output_indices_q8.get(), output_weights_q8.get(),
                true, false));
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        DeviceMoELayerRuntime after_fp32{};
        DeviceMoELayerRuntime after_q8{};
        ASSERT_EQ(hipMemcpy(&after_fp32, runtime_fp32, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(&after_q8, runtime_q8, sizeof(DeviceMoELayerRuntime), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipFree(runtime_fp32), hipSuccess);
        ASSERT_EQ(hipFree(runtime_q8), hipSuccess);

        output_indices_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_weights_fp32->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_indices_q8->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_weights_q8->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *legacy_indices_fp32 = output_indices_fp32->data();
        const float *legacy_weights_fp32 = output_weights_fp32->data();
        const float *legacy_indices_q8 = output_indices_q8->data();
        const float *legacy_weights_q8 = output_weights_q8->data();

        bool case_drifted = false;
        for (int k = 0; k < top_k; ++k)
        {
            EXPECT_FLOAT_EQ(legacy_indices_fp32[k], static_cast<float>(after_fp32.topk_expert_ids[k]));
            EXPECT_FLOAT_EQ(legacy_indices_q8[k], static_cast<float>(after_q8.topk_expert_ids[k]));
            EXPECT_NEAR(legacy_weights_fp32[k], after_fp32.topk_weights[k], 1.0e-6f);
            EXPECT_NEAR(legacy_weights_q8[k], after_q8.topk_weights[k], 1.0e-6f);

            if (after_q8.topk_expert_ids[k] != after_fp32.topk_expert_ids[k])
            {
                case_drifted = true;
                ++drift_slots;
            }
            EXPECT_EQ(after_q8.topk_expert_ids[k], after_fp32.topk_expert_ids[k])
                << "case=" << case_idx << " slot=" << k;

            const float diff = std::fabs(after_q8.topk_weights[k] - after_fp32.topk_weights[k]);
            max_weight_diff = std::max(max_weight_diff, diff);
            EXPECT_NEAR(after_q8.topk_weights[k], after_fp32.topk_weights[k], 2.0e-2f)
                << "case=" << case_idx << " slot=" << k;
        }
        if (case_drifted)
            ++drift_cases;
    }

    EXPECT_EQ(drift_cases, 0);
    EXPECT_EQ(drift_slots, 0);
    std::cout << "[DecodeRouteSelectQ8RouterMatchesFP32TopKAcrossSeeds] cases="
              << case_count << " drift_cases=" << drift_cases
              << " drift_slots=" << drift_slots << " max_weight_diff="
              << std::fixed << std::setprecision(8) << max_weight_diff << std::endl;
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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
    expectStrictVerifierSimilarity(
        "prefill router logits should match CPU routing row-by-row",
        gpu_result.router_logits.data(),
        cpu_result.router_logits.data(),
        gpu_result.router_logits.size(),
        static_cast<size_t>(num_experts));

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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
        if (gpu_batch[i] != cpu_batch[i])
            ++mismatches;
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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
    expectStrictVerifierSimilarity(
        "scatterAddWeighted should match CPU reference rows",
        gpu_output.data(),
        cpu_output.data(),
        gpu_output.size(),
        static_cast<size_t>(d_model));

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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
    expectStrictVerifierSimilarity(
        "sharedExpertGate decode should match CPU reference rows",
        gpu_shared_output_host.data(),
        cpu_shared_output.data(),
        cpu_shared_output.size(),
        static_cast<size_t>(d_model));

    std::cout << "[SharedExpertGate_Decode_Fused] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

TEST(Test__ROCmMoEKernel, SharedExpertGate_DecodeFusedFromTensorsMarksOutputDirty)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int seq_len = 1;
    const int d_model = 513;

    std::vector<float> input(seq_len * d_model);
    std::vector<float> gate_inp(d_model);
    std::vector<float> expected_shared_output(seq_len * d_model);
    fillRandom(input, -0.5f, 0.5f, 142);
    fillRandom(gate_inp, -0.25f, 0.25f, 223);
    fillRandom(expected_shared_output, -1.0f, 1.0f, 356);
    std::vector<float> initial_shared_output = expected_shared_output;

    CPUMoEKernel cpu_kernel;
    cpu_kernel.sharedExpertGate(input.data(), gate_inp.data(),
                                expected_shared_output.data(), seq_len, d_model);

    auto input_tensor = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto gate_tensor = TestTensorFactory::createFP32({static_cast<size_t>(d_model)});
    auto shared_output_tensor = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    std::copy(input.begin(), input.end(), input_tensor->mutable_data());
    std::copy(gate_inp.begin(), gate_inp.end(), gate_tensor->mutable_data());
    std::copy(initial_shared_output.begin(), initial_shared_output.end(), shared_output_tensor->mutable_data());

    ASSERT_TRUE(input_tensor->ensureOnDevice(device));
    ASSERT_TRUE(gate_tensor->ensureOnDevice(device));
    ASSERT_TRUE(shared_output_tensor->ensureOnDevice(device));

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    gpu_kernel.sharedExpertGateFromTensors(input_tensor.get(), gate_tensor.get(), shared_output_tensor.get(),
                                           seq_len, d_model);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(shared_output_tensor->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);

    const float *actual = shared_output_tensor->data();
    ASSERT_FALSE(hasNaNOrInf(actual, expected_shared_output.size()));
    double cosine = cosineSimilarity(actual, expected_shared_output.data(), expected_shared_output.size());
    double l2_err = relativeL2Error(actual, expected_shared_output.data(), expected_shared_output.size());
    expectStrictVerifierSimilarity(
        "sharedExpertGate tensor decode should match CPU reference rows",
        actual,
        expected_shared_output.data(),
        expected_shared_output.size(),
        static_cast<size_t>(d_model));

    std::cout << "[SharedExpertGate_DecodeFusedFromTensors] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

TEST(Test__ROCmMoEKernel, SharedExpertGateAddFromTensorsMatchesCPU)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int seq_len = 3;
    const int d_model = 513;

    std::vector<float> input(seq_len * d_model);
    std::vector<float> gate_inp(d_model);
    std::vector<float> shared_output(seq_len * d_model);
    std::vector<float> routed_residual(seq_len * d_model);
    fillRandom(input, -0.3f, 0.3f, 641);
    fillRandom(gate_inp, -0.2f, 0.2f, 642);
    fillRandom(shared_output, -1.0f, 1.0f, 643);
    fillRandom(routed_residual, -0.5f, 0.5f, 644);

    auto input_cpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto input_gpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto gate_cpu = TestTensorFactory::createFP32({static_cast<size_t>(d_model)});
    auto gate_gpu = TestTensorFactory::createFP32({static_cast<size_t>(d_model)});
    auto shared_cpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto shared_gpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto residual_cpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto residual_gpu = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto combined_tensor = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto expected_tensor = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    std::copy(input.begin(), input.end(), input_cpu->mutable_data());
    std::copy(input.begin(), input.end(), input_gpu->mutable_data());
    std::copy(gate_inp.begin(), gate_inp.end(), gate_cpu->mutable_data());
    std::copy(gate_inp.begin(), gate_inp.end(), gate_gpu->mutable_data());
    std::copy(shared_output.begin(), shared_output.end(), shared_cpu->mutable_data());
    std::copy(shared_output.begin(), shared_output.end(), shared_gpu->mutable_data());
    std::copy(routed_residual.begin(), routed_residual.end(), residual_cpu->mutable_data());
    std::copy(routed_residual.begin(), routed_residual.end(), residual_gpu->mutable_data());
    std::fill(combined_tensor->mutable_data(), combined_tensor->mutable_data() + combined_tensor->numel(), 0.0f);
    std::fill(expected_tensor->mutable_data(), expected_tensor->mutable_data() + expected_tensor->numel(), 0.0f);

    CPUMoEKernel cpu_kernel;
    cpu_kernel.sharedExpertGateAddFromTensors(
        input_cpu.get(), gate_cpu.get(), shared_cpu.get(),
        residual_cpu.get(), expected_tensor.get(), seq_len, d_model);

    ASSERT_TRUE(input_gpu->ensureOnDevice(device));
    ASSERT_TRUE(gate_gpu->ensureOnDevice(device));
    ASSERT_TRUE(shared_gpu->ensureOnDevice(device));
    ASSERT_TRUE(residual_gpu->ensureOnDevice(device));
    ASSERT_TRUE(combined_tensor->ensureOnDevice(device));

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    gpu_kernel.sharedExpertGateAddFromTensors(
        input_gpu.get(), gate_gpu.get(), shared_gpu.get(),
        residual_gpu.get(), combined_tensor.get(), seq_len, d_model);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(combined_tensor->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);

    const float *actual = combined_tensor->data();
    ASSERT_FALSE(hasNaNOrInf(actual, expected_tensor->numel()));
    double cosine = cosineSimilarity(actual, expected_tensor->data(), expected_tensor->numel());
    double l2_err = relativeL2Error(actual, expected_tensor->data(), expected_tensor->numel());
    expectStrictVerifierSimilarity(
        "sharedExpertGateAdd tensor path should match CPU reference rows",
        actual,
        expected_tensor->data(),
        expected_tensor->numel(),
        static_cast<size_t>(d_model));

    std::cout << "[SharedExpertGateAddFromTensors] cosine=" << std::fixed << std::setprecision(6)
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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
    expectStrictVerifierSimilarity(
        "sharedExpertGate prefill should match CPU reference rows",
        gpu_shared_output_host.data(),
        cpu_shared_output.data(),
        cpu_shared_output.size(),
        static_cast<size_t>(d_model));

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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
    expectStrictVerifierSimilarity(
        "swiGLU should match CPU reference rows",
        gpu_gate_host.data(),
        cpu_gate.data(),
        count,
        count);

    std::cout << "[SwiGLU] cosine=" << std::fixed << std::setprecision(6)
              << cosine << " l2_err=" << l2_err << std::endl;
}

// ============================================================================
// Test: grouped decode expert down path matches existing sequential ROCm path
// ============================================================================

TEST(Test__ROCmMoEKernel, GroupedExpertDownDecode_Q4_0MatchesSequential)
{
    SKIP_IF_NO_ROCM();

    const int d_model = 128;
    const int intermediate = 128;
    const int num_active = 4;
    const int expert_ids[num_active] = {2, 7, 3, 5};
    const float route_weights[num_active] = {0.39f, 0.27f, 0.21f, 0.13f};
    const DeviceId device = DeviceId::rocm(0);

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<std::unique_ptr<TensorBase>> weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;
    weight_tensors.reserve(num_active);
    packed_weights.reserve(num_active);
    down_kernels.reserve(num_active);

    for (int i = 0; i < num_active; ++i)
    {
        auto weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)});
        auto packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(weights.get(), *packed))
            << "packWeightsToROCm failed for expert " << expert_ids[i];
        ASSERT_FALSE(packed->native_vnni_payload.empty());

        auto kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(packed.get(), 0);
        weight_tensors.push_back(std::move(weights));
        packed_weights.push_back(std::move(packed));
        down_kernels.push_back(std::move(kernel));
    }

    auto reqs = down_kernels[0]->getWorkspaceRequirements(1, d_model, intermediate);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(reqs));
    for (auto &kernel : down_kernels)
    {
        kernel->bindWorkspace(workspace.get());
        kernel->prepareWeights();
    }

    std::vector<std::shared_ptr<FP32Tensor>> gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> up_tensors;
    gate_tensors.reserve(num_active);
    up_tensors.reserve(num_active);
    for (int i = 0; i < num_active; ++i)
    {
        auto gate = TestTensorFactory::createFP32Random(
            {1, static_cast<size_t>(intermediate)}, -2.0f, 2.0f, 100 + i);
        auto up = TestTensorFactory::createFP32Random(
            {1, static_cast<size_t>(intermediate)}, -2.0f, 2.0f, 200 + i);
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        gate_tensors.push_back(std::move(gate));
        up_tensors.push_back(std::move(up));
    }

    auto sequential_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto grouped_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto device_routed_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    ASSERT_TRUE(sequential_output->ensureOnDevice(device));
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    ASSERT_TRUE(device_routed_output->ensureOnDevice(device));
    moe_kernel.zeroBuffer(sequential_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(device_routed_output.get(), static_cast<size_t>(d_model) * sizeof(float));

    for (int i = 0; i < num_active; ++i)
    {
        auto expert_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
        ASSERT_TRUE(expert_output->ensureOnDevice(device));
        ASSERT_TRUE(down_kernels[i]->multiply_tensor_with_fused_swiglu(
            gate_tensors[i].get(), up_tensors[i].get(), expert_output.get(),
            1, d_model, intermediate));
        expert_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        moe_kernel.weightedAddFromTensors(
            sequential_output.get(), expert_output.get(), route_weights[i], d_model);
    }
    sequential_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    ITensor *gate_ptrs[num_active] = {};
    ITensor *up_ptrs[num_active] = {};
    DeviceNativeVNNIMatrixDesc descs[num_active] = {};
    for (int i = 0; i < num_active; ++i)
    {
        gate_ptrs[i] = gate_tensors[i].get();
        up_ptrs[i] = up_tensors[i].get();
        ASSERT_TRUE(down_kernels[i]->exportNativeVNNIMatrixDesc(descs[i]));
        ASSERT_EQ(descs[i].n, d_model);
        ASSERT_EQ(descs[i].k, intermediate);
    }

    ASSERT_TRUE(moe_kernel.groupedExpertDownDecode(
        gate_ptrs, up_ptrs, expert_ids, route_weights, descs,
        num_active, grouped_output.get(), d_model, intermediate));
    hipDeviceSynchronize();

    sequential_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    const float *seq = sequential_output->data();
    const float *grouped = grouped_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, d_model));
    expectStrictVerifierSimilarity(
        "ROCm grouped expert down decode Q4_0 must match sequential decode",
        grouped,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    const double cosine = cosineSimilarity(grouped, seq, d_model);
    const double l2_err = relativeL2Error(grouped, seq, d_model);

    std::cout << "[GroupedExpertDownDecode_Q4_0MatchesSequential] cosine="
              << std::fixed << std::setprecision(6) << cosine
              << " l2_err=" << l2_err << std::endl;
}

template <typename WeightFactory>
void runGroupedExpertDownDecodeFormatMatch(const char *label, WeightFactory create_weights)
{
    const int d_model = 128;
    const int intermediate = 256;
    const int num_experts = 8;
    const int num_active = 4;
    const int expert_ids[num_active] = {2, 7, 3, 5};
    const float route_weights[num_active] = {0.39f, 0.27f, 0.21f, 0.13f};
    const DeviceId device = DeviceId::rocm(0);

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<std::unique_ptr<TensorBase>> weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;
    weight_tensors.reserve(num_experts);
    packed_weights.reserve(num_experts);
    down_kernels.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto weights = create_weights(static_cast<size_t>(d_model), static_cast<size_t>(intermediate));
        auto packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(weights.get(), *packed))
            << "packWeightsToROCm failed for " << label << " expert " << expert_id;
        ASSERT_FALSE(packed->native_vnni_payload.empty());

        auto kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(packed.get(), 0);
        weight_tensors.push_back(std::move(weights));
        packed_weights.push_back(std::move(packed));
        down_kernels.push_back(std::move(kernel));
    }

    auto reqs = down_kernels[0]->getWorkspaceRequirements(1, d_model, intermediate);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(reqs));
    for (auto &kernel : down_kernels)
    {
        kernel->bindWorkspace(workspace.get());
        kernel->prepareWeights();
    }

    std::vector<std::shared_ptr<FP32Tensor>> gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> up_tensors;
    gate_tensors.reserve(num_active);
    up_tensors.reserve(num_active);
    for (int i = 0; i < num_active; ++i)
    {
        auto gate = TestTensorFactory::createFP32Random(
            {1, static_cast<size_t>(intermediate)}, -2.0f, 2.0f, 300 + i);
        auto up = TestTensorFactory::createFP32Random(
            {1, static_cast<size_t>(intermediate)}, -2.0f, 2.0f, 400 + i);
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        gate_tensors.push_back(std::move(gate));
        up_tensors.push_back(std::move(up));
    }

    auto sequential_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto grouped_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto device_routed_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto runtime_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto parallel_runtime_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    ASSERT_TRUE(sequential_output->ensureOnDevice(device));
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    ASSERT_TRUE(device_routed_output->ensureOnDevice(device));
    ASSERT_TRUE(runtime_output->ensureOnDevice(device));
    ASSERT_TRUE(parallel_runtime_output->ensureOnDevice(device));
    moe_kernel.zeroBuffer(sequential_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(device_routed_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(runtime_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(parallel_runtime_output.get(), static_cast<size_t>(d_model) * sizeof(float));

    for (int i = 0; i < num_active; ++i)
    {
        auto expert_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
        ASSERT_TRUE(expert_output->ensureOnDevice(device));
        ASSERT_TRUE(down_kernels[expert_ids[i]]->multiply_tensor_with_fused_swiglu(
            gate_tensors[i].get(), up_tensors[i].get(), expert_output.get(),
            1, d_model, intermediate));
        expert_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        moe_kernel.weightedAddFromTensors(
            sequential_output.get(), expert_output.get(), route_weights[i], d_model);
    }
    sequential_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    ITensor *gate_ptrs[num_active] = {};
    ITensor *up_ptrs[num_active] = {};
    std::vector<DeviceNativeVNNIMatrixDesc> descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(down_kernels[expert_id]->exportNativeVNNIMatrixDesc(descs[expert_id]));
        ASSERT_EQ(descs[expert_id].n, d_model);
        ASSERT_EQ(descs[expert_id].k, intermediate);
    }
    for (int i = 0; i < num_active; ++i)
    {
        gate_ptrs[i] = gate_tensors[i].get();
        up_ptrs[i] = up_tensors[i].get();
    }

    const int table_id = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(table_id, 0);

    ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
        gate_ptrs, up_ptrs, expert_ids, route_weights, table_id,
        num_active, grouped_output.get(), d_model, intermediate));

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(num_active)});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(num_active)});
    for (int i = 0; i < num_active; ++i)
    {
        routing_indices->mutable_data()[i] = static_cast<float>(expert_ids[i]);
        routing_weights->mutable_data()[i] = route_weights[i];
    }
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRouting(
        gate_ptrs, up_ptrs, routing_indices.get(), routing_weights.get(), table_id,
        num_active, device_routed_output.get(), d_model, intermediate));

    DeviceMoELayerRuntime host_runtime{};
    host_runtime.expert_count = static_cast<uint32_t>(num_experts);
    host_runtime.top_k = static_cast<uint32_t>(num_active);
    for (int i = 0; i < num_active; ++i)
    {
        host_runtime.topk_expert_ids[i] = expert_ids[i];
        host_runtime.topk_weights[i] = route_weights[i];
    }
    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(device_runtime, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    {
        ScopedROCmEnvOverride disable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "0");
        ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRuntime(
            gate_ptrs, up_ptrs, device_runtime, table_id,
            num_active, runtime_output.get(), d_model, intermediate));
    }

    {
        ScopedROCmEnvOverride enable_parallel_down("LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "1");
        ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRuntime(
            gate_ptrs, up_ptrs, device_runtime, table_id,
            num_active, parallel_runtime_output.get(), d_model, intermediate));
    }
    hipDeviceSynchronize();
    ASSERT_EQ(hipFree(device_runtime), hipSuccess);

    sequential_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    device_routed_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    runtime_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    parallel_runtime_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    const float *seq = sequential_output->data();
    const float *grouped = grouped_output->data();
    const float *device_routed = device_routed_output->data();
    const float *runtime = runtime_output->data();
    const float *parallel_runtime = parallel_runtime_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, d_model));
    ASSERT_FALSE(hasNaNOrInf(device_routed, d_model));
    ASSERT_FALSE(hasNaNOrInf(runtime, d_model));
    ASSERT_FALSE(hasNaNOrInf(parallel_runtime, d_model));
    const double cosine = cosineSimilarity(grouped, seq, d_model);
    const double l2_err = relativeL2Error(grouped, seq, d_model);
    const double device_cosine = cosineSimilarity(device_routed, seq, d_model);
    const double device_l2_err = relativeL2Error(device_routed, seq, d_model);
    const double runtime_cosine = cosineSimilarity(runtime, seq, d_model);
    const double runtime_l2_err = relativeL2Error(runtime, seq, d_model);
    const double parallel_runtime_cosine = cosineSimilarity(parallel_runtime, seq, d_model);
    const double parallel_runtime_l2_err = relativeL2Error(parallel_runtime, seq, d_model);
    const double parallel_vs_runtime_cosine = cosineSimilarity(parallel_runtime, runtime, d_model);
    const double parallel_vs_runtime_l2_err = relativeL2Error(parallel_runtime, runtime, d_model);
    const double parallel_vs_runtime_max_abs = maxAbsDiff(parallel_runtime, runtime, d_model);

    expectStrictVerifierSimilarity(
        (std::string(label) + " grouped expert down decode must match sequential").c_str(),
        grouped,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    expectStrictVerifierSimilarity(
        (std::string(label) + " device-routed grouped expert down decode must match sequential").c_str(),
        device_routed,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    expectStrictVerifierSimilarity(
        (std::string(label) + " runtime grouped expert down decode must match sequential").c_str(),
        runtime,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    expectStrictVerifierSimilarity(
        (std::string(label) + " parallel runtime grouped expert down decode must match sequential").c_str(),
        parallel_runtime,
        seq,
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));
    EXPECT_GE(parallel_vs_runtime_cosine, 0.99999)
        << label << " parallel runtime should closely match serial runtime, cosine=" << parallel_vs_runtime_cosine;
    EXPECT_LE(parallel_vs_runtime_l2_err, 0.002)
        << label << " parallel runtime should closely match serial runtime, relative L2=" << parallel_vs_runtime_l2_err;
    EXPECT_LE(parallel_vs_runtime_max_abs, 0.05)
        << label << " parallel runtime max abs diff vs serial runtime too high: " << parallel_vs_runtime_max_abs;

    std::cout << "[GroupedExpertDownDecode_" << label << "MatchesSequential] cosine="
              << std::fixed << std::setprecision(6) << cosine
              << " l2_err=" << l2_err
              << " device_cosine=" << device_cosine
              << " device_l2_err=" << device_l2_err
              << " runtime_cosine=" << runtime_cosine
              << " runtime_l2_err=" << runtime_l2_err
              << " parallel_runtime_cosine=" << parallel_runtime_cosine
              << " parallel_runtime_l2_err=" << parallel_runtime_l2_err
              << " parallel_vs_runtime_l2_err=" << parallel_vs_runtime_l2_err
              << " parallel_vs_runtime_max_abs=" << parallel_vs_runtime_max_abs << std::endl;
}

TEST(Test__ROCmMoEKernel, GroupedExpertDownDecode_Q5_KMatchesSequential)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertDownDecodeFormatMatch("Q5_K", [](size_t rows, size_t cols)
                                          {
                                              auto tensor = TestTensorFactory::createQ5_KRandom({rows, cols});
                                              injectNonZeroKQuantMins(tensor.get());
                                              return tensor;
                                          });
}

TEST(Test__ROCmMoEKernel, GroupedExpertDownDecode_Q4_KMatchesSequential)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertDownDecodeFormatMatch("Q4_K", [](size_t rows, size_t cols)
                                          {
                                              auto tensor = TestTensorFactory::createQ4_KRandom({rows, cols});
                                              injectNonZeroKQuantMins(tensor.get());
                                              return tensor;
                                          });
}

TEST(Test__ROCmMoEKernel, GroupedExpertDownDecode_Q6_KMatchesSequential)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertDownDecodeFormatMatch("Q6_K", [](size_t rows, size_t cols)
                                          { return TestTensorFactory::createQ6_KRandom({rows, cols}); });
}

template <typename WeightFactory>
void runGroupedExpertGateUpDecodeFormatMatch(const char *label, WeightFactory create_weights)
{
    const int d_model = 256;
    const int intermediate = 128;
    const int num_experts = 8;
    const int num_active = 4;
    const int expert_ids[num_active] = {2, 7, 3, 5};
    const DeviceId device = DeviceId::rocm(0);

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> gate_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> up_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> gate_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> up_kernels;
    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    gate_packed_weights.reserve(num_experts);
    up_packed_weights.reserve(num_experts);
    gate_kernels.reserve(num_experts);
    up_kernels.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = create_weights(static_cast<size_t>(intermediate), static_cast<size_t>(d_model));
        auto up_weights = create_weights(static_cast<size_t>(intermediate), static_cast<size_t>(d_model));
        auto gate_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto up_packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(gate_weights.get(), *gate_packed))
            << "packWeightsToROCm failed for " << label << " gate expert " << expert_id;
        ASSERT_TRUE(rocm::packWeightsToROCm(up_weights.get(), *up_packed))
            << "packWeightsToROCm failed for " << label << " up expert " << expert_id;
        ASSERT_FALSE(gate_packed->native_vnni_payload.empty());
        ASSERT_FALSE(up_packed->native_vnni_payload.empty());

        auto gate_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(gate_packed.get(), 0);
        auto up_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(up_packed.get(), 0);
        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        gate_packed_weights.push_back(std::move(gate_packed));
        up_packed_weights.push_back(std::move(up_packed));
        gate_kernels.push_back(std::move(gate_kernel));
        up_kernels.push_back(std::move(up_kernel));
    }

    auto reqs = gate_kernels[0]->getWorkspaceRequirements(1, intermediate, d_model);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(reqs));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_kernels[expert_id]->bindWorkspace(workspace.get());
        up_kernels[expert_id]->bindWorkspace(workspace.get());
        gate_kernels[expert_id]->prepareWeights();
        up_kernels[expert_id]->prepareWeights();
    }

    auto input = TestTensorFactory::createFP32Random(
        {1, static_cast<size_t>(d_model)}, -2.0f, 2.0f, 900);
    ASSERT_TRUE(input->ensureOnDevice(device));

    std::vector<std::shared_ptr<FP32Tensor>> seq_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> seq_up_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> grouped_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> grouped_up_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> device_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> device_up_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> runtime_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> runtime_up_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> kpart_gate_tensors;
    std::vector<std::shared_ptr<FP32Tensor>> kpart_up_tensors;
    seq_gate_tensors.reserve(num_active);
    seq_up_tensors.reserve(num_active);
    grouped_gate_tensors.reserve(num_active);
    grouped_up_tensors.reserve(num_active);
    device_gate_tensors.reserve(num_active);
    device_up_tensors.reserve(num_active);
    runtime_gate_tensors.reserve(num_active);
    runtime_up_tensors.reserve(num_active);
    kpart_gate_tensors.reserve(num_active);
    kpart_up_tensors.reserve(num_active);

    std::vector<ITensorGemm::TensorProjectionDesc> projections;
    projections.reserve(num_active * 2);
    for (int i = 0; i < num_active; ++i)
    {
        auto seq_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto seq_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto grouped_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto grouped_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto device_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto device_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto runtime_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto runtime_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto kpart_gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        auto kpart_up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        ASSERT_TRUE(seq_gate->ensureOnDevice(device));
        ASSERT_TRUE(seq_up->ensureOnDevice(device));
        ASSERT_TRUE(grouped_gate->ensureOnDevice(device));
        ASSERT_TRUE(grouped_up->ensureOnDevice(device));
        ASSERT_TRUE(device_gate->ensureOnDevice(device));
        ASSERT_TRUE(device_up->ensureOnDevice(device));
        ASSERT_TRUE(runtime_gate->ensureOnDevice(device));
        ASSERT_TRUE(runtime_up->ensureOnDevice(device));
        ASSERT_TRUE(kpart_gate->ensureOnDevice(device));
        ASSERT_TRUE(kpart_up->ensureOnDevice(device));

        const int expert_id = expert_ids[i];
        projections.emplace_back(gate_kernels[expert_id].get(), seq_gate.get(), intermediate, nullptr, "gate");
        projections.emplace_back(up_kernels[expert_id].get(), seq_up.get(), intermediate, nullptr, "up");

        seq_gate_tensors.push_back(std::move(seq_gate));
        seq_up_tensors.push_back(std::move(seq_up));
        grouped_gate_tensors.push_back(std::move(grouped_gate));
        grouped_up_tensors.push_back(std::move(grouped_up));
        device_gate_tensors.push_back(std::move(device_gate));
        device_up_tensors.push_back(std::move(device_up));
        runtime_gate_tensors.push_back(std::move(runtime_gate));
        runtime_up_tensors.push_back(std::move(runtime_up));
        kpart_gate_tensors.push_back(std::move(kpart_gate));
        kpart_up_tensors.push_back(std::move(kpart_up));
    }

    ASSERT_TRUE(gate_kernels[expert_ids[0]]->multiply_fused_tensor(
        input.get(), projections, 1, d_model));
    hipDeviceSynchronize();

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(gate_kernels[expert_id]->exportNativeVNNIMatrixDesc(gate_descs[expert_id]));
        ASSERT_TRUE(up_kernels[expert_id]->exportNativeVNNIMatrixDesc(up_descs[expert_id]));
        ASSERT_EQ(gate_descs[expert_id].n, intermediate);
        ASSERT_EQ(gate_descs[expert_id].k, d_model);
        ASSERT_EQ(up_descs[expert_id].n, intermediate);
        ASSERT_EQ(up_descs[expert_id].k, d_model);
    }

    ITensor *grouped_gate_ptrs[num_active] = {};
    ITensor *grouped_up_ptrs[num_active] = {};
    ITensor *device_gate_ptrs[num_active] = {};
    ITensor *device_up_ptrs[num_active] = {};
    ITensor *runtime_gate_ptrs[num_active] = {};
    ITensor *runtime_up_ptrs[num_active] = {};
    ITensor *kpart_gate_ptrs[num_active] = {};
    ITensor *kpart_up_ptrs[num_active] = {};
    for (int i = 0; i < num_active; ++i)
    {
        grouped_gate_ptrs[i] = grouped_gate_tensors[i].get();
        grouped_up_ptrs[i] = grouped_up_tensors[i].get();
        device_gate_ptrs[i] = device_gate_tensors[i].get();
        device_up_ptrs[i] = device_up_tensors[i].get();
        runtime_gate_ptrs[i] = runtime_gate_tensors[i].get();
        runtime_up_ptrs[i] = runtime_up_tensors[i].get();
        kpart_gate_ptrs[i] = kpart_gate_tensors[i].get();
        kpart_up_ptrs[i] = kpart_up_tensors[i].get();
    }

    const int table_id = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(table_id, 0);

    ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
        input.get(), expert_ids, table_id, num_active,
        grouped_gate_ptrs, grouped_up_ptrs, d_model, intermediate));

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(num_active)});
    for (int i = 0; i < num_active; ++i)
        routing_indices->mutable_data()[i] = static_cast<float>(expert_ids[i]);
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));

    ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRouting(
        input.get(), routing_indices.get(), table_id, num_active,
        device_gate_ptrs, device_up_ptrs, d_model, intermediate));

    DeviceMoELayerRuntime host_runtime{};
    host_runtime.expert_count = static_cast<uint32_t>(num_experts);
    host_runtime.top_k = static_cast<uint32_t>(num_active);
    for (int i = 0; i < num_active; ++i)
        host_runtime.topk_expert_ids[i] = expert_ids[i];
    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpy(device_runtime, &host_runtime, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice), hipSuccess);

    {
        ScopedROCmEnvOverride disable_kpart("LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE", "0");
        ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRuntime(
            device_runtime, input.get(), table_id, num_active,
            runtime_gate_ptrs, runtime_up_ptrs, d_model, intermediate));
    }

    constexpr const char *kKPartCounts[] = {"2", "4", "8"};
    for (const char *kparts : kKPartCounts)
    {
        ScopedROCmEnvOverride deterministic_off("LLAMINAR_DETERMINISTIC", "0");
        ScopedROCmEnvOverride enable_kpart("LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE", "1");
        ScopedROCmEnvOverride set_kparts("LLAMINAR_ROCM_MOE_GATEUP_KPARTS", kparts);
        ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRuntime(
            device_runtime, input.get(), table_id, num_active,
            kpart_gate_ptrs, kpart_up_ptrs, d_model, intermediate))
            << label << " K-partition runtime gate/up failed for kparts=" << kparts;
        hipDeviceSynchronize();

        for (int i = 0; i < num_active; ++i)
        {
            runtime_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            runtime_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            kpart_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            kpart_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

            const float *runtime_gate = runtime_gate_tensors[i]->data();
            const float *runtime_up = runtime_up_tensors[i]->data();
            const float *kpart_gate = kpart_gate_tensors[i]->data();
            const float *kpart_up = kpart_up_tensors[i]->data();

            const double kpart_gate_cosine = cosineSimilarity(kpart_gate, runtime_gate, intermediate);
            const double kpart_up_cosine = cosineSimilarity(kpart_up, runtime_up, intermediate);
            const double kpart_gate_l2 = relativeL2Error(kpart_gate, runtime_gate, intermediate);
            const double kpart_up_l2 = relativeL2Error(kpart_up, runtime_up, intermediate);
            const double kpart_gate_max_abs = maxAbsDiff(kpart_gate, runtime_gate, intermediate);
            const double kpart_up_max_abs = maxAbsDiff(kpart_up, runtime_up, intermediate);

            EXPECT_GE(kpart_gate_cosine, 0.99999)
                << label << " kpart gate cosine mismatch for active slot " << i << " kparts=" << kparts;
            EXPECT_GE(kpart_up_cosine, 0.99999)
                << label << " kpart up cosine mismatch for active slot " << i << " kparts=" << kparts;
            EXPECT_LE(kpart_gate_l2, 0.0015)
                << label << " kpart gate relative L2 too high for active slot " << i << " kparts=" << kparts;
            EXPECT_LE(kpart_up_l2, 0.0015)
                << label << " kpart up relative L2 too high for active slot " << i << " kparts=" << kparts;
            EXPECT_LE(kpart_gate_max_abs, 0.03)
                << label << " kpart gate max abs diff too high for active slot " << i << " kparts=" << kparts;
            EXPECT_LE(kpart_up_max_abs, 0.03)
                << label << " kpart up max abs diff too high for active slot " << i << " kparts=" << kparts;
        }
    }
    hipDeviceSynchronize();
    ASSERT_EQ(hipFree(device_runtime), hipSuccess);

    for (int i = 0; i < num_active; ++i)
    {
        seq_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        seq_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        grouped_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        grouped_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        device_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        device_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        runtime_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        runtime_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        kpart_gate_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        kpart_up_tensors[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        const float *seq_gate = seq_gate_tensors[i]->data();
        const float *seq_up = seq_up_tensors[i]->data();
        const float *grouped_gate = grouped_gate_tensors[i]->data();
        const float *grouped_up = grouped_up_tensors[i]->data();
        const float *device_gate = device_gate_tensors[i]->data();
        const float *device_up = device_up_tensors[i]->data();
        const float *runtime_gate = runtime_gate_tensors[i]->data();
        const float *runtime_up = runtime_up_tensors[i]->data();
        const float *kpart_gate = kpart_gate_tensors[i]->data();
        const float *kpart_up = kpart_up_tensors[i]->data();

        ASSERT_FALSE(hasNaNOrInf(grouped_gate, intermediate));
        ASSERT_FALSE(hasNaNOrInf(grouped_up, intermediate));
        ASSERT_FALSE(hasNaNOrInf(device_gate, intermediate));
        ASSERT_FALSE(hasNaNOrInf(device_up, intermediate));
        ASSERT_FALSE(hasNaNOrInf(runtime_gate, intermediate));
        ASSERT_FALSE(hasNaNOrInf(runtime_up, intermediate));
        ASSERT_FALSE(hasNaNOrInf(kpart_gate, intermediate));
        ASSERT_FALSE(hasNaNOrInf(kpart_up, intermediate));

        const double gate_cosine = cosineSimilarity(grouped_gate, seq_gate, intermediate);
        const double up_cosine = cosineSimilarity(grouped_up, seq_up, intermediate);
        const double gate_l2 = relativeL2Error(grouped_gate, seq_gate, intermediate);
        const double up_l2 = relativeL2Error(grouped_up, seq_up, intermediate);
        const double device_gate_cosine = cosineSimilarity(device_gate, seq_gate, intermediate);
        const double device_up_cosine = cosineSimilarity(device_up, seq_up, intermediate);
        const double device_gate_l2 = relativeL2Error(device_gate, seq_gate, intermediate);
        const double device_up_l2 = relativeL2Error(device_up, seq_up, intermediate);
        const double runtime_gate_cosine = cosineSimilarity(runtime_gate, seq_gate, intermediate);
        const double runtime_up_cosine = cosineSimilarity(runtime_up, seq_up, intermediate);
        const double runtime_gate_l2 = relativeL2Error(runtime_gate, seq_gate, intermediate);
        const double runtime_up_l2 = relativeL2Error(runtime_up, seq_up, intermediate);
        const double kpart_gate_cosine = cosineSimilarity(kpart_gate, runtime_gate, intermediate);
        const double kpart_up_cosine = cosineSimilarity(kpart_up, runtime_up, intermediate);
        const double kpart_gate_l2 = relativeL2Error(kpart_gate, runtime_gate, intermediate);
        const double kpart_up_l2 = relativeL2Error(kpart_up, runtime_up, intermediate);

        expectStrictVerifierSimilarity(
            (std::string(label) + " grouped gate output must match sequential").c_str(),
            grouped_gate,
            seq_gate,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " grouped up output must match sequential").c_str(),
            grouped_up,
            seq_up,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " device-routed grouped gate output must match sequential").c_str(),
            device_gate,
            seq_gate,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " device-routed grouped up output must match sequential").c_str(),
            device_up,
            seq_up,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " runtime grouped gate output must match sequential").c_str(),
            runtime_gate,
            seq_gate,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        expectStrictVerifierSimilarity(
            (std::string(label) + " runtime grouped up output must match sequential").c_str(),
            runtime_up,
            seq_up,
            static_cast<size_t>(intermediate),
            static_cast<size_t>(intermediate));
        EXPECT_GE(kpart_gate_cosine, 0.99999)
            << label << " kpart grouped gate cosine mismatch vs runtime for active slot " << i;
        EXPECT_GE(kpart_up_cosine, 0.99999)
            << label << " kpart grouped up cosine mismatch vs runtime for active slot " << i;
        EXPECT_LE(kpart_gate_l2, 0.0015)
            << label << " kpart grouped gate relative L2 too high vs runtime for active slot " << i;
        EXPECT_LE(kpart_up_l2, 0.0015)
            << label << " kpart grouped up relative L2 too high vs runtime for active slot " << i;

        std::cout << "[GroupedExpertGateUpDecode_" << label
                  << "] slot=" << i
                  << " gate_cosine=" << std::fixed << std::setprecision(6) << gate_cosine
                  << " up_cosine=" << up_cosine
                  << " gate_l2=" << gate_l2
                  << " up_l2=" << up_l2
                  << " device_gate_cosine=" << device_gate_cosine
                  << " device_up_cosine=" << device_up_cosine
                  << " device_gate_l2=" << device_gate_l2
                  << " device_up_l2=" << device_up_l2
                  << " runtime_gate_cosine=" << runtime_gate_cosine
                  << " runtime_up_cosine=" << runtime_up_cosine
                  << " runtime_gate_l2=" << runtime_gate_l2
                  << " runtime_up_l2=" << runtime_up_l2
                  << " kpart_gate_cosine=" << kpart_gate_cosine
                  << " kpart_up_cosine=" << kpart_up_cosine
                  << " kpart_gate_l2=" << kpart_gate_l2
                  << " kpart_up_l2=" << kpart_up_l2 << std::endl;
    }
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q4_0KPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q4_0", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ4_0Random({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_IQ4_NLKPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("IQ4_NL", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createIQ4_NLRandom({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q4_1KPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q4_1", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ4_1Random({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q5_0KPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q5_0", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ5_0Random({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q5_1KPartMatchesRuntime)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q5_1", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ5_1Random({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q5_KMatchesFusedProjection)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q5_K", [](size_t rows, size_t cols)
                                            {
                                                auto tensor = TestTensorFactory::createQ5_KRandom({rows, cols});
                                                injectNonZeroKQuantMins(tensor.get());
                                                return tensor;
                                            });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q4_KMatchesFusedProjection)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q4_K", [](size_t rows, size_t cols)
                                            {
                                                auto tensor = TestTensorFactory::createQ4_KRandom({rows, cols});
                                                injectNonZeroKQuantMins(tensor.get());
                                                return tensor;
                                            });
}

TEST(Test__ROCmMoEKernel, GroupedExpertGateUpDecode_Q6_KMatchesFusedProjection)
{
    SKIP_IF_NO_ROCM();
    runGroupedExpertGateUpDecodeFormatMatch("Q6_K", [](size_t rows, size_t cols)
                                            { return TestTensorFactory::createQ6_KRandom({rows, cols}); });
}

TEST(Test__ROCmMoEKernel, GroupedSharedExpertDecodeCapturesWithStablePointerCache)
{
    SKIP_IF_NO_ROCM();

    constexpr int d_model = 128;
    constexpr int intermediate = 128;
    constexpr int num_experts = 1;
    constexpr int num_active = 1;
    constexpr int expert_ids[num_active] = {0};
    constexpr float expert_weights[num_active] = {1.0f};
    const DeviceId device = DeviceId::rocm(0);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/1,
        d_model,
        intermediate,
        num_experts,
        /*top_k=*/1);

    auto make_kernel = [&](auto weights)
    {
        auto packed = std::make_unique<rocm::ROCmPackedWeights>();
        EXPECT_TRUE(rocm::packWeightsToROCm(weights.get(), *packed));
        auto kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(packed.get(), 0);
        static_cast<ITensorKernel *>(kernel.get())->setGPUStream(stream);
        return std::tuple{
            std::move(weights),
            std::move(packed),
            std::move(kernel)};
    };

    auto [gate_weights, gate_packed, gate_kernel] = make_kernel(
        TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}));
    auto [up_weights, up_packed, up_kernel] = make_kernel(
        TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}));
    auto [down_weights, down_packed, down_kernel] = make_kernel(
        TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}));

    WorkspaceRequirements gemm_reqs;
    gemm_reqs.merge(gate_kernel->getWorkspaceRequirements(1, intermediate, d_model));
    gemm_reqs.merge(up_kernel->getWorkspaceRequirements(1, intermediate, d_model));
    gemm_reqs.merge(down_kernel->getWorkspaceRequirements(1, d_model, intermediate));
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));
    for (auto *kernel : {gate_kernel.get(), up_kernel.get(), down_kernel.get()})
    {
        kernel->bindWorkspace(gemm_workspace.get());
        kernel->prepareWeights();
    }

    DeviceNativeVNNIMatrixDesc gate_desc{};
    DeviceNativeVNNIMatrixDesc up_desc{};
    DeviceNativeVNNIMatrixDesc down_desc{};
    ASSERT_TRUE(gate_kernel->exportNativeVNNIMatrixDesc(gate_desc));
    ASSERT_TRUE(up_kernel->exportNativeVNNIMatrixDesc(up_desc));
    ASSERT_TRUE(down_kernel->exportNativeVNNIMatrixDesc(down_desc));

    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        &gate_desc, &up_desc, num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        &down_desc, num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto hidden = TestTensorFactory::createFP32Random(
        {1, static_cast<size_t>(d_model)}, -0.5f, 0.5f, 5151);
    auto gate = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
    auto up = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
    auto output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
    ASSERT_TRUE(gate->ensureOnDevice(device, stream));
    ASSERT_TRUE(up->ensureOnDevice(device, stream));
    ASSERT_TRUE(output->ensureOnDevice(device, stream));

    std::array<ITensor *, num_active> gate_outputs = {gate.get()};
    std::array<ITensor *, num_active> up_outputs = {up.get()};

    ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids, gateup_table, num_active,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
        down_table, num_active, output.get(), d_model, intermediate));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
    const bool captured_gateup = moe_kernel.groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids, gateup_table, num_active,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate);
    const bool captured_down = moe_kernel.groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
        down_table, num_active, output.get(), d_model, intermediate);
    const hipError_t capture_status = hipStreamEndCapture(stream, &graph);
    EXPECT_TRUE(captured_gateup);
    EXPECT_TRUE(captured_down);
    ASSERT_EQ(capture_status, hipSuccess) << hipGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_FALSE(hasNaNOrInf(output->data(), static_cast<size_t>(d_model)));

    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, RuntimeGroupedDecodeFusedPathMatchesTwoStepAndCaptures)
{
    SKIP_IF_NO_ROCM();

    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");

    constexpr int d_model = 128;
    constexpr int intermediate = 128;
    constexpr int num_experts = 4;
    constexpr int top_k = 4;
    const DeviceId device = DeviceId::rocm(0);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ASSERT_TRUE(llaminar2::rocm::ensureIQGridTablesInitialized(0));

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/1,
        d_model,
        intermediate,
        num_experts,
        top_k);

    auto make_kernel = [&](auto weights)
    {
        auto packed = std::make_unique<rocm::ROCmPackedWeights>();
        EXPECT_TRUE(rocm::packWeightsToROCm(weights.get(), *packed));
        auto kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(packed.get(), 0);
        static_cast<ITensorKernel *>(kernel.get())->setGPUStream(stream);
        return std::tuple{
            std::move(weights),
            std::move(packed),
            std::move(kernel)};
    };

    std::vector<std::unique_ptr<TensorBase>> gate_weights;
    std::vector<std::unique_ptr<TensorBase>> up_weights;
    std::vector<std::unique_ptr<TensorBase>> down_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> gate_packed;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> up_packed;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> down_packed;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> gate_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> up_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;
    gate_weights.reserve(num_experts);
    up_weights.reserve(num_experts);
    down_weights.reserve(num_experts);
    gate_packed.reserve(num_experts);
    up_packed.reserve(num_experts);
    down_packed.reserve(num_experts);
    gate_kernels.reserve(num_experts);
    up_kernels.reserve(num_experts);
    down_kernels.reserve(num_experts);

    WorkspaceRequirements gemm_reqs;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        auto [gate_w, gate_p, gate_k] = make_kernel(
            TestTensorFactory::createQ4_0Random(
                {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}));
        auto [up_w, up_p, up_k] = make_kernel(
            TestTensorFactory::createQ4_0Random(
                {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}));
        auto [down_w, down_p, down_k] = make_kernel(
            TestTensorFactory::createQ4_0Random(
                {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}));

        gemm_reqs.merge(gate_k->getWorkspaceRequirements(1, intermediate, d_model));
        gemm_reqs.merge(up_k->getWorkspaceRequirements(1, intermediate, d_model));
        gemm_reqs.merge(down_k->getWorkspaceRequirements(1, d_model, intermediate));

        gate_weights.push_back(std::move(gate_w));
        up_weights.push_back(std::move(up_w));
        down_weights.push_back(std::move(down_w));
        gate_packed.push_back(std::move(gate_p));
        up_packed.push_back(std::move(up_p));
        down_packed.push_back(std::move(down_p));
        gate_kernels.push_back(std::move(gate_k));
        up_kernels.push_back(std::move(up_k));
        down_kernels.push_back(std::move(down_k));
    }

    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));
    for (int expert = 0; expert < num_experts; ++expert)
    {
        gate_kernels[expert]->bindWorkspace(gemm_workspace.get());
        up_kernels[expert]->bindWorkspace(gemm_workspace.get());
        down_kernels[expert]->bindWorkspace(gemm_workspace.get());
        gate_kernels[expert]->prepareWeights();
        up_kernels[expert]->prepareWeights();
        down_kernels[expert]->prepareWeights();
    }

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        ASSERT_TRUE(gate_kernels[expert]->exportNativeVNNIMatrixDesc(gate_descs[expert]));
        ASSERT_TRUE(up_kernels[expert]->exportNativeVNNIMatrixDesc(up_descs[expert]));
        ASSERT_TRUE(down_kernels[expert]->exportNativeVNNIMatrixDesc(down_descs[expert]));
    }

    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto hidden = TestTensorFactory::createFP32Random(
        {1, static_cast<size_t>(d_model)}, -0.5f, 0.5f, 8181);
    auto two_step_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    auto fused_output = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
    ASSERT_TRUE(two_step_output->ensureOnDevice(device, stream));
    ASSERT_TRUE(fused_output->ensureOnDevice(device, stream));

    std::array<std::shared_ptr<FP32Tensor>, top_k> gate_tensors;
    std::array<std::shared_ptr<FP32Tensor>, top_k> up_tensors;
    std::array<ITensor *, top_k> gate_ptrs = {};
    std::array<ITensor *, top_k> up_ptrs = {};
    for (int slot = 0; slot < top_k; ++slot)
    {
        gate_tensors[slot] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        up_tensors[slot] = TestTensorFactory::createFP32({1, static_cast<size_t>(intermediate)});
        ASSERT_TRUE(gate_tensors[slot]->ensureOnDevice(device, stream));
        ASSERT_TRUE(up_tensors[slot]->ensureOnDevice(device, stream));
        gate_ptrs[slot] = gate_tensors[slot].get();
        up_ptrs[slot] = up_tensors[slot].get();
    }

    DeviceMoELayerRuntime host_runtime{};
    host_runtime.expert_count = static_cast<uint32_t>(num_experts);
    host_runtime.top_k = static_cast<uint32_t>(top_k);
    for (int slot = 0; slot < top_k; ++slot)
    {
        host_runtime.topk_expert_ids[slot] = slot;
        host_runtime.topk_weights[slot] = 0.10f + 0.05f * static_cast<float>(slot);
    }
    DeviceMoELayerRuntime *device_runtime = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_runtime), sizeof(DeviceMoELayerRuntime)), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(
                  device_runtime, &host_runtime, sizeof(DeviceMoELayerRuntime),
                  hipMemcpyHostToDevice, stream),
              hipSuccess);

    moe_kernel.zeroBuffer(two_step_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    moe_kernel.zeroBuffer(fused_output.get(), static_cast<size_t>(d_model) * sizeof(float));
    ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromRuntime(
        device_runtime, hidden.get(), gateup_table, top_k,
        gate_ptrs.data(), up_ptrs.data(), d_model, intermediate));
    ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromRuntime(
        gate_ptrs.data(), up_ptrs.data(), device_runtime, down_table, top_k,
        two_step_output.get(), d_model, intermediate));

    PerfStatsCollector::reset();
    ASSERT_TRUE(moe_kernel.groupedExpertDecodeFromRuntime(
        device_runtime, hidden.get(), gateup_table, down_table, top_k,
        fused_output.get(), d_model, intermediate));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    two_step_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    fused_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    expectStrictVerifierSimilarity(
        "ROCm fused grouped decode must match two-step decode",
        fused_output->data(),
        two_step_output->data(),
        static_cast<size_t>(d_model),
        static_cast<size_t>(d_model));

    const auto counters = PerfStatsCollector::snapshot(
        {"kernel.rocm_moe_grouped_decode_fused_calls"});
    ASSERT_FALSE(counters.empty()) << PerfStatsCollector::summaryString(
        {"kernel.rocm_moe_grouped_decode_fused_calls"});

    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
    const bool captured_fused = moe_kernel.groupedExpertDecodeFromRuntime(
        device_runtime, hidden.get(), gateup_table, down_table, top_k,
        fused_output.get(), d_model, intermediate);
    const hipError_t capture_status = hipStreamEndCapture(stream, &graph);
    EXPECT_TRUE(captured_fused);
    ASSERT_EQ(capture_status, hipSuccess) << hipGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    ASSERT_EQ(hipFree(device_runtime), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, GroupedPrefill_Q4KGateUp_Q5KDownMatchesSequentialGemm)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 4;
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> gate_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> up_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> down_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> gate_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> up_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)});
        auto up_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)});
        auto down_weights = TestTensorFactory::createQ5_KRandom(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)});
        injectNonZeroKQuantMins(gate_weights.get());
        injectNonZeroKQuantMins(up_weights.get());
        injectNonZeroKQuantMins(down_weights.get());

        auto gate_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto up_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto down_packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(gate_weights.get(), *gate_packed));
        ASSERT_TRUE(rocm::packWeightsToROCm(up_weights.get(), *up_packed));
        ASSERT_TRUE(rocm::packWeightsToROCm(down_weights.get(), *down_packed));

        gate_kernels.push_back(std::make_unique<rocm::ROCmQuantisedGemmKernel>(gate_packed.get(), 0));
        up_kernels.push_back(std::make_unique<rocm::ROCmQuantisedGemmKernel>(up_packed.get(), 0));
        down_kernels.push_back(std::make_unique<rocm::ROCmQuantisedGemmKernel>(down_packed.get(), 0));
        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        down_weight_tensors.push_back(std::move(down_weights));
        gate_packed_weights.push_back(std::move(gate_packed));
        up_packed_weights.push_back(std::move(up_packed));
        down_packed_weights.push_back(std::move(down_packed));
    }

    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 128 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(gate_kernels[0]->getWorkspaceRequirements(seq_len, intermediate, d_model)));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_kernels[expert_id]->bindWorkspace(workspace.get());
        up_kernels[expert_id]->bindWorkspace(workspace.get());
        down_kernels[expert_id]->bindWorkspace(workspace.get());
        gate_kernels[expert_id]->prepareWeights();
        up_kernels[expert_id]->prepareWeights();
        down_kernels[expert_id]->prepareWeights();
    }

    auto make_q4k_parent = [&](const std::vector<std::unique_ptr<TensorBase>> &experts)
    {
        std::vector<uint8_t> raw;
        for (const auto &expert : experts)
        {
            const auto *bytes = static_cast<const uint8_t *>(expert->raw_data());
            raw.insert(raw.end(), bytes, bytes + expert->size_bytes());
        }
        return std::make_shared<Q4_KTensor>(
            std::vector<size_t>{static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)},
            raw);
    };
    auto make_q5k_parent = [&](const std::vector<std::unique_ptr<TensorBase>> &experts)
    {
        std::vector<uint8_t> raw;
        for (const auto &expert : experts)
        {
            const auto *bytes = static_cast<const uint8_t *>(expert->raw_data());
            raw.insert(raw.end(), bytes, bytes + expert->size_bytes());
        }
        return std::make_shared<Q5_KTensor>(
            std::vector<size_t>{static_cast<size_t>(intermediate), static_cast<size_t>(d_model), static_cast<size_t>(num_experts)},
            raw);
    };

    auto gate_parent = make_q4k_parent(gate_weight_tensors);
    auto up_parent = make_q4k_parent(up_weight_tensors);
    auto down_parent = make_q5k_parent(down_weight_tensors);
    std::vector<bool> expert_mask;
    std::vector<std::shared_ptr<TensorBase>> gpu_gate_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_up_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_down_views;
    std::vector<ITensorGemm *> gpu_gate_gemms;
    std::vector<ITensorGemm *> gpu_up_gemms;
    std::vector<ITensorGemm *> gpu_down_gemms;
    std::vector<std::shared_ptr<ITensorGemm>> gpu_owned_kernels;
    std::shared_ptr<void> gpu_gate_lifetime;
    std::shared_ptr<void> gpu_up_lifetime;
    std::shared_ptr<void> gpu_down_lifetime;
    MoEWeightContext gpu_ctx{
        device,
        num_experts,
        intermediate,
        d_model,
        0,
        num_experts,
        0,
        expert_mask,
        gate_parent.get(),
        up_parent.get(),
        down_parent.get(),
        gpu_gate_views,
        gpu_up_views,
        gpu_down_views,
        gpu_gate_gemms,
        gpu_up_gemms,
        gpu_down_gemms,
        gpu_owned_kernels,
        gpu_gate_lifetime,
        gpu_up_lifetime,
        gpu_down_lifetime};
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(gpu_ctx));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(gpu_ctx));

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -1.0f, 1.0f, 901);
    const float *input_host = input->data();
    ASSERT_TRUE(input->ensureOnDevice(device));

    const int routing_indices_data[total_slots] = {
        0, 1,
        2, 3,
        0, 2,
        3, 1};
    const float routing_weights_data[total_slots] = {
        0.70f, 0.30f,
        0.60f, 0.40f,
        0.55f, 0.45f,
        0.80f, 0.20f};

    std::vector<float> expected(static_cast<size_t>(seq_len) * d_model, 0.0f);
    std::vector<std::vector<std::pair<int, float>>> tokens_by_expert(num_experts);
    for (int token = 0; token < seq_len; ++token)
    {
        for (int route = 0; route < top_k; ++route)
        {
            const int slot = token * top_k + route;
            tokens_by_expert[static_cast<size_t>(routing_indices_data[slot])].push_back(
                {token, routing_weights_data[slot]});
        }
    }

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        const auto &routes = tokens_by_expert[static_cast<size_t>(expert_id)];
        if (routes.empty())
            continue;

        const int count = static_cast<int>(routes.size());
        auto batch = TestTensorFactory::createFP32(
            {static_cast<size_t>(count), static_cast<size_t>(d_model)});
        for (int row = 0; row < count; ++row)
        {
            const int token = routes[static_cast<size_t>(row)].first;
            std::copy(input_host + static_cast<size_t>(token) * d_model,
                      input_host + static_cast<size_t>(token + 1) * d_model,
                      batch->mutable_data() + static_cast<size_t>(row) * d_model);
        }
        ASSERT_TRUE(batch->ensureOnDevice(device));

        auto gate = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto up = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto expert_out = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(d_model)});
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        ASSERT_TRUE(expert_out->ensureOnDevice(device));

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {gate_kernels[expert_id].get(), gate.get(), intermediate, nullptr, "gate"},
            {up_kernels[expert_id].get(), up.get(), intermediate, nullptr, "up"}};
        ASSERT_TRUE(gate_kernels[expert_id]->multiply_fused_tensor(batch.get(), projections, count, d_model));
        gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        up->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        ASSERT_TRUE(down_kernels[expert_id]->multiply_tensor_with_fused_swiglu(
            gate.get(), up.get(), expert_out.get(), count, d_model, intermediate));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expert_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *expert_host = expert_out->data();
        for (int row = 0; row < count; ++row)
        {
            const int token = routes[static_cast<size_t>(row)].first;
            const float weight = routes[static_cast<size_t>(row)].second;
            for (int col = 0; col < d_model; ++col)
            {
                expected[static_cast<size_t>(token) * d_model + col] +=
                    weight * expert_host[static_cast<size_t>(row) * d_model + col];
            }
        }
    }

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    for (int slot = 0; slot < total_slots; ++slot)
    {
        routing_indices->mutable_data()[slot] = static_cast<float>(routing_indices_data[slot]);
        routing_weights->mutable_data()[slot] = routing_weights_data[slot];
    }
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));
    ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
        routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->exportNativeVNNIMatrixDesc(gate_descs[expert_id]));
        ASSERT_TRUE(gpu_up_gemms[expert_id]->exportNativeVNNIMatrixDesc(up_descs[expert_id]));
        ASSERT_TRUE(gpu_down_gemms[expert_id]->exportNativeVNNIMatrixDesc(down_descs[expert_id]));
    }

    const int gateup_table_id = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table_id, 0);
    const int down_table_id = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table_id, 0);

    auto grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(seq_len) * d_model * sizeof(float));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
        input.get(), grouped_output.get(), gateup_table_id, down_table_id,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *grouped = grouped_output->data();
    expectStrictVerifierSimilarity(
        "ROCm grouped prefill must match sequential GPU rows",
        grouped,
        expected.data(),
        expected.size(),
        static_cast<size_t>(d_model));
}

TEST(Test__ROCmMoEKernel, GroupedPrefill_Qwen35RouteTable_Q4KQ5KMatchesCpuDequantReference)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 9;
    constexpr int d_model = 512;
    constexpr int intermediate = 256;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int total_slots = seq_len * top_k;

    const std::vector<int> routing_indices = {
        112, 106, 107, 238, 181, 57, 200, 43,
        109, 41, 65, 24, 86, 29, 136, 2,
        134, 115, 220, 159, 41, 2, 224, 28,
        13, 131, 139, 47, 151, 159, 192, 207,
        13, 234, 242, 188, 144, 186, 217, 159,
        96, 187, 242, 117, 52, 133, 27, 250,
        181, 107, 82, 251, 80, 68, 59, 131,
        109, 41, 2, 24, 177, 131, 5, 225,
        139, 13, 159, 47, 116, 55, 220, 131};
    ASSERT_EQ(routing_indices.size(), static_cast<size_t>(total_slots));

    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    for (int token_idx = 0; token_idx < seq_len; ++token_idx)
    {
        float weight_sum = 0.0f;
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routing_weights[static_cast<size_t>(slot_idx)] =
                (1.0f / static_cast<float>(route_idx + 2)) + 0.003f * static_cast<float>(token_idx + 1);
            weight_sum += routing_weights[static_cast<size_t>(slot_idx)];
        }
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routing_weights[static_cast<size_t>(slot_idx)] /= weight_sum;
        }
    }

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    down_weight_tensors.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 10000 + expert_id);
        auto up_weights = TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 20000 + expert_id);
        auto down_weights = TestTensorFactory::createQ5_KRandom(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 30000 + expert_id);
        injectNonZeroKQuantMins(gate_weights.get());
        injectNonZeroKQuantMins(up_weights.get());
        injectNonZeroKQuantMins(down_weights.get());

        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        down_weight_tensors.push_back(std::move(down_weights));
    }

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -0.05f, 0.05f, 3519);
    const float *input_host = input->data();

    const std::vector<float> reference = computeCpuDequantMoEPrefillReference(
        input_host, seq_len, d_model, intermediate, num_experts, top_k,
        gate_weight_tensors, up_weight_tensors, down_weight_tensors,
        routing_indices, routing_weights);

    auto gate_parent = makeExpertParentTensor<Q4_KTensor>(
        gate_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto up_parent = makeExpertParentTensor<Q4_KTensor>(
        up_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto down_parent = makeExpertParentTensor<Q5_KTensor>(
        down_weight_tensors,
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model), static_cast<size_t>(num_experts)});

    std::vector<bool> expert_mask;
    std::vector<std::shared_ptr<TensorBase>> gpu_gate_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_up_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_down_views;
    std::vector<ITensorGemm *> gpu_gate_gemms;
    std::vector<ITensorGemm *> gpu_up_gemms;
    std::vector<ITensorGemm *> gpu_down_gemms;
    std::vector<std::shared_ptr<ITensorGemm>> gpu_owned_kernels;
    std::shared_ptr<void> gpu_gate_lifetime;
    std::shared_ptr<void> gpu_up_lifetime;
    std::shared_ptr<void> gpu_down_lifetime;
    MoEWeightContext gpu_ctx{
        device,
        num_experts,
        intermediate,
        d_model,
        0,
        num_experts,
        0,
        expert_mask,
        gate_parent.get(),
        up_parent.get(),
        down_parent.get(),
        gpu_gate_views,
        gpu_up_views,
        gpu_down_views,
        gpu_gate_gemms,
        gpu_up_gemms,
        gpu_down_gemms,
        gpu_owned_kernels,
        gpu_gate_lifetime,
        gpu_up_lifetime,
        gpu_down_lifetime};
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(gpu_ctx));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(gpu_ctx));

    auto *gate0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_gate_gemms[0]);
    auto *up0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_up_gemms[0]);
    auto *down0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_down_gemms[0]);
    ASSERT_NE(gate0_workspace, nullptr);
    ASSERT_NE(up0_workspace, nullptr);
    ASSERT_NE(down0_workspace, nullptr);

    WorkspaceRequirements sequential_reqs;
    sequential_reqs.merge(gate0_workspace->getWorkspaceRequirements(seq_len, intermediate, d_model));
    sequential_reqs.merge(up0_workspace->getWorkspaceRequirements(seq_len, intermediate, d_model));
    sequential_reqs.merge(down0_workspace->getWorkspaceRequirements(seq_len, d_model, intermediate));
    auto sequential_workspace = std::make_unique<DeviceWorkspaceManager>(device, 128 * 1024 * 1024);
    ASSERT_TRUE(sequential_workspace->allocate(sequential_reqs));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto *gate_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_gate_gemms[expert_id]);
        auto *up_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_up_gemms[expert_id]);
        auto *down_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_down_gemms[expert_id]);
        ASSERT_NE(gate_workspace, nullptr);
        ASSERT_NE(up_workspace, nullptr);
        ASSERT_NE(down_workspace, nullptr);
        gate_workspace->bindWorkspace(sequential_workspace.get());
        up_workspace->bindWorkspace(sequential_workspace.get());
        down_workspace->bindWorkspace(sequential_workspace.get());
    }

    std::vector<float> sequential_output(static_cast<size_t>(seq_len) * d_model, 0.0f);
    std::vector<std::vector<std::pair<int, float>>> routes_by_expert(static_cast<size_t>(num_experts));
    for (int token_idx = 0; token_idx < seq_len; ++token_idx)
    {
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routes_by_expert[static_cast<size_t>(routing_indices[static_cast<size_t>(slot_idx)])].push_back(
                {token_idx, routing_weights[static_cast<size_t>(slot_idx)]});
        }
    }

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        const auto &routes = routes_by_expert[static_cast<size_t>(expert_id)];
        if (routes.empty())
            continue;

        const int count = static_cast<int>(routes.size());
        auto batch = TestTensorFactory::createFP32(
            {static_cast<size_t>(count), static_cast<size_t>(d_model)});
        for (int row = 0; row < count; ++row)
        {
            const int token_idx = routes[static_cast<size_t>(row)].first;
            std::copy(input_host + static_cast<size_t>(token_idx) * d_model,
                      input_host + static_cast<size_t>(token_idx + 1) * d_model,
                      batch->mutable_data() + static_cast<size_t>(row) * d_model);
        }
        ASSERT_TRUE(batch->ensureOnDevice(device));

        auto gate = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto up = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto expert_out = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(d_model)});
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        ASSERT_TRUE(expert_out->ensureOnDevice(device));

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {gpu_gate_gemms[expert_id], gate.get(), intermediate, nullptr, "gate"},
            {gpu_up_gemms[expert_id], up.get(), intermediate, nullptr, "up"}};
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->multiply_fused_tensor(
            batch.get(), projections, count, d_model));
        gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        up->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        ASSERT_TRUE(gpu_down_gemms[expert_id]->multiply_tensor_with_fused_swiglu(
            gate.get(), up.get(), expert_out.get(), count, d_model, intermediate));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expert_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        const float *expert_host = expert_out->data();
        for (int row = 0; row < count; ++row)
        {
            const int token_idx = routes[static_cast<size_t>(row)].first;
            const float route_weight = routes[static_cast<size_t>(row)].second;
            for (int col = 0; col < d_model; ++col)
            {
                sequential_output[static_cast<size_t>(token_idx) * d_model + col] +=
                    route_weight * expert_host[static_cast<size_t>(row) * d_model + col];
            }
        }
    }

    const double sequential_cosine = cosineSimilarity(
        sequential_output.data(), reference.data(), reference.size());
    const double sequential_rel_l2 = relativeL2Error(
        sequential_output.data(), reference.data(), reference.size());
    const double sequential_max_abs = maxAbsDiff(
        sequential_output.data(), reference.data(), reference.size());
    EXPECT_GE(sequential_cosine, 0.960) << "relative L2=" << sequential_rel_l2
                                        << " max_abs=" << sequential_max_abs;
    EXPECT_LE(sequential_rel_l2, 0.35) << "cosine=" << sequential_cosine
                                       << " max_abs=" << sequential_max_abs;

    auto routing_indices_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    for (int slot_idx = 0; slot_idx < total_slots; ++slot_idx)
    {
        routing_indices_tensor->mutable_data()[slot_idx] =
            static_cast<float>(routing_indices[static_cast<size_t>(slot_idx)]);
        routing_weights_tensor->mutable_data()[slot_idx] = routing_weights[static_cast<size_t>(slot_idx)];
    }
    ASSERT_TRUE(input->ensureOnDevice(device));
    ASSERT_TRUE(routing_indices_tensor->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights_tensor->ensureOnDevice(device));

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);
    ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
        routing_indices_tensor.get(), routing_weights_tensor.get(), seq_len, num_experts, top_k));

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->exportNativeVNNIMatrixDesc(gate_descs[expert_id]));
        ASSERT_TRUE(gpu_up_gemms[expert_id]->exportNativeVNNIMatrixDesc(up_descs[expert_id]));
        ASSERT_TRUE(gpu_down_gemms[expert_id]->exportNativeVNNIMatrixDesc(down_descs[expert_id]));
    }

    const int gateup_table_id = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table_id, 0);
    const int down_table_id = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table_id, 0);

    auto grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(seq_len) * d_model * sizeof(float));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
        input.get(), grouped_output.get(), gateup_table_id, down_table_id,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *grouped = grouped_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, reference.size()));

    const double cosine = cosineSimilarity(grouped, reference.data(), reference.size());
    const double rel_l2 = relativeL2Error(grouped, reference.data(), reference.size());
    const double max_abs = maxAbsDiff(grouped, reference.data(), reference.size());
    EXPECT_GE(cosine, 0.985) << "relative L2=" << rel_l2 << " max_abs=" << max_abs;
    EXPECT_LE(rel_l2, 0.20) << "cosine=" << cosine << " max_abs=" << max_abs;

    std::cout << "[GroupedPrefill_Qwen35RouteTable_Q4KQ5KMatchesCpuDequantReference] cosine="
              << std::fixed << std::setprecision(6) << cosine
              << " rel_l2=" << rel_l2
              << " max_abs=" << max_abs << std::endl;
}

TEST(Test__ROCmMoEKernel, GroupedPrefill_Qwen36RouteTable_IQ2SGateUp_IQ4XSDownMatchesCpuDequantReference)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 9;
    constexpr int d_model = 512;
    constexpr int intermediate = 256;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int total_slots = seq_len * top_k;

    const std::vector<int> routing_indices = {
        112, 106, 107, 238, 181, 57, 200, 43,
        109, 41, 65, 24, 86, 29, 136, 2,
        134, 115, 220, 159, 41, 2, 224, 28,
        13, 131, 139, 47, 151, 159, 192, 207,
        13, 234, 242, 188, 144, 186, 217, 159,
        96, 187, 242, 117, 52, 133, 27, 250,
        181, 107, 82, 251, 80, 68, 59, 131,
        109, 41, 2, 24, 177, 131, 5, 225,
        139, 13, 159, 47, 116, 55, 220, 131};
    ASSERT_EQ(routing_indices.size(), static_cast<size_t>(total_slots));

    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    for (int token_idx = 0; token_idx < seq_len; ++token_idx)
    {
        float weight_sum = 0.0f;
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routing_weights[static_cast<size_t>(slot_idx)] =
                (1.0f / static_cast<float>(route_idx + 2)) + 0.003f * static_cast<float>(token_idx + 1);
            weight_sum += routing_weights[static_cast<size_t>(slot_idx)];
        }
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routing_weights[static_cast<size_t>(slot_idx)] /= weight_sum;
        }
    }

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    down_weight_tensors.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_weight_tensors.push_back(TestTensorFactory::createIQ2_SRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 40000 + expert_id));
        up_weight_tensors.push_back(TestTensorFactory::createIQ2_SRandom(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 50000 + expert_id));
        down_weight_tensors.push_back(TestTensorFactory::createIQ4_XSRandom(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 60000 + expert_id));
    }

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -0.05f, 0.05f, 3619);
    const float *input_host = input->data();

    const std::vector<float> reference = computeCpuDequantMoEPrefillReference(
        input_host, seq_len, d_model, intermediate, num_experts, top_k,
        gate_weight_tensors, up_weight_tensors, down_weight_tensors,
        routing_indices, routing_weights);

    auto gate_parent = makeExpertParentTensor<IQ2_STensor>(
        gate_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto up_parent = makeExpertParentTensor<IQ2_STensor>(
        up_weight_tensors,
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate), static_cast<size_t>(num_experts)});
    auto down_parent = makeExpertParentTensor<IQ4_XSTensor>(
        down_weight_tensors,
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model), static_cast<size_t>(num_experts)});

    std::vector<bool> expert_mask;
    std::vector<std::shared_ptr<TensorBase>> gpu_gate_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_up_views;
    std::vector<std::shared_ptr<TensorBase>> gpu_down_views;
    std::vector<ITensorGemm *> gpu_gate_gemms;
    std::vector<ITensorGemm *> gpu_up_gemms;
    std::vector<ITensorGemm *> gpu_down_gemms;
    std::vector<std::shared_ptr<ITensorGemm>> gpu_owned_kernels;
    std::shared_ptr<void> gpu_gate_lifetime;
    std::shared_ptr<void> gpu_up_lifetime;
    std::shared_ptr<void> gpu_down_lifetime;
    MoEWeightContext gpu_ctx{
        device,
        num_experts,
        intermediate,
        d_model,
        0,
        num_experts,
        0,
        expert_mask,
        gate_parent.get(),
        up_parent.get(),
        down_parent.get(),
        gpu_gate_views,
        gpu_up_views,
        gpu_down_views,
        gpu_gate_gemms,
        gpu_up_gemms,
        gpu_down_gemms,
        gpu_owned_kernels,
        gpu_gate_lifetime,
        gpu_up_lifetime,
        gpu_down_lifetime};
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(gpu_ctx));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(gpu_ctx));

    auto *gate0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_gate_gemms[0]);
    auto *up0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_up_gemms[0]);
    auto *down0_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_down_gemms[0]);
    ASSERT_NE(gate0_workspace, nullptr);
    ASSERT_NE(up0_workspace, nullptr);
    ASSERT_NE(down0_workspace, nullptr);

    WorkspaceRequirements sequential_reqs;
    sequential_reqs.merge(gate0_workspace->getWorkspaceRequirements(seq_len, intermediate, d_model));
    sequential_reqs.merge(up0_workspace->getWorkspaceRequirements(seq_len, intermediate, d_model));
    sequential_reqs.merge(down0_workspace->getWorkspaceRequirements(seq_len, d_model, intermediate));
    auto sequential_workspace = std::make_unique<DeviceWorkspaceManager>(device, 128 * 1024 * 1024);
    ASSERT_TRUE(sequential_workspace->allocate(sequential_reqs));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto *gate_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_gate_gemms[expert_id]);
        auto *up_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_up_gemms[expert_id]);
        auto *down_workspace = dynamic_cast<IWorkspaceConsumer *>(gpu_down_gemms[expert_id]);
        ASSERT_NE(gate_workspace, nullptr);
        ASSERT_NE(up_workspace, nullptr);
        ASSERT_NE(down_workspace, nullptr);
        gate_workspace->bindWorkspace(sequential_workspace.get());
        up_workspace->bindWorkspace(sequential_workspace.get());
        down_workspace->bindWorkspace(sequential_workspace.get());
    }

    std::vector<float> sequential_output(static_cast<size_t>(seq_len) * d_model, 0.0f);
    std::vector<std::vector<std::pair<int, float>>> routes_by_expert(static_cast<size_t>(num_experts));
    for (int token_idx = 0; token_idx < seq_len; ++token_idx)
    {
        for (int route_idx = 0; route_idx < top_k; ++route_idx)
        {
            const int slot_idx = token_idx * top_k + route_idx;
            routes_by_expert[static_cast<size_t>(routing_indices[static_cast<size_t>(slot_idx)])].push_back(
                {token_idx, routing_weights[static_cast<size_t>(slot_idx)]});
        }
    }

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        const auto &routes = routes_by_expert[static_cast<size_t>(expert_id)];
        if (routes.empty())
            continue;

        const int count = static_cast<int>(routes.size());
        auto batch = TestTensorFactory::createFP32(
            {static_cast<size_t>(count), static_cast<size_t>(d_model)});
        for (int row = 0; row < count; ++row)
        {
            const int token_idx = routes[static_cast<size_t>(row)].first;
            std::copy(input_host + static_cast<size_t>(token_idx) * d_model,
                      input_host + static_cast<size_t>(token_idx + 1) * d_model,
                      batch->mutable_data() + static_cast<size_t>(row) * d_model);
        }
        ASSERT_TRUE(batch->ensureOnDevice(device));

        auto gate = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto up = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(intermediate)});
        auto expert_out = TestTensorFactory::createFP32({static_cast<size_t>(count), static_cast<size_t>(d_model)});
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        ASSERT_TRUE(expert_out->ensureOnDevice(device));

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {gpu_gate_gemms[expert_id], gate.get(), intermediate, nullptr, "gate"},
            {gpu_up_gemms[expert_id], up.get(), intermediate, nullptr, "up"}};
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->multiply_fused_tensor(
            batch.get(), projections, count, d_model));
        gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        up->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        ASSERT_TRUE(gpu_down_gemms[expert_id]->multiply_tensor_with_fused_swiglu(
            gate.get(), up.get(), expert_out.get(), count, d_model, intermediate));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expert_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        const float *expert_host = expert_out->data();
        for (int row = 0; row < count; ++row)
        {
            const int token_idx = routes[static_cast<size_t>(row)].first;
            const float route_weight = routes[static_cast<size_t>(row)].second;
            for (int col = 0; col < d_model; ++col)
            {
                sequential_output[static_cast<size_t>(token_idx) * d_model + col] +=
                    route_weight * expert_host[static_cast<size_t>(row) * d_model + col];
            }
        }
    }

    const double sequential_cosine = cosineSimilarity(
        sequential_output.data(), reference.data(), reference.size());
    const double sequential_rel_l2 = relativeL2Error(
        sequential_output.data(), reference.data(), reference.size());
    const double sequential_max_abs = maxAbsDiff(
        sequential_output.data(), reference.data(), reference.size());
    EXPECT_GE(sequential_cosine, 0.960) << "relative L2=" << sequential_rel_l2
                                        << " max_abs=" << sequential_max_abs;
    EXPECT_LE(sequential_rel_l2, 0.35) << "cosine=" << sequential_cosine
                                       << " max_abs=" << sequential_max_abs;

    auto routing_indices_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    for (int slot_idx = 0; slot_idx < total_slots; ++slot_idx)
    {
        routing_indices_tensor->mutable_data()[slot_idx] =
            static_cast<float>(routing_indices[static_cast<size_t>(slot_idx)]);
        routing_weights_tensor->mutable_data()[slot_idx] = routing_weights[static_cast<size_t>(slot_idx)];
    }
    ASSERT_TRUE(input->ensureOnDevice(device));
    ASSERT_TRUE(routing_indices_tensor->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights_tensor->ensureOnDevice(device));

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);
    ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
        routing_indices_tensor.get(), routing_weights_tensor.get(), seq_len, num_experts, top_k));

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(num_experts);
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(num_experts);
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        ASSERT_TRUE(gpu_gate_gemms[expert_id]->exportNativeVNNIMatrixDesc(gate_descs[expert_id]));
        ASSERT_TRUE(gpu_up_gemms[expert_id]->exportNativeVNNIMatrixDesc(up_descs[expert_id]));
        ASSERT_TRUE(gpu_down_gemms[expert_id]->exportNativeVNNIMatrixDesc(down_descs[expert_id]));
    }

    const int gateup_table_id = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table_id, 0);
    const int down_table_id = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table_id, 0);

    auto grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    moe_kernel.zeroBuffer(grouped_output.get(), static_cast<size_t>(seq_len) * d_model * sizeof(float));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
        input.get(), grouped_output.get(), gateup_table_id, down_table_id,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *grouped = grouped_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, reference.size()));

    const double cosine = cosineSimilarity(grouped, reference.data(), reference.size());
    const double rel_l2 = relativeL2Error(grouped, reference.data(), reference.size());
    const double max_abs = maxAbsDiff(grouped, reference.data(), reference.size());
    EXPECT_GE(cosine, 0.960) << "relative L2=" << rel_l2 << " max_abs=" << max_abs;
    EXPECT_LE(rel_l2, 0.35) << "cosine=" << cosine << " max_abs=" << max_abs;

    std::cout << "[GroupedPrefill_Qwen36RouteTable_IQ2SGateUp_IQ4XSDownMatchesCpuDequantReference] cosine="
              << std::fixed << std::setprecision(6) << cosine
              << " rel_l2=" << rel_l2
              << " max_abs=" << max_abs << std::endl;
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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    gpu_kernel.recordHistogramDevice(d_indices, seq_len, top_k, layer_idx);

    // Verify something was recorded
    std::vector<uint64_t> counts_before(num_experts, 0);
    gpu_kernel.syncHistogramToHost(counts_before.data(), layer_idx, num_experts);
    uint64_t sum_before = 0;
    for (auto c : counts_before)
        sum_before += c;
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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
            if (result_weights[i] == original_weights[i])
                ++unchanged_count;
        }
        else
        {
            EXPECT_FLOAT_EQ(result_weights[i], 0.0f)
                << "Inactive expert " << expert << " weight at slot " << i
                << " should be zeroed";
            if (result_weights[i] == 0.0f)
                ++zeroed_count;
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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
        1, 3, // token 0
        0, 2, // token 1
        0, 1, // token 2
        2, 3, // token 3
        0, 3, // token 4
        1, 2, // token 5
        0, 1, // token 6
        2, 3  // token 7
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
    struct SlotInfo
    {
        int expert;
        int token;
        float weight;
    };
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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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

TEST(Test__ROCmMoEKernel, SmallFloatGrouping_VerifierSizedRoutesMatchHostGrouping)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 2;
    constexpr int num_experts = 16;
    constexpr int top_k = 8;
    constexpr int total_slots = seq_len * top_k;

    std::vector<float> routing_indices(static_cast<size_t>(total_slots));
    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    std::vector<std::vector<std::pair<int, float>>> expected_per_expert(static_cast<size_t>(num_experts));
    for (int slot = 0; slot < total_slots; ++slot)
    {
        const int token = slot / top_k;
        const int expert = (slot * 5 + token * 3) % num_experts;
        const float weight = 0.03125f * static_cast<float>(slot + 1);
        routing_indices[static_cast<size_t>(slot)] = static_cast<float>(expert);
        routing_weights[static_cast<size_t>(slot)] = weight;
        expected_per_expert[static_cast<size_t>(expert)].push_back({token, weight});
    }

    float *d_routing_indices = nullptr;
    float *d_routing_weights = nullptr;
    int *d_expert_offsets = nullptr;
    int *d_expert_counts = nullptr;
    int *d_grouped_indices = nullptr;
    int *d_original_to_grouped = nullptr;
    float *d_grouped_weights = nullptr;
    int *d_active_experts = nullptr;
    const int max_active_experts = std::min(total_slots, num_experts);

    ASSERT_EQ(hipMalloc(&d_routing_indices, total_slots * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_routing_weights, total_slots * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_expert_offsets, num_experts * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_expert_counts, num_experts * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_grouped_indices, total_slots * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_original_to_grouped, total_slots * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_grouped_weights, total_slots * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_active_experts, max_active_experts * sizeof(int)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_routing_indices, routing_indices.data(),
                        total_slots * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(d_routing_weights, routing_weights.data(),
                        total_slots * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_TRUE(hipMoE_group_tokens_small_float(
        d_routing_indices,
        d_routing_weights,
        d_expert_counts,
        d_expert_offsets,
        d_grouped_indices,
        d_original_to_grouped,
        d_grouped_weights,
        d_active_experts,
        total_slots,
        num_experts,
        top_k,
        max_active_experts,
        /*device_idx=*/0,
        stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    std::vector<int> host_offsets(static_cast<size_t>(num_experts));
    std::vector<int> host_counts(static_cast<size_t>(num_experts));
    std::vector<int> host_grouped_indices(static_cast<size_t>(total_slots));
    std::vector<int> host_original_to_grouped(static_cast<size_t>(total_slots));
    std::vector<float> host_grouped_weights(static_cast<size_t>(total_slots));
    std::vector<int> host_active_experts(static_cast<size_t>(max_active_experts));
    ASSERT_EQ(hipMemcpy(host_offsets.data(), d_expert_offsets,
                        num_experts * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_counts.data(), d_expert_counts,
                        num_experts * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_indices.data(), d_grouped_indices,
                        total_slots * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_original_to_grouped.data(), d_original_to_grouped,
                        total_slots * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_weights.data(), d_grouped_weights,
                        total_slots * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_active_experts.data(), d_active_experts,
                        max_active_experts * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);

    ASSERT_EQ(hipFree(d_routing_indices), hipSuccess);
    ASSERT_EQ(hipFree(d_routing_weights), hipSuccess);
    ASSERT_EQ(hipFree(d_expert_offsets), hipSuccess);
    ASSERT_EQ(hipFree(d_expert_counts), hipSuccess);
    ASSERT_EQ(hipFree(d_grouped_indices), hipSuccess);
    ASSERT_EQ(hipFree(d_original_to_grouped), hipSuccess);
    ASSERT_EQ(hipFree(d_grouped_weights), hipSuccess);
    ASSERT_EQ(hipFree(d_active_experts), hipSuccess);

    int running_offset = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const auto &expected = expected_per_expert[static_cast<size_t>(expert)];
        ASSERT_EQ(host_counts[static_cast<size_t>(expert)], static_cast<int>(expected.size()))
            << "expert=" << expert;
        ASSERT_EQ(host_offsets[static_cast<size_t>(expert)], running_offset)
            << "expert=" << expert;

        std::vector<std::pair<int, float>> actual;
        actual.reserve(expected.size());
        for (int i = 0; i < host_counts[static_cast<size_t>(expert)]; ++i)
        {
            const int grouped = running_offset + i;
            actual.push_back({host_grouped_indices[static_cast<size_t>(grouped)],
                              host_grouped_weights[static_cast<size_t>(grouped)]});
        }

        for (const auto &entry : expected)
        {
            EXPECT_NE(std::find(actual.begin(), actual.end(), entry), actual.end())
                << "expert=" << expert << " token=" << entry.first
                << " weight=" << entry.second;
        }

        running_offset += host_counts[static_cast<size_t>(expert)];
    }
    EXPECT_EQ(running_offset, total_slots);

    for (int slot = 0; slot < total_slots; ++slot)
    {
        const int token = slot / top_k;
        const int expert = static_cast<int>(routing_indices[static_cast<size_t>(slot)]);
        const int grouped = host_original_to_grouped[static_cast<size_t>(slot)];
        ASSERT_GE(grouped, 0) << "slot=" << slot;
        ASSERT_LT(grouped, total_slots) << "slot=" << slot;
        EXPECT_GE(grouped, host_offsets[static_cast<size_t>(expert)])
            << "slot=" << slot << " expert=" << expert;
        EXPECT_LT(grouped, host_offsets[static_cast<size_t>(expert)] +
                               host_counts[static_cast<size_t>(expert)])
            << "slot=" << slot << " expert=" << expert;
        EXPECT_EQ(host_grouped_indices[static_cast<size_t>(grouped)], token)
            << "slot=" << slot;
        EXPECT_FLOAT_EQ(host_grouped_weights[static_cast<size_t>(grouped)],
                        routing_weights[static_cast<size_t>(slot)])
            << "slot=" << slot;
    }

    std::vector<int> expected_active_experts;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        if (!expected_per_expert[static_cast<size_t>(expert)].empty())
            expected_active_experts.push_back(expert);
    }
    ASSERT_LE(expected_active_experts.size(), host_active_experts.size());
    for (size_t i = 0; i < expected_active_experts.size(); ++i)
        EXPECT_EQ(host_active_experts[i], expected_active_experts[i]) << "active slot " << i;
    for (size_t i = expected_active_experts.size(); i < host_active_experts.size(); ++i)
        EXPECT_EQ(host_active_experts[i], -1) << "inactive slot " << i;
}

TEST(Test__ROCmMoEKernel, SmallFloatGrouping_RejectsNullStream)
{
    SKIP_IF_NO_ROCM();

    EXPECT_FALSE(hipMoE_group_tokens_small_float(
        /*routing_indices=*/nullptr,
        /*routing_weights=*/nullptr,
        /*expert_counts=*/nullptr,
        /*expert_offsets=*/nullptr,
        /*grouped_token_indices=*/nullptr,
        /*original_to_grouped=*/nullptr,
        /*grouped_weights=*/nullptr,
        /*active_expert_ids=*/nullptr,
        /*total_slots=*/8,
        /*num_experts=*/16,
        /*top_k=*/8,
        /*max_active_experts=*/8,
        /*device_idx=*/0,
        /*stream=*/nullptr));
}

TEST(Test__ROCmMoEKernel, SmallMGateLogits_RejectsNullStream)
{
    SKIP_IF_NO_ROCM();

    float *d_hidden = nullptr;
    float *d_gate = nullptr;
    float *d_logits = nullptr;
    ASSERT_EQ(hipMalloc(&d_hidden, 2 * 32 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_gate, 8 * 32 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_logits, 2 * 8 * sizeof(float)), hipSuccess);

    EXPECT_FALSE(hipMoE_gate_logits_small_m(
        d_hidden,
        d_gate,
        d_logits,
        /*seq_len=*/2,
        /*d_model=*/32,
        /*num_experts=*/8,
        /*device_idx=*/0,
        /*stream=*/nullptr));

    hipFree(d_hidden);
    hipFree(d_gate);
    hipFree(d_logits);
}

TEST(Test__ROCmMoEKernel, SmallMGateLogits_ModelShapeMatchesSingleTokenLaunches)
{
    SKIP_IF_NO_ROCM();

    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    const DeviceId device = DeviceId::rocm(0);

    auto gate_weights = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)},
        -0.5f, 0.5f, 20260607);
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));

    for (int seq_len : {2, 3, 4})
    {
        auto hidden = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            -0.5f, 0.5f, 20260606 + seq_len);
        auto fused_logits = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(num_experts)});
        auto row_logits = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(num_experts)});

        ASSERT_TRUE(hidden->ensureOnDevice(device));
        ASSERT_TRUE(fused_logits->ensureOnDevice(device));
        ASSERT_TRUE(row_logits->ensureOnDevice(device));

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        ASSERT_TRUE(hipMoE_gate_logits_small_m(
            static_cast<const float *>(hidden->gpu_data_ptr()),
            static_cast<const float *>(gate_weights->gpu_data_ptr()),
            static_cast<float *>(fused_logits->gpu_data_ptr()),
            seq_len, d_model, num_experts,
            /*device_idx=*/0,
            stream));
        for (int row = 0; row < seq_len; ++row)
        {
            ASSERT_TRUE(hipMoE_gate_logits_single_token(
                static_cast<const float *>(hidden->gpu_data_ptr()) + static_cast<size_t>(row) * d_model,
                static_cast<const float *>(gate_weights->gpu_data_ptr()),
                static_cast<float *>(row_logits->gpu_data_ptr()) + static_cast<size_t>(row) * num_experts,
                d_model, num_experts,
                /*device_idx=*/0,
                stream));
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

        fused_logits->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        row_logits->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *fused = fused_logits->data();
        const float *rowwise = row_logits->data();

        float max_abs_diff = 0.0f;
        for (int i = 0; i < seq_len * num_experts; ++i)
            max_abs_diff = std::max(max_abs_diff, std::fabs(fused[i] - rowwise[i]));
        EXPECT_LE(max_abs_diff, 1e-5f) << "seq_len=" << seq_len;
    }
}

TEST(Test__ROCmMoEKernel, SmallMBF16GateLogits_ModelShapeMatchesSingleTokenLaunches)
{
    SKIP_IF_NO_ROCM();

    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    const DeviceId device = DeviceId::rocm(0);

    std::vector<float> gate_weights_fp32(
        static_cast<size_t>(num_experts) * d_model);
    fillRandom(gate_weights_fp32, -0.5f, 0.5f, 20260608);
    auto gate_weights = TestTensorFactory::createBF16(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    gate_weights->from_fp32(gate_weights_fp32.data(), gate_weights_fp32.size());
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));

    for (int seq_len : {2, 3, 4})
    {
        auto hidden = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            -0.5f, 0.5f, 20260609 + seq_len);
        auto fused_logits = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(num_experts)});
        auto row_logits = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(num_experts)});

        ASSERT_TRUE(hidden->ensureOnDevice(device));
        ASSERT_TRUE(fused_logits->ensureOnDevice(device));
        ASSERT_TRUE(row_logits->ensureOnDevice(device));

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        ASSERT_TRUE(hipMoE_gate_logits_small_m_bf16_weights(
            static_cast<const float *>(hidden->gpu_data_ptr()),
            gate_weights->gpu_data_ptr(),
            static_cast<float *>(fused_logits->gpu_data_ptr()),
            seq_len, d_model, num_experts,
            /*device_idx=*/0,
            stream));
        for (int row = 0; row < seq_len; ++row)
        {
            ASSERT_TRUE(hipMoE_gate_logits_single_token_bf16_weights(
                static_cast<const float *>(hidden->gpu_data_ptr()) +
                    static_cast<size_t>(row) * d_model,
                gate_weights->gpu_data_ptr(),
                static_cast<float *>(row_logits->gpu_data_ptr()) +
                    static_cast<size_t>(row) * num_experts,
                d_model, num_experts,
                /*device_idx=*/0,
                stream));
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

        fused_logits->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        row_logits->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *fused = fused_logits->data();
        const float *rowwise = row_logits->data();

        float max_abs_diff = 0.0f;
        for (int i = 0; i < seq_len * num_experts; ++i)
            max_abs_diff = std::max(max_abs_diff, std::fabs(fused[i] - rowwise[i]));
        EXPECT_LE(max_abs_diff, 1e-3f) << "seq_len=" << seq_len;
    }
}

TEST(Test__ROCmMoEKernel, SmallMBF16RouteWithTensorsUsesDecodeEquivalentRouter)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 4;
    constexpr int d_model = 64;
    constexpr int num_experts = 16;
    constexpr int top_k = 4;
    const DeviceId device = DeviceId::rocm(0);

    auto hidden = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        -0.25f, 0.25f, 20260610);
    std::vector<float> gate_weights_fp32(
        static_cast<size_t>(num_experts) * d_model);
    fillRandom(gate_weights_fp32, -0.25f, 0.25f, 20260611);
    auto gate_weights = TestTensorFactory::createBF16(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    gate_weights->from_fp32(gate_weights_fp32.data(), gate_weights_fp32.size());
    auto routing_indices = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
    auto routing_weights = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/seq_len,
        d_model,
        /*intermediate=*/128,
        num_experts,
        top_k);
    MoERoutingResult result;
    ASSERT_TRUE(moe_kernel.routeWithTensors(
        hidden.get(),
        gate_weights.get(),
        seq_len,
        d_model,
        num_experts,
        top_k,
        /*normalize_weights=*/true,
        routing_indices.get(),
        routing_weights.get(),
        result));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    double decode_equivalent_router_calls = 0.0;
    for (const auto &record : PerfStatsCollector::snapshot({"kernel.rocm_moe_decode_equivalent_small_m_router_calls"}))
        decode_equivalent_router_calls += record.value;
    EXPECT_GT(decode_equivalent_router_calls, 0.0)
        << "BF16 verifier-sized routing should use the decode-equivalent row router path";
    PerfStatsCollector::reset();

    ASSERT_EQ(result.expert_indices.size(), static_cast<size_t>(seq_len * top_k));
    ASSERT_EQ(result.expert_weights.size(), static_cast<size_t>(seq_len * top_k));
    for (int expert : result.expert_indices)
    {
        EXPECT_GE(expert, 0);
        EXPECT_LT(expert, num_experts);
    }
    for (float weight : result.expert_weights)
        EXPECT_TRUE(std::isfinite(weight));
}

TEST(Test__ROCmMoEKernel, SoftmaxTopKParallelSelectionPreservesTieOrder)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 2;
    constexpr int num_experts = 16;
    constexpr int top_k = 4;
    const DeviceId device = DeviceId::rocm(0);

    auto logits = TestTensorFactory::createFP32({seq_len, num_experts});
    auto indices = TestTensorFactory::createINT32({seq_len, top_k});
    auto weights = TestTensorFactory::createFP32({seq_len, top_k});

    // Row 0 has two equal winner pairs; row 1 is all ties. The expected expert
    // order therefore proves that the parallel reduction keeps the original
    // ascending serial-scan tie semantics.
    std::fill(logits->mutable_data(), logits->mutable_data() + seq_len * num_experts, 0.0f);
    logits->mutable_data()[2] = 5.0f;
    logits->mutable_data()[4] = 5.0f;
    logits->mutable_data()[1] = 4.0f;
    logits->mutable_data()[7] = 4.0f;
    for (int expert = 0; expert < num_experts; ++expert)
        logits->mutable_data()[num_experts + expert] = 1.0f;

    ASSERT_TRUE(logits->ensureOnDevice(device));
    ASSERT_TRUE(indices->ensureOnDevice(device));
    ASSERT_TRUE(weights->ensureOnDevice(device));

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_TRUE(hipMoE_softmax_topk(
        static_cast<float *>(logits->gpu_data_ptr()),
        static_cast<int *>(indices->gpu_data_ptr()),
        static_cast<float *>(weights->gpu_data_ptr()),
        seq_len,
        num_experts,
        top_k,
        /*normalize_weights=*/true,
        /*device_idx=*/0,
        stream));
    indices->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    weights->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ASSERT_TRUE(indices->ensureOnHost(stream));
    ASSERT_TRUE(weights->ensureOnHost(stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    const int *actual_indices = indices->int32_data();
    const float *actual_weights = weights->data();

    const int expected_indices[seq_len * top_k] = {2, 4, 1, 7, 0, 1, 2, 3};
    for (int i = 0; i < seq_len * top_k; ++i)
        EXPECT_EQ(actual_indices[i], expected_indices[i]) << "slot=" << i;

    const float row0_pair_sum = 2.0f * std::exp(5.0f) + 2.0f * std::exp(4.0f);
    const float expected_row0[4] = {
        std::exp(5.0f) / row0_pair_sum,
        std::exp(5.0f) / row0_pair_sum,
        std::exp(4.0f) / row0_pair_sum,
        std::exp(4.0f) / row0_pair_sum};
    for (int k = 0; k < top_k; ++k)
        EXPECT_NEAR(actual_weights[k], expected_row0[k], 1e-6f) << "row0 k=" << k;
    for (int k = 0; k < top_k; ++k)
        EXPECT_NEAR(actual_weights[top_k + k], 0.25f, 1e-6f) << "row1 k=" << k;
}

TEST(Test__ROCmMoEKernel, SmallMRouteWithTensorsUsesDecodeEquivalentRouterAndMatchesHostReference)
{
    SKIP_IF_NO_ROCM();

    constexpr int seq_len = 4;
    constexpr int d_model = 32;
    constexpr int num_experts = 12;
    constexpr int top_k = 4;
    const DeviceId device = DeviceId::rocm(0);

    auto hidden = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        -0.25f, 0.25f, 20260604);
    auto gate_weights = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)},
        -0.25f, 0.25f, 20260605);
    auto routing_indices = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
    auto routing_weights = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});

    ASSERT_TRUE(hidden->ensureOnDevice(device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(device));
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    ROCmMoEKernel moe_kernel(0);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);
    MoERoutingResult result;
    ASSERT_TRUE(moe_kernel.routeWithTensors(
        hidden.get(),
        gate_weights.get(),
        seq_len,
        d_model,
        num_experts,
        top_k,
        /*normalize_weights=*/true,
        routing_indices.get(),
        routing_weights.get(),
        result));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    double decode_equivalent_router_calls = 0.0;
    for (const auto &record : PerfStatsCollector::snapshot({"kernel.rocm_moe_decode_equivalent_small_m_router_calls"}))
        decode_equivalent_router_calls += record.value;
    EXPECT_GT(decode_equivalent_router_calls, 0.0)
        << "verifier-sized routing should use the decode-equivalent row router path";
    PerfStatsCollector::reset();

    ASSERT_EQ(result.expert_indices.size(), static_cast<size_t>(seq_len * top_k));
    ASSERT_EQ(result.expert_weights.size(), static_cast<size_t>(seq_len * top_k));

    const float *h = hidden->data();
    const float *g = gate_weights->data();
    for (int token = 0; token < seq_len; ++token)
    {
        std::vector<float> probs(static_cast<size_t>(num_experts));
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int expert = 0; expert < num_experts; ++expert)
        {
            float dot = 0.0f;
            for (int col = 0; col < d_model; ++col)
                dot += h[token * d_model + col] * g[expert * d_model + col];
            probs[static_cast<size_t>(expert)] = dot;
            max_logit = std::max(max_logit, dot);
        }

        float sum = 0.0f;
        for (float &prob : probs)
        {
            prob = std::exp(prob - max_logit);
            sum += prob;
        }
        for (float &prob : probs)
            prob /= sum;

        std::vector<int> expected_indices;
        std::vector<float> expected_weights;
        expected_indices.reserve(top_k);
        expected_weights.reserve(top_k);
        for (int k = 0; k < top_k; ++k)
        {
            int best_idx = 0;
            float best_val = -1.0f;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const bool already_picked =
                    std::find(expected_indices.begin(), expected_indices.end(), expert) != expected_indices.end();
                if (!already_picked && probs[static_cast<size_t>(expert)] > best_val)
                {
                    best_val = probs[static_cast<size_t>(expert)];
                    best_idx = expert;
                }
            }
            expected_indices.push_back(best_idx);
            expected_weights.push_back(best_val);
        }

        float topk_sum = 0.0f;
        for (float weight : expected_weights)
            topk_sum += weight;
        for (float &weight : expected_weights)
            weight /= topk_sum;

        for (int k = 0; k < top_k; ++k)
        {
            const int offset = token * top_k + k;
            EXPECT_EQ(result.expert_indices[static_cast<size_t>(offset)],
                      expected_indices[static_cast<size_t>(k)])
                << "token=" << token << " k=" << k;
            EXPECT_NEAR(result.expert_weights[static_cast<size_t>(offset)],
                        expected_weights[static_cast<size_t>(k)],
                        1e-5f)
                << "token=" << token << " k=" << k;
        }
    }
}

TEST(Test__ROCmMoEKernel, SmallMRouteWithTensorsMatchesSerialDecodeRouterForVerifierRows)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_kernel_workspace = bindDefaultMoEWorkspace(moe_kernel);

    DeviceMoERuntimeTable::Config config;
    config.device_id = device;
    config.num_layers = 1;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.mirror_to_device = true;
    MoERuntimeTable runtime_table(config);

    MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 0);
    update.replica_role.resize(num_experts, 0);
    for (int expert = 0; expert < num_experts; ++expert)
        update.experts[expert].logical_expert_id = expert;
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, update.epoch, stream));

    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.035f * std::sin(0.011f * static_cast<float>(i + 13)) +
                         0.027f * std::cos(0.017f * static_cast<float>(i + 19)) +
                         0.0007f * static_cast<float>(static_cast<int>(i % 23) - 11);
    auto gate_weights = TestTensorFactory::createFP32(
        {static_cast<size_t>(num_experts), static_cast<size_t>(d_model)});
    std::copy(gate_values.begin(), gate_values.end(), gate_weights->mutable_data());
    ASSERT_TRUE(gate_weights->ensureOnDevice(device, stream));

    for (int seq_len : {2, 3, 4})
    {
        std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
        for (size_t i = 0; i < hidden_values.size(); ++i)
            hidden_values[i] = 0.05f * std::sin(0.007f * static_cast<float>(i + 1 + seq_len)) -
                               0.031f * std::cos(0.013f * static_cast<float>(i + 5)) +
                               0.0013f * static_cast<float>(static_cast<int>(i % 17) - 8);

        auto hidden = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        auto batched_indices = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        auto batched_weights = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        std::copy(hidden_values.begin(), hidden_values.end(), hidden->mutable_data());
        ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
        ASSERT_TRUE(batched_indices->allocateOnDevice(device, stream));
        ASSERT_TRUE(batched_weights->allocateOnDevice(device, stream));

        MoERoutingResult batched_result;
        ASSERT_TRUE(moe_kernel.routeWithTensors(
            hidden.get(),
            gate_weights.get(),
            seq_len,
            d_model,
            num_experts,
            top_k,
            /*normalize_weights=*/true,
            batched_indices.get(),
            batched_weights.get(),
            batched_result))
            << "seq_len=" << seq_len;
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> batched_indices_host(static_cast<size_t>(seq_len) * top_k);
        std::vector<float> batched_weights_host(static_cast<size_t>(seq_len) * top_k);
        ASSERT_EQ(hipMemcpyAsync(batched_indices_host.data(), batched_indices->gpu_data_ptr(),
                                 batched_indices_host.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(batched_weights_host.data(), batched_weights->gpu_data_ptr(),
                                 batched_weights_host.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> serial_indices(static_cast<size_t>(seq_len) * top_k);
        std::vector<float> serial_weights(static_cast<size_t>(seq_len) * top_k);
        for (int row = 0; row < seq_len; ++row)
        {
            auto row_hidden = TestTensorFactory::createFP32({1, static_cast<size_t>(d_model)});
            auto row_indices = TestTensorFactory::createFP32({1, static_cast<size_t>(top_k)});
            auto row_weights = TestTensorFactory::createFP32({1, static_cast<size_t>(top_k)});
            std::copy_n(hidden_values.data() + static_cast<size_t>(row) * d_model,
                        d_model,
                        row_hidden->mutable_data());
            ASSERT_TRUE(row_hidden->ensureOnDevice(device, stream));
            ASSERT_TRUE(row_indices->allocateOnDevice(device, stream));
            ASSERT_TRUE(row_weights->allocateOnDevice(device, stream));

            ASSERT_TRUE(moe_kernel.decodeRouteSelect(
                runtime_table.deviceLayerState(0),
                row_hidden.get(),
                gate_weights.get(),
                d_model,
                num_experts,
                top_k,
                /*normalize_weights=*/true,
                row_indices.get(),
                row_weights.get(),
                /*write_legacy_outputs=*/true,
                /*update_runtime_histogram=*/false))
                << "seq_len=" << seq_len << " row=" << row;

            ASSERT_EQ(hipMemcpyAsync(serial_indices.data() + static_cast<size_t>(row) * top_k,
                                     row_indices->gpu_data_ptr(),
                                     static_cast<size_t>(top_k) * sizeof(float),
                                     hipMemcpyDeviceToHost, stream),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(serial_weights.data() + static_cast<size_t>(row) * top_k,
                                     row_weights->gpu_data_ptr(),
                                     static_cast<size_t>(top_k) * sizeof(float),
                                     hipMemcpyDeviceToHost, stream),
                      hipSuccess);
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        expectStrictTopKRowsEquivalent(
            batched_indices_host,
            batched_weights_host,
            serial_indices,
            serial_weights,
            seq_len,
            top_k);
    }

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(Test__ROCmMoEKernel, GroupPrefillRoutesRuntimeState_DeterministicSmall)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int seq_len = 8;
    const int num_experts = 4;
    const int top_k = 2;
    const int total_slots = seq_len * top_k;

    std::vector<int> routing_indices = {
        1, 3,
        0, 2,
        0, 1,
        2, 3,
        0, 3,
        1, 2,
        0, 1,
        2, 3};
    std::vector<float> routing_indices_f32(static_cast<size_t>(total_slots));
    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    for (int i = 0; i < total_slots; ++i)
    {
        routing_indices_f32[static_cast<size_t>(i)] = static_cast<float>(routing_indices[static_cast<size_t>(i)]);
        routing_weights[static_cast<size_t>(i)] = 0.125f + 0.03125f * static_cast<float>(i);
    }

    auto routing_index_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weight_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_f32.begin(), routing_indices_f32.end(), routing_index_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(), routing_weight_tensor->mutable_data());
    ASSERT_TRUE(routing_index_tensor->ensureOnDevice(device));
    ASSERT_TRUE(routing_weight_tensor->ensureOnDevice(device));

    DeviceMoERuntimeTable::Config config;
    config.device_id = device;
    config.num_layers = 1;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.mirror_to_device = true;
    config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(config);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_index_tensor.get(),
        routing_weight_tensor.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const auto &state = runtime_table.hostLayerState(0);
    ASSERT_EQ(state.prefill_route_capacity, static_cast<uint32_t>(total_slots));

    std::vector<int> host_route_ids(total_slots);
    std::vector<float> host_route_weights(total_slots);
    std::vector<int> host_counts(num_experts);
    std::vector<int> host_offsets(num_experts);
    std::vector<int> host_grouped_ids(total_slots);
    std::vector<float> host_grouped_weights(total_slots);
    ASSERT_EQ(hipMemcpy(host_route_ids.data(), state.route_expert_ids,
                        host_route_ids.size() * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_route_weights.data(), state.route_weights,
                        host_route_weights.size() * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_counts.data(), state.expert_counts,
                        host_counts.size() * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_offsets.data(), state.expert_offsets,
                        host_offsets.size() * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_ids.data(), state.grouped_token_ids,
                        host_grouped_ids.size() * sizeof(int), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(host_grouped_weights.data(), state.grouped_route_weights,
                        host_grouped_weights.size() * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);

    for (int slot = 0; slot < total_slots; ++slot)
    {
        EXPECT_EQ(host_route_ids[static_cast<size_t>(slot)], routing_indices[static_cast<size_t>(slot)]);
        EXPECT_FLOAT_EQ(host_route_weights[static_cast<size_t>(slot)], routing_weights[static_cast<size_t>(slot)]);
    }

    const std::vector<int> expected_counts = {4, 4, 4, 4};
    int running = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        EXPECT_EQ(host_counts[static_cast<size_t>(expert)], expected_counts[static_cast<size_t>(expert)]);
        EXPECT_EQ(host_offsets[static_cast<size_t>(expert)], running);
        running += expected_counts[static_cast<size_t>(expert)];
    }
    EXPECT_EQ(running, total_slots);

    std::vector<std::vector<std::pair<int, float>>> expected_per_expert(static_cast<size_t>(num_experts));
    for (int slot = 0; slot < total_slots; ++slot)
    {
        const int expert = routing_indices[static_cast<size_t>(slot)];
        expected_per_expert[static_cast<size_t>(expert)].push_back(
            {slot / top_k, routing_weights[static_cast<size_t>(slot)]});
    }

    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int offset = host_offsets[static_cast<size_t>(expert)];
        const auto &expected = expected_per_expert[static_cast<size_t>(expert)];
        ASSERT_EQ(static_cast<int>(expected.size()), host_counts[static_cast<size_t>(expert)]);
        for (int i = 0; i < static_cast<int>(expected.size()); ++i)
        {
            EXPECT_EQ(host_grouped_ids[static_cast<size_t>(offset + i)], expected[static_cast<size_t>(i)].first)
                << "expert=" << expert << " grouped row=" << i;
            EXPECT_FLOAT_EQ(host_grouped_weights[static_cast<size_t>(offset + i)],
                            expected[static_cast<size_t>(i)].second)
                << "expert=" << expert << " grouped row=" << i;
        }
    }
}

TEST(Test__ROCmMoEKernel, RuntimePrefillGatherScatter_ZeroCountExpertNoOps)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    const int seq_len = 4;
    const int d_model = 16;
    const int num_experts = 4;
    const int top_k = 2;
    const int total_slots = seq_len * top_k;

    std::vector<int> routing_indices = {
        0, 2,
        2, 0,
        0, 2,
        2, 0};
    std::vector<float> routing_indices_f32(static_cast<size_t>(total_slots));
    std::vector<float> routing_weights(static_cast<size_t>(total_slots), 0.5f);
    for (int i = 0; i < total_slots; ++i)
        routing_indices_f32[static_cast<size_t>(i)] = static_cast<float>(routing_indices[static_cast<size_t>(i)]);

    auto routing_index_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weight_tensor = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_f32.begin(), routing_indices_f32.end(), routing_index_tensor->mutable_data());
    std::copy(routing_weights.begin(), routing_weights.end(), routing_weight_tensor->mutable_data());
    ASSERT_TRUE(routing_index_tensor->ensureOnDevice(device));
    ASSERT_TRUE(routing_weight_tensor->ensureOnDevice(device));

    DeviceMoERuntimeTable::Config config;
    config.device_id = device;
    config.num_layers = 1;
    config.num_experts = num_experts;
    config.top_k = top_k;
    config.mirror_to_device = true;
    config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(config);

    ROCmMoEKernel gpu_kernel(0);
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
    ASSERT_TRUE(gpu_kernel.groupPrefillRoutes(
        runtime_table.deviceLayerState(0),
        routing_index_tensor.get(),
        routing_weight_tensor.get(),
        seq_len,
        seq_len,
        num_experts,
        top_k));

    auto hidden = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    for (int i = 0; i < seq_len * d_model; ++i)
        hidden->mutable_data()[i] = 0.01f * static_cast<float>(i + 1);
    ASSERT_TRUE(hidden->ensureOnDevice(device));

    auto batch = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(batch->ensureOnDevice(device));
    ASSERT_TRUE(gpu_kernel.gatherPrefillExpertBatchFromRuntime(
        runtime_table.deviceLayerState(0), hidden.get(), batch.get(),
        /*expert_id=*/1, seq_len, d_model));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    batch->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *zero_batch = batch->data();
    for (int i = 0; i < seq_len * d_model; ++i)
        EXPECT_FLOAT_EQ(zero_batch[i], 0.0f) << "zero-count gather wrote row element " << i;

    auto output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    std::fill(output->mutable_data(), output->mutable_data() + seq_len * d_model, 0.25f);
    ASSERT_TRUE(output->ensureOnDevice(device));
    auto expert_output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    std::fill(expert_output->mutable_data(), expert_output->mutable_data() + seq_len * d_model, 7.0f);
    ASSERT_TRUE(expert_output->ensureOnDevice(device));

    ASSERT_TRUE(gpu_kernel.scatterPrefillExpertResultsFromRuntime(
        output.get(), expert_output.get(), runtime_table.deviceLayerState(0),
        /*expert_id=*/1, seq_len, d_model));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *no_op_output = output->data();
    for (int i = 0; i < seq_len * d_model; ++i)
        EXPECT_FLOAT_EQ(no_op_output[i], 0.25f) << "zero-count scatter changed output element " << i;
}

TEST(Test__ROCmMoEKernel, FixedTopologyRuntimeGroupedPrefillMatchesExistingPrefillPath)
{
    SKIP_IF_NO_ROCM();
#ifdef ENABLE_PIPELINE_SNAPSHOTS
    GTEST_SKIP() << "Fixed-topology prefill capture path is release-only when snapshots are disabled";
#endif

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 2;
    constexpr int d_model = 64;
    constexpr int intermediate = 32;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int total_slots = seq_len * top_k;

    struct ScopedGroupedPrefillFlag
    {
        bool old_value;
        explicit ScopedGroupedPrefillFlag(bool value)
            : old_value(mutableDebugEnv().rocm.moe_grouped_prefill)
        {
            mutableDebugEnv().rocm.moe_grouped_prefill = value;
        }
        ~ScopedGroupedPrefillFlag()
        {
            mutableDebugEnv().rocm.moe_grouped_prefill = old_value;
        }
    } grouped_prefill_flag(true);
    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    std::vector<std::unique_ptr<TensorBase>> gate_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> up_weight_tensors;
    std::vector<std::unique_ptr<TensorBase>> down_weight_tensors;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> gate_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> up_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmPackedWeights>> down_packed_weights;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> gate_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> up_kernels;
    std::vector<std::unique_ptr<rocm::ROCmQuantisedGemmKernel>> down_kernels;

    gate_weight_tensors.reserve(num_experts);
    up_weight_tensors.reserve(num_experts);
    down_weight_tensors.reserve(num_experts);
    gate_packed_weights.reserve(num_experts);
    up_packed_weights.reserve(num_experts);
    down_packed_weights.reserve(num_experts);
    gate_kernels.reserve(num_experts);
    up_kernels.reserve(num_experts);
    down_kernels.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        auto gate_weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 100 + expert_id);
        auto up_weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 200 + expert_id);
        auto down_weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 300 + expert_id);

        auto gate_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto up_packed = std::make_unique<rocm::ROCmPackedWeights>();
        auto down_packed = std::make_unique<rocm::ROCmPackedWeights>();
        ASSERT_TRUE(rocm::packWeightsToROCm(gate_weights.get(), *gate_packed));
        ASSERT_TRUE(rocm::packWeightsToROCm(up_weights.get(), *up_packed));
        ASSERT_TRUE(rocm::packWeightsToROCm(down_weights.get(), *down_packed));

        auto gate_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(gate_packed.get(), 0);
        auto up_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(up_packed.get(), 0);
        auto down_kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(down_packed.get(), 0);

        gate_weight_tensors.push_back(std::move(gate_weights));
        up_weight_tensors.push_back(std::move(up_weights));
        down_weight_tensors.push_back(std::move(down_weights));
        gate_packed_weights.push_back(std::move(gate_packed));
        up_packed_weights.push_back(std::move(up_packed));
        down_packed_weights.push_back(std::move(down_packed));
        gate_kernels.push_back(std::move(gate_kernel));
        up_kernels.push_back(std::move(up_kernel));
        down_kernels.push_back(std::move(down_kernel));
    }

    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace->allocate(gate_kernels[0]->getWorkspaceRequirements(seq_len, intermediate, d_model)));
    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        gate_kernels[expert_id]->bindWorkspace(workspace.get());
        up_kernels[expert_id]->bindWorkspace(workspace.get());
        down_kernels[expert_id]->bindWorkspace(workspace.get());
        gate_kernels[expert_id]->prepareWeights();
        up_kernels[expert_id]->prepareWeights();
        down_kernels[expert_id]->prepareWeights();
    }

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -1.0f, 1.0f, 77);
    ASSERT_TRUE(input->ensureOnDevice(device));

    const float routing_indices_data[total_slots] = {
        0.0f, 1.0f,
        2.0f, 3.0f};
    const float routing_weights_data[total_slots] = {
        0.70f, 0.30f,
        0.60f, 0.40f};

    auto routing_indices = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    auto routing_weights = TestTensorFactory::createFP32({static_cast<size_t>(total_slots), 1});
    std::copy(routing_indices_data, routing_indices_data + total_slots, routing_indices->mutable_data());
    std::copy(routing_weights_data, routing_weights_data + total_slots, routing_weights->mutable_data());
    ASSERT_TRUE(routing_indices->ensureOnDevice(device));
    ASSERT_TRUE(routing_weights->ensureOnDevice(device));

    auto reference_output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto fixed_output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(reference_output->ensureOnDevice(device));
    ASSERT_TRUE(fixed_output->ensureOnDevice(device));

    auto gate_exps = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(num_experts * intermediate), static_cast<size_t>(d_model)}, 401);
    auto up_exps = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(num_experts * intermediate), static_cast<size_t>(d_model)}, 402);
    auto down_exps = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(num_experts * d_model), static_cast<size_t>(intermediate)}, 403);

    auto make_params = [&](TensorBase *output, IMoERuntimeTable *runtime_table)
    {
        MoEExpertComputeStage::Params params;
        params.device_id = device;
        params.input = input.get();
        params.seq_len = seq_len;
        params.d_model = d_model;
        params.num_experts = num_experts;
        params.top_k = top_k;
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.expert_intermediate = intermediate;
        params.local_expert_start = 0;
        params.local_expert_count = num_experts;
        params.layer_idx = 0;
        params.routing_indices = routing_indices.get();
        params.routing_weights = routing_weights.get();
        params.output = output;
        params.moe_runtime_table = runtime_table;
        params.prepared_gate_gemm.assign(num_experts, nullptr);
        params.prepared_up_gemm.assign(num_experts, nullptr);
        params.prepared_down_gemm.assign(num_experts, nullptr);
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            params.prepared_gate_gemm[expert_id] = gate_kernels[expert_id].get();
            params.prepared_up_gemm[expert_id] = up_kernels[expert_id].get();
            params.prepared_down_gemm[expert_id] = down_kernels[expert_id].get();
        }
        return params;
    };

    ROCmDeviceContext ctx(device, 0);

    mutableDebugEnv().rocm.moe_grouped_prefill = false;
    MoEExpertComputeStage reference_stage(make_params(reference_output.get(), nullptr));
    reference_stage.bindWorkspace(workspace.get());
    ASSERT_TRUE(reference_stage.execute(&ctx));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = device;
    runtime_config.num_layers = 1;
    runtime_config.num_experts = num_experts;
    runtime_config.top_k = top_k;
    runtime_config.mirror_to_device = true;
    runtime_config.prefill_token_capacity = seq_len;
    MoERuntimeTable runtime_table(runtime_config);

    mutableDebugEnv().rocm.moe_grouped_prefill = true;
    MoEExpertComputeStage fixed_stage(make_params(fixed_output.get(), &runtime_table));
    ASSERT_TRUE(fixed_stage.isGraphCapturable());
    fixed_stage.bindWorkspace(workspace.get());
    ASSERT_TRUE(fixed_stage.execute(&ctx));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    {
        const auto records = PerfStatsCollector::snapshot({"kernel.rocm_moe_small_prefill_grouping_calls"});
        double small_grouping_calls = 0.0;
        for (const auto &record : records)
            small_grouping_calls += record.value;
        EXPECT_GT(small_grouping_calls, 0.0)
            << "verifier-sized fixed-topology prefill should use fused small-M grouping";
    }
    {
        const auto records = PerfStatsCollector::snapshot({"kernel.rocm_moe_grouped_prefill_active_expert_grid_calls"});
        double active_grid_calls = 0.0;
        bool saw_verifier_tile = false;
        for (const auto &record : records)
        {
            active_grid_calls += record.value;
            const auto tile_it = record.tags.find("tile_m");
            saw_verifier_tile = saw_verifier_tile ||
                                (tile_it != record.tags.end() && tile_it->second == "2");
        }
        EXPECT_GT(active_grid_calls, 0.0)
            << "verifier-sized fixed-topology prefill should launch over compact active experts";
        EXPECT_TRUE(saw_verifier_tile)
            << "ROCm two-row verifier grouped prefill should use the M=2 kernel bucket";
    }
    PerfStatsCollector::reset();

    reference_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    fixed_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *reference = reference_output->data();
    const float *fixed = fixed_output->data();
    const size_t output_count = static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
    expectStrictVerifierSimilarity(
        "fixed-topology runtime grouped prefill should match existing prefill path",
        fixed,
        reference,
        output_count,
        static_cast<size_t>(d_model),
        /*min_cosine=*/0.99999,
        /*max_relative_l2=*/0.001,
        /*min_row_cosine=*/0.99999,
        /*max_row_relative_l2=*/0.0015,
        /*max_row_kl=*/1.0e-6);
}

TEST(Test__ROCmMoEKernel, SharedExpertGroupedPrefillMatchesSequentialPath)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int seq_len = 4;
    constexpr int d_model = 64;
    constexpr int intermediate = 32;
    constexpr int num_experts = 1;
    constexpr int top_k = 1;

    ScopedEnvOverride perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto gate_weights = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 6101);
    auto up_weights = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)}, 6102);
    auto down_weights = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)}, 6103);

    auto gate_packed = std::make_unique<rocm::ROCmPackedWeights>();
    auto up_packed = std::make_unique<rocm::ROCmPackedWeights>();
    auto down_packed = std::make_unique<rocm::ROCmPackedWeights>();
    ASSERT_TRUE(rocm::packWeightsToROCm(gate_weights.get(), *gate_packed));
    ASSERT_TRUE(rocm::packWeightsToROCm(up_weights.get(), *up_packed));
    ASSERT_TRUE(rocm::packWeightsToROCm(down_weights.get(), *down_packed));

    rocm::ROCmQuantisedGemmKernel gate_kernel(gate_packed.get(), 0);
    rocm::ROCmQuantisedGemmKernel up_kernel(up_packed.get(), 0);
    rocm::ROCmQuantisedGemmKernel down_kernel(down_packed.get(), 0);

    WorkspaceRequirements gemm_reqs;
    gemm_reqs.merge(gate_kernel.getWorkspaceRequirements(seq_len, intermediate, d_model));
    gemm_reqs.merge(up_kernel.getWorkspaceRequirements(seq_len, intermediate, d_model));
    gemm_reqs.merge(down_kernel.getWorkspaceRequirements(seq_len, d_model, intermediate));
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));
    gate_kernel.bindWorkspace(gemm_workspace.get());
    up_kernel.bindWorkspace(gemm_workspace.get());
    down_kernel.bindWorkspace(gemm_workspace.get());
    gate_kernel.prepareWeights();
    up_kernel.prepareWeights();
    down_kernel.prepareWeights();

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -0.5f, 0.5f, 6104);
    ASSERT_TRUE(input->ensureOnDevice(device));

    auto reference_gate = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});
    auto reference_up = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});
    auto reference_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(reference_gate->ensureOnDevice(device));
    ASSERT_TRUE(reference_up->ensureOnDevice(device));
    ASSERT_TRUE(reference_output->ensureOnDevice(device));

    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
        {&gate_kernel, reference_gate.get(), intermediate, nullptr, "shared_gate"},
        {&up_kernel, reference_up.get(), intermediate, nullptr, "shared_up"}};
    ASSERT_TRUE(gate_kernel.multiply_fused_tensor(
        input.get(), projections, seq_len, d_model));
    reference_gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    reference_up->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ASSERT_TRUE(down_kernel.multiply_tensor_with_fused_swiglu(
        reference_gate.get(), reference_up.get(), reference_output.get(),
        seq_len, d_model, intermediate));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    DeviceNativeVNNIMatrixDesc gate_desc;
    DeviceNativeVNNIMatrixDesc up_desc;
    DeviceNativeVNNIMatrixDesc down_desc;
    ASSERT_TRUE(gate_kernel.exportNativeVNNIMatrixDesc(gate_desc));
    ASSERT_TRUE(up_kernel.exportNativeVNNIMatrixDesc(up_desc));
    ASSERT_TRUE(down_kernel.exportNativeVNNIMatrixDesc(down_desc));

    ROCmMoEKernel moe_kernel(0);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel, seq_len, d_model, intermediate, num_experts, top_k);
    ASSERT_TRUE(moe_kernel.prepareSharedExpertPrefillGroup(seq_len));
    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        &gate_desc, &up_desc, num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        &down_desc, num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto grouped_output = TestTensorFactory::createFP32(
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_TRUE(grouped_output->ensureOnDevice(device));
    ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
        input.get(), grouped_output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    {
        const auto records = PerfStatsCollector::snapshot(
            {"kernel.rocm_moe_shared_expert_prefill_group_calls"});
        double calls = 0.0;
        for (const auto &record : records)
            calls += record.value;
        EXPECT_GT(calls, 0.0)
            << "shared expert verifier prefill must use the ROCm grouped preparation kernel";
    }

    reference_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    const size_t output_count = static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
    const float *reference = reference_output->data();
    const float *grouped = grouped_output->data();
    ASSERT_FALSE(hasNaNOrInf(grouped, output_count));

    expectStrictVerifierSimilarity(
        "ROCm shared-expert grouped verifier prefill must match sequential GEMV",
        grouped,
        reference,
        output_count,
        static_cast<size_t>(d_model));
}

TEST(Test__ROCmMoEKernel, RoutedOnlyVerifierPrefill_Qwen36ShapeM234MatchesRowByRowDecode)
{
    SKIP_IF_NO_ROCM();

    const DeviceId device = DeviceId::rocm(0);
    constexpr int d_model = 2048;
    constexpr int intermediate = 512;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;
    constexpr int routed_variants = 16;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

    ROCmMoEKernel moe_kernel(0);
    static_cast<ITensorKernel &>(moe_kernel).setGPUStream(stream);
    auto moe_workspace = bindDefaultMoEWorkspace(
        moe_kernel,
        /*max_seq_len=*/4,
        d_model,
        intermediate,
        num_experts,
        top_k);

    std::vector<std::unique_ptr<TensorBase>> owned_weights;
    std::vector<llaminar2::test::GpuPreparedGemm> prepared_weights;
    owned_weights.reserve(static_cast<size_t>(routed_variants * 3));
    prepared_weights.reserve(static_cast<size_t>(routed_variants * 3));

    auto add_prepared = [&](std::unique_ptr<TensorBase> weight,
                            int seed,
                            const char *role) -> ITensorGemm *
    {
        auto *weight_ptr = weight.get();
        owned_weights.push_back(std::move(weight));
        prepared_weights.push_back(llaminar2::test::makeGpuPreparedGemm(
            weight_ptr,
            device,
            std::string("test.rocm_moe.qwen36_routed_verifier.") + role +
                "." + std::to_string(seed),
            ModelContextId{890000 + static_cast<uint64_t>(seed)}));

        auto *kernel = prepared_weights.back().kernel;
        auto *tensor_kernel = dynamic_cast<ITensorKernel *>(kernel);
        if (!tensor_kernel)
            throw std::runtime_error("prepared ROCm routed verifier GEMM did not expose ITensorKernel");
        tensor_kernel->setGPUStream(stream);
        return kernel;
    };

    struct GemmTriplet
    {
        ITensorGemm *gate = nullptr;
        ITensorGemm *up = nullptr;
        ITensorGemm *down = nullptr;
    };

    std::array<GemmTriplet, routed_variants> routed{};
    for (int variant = 0; variant < routed_variants; ++variant)
    {
        routed[static_cast<size_t>(variant)].gate = add_prepared(
            TestTensorFactory::createIQ2_SRandom(
                {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)},
                891000 + variant),
            891000 + variant,
            "routed_gate");
        routed[static_cast<size_t>(variant)].up = add_prepared(
            TestTensorFactory::createIQ2_SRandom(
                {static_cast<size_t>(intermediate), static_cast<size_t>(d_model)},
                892000 + variant),
            892000 + variant,
            "routed_up");
        routed[static_cast<size_t>(variant)].down = add_prepared(
            TestTensorFactory::createIQ4_XSRandom(
                {static_cast<size_t>(d_model), static_cast<size_t>(intermediate)},
                893000 + variant),
            893000 + variant,
            "routed_down");
    }

    auto *workspace_probe = dynamic_cast<IWorkspaceConsumer *>(routed[0].gate);
    ASSERT_NE(workspace_probe, nullptr);
    WorkspaceRequirements gemm_reqs;
    gemm_reqs.merge(workspace_probe->getWorkspaceRequirements(4, intermediate, d_model));
    gemm_reqs.merge(workspace_probe->getWorkspaceRequirements(1, intermediate, d_model));
    if (auto *up_workspace = dynamic_cast<IWorkspaceConsumer *>(routed[0].up))
    {
        gemm_reqs.merge(up_workspace->getWorkspaceRequirements(4, intermediate, d_model));
        gemm_reqs.merge(up_workspace->getWorkspaceRequirements(1, intermediate, d_model));
    }
    if (auto *down_workspace = dynamic_cast<IWorkspaceConsumer *>(routed[0].down))
    {
        gemm_reqs.merge(down_workspace->getWorkspaceRequirements(4, d_model, intermediate));
        gemm_reqs.merge(down_workspace->getWorkspaceRequirements(1, d_model, intermediate));
    }
    auto gemm_workspace = std::make_unique<DeviceWorkspaceManager>(device, 256 * 1024 * 1024);
    ASSERT_TRUE(gemm_workspace->allocate(gemm_reqs));

    auto bind_gemm = [&](ITensorGemm *kernel)
    {
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        ASSERT_NE(consumer, nullptr);
        consumer->bindWorkspace(gemm_workspace.get());
        kernel->prepareWeights();
        ASSERT_TRUE(kernel->weights_converted());
    };
    for (const auto &triplet : routed)
    {
        bind_gemm(triplet.gate);
        bind_gemm(triplet.up);
        bind_gemm(triplet.down);
    }

    auto variant_for_expert = [&](int expert_id) -> size_t
    {
        return static_cast<size_t>(expert_id % routed_variants);
    };

    std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(static_cast<size_t>(num_experts));
    std::vector<DeviceNativeVNNIMatrixDesc> up_descs(static_cast<size_t>(num_experts));
    std::vector<DeviceNativeVNNIMatrixDesc> down_descs(static_cast<size_t>(num_experts));
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const GemmTriplet &triplet = routed[variant_for_expert(expert)];
        ASSERT_TRUE(triplet.gate->exportNativeVNNIMatrixDesc(gate_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.up->exportNativeVNNIMatrixDesc(up_descs[static_cast<size_t>(expert)]));
        ASSERT_TRUE(triplet.down->exportNativeVNNIMatrixDesc(down_descs[static_cast<size_t>(expert)]));
    }

    const int gateup_table = moe_kernel.uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);
    const int down_table = moe_kernel.uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);

    auto make_hidden = [](int seq_len)
    {
        auto hidden = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        for (size_t i = 0; i < hidden->numel(); ++i)
        {
            hidden->mutable_data()[i] =
                0.013f * static_cast<float>(static_cast<int>(i % 43) - 21) +
                0.004f * static_cast<float>(static_cast<int>((i / 17) % 19) - 9);
        }
        return hidden;
    };

    auto make_routes = [](int seq_len,
                          std::vector<float> &indices,
                          std::vector<float> &weights)
    {
        static constexpr std::array<int, top_k * 4> kExperts = {
            0, 13, 41, 96, 131, 159, 220, 238,
            3, 17, 42, 99, 144, 171, 221, 251,
            0, 17, 43, 96, 145, 159, 223, 251,
            5, 13, 42, 101, 131, 173, 220, 239};
        indices.resize(static_cast<size_t>(seq_len * top_k));
        weights.resize(static_cast<size_t>(seq_len * top_k));
        for (int row = 0; row < seq_len; ++row)
        {
            float sum = 0.0f;
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = row * top_k + route;
                indices[static_cast<size_t>(slot)] =
                    static_cast<float>(kExperts[static_cast<size_t>(slot)]);
                weights[static_cast<size_t>(slot)] =
                    0.09f + 0.013f * static_cast<float>((slot * 5 + 3) % 11);
                sum += weights[static_cast<size_t>(slot)];
            }
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = row * top_k + route;
                weights[static_cast<size_t>(slot)] /= sum;
            }
        }
    };

    for (int seq_len : {2, 3, 4})
    {
        auto hidden = make_hidden(seq_len);
        ASSERT_TRUE(hidden->ensureOnDevice(device, stream));

        std::vector<float> route_indices;
        std::vector<float> route_weights;
        make_routes(seq_len, route_indices, route_weights);
        auto routing_indices = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        auto routing_weights = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
        std::copy(route_indices.begin(), route_indices.end(), routing_indices->mutable_data());
        std::copy(route_weights.begin(), route_weights.end(), routing_weights->mutable_data());
        ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
        ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

        std::vector<float> row_by_row_expected(
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
        for (int row = 0; row < seq_len; ++row)
        {
            auto hidden_row = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            std::copy(hidden->data() + static_cast<size_t>(row) * d_model,
                      hidden->data() + static_cast<size_t>(row + 1) * d_model,
                      hidden_row->mutable_data());
            ASSERT_TRUE(hidden_row->ensureOnDevice(device, stream));

            std::array<int, top_k> expert_ids = {};
            std::array<float, top_k> expert_weights = {};
            for (int route = 0; route < top_k; ++route)
            {
                const int slot = row * top_k + route;
                expert_ids[static_cast<size_t>(route)] =
                    static_cast<int>(route_indices[static_cast<size_t>(slot)]);
                expert_weights[static_cast<size_t>(route)] =
                    route_weights[static_cast<size_t>(slot)];
            }

            std::array<std::shared_ptr<FP32Tensor>, top_k> gate_owned;
            std::array<std::shared_ptr<FP32Tensor>, top_k> up_owned;
            std::array<ITensor *, top_k> gate_outputs = {};
            std::array<ITensor *, top_k> up_outputs = {};
            for (int route = 0; route < top_k; ++route)
            {
                gate_owned[static_cast<size_t>(route)] =
                    TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
                up_owned[static_cast<size_t>(route)] =
                    TestTensorFactory::createFP32({1u, static_cast<size_t>(intermediate)});
                ASSERT_TRUE(gate_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream));
                ASSERT_TRUE(up_owned[static_cast<size_t>(route)]->ensureOnDevice(device, stream));
                gate_outputs[static_cast<size_t>(route)] =
                    gate_owned[static_cast<size_t>(route)].get();
                up_outputs[static_cast<size_t>(route)] =
                    up_owned[static_cast<size_t>(route)].get();
            }

            auto decode_output = TestTensorFactory::createFP32({1u, static_cast<size_t>(d_model)});
            ASSERT_TRUE(decode_output->ensureOnDevice(device, stream));
            ASSERT_TRUE(moe_kernel.groupedExpertGateUpDecodeFromTable(
                hidden_row.get(), expert_ids.data(), gateup_table, top_k,
                gate_outputs.data(), up_outputs.data(), d_model, intermediate));
            ASSERT_TRUE(moe_kernel.groupedExpertDownDecodeFromTable(
                gate_outputs.data(), up_outputs.data(), expert_ids.data(),
                expert_weights.data(), down_table, top_k, decode_output.get(),
                d_model, intermediate));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            decode_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
            std::copy(decode_output->data(),
                      decode_output->data() + d_model,
                      row_by_row_expected.begin() + static_cast<size_t>(row) * d_model);
        }

        auto grouped_output = TestTensorFactory::createFP32(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));
        ASSERT_TRUE(moe_kernel.prepareExpertGroupsAsync(
            routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));
        ASSERT_TRUE(moe_kernel.executeGroupedPrefillPipeline(
            hidden.get(),
            grouped_output.get(),
            gateup_table,
            down_table,
            seq_len,
            d_model,
            intermediate,
            num_experts,
            top_k));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        grouped_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);

        expectStrictVerifierSimilarity(
            ("ROCm Qwen3.6 routed-only verifier M=" + std::to_string(seq_len) +
             " must match row-by-row decode").c_str(),
            grouped_output->data(),
            row_by_row_expected.data(),
            grouped_output->numel(),
            static_cast<size_t>(d_model),
            /*min_cosine=*/0.99995,
            /*max_relative_l2=*/0.005,
            /*min_row_cosine=*/0.99995,
            /*max_row_relative_l2=*/0.005,
            /*max_row_kl=*/1.0e-4);
    }

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
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
    auto gpu_kernel_workspace = bindDefaultMoEWorkspace(gpu_kernel);
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
        if (!weights_match)
            break;
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
