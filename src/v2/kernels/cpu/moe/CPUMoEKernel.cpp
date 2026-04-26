/**
 * @file CPUMoEKernel.cpp
 * @brief CPU implementation of MoE kernel operations
 *
 * Extracted from MoEFFNStage.cpp to enable device-agnostic stage wiring.
 * Uses ISA-dispatched vector primitives for all compute-bound operations.
 */

#include "CPUMoEKernel.h"
#include "../../cpu/primitives/SoftmaxPrimitives_New.h"
#include "../../cpu/primitives/SwiGLUPrimitives.h"
#include "../../cpu/primitives/VectorPrimitives.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace llaminar2
{

    bool CPUMoEKernel::route(
        const float *hidden,
        const float *gate_weights,
        int seq_len, int d_model,
        int num_experts, int top_k,
        bool normalize_weights,
        MoERoutingResult &result)
    {
        result.expert_indices.resize(static_cast<size_t>(seq_len) * top_k);
        result.expert_weights.resize(static_cast<size_t>(seq_len) * top_k);
        result.router_logits.resize(static_cast<size_t>(seq_len) * num_experts);

        auto do_routing = [&]()
        {
            // Thread-local scratch (allocated per-thread inside parallel region)
            std::vector<float> logits(num_experts);
            std::vector<int> indices(num_experts);

#pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                const float *h = hidden + t * d_model;

                // Compute router logits via ISA-dispatched dot products
                for (int e = 0; e < num_experts; ++e)
                    logits[e] = primitives::vec_dot(gate_weights + e * d_model, h, d_model);

                // Vectorized softmax with fast_exp (SIMD-dispatched)
                primitives::softmax_row_fp32(logits.data(), num_experts);

                // Stash post-softmax probabilities
                std::copy(logits.begin(), logits.end(),
                          result.router_logits.begin() + static_cast<size_t>(t) * num_experts);

                // Top-k selection (reuse pre-allocated indices)
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                                  [&logits](int a, int b)
                                  { return logits[a] > logits[b]; });

                // Normalize top-k weights
                float topk_sum = 0.0f;
                for (int k = 0; k < top_k; ++k)
                    topk_sum += logits[indices[k]];

                for (int k = 0; k < top_k; ++k)
                {
                    result.expert_indices[t * top_k + k] = indices[k];
                    result.expert_weights[t * top_k + k] = normalize_weights
                                                                ? logits[indices[k]] / topk_sum
                                                                : logits[indices[k]];
                }
            }
        };

        // For single token (decode), parallelize the 256 dot products across
        // threads.  The gate-weight matrix is 256×d_model ≈ 2 MB — streaming it
        // through a single core is L3-bandwidth-bound (~15 GB/s → ~130 µs).
        // With 28 threads the aggregate L3 BW exceeds 400 GB/s, bringing the
        // dot-product phase down to ~5–10 µs.  The serial softmax + top-k that
        // follows costs only ~5 µs and is not worth parallelizing.
        //
        // For multi-token (prefill), parallelize across tokens as before.
        if (seq_len <= 1)
        {
            float *logits_ptr = result.router_logits.data();
            const float *h = hidden;

            // Phase 1: parallel dot products over experts
            auto do_dot_products = [&]()
            {
#pragma omp for schedule(static)
                for (int e = 0; e < num_experts; ++e)
                    logits_ptr[e] = primitives::vec_dot(
                        gate_weights + e * d_model, h, d_model);
            };
            OMP_WORKSHARE_REGION(do_dot_products);

            // Phase 2: vectorized softmax (uses fast_exp via SIMD primitives)
            primitives::softmax_row_fp32(logits_ptr, num_experts);

            // Phase 3: top-k selection (thread_local avoids per-call alloc)
            thread_local std::vector<int> indices;
            indices.resize(num_experts);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                              [logits_ptr](int a, int b)
                              { return logits_ptr[a] > logits_ptr[b]; });

            float topk_sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
                topk_sum += logits_ptr[indices[k]];

            for (int k = 0; k < top_k; ++k)
            {
                result.expert_indices[k] = indices[k];
                result.expert_weights[k] = normalize_weights
                                                ? logits_ptr[indices[k]] / topk_sum
                                                : logits_ptr[indices[k]];
            }
        }
        else
        {
            OMP_WORKSHARE_REGION(do_routing);
        }

        return true;
    }

    void CPUMoEKernel::gatherTokenBatch(
        const float *hidden,
        float *batch_buffer,
        const int *token_indices,
        int num_tokens, int d_model)
    {
        for (int i = 0; i < num_tokens; ++i)
        {
            const float *src = hidden + token_indices[i] * d_model;
            std::copy(src, src + d_model, batch_buffer + i * d_model);
        }
    }

    void CPUMoEKernel::scatterAddWeighted(
        float *output,
        const float *expert_output,
        const int *token_indices,
        const float *weights,
        int num_tokens, int d_model)
    {
        for (int i = 0; i < num_tokens; ++i)
        {
            primitives::vec_axpy(
                output + token_indices[i] * d_model,
                expert_output + i * d_model,
                weights[i], d_model);
        }
    }

    void CPUMoEKernel::sharedExpertGate(
        const float *input,
        const float *gate_inp,
        float *shared_output,
        int seq_len, int d_model)
    {
        // Fast serial path for decode (seq_len=1): OMP fork/join overhead
        // dominates for a single dot product + sigmoid + scale.
        if (seq_len <= 2)
        {
            for (int t = 0; t < seq_len; ++t)
            {
                const float *x = input + t * d_model;
                float dot = primitives::vec_dot(gate_inp, x, d_model);
                float gate = 1.0f / (1.0f + std::exp(-dot));
                float *out = shared_output + t * d_model;
                primitives::vec_scale(out, gate, d_model);
            }
            return;
        }

        auto do_work = [=]()
        {
#pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                const float *x = input + t * d_model;
                float dot = primitives::vec_dot(gate_inp, x, d_model);
                float gate = 1.0f / (1.0f + std::exp(-dot));

                float *out = shared_output + t * d_model;
                primitives::vec_scale(out, gate, d_model);
            }
        };
        OMP_WORKSHARE_REGION(do_work);
    }

    void CPUMoEKernel::swiGLU(float *gate, const float *up, int count)
    {
        primitives::compute_swiglu(gate, up, gate, count);
    }

} // namespace llaminar2
