/**
 * @file CPUMoEKernel.cpp
 * @brief CPU implementation of MoE kernel operations
 *
 * Extracted from MoEFFNStage.cpp to enable device-agnostic stage wiring.
 * Uses ISA-dispatched vector primitives for all compute-bound operations.
 */

#include "CPUMoEKernel.h"
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

                // Softmax (num_experts ~256: scalar is adequate for this size)
                float max_logit = *std::max_element(logits.begin(), logits.end());
                float sum_exp = 0.0f;
                for (int e = 0; e < num_experts; ++e)
                {
                    logits[e] = std::exp(logits[e] - max_logit);
                    sum_exp += logits[e];
                }
                const float inv_sum = 1.0f / sum_exp;
                for (int e = 0; e < num_experts; ++e)
                    logits[e] *= inv_sum;

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

        // For single token (decode), run without OpenMP overhead.
        // For multi-token (prefill), parallelize across tokens.
        if (seq_len <= 1)
        {
            do_routing();
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
