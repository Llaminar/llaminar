/**
 * @file IMoEKernel.h
 * @brief Device-agnostic Mixture-of-Experts kernel interface
 *
 * Defines the kernel contract for MoE-specific operations that are not
 * covered by ITensorGemm (which already handles device-agnostic GEMM).
 *
 * Operations:
 * - Router: gate logits → softmax → top-k selection
 * - Token gather: collect tokens for an expert batch
 * - Scatter-add: weighted accumulation of expert outputs
 * - Shared expert gate: sigmoid dot + elementwise scale
 * - SwiGLU fallback: activation when fused GEMM+SwiGLU is unavailable
 *
 * CPU implementation: CPUMoEKernel (src/v2/kernels/cpu/moe/)
 * GPU implementations can override for device-native execution.
 */

#pragma once

#include "../tensors/TensorKernels.h"

#include <vector>

namespace llaminar2
{

    /**
     * @brief Routing result from MoE gate computation
     *
     * Contains per-token expert assignments and weights after
     * softmax + top-k selection.
     */
    struct MoERoutingResult
    {
        std::vector<int> expert_indices;   ///< [seq_len * top_k] selected expert IDs
        std::vector<float> expert_weights; ///< [seq_len * top_k] normalized weights
        std::vector<float> router_logits;  ///< [seq_len * num_experts] post-softmax probs
    };

    /**
     * @brief Device-agnostic MoE kernel interface
     *
     * Encapsulates all non-GEMM MoE operations. GEMM (gate/up/down projections)
     * is already device-agnostic via ITensorGemm engines. This kernel handles
     * the MoE-specific orchestration primitives:
     *
     * - Routing (softmax top-k)
     * - Token gather/scatter for expert batching
     * - Shared expert sigmoid gating
     * - SwiGLU activation fallback
     */
    class IMoEKernel : public ITensorKernel
    {
    public:
        ~IMoEKernel() override = default;

        // =================================================================
        // Router: gate logits → softmax → top-k per token
        // =================================================================

        /**
         * @brief Compute MoE routing: gate logits, softmax, top-k selection
         *
         * For each token t in [0, seq_len):
         *   1. logits[e] = dot(hidden[t], gate_weights[e]) for all experts
         *   2. probs = softmax(logits)
         *   3. Select top-k experts by probability
         *   4. Optionally normalize top-k weights to sum to 1
         *
         * @param hidden           Input hidden states [seq_len, d_model]
         * @param gate_weights     Router gate matrix [num_experts, d_model]
         * @param seq_len          Number of tokens
         * @param d_model          Hidden dimension
         * @param num_experts      Total number of experts
         * @param top_k            Experts selected per token
         * @param normalize_weights Renormalize top-k weights to sum to 1
         * @param[out] result      Routing assignments and weights
         * @return true on success
         */
        virtual bool route(
            const float *hidden,
            const float *gate_weights,
            int seq_len, int d_model,
            int num_experts, int top_k,
            bool normalize_weights,
            MoERoutingResult &result) = 0;

        // =================================================================
        // Token gather/scatter for expert batching
        // =================================================================

        /**
         * @brief Gather tokens into a contiguous batch buffer for one expert
         *
         * Copies rows from hidden[token_indices[i]] into batch_buffer[i]
         * for i in [0, num_tokens).
         *
         * @param hidden        Full hidden states [seq_len, d_model]
         * @param batch_buffer  Output batch [num_tokens, d_model]
         * @param token_indices Token row indices to gather [num_tokens]
         * @param num_tokens    Number of tokens in this expert batch
         * @param d_model       Hidden dimension
         */
        virtual void gatherTokenBatch(
            const float *hidden,
            float *batch_buffer,
            const int *token_indices,
            int num_tokens, int d_model) = 0;

        /**
         * @brief Scatter weighted expert outputs back to combined output
         *
         * For each token i in [0, num_tokens):
         *   output[token_indices[i]] += weights[i] * expert_output[i]
         *
         * @param output         Accumulated output [seq_len, d_model] (must be pre-zeroed)
         * @param expert_output  Expert's output [num_tokens, d_model]
         * @param token_indices  Token row indices [num_tokens]
         * @param weights        Per-token routing weights [num_tokens]
         * @param num_tokens     Number of tokens in this expert batch
         * @param d_model        Hidden dimension
         */
        virtual void scatterAddWeighted(
            float *output,
            const float *expert_output,
            const int *token_indices,
            const float *weights,
            int num_tokens, int d_model) = 0;

        // =================================================================
        // Shared expert sigmoid gating
        // =================================================================

        /**
         * @brief Apply sigmoid gating to shared expert output (in-place)
         *
         * For each token t in [0, seq_len):
         *   gate = sigmoid(dot(gate_inp, input[t]))
         *   shared_output[t] *= gate
         *
         * @param input          Hidden states [seq_len, d_model]
         * @param gate_inp       Gate vector [d_model]
         * @param shared_output  Shared expert output, modified in-place [seq_len, d_model]
         * @param seq_len        Number of tokens
         * @param d_model        Hidden dimension
         */
        virtual void sharedExpertGate(
            const float *input,
            const float *gate_inp,
            float *shared_output,
            int seq_len, int d_model) = 0;

        // =================================================================
        // SwiGLU activation fallback
        // =================================================================

        /**
         * @brief SwiGLU activation: gate = silu(gate) * up
         *
         * In-place into gate buffer. Used as fallback when the GEMM engine
         * does not support fused SwiGLU+Down projection.
         *
         * @param gate  Gate projection output, modified in-place [count]
         * @param up    Up projection output [count]
         * @param count Total number of elements (batch_size * intermediate_dim)
         */
        virtual void swiGLU(float *gate, const float *up, int count) = 0;
    };

} // namespace llaminar2
