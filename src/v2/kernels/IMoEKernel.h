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

#include <cstdint>
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

        // =================================================================
        // Weighted vector addition (for expert output accumulation)
        // =================================================================

        /**
         * @brief Weighted add: output += weight * input
         *
         * Used to accumulate expert outputs into the combined MoE output.
         * CPU implementation is a simple loop. GPU implementations use
         * device-native kernels.
         *
         * @param output  Accumulated output buffer [count] (read+write)
         * @param input   Expert output buffer [count] (read-only)
         * @param weight  Scalar routing weight
         * @param count   Number of elements
         */
        virtual void weightedAdd(float *output, const float *input,
                                 float weight, int count)
        {
            // Default CPU implementation
            for (int i = 0; i < count; ++i)
                output[i] += weight * input[i];
        }

        // =================================================================
        // Tensor-aware API (device-agnostic)
        //
        // These methods accept ITensor* and handle coherence internally.
        // CPU defaults (in IMoEKernel.cpp) use data()/mutable_data().
        // GPU implementations override to use gpu_data_ptr() +
        // transitionTo(), keeping data on-device without H2D round-trips.
        //
        // Compute stages should use ONLY these methods — never raw pointers
        // or CUDA/HIP APIs directly.
        // =================================================================

        /// Tensor-aware route: computes routing, writes results to output
        /// tensors on the active device, and returns a host copy in
        /// host_result for CPU-side expert dispatch.  GPU implementations
        /// use D2D for tensors (no intermediate H2D).
        virtual bool routeWithTensors(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            MoERoutingResult &host_result);

        /// Zero a tensor's data buffer on the active device.
        /// GPU: zeros device memory, marks DEVICE_AUTHORITATIVE.
        /// CPU: zeros via mutable_data().
        virtual void zeroBuffer(ITensor *tensor, size_t bytes);

        /// Tensor-aware gather.  host_token_indices lives on the host;
        /// GPU implementations upload it to device staging internally.
        virtual void gatherTokenBatchFromTensors(
            ITensor *hidden, ITensor *batch_buffer,
            const int *host_token_indices, int num_tokens, int d_model);

        /// Tensor-aware scatter-add.  host indices/weights live on the host;
        /// GPU implementations upload them internally.
        virtual void scatterAddWeightedFromTensors(
            ITensor *output, ITensor *expert_output,
            const int *host_token_indices, const float *host_weights,
            int num_tokens, int d_model);

        /// Tensor-aware shared expert gate (sigmoid gating in-place).
        virtual void sharedExpertGateFromTensors(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            int seq_len, int d_model);

        /// Tensor-aware SwiGLU: gate = silu(gate) * up, on active device.
        virtual void swiGLUFromTensors(ITensor *gate, ITensor *up, int count);

        /// Tensor-aware weighted add: output += weight * input.
        virtual void weightedAddFromTensors(
            ITensor *output, ITensor *input, float weight, int count);

        // =================================================================
        // Device-resident histogram + expert mask (Phase 2)
        //
        // Default no-op implementations so CPU kernels compile unchanged.
        // GPU kernels override for on-device execution.
        // =================================================================

        /**
         * @brief Record routing decisions into device-resident histogram
         *
         * Atomically increments per-expert counters based on routing indices.
         *
         * @param d_routing_indices Device pointer: [seq_len * top_k] expert indices
         * @param seq_len           Number of tokens
         * @param top_k             Number of experts per token
         * @param layer_idx         Layer index (for per-layer histograms)
         */
        virtual void recordHistogramDevice(
            const int *d_routing_indices, int seq_len, int top_k, int layer_idx) {}

        /**
         * @brief Sync device histogram to host (async D2H copy + stream sync)
         *
         * @param host_counts Host buffer to receive counts: [num_experts] uint64_t
         * @param layer_idx   Which layer's histogram to sync
         * @param num_experts  Number of experts
         */
        virtual void syncHistogramToHost(
            uint64_t *host_counts, int layer_idx, int num_experts) {}

        /**
         * @brief Reset device histogram counters for a specific layer to zero
         *
         * @param layer_idx   Layer index
         * @param num_experts  Number of experts
         */
        virtual void resetHistogramDevice(int layer_idx, int num_experts) {}

        /**
         * @brief Upload expert mask to device (H2D)
         *
         * mask[e] = true means expert e is active (local to this device).
         *
         * @param mask        Host pointer: [num_experts] booleans
         * @param num_experts Number of experts
         */
        virtual void updateExpertMaskDevice(const bool *mask, int num_experts) {}

        /**
         * @brief Apply expert mask: zero out routing weights for masked-off experts
         *
         * For each token/expert slot, if the expert is masked off,
         * set its routing weight to 0.
         *
         * @param d_routing_weights  Device pointer: [seq_len * top_k] weights (modified in-place)
         * @param d_routing_indices  Device pointer: [seq_len * top_k] expert indices
         * @param seq_len            Number of tokens
         * @param top_k              Number of experts per token
         */
        virtual void applyExpertMaskDevice(
            float *d_routing_weights, const int *d_routing_indices,
            int seq_len, int top_k) {}

        // =================================================================
        // Device-side token grouping (Phase 3 — prefill optimization)
        //
        // Default no-op returning false so CPU kernels compile unchanged.
        // GPU kernels override for on-device execution.
        // =================================================================

        /**
         * @brief Group tokens by expert on-device (prefill optimization)
         *
         * After this call, tokens for expert e are at:
         *   grouped_token_indices[expert_offsets[e] .. expert_offsets[e] + expert_counts[e])
         *   grouped_weights[expert_offsets[e] .. expert_offsets[e] + expert_counts[e])
         * where grouped_token_indices[i] is the original token index (0..seq_len-1).
         *
         * All pointers are device pointers. expert_offsets and expert_counts
         * are output device buffers.
         *
         * @param d_routing_indices       Device pointer: [seq_len * top_k] expert indices
         * @param d_routing_weights       Device pointer: [seq_len * top_k] routing weights
         * @param seq_len                 Number of tokens
         * @param num_experts             Total number of experts
         * @param top_k                   Experts selected per token
         * @param d_expert_offsets        Output device pointer: [num_experts] exclusive prefix sums
         * @param d_expert_counts         Output device pointer: [num_experts] per-expert token counts
         * @param d_grouped_token_indices Output device pointer: [seq_len * top_k] grouped token indices
         * @param d_grouped_weights       Output device pointer: [seq_len * top_k] grouped weights
         * @return true on success
         */
        virtual bool groupTokensByExpertDevice(
            const int *d_routing_indices,
            const float *d_routing_weights,
            int seq_len, int num_experts, int top_k,
            int *d_expert_offsets,
            int *d_expert_counts,
            int *d_grouped_token_indices,
            float *d_grouped_weights) { return false; }

        // =================================================================
        // Phase 4: GPU-side expert dispatch for prefill
        //
        // prepareExpertGroups() does GPU-side token grouping from tensor
        // routing results (float→int conversion + grouping kernel).
        // After this, gatherExpertBatch/scatterExpertResults use device-
        // resident grouped indices — no per-call H2D staging needed.
        //
        // CPU default: falls back to host-side grouping.
        // =================================================================

        /**
         * @brief Prepare device-side token groups from routing tensor results.
         *
         * Converts float routing indices to int, runs GPU token grouping,
         * and D2H's the per-expert counts (small: num_experts ints).
         * After this call, gatherExpertBatch/scatterExpertResults use
         * pre-computed device-side offsets.
         *
         * @return true if GPU grouping succeeded; false → caller should
         *         fall back to host-side grouping.
         */
        virtual bool prepareExpertGroups(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k);

        /**
         * @brief Get number of tokens routed to a specific expert.
         * Only valid after prepareExpertGroups().
         */
        virtual int getExpertTokenCount(int expert_id) const;

        /**
         * @brief Gather tokens for expert_id using pre-grouped device indices.
         * Only valid after prepareExpertGroups().
         */
        virtual void gatherExpertBatch(
            ITensor *hidden, ITensor *batch_buffer,
            int expert_id, int d_model);

        /**
         * @brief Scatter weighted expert results using pre-grouped device data.
         * Only valid after prepareExpertGroups().
         */
        virtual void scatterExpertResults(
            ITensor *output, ITensor *expert_results,
            int expert_id, int d_model);

    protected:
        // State for CPU-default prepareExpertGroups / getExpertTokenCount /
        // gatherExpertBatch / scatterExpertResults.
        // GPU overrides manage their own device-resident equivalents.
        std::vector<int> host_expert_counts_;
        std::vector<int> host_expert_offsets_;
        std::vector<int> host_grouped_indices_;
        std::vector<float> host_grouped_weights_;
        int prepared_num_experts_ = 0;
    };

} // namespace llaminar2
