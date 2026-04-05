/**
 * @file GDNRecurrenceStage.h
 * @brief Gated Delta Net recurrence stage: delta rule linear attention
 *
 * Implements the gated delta rule recurrence from the Qwen 3.5 architecture:
 *
 * Decode (single step):
 *   S_t = exp(g_t) * S_{t-1}
 *   kv_mem = S_t * k_t  (contract over d_k)
 *   delta_t = (v_t - kv_mem) * beta_t
 *   S_t = S_t + outer(k_t, delta_t)
 *   o_t = S_t * q_t  (contract over d_k)
 *
 * Prefill (chunk-parallel):
 *   Processes the sequence in chunks of chunk_size, combining intra-chunk
 *   attention (causal masked matmul) with inter-chunk state propagation.
 *
 * The stage extracts raw pointers from tensors and delegates all computation
 * (L2 normalization, query scaling, gate computation, recurrence) to
 * the ITensorGatedDeltaNet kernel. This keeps the stage device-agnostic.
 *
 * Reference: torch_recurrent_gated_delta_rule() and torch_chunk_gated_delta_rule()
 *            from HuggingFace transformers 5.4.0
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>
#include <vector>

namespace llaminar2
{

    class ITensorGatedDeltaNet;

    /**
     * @brief Delta rule recurrence for GDN linear attention
     *
     * Two modes:
     * - Prefill (seq_len > 1): Chunk-parallel with intra-chunk causal attention
     * - Decode  (seq_len == 1): Single-step recurrence state update
     *
     * Requires: Q, K, V after conv1d + RoPE; alpha (A), beta (B) from projections;
     *           A_log and dt_bias for computing the gating signal g.
     *
     * Delegates to ITensorGatedDeltaNet* kernel for the core recurrence math.
     */
    class GDNRecurrenceStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input tensors (all FP32, already projected + conv'd + RoPE'd)
            const ITensor *Q = nullptr; ///< Query  [seq_len, n_heads * d_k]
            const ITensor *K = nullptr; ///< Key    [seq_len, n_heads * d_k]
            const ITensor *V = nullptr; ///< Value  [seq_len, n_heads * d_v]

            // Gate inputs (raw projections, gate computation done internally)
            const ITensor *alpha = nullptr; ///< A projection [seq_len, n_heads]
            const ITensor *beta = nullptr;  ///< B projection [seq_len, n_heads]

            // Weight parameters for gate computation: g = -exp(A_log) * softplus(alpha + dt_bias)
            const ITensor *A_log = nullptr;   ///< Learnable log-space gate [n_heads]
            const ITensor *dt_bias = nullptr; ///< Learnable dt bias [n_heads]

            ITensor *output = nullptr; ///< Output [seq_len, n_heads * d_v]

            // Recurrence state [n_heads, d_k, d_v] — persistent across decode steps
            float *recurrence_state = nullptr;

            int seq_len = 0;
            int n_heads = 0;     ///< Value head count (recurrence operates with this)
            int n_k_heads = 0;   ///< Key head count (for QKV split; 0 = same as n_heads)
            int d_k = 0;         ///< Key head dimension
            int d_v = 0;         ///< Value head dimension
            int chunk_size = 64; ///< Chunk size for prefill

            bool use_qk_l2norm = true; ///< Apply L2 normalization to Q and K

            int layer_idx = -1; ///< Layer index for logging

            /// Kernel implementation (set during graph construction)
            ITensorGatedDeltaNet *kernel = nullptr;

            // Optional BufferIds
            std::optional<BufferId> output_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit GDNRecurrenceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GDN_RECURRENCE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        const Params &getParams() const { return params_; }

    private:
        Params params_;

        // Reusable scratch for QKV deinterleaving (grow-only)
        mutable std::vector<float> q_deinterleave_;
        mutable std::vector<float> k_deinterleave_;
        mutable std::vector<float> v_deinterleave_;
    };

} // namespace llaminar2
