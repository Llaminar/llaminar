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
#include "../../../interfaces/IWorkspaceConsumer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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
    class GDNRecurrenceStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        static constexpr const char *WS_DEINTERLEAVE_SCRATCH = "gdn_deinterleave_scratch";
        static constexpr const char *WS_EFFECTIVE_SEQ_LEN_SCALAR = "gdn_effective_seq_len_scalar";
        static constexpr const char *WS_VERIFIER_STATE_CAPTURE = "gdn_verifier_state_capture";

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

            /// Global V-head offset for TP-aware Q/K selection.
            /// Under tensor parallelism with GDN modular repeat (repeat_type=1),
            /// Q/K are replicated across ranks while V is sharded. Each local
            /// v_head j maps to k_head (j + global_v_head_offset) % n_k_heads.
            /// Set to rank * n_v_heads_local by the graph builder. Default 0
            /// (single-device or TP=1).
            int global_v_head_offset = 0;

            int layer_idx = -1; ///< Layer index for logging
            int verifier_state_capture_rows = 0; ///< Candidate verifier rows to snapshot for MTP rollback.

            /// Kernel implementation (set during graph construction)
            ITensorGatedDeltaNet *kernel = nullptr;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> qkv_buffer_id;   ///< Arena: merged QKV tensor
            std::optional<BufferId> alpha_buffer_id; ///< Arena: alpha projection
            std::optional<BufferId> beta_buffer_id;  ///< Arena: beta projection
            std::optional<BufferId> output_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit GDNRecurrenceStage(Params params);
        ~GDNRecurrenceStage() override;

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GDN_RECURRENCE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            (void)pos_offset; // GDN layers don't use position offsets
            params_.seq_len = seq_len;
        }
        bool hasDynamicParams() const override { return true; }
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            prefill_effective_seq_len_ = 0;
            prefill_bucket_seq_len_ = 0;
            prefill_replay_params_set_ = false;
            if (params_.kernel)
                params_.kernel->setGPUStream(nullptr);
        }

        bool hasPrefillReplayParams() const override { return true; }
        void updatePrefillReplayParams(const PrefillReplayParams &replay) override;
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool hasVerifierStateCapture() const override;
        bool restoreVerifierStateCaptureRow(int row, void *stream = nullptr) override;
        bool restoreVerifierStateCaptureRowFromDeviceMetadata(
            const int32_t *device_committed_state_rows,
            int request_index,
            void *stream) override;
        /// @brief Allows cold GPU padded-prefill graph preflight before warmup allocates recurrence state.
        bool supportsPaddedPrefillGraphCapturePreflight() const override;

        bool isGraphCapturable() const override;

        const Params &getParams() const { return params_; }

    private:
        struct GpuEffectiveSeqLenState;

        Params params_;
        int prefill_effective_seq_len_ = 0;
        int prefill_bucket_seq_len_ = 0;
        bool prefill_replay_params_set_ = false;
        std::unique_ptr<GpuEffectiveSeqLenState> gpu_effective_seq_len_state_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        uint32_t workspace_slice_id_ = 0;
        bool verifier_capture_workspace_bound_ = false;
        int verifier_capture_rows_bound_ = 0;
        int verifier_capture_state_size_bound_ = 0;

        // Reusable scratch for QKV deinterleaving (grow-only)
        mutable std::vector<float> q_deinterleave_;
        mutable std::vector<float> k_deinterleave_;
        mutable std::vector<float> v_deinterleave_;

        int effectivePrefillSeqLen() const;
        bool shouldUseRealLengthContract() const;
        std::string workspaceStableId() const;
        std::string effectiveSeqLenScalarBufferName() const;
        std::string verifierStateCaptureBufferName() const;
        bool verifierStateCaptureWorkspaceRequired() const;
        bool ensureVerifierStateCaptureWorkspaceBound() const;
        bool ensureGpuEffectiveSeqLenStateInitialized();
        bool uploadGpuEffectiveSeqLen();
        void refreshPinnedEffectiveSeqLen();
        void releaseGpuEffectiveSeqLenState();
        void bindKernelWorkspace();
        size_t deinterleaveScratchFloats(int seq_len) const;
        bool ensureGpuDeinterleaveWorkspaceBound(int seq_len) const;
    };

} // namespace llaminar2
