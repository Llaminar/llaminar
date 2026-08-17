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
        static constexpr const char *WS_SPECULATIVE_STATE_SLOTS = "gdn_speculative_state_slots";
        static constexpr const char *WS_SPECULATIVE_STATE_WORK = "gdn_speculative_state_work";

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

            /**
             * @brief CPU-owned live recurrence state, or null for a GPU stage.
             *
             * CPU kernels receive this stable host pointer directly. CUDA and
             * ROCm kernels own their persistent live state internally, so GPU
             * graph parameters deliberately leave this pointer null. This
             * separation prevents a graph-captured GPU stage from adopting a
             * host mirror whose lifetime or contents can diverge from the
             * backend's resident state bank.
             */
            float *recurrence_state = nullptr;

            int seq_len = 0;
            int request_count = 1;   ///< Number of independent requests in the flattened verifier tensor.
            int request_seq_len = 0; ///< Per-request rows before flattening; 0 means seq_len for legacy graphs.
            /**
             * @brief Host-owned real row counts for native CPU request grouping.
             *
             * GPU graph replay deliberately ignores this mirror and consumes
             * `request_seq_lens_device`; CPU execution uses the stable host
             * vector without staging or per-request scalar dispatch.
             */
            const std::vector<int> *request_seq_lens_host = nullptr;
            /**
             * @brief Device-owned real row count for each request.
             *
             * The stable arena pointer is the sole GPU owner for padded
             * request-batched recurrence. It is consumed directly by the
             * grouped kernel during graph capture/replay and is never adopted
             * into a host scalar.
             */
            const int32_t *request_seq_lens_device = nullptr;
            int n_heads = 0;     ///< Value head count (recurrence operates with this)
            int n_k_heads = 0;   ///< Key head count (for QKV split; 0 = same as n_heads)
            int d_k = 0;         ///< Key head dimension
            int d_v = 0;         ///< Value head dimension
            int chunk_size = 64; ///< Chunk size for prefill

            bool use_qk_l2norm = true; ///< Apply L2 normalization to Q and K

            /// Global V-head offset for TP-aware Q/K selection.
            /// Global V-head offset for TP-aware Q/K selection. Qwen GDN uses
            /// modular Q/K tiling: local V-head j maps to
            /// (j + global_v_head_offset) % n_k_heads.
            int global_v_head_offset = 0;

            int layer_idx = -1; ///< Layer index for logging
            /**
             * @brief Stable graph/workspace namespace for capture-sensitive buffers.
             *
             * Main inference, grouped verifier, and live request-batch graphs can
             * execute independently on different streams. Their verifier-row
             * snapshots and mutable prefill scratch must therefore not alias.
             * Graph builders pass a graph-role prefix such as
             * `grouped_mtp_verifier`; the stage appends the logical layer only
             * for buffers whose contents are layer-persistent.
             *
             * An empty namespace denotes the main inference role and preserves
             * its historical workspace keys.
             */
            std::string workspace_namespace;
            int verifier_state_capture_rows = 0; ///< Compatibility spelling for speculative state slots.
            int speculative_state_slot_rows = 0; ///< Phase 13.8 temporary state slots for MTP verifier rows.

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
        ~GDNRecurrenceStage() override = default;

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

        /**
         * @brief Refresh the logical row geometry for the next graph execution.
         *
         * @param pos_offset Unused by GDN recurrence because logical history is
         *        represented by each request's recurrent state.
         * @param seq_len Number of rows contributed by one request.  The GDN
         *        grouped kernel sees `request_count * seq_len` flattened rows.
         *
         * Request-batched graph inputs describe their dynamic width per
         * request, while `Params::seq_len` describes the complete flattened
         * tensor passed to the backend.  Updating both fields here preserves
         * that distinction across cold execution, graph capture, and replay.
         */
        void updateDynamicParams(int pos_offset, int seq_len) override;
        bool hasDynamicParams() const override { return true; }
        bool supportsDeviceResidentDynamicPositionReplay() const override
        {
            return true;
        }
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            prefill_effective_seq_len_ = 0;
            prefill_bucket_seq_len_ = 0;
            prefill_replay_params_set_ = false;
            if (params_.kernel)
            {
                params_.kernel->clearGPUStreamBinding();
                clearKernelVerifierStateWorkspace();
            }
        }

        /**
         * @brief Reset request-local GDN metadata while preserving capture slots.
         *
         * Prefill graphs capture recurrent-state snapshot buffers by address
         * and read each request's real length from the stable device request
         * metadata allocation. Every explicit publication entry point establishes
         * this stage's verifier-workspace binding before enqueueing its restore
         * kernel; graph replay itself has no host callback. Do not clear the
         * verifier workspace binding while a Ready prefill executable is preserved.
         */
        void resetSessionStatePreservingCapturedReplay() override
        {
            IComputeStage::resetSessionState();
            prefill_effective_seq_len_ = 0;
            prefill_bucket_seq_len_ = 0;
            prefill_replay_params_set_ = false;
            if (params_.kernel)
                params_.kernel->clearGPUStreamBinding();
        }

        /**
         * @brief Preserve warmed GDN workspaces for a fresh capture attempt.
         */
        void resetSessionStatePreservingLazyInitialization() override
        {
            resetSessionStatePreservingCapturedReplay();
        }

        bool hasPrefillReplayParams() const override { return true; }
        void updatePrefillReplayParams(const PrefillReplayParams &replay) override;
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool hasVerifierStateCapture() const override;
        bool requiresVerifierStateCaptureForPublication() const override
        {
            return verifierStateCaptureWorkspaceRequired();
        }
        bool restoreVerifierStateCaptureRow(int row, void *stream = nullptr) override;
        bool restoreVerifierStateCaptureRows(
            const int *host_row_indices,
            int request_count,
            void *stream = nullptr) override;
        /**
         * @brief Expose the selected CPU recurrence snapshot as a byte-copy plan.
         *
         * Planning binds no shared backend state and performs no copy. The
         * central MTP publisher validates every layer plan before committing
         * all independent recurrence matrices in one OpenMP team.
         */
        CPUVerifierStateRestorePlan planCPUVerifierStateRestoreRow(int row) override;
        bool restoreVerifierStateCaptureRowFromDeviceIndex(
            const int *device_row_index,
            void *stream) override;
        /**
         * @brief Restore one captured recurrent state row per request.
         *
         * This is the request-batched companion to the scalar device-indexed
         * restore path.  It deliberately delegates to the backend tensor-kernel
         * contract rather than looping scalar restores, because GDN recurrence
         * state must be request-owned before batched publication is correct.
         */
        bool restoreVerifierStateCaptureRowsFromDeviceIndices(
            const int *device_row_indices,
            int request_count,
            int row_index_stride,
            void *stream) override;
        /**
         * @brief Publish every request's real terminal recurrence state on device.
         *
         * No host row list is materialized: the backend combines resident real
         * lengths with the fixed request row width inside one grouped launch.
         */
        bool restoreVerifierStateCaptureRequestTerminalRows(
            const int *device_request_seq_lens,
            int request_count,
            int request_row_width,
            void *stream) override;
        /**
         * @brief Report direct grouped publication into request-owned live banks.
         *
         * CPU, CUDA, and ROCm grouped request kernels all execute directly
         * against their request state bank when the bounded verifier snapshot
         * window cannot cover the complete padded request matrix. In that
         * geometry the terminal states are already live when execution returns,
         * so a second snapshot restore would be both redundant and incorrect.
         */
        bool requestBatchedTerminalStateCommittedDuringExecution(
            int request_count,
            int request_row_width) const override;
        void clearVerifierStateCaptureBindingAfterPublication() override;
        /// @brief Allows cold GPU prefill graph preflight before warmup allocates recurrence state.
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        /// @brief Allows cold GPU padded-prefill graph preflight before warmup allocates recurrence state.
        bool supportsPaddedPrefillGraphCapturePreflight() const override;

        bool isGraphCapturable() const override;

        const Params &getParams() const { return params_; }

    private:
        Params params_;
        int prefill_effective_seq_len_ = 0;
        int prefill_bucket_seq_len_ = 0;
        bool prefill_replay_params_set_ = false;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        uint32_t workspace_slice_id_ = 0;
        bool verifier_capture_workspace_bound_ = false;
        bool speculative_state_work_bound_ = false;
        int verifier_capture_rows_bound_ = 0;
        int verifier_capture_state_size_bound_ = 0;
        int effectivePrefillSeqLen() const;
        bool shouldUseScalarRealLengthContract() const;
        std::string workspaceStableId() const;
        std::string deinterleaveScratchBufferName() const;
        std::string speculativeStateSlotsBufferName() const;
        std::string speculativeStateWorkBufferName() const;
        int requestedSpeculativeStateSlotRows() const;
        bool verifierStateCaptureWorkspaceRequired() const;
        bool ensureVerifierStateCaptureWorkspaceBound() const;
        void bindKernelWorkspace();
        void clearKernelVerifierStateWorkspace();
        /**
         * @brief Resolve the writable CPU verifier slots from the bound manager.
         *
         * @return The exact manager-owned buffer address, or null when this is a
         *         GPU stage, no slots are bound, or the binding is incomplete.
         *
         * The stage deliberately owns no replacement host container. Returning
         * null therefore makes an omitted graph-family workspace participant a
         * fatal execution error instead of concealing it with private storage.
         */
        float *cpuVerifierStateCaptureWorkspace() const;
        /**
         * @brief Publish one accepted CPU verifier row into live recurrence state.
         * @param row Zero-based row in the manager-owned verifier slot matrix.
         * @return true after an exact state copy, otherwise false.
         */
        bool restoreCPUVerifierStateCaptureRowDirect(int row);
        size_t deinterleaveScratchFloats(int seq_len) const;
        bool ensureGpuDeinterleaveWorkspaceBound(int seq_len) const;
    };

} // namespace llaminar2
