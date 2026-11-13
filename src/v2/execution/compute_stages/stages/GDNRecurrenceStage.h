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
#include "GDNPreparedQKBinding.h"
#include "GDNSpeculativeWorkspaceContract.h"
#include "kernels/gdn/GDNPreparedRetainedInvocation.h"
#include "execution/local_execution/device/WorkspaceBufferLease.h"
#include <array>
#include <memory>
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
    class IKVCache;
    class CapturedDeviceStorageLease;

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
        static constexpr const char *WS_RETAINED_STATE_SLOTS = "gdn_exact_retained_state_slots_v1";

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            GDNQKInputContract qk_contract=GDNRawQKInput{}; ///< Immutable raw or paired prepared program.
            /// Cold grouped-verifier selection: exact geometry/output shared by requirements and native launch.
            std::optional<gdn_workspace::GPURetainedRecurrenceWorkspace> gpu_retained_workspace;

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

            /// Cache-owned sequence frontier; GPU recurrent-only slices advance
            /// it on the same stream after their final recurrent layer.
            IKVCache *sequence_state_cache = nullptr;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> qkv_buffer_id;   ///< Arena: merged QKV tensor
            std::optional<BufferId> alpha_buffer_id; ///< Arena: alpha projection
            std::optional<BufferId> beta_buffer_id;  ///< Arena: beta projection
            std::optional<BufferId> output_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit GDNRecurrenceStage(Params params);
        /** @return Immutable recurrence orientation included in retained graph identity. */
        uint64_t graphCaptureVariantSignature() const override;

        ~GDNRecurrenceStage() override = default;

        /** @brief Prepare an immutable retained invocation and physical borrows before recording.
         * @param ctx Exact participant context.
         * @param stream Exact non-default capture stream.
         * @return Prepared or already-identical; no arithmetic or host state read occurs.
         * @throws std::logic_error For missing/stale ownership or a foreign recording stream.
         * @throws std::invalid_argument For malformed native extents or default streams. */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        /** @return Retained programs prepare only at capture, never at replay. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        { return params_.gpu_retained_workspace ? GraphLaunchPreparationPolicy::CaptureOnly : GraphLaunchPreparationPolicy::None; }
        /** @return Whether cold preparation can bind this exact native program. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override
        { return params_.gpu_retained_workspace.has_value(); }
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
            if (params_.gpu_retained_workspace) return;
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
            if (params_.kernel && !params_.gpu_retained_workspace)
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
        /**
         * @brief Publish accepted CPU request prefixes from this stage's own slots.
         * @param host_row_indices One flattened live-row index per request; a
         *        negative entry rejects that request and preserves its live bank.
         * @param request_count Fixed request partition count of this graph.
         * @param stream Unused CPU stream argument retained by the common API.
         * @return True after the request-bank owner validates every selected head
         *         and publishes canonical bytes; false for an invalid transaction.
         * No shared kernel binding is changed. The explicit view borrows this
         * stage's generation authority and arena buffers through publication.
         */
        bool restoreVerifierStateCaptureRows(
            const int *host_row_indices,
            int request_count,
            void *stream = nullptr) override;
        /**
         * @brief Expose the selected CPU recurrence snapshot as a byte-copy plan.
         *
         * @param row Candidate row, or negative to request a no-op plan.
         * @return Ready plan borrowing immutable compact slots, NoOp for
         *         rejection, or Invalid before mutation for malformed bindings,
         *         metadata, candidate extents, or overlapping publication spans.
         *
         * Planning validates every head and returns the selected typed prefix.
         * It binds no shared backend state and changes no work or live bytes.
         * The central publisher validates every layer/source span before directly
         * scattering canonical bytes with one OpenMP team. Immutable slots remain
         * valid until this stage starts another execution, reset, or rebind.
         */
        CPUVerifierStateRestorePlan planCPUVerifierStateRestoreRow(int row) override;
        /** @brief Validate every CPU request and expose exact native snapshot-copy plans before mutation. */
        bool planCPUVerifierStateRestoreRows(const int *rows,int requests,
            std::vector<CPUVerifierStateRestorePlan> &plans) override;
        bool restoreVerifierStateCaptureRowFromDeviceIndex(
            std::span<const int> device_row_index_view,
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
            std::span<const int> device_row_indices_view,
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
        /** @brief Physical owner borrows for one immutable prepared retained invocation. */
        struct RetainedGPUInvocation final
        {
            DeviceWorkspaceManager *workspace = nullptr; ///< Borrowed original named-buffer manager.
            std::uint64_t workspace_owner_identity = 0; ///< Stable manager ID, not allocator address epoch.
            std::shared_ptr<const WorkspaceBufferLease> storage, terminal; ///< Retain original PMA claims.
            std::array<std::shared_ptr<const CapturedDeviceStorageLease>, 6> operands;
            std::array<std::vector<std::size_t>, 6> shapes;
            std::array<std::size_t, 6> bytes{};
            std::unique_ptr<gdn::IGDNPreparedRetainedInvocation> invocation; ///< Destroy before physical leases.
        };
        std::optional<RetainedGPUInvocation> retained_gpu_; ///< Immutable metadata only, never a live-state ledger.
        /** @return Current exact six tensor owners in native source/output order. */
        std::array<const TensorBase *, 6> retainedTensors() const;
        /** @return Raw original tuple without state/snapshot owners, borrowing device live lengths.
         * @throws std::logic_error For missing/non-FP32/foreign-device source storage. */
        gdn::MergedRecurrenceInput retainedSources() const;
        /** @brief Reject stale named buffers, tensor pointers/shapes, geometry or owner identity.
         * @throws std::logic_error On any changed or absent immutable preparation. */
        void requireRetainedInvocation() const;
        /** @brief Launch only this stage's frozen invocation under its exact execution token.
         * @return Native capture succeeded and normal sequence-frontier publication succeeded.
         * @throws std::logic_error For absent/stale preparation or a foreign execution stream. */
        bool executeRetainedInvocation();
        /** @brief Publish an existing typed device selection without rebinding a shared kernel.
         * @param selection Original owner-backed metadata meaning/extent.
         * @param stream Exact publisher invocation stream, possibly different from capture.
         * @return Successful publication; invalid or stale identity throws. */
        bool publishRetainedInvocation(const gdn::GDNDenseAcceptedSelection &selection, void *stream);
        Params params_;
        int prefill_effective_seq_len_ = 0;
        int prefill_bucket_seq_len_ = 0;
        bool prefill_replay_params_set_ = false;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        uint32_t workspace_slice_id_ = 0;
        /**
         * @brief Retire this stage's candidate headers and generation before reuse.
         * @throws std::overflow_error if its monotonically increasing generation
         *         cannot be represented; exhaustion is a fatal lifecycle error.
         * Only this stage's arena payload is affected. Another graph may clear or
         * rebind a shared backend without retiring these stage-owned candidates.
         */
        void retireCPUVerifierCandidates();
        uint64_t cpu_capture_generation_ = 1; ///< Stage-owned transaction lifetime; kernel rebinding cannot change it.
        bool verifier_capture_workspace_bound_ = false;
        bool speculative_state_work_bound_ = false;
        int verifier_capture_rows_bound_ = 0;
        int verifier_capture_state_size_bound_ = 0;
        int effectivePrefillSeqLen() const;
        bool shouldUseScalarRealLengthContract() const;
        std::string workspaceStableId() const;
        std::string deinterleaveScratchBufferName() const;
        std::string speculativeStateSlotsBufferName() const;
        /**
         * @brief Select a concurrency-correct mutable recurrence bank identity.
         * @return Role-shared GPU work name; CPU requests no mutable work buffer.
         * GPU dependencies serialize recurrence work within a role. CPU recurrence
         * writes retained checkpoints directly; neither inference nor publication
         * needs a cloned complete working bank.
         */
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
