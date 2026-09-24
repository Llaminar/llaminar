/**
 * @file CPUGatedDeltaNet.h
 * @brief CPU implementation of ITensorGatedDeltaNet
 *
 * Delta rule recurrence for GDN linear attention.
 * OpenMP-parallelized across heads.
 *
 * The kernel owns ALL preprocessing:
 * - L2 normalization of Q and K (when use_qk_l2norm is true)
 * - Query scaling by 1/sqrt(d_k)
 * - Gate computation: g = -exp(A_log) * softplus(alpha + dt_bias)
 * - Beta sigmoid: beta_sig = sigmoid(beta_raw)
 *
 * This ensures a future CUDA kernel can do all math on-device.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"

#include <vector>

namespace llaminar2
{

    class CPUGatedDeltaNet : public ITensorGatedDeltaNet
    {
    public:
        bool supportsPaddedPrefillRealLength() const override { return true; }
        bool supportsRequestLiveStateBank(int request_count, int state_size) const override
        {
            return request_count > 0 && state_size > 0;
        }
        /**
         * @brief Reset CPU-owned request and speculative state.
         *
         * @param stream Unused on CPU; present for the common state-publication
         *        contract.
         * @return true after all CPU-owned transient state is cleared.
         */
        bool resetGPUState(void *stream) override;
        void bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size) override;
        void bindSpeculativeStateWorkspace(float *workspace, int state_size) override;
        bool restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream) override;

        /**
         * @brief Commit one captured GDN recurrence state per host request.
         *
         * The method validates all requested rows before changing live state.
         * Single-request grouped transactions publish directly into
         * @p dst_state without allocating a request bank.  Multi-request
         * transactions publish into the bank created by grouped request
         * execution and mirror request zero for the public state ABI.
         *
         * @param dst_state Public live recurrence state for request zero.
         * @param host_row_indices Flat capture row for each request; negative
         *        entries preserve the corresponding live request state.
         * @param request_count Number of request row selections.
         * @param stream Unused by the CPU implementation.
         * @return true when publication is valid and completes atomically.
         */
        bool restoreVerifierStateCaptureRows(
            float *dst_state,
            const int *host_row_indices,
            int request_count,
            void *stream) override;

        bool chunk_forward(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm) override;

        /**
         * @brief Advance CPU recurrence directly from merged QKV source rows.
         *
         * Q/K preprocessing reads the mapped source heads in place and writes
         * only the canonical normalized scratch required by recurrence. V is
         * consumed through its original merged-row stride, so no temporary Q,
         * K, or V matrix is allocated or materialized.
         */
        bool chunkForwardMergedQKV(
            const float *merged_qkv, int qkv_stride,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_k_heads, int n_heads, int d_k, int d_v,
            int global_v_head_offset, int chunk_size,
            bool use_qk_l2norm) override;

        /**
         * @brief Advance separate Q/K/V rows and publish every post-row state.
         *
         * Snapshot rows are direct recurrence destinations, not copies made
         * after an alternate verifier kernel.  On successful return @p state is
         * advanced to the final snapshot, preserving the historical explicit
         * snapshot API contract while avoiding one copy per intermediate row.
         *
         * @return true when all rows are byte-equivalent to serial decode and
         *         the terminal state is published; false is a fatal bad contract.
         */
        bool chunkForwardWithStateSnapshots(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm,
            float *state_snapshots, int snapshot_stride_floats,
            int max_snapshot_rows) override;

        /**
         * @brief Run the production merged-QKV verifier without mutating live state.
         *
         * Row zero advances directly from @p state into snapshot zero. Every
         * later row advances the prior immutable snapshot into the next slot.
         * The input state therefore remains authoritative until the MTP
         * transaction explicitly publishes its accepted row.
         *
         * @return true after all output and snapshot rows are complete; false is
         *         a fatal layout, capacity, aliasing, or execution violation.
         */
        bool chunkForwardMergedQKVWithStateSnapshots(
            const float *merged_qkv, int qkv_stride,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_k_heads, int n_heads, int d_k, int d_v,
            int global_v_head_offset, int chunk_size, bool use_qk_l2norm,
            float *state_snapshots, int snapshot_stride_floats,
            int max_snapshot_rows) override;

        bool chunkForwardBatchedRequestsWithHostSeqLens(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int request_count, int request_seq_len,
            int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm,
            const int *host_request_seq_lens) override;

        bool chunkForwardBatchedMergedQKVWithHostSeqLens(
            const float *merged_qkv, int qkv_stride,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int request_count, int request_seq_len,
            int n_k_heads, int n_heads, int d_k, int d_v,
            int global_v_head_offset, bool use_qk_l2norm,
            const int *host_request_seq_lens) override;

        bool restoreStateFromSnapshot(
            float *state, const float *state_snapshots,
            int snapshot_row, int snapshot_stride_floats,
            int state_floats, void *stream = nullptr) override;

        bool recurrent_step(
            const float *q, const float *k, const float *v,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int n_heads, int d_k, int d_v,
            bool use_qk_l2norm) override;

    private:
        /**
         * @brief Read-only source layout consumed by the shared prefill kernel.
         *
         * Q, K, and V bases may belong to separate compact matrices or to
         * subregions of one merged row. Row strides and modular Q/K head mapping
         * make that distinction explicit without branching in model-graph code.
         */
        struct ChunkInputView
        {
            const float *q = nullptr;
            const float *k = nullptr;
            const float *v = nullptr;
            int q_row_stride = 0;
            int k_row_stride = 0;
            int v_row_stride = 0;
            int n_k_heads = 0;
            int global_v_head_offset = 0;
            const char *layout_name = nullptr;
        };

        /**
         * @brief Define whether snapshot execution commits its terminal state.
         *
         * Grouped verifier execution must leave live decode state untouched until
         * the accepted row is selected, whereas the legacy explicit snapshot API
         * promises to advance its supplied state to the terminal row.  Keeping
         * that distinction typed prevents a caller from accidentally selecting
         * ownership semantics through pointer aliasing or an informal boolean.
         */
        enum class InputStateDisposition
        {
            PublishTerminalState,
            PreserveInitialState,
        };

        /// Compute gate values: g = A_log * softplus(alpha + dt_bias), beta_sig = sigmoid(beta_raw)
        static void computeGates(
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *g_out, float *beta_sig_out,
            int seq_len, int n_heads);

        /// L2 normalize vectors per head
        static void l2normalize(float *data, int seq_len, int n_heads, int head_dim);

        /// Ensure scratch buffers are large enough, reallocating only when needed
        void ensureScratch(int seq_len, int n_heads, int d_k, int d_v);

        /**
         * @brief Canonical CPU recurrence over an explicit Q/K/V source view.
         *
         * This is the sole single-request recurrence implementation for ordinary
         * M=1 decode, prefill, and grouped verifier rows.  Every head is owned by
         * one OpenMP work item and advances rows in ascending serial-decode order.
         * Supplying @p state_snapshots only publishes each post-row state; it
         * never selects alternate arithmetic.
         *
         * @param input Typed view of separate or merged Q/K/V source rows.
         * @param alpha Per-row time-step projections.
         * @param beta_raw Per-row unnormalized delta gates.
         * @param A_log Per-head recurrence decay coefficients.
         * @param dt_bias Per-head time-step biases.
         * @param output Destination rows in local-value-head order.
         * @param state Mutable working recurrence state. Verifier callers pass a
         *        private speculative copy so live decode state is not mutated.
         * @param seq_len Number of rows to advance; every positive runtime M is
         *        accepted when the caller provides sufficient storage.
         * @param n_heads Number of local value heads.
         * @param d_k Key/query width and recurrence-state row count.
         * @param d_v Value width and recurrence-state column count.
         * @param chunk_size Retained chunk-policy input for the common interface.
         * @param use_qk_l2norm Whether to apply canonical per-head Q/K L2 norm.
         * @param state_snapshots Optional post-row state publication workspace.
         * @param snapshot_stride_floats Distance between snapshot row bases.
         * @param max_snapshot_rows Writable snapshot-row capacity; when snapshots
         *        are requested it must cover all @p seq_len rows.
         * @param input_state_disposition Whether snapshot execution publishes the
         *        terminal snapshot back to @p state or preserves its initial bytes.
         * @return true after all rows and requested snapshots are complete.
         */
        bool chunkForwardImpl(
            const ChunkInputView &input,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm,
            float *state_snapshots, int snapshot_stride_floats,
            int max_snapshot_rows,
            InputStateDisposition input_state_disposition);

        /**
         * @brief Shared grouped request recurrence over arbitrary row layouts.
         *
         * One OpenMP work item owns a `(request, head)` pair for every real
         * row.  That preserves the exact `recurrent_step()` arithmetic order
         * while exposing all independent request/head work to the CPU pool.
         */
        template <typename RowAccessor>
        bool chunkForwardBatchedDecodeEquivalentRows(
            RowAccessor &&row_accessor,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int request_count, int request_seq_len,
            int n_heads, int d_k, int d_v,
            bool use_qk_l2norm,
            const int *host_request_seq_lens);

        float *prepareSpeculativeState(float *live_state, int state_floats);
        bool ensureRequestStateBank(
            int request_count,
            int state_floats,
            const float *request_zero_state);

        // Reusable scratch buffers (grow-only, never shrink during lifetime)
        std::vector<float> q_scratch_;        ///< Preprocessed Q buffer
        std::vector<float> k_scratch_;        ///< Preprocessed K buffer
        std::vector<float> gate_scratch_;     ///< Gate values
        std::vector<float> beta_sig_scratch_; ///< Sigmoid of beta
        float *verifier_state_capture_ = nullptr;
        int verifier_state_capture_rows_ = 0;
        int verifier_state_capture_size_ = 0;
        float *speculative_state_work_ = nullptr;
        int speculative_state_work_size_ = 0;
        std::vector<float> owned_speculative_state_work_;
        std::vector<float> request_state_bank_;
        int request_state_size_ = 0;
        int request_state_capacity_ = 0;
    };

} // namespace llaminar2
