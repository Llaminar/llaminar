/**
 * @file IInferenceRunner.h
 * @brief Interface for inference execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * Interface implemented by DeviceGraphOrchestrator for inference execution.
 */

#pragma once

#include <optional>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

#include "../../../backends/DeviceId.h"
#include "../../mtp/MTPDecodeCatchup.h"
#include "../../prefix_cache/PrefixCacheStateProbe.h"
#include "../../prefix_cache/PrefixStateSnapshot.h"

namespace llaminar2
{
    struct MTPSpecDecodeMetadataBatch;

    // Forward declarations
    class TensorBase;
    struct PlacementPlan;
    struct GraphExecutorStats;
    struct SamplingParams;
    struct LogitPenalty;
    struct PrefillChunkSchedulerPolicy;
    class MoERebalanceController;

    /**
     * @brief Lightweight view of a device runner's local logits state
     *
     * Returned by getLogitsLocalInfo() to provide GPU pointer, device, and
     * shape information for column-parallel LM head gather and GPU-side sampling.
     * This decouples RankOrchestrator from DeviceGraphOrchestrator's
     * internal InferenceState struct.
     */
    struct LogitsLocalInfo
    {
        const void *gpu_ptr = nullptr;  ///< GPU buffer pointer (nullptr if CPU-only)
        std::optional<DeviceId> device; ///< GPU device for backend lookup
        size_t vocab_local = 0;         ///< Local vocab size (columns in logits_local)
        TensorBase *tensor = nullptr;   ///< Tensor pointer for CPU fallback (data())
        void *stream = nullptr;         ///< Explicit GPU stream (must match forward pass stream)

        // Device-resident scratch for the multi-block argmax reduction (owned by
        // the runner's BufferArena). Supplied to IBackend::argmaxF32() so the
        // CUDA two-pass reduction never has to allocate on the hot path. Null /
        // zero capacity means GPU-side argmax is unavailable for this runner.
        void *argmax_partial_vals = nullptr; ///< FP32 scratch [argmax_partial_capacity]
        void *argmax_partial_idxs = nullptr; ///< INT32 scratch [argmax_partial_capacity]
        int argmax_partial_capacity = 0;     ///< Number of entries in the scratch buffers

        /// True if this info is valid (has a tensor)
        explicit operator bool() const { return tensor != nullptr; }
    };

    enum class DeviceLogitsSource : uint8_t
    {
        Main,
        MTP,
        AllPosition
    };

    enum class DeviceDistributionBuffer : uint8_t
    {
        Target,
        Draft
    };

    struct DeviceSpeculativeVerifyResult
    {
        int32_t token = -1;
        bool accepted = false;
        float accept_probability = 0.0f;
        float accept_threshold = 0.0f;
    };

    /**
     * @brief Lightweight view of a captured snapshot with 2D shape metadata
     *
     * Returned by getSnapshotWithShape() to provide shape information
     * alongside the raw FP32 data pointer. The shape (rows, cols) comes
     * from the stage's getDumpInfo() at capture time, so stages own their
     * own dimension reporting.
     */
    struct SnapshotInfo
    {
        const float *data = nullptr; ///< Pointer to FP32 snapshot data (not owned)
        size_t size = 0;             ///< Total element count (rows * cols)
        size_t rows = 0;             ///< Logical rows (e.g. seq_len)
        size_t cols = 0;             ///< Logical cols (e.g. hidden_dim, kv_dim, d_ff)

        explicit operator bool() const { return data != nullptr && size > 0; }
    };

    /**
     * @brief Optional high-level decode-step result for orchestration-aware callers.
     *
     * Low-level runners expose forward() plus explicit sampling. Runners wrapped
     * around IOrchestrationRunner can expose decodeStep() so benchmark mode
     * measures MTP, rollback, and decode-boundary maintenance through the same
     * path as normal generation.
     */
    struct DecodeStepOutput
    {
        std::vector<int32_t> tokens;
        bool is_complete = false;
        std::string error;
    };

    /**
     * @brief Execution path type
     */
    enum class ExecutionPath
    {
        PIPELINE, ///< Traditional imperative pipeline (Qwen2Pipeline)
        GRAPH     ///< Graph-based execution (DeviceGraphOrchestrator)
    };

    /**
     * @brief Interface for inference execution
     *
     * Implemented by DeviceGraphOrchestrator for transformer model inference.
     */
    class IInferenceRunner
    {
    public:
        virtual ~IInferenceRunner() = default;

        // =====================================================================
        // Core Inference API
        // =====================================================================

        /**
         * @brief Run forward pass
         *
         * @param tokens Token IDs
         * @param seq_len Sequence length
         * @return true if forward succeeded
         */
        virtual bool forward(const int *tokens, int seq_len) = 0;

        /**
         * @brief Whether this runner can execute a prepared bucketed prefill
         *        chunk schedule for the current request state.
         */
        virtual bool supportsPrefillChunkSchedule(int seq_len) const
        {
            (void)seq_len;
            return false;
        }

        /**
         * @brief Execute a prompt/suffix through prepared bucketed prefill chunks.
         *
         * Default false keeps the path opt-in for runners that can preserve KV,
         * logits, and maintenance state across chunk boundaries.
         */
        virtual bool forwardPrefillChunkSchedule(
            const int *tokens,
            int seq_len,
            const PrefillChunkSchedulerPolicy &policy,
            int pad_token_id,
            bool allow_padded_execution)
        {
            (void)tokens;
            (void)seq_len;
            (void)policy;
            (void)pad_token_id;
            (void)allow_padded_execution;
            return false;
        }

        /**
         * @brief Get logits from last forward pass
         *
         * @return Pointer to logits [vocab_size], or nullptr if unavailable
         */
        virtual const float *logits() const = 0;

        virtual bool forwardMTP(int32_t draft_condition_token)
        {
            (void)draft_condition_token;
            return false;
        }

        /**
         * @brief True when the runner can chain depth-0 MTP sidecar calls.
         *
         * Chained drafts use the hidden state produced by the previous sidecar
         * as the next sidecar's terminal-hidden input, while appending shifted
         * MTP KV rows at explicit logical positions. Runners that cannot keep
         * this state contract should return false so callers hard-fail for
         * draft depths greater than one.
         */
        virtual bool supportsChainedMTPDrafts() const { return false; }

        /**
         * @brief True when forwardMTPAndSampleGreedy() can provide a runner-native
         *        combined sidecar/sample path.
         */
        virtual bool supportsMTPSidecarSampleFusion() const { return false; }

        /**
         * @brief True when MTP sidecar execution is isolated from main live state.
         *
         * Runners that return true guarantee forwardMTP*() may append speculative
         * MTP-side shifted KV rows and mutate MTP scratch/logits, but it does not
         * advance or otherwise mutate the main verifier state: main KV cache,
         * main hybrid/GDN state, main logits, terminal hidden, positions, or
         * decode bookkeeping. The caller may then skip restoring the verifier
         * base checkpoint after a sidecar draft and rely on explicit shifted-row
         * commit/truncate calls to discard speculative MTP rows.
         */
        virtual bool supportsMTPSidecarPreservesMainState() const { return false; }

        /**
         * @brief True when the verifier path must replay accepted tokens through
         *        normal one-token decode to preserve mutable model state exactly.
         *
         * Stateful architectures such as hybrid GDN may produce all-position
         * verifier logits whose token rows look plausible while the final
         * recurrent/conv state is not byte-for-byte decode-equivalent. Returning
         * true makes the MTP runner use the common sequential verifier replay
         * path instead of the batched all-position verifier for greedy decode.
         */
        virtual bool requiresMTPDecodeEquivalentVerifierReplay() const { return false; }

        /**
         * @brief True when the runner provides a backend-native implementation
         *        of the shared decode-equivalent catch-up contract.
         *
         * The optimized path must produce the same result and live-state
         * effects as runSharedStepwiseMTPDecodeCatchupGreedy(). Returning true
         * makes OrchestrationRunner call runOptimizedMTPDecodeCatchupGreedy()
         * for greedy decode-equivalent verification.
         */
        virtual bool supportsOptimizedMTPDecodeCatchupGreedy() const { return false; }

        /**
         * @brief Stable counter tag for the optimized catch-up implementation.
         */
        virtual const char *optimizedMTPDecodeCatchupGreedyName() const
        {
            return "optimized";
        }

        /**
         * @brief Run the backend-native optimized catch-up implementation.
         *
         * Implementations must preserve the MTPDecodeCatchupGreedyResult
         * contract and leave runner state decode-equivalent to the shared
         * stepwise oracle. The default body is a hard failure so an accidental
         * capability advertisement cannot silently fall back inside the hook.
         */
        virtual MTPDecodeCatchupGreedyResult runOptimizedMTPDecodeCatchupGreedy(
            const MTPDecodeCatchupGreedyRequest &request,
            const MTPDecodeCatchupGreedySampler &sample_after_forward)
        {
            (void)request;
            (void)sample_after_forward;
            MTPDecodeCatchupGreedyResult result;
            result.ok = false;
            result.error = "runner advertised optimized MTP decode catch-up without implementation";
            return result;
        }

        /**
         * @brief Run a chained MTP sidecar step from the previous sidecar hidden.
         *
         * @param draft_condition_token Token whose shifted MTP KV row is appended.
         * @param position_id Logical shifted-cache position for the append.
         */
        virtual bool forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id)
        {
            (void)draft_condition_token;
            (void)position_id;
            return false;
        }

        /**
         * @brief Run a sidecar step and greedily sample its logits as one logical
         *        operation.
         *
         * GPU runners may use this to keep captured sidecar replay and the argmax
         * reduction on one stream, avoiding an intermediate host synchronization.
         * The default implementation preserves the existing synchronous contract.
         */
        virtual bool forwardMTPAndSampleGreedy(int32_t draft_condition_token, int32_t *out_token)
        {
            if (!out_token)
                return false;
            if (!forwardMTP(draft_condition_token))
                return false;
            const int token = sampleGreedyFromMTPLogitsOnDevice();
            if (token < 0)
                return false;
            *out_token = token;
            return true;
        }

        /**
         * @brief Chained sidecar variant of forwardMTPAndSampleGreedy().
         */
        virtual bool forwardMTPFromLastDraftAndSampleGreedy(
            int32_t draft_condition_token,
            int position_id,
            int32_t *out_token)
        {
            if (!out_token)
                return false;
            if (!forwardMTPFromLastDraft(draft_condition_token, position_id))
                return false;
            const int token = sampleGreedyFromMTPLogitsOnDevice();
            if (token < 0)
                return false;
            *out_token = token;
            return true;
        }

        /**
         * @brief Flush deferred sidecar GPU work before checkpoint/verifier reads.
         *
         * Graph-captured MTP sidecars may defer their final stream sync so a
         * fused sampler can run on the same stream. Callers that need to read
         * or checkpoint sidecar-mutated KV/GDN state before sampling must make
         * that ordering explicit through this hook.
         */
        virtual bool flushPendingMTPWork() { return true; }

        /**
         * @brief Commit shifted MTP KV rows from the most recent main forward.
         *
         * MTP decode calls forwardMTP() before verifier/replay; that sidecar
         * step already appends the shifted row for tokens[0]. After the main
         * verifier or replay forward produces hidden rows for the accepted
         * token sequence, this method appends any remaining shifted rows so
         * the depth-0 MTP KV cache returns to the main_position - 1 invariant.
         *
         * @param tokens Accepted/correction tokens from the last main forward.
         * @param token_count Number of accepted/correction tokens.
         * @param already_appended_tokens Prefix of tokens already represented
         *        by the speculative sidecar KV append.
         */
        virtual bool commitMTPShiftedRowsFromLastForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens)
        {
            (void)tokens;
            return token_count <= already_appended_tokens;
        }

        /**
         * @brief Commit shifted MTP rows when the usable verifier hidden rows
         *        cover only a prefix of the emitted token sequence.
         *
         * Verifier-row restore can reuse mutable state for an accepted prefix
         * and then replay only a rejected correction suffix. In that case the
         * current hidden buffer may cover fewer rows than the logical token
         * span used to compute the shifted-cache position. Callers can pass an
         * explicit position_offset_override to keep those contracts separate.
         *
         * Set allow_speculative_discard only when the current MTP sidecar cache
         * is known to contain speculative rows produced by verifier-owned draft
         * steps. Generic callers should leave it false so unexpected extra rows
         * remain a hard failure.
         */
        virtual bool commitMTPShiftedRowsFromPartialForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens,
            int main_forward_token_count,
            bool allow_speculative_discard = false,
            int position_offset_override = -1)
        {
            (void)main_forward_token_count;
            (void)allow_speculative_discard;
            (void)position_offset_override;
            return commitMTPShiftedRowsFromLastForward(
                tokens,
                token_count,
                already_appended_tokens);
        }

        /**
         * @brief Append one shifted MTP KV row using the current terminal hidden.
         *
         * This is used by decode-equivalent replay paths that advance the main
         * model one token at a time. The shifted sidecar row for tokens[i] must
         * be produced before tokens[i] is forwarded by the main model, while
         * the current terminal hidden still represents tokens[i - 1].
         */
        virtual bool commitMTPShiftedRowFromCurrentTerminalHidden(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1)
        {
            (void)token;
            (void)already_appended_tokens;
            (void)allow_speculative_discard;
            (void)position_offset_override;
            return false;
        }

        virtual const float *mtpLogits() const
        {
            return nullptr;
        }

        virtual bool setComputeAllPositionLogits(bool enabled)
        {
            (void)enabled;
            return false;
        }

        virtual const float *getAllPositionLogits() const
        {
            return nullptr;
        }

        /**
         * @brief Check if this runner has column-parallel local all-position logits.
         *
         * Used by MTP verification when TP participants each own a vocabulary shard
         * for every verified position.
         */
        virtual bool hasAllPositionLogitsLocal() const { return false; }

        /**
         * @brief Get local all-position verifier logits info for TP gathering.
         */
        virtual LogitsLocalInfo getAllPositionLogitsLocalInfo() const { return {}; }

        virtual std::string mtpDecodeUnsupportedReason() const
        {
            return {};
        }

        /**
         * @brief True when this runner coordinates MTP draft-token choices across its domain.
         *
         * Multi-rank MTP decode requires every participant to verify and replay the
         * same draft sequence. Runners that span an MPI or TP domain should return
         * true only when their MTP sampling methods broadcast or otherwise agree on
         * MTP draft and verifier tokens.
         */
        virtual bool supportsMTPTokenCoordination() const { return false; }

        /**
         * @brief Sample the current MTP sidecar logits in greedy mode.
         *
         * Returns -1 when unavailable; callers may fall back to mtpLogits() on
         * single-rank paths. Multi-rank runners should coordinate the returned
         * token across all participants.
         */
        virtual int sampleGreedyFromMTPLogitsOnDevice() { return -1; }

        /**
         * @brief Sample one row from all-position verifier logits in greedy mode.
         *
         * @param row Logical verifier row to sample.
         * @return Token id, or -1 when unavailable.
         */
        virtual int sampleGreedyFromAllPositionLogitsOnDevice(int row)
        {
            (void)row;
            return -1;
        }

        /**
         * @brief Sample several contiguous verifier-logit rows in greedy mode.
         *
         * Implementations may use a backend batched argmax to avoid one
         * host/device synchronization per row. The default preserves the
         * existing contract by sampling each row individually, with a host-side
         * greedy scan fallback when all-position logits are already CPU-visible.
         */
        virtual bool sampleGreedyFromAllPositionLogitsOnDeviceRows(
            int start_row,
            int row_count,
            int32_t *out_tokens)
        {
            if (start_row < 0 || row_count <= 0 || !out_tokens)
                return false;

            const float *all_logits = getAllPositionLogits();
            const int vocab = vocab_size();
            for (int i = 0; i < row_count; ++i)
            {
                const int row = start_row + i;
                int token = sampleGreedyFromAllPositionLogitsOnDevice(row);
                if (token < 0 && all_logits && vocab > 0)
                {
                    const float *row_logits =
                        all_logits + static_cast<size_t>(row) * static_cast<size_t>(vocab);
                    int best = 0;
                    float best_value = row_logits[0];
                    for (int col = 1; col < vocab; ++col)
                    {
                        if (row_logits[col] > best_value)
                        {
                            best_value = row_logits[col];
                            best = col;
                        }
                    }
                    token = best;
                }
                if (token < 0)
                    return false;
                out_tokens[i] = static_cast<int32_t>(token);
            }
            return true;
        }

        /**
         * @brief Batched forward pass
         *
         * Process multiple sequences in parallel with automatic padding.
         *
         * @param token_batches Vector of token sequences
         * @return true if forward pass succeeded
         */
        virtual bool forward_batch(const std::vector<std::vector<int>> &token_batches)
        {
            (void)token_batches;
            return false; // Default: not implemented
        }

        /**
         * @brief Get logits for a specific sequence in batch
         *
         * Returns logits for the specified sequence index.
         * For E2E tests that compare all positions.
         *
         * @param seq_idx Sequence index in batch (default=0)
         * @return Pointer to logits [padded_seq_len, vocab_size], or nullptr
         */
        virtual const float *getLogits(int seq_idx = 0) const
        {
            (void)seq_idx;
            return logits(); // Default: return single-sequence logits
        }

        /**
         * @brief Get current batch size
         */
        virtual int batch_size() const { return 1; }

        /**
         * @brief Get padded sequence length for current batch
         */
        virtual int padded_seq_len() const { return 0; }

        /**
         * @brief Get sequence lengths for current batch
         *
         * @return Vector of actual (unpadded) sequence lengths
         */
        virtual const std::vector<int> &sequence_lengths() const
        {
            static const std::vector<int> empty;
            return empty;
        }

        /**
         * @brief Get vocabulary size
         */
        virtual int vocab_size() const = 0;

        /**
         * @brief Reset request/session state for a new sequence.
         *
         * Implementations must clear KV/recurrent/request-local state and reset
         * stage dynamic metadata, but preserve reusable graph topology, prepared
         * weights, workspace bindings, and device contexts. Destructive cache
         * invalidation should use an explicitly named implementation method.
         */
        virtual void clear_cache() = 0;

        /**
         * @brief GPU-side greedy sampling (skip D2H of full logits)
         *
         * For multi-GPU TP inference, this performs argmax on each device's
         * local logits partition on the GPU, then D2H only the (value, index)
         * result pairs (8 bytes per device vs ~600 KB for full logits).
         *
         * @return Token ID (>= 0) if on-device sampling succeeded,
         *         -1 if not supported (caller should fall back to logits() + CPU argmax)
         */
        virtual int sampleGreedyOnDevice() { return -1; }

        /**
         * @brief GPU-side sampling with full top-k/top-p support
         *
         * For greedy (temperature=0), delegates to sampleGreedyOnDevice().
         * For non-greedy, runs per-device GPU top-k selection, then performs
         * cross-device merge + softmax + top-p filtering + sampling on host
         * (operating on only k candidates, not the full vocabulary).
         *
         * @param params Sampling parameters (temperature, top_k, top_p, seed)
         * @return Token ID (>= 0) on success, -1 if not supported
         */
        virtual int sampleOnDevice(const SamplingParams &params)
        {
            (void)params;
            return -1;
        }

        /**
         * @brief Whether this runner can execute a full high-level decode step.
         */
        virtual bool supportsDecodeStep() const { return false; }

        /**
         * @brief Set sampling params consumed by decodeStepForBenchmark().
         */
        virtual void setDecodeSamplingParams(const SamplingParams & /*params*/) {}

        /**
         * @brief Limit how many tokens the next decode step may accept.
         */
        virtual void setDecodeStepTokenBudget(int /*max_tokens*/) {}

        /**
         * @brief Execute one high-level decode step.
         */
        virtual DecodeStepOutput decodeStepForBenchmark()
        {
            return DecodeStepOutput{{}, false, "decodeStepForBenchmark unsupported"};
        }

        /**
         * @brief Apply decode-boundary maintenance after a successful step.
         */
        virtual bool maybeApplyDecodeBoundaryMaintenance() { return true; }

        /**
         * @brief Apply sparse logit penalties on device (GPU-side)
         *
         * Uploads a sparse penalty map to the GPU and applies it in-place to the
         * logits tensor. This avoids a full D2H transfer of the logits tensor
         * (~600KB for 151K vocab) just to apply penalties.
         *
         * After this call, the penalized logits remain on GPU and can be sampled
         * via sampleGreedyOnDevice() or sampleOnDevice().
         *
         * @param penalties Sparse penalty entries (token_id, penalty) to subtract
         * @param vocab_size Vocabulary size (for bounds checking)
         * @return true if applied on device, false if not supported (caller should fall back)
         */
        virtual bool applyPenaltiesOnDevice(const std::vector<LogitPenalty> &penalties,
                                            int vocab_size)
        {
            (void)penalties;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief Apply sparse logit penalties to MTP sidecar logits on device.
         */
        virtual bool applyPenaltiesToMTPLogitsOnDevice(const std::vector<LogitPenalty> &penalties,
                                                       int vocab_size)
        {
            (void)penalties;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief Apply sparse logit penalties to one all-position verifier row on device.
         */
        virtual bool applyPenaltiesToAllPositionLogitsOnDeviceRow(
            int row,
            const std::vector<LogitPenalty> &penalties,
            int vocab_size)
        {
            (void)row;
            (void)penalties;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief True when compact device-side stochastic MTP distributions are available.
         *
         * This is intentionally narrower than generic sampling support: it means
         * the runner can build compact top-k/top-p probability tables for main,
         * MTP, and all-position verifier logits using explicit streams and
         * arena-owned buffers, then verify accept/residual decisions without
         * copying full logits to host.
         */
        virtual bool supportsDeviceStochasticMTPVerification() const { return false; }

        virtual bool buildStochasticDistributionOnDevice(
            DeviceLogitsSource source,
            int row,
            DeviceDistributionBuffer buffer,
            int slot,
            const SamplingParams &params,
            int vocab_size)
        {
            (void)source;
            (void)row;
            (void)buffer;
            (void)slot;
            (void)params;
            (void)vocab_size;
            return false;
        }

        virtual int sampleStochasticDistributionOnDevice(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold)
        {
            (void)buffer;
            (void)slot;
            (void)threshold;
            return -1;
        }

        virtual bool verifyStochasticDistributionsOnDevice(
            int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            DeviceSpeculativeVerifyResult *out)
        {
            (void)target_slot;
            (void)draft_slot;
            (void)draft_token;
            (void)accept_threshold;
            (void)residual_threshold;
            (void)out;
            return false;
        }

        /**
         * @brief Enable GPU-side decode sampling mode
         *
         * When enabled, forward() may skip gathering logits to host for decode calls.
         * Caller should use sampleGreedyOnDevice() instead of logits().
         * Default: no-op (not all runners support this).
         */
        virtual void setSkipLogitsGatherDecode(bool) {}

        /**
         * @brief Skip logits gather after prefill (seq_len > 1)
         *
         * In the standard generation flow, prefill logits are never consumed —
         * the first generated token comes from a decode step. Skipping the
         * D2H logits gather for prefill eliminates massive PCIe traffic
         * (e.g. 346 MB for 596 tokens × 152064 vocab across 2 devices).
         * Default: no-op (not all runners support this).
         */
        virtual void setSkipLogitsGatherPrefill(bool) {}

        /**
         * @brief Suppress GPU stage timeline output
         *
         * When enabled, the GPU stage timeline summary is not printed after
         * each forward pass. Used by BenchmarkRunner to exclude warmup runs
         * from overhead reporting.
         * Default: no-op (not all runners support this).
         */
        virtual void setSuppressTimeline(bool) {}

        /**
         * @brief Set prefill accumulation mode for benchmark
         *
         * When enabled, prefill GPU stage timelines are accumulated across
         * iterations instead of being printed immediately. Used by
         * BenchmarkRunner to avoid per-iteration prefill table spam.
         * Default: no-op (not all runners support this).
         */
        virtual void setAccumulatePrefill(bool) {}

        /**
         * @brief Flush accumulated GPU stage timeline data
         *
         * Prints accumulated decode stage timing summary and resets.
         * Called after decode phase completes (e.g., by BenchmarkRunner).
         * Default: no-op (not all runners support this).
         */
        virtual void flushStageTimeline() {}

        /**
         * @brief Get current position in cache
         */
        virtual int get_position() const = 0;

        /**
         * @brief Get execution path being used
         */
        virtual ExecutionPath executionPath() const = 0;

        /**
         * @brief Get architecture name (e.g., "qwen2")
         */
        virtual const char *architecture() const = 0;

        // =====================================================================
        // Snapshot Capture API (for E2E parity testing)
        // =====================================================================
        // These methods have default no-op implementations for builds without
        // snapshot support. Override in Pipeline (with ENABLE_PIPELINE_SNAPSHOTS)
        // or DeviceGraphOrchestrator (always available) for actual snapshot capture.

        /**
         * @brief Enable snapshot capture of intermediate activations
         *
         * When enabled, each forward pass will capture intermediate tensor
         * values at instrumented stages for comparison against ground truth.
         *
         * @param output_dir Optional directory to save snapshots (implementation-specific)
         */
        virtual void enableSnapshotCapture(const std::string &output_dir = "")
        {
            (void)output_dir; // No-op by default
        }

        /**
         * @brief Disable snapshot capture and clear stored snapshots
         */
        virtual void disableSnapshotCapture() {}

        /**
         * @brief Clear stored snapshots but keep capture enabled
         */
        virtual void clearSnapshots() {}

        /**
         * @brief Retrieve a captured snapshot by key
         *
         * @param key Snapshot identifier (e.g., "layer0_Q_PROJECTION", "EMBEDDING")
         * @param out_size Output parameter for snapshot size in bytes
         * @return Pointer to snapshot data (FP32), or nullptr if key doesn't exist
         */
        virtual const float *getSnapshot(const std::string &key, size_t &out_size) const
        {
            (void)key;
            out_size = 0;
            return nullptr; // No snapshot support by default
        }

        /**
         * @brief Retrieve a captured snapshot with 2D shape metadata
         *
         * Returns the snapshot data along with the rows/cols that the stage
         * reported via getDumpInfo() at capture time. This allows callers to
         * understand the 2D layout without model-specific inference logic.
         *
         * @param key Snapshot identifier (e.g., "layer0_Q_PROJECTION")
         * @return SnapshotInfo with data pointer and shape, or empty if not found
         */
        virtual SnapshotInfo getSnapshotWithShape(const std::string &key) const
        {
            (void)key;
            return {};
        }

        /**
         * @brief Get list of all captured snapshot keys
         *
         * @return Vector of snapshot identifiers
         */
        virtual std::vector<std::string> getSnapshotKeys() const
        {
            return {}; // No snapshots by default
        }

        // =====================================================================
        // Hidden State API (for Pipeline Parallelism)
        // =====================================================================

        /**
         * @brief Get final hidden state from last forward pass
         *
         * Returns the hidden state tensor after all transformer layers have
         * executed. This is used for Pipeline Parallelism to transfer
         * activations between stages.
         *
         * @return Pointer to hidden state tensor [seq_len, d_model], or nullptr
         */
        virtual TensorBase *getHiddenState() { return nullptr; }
        virtual const TensorBase *getHiddenState() const { return nullptr; }

        /**
         * @brief Set initial hidden state for forward pass
         *
         * For PP stages that don't have embedding (middle/final stages),
         * this sets the hidden state that would normally come from embedding.
         * The forward pass will skip embedding and use this tensor directly.
         *
         * @param hidden_state Tensor containing hidden state [seq_len, d_model]
         */
        virtual void setHiddenState(TensorBase *hidden_state) { (void)hidden_state; }

        /**
         * @brief Check if this runner has hidden state set for next forward
         */
        virtual bool hasHiddenStateInput() const { return false; }

        /**
         * @brief Clear hidden state input (reset to normal embedding mode)
         */
        virtual void clearHiddenStateInput() {}

        // =====================================================================
        // Device & Logits Local API
        // =====================================================================
        // These methods expose device identity and column-parallel logits state
        // so that RankOrchestrator can coordinate logits gathering and
        // GPU-side sampling without downcasting to a concrete runner type.

        /**
         * @brief Get the primary device this runner executes on
         *
         * @return DeviceId of the primary compute device (CPU by default)
         */
        virtual DeviceId primaryDeviceId() const { return DeviceId::cpu(); }

        /**
         * @brief Check if this runner has column-parallel local logits
         *
         * True when the LM head is column-parallel and logits_local is allocated.
         *
         * @return true if getLogitsLocalInfo() will return valid info
         */
        virtual bool hasLogitsLocal() const { return false; }

        /**
         * @brief Get local logits info for column-parallel gathering
         *
         * Returns GPU pointer, device, shape, and tensor pointer for the
         * per-device logits shard. Used by RankOrchestrator for
         * AllGather of column-parallel LM head output.
         *
         * @return LogitsLocalInfo (empty by default)
         */
        virtual LogitsLocalInfo getLogitsLocalInfo() const { return {}; }

        /**
         * @brief Check if this runner has column-parallel local MTP logits
         *
         * True when the MTP sidecar LM head writes a local vocabulary shard
         * instead of a full replicated logits row.
         */
        virtual bool hasMTPLogitsLocal() const { return false; }

        /**
         * @brief Get local MTP logits info for column-parallel gathering
         *
         * Uses the same LogitsLocalInfo contract as the main LM head so TP
         * orchestration can gather sidecar logits without knowing runner internals.
         */
        virtual LogitsLocalInfo getMTPLogitsLocalInfo() const { return {}; }

        // =====================================================================
        // Profiling API
        // =====================================================================

        /**
         * @brief Get executor statistics for profiling
         *
         * @return Pointer to GraphExecutorStats, or nullptr if not available
         */
        virtual const GraphExecutorStats *executorStats() const { return nullptr; }

        /**
         * @brief Reset executor statistics
         */
        virtual void resetExecutorStats() {}

        // =====================================================================
        // Orchestration API (for heterogeneous device placement)
        // =====================================================================

        /**
         * @brief Check if this runner has a PlacementPlan configured
         *
         * @return true if a PlacementPlan was provided during creation
         */
        virtual bool hasPlacementPlan() const { return false; }

        /**
         * @brief Get the PlacementPlan this runner is using
         *
         * @return Reference to the PlacementPlan
         * @throws std::runtime_error if no plan is configured (check hasPlacementPlan first)
         */
        virtual const PlacementPlan &getPlacementPlan() const
        {
            throw std::runtime_error("No PlacementPlan configured for this runner");
        }

        /**
         * @brief Domain-local MoE placement epoch used by prefix and graph caches.
         *
         * Non-MoE runners return 0. MoE runners increment this when expert
         * ownership, masks, replicas, or runtime-table placement changes.
         */
        virtual uint64_t moePlacementEpoch() const { return 0; }

        /**
         * @brief Enumerate MoE rebalance controllers owned by this runner.
         *
         * Single-device runners may own a primary controller plus routed-overlay
         * domain controllers. Composite runners return every local domain
         * controller so callers can avoid treating the first available device
         * controller as the multi-domain API.
         */
        virtual std::vector<MoERebalanceController *> moeRebalanceControllers() const { return {}; }

        /**
         * @brief Lookup a MoE rebalance controller by ExpertParallel domain id.
         */
        virtual MoERebalanceController *moeRebalanceControllerForDomain(
            const std::string &domain_id) const
        {
            (void)domain_id;
            return nullptr;
        }

        /**
         * @brief Participant index used for domain-scoped MoE rebalance actions.
         *
         * Single-device runners return 0. GlobalTP runners return rank-in-domain,
         * which is distinct from MPI local rank on multi-node or multi-domain runs.
         */
        virtual int moeRebalanceParticipantId() const { return 0; }

        /**
         * @brief Find the longest reusable prefix for a token sequence.
         *
         * Default runners do not support persistent prefix state yet. Concrete
         * implementations return a populated result only when the feature is
         * enabled and the active backend can import/export logical KV blocks.
         */
        virtual PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens)
        {
            (void)tokens;
            return {};
        }

        virtual bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0)
        {
            (void)hit;
            (void)seq_idx;
            return false;
        }

        virtual bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count)
        {
            (void)tokens;
            (void)prompt_token_count;
            return false;
        }

        virtual bool restorePrefixTerminalState(const PrefixLookupResult &hit)
        {
            (void)hit;
            return false;
        }

        virtual PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const
        {
            (void)seq_idx;
            return {};
        }

        virtual PrefixStateSnapshot captureLivePrefixCheckpoint(int seq_idx = 0) const
        {
            (void)seq_idx;
            return {};
        }

        virtual bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0)
        {
            (void)snapshot;
            (void)seq_idx;
            return false;
        }

        virtual bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0)
        {
            (void)cached_tokens;
            (void)seq_idx;
            return false;
        }

        /**
         * @brief Whether the runner can restore mutable MTP verifier state
         *        directly from the most recent verifier-row captures.
         *
         * When true, the MTP decode path may skip expensive post-sidecar
         * checkpoint payload captures for narrow greedy paths and require
         * restoreMTPVerifierStateRow() to succeed on rollback.
         */
        virtual bool supportsMTPVerifierStateRowRestore() const
        {
            return false;
        }

        /**
         * @brief Restore mutable verifier state from a captured all-position
         *        verifier row, without replaying the accepted verifier prefix.
         *
         * The target cached token count is the logical main-model position after
         * the restored row. Implementations must restore every mutable hybrid
         * state participant they own, truncate main KV state to target_cached_tokens,
         * and update decode position bookkeeping. Unsupported topologies return
         * false so callers can select the slower checkpoint replay path.
         */
        virtual bool restoreMTPVerifierStateRow(
            int verifier_row,
            int target_cached_tokens,
            int seq_idx = 0)
        {
            (void)verifier_row;
            (void)target_cached_tokens;
            (void)seq_idx;
            return false;
        }

        /**
         * @brief Upload graph-facing spec-decode metadata and publish the committed verifier state row.
         *
         * Phase 13.8 uses this to restore mutable hybrid state from
         * committed_state_rows[request_index] without host-side row selection.
         * Implementations must upload metadata on an explicit stream, restore
         * all participant-local stateful layers from that device metadata, then
         * truncate KV/bookkeeping to target_cached_tokens atomically for the
         * request. Unsupported implementations must return false.
         */
        virtual bool restoreMTPVerifierStateFromSpecDecodeMetadata(
            const MTPSpecDecodeMetadataBatch &batch,
            int request_index,
            int target_cached_tokens,
            int seq_idx = 0)
        {
            (void)batch;
            (void)request_index;
            (void)target_cached_tokens;
            (void)seq_idx;
            return false;
        }

        /**
         * @brief Read-only runtime state probe for prefix-cache/MTP development.
         *
         * This is diagnostic state only: callers must not mutate runner-owned
         * buffers through the returned value.
         */
        virtual PrefixRuntimeStateSnapshot prefixStateProbe() const { return {}; }
    };

} // namespace llaminar2
