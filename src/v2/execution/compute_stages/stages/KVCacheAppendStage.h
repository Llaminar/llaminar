/**
 * @file KVCacheAppendStage.h
 * @brief Explicit KV cache append stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "kernels/IKVCache.h"
#include "../../../memory/BufferId.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"

#include <algorithm>
#include <optional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llaminar2
{

    class FP16Tensor;
    class Q8_1Tensor;
    class TQ4Tensor;
    class TQ8Tensor;
    class TensorBase;
    class TurboQuantContext;
    class ActivationRotation;

    /**
     * @brief Mathematical publication contract for one KV append stage.
     *
     * This policy is selected declaratively by the model graph.  In particular,
     * a grouped MTP verifier append is not inferred from mutable cache length:
     * doing so would make graph topology depend on a stale host observation of
     * GPU-owned sequence state.  Both policies are economical backend kernels;
     * neither permits stage-level row replay.
     */
    enum class KVCacheAppendSemantics
    {
        Standard,                 ///< Normal prompt/decode cache publication.
        DecodeEquivalentVerifier ///< Grouped MTP rows with serial-decode semantics.
    };

    /**
     * @brief Explicit KV cache append stage
     *
     * Separates cache operations from attention computation, enabling:
     * - Pipelined execution: Append on one device while attending on another
     * - Explicit control: Manual cache management for advanced use cases
     * - Cross-device caches: Cache on GPU while computing on CPU
     *
     * VNNI-Safe Quantization (Q16_1 cache):
     * When the cache is Q16_1, this stage uses FIXED-SCALE quantization with
     * VNNI-safe clipping to prevent INT32 overflow during attention computation.
     * Set kv_cache_scale and head_dim to enable proper clipping limits.
     *
     * See: kernels/cpu/attention/q16_1/VNNISafetyConstants.h for clipping limits
     * See: docs/v2/projects/2025-12/PROJECT_Q16_INTEGER_ATTENTION_V2.md "VNNI OVERFLOW PREVENTION CONTRACT"
     */
    class KVCacheAppendStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const ITensor *K = nullptr; ///< Key to append
            const ITensor *V = nullptr; ///< Value to append
            IKVCache *kv_cache = nullptr;
            int layer_idx = 0;
            int seq_idx = 0;
            int num_tokens = 0;
            int batch_size = 1;
            int seq_len = 0;

            /**
             * @brief Mathematical cache-publication policy selected by the graph.
             *
             * MTP verifier graphs must set DecodeEquivalentVerifier explicitly.
             * The stage never guesses this policy from host cache metadata.
             */
            KVCacheAppendSemantics append_semantics =
                KVCacheAppendSemantics::Standard;

            /**
             * @brief Persistent device row containing one real length per request.
             *
             * A request-batched prefill graph has fixed `seq_len` geometry, but
             * each request may contain fewer real rows. The captured append and
             * sequence-advance kernels consume this persistent row directly;
             * there is no intermediate cache mailbox or pre-replay copy. This
             * pointer must remain valid for the lifetime of the captured graph.
             */
            const int32_t *request_sequence_lengths_device = nullptr;

            /// [Hybrid mode] Optional output for dequantized V (FP32)
            ITensor *V_dequant_out = nullptr;

            // =========================================================
            // VNNI-Safe Quantization Parameters (Q16_1 cache)
            // =========================================================

            /// Fixed scales for Q16_1 quantization (from GraphConfig).
            /// K and V use separate scales: K has large post-RoPE outliers,
            /// V values are much smaller. Separate scales maximize INT16 precision.
            float kv_cache_scale_k = 256.0f; ///< K scale (FP32 range ±scale_k)
            float kv_cache_scale_v = 32.0f;  ///< V scale (FP32 range ±scale_v)

            /// Attention head dimension (for VNNI clipping limits)
            /// Required for proper MAX_SAFE_INT16 selection. Common values: 64, 96, 128, 192.
            int head_dim = 128;

            /// TurboQuant context (rotation matrix) for TQ4 KV cache quantization.
            /// Required when cache precision is TQ4. Not owned by this struct.
            const TurboQuantContext *turboquant_ctx = nullptr;

            /// Block-diagonal orthogonal rotation for Q16_1 kurtosis reduction.
            /// When set, K and V are rotated before fixed-scale quantization.
            const ActivationRotation *kv_rotation = nullptr;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> k_buffer_id;
            std::optional<BufferId> v_buffer_id;
        };

        explicit KVCacheAppendStage(Params params);

        /**
         * @brief Return the immutable graph-declared append policy.
         *
         * This accessor is intended for graph construction tests and
         * diagnostics. Runtime mutation remains confined to the stage's
         * dynamic replay fields; callers must not cast away constness.
         */
        const Params &getParams() const { return params_; }

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::KV_CACHE_APPEND; }
        StageBufferContract bufferContract() const override;
        // KV cache append is graph-capturable when its kernels consume the
        // cache's canonical device head/count allocations directly.
        bool isGraphCapturable() const override
        {
            return params_.kv_cache && params_.kv_cache->isGraphCaptureReady();
        }
        bool hasDynamicParams() const override
        {
            return params_.kv_cache && params_.kv_cache->isGraphCaptureReady();
        }
        bool supportsDeviceResidentDynamicPositionReplay() const override
        {
            return true;
        }
        /**
         * @brief Bind the immutable device append-count source before capture.
         *
         * Setup-only graph materialization has no admitted request and therefore
         * does not execute the ordinary dynamic-parameter prelude.  The pointer
         * itself is nevertheless capture topology: padded prefill kernels must
         * embed the resident live-row scalar rather than a null exact-shape
         * source.  Values remain device-owned and are published by the captured
         * prefill materializer on every replay.
         *
         * @param ctx Device context selected for native graph capture.
         * @param stream Exact non-null capture stream.
         * @return true when every request-local cache row accepted its source.
         */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        /** @return CaptureOnly because the stable pointer is embedded once. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return params_.kv_cache && params_.kv_cache->isGraphCaptureReady()
                       ? GraphLaunchPreparationPolicy::CaptureOnly
                       : GraphLaunchPreparationPolicy::None;
        }
        /**
         * @brief Mark this append replay as consuming device-owned sequence state.
         *
         * KV append stages do not read absolute position IDs directly.  The
         * device-position replay hook is still meaningful for them because it
         * confirms that replay position and cache state have a common device
         * owner. Canonical GPU append no longer has a host-owned alternative,
         * so this hook intentionally performs no state transition.
         */
        void updateDynamicDevicePositionIds(const void *position_ids_device, int seq_len) override
        {
            (void)seq_len;
            (void)position_ids_device;
        }
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            (void)pos_offset;
            params_.seq_len = seq_len;
            if (params_.kv_cache && params_.kv_cache->isGraphCaptureReady())
            {
                // Bind the graph-shaped append to canonical device metadata.
                // Padded prefill buckets consume the persistent request-length
                // row directly; exact-shape graphs use their captured row count.
                void *stream = gpuStream();
                int append_tokens = params_.seq_len > 0 ? params_.seq_len : params_.num_tokens;
                if (params_.batch_size <= 1 && replay_advance_tokens_ > 0)
                {
                    append_tokens = replay_advance_tokens_;
                }
                const int bucket_tokens = params_.num_tokens > 0
                                              ? params_.num_tokens
                                              : append_tokens;
                if (append_tokens <= 0 ||
                    bucket_tokens <= 0 ||
                    append_tokens > bucket_tokens)
                {
                    LOG_ERROR("[KVCacheAppendStage] Invalid dynamic KV append token contract"
                              << " append_tokens=" << append_tokens
                              << " bucket_tokens=" << bucket_tokens
                              << " layer=" << params_.layer_idx
                              << " seq_idx=" << params_.seq_idx);
                    throw std::runtime_error("invalid dynamic KV append token contract");
                }
                const int request_count = std::max(1, params_.batch_size);
                const bool resident_request_lengths =
                    params_.request_sequence_lengths_device != nullptr;
                const int captured_request_tokens =
                    params_.seq_len > 0
                        ? params_.seq_len
                        : std::max(1, bucket_tokens / request_count);
                bool append_state_ready = true;
                for (int request = 0; request < request_count; ++request)
                {
                    const int seq_idx = params_.seq_idx + request;
                    const int32_t *append_count_source =
                        resident_request_lengths
                            ? params_.request_sequence_lengths_device + request
                            : nullptr;
                    const bool request_ready =
                        params_.kv_cache->bindGraphAppendCountSource(
                            params_.layer_idx,
                            seq_idx,
                            append_count_source,
                            captured_request_tokens,
                            stream);
                    append_state_ready = append_state_ready && request_ready;
                }
                if (!append_state_ready)
                {
                    LOG_ERROR("[KVCacheAppendStage] KV cache refused dynamic append state for graph replay"
                              << " resident_request_lengths="
                              << (resident_request_lengths ? "true" : "false")
                              << " layer=" << params_.layer_idx
                              << " first_seq_idx=" << params_.seq_idx
                              << " request_count=" << request_count
                              << " append_tokens=" << append_tokens);
                }
                if (debugEnv().attention.debug_kv_cache_snapshot &&
                    debugEnv().attention.debugKVCacheSnapshotLayerSelected(params_.layer_idx))
                {
                    const int cached_tokens =
                        params_.kv_cache->get_cached_tokens(params_.layer_idx, params_.seq_idx);
                    debug_cache_snapshot_rows_ =
                        static_cast<size_t>(
                            std::max(0, cached_tokens + append_tokens));
                    invalidateDumpInfoCache();
                }
            }
        }
        bool hasPrefillReplayParams() const override { return true; }
        void updatePrefillReplayParams(const PrefillReplayParams &replay) override
        {
            // Captured prefill append kernels may execute a padded bucket in a
            // later phase. Host cache metadata must advance by the real prompt
            // prefix only so padded rows remain invisible to attention/decode.
            replay_advance_tokens_ = replay.real_seq_len > 0 ? replay.real_seq_len : 0;
            const int bucket_tokens = params_.num_tokens > 0
                                          ? params_.num_tokens
                                          : (replay.bucket_seq_len > 0 ? replay.bucket_seq_len : 0);
            if (replay_advance_tokens_ > 0 &&
                bucket_tokens > 0 &&
                replay_advance_tokens_ > bucket_tokens)
            {
                LOG_ERROR("[KVCacheAppendStage] Prefill replay token count exceeds captured bucket"
                          << " real=" << replay_advance_tokens_
                          << " bucket=" << bucket_tokens
                          << " layer=" << params_.layer_idx
                          << " seq_idx=" << params_.seq_idx);
                throw std::runtime_error("prefill replay KV append token count exceeds bucket");
            }
            if (params_.kv_cache &&
                debugEnv().attention.debug_kv_cache_snapshot &&
                debugEnv().attention.debugKVCacheSnapshotLayerSelected(params_.layer_idx))
            {
                const int append_tokens =
                    replay_advance_tokens_ > 0 ? replay_advance_tokens_ : params_.num_tokens;
                const int cached_tokens =
                    params_.kv_cache->get_cached_tokens(params_.layer_idx, params_.seq_idx);
                debug_cache_snapshot_rows_ =
                    static_cast<size_t>(std::max(0, cached_tokens + append_tokens));
                invalidateDumpInfoCache();
            }
        }
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            replay_advance_tokens_ = 0;
            debug_append_source_k_rows_ = 0;
            debug_append_source_k_cols_ = 0;
            debug_append_source_v_rows_ = 0;
            debug_append_source_v_cols_ = 0;
            debug_cache_snapshot_rows_ = 0;
        }
        /**
         * @brief Clear host-side append bookkeeping for preserved graph replay.
         *
         * The captured append kernel reads stable tensor/cache pointers, while
         * replay_advance_tokens_ is restamped from PrefillReplayParams before
         * each launch. Normal request bookkeeping can therefore be cleared
         * without discarding the preserved prefill executable.
         */
        void resetSessionStatePreservingCapturedReplay() override
        {
            resetSessionState();
        }
        /**
         * @brief Keep append topology warm for capture-from-Initialized.
         */
        void resetSessionStatePreservingLazyInitialization() override
        {
            resetSessionState();
        }
        bool supportsBackend(ComputeBackendType backend) const override { return true; }
        StageBufferRequirements getBufferRequirements() const override;
        std::vector<BufferDescriptor> getDeclaredOutputs() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        bool producesVDequant() const { return params_.V_dequant_out != nullptr; }

    private:
        /**
         * @brief Bind resident append-count pointers for one graph geometry.
         *
         * @param stream Exact stream associated with the binding lifecycle.
         * @param runtime_seq_len Current physical sequence width, or zero to
         *        retain the immutable stage geometry.
         * @return true when all request rows accepted the binding.
         */
        bool bindCanonicalGraphAppendSources(
            void *stream,
            int runtime_seq_len);

        /**
         * @brief Validate and select grouped decode-equivalent publication.
         *
         * Grouped MTP verifier rows are mathematically decode rows, not prompt
         * prefill rows.  Cache publication must therefore use a dedicated
         * cache-level grouped contract that handles backend source layout and
         * updates ring metadata atomically.  Stage-level serial replay is not a
         * production implementation for Phase 9.8.
         */
        bool shouldUseDecodeEquivalentVerifierAppend(int request_rows) const;

        Params params_;
        std::unique_ptr<FP16Tensor> fp16_k_scratch_;
        std::unique_ptr<FP16Tensor> fp16_v_scratch_;
        std::unique_ptr<Q8_1Tensor> q8_k_scratch_;
        std::unique_ptr<Q8_1Tensor> q8_v_scratch_;
        std::shared_ptr<TQ4Tensor> tq4_k_scratch_;
        std::shared_ptr<TQ4Tensor> tq4_v_scratch_;
        std::shared_ptr<TQ8Tensor> tq8_k_scratch_; ///< TQ8 K for both public TurboQuant modes.
        std::shared_ptr<TQ8Tensor> tq8_v_scratch_; ///< TQ8 V for symmetric TQ8-K/TQ8-V.

        /// Workspace for kv_rotation: holds FP32 copy for in-place rotation
        /// before Q16_1 quantization. Lazy-allocated, reused across calls.
        std::vector<float> kv_rotation_scratch_;

        /// Debug-only source K/V metadata for tensor-backed graph snapshots.
        /// The actual bytes are captured by DeviceGraphExecutor's graph-stable
        /// D2D snapshot-copy path, never by host reads inside execute().
        size_t debug_append_source_k_rows_ = 0;
        size_t debug_append_source_k_cols_ = 0;
        size_t debug_append_source_v_rows_ = 0;
        size_t debug_append_source_v_cols_ = 0;
        size_t debug_cache_snapshot_rows_ = 0;

        /// Real token count to advance after prefill graph replay; 0 falls
        /// back to params_.num_tokens for decode and legacy exact-shape replay.
        int replay_advance_tokens_ = 0;

        /// One-shot marker set by device-resident replay preparation.  When
        /// true, updateDynamicParams() preserves GPU-owned cache head/count
        /// metadata and uploads only the append-count scalar.
    };

} // namespace llaminar2
