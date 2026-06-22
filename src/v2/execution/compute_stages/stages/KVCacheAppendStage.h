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

#include <optional>
#include <memory>

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

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::KV_CACHE_APPEND; }
        StageBufferContract bufferContract() const override;
        // KV cache append is graph-capturable when the KV cache supports
        // device-side head parameters. updateDynamicParams() uploads the head
        // to a stable device scalar before capture/replay; captured append
        // records only the dynamic kernel that reads that scalar.
        bool isGraphCapturable() const override
        {
            return params_.kv_cache && params_.kv_cache->isGraphCaptureReady();
        }
        bool hasDynamicParams() const override
        {
            return params_.kv_cache && params_.kv_cache->supportsDynamicAppendState();
        }
        bool supportsDeviceResidentDynamicPositionReplay() const override
        {
            return true;
        }
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            (void)pos_offset;
            params_.seq_len = seq_len;
            if (params_.kv_cache && params_.kv_cache->supportsDynamicAppendState())
            {
                // Upload current ring state and the real append length to
                // graph-owned device scalars. Padded prefill buckets keep a
                // bucket-shaped kernel for reuse, but sequence metadata must
                // advance by the real prompt length so padding never becomes
                // visible cache state.
                void *stream = gpuStream();
                int append_tokens = params_.seq_len > 0 ? params_.seq_len : params_.num_tokens;
                if (params_.batch_size <= 1 && replay_advance_tokens_ > 0)
                {
                    append_tokens = replay_advance_tokens_;
                }
                if (!params_.kv_cache->setDynamicAppendState(
                        params_.layer_idx, params_.seq_idx, append_tokens, stream))
                {
                    LOG_ERROR("[KVCacheAppendStage] KV cache refused dynamic append state for graph replay");
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
        }
        void onGraphReplayed() override
        {
            // Advance the ring buffer head and count on the host side.
            // Called by DeviceGraphExecutor AFTER the captured graph segment replays.
            if (params_.kv_cache)
            {
                const int advance_tokens = replay_advance_tokens_ > 0
                                               ? replay_advance_tokens_
                                               : params_.num_tokens;
                params_.kv_cache->advanceHead(params_.layer_idx, params_.seq_idx, advance_tokens);
            }
        }
        bool needsOnGraphReplayed() const override { return true; }
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            replay_advance_tokens_ = 0;
            debug_append_source_k_snapshot_.clear();
            debug_append_source_v_snapshot_.clear();
            debug_append_source_k_rows_ = 0;
            debug_append_source_k_cols_ = 0;
            debug_append_source_v_rows_ = 0;
            debug_append_source_v_cols_ = 0;
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
        const Params &getParams() const { return params_; }

    private:
        /**
         * @brief Returns true when a tiny verifier append needs grouped
         * decode-equivalent cache publication.
         *
         * Grouped MTP verifier rows are mathematically decode rows, not prompt
         * prefill rows.  Cache publication must therefore use a dedicated
         * cache-level grouped contract that handles backend source layout and
         * updates ring metadata atomically.  Stage-level serial replay is not a
         * production implementation for Phase 9.8.
         */
        bool shouldUseDecodeEquivalentVerifierAppend(int total_tokens,
                                                     int batch_size,
                                                     int seq_len) const;

        Params params_;
        std::unique_ptr<FP16Tensor> fp16_k_scratch_;
        std::unique_ptr<FP16Tensor> fp16_v_scratch_;
        std::unique_ptr<Q8_1Tensor> q8_k_scratch_;
        std::unique_ptr<Q8_1Tensor> q8_v_scratch_;
        std::shared_ptr<TQ4Tensor> tq4_k_scratch_;
        std::shared_ptr<TQ4Tensor> tq4_v_scratch_;
        std::shared_ptr<TQ8Tensor> tq8_k_scratch_; ///< For split TQ (TQ8 K + TQ4 V)

        /// Workspace for kv_rotation: holds FP32 copy for in-place rotation
        /// before Q16_1 quantization. Lazy-allocated, reused across calls.
        std::vector<float> kv_rotation_scratch_;

        /// Debug-only post-append KV snapshots. Populated only when
        /// LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT is enabled, so normal dump/coherence
        /// paths do not copy persistent cache state back to host.
        mutable std::vector<float> debug_cache_k_snapshot_;
        mutable std::vector<float> debug_cache_v_snapshot_;

        /// Debug-only source K/V copies captured immediately before append.
        /// This avoids reading persistent cache views while still proving what
        /// each request wrote into the cache append path.
        std::vector<float> debug_append_source_k_snapshot_;
        std::vector<float> debug_append_source_v_snapshot_;
        size_t debug_append_source_k_rows_ = 0;
        size_t debug_append_source_k_cols_ = 0;
        size_t debug_append_source_v_rows_ = 0;
        size_t debug_append_source_v_cols_ = 0;

        /// Real token count to advance after prefill graph replay; 0 falls
        /// back to params_.num_tokens for decode and legacy exact-shape replay.
        int replay_advance_tokens_ = 0;
    };

} // namespace llaminar2
