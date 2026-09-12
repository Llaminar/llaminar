/**
 * @file AttentionComputeStage.h
 * @brief Participant-local attention over explicit prepared cache read storage.
 *
 * Append remains a separate graph stage. Cache-owned read destinations are
 * described before capture, while device-owned head/count values determine
 * which rows are consumed at replay. Diagnostic views borrow those immutable
 * destinations and never trigger a cache read merely to discover its address.
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "kernels/IKVCache.h"
#include "../../../kernels/attention/AttentionExecutionPolicy.h"
#include "../../../memory/BufferId.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

namespace llaminar2
{
    // Forward declarations
    class ITensorAttention;
    class TurboQuantContext;

    /**
     * @brief Pure attention compute stage (no KV cache management)
     *
     * Delegates to KernelFactory::createAttention() for device-appropriate kernel
     * selection. This stage handles ONLY the attention computation:
     *   output = softmax(Q @ K^T / sqrt(head_dim) + mask) @ V
     *
     * For KV cache management, use KVCacheAppendStage separately in the DAG.
     *
     * **Workspace Management (ROCm GPU)**:
     * Implements IWorkspaceConsumerStage to delegate workspace requirements to the
     * underlying attention kernel. This enables zero-allocation GPU execution by
     * pre-binding workspace buffers during graph setup.
     */
    class AttentionComputeStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input/output tensors
            ITensor *Q = nullptr;
            ITensor *K = nullptr;
            ITensor *V = nullptr;
            ITensor *output = nullptr;

            // Dimensions
            int batch_size = 1;
            int seq_len = 0;
            int kv_len = 0;
            int n_heads = 0;
            int n_kv_heads = 0;
            int head_dim = 0;

            // Tensor-parallel GQA mapping (replicated KV heads)
            int head_start = 0; ///< Global Q head offset for this device
            int gqa_n_rep = 0;  ///< Global GQA repetition factor (n_heads_global / n_kv_heads_global, 0 = auto)

            // Attention configuration
            bool causal = true;
            int window_size = -1;

            /// Execution mode
            AttentionMode attention_mode = AttentionMode::PREFILL;
            bool auto_detect_mode = true;

            /**
             * @brief Complete declarative attention policy for this graph node.
             *
             * The model graph declares the permitted logical partition axis and
             * the physical encoding of persistent K cache bytes. Backend
             * machinery resolves `GeometrySelected` from immutable capture
             * geometry and must preserve that concrete topology across replay.
             * Live KV length is not part of this policy.
             */
            attention::AttentionExecutionPolicy execution_policy{};

            // Workspace buffers
            ITensor *workspace_scores = nullptr;
            ITensor *workspace_context = nullptr;
            ITensor *workspace_mask = nullptr;

            /**
             * @brief Sole post-append K/V source when non-null, on every backend.
             *
             * Cache-backed graphs order append before attention. A missing
             * cache selects the explicit K/V tensors instead; request length
             * and phase never change that source or its storage precision.
             */
            IKVCache *kv_cache = nullptr;
            int layer_idx = -1;

            /**
             * @brief Device-owned logical query width for a scalar GPU graph.
             *
             * A fixed-width grouped-verifier graph can launch more physical
             * rows than the current MTP transaction owns. Attention derives
             * serial-equivalent per-row KV horizons from this stable scalar;
             * no host length participates in capture or replay.
             */
            const int32_t *active_query_rows_device = nullptr;

            // Position offset for decode mode causal masking
            int position_offset = 0;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> q_buffer_id;
            std::optional<BufferId> output_buffer_id;
            std::optional<BufferId> workspace_scores_buffer_id;
            std::optional<BufferId> workspace_context_buffer_id;

            /// TurboQuant context for TQ4 KV cache dequantization
            const TurboQuantContext *turboquant_ctx = nullptr;

            /// Block-diagonal orthogonal rotation for Q16_1 kurtosis reduction.
            /// When set, Q is rotated before the dot product and the attention
            /// output is inverse-rotated after weighted-V accumulation.
            const ActivationRotation *kv_rotation = nullptr;

        };

        explicit AttentionComputeStage(Params params);

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override
        {
            // Device-owned native or transformed KV views consume replay-varying
            // ring geometry through persistent device parameters. Capture records
            // kernels only; no host conversion shadow participates.
            if (params_.kv_cache)
            {
                const auto kp = params_.kv_cache->k_precision();
                if (kp == ActivationPrecision::TQ4 || kp == ActivationPrecision::TQ8)
                    return params_.kv_cache->isGraphCaptureReady();
            }
            return true; // Device-side params buffer handles dynamic kv_len/position
        }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        /// Target device for coherence management

        /**
         * @brief Update request-local attention sequence metadata.
         *
         * The forward DAG runs KVCacheAppendStage immediately before this stage.
         * updateDynamicParams() is called before that append, so the KV cache's
         * host count is the pre-append history length.  This method records that
         * count as request-local stage state and uploads any host-owned dynamic
         * attention parameters needed by the next eager execution or graph replay.
         *
         * GPU graph-capturable KV caches expose a device-owned cached-token count.
         * For those caches, attention records a tiny in-graph derivation kernel
         * instead of relying on a host scalar.  That keeps one reusable prefill
         * graph valid whether it is captured for the first prompt chunk or later
         * replayed as a suffix after a prefix-cache restore.
         */
        bool hasDynamicParams() const override { return true; }
        bool supportsDeviceResidentDynamicPositionReplay() const override;
        void updateDynamicParams(int pos_offset, int seq_len) override;
        bool hasPrefillReplayParams() const override { return params_.kv_cache != nullptr; }
        void updatePrefillReplayParams(const PrefillReplayParams &replay) override;

        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            params_.position_offset = 0;
            prefill_replay_params_set_ = false;
            prefill_effective_seq_len_ = 0;
            prefill_bucket_seq_len_ = 0;
            dynamic_pre_append_cached_tokens_ = -1;
            dynamic_logical_seq_len_ = 0;
            dynamic_post_append_kv_len_ = 0;
            resetEffectiveKVDumpData();
            if (cached_kernel_)
            {
                cached_kernel_->resetDynamicState();
                cached_kernel_->clearGPUStreamBinding();
            }
        }

        /**
         * @brief Clear request mirrors while preserving captured attention params.
         *
         * Bucketed prefill replay updates the attention device-param row before
         * launch. A preserved CUDA/HIP graph still owns that row by address, so
         * request reset must not call resetDynamicState() on the kernel when the
         * graph executable is intentionally kept hot.
         */
        void resetSessionStatePreservingCapturedReplay() override
        {
            IComputeStage::resetSessionState();
            params_.position_offset = 0;
            prefill_replay_params_set_ = false;
            prefill_effective_seq_len_ = 0;
            prefill_bucket_seq_len_ = 0;
            dynamic_pre_append_cached_tokens_ = -1;
            dynamic_logical_seq_len_ = 0;
            dynamic_post_append_kv_len_ = 0;
            // A request resets payload state, not captured source addresses.
            resetEffectiveKVDumpData();
            if (cached_kernel_)
                cached_kernel_->clearGPUStreamBinding();
        }

        /**
         * @brief Keep warmed attention workspace/device-param storage for capture.
         */
        void resetSessionStatePreservingLazyInitialization() override
        {
            resetSessionStatePreservingCapturedReplay();
        }

        const Params &getParams() const { return params_; }

        // =================================================================
        // IWorkspaceConsumerStage Implementation
        // =================================================================

        /**
         * @brief Get the attention kernel as IWorkspaceConsumer for delegation
         *
         * Returns cached kernel (creates on first call). The same kernel is
         * returned on every call for this stage, enabling workspace binding.
         *
         * @return Kernel implementing IWorkspaceConsumer, or nullptr if not available
         */
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;

    private:
        Params params_;

        /** @return The one physical read policy used by execution and diagnostics. */
        IKVCache::DeviceReadStorageKind deviceCacheReadKind() const;

        /**
         * @brief Bind immutable GPU diagnostic views before snapshot sizing.
         * @throws std::runtime_error for missing storage or changed identity.
         * This performs no GPU work and never observes a live cache cursor.
         */
        void prepareEffectiveKVDumpStorage() const;

        /** @brief Reset CPU observations while retaining GPU topology metadata. */
        void resetEffectiveKVDumpData()
        {
            if (params_.device_id.is_gpu())
                return;
            debug_effective_k_tensor_ = nullptr;
            debug_effective_v_tensor_ = nullptr;
            debug_effective_k_rows_ = debug_effective_k_cols_ = 0;
            debug_effective_v_rows_ = debug_effective_v_cols_ = 0;
        }

        /// Cached attention kernel for workspace binding
        ITensorAttention *cached_kernel_ = nullptr;
        int cached_kernel_tensor_type_ = -1;

        /// Stage-owned non-owning views retain descriptor identity even when a
        /// different graph uses another request-count view of shared scratch.
        mutable std::unique_ptr<ITensor> debug_effective_k_view_;
        mutable std::unique_ptr<ITensor> debug_effective_v_view_;

        /// Debug-only effective K/V tensor metadata for graph snapshots. The
        /// executor performs graph-captured D2D snapshot copies and post-graph
        /// host materialization; attention execute() never performs D2H reads.
        mutable const ITensor *debug_effective_k_tensor_ = nullptr;
        mutable const ITensor *debug_effective_v_tensor_ = nullptr;
        mutable size_t debug_effective_k_rows_ = 0;
        mutable size_t debug_effective_k_cols_ = 0;
        mutable size_t debug_effective_v_rows_ = 0;
        mutable size_t debug_effective_v_cols_ = 0;

        /**
         * @brief Graph-stable non-owning views of canonical GPU KV metadata.
         *
         * These views are constructed with the stage, never during execute().
         * When effective-KV diagnostics are enabled, SnapshotCapture records
         * D2D copies of each request's count and ring head on the graph stream.
         * They make append-state ordering failures observable without a device
         * synchronization or a host-owned coherence mirror.
         */
        std::vector<std::unique_ptr<ITensor>> debug_device_kv_count_views_;
        std::vector<std::unique_ptr<ITensor>> debug_device_kv_head_views_;
        std::vector<std::string> debug_device_kv_count_names_;
        std::vector<std::string> debug_device_kv_head_names_;

        /// Real-token metadata for fixed-bucket prefill graph replay. The graph
        /// launch remains bucket-shaped, but dynamic attention/KV metadata must
        /// expose only rows that are real prompt tokens.
        bool prefill_replay_params_set_ = false;
        int prefill_effective_seq_len_ = 0;
        int prefill_bucket_seq_len_ = 0;

        /**
         * @brief Request geometry for the current dynamic stage pass.
         *
         * The admitted request cursor determines fixed host launch geometry; it
         * is not a cache-state mirror. GPU attention derives every live row
         * length from canonical device metadata after the captured append.
         */
        int dynamic_pre_append_cached_tokens_ = -1;
        int dynamic_logical_seq_len_ = 0;
        int dynamic_post_append_kv_len_ = 0;

        /**
         * @brief Stable per-request CPU KV descriptors for grouped decode.
         *
         * Request-batched CPU attention cannot flatten independent cache slots
         * into one scalar sequence. These vectors are sized with the graph and
         * reused on every execution so the grouped production path performs no
         * hot-loop descriptor allocation.
         */
        std::vector<const ITensor *> cpu_grouped_k_views_;
        std::vector<const ITensor *> cpu_grouped_v_views_;
        std::vector<int> cpu_grouped_kv_lens_;
        std::vector<attention::AttentionKVLogicalView>
            cpu_grouped_kv_logical_views_;

        /**
         * @brief Get or create the attention kernel
         * @return Pointer to cached kernel, or nullptr on failure
         */
        ITensorAttention *getOrCreateKernel();

        /**
         * @brief Number of row-local dynamic attention params needed.
         *
         * Multi-row MTP verifier decode can run as several row-local decode
         * kernels. The host-prepared and device-derived metadata paths must use
         * exactly the same row count or graph replay will read mismatched
         * `AttentionDeviceParams`.
         */
        int dynamicAttentionParamRows(int logical_seq_len, int kv_len) const;

    };

} // namespace llaminar2
