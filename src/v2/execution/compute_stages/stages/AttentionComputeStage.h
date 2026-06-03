/**
 * @file AttentionComputeStage.h
 * @brief Pure attention compute stage (no KV cache management)
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "kernels/IKVCache.h"
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

            // Workspace buffers
            ITensor *workspace_scores = nullptr;
            ITensor *workspace_context = nullptr;
            ITensor *workspace_mask = nullptr;

            // KV cache for dynamic length query at execution time
            IKVCache *kv_cache = nullptr;
            int layer_idx = -1;

            // When true, read K/V from kv_cache at execution time instead of
            // using the statically-wired K/V pointers. Enables GPU prefill to
            // use post-append FP16 cache tensors instead of Q8_1 projections.
            bool read_kv_from_cache = false;

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

            // RoPE-on-read: apply RoPE to K inside this attention stage.
            // When enabled, K in the KV cache is stored pre-RoPE, and position
            // embeddings are fused into the TQ4 dequant / applied in-place for FP32.
            bool apply_rope_to_k = false;
            float rope_theta = 10000.0f;        ///< RoPE frequency base (only used when apply_rope_to_k=true)
            float partial_rotary_factor = 1.0f; ///< Fraction of head dimensions to rotate (only used when apply_rope_to_k=true)
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
            if (params_.device_id.is_rocm() && params_.kv_cache && params_.layer_idx >= 0)
            {
                const bool decode_like =
                    params_.attention_mode == AttentionMode::DECODE ||
                    params_.attention_mode == AttentionMode::BATCHED_DECODE ||
                    (params_.auto_detect_mode && params_.seq_len <= 4);
                if (decode_like)
                {
                    // ROCm native decode attention currently chooses grid shape,
                    // split count, and block size from the host-side kv_len during
                    // capture. Device-side params update the logical values for
                    // replay, but cannot change the captured launch topology as
                    // the KV cache grows.
                    return false;
                }
            }

            // TQ KV cache dequant runs inside execute() via get_kv_converted() with
            // iteration-varying arguments (ring_pos, out_offset grow each step).
            // With device-side dynamic params, the captured H2D + kernel can be
            // replayed with updated values from pinned host memory.
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

        /// Update position offset for cached graph reuse.
        /// Also updates the kernel's pinned host device-params so the next
        /// graph replay picks up the new kv_len and position_offset via
        /// the captured H2D memcpy.
        /// Note: updateDynamicParams is called BEFORE KVCacheAppend runs for
        /// this step, so get_cached_tokens() returns previous step's count.
        /// We add seq_len to get the count after appending.
        bool hasDynamicParams() const override { return true; }
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            params_.position_offset = pos_offset;
            if (cached_kernel_ && params_.kv_cache && params_.layer_idx >= 0)
            {
                // Propagate current stage stream to kernel so setDynamicAttnParams
                // can issue H2D copies on the correct (capture/replay) stream.
                cached_kernel_->setGPUStream(gpuStream());
                int kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
                kv_len += seq_len; // This step will append seq_len tokens
                const int logical_pos_offset = std::max(0, kv_len - seq_len);
                cached_kernel_->setDynamicAttnParams(kv_len, logical_pos_offset, seq_len);

                // TQ dequant: update pinned host params for captured H2D.
                // setDynamicDequantParams computes ring_pos, out_offset, rope_position
                // from the pre-append entry state. On graph replay, the captured
                // H2D re-reads from pinned host, giving the kernel updated values.
                // position_start=0 matches execute() which always passes 0 — cache
                // rows are stored in position order, so position = entry.count.
                const float dequant_rope_theta =
                    params_.apply_rope_to_k ? params_.rope_theta : 0.0f;
                params_.kv_cache->setDynamicDequantParams(
                    params_.layer_idx, 0, dequant_rope_theta,
                    0, gpuStream());
            }
        }

        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            params_.position_offset = 0;
            debug_effective_k_snapshot_.clear();
            debug_effective_v_snapshot_.clear();
            debug_effective_k_rows_ = 0;
            debug_effective_k_cols_ = 0;
            debug_effective_v_rows_ = 0;
            debug_effective_v_cols_ = 0;
            if (cached_kernel_)
            {
                cached_kernel_->resetDynamicState();
                cached_kernel_->setGPUStream(nullptr);
            }
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

        /// Cached attention kernel for workspace binding
        ITensorAttention *cached_kernel_ = nullptr;
        int cached_kernel_tensor_type_ = -1;

        /// Debug-only FP32 copies of the effective K/V tensors passed to the
        /// attention kernel. Populated only when
        /// LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT is enabled.
        mutable std::vector<float> debug_effective_k_snapshot_;
        mutable std::vector<float> debug_effective_v_snapshot_;
        mutable size_t debug_effective_k_rows_ = 0;
        mutable size_t debug_effective_k_cols_ = 0;
        mutable size_t debug_effective_v_rows_ = 0;
        mutable size_t debug_effective_v_cols_ = 0;

        /**
         * @brief Get or create the attention kernel
         * @return Pointer to cached kernel, or nullptr on failure
         */
        ITensorAttention *getOrCreateKernel();
    };

} // namespace llaminar2
