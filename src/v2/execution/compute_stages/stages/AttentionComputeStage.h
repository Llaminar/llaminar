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

#include <memory>
#include <optional>

namespace llaminar2
{
    // Forward declarations
    class ITensorAttention;
    class FP32Tensor;

    /**
     * @brief Pure attention compute stage (no KV cache management)
     *
     * Delegates to KernelFactory::createAttention() for device-appropriate kernel
     * selection. This stage handles ONLY the attention computation:
     *   output = softmax(Q @ K^T / sqrt(head_dim) + mask) @ V
     *
     * For KV cache management, use KVCacheAppendStage separately in the DAG.
     * For integrated cache+attention, use AttentionWithKVCacheStage.
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
        };

        explicit AttentionComputeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return true; } // Device-side params buffer handles dynamic kv_len/position
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
                cached_kernel_->setDynamicAttnParams(kv_len, pos_offset);
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

        /// Persistent FP32 decode buffers for incremental FP16→FP32 conversion.
        /// Avoids re-converting the entire KV cache on every decode step.
        std::unique_ptr<FP32Tensor> decode_k_fp32_;
        std::unique_ptr<FP32Tensor> decode_v_fp32_;
        int decode_kv_fp32_rows_ = 0;

        /**
         * @brief Get or create the attention kernel
         * @return Pointer to cached kernel, or nullptr on failure
         */
        ITensorAttention *getOrCreateKernel();
    };

} // namespace llaminar2
