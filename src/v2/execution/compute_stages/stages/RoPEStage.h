/**
 * @file RoPEStage.h
 * @brief Rotary position encoding stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>
#include <vector>
#include <memory>

namespace llaminar2
{
    // Forward declarations
    class ITensorRoPE;

    /**
     * @brief Rotary position encoding stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses IActivationTensor::applyRoPE() for polymorphic device dispatch.
     */
    class RoPEStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Type-safe tensor pointers (required)
            ITensor *Q = nullptr; ///< Query tensor (IActivationTensor*, modified in-place)
            ITensor *K = nullptr; ///< Key tensor (IActivationTensor*, modified in-place, optional)

            // Hybrid mode output buffers (optional - when set, output goes here instead of in-place)
            ITensor *Q_out = nullptr; ///< FP32 output for Q after RoPE (Hybrid mode)
            ITensor *K_out = nullptr; ///< FP32 output for K after RoPE (Hybrid mode)

            // Configuration
            int n_heads = 0;             ///< Number of query heads
            int n_kv_heads = 0;          ///< Number of KV heads (for GQA)
            int head_dim = 0;            ///< Dimension per head
            int pos_offset = 0;          ///< Position offset (for KV cache) - DEPRECATED: use position_ids
            float theta_base = 10000.0f; ///< RoPE theta base
            int seq_len = 0;             ///< Explicit sequence length (for pre-allocated buffers)

            /// Partial rotary factor: fraction of head_dim that gets rotation.
            /// 1.0 = full RoPE (default), 0.5 = first half rotated, second half pass-through.
            /// Used by models like Qwen 3.5 GDN attention layers.
            float partial_rotary_factor = 1.0f;

            // Per-token position IDs for batched execution (optional, overrides pos_offset)
            // When set, this array should have seq_len elements (total_tokens for batched)
            // For batched mode with batch_size=2, seq_len=2: [pos0_t0, pos0_t1, pos1_t0, pos1_t1]
            const int *position_ids = nullptr; ///< Per-token position IDs array [seq_len]

            // Fixed-scale quantization for HybridQ16 mode (used when Q_out is Q16_1)
            // RoPE only processes K projections, so this is the K scale.
            float kv_cache_scale_k = 256.0f; ///< Fixed scale for K Q16 output (d = scale / 32767)

            // Dynamic-scale K output (for HybridQ16 K precision fix)
            // When K is Q16_1 from GEMM, RoPE outputs per-head scales for attention
            float *K_head_scales = nullptr; ///< Output: per-head K scales [seq_len * n_kv_heads]

            // RoPE-on-read mode: skip applying RoPE to K in this stage.
            // When true, only Q gets RoPE applied. K will be stored pre-RoPE in the
            // KV cache and RoPE will be fused into the attention dequant path.
            bool skip_k = false;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> q_buffer_id;
            std::optional<BufferId> k_buffer_id;
            std::optional<BufferId> q_out_buffer_id;
            std::optional<BufferId> k_out_buffer_id;
        };

        explicit RoPEStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ROPE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return true; }
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        bool needsGraphLaunchPreparation() const override { return params_.device_id.is_gpu(); }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        /// Update position offset for cached graph reuse.
        /// Also pre-uploads the kernel's device params on the explicit stage
        /// stream so captured RoPE execution records kernels only.
        bool hasDynamicParams() const override { return true; }
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            params_.pos_offset = pos_offset;
            params_.seq_len = seq_len;
            params_.position_ids = nullptr;
            position_ids_cache_.clear();
            if (cached_kernel_)
            {
                // Propagate current stage stream so setDynamicPosOffset can
                // pre-upload device params before capture/replay.
                cached_kernel_->setGPUStream(gpuStream());
                cached_kernel_->setDynamicPosOffset(pos_offset);
            }
        }

        /**
         * @brief Update explicit per-row position IDs for cached graph reuse.
         *
         * Request-batched MTP sidecars flatten multiple logical requests into
         * one tiny graph.  Rows can therefore share the same absolute position
         * (for example `[595, 595]` for two requests), which is not equivalent
         * to the contiguous scalar range `[595, 596]`.  This method keeps the
         * stable host copy alive for CPU/eager execution and asks GPU kernels
         * to pre-upload the workspace-owned device copy before graph capture.
         */
        void updateDynamicPositionIds(const int *position_ids, int seq_len)
        {
            params_.seq_len = seq_len;
            position_ids_cache_.clear();
            if (position_ids && seq_len > 0)
            {
                position_ids_cache_.assign(position_ids, position_ids + seq_len);
                params_.position_ids = position_ids_cache_.data();
                params_.pos_offset = position_ids_cache_.front();
            }
            else
            {
                params_.position_ids = nullptr;
                params_.pos_offset = 0;
            }

            if (cached_kernel_)
            {
                cached_kernel_->setGPUStream(gpuStream());
                cached_kernel_->setDynamicPositionIds(params_.position_ids, seq_len);
            }
        }

        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            params_.pos_offset = 0;
            params_.seq_len = 0;
            params_.position_ids = nullptr;
            position_ids_cache_.clear();
            if (cached_kernel_)
            {
                cached_kernel_->resetDynamicState();
                cached_kernel_->setGPUStream(nullptr);
            }
        }

        const Params &getParams() const { return params_; }

        // IWorkspaceConsumerStage implementation
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;

    private:
        ITensorRoPE *getOrCreateStageKernel(TensorBase *Q_base);

        Params params_;

        // Stage-owned RoPE kernel.  Graph-captured dynamic state such as the
        // active stream, workspace pointer, scalar pos_offset, and explicit
        // position-id upload validity is stage-local and must not be shared
        // with other RoPE stages on the same device.
        mutable std::unique_ptr<ITensorRoPE> owned_kernel_;
        mutable ITensorRoPE *cached_kernel_ = nullptr;
        mutable int cached_kernel_tensor_type_ = -1;

        // Mutable cache for getDumpInfo() to store transposed Q16 output
        // HybridQ16 RoPE outputs HEAD_MAJOR layout, but snapshot comparison expects SEQ_MAJOR
        mutable std::vector<float> q_transposed_cache_;
        mutable std::vector<float> k_transposed_cache_;

        // Stable host-side position_ids for graph capture (copied to device each step).
        mutable std::vector<int> position_ids_cache_;
    };

} // namespace llaminar2
