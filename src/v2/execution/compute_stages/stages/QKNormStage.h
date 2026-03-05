/**
 * @file QKNormStage.h
 * @brief Per-head QK RMSNorm stage for Qwen3
 *
 * Applies RMSNorm independently to each attention head within the Q and K
 * projections. This is used by Qwen3 which has separate attn_q_norm and
 * attn_k_norm weights of shape [head_dim].
 *
 * The stage reinterprets the Q/K tensor of shape [seq_len, n_heads * head_dim]
 * as [seq_len * n_heads, head_dim] and applies standard RMSNorm with gamma
 * of shape [head_dim], achieving per-head normalization without a separate kernel.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

namespace llaminar2
{
    // Forward declaration
    class ITensorRMSNorm;

    /**
     * @brief Per-head QK RMSNorm stage
     *
     * Normalizes Q or K projections per-head using RMSNorm with head_dim-sized gamma.
     * Operates in-place: output can be the same tensor as input.
     *
     * Dimension remapping:
     *   Input tensor:  [seq_len, n_heads * head_dim]
     *   Logical view:  [seq_len * n_heads, head_dim]
     *   Gamma weights: [head_dim]
     *
     * This reuses the existing RMSNorm kernel by passing adjusted dimensions.
     */
    class QKNormStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            /// Input Q or K tensor [seq_len, n_heads * head_dim]
            ITensor *input = nullptr;

            /// Output tensor (can be same as input for in-place)
            ITensor *output = nullptr;

            /// Gamma weights [head_dim]
            const ITensor *gamma = nullptr;

            /// Number of attention heads (Q heads or KV heads)
            int n_heads = 0;

            /// Dimension per head
            int head_dim = 0;

            /// Epsilon for numerical stability
            float eps = 1e-6f;

            /// Number of tokens (seq_len * batch_size)
            /// If 0, derives from input tensor dimensions
            int seq_len = 0;
        };

        static_assert(StageParamsRequired<Params>);

        explicit QKNormStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::QK_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;

        const Params &getParams() const { return params_; }

    private:
        Params params_;
        mutable ITensorRMSNorm *cached_kernel_ = nullptr;
        mutable int cached_kernel_tensor_type_ = -1;
    };

} // namespace llaminar2
