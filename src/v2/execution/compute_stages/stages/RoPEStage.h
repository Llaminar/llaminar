/**
 * @file RoPEStage.h
 * @brief Rotary position encoding stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include <vector>

namespace llaminar2
{

    /**
     * @brief Rotary position encoding stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses IActivationTensor::applyRoPE() for polymorphic device dispatch.
     */
    class RoPEStage : public IComputeStage
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

            // Per-token position IDs for batched execution (optional, overrides pos_offset)
            // When set, this array should have seq_len elements (total_tokens for batched)
            // For batched mode with batch_size=2, seq_len=2: [pos0_t0, pos0_t1, pos1_t0, pos1_t1]
            const int *position_ids = nullptr; ///< Per-token position IDs array [seq_len]

            // Fixed-scale quantization for HybridQ16 mode (used when Q_out is Q16_1)
            float kv_cache_scale = 256.0f; ///< Fixed scale for Q16 output (d = kv_cache_scale / 32767)

            // Dynamic-scale K output (for HybridQ16 K precision fix)
            // When K is Q16_1 from GEMM, RoPE outputs per-head scales for attention
            float *K_head_scales = nullptr; ///< Output: per-head K scales [seq_len * n_kv_heads]
        };

        explicit RoPEStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ROPE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /// Update position offset for cached graph reuse
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            params_.pos_offset = pos_offset;
            params_.seq_len = seq_len;
        }

        const Params &getParams() const { return params_; }

    private:
        Params params_;

        // Mutable cache for getDumpInfo() to store transposed Q16 output
        // HybridQ16 RoPE outputs HEAD_MAJOR layout, but snapshot comparison expects SEQ_MAJOR
        mutable std::vector<float> q_transposed_cache_;
        mutable std::vector<float> k_transposed_cache_;
    };

} // namespace llaminar2
