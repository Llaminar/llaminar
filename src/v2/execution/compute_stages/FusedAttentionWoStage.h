/**
 * @file FusedAttentionWoStage.h
 * @brief Fused attention + Wo projection stage
 */

#pragma once

#include "IComputeStage.h"
#include <memory>

namespace llaminar2
{
    // Forward declarations
    class ITensorFusedAttentionWo;

    /**
     * @brief Fused attention + Wo projection stage
     *
     * Combines attention computation and output projection (Wo) into a single
     * kernel for improved cache locality and reduced memory traffic.
     *
     * Benefits over separate AttentionComputeStage + GEMMStage:
     * - Eliminates attention context quantization round-trip (FP32 → Q8_1 → FP32)
     * - Context stays in registers through Wo projection
     * - Single pass over V and Wo weights
     *
     * Uses FusedAttentionWoKernel internally, which supports:
     * - JIT backend: AVX-512 VNNI optimized (default, fastest)
     * - TILED backend: Cache-blocked for correctness/fallback
     * - REFERENCE backend: Pure C++ for testing
     * - Q16_INTEGER backend: Pure integer Q16_1 via ITensorFusedAttentionWo
     */
    class FusedAttentionWoStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input tensors (Q8_1 format)
            TensorBase *Q = nullptr;
            TensorBase *K = nullptr;
            TensorBase *V = nullptr;

            // Weight tensor
            TensorBase *Wo = nullptr;

            // Output tensor
            TensorBase *output = nullptr;

            // Dimensions
            int batch_size = 1;
            int seq_len = 0;
            int kv_len = 0;
            int n_heads = 0;
            int n_kv_heads = 0;
            int head_dim = 0;
            int d_model = 0;

            // Attention configuration
            bool causal = true;
            int position_offset = 0;

            // Backend selection
            FusedAttentionBackend backend = FusedAttentionBackend::JIT;

            // KV cache for dynamic kv_len query
            IUnifiedKVCache *kv_cache = nullptr;
            int layer_idx = -1;

            // Optional MPI context
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;

            // Optional context snapshot buffer for debugging
            TensorBase *context_snapshot = nullptr;

            // Hybrid mode: use streaming dequantization for Wo projection
            bool use_hybrid_wo = false;

            // Q16_1 residual fusion (HybridQ16 mode)
            bool fuse_residual_add = false;
        };

        explicit FusedAttentionWoStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::FUSED_ATTENTION_WO; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            params_.position_offset = pos_offset;
            (void)seq_len;
        }

        const Params &getParams() const { return params_; }

    private:
        Params params_;
        std::unique_ptr<FusedAttentionWoKernel> kernel_;      ///< Q8_1 kernel (JIT/TILED/REFERENCE)
        std::unique_ptr<ITensorFusedAttentionWo> q16_kernel_; ///< Q16_1 kernel (Q16_INTEGER)
    };

} // namespace llaminar2
