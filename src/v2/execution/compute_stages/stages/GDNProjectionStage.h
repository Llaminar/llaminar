/**
 * @file GDNProjectionStage.h
 * @brief GDN projection stage: 4 separate GEMMs for Gated Delta Net layers
 *
 * Computes 4 independent linear projections from the same input:
 *   - in_proj_qkv: hidden → mixed QKV [seq_len, qkv_dim]
 *   - in_proj_z:   hidden → gate Z      [seq_len, n_heads * d_v]
 *   - in_proj_a:   hidden → alpha A      [seq_len, n_heads]
 *   - in_proj_b:   hidden → beta B       [seq_len, n_heads]
 *
 * All projections share the same input tensor (activation).
 * Used by Qwen 3.5 GDN (Gated Delta Network) layers.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    class ITensorGemm;

    /**
     * @brief 4-projection stage for GDN layers
     *
     * Performs: QKV = input × W_qkv, Z = input × W_z,
     *          A = input × W_a, B = input × W_b
     */
    class GDNProjectionStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const ITensor *input = nullptr; ///< Input activation [seq_len, d_model]
            int m = 0;                      ///< seq_len (rows of input)
            int k = 0;                      ///< d_model (cols of input)

            // QKV projection
            const ITensor *w_qkv = nullptr; ///< Weight [d_model, qkv_dim]
            ITensor *output_qkv = nullptr;  ///< Output [seq_len, qkv_dim]
            int n_qkv = 0;                  ///< qkv_dim = 2*n_heads*d_k + n_heads*d_v

            // Z (gate) projection
            const ITensor *w_z = nullptr; ///< Weight [d_model, n_heads * d_v]
            ITensor *output_z = nullptr;  ///< Output [seq_len, n_heads * d_v]
            int n_z = 0;                  ///< n_heads * d_v

            // A (alpha / dt) projection
            const ITensor *w_a = nullptr; ///< Weight [d_model, n_heads]
            ITensor *output_a = nullptr;  ///< Output [seq_len, n_heads]
            int n_a = 0;                  ///< n_heads

            // B (beta) projection
            const ITensor *w_b = nullptr; ///< Weight [d_model, n_heads]
            ITensor *output_b = nullptr;  ///< Output [seq_len, n_heads]
            int n_b = 0;                  ///< n_heads

            // Cached GEMM kernels (set during graph construction)
            ITensorGemm *gemm_qkv = nullptr;
            ITensorGemm *gemm_z = nullptr;
            ITensorGemm *gemm_a = nullptr;
            ITensorGemm *gemm_b = nullptr;

            // Optional BufferIds
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_qkv_buffer_id;
            std::optional<BufferId> output_z_buffer_id;
            std::optional<BufferId> output_a_buffer_id;
            std::optional<BufferId> output_b_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit GDNProjectionStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GDN_PROJECTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        const Params &getParams() const { return params_; }

    private:
        /// Lazily resolve a GEMM kernel from a weight tensor via KernelFactory.
        /// Caches the result in @p cached for subsequent calls (like GEMMStage).
        ITensorGemm *resolveGemm(
            const ITensor *weight, ITensorGemm *&cached, const char *name);

        Params params_;
    };

} // namespace llaminar2
