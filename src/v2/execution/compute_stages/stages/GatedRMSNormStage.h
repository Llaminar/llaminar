/**
 * @file GatedRMSNormStage.h
 * @brief Gated RMS normalization stage for models with multiplicative gating
 *
 * Computes: output = RMSNorm(input, gamma) * gate
 *
 * The gating tensor is a learned parameter (e.g., from a linear projection)
 * that modulates the normalized output element-wise. Used by architectures
 * like DeepSeek and Qwen 3.5 GDN layers.
 *
 * Supports optional `subtract_one` weight mode where the stored gamma
 * represents (gamma_effective - 1.0), so the effective gamma is (1.0 + gamma_stored).
 * This convention is used by DeepSeek and some Qwen variants.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    /**
     * @brief Gated RMS normalization
     *
     * output[i] = RMSNorm(input, gamma)[i] * gate[i]
     *
     * When subtract_one=true, gamma_effective = 1.0 + gamma_stored.
     */
    class GatedRMSNormStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *input = nullptr;       ///< Input activation [seq_len, d_model]
            ITensor *gate = nullptr;        ///< Gating tensor [seq_len, d_model]
            ITensor *output = nullptr;      ///< Output [seq_len, d_model]
            const ITensor *gamma = nullptr; ///< RMSNorm gamma weights [norm_dim]

            float eps = 1e-6f;         ///< Epsilon for numerical stability
            bool subtract_one = false; ///< gamma_effective = 1.0 + gamma_stored
            int seq_len = 0;           ///< Explicit sequence length

            /// Per-head normalization size. When > 0, normalizes over chunks of
            /// norm_dim elements (e.g., d_v=128 for head-level norm). When 0,
            /// normalizes over the full model dimension (d_model).
            int norm_dim = 0;

            /// Apply SiLU activation to gate before multiplying.
            /// Required for GDN layers: output = RMSNorm(input) * SiLU(gate).
            bool gate_silu = false;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> gate_buffer_id;
            std::optional<BufferId> output_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit GatedRMSNormStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GATED_RMS_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
