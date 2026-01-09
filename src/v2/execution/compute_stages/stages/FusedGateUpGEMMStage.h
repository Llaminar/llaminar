/**
 * @file FusedGateUpGEMMStage.h
 * @brief Fused Gate/Up projection stage for FFN
 */

#pragma once

#include "../IComputeStage.h"
#include "backends/DeviceId.h"

namespace llaminar2
{

    /**
     * @brief Fused Gate/Up projection stage for FFN
     *
     * Efficiently computes both gate and up projections from a shared input
     * by quantizing the input once and reusing the Q8_1 buffer for both
     * projections. This avoids redundant quantization and improves cache locality.
     *
     * Pattern: input → [gate_proj, up_proj] → SwiGLU → down_proj
     *
     * This stage delegates to QuantisedGemmKernel::multiply_fused_multi(), which
     * handles the quantization and multi-projection execution internally.
     */
    class FusedGateUpGEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const ITensor *input = nullptr; ///< Input activation tensor [m, k]
            int m = 0;                      ///< Batch size * seq_len
            int k = 0;                      ///< Input dimension (d_model)

            // Gate projection
            const ITensor *w_gate = nullptr;
            ITensor *output_gate = nullptr;
            int n_gate = 0;
            const float *bias_gate = nullptr;

            // Up projection
            const ITensor *w_up = nullptr;
            ITensor *output_up = nullptr;
            int n_up = 0;
            const float *bias_up = nullptr;

            // Optional MPI context
            const MPIContext *mpi_ctx = nullptr;
            DeviceId device_id = DeviceId::cpu();
        };

        explicit FusedGateUpGEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_GATE_UP; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
