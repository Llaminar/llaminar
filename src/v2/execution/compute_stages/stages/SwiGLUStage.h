/**
 * @file SwiGLUStage.h
 * @brief SwiGLU activation stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include <memory>

namespace llaminar2
{
    // Forward declaration
    class ITensorSwiGLU;

    /**
     * @brief SwiGLU activation stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses typed kernel dispatch based on tensor precision.
     */
    class SwiGLUStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Type-safe tensor pointers (required)
            const ITensor *gate = nullptr; ///< Gate tensor [seq_len, intermediate_dim]
            const ITensor *up = nullptr;   ///< Up tensor [seq_len, intermediate_dim]
            ITensor *output = nullptr;     ///< Output tensor [seq_len, intermediate_dim]

            // Explicit seq_len override (for pre-allocated buffers)
            // If 0, derives from tensor dimensions
            int seq_len = 0;
        };

        explicit SwiGLUStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SWIGLU; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /// Target device for coherence management


    private:
        Params params_;
        mutable std::unique_ptr<llaminar2::ITensorSwiGLU> cached_kernel_;
    };

} // namespace llaminar2
