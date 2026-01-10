/**
 * @file SwiGLUStage.h
 * @brief SwiGLU activation stage
 */

#pragma once

#include "../IComputeStage.h"
#include "backends/DeviceId.h"

namespace llaminar2
{

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
            // Type-safe tensor pointers (required)
            const ITensor *gate = nullptr; ///< Gate tensor [seq_len, intermediate_dim]
            const ITensor *up = nullptr;   ///< Up tensor [seq_len, intermediate_dim]
            ITensor *output = nullptr;     ///< Output tensor [seq_len, intermediate_dim]

            // Explicit seq_len override (for pre-allocated buffers)
            // If 0, derives from tensor dimensions
            int seq_len = 0;

            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            DeviceId device_id = DeviceId::cpu();
        };

        explicit SwiGLUStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SWIGLU; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /// Target device for coherence management
        DeviceId preferredDevice() const override { return params_.device_id; }

    private:
        Params params_;
    };

} // namespace llaminar2
