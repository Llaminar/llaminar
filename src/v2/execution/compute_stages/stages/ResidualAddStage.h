/**
 * @file ResidualAddStage.h
 * @brief Residual addition stage: output = input + residual
 */

#pragma once

#include "../IComputeStage.h"

namespace llaminar2
{

    /**
     * @brief Residual addition stage: output = input + residual
     *
     * Precision-aware implementation:
     * - FP32: Simple float addition
     * - BF16/FP16: Convert to FP32, add, convert back
     * - Q8_1: Native block addition via simd::q8_1_add_q8_1()
     *
     * The Q8_1 path performs fused dequant-add-requant in registers,
     * avoiding memory traffic for intermediate FP32 values.
     */
    class ResidualAddStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *input = nullptr;    ///< Input tensor (projection output)
            const TensorBase *residual = nullptr; ///< Residual tensor (previous hidden state)
            TensorBase *output = nullptr;         ///< Output tensor (can be same as residual for in-place)

            // Number of elements to process (0 = use input->numel())
            // IMPORTANT: For decode mode with pre-allocated buffers, this must be set to
            // seq_len * hidden_dim to avoid processing garbage data beyond the actual sequence.
            size_t num_elements = 0;

            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit ResidualAddStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ADD_RESIDUAL; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;

        // Type-specific implementations
        bool executeFP32(IDeviceContext *ctx, size_t num_elements);
        bool executeBF16(IDeviceContext *ctx, size_t num_elements);
        bool executeFP16(IDeviceContext *ctx, size_t num_elements);
        bool executeQ8_1(IDeviceContext *ctx, size_t num_elements);
        bool executeQ16_1(IDeviceContext *ctx, size_t num_elements); // Phase 6: Full integer residual

        // Mixed-type implementations for hybrid precision modes
        bool executeFP32_Q8_1_to_Q8_1(IDeviceContext *ctx, size_t num_elements);
        bool executeQ8_1_Q16_1_to_Q16_1(IDeviceContext *ctx, size_t num_elements);
        bool executeFP32_Q16_1_to_Q16_1(IDeviceContext *ctx, size_t num_elements);
        bool executeQ8_1_FP32_to_FP32(IDeviceContext *ctx, size_t num_elements);
    };

} // namespace llaminar2
