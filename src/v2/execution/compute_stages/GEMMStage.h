/**
 * @file GEMMStage.h
 * @brief GEMM stage: C = alpha * A * B + beta * C
 */

#pragma once

#include "IComputeStage.h"

namespace llaminar2
{

    /**
     * @brief GEMM stage: C = alpha * A * B + beta * C
     *
     * Takes FP32 activations and quantized/FP32 weights, handles quantization internally.
     * For multi-projection patterns (Q/K/V or gate/up), use FusedQKVGEMMStage or
     * FusedGateUpGEMMStage which efficiently share quantization across projections.
     *
     * **Tensor Parallelism Support (Phase 2)**:
     * When `output_range` is set, executes a row-sliced GEMM for tensor parallelism:
     * - Uses `KernelFactory::getOrCreateGemmSliced()` to create sliced kernel
     * - Only computes output rows in [output_range.start, output_range.end)
     * - Caller is responsible for MPI AllReduce after execution if needed
     */
    class GEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *A = nullptr; ///< Input activation tensor [m, k]
            const TensorBase *B = nullptr; ///< Weight tensor [k, n] (may be quantized)
            TensorBase *C = nullptr;       ///< Output tensor [m, n]
            int m = 0, n = 0, k = 0;       ///< Matrix dimensions
            float alpha = 1.0f;            ///< Scale factor for A*B
            float beta = 0.0f;             ///< Scale factor for existing C
            bool transpose_B = false;      ///< Whether B is transposed (n × k)

            // =================================================================
            // Bias Configuration (Enhanced for Tensor Parallelism)
            // =================================================================

            /**
             * @brief Raw bias pointer (legacy interface)
             * @deprecated Use bias_tensor instead for TensorSlice/TP compatibility.
             */
            const float *bias = nullptr;

            /**
             * @brief Bias tensor (preferred interface)
             */
            const TensorBase *bias_tensor = nullptr;

            /**
             * @brief Whether bias is required for this operation
             */
            bool bias_required = false;

            /**
             * @brief Get bias data pointer from either source
             */
            const float *getBiasData() const;

            /**
             * @brief Validate params and throw on configuration error
             */
            void validate(const std::string &stage_name = "GEMMStage") const;

            // SwiGLU fusion (extended fields)
            const TensorBase *gate_input = nullptr;
            bool do_swiglu = false;

            // MPI context for tensor-parallel execution (optional)
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;

            // =================================================================
            // Tensor Parallelism Parameters (Phase 2)
            // =================================================================

            /**
             * @brief Output row range for row-parallel GEMM
             */
            WorkRange output_range;

            /**
             * @brief Whether this GEMM requires AllReduce after execution
             */
            bool needs_allreduce = false;
        };

        explicit GEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;
        bool requiresAllreduce() const override { return params_.needs_allreduce; }

        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
