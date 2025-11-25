/**
 * @file FusedRMSNormQuantize.h
 * @brief RMSNorm kernel with optional output formats
 *
 * Provides RMSNorm with flexible output options:
 * - Standard FP32 output (default, for residual stream)
 * - Optional FP16/BF16 output (for memory-constrained scenarios)
 *
 * With the new FP32 residual architecture, the primary use case is
 * FP32 → RMSNorm → FP32, keeping the residual stream in full precision.
 * The Q8_1GemmKernel handles quantization internally.
 *
 * Note: The original "FusedRMSNormQuantize" design (FP32→INT8) is deprecated.
 * INT8 quantization is now handled on-the-fly inside Q8_1GemmKernel.
 *
 * Performance Benefits:
 * - Single-pass RMSNorm computation
 * - SIMD-optimized (AVX512/AVX2)
 * - Optional precision conversion on output
 *
 * Fusion Chain (New Architecture):
 *   [FP32 residual] → RMSNorm → [FP32] → Q8_1GemmKernel (quant inside)
 *
 * @author David Sanftenberg
 * @date 2025-11-22
 * @updated 2025-11-24 - Reworked for FP32 residual architecture
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../tensors/TensorKernels.h"
#include <cstdint>
#include <vector>
#include <memory>

namespace llaminar2
{
    /**
     * @brief RMSNorm kernel with FP32 output
     *
     * Standard RMSNorm: output = (input / rms) * gamma
     * where rms = sqrt(mean(input²) + eps)
     *
     * Usage:
     *   FusedRMSNormQuantize kernel;  // Name kept for backward compat
     *   kernel.apply(input_fp32, gamma, output_fp32, rows, cols, eps);
     */
    class FusedRMSNormQuantize : public CPUKernelBase, public ITensorRMSNorm
    {
    public:
        FusedRMSNormQuantize() = default;
        ~FusedRMSNormQuantize() override = default;

        // =============================================================================
        // CPUKernelBase Interface (Fusion Framework)
        // =============================================================================

        /**
         * @brief Get kernel I/O contract for fusion pattern detection
         */
        KernelContract get_contract() const override
        {
            return KernelContract{
                .accepted_input_formats = {TensorFormat::FP32}, // Accept FP32 input
                .output_format = TensorFormat::FP32,            // Produce FP32 output
                .supports_inplace = true,                       // Can write in-place
                .is_fusable = true                              // Can be fused with GEMM consumers
            };
        }

        bool supports_fusion() const override
        {
            return true;
        }

        TensorFormat preferred_fusion_format() const override
        {
            return TensorFormat::FP32; // FP32 output for fusion chains
        }

        // =============================================================================
        // ITensorRMSNorm Interface (Standard RMSNorm API)
        // =============================================================================

        /**
         * @brief Apply RMSNorm (ITensorRMSNorm interface)
         *
         * Computes: output = (input / rms) * weight
         * where rms = sqrt(mean(input²) + epsilon)
         */
        bool apply(
            const float *input, const float *weight, float *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)use_bf16;
            (void)mpi_ctx;
            (void)device_idx;
            // Not implemented - this kernel only supports fused INT8 output via execute()
            return false;
        }

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        /**
         * @brief Execute fused RMSNorm + INT8 quantization (ITensorRMSNorm::execute override)
         *
         * Implements the ITensorRMSNorm::execute() interface for fused quantization path.
         *
         * @param input Input tensor [rows, cols] FP32
         * @param weight RMSNorm scale parameters [cols] FP32 (gamma)
         * @param output Output tensor [rows, cols] INT8
         * @param scales Per-row quantization scales [rows] FP32
         * @param rows Number of rows (sequence length)
         * @param cols Hidden dimension (d_model)
         * @param epsilon RMSNorm epsilon for numerical stability (default: 1e-6)
         * @param mpi_ctx MPI context (unused for now)
         * @param device_idx Device index (must be -1 for CPU)
         * @return true on success, false on error
         *
         * Output:
         * - output[i*cols + j] = round((input[i*cols + j] / rms[i]) * weight[j] / scales[i])
         * - scales[i] = max(|normalized_row[i]|) / 127
         * - Quantized range: [-127, 127] (symmetric INT8)
         */
        bool execute(
            const float *input,
            const float *weight,
            int8_t *output,
            float *scales,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

    private:
        /**
         * @brief SIMD-optimized single-row processing
         *
         * Fuses: RMS computation → normalization → gamma scaling → quantization
         */
        void process_row_fused(
            const float *input_row,
            const float *gamma,
            int8_t *output_row,
            float &out_scale,
            int d_model,
            float epsilon);

        /**
         * @brief AVX512 implementation (16-way FP32)
         */
        void process_row_fused_avx512(
            const float *input_row,
            const float *gamma,
            int8_t *output_row,
            float &out_scale,
            int d_model,
            float epsilon);

        /**
         * @brief AVX2 implementation (8-way FP32)
         */
        void process_row_fused_avx2(
            const float *input_row,
            const float *gamma,
            int8_t *output_row,
            float &out_scale,
            int d_model,
            float epsilon);

        /**
         * @brief Scalar fallback (portable)
         */
        void process_row_fused_scalar(
            const float *input_row,
            const float *gamma,
            int8_t *output_row,
            float &out_scale,
            int d_model,
            float epsilon);
    };

} // namespace llaminar2
