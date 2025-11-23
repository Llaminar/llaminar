/**
 * @file FusedRMSNormQuantize.h
 * @brief Fused RMSNorm + INT8 quantization kernel (Phase 1: High-Impact Fusion)
 *
 * Fuses RMS normalization with per-row INT8 quantization in a single pass.
 * Eliminates intermediate FP32 buffer and redundant memory traffic.
 *
 * Performance Benefits:
 * - Saves 1 FP32 intermediate buffer allocation (seq_len × d_model × 4 bytes)
 * - Reduces memory bandwidth by ~33% (1 write instead of 2)
 * - Improves cache efficiency (single-pass algorithm)
 *
 * Expected Speedup: 5-10% per RMSNorm operation (2× per layer: attention + FFN)
 *
 * Algorithm:
 * 1. Compute RMS per row: rms = sqrt(mean(x²) + eps)
 * 2. Normalize: x_norm = x / rms
 * 3. Apply gamma: x_scaled = x_norm * gamma
 * 4. Quantize per-row: x_int8 = round(x_scaled / scale), scale = max(|x_scaled|) / 127
 *
 * SIMD Optimizations:
 * - AVX512: 16-way FP32 operations
 * - AVX2: 8-way FP32 operations
 * - Scalar fallback for tail elements
 *
 * @author David Sanftenberg
 * @date 2025-11-22
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
     * @brief Fused RMSNorm + INT8 quantization kernel
     *
     * Replaces: RMSNorm(FP32→FP32) → Quantize(FP32→INT8)
     * With: FusedRMSNormQuantize(FP32→INT8)
     *
     * Usage:
     *   FusedRMSNormQuantize kernel;
     *   kernel.execute(input_fp32, gamma, output_int8, output_scales,
     *                  seq_len, d_model, eps);
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
                .output_format = TensorFormat::INT8,            // Produce INT8 output
                .supports_inplace = false,                      // Cannot write in-place (type change)
                .is_fusable = true                              // Can be fused with GEMM consumers
            };
        }

        bool supports_fusion() const override
        {
            return true; // High-priority fusion candidate
        }

        TensorFormat preferred_fusion_format() const override
        {
            return TensorFormat::INT8; // Output format for fusion chains
        }

        // =============================================================================
        // ITensorRMSNorm Interface (Standard RMSNorm API)
        // =============================================================================

        /**
         * @brief Apply RMSNorm (ITensorRMSNorm interface - outputs FP32)
         *
         * This is the standard ITensorRMSNorm interface that outputs FP32.
         * For fused INT8 output, use execute() instead.
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
