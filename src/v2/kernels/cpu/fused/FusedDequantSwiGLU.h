/**
 * @file FusedDequantSwiGLU.h
 * @brief SwiGLU activation kernel for FP32 inputs
 *
 * Applies SwiGLU activation to FP32 gate and up projections:
 *   output = gate * silu(up)
 *   where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * With the new FP32 residual architecture, this kernel no longer needs
 * to perform dequantization - it receives FP32 inputs directly from
 * Q8_1GemmKernel outputs.
 *
 * Performance Benefits:
 * - SIMD-optimized SwiGLU (AVX512/AVX2 fused multiply-add)
 * - Single-pass computation
 * - Good cache locality
 *
 * Fusion Chain (New Architecture):
 *   FusedDualGEMM → FusedSwiGLU (this kernel)
 *   (FP32→FP32)   → (FP32→FP32)
 *
 * @author David Sanftenberg
 * @date 2025-11-23
 * @updated 2025-11-24 - Reworked for FP32 residual architecture (no dequant needed)
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../tensors/TensorKernels.h"
#include <cstdint>
#include <cmath>
#include <vector>

namespace llaminar2
{
    /**
     * @brief SwiGLU activation kernel for FP32 inputs
     *
     * Replaces the old FusedDequantSwiGLU that expected INT32 inputs.
     * Now operates on FP32 inputs directly.
     *
     * Usage:
     *   FusedSwiGLU kernel;
     *   kernel.execute(gate_fp32, up_fp32, output_fp32, m, n);
     */
    class FusedSwiGLU : public CPUKernelBase
    {
    public:
        FusedSwiGLU() = default;
        ~FusedSwiGLU() override = default;

        // =============================================================================
        // CPUKernelBase Interface (Fusion Framework)
        // =============================================================================

        /**
         * @brief Get kernel I/O contract for fusion pattern detection
         */
        KernelContract get_contract() const override
        {
            return KernelContract{
                .accepted_input_formats = {TensorFormat::FP32}, // Accept FP32 inputs
                .output_format = TensorFormat::FP32,            // Produce FP32 output
                .supports_inplace = true,                       // Can write to gate buffer
                .is_fusable = true                              // Can fuse with downstream ops
            };
        }

        bool supports_fusion() const override
        {
            return true;
        }

        TensorFormat preferred_fusion_format() const override
        {
            return TensorFormat::FP32;
        }

        // =============================================================================
        // Execution Interface
        // =============================================================================

        /**
         * @brief Execute SwiGLU activation
         *
         * Performs: output[i,j] = gate[i,j] * silu(up[i,j])
         *
         * @param gate Gate activations [m, n] FP32
         * @param up Up activations [m, n] FP32
         * @param output Output activations [m, n] FP32 (can alias gate for in-place)
         * @param m Batch size (sequence length)
         * @param n Hidden dimension (intermediate_size)
         * @return true on success, false on error
         */
        bool execute(
            const float *gate,
            const float *up,
            float *output,
            int m, int n);

        /**
         * @brief Check if kernel supports given device
         * @param device_idx Device index (-1 = CPU)
         * @return true if device is supported
         */
        bool supports_device(int device_idx) const
        {
            return device_idx == -1; // CPU only for now
        }
    };

    // Keep old name as alias for backward compatibility during transition
    using FusedDequantSwiGLU = FusedSwiGLU;

} // namespace llaminar2
