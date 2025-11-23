/**
 * @file CPUKernelBase.h
 * @brief Base class for all CPU kernel implementations with fusion support
 * @author David Sanftenberg
 * @date 2025-11-22
 */

#pragma once

#include <vector>
#include <algorithm>

namespace llaminar2
{
    /**
     * @brief Tensor format enumeration for kernel I/O contracts
     *
     * Used to describe what formats a kernel accepts/produces,
     * enabling automatic fusion pattern detection.
     */
    enum class TensorFormat
    {
        // Floating-point formats
        FP32, ///< 32-bit floating point
        BF16, ///< Brain float 16
        FP16, ///< IEEE 754 half precision

        // Integer formats (activations)
        INT8,  ///< 8-bit signed integer (quantized activations)
        INT32, ///< 32-bit signed integer (accumulator format)

        // Quantized weight formats
        Q4_0,    ///< 4-bit quantized (llama.cpp format)
        Q4_1,    ///< 4-bit quantized with offset
        Q5_0,    ///< 5-bit quantized
        Q5_1,    ///< 5-bit quantized with offset
        Q8_0,    ///< 8-bit quantized
        Q8_1,    ///< 8-bit quantized with offset
        Q2_K,    ///< 2-bit K-quantized
        Q3_K,    ///< 3-bit K-quantized
        Q4_K,    ///< 4-bit K-quantized
        Q5_K,    ///< 5-bit K-quantized
        Q6_K,    ///< 6-bit K-quantized
        Q8_K,    ///< 8-bit K-quantized
        IQ1_M,   ///< 1-bit importance quantized (medium)
        IQ1_S,   ///< 1-bit importance quantized (small)
        IQ2_XXS, ///< 2-bit importance quantized (extra extra small)
        IQ2_XS,  ///< 2-bit importance quantized (extra small)
        IQ3_XXS, ///< 3-bit importance quantized (extra extra small)
        IQ3_S,   ///< 3-bit importance quantized (small)
        IQ4_NL,  ///< 4-bit importance quantized (non-linear)
        IQ4_XS,  ///< 4-bit importance quantized (extra small)

        UNKNOWN ///< Unknown/unsupported format
    };

    /**
     * @brief Kernel I/O contract specification
     *
     * Describes what tensor formats a kernel accepts as input and produces as output.
     * Used by the fusion framework to detect compatible kernel chains.
     */
    struct KernelContract
    {
        /// Input formats this kernel can accept (empty = accepts any)
        std::vector<TensorFormat> accepted_input_formats;

        /// Output format this kernel produces
        TensorFormat output_format;

        /// Can this kernel write results in-place to input buffer?
        bool supports_inplace;

        /// Can this kernel be fused with adjacent kernels?
        bool is_fusable;

        /**
         * @brief Check if this kernel's output can be fused with the next kernel's input
         *
         * @param next Contract of the next kernel in sequence
         * @return true if fusion is possible
         */
        bool can_fuse_with(const KernelContract &next) const
        {
            if (!is_fusable || !next.is_fusable)
            {
                return false;
            }

            // If next kernel accepts any input, fusion is possible
            if (next.accepted_input_formats.empty())
            {
                return true;
            }

            // Check if our output format is in next kernel's accepted inputs
            return std::find(next.accepted_input_formats.begin(),
                             next.accepted_input_formats.end(),
                             output_format) != next.accepted_input_formats.end();
        }
    };

    /**
     * @brief Base class for all CPU kernel implementations
     *
     * Provides common functionality and interface for CPU-based kernels.
     * All CPU kernel classes should inherit from this base.
     *
     * Phase 1 (Fusion Framework):
     * - Added kernel contracts for fusion pattern detection
     * - Added virtual methods for fusion capabilities
     *
     * Future phases will use these contracts for automatic fusion optimization.
     */
    class CPUKernelBase
    {
    public:
        virtual ~CPUKernelBase() = default;

        /**
         * @brief Get the kernel's I/O contract
         *
         * @return KernelContract describing accepted inputs and produced output
         *
         * Default implementation returns a contract that accepts any input
         * and produces FP32 output (safe fallback for kernels without fusion support).
         */
        virtual KernelContract get_contract() const
        {
            return KernelContract{
                .accepted_input_formats = {}, // Accept any input
                .output_format = TensorFormat::FP32,
                .supports_inplace = false,
                .is_fusable = false};
        }

        /**
         * @brief Check if this kernel supports fusion with adjacent kernels
         *
         * @return true if kernel can be fused
         */
        virtual bool supports_fusion() const
        {
            return false; // Conservative default
        }

        /**
         * @brief Get the preferred intermediate format for fusion chains
         *
         * When this kernel is fused with others, what format should be used
         * for intermediate results? (e.g., INT8 for quantized paths, FP32 for native)
         *
         * @return Preferred TensorFormat for fusion intermediates
         */
        virtual TensorFormat preferred_fusion_format() const
        {
            return TensorFormat::FP32; // Safe default
        }

    protected:
        CPUKernelBase() = default;
    };

} // namespace llaminar2
