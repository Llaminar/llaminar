#pragma once

/**
 * @file ActivationQuantLayout.h
 * @brief Metadata for ROCm activation quantization layout
 *
 * Describes how FP32 activations are quantized to INT8 for ROCm GEMM/GEMV,
 * using blockwise scale granularity.
 */

#include <cstdint>
#include <cstddef>

namespace llaminar2::rocm
{

    /**
     * @brief Activation quantization granularity mode
     */
    enum class ActivationQuantMode : uint8_t
    {
        /// One symmetric scale per K-block of 32 elements.
        /// d_scales_A shape: [M × ceil(K/block_size)]
        BLOCKWISE = 1,
    };

    /**
     * @brief Describes the layout of quantized activation data and associated scales.
     *
     * This struct is a metadata descriptor — it does not own any memory.
     * It describes how to interpret the quantized activation buffer and
     * the activation scale buffer produced by the quantization kernel.
     *
     * ## Usage
     *
     * ```cpp
     * auto layout = ActivationQuantLayout::rowWise(M, K);
     * // or
     * auto layout = ActivationQuantLayout::blockwise(M, K, 32);
     *
     * size_t scale_bytes = layout.scaleBufferBytes();
     * ```
     */
    struct ActivationQuantLayout
    {
        ActivationQuantMode mode = ActivationQuantMode::BLOCKWISE;

        int rows = 0;           ///< M (number of activation rows)
        int cols = 0;           ///< K (number of activation columns / input features)
        int block_size = 0;     ///< Elements per quantization block (e.g. 32)
        int blocks_per_row = 0; ///< Number of scale blocks per row (ceil(K/block_size))

        /// Total number of FP32 scale values
        int scaleCount() const { return rows * blocks_per_row; }

        /// Bytes needed for the activation scale buffer
        size_t scaleBufferBytes() const
        {
            return static_cast<size_t>(scaleCount()) * sizeof(float);
        }

        /// Bytes needed for the quantized INT8 activation buffer
        size_t quantBufferBytes() const
        {
            return static_cast<size_t>(rows) * cols * sizeof(int8_t);
        }

        /// Is this blockwise mode?
        bool isBlockwise() const { return mode == ActivationQuantMode::BLOCKWISE; }

        // =========================================================================
        // Factory methods
        // =========================================================================

        /// Create a blockwise activation quant layout
        static ActivationQuantLayout blockwise(int M, int K, int block_sz = 32)
        {
            ActivationQuantLayout layout;
            layout.mode = ActivationQuantMode::BLOCKWISE;
            layout.rows = M;
            layout.cols = K;
            layout.block_size = block_sz;
            layout.blocks_per_row = (K + block_sz - 1) / block_sz;
            return layout;
        }
    };

} // namespace llaminar2::rocm
