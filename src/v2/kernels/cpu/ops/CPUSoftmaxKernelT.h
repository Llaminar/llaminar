/**
 * @file CPUSoftmaxKernelT.h
 * @brief Typed Softmax kernel with ActivationPrecision support
 * @author David Sanftenberg
 *
 * This kernel follows the "kernel black-box model" pattern from CPURoPEKernelT:
 * - External interface uses configured ActivationPrecision (FP32, BF16, FP16, Q8_1)
 * - Internal computation varies by precision
 * - Q8_1 uses integer-aware operations without full FP32 dequantization
 *
 * Usage:
 *   CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel_fp32;
 *   kernel_fp32.apply_typed(scores_fp32, rows, cols, causal, scale);
 *
 *   CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel_q8;
 *   kernel_q8.apply_typed(scores_q8, rows, n_blocks_per_row, causal, scale);
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../execution/RuntimeConfig.h"
#include "../../../tensors/BlockStructures.h"

#include <cstdint>

namespace llaminar2
{

    // Forward declarations
    class MPIContext;

    // =========================================================================
    // Precision Metadata Traits (same pattern as CPURoPEKernelT)
    // =========================================================================

    namespace detail
    {
        template <ActivationPrecision P>
        struct SoftmaxPrecisionMetadata;

        template <>
        struct SoftmaxPrecisionMetadata<ActivationPrecision::FP32>
        {
            using StorageType = float;
            static constexpr const char *name = "FP32";
            static constexpr float compression_ratio = 1.0f;
        };

        template <>
        struct SoftmaxPrecisionMetadata<ActivationPrecision::BF16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "BF16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct SoftmaxPrecisionMetadata<ActivationPrecision::FP16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "FP16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct SoftmaxPrecisionMetadata<ActivationPrecision::Q8_1>
        {
            using StorageType = Q8_1Block;
            static constexpr const char *name = "Q8_1";
            static constexpr float compression_ratio = 4.0f;
        };
    } // namespace detail

    // =========================================================================
    // Primary Template (Base Declaration)
    // =========================================================================

    /**
     * @brief Typed Softmax kernel template
     *
     * Primary template provides type traits and common metadata.
     * Specializations implement precision-specific apply_typed().
     */
    template <ActivationPrecision Precision>
    class CPUSoftmaxKernelT : public CPUKernelBase
    {
    public:
        using Metadata = detail::SoftmaxPrecisionMetadata<Precision>;
        using StorageType = typename Metadata::StorageType;

        virtual ~CPUSoftmaxKernelT() = default;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1; // CPU only
        }

        static constexpr ActivationPrecision precision() { return Precision; }
        static const char *precision_name() { return Metadata::name; }
        static constexpr float compression_ratio() { return Metadata::compression_ratio; }

        /**
         * @brief Apply softmax with typed storage
         *
         * @param data Input/output data (modified in-place)
         * @param rows Number of rows
         * @param cols Number of columns per row (for Q8_1: number of blocks per row)
         * @param use_causal_mask Apply causal masking (row i: only j <= i valid)
         * @param scale Scale factor applied before exp (typically 1/sqrt(d_k))
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            StorageType *data,
            int rows,
            int cols,
            bool use_causal_mask = false,
            float scale = 1.0f,
            int device_idx = -1);
    };

    // =========================================================================
    // FP32 Specialization
    // =========================================================================

    template <>
    class CPUSoftmaxKernelT<ActivationPrecision::FP32> : public CPUKernelBase
    {
    public:
        using StorageType = float;

        CPUSoftmaxKernelT() = default;
        ~CPUSoftmaxKernelT() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
        static const char *precision_name() { return "FP32"; }
        static constexpr float compression_ratio() { return 1.0f; }

        /**
         * @brief Apply softmax to FP32 data
         *
         * @param data Input/output FP32 data [rows, cols] (in-place)
         * @param rows Number of rows
         * @param cols Number of columns per row
         * @param use_causal_mask Apply causal masking
         * @param scale Scale factor applied before exp
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            float *data,
            int rows,
            int cols,
            bool use_causal_mask = false,
            float scale = 1.0f,
            int device_idx = -1);
    };

    // =========================================================================
    // BF16 Specialization
    // =========================================================================

    template <>
    class CPUSoftmaxKernelT<ActivationPrecision::BF16> : public CPUKernelBase
    {
    public:
        using StorageType = uint16_t;

        CPUSoftmaxKernelT() = default;
        ~CPUSoftmaxKernelT() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
        static const char *precision_name() { return "BF16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        /**
         * @brief Apply softmax to BF16 data
         *
         * @param data Input/output BF16 data [rows, cols] (in-place, stored as uint16_t)
         * @param rows Number of rows
         * @param cols Number of columns per row
         * @param use_causal_mask Apply causal masking
         * @param scale Scale factor applied before exp
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            uint16_t *data,
            int rows,
            int cols,
            bool use_causal_mask = false,
            float scale = 1.0f,
            int device_idx = -1);
    };

    // =========================================================================
    // FP16 Specialization
    // =========================================================================

    template <>
    class CPUSoftmaxKernelT<ActivationPrecision::FP16> : public CPUKernelBase
    {
    public:
        using StorageType = uint16_t;

        CPUSoftmaxKernelT() = default;
        ~CPUSoftmaxKernelT() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
        static const char *precision_name() { return "FP16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        /**
         * @brief Apply softmax to FP16 data
         *
         * @param data Input/output FP16 data [rows, cols] (in-place, stored as uint16_t)
         * @param rows Number of rows
         * @param cols Number of columns per row
         * @param use_causal_mask Apply causal masking
         * @param scale Scale factor applied before exp
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            uint16_t *data,
            int rows,
            int cols,
            bool use_causal_mask = false,
            float scale = 1.0f,
            int device_idx = -1);
    };

    // =========================================================================
    // Q8_1 Specialization (Integer-Aware)
    // =========================================================================

    /**
     * @brief Q8_1 specialization - integer-aware softmax without full FP32 dequantization
     *
     * This specialization uses the integer-domain softmax primitives:
     * - Integer max-finding within Q8_1 blocks
     * - Batched scale operations
     * - Direct requantization to output Q8_1 blocks
     *
     * Algorithm (3-pass):
     *   1. Find max in integer domain (compare qs values, adjust for scales)
     *   2. Compute exp sum with factored scale multiply
     *   3. Normalize and requantize directly to Q8_1
     *
     * @note cols parameter represents n_blocks_per_row (not element count)
     *       Actual column count = n_blocks_per_row * 32 (Q8_1 block size)
     */
    template <>
    class CPUSoftmaxKernelT<ActivationPrecision::Q8_1> : public CPUKernelBase
    {
    public:
        using StorageType = Q8_1Block;

        CPUSoftmaxKernelT() = default;
        ~CPUSoftmaxKernelT() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::Q8_1; }
        static const char *precision_name() { return "Q8_1"; }
        static constexpr float compression_ratio() { return 4.0f; }

        /**
         * @brief Apply softmax to Q8_1 data
         *
         * @param data Input/output Q8_1 blocks [rows * n_blocks_per_row] (in-place)
         * @param rows Number of rows
         * @param n_blocks_per_row Number of Q8_1 blocks per row (element count = n_blocks * 32)
         * @param use_causal_mask Apply causal masking
         * @param scale Scale factor applied before exp (typically 1/sqrt(d_k))
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            Q8_1Block *data,
            int rows,
            int n_blocks_per_row,
            bool use_causal_mask = false,
            float scale = 1.0f,
            int device_idx = -1);
    };

} // namespace llaminar2
