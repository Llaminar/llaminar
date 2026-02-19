/**
 * @file CPURMSNormKernelT.h
 * @brief Typed RMSNorm kernel with fused precision conversion
 *
 * This kernel implements the "kernel black-box model" for typed residuals:
 * - External interface uses configured ActivationPrecision
 * - Internal computation always uses FP32 for numerical stability
 * - Fused dequant on input (if input precision != FP32)
 * - Fused requant on output (if output precision != FP32)
 *
 * Design Contract:
 * - Pipeline instantiates CPURMSNormKernelT<Precision> based on config
 * - Kernel accepts input/output buffers in that precision format
 * - All precision conversion is internal and transparent to caller
 *
 * Memory Bandwidth Benefits (vs FP32):
 * - BF16: 2× reduction (input: 2B, output: 2B vs 4B each)
 * - FP16: 2× reduction (input: 2B, output: 2B vs 4B each)
 * - Q8_1: 3.5× reduction (36 bytes per 32 elements vs 128 bytes)
 *
 * @author David Sanftenberg
 * @date 2025-12-04
 */

#pragma once

#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/Tensors.h" // For FP32Tensor, BF16Tensor, etc. (apply_tensor)
#include "../CPUKernelBase.h"
#include "../../../utils/KernelProfiler.h"
#include <memory>
#include <cstdint>

namespace llaminar2
{

    // =========================================================================
    // Precision Metadata Traits (minimal - just storage type and metadata)
    // =========================================================================

    namespace detail
    {
        /**
         * @brief Primary template for precision metadata (must be specialized)
         */
        template <ActivationPrecision Precision>
        struct PrecisionMetadata;

        template <>
        struct PrecisionMetadata<ActivationPrecision::FP32>
        {
            using StorageType = float;
            static constexpr const char *name = "FP32";
            static constexpr float compression_ratio = 1.0f;
        };

        template <>
        struct PrecisionMetadata<ActivationPrecision::BF16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "BF16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct PrecisionMetadata<ActivationPrecision::FP16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "FP16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct PrecisionMetadata<ActivationPrecision::Q8_1>
        {
            using StorageType = Q8_1Block;
            static constexpr const char *name = "Q8_1";
            static constexpr float compression_ratio = 3.5f; // 36 bytes per 32 elements vs 128
        };

        template <>
        struct PrecisionMetadata<ActivationPrecision::Q16_1>
        {
            using StorageType = Q16_1Block;
            static constexpr const char *name = "Q16_1";
            static constexpr float compression_ratio = 1.778f; // 72 bytes per 32 elements vs 128
        };
    } // namespace detail

    /**
     * @brief Typed RMSNorm kernel with fused precision conversion
     *
     * @tparam Precision The ActivationPrecision for input/output buffers
     *
     * Usage:
     * @code
     * // Pipeline creates kernel based on configured precision
     * auto kernel = std::make_unique<CPURMSNormKernelT<ActivationPrecision::BF16>>();
     *
     * // Kernel takes BF16 input, computes in FP32, outputs BF16
     * kernel->apply_typed(bf16_input, gamma, bf16_output, rows, cols, epsilon);
     * @endcode
     */
    template <ActivationPrecision Precision>
    class CPURMSNormKernelT : public CPUKernelBase
    {
    public:
        using Metadata = detail::PrecisionMetadata<Precision>;
        using StorageType = typename Metadata::StorageType;

        CPURMSNormKernelT() = default;
        ~CPURMSNormKernelT() = default;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1; // CPU only
        }

        /**
         * @brief Apply RMSNorm with typed input/output (fused precision conversion)
         *
         * Algorithm:
         * 1. If input precision != FP32: fused dequantize to FP32
         * 2. Compute RMSNorm in FP32: output = (x / rms(x)) * gamma
         * 3. If output precision != FP32: fused quantize to output format
         *
         * @param input Input buffer in StorageType format [rows, cols]
         * @param gamma RMSNorm scale weights [cols] (always FP32)
         * @param output Output buffer in StorageType format [rows, cols]
         * @param rows Number of rows (batch_size * seq_len)
         * @param cols Hidden dimension (d_model)
         * @param epsilon Numerical stability epsilon (default: 1e-6f)
         * @param device_idx Device index (-1 = CPU)
         * @return true on success, false on error
         */
        bool apply_typed(
            const StorageType *input,
            const float *gamma,
            StorageType *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        /**
         * @brief Apply RMSNorm with fused residual addition
         *
         * Computes: output = RMSNorm(dequant(residual) + fp32_input) * gamma
         * Then quantizes output to configured precision.
         *
         * This fuses three operations:
         * 1. Dequantize typed residual to FP32
         * 2. Add FP32 input to dequantized residual
         * 3. Apply RMSNorm
         * 4. Quantize output to typed format
         *
         * @param residual Typed residual buffer [rows, cols]
         * @param fp32_input FP32 input to add to residual [rows, cols]
         * @param gamma RMSNorm scale weights [cols]
         * @param output Typed output buffer [rows, cols]
         * @param rows Number of rows
         * @param cols Hidden dimension
         * @param epsilon Numerical stability epsilon
         * @param device_idx Device index (-1 = CPU)
         * @return true on success, false on error
         */
        bool apply_with_residual_add(
            const StorageType *residual,
            const float *fp32_input,
            const float *gamma,
            StorageType *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        /**
         * @brief Get the precision of this kernel instance
         */
        static constexpr ActivationPrecision precision() { return Precision; }

        /**
         * @brief Get human-readable precision name
         */
        static const char *precision_name() { return Metadata::name; }

        /**
         * @brief Get compression ratio vs FP32
         */
        static constexpr float compression_ratio() { return Metadata::compression_ratio; }
    };

    // =========================================================================
    // FP32 Specialization (No Conversion Needed)
    // =========================================================================

    /**
     * @brief FP32 specialization - direct computation, no conversion
     *
     * This is the baseline case where input/output are already FP32.
     * No dequant/requant overhead.
     */
    template <>
    class CPURMSNormKernelT<ActivationPrecision::FP32> : public ITensorRMSNorm, public CPUKernelBase
    {
    public:
        using StorageType = float;

        CPURMSNormKernelT() = default;
        ~CPURMSNormKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        // ===== ITensorRMSNorm interface =====
        bool apply(
            const float *input, const float *weight, float *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)use_bf16;
            (void)mpi_ctx;
            return apply_typed(input, weight, output, rows, cols, epsilon, device_idx);
        }

        bool apply_bf16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // FP32 kernel doesn't handle BF16
        }

        bool apply_fp16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // FP32 kernel doesn't handle FP16
        }

        bool apply_q8_1(
            const Q8_1Block *input, const float *weight, Q8_1Block *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // FP32 kernel doesn't handle Q8_1
        }

        // ===== Tensor-based API (type-checked dispatch) =====
        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RMS_NORM);
            (void)mpi_ctx;
            // Type check: this kernel only handles FP32
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
                return false;
            return apply_typed(input->data(), weight->data(), output->mutable_data(),
                               rows, cols, epsilon, device_idx);
        }

        // ===== Typed API =====
        /**
         * @brief Apply RMSNorm with FP32 input/output (no conversion)
         */
        bool apply_typed(
            const float *input,
            const float *gamma,
            float *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        /**
         * @brief Apply RMSNorm with fused FP32 residual addition
         */
        bool apply_with_residual_add(
            const float *residual,
            const float *fp32_input,
            const float *gamma,
            float *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
        static const char *precision_name() { return "FP32"; }
        static constexpr float compression_ratio() { return 1.0f; }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rmsnorm()
                .withInput("input", "input activations [rows, cols]", KernelBufferDtype::FP32)
                .withWeight("gamma", "normalization weights [cols]", KernelBufferDtype::FP32)
                .withOutput("output", "normalized output [rows, cols]", KernelBufferDtype::FP32)
                .withScalar("epsilon", "numerical stability constant");
        }
    };

    // =========================================================================
    // BF16 Specialization (2x Compression)
    // =========================================================================

    template <>
    class CPURMSNormKernelT<ActivationPrecision::BF16> : public ITensorRMSNorm, public CPUKernelBase
    {
    public:
        using StorageType = uint16_t; // BF16 stored as uint16_t

        CPURMSNormKernelT() = default;
        ~CPURMSNormKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        // ===== ITensorRMSNorm interface =====
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
            return false; // BF16 kernel doesn't handle FP32
        }

        bool apply_bf16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            return apply_typed(input, weight, output, rows, cols, epsilon, device_idx);
        }

        bool apply_fp16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // BF16 kernel doesn't handle FP16
        }

        bool apply_q8_1(
            const Q8_1Block *input, const float *weight, Q8_1Block *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // BF16 kernel doesn't handle Q8_1
        }

        // ===== Tensor-based API (type-checked dispatch) =====
        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RMS_NORM);
            (void)mpi_ctx;
            // Type check: this kernel only handles BF16
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::BF16 ||
                output->native_type() != TensorType::BF16)
                return false;
            auto *bf16_in = static_cast<const BF16Tensor *>(input);
            auto *bf16_out = static_cast<BF16Tensor *>(output);
            return apply_typed(bf16_in->typed_data(), weight->data(), bf16_out->mutable_typed_data(),
                               rows, cols, epsilon, device_idx);
        }

        // ===== Typed API =====
        /**
         * @brief Apply RMSNorm with BF16 input/output
         *
         * Algorithm:
         * 1. Dequantize BF16 input to FP32 (vectorized)
         * 2. Compute RMSNorm in FP32
         * 3. Quantize FP32 output to BF16 (vectorized)
         */
        bool apply_typed(
            const uint16_t *input,
            const float *gamma,
            uint16_t *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        /**
         * @brief Apply RMSNorm with fused BF16 residual addition
         */
        bool apply_with_residual_add(
            const uint16_t *residual,
            const float *fp32_input,
            const float *gamma,
            uint16_t *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
        static const char *precision_name() { return "BF16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rmsnorm()
                .withInput("input", "input activations [rows, cols]", KernelBufferDtype::BF16)
                .withWeight("gamma", "normalization weights [cols]", KernelBufferDtype::FP32)
                .withOutput("output", "normalized output [rows, cols]", KernelBufferDtype::BF16)
                .withScalar("epsilon", "numerical stability constant");
        }
    };

    // =========================================================================
    // FP16 Specialization (2x Compression)
    // =========================================================================

    template <>
    class CPURMSNormKernelT<ActivationPrecision::FP16> : public ITensorRMSNorm, public CPUKernelBase
    {
    public:
        using StorageType = uint16_t; // FP16 stored as uint16_t

        CPURMSNormKernelT() = default;
        ~CPURMSNormKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        // ===== ITensorRMSNorm interface =====
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
            return false; // FP16 kernel doesn't handle FP32
        }

        bool apply_bf16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // FP16 kernel doesn't handle BF16
        }

        bool apply_fp16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            return apply_typed(input, weight, output, rows, cols, epsilon, device_idx);
        }

        bool apply_q8_1(
            const Q8_1Block *input, const float *weight, Q8_1Block *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // FP16 kernel doesn't handle Q8_1
        }

        // ===== Tensor-based API (type-checked dispatch) =====
        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RMS_NORM);
            (void)mpi_ctx;
            // Type check: this kernel only handles FP16
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::FP16 ||
                output->native_type() != TensorType::FP16)
                return false;
            auto *fp16_in = static_cast<const FP16Tensor *>(input);
            auto *fp16_out = static_cast<FP16Tensor *>(output);
            return apply_typed(fp16_in->typed_data(), weight->data(), fp16_out->mutable_typed_data(),
                               rows, cols, epsilon, device_idx);
        }

        // ===== Typed API =====
        /**
         * @brief Apply RMSNorm with FP16 input/output
         */
        bool apply_typed(
            const uint16_t *input,
            const float *gamma,
            uint16_t *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        /**
         * @brief Apply RMSNorm with fused FP16 residual addition
         */
        bool apply_with_residual_add(
            const uint16_t *residual,
            const float *fp32_input,
            const float *gamma,
            uint16_t *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
        static const char *precision_name() { return "FP16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rmsnorm()
                .withInput("input", "input activations [rows, cols]", KernelBufferDtype::FP16)
                .withWeight("gamma", "normalization weights [cols]", KernelBufferDtype::FP32)
                .withOutput("output", "normalized output [rows, cols]", KernelBufferDtype::FP16)
                .withScalar("epsilon", "numerical stability constant");
        }
    };

    // =========================================================================
    // Q8_1 Specialization (3.5x Compression)
    // =========================================================================

    // Forward declaration for Q8_1Block
    struct Q8_1Block;

    template <>
    class CPURMSNormKernelT<ActivationPrecision::Q8_1> : public ITensorRMSNorm, public CPUKernelBase
    {
    public:
        using StorageType = Q8_1Block;

        CPURMSNormKernelT() = default;
        ~CPURMSNormKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        // ===== ITensorRMSNorm interface =====
        bool apply(
            const float *input, const float *weight, float *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override; // Implemented in .cpp for mutable Q8_1 tensors

        bool apply_bf16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // Q8_1 kernel doesn't handle BF16
        }

        bool apply_fp16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // Q8_1 kernel doesn't handle FP16
        }

        bool apply_q8_1(
            const Q8_1Block *input, const float *weight, Q8_1Block *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            return apply_typed(input, weight, output, rows, cols, epsilon, device_idx);
        }

        // ===== Tensor-based API (type-checked dispatch) =====
        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RMS_NORM);
            (void)mpi_ctx;
            // Type check: this kernel only handles Q8_1
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::Q8_1 ||
                output->native_type() != TensorType::Q8_1)
                return false;
            auto *q8_in = static_cast<const Q8_1Tensor *>(input);
            auto *q8_out = static_cast<Q8_1Tensor *>(output);
            return apply_typed(q8_in->typed_data(), weight->data(), q8_out->mutable_typed_data(),
                               rows, cols, epsilon, device_idx);
        }

        // ===== Typed API =====
        /**
         * @brief Apply RMSNorm with Q8_1 input/output
         *
         * Algorithm:
         * 1. Dequantize Q8_1 blocks to FP32 (per-block scale × int8 values)
         * 2. Compute RMSNorm in FP32
         * 3. Quantize FP32 output to Q8_1 blocks (find scale, quantize)
         *
         * Note: cols must be a multiple of 32 (Q8_1 block size)
         */
        bool apply_typed(
            const Q8_1Block *input,
            const float *gamma,
            Q8_1Block *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        /**
         * @brief Apply RMSNorm with fused Q8_1 residual addition
         *
         * Optimized for the typed residual pattern where:
         * - residual is stored compressed as Q8_1
         * - fp32_input is the new contribution to add
         * - output is stored compressed as Q8_1
         */
        bool apply_with_residual_add(
            const Q8_1Block *residual,
            const float *fp32_input,
            const float *gamma,
            Q8_1Block *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        static constexpr ActivationPrecision precision() { return ActivationPrecision::Q8_1; }
        static const char *precision_name() { return "Q8_1"; }
        static constexpr float compression_ratio() { return 3.556f; }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rmsnorm()
                .withInput("input", "input activations [rows, cols]", KernelBufferDtype::Q8_1)
                .withWeight("gamma", "normalization weights [cols]", KernelBufferDtype::FP32)
                .withOutput("output", "normalized output [rows, cols]", KernelBufferDtype::Q8_1)
                .withScalar("epsilon", "numerical stability constant");
        }
    };

    // =========================================================================
    // Q16_1 Specialization (Q16_1 input → FP32 output)
    // =========================================================================

    /**
     * @brief Q16_1 specialization for typed residual pattern
     *
     * Special kernel for reading high-precision Q16_1 residuals and outputting FP32.
     * Used in the typed residual stream where:
     * - Residual is stored in Q16_1 format (72 bytes per 32 elements)
     * - Pre-attention/pre-FFN operations need FP32 input
     * - This kernel dequantizes on-the-fly during RMSNorm computation
     *
     * Note: This kernel ALWAYS outputs FP32, unlike other specializations
     * that output the same type as input.
     */
    template <>
    class CPURMSNormKernelT<ActivationPrecision::Q16_1> : public ITensorRMSNorm, public CPUKernelBase
    {
    public:
        using StorageType = Q16_1Block;

        CPURMSNormKernelT() = default;
        ~CPURMSNormKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        // ===== ITensorRMSNorm interface =====
        // Note: Q16_1 kernel primarily supports Q16_1→FP32 (apply_q16_1_to_fp32)
        // The other methods return false or delegate where possible

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
            return false; // Q16_1 kernel doesn't handle FP32→FP32
        }

        bool apply_bf16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // Q16_1 kernel doesn't handle BF16
        }

        bool apply_fp16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // Q16_1 kernel doesn't handle FP16
        }

        bool apply_q8_1(
            const Q8_1Block *input, const float *weight, Q8_1Block *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)device_idx;
            return false; // Q16_1 kernel doesn't handle Q8_1
        }

        bool apply_q16_1_to_fp32(
            const Q16_1Block *input, const float *weight, float *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
        {
            return apply_typed(input, weight, output, rows, cols, epsilon, device_idx);
        }

        // ===== Tensor-based API (type-checked dispatch) =====
        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RMS_NORM);
            (void)mpi_ctx;
            // Type check: Q16_1 input → FP32 output
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::Q16_1 ||
                output->native_type() != TensorType::FP32)
                return false;
            auto *q16_in = static_cast<const Q16_1Tensor *>(input);
            auto *fp32_out = static_cast<FP32Tensor *>(output);
            return apply_typed(q16_in->typed_data(), weight->data(), fp32_out->mutable_typed_data(),
                               rows, cols, epsilon, device_idx);
        }

        // ===== Typed API =====
        /**
         * @brief Apply RMSNorm with Q16_1 input, FP32 output
         *
         * Algorithm:
         * 1. Dequantize Q16_1 blocks to FP32 on-the-fly (per-block FP32 scale × int16 values)
         * 2. Compute RMSNorm in FP32
         * 3. Output is FP32 (no re-quantization)
         *
         * Note: cols must be a multiple of 32 (Q16_1 block size)
         */
        bool apply_typed(
            const Q16_1Block *input,
            const float *gamma,
            float *output,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            int device_idx = -1);

        static constexpr ActivationPrecision precision() { return ActivationPrecision::Q16_1; }
        static const char *precision_name() { return "Q16_1"; }
        static constexpr float compression_ratio() { return 1.778f; }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rmsnorm()
                .withInput("input", "input activations [rows, cols]", KernelBufferDtype::Q16_1)
                .withWeight("gamma", "normalization weights [cols]", KernelBufferDtype::FP32)
                .withOutput("output", "normalized output [rows, cols]", KernelBufferDtype::Q16_1)
                .withScalar("epsilon", "numerical stability constant");
        }
    };

} // namespace llaminar2
