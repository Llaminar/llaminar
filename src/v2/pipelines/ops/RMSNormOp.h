/**
 * @file RMSNormOp.h
 * @brief Typed self-validating RMS Normalization operation
 *
 * Template-based operation supporting all activation precisions (FP32, BF16, FP16, Q8_1).
 * Uses compile-time dispatch via ActivationPrecision enum for zero-overhead precision handling.
 *
 * Encapsulates the full RMSNorm workflow:
 * 1. Validate input/weight/output tensors
 * 2. Create RMSNorm kernel from activation tensor
 * 3. Execute kernel with precision-appropriate paths
 * 4. Capture snapshot (if enabled)
 *
 * Supports native precision execution:
 * - FP32Tensor: apply() with float*
 * - BF16Tensor: apply_bf16() with uint16_t*
 * - FP16Tensor: apply_fp16() with uint16_t*
 * - Q8_1Tensor: apply_q8_1() with Q8_1Block*
 *
 * Usage:
 * @code
 * // Create typed op at initialization
 * auto rmsnorm = createRMSNormOp(ActivationPrecision::Q8_1);
 *
 * // Execute (polymorphic call, type-specific implementation)
 * TRY_OP(rmsnorm->execute(input, weight, output, rows, cols, eps, nullptr, device));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "GemmOp.h" // For detail::PrecisionTensor
#include "../RuntimeConfig.h"
#include "../../tensors/Tensors.h"
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // IRMSNormOp Interface
    // =========================================================================

    /**
     * @brief Interface for typed RMSNorm operations
     *
     * Provides polymorphic access to precision-specific RMSNorm implementations.
     * Use createRMSNormOp() factory to create instances at runtime.
     */
    class IRMSNormOp
    {
    public:
        virtual ~IRMSNormOp() = default;

        /**
         * @brief Get the activation precision this op was created for
         */
        virtual ActivationPrecision precision() const = 0;

        /**
         * @brief Execute RMS normalization
         *
         * @param input Input tensor [rows, cols]
         * @param weight Gamma weights [cols]
         * @param output Output tensor [rows, cols] (can be same as input)
         * @param rows Number of rows (sequence length)
         * @param cols Number of columns (model dimension)
         * @param eps Epsilon for numerical stability
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on failure
         */
        virtual bool execute(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute RMS normalization with raw float pointers
         *
         * Note: Only supports FP32 data.
         */
        virtual bool execute_raw(
            const float *input_data,
            const float *weight_data,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    // =========================================================================
    // RMSNormOpTyped<Precision> Template Implementation
    // =========================================================================

    /**
     * @brief Typed RMSNorm operation with compile-time precision dispatch
     *
     * @tparam Precision Activation precision (FP32, BF16, FP16, Q8_1)
     */
    template <ActivationPrecision Precision>
    class RMSNormOpTyped : public IRMSNormOp, public OpBase
    {
    public:
        using TensorT = typename detail::PrecisionTensor<Precision>::Type;
        using ElementType = typename detail::PrecisionTensor<Precision>::ElementType;

        const char *name() const override { return "RMSNormOpTyped"; }
        ActivationPrecision precision() const override { return Precision; }

        /**
         * @brief Get the expected TensorType for this precision
         */
        static constexpr TensorType expected_tensor_type()
        {
            if constexpr (Precision == ActivationPrecision::FP32)
                return TensorType::FP32;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            else
                return TensorType::FP32;
        }

        bool execute(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // Compile-time validation: ensure TensorT trait is defined
            static_assert(sizeof(TensorT) > 0,
                          "PrecisionTensor trait must be defined for this ActivationPrecision");

            // 1. Validate inputs
            if (!validateTensor(input, "input"))
                return false;
            if (!validateTensor(weight, "weight"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "RMSNorm"))
                return false;

            // 2. Type validation: ensure input/output match expected precision
            constexpr TensorType expected = expected_tensor_type();
            if (input->native_type() != expected)
            {
                logError(("input tensor type mismatch: expected " +
                          std::to_string(static_cast<int>(expected)) + ", got " +
                          std::to_string(static_cast<int>(input->native_type())))
                             .c_str());
                return false;
            }
            if (output->native_type() != expected)
            {
                logError(("output tensor type mismatch: expected " +
                          std::to_string(static_cast<int>(expected)) + ", got " +
                          std::to_string(static_cast<int>(output->native_type())))
                             .c_str());
                return false;
            }

            // 3. Create kernel from output tensor
            auto *activation = dynamic_cast<IActivationTensor *>(output);
            if (!activation)
            {
                logError("output tensor must be IActivationTensor");
                return false;
            }

            auto kernel = activation->createRMSNorm();
            if (!kernel)
            {
                logError("failed to create RMSNorm kernel");
                return false;
            }

            // 4. Delegate to kernel's apply_tensor - handles all type dispatch internally
            if (!kernel->apply_tensor(input, weight, output, rows, cols, eps, mpi_ctx, device_idx))
            {
                logError("kernel execution failed");
                return false;
            }

            return true;
        }

        bool execute_raw(
            const float *input_data,
            const float *weight_data,
            TensorBase *output,
            int rows, int cols,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // 1. Validate inputs
            if (!validatePointer(input_data, "input data"))
                return false;
            if (!validatePointer(weight_data, "weight data"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "RMSNorm"))
                return false;

            // This overload only supports FP32 output
            if (output->native_type() != TensorType::FP32)
            {
                logError("raw float pointer overload only supports FP32 output tensor");
                return false;
            }

            // 2. Create kernel from output tensor
            auto *activation = dynamic_cast<IActivationTensor *>(output);
            if (!activation)
            {
                logError("output tensor must be IActivationTensor");
                return false;
            }

            auto kernel = activation->createRMSNorm();
            if (!kernel)
            {
                logError("failed to create RMSNorm kernel");
                return false;
            }

            // 3. Execute
            if (!kernel->apply(
                    input_data,
                    weight_data,
                    output->mutable_data(),
                    rows, cols,
                    eps,
                    false,
                    mpi_ctx,
                    device_idx))
            {
                logError("kernel execution failed");
                return false;
            }

            return true;
        }
    };

    // =========================================================================
    // Factory Function
    // =========================================================================

    /**
     * @brief Create typed RMSNorm op for given precision
     *
     * @param precision Activation precision
     * @return Unique pointer to precision-specific RMSNorm op
     */
    inline std::unique_ptr<IRMSNormOp> createRMSNormOp(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::Q8_1:
            return std::make_unique<RMSNormOpTyped<ActivationPrecision::Q8_1>>();
        case ActivationPrecision::BF16:
            return std::make_unique<RMSNormOpTyped<ActivationPrecision::BF16>>();
        case ActivationPrecision::FP16:
            return std::make_unique<RMSNormOpTyped<ActivationPrecision::FP16>>();
        case ActivationPrecision::FP32:
        default:
            return std::make_unique<RMSNormOpTyped<ActivationPrecision::FP32>>();
        }
    }

} // namespace llaminar2
