/**
 * @file RMSNormOp.h
 * @brief Self-validating RMS Normalization operation
 *
 * Encapsulates the full RMSNorm workflow:
 * 1. Validate input/weight/output tensors
 * 2. Create RMSNorm kernel from activation tensor
 * 3. Execute kernel with error handling (using native precision)
 * 4. Capture snapshot (if enabled)
 *
 * Usage:
 * @code
 * RMSNormOp rmsnorm;
 *
 * // Full signature with snapshot
 * if (!rmsnorm(input, weight, output, rows, cols, eps, "layer0_ATTN_NORM", mpi, device))
 *     return false;
 *
 * // Minimal signature (no snapshot)
 * if (!rmsnorm(input, weight, output, rows, cols))
 *     return false;
 * @endcode
 *
 * Supports native precision execution:
 * - FP32Tensor: apply() with float*
 * - BF16Tensor: apply_bf16() with uint16_t*
 * - FP16Tensor: apply_fp16() with uint16_t*
 * - Q8_1Tensor: apply_q8_1() with Q8_1Block*
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "../../tensors/Tensors.h" // For FP16Tensor, BF16Tensor, Q8_1Tensor

namespace llaminar2
{

    /**
     * @brief Self-validating RMS Normalization operation
     *
     * Replaces the verbose pattern:
     * @code
     * auto *activation = dynamic_cast<IActivationTensor*>(output.get());
     * VALIDATE_POINTER(activation, "activation tensor");
     * auto kernel = activation->createRMSNorm();
     * VALIDATE_POINTER(kernel, "RMSNorm kernel");
     * VALIDATE_OP(kernel->apply(...), "RMSNorm");
     * CAPTURE_SNAPSHOT_VIEW("stage", output, rows, cols);
     * @endcode
     *
     * With a single call:
     * @code
     * TRY_OP(rmsnorm(input, weight, output, rows, cols, eps, "stage", mpi, device));
     * @endcode
     */
    class RMSNormOp : public OpBase
    {
    public:
        const char *name() const override { return "RMSNormOp"; }

        /**
         * @brief Execute RMS normalization
         *
         * @param input Input tensor data (source for normalization)
         * @param weight Gamma weights [cols] (scale parameters)
         * @param output Output tensor (must be IActivationTensor for kernel creation)
         * @param rows Number of rows (sequence length)
         * @param cols Number of columns (model dimension)
         * @param eps Epsilon for numerical stability (default: 1e-6f)
         * @param snapshot_key Snapshot identifier (nullptr to skip snapshot)
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on validation or execution failure
         */
        bool operator()(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows,
            int cols,
            float eps = 1e-6f,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // 1. Validate inputs
            if (!validateTensor(input, "input"))
                return false;
            if (!validateTensor(weight, "weight"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "RMSNorm"))
                return false;

            // 2. Create kernel from output tensor (activation tensor creates the kernel)
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

            // 3. Execute kernel using native precision based on output tensor type
            bool success = false;
            const TensorType out_type = output->native_type();

            switch (out_type)
            {
            case TensorType::FP32:
                // FP32: Use standard apply() with float*
                success = kernel->apply(
                    input->data(),
                    weight->data(),
                    output->mutable_data(),
                    rows, cols,
                    eps,
                    false, // use_bf16
                    mpi_ctx,
                    device_idx);
                break;

            case TensorType::BF16:
            {
                // BF16: Use native apply_bf16() with uint16_t*
                auto *bf16_input = dynamic_cast<const BF16Tensor *>(input);
                auto *bf16_output = dynamic_cast<BF16Tensor *>(output);
                if (!bf16_input || !bf16_output)
                {
                    logError("BF16 RMSNorm requires BF16 input and output tensors");
                    return false;
                }
                success = kernel->apply_bf16(
                    bf16_input->bf16_data(),
                    weight->data(),
                    bf16_output->mutable_bf16_data(),
                    rows, cols, eps, device_idx);
                break;
            }

            case TensorType::FP16:
            {
                // FP16: Use native apply_fp16() with uint16_t*
                auto *fp16_input = dynamic_cast<const FP16Tensor *>(input);
                auto *fp16_output = dynamic_cast<FP16Tensor *>(output);
                if (!fp16_input || !fp16_output)
                {
                    logError("FP16 RMSNorm requires FP16 input and output tensors");
                    return false;
                }
                success = kernel->apply_fp16(
                    fp16_input->fp16_data(),
                    weight->data(),
                    fp16_output->mutable_fp16_data(),
                    rows, cols, eps, device_idx);
                break;
            }

            case TensorType::Q8_1:
            {
                // Q8_1: Use native apply_q8_1() with Q8_1Block*
                auto *q8_1_input = dynamic_cast<const Q8_1Tensor *>(input);
                auto *q8_1_output = dynamic_cast<Q8_1Tensor *>(output);
                if (!q8_1_input || !q8_1_output)
                {
                    logError("Q8_1 RMSNorm requires Q8_1 input and output tensors");
                    return false;
                }
                success = kernel->apply_q8_1(
                    q8_1_input->q8_1_blocks(),
                    weight->data(),
                    q8_1_output->mutable_q8_1_blocks(),
                    rows, cols, eps, device_idx);
                break;
            }

            default:
                logError("unsupported output tensor type for RMSNorm");
                return false;
            }

            if (!success)
            {
                logError("kernel execution failed");
                return false;
            }

            // Note: Snapshot capture is handled by the calling pipeline
            // The snapshot_key parameter is reserved for future use
            (void)snapshot_key;

            return true;
        }

        /**
         * @brief Execute RMS normalization with raw float pointers
         *
         * Alternative signature for cases where input is not a TensorBase
         * (e.g., using buffers from another location).
         *
         * Note: This signature only supports FP32 data.
         */
        bool operator()(
            const float *input_data,
            const float *weight_data,
            TensorBase *output,
            int rows,
            int cols,
            float eps = 1e-6f,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
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

            // 3. Execute kernel
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

            // Note: Snapshot capture is handled by the calling pipeline
            (void)snapshot_key;

            return true;
        }
    };

} // namespace llaminar2
