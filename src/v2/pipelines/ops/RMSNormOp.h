/**
 * @file RMSNormOp.h
 * @brief Self-validating RMS Normalization operation
 *
 * Encapsulates the full RMSNorm workflow:
 * 1. Validate input/weight/output tensors
 * 2. Create RMSNorm kernel from activation tensor
 * 3. Execute kernel with error handling
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
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"

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

            // 3. Execute kernel
            if (!kernel->apply(
                    input->data(),
                    weight->data(),
                    output->mutable_data(),
                    rows, cols,
                    eps,
                    false, // use_bf16
                    mpi_ctx,
                    device_idx))
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
