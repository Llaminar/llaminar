/**
 * @file SwiGLUOp.h
 * @brief Self-validating SwiGLU activation operation
 *
 * SwiGLU is used in FFN blocks: output = silu(gate) * up
 * Where silu(x) = x * sigmoid(x)
 *
 * Per HuggingFace: down_proj(act_fn(gate_proj(x)) * up_proj(x))
 *
 * Encapsulates the full SwiGLU workflow:
 * 1. Validate gate/up/output tensors
 * 2. Create SwiGLU kernel from activation tensor
 * 3. Execute kernel with error handling
 * 4. Capture snapshot (if enabled)
 *
 * Usage:
 * @code
 * SwiGLUOp swiglu;
 *
 * // Apply SwiGLU (output in-place in up buffer)
 * TRY_OP(swiglu(gate, up, up, rows, cols, "layer0_FFN_SWIGLU", mpi, device));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "../../kernels/cpu/ops/CPUSwiGLUKernelT.h"

namespace llaminar2
{

    /**
     * @brief Self-validating SwiGLU activation operation
     *
     * Qwen2 SwiGLU formulation: output = silu(gate) * up
     * Per HuggingFace: down_proj(act_fn(gate_proj(x)) * up_proj(x))
     * - gate: Linear projection output, gets silu activation
     * - up: Linear projection output (no activation)
     * - output: Element-wise product
     *
     * Replaces:
     * @code
     * static CPUSwiGLUKernelT<FP32Tensor> swiglu_kernel;
     * VALIDATE_OP(swiglu_kernel.apply(gate->data(), up->data(), output->mutable_data(),
     *             rows, cols, false, mpi, device), "SwiGLU");
     * CAPTURE_SNAPSHOT_VIEW("FFN_SWIGLU", output, rows, cols);
     * @endcode
     *
     * With:
     * @code
     * TRY_OP(swiglu(gate, up, output, rows, cols, "FFN_SWIGLU", mpi, device));
     * @endcode
     */
    class SwiGLUOp : public OpBase
    {
    public:
        const char *name() const override { return "SwiGLUOp"; }

        /**
         * @brief Execute SwiGLU activation: output = silu(gate) * up
         *
         * @param gate Gate tensor [rows, cols] - gets silu activation
         * @param up Up tensor [rows, cols] - linear term
         * @param output Output tensor [rows, cols] - can be same as up for in-place
         * @param rows Number of rows (sequence length)
         * @param cols Number of columns (FFN intermediate size)
         * @param snapshot_key Snapshot identifier (nullptr to skip)
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on validation or execution failure
         */
        bool operator()(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows,
            int cols,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // 1. Validate inputs
            if (!validateTensor(gate, "gate"))
                return false;
            if (!validateTensor(up, "up"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "SwiGLU"))
                return false;

            // 2. Execute SwiGLU kernel
            // Note: Using static kernel instance (stateless, thread-safe)
            if (!swiglu_kernel_.apply(
                    gate->data(),
                    up->data(),
                    output->mutable_data(),
                    rows, cols,
                    false, // no residual
                    mpi_ctx,
                    device_idx))
            {
                logError("SwiGLU kernel execution failed");
                return false;
            }

            // Note: Snapshot capture is handled by the calling pipeline
            (void)snapshot_key;

            return true;
        }

        /**
         * @brief Execute SwiGLU with raw float pointers
         */
        bool operator()(
            const float *gate_data,
            const float *up_data,
            float *output_data,
            int rows,
            int cols,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // 1. Validate inputs
            if (!validatePointer(gate_data, "gate data"))
                return false;
            if (!validatePointer(up_data, "up data"))
                return false;
            if (!validatePointer(output_data, "output data"))
                return false;
            if (!validateDimensions(rows, cols, "SwiGLU"))
                return false;

            // 2. Execute
            if (!swiglu_kernel_.apply(
                    gate_data,
                    up_data,
                    output_data,
                    rows, cols,
                    false,
                    mpi_ctx,
                    device_idx))
            {
                logError("SwiGLU kernel execution failed");
                return false;
            }

            return true;
        }

    private:
        CPUSwiGLUKernelT<FP32Tensor> swiglu_kernel_;
    };

} // namespace llaminar2
