/**
 * @file ROCmSwiGLUKernelT.cpp
 * @brief Implementation of ROCm SwiGLU kernel ITensorSwiGLU methods
 *
 * This file contains the C++ method implementations for ROCmSwiGLUKernelT.
 * It calls the extern "C" wrapper functions defined in ROCmSwiGLUKernels.hip.
 *
 * Separating .hip and .cpp allows hipcc to compile only the HIP code without
 * encountering issues with MPI or other complex C++ headers.
 *
 * @author Llaminar Team
 * @date 2026-01-17
 */

#include "ROCmSwiGLUKernelT.h"
#include "../../../tensors/Tensors.h"
#include "../../../backends/DeviceId.h"
#include "../../../utils/KernelProfiler.h"

#include <hip/hip_runtime.h>
#include <cstdio>

// =========================================================================
// Extern "C" declarations for HIP kernel wrappers
// =========================================================================

extern "C"
{
    // SwiGLU
    bool hipOps_swiglu_fp32(
        const float *gate, const float *up, float *output,
        int size, int device_idx);
    bool hipOps_swiglu_bf16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int size, int device_idx);
    bool hipOps_swiglu_fp16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int size, int device_idx);
}

namespace llaminar2
{
    namespace rocm
    {

        // =========================================================================
        // ROCmSwiGLUKernelT<FP32> Implementation
        // =========================================================================

        bool ROCmSwiGLUKernelT<ActivationPrecision::FP32>::apply(
            const float *gate, const float *up, float *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual; // TODO: implement residual addition
            (void)mpi_ctx;
            int size = rows * cols;
            return apply_typed(gate, up, output, size, device_idx);
        }

        bool ROCmSwiGLUKernelT<ActivationPrecision::FP32>::apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            KERNEL_PROFILE_SCOPE(KernelType::SWIGLU);
            (void)add_residual;
            (void)mpi_ctx;
            if (!gate || !up || !output)
                return false;
            if (gate->native_type() != TensorType::FP32 ||
                up->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Cast to FP32Tensor for GPU operations
            auto *gate_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(gate));
            auto *up_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(up));
            auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);

            if (!gate_fp32 || !up_fp32 || !output_fp32)
                return false;

            // Coherence handled automatically by GraphExecutor

            // Get device pointers
            const float *d_gate = static_cast<const float *>(gate_fp32->gpu_data_ptr());
            const float *d_up = static_cast<const float *>(up_fp32->gpu_data_ptr());
            float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());

            int size = rows * cols;
            // No sync needed - coherence system handles sync when data is read
            return hipOps_swiglu_fp32(d_gate, d_up, d_output, size, dev);
        }

        bool ROCmSwiGLUKernelT<ActivationPrecision::FP32>::apply_typed(
            const float *gate,
            const float *up,
            float *output,
            int size,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            // No sync needed - coherence system handles sync when data is read
            return hipOps_swiglu_fp32(gate, up, output, size, dev);
        }

        // =========================================================================
        // ROCmSwiGLUKernelT<BF16> Implementation
        // =========================================================================

        bool ROCmSwiGLUKernelT<ActivationPrecision::BF16>::apply_bf16(
            const uint16_t *gate, const uint16_t *up, uint16_t *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            int size = rows * cols;
            return apply_typed(gate, up, output, size, device_idx);
        }

        bool ROCmSwiGLUKernelT<ActivationPrecision::BF16>::apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            KERNEL_PROFILE_SCOPE(KernelType::SWIGLU);
            (void)add_residual;
            (void)mpi_ctx;
            if (!gate || !up || !output)
                return false;
            if (gate->native_type() != TensorType::BF16 ||
                up->native_type() != TensorType::BF16 ||
                output->native_type() != TensorType::BF16)
                return false;

            auto *gate_bf16 = const_cast<BF16Tensor *>(dynamic_cast<const BF16Tensor *>(gate));
            auto *up_bf16 = const_cast<BF16Tensor *>(dynamic_cast<const BF16Tensor *>(up));
            auto *out_bf16 = dynamic_cast<BF16Tensor *>(output);
            if (!gate_bf16 || !up_bf16 || !out_bf16)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by GraphExecutor

            // Get device pointers
            const uint16_t *d_gate = static_cast<const uint16_t *>(gate_bf16->gpu_data_ptr());
            const uint16_t *d_up = static_cast<const uint16_t *>(up_bf16->gpu_data_ptr());
            uint16_t *d_output = static_cast<uint16_t *>(out_bf16->gpu_data_ptr());

            int size = rows * cols;
            // No sync needed - coherence system handles sync when data is read
            return hipOps_swiglu_bf16(d_gate, d_up, d_output, size, dev);
        }

        bool ROCmSwiGLUKernelT<ActivationPrecision::BF16>::apply_typed(
            const uint16_t *gate,
            const uint16_t *up,
            uint16_t *output,
            int size,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            // No sync needed - coherence system handles sync when data is read
            return hipOps_swiglu_bf16(gate, up, output, size, dev);
        }

        // =========================================================================
        // ROCmSwiGLUKernelT<FP16> Implementation
        // =========================================================================

        bool ROCmSwiGLUKernelT<ActivationPrecision::FP16>::apply_fp16(
            const uint16_t *gate, const uint16_t *up, uint16_t *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            int size = rows * cols;
            return apply_typed(gate, up, output, size, device_idx);
        }

        bool ROCmSwiGLUKernelT<ActivationPrecision::FP16>::apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            KERNEL_PROFILE_SCOPE(KernelType::SWIGLU);
            (void)add_residual;
            (void)mpi_ctx;
            if (!gate || !up || !output)
                return false;
            if (gate->native_type() != TensorType::FP16 ||
                up->native_type() != TensorType::FP16 ||
                output->native_type() != TensorType::FP16)
                return false;

            auto *gate_fp16 = const_cast<FP16Tensor *>(dynamic_cast<const FP16Tensor *>(gate));
            auto *up_fp16 = const_cast<FP16Tensor *>(dynamic_cast<const FP16Tensor *>(up));
            auto *out_fp16 = dynamic_cast<FP16Tensor *>(output);
            if (!gate_fp16 || !up_fp16 || !out_fp16)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by GraphExecutor

            // Get device pointers
            const uint16_t *d_gate = static_cast<const uint16_t *>(gate_fp16->gpu_data_ptr());
            const uint16_t *d_up = static_cast<const uint16_t *>(up_fp16->gpu_data_ptr());
            uint16_t *d_output = static_cast<uint16_t *>(out_fp16->gpu_data_ptr());

            int size = rows * cols;
            // No sync needed - coherence system handles sync when data is read
            return hipOps_swiglu_fp16(d_gate, d_up, d_output, size, dev);
        }

        bool ROCmSwiGLUKernelT<ActivationPrecision::FP16>::apply_typed(
            const uint16_t *gate,
            const uint16_t *up,
            uint16_t *output,
            int size,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            // No sync needed - coherence system handles sync when data is read
            return hipOps_swiglu_fp16(gate, up, output, size, dev);
        }

    } // namespace rocm
} // namespace llaminar2
