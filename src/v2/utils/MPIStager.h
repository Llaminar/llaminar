/**
 * @file MPIStager.h
 * @brief MPI host staging utilities for GPU↔Host transfers
 *
 * **Purpose**: All MPI collectives operate on host memory. When tensors reside
 * on GPU devices, we must stage them to host before MPI operations and back to
 * device after.
 *
 * **Phase 2 Objective**: Eliminate "MPI operations on device memory" bugs by
 * enforcing explicit staging at call sites.
 *
 * **Usage Pattern**:
 * ```cpp
 * // Before MPI collective (stage GPU→Host)
 * auto host_buffer = MPIStager::toHost(gpu_tensor);
 *
 * // MPI operation on host memory
 * MPI_Allreduce(host_buffer.data(), result.data(), count, MPI_FLOAT, MPI_SUM, comm);
 *
 * // After MPI collective (stage Host→GPU)
 * MPIStager::toDevice(result, gpu_tensor);
 * ```
 *
 * @author David Sanftenberg
 */

#pragma once

#include <vector>
#include <memory>
#include <mpi.h>
#include "../tensors/Tensors.h"
#include "../utils/MPIContext.h"

namespace llaminar2
{

    /**
     * @class MPIStager
     * @brief Utilities for staging tensors between device and host for MPI operations
     *
     * **Design Rationale**:
     * - MPI libraries (OpenMPI, MPICH) do NOT support device pointers by default
     * - Even with CUDA-aware MPI, explicit staging is safer and more portable
     * - Avoids subtle bugs where MPI reads garbage from device memory
     *
     * **Performance Characteristics**:
     * - Small tensors (<1MB): Staging overhead <5%
     * - Large tensors (>10MB): Staging amortized by network latency
     * - Batched staging (future): Reduce memcpy calls
     */
    class MPIStager
    {
    public:
        /**
         * @brief Stage tensor from device to host memory for MPI operation
         *
         * @param tensor Input tensor (may be CPU or GPU resident)
         * @return Host-resident buffer with tensor data
         *
         * **Behavior**:
         * - If tensor->device_id() < 0: No-op, return direct pointer
         * - If tensor->device_id() >= 0: cudaMemcpy/hipMemcpy D2H
         *
         * **Thread Safety**: Safe (no shared state)
         * **MPI Context**: Independent of MPI rank
         */
        static std::vector<float> toHost(const TensorBase *tensor);

        /**
         * @brief Stage host buffer back to device tensor after MPI operation
         *
         * @param host_buffer Host memory with MPI result
         * @param tensor Output tensor (must match original shape)
         *
         * **Behavior**:
         * - If tensor->device_id() < 0: Memcpy to CPU tensor data
         * - If tensor->device_id() >= 0: cudaMemcpy/hipMemcpy H2D
         *
         * **Preconditions**:
         * - host_buffer.size() == tensor->numel()
         * - tensor must be pre-allocated
         *
         * **Thread Safety**: Safe (no shared state)
         */
        static void toDevice(const std::vector<float> &host_buffer, TensorBase *tensor);

        /**
         * @brief Check if tensor requires staging (is GPU-resident)
         *
         * @param tensor Tensor to check
         * @return true if tensor is on GPU (requires staging)
         */
        static bool requiresStaging(const TensorBase *tensor);

        /**
         * @brief Synchronize device before MPI operation
         *
         * @param device_id GPU device ID (-1 for CPU)
         *
         * **Purpose**: Ensure all device kernels complete before staging
         * **Behavior**:
         * - If device_id < 0: No-op
         * - If HAVE_CUDA: cudaDeviceSynchronize()
         * - If HAVE_ROCM: hipDeviceSynchronize()
         */
        static void synchronizeDevice(int device_id);

        // ====================================================================
        // Typed staging variants (for future BF16, INT8, etc.)
        // ====================================================================

        /**
         * @brief Stage BF16 tensor to FP32 host buffer
         *
         * @param tensor BF16 tensor (GPU or CPU)
         * @return FP32 host buffer for MPI operation
         *
         * **Note**: MPI_BFLOAT16 not widely supported, so we upcast
         */
        static std::vector<float> toHostBF16(const TensorBase *tensor);

        /**
         * @brief Stage FP32 host buffer back to BF16 device tensor
         *
         * @param host_buffer FP32 MPI result
         * @param tensor BF16 tensor (GPU or CPU)
         *
         * **Note**: Downcasts FP32→BF16 with rounding
         */
        static void toDeviceBF16(const std::vector<float> &host_buffer, TensorBase *tensor);

    private:
        /**
         * @brief GPU→Host memcpy implementation
         *
         * @param dst Host destination
         * @param src Device source
         * @param count Number of floats
         * @param device_id GPU device ID
         */
        static void deviceToHost(float *dst, const float *src, size_t count, int device_id);

        /**
         * @brief Host→GPU memcpy implementation
         *
         * @param dst Device destination
         * @param src Host source
         * @param count Number of floats
         * @param device_id GPU device ID
         */
        static void hostToDevice(float *dst, const float *src, size_t count, int device_id);
    };

} // namespace llaminar2
