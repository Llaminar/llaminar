#pragma once
/**
 * @file TensorBase_GPU.h
 * @brief Minimal GPU interface for CUDA compilation
 * @author David Sanftenberg
 *
 * This header provides the minimal TensorBase interface needed for CUDA code to access
 * GPU-related methods. Full Tensors.h cannot be included in .cu files because it pulls
 * in x86 SIMD intrinsics via SIMDHelpers.h that nvcc cannot compile.
 *
 * CUDA code should include this header instead of Tensors.h when only GPU data access
 * is needed. The interface matches TensorBase exactly for the GPU-related methods.
 *
 * IMPORTANT: This header defines TensorBase differently for CUDA vs CPU compilation:
 * - When __CUDACC__ is defined: Provides a minimal abstract interface
 * - When __CUDACC__ is NOT defined: Includes full Tensors.h
 *
 * This allows the same code to work both in CUDA (.cu) and CPU (.cpp) files.
 */

#ifndef __CUDACC__
// For non-CUDA compilation, just include the full Tensors.h
#include "Tensors.h"
#else
// For CUDA compilation, provide minimal interface

#include <vector>
#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Minimal TensorBase interface for CUDA compilation
     *
     * This provides only the methods that CUDA code typically needs to access.
     * The actual TensorBase class inherits from this interface (implicitly through
     * virtual method matching).
     */
    class TensorBase
    {
    public:
        virtual ~TensorBase() = default;

        // =====================================================================
        // GPU Data Access - the primary interface CUDA code needs
        // =====================================================================

        /**
         * @brief Get the raw device (GPU) data pointer
         * @return void* pointer to device memory, or nullptr if not on GPU
         */
        virtual void *gpu_data_ptr() = 0;
        virtual const void *gpu_data_ptr() const = 0;

        /**
         * @brief Check if tensor data is currently resident on GPU
         * @return true if GPU buffer is allocated and contains valid data
         */
        virtual bool isOnGPU() const = 0;

        // =====================================================================
        // Shape Access - commonly needed alongside GPU data
        // =====================================================================

        /**
         * @brief Get the shape of this tensor
         * @return Vector of dimension sizes
         */
        virtual const std::vector<size_t> &shape() const = 0;

        /**
         * @brief Get the total number of elements in the tensor
         * @return Product of all dimensions
         */
        virtual size_t numel() const = 0;

        /**
         * @brief Get the number of rows (first dimension)
         * @return shape()[0] or 1 if 1D
         */
        virtual size_t rows() const = 0;

        /**
         * @brief Get the number of columns (second dimension)
         * @return shape()[1] or numel() if 1D
         */
        virtual size_t cols() const = 0;

        /**
         * @brief Get the size of the tensor data in bytes
         * @return Total bytes for storage
         */
        virtual size_t size_bytes() const = 0;

        // =====================================================================
        // GPU Dirty Flag - track device modifications for lazy sync
        // =====================================================================

        /**
         * @brief Mark tensor as modified on device (requires sync to host before host access)
         * @note Call this after GPU kernels write to the tensor via gpu_data_ptr()
         */
        virtual void mark_device_dirty() = 0;
    };

} // namespace llaminar2

#endif // __CUDACC__
