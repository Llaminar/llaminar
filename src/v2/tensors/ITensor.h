/**
 * @file ITensor.h
 * @brief Runtime tensor interface for polymorphic tensor access
 * @author David Sanftenberg
 *
 * This file defines the ITensor interface - a pure runtime polymorphism layer that enables:
 * 1. Type-safe downcasting via typed_as<T>() and try_as<T>()
 * 2. Raw data access when type is known at runtime
 * 3. Runtime type introspection via native_type()
 *
 * Design philosophy:
 * - ITensor is the minimal runtime interface for heterogeneous tensor collections
 * - TensorBase extends ITensor with common functionality (shape, device, kernel creation)
 * - Concrete tensors can provide type-safe data() via CRTP (TypedTensorBase) or virtual methods
 *
 * Usage patterns:
 *
 * 1. Runtime dispatch with type-safe access:
 *    @code
 *    void process(ITensor& tensor) {
 *        if (auto* fp32 = tensor.try_as<FP32Tensor>()) {
 *            float* data = fp32->data();  // Type-safe!
 *        } else if (auto* bf16 = tensor.try_as<BF16Tensor>()) {
 *            uint16_t* data = bf16->native_data();  // Type-safe!
 *        }
 *    }
 *    @endcode
 *
 * 2. Known type access (asserts on mismatch):
 *    @code
 *    void process_fp32(ITensor& tensor) {
 *        auto& fp32 = tensor.typed_as<FP32Tensor>();  // Asserts if not FP32
 *        float* data = fp32.data();
 *    }
 *    @endcode
 *
 * 3. Generic algorithms with visitor pattern:
 *    @code
 *    template<typename Func>
 *    void visit_tensor(ITensor& tensor, Func&& func);  // See Tensors.h
 *    @endcode
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <memory>
#include <string>
#include <stdexcept>
#include "TensorType.h"           // For TensorType enum (CUDA-safe, no SIMD)
#include "../backends/DeviceId.h" // Type-safe device identification

namespace llaminar2
{
    // Forward declarations for typed_as/try_as
    class FP32Tensor;
    class BF16Tensor;
    class FP16Tensor;
    class INT8Tensor;
    class INT32Tensor;
    class Q8_0Tensor;
    class Q8_1Tensor;
    class Q4_0Tensor;
    class Q4_1Tensor;
    class Q5_0Tensor;
    class Q5_1Tensor;
    class Q6_KTensor;
    class IQ4_NLTensor;

    /**
     * @brief Runtime tensor interface for polymorphic tensor access
     *
     * ITensor provides the minimal interface needed for runtime polymorphism:
     * - Type introspection (native_type)
     * - Shape and size queries
     * - Raw data access (for when caller knows the type)
     * - Type-safe downcasting helpers
     *
     * This interface is intentionally minimal. TensorBase extends it with
     * device management, kernel creation, and format conversion.
     */
    class ITensor
    {
    public:
        virtual ~ITensor() = default;

        // =========================================================================
        // Type Information
        // =========================================================================

        /**
         * @brief Get the native storage type of this tensor
         * @return TensorType enum indicating the tensor's data format
         * @note Forward declared here, defined in Tensors.h (avoids circular dependency)
         */
        virtual int native_type_id() const = 0;

        /**
         * @brief Get the native storage type as TensorType enum
         * @return TensorType enum indicating the tensor's data format
         * @note Convenience method - derived from native_type_id()
         */
        TensorType native_type() const
        {
            return static_cast<TensorType>(native_type_id());
        }

        /**
         * @brief Get human-readable dtype name
         * @return String like "FP32", "Q8_0", "IQ4_NL", etc.
         * @note Convenience method - uses tensorTypeName()
         */
        const char *dtype_name() const
        {
            return tensorTypeName(native_type());
        }

        /**
         * @brief Get the shape of this tensor
         * @return Reference to shape vector (dimensions)
         */
        virtual const std::vector<size_t> &shape() const = 0;

        /**
         * @brief Get number of rows (first dimension)
         * @return shape()[0], or 1 if shape is empty
         * @note Convenience method for 2D matrix access pattern
         */
        size_t rows() const
        {
            const auto &s = shape();
            return s.empty() ? 1 : s[0];
        }

        /**
         * @brief Get number of columns (second dimension)
         * @return shape()[1], or 1 if shape has fewer than 2 dimensions
         * @note Convenience method for 2D matrix access pattern
         */
        size_t cols() const
        {
            const auto &s = shape();
            return s.size() < 2 ? 1 : s[1];
        }

        /**
         * @brief Get total number of logical elements
         * @return Product of all dimensions
         */
        virtual size_t numel() const = 0;

        /**
         * @brief Get size in bytes of the raw data buffer
         * @return Number of bytes for the underlying storage
         * @note For quantized tensors, this is the packed block size, not numel * sizeof(element)
         */
        virtual size_t size_bytes() const = 0;

        // =========================================================================
        // Device Location API (Type-Safe)
        // =========================================================================

        /**
         * @brief Get the device where this tensor's data resides
         * @return DeviceId identifying CPU, CUDA, or ROCm device
         * @note This is the PRIMARY device API - prefer over legacy home_dm_device_index()
         */
        virtual DeviceId home_device() const = 0;

        // =========================================================================
        // Device Location Convenience Methods
        // =========================================================================

        /**
         * @brief Check if tensor data is on CPU (host memory)
         * @return true if tensor's primary data is on CPU
         */
        virtual bool is_on_cpu() const
        {
            return home_device().is_cpu();
        }

        /**
         * @brief Check if tensor data is on GPU (device memory)
         * @return true if tensor's primary data is on a GPU device
         */
        virtual bool is_on_gpu() const
        {
            return home_device().is_gpu();
        }

        // =========================================================================
        // FP32 Data Access
        // =========================================================================

        /**
         * @brief Get const pointer to FP32 data on host
         * @return Const pointer to float data, or nullptr if not available
         * @note For CPU tensors, returns host pointer directly
         * @note For GPU tensors, may sync to host or return nullptr
         * @note For non-FP32 tensors, may throw or return cached dequantized data
         */
        virtual const float *data() const = 0;

        /**
         * @brief Get mutable pointer to FP32 data on host
         * @return Mutable pointer to float data, or nullptr if not supported
         * @note For CPU activation tensors, returns mutable host pointer
         * @note For GPU tensors or read-only weight tensors, may return nullptr or throw
         */
        virtual float *mutable_data() = 0;

        /**
         * @brief Explicit FP32 dequantization accessor
         *
         * For most tensor types, this is equivalent to data(). For quantized tensors
         * like Q8_1Tensor, this performs explicit dequantization while data() may throw
         * to prevent accidental implicit dequantization.
         *
         * Use this method when you INTENTIONALLY need FP32 data from a quantized tensor
         * (e.g., for snapshot capture, debugging, or interop with FP32-only code).
         *
         * @return Const pointer to FP32 data (may be cached/lazily computed)
         * @note Default implementation returns data() - overridden by quantized tensors
         */
        virtual const float *fp32_data() const { return data(); }

        // =========================================================================
        // Raw Data Access (type-unsafe, use with caution)
        // =========================================================================

        /**
         * @brief Get raw pointer to HOST data buffer (const)
         * @return Pointer to start of HOST data buffer, caller must know the actual type
         * @note Use typed_as<T>() for type-safe access when possible
         * @note Always returns HOST pointer even if tensor is on GPU
         * @see active_data_ptr() for device-aware pointer access
         */
        virtual const void *raw_data() const = 0;

        /**
         * @brief Get raw mutable pointer to HOST data buffer
         * @return Mutable pointer to start of HOST data buffer
         * @note Use typed_as<T>() for type-safe access when possible
         * @note Always returns HOST pointer even if tensor is on GPU
         * @see active_mutable_data_ptr() for device-aware pointer access
         */
        virtual void *raw_mutable_data() = 0;

        // =========================================================================
        // Device-Aware Data Access
        // =========================================================================

        /**
         * @brief Get pointer to data where it currently resides (const)
         * @return GPU pointer if is_on_gpu(), else host pointer from raw_data()
         * @note Use this when you need the "active" pointer for kernel dispatch
         * @note Default implementation returns raw_data() - override in GPU-capable tensors
         */
        virtual const void *active_data_ptr() const { return raw_data(); }

        /**
         * @brief Get mutable pointer to data where it currently resides
         * @return GPU pointer if is_on_gpu(), else host pointer from raw_mutable_data()
         * @note Use this when you need the "active" pointer for kernel dispatch
         * @note Default implementation returns raw_mutable_data() - override in GPU-capable tensors
         */
        virtual void *active_mutable_data_ptr() { return raw_mutable_data(); }

        // =========================================================================
        // GPU Data Access
        // =========================================================================

        /**
         * @brief Get const pointer to GPU device memory
         * @return Pointer to GPU buffer, or nullptr if tensor has no GPU allocation
         * @note Returns nullptr if ensureOnDevice() has not been called
         */
        virtual const void *gpu_data_ptr() const = 0;

        /**
         * @brief Get mutable pointer to GPU device memory
         * @return Pointer to GPU buffer, or nullptr if tensor has no GPU allocation
         * @note Returns nullptr if ensureOnDevice() has not been called
         */
        virtual void *gpu_data_ptr() = 0;

        // =========================================================================
        // Device Coherence API
        // =========================================================================

        /**
         * @brief Mark tensor as modified on device (GPU data is current, host is stale)
         * @note Call this after GPU kernels write to the tensor via gpu_data_ptr()
         *
         * After calling this:
         *   - GPU data is considered current
         *   - Host data is considered stale
         *   - Next data() call will sync from device
         */
        virtual void mark_device_dirty() {}

        /**
         * @brief Mark tensor as modified on host (host data is current, GPU is stale)
         * @note Call this after CPU code writes to the tensor via mutable_data()
         *
         * After calling this:
         *   - Host data is considered current
         *   - GPU data is considered stale (if any)
         *   - Next ensureOnDevice() will sync to device
         */
        virtual void mark_host_dirty() {}

        /**
         * @brief Ensure tensor data is present on the specified device
         * @param target_device Device to ensure data is on
         * @return true if data is now available on target device
         * @note For CPU tensors, this may upload to GPU; for GPU tensors, may be no-op
         */
        virtual bool ensureOnDevice(DeviceId target_device)
        {
            (void)target_device;
            return true; // CPU tensors are always "on device" (CPU)
        }

        /**
         * @brief Ensure tensor data is synchronized to host (CPU)
         * @return true if host data is now current
         * @note Downloads from GPU if tensor was modified on device
         */
        virtual bool ensureOnHost() { return true; }

        /**
         * @brief Check if tensor data is valid on host
         * @return true if host buffer contains current data
         */
        virtual bool isHostValid() const { return true; }

        /**
         * @brief Check if tensor data is valid on device
         * @return true if device buffer is allocated AND contains current data
         */
        virtual bool isDeviceValid() const { return false; }

        // =========================================================================
        // Type-Safe Downcasting
        // =========================================================================

        /**
         * @brief Check if this tensor can be cast to type T
         * @tparam T Concrete tensor type (e.g., FP32Tensor, BF16Tensor)
         * @return true if this tensor is of type T
         *
         * Usage:
         * @code
         * if (tensor.is<FP32Tensor>()) {
         *     // safe to call typed_as<FP32Tensor>()
         * }
         * @endcode
         */
        template <typename T>
        bool is() const
        {
            return native_type_id() == T::static_type_id();
        }

        /**
         * @brief Cast to typed tensor reference (asserts on type mismatch)
         * @tparam T Concrete tensor type
         * @return Reference to this tensor as type T
         * @note Asserts if type doesn't match - use for known-type scenarios
         *
         * Usage:
         * @code
         * auto& fp32 = tensor.typed_as<FP32Tensor>();
         * float* data = fp32.data();  // Type-safe access
         * @endcode
         *
         * @note Uses dynamic_cast due to virtual inheritance. Still fast due to assert
         * checking the type match before cast.
         */
        template <typename T>
        T &typed_as()
        {
            assert(is<T>() && "ITensor::typed_as<T>(): tensor type mismatch");
            // dynamic_cast required for virtual inheritance
            return dynamic_cast<T &>(*this);
        }

        template <typename T>
        const T &typed_as() const
        {
            assert(is<T>() && "ITensor::typed_as<T>(): tensor type mismatch");
            // dynamic_cast required for virtual inheritance
            return dynamic_cast<const T &>(*this);
        }

        /**
         * @brief Try to cast to typed tensor pointer (returns nullptr on mismatch)
         * @tparam T Concrete tensor type
         * @return Pointer to this tensor as type T, or nullptr if type doesn't match
         *
         * Usage:
         * @code
         * if (auto* fp32 = tensor.try_as<FP32Tensor>()) {
         *     float* data = fp32->data();  // Type-safe access
         * } else if (auto* bf16 = tensor.try_as<BF16Tensor>()) {
         *     uint16_t* data = bf16->native_data();
         * }
         * @endcode
         *
         * @note Uses dynamic_cast due to virtual inheritance in TensorBase hierarchy.
         */
        template <typename T>
        T *try_as()
        {
            // dynamic_cast required for virtual inheritance
            return is<T>() ? dynamic_cast<T *>(this) : nullptr;
        }

        template <typename T>
        const T *try_as() const
        {
            // dynamic_cast required for virtual inheritance
            return is<T>() ? dynamic_cast<const T *>(this) : nullptr;
        }

        // =========================================================================
        // Conversion Methods
        // =========================================================================

        /**
         * @brief Convert tensor data to FP32 format
         * @param dst Destination buffer (must be pre-allocated with numel() floats)
         * @note For FP32 tensors this is a memcpy, for quantized it dequantizes
         */
        virtual void to_fp32(float *dst) const = 0;
    };

    // =========================================================================
    // Type ID Constants
    // =========================================================================

    /**
     * @brief Compile-time type IDs for tensor types
     *
     * These MUST match the TensorType enum values in Tensors.h exactly.
     * Order: FP32=0, BF16=1, FP16=2, INT8=3, INT32=4, IQ4_NL=5, IQ4_XS=6, Q8_0=7, Q8_1=8, Q16_1=9, ...
     * Defined here as constexpr to enable template specialization without circular dependency.
     */
    namespace TensorTypeId
    {
        // Dense floating point formats
        constexpr int FP32 = 0;
        constexpr int BF16 = 1;
        constexpr int FP16 = 2;

        // Integer formats
        constexpr int INT8 = 3;
        constexpr int INT32 = 4;

        // Quantized formats (order matches TensorType enum in Tensors.h)
        constexpr int IQ4_NL = 5;
        constexpr int IQ4_XS = 6;
        constexpr int Q8_0 = 7;
        constexpr int Q8_1 = 8;
        constexpr int Q16_1 = 9; // High-precision residual format
        constexpr int Q4_0 = 10;
        constexpr int Q4_1 = 11;
        constexpr int Q5_0 = 12;
        constexpr int Q5_1 = 13;
        constexpr int Q6_K = 14;
        constexpr int Q2_K = 15;
        constexpr int Q5_K = 16;
        constexpr int Q3_K = 17;
        constexpr int Q4_K = 18;
        constexpr int Q8_K = 19;
        constexpr int IQ2_XXS = 20;
        constexpr int IQ2_XS = 21;
        constexpr int IQ3_XXS = 22;
        constexpr int IQ2_S = 23;
        constexpr int IQ3_S = 24;
        constexpr int IQ1_S = 25;
        constexpr int IQ1_M = 26;
        constexpr int TQ4 = 27;
        constexpr int TQ8 = 28;
    }

} // namespace llaminar2
