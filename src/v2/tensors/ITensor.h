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
         * @brief Get the shape of this tensor
         * @return Reference to shape vector (dimensions)
         */
        virtual const std::vector<size_t> &shape() const = 0;

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

        /**
         * @brief Get the device index where this tensor resides
         * @return -1 for CPU/host, >= 0 for GPU device index
         */
        virtual int device_index() const = 0;

        // =========================================================================
        // Raw Data Access (type-unsafe, use with caution)
        // =========================================================================

        /**
         * @brief Get raw pointer to the data buffer (const)
         * @return Pointer to start of data buffer, caller must know the actual type
         * @note Use typed_as<T>() for type-safe access when possible
         */
        virtual const void *raw_data() const = 0;

        /**
         * @brief Get raw mutable pointer to the data buffer
         * @return Mutable pointer to start of data buffer
         * @note Use typed_as<T>() for type-safe access when possible
         */
        virtual void *raw_mutable_data() = 0;

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
         */
        template <typename T>
        T &typed_as()
        {
            assert(is<T>() && "ITensor::typed_as<T>(): tensor type mismatch");
            return static_cast<T &>(*this);
        }

        template <typename T>
        const T &typed_as() const
        {
            assert(is<T>() && "ITensor::typed_as<T>(): tensor type mismatch");
            return static_cast<const T &>(*this);
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
         */
        template <typename T>
        T *try_as()
        {
            return is<T>() ? static_cast<T *>(this) : nullptr;
        }

        template <typename T>
        const T *try_as() const
        {
            return is<T>() ? static_cast<const T *>(this) : nullptr;
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
    }

} // namespace llaminar2
