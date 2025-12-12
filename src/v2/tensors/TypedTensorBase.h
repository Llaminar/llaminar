/**
 * @file TypedTensorBase.h
 * @brief CRTP base class for type-safe tensor data access
 * @author David Sanftenberg
 *
 * This file implements the Curiously Recurring Template Pattern (CRTP) to provide
 * type-safe data() and mutable_data() accessors with zero virtual dispatch overhead
 * when the concrete type is known at compile time.
 *
 * Design goals:
 * 1. Type-safe accessors: data() returns the correct native type (float*, uint16_t*, Q8_1Block*, etc.)
 * 2. Zero overhead: No virtual dispatch when type is known at compile time
 * 3. Runtime polymorphism: Compatible with ITensor for heterogeneous collections
 * 4. Backward compatibility: Works alongside existing TensorBase hierarchy
 *
 * Usage pattern for concrete tensor classes:
 *
 * @code
 * // Define a typed tensor by inheriting from TypedTensorBase
 * class FP32Tensor : public TypedTensorBase<FP32Tensor, float> {
 * public:
 *     // CRTP implementation - called by TypedTensorBase::data()
 *     const float* data_impl() const { return host_data_.data(); }
 *     float* mutable_data_impl() { return host_data_.data(); }
 *
 *     // Static type ID for ITensor::is<T>() and typed_as<T>()
 *     static constexpr int static_type_id() { return TensorTypeId::FP32; }
 *
 * private:
 *     std::vector<float> host_data_;
 * };
 *
 * // Usage in templated code (zero overhead)
 * template<typename T>
 * void process(T& tensor) {
 *     auto* data = tensor.data();  // Returns T::value_type*
 *     // ... work with native type
 * }
 *
 * // Usage with runtime polymorphism
 * void process(ITensor& tensor) {
 *     if (auto* fp32 = tensor.try_as<FP32Tensor>()) {
 *         float* data = fp32->data();  // Type-safe!
 *     }
 * }
 * @endcode
 *
 * Note: This CRTP layer is optional. Tensors can directly implement ITensor
 * without using TypedTensorBase if they need custom data() semantics.
 */

#pragma once

#include "ITensor.h"
#include <type_traits>

namespace llaminar2
{

    /**
     * @brief CRTP base class providing type-safe data accessors
     *
     * TypedTensorBase uses the Curiously Recurring Template Pattern to provide
     * type-safe data() and mutable_data() methods that return the correct
     * native pointer type (float*, uint16_t*, Q8_1Block*, etc.).
     *
     * When the concrete type is known at compile time, data() resolves directly
     * to the derived class's data_impl() with no virtual dispatch.
     *
     * @tparam Derived The concrete tensor class (CRTP pattern)
     * @tparam DataType The native storage type (float, uint16_t, Q8_1Block, etc.)
     *
     * Requirements for Derived class:
     * - Implement `const DataType* data_impl() const`
     * - Implement `DataType* mutable_data_impl()`
     * - Implement `static constexpr int static_type_id()`
     * - Implement ITensor interface: shape(), numel(), size_bytes(), device_index()
     */
    template <typename Derived, typename DataType>
    class TypedTensorBase : public ITensor
    {
    public:
        /// Native storage type for this tensor
        using value_type = DataType;
        using pointer = DataType *;
        using const_pointer = const DataType *;
        using reference = DataType &;
        using const_reference = const DataType &;

        // =========================================================================
        // Type-Safe Data Access (CRTP - no virtual overhead when type is known)
        // =========================================================================

        /**
         * @brief Get const pointer to native data (type-safe, zero overhead)
         * @return Pointer to data in native format (float*, uint16_t*, Q8_1Block*, etc.)
         *
         * This method uses CRTP to call the derived class's data_impl() directly,
         * avoiding virtual dispatch when the concrete type is known at compile time.
         */
        const_pointer data() const
        {
            return static_cast<const Derived *>(this)->data_impl();
        }

        /**
         * @brief Get mutable pointer to native data (type-safe, zero overhead)
         * @return Mutable pointer to data in native format
         *
         * Note: Some tensor types (e.g., quantized weights) may throw from mutable_data_impl()
         * as they are read-only.
         */
        pointer mutable_data()
        {
            return static_cast<Derived *>(this)->mutable_data_impl();
        }

        // =========================================================================
        // ITensor Interface Implementation
        // =========================================================================

        /**
         * @brief Get raw data pointer (implements ITensor)
         * @return Void pointer to native data, caller must know the actual type
         */
        const void *raw_data() const override
        {
            return static_cast<const void *>(data());
        }

        /**
         * @brief Get raw mutable data pointer (implements ITensor)
         * @return Void pointer to native data
         */
        void *raw_mutable_data() override
        {
            return static_cast<void *>(mutable_data());
        }

        /**
         * @brief Get runtime type ID (implements ITensor)
         * @return Integer type ID matching TensorTypeId constants
         */
        int native_type_id() const override
        {
            return Derived::static_type_id();
        }

        /**
         * @brief Get size in bytes (default implementation)
         * @return numel() * sizeof(DataType)
         * @note Override in derived class for block-quantized formats
         */
        size_t size_bytes() const override
        {
            return this->numel() * sizeof(DataType);
        }

        // =========================================================================
        // Static Type Information
        // =========================================================================

        /**
         * @brief Get the static type ID for this tensor class
         * @return Integer type ID from TensorTypeId namespace
         *
         * Must be implemented in Derived class as:
         * @code
         * static constexpr int static_type_id() { return TensorTypeId::FP32; }
         * @endcode
         */
        static constexpr int static_type_id()
        {
            return Derived::static_type_id();
        }

    protected:
        TypedTensorBase() = default;
        ~TypedTensorBase() override = default;

        // Non-copyable (tensors contain mutexes, cached kernels, device pointers)
        TypedTensorBase(const TypedTensorBase &) = delete;
        TypedTensorBase &operator=(const TypedTensorBase &) = delete;

        // Move disabled (enable_shared_from_this semantics)
        TypedTensorBase(TypedTensorBase &&) = delete;
        TypedTensorBase &operator=(TypedTensorBase &&) = delete;
    };

    // =========================================================================
    // Type Traits for CRTP Tensors
    // =========================================================================

    /**
     * @brief Type trait to detect if a type is a TypedTensorBase-derived tensor
     */
    template <typename T, typename = void>
    struct is_typed_tensor : std::false_type
    {
    };

    template <typename T>
    struct is_typed_tensor<T, std::void_t<typename T::value_type>> : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_typed_tensor_v = is_typed_tensor<T>::value;

    /**
     * @brief Type trait to get the native data type of a tensor
     */
    template <typename T, typename = void>
    struct tensor_value_type
    {
        using type = void;
    };

    template <typename T>
    struct tensor_value_type<T, std::void_t<typename T::value_type>>
    {
        using type = typename T::value_type;
    };

    template <typename T>
    using tensor_value_type_t = typename tensor_value_type<T>::type;

} // namespace llaminar2
