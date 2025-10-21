#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace llaminar
{
    // Forward declaration
    struct Tensor;

    // Forward declarations for cache
    class QuantSlabCache;
    enum class CachedDataType : uint8_t;

    /**
     * @brief Native data type of a tensor
     *
     * Used for fast-path optimization in pull-through cache.
     */
    enum class TensorDataType
    {
        FP32,      ///< 32-bit float (SimpleTensor)
        BF16,      ///< 16-bit bfloat16 (BF16Tensor)
        QUANTIZED, ///< Quantized format (QuantizedTensor)
        OTHER      ///< Other/custom formats
    };

    /**
     * Abstract base class for all tensor types in the llaminar system.
     * Provides a common interface that can be implemented by both simple
     * in-memory tensors and COSMA-optimized distributed tensors.
     */
    class TensorBase
    {
    public:
        virtual ~TensorBase() = default;

        // Shape and size information
        virtual const std::vector<int> &shape() const = 0;
        virtual int size() const = 0;
        virtual int ndim() const = 0;

        // Data access - returns pointer to the local data portion
        virtual float *data() = 0;
        virtual const float *data() const = 0;

        // ===========================
        // Pull-Through Cache Interface (NEW)
        // ===========================

        /**
         * @brief Get FP32 data pointer (may decode via cache)
         *
         * This method implements the pull-through cache pattern:
         * - If tensor is natively FP32: returns direct pointer (fast path)
         * - Otherwise: checks shared cache, decodes if miss, returns cached pointer
         *
         * The returned pointer is valid until the cache evicts it (typically
         * safe within the same forward pass).
         *
         * @return const float* Pointer to FP32 data
         */
        virtual const float *data_fp32() const;

        /**
         * @brief Get BF16 data pointer (may decode via cache)
         *
         * Similar to data_fp32() but for bfloat16 format.
         *
         * @return const bfloat16* Pointer to BF16 data
         */
        virtual const void *data_bf16() const; // void* to avoid bfloat16 include here

        /**
         * @brief Get native data type of this tensor
         *
         * Used for fast-path optimization (avoid cache lookup for native type).
         *
         * @return TensorDataType Native storage format
         */
        virtual TensorDataType native_type() const = 0;

        /**
         * @brief Get total number of elements in tensor
         *
         * @return size_t Element count
         */
        virtual size_t element_count() const = 0;

        // Tensor type identification
        virtual std::string type_name() const = 0;
        virtual bool is_distributed() const = 0;

        // Utility methods
        virtual void zero() = 0;
        virtual void fill(float value) = 0;

        // Basic tensor operations
        virtual std::shared_ptr<TensorBase> copy() const = 0;
        virtual void copy_from(const TensorBase &other) = 0;

        // Shape utilities
        int total_elements() const
        {
            int total = 1;
            const auto &dims = shape();
            for (int dim : dims)
            {
                total *= dim;
            }
            return total;
        }

        // Matrix-specific accessors (for 2D tensors)
        int rows() const
        {
            const auto &dims = shape();
            return dims.size() >= 1 ? dims[0] : 0;
        }

        int cols() const
        {
            const auto &dims = shape();
            return dims.size() >= 2 ? dims[1] : 1;
        }

        // Validation
        bool is_matrix() const { return ndim() == 2; }
        bool is_vector() const { return ndim() == 1; }
        bool is_scalar() const { return ndim() == 0; }

        bool is_compatible_shape(const TensorBase &other) const
        {
            return shape() == other.shape();
        }

    protected:
        // ===========================
        // Decode Hooks for Subclasses
        // ===========================

        /**
         * @brief Fast-path accessor for native FP32 tensors
         *
         * Override this in SimpleTensor to return direct pointer.
         * Default returns nullptr (triggers cache path).
         */
        virtual const float *data_native_fp32() const { return nullptr; }

        /**
         * @brief Fast-path accessor for native BF16 tensors
         *
         * Override this in BF16Tensor to return direct pointer.
         * Default returns nullptr (triggers cache path).
         */
        virtual const void *data_native_bf16() const { return nullptr; }

        /**
         * @brief Decode entire tensor to FP32
         *
         * Called on cache miss when data_fp32() requested.
         * Must fill dst with element_count() floats.
         *
         * @param dst Destination buffer (pre-allocated)
         */
        virtual void decode_to_fp32(float *dst) const = 0;

        /**
         * @brief Decode entire tensor to BF16
         *
         * Called on cache miss when data_bf16() requested.
         * Must fill dst with element_count() bfloat16 values.
         *
         * @param dst Destination buffer (pre-allocated)
         */
        virtual void decode_to_bf16(void *dst) const = 0; // void* to avoid bfloat16 include
    };

    // Forward declaration - full definition in TensorFactory.h
    class TensorFactory;

} // namespace llaminar