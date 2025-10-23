/**
 * @file TensorBase.cpp
 * @brief Implementation of TensorBase pull-through cache methods
 * @author David Sanftenberg
 * @date October 20, 2025
 */

#include "TensorBase.h"
#include "../operators/QuantSlabCache.h"
#include "../utils/BFloat16.h"

namespace llaminar
{

    const float *TensorBase::data_fp32() const
    {
        // Handle empty tensors (e.g., [0, 896] shape from rank with no tokens)
        if (element_count() == 0)
        {
            static float empty_buffer[1] = {0.0f}; // Non-const to allow const_cast in data()
            return empty_buffer;
        }

        // Fast path: If tensor is natively FP32, return direct pointer
        const float *native_ptr = data_native_fp32();
        if (native_ptr != nullptr)
        {
            return native_ptr;
        }

        // Pull-through cache path
        auto &cache = QuantSlabCache::instance();

        const float *result = cache.getOrDecodeTensor<float>(
            this,                 // tensor identity
            element_count(),      // size
            CachedDataType::FP32, // requested type
            [this](float *dst) {  // decode callback
                this->decode_to_fp32(dst);
            });

        // Sanity check
        if (!result)
        {
            throw std::runtime_error("QuantSlabCache returned nullptr for " + type_name());
        }

        return result;
    }

    const void *TensorBase::data_bf16() const
    {
        // Handle empty tensors (e.g., [0, 896] shape from rank with no tokens)
        if (element_count() == 0)
        {
            static bfloat16 empty_buffer[1] = {bfloat16(0.0f)}; // Non-const to allow const_cast
            return empty_buffer;
        }

        // Fast path: If tensor is natively BF16, return direct pointer
        const void *native_ptr = data_native_bf16();
        if (native_ptr != nullptr)
        {
            return native_ptr;
        }

        // Pull-through cache path
        auto &cache = QuantSlabCache::instance();

        return cache.getOrDecodeTensor<bfloat16>(
            this,                   // tensor identity
            element_count(),        // size
            CachedDataType::BF16,   // requested type
            [this](bfloat16 *dst) { // decode callback
                this->decode_to_bf16(reinterpret_cast<void *>(dst));
            });
    }

} // namespace llaminar
