/**
 * @file ComputeStageUtils.h
 * @brief Shared utilities for compute stage implementations
 */
#pragma once

#include "../../tensors/Tensors.h"

namespace llaminar2
{

    /**
     * @brief Safe FP32 data access for getDumpInfo()
     *
     * For Q8_1 tensors, uses fp32_data() (explicit dequant for debugging only)
     * For other tensors, uses data()
     */
    inline const float *getSafeFp32Data(const TensorBase *tensor)
    {
        if (!tensor)
            return nullptr;

        if (tensor->native_type() == TensorType::Q8_1)
        {
            // Q8_1 tensors throw on data() - use explicit fp32_data() for debugging
            auto *q8_1 = dynamic_cast<const Q8_1Tensor *>(tensor);
            return q8_1 ? q8_1->fp32_data() : nullptr;
        }

        if (tensor->native_type() == TensorType::Q16_1)
        {
            // Q16_1 tensors also need special handling
            auto *q16_1 = dynamic_cast<const Q16_1Tensor *>(tensor);
            return q16_1 ? q16_1->fp32_data() : nullptr;
        }

        // Standard FP32/BF16/FP16 tensors
        return tensor->data();
    }

} // namespace llaminar2
