/**
 * @file TensorValidation.cpp
 * @brief Factory dispatcher for GPU tensor validators
 *
 * @author David Sanftenberg
 */

#include "GPUTensorVerification.h"
#include "../utils/Logger.h"

// Check for GPU backend availability at compile time
#ifdef HAVE_CUDA
extern "C" llaminar2::ITensorValidator *llaminar2_getCUDATensorValidator();
extern "C" llaminar2::ITensorValidator *llaminar2_getCUDATensorValidatorForDevice(int device_id);
#endif

#ifdef HAVE_ROCM
extern "C" llaminar2::ITensorValidator *llaminar2_getROCmTensorValidator();
extern "C" llaminar2::ITensorValidator *llaminar2_getROCmTensorValidatorForDevice(int device_id);
#endif

namespace llaminar2
{

    ITensorValidator *getTensorValidator(DeviceType device_type)
    {
        switch (device_type)
        {
        case DeviceType::CUDA:
#ifdef HAVE_CUDA
            return llaminar2_getCUDATensorValidator();
#else
            LOG_DEBUG("[TensorValidation] CUDA not available");
            return nullptr;
#endif

        case DeviceType::ROCm:
#ifdef HAVE_ROCM
            return llaminar2_getROCmTensorValidator();
#else
            LOG_DEBUG("[TensorValidation] ROCm not available");
            return nullptr;
#endif

        default:
            LOG_DEBUG("[TensorValidation] No GPU validator for device type "
                      << static_cast<int>(device_type));
            return nullptr;
        }
    }

    ITensorValidator *getTensorValidator(DeviceType device_type, int device_id)
    {
        if (device_id < 0)
        {
            LOG_ERROR("[TensorValidation] Invalid GPU validator device ordinal " << device_id);
            return nullptr;
        }

        switch (device_type)
        {
        case DeviceType::CUDA:
#ifdef HAVE_CUDA
            return llaminar2_getCUDATensorValidatorForDevice(device_id);
#else
            LOG_DEBUG("[TensorValidation] CUDA not available");
            return nullptr;
#endif

        case DeviceType::ROCm:
#ifdef HAVE_ROCM
            return llaminar2_getROCmTensorValidatorForDevice(device_id);
#else
            LOG_DEBUG("[TensorValidation] ROCm not available");
            return nullptr;
#endif

        default:
            LOG_DEBUG("[TensorValidation] No GPU validator for device type "
                      << static_cast<int>(device_type));
            return nullptr;
        }
    }

#ifndef HAVE_CUDA
    ITensorValidator *getCUDATensorValidator()
    {
        return nullptr;
    }

    ITensorValidator *getCUDATensorValidator(int)
    {
        return nullptr;
    }
#endif

#ifndef HAVE_ROCM
    ITensorValidator *getROCmTensorValidator()
    {
        return nullptr;
    }

    ITensorValidator *getROCmTensorValidator(int)
    {
        return nullptr;
    }
#endif

} // namespace llaminar2
