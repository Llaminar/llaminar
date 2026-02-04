/**
 * @file TPDomainConfig.cpp
 * @brief Implementation of TPDomainConfig
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "TPDomainConfig.h"
#include "collective/ILocalTPContext.h"
#include <cmath>

namespace llaminar2
{

    DeviceId TPDomainConfig::primaryDevice() const
    {
        if (devices.empty())
        {
            return DeviceId::cpu();
        }
        return devices[0];
    }

    int TPDomainConfig::degree() const
    {
        return static_cast<int>(devices.size());
    }

    bool TPDomainConfig::isHomogeneous() const
    {
        if (devices.size() <= 1)
        {
            return true; // Single or no device is trivially homogeneous
        }

        DeviceType first_type = devices[0].type;
        for (size_t i = 1; i < devices.size(); ++i)
        {
            if (devices[i].type != first_type)
            {
                return false;
            }
        }
        return true;
    }

    bool TPDomainConfig::isGPUDomain() const
    {
        if (devices.empty())
        {
            return false;
        }

        for (const auto &device : devices)
        {
            if (!device.is_gpu())
            {
                return false;
            }
        }
        return true;
    }

    bool TPDomainConfig::isCPUDomain() const
    {
        if (devices.empty())
        {
            return false;
        }

        for (const auto &device : devices)
        {
            if (!device.is_cpu())
            {
                return false;
            }
        }
        return true;
    }

    bool TPDomainConfig::isCrossVendor() const
    {
        bool has_cuda = false;
        bool has_rocm = false;

        for (const auto &device : devices)
        {
            if (device.is_cuda())
            {
                has_cuda = true;
            }
            else if (device.is_rocm())
            {
                has_rocm = true;
            }
        }

        return has_cuda && has_rocm;
    }

    bool TPDomainConfig::validate(std::string *error_msg) const
    {
        // Check name
        if (name.empty())
        {
            if (error_msg)
            {
                *error_msg = "TPDomainConfig: name cannot be empty";
            }
            return false;
        }

        // Check devices
        if (devices.empty())
        {
            if (error_msg)
            {
                *error_msg = "TPDomainConfig: at least one device is required";
            }
            return false;
        }

        // Check device validity
        for (size_t i = 0; i < devices.size(); ++i)
        {
            if (!devices[i].is_valid())
            {
                if (error_msg)
                {
                    *error_msg = "TPDomainConfig: device[" + std::to_string(i) + "] is invalid";
                }
                return false;
            }
        }

        // Check weights if provided
        if (!device_weights.empty())
        {
            // Weight count must match device count
            if (device_weights.size() != devices.size())
            {
                if (error_msg)
                {
                    *error_msg = "TPDomainConfig: weight count (" +
                                 std::to_string(device_weights.size()) +
                                 ") does not match device count (" +
                                 std::to_string(devices.size()) + ")";
                }
                return false;
            }

            // Weights must sum to approximately 1.0
            float sum = 0.0f;
            for (float w : device_weights)
            {
                if (w < 0.0f)
                {
                    if (error_msg)
                    {
                        *error_msg = "TPDomainConfig: weights cannot be negative";
                    }
                    return false;
                }
                sum += w;
            }

            const float tolerance = 0.01f;
            if (std::fabs(sum - 1.0f) > tolerance)
            {
                if (error_msg)
                {
                    *error_msg = "TPDomainConfig: weights must sum to 1.0 (got " +
                                 std::to_string(sum) + ")";
                }
                return false;
            }
        }

        // Check backend compatibility
        if (tp_backend == CollectiveBackendType::NCCL)
        {
            // NCCL requires all CUDA devices
            for (const auto &device : devices)
            {
                if (!device.is_cuda())
                {
                    if (error_msg)
                    {
                        *error_msg = "TPDomainConfig: NCCL backend requires all CUDA devices";
                    }
                    return false;
                }
            }
        }
        else if (tp_backend == CollectiveBackendType::RCCL)
        {
            // RCCL requires all ROCm devices
            for (const auto &device : devices)
            {
                if (!device.is_rocm())
                {
                    if (error_msg)
                    {
                        *error_msg = "TPDomainConfig: RCCL backend requires all ROCm devices";
                    }
                    return false;
                }
            }
        }

        return true;
    }

    std::unique_ptr<ILocalTPContext> TPDomainConfig::createTPContext() const
    {
        // TODO: Wire up to LocalTPContext factory in Phase 1.2
        // This will instantiate the appropriate context based on:
        // - Backend type (NCCL, RCCL, PCIeBAR, HOST)
        // - Device configuration
        // - Weights for proportional TP
        return nullptr;
    }

} // namespace llaminar2
