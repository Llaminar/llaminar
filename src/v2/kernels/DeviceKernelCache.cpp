/**
 * @file DeviceKernelCache.cpp
 * @brief Implementation of universal device kernel cache
 */

#include "DeviceKernelCache.h"
#include "utils/Logger.h"
#include <sstream>

namespace llaminar2
{
    // Static member definitions
    std::mutex DeviceKernelCache::mutex_;
    std::unordered_map<DeviceKernelKey, std::unique_ptr<IDeviceKernel>, DeviceKernelKeyHash>
        DeviceKernelCache::cache_;
    std::unordered_map<DeviceKernelCache::FactoryKey, DeviceKernelCache::KernelFactory,
                       DeviceKernelCache::FactoryKeyHash>
        DeviceKernelCache::factories_;

    // =========================================================================
    // Kernel Access
    // =========================================================================

    IDeviceKernel* DeviceKernelCache::getOrCreate(const DeviceId& device, KernelType type)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        DeviceKernelKey key{device, type};

        // Check cache first
        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            return it->second.get();
        }

        // Look for factory
        FactoryKey factory_key{device.type, type};
        auto factory_it = factories_.find(factory_key);
        if (factory_it == factories_.end())
        {
            std::ostringstream oss;
            oss << "DeviceKernelCache: no factory registered for "
                << kernelTypeName(type) << " on " << device.to_string();
            throw std::runtime_error(oss.str());
        }

        // Create kernel using factory
        LOG_DEBUG("DeviceKernelCache: creating " << kernelTypeName(type)
                                                  << " kernel for " << device.to_string());
        auto kernel = factory_it->second(device);
        if (!kernel)
        {
            throw std::runtime_error(
                std::string("DeviceKernelCache: factory returned null for ") +
                kernelTypeName(type));
        }

        // Verify kernel type and device
        if (kernel->type() != type)
        {
            throw std::runtime_error(
                std::string("DeviceKernelCache: factory returned wrong kernel type for ") +
                kernelTypeName(type));
        }

        IDeviceKernel* ptr = kernel.get();
        cache_.emplace(key, std::move(kernel));
        return ptr;
    }

    bool DeviceKernelCache::hasKernel(const DeviceId& device, KernelType type)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        DeviceKernelKey key{device, type};
        return cache_.find(key) != cache_.end();
    }

    // =========================================================================
    // Factory Registration
    // =========================================================================

    void DeviceKernelCache::registerFactory(
        DeviceType device_type,
        KernelType kernel_type,
        KernelFactory factory)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        FactoryKey key{device_type, kernel_type};

        if (factories_.find(key) != factories_.end())
        {
            LOG_WARN("DeviceKernelCache: overwriting factory for "
                     << kernelTypeName(kernel_type) << " on device type "
                     << static_cast<int>(device_type));
        }

        factories_[key] = std::move(factory);
        LOG_DEBUG("DeviceKernelCache: registered factory for "
                  << kernelTypeName(kernel_type) << " on device type "
                  << static_cast<int>(device_type));
    }

    // =========================================================================
    // Cache Management
    // =========================================================================

    void DeviceKernelCache::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        LOG_INFO("DeviceKernelCache: clearing " << cache_.size() << " cached kernels");
        cache_.clear();
    }

    void DeviceKernelCache::clearDevice(const DeviceId& device)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t removed = 0;
        for (auto it = cache_.begin(); it != cache_.end();)
        {
            if (it->first.device == device)
            {
                it = cache_.erase(it);
                ++removed;
            }
            else
            {
                ++it;
            }
        }

        LOG_DEBUG("DeviceKernelCache: cleared " << removed << " kernels for "
                                                 << device.to_string());
    }

    size_t DeviceKernelCache::size()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    std::pair<size_t, size_t> DeviceKernelCache::stats()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Basic memory estimate (just unique_ptr overhead, not actual kernel memory)
        size_t memory = cache_.size() * (sizeof(DeviceKernelKey) + sizeof(std::unique_ptr<IDeviceKernel>) + 64);

        return {cache_.size(), memory};
    }

} // namespace llaminar2
