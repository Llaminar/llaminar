/**
 * @file AMDContextFactory.cpp
 * @brief ROCm/HIP factory registration for GPUDeviceContextPool
 *
 * This file registers the AMD device context factory with the
 * GPUDeviceContextPool during static initialization. This allows
 * GPUDeviceContextPool (which lives in llaminar2_core) to create
 * AMDDeviceContext instances without directly depending on HIP headers.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "AMDDeviceContext.h"
#include "../GPUDeviceContextPool.h"
#include "../GPUEnumeration.h"
#include "../../utils/Logger.h"
#include <memory>

namespace llaminar2
{
    namespace
    {

        /**
         * @brief Factory function to create AMDDeviceContext instances
         *
         * This function is registered with GPUDeviceContextPool and called
         * when a ROCm context is requested via getAMDContext().
         *
         * @param device_ordinal HIP device index (0, 1, 2, ...)
         * @return Unique pointer to new AMDDeviceContext (as IWorkerGPUContext)
         */
        std::unique_ptr<IWorkerGPUContext> createAMDContext(int device_ordinal)
        {
            return std::make_unique<AMDDeviceContext>(device_ordinal);
        }

        /**
         * @brief RAII helper for static initialization registration
         *
         * This struct's constructor runs during static initialization (before main)
         * to register the AMD factory with GPUDeviceContextPool.
         */
        struct AMDFactoryRegistrar
        {
            AMDFactoryRegistrar()
            {
                try
                {
                    // Get device count via enumeration
                    auto devices = rocm_enumeration::enumerate_rocm_devices();
                    int device_count = static_cast<int>(devices.size());

                    if (device_count > 0)
                    {
                        // Register factory with device pool
                        GPUDeviceContextPool::instance().registerAMDFactory(
                            createAMDContext, device_count);

                        LOG_DEBUG("[AMDContextFactory] Registered ROCm factory with "
                                  << device_count << " devices");
                    }
                    else
                    {
                        LOG_DEBUG("[AMDContextFactory] No ROCm devices found, factory not registered");
                    }
                }
                catch (const std::exception &e)
                {
                    LOG_WARN("[AMDContextFactory] Failed to register ROCm factory: " << e.what());
                }
            }
        };

        // Static instance triggers registration during library load
        static AMDFactoryRegistrar amd_factory_registrar;

    } // anonymous namespace
} // namespace llaminar2
