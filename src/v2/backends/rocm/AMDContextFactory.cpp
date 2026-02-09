/**
 * @file AMDContextFactory.cpp
 * @brief ROCm/HIP factory registration for GPUDeviceContextPool
 *
 * This file registers the AMD device context factory with the
 * GPUDeviceContextPool during static initialization. This allows
 * GPUDeviceContextPool (which lives in llaminar2_core) to create
 * AMDDeviceContext instances without directly depending on HIP headers.
 *
 * NOTE: The static initializer (AMDFactoryRegistrar) may be stripped by the
 * linker when this TU is in a static library and no symbols are referenced.
 * The public function ensureAMDFactoryRegistered() provides an explicit
 * registration path that also forces the linker to include this TU.
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
         * @brief Perform the actual factory registration with the pool
         *
         * Shared logic used by both the static initializer and the explicit
         * ensureAMDFactoryRegistered() function.
         */
        void doRegisterAMDFactory()
        {
            auto &pool = GPUDeviceContextPool::instance();
            if (pool.hasAMDSupport())
                return; // Already registered

            auto devices = rocm_enumeration::enumerate_rocm_devices();
            int device_count = static_cast<int>(devices.size());

            if (device_count > 0)
            {
                pool.registerAMDFactory(createAMDContext, device_count);
                LOG_INFO("[AMDContextFactory] Registered ROCm factory with "
                         << device_count << " devices");
            }
            else
            {
                LOG_DEBUG("[AMDContextFactory] No ROCm devices found, factory not registered");
            }
        }

        /**
         * @brief RAII helper for static initialization registration
         *
         * This struct's constructor runs during static initialization (before main)
         * to register the AMD factory with GPUDeviceContextPool.
         *
         * NOTE: This may not run if the linker strips this TU from the static
         * library. Use ensureAMDFactoryRegistered() as an explicit fallback.
         */
        struct AMDFactoryRegistrar
        {
            AMDFactoryRegistrar()
            {
                try
                {
                    doRegisterAMDFactory();
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

    // =========================================================================
    // Explicit registration (called when static init is unreliable)
    // =========================================================================

    void ensureAMDFactoryRegistered()
    {
        try
        {
            doRegisterAMDFactory();
        }
        catch (const std::exception &e)
        {
            LOG_WARN("[AMDContextFactory] Explicit registration failed: " << e.what());
        }
    }

} // namespace llaminar2
