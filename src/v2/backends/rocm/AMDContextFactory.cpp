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
#include <cstdlib>
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
            const char *cpu_only_env = std::getenv("LLAMINAR_FORCE_CPU_ONLY_STARTUP");
            if (cpu_only_env && std::atoi(cpu_only_env) != 0)
            {
                LOG_INFO("[AMDContextFactory] Skipping ROCm factory registration (LLAMINAR_FORCE_CPU_ONLY_STARTUP=1)");
                return;
            }

            const char *skip_rocm_env = std::getenv("LLAMINAR_SKIP_ROCM_STARTUP");
            if (skip_rocm_env && std::atoi(skip_rocm_env) != 0)
            {
                LOG_INFO("[AMDContextFactory] Skipping ROCm factory registration (LLAMINAR_SKIP_ROCM_STARTUP=1)");
                return;
            }

            auto &pool = GPUDeviceContextPool::instance();
            if (pool.hasAMDSupport())
                return; // Already registered

            auto devices = rocm_enumeration::enumerate_rocm_devices();
            int device_count = static_cast<int>(devices.size());

            if (device_count > 0)
            {
                pool.registerAMDFactory(createAMDContext, device_count);
                LOG_DEBUG("[AMDContextFactory] Registered ROCm factory with "
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
         * NOTE: The static constructor intentionally does NOT enumerate ROCm
         * devices. GPU enumeration during static init runs before main() and
         * cannot be controlled by environment variables or CLI flags, adding
         * measurable latency to every process start (even for `--help`) and
         * emitting log lines before logging has been configured by the app.
         *
         * Actual registration is deferred to ensureAMDFactoryRegistered()
         * which is called explicitly from GPUDeviceContextPool when an AMD
         * context is first needed. This makes startup zero-cost when only
         * CUDA or CPU backends are used, and avoids spurious ROCm output
         * during early-exit paths like `--help` and `--version`.
         */
        struct AMDFactoryRegistrar
        {
            AMDFactoryRegistrar()
            {
                // Intentionally empty - registration deferred to
                // ensureAMDFactoryRegistered() for controllable startup.
            }
        };

        // Static instance kept for linker inclusion (ensures TU is not stripped)
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
