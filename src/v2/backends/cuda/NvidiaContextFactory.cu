/**
 * @file NvidiaContextFactory.cu
 * @brief CUDA factory registration for GPUDeviceContextPool
 *
 * This file registers the NVIDIA device context factory with the
 * GPUDeviceContextPool during static initialization. This allows
 * GPUDeviceContextPool (which lives in llaminar2_core) to create
 * NvidiaDeviceContext instances without directly depending on CUDA headers.
 *
 * NOTE: The static initializer is intentionally a no-op to avoid ~500-800ms
 * CUDA enumeration during process startup (before main). Registration is
 * deferred to ensureNvidiaFactoryRegistered(), called explicitly by
 * DeviceGraphOrchestrator when a CUDA context is first needed.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "NvidiaDeviceContext.h"
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
         * @brief Factory function to create NvidiaDeviceContext instances
         *
         * This function is registered with GPUDeviceContextPool and called
         * when a CUDA context is requested via getNvidiaContext().
         *
         * @param device_ordinal CUDA device index (0, 1, 2, ...)
         * @return Unique pointer to new NvidiaDeviceContext (as IWorkerGPUContext)
         */
        std::unique_ptr<IWorkerGPUContext> createNvidiaContext(int device_ordinal)
        {
            return std::make_unique<NvidiaDeviceContext>(device_ordinal);
        }

        /**
         * @brief Perform the actual factory registration with the pool
         *
         * Shared logic used by both the static initializer and the explicit
         * ensureNvidiaFactoryRegistered() function.
         */
        void doRegisterNvidiaFactory()
        {
            const char *cpu_only_env = std::getenv("LLAMINAR_FORCE_CPU_ONLY_STARTUP");
            if (cpu_only_env && std::atoi(cpu_only_env) != 0)
            {
                LOG_INFO("[NvidiaContextFactory] Skipping CUDA factory registration (LLAMINAR_FORCE_CPU_ONLY_STARTUP=1)");
                return;
            }

            const char *skip_cuda_env = std::getenv("LLAMINAR_SKIP_CUDA_STARTUP");
            if (skip_cuda_env && std::atoi(skip_cuda_env) != 0)
            {
                LOG_INFO("[NvidiaContextFactory] Skipping CUDA factory registration (LLAMINAR_SKIP_CUDA_STARTUP=1)");
                return;
            }

            auto &pool = GPUDeviceContextPool::instance();
            if (pool.hasNvidiaSupport())
                return; // Already registered

            auto devices = cuda_enumeration::enumerate_cuda_devices();
            int device_count = static_cast<int>(devices.size());

            if (device_count > 0)
            {
                pool.registerNvidiaFactory(createNvidiaContext, device_count);
                LOG_INFO("[NvidiaContextFactory] Registered CUDA factory with "
                         << device_count << " devices");
            }
            else
            {
                LOG_DEBUG("[NvidiaContextFactory] No CUDA devices found, factory not registered");
            }
        }

        /**
         * @brief RAII helper for static initialization registration
         *
         * NOTE: The static constructor intentionally does NOT enumerate CUDA
         * devices. GPU enumeration during static init runs before main() and
         * cannot be controlled by environment variables or CLI flags, adding
         * ~500-800ms to every process start even when CUDA is not needed.
         *
         * Actual registration is deferred to ensureNvidiaFactoryRegistered()
         * which is called explicitly from DeviceGraphOrchestrator when a CUDA
         * context is first needed. This makes startup zero-cost when only
         * ROCm or CPU backends are used.
         */
        struct NvidiaFactoryRegistrar
        {
            NvidiaFactoryRegistrar()
            {
                // Intentionally empty - registration deferred to
                // ensureNvidiaFactoryRegistered() for controllable startup.
            }
        };

        // Static instance kept for linker inclusion (ensures TU is not stripped)
        static NvidiaFactoryRegistrar nvidia_factory_registrar;

    } // anonymous namespace

    // =========================================================================
    // Explicit registration (called when static init is unreliable)
    // =========================================================================

    void ensureNvidiaFactoryRegistered()
    {
        try
        {
            doRegisterNvidiaFactory();
        }
        catch (const std::exception &e)
        {
            LOG_WARN("[NvidiaContextFactory] Explicit registration failed: " << e.what());
        }
    }

} // namespace llaminar2
