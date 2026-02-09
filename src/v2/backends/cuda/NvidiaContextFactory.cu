/**
 * @file NvidiaContextFactory.cu
 * @brief CUDA factory registration for GPUDeviceContextPool
 *
 * This file registers the NVIDIA device context factory with the
 * GPUDeviceContextPool during static initialization. This allows
 * GPUDeviceContextPool (which lives in llaminar2_core) to create
 * NvidiaDeviceContext instances without directly depending on CUDA headers.
 *
 * NOTE: The static initializer works reliably here because nvcc device linking
 * forces inclusion of all .cu object files. The public function
 * ensureNvidiaFactoryRegistered() is provided for symmetry with the AMD path.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "NvidiaDeviceContext.h"
#include "../GPUDeviceContextPool.h"
#include "../GPUEnumeration.h"
#include "../../utils/Logger.h"
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
         * This struct's constructor runs during static initialization (before main)
         * to register the NVIDIA factory with GPUDeviceContextPool.
         */
        struct NvidiaFactoryRegistrar
        {
            NvidiaFactoryRegistrar()
            {
                try
                {
                    doRegisterNvidiaFactory();
                }
                catch (const std::exception &e)
                {
                    LOG_WARN("[NvidiaContextFactory] Failed to register CUDA factory: " << e.what());
                }
            }
        };

        // Static instance triggers registration during library load
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
