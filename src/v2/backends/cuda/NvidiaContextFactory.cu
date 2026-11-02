/**
 * @file NvidiaContextFactory.cu
 * @brief CUDA factory registration for GPUDeviceContextPool
 *
 * This file explicitly registers the NVIDIA device context factory with the
 * GPUDeviceContextPool after immutable physical inventory admission. This allows
 * GPUDeviceContextPool (which lives in llaminar2_core) to create
 * NvidiaDeviceContext instances without directly depending on CUDA headers.
 * Cold discovery retains a short process-lifetime borrow before any native
 * query, so terminal retirement cannot race factory setup before a worker exists.
 *
 * NOTE: The static initializer is intentionally a no-op to avoid ~500-800ms
 * CUDA enumeration during process startup (before main). Registration is
 * deferred to ensureNvidiaFactoryRegistered(), called explicitly by
 * DeviceGraphOrchestrator when a CUDA context is first needed.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "../NativeContextRuntimePolicy.h"
#include "NvidiaDeviceContext.h"
#include "../NativeContextPreparation.h"
#include <mutex>
#include "../GPUDeviceContextPool.h"
#include "CUDAHardwareMetadata.h"
#include "CUDADriverApi.h"
#include "../DeviceUUID.h"
#include <span>
#include "../GPUDeviceProcessLifetime.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include <memory>

namespace llaminar2
{
    NativeContextBirthKey PreparedGPUContextFactory::cudaBirthKey(int ordinal)
    {
        // This is the post-inventory selected-worker population. Snapshot never
        // initializes metadata or creates an unrelated primary context.
        const auto& inventory=cudaHardwareMetadata();const auto& hardware=inventory.at(ordinal);
        auto key=commonBirthKey(DeviceId::cuda(ordinal));
        key.compute_units=hardware.sm_count;key.native_wave_width=hardware.native_warp_width;
        key.runtime_policy_sha256=NativeContextRuntimePolicy::original(DeviceType::CUDA).identity();
        key.physical_uuid=formatDeviceUUID(std::span<const char,16>{reinterpret_cast<const char*>(hardware.uuid.data()),16});
        key.architecture="sm_"+std::to_string(hardware.sm_major)+std::to_string(hardware.sm_minor);
        key.visible_device_count=inventory.size();std::string visibility;
        for(std::size_t i=0;i<inventory.size();++i){const auto& record=inventory.at(static_cast<int>(i));
            visibility+=formatDeviceUUID(std::span<const char,16>{reinterpret_cast<const char*>(record.uuid.data()),16})+";";}
        key.visibility_identity=digestIdentity(visibility+":"+NativeContextRuntimePolicy::original(DeviceType::CUDA).liveIdentity());
        int runtime=0,driver=0;
        if(cudaRuntimeGetVersion(&runtime)!=cudaSuccess || cudaDriverGetVersion(&driver)!=cudaSuccess || runtime<=0 || driver<=0)
            throw std::runtime_error("Selected CUDA worker lacks its actual runtime/driver version identity");
        key.runtime_version=std::to_string(runtime);key.driver_version=std::to_string(driver);
        key.runtime_sha256=libraryIdentity(reinterpret_cast<const void*>(&cudaRuntimeGetVersion));
        key.driver_sha256=libraryIdentity(reinterpret_cast<const void*>(CUDADriverApi::instance().init));
        key.backend_sha256=digestIdentity(libraryIdentity(reinterpret_cast<const void*>(&cublasCreate_v2))+":"+
            libraryIdentity(reinterpret_cast<const void*>(&cublasLtCreate)));
        key.requireComplete();return key;
    }
    PreparedGPUContextObjectGeometry PreparedGPUContextFactory::cudaObjectGeometry()
    {return {sizeof(NvidiaDeviceContext),alignof(NvidiaDeviceContext)};}
    std::shared_ptr<IWorkerGPUContext> PreparedGPUContextFactory::cuda(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> stack)
    {
        if(!stack)throw std::invalid_argument("Prepared original worker needs its source-owned stack");
        auto* storage=stack->contextObjectStorage(sizeof(NvidiaDeviceContext),alignof(NvidiaDeviceContext));
        auto* worker=new(storage) NvidiaDeviceContext(ordinal,stack);
        return std::shared_ptr<IWorkerGPUContext>(worker,[stack=std::move(stack)](IWorkerGPUContext* base){
            std::destroy_at(static_cast<NvidiaDeviceContext*>(base));
        });
    }
    /** @brief Pure creator registration; original constructor owns native initialization after source claims. */
    void ensureNvidiaPreparedFactoryRegistered()
    {
        static std::once_flag registered;
        std::call_once(registered,[]{GPUDeviceContextPool::instance().registerPreparedFactory(DeviceType::CUDA,PreparedGPUContextFactory::cuda);});
    }
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
            const auto &startup = debugEnv().backend_startup;
            if (startup.force_cpu_only)
            {
                LOG_INFO("[NvidiaContextFactory] Skipping CUDA factory registration (LLAMINAR_FORCE_CPU_ONLY_STARTUP=1)");
                return;
            }

            if (!startup.cudaEnabled())
            {
                LOG_INFO("[NvidiaContextFactory] Skipping CUDA factory registration (LLAMINAR_SKIP_CUDA_STARTUP=1)");
                return;
            }

            auto &pool = GPUDeviceContextPool::instance();
            if (pool.hasNvidiaSupport())
                return; // Already registered

            const auto discovery_lifetime = GPUDeviceProcessLifetime::acquireNativeContext();
            initializeCUDAHardwareMetadata();
            const int device_count = static_cast<int>(cudaHardwareMetadata().size());

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

    /**
     * @brief Register the exact admitted CUDA visibility after explicit startup.
     * @throws Discovery or inventory validation errors without hiding failed admission.
     */
    void ensureNvidiaFactoryRegistered()
    {
        doRegisterNvidiaFactory();
    }

} // namespace llaminar2
