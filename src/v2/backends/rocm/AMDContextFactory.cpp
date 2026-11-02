/**
 * @file AMDContextFactory.cpp
 * @brief ROCm/HIP factory registration for GPUDeviceContextPool
 *
 * This file registers the AMD device context factory with the
 * GPUDeviceContextPool at explicit startup after immutable inventory admission. This allows
 * GPUDeviceContextPool (which lives in llaminar2_core) to create
 * AMDDeviceContext instances without directly depending on HIP headers.
 *
 * NOTE: The static initializer (AMDFactoryRegistrar) may be stripped by the
 * linker when this TU is in a static library and no symbols are referenced.
 * The public function ensureAMDFactoryRegistered() provides an explicit
 * registration path that also forces the linker to include this TU.
 * Cold discovery retains a short process-lifetime borrow before native queries;
 * it cannot race terminal retirement before the first worker owns its own lease.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "../NativeContextRuntimePolicy.h"
#include <dlfcn.h>
#include "AMDDeviceContext.h"
#include "../NativeContextPreparation.h"
#include <mutex>
#include "../GPUDeviceContextPool.h"
#include "ROCmHardwareMetadata.h"
#include "../DeviceUUID.h"
#include <span>
#include <fstream>
#include <sstream>
#include <sys/utsname.h>
#include "../GPUDeviceProcessLifetime.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include <memory>

namespace llaminar2
{
    NativeContextBirthKey PreparedGPUContextFactory::rocmBirthKey(int ordinal)
    {
        // Original immutable startup snapshot only; HIP_INIT's separate global
        // population must not be mistaken for newly owned selected-worker bytes.
        const auto& inventory=rocmHardwareMetadata();const auto& hardware=inventory.at(ordinal);
        auto key=commonBirthKey(DeviceId::rocm(ordinal));
        key.compute_units=hardware.compute_units;key.native_wave_width=hardware.native_wave_width;
        key.runtime_policy_sha256=NativeContextRuntimePolicy::original(DeviceType::ROCm).identity();
        key.physical_uuid=formatDeviceUUID(std::span<const char,16>{reinterpret_cast<const char*>(hardware.uuid.data()),16});
        key.architecture=hardware.isa_target;key.visible_device_count=inventory.size();std::string visibility;
        for(std::size_t i=0;i<inventory.size();++i){const auto& record=inventory.at(static_cast<int>(i));
            visibility+=formatDeviceUUID(std::span<const char,16>{reinterpret_cast<const char*>(record.uuid.data()),16})+";";}
        key.visibility_identity=digestIdentity(visibility+":"+NativeContextRuntimePolicy::original(DeviceType::ROCm).liveIdentity());
        int runtime=0;if(hipRuntimeGetVersion(&runtime)!=hipSuccess || runtime<=0)
            throw std::runtime_error("Selected ROCm worker lacks its actual runtime identity");
        const auto read=[](const char* path){std::ifstream file(path);std::ostringstream text;text<<file.rdbuf();
            if(!file || text.str().empty())throw std::runtime_error("Selected ROCm worker lacks original kernel-driver source identity");return text.str();};
        struct utsname kernel{};if(uname(&kernel)!=0)throw std::runtime_error("Selected ROCm worker lacks kernel release identity");
        key.runtime_version=std::to_string(runtime);key.driver_version=read("/sys/module/amdgpu/version")+":"+kernel.release;
        key.driver_sha256=digestIdentity(key.driver_version+":"+read("/sys/module/amdgpu/srcversion"));
        key.hsa_sha256=libraryClosureIdentity(dlsym(RTLD_DEFAULT,"hsa_init"));
        key.runtime_sha256=libraryIdentity(reinterpret_cast<const void*>(&hipRuntimeGetVersion));
        key.backend_sha256=digestIdentity(libraryIdentity(reinterpret_cast<const void*>(&hipblasCreate))+":"+
            libraryIdentity(reinterpret_cast<const void*>(&hipblasLtCreate)));
        key.requireComplete();return key;
    }
    PreparedGPUContextObjectGeometry PreparedGPUContextFactory::rocmObjectGeometry()
    {return {sizeof(AMDDeviceContext),alignof(AMDDeviceContext)};}
    std::shared_ptr<IWorkerGPUContext> PreparedGPUContextFactory::rocm(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> stack)
    {
        if(!stack)throw std::invalid_argument("Prepared original worker needs its source-owned stack");
        auto* storage=stack->contextObjectStorage(sizeof(AMDDeviceContext),alignof(AMDDeviceContext));
        auto* worker=new(storage) AMDDeviceContext(ordinal,stack);
        return std::shared_ptr<IWorkerGPUContext>(worker,[stack=std::move(stack)](IWorkerGPUContext* base){
            std::destroy_at(static_cast<AMDDeviceContext*>(base));
        });
    }
    /** @brief Pure creator registration; original constructor owns native initialization after source claims. */
    void ensureAMDPreparedFactoryRegistered()
    {
        static std::once_flag registered;
        std::call_once(registered,[]{GPUDeviceContextPool::instance().registerPreparedFactory(DeviceType::ROCm,PreparedGPUContextFactory::rocm);});
    }
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
            const auto &startup = debugEnv().backend_startup;
            if (startup.force_cpu_only)
            {
                LOG_INFO("[AMDContextFactory] Skipping ROCm factory registration (LLAMINAR_FORCE_CPU_ONLY_STARTUP=1)");
                return;
            }

            if (!startup.rocmEnabled())
            {
                LOG_INFO("[AMDContextFactory] Skipping ROCm factory registration (LLAMINAR_SKIP_ROCM_STARTUP=1)");
                return;
            }

            auto &pool = GPUDeviceContextPool::instance();
            if (pool.hasAMDSupport())
                return; // Already registered

            const auto discovery_lifetime = GPUDeviceProcessLifetime::acquireNativeContext();
            initializeROCmHardwareMetadata();
            const int device_count = static_cast<int>(rocmHardwareMetadata().size());

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

    /**
     * @brief Register exact admitted ROCm visibility at the enabled startup boundary.
     * @throws Discovery, lifecycle or inventory errors without hiding failed admission.
     */
    void ensureAMDFactoryRegistered()
    {
        doRegisterAMDFactory();
    }

} // namespace llaminar2
