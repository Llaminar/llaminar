/**
 * @file CUDAHardwareMetadata.cu
 * @brief Sole checked native reader for immutable process-visible CUDA facts.
 *
 * Explicit startup may initialize opaque driver/runtime resources; it issues no
 * allocation, stream, set-device or reset operation and grants no memory credit.
 * All records validate before publication. Free-capacity and loaded-function
 * observations retain their distinct admission and runtime-generation owners.
 */
#include "../NativeContextRuntimePolicy.h"
#include "CUDAHardwareMetadata.h"
#include "CUDADiscoveryAdmission.h"
#include "CUDADriverApi.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

namespace llaminar2
{
    namespace
    {
        /** @return One process-lifetime physical publication owner. */
        CUDAHardwareAuthority &authority()
        {
            static CUDAHardwareAuthority instance;
            return instance;
        }
        /**
         * @brief Enumerate one complete native visibility set without selecting a context.
         * @return Exact ordered records; any genuine no-device result is empty.
         * @throws std::runtime_error On a failed native query, never publishing a partial set.
         *
         * Admission follows the typed discovery policy in CUDADiscoveryAdmission.h:
         * only a positively driverless member (the loader finds no libcuda.so.1
         * anywhere) may answer an insufficient-driver runtime status with an
         * empty visibility set, because such a member has no CUDA capability at
         * all and full-backend binaries must keep serving CPU-only cluster
         * members. Any present driver installation that the runtime cannot use,
         * a system driver mismatch, and every other query failure keep the
         * original fatal fault; an explicit CUDA request still fails at
         * admission with its own device diagnostic.
         */
        std::vector<CUDADeviceHardware> readHardware()
        {
            (void)NativeContextRuntimePolicy::original(DeviceType::CUDA).identity();
            // Bind terminal process exclusion before the first native query.
            // Only this once-source borrows lifecycle; published reads stay pure.
            const auto discovery_lifetime = GPUDeviceProcessLifetime::acquireNativeContext();
            int count = 0;
            const auto status = cudaGetDeviceCount(&count);
            if (status == cudaErrorNoDevice) return {};
            if (status == cudaErrorInsufficientDriver &&
                classifyCUDADiscovery(CUDADeviceCountStatus::InsufficientDriver,
                    CUDADriverApi::probeDriverBinding()) == CUDADiscoveryDisposition::EmptyVisibility)
            {
                LOG_WARN("CUDA discovery found no installed NVIDIA driver ("
                    << cudaGetErrorString(status)
                    << "); this member publishes an empty CUDA visibility set");
                return {};
            }
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("CUDA hardware enumeration failed: ") + cudaGetErrorString(status));
            if (count < 0)
                throw std::runtime_error("CUDA hardware enumeration returned negative count " + std::to_string(count));
            std::vector<CUDADeviceHardware> devices;
            devices.reserve(static_cast<std::size_t>(count));
            for (int ordinal = 0; ordinal < count; ++ordinal)
            {
                cudaDeviceProp properties{};
                const auto result = cudaGetDeviceProperties(&properties, ordinal);
                if (result != cudaSuccess)
                    throw std::runtime_error("CUDA physical metadata query failed at ordinal " +
                        std::to_string(ordinal) + ": " + cudaGetErrorString(result));
                char pci[32]{};
                const auto pci_status = cudaDeviceGetPCIBusId(pci, sizeof(pci), ordinal);
                if (pci_status != cudaSuccess)
                    throw std::runtime_error("CUDA PCI identity query failed at ordinal " +
                        std::to_string(ordinal) + ": " + cudaGetErrorString(pci_status));
                CUDADeviceHardware hardware{
                    .ordinal = ordinal, .name = properties.name,
                    .pci_address = DevicePCIAddress::parse(pci).toString(),
                    .pci_domain = properties.pciDomainID, .pci_bus = properties.pciBusID,
                    .pci_device = properties.pciDeviceID,
                    .sm_major = properties.major, .sm_minor = properties.minor,
                    .sm_count = properties.multiProcessorCount, .native_warp_width = properties.warpSize,
                    .maximum_threads_per_block = properties.maxThreadsPerBlock,
                    .maximum_threads_per_sm = properties.maxThreadsPerMultiProcessor,
                    .maximum_blocks_per_sm = properties.maxBlocksPerMultiProcessor,
                    .registers_per_block = properties.regsPerBlock,
                    .registers_per_sm = properties.regsPerMultiprocessor,
                    .shared_memory_per_block = properties.sharedMemPerBlock,
                    .shared_memory_optin_per_block = properties.sharedMemPerBlockOptin,
                    .shared_memory_per_sm = properties.sharedMemPerMultiprocessor,
                    .total_memory = properties.totalGlobalMem,
                    .last_level_cache = static_cast<std::size_t>(std::max(properties.l2CacheSize, 0)),
                };
                static_assert(sizeof(properties.uuid.bytes) == 16);
                std::memcpy(hardware.uuid.data(), properties.uuid.bytes, hardware.uuid.size());
                devices.push_back(std::move(hardware));
            }
            return devices;
        }
    }
    void initializeCUDAHardwareMetadata()
    {
        if (!debugEnv().backend_startup.cudaEnabled())
            throw std::logic_error("CUDA physical metadata requested while CUDA startup is disabled");
        authority().initialize(readHardware);
    }
    const CUDAHardwareInventory &cudaHardwareMetadata()
    {
        return authority().snapshot();
    }
}
