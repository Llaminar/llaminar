/**
 * @file ROCmHardwareMetadata.cpp
 * @brief Native HIP source for the process-wide immutable hardware inventory.
 *
 * Property and PCI queries occur exactly once after the host-memory startup
 * policy is validated. Those HIP queries may initialize opaque native runtime
 * resources; this boundary issues no explicit set-device, reset, stream or
 * memory operation, and grants no capacity credit for opaque admission work.
 * Runtime resource generations remain owned by existing lifecycle machinery.
 * Driver errors reject the
 * whole inventory; they cannot become an invented empty visibility set.
 */
#include "ROCmHardwareMetadata.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "ROCmRuntimeStartup.h"
#include "backends/DevicePCIAddress.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>

namespace llaminar2
{
    namespace
    {
        /** @return Sole process-lifetime hardware publication authority. */
        ROCmHardwareAuthority &authority()
        {
            static ROCmHardwareAuthority instance;
            return instance;
        }
        /**
         * @brief Preserve the old backend precision-query answers without qualifying native variants.
         * @param isa Exact driver ISA identity.
         * @return Legacy compatibility booleans, never an instruction dispatch policy.
         */
        ROCmLegacyPrecisionQueries legacyPrecision(const std::string &isa)
        {
            const bool general = isa.find("gfx9") != std::string::npos ||
                                 isa.find("gfx10") != std::string::npos ||
                                 isa.find("gfx11") != std::string::npos;
            return {.fp16 = general,
                    .bf16 = isa.find("gfx90a") != std::string::npos ||
                            isa.find("gfx940") != std::string::npos ||
                            isa.find("gfx941") != std::string::npos ||
                            isa.find("gfx942") != std::string::npos,
                    .int8 = general};
        }
        /** @return Complete visible HIP records; a genuine no-device result is empty. */
        std::vector<ROCmDeviceHardware> readHardware()
        {
            // Bind terminal process exclusion before the first native query.
            // Only this once-source borrows lifecycle; published reads stay pure.
            const auto discovery_lifetime = GPUDeviceProcessLifetime::acquireNativeContext();
            int count = 0;
            const auto status = hipGetDeviceCount(&count);
            if (status == hipErrorNoDevice) return {};
            if (status != hipSuccess)
                throw std::runtime_error(std::string("ROCm hardware enumeration failed: ") + hipGetErrorString(status));
            if (count < 0)
                throw std::runtime_error("ROCm hardware enumeration returned negative count " + std::to_string(count));
            std::vector<ROCmDeviceHardware> devices;
            devices.reserve(static_cast<std::size_t>(count));
            for (int ordinal = 0; ordinal < count; ++ordinal)
            {
                hipDeviceProp_t properties{};
                const auto result = hipGetDeviceProperties(&properties, ordinal);
                if (result != hipSuccess)
                    throw std::runtime_error("ROCm physical metadata query failed at ordinal " +
                        std::to_string(ordinal) + ": " + hipGetErrorString(result));
                char pci[32]{};
                const auto pci_status = hipDeviceGetPCIBusId(pci, sizeof(pci), ordinal);
                if (pci_status != hipSuccess)
                    throw std::runtime_error("ROCm PCI identity query failed at ordinal " +
                        std::to_string(ordinal) + ": " + hipGetErrorString(pci_status));
                int wall_clock_rate_khz = 0;
                const auto clock_status = hipDeviceGetAttribute(&wall_clock_rate_khz,
                    hipDeviceAttributeWallClockRate, ordinal);
                if (clock_status != hipSuccess)
                    throw std::runtime_error("ROCm wall-clock frequency query failed at ordinal " +
                        std::to_string(ordinal) + ": " + hipGetErrorString(clock_status));
                ROCmDeviceHardware hardware{
                    .ordinal = ordinal,
                    .name = properties.name,
                    .isa_target = properties.gcnArchName,
                    .pci_address = DevicePCIAddress::parse(pci).toString(),
                    .pci_domain = properties.pciDomainID,
                    .pci_bus = properties.pciBusID,
                    .pci_device = properties.pciDeviceID,
                    .compute_units = properties.multiProcessorCount,
                    .native_wave_width = properties.warpSize,
                    .maximum_threads_per_block = properties.maxThreadsPerBlock,
                    .can_use_stream_wait_value = observeROCmStreamWaitValueCapability(ordinal,
                        [](int *supported, int device) {
                            return hipDeviceGetAttribute(supported, hipDeviceAttributeCanUseStreamWaitValue, device);
                        }),
                    .wall_clock_rate_khz = wall_clock_rate_khz,
                    .lds_per_block = properties.sharedMemPerBlock,
                    .total_memory = properties.totalGlobalMem,
                    .last_level_cache = static_cast<std::size_t>(std::max(properties.l2CacheSize, 0)),
                    .legacy_precision = legacyPrecision(properties.gcnArchName),
                };
                static_assert(sizeof(properties.uuid.bytes) == 16);
                std::memcpy(hardware.uuid.data(), properties.uuid.bytes, hardware.uuid.size());
                devices.push_back(std::move(hardware));
            }
            return devices;
        }
    }
    void initializeROCmHardwareMetadata()
    {
        requireROCmRuntimeStartup();
        authority().initialize(readHardware);
    }
    const ROCmHardwareInventory &rocmHardwareMetadata()
    {
        return authority().snapshot();
    }
}
