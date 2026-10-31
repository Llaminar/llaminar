/**
 * @file ROCmEnumeration.cpp
 * @brief ROCm device enumeration (separate compilation unit to avoid header conflicts)
 *
 * This file is compiled ONLY when HAVE_ROCM is defined, ensuring no CUDA header conflicts.
 * Compiled with hipcc to enable HIP runtime.
 * Preserve driver UUIDs instead of using rank-local ordinals as physical
 * identity: several ranks may legitimately observe the same accelerator.
 *
 * @author David Sanftenberg
 * Physical identity and limits share the admitted process inventory; live free
 * memory and checked primary-context reference-release requests remain runtime
 * observations. A successful release is not proof of context destruction or
 * returned capacity; complete opaque context ownership is a separate contract.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h> // For hipDevicePrimaryCtxRelease, hipDevicePrimaryCtxGetState
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include "ComputeBackend.h"
#include "GPUEnumeration.h"
#include "DeviceUUID.h"
#include "DevicePCIAddress.h"
#include "rocm/ROCmRuntimeStartup.h"
#include "rocm/ROCmHardwareMetadata.h"
#include "../utils/Logger.h"

namespace llaminar2
{
    namespace rocm_enumeration
    {

        namespace
        {
            /**
             * @brief Restore the exact caller ordinal and check existing primary-reference releases.
             *
             * Context-state observations precede every selection. A previously
             * inactive context is eligible for the existing release request only
             * after a successful selection. This scope never reports context
             * destruction, released VRAM or an opaque resource-admission proof.
             */
            class ROCmEnumerationDeviceScope final
            {
            public:
                /** @brief Authenticate current selection and all original context states before touching capacity. */
                explicit ROCmEnumerationDeviceScope(const ROCmHardwareInventory& inventory)
                    : contexts_(inventory.size())
                {
                    const auto status=hipGetDevice(&saved_);
                    if(status!=hipSuccess)
                        throw std::runtime_error(std::string("ROCm discovery cannot preserve current device: ")+hipGetErrorString(status));
                    (void)inventory.at(saved_);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                    for(std::size_t i=0;i<contexts_.size();++i){
                        auto& context=contexts_[i];
                        const auto device_status=hipDeviceGet(&context.device,static_cast<int>(i));
                        if(device_status!=hipSuccess)
                            throw std::runtime_error("ROCm context observation hipDeviceGet failed at ordinal "+std::to_string(i)+": "+hipGetErrorString(device_status));
                        unsigned int flags=0;int active=0;
                        const auto state_status=hipDevicePrimaryCtxGetState(context.device,&flags,&active);
                        if(state_status!=hipSuccess)
                            throw std::runtime_error("ROCm context observation hipDevicePrimaryCtxGetState failed at ordinal "+std::to_string(i)+": "+hipGetErrorString(state_status));
                        context.had_context=active!=0;
                    }
#pragma GCC diagnostic pop
                }
                /** @brief Unwinding must restore/release successfully; cleanup errors cannot become successful discovery. */
                ~ROCmEnumerationDeviceScope()
                {
                    if(!active_)return;
                    const auto result=cleanup();
                    if(result.restore!=hipSuccess || result.release!=hipSuccess){
                        LOG_ERROR("ROCm discovery cleanup failed: restore="<<hipGetErrorString(result.restore)
                            <<", primary-reference release ordinal="<<result.ordinal<<" status="<<hipGetErrorString(result.release));
                        std::terminate();
                    }
                }
                /** @brief Select one exact inventory member, retaining no successful status from a failed native call. */
                void select(int ordinal)
                {
                    auto& context=contexts_.at(static_cast<std::size_t>(ordinal));
                    const auto status=hipSetDevice(ordinal);
                    if(status!=hipSuccess)
                        throw std::runtime_error("ROCm capacity observation could not select ordinal "+std::to_string(ordinal)+": "+hipGetErrorString(status));
                    context.selected=true;
                }
                /** @brief Complete checked restoration/releases once before publishing any inventory projection. */
                void finish()
                {
                    const auto result=cleanup();
                    if(result.restore!=hipSuccess || result.release!=hipSuccess){
                        std::string error;
                        if(result.restore!=hipSuccess)
                            error="ROCm discovery failed to restore exact current ordinal "+std::to_string(saved_)+": "+hipGetErrorString(result.restore);
                        if(result.release!=hipSuccess){
                            if(!error.empty())error+="; ";
                            error+="ROCm discovery hipDevicePrimaryCtxRelease failed at ordinal "+std::to_string(result.ordinal)+": "+hipGetErrorString(result.release);
                        }
                        throw std::runtime_error(error);
                    }
                }
                ROCmEnumerationDeviceScope(const ROCmEnumerationDeviceScope&)=delete;
                ROCmEnumerationDeviceScope& operator=(const ROCmEnumerationDeviceScope&)=delete;
            private:
                /** @brief Original primary-context observation and whether this scope successfully selected it. */
                struct Context {hipDevice_t device{};bool had_context=false,selected=false;};
                /** @brief Native cleanup outcome, with no throwing allocation during unwinding. */
                struct Cleanup {hipError_t restore=hipSuccess,release=hipSuccess;int ordinal=-1;};
                /** @brief Restore once, then check every eligible release request without retrying native failures. */
                Cleanup cleanup()noexcept
                {
                    active_=false;Cleanup result;
                    result.restore=hipSetDevice(saved_);
                    if(result.restore==hipSuccess)contexts_[static_cast<std::size_t>(saved_)].selected=true;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                    for(std::size_t i=0;i<contexts_.size();++i){
                        const auto& context=contexts_[i];
                        if(!context.selected || context.had_context)continue;
                        const auto status=hipDevicePrimaryCtxRelease(context.device);
                        if(status!=hipSuccess && result.release==hipSuccess){result.release=status;result.ordinal=static_cast<int>(i);}
                    }
#pragma GCC diagnostic pop
                    return result;
                }
                std::vector<Context> contexts_;
                int saved_=-1;
                bool active_=true;
            };
        }

        /** @return Observed devices with immutable UUIDs and current capacities. */
        std::vector<ComputeDevice> enumerate_rocm_devices()
        {
            requireROCmRuntimeStartup();
            std::vector<ComputeDevice> devices;

            LOG_DEBUG("[ROCm] enumerate_rocm_devices() called");

            initializeROCmHardwareMetadata();
            const int device_count = static_cast<int>(rocmHardwareMetadata().size());

            if (device_count == 0)
            {
                LOG_DEBUG("[ROCm] No ROCm devices found");
                return devices;
            }

            LOG_DEBUG("[ROCm] Found " << device_count << " ROCm device(s)");

            // Preserve the exact current ordinal and acquire checked original
            // context observations before any volatile capacity selection.
            ROCmEnumerationDeviceScope current_device(rocmHardwareMetadata());

            for (int i = 0; i < device_count; ++i)
            {
                const auto &prop = rocmHardwareMetadata().at(i);

                ComputeDevice dev;
                dev.type = ComputeBackendType::GPU_ROCM;
                dev.device_id = i;
                dev.uuid = formatDeviceUUID(std::span<const char, 16>{
                    reinterpret_cast<const char *>(prop.uuid.data()), 16});
                // Preserve the driver's full endpoint, including PCI function.
                // Reconstructing from ordinal or link bottleneck loses identity.
                dev.pci_bus_address = prop.pci_address;
                dev.compute_units = prop.compute_units;
                dev.last_level_cache_bytes = prop.last_level_cache > 0 ? static_cast<size_t>(prop.last_level_cache) : 0;
                dev.total_memory_bytes = prop.total_memory;

                // Parse gcnArchName for architecture info
                // Format: "gfx906:sramecc+:xnack-" or similar
                std::string arch_name = prop.isa_target;
                std::string gfx_name;

                // Extract gfx number (e.g., "gfx906" -> "906")
                size_t gfx_pos = arch_name.find("gfx");
                if (gfx_pos != std::string::npos)
                {
                    size_t end_pos = arch_name.find_first_of(":", gfx_pos);
                    if (end_pos == std::string::npos)
                    {
                        end_pos = arch_name.length();
                    }
                    gfx_name = arch_name.substr(gfx_pos, end_pos - gfx_pos);
                }

                // Build device name
                dev.name = std::string(prop.name);
                if (!gfx_name.empty())
                {
                    dev.name += " (" + gfx_name + ")";
                }

                // Compute capability equivalent (for sorting/comparison)
                // gfx906 -> 906, gfx90a -> 910, gfx940 -> 940
                int arch_num = 0;
                if (gfx_name.length() > 3)
                {
                    std::string num_part = gfx_name.substr(3);
                    // Handle hex-like suffixes (90a -> 910)
                    if (num_part.back() == 'a')
                    {
                        num_part.back() = '0';
                        arch_num = std::stoi(num_part) + 10;
                    }
                    else
                    {
                        try
                        {
                            arch_num = std::stoi(num_part);
                        }
                        catch (...)
                        {
                            arch_num = 900; // Default
                        }
                    }
                }
                dev.compute_capability = arch_num;

                // Get free memory
                size_t free_bytes = 0, total_bytes = 0;
                current_device.select(i);
                const auto memory_status=hipMemGetInfo(&free_bytes,&total_bytes);
                if(memory_status!=hipSuccess)
                    throw std::runtime_error("ROCm capacity observation failed at ordinal "+std::to_string(i)+": "+hipGetErrorString(memory_status));
                dev.free_memory_bytes=free_bytes;

                // Feature detection based on architecture
                // gfx900+ supports FP16, gfx908+ supports BF16 and matrix cores
                dev.supports_fp16 = (arch_num >= 900);
                dev.supports_bf16 = (arch_num >= 908);
                dev.supports_int8 = (arch_num >= 906);

                LOG_DEBUG("[ROCm] Device " << i << ": " << dev.name
                                           << " (" << (dev.total_memory_bytes / (1024 * 1024 * 1024)) << " GB)");

                // Read PCIe link information from sysfs
                dev.pcie = pcie_enumeration::read_pcie_link_info(
                    prop.pci_domain, prop.pci_bus, prop.pci_device);

                devices.push_back(dev);
            }

            // No capacity projection escapes after failed restoration or an
            // unchecked primary-reference release. This proves no destruction.
            current_device.finish();

            return devices;
        }

        /** @return Observed PCIe NUMA affinity, or -1 when sysfs has no locality. */
        int get_rocm_device_numa_node(int device_id)
        {
            initializeROCmHardwareMetadata();
            const auto &prop = rocmHardwareMetadata().at(device_id);

            // Read from sysfs using PCI location
            char path[256];
            snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node",
                     prop.pci_address.c_str());

            FILE *f = fopen(path, "r");
            if (!f)
            {
                return -1;
            }

            int numa_node = -1;
            if (fscanf(f, "%d", &numa_node) != 1)
            {
                numa_node = -1;
            }
            fclose(f);

            return numa_node;
        }

        /** @return Directed driver P2P edges in the supplied ordinal order. */
        P2PMatrix query_p2p_matrix(const std::vector<ComputeDevice> &devices)
        {
            P2PMatrix matrix;
            matrix.backend = ComputeBackendType::GPU_ROCM;
            const int n = static_cast<int>(devices.size());
            matrix.device_ids.resize(n);
            matrix.can_access.resize(n, std::vector<bool>(n, false));

            for (int i = 0; i < n; ++i)
            {
                matrix.device_ids[i] = devices[i].device_id;
                matrix.can_access[i][i] = true;
            }

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (i == j)
                        continue;
                    int can_access = 0;
                    hipError_t err = hipDeviceCanAccessPeer(&can_access,
                                                            devices[i].device_id, devices[j].device_id);
                    if(err!=hipSuccess)
                        throw std::runtime_error("ROCm P2P observation failed for ordinals "+std::to_string(devices[i].device_id)+
                            " -> "+std::to_string(devices[j].device_id)+": "+hipGetErrorString(err));
                    matrix.can_access[i][j]=can_access!=0;
                }
            }

            return matrix;
        }

    } // namespace rocm_enumeration
} // namespace llaminar2
