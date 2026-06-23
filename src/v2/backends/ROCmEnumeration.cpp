/**
 * @file ROCmEnumeration.cpp
 * @brief ROCm device enumeration (separate compilation unit to avoid header conflicts)
 *
 * This file is compiled ONLY when HAVE_ROCM is defined, ensuring no CUDA header conflicts.
 * Compiled with hipcc to enable HIP runtime.
 *
 * @author David Sanftenberg
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h> // For hipDevicePrimaryCtxRelease, hipDevicePrimaryCtxGetState
#include <hipblas/hipblas.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include "ComputeBackend.h"
#include "GPUEnumeration.h"
#include "../utils/Logger.h"

namespace llaminar2
{
    namespace rocm_enumeration
    {

        std::vector<ComputeDevice> enumerate_rocm_devices()
        {
            std::vector<ComputeDevice> devices;

            LOG_DEBUG("[ROCm] enumerate_rocm_devices() called");

            int device_count = 0;
            hipError_t err = hipGetDeviceCount(&device_count);

            if (err != hipSuccess)
            {
                LOG_DEBUG("[ROCm] hipGetDeviceCount failed: " << hipGetErrorString(err));
                return devices;
            }

            if (device_count == 0)
            {
                LOG_DEBUG("[ROCm] No ROCm devices found");
                return devices;
            }

            LOG_DEBUG("[ROCm] Found " << device_count << " ROCm device(s)");

            // Save the current device so we can restore it after enumeration
            int saved_device = 0;
            (void)hipGetDevice(&saved_device);

            // Track which devices already had an active primary context before
            // we touch them.  We will release contexts we created so that
            // enumeration-only processes don't keep GPU memory alive.
            std::vector<bool> had_context(device_count, false);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            for (int i = 0; i < device_count; ++i)
            {
                unsigned int flags = 0;
                int active = 0;
                hipDevice_t hip_dev;
                if (hipDeviceGet(&hip_dev, i) == hipSuccess &&
                    hipDevicePrimaryCtxGetState(hip_dev, &flags, &active) == hipSuccess)
                {
                    had_context[i] = (active != 0);
                }
            }
#pragma GCC diagnostic pop

            for (int i = 0; i < device_count; ++i)
            {
                hipDeviceProp_t prop;
                if (hipGetDeviceProperties(&prop, i) != hipSuccess)
                {
                    continue;
                }

                ComputeDevice dev;
                dev.type = ComputeBackendType::GPU_ROCM;
                dev.device_id = i;
                dev.total_memory_bytes = prop.totalGlobalMem;

                // Parse gcnArchName for architecture info
                // Format: "gfx906:sramecc+:xnack-" or similar
                std::string arch_name = prop.gcnArchName;
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
                if (hipSetDevice(i) == hipSuccess)
                {
                    (void)hipMemGetInfo(&free_bytes, &total_bytes);
                    dev.free_memory_bytes = free_bytes;
                }
                else
                {
                    dev.free_memory_bytes = dev.total_memory_bytes;
                }

                // Feature detection based on architecture
                // gfx900+ supports FP16, gfx908+ supports BF16 and matrix cores
                dev.supports_fp16 = (arch_num >= 900);
                dev.supports_bf16 = (arch_num >= 908);
                dev.supports_int8 = (arch_num >= 906);

                LOG_DEBUG("[ROCm] Device " << i << ": " << dev.name
                                           << " (" << (dev.total_memory_bytes / (1024 * 1024 * 1024)) << " GB)");

                // Read PCIe link information from sysfs
                dev.pcie = pcie_enumeration::read_pcie_link_info(
                    prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

                devices.push_back(dev);
            }

            // Restore the original device to avoid side effects on callers
            (void)hipSetDevice(saved_device);

            // Release primary contexts that we created during enumeration.
            // hipSetDevice() implicitly retains the primary context; if we
            // don't release it, each enumerated device keeps GPU memory
            // allocated even when the process never uses that GPU.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            for (int i = 0; i < device_count; ++i)
            {
                if (!had_context[i])
                {
                    hipDevice_t hip_dev;
                    if (hipDeviceGet(&hip_dev, i) == hipSuccess)
                    {
                        (void)hipDevicePrimaryCtxRelease(hip_dev);
                    }
                }
            }
#pragma GCC diagnostic pop

            return devices;
        }

        int get_rocm_device_numa_node(int device_id)
        {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, device_id) != hipSuccess)
            {
                return -1;
            }

            // Read from sysfs using PCI location
            char path[256];
            snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node",
                     prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

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
                    if (err == hipSuccess && can_access)
                    {
                        matrix.can_access[i][j] = true;
                    }
                }
            }

            return matrix;
        }

    } // namespace rocm_enumeration
} // namespace llaminar2
