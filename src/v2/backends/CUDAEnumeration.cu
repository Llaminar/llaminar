/**
 * @file CUDAEnumeration.cu
 * @brief CUDA device enumeration (separate compilation unit to avoid header conflicts)
 *
 * This file is compiled ONLY when HAVE_CUDA is defined, ensuring no HIP header conflicts.
 *
 * @author David Sanftenberg
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include "ComputeBackend.h"
#include "../utils/Logger.h"

namespace llaminar2
{
    namespace cuda_enumeration
    {

        std::vector<ComputeDevice> enumerate_cuda_devices()
        {
            std::vector<ComputeDevice> devices;

            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);

            if (err != cudaSuccess || device_count == 0)
            {
                LOG_DEBUG("[CUDA] No CUDA devices found or cudaGetDeviceCount failed: "
                          << cudaGetErrorString(err));
                return devices;
            }

            LOG_INFO("[CUDA] Found " << device_count << " CUDA device(s)");

            for (int i = 0; i < device_count; ++i)
            {
                cudaDeviceProp prop;
                if (cudaGetDeviceProperties(&prop, i) != cudaSuccess)
                {
                    continue;
                }

                ComputeDevice dev;
                dev.type = ComputeBackendType::GPU_CUDA;
                dev.name = std::string(prop.name);
                dev.device_id = i;
                dev.compute_capability = prop.major * 10 + prop.minor;
                dev.total_memory_bytes = prop.totalGlobalMem;

                // Get free memory
                size_t free_bytes = 0, total_bytes = 0;
                if (cudaSetDevice(i) == cudaSuccess)
                {
                    cudaMemGetInfo(&free_bytes, &total_bytes);
                    dev.free_memory_bytes = free_bytes;
                }
                else
                {
                    dev.free_memory_bytes = dev.total_memory_bytes;
                }

                // Feature detection based on compute capability
                dev.supports_fp16 = (prop.major >= 6); // Pascal (SM 6.0+)
                dev.supports_bf16 = (prop.major >= 8); // Ampere (SM 8.0+)
                dev.supports_int8 = (prop.major >= 6); // DP4A on Pascal+

                LOG_INFO("[CUDA] Device " << i << ": " << dev.name
                                          << " (SM " << prop.major << "." << prop.minor
                                          << ", " << (dev.total_memory_bytes / (1024 * 1024 * 1024)) << " GB)");

                devices.push_back(dev);
            }

            return devices;
        }

        int get_cuda_device_numa_node(int device_id)
        {
            // Try to get NUMA node from PCI bus location
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
            {
                return -1;
            }

            // Read from sysfs: /sys/bus/pci/devices/0000:XX:XX.X/numa_node
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

    } // namespace cuda_enumeration
} // namespace llaminar2
