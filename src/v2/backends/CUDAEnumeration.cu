/**
 * @file CUDAEnumeration.cu
 * @brief CUDA device enumeration (separate compilation unit to avoid header conflicts)
 *
 * This file is compiled ONLY when HAVE_CUDA is defined, ensuring no HIP header conflicts.
 *
 * @author David Sanftenberg
 */

#include <cuda_runtime.h>
#include <cuda.h> // For cuDevicePrimaryCtxRelease, cuDevicePrimaryCtxGetState
#include <cublas_v2.h>
#include <vector>
#include <string>
#include "ComputeBackend.h"
#include "GPUEnumeration.h"
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

            // Save current device to restore after enumeration
            // (enumeration changes current device via cudaSetDevice for memory queries)
            int original_device = 0;
            cudaGetDevice(&original_device);

            // Track which devices already had an active primary context before
            // we touch them.  We will release contexts we created so that
            // enumeration-only processes don't keep ~256 MiB per GPU alive.
            std::vector<bool> had_context(device_count, false);
            for (int i = 0; i < device_count; ++i)
            {
                unsigned int flags = 0;
                int active = 0;
                CUdevice cu_dev;
                if (cuDeviceGet(&cu_dev, i) == CUDA_SUCCESS &&
                    cuDevicePrimaryCtxGetState(cu_dev, &flags, &active) == CUDA_SUCCESS)
                {
                    had_context[i] = (active != 0);
                }
            }

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

                // Read PCIe link information from sysfs
                dev.pcie = pcie_enumeration::read_pcie_link_info(
                    prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

                devices.push_back(dev);
            }

            // Restore original device context to avoid disrupting caller's CUDA state
            cudaSetDevice(original_device);

            // Release primary contexts that we created during enumeration.
            // cudaSetDevice() implicitly retains the primary context; if we
            // don't release it, each enumerated device keeps ~256 MiB of GPU
            // memory allocated even when the process never uses that GPU.
            // This is safe: cuDevicePrimaryCtxRelease only decrements the
            // refcount.  If another part of the process already held the
            // context, it stays alive.
            for (int i = 0; i < device_count; ++i)
            {
                if (!had_context[i])
                {
                    CUdevice cu_dev;
                    if (cuDeviceGet(&cu_dev, i) == CUDA_SUCCESS)
                    {
                        cuDevicePrimaryCtxRelease(cu_dev);
                    }
                }
            }

            return devices;
        }

        int get_cuda_device_numa_node(int device_id)
        {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
            {
                return -1;
            }

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
            matrix.backend = ComputeBackendType::GPU_CUDA;
            const int n = static_cast<int>(devices.size());
            matrix.device_ids.resize(n);
            matrix.can_access.resize(n, std::vector<bool>(n, false));

            for (int i = 0; i < n; ++i)
            {
                matrix.device_ids[i] = devices[i].device_id;
                matrix.can_access[i][i] = true; // self-access always works
            }

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (i == j)
                        continue;
                    int can_access = 0;
                    cudaError_t err = cudaDeviceCanAccessPeer(&can_access,
                                                              devices[i].device_id, devices[j].device_id);
                    if (err == cudaSuccess && can_access)
                    {
                        matrix.can_access[i][j] = true;
                    }
                }
            }

            return matrix;
        }

    } // namespace cuda_enumeration
} // namespace llaminar2
