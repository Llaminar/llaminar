/**
 * @file CUDAEnumeration.cu
 * @brief CUDA device enumeration (separate compilation unit to avoid header conflicts)
 *
 * This file is compiled ONLY when HAVE_CUDA is defined, ensuring no HIP header conflicts.
 * Driver UUIDs remain independent of visible ordinals so MPI discovery cannot
 * count two process views of one GPU as two physical allocations. Immutable
 * properties come only from admitted CUDA metadata; current free capacity is
 * a separate checked setup observation and never a cached physical fact.
 *
 * @author David Sanftenberg
 */

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <exception>
#include <span>
#include "cuda/CUDAHardwareMetadata.h"
#include "ComputeBackend.h"
#include "GPUEnumeration.h"
#include "DeviceUUID.h"
#include "DevicePCIAddress.h"
#include "../utils/Logger.h"

namespace llaminar2
{
    namespace cuda_enumeration
    {

        namespace
        {
            /** @brief Restore the exact native current device on observation failure. */
            class CUDAEnumerationDeviceScope final
            {
            public:
                /** @brief Authenticate the caller's current ordinal before any selection. */
                CUDAEnumerationDeviceScope()
                {
                    const auto status = cudaGetDevice(&saved_);
                    if (status != cudaSuccess)
                        throw std::runtime_error(std::string("CUDA discovery cannot preserve current device: ") +
                                                 cudaGetErrorString(status));
                    (void)cudaHardwareMetadata().at(saved_);
                }
                /** @brief Restore during failure unwinding; failed physical restoration is fatal. */
                ~CUDAEnumerationDeviceScope()
                {
                    if (active_ && cudaSetDevice(saved_) != cudaSuccess)
                    {
                        LOG_ERROR("CUDA discovery failed to restore exact current ordinal " << saved_);
                        std::terminate();
                    }
                }
                /**
                 * @brief Restore before publishing the capacity projection.
                 * @throws std::runtime_error On a failed native restoration, without retry.
                 */
                void restore()
                {
                    active_ = false;
                    const auto status = cudaSetDevice(saved_);
                    if (status != cudaSuccess)
                        throw std::runtime_error("CUDA discovery failed to restore exact current ordinal " +
                            std::to_string(saved_) + ": " + cudaGetErrorString(status));
                }
                CUDAEnumerationDeviceScope(const CUDAEnumerationDeviceScope &) = delete;
                CUDAEnumerationDeviceScope &operator=(const CUDAEnumerationDeviceScope &) = delete;
            private:
                int saved_ = -1;
                bool active_ = true;
            };
        }

        /** @return Observed devices with immutable UUIDs and current capacities. */
        std::vector<ComputeDevice> enumerate_cuda_devices()
        {
            std::vector<ComputeDevice> devices;

            initializeCUDAHardwareMetadata();
            const auto &inventory = cudaHardwareMetadata();
            const int device_count = static_cast<int>(inventory.size());
            if (device_count == 0)
                return devices;

            // Volatile capacity observations retain their explicit context
            // selection scope; physical discovery itself never selects one.
            CUDAEnumerationDeviceScope current_device;
            for (int i = 0; i < device_count; ++i)
            {
                const auto &prop = inventory.at(i);
                ComputeDevice dev;
                dev.type = ComputeBackendType::GPU_CUDA;
                dev.name = prop.name;
                dev.device_id = i;
                dev.uuid = formatDeviceUUID(std::span<const char, 16>{
                    reinterpret_cast<const char *>(prop.uuid.data()), 16});
                dev.pci_bus_address = prop.pci_address;
                dev.compute_capability = prop.sm_major * 10 + prop.sm_minor;
                dev.compute_units = prop.sm_count;
                dev.last_level_cache_bytes = prop.last_level_cache;
                dev.total_memory_bytes = prop.total_memory;

                // Query failure cannot become an invented full-capacity GPU.
                size_t free_bytes = 0, total_bytes = 0;
                const auto select_status = cudaSetDevice(i);
                if (select_status != cudaSuccess)
                    throw std::runtime_error("CUDA capacity observation could not select ordinal " +
                        std::to_string(i) + ": " + cudaGetErrorString(select_status));
                const auto memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
                if (memory_status != cudaSuccess)
                    throw std::runtime_error("CUDA capacity observation failed at ordinal " +
                        std::to_string(i) + ": " + cudaGetErrorString(memory_status));
                dev.free_memory_bytes = free_bytes;

                // Feature detection based on compute capability
                dev.supports_fp16 = (prop.sm_major >= 6); // Pascal (SM 6.0+)
                dev.supports_bf16 = (prop.sm_major >= 8); // Ampere (SM 8.0+)
                dev.supports_int8 = (prop.sm_major >= 6); // DP4A on Pascal+

                LOG_DEBUG("[CUDA] Device " << i << ": " << dev.name
                                           << " (SM " << prop.sm_major << "." << prop.sm_minor
                                           << ", " << (dev.total_memory_bytes / (1024 * 1024 * 1024)) << " GB)");

                // Read PCIe link information from sysfs
                dev.pcie = pcie_enumeration::read_pcie_link_info(
                    prop.pci_domain, prop.pci_bus, prop.pci_device);

                devices.push_back(dev);
            }

            // Restore original device context to avoid disrupting caller's CUDA state
            current_device.restore();

            return devices;
        }

        /** @return Observed PCIe NUMA affinity, or -1 when sysfs has no locality. */
        int get_cuda_device_numa_node(int device_id)
        {
            const auto &prop = cudaHardwareMetadata().at(device_id);
            const std::string path = "/sys/bus/pci/devices/" + prop.pci_address + "/numa_node";

            FILE *f = fopen(path.c_str(), "r");
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
                    if (err != cudaSuccess)
                        throw std::runtime_error("CUDA P2P observation failed for ordinals " +
                            std::to_string(devices[i].device_id) + " -> " +
                            std::to_string(devices[j].device_id) + ": " + cudaGetErrorString(err));
                    matrix.can_access[i][j] = can_access != 0;
                }
            }

            return matrix;
        }

    } // namespace cuda_enumeration
} // namespace llaminar2
