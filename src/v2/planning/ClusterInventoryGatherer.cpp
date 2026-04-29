/**
 * @file ClusterInventoryGatherer.cpp
 * @brief Implementation of cluster-wide device inventory gathering
 *
 * Extracted from OrchestrationRunner::gatherClusterInventory().
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include "planning/ClusterInventoryGatherer.h"
#include "backends/ComputeBackend.h"
#include "utils/Logger.h"
#include "utils/MPITopology.h"
#include "utils/NodeDetection.h"
#include "utils/NUMATopology.h"

#include <mpi.h>
#include <unistd.h>

namespace llaminar2
{

    ClusterInventory gatherClusterInventory(
        const std::shared_ptr<IMPIContext> &mpi_ctx,
        const std::vector<GlobalDeviceAddress> &explicit_tp_devices,
        const std::string &hostfile)
    {
        ClusterInventory inventory;

        // Ensure DeviceManager is initialized with NUMA-aware filtering by default.
        // This avoids accidentally broadening visibility to cross-socket devices in local execution.
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
        {
            auto numa_info = NUMATopology::detectLocalNUMANode();
            int target_numa_node = 0;
            if (numa_info.detection_succeeded && numa_info.local_numa_node >= 0)
            {
                target_numa_node = numa_info.local_numa_node;
            }
            dm.initialize(target_numa_node, false); // Tables already printed pre-MPI
        }
        const auto &devices = dm.devices();

        // Helper to convert ComputeBackendType to DeviceType
        auto toDeviceType = [](ComputeBackendType backend) -> DeviceType
        {
            switch (backend)
            {
            case ComputeBackendType::GPU_CUDA:
                return DeviceType::CUDA;
            case ComputeBackendType::GPU_ROCM:
                return DeviceType::ROCm;
            case ComputeBackendType::GPU_VULKAN:
                return DeviceType::Vulkan;
            case ComputeBackendType::GPU_METAL:
                return DeviceType::Metal;
            case ComputeBackendType::CPU:
            default:
                return DeviceType::CPU;
            }
        };

        // For single-rank execution, create a simple inventory
        if (!mpi_ctx || mpi_ctx->world_size() == 1)
        {
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.hostname = "localhost";
            rank_inv.numa_nodes = 1;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;

            // Add CPU by default
            rank_inv.cpu.type = DeviceType::CPU;
            rank_inv.cpu.local_device_id = 0;
            rank_inv.cpu_cores = 1;

            // Query system RAM via POSIX sysconf
            long pages = sysconf(_SC_PHYS_PAGES);
            long page_size = sysconf(_SC_PAGE_SIZE);
            if (pages > 0 && page_size > 0)
                rank_inv.cpu_memory_bytes = static_cast<size_t>(pages) * static_cast<size_t>(page_size);

            // Enumerate actual GPUs from DeviceManager
            for (const auto &dev : devices)
            {
                if (dev.type != ComputeBackendType::CPU)
                {
                    DeviceInfo gpu;
                    gpu.type = toDeviceType(dev.type);
                    gpu.local_device_id = dev.device_id;
                    gpu.memory_bytes = dev.total_memory_bytes;
                    gpu.free_memory_bytes = dev.free_memory_bytes;
                    gpu.name = dev.name;
                    gpu.numa_node = dev.numa_node;
                    gpu.compute_capability_major = dev.compute_capability / 10;
                    gpu.compute_capability_minor = dev.compute_capability % 10;
                    rank_inv.gpus.push_back(gpu);

                    LOG_DEBUG("[gatherClusterInventory] Found GPU: " << dev.name
                                                                     << " (" << dev.total_memory_bytes / (1024 * 1024 * 1024) << " GB)");
                }
            }

            // If explicit tp_devices are configured, filter the enumerated list to match
            if (!explicit_tp_devices.empty())
            {
                LOG_DEBUG("[gatherClusterInventory] Using explicitly configured TP devices (count="
                          << explicit_tp_devices.size() << ")");

                // Map DeviceType → ComputeBackendType for lookup
                auto toBackendType = [](DeviceType dt) -> ComputeBackendType
                {
                    switch (dt)
                    {
                    case DeviceType::CUDA:   return ComputeBackendType::GPU_CUDA;
                    case DeviceType::ROCm:   return ComputeBackendType::GPU_ROCM;
                    case DeviceType::Vulkan: return ComputeBackendType::GPU_VULKAN;
                    case DeviceType::Metal:  return ComputeBackendType::GPU_METAL;
                    default:                 return ComputeBackendType::CPU;
                    }
                };

                rank_inv.gpus.clear();
                for (size_t i = 0; i < explicit_tp_devices.size(); ++i)
                {
                    const auto &addr = explicit_tp_devices[i];
                    DeviceInfo gpu;
                    gpu.type = addr.device_type;
                    gpu.local_device_id = addr.device_ordinal;

                    // Look up actual device info from DeviceManager enumeration
                    int dev_idx = dm.find_device(toBackendType(addr.device_type), addr.device_ordinal);
                    if (dev_idx >= 0)
                    {
                        const auto &dev = devices[static_cast<size_t>(dev_idx)];
                        gpu.memory_bytes = dev.total_memory_bytes;
                        gpu.free_memory_bytes = dev.free_memory_bytes;
                        gpu.name = dev.name;
                        gpu.numa_node = dev.numa_node;
                        gpu.compute_capability_major = dev.compute_capability / 10;
                        gpu.compute_capability_minor = dev.compute_capability % 10;
                        LOG_DEBUG("[gatherClusterInventory] Explicit TP device " << i << ": "
                                  << dev.name << " (ordinal=" << addr.device_ordinal
                                  << ", " << dev.total_memory_bytes / (1024 * 1024) << " MB total, "
                                  << dev.free_memory_bytes / (1024 * 1024) << " MB free)");
                    }
                    else
                    {
                        LOG_WARN("[gatherClusterInventory] Explicit TP device " << i
                                  << " (ordinal=" << addr.device_ordinal
                                  << ") not found in DeviceManager — memory info unavailable");
                    }
                    rank_inv.gpus.push_back(gpu);
                }
            }

            inventory.ranks.push_back(rank_inv);
            inventory.world_size = 1;
            inventory.node_count = 1;
            inventory.total_gpus = static_cast<int>(rank_inv.gpus.size());

            LOG_INFO("[gatherClusterInventory] Discovered " << inventory.total_gpus << " GPU(s)");
            return inventory;
        }

        // Multi-rank execution: build local RankInventory and exchange via MPI_Allgatherv.
        const int world_size = mpi_ctx->world_size();
        const int rank = mpi_ctx->rank();

        MPI_Comm comm = mpi_ctx->communicator();

        RankInventory local_rank_inv;
        local_rank_inv.rank = rank;
        local_rank_inv.node_id = -1;
        local_rank_inv.local_rank = 0;
        local_rank_inv.numa_nodes = 1;

        // Hostname
        char hostname_buf[MPI_MAX_PROCESSOR_NAME] = {0};
        int hostname_len = 0;
        if (MPI_Get_processor_name(hostname_buf, &hostname_len) == MPI_SUCCESS && hostname_len > 0)
        {
            local_rank_inv.hostname.assign(hostname_buf, static_cast<size_t>(hostname_len));
        }
        else
        {
            local_rank_inv.hostname = "unknown";
        }

        // Detect local rank within physical node (shared-memory communicator)
        MPI_Comm local_comm = MPI_COMM_NULL;
        if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm) == MPI_SUCCESS)
        {
            int local_rank = 0;
            int local_world = 1;
            MPI_Comm_rank(local_comm, &local_rank);
            MPI_Comm_size(local_comm, &local_world);
            local_rank_inv.local_rank = local_rank;
            local_rank_inv.numa_nodes = local_world;
            MPI_Comm_free(&local_comm);
        }

        // Populate CPU info
        local_rank_inv.cpu.type = DeviceType::CPU;
        local_rank_inv.cpu.local_device_id = 0;
        local_rank_inv.cpu_cores = 1;

        // Query system RAM via POSIX sysconf
        {
            long pages = sysconf(_SC_PHYS_PAGES);
            long page_size = sysconf(_SC_PAGE_SIZE);
            if (pages > 0 && page_size > 0)
                local_rank_inv.cpu_memory_bytes = static_cast<size_t>(pages) * static_cast<size_t>(page_size);
        }

        // Populate GPU info from DeviceManager
        for (const auto &dev : devices)
        {
            if (dev.type == ComputeBackendType::CPU)
            {
                continue;
            }

            DeviceInfo gpu;
            gpu.type = toDeviceType(dev.type);
            gpu.local_device_id = dev.device_id;
            gpu.memory_bytes = dev.total_memory_bytes;
            gpu.free_memory_bytes = dev.free_memory_bytes;
            gpu.name = dev.name;
            gpu.numa_node = dev.numa_node;
            gpu.compute_capability_major = dev.compute_capability / 10;
            gpu.compute_capability_minor = dev.compute_capability % 10;
            local_rank_inv.gpus.push_back(gpu);
        }

        // Override with explicit tp_devices if configured
        if (!explicit_tp_devices.empty())
        {
            local_rank_inv.gpus.clear();
            for (size_t i = 0; i < explicit_tp_devices.size(); ++i)
            {
                const auto &addr = explicit_tp_devices[i];
                DeviceInfo gpu;
                gpu.type = addr.device_type;
                gpu.local_device_id = static_cast<int>(i);
                gpu.memory_bytes = 0;
                local_rank_inv.gpus.push_back(gpu);
            }
        }

        // Serialize local inventory
        std::vector<uint8_t> local_data = MPITopology::serializeRankInventory(local_rank_inv);
        const int local_size = static_cast<int>(local_data.size());

        // Gather serialized sizes from all ranks
        std::vector<int> all_sizes(world_size, 0);
        MPI_Allgather(
            &local_size, 1, MPI_INT,
            all_sizes.data(), 1, MPI_INT,
            comm);

        // Compute displacements for allgatherv
        std::vector<int> displacements(world_size, 0);
        int total_size = 0;
        for (int r = 0; r < world_size; ++r)
        {
            displacements[r] = total_size;
            total_size += all_sizes[r];
        }

        std::vector<uint8_t> all_data(static_cast<size_t>(total_size));
        MPI_Allgatherv(
            local_data.data(), local_size, MPI_BYTE,
            all_data.data(), all_sizes.data(), displacements.data(), MPI_BYTE,
            comm);

        // Deserialize inventories from all ranks
        inventory.world_size = world_size;
        inventory.ranks.resize(world_size);
        for (int r = 0; r < world_size; ++r)
        {
            const uint8_t *ptr = all_data.data() + displacements[r];
            const size_t size = static_cast<size_t>(all_sizes[r]);
            try
            {
                inventory.ranks[r] = MPITopology::deserializeRankInventory(ptr, size);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[gatherClusterInventory] Failed to deserialize rank " << r << ": " << e.what());
                inventory.ranks[r].rank = r;
                inventory.ranks[r].hostname = "error";
                inventory.ranks[r].node_id = -1;
            }
        }

        // Build deterministic node_id mapping from hostname (single source of truth).
        // Rank 0 parses the hostfile (if any) and computes node IDs, then broadcasts
        // to all ranks. This avoids requiring the hostfile to exist on remote hosts.
        std::vector<std::string> hostnames;
        hostnames.reserve(static_cast<size_t>(world_size));
        for (const auto &rank_inv : inventory.ranks)
        {
            hostnames.push_back(rank_inv.hostname);
        }

        // node_ids: one int per rank. node_count appended as element [world_size].
        std::vector<int> node_id_buf(static_cast<size_t>(world_size) + 1);

        if (rank == 0)
        {
            NodeDetectionResult detection;
            if (!hostfile.empty())
            {
                detection = NodeDetection::fromHostnames(hostnames, hostfile);
            }
            else
            {
                detection = NodeDetection::fromHostnames(hostnames);
            }
            for (int r = 0; r < world_size; ++r)
            {
                node_id_buf[r] = detection.node_ids[r];
            }
            node_id_buf[world_size] = detection.node_count;
        }

        MPI_Bcast(node_id_buf.data(),
                  static_cast<int>(node_id_buf.size()),
                  MPI_INT, 0, comm);

        for (int r = 0; r < world_size; ++r)
        {
            inventory.ranks[r].node_id = node_id_buf[r];
        }

        inventory.node_count = node_id_buf[world_size];
        inventory.buildNodeAggregations();

        LOG_INFO("[gatherClusterInventory] Discovered " << inventory.total_gpus
                                                        << " GPU(s) across " << world_size
                                                        << " ranks on " << inventory.node_count << " node(s)");
        return inventory;
    }

} // namespace llaminar2
