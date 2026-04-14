/**
 * @file DeviceInventory.h
 * @brief Device inventory structures for hierarchical placement
 *
 * This file defines the data structures for representing device capabilities
 * across an entire MPI cluster. The hierarchy is:
 *
 *   ClusterInventory (all nodes)
 *     └── NodeInventory (per physical machine)
 *           └── RankInventory (per MPI rank)
 *                 └── DeviceInfo (per GPU/accelerator)
 *
 * Workflow:
 * 1. Each MPI rank discovers its local devices at startup
 * 2. All ranks exchange their inventories via MPI_Allgather
 * 3. All ranks now have identical ClusterInventory
 * 4. PlacementStrategy uses ClusterInventory to compute placement
 * 5. Each rank extracts its portion of the placement plan
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../backends/DeviceType.h"
#include "../../backends/CPUSocketInfo.h"
#include <cstddef>
#include <string>
#include <sstream>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Convert DeviceType to string
     */
    inline const char *deviceTypeToString(DeviceType type)
    {
        switch (type)
        {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::CUDA:
            return "CUDA";
        case DeviceType::ROCm:
            return "ROCm";
        case DeviceType::Vulkan:
            return "Vulkan";
        case DeviceType::Metal:
            return "Metal";
        }
        return "Unknown";
    }

    /**
     * @brief Information about a single device (GPU or accelerator)
     *
     * This represents one compute device available to an MPI rank.
     * For CPU, there's typically one "device" per rank representing
     * the host processor.
     */
    struct DeviceInfo
    {
        DeviceType type = DeviceType::CPU;
        int local_device_id = 0;      ///< Device index within this rank (0, 1, 2, ...)
        size_t memory_bytes = 0;      ///< Total device memory (VRAM for GPU, RAM for CPU)
        size_t free_memory_bytes = 0; ///< Available memory at discovery time

        // Compute capability
        int compute_units = 0;            ///< SM count (GPU) or core count (CPU)
        int compute_capability_major = 0; ///< CUDA compute capability major
        int compute_capability_minor = 0; ///< CUDA compute capability minor
        float tflops_fp16 = 0.0f;         ///< Estimated FP16 TFLOPS
        float tflops_int8 = 0.0f;         ///< Estimated INT8 TOPS

        // Memory bandwidth
        float memory_bandwidth_gbps = 0.0f; ///< Memory bandwidth in GB/s

        // Identification
        std::string name; ///< Device name (e.g., "NVIDIA A100-SXM4-80GB")
        std::string uuid; ///< Unique device identifier (for multi-node dedup)

        // Connectivity (for peer-to-peer)
        bool supports_p2p = false; ///< Can do GPU-direct P2P
        int pcie_bus_id = 0;       ///< PCIe bus ID (for locality)
        int numa_node = -1;        ///< Associated NUMA node (-1 if unknown)

        // PCIe link info (effective = bottleneck-aware after upstream walk)
        int pcie_gen = 0;              ///< Effective PCIe generation (3/4/5/6)
        int pcie_width = 0;            ///< Effective link width (x8, x16)
        double pcie_speed_gts = 0.0;   ///< Effective speed in GT/s
        int pcie_max_width = 0;        ///< Max capable width (endpoint)
        double pcie_max_speed_gts = 0; ///< Max capable speed in GT/s (endpoint)
        bool pcie_degraded = false;    ///< True if running below max capability
        std::string pcie_bottleneck_bdf; ///< BDF of upstream bridge causing bottleneck (empty if none)

        /// Check if this is a GPU (any type)
        bool isGPU() const
        {
            return type == DeviceType::CUDA ||
                   type == DeviceType::ROCm ||
                   type == DeviceType::Vulkan ||
                   type == DeviceType::Metal;
        }

        /// Get relative compute weight for load balancing
        float computeWeight() const
        {
            if (tflops_int8 > 0)
                return tflops_int8;
            if (tflops_fp16 > 0)
                return tflops_fp16;
            return static_cast<float>(compute_units) * 0.1f;
        }
    };

    /**
     * @brief Device inventory for a single MPI rank
     *
     * Contains all devices accessible to one MPI process.
     * Typically includes one CPU device and zero or more GPUs.
     */
    struct RankInventory
    {
        int rank = -1;        ///< MPI rank ID
        int node_id = -1;     ///< Physical node ID (ranks on same node share this)
        int local_rank = -1;  ///< Rank within node (0..ranks_per_node-1)
        std::string hostname; ///< Node hostname

        // CPU info
        DeviceInfo cpu;              ///< Host CPU capabilities
        int cpu_cores = 0;           ///< Total CPU cores
        int cpu_sockets = 0;         ///< CPU sockets
        int numa_nodes = 0;          ///< NUMA nodes
        size_t cpu_memory_bytes = 0; ///< System RAM

        // GPU/accelerator info
        std::vector<DeviceInfo> gpus; ///< GPU devices accessible to this rank

        // Per-socket CPU detail (detected from sysfs)
        std::vector<CPUSocketInfo> cpu_socket_info; ///< Per-socket CPU topology

        // P2P access matrices (per GPU backend, populated if >=2 devices)
        // Stored as flat bool vectors: p2p_cuda[i * cuda_count + j] = can_access
        std::vector<bool> p2p_cuda; ///< CUDA P2P matrix (cuda_count x cuda_count)
        std::vector<bool> p2p_rocm; ///< ROCm P2P matrix (rocm_count x rocm_count)
        int p2p_cuda_count = 0;     ///< Number of CUDA devices in P2P matrix
        int p2p_rocm_count = 0;     ///< Number of ROCm devices in P2P matrix

        /// Total GPU count for this rank
        int gpuCount() const { return static_cast<int>(gpus.size()); }

        /// Total GPU memory for this rank
        size_t totalGPUMemory() const
        {
            size_t total = 0;
            for (const auto &gpu : gpus)
            {
                total += gpu.memory_bytes;
            }
            return total;
        }

        /// Check if rank has any GPUs
        bool hasGPU() const { return !gpus.empty(); }

        /// Get total compute weight for this rank
        float totalComputeWeight() const
        {
            float weight = cpu.computeWeight();
            for (const auto &gpu : gpus)
            {
                weight += gpu.computeWeight();
            }
            return weight;
        }
    };

    /**
     * @brief Device inventory for a physical node (machine)
     *
     * Aggregates the inventories of all MPI ranks on the same node.
     * Useful for intra-node placement optimization.
     */
    struct NodeInventory
    {
        int node_id = -1;       ///< Node ID (0..node_count-1)
        std::string hostname;   ///< Node hostname
        std::vector<int> ranks; ///< MPI ranks on this node

        // Aggregated hardware info
        int total_gpus = 0;          ///< Total GPUs on node
        size_t total_gpu_memory = 0; ///< Total GPU memory on node
        size_t total_cpu_memory = 0; ///< Total CPU memory on node
        int total_cpu_cores = 0;     ///< Total CPU cores on node

        /// Get ranks per node
        int ranksPerNode() const { return static_cast<int>(ranks.size()); }
    };

    /**
     * @brief Complete device inventory for the entire MPI cluster
     *
     * This is the top-level structure containing device information
     * for all ranks across all nodes. Built by exchanging RankInventory
     * via MPI_Allgather.
     *
     * All ranks have identical copies of this after exchange.
     */
    struct ClusterInventory
    {
        int world_size = 0; ///< Total MPI ranks
        int node_count = 0; ///< Physical node count

        std::vector<RankInventory> ranks; ///< Per-rank inventories
        std::vector<NodeInventory> nodes; ///< Per-node aggregations

        // Cluster-wide totals
        int total_gpus = 0;          ///< Total GPUs in cluster
        size_t total_gpu_memory = 0; ///< Total GPU memory in cluster
        size_t total_cpu_memory = 0; ///< Total CPU memory in cluster

        /// Check if any rank has GPU
        bool hasAnyGPU() const { return total_gpus > 0; }

        /// Get rank inventory by rank ID
        const RankInventory &getRank(int rank) const
        {
            static RankInventory empty;
            if (rank < 0 || rank >= static_cast<int>(ranks.size()))
            {
                return empty;
            }
            return ranks[rank];
        }

        /// Get node inventory by node ID
        const NodeInventory &getNode(int node_id) const
        {
            static NodeInventory empty;
            if (node_id < 0 || node_id >= static_cast<int>(nodes.size()))
            {
                return empty;
            }
            return nodes[node_id];
        }

        /// Build node aggregations from rank inventories
        void buildNodeAggregations()
        {
            // Find unique node IDs
            int max_node = 0;
            for (const auto &r : ranks)
            {
                if (r.node_id > max_node)
                    max_node = r.node_id;
            }
            node_count = max_node + 1;
            nodes.resize(node_count);

            // Initialize nodes
            for (int n = 0; n < node_count; ++n)
            {
                nodes[n].node_id = n;
            }

            // Aggregate per node
            for (const auto &r : ranks)
            {
                if (r.node_id < 0 || r.node_id >= node_count)
                    continue;

                auto &node = nodes[r.node_id];
                node.hostname = r.hostname;
                node.ranks.push_back(r.rank);
                node.total_gpus += r.gpuCount();
                node.total_gpu_memory += r.totalGPUMemory();
                node.total_cpu_memory += r.cpu_memory_bytes;
                node.total_cpu_cores += r.cpu_cores;
            }

            // Update cluster totals
            total_gpus = 0;
            total_gpu_memory = 0;
            total_cpu_memory = 0;
            for (const auto &node : nodes)
            {
                total_gpus += node.total_gpus;
                total_gpu_memory += node.total_gpu_memory;
                total_cpu_memory += node.total_cpu_memory;
            }
        }

        /// Generate human-readable summary
        std::string toString() const;
    };

    /**
     * @brief Global device identifier (rank + local device)
     *
     * Uniquely identifies a device across the entire cluster.
     * Used in placement plans to specify exactly where work goes.
     */
    struct GlobalDeviceId
    {
        int rank = 0;                      ///< MPI rank owning the device
        DeviceType type = DeviceType::CPU; ///< Device type
        int local_device_id = 0;           ///< Device index within rank (0 for CPU)

        /// Check if this is a CPU device
        bool isCPU() const { return type == DeviceType::CPU; }

        /// Check if this is a GPU device
        bool isGPU() const
        {
            return type == DeviceType::CUDA ||
                   type == DeviceType::ROCm ||
                   type == DeviceType::Vulkan ||
                   type == DeviceType::Metal;
        }

        /// Create CPU device ID for a rank
        static GlobalDeviceId cpu(int rank)
        {
            return {rank, DeviceType::CPU, 0};
        }

        /// Create GPU device ID for a rank
        static GlobalDeviceId gpu(int rank, int local_id, DeviceType type = DeviceType::CUDA)
        {
            return {rank, type, local_id};
        }

        /// Equality comparison
        bool operator==(const GlobalDeviceId &other) const
        {
            return rank == other.rank &&
                   type == other.type &&
                   local_device_id == other.local_device_id;
        }

        bool operator!=(const GlobalDeviceId &other) const
        {
            return !(*this == other);
        }

        /// Less-than comparison for use in maps/sets
        bool operator<(const GlobalDeviceId &other) const
        {
            if (rank != other.rank)
                return rank < other.rank;
            if (type != other.type)
                return type < other.type;
            return local_device_id < other.local_device_id;
        }

        /// String representation: "rank{N}_{type}{ordinal}"
        /// Example: "rank0_cuda0", "rank1_rocm0", "rank0_cpu"
        std::string toString() const
        {
            std::ostringstream oss;
            oss << "rank" << rank << "_";
            switch (type)
            {
            case DeviceType::CUDA:
                oss << "cuda" << local_device_id;
                break;
            case DeviceType::ROCm:
                oss << "rocm" << local_device_id;
                break;
            case DeviceType::Vulkan:
                oss << "vulkan" << local_device_id;
                break;
            case DeviceType::Metal:
                oss << "metal" << local_device_id;
                break;
            case DeviceType::CPU:
            default:
                oss << "cpu";
                break;
            }
            return oss.str();
        }
    };

} // namespace llaminar2
