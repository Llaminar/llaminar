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
 * Rank records describe visibility, not exclusive physical ownership. Node
 * summaries union observed UUIDs and NUMA resources; they are immutable
 * discovery facts, never a second live allocation/admission ledger.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once
#include "backends/CPUExecutionGeometry.h"

#include "../../backends/DeviceType.h"
#include "../../backends/CPUSocketInfo.h"
#include <cstddef>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace llaminar2
{

    /** @brief Physical membership, independent of link speed and transport choice. */
    enum class RankConnectionLocality { SameRank, SameNode, CrossNode };

    /**
     * @brief Immutable endpoint membership projected from one cluster inventory.
     *
     * Rank and node ordinals belong to that inventory's communicator namespace.
     * Node identity is not an OS socket/NUMA index or hostname. Measurements can
     * attach costs to this value but cannot change its physical classification.
     */
    class RankConnectionTopology final
    {
    public:
        /** @return First endpoint's rank in the owning inventory. */
        int sourceRank() const noexcept { return source_rank_; }
        /** @return Second endpoint's rank in the owning inventory. */
        int destinationRank() const noexcept { return destination_rank_; }
        /** @return First endpoint's physical-node membership label. */
        int sourceNode() const noexcept { return source_node_; }
        /** @return Second endpoint's physical-node membership label. */
        int destinationNode() const noexcept { return destination_node_; }
        /** @return Classification derived only from authenticated rank/node identity. */
        RankConnectionLocality locality() const noexcept
        {
            if (source_rank_ == destination_rank_) return RankConnectionLocality::SameRank;
            return source_node_ == destination_node_ ? RankConnectionLocality::SameNode : RankConnectionLocality::CrossNode;
        }
        bool operator==(const RankConnectionTopology &) const = default;

    private:
        friend struct ClusterInventory;
        /** @brief Only a validated inventory lookup may bind endpoint membership. */
        RankConnectionTopology(int source_rank, int destination_rank, int source_node, int destination_node)
            : source_rank_(source_rank), destination_rank_(destination_rank),
              source_node_(source_node), destination_node_(destination_node) {}
        int source_rank_, destination_rank_, source_node_, destination_node_;
    };

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
        /** Observed device-wide last-level cache; required to authenticate streaming memory samples. */
        size_t last_level_cache_bytes = 0;

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
        CPUExecutionGeometry cpu_execution; ///< Exact local CPU policy, never reconstructed on root.
        int rank = -1;        ///< MPI rank ID
        int node_id = -1;     ///< Physical node ID (ranks on same node share this)
        int local_rank = -1;  ///< Rank within node (0..ranks_per_node-1)
        std::string hostname; ///< Node hostname

        // CPU info
        DeviceInfo cpu;              ///< Host CPU capabilities
        int cpu_cores = 0;           ///< Physical cores in the observed CPU locality, not a worker budget.
        int cpu_worker_threads = 0;  ///< Observed OpenMP team budget; zero means no execution observation.
        int cpu_sockets = 0;         ///< CPU sockets
        int numa_nodes = 0;          ///< NUMA nodes
        size_t cpu_memory_bytes = 0; ///< System RAM

        /**
         * @return Exact rank-local execution budget published after startup thread policy.
         * @throws std::invalid_argument when only physical hardware was observed.
         *
         * Physical cores are topology, not permission to silently enlarge a
         * requested team. Workspace admission and measured service must consume
         * this same budget, including explicit one-thread and remote-rank cases.
         */
        int cpuWorkerThreads() const
        {
            if (cpu_worker_threads <= 0)
                throw std::invalid_argument("Rank inventory has no positive CPU worker observation");
            return cpu_worker_threads;
        }

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

        /**
         * @brief Resolve physical membership without discovery or measurements.
         * @throws std::invalid_argument for absent ranks or malformed membership.
         *
         * Hostname aliases, NUMA IDs, rank adjacency and observed link latency
         * never establish locality. Consumers of a selected execution inventory
         * pass execution ranks; discovery observations use discovery ranks.
         * A missing record is an error, not a local or remote default.
         */
        RankConnectionTopology connectionBetweenRanks(int source, int destination) const;

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

        /**
         * @brief Rebuild physical summaries without counting overlapping views.
         *
         * UUIDs identify GPUs within one physical node even when visibility
         * filters change their process-local ordinals. NUMA observations and
         * physical core IDs similarly identify CPU resources. Free-memory
         * samples may differ between observers; immutable physical capacities
         * may not. Zero-capacity collective-only records carry membership, not
         * an admissible physical resource.
         *
         * All work is staged locally and published together. A rejected record
         * leaves the previous summary intact, and repeated calls are idempotent.
         * @throws std::invalid_argument for ambiguous or conflicting resources.
         * @throws std::overflow_error if a summary cannot represent its inputs.
         */
        void buildNodeAggregations();

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
