/**
 * @file NodeTopology.h
 * @brief System topology detection for NUMA-aware CPU parallelism
 *
 * Parses Linux sysfs to detect NUMA topology, CPU sockets, and inter-socket
 * links (UPI/QPI/Infinity Fabric) for cross-socket CPU tensor parallelism.
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <optional>

namespace llaminar2
{

    /**
     * Detected NUMA node information
     */
    struct NUMANode
    {
        int node_id;              ///< NUMA node ID (0-indexed)
        std::vector<int> cpu_ids; ///< CPUs in this NUMA node
        size_t memory_bytes;      ///< Total memory on this node
        int socket_id;            ///< Physical socket this node belongs to
    };

    /**
     * Inter-socket link information (UPI/QPI/Infinity Fabric)
     */
    struct InterSocketLink
    {
        int socket_a;         ///< First socket ID
        int socket_b;         ///< Second socket ID
        float bandwidth_gbps; ///< Estimated bandwidth (typically ~50 GB/s for UPI)
        float latency_ns;     ///< Estimated latency (typically ~80-150ns for UPI)
    };

    /**
     * System topology detection for NUMA-aware parallelism.
     *
     * Parses /sys/devices/system/node/ and /sys/devices/system/cpu/
     * to detect:
     * - Number of CPU sockets
     * - NUMA nodes per socket
     * - CPUs per NUMA node
     * - Inter-socket links (UPI/QPI)
     *
     * This class is designed for cross-socket CPU tensor parallelism where
     * MPI ranks on different sockets communicate via UPI links.
     */
    class NodeTopology
    {
    public:
        /**
         * Detect system topology.
         * @return NodeTopology with detected configuration, or default single-socket if detection fails
         */
        static NodeTopology detect();

        /**
         * Create topology from explicit configuration (for testing)
         * @param num_sockets Number of CPU sockets
         * @param numa_nodes_per_socket NUMA nodes per socket
         * @param cores_per_numa_node Cores per NUMA node
         * @return NodeTopology with specified configuration
         */
        static NodeTopology createForTest(int num_sockets, int numa_nodes_per_socket,
                                          int cores_per_numa_node);

        // =========================================================================
        // Accessors
        // =========================================================================

        /** @return Number of physical CPU sockets */
        int numSockets() const { return num_sockets_; }

        /** @return Total number of NUMA nodes */
        int numNUMANodes() const { return static_cast<int>(numa_nodes_.size()); }

        /** @return Total number of CPUs (logical cores) */
        int numCPUs() const { return total_cpus_; }

        /** @return All detected NUMA nodes */
        const std::vector<NUMANode> &numaNodes() const { return numa_nodes_; }

        /** @return All detected inter-socket links */
        const std::vector<InterSocketLink> &interSocketLinks() const { return inter_socket_links_; }

        // =========================================================================
        // Queries
        // =========================================================================

        /**
         * Get NUMA nodes for a specific socket
         * @param socket_id Socket ID to query
         * @return Vector of pointers to NUMA nodes on this socket
         */
        std::vector<const NUMANode *> numaNodesForSocket(int socket_id) const;

        /**
         * Get the socket ID for a given CPU ID
         * @param cpu_id CPU ID to query
         * @return Socket ID, or -1 if CPU not found
         */
        int socketForCPU(int cpu_id) const;

        /**
         * Get the NUMA node ID for a given CPU ID
         * @param cpu_id CPU ID to query
         * @return NUMA node ID, or -1 if CPU not found
         */
        int numaNodeForCPU(int cpu_id) const;

        /**
         * Check if system has multiple sockets with inter-socket links
         * @return true if there are inter-socket links (UPI/QPI)
         */
        bool hasInterSocketLinks() const { return !inter_socket_links_.empty(); }

        /**
         * Get estimated UPI/inter-socket bandwidth in GB/s
         * @return Bandwidth in GB/s, or 0 if single-socket
         */
        float interSocketBandwidthGBps() const;

        // =========================================================================
        // MPI Rank Mapping
        // =========================================================================

        /**
         * Map MPI ranks to sockets (for TP domain creation)
         * @param world_size Total MPI ranks
         * @param local_rank This process's rank
         * @return Socket ID for the given rank
         */
        int suggestSocketForRank(int world_size, int local_rank) const;

        /**
         * Group ranks by socket for creating TP domains
         * @param world_size Total MPI ranks
         * @return Vector of vectors: ranks_per_socket[socket_id] = {rank0, rank1, ...}
         */
        std::vector<std::vector<int>> groupRanksBySocket(int world_size) const;

        /**
         * Get string representation for logging
         * @return Human-readable topology description
         */
        std::string toString() const;

    private:
        NodeTopology() = default;

        // =========================================================================
        // Detection helpers (static)
        // =========================================================================

        /**
         * Detect all NUMA nodes from /sys/devices/system/node/
         * @return Vector of detected NUMA nodes
         */
        static std::vector<NUMANode> detectNUMANodes();

        /**
         * Detect inter-socket links (heuristic-based)
         * @param num_sockets Number of detected sockets
         * @return Vector of inter-socket links
         */
        static std::vector<InterSocketLink> detectInterSocketLinks(int num_sockets);

        /**
         * Determine number of sockets from NUMA node data
         * @param nodes Detected NUMA nodes
         * @return Number of unique sockets
         */
        static int detectNumSockets(const std::vector<NUMANode> &nodes);

        /**
         * Parse CPU list from sysfs (e.g., "0-7,16-23")
         * @param path Path to cpulist file
         * @return Vector of CPU IDs
         */
        static std::vector<int> parseCPUList(const std::string &path);

        /**
         * Parse a CPU list string (e.g., "0-7,16-23")
         * @param cpu_list_str CPU list string
         * @return Vector of CPU IDs
         */
        static std::vector<int> parseCPUListString(const std::string &cpu_list_str);

        /**
         * Parse memory info from sysfs
         * @param path Path to meminfo file
         * @return Memory size in bytes
         */
        static size_t parseMemoryInfo(const std::string &path);

        /**
         * Get socket ID for a CPU from topology
         * @param cpu_id CPU ID to query
         * @return Socket (physical_package_id), or 0 if detection fails
         */
        static int parseSocketId(int cpu_id);

        // =========================================================================
        // Instance data
        // =========================================================================

        int num_sockets_ = 1;                             ///< Number of physical sockets
        int total_cpus_ = 1;                              ///< Total logical CPUs
        std::vector<NUMANode> numa_nodes_;                ///< All NUMA nodes
        std::vector<InterSocketLink> inter_socket_links_; ///< Inter-socket links
        std::unordered_map<int, int> cpu_to_numa_;        ///< cpu_id -> numa_node_id
        std::unordered_map<int, int> cpu_to_socket_;      ///< cpu_id -> socket_id
    };

} // namespace llaminar2
