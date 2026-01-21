/**
 * @file NodeTopology.cpp
 * @brief System topology detection implementation
 *
 * Parses Linux sysfs to detect NUMA topology, CPU sockets, and inter-socket
 * links for cross-socket CPU tensor parallelism.
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#include "NodeTopology.h"
#include "Logger.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <set>
#include <thread>

namespace llaminar2
{

    namespace
    {
        // =========================================================================
        // Constants for heuristic inter-socket bandwidth estimation
        // =========================================================================

        // Intel UPI (Ultra Path Interconnect) bandwidth estimates
        constexpr float UPI_BANDWIDTH_GBPS = 50.0f; // ~50 GB/s bidirectional per link
        constexpr float UPI_LATENCY_NS = 100.0f;    // ~80-150ns typical

        // AMD Infinity Fabric bandwidth estimates (typically higher than UPI)
        constexpr float INFINITY_FABRIC_BANDWIDTH_GBPS = 100.0f; // ~100 GB/s per link
        constexpr float INFINITY_FABRIC_LATENCY_NS = 80.0f;      // ~60-100ns typical

        // Sysfs paths
        constexpr const char *SYSFS_NODE_PATH = "/sys/devices/system/node";
        constexpr const char *SYSFS_CPU_PATH = "/sys/devices/system/cpu";

        /**
         * Read entire file contents as string
         */
        std::string readFile(const std::string &path)
        {
            std::ifstream file(path);
            if (!file.is_open())
            {
                return "";
            }
            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }

        /**
         * Trim whitespace from string
         */
        std::string trim(const std::string &str)
        {
            size_t start = str.find_first_not_of(" \t\n\r");
            if (start == std::string::npos)
                return "";
            size_t end = str.find_last_not_of(" \t\n\r");
            return str.substr(start, end - start + 1);
        }

        /**
         * Check if CPU vendor is AMD (for Infinity Fabric heuristics)
         */
        bool isAMDProcessor()
        {
            std::ifstream cpuinfo("/proc/cpuinfo");
            if (!cpuinfo.is_open())
            {
                return false;
            }
            std::string line;
            while (std::getline(cpuinfo, line))
            {
                if (line.find("vendor_id") != std::string::npos &&
                    line.find("AMD") != std::string::npos)
                {
                    return true;
                }
            }
            return false;
        }
    }

    // =============================================================================
    // NodeTopology::detect()
    // =============================================================================

    NodeTopology NodeTopology::detect()
    {
        NodeTopology topology;

        // Detect NUMA nodes
        topology.numa_nodes_ = detectNUMANodes();

        // If detection failed, create a minimal single-socket topology
        if (topology.numa_nodes_.empty())
        {
            LOG_WARN("[NodeTopology] NUMA detection failed, creating single-socket fallback");

            // Get CPU count from hardware_concurrency or fallback
            int num_cpus = static_cast<int>(std::thread::hardware_concurrency());
            if (num_cpus <= 0)
            {
                num_cpus = 1;
            }

            // Create a single NUMA node with all CPUs
            NUMANode node;
            node.node_id = 0;
            node.socket_id = 0;
            node.memory_bytes = 0; // Unknown
            for (int i = 0; i < num_cpus; ++i)
            {
                node.cpu_ids.push_back(i);
            }
            topology.numa_nodes_.push_back(node);
            topology.num_sockets_ = 1;
            topology.total_cpus_ = num_cpus;

            // Build lookup maps
            for (int i = 0; i < num_cpus; ++i)
            {
                topology.cpu_to_numa_[i] = 0;
                topology.cpu_to_socket_[i] = 0;
            }

            LOG_INFO("[NodeTopology] Fallback: 1 socket, " << num_cpus << " CPUs");
            return topology;
        }

        // Determine number of sockets from detected nodes
        topology.num_sockets_ = detectNumSockets(topology.numa_nodes_);

        // Count total CPUs and build lookup maps
        topology.total_cpus_ = 0;
        for (const auto &node : topology.numa_nodes_)
        {
            topology.total_cpus_ += static_cast<int>(node.cpu_ids.size());
            for (int cpu_id : node.cpu_ids)
            {
                topology.cpu_to_numa_[cpu_id] = node.node_id;
                topology.cpu_to_socket_[cpu_id] = node.socket_id;
            }
        }

        // Detect inter-socket links if multi-socket
        if (topology.num_sockets_ > 1)
        {
            topology.inter_socket_links_ = detectInterSocketLinks(topology.num_sockets_);
        }

        LOG_INFO("[NodeTopology] Detected: " << topology.num_sockets_ << " socket(s), "
                                             << topology.numa_nodes_.size() << " NUMA node(s), "
                                             << topology.total_cpus_ << " CPU(s)");

        if (!topology.inter_socket_links_.empty())
        {
            LOG_INFO("[NodeTopology] Inter-socket links: " << topology.inter_socket_links_.size()
                                                           << " (bandwidth ~" << topology.interSocketBandwidthGBps() << " GB/s)");
        }

        return topology;
    }

    // =============================================================================
    // NodeTopology::createForTest()
    // =============================================================================

    NodeTopology NodeTopology::createForTest(int num_sockets, int numa_nodes_per_socket,
                                             int cores_per_numa_node)
    {
        NodeTopology topology;

        topology.num_sockets_ = num_sockets;
        topology.total_cpus_ = num_sockets * numa_nodes_per_socket * cores_per_numa_node;

        int cpu_id = 0;
        int node_id = 0;

        for (int socket = 0; socket < num_sockets; ++socket)
        {
            for (int numa = 0; numa < numa_nodes_per_socket; ++numa)
            {
                NUMANode node;
                node.node_id = node_id;
                node.socket_id = socket;
                node.memory_bytes = 64ULL * 1024 * 1024 * 1024; // 64 GB per node (test value)

                for (int core = 0; core < cores_per_numa_node; ++core)
                {
                    node.cpu_ids.push_back(cpu_id);
                    topology.cpu_to_numa_[cpu_id] = node_id;
                    topology.cpu_to_socket_[cpu_id] = socket;
                    ++cpu_id;
                }

                topology.numa_nodes_.push_back(node);
                ++node_id;
            }
        }

        // Create inter-socket links for multi-socket configs
        if (num_sockets > 1)
        {
            topology.inter_socket_links_ = detectInterSocketLinks(num_sockets);
        }

        return topology;
    }

    // =============================================================================
    // Query Methods
    // =============================================================================

    std::vector<const NUMANode *> NodeTopology::numaNodesForSocket(int socket_id) const
    {
        std::vector<const NUMANode *> result;
        for (const auto &node : numa_nodes_)
        {
            if (node.socket_id == socket_id)
            {
                result.push_back(&node);
            }
        }
        return result;
    }

    int NodeTopology::socketForCPU(int cpu_id) const
    {
        auto it = cpu_to_socket_.find(cpu_id);
        return (it != cpu_to_socket_.end()) ? it->second : -1;
    }

    int NodeTopology::numaNodeForCPU(int cpu_id) const
    {
        auto it = cpu_to_numa_.find(cpu_id);
        return (it != cpu_to_numa_.end()) ? it->second : -1;
    }

    float NodeTopology::interSocketBandwidthGBps() const
    {
        if (inter_socket_links_.empty())
        {
            return 0.0f;
        }
        // Return bandwidth of first link (they're typically identical)
        return inter_socket_links_[0].bandwidth_gbps;
    }

    // =============================================================================
    // MPI Rank Mapping
    // =============================================================================

    int NodeTopology::suggestSocketForRank(int world_size, int local_rank) const
    {
        if (num_sockets_ <= 1 || world_size <= 1)
        {
            return 0;
        }

        // Distribute ranks across sockets round-robin
        // For world_size=4, num_sockets=2: rank0->socket0, rank1->socket1, rank2->socket0, rank3->socket1
        return local_rank % num_sockets_;
    }

    std::vector<std::vector<int>> NodeTopology::groupRanksBySocket(int world_size) const
    {
        std::vector<std::vector<int>> groups(num_sockets_);

        for (int rank = 0; rank < world_size; ++rank)
        {
            int socket = suggestSocketForRank(world_size, rank);
            groups[socket].push_back(rank);
        }

        return groups;
    }

    // =============================================================================
    // toString()
    // =============================================================================

    std::string NodeTopology::toString() const
    {
        std::ostringstream ss;
        ss << "NodeTopology {\n";
        ss << "  sockets: " << num_sockets_ << "\n";
        ss << "  numa_nodes: " << numa_nodes_.size() << "\n";
        ss << "  total_cpus: " << total_cpus_ << "\n";

        for (const auto &node : numa_nodes_)
        {
            ss << "  NUMA" << node.node_id << " (socket " << node.socket_id << "): ";
            if (node.cpu_ids.size() <= 8)
            {
                ss << "CPUs [";
                for (size_t i = 0; i < node.cpu_ids.size(); ++i)
                {
                    if (i > 0)
                        ss << ",";
                    ss << node.cpu_ids[i];
                }
                ss << "]";
            }
            else
            {
                ss << node.cpu_ids.size() << " CPUs [" << node.cpu_ids.front() << "-" << node.cpu_ids.back() << "]";
            }
            if (node.memory_bytes > 0)
            {
                ss << ", " << (node.memory_bytes / (1024ULL * 1024 * 1024)) << " GB";
            }
            ss << "\n";
        }

        if (!inter_socket_links_.empty())
        {
            ss << "  inter_socket_links:\n";
            for (const auto &link : inter_socket_links_)
            {
                ss << "    socket" << link.socket_a << " <-> socket" << link.socket_b
                   << ": " << link.bandwidth_gbps << " GB/s, " << link.latency_ns << " ns\n";
            }
        }

        ss << "}";
        return ss.str();
    }

    // =============================================================================
    // Detection Helpers
    // =============================================================================

    std::vector<NUMANode> NodeTopology::detectNUMANodes()
    {
        std::vector<NUMANode> nodes;

        // Check if sysfs node path exists
        if (!std::filesystem::exists(SYSFS_NODE_PATH))
        {
            LOG_DEBUG("[NodeTopology] " << SYSFS_NODE_PATH << " not found");
            return nodes;
        }

        // Iterate over node directories
        try
        {
            for (const auto &entry : std::filesystem::directory_iterator(SYSFS_NODE_PATH))
            {
                const std::string &name = entry.path().filename().string();

                // Look for directories named "node0", "node1", etc.
                if (name.substr(0, 4) != "node")
                {
                    continue;
                }

                int node_id = -1;
                try
                {
                    node_id = std::stoi(name.substr(4));
                }
                catch (...)
                {
                    continue;
                }

                NUMANode node;
                node.node_id = node_id;

                // Parse CPU list
                std::string cpulist_path = entry.path().string() + "/cpulist";
                node.cpu_ids = parseCPUList(cpulist_path);

                if (node.cpu_ids.empty())
                {
                    LOG_DEBUG("[NodeTopology] No CPUs found for node " << node_id);
                    continue;
                }

                // Determine socket ID from first CPU's physical_package_id
                node.socket_id = parseSocketId(node.cpu_ids[0]);

                // Parse memory info
                std::string meminfo_path = entry.path().string() + "/meminfo";
                node.memory_bytes = parseMemoryInfo(meminfo_path);

                nodes.push_back(node);
                LOG_DEBUG("[NodeTopology] Found NUMA node " << node_id << " on socket " << node.socket_id
                                                            << " with " << node.cpu_ids.size() << " CPUs");
            }
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            LOG_WARN("[NodeTopology] Error iterating " << SYSFS_NODE_PATH << ": " << e.what());
        }

        // Sort by node_id
        std::sort(nodes.begin(), nodes.end(),
                  [](const NUMANode &a, const NUMANode &b)
                  { return a.node_id < b.node_id; });

        return nodes;
    }

    std::vector<InterSocketLink> NodeTopology::detectInterSocketLinks(int num_sockets)
    {
        std::vector<InterSocketLink> links;

        if (num_sockets <= 1)
        {
            return links;
        }

        // Use heuristics based on processor vendor
        bool is_amd = isAMDProcessor();
        float bandwidth = is_amd ? INFINITY_FABRIC_BANDWIDTH_GBPS : UPI_BANDWIDTH_GBPS;
        float latency = is_amd ? INFINITY_FABRIC_LATENCY_NS : UPI_LATENCY_NS;

        // Create links between all socket pairs
        // For 2 sockets: socket0 <-> socket1
        // For 4 sockets: typically mesh topology (socket0 <-> socket1, socket0 <-> socket2, etc.)
        for (int i = 0; i < num_sockets; ++i)
        {
            for (int j = i + 1; j < num_sockets; ++j)
            {
                InterSocketLink link;
                link.socket_a = i;
                link.socket_b = j;
                link.bandwidth_gbps = bandwidth;
                link.latency_ns = latency;
                links.push_back(link);
            }
        }

        LOG_DEBUG("[NodeTopology] Created " << links.size() << " inter-socket links ("
                                            << (is_amd ? "AMD Infinity Fabric" : "Intel UPI") << " heuristic)");

        return links;
    }

    int NodeTopology::detectNumSockets(const std::vector<NUMANode> &nodes)
    {
        if (nodes.empty())
        {
            return 1;
        }

        std::set<int> sockets;
        for (const auto &node : nodes)
        {
            sockets.insert(node.socket_id);
        }

        return static_cast<int>(sockets.size());
    }

    std::vector<int> NodeTopology::parseCPUList(const std::string &path)
    {
        std::string content = readFile(path);
        if (content.empty())
        {
            return {};
        }
        return parseCPUListString(trim(content));
    }

    std::vector<int> NodeTopology::parseCPUListString(const std::string &cpu_list_str)
    {
        std::vector<int> cpus;

        if (cpu_list_str.empty())
        {
            return cpus;
        }

        // Parse comma-separated ranges like "0-7,16-23,32"
        std::istringstream stream(cpu_list_str);
        std::string token;

        while (std::getline(stream, token, ','))
        {
            token = trim(token);
            if (token.empty())
            {
                continue;
            }

            size_t dash = token.find('-');
            if (dash != std::string::npos)
            {
                // Range: "0-7"
                try
                {
                    int start = std::stoi(token.substr(0, dash));
                    int end = std::stoi(token.substr(dash + 1));
                    for (int i = start; i <= end; ++i)
                    {
                        cpus.push_back(i);
                    }
                }
                catch (...)
                {
                    LOG_WARN("[NodeTopology] Failed to parse CPU range: " << token);
                }
            }
            else
            {
                // Single CPU: "32"
                try
                {
                    cpus.push_back(std::stoi(token));
                }
                catch (...)
                {
                    LOG_WARN("[NodeTopology] Failed to parse CPU ID: " << token);
                }
            }
        }

        return cpus;
    }

    size_t NodeTopology::parseMemoryInfo(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            return 0;
        }

        std::string line;
        while (std::getline(file, line))
        {
            // Look for "MemTotal:" line
            if (line.find("MemTotal:") != std::string::npos)
            {
                // Format: "Node 0 MemTotal:       131072000 kB"
                // Find the number
                std::istringstream iss(line);
                std::string word;
                size_t mem_kb = 0;
                while (iss >> word)
                {
                    try
                    {
                        mem_kb = std::stoull(word);
                        // If we parsed a number, check if next word is "kB"
                        std::string unit;
                        if (iss >> unit && (unit == "kB" || unit == "KB"))
                        {
                            return mem_kb * 1024; // Convert to bytes
                        }
                    }
                    catch (...)
                    {
                        // Not a number, continue
                    }
                }
            }
        }

        return 0;
    }

    int NodeTopology::parseSocketId(int cpu_id)
    {
        std::ostringstream path;
        path << SYSFS_CPU_PATH << "/cpu" << cpu_id << "/topology/physical_package_id";

        std::string content = readFile(path.str());
        if (content.empty())
        {
            LOG_DEBUG("[NodeTopology] Could not read physical_package_id for CPU " << cpu_id);
            return 0; // Assume socket 0
        }

        try
        {
            return std::stoi(trim(content));
        }
        catch (...)
        {
            LOG_WARN("[NodeTopology] Failed to parse physical_package_id for CPU " << cpu_id);
            return 0;
        }
    }

} // namespace llaminar2
