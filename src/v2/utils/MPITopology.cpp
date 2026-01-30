/**
 * @file MPITopology.cpp
 * @brief Implementation of MPI topology abstraction
 *
 * Design principles:
 * - ALL ranks (including rank 0) participate in compute by default
 * - Equal work division initially; future support for weighted distribution
 * - Integrates with existing SliceMetadata from tensors/TensorSlice.h
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include "MPITopology.h"
#include "NUMATopology.h"
#include "Logger.h"
#include "../tensors/TensorSlice.h"
#include "../execution/mpi_orchestration/PlacementStrategy.h"

#include <sstream>
#include <algorithm>
#include <numeric>
#include <unistd.h> // gethostname
#include <cstring>  // memcpy

namespace llaminar2
{

    // =========================================================================
    // Serialization Helpers
    // =========================================================================

    namespace
    {
        // Helper to write a value to a byte buffer
        template <typename T>
        void writeValue(std::vector<uint8_t> &buffer, T value)
        {
            const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&value);
            buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
        }

        // Helper to write a string to a byte buffer (length-prefixed)
        void writeString(std::vector<uint8_t> &buffer, const std::string &str)
        {
            uint32_t len = static_cast<uint32_t>(str.size());
            writeValue(buffer, len);
            buffer.insert(buffer.end(), str.begin(), str.end());
        }

        // Helper to read a value from a byte buffer
        template <typename T>
        T readValue(const uint8_t *&ptr, const uint8_t *end)
        {
            if (ptr + sizeof(T) > end)
            {
                throw std::runtime_error("Buffer underflow in RankInventory deserialization");
            }
            T value;
            std::memcpy(&value, ptr, sizeof(T));
            ptr += sizeof(T);
            return value;
        }

        // Helper to read a string from a byte buffer (length-prefixed)
        std::string readString(const uint8_t *&ptr, const uint8_t *end)
        {
            uint32_t len = readValue<uint32_t>(ptr, end);
            if (ptr + len > end)
            {
                throw std::runtime_error("Buffer underflow reading string in RankInventory deserialization");
            }
            std::string str(reinterpret_cast<const char *>(ptr), len);
            ptr += len;
            return str;
        }

        // Serialize a single DeviceInfo
        void serializeDeviceInfo(std::vector<uint8_t> &buffer, const DeviceInfo &info)
        {
            writeValue(buffer, static_cast<int32_t>(info.type));
            writeValue(buffer, static_cast<int32_t>(info.local_device_id));
            writeValue(buffer, static_cast<uint64_t>(info.memory_bytes));
            writeValue(buffer, static_cast<uint64_t>(info.free_memory_bytes));
            writeValue(buffer, static_cast<int32_t>(info.compute_units));
            writeValue(buffer, static_cast<int32_t>(info.compute_capability_major));
            writeValue(buffer, static_cast<int32_t>(info.compute_capability_minor));
            writeValue(buffer, info.tflops_fp16);
            writeValue(buffer, info.tflops_int8);
            writeValue(buffer, info.memory_bandwidth_gbps);
            writeString(buffer, info.name);
            writeString(buffer, info.uuid);
            writeValue(buffer, static_cast<uint8_t>(info.supports_p2p ? 1 : 0));
            writeValue(buffer, static_cast<int32_t>(info.pcie_bus_id));
            writeValue(buffer, static_cast<int32_t>(info.numa_node));
        }

        // Deserialize a single DeviceInfo
        DeviceInfo deserializeDeviceInfo(const uint8_t *&ptr, const uint8_t *end)
        {
            DeviceInfo info;
            info.type = static_cast<DeviceType>(readValue<int32_t>(ptr, end));
            info.local_device_id = readValue<int32_t>(ptr, end);
            info.memory_bytes = readValue<uint64_t>(ptr, end);
            info.free_memory_bytes = readValue<uint64_t>(ptr, end);
            info.compute_units = readValue<int32_t>(ptr, end);
            info.compute_capability_major = readValue<int32_t>(ptr, end);
            info.compute_capability_minor = readValue<int32_t>(ptr, end);
            info.tflops_fp16 = readValue<float>(ptr, end);
            info.tflops_int8 = readValue<float>(ptr, end);
            info.memory_bandwidth_gbps = readValue<float>(ptr, end);
            info.name = readString(ptr, end);
            info.uuid = readString(ptr, end);
            info.supports_p2p = (readValue<uint8_t>(ptr, end) != 0);
            info.pcie_bus_id = readValue<int32_t>(ptr, end);
            info.numa_node = readValue<int32_t>(ptr, end);
            return info;
        }
    } // anonymous namespace

    // =========================================================================
    // RankInventory Serialization
    // =========================================================================

    std::vector<uint8_t> MPITopology::serializeRankInventory(const RankInventory &inventory)
    {
        std::vector<uint8_t> buffer;
        buffer.reserve(512); // Pre-allocate reasonable size

        // Write rank identification
        writeValue(buffer, static_cast<int32_t>(inventory.rank));
        writeValue(buffer, static_cast<int32_t>(inventory.node_id));
        writeValue(buffer, static_cast<int32_t>(inventory.local_rank));
        writeString(buffer, inventory.hostname);

        // Write CPU info
        writeValue(buffer, static_cast<int32_t>(inventory.cpu_cores));
        writeValue(buffer, static_cast<int32_t>(inventory.cpu_sockets));
        writeValue(buffer, static_cast<int32_t>(inventory.numa_nodes));
        writeValue(buffer, static_cast<uint64_t>(inventory.cpu_memory_bytes));

        // Write CPU device info
        serializeDeviceInfo(buffer, inventory.cpu);

        // Write GPU count and GPU device infos
        writeValue(buffer, static_cast<int32_t>(inventory.gpus.size()));
        for (const auto &gpu : inventory.gpus)
        {
            serializeDeviceInfo(buffer, gpu);
        }

        return buffer;
    }

    RankInventory MPITopology::deserializeRankInventory(const uint8_t *data, size_t size)
    {
        const uint8_t *ptr = data;
        const uint8_t *end = data + size;

        RankInventory inventory;

        // Read rank identification
        inventory.rank = readValue<int32_t>(ptr, end);
        inventory.node_id = readValue<int32_t>(ptr, end);
        inventory.local_rank = readValue<int32_t>(ptr, end);
        inventory.hostname = readString(ptr, end);

        // Read CPU info
        inventory.cpu_cores = readValue<int32_t>(ptr, end);
        inventory.cpu_sockets = readValue<int32_t>(ptr, end);
        inventory.numa_nodes = readValue<int32_t>(ptr, end);
        inventory.cpu_memory_bytes = readValue<uint64_t>(ptr, end);

        // Read CPU device info
        inventory.cpu = deserializeDeviceInfo(ptr, end);

        // Read GPU count and GPU device infos
        int32_t gpu_count = readValue<int32_t>(ptr, end);
        inventory.gpus.reserve(gpu_count);
        for (int32_t i = 0; i < gpu_count; ++i)
        {
            inventory.gpus.push_back(deserializeDeviceInfo(ptr, end));
        }

        return inventory;
    }

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    MPITopology::MPITopology(MPI_Comm comm)
        : world_comm_(comm),
          intra_node_comm_(MPI_COMM_NULL),
          inter_node_comm_(MPI_COMM_NULL),
          owns_comms_(true)
    {
        // Get basic rank info
        MPI_Comm_rank(world_comm_, &rank_);
        MPI_Comm_size(world_comm_, &world_size_);

        // Auto-detect everything
        detect_topology();
        setup_communicators();
        detect_numa_placement();
        detect_device_capabilities();

        // Exchange capabilities with all ranks
        exchangeCapabilities();

        LOG_DEBUG("[MPITopology] Initialized: rank=" << rank_
                                                     << "/" << world_size_
                                                     << " node=" << placement_.node_id
                                                     << "/" << node_count_
                                                     << " local_rank=" << placement_.local_rank
                                                     << " devices=" << placement_.devices.size()
                                                     << " compute_participant=" << compute_participant_);
    }

    MPITopology::MPITopology(int rank, int world_size, int ranks_per_node, MPI_Comm comm)
        : rank_(rank),
          world_size_(world_size),
          ranks_per_node_(ranks_per_node),
          world_comm_(comm),
          intra_node_comm_(MPI_COMM_NULL),
          inter_node_comm_(MPI_COMM_NULL),
          owns_comms_(false) // Don't create communicators in explicit mode
    {
        // Calculate derived values
        node_count_ = (world_size_ + ranks_per_node_ - 1) / ranks_per_node_;

        // Set placement
        placement_.rank = rank_;
        placement_.node_id = rank_ / ranks_per_node_;
        placement_.local_rank = rank_ % ranks_per_node_;
        placement_.socket_id = placement_.local_rank; // Assume socket = local_rank
        placement_.numa_node = placement_.socket_id;
        placement_.hostname = "explicit";

        // Add default CPU device
        DeviceCapability cpu_dev;
        cpu_dev.type = DeviceCapability::Type::CPU;
        cpu_dev.device_id = 0;
        cpu_dev.relative_compute = 1.0f;
        cpu_dev.memory_bytes = 0; // Unknown in explicit mode
        cpu_dev.name = "CPU";
        placement_.devices.push_back(cpu_dev);

        // Initialize all_placements_ with this rank's info
        all_placements_.resize(world_size_);
        all_placements_[rank_] = placement_;

        LOG_DEBUG("[MPITopology] Explicit init: rank=" << rank_
                                                       << "/" << world_size_
                                                       << " ranks_per_node=" << ranks_per_node_);
    }

    MPITopology::~MPITopology()
    {
        if (owns_comms_)
        {
            if (intra_node_comm_ != MPI_COMM_NULL)
            {
                MPI_Comm_free(&intra_node_comm_);
            }
            if (inter_node_comm_ != MPI_COMM_NULL)
            {
                MPI_Comm_free(&inter_node_comm_);
            }
        }
    }

    MPITopology::MPITopology(MPITopology &&other) noexcept
        : rank_(other.rank_),
          world_size_(other.world_size_),
          node_count_(other.node_count_),
          ranks_per_node_(other.ranks_per_node_),
          compute_participant_(other.compute_participant_),
          placement_(std::move(other.placement_)),
          all_placements_(std::move(other.all_placements_)),
          world_comm_(other.world_comm_),
          intra_node_comm_(other.intra_node_comm_),
          inter_node_comm_(other.inter_node_comm_),
          owns_comms_(other.owns_comms_)
    {
        other.intra_node_comm_ = MPI_COMM_NULL;
        other.inter_node_comm_ = MPI_COMM_NULL;
        other.owns_comms_ = false;
    }

    MPITopology &MPITopology::operator=(MPITopology &&other) noexcept
    {
        if (this != &other)
        {
            // Clean up existing
            if (owns_comms_)
            {
                if (intra_node_comm_ != MPI_COMM_NULL)
                    MPI_Comm_free(&intra_node_comm_);
                if (inter_node_comm_ != MPI_COMM_NULL)
                    MPI_Comm_free(&inter_node_comm_);
            }

            // Move
            rank_ = other.rank_;
            world_size_ = other.world_size_;
            node_count_ = other.node_count_;
            ranks_per_node_ = other.ranks_per_node_;
            compute_participant_ = other.compute_participant_;
            placement_ = std::move(other.placement_);
            all_placements_ = std::move(other.all_placements_);
            world_comm_ = other.world_comm_;
            intra_node_comm_ = other.intra_node_comm_;
            inter_node_comm_ = other.inter_node_comm_;
            owns_comms_ = other.owns_comms_;

            other.intra_node_comm_ = MPI_COMM_NULL;
            other.inter_node_comm_ = MPI_COMM_NULL;
            other.owns_comms_ = false;
        }
        return *this;
    }

    // =========================================================================
    // Topology Detection
    // =========================================================================

    void MPITopology::detect_topology()
    {
        // Get hostname for this rank
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        placement_.hostname = hostname;
        placement_.rank = rank_;

        // Use MPI_Comm_split_type to identify ranks on same node
        // This is the most reliable way to detect shared memory boundaries
        MPI_Comm shared_comm;
        MPI_Comm_split_type(world_comm_, MPI_COMM_TYPE_SHARED, rank_,
                            MPI_INFO_NULL, &shared_comm);

        int local_size, local_rank;
        MPI_Comm_size(shared_comm, &local_size);
        MPI_Comm_rank(shared_comm, &local_rank);

        placement_.local_rank = local_rank;
        ranks_per_node_ = local_size;

        // Calculate node_id: gather local_rank==0 from each node to determine ordering
        // Simple approach: node_id = rank / ranks_per_node (assumes round-robin assignment)
        // More robust: use hostname hashing
        placement_.node_id = rank_ / ranks_per_node_;
        node_count_ = (world_size_ + ranks_per_node_ - 1) / ranks_per_node_;

        // Store shared comm temporarily for setup_communicators
        intra_node_comm_ = shared_comm;

        LOG_TRACE("[MPITopology] detect_topology: hostname=" << placement_.hostname
                                                             << " local_rank=" << local_rank
                                                             << "/" << local_size
                                                             << " node_id=" << placement_.node_id);
    }

    void MPITopology::setup_communicators()
    {
        // intra_node_comm_ was already set in detect_topology via MPI_Comm_split_type

        // Create inter-node communicator (one rank per node)
        // Only local_rank 0 participates
        int color = is_node_leader() ? 0 : MPI_UNDEFINED;
        MPI_Comm_split(world_comm_, color, rank_, &inter_node_comm_);

        if (is_node_leader() && inter_node_comm_ != MPI_COMM_NULL)
        {
            int inter_size;
            MPI_Comm_size(inter_node_comm_, &inter_size);
            LOG_TRACE("[MPITopology] Inter-node comm created with " << inter_size << " ranks");
        }
    }

    void MPITopology::detect_numa_placement()
    {
        // Use NUMATopology to detect local NUMA node
        NUMAInfo numa_info = NUMATopology::detectLocalNUMANode();

        placement_.numa_node = numa_info.local_numa_node;
        placement_.socket_id = numa_info.local_numa_node; // Typically NUMA node == socket

        LOG_TRACE("[MPITopology] NUMA placement: node=" << placement_.numa_node
                                                        << " method=" << numa_info.detection_method);
    }

    void MPITopology::detect_device_capabilities()
    {
        placement_.devices.clear();

        // Detect CPU memory bandwidth first (for phase-aware decode placement)
        auto cpu_bw_info = NUMATopology::estimateCPUBandwidth();

        // Always add CPU as a device - CPU is a FIRST-CLASS decode participant!
        DeviceCapability cpu_dev;
        cpu_dev.type = DeviceCapability::Type::CPU;
        cpu_dev.device_id = 0;
        cpu_dev.relative_compute = 1.0f; // Baseline (for prefill compute weight)
        cpu_dev.memory_bytes = 0;        // TODO: Detect system memory
        cpu_dev.name = "CPU (socket " + std::to_string(placement_.socket_id) + ")";

        // Set CPU memory bandwidth - critical for decode phase placement!
        // This determines CPU's share of decode work (bandwidth-proportional sharding)
        // Per-socket bandwidth (divide by num_sockets since this is per-rank)
        if (cpu_bw_info.num_sockets > 0)
        {
            cpu_dev.compute_units = cpu_bw_info.memory_channels; // Channels as "units"
            // Note: For multi-rank-per-node, bandwidth is shared among local ranks
            // TODO: Better handling of intra-node bandwidth sharing
        }

        placement_.devices.push_back(cpu_dev);

        // Check environment for CUDA devices
        const char *cuda_visible = std::getenv("CUDA_VISIBLE_DEVICES");
        if (cuda_visible && strlen(cuda_visible) > 0)
        {
            std::string devices(cuda_visible);
            std::stringstream ss(devices);
            std::string item;
            while (std::getline(ss, item, ','))
            {
                try
                {
                    int device_id = std::stoi(item);
                    DeviceCapability cuda_dev;
                    cuda_dev.type = DeviceCapability::Type::CUDA;
                    cuda_dev.device_id = device_id;
                    cuda_dev.relative_compute = 10.0f; // GPUs typically 10x faster for GEMM
                    cuda_dev.memory_bytes = 0;         // TODO: Query CUDA runtime
                    cuda_dev.name = "CUDA:" + std::to_string(device_id);
                    placement_.devices.push_back(cuda_dev);
                }
                catch (...)
                {
                    // Ignore invalid device IDs
                }
            }
        }

        // Check environment for ROCm devices
        const char *hip_visible = std::getenv("HIP_VISIBLE_DEVICES");
        if (hip_visible && strlen(hip_visible) > 0)
        {
            std::string devices(hip_visible);
            std::stringstream ss(devices);
            std::string item;
            while (std::getline(ss, item, ','))
            {
                try
                {
                    int device_id = std::stoi(item);
                    DeviceCapability rocm_dev;
                    rocm_dev.type = DeviceCapability::Type::ROCm;
                    rocm_dev.device_id = device_id;
                    rocm_dev.relative_compute = 10.0f; // GPUs typically 10x faster for GEMM
                    rocm_dev.memory_bytes = 0;         // TODO: Query HIP runtime
                    rocm_dev.name = "ROCm:" + std::to_string(device_id);
                    placement_.devices.push_back(rocm_dev);
                }
                catch (...)
                {
                    // Ignore invalid device IDs
                }
            }
        }

        LOG_TRACE("[MPITopology] Device capabilities: " << placement_.devices.size() << " devices");
    }

    // =========================================================================
    // Device Capability Exchange
    // =========================================================================

    void MPITopology::exchangeCapabilities()
    {
        // Build local RankInventory from placement info
        RankInventory local_inventory;
        local_inventory.rank = rank_;
        local_inventory.node_id = placement_.node_id;
        local_inventory.local_rank = placement_.local_rank;
        local_inventory.hostname = placement_.hostname;
        local_inventory.numa_nodes = 1; // Default

        // Convert DeviceCapability to DeviceInfo for CPU and GPUs
        for (const auto &dev : placement_.devices)
        {
            DeviceInfo info;
            info.local_device_id = dev.device_id;
            info.memory_bytes = dev.memory_bytes;
            info.compute_units = static_cast<int>(dev.compute_units);
            info.name = dev.name;
            // Set relative TFLOPS based on relative_compute for weight calculations
            info.tflops_int8 = dev.relative_compute;

            if (dev.type == DeviceCapability::Type::CPU)
            {
                info.type = DeviceType::CPU;
                local_inventory.cpu = info;
            }
            else if (dev.type == DeviceCapability::Type::CUDA)
            {
                info.type = DeviceType::CUDA;
                local_inventory.gpus.push_back(info);
            }
            else if (dev.type == DeviceCapability::Type::ROCm)
            {
                info.type = DeviceType::ROCm;
                local_inventory.gpus.push_back(info);
            }
        }

        // Serialize local inventory
        std::vector<uint8_t> local_data = serializeRankInventory(local_inventory);
        int local_size = static_cast<int>(local_data.size());

        LOG_DEBUG("[MPITopology] Serialized local RankInventory: " << local_size << " bytes, "
                                                                   << local_inventory.gpus.size() << " GPUs");

        // Gather all sizes first (MPI_Allgather of sizes)
        std::vector<int> all_sizes(world_size_);
        MPI_Allgather(&local_size, 1, MPI_INT,
                      all_sizes.data(), 1, MPI_INT,
                      world_comm_);

        // Calculate displacements for MPI_Allgatherv
        std::vector<int> displacements(world_size_);
        int total_size = 0;
        for (int r = 0; r < world_size_; ++r)
        {
            displacements[r] = total_size;
            total_size += all_sizes[r];
        }

        // Allocate receive buffer
        std::vector<uint8_t> all_data(total_size);

        // Gather all serialized inventories
        MPI_Allgatherv(local_data.data(), local_size, MPI_BYTE,
                       all_data.data(), all_sizes.data(), displacements.data(), MPI_BYTE,
                       world_comm_);

        // Deserialize all inventories into ClusterInventory
        cluster_inventory_.world_size = world_size_;
        cluster_inventory_.node_count = node_count_;
        cluster_inventory_.ranks.resize(world_size_);

        bool has_cuda = false;
        bool has_rocm = false;

        for (int r = 0; r < world_size_; ++r)
        {
            const uint8_t *ptr = all_data.data() + displacements[r];
            size_t size = static_cast<size_t>(all_sizes[r]);

            try
            {
                cluster_inventory_.ranks[r] = deserializeRankInventory(ptr, size);

                // Track GPU types for heterogeneous detection
                for (const auto &gpu : cluster_inventory_.ranks[r].gpus)
                {
                    if (gpu.type == DeviceType::CUDA)
                    {
                        has_cuda = true;
                    }
                    else if (gpu.type == DeviceType::ROCm)
                    {
                        has_rocm = true;
                    }
                }

                LOG_TRACE("[MPITopology] Deserialized rank " << r << ": "
                                                             << cluster_inventory_.ranks[r].hostname << ", "
                                                             << cluster_inventory_.ranks[r].gpus.size() << " GPUs");
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[MPITopology] Failed to deserialize inventory for rank " << r
                                                                                    << ": " << e.what());
                // Create empty placeholder
                cluster_inventory_.ranks[r].rank = r;
                cluster_inventory_.ranks[r].hostname = "error";
            }
        }

        // Build node aggregations
        cluster_inventory_.buildNodeAggregations();
        cluster_inventory_built_ = true;

        // Also update all_placements_ for backward compatibility
        all_placements_.resize(world_size_);
        all_placements_[rank_] = placement_;

        for (int r = 0; r < world_size_; ++r)
        {
            if (r != rank_)
            {
                const auto &ri = cluster_inventory_.ranks[r];
                RankPlacement remote;
                remote.rank = ri.rank;
                remote.node_id = ri.node_id;
                remote.local_rank = ri.local_rank;
                remote.socket_id = ri.local_rank; // Simplified
                remote.numa_node = ri.local_rank; // Simplified
                remote.hostname = ri.hostname;

                // Convert DeviceInfo back to DeviceCapability
                DeviceCapability cpu_cap;
                cpu_cap.type = DeviceCapability::Type::CPU;
                cpu_cap.device_id = ri.cpu.local_device_id;
                cpu_cap.memory_bytes = ri.cpu.memory_bytes;
                cpu_cap.compute_units = ri.cpu.compute_units;
                cpu_cap.relative_compute = ri.cpu.tflops_int8; // Stored here
                cpu_cap.name = ri.cpu.name;
                remote.devices.push_back(cpu_cap);

                for (const auto &gpu : ri.gpus)
                {
                    DeviceCapability gpu_cap;
                    if (gpu.type == DeviceType::CUDA)
                    {
                        gpu_cap.type = DeviceCapability::Type::CUDA;
                    }
                    else if (gpu.type == DeviceType::ROCm)
                    {
                        gpu_cap.type = DeviceCapability::Type::ROCm;
                    }
                    else
                    {
                        gpu_cap.type = DeviceCapability::Type::Unknown;
                    }
                    gpu_cap.device_id = gpu.local_device_id;
                    gpu_cap.memory_bytes = gpu.memory_bytes;
                    gpu_cap.compute_units = gpu.compute_units;
                    gpu_cap.relative_compute = gpu.tflops_int8; // Stored here
                    gpu_cap.name = gpu.name;
                    remote.devices.push_back(gpu_cap);
                }

                all_placements_[r] = remote;
            }
        }

        // Log heterogeneous status
        if (has_cuda && has_rocm)
        {
            LOG_INFO("[MPITopology] Heterogeneous cluster detected: CUDA + ROCm GPUs present");
        }

        LOG_DEBUG("[MPITopology] Full capability exchange complete for " << world_size_ << " ranks, "
                                                                         << cluster_inventory_.total_gpus << " total GPUs");
    }

    // =========================================================================
    // Topology Queries
    // =========================================================================

    const RankPlacement &MPITopology::get_placement(int rank) const
    {
        if (rank < 0 || rank >= world_size_ || static_cast<size_t>(rank) >= all_placements_.size())
        {
            LOG_ERROR("[MPITopology] Invalid rank " << rank << " requested");
            return placement_; // Return local placement as fallback
        }
        return all_placements_[rank];
    }

    int MPITopology::compute_world_size() const
    {
        // Count ranks that participate in compute
        // For now, all ranks participate
        return world_size_;
    }

    bool MPITopology::same_node(int rank_a, int rank_b) const
    {
        return (rank_a / ranks_per_node_) == (rank_b / ranks_per_node_);
    }

    // =========================================================================
    // Work Distribution
    // =========================================================================

    WorkRange MPITopology::get_head_range(int total_heads) const
    {
        return WorkRange::for_rank_equal(total_heads, rank_, world_size_);
    }

    WorkRange MPITopology::get_kv_head_range(int total_kv_heads) const
    {
        // GQA-aware: if fewer KV heads than ranks, some ranks get empty ranges
        // This is intentional - those ranks still compute Q heads but share KV
        if (total_kv_heads < world_size_)
        {
            // Each KV head is assigned to one rank
            if (rank_ < total_kv_heads)
            {
                return {static_cast<size_t>(rank_), static_cast<size_t>(rank_ + 1)};
            }
            else
            {
                return {0, 0}; // Empty range
            }
        }
        return WorkRange::for_rank_equal(total_kv_heads, rank_, world_size_);
    }

    WorkRange MPITopology::get_column_range(size_t total_cols) const
    {
        return WorkRange::for_rank_equal(total_cols, rank_, world_size_);
    }

    WorkRange MPITopology::get_row_range(size_t total_rows) const
    {
        return WorkRange::for_rank_equal(total_rows, rank_, world_size_);
    }

    WorkRange MPITopology::get_vocab_range(size_t vocab_size) const
    {
        return WorkRange::for_rank_equal(vocab_size, rank_, world_size_);
    }

    WorkRange MPITopology::get_ffn_range(size_t ffn_dim) const
    {
        return WorkRange::for_rank_equal(ffn_dim, rank_, world_size_);
    }

    // =========================================================================
    // SliceMetadata Creation
    // =========================================================================

    SliceMetadata MPITopology::createRowParallelMeta(
        size_t original_rows,
        size_t original_cols,
        bool inner_is_presliced) const
    {
        return SliceMetadata::forRowParallel(
            original_rows, original_cols,
            rank_, world_size_,
            inner_is_presliced);
    }

    SliceMetadata MPITopology::createColumnParallelMeta(
        size_t original_rows,
        size_t original_cols,
        bool inner_is_presliced) const
    {
        return SliceMetadata::forColumnParallel(
            original_rows, original_cols,
            rank_, world_size_,
            inner_is_presliced);
    }

    // =========================================================================
    // Compute Weights
    // =========================================================================

    std::vector<float> MPITopology::get_compute_weights() const
    {
        std::vector<float> weights(world_size_);
        for (int r = 0; r < world_size_; ++r)
        {
            const auto &placement = all_placements_[r];
            float total = 0.0f;
            for (const auto &dev : placement.devices)
            {
                total += dev.relative_compute;
            }
            weights[r] = total;
        }
        return weights;
    }

    // =========================================================================
    // Device Mapping
    // =========================================================================

    int MPITopology::get_device() const
    {
        // Return first accelerator if available, otherwise CPU
        for (const auto &dev : placement_.devices)
        {
            if (dev.type == DeviceCapability::Type::CUDA ||
                dev.type == DeviceCapability::Type::ROCm)
            {
                return dev.device_id;
            }
        }
        return 0; // Default to device 0 (CPU)
    }

    bool MPITopology::has_accelerator() const
    {
        for (const auto &dev : placement_.devices)
        {
            if (dev.type == DeviceCapability::Type::CUDA ||
                dev.type == DeviceCapability::Type::ROCm)
            {
                return true;
            }
        }
        return false;
    }

    // =========================================================================
    // Debugging
    // =========================================================================

    std::string MPITopology::to_string() const
    {
        std::ostringstream oss;
        oss << "MPITopology{"
            << "rank=" << rank_ << "/" << world_size_
            << ", node=" << placement_.node_id << "/" << node_count_
            << ", local_rank=" << placement_.local_rank << "/" << ranks_per_node_
            << ", numa=" << placement_.numa_node
            << ", hostname=" << placement_.hostname
            << ", compute_participant=" << (compute_participant_ ? "yes" : "no")
            << ", devices=[";
        for (size_t i = 0; i < placement_.devices.size(); ++i)
        {
            if (i > 0)
                oss << ", ";
            const auto &dev = placement_.devices[i];
            oss << dev.name << "(compute=" << dev.relative_compute << ")";
        }
        oss << "]}";
        return oss.str();
    }

    void MPITopology::print_topology() const
    {
        // Gather all rank info to rank 0 for unified printing
        std::string local_info = to_string();

        if (is_coordinator())
        {
            LOG_INFO("=== MPI Topology ===");
            LOG_INFO("  Total ranks: " << world_size_);
            LOG_INFO("  Nodes: " << node_count_);
            LOG_INFO("  Ranks per node: " << ranks_per_node_);
            LOG_INFO("  ALL ranks participate in compute (default)");
        }

        // Barrier to ensure clean output
        MPI_Barrier(world_comm_);

        // Each rank prints its info in order
        for (int r = 0; r < world_size_; ++r)
        {
            if (rank_ == r)
            {
                LOG_INFO("  Rank " << r << ": " << local_info);
            }
            MPI_Barrier(world_comm_);
        }
    }

    // =========================================================================
    // Placement Strategy
    // =========================================================================

    PlacementPlan MPITopology::computePlacement(
        const std::string &architecture,
        int n_layers,
        size_t d_model,
        size_t d_ff,
        size_t vocab_size,
        size_t n_heads,
        size_t n_kv_heads,
        const std::string &quant_type,
        size_t estimated_memory,
        const std::string &strategy_name) const
    {
        PlacementInput input;
        input.architecture = architecture;
        input.n_layers = n_layers;
        input.d_model = d_model;
        input.d_ff = d_ff;
        input.vocab_size = vocab_size;
        input.n_heads = n_heads;
        input.n_kv_heads = n_kv_heads;
        input.quant_type = quant_type;
        input.estimated_memory_bytes = estimated_memory;
        input.preferred_strategy = strategy_name;

        return computePlacement(input);
    }

    PlacementPlan MPITopology::computePlacement(const PlacementInput &input_ref) const
    {
        // Make a copy since we need to modify it
        PlacementInput input = input_ref;

        // Fill in topology fields from our gathered data
        input.world_size = world_size_;
        input.ranks_per_node = ranks_per_node_;
        input.node_count = node_count_;

        // Compute aggregated device info from all_placements_
        input.rank_compute_weights.resize(world_size_);
        input.any_rank_has_gpu = false;
        input.total_gpu_memory = 0;
        input.total_cpu_memory = 0;

        for (int r = 0; r < world_size_; ++r)
        {
            const auto &rp = (r < static_cast<int>(all_placements_.size()))
                                 ? all_placements_[r]
                                 : placement_;
            input.rank_compute_weights[r] = rp.total_compute_power();

            for (const auto &dev : rp.devices)
            {
                if (dev.type == DeviceCapability::Type::CUDA ||
                    dev.type == DeviceCapability::Type::ROCm)
                {
                    input.any_rank_has_gpu = true;
                    input.total_gpu_memory += dev.memory_bytes;
                }
                else if (dev.type == DeviceCapability::Type::CPU)
                {
                    input.total_cpu_memory += dev.memory_bytes;
                }
            }
        }

        // Auto-select and run strategy
        auto strategy = PlacementStrategyFactory::autoSelect(input);
        if (!strategy)
        {
            LOG_ERROR("[MPITopology] Failed to select placement strategy");
            // Return empty plan
            PlacementPlan empty;
            empty.strategy_name = "ERROR";
            return empty;
        }

        LOG_DEBUG("[MPITopology] Computing placement with strategy: " << strategy->name());
        PlacementPlan plan = strategy->compute(input);

        if (rank_ == 0)
        {
            LOG_INFO("[MPITopology] Placement plan computed:\n"
                     << plan.toString());
        }

        return plan;
    }

    // =========================================================================
    // ClusterInventory (IMPITopology interface)
    // =========================================================================

    const ClusterInventory &MPITopology::clusterInventory() const
    {
        if (!cluster_inventory_built_)
        {
            buildClusterInventory();
        }
        return cluster_inventory_;
    }

    void MPITopology::buildClusterInventory() const
    {
        cluster_inventory_.world_size = world_size_;
        cluster_inventory_.node_count = node_count_;
        cluster_inventory_.ranks.resize(world_size_);

        // Convert RankPlacement to RankInventory
        for (int r = 0; r < world_size_; ++r)
        {
            const auto &rp = (r < static_cast<int>(all_placements_.size()))
                                 ? all_placements_[r]
                                 : placement_;

            auto &ri = cluster_inventory_.ranks[r];
            ri.rank = rp.rank;
            ri.node_id = rp.node_id;
            ri.local_rank = rp.local_rank;
            ri.hostname = rp.hostname;
            ri.numa_nodes = 1; // Simplified from RankPlacement

            // Convert DeviceCapability to DeviceInfo
            for (const auto &dev : rp.devices)
            {
                if (dev.type == DeviceCapability::Type::CPU)
                {
                    ri.cpu.type = DeviceType::CPU;
                    ri.cpu.local_device_id = dev.device_id;
                    ri.cpu.memory_bytes = dev.memory_bytes;
                    ri.cpu.compute_units = static_cast<int>(dev.compute_units);
                    ri.cpu.name = dev.name;
                }
                else
                {
                    DeviceInfo gpu;
                    if (dev.type == DeviceCapability::Type::CUDA)
                    {
                        gpu.type = DeviceType::CUDA;
                    }
                    else if (dev.type == DeviceCapability::Type::ROCm)
                    {
                        gpu.type = DeviceType::ROCm;
                    }
                    gpu.local_device_id = dev.device_id;
                    gpu.memory_bytes = dev.memory_bytes;
                    gpu.compute_units = static_cast<int>(dev.compute_units);
                    gpu.name = dev.name;
                    ri.gpus.push_back(gpu);
                }
            }
        }

        // Build node aggregations
        cluster_inventory_.buildNodeAggregations();
        cluster_inventory_built_ = true;
    }

    // =========================================================================
    // Heterogeneous Device Detection
    // =========================================================================

    bool MPITopology::hasHeterogeneousGPUs() const
    {
        // Ensure cluster inventory is built
        const auto &inventory = clusterInventory();

        bool has_cuda = false;
        bool has_rocm = false;

        for (const auto &rank_inv : inventory.ranks)
        {
            for (const auto &gpu : rank_inv.gpus)
            {
                if (gpu.type == DeviceType::CUDA)
                {
                    has_cuda = true;
                }
                else if (gpu.type == DeviceType::ROCm)
                {
                    has_rocm = true;
                }

                // Early exit if both found
                if (has_cuda && has_rocm)
                {
                    return true;
                }
            }
        }

        return false;
    }

    const RankInventory &MPITopology::getRankInventory(int rank) const
    {
        const auto &inventory = clusterInventory();
        return inventory.getRank(rank);
    }

} // namespace llaminar2
