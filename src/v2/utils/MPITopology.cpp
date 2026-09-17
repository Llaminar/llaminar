/**
 * @file MPITopology.cpp
 * @brief Implementation of MPI topology abstraction
 *
 * Design principles:
 * - ALL ranks (including rank 0) participate in compute by default
 * - Equal work division initially; future support for weighted distribution
 * - Integrates with existing SliceMetadata from tensors/TensorSlice.h
 * - ClusterInventoryGatherer owns physical observations and exchange;
 *   RankPlacement is a read-only projection, never a second discovery path.
 * - Inventory access is collective-free; the complete snapshot is installed
 *   during construction and follows the topology through moves.
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include "MPITopology.h"
#include "Logger.h"
#include "MPIContext.h"
#include "../tensors/TensorSlice.h"
#include "../execution/mpi_orchestration/PlacementStrategy.h"

#include <sstream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstring> // memcpy
#include <unistd.h>

namespace llaminar2
{

    // =========================================================================
    // Serialization Helpers
    // =========================================================================

    namespace
    {
        constexpr uint32_t kInventoryMagic = 0x4c494e56u; ///< LINV, not an unversioned rank id.
        constexpr uint32_t kInventoryVersion = 3u; ///< Distinguishes CPU execution workers from physical cores.
        /** @brief Lend a checked communicator to a standalone topology constructor. */
        MPIContext standaloneTopologyContext(MPI_Comm comm)
        {
            if (comm == MPI_COMM_NULL)
                throw std::invalid_argument("MPI topology requires a non-null communicator");
            int rank = -1, size = 0;
            if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS ||
                MPI_Comm_size(comm, &size) != MPI_SUCCESS)
                throw std::runtime_error("Could not inspect MPI topology communicator");
            return MPIContext(rank, size, comm);
        }

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

        // Serialize a vector of ints (length-prefixed)
        void writeIntVector(std::vector<uint8_t> &buffer, const std::vector<int> &vec)
        {
            writeValue(buffer, static_cast<int32_t>(vec.size()));
            for (int v : vec)
            {
                writeValue(buffer, static_cast<int32_t>(v));
            }
        }

        // Deserialize a vector of ints
        std::vector<int> readIntVector(const uint8_t *&ptr, const uint8_t *end)
        {
            int32_t count = readValue<int32_t>(ptr, end);
            std::vector<int> vec;
            vec.reserve(count);
            for (int32_t i = 0; i < count; ++i)
            {
                vec.push_back(readValue<int32_t>(ptr, end));
            }
            return vec;
        }

        // Serialize a CPUSocketInfo
        void serializeCPUSocketInfo(std::vector<uint8_t> &buffer, const CPUSocketInfo &info)
        {
            writeValue(buffer, static_cast<int32_t>(info.socket_id));
            writeValue(buffer, static_cast<int32_t>(info.numa_node));
            writeString(buffer, info.model_name);
            writeIntVector(buffer, info.physical_cores);
            writeIntVector(buffer, info.ht_threads);
            writeValue(buffer, static_cast<uint64_t>(info.memory_bytes));
            writeValue(buffer, static_cast<uint64_t>(info.available_memory_bytes));
        }

        // Deserialize a CPUSocketInfo
        CPUSocketInfo deserializeCPUSocketInfo(const uint8_t *&ptr, const uint8_t *end)
        {
            CPUSocketInfo info;
            info.socket_id = readValue<int32_t>(ptr, end);
            info.numa_node = readValue<int32_t>(ptr, end);
            info.model_name = readString(ptr, end);
            info.physical_cores = readIntVector(ptr, end);
            info.ht_threads = readIntVector(ptr, end);
            info.memory_bytes = readValue<uint64_t>(ptr, end);
            info.available_memory_bytes = readValue<uint64_t>(ptr, end);
            return info;
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
            // PCIe link info
            writeValue(buffer, static_cast<int32_t>(info.pcie_gen));
            writeValue(buffer, static_cast<int32_t>(info.pcie_width));
            writeValue(buffer, info.pcie_speed_gts);
            writeValue(buffer, static_cast<int32_t>(info.pcie_max_width));
            writeValue(buffer, info.pcie_max_speed_gts);
            writeValue(buffer, static_cast<uint8_t>(info.pcie_degraded ? 1 : 0));
            writeString(buffer, info.pcie_bottleneck_bdf);
            writeValue(buffer, static_cast<uint64_t>(info.last_level_cache_bytes));
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
            // PCIe link info
            info.pcie_gen = readValue<int32_t>(ptr, end);
            info.pcie_width = readValue<int32_t>(ptr, end);
            info.pcie_speed_gts = readValue<double>(ptr, end);
            info.pcie_max_width = readValue<int32_t>(ptr, end);
            info.pcie_max_speed_gts = readValue<double>(ptr, end);
            info.pcie_degraded = (readValue<uint8_t>(ptr, end) != 0);
            info.pcie_bottleneck_bdf = readString(ptr, end);
            info.last_level_cache_bytes = readValue<uint64_t>(ptr, end);
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

        // Every rank in one launch runs the same certified artifact. Reject
        // incompatible inventories instead of inventing missing CPU geometry.
        writeValue(buffer, kInventoryMagic);
        writeValue(buffer, kInventoryVersion);

        // Write rank identification
        writeValue(buffer, static_cast<int32_t>(inventory.rank));
        writeValue(buffer, static_cast<int32_t>(inventory.node_id));
        writeValue(buffer, static_cast<int32_t>(inventory.local_rank));
        writeString(buffer, inventory.hostname);

        // Write CPU info
        writeValue(buffer, static_cast<int32_t>(inventory.cpu_cores));
        writeValue(buffer, static_cast<int32_t>(inventory.cpu_worker_threads));
        writeValue(buffer, static_cast<int32_t>(inventory.cpu_sockets));
        writeValue(buffer, static_cast<int32_t>(inventory.numa_nodes));
        writeValue(buffer, static_cast<uint64_t>(inventory.cpu_memory_bytes));

        // Write CPU device info
        serializeDeviceInfo(buffer, inventory.cpu);
        writeValue(buffer, inventory.cpu_execution.cache.private_l2_bytes);
        writeValue(buffer, inventory.cpu_execution.cache.shared_l3_bytes);
        writeValue(buffer, inventory.cpu_execution.cache.private_l2_ways);
        writeValue(buffer, inventory.cpu_execution.cache.shared_l3_ways);
        writeValue(buffer, inventory.cpu_execution.maximum_native_row_tile);

        // Write GPU count and GPU device infos
        writeValue(buffer, static_cast<int32_t>(inventory.gpus.size()));
        for (const auto &gpu : inventory.gpus)
        {
            serializeDeviceInfo(buffer, gpu);
        }

        // Write per-socket CPU info
        writeValue(buffer, static_cast<int32_t>(inventory.cpu_socket_info.size()));
        for (const auto &sock : inventory.cpu_socket_info)
        {
            serializeCPUSocketInfo(buffer, sock);
        }

        // Write P2P matrices
        writeValue(buffer, static_cast<int32_t>(inventory.p2p_cuda_count));
        for (int i = 0; i < static_cast<int>(inventory.p2p_cuda.size()); ++i)
        {
            writeValue(buffer, static_cast<uint8_t>(inventory.p2p_cuda[i] ? 1 : 0));
        }
        writeValue(buffer, static_cast<int32_t>(inventory.p2p_rocm_count));
        for (int i = 0; i < static_cast<int>(inventory.p2p_rocm.size()); ++i)
        {
            writeValue(buffer, static_cast<uint8_t>(inventory.p2p_rocm[i] ? 1 : 0));
        }

        return buffer;
    }

    RankInventory MPITopology::deserializeRankInventory(const uint8_t *data, size_t size)
    {
        const uint8_t *ptr = data;
        const uint8_t *end = data + size;

        RankInventory inventory;

        if (readValue<uint32_t>(ptr, end) != kInventoryMagic ||
            readValue<uint32_t>(ptr, end) != kInventoryVersion)
            throw std::runtime_error("Incompatible MPI inventory ABI; all ranks must run the same artifact");

        // Read rank identification
        inventory.rank = readValue<int32_t>(ptr, end);
        inventory.node_id = readValue<int32_t>(ptr, end);
        inventory.local_rank = readValue<int32_t>(ptr, end);
        inventory.hostname = readString(ptr, end);

        // Read CPU info
        inventory.cpu_cores = readValue<int32_t>(ptr, end);
        inventory.cpu_worker_threads = readValue<int32_t>(ptr, end);
        if (inventory.cpu_worker_threads < 0)
            throw std::invalid_argument("MPI inventory contains a negative CPU worker budget");
        inventory.cpu_sockets = readValue<int32_t>(ptr, end);
        inventory.numa_nodes = readValue<int32_t>(ptr, end);
        inventory.cpu_memory_bytes = readValue<uint64_t>(ptr, end);

        // Read CPU device info
        inventory.cpu = deserializeDeviceInfo(ptr, end);
        inventory.cpu_execution.cache.private_l2_bytes = readValue<uint64_t>(ptr, end);
        inventory.cpu_execution.cache.shared_l3_bytes = readValue<uint64_t>(ptr, end);
        inventory.cpu_execution.cache.private_l2_ways = readValue<uint32_t>(ptr, end);
        inventory.cpu_execution.cache.shared_l3_ways = readValue<uint32_t>(ptr, end);
        inventory.cpu_execution.maximum_native_row_tile = readValue<uint32_t>(ptr, end);

        // Read GPU count and GPU device infos
        int32_t gpu_count = readValue<int32_t>(ptr, end);
        inventory.gpus.reserve(gpu_count);
        for (int32_t i = 0; i < gpu_count; ++i)
        {
            inventory.gpus.push_back(deserializeDeviceInfo(ptr, end));
        }

        // Socket and directed peer observations are mandatory in this ABI.
        {
            int32_t socket_count = readValue<int32_t>(ptr, end);
            inventory.cpu_socket_info.reserve(socket_count);
            for (int32_t i = 0; i < socket_count; ++i)
            {
                inventory.cpu_socket_info.push_back(deserializeCPUSocketInfo(ptr, end));
            }
        }

        {
            inventory.p2p_cuda_count = readValue<int32_t>(ptr, end);
            int cuda_matrix_size = inventory.p2p_cuda_count * inventory.p2p_cuda_count;
            inventory.p2p_cuda.resize(cuda_matrix_size);
            for (int i = 0; i < cuda_matrix_size; ++i)
            {
                inventory.p2p_cuda[i] = (readValue<uint8_t>(ptr, end) != 0);
            }
            inventory.p2p_rocm_count = readValue<int32_t>(ptr, end);
            int rocm_matrix_size = inventory.p2p_rocm_count * inventory.p2p_rocm_count;
            inventory.p2p_rocm.resize(rocm_matrix_size);
            for (int i = 0; i < rocm_matrix_size; ++i)
            {
                inventory.p2p_rocm[i] = (readValue<uint8_t>(ptr, end) != 0);
            }
        }

        if (ptr != end)
            throw std::runtime_error("Unexpected trailing bytes in MPI inventory");
        return inventory;
    }

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    MPITopology::MPITopology(MPI_Comm comm)
        : MPITopology(standaloneTopologyContext(comm))
    {
    }

    MPITopology::MPITopology(const IMPIContext &context)
        : rank_(context.rank()),
          world_size_(context.world_size()),
          world_comm_(context.communicator()),
          intra_node_comm_(MPI_COMM_NULL),
          inter_node_comm_(MPI_COMM_NULL),
          owns_comms_(true)
    {
        if (world_comm_ == MPI_COMM_NULL || world_size_ <= 0 || rank_ < 0 || rank_ >= world_size_)
            throw std::invalid_argument("MPI topology requires an exact live context");
        const auto inventory = context.clusterInventory();
        if (!inventory || inventory->world_size != world_size_ ||
            inventory->ranks.size() != static_cast<std::size_t>(world_size_))
            throw std::invalid_argument("MPI context published incomplete topology inventory");
        node_count_ = inventory->node_count;
        for (const auto &peer : inventory->ranks) rank_node_ids_.push_back(peer.node_id);
        installInventory(inventory);
        try
        {
            // Only communicator resources remain to be constructed. Physical
            // identity already belongs to the context's frozen observation.
            detect_topology();
            setup_communicators();
        }
        catch (...)
        {
            releaseCommunicators();
            throw;
        }

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
        // Use hostname-based node detection even in explicit mode
        // All ranks share hostname "explicit", so they all land on node 0
        std::vector<std::string> hostnames(static_cast<size_t>(world_size), "explicit");
        auto detection = NodeDetection::fromHostnames(hostnames);
        rank_node_ids_ = std::move(detection.node_ids);
        node_count_ = detection.node_count;

        // Set placement
        placement_.rank = rank_;
        placement_.node_id = rank_node_ids_[rank_];
        placement_.local_rank = rank_ % ranks_per_node_;
        placement_.socket_id = placement_.local_rank; // Assume socket = local_rank
        placement_.numa_node = placement_.socket_id;
        placement_.hostname = "explicit";

        /*
         * Explicit topology objects are test/setup fixtures and do not own a
         * live shared communicator.  Give every fixture for the same declared
         * topology one stable non-zero namespace so locality-dependent pure
         * planning remains deterministic without pretending it is a fresh
         * production MPI run.
         */
        node_shared_memory_namespace_ =
            0x4d5049544f504f4cULL ^
            (static_cast<uint64_t>(static_cast<uint32_t>(world_size_)) << 16u) ^
            static_cast<uint64_t>(static_cast<uint32_t>(ranks_per_node_));

        // Add default CPU device
        DeviceCapability cpu_dev;
        cpu_dev.type = DeviceCapability::Type::CPU;
        cpu_dev.device_id = 0;
        cpu_dev.relative_compute = 1.0f;
        cpu_dev.memory_bytes = 0; // Unknown in explicit mode
        cpu_dev.name = "CPU";
        placement_.devices.push_back(cpu_dev);

        // Explicit constructors are device-free synthetic fixtures. Populate
        // every record now, without a later discovery or reconstruction from
        // a partially initialized placement vector.
        ClusterInventory fixture;
        fixture.world_size = world_size_;
        fixture.node_count = node_count_;
        for (int peer = 0; peer < world_size_; ++peer)
        {
            RankInventory record;
            record.rank = peer;
            record.node_id = rank_node_ids_[peer];
            record.local_rank = peer % ranks_per_node_;
            record.hostname = "explicit";
            record.cpu.type = DeviceType::CPU;
            record.cpu.numa_node = record.local_rank;
            record.cpu.name = "CPU";
            record.cpu.compute_units = 10; // Fixture baseline relative weight 1.
            fixture.ranks.push_back(std::move(record));
        }
        fixture.buildNodeAggregations();
        installInventory(std::make_shared<const ClusterInventory>(std::move(fixture)));

        LOG_DEBUG("[MPITopology] Explicit init: rank=" << rank_
                                                       << "/" << world_size_
                                                       << " ranks_per_node=" << ranks_per_node_);
    }

    MPITopology::~MPITopology()
    {
        releaseCommunicators();
    }

    void MPITopology::releaseCommunicators() noexcept
    {
        if (owns_comms_)
        {
            // Guard against static destruction after MPI_Finalize
            int mpi_finalized = 0;
            MPI_Finalized(&mpi_finalized);
            if (mpi_finalized)
                return;

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
          rank_node_ids_(std::move(other.rank_node_ids_)),
          all_placements_(std::move(other.all_placements_)),
          node_shared_memory_namespace_(other.node_shared_memory_namespace_),
          cluster_inventory_(std::move(other.cluster_inventory_)),
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
            releaseCommunicators();

            // Move
            rank_ = other.rank_;
            world_size_ = other.world_size_;
            node_count_ = other.node_count_;
            ranks_per_node_ = other.ranks_per_node_;
            compute_participant_ = other.compute_participant_;
            placement_ = std::move(other.placement_);
            rank_node_ids_ = std::move(other.rank_node_ids_);
            all_placements_ = std::move(other.all_placements_);
            cluster_inventory_ = std::move(other.cluster_inventory_);
            node_shared_memory_namespace_ =
                other.node_shared_memory_namespace_;
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
        placement_.rank = rank_;

        // Use MPI_Comm_split_type to identify ranks on same node
        // This is the most reliable way to detect shared memory boundaries
        if (MPI_Comm_split_type(world_comm_, MPI_COMM_TYPE_SHARED, rank_,
                               MPI_INFO_NULL, &intra_node_comm_) != MPI_SUCCESS)
            throw std::runtime_error("Could not create node-local topology communicator");
        const MPI_Comm shared_comm = intra_node_comm_;

        int local_size, local_rank;
        if (MPI_Comm_size(shared_comm, &local_size) != MPI_SUCCESS ||
            MPI_Comm_rank(shared_comm, &local_rank) != MPI_SUCCESS)
            throw std::runtime_error("Could not inspect node-local topology communicator");
        if (local_rank != placement_.local_rank ||
            local_size != std::count(rank_node_ids_.begin(), rank_node_ids_.end(), placement_.node_id))
            throw std::invalid_argument("Observed inventory disagrees with node-local communicator membership");
        ranks_per_node_ = local_size;

        /*
         * Only the node leader invents the run identity.  Broadcasting on the
         * communicator returned by MPI_COMM_TYPE_SHARED proves that every
         * process deriving a POSIX channel name sees the same value without
         * imposing a world-wide setup barrier.  PID and monotonic time make a
         * stale segment from a crashed prior run unable to alias this run.
         */
        if (local_rank == 0)
        {
            const uint64_t now = static_cast<uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            const uint64_t pid =
                static_cast<uint64_t>(static_cast<uint32_t>(::getpid()));
            node_shared_memory_namespace_ =
                now ^ (pid << 32u) ^
                (static_cast<uint64_t>(
                     static_cast<uint32_t>(placement_.node_id))
                 << 1u) ^
                0x4c4c414d494e4152ULL;
            if (node_shared_memory_namespace_ == 0)
                node_shared_memory_namespace_ = 1;
        }
        if (MPI_Bcast(
                &node_shared_memory_namespace_,
                1,
                MPI_UINT64_T,
                0,
                shared_comm) != MPI_SUCCESS ||
            node_shared_memory_namespace_ == 0)
        {
            throw std::runtime_error(
                "MPITopology could not publish a node-local shared-memory namespace");
        }

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
        if (MPI_Comm_split(world_comm_, color, rank_, &inter_node_comm_) != MPI_SUCCESS)
            throw std::runtime_error("Could not create inter-node topology communicator");

        if (is_node_leader() && inter_node_comm_ != MPI_COMM_NULL)
        {
            int inter_size;
            MPI_Comm_size(inter_node_comm_, &inter_size);
            LOG_TRACE("[MPITopology] Inter-node comm created with " << inter_size << " ranks");
        }
    }

    void MPITopology::installInventory(std::shared_ptr<const ClusterInventory> snapshot)
    {
        if (!snapshot) throw std::invalid_argument("MPI topology needs an inventory publication");
        const auto &inventory = *snapshot;
        if (inventory.world_size != world_size_ ||
            inventory.ranks.size() != static_cast<std::size_t>(world_size_) ||
            inventory.node_count != node_count_)
            throw std::invalid_argument("Inventory does not describe this MPI topology");

        std::vector<RankPlacement> projections;
        projections.reserve(inventory.ranks.size());
        for (int rank = 0; rank < world_size_; ++rank)
        {
            const auto &observed = inventory.ranks[rank];
            if (observed.rank != rank || observed.node_id != rank_node_ids_[rank])
                throw std::invalid_argument("Inventory rank identity disagrees with MPI topology");
            RankPlacement projection;
            projection.rank = observed.rank;
            projection.node_id = observed.node_id;
            projection.local_rank = observed.local_rank;
            projection.numa_node = observed.cpu.numa_node;
            projection.socket_id = -1;
            projection.hostname = observed.hostname;
            for (const auto &socket : observed.cpu_socket_info)
                if (socket.numa_node == observed.cpu.numa_node)
                    projection.socket_id = socket.socket_id;

            const auto append = [&](const DeviceInfo &device)
            {
                DeviceCapability capability;
                switch (device.type)
                {
                case DeviceType::CPU: capability.type = DeviceCapability::Type::CPU; break;
                case DeviceType::CUDA: capability.type = DeviceCapability::Type::CUDA; break;
                case DeviceType::ROCm: capability.type = DeviceCapability::Type::ROCm; break;
                default: capability.type = DeviceCapability::Type::Unknown; break;
                }
                capability.device_id = device.local_device_id;
                capability.memory_bytes = device.memory_bytes;
                capability.compute_units = device.compute_units;
                capability.name = device.name;
                // Preserve the established placement heuristic as a derived
                // view. It is not measured TFLOPS and must never be written
                // back into the canonical hardware observation as such.
                capability.relative_compute = device.computeWeight();
                projection.devices.push_back(std::move(capability));
            };
            append(observed.cpu);
            for (const auto &gpu : observed.gpus) append(gpu);
            projections.push_back(std::move(projection));
        }
        // Install only after the entire snapshot validates. Accessors neither
        // discover hardware nor perform collectives, including on rank zero.
        cluster_inventory_ = std::move(snapshot);
        all_placements_ = std::move(projections);
        placement_ = all_placements_.at(rank_);
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
        if (!cluster_inventory_)
            throw std::logic_error("Physical locality requires the canonical cluster inventory");
        return cluster_inventory_->connectionBetweenRanks(rank_a, rank_b).locality() !=
               RankConnectionLocality::CrossNode;
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
        const std::string &kv_cache_precision,
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
        input.kv_cache_precision = kv_cache_precision;
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
        if (!cluster_inventory_)
            throw std::logic_error("Moved-from MPI topology has no inventory publication");
        return *cluster_inventory_;
    }

    // =========================================================================
    // Heterogeneous Device Detection
    // =========================================================================

    bool MPITopology::hasHeterogeneousGPUs() const
    {
        // Pure query of the snapshot installed at construction.
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
