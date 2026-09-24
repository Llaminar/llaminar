/**
 * @file ClusterInventoryGatherer.cpp
 * @brief One hardware projection and MPI exchange for planning and execution.
 *
 * DeviceManager owns physical discovery. This module preserves those facts,
 * attaches actual rank/NUMA membership, and exchanges them collectively.
 * Configuration is deliberately absent: selection consumes inventory, never
 * rewrites it. Local errors are published before payload collectives so peers
 * cannot proceed with an invented or partially empty record.
 */
#include "planning/ClusterInventoryGatherer.h"
#include "planning/ExecutionRankMembership.h"
#include "utils/MPITopology.h"
#include "utils/NodeDetection.h"
#include "utils/NUMATopology.h"
#include "utils/MPIContext.h"
#include <algorithm>
#include <array>
#include <climits>
#include <set>
#include <stdexcept>
#include <omp.h>

namespace llaminar2
{
    namespace
    {
        /** @brief Translate a discovered backend without inventing CPU devices. */
        DeviceType deviceType(ComputeBackendType backend)
        {
            switch (backend)
            {
            case ComputeBackendType::CPU: return DeviceType::CPU;
            case ComputeBackendType::GPU_CUDA: return DeviceType::CUDA;
            case ComputeBackendType::GPU_ROCM: return DeviceType::ROCm;
            case ComputeBackendType::GPU_VULKAN: return DeviceType::Vulkan;
            case ComputeBackendType::GPU_METAL: return DeviceType::Metal;
            }
            throw std::invalid_argument("Unknown backend in hardware observation");
        }

        /** @brief Reject MPI setup errors at the operation which produced them. */
        void requireMPI(int status, const char *operation)
        {
            if (status != MPI_SUCCESS)
                throw std::runtime_error(std::string("Cluster inventory exchange failed: ") + operation);
        }

        /**
         * @brief Publish a local setup failure before any variable-size exchange.
         * @param comm All participants in the inventory transaction.
         * @param size Number of participants.
         * @param error Local diagnostic, empty on success.
         *
         * Fixed-size diagnostics are initialization-only. All observers throw
         * the same first-rank failure; none create a placeholder inventory.
         */
        void publishDiscoveryStatus(MPI_Comm comm, int size, const std::string &error)
        {
            std::array<char, 1024> local{};
            std::copy_n(error.data(), std::min(error.size(), local.size() - 1), local.data());
            std::vector<char> errors(static_cast<std::size_t>(size) * local.size());
            requireMPI(MPI_Allgather(local.data(), local.size(), MPI_CHAR,
                                    errors.data(), local.size(), MPI_CHAR, comm), "discovery status");
            for (int rank = 0; rank < size; ++rank)
                if (errors[static_cast<std::size_t>(rank) * local.size()] != '\0')
                    throw std::runtime_error("Cluster inventory rank " + std::to_string(rank) +
                        ": " + (errors.data() + static_cast<std::size_t>(rank) * local.size()));
        }

        /**
         * @brief Reindex measured P2P edges to the rank's visible device order.
         * @param rank Record whose GPU list is already complete.
         * @param backend Backend of the matrix.
         * @param observed Optional real driver observation.
         *
         * Backend ordinals can be sparse or filtered; matrix index is never
         * mistaken for a device ID. An absent observation remains absent.
         */
        void projectPeers(RankInventory &rank, DeviceType backend,
                          const std::optional<P2PMatrix> &observed)
        {
            if (!observed) return;
            const auto &source = *observed;
            if (source.can_access.size() != source.device_ids.size())
                throw std::invalid_argument("Malformed observed P2P matrix");
            std::set<int> unique;
            for (std::size_t row = 0; row < source.device_ids.size(); ++row)
                if (source.device_ids[row] < 0 || !unique.insert(source.device_ids[row]).second ||
                    source.can_access[row].size() != source.device_ids.size())
                    throw std::invalid_argument("Malformed observed P2P row or ordinal");
            std::vector<int> indices;
            for (const auto &gpu : rank.gpus)
                if (gpu.type == backend)
                {
                    const auto index = source.indexForDevice(gpu.local_device_id);
                    if (!index) throw std::invalid_argument("Visible GPU missing from observed P2P matrix");
                    indices.push_back(*index);
                }
            auto &count = backend == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
            auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
            count = static_cast<int>(indices.size());
            for (int from : indices)
                for (int to : indices)
                    matrix.push_back(source.can_access[from][to]);
        }
    }

    RankInventory makeRankInventory(const HardwareInventory &hardware,
                                    const RankHardwareLocation &location)
    {
        // This is a startup-only projection of the already owned observation,
        // not another driver query or allocation ledger. CPU binding must not
        // remove a GPU on a different socket from automatic planning.
        std::vector<ComputeDevice> accelerators;
        accelerators.reserve(hardware.cuda_devices.size() + hardware.rocm_devices.size());
        accelerators.insert(accelerators.end(), hardware.cuda_devices.begin(), hardware.cuda_devices.end());
        accelerators.insert(accelerators.end(), hardware.rocm_devices.begin(), hardware.rocm_devices.end());
        return makeRankInventory(hardware, accelerators, location);
    }

    RankInventory makeRankInventory(const HardwareInventory &hardware,
                                    std::span<const ComputeDevice> visible_devices,
                                    const RankHardwareLocation &location)
    {
        if (location.rank < 0 || location.node < 0 || location.local_rank < 0 ||
            hardware.cpu_sockets.empty() || location.cpu_worker_threads < 0)
            throw std::invalid_argument("Incomplete rank hardware identity or CPU observation");
        RankInventory rank;
        rank.rank = location.rank;
        rank.node_id = location.node;
        rank.local_rank = location.local_rank;
        rank.hostname = location.hostname;
        rank.cpu.type = DeviceType::CPU;
        rank.cpu.numa_node = location.cpu_numa_node.value_or(-1);
        rank.cpu_sockets = static_cast<int>(hardware.cpu_sockets.size());
        rank.cpu_socket_info = hardware.cpu_sockets;
        rank.cpu_execution = hardware.cpu_execution;
        rank.cpu_worker_threads = location.cpu_worker_threads;
        const auto cpu = hardware.cpuDevice(location.cpu_numa_node.value_or(-1));
        rank.cpu.memory_bytes = cpu.total_memory_bytes;
        rank.cpu.free_memory_bytes = cpu.free_memory_bytes;
        rank.cpu.compute_units = cpu.compute_units;
        rank.cpu.last_level_cache_bytes = cpu.last_level_cache_bytes;
        rank.cpu.name = cpu.name;

        // CPU ownership follows actual NUMA binding, not local MPI rank.
        // Whole-host ownership is explicit: socket zero must not silently cap
        // a single-process CPU plan or be counted again as a second socket.
        std::set<int> nodes;
        for (const auto &socket : hardware.cpu_sockets)
        {
            nodes.insert(socket.numa_node);
            if (location.cpu_numa_node && socket.numa_node != *location.cpu_numa_node)
                continue;
            rank.cpu_cores += socket.num_physical_cores();
        }
        rank.numa_nodes = static_cast<int>(nodes.size());
        rank.cpu_memory_bytes = rank.cpu.memory_bytes;

        std::set<std::pair<DeviceType, int>> ordinals;
        for (const auto &device : visible_devices)
        {
            if (device.type == ComputeBackendType::CPU) continue;
            const auto type = deviceType(device.type);
            if (device.device_id < 0 || !ordinals.emplace(type, device.device_id).second ||
                device.total_memory_bytes == 0 || device.free_memory_bytes > device.total_memory_bytes)
                throw std::invalid_argument("Invalid GPU ordinal or capacity in hardware observation");
            DeviceInfo gpu;
            gpu.type = type;
            gpu.local_device_id = device.device_id;
            gpu.memory_bytes = device.total_memory_bytes;
            gpu.free_memory_bytes = device.free_memory_bytes;
            gpu.compute_units = device.compute_units;
            gpu.last_level_cache_bytes = device.last_level_cache_bytes;
            gpu.name = device.name;
            gpu.uuid = device.uuid;
            gpu.numa_node = device.numa_node;
            gpu.compute_capability_major = device.compute_capability / 10;
            gpu.compute_capability_minor = device.compute_capability % 10;
            gpu.pcie_gen = device.pcie.pcie_gen;
            gpu.pcie_width = device.pcie.link_width;
            gpu.pcie_speed_gts = device.pcie.link_speed_gts;
            gpu.pcie_max_width = device.pcie.max_width;
            gpu.pcie_max_speed_gts = device.pcie.max_speed_gts;
            gpu.pcie_degraded = device.pcie.degraded;
            gpu.pcie_bottleneck_bdf = device.pcie.bottleneck_bdf;
            rank.gpus.push_back(std::move(gpu));
        }
        projectPeers(rank, DeviceType::CUDA, hardware.cuda_p2p);
        projectPeers(rank, DeviceType::ROCm, hardware.rocm_p2p);
        return rank;
    }

    namespace
    {
    /**
     * @brief Execute the sole observation transaction, before publishing it.
     * @param mpi_ctx Exact context identity; null is explicit local-only discovery.
     * @return Complete observed records. No placement or allocation is performed.
     *
     * This private implementation is entered by the context's once-only owner,
     * never by topology readers or by a second public refresh API.
     */
    ClusterInventory discoverClusterInventory(const IMPIContext *mpi_ctx)
    {
        const int declared_size = mpi_ctx ? mpi_ctx->world_size() : 1;
        const int declared_rank = mpi_ctx ? mpi_ctx->rank() : 0;
        int size = declared_size;
        int rank = declared_rank;
        const MPI_Comm comm = mpi_ctx ? mpi_ctx->communicator() : MPI_COMM_NULL;
        if (mpi_ctx)
        {
            if (comm == MPI_COMM_NULL)
                throw std::invalid_argument("Cluster inventory requires a valid communicator identity");
            int initialized = 0, finalized = 0;
            requireMPI(MPI_Initialized(&initialized), "MPI initialization query");
            if (initialized) requireMPI(MPI_Finalized(&finalized), "MPI retirement query");
            if (!initialized || finalized)
                throw std::logic_error("Cluster inventory requires a live MPI session");
            // Actual membership sizes all protocol buffers. A malformed wrapper
            // cannot turn a multi-rank transaction into a local observation or
            // allocate a receive buffer smaller than its communicator.
            requireMPI(MPI_Comm_size(comm, &size), "communicator size");
            requireMPI(MPI_Comm_rank(comm, &rank), "communicator rank");
        }
        const bool distributed = size > 1;
        if (size < 1 || rank < 0 || rank >= size || (distributed && comm == MPI_COMM_NULL))
            throw std::invalid_argument("Cluster inventory requires a valid communicator identity");

        NodeDetectionResult nodes;
        RankHardwareLocation location;
        location.rank = rank;
        if (distributed)
        {
            nodes = NodeDetection::detect(comm);
            location.node = nodes.node_ids.at(rank);
            location.hostname = nodes.hostnames.at(rank);
            location.local_rank = static_cast<int>(std::count(
                nodes.node_ids.begin(), nodes.node_ids.begin() + rank, location.node));
        }

        RankInventory local;
        std::vector<uint8_t> local_data;
        std::string error;
        try
        {
            if (declared_size != size || declared_rank != rank)
                throw std::invalid_argument("MPI context identity disagrees with its communicator");
            // Capture the configured execution team independently of physical
            // cores. --threads and per-host bootstrap can legitimately choose
            // a different budget on each rank; root must not reconstruct it.
            if (omp_in_parallel() || omp_get_dynamic() || omp_get_max_threads() <= 0 ||
                omp_get_thread_limit() < omp_get_max_threads())
                throw std::invalid_argument("Inventory requires a fixed, non-nested CPU execution team");
            location.cpu_worker_threads = omp_get_max_threads();
            if (distributed)
            {
                const auto numa = NUMATopology::detectLocalNUMANode();
                if (!numa.detection_succeeded || numa.local_numa_node < 0)
                    throw std::runtime_error("Multi-rank hardware discovery requires exact CPU NUMA affinity");
                location.cpu_numa_node = numa.local_numa_node;
            }
            auto &manager = DeviceManager::instance();
            // Reuse the complete observation even if another startup caller
            // installed a socket-local execution view. Visibility projection
            // cannot require re-enumeration or mutate that caller's view.
            if (!manager.hardware())
                manager.initialize(location.cpu_numa_node.value_or(-1), false);
            const auto *hardware = manager.hardware();
            if (!hardware) throw std::runtime_error("DeviceManager did not publish its hardware observation");
            local = makeRankInventory(*hardware, location);
            local_data = MPITopology::serializeRankInventory(local);
            if (local_data.size() > INT_MAX)
                throw std::runtime_error("Rank inventory exceeds MPI payload capacity");
        }
        catch (const std::exception &failure) { error = failure.what(); }
        if (distributed) publishDiscoveryStatus(comm, size, error);
        else if (!error.empty()) throw std::runtime_error(error);

        ClusterInventory inventory;
        inventory.world_size = size;
        inventory.node_count = distributed ? nodes.node_count : 1;
        if (!distributed)
        {
            inventory.ranks.push_back(std::move(local));
            inventory.buildNodeAggregations();
            return inventory;
        }

        const int local_size = static_cast<int>(local_data.size());
        std::vector<int> sizes(size), offsets(size);
        requireMPI(MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm), "payload sizes");
        int total = 0;
        for (int peer = 0; peer < size; ++peer)
        {
            if (sizes[peer] <= 0 || sizes[peer] > INT_MAX - total)
                throw std::runtime_error("Invalid cluster inventory payload extent");
            offsets[peer] = total;
            total += sizes[peer];
        }
        std::vector<uint8_t> payload(static_cast<std::size_t>(total));
        requireMPI(MPI_Allgatherv(local_data.data(), local_size, MPI_BYTE,
                                 payload.data(), sizes.data(), offsets.data(), MPI_BYTE, comm), "rank payloads");
        for (int peer = 0; peer < size; ++peer)
        {
            // Every observer validates identical bytes. A malformed record is
            // fatal, never an empty rank or zero-byte invented GPU.
            auto record = MPITopology::deserializeRankInventory(payload.data() + offsets[peer], sizes[peer]);
            if (record.rank != peer || record.node_id != nodes.node_ids[peer] ||
                record.hostname != nodes.hostnames[peer])
                throw std::runtime_error("Inventory payload identity disagrees with MPI membership");
            inventory.ranks.push_back(std::move(record));
        }
        inventory.buildNodeAggregations();
        return inventory;
    }
    } // namespace

    std::shared_ptr<const ClusterInventory> MPIContext::clusterInventory() const
    {
        // The alias retains the immutable membership owner without copying its
        // projection or entering discovery on an already selected communicator.
        if (membership_)
            return std::shared_ptr<const ClusterInventory>(membership_, &membership_->inventory());
        std::call_once(inventory_once_, [this] {
            try
            {
                inventory_publication_ = std::make_shared<const ClusterInventory>(
                    discoverClusterInventory(this));
            }
            catch (...) { inventory_publication_ = std::current_exception(); }
        });
        if (const auto *failure = std::get_if<std::exception_ptr>(&inventory_publication_))
            std::rethrow_exception(*failure);
        return std::get<std::shared_ptr<const ClusterInventory>>(inventory_publication_);
    }

    std::shared_ptr<const ClusterInventory> gatherClusterInventory(const std::shared_ptr<IMPIContext> &mpi_ctx,
                                            const std::string & /*hostfile*/)
    {
        // Hostfile is launch provenance, not a second source of MPI membership.
        // Context-bound readers all consume the very same immutable observation.
        return mpi_ctx ? mpi_ctx->clusterInventory() :
            std::make_shared<const ClusterInventory>(discoverClusterInventory(nullptr));
    }
} // namespace llaminar2
