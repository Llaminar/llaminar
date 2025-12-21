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
#include "../execution/PlacementStrategy.h"

#include <sstream>
#include <algorithm>
#include <numeric>
#include <unistd.h> // gethostname

namespace llaminar2
{

    // =========================================================================
    // WorkRange Static Methods
    // =========================================================================

    WorkRange WorkRange::for_rank_equal(size_t total, int rank, int world_size)
    {
        if (world_size <= 0 || rank < 0 || rank >= world_size)
        {
            return {0, 0, 0}; // Invalid
        }

        size_t per_rank = total / world_size;
        size_t remainder = total % world_size;

        // Distribute remainder among first 'remainder' ranks
        size_t start = rank * per_rank + std::min(static_cast<size_t>(rank), remainder);
        size_t end = start + per_rank + (static_cast<size_t>(rank) < remainder ? 1 : 0);

        return {start, end, end - start};
    }

    WorkRange WorkRange::for_rank_weighted(size_t total, int rank, int world_size,
                                           const std::vector<float> &weights)
    {
        if (world_size <= 0 || rank < 0 || rank >= world_size)
        {
            return {0, 0, 0}; // Invalid
        }

        if (weights.size() != static_cast<size_t>(world_size))
        {
            // Fall back to equal distribution if weights don't match
            return for_rank_equal(total, rank, world_size);
        }

        // Normalize weights
        float total_weight = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (total_weight <= 0.0f)
        {
            return for_rank_equal(total, rank, world_size);
        }

        // Calculate this rank's portion
        float cumulative = 0.0f;
        for (int r = 0; r < rank; ++r)
        {
            cumulative += weights[r];
        }

        size_t start = static_cast<size_t>((cumulative / total_weight) * total);
        cumulative += weights[rank];
        size_t end = static_cast<size_t>((cumulative / total_weight) * total);

        // Ensure last rank gets everything remaining
        if (rank == world_size - 1)
        {
            end = total;
        }

        return {start, end, end - start};
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

        // Always add CPU as a device
        DeviceCapability cpu_dev;
        cpu_dev.type = DeviceCapability::Type::CPU;
        cpu_dev.device_id = 0;
        cpu_dev.relative_compute = 1.0f; // Baseline
        cpu_dev.memory_bytes = 0;        // TODO: Detect system memory
        cpu_dev.name = "CPU (socket " + std::to_string(placement_.socket_id) + ")";
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
        // For now, use simple all-to-all exchange
        // Each rank broadcasts its placement to all others

        all_placements_.resize(world_size_);
        all_placements_[rank_] = placement_;

        // Serialize minimal info for exchange (hostname, device count, compute power)
        // Full exchange would require custom MPI datatype or serialization
        // For now, just exchange compute weights

        std::vector<float> local_compute(1);
        local_compute[0] = 0.0f;
        for (const auto &dev : placement_.devices)
        {
            local_compute[0] += dev.relative_compute;
        }

        std::vector<float> all_compute(world_size_);
        MPI_Allgather(local_compute.data(), 1, MPI_FLOAT,
                      all_compute.data(), 1, MPI_FLOAT,
                      world_comm_);

        // Store compute weights in placements
        for (int r = 0; r < world_size_; ++r)
        {
            if (r != rank_)
            {
                // Create placeholder placement for remote ranks
                RankPlacement remote;
                remote.rank = r;
                remote.node_id = r / ranks_per_node_;
                remote.local_rank = r % ranks_per_node_;
                remote.socket_id = remote.local_rank;
                remote.numa_node = remote.local_rank;
                remote.hostname = "remote";

                // Add placeholder device with gathered compute power
                DeviceCapability placeholder;
                placeholder.type = DeviceCapability::Type::CPU;
                placeholder.device_id = 0;
                placeholder.relative_compute = all_compute[r];
                placeholder.memory_bytes = 0;
                placeholder.name = "remote_compute";
                remote.devices.push_back(placeholder);

                all_placements_[r] = remote;
            }
        }

        LOG_TRACE("[MPITopology] Capability exchange complete, total compute weights:");
        for (int r = 0; r < world_size_; ++r)
        {
            LOG_TRACE("  Rank " << r << ": " << all_compute[r]);
        }
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
                return {static_cast<size_t>(rank_), static_cast<size_t>(rank_ + 1), 1};
            }
            else
            {
                return {0, 0, 0}; // Empty range
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

        return computePlacement(std::move(input));
    }

    PlacementPlan MPITopology::computePlacement(PlacementInput input) const
    {
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

} // namespace llaminar2
