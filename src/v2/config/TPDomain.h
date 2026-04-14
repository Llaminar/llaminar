/**
 * @file TPDomain.h
 * @brief Tensor parallel domain management for hybrid parallelism
 *
 * Supports two types of tensor parallel domains:
 * 1. GPU_INTRA_RANK: NVIDIA↔AMD GPUs on the same socket communicate via HOST backend
 * 2. CPU_CROSS_RANK: CPUs across sockets communicate via MPI over UPI (~50 GB/s)
 *
 * Design:
 * - GPU domains are INTRA-RANK only (no cross-rank GPU TP for latency reasons)
 * - CPU domains can be CROSS-RANK (using MPI collective operations)
 * - Each domain has its own MPI communicator for collective ops
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "backends/DeviceId.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace llaminar2
{

    // Forward declarations
    class NodeTopology;

    /**
     * Type of tensor parallel domain
     */
    enum class TPDomainType
    {
        GPU_INTRA_RANK, ///< HOST backend for NVIDIA↔AMD on same socket (intra-rank only)
        CPU_CROSS_RANK, ///< MPI over UPI for CPUs across sockets (cross-rank)
    };

    /**
     * Convert TPDomainType to human-readable string
     * @param type Domain type
     * @return String representation
     */
    const char *tpDomainTypeToString(TPDomainType type);

    /**
     * A tensor parallel domain groups devices that participate in collective operations.
     *
     * Key constraints:
     * - GPU domains are INTRA-RANK only (no cross-rank GPU TP)
     * - CPU domains can be CROSS-RANK (using MPI over UPI)
     * - Each domain has its own MPI communicator
     */
    struct TPDomain
    {
        TPDomainType type;
        MPI_Comm communicator;         ///< Domain-specific communicator (MPI_COMM_NULL if not applicable)
        std::vector<DeviceId> devices; ///< Devices participating in this domain
        int local_rank_in_domain;      ///< Our rank within this domain (0 to domain_size-1)
        int domain_size;               ///< Number of participants in this domain
        std::string name;              ///< Human-readable name (e.g., "gpu_tp_socket0", "cpu_tp_cross")

        /**
         * Default constructor - creates an invalid/empty domain
         */
        TPDomain()
            : type(TPDomainType::GPU_INTRA_RANK), communicator(MPI_COMM_NULL), local_rank_in_domain(0), domain_size(0), name("invalid")
        {
        }

        /**
         * Check if this domain requires cross-rank communication
         * @return true if CPU_CROSS_RANK type
         */
        bool isCrossRank() const { return type == TPDomainType::CPU_CROSS_RANK; }

        /**
         * Check if this is a trivial domain (size 1, no communication needed)
         * @return true if domain_size <= 1
         */
        bool isTrivial() const { return domain_size <= 1; }

        /**
         * Check if domain is valid (has devices and valid size)
         * @return true if domain has at least one device
         */
        bool isValid() const { return domain_size > 0 && !devices.empty(); }

        /**
         * Get string representation for logging
         * @return Human-readable description
         */
        std::string toString() const;
    };

    /**
     * Configuration for multi-domain tensor parallelism.
     *
     * Manages multiple TP domains that can operate concurrently:
     * - GPU domain for attention head parallelism (intra-rank)
     * - CPU domain for FFN parallelism (optional, cross-rank)
     */
    class MultiDomainTPConfig
    {
    public:
        /**
         * Create configuration with automatic domain detection
         * @param topology System topology for NUMA-aware domain creation
         * @param mpi_comm Base MPI communicator (typically MPI_COMM_WORLD)
         * @param local_devices Devices available on this rank
         * @return Configured MultiDomainTPConfig
         */
        static MultiDomainTPConfig create(
            const NodeTopology &topology,
            MPI_Comm mpi_comm,
            const std::vector<DeviceId> &local_devices);

        /**
         * Create configuration for testing (no MPI)
         * @param domains Pre-configured domains
         * @return MultiDomainTPConfig with given domains
         */
        static MultiDomainTPConfig createForTest(
            const std::vector<TPDomain> &domains);

        // Default constructor for empty config
        MultiDomainTPConfig() = default;

        // Move operations
        MultiDomainTPConfig(MultiDomainTPConfig &&other) noexcept;
        MultiDomainTPConfig &operator=(MultiDomainTPConfig &&other) noexcept;

        // No copy (owns MPI communicators)
        MultiDomainTPConfig(const MultiDomainTPConfig &) = delete;
        MultiDomainTPConfig &operator=(const MultiDomainTPConfig &) = delete;

        /**
         * Get all domains
         * @return Reference to vector of all domains
         */
        const std::vector<TPDomain> &domains() const { return domains_; }

        /**
         * Get GPU tensor parallel domain for this rank
         * @return Pointer to GPU domain, or nullptr if no GPU TP
         */
        const TPDomain *gpuDomain() const { return gpu_domain_; }

        /**
         * Get CPU tensor parallel domain
         * @return Pointer to CPU domain, or nullptr if no CPU TP
         */
        const TPDomain *cpuDomain() const { return cpu_domain_; }

        /**
         * Get domain for a specific layer
         * @param layer_idx Layer index
         * @param is_attention True for attention, false for FFN
         * @return Domain to use, or nullptr for no TP
         */
        const TPDomain *domainForLayer(int layer_idx, bool is_attention) const;

        /**
         * Set which domain handles which layers
         * @param attention_domains Map of layer_idx -> TPDomainType for attention
         * @param ffn_domains Map of layer_idx -> TPDomainType for FFN
         */
        void setLayerDomainMapping(
            const std::unordered_map<int, TPDomainType> &attention_domains,
            const std::unordered_map<int, TPDomainType> &ffn_domains);

        /**
         * Check if any cross-rank TP is configured
         * @return true if there's a non-trivial CPU_CROSS_RANK domain
         */
        bool hasCrossRankTP() const;

        /**
         * Get string representation for logging
         * @return Human-readable description
         */
        std::string toString() const;

        /**
         * Cleanup MPI communicators (call before MPI_Finalize)
         */
        void cleanup();

        /**
         * Destructor - automatically calls cleanup()
         */
        ~MultiDomainTPConfig();

    private:
        std::vector<TPDomain> domains_;
        TPDomain *gpu_domain_ = nullptr; ///< Points into domains_
        TPDomain *cpu_domain_ = nullptr; ///< Points into domains_

        // Layer -> domain mapping
        std::unordered_map<int, TPDomainType> attention_layer_domains_;
        std::unordered_map<int, TPDomainType> ffn_layer_domains_;

        bool owns_communicators_ = false; ///< True if we created comms and need to free them

        // Update pointers after domains_ modification
        void updateDomainPointers();

        // Find domain by type
        TPDomain *findDomainByType(TPDomainType type);
        const TPDomain *findDomainByType(TPDomainType type) const;
    };

    /**
     * Builder for creating TP domains with MPI communicator splitting.
     *
     * Usage:
     *   TPDomainBuilder builder(MPI_COMM_WORLD);
     *   auto gpu_domain = builder.createGPUIntraRankDomain(gpus, "gpu_tp");
     *   auto cpu_domain = builder.createCPUCrossRankDomain(color, key, "cpu_tp");
     */
    class TPDomainBuilder
    {
    public:
        /**
         * Create builder with base MPI communicator
         * @param base_comm Base communicator (typically MPI_COMM_WORLD)
         */
        explicit TPDomainBuilder(MPI_Comm base_comm);

        /**
         * Destructor - does NOT free communicators (caller owns them)
         */
        ~TPDomainBuilder() = default;

        /**
         * Create a GPU intra-rank domain (no MPI communicator needed - uses HOST backend)
         * @param gpus GPU devices in this domain
         * @param name Domain name for logging
         * @return Configured TPDomain
         */
        TPDomain createGPUIntraRankDomain(
            const std::vector<DeviceId> &gpus,
            const std::string &name);

        /**
         * Create a CPU cross-rank domain with MPI_Comm_split
         * @param color Ranks with same color join same domain
         * @param key Ordering key within domain
         * @param name Domain name for logging
         * @return Configured TPDomain with its own communicator
         */
        TPDomain createCPUCrossRankDomain(
            int color,
            int key,
            const std::string &name);

        /**
         * Create domain communicator using MPI_Comm_split
         * @param participating True if this rank participates
         * @param name Domain name for logging
         * @return MPI communicator (MPI_COMM_NULL if not participating)
         */
        MPI_Comm splitCommunicator(bool participating, const std::string &name);

        /**
         * Get world rank in base communicator
         * @return This process's rank
         */
        int worldRank() const { return world_rank_; }

        /**
         * Get world size of base communicator
         * @return Total number of ranks
         */
        int worldSize() const { return world_size_; }

        /**
         * Get list of created communicators (for cleanup tracking)
         * @return Reference to vector of created communicators
         */
        const std::vector<MPI_Comm> &createdCommunicators() const { return created_comms_; }

    private:
        MPI_Comm base_comm_;
        int world_rank_;
        int world_size_;
        std::vector<MPI_Comm> created_comms_; ///< Track for cleanup
        int split_counter_ = 0;               ///< Unique color base for splits
    };

} // namespace llaminar2
