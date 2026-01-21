/**
 * @file TPDomain.cpp
 * @brief Implementation of tensor parallel domain management
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "TPDomain.h"
#include "utils/NodeTopology.h"
#include "utils/Logger.h"
#include <sstream>
#include <algorithm>

namespace llaminar2
{

    // =============================================================================
    // TPDomainType to string conversion
    // =============================================================================

    const char *tpDomainTypeToString(TPDomainType type)
    {
        switch (type)
        {
        case TPDomainType::GPU_INTRA_RANK:
            return "GPU_INTRA_RANK";
        case TPDomainType::CPU_CROSS_RANK:
            return "CPU_CROSS_RANK";
        default:
            return "UNKNOWN";
        }
    }

    // =============================================================================
    // TPDomain Implementation
    // =============================================================================

    std::string TPDomain::toString() const
    {
        std::ostringstream oss;
        oss << "TPDomain{name=" << name
            << ", type=" << tpDomainTypeToString(type)
            << ", size=" << domain_size
            << ", local_rank=" << local_rank_in_domain
            << ", devices=[";

        for (size_t i = 0; i < devices.size(); ++i)
        {
            if (i > 0)
                oss << ", ";
            oss << devices[i].to_string();
        }
        oss << "]";

        if (communicator != MPI_COMM_NULL)
        {
            oss << ", has_comm=true";
        }
        oss << "}";

        return oss.str();
    }

    // =============================================================================
    // MultiDomainTPConfig Implementation
    // =============================================================================

    MultiDomainTPConfig::MultiDomainTPConfig(MultiDomainTPConfig &&other) noexcept
        : domains_(std::move(other.domains_)), attention_layer_domains_(std::move(other.attention_layer_domains_)), ffn_layer_domains_(std::move(other.ffn_layer_domains_)), owns_communicators_(other.owns_communicators_)
    {
        // Transfer ownership
        other.owns_communicators_ = false;
        other.gpu_domain_ = nullptr;
        other.cpu_domain_ = nullptr;

        // Update our pointers
        updateDomainPointers();
    }

    MultiDomainTPConfig &MultiDomainTPConfig::operator=(MultiDomainTPConfig &&other) noexcept
    {
        if (this != &other)
        {
            // Clean up our communicators first
            cleanup();

            // Move data
            domains_ = std::move(other.domains_);
            attention_layer_domains_ = std::move(other.attention_layer_domains_);
            ffn_layer_domains_ = std::move(other.ffn_layer_domains_);
            owns_communicators_ = other.owns_communicators_;

            // Transfer ownership
            other.owns_communicators_ = false;
            other.gpu_domain_ = nullptr;
            other.cpu_domain_ = nullptr;

            // Update our pointers
            updateDomainPointers();
        }
        return *this;
    }

    MultiDomainTPConfig::~MultiDomainTPConfig()
    {
        cleanup();
    }

    void MultiDomainTPConfig::updateDomainPointers()
    {
        gpu_domain_ = nullptr;
        cpu_domain_ = nullptr;

        for (auto &domain : domains_)
        {
            if (domain.type == TPDomainType::GPU_INTRA_RANK && !gpu_domain_)
            {
                gpu_domain_ = &domain;
            }
            else if (domain.type == TPDomainType::CPU_CROSS_RANK && !cpu_domain_)
            {
                cpu_domain_ = &domain;
            }
        }
    }

    TPDomain *MultiDomainTPConfig::findDomainByType(TPDomainType type)
    {
        for (auto &domain : domains_)
        {
            if (domain.type == type)
            {
                return &domain;
            }
        }
        return nullptr;
    }

    const TPDomain *MultiDomainTPConfig::findDomainByType(TPDomainType type) const
    {
        for (const auto &domain : domains_)
        {
            if (domain.type == type)
            {
                return &domain;
            }
        }
        return nullptr;
    }

    MultiDomainTPConfig MultiDomainTPConfig::create(
        const NodeTopology &topology,
        MPI_Comm mpi_comm,
        const std::vector<DeviceId> &local_devices)
    {
        MultiDomainTPConfig config;
        TPDomainBuilder builder(mpi_comm);

        // Separate GPUs and CPUs
        std::vector<DeviceId> gpus;
        std::vector<DeviceId> cpus;

        for (const auto &device : local_devices)
        {
            if (device.is_gpu())
            {
                gpus.push_back(device);
            }
            else if (device.is_cpu())
            {
                cpus.push_back(device);
            }
        }

        // Create GPU domain if we have GPUs (intra-rank only)
        if (!gpus.empty())
        {
            std::string gpu_name = "gpu_tp_rank" + std::to_string(builder.worldRank());
            auto gpu_domain = builder.createGPUIntraRankDomain(gpus, gpu_name);
            config.domains_.push_back(std::move(gpu_domain));
        }

        // Create CPU domain for cross-rank TP if multi-socket system
        // All ranks with CPUs participate
        if (topology.numSockets() > 1 || builder.worldSize() > 1)
        {
            // Use rank as ordering key for deterministic domain ordering
            int color = cpus.empty() ? MPI_UNDEFINED : 0; // All participating ranks get same color
            int key = builder.worldRank();

            auto cpu_domain = builder.createCPUCrossRankDomain(color, key, "cpu_tp_cross");

            // Add CPU device if not already in the domain devices
            if (!cpus.empty() && cpu_domain.isValid())
            {
                cpu_domain.devices = cpus;
                config.domains_.push_back(std::move(cpu_domain));
                config.owns_communicators_ = true;
            }
        }

        // Update cached pointers
        config.updateDomainPointers();

        LOG_DEBUG("MultiDomainTPConfig created: " << config.toString());

        return config;
    }

    MultiDomainTPConfig MultiDomainTPConfig::createForTest(const std::vector<TPDomain> &domains)
    {
        MultiDomainTPConfig config;
        config.domains_ = domains;
        config.owns_communicators_ = false; // Test domains don't own communicators
        config.updateDomainPointers();
        return config;
    }

    const TPDomain *MultiDomainTPConfig::domainForLayer(int layer_idx, bool is_attention) const
    {
        const auto &layer_map = is_attention ? attention_layer_domains_ : ffn_layer_domains_;

        auto it = layer_map.find(layer_idx);
        if (it != layer_map.end())
        {
            return findDomainByType(it->second);
        }

        // Default: attention uses GPU domain, FFN uses CPU domain (if available)
        if (is_attention)
        {
            return gpu_domain_;
        }
        else
        {
            // For FFN, prefer CPU domain if available, else GPU
            return cpu_domain_ ? cpu_domain_ : gpu_domain_;
        }
    }

    void MultiDomainTPConfig::setLayerDomainMapping(
        const std::unordered_map<int, TPDomainType> &attention_domains,
        const std::unordered_map<int, TPDomainType> &ffn_domains)
    {
        attention_layer_domains_ = attention_domains;
        ffn_layer_domains_ = ffn_domains;
    }

    bool MultiDomainTPConfig::hasCrossRankTP() const
    {
        for (const auto &domain : domains_)
        {
            if (domain.isCrossRank() && !domain.isTrivial())
            {
                return true;
            }
        }
        return false;
    }

    std::string MultiDomainTPConfig::toString() const
    {
        std::ostringstream oss;
        oss << "MultiDomainTPConfig{domains=" << domains_.size();

        if (gpu_domain_)
        {
            oss << ", gpu=" << gpu_domain_->name;
        }
        if (cpu_domain_)
        {
            oss << ", cpu=" << cpu_domain_->name;
        }

        oss << ", cross_rank=" << (hasCrossRankTP() ? "yes" : "no");
        oss << ", owns_comms=" << (owns_communicators_ ? "yes" : "no");

        if (!domains_.empty())
        {
            oss << ", [";
            for (size_t i = 0; i < domains_.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << domains_[i].toString();
            }
            oss << "]";
        }

        oss << "}";
        return oss.str();
    }

    void MultiDomainTPConfig::cleanup()
    {
        if (!owns_communicators_)
        {
            return;
        }

        for (auto &domain : domains_)
        {
            if (domain.communicator != MPI_COMM_NULL &&
                domain.communicator != MPI_COMM_WORLD &&
                domain.communicator != MPI_COMM_SELF)
            {
                // Check if MPI is still initialized before freeing
                int mpi_initialized = 0;
                int mpi_finalized = 0;
                MPI_Initialized(&mpi_initialized);
                MPI_Finalized(&mpi_finalized);

                if (mpi_initialized && !mpi_finalized)
                {
                    MPI_Comm_free(&domain.communicator);
                }
                domain.communicator = MPI_COMM_NULL;
            }
        }

        owns_communicators_ = false;
    }

    // =============================================================================
    // TPDomainBuilder Implementation
    // =============================================================================

    TPDomainBuilder::TPDomainBuilder(MPI_Comm base_comm)
        : base_comm_(base_comm), world_rank_(0), world_size_(1)
    {
        // Get rank and size from base communicator
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);

        if (mpi_initialized && base_comm != MPI_COMM_NULL)
        {
            MPI_Comm_rank(base_comm, &world_rank_);
            MPI_Comm_size(base_comm, &world_size_);
        }
    }

    TPDomain TPDomainBuilder::createGPUIntraRankDomain(
        const std::vector<DeviceId> &gpus,
        const std::string &name)
    {
        TPDomain domain;
        domain.type = TPDomainType::GPU_INTRA_RANK;
        domain.communicator = MPI_COMM_NULL; // No MPI comm needed - PCIeBAR direct
        domain.devices = gpus;
        domain.local_rank_in_domain = 0; // Intra-rank, so we're always rank 0
        domain.domain_size = static_cast<int>(gpus.size());
        domain.name = name;

        LOG_DEBUG("Created GPU intra-rank domain: " << domain.toString());

        return domain;
    }

    TPDomain TPDomainBuilder::createCPUCrossRankDomain(
        int color,
        int key,
        const std::string &name)
    {
        TPDomain domain;
        domain.type = TPDomainType::CPU_CROSS_RANK;
        domain.name = name;

        // Split communicator
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);

        if (!mpi_initialized || base_comm_ == MPI_COMM_NULL)
        {
            // No MPI - create trivial single-rank domain
            domain.communicator = MPI_COMM_NULL;
            domain.local_rank_in_domain = 0;
            domain.domain_size = 1;
            domain.devices.push_back(DeviceId::cpu());
            return domain;
        }

        MPI_Comm new_comm = MPI_COMM_NULL;
        int result = MPI_Comm_split(base_comm_, color, key, &new_comm);

        if (result != MPI_SUCCESS || new_comm == MPI_COMM_NULL)
        {
            // Non-participating rank or split failed
            domain.communicator = MPI_COMM_NULL;
            domain.local_rank_in_domain = 0;
            domain.domain_size = 0;
            LOG_DEBUG("Rank " << world_rank_ << " not participating in CPU domain '" << name << "'");
            return domain;
        }

        // Get our rank and size within the new communicator
        int domain_rank, domain_size;
        MPI_Comm_rank(new_comm, &domain_rank);
        MPI_Comm_size(new_comm, &domain_size);

        domain.communicator = new_comm;
        domain.local_rank_in_domain = domain_rank;
        domain.domain_size = domain_size;
        domain.devices.push_back(DeviceId::cpu());

        created_comms_.push_back(new_comm);

        LOG_DEBUG("Created CPU cross-rank domain: " << domain.toString());

        return domain;
    }

    MPI_Comm TPDomainBuilder::splitCommunicator(bool participating, const std::string &name)
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);

        if (!mpi_initialized || base_comm_ == MPI_COMM_NULL)
        {
            return MPI_COMM_NULL;
        }

        // Use unique color for each split, MPI_UNDEFINED for non-participants
        int color = participating ? split_counter_ : MPI_UNDEFINED;
        int key = world_rank_; // Ordering by world rank

        split_counter_++;

        MPI_Comm new_comm = MPI_COMM_NULL;
        int result = MPI_Comm_split(base_comm_, color, key, &new_comm);

        if (result != MPI_SUCCESS)
        {
            LOG_WARN("MPI_Comm_split failed for '" << name << "': " << result);
            return MPI_COMM_NULL;
        }

        if (new_comm != MPI_COMM_NULL)
        {
            created_comms_.push_back(new_comm);

            int new_rank, new_size;
            MPI_Comm_rank(new_comm, &new_rank);
            MPI_Comm_size(new_comm, &new_size);

            LOG_DEBUG("Split communicator '" << name << "': rank " << new_rank
                                             << "/" << new_size << " (world rank " << world_rank_ << ")");
        }

        return new_comm;
    }

} // namespace llaminar2
