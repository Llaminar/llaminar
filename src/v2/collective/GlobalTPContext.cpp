/**
 * @file GlobalTPContext.cpp
 * @brief Implementation of GLOBAL tensor parallelism context
 *
 * Provides concrete implementation of IGlobalTPContext for managing
 * tensor parallelism across multiple MPI ranks using UPI/MPI.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "GlobalTPContext.h"
#include "DeviceGroup.h"
#include "../config/TPDomain.h"
#include "../tensors/Tensors.h"
#include "../utils/Logger.h"
#include <mpi.h>
#include <utility>

namespace llaminar2
{

    // =============================================================================
    // Private Constructor
    // =============================================================================

    GlobalTPContext::GlobalTPContext(
        MPI_Comm domain_comm,
        int domain_id,
        int my_rank_in_domain,
        int domain_size,
        std::vector<int> world_ranks,
        bool owns_communicator,
        CollectiveBackendType backend_type)
        : domain_comm_(domain_comm), domain_id_(domain_id), my_rank_in_domain_(my_rank_in_domain), domain_size_(domain_size), world_ranks_(std::move(world_ranks)), owns_communicator_(owns_communicator), backend_type_(backend_type), backend_(nullptr)
    {
        // Create UPI backend for collective operations
        if (domain_comm_ != MPI_COMM_NULL)
        {
            backend_ = std::make_unique<UPICollectiveBackend>(domain_comm_, nullptr);
            
            // Initialize the backend with a minimal device group for CPU TP
            DeviceGroup group;
            group.name = "global_tp_domain_" + std::to_string(domain_id);
            group.scope = CollectiveScope::LOCAL; // Cross-socket on same node
            for (int i = 0; i < domain_size_; ++i) {
                group.devices.push_back(DeviceId::cpu());
            }
            backend_->initialize(group);

            LOG_DEBUG("GlobalTPContext: Created for domain " << domain_id_
                                                             << " with rank " << my_rank_in_domain_
                                                             << "/" << domain_size_
                                                             << " (owns_comm=" << owns_communicator_ << ")");
        }
    }

    // =============================================================================
    // Factory Methods
    // =============================================================================

    std::unique_ptr<GlobalTPContext> GlobalTPContext::create(const TPDomain &domain)
    {
        if (domain.communicator == MPI_COMM_NULL)
        {
            LOG_ERROR("GlobalTPContext::create - TPDomain has null communicator");
            return nullptr;
        }

        if (!domain.isValid())
        {
            LOG_ERROR("GlobalTPContext::create - TPDomain is invalid (size="
                      << domain.domain_size << ", devices=" << domain.devices.size() << ")");
            return nullptr;
        }

        // Get our position in the domain communicator
        int my_rank, domain_size;
        MPI_Comm_rank(domain.communicator, &my_rank);
        MPI_Comm_size(domain.communicator, &domain_size);

        // Verify domain_size matches TPDomain
        if (domain_size != domain.domain_size)
        {
            LOG_WARN("GlobalTPContext::create - Communicator size (" << domain_size
                                                                     << ") differs from TPDomain size (" << domain.domain_size << ")");
        }

        // Build world_ranks by gathering world ranks from all domain participants
        // First, get our world rank
        int my_world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);

        // Gather all world ranks
        std::vector<int> world_ranks(domain_size);
        MPI_Allgather(&my_world_rank, 1, MPI_INT,
                      world_ranks.data(), 1, MPI_INT,
                      domain.communicator);

        LOG_DEBUG("GlobalTPContext::create - Domain " << domain.name
                                                      << ": gathered world_ranks from " << domain_size << " participants");

        // Create context - TPDomain owns the communicator, we don't
        return std::unique_ptr<GlobalTPContext>(new GlobalTPContext(
            domain.communicator,
            0, // domain_id (use 0 for domains from TPDomain)
            my_rank,
            domain_size,
            std::move(world_ranks),
            false // owns_communicator = false (TPDomain owns it)
            ));
    }

    std::unique_ptr<GlobalTPContext> GlobalTPContext::createWithSplit(
        MPI_Comm base_comm,
        int domain_id,
        int color,
        int key)
    {
        if (base_comm == MPI_COMM_NULL)
        {
            LOG_ERROR("GlobalTPContext::createWithSplit - base_comm is null");
            return nullptr;
        }

        // Create new communicator via MPI_Comm_split
        MPI_Comm new_comm;
        int err = MPI_Comm_split(base_comm, color, key, &new_comm);
        if (err != MPI_SUCCESS)
        {
            LOG_ERROR("GlobalTPContext::createWithSplit - MPI_Comm_split failed with error " << err);
            return nullptr;
        }

        // Get our position in the new domain communicator
        int my_rank, domain_size;
        MPI_Comm_rank(new_comm, &my_rank);
        MPI_Comm_size(new_comm, &domain_size);

        // Get our world rank from the base communicator
        int my_base_rank;
        MPI_Comm_rank(base_comm, &my_base_rank);

        // Gather all base_comm ranks that are in this domain
        // This gives us the "world ranks" relative to base_comm
        std::vector<int> world_ranks(domain_size);
        MPI_Allgather(&my_base_rank, 1, MPI_INT,
                      world_ranks.data(), 1, MPI_INT,
                      new_comm);

        LOG_DEBUG("GlobalTPContext::createWithSplit - Created domain " << domain_id
                                                                       << " with color=" << color << ", key=" << key
                                                                       << ", size=" << domain_size);

        // Create context - we own the communicator since we created it
        return std::unique_ptr<GlobalTPContext>(new GlobalTPContext(
            new_comm,
            domain_id,
            my_rank,
            domain_size,
            std::move(world_ranks),
            true // owns_communicator = true (we created it)
            ));
    }

    std::unique_ptr<GlobalTPContext> GlobalTPContext::createForTest(
        MPI_Comm comm,
        int domain_id,
        std::vector<int> world_ranks)
    {
        if (comm == MPI_COMM_NULL)
        {
            LOG_ERROR("GlobalTPContext::createForTest - communicator is null");
            return nullptr;
        }

        // Get rank and size from the communicator
        int my_rank, domain_size;
        MPI_Comm_rank(comm, &my_rank);
        MPI_Comm_size(comm, &domain_size);

        // Validate world_ranks size
        if (static_cast<int>(world_ranks.size()) != domain_size)
        {
            LOG_WARN("GlobalTPContext::createForTest - world_ranks size (" << world_ranks.size()
                                                                           << ") differs from comm size (" << domain_size << ")");
        }

        LOG_DEBUG("GlobalTPContext::createForTest - Created test context for domain " << domain_id
                                                                                      << " with size=" << domain_size);

        // Create context - test controls lifetime, we don't own it
        return std::unique_ptr<GlobalTPContext>(new GlobalTPContext(
            comm,
            domain_id,
            my_rank,
            domain_size,
            std::move(world_ranks),
            false // owns_communicator = false (test owns it)
            ));
    }

    // =============================================================================
    // Destructor
    // =============================================================================

    GlobalTPContext::~GlobalTPContext()
    {
        // Free the communicator if we own it
        if (owns_communicator_ && domain_comm_ != MPI_COMM_NULL)
        {
            LOG_DEBUG("GlobalTPContext: Freeing owned communicator for domain " << domain_id_);
            MPI_Comm_free(&domain_comm_);
            domain_comm_ = MPI_COMM_NULL;
        }
    }

    // =============================================================================
    // Move Operations
    // =============================================================================

    GlobalTPContext::GlobalTPContext(GlobalTPContext &&other) noexcept
        : domain_comm_(other.domain_comm_), domain_id_(other.domain_id_), my_rank_in_domain_(other.my_rank_in_domain_), domain_size_(other.domain_size_), world_ranks_(std::move(other.world_ranks_)), owns_communicator_(other.owns_communicator_), backend_(std::move(other.backend_))
    {
        // Clear source to prevent double-free
        other.domain_comm_ = MPI_COMM_NULL;
        other.owns_communicator_ = false;
        other.domain_id_ = -1;
        other.my_rank_in_domain_ = -1;
        other.domain_size_ = 0;
    }

    GlobalTPContext &GlobalTPContext::operator=(GlobalTPContext &&other) noexcept
    {
        if (this != &other)
        {
            // Free our communicator if we own it
            if (owns_communicator_ && domain_comm_ != MPI_COMM_NULL)
            {
                MPI_Comm_free(&domain_comm_);
            }

            // Transfer ownership
            domain_comm_ = other.domain_comm_;
            domain_id_ = other.domain_id_;
            my_rank_in_domain_ = other.my_rank_in_domain_;
            domain_size_ = other.domain_size_;
            world_ranks_ = std::move(other.world_ranks_);
            owns_communicator_ = other.owns_communicator_;
            backend_ = std::move(other.backend_);

            // Clear source to prevent double-free
            other.domain_comm_ = MPI_COMM_NULL;
            other.owns_communicator_ = false;
            other.domain_id_ = -1;
            other.my_rank_in_domain_ = -1;
            other.domain_size_ = 0;
        }
        return *this;
    }

    // =============================================================================
    // ITPContext Implementation
    // =============================================================================

    int GlobalTPContext::degree() const
    {
        return domain_size_;
    }

    int GlobalTPContext::myIndex() const
    {
        return my_rank_in_domain_;
    }

    bool GlobalTPContext::allreduce(TensorBase *tensor)
    {
        if (!tensor)
        {
            LOG_ERROR("GlobalTPContext::allreduce - tensor is null");
            return false;
        }

        if (!backend_)
        {
            LOG_ERROR("GlobalTPContext::allreduce - backend not initialized");
            return false;
        }

        // Get mutable data pointer and element count
        float *data = tensor->mutable_data();
        if (!data)
        {
            LOG_ERROR("GlobalTPContext::allreduce - tensor has no mutable data (may be read-only or non-FP32)");
            return false;
        }

        size_t count = tensor->numel();
        if (count == 0)
        {
            LOG_WARN("GlobalTPContext::allreduce - tensor has zero elements");
            return true; // Nothing to reduce
        }

        // Delegate to UPI backend
        return backend_->allreduce(data, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
    }

    bool GlobalTPContext::broadcast(TensorBase *tensor, int source_index)
    {
        if (!tensor)
        {
            LOG_ERROR("GlobalTPContext::broadcast - tensor is null");
            return false;
        }

        if (!backend_)
        {
            LOG_ERROR("GlobalTPContext::broadcast - backend not initialized");
            return false;
        }

        if (source_index < 0 || source_index >= domain_size_)
        {
            LOG_ERROR("GlobalTPContext::broadcast - invalid source_index " << source_index
                                                                           << " (domain_size=" << domain_size_ << ")");
            return false;
        }

        // Get mutable data pointer and element count
        float *data = tensor->mutable_data();
        if (!data)
        {
            LOG_ERROR("GlobalTPContext::broadcast - tensor has no mutable data");
            return false;
        }

        size_t count = tensor->numel();
        if (count == 0)
        {
            LOG_WARN("GlobalTPContext::broadcast - tensor has zero elements");
            return true; // Nothing to broadcast
        }

        // Delegate to UPI backend
        return backend_->broadcast(data, count, CollectiveDataType::FLOAT32, source_index);
    }

    bool GlobalTPContext::allgather(const TensorBase *local_shard, TensorBase *global_tensor)
    {
        if (!local_shard)
        {
            LOG_ERROR("GlobalTPContext::allgather - local_shard is null");
            return false;
        }

        if (!global_tensor)
        {
            LOG_ERROR("GlobalTPContext::allgather - global_tensor is null");
            return false;
        }

        if (!backend_)
        {
            LOG_ERROR("GlobalTPContext::allgather - backend not initialized");
            return false;
        }

        // Get const data pointer from local shard
        const float *send_data = local_shard->data();
        if (!send_data)
        {
            LOG_ERROR("GlobalTPContext::allgather - local_shard has no data");
            return false;
        }

        // Get mutable data pointer for global tensor
        float *recv_data = global_tensor->mutable_data();
        if (!recv_data)
        {
            LOG_ERROR("GlobalTPContext::allgather - global_tensor has no mutable data");
            return false;
        }

        size_t local_count = local_shard->numel();
        size_t expected_global = local_count * domain_size_;
        size_t actual_global = global_tensor->numel();

        if (actual_global != expected_global)
        {
            LOG_ERROR("GlobalTPContext::allgather - size mismatch: local=" << local_count
                                                                           << ", expected_global=" << expected_global
                                                                           << ", actual_global=" << actual_global);
            return false;
        }

        // Delegate to UPI backend
        return backend_->allgather(send_data, recv_data, local_count, CollectiveDataType::FLOAT32);
    }

    // =============================================================================
    // IGlobalTPContext Implementation
    // =============================================================================

    MPI_Comm GlobalTPContext::communicator() const
    {
        return domain_comm_;
    }

    int GlobalTPContext::domainId() const
    {
        return domain_id_;
    }

    const std::vector<int> &GlobalTPContext::worldRanks() const
    {
        return world_ranks_;
    }

    GlobalDeviceAddress GlobalTPContext::localDevice() const
    {
        // Get our world rank to construct the GlobalDeviceAddress
        int world_rank = 0;
        if (!world_ranks_.empty() && my_rank_in_domain_ >= 0 && 
            static_cast<size_t>(my_rank_in_domain_) < world_ranks_.size())
        {
            world_rank = world_ranks_[my_rank_in_domain_];
        }
        else
        {
            // Fallback: query MPI directly
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        }
        
        // Global TP is CPU-only, so return CPU device for this rank
        // Use "rank<N>" as hostname to uniquely identify each rank's CPU
        std::string hostname = "rank" + std::to_string(world_rank);
        return GlobalDeviceAddress::cpu(0, hostname);
    }

    void GlobalTPContext::barrier() const
    {
        if (domain_comm_ != MPI_COMM_NULL)
        {
            MPI_Barrier(domain_comm_);
        }
    }

    bool GlobalTPContext::send(const TensorBase *tensor, int dest_index)
    {
        if (!tensor)
        {
            LOG_ERROR("GlobalTPContext::send - tensor is null");
            return false;
        }

        if (dest_index < 0 || dest_index >= domain_size_)
        {
            LOG_ERROR("GlobalTPContext::send - invalid dest_index " << dest_index
                                                                    << " (domain_size=" << domain_size_ << ")");
            return false;
        }

        if (domain_comm_ == MPI_COMM_NULL)
        {
            LOG_ERROR("GlobalTPContext::send - communicator is null");
            return false;
        }

        // Get const data pointer
        const float *data = tensor->data();
        if (!data)
        {
            LOG_ERROR("GlobalTPContext::send - tensor has no data");
            return false;
        }

        size_t count = tensor->numel();
        if (count == 0)
        {
            LOG_WARN("GlobalTPContext::send - tensor has zero elements");
            return true; // Nothing to send
        }

        // Map dest_index to world rank for logging, but use dest_index for domain comm
        int dest_world_rank = (static_cast<size_t>(dest_index) < world_ranks_.size())
                                  ? world_ranks_[dest_index]
                                  : -1;

        LOG_DEBUG("GlobalTPContext::send - Sending " << count << " floats to domain_index="
                                                     << dest_index << " (world_rank=" << dest_world_rank << ")");

        // Use MPI_Send with dest_index (rank within domain communicator)
        int err = MPI_Send(data, static_cast<int>(count), MPI_FLOAT,
                           dest_index, 0, domain_comm_);

        if (err != MPI_SUCCESS)
        {
            LOG_ERROR("GlobalTPContext::send - MPI_Send failed with error " << err);
            return false;
        }

        return true;
    }

    bool GlobalTPContext::recv(TensorBase *tensor, int source_index)
    {
        if (!tensor)
        {
            LOG_ERROR("GlobalTPContext::recv - tensor is null");
            return false;
        }

        if (source_index < 0 || source_index >= domain_size_)
        {
            LOG_ERROR("GlobalTPContext::recv - invalid source_index " << source_index
                                                                      << " (domain_size=" << domain_size_ << ")");
            return false;
        }

        if (domain_comm_ == MPI_COMM_NULL)
        {
            LOG_ERROR("GlobalTPContext::recv - communicator is null");
            return false;
        }

        // Get mutable data pointer
        float *data = tensor->mutable_data();
        if (!data)
        {
            LOG_ERROR("GlobalTPContext::recv - tensor has no mutable data");
            return false;
        }

        size_t count = tensor->numel();
        if (count == 0)
        {
            LOG_WARN("GlobalTPContext::recv - tensor has zero elements");
            return true; // Nothing to receive
        }

        // Map source_index to world rank for logging
        int source_world_rank = (static_cast<size_t>(source_index) < world_ranks_.size())
                                    ? world_ranks_[source_index]
                                    : -1;

        LOG_DEBUG("GlobalTPContext::recv - Receiving " << count << " floats from domain_index="
                                                       << source_index << " (world_rank=" << source_world_rank << ")");

        // Use MPI_Recv with source_index (rank within domain communicator)
        MPI_Status status;
        int err = MPI_Recv(data, static_cast<int>(count), MPI_FLOAT,
                           source_index, 0, domain_comm_, &status);

        if (err != MPI_SUCCESS)
        {
            LOG_ERROR("GlobalTPContext::recv - MPI_Recv failed with error " << err);
            return false;
        }

        return true;
    }

    // =============================================================================
    // Additional Methods
    // =============================================================================

    bool GlobalTPContext::isValid() const
    {
        return domain_comm_ != MPI_COMM_NULL && domain_size_ > 0;
    }

} // namespace llaminar2
