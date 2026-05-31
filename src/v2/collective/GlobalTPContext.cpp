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
#include "backends/ShmemSpinBackend.h"
#include "../config/TPDomain.h"
#include "../tensors/Tensors.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include "../utils/NodeDetection.h"
#include <chrono>
#include <mpi.h>
#include <thread>
#include <utility>
#include <algorithm>
#include <limits>
#include <set>

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
        CollectiveBackendType backend_type,
        std::vector<int> node_ids)
        : domain_comm_(domain_comm), domain_id_(domain_id), my_rank_in_domain_(my_rank_in_domain), domain_size_(domain_size), world_ranks_(std::move(world_ranks)), node_ids_(std::move(node_ids)), all_same_node_(false), node_count_(0), owns_communicator_(owns_communicator), backend_type_(backend_type), backend_(nullptr)
    {
        // Auto-detect node IDs if not provided
        if (node_ids_.empty() && domain_comm_ != MPI_COMM_NULL)
        {
            detectNodeIds();
        }

        // Compute cached node topology fields
        if (!node_ids_.empty())
        {
            std::set<int> unique_nodes(node_ids_.begin(), node_ids_.end());
            node_count_ = static_cast<int>(unique_nodes.size());
            all_same_node_ = (node_count_ == 1);
        }
        else
        {
            // No node info available (e.g., null communicator)
            node_count_ = domain_size_;
            all_same_node_ = (domain_size_ <= 1);
        }
        // Create collective backend. Same-node UPI domains use ShmemSpin as an
        // opportunistic fast path for supported allreduces, while retaining UPI
        // as the fallback backend for unsupported collectives and init failures.
        if (domain_comm_ != MPI_COMM_NULL)
        {
            // Build the device group (needed by both backends)
            DeviceGroup group;
            group.name = "global_tp_domain_" + std::to_string(domain_id);
            group.scope = CollectiveScope::LOCAL; // Cross-socket on same node
            for (int i = 0; i < domain_size_; ++i)
            {
                group.devices.push_back(DeviceId::cpu());
            }

            CollectiveBackendType effective_backend = backend_type_;
            if (effective_backend == CollectiveBackendType::AUTO)
                effective_backend = CollectiveBackendType::UPI;

            auto create_upi_backend = [&]() -> std::unique_ptr<ICollectiveBackend>
            {
                auto upi = std::make_unique<UPICollectiveBackend>(domain_comm_, nullptr);
                if (!upi->initialize(group))
                {
                    LOG_ERROR("GlobalTPContext: Failed to initialize UPICollectiveBackend for domain "
                              << domain_id_ << " requested_backend="
                              << collectiveBackendTypeToString(backend_type_)
                              << " error=" << upi->lastError());
                    return nullptr;
                }
                return upi;
            };

            if (effective_backend == CollectiveBackendType::UPI && all_same_node_ && domain_size_ > 1)
            {
                backend_type_ = effective_backend;
                auto upi_fallback = std::make_unique<UPICollectiveBackend>(domain_comm_, nullptr);
                auto shmem = std::make_unique<ShmemSpinBackend>(
                    domain_id_, my_rank_in_domain_, std::move(upi_fallback));

                if (shmem->initialize(group))
                {
                    backend_ = std::move(shmem);
                    LOG_DEBUG("GlobalTPContext: Using ShmemSpinBackend for domain " << domain_id_
                                                                                    << " (same-node UPI fast path, domain_size="
                                                                                    << domain_size_ << ")");
                }
                else
                {
                    LOG_WARN("GlobalTPContext: ShmemSpinBackend unavailable for domain " << domain_id_
                                                                                         << "; falling back to UPI (error="
                                                                                         << shmem->lastError() << ")");
                    backend_ = create_upi_backend();
                }
            }
            else if (effective_backend == CollectiveBackendType::UPI ||
                     effective_backend == CollectiveBackendType::MPI)
            {
                backend_type_ = effective_backend;
                backend_ = create_upi_backend();
                if (backend_)
                {
                    LOG_DEBUG("GlobalTPContext: Using UPICollectiveBackend for domain " << domain_id_
                                                                                        << " (domain_size=" << domain_size_
                                                                                        << ", all_same_node=" << all_same_node_
                                                                                        << ", requested_backend="
                                                                                        << collectiveBackendTypeToString(backend_type_) << ")");
                }
            }
            else
            {
                LOG_ERROR("GlobalTPContext: Unsupported backend "
                          << collectiveBackendTypeToString(backend_type_)
                          << " for CPU cross-rank GlobalTP domain " << domain_id_);
            }

            if (backend_)
            {
                LOG_DEBUG("GlobalTPContext: Created for domain " << domain_id_
                                                                 << " with rank " << my_rank_in_domain_
                                                                 << "/" << domain_size_
                                                                 << " (owns_comm=" << owns_communicator_
                                                                 << ", all_same_node=" << all_same_node_
                                                                 << ", node_count=" << node_count_
                                                                 << ", backend=" << backend_->name() << ")");
            }
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
        int key,
        const std::string &hostfile_path,
        CollectiveBackendType backend_type)
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

        // Non-participant: color == MPI_UNDEFINED produces MPI_COMM_NULL.
        // Return nullptr gracefully so the caller knows this rank is not in the domain.
        if (new_comm == MPI_COMM_NULL)
        {
            LOG_DEBUG("GlobalTPContext::createWithSplit - rank excluded from domain " << domain_id
                                                                                      << " (color=MPI_UNDEFINED)");
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

        // Detect node IDs using hostfile if provided, else auto-detect
        // Doing this here (before constructor) means we pass node_ids directly,
        // and the constructor skips detectNodeIds() since node_ids will be non-empty.
        auto detection = NodeDetection::detect(new_comm, hostfile_path);

        // Create context - we own the communicator since we created it
        return std::unique_ptr<GlobalTPContext>(new GlobalTPContext(
            new_comm,
            domain_id,
            my_rank,
            domain_size,
            std::move(world_ranks),
            true, // owns_communicator = true (we created it)
            backend_type,
            std::move(detection.node_ids)));
    }

    std::unique_ptr<GlobalTPContext> GlobalTPContext::createForTest(
        MPI_Comm comm,
        int domain_id,
        std::vector<int> world_ranks,
        std::vector<int> node_ids,
        CollectiveBackendType backend_type)
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
            false, // owns_communicator = false (test owns it)
            backend_type,
            std::move(node_ids)));
    }

    // =============================================================================
    // Destructor
    // =============================================================================

    GlobalTPContext::~GlobalTPContext()
    {
        // Free the communicator if we own it
        if (owns_communicator_ && domain_comm_ != MPI_COMM_NULL)
        {
            // Guard against static destruction after MPI_Finalize
            int mpi_finalized = 0;
            MPI_Finalized(&mpi_finalized);
            if (!mpi_finalized)
            {
                LOG_DEBUG("GlobalTPContext: Freeing owned communicator for domain " << domain_id_);
                MPI_Comm_free(&domain_comm_);
            }
            domain_comm_ = MPI_COMM_NULL;
        }
    }

    // =============================================================================
    // Move Operations
    // =============================================================================

    GlobalTPContext::GlobalTPContext(GlobalTPContext &&other) noexcept
        : domain_comm_(other.domain_comm_), domain_id_(other.domain_id_), my_rank_in_domain_(other.my_rank_in_domain_), domain_size_(other.domain_size_), world_ranks_(std::move(other.world_ranks_)), node_ids_(std::move(other.node_ids_)), all_same_node_(other.all_same_node_), node_count_(other.node_count_), owns_communicator_(other.owns_communicator_), backend_type_(other.backend_type_), backend_(std::move(other.backend_)), abort_requested_(other.abort_requested_.load(std::memory_order_acquire))
    {
        // Clear source to prevent double-free
        other.domain_comm_ = MPI_COMM_NULL;
        other.owns_communicator_ = false;
        other.domain_id_ = -1;
        other.my_rank_in_domain_ = -1;
        other.domain_size_ = 0;
        other.all_same_node_ = false;
        other.node_count_ = 0;
    }

    GlobalTPContext &GlobalTPContext::operator=(GlobalTPContext &&other) noexcept
    {
        if (this != &other)
        {
            // Free our communicator if we own it
            if (owns_communicator_ && domain_comm_ != MPI_COMM_NULL)
            {
                int mpi_finalized = 0;
                MPI_Finalized(&mpi_finalized);
                if (!mpi_finalized)
                    MPI_Comm_free(&domain_comm_);
            }

            // Transfer ownership
            domain_comm_ = other.domain_comm_;
            domain_id_ = other.domain_id_;
            my_rank_in_domain_ = other.my_rank_in_domain_;
            domain_size_ = other.domain_size_;
            world_ranks_ = std::move(other.world_ranks_);
            node_ids_ = std::move(other.node_ids_);
            all_same_node_ = other.all_same_node_;
            node_count_ = other.node_count_;
            owns_communicator_ = other.owns_communicator_;
            backend_type_ = other.backend_type_;
            backend_ = std::move(other.backend_);
            abort_requested_.store(other.abort_requested_.load(std::memory_order_acquire), std::memory_order_release);

            // Clear source to prevent double-free
            other.domain_comm_ = MPI_COMM_NULL;
            other.owns_communicator_ = false;
            other.domain_id_ = -1;
            other.my_rank_in_domain_ = -1;
            other.domain_size_ = 0;
            other.all_same_node_ = false;
            other.node_count_ = 0;
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
        return allreduce(tensor, "", 0);
    }

    bool GlobalTPContext::allreduce(TensorBase *tensor, const std::string &stage_name, size_t count)
    {
        (void)stage_name; // Stage name currently not used by GLOBAL TP backend

        if (isAbortRequested())
        {
            LOG_ERROR("GlobalTPContext::allreduce - abort already requested for domain " << domain_id_);
            return false;
        }

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

        const size_t tensor_numel = tensor->numel();
        const size_t effective_count = (count > 0) ? count : tensor_numel;

        if (effective_count > tensor_numel)
        {
            LOG_ERROR("GlobalTPContext::allreduce - effective_count=" << effective_count
                                                                      << " exceeds tensor numel=" << tensor_numel);
            return false;
        }

        if (effective_count == 0)
        {
            LOG_WARN("GlobalTPContext::allreduce - tensor has zero elements");
            return true; // Nothing to reduce
        }

        // Delegate to UPI backend
        return backend_->allreduce(data, effective_count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
    }

    bool GlobalTPContext::broadcast(TensorBase *tensor, int source_index)
    {
        if (isAbortRequested())
        {
            LOG_ERROR("GlobalTPContext::broadcast - abort already requested for domain " << domain_id_);
            return false;
        }

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
        if (isAbortRequested())
        {
            LOG_ERROR("GlobalTPContext::allgather - abort already requested for domain " << domain_id_);
            return false;
        }

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

    void GlobalTPContext::requestAbort()
    {
        const bool was_set = abort_requested_.exchange(true, std::memory_order_acq_rel);
        if (was_set)
            return;

        LOG_ERROR("GlobalTPContext: Abort requested for domain " << domain_id_
                                                                 << " rank " << my_rank_in_domain_ << "/" << domain_size_
                                                                 << "; aborting backend and MPI job to avoid rank desynchronization");
        if (backend_)
            backend_->abort();

        if (domain_comm_ != MPI_COMM_NULL && domain_size_ > 1)
            MPI_Abort(MPI_COMM_WORLD, 1);
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
        if (domain_comm_ == MPI_COMM_NULL)
            return;

        const int timeout_ms = debugEnv().tp_collect_timeout_ms;
        if (timeout_ms <= 0)
        {
            MPI_Barrier(domain_comm_);
            return;
        }

        MPI_Request request = MPI_REQUEST_NULL;
        int result = MPI_Ibarrier(domain_comm_, &request);
        if (result != MPI_SUCCESS)
        {
            LOG_ERROR("GlobalTPContext::barrier - MPI_Ibarrier failed with code " << result
                                                                                  << " on domain " << domain_id_ << " rank " << my_rank_in_domain_
                                                                                  << "/" << domain_size_);
            return;
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        int complete = 0;
        while (!complete)
        {
            result = MPI_Test(&request, &complete, MPI_STATUS_IGNORE);
            if (result != MPI_SUCCESS || complete)
                break;

            if (std::chrono::steady_clock::now() >= deadline)
            {
                LOG_ERROR("GlobalTPContext::barrier timed out after " << timeout_ms
                                                                      << "ms on domain " << domain_id_ << " rank " << my_rank_in_domain_
                                                                      << "/" << domain_size_
                                                                      << "; aborting MPI job to avoid rank desynchronization");
                if (backend_)
                    backend_->abort();
                MPI_Abort(MPI_COMM_WORLD, 1);
                return;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (result != MPI_SUCCESS)
        {
            LOG_ERROR("GlobalTPContext::barrier - MPI_Test failed with code " << result
                                                                              << " on domain " << domain_id_ << " rank " << my_rank_in_domain_
                                                                              << "/" << domain_size_);
        }
    }

    bool GlobalTPContext::allgatherBytes(const void *send_data, void *recv_data, size_t byte_count) const
    {
        if (isAbortRequested())
        {
            LOG_ERROR("GlobalTPContext::allgatherBytes - abort already requested for domain "
                      << domain_id_);
            return false;
        }
        if (domain_comm_ == MPI_COMM_NULL || !send_data || !recv_data || byte_count == 0)
        {
            LOG_ERROR("GlobalTPContext::allgatherBytes - invalid arguments for domain "
                      << domain_id_);
            return false;
        }
        if (byte_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("GlobalTPContext::allgatherBytes - byte_count too large for MPI_Allgather: "
                      << byte_count);
            return false;
        }

        const int count = static_cast<int>(byte_count);
        const int result = MPI_Allgather(send_data, count, MPI_BYTE,
                                         recv_data, count, MPI_BYTE,
                                         domain_comm_);
        if (result != MPI_SUCCESS)
        {
            LOG_ERROR("GlobalTPContext::allgatherBytes - MPI_Allgather failed with code "
                      << result << " on domain " << domain_id_ << " rank "
                      << my_rank_in_domain_ << "/" << domain_size_);
            return false;
        }
        return true;
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

    // =============================================================================
    // Node Awareness
    // =============================================================================

    TPScope GlobalTPContext::scope() const
    {
        return all_same_node_ ? TPScope::NODE_LOCAL : TPScope::GLOBAL;
    }

    bool GlobalTPContext::isAllRanksOnSameNode() const
    {
        return all_same_node_;
    }

    int GlobalTPContext::nodeId(int domain_index) const
    {
        if (domain_index < 0 || static_cast<size_t>(domain_index) >= node_ids_.size())
        {
            return -1;
        }
        return node_ids_[domain_index];
    }

    const std::vector<int> &GlobalTPContext::nodeIds() const
    {
        return node_ids_;
    }

    int GlobalTPContext::nodeCount() const
    {
        return node_count_;
    }

    std::string GlobalTPContext::collectiveBackendNameForDiagnostics() const
    {
        return backend_ ? backend_->name() : "none";
    }

    void GlobalTPContext::detectNodeIds()
    {
        if (domain_comm_ == MPI_COMM_NULL || domain_size_ <= 0)
        {
            return;
        }

        // Delegate to the canonical hostname-based node detection
        auto detection = NodeDetection::detect(domain_comm_);
        node_ids_ = std::move(detection.node_ids);
        // node_count_ and all_same_node_ are computed by the caller
    }

} // namespace llaminar2
