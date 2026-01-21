/**
 * @file CollectiveContext.cpp
 * @brief CollectiveContext implementation for runtime collective operations
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CollectiveContext.h"
#include "DeviceInventory.h"
#include "../collective/backends/PCIeBARBackend.h"
#include "../config/TPDomain.h"
#include "../tensors/ITensor.h"
#include "../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // CollectiveContext Implementation
    // =========================================================================

    CollectiveContext::CollectiveContext(const Config &config)
        : config_(config), mpi_ctx_(config.mpi_ctx)
    {
        // Build local device list from cluster inventory
        if (config.cluster_inventory && mpi_ctx_)
        {
            int rank = mpi_ctx_->rank();
            if (rank < static_cast<int>(config.cluster_inventory->ranks.size()))
            {
                const RankInventory &inv = config.cluster_inventory->ranks[rank];
                for (const auto &gpu : inv.gpus)
                {
                    if (gpu.type == DeviceType::CUDA)
                    {
                        local_devices_.push_back(DeviceId::cuda(gpu.local_device_id));
                    }
                    else if (gpu.type == DeviceType::ROCm)
                    {
                        local_devices_.push_back(DeviceId::rocm(gpu.local_device_id));
                    }
                }
                // Always include CPU
                local_devices_.push_back(DeviceId::cpu());
            }
        }
        else
        {
            // Default: single CPU device
            local_devices_.push_back(DeviceId::cpu());
        }

        // Create router
        if (config.cluster_inventory)
        {
            router_ = std::make_unique<BackendRouter>(
                mpi_ctx_, *config.cluster_inventory);
        }

        // Set world size
        world_size_ = mpi_ctx_ ? mpi_ctx_->world_size() : 1;
    }

    CollectiveContext::CollectiveContext(
        std::unique_ptr<IBackendRouter> router,
        std::shared_ptr<MPIContext> mpi_ctx,
        std::vector<DeviceId> local_devices)
        : mpi_ctx_(std::move(mpi_ctx)),
          router_(std::move(router)),
          local_devices_(std::move(local_devices))
    {
        world_size_ = mpi_ctx_ ? mpi_ctx_->world_size() : 1;
    }

    CollectiveContext::~CollectiveContext() = default;

    bool CollectiveContext::executeAllreduce(
        ITensor *buffer,
        size_t count,
        DeviceId tensor_device,
        CollectiveOp op)
    {
        if (!router_)
        {
            LOG_ERROR("CollectiveContext: No router available");
            return false;
        }

        // Build device group for this collective
        DeviceGroup group = buildDeviceGroup(tensor_device);

        // Get backend for this group
        ICollectiveBackend *backend = router_->getBackend(group);
        if (!backend)
        {
            LOG_ERROR("CollectiveContext: No backend available for device group " << group.name);
            return false;
        }

        // Determine data type from buffer
        CollectiveDataType dtype = CollectiveDataType::FLOAT32; // Default
        // TODO: Query buffer for actual dtype

        // Execute collective
        void *data_ptr = buffer->mutable_data();
        size_t actual_count = count > 0 ? count : buffer->numel();

        return backend->allreduce(data_ptr, actual_count, dtype, op);
    }

    bool CollectiveContext::executeAllgather(
        ITensor *local_input,
        ITensor *full_output,
        size_t actual_seq_len,
        DeviceId tensor_device)
    {
        if (!router_)
        {
            LOG_ERROR("CollectiveContext: No router available");
            return false;
        }

        // Build device group
        DeviceGroup group = buildDeviceGroup(tensor_device);

        // Get backend
        ICollectiveBackend *backend = router_->getBackend(group);
        if (!backend)
        {
            LOG_ERROR("CollectiveContext: No backend available for device group " << group.name);
            return false;
        }

        // Execute collective
        const void *send_buf = local_input->data();
        void *recv_buf = full_output->mutable_data();
        size_t send_count = actual_seq_len > 0 ? actual_seq_len : local_input->numel();

        return backend->allgather(send_buf, recv_buf, send_count, CollectiveDataType::FLOAT32);
    }

    bool CollectiveContext::executeAllgatherv(
        ITensor *local_input,
        ITensor *full_output,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        size_t actual_seq_len,
        DeviceId tensor_device)
    {
        if (!router_)
        {
            LOG_ERROR("CollectiveContext: No router available for allgatherv");
            return false;
        }

        // Build device group
        DeviceGroup group = buildDeviceGroup(tensor_device);

        // Get backend
        ICollectiveBackend *backend = router_->getBackend(group);
        if (!backend)
        {
            LOG_ERROR("CollectiveContext: No backend available for device group " << group.name);
            return false;
        }

        // Get shapes and calculate send count
        const auto &local_shape = local_input->shape();
        size_t buffer_seq_len = local_shape.size() >= 2 ? local_shape[0] : 1;
        size_t local_dim = local_shape.size() >= 2 ? local_shape[1] : local_shape[0];
        size_t seq_len = actual_seq_len > 0 ? actual_seq_len : buffer_seq_len;

        // Calculate send count for this rank (seq_len * local_dim)
        size_t send_count = seq_len * local_dim;

        // Scale recv_counts and displacements by seq_len
        std::vector<int> scaled_recv_counts(recv_counts.size());
        std::vector<int> scaled_displacements(displacements.size());
        for (size_t i = 0; i < recv_counts.size(); ++i)
        {
            scaled_recv_counts[i] = static_cast<int>(seq_len) * recv_counts[i];
            scaled_displacements[i] = static_cast<int>(seq_len) * displacements[i];
        }

        // Execute collective
        const void *send_buf = local_input->data();
        void *recv_buf = full_output->mutable_data();

        return backend->allgatherv(send_buf, send_count, recv_buf,
                                   scaled_recv_counts, scaled_displacements,
                                   CollectiveDataType::FLOAT32);
    }

    bool CollectiveContext::executeBroadcast(
        ITensor *buffer,
        size_t count,
        int root_rank,
        DeviceId tensor_device)
    {
        if (!router_)
        {
            LOG_ERROR("CollectiveContext: No router available");
            return false;
        }

        // Build device group
        DeviceGroup group = buildDeviceGroup(tensor_device);

        // Get backend
        ICollectiveBackend *backend = router_->getBackend(group);
        if (!backend)
        {
            LOG_ERROR("CollectiveContext: No backend available for device group " << group.name);
            return false;
        }

        // Execute collective
        void *data_ptr = buffer->mutable_data();
        size_t actual_count = count > 0 ? count : buffer->numel();

        return backend->broadcast(data_ptr, actual_count, CollectiveDataType::FLOAT32, root_rank);
    }

    bool CollectiveContext::requiresCollectives() const
    {
        return world_size_ > 1 || local_devices_.size() > 1;
    }

    int CollectiveContext::worldSize() const
    {
        return world_size_;
    }

    int CollectiveContext::rank() const
    {
        return mpi_ctx_ ? mpi_ctx_->rank() : 0;
    }

    bool CollectiveContext::isBackendAvailable(CollectiveBackendType type) const
    {
        return router_ ? router_->isAvailable(type) : false;
    }

    // =========================================================================
    // Buffer Registration Support (Phase 3)
    // =========================================================================

    PCIeBARBackend *CollectiveContext::getPCIeBarBackend() const
    {
        if (!router_)
        {
            return nullptr;
        }

        // Check if PCIe BAR backend is available
        if (!router_->isAvailable(CollectiveBackendType::PCIE_BAR))
        {
            return nullptr;
        }

        // Build a device group to get the backend
        // Use first local device to query
        DeviceGroup group;
        if (!local_devices_.empty())
        {
            DeviceGroupBuilder builder;
            builder.setName("bar_query_group")
                .setScope(CollectiveScope::LOCAL);
            for (const auto &device : local_devices_)
            {
                builder.addDevice(device);
            }
            group = builder.build();
        }
        else
        {
            return nullptr;
        }

        ICollectiveBackend *backend = router_->getBackend(group);
        if (!backend)
        {
            return nullptr;
        }

        // Check if it's actually a PCIeBARBackend
        if (backend->type() != CollectiveBackendType::PCIE_BAR)
        {
            return nullptr;
        }

        return static_cast<PCIeBARBackend *>(backend);
    }

    bool CollectiveContext::requiresBufferRegistration() const
    {
        // Check if any available backend requires buffer registration
        PCIeBARBackend *bar_backend = getPCIeBarBackend();
        if (bar_backend && bar_backend->requiresBufferRegistration())
        {
            return true;
        }

        return false;
    }

    ICollectiveBackend *CollectiveContext::selectBackend(DeviceId tensor_device)
    {
        if (!router_)
        {
            return nullptr;
        }

        DeviceGroup group = buildDeviceGroup(tensor_device);
        return router_->getBackend(group);
    }

    DeviceGroup CollectiveContext::buildDeviceGroup(DeviceId tensor_device) const
    {
        DeviceGroupBuilder builder;

        // For MPI-based collectives, create a global group
        if (world_size_ > 1)
        {
            builder.setName("global_mpi_group")
                .setScope(CollectiveScope::GLOBAL)
                .setLocalRank(rank());

            // Add local device for this rank
            builder.addDevice(tensor_device);
        }
        else
        {
            // Single-rank: local group with available devices
            builder.setName("local_group")
                .setScope(CollectiveScope::LOCAL)
                .setLocalRank(0);

            for (const auto &device : local_devices_)
            {
                builder.addDevice(device);
            }
        }

        return builder.build();
    }

    // =========================================================================
    // Domain-Aware Collective Operations
    // =========================================================================

    bool CollectiveContext::executeAllreduceInDomain(
        ITensor *buffer,
        size_t count,
        DeviceId tensor_device,
        CollectiveOp op,
        const TPDomain *domain)
    {
        // Fallback to legacy path when domain is nullptr
        if (!domain)
        {
            LOG_DEBUG("CollectiveContext::executeAllreduceInDomain: No domain specified, "
                      "falling back to non-domain executeAllreduce");
            return executeAllreduce(buffer, count, tensor_device, op);
        }

        if (!router_)
        {
            LOG_ERROR("CollectiveContext: No router available for domain-aware allreduce");
            return false;
        }

        // Skip trivial domains (size 1, no communication needed)
        if (domain->isTrivial())
        {
            LOG_DEBUG("CollectiveContext::executeAllreduceInDomain: Domain '" << domain->name
                                                                              << "' is trivial (size=" << domain->domain_size << "), skipping");
            return true;
        }

        // Use domain-aware backend selection
        ICollectiveBackend *backend = router_->selectBackendForDomain(domain);
        if (!backend)
        {
            LOG_ERROR("CollectiveContext: No backend available for domain '" << domain->name << "'");
            return false;
        }

        LOG_DEBUG("CollectiveContext: Executing allreduce in domain '" << domain->name
                                                                       << "' (type=" << tpDomainTypeToString(domain->type)
                                                                       << ", size=" << domain->domain_size
                                                                       << ") via " << toString(backend->type()));

        // Determine data type from buffer
        CollectiveDataType dtype = CollectiveDataType::FLOAT32; // Default
        // TODO: Query buffer for actual dtype

        // Execute collective
        void *data_ptr = buffer->mutable_data();
        size_t actual_count = count > 0 ? count : buffer->numel();

        return backend->allreduce(data_ptr, actual_count, dtype, op);
    }

    bool CollectiveContext::executeAllgatherInDomain(
        ITensor *local_input,
        ITensor *full_output,
        size_t actual_seq_len,
        DeviceId tensor_device,
        const TPDomain *domain)
    {
        // Fallback to legacy path when domain is nullptr
        if (!domain)
        {
            LOG_DEBUG("CollectiveContext::executeAllgatherInDomain: No domain specified, "
                      "falling back to non-domain executeAllgather");
            return executeAllgather(local_input, full_output, actual_seq_len, tensor_device);
        }

        if (!router_)
        {
            LOG_ERROR("CollectiveContext: No router available for domain-aware allgather");
            return false;
        }

        // Skip trivial domains (size 1, no communication needed)
        if (domain->isTrivial())
        {
            LOG_DEBUG("CollectiveContext::executeAllgatherInDomain: Domain '" << domain->name
                                                                              << "' is trivial, skipping");
            return true;
        }

        // Use domain-aware backend selection
        ICollectiveBackend *backend = router_->selectBackendForDomain(domain);
        if (!backend)
        {
            LOG_ERROR("CollectiveContext: No backend available for domain '" << domain->name << "'");
            return false;
        }

        LOG_DEBUG("CollectiveContext: Executing allgather in domain '" << domain->name
                                                                       << "' (type=" << tpDomainTypeToString(domain->type)
                                                                       << ", size=" << domain->domain_size
                                                                       << ") via " << toString(backend->type()));

        // Execute collective
        const void *send_buf = local_input->data();
        void *recv_buf = full_output->mutable_data();
        size_t send_count = actual_seq_len > 0 ? actual_seq_len : local_input->numel();

        return backend->allgather(send_buf, recv_buf, send_count, CollectiveDataType::FLOAT32);
    }

    bool CollectiveContext::executeAllgathervInDomain(
        ITensor *local_input,
        ITensor *full_output,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        size_t actual_seq_len,
        DeviceId tensor_device,
        const TPDomain *domain)
    {
        // Fallback to legacy path when domain is nullptr
        if (!domain)
        {
            LOG_DEBUG("CollectiveContext::executeAllgathervInDomain: No domain specified, "
                      "falling back to non-domain executeAllgatherv");
            return executeAllgatherv(local_input, full_output, recv_counts, displacements,
                                     actual_seq_len, tensor_device);
        }

        if (!router_)
        {
            LOG_ERROR("CollectiveContext: No router available for domain-aware allgatherv");
            return false;
        }

        // Skip trivial domains (size 1, no communication needed)
        if (domain->isTrivial())
        {
            LOG_DEBUG("CollectiveContext::executeAllgathervInDomain: Domain '" << domain->name
                                                                               << "' is trivial, skipping");
            return true;
        }

        // Use domain-aware backend selection
        ICollectiveBackend *backend = router_->selectBackendForDomain(domain);
        if (!backend)
        {
            LOG_ERROR("CollectiveContext: No backend available for domain '" << domain->name << "'");
            return false;
        }

        LOG_DEBUG("CollectiveContext: Executing allgatherv in domain '" << domain->name
                                                                        << "' (type=" << tpDomainTypeToString(domain->type)
                                                                        << ", size=" << domain->domain_size
                                                                        << ") via " << toString(backend->type()));

        // Get shapes and calculate send count
        const auto &local_shape = local_input->shape();
        size_t buffer_seq_len = local_shape.size() >= 2 ? local_shape[0] : 1;
        size_t local_dim = local_shape.size() >= 2 ? local_shape[1] : local_shape[0];
        size_t seq_len = actual_seq_len > 0 ? actual_seq_len : buffer_seq_len;

        // Calculate send count for this rank (seq_len * local_dim)
        size_t send_count = seq_len * local_dim;

        // Scale recv_counts and displacements by seq_len
        std::vector<int> scaled_recv_counts(recv_counts.size());
        std::vector<int> scaled_displacements(displacements.size());
        for (size_t i = 0; i < recv_counts.size(); ++i)
        {
            scaled_recv_counts[i] = static_cast<int>(seq_len) * recv_counts[i];
            scaled_displacements[i] = static_cast<int>(seq_len) * displacements[i];
        }

        // Execute collective
        const void *send_buf = local_input->data();
        void *recv_buf = full_output->mutable_data();

        return backend->allgatherv(send_buf, send_count, recv_buf,
                                   scaled_recv_counts, scaled_displacements,
                                   CollectiveDataType::FLOAT32);
    }

    // =========================================================================
    // CollectiveContextFactory Implementation
    // =========================================================================

    std::unique_ptr<CollectiveContext> CollectiveContextFactory::createSingleDevice()
    {
        CollectiveContext::Config config;
        // No MPI, no cluster inventory - single device
        return std::make_unique<CollectiveContext>(config);
    }

    std::unique_ptr<CollectiveContext> CollectiveContextFactory::createMPI(
        std::shared_ptr<MPIContext> mpi_ctx)
    {
        CollectiveContext::Config config;
        config.mpi_ctx = std::move(mpi_ctx);
        return std::make_unique<CollectiveContext>(config);
    }

    std::unique_ptr<CollectiveContext> CollectiveContextFactory::createIntraNode(
        const ClusterInventory &inventory,
        std::shared_ptr<MPIContext> mpi_ctx)
    {
        CollectiveContext::Config config;
        config.mpi_ctx = std::move(mpi_ctx);
        config.cluster_inventory = &inventory;
        return std::make_unique<CollectiveContext>(config);
    }

    std::unique_ptr<CollectiveContext> CollectiveContextFactory::createWithRouter(
        std::unique_ptr<IBackendRouter> router,
        std::shared_ptr<MPIContext> mpi_ctx,
        std::vector<DeviceId> local_devices)
    {
        return std::make_unique<CollectiveContext>(
            std::move(router), std::move(mpi_ctx), std::move(local_devices));
    }

    std::unique_ptr<CollectiveContext> CollectiveContextFactory::create(
        const CollectiveContext::Config &config)
    {
        return std::make_unique<CollectiveContext>(config);
    }

} // namespace llaminar2
