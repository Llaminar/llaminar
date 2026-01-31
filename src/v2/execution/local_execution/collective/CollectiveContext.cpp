/**
 * @file CollectiveContext.cpp
 * @brief CollectiveContext implementation for runtime collective operations
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CollectiveContext.h"
#include "../../mpi_orchestration/DeviceInventory.h"
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "../../../collective/backends/PCIeBARBackend.h"
#endif
#ifdef HAVE_CUDA
#include "../../../collective/backends/NCCLBackend.h"
#endif
#include "../../../config/TPDomain.h"
#include "../../../tensors/ITensor.h"
#include "../../../tensors/TensorClasses.h" // For TensorBase (MPIStager compatibility)
#include "../../../utils/Logger.h"
#include "../../../utils/MPIStager.h" // For GPU↔Host staging with MPI backend

namespace llaminar2
{

    // =========================================================================
    // Helper: Convert TensorType to CollectiveDataType
    // =========================================================================

    /**
     * @brief Convert tensor native type to collective data type for MPI/NCCL/RCCL
     * @param tensor The tensor to query (must not be nullptr)
     * @return CollectiveDataType matching the tensor's storage format
     * @throws std::invalid_argument if tensor is nullptr
     * @throws std::invalid_argument if tensor type is not supported for collectives
     */
    CollectiveDataType tensorToCollectiveDataType(const ITensor *tensor)
    {
        if (!tensor)
        {
            throw std::invalid_argument(
                "tensorToCollectiveDataType: tensor is nullptr");
        }

        switch (tensor->native_type())
        {
        case TensorType::FP32:
            return CollectiveDataType::FLOAT32;
        case TensorType::FP16:
            return CollectiveDataType::FLOAT16;
        case TensorType::BF16:
            return CollectiveDataType::BFLOAT16;
        case TensorType::INT32:
            return CollectiveDataType::INT32;
        case TensorType::INT8:
            return CollectiveDataType::INT8;
        default:
            // Quantized types (Q8_0, Q4_0, IQ4_NL, etc.) are not valid for
            // collective operations. Collectives should operate on activation
            // buffers (FP32/FP16/BF16), not weight buffers (quantized).
            throw std::invalid_argument(
                std::string("tensorToCollectiveDataType: unsupported tensor type '") +
                tensor->dtype_name() + "' for collective operation");
        }
    }

    // =========================================================================
    // CollectiveContext Implementation
    // =========================================================================

    CollectiveContext::CollectiveContext(const Config &config)
        : config_(config), mpi_ctx_(config.mpi_ctx)
    {
        // Build local device list from cluster inventory
        if (config.cluster_inventory && !config.cluster_inventory->ranks.empty())
        {
            // Note: buildLocalClusterInventory creates a single-entry ranks array
            // containing this rank's local devices. The entry is always at index 0,
            // not at index == mpi_rank. Use the first (and only) entry.
            const RankInventory &inv = config.cluster_inventory->ranks[0];
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
        // Single-device: AllReduce is a no-op (nothing to reduce across)
        if (!requiresCollectives())
        {
            LOG_DEBUG("CollectiveContext: Single-device allreduce is a no-op");
            return true;
        }

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

        // Query buffer for actual dtype
        CollectiveDataType dtype = tensorToCollectiveDataType(buffer);
        size_t actual_count = count > 0 ? count : buffer->numel();

        // =========================================================================
        // MPI Backend with GPU Buffer: Requires Host Staging
        // =========================================================================
        // MPI cannot access GPU memory directly (CUDA/ROCm pointers).
        // When using MPI backend with GPU-resident tensors, we must:
        // 1. Stage GPU→Host
        // 2. MPI_Allreduce on host buffer
        // 3. Stage Host→GPU
        // =========================================================================
        if (backend->type() == CollectiveBackendType::MPI && buffer->is_on_gpu())
        {
            LOG_DEBUG("CollectiveContext: MPI+GPU staging for allreduce (count=" << actual_count << ")");

            // Cast to TensorBase for MPIStager compatibility
            auto *tensor_base = dynamic_cast<TensorBase *>(buffer);
            if (!tensor_base)
            {
                LOG_ERROR("CollectiveContext: Buffer is not TensorBase, cannot stage for MPI");
                return false;
            }

            // Ensure GPU kernels complete before staging
            MPIStager::synchronizeDevice(tensor_base->home_device().gpu_ordinal());

            // Stage GPU→Host
            std::vector<float> host_buffer = MPIStager::toHost(tensor_base);

            // MPI allreduce on host memory (in-place)
            if (!backend->allreduce(host_buffer.data(), actual_count, dtype, op))
            {
                LOG_ERROR("CollectiveContext: MPI allreduce failed during staged operation");
                return false;
            }

            // Stage Host→GPU (copy result back to device)
            MPIStager::toDevice(host_buffer, tensor_base);

            return true;
        }

        // =========================================================================
        // Standard Path: Direct backend call (NCCL, RCCL, PCIeBAR, or MPI+CPU)
        // =========================================================================
        // CRITICAL: Use active_mutable_data_ptr() to get GPU pointer without invalidating
        // GPU data. Using mutable_data() would mark device_valid_=false, causing expensive
        // re-uploads on subsequent operations that need this tensor on GPU.
        void *data_ptr = buffer->active_mutable_data_ptr();

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

        // Query buffer for actual dtype (use input tensor as reference)
        CollectiveDataType dtype = tensorToCollectiveDataType(local_input);
        size_t send_count = actual_seq_len > 0 ? actual_seq_len : local_input->numel();

        // =========================================================================
        // MPI Backend with GPU Buffers: Requires Host Staging
        // =========================================================================
        bool input_on_gpu = local_input->is_on_gpu();
        bool output_on_gpu = full_output->is_on_gpu();

        if (backend->type() == CollectiveBackendType::MPI && (input_on_gpu || output_on_gpu))
        {
            LOG_DEBUG("CollectiveContext: MPI+GPU staging for allgather (send_count=" << send_count << ")");

            // Cast to TensorBase for MPIStager compatibility
            auto *input_base = dynamic_cast<TensorBase *>(local_input);
            auto *output_base = dynamic_cast<TensorBase *>(full_output);
            if (!input_base || !output_base)
            {
                LOG_ERROR("CollectiveContext: Buffers are not TensorBase, cannot stage for MPI");
                return false;
            }

            // Sync input device if on GPU
            if (input_on_gpu)
            {
                MPIStager::synchronizeDevice(input_base->home_device().gpu_ordinal());
            }

            // Stage input GPU→Host
            std::vector<float> host_send = input_on_gpu ? MPIStager::toHost(input_base)
                                                        : std::vector<float>(input_base->data(), input_base->data() + input_base->numel());

            // Allocate host receive buffer (full size for all ranks)
            size_t recv_count = send_count * (mpi_ctx_ ? mpi_ctx_->world_size() : 1);
            std::vector<float> host_recv(recv_count);

            // MPI allgather on host memory
            if (!backend->allgather(host_send.data(), host_recv.data(), send_count, dtype))
            {
                LOG_ERROR("CollectiveContext: MPI allgather failed during staged operation");
                return false;
            }

            // Stage Host→GPU for output (or copy to CPU tensor)
            if (output_on_gpu)
            {
                MPIStager::toDevice(host_recv, output_base);
            }
            else
            {
                std::memcpy(output_base->mutable_data(), host_recv.data(), host_recv.size() * sizeof(float));
            }

            return true;
        }

        // =========================================================================
        // Standard Path: Direct backend call (NCCL, RCCL, PCIeBAR, or MPI+CPU)
        // =========================================================================
        // Use active_data_ptr() for send and active_mutable_data_ptr() for receive
        // to avoid invalidating GPU data on tensors that should remain on device
        const void *send_buf = local_input->active_data_ptr();
        void *recv_buf = full_output->active_mutable_data_ptr();

        return backend->allgather(send_buf, recv_buf, send_count, dtype);
    }

    bool CollectiveContext::executeStridedAllgather(
        ITensor *local_input,
        ITensor *full_output,
        size_t actual_seq_len,
        DeviceId tensor_device)
    {
#ifndef HAVE_CUDA
        // Strided allgather requires NCCL which requires CUDA
        (void)local_input;
        (void)full_output;
        (void)actual_seq_len;
        (void)tensor_device;
        LOG_ERROR("CollectiveContext: stridedAllgather requires CUDA support");
        return false;
#else
        if (!router_)
        {
            LOG_ERROR("CollectiveContext: No router available for stridedAllgather");
            return false;
        }

        // Strided allgather only works on CUDA devices (uses NCCL + CUDA kernel)
        if (tensor_device.type != DeviceType::CUDA)
        {
            LOG_DEBUG("CollectiveContext: stridedAllgather requires CUDA device, got "
                      << tensor_device.to_string());
            return false;
        }

        // Build device group
        DeviceGroup group = buildDeviceGroup(tensor_device);

        // Get backend - must be NCCL for strided allgather
        ICollectiveBackend *backend = router_->getBackend(group);
        if (!backend || backend->type() != CollectiveBackendType::NCCL)
        {
            LOG_DEBUG("CollectiveContext: stridedAllgather requires NCCL backend, got "
                      << (backend ? toString(backend->type()) : "none"));
            return false;
        }

        // Cast to NCCLBackend to access stridedAllgather
        NCCLBackend *nccl_backend = dynamic_cast<NCCLBackend *>(backend);
        if (!nccl_backend)
        {
            LOG_ERROR("CollectiveContext: Failed to cast backend to NCCLBackend");
            return false;
        }

        // Only supports FP32 currently
        CollectiveDataType dtype = tensorToCollectiveDataType(local_input);
        if (dtype != CollectiveDataType::FLOAT32)
        {
            LOG_DEBUG("CollectiveContext: stridedAllgather only supports FP32, got "
                      << static_cast<int>(dtype));
            return false;
        }

        // Get tensor dimensions
        const auto &local_shape = local_input->shape();
        const auto &full_shape = full_output->shape();

        if (local_shape.size() != 2 || full_shape.size() != 2)
        {
            LOG_ERROR("CollectiveContext: stridedAllgather requires 2D tensors");
            return false;
        }

        size_t buffer_seq_len = local_shape[0];
        size_t seq_len = actual_seq_len > 0 ? actual_seq_len : buffer_seq_len;
        size_t local_dim = local_shape[1];

        // Use active_data_ptr for GPU memory access
        const void *send_buf = local_input->active_data_ptr();
        void *recv_buf = full_output->active_mutable_data_ptr();

        bool success = nccl_backend->stridedAllgather(
            send_buf,
            recv_buf,
            seq_len,
            local_dim,
            dtype);

        if (success)
        {
            LOG_DEBUG("CollectiveContext: stridedAllgather completed seq_len=" << seq_len
                                                                               << " local_dim=" << local_dim);
        }
        else
        {
            LOG_ERROR("CollectiveContext: stridedAllgather failed: " << nccl_backend->lastError());
        }

        return success;
#endif // HAVE_CUDA
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

        // Query buffer for actual dtype (use input tensor as reference)
        CollectiveDataType dtype = tensorToCollectiveDataType(local_input);

        // =========================================================================
        // MPI Backend with GPU Buffers: Requires Host Staging
        // =========================================================================
        bool input_on_gpu = local_input->is_on_gpu();
        bool output_on_gpu = full_output->is_on_gpu();

        if (backend->type() == CollectiveBackendType::MPI && (input_on_gpu || output_on_gpu))
        {
            LOG_DEBUG("CollectiveContext: MPI+GPU staging for allgatherv (send_count=" << send_count << ")");

            // Cast to TensorBase for MPIStager compatibility
            auto *input_base = dynamic_cast<TensorBase *>(local_input);
            auto *output_base = dynamic_cast<TensorBase *>(full_output);
            if (!input_base || !output_base)
            {
                LOG_ERROR("CollectiveContext: Buffers are not TensorBase, cannot stage for MPI");
                return false;
            }

            // Sync input device if on GPU
            if (input_on_gpu)
            {
                MPIStager::synchronizeDevice(input_base->home_device().gpu_ordinal());
            }

            // Stage input GPU→Host
            std::vector<float> host_send = input_on_gpu ? MPIStager::toHost(input_base)
                                                        : std::vector<float>(input_base->data(), input_base->data() + input_base->numel());

            // Calculate total receive size
            size_t total_recv = 0;
            for (int count_i : scaled_recv_counts)
            {
                total_recv += static_cast<size_t>(count_i);
            }
            std::vector<float> host_recv(total_recv);

            // MPI allgatherv on host memory
            if (!backend->allgatherv(host_send.data(), send_count, host_recv.data(),
                                     scaled_recv_counts, scaled_displacements, dtype))
            {
                LOG_ERROR("CollectiveContext: MPI allgatherv failed during staged operation");
                return false;
            }

            // Stage Host→GPU for output (or copy to CPU tensor)
            if (output_on_gpu)
            {
                MPIStager::toDevice(host_recv, output_base);
            }
            else
            {
                std::memcpy(output_base->mutable_data(), host_recv.data(), host_recv.size() * sizeof(float));
            }

            return true;
        }

        // =========================================================================
        // Standard Path: Direct backend call (NCCL, RCCL, PCIeBAR, or MPI+CPU)
        // =========================================================================
        // Use active_data_ptr() for send and active_mutable_data_ptr() for receive
        // to avoid invalidating GPU data on tensors that should remain on device
        const void *send_buf = local_input->active_data_ptr();
        void *recv_buf = full_output->active_mutable_data_ptr();

        return backend->allgatherv(send_buf, send_count, recv_buf,
                                   scaled_recv_counts, scaled_displacements,
                                   dtype);
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

        // Query buffer for actual dtype
        CollectiveDataType dtype = tensorToCollectiveDataType(buffer);
        size_t actual_count = count > 0 ? count : buffer->numel();

        // =========================================================================
        // MPI Backend with GPU Buffer: Requires Host Staging
        // =========================================================================
        if (backend->type() == CollectiveBackendType::MPI && buffer->is_on_gpu())
        {
            LOG_DEBUG("CollectiveContext: MPI+GPU staging for broadcast (count=" << actual_count << ")");

            // Cast to TensorBase for MPIStager compatibility
            auto *tensor_base = dynamic_cast<TensorBase *>(buffer);
            if (!tensor_base)
            {
                LOG_ERROR("CollectiveContext: Buffer is not TensorBase, cannot stage for MPI");
                return false;
            }

            // Ensure GPU kernels complete before staging
            MPIStager::synchronizeDevice(tensor_base->home_device().gpu_ordinal());

            // Stage GPU→Host (root has data, others will receive)
            std::vector<float> host_buffer = MPIStager::toHost(tensor_base);

            // MPI broadcast on host memory
            if (!backend->broadcast(host_buffer.data(), actual_count, dtype, root_rank))
            {
                LOG_ERROR("CollectiveContext: MPI broadcast failed during staged operation");
                return false;
            }

            // Stage Host→GPU (all ranks copy result back to device)
            MPIStager::toDevice(host_buffer, tensor_base);

            return true;
        }

        // =========================================================================
        // Standard Path: Direct backend call (NCCL, RCCL, PCIeBAR, or MPI+CPU)
        // =========================================================================
        // Use active_mutable_data_ptr() to get GPU pointer without invalidating GPU data
        void *data_ptr = buffer->active_mutable_data_ptr();

        return backend->broadcast(data_ptr, actual_count, dtype, root_rank);
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

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        return static_cast<PCIeBARBackend *>(backend);
#else
        return nullptr;
#endif
    }

    bool CollectiveContext::requiresBufferRegistration() const
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Check if any available backend requires buffer registration
        PCIeBARBackend *bar_backend = getPCIeBarBackend();
        if (bar_backend && bar_backend->requiresBufferRegistration())
        {
            return true;
        }
#endif

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
                .setScope(CollectiveScope::GLOBAL);

            // CRITICAL: Add ALL local devices to enable heterogeneous detection.
            // This allows the backend router to see if this node has both CUDA and ROCm,
            // which indicates potential cross-rank heterogeneity.
            //
            // The local_devices_ list contains all GPUs available to this rank,
            // populated from the cluster_inventory during construction.
            int gpu_count = 0;
            for (const auto &device : local_devices_)
            {
                if (device.type == DeviceType::CUDA || device.type == DeviceType::ROCm)
                {
                    builder.addDevice(device);
                    gpu_count++;
                }
            }

            // Fallback: if no GPUs in local_devices_, add the tensor device
            if (gpu_count == 0)
            {
                builder.addDevice(tensor_device);
                gpu_count = 1;
            }

            // local_rank determines which device this process uses in the group.
            // For multi-process NCCL/RCCL, each rank should use a different GPU.
            // Use MPI rank modulo number of devices to assign GPUs round-robin.
            int mpi_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            int local_rank = mpi_rank % gpu_count;
            builder.setLocalRank(local_rank);
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

        // Skip trivial domains (size 1, no communication needed)
        // Check this BEFORE checking for router - trivial domains don't need a router
        if (domain->isTrivial())
        {
            LOG_DEBUG("CollectiveContext::executeAllreduceInDomain: Domain '" << domain->name
                                                                              << "' is trivial (size=" << domain->domain_size << "), skipping");
            return true;
        }

        // Single-device contexts without a router can only handle trivial domains
        if (!router_)
        {
            LOG_ERROR("CollectiveContext: No router available for domain-aware allreduce "
                      "(domain '"
                      << domain->name << "' has size " << domain->domain_size << ")");
            return false;
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

        // Query buffer for actual dtype
        CollectiveDataType dtype = tensorToCollectiveDataType(buffer);

        // Execute collective
        // CRITICAL: Use active_mutable_data_ptr() to get GPU pointer without invalidating
        // GPU data. Using mutable_data() would mark device_valid_=false, causing expensive
        // re-uploads on subsequent operations that need this tensor on GPU.
        void *data_ptr = buffer->active_mutable_data_ptr();
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

        // Query buffer for actual dtype (use input tensor as reference)
        CollectiveDataType dtype = tensorToCollectiveDataType(local_input);

        // Execute collective
        // Use active_data_ptr() for send and active_mutable_data_ptr() for receive
        // to avoid invalidating GPU data on tensors that should remain on device
        const void *send_buf = local_input->active_data_ptr();
        void *recv_buf = full_output->active_mutable_data_ptr();
        size_t send_count = actual_seq_len > 0 ? actual_seq_len : local_input->numel();

        return backend->allgather(send_buf, recv_buf, send_count, dtype);
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

        // Query buffer for actual dtype (use input tensor as reference)
        CollectiveDataType dtype = tensorToCollectiveDataType(local_input);

        // Execute collective
        // Use active_data_ptr() for send and active_mutable_data_ptr() for receive
        // to avoid invalidating GPU data on tensors that should remain on device
        const void *send_buf = local_input->active_data_ptr();
        void *recv_buf = full_output->active_mutable_data_ptr();

        return backend->allgatherv(send_buf, send_count, recv_buf,
                                   scaled_recv_counts, scaled_displacements,
                                   dtype);
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
