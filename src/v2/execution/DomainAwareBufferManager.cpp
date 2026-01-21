/**
 * @file DomainAwareBufferManager.cpp
 * @brief Implementation of domain-aware buffer allocation
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#include "DomainAwareBufferManager.h"
#include "BufferRole.h" // For BufferTensorType
#include "../tensors/TensorFactory.h"
#include "../tensors/cpu/CPUTensors.h"
#include "../utils/Logger.h"
#include <stdexcept>
#include <sstream>

namespace llaminar2
{

    // =============================================================================
    // Construction / Destruction
    // =============================================================================

    DomainAwareBufferManager::DomainAwareBufferManager(DomainAwareBufferConfig config)
        : config_(std::move(config))
    {
        // Create TensorFactory for tensor allocation
        // Use provided MPI context or create a default single-rank context
        if (config_.mpi_ctx)
        {
            tensor_factory_ = std::make_unique<TensorFactory>(*config_.mpi_ctx);
        }
        else
        {
            // Create a minimal MPI context for single-rank operation
            // MPIContext requires explicit rank/world_size
            static MPIContext single_rank_ctx(0, 1); // rank=0, world_size=1
            tensor_factory_ = std::make_unique<TensorFactory>(single_rank_ctx);
        }

        LOG_DEBUG("[DomainAwareBufferManager] Initialized with placement_config="
                  << (config_.placement_config ? "yes" : "no")
                  << ", numa_allocator=" << (config_.numa_allocator ? "yes" : "no")
                  << ", default_device=" << config_.default_device.to_string());
    }

    DomainAwareBufferManager::~DomainAwareBufferManager()
    {
        releaseAll();
    }

    // =============================================================================
    // Layer-Aware Allocation
    // =============================================================================

    TensorBase *DomainAwareBufferManager::allocateForLayer(
        int layer_idx,
        const std::string &buffer_name,
        const std::vector<size_t> &shape,
        BufferTensorType dtype)
    {
        // Determine device for this layer
        DeviceId device = getDeviceForLayer(layer_idx);

        LOG_DEBUG("[DomainAwareBufferManager::allocateForLayer] layer=" << layer_idx
                                                                        << ", buffer=" << buffer_name
                                                                        << ", device=" << device.to_string());

        // Create unique buffer name including layer
        std::string full_name = "layer" + std::to_string(layer_idx) + "_" + buffer_name;

        return allocateOnDevice(device, full_name, shape, dtype);
    }

    // =============================================================================
    // Device-Specific Allocation
    // =============================================================================

    TensorBase *DomainAwareBufferManager::allocateOnDevice(
        DeviceId device,
        const std::string &buffer_name,
        const std::vector<size_t> &shape,
        BufferTensorType dtype)
    {
        // Check for duplicate allocation
        if (hasBuffer(buffer_name))
        {
            LOG_WARN("[DomainAwareBufferManager] Buffer '" << buffer_name
                                                           << "' already exists, returning existing buffer");
            return getBuffer(buffer_name);
        }

        // Create tensor on the target device
        auto tensor = createTensorOnDevice(device, shape, dtype);
        if (!tensor)
        {
            LOG_ERROR("[DomainAwareBufferManager] Failed to allocate buffer '"
                      << buffer_name << "' on device " << device.to_string());
            return nullptr;
        }

        // Compute allocation size for stats
        size_t bytes = computeTensorBytes(shape, dtype);

        // Determine NUMA node for CPU allocations
        int numa_node = -1;
        if (device.is_cpu())
        {
            if (config_.numa_allocator)
            {
                numa_node = config_.default_numa_node >= 0
                                ? config_.default_numa_node
                                : config_.numa_allocator->getCurrentNUMANode();
            }
        }

        // Update statistics
        updateStats(device, bytes, numa_node);

        // Store and return
        TensorBase *result = tensor.get();
        owned_buffers_[buffer_name] = std::move(tensor);

        LOG_DEBUG("[DomainAwareBufferManager::allocateOnDevice] Allocated '" << buffer_name
                                                                             << "' on " << device.to_string() << " (" << bytes << " bytes)");

        return result;
    }

    TensorBase *DomainAwareBufferManager::allocateNUMALocal(
        int numa_node,
        const std::string &buffer_name,
        const std::vector<size_t> &shape,
        BufferTensorType dtype)
    {
        // Check for duplicate allocation
        if (hasBuffer(buffer_name))
        {
            LOG_WARN("[DomainAwareBufferManager] Buffer '" << buffer_name
                                                           << "' already exists, returning existing buffer");
            return getBuffer(buffer_name);
        }

        // Resolve NUMA node (-1 means local node)
        int resolved_node = numa_node;
        if (resolved_node < 0 && config_.numa_allocator)
        {
            resolved_node = config_.numa_allocator->getCurrentNUMANode();
        }

        // Create tensor on CPU
        auto tensor = createTensorOnDevice(DeviceId::cpu(), shape, dtype);
        if (!tensor)
        {
            LOG_ERROR("[DomainAwareBufferManager] Failed to allocate NUMA-local buffer '"
                      << buffer_name << "'");
            return nullptr;
        }

        // If NUMA allocator is available, touch memory to ensure placement
        // Note: TensorFactory already handles NUMA binding, but explicit touch
        // ensures correct placement for first-touch policy
        if (config_.numa_allocator && resolved_node >= 0)
        {
            // The tensor is already allocated, but we could verify placement
            // For now, we trust TensorFactory's NUMA binding
            LOG_TRACE("[DomainAwareBufferManager] NUMA-local allocation on node "
                      << resolved_node);
        }

        // Compute allocation size for stats
        size_t bytes = computeTensorBytes(shape, dtype);

        // Update statistics
        updateStats(DeviceId::cpu(), bytes, resolved_node);

        // Store and return
        TensorBase *result = tensor.get();
        owned_buffers_[buffer_name] = std::move(tensor);

        LOG_DEBUG("[DomainAwareBufferManager::allocateNUMALocal] Allocated '" << buffer_name
                                                                              << "' on NUMA node " << resolved_node << " (" << bytes << " bytes)");

        return result;
    }

    // =============================================================================
    // Query Methods
    // =============================================================================

    DeviceId DomainAwareBufferManager::getDeviceForLayer(int layer_idx) const
    {
        if (config_.placement_config)
        {
            // Use placement configuration
            return config_.placement_config->deviceForLayer(layer_idx);
        }

        // Fall back to default device
        return config_.default_device;
    }

    bool DomainAwareBufferManager::isGPULayer(int layer_idx) const
    {
        return getDeviceForLayer(layer_idx).is_gpu();
    }

    bool DomainAwareBufferManager::isCPULayer(int layer_idx) const
    {
        return getDeviceForLayer(layer_idx).is_cpu();
    }

    // =============================================================================
    // Buffer Retrieval
    // =============================================================================

    TensorBase *DomainAwareBufferManager::getBuffer(const std::string &buffer_name)
    {
        auto it = owned_buffers_.find(buffer_name);
        if (it != owned_buffers_.end())
        {
            return it->second.get();
        }
        return nullptr;
    }

    bool DomainAwareBufferManager::hasBuffer(const std::string &buffer_name) const
    {
        return owned_buffers_.find(buffer_name) != owned_buffers_.end();
    }

    // =============================================================================
    // Lifecycle Management
    // =============================================================================

    void DomainAwareBufferManager::releaseAll()
    {
        LOG_DEBUG("[DomainAwareBufferManager::releaseAll] Releasing "
                  << owned_buffers_.size() << " buffers");

        owned_buffers_.clear();
        stats_.reset();
    }

    // =============================================================================
    // Statistics and Debugging
    // =============================================================================

    void DomainAwareBufferManager::dumpBufferInventory() const
    {
        LOG_INFO("[DomainAwareBufferManager] Buffer Inventory:");
        LOG_INFO("  Total buffers: " << stats_.total_buffers());
        LOG_INFO("  Total bytes: " << stats_.total_bytes());
        LOG_INFO("  GPU buffers: " << stats_.gpu_buffer_count
                                   << " (" << stats_.gpu_bytes_allocated << " bytes)");
        LOG_INFO("  CPU buffers: " << stats_.cpu_buffer_count
                                   << " (" << stats_.cpu_bytes_allocated << " bytes)");

        // Per-device breakdown
        for (const auto &[device, bytes] : stats_.bytes_per_device)
        {
            int count = stats_.buffers_per_device.count(device)
                            ? stats_.buffers_per_device.at(device)
                            : 0;
            LOG_INFO("  " << device.to_string() << ": " << count
                          << " buffers (" << bytes << " bytes)");
        }

        // NUMA breakdown
        if (!stats_.bytes_per_numa_node.empty())
        {
            LOG_INFO("  NUMA node breakdown:");
            for (const auto &[node, bytes] : stats_.bytes_per_numa_node)
            {
                LOG_INFO("    Node " << node << ": " << bytes << " bytes");
            }
        }

        // Individual buffers
        LOG_DEBUG("  Individual buffers:");
        for (const auto &[name, tensor] : owned_buffers_)
        {
            if (tensor)
            {
                LOG_DEBUG("    " << name << ": " << tensor->size_bytes() << " bytes"
                                 << ", device=" << tensor->home_device().to_string());
            }
        }
    }

    // =============================================================================
    // Internal Helpers
    // =============================================================================

    std::unique_ptr<TensorBase> DomainAwareBufferManager::createTensorOnDevice(
        DeviceId device,
        const std::vector<size_t> &shape,
        BufferTensorType dtype)
    {
        switch (dtype)
        {
        case BufferTensorType::FP32:
            return tensor_factory_->createFP32(shape, device);

        case BufferTensorType::FP16:
            return tensor_factory_->createFP16(shape);

        case BufferTensorType::BF16:
            return tensor_factory_->createBF16(shape);

        case BufferTensorType::Q8_1:
            return tensor_factory_->createQ8_1(shape, device);

        case BufferTensorType::INT32:
            return tensor_factory_->createINT32(shape);

        default:
            LOG_ERROR("[DomainAwareBufferManager] Unsupported dtype for allocation: "
                      << static_cast<int>(dtype));
            return nullptr;
        }
    }

    void DomainAwareBufferManager::updateStats(DeviceId device, size_t bytes, int numa_node)
    {
        // Update aggregate stats
        if (device.is_gpu())
        {
            stats_.gpu_bytes_allocated += bytes;
            stats_.gpu_buffer_count++;
        }
        else
        {
            stats_.cpu_bytes_allocated += bytes;
            stats_.cpu_buffer_count++;

            // Track NUMA node if known
            if (numa_node >= 0)
            {
                stats_.bytes_per_numa_node[numa_node] += bytes;
            }
        }

        // Update per-device stats
        stats_.bytes_per_device[device] += bytes;
        stats_.buffers_per_device[device]++;
    }

    size_t DomainAwareBufferManager::computeTensorBytes(
        const std::vector<size_t> &shape,
        BufferTensorType dtype) const
    {
        size_t numel = 1;
        for (size_t dim : shape)
        {
            numel *= dim;
        }

        size_t element_size = 1;
        switch (dtype)
        {
        case BufferTensorType::FP32:
        case BufferTensorType::INT32:
            element_size = 4;
            break;
        case BufferTensorType::FP16:
        case BufferTensorType::BF16:
            element_size = 2;
            break;
        case BufferTensorType::Q8_1:
            // Q8_1 has 32 elements per block, each block is 36 bytes
            return ((numel + 31) / 32) * 36;
        case BufferTensorType::Q8_0:
            // Q8_0 has 32 elements per block, each block is 34 bytes
            return ((numel + 31) / 32) * 34;
        case BufferTensorType::Q4_0:
        case BufferTensorType::IQ4_NL:
            // Q4 has 32 elements per block, each block is 18 bytes
            return ((numel + 31) / 32) * 18;
        default:
            element_size = 4; // Default to FP32
            break;
        }

        return numel * element_size;
    }

} // namespace llaminar2
