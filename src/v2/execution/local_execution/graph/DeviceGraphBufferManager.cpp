/**
 * @file DeviceGraphBufferManager.cpp
 * @brief Implementation of DeviceGraphBufferManager
 * @author David Sanftenberg
 * @date December 2025
 */

#include "DeviceGraphBufferManager.h"
#include "DeviceGraphExecutor.h"
#include "LivenessAnalyzer.h"
#include "../collective/CollectiveContext.h"
#include "../device/DeviceWorkspaceManager.h"
#include "../../../collective/backends/PCIeBARBackend.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../backends/BackendManager.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../compute_stages/IComputeStage.h"
#include "../../../models/qwen/Qwen2BufferSpec.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include <algorithm>
#include <cctype>

namespace llaminar2
{

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    DeviceGraphBufferManager::DeviceGraphBufferManager(TensorFactory *factory, const MPIContext *mpi_ctx,
                                                       const GraphBufferManagerConfig &config)
        : factory_(factory), mpi_ctx_(mpi_ctx), config_(config)
    {
        if (!factory_)
        {
            LOG_WARN("DeviceGraphBufferManager created with null TensorFactory - allocations will fail");
        }
        if (config_.use_mapped_memory)
        {
            LOG_DEBUG("DeviceGraphBufferManager: Mapped memory mode enabled for activation buffers");
        }
    }

    DeviceGraphBufferManager::~DeviceGraphBufferManager()
    {
        // Unique_ptrs handle cleanup automatically
        // Just log if we're destroying with buffers still allocated
        if (!buffers_.empty())
        {
            LOG_DEBUG("DeviceGraphBufferManager destroying with " << buffers_.size() << " buffers still allocated");
        }
    }

    // =========================================================================
    // Collective Context (Phase 3: Buffer Registration API)
    // =========================================================================

    void DeviceGraphBufferManager::setCollectiveContext(std::shared_ptr<CollectiveContext> ctx)
    {
        collective_ctx_ = std::move(ctx);
        if (collective_ctx_)
        {
            bool requires_reg = collective_ctx_->requiresBufferRegistration();
            LOG_DEBUG("DeviceGraphBufferManager: CollectiveContext set (requires_registration="
                      << (requires_reg ? "true" : "false") << ")");
        }
        else
        {
            LOG_DEBUG("DeviceGraphBufferManager: CollectiveContext cleared");
        }
    }

    bool DeviceGraphBufferManager::shouldUseBarAllocation(const BufferDescriptor &desc) const
    {
        // =====================================================================
        // Path 1: Explicit collective buffer marking (existing behavior)
        // =====================================================================
        // If buffer is explicitly marked as collective, use existing logic
        if (desc.participates_in_collective && !desc.collective_id.empty())
        {
            // Must target a ROCm device (BAR allocation is for AMD GPU memory)
            if (!desc.device.is_rocm())
            {
                return false;
            }

            // Must have a collective context with BAR backend
            if (!collective_ctx_)
            {
                return false;
            }

            // Check if the context has a PCIeBARBackend
            PCIeBARBackend *bar_backend = collective_ctx_->getPCIeBarBackend();
            if (!bar_backend)
            {
                return false;
            }

            // Check if the backend requires registration
            return bar_backend->requiresBufferRegistration();
        }

        // =====================================================================
        // Path 2: Automatic detection via Qwen2BufferSpec (Phase 3)
        // =====================================================================
        // Check if buffer name matches row-parallel outputs AND config enables
        // BAR allocation (LOCAL TP with PCIeBAR backend)
        if (Qwen2BufferSpec::requiresBARBacked(desc.name, config_.collective_backend, config_.tp_degree))
        {
            // Must target a ROCm device for BAR allocation
            if (!desc.device.is_rocm())
            {
                LOG_TRACE("[DeviceGraphBufferManager] Buffer '" << desc.name
                                                                << "' requires BAR but device is not ROCm ("
                                                                << desc.device.toString() << ")");
                return false;
            }

            // Check if we have a collective context with BAR backend
            if (collective_ctx_)
            {
                PCIeBARBackend *bar_backend = collective_ctx_->getPCIeBarBackend();
                if (bar_backend && bar_backend->requiresBufferRegistration())
                {
                    LOG_DEBUG("[DeviceGraphBufferManager] Auto-detected BAR requirement for '"
                              << desc.name << "' (row-parallel output with PCIeBAR backend)");
                    return true;
                }
            }

            // No collective context yet, but config indicates we need BAR allocation
            // This can happen if collective context is set after buffer allocation
            LOG_TRACE("[DeviceGraphBufferManager] Buffer '" << desc.name
                                                            << "' identified as needing BAR by Qwen2BufferSpec, "
                                                            << "but no CollectiveContext with BAR backend available yet");
        }

        return false;
    }

    std::unique_ptr<TensorBase> DeviceGraphBufferManager::allocateFromBarRegion(
        const std::string &node_name,
        const BufferDescriptor &desc)
    {
        if (!collective_ctx_)
        {
            LOG_ERROR("DeviceGraphBufferManager::allocateFromBarRegion: no collective context");
            return nullptr;
        }

        PCIeBARBackend *bar_backend = collective_ctx_->getPCIeBarBackend();
        if (!bar_backend)
        {
            LOG_ERROR("DeviceGraphBufferManager::allocateFromBarRegion: no PCIeBARBackend");
            return nullptr;
        }

        // Calculate size needed
        size_t size_bytes = desc.sizeBytes();
        if (size_bytes == 0)
        {
            LOG_ERROR("DeviceGraphBufferManager::allocateFromBarRegion: zero size for '" << desc.name << "'");
            return nullptr;
        }

        // Allocate from BAR region
        auto result = bar_backend->allocateInBarRegion(size_bytes);
        if (!result)
        {
            LOG_ERROR("DeviceGraphBufferManager::allocateFromBarRegion: BAR allocation failed for '"
                      << desc.name << "' (" << size_bytes << " bytes)");
            return nullptr;
        }

        auto [ptr, offset] = *result;

        // Register the buffer with the backend
        bool registered = bar_backend->registerBuffer(
            desc.collective_id,
            desc.device,
            ptr,
            size_bytes);

        if (!registered)
        {
            LOG_ERROR("DeviceGraphBufferManager::allocateFromBarRegion: buffer registration failed for '"
                      << desc.collective_id << "'");
            bar_backend->freeBarBuffer(ptr);
            return nullptr;
        }

        LOG_DEBUG("[BAR_ALLOC] '" << node_name << "." << desc.name
                  << "' collective='" << desc.collective_id
                  << "' bar_ptr=" << ptr
                  << " offset=" << offset
                  << " bytes=" << size_bytes
                  << " device=" << desc.device.toString());

        // Create a tensor wrapping the external BAR memory
        // Note: For now, we use the standard factory and rely on the caller
        // to handle the external memory correctly. In a full implementation,
        // we would need a factory method like createFromExternalMemory().
        //
        // The tensor created here wraps the BAR pointer. The underlying memory
        // is NOT owned by the tensor - it's owned by the PCIeBARBackend.
        //
        // TODO: Add TensorFactory::createFromExternalMemory() for proper
        // external memory wrapping with custom deallocator.

        auto tensor = createTensorFromDescriptor(desc);
        if (!tensor)
        {
            LOG_ERROR("DeviceGraphBufferManager::allocateFromBarRegion: failed to create tensor wrapper");
            bar_backend->unregisterBuffer(desc.collective_id, desc.device);
            bar_backend->freeBarBuffer(ptr);
            return nullptr;
        }

        // Note: The tensor created above allocates its own memory. For a complete
        // implementation, we would need to create a tensor that wraps the BAR
        // memory directly. For Phase 3, we register the buffer location but
        // the actual memory layout integration happens in Phase 4.

        // Store the BAR pointer association in the registered buffer metadata
        // so that collective operations can find it later.

        return tensor;
    }

    // =========================================================================
    // Allocation
    // =========================================================================

    bool DeviceGraphBufferManager::allocateForGraph(ComputeGraph &graph)
    {
        if (!factory_)
        {
            LOG_ERROR("Cannot allocate buffers: TensorFactory is null");
            return false;
        }

        LOG_DEBUG("DeviceGraphBufferManager: Collecting buffer requirements from graph");

        // Get execution order to process nodes
        auto execution_order = graph.getExecutionOrder();

        size_t stages_with_requirements = 0;
        size_t total_requirements = 0;

        // Collect all stages for workspace allocation
        std::vector<IComputeStage *> all_stages;
        all_stages.reserve(execution_order.size());

        for (const auto &node_name : execution_order)
        {
            auto *node = graph.getNode(node_name);
            if (!node || !node->stage)
            {
                LOG_WARN("DeviceGraphBufferManager: Node '" << node_name << "' has no stage, skipping");
                continue;
            }

            // Collect stage for potential workspace allocation
            all_stages.push_back(node->stage.get());

            // Get requirements from stage
            auto reqs = node->stage->getBufferRequirements();
            if (reqs.empty())
            {
                // Stage hasn't implemented getBufferRequirements() yet
                continue;
            }

            stages_with_requirements++;
            total_requirements += reqs.size();

            // Allocate each buffer
            for (const auto &desc : reqs.buffers)
            {
                // Skip INPUT and WEIGHT - they come from external sources
                if (desc.role == BufferRole::INPUT || desc.role == BufferRole::WEIGHT)
                {
                    // Store descriptor for metadata tracking but don't allocate
                    BufferKey key{node_name, desc.name};
                    descriptors_[key] = desc;
                    continue;
                }

                // Allocate OUTPUT, INOUT, SCRATCH buffers
                if (!allocateBuffer(node_name, desc))
                {
                    LOG_ERROR("DeviceGraphBufferManager: Failed to allocate buffer '"
                              << desc.name << "' for node '" << node_name << "'");
                    return false;
                }
            }
        }

        LOG_DEBUG("DeviceGraphBufferManager: Allocated " << buffers_.size() << " buffers from "
                                                         << stages_with_requirements << " stages ("
                                                         << total_requirements << " requirements total)");
        LOG_DEBUG("DeviceGraphBufferManager: Total allocated: " << (stats_.total_bytes / 1024.0 / 1024.0) << " MB");

        // Allocate GPU workspace for stages that need it (IWorkspaceConsumer implementations)
        if (!all_stages.empty())
        {
            WorkspaceBudgetConfig workspace_config;
            if (!allocateDeviceWorkspace(all_stages, workspace_config))
            {
                LOG_ERROR("DeviceGraphBufferManager: GPU workspace allocation failed!");
                return false;
            }
        }

        return true;
    }

    bool DeviceGraphBufferManager::allocateBuffer(const std::string &node_name, const BufferDescriptor &desc)
    {
        BufferKey key{node_name, desc.name};

        // Check for duplicate
        if (buffers_.find(key) != buffers_.end())
        {
            LOG_WARN("DeviceGraphBufferManager: Buffer '" << desc.name << "' already allocated for node '"
                                                          << node_name << "', skipping");
            return true; // Not a failure, just skip
        }

        std::unique_ptr<TensorBase> tensor;

        // Check if this is a collective buffer that should use BAR allocation
        if (shouldUseBarAllocation(desc))
        {
            tensor = allocateFromBarRegion(node_name, desc);
            if (!tensor)
            {
                // BAR allocation failed - fall back to standard allocation
                LOG_WARN("DeviceGraphBufferManager: BAR allocation failed for '" << desc.name
                                                                                 << "', falling back to standard allocation");
                tensor = createTensorFromDescriptor(desc);
            }
        }
        else
        {
            // Standard allocation path
            tensor = createTensorFromDescriptor(desc);
        }

        if (!tensor)
        {
            LOG_ERROR("DeviceGraphBufferManager: Failed to create tensor for buffer '"
                      << desc.name << "' (shape invalid or allocation failed)");
            return false;
        }

        // Track allocation
        size_t allocated_bytes = tensor->size_bytes();
        updateStats(desc, allocated_bytes);

        // Log pointer addresses for multi-GPU debugging
        LOG_DEBUG("[BUFFER_ALLOC] '" << node_name << "." << desc.name << "'"
                  << " role=" << bufferRoleName(desc.role)
                  << " host_ptr=" << static_cast<const void*>(tensor->raw_data())
                  << " gpu_ptr=" << tensor->gpu_data_ptr()
                  << " device=" << desc.device.toString()
                  << " bytes=" << allocated_bytes
                  << " tensor_obj=" << static_cast<void*>(tensor.get())
                  << (desc.isCollectiveBuffer() ? " [COLLECTIVE]" : ""));

        // Store buffer and descriptor
        descriptors_[key] = desc;
        buffers_[key] = std::move(tensor);

        return true;
    }

    void DeviceGraphBufferManager::releaseAll()
    {
        LOG_DEBUG("DeviceGraphBufferManager: Releasing " << buffers_.size() << " buffers ("
                                                         << (stats_.total_bytes / 1024.0 / 1024.0) << " MB)");

        releaseDeviceWorkspace();
        buffers_.clear();
        descriptors_.clear();
        aliased_buffers_.clear();
        buffer_to_group_.clear();
        aliasing_groups_.clear();
        aliasing_savings_percent_ = 0.0;
        stats_.reset();
    }

    // =========================================================================
    // Aliasing-Aware Allocation
    // =========================================================================

    bool DeviceGraphBufferManager::allocateWithAliasing(ComputeGraph &graph)
    {
        if (!factory_)
        {
            LOG_ERROR("Cannot allocate buffers: TensorFactory is null");
            return false;
        }

        LOG_DEBUG("DeviceGraphBufferManager: Allocating with aliasing optimization");

        // Step 1: Run liveness analysis
        LivenessAnalyzer analyzer;
        auto lifetimes = analyzer.analyze(graph);

        if (lifetimes.empty())
        {
            LOG_DEBUG("DeviceGraphBufferManager: No buffers to allocate (empty graph or no requirements)");
            return true;
        }

        // Step 2: Compute aliasing groups
        aliasing_groups_ = analyzer.computeAliasingGroups(lifetimes);

        // Step 3: Calculate savings
        auto [original_bytes, optimized_bytes] = analyzer.computeMemoryUsage(lifetimes, aliasing_groups_);
        aliasing_savings_percent_ = analyzer.computeSavingsPercent(lifetimes, aliasing_groups_);

        LOG_INFO("DeviceGraphBufferManager: Aliasing analysis complete - "
                 << "original=" << (original_bytes / 1024.0 / 1024.0) << " MB, "
                 << "optimized=" << (optimized_bytes / 1024.0 / 1024.0) << " MB, "
                 << "savings=" << aliasing_savings_percent_ << "%, "
                 << "groups=" << aliasing_groups_.size());

        // Step 4: Allocate aliased SCRATCH buffers
        if (!allocateAliasingGroups(lifetimes))
        {
            LOG_ERROR("DeviceGraphBufferManager: Failed to allocate aliasing groups");
            return false;
        }

        // Step 5: Allocate non-SCRATCH buffers normally
        auto execution_order = graph.getExecutionOrder();
        for (const auto &node_name : execution_order)
        {
            auto *node = graph.getNode(node_name);
            if (!node || !node->stage)
            {
                continue;
            }

            auto reqs = node->stage->getBufferRequirements();
            for (const auto &desc : reqs.buffers)
            {
                // Skip INPUT and WEIGHT (external)
                if (desc.role == BufferRole::INPUT || desc.role == BufferRole::WEIGHT)
                {
                    BufferKey key{node_name, desc.name};
                    descriptors_[key] = desc;
                    continue;
                }

                // SCRATCH buffers handled by aliasing groups
                if (desc.role == BufferRole::SCRATCH)
                {
                    // Already allocated via aliasing groups
                    continue;
                }

                // Allocate OUTPUT and INOUT normally
                if (!allocateBuffer(node_name, desc))
                {
                    LOG_ERROR("DeviceGraphBufferManager: Failed to allocate buffer '"
                              << desc.name << "' for node '" << node_name << "'");
                    return false;
                }
            }
        }

        LOG_DEBUG("DeviceGraphBufferManager: Allocated " << buffers_.size() << " individual buffers + "
                                                         << aliased_buffers_.size() << " aliased groups");

        return true;
    }

    bool DeviceGraphBufferManager::allocateAliasingGroups(const std::vector<BufferLiveness> &lifetimes)
    {
        // Build map from full buffer name to its liveness info (for descriptor lookup)
        std::unordered_map<std::string, const BufferLiveness *> liveness_map;
        for (const auto &l : lifetimes)
        {
            liveness_map[l.buffer_name] = &l;
        }

        // Helper to extract short buffer name from full name (stage_name::buffer_name)
        auto extractShortName = [](const std::string &full_name) -> std::string
        {
            size_t pos = full_name.find("::");
            if (pos != std::string::npos && pos + 2 < full_name.size())
            {
                return full_name.substr(pos + 2);
            }
            return full_name; // Fallback: return as-is
        };

        // Allocate each aliasing group
        for (size_t group_idx = 0; group_idx < aliasing_groups_.size(); ++group_idx)
        {
            const auto &group = aliasing_groups_[group_idx];

            if (group.buffer_names.empty())
            {
                continue;
            }

            // Create physical buffer for the group (sized to max)
            BufferDescriptor group_desc;
            group_desc.name = "aliased_group_" + std::to_string(group_idx);
            group_desc.role = BufferRole::SCRATCH;
            group_desc.tensor_type = group.tensor_type;

            // Infer shape from max size and first buffer's shape ratio
            // For simplicity, use a 1D shape with max_size_bytes / element_size
            size_t element_size = (group.tensor_type == BufferTensorType::FP32) ? 4 : (group.tensor_type == BufferTensorType::BF16 || group.tensor_type == BufferTensorType::FP16) ? 2
                                                                                                                                                                                   : 4;
            size_t num_elements = group.max_size_bytes / element_size;
            group_desc.shape = {num_elements};
            group_desc.device = DeviceId::cpu();

            auto tensor = createTensorFromDescriptor(group_desc);
            if (!tensor)
            {
                LOG_ERROR("DeviceGraphBufferManager: Failed to create aliased buffer for group " << group_idx);
                return false;
            }

            // Track allocation in stats
            stats_.total_bytes += tensor->size_bytes();
            stats_.scratch_bytes += tensor->size_bytes();
            stats_.scratch_buffers_aliased += group.buffer_names.size();

            // Store the physical buffer
            aliased_buffers_[group_idx] = std::move(tensor);

            // Map each logical buffer to this group
            // NOTE: buffer names in group.buffer_names are full names like "stage0::scratch_a"
            for (const auto &full_name : group.buffer_names)
            {
                auto it = liveness_map.find(full_name);
                if (it != liveness_map.end())
                {
                    const auto &liveness = *it->second;
                    // Use stage_name from liveness + short buffer name for the key
                    std::string short_name = extractShortName(full_name);
                    BufferKey key{liveness.stage_name, short_name};
                    buffer_to_group_[key] = group_idx;

                    // Store descriptor for metadata
                    BufferDescriptor desc;
                    desc.name = short_name;
                    desc.role = BufferRole::SCRATCH;
                    desc.tensor_type = liveness.tensor_type;
                    desc.shape = liveness.shape;
                    descriptors_[key] = desc;

                    LOG_TRACE("DeviceGraphBufferManager: Buffer '" << key.node_name << "." << short_name
                                                                   << "' -> aliased group " << group_idx);
                }
            }
        }

        return true;
    }

    // =========================================================================
    // Buffer Retrieval
    // =========================================================================

    TensorBase *DeviceGraphBufferManager::getBuffer(const std::string &node_name, const std::string &buffer_name)
    {
        return getBuffer(BufferKey{node_name, buffer_name});
    }

    TensorBase *DeviceGraphBufferManager::getBuffer(const BufferKey &key)
    {
        // First check individual buffers
        auto it = buffers_.find(key);
        if (it != buffers_.end())
        {
            return it->second.get();
        }

        // Check aliased buffers
        auto group_it = buffer_to_group_.find(key);
        if (group_it != buffer_to_group_.end())
        {
            size_t group_idx = group_it->second;
            auto aliased_it = aliased_buffers_.find(group_idx);
            if (aliased_it != aliased_buffers_.end())
            {
                return aliased_it->second.get();
            }
        }

        return nullptr;
    }

    bool DeviceGraphBufferManager::hasBuffer(const std::string &node_name, const std::string &buffer_name) const
    {
        BufferKey key{node_name, buffer_name};
        // Check both individual buffers and aliased groups
        return buffers_.find(key) != buffers_.end() ||
               buffer_to_group_.find(key) != buffer_to_group_.end();
    }

    std::vector<BufferKey> DeviceGraphBufferManager::getAllBufferKeys() const
    {
        std::vector<BufferKey> keys;
        keys.reserve(buffers_.size());
        for (const auto &pair : buffers_)
        {
            keys.push_back(pair.first);
        }
        return keys;
    }

    // =========================================================================
    // Binding
    // =========================================================================

    bool DeviceGraphBufferManager::bindBuffer(const std::string &node_name,
                                              const std::string &buffer_name,
                                              TensorBase **target_ptr)
    {
        if (!target_ptr)
        {
            LOG_ERROR("DeviceGraphBufferManager::bindBuffer: null target_ptr");
            return false;
        }

        auto *buffer = getBuffer(node_name, buffer_name);
        if (!buffer)
        {
            LOG_ERROR("DeviceGraphBufferManager::bindBuffer: buffer not found: "
                      << node_name << "." << buffer_name);
            return false;
        }

        *target_ptr = buffer;
        return true;
    }

    // =========================================================================
    // Debug
    // =========================================================================

    void DeviceGraphBufferManager::dumpBufferInventory() const
    {
        LOG_INFO("=== DeviceGraphBufferManager Buffer Inventory ===");
        LOG_INFO("Total buffers: " << buffers_.size());
        LOG_INFO("Total allocated: " << (stats_.total_bytes / 1024.0 / 1024.0) << " MB");
        LOG_INFO("  INPUT:   " << (stats_.input_bytes / 1024.0 / 1024.0) << " MB (tracked, not allocated)");
        LOG_INFO("  OUTPUT:  " << (stats_.output_bytes / 1024.0 / 1024.0) << " MB");
        LOG_INFO("  INOUT:   " << (stats_.inout_bytes / 1024.0 / 1024.0) << " MB");
        LOG_INFO("  SCRATCH: " << (stats_.scratch_bytes / 1024.0 / 1024.0) << " MB");
        LOG_INFO("  WEIGHT:  " << (stats_.weight_bytes / 1024.0 / 1024.0) << " MB (tracked, not allocated)");

        if (buffers_.empty())
        {
            LOG_INFO("(no buffers)");
            return;
        }

        // Sort keys for consistent output
        auto keys = getAllBufferKeys();
        std::sort(keys.begin(), keys.end(), [](const BufferKey &a, const BufferKey &b)
                  {
            if (a.node_name != b.node_name) return a.node_name < b.node_name;
            return a.buffer_name < b.buffer_name; });

        for (const auto &key : keys)
        {
            auto it = descriptors_.find(key);
            auto bit = buffers_.find(key);
            if (it != descriptors_.end() && bit != buffers_.end())
            {
                const auto &desc = it->second;
                const auto &tensor = bit->second;
                LOG_INFO("  " << key.node_name << "." << key.buffer_name
                              << " [" << bufferRoleName(desc.role) << "] "
                              << bufferTensorTypeName(desc.tensor_type) << " "
                              << tensor->size_bytes() << " bytes");
            }
        }
        LOG_INFO("============================================");
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    std::unique_ptr<TensorBase> DeviceGraphBufferManager::createTensorFromDescriptor(const BufferDescriptor &desc)
    {
        if (desc.shape.empty())
        {
            LOG_WARN("DeviceGraphBufferManager: Cannot create tensor with empty shape for '"
                     << desc.name << "'");
            return nullptr;
        }

        if (!factory_)
        {
            LOG_ERROR("DeviceGraphBufferManager: TensorFactory is null");
            return nullptr;
        }

        // Convert shape to vector<size_t>
        std::vector<size_t> shape = desc.shape;
        DeviceId device = desc.device;

        switch (desc.tensor_type)
        {
        case BufferTensorType::FP32:
        {
            // =========================================================
            // BAR-Backed Allocation Path (Phase 3)
            // =========================================================
            // Check if this buffer should be BAR-backed for PCIeBAR allreduce
            // Conditions:
            // 1. Buffer identified as row-parallel output by Qwen2BufferSpec
            // 2. TensorFactory can create BAR-backed tensors (P2P configured)
            // 3. Config has valid ROCm and CUDA device pair
            // 4. THIS DEVICE is the ROCm device (not CUDA!)
            //    - CUDA device gets standard FP32Tensor
            //    - ROCm device gets BAR-backed tensor for cross-device access
            if (Qwen2BufferSpec::requiresBARBacked(desc.name, config_.collective_backend, config_.tp_degree) &&
                factory_->canCreateBARBacked() &&
                config_.rocm_device.is_rocm() &&
                config_.cuda_device.is_cuda() &&
                device.is_rocm()) // Only BAR-backed for ROCm device!
            {
                try
                {
                    auto bar_tensor = factory_->createFP32BARBacked(
                        shape, config_.rocm_device, config_.cuda_device);
                    if (bar_tensor)
                    {
                        LOG_DEBUG("DeviceGraphBufferManager: Created BAR-backed FP32 tensor for '"
                                  << desc.name << "' (row-parallel output on ROCm device, PCIeBAR allreduce)");
                        return bar_tensor;
                    }
                }
                catch (const std::exception &e)
                {
                    LOG_WARN("DeviceGraphBufferManager: BAR-backed allocation failed for '"
                             << desc.name << "': " << e.what()
                             << " - falling back to regular allocation");
                }
            }

            // =========================================================
            // Mapped Memory Path (for snapshot/debugging)
            // =========================================================
            // Use mapped memory for FP32 activation buffers when configured
            // This enables zero-copy GPU↔CPU access for snapshot/debugging mode
            if (config_.use_mapped_memory && device.is_gpu())
            {
                auto mapped_tensor = FP32Tensor::createMapped(shape, device);
                if (mapped_tensor && mapped_tensor->isMapped())
                {
                    LOG_DEBUG("DeviceGraphBufferManager: Created mapped FP32 tensor for '"
                              << desc.name << "' on " << device.toString());
                    return mapped_tensor;
                }
                // Fall through to regular allocation if mapped allocation failed
                LOG_WARN("DeviceGraphBufferManager: Mapped allocation failed for '"
                         << desc.name << "', using regular allocation");
            }

            // =========================================================
            // Standard Allocation Path
            // =========================================================
            return factory_->createFP32(shape, device);
        }

        case BufferTensorType::FP16:
            return factory_->createFP16(shape);

        case BufferTensorType::BF16:
            return factory_->createBF16(shape);

        case BufferTensorType::INT32:
            return factory_->createINT32(shape);

        case BufferTensorType::Q8_1:
            return factory_->createQ8_1(shape);

        case BufferTensorType::Q8_0:
        case BufferTensorType::Q4_0:
        case BufferTensorType::IQ4_NL:
            // These quantized types need raw data to create
            // For SCRATCH/OUTPUT buffers, we typically use FP32 or Q8_1
            LOG_WARN("DeviceGraphBufferManager: Quantized type " << bufferTensorTypeName(desc.tensor_type)
                                                                 << " not directly creatable without raw data, "
                                                                 << "falling back to FP32 for buffer '" << desc.name << "'");
            return factory_->createFP32(shape, device);

        case BufferTensorType::UNKNOWN:
        default:
            // Default to FP32
            LOG_DEBUG("DeviceGraphBufferManager: Unknown tensor type, defaulting to FP32 for '"
                      << desc.name << "'");
            return factory_->createFP32(shape, device);
        }
    }

    void DeviceGraphBufferManager::updateStats(const BufferDescriptor &desc, size_t allocated_bytes)
    {
        stats_.total_buffers++;
        stats_.total_bytes += allocated_bytes;

        switch (desc.role)
        {
        case BufferRole::INPUT:
            stats_.input_bytes += allocated_bytes;
            break;
        case BufferRole::OUTPUT:
            stats_.output_bytes += allocated_bytes;
            break;
        case BufferRole::INOUT:
            stats_.inout_bytes += allocated_bytes;
            break;
        case BufferRole::SCRATCH:
            stats_.scratch_bytes += allocated_bytes;
            break;
        case BufferRole::WEIGHT:
            stats_.weight_bytes += allocated_bytes;
            break;
        }
    }

    // =========================================================================
    // GPU Workspace Management (Phase 4: Memory Budget Enforcement)
    // =========================================================================

    size_t DeviceGraphBufferManager::queryAvailableMemory(DeviceId device)
    {
        // Handle invalid device early
        if (!device.is_valid())
        {
            LOG_WARN("[DeviceGraphBufferManager] Cannot query memory for invalid device");
            return 0;
        }

        IBackend *backend = getBackendFor(device);
        if (!backend)
        {
            LOG_WARN("[DeviceGraphBufferManager] No backend available for " << device.toString());
            return 0;
        }

        // For GPU: use gpu_ordinal, for CPU: always device 0 (rank-local view)
        int device_idx = device.is_cpu() ? 0 : device.gpu_ordinal();
        return backend->deviceMemoryFree(device_idx);
    }

    size_t DeviceGraphBufferManager::computeWorkspaceBudget(DeviceId device,
                                                            const WorkspaceBudgetConfig &config)
    {
        size_t available = queryAvailableMemory(device);
        if (available == 0)
        {
            LOG_DEBUG("[DeviceGraphBufferManager] No memory available for " << device.toString());
            return 0;
        }

        // Apply fraction based on device type
        float fraction = device.is_cpu() ? config.cpu_fraction : config.gpu_fraction;
        size_t budget = static_cast<size_t>(static_cast<double>(available) * fraction);

        // Apply headroom
        if (budget > config.headroom)
        {
            budget -= config.headroom;
        }
        else
        {
            budget = 0;
        }

        // Clamp to min/max
        budget = std::max(budget, config.min_budget);
        budget = std::min(budget, config.max_budget);

        LOG_INFO("[DeviceGraphBufferManager] " << device.toString()
                                               << " available=" << (available / (1024 * 1024)) << "MB"
                                               << ", budget=" << (budget / (1024 * 1024)) << "MB"
                                               << " (fraction=" << fraction << ", headroom=" << (config.headroom / (1024 * 1024)) << "MB)");

        return budget;
    }

    DeviceWorkspaceManager *DeviceGraphBufferManager::getDeviceWorkspace(DeviceId device)
    {
        auto it = device_workspaces_.find(device);
        return (it != device_workspaces_.end()) ? it->second.get() : nullptr;
    }

    bool DeviceGraphBufferManager::allocateDeviceWorkspace(
        const std::vector<IComputeStage *> &stages,
        const WorkspaceBudgetConfig &config)
    {
        // Step 1: Collect all unique devices from stages that are workspace consumers
        std::unordered_map<DeviceId, std::vector<IWorkspaceConsumer *>> device_consumers;

        for (auto *stage : stages)
        {
            if (!stage)
            {
                continue;
            }

            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(stage);
            if (consumer)
            {
                DeviceId device = stage->device();
                device_consumers[device].push_back(consumer);
            }
        }

        if (device_consumers.empty())
        {
            LOG_DEBUG("[DeviceGraphBufferManager] No workspace consumers found in " << stages.size() << " stages");
            return true;
        }

        LOG_DEBUG("[DeviceGraphBufferManager] Found " << device_consumers.size()
                                                      << " devices with workspace consumers");

        // Step 2: For each device, allocate workspace and bind to consumers
        for (const auto &[device, consumers] : device_consumers)
        {
            // Skip invalid devices
            if (!device.is_valid())
            {
                LOG_WARN("[DeviceGraphBufferManager] Skipping invalid device from stage");
                continue;
            }

            // Compute budget for this device
            size_t budget = computeWorkspaceBudget(device, config);
            if (budget == 0)
            {
                LOG_WARN("[DeviceGraphBufferManager] Zero budget for " << device.toString()
                                                                       << ", skipping workspace allocation");
                continue;
            }

            // Collect requirements from all consumers on this device
            // Use reasonable default dimensions (consumers can request larger)
            WorkspaceRequirements combined;
            for (auto *consumer : consumers)
            {
                // Get requirements for maximum expected dimensions
                // Default to 4096 for max_m if not specified
                auto reqs = consumer->getWorkspaceRequirements(/*max_m=*/4096);
                combined.merge(reqs);
            }

            // If no requirements after merging, skip this device
            if (combined.buffers.empty())
            {
                LOG_DEBUG("[DeviceGraphBufferManager] No workspace requirements for device "
                          << device.toString());
                continue;
            }

            LOG_DEBUG("[DeviceGraphBufferManager] Device " << device.toString()
                                                           << ": " << consumers.size() << " consumers, "
                                                           << combined.buffers.size() << " buffers, "
                                                           << combined.total_bytes_with_alignment() << " bytes needed");

            // Create manager and allocate
            auto manager = std::make_unique<DeviceWorkspaceManager>(device, budget);
            if (!manager->allocate(combined))
            {
                LOG_ERROR("[DeviceGraphBufferManager] Failed to allocate workspace on "
                          << device.toString()
                          << " (needed=" << combined.total_bytes_with_alignment()
                          << ", budget=" << budget << ")");
                return false;
            }

            // Bind workspace to all consumers on this device
            for (auto *consumer : consumers)
            {
                consumer->bindWorkspace(manager.get());
            }

            LOG_INFO("[DeviceGraphBufferManager] Allocated " << (manager->used() / (1024 * 1024))
                                                             << "MB workspace on " << device.toString()
                                                             << " (" << manager->bufferCount() << " buffers)");

            // Store the manager and budget
            device_workspace_budgets_[device] = budget;
            device_workspaces_[device] = std::move(manager);
        }

        return true;
    }

    size_t DeviceGraphBufferManager::computeModelAwareBudgetFloor(const WorkspaceSizingHints &hints) const
    {
        const int max_seq_len = std::max(1, hints.max_seq_len);
        const int vocab_size = std::max(1, hints.vocab_size);
        const int d_model = std::max(1, hints.d_model);

        const size_t mn_buffer_size = static_cast<size_t>(max_seq_len) * static_cast<size_t>(vocab_size) * sizeof(float);
        const size_t lm_head_workspace = 3 * mn_buffer_size;
        const size_t mk_overhead = static_cast<size_t>(max_seq_len) * static_cast<size_t>(d_model) * sizeof(float) * 2;
        const size_t padded_n_buffer = 8ULL * static_cast<size_t>(vocab_size) * sizeof(float);
        const size_t embed_table_temp = static_cast<size_t>(vocab_size) * static_cast<size_t>(d_model) * sizeof(float);
        const size_t base_workspace = lm_head_workspace + mk_overhead + padded_n_buffer + embed_table_temp;
        const size_t safety_margin = base_workspace / 10;
        const size_t min_budget = 768ULL * 1024 * 1024;
        return std::max(min_budget, base_workspace + safety_margin);
    }

    bool DeviceGraphBufferManager::allocateDeviceWorkspaceForGraph(
        const ComputeGraph &graph,
        const WorkspaceSizingHints &hints,
        const std::vector<WorkspaceConsumerRequest> &extra_consumers,
        const WorkspaceBudgetConfig &config)
    {
        struct ConsumerBinding
        {
            IWorkspaceConsumer *consumer = nullptr;
            int m = 4096;
            int n = 0;
            int k = 0;
        };

        std::unordered_map<DeviceId, std::vector<ConsumerBinding>> consumers_by_device;

        const auto execution_order = graph.getExecutionOrder();
        for (const auto &node_name : execution_order)
        {
            const ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage)
            {
                continue;
            }

            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(node->stage.get());
            if (!consumer)
            {
                continue;
            }

            const DeviceId device = node->device;
            if (!device.is_gpu())
            {
                continue;
            }

            std::string lowered_name = node_name;
            std::transform(
                lowered_name.begin(), lowered_name.end(), lowered_name.begin(),
                [](unsigned char c)
                { return static_cast<char>(std::tolower(c)); });

            const bool is_embedding = (lowered_name == "embedding") || (lowered_name.find("embed") != std::string::npos);
            const bool is_attention = (lowered_name.find("attention") != std::string::npos);

            ConsumerBinding binding;
            binding.consumer = consumer;

            if (is_attention)
            {
                binding.m = std::max(1, hints.batch_size);
                binding.n = std::max(0, hints.n_heads);
                binding.k = std::max(0, hints.head_dim);
            }
            else if (is_embedding)
            {
                binding.m = std::max(1, hints.max_seq_len);
                binding.n = std::max(1, hints.vocab_size);
                binding.k = std::max(0, hints.d_model);
            }
            else
            {
                binding.m = std::max(1, hints.max_seq_len);
                binding.n = 0;
                binding.k = 0;
            }

            consumers_by_device[device].push_back(binding);
        }

        for (const auto &request : extra_consumers)
        {
            if (!request.consumer || !request.device.is_gpu())
            {
                continue;
            }

            consumers_by_device[request.device].push_back(ConsumerBinding{
                request.consumer,
                std::max(1, request.m),
                request.n,
                request.k,
            });
        }

        if (consumers_by_device.empty())
        {
            LOG_DEBUG("[DeviceGraphBufferManager] No GPU workspace consumers found in graph");
            return true;
        }

        const size_t model_floor_budget = computeModelAwareBudgetFloor(hints);

        for (const auto &[device, consumers] : consumers_by_device)
        {
            if (!device.is_valid())
            {
                LOG_WARN("[DeviceGraphBufferManager] Skipping invalid device from graph consumer");
                continue;
            }

            auto existing = device_workspaces_.find(device);
            if (existing != device_workspaces_.end() && existing->second)
            {
                for (const auto &consumer_binding : consumers)
                {
                    consumer_binding.consumer->bindWorkspace(existing->second.get());
                }
                continue;
            }

            size_t budget = computeWorkspaceBudget(device, config);
            budget = std::max(budget, model_floor_budget);

            WorkspaceRequirements combined;
            for (const auto &consumer_binding : consumers)
            {
                auto reqs = consumer_binding.consumer->getWorkspaceRequirements(
                    consumer_binding.m,
                    consumer_binding.n,
                    consumer_binding.k);
                combined.merge(reqs);
            }

            if (combined.buffers.empty())
            {
                LOG_DEBUG("[DeviceGraphBufferManager] No workspace requirements for device "
                          << device.toString());
                continue;
            }

            auto manager = std::make_unique<DeviceWorkspaceManager>(device, budget);
            if (!manager->allocate(combined))
            {
                LOG_ERROR("[DeviceGraphBufferManager] Failed to allocate workspace on "
                          << device.toString()
                          << " (needed=" << combined.total_bytes_with_alignment()
                          << ", budget=" << budget << ")");
                return false;
            }

            for (const auto &consumer_binding : consumers)
            {
                consumer_binding.consumer->bindWorkspace(manager.get());
            }

            LOG_INFO("[DeviceGraphBufferManager] Allocated " << (manager->used() / (1024 * 1024))
                                                             << "MB workspace on " << device.toString()
                                                             << " (" << manager->bufferCount() << " buffers, model-aware budget)");

            device_workspace_budgets_[device] = budget;
            device_workspaces_[device] = std::move(manager);
        }

        return true;
    }

    void DeviceGraphBufferManager::releaseDeviceWorkspace()
    {
        if (!device_workspaces_.empty())
        {
            LOG_DEBUG("[DeviceGraphBufferManager] Releasing " << device_workspaces_.size()
                                                              << " GPU workspace managers");
        }

        // Clear all workspace managers (destructors release memory)
        device_workspaces_.clear();
        device_workspace_budgets_.clear();
    }

    size_t DeviceGraphBufferManager::totalDeviceWorkspaceAllocated() const
    {
        size_t total = 0;
        for (const auto &[device, mgr] : device_workspaces_)
        {
            total += mgr->used();
        }
        return total;
    }

    size_t DeviceGraphBufferManager::deviceWorkspaceAllocated(DeviceId device) const
    {
        auto it = device_workspaces_.find(device);
        return (it != device_workspaces_.end()) ? it->second->used() : 0;
    }

} // namespace llaminar2
