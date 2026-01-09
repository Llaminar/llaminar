/**
 * @file GraphBufferManager.cpp
 * @brief Implementation of GraphBufferManager
 * @author David Sanftenberg
 * @date December 2025
 */

#include "GraphBufferManager.h"
#include "GraphExecutor.h"
#include "LivenessAnalyzer.h"
#include "../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    GraphBufferManager::GraphBufferManager(TensorFactory *factory, const MPIContext *mpi_ctx)
        : factory_(factory), mpi_ctx_(mpi_ctx)
    {
        if (!factory_)
        {
            LOG_WARN("GraphBufferManager created with null TensorFactory - allocations will fail");
        }
    }

    GraphBufferManager::~GraphBufferManager()
    {
        // Unique_ptrs handle cleanup automatically
        // Just log if we're destroying with buffers still allocated
        if (!buffers_.empty())
        {
            LOG_DEBUG("GraphBufferManager destroying with " << buffers_.size() << " buffers still allocated");
        }
    }

    // =========================================================================
    // Allocation
    // =========================================================================

    bool GraphBufferManager::allocateForGraph(ComputeGraph &graph)
    {
        if (!factory_)
        {
            LOG_ERROR("Cannot allocate buffers: TensorFactory is null");
            return false;
        }

        LOG_DEBUG("GraphBufferManager: Collecting buffer requirements from graph");

        // Get execution order to process nodes
        auto execution_order = graph.getExecutionOrder();

        size_t stages_with_requirements = 0;
        size_t total_requirements = 0;

        for (const auto &node_name : execution_order)
        {
            auto *node = graph.getNode(node_name);
            if (!node || !node->stage)
            {
                LOG_WARN("GraphBufferManager: Node '" << node_name << "' has no stage, skipping");
                continue;
            }

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
                    LOG_ERROR("GraphBufferManager: Failed to allocate buffer '"
                              << desc.name << "' for node '" << node_name << "'");
                    return false;
                }
            }
        }

        LOG_DEBUG("GraphBufferManager: Allocated " << buffers_.size() << " buffers from "
                                                   << stages_with_requirements << " stages ("
                                                   << total_requirements << " requirements total)");
        LOG_DEBUG("GraphBufferManager: Total allocated: " << (stats_.total_bytes / 1024.0 / 1024.0) << " MB");

        return true;
    }

    bool GraphBufferManager::allocateBuffer(const std::string &node_name, const BufferDescriptor &desc)
    {
        BufferKey key{node_name, desc.name};

        // Check for duplicate
        if (buffers_.find(key) != buffers_.end())
        {
            LOG_WARN("GraphBufferManager: Buffer '" << desc.name << "' already allocated for node '"
                                                    << node_name << "', skipping");
            return true; // Not a failure, just skip
        }

        // Create tensor from descriptor
        auto tensor = createTensorFromDescriptor(desc);
        if (!tensor)
        {
            LOG_ERROR("GraphBufferManager: Failed to create tensor for buffer '"
                      << desc.name << "' (shape invalid or allocation failed)");
            return false;
        }

        // Track allocation
        size_t allocated_bytes = tensor->size_bytes();
        updateStats(desc, allocated_bytes);

        // Store buffer and descriptor
        descriptors_[key] = desc;
        buffers_[key] = std::move(tensor);

        LOG_TRACE("GraphBufferManager: Allocated " << bufferRoleName(desc.role) << " buffer '"
                                                   << node_name << "." << desc.name << "' ("
                                                   << allocated_bytes << " bytes)");

        return true;
    }

    void GraphBufferManager::releaseAll()
    {
        LOG_DEBUG("GraphBufferManager: Releasing " << buffers_.size() << " buffers ("
                                                   << (stats_.total_bytes / 1024.0 / 1024.0) << " MB)");

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

    bool GraphBufferManager::allocateWithAliasing(ComputeGraph &graph)
    {
        if (!factory_)
        {
            LOG_ERROR("Cannot allocate buffers: TensorFactory is null");
            return false;
        }

        LOG_DEBUG("GraphBufferManager: Allocating with aliasing optimization");

        // Step 1: Run liveness analysis
        LivenessAnalyzer analyzer;
        auto lifetimes = analyzer.analyze(graph);

        if (lifetimes.empty())
        {
            LOG_DEBUG("GraphBufferManager: No buffers to allocate (empty graph or no requirements)");
            return true;
        }

        // Step 2: Compute aliasing groups
        aliasing_groups_ = analyzer.computeAliasingGroups(lifetimes);

        // Step 3: Calculate savings
        auto [original_bytes, optimized_bytes] = analyzer.computeMemoryUsage(lifetimes, aliasing_groups_);
        aliasing_savings_percent_ = analyzer.computeSavingsPercent(lifetimes, aliasing_groups_);

        LOG_INFO("GraphBufferManager: Aliasing analysis complete - "
                 << "original=" << (original_bytes / 1024.0 / 1024.0) << " MB, "
                 << "optimized=" << (optimized_bytes / 1024.0 / 1024.0) << " MB, "
                 << "savings=" << aliasing_savings_percent_ << "%, "
                 << "groups=" << aliasing_groups_.size());

        // Step 4: Allocate aliased SCRATCH buffers
        if (!allocateAliasingGroups(lifetimes))
        {
            LOG_ERROR("GraphBufferManager: Failed to allocate aliasing groups");
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
                    LOG_ERROR("GraphBufferManager: Failed to allocate buffer '"
                              << desc.name << "' for node '" << node_name << "'");
                    return false;
                }
            }
        }

        LOG_DEBUG("GraphBufferManager: Allocated " << buffers_.size() << " individual buffers + "
                                                   << aliased_buffers_.size() << " aliased groups");

        return true;
    }

    bool GraphBufferManager::allocateAliasingGroups(const std::vector<BufferLiveness> &lifetimes)
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
                LOG_ERROR("GraphBufferManager: Failed to create aliased buffer for group " << group_idx);
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

                    LOG_TRACE("GraphBufferManager: Buffer '" << key.node_name << "." << short_name
                                                             << "' -> aliased group " << group_idx);
                }
            }
        }

        return true;
    }

    // =========================================================================
    // Buffer Retrieval
    // =========================================================================

    TensorBase *GraphBufferManager::getBuffer(const std::string &node_name, const std::string &buffer_name)
    {
        return getBuffer(BufferKey{node_name, buffer_name});
    }

    TensorBase *GraphBufferManager::getBuffer(const BufferKey &key)
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

    bool GraphBufferManager::hasBuffer(const std::string &node_name, const std::string &buffer_name) const
    {
        BufferKey key{node_name, buffer_name};
        // Check both individual buffers and aliased groups
        return buffers_.find(key) != buffers_.end() ||
               buffer_to_group_.find(key) != buffer_to_group_.end();
    }

    std::vector<BufferKey> GraphBufferManager::getAllBufferKeys() const
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

    bool GraphBufferManager::bindBuffer(const std::string &node_name,
                                        const std::string &buffer_name,
                                        TensorBase **target_ptr)
    {
        if (!target_ptr)
        {
            LOG_ERROR("GraphBufferManager::bindBuffer: null target_ptr");
            return false;
        }

        auto *buffer = getBuffer(node_name, buffer_name);
        if (!buffer)
        {
            LOG_ERROR("GraphBufferManager::bindBuffer: buffer not found: "
                      << node_name << "." << buffer_name);
            return false;
        }

        *target_ptr = buffer;
        return true;
    }

    // =========================================================================
    // Debug
    // =========================================================================

    void GraphBufferManager::dumpBufferInventory() const
    {
        LOG_INFO("=== GraphBufferManager Buffer Inventory ===");
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

    std::unique_ptr<TensorBase> GraphBufferManager::createTensorFromDescriptor(const BufferDescriptor &desc)
    {
        if (desc.shape.empty())
        {
            LOG_WARN("GraphBufferManager: Cannot create tensor with empty shape for '"
                     << desc.name << "'");
            return nullptr;
        }

        if (!factory_)
        {
            LOG_ERROR("GraphBufferManager: TensorFactory is null");
            return nullptr;
        }

        // Convert shape to vector<size_t>
        std::vector<size_t> shape = desc.shape;
        DeviceId device = desc.device;

        switch (desc.tensor_type)
        {
        case BufferTensorType::FP32:
            return factory_->createFP32(shape, device);

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
            LOG_WARN("GraphBufferManager: Quantized type " << bufferTensorTypeName(desc.tensor_type)
                                                           << " not directly creatable without raw data, "
                                                           << "falling back to FP32 for buffer '" << desc.name << "'");
            return factory_->createFP32(shape, device);

        case BufferTensorType::UNKNOWN:
        default:
            // Default to FP32
            LOG_DEBUG("GraphBufferManager: Unknown tensor type, defaulting to FP32 for '"
                      << desc.name << "'");
            return factory_->createFP32(shape, device);
        }
    }

    void GraphBufferManager::updateStats(const BufferDescriptor &desc, size_t allocated_bytes)
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

} // namespace llaminar2
