/**
 * @file LivenessAnalyzer.cpp
 * @brief Buffer liveness analysis implementation
 * @author David Sanftenberg
 * @date December 2025
 */

#include "LivenessAnalyzer.h"
#include "DeviceGraphExecutor.h"
#include "../../compute_stages/ComputeStages.h"
#include "../../../utils/Logger.h"
#include <algorithm>
#include <unordered_set>

namespace llaminar2
{

    // =============================================================================
    // Helper: Compute buffer size in bytes
    // =============================================================================

    size_t LivenessAnalyzer::computeBufferSize(const BufferDescriptor &desc)
    {
        if (desc.shape.empty())
        {
            return 0;
        }

        size_t num_elements = 1;
        for (size_t dim : desc.shape)
        {
            num_elements *= dim;
        }

        // Size per element based on tensor type
        size_t element_size = 4; // Default to FP32
        switch (desc.tensor_type)
        {
        case BufferTensorType::FP32:
        case BufferTensorType::INT32:
            element_size = 4;
            break;
        case BufferTensorType::BF16:
        case BufferTensorType::FP16:
            element_size = 2;
            break;
        case BufferTensorType::Q8_0:
        case BufferTensorType::Q8_1:
            // Q8 formats: ~1 byte per element + scale overhead
            element_size = 1;
            break;
        case BufferTensorType::Q4_0:
        case BufferTensorType::IQ4_NL:
            // Q4 formats: ~0.5 bytes per element + scale overhead
            // Round up to be safe
            return (num_elements + 1) / 2 + (num_elements / 32) * 4;
        case BufferTensorType::UNKNOWN:
        default:
            element_size = 4; // Conservative default
            break;
        }

        return num_elements * element_size;
    }

    // =============================================================================
    // Helper: Check tensor type compatibility for aliasing
    // =============================================================================

    bool LivenessAnalyzer::areTypesCompatible(BufferTensorType a, BufferTensorType b)
    {
        // Same type always compatible
        if (a == b)
        {
            return true;
        }

        // All floating point types compatible (FP32, BF16, FP16)
        auto isFloating = [](BufferTensorType t)
        {
            return t == BufferTensorType::FP32 ||
                   t == BufferTensorType::BF16 ||
                   t == BufferTensorType::FP16;
        };

        if (isFloating(a) && isFloating(b))
        {
            return true;
        }

        // Quantized types only compatible with same type
        // (different quantization schemes have different block structures)
        return false;
    }

    // =============================================================================
    // analyze(): Compute buffer lifetimes from graph
    // =============================================================================

    std::vector<BufferLiveness> LivenessAnalyzer::analyze(const ComputeGraph &graph) const
    {
        std::vector<BufferLiveness> lifetimes;

        // Get execution order (topological sort)
        auto exec_order = graph.getExecutionOrder();
        if (exec_order.empty())
        {
            return lifetimes;
        }

        // Map buffer names to their lifetime entries
        std::unordered_map<std::string, size_t> buffer_indices;

        // Process each stage in execution order
        for (size_t stage_idx = 0; stage_idx < exec_order.size(); ++stage_idx)
        {
            const std::string &stage_name = exec_order[stage_idx];
            const ComputeNode *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                continue;
            }

            // Get buffer requirements for this stage
            auto reqs = node->stage->getBufferRequirements();

            for (const auto &buf : reqs.buffers)
            {
                // Create unique buffer name: stage_name::buffer_name
                std::string full_name = stage_name + "::" + buf.name;

                auto it = buffer_indices.find(full_name);
                if (it == buffer_indices.end())
                {
                    // First time seeing this buffer
                    BufferLiveness liveness;
                    liveness.buffer_name = full_name;
                    liveness.stage_name = stage_name;
                    liveness.first_use_idx = stage_idx;
                    liveness.last_use_idx = stage_idx;
                    liveness.role = buf.role;
                    liveness.tensor_type = buf.tensor_type;
                    liveness.shape = buf.shape;
                    liveness.size_bytes = computeBufferSize(buf);

                    buffer_indices[full_name] = lifetimes.size();
                    lifetimes.push_back(std::move(liveness));
                }
                else
                {
                    // Update last use
                    lifetimes[it->second].last_use_idx = stage_idx;
                }
            }
        }

        LOG_DEBUG("[LivenessAnalyzer] Analyzed " << lifetimes.size()
                                                 << " buffers across " << exec_order.size() << " stages");

        return lifetimes;
    }

    // =============================================================================
    // filterScratchBuffers(): Get only SCRATCH buffers
    // =============================================================================

    std::vector<BufferLiveness> LivenessAnalyzer::filterScratchBuffers(
        const std::vector<BufferLiveness> &lifetimes)
    {
        std::vector<BufferLiveness> scratch_only;
        scratch_only.reserve(lifetimes.size());

        for (const auto &l : lifetimes)
        {
            if (l.role == BufferRole::SCRATCH)
            {
                scratch_only.push_back(l);
            }
        }

        return scratch_only;
    }

    // =============================================================================
    // canAlias(): Check if two buffers can share memory
    // =============================================================================

    bool LivenessAnalyzer::canAlias(const BufferLiveness &a, const BufferLiveness &b)
    {
        // Only SCRATCH buffers can alias
        if (a.role != BufferRole::SCRATCH || b.role != BufferRole::SCRATCH)
        {
            return false;
        }

        // Lifetimes must not overlap
        if (a.overlaps(b))
        {
            return false;
        }

        // Types must be compatible
        if (!areTypesCompatible(a.tensor_type, b.tensor_type))
        {
            return false;
        }

        return true;
    }

    // =============================================================================
    // computeAliasingGroups(): Interval graph coloring for optimal aliasing
    // =============================================================================

    std::vector<AliasingGroup> LivenessAnalyzer::computeAliasingGroups(
        const std::vector<BufferLiveness> &lifetimes) const
    {
        // Filter to only SCRATCH buffers
        auto scratch_buffers = filterScratchBuffers(lifetimes);

        if (scratch_buffers.empty())
        {
            return {};
        }

        // Sort by first_use_idx for greedy interval coloring
        std::sort(scratch_buffers.begin(), scratch_buffers.end(),
                  [](const BufferLiveness &a, const BufferLiveness &b)
                  {
                      return a.first_use_idx < b.first_use_idx;
                  });

        std::vector<AliasingGroup> groups;
        std::vector<size_t> group_end_times; // Track when each group's last buffer ends

        for (const auto &buf : scratch_buffers)
        {
            // Try to find an existing group this buffer can join
            bool found_group = false;
            for (size_t g = 0; g < groups.size(); ++g)
            {
                // Check if this buffer starts after the group's last buffer ends
                // AND types are compatible
                if (buf.first_use_idx > group_end_times[g] &&
                    areTypesCompatible(buf.tensor_type, groups[g].tensor_type))
                {
                    // Can join this group
                    groups[g].buffer_names.push_back(buf.buffer_name);
                    groups[g].max_size_bytes = std::max(groups[g].max_size_bytes, buf.size_bytes);
                    group_end_times[g] = buf.last_use_idx;
                    found_group = true;
                    break;
                }
            }

            if (!found_group)
            {
                // Create new group
                AliasingGroup group;
                group.buffer_names.push_back(buf.buffer_name);
                group.max_size_bytes = buf.size_bytes;
                group.tensor_type = buf.tensor_type;
                groups.push_back(std::move(group));
                group_end_times.push_back(buf.last_use_idx);
            }
        }

        LOG_DEBUG("[LivenessAnalyzer] Created " << groups.size()
                                                << " aliasing groups from " << scratch_buffers.size()
                                                << " SCRATCH buffers");

        return groups;
    }

    // =============================================================================
    // computeMemoryUsage(): Calculate original vs optimized memory
    // =============================================================================

    std::pair<size_t, size_t> LivenessAnalyzer::computeMemoryUsage(
        const std::vector<BufferLiveness> &lifetimes,
        const std::vector<AliasingGroup> &groups) const
    {
        // Original: sum of all buffer sizes
        size_t original = 0;
        for (const auto &l : lifetimes)
        {
            original += l.size_bytes;
        }

        // Optimized: non-SCRATCH buffers + aliased group sizes
        size_t optimized = 0;

        // Add non-SCRATCH buffers (cannot be aliased)
        std::unordered_set<std::string> aliased_buffers;
        for (const auto &group : groups)
        {
            for (const auto &name : group.buffer_names)
            {
                aliased_buffers.insert(name);
            }
        }

        for (const auto &l : lifetimes)
        {
            if (l.role != BufferRole::SCRATCH)
            {
                optimized += l.size_bytes;
            }
        }

        // Add aliased group sizes (max of each group)
        for (const auto &group : groups)
        {
            optimized += group.max_size_bytes;
        }

        return {original, optimized};
    }

    // =============================================================================
    // computeSavingsPercent(): Get savings as percentage
    // =============================================================================

    double LivenessAnalyzer::computeSavingsPercent(
        const std::vector<BufferLiveness> &lifetimes,
        const std::vector<AliasingGroup> &groups) const
    {
        auto [original, optimized] = computeMemoryUsage(lifetimes, groups);

        if (original == 0)
        {
            return 0.0;
        }

        return 100.0 * static_cast<double>(original - optimized) / static_cast<double>(original);
    }

} // namespace llaminar2
