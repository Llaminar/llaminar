/**
 * @file WorkDistributor.cpp
 * @brief Implementation of hierarchical work distribution
 * @author David Sanftenberg
 * @date December 2025
 */

#include "WorkDistributor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <utility>

namespace llaminar2
{

    // =============================================================================
    // Construction
    // =============================================================================

    WorkDistributor::WorkDistributor(Config config)
        : config_(std::move(config))
    {
        if (config_.world_size < 1)
        {
            throw std::invalid_argument("WorkDistributor: world_size must be >= 1");
        }
        if (config_.rank < 0 || config_.rank >= config_.world_size)
        {
            throw std::invalid_argument("WorkDistributor: rank out of range");
        }

        // If no devices specified, assume single CPU device at index 0
        if (config_.devices.empty())
        {
            config_.devices = {0};
        }

        // If weights specified, must match device count
        if (!config_.device_weights.empty() &&
            config_.device_weights.size() != config_.devices.size())
        {
            throw std::invalid_argument("WorkDistributor: device_weights size must match devices size");
        }

        // Validate weights are positive
        for (float w : config_.device_weights)
        {
            if (w <= 0.0f)
            {
                throw std::invalid_argument("WorkDistributor: device_weights must be positive");
            }
        }
    }

    WorkDistributor::WorkDistributor(int world_size, int rank, DeviceId device)
        : WorkDistributor(Config{
              .world_size = world_size,
              .rank = rank,
              .devices = device.is_valid() ? std::vector<int>{device.toLegacyIndex()} : std::vector<int>{0},
              .device_weights = {}})
    {
    }

    // =============================================================================
    // Rank-Level Distribution
    // =============================================================================

    WorkDistributor::WorkSlice WorkDistributor::getRankSlice(size_t total_elements) const
    {
        if (total_elements == 0)
        {
            return WorkSlice{0, 0, 0, config_.rank};
        }

        // Even distribution with remainder to last rank
        size_t base_count = total_elements / config_.world_size;
        size_t remainder = total_elements % config_.world_size;

        // Distribute remainder to first N ranks (more balanced than giving all to last)
        size_t start = 0;
        for (int r = 0; r < config_.rank; ++r)
        {
            size_t this_rank_count = base_count + (static_cast<size_t>(r) < remainder ? 1 : 0);
            start += this_rank_count;
        }

        size_t count = base_count + (static_cast<size_t>(config_.rank) < remainder ? 1 : 0);

        return WorkSlice{
            .start = start,
            .end = start + count,
            .count = count,
            .owner = config_.rank};
    }

    std::vector<WorkDistributor::WorkSlice> WorkDistributor::getAllRankSlices(size_t total_elements) const
    {
        std::vector<WorkSlice> slices;
        slices.reserve(config_.world_size);

        if (total_elements == 0)
        {
            for (int r = 0; r < config_.world_size; ++r)
            {
                slices.push_back(WorkSlice{0, 0, 0, r});
            }
            return slices;
        }

        size_t base_count = total_elements / config_.world_size;
        size_t remainder = total_elements % config_.world_size;
        size_t current_start = 0;

        for (int r = 0; r < config_.world_size; ++r)
        {
            size_t count = base_count + (static_cast<size_t>(r) < remainder ? 1 : 0);
            slices.push_back(WorkSlice{
                .start = current_start,
                .end = current_start + count,
                .count = count,
                .owner = r});
            current_start += count;
        }

        return slices;
    }

    bool WorkDistributor::rankHasWork(size_t total_elements) const
    {
        return getRankSlice(total_elements).count > 0;
    }

    // =============================================================================
    // Device-Level Distribution
    // =============================================================================

    std::vector<float> WorkDistributor::getNormalizedWeights() const
    {
        std::vector<float> weights;

        if (config_.device_weights.empty())
        {
            // Equal distribution
            float equal_weight = 1.0f / config_.devices.size();
            weights.assign(config_.devices.size(), equal_weight);
        }
        else
        {
            // Normalize provided weights
            float sum = std::accumulate(config_.device_weights.begin(),
                                        config_.device_weights.end(), 0.0f);
            weights.reserve(config_.device_weights.size());
            for (float w : config_.device_weights)
            {
                weights.push_back(w / sum);
            }
        }

        return weights;
    }

    WorkDistributor::WorkSlice WorkDistributor::getDeviceSlice(size_t rank_elements, DeviceId device) const
    {
        // Find device position in our list
        int device_idx = device.toLegacyIndex();
        auto it = std::find(config_.devices.begin(), config_.devices.end(), device_idx);
        if (it == config_.devices.end())
        {
            throw std::invalid_argument("WorkDistributor::getDeviceSlice: device not in devices list");
        }
        size_t device_pos = std::distance(config_.devices.begin(), it);

        if (rank_elements == 0)
        {
            return WorkSlice{0, 0, 0, device_idx};
        }

        auto weights = getNormalizedWeights();

        // Compute start and count using weights
        size_t start = 0;
        for (size_t i = 0; i < device_pos; ++i)
        {
            start += static_cast<size_t>(weights[i] * rank_elements);
        }

        size_t end;
        if (device_pos == config_.devices.size() - 1)
        {
            // Last device gets remainder
            end = rank_elements;
        }
        else
        {
            size_t next_start = 0;
            for (size_t i = 0; i <= device_pos; ++i)
            {
                next_start += static_cast<size_t>(weights[i] * rank_elements);
            }
            end = next_start;
        }

        return WorkSlice{
            .start = start,
            .end = end,
            .count = end - start,
            .owner = device_idx};
    }

    std::vector<WorkDistributor::WorkSlice> WorkDistributor::getAllDeviceSlices(size_t rank_elements) const
    {
        std::vector<WorkSlice> slices;
        slices.reserve(config_.devices.size());

        if (rank_elements == 0)
        {
            for (int dev_idx : config_.devices)
            {
                slices.push_back(WorkSlice{0, 0, 0, dev_idx});
            }
            return slices;
        }

        auto weights = getNormalizedWeights();
        size_t current_start = 0;

        for (size_t i = 0; i < config_.devices.size(); ++i)
        {
            size_t count;
            if (i == config_.devices.size() - 1)
            {
                // Last device gets remainder
                count = rank_elements - current_start;
            }
            else
            {
                count = static_cast<size_t>(weights[i] * rank_elements);
            }

            slices.push_back(WorkSlice{
                .start = current_start,
                .end = current_start + count,
                .count = count,
                .owner = config_.devices[i]});
            current_start += count;
        }

        return slices;
    }

    int WorkDistributor::getDeviceForElement(size_t element_idx, size_t rank_elements) const
    {
        auto slices = getAllDeviceSlices(rank_elements);
        for (const auto &slice : slices)
        {
            if (slice.contains(element_idx))
            {
                return slice.owner;
            }
        }
        // Should not reach here if element_idx < rank_elements
        return config_.devices.back();
    }

    // =============================================================================
    // Full Hierarchy Distribution
    // =============================================================================

    std::vector<WorkDistributor::HierarchicalSlice> WorkDistributor::distribute(size_t total_elements) const
    {
        std::vector<HierarchicalSlice> result;

        // Get this rank's slice
        auto rank_slice = getRankSlice(total_elements);
        if (rank_slice.count == 0)
        {
            return result; // This rank has no work
        }

        // Distribute this rank's work across devices
        auto device_slices = getAllDeviceSlices(rank_slice.count);

        result.reserve(device_slices.size());
        for (const auto &dev_slice : device_slices)
        {
            result.push_back(HierarchicalSlice{
                .rank = config_.rank,
                .device = DeviceId::fromLegacyIndex(dev_slice.owner),
                .global_start = rank_slice.start + dev_slice.start,
                .global_end = rank_slice.start + dev_slice.end,
                .local_start = dev_slice.start,
                .local_count = dev_slice.count});
        }

        return result;
    }

    WorkDistributor::HierarchicalSlice WorkDistributor::getPrimaryDeviceSlice(size_t total_elements) const
    {
        auto all_slices = distribute(total_elements);
        if (all_slices.empty())
        {
            return HierarchicalSlice{
                .rank = config_.rank,
                .device = DeviceId::fromLegacyIndex(config_.devices.empty() ? 0 : config_.devices[0]),
                .global_start = 0,
                .global_end = 0,
                .local_start = 0,
                .local_count = 0};
        }
        return all_slices[0];
    }

    // =============================================================================
    // Utility Methods
    // =============================================================================

    size_t WorkDistributor::estimateMemoryPerDevice(size_t total_bytes) const
    {
        if (config_.devices.empty())
        {
            return total_bytes / config_.world_size;
        }

        // Per-rank bytes
        size_t rank_bytes = total_bytes / config_.world_size;

        // Distribute across devices
        auto weights = getNormalizedWeights();
        size_t max_per_device = 0;
        for (float w : weights)
        {
            size_t device_bytes = static_cast<size_t>(w * rank_bytes);
            max_per_device = std::max(max_per_device, device_bytes);
        }

        return max_per_device;
    }

    std::vector<size_t> WorkDistributor::getElementCountsPerDevice(size_t total_elements) const
    {
        auto rank_slice = getRankSlice(total_elements);
        auto device_slices = getAllDeviceSlices(rank_slice.count);

        std::vector<size_t> counts;
        counts.reserve(device_slices.size());
        for (const auto &slice : device_slices)
        {
            counts.push_back(slice.count);
        }

        return counts;
    }

    // =============================================================================
    // MoE Expert Distribution
    // =============================================================================

    std::vector<WorkDistributor::ExpertAssignment> WorkDistributor::distributeExperts(int num_experts) const
    {
        std::vector<ExpertAssignment> assignments;
        assignments.reserve(num_experts);

        if (config_.devices.empty() || num_experts == 0)
        {
            // No devices configured - all experts on invalid device
            for (int e = 0; e < num_experts; ++e)
            {
                assignments.push_back(ExpertAssignment{
                    .expert_id = e,
                    .device = DeviceId::invalid(),
                    .rank = config_.rank});
            }
            return assignments;
        }

        // Distribute experts across devices according to weights
        // This is similar to element distribution but for discrete experts
        auto weights = getNormalizedWeights();
        int num_devices = static_cast<int>(config_.devices.size());

        // Calculate experts per device (round-robin for remainder)
        std::vector<int> experts_per_device(num_devices, 0);
        int assigned = 0;

        for (int d = 0; d < num_devices; ++d)
        {
            // Weighted allocation
            int count = static_cast<int>(weights[d] * num_experts);
            if (d == num_devices - 1)
            {
                // Last device gets remainder
                count = num_experts - assigned;
            }
            experts_per_device[d] = count;
            assigned += count;
        }

        // Create assignments
        int expert_idx = 0;
        for (int d = 0; d < num_devices; ++d)
        {
            for (int i = 0; i < experts_per_device[d]; ++i)
            {
                assignments.push_back(ExpertAssignment{
                    .expert_id = expert_idx++,
                    .device = DeviceId::fromLegacyIndex(config_.devices[d]),
                    .rank = config_.rank});
            }
        }

        return assignments;
    }

    std::vector<WorkDistributor::TokenRouting> WorkDistributor::routeTokensToExperts(
        const float *router_output,
        const std::vector<ExpertAssignment> &expert_assignments,
        int top_k,
        int seq_len,
        int num_experts) const
    {
        // Build expert-to-device lookup
        std::vector<DeviceId> expert_to_device(num_experts, DeviceId::invalid());
        for (const auto &assignment : expert_assignments)
        {
            if (assignment.expert_id >= 0 && assignment.expert_id < num_experts)
            {
                expert_to_device[assignment.expert_id] = assignment.device;
            }
        }

        // For each token, find top-k experts and their weights
        // Using a simple O(seq_len * num_experts) approach for clarity
        std::vector<TokenRouting> routings(num_experts);
        for (int e = 0; e < num_experts; ++e)
        {
            routings[e].expert_id = e;
            routings[e].device = expert_to_device[e];
        }

        for (int s = 0; s < seq_len; ++s)
        {
            const float *scores = router_output + s * num_experts;

            // Find top-k experts for this token
            // Simple selection (could optimize with partial_sort)
            std::vector<std::pair<float, int>> scored_experts;
            scored_experts.reserve(num_experts);
            for (int e = 0; e < num_experts; ++e)
            {
                scored_experts.emplace_back(scores[e], e);
            }

            // Partial sort to get top-k
            std::partial_sort(
                scored_experts.begin(),
                scored_experts.begin() + std::min(top_k, num_experts),
                scored_experts.end(),
                [](const auto &a, const auto &b)
                { return a.first > b.first; });

            // Compute softmax over top-k for weights
            float max_score = scored_experts[0].first;
            float sum_exp = 0.0f;
            std::vector<float> exp_scores(top_k);
            for (int k = 0; k < top_k && k < num_experts; ++k)
            {
                exp_scores[k] = std::exp(scored_experts[k].first - max_score);
                sum_exp += exp_scores[k];
            }

            // Add token to selected experts
            for (int k = 0; k < top_k && k < num_experts; ++k)
            {
                int expert_id = scored_experts[k].second;
                float weight = exp_scores[k] / sum_exp;
                routings[expert_id].token_indices.push_back(s);
                routings[expert_id].weights.push_back(weight);
            }
        }

        // Remove empty routings (experts with no tokens)
        std::vector<TokenRouting> active_routings;
        for (auto &r : routings)
        {
            if (!r.token_indices.empty())
            {
                active_routings.push_back(std::move(r));
            }
        }

        return active_routings;
    }

    std::vector<int> WorkDistributor::getExpertsForDevice(
        const std::vector<ExpertAssignment> &expert_assignments,
        DeviceId device)
    {
        std::vector<int> experts;
        for (const auto &assignment : expert_assignments)
        {
            if (assignment.device == device)
            {
                experts.push_back(assignment.expert_id);
            }
        }
        return experts;
    }

} // namespace llaminar2
