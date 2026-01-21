/**
 * @file TensorParallelConfig.cpp
 * @brief Implementation of tensor parallelism configuration
 */

#include "TensorParallelConfig.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>

namespace llaminar2
{

    // =========================================================================
    // DeviceShardingAssignment
    // =========================================================================

    std::string DeviceShardingAssignment::toString() const
    {
        std::ostringstream oss;
        oss << "Rank " << local_rank << " (" << device.to_string() << "): "
            << "heads[" << head_start << ":" << headEnd() << "]=" << head_count
            << ", kv_heads[" << kv_head_start << ":" << kvHeadEnd() << "]=" << kv_head_count
            << ", d_ff[" << d_ff_start << ":" << dFFEnd() << "]=" << d_ff_count
            << ", vocab[" << vocab_start << ":" << vocabEnd() << "]=" << vocab_count
            << " (" << (work_fraction * 100.0f) << "%)";
        return oss.str();
    }

    // =========================================================================
    // TensorParallelConfig - Constructor
    // =========================================================================

    TensorParallelConfig::TensorParallelConfig(std::vector<DeviceShardingAssignment> assignments)
        : assignments_(std::move(assignments))
    {
        computeTotals();

        // Detect if proportional (check if all work fractions are equal)
        if (assignments_.size() > 1)
        {
            float first_fraction = assignments_[0].work_fraction;
            float tolerance = 0.01f; // 1% tolerance
            is_proportional_ = std::any_of(
                assignments_.begin() + 1, assignments_.end(),
                [first_fraction, tolerance](const DeviceShardingAssignment &a)
                {
                    return std::abs(a.work_fraction - first_fraction) > tolerance;
                });
        }
    }

    // =========================================================================
    // TensorParallelConfig - Accessors
    // =========================================================================

    const DeviceShardingAssignment &TensorParallelConfig::forDevice(DeviceId device) const
    {
        auto it = std::find_if(
            assignments_.begin(), assignments_.end(),
            [device](const DeviceShardingAssignment &a)
            { return a.device == device; });

        if (it == assignments_.end())
        {
            throw std::out_of_range(
                "Device not found in TensorParallelConfig: " + device.to_string());
        }
        return *it;
    }

    const DeviceShardingAssignment &TensorParallelConfig::forRank(int rank) const
    {
        if (rank < 0 || rank >= static_cast<int>(assignments_.size()))
        {
            throw std::out_of_range(
                "Rank " + std::to_string(rank) + " out of bounds [0, " +
                std::to_string(assignments_.size()) + ")");
        }
        return assignments_[rank];
    }

    // =========================================================================
    // TensorParallelConfig - Validation
    // =========================================================================

    void TensorParallelConfig::computeTotals()
    {
        total_heads_ = 0;
        total_kv_heads_ = 0;
        total_d_ff_ = 0;
        total_vocab_ = 0;

        for (const auto &a : assignments_)
        {
            total_heads_ += a.head_count;
            total_kv_heads_ += a.kv_head_count;
            total_d_ff_ += a.d_ff_count;
            total_vocab_ += a.vocab_count;
        }
    }

    bool TensorParallelConfig::validate() const
    {
        return validationError().empty();
    }

    std::string TensorParallelConfig::validationError() const
    {
        if (assignments_.empty())
        {
            return "No device assignments";
        }

        // Check for valid devices
        for (size_t i = 0; i < assignments_.size(); ++i)
        {
            if (!assignments_[i].device.is_valid())
            {
                return "Assignment " + std::to_string(i) + " has invalid device";
            }
        }

        // Check for duplicate devices
        for (size_t i = 0; i < assignments_.size(); ++i)
        {
            for (size_t j = i + 1; j < assignments_.size(); ++j)
            {
                if (assignments_[i].device == assignments_[j].device)
                {
                    return "Duplicate device: " + assignments_[i].device.to_string();
                }
            }
        }

        // Check for overlapping head ranges
        for (size_t i = 0; i < assignments_.size(); ++i)
        {
            for (size_t j = i + 1; j < assignments_.size(); ++j)
            {
                const auto &a = assignments_[i];
                const auto &b = assignments_[j];

                // Check Q head overlap
                if (a.head_start < b.headEnd() && b.head_start < a.headEnd())
                {
                    return "Overlapping Q head ranges between rank " +
                           std::to_string(i) + " and " + std::to_string(j);
                }

                // Check KV head overlap
                if (a.kv_head_start < b.kvHeadEnd() && b.kv_head_start < a.kvHeadEnd())
                {
                    return "Overlapping KV head ranges between rank " +
                           std::to_string(i) + " and " + std::to_string(j);
                }

                // Check d_ff overlap
                if (a.d_ff_start < b.dFFEnd() && b.d_ff_start < a.dFFEnd())
                {
                    return "Overlapping d_ff ranges between rank " +
                           std::to_string(i) + " and " + std::to_string(j);
                }

                // Check vocab overlap
                if (a.vocab_start < b.vocabEnd() && b.vocab_start < a.vocabEnd())
                {
                    return "Overlapping vocab ranges between rank " +
                           std::to_string(i) + " and " + std::to_string(j);
                }
            }
        }

        // Check contiguity (ranges should be contiguous with no gaps)
        std::vector<const DeviceShardingAssignment *> sorted;
        for (const auto &a : assignments_)
        {
            sorted.push_back(&a);
        }

        // Sort by head_start to check contiguity
        std::sort(sorted.begin(), sorted.end(),
                  [](const DeviceShardingAssignment *a, const DeviceShardingAssignment *b)
                  {
                      return a->head_start < b->head_start;
                  });

        // First assignment should start at 0
        if (sorted[0]->head_start != 0)
        {
            return "Head ranges do not start at 0 (starts at " +
                   std::to_string(sorted[0]->head_start) + ")";
        }

        // Each subsequent assignment should start where previous ended
        for (size_t i = 1; i < sorted.size(); ++i)
        {
            if (sorted[i]->head_start != sorted[i - 1]->headEnd())
            {
                return "Gap in head ranges between " +
                       std::to_string(sorted[i - 1]->headEnd()) + " and " +
                       std::to_string(sorted[i]->head_start);
            }
        }

        return ""; // No error
    }

    // =========================================================================
    // TensorParallelConfig - Factory Methods
    // =========================================================================

    TensorParallelConfig TensorParallelConfig::equalSplit(
        int world_size,
        int n_heads,
        int n_kv_heads,
        int d_ff,
        int vocab_size,
        std::optional<std::vector<DeviceId>> devices)
    {
        if (world_size <= 0)
        {
            throw std::invalid_argument("world_size must be positive");
        }
        if (n_heads <= 0 || n_kv_heads <= 0 || d_ff <= 0 || vocab_size <= 0)
        {
            throw std::invalid_argument("All dimension parameters must be positive");
        }
        if (n_heads < world_size)
        {
            throw std::invalid_argument("Cannot split " + std::to_string(n_heads) +
                                        " heads across " + std::to_string(world_size) + " devices");
        }

        // Create uniform fractions
        std::vector<float> fractions(world_size, 1.0f / world_size);

        // Create default devices if not provided
        std::vector<DeviceId> device_list;
        if (devices.has_value())
        {
            device_list = devices.value();
            if (static_cast<int>(device_list.size()) != world_size)
            {
                throw std::invalid_argument("devices.size() must equal world_size");
            }
        }
        else
        {
            for (int i = 0; i < world_size; ++i)
            {
                device_list.push_back(DeviceId::cuda(i));
            }
        }

        // Use proportionalSplit with equal fractions
        auto config = proportionalSplit(device_list, fractions, n_heads, n_kv_heads, d_ff, vocab_size);

        // Mark as non-proportional (equal split)
        config.is_proportional_ = false;

        return config;
    }

    TensorParallelConfig TensorParallelConfig::proportionalSplit(
        const std::vector<DeviceId> &devices,
        const std::vector<float> &work_fractions,
        int n_heads,
        int n_kv_heads,
        int d_ff,
        int vocab_size)
    {
        if (devices.size() != work_fractions.size())
        {
            throw std::invalid_argument("devices.size() != work_fractions.size()");
        }
        if (devices.empty())
        {
            throw std::invalid_argument("devices must not be empty");
        }
        if (n_heads <= 0 || n_kv_heads <= 0 || d_ff <= 0 || vocab_size <= 0)
        {
            throw std::invalid_argument("All dimension parameters must be positive");
        }

        // Validate work fractions
        for (float f : work_fractions)
        {
            if (f < 0.0f)
            {
                throw std::invalid_argument("Work fractions must be non-negative");
            }
        }

        // Normalize work fractions to sum to 1.0
        float sum = std::accumulate(work_fractions.begin(), work_fractions.end(), 0.0f);
        if (sum <= 0.0f)
        {
            throw std::invalid_argument("Sum of work fractions must be positive");
        }

        std::vector<float> normalized(work_fractions.size());
        for (size_t i = 0; i < work_fractions.size(); ++i)
        {
            normalized[i] = work_fractions[i] / sum;
        }

        // Distribute heads, KV heads, d_ff, and vocab proportionally
        std::vector<int> head_counts = distributeProportionally(n_heads, normalized, 1);
        std::vector<int> kv_head_counts = distributeProportionally(n_kv_heads, normalized, 1);
        std::vector<int> d_ff_counts = distributeProportionally(d_ff, normalized, 32); // 32-aligned
        std::vector<int> vocab_counts = distributeProportionally(vocab_size, normalized, 1);

        // Build assignments
        std::vector<DeviceShardingAssignment> assignments;
        int head_offset = 0;
        int kv_offset = 0;
        int d_ff_offset = 0;
        int vocab_offset = 0;

        for (size_t i = 0; i < devices.size(); ++i)
        {
            DeviceShardingAssignment a;
            a.device = devices[i];
            a.local_rank = static_cast<int>(i);
            a.work_fraction = normalized[i];

            a.head_start = head_offset;
            a.head_count = head_counts[i];
            head_offset += head_counts[i];

            a.kv_head_start = kv_offset;
            a.kv_head_count = kv_head_counts[i];
            kv_offset += kv_head_counts[i];

            a.d_ff_start = d_ff_offset;
            a.d_ff_count = d_ff_counts[i];
            d_ff_offset += d_ff_counts[i];

            a.vocab_start = vocab_offset;
            a.vocab_count = vocab_counts[i];
            vocab_offset += vocab_counts[i];

            assignments.push_back(a);
        }

        TensorParallelConfig config(std::move(assignments));
        config.is_proportional_ = true;
        return config;
    }

    TensorParallelConfig TensorParallelConfig::singleDevice(
        DeviceId device,
        int n_heads,
        int n_kv_heads,
        int d_ff,
        int vocab_size)
    {
        DeviceShardingAssignment a;
        a.device = device;
        a.local_rank = 0;
        a.work_fraction = 1.0f;
        a.head_start = 0;
        a.head_count = n_heads;
        a.kv_head_start = 0;
        a.kv_head_count = n_kv_heads;
        a.d_ff_start = 0;
        a.d_ff_count = d_ff;
        a.vocab_start = 0;
        a.vocab_count = vocab_size;

        return TensorParallelConfig({a});
    }

    // =========================================================================
    // TensorParallelConfig - Private Helpers
    // =========================================================================

    int TensorParallelConfig::roundToAlignment(int value, int alignment)
    {
        if (alignment <= 1)
            return value;
        return (value / alignment) * alignment;
    }

    std::vector<int> TensorParallelConfig::distributeProportionally(
        int total,
        const std::vector<float> &fractions,
        int alignment)
    {
        size_t n = fractions.size();
        std::vector<int> result(n, 0);

        if (n == 0 || total <= 0)
        {
            return result;
        }

        // First pass: assign floor of proportional amounts (with alignment)
        int assigned = 0;
        for (size_t i = 0; i < n; ++i)
        {
            int ideal = static_cast<int>(std::floor(fractions[i] * total));
            if (alignment > 1)
            {
                ideal = roundToAlignment(ideal, alignment);
            }
            result[i] = ideal;
            assigned += ideal;
        }

        // Second pass: distribute remainder to devices with largest fractional parts
        // But respect alignment - only add alignment-sized chunks
        int remainder = total - assigned;

        if (alignment > 1)
        {
            // With alignment, we need to be more careful
            // Distribute full alignment chunks, prioritizing devices by fraction
            while (remainder >= alignment)
            {
                // Find device with smallest current assignment relative to its fraction
                size_t best_idx = 0;
                float best_deficit = -1.0f;
                for (size_t i = 0; i < n; ++i)
                {
                    float expected = fractions[i] * total;
                    float deficit = expected - result[i];
                    if (deficit > best_deficit)
                    {
                        best_deficit = deficit;
                        best_idx = i;
                    }
                }
                result[best_idx] += alignment;
                remainder -= alignment;
            }

            // Handle any sub-alignment remainder by giving to first device
            // This ensures we don't lose elements
            if (remainder > 0)
            {
                result[0] += remainder;
            }
        }
        else
        {
            // No alignment constraint - distribute one at a time
            // Sort indices by fractional part (descending)
            std::vector<std::pair<float, size_t>> fractional_parts;
            for (size_t i = 0; i < n; ++i)
            {
                float ideal = fractions[i] * total;
                float frac = ideal - std::floor(ideal);
                fractional_parts.push_back({frac, i});
            }
            std::sort(fractional_parts.begin(), fractional_parts.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

            // Assign remainder to devices with largest fractional parts
            for (int r = 0; r < remainder && r < static_cast<int>(n); ++r)
            {
                result[fractional_parts[r].second]++;
            }
        }

        return result;
    }

    std::string TensorParallelConfig::toString() const
    {
        std::ostringstream oss;
        oss << "TensorParallelConfig(world_size=" << worldSize()
            << ", proportional=" << (is_proportional_ ? "true" : "false")
            << ", heads=" << total_heads_
            << ", kv_heads=" << total_kv_heads_
            << ", d_ff=" << total_d_ff_
            << ", vocab=" << total_vocab_ << "):\n";
        for (const auto &a : assignments_)
        {
            oss << "  " << a.toString() << "\n";
        }
        return oss.str();
    }

} // namespace llaminar2
