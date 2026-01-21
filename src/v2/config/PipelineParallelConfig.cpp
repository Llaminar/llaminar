/**
 * @file PipelineParallelConfig.cpp
 * @brief Implementation of pipeline parallelism configuration
 */

#include "PipelineParallelConfig.h"
#include <algorithm>
#include <sstream>

namespace llaminar2
{

    // =========================================================================
    // LayerRange
    // =========================================================================

    std::string LayerRange::toString() const
    {
        std::ostringstream oss;
        oss << "Rank " << owning_rank << ": layers [" << first_layer << ", "
            << last_layer << "] (" << count() << " layers)";
        return oss.str();
    }

    // =========================================================================
    // PipelineParallelConfig - Constructor
    // =========================================================================

    PipelineParallelConfig::PipelineParallelConfig(std::vector<LayerRange> stage_layers)
        : stages_(std::move(stage_layers))
    {
        computeTotals();

        // Validate on construction
        if (!validate())
        {
            throw std::invalid_argument(
                "Invalid PipelineParallelConfig: " + validationError());
        }
    }

    // =========================================================================
    // PipelineParallelConfig - Private Helpers
    // =========================================================================

    void PipelineParallelConfig::computeTotals()
    {
        total_layers_ = 0;
        for (const auto &stage : stages_)
        {
            total_layers_ += stage.count();
        }
    }

    // =========================================================================
    // PipelineParallelConfig - Lookup Methods
    // =========================================================================

    const LayerRange &PipelineParallelConfig::forRank(int rank) const
    {
        if (rank < 0 || rank >= static_cast<int>(stages_.size()))
        {
            throw std::out_of_range(
                "Rank " + std::to_string(rank) + " out of bounds [0, " +
                std::to_string(stages_.size()) + ")");
        }
        return stages_[rank];
    }

    int PipelineParallelConfig::rankForLayer(int layer) const
    {
        for (const auto &stage : stages_)
        {
            if (stage.contains(layer))
            {
                return stage.owning_rank;
            }
        }
        throw std::out_of_range(
            "Layer " + std::to_string(layer) + " not found in any stage");
    }

    bool PipelineParallelConfig::ownsLayer(int rank, int layer) const
    {
        if (rank < 0 || rank >= static_cast<int>(stages_.size()))
        {
            return false;
        }
        return stages_[rank].contains(layer);
    }

    // =========================================================================
    // PipelineParallelConfig - Pipeline Topology
    // =========================================================================

    int PipelineParallelConfig::prevRank(int rank) const
    {
        if (rank <= 0 || rank >= static_cast<int>(stages_.size()))
        {
            return -1;
        }
        return rank - 1;
    }

    int PipelineParallelConfig::nextRank(int rank) const
    {
        if (rank < 0 || rank >= static_cast<int>(stages_.size()) - 1)
        {
            return -1;
        }
        return rank + 1;
    }

    // =========================================================================
    // PipelineParallelConfig - Validation
    // =========================================================================

    bool PipelineParallelConfig::validate() const
    {
        return validationError().empty();
    }

    std::string PipelineParallelConfig::validationError() const
    {
        // Check for empty configuration
        if (stages_.empty())
        {
            return "Configuration has no stages";
        }

        // Check that ranks are sequential starting from 0
        for (int i = 0; i < static_cast<int>(stages_.size()); ++i)
        {
            if (stages_[i].owning_rank != i)
            {
                return "Stage " + std::to_string(i) + " has owning_rank " +
                       std::to_string(stages_[i].owning_rank) + " (expected " +
                       std::to_string(i) + ")";
            }
        }

        // Check that each stage has valid layer range
        for (const auto &stage : stages_)
        {
            if (stage.first_layer < 0)
            {
                return "Rank " + std::to_string(stage.owning_rank) +
                       " has negative first_layer: " + std::to_string(stage.first_layer);
            }
            if (stage.last_layer < stage.first_layer)
            {
                return "Rank " + std::to_string(stage.owning_rank) +
                       " has last_layer < first_layer: [" +
                       std::to_string(stage.first_layer) + ", " +
                       std::to_string(stage.last_layer) + "]";
            }
        }

        // Check that first stage starts at layer 0
        if (stages_[0].first_layer != 0)
        {
            return "First stage must start at layer 0, but starts at " +
                   std::to_string(stages_[0].first_layer);
        }

        // Check for gaps and overlaps between consecutive stages
        for (size_t i = 1; i < stages_.size(); ++i)
        {
            int prev_last = stages_[i - 1].last_layer;
            int curr_first = stages_[i].first_layer;

            if (curr_first != prev_last + 1)
            {
                if (curr_first <= prev_last)
                {
                    return "Overlap between rank " + std::to_string(i - 1) +
                           " (last_layer=" + std::to_string(prev_last) +
                           ") and rank " + std::to_string(i) +
                           " (first_layer=" + std::to_string(curr_first) + ")";
                }
                else
                {
                    return "Gap between rank " + std::to_string(i - 1) +
                           " (last_layer=" + std::to_string(prev_last) +
                           ") and rank " + std::to_string(i) +
                           " (first_layer=" + std::to_string(curr_first) + ")";
                }
            }
        }

        return ""; // Valid
    }

    // =========================================================================
    // PipelineParallelConfig - Factory Methods
    // =========================================================================

    PipelineParallelConfig PipelineParallelConfig::equalSplit(int num_ranks, int total_layers)
    {
        if (num_ranks <= 0)
        {
            throw std::invalid_argument("num_ranks must be positive");
        }
        if (total_layers <= 0)
        {
            throw std::invalid_argument("total_layers must be positive");
        }
        if (num_ranks > total_layers)
        {
            throw std::invalid_argument(
                "num_ranks (" + std::to_string(num_ranks) +
                ") cannot exceed total_layers (" + std::to_string(total_layers) + ")");
        }

        std::vector<LayerRange> stages;
        stages.reserve(num_ranks);

        int base_layers = total_layers / num_ranks;
        int remainder = total_layers % num_ranks;
        int current_layer = 0;

        for (int rank = 0; rank < num_ranks; ++rank)
        {
            // Earlier ranks get one extra layer if there's a remainder
            int layers_for_rank = base_layers + (rank < remainder ? 1 : 0);

            LayerRange range;
            range.first_layer = current_layer;
            range.last_layer = current_layer + layers_for_rank - 1;
            range.owning_rank = rank;

            stages.push_back(range);
            current_layer += layers_for_rank;
        }

        return PipelineParallelConfig(std::move(stages));
    }

    PipelineParallelConfig PipelineParallelConfig::customSplit(
        const std::vector<std::pair<int, int>> &layer_ranges)
    {
        if (layer_ranges.empty())
        {
            throw std::invalid_argument("layer_ranges cannot be empty");
        }

        std::vector<LayerRange> stages;
        stages.reserve(layer_ranges.size());

        for (size_t i = 0; i < layer_ranges.size(); ++i)
        {
            LayerRange range;
            range.first_layer = layer_ranges[i].first;
            range.last_layer = layer_ranges[i].second;
            range.owning_rank = static_cast<int>(i);
            stages.push_back(range);
        }

        return PipelineParallelConfig(std::move(stages));
    }

    PipelineParallelConfig PipelineParallelConfig::singleRank(int total_layers)
    {
        if (total_layers <= 0)
        {
            throw std::invalid_argument("total_layers must be positive");
        }

        LayerRange range;
        range.first_layer = 0;
        range.last_layer = total_layers - 1;
        range.owning_rank = 0;

        return PipelineParallelConfig({range});
    }

} // namespace llaminar2
