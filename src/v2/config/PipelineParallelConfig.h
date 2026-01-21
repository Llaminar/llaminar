/**
 * @file PipelineParallelConfig.h
 * @brief Configuration for pipeline parallelism across MPI ranks
 *
 * Defines how transformer layers are distributed across MPI ranks for
 * pipeline parallelism. Each rank owns a contiguous range of layers.
 *
 * Design:
 * - LayerRange: Defines a contiguous range of layers owned by a rank
 * - PipelineParallelConfig: Container with factory methods for common patterns
 *
 * Usage:
 *   // Equal split across 2 ranks: 28 layers = 14 + 14
 *   auto config = PipelineParallelConfig::equalSplit(2, 28);
 *
 *   // Custom split: rank 0 gets layers 0-9, rank 1 gets layers 10-27
 *   auto config = PipelineParallelConfig::customSplit({{0, 9}, {10, 27}});
 *
 *   // Single rank (no pipeline parallelism)
 *   auto config = PipelineParallelConfig::singleRank(28);
 */

#pragma once

#include <vector>
#include <optional>
#include <stdexcept>
#include <string>

namespace llaminar2
{

    /**
     * @brief Defines a range of layers assigned to a pipeline stage
     *
     * Each pipeline stage (MPI rank) owns a contiguous range of transformer layers.
     * Layers are 0-indexed, and both first_layer and last_layer are inclusive.
     */
    struct LayerRange
    {
        int first_layer; ///< First layer index (inclusive, 0-indexed)
        int last_layer;  ///< Last layer index (inclusive)
        int owning_rank; ///< MPI rank that owns these layers

        /**
         * @brief Get the number of layers in this range
         */
        int count() const { return last_layer - first_layer + 1; }

        /**
         * @brief Check if a layer index is contained in this range
         */
        bool contains(int layer) const
        {
            return layer >= first_layer && layer <= last_layer;
        }

        /**
         * @brief String representation for logging
         */
        std::string toString() const;
    };

    /**
     * @brief Configuration for pipeline parallelism
     *
     * Defines how transformer layers are distributed across MPI ranks.
     * Provides lookup methods to determine layer ownership and pipeline topology.
     */
    class PipelineParallelConfig
    {
    public:
        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Default constructor (empty configuration)
         */
        PipelineParallelConfig() = default;

        /**
         * @brief Construct from a list of layer ranges
         *
         * @param stage_layers Vector of LayerRange, one per pipeline stage
         * @throws std::invalid_argument if validation fails
         */
        explicit PipelineParallelConfig(std::vector<LayerRange> stage_layers);

        // =====================================================================
        // Accessors
        // =====================================================================

        /**
         * @brief Get all pipeline stages
         */
        const std::vector<LayerRange> &stages() const { return stages_; }

        /**
         * @brief Get the number of pipeline stages
         */
        int numStages() const { return static_cast<int>(stages_.size()); }

        /**
         * @brief Get the total number of layers across all stages
         */
        int totalLayers() const { return total_layers_; }

        // =====================================================================
        // Lookup Methods
        // =====================================================================

        /**
         * @brief Get the layer range for a specific rank
         *
         * @param rank MPI rank (0-indexed)
         * @return Reference to the LayerRange for that rank
         * @throws std::out_of_range if rank is invalid
         */
        const LayerRange &forRank(int rank) const;

        /**
         * @brief Find which rank owns a specific layer
         *
         * @param layer Layer index (0-indexed)
         * @return MPI rank that owns the layer
         * @throws std::out_of_range if layer is not in any range
         */
        int rankForLayer(int layer) const;

        /**
         * @brief Check if a rank owns a specific layer
         *
         * @param rank MPI rank
         * @param layer Layer index
         * @return true if the rank owns the layer
         */
        bool ownsLayer(int rank, int layer) const;

        // =====================================================================
        // Pipeline Topology
        // =====================================================================

        /**
         * @brief Get the previous rank in the pipeline
         *
         * @param rank Current rank
         * @return Previous rank, or -1 if this is the first stage
         */
        int prevRank(int rank) const;

        /**
         * @brief Get the next rank in the pipeline
         *
         * @param rank Current rank
         * @return Next rank, or -1 if this is the last stage
         */
        int nextRank(int rank) const;

        /**
         * @brief Check if a rank is the first stage in the pipeline
         */
        bool isFirstStage(int rank) const { return prevRank(rank) == -1; }

        /**
         * @brief Check if a rank is the last stage in the pipeline
         */
        bool isLastStage(int rank) const { return nextRank(rank) == -1; }

        // =====================================================================
        // Validation
        // =====================================================================

        /**
         * @brief Validate the configuration
         *
         * Checks for:
         * - Non-empty configuration
         * - No gaps in layer coverage
         * - No overlapping layers
         * - Contiguous layer ranges starting from 0
         *
         * @return true if valid
         */
        bool validate() const;

        /**
         * @brief Get the validation error message
         *
         * @return Error message, or empty string if valid
         */
        std::string validationError() const;

        // =====================================================================
        // Factory Methods
        // =====================================================================

        /**
         * @brief Create an equal split of layers across ranks
         *
         * Divides layers as evenly as possible. If layers don't divide evenly,
         * earlier ranks get one extra layer each.
         *
         * Example: 28 layers / 2 ranks = 14 + 14
         * Example: 29 layers / 2 ranks = 15 + 14
         * Example: 28 layers / 3 ranks = 10 + 10 + 8
         *
         * @param num_ranks Number of MPI ranks
         * @param total_layers Total number of transformer layers
         * @return PipelineParallelConfig with equal splits
         */
        static PipelineParallelConfig equalSplit(int num_ranks, int total_layers);

        /**
         * @brief Create a custom split from explicit layer ranges
         *
         * Each pair specifies {first_layer, last_layer} for each rank in order.
         *
         * @param layer_ranges Vector of {first, last} pairs, one per rank
         * @return PipelineParallelConfig with custom splits
         * @throws std::invalid_argument if validation fails
         */
        static PipelineParallelConfig customSplit(
            const std::vector<std::pair<int, int>> &layer_ranges);

        /**
         * @brief Create a single-rank configuration (no pipeline parallelism)
         *
         * @param total_layers Total number of transformer layers
         * @return PipelineParallelConfig with all layers on rank 0
         */
        static PipelineParallelConfig singleRank(int total_layers);

    private:
        std::vector<LayerRange> stages_; ///< Layer ranges for each stage
        int total_layers_ = 0;           ///< Total layers across all stages

        /**
         * @brief Compute total layers from stages
         */
        void computeTotals();
    };

} // namespace llaminar2
