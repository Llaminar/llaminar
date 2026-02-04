#pragma once

#include "WeightManager.h"
#include "../utils/MPIContext.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2 {

// Forward declarations to avoid circular dependency
struct RankExecutionPlan;
class TensorFactory;

/**
 * @brief Configuration for ModelContext creation
 *
 * Unified configuration that supports all scenarios:
 * - Single device (full model)
 * - Tensor Parallelism (sharded weights)
 * - Pipeline Parallelism (layer partitions)
 * - Combined TP + PP
 */
struct ModelContextConfig {
    // Core Settings
    std::shared_ptr<MPIContext> mpi_ctx;
    WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED;
    WeightPrecision weight_precision = WeightPrecision::NATIVE;

    // Pipeline Parallelism (Layer Range)
    int first_layer = 0;       // First layer to load (inclusive, 0-indexed)
    int last_layer = -1;       // Last layer to load (inclusive, -1 = all)
    bool has_embedding = true; // Load embedding weights?
    bool has_lm_head = true;   // Load output_norm and lm_head?

    // Tensor Parallelism (Weight Sharding)
    int shard_index = 0;       // Which shard (0-indexed)
    int total_shards = 1;      // Total shards (1 = no sharding)
    float work_fraction = 1.0f; // For proportional TP

    // Advanced Settings
    std::shared_ptr<WeightPlacementMap> placement_map;
    TensorFactory* factory = nullptr;

    // Factory Helpers
    static ModelContextConfig defaults();
    static ModelContextConfig forPPStage(int stage_idx, int total_stages, int n_layers);
    static ModelContextConfig forTPShard(int shard_idx, int total_shards);
    static ModelContextConfig fromExecutionPlan(const RankExecutionPlan& plan);

    // Validation
    std::vector<std::string> validate() const;
    bool isLayerPartitioned() const;
    bool isSharded() const;
    std::string toString() const;
};

} // namespace llaminar2
