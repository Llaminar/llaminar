#include "ModelContextConfig.h"
#include "../execution/mpi_orchestration/RankExecutionPlan.h"
#include <algorithm>
#include <sstream>

namespace llaminar2 {

ModelContextConfig ModelContextConfig::defaults() {
    ModelContextConfig config;
    config.mpi_ctx = nullptr;
    config.strategy = WeightDistributionStrategy::REPLICATED;
    config.weight_precision = WeightPrecision::NATIVE;
    config.first_layer = 0;
    config.last_layer = -1;
    config.has_embedding = true;
    config.has_lm_head = true;
    config.shard_index = 0;
    config.total_shards = 1;
    config.work_fraction = 1.0f;
    config.placement_map = nullptr;
    config.factory = nullptr;
    return config;
}

ModelContextConfig ModelContextConfig::forPPStage(int stage_idx, int total_stages, int n_layers) {
    ModelContextConfig config = defaults();

    if (total_stages <= 0 || stage_idx < 0 || stage_idx >= total_stages || n_layers <= 0) {
        // Return invalid config that will fail validation
        config.first_layer = -1;
        return config;
    }

    // Divide layers evenly across stages
    int layers_per_stage = n_layers / total_stages;
    int remainder = n_layers % total_stages;

    // Calculate first_layer for this stage
    // Earlier stages get extra layers from remainder
    int first = 0;
    for (int i = 0; i < stage_idx; ++i) {
        first += layers_per_stage + (i < remainder ? 1 : 0);
    }

    // Calculate last_layer for this stage
    int layers_this_stage = layers_per_stage + (stage_idx < remainder ? 1 : 0);
    int last = first + layers_this_stage - 1;

    config.first_layer = first;
    config.last_layer = last;

    // First stage gets embedding, last stage gets lm_head
    config.has_embedding = (stage_idx == 0);
    config.has_lm_head = (stage_idx == total_stages - 1);

    return config;
}

ModelContextConfig ModelContextConfig::forTPShard(int shard_idx, int total_shards) {
    ModelContextConfig config = defaults();

    config.shard_index = shard_idx;
    config.total_shards = total_shards;
    config.work_fraction = 1.0f / static_cast<float>(total_shards);
    config.strategy = WeightDistributionStrategy::SHARDED;

    return config;
}

ModelContextConfig ModelContextConfig::fromExecutionPlan(const RankExecutionPlan& plan) {
    ModelContextConfig config = defaults();

    // Copy layer partition info
    config.first_layer = plan.first_layer;
    config.last_layer = plan.last_layer;
    config.has_embedding = plan.has_embedding;
    config.has_lm_head = plan.has_lm_head;

    // Copy weight shard info
    config.shard_index = plan.weight_shard.shard_index;
    config.total_shards = plan.weight_shard.total_shards;
    config.work_fraction = plan.weight_shard.work_fraction;

    // Auto-select strategy based on configuration
    bool is_partitioned = (plan.first_layer != 0) || (plan.last_layer != -1) ||
                          !plan.has_embedding || !plan.has_lm_head;
    bool is_sharded = plan.weight_shard.total_shards > 1;

    if (is_sharded) {
        config.strategy = WeightDistributionStrategy::SHARDED;
    } else if (is_partitioned) {
        // Layer partitioned but not sharded - still replicated within the partition
        config.strategy = WeightDistributionStrategy::REPLICATED;
    } else {
        config.strategy = WeightDistributionStrategy::REPLICATED;
    }

    return config;
}

std::vector<std::string> ModelContextConfig::validate() const {
    std::vector<std::string> errors;

    // first_layer >= 0
    if (first_layer < 0) {
        errors.push_back("first_layer must be >= 0, got " + std::to_string(first_layer));
    }

    // last_layer >= first_layer (or -1)
    if (last_layer != -1 && last_layer < first_layer) {
        errors.push_back("last_layer (" + std::to_string(last_layer) +
                         ") must be >= first_layer (" + std::to_string(first_layer) +
                         ") or -1");
    }

    // shard_index >= 0 and < total_shards
    if (shard_index < 0) {
        errors.push_back("shard_index must be >= 0, got " + std::to_string(shard_index));
    }
    if (shard_index >= total_shards) {
        errors.push_back("shard_index (" + std::to_string(shard_index) +
                         ") must be < total_shards (" + std::to_string(total_shards) + ")");
    }

    // total_shards >= 1
    if (total_shards < 1) {
        errors.push_back("total_shards must be >= 1, got " + std::to_string(total_shards));
    }

    // work_fraction in (0.0, 1.0]
    if (work_fraction <= 0.0f || work_fraction > 1.0f) {
        std::ostringstream oss;
        oss << "work_fraction must be in (0.0, 1.0], got " << work_fraction;
        errors.push_back(oss.str());
    }

    return errors;
}

bool ModelContextConfig::isLayerPartitioned() const {
    return first_layer != 0 || last_layer != -1 || !has_embedding || !has_lm_head;
}

bool ModelContextConfig::isSharded() const {
    return total_shards > 1;
}

std::string ModelContextConfig::toString() const {
    std::ostringstream oss;
    oss << "ModelContextConfig{layers=[" << first_layer << ","
        << (last_layer == -1 ? "all" : std::to_string(last_layer)) << "]"
        << ", emb=" << (has_embedding ? "true" : "false")
        << ", lm=" << (has_lm_head ? "true" : "false")
        << ", shard=" << shard_index << "/" << total_shards
        << "}";
    return oss.str();
}

} // namespace llaminar2
