/**
 * @file RankExecutionPlan.h
 * @brief Contract between cluster orchestration (Layer 2) and rank-local execution (Layer 3)
 *
 * Each MPI rank receives one RankExecutionPlan and executes it independently.
 * The plan contains all information needed to:
 * - Build the correct subset of the model (layers, embedding, LM head)
 * - Shard weights appropriately
 * - Configure TP/PP communication
 *
 * Key principle: Layer 3 receives RankExecutionPlan and executes it WITHOUT
 * needing cluster-wide context. It doesn't know what other ranks are doing;
 * it just follows its plan.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../backends/GlobalDeviceAddress.h"
#include "../../config/OrchestrationConfig.h"
#include <optional>
#include <vector>
#include <string>
#include <sstream>

namespace llaminar2
{

    // =========================================================================
    // WeightShardInfo
    // =========================================================================

    /**
     * @brief Weight shard information for this rank
     *
     * Describes how weights should be loaded and partitioned.
     * For proportional TP (mixed GPU types), work_fraction may be < 1.0/total_shards.
     */
    struct WeightShardInfo
    {
        int shard_index = 0;        ///< Which shard this rank loads (0-based)
        int total_shards = 1;       ///< Total number of shards
        float work_fraction = 1.0f; ///< Fraction of work for proportional TP

        /**
         * @brief Check if weights are sharded across multiple devices/ranks
         * @return true if total_shards > 1
         */
        bool isSharded() const { return total_shards > 1; }

        /**
         * @brief Validate shard info is consistent
         * @return Empty vector if valid, otherwise list of error messages
         */
        std::vector<std::string> validate() const
        {
            std::vector<std::string> errors;
            if (shard_index < 0)
            {
                errors.push_back("shard_index must be >= 0");
            }
            if (total_shards < 1)
            {
                errors.push_back("total_shards must be >= 1");
            }
            if (shard_index >= total_shards)
            {
                errors.push_back("shard_index (" + std::to_string(shard_index) +
                                 ") must be < total_shards (" + std::to_string(total_shards) + ")");
            }
            if (work_fraction <= 0.0f || work_fraction > 1.0f)
            {
                errors.push_back("work_fraction must be in (0.0, 1.0]");
            }
            return errors;
        }

        /**
         * @brief String representation for debugging
         */
        std::string toString() const
        {
            std::ostringstream ss;
            ss << "WeightShard{index=" << shard_index
               << ", total=" << total_shards
               << ", work=" << work_fraction << "}";
            return ss.str();
        }
    };

    // =========================================================================
    // TPDomainParticipation
    // =========================================================================

    /**
     * @brief TP domain participation information for this rank
     *
     * Describes which TP domain(s) this rank participates in,
     * including device list, work distribution, and collective backend.
     */
    struct TPDomainParticipation
    {
        int domain_id = -1;                       ///< Unique domain identifier
        std::string domain_name;                  ///< Human-readable domain name
        std::vector<GlobalDeviceAddress> devices; ///< All devices in this domain
        std::vector<float> weights;               ///< Work distribution weights
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        int my_index_in_domain = 0; ///< This rank's position in domain

        /**
         * @brief Get the device this rank should use within the domain
         * @return GlobalDeviceAddress for this rank
         */
        GlobalDeviceAddress myDevice() const
        {
            if (my_index_in_domain >= 0 &&
                my_index_in_domain < static_cast<int>(devices.size()))
            {
                return devices[my_index_in_domain];
            }
            return GlobalDeviceAddress::cpu();
        }

        /**
         * @brief Get work fraction for this rank
         * @return Weight for this rank (defaults to 1/N if weights empty)
         */
        float myWeight() const
        {
            if (weights.empty())
            {
                return devices.empty() ? 1.0f : (1.0f / devices.size());
            }
            if (my_index_in_domain >= 0 &&
                my_index_in_domain < static_cast<int>(weights.size()))
            {
                return weights[my_index_in_domain];
            }
            return 1.0f / devices.size();
        }

        /**
         * @brief Validate domain participation
         * @return Empty vector if valid, otherwise list of errors
         */
        std::vector<std::string> validate() const
        {
            std::vector<std::string> errors;
            if (domain_id < 0)
            {
                errors.push_back("domain_id must be >= 0");
            }
            if (devices.empty())
            {
                errors.push_back("domain must have at least one device");
            }
            if (!weights.empty() && weights.size() != devices.size())
            {
                errors.push_back("weights count (" + std::to_string(weights.size()) +
                                 ") must match device count (" + std::to_string(devices.size()) + ")");
            }
            if (my_index_in_domain < 0 || my_index_in_domain >= static_cast<int>(devices.size()))
            {
                errors.push_back("my_index_in_domain (" + std::to_string(my_index_in_domain) +
                                 ") out of range [0, " + std::to_string(devices.size()) + ")");
            }
            return errors;
        }

        /**
         * @brief String representation for debugging
         */
        std::string toString() const
        {
            std::ostringstream ss;
            ss << "TPDomain{id=" << domain_id
               << ", name='" << domain_name << "'"
               << ", devices=[";
            for (size_t i = 0; i < devices.size(); ++i)
            {
                if (i > 0)
                    ss << ", ";
                ss << devices[i].toString();
            }
            ss << "], my_idx=" << my_index_in_domain
               << ", backend=" << collectiveBackendTypeToString(backend) << "}";
            return ss.str();
        }
    };

    // =========================================================================
    // RankExecutionPlan
    // =========================================================================

    /**
     * @brief The contract between Layer 2 (cluster orchestration) and Layer 3 (rank-local execution)
     *
     * Each MPI rank receives one RankExecutionPlan containing all information
     * needed to execute its portion of inference:
     * - Identity (rank, hostname, NUMA)
     * - Pipeline parallelism (which layers, PP neighbors)
     * - Tensor parallelism (local devices, global participation)
     * - Weight loading (shard index, total shards)
     */
    struct RankExecutionPlan
    {
        // =====================================================================
        // Identity
        // =====================================================================

        int rank = 0;         ///< This MPI rank
        std::string hostname; ///< Machine we're on
        int numa_node = 0;    ///< NUMA node we're bound to

        // =====================================================================
        // Pipeline Parallelism (what layers to run)
        // =====================================================================

        int pp_stage_id = 0;       ///< Which PP stage this rank owns
        int first_layer = 0;       ///< First layer index to build/execute
        int last_layer = -1;       ///< Last layer index (inclusive), -1 = all
        bool has_embedding = true; ///< Build embedding stage?
        bool has_lm_head = true;   ///< Build LM head stage?

        // PP Communication
        std::optional<int> prev_rank; ///< Rank to receive activations from
        std::optional<int> next_rank; ///< Rank to send activations to

        // =====================================================================
        // Tensor Parallelism
        // =====================================================================

        TPScope tp_scope = TPScope::AUTO;

        // LOCAL TP (devices within this rank)
        std::vector<GlobalDeviceAddress> local_tp_devices;
        std::vector<float> local_tp_weights;
        CollectiveBackendType local_tp_backend = CollectiveBackendType::AUTO;

        // GLOBAL TP (participation in cross-rank TP)
        std::optional<int> global_tp_domain_id;
        int global_tp_rank_in_domain = 0;
        int global_tp_domain_size = 1;

        // =====================================================================
        // Weight Loading
        // =====================================================================

        WeightShardInfo weight_shard;

        // =====================================================================
        // All TP domains this rank participates in
        // =====================================================================

        std::vector<TPDomainParticipation> my_domains;

        // =====================================================================
        // Cross-rank communication
        // =====================================================================

        CollectiveBackendType cross_rank_backend = CollectiveBackendType::MPI;

        // =====================================================================
        // Primary device for this rank
        // =====================================================================

        GlobalDeviceAddress primary_device;

        // =====================================================================
        // Convenience Methods
        // =====================================================================

        /**
         * @brief Check if this plan uses pipeline parallelism
         * @return true if there are PP neighbors
         */
        bool usesPipelineParallel() const
        {
            return prev_rank.has_value() || next_rank.has_value();
        }

        /**
         * @brief Check if this plan uses local tensor parallelism
         * @return true if multiple local devices
         */
        bool usesLocalTP() const
        {
            return local_tp_devices.size() > 1;
        }

        /**
         * @brief Check if this plan uses global tensor parallelism
         * @return true if participating in a global TP domain
         */
        bool usesGlobalTP() const
        {
            return global_tp_domain_id.has_value();
        }

        /**
         * @brief Get number of layers this rank is responsible for
         * @return Layer count (0 if no layers assigned)
         */
        int layerCount() const
        {
            return (last_layer >= first_layer) ? (last_layer - first_layer + 1) : 0;
        }

        /**
         * @brief Check if this rank handles a specific layer
         * @param layer_idx Layer index to check
         * @return true if layer is in this rank's range
         */
        bool hasLayer(int layer_idx) const
        {
            return layer_idx >= first_layer && layer_idx <= last_layer;
        }

        /**
         * @brief Check if this is the first PP stage
         * @return true if no previous rank
         */
        bool isFirstStage() const
        {
            return !prev_rank.has_value();
        }

        /**
         * @brief Check if this is the last PP stage
         * @return true if no next rank
         */
        bool isLastStage() const
        {
            return !next_rank.has_value();
        }

        /**
         * @brief Get total TP degree (local * global)
         * @return Total tensor parallelism degree
         */
        int totalTPDegree() const
        {
            int local_deg = std::max(1, static_cast<int>(local_tp_devices.size()));
            int global_deg = global_tp_domain_size;
            return local_deg * global_deg;
        }

        // =====================================================================
        // Validation
        // =====================================================================

        /**
         * @brief Validate the plan is internally consistent
         * @return Empty vector if valid, otherwise list of error messages
         */
        std::vector<std::string> validate() const
        {
            std::vector<std::string> errors;

            // Rank must be non-negative
            if (rank < 0)
            {
                errors.push_back("rank must be >= 0");
            }

            // Layer range must be valid
            if (last_layer >= 0 && first_layer > last_layer)
            {
                errors.push_back("first_layer (" + std::to_string(first_layer) +
                                 ") > last_layer (" + std::to_string(last_layer) + ")");
            }

            // PP stage must be non-negative
            if (pp_stage_id < 0)
            {
                errors.push_back("pp_stage_id must be >= 0");
            }

            // First stage should have embedding
            if (!prev_rank.has_value() && !has_embedding)
            {
                errors.push_back("First PP stage should have embedding");
            }

            // Last stage should have LM head
            if (!next_rank.has_value() && !has_lm_head)
            {
                errors.push_back("Last PP stage should have LM head");
            }

            // Local TP weights must match device count if specified
            if (!local_tp_weights.empty() &&
                local_tp_weights.size() != local_tp_devices.size())
            {
                errors.push_back("local_tp_weights count (" +
                                 std::to_string(local_tp_weights.size()) +
                                 ") must match local_tp_devices count (" +
                                 std::to_string(local_tp_devices.size()) + ")");
            }

            // Global TP rank must be within domain
            if (global_tp_domain_id.has_value())
            {
                if (global_tp_rank_in_domain < 0 ||
                    global_tp_rank_in_domain >= global_tp_domain_size)
                {
                    errors.push_back("global_tp_rank_in_domain (" +
                                     std::to_string(global_tp_rank_in_domain) +
                                     ") out of range [0, " +
                                     std::to_string(global_tp_domain_size) + ")");
                }
            }

            // Validate weight shard
            auto shard_errors = weight_shard.validate();
            errors.insert(errors.end(), shard_errors.begin(), shard_errors.end());

            // Validate domain participations
            for (size_t i = 0; i < my_domains.size(); ++i)
            {
                auto domain_errors = my_domains[i].validate();
                for (auto &e : domain_errors)
                {
                    errors.push_back("my_domains[" + std::to_string(i) + "]: " + e);
                }
            }

            return errors;
        }

        // =====================================================================
        // Serialization
        // =====================================================================

        /**
         * @brief Convert to human-readable string for debugging
         * @return Multi-line string representation
         */
        std::string toString() const
        {
            std::ostringstream ss;
            ss << "RankExecutionPlan {\n";
            ss << "  Identity:\n";
            ss << "    rank: " << rank << "\n";
            ss << "    hostname: " << hostname << "\n";
            ss << "    numa_node: " << numa_node << "\n";
            ss << "    primary_device: " << primary_device.toString() << "\n";

            ss << "  Pipeline Parallelism:\n";
            ss << "    pp_stage_id: " << pp_stage_id << "\n";
            ss << "    layers: [" << first_layer << ", " << last_layer << "]\n";
            ss << "    has_embedding: " << (has_embedding ? "true" : "false") << "\n";
            ss << "    has_lm_head: " << (has_lm_head ? "true" : "false") << "\n";
            ss << "    prev_rank: " << (prev_rank.has_value() ? std::to_string(*prev_rank) : "none") << "\n";
            ss << "    next_rank: " << (next_rank.has_value() ? std::to_string(*next_rank) : "none") << "\n";

            ss << "  Tensor Parallelism:\n";
            ss << "    tp_scope: " << tpScopeToString(tp_scope) << "\n";
            ss << "    local_tp_devices: [";
            for (size_t i = 0; i < local_tp_devices.size(); ++i)
            {
                if (i > 0)
                    ss << ", ";
                ss << local_tp_devices[i].toShortString();
            }
            ss << "]\n";
            if (!local_tp_weights.empty())
            {
                ss << "    local_tp_weights: [";
                for (size_t i = 0; i < local_tp_weights.size(); ++i)
                {
                    if (i > 0)
                        ss << ", ";
                    ss << local_tp_weights[i];
                }
                ss << "]\n";
            }
            ss << "    local_tp_backend: " << collectiveBackendTypeToString(local_tp_backend) << "\n";

            if (global_tp_domain_id.has_value())
            {
                ss << "    global_tp_domain_id: " << *global_tp_domain_id << "\n";
                ss << "    global_tp_rank_in_domain: " << global_tp_rank_in_domain << "\n";
                ss << "    global_tp_domain_size: " << global_tp_domain_size << "\n";
            }

            ss << "  Weight Shard:\n";
            ss << "    " << weight_shard.toString() << "\n";

            if (!my_domains.empty())
            {
                ss << "  Domain Participation:\n";
                for (const auto &domain : my_domains)
                {
                    ss << "    " << domain.toString() << "\n";
                }
            }

            ss << "  Cross-rank backend: " << collectiveBackendTypeToString(cross_rank_backend) << "\n";
            ss << "}";
            return ss.str();
        }
    };

} // namespace llaminar2
