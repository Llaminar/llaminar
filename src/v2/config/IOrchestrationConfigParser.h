/**
 * @file IOrchestrationConfigParser.h
 * @brief Interface for orchestration configuration parsing
 *
 * Provides a mockable interface for parsing orchestration configuration
 * from CLI arguments, YAML files, or YAML strings.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "OrchestrationConfig.h"
#include <string>
#include <memory>

namespace llaminar2
{

    /**
     * @brief Interface for orchestration configuration parsing
     *
     * Enables dependency injection and mocking for testing.
     * Implementations can parse from:
     * - Command line arguments
     * - YAML configuration files
     * - YAML strings (for testing)
     */
    class IOrchestrationConfigParser
    {
    public:
        virtual ~IOrchestrationConfigParser() = default;

        /**
         * @brief Parse configuration from command line arguments
         *
         * Supported options:
         *   --dry-run                   Show configuration without executing
         *   --explain-placement         Explain device placement decisions
         *   --show-topology             Show detected topology and exit
         *   --show-numa                 Show NUMA configuration and exit
         *   --validate-only             Validate configuration without running
         *
         *   --device <spec>             Device for this rank (e.g., cuda:0)
         *   --device-mode <mode>        Device assignment mode (auto, local_gpu, round_robin, explicit)
         *   --device-map <map>          Explicit device mapping (e.g., "0=cuda:0,1=cuda:1")
         *
         *   --define-domain <spec>      Define a named domain (e.g., "gpu_tp=cuda:0,cuda:1")
         *   --pp-stage <spec>           Define PP stage mapping (e.g., "0=gpu_tp:0-13")
         *
         *   --tp <degree>               Tensor parallelism degree
         *   --tp-scope <scope>          TP scope (auto, local, global, hybrid)
         *   --tp-devices <specs>        TP device list (e.g., "cuda:0,cuda:1")
         *   --tp-weights <weights>      TP weight distribution (e.g., "0.73,0.27")
         *   --tp-local <degree>         Local TP degree for hybrid
         *   --tp-global <degree>        Global TP degree for hybrid
         *
         *   --pp <degree>               Pipeline parallelism degree
         *   --pp-split <mode>           PP layer split mode (equal, weighted, manual)
         *
         *   --cpu-layers <count>        Number of layers on CPU
         *   --cpu-layers-first          Put CPU layers at the beginning
         *
         *   --backend <type>            Default collective backend (auto, nccl, rccl, etc.)
         *   --config <path>             Path to YAML configuration file
         *
         *   MoE expert overlay uses named domains as the canonical hardware
         *   inventory. Use --moe-expert-overlay-continuation for activation /
         *   logits ownership and --moe-expert-overlay-base-domain for dense /
         *   non-expert model placement (defaults to continuation). Do not
         *   combine explicit overlay placements with --device/-d.
         *
         * @param argc Argument count
         * @param argv Argument values
         * @return Parsed OrchestrationConfig
         * @throws std::invalid_argument if parsing fails
         */
        virtual OrchestrationConfig parseArgs(int argc, char **argv) = 0;

        /**
         * @brief Parse configuration from YAML file
         *
         * YAML format example:
         * ```yaml
         * orchestration:
         *   tp_degree: 2
         *   tp_scope: local
         *   domains:
         *     - name: gpu_tp
         *       devices: [cuda:0, cuda:1]
         *       weights: [0.73, 0.27]
         *       backend: nccl
         *   pp_stages:
         *     - stage: 0
         *       domain: gpu_tp
         *       layers: [0, 13]
         *
         * moe_expert_parallel:
         *   enabled: true
         *   execution_kind: tiered
         *   continuation_domain: rocm_hot
         *   base_model_domain: rocm_hot
         *   shared_expert_domain: rocm_hot
         *   routed_tiers:
         *     - "hot@rocm_hot;priority=0"
         *     - "cold@cpu_cold;priority=1;fallback=true"
         * ```
         *
         * @param path Path to YAML file
         * @return Parsed OrchestrationConfig
         * @throws std::invalid_argument if file doesn't exist or parsing fails
         */
        virtual OrchestrationConfig parseYamlFile(const std::string &path) = 0;

        /**
         * @brief Parse configuration from YAML string (for testing)
         *
         * @param yaml YAML string content
         * @return Parsed OrchestrationConfig
         * @throws std::invalid_argument if parsing fails
         */
        virtual OrchestrationConfig parseYamlString(const std::string &yaml) = 0;
    };

    /**
     * @brief Factory function to create the default parser implementation
     * @return Unique pointer to parser implementation
     */
    std::unique_ptr<IOrchestrationConfigParser> createOrchestrationConfigParser();

} // namespace llaminar2
