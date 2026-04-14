/**
 * @file PipelineConfig.h
 * @brief Complete pipeline configuration for PP + TP composition
 *
 * Part of the Unified PP Graph Architecture (Phase 1.3).
 * Combines TP domains and PP stages into a unified pipeline specification.
 * Enables complex compositions like PP(TP(cuda:0,cuda:1) + TP(rocm:0,rocm:1) + TP(cpu:0,cpu:1)).
 *
 * @see docs/v2/UNIFIED_PP_GRAPH_ARCHITECTURE_PLAN.md
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "TPDomainConfig.h"
#include "PPStageConfig.h"
#include "OrchestrationConfig.h" // For CollectiveBackendType
#include "backends/DeviceId.h"
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Complete pipeline configuration for PP + TP composition
     *
     * Combines TP domains and PP stages into a unified pipeline specification.
     * Enables complex compositions like PP(TP(cuda:0,cuda:1) + TP(rocm:0,rocm:1) + TP(cpu:0,cpu:1)).
     *
     * ## Key Concepts
     *
     * **TP Domain**: A group of devices that work together on tensor-parallel
     * sharded operations. Each domain has a collective backend (NCCL, RCCL, etc.)
     * for allreduce operations.
     *
     * **PP Stage**: A contiguous range of transformer layers executed on a specific
     * TP domain. Stages are executed sequentially with activation transfers between them.
     *
     * **Transfer Backend**: The communication mechanism used to transfer activations
     * between PP stages (e.g., host memory staging, NCCL P2P).
     *
     * ## Usage Examples
     *
     * @code
     * // Single device, no parallelism
     * auto config = PipelineConfig::singleDevice(24, DeviceId::cuda(0));
     *
     * // 2-way tensor parallel on CUDA GPUs
     * auto config = PipelineConfig::tensorParallel(24,
     *     {DeviceId::cuda(0), DeviceId::cuda(1)},
     *     CollectiveBackendType::NCCL);
     *
     * // 2-stage pipeline parallel
     * auto config = PipelineConfig::pipelineParallel2Stage(24,
     *     DeviceId::cuda(0), 12,  // First 12 layers on cuda:0
     *     DeviceId::rocm(0),      // Last 12 layers on rocm:0
     *     CollectiveBackendType::HOST);
     *
     * // Complex PP + TP composition (manual construction)
     * PipelineConfig config;
     * config.total_layers = 28;
     * config.tp_domains = {
     *     {"gpu_nvidia", {DeviceId::cuda(0), DeviceId::cuda(1)}, CollectiveBackendType::NCCL},
     *     {"gpu_amd", {DeviceId::rocm(0), DeviceId::rocm(1)}, CollectiveBackendType::RCCL}
     * };
     * config.pp_stages = {
     *     PPStageConfig::firstStage(0, "gpu_nvidia", 0, 14),
     *     PPStageConfig::lastStage(1, "gpu_amd", 14, 28)
     * };
     * config.pp_transfer_backends[{0, 1}] = CollectiveBackendType::HOST;
     * @endcode
     */
    struct PipelineConfig
    {
        // =========================================================================
        // Configuration Data
        // =========================================================================

        /// TP domains (groups of devices for tensor parallelism)
        std::vector<TPDomainConfig> tp_domains;

        /// PP stages in execution order
        std::vector<PPStageConfig> pp_stages;

        /// PP transfer backends between stages
        /// Key: {from_stage_id, to_stage_id}, Value: backend type
        std::map<std::pair<int, int>, CollectiveBackendType> pp_transfer_backends;

        /// Total number of layers in the model
        int total_layers = 0;

        // =========================================================================
        // Lookup Methods
        // =========================================================================

        /**
         * @brief Get a TP domain by name
         * @param name Domain name to look up
         * @return Pointer to domain, or nullptr if not found
         */
        [[nodiscard]] const TPDomainConfig *getDomain(const std::string &name) const;

        /**
         * @brief Get the TP domain for a specific PP stage
         * @param stage_id Stage ID to look up
         * @return Pointer to domain, or nullptr if stage not found or domain missing
         */
        [[nodiscard]] const TPDomainConfig *getDomainForStage(int stage_id) const;

        /**
         * @brief Get the PP stage that contains a specific layer
         * @param layer_idx Layer index to look up
         * @return Pointer to stage, or nullptr if layer not covered by any stage
         */
        [[nodiscard]] const PPStageConfig *getStageForLayer(int layer_idx) const;

        /**
         * @brief Get the primary device for a specific layer
         *
         * Finds the PP stage containing the layer, then returns the primary device
         * of that stage's TP domain.
         *
         * @param layer_idx Layer index to look up
         * @return Device ID for the layer, or DeviceId::cpu() if not found
         */
        [[nodiscard]] DeviceId getDeviceForLayer(int layer_idx) const;

        /**
         * @brief Check if a PP transfer is needed between two layers
         *
         * Returns true if the two layers are in different PP stages, meaning
         * an activation transfer is required between them.
         *
         * @param from_layer Source layer index
         * @param to_layer Destination layer index
         * @return true if transfer needed (different stages)
         */
        [[nodiscard]] bool needsPPTransfer(int from_layer, int to_layer) const;

        /**
         * @brief Get the transfer backend between two stages
         * @param from_stage Source stage ID
         * @param to_stage Destination stage ID
         * @return Backend type, or AUTO if not explicitly configured
         */
        [[nodiscard]] CollectiveBackendType getTransferBackend(int from_stage, int to_stage) const;

        /**
         * @brief Get the stage ID for a layer
         * @param layer_idx Layer index to look up
         * @return Stage ID, or -1 if layer not covered by any stage
         */
        [[nodiscard]] int getStageIdForLayer(int layer_idx) const;

        // =========================================================================
        // Query Methods
        // =========================================================================

        /**
         * @brief Get the number of PP stages
         * @return Number of stages in pp_stages
         */
        [[nodiscard]] int numStages() const;

        /**
         * @brief Get the maximum TP degree across all domains
         * @return Largest device count among all domains (0 if no domains)
         */
        [[nodiscard]] int maxTPDegree() const;

        /**
         * @brief Check if this is a simple single-stage pipeline (no PP)
         * @return true if there is exactly one PP stage
         */
        [[nodiscard]] bool isSingleStage() const;

        /**
         * @brief Check if any domain has internal TP (degree > 1)
         * @return true if any domain has more than one device
         */
        [[nodiscard]] bool hasTP() const;

        /**
         * @brief Check if there are multiple PP stages
         * @return true if there are 2 or more PP stages
         */
        [[nodiscard]] bool hasPP() const;

        /**
         * @brief Get all unique devices across all domains
         *
         * Collects all devices from all TP domains, removing duplicates.
         * Order is preserved (first occurrence wins).
         *
         * @return Vector of unique DeviceIds
         */
        [[nodiscard]] std::vector<DeviceId> getAllDevices() const;

        // =========================================================================
        // Validation
        // =========================================================================

        /**
         * @brief Validate the complete pipeline configuration
         *
         * Checks:
         * - At least one domain exists
         * - At least one stage exists
         * - All stage domain_names reference existing domains
         * - Stages cover all layers [0, total_layers) with no gaps or overlaps
         * - Exactly one stage has has_embedding=true (stage with first_layer=0)
         * - Exactly one stage has has_lm_head=true (stage with last_layer=total_layers)
         * - All domains pass their own validation
         * - All stages pass their own validation
         * - total_layers > 0
         *
         * @param error_msg Output: error message if invalid (optional)
         * @return true if valid
         */
        [[nodiscard]] bool validate(std::string *error_msg = nullptr) const;

        // =========================================================================
        // Auto-Completion Methods
        // =========================================================================

        /**
         * @brief Auto-complete missing backend selections
         *
         * For any {from_stage, to_stage} pair not in pp_transfer_backends,
         * uses BackendSelector to determine the optimal backend based on
         * the stage devices.
         *
         * Also resolves tp_backend=AUTO domains to their optimal backends.
         */
        void autoSelectBackends();

        /**
         * @brief Validate and auto-complete (convenience method)
         *
         * Calls autoSelectBackends() then validate().
         *
         * @param error_msg Output: error message if invalid
         * @return true if valid after auto-completion
         */
        [[nodiscard]] bool completeAndValidate(std::string *error_msg = nullptr);

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /**
         * @brief Create a single-device pipeline (no parallelism)
         *
         * Creates the simplest possible configuration: one domain with one device,
         * one stage covering all layers.
         *
         * @param num_layers Total number of transformer layers
         * @param device The single device to run on
         * @return Valid PipelineConfig
         */
        static PipelineConfig singleDevice(int num_layers, DeviceId device);

        /**
         * @brief Create a single-stage TP pipeline
         *
         * Creates a configuration with one domain containing multiple devices
         * (tensor parallelism), and one stage covering all layers.
         *
         * @param num_layers Total number of transformer layers
         * @param devices Devices for tensor parallelism
         * @param backend Collective backend for TP allreduce
         * @return Valid PipelineConfig
         */
        static PipelineConfig tensorParallel(int num_layers,
                                             const std::vector<DeviceId> &devices,
                                             CollectiveBackendType backend);

        /**
         * @brief Create a simple 2-stage PP pipeline (no internal TP)
         *
         * Creates a configuration with two stages, each with one device.
         * First stage has layers [0, split_layer), second has [split_layer, num_layers).
         *
         * @param num_layers Total number of transformer layers
         * @param device0 Device for first stage
         * @param split_layer Layer index where to split (first layer of second stage)
         * @param device1 Device for second stage
         * @param transfer_backend Backend for PP activation transfer
         * @return Valid PipelineConfig
         */
        static PipelineConfig pipelineParallel2Stage(int num_layers,
                                                     DeviceId device0, int split_layer,
                                                     DeviceId device1,
                                                     CollectiveBackendType transfer_backend);
    };

} // namespace llaminar2
