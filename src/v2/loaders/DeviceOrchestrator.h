/**
 * @file DeviceOrchestrator.h
 * @brief High-level orchestration of device placement strategies
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#pragma once

#include "WeightPlacementMap.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"
#include "ModelContext.h"
#include <memory>
#include <string>
#include <optional>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Device map rule types for CUSTOM strategy
     */
    enum class DeviceMapRuleType
    {
        INVALID,     ///< Invalid/unparseable rule
        LAYER_RANGE, ///< Layer range: "0-11:gpu:0"
        PERCENTAGE,  ///< Percentage: "first_50%:gpu" or "last_25%:cpu"
        PATTERN      ///< Pattern match: "embed:gpu", "*experts.0*:cpu"
    };

    /**
     * @brief Device map rule for custom placement
     */
    struct DeviceMapRule
    {
        DeviceMapRuleType type = DeviceMapRuleType::INVALID;
        DeviceId device = DeviceId::cpu();

        // LAYER_RANGE fields
        int start_layer = -1;
        int end_layer = -1;

        // PERCENTAGE fields
        float percentage = 0.0f;
        bool is_first = true; // true = first_N%, false = last_N%

        // PATTERN fields
        std::string pattern;
    };

    /**
     * @brief Weight placement strategy types supported by the orchestrator
     *
     * Controls where model weights are stored (CPU vs GPU memory).
     * Distinct from LayerPlacementStrategy which controls where compute runs.
     */
    enum class WeightPlacementStrategy
    {
        ALL_GPU,       ///< All weights on GPU device 0
        ALL_CPU,       ///< All weights on CPU
        LAYER_SPLIT,   ///< First N layers on GPU, rest on CPU
        AUTO,          ///< Automatic based on available memory
        MEMORY_AWARE,  ///< Auto-fit layers within memory budget (Phase 2)
        MOE_OPTIMIZED, ///< MoE-aware placement (shared experts GPU, sparse CPU) (Phase 2)
        CUSTOM,        ///< User-provided custom device map string (Phase 2)
        MULTI_GPU      ///< Distribute layers across multiple GPUs (Phase 6)
    };

    /**
     * @brief Configuration for device orchestration
     */
    struct OrchestrationConfig
    {
        WeightPlacementStrategy strategy = WeightPlacementStrategy::AUTO;
        DeviceId gpu_device = DeviceId::cuda(0); ///< Which GPU to use (if multiple)
        DeviceId cpu_device = DeviceId::cpu();   ///< CPU device
        int offload_layers = 0;                  ///< Number of layers to keep on GPU (LAYER_SPLIT)

        // Phase 2: Custom device map
        std::string device_map; ///< Custom device map string (e.g., "0-11:gpu:0,12-23:cpu")

        // Phase 2: Memory constraints
        std::optional<size_t> max_gpu_memory_mb; ///< Max GPU memory budget (MB)
        std::optional<size_t> max_cpu_memory_mb; ///< Max CPU memory budget (MB)

        // Phase 2: MoE-specific
        bool moe_shared_experts_gpu = true; ///< Put MoE shared experts on GPU
        bool moe_sparse_experts_cpu = true; ///< Put MoE sparse experts on CPU

        // Phase 6: Multi-GPU
        bool multi_gpu = false;            ///< Enable multi-GPU distribution
        std::string gpu_split = "even";    ///< GPU split strategy: "even", "weighted", or "0.6,0.4"
        std::vector<DeviceId> gpu_devices; ///< Specific GPU devices to use (empty = all GPUs)

        // Phase 6.5: Heterogeneous Multi-Domain Parallelism
        bool heterogeneous_mode = false;   ///< Enable heterogeneous multi-domain TP/PP
        float cpu_compute_fraction = 0.2f; ///< Fraction of layers for CPU domains (0.0-1.0)
        bool enable_gpu_tp = true;         ///< Use GPU tensor parallelism
        bool enable_cpu_tp = true;         ///< Use CPU tensor parallelism
        int min_layers_per_domain = 2;     ///< Minimum layers per TP domain

        bool verbose = false; ///< Log placement decisions
    };

    /**
     * @brief Orchestrates device placement strategies for model weights.
     *
     * The DeviceOrchestrator is the high-level decision maker that considers:
     * - Available hardware (from DeviceManager)
     * - Model architecture (from ModelContext metadata)
     * - User preferences (from CLI flags)
     * - MPI topology (from MPIContext)
     *
     * It produces a WeightPlacementMap that encodes fine-grained placement decisions,
     * which the WeightManager then executes.
     *
     * Phase 1 Strategies:
     * - ALL_GPU: Put everything on GPU device 0
     * - ALL_CPU: Put everything on CPU
     * - LAYER_SPLIT: First N layers on GPU, rest on CPU
     * - AUTO: Fit what we can on GPU based on memory
     *
     * Future Phases:
     * - MoE-aware strategies (shared experts on GPU, sparse on CPU)
     * - Multi-GPU load balancing
     * - Memory-aware auto-tuning
     * - Dynamic migration strategies
     */
    class DeviceOrchestrator
    {
    public:
        /**
         * @brief Construct orchestrator with hardware and model context
         * @param device_mgr Device manager (hardware enumeration)
         * @param mpi_ctx MPI context (rank topology)
         * @param config User-provided configuration
         */
        DeviceOrchestrator(std::shared_ptr<DeviceManager> device_mgr,
                           std::shared_ptr<MPIContext> mpi_ctx,
                           const OrchestrationConfig &config);

        /**
         * @brief Create a placement map for a specific model
         *
         * This is the main entry point. It analyzes the model metadata,
         * applies the configured strategy, and produces a placement map.
         *
         * @param model_ctx Model context with loaded metadata
         * @return Placement map encoding weight→device decisions
         */
        std::shared_ptr<WeightPlacementMap> createPlacementMap(
            const std::shared_ptr<ModelContext> &model_ctx);

        /**
         * @brief Get current strategy
         */
        WeightPlacementStrategy strategy() const { return config_.strategy; }

        /**
         * @brief Get configuration
         */
        const OrchestrationConfig &config() const { return config_; }

        // ========================================================================
        // Public Testing Interface (Phase 2)
        // ========================================================================
        // These methods are public for testing purposes, allowing direct
        // verification of device map parsing logic.

        /**
         * @brief Parse device map string into rules (exposed for testing)
         */
        std::vector<DeviceMapRule> parseDeviceMapString(
            const std::string &device_map_str) const;

    private:
        /**
         * @brief Create ALL_GPU placement map
         */
        std::shared_ptr<WeightPlacementMap> createAllGPUMap(
            const std::shared_ptr<ModelContext> &model_ctx);

        /**
         * @brief Create ALL_CPU placement map
         */
        std::shared_ptr<WeightPlacementMap> createAllCPUMap(
            const std::shared_ptr<ModelContext> &model_ctx);

        /**
         * @brief Create LAYER_SPLIT placement map
         */
        std::shared_ptr<WeightPlacementMap> createLayerSplitMap(
            const std::shared_ptr<ModelContext> &model_ctx);

        /**
         * @brief Create AUTO placement map (memory-aware)
         */
        std::shared_ptr<WeightPlacementMap> createAutoMap(
            const std::shared_ptr<ModelContext> &model_ctx);

        /**
         * @brief Create MEMORY_AWARE placement map (Phase 2)
         */
        std::shared_ptr<WeightPlacementMap> createMemoryAwareMap(
            const std::shared_ptr<ModelContext> &model_ctx);

        /**
         * @brief Create MOE_OPTIMIZED placement map (Phase 2)
         */
        std::shared_ptr<WeightPlacementMap> createMoEOptimizedMap(
            const std::shared_ptr<ModelContext> &model_ctx);

        /**
         * @brief Create CUSTOM placement map from device map string (Phase 2)
         */
        std::shared_ptr<WeightPlacementMap> createCustomMap(
            const std::shared_ptr<ModelContext> &model_ctx);

        /**
         * @brief Create MULTI_GPU placement map (Phase 6)
         *
         * Distributes layers across multiple GPUs using round-robin or
         * weighted distribution based on gpu_split config.
         */
        std::shared_ptr<WeightPlacementMap> createMultiGPUMap(
            const std::shared_ptr<ModelContext> &model_ctx);

        /**
         * @brief Get list of available GPU device indices
         *
         * @return Vector of device indices for GPUs (CUDA or ROCm)
         */
        std::vector<int> getAvailableGPUs() const;

        /**
         * @brief Parse individual device map rule (Phase 2)
         */
        DeviceMapRule parseDeviceMapRule(const std::string &rule_str) const;

        /**
         * @brief Parse device string to DeviceId
         * @param device_type Device type string ("cpu", "cuda", "gpu", "rocm")
         * @param device_id Device ordinal (e.g., 0, 1)
         * @return DeviceId for the specified device
         */
        DeviceId parseDeviceString(const std::string &device_type, int device_id) const;

        /**
         * @brief Apply parsed device map rule to placement map (Phase 2)
         */
        void applyDeviceMapRule(
            std::shared_ptr<WeightPlacementMap> &map,
            const DeviceMapRule &rule,
            const std::shared_ptr<ModelContext> &model_ctx) const;

        /**
         * @brief Detect CPU device index from DeviceManager
         */
        int detectCPUDeviceIndex() const;

        /**
         * @brief Get number of layers from model metadata
         */
        int getLayerCount(const std::shared_ptr<ModelContext> &model_ctx) const;

        /**
         * @brief Estimate memory required for model
         */
        size_t estimateModelMemory(const std::shared_ptr<ModelContext> &model_ctx) const;

        /**
         * @brief Log placement decisions (if verbose enabled)
         */
        void logPlacementDecision(const std::string &message) const;

        std::shared_ptr<DeviceManager> device_mgr_;
        std::shared_ptr<MPIContext> mpi_ctx_;
        OrchestrationConfig config_;
    };

} // namespace llaminar2
