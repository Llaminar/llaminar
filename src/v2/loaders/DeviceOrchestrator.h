/**
 * @file DeviceOrchestrator.h
 * @brief High-level orchestration of device placement strategies
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#pragma once

#include "WeightPlacementMap.h"
#include "backends/ComputeBackend.h"
#include "utils/MPIContext.h"
#include "ModelContext.h"
#include <memory>
#include <string>

namespace llaminar2 {

/**
 * @brief Placement strategy types supported by the orchestrator
 */
enum class PlacementStrategy {
    ALL_GPU,        ///< All weights on GPU device 0
    ALL_CPU,        ///< All weights on CPU
    LAYER_SPLIT,    ///< First N layers on GPU, rest on CPU
    AUTO,           ///< Automatic based on available memory
    CUSTOM          ///< User-provided custom mapping
};

/**
 * @brief Configuration for device orchestration
 */
struct OrchestrationConfig {
    PlacementStrategy strategy = PlacementStrategy::AUTO;
    int gpu_device_idx = 0;           ///< Which GPU to use (if multiple)
    int cpu_device_idx = -1;          ///< CPU device index (or -1 for auto-detect)
    int offload_layers = 0;           ///< Number of layers to keep on GPU (LAYER_SPLIT)
    std::string device_map_str;       ///< Custom device map string (future)
    bool verbose = false;             ///< Log placement decisions
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
class DeviceOrchestrator {
public:
    /**
     * @brief Construct orchestrator with hardware and model context
     * @param device_mgr Device manager (hardware enumeration)
     * @param mpi_ctx MPI context (rank topology)
     * @param config User-provided configuration
     */
    DeviceOrchestrator(std::shared_ptr<DeviceManager> device_mgr,
                      std::shared_ptr<MPIContext> mpi_ctx,
                      const OrchestrationConfig& config);

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
        const std::shared_ptr<ModelContext>& model_ctx);

    /**
     * @brief Get current strategy
     */
    PlacementStrategy strategy() const { return config_.strategy; }

    /**
     * @brief Get configuration
     */
    const OrchestrationConfig& config() const { return config_; }

private:
    /**
     * @brief Create ALL_GPU placement map
     */
    std::shared_ptr<WeightPlacementMap> createAllGPUMap(
        const std::shared_ptr<ModelContext>& model_ctx);

    /**
     * @brief Create ALL_CPU placement map
     */
    std::shared_ptr<WeightPlacementMap> createAllCPUMap(
        const std::shared_ptr<ModelContext>& model_ctx);

    /**
     * @brief Create LAYER_SPLIT placement map
     */
    std::shared_ptr<WeightPlacementMap> createLayerSplitMap(
        const std::shared_ptr<ModelContext>& model_ctx);

    /**
     * @brief Create AUTO placement map (memory-aware)
     */
    std::shared_ptr<WeightPlacementMap> createAutoMap(
        const std::shared_ptr<ModelContext>& model_ctx);

    /**
     * @brief Detect CPU device index from DeviceManager
     */
    int detectCPUDeviceIndex() const;

    /**
     * @brief Get number of layers from model metadata
     */
    int getLayerCount(const std::shared_ptr<ModelContext>& model_ctx) const;

    /**
     * @brief Estimate memory required for model
     */
    size_t estimateModelMemory(const std::shared_ptr<ModelContext>& model_ctx) const;

    /**
     * @brief Log placement decisions (if verbose enabled)
     */
    void logPlacementDecision(const std::string& message) const;

    std::shared_ptr<DeviceManager> device_mgr_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    OrchestrationConfig config_;
};

} // namespace llaminar2
