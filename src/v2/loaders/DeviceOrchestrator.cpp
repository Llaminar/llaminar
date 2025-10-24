/**
 * @file DeviceOrchestrator.cpp
 * @brief Implementation of device placement orchestration
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include "DeviceOrchestrator.h"
#include <iostream>
#include <algorithm>

namespace llaminar2 {

DeviceOrchestrator::DeviceOrchestrator(
    std::shared_ptr<DeviceManager> device_mgr,
    std::shared_ptr<MPIContext> mpi_ctx,
    const OrchestrationConfig& config)
    : device_mgr_(device_mgr)
    , mpi_ctx_(mpi_ctx)
    , config_(config) {
    
    // Auto-detect CPU device if not specified
    if (config_.cpu_device_idx < 0) {
        config_.cpu_device_idx = detectCPUDeviceIndex();
    }
    
    logPlacementDecision("DeviceOrchestrator initialized with strategy: " + 
                        std::to_string(static_cast<int>(config_.strategy)));
}

std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createPlacementMap(
    const std::shared_ptr<ModelContext>& model_ctx) {
    
    logPlacementDecision("Creating placement map for model: " + model_ctx->path());
    
    switch (config_.strategy) {
        case PlacementStrategy::ALL_GPU:
            return createAllGPUMap(model_ctx);
        
        case PlacementStrategy::ALL_CPU:
            return createAllCPUMap(model_ctx);
        
        case PlacementStrategy::LAYER_SPLIT:
            return createLayerSplitMap(model_ctx);
        
        case PlacementStrategy::AUTO:
            return createAutoMap(model_ctx);
        
        case PlacementStrategy::CUSTOM:
            // TODO: Parse custom device map string
            std::cerr << "[DeviceOrchestrator] CUSTOM strategy not yet implemented, falling back to AUTO" << std::endl;
            return createAutoMap(model_ctx);
        
        default:
            std::cerr << "[DeviceOrchestrator] Unknown strategy, falling back to AUTO" << std::endl;
            return createAutoMap(model_ctx);
    }
}

std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createAllGPUMap(
    const std::shared_ptr<ModelContext>& model_ctx) {
    
    logPlacementDecision("ALL_GPU: Placing all weights on GPU device " + 
                        std::to_string(config_.gpu_device_idx));
    
    auto map = std::make_shared<WeightPlacementMap>(config_.gpu_device_idx);
    
    // No additional rules needed - default device is GPU
    return map;
}

std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createAllCPUMap(
    const std::shared_ptr<ModelContext>& model_ctx) {
    
    logPlacementDecision("ALL_CPU: Placing all weights on CPU device " + 
                        std::to_string(config_.cpu_device_idx));
    
    auto map = std::make_shared<WeightPlacementMap>(config_.cpu_device_idx);
    
    // No additional rules needed - default device is CPU
    return map;
}

std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createLayerSplitMap(
    const std::shared_ptr<ModelContext>& model_ctx) {
    
    int layer_count = getLayerCount(model_ctx);
    int gpu_layers = config_.offload_layers;
    
    // Clamp to valid range (if layer_count is 0, assume unlimited)
    if (layer_count > 0) {
        gpu_layers = std::max(0, std::min(gpu_layers, layer_count));
    }
    
    logPlacementDecision("LAYER_SPLIT: " + std::to_string(gpu_layers) + 
                        " layers on GPU" + 
                        (layer_count > 0 ? (", " + std::to_string(layer_count - gpu_layers) + " layers on CPU") : ""));
    
    // Default to CPU, then override GPU layers
    auto map = std::make_shared<WeightPlacementMap>(config_.cpu_device_idx);
    
    // First N layers on GPU
    if (gpu_layers > 0) {
        map->setLayerRange(0, gpu_layers - 1, config_.gpu_device_idx);
    }
    
    // Embeddings typically on GPU for best performance
    map->setPatternDevice("*embd*", config_.gpu_device_idx);
    map->setPatternDevice("token_embd.weight", config_.gpu_device_idx);
    
    // Output head on GPU (accessed every token)
    map->setPatternDevice("output.weight", config_.gpu_device_idx);
    map->setPatternDevice("*lm_head*", config_.gpu_device_idx);
    
    return map;
}

std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createAutoMap(
    const std::shared_ptr<ModelContext>& model_ctx) {
    
    // For Phase 1, AUTO = ALL_GPU if GPU available, else ALL_CPU
    // Future: Memory-aware fitting
    
    auto devices = device_mgr_->devices();
    bool has_gpu = false;
    
    for (const auto& device : devices) {
        if (device.type == ComputeBackendType::GPU_CUDA || device.type == ComputeBackendType::GPU_ROCM) {
            has_gpu = true;
            break;
        }
    }
    
    if (has_gpu) {
        logPlacementDecision("AUTO: GPU detected, using ALL_GPU strategy");
        return createAllGPUMap(model_ctx);
    } else {
        logPlacementDecision("AUTO: No GPU detected, using ALL_CPU strategy");
        return createAllCPUMap(model_ctx);
    }
}

int DeviceOrchestrator::detectCPUDeviceIndex() const {
    auto devices = device_mgr_->devices();
    
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].type == ComputeBackendType::CPU_OPENBLAS) {
            return static_cast<int>(i);
        }
    }
    
    // No explicit CPU device found, return 0 as fallback
    return 0;
}

int DeviceOrchestrator::getLayerCount(const std::shared_ptr<ModelContext>& model_ctx) const {
    // Try to get from model metadata
    try {
        auto model = model_ctx->model();
        return model.block_count;
    } catch (...) {
        // Model not loaded yet
    }
    
    // Fallback: return 0 (will be handled gracefully)
    return 0;
}

size_t DeviceOrchestrator::estimateModelMemory(
    const std::shared_ptr<ModelContext>& model_ctx) const {
    
    // TODO: Implement memory estimation based on:
    // - Parameter count from metadata
    // - Quantization type
    // - Activation memory requirements
    
    // For now, return 0 (not used in Phase 1)
    return 0;
}

void DeviceOrchestrator::logPlacementDecision(const std::string& message) const {
    if (config_.verbose) {
        int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        if (rank == 0) {
            std::cout << "[DeviceOrchestrator] " << message << std::endl;
        }
    }
}

} // namespace llaminar2
