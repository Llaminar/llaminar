/**
 * @file DeviceOrchestrator.cpp
 * @brief Implementation of device placement orchestration
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include "DeviceOrchestrator.h"
#include "../utils/Logger.h"
#include <iostream>
#include <algorithm>

namespace llaminar2
{

    DeviceOrchestrator::DeviceOrchestrator(
        std::shared_ptr<DeviceManager> device_mgr,
        std::shared_ptr<MPIContext> mpi_ctx,
        const OrchestrationConfig &config)
        : device_mgr_(device_mgr), mpi_ctx_(mpi_ctx), config_(config)
    {

        // Auto-detect CPU device if not specified
        if (config_.cpu_device_idx < 0)
        {
            config_.cpu_device_idx = detectCPUDeviceIndex();
        }

        logPlacementDecision("DeviceOrchestrator initialized with strategy: " +
                             std::to_string(static_cast<int>(config_.strategy)));
    }

    std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createPlacementMap(
        const std::shared_ptr<ModelContext> &model_ctx)
    {

        logPlacementDecision("Creating placement map for model: " + model_ctx->path());

        switch (config_.strategy)
        {
        case PlacementStrategy::ALL_GPU:
            return createAllGPUMap(model_ctx);

        case PlacementStrategy::ALL_CPU:
            return createAllCPUMap(model_ctx);

        case PlacementStrategy::LAYER_SPLIT:
            return createLayerSplitMap(model_ctx);

        case PlacementStrategy::AUTO:
            return createAutoMap(model_ctx);

        case PlacementStrategy::MEMORY_AWARE:
            return createMemoryAwareMap(model_ctx);

        case PlacementStrategy::MOE_OPTIMIZED:
            return createMoEOptimizedMap(model_ctx);

        case PlacementStrategy::CUSTOM:
            return createCustomMap(model_ctx);

        default:
            LOG_ERROR("[DeviceOrchestrator] Unknown strategy, falling back to AUTO");
            return createAutoMap(model_ctx);
        }
    }

    std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createAllGPUMap(
        const std::shared_ptr<ModelContext> &model_ctx)
    {

        logPlacementDecision("ALL_GPU: Placing all weights on GPU device " +
                             std::to_string(config_.gpu_device_idx));

        auto map = std::make_shared<WeightPlacementMap>(config_.gpu_device_idx);

        // No additional rules needed - default device is GPU
        return map;
    }

    std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createAllCPUMap(
        const std::shared_ptr<ModelContext> &model_ctx)
    {

        logPlacementDecision("ALL_CPU: Placing all weights on CPU device " +
                             std::to_string(config_.cpu_device_idx));

        auto map = std::make_shared<WeightPlacementMap>(config_.cpu_device_idx);

        // No additional rules needed - default device is CPU
        return map;
    }

    std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createLayerSplitMap(
        const std::shared_ptr<ModelContext> &model_ctx)
    {

        int layer_count = getLayerCount(model_ctx);
        int gpu_layers = config_.offload_layers;

        // Clamp to valid range (if layer_count is 0, assume unlimited)
        if (layer_count > 0)
        {
            gpu_layers = std::max(0, std::min(gpu_layers, layer_count));
        }

        logPlacementDecision("LAYER_SPLIT: " + std::to_string(gpu_layers) +
                             " layers on GPU" +
                             (layer_count > 0 ? (", " + std::to_string(layer_count - gpu_layers) + " layers on CPU") : ""));

        // Default to CPU, then override GPU layers
        auto map = std::make_shared<WeightPlacementMap>(config_.cpu_device_idx);

        // First N layers on GPU
        if (gpu_layers > 0)
        {
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
        const std::shared_ptr<ModelContext> &model_ctx)
    {

        // For Phase 1, AUTO = ALL_GPU if GPU available, else ALL_CPU
        // Future: Memory-aware fitting

        auto devices = device_mgr_->devices();
        bool has_gpu = false;

        for (const auto &device : devices)
        {
            if (device.type == ComputeBackendType::GPU_CUDA || device.type == ComputeBackendType::GPU_ROCM)
            {
                has_gpu = true;
                break;
            }
        }

        if (has_gpu)
        {
            logPlacementDecision("AUTO: GPU detected, using ALL_GPU strategy");
            return createAllGPUMap(model_ctx);
        }
        else
        {
            logPlacementDecision("AUTO: No GPU detected, using ALL_CPU strategy");
            return createAllCPUMap(model_ctx);
        }
    }

    int DeviceOrchestrator::detectCPUDeviceIndex() const
    {
        auto devices = device_mgr_->devices();

        for (size_t i = 0; i < devices.size(); ++i)
        {
            if (devices[i].type == ComputeBackendType::CPU_OPENBLAS)
            {
                return static_cast<int>(i);
            }
        }

        // No explicit CPU device found, return 0 as fallback
        return 0;
    }

    int DeviceOrchestrator::getLayerCount(const std::shared_ptr<ModelContext> &model_ctx) const
    {
        // Try to get from model metadata
        try
        {
            auto model = model_ctx->model();
            return model.block_count;
        }
        catch (...)
        {
            // Model not loaded yet
        }

        // Fallback: return 0 (will be handled gracefully)
        return 0;
    }

    size_t DeviceOrchestrator::estimateModelMemory(
        const std::shared_ptr<ModelContext> &model_ctx) const
    {

        // Estimate total model memory based on tensors
        try
        {
            auto model = model_ctx->model();

            // Sum up tensor sizes (already computed in size_bytes)
            size_t total_bytes = 0;
            for (const auto &tensor_info : model.tensors)
            {
                total_bytes += tensor_info.size_bytes;
            }

            // Add activation memory (rough estimate: 10-20% of weights)
            size_t activation_memory_bytes = total_bytes / 5;

            return total_bytes + activation_memory_bytes;
        }
        catch (...)
        {
            // Model not loaded, can't estimate
            return 0;
        }
    }

    void DeviceOrchestrator::logPlacementDecision(const std::string &message) const
    {
        if (config_.verbose)
        {
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            if (rank == 0)
            {
                LOG_INFO("[DeviceOrchestrator] " << message);
            }
        }
    }

    // ============================================================================
    // Phase 2: Advanced Strategies
    // ============================================================================

    std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createMemoryAwareMap(
        const std::shared_ptr<ModelContext> &model_ctx)
    {

        logPlacementDecision("MEMORY_AWARE: Auto-fitting layers within memory budget");

        // Estimate total model memory
        size_t model_memory_bytes = estimateModelMemory(model_ctx);
        int layer_count = getLayerCount(model_ctx);

        if (layer_count == 0)
        {
            logPlacementDecision("MEMORY_AWARE: Layer count unknown, falling back to AUTO");
            return createAutoMap(model_ctx);
        }

        // Get available GPU memory
        size_t available_gpu_memory_bytes = 0;
        if (config_.max_gpu_memory_mb.has_value())
        {
            available_gpu_memory_bytes = config_.max_gpu_memory_mb.value() * 1024 * 1024;
        }
        else
        {
            // Query from device manager
            auto devices = device_mgr_->devices();
            if (config_.gpu_device_idx >= 0 &&
                static_cast<size_t>(config_.gpu_device_idx) < devices.size())
            {
                available_gpu_memory_bytes = devices[config_.gpu_device_idx].free_memory_bytes;
            }
        }

        if (available_gpu_memory_bytes == 0)
        {
            logPlacementDecision("MEMORY_AWARE: No GPU memory available, using ALL_CPU");
            return createAllCPUMap(model_ctx);
        }

        // Estimate memory per layer (assume uniform distribution)
        size_t memory_per_layer = model_memory_bytes / layer_count;

        // Calculate how many layers fit on GPU
        int gpu_layers = std::min(
            layer_count,
            static_cast<int>(available_gpu_memory_bytes / memory_per_layer));

        logPlacementDecision("MEMORY_AWARE: Fitting " + std::to_string(gpu_layers) +
                             " / " + std::to_string(layer_count) + " layers on GPU (" +
                             std::to_string(available_gpu_memory_bytes / (1024 * 1024)) + " MB available)");

        // Use LAYER_SPLIT strategy with calculated layer count
        OrchestrationConfig temp_config = config_;
        temp_config.strategy = PlacementStrategy::LAYER_SPLIT;
        temp_config.offload_layers = gpu_layers;

        DeviceOrchestrator temp_orchestrator(device_mgr_, mpi_ctx_, temp_config);
        return temp_orchestrator.createLayerSplitMap(model_ctx);
    }

    std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createMoEOptimizedMap(
        const std::shared_ptr<ModelContext> &model_ctx)
    {

        logPlacementDecision("MOE_OPTIMIZED: MoE-aware placement strategy");

        // Start with AUTO as base
        auto map = createAutoMap(model_ctx);

        // Override with MoE-specific rules
        if (config_.moe_shared_experts_gpu)
        {
            // Shared experts on GPU (accessed every token)
            map->setPatternDevice("*shared_expert*", config_.gpu_device_idx);
            map->setPatternDevice("*shared_experts*", config_.gpu_device_idx);
            map->setPatternDevice("*gate*", config_.gpu_device_idx);
            logPlacementDecision("MOE_OPTIMIZED: Shared experts placed on GPU");
        }
        else
        {
            map->setPatternDevice("*shared_expert*", config_.cpu_device_idx);
            map->setPatternDevice("*shared_experts*", config_.cpu_device_idx);
            logPlacementDecision("MOE_OPTIMIZED: Shared experts placed on CPU");
        }

        if (config_.moe_sparse_experts_cpu)
        {
            // Sparse experts on CPU (rarely accessed)
            map->setPatternDevice("*experts.0.*", config_.cpu_device_idx);
            map->setPatternDevice("*experts.1.*", config_.cpu_device_idx);
            map->setPatternDevice("*experts.2.*", config_.cpu_device_idx);
            map->setPatternDevice("*experts.3.*", config_.cpu_device_idx);
            map->setPatternDevice("*experts.4.*", config_.cpu_device_idx);
            map->setPatternDevice("*experts.5.*", config_.cpu_device_idx);
            map->setPatternDevice("*experts.6.*", config_.cpu_device_idx);
            map->setPatternDevice("*experts.7.*", config_.cpu_device_idx);
            logPlacementDecision("MOE_OPTIMIZED: Sparse experts placed on CPU");
        }
        else
        {
            // Sparse experts on GPU
            map->setPatternDevice("*experts.*", config_.gpu_device_idx);
            logPlacementDecision("MOE_OPTIMIZED: Sparse experts placed on GPU");
        }

        return map;
    }

    std::shared_ptr<WeightPlacementMap> DeviceOrchestrator::createCustomMap(
        const std::shared_ptr<ModelContext> &model_ctx)
    {

        if (config_.device_map.empty())
        {
            logPlacementDecision("CUSTOM: No device map provided, falling back to AUTO");
            return createAutoMap(model_ctx);
        }

        logPlacementDecision("CUSTOM: Parsing device map: " + config_.device_map);

        // Start with CPU as default
        auto map = std::make_shared<WeightPlacementMap>(config_.cpu_device_idx);

        // Parse device map string (format: "0-11:gpu:0,12-23:cpu,embed:gpu:0")
        auto rules = parseDeviceMapString(config_.device_map);

        for (const auto &rule : rules)
        {
            applyDeviceMapRule(map, rule, model_ctx);
        }

        return map;
    }

    std::vector<DeviceMapRule> DeviceOrchestrator::parseDeviceMapString(
        const std::string &device_map_str) const
    {

        std::vector<DeviceMapRule> rules;

        // Split by comma
        size_t start = 0;

        while (start < device_map_str.length())
        {
            size_t end = device_map_str.find(',', start);

            std::string rule_str;
            if (end == std::string::npos)
            {
                // Last rule (no more commas)
                rule_str = device_map_str.substr(start);
                start = device_map_str.length(); // Exit loop after this
            }
            else
            {
                // More rules to come
                rule_str = device_map_str.substr(start, end - start);
                start = end + 1; // Move past the comma
            }

            // Parse individual rule
            DeviceMapRule rule = parseDeviceMapRule(rule_str);
            if (rule.type != DeviceMapRuleType::INVALID)
            {
                rules.push_back(rule);
            }
        }

        logPlacementDecision("CUSTOM: Parsed " + std::to_string(rules.size()) + " device map rules");
        return rules;
    }

    DeviceMapRule DeviceOrchestrator::parseDeviceMapRule(
        const std::string &rule_str) const
    {

        DeviceMapRule rule;
        rule.type = DeviceMapRuleType::INVALID;

        // Trim whitespace
        std::string trimmed = rule_str;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        trimmed.erase(trimmed.find_last_not_of(" \t") + 1);

        // Split by colon
        size_t first_colon = trimmed.find(':');
        if (first_colon == std::string::npos)
        {
            logPlacementDecision("CUSTOM: Invalid rule format: " + rule_str);
            return rule;
        }

        std::string target = trimmed.substr(0, first_colon);
        std::string rest = trimmed.substr(first_colon + 1);

        size_t second_colon = rest.find(':');
        std::string device_type = rest;
        int device_id = 0;

        if (second_colon != std::string::npos)
        {
            device_type = rest.substr(0, second_colon);
            device_id = std::stoi(rest.substr(second_colon + 1));
        }

        // Determine device index from type and id
        rule.device_idx = parseDeviceString(device_type, device_id);

        // Parse target (layer range, percentage, or pattern)
        if (target.find('-') != std::string::npos)
        {
            // Layer range: "0-11"
            size_t dash = target.find('-');
            rule.type = DeviceMapRuleType::LAYER_RANGE;
            rule.start_layer = std::stoi(target.substr(0, dash));
            rule.end_layer = std::stoi(target.substr(dash + 1));
        }
        else if (target.find('%') != std::string::npos)
        {
            // Percentage: "first_50%" or "last_25%"
            rule.type = DeviceMapRuleType::PERCENTAGE;

            if (target.find("first_") == 0)
            {
                rule.is_first = true;
                size_t percent_pos = target.find('%');
                rule.percentage = std::stof(target.substr(6, percent_pos - 6));
            }
            else if (target.find("last_") == 0)
            {
                rule.is_first = false;
                size_t percent_pos = target.find('%');
                rule.percentage = std::stof(target.substr(5, percent_pos - 5));
            }
            else
            {
                logPlacementDecision("CUSTOM: Invalid percentage format: " + target);
                return rule;
            }
        }
        else
        {
            // Pattern: "embed", "experts.0", "*lm_head*"
            rule.type = DeviceMapRuleType::PATTERN;
            rule.pattern = target;
        }

        return rule;
    }

    int DeviceOrchestrator::parseDeviceString(
        const std::string &device_type, int device_id) const
    {

        if (device_type == "cpu")
        {
            return config_.cpu_device_idx;
        }
        else if (device_type == "gpu" || device_type == "cuda")
        {
            // Find GPU device by ID
            auto devices = device_mgr_->devices();
            for (size_t i = 0; i < devices.size(); ++i)
            {
                if (devices[i].type == ComputeBackendType::GPU_CUDA &&
                    devices[i].device_id == device_id)
                {
                    return static_cast<int>(i);
                }
            }
            // Fallback to configured GPU device
            return config_.gpu_device_idx;
        }
        else if (device_type == "rocm")
        {
            // Find ROCm device by ID
            auto devices = device_mgr_->devices();
            for (size_t i = 0; i < devices.size(); ++i)
            {
                if (devices[i].type == ComputeBackendType::GPU_ROCM &&
                    devices[i].device_id == device_id)
                {
                    return static_cast<int>(i);
                }
            }
            return config_.gpu_device_idx;
        }

        return config_.cpu_device_idx;
    }

    void DeviceOrchestrator::applyDeviceMapRule(
        std::shared_ptr<WeightPlacementMap> &map,
        const DeviceMapRule &rule,
        const std::shared_ptr<ModelContext> &model_ctx) const
    {

        switch (rule.type)
        {
        case DeviceMapRuleType::LAYER_RANGE:
            map->setLayerRange(rule.start_layer, rule.end_layer, rule.device_idx);
            logPlacementDecision("CUSTOM: Layers " + std::to_string(rule.start_layer) +
                                 "-" + std::to_string(rule.end_layer) +
                                 " -> device " + std::to_string(rule.device_idx));
            break;

        case DeviceMapRuleType::PERCENTAGE:
        {
            int layer_count = getLayerCount(model_ctx);
            if (layer_count > 0)
            {
                int num_layers = static_cast<int>(layer_count * rule.percentage / 100.0f);
                int start = rule.is_first ? 0 : (layer_count - num_layers);
                int end = rule.is_first ? (num_layers - 1) : (layer_count - 1);
                map->setLayerRange(start, end, rule.device_idx);
                logPlacementDecision("CUSTOM: " + std::string(rule.is_first ? "First" : "Last") +
                                     " " + std::to_string(static_cast<int>(rule.percentage)) + "% (" +
                                     std::to_string(num_layers) + " layers) -> device " +
                                     std::to_string(rule.device_idx));
            }
            break;
        }

        case DeviceMapRuleType::PATTERN:
            map->setPatternDevice(rule.pattern, rule.device_idx);
            logPlacementDecision("CUSTOM: Pattern '" + rule.pattern +
                                 "' -> device " + std::to_string(rule.device_idx));
            break;

        case DeviceMapRuleType::INVALID:
            // Skip invalid rules
            break;
        }
    }

} // namespace llaminar2
