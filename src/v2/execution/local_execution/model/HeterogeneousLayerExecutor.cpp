/**
 * @file HeterogeneousLayerExecutor.cpp
 * @brief Implementation of heterogeneous layer executor
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "HeterogeneousLayerExecutor.h"
#include "../graph/GraphExecutor.h"
#include "../device/DeviceContext.h"
#include "../graph/GraphBufferManager.h"
#include "../../../interfaces/ICollectiveContext.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

    // =============================================================================
    // Construction / Destruction
    // =============================================================================

    HeterogeneousLayerExecutor::HeterogeneousLayerExecutor(Config config)
        : config_(std::move(config)), stats_{}
    {
        if (!config_.placement_config)
        {
            throw std::invalid_argument(
                "HeterogeneousLayerExecutor requires a non-null placement_config");
        }

        LOG_DEBUG("HeterogeneousLayerExecutor created with "
                  << config_.placement_config->numLayers() << " layers, "
                  << config_.placement_config->deviceCount() << " devices");

        if (config_.enable_profiling)
        {
            LOG_DEBUG("HeterogeneousLayerExecutor profiling enabled");
        }
    }

    HeterogeneousLayerExecutor::~HeterogeneousLayerExecutor() = default;

    HeterogeneousLayerExecutor::HeterogeneousLayerExecutor(HeterogeneousLayerExecutor &&) noexcept = default;
    HeterogeneousLayerExecutor &HeterogeneousLayerExecutor::operator=(HeterogeneousLayerExecutor &&) noexcept = default;

    // =============================================================================
    // Layer Execution
    // =============================================================================

    bool HeterogeneousLayerExecutor::executeLayer(int layer_idx, ComputeGraph *graph)
    {
        if (!graph)
        {
            LOG_ERROR("HeterogeneousLayerExecutor::executeLayer: null graph");
            return false;
        }

        // Get the device for this layer
        DeviceId device = getDeviceForLayer(layer_idx);

        LOG_TRACE("Executing layer " << layer_idx << " on " << device.to_string());

        // Dispatch to appropriate device executor
        if (device.is_gpu())
        {
            return executeOnGPU(layer_idx, graph);
        }
        else
        {
            return executeOnCPU(layer_idx, graph);
        }
    }

    bool HeterogeneousLayerExecutor::executeLayerRange(int start_layer, int end_layer, ComputeGraph *graph)
    {
        if (!graph)
        {
            LOG_ERROR("HeterogeneousLayerExecutor::executeLayerRange: null graph");
            return false;
        }

        if (start_layer >= end_layer)
        {
            LOG_WARN("HeterogeneousLayerExecutor::executeLayerRange: empty range ["
                     << start_layer << ", " << end_layer << ")");
            return true; // Empty range is a no-op success
        }

        LOG_DEBUG("HeterogeneousLayerExecutor executing layers [" << start_layer
                                                                  << ", " << end_layer << ")");

        for (int layer_idx = start_layer; layer_idx < end_layer; ++layer_idx)
        {
            // Check for cross-domain transfer before this layer
            if (layer_idx > start_layer && requiresCrossDomainTransfer(layer_idx - 1, layer_idx))
            {
                LOG_DEBUG("Cross-domain transfer required: layer " << (layer_idx - 1)
                                                                   << " -> layer " << layer_idx);

                auto transfer_start = Clock::now();

                if (!transferActivations(layer_idx - 1, layer_idx, graph))
                {
                    LOG_ERROR("Failed to transfer activations between layers "
                              << (layer_idx - 1) << " and " << layer_idx);
                    return false;
                }

                if (config_.enable_profiling)
                {
                    auto transfer_end = Clock::now();
                    double elapsed_ms = std::chrono::duration<double, std::milli>(
                                            transfer_end - transfer_start)
                                            .count();
                    stats_.transfer_time_ms += elapsed_ms;
                }

                stats_.cross_domain_transfers++;
            }

            // Execute the layer
            if (!executeLayer(layer_idx, graph))
            {
                LOG_ERROR("Failed to execute layer " << layer_idx);
                return false;
            }
        }

        return true;
    }

    // =============================================================================
    // Device/Domain Queries
    // =============================================================================

    DeviceId HeterogeneousLayerExecutor::getDeviceForLayer(int layer_idx) const
    {
        return config_.placement_config->deviceForLayer(layer_idx);
    }

    const TPDomain *HeterogeneousLayerExecutor::getDomainForLayer(int layer_idx) const
    {
        if (!config_.tp_config)
        {
            return nullptr;
        }

        // Default to attention domain; caller can specify differently if needed
        // For heterogeneous execution, attention and FFN may have different domains
        return config_.tp_config->domainForLayer(layer_idx, /*is_attention=*/true);
    }

    bool HeterogeneousLayerExecutor::requiresCrossDomainTransfer(int from_layer, int to_layer) const
    {
        // Bounds check
        if (from_layer < 0 || to_layer < 0 ||
            from_layer >= config_.placement_config->numLayers() ||
            to_layer >= config_.placement_config->numLayers())
        {
            return false;
        }

        DeviceId from_device = config_.placement_config->deviceForLayer(from_layer);
        DeviceId to_device = config_.placement_config->deviceForLayer(to_layer);

        // Transfer required if devices are different
        return from_device != to_device;
    }

    // =============================================================================
    // Internal Execution Methods
    // =============================================================================

    bool HeterogeneousLayerExecutor::executeOnGPU(int layer_idx, ComputeGraph *graph)
    {
        auto start = Clock::now();

        bool success = executeLayerStages(layer_idx, graph, config_.gpu_context);

        if (config_.enable_profiling)
        {
            auto end = Clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            stats_.gpu_time_ms += elapsed_ms;
            LOG_TRACE("Layer " << layer_idx << " GPU execution: " << elapsed_ms << " ms");
        }

        if (success)
        {
            stats_.gpu_layers_executed++;
        }

        return success;
    }

    bool HeterogeneousLayerExecutor::executeOnCPU(int layer_idx, ComputeGraph *graph)
    {
        auto start = Clock::now();

        bool success = executeLayerStages(layer_idx, graph, config_.cpu_context);

        if (config_.enable_profiling)
        {
            auto end = Clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            stats_.cpu_time_ms += elapsed_ms;
            LOG_TRACE("Layer " << layer_idx << " CPU execution: " << elapsed_ms << " ms");
        }

        if (success)
        {
            stats_.cpu_layers_executed++;
        }

        return success;
    }

    bool HeterogeneousLayerExecutor::executeLayerStages(
        int layer_idx,
        ComputeGraph *graph,
        IDeviceContext *ctx)
    {
        // Phase 5.1: Basic implementation - delegate to existing infrastructure
        //
        // In a full implementation, this would:
        // 1. Get the execution order from the graph
        // 2. Filter to stages for this specific layer
        // 3. Execute them in order on the given device context
        //
        // For now, we mark success and track stats. The actual stage execution
        // is handled by the pipeline/DeviceGraphOrchestrator which already calls
        // GraphExecutor with the appropriate context.

        LOG_TRACE("executeLayerStages: layer " << layer_idx
                                               << " with " << (ctx ? "valid" : "null") << " context");

        // If no context provided, the stages will use their default execution path
        // This is acceptable for Phase 5.1 where we're establishing the routing framework

        // In Phase 5.3, this will integrate with GraphExecutor::executeMultiDevice()
        // to actually run the stages on the appropriate device

        return true; // Success - actual execution delegated to existing infrastructure
    }

    bool HeterogeneousLayerExecutor::transferActivations(
        int from_layer,
        int to_layer,
        ComputeGraph *graph)
    {
        // Phase 5.1: Placeholder implementation
        //
        // In a full implementation (Phase 5.3), this would:
        // 1. Identify the activation tensors that flow between layers
        // 2. Copy them from from_layer's device to to_layer's device
        // 3. Handle the specific transfer type (GPU->CPU, CPU->GPU, or GPU->GPU)
        //
        // Transfer types:
        // - CPU->GPU: cudaMemcpyHostToDevice / hipMemcpyHostToDevice
        // - GPU->CPU: cudaMemcpyDeviceToHost / hipMemcpyDeviceToHost
        // - GPU->GPU: P2P copy or staged through host

        DeviceId from_device = config_.placement_config->deviceForLayer(from_layer);
        DeviceId to_device = config_.placement_config->deviceForLayer(to_layer);

        LOG_DEBUG("Activation transfer: " << from_device.to_string()
                                          << " -> " << to_device.to_string()
                                          << " (layer " << from_layer << " -> " << to_layer << ")");

        // Determine transfer direction for logging
        if (from_device.is_cpu() && to_device.is_gpu())
        {
            LOG_TRACE("Transfer type: CPU -> GPU (cudaMemcpyHostToDevice)");
        }
        else if (from_device.is_gpu() && to_device.is_cpu())
        {
            LOG_TRACE("Transfer type: GPU -> CPU (cudaMemcpyDeviceToHost)");
        }
        else if (from_device.is_gpu() && to_device.is_gpu())
        {
            LOG_TRACE("Transfer type: GPU -> GPU (P2P or staged)");
        }

        // Phase 5.1: Return success as placeholder
        // Actual transfer implementation will come in Phase 5.3
        return true;
    }

} // namespace llaminar2
