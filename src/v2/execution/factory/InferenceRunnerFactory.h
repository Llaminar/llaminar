/**
 * @file InferenceRunnerFactory.h
 * @brief Factory for creating IInferenceRunner implementations
 * @author David Sanftenberg
 * @date December 2025
 *
 * @deprecated This factory is now an INTERNAL implementation detail.
 *             Use IOrchestrationRunnerFactory for new code.
 *
 * This file provides factory functions for creating inference runners.
 * The factory handles:
 * - Building Qwen2GraphConfig from GGUF model metadata
 * - Configuring tensor parallelism (GLOBAL via MPI or LOCAL via ILocalTPContext)
 * - Loading and wiring up model weights
 * - Initializing DeviceGraphOrchestrator with proper state
 *
 * MIGRATION GUIDE:
 * ================
 * Old API (deprecated):
 *   #include "execution/factory/InferenceRunnerFactory.h"
 *   auto runner = createInferenceRunner(model_ctx, mpi_ctx, device, config);
 *   runner->forward(tokens.data(), seq_len);
 *
 * New API (recommended):
 *   #include "execution/runner/IOrchestrationRunnerFactory.h"
 *   auto factory = createOrchestrationRunnerFactory();
 *   auto runner = factory->createSimple(model_path, "cuda:0");
 *   runner->prefill(tokens);
 *   auto result = runner->decodeStep();
 *
 * For tests, use TestOrchestrationHelper:
 *   #include "utils/TestOrchestrationHelper.h"
 *   auto runner = TestOrchestrationHelper::createSimple(model_path, DeviceId::cuda(0));
 *
 * Two TP modes are supported:
 *
 * 1. GLOBAL TP (MPI-based): Multiple MPI ranks, each with one device
 *    - Configured via mpi_ctx with world_size > 1
 *    - Collectives via MPI with host staging
 *    - Use: mpirun -np 2 llaminar2 --tp 2 --tp-scope global
 *
 * 2. LOCAL TP (single-rank): One MPI rank with multiple devices
 *    - Configured via ILocalTPContext in InferenceRunnerConfig
 *    - Collectives via NCCL/RCCL/PCIeBAR (no host staging)
 *    - Use: llaminar2 --tp 2 --tp-scope local --tp-devices cuda:0,rocm:0
 *
 * @code
 * // GLOBAL TP: via MPI world_size
 * auto runner = createInferenceRunner(model_ctx, mpi_ctx, device, config);
 *
 * // LOCAL TP: via ILocalTPContext
 * auto tp_ctx = createLocalTPContext({DeviceId::cuda(0), DeviceId::rocm(0)}, ...);
 * InferenceRunnerConfig config;
 * config.local_tp_ctx = tp_ctx.get();
 * auto runner = createInferenceRunner(model_ctx, nullptr, DeviceId::cuda(0), config);
 * @endcode
 */

#pragma once

#include "../local_execution/orchestrators/IInferenceRunner.h"
#include "../../loaders/ModelContext.h"
#include "../../interfaces/IModelContext.h"
#include "../config/RuntimeConfig.h"
#include "../local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "../../backends/DeviceId.h"
#include "../../utils/MPIContext.h"
#include "../mpi_orchestration/PlacementPlan.h"
#include "../../config/PipelineConfig.h"
#include "FactoryPPStageConfig.h"
#include <memory>
#include <optional>

namespace llaminar2
{
    // Forward declarations to avoid pulling in full headers
    class DeviceGraphOrchestrator;
    class ILocalTPContext;
    class IMultiDeviceOrchestrator;

    // Note: FactoryPPStageConfig is now defined in FactoryPPStageConfig.h
    // to avoid circular dependencies with MultiDeviceOrchestrator.h

    /**
     * @brief Configuration for inference runner creation
     */
    struct InferenceRunnerConfig
    {
        int max_seq_len = 4096;
        int batch_size = 1;

        // Explicit graph path selection (only graph path is supported)
        bool force_graph = false;

        // Activation precision
        ActivationPrecision activation_precision = ActivationPrecision::FP32;

        // Fused attention backend selection
        // - JIT: AVX-512 VNNI optimized (default)
        // - REFERENCE: Pure C++ implementation for testing
        // - TILED: Cache-blocked implementation
        // - Q16_INTEGER: Pure Q16 integer-domain kernel (experimental)
        FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;

        // Fixed scale for Q16_1 KV cache quantization (FP32 range: ±kv_cache_scale)
        // See RuntimeConfig.h for detailed documentation on scale selection.
        // Default 64.0f covers Q8_1 activation ranges (max ±35) with ~2× headroom.
        float kv_cache_scale = 256.0f; ///< Fixed Q16 scale. Must cover Q projection max (~130 for Qwen2)

        // Use mapped memory for GPU tensor allocation (zero-copy host access)
        // When true, FP32 activation buffers are allocated using cudaHostAllocMapped /
        // hipHostMallocMapped, enabling direct host access without memcpy.
        // Essential for snapshot capture mode on GPU to avoid slow D2H syncs.
        bool use_mapped_memory = false;

        // Orchestration: computed PlacementPlan for layer device assignment
        // When set, the inference runner will use this plan to determine which
        // device executes each layer. If not set, defaults to single-device execution.
        std::optional<PlacementPlan> placement_plan;

        // =====================================================================
        // LOCAL Tensor Parallelism (single MPI rank, multiple devices)
        // =====================================================================
        // When local_tp_ctx is set, the factory configures LOCAL TP mode:
        //   - Weight sharding based on local_tp_ctx->devices() and weights()
        //   - Collectives via local_tp_ctx (NCCL/RCCL/PCIeBAR, not MPI)
        //   - The 'device' parameter becomes the "primary" device for this runner
        //
        // This is mutually exclusive with GLOBAL TP (mpi_ctx->world_size() > 1).
        // If both are set, LOCAL TP takes precedence.
        //
        // Lifetime: Caller owns the ILocalTPContext and must keep it alive
        // for the duration of the IInferenceRunner's lifetime.
        // =====================================================================
        ILocalTPContext *local_tp_ctx = nullptr;

        /// Device index within the LOCAL TP context (0 to degree-1).
        /// Determines which portion of sharded weights this runner loads.
        /// Only used when local_tp_ctx is set.
        int local_tp_device_index = 0;

        /// Allocate hidden state in PCIe BAR region for cross-vendor PP transfers.
        /// When true and device is ROCm, the final hidden state buffer is allocated
        /// via PCIeBARBackend::allocateInBarRegion() to enable zero-copy transfers
        /// to CUDA devices in subsequent PP stages.
        bool use_bar_backed_hidden = false;
    };

    /**
     * @brief Factory function to create DeviceGraphOrchestrator inference runner
     *
     * @param model_ctx Model context with weights
     * @param mpi_ctx MPI context (nullptr for single-rank)
     * @param device Target device
     * @param config Runner configuration
     * @return Unique pointer to IInferenceRunner, or nullptr on failure
     */
    std::unique_ptr<IInferenceRunner> createInferenceRunner(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config = {});

    /**
     * @brief Factory function to create DeviceGraphOrchestrator with injected dependencies
     *
     * This factory function enables unit testing by accepting interface types
     * instead of concrete implementations. Use MockModelContext, MockMPITopology,
     * etc. for isolated testing without GGUF files or MPI runtime.
     *
     * @param model_ctx Model context interface (can be MockModelContext)
     * @param device Target device
     * @param config Runner configuration
     * @return Unique pointer to IInferenceRunner, or nullptr on failure
     *
     * @code
     * // Unit test example
     * auto mock_ctx = MockModelContextBuilder()
     *     .usePreset(ModelPreset::QWEN2_05B)
     *     .build();
     * auto runner = createTestableInferenceRunner(mock_ctx, DeviceId::cpu());
     * runner->forward(tokens.data(), seq_len);
     * @endcode
     */
    std::unique_ptr<IInferenceRunner> createTestableInferenceRunner(
        std::shared_ptr<IModelContext> model_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config = {});

    /**
     * @brief Factory function to create a unified LOCAL PP runner
     *
     * Creates a DeviceGraphOrchestrator configured for LOCAL Pipeline Parallelism
     * (multiple PP stages on multiple local devices within a single MPI rank).
     * The factory handles:
     * - Building Qwen2GraphConfig from model metadata
     * - Calling setPipelineConfig() on the orchestrator
     * - Auto-configuring weights for each layer's device using getWeightForDevice()
     * - Initializing PP contexts for inter-stage activation transfers
     *
     * This is the **production entry point** for LOCAL PP. Tests and production
     * code should use this instead of manually wiring weights.
     *
     * @param model_ctx Model context with weights (REPLICATED strategy recommended)
     * @param pipeline_config Complete pipeline configuration (TP domains + PP stages)
     * @param config General runner configuration
     * @return Unique pointer to IInferenceRunner, or nullptr on failure
     *
     * @code
     * // Example: 2-stage LOCAL PP with CUDA and CPU
     * auto pipeline_config = std::make_shared<PipelineConfig>();
     * pipeline_config->total_layers = 24;
     * pipeline_config->tp_domains = {
     *     {"stage0_domain", {DeviceId::cuda(0)}, CollectiveBackendType::HOST},
     *     {"stage1_domain", {DeviceId::cpu()}, CollectiveBackendType::HOST}
     * };
     * pipeline_config->pp_stages = {
     *     PPStageConfig::firstStage(0, "stage0_domain", 0, 12),
     *     PPStageConfig::lastStage(1, "stage1_domain", 12, 24)
     * };
     * pipeline_config->pp_transfer_backends[{0, 1}] = CollectiveBackendType::HOST;
     *
     * auto runner = createUnifiedPipelineRunner(model_ctx, pipeline_config);
     * runner->forward(tokens.data(), seq_len);
     * @endcode
     */
    std::unique_ptr<IInferenceRunner> createUnifiedPipelineRunner(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<PipelineConfig> pipeline_config,
        const InferenceRunnerConfig &config = {});

    /**
     * @brief Factory function to create a Pipeline Parallelism stage runner
     *
     * Creates an inference runner that executes only a subset of transformer layers,
     * suitable for Pipeline Parallelism deployments. The stage receives activations
     * from the previous stage and produces activations for the next stage.
     *
     * @param stage_ctx Model context with weights (may contain only this stage's weights)
     * @param device Target device for this stage
     * @param pp_config Pipeline parallelism stage configuration
     * @param config General runner configuration
     * @return Unique pointer to IInferenceRunner, or nullptr on failure
     *
     * @note Implementation is provided in a future commit. This is a forward declaration.
     *
     * @code
     * // Example: Create stage 0 of a 2-stage PP setup
     * FactoryPPStageConfig pp_config{
     *     .first_layer = 0,
     *     .last_layer = 12,
     *     .has_embedding = true,
     *     .has_lm_head = false
     * };
     * auto stage_runner = createPPStageRunner(model_ctx, DeviceId::cuda(0), pp_config);
     * @endcode
     */
    std::unique_ptr<IInferenceRunner> createPPStageRunner(
        std::shared_ptr<ModelContext> stage_ctx,
        DeviceId device,
        const FactoryPPStageConfig &pp_config,
        const InferenceRunnerConfig &config = {});

    /**
     * @brief Factory function to create MultiDeviceOrchestrator for LOCAL TP
     *
     * Creates a MultiDeviceOrchestrator that coordinates multiple devices within
     * a single MPI rank. Use this when LOCAL TP is configured via --tp-scope local.
     *
     * @param model_ctx Model context with weights
     * @param tp_ctx Pre-constructed LOCAL TP context (ownership transferred)
     * @param config Multi-device configuration
     * @return Unique pointer to IMultiDeviceOrchestrator (extends IInferenceRunner)
     *
     * @code
     * // Create LOCAL TP context
     * auto tp_ctx = createLocalTPContext(
     *     {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
     *     {0.5f, 0.5f},
     *     CollectiveBackendType::NCCL);
     *
     * // Create multi-device config
     * MultiDeviceOrchestrator::Config config;
     * config.devices = tp_ctx->devices();
     * config.weights = tp_ctx->weights();
     * config.backend = CollectiveBackendType::NCCL;
     *
     * // Create orchestrator (returns IInferenceRunner-compatible)
     * auto runner = createMultiDeviceOrchestrator(model_ctx, std::move(tp_ctx), config);
     * @endcode
     */
    std::unique_ptr<IMultiDeviceOrchestrator> createMultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const MultiDeviceOrchestrator::Config &config);

    /**
     * @brief Factory function to create MultiDeviceOrchestrator with testable dependencies
     *
     * For unit testing: allows injecting mock device runners and TP context.
     *
     * @param model_ctx Model context (can be MockModelContext)
     * @param device_runners Pre-constructed per-device runners (ownership transferred)
     * @param tp_ctx Pre-constructed LOCAL TP context (can be mock)
     * @param config Multi-device configuration
     * @return Unique pointer to IMultiDeviceOrchestrator
     */
    std::unique_ptr<IMultiDeviceOrchestrator> createTestableMultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const MultiDeviceOrchestrator::Config &config);

} // namespace llaminar2
