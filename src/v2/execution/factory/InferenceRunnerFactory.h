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
 * - Building GraphConfig from GGUF model metadata
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
 *    - Collectives via NCCL/RCCL/HOST
 *    - Use: llaminar2 --tp 2 --tp-scope local --tp-devices cuda:0,rocm:0
 *
 * @code
 * // GLOBAL TP: via MPI world_size
 * auto runner = createInferenceRunner(model_ctx, mpi_ctx, device, config);
 *
 * // LOCAL TP: via ILocalTPContext
 * auto tp_ctx = createLocalTPContext({DeviceId::cuda(0), DeviceId::rocm(0)}, ...);
 * InferenceRunnerConfig config;
 * config.tp_ctx = tp_ctx.get();
 * auto runner = createInferenceRunner(model_ctx, nullptr, DeviceId::cuda(0), config);
 * @endcode
 */

#pragma once

#include "../local_execution/orchestrators/IInferenceRunner.h"
#include "../../loaders/ModelContext.h"
#include "../../interfaces/IModelContext.h"
#include "../config/RuntimeConfig.h"
#include "../local_execution/orchestrators/RankOrchestrator.h"
#include "../../backends/DeviceId.h"
#include "../../utils/MPIContext.h"
#include "../mpi_orchestration/PlacementPlan.h"
#include "../mpi_orchestration/RankExecutionPlan.h"
#include "../../config/PipelineConfig.h"
#include "FactoryPPStageConfig.h"
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace llaminar2
{
    // Forward declarations to avoid pulling in full headers
    class DeviceGraphOrchestrator;
    class ITPContext;
    class ILocalTPContext;
    class IRankOrchestrator;
    class PreparedWeightStore;
    struct GraphConfig;
    struct MoEExpertParallelPlan;

    using DomainLocalTPContextMap = std::map<std::string, std::shared_ptr<ILocalTPContext>>;

    // Note: FactoryPPStageConfig is now defined in FactoryPPStageConfig.h
    // to avoid circular dependencies with RankOrchestrator.h

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

        // Fixed scales for Q16_1 KV cache quantization (K and V separate)
        // K has large post-RoPE outliers; V values are much smaller.
        float kv_cache_scale_k = 256.0f; ///< K scale (FP32 range ±scale_k)
        float kv_cache_scale_v = 32.0f;  ///< V scale (FP32 range ±scale_v)

        // Explicit KV cache precision control.
        // AUTO preserves legacy behavior (derived from activation precision mode).
        KVCachePrecision kv_cache_precision = KVCachePrecision::AUTO;

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
        // Tensor Parallelism (Polymorphic — LOCAL or GLOBAL)
        // =====================================================================
        // When tp_ctx is set, the factory configures TP mode:
        //   - LOCAL TP (isLocal()=true): Weight sharding via device list + weights
        //   - GLOBAL TP (isLocal()=false): Weight sharding via MPI rank assignment
        //   - Collectives via ITPContext polymorphism (TPAllreduceStage)
        //
        // In a nested PP+TP topology, each PP stage runner gets its own tp_ctx.
        //
        // Lifetime: Caller owns the ITPContext and must keep it alive
        // for the duration of the IInferenceRunner's lifetime.
        // =====================================================================
        ITPContext *tp_ctx = nullptr;

        /// Device index within the TP context (0 to degree-1).
        /// Determines which portion of sharded weights this runner loads.
        /// For LOCAL TP: index into ILocalTPContext::devices().
        /// For GLOBAL TP: rank within the global TP domain.
        int tp_device_index = 0;

        /// PP stage config for nested TP-in-PP runners.
        /// When set, the factory uses partial weight loading (only embedding/lm_head
        /// as indicated by the PP stage config) and sets pp_layer_offset on the graph.
        std::optional<FactoryPPStageConfig> pp_stage_config;

        /// Optional MPI hostfile path for hostfile-aware node detection.
        /// When set, NodeDetection uses hostfile hostname ordering to assign node IDs
        /// instead of relying purely on first-appearance ordering from MPI_Allgather.
        std::string hostfile;

        /// Optional stage-local prepared store. When supplied, concrete factory
        /// paths install this store before materializing/preparing weights.
        std::shared_ptr<PreparedWeightStore> prepared_weight_store;

        /// Optional same-layer MoE expert overlay plan injected by orchestration
        /// and parity tests. When present, graph builders receive it through
        /// GraphConfig::moe.expert_parallel_plan.
        std::shared_ptr<MoEExpertParallelPlan> moe_expert_parallel_plan;

        /**
         * @brief Canonical factory: build InferenceRunnerConfig from a RankExecutionPlan
         *
         * Copies runtime fields (max_seq_len, activation_precision, etc.) from
         * plan.runtime which was pre-parsed in ExecutionPlanBuilder.
         *
         * @param plan The rank execution plan
         * @return Populated config
         */
        static InferenceRunnerConfig fromPlan(const RankExecutionPlan &plan)
        {
            InferenceRunnerConfig config;
            config.max_seq_len = plan.runtime.max_seq_len;
            config.batch_size = plan.runtime.batch_size;
            config.activation_precision = plan.runtime.activation_precision;
            config.kv_cache_precision = plan.runtime.kv_cache_precision;
            config.fused_attention_backend = plan.runtime.fused_attention_backend;
            config.kv_cache_scale_k = plan.runtime.kv_cache_scale_k;
            config.kv_cache_scale_v = plan.runtime.kv_cache_scale_v;
            return config;
        }
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
        std::shared_ptr<IMPIContext> mpi_ctx,
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
     * @brief Resolve a requested MoE expert overlay plan against loaded model metadata.
     *
     * If no plan is configured, returns nullptr. If the plan is enabled and has no
     * explicit placements, this runs the production MoEExpertParallelPlanner and
     * returns a planned copy. If explicit placements are present, validates them
     * against the model and returns the original shared plan.
     *
     * Throws std::invalid_argument/std::runtime_error with diagnostics on invalid
     * metadata, validation, or planning failures.
     */
    std::shared_ptr<MoEExpertParallelPlan> resolveMoEExpertParallelPlanForModel(
        IModelContext &model_ctx,
        const InferenceRunnerConfig &config);

    /**
     * @brief Populate production-owned domain-scoped LocalTP contexts for active
     * MoE expert overlay LocalTP TensorParallelExperts domains.
     *
     * The returned/updated owner map must stay alive for the graph builder and
     * executor lifetime because GraphConfig stores raw ILocalTPContext pointers.
     * Existing GraphConfig::domain_tp_contexts entries are preserved.
     */
    bool populateMoEExpertOverlayDomainTPContextsForGraph(
        GraphConfig &graph_config,
        DomainLocalTPContextMap &owned_contexts,
        const char *context);

    /**
     * @brief Factory function to create a unified LOCAL PP runner
     *
     * Creates a DeviceGraphOrchestrator configured for LOCAL Pipeline Parallelism
     * (multiple PP stages on multiple local devices within a single MPI rank).
     * The factory handles:
     * - Building GraphConfig from model metadata
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
        std::shared_ptr<ModelContext> model_ctx,
        DeviceId device,
        const FactoryPPStageConfig &pp_config,
        const InferenceRunnerConfig &config = {});

    /**
     * @brief Factory function to create RankOrchestrator for LOCAL TP
     *
     * Creates a RankOrchestrator that coordinates multiple devices within
     * a single MPI rank. Use this when LOCAL TP is configured via --tp-scope local.
     *
     * @param model_ctx Model context with weights
     * @param tp_ctx Pre-constructed LOCAL TP context (ownership transferred)
     * @param config Multi-device configuration
     * @return Unique pointer to IRankOrchestrator (extends IInferenceRunner)
     *
     * @code
     * // Create LOCAL TP context
     * auto tp_ctx = createLocalTPContext(
     *     {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
     *     {0.5f, 0.5f},
     *     CollectiveBackendType::NCCL);
     *
     * // Create multi-device config
     * RankOrchestrator::Config config;
     * config.devices = tp_ctx->devices();
     * config.weights = tp_ctx->weights();
     * config.backend = CollectiveBackendType::NCCL;
     *
     * // Create orchestrator (returns IInferenceRunner-compatible)
     * auto runner = createRankOrchestrator(model_ctx, std::move(tp_ctx), config);
     * @endcode
     */
    std::unique_ptr<IRankOrchestrator> createRankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const RankOrchestrator::Config &config);

    /**
     * @brief Factory function to create RankOrchestrator with testable dependencies
     *
     * For unit testing: allows injecting mock device runners and TP context.
     *
     * @param model_ctx Model context (can be MockModelContext)
     * @param device_runners Pre-constructed per-device runners (ownership transferred)
     * @param tp_ctx Pre-constructed LOCAL TP context (can be mock)
     * @param config Multi-device configuration
     * @return Unique pointer to IRankOrchestrator
     */
    std::unique_ptr<IRankOrchestrator> createTestableRankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const RankOrchestrator::Config &config);

} // namespace llaminar2
