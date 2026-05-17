/**
 * @file OrchestrationRunner.cpp
 * @brief Implementation of OrchestrationRunner
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "OrchestrationRunner.h"
#include "../../config/OrchestrationConfigParser.h"
#include "../../config/TPPPValidator.h"
#include "../mpi_orchestration/ExecutionPlanBuilder.h"
#include "../factory/InferenceRunnerFactory.h"
#include "../local_execution/orchestrators/RankOrchestrator.h"
#include "../parallelism_tree/ParallelismTree.h"
#include "../parallelism_tree/TreeToRunnerCompiler.h"
#include "../../collective/LocalTPContext.h"
#include "../../collective/ILocalPPContext.h"
#include "../../collective/BackendRouter.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/ModelContextConfig.h"
#include "../../loaders/ModelLoader.h"
#include "../../loaders/MmapRegion.h"
#include "../local_execution/graph/SchemaFactoryRegistry.h"
#include "../../backends/ComputeBackend.h"
#include "../../planning/ClusterInventoryGatherer.h"
#include "../../planning/ModelMemoryProfile.h"
#include "../../backends/DeviceAddressAdapter.h"
#include "../../tensors/TensorFactory.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/MPITopology.h"
#include "../../utils/NodeDetection.h"
#include "../../utils/NUMATopology.h"
#include "../../utils/WeightLoadingProfiler.h"
#include "../local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "../../execution/moe/MoERebalanceController.h"
#include "../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../execution/moe/ExpertWeightTransfer.h"
#include "../../execution/moe/MoEExpertParallelPlan.h"
#include "../../execution/moe/MoEExpertOverlayExecutionPlan.h"

#include <algorithm>
#include <cctype>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace llaminar2
{
    namespace
    {
        bool requiresOverlayMPIWorld(const std::shared_ptr<MoEExpertParallelPlan> &plan)
        {
            if (!plan || !plan->isTieredOverlay())
                return false;

            for (const auto &domain : plan->domains)
            {
                if (domain.owner_rank > 0)
                    return true;
                if (domain.world_ranks.size() > 1)
                    return true;
                if (domain.kind == ExpertDomainKind::NodeLocalTP && domain.participants.size() > 1)
                    return true;
            }
            return false;
        }

        std::shared_ptr<const MoEExpertOverlayExecutionPlan> resolveOverlayExecutionPlanForRunner(
            const std::shared_ptr<const MoEExpertParallelPlan> &plan,
            const std::shared_ptr<IMPIContext> &mpi_ctx)
        {
            if (!plan || !plan->isTieredOverlay() || !mpi_ctx)
                return nullptr;

            return std::make_shared<MoEExpertOverlayExecutionPlan>(
                resolveMoEExpertOverlayExecutionPlan(
                    plan,
                    MoEExpertOverlayExecutionPlanResolverOptions{
                        .current_world_rank = mpi_ctx->rank(),
                        .world_size = mpi_ctx->world_size(),
                    }));
        }

        const ExpertComputeDomain *overlayDomainForName(
            const MoEExpertParallelPlan &plan,
            const std::string &domain_name)
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                                   [&](const ExpertComputeDomain &domain)
                                   {
                                       return domain.name == domain_name;
                                   });
            return it == plan.domains.end() ? nullptr : &(*it);
        }

        bool applyOverlayRankRoleToExecutionPlan(
            RankExecutionPlan &rank_plan,
            const MoEExpertParallelPlan &overlay_plan,
            const MoEExpertOverlayExecutionPlan &execution_plan,
            std::string &error)
        {
            const auto &overlay_rank = execution_plan.currentRankPlan();
            if (!overlay_rank.builds_root_graph)
            {
                rank_plan.local_tp_devices.clear();
                rank_plan.local_tp_weights.clear();
                rank_plan.local_tp_backend = CollectiveBackendType::AUTO;
                rank_plan.local_pp_devices.clear();
                rank_plan.local_pp_layer_boundaries.clear();
                rank_plan.local_pp_stage_tp_info.clear();
                rank_plan.primary_device = GlobalDeviceAddress::cpu();
                rank_plan.weight_shard = {};
                return true;
            }

            const std::string base_domain_name = overlay_plan.effectiveBaseModelDomain();
            const auto *base_domain = overlayDomainForName(overlay_plan, base_domain_name);
            if (!base_domain)
            {
                error = "MoE overlay base/continuation domain '" + base_domain_name + "' is not defined";
                return false;
            }
            if (base_domain->participants.empty())
            {
                error = "MoE overlay base/continuation domain '" + base_domain_name + "' has no participants";
                return false;
            }
            if (base_domain->kind == ExpertDomainKind::NodeLocalTP)
            {
                error = "MoE overlay base/continuation domain '" + base_domain_name +
                        "' cannot be NodeLocalTP; NodeLocalTP is reserved for CPU fallback expert domains";
                return false;
            }

            rank_plan.local_pp_devices.clear();
            rank_plan.local_pp_layer_boundaries.clear();
            rank_plan.local_pp_stage_tp_info.clear();
            rank_plan.primary_device = base_domain->participants.front();
            rank_plan.local_tp_backend = base_domain->backend;
            rank_plan.local_tp_weights = base_domain->weights;
            rank_plan.weight_shard = {};

            if (base_domain->kind == ExpertDomainKind::LocalTP)
            {
                rank_plan.tp_scope = TPScope::LOCAL;
                rank_plan.local_tp_devices = base_domain->participants;
            }
            else
            {
                rank_plan.local_tp_devices.clear();
                rank_plan.local_tp_weights.clear();
                rank_plan.tp_scope = TPScope::AUTO;
            }

            return true;
        }

    }

    std::shared_ptr<MoEExpertParallelPlan> freezeMoEExpertOverlayPlanForModel(
        IModelContext &model_ctx,
        const std::shared_ptr<MoEExpertParallelPlan> &plan)
    {
        if (!plan)
            return nullptr;

        InferenceRunnerConfig runner_config;
        runner_config.moe_expert_parallel_plan = plan;
        return resolveMoEExpertParallelPlanForModel(model_ctx, runner_config);
    }

    // =========================================================================
    // Construction
    // =========================================================================

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder)
        : config_(std::move(config)), plan_builder_(std::move(plan_builder)), sampler_(0)
    {
        if (!plan_builder_)
        {
            plan_builder_ = createExecutionPlanBuilder();
        }
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true), sampler_(0)
    {
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan,
        std::unique_ptr<IInferenceRunner> runner)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true),
          runner_(std::move(runner)), initialized_(true), sampler_(0)
    {
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan,
        std::unique_ptr<IInferenceRunner> runner,
        std::shared_ptr<IMPIContext> mpi_ctx)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true),
          mpi_ctx_(std::move(mpi_ctx)), runner_(std::move(runner)),
          initialized_(true), sampler_(0)
    {
    }

    OrchestrationRunner::~OrchestrationRunner()
    {
        shutdown();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool OrchestrationRunner::initialize()
    {
        if (initialized_)
        {
            return true;
        }

        auto syncInitStep = [&](bool local_ok, const char *step_name) -> bool
        {
            if (!mpi_ctx_ || mpi_ctx_->world_size() <= 1)
            {
                return local_ok;
            }

            int ok = local_ok ? 1 : 0;
            int global_ok = 0;
            MPI_Allreduce(&ok, &global_ok, 1, MPI_INT, MPI_MIN, mpi_ctx_->communicator());
            if (global_ok == 0)
            {
                if (local_ok)
                {
                    setError(std::string("Initialization failed on another rank at step: ") + step_name);
                }
                return false;
            }
            return true;
        };

        try
        {
            // Step 1: Initialize MPI if needed
            if (!initializeMPI())
            {
                return false;
            }
            if (!syncInitStep(true, "initializeMPI"))
            {
                return false;
            }

            // Step 2: Build execution plan (if not pre-built)
            if (!buildExecutionPlan())
            {
                syncInitStep(false, "buildExecutionPlan");
                return false;
            }
            if (!syncInitStep(true, "buildExecutionPlan"))
            {
                return false;
            }

            // Step 3: Setup LOCAL TP context
            if (!setupLocalTPContext())
            {
                syncInitStep(false, "setupLocalTPContext");
                return false;
            }
            if (!syncInitStep(true, "setupLocalTPContext"))
            {
                return false;
            }

            // Step 3.5: Setup LOCAL PP context
            if (!setupLocalPPContext())
            {
                syncInitStep(false, "setupLocalPPContext");
                return false;
            }
            if (!syncInitStep(true, "setupLocalPPContext"))
            {
                return false;
            }

            // Step 4: Load model weights
            if (!loadWeights())
            {
                syncInitStep(false, "loadWeights");
                return false;
            }
            if (!syncInitStep(true, "loadWeights"))
            {
                return false;
            }

            // Step 4b: Freeze model-aware MoE overlay placements before any
            // endpoint or root graph is constructed. The raw YAML plan may
            // intentionally omit placements and rely on the planner.
            if (!freezeMoEExpertOverlayPlanForLoadedModel())
            {
                syncInitStep(false, "freezeMoEExpertOverlayPlanForLoadedModel");
                return false;
            }
            if (!syncInitStep(true, "freezeMoEExpertOverlayPlanForLoadedModel"))
            {
                return false;
            }

            // Step 5: Validate TP/PP configuration against model architecture
            if (!validateTPPPConfiguration())
            {
                syncInitStep(false, "validateTPPPConfiguration");
                return false;
            }
            if (!syncInitStep(true, "validateTPPPConfiguration"))
            {
                return false;
            }

            // Step 5b: Validate context length against model max
            if (!validateContextLength())
            {
                syncInitStep(false, "validateContextLength");
                return false;
            }
            if (!syncInitStep(true, "validateContextLength"))
            {
                return false;
            }

            // Step 5c: Validate memory plan
            if (!validateMemoryPlan())
            {
                syncInitStep(false, "validateMemoryPlan");
                return false;
            }
            if (!syncInitStep(true, "validateMemoryPlan"))
            {
                return false;
            }

            // Step 6: Build compute graph
            if (!buildComputeGraph())
            {
                syncInitStep(false, "buildComputeGraph");
                return false;
            }
            if (!syncInitStep(true, "buildComputeGraph"))
            {
                return false;
            }

            initialized_ = true;

            // Cache model-recommended sampling params (for API consumers)
            if (model_ctx_)
            {
                const std::string arch = model_ctx_->architecture();
                if (SchemaFactoryRegistry::isSupported(arch))
                {
                    auto factory = SchemaFactoryRegistry::getFactory(arch);
                    if (factory)
                    {
                        recommended_sampling_params_ = factory->getRecommendedSamplingParams();
                        stop_thinking_prompt_ = factory->getStopThinkingPrompt();
                        tool_call_format_ = factory->getToolCallFormat();
                        if (recommended_sampling_params_.has_penalties() || recommended_sampling_params_.temperature != 1.0f)
                        {
                            LOG_INFO("[OrchestrationRunner] Model-recommended sampling: "
                                     << "temp=" << recommended_sampling_params_.temperature
                                     << " top_p=" << recommended_sampling_params_.top_p
                                     << " top_k=" << recommended_sampling_params_.top_k
                                     << " presence_penalty=" << recommended_sampling_params_.presence_penalty
                                     << " frequency_penalty=" << recommended_sampling_params_.frequency_penalty);
                        }
                        if (!stop_thinking_prompt_.empty())
                        {
                            LOG_INFO("[OrchestrationRunner] Stop-thinking prompt configured ("
                                     << stop_thinking_prompt_.size() << " chars)");
                        }
                    }
                }
            }

            LOG_INFO("OrchestrationRunner initialized successfully");
            return true;
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Initialization failed: ") + e.what());
        }
    }

    void OrchestrationRunner::shutdown()
    {
        if (!initialized_)
        {
            return;
        }

        // Release resources in reverse order
        runner_.reset();
        local_pp_ctx_.reset();
        local_tp_ctx_.reset();
        model_ctx_.reset();

        initialized_ = false;
        LOG_DEBUG("OrchestrationRunner shut down");
    }

    // =========================================================================
    // Inference
    // =========================================================================

    bool OrchestrationRunner::prefill(const std::vector<int32_t> &prompt_tokens)
    {
        if (!initialized_)
        {
            setError("Runner not initialized");
            return false;
        }

        if (prompt_tokens.empty())
        {
            setError("Empty prompt tokens");
            return false;
        }

        // Broadcast to worker ranks so they prefill with the same tokens
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::PREFILL);
            int32_t n_tokens = static_cast<int32_t>(prompt_tokens.size());
            mpi_ctx_->broadcast_int32(&n_tokens, 1, 0);
            // const_cast is safe: rank 0 is the sender, buffer is not modified
            mpi_ctx_->broadcast_int32(const_cast<int32_t *>(prompt_tokens.data()),
                                      static_cast<size_t>(n_tokens), 0);
        }

        // Run forward pass
        try
        {
            if (!runner_->forward(prompt_tokens.data(),
                                  static_cast<int>(prompt_tokens.size())))
            {
                return setError("Forward pass failed during prefill");
            }
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Prefill failed: ") + e.what());
        }

        // Signal that prefill logits are ready for sampling.
        // The first decodeStep() will sample from these logits directly
        // instead of re-feeding the last prompt token (which would cause
        // the model to see it twice at consecutive positions, corrupting
        // GDN recurrence state and KV cache entries).
        prefill_logits_ready_ = true;

        return true;
    }

    GenerationResult OrchestrationRunner::decodeStep()
    {
        GenerationResult result;

        if (!initialized_)
        {
            result.error = "Runner not initialized";
            return result;
        }

        // Broadcast to worker ranks so they run decode in lockstep
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
            broadcastCommand(MPICommand::DECODE_STEP);

        if (prefill_logits_ready_)
        {
            // First decode step after prefill: sample from the already-computed
            // prefill logits instead of re-feeding the last prompt token.
            // This avoids processing the last token twice (which corrupts GDN
            // recurrence state and creates duplicate KV cache entries).
            prefill_logits_ready_ = false;
        }
        else
        {
            // Run single-token forward with last token
            if (!runner_->forward(&last_token_, 1))
            {
                result.error = "Forward pass failed during decode";
                return result;
            }
        }

        // Tail stage: try GPU-side sampling first, fall back to CPU
        // When penalties are active, compute the sparse penalty map on CPU,
        // upload to GPU, apply in-place, then sample on GPU.
        // This avoids the full ~600KB D2H transfer of logits.
        int token = -1;

        if (active_sampling_params_.has_penalties())
        {
            // Compute sparse penalty map on CPU (presence + frequency + DRY)
            int vocab = vocabSize();
            auto penalty_map = sampler_.compute_penalty_map(active_sampling_params_, vocab);

            if (!penalty_map.empty())
            {
                // Try GPU-side penalty application + sampling
                bool gpu_penalties_applied = runner_->applyPenaltiesOnDevice(penalty_map, vocab);
                if (gpu_penalties_applied)
                {
                    // Penalties applied on GPU — now sample from penalized logits
                    if (active_sampling_params_.is_greedy())
                    {
                        token = runner_->sampleGreedyOnDevice();
                    }
                    else
                    {
                        token = runner_->sampleOnDevice(active_sampling_params_);
                    }
                }
            }
            else
            {
                // No penalties to apply — sample directly on GPU
                if (active_sampling_params_.is_greedy())
                {
                    token = runner_->sampleGreedyOnDevice();
                }
                else
                {
                    token = runner_->sampleOnDevice(active_sampling_params_);
                }
            }
            // If GPU path failed, fall through to CPU fallback below
        }
        else if (active_sampling_params_.is_greedy())
        {
            // Try GPU-side greedy (argmax)
            token = runner_->sampleGreedyOnDevice();
        }
        else
        {
            // Try GPU-side top-k/top-p
            token = runner_->sampleOnDevice(active_sampling_params_);
            if (token >= 0)
            {
                LOG_TRACE("[decodeStep] GPU top-k/top-p sampled token=" << token);
            }
        }

        if (token < 0)
        {
            // Fallback: CPU-side sampling (requires logits D2H)
            LOG_TRACE("[decodeStep] GPU sampling returned -1, falling back to CPU");
            const float *logits = runner_->logits();
            if (!logits)
            {
                result.error = "No logits available";
                return result;
            }
            int vocab = vocabSize();
            token = sampler_.sample(logits, static_cast<size_t>(vocab), active_sampling_params_);
        }

        // Record token for presence/frequency penalty tracking
        sampler_.record_token(token);

        result.tokens.push_back(token);
        last_token_ = token; // Store for next decode step

        // Check stop tokens
        for (int32_t stop : stop_tokens_)
        {
            if (token == stop)
            {
                result.is_complete = true;
                break;
            }
        }

        return result;
    }

    GenerationResult OrchestrationRunner::generate(
        const std::vector<int32_t> &prompt_tokens,
        int max_new_tokens,
        const SamplingParams &sampling)
    {
        GenerationResult result;

        if (!initialized_)
        {
            result.error = "Runner not initialized";
            return result;
        }

        // Prefill
        // After prefill, the first decodeStep() samples from the prefill logits
        // directly (via prefill_logits_ready_ flag) instead of re-feeding the
        // last prompt token. GPU-side sampling (sampleGreedyOnDevice) works on
        // device logits without D2H, so no gather is needed for GPU models.
        // For CPU models, logits are already on host.
        if (!prefill(prompt_tokens))
        {
            result.error = last_error_;
            return result;
        }

        // Store sampling params for decodeStep() and configure GPU-side decode
        active_sampling_params_ = sampling;
        sampler_ = Sampler(sampling.seed);

        // Enable GPU-side logits skip for decode (GPU sampling avoids full D2H)
        runner_->setSkipLogitsGatherDecode(true);

        for (int i = 0; i < max_new_tokens; ++i)
        {
            // Use decodeStep() which uses last_token_ internally
            GenerationResult step = decodeStep();

            if (!step.error.empty())
            {
                result.error = step.error;
                break;
            }

            // Collect tokens from step (on tail stage)
            for (int32_t token : step.tokens)
            {
                result.tokens.push_back(token);
            }

            if (step.is_complete)
            {
                result.is_complete = true;
                break;
            }

            if (!maybeApplyMoERebalance())
            {
                result.error = last_error_.empty() ? "MoE rebalance failed" : last_error_;
                break;
            }
        }

        // Restore normal logits gathering after generation
        runner_->setSkipLogitsGatherDecode(false);

        return result;
    }

    bool OrchestrationRunner::maybeApplyMoERebalance()
    {
        auto *controller = moeRebalanceController();
        if (!controller || !controller->shouldRebalance())
            return true;

        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::APPLY_MOE_REBALANCE);
        }

        if (!applyMoERebalanceWithReplicas())
        {
            setError("MoE rebalance failed");
            return false;
        }
        return true;
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    const RankExecutionPlan &OrchestrationRunner::executionPlan() const
    {
        return plan_;
    }

    const OrchestrationConfig &OrchestrationRunner::config() const
    {
        return config_;
    }

    // =========================================================================
    // Status
    // =========================================================================

    bool OrchestrationRunner::isInitialized() const
    {
        return initialized_;
    }

    const std::string &OrchestrationRunner::lastError() const
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return last_error_;
    }

    int OrchestrationRunner::vocabSize() const
    {
        if (!runner_)
        {
            return 0;
        }
        return runner_->vocab_size();
    }

    int OrchestrationRunner::currentPosition() const
    {
        if (!runner_)
        {
            return 0;
        }
        return runner_->get_position();
    }

    void OrchestrationRunner::clearCache()
    {
        // Broadcast to worker ranks so they clear their KV caches in lockstep
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
            broadcastCommand(MPICommand::CLEAR_CACHE);

        if (runner_)
        {
            runner_->clear_cache();
        }
#if defined(__GLIBC__)
        ::malloc_trim(0);
#endif
        prefill_logits_ready_ = false;
    }

    // =========================================================================
    // Advanced
    // =========================================================================

    const float *OrchestrationRunner::lastLogits() const
    {
        if (!runner_)
        {
            return nullptr;
        }
        return runner_->logits();
    }

    void OrchestrationRunner::setStopTokens(const std::vector<int32_t> &stop_tokens)
    {
        stop_tokens_ = stop_tokens;
    }

    // =========================================================================
    // Initialization Helpers
    // =========================================================================

    bool OrchestrationRunner::initializeMPI()
    {
        // Check if multi-rank MPI is needed
        bool needs_multi_rank_mpi = config_.pp_degree > 1 ||
                                    config_.tp_scope == TPScope::GLOBAL ||
                                    config_.tp_scope == TPScope::HYBRID ||
                                    requiresOverlayMPIWorld(config_.moe_expert_parallel_plan);

        if (!needs_multi_rank_mpi)
        {
            // For single-rank execution, create a local-only MPI context
            // This is needed for TensorFactory creation in ModelContext
            LOG_DEBUG("Creating single-rank MPI context for local execution");
            mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            return true;
        }

        // Create or reuse MPI context using factory
        mpi_ctx_ = MPIContextFactory::global();
        if (!mpi_ctx_)
        {
            return setError("Failed to get MPI context");
        }

        LOG_INFO("MPI initialized: rank " << mpi_ctx_->rank()
                                          << " of " << mpi_ctx_->world_size());

        return true;
    }

    bool OrchestrationRunner::buildExecutionPlan()
    {
        if (plan_built_)
        {
            LOG_DEBUG("Using pre-built execution plan");
            return true;
        }

        if (!plan_builder_)
        {
            return setError("No plan builder available");
        }

        // Gather cluster inventory
        cluster_inventory_ = gatherClusterInventory();

        // Get model config (need to load model metadata first)
        // Read actual model metadata from the GGUF file for accurate plan building
        ModelConfig model_config;
        if (!config_.model_path.empty())
        {
            std::shared_ptr<IMPIContext> metadata_mpi_ctx = mpi_ctx_;
            if (!metadata_mpi_ctx)
            {
                metadata_mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            }
            TensorFactory metadata_factory(*metadata_mpi_ctx);
            ModelLoader metadata_loader(&metadata_factory);
            metadata_loader.setUseMmap(false); // Only reading header metadata, skip mmap
            bool metadata_ok = false;
            try
            {
                metadata_ok = metadata_loader.loadModel(config_.model_path);
            }
            catch (const std::exception &e)
            {
                LOG_WARN("Failed to read model metadata from " << config_.model_path
                                                               << " (" << e.what() << "), using defaults for plan building");
                metadata_ok = false;
            }
            if (metadata_ok)
            {
                model_config.n_layers = static_cast<int>(metadata_loader.blockCount());
                model_config.n_heads = static_cast<int>(metadata_loader.headCount());
                model_config.n_kv_heads = static_cast<int>(metadata_loader.headCountKV());
                model_config.hidden_size = static_cast<int>(metadata_loader.embeddingLength());
                LOG_DEBUG("Model metadata for plan building: n_layers=" << model_config.n_layers
                                                                        << " n_heads=" << model_config.n_heads
                                                                        << " n_kv_heads=" << model_config.n_kv_heads
                                                                        << " hidden_size=" << model_config.hidden_size);
            }
            else
            {
                LOG_WARN("Failed to read model metadata from " << config_.model_path
                                                               << ", using defaults for plan building");
                model_config.n_layers = 24;
                model_config.n_heads = 32;
                model_config.n_kv_heads = 8;
                model_config.hidden_size = 4096;
            }
        }
        else
        {
            // No model path (testing) - use defaults
            model_config.n_layers = 24;
            model_config.n_heads = 32;
            model_config.n_kv_heads = 8;
            model_config.hidden_size = 4096;
        }

        // Validate config
        auto errors = plan_builder_->validateConfig(config_, model_config, cluster_inventory_);
        if (!errors.empty())
        {
            std::string error_msg = "Config validation failed:";
            for (const auto &e : errors)
            {
                error_msg += "\n  - " + e;
            }
            return setError(error_msg);
        }

        // Build plan for this rank
        int my_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        plan_ = plan_builder_->buildPlanForRank(config_, model_config, cluster_inventory_, my_rank);

        auto overlay_execution_plan = resolveOverlayExecutionPlanForRunner(
            config_.moe_expert_parallel_plan,
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_);
        if (overlay_execution_plan)
        {
            std::string overlay_plan_error;
            if (!applyOverlayRankRoleToExecutionPlan(
                    plan_,
                    *config_.moe_expert_parallel_plan,
                    *overlay_execution_plan,
                    overlay_plan_error))
            {
                return setError(overlay_plan_error);
            }

            if (overlay_execution_plan->buildsRootGraph())
            {
                LOG_INFO("[OrchestrationRunner] MoE overlay root plan bound to base domain '"
                         << config_.moe_expert_parallel_plan->effectiveBaseModelDomain()
                         << "' devices=" << plan_.local_tp_devices.size());
            }
            else
            {
                LOG_INFO("[OrchestrationRunner] MoE overlay non-root plan narrowed to participant endpoint role "
                         << toString(overlay_execution_plan->currentRankPlan().role));
            }
        }

        // Validate the built plan
        auto plan_errors = plan_.validate();
        if (!plan_errors.empty())
        {
            std::string error_msg = "Plan validation failed:";
            for (const auto &e : plan_errors)
            {
                error_msg += "\n  - " + e;
            }
            return setError(error_msg);
        }

        plan_built_ = true;
        return true;
    }

    ClusterInventory OrchestrationRunner::gatherClusterInventory()
    {
        return llaminar2::gatherClusterInventory(mpi_ctx_, config_.tp_devices, config_.hostfile);
    }

    bool OrchestrationRunner::setupLocalTPContext()
    {
        // Check if LOCAL TP is configured
        if (plan_.local_tp_devices.empty())
        {
            LOG_DEBUG("No LOCAL TP devices configured");
            return true;
        }

        // When LOCAL PP with TP domains is active, TP is per-stage (each PP stage
        // creates its own TP context inside the nested MDO). Skip global TP context.
        if (plan_.usesLocalPP())
        {
            LOG_DEBUG("LOCAL PP active — TP will be handled per-stage by nested MDOs");
            return true;
        }

        // Create LOCAL TP context using factory function
        local_tp_ctx_ = createLocalTPContext(
            plan_.local_tp_devices,
            plan_.local_tp_weights,
            plan_.local_tp_backend);
        if (!local_tp_ctx_)
        {
            return setError("Failed to create LOCAL TP context");
        }

        LOG_INFO("LOCAL TP context created with " << plan_.local_tp_devices.size() << " devices");
        return true;
    }

    bool OrchestrationRunner::setupLocalPPContext()
    {
        // Check if LOCAL PP is configured
        if (plan_.local_pp_devices.size() <= 1)
        {
            LOG_DEBUG("No LOCAL PP devices configured (or only single device)");
            return true;
        }

        // Build LocalPPConfig from execution plan
        LocalPPConfig pp_config;
        pp_config.stage_devices = plan_.local_pp_devices;
        pp_config.layer_boundaries = plan_.local_pp_layer_boundaries;

        // Validate configuration
        if (!pp_config.isValid())
        {
            return setError("Invalid LOCAL PP configuration");
        }

        // Create LOCAL PP context using factory function
        local_pp_ctx_ = createLocalPPContext(pp_config);
        if (!local_pp_ctx_)
        {
            return setError("Failed to create LOCAL PP context");
        }

        LOG_INFO("LOCAL PP context created with " << pp_config.numStages()
                                                  << " stages on " << plan_.local_pp_devices.size() << " devices");
        return true;
    }

    bool OrchestrationRunner::loadWeights()
    {
        // Get model path from config
        std::string model_path = config_.model_path;

        // Skip weight loading if no model path (for testing)
        if (model_path.empty())
        {
            LOG_DEBUG("No model path specified, skipping weight loading");
            return true;
        }

        // Create ModelContextConfig from the execution plan
        // This automatically configures:
        // - Layer range (first_layer, last_layer) for PP
        // - Global weight flags (has_embedding, has_lm_head) for PP
        // - Shard info (shard_index, total_shards, work_fraction) for TP
        // - Appropriate strategy (REPLICATED, SHARDED, or layer-partitioned)
        ModelContextConfig weight_config = ModelContextConfig::fromExecutionPlan(plan_);
        weight_config.mpi_ctx = mpi_ctx_;
        weight_config.weight_precision = WeightPrecision::NATIVE;
        weight_config.use_mmap = config_.use_mmap;

        // Multi-rank page cache pre-population:
        // In multi-rank mode, each rank independently mmaps and first-touches the
        // same file. Without coordination, this creates N concurrent page fault
        // streams that destroy disk readahead and cause ~60 MB/s throughput
        // instead of ~1 GB/s. Fix: node leaders read the file sequentially to warm
        // the page cache, then all ranks on that node skip POSIX_FADV_DONTNEED so
        // the OMP first-touch loop faults from cache (memory speed) instead of disk.
        //
        // Multi-node scaling: each physical machine has its own page cache, so we
        // use the node-leader (lowest local rank) on each node to prepopulate
        // independently. An intra-node barrier ensures same-node ranks wait only
        // for their own leader, not a global rank-0.
        const bool is_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;
        if (is_multi_rank && config_.use_mmap)
        {
            const auto *topo = mpi_ctx_->topology();
            if (topo)
            {
                // Per-node prepopulation: each node leader warms its own page cache
                if (topo->is_node_leader())
                {
                    LOG_INFO("Node leader (rank " << mpi_ctx_->rank()
                                                  << ", node " << topo->placement().node_id
                                                  << ") pre-populating page cache for multi-rank mmap...");
                    MmapRegion::prepopulatePageCache(model_path);
                }
                // Intra-node barrier: same-node ranks wait for their node leader only.
                // Ranks on other nodes proceed independently with their own leader.
                MPI_Comm intra = mpi_ctx_->intra_node_comm();
                if (intra != MPI_COMM_NULL)
                    MPI_Barrier(intra);
                else
                    MPI_Barrier(mpi_ctx_->communicator());
            }
            else
            {
                // Fallback: no topology available (mock or non-standard context)
                if (mpi_ctx_->rank() == 0)
                {
                    LOG_INFO("Pre-populating page cache for multi-rank mmap (rank 0 fallback)...");
                    MmapRegion::prepopulatePageCache(model_path);
                }
                MPI_Barrier(mpi_ctx_->communicator());
            }
            weight_config.skip_mmap_cache_eviction = true;
        }

        // Validate config
        auto errors = weight_config.validate();
        if (!errors.empty())
        {
            std::ostringstream oss;
            oss << "Invalid ModelContextConfig from execution plan:\n";
            for (const auto &err : errors)
            {
                oss << "  - " << err << "\n";
            }
            return setError(oss.str());
        }

        LOG_DEBUG("Weight loading config: " << weight_config.toString());

        // Create ModelContext using the unified config-based factory method
        // Use NATIVE weight precision to preserve quantization (Q4_0, Q8_0, etc.)
        // for efficient GPU kernels rather than dequantizing to FP32
        {
            ScopedWeightLoadTimer timer(WeightLoadPhase::GGUF_PARSE);
            model_ctx_ = ModelContext::create(model_path, weight_config);
        }

        if (!model_ctx_)
        {
            return setError("Failed to create ModelContext for: " + model_path);
        }

        // Create tokenizer from model context
        tokenizer_ = createTokenizer(model_ctx_);
        if (!tokenizer_)
        {
            LOG_WARN("Failed to create tokenizer from model context");
        }

        LOG_INFO("Model context created from: " << model_path
                                                << " (layers " << weight_config.first_layer << "-" << weight_config.last_layer
                                                << ", embedding=" << weight_config.has_embedding
                                                << ", lm_head=" << weight_config.has_lm_head << ")");

        return true;
    }

    bool OrchestrationRunner::freezeMoEExpertOverlayPlanForLoadedModel()
    {
        if (!model_ctx_ || !config_.moe_expert_parallel_plan)
            return true;

        const bool requested_without_placements =
            config_.moe_expert_parallel_plan->isTieredOverlay() &&
            config_.moe_expert_parallel_plan->placements.empty();

        try
        {
            auto frozen_plan = freezeMoEExpertOverlayPlanForModel(
                *model_ctx_,
                config_.moe_expert_parallel_plan);
            if (!frozen_plan)
                return true;

            config_.moe_expert_parallel_plan = std::move(frozen_plan);
            if (config_.moe_expert_parallel_plan->isTieredOverlay())
            {
                LOG_INFO("[OrchestrationRunner] MoE expert overlay plan frozen: placements="
                         << config_.moe_expert_parallel_plan->placements.size()
                         << " routed_tiers=" << config_.moe_expert_parallel_plan->routed_tiers.size()
                         << " domains=" << config_.moe_expert_parallel_plan->domains.size()
                         << (requested_without_placements ? " (planned from model metadata)" : ""));
            }
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Failed to freeze MoE expert overlay plan from model metadata: ") + e.what());
        }

        return true;
    }

    bool OrchestrationRunner::validateTPPPConfiguration()
    {
        // Skip validation if no model loaded (testing mode)
        if (!model_ctx_)
        {
            LOG_DEBUG("No model context, skipping TP/PP validation");
            return true;
        }

        // Run validation
        auto result = TPPPValidator::validate(config_, *model_ctx_);

        // Log warnings (but don't fail)
        for (const auto &warning : result.warnings)
        {
            LOG_WARN("[TP/PP Config] " << warning);
        }

        // Check for errors
        if (!result.valid)
        {
            std::ostringstream oss;
            oss << "TP/PP configuration is incompatible with model architecture:\n";
            for (const auto &error : result.errors)
            {
                oss << "  - " << error << "\n";
            }
            return setError(oss.str());
        }

        LOG_INFO("TP/PP configuration validated against model architecture");
        return true;
    }

    bool OrchestrationRunner::validateContextLength()
    {
        if (!model_ctx_)
            return true;

        const int model_max = model_ctx_->contextLength();
        if (model_max <= 0)
        {
            LOG_DEBUG("Model does not report max context length, skipping validation");
            return true;
        }

        if (config_.max_seq_len > model_max)
        {
            LOG_ERROR("Requested context length " << config_.max_seq_len
                                                  << " exceeds model maximum of " << model_max
                                                  << ". Use -c " << model_max << " or smaller.");
            return setError("Context length " + std::to_string(config_.max_seq_len) +
                            " exceeds model maximum of " + std::to_string(model_max));
        }

        LOG_INFO("Context length: " << config_.max_seq_len
                                    << " / " << model_max << " (model max)");
        return true;
    }

    bool OrchestrationRunner::validateMemoryPlan()
    {
        if (!model_ctx_)
        {
            LOG_WARN("[MemoryPlanner] No model context — skipping memory validation");
            return true;
        }

        // Build memory profile from the loaded model
        auto profile = ModelMemoryProfile::fromGGUF(model_ctx_->model());

        auto memoryForDevice = [&](DeviceId device)
        {
            std::pair<size_t, size_t> memory{0, 0};
            int my_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            if (my_rank < static_cast<int>(cluster_inventory_.ranks.size()))
            {
                const auto &rank_inv = cluster_inventory_.ranks[my_rank];
                if (device.is_gpu())
                {
                    for (const auto &gpu : rank_inv.gpus)
                    {
                        if (gpu.type == device.type && gpu.local_device_id == device.ordinal)
                        {
                            memory.first = gpu.memory_bytes;
                            memory.second = gpu.free_memory_bytes;
                            break;
                        }
                    }
                }
                else
                {
                    memory.first = rank_inv.cpu_memory_bytes;
                    memory.second = rank_inv.cpu_memory_bytes;
                }
            }
            return memory;
        };

        std::vector<DevicePlanConfig> device_configs;
        auto makeConfigForDevice = [&](DeviceId device, int shard_index, int total_shards)
        {
            auto [device_total, device_free] = memoryForDevice(device);

            DevicePlanConfig cfg;
            cfg.device = device;
            cfg.device_total_bytes = device_total;
            cfg.device_free_bytes = device_free;
            cfg.shard_index = shard_index;
            cfg.total_shards = total_shards;
            cfg.first_layer = plan_.first_layer;
            cfg.last_layer = plan_.last_layer;
            cfg.batch_size = plan_.runtime.batch_size;
            cfg.max_seq_len = plan_.runtime.max_seq_len;

            switch (plan_.runtime.kv_cache_precision)
            {
            case KVCachePrecision::FP16:
                cfg.kv_precision = "fp16";
                break;
            case KVCachePrecision::FP32:
                cfg.kv_precision = "fp32";
                break;
            case KVCachePrecision::Q8_1:
                cfg.kv_precision = "q8_1";
                break;
            default:
                cfg.kv_precision = "fp16";
                break;
            }

            if (total_shards > 1 && profile.n_kv_heads > 0)
            {
                cfg.local_kv_heads = profile.n_kv_heads / total_shards;
                if (cfg.local_kv_heads < 1)
                    cfg.local_kv_heads = 1;
            }
            return cfg;
        };

        if (plan_.usesLocalTP())
        {
            const int total_shards = static_cast<int>(plan_.local_tp_devices.size());
            device_configs.reserve(plan_.local_tp_devices.size());
            for (int index = 0; index < total_shards; ++index)
            {
                device_configs.push_back(makeConfigForDevice(
                    plan_.local_tp_devices[static_cast<size_t>(index)].toLocalDeviceId(),
                    index,
                    total_shards));
            }
        }
        else
        {
            DeviceId device = DeviceAddressAdapter::toDeviceId(plan_.primary_device);
            device_configs.push_back(makeConfigForDevice(
                device,
                plan_.weight_shard.shard_index,
                plan_.weight_shard.total_shards));
        }

        auto plan = MemoryPlanner::plan(profile, device_configs);

        if (!plan.fits())
        {
            std::string msg = "Memory plan validation failed — model does not fit on assigned device(s):\n";
            msg += plan.renderTable();
            for (const auto &d : plan.diagnostics)
            {
                msg += "\n  " + d;
            }
            return setError(msg);
        }

        LOG_INFO("[MemoryPlanner] Memory validation passed:\n"
                 << plan.renderTable());
        return true;
    }

    bool OrchestrationRunner::buildComputeGraph()
    {
        ScopedWeightLoadTimer timer(WeightLoadPhase::GRAPH_BUILD);

        auto overlay_execution_plan = resolveOverlayExecutionPlanForRunner(
            config_.moe_expert_parallel_plan,
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_);
        if (overlay_execution_plan && !overlay_execution_plan->buildsRootGraph())
        {
            const auto &rank_plan = overlay_execution_plan->currentRankPlan();
            return setError(std::string("MoE overlay rank ") + std::to_string(rank_plan.world_rank) +
                            " has role " + toString(rank_plan.role) +
                            " but sidecar endpoint ranks were removed by graph-native MoE productionization");
        }

        // Check if LOCAL PP is configured (takes priority over TP, because
        // TP-in-PP composition creates per-stage TP contexts inside the MDO)
        if (plan_.usesLocalPP())
        {
            return buildLocalPPComputeGraph();
        }

        // Check if LOCAL TP is configured (multiple devices within this rank)
        if (hasLocalTP())
        {
            return buildMultiDeviceComputeGraph();
        }

        // Single-device path
        return buildSingleDeviceComputeGraph();
    }

    bool OrchestrationRunner::hasLocalTP() const
    {
        return plan_.local_tp_devices.size() > 1;
    }

    std::shared_ptr<ITokenizer> OrchestrationRunner::tokenizer() const
    {
        return tokenizer_;
    }

    const std::string &OrchestrationRunner::architecture() const
    {
        static const std::string kEmpty;
        return model_ctx_ ? model_ctx_->architecture() : kEmpty;
    }

    bool OrchestrationRunner::buildMultiDeviceComputeGraph()
    {
        // Validate that all requested devices actually exist in hardware
        const auto &dm = DeviceManager::instance();
        for (size_t i = 0; i < plan_.local_tp_devices.size(); ++i)
        {
            auto local_device = plan_.local_tp_devices[i].toLocalDeviceId();
            if (!dm.deviceExists(local_device))
            {
                return setError("TP device " + std::to_string(i) + " (" +
                                local_device.toString() +
                                ") is not available. Available devices: " +
                                dm.availableDevicesString());
            }
        }

        LOG_INFO("[OrchestrationRunner] Execution strategy: MULTI-DEVICE (LOCAL TP)");
        LOG_INFO("[OrchestrationRunner]   TP degree: " << plan_.local_tp_devices.size());

        // Log each device
        for (size_t i = 0; i < plan_.local_tp_devices.size(); ++i)
        {
            const auto &dev = plan_.local_tp_devices[i];
            std::string weight_str = "";
            if (i < plan_.local_tp_weights.size())
            {
                weight_str = " (weight=" + std::to_string(plan_.local_tp_weights[i]) + ")";
            }
            LOG_INFO("[OrchestrationRunner]   Device " << i << ": " << dev.toString() << weight_str);
        }

        // Build config from execution plan via canonical factory
        auto mdo_config = RankOrchestrator::Config::fromPlan(plan_);
        mdo_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
        mdo_config.moe_expert_overlay_mpi_ctx = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;

        LOG_INFO("[OrchestrationRunner] Multi-device precision config: activation="
                 << activationPrecisionToString(mdo_config.activation_precision)
                 << ", kv_cache=" << kvCachePrecisionToString(mdo_config.kv_cache_precision));

        // Validate config
        if (!mdo_config.validate())
        {
            return setError("Invalid multi-device configuration");
        }

        // Create RankOrchestrator via factory
        // Note: local_tp_ctx_ was already created in setupLocalTPContext()
        auto multi_orchestrator = createRankOrchestrator(
            model_ctx_,
            std::move(local_tp_ctx_),
            mdo_config);

        if (!multi_orchestrator)
        {
            return setError("Failed to create RankOrchestrator");
        }

        // Store as IInferenceRunner (RankOrchestrator extends it)
        runner_ = std::move(multi_orchestrator);

        LOG_INFO("Multi-device compute graph built successfully");
        return true;
    }

    bool OrchestrationRunner::buildLocalPPComputeGraph()
    {
        const auto &pp_devices = plan_.local_pp_devices;
        const auto &boundaries = plan_.local_pp_layer_boundaries;

        if (pp_devices.size() < 2 || boundaries.size() < pp_devices.size() + 1)
        {
            return setError("Invalid LOCAL PP plan: need >=2 devices and matching layer boundaries");
        }

        LOG_INFO("[OrchestrationRunner] Execution strategy: LOCAL PIPELINE PARALLEL");
        LOG_INFO("[OrchestrationRunner]   PP stages: " << pp_devices.size());

        // Validate all devices exist
        const auto &dm = DeviceManager::instance();
        for (size_t i = 0; i < pp_devices.size(); ++i)
        {
            auto local_device = pp_devices[i].toLocalDeviceId();
            if (!dm.deviceExists(local_device))
            {
                return setError("PP device " + std::to_string(i) + " (" +
                                local_device.toString() +
                                ") is not available. Available devices: " +
                                dm.availableDevicesString());
            }
        }

        // Build config from execution plan via canonical factory
        auto mdo_config = RankOrchestrator::Config::fromPlan(plan_);
        mdo_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
        mdo_config.moe_expert_overlay_mpi_ctx = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;

        // Log PP stage details
        for (size_t i = 0; i < mdo_config.pp_stages.size(); ++i)
        {
            const auto &stage = mdo_config.pp_stages[i];
            LOG_INFO("[OrchestrationRunner]   Stage " << i << ": "
                                                      << pp_devices[i].toString()
                                                      << " layers [" << stage.first_layer << ", "
                                                      << stage.last_layer << ") "
                                                      << (stage.has_embedding ? "[+embed] " : "")
                                                      << (stage.has_lm_head ? "[+lm_head] " : ""));
        }

        if (!mdo_config.validate())
        {
            return setError("Invalid LOCAL PP configuration");
        }

        // Ensure GlobalBackendRouter is initialized for inter-stage transfers.
        // LOCAL PP uses TensorBase::transferTo() which routes through the backend router.
        GlobalBackendRouter::initForTests();

        std::unique_ptr<RankOrchestrator> orch;
        orch = std::make_unique<RankOrchestrator>(model_ctx_, mdo_config);
        runner_ = std::move(orch);

        LOG_INFO("Local PP compute graph built successfully");
        return true;
    }

    bool OrchestrationRunner::buildSingleDeviceComputeGraph()
    {
        // Determine target device from execution plan
        DeviceId device = DeviceId::cpu();
        std::string device_source = "default (CPU)";

        if (!plan_.local_tp_devices.empty())
        {
            device = plan_.primary_device.toLocalDeviceId();
            device_source = "plan.local_tp_devices[0]";
        }
        else if (!plan_.primary_device.hostname.empty())
        {
            device = plan_.primary_device.toLocalDeviceId();
            device_source = "plan.primary_device";
        }

        // Validate that the requested device actually exists in hardware
        const auto &dm = DeviceManager::instance();
        const bool strict_numa = plan_.primary_device_numa_explicit;

        const bool device_available = strict_numa
                                          ? dm.deviceExists(plan_.primary_device, true)
                                          : dm.deviceExists(device);

        if (!device_available)
        {
            if (strict_numa)
            {
                return setError("Requested device " + plan_.primary_device.toString() +
                                " is not available on the specified NUMA node. Available devices: " +
                                dm.availableDevicesString());
            }

            return setError("Requested device " + device.toString() +
                            " is not available. Available devices: " +
                            dm.availableDevicesString());
        }

        // Log execution strategy decision
        LOG_INFO("[OrchestrationRunner] Execution strategy: SINGLE-DEVICE");
        LOG_INFO("[OrchestrationRunner]   Target device: " << device.toString());
        LOG_INFO("[OrchestrationRunner]   Device source: " << device_source);
        if (device.is_cpu())
        {
            LOG_INFO("[OrchestrationRunner]   Backend: CPU (OneDNN/AVX-512)");
        }
        else if (device.is_cuda())
        {
            LOG_INFO("[OrchestrationRunner]   Backend: CUDA (GPU " << device.ordinal << ")");
        }
        else if (device.is_rocm())
        {
            LOG_INFO("[OrchestrationRunner]   Backend: ROCm (GPU " << device.ordinal << ")");
        }

        // Build config from execution plan via canonical factory
        auto runner_config = InferenceRunnerConfig::fromPlan(plan_);
        runner_config.hostfile = config_.hostfile;
        runner_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
        runner_config.moe_expert_overlay_mpi_ctx = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;

        LOG_INFO("[OrchestrationRunner] Single-device precision config: activation="
                 << activationPrecisionToString(runner_config.activation_precision)
                 << ", kv_cache=" << kvCachePrecisionToString(runner_config.kv_cache_precision));

        // Create runner via factory (returns IInferenceRunner)
        if (model_ctx_)
        {
            runner_ = createInferenceRunner(
                model_ctx_,
                mpi_ctx_,
                device,
                runner_config);
        }

        if (!runner_ && model_ctx_)
        {
            return setError("Failed to create inference runner");
        }

        LOG_INFO("[OrchestrationRunner] Compute graph built successfully");
        return true;
    }

    // =========================================================================
    // Error Handling
    // =========================================================================

    bool OrchestrationRunner::setError(const std::string &error)
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = error;
        LOG_ERROR(error);
        return false;
    }

    // =========================================================================
    // Snapshot API
    // =========================================================================

    void OrchestrationRunner::enableSnapshotCapture(const std::string &output_dir)
    {
        if (runner_)
        {
            runner_->enableSnapshotCapture(output_dir);
        }
    }

    void OrchestrationRunner::disableSnapshotCapture()
    {
        if (runner_)
        {
            runner_->disableSnapshotCapture();
        }
    }

    void OrchestrationRunner::clearSnapshots()
    {
        if (runner_)
        {
            runner_->clearSnapshots();
        }
    }

    const float *OrchestrationRunner::getSnapshot(const std::string &key, size_t &out_size) const
    {
        if (runner_)
        {
            return runner_->getSnapshot(key, out_size);
        }
        out_size = 0;
        return nullptr;
    }

    std::vector<std::string> OrchestrationRunner::getSnapshotKeys() const
    {
        if (runner_)
        {
            return runner_->getSnapshotKeys();
        }
        return {};
    }

    // =========================================================================
    // Profiling
    // =========================================================================

    const GraphExecutorStats *OrchestrationRunner::executorStats() const
    {
        if (runner_)
        {
            return runner_->executorStats();
        }
        return nullptr;
    }

    void OrchestrationRunner::resetExecutorStats()
    {
        if (runner_)
        {
            runner_->resetExecutorStats();
        }
    }

    MoERebalanceController *OrchestrationRunner::moeRebalanceController() const
    {
        if (runner_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner_.get()))
            {
                return dgo->moeRebalanceController();
            }
            if (auto *rank = dynamic_cast<RankOrchestrator *>(runner_.get()))
            {
                return rank->moeRebalanceController();
            }
        }
        return nullptr;
    }

    void OrchestrationRunner::applyMoEExpertMasks(
        const std::vector<std::vector<bool>> &masks,
        const ReceivedWeightsMap &received)
    {
        if (runner_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner_.get()))
            {
                dgo->applyExpertMasks(masks, received);
            }
        }
    }

    bool OrchestrationRunner::applyMoEExpertMasksForAllLocalDevices(
        const MoERebalanceController &controller)
    {
        if (!runner_)
            return false;
        if (auto *rank = dynamic_cast<RankOrchestrator *>(runner_.get()))
        {
            rank->applyMoEExpertMasksForAllDevices(controller);
            return true;
        }
        return false;
    }

    bool OrchestrationRunner::applyMoEExpertMasksForAllLocalDevices(
        const std::vector<std::vector<std::vector<bool>>> &masks_by_socket)
    {
        if (!runner_)
            return false;
        if (auto *rank = dynamic_cast<RankOrchestrator *>(runner_.get()))
        {
            rank->applyMoEExpertMasksForAllDevices(masks_by_socket);
            return true;
        }
        return false;
    }

    void OrchestrationRunner::setExpertReplicaSet(
        const ExpertReplicaSet &replicas, int socket_id)
    {
        if (runner_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner_.get()))
            {
                dgo->setExpertReplicaSet(replicas, socket_id);
            }
            else if (auto *rank = dynamic_cast<RankOrchestrator *>(runner_.get()))
            {
                rank->setExpertReplicaSetForAllDevices(replicas);
            }
        }
    }

    bool OrchestrationRunner::applyMoERebalanceWithReplicas(bool log_histogram_summary)
    {
        auto *controller = moeRebalanceController();
        if (!controller)
            return true;

        if (log_histogram_summary)
            controller->logHistogramSummary();

        std::vector<std::vector<std::vector<bool>>> gpu_cache_masks;
        const int gpu_cache_experts = debugEnv().moe_rebalance.gpu_cache_experts_per_layer;
        if (gpu_cache_experts > 0)
            gpu_cache_masks = controller->computeGpuCacheExpertMasks(gpu_cache_experts);

        const auto old_placement = controller->currentPlacement();
        std::vector<int> new_placement;

        const int max_replicas = controller->maxReplicasPerSocket();
        if (max_replicas > 0)
        {
            controller->proposeReplicas(max_replicas);
            if (controller->hasReplicas())
            {
                if (!mpi_ctx_ || mpi_ctx_->rank() == 0)
                {
                    LOG_INFO("[MoE] Expert replication: "
                             << controller->currentReplicas().num_replicated
                             << " experts replicated (cap=" << max_replicas
                             << " per rank/device, hot_cache="
                             << config_.moe_hot_expert_cache.toString() << ")");
                    LOG_INFO("[MoE] Keeping base expert ownership stable while applying hot-expert replicas");
                }
                controller->resetRebalanceWindow();
            }
        }

        if (!controller->hasReplicas())
        {
            new_placement = controller->rebalance();
            controller->syncReplicaPlacement();
        }

        if (new_placement.empty() && !controller->hasReplicas() && gpu_cache_masks.empty())
            return true;

        ReceivedWeightsMap received;
        if (controller->hasReplicas())
        {
            received = transferReplicaWeights(controller->currentReplicas(), controller->numLayers());
        }
        else if (!new_placement.empty())
        {
            auto manifest = ExpertWeightTransfer::buildManifest(old_placement, new_placement);
            if (!manifest.empty())
                received = transferExpertWeights(manifest, controller->numLayers());
        }

        const int socket_id = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        if (!gpu_cache_masks.empty())
        {
            if (!applyMoEExpertMasksForAllLocalDevices(gpu_cache_masks))
            {
                if (socket_id >= 0 && socket_id < static_cast<int>(gpu_cache_masks.size()))
                    applyMoEExpertMasks(gpu_cache_masks[socket_id], received);
            }
        }
        else if (!applyMoEExpertMasksForAllLocalDevices(*controller))
        {
            auto masks = controller->computeExpertMasks(socket_id);
            applyMoEExpertMasks(masks, received);
        }

        if (controller->hasReplicas())
            setExpertReplicaSet(controller->currentReplicas(), socket_id);

        if (config_.moe_rebalance.release_raw_expert_weights || debugEnv().moe_rebalance.release_raw_weights)
        {
            const size_t freed = releaseRawExpertWeights();
            if (!mpi_ctx_ || mpi_ctx_->rank() == 0)
                LOG_INFO("[MoE] Released " << (freed >> 20) << " MB raw expert weights");
        }

        return true;
    }

    ReceivedWeightsMap OrchestrationRunner::transferExpertWeights(
        const std::vector<ExpertMigration> &manifest, int num_layers)
    {
        if (runner_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner_.get()))
            {
                return dgo->transferExpertWeights(manifest, num_layers);
            }
        }
        return {};
    }

    ReceivedWeightsMap OrchestrationRunner::transferReplicaWeights(
        const ExpertReplicaSet &replicas, int num_layers)
    {
        if (runner_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner_.get()))
            {
                return dgo->transferReplicaWeights(replicas, num_layers);
            }
        }
        return {};
    }

    size_t OrchestrationRunner::releaseRawExpertWeights()
    {
        if (runner_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner_.get()))
            {
                return dgo->releaseRawExpertWeights();
            }
        }
        return 0;
    }

    int OrchestrationRunner::sampleGreedyOnDevice()
    {
        if (runner_)
        {
            return runner_->sampleGreedyOnDevice();
        }
        return -1;
    }

    int OrchestrationRunner::sampleOnDevice(const SamplingParams &params)
    {
        if (runner_)
        {
            return runner_->sampleOnDevice(params);
        }
        return -1;
    }

    void OrchestrationRunner::setSkipLogitsGatherDecode(bool skip)
    {
        // Broadcast to worker ranks
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SKIP_LOGITS_DECODE);
            int32_t val = skip ? 1 : 0;
            mpi_ctx_->broadcast_int32(&val, 1, 0);
        }

        if (runner_)
        {
            runner_->setSkipLogitsGatherDecode(skip);
        }
    }

    void OrchestrationRunner::setSkipLogitsGatherPrefill(bool skip)
    {
        if (runner_)
        {
            runner_->setSkipLogitsGatherPrefill(skip);
        }
    }

    void OrchestrationRunner::setSuppressTimeline(bool suppress)
    {
        if (runner_)
        {
            runner_->setSuppressTimeline(suppress);
        }
    }

    void OrchestrationRunner::setAccumulatePrefill(bool accumulate)
    {
        if (runner_)
        {
            runner_->setAccumulatePrefill(accumulate);
        }
    }

    void OrchestrationRunner::flushStageTimeline()
    {
        if (runner_)
        {
            runner_->flushStageTimeline();
        }
        MoEExpertOverlayProfiler::flush();
    }

    void OrchestrationRunner::setSamplingParams(const SamplingParams &params)
    {
        // Broadcast to worker ranks
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SET_SAMPLING);
            float params_buf[4] = {
                params.temperature,
                params.top_p,
                static_cast<float>(params.top_k),
                static_cast<float>(params.seed)};
            mpi_ctx_->broadcast(params_buf, 4, 0);
        }

        active_sampling_params_ = params;
        // Reset token history for new conversation/request so penalties start fresh
        sampler_.reset_history();
    }

    SamplingParams OrchestrationRunner::getRecommendedSamplingParams() const
    {
        return recommended_sampling_params_;
    }

    std::string OrchestrationRunner::getStopThinkingPrompt() const
    {
        return stop_thinking_prompt_;
    }

    ToolCallFormat OrchestrationRunner::getToolCallFormat() const
    {
        return tool_call_format_;
    }

    // =========================================================================
    // MPI Worker Loop (non-root ranks in server mode)
    // =========================================================================

    void OrchestrationRunner::broadcastCommand(MPICommand cmd)
    {
        if (!mpi_coordinated_mode_ || !mpi_ctx_ || mpi_ctx_->world_size() <= 1)
            return;

        int32_t tag = static_cast<int32_t>(cmd);
        mpi_ctx_->broadcast_int32(&tag, 1, 0);
    }

    void OrchestrationRunner::shutdownMPIWorkers()
    {
        if (!mpi_ctx_ || mpi_ctx_->world_size() <= 1)
            return;

        LOG_INFO("[MPI] Rank 0 sending SHUTDOWN to worker ranks");
        broadcastCommand(MPICommand::SHUTDOWN);
    }

    void OrchestrationRunner::runMPIWorkerLoop()
    {
        if (!mpi_ctx_ || mpi_ctx_->rank() == 0)
        {
            LOG_WARN("[MPIWorkerLoop] Should only be called on non-root ranks");
            return;
        }

        LOG_INFO("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
                                         << " entering worker loop");

        while (true)
        {
            // Wait for command from rank 0
            int32_t tag = 0;
            mpi_ctx_->broadcast_int32(&tag, 1, 0);
            auto cmd = static_cast<MPICommand>(tag);

            switch (cmd)
            {
            case MPICommand::CLEAR_CACHE:
            {
                clearCache();
                break;
            }

            case MPICommand::SET_SAMPLING:
            {
                // Receive sampling params
                float params_buf[4]; // temperature, top_p, top_k, seed
                mpi_ctx_->broadcast(params_buf, 4, 0);
                SamplingParams sp;
                sp.temperature = params_buf[0];
                sp.top_p = params_buf[1];
                sp.top_k = static_cast<int>(params_buf[2]);
                sp.seed = static_cast<uint64_t>(params_buf[3]);
                setSamplingParams(sp);
                break;
            }

            case MPICommand::PREFILL:
            {
                // Receive token count then tokens
                int32_t n_tokens = 0;
                mpi_ctx_->broadcast_int32(&n_tokens, 1, 0);

                std::vector<int32_t> tokens(n_tokens);
                mpi_ctx_->broadcast_int32(tokens.data(), static_cast<size_t>(n_tokens), 0);

                prefill(tokens);
                break;
            }

            case MPICommand::DECODE_STEP:
            {
                decodeStep();
                break;
            }

            case MPICommand::SKIP_LOGITS_DECODE:
            {
                int32_t skip = 0;
                mpi_ctx_->broadcast_int32(&skip, 1, 0);
                runner_->setSkipLogitsGatherDecode(skip != 0);
                break;
            }

            case MPICommand::APPLY_MOE_REBALANCE:
            {
                if (!applyMoERebalanceWithReplicas())
                    LOG_ERROR("[MPIWorkerLoop] MoE rebalance failed on rank " << mpi_ctx_->rank());
                break;
            }

            case MPICommand::SHUTDOWN:
            {
                LOG_INFO("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
                                                 << " received SHUTDOWN");
                return;
            }

            default:
                LOG_WARN("[MPIWorkerLoop] Unknown command: " << tag);
                break;
            }
        }
    }

} // namespace llaminar2
