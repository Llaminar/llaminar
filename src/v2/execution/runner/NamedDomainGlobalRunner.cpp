/**
 * @file NamedDomainGlobalRunner.cpp
 * @brief Implementation of NamedDomainGlobalRunner
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#include "NamedDomainGlobalRunner.h"
#include "../../utils/Logger.h"
#include "../../utils/MPIContext.h"
#include "../../planning/ClusterInventoryGatherer.h"
#include "../../loaders/ModelLoader.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/ModelContextConfig.h"
#include "../../loaders/WeightPlan.h"
#include "../../loaders/WeightManager.h"
#include "../../tensors/TensorFactory.h"
#include "../mpi_orchestration/ExecutionPlanBuilder.h"
#include "../global_pp/GlobalPPRankPlanBuilder.h"
#include "../global/DomainCommunicatorRegistry.h"
#include "../global/StageRunnerFactory.h"
#include "../global/GlobalOrchestrator.h"
#include "../factory/InferenceRunnerFactory.h"
#include "../config/RuntimeConfig.h"
#include "../../utils/Tokenizer.h"
#include "../../config/OrchestrationConfig.h"

#include <algorithm>
#include <set>

namespace llaminar2
{

    // =========================================================================
    // shouldUse
    // =========================================================================

    bool NamedDomainGlobalRunner::shouldUse(const OrchestrationConfig &config)
    {
        if (!config.usesNamedDomains())
            return false;
        if (config.pp_stage_definitions.empty())
            return false;

        for (const auto &domain : config.domain_definitions)
        {
            if (domain.scope == TPScope::NODE_LOCAL || domain.scope == TPScope::GLOBAL)
            {
                return true;
            }
            if (domain.explicit_ranks.size() > 1)
            {
                return true;
            }
            // AUTO scope: check if devices span multiple unique rank-qualified hostnames.
            // Devices with NUMA-qualified addresses like "0:cpu:0" and "1:cpu:0" imply
            // different ranks when the NUMA index equals the rank ordinal.
            // A lightweight heuristic: if the device list has >1 unique hostname prefix.
            std::set<std::string> hosts;
            for (const auto &dev : domain.devices)
            {
                hosts.insert(dev.hostname);
            }
            if (hosts.size() > 1)
            {
                return true;
            }
        }
        return false;
    }

    // =========================================================================
    // Construction
    // =========================================================================

    NamedDomainGlobalRunner::NamedDomainGlobalRunner(
        OrchestrationConfig config,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder)
        : config_(std::move(config)), plan_builder_(std::move(plan_builder))
    {
        if (!plan_builder_)
        {
            plan_builder_ = createExecutionPlanBuilder();
        }
    }

    NamedDomainGlobalRunner::~NamedDomainGlobalRunner()
    {
        if (initialized_)
        {
            shutdown();
        }
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool NamedDomainGlobalRunner::initialize()
    {
        if (initialized_)
            return true;

        try
        {
            // ----------------------------------------------------------
            // Step 1: MPI context
            // ----------------------------------------------------------
            auto mpi_ctx = MPIContextFactory::global();
            if (!mpi_ctx)
            {
                return setError("Failed to acquire MPI context");
            }
            const int my_rank = mpi_ctx->rank();
            const int world_size = mpi_ctx->world_size();

            LOG_DEBUG("NamedDomainGlobalRunner: initializing on rank " << my_rank
                                                                      << " of " << world_size);

            // ----------------------------------------------------------
            // Step 2: Read model metadata for layer count
            // ----------------------------------------------------------
            ModelConfig model_config;
            if (!config_.model_path.empty())
            {
                auto meta_mpi = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
                TensorFactory meta_factory(*meta_mpi);
                ModelLoader meta_loader(&meta_factory);
                meta_loader.setUseMmap(false);

                bool ok = false;
                try
                {
                    ok = meta_loader.loadModel(config_.model_path);
                }
                catch (const std::exception &e)
                {
                    LOG_WARN("NamedDomainGlobalRunner: failed to read model metadata ("
                             << e.what() << "), using defaults");
                }

                if (ok)
                {
                    model_config.n_layers = static_cast<int>(meta_loader.blockCount());
                    model_config.n_heads = static_cast<int>(meta_loader.headCount());
                    model_config.n_kv_heads = static_cast<int>(meta_loader.headCountKV());
                    model_config.hidden_size = static_cast<int>(meta_loader.embeddingLength());
                    model_config.vocab_size = static_cast<int>(meta_loader.vocabSize());
                    LOG_DEBUG("NamedDomainGlobalRunner: model metadata n_layers=" << model_config.n_layers);
                }
                else
                {
                    LOG_WARN("NamedDomainGlobalRunner: using default model_config (n_layers=24)");
                    model_config.n_layers = 24;
                }
            }
            else
            {
                LOG_WARN("NamedDomainGlobalRunner: no model_path; using default model_config");
                model_config.n_layers = 24;
                model_config.vocab_size = 151936;
            }

            // ----------------------------------------------------------
            // Step 3: Gather cluster inventory
            // ----------------------------------------------------------
            auto cluster_inventory = gatherClusterInventory(mpi_ctx,
                                                            config_.tp_devices,
                                                            config_.hostfile);

            // ----------------------------------------------------------
            // Step 4: Build GlobalPPTopology
            // ----------------------------------------------------------
            auto *concrete_builder =
                dynamic_cast<ExecutionPlanBuilder *>(plan_builder_.get());
            if (!concrete_builder)
            {
                return setError("plan_builder_ is not a concrete ExecutionPlanBuilder; "
                                "cannot build GlobalPPTopology");
            }

            GlobalPPTopology topology = concrete_builder->buildGlobalPPTopology(
                config_, model_config, cluster_inventory);

            // Validate topology
            {
                auto errs = topology.validate();
                if (!errs.empty())
                {
                    std::string msg = "GlobalPPTopology validation failed:";
                    for (const auto &e : errs)
                        msg += "\n  - " + e;
                    return setError(msg);
                }
            }

            // ----------------------------------------------------------
            // Step 5: --show-topology output (rank 0 only)
            // ----------------------------------------------------------
            if (config_.show_topology && my_rank == 0)
            {
                LOG_DEBUG("Multi-domain topology:\n"
                         << renderMultiDomainTopologyInfo(topology, world_size));
            }

            // ----------------------------------------------------------
            // Step 6: DomainCommunicatorRegistry (global TP stages)
            // ----------------------------------------------------------
            auto domain_registry = std::make_unique<DomainCommunicatorRegistry>();
            {
                bool has_global_tp = false;
                for (const auto &s : topology.stages)
                {
                    if (s.is_global_tp)
                    {
                        has_global_tp = true;
                        break;
                    }
                }
                if (has_global_tp)
                {
                    domain_registry->initialize(topology, mpi_ctx->communicator(), my_rank);
                }
            }

            // ----------------------------------------------------------
            // Step 7: Per-rank plan
            // ----------------------------------------------------------
            GlobalPPRankPlan rank_plan = GlobalPPRankPlanBuilder::build(topology, my_rank);

            // ----------------------------------------------------------
            // Step 8: Model loading + stage runner construction
            // ----------------------------------------------------------
            // Full model loading is required for real inference.  Without a
            // model path the runner is useful only for topology validation.
            if (config_.model_path.empty())
            {
                return setError("NamedDomainGlobalRunner: model_path is required for full initialization");
            }

            // Build InferenceRunnerConfig base
            auto runtime_cfg = RuntimeConfig::fromOrchestrationConfig(
                config_.max_seq_len,
                config_.activation_precision,
                config_.kv_cache_precision,
                config_.fused_attention_backend,
                config_.moe_expert_mode,
                config_.moe_hot_expert_cache,
                config_.moe_rebalance,
                config_.prefix_cache,
                config_.mtp);
            InferenceRunnerConfig base_runner_cfg;
            base_runner_cfg.max_seq_len = runtime_cfg.max_seq_len;
            base_runner_cfg.batch_size = config_.batch_size;
            base_runner_cfg.activation_precision = runtime_cfg.activation_precision;
            base_runner_cfg.kv_cache_precision = runtime_cfg.kv_cache_precision;
            base_runner_cfg.fused_attention_backend = runtime_cfg.fused_attention_backend;
            base_runner_cfg.moe_expert_mode = runtime_cfg.moe_expert_mode;
            base_runner_cfg.moe_hot_expert_cache = runtime_cfg.moe_hot_expert_cache;
            base_runner_cfg.moe_rebalance = runtime_cfg.moe_rebalance;
            base_runner_cfg.prefix_cache = runtime_cfg.prefix_cache;
            base_runner_cfg.mtp = runtime_cfg.mtp;

            // Load model context once; each stage runner slices its own weight plan
            std::shared_ptr<ModelContext> model_ctx;
            {
                // Build a per-rank plan that covers the full layer range this rank touches
                auto all_execute = rank_plan.executeStages();
                if (all_execute.empty())
                {
                    // This rank has no EXECUTE actions — it's an MPI relay rank.
                    // Still need a model_ctx for vocab size etc.
                }

                // Use first EXECUTE stage's layer range as baseline config;
                // full model loading is always done and weight slicing is handled
                // inside StageRunnerFactory per stage.
                RankExecutionPlan tmp_plan;
                tmp_plan.rank = my_rank;
                tmp_plan.first_layer = 0;
                tmp_plan.last_layer = model_config.n_layers - 1;
                tmp_plan.has_embedding = true;
                tmp_plan.has_lm_head = true;

                ModelContextConfig weight_cfg = ModelContextConfig::fromExecutionPlan(tmp_plan);
                model_ctx = ModelContext::create(config_.model_path, weight_cfg);
                if (!model_ctx)
                {
                    return setError("Failed to load ModelContext from: " + config_.model_path);
                }
            }

            // Build stage runners for each EXECUTE action this rank performs
            StageBuildContext build_ctx;
            build_ctx.model_ctx = model_ctx;
            build_ctx.mpi_ctx = mpi_ctx;
            build_ctx.runner_config = base_runner_cfg;
            build_ctx.domain_registry = domain_registry.get();

            std::vector<StageRunnerEntry> stage_runners;
            for (const auto &step : rank_plan.steps)
            {
                if (step.type != GlobalPPRankPlan::Step::Type::EXECUTE_STAGE)
                    continue;
                const auto &action = step.stage_action;
                if (action.role != RankStageAction::Role::EXECUTE)
                    continue;

                // Find matching stage spec
                const GlobalPPStageSpec *spec_ptr = nullptr;
                for (const auto &s : topology.stages)
                {
                    if (s.stage_id == action.stage_id)
                    {
                        spec_ptr = &s;
                        break;
                    }
                }
                if (!spec_ptr)
                {
                    return setError("No stage spec found for stage_id=" +
                                    std::to_string(action.stage_id));
                }

                try
                {
                    auto entry = StageRunnerFactory::create(*spec_ptr, action, build_ctx);
                    stage_runners.push_back(std::move(entry));
                }
                catch (const std::exception &e)
                {
                    return setError(std::string("StageRunnerFactory::create failed for stage ") +
                                    std::to_string(action.stage_id) + ": " + e.what());
                }
            }

            // ----------------------------------------------------------
            // Step 9: Build GlobalOrchestrator
            // ----------------------------------------------------------
            GlobalOrchestrator::Config go_cfg;
            go_cfg.topology = topology;
            go_cfg.rank = my_rank;
            go_cfg.world_size = world_size;
            go_cfg.mpi_ctx = mpi_ctx.get();
            go_cfg.vocab_size = model_ctx ? model_ctx->vocabSize() : model_config.vocab_size;
            go_cfg.d_model = model_ctx ? static_cast<int>(model_ctx->embeddingLength()) : 896;
            go_cfg.architecture_name = model_ctx ? model_ctx->architecture() : "unknown";

            if (!stage_runners.empty())
            {
                go_cfg.stage_runners = std::move(stage_runners);
            }

            auto global_orch = std::make_unique<GlobalOrchestrator>(std::move(go_cfg));

            // ----------------------------------------------------------
            // Step 10: Build GlobalOrchestratorRunner and delegate
            // ----------------------------------------------------------
            std::shared_ptr<ITokenizer> tokenizer;
            if (model_ctx)
            {
                tokenizer = createTokenizer(model_ctx);
            }

            GlobalOrchestratorRunner::Config runner_cfg;
            runner_cfg.orchestration_config = config_;
            runner_cfg.topology = topology;
            runner_cfg.mpi_ctx = mpi_ctx;
            runner_cfg.global_orchestrator = std::move(global_orch);
            runner_cfg.tokenizer = tokenizer;

            inner_ = std::make_unique<GlobalOrchestratorRunner>(std::move(runner_cfg));
            if (!inner_->initialize())
            {
                return setError("GlobalOrchestratorRunner::initialize() failed: " +
                                inner_->lastError());
            }

            initialized_ = true;
            LOG_DEBUG("NamedDomainGlobalRunner: initialized on rank " << my_rank);
            return true;
        }
        catch (const std::exception &e)
        {
            return setError(std::string("NamedDomainGlobalRunner::initialize() threw: ") + e.what());
        }
    }

    void NamedDomainGlobalRunner::shutdown()
    {
        if (inner_)
            inner_->shutdown();
        inner_.reset();
        initialized_ = false;
    }

    // =========================================================================
    // Inference delegation
    // =========================================================================

    bool NamedDomainGlobalRunner::prefill(const std::vector<int32_t> &tokens)
    {
        if (!initialized_ || !inner_)
            return setError("Not initialized");
        return inner_->prefill(tokens);
    }

    GenerationResult NamedDomainGlobalRunner::decodeStep()
    {
        if (!initialized_ || !inner_)
        {
            GenerationResult r;
            r.error = "Not initialized";
            return r;
        }
        return inner_->decodeStep();
    }

    GenerationResult NamedDomainGlobalRunner::generate(
        const std::vector<int32_t> &prompt_tokens,
        int max_new_tokens,
        const SamplingParams &sampling)
    {
        if (!initialized_ || !inner_)
        {
            GenerationResult r;
            r.error = "Not initialized";
            return r;
        }
        return inner_->generate(prompt_tokens, max_new_tokens, sampling);
    }

    // =========================================================================
    // Configuration / Status delegation
    // =========================================================================

    const RankExecutionPlan &NamedDomainGlobalRunner::executionPlan() const
    {
        return empty_plan_;
    }

    const OrchestrationConfig &NamedDomainGlobalRunner::config() const
    {
        return config_;
    }

    bool NamedDomainGlobalRunner::isInitialized() const
    {
        return initialized_;
    }

    const std::string &NamedDomainGlobalRunner::lastError() const
    {
        return last_error_;
    }

    int NamedDomainGlobalRunner::vocabSize() const
    {
        return inner_ ? inner_->vocabSize() : 0;
    }

    int NamedDomainGlobalRunner::currentPosition() const
    {
        return inner_ ? inner_->currentPosition() : 0;
    }

    void NamedDomainGlobalRunner::clearCache()
    {
        if (inner_)
            inner_->clearCache();
    }

    PrefixRuntimeStateSnapshot NamedDomainGlobalRunner::prefixStateProbe() const
    {
        return inner_ ? inner_->prefixStateProbe() : PrefixRuntimeStateSnapshot{};
    }

    const float *NamedDomainGlobalRunner::lastLogits() const
    {
        return inner_ ? inner_->lastLogits() : nullptr;
    }

    void NamedDomainGlobalRunner::setStopTokens(const std::vector<int32_t> &stop_tokens)
    {
        if (inner_)
            inner_->setStopTokens(stop_tokens);
    }

    std::shared_ptr<ITokenizer> NamedDomainGlobalRunner::tokenizer() const
    {
        return inner_ ? inner_->tokenizer() : nullptr;
    }

    const std::string &NamedDomainGlobalRunner::architecture() const
    {
        static const std::string empty;
        return inner_ ? inner_->architecture() : empty;
    }

    void NamedDomainGlobalRunner::enableSnapshotCapture(const std::string &output_dir)
    {
        if (inner_)
            inner_->enableSnapshotCapture(output_dir);
    }

    void NamedDomainGlobalRunner::disableSnapshotCapture()
    {
        if (inner_)
            inner_->disableSnapshotCapture();
    }

    void NamedDomainGlobalRunner::clearSnapshots()
    {
        if (inner_)
            inner_->clearSnapshots();
    }

    const float *NamedDomainGlobalRunner::getSnapshot(const std::string &key, size_t &out_size) const
    {
        if (inner_)
            return inner_->getSnapshot(key, out_size);
        out_size = 0;
        return nullptr;
    }

    std::vector<std::string> NamedDomainGlobalRunner::getSnapshotKeys() const
    {
        if (inner_)
            return inner_->getSnapshotKeys();
        return {};
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    bool NamedDomainGlobalRunner::setError(const std::string &msg)
    {
        last_error_ = msg;
        LOG_ERROR("[NamedDomainGlobalRunner] " << msg);
        return false;
    }

} // namespace llaminar2
