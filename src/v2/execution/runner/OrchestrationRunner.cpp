/**
 * @file OrchestrationRunner.cpp
 * @brief Implementation of OrchestrationRunner
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "OrchestrationRunner.h"
#include "MTPVerifierForwardExecutor.h"
#include "../../app/StartupBanner.h"
#include "../../config/OrchestrationConfigParser.h"
#include "../../config/TPPPValidator.h"
#include "../mpi_orchestration/ExecutionPlanBuilder.h"
#include "../factory/InferenceRunnerFactory.h"
#include "../mtp/MTPStateTransaction.h"
#include "../mtp/MTPDecodeCatchup.h"
#include "../mtp/MTPRejectionSampler.h"
#include "../mtp/MTPSpecDecodeMetadata.h"
#include "../mtp/MTPSpecDecodeTransaction.h"
#include "../mtp/MTPSpecStateContract.h"
#include "../mtp/MTPSpecTransactionDriver.h"
#include "../mtp/MTPVerifierPolicy.h"
#include "../mtp/MTPWeightManifest.h"
#include "../prefix_cache/PrefixCacheCoordinator.h"
#include "../local_execution/engine/PrefillBucketUtils.h"
#include "../local_execution/orchestrators/RankOrchestrator.h"
#include "../../kernels/common/SamplingMath.h"
#include "../parallelism_tree/ParallelismTree.h"
#include "../parallelism_tree/TreeToRunnerCompiler.h"
#include "../../collective/LocalTPContext.h"
#include "../../collective/ILocalPPContext.h"
#include "../../collective/BackendRouter.h"
#include "../../backends/BackendManager.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/ModelContextConfig.h"
#include "../../loaders/ModelLoader.h"
#include "../../loaders/MmapRegion.h"
#include "../local_execution/graph/SchemaFactoryRegistry.h"
#include "../../backends/ComputeBackend.h"
#include "../../planning/ClusterInventoryGatherer.h"
#include "../../planning/ModelMemoryProfile.h"
#include "../../planning/ActivationBufferSizing.h"
#include "../../backends/DeviceAddressAdapter.h"
#include "../../kernels/KernelFactory.h"
#include "../../tensors/TensorFactory.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/MPITopology.h"
#include "../../utils/NodeDetection.h"
#include "../../utils/NUMATopology.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/WeightLoadingProfiler.h"
#include "../local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "../../execution/moe/MoERebalanceController.h"
#include "../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../execution/moe/ExpertWeightTransfer.h"
#include "../../execution/moe/MoEExpertParallelPlan.h"
#include "../../execution/moe/MoEExpertOverlayExecutionPlan.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <print>
#include <sstream>
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

        /**
         * @brief Infer the coarse model class used by generated MTP depth rules.
         *
         * The controller should not depend on exact GGUF architecture strings,
         * but MoE and dense models have different speculative-depth economics.
         * Collapse the metadata string into the smallest useful policy key.
         */
        MTPDepthPolicyModelClass inferMTPDepthPolicyModelClass(
            const std::shared_ptr<ModelContext> &model_ctx)
        {
            if (!model_ctx)
                return MTPDepthPolicyModelClass::Any;

            std::string architecture = model_ctx->architecture();
            std::transform(
                architecture.begin(),
                architecture.end(),
                architecture.begin(),
                [](unsigned char c)
                { return static_cast<char>(std::tolower(c)); });
            if (architecture.empty())
                return MTPDepthPolicyModelClass::Any;
            if (architecture.find("moe") != std::string::npos)
                return MTPDepthPolicyModelClass::MoE;
            return MTPDepthPolicyModelClass::Dense;
        }

        bool samplingParamsEqual(const SamplingParams &a, const SamplingParams &b)
        {
            return a.temperature == b.temperature &&
                   a.top_k == b.top_k &&
                   a.top_p == b.top_p &&
                   a.seed == b.seed &&
                   a.presence_penalty == b.presence_penalty &&
                   a.frequency_penalty == b.frequency_penalty &&
                   a.dry_multiplier == b.dry_multiplier &&
                   a.dry_base == b.dry_base &&
                   a.dry_allowed_length == b.dry_allowed_length &&
                   a.dry_penalty_last_n == b.dry_penalty_last_n &&
                   a.dry_sequence_breakers == b.dry_sequence_breakers;
        }

        int sampleDistributionWithThreshold(
            const std::vector<SamplingDistributionEntry> &distribution,
            float threshold)
        {
            if (distribution.empty())
                return -1;

            const float clamped_threshold =
                std::clamp(threshold, 0.0f, std::nextafter(1.0f, 0.0f));
            float cumulative = 0.0f;
            int fallback_token = -1;
            for (const auto &entry : distribution)
            {
                if (entry.token_id < 0 || !(entry.probability > 0.0f))
                    continue;
                cumulative += entry.probability;
                fallback_token = entry.token_id;
                if (clamped_threshold <= cumulative)
                    return entry.token_id;
            }
            return fallback_token;
        }

        int sampleResidualDistributionWithThreshold(
            const std::vector<SamplingDistributionEntry> &target,
            const std::vector<SamplingDistributionEntry> &draft,
            float threshold)
        {
            return sampleDistributionWithThreshold(
                Sampler::residual_distribution(target, draft),
                threshold);
        }

        /**
         * @brief Build the current single-request target-verifier row plan.
         *
         * The transaction pipeline still runs one request per runner today.
         * Centralizing this conversion keeps the compact verifier row contract
         * next to the metadata model instead of leaving it implicit in the
         * runner's draft-token vector handling.
         */
        MTPSpecDecodeVerifierInputPlan buildSingleRequestVerifierInputPlan(
            const std::vector<int32_t> &draft_tokens)
        {
            MTPSpecDecodeMetadataShape shape;
            shape.max_requests = 1;
            shape.max_draft_tokens = static_cast<int>(draft_tokens.size());

            MTPSpecDecodeVerifierDraftRequest request;
            request.request_id = 0;
            request.draft_tokens = draft_tokens;
            return buildMTPSpecDecodeVerifierInputPlan(shape, {request});
        }

        /**
         * @brief Validate the compact verifier row metadata before graph install.
         *
         * The graph builder now accepts arbitrary compact source rows. This
         * helper only checks the metadata shape invariants that the sampler
         * needs before handing the full plan to the runner for graph-specific
         * row validation and, on GPU, workspace upload.
         */
        bool verifierInputPlanHasCompactRows(
            const MTPSpecDecodeVerifierInputPlan &plan)
        {
            if (!plan.ok ||
                plan.compact_logit_row_count !=
                    static_cast<int>(plan.verifier_logit_rows.size()))
            {
                return false;
            }
            for (int row = 0; row < plan.compact_logit_row_count; ++row)
            {
                if (plan.verifier_logit_rows[static_cast<size_t>(row)] < 0)
                    return false;
            }
            return true;
        }

        class ScopedMTPAllPositionVerifierSyncDeferral
        {
        public:
            ScopedMTPAllPositionVerifierSyncDeferral(
                IInferenceRunner *runner,
                bool enabled)
                : runner_(runner),
                  enabled_(enabled && runner != nullptr)
            {
                if (enabled_)
                    runner_->setMTPAllPositionVerifierSyncDeferralEnabled(true);
            }

            ~ScopedMTPAllPositionVerifierSyncDeferral()
            {
                if (enabled_)
                    runner_->setMTPAllPositionVerifierSyncDeferralEnabled(false);
            }

            ScopedMTPAllPositionVerifierSyncDeferral(
                const ScopedMTPAllPositionVerifierSyncDeferral &) = delete;
            ScopedMTPAllPositionVerifierSyncDeferral &operator=(
                const ScopedMTPAllPositionVerifierSyncDeferral &) = delete;

        private:
            IInferenceRunner *runner_ = nullptr;
            bool enabled_ = false;
        };

        /**
         * @brief Installs a verifier row plan for one scoped forward call.
         *
         * Device runners upload this plan into their graph metadata workspace
         * immediately before the row-indexed all-position verifier executes.
         * The destructor clears the plan so a cached verifier graph can never
         * accidentally replay with row metadata from an older speculative step.
         */
        class ScopedMTPSpecVerifierInputPlan
        {
        public:
            ScopedMTPSpecVerifierInputPlan(
                IInferenceRunner *runner,
                const MTPSpecDecodeVerifierInputPlan &plan)
                : runner_(runner),
                  installed_(runner != nullptr &&
                             runner->setMTPSpecVerifierInputPlan(plan))
            {
            }

            ~ScopedMTPSpecVerifierInputPlan()
            {
                if (runner_)
                    runner_->clearMTPSpecVerifierInputPlan();
            }

            bool installed() const { return installed_; }

            ScopedMTPSpecVerifierInputPlan(
                const ScopedMTPSpecVerifierInputPlan &) = delete;
            ScopedMTPSpecVerifierInputPlan &operator=(
                const ScopedMTPSpecVerifierInputPlan &) = delete;

        private:
            IInferenceRunner *runner_ = nullptr;
            bool installed_ = false;
        };

        void synchronizeRunnerPrimaryDeviceBeforeRelease(const IInferenceRunner *runner)
        {
            if (!runner)
                return;

            const DeviceId device = runner->primaryDeviceId();
            if (!device.is_gpu())
                return;

            IBackend *backend = getBackendFor(device);
            if (!backend)
            {
                LOG_WARN("[OrchestrationRunner] Could not synchronize "
                         << device.toString()
                         << " before runner shutdown: backend unavailable");
                return;
            }

            if (!backend->synchronize(device.gpu_ordinal()))
            {
                LOG_WARN("[OrchestrationRunner] Device synchronization failed before runner shutdown on "
                         << device.toString());
            }
        }

        const char *prefixStorageTierName(PrefixStorageTier tier)
        {
            switch (tier)
            {
            case PrefixStorageTier::Ram:
                return "ram";
            case PrefixStorageTier::DeviceHot:
                return "device-hot";
            case PrefixStorageTier::Disk:
                return "disk-hydrated";
            }
            return "none";
        }

        std::string summarizePrefixStorageTiers(const std::vector<PrefixBlockHandle> &blocks)
        {
            if (blocks.empty())
                return "none";

            PrefixStorageTier first_tier = blocks.front().tier;
            for (const auto &block : blocks)
            {
                if (block.tier != first_tier)
                    return "mixed";
            }
            return prefixStorageTierName(first_tier);
        }

        bool prefixBlocksContainHybridState(const std::vector<PrefixBlockHandle> &blocks)
        {
            return std::any_of(blocks.begin(), blocks.end(),
                               [](const PrefixBlockHandle &block)
                               {
                                   return block.has_hybrid_state;
                               });
        }

        bool prefixBlocksContainMTPState(const std::vector<PrefixBlockHandle> &blocks)
        {
            return std::any_of(blocks.begin(), blocks.end(),
                               [](const PrefixBlockHandle &block)
                               {
                                   return block.mtp_payload != nullptr ||
                                          (block.mtp_storage && !block.mtp_storage->empty()) ||
                                          block.device_mtp_storage != nullptr;
                               });
        }

        int snapshotShiftedMTPTokens(const PrefixStateSnapshot &snapshot)
        {
            int tokens = -1;
            for (int count : snapshot.mtp_cached_tokens)
            {
                if (count >= 0)
                    tokens = std::max(tokens, count);
            }
            if (tokens >= 0)
                return tokens;

            for (const auto &block : snapshot.mtp_blocks)
            {
                tokens = std::max(tokens, block.key.token_count);
            }
            if (tokens >= 0)
                return tokens;

            return expectedShiftedMTPTokens(snapshot.cached_tokens);
        }

        MTPDecodeStateStamp makeMTPStateStamp(
            const PrefixStateSnapshot &snapshot,
            std::string label,
            bool has_terminal_hidden,
            bool has_terminal_logits,
            bool has_ready_token)
        {
            MTPDecodeStateStamp stamp;
            stamp.valid = snapshot.valid;
            stamp.logical_tokens = snapshot.cached_tokens;
            stamp.main_kv_tokens = snapshot.cached_tokens;
            stamp.shifted_mtp_kv_tokens = snapshotShiftedMTPTokens(snapshot);
            stamp.position = snapshot.cached_tokens;
            stamp.has_terminal_hidden = has_terminal_hidden;
            stamp.has_terminal_logits = has_terminal_logits;
            stamp.has_ready_token = has_ready_token;
            stamp.provenance = snapshot.provenance;
            stamp.label = std::move(label);
            return stamp;
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

            // Print consolidated startup banner (rank 0 only, after all preflight passes)
            printStartupBanner();

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
                            LOG_DEBUG("[OrchestrationRunner] Model-recommended sampling: "
                                      << "temp=" << recommended_sampling_params_.temperature
                                      << " top_p=" << recommended_sampling_params_.top_p
                                      << " top_k=" << recommended_sampling_params_.top_k
                                      << " presence_penalty=" << recommended_sampling_params_.presence_penalty
                                      << " frequency_penalty=" << recommended_sampling_params_.frequency_penalty);
                        }
                        if (!stop_thinking_prompt_.empty())
                        {
                            LOG_DEBUG("[OrchestrationRunner] Stop-thinking prompt configured ("
                                      << stop_thinking_prompt_.size() << " chars)");
                        }
                    }
                }
            }

            LOG_DEBUG("OrchestrationRunner initialized successfully");
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

        PerfStatsCollector::flushFromEnv();

        if (runner_)
        {
            runner_->clear_cache();
            synchronizeRunnerPrimaryDeviceBeforeRelease(runner_.get());
        }

        // Release resources in reverse order
        runner_.reset();
        llaminar::v2::kernels::KernelFactory::clearCache();
        local_pp_ctx_.reset();
        local_tp_ctx_.reset();
        model_ctx_.reset();

        initialized_ = false;
        LOG_DEBUG("OrchestrationRunner shut down");
    }

    // =========================================================================
    // Inference
    // =========================================================================

    bool OrchestrationRunner::forwardPrefillTokens(
        const int *tokens,
        int token_count,
        const std::string &failure_message)
    {
        if (!runner_ || !tokens || token_count <= 0)
            return setError(failure_message);

        const auto &exec = debugEnv().execution;
        const auto buckets = normalizePrefillGraphBuckets(exec.prefill_graph_bucket_sizes);
        const bool long_bucketed_prefill =
            exec.gpu_graphs &&
            exec.prefill_graph_buckets &&
            token_count >= exec.prefill_graph_min_seq &&
            !buckets.empty() &&
            token_count > buckets.back();

        if (long_bucketed_prefill && runner_->supportsPrefillChunkSchedule(token_count))
        {
            PrefillChunkSchedulerPolicy policy;
            policy.bucket_sizes = buckets;
            policy.fixed_chunk_real_tokens = buckets.back();
            policy.min_rebalance_interval_tokens = buckets.back();
            policy.max_rebalance_interval_tokens = 0;
            policy.real_token_start = runner_->get_position();
            policy.real_token_count = token_count;

            PrefillChunkSchedule chunk_schedule = planPrefillChunkSchedule(policy);
            if (!chunk_schedule)
            {
                ++prefill_chunk_stats_.schedules;
                ++prefill_chunk_stats_.failures;
                return setError(failure_message + " (chunk planning failed: " +
                                chunk_schedule.error + ")");
            }

            uint64_t padded_tokens = 0;
            for (const auto &chunk : chunk_schedule.chunks)
            {
                padded_tokens += static_cast<uint64_t>(
                    std::max(0, chunk.bucket_seq_len - chunk.real_count));
            }

            ++prefill_chunk_stats_.schedules;
            if (runner_->forwardPrefillChunkSchedule(
                    tokens,
                    token_count,
                    policy,
                    exec.prefill_graph_pad_token_id,
                    /*allow_padded_execution=*/true))
            {
                ++prefill_chunk_stats_.successful_schedules;
                prefill_chunk_stats_.chunks +=
                    static_cast<uint64_t>(chunk_schedule.chunks.size());
                prefill_chunk_stats_.real_tokens += static_cast<uint64_t>(token_count);
                prefill_chunk_stats_.padded_tokens += padded_tokens;
                return true;
            }

            ++prefill_chunk_stats_.failures;
            return setError(failure_message + " (chunked prefill failed)");
        }

        if (!runner_->forward(tokens, token_count))
            return setError(failure_message);
        return true;
    }

    void OrchestrationRunner::clearBatchedDecodeState()
    {
        batched_decode_active_ = false;
        batched_request_states_.clear();
    }

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

        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        mtp_stats_ = {};
        ready_sampled_token_.reset();
        ready_sampled_params_.reset();
        clearBatchedDecodeState();
        last_token_ = prompt_tokens.back();

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

        const auto &plan_prefix = plan_.runtime.prefix_cache;
        const auto &config_prefix = config_.prefix_cache;
        const MTPRuntimeConfig &active_mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        const bool prefix_cache_enabled =
            (plan_prefix.enabled || config_prefix.enabled) &&
            plan_prefix.storage_mode != PrefixCacheStorageMode::Disabled &&
            config_prefix.storage_mode != PrefixCacheStorageMode::Disabled;
        const bool mtp_full_hit_requires_terminal_hidden =
            active_mtp.enabled && active_mtp.require_terminal_hidden_for_full_hit;
        prefix_request_summary_ = {};
        prefix_request_summary_.enabled = prefix_cache_enabled;
        prefix_request_summary_.requested_tokens = static_cast<int>(prompt_tokens.size());

        if (active_mtp.enabled && runner_)
        {
            if (!ensureMTPDepthController(active_mtp))
            {
                return false;
            }
            const int effective_max_draft_tokens = effectiveMTPMaxDraftDepth(active_mtp);
            if (effective_max_draft_tokens < 1 || effective_max_draft_tokens > 3)
            {
                return setError(
                    "MTP decode supports --mtp-draft-tokens in the range [1, 3] for verifier M=2..4");
            }
            if (effective_max_draft_tokens > 1 && !runner_->supportsChainedMTPDrafts())
            {
                return setError(
                    "MTP decode with --mtp-draft-tokens > 1 requires runner support for chained MTP sidecars");
            }
        }
        else
        {
            mtp_depth_controller_.reset();
        }

        if (prefix_cache_enabled)
        {
            try
            {
                PrefixLookupResult local_hit = runner_->lookupPrefix(prompt_tokens);
                PrefixParticipantLookup participant = makePrefixParticipantLookup(
                    mpi_ctx_ ? mpi_ctx_->rank() : 0,
                    runner_->primaryDeviceId(),
                    local_hit,
                    {},
                    runner_->moePlacementEpoch());

                PrefixCoordinationResult coordination;
                if (mpi_ctx_ && mpi_ctx_->world_size() > 1 &&
                    mpi_ctx_->communicator() != MPI_COMM_NULL)
                {
                    MPIPrefixCollectiveCoordinator domain_coordinator(mpi_ctx_->communicator());
                    coordination = coordinatePrefixLookups({participant}, &domain_coordinator);
                }
                else
                {
                    coordination = coordinatePrefixLookups({participant});
                }
                const int coordination_block_size =
                    local_hit.block_size > 0 ? local_hit.block_size : plan_prefix.block_size;
                PrefixLookupResult coordinated_hit =
                    makePrefixLookupResult(coordination, coordination_block_size);
                int matched_tokens = coordinated_hit.cached_tokens;
                LOG_DEBUG("[OrchestrationRunner] Prefix cache lookup summary: local_tokens="
                          << local_hit.cached_tokens
                          << " coordinated_tokens=" << coordinated_hit.cached_tokens
                          << " supported=" << coordinated_hit.supported
                          << " terminal_logits=" << coordinated_hit.has_terminal_logits
                          << " terminal_hidden=" << coordinated_hit.has_terminal_hidden
                          << " requires_terminal_logits=" << coordinated_hit.requires_terminal_logits
                          << " requires_terminal_hidden=" << coordinated_hit.requires_terminal_hidden
                          << " blocks=" << coordinated_hit.blocks.size()
                          << " bypass_reason=" << coordinated_hit.bypass_reason);

                runner_->clear_cache();
                prefill_logits_ready_ = false;
                ready_sampled_token_.reset();
                ready_sampled_params_.reset();

                auto make_common_hit = [&]()
                {
                    PrefixLookupResult hit = local_hit.clampedTo(matched_tokens);
                    hit.cache_enabled = coordinated_hit.cache_enabled;
                    hit.supported = coordinated_hit.supported;
                    hit.fingerprint_key = coordinated_hit.fingerprint_key != 0
                                              ? coordinated_hit.fingerprint_key
                                              : hit.fingerprint_key;
                    hit.placement_epoch = coordinated_hit.placement_epoch;
                    hit.bypass_reason = coordinated_hit.bypass_reason;
                    hit.has_terminal_logits =
                        hit.has_terminal_logits && coordinated_hit.has_terminal_logits;
                    hit.has_terminal_hidden =
                        hit.has_terminal_hidden && coordinated_hit.has_terminal_hidden;
                    return hit;
                };

                PrefixLookupResult common_hit = make_common_hit();
                prefix_request_summary_.bypassed = !coordinated_hit.supported;
                prefix_request_summary_.bypass_reason = coordinated_hit.bypass_reason;
                if (active_mtp.enabled &&
                    matched_tokens > 0 &&
                    matched_tokens < static_cast<int>(prompt_tokens.size()) &&
                    !common_hit.has_terminal_hidden)
                {
                    const int block_size =
                        common_hit.block_size > 0 ? common_hit.block_size : plan_prefix.block_size;
                    matched_tokens = std::max(0, matched_tokens - std::max(1, block_size));
                    common_hit = make_common_hit();
                }

                if (matched_tokens > 0 && !runner_->populatePrefix(common_hit))
                {
                    LOG_DEBUG("[OrchestrationRunner] Prefix cache populate failed at matched_tokens="
                              << matched_tokens);
                    matched_tokens = 0;
                    common_hit = make_common_hit();
                    runner_->clear_cache();
                }

                int suffix_start = matched_tokens;
                int suffix_len = static_cast<int>(prompt_tokens.size()) - suffix_start;
                bool terminal_state_restored = false;

                if (suffix_len > 0)
                {
                    if (!forwardPrefillTokens(prompt_tokens.data() + suffix_start,
                                              suffix_len,
                                              "Forward pass failed during prefix-cache suffix prefill"))
                        return false;
                    prefill_logits_ready_ = true;
                }
                else if (common_hit.has_terminal_logits &&
                         (!mtp_full_hit_requires_terminal_hidden ||
                          common_hit.has_terminal_hidden) &&
                         runner_->restorePrefixTerminalState(common_hit))
                {
                    prefill_logits_ready_ = true;
                    terminal_state_restored = true;
                }
                else
                {
                    LOG_DEBUG("[OrchestrationRunner] Prefix cache terminal restore unavailable; "
                              << "matched_tokens=" << matched_tokens
                              << " has_terminal_logits=" << common_hit.has_terminal_logits
                              << " has_terminal_hidden=" << common_hit.has_terminal_hidden
                              << " mtp_requires_hidden=" << mtp_full_hit_requires_terminal_hidden);
                    const int block_size =
                        common_hit.block_size > 0 ? common_hit.block_size : plan_prefix.block_size;
                    matched_tokens = std::max(0, matched_tokens - std::max(1, block_size));
                    common_hit = make_common_hit();
                    runner_->clear_cache();
                    if (matched_tokens > 0 && !runner_->populatePrefix(common_hit))
                    {
                        matched_tokens = 0;
                        runner_->clear_cache();
                    }
                    suffix_start = matched_tokens;
                    suffix_len = static_cast<int>(prompt_tokens.size()) - suffix_start;
                    if (!forwardPrefillTokens(prompt_tokens.data() + suffix_start,
                                              suffix_len,
                                              "Forward pass failed during prefix-cache terminal recompute"))
                        return false;
                    prefill_logits_ready_ = true;
                }

                runner_->harvestPrefix(prompt_tokens, static_cast<int>(prompt_tokens.size()));

                const bool full_hit = matched_tokens == static_cast<int>(prompt_tokens.size());
                prefix_request_summary_.hit = matched_tokens > 0 && full_hit;
                prefix_request_summary_.partial_hit = matched_tokens > 0 && !full_hit;
                prefix_request_summary_.matched_tokens = matched_tokens;
                const int summary_block_size =
                    common_hit.block_size > 0 ? common_hit.block_size : plan_prefix.block_size;
                prefix_request_summary_.matched_blocks =
                    !common_hit.blocks.empty()
                        ? static_cast<int>(common_hit.blocks.size())
                        : (summary_block_size > 0 ? matched_tokens / summary_block_size : 0);
                prefix_request_summary_.terminal_logits_restored = terminal_state_restored;
                prefix_request_summary_.terminal_hidden_restored =
                    terminal_state_restored && common_hit.has_terminal_hidden;
                prefix_request_summary_.mtp_state_restored =
                    matched_tokens > 0 && prefixBlocksContainMTPState(common_hit.blocks);
                prefix_request_summary_.hybrid_state_restored =
                    matched_tokens > 0 && prefixBlocksContainHybridState(common_hit.blocks);
                prefix_request_summary_.storage_tier = summarizePrefixStorageTiers(common_hit.blocks);

                LOG_INFO("[OrchestrationRunner] Prefix cache request: "
                         << (matched_tokens > 0 ? (full_hit ? "hit" : "partial-hit") : "miss")
                         << " matched_tokens=" << matched_tokens
                         << " prompt_tokens=" << prompt_tokens.size()
                         << " terminal_logits="
                         << (common_hit.has_terminal_logits ? "yes" : "no"));
                return true;
            }
            catch (const std::exception &e)
            {
                return setError(std::string("Prefill with prefix cache failed: ") + e.what());
            }
        }

        // Run forward pass
        try
        {
            if (!forwardPrefillTokens(prompt_tokens.data(),
                                      static_cast<int>(prompt_tokens.size()),
                                      "Forward pass failed during prefill"))
                return false;
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

    bool OrchestrationRunner::supportsPrefillBatch(int request_batch) const
    {
        if (!initialized_ || !runner_ || request_batch <= 1)
            return false;

        const MTPRuntimeConfig &mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        if (!mtp.enabled || request_batch > mtp.max_request_batch)
            return false;

        const auto &plan_prefix = plan_.runtime.prefix_cache;
        const auto &config_prefix = config_.prefix_cache;
        const bool prefix_cache_enabled =
            (plan_prefix.enabled || config_prefix.enabled) &&
            plan_prefix.storage_mode != PrefixCacheStorageMode::Disabled &&
            config_prefix.storage_mode != PrefixCacheStorageMode::Disabled;
        if (prefix_cache_enabled)
            return false;

        if (plan_.usesLocalTP() ||
            plan_.usesLocalPP() ||
            plan_.usesGlobalTP() ||
            plan_.usesPipelineParallel())
        {
            return false;
        }

        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
            return false;

        return runner_->batch_size() >= request_batch;
    }

    bool OrchestrationRunner::prefillBatch(
        const std::vector<std::vector<int32_t>> &token_batches)
    {
        if (!initialized_)
            return setError("Runner not initialized");
        if (!runner_)
            return setError("Runner unavailable");

        const int request_batch = static_cast<int>(token_batches.size());
        if (request_batch <= 1)
        {
            return setError(
                "Request-batched prefill requires at least two logical requests");
        }

        const MTPRuntimeConfig &mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        if (!mtp.enabled)
            return setError("Request-batched prefill requires MTP to be enabled");
        if (request_batch > mtp.max_request_batch)
        {
            return setError(
                "Request-batched prefill exceeds configured MTP max_request_batch");
        }

        const auto &plan_prefix = plan_.runtime.prefix_cache;
        const auto &config_prefix = config_.prefix_cache;
        const bool prefix_cache_enabled =
            (plan_prefix.enabled || config_prefix.enabled) &&
            plan_prefix.storage_mode != PrefixCacheStorageMode::Disabled &&
            config_prefix.storage_mode != PrefixCacheStorageMode::Disabled;
        if (prefix_cache_enabled)
        {
            return setError(
                "Request-batched prefill with prefix cache requires Phase 9 "
                "common-prefix coordination");
        }

        if (plan_.usesLocalTP() ||
            plan_.usesLocalPP() ||
            plan_.usesGlobalTP() ||
            plan_.usesPipelineParallel())
        {
            return setError(
                "Request-batched prefill is currently implemented only for "
                "SingleDevice runners");
        }

        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            return setError(
                "Request-batched prefill is not enabled for MPI multi-rank runners");
        }

        if (runner_->batch_size() < request_batch)
        {
            return setError(
                "Request-batched prefill exceeds initialized runner batch capacity");
        }

        std::vector<std::vector<int>> converted;
        converted.reserve(token_batches.size());
        std::vector<BatchedDecodeRequestState> next_states;
        next_states.reserve(token_batches.size());
        for (const std::vector<int32_t> &tokens : token_batches)
        {
            if (tokens.empty())
                return setError("Request-batched prefill received an empty prompt");

            converted.emplace_back(tokens.begin(), tokens.end());

            BatchedDecodeRequestState state;
            state.last_token = tokens.back();
            state.prefill_logits_ready = true;
            next_states.push_back(std::move(state));
        }

        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        mtp_stats_ = {};
        prefix_request_summary_ = {};
        ready_sampled_token_.reset();
        ready_sampled_params_.reset();
        prefill_logits_ready_ = false;
        last_token_ = next_states.front().last_token;

        if (!runner_->forward_batch(converted))
        {
            clearBatchedDecodeState();
            return setError("Forward batch failed during request-batched prefill");
        }

        /*
         * From this point onward the scalar decode state is intentionally
         * invalid. decodeStepBatch() is the only API allowed to consume this
         * request set, because it must advance and publish every request slot
         * under the same ownership transaction.
         */
        batched_request_states_ = std::move(next_states);
        batched_decode_active_ = true;
        return true;
    }

    bool OrchestrationRunner::shouldUseMTPDecode() const
    {
        const MTPRuntimeConfig &mtp = plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        return mtp.enabled &&
               mtpDecodeHardFailureReason().empty() &&
               mtpDecodeBypassReason().empty();
    }

    std::string OrchestrationRunner::mtpDecodeHardFailureReason() const
    {
        const MTPRuntimeConfig &mtp = plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        if (!mtp.enabled || !runner_)
            return {};

        const int effective_max_draft_tokens = effectiveMTPMaxDraftDepth(mtp);
        if (effective_max_draft_tokens < 1 || effective_max_draft_tokens > 3)
        {
            return "MTP decode supports --mtp-draft-tokens in the range [1, 3] for verifier M=2..4";
        }
        if (mtp.max_request_batch != 1)
        {
            return "MTP request-batched speculative transactions are not executable yet; use --mtp-max-request-batch 1 until Phase 8 runner batching lands";
        }
        if (effective_max_draft_tokens > 1 && !runner_->supportsChainedMTPDrafts())
        {
            return "MTP decode with --mtp-draft-tokens > 1 requires runner support for chained MTP sidecars";
        }
        /*
         * The adaptive controller is intentionally owned by this
         * OrchestrationRunner, not by child device runners. LocalTP and LocalPP
         * are safe because one in-process runner chooses a single depth and
         * then fans out that same request to every participant or final-stage
         * sidecar. Multi-process domains add their own scalar coordination.
         */
        if (mtp.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            !active_sampling_params_.is_greedy() &&
            (plan_.usesLocalTP() ||
             plan_.usesGlobalTP() ||
             (mpi_ctx_ && mpi_ctx_->world_size() > 1)))
        {
            return "MTP speculative sampling verification is currently implemented only for SingleDevice and LocalPP full-logit execution";
        }
        if (runner_->primaryDeviceId().is_rocm() && debugEnv().rocm.concurrent_decode)
        {
            return "ROCm MTP decode is incompatible with LLAMINAR_ROCM_CONCURRENT_DECODE; use LLAMINAR_ROCM_CONCURRENT_M2_ROWS for M=2 verifier experiments";
        }
        if (runner_->primaryDeviceId().is_rocm() &&
            debugEnv().execution.gpu_graphs &&
            debugEnv().rocm.concurrent_m2_rows)
        {
            return "ROCm MTP decode is incompatible with LLAMINAR_ROCM_CONCURRENT_M2_ROWS when LLAMINAR_GPU_GRAPHS=1; M=2 row-overlap launches side streams that are not graph-capture safe";
        }
        if (debugEnv().execution.gpu_graphs &&
            debugEnv().execution.gpu_graph_collective_segmented &&
            plan_.usesLocalTP())
        {
            const bool has_rocm_participant =
                std::any_of(plan_.local_tp_devices.begin(), plan_.local_tp_devices.end(),
                            [](const GlobalDeviceAddress &address)
                            {
                                return address.toLocalDeviceId().is_rocm();
                            });
            if (has_rocm_participant)
            {
                return "ROCm LocalTP MTP decode is incompatible with LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED; RCCL segmented collective replay for MTP sidecar execution is not implemented";
            }
        }

        return {};
    }

    std::string OrchestrationRunner::mtpDecodeBypassReason() const
    {
        const MTPRuntimeConfig &mtp = plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        if (!mtp.enabled)
        {
            return "feature disabled";
        }
        if (!runner_)
        {
            return "runner unavailable";
        }
        if (!active_sampling_params_.is_greedy() &&
            mtp.verify_mode != MTPVerifyMode::SpeculativeSampling)
        {
            return "sampling is not greedy";
        }
        const std::string runner_reason = runner_->mtpDecodeUnsupportedReason();
        if (!runner_reason.empty())
        {
            return runner_reason;
        }
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1 &&
            !runner_->supportsMTPTokenCoordination())
        {
            return "MTP decode is not enabled for MPI world_size > 1";
        }
        return {};
    }

    void OrchestrationRunner::recordMTPBypass(const std::string &reason)
    {
        if (reason.empty() || reason == "feature disabled")
        {
            return;
        }
        mtp_bypassed_ = true;
        mtp_bypass_reason_ = reason;
        if (!mtp_bypass_recorded_for_request_)
        {
            ++mtp_stats_.bypasses;
            mtp_bypass_recorded_for_request_ = true;
            LOG_DEBUG("[OrchestrationRunner] MTP bypassed: " << reason);
        }
    }

    int OrchestrationRunner::effectiveMTPMaxDraftDepth(const MTPRuntimeConfig &mtp) const
    {
        if (mtp.depth_policy.mode == MTPDepthPolicyMode::Fixed)
        {
            return mtp.draft_tokens;
        }
        return mtp.depth_policy.max_depth > 0 ? mtp.depth_policy.max_depth : mtp.draft_tokens;
    }

    bool OrchestrationRunner::ensureMTPDepthController(const MTPRuntimeConfig &mtp)
    {
        try
        {
            if (!mtp_depth_controller_)
            {
                MTPDepthPolicyConfig depth_policy = mtp.depth_policy;
                const DeviceId primary_device = runner_->primaryDeviceId();
                if (primary_device.is_cuda())
                    depth_policy.backend = MTPDepthPolicyBackend::CUDA;
                else if (primary_device.is_rocm())
                    depth_policy.backend = MTPDepthPolicyBackend::ROCm;
                else if (primary_device.is_cpu())
                    depth_policy.backend = MTPDepthPolicyBackend::CPU;
                else
                    depth_policy.backend = MTPDepthPolicyBackend::Any;
                depth_policy.model_class =
                    inferMTPDepthPolicyModelClass(model_ctx_);

                mtp_depth_controller_ =
                    std::make_unique<MTPDepthController>(
                        depth_policy,
                        mtp.draft_tokens,
                        mtp.verify_mode);
            }
            return true;
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Invalid MTP depth policy: ") + e.what());
        }
    }

    int OrchestrationRunner::currentMTPDraftDepth(const MTPRuntimeConfig &mtp)
    {
        if (!ensureMTPDepthController(mtp) || !mtp_depth_controller_)
        {
            return std::max(1, mtp.draft_tokens);
        }

        int depth = mtp_depth_controller_->requestedDepthForStep();

        /*
         * Dynamic depth is a request-level scheduling decision.  In NodeLocalTP /
         * GlobalTP every rank must execute the same sidecar/verifier shape in the
         * same order, so rank 0's controller decision is treated as the scalar
         * source of truth and broadcast before the step begins.  This mirrors the
         * vLLM-style contract: the speculative batch shape is coordinated once,
         * while tensor data still moves through the graph and collective layers.
         */
        if (mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic &&
            mpi_ctx_ &&
            mpi_ctx_->world_size() > 1)
        {
            int32_t coordinated_depth =
                mpi_ctx_->rank() == 0 ? static_cast<int32_t>(depth) : 0;
            mpi_ctx_->broadcast_int32(&coordinated_depth, 1, 0);
            depth = static_cast<int>(coordinated_depth);

            PerfStatsCollector::addCounter(
                "mtp",
                "depth_policy_mpi_depth_broadcasts",
                1.0,
                "decode",
                {},
                {{"depth", std::to_string(depth)},
                 {"rank", std::to_string(mpi_ctx_->rank())},
                 {"world_size", std::to_string(mpi_ctx_->world_size())}});
        }

        return depth;
    }

    void OrchestrationRunner::recordMTPDepthZeroBypass()
    {
        if (!mtp_depth_controller_)
        {
            return;
        }
        const MTPDepthDecision decision = mtp_depth_controller_->recordBypassStep();
        mtp_stats_.current_depth = mtp_depth_controller_->currentDepth();
        mtp_stats_.min_depth = mtp_depth_controller_->minDepth();
        mtp_stats_.max_depth = mtp_depth_controller_->maxDepth();

        PerfStatsCollector::addCounter(
            "mtp",
            "depth_policy_zero_depth_bypasses",
            1.0,
            "decode",
            {},
            {{"current_depth", std::to_string(decision.new_depth)},
             {"next_requested_depth", std::to_string(mtp_depth_controller_->requestedDepthForStep())},
             {"reason", toString(decision.reason)}});
    }

    void OrchestrationRunner::recordMTPDepthObservation(
        int requested_depth,
        int effective_depth,
        int accepted_speculative_prefix,
        bool budget_limited,
        bool rollback)
    {
        if (!mtp_depth_controller_)
        {
            return;
        }
        const auto before = mtp_depth_controller_->stats();
        const MTPDepthDecision decision = mtp_depth_controller_->recordStep(
            MTPDepthObservation{
                .requested_depth = requested_depth,
                .effective_depth = effective_depth,
                .accepted_speculative_prefix = accepted_speculative_prefix,
                .budget_limited = budget_limited,
                .rollback = rollback,
            });
        const auto after = mtp_depth_controller_->stats();

        mtp_stats_.depth_policy_windows += after.windows - before.windows;
        mtp_stats_.depth_policy_updates += after.updates - before.updates;
        mtp_stats_.depth_policy_promotions += after.promotions - before.promotions;
        mtp_stats_.depth_policy_demotions += after.demotions - before.demotions;
        mtp_stats_.depth_policy_observe_recommendations +=
            after.observe_recommendations - before.observe_recommendations;
        mtp_stats_.current_depth = mtp_depth_controller_->currentDepth();
        mtp_stats_.min_depth = mtp_depth_controller_->minDepth();
        mtp_stats_.max_depth = mtp_depth_controller_->maxDepth();

        if (decision.evaluated)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "depth_policy_windows",
                1.0,
                "decode",
                {},
                {{"old_depth", std::to_string(decision.old_depth)},
                 {"new_depth", std::to_string(decision.new_depth)},
                 {"recommended_depth", std::to_string(decision.recommended_depth)},
                 {"reason", toString(decision.reason)},
                 {"changed", decision.changed ? "true" : "false"},
                 {"observe_recommendation", decision.observe_recommendation ? "true" : "false"},
                 {"acceptance_rate", std::to_string(decision.acceptance_rate)},
                 {"zero_accept_rate", std::to_string(decision.zero_accept_rate)},
                 {"full_accept_rate", std::to_string(decision.full_accept_rate)},
                 {"window_size", std::to_string(decision.window.verifier_runs)}});
        }
        if (decision.changed)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                decision.new_depth > decision.old_depth
                    ? "depth_policy_promotions"
                    : "depth_policy_demotions",
                1.0,
                "decode",
                {},
                {{"old_depth", std::to_string(decision.old_depth)},
                 {"new_depth", std::to_string(decision.new_depth)},
                 {"reason", toString(decision.reason)}});
        }
        else if (decision.observe_recommendation)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "depth_policy_observe_recommendations",
                1.0,
                "decode",
                {},
                {{"current_depth", std::to_string(decision.old_depth)},
                 {"recommended_depth", std::to_string(decision.recommended_depth)},
                 {"reason", toString(decision.reason)}});
        }
    }

    GenerationResult OrchestrationRunner::decodeStepMTP()
    {
        PerfStatsCollector::ScopedTimer step_timer("mtp", "decode_step_total", "decode");
        PerfStatsCollector::addCounter("mtp", "decode_step_calls", 1.0, "decode");

        GenerationResult result;
        const int vocab = vocabSize();
        if (vocab <= 0)
        {
            result.error = "Invalid vocabulary size for MTP decode";
            return result;
        }

        PrefixStateSnapshot checkpoint;
        {
            PerfStatsCollector::ScopedTimer timer("mtp", "capture_live_prefix_state", "decode");
            checkpoint = runner_->captureLivePrefixCheckpoint();
        }
        if (!checkpoint.valid)
        {
            PerfStatsCollector::addCounter("mtp", "capture_live_prefix_state_failures", 1.0, "decode");
            result.error = "MTP decode could not capture live prefix state";
            return result;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            checkpoint.logical_checkpoint ? "live_prefix_checkpoint_logical" : "live_prefix_checkpoint_payload",
            1.0,
            "decode");

        const bool use_ready_logits = prefill_logits_ready_;
        const std::optional<int32_t> ready_sampled_token = ready_sampled_token_;
        const std::optional<SamplingParams> ready_sampled_params = ready_sampled_params_;
        prefill_logits_ready_ = false;
        ready_sampled_token_.reset();
        ready_sampled_params_.reset();
        PrefixStateSnapshot verifier_base_checkpoint = checkpoint;

        auto fail_after_checkpoint = [&](const std::string &message) -> GenerationResult
        {
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "disable_all_position_logits_after_failure", "decode");
                runner_->setComputeAllPositionLogits(false);
                runner_->setComputeRowIndexedAllPositionLogits(false, 0);
            }
            bool restored = false;
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "restore_live_prefix_state_after_failure", "decode");
                restored = runner_->restoreLivePrefixState(checkpoint);
            }
            if (restored)
            {
                ++mtp_stats_.rollbacks;
                ++mtp_stats_.transaction_rollbacks;
                PerfStatsCollector::addCounter("mtp", "rollbacks", 1.0, "decode");
                PerfStatsCollector::addCounter("mtp", "transaction_rollbacks", 1.0, "decode");
            }
            PerfStatsCollector::addCounter("mtp", "decode_step_failures", 1.0, "decode",
                                           std::string{}, {{"reason", message}});
            prefill_logits_ready_ = use_ready_logits;
            ready_sampled_token_ = ready_sampled_token;
            ready_sampled_params_ = ready_sampled_params;
            result.error = message;
            return result;
        };

        if (use_ready_logits && ready_sampled_token.has_value())
        {
            /*
             * A ready verifier token is sampled one decode step before it is
             * consumed. Treat it as part of the atomic MTP transaction: if the
             * active sampling contract changed, consuming the cached token would
             * silently mix two sampling regimes in one request.
             */
            if (!ready_sampled_params.has_value())
            {
                return fail_after_checkpoint(
                    "Ready MTP verifier token is missing the sampling parameters that produced it");
            }
            if (!samplingParamsEqual(*ready_sampled_params, active_sampling_params_))
            {
                return fail_after_checkpoint(
                    "Ready MTP verifier token was sampled with different sampling parameters");
            }
        }

        const MTPRuntimeConfig &mtp = plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        const bool stochastic_verify =
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            !active_sampling_params_.is_greedy();
        const bool stochastic_device_verify =
            stochastic_verify &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsDeviceStochasticMTPVerification();
        const bool stochastic_host_verify =
            stochastic_verify &&
            !runner_->primaryDeviceId().is_gpu();
        const bool use_sampling_penalties =
            active_sampling_params_.has_penalties() && !stochastic_verify;
        const bool supports_all_position_state_publication =
            runner_->supportsMTPSpecStatePublication() &&
            (!stochastic_verify || stochastic_device_verify || stochastic_host_verify);
        const MTPVerifierPolicyDecision verifier_policy =
            chooseMTPVerifierPolicy(
                MTPVerifierPolicyInput{
                    .greedy_sampling = active_sampling_params_.is_greedy(),
                    .stochastic_verify = stochastic_verify,
                    .uses_sampling_penalties = use_sampling_penalties,
                    .supports_spec_state_publication =
                        supports_all_position_state_publication,
                });
        if (verifier_policy.path == MTPVerifierExecutionPath::Unsupported)
        {
            return fail_after_checkpoint(
                std::string("MTP verifier policy selected unsupported path: ") +
                verifier_policy.reason);
        }
        const bool use_all_position_state_publication_verifier =
            verifier_policy.path ==
            MTPVerifierExecutionPath::AllPositionStatePublication;
        const bool verify_sidecar_preserves_main_state =
            DebugEnv::isTruthyEnv("LLAMINAR_MTP_VERIFY_SIDECAR_PRESERVES_MAIN_STATE");
        const bool verify_commit_replay_check =
            DebugEnv::isTruthyEnv("LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_CHECK") &&
            !stochastic_verify &&
            !use_sampling_penalties &&
            active_sampling_params_.is_greedy();
        const bool can_synthesize_verifier_base_checkpoint =
            use_all_position_state_publication_verifier &&
            runner_->supportsMTPSidecarPreservesMainState() &&
            !verify_sidecar_preserves_main_state &&
            !verify_commit_replay_check;

        auto join_tokens = [](const std::vector<int32_t> &tokens) -> std::string
        {
            std::ostringstream oss;
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (i)
                    oss << ",";
                oss << tokens[i];
            }
            return oss.str();
        };

        enum class StochasticDrawPurpose : int
        {
            Sample = 0,
            Accept = 1,
            Residual = 2,
        };

        auto stochastic_threshold_for_position = [&](
                                                     Sampler &fallback_sampler,
                                                     int logical_position,
                                                     StochasticDrawPurpose purpose)
            -> float
        {
            if (active_sampling_params_.seed == 0)
            {
                return fallback_sampler.random_uniform_01();
            }

            /*
             * Seeded MTP sampling must not depend on when a token is sampled.
             * A ready token may be sampled as a bonus row in step N or as the
             * first token of step N+1.  Keying the threshold by logical output
             * position and purpose makes those two paths equivalent.
             */
            const uint64_t position =
                static_cast<uint64_t>(std::max(0, logical_position));
            constexpr uint64_t kDrawPurposesPerToken = 8;
            const uint64_t offset =
                position * kDrawPurposesPerToken +
                static_cast<uint64_t>(purpose);
            return sampling_math::uniform01(
                static_cast<uint64_t>(active_sampling_params_.seed),
                offset);
        };

        auto sample_threshold_for_position =
            [&](Sampler &fallback_sampler, int logical_position) -> float
        {
            return stochastic_threshold_for_position(
                fallback_sampler,
                logical_position,
                StochasticDrawPurpose::Sample);
        };

        auto format_stochastic_threshold = [](float threshold) -> std::string
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(9) << threshold;
            return oss.str();
        };

        auto accept_threshold_for_position =
            [&](Sampler &fallback_sampler, int logical_position) -> float
        {
            return stochastic_threshold_for_position(
                fallback_sampler,
                logical_position,
                StochasticDrawPurpose::Accept);
        };

        auto residual_threshold_for_position =
            [&](Sampler &fallback_sampler, int logical_position) -> float
        {
            return stochastic_threshold_for_position(
                fallback_sampler,
                logical_position,
                StochasticDrawPurpose::Residual);
        };

        auto validate_mtp_transaction = [&](
                                            const char *path,
                                            const PrefixStateSnapshot &base,
                                            int emitted_tokens,
                                            int state_advanced_tokens,
                                            PrefixStateProvenance verifier_source,
                                            bool has_terminal_logits,
                                            bool has_ready_token)
            -> std::optional<std::string>
        {
            if (!base.valid)
                return std::string("MTP transaction base snapshot is invalid");
            if (state_advanced_tokens < 0 ||
                state_advanced_tokens > emitted_tokens)
            {
                return std::string("MTP transaction advanced-state token count is outside emitted output count");
            }

            MTPCommitValidationOptions options;
            options.require_decode_equivalent_source = true;
            options.require_base_shifted_mtp_kv = false;
            options.require_committed_shifted_mtp_kv = true;
            options.require_terminal_hidden = true;
            options.require_terminal_logits = has_terminal_logits;
            options.require_ready_token = has_ready_token;

            MTPDecodeStateStamp base_stamp = makeMTPStateStamp(
                base,
                std::string(path) + ".base",
                /*has_terminal_hidden=*/true,
                /*has_terminal_logits=*/true,
                /*has_ready_token=*/true);

            MTPDecodeStateStamp committed_stamp;
            committed_stamp.valid = base.valid;
            committed_stamp.logical_tokens =
                base.cached_tokens + state_advanced_tokens;
            committed_stamp.main_kv_tokens = committed_stamp.logical_tokens;
            committed_stamp.shifted_mtp_kv_tokens =
                expectedShiftedMTPTokens(committed_stamp.logical_tokens);
            committed_stamp.position = committed_stamp.logical_tokens;
            committed_stamp.has_terminal_hidden = true;
            committed_stamp.has_terminal_logits = has_terminal_logits;
            committed_stamp.has_ready_token = has_ready_token;
            committed_stamp.provenance = verifier_source;
            committed_stamp.label = std::string(path) + ".committed";

            MTPStateValidationResult validation = validateAtomicMTPCommit(
                base_stamp,
                committed_stamp,
                state_advanced_tokens,
                verifier_source,
                options);
            if (!validation)
            {
                ++mtp_stats_.transaction_validation_failures;
                if (!base_stamp.decodeEquivalent() ||
                    !committed_stamp.decodeEquivalent() ||
                    !isDecodeEquivalent(verifier_source))
                {
                    ++mtp_stats_.unsafe_verifier_state_rejections;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "unsafe_verifier_state_rejections",
                        1.0,
                        "decode",
                        {},
                        {{"path", path},
                         {"source", toString(verifier_source)}});
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "transaction_validation_failures",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"reason", validation.reason},
                     {"source", toString(verifier_source)}});
                return std::string("MTP transaction validation failed on ") +
                       path + ": " + validation.reason;
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "transaction_validation_passes",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"emitted_tokens", std::to_string(emitted_tokens)},
                 {"state_advanced_tokens", std::to_string(state_advanced_tokens)},
                 {"source", toString(verifier_source)}});
            return std::nullopt;
        };

        auto commit_mtp_transaction_outputs = [&](
                                                  const char *path,
                                                  const PrefixStateSnapshot &base,
                                                  const std::vector<int32_t> &tokens,
                                                  std::optional<int32_t> ready_token,
                                                  bool terminal_logits_ready,
                                                  bool is_complete,
                                                  PrefixStateProvenance verifier_source,
                                                  bool state_advanced,
                                                  int state_advanced_token_count = -1)
            -> std::optional<std::string>
        {
            if (tokens.empty())
                return std::string("MTP transaction produced no output tokens");

            if (state_advanced)
            {
                const int advanced_tokens =
                    state_advanced_token_count >= 0
                        ? state_advanced_token_count
                        : static_cast<int>(tokens.size());
                if (auto validation_error = validate_mtp_transaction(
                        path,
                        base,
                        static_cast<int>(tokens.size()),
                        advanced_tokens,
                        verifier_source,
                        terminal_logits_ready && !is_complete,
                        ready_token.has_value() && !is_complete))
                {
                    return validation_error;
                }
            }

            prefill_logits_ready_ = terminal_logits_ready && !is_complete;
            if (prefill_logits_ready_ && ready_token.has_value())
            {
                ready_sampled_token_ = *ready_token;
                ready_sampled_params_ = active_sampling_params_;
            }
            else
            {
                ready_sampled_token_.reset();
                ready_sampled_params_.reset();
            }

            for (int32_t token : tokens)
            {
                sampler_.record_token(token);
                result.tokens.push_back(token);
            }
            last_token_ = tokens.back();
            result.is_complete = result.is_complete || is_complete;

            ++mtp_stats_.transaction_commits;
            PerfStatsCollector::addCounter(
                "mtp",
                "transaction_commits",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"tokens", join_tokens(tokens)},
                 {"ready_token", ready_token.has_value()
                                     ? std::to_string(*ready_token)
                                     : std::string("none")},
                 {"state_advanced", state_advanced ? "true" : "false"},
                 {"state_advanced_tokens",
                  std::to_string(state_advanced
                                     ? (state_advanced_token_count >= 0
                                            ? state_advanced_token_count
                                            : static_cast<int>(tokens.size()))
                                     : 0)},
                 {"complete", is_complete ? "true" : "false"},
                 {"source", toString(verifier_source)}});
            return std::nullopt;
        };

        auto validate_spec_decode_transaction = [&](
                                                        const char *path,
                                                        const std::string &implementation,
                                                        const std::vector<int32_t> &draft_tokens_for_tx,
                                                        const std::vector<int32_t> &committed_output_tokens,
                                                        std::optional<int32_t> ready_token,
                                                        bool all_drafts_accepted,
                                                        bool stopped_on_output,
                                                        int accepted_mtp_draft_prefix)
            -> std::optional<std::string>
        {
            if (draft_tokens_for_tx.empty())
                return std::string("MTP spec-decode transaction has no draft tokens");
            if (committed_output_tokens.empty())
                return std::string("MTP spec-decode transaction has no committed output tokens");
            if (!stopped_on_output && all_drafts_accepted && !ready_token.has_value())
                return std::string("MTP spec-decode transaction accepted all drafts without a ready token");

            MTPSpecDecodeMetadataShape metadata_shape;
            metadata_shape.max_requests = 1;
            metadata_shape.max_draft_tokens =
                static_cast<int>(draft_tokens_for_tx.size());

            MTPDecodeCatchupGreedyRequest catchup_request_for_tx;
            catchup_request_for_tx.draft_tokens = draft_tokens_for_tx;
            MTPDecodeCatchupGreedyResult catchup_result_for_tx;
            catchup_result_for_tx.ok = true;
            catchup_result_for_tx.accepted_tokens = committed_output_tokens;
            catchup_result_for_tx.all_speculative_accepted = all_drafts_accepted;
            catchup_result_for_tx.stopped_on_output = stopped_on_output;
            catchup_result_for_tx.accepted_speculative_prefix =
                accepted_mtp_draft_prefix;
            catchup_result_for_tx.ready_token =
                ready_token.value_or(kMTPSpecDecodeInvalidToken);

            MTPSpecDecodeMetadataBatch metadata =
                buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
                    metadata_shape,
                    /*request_id=*/0,
                    vocab,
                    catchup_request_for_tx,
                    catchup_result_for_tx);
            if (!metadata.ok)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "spec_decode_transaction_metadata_failures",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"implementation", implementation},
                     {"reason", metadata.error}});
                return std::string("MTP spec-decode metadata batch failed on ") +
                       path + ": " + metadata.error;
            }
            if (metadata.transactions.empty())
                return std::string("MTP spec-decode metadata batch produced no transaction");

            const MTPSpecDecodeTransaction &tx = metadata.transactions.front();
            if (!tx.ok)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "spec_decode_transaction_metadata_failures",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"implementation", implementation},
                     {"reason", tx.error}});
                return std::string("MTP spec-decode transaction metadata failed on ") +
                       path + ": " + tx.error;
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "spec_decode_transaction_metadata",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"implementation", implementation},
                 {"target_query_len", std::to_string(tx.target_query_len)},
                 {"metadata_total_target_query_tokens",
                  std::to_string(metadata.total_target_query_tokens)},
                 {"valid_sampled_count", std::to_string(tx.valid_sampled_count)},
                 {"committed_output_count",
                  std::to_string(metadata.committed_output_counts.front())},
                 {"accepted_state_count",
                  std::to_string(metadata.accepted_state_counts.front())},
                 {"committed_state_row",
                  std::to_string(metadata.committed_state_rows.front())},
                 {"committed_state_index",
                  std::to_string(metadata.committed_state_indices.front())},
                 {"accepted_state_slot_index",
                  std::to_string(metadata.accepted_state_slot_indices.front())},
                 {"bonus_ready_token_row",
                  std::to_string(metadata.bonus_ready_token_rows.front())},
                 {"bonus_ready_token_index",
                  std::to_string(metadata.bonus_ready_token_indices.front())},
                 {"bonus_ready_state_slot_index",
                  std::to_string(metadata.bonus_ready_state_slot_indices.front())},
                 {"accepted_verifier_input_prefix",
                  std::to_string(tx.accepted_speculative_prefix)},
                 {"accepted_mtp_draft_prefix",
                  std::to_string(std::max(0, tx.accepted_speculative_prefix - 1))},
                 {"rejected_token_count", std::to_string(tx.rejected_token_count)},
                 {"token_index_to_sample", std::to_string(tx.token_index_to_sample)},
                 {"next_condition_token", std::to_string(tx.next_condition_token)},
                 {"all_drafts_accepted", tx.allDraftsAccepted() ? "true" : "false"},
                 {"stopped_on_output", stopped_on_output ? "true" : "false"},
                 {"draft_tokens", join_tokens(draft_tokens_for_tx)},
                 {"committed_output_tokens", join_tokens(committed_output_tokens)}});
            return std::nullopt;
        };

        auto validate_spec_decode_accepted_outcome = [&](
                                                        const char *path,
                                                        const std::string &implementation,
                                                        const MTPSpecDecodeAcceptedOutcome &outcome)
            -> std::optional<std::string>
        {
            if (outcome.draft_count <= 0)
                return std::string("MTP spec-decode accepted outcome has no draft rows");
            if (outcome.committed_output_tokens.empty())
                return std::string("MTP spec-decode accepted outcome has no committed output tokens");
            if (!outcome.stopped_on_output &&
                outcome.all_drafts_accepted &&
                !outcome.bonus_ready_token.has_value())
            {
                return std::string("MTP spec-decode accepted outcome accepted all drafts without a ready token");
            }

            MTPSpecDecodeMetadataShape metadata_shape;
            metadata_shape.max_requests = 1;
            metadata_shape.max_draft_tokens = outcome.draft_count;

            MTPSpecDecodeMetadataBatch metadata =
                buildMTPSpecDecodeMetadataBatchFromAcceptedOutcome(
                    metadata_shape,
                    outcome);
            if (!metadata.ok)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "spec_decode_transaction_metadata_failures",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"implementation", implementation},
                     {"reason", metadata.error}});
                return std::string("MTP spec-decode accepted-outcome metadata failed on ") +
                       path + ": " + metadata.error;
            }
            if (metadata.transactions.empty())
                return std::string("MTP spec-decode accepted-outcome metadata produced no transaction");

            const MTPSpecDecodeTransaction &tx = metadata.transactions.front();
            if (!tx.ok)
                return std::string("MTP spec-decode accepted-outcome transaction is invalid: ") + tx.error;

            PerfStatsCollector::addCounter(
                "mtp",
                "spec_decode_transaction_metadata",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"implementation", implementation},
                 {"target_query_len", std::to_string(tx.target_query_len)},
                 {"metadata_total_target_query_tokens",
                  std::to_string(metadata.total_target_query_tokens)},
                 {"valid_sampled_count", std::to_string(tx.valid_sampled_count)},
                 {"committed_output_count",
                  std::to_string(metadata.committed_output_counts.front())},
                 {"accepted_state_count",
                  std::to_string(metadata.accepted_state_counts.front())},
                 {"committed_state_row",
                  std::to_string(metadata.committed_state_rows.front())},
                 {"committed_state_index",
                  std::to_string(metadata.committed_state_indices.front())},
                 {"accepted_state_slot_index",
                  std::to_string(metadata.accepted_state_slot_indices.front())},
                 {"bonus_ready_token_row",
                  std::to_string(metadata.bonus_ready_token_rows.front())},
                 {"bonus_ready_token_index",
                  std::to_string(metadata.bonus_ready_token_indices.front())},
                 {"bonus_ready_state_slot_index",
                  std::to_string(metadata.bonus_ready_state_slot_indices.front())},
                 {"accepted_verifier_input_prefix",
                  std::to_string(tx.accepted_speculative_prefix)},
                 {"accepted_mtp_draft_prefix",
                  std::to_string(std::max(0, tx.accepted_speculative_prefix - 1))},
                 {"rejected_token_count", std::to_string(tx.rejected_token_count)},
                 {"token_index_to_sample", std::to_string(tx.token_index_to_sample)},
                 {"next_condition_token", std::to_string(tx.next_condition_token)},
                 {"all_drafts_accepted", tx.allDraftsAccepted() ? "true" : "false"},
                 {"stopped_on_output", outcome.stopped_on_output ? "true" : "false"},
                 {"draft_tokens", std::string("device_deferred:") +
                                      std::to_string(outcome.draft_count)},
                 {"committed_output_tokens",
                  join_tokens(outcome.committed_output_tokens)}});
            return std::nullopt;
        };

        const bool can_defer_main_decode_sync =
            runner_->primaryDeviceId().is_gpu() &&
            !active_sampling_params_.has_penalties() &&
            (active_sampling_params_.is_greedy() || stochastic_device_verify);

        const int32_t condition_token = last_token_;
        if (!use_ready_logits)
        {
            bool ok = false;
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "condition_forward", "decode");
                /*
                 * The condition forward's logits are consumed immediately by a
                 * GPU sampler or distribution-builder.  Arm a one-shot stream
                 * handoff so graph replay can skip the CPU sync boundary and
                * let that consumer enforce ordering on the same stream.
                */
                runner_->setMTPMainDecodeSyncDeferralEnabled(can_defer_main_decode_sync);
                ok = runner_->forward(&condition_token, 1);
                if (!ok)
                {
                    runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                }
            }
            if (!ok)
                return fail_after_checkpoint("Forward pass failed during MTP condition decode");
            if (can_synthesize_verifier_base_checkpoint)
            {
                verifier_base_checkpoint =
                    makeLogicalMTPVerifierBaseSnapshot(runner_->get_position());
                PerfStatsCollector::addCounter(
                    "mtp",
                    "capture_verifier_base_prefix_state_skipped_all_position_publication",
                    1.0,
                    "decode",
                    {},
                    {{"cached_tokens", std::to_string(verifier_base_checkpoint.cached_tokens)}});
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "capture_verifier_base_prefix_state",
                    "decode");
                verifier_base_checkpoint = runner_->captureLivePrefixCheckpoint();
            }
            if (!verifier_base_checkpoint.valid)
            {
                return fail_after_checkpoint(
                    "MTP decode could not capture verifier base state after condition forward");
            }
        }
        else if (use_ready_logits)
        {
            PerfStatsCollector::addCounter("mtp", "condition_forward_skipped_ready_logits", 1.0, "decode");
        }

        const int requested_speculative_draft_count = currentMTPDraftDepth(mtp);
        const int pre_sample_effective_draft_count =
            decode_step_token_budget_ > 0
                ? std::min(
                      requested_speculative_draft_count,
                      std::max(0, decode_step_token_budget_ - 1))
                : requested_speculative_draft_count;
        constexpr int32_t kDeferredMTPFirstTokenShadow = -3;
        const bool can_defer_stochastic_first_host_read =
            stochastic_device_verify &&
            use_all_position_state_publication_verifier &&
            runner_->primaryDeviceId().is_gpu() &&
            !active_sampling_params_.has_penalties() &&
            pre_sample_effective_draft_count > 0 &&
            stop_tokens_.size() <=
                static_cast<size_t>(
                    sampling_math::kSpeculativeBatchMaxStopTokens) &&
            runner_->supportsMTPDeviceDraftTokenInput();

        int32_t first_token = -1;
        if (use_ready_logits && ready_sampled_token.has_value())
        {
            first_token = *ready_sampled_token;
            PerfStatsCollector::addCounter("mtp", "first_token_ready_cache_hits", 1.0, "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_ready_cache_token",
                1.0,
                "decode",
                {},
                {{"token", std::to_string(first_token)}});
        }
        else
        {
            if (stochastic_verify)
            {
                if (runner_->primaryDeviceId().is_gpu())
                {
                    if (active_sampling_params_.top_k <= 0 ||
                        active_sampling_params_.top_k > 256)
                    {
                        return fail_after_checkpoint(
                            "GPU stochastic MTP sampling requires 1 <= top_k <= 256");
                    }
                    if (!stochastic_device_verify)
                    {
                        return fail_after_checkpoint(
                            "GPU stochastic MTP requires device-resident distribution verification");
                    }
                    auto penalty_map = sampler_.compute_penalty_map(active_sampling_params_, vocab);
                    if (!penalty_map.empty() &&
                        !runner_->applyPenaltiesOnDevice(penalty_map, vocab))
                    {
                        return fail_after_checkpoint("MTP stochastic first-token GPU penalty application failed");
                    }
                    {
                        PerfStatsCollector::ScopedTimer timer(
                            "mtp",
                            "sample_first_token_stochastic_device",
                            "decode");
                        const int first_token_logical_position =
                            checkpoint.cached_tokens;
                        const float first_token_threshold =
                            sample_threshold_for_position(
                                sampler_,
                                first_token_logical_position);
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "first_token_stochastic_draw",
                            1.0,
                            "decode",
                            {},
                            {{"logical_position", std::to_string(first_token_logical_position)},
                             {"threshold", format_stochastic_threshold(first_token_threshold)},
                             {"deferred", can_defer_stochastic_first_host_read ? "true" : "false"}});
                        if (!runner_->buildStochasticDistributionOnDevice(
                                DeviceLogitsSource::Main,
                                0,
                                DeviceDistributionBuffer::Target,
                                0,
                                active_sampling_params_,
                                vocab))
                        {
                            return fail_after_checkpoint("MTP stochastic first-token GPU distribution build failed");
                        }
                        if (can_defer_stochastic_first_host_read)
                        {
                            if (!runner_->sampleStochasticDistributionOnDeviceDeferred(
                                    DeviceDistributionBuffer::Target,
                                    0,
                                    first_token_threshold))
                            {
                                return fail_after_checkpoint("MTP stochastic first-token GPU deferred sampling failed");
                            }
                            first_token = kDeferredMTPFirstTokenShadow;
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "first_token_stochastic_deferred_host_reads",
                                1.0,
                                "decode");
                        }
                        else
                        {
                            first_token = runner_->sampleStochasticDistributionOnDevice(
                                DeviceDistributionBuffer::Target,
                                0,
                                first_token_threshold);
                        }
                    }
                    if (first_token < 0 &&
                        first_token != kDeferredMTPFirstTokenShadow)
                    {
                        return fail_after_checkpoint("MTP stochastic first-token GPU sampling failed");
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "first_token_stochastic_device_samples",
                        1.0,
                        "decode");
                }
                else
                {
                    const float *main_logits = runner_->logits();
                    if (!main_logits)
                    {
                        return fail_after_checkpoint("No logits available for stochastic MTP first token");
                    }
                    {
                        PerfStatsCollector::ScopedTimer timer("mtp", "sample_first_token_stochastic", "decode");
                        first_token = sampler_.sample(
                            main_logits,
                            static_cast<size_t>(vocab),
                            active_sampling_params_);
                    }
                    PerfStatsCollector::addCounter("mtp", "first_token_stochastic_samples", 1.0, "decode");
                }
            }
            else
            {
                if (use_sampling_penalties)
                {
                    auto penalty_map = sampler_.compute_penalty_map(active_sampling_params_, vocab);
                    if (!runner_->applyPenaltiesOnDevice(penalty_map, vocab))
                    {
                        return fail_after_checkpoint("MTP first-token GPU penalty application failed");
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "first_token_gpu_penalty_applications",
                        1.0,
                        "decode");
                }
                {
                    PerfStatsCollector::ScopedTimer timer("mtp", "sample_first_token_device", "decode");
                    first_token = runner_->sampleGreedyOnDevice();
                }
                if (first_token < 0)
                {
                    if (use_sampling_penalties)
                    {
                        return fail_after_checkpoint("MTP first-token penalized GPU sampling failed");
                    }
                    PerfStatsCollector::addCounter("mtp", "first_token_host_sampling_fallbacks", 1.0, "decode");
                    const float *main_logits = runner_->logits();
                    if (!main_logits)
                    {
                        return fail_after_checkpoint("No logits available for MTP first draft token");
                    }
                    {
                        PerfStatsCollector::ScopedTimer timer("mtp", "sample_first_token_host", "decode");
                        first_token = sampler_.sample(
                            main_logits,
                            static_cast<size_t>(vocab),
                            active_sampling_params_);
                    }
                }
                else
                {
                    PerfStatsCollector::addCounter("mtp", "first_token_device_samples", 1.0, "decode");
                }
            }
        }

        int speculative_draft_count = requested_speculative_draft_count;
        bool draft_count_budget_limited = false;
        if (decode_step_token_budget_ > 0)
        {
            const int budgeted_speculative_outputs =
                std::max(0, decode_step_token_budget_ - 1);
            speculative_draft_count =
                std::min(speculative_draft_count, budgeted_speculative_outputs);
            draft_count_budget_limited =
                speculative_draft_count != requested_speculative_draft_count;
            if (draft_count_budget_limited)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "draft_steps_budget_clamped",
                    1.0,
                    "decode",
                    {},
                    {{"configured", std::to_string(requested_speculative_draft_count)},
                     {"effective", std::to_string(speculative_draft_count)},
                     {"token_budget", std::to_string(decode_step_token_budget_)}});
                PerfStatsCollector::addCounter(
                    "mtp",
                    "draft_steps_budget_skipped",
                    static_cast<double>(requested_speculative_draft_count - speculative_draft_count),
                    "decode");
            }
        }

        if (speculative_draft_count == 0)
        {
            PerfStatsCollector::addCounter("mtp", "budget_limited_direct_emits", 1.0, "decode");
            PerfStatsCollector::addCounter("mtp", "output_tokens", 1.0, "decode");

            const bool first_token_is_stop =
                std::find(stop_tokens_.begin(), stop_tokens_.end(), first_token) != stop_tokens_.end();
            if (!first_token_is_stop)
            {
                const int base_sidecar_position = runner_->get_position();
                bool shifted_commit_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "budget_limited_direct_emit_shifted_commit",
                        "decode");
                    shifted_commit_ok =
                        runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                            first_token,
                            /*already_appended_tokens=*/0,
                            /*allow_speculative_discard=*/true,
                            base_sidecar_position);
                }
                if (!shifted_commit_ok)
                {
                    return fail_after_checkpoint(
                        "MTP budget-limited direct emit shifted-cache commit failed");
                }

                bool advance_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "budget_limited_direct_emit_forward",
                        "decode");
                    /*
                     * Depth-zero MTP still advances the main graph so the
                     * next decode call can consume ready terminal logits.  On
                     * GPU, that next consumer is a device sampler or
                     * distribution builder, so publish the producer stream and
                     * let the consumer preserve ordering without a CPU sync.
                     */
                    runner_->setMTPMainDecodeSyncDeferralEnabled(
                        can_defer_main_decode_sync);
                    advance_ok = runner_->forward(&first_token, 1);
                    if (!advance_ok)
                    {
                        runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                    }
                }
                if (!advance_ok)
                {
                    return fail_after_checkpoint(
                        "MTP budget-limited direct emit state advance failed");
                }
            }

            if (first_token_is_stop)
            {
                if (auto commit_error = commit_mtp_transaction_outputs(
                        "budget_limited_direct_stop",
                        verifier_base_checkpoint,
                        std::vector<int32_t>{first_token},
                        std::nullopt,
                        /*terminal_logits_ready=*/false,
                        /*is_complete=*/true,
                        PrefixStateProvenance::DecodeEquivalent,
                        /*state_advanced=*/false))
                {
                    return fail_after_checkpoint(*commit_error);
                }
            }
            else
            {
                if (auto commit_error = commit_mtp_transaction_outputs(
                        "budget_limited_direct_emit",
                        verifier_base_checkpoint,
                        std::vector<int32_t>{first_token},
                        std::nullopt,
                        /*terminal_logits_ready=*/true,
                        /*is_complete=*/false,
                        PrefixStateProvenance::DecodeEquivalent,
                        /*state_advanced=*/true))
                {
                    return fail_after_checkpoint(*commit_error);
                }
            }
            return result;
        }

        std::optional<PrefixStateSnapshot> verifier_replay_base_checkpoint;
        if (verify_commit_replay_check)
        {
            verifier_replay_base_checkpoint = verifier_base_checkpoint;
        }

        const int base_sidecar_position = runner_->get_position();
        bool first_token_is_stop =
            first_token != kDeferredMTPFirstTokenShadow &&
            std::find(stop_tokens_.begin(), stop_tokens_.end(), first_token) != stop_tokens_.end();
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(static_cast<size_t>(speculative_draft_count) + 1);
        draft_tokens.push_back(first_token);
        Sampler draft_sampler = sampler_;
        if ((stochastic_verify || use_sampling_penalties) &&
            first_token != kDeferredMTPFirstTokenShadow)
        {
            draft_sampler.record_token(first_token);
        }

        std::vector<PrefixStateSnapshot> sidecar_checkpoints;
        sidecar_checkpoints.reserve(1);
        std::vector<std::vector<SamplingDistributionEntry>> host_mtp_draft_distributions(
            static_cast<size_t>(std::max(0, speculative_draft_count)));
        constexpr int32_t kDeferredMTPDraftTokenShadow = -2;

        auto sample_mtp_token = [&](int draft_idx, bool defer_host_read) -> int32_t
        {
            int32_t token = -1;
            if (stochastic_verify)
            {
                if (stochastic_device_verify)
                {
                    /*
                     * vLLM's default draft-sample mode is greedy: the draft
                     * side emits only a token, and target-side rejection treats
                     * q as a one-hot distribution. That avoids building full
                     * draft probability rows on every MTP sidecar step while
                     * preserving stochastic target correction semantics.
                     */
                    {
                        PerfStatsCollector::ScopedTimer timer("mtp", "sample_mtp_token_stochastic_device", "decode");
                        const float threshold =
                            sample_threshold_for_position(
                                draft_sampler,
                                checkpoint.cached_tokens + 1 + draft_idx);
                        if (defer_host_read)
                        {
                            if (!runner_->sampleStochasticDraftProposalOnDeviceDeferred(
                                    DeviceLogitsSource::MTP,
                                    0,
                                    draft_idx,
                                    active_sampling_params_,
                                    vocab,
                                    threshold))
                            {
                                return -1;
                            }
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "mtp_token_stochastic_deferred_host_reads",
                                1.0,
                                "decode",
                                {},
                                {{"draft_idx", std::to_string(draft_idx)}});
                            return kDeferredMTPDraftTokenShadow;
                        }
                        token = runner_->sampleStochasticDraftProposalOnDevice(
                            DeviceLogitsSource::MTP,
                            0,
                            draft_idx,
                            active_sampling_params_,
                            vocab,
                            threshold);
                    }
                    if (token < 0)
                    {
                        return -1;
                    }
                    PerfStatsCollector::addCounter("mtp", "mtp_token_stochastic_device_samples", 1.0, "decode");
                    return token;
                }

                const float *mtp_logits = runner_->mtpLogits();
                if (!mtp_logits)
                {
                    return -1;
                }
                auto distribution =
                    draft_sampler.compute_distribution(
                        mtp_logits,
                        static_cast<size_t>(vocab),
                        active_sampling_params_);
                if (draft_idx >= 0 &&
                    draft_idx < static_cast<int>(host_mtp_draft_distributions.size()))
                {
                    host_mtp_draft_distributions[static_cast<size_t>(draft_idx)] =
                        distribution;
                }
                {
                    PerfStatsCollector::ScopedTimer timer("mtp", "sample_mtp_token_stochastic", "decode");
                    token = draft_sampler.sample_from_distribution(distribution);
                }
                PerfStatsCollector::addCounter("mtp", "mtp_token_stochastic_samples", 1.0, "decode");
                return token;
            }

            if (use_sampling_penalties)
            {
                auto penalty_map =
                    draft_sampler.compute_penalty_map(active_sampling_params_, vocab);
                if (!runner_->applyPenaltiesToMTPLogitsOnDevice(penalty_map, vocab))
                {
                    return -1;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "mtp_token_gpu_penalty_applications",
                    1.0,
                    "decode");
            }
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "sample_mtp_token_device", "decode");
                token = runner_->sampleGreedyFromMTPLogitsOnDevice();
            }
            if (token >= 0)
            {
                PerfStatsCollector::addCounter("mtp", "mtp_token_device_samples", 1.0, "decode");
                return token;
            }

            if (use_sampling_penalties)
            {
                return -1;
            }

            PerfStatsCollector::addCounter("mtp", "mtp_token_host_sampling_fallbacks", 1.0, "decode");
            const float *mtp_logits = runner_->mtpLogits();
            if (!mtp_logits)
            {
                return -1;
            }
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "sample_mtp_token_host", "decode");
                token = sampler_.sample(
                    mtp_logits,
                    static_cast<size_t>(vocab),
                    active_sampling_params_);
            }
            return token;
        };

        const bool use_sidecar_sample_fusion =
            runner_->supportsMTPSidecarSampleFusion() && !use_sampling_penalties && !stochastic_verify;
        /*
         * Penalty-free stochastic MTP can hand sidecar logits directly to the
         * compact device distribution builder. This avoids the sync that used
         * to sit between sidecar replay and draft-token sampling. Penalty
         * paths remain synchronized because accepted-token history mutates the
         * logits before each sample.
         */
        const bool use_sidecar_stream_handoff_for_stochastic =
            stochastic_verify &&
            stochastic_device_verify &&
            runner_->primaryDeviceId().is_gpu() &&
            !active_sampling_params_.has_penalties() &&
            runner_->supportsMTPSidecarLogitsStreamHandoff();
        const bool use_device_draft_token_sidecar =
            use_sidecar_stream_handoff_for_stochastic &&
            runner_->supportsMTPDeviceDraftTokenInput();
        const bool can_defer_stochastic_draft_host_reads =
            use_sidecar_stream_handoff_for_stochastic &&
            use_all_position_state_publication_verifier &&
            !active_sampling_params_.has_penalties();
        for (int draft_idx = 0; draft_idx < speculative_draft_count; ++draft_idx)
        {
            bool sidecar_ok = false;
            int32_t mtp_token = -1;
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "sidecar_forward", "decode");
                if (draft_idx == 0)
                {
                    if (use_sidecar_sample_fusion)
                    {
                        sidecar_ok = runner_->forwardMTPAndSampleGreedy(
                            draft_tokens.back(),
                            &mtp_token);
                    }
                    else if (use_sidecar_stream_handoff_for_stochastic)
                    {
                        if (first_token == kDeferredMTPFirstTokenShadow)
                        {
                            sidecar_ok =
                                use_device_draft_token_sidecar &&
                                runner_->forwardMTPFromDeviceTargetForDeviceSampling(
                                    /*target_sample_slot=*/0,
                                    base_sidecar_position);
                            if (sidecar_ok)
                            {
                                PerfStatsCollector::addCounter(
                                    "mtp",
                                    "stochastic_first_sidecar_device_target_inputs",
                                    1.0,
                                    "decode");
                            }
                        }
                        else
                        {
                            sidecar_ok = runner_->forwardMTPForDeviceSampling(
                                draft_tokens.back());
                        }
                    }
                    else
                    {
                        sidecar_ok = runner_->forwardMTP(draft_tokens.back());
                    }
                }
                else
                {
                    if (use_sidecar_sample_fusion)
                    {
                        sidecar_ok = runner_->forwardMTPFromLastDraftAndSampleGreedy(
                            draft_tokens.back(),
                            base_sidecar_position + draft_idx,
                            &mtp_token);
                    }
                    else if (use_device_draft_token_sidecar)
                    {
                        /*
                         * The previous iteration sampled MTP draft token
                         * draft_idx - 1 into the runner-owned device slot with
                         * the same index. Feed that slot directly into the next
                         * sidecar embedding instead of uploading draft_tokens.back().
                         */
                        sidecar_ok =
                            runner_->forwardMTPFromDeviceDraftForDeviceSampling(
                                draft_idx - 1,
                                base_sidecar_position + draft_idx);
                    }
                    else if (use_sidecar_stream_handoff_for_stochastic)
                    {
                        sidecar_ok =
                            runner_->forwardMTPFromLastDraftForDeviceSampling(
                                draft_tokens.back(),
                                base_sidecar_position + draft_idx);
                    }
                    else
                    {
                        sidecar_ok = runner_->forwardMTPFromLastDraft(
                            draft_tokens.back(),
                            base_sidecar_position + draft_idx);
                    }
                }
            }
            if (!sidecar_ok)
            {
                return fail_after_checkpoint(
                    draft_idx == 0
                        ? "MTP sidecar forward failed"
                        : "Chained MTP sidecar forward failed");
            }
            if (use_sidecar_stream_handoff_for_stochastic)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_sidecar_stream_handoff_attempts",
                    1.0,
                    "decode",
                    {},
                    {{"draft_idx", std::to_string(draft_idx)}});
                if (draft_idx > 0 && use_device_draft_token_sidecar)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_sidecar_device_token_inputs",
                        1.0,
                        "decode",
                        {},
                        {{"draft_idx", std::to_string(draft_idx)}});
                }
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "sidecar_iteration_flush",
                    "decode");
                if (!runner_->flushPendingMTPWork())
                {
                    return fail_after_checkpoint("MTP sidecar stream flush failed");
                }
            }

            if (draft_idx == 0)
            {
                if (use_all_position_state_publication_verifier)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "post_sidecar_checkpoint_skipped_all_position_publication",
                        1.0,
                        "decode");
                }
                else
                {
                    PerfStatsCollector::ScopedTimer timer("mtp", "capture_post_sidecar_prefix_state", "decode");
                    sidecar_checkpoints.push_back(runner_->captureLivePrefixCheckpoint());
                    if (!sidecar_checkpoints.back().valid)
                    {
                        return fail_after_checkpoint("MTP decode could not capture post-sidecar shifted state");
                    }
                }
            }
            else
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "post_sidecar_checkpoint_skipped_speculative",
                    1.0,
                    "decode");
            }

            if (!use_sidecar_sample_fusion)
            {
                const bool next_sidecar_needs_host_token =
                    draft_idx + 1 < speculative_draft_count &&
                    !use_device_draft_token_sidecar;
                const bool defer_draft_host_read =
                    can_defer_stochastic_draft_host_reads &&
                    !next_sidecar_needs_host_token;
                mtp_token = sample_mtp_token(draft_idx, defer_draft_host_read);
            }
            if (mtp_token < 0)
            {
                if (mtp_token != kDeferredMTPDraftTokenShadow)
                    return fail_after_checkpoint("No MTP logits available");
            }
            if (use_sidecar_sample_fusion)
            {
                PerfStatsCollector::addCounter("mtp", "mtp_token_device_samples", 1.0, "decode");
            }
            draft_tokens.push_back(mtp_token);
            if ((stochastic_verify || use_sampling_penalties) &&
                mtp_token != kDeferredMTPDraftTokenShadow)
            {
                draft_sampler.record_token(mtp_token);
            }

            ++mtp_stats_.draft_steps;
            PerfStatsCollector::addCounter("mtp", "draft_steps", 1.0, "decode");
        }

        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_final_flush_before_verification",
                "decode");
            if (!runner_->flushPendingMTPWork())
            {
                return fail_after_checkpoint("MTP sidecar stream flush failed before verification");
            }
        }

        if (DebugEnv::isTruthyEnv("LLAMINAR_MTP_VERIFY_SIDECAR_PRESERVES_MAIN_STATE"))
        {
            auto join_debug_tokens = [](const std::vector<int32_t> &tokens) -> std::string
            {
                std::ostringstream oss;
                for (size_t i = 0; i < tokens.size(); ++i)
                {
                    if (i)
                        oss << ",";
                    oss << tokens[i];
                }
                return oss.str();
            };

            PrefixStateSnapshot sidecar_state = runner_->captureLivePrefixState();
            if (!sidecar_state.valid)
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not capture sidecar state");
            }

            auto verifier_rows_from_current_state = [&]()
                -> std::optional<std::vector<int32_t>>
            {
                const MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                    buildSingleRequestVerifierInputPlan(draft_tokens);
                if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                    return std::nullopt;
                const int verifier_row_count =
                    verifier_input_plan.compact_logit_row_count;
                ScopedMTPSpecVerifierInputPlan verifier_plan_scope(
                    runner_.get(),
                    verifier_input_plan);
                if (!verifier_plan_scope.installed())
                    return std::nullopt;
                if (!runner_->setComputeRowIndexedAllPositionLogits(true, verifier_row_count))
                    return std::nullopt;
                if (!runner_->setComputeAllPositionLogits(true))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return std::nullopt;
                }
                const MTPVerifierForwardExecutionResult forward_result =
                    executeMTPSpecVerifierForward(
                        *runner_,
                        verifier_input_plan);
                if (!forward_result.ok)
                {
                    runner_->setComputeAllPositionLogits(false);
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return std::nullopt;
                }
                if (!runner_->setComputeAllPositionLogits(false))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return std::nullopt;
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(false, 0))
                    return std::nullopt;
                std::vector<int32_t> rows(
                    static_cast<size_t>(verifier_row_count),
                    -1);
                if (!runner_->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                        0,
                        static_cast<int>(rows.size()),
                        rows.data()))
                {
                    return std::nullopt;
                }
                return rows;
            };

            std::optional<std::vector<int32_t>> sidecar_rows =
                verifier_rows_from_current_state();
            if (!sidecar_rows)
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not sample sidecar verifier rows");
            }
            if (!runner_->restoreLivePrefixState(verifier_base_checkpoint))
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not restore verifier base checkpoint");
            }
            std::optional<std::vector<int32_t>> base_rows =
                verifier_rows_from_current_state();
            if (!base_rows)
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not sample base verifier rows");
            }
            if (!runner_->restoreLivePrefixState(sidecar_state))
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not restore sidecar state");
            }
            if (*sidecar_rows != *base_rows)
            {
                return fail_after_checkpoint(
                    "MTP sidecar mutated main verifier state: condition_token=" +
                    std::to_string(condition_token) +
                    " first_token=" + std::to_string(first_token) +
                    " draft_tokens=" + join_debug_tokens(draft_tokens) +
                    " sidecar_rows=" + join_debug_tokens(*sidecar_rows) +
                    " base_rows=" + join_debug_tokens(*base_rows) +
                    " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false")));
            }
        }

        auto verify_committed_prefix_replay = [&](
                                                  const char *path,
                                                  const std::vector<int32_t> &tokens_to_replay,
                                                  int32_t expected_next_token,
                                                  const std::string &debug_context = {})
            -> std::optional<std::string>
        {
            if (!verify_commit_replay_check)
            {
                return std::nullopt;
            }
            if (tokens_to_replay.empty() ||
                !verifier_replay_base_checkpoint.has_value())
            {
                return std::nullopt;
            }

            PrefixStateSnapshot committed_checkpoint =
                runner_->captureLivePrefixState();
            if (!committed_checkpoint.valid)
            {
                return std::string("MTP commit replay check could not capture committed state");
            }
            auto summarize_probe = [](const PrefixRuntimeStateSnapshot &probe)
            {
                auto summarize_cache = [](const std::vector<PrefixKVCacheProbe> &caches)
                {
                    std::string out;
                    for (const auto &cache : caches)
                    {
                        if (!out.empty())
                            out += ";";
                        out += cache.owner + ":";
                        const size_t limit = std::min<size_t>(cache.layers.size(), 6);
                        for (size_t i = 0; i < limit; ++i)
                        {
                            if (i > 0)
                                out += ",";
                            const auto &layer = cache.layers[i];
                            out += "L" + std::to_string(layer.global_layer) +
                                   "/S" + std::to_string(layer.seq_idx) +
                                   "=" + std::to_string(layer.cached_tokens) +
                                   "@" + std::to_string(layer.ring_head);
                        }
                        if (cache.layers.size() > limit)
                            out += ",...";
                    }
                    return out.empty() ? std::string("none") : out;
                };

                std::string positions;
                for (size_t i = 0; i < probe.positions.size(); ++i)
                {
                    if (i > 0)
                        positions += ",";
                    positions += std::to_string(probe.positions[i]);
                }
                std::string seqs;
                for (size_t i = 0; i < probe.sequence_lengths.size(); ++i)
                {
                    if (i > 0)
                        seqs += ",";
                    seqs += std::to_string(probe.sequence_lengths[i]);
                }
                std::string gdn;
                const size_t gdn_limit = std::min<size_t>(probe.gdn_layers.size(), 4);
                for (size_t i = 0; i < gdn_limit; ++i)
                {
                    if (i > 0)
                        gdn += ",";
                    const auto &layer = probe.gdn_layers[i];
                    gdn += "L" + std::to_string(layer.global_layer) +
                           "/r=" + std::to_string(layer.recurrence_hash) +
                           "/c=" + std::to_string(layer.conv_hash);
                }
                if (probe.gdn_layers.size() > gdn_limit)
                    gdn += ",...";
                if (gdn.empty())
                    gdn = "none";

                return std::string("pos=[") + positions +
                       "] seq=[" + seqs +
                       "] kv={" + summarize_cache(probe.kv_caches) +
                       "} mtp={" + summarize_cache(probe.mtp_kv_caches) +
                       "} gdn={" + gdn + "}";
            };
            const PrefixRuntimeStateSnapshot committed_probe_before =
                runner_->prefixStateProbe();

            int continuation_check_depth = 1;
            if (const char *depth_env =
                    DebugEnv::envValue("LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_DEPTH"))
            {
                char *end = nullptr;
                const long parsed = std::strtol(depth_env, &end, 10);
                if (end != depth_env && parsed > 0)
                {
                    continuation_check_depth =
                        static_cast<int>(std::min<long>(parsed, 16));
                }
            }

            auto sample_continuation = [&](int32_t first_input)
                -> std::optional<std::vector<int32_t>>
            {
                std::vector<int32_t> tokens;
                tokens.reserve(static_cast<size_t>(continuation_check_depth));
                int32_t input = first_input;
                for (int i = 0; i < continuation_check_depth; ++i)
                {
                    if (!runner_->forward(&input, 1))
                    {
                        return std::nullopt;
                    }
                    const int32_t sampled = runner_->sampleGreedyOnDevice();
                    if (sampled < 0)
                    {
                        return std::nullopt;
                    }
                    tokens.push_back(sampled);
                    input = sampled;
                }
                return tokens;
            };

            const bool derived_next_token_from_deferred_condition =
                expected_next_token < 0;
            auto prepare_committed_ready_state = [&]()
                -> std::optional<int32_t>
            {
                if (!derived_next_token_from_deferred_condition)
                    return expected_next_token;

                /*
                 * A forced reject has no ready token yet: publication advances
                 * only the accepted verifier prefix, while the rejected
                 * correction is the next ordinary condition token.  The debug
                 * oracle therefore has to run that one condition forward before
                 * comparing the committed state against a full replay.
                 */
                const int32_t deferred_condition_token = tokens_to_replay.back();
                if (!runner_->forward(&deferred_condition_token, 1))
                    return std::nullopt;
                const int32_t sampled = runner_->sampleGreedyOnDevice();
                if (sampled < 0)
                    return std::nullopt;
                PerfStatsCollector::addCounter(
                    "mtp",
                    "commit_replay_check_derived_next_tokens",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"deferred_condition_token",
                      std::to_string(deferred_condition_token)},
                     {"next_token", std::to_string(sampled)}});
                return sampled;
            };

            std::optional<int32_t> prepared_next_token =
                prepare_committed_ready_state();
            if (!prepared_next_token)
            {
                return derived_next_token_from_deferred_condition
                           ? std::string("MTP commit replay check deferred condition forward failed")
                           : std::string("MTP commit replay check missing expected next token");
            }
            expected_next_token = *prepared_next_token;

            std::optional<std::vector<int32_t>> live_committed_continuation =
                sample_continuation(expected_next_token);
            if (!live_committed_continuation)
            {
                return std::string("MTP commit replay check live committed continuation forward failed");
            }
            if (!runner_->restoreLivePrefixState(committed_checkpoint))
            {
                return std::string("MTP commit replay check could not restore committed state after live continuation check");
            }

            const PrefixStateSnapshot &base = *verifier_replay_base_checkpoint;
            if (!runner_->restoreLivePrefixState(base))
            {
                return std::string("MTP commit replay check could not restore verifier base state");
            }
            bool sequential_replay_ok = true;
            for (int32_t replay_token : tokens_to_replay)
            {
                if (!runner_->forward(&replay_token, 1))
                {
                    sequential_replay_ok = false;
                    break;
                }
            }
            if (!sequential_replay_ok)
            {
                return std::string("MTP commit replay check sequential replay forward failed");
            }
            const int32_t replay_next_token = runner_->sampleGreedyOnDevice();
            if (replay_next_token < 0)
            {
                return std::string("MTP commit replay check full replay sampling failed");
            }
            if (!runner_->restoreLivePrefixState(committed_checkpoint))
            {
                return std::string("MTP commit replay check could not restore committed state");
            }
            if (replay_next_token != expected_next_token)
            {
                return std::string("MTP committed state mismatch against full replay: path=") +
                       path +
                       " condition_token=" + std::to_string(condition_token) +
                       " accepted_tokens=" + join_tokens(tokens_to_replay) +
                       " committed_next=" + std::to_string(expected_next_token) +
                       " replay_next=" + std::to_string(replay_next_token) +
                       " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false")) +
                       (debug_context.empty() ? std::string{} : " " + debug_context);
            }

            std::optional<int32_t> committed_ready_token =
                prepare_committed_ready_state();
            if (!committed_ready_token ||
                *committed_ready_token != expected_next_token)
            {
                return std::string("MTP commit replay check committed ready-token derivation mismatch");
            }
            std::optional<std::vector<int32_t>> committed_continuation =
                sample_continuation(expected_next_token);
            if (!committed_continuation)
            {
                return std::string("MTP commit replay check committed continuation forward failed");
            }
            if (!runner_->restoreLivePrefixState(base))
            {
                return std::string("MTP commit replay check could not restore verifier base for continuation replay");
            }
            bool sequential_continuation_ok = true;
            for (int32_t replay_token : tokens_to_replay)
            {
                if (!runner_->forward(&replay_token, 1))
                {
                    sequential_continuation_ok = false;
                    break;
                }
            }
            if (sequential_continuation_ok)
            {
                std::optional<std::vector<int32_t>> replay_continuation =
                    sample_continuation(expected_next_token);
                if (!replay_continuation)
                {
                    return std::string("MTP commit replay check continuation replay forward failed");
                }
                if (!runner_->restoreLivePrefixState(committed_checkpoint))
                {
                    return std::string("MTP commit replay check could not restore committed state after continuation check");
                }
                auto summarize_prefix_replays = [&]() -> std::string
                {
                    std::string summary;
                    for (size_t prefix_len = 0;
                         prefix_len <= tokens_to_replay.size();
                         ++prefix_len)
                    {
                        if (!runner_->restoreLivePrefixState(base))
                        {
                            summary += " len" + std::to_string(prefix_len) + "=restore_failed";
                            continue;
                        }
                        bool prefix_ok = true;
                        for (size_t i = 0; i < prefix_len; ++i)
                        {
                            const int32_t replay_token = tokens_to_replay[i];
                            if (!runner_->forward(&replay_token, 1))
                            {
                                prefix_ok = false;
                                break;
                            }
                        }
                        if (!prefix_ok)
                        {
                            summary += " len" + std::to_string(prefix_len) + "=forward_failed";
                            continue;
                        }
                        std::optional<std::vector<int32_t>> prefix_continuation =
                            sample_continuation(expected_next_token);
                        summary += " len" + std::to_string(prefix_len) + "=";
                        summary += prefix_continuation
                                       ? join_tokens(*prefix_continuation)
                                       : std::string("sample_failed");
                    }
                    (void)runner_->restoreLivePrefixState(committed_checkpoint);
                    return summary;
                };
                if (*live_committed_continuation != *replay_continuation)
                {
                    const PrefixRuntimeStateSnapshot mismatch_probe =
                        runner_->prefixStateProbe();
                    return std::string("MTP live committed state continuation mismatch against full replay: path=") +
                           path +
                           " condition_token=" + std::to_string(condition_token) +
                           " accepted_tokens=" + join_tokens(tokens_to_replay) +
                           " next_token=" + std::to_string(expected_next_token) +
                           " live_committed_continuation=" + join_tokens(*live_committed_continuation) +
                           " committed_continuation=" + join_tokens(*committed_continuation) +
                           " replay_continuation=" + join_tokens(*replay_continuation) +
                           " prefix_replay_continuations=" + summarize_prefix_replays() +
                           " committed_probe_before={" + summarize_probe(committed_probe_before) + "}" +
                           " mismatch_probe={" + summarize_probe(mismatch_probe) + "}" +
                           " continuation_depth=" + std::to_string(continuation_check_depth) +
                           " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false"));
                }
                if (*committed_continuation != *replay_continuation)
                {
                    const PrefixRuntimeStateSnapshot mismatch_probe =
                        runner_->prefixStateProbe();
                    return std::string("MTP committed state continuation mismatch against full replay: path=") +
                           path +
                           " condition_token=" + std::to_string(condition_token) +
                           " accepted_tokens=" + join_tokens(tokens_to_replay) +
                           " next_token=" + std::to_string(expected_next_token) +
                           " live_committed_continuation=" + join_tokens(*live_committed_continuation) +
                           " committed_continuation=" + join_tokens(*committed_continuation) +
                           " replay_continuation=" + join_tokens(*replay_continuation) +
                           " prefix_replay_continuations=" + summarize_prefix_replays() +
                           " committed_probe_before={" + summarize_probe(committed_probe_before) + "}" +
                           " mismatch_probe={" + summarize_probe(mismatch_probe) + "}" +
                           " continuation_depth=" + std::to_string(continuation_check_depth) +
                           " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false"));
                }
            }
            if (!sequential_continuation_ok)
            {
                return std::string("MTP commit replay check continuation replay forward failed");
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "commit_replay_check_matches",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"accepted_tokens", join_tokens(tokens_to_replay)},
                 {"next_token", std::to_string(expected_next_token)},
                 {"continuation_depth", std::to_string(continuation_check_depth)},
                 {"derived_next_token",
                  derived_next_token_from_deferred_condition ? "true" : "false"},
                 {"used_ready_logits", use_ready_logits ? "true" : "false"}});
            return std::nullopt;
        };

        if (use_all_position_state_publication_verifier)
        {
            const bool sidecar_preserves_main_state =
                runner_->supportsMTPSidecarPreservesMainState();
            bool restored_verifier_base = sidecar_preserves_main_state;
            if (sidecar_preserves_main_state)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_verifier_base_restore_skipped_sidecar_preserved",
                    1.0,
                    "decode",
                    {},
                    {{"draft_tokens", std::to_string(draft_tokens.size())},
                     {"cached_tokens", std::to_string(verifier_base_checkpoint.cached_tokens)}});
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "all_position_verifier_restore_base_checkpoint",
                    "decode");
                restored_verifier_base =
                    runner_->restoreLivePrefixState(verifier_base_checkpoint);
                if (restored_verifier_base)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "all_position_verifier_base_restores",
                        1.0,
                        "decode",
                        {},
                        {{"draft_tokens", std::to_string(draft_tokens.size())},
                         {"cached_tokens", std::to_string(verifier_base_checkpoint.cached_tokens)}});
                }
            }
            if (!restored_verifier_base)
            {
                return fail_after_checkpoint(
                    "All-position MTP verifier could not restore verifier base checkpoint after sidecar draft");
            }

            bool first_token_is_stop =
                first_token != kDeferredMTPFirstTokenShadow &&
                std::find(stop_tokens_.begin(),
                          stop_tokens_.end(),
                          first_token) != stop_tokens_.end();
            /*
             * The first sidecar draft already consumes the first emitted token
             * and appends the corresponding shifted MTP KV row.  When the
             * sidecar is declared main-state preserving we can keep that row
             * and let later prefix/publication truncation discard any deeper
             * speculative rows.  Re-running a KV-only sidecar here would
             * duplicate the same first-row work on every speculative step.
             */
            const bool first_shifted_row_available_from_sidecar =
                sidecar_preserves_main_state && !first_token_is_stop;
            if (!first_token_is_stop &&
                !first_shifted_row_available_from_sidecar)
            {
                bool shifted_commit_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "all_position_initial_shifted_commit",
                        "decode");
                    if (first_token == kDeferredMTPFirstTokenShadow)
                    {
                        shifted_commit_ok =
                            runner_->commitMTPShiftedRowFromDeviceTargetSample(
                                /*target_sample_slot=*/0,
                                /*already_appended_tokens=*/0,
                                /*allow_speculative_discard=*/true,
                                base_sidecar_position);
                    }
                    else
                    {
                        shifted_commit_ok =
                            runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                                first_token,
                                /*already_appended_tokens=*/0,
                                /*allow_speculative_discard=*/true,
                                base_sidecar_position);
                    }
                }
                if (!shifted_commit_ok)
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier initial shifted-cache commit failed");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_initial_shifted_commits",
                    1.0,
                    "decode");
            }
            else if (first_shifted_row_available_from_sidecar)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_initial_shifted_reused_sidecar_rows",
                    1.0,
                    "decode",
                    {},
                    {{"draft_tokens", std::to_string(draft_tokens.size())},
                     {"first_token_deferred",
                      first_token == kDeferredMTPFirstTokenShadow ? "true" : "false"}});
            }

            const MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                buildSingleRequestVerifierInputPlan(draft_tokens);
            if (!verifier_input_plan.ok)
            {
                return fail_after_checkpoint(
                    std::string("All-position MTP verifier input metadata failed: ") +
                    verifier_input_plan.error);
            }
            if (!verifierInputPlanHasCompactRows(verifier_input_plan))
            {
                return fail_after_checkpoint(
                    "All-position MTP verifier row metadata is malformed");
            }

            std::vector<int32_t> sampled_verifier_rows(
                static_cast<size_t>(verifier_input_plan.compact_logit_row_count),
                -1);
            /*
             * Greedy row sampling already consumes a deferred verifier stream.
             * Stochastic can do the same only for the penalty-free batched
             * device lane: every target distribution is built from immutable
             * verifier rows, then the backend batch-outcome reducer performs
             * the single host-visible synchronization. Penalty-bearing rows
             * still depend on sampler history between accepted tokens, so they
             * keep the synchronized verifier boundary.
             */
            const bool can_defer_stochastic_batch_verifier_sync =
                stochastic_verify &&
                stochastic_device_verify &&
                !active_sampling_params_.has_penalties() &&
                draft_tokens.size() > 1 &&
                !first_token_is_stop &&
                stop_tokens_.size() <=
                    static_cast<size_t>(
                        sampling_math::kSpeculativeBatchMaxStopTokens);
            const bool defer_all_position_verifier_sync =
                runner_->primaryDeviceId().is_gpu() &&
                (!stochastic_verify ||
                 can_defer_stochastic_batch_verifier_sync);
            ScopedMTPAllPositionVerifierSyncDeferral verifier_sync_deferral(
                runner_.get(),
                defer_all_position_verifier_sync);
            {
                PerfStatsCollector::ScopedTimer verifier_timer(
                    "mtp",
                    "verifier_forward",
                    "decode",
                    {},
                    {{"implementation", "all_position_state_publication"},
                     {"verifier_path", "all_position_state_publication"}});
                const int verifier_row_count =
                    verifier_input_plan.compact_logit_row_count;
                // The verifier forward still consumes every draft token so KV,
                // GDN, and MoE state publication can see the full sequence.
                // Row-indexed logits only shrink the LM-head projection rows.
                ScopedMTPSpecVerifierInputPlan verifier_plan_scope(
                    runner_.get(),
                    verifier_input_plan);
                if (!verifier_plan_scope.installed())
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not install row metadata plan");
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(
                        true,
                        verifier_row_count))
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not enable row-indexed logits");
                }
                if (!runner_->setComputeAllPositionLogits(true))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not enable all-position logits");
                }
                const void *verifier_input_tokens_device = nullptr;
                if (can_defer_stochastic_batch_verifier_sync)
                {
                    /*
                     * Penalty-free stochastic verification already stores the
                     * sampled sidecar draft tokens in runner-owned device slots.
                     * Ask the runner to compose the compact verifier input row
                     * on device so the embedding graph can read it directly.
                     */
                    verifier_input_tokens_device =
                        first_token == kDeferredMTPFirstTokenShadow
                            ? runner_->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
                                  /*first_target_sample_slot=*/0,
                                  /*first_draft_slot=*/0,
                                  verifier_input_plan.total_verifier_input_tokens - 1,
                                  verifier_input_plan.total_verifier_input_tokens)
                            : runner_->prepareMTPVerifierInputTokensOnDevice(
                                  first_token,
                                  /*first_draft_slot=*/0,
                                  verifier_input_plan.total_verifier_input_tokens - 1,
                                  verifier_input_plan.total_verifier_input_tokens);
                    if (!verifier_input_tokens_device)
                    {
                        runner_->setComputeAllPositionLogits(false);
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        return fail_after_checkpoint(
                            "All-position MTP verifier could not prepare device token input");
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "all_position_verifier_device_token_inputs",
                        1.0,
                        "decode",
                        {},
                        {{"total_tokens",
                          std::to_string(
                              verifier_input_plan.total_verifier_input_tokens)}});
                }

                MTPVerifierForwardExecutionOptions verifier_forward_options;
                verifier_forward_options.device_token_ids =
                    verifier_input_tokens_device;
                const MTPVerifierForwardExecutionResult verifier_forward =
                    executeMTPSpecVerifierForward(
                        *runner_,
                        verifier_input_plan,
                        verifier_forward_options);
                if (!verifier_forward.ok)
                {
                    runner_->setComputeAllPositionLogits(false);
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return fail_after_checkpoint(
                        std::string("All-position MTP verifier forward failed: ") +
                        verifier_forward.error);
                }
                if (!runner_->setComputeAllPositionLogits(false))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not disable all-position logits");
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(false, 0))
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not disable row-indexed logits");
                }
            }

            MTPDecodeCatchupGreedyRequest catchup_request;
            catchup_request.draft_tokens = draft_tokens;
            catchup_request.stop_tokens = stop_tokens_;
            catchup_request.base_sidecar_position = base_sidecar_position;
            catchup_request.allow_speculative_discard = true;
            catchup_request.verifier_path = "all_position_state_publication";
            catchup_request.implementation_name = "all_position_state_publication";
            catchup_request.verifier_base_checkpoint = &verifier_base_checkpoint;

            Sampler all_position_stochastic_penalty_sampler = sampler_;
            MTPDecodeCatchupGreedyResult catchup;
            std::optional<DeviceSpeculativeVerifyBatchOutcome>
                device_batch_outcome_for_transaction;
            if (stochastic_verify)
            {
                if (!stochastic_device_verify && !stochastic_host_verify)
                {
                    return fail_after_checkpoint(
                        "All-position stochastic MTP verifier requires device-resident or host distribution verification");
                }

                std::vector<MTPRejectionSampleRowResult> stochastic_rows;
                stochastic_rows.reserve(
                    draft_tokens.size() > 0 ? draft_tokens.size() - 1 : 0);
                bool stochastic_stopped_on_output = false;
                std::optional<int32_t> bonus_ready_token;
                if (first_token != kDeferredMTPFirstTokenShadow)
                {
                    all_position_stochastic_penalty_sampler.record_token(first_token);
                }

                if (std::find(stop_tokens_.begin(),
                              stop_tokens_.end(),
                              first_token) != stop_tokens_.end())
                {
                    stochastic_stopped_on_output = true;
                }

                std::vector<SamplingDistributionEntry> host_target_distribution;
                auto build_all_position_target_distribution =
                    [&](int row, int slot) -> bool
                {
                    if (stochastic_host_verify)
                    {
                        const float *all_position_logits =
                            runner_->getAllPositionLogits();
                        if (!all_position_logits || row < 0)
                            return false;

                        const float *row_logits =
                            all_position_logits +
                            static_cast<size_t>(row) * static_cast<size_t>(vocab);
                        PerfStatsCollector::ScopedTimer timer(
                            "mtp",
                            "all_position_stochastic_host_target_distribution",
                            "decode",
                            {},
                            {{"implementation", "all_position_state_publication"}});
                        host_target_distribution =
                            all_position_stochastic_penalty_sampler.compute_distribution(
                                row_logits,
                                static_cast<size_t>(vocab),
                                active_sampling_params_);
                        return !host_target_distribution.empty();
                    }

                    auto penalty_map =
                        all_position_stochastic_penalty_sampler
                            .compute_penalty_map(active_sampling_params_, vocab);
                    if (!penalty_map.empty() &&
                        !runner_->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                            row,
                            penalty_map,
                            vocab))
                    {
                        return false;
                    }
	                    return runner_->buildStochasticDistributionOnDevice(
	                        DeviceLogitsSource::AllPosition,
	                        row,
	                        DeviceDistributionBuffer::Target,
                        slot,
                        active_sampling_params_,
	                        vocab);
	                };

                auto inverse_sample_seed_for_thresholds =
                    [&](const float *thresholds, size_t count) -> uint64_t
                {
                    if (active_sampling_params_.seed != 0)
                    {
                        return static_cast<uint64_t>(
                            active_sampling_params_.seed);
                    }

                    /*
                     * Unseeded stochastic decode still needs GPU-side random
                     * inverse-exponential rows for vLLM rejection recovery.
                     * Mix the residual draws that already belong to this
                     * verifier step so captured and uncaptured execution use a
                     * stable per-step random matrix without a host full-vocab
                     * upload.
                     */
                    uint64_t seed = 0xD1B54A32D192ED03ull;
                    for (size_t i = 0; i < count; ++i)
                    {
                        uint32_t bits = 0;
                        std::memcpy(&bits, thresholds + i, sizeof(bits));
                        seed = sampling_math::splitmix64(
                            seed ^ static_cast<uint64_t>(bits));
                    }
                    return seed;
                };

                auto apply_vllm_row_penalties =
                    [&](int compare_rows, int bonus_row) -> bool
                {
                    if (!active_sampling_params_.has_penalties())
                        return true;
                    if (first_token == kDeferredMTPFirstTokenShadow)
                        return false;

                    /*
                     * Row i is consumed only on the branch where all previous
                     * draft rows accepted.  That makes the row-local penalty
                     * history deterministic: base sampler history, first
                     * target token, then draft[1..i-1].  This is the same
                     * speculative-row metadata shape vLLM uses; it preserves
                     * exact sampler history without a per-row host sync.
                     */
                    Sampler row_penalty_sampler = sampler_;
                    row_penalty_sampler.record_token(first_token);
                    for (int row = 0; row < compare_rows; ++row)
                    {
                        auto penalty_map =
                            row_penalty_sampler.compute_penalty_map(
                                active_sampling_params_,
                                vocab);
                        if (!penalty_map.empty() &&
                            !runner_->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                                row,
                                penalty_map,
                                vocab))
                        {
                            return false;
                        }
                        row_penalty_sampler.record_token(
                            draft_tokens[static_cast<size_t>(row + 1)]);
                    }

                    auto bonus_penalty_map =
                        row_penalty_sampler.compute_penalty_map(
                            active_sampling_params_,
                            vocab);
                    if (!bonus_penalty_map.empty() &&
                        !runner_->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                            bonus_row,
                            bonus_penalty_map,
                            vocab))
                    {
                        return false;
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_vllm_penalty_rows_preapplied",
                        static_cast<double>(compare_rows + 1),
                        "decode",
                        {},
                        {{"verifier_path", "all_position_state_publication"}});
                    return true;
                };

                const bool can_batch_penalty_rows =
                    !active_sampling_params_.has_penalties() ||
                    first_token != kDeferredMTPFirstTokenShadow;
                const bool batched_device_rejection =
                    stochastic_device_verify &&
                    can_batch_penalty_rows &&
                    draft_tokens.size() > 1 &&
                    !stochastic_stopped_on_output &&
                    stop_tokens_.size() <=
                        static_cast<size_t>(
                            sampling_math::kSpeculativeBatchMaxStopTokens);
                bool used_device_batch_outcome = false;
                std::vector<float> batched_accept_thresholds;
                std::vector<float> batched_residual_thresholds;
                if (batched_device_rejection)
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "all_position_stochastic_device_batch_outcome",
                        "decode",
                        {},
                        {{"verifier_path", "all_position_state_publication"}});
                    const int compare_rows =
                        static_cast<int>(draft_tokens.size()) - 1;
                    batched_accept_thresholds.reserve(static_cast<size_t>(compare_rows));
                    batched_residual_thresholds.reserve(static_cast<size_t>(compare_rows));

                    const int bonus_row = compare_rows;
                    if (!apply_vllm_row_penalties(compare_rows, bonus_row))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP vLLM row penalty application failed");
                    }
                    if (!runner_->buildStochasticProcessedLogitRowsOnDevice(
                            DeviceLogitsSource::AllPosition,
                            /*first_row=*/0,
                            DeviceDistributionBuffer::Target,
                            /*first_slot=*/0,
                            /*row_count=*/compare_rows + 1,
                            active_sampling_params_,
                            vocab))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP batched target processed-row build failed");
                    }

                    for (int row = 0; row < compare_rows; ++row)
                    {
                        const int row_logical_position =
                            checkpoint.cached_tokens + 1 + row;
                        batched_accept_thresholds.push_back(
                            accept_threshold_for_position(
                                sampler_,
                                row_logical_position));
                        batched_residual_thresholds.push_back(
                            residual_threshold_for_position(
                                sampler_,
                                row_logical_position));
                    }

                    // Preserve seeded RNG semantics: the bonus threshold is
                    // drawn from a copy and committed only when the device
                    // summary says the bonus token was semantically consumed.
                    Sampler bonus_sampler = sampler_;
                    const float bonus_threshold =
                        sample_threshold_for_position(
                            bonus_sampler,
                            checkpoint.cached_tokens +
                                static_cast<int>(draft_tokens.size()));
                    const uint64_t inverse_sample_seed =
                        inverse_sample_seed_for_thresholds(
                            batched_residual_thresholds.data(),
                            batched_residual_thresholds.size());
                    const int inverse_sample_first_logical_position =
                        checkpoint.cached_tokens + 1;
                    DeviceSpeculativeVerifyBatchOutcome device_outcome;
                    const bool device_outcome_ok =
                        first_token == kDeferredMTPFirstTokenShadow
                            ? runner_->verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
                                  /*first_target_slot=*/0,
                                  /*first_draft_slot=*/0,
                                  /*draft_tokens=*/nullptr,
                                  batched_accept_thresholds.data(),
                                  batched_residual_thresholds.data(),
                                  compare_rows,
                                  /*first_target_sample_slot=*/0,
                                  stop_tokens_.data(),
                                  static_cast<int>(stop_tokens_.size()),
                                  bonus_row,
                                  bonus_threshold,
                                  &device_outcome,
                                  inverse_sample_seed,
                                  inverse_sample_first_logical_position,
                                  /*use_vllm_probability_rejection=*/true)
                            : runner_->verifyStochasticDistributionsBatchOutcomeOnDevice(
                                  /*first_target_slot=*/0,
                                  /*first_draft_slot=*/0,
                                  /*draft_tokens=*/nullptr,
                                  batched_accept_thresholds.data(),
                                  batched_residual_thresholds.data(),
                                  compare_rows,
                                  first_token,
                                  stop_tokens_.data(),
                                  static_cast<int>(stop_tokens_.size()),
                                  bonus_row,
                                  bonus_threshold,
                                  &device_outcome,
                                  inverse_sample_seed,
                                  inverse_sample_first_logical_position,
                                  /*use_vllm_probability_rejection=*/true);
                    if (!device_outcome_ok)
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP batched device outcome verifier failed");
                    }
                    if (device_outcome.sampled_terminal)
                        sampler_ = bonus_sampler;

                    catchup =
                        buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                            catchup_request,
                            device_outcome);
                    if (!catchup.ok)
                        return fail_after_checkpoint(catchup.error);
                    device_batch_outcome_for_transaction = device_outcome;

                    for (size_t i = 0;
                         i < catchup.verifier_tokens.size() &&
                         i < sampled_verifier_rows.size();
                         ++i)
                    {
                        sampled_verifier_rows[i] = catchup.verifier_tokens[i];
                    }
                    if (device_outcome.sampled_terminal &&
                        device_outcome.consumed_verifier_rows >= 0 &&
                        static_cast<size_t>(device_outcome.consumed_verifier_rows) <
                            sampled_verifier_rows.size())
                    {
                        sampled_verifier_rows[
                            static_cast<size_t>(
                                device_outcome.consumed_verifier_rows)] =
                            device_outcome.ready_token;
                    }

                    mtp_stats_.stochastic_accept_tests +=
                        static_cast<uint64_t>(
                            std::max(0, device_outcome.consumed_verifier_rows));
                    mtp_stats_.stochastic_accepts +=
                        static_cast<uint64_t>(
                            std::max(0, device_outcome.accepted_speculative_prefix));
                    if (!device_outcome.all_speculative_accepted &&
                        device_outcome.rejected_verified_token >= 0)
                    {
                        ++mtp_stats_.stochastic_residual_samples;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "stochastic_residual_device_samples",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path", "all_position_state_publication"},
                             {"implementation", "device_batch_outcome"}});
                    }
                    if (device_outcome.sampled_terminal)
                    {
                        ++mtp_stats_.stochastic_terminal_samples;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "stochastic_terminal_device_samples",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path", "all_position_state_publication"},
                             {"implementation", "device_batch_outcome"}});
                    }

                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_accept_tests",
                        static_cast<double>(
                            std::max(0, device_outcome.consumed_verifier_rows)),
                        "decode",
                        {},
                        {{"device_resident", "true"},
                         {"verifier_path", "all_position_state_publication"},
                         {"implementation", "device_batch_outcome"}});
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_accepts",
                        static_cast<double>(
                            std::max(0, device_outcome.accepted_speculative_prefix)),
                        "decode",
                        {},
                        {{"verifier_path", "all_position_state_publication"},
                         {"implementation", "device_batch_outcome"}});
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "all_position_stochastic_device_batched_rows",
                        static_cast<double>(compare_rows),
                        "decode",
                        {},
                        {{"verifier_path", "all_position_state_publication"},
                         {"implementation", "device_batch_outcome"}});
                    used_device_batch_outcome = true;
                }

                if (!used_device_batch_outcome &&
                    stochastic_device_verify &&
                    !stochastic_stopped_on_output &&
                    draft_tokens.size() > 1)
                {
                    return fail_after_checkpoint(
                        "GPU stochastic MTP requires the vLLM batched device outcome verifier; "
                        "the legacy scalar full-probability row verifier has been removed");
                }

                for (int draft_idx = 1;
                     !used_device_batch_outcome &&
                     !stochastic_stopped_on_output &&
                     draft_idx < static_cast<int>(draft_tokens.size());
                     ++draft_idx)
                {
                    const int row = draft_idx - 1;
                    if (!build_all_position_target_distribution(row, row))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP target distribution build failed");
                    }

                    const int32_t draft_token =
                        draft_tokens[static_cast<size_t>(draft_idx)];
                    const float accept_threshold =
                        accept_threshold_for_position(
                            sampler_,
                            checkpoint.cached_tokens + draft_idx);
                    const float residual_threshold =
                        residual_threshold_for_position(
                            sampler_,
                            checkpoint.cached_tokens + draft_idx);
                    MTPRejectionSampleRowResult row_result;
                    if (row < 0 ||
                        row >= static_cast<int>(host_mtp_draft_distributions.size()) ||
                        host_mtp_draft_distributions[static_cast<size_t>(row)].empty() ||
                        host_target_distribution.empty())
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP host verifier missing distributions");
                    }
                    const auto &draft_distribution =
                        host_mtp_draft_distributions[static_cast<size_t>(row)];
                    row_result = sampleMTPRejectionRowFromDistributions(
                        host_target_distribution,
                        draft_distribution,
                        draft_token,
                        accept_threshold,
                        residual_threshold);
                    if (!row_result.ok)
                    {
                        return fail_after_checkpoint(
                            std::string("All-position stochastic MTP verifier row failed: ") +
                            row_result.error);
                    }

                    ++mtp_stats_.stochastic_accept_tests;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_accept_tests",
                        1.0,
                        "decode",
                        {},
                        {{"row", std::to_string(row)},
                         {"draft_token", std::to_string(draft_token)},
                         {"accept_probability", std::to_string(row_result.accept_probability)},
                         {"threshold", std::to_string(row_result.accept_threshold)},
                         {"device_resident", stochastic_device_verify ? "true" : "false"},
                         {"verifier_path", "all_position_state_publication"}});

                    const int32_t output_token = row_result.token;
                    stochastic_rows.push_back(row_result);
                    if (row_result.accepted)
                    {
                        sampled_verifier_rows[static_cast<size_t>(row)] =
                            draft_token;
                        ++mtp_stats_.stochastic_accepts;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "stochastic_accepts",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path", "all_position_state_publication"}});
                    }
                    else
                    {
                        if (output_token < 0)
                        {
                            return fail_after_checkpoint(
                                "All-position stochastic MTP residual verifier produced no correction token");
                        }
                        sampled_verifier_rows[static_cast<size_t>(row)] =
                            output_token;
                        ++mtp_stats_.stochastic_residual_samples;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            stochastic_device_verify
                                ? "stochastic_residual_device_samples"
                                : "stochastic_residual_host_samples",
                            1.0,
                            "decode",
                            {},
                            {{"row", std::to_string(row)},
                             {"draft_token", std::to_string(draft_token)},
                             {"correction_token", std::to_string(output_token)},
                             {"verifier_path", "all_position_state_publication"}});
                    }

                    all_position_stochastic_penalty_sampler.record_token(output_token);
                    if (std::find(stop_tokens_.begin(),
                                  stop_tokens_.end(),
                                  output_token) != stop_tokens_.end())
                    {
                        stochastic_stopped_on_output = true;
                        break;
                    }
                    if (!row_result.accepted)
                        break;
                }

                const bool all_rows_verified =
                    stochastic_rows.size() + 1 == draft_tokens.size();
                const bool all_rows_accepted =
                    std::all_of(
                        stochastic_rows.begin(),
                        stochastic_rows.end(),
                        [](const MTPRejectionSampleRowResult &row)
                        {
                            return row.accepted;
                        });

                if (!used_device_batch_outcome &&
                    !stochastic_stopped_on_output &&
                    all_rows_verified &&
                    all_rows_accepted)
                {
                    const int bonus_row =
                        static_cast<int>(draft_tokens.size()) - 1;
                    if (!build_all_position_target_distribution(
                            bonus_row,
                            bonus_row))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP bonus distribution build failed");
                    }
                    const int32_t ready_token =
                        stochastic_device_verify
                            ? runner_->sampleStochasticDistributionOnDevice(
                                  DeviceDistributionBuffer::Target,
                                  bonus_row,
                                  sample_threshold_for_position(
                                      sampler_,
                                      checkpoint.cached_tokens +
                                          static_cast<int>(draft_tokens.size())))
                            : sampleMTPDistributionWithThreshold(
                                  host_target_distribution,
                                  sample_threshold_for_position(
                                      sampler_,
                                      checkpoint.cached_tokens +
                                          static_cast<int>(draft_tokens.size())));
                    if (ready_token < 0)
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP bonus ready-token sampling failed");
                    }
                    bonus_ready_token = ready_token;
                    sampled_verifier_rows[static_cast<size_t>(bonus_row)] =
                        ready_token;
                    ++mtp_stats_.stochastic_terminal_samples;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        stochastic_device_verify
                            ? "stochastic_terminal_device_samples"
                            : "stochastic_terminal_host_samples",
                        1.0,
                        "decode",
                            {},
                            {{"verifier_path", "all_position_state_publication"}});
                }
                if (!used_device_batch_outcome)
                {
                    catchup = buildAllPositionMTPDecodeCatchupStochasticResult(
                        catchup_request,
                        stochastic_rows,
                        bonus_ready_token);
                }
            }
            else
            {
                const bool use_greedy_device_batch_outcome =
                    runner_->primaryDeviceId().is_gpu() &&
                    runner_->supportsGreedyAllPositionBatchOutcomeOnDevice() &&
                    first_token != kDeferredMTPFirstTokenShadow &&
                    stop_tokens_.size() <=
                        static_cast<size_t>(
                            sampling_math::kSpeculativeBatchMaxStopTokens) &&
                    draft_tokens.size() <=
                        static_cast<size_t>(
                            sampling_math::kSpeculativeBatchMaxOutputTokens);
                if (use_greedy_device_batch_outcome)
                {
                    PerfStatsCollector::ScopedTimer sample_timer(
                        "mtp",
                        "all_position_verifier_greedy_device_summary",
                        "decode");
                    DeviceSpeculativeVerifyBatchOutcome device_outcome;
                    if (!runner_->verifyGreedyAllPositionBatchOutcomeOnDevice(
                            draft_tokens.data(),
                            static_cast<int>(draft_tokens.size()),
                            stop_tokens_.data(),
                            static_cast<int>(stop_tokens_.size()),
                            &device_outcome))
                    {
                        return fail_after_checkpoint(
                            "All-position greedy MTP compact device outcome verifier failed");
                    }
                    catchup =
                        buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                            catchup_request,
                            device_outcome);
                    if (!catchup.ok)
                        return fail_after_checkpoint(catchup.error);

                    for (size_t i = 0;
                         i < catchup.verifier_tokens.size() &&
                         i < sampled_verifier_rows.size();
                         ++i)
                    {
                        sampled_verifier_rows[i] = catchup.verifier_tokens[i];
                    }
                    if (device_outcome.sampled_terminal &&
                        device_outcome.consumed_verifier_rows >= 0 &&
                        static_cast<size_t>(device_outcome.consumed_verifier_rows) <
                            sampled_verifier_rows.size())
                    {
                        sampled_verifier_rows[
                            static_cast<size_t>(
                                device_outcome.consumed_verifier_rows)] =
                            device_outcome.ready_token;
                    }

                    PerfStatsCollector::addCounter(
                        "mtp",
                        "all_position_greedy_device_batch_outcomes",
                        1.0,
                        "decode",
                        {},
                        {{"draft_tokens", std::to_string(draft_tokens.size())},
                         {"accepted_prefix",
                          std::to_string(
                              device_outcome.accepted_speculative_prefix)}});
                }
                else
                {
                    PerfStatsCollector::ScopedTimer sample_timer(
                        "mtp",
                        "all_position_verifier_sample_rows",
                        "decode");
                    if (!runner_->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                            0,
                            static_cast<int>(sampled_verifier_rows.size()),
                            sampled_verifier_rows.data()))
                    {
                        return fail_after_checkpoint(
                            "All-position MTP verifier could not sample verifier rows");
                    }
                    catchup = buildAllPositionMTPDecodeCatchupGreedyResult(
                        catchup_request,
                        sampled_verifier_rows);
                }
            }
            if (!catchup.ok)
                return fail_after_checkpoint(catchup.error);

            const bool has_deferred_stochastic_metadata =
                std::find(
                    draft_tokens.begin(),
                    draft_tokens.end(),
                    kDeferredMTPDraftTokenShadow) != draft_tokens.end() ||
                first_token == kDeferredMTPFirstTokenShadow;
            if (has_deferred_stochastic_metadata)
            {
                const bool metadata_first_token_was_deferred =
                    first_token == kDeferredMTPFirstTokenShadow;
                if (catchup.accepted_tokens.empty())
                    return fail_after_checkpoint(
                        "Deferred stochastic MTP metadata requires at least one committed output token");
                if (first_token == kDeferredMTPFirstTokenShadow)
                {
                    first_token = catchup.accepted_tokens.front();
                    first_token_is_stop =
                        std::find(stop_tokens_.begin(),
                                  stop_tokens_.end(),
                                  first_token) != stop_tokens_.end();
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "deferred_stochastic_accepted_outcome_metadata",
                    1.0,
                    "decode",
                    {},
                    {{"draft_tokens", std::to_string(draft_tokens.size())},
                     {"first_token_deferred",
                      metadata_first_token_was_deferred ? "true" : "false"},
                     {"accepted_prefix",
                      std::to_string(catchup.accepted_speculative_prefix)}});
            }

            MTPSpecDecodeMetadataShape metadata_shape;
            metadata_shape.max_requests = 1;
            metadata_shape.max_draft_tokens =
                static_cast<int>(draft_tokens.size());
            const int accepted_verifier_input_prefix =
                std::min<int>(
                    static_cast<int>(draft_tokens.size()),
                    std::max(0, catchup.accepted_speculative_prefix) + 1);
            std::optional<MTPSpecDecodeAcceptedOutcome> deferred_accepted_outcome;
            if (has_deferred_stochastic_metadata)
            {
                deferred_accepted_outcome = MTPSpecDecodeAcceptedOutcome{
                    .request_id = 0,
                    .vocab_size = vocab,
                    .draft_count = static_cast<int>(draft_tokens.size()),
                    .committed_output_tokens = catchup.accepted_tokens,
                    .bonus_ready_token =
                        (!catchup.stopped_on_output &&
                         catchup.all_speculative_accepted &&
                         catchup.ready_token >= 0)
                            ? std::optional<int32_t>{catchup.ready_token}
                            : std::optional<int32_t>{},
                    .accepted_verifier_input_prefix =
                        accepted_verifier_input_prefix,
                    .target_verifier_state_commit_count =
                        catchup.target_verifier_state_commit_count,
                    .all_drafts_accepted = catchup.all_speculative_accepted,
                    .stopped_on_output = catchup.stopped_on_output};
            }
            const int32_t verifier_base_cached_tokens =
                static_cast<int32_t>(verifier_base_checkpoint.cached_tokens);
            MTPSpecTransactionBatchPlan transaction_plan;
            if (device_batch_outcome_for_transaction.has_value())
            {
                /*
                 * Device stochastic verification has already reduced the row
                 * decisions into accepted counts and committed tokens.  Route
                 * that compact outcome through the same batched transaction
                 * driver that future request scheduling will use.
                 */
                const std::vector<int> request_ids{0};
                const std::vector<MTPDecodeCatchupGreedyRequest> requests{
                    catchup_request};
                const std::vector<MTPDeviceRejectionBatchOutcome> device_outcomes{
                    *device_batch_outcome_for_transaction};
                const std::vector<int32_t> base_cached_tokens{
                    verifier_base_cached_tokens};
                transaction_plan =
                    buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
                        metadata_shape,
                        request_ids,
                        vocab,
                        requests,
                        device_outcomes,
                        base_cached_tokens);
            }
            else if (deferred_accepted_outcome.has_value())
            {
                transaction_plan =
                    buildMTPSpecTransactionBatchPlanFromAcceptedOutcome(
                        metadata_shape,
                        *deferred_accepted_outcome,
                        verifier_base_cached_tokens);
            }
            else
            {
                transaction_plan =
                    buildMTPSpecTransactionBatchPlanFromGreedyCatchup(
                        metadata_shape,
                        /*request_id=*/0,
                        vocab,
                        catchup_request,
                        catchup,
                        verifier_base_cached_tokens);
            }
            if (!transaction_plan.ok)
            {
                return fail_after_checkpoint(
                    std::string("All-position MTP verifier transaction plan failed: ") +
                    transaction_plan.error);
            }
            const MTPSpecStepPlanBatch &step_plans =
                transaction_plan.step_plans;
            if (step_plans.steps.size() != 1)
            {
                return fail_after_checkpoint(
                    std::string("All-position MTP verifier step-plan failed: ") +
                    "missing single-request step");
            }

            const MTPSpecStepPlan &step = step_plans.steps.front();
            const int accepted_state_count = std::max(0, step.accepted_count);
            if (!first_token_is_stop && accepted_state_count > 1)
            {
                if (accepted_state_count >
                    static_cast<int>(catchup.accepted_tokens.size()))
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier accepted-state publication exceeds committed outputs");
                }
                bool shifted_catchup_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "all_position_shifted_prefix_commit",
                        "decode");
                    shifted_catchup_ok =
                        runner_->commitMTPShiftedRowsFromPartialForward(
                            catchup.accepted_tokens.data(),
                            accepted_state_count,
                            /*already_appended_tokens=*/1,
                            catchup.main_forward_token_count,
                            /*allow_speculative_discard=*/true,
                            base_sidecar_position);
                }
                if (!shifted_catchup_ok)
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier shifted-cache accepted-prefix commit failed");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_shifted_prefix_commits",
                    static_cast<double>(accepted_state_count - 1),
                    "decode");
            }

            std::string publication_error;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "all_position_publish_accepted_state",
                    "decode");
                if (!runner_->publishAcceptedMTPSpecStateBatch(
                        step_plans,
                        &publication_error))
                {
                    return fail_after_checkpoint(
                        std::string("All-position MTP verifier state publication failed: ") +
                        publication_error);
                }
            }

            int correction_forward_count = 0;
            int deferred_correction_condition_count = 0;
            if (step.requiresCorrectionReplay())
            {
                const int replay_start = step.correction_replay_start_index;
                const int replay_count = step.correction_replay_count;
                if (replay_start < 0 ||
                    replay_start + replay_count >
                        static_cast<int>(catchup.accepted_tokens.size()))
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier deferred correction plan is outside committed outputs");
                }
                correction_forward_count = 0;
                deferred_correction_condition_count = replay_count;
                for (int i = 0; i < replay_count; ++i)
                {
                    const int token_index = replay_start + i;
                    const int32_t replay_token =
                        catchup.accepted_tokens[static_cast<size_t>(token_index)];
                    bool shifted_commit_ok = false;
                    {
                        PerfStatsCollector::ScopedTimer timer(
                            "mtp",
                            "all_position_deferred_correction_shifted_commit",
                            "decode");
                        shifted_commit_ok =
                            runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                                replay_token,
                                token_index,
                                /*allow_speculative_discard=*/true,
                                base_sidecar_position);
                    }
                    if (!shifted_commit_ok)
                    {
                        return fail_after_checkpoint(
                            "All-position MTP verifier deferred correction shifted-cache commit failed");
                    }
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_deferred_correction_condition_tokens",
                    static_cast<double>(deferred_correction_condition_count),
                    "decode",
                    {},
                    {{"verifier_path", "all_position_state_publication"},
                     {"start_index", std::to_string(replay_start)}});
            }

            std::vector<int32_t> accepted_tokens =
                std::move(catchup.accepted_tokens);
            std::vector<int32_t> verifier_tokens =
                std::move(catchup.verifier_tokens);
            const bool all_speculative_accepted =
                catchup.all_speculative_accepted;
            const int accepted_speculative_prefix =
                catchup.accepted_speculative_prefix;
            const int32_t rejected_verified_token =
                catchup.rejected_verified_token;
            const int32_t ready_token = catchup.ready_token;
            const bool stopped_on_output = catchup.stopped_on_output;
            const int main_forward_token_count =
                catchup.main_forward_token_count + correction_forward_count;
            result.is_complete = result.is_complete || stopped_on_output;

            if (!all_speculative_accepted &&
                !stopped_on_output &&
                ready_token < 0)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_rejection_without_ready_token",
                    1.0,
                    "decode",
                    {},
                    {{"deferred_condition_tokens",
                      std::to_string(deferred_correction_condition_count)}});
            }

            std::optional<std::string> tx_error =
                deferred_accepted_outcome.has_value()
                    ? validate_spec_decode_accepted_outcome(
                          "all_position_state_publication_verifier",
                          "all_position_state_publication",
                          *deferred_accepted_outcome)
                    : validate_spec_decode_transaction(
                          "all_position_state_publication_verifier",
                          "all_position_state_publication",
                          draft_tokens,
                          accepted_tokens,
                          stopped_on_output || ready_token < 0
                              ? std::optional<int32_t>{}
                              : std::optional<int32_t>{ready_token},
                          all_speculative_accepted,
                          stopped_on_output,
                          accepted_speculative_prefix);
            if (tx_error)
            {
                return fail_after_checkpoint(*tx_error);
            }

            ++mtp_stats_.verifier_runs;
            mtp_stats_.verifier_token_count +=
                static_cast<uint64_t>(main_forward_token_count);
            PerfStatsCollector::addCounter("mtp", "verifier_runs", 1.0, "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_tokens",
                static_cast<double>(main_forward_token_count),
                "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "all_position_state_publication_verifier_runs",
                1.0,
                "decode",
                {},
                {{"forward_tokens", std::to_string(main_forward_token_count)},
                 {"verifier_rows", std::to_string(sampled_verifier_rows.size())},
                 {"correction_replay_tokens", std::to_string(correction_forward_count)},
                 {"draft_tokens", std::to_string(draft_tokens.size())},
                 {"accepted_state_count", std::to_string(step.accepted_count)},
                 {"target_cached_tokens", std::to_string(step.target_cached_tokens)},
                 {"restored_verifier_base", restored_verifier_base ? "true" : "false"}});

            recordMTPDepthObservation(
                requested_speculative_draft_count,
                speculative_draft_count,
                accepted_speculative_prefix,
                draft_count_budget_limited,
                /*rollback=*/false);

            if (!all_speculative_accepted)
            {
                ++mtp_stats_.rejected_tokens;
                PerfStatsCollector::addCounter("mtp", "rejected_tokens", 1.0, "decode");
            }

            if (accepted_speculative_prefix > 0)
            {
                mtp_stats_.accepted_tokens +=
                    static_cast<uint64_t>(accepted_speculative_prefix);
                PerfStatsCollector::addCounter(
                    "mtp",
                    "accepted_tokens",
                    static_cast<double>(accepted_speculative_prefix),
                    "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "accepted_second_draft_tokens",
                    accepted_speculative_prefix > 0 ? 1.0 : 0.0,
                    "decode");
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "output_tokens",
                static_cast<double>(accepted_tokens.size()),
                "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "acceptance_trace",
                1.0,
                "decode",
                {},
                {{"draft_step", std::to_string(mtp_stats_.draft_steps)},
                 {"condition_token", std::to_string(condition_token)},
                 {"first_token", std::to_string(first_token)},
                 {"draft_tokens", join_tokens(draft_tokens)},
                 {"verifier_tokens", join_tokens(verifier_tokens)},
                 {"all_position_rows", join_tokens(sampled_verifier_rows)},
                 {"rejected_verified_token", std::to_string(rejected_verified_token)},
                 {"accepted_speculative_prefix", std::to_string(accepted_speculative_prefix)},
                 {"all_speculative_accepted", all_speculative_accepted ? "true" : "false"},
                 {"verifier_state_matches_output", "true"},
                 {"verifier_path", "all_position_state_publication"},
                 {"catchup_implementation", "all_position_state_publication"},
                 {"decode_equivalent_replay_required", "false"},
                 {"correction_replay_tokens", std::to_string(correction_forward_count)},
                 {"deferred_correction_condition_tokens",
                  std::to_string(deferred_correction_condition_count)},
                 {"output_tokens", std::to_string(accepted_tokens.size())},
                 {"ready_token", std::to_string(ready_token)},
                 {"used_ready_logits", use_ready_logits ? "true" : "false"}});

            if (!stopped_on_output &&
                (ready_token >= 0 || verify_commit_replay_check))
            {
                std::ostringstream replay_context;
                replay_context
                    << "draft_tokens=" << join_tokens(draft_tokens)
                    << " verifier_tokens=" << join_tokens(verifier_tokens)
                    << " all_position_rows=" << join_tokens(sampled_verifier_rows)
                    << " accepted_state_count=" << step.accepted_count
                    << " target_cached_tokens=" << step.target_cached_tokens
                    << " main_forward_token_count=" << main_forward_token_count
                    << " all_speculative_accepted="
                    << (all_speculative_accepted ? "true" : "false")
                    << " accepted_speculative_prefix="
                    << accepted_speculative_prefix;
                if (auto mismatch = verify_committed_prefix_replay(
                        "all_position_state_publication_verifier",
                        accepted_tokens,
                        ready_token,
                        replay_context.str()))
                {
                    return fail_after_checkpoint(*mismatch);
                }
            }

            if (auto commit_error = commit_mtp_transaction_outputs(
                    "all_position_state_publication_verifier",
                    verifier_base_checkpoint,
                    accepted_tokens,
                    stopped_on_output || ready_token < 0
                        ? std::optional<int32_t>{}
                        : std::optional<int32_t>{ready_token},
                    /*terminal_logits_ready=*/!stopped_on_output && ready_token >= 0,
                    /*is_complete=*/stopped_on_output,
                    PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent,
                    /*state_advanced=*/true,
                    /*state_advanced_token_count=*/accepted_state_count))
            {
                return fail_after_checkpoint(*commit_error);
            }

            return result;
        }

        const bool use_decode_equivalent_sequential_greedy_verifier =
            verifier_policy.path ==
            MTPVerifierExecutionPath::DecodeEquivalentSequential;
        if (use_decode_equivalent_sequential_greedy_verifier)
        {
            if (sidecar_checkpoints.empty())
            {
                return fail_after_checkpoint(
                    "Decode-equivalent sequential MTP verifier requires a post-sidecar checkpoint");
            }

            const PrefixStateSnapshot &sidecar_checkpoint = sidecar_checkpoints.front();
            if (!sidecar_checkpoint.valid)
            {
                return fail_after_checkpoint(
                    "Decode-equivalent sequential MTP verifier received an invalid post-sidecar checkpoint");
            }

            const bool sidecar_preserves_main_state =
                runner_->supportsMTPSidecarPreservesMainState();
            bool restored_verifier_base = sidecar_preserves_main_state;
            if (sidecar_preserves_main_state)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "decode_equivalent_sequential_verifier_base_restore_skipped_sidecar_preserved",
                    1.0,
                    "decode",
                    {},
                    {{"draft_tokens", std::to_string(draft_tokens.size())},
                     {"cached_tokens", std::to_string(verifier_base_checkpoint.cached_tokens)},
                     {"discarded_sidecar_checkpoint", sidecar_checkpoint.valid ? "true" : "false"}});
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "decode_equivalent_sequential_verifier_restore_base_checkpoint",
                    "decode");
                restored_verifier_base =
                    runner_->restoreLivePrefixState(verifier_base_checkpoint);
                if (restored_verifier_base)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "decode_equivalent_sequential_verifier_base_restores",
                        1.0,
                        "decode",
                        {},
                        {{"draft_tokens", std::to_string(draft_tokens.size())},
                         {"cached_tokens", std::to_string(verifier_base_checkpoint.cached_tokens)},
                         {"discarded_sidecar_checkpoint", sidecar_checkpoint.valid ? "true" : "false"}});
                }
            }
            if (!restored_verifier_base)
            {
                return fail_after_checkpoint(
                    "Decode-equivalent sequential MTP verifier could not restore verifier base checkpoint after sidecar draft");
            }

            if (stochastic_verify)
            {
                if (runner_->primaryDeviceId().is_gpu() && !stochastic_device_verify)
                {
                    return fail_after_checkpoint(
                        "Decode-equivalent stochastic MTP verifier requires device-resident distribution verification");
                }

                std::vector<int32_t> accepted_tokens;
                accepted_tokens.reserve(draft_tokens.size());
                std::vector<int32_t> verifier_tokens;
                verifier_tokens.reserve(draft_tokens.size());

                accepted_tokens.push_back(first_token);
                bool all_speculative_accepted = true;
                bool stopped_on_output = first_token_is_stop;
                int accepted_speculative_prefix = 0;
                int32_t rejected_verified_token = -1;
                int32_t ready_token = -1;
                int main_forward_token_count = 0;
                int shifted_commit_count = 0;
                std::vector<SamplingDistributionEntry> host_target_distribution;

                auto commit_shifted_before_forward =
                    [&](int32_t token, int token_index) -> bool
                {
                    bool ok = false;
                    {
                        PerfStatsCollector::ScopedTimer timer(
                            "mtp",
                            "decode_equivalent_stochastic_shifted_commit",
                            "decode",
                            {},
                            {{"implementation", "shared_stepwise_stochastic"}});
                        ok = runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                            token,
                            token_index,
                            /*allow_speculative_discard=*/true,
                            base_sidecar_position);
                    }
                    if (ok)
                        ++shifted_commit_count;
                    return ok;
                };

                auto forward_one = [&](int32_t token) -> bool
                {
                    int forward_token = static_cast<int>(token);
                    bool ok = false;
                    {
                        PerfStatsCollector::ScopedTimer timer(
                            "mtp",
                            "decode_equivalent_stochastic_forward_one",
                            "decode",
                            {},
                            {{"implementation", "shared_stepwise_stochastic"}});
                        ok = runner_->forward(&forward_token, 1);
                    }
                    if (ok)
                        ++main_forward_token_count;
                    return ok;
                };

                Sampler verifier_penalty_sampler = sampler_;
                auto build_target_distribution = [&]() -> bool
                {
                    if (stochastic_host_verify)
                    {
                        const float *main_logits = runner_->logits();
                        if (!main_logits)
                            return false;
                        PerfStatsCollector::ScopedTimer timer(
                            "mtp",
                            "decode_equivalent_stochastic_host_target_distribution",
                            "decode",
                            {},
                            {{"implementation", "shared_stepwise_stochastic"}});
                        host_target_distribution =
                            verifier_penalty_sampler.compute_distribution(
                                main_logits,
                                static_cast<size_t>(vocab),
                                active_sampling_params_);
                        return !host_target_distribution.empty();
                    }

                    auto penalty_map =
                        verifier_penalty_sampler.compute_penalty_map(
                            active_sampling_params_,
                            vocab);
                    if (!penalty_map.empty() &&
                        !runner_->applyPenaltiesOnDevice(penalty_map, vocab))
                    {
                        return false;
                    }
                    return runner_->buildStochasticDistributionOnDevice(
                        DeviceLogitsSource::Main,
                        0,
                        DeviceDistributionBuffer::Target,
                        0,
                        active_sampling_params_,
                        vocab);
                };

                if (!commit_shifted_before_forward(first_token, 0))
                {
                    return fail_after_checkpoint(
                        "Decode-equivalent stochastic MTP initial shifted-cache commit failed");
                }
                if (!forward_one(first_token))
                {
                    return fail_after_checkpoint(
                        "Decode-equivalent stochastic MTP failed to forward first token");
                }
                verifier_penalty_sampler.record_token(first_token);

                for (int draft_idx = 1;
                     !stopped_on_output &&
                     draft_idx < static_cast<int>(draft_tokens.size());
                     ++draft_idx)
                {
                    if (!build_target_distribution())
                    {
                        return fail_after_checkpoint(
                            "Decode-equivalent stochastic MTP target distribution build failed");
                    }

                    const int row = draft_idx - 1;
                    const int32_t draft_token =
                        draft_tokens[static_cast<size_t>(draft_idx)];
                    const int row_logical_position =
                        checkpoint.cached_tokens + draft_idx;
                    const float accept_threshold =
                        accept_threshold_for_position(
                            sampler_,
                            row_logical_position);
                    const float residual_threshold =
                        residual_threshold_for_position(
                            sampler_,
                            row_logical_position);
                    DeviceSpeculativeVerifyResult verify_result;
                    if (stochastic_device_verify)
                    {
                        if (!runner_->verifyStochasticDistributionsBatchOnDevice(
                                /*first_target_slot=*/0,
                                /*first_draft_slot=*/row,
                                &draft_token,
                                &accept_threshold,
                                &residual_threshold,
                                /*row_count=*/1,
                                &verify_result))
                        {
                            return fail_after_checkpoint(
                                "Decode-equivalent stochastic MTP device verifier failed");
                        }
                    }
                    else
                    {
                        if (row < 0 ||
                            row >= static_cast<int>(host_mtp_draft_distributions.size()) ||
                            host_mtp_draft_distributions[static_cast<size_t>(row)].empty() ||
                            host_target_distribution.empty())
                        {
                            return fail_after_checkpoint(
                                "Decode-equivalent stochastic MTP host verifier missing distributions");
                        }
                        const auto &draft_distribution =
                            host_mtp_draft_distributions[static_cast<size_t>(row)];
                        const float p =
                            Sampler::probability_of_token(host_target_distribution, draft_token);
                        const float q =
                            Sampler::probability_of_token(draft_distribution, draft_token);
                        verify_result.accept_probability =
                            Sampler::speculative_accept_probability(p, q);
                        verify_result.accept_threshold = accept_threshold;
                        verify_result.accepted =
                            accept_threshold < verify_result.accept_probability;
                        verify_result.token = verify_result.accepted
                                                  ? draft_token
                                                  : sampleResidualDistributionWithThreshold(
                                                        host_target_distribution,
                                                        draft_distribution,
                                                        residual_threshold);
                        if (verify_result.token < 0)
                        {
                            return fail_after_checkpoint(
                                "Decode-equivalent stochastic MTP host residual verifier failed");
                        }
                    }

                    ++mtp_stats_.stochastic_accept_tests;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_accept_tests",
                        1.0,
                        "decode",
                        {},
                        {{"row", std::to_string(row)},
                         {"draft_token", std::to_string(draft_token)},
                         {"accept_probability", std::to_string(verify_result.accept_probability)},
                         {"threshold", std::to_string(verify_result.accept_threshold)},
                         {"device_resident", stochastic_device_verify ? "true" : "false"},
                         {"verifier_path", "decode_equivalent_stochastic"}});

                    int32_t output_token = -1;
                    if (verify_result.accepted)
                    {
                        output_token = draft_token;
                        verifier_tokens.push_back(draft_token);
                        ++accepted_speculative_prefix;
                        ++mtp_stats_.stochastic_accepts;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "stochastic_accepts",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path", "decode_equivalent_stochastic"}});
                    }
                    else
                    {
                        output_token = verify_result.token;
                        if (output_token < 0)
                        {
                            return fail_after_checkpoint(
                                "Decode-equivalent stochastic MTP residual verifier produced no correction token");
                        }
                        all_speculative_accepted = false;
                        rejected_verified_token = output_token;
                        verifier_tokens.push_back(output_token);
                        ++mtp_stats_.stochastic_residual_samples;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            stochastic_device_verify
                                ? "stochastic_residual_device_samples"
                                : "stochastic_residual_host_samples",
                            1.0,
                            "decode",
                            {},
                            {{"row", std::to_string(row)},
                             {"draft_token", std::to_string(draft_token)},
                             {"correction_token", std::to_string(output_token)},
                             {"verifier_path", "decode_equivalent_stochastic"}});
                    }

                    accepted_tokens.push_back(output_token);
                    const int token_index =
                        static_cast<int>(accepted_tokens.size()) - 1;
                    if (!commit_shifted_before_forward(output_token, token_index))
                    {
                        return fail_after_checkpoint(
                            "Decode-equivalent stochastic MTP shifted-cache commit failed");
                    }
                    if (!forward_one(output_token))
                    {
                        return fail_after_checkpoint(
                            "Decode-equivalent stochastic MTP failed while forwarding accepted output");
                    }
                    verifier_penalty_sampler.record_token(output_token);

                    if (std::find(stop_tokens_.begin(),
                                  stop_tokens_.end(),
                                  output_token) != stop_tokens_.end())
                    {
                        stopped_on_output = true;
                        break;
                    }
                    if (!verify_result.accepted)
                        break;
                }

                if (!stopped_on_output)
                {
                    if (!build_target_distribution())
                    {
                        return fail_after_checkpoint(
                            "Decode-equivalent stochastic MTP ready-token distribution build failed");
                    }
                    ready_token = stochastic_device_verify
                                      ? runner_->sampleStochasticDistributionOnDevice(
                                            DeviceDistributionBuffer::Target,
                                            0,
                                            sample_threshold_for_position(
                                                sampler_,
                                                checkpoint.cached_tokens +
                                                    static_cast<int>(accepted_tokens.size())))
                                      : sampleDistributionWithThreshold(
                                            host_target_distribution,
                                            sample_threshold_for_position(
                                                sampler_,
                                                checkpoint.cached_tokens +
                                                    static_cast<int>(accepted_tokens.size())));
                    if (ready_token < 0)
                    {
                        return fail_after_checkpoint(
                            "Decode-equivalent stochastic MTP ready-token sampling failed");
                    }
                    if (all_speculative_accepted)
                    {
                        ++mtp_stats_.stochastic_terminal_samples;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            stochastic_device_verify
                                ? "stochastic_terminal_device_samples"
                                : "stochastic_terminal_host_samples",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path", "decode_equivalent_stochastic"}});
                    }
                    else
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "phase138_stochastic_correction_ready_samples",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path", "decode_equivalent_stochastic"}});
                    }
                }

                result.is_complete = result.is_complete || stopped_on_output;

                ++mtp_stats_.verifier_runs;
                mtp_stats_.verifier_token_count +=
                    static_cast<uint64_t>(main_forward_token_count);
                PerfStatsCollector::addCounter("mtp", "verifier_runs", 1.0, "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "verifier_tokens",
                    static_cast<double>(main_forward_token_count),
                    "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "decode_equivalent_stochastic_verifier_runs",
                    1.0,
                    "decode",
                    {},
                    {{"forward_tokens", std::to_string(main_forward_token_count)},
                     {"draft_tokens", std::to_string(draft_tokens.size())},
                     {"accepted_tokens", std::to_string(accepted_tokens.size())},
                     {"shifted_commits", std::to_string(shifted_commit_count)},
                     {"restored_verifier_base", restored_verifier_base ? "true" : "false"}});

                recordMTPDepthObservation(
                    requested_speculative_draft_count,
                    speculative_draft_count,
                    accepted_speculative_prefix,
                    draft_count_budget_limited,
                    !all_speculative_accepted);

                if (!all_speculative_accepted)
                {
                    ++mtp_stats_.rejected_tokens;
                    ++mtp_stats_.rollbacks;
                    ++mtp_stats_.transaction_rollbacks;
                    PerfStatsCollector::addCounter("mtp", "rejected_tokens", 1.0, "decode");
                    PerfStatsCollector::addCounter("mtp", "rollbacks", 1.0, "decode");
                    PerfStatsCollector::addCounter("mtp", "transaction_rollbacks", 1.0, "decode");
                }

                if (accepted_speculative_prefix > 0)
                {
                    mtp_stats_.accepted_tokens +=
                        static_cast<uint64_t>(accepted_speculative_prefix);
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "accepted_tokens",
                        static_cast<double>(accepted_speculative_prefix),
                        "decode");
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "accepted_second_draft_tokens",
                        accepted_speculative_prefix > 0 ? 1.0 : 0.0,
                        "decode");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "output_tokens",
                    static_cast<double>(accepted_tokens.size()),
                    "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "acceptance_trace",
                    1.0,
                    "decode",
                    {},
                    {{"request_epoch", std::to_string(request_epoch_)},
                     {"draft_step", std::to_string(mtp_stats_.draft_steps)},
                     {"condition_token", std::to_string(condition_token)},
                     {"first_token", std::to_string(first_token)},
                     {"draft_tokens", join_tokens(draft_tokens)},
                     {"verifier_tokens", join_tokens(verifier_tokens)},
                     {"rejected_verified_token", std::to_string(rejected_verified_token)},
                     {"accepted_speculative_prefix", std::to_string(accepted_speculative_prefix)},
                     {"all_speculative_accepted", all_speculative_accepted ? "true" : "false"},
                     {"verifier_state_matches_output", "true"},
                     {"verifier_path", "decode_equivalent_stochastic"},
                     {"catchup_implementation", "shared_stepwise_stochastic"},
                     {"decode_equivalent_replay_required", "true"},
                     {"output_tokens", std::to_string(accepted_tokens.size())},
                     {"ready_token", std::to_string(ready_token)},
                     {"used_ready_logits", use_ready_logits ? "true" : "false"}});

                if (!stopped_on_output && ready_token >= 0)
                {
                    if (auto mismatch = verify_committed_prefix_replay(
                            "decode_equivalent_stochastic_verifier",
                            accepted_tokens,
                            ready_token))
                    {
                        return fail_after_checkpoint(*mismatch);
                    }
                }

                if (auto commit_error = commit_mtp_transaction_outputs(
                        "decode_equivalent_stochastic_verifier",
                        verifier_base_checkpoint,
                        accepted_tokens,
                        stopped_on_output || ready_token < 0
                            ? std::optional<int32_t>{}
                            : std::optional<int32_t>{ready_token},
                        /*terminal_logits_ready=*/!stopped_on_output && ready_token >= 0,
                        /*is_complete=*/stopped_on_output,
                        PrefixStateProvenance::DecodeEquivalent,
                        /*state_advanced=*/true))
                {
                    return fail_after_checkpoint(*commit_error);
                }

                return result;
            }

            auto sample_after_forward = [&]() -> int32_t
            {
                int32_t sampled = runner_->sampleGreedyOnDevice();
                if (sampled >= 0)
                    return sampled;

                const float *main_logits = runner_->logits();
                if (!main_logits)
                    return -1;

                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "decode_equivalent_catchup_sample_one_host",
                    "decode",
                    {},
                    {{"implementation", "shared_stepwise"}});
                return sampler_.sample(
                    main_logits,
                    static_cast<size_t>(vocab),
                    active_sampling_params_);
            };

            MTPDecodeCatchupGreedyRequest catchup_request;
            catchup_request.draft_tokens = draft_tokens;
            catchup_request.stop_tokens = stop_tokens_;
            catchup_request.base_sidecar_position = base_sidecar_position;
            catchup_request.allow_speculative_discard = true;
            catchup_request.verifier_path = "decode_equivalent_catchup";
            catchup_request.verifier_base_checkpoint = &verifier_base_checkpoint;

            std::string catchup_implementation = "shared_stepwise";
            MTPDecodeCatchupGreedyResult catchup;
            {
                PerfStatsCollector::ScopedTimer verifier_timer(
                    "mtp",
                    "verifier_forward",
                    "decode",
                    {},
                    {{"implementation", catchup_implementation},
                     {"verifier_path", "decode_equivalent_catchup"}});
                catchup = runSharedStepwiseMTPDecodeCatchupGreedy(
                    *runner_,
                    catchup_request,
                    sample_after_forward);
            }
            if (!catchup.ok)
            {
                return fail_after_checkpoint(catchup.error);
            }

            std::vector<int32_t> accepted_tokens = std::move(catchup.accepted_tokens);
            std::vector<int32_t> verifier_tokens = std::move(catchup.verifier_tokens);
            const bool all_speculative_accepted = catchup.all_speculative_accepted;
            const int accepted_speculative_prefix = catchup.accepted_speculative_prefix;
            const int32_t rejected_verified_token = catchup.rejected_verified_token;
            const int32_t ready_token = catchup.ready_token;
            const bool stopped_on_output = catchup.stopped_on_output;
            const int main_forward_token_count = catchup.main_forward_token_count;
            result.is_complete = result.is_complete || stopped_on_output;

            if (!all_speculative_accepted &&
                !stopped_on_output &&
                ready_token < 0)
            {
                return fail_after_checkpoint(
                    "MTP optimized catch-up returned a rejected transaction without advancing correction state or producing a ready token");
            }

            if (auto tx_error = validate_spec_decode_transaction(
                    "decode_equivalent_sequential_verifier",
                    catchup_implementation,
                    draft_tokens,
                    accepted_tokens,
                    stopped_on_output || ready_token < 0
                        ? std::optional<int32_t>{}
                        : std::optional<int32_t>{ready_token},
                    all_speculative_accepted,
                    stopped_on_output,
                    accepted_speculative_prefix))
            {
                return fail_after_checkpoint(*tx_error);
            }

            ++mtp_stats_.verifier_runs;
            mtp_stats_.verifier_token_count +=
                static_cast<uint64_t>(main_forward_token_count);
            PerfStatsCollector::addCounter("mtp", "verifier_runs", 1.0, "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_tokens",
                static_cast<double>(main_forward_token_count),
                "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "decode_equivalent_sequential_verifier_runs",
                1.0,
                "decode",
                {},
                {{"forward_tokens", std::to_string(main_forward_token_count)},
                 {"draft_tokens", std::to_string(draft_tokens.size())},
                 {"restored_verifier_base", restored_verifier_base ? "true" : "false"},
                 {"catchup_implementation", catchup_implementation}});

            recordMTPDepthObservation(
                requested_speculative_draft_count,
                speculative_draft_count,
                accepted_speculative_prefix,
                draft_count_budget_limited,
                /*rollback=*/false);

            if (!all_speculative_accepted)
            {
                ++mtp_stats_.rejected_tokens;
                PerfStatsCollector::addCounter("mtp", "rejected_tokens", 1.0, "decode");
            }

            if (accepted_speculative_prefix > 0)
            {
                mtp_stats_.accepted_tokens +=
                    static_cast<uint64_t>(accepted_speculative_prefix);
                PerfStatsCollector::addCounter(
                    "mtp",
                    "accepted_tokens",
                    static_cast<double>(accepted_speculative_prefix),
                    "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "accepted_second_draft_tokens",
                    accepted_speculative_prefix > 0 ? 1.0 : 0.0,
                    "decode");
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "output_tokens",
                static_cast<double>(accepted_tokens.size()),
                "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "acceptance_trace",
                1.0,
                "decode",
                {},
                 {{"draft_step", std::to_string(mtp_stats_.draft_steps)},
                  {"condition_token", std::to_string(condition_token)},
                  {"first_token", std::to_string(first_token)},
                  {"draft_tokens", join_tokens(draft_tokens)},
                  {"verifier_tokens", join_tokens(verifier_tokens)},
                  {"rejected_verified_token", std::to_string(rejected_verified_token)},
                  {"accepted_speculative_prefix", std::to_string(accepted_speculative_prefix)},
                  {"all_speculative_accepted", all_speculative_accepted ? "true" : "false"},
                 {"verifier_state_matches_output", "true"},
                 {"verifier_path", "decode_equivalent_catchup"},
                 {"catchup_implementation", catchup_implementation},
                 {"decode_equivalent_replay_required", "true"},
                 {"output_tokens", std::to_string(accepted_tokens.size())},
                 {"ready_token", std::to_string(ready_token)},
                 {"used_ready_logits", use_ready_logits ? "true" : "false"}});

            if (!stopped_on_output && ready_token >= 0)
            {
                if (auto mismatch = verify_committed_prefix_replay(
                        "decode_equivalent_sequential_verifier",
                        accepted_tokens,
                        ready_token))
                {
                    return fail_after_checkpoint(*mismatch);
                }
            }

            if (auto commit_error = commit_mtp_transaction_outputs(
                    "decode_equivalent_sequential_verifier",
                    verifier_base_checkpoint,
                    accepted_tokens,
                    stopped_on_output || ready_token < 0
                        ? std::optional<int32_t>{}
                        : std::optional<int32_t>{ready_token},
                    /*terminal_logits_ready=*/!stopped_on_output && ready_token >= 0,
                    /*is_complete=*/stopped_on_output,
                    PrefixStateProvenance::DecodeEquivalent,
                    /*state_advanced=*/true))
            {
                return fail_after_checkpoint(*commit_error);
            }

            return result;
        }

        return fail_after_checkpoint(
            std::string("MTP verifier policy selected unsupported path: ") +
            verifier_policy.reason);
    }

    GenerationResult OrchestrationRunner::decodeStep()
    {
        GenerationResult result;

        if (!initialized_)
        {
            result.error = "Runner not initialized";
            return result;
        }
        if (batched_decode_active_)
        {
            result.error =
                "decodeStep() cannot consume request-batched prefill state; "
                "use decodeStepBatch()";
            return result;
        }

        // Broadcast to worker ranks so they run decode in lockstep
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
            broadcastCommand(MPICommand::DECODE_STEP);

        const MTPRuntimeConfig &mtp = plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        bool adaptive_depth_zero_step = false;
        if (mtp.enabled)
        {
            const std::string mtp_hard_failure = mtpDecodeHardFailureReason();
            if (!mtp_hard_failure.empty())
            {
                result.error = mtp_hard_failure;
                return result;
            }

            const std::string mtp_bypass_reason = mtpDecodeBypassReason();
            if (mtp_bypass_reason.empty())
            {
                if (!ensureMTPDepthController(mtp))
                {
                    result.error = last_error_.empty()
                                       ? "Invalid MTP depth policy"
                                       : last_error_;
                    return result;
                }
                if (currentMTPDraftDepth(mtp) > 0)
                {
                    return decodeStepMTP();
                }
                recordMTPDepthZeroBypass();
                adaptive_depth_zero_step = true;
            }
            else
            {
                recordMTPBypass(mtp_bypass_reason);
            }
        }

        std::optional<int32_t> ready_token_for_decode;
        const bool can_defer_decode_sampling_sync =
            runner_->primaryDeviceId().is_gpu() &&
            !active_sampling_params_.has_penalties() &&
            (active_sampling_params_.is_greedy() ||
             (active_sampling_params_.top_k > 0 &&
              active_sampling_params_.top_k <= 256));
        bool decode_sampling_sync_deferred = false;
        if (prefill_logits_ready_)
        {
            // First decode step after prefill: sample from the already-computed
            // prefill logits instead of re-feeding the last prompt token.
            // This avoids processing the last token twice (which corrupts GDN
            // recurrence state and creates duplicate KV cache entries).
            if (ready_sampled_token_.has_value())
            {
                if (!ready_sampled_params_.has_value())
                {
                    result.error =
                        "Ready MTP verifier token is missing the sampling parameters that produced it";
                    return result;
                }
                if (!samplingParamsEqual(*ready_sampled_params_, active_sampling_params_))
                {
                    result.error =
                        "Ready MTP verifier token was sampled with different sampling parameters";
                    return result;
                }
                ready_token_for_decode = ready_sampled_token_;
                ready_sampled_token_.reset();
                ready_sampled_params_.reset();
            }
            prefill_logits_ready_ = false;
            LOG_TRACE("[decodeStep] Using prefill logits (skipping forward)");
        }
        else
        {
            ready_sampled_token_.reset();
            ready_sampled_params_.reset();
            LOG_TRACE("[decodeStep] Running forward with last_token_=" << last_token_);
            /*
             * Single-token decode produces logits that are immediately
             * consumed by GPU sampling below.  When the backend can keep that
             * consumer on device, arm the same one-shot stream handoff used by
             * MTP verification so graph replay does not synchronize merely to
             * hand the logits to the next GPU kernel.
             */
            runner_->setMTPMainDecodeSyncDeferralEnabled(
                can_defer_decode_sampling_sync);
            decode_sampling_sync_deferred = can_defer_decode_sampling_sync;
            // Run single-token forward with last token.
            if (!runner_->forward(&last_token_, 1))
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                result.error = "Forward pass failed during decode";
                return result;
            }
        }

        // Tail stage: try GPU-side sampling first, fall back to CPU
        // When penalties are active, compute the sparse penalty map on CPU,
        // upload to GPU, apply in-place, then sample on GPU.
        // This avoids the full ~600KB D2H transfer of logits.
        int token = -1;

        if (ready_token_for_decode.has_value())
        {
            token = *ready_token_for_decode;
            PerfStatsCollector::addCounter("mtp", "ready_token_direct_emits", 1.0, "decode");
        }
        else if (active_sampling_params_.has_penalties())
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
            if (decode_sampling_sync_deferred)
            {
                result.error =
                    "GPU decode sampling failed after deferred logits sync; "
                    "CPU fallback would read unsynchronized logits";
                return result;
            }
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

        const bool token_is_stop =
            std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
        if (adaptive_depth_zero_step && !token_is_stop)
        {
            const int base_sidecar_position = runner_->get_position();
            bool shifted_commit_ok = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "depth_zero_bypass_shifted_commit",
                    "decode");
                shifted_commit_ok =
                    runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                        token,
                        /*already_appended_tokens=*/0,
                        /*allow_speculative_discard=*/true,
                        base_sidecar_position);
            }
            if (!shifted_commit_ok)
            {
                result.error =
                    "MTP dynamic depth-zero shifted-cache maintenance failed";
                return result;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "depth_zero_bypass_shifted_commits",
                1.0,
                "decode");
        }

        // Record token for presence/frequency penalty tracking
        sampler_.record_token(token);

        LOG_TRACE("[decodeStep] sampled token=" << token << " stop_tokens_size=" << stop_tokens_.size());

        result.tokens.push_back(token);
        last_token_ = token; // Store for next decode step

        // Check stop tokens
        result.is_complete = token_is_stop;

        return result;
    }

    void OrchestrationRunner::setDecodeStepTokenBudget(int max_tokens)
    {
        decode_step_token_budget_ = std::max(0, max_tokens);
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

        while (static_cast<int>(result.tokens.size()) < max_new_tokens)
        {
            // Use decodeStep() which uses last_token_ internally
            decode_step_token_budget_ = max_new_tokens - static_cast<int>(result.tokens.size());
            GenerationResult step = decodeStep();
            decode_step_token_budget_ = 0;

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

        const MTPRuntimeConfig &mtp = plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        if (mtp.enabled)
        {
            LOG_INFO("[OrchestrationRunner] MTP summary: draft_steps="
                     << mtp_stats_.draft_steps
                     << " accepted_tokens=" << mtp_stats_.accepted_tokens
                     << " rejected_tokens=" << mtp_stats_.rejected_tokens
                     << " rollbacks=" << mtp_stats_.rollbacks
                     << " bypasses=" << mtp_stats_.bypasses
                     << " last_bypass_reason="
                     << (mtp_bypass_reason_.empty() ? "none" : mtp_bypass_reason_)
                     << " verifier_runs=" << mtp_stats_.verifier_runs
                     << " verifier_tokens=" << mtp_stats_.verifier_token_count
                     << " verify_mode=" << mtpVerifyModeToString(mtp.verify_mode)
                     << " stochastic_accept_tests=" << mtp_stats_.stochastic_accept_tests
                     << " stochastic_residual_samples=" << mtp_stats_.stochastic_residual_samples
                     << " stochastic_terminal_samples=" << mtp_stats_.stochastic_terminal_samples
                     << " depth_policy=" << mtpDepthPolicyModeToString(mtp.depth_policy.mode)
                     << " current_depth="
                     << (mtp_depth_controller_ ? mtp_depth_controller_->currentDepth() : mtp.draft_tokens)
                     << " depth_updates=" << mtp_stats_.depth_policy_updates);
        }

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
        // Request-boundary reset: broadcast to worker ranks so they clear
        // KV/recurrent state in lockstep while preserving reusable graph caches.
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
            broadcastCommand(MPICommand::CLEAR_CACHE);

        if (runner_)
        {
            runner_->clear_cache();
        }
        ++request_epoch_;
#if defined(__GLIBC__)
        ::malloc_trim(0);
#endif
        prefill_logits_ready_ = false;
        ready_sampled_token_.reset();
        ready_sampled_params_.reset();
        clearBatchedDecodeState();
        sampler_ = Sampler(active_sampling_params_.seed);
        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        /*
         * clearCache() is the request boundary used by benchmark iterations
         * and server sessions. Keep adaptive-depth state request-scoped so the
         * reported counters and current depth describe the same request.
         */
        if (mtp_depth_controller_)
        {
            mtp_depth_controller_->reset();
        }
        mtp_stats_ = {};
    }

    PrefixRuntimeStateSnapshot OrchestrationRunner::prefixStateProbe() const
    {
        PrefixRuntimeStateSnapshot snapshot = runner_ ? runner_->prefixStateProbe()
                                                      : PrefixRuntimeStateSnapshot{};
        snapshot.initialized = initialized_;
        snapshot.prefill_logits_ready = prefill_logits_ready_;
        snapshot.current_position = currentPosition();
        const MTPRuntimeConfig &mtp = plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        snapshot.mtp_config_enabled = mtp.enabled;
        snapshot.mtp_bypassed = mtp_bypassed_;
        snapshot.mtp_bypass_reason = mtp_bypass_reason_;
        snapshot.mtp_draft_steps = mtp_stats_.draft_steps;
        snapshot.mtp_accepted_tokens = mtp_stats_.accepted_tokens;
        snapshot.mtp_rejected_tokens = mtp_stats_.rejected_tokens;
        snapshot.mtp_rollbacks = mtp_stats_.rollbacks;
        snapshot.mtp_bypasses = mtp_stats_.bypasses;
        snapshot.mtp_verifier_runs = mtp_stats_.verifier_runs;
        snapshot.mtp_verifier_token_count = mtp_stats_.verifier_token_count;
        snapshot.mtp_stochastic_accept_tests = mtp_stats_.stochastic_accept_tests;
        snapshot.mtp_stochastic_accepts = mtp_stats_.stochastic_accepts;
        snapshot.mtp_stochastic_residual_samples = mtp_stats_.stochastic_residual_samples;
        snapshot.mtp_stochastic_terminal_samples = mtp_stats_.stochastic_terminal_samples;
        snapshot.mtp_transaction_commits = mtp_stats_.transaction_commits;
        snapshot.mtp_transaction_rollbacks = mtp_stats_.transaction_rollbacks;
        snapshot.mtp_transaction_validation_failures =
            mtp_stats_.transaction_validation_failures;
        snapshot.mtp_unsafe_verifier_state_rejections =
            mtp_stats_.unsafe_verifier_state_rejections;
        snapshot.mtp_depth_policy_windows = mtp_stats_.depth_policy_windows;
        snapshot.mtp_depth_policy_updates = mtp_stats_.depth_policy_updates;
        snapshot.mtp_depth_policy_promotions = mtp_stats_.depth_policy_promotions;
        snapshot.mtp_depth_policy_demotions = mtp_stats_.depth_policy_demotions;
        snapshot.mtp_depth_policy_observe_recommendations =
            mtp_stats_.depth_policy_observe_recommendations;
        snapshot.mtp_current_depth =
            mtp_depth_controller_ ? mtp_depth_controller_->currentDepth()
                                  : std::max(0, mtp.draft_tokens);
        snapshot.mtp_min_depth =
            mtp_depth_controller_ ? mtp_depth_controller_->minDepth()
                                  : std::max(0, mtp.draft_tokens);
        snapshot.mtp_max_depth =
            mtp_depth_controller_ ? mtp_depth_controller_->maxDepth()
                                  : std::max(0, mtp.draft_tokens);
        snapshot.prefill_chunk_schedules = prefill_chunk_stats_.schedules;
        snapshot.prefill_chunk_successful_schedules = prefill_chunk_stats_.successful_schedules;
        snapshot.prefill_chunks = prefill_chunk_stats_.chunks;
        snapshot.prefill_chunk_real_tokens = prefill_chunk_stats_.real_tokens;
        snapshot.prefill_chunk_padded_tokens = prefill_chunk_stats_.padded_tokens;
        snapshot.prefill_chunk_failures = prefill_chunk_stats_.failures;
        snapshot.prefix_request = prefix_request_summary_;
        snapshot.mtp_request.enabled = mtp.enabled;
        snapshot.mtp_request.bypassed = mtp_bypassed_;
        snapshot.mtp_request.bypass_reason = mtp_bypass_reason_;
        snapshot.mtp_request.verify_mode = mtpVerifyModeToString(mtp.verify_mode);
        snapshot.mtp_request.stochastic_verify =
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling;
        snapshot.mtp_request.adaptive_depth_enabled =
            mtp.depth_policy.mode != MTPDepthPolicyMode::Fixed;
        snapshot.mtp_request.depth_policy_mode =
            mtpDepthPolicyModeToString(mtp.depth_policy.mode);
        snapshot.mtp_request.current_depth = snapshot.mtp_current_depth;
        snapshot.mtp_request.min_depth = snapshot.mtp_min_depth;
        snapshot.mtp_request.max_depth = snapshot.mtp_max_depth;
        snapshot.mtp_request.depth_policy_updates = mtp_stats_.depth_policy_updates;
        if (mtp_depth_controller_)
        {
            snapshot.mtp_request.last_depth_policy_reason =
                toString(mtp_depth_controller_->lastDecision().reason);
        }
        snapshot.mtp_request.draft_steps = mtp_stats_.draft_steps;
        snapshot.mtp_request.accepted_tokens = mtp_stats_.accepted_tokens;
        snapshot.mtp_request.rejected_tokens = mtp_stats_.rejected_tokens;
        snapshot.mtp_request.rollbacks = mtp_stats_.rollbacks;
        const uint64_t mtp_total_tokens = mtp_stats_.accepted_tokens + mtp_stats_.rejected_tokens;
        snapshot.mtp_request.acceptance_rate =
            mtp_total_tokens > 0
                ? static_cast<double>(mtp_stats_.accepted_tokens) / static_cast<double>(mtp_total_tokens)
                : 0.0;
        snapshot.mtp_request.stochastic_accept_tests = mtp_stats_.stochastic_accept_tests;
        snapshot.mtp_request.stochastic_accepts = mtp_stats_.stochastic_accepts;
        snapshot.mtp_request.stochastic_residual_samples =
            mtp_stats_.stochastic_residual_samples;
        snapshot.mtp_request.stochastic_terminal_samples =
            mtp_stats_.stochastic_terminal_samples;
        snapshot.mtp_request.stochastic_acceptance_rate =
            mtp_stats_.stochastic_accept_tests > 0
                ? static_cast<double>(mtp_stats_.stochastic_accepts) /
                      static_cast<double>(mtp_stats_.stochastic_accept_tests)
                : 0.0;
        if (snapshot.architecture.empty() && model_ctx_)
        {
            snapshot.architecture = model_ctx_->architecture();
        }
        return snapshot;
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
                                    config_.tp_scope == TPScope::NODE_LOCAL ||
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

        LOG_DEBUG("MPI initialized: rank " << mpi_ctx_->rank()
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
            // Hard fail immediately if the model file does not exist — downstream
            // stages (PP layer boundaries, memory planning) require accurate metadata.
            // Falling back to defaults would silently produce invalid configurations.
            std::ifstream probe(config_.model_path, std::ios::binary);
            if (!probe.good())
            {
                return setError("Model file not found: " + config_.model_path);
            }
            probe.close();

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
                return setError("Failed to read model metadata from " + config_.model_path
                                + ": " + e.what());
            }
            if (!metadata_ok)
            {
                return setError("Failed to read model metadata from " + config_.model_path
                                + " (file exists but GGUF parsing failed)");
            }

            const int raw_layers = static_cast<int>(metadata_loader.blockCount());
            model_config.n_layers = mainLayerCountExcludingMTP(
                metadata_loader,
                metadata_loader.architecture(),
                raw_layers);
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
            // No model path (testing only) - use defaults
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
                LOG_DEBUG("[OrchestrationRunner] MoE overlay root plan bound to base domain '"
                          << config_.moe_expert_parallel_plan->effectiveBaseModelDomain()
                          << "' devices=" << plan_.local_tp_devices.size());
            }
            else
            {
                LOG_DEBUG("[OrchestrationRunner] MoE overlay non-root plan narrowed to participant endpoint role "
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

        LOG_DEBUG("LOCAL TP context created with " << plan_.local_tp_devices.size() << " devices");
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

        LOG_DEBUG("LOCAL PP context created with " << pp_config.numStages()
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

        // For GPU targets, skip NUMA mmap binding — weights are uploaded to VRAM,
        // so the host staging mmap doesn't need NUMA placement. This avoids the
        // catastrophic POSIX_FADV_DONTNEED + cold OMP first-touch path that can
        // turn a 4-second model load into a 100+ second ordeal.
        weight_config.target_is_gpu = (plan_.primary_device.device_type != DeviceType::CPU);

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
                    LOG_DEBUG("Node leader (rank " << mpi_ctx_->rank()
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
                    LOG_DEBUG("Pre-populating page cache for multi-rank mmap (rank 0 fallback)...");
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

        LOG_DEBUG("Model context created from: " << model_path
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
                LOG_DEBUG("[OrchestrationRunner] MoE expert overlay plan frozen: placements="
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

        LOG_DEBUG("TP/PP configuration validated against model architecture");
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

        LOG_DEBUG("Context length: " << config_.max_seq_len
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
            cfg.activation_seq_len = resolveActivationBufferSeqLen(cfg.max_seq_len, device);

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

        if (plan_.usesLocalPP())
        {
            // LOCAL PP: each PP stage has its own layer range. Create per-device
            // configs with the correct layer boundaries for each stage.
            const auto &pp_devices = plan_.local_pp_devices;
            const auto &boundaries = plan_.local_pp_layer_boundaries;
            const auto &stage_tp = plan_.local_pp_stage_tp_info;

            for (size_t stage = 0; stage < pp_devices.size(); ++stage)
            {
                int stage_first = boundaries[stage];
                int stage_last = boundaries[stage + 1] - 1;

                // Check if this PP stage has TP composition (multiple devices per stage)
                if (stage < stage_tp.size() && stage_tp[stage].devices.size() > 1)
                {
                    // PP+TP: each device in this stage gets the stage's layer range + TP shard
                    const auto &tp_info = stage_tp[stage];
                    int tp_degree = static_cast<int>(tp_info.devices.size());
                    for (int tp_idx = 0; tp_idx < tp_degree; ++tp_idx)
                    {
                        auto cfg = makeConfigForDevice(
                            tp_info.devices[tp_idx].toLocalDeviceId(),
                            tp_idx, tp_degree);
                        cfg.first_layer = stage_first;
                        cfg.last_layer = stage_last;
                        device_configs.push_back(cfg);
                    }
                }
                else
                {
                    // PP only: single device per stage with that stage's full layer range
                    auto cfg = makeConfigForDevice(
                        pp_devices[stage].toLocalDeviceId(), 0, 1);
                    cfg.first_layer = stage_first;
                    cfg.last_layer = stage_last;
                    device_configs.push_back(cfg);
                }
            }
        }
        else if (plan_.usesLocalTP())
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

        LOG_DEBUG("[MemoryPlanner] Memory validation passed:\n"
                  << plan.renderTable());
        return true;
    }

    void OrchestrationRunner::printStartupBanner()
    {
        // Only rank 0 prints the banner
        if (mpi_ctx_ && mpi_ctx_->rank() != 0)
            return;

        StartupBannerData data;

        // Phase 1: Cluster topology
        data.cluster = &cluster_inventory_;
        if (config_.n_threads > 0)
            data.threads_per_rank = config_.n_threads;
        else
        {
            // cpu_cores is per-socket (this rank's local cores) — use directly as threads/rank
            if (!cluster_inventory_.ranks.empty())
            {
                data.threads_per_rank = cluster_inventory_.ranks[0].cpu_cores;
            }
        }
        data.bind_policy = "socket";

        // Phase 2: Inference configuration
        {
            DeviceId device = DeviceAddressAdapter::toDeviceId(plan_.primary_device);
            std::ostringstream dev_oss;
            dev_oss << device.to_string();
            if (device.is_cpu() && cluster_inventory_.world_size > 1)
            {
                int sockets = cluster_inventory_.ranks[0].cpu_sockets;
                int cores_per_socket = cluster_inventory_.ranks[0].cpu_cores;
                dev_oss << " (" << sockets << "S x " << cores_per_socket << "C, TP=" << cluster_inventory_.world_size << ")";
            }
            data.device_description = dev_oss.str();

            // Parallelism
            std::ostringstream par_oss;
            int effective_tp = plan_.totalTPDegree();
            par_oss << "TP=" << effective_tp;
            if (effective_tp > 1)
            {
                if (config_.tp_scope == TPScope::GLOBAL || config_.cpu_global_tp_all_local)
                    par_oss << " (global)";
                else if (plan_.usesLocalTP())
                    par_oss << " (local)";
            }
            par_oss << " | PP=" << config_.pp_degree;
            data.parallelism = par_oss.str();

            // Precision
            std::ostringstream prec_oss;
            prec_oss << "Activations: FP32";
            if (device.is_cpu())
                prec_oss << " | KV Cache: Q16_1";
            else
                prec_oss << " | KV Cache: FP16";
            data.precision = prec_oss.str();

            // Context length
            std::ostringstream ctx_oss;
            int model_max = model_ctx_ ? static_cast<int>(model_ctx_->contextLength()) : 0;
            ctx_oss << config_.max_seq_len;
            if (model_max > 0)
                ctx_oss << " / " << model_max << " (model max)";
            data.context_length = ctx_oss.str();

            // Backend
            if (device.is_cpu())
                data.backend = "CPU (OneDNN/AVX-512)";
            else if (device.is_cuda())
                data.backend = "CUDA (GPU " + std::to_string(device.ordinal) + ")";
            else if (device.is_rocm())
                data.backend = "ROCm (GPU " + std::to_string(device.ordinal) + ")";
        }

        // Phase 3: Model
        if (model_ctx_)
        {
            // Filename (basename)
            const std::string &path = config_.model_path;
            size_t slash = path.find_last_of('/');
            data.model_filename = (slash != std::string::npos) ? path.substr(slash + 1) : path;

            // File size
            const auto &model = model_ctx_->model();
            size_t file_bytes = 0;
            for (const auto &t : model.tensors)
                file_bytes += t.size_bytes;
            double file_gb = static_cast<double>(file_bytes) / (1024.0 * 1024.0 * 1024.0);
            char size_buf[32];
            snprintf(size_buf, sizeof(size_buf), "%.1f GB", file_gb);
            data.model_size = size_buf;

            // Architecture
            std::ostringstream arch_oss;
            arch_oss << model.architecture << " (" << model.block_count << " layers";
            // Check for MoE
            uint64_t n_experts = 0;
            auto it_experts = model.metadata.find("expert_count");
            if (it_experts != model.metadata.end())
                n_experts = it_experts->second.asUInt64();
            if (n_experts == 0)
            {
                auto it2 = model.metadata.find(model.architecture + ".expert_count");
                if (it2 != model.metadata.end())
                    n_experts = it2->second.asUInt64();
            }
            if (n_experts > 0)
            {
                arch_oss << ", " << n_experts << " experts";
                // top-k
                uint64_t top_k = 0;
                auto it_topk = model.metadata.find("expert_used_count");
                if (it_topk != model.metadata.end())
                    top_k = it_topk->second.asUInt64();
                if (top_k == 0)
                {
                    auto it2 = model.metadata.find(model.architecture + ".expert_used_count");
                    if (it2 != model.metadata.end())
                        top_k = it2->second.asUInt64();
                }
                if (top_k > 0)
                    arch_oss << ", top-" << top_k;
            }
            arch_oss << ")";
            data.architecture = arch_oss.str();

            // Vocab
            std::ostringstream vocab_oss;
            if (model.vocab_size > 0)
            {
                // Format with comma separators
                std::string vs = std::to_string(model.vocab_size);
                std::string formatted;
                int count = 0;
                for (int i = static_cast<int>(vs.size()) - 1; i >= 0; --i)
                {
                    if (count > 0 && count % 3 == 0)
                        formatted = "," + formatted;
                    formatted = vs[static_cast<size_t>(i)] + formatted;
                    count++;
                }
                vocab_oss << formatted << " tokens";
            }
            data.vocab = vocab_oss.str();

            // Thinking model detection
            auto it_think = model.metadata.find("tokenizer.chat_template");
            if (it_think != model.metadata.end())
            {
                const std::string &tmpl = it_think->second.asString();
                if (tmpl.find("<think>") != std::string::npos)
                    data.thinking = "Enabled (<think>...</think>)";
            }
        }

        // Phase 4: Preflight checks (all passed if we got here)
        {
            // Host RAM — we know it passed since we're past validateMemoryPlan
            PreflightCheckResult ram_check;
            ram_check.name = "Host RAM (weight staging)";
            ram_check.passed = true;
            if (model_ctx_)
            {
                const auto &model = model_ctx_->model();
                size_t weight_bytes = 0;
                for (const auto &t : model.tensors)
                    weight_bytes += t.size_bytes;
                double weight_gb = static_cast<double>(weight_bytes) / (1024.0 * 1024.0 * 1024.0);
                char buf[64];
                snprintf(buf, sizeof(buf), "%.1f GB required", weight_gb);
                ram_check.detail = buf;
            }
            data.preflight_checks.push_back(ram_check);

            PreflightCheckResult mem_check;
            mem_check.name = "Device memory (weights + KV + activ.)";
            mem_check.passed = true;
            mem_check.detail = "fits";
            data.preflight_checks.push_back(mem_check);

            PreflightCheckResult schema_check;
            schema_check.name = "Weight schema validation";
            schema_check.passed = true;
            if (model_ctx_)
            {
                char buf[64];
                snprintf(buf, sizeof(buf), "%lu tensors in model",
                         static_cast<unsigned long>(model_ctx_->model().tensor_count));
                schema_check.detail = buf;
            }
            data.preflight_checks.push_back(schema_check);
        }

        // Render and print
        bool use_color = StartupBanner::shouldUseColor();
        std::string banner = StartupBanner::render(data, use_color);

        // Print directly to stderr (bypassing LOG_INFO) to preserve ANSI colors.
        // LOG_INFO strips escape codes via its formatting pipeline.
        if (!banner.empty())
        {
            std::print(stderr, "{}\n", banner);
        }
    }

    bool OrchestrationRunner::buildComputeGraph()
    {
        ScopedWeightLoadTimer timer(WeightLoadPhase::GRAPH_BUILD);

        auto overlay_execution_plan = resolveOverlayExecutionPlanForRunner(
            config_.moe_expert_parallel_plan,
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_);
        if (overlay_execution_plan)
        {
            if (auto blocker = graphNativeMoEOverlayBuildBlocker(*overlay_execution_plan))
                return setError(*blocker);
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

        LOG_DEBUG("[OrchestrationRunner] Execution strategy: MULTI-DEVICE (LOCAL TP)");
        LOG_DEBUG("[OrchestrationRunner]   TP degree: " << plan_.local_tp_devices.size());

        // Log each device
        for (size_t i = 0; i < plan_.local_tp_devices.size(); ++i)
        {
            const auto &dev = plan_.local_tp_devices[i];
            std::string weight_str = "";
            if (i < plan_.local_tp_weights.size())
            {
                weight_str = " (weight=" + std::to_string(plan_.local_tp_weights[i]) + ")";
            }
            LOG_DEBUG("[OrchestrationRunner]   Device " << i << ": " << dev.toString() << weight_str);
        }

        // Build config from execution plan via canonical factory
        auto mdo_config = RankOrchestrator::Config::fromPlan(plan_);
        mdo_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
        mdo_config.moe_expert_overlay_mpi_ctx = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;

        LOG_DEBUG("[OrchestrationRunner] Multi-device precision config: activation="
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

        LOG_DEBUG("Multi-device compute graph built successfully");
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

        LOG_DEBUG("[OrchestrationRunner] Execution strategy: LOCAL PIPELINE PARALLEL");
        LOG_DEBUG("[OrchestrationRunner]   PP stages: " << pp_devices.size());

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
            LOG_DEBUG("[OrchestrationRunner]   Stage " << i << ": "
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

        LOG_DEBUG("Local PP compute graph built successfully");
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
        LOG_DEBUG("[OrchestrationRunner] Execution strategy: SINGLE-DEVICE");
        LOG_DEBUG("[OrchestrationRunner]   Target device: " << device.toString());
        LOG_DEBUG("[OrchestrationRunner]   Device source: " << device_source);
        if (device.is_cpu())
        {
            LOG_DEBUG("[OrchestrationRunner]   Backend: CPU (OneDNN/AVX-512)");
        }
        else if (device.is_cuda())
        {
            LOG_DEBUG("[OrchestrationRunner]   Backend: CUDA (GPU " << device.ordinal << ")");
        }
        else if (device.is_rocm())
        {
            LOG_DEBUG("[OrchestrationRunner]   Backend: ROCm (GPU " << device.ordinal << ")");
        }

        // Build config from execution plan via canonical factory
        auto runner_config = InferenceRunnerConfig::fromPlan(plan_);
        runner_config.hostfile = config_.hostfile;
        runner_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
        runner_config.moe_expert_overlay_mpi_ctx = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;

        LOG_DEBUG("[OrchestrationRunner] Single-device precision config: activation="
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

        LOG_DEBUG("[OrchestrationRunner] Compute graph built successfully");
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
        auto controllers = moeRebalanceControllers();
        return controllers.empty() ? nullptr : controllers.front();
    }

    std::vector<MoERebalanceController *> OrchestrationRunner::moeRebalanceControllers() const
    {
        if (!runner_)
            return {};
        return runner_->moeRebalanceControllers();
    }

    MoERebalanceController *OrchestrationRunner::moeRebalanceControllerForDomain(
        const std::string &domain_id) const
    {
        if (!runner_)
            return nullptr;
        return runner_->moeRebalanceControllerForDomain(domain_id);
    }

    void OrchestrationRunner::applyMoEExpertMasks(
        const std::vector<std::vector<bool>> &masks,
        const ReceivedWeightsMap &received,
        const std::string &domain_id)
    {
        if (runner_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner_.get()))
            {
                dgo->applyExpertMasksForDomain(domain_id, masks, received);
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
        const std::vector<std::vector<std::vector<bool>>> &masks_by_participant,
        const std::string &domain_id)
    {
        if (!runner_)
            return false;
        if (auto *rank = dynamic_cast<RankOrchestrator *>(runner_.get()))
        {
            rank->applyMoEExpertMasksForAllDevices(masks_by_participant, domain_id);
            return true;
        }
        return false;
    }

    void OrchestrationRunner::setExpertReplicaSet(
        const ExpertReplicaSet &replicas, int participant_id)
    {
        if (runner_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner_.get()))
            {
                dgo->setExpertReplicaSetForParticipant(replicas, participant_id);
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

        std::vector<std::vector<std::vector<bool>>> gpu_cache_masks_by_participant;
        const int gpu_cache_experts = debugEnv().moe_rebalance.gpu_cache_experts_per_layer;
        if (gpu_cache_experts > 0)
            gpu_cache_masks_by_participant = controller->computeGpuCacheExpertMasks(gpu_cache_experts);

        const auto old_placement = controller->currentPlacement();
        const ExpertReplicaSet previous_replicas = controller->currentReplicas();
        const bool had_replicas = previous_replicas.num_replicated > 0;
        std::vector<int> new_placement;
        ExpertReplicaSet replica_arrivals;
        bool replica_state_changed = false;

        const int max_replicas = controller->maxReplicasPerSocket();
        if (max_replicas > 0)
        {
            controller->proposeReplicasForParticipants(max_replicas);
            if (controller->hasReplicas())
            {
                const auto &current_replicas = controller->currentReplicas();
                replica_state_changed = !current_replicas.sameReplicaPlacement(previous_replicas);
                replica_arrivals = current_replicas.arrivalsSince(previous_replicas);

                if (!mpi_ctx_ || mpi_ctx_->rank() == 0)
                {
                    LOG_DEBUG("[MoE] Expert replication: "
                              << current_replicas.num_replicated
                              << " experts replicated (cap=" << max_replicas
                              << " per rank/device, hot_cache="
                              << config_.moe_hot_expert_cache.toString() << ")");
                    LOG_DEBUG("[MoE] Keeping base expert ownership stable while applying hot-expert replicas");
                    if (!replica_state_changed)
                    {
                        LOG_DEBUG("[MoE] Hot expert replica set unchanged; skipping replica transfer and mask reapply");
                    }
                    else if (replica_arrivals.num_replicated < current_replicas.num_replicated)
                    {
                        LOG_DEBUG("[MoE] Transferring " << replica_arrivals.num_replicated
                                                        << " newly-arrived hot replicas; "
                                                        << (current_replicas.num_replicated - replica_arrivals.num_replicated)
                                                        << " already resident");
                    }
                }
                controller->resetRebalanceWindow();
            }
            else if (had_replicas)
            {
                replica_state_changed = true;
                controller->resetRebalanceWindow();
                if (!mpi_ctx_ || mpi_ctx_->rank() == 0)
                    LOG_DEBUG("[MoE] Hot expert replica set is now empty; releasing previous replicas");
            }
        }

        if (!controller->hasReplicas())
        {
            new_placement = controller->rebalance();
            controller->syncReplicaPlacement();
        }

        if (controller->hasReplicas() && !replica_state_changed && gpu_cache_masks_by_participant.empty())
            return true;

        if (new_placement.empty() && !controller->hasReplicas() && !replica_state_changed && gpu_cache_masks_by_participant.empty())
            return true;

        ReceivedWeightsMap received;
        if (controller->hasReplicas())
        {
            if (replica_arrivals.num_replicated > 0)
                received = transferReplicaWeights(replica_arrivals, controller->numLayers());
        }
        else if (!new_placement.empty())
        {
            auto manifest = ExpertWeightTransfer::buildManifest(old_placement, new_placement);
            if (!manifest.empty())
                received = transferExpertWeights(manifest, controller->numLayers());
        }

        const int participant_id = runner_ ? runner_->moeRebalanceParticipantId() : 0;
        if (!gpu_cache_masks_by_participant.empty())
        {
            if (!applyMoEExpertMasksForAllLocalDevices(gpu_cache_masks_by_participant, controller->domainId()))
            {
                if (participant_id >= 0 && participant_id < static_cast<int>(gpu_cache_masks_by_participant.size()))
                    applyMoEExpertMasks(gpu_cache_masks_by_participant[participant_id], received, controller->domainId());
            }
        }
        else if (!applyMoEExpertMasksForAllLocalDevices(*controller))
        {
            auto masks = controller->computeExpertMasksForParticipant(participant_id);
            applyMoEExpertMasks(masks, received, controller->domainId());
        }

        if (controller->hasReplicas())
            setExpertReplicaSet(controller->currentReplicas(), participant_id);
        else if (had_replicas && replica_state_changed)
            setExpertReplicaSet(controller->currentReplicas(), participant_id);

        if (config_.moe_rebalance.release_raw_expert_weights || debugEnv().moe_rebalance.release_raw_weights)
        {
            const size_t freed = releaseRawExpertWeights();
            if (!mpi_ctx_ || mpi_ctx_->rank() == 0)
                LOG_DEBUG("[MoE] Released " << (freed >> 20) << " MB raw expert weights");
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
        // Reset token history and deterministic RNG for a new conversation/request.
        sampler_ = Sampler(params.seed);
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

        LOG_DEBUG("[MPI] Rank 0 sending SHUTDOWN to worker ranks");
        broadcastCommand(MPICommand::SHUTDOWN);
    }

    void OrchestrationRunner::runMPIWorkerLoop()
    {
        if (!mpi_ctx_ || mpi_ctx_->rank() == 0)
        {
            LOG_WARN("[MPIWorkerLoop] Should only be called on non-root ranks");
            return;
        }

        LOG_DEBUG("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
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
                LOG_DEBUG("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
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
