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
#include <filesystem>
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

        enum class MTPSpecStochasticDrawPurpose : int
        {
            Sample = 0,
            Accept = 1,
            Residual = 2,
        };

        /**
         * @brief Deterministic per-logical-position stochastic draw.
         *
         * vLLM-style MTP may sample a token as a bonus row in step N or as the
         * first token in step N+1.  Seeded runs must see the same threshold in
         * both cases, so the draw key is the logical output position plus the
         * draw purpose, not the wall-clock point where the code asks for RNG.
         */
        float mtpSpecStochasticThresholdForPosition(
            const SamplingParams &params,
            Sampler &fallback_sampler,
            int logical_position,
            MTPSpecStochasticDrawPurpose purpose)
        {
            if (params.seed == 0)
                return fallback_sampler.random_uniform_01();

            return sampling_math::mtp_spec_threshold_from_seed(
                static_cast<uint64_t>(params.seed),
                logical_position,
                static_cast<int>(purpose));
        }

        uint64_t mtpSpecInverseSampleSeedForThresholds(
            const SamplingParams &params,
            const float *thresholds,
            size_t count)
        {
            if (params.seed != 0)
                return static_cast<uint64_t>(params.seed);

            uint64_t seed = 0xD1B54A32D192ED03ull;
            for (size_t i = 0; i < count; ++i)
            {
                uint32_t bits = 0;
                std::memcpy(&bits, thresholds + i, sizeof(bits));
                seed = sampling_math::splitmix64(
                    seed ^ static_cast<uint64_t>(bits));
            }
            return seed;
        }

        bool traceChatGeneratedTokensEnabled()
        {
            return debugEnv().runtime_debug.trace_generated_tokens;
        }

        const char *perfBool(bool value)
        {
            return value ? "true" : "false";
        }

        bool laneSupportsGroupedDecodeEquivalentOutcome(
            const MTPVerifierEconomyLane &lane,
            int rows,
            bool stochastic_requested)
        {
            return lane.supportsRows(rows, stochastic_requested) &&
                   lane.grouped_decode_equivalent &&
                   lane.row_indexed_lm_head &&
                   lane.device_resident_input &&
                   lane.device_resident_outcome &&
                   lane.graph_capturable;
        }

        /**
         * @brief True when the active model lane has proven grouped verifier
         *        outcomes but may still require replay to publish live state.
         *
         * This deliberately does not check device_resident_publication or
         * host_bridge_free_hot_path.  Those fields describe the stronger
         * economical vLLM-style path; Phase 9.8 needs to represent the earlier
         * milestone where grouped row math is green but live KV/GDN ownership is
         * still being advanced by the safe decode-equivalent replay contract.
         */
        bool supportsGroupedVerifierOutcomeForModel(
            const MTPVerifierEconomyCapability &capability,
            MTPDepthPolicyModelClass model_class,
            int rows,
            bool stochastic_requested)
        {
            switch (model_class)
            {
            case MTPDepthPolicyModelClass::Dense:
                return laneSupportsGroupedDecodeEquivalentOutcome(
                    capability.dense,
                    rows,
                    stochastic_requested);
            case MTPDepthPolicyModelClass::MoE:
                return laneSupportsGroupedDecodeEquivalentOutcome(
                    capability.moe,
                    rows,
                    stochastic_requested);
            case MTPDepthPolicyModelClass::Any:
                return laneSupportsGroupedDecodeEquivalentOutcome(
                           capability.dense,
                           rows,
                           stochastic_requested) ||
                       laneSupportsGroupedDecodeEquivalentOutcome(
                           capability.moe,
                           rows,
                           stochastic_requested);
            }
            return false;
        }

        const char *mtpDepthPolicyModelClassName(MTPDepthPolicyModelClass model_class)
        {
            switch (model_class)
            {
            case MTPDepthPolicyModelClass::Dense:
                return "dense";
            case MTPDepthPolicyModelClass::MoE:
                return "moe";
            case MTPDepthPolicyModelClass::Any:
                return "any";
            }
            return "any";
        }

        /**
         * @brief End-of-command fence for MPI coordinated server commands.
         *
         * The server command stream shares the same communicator as model
         * collectives.  Rank 0 must not publish a new command until every worker
         * has fully left the previous command body, otherwise a worker can
         * consume the next command tag at an inner collective site.  This small
         * guard makes that ownership handoff explicit for commands that do not
         * already end with a protocol broadcast.
         */
        class ScopedMPICoordinatedCommandFence
        {
        public:
            ScopedMPICoordinatedCommandFence(
                const IMPIContext *mpi_ctx,
                bool active,
                const char *label)
                : mpi_ctx_(mpi_ctx), active_(active), label_(label ? label : "unknown")
            {
            }

            ~ScopedMPICoordinatedCommandFence()
            {
                if (!active_ || !mpi_ctx_)
                    return;

                try
                {
                    mpi_ctx_->barrier();
                    if (traceChatGeneratedTokensEnabled())
                    {
                        LOG_INFO("[MPIWorkerLoop] coordinated command fence label="
                                 << label_ << " rank=" << mpi_ctx_->rank());
                    }
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[MPIWorkerLoop] coordinated command fence failed label="
                              << label_ << " rank=" << mpi_ctx_->rank()
                              << ": " << e.what());
                }
                catch (...)
                {
                    LOG_ERROR("[MPIWorkerLoop] coordinated command fence failed label="
                              << label_ << " rank=" << mpi_ctx_->rank()
                              << ": unknown exception");
                }
            }

            ScopedMPICoordinatedCommandFence(const ScopedMPICoordinatedCommandFence &) = delete;
            ScopedMPICoordinatedCommandFence &operator=(const ScopedMPICoordinatedCommandFence &) = delete;

        private:
            const IMPIContext *mpi_ctx_{nullptr};
            bool active_{false};
            const char *label_{"unknown"};
        };

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
        ready_sampled_resident_state_.reset();
        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
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
                ready_sampled_resident_state_.reset();
                pending_mtp_condition_token_.reset();
                pending_mtp_condition_params_.reset();
                pending_mtp_condition_resident_state_.reset();
                prelaunched_mtp_first_sidecar_resident_state_.reset();
                prelaunched_mtp_first_sidecar_params_.reset();

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
            state.sampler = Sampler(active_sampling_params_.seed);
            next_states.push_back(std::move(state));
        }

        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        mtp_stats_ = {};
        prefix_request_summary_ = {};
        ready_sampled_token_.reset();
        ready_sampled_params_.reset();
        ready_sampled_resident_state_.reset();
        prefill_logits_ready_ = false;
        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
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

    bool OrchestrationRunner::supportsDecodeStepBatch(int request_batch) const
    {
        if (!initialized_ || !runner_ || request_batch <= 1)
            return false;
        if (!batched_decode_active_)
            return false;
        if (static_cast<int>(batched_request_states_.size()) != request_batch)
            return false;
        if (runner_->vocab_size() <= 0)
            return false;

        bool has_ready_prefill_logits = false;
        bool has_verifier_continuation = false;
        bool has_completed_requests = false;
        for (const BatchedDecodeRequestState &state : batched_request_states_)
        {
            if (state.is_complete)
            {
                has_completed_requests = true;
                continue;
            }
            has_ready_prefill_logits =
                has_ready_prefill_logits || state.prefill_logits_ready;
            has_verifier_continuation =
                has_verifier_continuation || !state.prefill_logits_ready;
        }
        if (has_ready_prefill_logits && has_verifier_continuation)
            return false;
        if (has_ready_prefill_logits)
            return true;

        const MTPRuntimeConfig &mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        const int draft_depth = effectiveMTPMaxDraftDepth(mtp);
        const bool greedy_batch_verify =
            mtp.verify_mode == MTPVerifyMode::Greedy &&
            active_sampling_params_.is_greedy();
        const bool stochastic_batch_verify =
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            !active_sampling_params_.is_greedy() &&
            !active_sampling_params_.has_penalties() &&
            runner_->supportsDeviceStochasticMTPVerification();
        const bool stochastic_publication_ok =
            !stochastic_batch_verify ||
            !runner_->primaryDeviceId().is_gpu() ||
            runner_->supportsDeviceResidentMTPSpecStatePublication();
        const bool chained_ok =
            draft_depth <= 1 || runner_->supportsChainedMTPDrafts();
        return has_verifier_continuation &&
               !has_completed_requests &&
               mtp.enabled &&
               (greedy_batch_verify || stochastic_batch_verify) &&
               draft_depth >= 1 &&
               draft_depth <= 3 &&
               chained_ok &&
               stochastic_publication_ok &&
               runner_->supportsMTPSpecStatePublication();
    }

    GenerationBatchResult OrchestrationRunner::decodeStepBatch(int request_batch)
    {
        GenerationBatchResult batch_result;

        if (!initialized_)
        {
            batch_result.error = "Runner not initialized";
            return batch_result;
        }
        if (!runner_)
        {
            batch_result.error = "Runner unavailable";
            return batch_result;
        }
        if (request_batch <= 1)
        {
            batch_result.error =
                "decodeStepBatch() requires at least two logical requests";
            return batch_result;
        }
        if (!batched_decode_active_)
        {
            batch_result.error =
                "decodeStepBatch() requires a preceding prefillBatch() call";
            return batch_result;
        }
        if (static_cast<int>(batched_request_states_.size()) != request_batch)
        {
            batch_result.error =
                "decodeStepBatch() request count does not match active "
                "request-batched prefill state";
            return batch_result;
        }

        const int vocab = vocabSize();
        if (vocab <= 0)
        {
            batch_result.error = "decodeStepBatch() requires a positive vocabulary size";
            return batch_result;
        }

        const std::vector<int> &sequence_lengths = runner_->sequence_lengths();
        const int padded_seq_len = runner_->padded_seq_len();
        if (static_cast<int>(sequence_lengths.size()) < request_batch ||
            padded_seq_len <= 0)
        {
            batch_result.error =
                "decodeStepBatch() requires per-request batch sequence metadata";
            return batch_result;
        }

        batch_result.requests.resize(static_cast<size_t>(request_batch));

        bool has_ready_prefill_logits = false;
        bool has_verifier_continuation = false;
        bool has_completed_requests = false;
        for (int request = 0; request < request_batch; ++request)
        {
            const BatchedDecodeRequestState &state =
                batched_request_states_[static_cast<size_t>(request)];
            if (state.is_complete)
            {
                has_completed_requests = true;
                continue;
            }
            has_ready_prefill_logits =
                has_ready_prefill_logits || state.prefill_logits_ready;
            has_verifier_continuation =
                has_verifier_continuation || !state.prefill_logits_ready;
        }

        if (has_ready_prefill_logits && has_verifier_continuation)
        {
            batch_result.error =
                "decodeStepBatch() cannot mix terminal-prefill sampling and "
                "MTP verifier continuation in one live batch";
            return batch_result;
        }

        if (has_ready_prefill_logits)
        {
            bool needs_device_prefill_sampling = false;
            std::vector<int32_t> device_prefill_tokens(
                static_cast<size_t>(request_batch),
                kMTPSpecDecodeInvalidToken);
            std::vector<float> device_prefill_thresholds;

            if (runner_->primaryDeviceId().is_gpu())
            {
                for (int request = 0; request < request_batch; ++request)
                {
                    const BatchedDecodeRequestState &state =
                        batched_request_states_[static_cast<size_t>(request)];
                    if (!state.is_complete && !state.ready_sampled_token.has_value())
                    {
                        needs_device_prefill_sampling = true;
                        break;
                    }
                }

                if (needs_device_prefill_sampling &&
                    !active_sampling_params_.is_greedy())
                {
                    device_prefill_thresholds.resize(
                        static_cast<size_t>(request_batch), 0.0f);
                    for (int request = 0; request < request_batch; ++request)
                    {
                        BatchedDecodeRequestState &state =
                            batched_request_states_[static_cast<size_t>(request)];
                        const int logical_position =
                            sequence_lengths[static_cast<size_t>(request)];
                        /*
                         * Keep request-batched first-token sampling identical
                         * to scalar stochastic MTP. The draw is keyed by the
                         * logical output position, not by "row N in this batch",
                         * so a duplicate prompt in a request batch sees the
                         * same first-token threshold as a standalone request.
                         */
                        device_prefill_thresholds[static_cast<size_t>(request)] =
                            mtpSpecStochasticThresholdForPosition(
                                active_sampling_params_,
                                state.sampler,
                                logical_position,
                                MTPSpecStochasticDrawPurpose::Sample);
                    }
                }

                if (needs_device_prefill_sampling &&
                    !runner_->sampleMainLogitsBatchRowsOnDevice(
                        request_batch,
                        active_sampling_params_,
                        device_prefill_tokens.data(),
                        device_prefill_thresholds.empty()
                            ? nullptr
                            : device_prefill_thresholds.data()))
                {
                    batch_result.error =
                        "decodeStepBatch() could not sample request-batched "
                        "prefill logits on device";
                    return batch_result;
                }
            }

            for (int request = 0; request < request_batch; ++request)
            {
                BatchedDecodeRequestState &state =
                    batched_request_states_[static_cast<size_t>(request)];
                GenerationResult &request_result =
                    batch_result.requests[static_cast<size_t>(request)];

                if (state.is_complete)
                {
                    request_result.is_complete = true;
                    continue;
                }
                if (!state.prefill_logits_ready)
                {
                    batch_result.error =
                        "decodeStepBatch() expected every active request to "
                        "own terminal prefill logits";
                    return batch_result;
                }

                int token = kMTPSpecDecodeInvalidToken;
                if (state.ready_sampled_token.has_value())
                {
                    if (!state.ready_sampled_params.has_value())
                    {
                        batch_result.error =
                            "decodeStepBatch() ready token is missing the "
                            "sampling parameters that produced it";
                        return batch_result;
                    }
                    if (!samplingParamsEqual(
                            *state.ready_sampled_params,
                            active_sampling_params_))
                    {
                        batch_result.error =
                            "decodeStepBatch() ready token was sampled with "
                            "different sampling parameters";
                        return batch_result;
                    }
                    token = *state.ready_sampled_token;
                }
                else if (runner_->primaryDeviceId().is_gpu())
                {
                    token = device_prefill_tokens[static_cast<size_t>(request)];
                    if (token < 0)
                    {
                        batch_result.error =
                            "decodeStepBatch() device prefill sampler returned "
                            "an invalid token";
                        return batch_result;
                    }
                }
                else
                {
                    const int logical_length = sequence_lengths[static_cast<size_t>(request)];
                    if (logical_length <= 0 || logical_length > padded_seq_len)
                    {
                        batch_result.error =
                            "decodeStepBatch() received invalid per-request sequence length";
                        return batch_result;
                    }

                    const float *sequence_logits = runner_->getLogits(request);
                    if (!sequence_logits)
                    {
                        batch_result.error =
                            "decodeStepBatch() could not access per-request logits";
                        return batch_result;
                    }

                    const float *terminal_logits =
                        sequence_logits +
                        static_cast<size_t>(logical_length - 1) *
                            static_cast<size_t>(vocab);
                    token = state.sampler.sample(
                        terminal_logits,
                        static_cast<size_t>(vocab),
                        active_sampling_params_);
                }

                state.prefill_logits_ready = false;
                state.ready_sampled_token.reset();
                state.ready_sampled_params.reset();
                state.last_token = token;

                request_result.tokens.push_back(token);
                request_result.is_complete =
                    std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
                    stop_tokens_.end();
                state.is_complete = request_result.is_complete;

                state.sampler.record_token(token);
            }

            return batch_result;
        }

        if (!has_verifier_continuation)
            return batch_result;
        if (has_completed_requests)
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires every request lane to be active";
            return batch_result;
        }

        const MTPRuntimeConfig &mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        if (!mtp.enabled)
        {
            batch_result.error =
                "decodeStepBatch() MTP verifier continuation requires MTP";
            return batch_result;
        }
        recordMTPVerifierEconomyPerfStatsIfNeeded();
        const bool greedy_batch_verify =
            mtp.verify_mode == MTPVerifyMode::Greedy &&
            active_sampling_params_.is_greedy();
        const bool stochastic_batch_verify =
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            !active_sampling_params_.is_greedy();
        if (!greedy_batch_verify && !stochastic_batch_verify)
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires greedy verification or stochastic speculative sampling";
            return batch_result;
        }
        if (stochastic_batch_verify && active_sampling_params_.has_penalties())
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "does not yet support penalty-mutated stochastic sampling";
            return batch_result;
        }
        if (stochastic_batch_verify &&
            !runner_->supportsDeviceStochasticMTPVerification())
        {
            batch_result.error =
                "decodeStepBatch() stochastic request batching requires "
                "device-resident stochastic MTP verification";
            return batch_result;
        }
        if (greedy_batch_verify && active_sampling_params_.has_penalties())
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "does not yet support penalty-mutated greedy sampling";
            return batch_result;
        }
        const int draft_depth = effectiveMTPMaxDraftDepth(mtp);
        if (draft_depth < 1 || draft_depth > 3)
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires --mtp-draft-tokens in [1, 3]";
            return batch_result;
        }
        if (draft_depth > 1 && !runner_->supportsChainedMTPDrafts())
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires batched chained MTP draft support for depth > 1";
            return batch_result;
        }
        if (!runner_->supportsMTPSpecStatePublication())
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires runner MTP spec-state publication";
            return batch_result;
        }

        if (!runner_->ensureMTPCheckpointTerminalHidden())
        {
            batch_result.error =
                "decodeStepBatch() could not materialize MTP checkpoint terminal hidden";
            return batch_result;
        }
        PrefixStateSnapshot checkpoint = runner_->captureLivePrefixCheckpoint();
        if (!checkpoint.valid)
        {
            batch_result.error =
                "decodeStepBatch() could not capture live prefix checkpoint";
            return batch_result;
        }

        auto fail_after_checkpoint =
            [&](const std::string &message) -> GenerationBatchResult
        {
            runner_->setComputeAllPositionLogits(false);
            runner_->setComputeRowIndexedAllPositionLogits(false, 0);
            std::string restored_suffix;
            if (!runner_->restoreLivePrefixState(checkpoint))
                restored_suffix = "; checkpoint restore failed";
            batch_result.error = message + restored_suffix;
            return batch_result;
        };

        std::vector<int32_t> condition_tokens(
            static_cast<size_t>(request_batch),
            kMTPSpecDecodeInvalidToken);
        std::vector<int> position_ids(
            static_cast<size_t>(request_batch),
            -1);
        std::vector<int32_t> sidecar_drafts(
            static_cast<size_t>(request_batch),
            kMTPSpecDecodeInvalidToken);
        std::vector<std::vector<int32_t>> request_drafts(
            static_cast<size_t>(request_batch));
        const bool use_request_batch_device_draft_slots =
            stochastic_batch_verify &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsDeviceStochasticMTPVerification();
        for (int request = 0; request < request_batch; ++request)
        {
            const BatchedDecodeRequestState &state =
                batched_request_states_[static_cast<size_t>(request)];
            if (state.is_complete)
            {
                batch_result.requests[static_cast<size_t>(request)].is_complete = true;
                continue;
            }
            const int logical_length = sequence_lengths[static_cast<size_t>(request)];
            if (logical_length <= 0)
            {
                batch_result.error =
                    "decodeStepBatch() received invalid per-request sequence length";
                return batch_result;
            }
            condition_tokens[static_cast<size_t>(request)] = state.last_token;
            position_ids[static_cast<size_t>(request)] = logical_length;
        }

        /*
         * One true sidecar graph launch amortizes the first MTP draft across
         * the request batch. Completed rows are still given a harmless token so
         * the runner sees a dense request-batch shape; their results are ignored
         * because they are not enqueued into the owner below.
         */
        for (int request = 0; request < request_batch; ++request)
        {
            if (condition_tokens[static_cast<size_t>(request)] ==
                kMTPSpecDecodeInvalidToken)
            {
                condition_tokens[static_cast<size_t>(request)] = 0;
                position_ids[static_cast<size_t>(request)] =
                    std::max(0, sequence_lengths[static_cast<size_t>(request)]);
            }
        }
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                use_request_batch_device_draft_slots
                    ? "request_batch_sidecar_forward_device_drafts"
                    : "request_batch_sidecar_forward",
                "decode");
            const bool sidecar_ok =
                use_request_batch_device_draft_slots
                    ? runner_->forwardMTPBatchAndSampleGreedyToDeviceDraftSlots(
                          condition_tokens.data(),
                          position_ids.data(),
                          request_batch,
                          /*first_draft_slot=*/0,
                          /*slot_stride=*/draft_depth,
                          sidecar_drafts.data())
                    : runner_->forwardMTPBatchAndSampleGreedy(
                          condition_tokens.data(),
                          position_ids.data(),
                          request_batch,
                          sidecar_drafts.data());
            if (!sidecar_ok)
            {
                return fail_after_checkpoint(
                    use_request_batch_device_draft_slots
                        ? "decodeStepBatch() request-batched MTP sidecar could not produce device draft slots"
                        : "decodeStepBatch() request-batched MTP sidecar failed");
            }
        }
        for (int request = 0; request < request_batch; ++request)
        {
            const int32_t draft = sidecar_drafts[static_cast<size_t>(request)];
            if (draft < 0)
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() request-batched MTP sidecar produced "
                    "an invalid first draft token");
            }
            request_drafts[static_cast<size_t>(request)].push_back(draft);
        }

        for (int draft_index = 1; draft_index < draft_depth; ++draft_index)
        {
            for (int request = 0; request < request_batch; ++request)
            {
                condition_tokens[static_cast<size_t>(request)] =
                    request_drafts[static_cast<size_t>(request)].back();
                position_ids[static_cast<size_t>(request)] =
                    sequence_lengths[static_cast<size_t>(request)] +
                    draft_index;
                sidecar_drafts[static_cast<size_t>(request)] =
                    kMTPSpecDecodeInvalidToken;
            }

            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "request_batch_chained_sidecar_forward",
                "decode",
                {},
                {{"draft_index", std::to_string(draft_index)}});
            const bool chained_sidecar_ok =
                use_request_batch_device_draft_slots
                    ? runner_->forwardMTPBatchFromLastDraftAndSampleGreedyToDeviceDraftSlots(
                          condition_tokens.data(),
                          position_ids.data(),
                          request_batch,
                          /*first_draft_slot=*/draft_index,
                          /*slot_stride=*/draft_depth,
                          sidecar_drafts.data())
                    : runner_->forwardMTPBatchFromLastDraftAndSampleGreedy(
                          condition_tokens.data(),
                          position_ids.data(),
                          request_batch,
                          sidecar_drafts.data());
            if (!chained_sidecar_ok)
            {
                return fail_after_checkpoint(
                    use_request_batch_device_draft_slots
                        ? "decodeStepBatch() request-batched chained MTP sidecar could not produce device draft slots"
                        : "decodeStepBatch() request-batched chained MTP sidecar failed");
            }
            for (int request = 0; request < request_batch; ++request)
            {
                const int32_t draft =
                    sidecar_drafts[static_cast<size_t>(request)];
                if (draft < 0)
                {
                    return fail_after_checkpoint(
                        "decodeStepBatch() request-batched chained MTP "
                        "sidecar produced an invalid draft token");
                }
                request_drafts[static_cast<size_t>(request)].push_back(draft);
            }
        }

        if (!runner_->flushPendingMTPWork())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() request-batched MTP sidecar flush failed");
        }

        MTPSpecRequestBatchOwner owner;
        const std::string compatibility_key =
            std::string("singledevice:") + runner_->architecture() +
            (stochastic_batch_verify ? ":stochastic:d" : ":greedy:d") +
            std::to_string(draft_depth);
        for (int request = 0; request < request_batch; ++request)
        {
            const BatchedDecodeRequestState &state =
                batched_request_states_[static_cast<size_t>(request)];
            if (state.is_complete)
                continue;
            const int32_t draft =
                sidecar_drafts[static_cast<size_t>(request)];
            if (draft < 0)
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() request-batched MTP sidecar produced "
                    "an invalid draft token");
            }

            MTPSpecSchedulableRequest pending;
            pending.request_id = request;
            pending.ready = true;
            pending.mode = stochastic_batch_verify
                               ? MTPSpecRequestBatchMode::STOCHASTIC
                               : MTPSpecRequestBatchMode::GREEDY;
            pending.verifier_input = MTPSpecVerifierInputPlacement::HOST_TOKENS;
            pending.compatibility_key = compatibility_key;
            pending.vocab_size = vocab;
            pending.base_cached_tokens =
                static_cast<int32_t>(sequence_lengths[static_cast<size_t>(request)]);
            pending.requires_shifted_kv_publication = true;
            pending.greedy_request.draft_tokens.clear();
            pending.greedy_request.draft_tokens.reserve(
                static_cast<size_t>(draft_depth) + 1u);
            pending.greedy_request.draft_tokens.push_back(state.last_token);
            pending.greedy_request.draft_tokens.insert(
                pending.greedy_request.draft_tokens.end(),
                request_drafts[static_cast<size_t>(request)].begin(),
                request_drafts[static_cast<size_t>(request)].end());
            pending.greedy_request.stop_tokens = stop_tokens_;
            pending.greedy_request.base_sidecar_position =
                sequence_lengths[static_cast<size_t>(request)];
            pending.greedy_request.verifier_path =
                "request_batched_all_position_state_publication";
            pending.greedy_request.implementation_name =
                stochastic_batch_verify
                    ? "request_batched_stochastic_d" + std::to_string(draft_depth)
                    : "request_batched_greedy_d" + std::to_string(draft_depth);

            std::string enqueue_error;
            if (!owner.enqueueRequest(std::move(pending), &enqueue_error))
            {
                return fail_after_checkpoint(
                    std::string("decodeStepBatch() could not enqueue MTP request: ") +
                    enqueue_error);
            }
        }

        MTPSpecRequestBatchScheduler scheduler(
            MTPSpecRequestBatchSchedulerConfig{
                .max_request_batch = request_batch,
                .max_draft_tokens = draft_depth + 1,
                .mode = stochastic_batch_verify
                            ? MTPSpecRequestBatchMode::STOCHASTIC
                            : MTPSpecRequestBatchMode::GREEDY});

        const bool use_device_resident_request_batch_publication =
            stochastic_batch_verify && runner_->primaryDeviceId().is_gpu();
        if (use_device_resident_request_batch_publication &&
            !runner_->supportsDeviceResidentMTPSpecStatePublication())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() GPU stochastic request batching requires "
                "device-resident MTP state publication");
        }

        DeviceSpeculativeOutcomeHandle resident_request_batch_outcome;
        bool resident_request_batch_outcome_ready = false;
        int resident_request_batch_verifier_rows = 0;
        std::vector<DeviceStochasticBatchOutcomeRequest>
            resident_request_batch_outcome_requests;
        std::vector<Sampler> resident_request_batch_bonus_samplers;
        std::vector<std::optional<Sampler>> sampled_terminal_samplers(
            static_cast<size_t>(request_batch));

        auto publish = [&](const MTPSpecTransactionBatchPlan &plan,
                           std::string *error) -> bool
        {
            if (plan.requiresDecodeEquivalentReplayPublication())
            {
                if (error)
                {
                    *error =
                        "request-batched MTP direct publication received a replay-required transaction plan";
                    if (!plan.publication_contract_reason.empty())
                    {
                        *error += ": ";
                        *error += plan.publication_contract_reason;
                    }
                }
                return false;
            }
            return runner_->publishAcceptedMTPSpecStateBatch(
                plan.step_plans,
                error);
        };

        auto process_stochastic_host_outcomes =
            [&](const std::vector<DeviceStochasticBatchOutcomeRequest> &outcome_requests,
                const std::vector<Sampler> &bonus_samplers,
                const std::vector<MTPDeviceRejectionBatchOutcome> &outcomes,
                std::string *error) -> bool
        {
            auto set_error = [&](std::string message) -> bool
            {
                if (error)
                    *error = std::move(message);
                return false;
            };

            if (outcome_requests.size() != outcomes.size())
            {
                return set_error(
                    "stochastic request-batch outcome vector size mismatch");
            }

            for (size_t i = 0; i < outcome_requests.size(); ++i)
            {
                const int request_id = outcome_requests[i].request_id;
                if (request_id < 0 || request_id >= request_batch)
                {
                    return set_error(
                        "stochastic request-batch outcome descriptor has an out-of-range request id");
                }
                if (static_cast<size_t>(request_id) >= bonus_samplers.size())
                {
                    return set_error(
                        "stochastic request-batch bonus sampler vector is undersized");
                }
                if (outcomes[i].sampled_terminal)
                {
                    sampled_terminal_samplers[static_cast<size_t>(request_id)] =
                        bonus_samplers[static_cast<size_t>(request_id)];
                }

                const int physical_rows =
                    std::max(0, outcome_requests[i].row_count);
                const int semantic_rows =
                    std::min(
                        physical_rows,
                        std::max(0, outcomes[i].consumed_verifier_rows));
                const int post_reject_rows =
                    std::max(0, physical_rows - semantic_rows);
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_device_physical_verify_rows",
                    static_cast<double>(physical_rows),
                    "decode",
                    {},
                    {{"implementation", "request_batch_device_outcome"},
                     {"request_batch", "true"}});
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_device_semantic_verify_rows",
                    static_cast<double>(semantic_rows),
                    "decode",
                    {},
                    {{"implementation", "request_batch_device_outcome"},
                     {"request_batch", "true"}});
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_device_post_reject_rows",
                    static_cast<double>(post_reject_rows),
                    "decode",
                    {},
                    {{"implementation", "request_batch_device_outcome"},
                     {"request_batch", "true"}});
            }
            return true;
        };

        std::vector<int> scheduled_request_ids;
        std::vector<MTPDecodeCatchupGreedyResult> catchup_results;

        if (!stochastic_batch_verify)
        {
            MTPOwnedGreedyVerifierBatchTransactionResult tx =
                executeOwnedMTPGreedyVerifierScheduledBatchTransactionAndPublish(
                    *runner_,
                    owner,
                    scheduler,
                    publish);
            if (!tx.ok)
            {
                return fail_after_checkpoint(
                    std::string("decodeStepBatch() request-batched verifier "
                                "transaction failed: ") +
                    tx.error);
            }

            scheduled_request_ids = tx.scheduled_batch.request_ids;
            catchup_results = tx.transaction.catchup.results;
        }
        else
        {
            auto produce_stochastic_outcomes =
                [&](const MTPSpecRequestBatch &scheduled_batch,
                    std::vector<MTPDeviceRejectionBatchOutcome> *outcomes,
                    std::string *error) -> bool
            {
                auto set_producer_error = [&](std::string message) -> bool
                {
                    if (error)
                        *error = std::move(message);
                    return false;
                };

                if (!outcomes)
                    return set_producer_error("stochastic outcome output vector is null");
                if (!scheduled_batch.ok)
                    return set_producer_error("scheduled stochastic batch is invalid");
                if (scheduled_batch.request_count <= 0)
                    return set_producer_error("scheduled stochastic batch is empty");

                resident_request_batch_outcome = {};
                resident_request_batch_outcome_ready = false;
                resident_request_batch_verifier_rows = 0;
                resident_request_batch_outcome_requests.clear();
                resident_request_batch_bonus_samplers.clear();

                std::vector<MTPSpecDecodeVerifierDraftRequest> verifier_requests;
                verifier_requests.reserve(scheduled_batch.greedy_requests.size());
                for (size_t i = 0; i < scheduled_batch.greedy_requests.size(); ++i)
                {
                    MTPSpecDecodeVerifierDraftRequest request;
                    request.request_id = scheduled_batch.request_ids[i];
                    request.draft_tokens =
                        scheduled_batch.greedy_requests[i].draft_tokens;
                    verifier_requests.push_back(std::move(request));
                }

                MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                    buildMTPSpecDecodeVerifierInputPlan(
                        scheduled_batch.shape,
                        verifier_requests);
                if (!verifier_input_plan.ok)
                {
                    return set_producer_error(
                        std::string("stochastic request-batch verifier input plan failed: ") +
                        verifier_input_plan.error);
                }
                if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                {
                    return set_producer_error(
                        "stochastic request-batch verifier row metadata is malformed");
                }

                bool row_indexed_enabled = false;
                bool all_position_enabled = false;
                auto cleanup_row_modes = [&]() -> bool
                {
                    bool ok = true;
                    runner_->clearMTPSpecVerifierInputPlan();
                    if (all_position_enabled)
                    {
                        all_position_enabled = false;
                        ok = runner_->setComputeAllPositionLogits(false) && ok;
                    }
                    if (row_indexed_enabled)
                    {
                        row_indexed_enabled = false;
                        ok = runner_->setComputeRowIndexedAllPositionLogits(false, 0) && ok;
                    }
                    return ok;
                };

                const int compact_row_count =
                    verifier_input_plan.compact_logit_row_count;
                if (!runner_->setComputeRowIndexedAllPositionLogits(
                        true,
                        compact_row_count))
                {
                    return set_producer_error(
                        "stochastic request-batch verifier could not enable row-indexed logits");
                }
                row_indexed_enabled = true;
                if (!runner_->setMTPSpecVerifierInputPlan(verifier_input_plan))
                {
                    const bool cleanup_ok = cleanup_row_modes();
                    return set_producer_error(
                        cleanup_ok
                            ? "stochastic request-batch verifier could not install row plan"
                            : "stochastic request-batch verifier could not install row plan and cleanup failed");
                }
                if (!runner_->setComputeAllPositionLogits(true))
                {
                    const bool cleanup_ok = cleanup_row_modes();
                    return set_producer_error(
                        cleanup_ok
                            ? "stochastic request-batch verifier could not enable all-position logits"
                            : "stochastic request-batch verifier could not enable all-position logits and cleanup failed");
                }
                all_position_enabled = true;

                {
                    ScopedMTPAllPositionVerifierSyncDeferral verifier_sync_deferral(
                        runner_.get(),
                        runner_->primaryDeviceId().is_gpu());
                    PerfStatsCollector::ScopedTimer verifier_timer(
                        "mtp",
                        "request_batch_stochastic_verifier_forward",
                        "decode",
                        {},
                        {{"requests", std::to_string(scheduled_batch.request_count)},
                         {"draft_depth", std::to_string(draft_depth)}});
                    const MTPVerifierForwardExecutionResult forward =
                        executeMTPSpecVerifierForward(
                            *runner_,
                            verifier_input_plan);
                    if (!forward.ok)
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? std::string("stochastic request-batch verifier forward failed: ") +
                                      forward.error
                                : std::string("stochastic request-batch verifier forward failed and cleanup failed: ") +
                                      forward.error);
                    }
                }

                if (!cleanup_row_modes())
                {
                    return set_producer_error(
                        "stochastic request-batch verifier could not disable row-indexed logits");
                }

                std::vector<DeviceStochasticBatchOutcomeRequest>
                    outcome_requests;
                outcome_requests.reserve(
                    scheduled_batch.greedy_requests.size());
                std::vector<Sampler> bonus_samplers(
                    static_cast<size_t>(request_batch));
                int next_draft_slot = 0;
                for (size_t i = 0; i < scheduled_batch.greedy_requests.size(); ++i)
                {
                    const int request_id = scheduled_batch.request_ids[i];
                    if (request_id < 0 || request_id >= request_batch)
                    {
                        return set_producer_error(
                            "stochastic request-batch verifier returned an out-of-range request id");
                    }

                    const MTPDecodeCatchupGreedyRequest &request =
                        scheduled_batch.greedy_requests[i];
                    const int verifier_token_count =
                        static_cast<int>(request.draft_tokens.size());
                    const int compare_rows = verifier_token_count - 1;
                    if (compare_rows <= 0 ||
                        compare_rows >
                            sampling_math::kSpeculativeBatchMaxRows ||
                        static_cast<int>(verifier_input_plan.query_start_locs.size()) <=
                            static_cast<int>(i))
                    {
                        return set_producer_error(
                            "stochastic request-batch verifier request shape is invalid");
                    }
                    if (stop_tokens_.size() >
                        static_cast<size_t>(
                            sampling_math::kSpeculativeBatchMaxStopTokens))
                    {
                        return set_producer_error(
                            "stochastic request-batch stop-token count exceeds device summary capacity");
                    }

                    const int first_compact_row =
                        verifier_input_plan.query_start_locs[i];
                    const int first_draft_slot = next_draft_slot;
                    next_draft_slot += compare_rows;
                    const int bonus_row = compare_rows;
                    if (!runner_->buildStochasticDistributionsOnDevice(
                            DeviceLogitsSource::AllPosition,
                            first_compact_row,
                            DeviceDistributionBuffer::Target,
                            first_compact_row,
                            compare_rows + 1,
                            active_sampling_params_,
                            vocab))
                    {
                        return set_producer_error(
                            "stochastic request-batch compact target-row build failed");
                    }

                    if (!use_request_batch_device_draft_slots &&
                        !runner_->stageStochasticDraftTokensForDeviceVerification(
                            request.draft_tokens.data() + 1,
                            compare_rows,
                            first_draft_slot))
                    {
                        return set_producer_error(
                            "stochastic request-batch draft-token staging failed");
                    }

                    Sampler &request_sampler =
                        batched_request_states_[static_cast<size_t>(request_id)]
                            .sampler;
                    DeviceStochasticBatchOutcomeRequest descriptor;
                    descriptor.request_id = request_id;
                    descriptor.first_target_slot = first_compact_row;
                    descriptor.first_draft_slot = first_draft_slot;
                    descriptor.row_count = compare_rows;
                    descriptor.first_token = request.draft_tokens.front();
                    descriptor.first_token_from_device = false;
                    descriptor.bonus_target_slot = first_compact_row + bonus_row;
                    descriptor.use_device_draft_tokens = true;
                    descriptor.use_vllm_probability_rejection = true;
                    const int base_cached_tokens =
                        scheduled_batch.base_cached_tokens[i];
                    for (int row = 0; row < compare_rows; ++row)
                    {
                        const int logical_position =
                            base_cached_tokens + 1 + row;
                        descriptor.accept_thresholds[static_cast<size_t>(row)] =
                            mtpSpecStochasticThresholdForPosition(
                                active_sampling_params_,
                                request_sampler,
                                logical_position,
                                MTPSpecStochasticDrawPurpose::Accept);
                        descriptor.residual_thresholds[static_cast<size_t>(row)] =
                            mtpSpecStochasticThresholdForPosition(
                                active_sampling_params_,
                                request_sampler,
                                logical_position,
                                MTPSpecStochasticDrawPurpose::Residual);
                        descriptor.draft_tokens[static_cast<size_t>(row)] =
                            request.draft_tokens[static_cast<size_t>(row + 1)];
                    }

                    Sampler bonus_sampler = request_sampler;
                    descriptor.bonus_threshold =
                        mtpSpecStochasticThresholdForPosition(
                            active_sampling_params_,
                            bonus_sampler,
                            base_cached_tokens + verifier_token_count,
                            MTPSpecStochasticDrawPurpose::Sample);
                    bonus_samplers[static_cast<size_t>(request_id)] =
                        bonus_sampler;
                    descriptor.inverse_sample_seed =
                        mtpSpecInverseSampleSeedForThresholds(
                            active_sampling_params_,
                            descriptor.residual_thresholds.data(),
                            static_cast<size_t>(compare_rows));
                    descriptor.inverse_sample_first_logical_position =
                        base_cached_tokens + 1;
                    descriptor.stop_token_count =
                        static_cast<int>(stop_tokens_.size());
                    for (int stop_index = 0;
                         stop_index < descriptor.stop_token_count;
                         ++stop_index)
                    {
                        descriptor.stop_tokens[static_cast<size_t>(stop_index)] =
                            stop_tokens_[static_cast<size_t>(stop_index)];
                    }

                    outcome_requests.push_back(std::move(descriptor));
                }

                outcomes->assign(
                    outcome_requests.size(),
                    MTPDeviceRejectionBatchOutcome{});
                if (use_device_resident_request_batch_publication)
                {
                    DeviceSpeculativeOutcomeHandle resident_handle;
                    if (!runner_->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                            outcome_requests.data(),
                            static_cast<int>(outcome_requests.size()),
                            &resident_handle))
                    {
                        return set_producer_error(
                            "stochastic request-batch resident device outcome verifier failed");
                    }

                    resident_request_batch_outcome = resident_handle;
                    resident_request_batch_outcome_ready = true;
                    resident_request_batch_verifier_rows =
                        scheduled_batch.shape.max_draft_tokens;
                    resident_request_batch_outcome_requests =
                        std::move(outcome_requests);
                    resident_request_batch_bonus_samplers =
                        std::move(bonus_samplers);
                    outcomes->clear();
                    return true;
                }
                else if (!runner_->verifyStochasticDistributionsRequestBatchOutcomesOnDevice(
                             outcome_requests.data(),
                             static_cast<int>(outcome_requests.size()),
                             outcomes->data()))
                {
                    return set_producer_error(
                        "stochastic request-batch device outcome verifier failed");
                }

                if (!process_stochastic_host_outcomes(
                        outcome_requests,
                        bonus_samplers,
                        *outcomes,
                        error))
                {
                    return false;
                }
                return true;
            };

            MTPOwnedDeviceOutcomeBatchTransactionResult tx;
            if (use_device_resident_request_batch_publication)
            {
                auto release_and_fail =
                    [&](std::string message) -> GenerationBatchResult
                {
                    if (owner.hasInFlightBatch())
                    {
                        std::string release_error;
                        tx.released =
                            owner.releaseInFlightBatch(&release_error);
                        if (!tx.released)
                        {
                            message += "; release failed: ";
                            message += release_error;
                        }
                    }
                    return fail_after_checkpoint(message);
                };

                tx.scheduled_batch = owner.scheduleNextBatch(scheduler);
                if (!tx.scheduled_batch.ok)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched stochastic "
                                    "verifier scheduling failed: ") +
                        tx.scheduled_batch.error);
                }

                std::string produce_error;
                tx.produced =
                    produce_stochastic_outcomes(
                        tx.scheduled_batch,
                        &tx.device_outcomes,
                        &produce_error);
                if (!tx.produced)
                {
                    std::string message =
                        "decodeStepBatch() request-batched stochastic "
                        "resident verifier production failed";
                    if (!produce_error.empty())
                    {
                        message += ": ";
                        message += produce_error;
                    }
                    return release_and_fail(std::move(message));
                }

                if (!resident_request_batch_outcome_ready ||
                    !resident_request_batch_outcome.valid() ||
                    resident_request_batch_verifier_rows <= 0)
                {
                    return release_and_fail(
                        "decodeStepBatch() request-batched stochastic "
                        "resident verifier produced no valid device outcome");
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = resident_request_batch_outcome;
                publication_request.request_count =
                    tx.scheduled_batch.request_count;
                publication_request.max_draft_tokens =
                    resident_request_batch_verifier_rows;
                publication_request.base_sidecar_position = 0;
                publication_request.publish_mtp_shifted_kv =
                    tx.scheduled_batch.requires_shifted_kv_publication;

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer publication_timer(
                        "mtp",
                        "request_batch_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"sampling", "stochastic"},
                         {"request_count",
                          std::to_string(publication_request.request_count)},
                         {"max_draft_tokens",
                          std::to_string(
                              publication_request.max_draft_tokens)}});
                    tx.published =
                        runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                            publication_request,
                            &publication_error);
                }
                if (!tx.published)
                {
                    std::string message =
                        "decodeStepBatch() request-batched stochastic "
                        "resident state publication failed";
                    if (!publication_error.empty())
                    {
                        message += ": ";
                        message += publication_error;
                    }
                    return release_and_fail(std::move(message));
                }

                DeviceResidentHostStateAdoptionRequest adoption_request;
                adoption_request.logical_state =
                    runner_->deviceResidentLogicalSequenceState();
                adoption_request.base_cached_tokens =
                    tx.scheduled_batch.base_cached_tokens;
                adoption_request.publish_mtp_shifted_kv =
                    tx.scheduled_batch.requires_shifted_kv_publication;

                std::string adoption_error;
                {
                    PerfStatsCollector::ScopedTimer adoption_timer(
                        "mtp",
                        "request_batch_device_resident_host_metadata_adoption",
                        "decode",
                        {},
                        {{"request_count",
                          std::to_string(tx.scheduled_batch.request_count)}});
                    if (!runner_->adoptDeviceResidentMTPSpecPublishedHostStateFromDeviceMetadata(
                            adoption_request,
                            &adoption_error))
                    {
                        std::string message =
                            "decodeStepBatch() request-batched stochastic "
                            "resident host-state metadata adoption failed";
                        if (!adoption_error.empty())
                        {
                            message += ": ";
                            message += adoption_error;
                        }
                        return release_and_fail(std::move(message));
                    }
                }

                /*
                 * Full compact outcome materialization is deliberately after
                 * live-state publication and host-mirror adoption.  It now
                 * exists only so decodeStepBatch() can return response tokens
                 * and update sampler bookkeeping; state planning and adoption
                 * above consume the resident logical-state mailbox only.
                 */
                tx.device_outcomes.assign(
                    static_cast<size_t>(tx.scheduled_batch.request_count),
                    MTPDeviceRejectionBatchOutcome{});
                {
                    PerfStatsCollector::ScopedTimer bridge_timer(
                        "mtp",
                        "request_batch_stochastic_device_outcome_host_bridge",
                        "decode",
                        {},
                        {{"request_count",
                          std::to_string(tx.scheduled_batch.request_count)}});
                    if (!runner_->materializeDeviceSpeculativeOutcomesForHostResponse(
                            resident_request_batch_outcome,
                            tx.device_outcomes.data()))
                    {
                        return release_and_fail(
                            "decodeStepBatch() request-batched stochastic "
                            "resident outcome host-response materialization failed");
                    }
                }

                std::string process_error;
                if (!process_stochastic_host_outcomes(
                        resident_request_batch_outcome_requests,
                        resident_request_batch_bonus_samplers,
                        tx.device_outcomes,
                        &process_error))
                {
                    std::string message =
                        "decodeStepBatch() request-batched stochastic "
                        "resident host outcome processing failed";
                    if (!process_error.empty())
                    {
                        message += ": ";
                        message += process_error;
                    }
                    return release_and_fail(std::move(message));
                }

                std::string commit_error;
                tx.committed = owner.commitInFlightBatch(&commit_error);
                if (!tx.committed)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched stochastic "
                                    "resident publication succeeded but owner "
                                    "commit failed: ") +
                        commit_error);
                }

                PerfStatsCollector::addCounter(
                    "mtp",
                    "request_batch_device_resident_state_publications",
                    1.0,
                    "decode",
                    {},
                    {{"request_count",
                      std::to_string(publication_request.request_count)},
                     {"max_draft_tokens",
                      std::to_string(
                          publication_request.max_draft_tokens)}});
                tx.ok = true;
            }
            else
            {
                tx =
                    executeOwnedMTPDeviceOutcomeScheduledBatchTransactionAndPublish(
                        owner,
                        scheduler,
                        produce_stochastic_outcomes,
                        publish);
                if (!tx.ok)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched stochastic "
                                    "verifier transaction failed: ") +
                        tx.error);
                }
            }

            scheduled_request_ids = tx.scheduled_batch.request_ids;
            catchup_results.reserve(tx.device_outcomes.size());
            for (size_t i = 0; i < tx.device_outcomes.size(); ++i)
            {
                MTPDecodeCatchupGreedyResult catchup =
                    buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                        tx.scheduled_batch.greedy_requests[i],
                        tx.device_outcomes[i]);
                if (!catchup.ok)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() stochastic catch-up "
                                    "summary failed: ") +
                        catchup.error);
                }
                catchup_results.push_back(std::move(catchup));
            }
        }

        if (static_cast<int>(scheduled_request_ids.size()) !=
                static_cast<int>(catchup_results.size()) ||
            scheduled_request_ids.empty())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() request-batched verifier transaction "
                "returned inconsistent request vectors");
        }

        bool batch_has_ready_token = false;
        bool batch_has_non_ready_lane = false;
        for (const MTPDecodeCatchupGreedyResult &catchup : catchup_results)
        {
            const bool has_ready_token =
                !catchup.stopped_on_output &&
                catchup.all_speculative_accepted &&
                catchup.ready_token >= 0;
            batch_has_ready_token = batch_has_ready_token || has_ready_token;
            batch_has_non_ready_lane = batch_has_non_ready_lane || !has_ready_token;
        }
        const bool inline_mixed_ready_tokens =
            batch_has_ready_token && batch_has_non_ready_lane;

        for (size_t i = 0; i < scheduled_request_ids.size(); ++i)
        {
            const int request = scheduled_request_ids[i];
            if (request < 0 || request >= request_batch)
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() request-batched verifier returned an "
                    "out-of-range request id");
            }

            BatchedDecodeRequestState &state =
                batched_request_states_[static_cast<size_t>(request)];
            GenerationResult &request_result =
                batch_result.requests[static_cast<size_t>(request)];
            const MTPDecodeCatchupGreedyResult &catchup =
                catchup_results[i];
            if (!catchup.ok || catchup.accepted_tokens.empty())
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() request-batched verifier produced an "
                    "invalid catch-up result");
            }

            /*
             * The verifier input prefix includes the already-returned
             * condition token so model state can be published atomically from
             * the pre-verifier base. Do not emit or record that token twice.
             */
            auto new_token_begin = catchup.accepted_tokens.begin() + 1;
            request_result.tokens.insert(
                request_result.tokens.end(),
                new_token_begin,
                catchup.accepted_tokens.end());
            if (sampled_terminal_samplers[static_cast<size_t>(request)].has_value())
            {
                state.sampler =
                    *sampled_terminal_samplers[static_cast<size_t>(request)];
            }

            const bool has_ready_token =
                !catchup.stopped_on_output &&
                catchup.all_speculative_accepted &&
                catchup.ready_token >= 0;
            if (inline_mixed_ready_tokens && has_ready_token)
            {
                /*
                 * If only some request lanes have a bonus-ready token, deferring
                 * those tokens would leave the batch half in terminal-logit
                 * sampling and half in verifier continuation. vLLM-style
                 * request batches stay lockstep: emit the bonus token now, but
                 * do not publish bonus recurrent/KV state. The next verifier
                 * forward will consume this token as the request condition.
                 */
                request_result.tokens.push_back(catchup.ready_token);
            }

            request_result.is_complete =
                catchup.stopped_on_output ||
                (inline_mixed_ready_tokens &&
                 has_ready_token &&
                 std::find(stop_tokens_.begin(), stop_tokens_.end(),
                           catchup.ready_token) != stop_tokens_.end());
            state.is_complete = request_result.is_complete;
            state.last_token = !request_result.tokens.empty()
                                   ? request_result.tokens.back()
                                   : catchup.accepted_tokens.back();

            if (has_ready_token && !inline_mixed_ready_tokens)
            {
                state.prefill_logits_ready = true;
                state.ready_sampled_token = catchup.ready_token;
                state.ready_sampled_params = active_sampling_params_;
            }
            else
            {
                state.prefill_logits_ready = false;
                state.ready_sampled_token.reset();
                state.ready_sampled_params.reset();
            }

            for (auto it = new_token_begin; it != catchup.accepted_tokens.end(); ++it)
            {
                const int32_t token = *it;
                state.sampler.record_token(token);
            }
            if (inline_mixed_ready_tokens && has_ready_token)
            {
                state.sampler.record_token(catchup.ready_token);
            }

            ++mtp_stats_.verifier_runs;
            mtp_stats_.verifier_token_count +=
                static_cast<uint64_t>(catchup.main_forward_token_count);
            const int accepted_speculative =
                std::max(0, catchup.accepted_speculative_prefix);
            mtp_stats_.accepted_tokens +=
                static_cast<uint64_t>(accepted_speculative);
            const int rejected =
                catchup.all_speculative_accepted ? 0 : 1;
            mtp_stats_.rejected_tokens += static_cast<uint64_t>(rejected);
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "request_batched_verifier_transactions",
            1.0,
            "decode",
            {},
            {{"requests", std::to_string(scheduled_request_ids.size())},
             {"draft_depth", std::to_string(draft_depth)},
             {"mode", stochastic_batch_verify ? "stochastic" : "greedy"}});
        return batch_result;
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

    void OrchestrationRunner::recordMTPVerifierEconomyPerfStatsIfNeeded()
    {
        if (mtp_verifier_economy_perfstats_emitted_ ||
            !PerfStatsCollector::isEnabled() ||
            !runner_)
        {
            return;
        }

        mtp_verifier_economy_perfstats_emitted_ = true;
        const MTPVerifierEconomyCapability capability =
            runner_->mtpVerifierEconomyCapability();
        const std::string device = runner_->primaryDeviceId().is_valid()
                                       ? runner_->primaryDeviceId().toString()
                                       : std::string{};
        const char *active_model_class =
            mtpDepthPolicyModelClassName(inferMTPDepthPolicyModelClass(model_ctx_));

        auto emit_lane = [&](const char *lane_name, const MTPVerifierEconomyLane &lane)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_economy_capability",
                static_cast<double>(std::max(0, lane.max_rows)),
                "decode",
                device,
                {{"lane", lane_name},
                 {"active_model_class", active_model_class},
                 {"correct", perfBool(lane.correct)},
                 {"serial_decode_equivalent_fallback", perfBool(lane.serial_decode_equivalent_fallback)},
                 {"grouped_decode_equivalent", perfBool(lane.grouped_decode_equivalent)},
                 {"row_indexed_lm_head", perfBool(lane.row_indexed_lm_head)},
                 {"device_resident_input", perfBool(lane.device_resident_input)},
                 {"device_resident_outcome", perfBool(lane.device_resident_outcome)},
                 {"device_resident_publication", perfBool(lane.device_resident_publication)},
                 {"host_bridge_free_hot_path", perfBool(lane.host_bridge_free_hot_path)},
                 {"graph_capturable", perfBool(lane.graph_capturable)},
                 {"greedy", perfBool(lane.greedy)},
                 {"stochastic", perfBool(lane.stochastic)},
                 {"max_rows", std::to_string(std::max(0, lane.max_rows))},
                 {"perf_gate_status", lane.perf_gate_status}});
        };

        emit_lane("dense", capability.dense);
        emit_lane("moe", capability.moe);
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

    std::optional<int> OrchestrationRunner::currentMTPBaseSidecarPositionForPlanning(
        const char *context,
        std::string *error) const
    {
        if (!runner_)
        {
            if (error)
                *error = "MTP sidecar position planning requires an initialized runner";
            return std::nullopt;
        }

        const DeviceResidentLogicalSequenceStateHandle resident_state =
            runner_->deviceResidentLogicalSequenceState();
        if (resident_state.valid() &&
            !runner_->hostLogicalStateMirrorsDeviceResidentState())
        {
            std::string message =
                "MTP sidecar position planning refused stale host logical state";
            if (context && context[0] != '\0')
                message += std::string(" for ") + context;
            if (error)
                *error = std::move(message);
            return std::nullopt;
        }

        const int position = runner_->get_position();
        if (position < 0)
        {
            if (error)
                *error = "MTP sidecar position planning received a negative host position";
            return std::nullopt;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "sidecar_position_planning_host_reads",
            1.0,
            "decode",
            {},
            {{"context", context ? context : "unknown"},
             {"resident_mailbox", resident_state.valid() ? "true" : "false"}});
        return position;
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
        recordMTPVerifierEconomyPerfStatsIfNeeded();

        GenerationResult result;
        const int vocab = vocabSize();
        if (vocab <= 0)
        {
            result.error = "Invalid vocabulary size for MTP decode";
            return result;
        }

        const bool use_ready_logits = prefill_logits_ready_;
        const std::optional<int32_t> ready_sampled_token = ready_sampled_token_;
        const std::optional<SamplingParams> ready_sampled_params = ready_sampled_params_;
        const std::optional<DeviceResidentLogicalSequenceStateHandle>
            ready_sampled_resident_state = ready_sampled_resident_state_;
        const std::optional<int32_t> pending_condition_token =
            pending_mtp_condition_token_;
        const std::optional<SamplingParams> pending_condition_params =
            pending_mtp_condition_params_;
        const std::optional<DeviceResidentLogicalSequenceStateHandle>
            pending_condition_resident_state =
                pending_mtp_condition_resident_state_;
        const std::optional<DeviceResidentLogicalSequenceStateHandle>
            prelaunched_first_sidecar_resident_state =
                prelaunched_mtp_first_sidecar_resident_state_;
        const std::optional<SamplingParams> prelaunched_first_sidecar_params =
            prelaunched_mtp_first_sidecar_params_;
        prefill_logits_ready_ = false;
        ready_sampled_token_.reset();
        ready_sampled_params_.reset();
        ready_sampled_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        PrefixStateSnapshot rollback_checkpoint;
        bool rollback_checkpoint_captured = false;
        PrefixStateSnapshot verifier_base_checkpoint;
        int transaction_base_cached_tokens = -1;

        auto fail_without_checkpoint = [&](const std::string &message) -> GenerationResult
        {
            PerfStatsCollector::addCounter("mtp", "decode_step_failures", 1.0, "decode",
                                           std::string{}, {{"reason", message}});
            prefill_logits_ready_ = use_ready_logits;
            ready_sampled_token_ = ready_sampled_token;
            ready_sampled_params_ = ready_sampled_params;
            ready_sampled_resident_state_ = ready_sampled_resident_state;
            pending_mtp_condition_token_ = pending_condition_token;
            pending_mtp_condition_params_ = pending_condition_params;
            pending_mtp_condition_resident_state_ =
                pending_condition_resident_state;
            prelaunched_mtp_first_sidecar_resident_state_.reset();
            prelaunched_mtp_first_sidecar_params_.reset();
            result.error = message;
            return result;
        };

        auto capture_rollback_checkpoint = [&]() -> bool
        {
            if (rollback_checkpoint_captured)
                return true;

            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "capture_live_prefix_state",
                    "decode");
                if (!runner_->ensureMTPCheckpointTerminalHidden())
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "capture_live_prefix_terminal_hidden_failures",
                        1.0,
                        "decode");
                    return false;
                }
                rollback_checkpoint = runner_->captureLivePrefixCheckpoint();
            }
            if (!rollback_checkpoint.valid)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "capture_live_prefix_state_failures",
                    1.0,
                    "decode");
                return false;
            }
            rollback_checkpoint_captured = true;
            transaction_base_cached_tokens = rollback_checkpoint.cached_tokens;
            PerfStatsCollector::addCounter(
                "mtp",
                rollback_checkpoint.logical_checkpoint
                    ? "live_prefix_checkpoint_logical"
                    : "live_prefix_checkpoint_payload",
                1.0,
                "decode");
            return true;
        };

        auto fail_after_checkpoint = [&](const std::string &message) -> GenerationResult
        {
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "disable_all_position_logits_after_failure", "decode");
                runner_->setComputeAllPositionLogits(false);
                runner_->setComputeRowIndexedAllPositionLogits(false, 0);
            }
            bool restored = false;
            if (rollback_checkpoint_captured)
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "restore_live_prefix_state_after_failure", "decode");
                restored = runner_->restoreLivePrefixState(rollback_checkpoint);
            }
            else
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "decode_step_failures_without_rollback_checkpoint",
                    1.0,
                    "decode",
                    {},
                    {{"reason", message}});
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
            ready_sampled_resident_state_ = ready_sampled_resident_state;
            pending_mtp_condition_token_ = pending_condition_token;
            pending_mtp_condition_params_ = pending_condition_params;
            pending_mtp_condition_resident_state_ =
                pending_condition_resident_state;
            prelaunched_mtp_first_sidecar_resident_state_.reset();
            prelaunched_mtp_first_sidecar_params_.reset();
            result.error = message;
            return result;
        };

        const bool needs_mpi_mtp_boundary_fence =
            mpi_ctx_ && mpi_ctx_->world_size() > 1;
        auto fence_mpi_mtp_boundary =
            [&](const char *boundary) -> std::optional<std::string>
        {
            if (!needs_mpi_mtp_boundary_fence)
                return std::nullopt;

            /*
             * Global/NodeLocal TP ranks must enter sidecar and verifier
             * collectives in the same logical order. A local runner flush only
             * drains the participant's own work; it does not stop rank 0 from
             * starting target-verifier allreduces while rank 1 is still inside
             * the MTP sidecar. The shared-memory fast path matches by epoch and
             * payload size, so this boundary fence is part of the transaction
             * contract rather than a best-effort scheduling hint.
             */
            const std::string boundary_name = boundary ? boundary : "unknown";
            try
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "sidecar_mpi_boundary_fence",
                    "decode",
                    std::string{},
                    {{"boundary", boundary_name},
                     {"rank", std::to_string(mpi_ctx_->rank())},
                     {"world_size", std::to_string(mpi_ctx_->world_size())}});
                mpi_ctx_->barrier();
                PerfStatsCollector::addCounter(
                    "mtp",
                    "sidecar_mpi_boundary_fences",
                    1.0,
                    "decode",
                    std::string{},
                    {{"boundary", boundary_name},
                     {"rank", std::to_string(mpi_ctx_->rank())},
                     {"world_size", std::to_string(mpi_ctx_->world_size())}});
            }
            catch (const std::exception &ex)
            {
                return std::string("MTP MPI sidecar boundary fence failed at ") +
                       boundary_name + ": " + ex.what();
            }
            catch (...)
            {
                return std::string("MTP MPI sidecar boundary fence failed at ") +
                       boundary_name;
            }
            return std::nullopt;
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
            if (ready_sampled_resident_state.has_value() &&
                !ready_sampled_resident_state->valid())
            {
                return fail_after_checkpoint(
                    "Ready MTP verifier token resident logical-state handle is stale or incomplete");
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
        const MTPVerifierEconomyCapability verifier_economy =
            runner_->mtpVerifierEconomyCapability();
        const MTPDepthPolicyModelClass verifier_model_class =
            inferMTPDepthPolicyModelClass(model_ctx_);
        const int verifier_policy_probe_rows =
            std::max(1, effectiveMTPMaxDraftDepth(mtp));
        const bool supports_grouped_decode_equivalent_outcome =
            supportsGroupedVerifierOutcomeForModel(
                verifier_economy,
                verifier_model_class,
                verifier_policy_probe_rows,
                stochastic_verify);
        const MTPVerifierPolicyDecision verifier_policy =
            chooseMTPVerifierPolicy(
                MTPVerifierPolicyInput{
                    .greedy_sampling = active_sampling_params_.is_greedy(),
                    .stochastic_verify = stochastic_verify,
                    .uses_sampling_penalties = use_sampling_penalties,
                    .supports_row_local_penalty_application =
                        !use_sampling_penalties ||
                        runner_->supportsRowLocalAllPositionPenaltyApplication(),
                    .supports_spec_state_publication =
                        supports_all_position_state_publication,
                    .supports_grouped_decode_equivalent_outcome =
                        supports_grouped_decode_equivalent_outcome,
                });
        if (verifier_policy.path == MTPVerifierExecutionPath::Unsupported)
        {
            return fail_after_checkpoint(
                std::string("MTP verifier policy selected unsupported path: ") +
                verifier_policy.reason);
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_policy_selections",
            1.0,
            "decode",
            {},
            {{"path", std::to_string(static_cast<int>(verifier_policy.path))},
             {"reason", verifier_policy.reason},
             {"model_class", mtpDepthPolicyModelClassName(verifier_model_class)},
             {"probe_rows", std::to_string(verifier_policy_probe_rows)},
             {"grouped_outcome_supported", perfBool(supports_grouped_decode_equivalent_outcome)},
             {"direct_publication_supported", perfBool(supports_all_position_state_publication)}});
        const bool use_all_position_state_publication_verifier =
            verifier_policy.path ==
            MTPVerifierExecutionPath::AllPositionStatePublication;
        const bool use_grouped_outcome_device_resident_publication_verifier =
            verifier_policy.path ==
                MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome &&
            (!stochastic_verify || stochastic_device_verify) &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsDeviceResidentMTPSpecStatePublication();
        const bool use_decode_equivalent_replay_publication_verifier =
            verifier_policy.path ==
                MTPVerifierExecutionPath::DecodeEquivalentSequential ||
            verifier_policy.path ==
                MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome;
        if (use_ready_logits && pending_condition_token.has_value())
        {
            return fail_after_checkpoint(
                "MTP decode found both ready terminal logits and a pending condition token");
        }
        const bool pending_condition_has_resident_state =
            pending_condition_resident_state.has_value() &&
            pending_condition_resident_state->valid();
        const bool pending_condition_candidate =
            pending_condition_token.has_value() && !use_ready_logits;
        const bool use_pending_condition_row =
            pending_condition_candidate &&
            (use_all_position_state_publication_verifier ||
             use_grouped_outcome_device_resident_publication_verifier);
        const bool ready_sampled_has_resident_state =
            use_ready_logits &&
            ready_sampled_token.has_value() &&
            ready_sampled_resident_state.has_value() &&
            ready_sampled_resident_state->valid();
        if (use_pending_condition_row)
        {
            if (!pending_condition_params.has_value())
            {
                return fail_after_checkpoint(
                    "Pending MTP condition token is missing the sampling parameters that produced it");
            }
            if (!samplingParamsEqual(*pending_condition_params, active_sampling_params_))
            {
                return fail_after_checkpoint(
                    "Pending MTP condition token was sampled with different sampling parameters");
            }
            if (pending_condition_resident_state.has_value() &&
                !pending_condition_has_resident_state)
            {
                return fail_after_checkpoint(
                    "Pending MTP condition resident logical-state handle is stale or incomplete");
            }
            if (use_grouped_outcome_device_resident_publication_verifier &&
                !pending_condition_has_resident_state)
            {
                return fail_after_checkpoint(
                    "Grouped-outcome MTP pending condition is missing device-resident logical state");
            }
        }
        if (pending_condition_token.has_value() &&
            !use_pending_condition_row)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_condition_fast_path_bypasses",
                1.0,
                "decode",
                {},
                {{"verifier_path",
                  std::to_string(static_cast<int>(verifier_policy.path))}});
        }
        const bool verify_sidecar_preserves_main_state =
            DebugEnv::isTruthyEnv("LLAMINAR_MTP_VERIFY_SIDECAR_PRESERVES_MAIN_STATE");
        const bool verify_commit_replay_check =
            DebugEnv::isTruthyEnv("LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_CHECK") &&
            !stochastic_verify &&
            !use_sampling_penalties &&
            active_sampling_params_.is_greedy();
        /*
         * The vLLM-style GPU lanes publish from device-resident spec slots and
         * treat the live transaction as atomic. They only need a logical base
         * stamp on the success path; rollback checkpoints are reserved for
         * verifier lanes that still mutate/restore host-owned state.
         *
         * This is deliberately not the same contract as
         * supportsLogicalMTPVerifierBaseCheckpoint().  Hybrid MoE/GDN runners may
         * be unable to restore recurrent payload state from a token-count-only
         * snapshot in replay/debug code, while still being able to publish the
         * just-produced verifier rows directly on device without ever restoring
         * the verifier base on the success path.  Grouped-outcome MoE uses the
         * same success-path contract once it proves sidecar drafting preserves
         * the main state and publishes the accepted rows from compact device
         * metadata.
         */
        const bool use_device_publication_without_rollback_checkpoint =
            (use_all_position_state_publication_verifier ||
             use_grouped_outcome_device_resident_publication_verifier) &&
            (!stochastic_verify || stochastic_device_verify) &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsDeviceResidentMTPSpecStatePublication() &&
            runner_->supportsMTPSidecarPreservesMainState() &&
            !verify_sidecar_preserves_main_state &&
            !verify_commit_replay_check;
        const bool can_synthesize_verifier_base_checkpoint =
            (use_all_position_state_publication_verifier ||
             use_grouped_outcome_device_resident_publication_verifier) &&
            runner_->supportsMTPSidecarPreservesMainState() &&
            (runner_->supportsLogicalMTPVerifierBaseCheckpoint() ||
             use_device_publication_without_rollback_checkpoint) &&
            !verify_sidecar_preserves_main_state &&
            !verify_commit_replay_check;

        if (use_device_publication_without_rollback_checkpoint)
        {
            std::string position_error;
            const std::optional<int> base_position =
                currentMTPBaseSidecarPositionForPlanning(
                    "device-publication transaction base",
                    &position_error);
            if (!base_position)
                return fail_without_checkpoint(position_error);

            transaction_base_cached_tokens = *base_position;
            verifier_base_checkpoint =
                makeLogicalMTPVerifierBaseSnapshot(transaction_base_cached_tokens);
            PerfStatsCollector::addCounter(
                "mtp",
                "live_prefix_checkpoint_skipped_direct_publication",
                1.0,
                "decode",
                {},
                {{"cached_tokens", std::to_string(transaction_base_cached_tokens)},
                 {"verifier_path",
                  use_grouped_outcome_device_resident_publication_verifier
                      ? "grouped_decode_equivalent_outcome"
                      : "all_position_state_publication"}});
        }
        else
        {
            if (!capture_rollback_checkpoint())
                return fail_without_checkpoint("MTP decode could not capture live prefix state");
            verifier_base_checkpoint = rollback_checkpoint;
        }

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
                                            int committed_token_count,
                                            int state_advanced_tokens,
                                            PrefixStateProvenance verifier_source,
                                            bool has_terminal_logits,
                                            bool has_ready_token)
            -> std::optional<std::string>
        {
            if (!base.valid)
                return std::string("MTP transaction base snapshot is invalid");
            if (state_advanced_tokens < 0 ||
                state_advanced_tokens > committed_token_count)
            {
                return std::string("MTP transaction advanced-state token count is outside committed token count");
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
                 {"committed_tokens", std::to_string(committed_token_count)},
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
                                                  int state_advanced_token_count = -1,
                                                  int emitted_token_start_index = 0,
                                                  std::optional<int32_t> next_pending_condition_token = std::nullopt,
                                                  std::optional<DeviceResidentLogicalSequenceStateHandle>
                                                      next_pending_condition_resident_state = std::nullopt,
                                                  std::optional<DeviceResidentLogicalSequenceStateHandle>
                                                      ready_condition_resident_state = std::nullopt)
            -> std::optional<std::string>
        {
            if (tokens.empty())
                return std::string("MTP transaction produced no committed tokens");
            if (emitted_token_start_index < 0 ||
                emitted_token_start_index > static_cast<int>(tokens.size()))
            {
                return std::string("MTP transaction emitted-token start is outside committed tokens");
            }
            PerfStatsCollector::ScopedTimer commit_timer(
                "mtp",
                "transaction_output_commit",
                "decode",
                {},
                {{"path", path},
                 {"source", toString(verifier_source)}});

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

            pending_mtp_condition_token_ = next_pending_condition_token;
            pending_mtp_condition_params_ =
                next_pending_condition_token.has_value()
                    ? std::optional<SamplingParams>{active_sampling_params_}
                    : std::optional<SamplingParams>{};
            pending_mtp_condition_resident_state_ =
                next_pending_condition_token.has_value()
                    ? next_pending_condition_resident_state
                    : std::nullopt;

            prefill_logits_ready_ = terminal_logits_ready && !is_complete;
            if (prefill_logits_ready_ && ready_token.has_value())
            {
                if (ready_condition_resident_state.has_value() &&
                    !ready_condition_resident_state->valid())
                {
                    return std::string(
                        "MTP transaction produced a stale ready-token resident logical-state handle");
                }
                ready_sampled_token_ = *ready_token;
                ready_sampled_params_ = active_sampling_params_;
                ready_sampled_resident_state_ = ready_condition_resident_state;
                pending_mtp_condition_token_.reset();
                pending_mtp_condition_params_.reset();
                pending_mtp_condition_resident_state_.reset();
            }
            else
            {
                ready_sampled_token_.reset();
                ready_sampled_params_.reset();
                ready_sampled_resident_state_.reset();
            }

            for (size_t i = static_cast<size_t>(emitted_token_start_index);
                 i < tokens.size();
                 ++i)
            {
                const int32_t token = tokens[i];
                sampler_.record_token(token);
                result.tokens.push_back(token);
            }
            last_token_ = tokens.back();
            result.is_complete = result.is_complete || is_complete;
            if (is_complete)
            {
                /*
                 * A first-sidecar prelaunch is intentionally speculative: it
                 * only prepares the next step's draft proposal.  Stop-token
                 * detection still becomes host-visible at the response
                 * boundary, so a completed request must discard any sidecar
                 * that was queued before the compatibility response bridge.
                 */
                if (prelaunched_mtp_first_sidecar_resident_state_.has_value())
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_first_sidecar_prelaunch_discarded_complete",
                        1.0,
                        "decode",
                        {},
                        {{"path", path}});
                }
                prelaunched_mtp_first_sidecar_resident_state_.reset();
                prelaunched_mtp_first_sidecar_params_.reset();
            }

            ++mtp_stats_.transaction_commits;
            PerfStatsCollector::addCounter(
                "mtp",
                "transaction_commits",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"tokens", join_tokens(tokens)},
                 {"emitted_token_start_index",
                  std::to_string(emitted_token_start_index)},
                 {"emitted_tokens",
                  std::to_string(static_cast<int>(tokens.size()) -
                                 emitted_token_start_index)},
                 {"ready_token", ready_token.has_value()
                                     ? std::to_string(*ready_token)
                                     : std::string("none")},
                 {"next_pending_condition_token",
                  next_pending_condition_token.has_value()
                      ? std::to_string(*next_pending_condition_token)
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

        const int32_t condition_token =
            use_pending_condition_row ? *pending_condition_token : last_token_;
        if (use_pending_condition_row)
        {
            /*
             * The pending token was already emitted and recorded when the
             * previous MTP transaction rejected a draft.  Keep the main state
             * at the accepted prefix and let the verifier consume this token as
             * row zero.  The token is already visible to the response stream, so
             * commit code below must start output emission after row zero.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_condition_verifier_rows",
                1.0,
                "decode",
                {},
                {{"token", std::to_string(condition_token)},
                 {"cached_tokens", std::to_string(transaction_base_cached_tokens)}});
            PerfStatsCollector::addCounter(
                "mtp",
                "condition_forward_skipped_pending_condition",
                1.0,
                "decode");
        }
        else if (!use_ready_logits)
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
                std::string position_error;
                const std::optional<int> verifier_base_position =
                    currentMTPBaseSidecarPositionForPlanning(
                        "verifier-base checkpoint synthesis",
                        &position_error);
                if (!verifier_base_position)
                    return fail_after_checkpoint(position_error);
                verifier_base_checkpoint =
                    makeLogicalMTPVerifierBaseSnapshot(*verifier_base_position);
                PerfStatsCollector::addCounter(
                    "mtp",
                    "capture_verifier_base_prefix_state_skipped_all_position_publication",
                    1.0,
                    "decode",
                    {},
                    {{"cached_tokens", std::to_string(verifier_base_checkpoint.cached_tokens)},
                     {"verifier_path",
                      use_grouped_outcome_device_resident_publication_verifier
                          ? "grouped_decode_equivalent_outcome"
                          : "all_position_state_publication"}});
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "capture_verifier_base_prefix_state",
                    "decode");
                if (!runner_->ensureMTPCheckpointTerminalHidden())
                {
                    return fail_after_checkpoint(
                        "MTP decode could not materialize verifier base terminal hidden");
                }
                verifier_base_checkpoint = runner_->captureLivePrefixCheckpoint();
            }
            if (!verifier_base_checkpoint.valid)
            {
                return fail_after_checkpoint(
                    "MTP decode could not capture verifier base state after condition forward");
            }
            pending_mtp_condition_token_.reset();
            pending_mtp_condition_params_.reset();
            pending_mtp_condition_resident_state_.reset();
        }
        else if (use_ready_logits)
        {
            PerfStatsCollector::addCounter("mtp", "condition_forward_skipped_ready_logits", 1.0, "decode");
        }

        const int requested_speculative_draft_count = currentMTPDraftDepth(mtp);
        const int first_token_output_budget_cost =
            use_pending_condition_row ? 0 : 1;
        const int pre_sample_effective_draft_count =
            decode_step_token_budget_ > 0
                ? std::min(
                      requested_speculative_draft_count,
                      std::max(0, decode_step_token_budget_ -
                                      first_token_output_budget_cost))
                : requested_speculative_draft_count;
        constexpr int32_t kDeferredMTPFirstTokenShadow = -3;
        const bool verifier_accepts_device_first_token =
            use_all_position_state_publication_verifier ||
            use_grouped_outcome_device_resident_publication_verifier;
        const bool can_defer_stochastic_first_host_read =
            stochastic_device_verify &&
            verifier_accepts_device_first_token &&
            runner_->primaryDeviceId().is_gpu() &&
            !active_sampling_params_.has_penalties() &&
            pre_sample_effective_draft_count > 0 &&
            stop_tokens_.size() <=
                static_cast<size_t>(
                    sampling_math::kSpeculativeBatchMaxStopTokens) &&
            runner_->supportsMTPDeviceDraftTokenInput();
        const bool can_defer_greedy_first_host_read =
            !stochastic_verify &&
            (use_all_position_state_publication_verifier ||
             use_grouped_outcome_device_resident_publication_verifier) &&
            runner_->primaryDeviceId().is_gpu() &&
            !use_sampling_penalties &&
            pre_sample_effective_draft_count > 0 &&
            stop_tokens_.size() <=
                static_cast<size_t>(
                    sampling_math::kSpeculativeBatchMaxStopTokens) &&
            runner_->supportsMTPDeviceDraftTokenInput();

        int32_t first_token = -1;
        bool first_token_is_pending_condition = false;
        if (use_pending_condition_row)
        {
            first_token = condition_token;
            first_token_is_pending_condition = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_pending_condition_rows",
                1.0,
                "decode",
                {},
                {{"token", std::to_string(first_token)}});
        }
        else if (use_ready_logits && ready_sampled_token.has_value())
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
                            transaction_base_cached_tokens;
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
                if (can_defer_greedy_first_host_read)
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "sample_first_token_greedy_device_target_slot",
                        "decode");
                    if (!runner_->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                            /*target_sample_slot=*/0,
                            /*out_token=*/nullptr))
                    {
                        return fail_after_checkpoint(
                            "MTP greedy first-token GPU deferred sampling failed");
                    }
                    first_token = kDeferredMTPFirstTokenShadow;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "first_token_greedy_deferred_host_reads",
                        1.0,
                        "decode");
                }
                else
                {
                    PerfStatsCollector::ScopedTimer timer("mtp", "sample_first_token_device", "decode");
                    first_token = runner_->sampleGreedyOnDevice();
                }
                if (first_token < 0 &&
                    first_token != kDeferredMTPFirstTokenShadow)
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
                std::max(0, decode_step_token_budget_ -
                                  first_token_output_budget_cost);
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
                std::string position_error;
                const std::optional<int> base_sidecar_position =
                    currentMTPBaseSidecarPositionForPlanning(
                        "budget-limited direct emit",
                        &position_error);
                if (!base_sidecar_position)
                    return fail_after_checkpoint(position_error);
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
                            *base_sidecar_position);
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

        std::string base_position_error;
        const std::optional<int> maybe_base_sidecar_position =
            currentMTPBaseSidecarPositionForPlanning(
                "speculative sidecar",
                &base_position_error);
        if (!maybe_base_sidecar_position)
            return fail_after_checkpoint(base_position_error);
        const int base_sidecar_position = *maybe_base_sidecar_position;
        bool first_token_is_stop =
            first_token != kDeferredMTPFirstTokenShadow &&
            std::find(stop_tokens_.begin(), stop_tokens_.end(), first_token) != stop_tokens_.end();
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(static_cast<size_t>(speculative_draft_count) + 1);
        draft_tokens.push_back(first_token);
        Sampler draft_sampler = sampler_;
        if ((stochastic_verify || use_sampling_penalties) &&
            !first_token_is_pending_condition &&
            first_token != kDeferredMTPFirstTokenShadow)
        {
            draft_sampler.record_token(first_token);
        }

        std::vector<PrefixStateSnapshot> sidecar_checkpoints;
        sidecar_checkpoints.reserve(1);
        std::vector<std::vector<SamplingDistributionEntry>> host_mtp_draft_distributions(
            static_cast<size_t>(std::max(0, speculative_draft_count)));
        constexpr int32_t kDeferredMTPDraftTokenShadow = -2;
        const bool use_greedy_device_draft_slots =
            (use_all_position_state_publication_verifier ||
             use_grouped_outcome_device_resident_publication_verifier) &&
            !stochastic_verify &&
            !use_sampling_penalties &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsMTPDeviceDraftTokenInput();

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
                                transaction_base_cached_tokens + 1 + draft_idx);
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
                if (use_greedy_device_draft_slots)
                {
                    bool sampled_to_slot = false;
                    {
                        PerfStatsCollector::ScopedTimer timer(
                            "mtp",
                            "sample_mtp_token_greedy_device_slot",
                            "decode",
                            {},
                            {{"draft_idx", std::to_string(draft_idx)}});
                        int32_t *host_shadow =
                            defer_host_read ? nullptr : &token;
                        sampled_to_slot =
                            runner_->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                                draft_idx,
                                host_shadow);
                    }
                    if (sampled_to_slot && defer_host_read)
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "mtp_token_greedy_device_slot_deferred_host_reads",
                            1.0,
                            "decode",
                            {},
                            {{"draft_idx", std::to_string(draft_idx)}});
                        return kDeferredMTPDraftTokenShadow;
                    }
                    if (sampled_to_slot && token >= 0)
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "mtp_token_greedy_device_slot_samples",
                            1.0,
                            "decode",
                            {},
                            {{"draft_idx", std::to_string(draft_idx)}});
                        return token;
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "mtp_token_greedy_device_slot_failures",
                        1.0,
                        "decode",
                        {},
                        {{"draft_idx", std::to_string(draft_idx)}});
                    return -1;
                }
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
        const bool use_device_resident_sidecar_stream_handoff =
            runner_->primaryDeviceId().is_gpu() &&
            !active_sampling_params_.has_penalties() &&
            runner_->supportsMTPSidecarLogitsStreamHandoff();
        const bool use_sidecar_stream_handoff_for_stochastic =
            stochastic_verify &&
            stochastic_device_verify &&
            use_device_resident_sidecar_stream_handoff;
        const bool use_sidecar_stream_handoff_for_grouped_greedy =
            !stochastic_verify &&
            use_grouped_outcome_device_resident_publication_verifier &&
            use_device_resident_sidecar_stream_handoff;
        const bool use_device_draft_token_sidecar =
            (use_sidecar_stream_handoff_for_stochastic ||
             use_sidecar_stream_handoff_for_grouped_greedy) &&
            runner_->supportsMTPDeviceDraftTokenInput();
        const bool use_resident_pending_condition_sidecar =
            use_pending_condition_row &&
            pending_condition_has_resident_state &&
            use_device_draft_token_sidecar;
        const bool use_resident_ready_condition_sidecar =
            ready_sampled_has_resident_state &&
            use_device_draft_token_sidecar;
        auto prelaunch_matches_resident_state =
            [&](const std::optional<DeviceResidentLogicalSequenceStateHandle> &expected)
        {
            return expected.has_value() &&
                   expected->valid() &&
                   prelaunched_first_sidecar_resident_state.has_value() &&
                   prelaunched_first_sidecar_resident_state->valid() &&
                   prelaunched_first_sidecar_params.has_value() &&
                   samplingParamsEqual(
                       *prelaunched_first_sidecar_params,
                       active_sampling_params_) &&
                   prelaunched_first_sidecar_resident_state->sameMailboxAs(
                       *expected);
        };
        const bool use_prelaunched_first_sidecar =
            (use_sidecar_stream_handoff_for_stochastic ||
             use_sidecar_stream_handoff_for_grouped_greedy) &&
            ((use_resident_ready_condition_sidecar &&
              prelaunch_matches_resident_state(ready_sampled_resident_state)) ||
             (use_resident_pending_condition_sidecar &&
              prelaunch_matches_resident_state(pending_condition_resident_state)));
        if (prelaunched_first_sidecar_resident_state.has_value() &&
            !use_prelaunched_first_sidecar)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_prelaunched_first_sidecar_dropped",
                1.0,
                "decode",
                {},
                {{"has_ready_resident",
                  use_resident_ready_condition_sidecar ? "true" : "false"},
                 {"has_pending_resident",
                  use_resident_pending_condition_sidecar ? "true" : "false"}});
        }
        const bool can_defer_stochastic_draft_host_reads =
            use_sidecar_stream_handoff_for_stochastic &&
            verifier_accepts_device_first_token &&
            !active_sampling_params_.has_penalties();
        const bool can_defer_greedy_draft_host_reads =
            !stochastic_verify &&
            use_greedy_device_draft_slots &&
            verifier_accepts_device_first_token &&
            runner_->primaryDeviceId().is_gpu() &&
            !use_sampling_penalties &&
            !active_sampling_params_.has_penalties();
        for (int draft_idx = 0; draft_idx < speculative_draft_count; ++draft_idx)
        {
            bool sidecar_ok = false;
            bool used_prelaunched_first_sidecar = false;
            int32_t mtp_token = -1;
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "sidecar_forward", "decode");
                if (draft_idx == 0)
                {
                    if (use_prelaunched_first_sidecar)
                    {
                        /*
                         * The previous decode step already enqueued this
                         * first sidecar from the same resident mailbox before
                         * flushing host-visible response tokens. Reuse the
                         * pending MTP logits instead of replaying the sidecar
                         * and duplicating shifted-cache work. Greedy lanes
                         * sample those pending logits into the usual device
                         * draft slot below so verifier setup stays device
                         * resident.
                         */
                        sidecar_ok = true;
                        used_prelaunched_first_sidecar = true;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "stochastic_first_sidecar_prelaunch_reuses",
                            1.0,
                            "decode",
                            {},
                            {{"sampling", stochastic_verify ? "stochastic" : "greedy"}});
                    }
                    else if (use_sidecar_sample_fusion)
                    {
                        const bool defer_fused_sample =
                            can_defer_greedy_draft_host_reads;
                        int32_t *sample_host_shadow =
                            defer_fused_sample ? nullptr : &mtp_token;
                        if (first_token == kDeferredMTPFirstTokenShadow &&
                            use_greedy_device_draft_slots)
                        {
                            sidecar_ok =
                                runner_->forwardMTPFromDeviceTargetAndSampleGreedyToDeviceDraftSlot(
                                    /*target_sample_slot=*/0,
                                    base_sidecar_position,
                                    draft_idx,
                                    sample_host_shadow);
                        }
                        else
                        {
                            sidecar_ok =
                                use_greedy_device_draft_slots
                                    ? runner_->forwardMTPAndSampleGreedyToDeviceDraftSlot(
                                          draft_tokens.back(),
                                          draft_idx,
                                          sample_host_shadow)
                                    : runner_->forwardMTPAndSampleGreedy(
                                          draft_tokens.back(),
                                          &mtp_token);
                        }
                        if (sidecar_ok && defer_fused_sample)
                        {
                            mtp_token = kDeferredMTPDraftTokenShadow;
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "mtp_token_greedy_device_slot_deferred_host_reads",
                                1.0,
                                "decode",
                                {},
                                {{"draft_idx", std::to_string(draft_idx)},
                                 {"path", "fused_first_sidecar"}});
                        }
                    }
                    else if (use_sidecar_stream_handoff_for_stochastic)
                    {
                        if (use_resident_ready_condition_sidecar)
                        {
                            /*
                             * The ready token has already been sampled from a
                             * verifier bonus row and will be emitted to the
                             * caller at this step boundary.  The next sidecar
                             * consumes the same token and logical position from
                             * the device mailbox so the hot path does not
                             * round-trip that token through the CPU.
                             */
                            sidecar_ok =
                                runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                    *ready_sampled_resident_state,
                                    /*request_index=*/0);
                            if (sidecar_ok)
                            {
                                PerfStatsCollector::addCounter(
                                    "mtp",
                                    "stochastic_first_sidecar_resident_ready_inputs",
                                    1.0,
                                    "decode");
                            }
                        }
                        else if (use_resident_pending_condition_sidecar)
                        {
                            /*
                             * The host token is only the response shadow.  The
                             * next sidecar consumes the correction token and
                             * logical position from the mailbox produced by
                             * direct device publication, keeping the fixed
                             * depth path aligned with vLLM's resident
                             * transaction model.
                             */
                            sidecar_ok =
                                runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                    *pending_condition_resident_state,
                                    /*request_index=*/0);
                            if (sidecar_ok)
                            {
                                PerfStatsCollector::addCounter(
                                    "mtp",
                                    "stochastic_first_sidecar_resident_condition_inputs",
                                    1.0,
                                    "decode");
                            }
                        }
                        else if (first_token == kDeferredMTPFirstTokenShadow)
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
                    else if (first_token == kDeferredMTPFirstTokenShadow &&
                             use_greedy_device_draft_slots)
                    {
                        sidecar_ok =
                            runner_->forwardMTPFromDeviceTargetForDeviceSampling(
                                /*target_sample_slot=*/0,
                                base_sidecar_position);
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
                        const bool defer_fused_sample =
                            can_defer_greedy_draft_host_reads;
                        int32_t *sample_host_shadow =
                            defer_fused_sample ? nullptr : &mtp_token;
                        if (use_greedy_device_draft_slots && defer_fused_sample)
                        {
                            /*
                             * The previous draft row was sampled into the
                             * runner-owned device slot with index draft_idx-1.
                             * Consume that slot directly and write the next
                             * proposal to draft_idx.  The compact verifier
                             * outcome will materialize response tokens later,
                             * so no intermediate D2H token copy is required.
                             */
                            sidecar_ok =
                                runner_->forwardMTPFromDeviceDraftAndSampleGreedyToDeviceDraftSlot(
                                    draft_idx - 1,
                                    base_sidecar_position + draft_idx,
                                    draft_idx,
                                    sample_host_shadow);
                        }
                        else
                        {
                            sidecar_ok =
                                use_greedy_device_draft_slots
                                    ? runner_->forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot(
                                          draft_tokens.back(),
                                          base_sidecar_position + draft_idx,
                                          draft_idx,
                                          sample_host_shadow)
                                    : runner_->forwardMTPFromLastDraftAndSampleGreedy(
                                          draft_tokens.back(),
                                          base_sidecar_position + draft_idx,
                                          &mtp_token);
                        }
                        if (sidecar_ok && defer_fused_sample)
                        {
                            mtp_token = kDeferredMTPDraftTokenShadow;
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "mtp_token_greedy_device_slot_deferred_host_reads",
                                1.0,
                                "decode",
                                {},
                                {{"draft_idx", std::to_string(draft_idx)},
                                 {"path", "fused_chained_sidecar"}});
                        }
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
            if (use_sidecar_stream_handoff_for_stochastic ||
                used_prelaunched_first_sidecar)
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
                if (needs_mpi_mtp_boundary_fence)
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
            if (auto fence_error =
                    fence_mpi_mtp_boundary("after_sidecar_iteration_flush"))
            {
                return fail_after_checkpoint(*fence_error);
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
                else if (use_decode_equivalent_replay_publication_verifier &&
                         runner_->supportsMTPSidecarPreservesMainState())
                {
                    /*
                     * Decode-equivalent replay still uses the main verifier
                     * base checkpoint, but graph-native sidecars prove they do
                     * not mutate that base while drafting. Capturing a
                     * post-sidecar payload checkpoint here would only export
                     * hybrid KV/GDN state that the replay branch immediately
                     * discards.
                     */
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "post_sidecar_checkpoint_skipped_sidecar_preserved",
                        1.0,
                        "decode",
                        {},
                        {{"verifier_path",
                          verifier_policy.path ==
                                  MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome
                              ? "grouped_decode_equivalent_outcome"
                              : "decode_equivalent_sequential"}});
                }
                else
                {
                    PerfStatsCollector::ScopedTimer timer("mtp", "capture_post_sidecar_prefix_state", "decode");
                    if (!runner_->ensureMTPCheckpointTerminalHidden())
                    {
                        return fail_after_checkpoint(
                            "MTP decode could not materialize post-sidecar terminal hidden");
                    }
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

            const bool sidecar_sample_already_done =
                use_sidecar_sample_fusion &&
                !used_prelaunched_first_sidecar;
            if (!sidecar_sample_already_done)
            {
                const bool next_sidecar_needs_host_token =
                    draft_idx + 1 < speculative_draft_count &&
                    !use_device_draft_token_sidecar;
                const bool greedy_next_sidecar_can_consume_device_token =
                    can_defer_greedy_draft_host_reads &&
                    use_device_draft_token_sidecar &&
                    draft_idx + 1 < speculative_draft_count;
                const bool defer_draft_host_read =
                    (can_defer_stochastic_draft_host_reads &&
                     !next_sidecar_needs_host_token) ||
                    greedy_next_sidecar_can_consume_device_token;
                mtp_token = sample_mtp_token(draft_idx, defer_draft_host_read);
            }
            if (mtp_token < 0)
            {
                if (mtp_token != kDeferredMTPDraftTokenShadow)
                    return fail_after_checkpoint("No MTP logits available");
            }
            if (sidecar_sample_already_done)
            {
                if (use_greedy_device_draft_slots)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "mtp_token_greedy_device_slot_samples",
                        1.0,
                        "decode",
                        {},
                        {{"draft_idx", std::to_string(draft_idx)}});
                }
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

        /*
         * vLLM-style stochastic verification can keep every draft token
         * sidecar-resident: the verifier input row is materialized later on
         * the verifier graph stream, and that materialization waits on the
         * target/draft sample-ready events before copying device tokens. In
         * that lane, synchronizing the sidecar stream here is pure host-side
         * performance debt. Host-token and debug-preservation paths still need
         * the flush because they inspect sidecar-produced state immediately.
         */
        const bool can_skip_sidecar_flush_before_verifier =
            stochastic_verify &&
            stochastic_device_verify &&
            !active_sampling_params_.has_penalties() &&
            runner_->primaryDeviceId().is_gpu() &&
            use_all_position_state_publication_verifier &&
            draft_tokens.size() > 1 &&
            !first_token_is_stop &&
            !needs_mpi_mtp_boundary_fence &&
            stop_tokens_.size() <=
                static_cast<size_t>(
                    sampling_math::kSpeculativeBatchMaxStopTokens) &&
            runner_->supportsMTPDeviceDraftTokenInput() &&
            !DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_FORCE_SIDECAR_FLUSH_BEFORE_VERIFIER") &&
            !DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_VERIFY_SIDECAR_PRESERVES_MAIN_STATE");
        if (can_skip_sidecar_flush_before_verifier)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_final_flush_skipped_device_verifier_tokens",
                1.0,
                "decode",
                {},
                {{"draft_tokens", std::to_string(draft_tokens.size())},
                 {"first_token_deferred",
                  first_token == kDeferredMTPFirstTokenShadow ? "true" : "false"}});
        }
        else
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_final_flush_before_verification",
                "decode");
            if (!runner_->flushPendingMTPWork())
            {
                return fail_after_checkpoint("MTP sidecar stream flush failed before verification");
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_final_flush_required_before_verification",
                1.0,
                "decode",
                {},
                {{"stochastic_verify", stochastic_verify ? "true" : "false"},
                 {"stochastic_device_verify",
                  stochastic_device_verify ? "true" : "false"},
                 {"has_penalties",
                  active_sampling_params_.has_penalties() ? "true" : "false"},
                 {"all_position",
                  use_all_position_state_publication_verifier ? "true" : "false"}});
        }
        if (auto fence_error =
                fence_mpi_mtp_boundary("before_target_verifier"))
        {
            return fail_after_checkpoint(*fence_error);
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

            if (!runner_->ensureMTPCheckpointTerminalHidden())
            {
                return fail_after_checkpoint(
                    "MTP sidecar preservation check could not materialize terminal hidden");
            }
            PrefixStateSnapshot sidecar_state = runner_->captureLivePrefixState();
            if (!sidecar_state.valid)
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not capture sidecar state");
            }
            /*
             * Row-token equality alone can miss the coherence bug class that
             * motivated Phase 9.5: sidecar replay may leave host-visible rows
             * correct while mutating main KV, GDN/short-conv state, or logical
             * positions.  Compare the runtime state surface that the next
             * verifier and prefix-cache probes observe.  Shifted MTP KV is
             * intentionally excluded here; MoE still publishes accepted shifted
             * rows from verifier rows rather than reusing the sidecar draft row.
             */
            const PrefixRuntimeStateSnapshot sidecar_runtime_state =
                runner_->prefixStateProbe();

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
            const PrefixRuntimeStateSnapshot base_runtime_state =
                runner_->prefixStateProbe();
            MTPRuntimeSnapshotComparisonOptions sidecar_state_compare;
            sidecar_state_compare.compare_shifted_mtp_kv = false;
            const MTPStateValidationResult sidecar_state_result =
                compareMTPRuntimeStateSnapshots(
                    base_runtime_state,
                    sidecar_runtime_state,
                    sidecar_state_compare);
            if (!sidecar_state_result)
            {
                return fail_after_checkpoint(
                    "MTP sidecar mutated main runtime state: " +
                    sidecar_state_result.reason);
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
                                                  int state_advanced_token_count = -1,
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

            const bool ready_token_was_provided = expected_next_token >= 0;
            const int state_advanced_count =
                state_advanced_token_count >= 0
                    ? std::clamp(
                          state_advanced_token_count,
                          0,
                          static_cast<int>(tokens_to_replay.size()))
                    : static_cast<int>(tokens_to_replay.size());
            /*
             * The compact all-position verifier publishes state in verifier-row
             * space, not in outer response-token space.  When the verifier also
             * produces a ready token, the correction suffix has already been
             * accounted for by the ready-token contract and the continuation
             * oracle should replay the full committed output stream.  When no
             * ready token exists, the suffix is still a pending condition token
             * and must be fed before comparing continuations.
             */
            const int replay_prefix_count =
                ready_token_was_provided
                    ? static_cast<int>(tokens_to_replay.size())
                    : state_advanced_count;
            const std::vector<int32_t> state_replay_tokens(
                tokens_to_replay.begin(),
                tokens_to_replay.begin() + replay_prefix_count);
            const std::vector<int32_t> pending_condition_inputs(
                tokens_to_replay.begin() + replay_prefix_count,
                tokens_to_replay.end());

            if (!runner_->ensureMTPCheckpointTerminalHidden())
            {
                return std::string("MTP commit replay check could not materialize committed terminal hidden");
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
                            if (layer.payload_hash_available)
                            {
                                out += "/kb=" + std::to_string(layer.k_payload_bytes) +
                                       "/vb=" + std::to_string(layer.v_payload_bytes) +
                                       "/kh=" + std::to_string(layer.k_payload_hash) +
                                       "/vh=" + std::to_string(layer.v_payload_hash);
                            }
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
                    if (layer.device_state_hash_available)
                    {
                        gdn += "/dr=" +
                               std::to_string(layer.recurrence_device_hash) +
                               "/dc=" +
                               std::to_string(layer.conv_device_hash);
                    }
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
            auto summarize_snapshot = [](const PrefixStateSnapshot &snapshot)
            {
                auto summarize_blocks = [](const std::vector<PrefixBlockHandle> &blocks)
                {
                    std::ostringstream out;
                    out << "count=" << blocks.size();
                    const size_t limit = std::min<size_t>(blocks.size(), 4);
                    for (size_t i = 0; i < limit; ++i)
                    {
                        const PrefixBlockHandle &block = blocks[i];
                        out << "|i=" << i
                            << "/tokens=" << block.key.token_count
                            << "/start=" << block.key.token_start
                            << "/idx=" << block.key.block_index
                            << "/kv=" << block.layout.faKVBytes()
                            << "/hybrid=" << block.layout.hybrid_state_bytes
                            << "/hybrid_host=" << block.layout.hybrid_host_state_bytes
                            << "/hybrid_device=" << block.layout.hybrid_device_state_bytes
                            << "/term_h=" << block.layout.terminal_hidden_bytes
                            << "/has_hybrid=" << (block.has_hybrid_state ? "1" : "0")
                            << "/has_term_h=" << (block.has_terminal_hidden ? "1" : "0")
                            << "/dev_hybrid=" << (block.device_hybrid_storage ? "1" : "0");
                    }
                    if (blocks.size() > limit)
                        out << "|...";
                    return out.str();
                };

                std::ostringstream out;
                out << "valid=" << (snapshot.valid ? "1" : "0")
                    << " logical=" << (snapshot.logical_checkpoint ? "1" : "0")
                    << " provenance=" << toString(snapshot.provenance)
                    << " cached=" << snapshot.cached_tokens
                    << " ready=" << (snapshot.ready_event_valid ? "1" : "0")
                    << " participants=" << snapshot.participant_snapshots.size()
                    << " blocks={" << summarize_blocks(snapshot.blocks) << "}"
                    << " mtp_blocks={" << summarize_blocks(snapshot.mtp_blocks) << "}";
                if (!snapshot.mtp_cached_tokens.empty())
                {
                    out << " mtp_cached=[";
                    for (size_t i = 0; i < snapshot.mtp_cached_tokens.size(); ++i)
                    {
                        if (i > 0)
                            out << ",";
                        out << snapshot.mtp_cached_tokens[i];
                    }
                    out << "]";
                }
                const size_t participant_limit =
                    std::min<size_t>(snapshot.participant_snapshots.size(), 4);
                for (size_t i = 0; i < participant_limit; ++i)
                {
                    const PrefixStateSnapshot &child =
                        snapshot.participant_snapshots[i];
                    out << " child" << i
                        << "{prov=" << toString(child.provenance)
                        << " logical=" << (child.logical_checkpoint ? "1" : "0")
                        << " cached=" << child.cached_tokens
                        << " blocks=" << child.blocks.size()
                        << " mtp_blocks=" << child.mtp_blocks.size()
                        << " ready=" << (child.ready_event_valid ? "1" : "0")
                        << "}";
                }
                if (snapshot.participant_snapshots.size() > participant_limit)
                    out << " child...";
                return out.str();
            };
            const PrefixRuntimeStateSnapshot committed_probe_before =
                runner_->prefixStateProbe();
            auto first_gdn_mismatch = [](
                                          const PrefixRuntimeStateSnapshot &lhs,
                                          const PrefixRuntimeStateSnapshot &rhs,
                                          const char *lhs_name,
                                          const char *rhs_name) -> std::string
            {
                const size_t count =
                    std::min(lhs.gdn_layers.size(), rhs.gdn_layers.size());
                if (lhs.gdn_layers.size() != rhs.gdn_layers.size())
                {
                    return std::string(lhs_name) + "_gdn_layers=" +
                           std::to_string(lhs.gdn_layers.size()) + " " +
                           rhs_name + "_gdn_layers=" +
                           std::to_string(rhs.gdn_layers.size());
                }
                for (size_t i = 0; i < count; ++i)
                {
                    const PrefixGDNLayerProbe &a = lhs.gdn_layers[i];
                    const PrefixGDNLayerProbe &b = rhs.gdn_layers[i];
                    const bool both_have_device_state =
                        a.device_state_hash_available &&
                        b.device_state_hash_available;
                    const bool mismatch =
                        a.global_layer != b.global_layer ||
                        (both_have_device_state
                             ? (a.recurrence_device_bytes != b.recurrence_device_bytes ||
                                a.conv_device_bytes != b.conv_device_bytes ||
                                a.recurrence_device_hash != b.recurrence_device_hash ||
                                a.conv_device_hash != b.conv_device_hash)
                             : (a.recurrence_hash != b.recurrence_hash ||
                                a.conv_hash != b.conv_hash ||
                                a.device_state_hash_available != b.device_state_hash_available ||
                                a.recurrence_all_zero != b.recurrence_all_zero ||
                                a.conv_all_zero != b.conv_all_zero));
                    if (mismatch)
                    {
                        std::ostringstream oss;
                        oss << "layer=" << a.global_layer
                            << " " << lhs_name << "_rec="
                            << a.recurrence_hash
                            << " " << rhs_name << "_rec="
                            << b.recurrence_hash
                            << " " << lhs_name << "_conv="
                            << a.conv_hash
                            << " " << rhs_name << "_conv="
                            << b.conv_hash
                            << " " << lhs_name << "_dev_rec="
                            << a.recurrence_device_hash
                            << " " << rhs_name << "_dev_rec="
                            << b.recurrence_device_hash
                            << " " << lhs_name << "_dev_conv="
                            << a.conv_device_hash
                            << " " << rhs_name << "_dev_conv="
                            << b.conv_device_hash
                            << " " << lhs_name << "_rec_zero="
                            << (a.recurrence_all_zero ? "true" : "false")
                            << " " << rhs_name << "_rec_zero="
                            << (b.recurrence_all_zero ? "true" : "false")
                            << " " << lhs_name << "_conv_zero="
                            << (a.conv_all_zero ? "true" : "false")
                            << " " << rhs_name << "_conv_zero="
                            << (b.conv_all_zero ? "true" : "false");
                        return oss.str();
                    }
                }
                return "none";
            };

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

            std::string continuation_failure_detail;
            PrefixRuntimeStateSnapshot first_committed_restore_probe;
            PrefixRuntimeStateSnapshot second_committed_restore_probe;
            auto continuation_failure_suffix = [&]() -> std::string
            {
                if (continuation_failure_detail.empty())
                    return {};
                return std::string(": ") + continuation_failure_detail +
                       " current_probe={" + summarize_probe(runner_->prefixStateProbe()) + "}" +
                       " committed_probe_before={" + summarize_probe(committed_probe_before) + "}" +
                       " first_committed_restore_probe={" + summarize_probe(first_committed_restore_probe) + "}" +
                       " second_committed_restore_probe={" + summarize_probe(second_committed_restore_probe) + "}" +
                       " committed_restore_gdn_delta={" +
                       first_gdn_mismatch(first_committed_restore_probe,
                                          second_committed_restore_probe,
                                          "first",
                                          "second") +
                       "}" +
                       " committed_snapshot={" + summarize_snapshot(committed_checkpoint) + "}";
            };

            auto commit_shifted_replay_row = [&](
                                                  int32_t token,
                                                  int token_index,
                                                  const char *context) -> bool
            {
                auto current_shifted_mtp_tokens = [&]() -> int
                {
                    const PrefixRuntimeStateSnapshot probe =
                        runner_->prefixStateProbe();
                    int max_tokens = -1;
                    for (const PrefixKVCacheProbe &cache : probe.mtp_kv_caches)
                    {
                        for (const PrefixKVLayerProbe &layer : cache.layers)
                        {
                            if (layer.seq_idx == 0)
                            {
                                max_tokens =
                                    std::max(max_tokens, layer.cached_tokens);
                            }
                        }
                    }
                    return max_tokens;
                };

                /*
                 * token_index is the committed output/verifier-row index.  It is
                 * not necessarily the number of shifted MTP KV rows currently
                 * resident in this replay timeline: MoE/non-reuse paths can
                 * restore a verifier base where the sidecar's row-zero append has
                 * intentionally been discarded.  The replay checker runs only
                 * under LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_CHECK, so it can afford
                 * to inspect the debug probe and derive a position offset that
                 * appends exactly one shifted row from the live cache count.
                 */
                const int shifted_before = current_shifted_mtp_tokens();
                if (shifted_before < 0)
                {
                    continuation_failure_detail =
                        std::string("shifted MTP replay commit could not inspect current shifted cache in ") +
                        context +
                        " token_index=" + std::to_string(token_index);
                    return false;
                }
                const int shifted_rows_committed_in_this_replay = 0;
                const int replay_position_offset =
                    shifted_before + 1 - shifted_rows_committed_in_this_replay;
                bool ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "commit_replay_check_shifted_commit",
                        "decode",
                        {},
                        {{"path", path},
                         {"context", context},
                         {"token_index", std::to_string(token_index)}});
                    /*
                     * The replay oracle must match the shared stepwise verifier
                     * boundary exactly: every token forwarded from the verifier
                     * base first publishes the shifted MTP row derived from the
                     * current terminal hidden.  Otherwise the main KV/GDN state
                     * can look serial-equivalent while the sidecar cache remains
                     * at the base position, which makes the next MTP step
                     * compare a live committed state with a stale replay.
                     */
                    ok = runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                        token,
                        shifted_rows_committed_in_this_replay,
                        /*allow_speculative_discard=*/true,
                        replay_position_offset);
                }
                if (!ok)
                {
                    continuation_failure_detail =
                        std::string("shifted MTP replay commit failed in ") +
                        context +
                        " token_index=" + std::to_string(token_index) +
                        " token=" + std::to_string(token) +
                        " base_sidecar_position=" +
                        std::to_string(base_sidecar_position) +
                        " shifted_before=" + std::to_string(shifted_before) +
                        " replay_position_offset=" +
                        std::to_string(replay_position_offset);
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "commit_replay_check_shifted_commits",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"context", context}});
                return true;
            };

            auto forward_replay_token = [&](
                                            int32_t token,
                                            int token_index,
                                            const char *context) -> bool
            {
                if (!commit_shifted_replay_row(token, token_index, context))
                    return false;
                int forward_token = static_cast<int>(token);
                if (!runner_->forward(&forward_token, 1))
                {
                    continuation_failure_detail =
                        std::string("replay forward failed in ") +
                        context +
                        " token_index=" + std::to_string(token_index) +
                        " token=" + std::to_string(token);
                    return false;
                }
                return true;
            };

            auto sample_continuation = [&](int32_t first_input,
                                           int first_token_index)
                -> std::optional<std::vector<int32_t>>
            {
                continuation_failure_detail.clear();
                std::vector<int32_t> tokens;
                tokens.reserve(static_cast<size_t>(continuation_check_depth));
                int32_t input = first_input;
                for (int i = 0; i < continuation_check_depth; ++i)
                {
                    const int token_index = first_token_index + i;
                    if (!forward_replay_token(
                            input,
                            token_index,
                            "continuation"))
                    {
                        continuation_failure_detail +=
                            " depth=" + std::to_string(i);
                        return std::nullopt;
                    }
                    const int32_t sampled = runner_->sampleGreedyOnDevice();
                    if (sampled < 0)
                    {
                        continuation_failure_detail =
                            std::string("main continuation greedy sampling failed at depth=") +
                            std::to_string(i) +
                            " input=" + std::to_string(input);
                        return std::nullopt;
                    }
                    tokens.push_back(sampled);
                    input = sampled;
                }
                return tokens;
            };

            /*
             * vLLM-style publication can make tokens host-visible before every
             * one of those tokens has been consumed by the live verifier state.
             * A first-row rejection is the common case: the correction token is
             * emitted as output, but it remains the next condition token.  The
             * replay oracle must therefore feed the unadvanced emitted suffix
             * first, prove it produces the expected ready token, and only then
             * continue past that ready token.
             */
            auto sample_continuation_after_pending_inputs = [&](
                                                                const std::vector<int32_t> &pending_inputs,
                                                                int32_t ready_token,
                                                                int pending_token_index_base)
                -> std::optional<std::vector<int32_t>>
            {
                if (ready_token < 0)
                    return std::nullopt;

                const PrefixRuntimeStateSnapshot pending_start_probe =
                    runner_->prefixStateProbe();
                for (size_t i = 0; i < pending_inputs.size(); ++i)
                {
                    const int32_t pending = pending_inputs[i];
                    const int token_index =
                        pending_token_index_base +
                        static_cast<int>(i);
                    if (!forward_replay_token(
                            pending,
                            token_index,
                            "pending_condition"))
                    {
                        continuation_failure_detail +=
                            " pending_index=" + std::to_string(i);
                        return std::nullopt;
                    }
                    const int32_t sampled = runner_->sampleGreedyOnDevice();
                    if (sampled < 0)
                    {
                        continuation_failure_detail =
                            std::string("pending-condition greedy sampling failed at index=") +
                            std::to_string(i) +
                            " input=" + std::to_string(pending);
                        return std::nullopt;
                    }

                    const int32_t expected =
                        (i + 1 < pending_inputs.size())
                            ? pending_inputs[i + 1]
                            : ready_token;
                    if (sampled != expected)
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "commit_replay_check_pending_condition_mismatches",
                            1.0,
                            "decode",
                            {},
                            {{"path", path},
                             {"pending_index", std::to_string(i)},
                             {"pending_token", std::to_string(pending)},
                             {"sampled", std::to_string(sampled)},
                             {"expected", std::to_string(expected)}});
                        continuation_failure_detail =
                            std::string("pending-condition token mismatch at index=") +
                            std::to_string(i) +
                            " input=" + std::to_string(pending) +
                            " sampled=" + std::to_string(sampled) +
                            " expected=" + std::to_string(expected) +
                            " pending_start_probe={" +
                            summarize_probe(pending_start_probe) + "}";
                        return std::nullopt;
                    }
                }

                return sample_continuation(
                    ready_token,
                    pending_token_index_base +
                        static_cast<int>(pending_inputs.size()));
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
                const int deferred_token_index =
                    std::max(
                        0,
                        static_cast<int>(tokens_to_replay.size()) - 1);
                if (!forward_replay_token(
                        deferred_condition_token,
                        deferred_token_index,
                        "deferred_condition_ready"))
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

            const PrefixStateSnapshot &base = *verifier_replay_base_checkpoint;
            if (!runner_->restoreLivePrefixState(base))
            {
                return std::string("MTP commit replay check could not restore verifier base state");
            }
            bool sequential_replay_ok = true;
            for (size_t i = 0; i < tokens_to_replay.size(); ++i)
            {
                const int32_t replay_token = tokens_to_replay[i];
                if (!forward_replay_token(
                        replay_token,
                        static_cast<int>(i),
                        "full_replay"))
                {
                    sequential_replay_ok = false;
                    break;
                }
            }
            if (!sequential_replay_ok)
            {
                return std::string("MTP commit replay check sequential replay failed") +
                       continuation_failure_suffix();
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
            first_committed_restore_probe = runner_->prefixStateProbe();
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

            /*
             * Keep the first replay assertion side-effect-free with respect to
             * future continuation work.  The continuation probe below is useful,
             * but it intentionally mutates live decode state; running it before
             * the base replay can turn a restore-ordering bug into a misleading
             * ready-token mismatch.
             */
            std::optional<std::vector<int32_t>> live_committed_continuation =
                sample_continuation_after_pending_inputs(
                    pending_condition_inputs,
                    expected_next_token,
                    replay_prefix_count);
            if (!live_committed_continuation)
            {
                return std::string("MTP commit replay check live committed continuation forward failed") +
                       continuation_failure_suffix();
            }
            if (!runner_->restoreLivePrefixState(committed_checkpoint))
            {
                return std::string("MTP commit replay check could not restore committed state after live continuation check");
            }
            second_committed_restore_probe = runner_->prefixStateProbe();

            std::optional<int32_t> committed_ready_token =
                prepare_committed_ready_state();
            if (!committed_ready_token ||
                *committed_ready_token != expected_next_token)
            {
                return std::string("MTP commit replay check committed ready-token derivation mismatch");
            }
            if (derived_next_token_from_deferred_condition)
            {
                /*
                 * Deriving the ready token for a forced reject is itself an
                 * ordinary one-token forward from the committed checkpoint.
                 * The continuation oracle below intentionally feeds the same
                 * pending condition token to prove the restored state, so put
                 * the runner back at the committed checkpoint before that
                 * second probe.
                 */
                if (!runner_->restoreLivePrefixState(committed_checkpoint))
                {
                    return std::string("MTP commit replay check could not restore committed state after ready-token derivation");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "commit_replay_check_restores_after_ready_derivation",
                    1.0,
                    "decode",
                    {},
                    {{"path", path}});
            }
            std::optional<std::vector<int32_t>> committed_continuation =
                sample_continuation_after_pending_inputs(
                    pending_condition_inputs,
                    expected_next_token,
                    replay_prefix_count);
            if (!committed_continuation)
            {
                return std::string("MTP commit replay check committed continuation forward failed") +
                       continuation_failure_suffix();
            }
            if (!runner_->restoreLivePrefixState(base))
            {
                return std::string("MTP commit replay check could not restore verifier base for continuation replay");
            }
            bool sequential_continuation_ok = true;
            for (size_t i = 0; i < state_replay_tokens.size(); ++i)
            {
                const int32_t replay_token = state_replay_tokens[i];
                if (!forward_replay_token(
                        replay_token,
                        static_cast<int>(i),
                        "state_replay"))
                {
                    sequential_continuation_ok = false;
                    break;
                }
            }
            PrefixRuntimeStateSnapshot serial_probe_after_replay;
            bool have_serial_probe_after_replay = false;
            if (sequential_continuation_ok)
            {
                serial_probe_after_replay = runner_->prefixStateProbe();
                have_serial_probe_after_replay = true;
                std::optional<std::vector<int32_t>> replay_continuation =
                    sample_continuation_after_pending_inputs(
                        pending_condition_inputs,
                        expected_next_token,
                        replay_prefix_count);
                if (!replay_continuation)
                {
                    return std::string("MTP commit replay check continuation replay forward failed") +
                           continuation_failure_suffix();
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
                            if (!forward_replay_token(
                                    replay_token,
                                    static_cast<int>(i),
                                    "prefix_replay_summary"))
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
                            sample_continuation_after_pending_inputs(
                                pending_condition_inputs,
                                expected_next_token,
                                replay_prefix_count);
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
                           (have_serial_probe_after_replay
                                ? " serial_probe_after_replay={" +
                                      summarize_probe(serial_probe_after_replay) +
                                      "} gdn_first_mismatch={" +
                                      first_gdn_mismatch(
                                          committed_probe_before,
                                          serial_probe_after_replay,
                                          "committed",
                                          "serial") +
                                      "}"
                                : std::string{}) +
                           " mismatch_probe={" + summarize_probe(mismatch_probe) + "}" +
                           " continuation_depth=" + std::to_string(continuation_check_depth) +
                           " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false")) +
                           (debug_context.empty() ? std::string{} : " " + debug_context);
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
                           (have_serial_probe_after_replay
                                ? " serial_probe_after_replay={" +
                                      summarize_probe(serial_probe_after_replay) +
                                      "} gdn_first_mismatch={" +
                                      first_gdn_mismatch(
                                          committed_probe_before,
                                          serial_probe_after_replay,
                                          "committed",
                                          "serial") +
                                      "}"
                                : std::string{}) +
                           " mismatch_probe={" + summarize_probe(mismatch_probe) + "}" +
                           " continuation_depth=" + std::to_string(continuation_check_depth) +
                           " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false")) +
                           (debug_context.empty() ? std::string{} : " " + debug_context);
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
                 {"state_advanced_tokens", std::to_string(state_advanced_count)},
                 {"pending_condition_inputs", join_tokens(pending_condition_inputs)},
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
             * The first sidecar draft can only be kept as live shifted-MTP
             * state when the backend explicitly proves that its sidecar row is
             * equivalent to the first accepted target row.  Otherwise the
             * shifted cache is published only from accepted verifier rows below.
             * Synthesizing row zero from the host-visible first token is not
             * decode-equivalent for MoE: a rejected correction is an output
             * token, not yet a condition-token state boundary.
             */
            const bool first_shifted_row_available_from_sidecar =
                sidecar_preserves_main_state &&
                runner_->supportsMTPShiftedRowReuseFromSidecar() &&
                !first_token_is_stop;
            if (first_shifted_row_available_from_sidecar)
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
            else if (!first_token_is_stop)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_initial_shifted_deferred_to_verifier_rows",
                    1.0,
                    "decode",
                    {},
                    {{"draft_tokens", std::to_string(draft_tokens.size())},
                     {"first_token_deferred",
                      first_token == kDeferredMTPFirstTokenShadow ? "true" : "false"},
                     {"sidecar_preserves_main_state",
                      sidecar_preserves_main_state ? "true" : "false"}});
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
            const bool can_prepare_greedy_device_verifier_tokens =
                !stochastic_verify &&
                use_greedy_device_draft_slots &&
                draft_tokens.size() > 1 &&
                !first_token_is_stop;
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
                if (can_defer_stochastic_batch_verifier_sync ||
                    can_prepare_greedy_device_verifier_tokens)
                {
                    /*
                     * Penalty-free stochastic verification already stores the
                     * sampled sidecar draft tokens in runner-owned device slots;
                     * the penalty-free greedy lane can now do the same. Ask the
                     * runner to compose the compact verifier input row on device
                     * so the embedding graph and compact reducer read the same
                     * device-resident token row.
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
            bool state_published_from_device_outcome = false;
            std::vector<int32_t> direct_sampled_verifier_rows_for_replay_check;
            auto apply_all_position_row_penalties_for_history =
                [&](int compare_rows,
                    int bonus_row,
                    const char *counter_name) -> bool
            {
                if (!active_sampling_params_.has_penalties())
                    return true;
                if (first_token == kDeferredMTPFirstTokenShadow)
                    return false;

                /*
                 * Row i is consumed only when every previous speculative row
                 * accepted.  Its sampler history is therefore deterministic:
                 * request base history, first target token, then
                 * draft[1..i-1].  Mutating the row on the verifier stream lets
                 * compact greedy/stochastic reducers see the exact same logits
                 * as serial decode without reading full rows back to host.
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
                    counter_name,
                    static_cast<double>(compare_rows + 1),
                    "decode",
                    {},
                    {{"verifier_path", "all_position_state_publication"}});
                return true;
            };
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
                    if (!apply_all_position_row_penalties_for_history(
                            compare_rows,
                            bonus_row,
                            "stochastic_vllm_penalty_rows_preapplied"))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP vLLM row penalty application failed");
                    }
                    if (!runner_->buildStochasticDistributionsOnDevice(
                            DeviceLogitsSource::AllPosition,
                            /*first_row=*/0,
                            DeviceDistributionBuffer::Target,
                            /*first_slot=*/0,
                            /*row_count=*/compare_rows + 1,
                            active_sampling_params_,
                            vocab))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP batched compact target-row build failed");
                    }

                    for (int row = 0; row < compare_rows; ++row)
                    {
                        const int row_logical_position =
                            transaction_base_cached_tokens + 1 + row;
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
                            transaction_base_cached_tokens +
                                static_cast<int>(draft_tokens.size()));
                    const uint64_t inverse_sample_seed =
                        inverse_sample_seed_for_thresholds(
                            batched_residual_thresholds.data(),
                            batched_residual_thresholds.size());
                    const int inverse_sample_first_logical_position =
                        transaction_base_cached_tokens + 1;
                    DeviceSpeculativeOutcomeHandle device_outcome_handle;
                    bool resident_outcome_ok = false;
                    {
                        PerfStatsCollector::ScopedTimer resident_timer(
                            "mtp",
                            "all_position_stochastic_device_resident_outcome_enqueue",
                            "decode",
                            {},
                            {{"verifier_path", "all_position_state_publication"}});
                        /*
                         * Phase 10's no-D2H target starts here: the verifier
                         * reducer leaves compact output tokens and metadata on
                         * the verifier stream.  The host bridge below remains
                         * compatibility scaffolding until publication and
                         * continuation-token staging can consume this handle
                         * directly.
                         */
                        resident_outcome_ok =
                            first_token == kDeferredMTPFirstTokenShadow
                                ? runner_->verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(
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
                                      &device_outcome_handle,
                                      inverse_sample_seed,
                                      inverse_sample_first_logical_position,
                                      /*use_vllm_probability_rejection=*/true)
                                : runner_->verifyStochasticDistributionsBatchOutcomeOnDeviceResident(
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
                                      &device_outcome_handle,
                                      inverse_sample_seed,
                                      inverse_sample_first_logical_position,
                                      /*use_vllm_probability_rejection=*/true);
                    }
                    if (!resident_outcome_ok)
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP resident device outcome verifier failed");
                    }

                    if (runner_->supportsDeviceResidentMTPSpecStatePublication())
                    {
                        DeviceSpeculativePublicationRequest publication_request;
                        publication_request.outcome = device_outcome_handle;
                        publication_request.request_count = 1;
                        publication_request.max_draft_tokens =
                            static_cast<int>(draft_tokens.size());
                        publication_request.base_sidecar_position =
                            base_sidecar_position;
                        publication_request.publish_mtp_shifted_kv = true;

                        std::string publication_error;
                        PerfStatsCollector::ScopedTimer direct_publish_timer(
                            "mtp",
                            "all_position_publish_accepted_state_device_resident",
                            "decode",
                            {},
                            {{"verifier_path",
                              "all_position_state_publication"}});
                        if (!runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                                publication_request,
                                &publication_error))
                        {
                            return fail_after_checkpoint(
                                std::string("All-position stochastic MTP device-resident state publication failed: ") +
                                publication_error);
                        }
                        state_published_from_device_outcome = true;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "device_resident_state_publications",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path",
                              "all_position_state_publication"},
                             {"request_count", "1"},
                             {"max_draft_tokens",
                              std::to_string(
                                  publication_request.max_draft_tokens)}});

                        const bool can_prelaunch_next_first_sidecar =
                            use_sidecar_stream_handoff_for_stochastic &&
                            use_device_draft_token_sidecar &&
                            runner_->supportsMTPSidecarPreservesMainState() &&
                            requested_speculative_draft_count > 0;
                        if (can_prelaunch_next_first_sidecar)
                        {
                            DeviceResidentLogicalSequenceStateHandle handle =
                                runner_->deviceResidentLogicalSequenceState();
                            if (!handle.valid())
                            {
                                return fail_after_checkpoint(
                                    "All-position stochastic MTP direct publication produced no resident logical-state row for sidecar prelaunch");
                            }
                            {
                                PerfStatsCollector::ScopedTimer prelaunch_timer(
                                    "mtp",
                                    "stochastic_first_sidecar_prelaunch_enqueue",
                                    "decode",
                                    {},
                                    {{"verifier_path",
                                      "all_position_state_publication"}});
                                if (!runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                        handle,
                                        /*request_index=*/0))
                                {
                                    return fail_after_checkpoint(
                                        "All-position stochastic MTP resident first-sidecar prelaunch failed");
                                }
                            }
                            prelaunched_mtp_first_sidecar_resident_state_ =
                                handle;
                            prelaunched_mtp_first_sidecar_params_ =
                                active_sampling_params_;
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "stochastic_first_sidecar_prelaunches",
                                1.0,
                                "decode",
                                {},
                                {{"request_index", "0"},
                                 {"stop_tokens",
                                  std::to_string(stop_tokens_.size())}});
                        }
                    }

                    DeviceSpeculativeVerifyBatchOutcome device_outcome;
                    {
                        PerfStatsCollector::ScopedTimer bridge_timer(
                            "mtp",
                            "all_position_stochastic_device_outcome_host_bridge",
                            "decode",
                            {},
                            {{"verifier_path", "all_position_state_publication"}});
                        if (!runner_->materializeDeviceSpeculativeOutcomesForHostResponse(
                                device_outcome_handle,
                                &device_outcome))
                        {
                            return fail_after_checkpoint(
                                "All-position stochastic MTP resident outcome host-response materialization failed");
                        }
                    }
                    if (device_outcome.sampled_terminal)
                        sampler_ = bonus_sampler;

                    {
                        PerfStatsCollector::ScopedTimer catchup_timer(
                            "mtp",
                            "all_position_stochastic_device_outcome_catchup_plan",
                            "decode",
                            {},
                            {{"verifier_path",
                              "all_position_state_publication"}});
                        catchup =
                            buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                                catchup_request,
                                device_outcome);
                    }
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
                    const int physical_rows = compare_rows;
                    const int semantic_rows =
                        std::min(
                            physical_rows,
                            std::max(0, device_outcome.consumed_verifier_rows));
                    const int post_reject_rows =
                        std::max(0, physical_rows - semantic_rows);
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_device_physical_verify_rows",
                        static_cast<double>(physical_rows),
                        "decode",
                        {},
                        {{"verifier_path", "all_position_state_publication"},
                         {"implementation", "device_batch_outcome"},
                         {"request_batch", "false"}});
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_device_semantic_verify_rows",
                        static_cast<double>(semantic_rows),
                        "decode",
                        {},
                        {{"verifier_path", "all_position_state_publication"},
                         {"implementation", "device_batch_outcome"},
                         {"request_batch", "false"}});
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_device_post_reject_rows",
                        static_cast<double>(post_reject_rows),
                        "decode",
                        {},
                        {{"verifier_path", "all_position_state_publication"},
                         {"implementation", "device_batch_outcome"},
                         {"request_batch", "false"}});
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
                            transaction_base_cached_tokens + draft_idx);
                    const float residual_threshold =
                        residual_threshold_for_position(
                            sampler_,
                            transaction_base_cached_tokens + draft_idx);
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
                                      transaction_base_cached_tokens +
                                          static_cast<int>(draft_tokens.size())))
                            : sampleMTPDistributionWithThreshold(
                                  host_target_distribution,
                                  sample_threshold_for_position(
                                      sampler_,
                                      transaction_base_cached_tokens +
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
                    stop_tokens_.size() <=
                        static_cast<size_t>(
                            sampling_math::kSpeculativeBatchMaxStopTokens) &&
                    draft_tokens.size() <=
                        static_cast<size_t>(
                            sampling_math::kSpeculativeBatchMaxOutputTokens);
                if (use_greedy_device_batch_outcome)
                {
                    if (verify_commit_replay_check)
                    {
                        /*
                         * The compact greedy reducer is the production fast path,
                         * but replay-check failures need to know whether the
                         * compact reducer disagreed with ordinary row argmax or
                         * whether the verifier graph itself produced different
                         * rows.  This diagnostic intentionally runs only under
                         * the expensive commit-replay guard; it may consume and
                         * synchronize the deferred verifier stream before the
                         * compact reducer, so it is never part of the hot path.
                         */
                        direct_sampled_verifier_rows_for_replay_check.assign(
                            sampled_verifier_rows.size(),
                            -1);
                        if (!runner_->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                                0,
                                static_cast<int>(
                                    direct_sampled_verifier_rows_for_replay_check.size()),
                                direct_sampled_verifier_rows_for_replay_check.data()))
                        {
                            direct_sampled_verifier_rows_for_replay_check.clear();
                        }
                    }

                    PerfStatsCollector::ScopedTimer sample_timer(
                        "mtp",
                        "all_position_verifier_greedy_device_summary",
                        "decode");
                    const int compare_rows =
                        static_cast<int>(draft_tokens.size()) - 1;
                    const int bonus_row = compare_rows;
                    if (!apply_all_position_row_penalties_for_history(
                            compare_rows,
                            bonus_row,
                            "greedy_vllm_penalty_rows_preapplied"))
                    {
                        return fail_after_checkpoint(
                            "All-position greedy MTP row penalty application failed");
                    }
                    DeviceSpeculativeOutcomeHandle device_outcome_handle;
                    if (!runner_->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                            draft_tokens.data(),
                            static_cast<int>(draft_tokens.size()),
                            stop_tokens_.data(),
                            static_cast<int>(stop_tokens_.size()),
                            &device_outcome_handle))
                    {
                        return fail_after_checkpoint(
                            "All-position greedy MTP resident compact device outcome verifier failed");
                    }

                    if (runner_->supportsDeviceResidentMTPSpecStatePublication())
                    {
                        DeviceSpeculativePublicationRequest publication_request;
                        publication_request.outcome = device_outcome_handle;
                        publication_request.request_count = 1;
                        publication_request.max_draft_tokens =
                            static_cast<int>(draft_tokens.size());
                        publication_request.base_sidecar_position =
                            base_sidecar_position;
                        publication_request.publish_mtp_shifted_kv = true;

                        std::string publication_error;
                        PerfStatsCollector::ScopedTimer direct_publish_timer(
                            "mtp",
                            "all_position_publish_accepted_state_device_resident",
                            "decode",
                            {},
                            {{"verifier_path",
                              "all_position_state_publication"},
                             {"sampling", "greedy"}});
                        if (!runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                                publication_request,
                                &publication_error))
                        {
                            return fail_after_checkpoint(
                                std::string("All-position greedy MTP device-resident state publication failed: ") +
                                publication_error);
                        }
                        state_published_from_device_outcome = true;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "device_resident_state_publications",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path",
                              "all_position_state_publication"},
                             {"sampling", "greedy"},
                             {"request_count", "1"},
                             {"max_draft_tokens",
                              std::to_string(
                                  publication_request.max_draft_tokens)}});
                    }

                    DeviceSpeculativeVerifyBatchOutcome device_outcome;
                    {
                        PerfStatsCollector::ScopedTimer bridge_timer(
                            "mtp",
                            "all_position_greedy_device_outcome_host_bridge",
                            "decode",
                            {},
                            {{"verifier_path",
                              "all_position_state_publication"}});
                        if (!runner_->materializeDeviceSpeculativeOutcomesForHostResponse(
                                device_outcome_handle,
                                &device_outcome))
                        {
                            return fail_after_checkpoint(
                                "All-position greedy MTP resident outcome host-response materialization failed");
                        }
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
            {
                PerfStatsCollector::ScopedTimer transaction_plan_timer(
                    "mtp",
                    "all_position_transaction_plan_build",
                    "decode",
                    {},
                    {{"source",
                      device_batch_outcome_for_transaction.has_value()
                          ? "device_rejection_outcome"
                          : (deferred_accepted_outcome.has_value()
                                 ? "accepted_outcome"
                                 : "greedy_catchup")}});
                if (device_batch_outcome_for_transaction.has_value())
                {
                    /*
                     * Device stochastic verification has already reduced the
                     * row decisions into accepted counts and committed tokens.
                     * Route that compact outcome through the same batched
                     * transaction driver that future request scheduling will
                     * use.
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
            }
            if (!transaction_plan.ok)
            {
                return fail_after_checkpoint(
                    std::string("All-position MTP verifier transaction plan failed: ") +
                    transaction_plan.error);
            }
            MTPSpecStepPlanBatch &step_plans =
                transaction_plan.step_plans;
            if (step_plans.steps.size() != 1)
            {
                return fail_after_checkpoint(
                    std::string("All-position MTP verifier step-plan failed: ") +
                    "missing single-request step");
            }

            MTPSpecStepPlan &mutable_step = step_plans.steps.front();
            const int accepted_state_count =
                std::max(0, mutable_step.accepted_count);
            bool first_shifted_row_available_for_publication =
                first_shifted_row_available_from_sidecar;
            if (!state_published_from_device_outcome &&
                !first_shifted_row_available_for_publication &&
                !first_token_is_stop &&
                accepted_state_count > 0)
            {
                if (catchup.accepted_tokens.empty())
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier has no token for initial shifted-cache publication");
                }

                bool initial_shifted_commit_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "all_position_initial_shifted_commit",
                        "decode");
                    /*
                     * Non-preserving MoE/TP sidecars are restored away before
                     * verifier publication, but the restored verifier base
                     * still owns terminal hidden for the previous token.  Use
                     * the normal sequential shifted-row helper to publish row
                     * zero exactly as serial decode would, then let verifier
                     * hidden rows publish any later accepted rows below.
                     */
                    initial_shifted_commit_ok =
                        runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                            catchup.accepted_tokens.front(),
                            /*already_appended_tokens=*/0,
                            /*allow_speculative_discard=*/true,
                            static_cast<int>(verifier_base_checkpoint.cached_tokens));
                }
                if (!initial_shifted_commit_ok)
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier initial shifted-cache commit failed");
                }
                first_shifted_row_available_for_publication = true;
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_initial_shifted_commits",
                    1.0,
                    "decode",
                    {},
                    {{"source", "verifier_base_terminal_hidden"}});
            }

            mutable_step.reuse_initial_mtp_shifted_kv_row =
                first_shifted_row_available_for_publication;
            const MTPSpecStepPlan &step = mutable_step;
            if (!state_published_from_device_outcome &&
                !first_token_is_stop &&
                accepted_state_count > 1)
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
                    /*
                     * `already_appended_tokens` is a verifier-row indexing
                     * count.  Row zero hidden is the source for accepted token
                     * one, even when the sidecar row is not a reusable shifted
                     * KV boundary.  Shifted-KV residency is a separate count:
                     * reusable sidecars own one skipped shifted row; restored
                     * MoE/TP verifier-base paths own zero skipped verifier
                     * rows and must be anchored to the verifier-base cached
                     * token count so the pre-commit expectation is
                     * `verifier_base - 1`.
                     */
                    const int shifted_commit_position_offset =
                        first_shifted_row_available_from_sidecar
                            ? base_sidecar_position
                            : static_cast<int>(verifier_base_checkpoint.cached_tokens);
                    const int already_appended_shifted_kv_tokens =
                        first_shifted_row_available_for_publication ? 1 : 0;
                    shifted_catchup_ok =
                        runner_->commitMTPShiftedRowsFromPartialForward(
                            catchup.accepted_tokens.data(),
                            accepted_state_count,
                            /*already_appended_tokens=*/1,
                            catchup.main_forward_token_count,
                            /*allow_speculative_discard=*/true,
                            shifted_commit_position_offset,
                            already_appended_shifted_kv_tokens);
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
            if (!state_published_from_device_outcome)
            {
                if (transaction_plan.requiresDecodeEquivalentReplayPublication())
                {
                    return fail_after_checkpoint(
                        std::string("All-position MTP direct publication received a replay-required transaction plan: ") +
                        transaction_plan.publication_contract_reason);
                }
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
            else
            {
                std::string host_state_error;
                bool host_adoption_ok = false;
                {
                    PerfStatsCollector::ScopedTimer adoption_timer(
                        "mtp",
                        "device_resident_publication_host_adoption",
                        "decode",
                        {},
                        {{"verifier_path",
                          "all_position_state_publication"}});
                    host_adoption_ok =
                        runner_->adoptDeviceResidentMTPSpecPublishedHostState(
                            step_plans,
                            &host_state_error);
                }
                if (!host_adoption_ok)
                {
                    return fail_after_checkpoint(
                        std::string("All-position MTP verifier device-resident host-state adoption failed: ") +
                        host_state_error);
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "device_resident_state_publication_host_plan_checks",
                    1.0,
                    "decode",
                    {},
                    {{"accepted_state_count",
                      std::to_string(accepted_state_count)},
                     {"requires_correction_replay",
                      step.requiresCorrectionReplay() ? "true" : "false"}});
            }

            int correction_forward_count = 0;
            int deferred_correction_condition_count = 0;
            int deferred_correction_start_index = kMTPSpecDecodeInvalidToken;
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
                deferred_correction_start_index = replay_start;
                /*
                 * A rejected correction token is host-visible output, but it is
                 * not live model state yet.  Do not append its shifted-MTP KV
                 * row at the rejecting step: doing so leaves the sidecar cache
                 * one row ahead of the accepted verifier prefix.  The next
                 * all-position verifier step consumes the pending condition row
                 * exactly once and lets the normal sidecar graph append the
                 * shifted row at the same boundary as a serial decode.
                 */
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
            const bool stopped_on_output = catchup.stopped_on_output;
            if (!step.requiresCorrectionReplay() &&
                !all_speculative_accepted &&
                !stopped_on_output)
            {
                const int derived_correction_count =
                    std::max(
                        0,
                        static_cast<int>(accepted_tokens.size()) -
                            accepted_state_count);
                if (derived_correction_count > 0)
                {
                    deferred_correction_start_index = accepted_state_count;
                    deferred_correction_condition_count =
                        derived_correction_count;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "all_position_derived_deferred_correction_condition_tokens",
                        static_cast<double>(
                            deferred_correction_condition_count),
                        "decode",
                        {},
                        {{"verifier_path",
                          "all_position_state_publication"},
                         {"accepted_state_count",
                          std::to_string(accepted_state_count)},
                         {"committed_output_count",
                          std::to_string(accepted_tokens.size())}});
                }
            }
            const int32_t raw_ready_token = catchup.ready_token;
            int32_t ready_token = raw_ready_token;
            const bool has_deferred_correction_condition =
                !all_speculative_accepted &&
                !stopped_on_output &&
                deferred_correction_condition_count > 0;
            if (has_deferred_correction_condition && ready_token >= 0)
            {
                /*
                 * Some verifier paths can already have sampled a token after
                 * the rejected correction row.  That token is useful evidence
                 * for diagnostics, but it is not a serial decode boundary: the
                 * correction token itself has only been emitted, not consumed
                 * as the next condition row.  Suppress terminal ready state and
                 * let the following decode step consume the pending correction
                 * exactly once.
                 */
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_deferred_correction_ready_tokens_suppressed",
                    1.0,
                    "decode",
                    {},
                    {{"verifier_path", "all_position_state_publication"},
                     {"raw_ready_token", std::to_string(raw_ready_token)},
                     {"deferred_condition_tokens",
                      std::to_string(deferred_correction_condition_count)}});
                ready_token = -1;
            }
            const int main_forward_token_count =
                catchup.main_forward_token_count + correction_forward_count;
            result.is_complete = result.is_complete || stopped_on_output;
            const int emitted_token_start_index =
                first_token_is_pending_condition ? 1 : 0;
            if (emitted_token_start_index >
                static_cast<int>(accepted_tokens.size()))
            {
                return fail_after_checkpoint(
                    "All-position MTP pending-condition commit has no matching committed row");
            }
            const int newly_emitted_token_count =
                static_cast<int>(accepted_tokens.size()) -
                emitted_token_start_index;
            std::optional<int32_t> next_pending_condition_token;
            std::optional<DeviceResidentLogicalSequenceStateHandle>
                next_pending_condition_resident_state;
            std::optional<DeviceResidentLogicalSequenceStateHandle>
                ready_condition_resident_state;

            if (!all_speculative_accepted &&
                !stopped_on_output &&
                (ready_token < 0 || has_deferred_correction_condition))
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_rejection_without_ready_token",
                    1.0,
                    "decode",
                    {},
                    {{"deferred_condition_tokens",
                      std::to_string(deferred_correction_condition_count)}});
                if (deferred_correction_condition_count == 1)
                {
                    const int replay_start = deferred_correction_start_index;
                    if (replay_start < 0 ||
                        replay_start >=
                            static_cast<int>(accepted_tokens.size()))
                    {
                        return fail_after_checkpoint(
                            "All-position MTP pending correction row is outside committed outputs");
                    }
                    next_pending_condition_token =
                        accepted_tokens[static_cast<size_t>(replay_start)];
                    if (state_published_from_device_outcome)
                    {
                        DeviceResidentLogicalSequenceStateHandle handle =
                            runner_->deviceResidentLogicalSequenceState();
                        if (!handle.valid())
                        {
                            return fail_after_checkpoint(
                                "All-position MTP direct publication produced no resident logical-state row for the next pending condition");
                        }
                        next_pending_condition_resident_state = handle;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "pending_condition_resident_mailboxes",
                            1.0,
                            "decode",
                            {},
                            {{"request_index", "0"},
                             {"replay_start", std::to_string(replay_start)}});
                    }
                }
                else if (deferred_correction_condition_count > 1)
                {
                    return fail_after_checkpoint(
                        "All-position MTP pending-condition fast path supports one correction row");
                }
            }
            else if (all_speculative_accepted &&
                     !stopped_on_output &&
                     ready_token >= 0 &&
                     state_published_from_device_outcome)
            {
                DeviceResidentLogicalSequenceStateHandle handle =
                    runner_->deviceResidentLogicalSequenceState();
                if (!handle.valid())
                {
                    return fail_after_checkpoint(
                        "All-position MTP direct publication produced no resident logical-state row for the ready token");
                }
                ready_condition_resident_state = handle;
                PerfStatsCollector::addCounter(
                    "mtp",
                    "ready_token_resident_mailboxes",
                    1.0,
                    "decode",
                    {},
                    {{"request_index", "0"},
                     {"ready_token", std::to_string(ready_token)}});
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
                static_cast<double>(newly_emitted_token_count),
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
                 {"output_tokens", std::to_string(newly_emitted_token_count)},
                 {"ready_token", std::to_string(ready_token)},
                 {"raw_ready_token", std::to_string(raw_ready_token)},
                 {"pending_condition_input",
                  first_token_is_pending_condition ? "true" : "false"},
                 {"next_pending_condition_token",
                  next_pending_condition_token.has_value()
                      ? std::to_string(*next_pending_condition_token)
                      : std::string("none")},
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
                if (!direct_sampled_verifier_rows_for_replay_check.empty())
                {
                    replay_context
                        << " direct_all_position_rows="
                        << join_tokens(direct_sampled_verifier_rows_for_replay_check);
                }
                if (auto mismatch = verify_committed_prefix_replay(
                        "all_position_state_publication_verifier",
                        accepted_tokens,
                        ready_token,
                        accepted_state_count,
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
                    /*state_advanced_token_count=*/accepted_state_count,
                    emitted_token_start_index,
                    next_pending_condition_token,
                    next_pending_condition_resident_state,
                    ready_condition_resident_state))
            {
                return fail_after_checkpoint(*commit_error);
            }

            return result;
        }

        if (use_decode_equivalent_replay_publication_verifier)
        {
            const bool grouped_outcome_device_resident_publication =
                use_grouped_outcome_device_resident_publication_verifier;
            if (grouped_outcome_device_resident_publication)
            {
                /*
                 * This is intentionally stricter than "policy selected the
                 * grouped-outcome lane". LocalTP can prove grouped verifier
                 * row math while still lacking a single compact reducer across
                 * all TP shards. Only runners that passed the earlier
                 * device-resident publication gate may enter this hot path.
                 */
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_device_resident_publication_uses",
                    1.0,
                    "decode",
                    {},
                    {{"model_class", mtpDepthPolicyModelClassName(verifier_model_class)},
                     {"probe_rows", std::to_string(verifier_policy_probe_rows)},
                     {"reason", verifier_policy.reason}});
            }
            const bool sidecar_preserves_main_state =
                runner_->supportsMTPSidecarPreservesMainState();
            const PrefixStateSnapshot *sidecar_checkpoint = nullptr;
            if (!sidecar_preserves_main_state)
            {
                if (sidecar_checkpoints.empty())
                {
                    return fail_after_checkpoint(
                        "Decode-equivalent sequential MTP verifier requires a post-sidecar checkpoint");
                }

                sidecar_checkpoint = &sidecar_checkpoints.front();
                if (!sidecar_checkpoint->valid)
                {
                    return fail_after_checkpoint(
                        "Decode-equivalent sequential MTP verifier received an invalid post-sidecar checkpoint");
                }
            }
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
                     {"discarded_sidecar_checkpoint",
                      sidecar_checkpoint ? "true" : "false"}});
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
                         {"discarded_sidecar_checkpoint",
                          sidecar_checkpoint ? "true" : "false"}});
                }
            }
            if (!restored_verifier_base)
            {
                return fail_after_checkpoint(
                    "Decode-equivalent sequential MTP verifier could not restore verifier base checkpoint after sidecar draft");
            }

            if (stochastic_verify && grouped_outcome_device_resident_publication)
            {
                if (!runner_->primaryDeviceId().is_gpu() || !stochastic_device_verify)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP requires GPU device-resident verification");
                }
                if (draft_tokens.size() <= 1)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP requires at least one speculative draft row");
                }
                if (stop_tokens_.size() >
                    static_cast<size_t>(
                        sampling_math::kSpeculativeBatchMaxStopTokens))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP has too many stop tokens for the compact device summary");
                }

                const bool first_token_deferred =
                    first_token == kDeferredMTPFirstTokenShadow;
                const bool first_token_is_stop =
                    !first_token_deferred &&
                    std::find(stop_tokens_.begin(),
                              stop_tokens_.end(),
                              first_token) != stop_tokens_.end();
                if (first_token_is_stop)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP stop-on-first-token short-circuit is not implemented");
                }

                const int compare_rows =
                    static_cast<int>(draft_tokens.size()) - 1;
                bool has_deferred_draft_token = false;
                bool has_host_visible_draft_token = false;
                for (int row = 0; row < compare_rows; ++row)
                {
                    const int32_t draft_token =
                        draft_tokens[static_cast<size_t>(row + 1)];
                    if (draft_token == kDeferredMTPDraftTokenShadow)
                        has_deferred_draft_token = true;
                    else if (draft_token >= 0)
                        has_host_visible_draft_token = true;
                    else
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome stochastic MTP found an invalid draft token");
                    }
                }
                if (has_deferred_draft_token && has_host_visible_draft_token)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP does not support mixed host/deferred draft-token ownership");
                }
                if (has_host_visible_draft_token)
                {
                    if (!runner_->stageStochasticDraftTokensForDeviceVerification(
                            draft_tokens.data() + 1,
                            compare_rows,
                            /*first_draft_slot=*/0))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome stochastic MTP draft-token device staging failed");
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "grouped_outcome_stochastic_draft_token_stages",
                        static_cast<double>(compare_rows),
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"}});
                }

                const MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                    buildSingleRequestVerifierInputPlan(draft_tokens);
                if (!verifier_input_plan.ok)
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome MTP verifier input metadata failed: ") +
                        verifier_input_plan.error);
                }
                if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome MTP verifier row metadata is malformed");
                }

                const int verifier_row_count =
                    verifier_input_plan.compact_logit_row_count;
                const bool needs_device_verifier_tokens =
                    first_token_deferred || has_deferred_draft_token;
                const bool can_defer_grouped_verifier_sync =
                    !active_sampling_params_.has_penalties() &&
                    !first_token_is_stop;
                ScopedMTPAllPositionVerifierSyncDeferral verifier_sync_deferral(
                    runner_.get(),
                    can_defer_grouped_verifier_sync);
                const void *verifier_input_tokens_device = nullptr;
                {
                    PerfStatsCollector::ScopedTimer verifier_timer(
                        "mtp",
                        "grouped_outcome_stochastic_verifier_forward",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"rows", std::to_string(verifier_row_count)}});
                    ScopedMTPSpecVerifierInputPlan verifier_plan_scope(
                        runner_.get(),
                        verifier_input_plan);
                    if (!verifier_plan_scope.installed())
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome MTP verifier could not install row metadata plan");
                    }
                    if (!runner_->setComputeRowIndexedAllPositionLogits(
                            true,
                            verifier_row_count))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome MTP verifier could not enable row-indexed logits");
                    }
                    if (!runner_->setComputeAllPositionLogits(true))
                    {
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        return fail_after_checkpoint(
                            "Grouped-outcome MTP verifier could not enable all-position logits");
                    }
                    if (needs_device_verifier_tokens)
                    {
                        if (first_token_deferred)
                        {
                            verifier_input_tokens_device =
                                runner_->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
                                    /*first_target_sample_slot=*/0,
                                    /*first_draft_slot=*/0,
                                    verifier_input_plan.total_verifier_input_tokens - 1,
                                    verifier_input_plan.total_verifier_input_tokens);
                        }
                        else
                        {
                            verifier_input_tokens_device =
                                runner_->prepareMTPVerifierInputTokensOnDevice(
                                    first_token,
                                    /*first_draft_slot=*/0,
                                    verifier_input_plan.total_verifier_input_tokens - 1,
                                    verifier_input_plan.total_verifier_input_tokens);
                        }
                        if (!verifier_input_tokens_device)
                        {
                            runner_->setComputeAllPositionLogits(false);
                            runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                            return fail_after_checkpoint(
                                "Grouped-outcome MTP verifier could not prepare device token input");
                        }
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_verifier_device_token_inputs",
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
                            std::string("Grouped-outcome MTP verifier forward failed: ") +
                            verifier_forward.error);
                    }
                    if (!runner_->setComputeAllPositionLogits(false))
                    {
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        return fail_after_checkpoint(
                            "Grouped-outcome MTP verifier could not disable all-position logits");
                    }
                    if (!runner_->setComputeRowIndexedAllPositionLogits(false, 0))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome MTP verifier could not disable row-indexed logits");
                    }
                }

                auto inverse_sample_seed_for_thresholds =
                    [&](const float *thresholds, size_t count) -> uint64_t
                {
                    if (active_sampling_params_.seed != 0)
                    {
                        return static_cast<uint64_t>(
                            active_sampling_params_.seed);
                    }

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

                auto apply_grouped_row_penalties_for_history =
                    [&](int bonus_row) -> bool
                {
                    if (!active_sampling_params_.has_penalties())
                        return true;
                    if (first_token_deferred || has_deferred_draft_token)
                        return false;

                    /*
                     * Each verifier row observes the same branch-local sampler
                     * history it would have seen in serial decode: base request
                     * history, the first emitted token, then only earlier draft
                     * rows.  Applying those sparse maps directly to the compact
                     * all-position rows lets the grouped outcome reducer remain
                     * decode-equivalent while publication is still replayed.
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
                        "grouped_outcome_stochastic_penalty_rows_preapplied",
                        static_cast<double>(compare_rows + 1),
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"}});
                    return true;
                };

                const int bonus_row = compare_rows;
                if (!apply_grouped_row_penalties_for_history(bonus_row))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP row penalty application failed");
                }
                if (!runner_->buildStochasticDistributionsOnDevice(
                        DeviceLogitsSource::AllPosition,
                        /*first_row=*/0,
                        DeviceDistributionBuffer::Target,
                        /*first_slot=*/0,
                        /*row_count=*/compare_rows + 1,
                        active_sampling_params_,
                        vocab))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP compact target-row build failed");
                }

                std::vector<float> accept_thresholds;
                std::vector<float> residual_thresholds;
                accept_thresholds.reserve(static_cast<size_t>(compare_rows));
                residual_thresholds.reserve(static_cast<size_t>(compare_rows));
                for (int row = 0; row < compare_rows; ++row)
                {
                    const int row_logical_position =
                        transaction_base_cached_tokens + 1 + row;
                    accept_thresholds.push_back(
                        accept_threshold_for_position(
                            sampler_,
                            row_logical_position));
                    residual_thresholds.push_back(
                        residual_threshold_for_position(
                            sampler_,
                            row_logical_position));
                }

                Sampler bonus_sampler = sampler_;
                const float bonus_threshold =
                    sample_threshold_for_position(
                        bonus_sampler,
                        transaction_base_cached_tokens +
                            static_cast<int>(draft_tokens.size()));
                const uint64_t inverse_sample_seed =
                    inverse_sample_seed_for_thresholds(
                        residual_thresholds.data(),
                        residual_thresholds.size());
                const int inverse_sample_first_logical_position =
                    transaction_base_cached_tokens + 1;

                DeviceSpeculativeOutcomeHandle outcome_handle;
                bool resident_outcome_ok = false;
                {
                    PerfStatsCollector::ScopedTimer outcome_timer(
                        "mtp",
                        "grouped_outcome_stochastic_device_resident_outcome_enqueue",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"rows", std::to_string(compare_rows)}});
                    resident_outcome_ok =
                        first_token_deferred
                            ? runner_->verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(
                                  /*first_target_slot=*/0,
                                  /*first_draft_slot=*/0,
                                  /*draft_tokens=*/nullptr,
                                  accept_thresholds.data(),
                                  residual_thresholds.data(),
                                  compare_rows,
                                  /*first_target_sample_slot=*/0,
                                  stop_tokens_.data(),
                                  static_cast<int>(stop_tokens_.size()),
                                  bonus_row,
                                  bonus_threshold,
                                  &outcome_handle,
                                  inverse_sample_seed,
                                  inverse_sample_first_logical_position,
                                  /*use_vllm_probability_rejection=*/true)
                            : runner_->verifyStochasticDistributionsBatchOutcomeOnDeviceResident(
                                  /*first_target_slot=*/0,
                                  /*first_draft_slot=*/0,
                                  /*draft_tokens=*/nullptr,
                                  accept_thresholds.data(),
                                  residual_thresholds.data(),
                                  compare_rows,
                                  first_token,
                                  stop_tokens_.data(),
                                  static_cast<int>(stop_tokens_.size()),
                                  bonus_row,
                                  bonus_threshold,
                                  &outcome_handle,
                                  inverse_sample_seed,
                                  inverse_sample_first_logical_position,
                                  /*use_vllm_probability_rejection=*/true);
                }
                if (!resident_outcome_ok)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP resident device outcome verifier failed");
                }

                MTPDecodeCatchupGreedyRequest catchup_request;
                catchup_request.draft_tokens = draft_tokens;
                catchup_request.stop_tokens = stop_tokens_;
                catchup_request.base_sidecar_position = base_sidecar_position;
                catchup_request.allow_speculative_discard = true;
                catchup_request.verifier_path =
                    "grouped_decode_equivalent_stochastic";
                catchup_request.implementation_name =
                    "device_batch_outcome_device_resident_publication";
                catchup_request.verifier_base_checkpoint =
                    &verifier_base_checkpoint;

                if (!runner_->supportsDeviceResidentMTPSpecStatePublication())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP requires device-resident accepted-state publication; replay publication is not an accepted production path");
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = outcome_handle;
                publication_request.request_count = 1;
                publication_request.max_draft_tokens =
                    static_cast<int>(draft_tokens.size());
                publication_request.base_sidecar_position =
                    base_sidecar_position;
                publication_request.publish_mtp_shifted_kv = true;

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer direct_publish_timer(
                        "mtp",
                        "grouped_outcome_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"request_count", "1"},
                         {"max_draft_tokens",
                          std::to_string(
                              publication_request.max_draft_tokens)}});
                    if (!runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                            publication_request,
                            &publication_error))
                    {
                        return fail_after_checkpoint(
                            std::string("Grouped-outcome stochastic MTP device-resident state publication failed: ") +
                            publication_error);
                    }
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_device_resident_state_publications",
                    1.0,
                    "decode",
                    {},
                    {{"policy_path", "grouped_outcome_device_resident_publication"},
                     {"request_count", "1"},
                     {"max_draft_tokens",
                      std::to_string(publication_request.max_draft_tokens)}});

                /*
                 * Device publication derives next-condition rows into the
                 * runner mailbox before any compatibility D2H bridge.  Launch
                 * the next first-depth sidecar from that mailbox now so its
                 * stream can overlap the response-token bridge.  This is safe
                 * only because the mailbox owns device-side next-condition
                 * tokens and a readiness event; host transaction state remains
                 * a later mirror, not the producer of sidecar input.
                 */
                bool grouped_first_sidecar_prelaunched = false;
                const bool can_prelaunch_next_first_sidecar =
                    use_sidecar_stream_handoff_for_stochastic &&
                    use_device_draft_token_sidecar &&
                    runner_->supportsMTPSidecarPreservesMainState() &&
                    requested_speculative_draft_count > 0;
                if (can_prelaunch_next_first_sidecar)
                {
                    DeviceResidentLogicalSequenceStateHandle handle =
                        runner_->deviceResidentLogicalSequenceState();
                    if (!handle.valid())
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome stochastic MTP direct publication produced no resident logical-state row for sidecar prelaunch");
                    }
                    {
                        PerfStatsCollector::ScopedTimer prelaunch_timer(
                            "mtp",
                            "stochastic_first_sidecar_prelaunch_enqueue",
                            "decode",
                            {},
                            {{"verifier_path",
                              "grouped_outcome_device_resident_publication"},
                             {"resident_state_kind",
                              "device_publication_mailbox"},
                             {"prelaunch_timing", "pre_bridge"}});
                        if (!runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                handle,
                                /*request_index=*/0))
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome stochastic MTP resident first-sidecar prelaunch failed");
                        }
                    }
                    prelaunched_mtp_first_sidecar_resident_state_ = handle;
                    prelaunched_mtp_first_sidecar_params_ =
                        active_sampling_params_;
                    grouped_first_sidecar_prelaunched = true;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_first_sidecar_prelaunches",
                        1.0,
                        "decode",
                        {},
                        {{"request_index", "0"},
                         {"path", "grouped_outcome_device_resident_publication"},
                         {"resident_state_kind",
                          "device_publication_mailbox"},
                         {"prelaunch_timing", "pre_bridge"},
                         {"stop_tokens",
                          std::to_string(stop_tokens_.size())}});
                }

                DeviceSpeculativeVerifyBatchOutcome device_outcome;
                {
                    /*
                     * Live state has already been published from device
                     * metadata. The compatibility bridge below is only for
                     * served response tokens and temporary host mirror
                     * adoption; it must not be a state mutation dependency.
                     */
                    PerfStatsCollector::ScopedTimer bridge_timer(
                        "mtp",
                        "grouped_outcome_stochastic_device_outcome_host_bridge",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"}});
                    if (!runner_->materializeDeviceSpeculativeOutcomesForHostResponse(
                            outcome_handle,
                            &device_outcome))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome stochastic MTP resident outcome materialization failed");
                    }
                }
                if (device_outcome.sampled_terminal)
                    sampler_ = bonus_sampler;

                MTPDecodeCatchupGreedyResult catchup =
                    buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                        catchup_request,
                        device_outcome);
                if (!catchup.ok)
                    return fail_after_checkpoint(catchup.error);

                MTPSpecDecodeMetadataShape metadata_shape;
                metadata_shape.max_requests = 1;
                metadata_shape.max_draft_tokens =
                    static_cast<int>(draft_tokens.size());
                const std::vector<int> request_ids{0};
                const std::vector<MTPDecodeCatchupGreedyRequest> requests{
                    catchup_request};
                const std::vector<MTPDeviceRejectionBatchOutcome> device_outcomes{
                    device_outcome};
                const std::vector<int32_t> base_cached_tokens{
                    static_cast<int32_t>(
                        verifier_base_checkpoint.cached_tokens)};

                MTPSpecTransactionBatchPlan transaction_plan;
                {
                    PerfStatsCollector::ScopedTimer transaction_plan_timer(
                        "mtp",
                        "grouped_outcome_transaction_plan_build",
                        "decode",
                        {},
                        {{"source", "device_rejection_outcome"}});
                    transaction_plan =
                        buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
                            metadata_shape,
                            request_ids,
                            vocab,
                            requests,
                            device_outcomes,
                            base_cached_tokens);
                }
                if (!transaction_plan.ok)
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome MTP verifier transaction plan failed: ") +
                        transaction_plan.error);
                }
                if (transaction_plan.requiresDecodeEquivalentReplayPublication())
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome MTP direct publication received a replay-required transaction plan: ") +
                        transaction_plan.publication_contract_reason);
                }
                MTPSpecStepPlanBatch &step_plans =
                    transaction_plan.step_plans;
                if (step_plans.steps.size() != 1)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome MTP verifier step-plan failed: missing single-request step");
                }
                const MTPSpecStepPlan &step = step_plans.steps.front();
                const int accepted_state_count =
                    std::max(0, step.accepted_count);

                std::string host_state_error;
                bool host_adoption_ok = false;
                {
                    PerfStatsCollector::ScopedTimer adoption_timer(
                        "mtp",
                        "grouped_outcome_device_resident_host_adoption",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"}});
                    host_adoption_ok =
                        runner_->adoptDeviceResidentMTPSpecPublishedHostState(
                            step_plans,
                            &host_state_error);
                }
                if (!host_adoption_ok)
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome MTP verifier device-resident host-state adoption failed: ") +
                        host_state_error);
                }

                const std::vector<int32_t> accepted_tokens =
                    catchup.accepted_tokens;
                const std::vector<int32_t> verifier_tokens =
                    catchup.verifier_tokens;
                const bool all_speculative_accepted =
                    catchup.all_speculative_accepted;
                const int accepted_speculative_prefix =
                    catchup.accepted_speculative_prefix;
                const int32_t rejected_verified_token =
                    catchup.rejected_verified_token;
                const int32_t raw_ready_token = catchup.ready_token;
                int32_t ready_token = raw_ready_token;
                const bool stopped_on_output = catchup.stopped_on_output;
                result.is_complete = result.is_complete || stopped_on_output;
                const int emitted_token_start_index =
                    first_token_is_pending_condition ? 1 : 0;
                if (emitted_token_start_index >
                    static_cast<int>(accepted_tokens.size()))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome MTP pending-condition commit has no matching committed row");
                }
                const int newly_emitted_token_count =
                    static_cast<int>(accepted_tokens.size()) -
                    emitted_token_start_index;
                if (accepted_state_count >
                    static_cast<int>(accepted_tokens.size()))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome MTP verifier accepted-state publication exceeds committed outputs");
                }

                std::optional<int32_t> next_pending_condition_token;
                std::optional<DeviceResidentLogicalSequenceStateHandle>
                    next_pending_condition_resident_state;
                std::optional<DeviceResidentLogicalSequenceStateHandle>
                    ready_condition_resident_state;

                /*
                 * After a stochastic rejection the correction token is output
                 * to the user but not yet part of live model state.  Publish
                 * only the accepted verifier prefix and carry that correction
                 * as the next device-resident condition row.
                 */
                const int deferred_correction_count =
                    (!all_speculative_accepted && !stopped_on_output)
                        ? std::max(
                              0,
                              static_cast<int>(accepted_tokens.size()) -
                                  accepted_state_count)
                        : 0;
                if (deferred_correction_count == 1)
                {
                    const int replay_start = accepted_state_count;
                    if (replay_start < 0 ||
                        replay_start >=
                            static_cast<int>(accepted_tokens.size()))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome MTP pending correction row is outside committed outputs");
                    }
                    next_pending_condition_token =
                        accepted_tokens[static_cast<size_t>(replay_start)];
                    DeviceResidentLogicalSequenceStateHandle handle =
                        runner_->deviceResidentLogicalSequenceState();
                    if (!handle.valid())
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome MTP direct publication produced no resident logical-state row for the pending correction");
                    }
                    next_pending_condition_resident_state = handle;
                    if (ready_token >= 0)
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_deferred_correction_ready_tokens_suppressed",
                            1.0,
                            "decode",
                            {},
                            {{"raw_ready_token", std::to_string(raw_ready_token)},
                             {"policy_path",
                              "grouped_outcome_device_resident_publication"}});
                        ready_token = -1;
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "grouped_outcome_pending_condition_resident_mailboxes",
                        1.0,
                        "decode",
                        {},
                        {{"request_index", "0"},
                         {"replay_start", std::to_string(replay_start)}});
                }
                else if (deferred_correction_count > 1)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome MTP pending-condition fast path supports one correction row");
                }
                else if (all_speculative_accepted &&
                         !stopped_on_output &&
                         ready_token >= 0)
                {
                    DeviceResidentLogicalSequenceStateHandle handle =
                        runner_->deviceResidentLogicalSequenceState();
                    if (!handle.valid())
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome MTP direct publication produced no resident logical-state row for the ready token");
                    }
                    ready_condition_resident_state = handle;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "grouped_outcome_ready_token_resident_mailboxes",
                        1.0,
                        "decode",
                        {},
                        {{"request_index", "0"},
                         {"ready_token", std::to_string(ready_token)}});
                }

                /*
                 * The grouped verifier can resolve to either a ready bonus
                 * row or a pending correction row.  Do not enqueue the next
                 * sidecar from the raw live mailbox before that outcome is
                 * known: a rejection can make the correct next condition
                 * differ from the currently published accepted-state row.
                 * Once the transaction has selected the resident continuation
                 * handle, prelaunch from exactly the handle that commit will
                 * hand to the next decode step.
                 */
                std::optional<DeviceResidentLogicalSequenceStateHandle>
                    resolved_prelaunch_state;
                const char *resolved_prelaunch_kind = "none";
                if (next_pending_condition_resident_state.has_value())
                {
                    resolved_prelaunch_state =
                        next_pending_condition_resident_state;
                    resolved_prelaunch_kind = "pending_condition";
                }
                else if (ready_condition_resident_state.has_value())
                {
                    resolved_prelaunch_state = ready_condition_resident_state;
                    resolved_prelaunch_kind = "ready_token";
                }
                if (can_prelaunch_next_first_sidecar &&
                    !grouped_first_sidecar_prelaunched &&
                    !stopped_on_output &&
                    resolved_prelaunch_state.has_value())
                {
                    if (!resolved_prelaunch_state->valid())
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome stochastic MTP selected a stale resident logical-state row for sidecar prelaunch");
                    }
                    {
                        PerfStatsCollector::ScopedTimer prelaunch_timer(
                            "mtp",
                            "stochastic_first_sidecar_prelaunch_enqueue",
                            "decode",
                            {},
                            {{"verifier_path",
                              "grouped_outcome_device_resident_publication"},
                             {"resident_state_kind", resolved_prelaunch_kind},
                             {"prelaunch_timing",
                              "post_outcome_fallback"}});
                        if (!runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                *resolved_prelaunch_state,
                                /*request_index=*/0))
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome stochastic MTP resolved resident first-sidecar prelaunch failed");
                        }
                    }
                    prelaunched_mtp_first_sidecar_resident_state_ =
                        *resolved_prelaunch_state;
                    prelaunched_mtp_first_sidecar_params_ =
                        active_sampling_params_;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_first_sidecar_prelaunches",
                        1.0,
                        "decode",
                        {},
                        {{"request_index", "0"},
                         {"path", "grouped_outcome_device_resident_publication"},
                         {"resident_state_kind", resolved_prelaunch_kind},
                         {"prelaunch_timing", "post_outcome_fallback"},
                         {"stop_tokens",
                          std::to_string(stop_tokens_.size())}});
                }

                const bool grouped_metadata_has_deferred_tokens =
                    first_token_deferred || has_deferred_draft_token;
                if (grouped_metadata_has_deferred_tokens)
                {
                    /*
                     * Device-resident grouped verification intentionally keeps
                     * sampled draft ids in device slots.  The host mirror uses
                     * negative sentinels, so validate transaction metadata from
                     * the already-reduced device outcome instead of pretending
                     * those sentinels are real vocabulary tokens.
                     */
                    MTPSpecDecodeAcceptedOutcome accepted_outcome;
                    accepted_outcome.request_id = 0;
                    accepted_outcome.vocab_size = vocab;
                    accepted_outcome.draft_count =
                        static_cast<int>(draft_tokens.size());
                    accepted_outcome.committed_output_tokens =
                        accepted_tokens;
                    if (!stopped_on_output &&
                        all_speculative_accepted &&
                        ready_token >= 0)
                    {
                        accepted_outcome.bonus_ready_token = ready_token;
                    }
                    accepted_outcome.accepted_verifier_input_prefix =
                        std::min<int>(
                            static_cast<int>(draft_tokens.size()),
                            std::max(0, accepted_speculative_prefix) + 1);
                    accepted_outcome.target_verifier_state_commit_count =
                        catchup.target_verifier_state_commit_count;
                    accepted_outcome.all_drafts_accepted =
                        all_speculative_accepted;
                    accepted_outcome.stopped_on_output = stopped_on_output;

                    if (auto tx_error = validate_spec_decode_accepted_outcome(
                            "grouped_decode_equivalent_stochastic_verifier",
                            "device_batch_outcome_device_resident_publication",
                            accepted_outcome))
                    {
                        return fail_after_checkpoint(*tx_error);
                    }
                }
                else if (auto tx_error = validate_spec_decode_transaction(
                             "grouped_decode_equivalent_stochastic_verifier",
                             "device_batch_outcome_device_resident_publication",
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
                    static_cast<uint64_t>(
                        verifier_input_plan.total_verifier_input_tokens);
                PerfStatsCollector::addCounter(
                    "mtp",
                    "verifier_runs",
                    1.0,
                    "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "verifier_tokens",
                    static_cast<double>(
                        verifier_input_plan.total_verifier_input_tokens),
                    "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_decode_equivalent_stochastic_verifier_runs",
                    1.0,
                    "decode",
                    {},
                    {{"verifier_forward_tokens",
                     std::to_string(verifier_input_plan.total_verifier_input_tokens)},
                     {"verifier_rows", std::to_string(verifier_row_count)},
                     {"replay_forward_tokens", "0"},
                     {"shifted_commits", "0"},
                     {"accepted_tokens",
                      std::to_string(accepted_tokens.size())},
                     {"state_publication", "device_resident"}});

                mtp_stats_.stochastic_accept_tests +=
                    static_cast<uint64_t>(
                        std::max(0, device_outcome.consumed_verifier_rows));
                mtp_stats_.stochastic_accepts +=
                    static_cast<uint64_t>(
                        std::max(0, device_outcome.accepted_speculative_prefix));
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_accept_tests",
                    static_cast<double>(
                        std::max(0, device_outcome.consumed_verifier_rows)),
                    "decode",
                    {},
                    {{"device_resident", "true"},
                     {"verifier_path", "grouped_decode_equivalent_stochastic"},
                     {"implementation", "device_batch_outcome_device_resident_publication"}});
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_accepts",
                    static_cast<double>(
                        std::max(0, device_outcome.accepted_speculative_prefix)),
                    "decode",
                    {},
                    {{"verifier_path", "grouped_decode_equivalent_stochastic"},
                     {"implementation", "device_batch_outcome_device_resident_publication"}});

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
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "rejected_tokens",
                        1.0,
                        "decode");
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "rollbacks",
                        1.0,
                        "decode");
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "transaction_rollbacks",
                        1.0,
                        "decode");
                    if (device_outcome.rejected_verified_token >= 0)
                    {
                        ++mtp_stats_.stochastic_residual_samples;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "stochastic_residual_device_samples",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path",
                              "grouped_decode_equivalent_stochastic"},
                             {"implementation",
                              "device_batch_outcome_device_resident_publication"}});
                    }
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
                        {{"verifier_path",
                          "grouped_decode_equivalent_stochastic"},
                         {"implementation",
                          "device_batch_outcome_device_resident_publication"}});
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
                    static_cast<double>(newly_emitted_token_count),
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
                     {"rejected_verified_token",
                      std::to_string(rejected_verified_token)},
                     {"accepted_speculative_prefix",
                      std::to_string(accepted_speculative_prefix)},
                     {"all_speculative_accepted",
                      all_speculative_accepted ? "true" : "false"},
                     {"verifier_state_matches_output", "true"},
                     {"verifier_path",
                      "grouped_decode_equivalent_stochastic"},
                     {"catchup_implementation",
                      "device_batch_outcome_device_resident_publication"},
                     {"policy_path", "grouped_outcome_device_resident_publication"},
                     {"decode_equivalent_replay_required", "false"},
                     {"output_tokens", std::to_string(newly_emitted_token_count)},
                     {"ready_token", std::to_string(ready_token)},
                     {"raw_ready_token", std::to_string(raw_ready_token)},
                     {"accepted_state_count",
                      std::to_string(accepted_state_count)},
                     {"pending_condition_input",
                      first_token_is_pending_condition ? "true" : "false"},
                     {"next_pending_condition_token",
                      next_pending_condition_token.has_value()
                          ? std::to_string(*next_pending_condition_token)
                          : std::string("none")},
                     {"used_ready_logits", use_ready_logits ? "true" : "false"}});

                if (!stopped_on_output && ready_token >= 0)
                {
                    if (auto mismatch = verify_committed_prefix_replay(
                            "grouped_decode_equivalent_stochastic_verifier",
                            accepted_tokens,
                            ready_token))
                    {
                        return fail_after_checkpoint(*mismatch);
                    }
                }

                if (auto commit_error = commit_mtp_transaction_outputs(
                        "grouped_decode_equivalent_stochastic_verifier",
                        verifier_base_checkpoint,
                        accepted_tokens,
                        stopped_on_output || ready_token < 0
                            ? std::optional<int32_t>{}
                            : std::optional<int32_t>{ready_token},
                        /*terminal_logits_ready=*/!stopped_on_output && ready_token >= 0,
                        /*is_complete=*/stopped_on_output,
                        PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent,
                        /*state_advanced=*/true,
                        accepted_state_count,
                        emitted_token_start_index,
                        next_pending_condition_token,
                        next_pending_condition_resident_state,
                        ready_condition_resident_state))
                {
                    return fail_after_checkpoint(*commit_error);
                }

                return result;
            }

            if (!stochastic_verify && grouped_outcome_device_resident_publication)
            {
                if (!runner_->primaryDeviceId().is_gpu())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP requires a GPU device-resident verifier");
                }
                if (!runner_->supportsGreedyAllPositionBatchOutcomeOnDevice())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP requires compact greedy device outcome support");
                }
                if (!runner_->supportsDeviceResidentMTPSpecStatePublication())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP requires device-resident accepted-state publication");
                }
                if (stop_tokens_.size() >
                    static_cast<size_t>(
                        sampling_math::kSpeculativeBatchMaxStopTokens))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP has too many stop tokens for the compact device summary");
                }
                if (draft_tokens.empty() ||
                    draft_tokens.size() >
                        static_cast<size_t>(
                            sampling_math::kSpeculativeBatchMaxOutputTokens))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP draft width is outside compact verifier limits");
                }

                const bool first_token_deferred =
                    first_token == kDeferredMTPFirstTokenShadow;
                const bool has_deferred_draft_token =
                    std::find(
                        draft_tokens.begin() + 1,
                        draft_tokens.end(),
                        kDeferredMTPDraftTokenShadow) != draft_tokens.end();
                if (use_sampling_penalties &&
                    (first_token_deferred || has_deferred_draft_token))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP cannot apply row-local penalties to deferred token shadows");
                }

                const MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                    buildSingleRequestVerifierInputPlan(draft_tokens);
                if (!verifier_input_plan.ok)
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome greedy MTP verifier input metadata failed: ") +
                        verifier_input_plan.error);
                }
                if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP verifier row metadata is malformed");
                }

                const int verifier_row_count =
                    verifier_input_plan.compact_logit_row_count;
                const bool needs_device_verifier_tokens =
                    first_token_deferred || has_deferred_draft_token ||
                    use_greedy_device_draft_slots ||
                    runner_->primaryDeviceId().is_gpu();
                ScopedMTPAllPositionVerifierSyncDeferral verifier_sync_deferral(
                    runner_.get(),
                    true);
                const void *verifier_input_tokens_device = nullptr;
                {
                    PerfStatsCollector::ScopedTimer verifier_timer(
                        "mtp",
                        "grouped_outcome_greedy_verifier_forward",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"rows", std::to_string(verifier_row_count)}});
                    ScopedMTPSpecVerifierInputPlan verifier_plan_scope(
                        runner_.get(),
                        verifier_input_plan);
                    if (!verifier_plan_scope.installed())
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP verifier could not install row metadata plan");
                    }
                    if (!runner_->setComputeRowIndexedAllPositionLogits(
                            true,
                            verifier_row_count))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP verifier could not enable row-indexed logits");
                    }
                    if (!runner_->setComputeAllPositionLogits(true))
                    {
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP verifier could not enable all-position logits");
                    }
                    if (needs_device_verifier_tokens)
                    {
                        if (first_token_deferred)
                        {
                            verifier_input_tokens_device =
                                runner_->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
                                    /*first_target_sample_slot=*/0,
                                    /*first_draft_slot=*/0,
                                    verifier_input_plan.total_verifier_input_tokens - 1,
                                    verifier_input_plan.total_verifier_input_tokens);
                        }
                        else if (use_greedy_device_draft_slots ||
                                 has_deferred_draft_token)
                        {
                            verifier_input_tokens_device =
                                runner_->prepareMTPVerifierInputTokensOnDevice(
                                    first_token,
                                    /*first_draft_slot=*/0,
                                    verifier_input_plan.total_verifier_input_tokens - 1,
                                    verifier_input_plan.total_verifier_input_tokens);
                        }
                        else
                        {
                            /*
                             * Host-visible warmup rows still feed the compact
                             * resident reducer.  Stage the complete row once on
                             * the verifier stream so the forward graph and the
                             * outcome reducer consume one coherent device row.
                             */
                            verifier_input_tokens_device =
                                runner_->prepareMTPVerifierInputTokensOnDeviceFromHostRow(
                                    verifier_input_plan.verifier_input_tokens.data(),
                                    verifier_input_plan.total_verifier_input_tokens,
                                    verifier_input_plan.total_verifier_input_tokens - 1);
                        }
                        if (!verifier_input_tokens_device)
                        {
                            runner_->setComputeAllPositionLogits(false);
                            runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                            return fail_after_checkpoint(
                                "Grouped-outcome greedy MTP verifier could not prepare device token input");
                        }
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_verifier_device_token_inputs",
                            1.0,
                            "decode",
                            {},
                            {{"total_tokens",
                              std::to_string(
                                  verifier_input_plan.total_verifier_input_tokens)},
                             {"sampling", "greedy"}});
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
                            std::string("Grouped-outcome greedy MTP verifier forward failed: ") +
                            verifier_forward.error);
                    }
                    if (!runner_->setComputeAllPositionLogits(false))
                    {
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP verifier could not disable all-position logits");
                    }
                    if (!runner_->setComputeRowIndexedAllPositionLogits(false, 0))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP verifier could not disable row-indexed logits");
                    }
                }

                auto apply_grouped_greedy_row_penalties_for_history =
                    [&](int bonus_row) -> bool
                {
                    if (!use_sampling_penalties)
                        return true;

                    Sampler row_penalty_sampler = sampler_;
                    row_penalty_sampler.record_token(first_token);
                    const int compare_rows =
                        static_cast<int>(draft_tokens.size()) - 1;
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
                        "grouped_outcome_greedy_penalty_rows_preapplied",
                        static_cast<double>(compare_rows + 1),
                        "decode",
                        {},
                        {{"policy_path",
                          "grouped_outcome_device_resident_publication"}});
                    return true;
                };

                const int compare_rows =
                    static_cast<int>(draft_tokens.size()) - 1;
                const int bonus_row = compare_rows;
                if (!apply_grouped_greedy_row_penalties_for_history(bonus_row))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP row penalty application failed");
                }

                DeviceSpeculativeOutcomeHandle outcome_handle;
                {
                    PerfStatsCollector::ScopedTimer outcome_timer(
                        "mtp",
                        "grouped_outcome_greedy_device_resident_outcome_enqueue",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"rows", std::to_string(compare_rows)}});
                    if (!runner_->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                            draft_tokens.data(),
                            static_cast<int>(draft_tokens.size()),
                            stop_tokens_.data(),
                            static_cast<int>(stop_tokens_.size()),
                            &outcome_handle))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP resident device outcome verifier failed");
                    }
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = outcome_handle;
                publication_request.request_count = 1;
                publication_request.max_draft_tokens =
                    static_cast<int>(draft_tokens.size());
                publication_request.base_sidecar_position =
                    base_sidecar_position;
                publication_request.publish_mtp_shifted_kv = true;

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer direct_publish_timer(
                        "mtp",
                        "grouped_outcome_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"request_count", "1"},
                         {"max_draft_tokens",
                          std::to_string(
                              publication_request.max_draft_tokens)},
                         {"sampling", "greedy"}});
                    if (!runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                            publication_request,
                            &publication_error))
                    {
                        return fail_after_checkpoint(
                            std::string("Grouped-outcome greedy MTP device-resident state publication failed: ") +
                            publication_error);
                    }
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_device_resident_state_publications",
                    1.0,
                    "decode",
                    {},
                    {{"policy_path", "grouped_outcome_device_resident_publication"},
                     {"request_count", "1"},
                     {"max_draft_tokens",
                      std::to_string(publication_request.max_draft_tokens)},
                     {"sampling", "greedy"}});

                const bool can_prelaunch_next_first_sidecar =
                    use_sidecar_stream_handoff_for_grouped_greedy &&
                    use_device_draft_token_sidecar &&
                    runner_->supportsMTPSidecarPreservesMainState() &&
                    requested_speculative_draft_count > 0;

                DeviceSpeculativeVerifyBatchOutcome device_outcome;
                {
                    /*
                     * State has already been committed from compact device
                     * metadata.  This bridge is only the compatibility mirror
                     * for response tokens and host transaction bookkeeping.
                     */
                    PerfStatsCollector::ScopedTimer bridge_timer(
                        "mtp",
                        "grouped_outcome_greedy_device_outcome_host_bridge",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"}});
                    if (!runner_->materializeDeviceSpeculativeOutcomesForHostResponse(
                            outcome_handle,
                            &device_outcome))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP resident outcome materialization failed");
                    }
                }
                /*
                 * Keep compatibility response materialization as the first
                 * host-visible boundary after verifier outcome reduction.  A
                 * previous pre-bridge prelaunch could make ROCm's response
                 * bridge wait behind speculative next-step sidecar work.  The
                 * sidecar is still prelaunched for reuse by the following
                 * decode step, but only after emitted response tokens are
                 * already materialized.
                 */
                if (can_prelaunch_next_first_sidecar)
                {
                    DeviceResidentLogicalSequenceStateHandle handle =
                        runner_->deviceResidentLogicalSequenceState();
                    if (!handle.valid())
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP direct publication produced no resident logical-state row for sidecar prelaunch");
                    }
                    {
                        PerfStatsCollector::ScopedTimer prelaunch_timer(
                            "mtp",
                            "stochastic_first_sidecar_prelaunch_enqueue",
                            "decode",
                            {},
                            {{"verifier_path",
                              "grouped_outcome_device_resident_publication"},
                             {"resident_state_kind",
                              "device_publication_mailbox"},
                             {"prelaunch_timing", "post_bridge"},
                             {"sampling", "greedy"}});
                        if (!runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                handle,
                                /*request_index=*/0))
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome greedy MTP resident first-sidecar prelaunch failed");
                        }
                    }
                    prelaunched_mtp_first_sidecar_resident_state_ = handle;
                    prelaunched_mtp_first_sidecar_params_ =
                        active_sampling_params_;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_first_sidecar_prelaunches",
                        1.0,
                        "decode",
                        {},
                        {{"request_index", "0"},
                         {"path", "grouped_outcome_device_resident_publication"},
                         {"resident_state_kind",
                          "device_publication_mailbox"},
                         {"prelaunch_timing", "post_bridge"},
                         {"sampling", "greedy"},
                         {"stop_tokens",
                          std::to_string(stop_tokens_.size())}});
                }

                MTPDecodeCatchupGreedyRequest catchup_request;
                catchup_request.draft_tokens = draft_tokens;
                catchup_request.stop_tokens = stop_tokens_;
                catchup_request.base_sidecar_position = base_sidecar_position;
                catchup_request.allow_speculative_discard = true;
                catchup_request.verifier_path =
                    "grouped_decode_equivalent_greedy";
                catchup_request.implementation_name =
                    "device_batch_outcome_device_resident_publication";
                catchup_request.verifier_base_checkpoint =
                    &verifier_base_checkpoint;

                MTPDecodeCatchupGreedyResult catchup =
                    buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                        catchup_request,
                        device_outcome);
                if (!catchup.ok)
                    return fail_after_checkpoint(catchup.error);

                MTPSpecDecodeMetadataShape metadata_shape;
                metadata_shape.max_requests = 1;
                metadata_shape.max_draft_tokens =
                    static_cast<int>(draft_tokens.size());
                const std::vector<int> request_ids{0};
                const std::vector<MTPDecodeCatchupGreedyRequest> requests{
                    catchup_request};
                const std::vector<MTPDeviceRejectionBatchOutcome> device_outcomes{
                    device_outcome};
                const std::vector<int32_t> base_cached_tokens{
                    static_cast<int32_t>(
                        verifier_base_checkpoint.cached_tokens)};

                MTPSpecTransactionBatchPlan transaction_plan;
                {
                    PerfStatsCollector::ScopedTimer transaction_plan_timer(
                        "mtp",
                        "grouped_outcome_transaction_plan_build",
                        "decode",
                        {},
                        {{"source", "device_rejection_outcome"},
                         {"sampling", "greedy"}});
                    transaction_plan =
                        buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
                            metadata_shape,
                            request_ids,
                            vocab,
                            requests,
                            device_outcomes,
                            base_cached_tokens);
                }
                if (!transaction_plan.ok)
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome greedy MTP verifier transaction plan failed: ") +
                        transaction_plan.error);
                }
                if (transaction_plan.requiresDecodeEquivalentReplayPublication())
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome greedy MTP direct publication received a replay-required transaction plan: ") +
                        transaction_plan.publication_contract_reason);
                }

                MTPSpecStepPlanBatch &step_plans =
                    transaction_plan.step_plans;
                if (step_plans.steps.size() != 1)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP verifier step-plan failed: missing single-request step");
                }
                const MTPSpecStepPlan &step = step_plans.steps.front();
                const int accepted_state_count =
                    std::max(0, step.accepted_count);

                std::string host_state_error;
                bool host_adoption_ok = false;
                {
                    PerfStatsCollector::ScopedTimer adoption_timer(
                        "mtp",
                        "grouped_outcome_device_resident_host_adoption",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"sampling", "greedy"}});
                    host_adoption_ok =
                        runner_->adoptDeviceResidentMTPSpecPublishedHostState(
                            step_plans,
                            &host_state_error);
                }
                if (!host_adoption_ok)
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome greedy MTP verifier device-resident host-state adoption failed: ") +
                        host_state_error);
                }

                const std::vector<int32_t> accepted_tokens =
                    catchup.accepted_tokens;
                const std::vector<int32_t> verifier_tokens =
                    catchup.verifier_tokens;
                const bool all_speculative_accepted =
                    catchup.all_speculative_accepted;
                const int accepted_speculative_prefix =
                    catchup.accepted_speculative_prefix;
                const int32_t rejected_verified_token =
                    catchup.rejected_verified_token;
                const int32_t raw_ready_token = catchup.ready_token;
                int32_t ready_token = raw_ready_token;
                const bool stopped_on_output = catchup.stopped_on_output;
                result.is_complete = result.is_complete || stopped_on_output;
                const int emitted_token_start_index =
                    first_token_is_pending_condition ? 1 : 0;
                if (emitted_token_start_index >
                    static_cast<int>(accepted_tokens.size()))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP pending-condition commit has no matching committed row");
                }
                const int newly_emitted_token_count =
                    static_cast<int>(accepted_tokens.size()) -
                    emitted_token_start_index;
                if (accepted_state_count >
                    static_cast<int>(accepted_tokens.size()))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP accepted-state publication exceeds committed outputs");
                }

                std::optional<int32_t> next_pending_condition_token;
                std::optional<DeviceResidentLogicalSequenceStateHandle>
                    next_pending_condition_resident_state;
                std::optional<DeviceResidentLogicalSequenceStateHandle>
                    ready_condition_resident_state;

                const int deferred_correction_count =
                    (!all_speculative_accepted && !stopped_on_output)
                        ? std::max(
                              0,
                              static_cast<int>(accepted_tokens.size()) -
                                  accepted_state_count)
                        : 0;
                if (deferred_correction_count == 1)
                {
                    const int replay_start = accepted_state_count;
                    if (replay_start < 0 ||
                        replay_start >=
                            static_cast<int>(accepted_tokens.size()))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP pending correction row is outside committed outputs");
                    }
                    next_pending_condition_token =
                        accepted_tokens[static_cast<size_t>(replay_start)];
                    DeviceResidentLogicalSequenceStateHandle handle =
                        runner_->deviceResidentLogicalSequenceState();
                    if (!handle.valid())
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP direct publication produced no resident logical-state row for the pending correction");
                    }
                    next_pending_condition_resident_state = handle;
                    if (ready_token >= 0)
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_deferred_correction_ready_tokens_suppressed",
                            1.0,
                            "decode",
                            {},
                            {{"raw_ready_token", std::to_string(raw_ready_token)},
                             {"policy_path",
                              "grouped_outcome_device_resident_publication"},
                             {"sampling", "greedy"}});
                        ready_token = -1;
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "grouped_outcome_pending_condition_resident_mailboxes",
                        1.0,
                        "decode",
                        {},
                        {{"request_index", "0"},
                         {"replay_start", std::to_string(replay_start)},
                         {"sampling", "greedy"}});
                }
                else if (deferred_correction_count > 1)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP pending-condition fast path supports one correction row");
                }
                else if (all_speculative_accepted &&
                         !stopped_on_output &&
                         ready_token >= 0)
                {
                    DeviceResidentLogicalSequenceStateHandle handle =
                        runner_->deviceResidentLogicalSequenceState();
                    if (!handle.valid())
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP direct publication produced no resident logical-state row for the ready token");
                    }
                    ready_condition_resident_state = handle;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "grouped_outcome_ready_token_resident_mailboxes",
                        1.0,
                        "decode",
                        {},
                        {{"request_index", "0"},
                         {"ready_token", std::to_string(ready_token)},
                         {"sampling", "greedy"}});
                }

                const bool grouped_metadata_has_deferred_tokens =
                    first_token_deferred || has_deferred_draft_token;
                if (grouped_metadata_has_deferred_tokens)
                {
                    MTPSpecDecodeAcceptedOutcome accepted_outcome;
                    accepted_outcome.request_id = 0;
                    accepted_outcome.vocab_size = vocab;
                    accepted_outcome.draft_count =
                        static_cast<int>(draft_tokens.size());
                    accepted_outcome.committed_output_tokens =
                        accepted_tokens;
                    if (!stopped_on_output &&
                        all_speculative_accepted &&
                        ready_token >= 0)
                    {
                        accepted_outcome.bonus_ready_token = ready_token;
                    }
                    accepted_outcome.accepted_verifier_input_prefix =
                        std::min<int>(
                            static_cast<int>(draft_tokens.size()),
                            std::max(0, accepted_speculative_prefix) + 1);
                    accepted_outcome.target_verifier_state_commit_count =
                        catchup.target_verifier_state_commit_count;
                    accepted_outcome.all_drafts_accepted =
                        all_speculative_accepted;
                    accepted_outcome.stopped_on_output = stopped_on_output;

                    if (auto tx_error = validate_spec_decode_accepted_outcome(
                            "grouped_decode_equivalent_greedy_verifier",
                            "device_batch_outcome_device_resident_publication",
                            accepted_outcome))
                    {
                        return fail_after_checkpoint(*tx_error);
                    }
                }
                else if (auto tx_error = validate_spec_decode_transaction(
                             "grouped_decode_equivalent_greedy_verifier",
                             "device_batch_outcome_device_resident_publication",
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
                    static_cast<uint64_t>(
                        verifier_input_plan.total_verifier_input_tokens);
                PerfStatsCollector::addCounter(
                    "mtp",
                    "verifier_runs",
                    1.0,
                    "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "verifier_tokens",
                    static_cast<double>(
                        verifier_input_plan.total_verifier_input_tokens),
                    "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_decode_equivalent_greedy_verifier_runs",
                    1.0,
                    "decode",
                    {},
                    {{"verifier_forward_tokens",
                      std::to_string(
                          verifier_input_plan.total_verifier_input_tokens)},
                     {"verifier_rows", std::to_string(verifier_row_count)},
                     {"replay_forward_tokens", "0"},
                     {"shifted_commits", "0"},
                     {"accepted_tokens",
                      std::to_string(accepted_tokens.size())},
                     {"state_publication", "device_resident"}});

                recordMTPDepthObservation(
                    requested_speculative_draft_count,
                    speculative_draft_count,
                    accepted_speculative_prefix,
                    draft_count_budget_limited,
                    !all_speculative_accepted);

                if (!all_speculative_accepted)
                {
                    ++mtp_stats_.rejected_tokens;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "rejected_tokens",
                        1.0,
                        "decode");
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
                    static_cast<double>(newly_emitted_token_count),
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
                     {"rejected_verified_token",
                      std::to_string(rejected_verified_token)},
                     {"accepted_speculative_prefix",
                      std::to_string(accepted_speculative_prefix)},
                     {"all_speculative_accepted",
                      all_speculative_accepted ? "true" : "false"},
                     {"verifier_state_matches_output", "true"},
                     {"verifier_path",
                      "grouped_decode_equivalent_greedy"},
                     {"catchup_implementation",
                      "device_batch_outcome_device_resident_publication"},
                     {"policy_path", "grouped_outcome_device_resident_publication"},
                     {"decode_equivalent_replay_required", "false"},
                     {"output_tokens", std::to_string(newly_emitted_token_count)},
                     {"ready_token", std::to_string(ready_token)},
                     {"raw_ready_token", std::to_string(raw_ready_token)},
                     {"accepted_state_count",
                      std::to_string(accepted_state_count)},
                     {"pending_condition_input",
                      first_token_is_pending_condition ? "true" : "false"},
                     {"next_pending_condition_token",
                      next_pending_condition_token.has_value()
                          ? std::to_string(*next_pending_condition_token)
                          : std::string("none")},
                     {"used_ready_logits", use_ready_logits ? "true" : "false"}});

                if (!stopped_on_output && ready_token >= 0)
                {
                    if (auto mismatch = verify_committed_prefix_replay(
                            "grouped_decode_equivalent_greedy_verifier",
                            accepted_tokens,
                            ready_token))
                    {
                        return fail_after_checkpoint(*mismatch);
                    }
                }

                if (auto commit_error = commit_mtp_transaction_outputs(
                        "grouped_decode_equivalent_greedy_verifier",
                        verifier_base_checkpoint,
                        accepted_tokens,
                        stopped_on_output || ready_token < 0
                            ? std::optional<int32_t>{}
                            : std::optional<int32_t>{ready_token},
                        /*terminal_logits_ready=*/!stopped_on_output && ready_token >= 0,
                        /*is_complete=*/stopped_on_output,
                        PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent,
                        /*state_advanced=*/true,
                        accepted_state_count,
                        emitted_token_start_index,
                        next_pending_condition_token,
                        next_pending_condition_resident_state,
                        ready_condition_resident_state))
                {
                    return fail_after_checkpoint(*commit_error);
                }

                return result;
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
                        transaction_base_cached_tokens + draft_idx;
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
                        if (draft_token == kDeferredMTPDraftTokenShadow)
                        {
                            return fail_after_checkpoint(
                                "Decode-equivalent stochastic MTP cannot verify a deferred draft token without a host-visible shadow");
                        }
                        if (draft_token < 0)
                        {
                            return fail_after_checkpoint(
                                "Decode-equivalent stochastic MTP found an invalid draft token before device verifier staging");
                        }
                        /*
                         * The decode-equivalent verifier restores the main
                         * base checkpoint after sidecar drafting.  That restore
                         * can deliberately clear runner-local device readiness
                         * metadata, so publish the host-visible draft shadow
                         * into the verifier-owned device slot at the exact row
                         * boundary where it is consumed.  CUDA and ROCm share
                         * this runner contract; backend code only sees a
                         * prepared device token slot plus an explicit stream
                         * readiness event.
                         */
                        if (!runner_->stageStochasticDraftTokensForDeviceVerification(
                                &draft_token,
                                /*draft_token_count=*/1,
                                /*first_draft_slot=*/row))
                        {
                            return fail_after_checkpoint(
                                "Decode-equivalent stochastic MTP draft-token device staging failed");
                        }
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "decode_equivalent_stochastic_draft_token_stages",
                            1.0,
                            "decode",
                            {},
                            {{"row", std::to_string(row)}});
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
                                                transaction_base_cached_tokens +
                                                    static_cast<int>(accepted_tokens.size())))
                                      : sampleDistributionWithThreshold(
                                            host_target_distribution,
                                            sample_threshold_for_position(
                                                sampler_,
                                                transaction_base_cached_tokens +
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

            Sampler verifier_penalty_sampler = sampler_;
            auto sample_after_forward = [&](int32_t forwarded_token) -> int32_t
            {
                if (forwarded_token >= 0)
                {
                    verifier_penalty_sampler.record_token(forwarded_token);
                }

                bool penalties_applied_to_logits = false;
                if (use_sampling_penalties)
                {
                    auto penalty_map =
                        verifier_penalty_sampler.compute_penalty_map(
                            active_sampling_params_,
                            vocab);
                    if (!penalty_map.empty())
                    {
                        if (!runner_->applyPenaltiesOnDevice(penalty_map, vocab))
                        {
                            return -1;
                        }
                        penalties_applied_to_logits = true;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "decode_equivalent_catchup_penalty_applications",
                            1.0,
                            "decode",
                            {},
                            {{"implementation", "shared_stepwise"}});
                    }
                }

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
                SamplingParams host_params = active_sampling_params_;
                if (penalties_applied_to_logits)
                {
                    host_params.presence_penalty = 0.0f;
                    host_params.frequency_penalty = 0.0f;
                    host_params.dry_multiplier = 0.0f;
                    host_params.dry_penalty_last_n = 0;
                }
                return verifier_penalty_sampler.sample(
                    main_logits,
                    static_cast<size_t>(vocab),
                    host_params);
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
                 {"catchup_implementation", catchup_implementation},
                 {"policy_path", grouped_outcome_device_resident_publication
                                     ? "grouped_outcome_device_resident_publication"
                                     : "decode_equivalent_sequential"}});

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
                 {"policy_path", grouped_outcome_device_resident_publication
                                      ? "grouped_outcome_device_resident_publication"
                                      : "decode_equivalent_sequential"},
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

        // Broadcast to worker ranks so they run decode in lockstep.  The
        // current token budget is part of the decode command: Qwen thinking
        // budget handling calls setDecodeStepTokenBudget(1) only on rank 0,
        // and MTP draft-depth clamping must be identical on every rank or TP
        // collectives diverge inside the sidecar/verifier graphs.
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::DECODE_STEP);
            int32_t token_budget = static_cast<int32_t>(decode_step_token_budget_);
            mpi_ctx_->broadcast_int32(&token_budget, 1, 0);
        }

        const bool mpi_coordinated_world =
            mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->world_size() > 1;
        const bool mpi_worker_rank =
            mpi_coordinated_world && mpi_ctx_->rank() != 0;
        auto trace_position = [this](const char *context) -> int
        {
            return currentMTPBaseSidecarPositionForPlanning(context, nullptr)
                .value_or(-1);
        };
        if (traceChatGeneratedTokensEnabled())
        {
            LOG_INFO("[OrchestrationRunner/decodeStep] begin rank="
                     << (mpi_ctx_ ? mpi_ctx_->rank() : 0)
                     << " coordinated=" << (mpi_coordinated_world ? "true" : "false")
                     << " worker=" << (mpi_worker_rank ? "true" : "false")
                     << " prefill_logits_ready=" << (prefill_logits_ready_ ? "true" : "false")
                     << " position=" << trace_position("trace_decode_step_begin"));
        }

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
                prelaunched_mtp_first_sidecar_resident_state_.reset();
                prelaunched_mtp_first_sidecar_params_.reset();
                recordMTPDepthZeroBypass();
                adaptive_depth_zero_step = true;
            }
            else
            {
                prelaunched_mtp_first_sidecar_resident_state_.reset();
                prelaunched_mtp_first_sidecar_params_.reset();
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
                ready_sampled_resident_state_.reset();
            }
            prefill_logits_ready_ = false;
            LOG_TRACE("[decodeStep] Using prefill logits (skipping forward)");
        }
        else
        {
            ready_sampled_token_.reset();
            ready_sampled_params_.reset();
            ready_sampled_resident_state_.reset();
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
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << (mpi_ctx_ ? mpi_ctx_->rank() : 0)
                         << " forward begin token=" << last_token_);
            }
            if (!runner_->forward(&last_token_, 1))
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                result.error = "Forward pass failed during decode";
                return result;
            }
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << (mpi_ctx_ ? mpi_ctx_->rank() : 0)
                         << " forward complete position="
                         << trace_position("trace_decode_step_forward_complete"));
            }
            pending_mtp_condition_token_.reset();
            pending_mtp_condition_params_.reset();
            pending_mtp_condition_resident_state_.reset();
        }

        // Tail stage: try GPU-side sampling first, fall back to CPU
        // When penalties are active, compute the sparse penalty map on CPU,
        // upload to GPU, apply in-place, then sample on GPU.
        // This avoids the full ~600KB D2H transfer of logits.
        int token = -1;

        const bool mpi_sampling_collective_required =
            mpi_coordinated_world &&
            runner_->requiresMPICoordinatedDecodeSampling(active_sampling_params_);
        const bool worker_sampling_required =
            mpi_worker_rank && mpi_sampling_collective_required;
        if (mpi_worker_rank && !ready_token_for_decode.has_value())
        {
            /*
             * Worker ranks must participate in any device/distributed sampling
             * collectives, but rank 0 owns the committed token.  Do not run
             * root-only CPU fallback here; after the post-sampling fence below,
             * workers receive the authoritative token and record that in their
             * local sampler history.
             */
            if (!worker_sampling_required)
            {
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                             << mpi_ctx_->rank()
                             << " worker sampling skipped; awaiting root token");
                }
                token = 0;
            }
            else if (active_sampling_params_.has_penalties())
            {
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                             << mpi_ctx_->rank()
                             << " worker sampling with penalties begin");
                }
                int vocab = vocabSize();
                auto penalty_map =
                    sampler_.compute_penalty_map(active_sampling_params_, vocab);
                bool gpu_penalties_applied = penalty_map.empty() ||
                    runner_->applyPenaltiesOnDevice(penalty_map, vocab);
                if (gpu_penalties_applied)
                {
                    token = active_sampling_params_.is_greedy()
                                ? runner_->sampleGreedyOnDevice()
                                : runner_->sampleOnDevice(active_sampling_params_);
                }
            }
            else
            {
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                             << mpi_ctx_->rank()
                             << " worker sampling begin");
                }
                token = active_sampling_params_.is_greedy()
                            ? runner_->sampleGreedyOnDevice()
                            : runner_->sampleOnDevice(active_sampling_params_);
            }
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " worker sampling complete token=" << token);
            }
            if (token < 0)
                token = 0;
        }
        else if (ready_token_for_decode.has_value())
        {
            token = *ready_token_for_decode;
            PerfStatsCollector::addCounter("mtp", "ready_token_direct_emits", 1.0, "decode");
        }
        else if (active_sampling_params_.has_penalties())
        {
            // Compute sparse penalty map on CPU (presence + frequency + DRY)
            int vocab = vocabSize();
            auto penalty_map = sampler_.compute_penalty_map(active_sampling_params_, vocab);

            if (!mpi_coordinated_world || mpi_sampling_collective_required)
            {
                if (!penalty_map.empty())
                {
                    // Try GPU-side penalty application + sampling.
                    bool gpu_penalties_applied = runner_->applyPenaltiesOnDevice(penalty_map, vocab);
                    if (gpu_penalties_applied)
                    {
                        // Penalties applied on GPU — now sample from penalized logits.
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
                    // No penalties to apply — sample directly on GPU.
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
            // If GPU path failed, fall through to CPU fallback below
        }
        else if (active_sampling_params_.is_greedy())
        {
            // Try GPU-side greedy (argmax)
            if (!mpi_coordinated_world || mpi_sampling_collective_required)
            {
                token = runner_->sampleGreedyOnDevice();
            }
        }
        else
        {
            // Try GPU-side top-k/top-p
            if (!mpi_coordinated_world || mpi_sampling_collective_required)
            {
                token = runner_->sampleOnDevice(active_sampling_params_);
                if (token >= 0)
                {
                    LOG_TRACE("[decodeStep] GPU top-k/top-p sampled token=" << token);
                }
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

        if (mpi_coordinated_world)
        {
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " entering post-sampling coordinated fence");
            }
            mpi_ctx_->barrier();
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " leaving post-sampling coordinated fence");
            }

            int32_t coordinated_token = static_cast<int32_t>(token);
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " publishing/receiving coordinated token candidate="
                         << coordinated_token);
            }
            mpi_ctx_->broadcast_int32(&coordinated_token, 1, 0);
            token = coordinated_token;
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " coordinated token=" << token);
            }
        }

        const bool token_is_stop =
            std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
        if (adaptive_depth_zero_step && !token_is_stop)
        {
            std::string position_error;
            const std::optional<int> base_sidecar_position =
                currentMTPBaseSidecarPositionForPlanning(
                    "dynamic depth-zero bypass",
                    &position_error);
            if (!base_sidecar_position)
            {
                result.error = position_error;
                return result;
            }
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
                        *base_sidecar_position);
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

    GenerationResult OrchestrationRunner::forceDecodeToken(int32_t token)
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
                "forceDecodeToken() cannot consume request-batched prefill state";
            return result;
        }

        const int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        auto trace_position = [this](const char *context) -> int
        {
            return currentMTPBaseSidecarPositionForPlanning(context, nullptr)
                .value_or(-1);
        };
        if (traceChatGeneratedTokensEnabled())
        {
            LOG_INFO("[OrchestrationRunner/forceDecodeToken] begin rank="
                     << rank
                     << " token=" << token
                     << " last_token=" << last_token_
                     << " prefill_logits_ready=" << (prefill_logits_ready_ ? "true" : "false")
                     << " position=" << trace_position("trace_force_decode_begin"));
        }

        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 &&
            mpi_ctx_->world_size() > 1)
        {
            if (traceChatGeneratedTokensEnabled())
                LOG_INFO("[OrchestrationRunner/forceDecodeToken] broadcasting forced token");
            broadcastCommand(MPICommand::FORCE_DECODE_TOKEN);
            int32_t forced = token;
            mpi_ctx_->broadcast_int32(&forced, 1, 0);
        }

        const bool coordinated_force_command =
            mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->world_size() > 1;
        ScopedMPICoordinatedCommandFence force_command_fence(
            mpi_ctx_.get(),
            coordinated_force_command,
            "forceDecodeToken");

        /*
         * A forced token is a real generated token chosen by request policy
         * rather than by the sampler.  If terminal logits are already ready
         * from prefill or MTP publication, no main-state row has been appended
         * for the next position yet; replacing the sampled choice starts as a
         * metadata update.  Otherwise we must first append the previous
         * last_token_ with a normal one-token forward so KV/GDN state reaches
         * the forced token's logical position.
         */
        if (prefill_logits_ready_)
        {
            prefill_logits_ready_ = false;
            ready_sampled_token_.reset();
            ready_sampled_params_.reset();
            ready_sampled_resident_state_.reset();
        }
        else
        {
            runner_->setMTPMainDecodeSyncDeferralEnabled(false);
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/forceDecodeToken] rank="
                         << rank << " forwarding previous token " << last_token_);
            }
            if (!runner_->forward(&last_token_, 1))
            {
                result.error = "Forward pass failed while committing forced token";
                return result;
            }
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/forceDecodeToken] rank="
                         << rank
                         << " forward complete position="
                         << trace_position("trace_force_decode_forward_complete"));
            }
        }

        /*
         * MTP maintains a shifted sidecar KV stream: each emitted non-stop token
         * must publish a row derived from the terminal hidden state immediately
         * before that token.  Normal decode and verifier catch-up do this before
         * forwarding each accepted token.  Forced policy tokens, such as the
         * Qwen stop-thinking phrase, must follow the same contract or the main
         * KV/GDN state advances while the shifted sidecar cache stays behind.
         */
        const bool token_is_stop =
            std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
        if (shouldUseMTPDecode() && !token_is_stop)
        {
            std::string position_error;
            const std::optional<int> base_sidecar_position =
                currentMTPBaseSidecarPositionForPlanning(
                    "forced decode token",
                    &position_error);
            if (!base_sidecar_position)
            {
                result.error = position_error;
                return result;
            }

            bool shifted_commit_ok = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "forced_token_shifted_commit",
                    "decode");
                shifted_commit_ok =
                    runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                        token,
                        /*already_appended_tokens=*/0,
                        /*allow_speculative_discard=*/true,
                        *base_sidecar_position);
            }
            if (!shifted_commit_ok)
            {
                result.error =
                    "MTP forced-token shifted-cache maintenance failed";
                return result;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "forced_token_shifted_commits",
                1.0,
                "decode");
        }

        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();

        sampler_.record_token(token);
        last_token_ = token;
        result.tokens.push_back(token);
        result.is_complete = token_is_stop;
        if (traceChatGeneratedTokensEnabled())
        {
            LOG_INFO("[OrchestrationRunner/forceDecodeToken] done rank="
                     << rank
                     << " token=" << token
                     << " position=" << trace_position("trace_force_decode_done"));
        }
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
        ready_sampled_resident_state_.reset();
        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
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
        const auto prepopulate_page_cache = [&](const std::string &reason)
        {
            LOG_DEBUG(reason << " pre-populating page cache for mmap load...");
            uintmax_t model_file_bytes = 0;
            try
            {
                model_file_bytes = std::filesystem::file_size(model_path);
            }
            catch (const std::exception &)
            {
                model_file_bytes = 0;
            }
            auto start = std::chrono::steady_clock::now();
            const bool ok = MmapRegion::prepopulatePageCache(model_path);
            auto elapsed = std::chrono::steady_clock::now() - start;
            const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
            const double elapsed_s = elapsed_ms / 1000.0;
            const double mb = static_cast<double>(model_file_bytes) / (1024.0 * 1024.0);
            const double mbps = (model_file_bytes > 0 && elapsed_s > 0.0) ? (mb / elapsed_s) : 0.0;
            WeightLoadingProfiler::addDetail("weights.mmap_page_cache_prepopulate", elapsed_ms);
            PerfStatsCollector::addCounter("weight_loading", "mmap_page_cache_prepopulate_success",
                                           ok ? 1.0 : 0.0, "load", {},
                                           {{"reason", reason}});
            PerfStatsCollector::addCounter("weight_loading", "mmap_page_cache_prepopulate_bytes",
                                           static_cast<double>(model_file_bytes), "load", {},
                                           {{"reason", reason},
                                            {"success", perfBool(ok)}});
            PerfStatsCollector::addCounter("weight_loading", "mmap_page_cache_prepopulate_mbps",
                                           mbps, "load", {},
                                           {{"reason", reason},
                                            {"success", perfBool(ok)}});
            if (!ok)
            {
                LOG_WARN(reason << " page-cache prepopulation failed; continuing with mmap demand faults");
            }
            else if (model_file_bytes > 0 && mbps > 0.0 && mbps < 256.0)
            {
                LOG_INFO(reason << " page-cache prepopulation was slow: "
                                << std::fixed << std::setprecision(0) << mbps
                                << " MB/s for " << std::fixed << std::setprecision(0)
                                << mb << " MB. Subsequent counters can distinguish disk/cache speed from GPU staging.");
            }
            return ok;
        };

        if (!is_multi_rank && config_.use_mmap)
        {
            // Single-rank loads need the same protection as multi-rank loads.
            //
            // CPU: NUMA-bound parallel first-touch otherwise creates many cold
            // page-fault streams.
            //
            // GPU: the demand-paged upload path copies tensor-sized mmap ranges
            // into pinned staging slots.  On a cold file those copies can also
            // defeat disk readahead and collapse to ~100 MB/s.  A single
            // sequential prewarm keeps the upload path demand-paged while making
            // the backing reads deterministic and full-bandwidth.
            prepopulate_page_cache(weight_config.target_is_gpu
                                       ? "Single-rank GPU"
                                       : "Single-rank CPU");
            weight_config.skip_mmap_cache_eviction = true;
        }
        else if (is_multi_rank && config_.use_mmap)
        {
            const auto *topo = mpi_ctx_->topology();
            if (topo)
            {
                // Per-node prepopulation: each node leader warms its own page cache
                if (topo->is_node_leader())
                {
                    std::ostringstream reason;
                    reason << "Node leader (rank " << mpi_ctx_->rank()
                           << ", node " << topo->placement().node_id << ")";
                    prepopulate_page_cache(reason.str());
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
                    prepopulate_page_cache("Rank 0 fallback");
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
        mtp_verifier_economy_perfstats_emitted_ = false;
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
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[MPIWorkerLoop] rank " << mpi_ctx_->rank()
                                                 << " received command tag " << tag);
            }

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
                int32_t token_budget = 0;
                mpi_ctx_->broadcast_int32(&token_budget, 1, 0);
                setDecodeStepTokenBudget(token_budget);
                decodeStep();
                // Rank 0 scopes thinking-budget decode through ChatCompletionHandler;
                // reset workers too so forced-token or later control commands never
                // observe a stale request-local budget.
                setDecodeStepTokenBudget(0);
                break;
            }

            case MPICommand::FORCE_DECODE_TOKEN:
            {
                int32_t forced = 0;
                mpi_ctx_->broadcast_int32(&forced, 1, 0);
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[MPIWorkerLoop] rank " << mpi_ctx_->rank()
                                                     << " received forced token "
                                                     << forced);
                }
                GenerationResult forced_result = forceDecodeToken(forced);
                if (!forced_result.success())
                    LOG_ERROR("[MPIWorkerLoop] forceDecodeToken failed on rank "
                              << mpi_ctx_->rank() << ": " << forced_result.error);
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
