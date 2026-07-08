/**
 * @file RankOrchestrator.cpp
 * @brief Multi-device orchestrator implementation for LOCAL tensor parallelism
 * @author David Sanftenberg
 * @date January 2026
 *
 * Implements coordination of multiple DeviceGraphOrchestrator instances for LOCAL
 * tensor parallelism across multiple devices within a single MPI rank.
 *
 * Key features:
 * - Parallel forward pass execution across devices via TPWorkerPool (TP) / std::async (PP)
 * - AllGather for combining partial logits from column-parallel LM head
 * - Unified snapshot/profiling API across all device runners
 */

#include "RankOrchestrator.h"
#include "LogitsGatherer.h"
#include "DeviceSampler.h"
#include "DeviceGraphOrchestrator.h"
#include "../../../collective/CollectiveTimeoutPolicy.h"
#include "../../mtp/MTPSpecTransactionDriver.h"
#include "../../mtp/MTPSpecStateContract.h"
#include "../../factory/InferenceRunnerFactory.h"
#include "../../moe/MoEExpertOverlayRuntimePlan.h"
#include "../../prefix_cache/PrefixCacheCoordinator.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../kernels/cpu/sampling/CPUSamplerPrimitives.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../collective/ILocalPPContext.h"
#include "../../../config/TensorParallelConfig.h"
#include "../../../interfaces/IModelContext.h"
#include "../../../loaders/ModelContext.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../loaders/WeightManager.h"
#include "../graph/SchemaFactoryRegistry.h" // Model-agnostic sharding config access
#include "../../../tensors/TensorClasses.h"
#include "../../../tensors/TensorFactory.h"
#include "../../../backends/BackendManager.h"       // getBackendFor() for partial D2H in gatherLogits
#include "../../../backends/GPUDeviceContextPool.h" // Compute stream registration for event-based collective sync
#include "../../../utils/Logger.h"
#include "../../../utils/Sampler.h"            // SamplingParams for sampleOnDevice()
#include "../../../utils/KernelProfiler.h"     // Phase propagation to worker threads
#include "../../../utils/ROCmKernelProfiler.h" // Phase propagation to worker threads
#include "../../../utils/CUDAKernelProfiler.h" // Phase propagation to worker threads
#include "../../../utils/KVCacheProfiler.h"    // Phase propagation to worker threads
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/VramBillOfMaterials.h"
#include "../../mpi_orchestration/RankExecutionPlan.h" // For Config::fromPlan()
#include "../../../collective/PPActivationContract.h"  // PPActivationContract for forwardPP
#include "fort.hpp"                                    // libfort for TP profiling summary table
#include <algorithm>
#include <array>
#include <future>
#include <iomanip>
#include <limits>
#include <numeric>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <utility>
#ifdef __linux__
#include <malloc.h> // malloc_trim for deferred host weight release
#endif

namespace llaminar2
{
    namespace rank_orchestrator_detail
    {
        /**
         * @brief Emit a structured reason when a rank-level prefix snapshot cannot be built.
         *
         * Multi-device MTP treats rollback checkpoint capture as an all-or-nothing
         * transaction.  If one participant cannot provide a snapshot, or if two
         * participants disagree about the live token count, the parent
         * RankOrchestrator must reject the entire checkpoint.  This helper keeps
         * that rejection observable in perfstats and logs so a failed end-to-end
         * parity cell does not collapse into the generic
         * "could not capture live prefix state" message.
         *
         * @param operation Human-readable operation name, for example
         *        "capture_live_prefix_checkpoint".
         * @param runner_group Which child collection was being traversed
         *        ("device" for LocalTP or "pp" for LocalPP).
         * @param participant_index Stable index among non-null participants.
         * @param runner Child runner that produced the failure, or nullptr if
         *        the failure happened before a child call.
         * @param reason Short machine-readable reason tag.
         * @param expected_tokens Common token count already observed, or -1 when
         *        no common count exists yet.
         * @param actual_tokens Token count from the current child, or -1 when
         *        unavailable.
         */
        void recordRankPrefixSnapshotFailure(
            const char *operation,
            const char *runner_group,
            size_t participant_index,
            const IInferenceRunner *runner,
            const std::string &reason,
            int expected_tokens = -1,
            int actual_tokens = -1)
        {
            const std::string operation_name =
                operation && operation[0] != '\0'
                    ? operation
                    : "capture_live_prefix_snapshot";
            const std::string group_name =
                runner_group && runner_group[0] != '\0'
                    ? runner_group
                    : "unknown";
            const std::string device_name =
                runner ? runner->primaryDeviceId().toString() : "unknown";

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_live_prefix_snapshot_failures",
                1.0,
                "decode",
                device_name,
                {{"operation", operation_name},
                 {"runner_group", group_name},
                 {"participant_index", std::to_string(participant_index)},
                 {"reason", reason},
                 {"expected_tokens", std::to_string(expected_tokens)},
                 {"actual_tokens", std::to_string(actual_tokens)}});

            LOG_WARN("[RankOrchestrator] " << operation_name
                                           << " failed for " << group_name
                                           << " participant=" << participant_index
                                           << " device=" << device_name
                                           << " reason=" << reason
                                           << " expected_tokens=" << expected_tokens
                                           << " actual_tokens=" << actual_tokens);
        }

        bool sameBackendGpuExpertTransferIsDirectOnly(DeviceId destination, DeviceId source)
        {
            return destination.is_gpu() &&
                   source.is_gpu() &&
                   destination.type == source.type;
        }

        std::vector<std::vector<std::vector<bool>>> buildReplicaArrivalTransferMasks(
            const MoERebalanceController &controller,
            const ExpertReplicaSet &arrivals)
        {
            int participants = arrivals.num_sockets;
            if (participants <= 0)
                participants = controller.currentReplicas().num_sockets;
            if (participants <= 0 && !arrivals.owner_socket.empty())
            {
                const auto max_owner = std::max_element(
                    arrivals.owner_socket.begin(),
                    arrivals.owner_socket.end());
                if (max_owner != arrivals.owner_socket.end() && *max_owner >= 0)
                    participants = *max_owner + 1;
            }

            const int num_layers = controller.numLayers();
            const int num_experts = controller.numExperts();
            std::vector<std::vector<std::vector<bool>>> masks_by_participant(
                static_cast<size_t>(std::max(0, participants)),
                std::vector<std::vector<bool>>(
                    static_cast<size_t>(std::max(0, num_layers)),
                    std::vector<bool>(static_cast<size_t>(std::max(0, num_experts)), false)));

            if (participants <= 0 || num_layers <= 0 || num_experts <= 0)
                return masks_by_participant;

            const int expert_limit = std::min(
                num_experts,
                static_cast<int>(arrivals.owner_socket.size()));
            for (int participant = 0; participant < participants; ++participant)
            {
                for (int layer = 0; layer < num_layers; ++layer)
                {
                    for (int expert_id = 0; expert_id < expert_limit; ++expert_id)
                    {
                        const int owner = arrivals.owner_socket[static_cast<size_t>(expert_id)];
                        if (owner < 0 || owner >= participants || participant == owner)
                            continue;
                        if (arrivals.hasReplicaOnParticipant(layer, expert_id, participant))
                            masks_by_participant[static_cast<size_t>(participant)][static_cast<size_t>(layer)][static_cast<size_t>(expert_id)] = true;
                    }
                }
            }

            return masks_by_participant;
        }

        std::vector<std::vector<std::vector<bool>>> buildOwnershipArrivalTransferMasks(
            const MoERebalanceController &controller,
            const std::vector<int> &previous_placement)
        {
            const int participants = controller.participantCount();
            const int num_layers = controller.numLayers();
            const int num_experts = controller.numExperts();
            std::vector<std::vector<std::vector<bool>>> masks_by_participant(
                static_cast<size_t>(std::max(0, participants)),
                std::vector<std::vector<bool>>(
                    static_cast<size_t>(std::max(0, num_layers)),
                    std::vector<bool>(static_cast<size_t>(std::max(0, num_experts)), false)));

            if (participants <= 0 || num_layers <= 0 || num_experts <= 0)
                return masks_by_participant;

            const auto &current_placement = controller.currentParticipantPlacement();
            const int expert_limit = std::min({
                num_experts,
                static_cast<int>(previous_placement.size()),
                static_cast<int>(current_placement.size())});

            for (int expert_id = 0; expert_id < expert_limit; ++expert_id)
            {
                const int previous_owner = previous_placement[static_cast<size_t>(expert_id)];
                const int current_owner = current_placement[static_cast<size_t>(expert_id)];
                if (current_owner < 0 || current_owner >= participants ||
                    previous_owner == current_owner)
                {
                    continue;
                }

                for (int layer = 0; layer < num_layers; ++layer)
                    masks_by_participant[static_cast<size_t>(current_owner)][static_cast<size_t>(layer)][static_cast<size_t>(expert_id)] = true;
            }

            return masks_by_participant;
        }
    }

    namespace
    {
        /**
         * @brief Decode shared SamplingMath metadata into the public outcome ABI.
         *
         * DeviceGraphOrchestrator uses the same metadata layout after its CUDA
         * and ROCm reducers run.  RankOrchestrator's LocalTP greedy reducer
         * produces the compact row tokens by reducing child-local vocab shards,
         * then calls the same SamplingMath summarizer so accepted-prefix,
         * ready-token, stop-token, and consumed-row semantics cannot drift from
         * the single-device path.
         */
        bool fillRankSpeculativeVerifyOutcomeFromMeta(
            const std::array<int32_t, sampling_math::kSpeculativeBatchMaxOutputTokens> &output_tokens,
            const std::array<int, sampling_math::kSpeculativeBatchMetaCount> &meta,
            DeviceSpeculativeVerifyBatchOutcome *out)
        {
            using namespace sampling_math;
            if (!out ||
                meta[kSpecBatchMetaOk] == 0 ||
                meta[kSpecBatchMetaOutputCount] < 0 ||
                meta[kSpecBatchMetaOutputCount] > kSpeculativeBatchMaxOutputTokens)
            {
                return false;
            }
            for (int i = 0; i < meta[kSpecBatchMetaOutputCount]; ++i)
            {
                if (output_tokens[static_cast<size_t>(i)] < 0)
                    return false;
            }

            out->ok = true;
            out->output_tokens = output_tokens;
            out->output_token_count = meta[kSpecBatchMetaOutputCount];
            out->accepted_speculative_prefix =
                meta[kSpecBatchMetaAcceptedSpeculativePrefix];
            out->target_verifier_state_commit_count =
                meta[kSpecBatchMetaTargetVerifierStateCommitCount];
            out->ready_token = meta[kSpecBatchMetaReadyToken];
            out->rejected_verified_token = meta[kSpecBatchMetaRejectedVerifiedToken];
            out->stopped_on_output = meta[kSpecBatchMetaStoppedOnOutput] != 0;
            out->all_speculative_accepted =
                meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
            out->consumed_verifier_rows = meta[kSpecBatchMetaConsumedVerifierRows];
            out->sampled_terminal = meta[kSpecBatchMetaSampledTerminal] != 0;
            return true;
        }

        std::vector<DeviceId> mmapReleaseSyncDevices(const RankOrchestrator::Config &config)
        {
            std::vector<DeviceId> devices;

            auto add_device = [&devices](const GlobalDeviceAddress &address)
            {
                if (!address.isGPU())
                    return;

                const DeviceId local_device = address.toLocalDeviceId();
                if (std::find(devices.begin(), devices.end(), local_device) == devices.end())
                    devices.push_back(local_device);
            };

            for (const auto &device : config.devices)
                add_device(device);

            for (const auto &stage : config.pp_stages)
            {
                for (const auto &device : stage.stage_devices)
                    add_device(device);
            }

            return devices;
        }

        bool synchronizeGpuBackendsBeforeRankMmapRelease(const RankOrchestrator::Config &config)
        {
            bool ok = true;
            for (const DeviceId &device : mmapReleaseSyncDevices(config))
            {
                IBackend *backend = getBackendFor(device);
                if (!backend)
                {
                    LOG_ERROR("RankOrchestrator: no backend available for " << device
                                                                            << " before mmap DONTNEED");
                    ok = false;
                    continue;
                }

                if (debugEnv().vram_trace)
                {
                    LOG_TRACE("[VRAM_TRACE] rank_mmap_release.before_sync device=" << device);
                }
                else
                {
                    LOG_DEBUG("RankOrchestrator: synchronizing " << device
                                                                 << " before mmap DONTNEED");
                }

                if (!backend->synchronize(device.gpu_ordinal()))
                {
                    LOG_ERROR("RankOrchestrator: failed to synchronize " << device
                                                                         << " before mmap DONTNEED");
                    ok = false;
                }
                else if (debugEnv().vram_trace)
                {
                    LOG_TRACE("[VRAM_TRACE] rank_mmap_release.after_sync device=" << device);
                }
            }
            return ok;
        }

        const ExpertComputeDomain *findMoEExpertDomain(
            const MoEExpertParallelPlan &plan,
            const std::string &name)
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                                   [&](const auto &domain)
                                   {
                                       return domain.name == name;
                                   });
            return it == plan.domains.end() ? nullptr : &*it;
        }

        bool routedOverlayUsesApportionedExperts(const std::shared_ptr<MoEExpertParallelPlan> &plan)
        {
            if (!plan || !plan->isTieredOverlay())
                return false;

            for (const auto &tier : plan->routed_tiers)
            {
                const auto *domain = findMoEExpertDomain(*plan, tier.domain);
                if (domain && domain->compute_kind == ExpertDomainComputeKind::ApportionedExperts)
                    return true;
            }
            return false;
        }

        bool moeOverlayDenseTPEnabled(const std::shared_ptr<MoEExpertParallelPlan> &plan)
        {
            if (!plan || !plan->isTieredOverlay())
                return true;
            return plan->continuation_domain_spec.dense_tp_enabled;
        }

        void forceRoutedMoEExpertParentsReplicated(WeightShardingConfig &sharding)
        {
            for (auto &pattern : sharding.patterns)
            {
                if (pattern.pattern == "ffn_gate_exps.weight" ||
                    pattern.pattern == "ffn_up_exps.weight" ||
                    pattern.pattern == "ffn_down_exps.weight")
                {
                    pattern.mode = WeightShardingMode::Replicate;
                    pattern.description = "MoE routed expert weights - replicated for graph-native overlay";
                }
            }
        }

        [[noreturn]] void abortAfterTPWorkerTimeout(
            const char *operation,
            int timeout_ms,
            size_t completed,
            size_t expected)
        {
            LOG_ERROR("RankOrchestrator::" << operation
                                           << ": TP worker timeout after "
                                           << timeout_ms
                                           << "ms (completed "
                                           << completed
                                           << "/" << expected
                                           << "). Aborting process to avoid "
                                           << "hanging on stuck worker teardown.");
            std::abort();
        }
    }

    // =========================================================================
    // Config Implementation
    // =========================================================================

    bool RankOrchestrator::PPStageConfig::validate() const
    {
        // Layer range must be valid
        if (last_layer <= first_layer)
        {
            LOG_ERROR("PPStageConfig: Invalid layer range [" << first_layer << ", " << last_layer << ")");
            return false;
        }

        // Must have at least one device
        if (stage_devices.empty())
        {
            LOG_ERROR("PPStageConfig: No stage devices specified");
            return false;
        }

        // If TP weights are provided, must match device count
        if (!tp_weights.empty() && tp_weights.size() != stage_devices.size())
        {
            LOG_ERROR("PPStageConfig: TP weights count (" << tp_weights.size()
                                                          << ") doesn't match device count (" << stage_devices.size() << ")");
            return false;
        }

        // If TP weights are provided, must sum to approximately 1.0
        if (!tp_weights.empty())
        {
            float sum = std::accumulate(tp_weights.begin(), tp_weights.end(), 0.0f);
            if (std::abs(sum - 1.0f) > 0.01f)
            {
                LOG_ERROR("PPStageConfig: TP weights sum to " << sum << ", expected 1.0");
                return false;
            }
        }

        return true;
    }

    RankOrchestrator::ParallelismMode
    RankOrchestrator::Config::detectMode() const
    {
        if (pp_stages.empty())
        {
            // No PP stages - pure TP mode
            return ParallelismMode::TP;
        }

        // Check if any PP stage is a TP domain
        bool has_tp_stages = std::any_of(pp_stages.begin(), pp_stages.end(),
                                         [](const PPStageConfig &stage)
                                         { return stage.isTPDomain(); });

        return has_tp_stages ? ParallelismMode::TP_PP : ParallelismMode::PP;
    }

    std::vector<int> RankOrchestrator::Config::buildLayerBoundaries() const
    {
        std::vector<int> boundaries;
        if (pp_stages.empty())
        {
            return boundaries;
        }

        boundaries.push_back(0);
        for (const auto &stage : pp_stages)
        {
            boundaries.push_back(stage.last_layer);
        }
        return boundaries;
    }

    bool RankOrchestrator::Config::validate() const
    {
        ParallelismMode effective = effectiveMode();

        if (effective == ParallelismMode::TP)
        {
            // TP mode validation
            if (devices.empty())
            {
                LOG_ERROR("RankOrchestrator::Config: No devices specified for TP mode");
                return false;
            }

            // If weights are provided, must match device count
            if (!weights.empty() && weights.size() != devices.size())
            {
                LOG_ERROR("RankOrchestrator::Config: Weights count (" << weights.size()
                                                                      << ") doesn't match device count (" << devices.size() << ")");
                return false;
            }

            // If weights are provided, must sum to approximately 1.0
            if (!weights.empty())
            {
                float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
                if (std::abs(sum - 1.0f) > 0.01f)
                {
                    LOG_ERROR("RankOrchestrator::Config: Weights sum to " << sum << ", expected 1.0");
                    return false;
                }
            }
        }
        else
        {
            // PP or TP_PP mode validation
            if (pp_stages.empty())
            {
                LOG_ERROR("RankOrchestrator::Config: No PP stages specified for PP mode");
                return false;
            }

            // Validate each stage
            for (size_t i = 0; i < pp_stages.size(); ++i)
            {
                if (!pp_stages[i].validate())
                {
                    LOG_ERROR("RankOrchestrator::Config: PP stage " << i << " validation failed");
                    return false;
                }
            }

            // Check layer continuity (no gaps)
            int expected_first = 0;
            for (size_t i = 0; i < pp_stages.size(); ++i)
            {
                if (pp_stages[i].first_layer != expected_first)
                {
                    LOG_ERROR("RankOrchestrator::Config: PP stage " << i
                                                                    << " first_layer=" << pp_stages[i].first_layer
                                                                    << " but expected " << expected_first << " (gap in layers)");
                    return false;
                }
                expected_first = pp_stages[i].last_layer;
            }

            // First stage should have embedding, last should have LM head
            if (!pp_stages.front().has_embedding)
            {
                LOG_WARN("RankOrchestrator::Config: First PP stage doesn't have embedding flag set");
            }
            if (!pp_stages.back().has_lm_head)
            {
                LOG_WARN("RankOrchestrator::Config: Last PP stage doesn't have lm_head flag set");
            }
        }

        return true;
    }

    std::vector<float> RankOrchestrator::Config::getNormalizedWeights() const
    {
        if (weights.empty() || weights.size() != devices.size())
        {
            // Equal distribution
            float equal_weight = 1.0f / static_cast<float>(devices.size());
            return std::vector<float>(devices.size(), equal_weight);
        }

        // Normalize to ensure sum is exactly 1.0
        float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (sum <= 0.0f)
        {
            float equal_weight = 1.0f / static_cast<float>(devices.size());
            return std::vector<float>(devices.size(), equal_weight);
        }

        std::vector<float> normalized(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            normalized[i] = weights[i] / sum;
        }
        return normalized;
    }

    // =========================================================================
    // Config::fromPlan — Canonical translation from RankExecutionPlan
    // =========================================================================

    RankOrchestrator::Config
    RankOrchestrator::Config::fromPlan(const RankExecutionPlan &plan)
    {
        Config config;

        // Runtime fields from pre-parsed RuntimeConfig
        config.max_seq_len = plan.runtime.max_seq_len;
        config.batch_size = plan.runtime.batch_size;
        config.activation_precision = plan.runtime.activation_precision;
        config.kv_cache_precision = plan.runtime.kv_cache_precision;
        config.tp_allreduce_precision_override =
            plan.runtime.tp_allreduce_precision_override;
        config.prefix_cache = plan.runtime.prefix_cache;
        config.mtp = plan.runtime.mtp;
        config.moe_expert_mode = plan.runtime.moe_expert_mode;
        config.moe_hot_expert_cache = plan.runtime.moe_hot_expert_cache;
        config.moe_rebalance = plan.runtime.moe_rebalance;

        if (plan.usesLocalPP())
        {
            // PP mode: build stage configs from plan boundaries
            config.mode = ParallelismMode::PP;

            const auto &pp_devices = plan.local_pp_devices;
            const auto &boundaries = plan.local_pp_layer_boundaries;
            const auto &stage_tp_info = plan.local_pp_stage_tp_info;

            for (size_t i = 0; i < pp_devices.size(); ++i)
            {
                PPStageConfig stage_cfg;
                stage_cfg.first_layer = boundaries[i];
                stage_cfg.last_layer = boundaries[i + 1]; // exclusive
                stage_cfg.has_embedding = (i == 0);
                stage_cfg.has_lm_head = (i == pp_devices.size() - 1);

                // Use per-stage TP info if available (TP-in-PP composition)
                if (i < stage_tp_info.size() && stage_tp_info[i].devices.size() > 1)
                {
                    stage_cfg.stage_devices = stage_tp_info[i].devices;
                    stage_cfg.tp_weights = stage_tp_info[i].tp_weights;
                    stage_cfg.tp_backend = stage_tp_info[i].tp_backend;
                }
                else
                {
                    stage_cfg.stage_devices = {pp_devices[i]};
                }

                // Cross-vendor detection for host-staged hidden state transfer
                // Compare primary devices of adjacent stages
                auto primaryDeviceType = [&](size_t idx) -> DeviceType
                {
                    if (idx < stage_tp_info.size() && !stage_tp_info[idx].devices.empty())
                        return stage_tp_info[idx].devices[0].device_type;
                    if (idx < pp_devices.size())
                        return pp_devices[idx].device_type;
                    return DeviceType::CPU;
                };
                if (i + 1 < pp_devices.size() &&
                    primaryDeviceType(i) != primaryDeviceType(i + 1))
                {
                    // Cross-vendor PP stage boundary detected
                }

                config.pp_stages.push_back(std::move(stage_cfg));
            }
        }
        else
        {
            // TP mode: copy devices, weights, backend
            config.devices = plan.local_tp_devices;
            if (!plan.local_tp_weights.empty())
            {
                config.weights = plan.local_tp_weights;
            }
            config.backend = plan.local_tp_backend;
        }

        return config;
    }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    std::unique_ptr<RankOrchestrator> RankOrchestrator::createForTest(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
    {
        // Use the private constructor
        return std::unique_ptr<RankOrchestrator>(
            new RankOrchestrator(
                std::move(model_ctx),
                std::move(device_runners),
                std::move(tp_ctx),
                config));
    }

    std::unique_ptr<RankOrchestrator> RankOrchestrator::createForTestWithPipelineStages(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> pp_stage_runners,
        const Config &config)
    {
        auto orchestrator = createForTest(
            std::move(model_ctx),
            {},
            nullptr,
            config);
        orchestrator->mode_ = ParallelismMode::PP;
        orchestrator->pp_stage_runners_ = std::move(pp_stage_runners);
        return orchestrator;
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    RankOrchestrator::RankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        const Config &config,
        std::unique_ptr<ILocalTPContext> tp_ctx)
        : model_ctx_(std::move(model_ctx)), config_(config)
    {
        if (!config_.validate())
        {
            throw std::invalid_argument("Invalid RankOrchestrator configuration");
        }

        // Initialize stage sharding map from model architecture
        stage_sharding_map_ = SchemaFactoryRegistry::getStageShardingConfig(model_ctx_->architecture());

        if (tp_ctx)
        {
            // Pre-existing TP context provided — force TP mode
            tp_ctx_ = std::move(tp_ctx);
            mode_ = ParallelismMode::TP;
            applyRuntimeSnapshotShardingOverrides();

            if (tp_ctx_->degree() < 2)
            {
                LOG_WARN("RankOrchestrator: TP degree is " << tp_ctx_->degree()
                                                           << ", multi-device orchestration may not be beneficial");
            }

            LOG_DEBUG("RankOrchestrator: Creating with pre-existing TP context, "
                      << tp_ctx_->degree() << " devices");

            initializeDeviceRunners();

            LOG_DEBUG("RankOrchestrator: Initialized with " << device_runners_.size() << " device runners");
        }
        else
        {
            // Auto-detect mode from config
            mode_ = config_.effectiveMode();

            LOG_DEBUG("RankOrchestrator: Creating with mode="
                      << (mode_ == ParallelismMode::TP ? "TP" : mode_ == ParallelismMode::PP ? "PP"
                                                                                             : "TP_PP")
                      << ", backend=" << static_cast<int>(config_.backend));

            if (mode_ == ParallelismMode::TP)
            {
                // Pure TP mode - create LOCAL TP context from config
                tp_ctx_ = createLocalTPContext(
                    config_.devices,
                    config_.getNormalizedWeights(),
                    config_.backend);

                if (!tp_ctx_)
                {
                    throw std::runtime_error("Failed to create LOCAL TP context");
                }
                applyRuntimeSnapshotShardingOverrides();

                if (tp_ctx_->degree() < 2)
                {
                    LOG_WARN("RankOrchestrator: TP degree is " << tp_ctx_->degree()
                                                               << ", multi-device orchestration may not be beneficial");
                }

                initializeDeviceRunners();

                LOG_DEBUG("RankOrchestrator: Initialized TP mode with " << device_runners_.size() << " device runners");
            }
            else
            {
                // PP or TP_PP mode - create PP context and stage runners
                initializePPDeviceRunners();
                initializePPContext();

                LOG_DEBUG("RankOrchestrator: Initialized PP mode with " << config_.pp_stages.size() << " stages");
            }
        }
    }

    // Private constructor for createForTest
    RankOrchestrator::RankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
        : model_ctx_(std::move(model_ctx)),
          tp_ctx_(std::move(tp_ctx)),
          mode_(ParallelismMode::TP), // Test factory currently only supports TP mode
          device_runners_(std::move(device_runners)),
          config_(config)
    {
        // Initialize stage sharding map from model architecture (if registered)
        const auto arch = model_ctx_->architecture();
        if (SchemaFactoryRegistry::isSupported(arch))
        {
            stage_sharding_map_ = SchemaFactoryRegistry::getStageShardingConfig(arch);
            applyRuntimeSnapshotShardingOverrides();
        }

        LOG_DEBUG("RankOrchestrator: Created via createForTest with "
                  << device_runners_.size() << " injected device runners");

        if (model_ctx_ && !device_runners_.empty())
        {
            int vocab = vocab_size();
            if (vocab > 0)
            {
                size_t max_tokens = static_cast<size_t>(config_.batch_size) *
                                    static_cast<size_t>(config_.max_seq_len);
                logits_gatherer_ = std::make_unique<LogitsGatherer>(vocab, max_tokens);

                if (device_runners_.size() > 1)
                {
                    DeviceId primary_dev = device_runners_[0]->primaryDeviceId();
                    if (primary_dev.is_gpu())
                    {
                        logits_gatherer_->pinForDevice(primary_dev);
                    }
                }
                applyLogitsGatherSkipFlags();
            }
        }

        wireLocalTPMoERuntimeHistogramSyncs();
    }

    RankOrchestrator::~RankOrchestrator() = default;

    // Move operations
    RankOrchestrator::RankOrchestrator(RankOrchestrator &&) noexcept = default;
    RankOrchestrator &RankOrchestrator::operator=(RankOrchestrator &&) noexcept = default;

    // =========================================================================
    // Private Methods
    // =========================================================================

    void RankOrchestrator::applyRuntimeSnapshotShardingOverrides()
    {
        if (!model_ctx_ || !tp_ctx_)
            return;

        if (model_ctx_->headCountKV() >= tp_ctx_->degree())
            return;

        /*
         * GQA models with fewer KV heads than TP participants replicate the K/V
         * stream.  Keep every semantic snapshot at that lifetime boundary in
         * lockstep so diagnostics do not combine projections one way and cache
         * state another way.
         */
        static constexpr const char *kv_keys[] = {
            "K_PROJECTION",
            "V_PROJECTION",
            "K_ROPE",
            "KV_APPEND_SOURCE_K",
            "KV_APPEND_SOURCE_V",
            "KV_CACHE_K",
            "KV_CACHE_V",
            "ATTENTION_EFFECTIVE_K",
            "ATTENTION_EFFECTIVE_V",
        };

        for (const char *key : kv_keys)
        {
            stage_sharding_map_[key] = SnapshotShardingMode::REPLICATED;
        }
    }

    void RankOrchestrator::initializeDeviceRunners()
    {
        if (!tp_ctx_)
        {
            throw std::runtime_error("Cannot initialize device runners: tp_ctx_ is null");
        }

        const auto &devices = tp_ctx_->devices();
        device_runners_.reserve(devices.size());

        LOG_DEBUG("RankOrchestrator: Initializing " << devices.size() << " device runners");

        // =====================================================================
        // BUILD WEIGHTMANAGERCONFIG FOR LOCAL TP WEIGHT SHARDING
        // =====================================================================
        // Configure WeightManager with TP config, model dimensions, and sharding
        // in a single configure() call. This replaces the previous multi-setter chain.
        // =====================================================================
        {
            auto weight_mgr = model_ctx_->weightManager();
            if (weight_mgr && tp_ctx_)
            {
                // Get model dimensions
                int n_heads = model_ctx_->headCount();
                int n_kv_heads = model_ctx_->headCountKV();
                int d_ff = model_ctx_->feedForwardLength();
                int vocab_size = model_ctx_->vocabSize();

                // feedForwardLength must be available from model metadata for correct TP sharding
                if (d_ff <= 0)
                {
                    throw std::runtime_error(
                        "RankOrchestrator: feedForwardLength() unavailable from model context. "
                        "This value is required for correct tensor parallel weight sharding. "
                        "Ensure the GGUF model contains the feed_forward_length metadata field.");
                }

                const bool dense_tp_enabled =
                    moeOverlayDenseTPEnabled(config_.moe_expert_parallel_plan);

                std::shared_ptr<TensorParallelConfig> tp_config;
                if (dense_tp_enabled)
                {
                    tp_config = std::make_shared<TensorParallelConfig>(
                        TensorParallelConfig::fromLocalTPContext(
                            *tp_ctx_, n_heads, n_kv_heads, d_ff, vocab_size));
                }

                // Build unified WeightManagerConfig
                WeightManagerConfig wm_config;
                wm_config.tp_config = dense_tp_enabled ? tp_config : nullptr;
                wm_config.sharding = SchemaFactoryRegistry::getWeightShardingConfig(model_ctx_->architecture());
                if (routedOverlayUsesApportionedExperts(config_.moe_expert_parallel_plan))
                {
                    forceRoutedMoEExpertParentsReplicated(wm_config.sharding);
                    LOG_DEBUG("RankOrchestrator: forcing routed MoE expert parent weights replicated for graph-native overlay");
                }

                // Model head dimensions for FusedQKV sub-block slicing
                const int embed_len = model_ctx_->embeddingLength();
                const int head_dim = (n_heads > 0) ? (embed_len / n_heads) : 0;
                if (n_heads > 0 && head_dim > 0)
                {
                    wm_config.dimensions.n_heads = n_heads;
                    wm_config.dimensions.n_kv_heads = n_kv_heads;
                    wm_config.dimensions.head_dim = head_dim;
                }

                // GDN dimensions from GGUF metadata for asymmetric FusedQKV
                auto *concrete_ctx = dynamic_cast<ModelContext *>(model_ctx_.get());
                if (concrete_ctx)
                {
                    const std::string arch_prefix = model_ctx_->architecture();
                    ModelLoader &loader = concrete_ctx->concreteLoader();
                    const int gdn_group_count = loader.getInt(arch_prefix + ".ssm.group_count", 0);
                    const int gdn_time_step_rank = loader.getInt(arch_prefix + ".ssm.time_step_rank", 0);
                    const int gdn_state_size = loader.getInt(arch_prefix + ".ssm.state_size", 0);

                    if (gdn_group_count > 0 && gdn_time_step_rank > 0 && gdn_state_size > 0)
                    {
                        wm_config.dimensions.gdn_n_k_heads = gdn_group_count;
                        wm_config.dimensions.gdn_n_v_heads = gdn_time_step_rank;
                        wm_config.dimensions.gdn_d_state = gdn_state_size;
                        LOG_DEBUG("RankOrchestrator: GDN dimensions for FusedQKV slicing"
                                  << " (group_count=" << gdn_group_count
                                  << " time_step_rank=" << gdn_time_step_rank
                                  << " state_size=" << gdn_state_size << ")");
                    }
                }

                // Apply all configuration in a single call
                weight_mgr->configure(wm_config);

                LOG_DEBUG("RankOrchestrator: Configured WeightManager for LOCAL TP ("
                          << tp_ctx_->degree() << " devices, "
                          << "heads=" << n_heads << ", kv_heads=" << n_kv_heads
                          << ", d_ff=" << d_ff << ", vocab=" << vocab_size
                          << ", dense_tp=" << (dense_tp_enabled ? "true" : "false") << ")");

                // Print per-device assignments for debugging TP parity issues
                for (int dev_idx = 0; dense_tp_enabled && dev_idx < tp_ctx_->degree(); ++dev_idx)
                {
                    const auto &addr = tp_ctx_->devices()[dev_idx];
                    DeviceId dev_id = addr.toLocalDeviceId();
                    try
                    {
                        const auto &assignment = tp_config->forDevice(dev_id);
                        LOG_DEBUG("RankOrchestrator: Device " << dev_idx << " (" << dev_id.to_string() << ") assignment:"
                                                              << " head_start=" << assignment.head_start
                                                              << " head_count=" << assignment.head_count
                                                              << " kv_head_start=" << assignment.kv_head_start
                                                              << " kv_head_count=" << assignment.kv_head_count
                                                              << " d_ff_start=" << assignment.d_ff_start
                                                              << " d_ff_count=" << assignment.d_ff_count);
                    }
                    catch (const std::out_of_range &e)
                    {
                        LOG_WARN("RankOrchestrator: Device " << dev_idx << " (" << dev_id.to_string() << ") NOT in TensorParallelConfig!");
                    }
                }

                // GQA-aware stage sharding override: when n_kv_heads < tp_degree,
                // K/V are replicated (not column-parallel). Override the snapshot
                // sharding map so parity tests compare K/V outputs correctly.
                if (dense_tp_enabled && n_kv_heads < tp_ctx_->degree())
                {
                    stage_sharding_map_["K_PROJECTION"] = SnapshotShardingMode::REPLICATED;
                    stage_sharding_map_["V_PROJECTION"] = SnapshotShardingMode::REPLICATED;
                    stage_sharding_map_["K_ROPE"] = SnapshotShardingMode::REPLICATED;
                    LOG_DEBUG("RankOrchestrator: GQA override: K/V stages set to REPLICATED "
                              "(n_kv_heads="
                              << n_kv_heads << " < tp_degree=" << tp_ctx_->degree() << ")");
                }
            }
        }

        // =====================================================================
        // PRE-RESERVE COLLECTIVE TEMP BUFFER
        // =====================================================================
        // Pre-allocate temp buffer for allreduce operations based on model dimensions
        // and activation precision. This avoids allocation in the hot path.
        // Buffer uses grow-only semantics (never shrinks during inference).
        // =====================================================================
        {
            size_t hidden_size = model_ctx_->embeddingLength();
            size_t max_elements = config_.max_seq_len * hidden_size;

            // Calculate buffer bytes based on activation precision (handles block quantization alignment)
            size_t buffer_bytes = activationPrecisionBufferBytes(max_elements, config_.activation_precision);

            // Add 10% margin for safety
            size_t buffer_with_margin = static_cast<size_t>(buffer_bytes * 1.1);

            const bool reserved = tp_ctx_->reserveTempBufferBytes(buffer_with_margin);
            logVramBomLine(
                "collective_temp_reservation",
                "source=RankOrchestrator backend=local_tp max_seq_len=" + std::to_string(config_.max_seq_len) +
                    " hidden_size=" + std::to_string(hidden_size) +
                    " precision=" + activationPrecisionToString(config_.activation_precision) +
                    " status=" + (reserved ? "pass" : "fail") +
                    " " + vramBomBytes(buffer_with_margin));
            if (reserved)
            {
                LOG_DEBUG("RankOrchestrator: Reserved collective temp buffer: "
                          << buffer_with_margin << " bytes ("
                          << "max_seq_len=" << config_.max_seq_len
                          << ", hidden_size=" << hidden_size
                          << ", precision=" << activationPrecisionToString(config_.activation_precision) << ")");
            }
            else
            {
                LOG_WARN("RankOrchestrator: Failed to reserve collective temp buffer");
            }
        }

        // =====================================================================
        // PRE-LOAD WEIGHTS FOR ALL DEVICES
        // =====================================================================
        // This is critical for multi-device operation:
        // - Creates device-specific clones of shared tensors (embedding, norms)
        // - Uploads each clone to its target device BEFORE parallel execution
        // - Avoids race condition where multiple devices try to upload same tensor
        //
        // The WeightManager now handles all device-aware weight management centrally.
        // =====================================================================
        bool overlay_weights_prepared_by_parent = false;
        {
            // Collect device IDs for preloading
            std::vector<DeviceId> device_ids;
            device_ids.reserve(devices.size());
            for (const auto &device_addr : devices)
            {
                device_ids.push_back(device_addr.toLocalDeviceId());
            }

            // Finalize weights for all devices: clone, upload, and prepare GEMM
            // weights. Host data release is deferred until after device runners are
            // created because graph construction resolves prepared weight bindings.
            auto weight_mgr = model_ctx_->weightManager();
            if (weight_mgr)
            {
                if (auto concrete_weight_mgr = std::dynamic_pointer_cast<WeightManager>(weight_mgr))
                {
                    if (config_.prepared_weight_store)
                    {
                        concrete_weight_mgr->setPreparedWeightStore(config_.prepared_weight_store);
                    }
                    else
                    {
                        config_.prepared_weight_store = std::make_shared<PreparedWeightStore>(
                            ModelContextId{reinterpret_cast<uint64_t>(model_ctx_.get())});
                        concrete_weight_mgr->setPreparedWeightStore(config_.prepared_weight_store);
                    }
                }
                if (config_.nested_pp_stage_config.has_value())
                {
                    LOG_DEBUG("RankOrchestrator: Preloading weights for "
                              << device_ids.size() << " nested TP-in-PP devices"
                              << " (binding-driven preparation deferred to runner materialization)");
                    if (!weight_mgr->preloadForDevices(device_ids))
                    {
                        LOG_WARN("RankOrchestrator: Weight preload failed; runner materialization may fail");
                    }
                }
                else
                {
                    const bool include_expert_jobs =
                        !(config_.moe_expert_parallel_plan &&
                          config_.moe_expert_parallel_plan->isTieredOverlay());
                    LOG_DEBUG("RankOrchestrator: Finalizing weights for "
                              << device_ids.size() << " devices"
                              << " (release_host_data=false, deferred until after graph build"
                              << ", include_expert_jobs=" << include_expert_jobs << ")");
                    if (!weight_mgr->finalizeForDevices(
                            device_ids,
                            /*release_host_data=*/false,
                            include_expert_jobs))
                    {
                        LOG_WARN("RankOrchestrator: Weight finalization failed; prepared kernels may be unavailable");
                    }

                    if (config_.moe_expert_parallel_plan &&
                        config_.moe_expert_parallel_plan->isTieredOverlay())
                    {
                        if (auto concrete_weight_mgr = std::dynamic_pointer_cast<WeightManager>(weight_mgr))
                        {
                            const int overlay_rank = config_.moe_expert_overlay_mpi_ctx
                                                         ? config_.moe_expert_overlay_mpi_ctx->rank()
                                                         : 0;
                            auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(
                                config_.moe_expert_parallel_plan,
                                MoEExpertOverlayRuntimeResolverOptions{
                                    .current_world_rank = overlay_rank,
                                });
                            LOG_DEBUG("RankOrchestrator: Preparing LocalTP MoE overlay expert weights once for "
                                      << device_ids.size() << " device(s)");
                            if (!concrete_weight_mgr->prepareMoEExpertOverlayWeights(*runtime_plan))
                            {
                                throw std::runtime_error(
                                    "RankOrchestrator: failed to prepare LocalTP MoE overlay expert weights");
                            }
                            overlay_weights_prepared_by_parent = true;
                        }
                    }
                }
            }
        }

        // =====================================================================
        // Create device runners in parallel
        // =====================================================================
        // After preloadForDevices(), each runner creation is independent:
        //   - Different device_id, different device_idx
        //   - WeightManager cache hits are mutex-protected
        //   - Graph construction reads immutable model config
        // Parallelizing this cuts ~50% off the MDO init time for 2+ devices.
        // =====================================================================

        struct RunnerResult
        {
            std::unique_ptr<IInferenceRunner> runner;
            int device_idx;
            std::string error;
        };

        const int num_devices = static_cast<int>(devices.size());
        std::vector<std::future<RunnerResult>> futures;
        futures.reserve(num_devices);

        for (int device_idx = 0; device_idx < num_devices; ++device_idx)
        {
            const auto &device_addr = devices[device_idx];
            DeviceId device_id = device_addr.toLocalDeviceId();

            futures.push_back(std::async(std::launch::async,
                                         [this, device_idx, device_id, overlay_weights_prepared_by_parent]() -> RunnerResult
                                         {
                                             RunnerResult result;
                                             result.device_idx = device_idx;
                                             try
                                             {
                                                 LOG_DEBUG("RankOrchestrator: Creating runner for device " << device_idx
                                                                                                           << " (" << device_id.toString() << ")");

                                                 // Build InferenceRunnerConfig for LOCAL TP
                                                 InferenceRunnerConfig runner_config;
                                                 runner_config.max_seq_len = static_cast<int>(config_.max_seq_len);
                                                 runner_config.batch_size = config_.batch_size;
                                                 runner_config.activation_precision = config_.activation_precision;
                                                 runner_config.kv_cache_scale_k = config_.kv_cache_scale_k;
                                                 runner_config.kv_cache_scale_v = config_.kv_cache_scale_v;
                                                 runner_config.kv_cache_precision = config_.kv_cache_precision;
                                                 runner_config.tp_allreduce_precision_override =
                                                     config_.tp_allreduce_precision_override;
                                                 runner_config.prefix_cache = config_.prefix_cache;
                                                 runner_config.mtp = config_.mtp;
                                                 runner_config.moe_expert_mode = config_.moe_expert_mode;
                                                 runner_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
                                                 runner_config.moe_rebalance = config_.moe_rebalance;
                                                 runner_config.use_mapped_memory = config_.use_mapped_memory;
                                                 runner_config.prepared_weight_store = config_.prepared_weight_store;
                                                 runner_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
                                                 runner_config.moe_expert_overlay_weights_prepared_by_parent =
                                                     overlay_weights_prepared_by_parent;
                                                 runner_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
                                                 runner_config.cancellation_requested = [this]()
                                                 {
                                                     return tp_ctx_ && tp_ctx_->isAbortRequested();
                                                 };
                                                 runner_config.stage_failure_callback = [this, device_idx](const std::string &stage_name, const std::string &reason)
                                                 {
                                                     LOG_WARN("RankOrchestrator: device " << device_idx
                                                                                          << " stage failure triggers TP abort"
                                                                                          << " stage='" << stage_name << "' reason='" << reason << "'");
                                                     if (tp_ctx_)
                                                         tp_ctx_->requestAbort();
                                                 };

                                                 // Set TP parameters (LOCAL TP context here)
                                                 runner_config.tp_ctx = tp_ctx_.get();
                                                 runner_config.tp_device_index = device_idx;

                                                 // Pass PP stage config for nested TP-in-PP
                                                 if (config_.nested_pp_stage_config.has_value())
                                                 {
                                                     runner_config.pp_stage_config = config_.nested_pp_stage_config;
                                                 }

                                                 // Create the inference runner
                                                 result.runner = createTestableInferenceRunner(model_ctx_, device_id, runner_config);
                                                 if (!result.runner)
                                                 {
                                                     result.error = "Failed to create inference runner for device " +
                                                                    std::to_string(device_idx);
                                                 }
                                             }
                                             catch (const std::exception &e)
                                             {
                                                 result.error = std::string("Exception creating runner for device ") +
                                                                std::to_string(device_idx) + ": " + e.what();
                                             }
                                             return result;
                                         }));
        }

        // Collect results in device order
        for (auto &fut : futures)
        {
            auto result = fut.get();
            if (!result.error.empty())
            {
                throw std::runtime_error(result.error);
            }

            // Cast to DeviceGraphOrchestrator for setPPStageConfig (construction-time only)
            if (config_.nested_pp_stage_config.has_value())
            {
                auto *device_orchestrator = dynamic_cast<DeviceGraphOrchestrator *>(result.runner.get());
                if (device_orchestrator)
                {
                    device_orchestrator->setHostResidentReleaseEnabled(false);
                    device_orchestrator->setPPStageConfig(config_.nested_pp_stage_config.value());
                    LOG_DEBUG("RankOrchestrator: Set PP stage config on device " << result.device_idx
                                                                                 << " (layers " << config_.nested_pp_stage_config->first_layer
                                                                                 << "-" << config_.nested_pp_stage_config->last_layer
                                                                                 << " has_lm_head=" << config_.nested_pp_stage_config->has_lm_head << ")");
                }
                else if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(result.runner.get()))
                {
                    dgo->setHostResidentReleaseEnabled(false);
                }
            }
            else if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(result.runner.get()))
            {
                dgo->setHostResidentReleaseEnabled(false);
            }

            // Store as IInferenceRunner (decoupled from concrete DGO type)
            device_runners_.push_back(std::move(result.runner));

            LOG_DEBUG("RankOrchestrator: Successfully created runner for device " << result.device_idx);
        }

        // =====================================================================
        // SYNCHRONOUS HOST WEIGHT RELEASE (Phase 9 final)
        // =====================================================================
        // All expert GEMM preparation is resolved before host release: GPU paths
        // consume the unified ExpertGemmRegistry and CPU paths prepare inline.
        // There is no lazy graph building path.
        // Dynamic MoE rebalancing uses serialized transfer blobs, not raw host data.
        //
        // Mark graph_materialization_complete and release host data immediately.
        // =====================================================================
        {
            const bool is_nested_tp_in_pp = config_.nested_pp_stage_config.has_value();
            if (!is_nested_tp_in_pp)
            {
                auto weight_mgr = model_ctx_->weightManager();
                if (weight_mgr)
                {
                    weight_mgr->markGraphMaterializationComplete();

                    size_t released = weight_mgr->releaseAllHostWeightData();
                    if (released > 0)
                    {
                        LOG_DEBUG("RankOrchestrator: Host weight release after graph materialization: "
                                  << released << " tensors released");
#ifdef __linux__
                        ::malloc_trim(0);
#endif
                    }
                }
            }
            else
            {
                LOG_DEBUG("RankOrchestrator: Retaining host weight data "
                          "(nested TP-in-PP; outer caller will release)");
            }
        }

        // Create logits gatherer for combined logits buffer management
        if (model_ctx_ && device_runners_.size() > 0)
        {
            int vocab = vocab_size();
            if (vocab > 0)
            {
                size_t max_tokens = static_cast<size_t>(config_.batch_size) *
                                    static_cast<size_t>(config_.max_seq_len);
                logits_gatherer_ = std::make_unique<LogitsGatherer>(vocab, max_tokens);

                // Pin the logits buffer for faster D2H DMA
                if (device_runners_.size() > 1)
                {
                    DeviceId primary_dev = device_runners_[0]->primaryDeviceId();
                    if (primary_dev.is_gpu())
                    {
                        logits_gatherer_->pinForDevice(primary_dev);
                    }
                }
                applyLogitsGatherSkipFlags();
            }
        }

        // =====================================================================
        // REGISTER COMPUTE STREAMS WITH COLLECTIVE BACKEND
        // =====================================================================
        // Enables event-based pre-synchronization in RCCL/NCCL coordinators instead
        // of hipDeviceSynchronize/cudaDeviceSynchronize. Each device's compute stream
        // is passed so the coordinator can do hipEventRecord(compute_stream) +
        // hipStreamWaitEvent(rccl_stream) before collectives — zero host stall.
        // =====================================================================
        if (tp_ctx_ && tp_ctx_->degree() > 1)
        {
            const auto &devices = tp_ctx_->devices();

            auto &pool = GPUDeviceContextPool::instance();
            std::vector<void *> compute_streams;
            compute_streams.reserve(devices.size());

            for (const auto &dev : devices)
            {
                if (dev.device_type == DeviceType::ROCm || dev.device_type == DeviceType::CUDA)
                {
                    compute_streams.push_back(
                        pool.getContext(DeviceId(dev.device_type, dev.device_ordinal)).defaultStream());
                }
                else
                {
                    LOG_WARN("RankOrchestrator: Skipping compute stream registration for non-GPU device "
                             << dev.toString());
                    compute_streams.clear();
                    break;
                }
            }

            if (!compute_streams.empty())
            {
                tp_ctx_->setComputeStreams(compute_streams);
                LOG_DEBUG("RankOrchestrator: Registered " << compute_streams.size()
                                                          << " compute streams for event-based collective sync");
            }
        }

        wireLocalTPMoERuntimeHistogramSyncs();
    }

    // =========================================================================
    // PP Mode Initialization
    // =========================================================================

    void RankOrchestrator::initializePPDeviceRunners()
    {
        if (config_.pp_stages.empty())
        {
            throw std::runtime_error("Cannot initialize PP device runners: no PP stages configured");
        }

        LOG_DEBUG("RankOrchestrator: Initializing " << config_.pp_stages.size() << " PP stage runners");

        // Cast to concrete ModelContext — needed by createPPStageRunner for concreteLoader() access
        auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(model_ctx_);
        if (!concrete_model_ctx)
        {
            throw std::runtime_error("RankOrchestrator: model_ctx_ must be a concrete ModelContext for PP mode");
        }

        const size_t num_stages = config_.pp_stages.size();

        // =========================================================================
        // Cross-Vendor PP Detection
        // =========================================================================
        // Check if any PP transfer is cross-vendor (ROCm→CUDA or CUDA→ROCm).
        // If so, the source stage's hidden state tensor needs host-staged allocation
        // to enable cross-vendor transfers via HOST backend.
        // =========================================================================
        auto isCrossVendorTransfer = [](const PPStageConfig &src, const PPStageConfig &dst) -> bool
        {
            if (src.stage_devices.empty() || dst.stage_devices.empty())
            {
                return false;
            }
            DeviceId src_dev = src.stage_devices[0].toLocalDeviceId();
            DeviceId dst_dev = dst.stage_devices[0].toLocalDeviceId();

            bool src_cuda = src_dev.is_cuda();
            bool src_rocm = src_dev.is_rocm();
            bool dst_cuda = dst_dev.is_cuda();
            bool dst_rocm = dst_dev.is_rocm();

            return (src_cuda && dst_rocm) || (src_rocm && dst_cuda);
        };

        // Check for cross-vendor transfers for logging purposes
        for (size_t i = 0; i + 1 < num_stages; ++i)
        {
            if (isCrossVendorTransfer(config_.pp_stages[i], config_.pp_stages[i + 1]))
            {
                LOG_DEBUG("RankOrchestrator: PP stage " << i
                                                        << " outputs to cross-vendor stage " << (i + 1)
                                                        << " - will use host-staged transfer");
            }
        }

        pp_stage_runners_.reserve(num_stages);

        for (size_t stage_idx = 0; stage_idx < num_stages; ++stage_idx)
        {
            const auto &stage_config = config_.pp_stages[stage_idx];

            LOG_DEBUG("RankOrchestrator: Creating PP stage " << stage_idx
                                                             << " [layers " << stage_config.first_layer
                                                             << "-" << stage_config.last_layer << ")"
                                                             << " has_embedding=" << stage_config.has_embedding
                                                             << " has_lm_head=" << stage_config.has_lm_head);

            // Validate stage has at least one device
            if (stage_config.stage_devices.empty())
            {
                throw std::runtime_error("PP stage " + std::to_string(stage_idx) + " has no devices configured");
            }

            // Get primary device for this stage
            DeviceId primary_device = stage_config.stage_devices[0].toLocalDeviceId();

            // =====================================================================
            // Build InferenceRunnerConfig for this stage
            // =====================================================================
            InferenceRunnerConfig runner_config;
            runner_config.max_seq_len = static_cast<int>(config_.max_seq_len);
            runner_config.batch_size = config_.batch_size;
            runner_config.activation_precision = config_.activation_precision;
            runner_config.kv_cache_scale_k = config_.kv_cache_scale_k;
            runner_config.kv_cache_scale_v = config_.kv_cache_scale_v;
            runner_config.kv_cache_precision = config_.kv_cache_precision;
            runner_config.tp_allreduce_precision_override =
                config_.tp_allreduce_precision_override;
            runner_config.prefix_cache = config_.prefix_cache;
            runner_config.mtp = config_.mtp;
            runner_config.moe_expert_mode = config_.moe_expert_mode;
            runner_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
            runner_config.moe_rebalance = config_.moe_rebalance;
            runner_config.use_mapped_memory = config_.use_mapped_memory;
            runner_config.prepared_weight_store = config_.prepared_weight_store;
            runner_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
            runner_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
            // =====================================================================
            // Build FactoryPPStageConfig for the createPPStageRunner factory
            // =====================================================================
            FactoryPPStageConfig factory_pp_config;
            factory_pp_config.first_layer = stage_config.first_layer;
            factory_pp_config.last_layer = stage_config.last_layer;
            factory_pp_config.has_embedding = stage_config.has_embedding;
            factory_pp_config.has_lm_head = stage_config.has_lm_head;

            // =====================================================================
            // Handle single-device vs TP-domain stages
            // =====================================================================
            if (stage_config.isTPDomain())
            {
                // =====================================================================
                // TP Domain Stage: Create nested RankOrchestrator in TP mode
                // =====================================================================
                LOG_DEBUG("RankOrchestrator: PP stage " << stage_idx
                                                        << " is a TP domain with " << stage_config.stage_devices.size() << " devices");

                // Build TP configuration for the nested orchestrator
                Config nested_config;
                nested_config.mode = ParallelismMode::TP;
                nested_config.devices = stage_config.stage_devices;
                nested_config.weights = stage_config.tp_weights;
                nested_config.backend = stage_config.tp_backend;
                nested_config.max_seq_len = config_.max_seq_len;
                nested_config.batch_size = config_.batch_size;
                nested_config.activation_precision = config_.activation_precision;
                nested_config.kv_cache_scale_k = config_.kv_cache_scale_k;
                nested_config.kv_cache_scale_v = config_.kv_cache_scale_v;
                nested_config.kv_cache_precision = config_.kv_cache_precision;
                nested_config.tp_allreduce_precision_override =
                    config_.tp_allreduce_precision_override;
                nested_config.prefix_cache = config_.prefix_cache;
                nested_config.mtp = config_.mtp;
                nested_config.moe_expert_mode = config_.moe_expert_mode;
                nested_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
                nested_config.moe_rebalance = config_.moe_rebalance;
                nested_config.use_mapped_memory = config_.use_mapped_memory;
                nested_config.prepared_weight_store = config_.prepared_weight_store;
                nested_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
                nested_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
                // CRITICAL: Pass PP stage config to nested TP MDO so its DeviceGraphOrchestrators
                // build partial graphs instead of full graphs. Without this, the TP devices would
                // build LM_HEAD stages even though this PP stage doesn't own LM_HEAD.
                nested_config.nested_pp_stage_config = factory_pp_config;

                // Create the nested RankOrchestrator
                // Note: model_ctx_ is the shared ModelContext — the nested MDO's
                // createPPStageRunner uses the shared WeightManager from ModelContext
                auto nested_mdo = std::make_unique<RankOrchestrator>(concrete_model_ctx, nested_config);

                if (!nested_mdo)
                {
                    throw std::runtime_error("Failed to create nested RankOrchestrator for PP stage " +
                                             std::to_string(stage_idx));
                }

                pp_stage_runners_.push_back(std::move(nested_mdo));

                LOG_DEBUG("RankOrchestrator: Created TP domain PP stage " << stage_idx
                                                                          << " with " << stage_config.stage_devices.size() << " devices"
                                                                          << " (layers " << stage_config.first_layer << "-" << stage_config.last_layer << ")");
            }
            else
            {
                // =====================================================================
                // Single Device Stage: Use existing factory path
                // =====================================================================
                auto runner = createPPStageRunner(concrete_model_ctx, primary_device, factory_pp_config, runner_config);

                if (!runner)
                {
                    throw std::runtime_error("Failed to create PP stage runner for stage " +
                                             std::to_string(stage_idx) + " on device " +
                                             primary_device.to_string());
                }

                if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                    dgo->setHostResidentReleaseEnabled(false);

                pp_stage_runners_.push_back(std::move(runner));

                LOG_DEBUG("RankOrchestrator: Created PP stage " << stage_idx
                                                                << " runner on device " << primary_device.to_string()
                                                                << " (layers " << stage_config.first_layer << "-" << stage_config.last_layer << ")");
            }
        }

        LOG_DEBUG("RankOrchestrator: Successfully initialized " << pp_stage_runners_.size() << " PP stage runners");

        // =====================================================================
        // Release host weight copies now that all PP stages are prepared.
        // With single-WM architecture, host copies are retained across stages
        // and only released after ALL stage preparation is complete.
        //
        // CPU safety: If any PP stage runs on CPU, retain host weight data —
        // CPU stages read weights directly from host memory.
        // =====================================================================
        bool has_cpu_stage = false;
        for (const auto &stage_config : config_.pp_stages)
        {
            if (!stage_config.stage_devices.empty() &&
                stage_config.stage_devices[0].toLocalDeviceId().is_cpu())
            {
                has_cpu_stage = true;
                break;
            }
        }

        if (has_cpu_stage)
        {
            LOG_DEBUG("RankOrchestrator: Retaining host weight data (CPU PP stages present)");
        }
        else if (auto *concrete_wm = dynamic_cast<WeightManager *>(model_ctx_->weightManager().get()))
        {
            size_t released = concrete_wm->releaseAllHostWeightData();
            LOG_DEBUG("RankOrchestrator: Released " << released << " host weight copies after PP preparation");
        }

        // Build PPActivationContract describing inter-stage data transfers
        if (config_.pp_stages.size() > 1)
        {
            pp_activation_contract_ = std::make_unique<PPActivationContract>();

            const size_t embedding_dim = model_ctx_->embeddingLength();
            const size_t max_seq_len = model_ctx_->contextLength();

            for (size_t i = 0; i + 1 < config_.pp_stages.size(); ++i)
            {
                const auto &src = config_.pp_stages[i];
                const auto &dst = config_.pp_stages[i + 1];

                PPStageTransferContract xfer;
                xfer.source_stage = static_cast<int>(i);
                xfer.target_stage = static_cast<int>(i + 1);
                xfer.source_device = src.stage_devices[0].toLocalDeviceId();
                xfer.target_device = dst.stage_devices[0].toLocalDeviceId();
                xfer.embedding_dim = embedding_dim;
                xfer.dtype = ActivationDType::FP32;
                xfer.max_seq_len = max_seq_len;

                pp_activation_contract_->addTransfer(std::move(xfer));
            }

            auto validation_err = pp_activation_contract_->validate();
            if (!validation_err.empty())
            {
                LOG_ERROR("RankOrchestrator: PPActivationContract validation failed: " << validation_err);
                pp_activation_contract_.reset();
            }
            else
            {
                LOG_DEBUG("RankOrchestrator: PPActivationContract built with "
                          << pp_activation_contract_->numTransfers() << " transfers"
                          << " (embedding=" << embedding_dim << ", max_seq=" << max_seq_len << ")");
            }
        }

        // Note: PP context initialization is done by the caller (constructor)
        // after this method returns, to avoid double-initialization
    }

    void RankOrchestrator::initializePPContext()
    {
        if (pp_stage_runners_.empty())
        {
            // PP stage runners not initialized - this is expected for Phase 1
            LOG_DEBUG("RankOrchestrator::initializePPContext: No PP stage runners yet");
            return;
        }

        // Build LocalPPConfig for simple single-device stages
        // TODO: Phase 5 - use HierarchicalPPConfig for TP domain stages
        LocalPPConfig pp_config;
        pp_config.stage_devices.reserve(config_.pp_stages.size());
        pp_config.layer_boundaries.reserve(config_.pp_stages.size() + 1);

        // Add first boundary (start of first stage)
        pp_config.layer_boundaries.push_back(config_.pp_stages[0].first_layer);

        for (size_t i = 0; i < config_.pp_stages.size(); ++i)
        {
            const auto &stage_config = config_.pp_stages[i];
            // Use first device in stage_devices for single-device stages
            if (stage_config.stage_devices.empty())
            {
                throw std::runtime_error("PP stage " + std::to_string(i) + " has no devices");
            }
            pp_config.stage_devices.push_back(stage_config.stage_devices[0]);
            // Each boundary is the exclusive end = last_layer
            pp_config.layer_boundaries.push_back(stage_config.last_layer);
        }

        // Validate config
        if (!pp_config.isValid())
        {
            throw std::runtime_error("Generated LocalPPConfig is invalid");
        }

        LOG_DEBUG("RankOrchestrator::initializePPContext: Creating LocalPPContext with "
                  << pp_config.numStages() << " stages");

        // Create the LocalPPContext using the factory function
        pp_ctx_ = createLocalPPContext(pp_config);

        if (!pp_ctx_)
        {
            throw std::runtime_error("Failed to create LocalPPContext");
        }

        LOG_DEBUG("RankOrchestrator: Initialized LocalPPContext with " << pp_config.numStages() << " stages");
    }

    void RankOrchestrator::aggregateStats() const
    {
        if (!stats_dirty_ || device_runners_.empty())
        {
            return;
        }

        // Reset or create aggregated stats
        if (!aggregated_stats_)
        {
            aggregated_stats_ = std::make_unique<GraphExecutorStats>();
        }
        aggregated_stats_->reset();

        // Aggregate from all device runners
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                const auto *stats = runner->executorStats();
                if (stats)
                {
                    aggregated_stats_->total_stages_executed += stats->total_stages_executed;
                    aggregated_stats_->total_flops += stats->total_flops;
                    aggregated_stats_->total_time_ms += stats->total_time_ms;
                    aggregated_stats_->total_execute_ms += stats->total_execute_ms;
                    aggregated_stats_->total_collective_ms += stats->total_collective_ms;
                    aggregated_stats_->total_collective_calls += stats->total_collective_calls;

                    // Merge stage times
                    for (const auto &[stage_name, time_ms] : stats->stage_times_ms)
                    {
                        aggregated_stats_->stage_times_ms[stage_name] += time_ms;
                    }

                    // Merge stage type execution times and counts
                    for (const auto &[type_name, time_ms] : stats->stage_type_execute_ms)
                    {
                        aggregated_stats_->stage_type_execute_ms[type_name] += time_ms;
                    }
                    for (const auto &[type_name, count] : stats->stage_type_counts)
                    {
                        aggregated_stats_->stage_type_counts[type_name] += count;
                    }

                    // Merge overhead breakdown
                    aggregated_stats_->overhead += stats->overhead;

                    // Merge phase-split stats (prefill / decode)
                    auto mergePhase = [](PhaseStats &dst, const PhaseStats &src)
                    {
                        dst.total_execute_ms += src.total_execute_ms;
                        dst.total_stages_executed += src.total_stages_executed;
                        dst.total_collective_ms += src.total_collective_ms;
                        dst.total_collective_calls += src.total_collective_calls;
                        for (const auto &[type_name, time_ms] : src.stage_type_execute_ms)
                            dst.stage_type_execute_ms[type_name] += time_ms;
                        for (const auto &[type_name, cnt] : src.stage_type_counts)
                            dst.stage_type_counts[type_name] += cnt;
                    };
                    mergePhase(aggregated_stats_->prefill, stats->prefill);
                    mergePhase(aggregated_stats_->decode, stats->decode);
                }
            }
        }

        // Average the times (since devices run in parallel)
        if (!device_runners_.empty())
        {
            size_t count = device_runners_.size();
            double dcount = static_cast<double>(count);
            aggregated_stats_->total_time_ms /= dcount;
            aggregated_stats_->total_execute_ms /= dcount;
            aggregated_stats_->total_collective_ms /= dcount;
            aggregated_stats_->total_collective_calls /= count;
            for (auto &[stage_name, time_ms] : aggregated_stats_->stage_times_ms)
            {
                time_ms /= dcount;
            }
            for (auto &[type_name, time_ms] : aggregated_stats_->stage_type_execute_ms)
            {
                time_ms /= dcount;
            }
            for (auto &[type_name, cnt] : aggregated_stats_->stage_type_counts)
            {
                cnt /= count;
            }

            // Average overhead (each device incurs its own overhead in parallel)
            aggregated_stats_->overhead.input_cohere_ms /= dcount;
            aggregated_stats_->overhead.weight_cohere_ms /= dcount;
            aggregated_stats_->overhead.output_alloc_ms /= dcount;
            aggregated_stats_->overhead.mark_dirty_ms /= dcount;
            aggregated_stats_->overhead.dump_input_ms /= dcount;
            aggregated_stats_->overhead.dump_output_ms /= dcount;
            aggregated_stats_->overhead.verify_ms /= dcount;
            aggregated_stats_->overhead.callback_ms /= dcount;
            aggregated_stats_->overhead.get_dump_info_ms /= dcount;

            // Average phase stats (devices run in parallel)
            auto avgPhase = [dcount, count](PhaseStats &ps)
            {
                ps.total_execute_ms /= dcount;
                ps.total_stages_executed /= count;
                ps.total_collective_ms /= dcount;
                ps.total_collective_calls /= count;
                for (auto &[_, ms] : ps.stage_type_execute_ms)
                    ms /= dcount;
                for (auto &[_, cnt] : ps.stage_type_counts)
                    cnt /= count;
            };
            avgPhase(aggregated_stats_->prefill);
            avgPhase(aggregated_stats_->decode);
        }

        stats_dirty_ = false;
    }

    // =========================================================================
    // IInferenceRunner Interface Implementation
    // =========================================================================

    bool RankOrchestrator::forward(const int *tokens, int seq_len)
    {
        bool success = false;

        // Dispatch to appropriate implementation based on parallelism mode
        switch (mode_)
        {
        case ParallelismMode::TP:
            success = forwardTP(tokens, seq_len, /*force_prefill_phase=*/false);
            break;
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            // PP and TP_PP both use sequential stage execution
            // The difference is that TP_PP stages may be nested MDOs (TP domains)
            // but forwardPP() works through IInferenceRunner interface regardless
            success = forwardPP(tokens, seq_len, /*force_prefill_phase=*/false);
            break;
        default:
            LOG_ERROR("RankOrchestrator::forward: Unknown parallelism mode");
            return false;
        }

        return success;
    }

    bool RankOrchestrator::forwardPrefill(const int *tokens, int seq_len)
    {
        switch (mode_)
        {
        case ParallelismMode::TP:
            return forwardTP(tokens, seq_len, /*force_prefill_phase=*/true);
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            return forwardPP(tokens, seq_len, /*force_prefill_phase=*/true);
        default:
            LOG_ERROR("RankOrchestrator::forwardPrefill: Unknown parallelism mode");
            return false;
        }
    }

    bool RankOrchestrator::forwardWithDeviceTokenIds(
        const int *token_shadow,
        const void *token_ids_device,
        int seq_len)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardWithDeviceTokenIds(
                token_shadow,
                token_ids_device,
                seq_len);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardWithDeviceTokenIds(
                token_shadow,
                token_ids_device,
                seq_len);
        }
        if (device_runners_.size() < 2 ||
            !token_shadow ||
            seq_len <= 0 ||
            token_ids_device != rank_mtp_verifier_child_token_inputs_.data() ||
            rank_mtp_verifier_child_token_inputs_.size() !=
                device_runners_.size() ||
            rank_mtp_verifier_child_token_count_ != seq_len)
        {
            LOG_ERROR("RankOrchestrator::forwardWithDeviceTokenIds: invalid LocalTP verifier token bundle");
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i] ||
                !rank_mtp_verifier_child_token_inputs_[i])
            {
                LOG_ERROR("RankOrchestrator::forwardWithDeviceTokenIds: participant "
                          << i << " has no staged verifier token row");
                return false;
            }
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] device-token verifier forward failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        tp_worker_pool_->dispatch(
            [this,
             token_shadow,
             seq_len,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                if (debugEnv().tp_collective_contract_trace)
                {
                    LOG_DEBUG("[TP_WORKER_CONTRACT] event=device_token_forward_enter"
                              << " worker=" << i
                              << " device=" << device_id.toString()
                              << " seq_len=" << seq_len);
                }

                const bool ok =
                    device_runners_[i]->forwardWithDeviceTokenIds(
                        token_shadow,
                        rank_mtp_verifier_child_token_inputs_[i],
                        seq_len);

                if (debugEnv().tp_collective_contract_trace)
                {
                    LOG_DEBUG("[TP_WORKER_CONTRACT] event=device_token_forward_leave"
                              << " worker=" << i
                              << " device=" << device_id.toString()
                              << " success=" << (ok ? 1 : 0));
                }
                return ok;
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                worker_timeout = true;
                all_success = false;
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                continue;
            }
            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
                all_success = false;
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardWithDeviceTokenIds",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardWithDeviceTokenIds: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        current_position_ += seq_len;
        current_padded_seq_len_ = seq_len;
        if (current_sequence_lengths_.empty())
        {
            current_sequence_lengths_.assign(
                static_cast<size_t>(std::max(1, config_.batch_size)),
                current_position_ - seq_len);
        }
        for (int &length : current_sequence_lengths_)
            length += seq_len;
        stats_dirty_ = true;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_device_token_forwards",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"seq_len", std::to_string(seq_len)}});
        return true;
    }

    bool RankOrchestrator::supportsPrefillChunkSchedule(int seq_len) const
    {
        if (mode_ != ParallelismMode::TP || device_runners_.empty())
            return false;

        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->supportsPrefillChunkSchedule(seq_len))
                return false;
        }
        return true;
    }

    bool RankOrchestrator::forwardPrefillChunkSchedule(
        const int *tokens,
        int seq_len,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        switch (mode_)
        {
        case ParallelismMode::TP:
            return forwardTP(
                tokens,
                seq_len,
                /*force_prefill_phase=*/false,
                &policy,
                pad_token_id,
                allow_padded_execution);
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            LOG_ERROR("RankOrchestrator::forwardPrefillChunkSchedule: chunked prefill is not implemented for PP/TP_PP");
            return false;
        default:
            LOG_ERROR("RankOrchestrator::forwardPrefillChunkSchedule: Unknown parallelism mode");
            return false;
        }
    }

    // =========================================================================
    // TP Mode Forward Implementation (existing parallel execution)
    // =========================================================================
    int RankOrchestrator::effectiveTPWorkerJoinTimeoutMs() const
    {
        return collective_timeout_policy::effectiveWorkerJoinTimeoutMs(
            debugEnv().tp_collect_timeout_ms);
    }

    bool RankOrchestrator::forwardTP(
        const int *tokens,
        int seq_len,
        bool force_prefill_phase,
        const PrefillChunkSchedulerPolicy *chunk_schedule_policy,
        int chunk_schedule_pad_token_id,
        bool chunk_schedule_allow_padded_execution)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::forwardTP: No device runners available");
            return false;
        }

        // TP timing diagnostic — enabled via LLAMINAR_TP_TIMING=1 or LLAMINAR_PROFILING=1
        const bool tp_timing = debugEnv().tp_timing;
        const bool tp_profiling = debugEnv().execution.executor_profiling;
        const bool collect_timing = tp_timing || tp_profiling;
        auto tp_t0 = collect_timing ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};

        LOG_DEBUG("RankOrchestrator::forwardTP: seq_len=" << seq_len
                                                          << ", devices=" << device_runners_.size());

        // Decode latency breakdown timing
        const bool decode_breakdown = collect_timing && seq_len == 1;
        auto launch_t0 = decode_breakdown ? std::chrono::high_resolution_clock::now()
                                          : std::chrono::high_resolution_clock::time_point{};
        auto launch_t1 = launch_t0; // Set properly after dispatch in parallel mode

        // DIAGNOSTIC: Run forward passes SEQUENTIALLY to test if concurrency causes crash.
        // If sequential execution works but parallel crashes, it's a concurrent HIP issue.
        const bool serialize_devices = debugEnv().runtime_debug.serialize_tp_forward;

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;

        if (serialize_devices)
        {
            // ----- SERIAL MODE (diagnostic fallback) -----
            for (size_t i = 0; i < device_runners_.size(); ++i)
            {
                auto &runner = device_runners_[i];
                if (!runner)
                    continue;
                // Cast to IInferenceRunner* to disambiguate forward() overloads
                IInferenceRunner *runner_iface = runner.get();
                LOG_WARN("RankOrchestrator::forwardTP: SERIAL mode - device "
                         << i << " running synchronously");
                try
                {
                    const bool ok = chunk_schedule_policy
                                        ? runner_iface->forwardPrefillChunkSchedule(
                                              tokens,
                                              seq_len,
                                              *chunk_schedule_policy,
                                              chunk_schedule_pad_token_id,
                                              chunk_schedule_allow_padded_execution)
                                        : (force_prefill_phase
                                               ? runner_iface->forwardPrefill(tokens, seq_len)
                                               : runner_iface->forward(tokens, seq_len));
                    if (!ok)
                    {
                        LOG_ERROR("RankOrchestrator::forwardTP: Device "
                                  << i << " forward failed");
                        all_success = false;
                    }
                }
                catch (...)
                {
                    all_success = false;
                    if (!first_exception)
                    {
                        first_exception = std::current_exception();
                        first_exception_device = i;
                    }
                }
            }
        }
        else
        {
            // ----- PARALLEL MODE: persistent thread pool -----
            // Lazy-initialize on first TP forward call. Workers persist for the
            // lifetime of the orchestrator, eliminating per-step thread overhead.
            if (!tp_worker_pool_)
            {
                tp_worker_pool_ = std::make_unique<TPWorkerPool>(device_runners_.size());

                // Wire abort callback: when one worker fails (exception or false return),
                // abort the collective backend to unblock any workers stuck in NCCL/RCCL
                // collective calls. Without this, a failed worker exits the forward pass
                // while other workers block forever in ncclAllReduce waiting for all ranks.
                if (tp_ctx_)
                {
                    tp_worker_pool_->setFailureCallback([this]()
                                                        {
                        LOG_WARN("[TPWorkerPool] Worker failure detected — aborting collective backend");
                        tp_ctx_->requestAbort(); });
                }

                LOG_DEBUG("[TPWorkerPool] Created " << device_runners_.size()
                                                    << " persistent worker threads");
            }

            // Dispatch parallel forward passes to persistent worker threads.
            // dispatch() wakes all workers via condition_variable and returns
            // immediately — no thread creation, no pthread_create syscall.
            //
            // Capture profiler phases from the caller thread and propagate to
            // workers. All profilers use thread-local phase storage, so worker
            // threads must explicitly set their phase to match the caller.
            auto kernel_phase = KernelProfiler::getCurrentPhase();
            auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            auto kv_phase = KVCacheProfiler::getCurrentPhase();
            auto executor_phase = GraphExecutorStats::currentPhase();

            std::vector<DeviceGraphOrchestrator *> pre_execution_rendezvous_runners;
            std::shared_ptr<ForwardGraphExecutionRendezvous> pre_execution_rendezvous;
            if (!tp_first_forward_completed_ && device_runners_.size() > 1)
            {
                pre_execution_rendezvous_runners.reserve(device_runners_.size());
                for (auto &runner : device_runners_)
                {
                    auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get());
                    if (dgo)
                        pre_execution_rendezvous_runners.push_back(dgo);
                }

                if (pre_execution_rendezvous_runners.size() == device_runners_.size())
                {
                    pre_execution_rendezvous =
                        std::make_shared<ForwardGraphExecutionRendezvous>(
                            device_runners_.size(),
                            "rank_tp_first_forward");
                    for (auto *dgo : pre_execution_rendezvous_runners)
                        dgo->setForwardGraphExecutionRendezvous(pre_execution_rendezvous);
                    LOG_DEBUG("RankOrchestrator::forwardTP: armed first-forward pre-execution rendezvous for "
                              << device_runners_.size() << " TP child graph(s)");
                }
                else if (!pre_execution_rendezvous_runners.empty())
                {
                    LOG_WARN("RankOrchestrator::forwardTP: skipping first-forward pre-execution rendezvous; "
                             << pre_execution_rendezvous_runners.size() << "/"
                             << device_runners_.size()
                             << " child runners expose DeviceGraphOrchestrator");
                    pre_execution_rendezvous_runners.clear();
                }
            }

            struct ScopedForwardExecutionRendezvousClear
            {
                std::vector<DeviceGraphOrchestrator *> runners;
                ~ScopedForwardExecutionRendezvousClear()
                {
                    for (auto *runner : runners)
                    {
                        if (runner)
                            runner->setForwardGraphExecutionRendezvous(nullptr);
                    }
                }
            } rendezvous_clear_guard{pre_execution_rendezvous_runners};

            tp_worker_pool_->dispatch(
            [this,
             tokens,
             seq_len,
             force_prefill_phase,
             chunk_schedule_policy,
             chunk_schedule_pad_token_id,
             chunk_schedule_allow_padded_execution,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
                {
                    // Propagate profiler phases from caller thread
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    // Set per-device profiler context so ROCm/CUDA profilers
                    // can attribute kernel times to specific GPUs
                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    IInferenceRunner *runner_iface = device_runners_[i].get();
                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " runner=" << static_cast<void *>(runner_iface)
                                  << " seq_len=" << seq_len);
                    }
                    try
                    {
                        const bool ok = chunk_schedule_policy
                                            ? runner_iface->forwardPrefillChunkSchedule(
                                                  tokens,
                                                  seq_len,
                                                  *chunk_schedule_policy,
                                                  chunk_schedule_pad_token_id,
                                                  chunk_schedule_allow_padded_execution)
                                            : (force_prefill_phase
                                                   ? runner_iface->forwardPrefill(tokens, seq_len)
                                                   : runner_iface->forward(tokens, seq_len));
                        if (debugEnv().tp_collective_contract_trace)
                        {
                            LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_leave"
                                      << " worker=" << i
                                      << " device=" << device_id.toString()
                                      << " success=" << (ok ? 1 : 0));
                        }
                        return ok;
                    }
                    catch (...)
                    {
                        if (debugEnv().tp_collective_contract_trace)
                        {
                            LOG_ERROR("[TP_WORKER_CONTRACT] event=forward_exception"
                                      << " worker=" << i
                                      << " device=" << device_id.toString());
                        }
                        throw;
                    }
                });

            launch_t1 = decode_breakdown ? std::chrono::high_resolution_clock::now()
                                         : std::chrono::high_resolution_clock::time_point{};

            // Collect results from all workers. LLAMINAR_TP_COLLECT_TIMEOUT_MS
            // is enforced inside LocalTP/NCCL/RCCL rendezvous calls; it must not
            // be treated as a wall-clock limit for the whole runner forward.
            const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            auto results = tp_worker_pool_->collectAll(collect_timeout_ms);

            // Process results with fault-tolerant exception handling.
            // IMPORTANT: Store the FIRST substantive exception. When one device
            // throws (e.g., VerificationFailure), it can cause CUDA/HIP context
            // destruction, making other devices fail with misleading "context is
            // destroyed" errors. We want to surface the original root cause.
            bool worker_timeout = false;
            for (auto &r : results)
            {
                if (!r.completed)
                {
                    LOG_ERROR("RankOrchestrator::forwardTP: Device "
                              << r.worker_index << " did not complete (stuck)");
                    worker_timeout = true;
                    all_success = false;
                    if (collect_timeout_ms > 0)
                    {
                        abortAfterTPWorkerTimeout(
                            "forwardTP",
                            collect_timeout_ms,
                            tp_worker_pool_->completedCount(),
                            tp_worker_pool_->numWorkers());
                    }
                    if (tp_ctx_)
                    {
                        LOG_WARN("RankOrchestrator::forwardTP: requesting TP context abort after worker timeout");
                        tp_ctx_->requestAbort();
                    }
                    continue;
                }

                if (r.exception)
                {
                    all_success = false;
                    try
                    {
                        std::rethrow_exception(r.exception);
                    }
                    catch (const std::exception &e)
                    {
                        std::string error_msg = e.what();
                        bool is_context_destroyed =
                            (error_msg.find("context is destroyed") != std::string::npos ||
                             error_msg.find("context destroyed") != std::string::npos ||
                             error_msg.find("error 709") != std::string::npos);

                        if (!first_exception)
                        {
                            first_exception = r.exception;
                            first_exception_device = r.worker_index;
                            LOG_ERROR("RankOrchestrator::forwardTP: Device "
                                      << r.worker_index
                                      << " threw PRIMARY exception: " << error_msg);
                        }
                        else if (!is_context_destroyed)
                        {
                            // Substantive error — replace if first was a context error
                            try
                            {
                                std::rethrow_exception(first_exception);
                            }
                            catch (const std::exception &first_e)
                            {
                                std::string first_msg = first_e.what();
                                bool first_is_ctx =
                                    (first_msg.find("context is destroyed") != std::string::npos ||
                                     first_msg.find("context destroyed") != std::string::npos ||
                                     first_msg.find("error 709") != std::string::npos);
                                if (first_is_ctx)
                                {
                                    LOG_WARN("RankOrchestrator::forwardTP: Replacing "
                                             "secondary context error with primary error from device "
                                             << r.worker_index);
                                    first_exception = r.exception;
                                    first_exception_device = r.worker_index;
                                }
                            }
                            LOG_ERROR("RankOrchestrator::forwardTP: Device "
                                      << r.worker_index
                                      << " threw exception: " << error_msg);
                        }
                        else
                        {
                            LOG_WARN("RankOrchestrator::forwardTP: Device "
                                     << r.worker_index
                                     << " threw SECONDARY exception (likely due to primary failure): "
                                     << error_msg);
                        }
                    }
                }
                else if (!r.success)
                {
                    LOG_ERROR("RankOrchestrator::forwardTP: Device "
                              << r.worker_index << " forward failed");
                    all_success = false;
                }
            }
            if (worker_timeout && collect_timeout_ms > 0)
            {
                abortAfterTPWorkerTimeout(
                    "forwardTP",
                    collect_timeout_ms,
                    tp_worker_pool_->completedCount(),
                    tp_worker_pool_->numWorkers());
            }

            // Capture timing for decode breakdown (only in parallel mode)
            if (decode_breakdown)
            {
                auto collect_t1 = std::chrono::high_resolution_clock::now();
                double launch_us = std::chrono::duration<double, std::micro>(
                                       launch_t1 - launch_t0)
                                       .count();
                double wait_us = std::chrono::duration<double, std::micro>(
                                     collect_t1 - launch_t1)
                                     .count();
                double total_us = std::chrono::duration<double, std::micro>(
                                      collect_t1 - launch_t0)
                                      .count();
                if (tp_timing)
                {
                    LOG_DEBUG("[DECODE_BREAKDOWN] launch=" << std::fixed << std::setprecision(1)
                                                           << launch_us << "us"
                                                           << " wait=" << wait_us << "us"
                                                           << " total=" << total_us << "us");
                }
            }
        }

        // If we captured an exception, re-throw it with full context
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardTP: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        auto tp_t1 = collect_timing ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};

        if (all_success)
        {
            // Gather logits from all devices (delegates to LogitsGatherer)
            bool need_gather = logits_gatherer_ && logits_gatherer_->needsGather(seq_len);

            if (!need_gather && seq_len > 1)
            {
                static bool logged_prefill_skip = false;
                if (!logged_prefill_skip)
                {
                    LOG_DEBUG("[forwardTP] Skipping prefill logits gather (seq_len="
                              << seq_len << ") — prefill logits are not consumed");
                    logged_prefill_skip = true;
                }
            }

            // During prefill the LM head computes only 1 row of logits (the
            // last-token position) written to row 0 of logits_local.  We must
            // gather exactly 1 row; gathering seq_len rows would include
            // uninitialised data in rows 1..seq_len-1.
            size_t gather_rows = (seq_len > 1) ? 1 : static_cast<size_t>(seq_len);
            if (need_gather && !logits_gatherer_->gather(device_runners_, gather_rows, vocab_size()))
            {
                LOG_ERROR("RankOrchestrator::forwardTP: Failed to gather logits");
                all_success = false;
            }

            // Update position tracking
            current_position_ += seq_len;
            current_padded_seq_len_ = seq_len;
            if (current_sequence_lengths_.empty())
            {
                current_sequence_lengths_.assign(
                    static_cast<size_t>(std::max(1, config_.batch_size)),
                    current_position_ - seq_len);
            }
            for (int &length : current_sequence_lengths_)
            {
                length += seq_len;
            }
            stats_dirty_ = true;

            // After first prefill, release host-resident weight data.
            // GPU kernels (e.g., embedding repack) have now uploaded their own
            // device copies, and later MoE migration must use prepared/packed
            // weight transfer rather than falling back to raw GGUF bytes.
            if (!host_resident_released_ && seq_len > 1 && model_ctx_)
            {
                host_resident_released_ = true;
                if (auto wm = model_ctx_->weightManager())
                {
                    wm->releaseHostResidentWeightData();
                    if (!mmap_dontneed_advised_)
                    {
                        mmap_dontneed_advised_ = true;
                        if (synchronizeGpuBackendsBeforeRankMmapRelease(config_))
                        {
                            if (debugEnv().vram_trace)
                                LOG_TRACE("[VRAM_TRACE] rank_mmap_release.before_advise phase=after_first_prefill");
                            const size_t advised_bytes = wm->adviseMmapDontneed();
                            if (debugEnv().vram_trace)
                                LOG_TRACE("[VRAM_TRACE] rank_mmap_release.after_advise phase=after_first_prefill bytes="
                                         << advised_bytes);
                        }
                        else
                        {
                            LOG_WARN("RankOrchestrator: skipping mmap DONTNEED after prefill because GPU synchronization failed");
                        }
                    }
                }
            }
        }

        if (collect_timing)
        {
            auto tp_t2 = std::chrono::high_resolution_clock::now();
            double forward_ms = std::chrono::duration<double, std::milli>(tp_t1 - tp_t0).count();
            double gather_ms = std::chrono::duration<double, std::milli>(tp_t2 - tp_t1).count();
            double total_ms = std::chrono::duration<double, std::milli>(tp_t2 - tp_t0).count();

            if (tp_timing)
            {
                LOG_DEBUG("[TP_TIMING] seq_len=" << seq_len
                                                 << " forward=" << std::fixed << std::setprecision(3) << forward_ms << "ms"
                                                 << " gather=" << std::fixed << std::setprecision(3) << gather_ms << "ms"
                                                 << " total=" << std::fixed << std::setprecision(3) << total_ms << "ms");
            }

            // Accumulate TP decode stats for profiling summary at benchmark end.
            // dispatch_ms = time to wake workers, wait_ms = time blocked on collect.
            // These are computed from DECODE_BREAKDOWN timestamps which are only
            // available in parallel (non-serialized) mode for decode (seq_len==1).
            if (tp_profiling && seq_len == 1)
            {
                double dispatch_ms = 0, wait_ms = 0;
                if (decode_breakdown && !serialize_devices)
                {
                    dispatch_ms = std::chrono::duration<double, std::milli>(launch_t1 - launch_t0).count();
                    wait_ms = std::chrono::duration<double, std::milli>(tp_t1 - launch_t1).count();
                }
                else
                {
                    // Serial mode or timing unavailable: entire forward time is wait
                    wait_ms = forward_ms;
                }
                tp_decode_stats_.record(total_ms, dispatch_ms, wait_ms, gather_ms);
            }
        }

        if (all_success)
            tp_first_forward_completed_ = true;

        return all_success;
    }

    // =========================================================================
    // PP Mode Forward Implementation (sequential pipeline execution)
    // =========================================================================
    bool RankOrchestrator::forwardPP(const int *tokens, int seq_len, bool force_prefill_phase)
    {
        if (pp_stage_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::forwardPP: No PP stage runners available");
            return false;
        }

        if (!pp_ctx_)
        {
            LOG_ERROR("RankOrchestrator::forwardPP: No LocalPPContext available for transfers");
            return false;
        }

        const size_t num_stages = pp_stage_runners_.size();

        LOG_DEBUG("RankOrchestrator::forwardPP: seq_len=" << seq_len
                                                          << " num_stages=" << num_stages);

        // =====================================================================
        // Stage 0: Embedding + first layers (receives tokens as input)
        // =====================================================================
        auto &stage0_runner = pp_stage_runners_[0];
        if (!stage0_runner)
        {
            LOG_ERROR("RankOrchestrator::forwardPP: Stage 0 runner is null");
            return false;
        }

        // PP device coherence: After PP transfer in the previous iteration, the
        // hidden state tensor's gpu_device_ may point to the *destination* device
        // (e.g., ROCm:1) instead of this stage's device (ROCm:0). Promote the
        // secondary buffer back to primary so kernels and transitionToWithEvent
        // operate on the correct device.
        if (pp_ctx_ && num_stages > 1)
        {
            TensorBase *hidden = stage0_runner->getHiddenState();
            if (hidden)
            {
                DeviceId stage0_device = pp_ctx_->deviceForStage(0).toLocalDeviceId();
                auto cur = hidden->current_device();
                if (cur.has_value() && *cur != stage0_device)
                {
                    LOG_DEBUG("RankOrchestrator::forwardPP: Re-asserting hidden state device from "
                              << cur->toString() << " to " << stage0_device.toString());
                    hidden->allocateOnDevice(stage0_device);
                }
            }
        }

        LOG_DEBUG("RankOrchestrator::forwardPP: Executing stage 0 (embedding)");
        if (!(force_prefill_phase
                  ? stage0_runner->forwardPrefill(tokens, seq_len)
                  : stage0_runner->forward(tokens, seq_len)))
        {
            LOG_ERROR("RankOrchestrator::forwardPP: Stage 0 forward failed");
            return false;
        }

        // =====================================================================
        // Intermediate stages: Transfer activations and continue execution
        // =====================================================================
        for (size_t stage_idx = 1; stage_idx < num_stages; ++stage_idx)
        {
            auto &prev_runner = pp_stage_runners_[stage_idx - 1];
            auto &curr_runner = pp_stage_runners_[stage_idx];

            if (!curr_runner)
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << stage_idx << " runner is null");
                return false;
            }

            // Get hidden state from previous stage
            // DeviceGraphOrchestrator stores hidden state in InferenceState
            TensorBase *hidden_state = prev_runner->getHiddenState();
            if (!hidden_state)
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << (stage_idx - 1)
                                                                << " has no hidden state to transfer");
                return false;
            }

            // Transfer activations from previous stage to current stage
            // Only transfer the active region (seq_len tokens), not the full buffer
            const size_t active_bytes = pp_activation_contract_
                                            ? pp_activation_contract_->transfer(static_cast<int>(stage_idx - 1)).activeBytes(seq_len)
                                            : static_cast<size_t>(seq_len) * model_ctx_->embeddingLength() * sizeof(float);

            LOG_DEBUG("RankOrchestrator::forwardPP: Transferring hidden state from stage "
                      << (stage_idx - 1) << " to stage " << stage_idx
                      << " (" << active_bytes << " bytes for " << seq_len << " tokens)");

            if (!pp_ctx_->transfer(hidden_state, static_cast<int>(stage_idx - 1),
                                   static_cast<int>(stage_idx), active_bytes))
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Transfer from stage "
                          << (stage_idx - 1) << " to stage " << stage_idx << " failed");
                return false;
            }

            // Set the hidden state as input for current stage
            curr_runner->setHiddenState(hidden_state);

            LOG_DEBUG("RankOrchestrator::forwardPP: Executing stage " << stage_idx);

            /*
             * Non-head PP stages consume transferred hidden activations rather
             * than embedding token ids. The final stage still needs the raw
             * token ids when MTP is enabled so its shifted prefill cache can be
             * populated for the sidecar. Passing tokens here does not make the
             * stage re-run embedding; has_embedding=false keeps the graph on
             * the hidden-state input path.
             */
            const int *stage_tokens =
                curr_runner.get() == finalPPSidecarRunner() ? tokens : nullptr;
            if (!(force_prefill_phase
                      ? curr_runner->forwardPrefill(stage_tokens, seq_len)
                      : curr_runner->forward(stage_tokens, seq_len)))
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << stage_idx << " forward failed");
                return false;
            }

            // Clear hidden state input for clean state on next forward
            curr_runner->clearHiddenStateInput();
        }

        // =====================================================================
        // Copy logits from last stage to combined buffer
        // =====================================================================
        {
            int last_stage = static_cast<int>(num_stages - 1);
            if (last_stage >= 0 && static_cast<size_t>(last_stage) < pp_stage_runners_.size() && pp_stage_runners_[last_stage])
            {
                if (!logits_gatherer_)
                {
                    logits_gatherer_ = std::make_unique<LogitsGatherer>(0, 0);
                    applyLogitsGatherSkipFlags();
                }
                logits_gatherer_->copyFromStage(*pp_stage_runners_[last_stage],
                                                0, config_.batch_size, config_.max_seq_len);
            }
        }

        // =====================================================================
        // Update position tracking for PP mode
        // Each PP stage runner updates its own position internally, but we also
        // need to update current_position_ for consistency with TP mode and
        // get_position() queries.
        // =====================================================================
        current_position_ += seq_len;

        LOG_DEBUG("RankOrchestrator::forwardPP: Complete, all " << num_stages << " stages executed"
                                                                << ", position now " << current_position_);
        return true;
    }

    IInferenceRunner *RankOrchestrator::finalPPSidecarRunner()
    {
        if (pp_stage_runners_.empty())
            return nullptr;
        return pp_stage_runners_.back().get();
    }

    const IInferenceRunner *RankOrchestrator::finalPPSidecarRunner() const
    {
        if (pp_stage_runners_.empty())
            return nullptr;
        return pp_stage_runners_.back().get();
    }

    void RankOrchestrator::refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication()
    {
        /*
         * The child runners own the actual published state.  RankOrchestrator
         * mirrors a small public bookkeeping surface for callers that cannot
         * delegate to a single child, so after MTP publication we adopt the
         * primary child's accepted-prefix boundary instead of keeping the
         * verifier graph's temporary row count.
         */
        const IInferenceRunner *primary = nullptr;
        if (!pp_stage_runners_.empty())
        {
            primary = pp_stage_runners_.front().get();
        }
        else if (!device_runners_.empty())
        {
            primary = device_runners_.front().get();
        }

        if (!primary)
        {
            current_position_ = 0;
            current_batch_size_ = std::max(1, config_.batch_size);
            current_padded_seq_len_ = 0;
            current_sequence_lengths_.assign(
                static_cast<size_t>(current_batch_size_),
                0);
            stats_dirty_ = true;
            return;
        }

        current_position_ = primary->get_position();
        current_batch_size_ = std::max(1, primary->batch_size());
        current_padded_seq_len_ = primary->padded_seq_len();

        const std::vector<int> &child_lengths = primary->sequence_lengths();
        if (!child_lengths.empty())
        {
            current_sequence_lengths_ = child_lengths;
        }
        else
        {
            current_sequence_lengths_.assign(
                static_cast<size_t>(current_batch_size_),
                current_position_);
        }
        if (current_sequence_lengths_.size() <
            static_cast<size_t>(current_batch_size_))
        {
            current_sequence_lengths_.resize(
                static_cast<size_t>(current_batch_size_),
                current_position_);
        }
        stats_dirty_ = true;
    }

    int RankOrchestrator::sampleGreedyOnDevice()
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyOnDevice();
        }
        if (mode_ != ParallelismMode::TP || device_runners_.size() < 2)
            return -1;

        bool any_local_logits = false;
        bool all_local_logits = true;
        for (const auto &runner : device_runners_)
        {
            const bool has_local = runner && runner->hasLogitsLocal();
            any_local_logits = any_local_logits || has_local;
            all_local_logits = all_local_logits && has_local;
        }

        if (all_local_logits)
            return DeviceSampler::sampleGreedy(device_runners_);
        if (!any_local_logits && !device_runners_.empty() && device_runners_[0])
            return device_runners_[0]->sampleGreedyOnDevice();
        return -1;
    }

    int RankOrchestrator::sampleOnDevice(const SamplingParams &params)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleOnDevice(params);
        }
        if (params.is_greedy())
            return sampleGreedyOnDevice();
        if (mode_ != ParallelismMode::TP || device_runners_.size() < 2)
            return -1;

        bool any_local_logits = false;
        bool all_local_logits = true;
        for (const auto &runner : device_runners_)
        {
            const bool has_local = runner && runner->hasLogitsLocal();
            any_local_logits = any_local_logits || has_local;
            all_local_logits = all_local_logits && has_local;
        }

        if (all_local_logits)
            return DeviceSampler::sample(device_runners_, params);
        if (!any_local_logits && !device_runners_.empty() && device_runners_[0])
            return device_runners_[0]->sampleOnDevice(params);
        return -1;
    }

    bool RankOrchestrator::requiresMPICoordinatedDecodeSampling(
        const SamplingParams &params) const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->requiresMPICoordinatedDecodeSampling(params);
        }

        /*
         * This method must mirror RankOrchestrator::sampleGreedyOnDevice() /
         * sampleOnDevice(), not a child runner's latent capability.  The TP path
         * samples through DeviceSampler over local child runners or falls back
         * to rank-local gathered logits; it does not call
         * child->sampleGreedyOnDevice(), so worker ranks must not enter child
         * sampling collectives on its behalf.
         */
        return false;
    }

    const float *RankOrchestrator::logits() const
    {
        PerfStatsCollector::addCounter(
            "sampling",
            "host_logits_access",
            1.0,
            {},
            primaryDeviceId().toString(),
            {{"source", "rank_orchestrator_logits"},
             {"mode", std::to_string(static_cast<int>(mode_))},
             {"tp_children", std::to_string(device_runners_.size())},
             {"pp_children", std::to_string(pp_stage_runners_.size())},
             {"gatherer_allocated",
              (logits_gatherer_ && logits_gatherer_->isAllocated()) ? "true" : "false"}});

        // For PP mode: return combined logits (copied from final stage)
        if (mode_ == ParallelismMode::PP || mode_ == ParallelismMode::TP_PP)
        {
            if (logits_gatherer_ && logits_gatherer_->isAllocated())
            {
                return logits_gatherer_->data();
            }
            // Fallback: try to get from final PP stage
            if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
            {
                return pp_stage_runners_.back()->logits();
            }
            return nullptr;
        }

        // For TP mode: return combined logits if available (multi-device)
        if (logits_gatherer_ && logits_gatherer_->isAllocated() && device_runners_.size() > 1)
        {
            return logits_gatherer_->data();
        }

        // For single device, return primary device's logits
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->logits();
        }

        return nullptr;
    }

    DeviceId RankOrchestrator::primaryDeviceId() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->primaryDeviceId();
        }
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->primaryDeviceId();
        }
        return DeviceId::cpu();
    }

    bool RankOrchestrator::forwardMTP(int32_t draft_condition_token)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_forward_mtp_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * PP verifier replay stays pipeline-wide through forwardPP(), but
             * the MTP sidecar consumes the terminal hidden row and produces MTP
             * logits on the final stage.  Running it on earlier stages would be
             * nonsensical: they do not own final norm, LM head, or sidecar
             * logits.  This is intentionally not TP fan-out.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_pp_final_stage_calls",
                1.0,
                "decode",
                "rank");
            return pp_sidecar->forwardMTP(draft_condition_token);
        }
        if (device_runners_.empty())
        {
            return false;
        }

        if (device_runners_.size() == 1)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_single_participant_calls",
                1.0,
                "decode",
                "rank");
            return device_runners_[0] && device_runners_[0]->forwardMTP(draft_condition_token);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP worker failure detected — aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_forward_mtp_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, draft_condition_token, kernel_phase, rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_mtp_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token=" << draft_condition_token);
                    }

                    const bool ok = device_runners_[i] &&
                                    device_runners_[i]->forwardMTP(draft_condition_token);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_mtp_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_forward_mtp_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::forwardMTP: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "forwardMTP",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::forwardMTP: requesting TP context abort after worker timeout");
                    tp_ctx_->requestAbort();
                }
                continue;
            }

            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }

            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::forwardMTP: Device "
                          << r.worker_index << " MTP forward failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTP",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTP: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        return all_success;
    }

    bool RankOrchestrator::forwardMTPForDeviceSampling(int32_t draft_condition_token)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardMTPForDeviceSampling(draft_condition_token);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardMTPForDeviceSampling(
                draft_condition_token);
        }
        if (!supportsMTPSidecarLogitsStreamHandoff())
        {
            LOG_ERROR("[RankOrchestrator] LocalTP MTP device-sampling sidecar requires stream handoff on every participant");
            return false;
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP device-sampling worker failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             draft_condition_token,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return device_runners_[i] &&
                       device_runners_[i]->forwardMTPForDeviceSampling(
                           draft_condition_token);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPForDeviceSampling: Device "
                          << r.worker_index << " did not complete");
                worker_timeout = true;
                all_success = false;
                continue;
            }
            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPForDeviceSampling: Device "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPForDeviceSampling",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPForDeviceSampling: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_device_sampling_calls",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
        }
        return all_success;
    }

    bool RankOrchestrator::supportsChainedMTPDrafts() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsChainedMTPDrafts();
        }

        if (device_runners_.empty())
        {
            return false;
        }

        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->supportsChainedMTPDrafts())
            {
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_forward_mtp_chained_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            if (!pp_sidecar->supportsChainedMTPDrafts())
            {
                LOG_ERROR("[RankOrchestrator] Final PP stage does not support chained MTP sidecar execution");
                return false;
            }
            return pp_sidecar->forwardMTPFromLastDraft(
                draft_condition_token,
                position_id);
        }
        if (device_runners_.empty())
        {
            LOG_ERROR("[RankOrchestrator] Chained MTP drafts require at least one rank participant");
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i] ||
                !device_runners_[i]->supportsChainedMTPDrafts())
            {
                LOG_ERROR("[RankOrchestrator] Chained MTP draft participant "
                          << i << " does not support chained sidecar execution");
                return false;
            }
        }

        if (device_runners_.size() == 1)
        {
            return device_runners_[0]->forwardMTPFromLastDraft(
                draft_condition_token,
                position_id);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] Chained MTP worker failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_forward_mtp_chained_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, draft_condition_token, position_id, kernel_phase,
                 rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_mtp_chained_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token=" << draft_condition_token
                                  << " position=" << position_id);
                    }

                    const bool ok =
                        device_runners_[i] &&
                        device_runners_[i]->forwardMTPFromLastDraft(
                            draft_condition_token,
                            position_id);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_mtp_chained_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_forward_mtp_chained_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }

        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromLastDraft: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "forwardMTPFromLastDraft",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::forwardMTPFromLastDraft: requesting TP context abort after worker timeout");
                    tp_ctx_->requestAbort();
                }
                continue;
            }

            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }

            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromLastDraft: Device "
                          << r.worker_index << " chained MTP forward failed");
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPFromLastDraft",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPFromLastDraft: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        return all_success;
    }

    bool RankOrchestrator::forwardMTPFromLastDraftForDeviceSampling(
        int32_t draft_condition_token,
        int position_id)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardMTPFromLastDraftForDeviceSampling(
                draft_condition_token,
                position_id);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardMTPFromLastDraftForDeviceSampling(
                draft_condition_token,
                position_id);
        }
        return false;
    }

    bool RankOrchestrator::forwardMTPFromDeviceDraftForDeviceSampling(
        int draft_sample_slot,
        int position_id)
    {
        /*
         * LocalPP samples draft tokens in final-stage device memory, while the
         * next verifier input starts at pipeline stage 0.  Until the pipeline
         * has a first-stage-owned token slot or an explicit transfer contract,
         * rank-level PP must not advertise device-token sidecar chaining.
         */
        if (finalPPSidecarRunner())
        {
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardMTPFromDeviceDraftForDeviceSampling(
                draft_sample_slot,
                position_id);
        }
        if (!supportsMTPDeviceDraftTokenInput() ||
            draft_sample_slot < 0 ||
            draft_sample_slot >= kRankStochasticMaxSlots ||
            position_id < 0)
        {
            LOG_ERROR("[RankOrchestrator] LocalTP device-draft sidecar requires staged child draft slots on every participant");
            return false;
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP device-draft worker failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             draft_sample_slot,
             position_id,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return device_runners_[i] &&
                       device_runners_[i]->forwardMTPFromDeviceDraftForDeviceSampling(
                           draft_sample_slot,
                           position_id);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceDraftForDeviceSampling: Device "
                          << r.worker_index << " did not complete");
                worker_timeout = true;
                all_success = false;
                continue;
            }
            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceDraftForDeviceSampling: Device "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPFromDeviceDraftForDeviceSampling",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceDraftForDeviceSampling: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_device_draft_slot_calls",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(draft_sample_slot)}});
        }
        return all_success;
    }

    bool RankOrchestrator::forwardMTPFromDeviceTargetForDeviceSampling(
        int target_sample_slot,
        int position_id)
    {
        if (finalPPSidecarRunner())
        {
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardMTPFromDeviceTargetForDeviceSampling(
                target_sample_slot,
                position_id);
        }
        if (!supportsMTPDeviceDraftTokenInput() ||
            target_sample_slot < 0 ||
            target_sample_slot >= kRankStochasticMaxSlots ||
            position_id < 0)
        {
            LOG_ERROR("[RankOrchestrator] LocalTP device-target sidecar requires staged child target slots on every participant");
            return false;
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP device-target worker failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             target_sample_slot,
             position_id,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return device_runners_[i] &&
                       device_runners_[i]->forwardMTPFromDeviceTargetForDeviceSampling(
                           target_sample_slot,
                           position_id);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceTargetForDeviceSampling: Device "
                          << r.worker_index << " did not complete");
                worker_timeout = true;
                all_success = false;
                continue;
            }
            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceTargetForDeviceSampling: Device "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPFromDeviceTargetForDeviceSampling",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceTargetForDeviceSampling: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_device_target_slot_calls",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(target_sample_slot)}});
        }
        return all_success;
    }

    bool RankOrchestrator::forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index)
    {
        if (finalPPSidecarRunner())
        {
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                logical_state,
                request_index);
        }
        if (!logical_state.valid() ||
            logical_state.target_positions_device !=
                rank_resident_logical_state_marker_.data() ||
            logical_state.live_state_epoch !=
                rank_resident_logical_state_epoch_ ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            LOG_ERROR("[RankOrchestrator] Resident logical-state sidecar received a non-owned LocalTP mailbox");
            return false;
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] resident logical-state sidecar failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        std::vector<std::string> child_errors(device_runners_.size());
        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        tp_worker_pool_->dispatch(
            [this,
             request_index,
             &child_errors,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                const bool ok =
                    device_runners_[i] &&
                    device_runners_[i]->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                        rank_resident_child_logical_state_handles_[i],
                        request_index);
                if (!ok)
                    child_errors[i] = "resident logical-state sidecar failed";
                return ok;
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        auto results =
            tp_worker_pool_->collectAll(effectiveTPWorkerJoinTimeoutMs());
        for (auto &r : results)
        {
            if (!r.completed || !r.success)
                all_success = false;
            if (r.exception && !first_exception)
            {
                first_exception = r.exception;
                first_exception_device = r.worker_index;
                all_success = false;
            }
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Resident logical-state sidecar rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_resident_logical_state_sidecar_prelaunches",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_index", std::to_string(request_index)}});
        }
        return all_success;
    }

    DeviceResidentLogicalSequenceStateHandle
    RankOrchestrator::deviceResidentLogicalSequenceState() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->deviceResidentLogicalSequenceState();
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->deviceResidentLogicalSequenceState();
        }
        if (device_runners_.size() < 2 ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            return {};
        }

        int request_count = -1;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            const DeviceResidentLogicalSequenceStateHandle &child =
                rank_resident_child_logical_state_handles_[i];
            if (!child.valid())
                return {};
            if (request_count < 0)
                request_count = child.request_count;
            if (child.request_count != request_count)
                return {};
        }
        if (request_count <= 0)
            return {};

        rank_resident_logical_state_marker_[0] = request_count;
        DeviceResidentLogicalSequenceStateHandle handle;
        handle.target_positions_device =
            rank_resident_logical_state_marker_.data();
        handle.target_sequence_lengths_device =
            rank_resident_logical_state_marker_.data();
        handle.accepted_state_counts_device =
            rank_resident_logical_state_marker_.data();
        handle.next_condition_tokens_device =
            rank_resident_logical_state_marker_.data();
        handle.all_drafts_accepted_flags_device =
            rank_resident_logical_state_marker_.data();
        handle.stopped_flags_device =
            rank_resident_logical_state_marker_.data();
        handle.publication_ok_flags_device =
            rank_resident_logical_state_marker_.data();
        handle.request_count = request_count;
        handle.device = primaryDeviceId();
        handle.stream = &rank_resident_logical_state_stream_token_;
        handle.ready_event = &rank_resident_logical_state_ready_event_token_;
        handle.live_state_epoch = rank_resident_logical_state_epoch_;
        return handle;
    }

    bool RankOrchestrator::commitMTPShiftedRowsFromLastForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens)
    {
        return commitMTPShiftedRowsFromPartialForward(
            tokens,
            token_count,
            already_appended_tokens,
            token_count);
    }

    bool RankOrchestrator::commitMTPShiftedRowsFromPartialForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens,
        int main_forward_token_count,
        bool allow_speculative_discard,
        int position_offset_override,
        int already_appended_shifted_kv_tokens)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_shifted_commit_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (token_count <= already_appended_tokens)
            return true;
        if (!tokens || token_count <= 0)
            return false;
        const int catchup_token_count = token_count - already_appended_tokens;
        const int hidden_source_row_start = already_appended_tokens - 1;
        const int hidden_source_row_end = hidden_source_row_start + catchup_token_count;
        if (main_forward_token_count <= 0 ||
            hidden_source_row_start < 0 ||
            hidden_source_row_end > main_forward_token_count)
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowsFromPartialForward: invalid main_forward_token_count="
                      << main_forward_token_count << " token_count=" << token_count
                      << " already_appended_tokens=" << already_appended_tokens
                      << " catchup_token_count=" << catchup_token_count
                      << " hidden_source_row_start=" << hidden_source_row_start
                      << " hidden_source_row_end=" << hidden_source_row_end);
            return false;
        }
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * PP replay has already produced the accepted hidden rows on the
             * final stage.  Only that stage owns the shifted MTP KV cache, so
             * commit the accepted sidecar rows there instead of iterating every
             * pipeline participant.
             */
            return pp_sidecar->commitMTPShiftedRowsFromPartialForward(
                tokens,
                token_count,
                already_appended_tokens,
                main_forward_token_count,
                allow_speculative_discard,
                position_offset_override,
                already_appended_shifted_kv_tokens);
        }
        if (device_runners_.empty())
            return false;

        if (device_runners_.size() == 1)
        {
            return device_runners_[0] &&
                   device_runners_[0]->commitMTPShiftedRowsFromPartialForward(
                       tokens,
                       token_count,
                       already_appended_tokens,
                       main_forward_token_count,
                       allow_speculative_discard,
                       position_offset_override,
                       already_appended_shifted_kv_tokens);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP shifted-row commit failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        const std::vector<int32_t> committed_tokens(tokens, tokens + token_count);
        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_shifted_commit_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, &committed_tokens, token_count, already_appended_tokens,
                 main_forward_token_count, allow_speculative_discard,
                 position_offset_override, already_appended_shifted_kv_tokens,
                 kernel_phase, rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_commit_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token_count=" << token_count
                                  << " already_appended=" << already_appended_tokens
                                  << " main_forward_token_count=" << main_forward_token_count);
                    }

                    const bool ok = device_runners_[i] &&
                                    device_runners_[i]->commitMTPShiftedRowsFromPartialForward(
                                        committed_tokens.data(),
                                        token_count,
                                        already_appended_tokens,
                                        main_forward_token_count,
                                        allow_speculative_discard,
                                        position_offset_override,
                                        already_appended_shifted_kv_tokens);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_commit_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_mtp_shifted_commit_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }

        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowsFromLastForward: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "commitMTPShiftedRowsFromLastForward",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::commitMTPShiftedRowsFromLastForward: requesting TP context abort after worker timeout");
                    tp_ctx_->requestAbort();
                }
                continue;
            }

            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }

            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowsFromLastForward: Device "
                          << r.worker_index << " shifted-row commit failed");
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "commitMTPShiftedRowsFromLastForward",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowsFromLastForward: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        return all_success;
    }

    bool RankOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(
        int target_sample_slot,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->commitMTPShiftedRowFromDeviceTargetSample(
                target_sample_slot,
                already_appended_tokens,
                allow_speculative_discard,
                position_offset_override);
        }
        if (device_runners_.size() != 1 || !device_runners_[0])
        {
            LOG_ERROR("[RankOrchestrator] Device-target shifted MTP commit is not enabled for multi-participant TP domains yet");
            return false;
        }
        return device_runners_[0]->commitMTPShiftedRowFromDeviceTargetSample(
            target_sample_slot,
            already_appended_tokens,
            allow_speculative_discard,
            position_offset_override);
    }

    bool RankOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->commitMTPShiftedRowFromDeviceResidentLogicalState(
                logical_state,
                request_index,
                already_appended_tokens,
                allow_speculative_discard,
                position_offset_override);
        }
        if (device_runners_.size() != 1 || !device_runners_[0])
        {
            if (!logical_state.valid() ||
                logical_state.target_positions_device !=
                    rank_resident_logical_state_marker_.data() ||
                logical_state.live_state_epoch !=
                    rank_resident_logical_state_epoch_ ||
                rank_resident_child_logical_state_handles_.size() !=
                    device_runners_.size())
            {
                LOG_ERROR("[RankOrchestrator] Resident logical-state shifted MTP commit received a non-owned LocalTP mailbox");
                return false;
            }

            if (!tp_worker_pool_)
            {
                tp_worker_pool_ =
                    std::make_unique<TPWorkerPool>(device_runners_.size());
                if (tp_ctx_)
                {
                    tp_worker_pool_->setFailureCallback([this]()
                                                        {
                        LOG_WARN("[TPWorkerPool] resident logical-state shifted commit failure detected - aborting collective backend");
                        tp_ctx_->requestAbort(); });
                }
            }

            auto kernel_phase = KernelProfiler::getCurrentPhase();
            auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            auto kv_phase = KVCacheProfiler::getCurrentPhase();
            auto executor_phase = GraphExecutorStats::currentPhase();

            tp_worker_pool_->dispatch(
                [this,
                 request_index,
                 already_appended_tokens,
                 allow_speculative_discard,
                 position_offset_override,
                 kernel_phase,
                 rocm_phase,
                 cuda_phase,
                 kv_phase,
                 executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    return device_runners_[i] &&
                           device_runners_[i]->commitMTPShiftedRowFromDeviceResidentLogicalState(
                               rank_resident_child_logical_state_handles_[i],
                               request_index,
                               already_appended_tokens,
                               allow_speculative_discard,
                               position_offset_override);
                });

            bool all_success = true;
            std::exception_ptr first_exception = nullptr;
            size_t first_exception_device = 0;
            auto results =
                tp_worker_pool_->collectAll(effectiveTPWorkerJoinTimeoutMs());
            for (auto &r : results)
            {
                if (!r.completed || !r.success)
                    all_success = false;
                if (r.exception && !first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                    all_success = false;
                }
            }
            if (first_exception)
            {
                LOG_ERROR("[RankOrchestrator] Resident logical-state shifted commit rethrowing exception from participant "
                          << first_exception_device);
                std::rethrow_exception(first_exception);
            }
            if (all_success)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "rank_resident_logical_state_shifted_commits",
                    1.0,
                    "decode",
                    "rank",
                    {{"participants", std::to_string(device_runners_.size())},
                     {"request_index", std::to_string(request_index)}});
            }
            return all_success;
        }
        return device_runners_[0]->commitMTPShiftedRowFromDeviceResidentLogicalState(
            logical_state,
            request_index,
            already_appended_tokens,
            allow_speculative_discard,
            position_offset_override);
    }

    bool RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden(
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_shifted_sequential_commit_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            if (already_appended_tokens < 0)
                return false;
            return pp_sidecar->commitMTPShiftedRowFromCurrentTerminalHidden(
                token,
                already_appended_tokens,
                allow_speculative_discard,
                position_offset_override);
        }
        if (already_appended_tokens < 0 || device_runners_.empty())
            return false;

        if (device_runners_.size() == 1)
        {
            return device_runners_[0] &&
                   device_runners_[0]->commitMTPShiftedRowFromCurrentTerminalHidden(
                       token,
                       already_appended_tokens,
                       allow_speculative_discard,
                       position_offset_override);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP sequential shifted-row commit failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_shifted_sequential_commit_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, token, already_appended_tokens, allow_speculative_discard,
                 position_offset_override, kernel_phase, rocm_phase, cuda_phase,
                 kv_phase, executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_sequential_commit_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token=" << token
                                  << " already_appended=" << already_appended_tokens);
                    }

                    const bool ok = device_runners_[i] &&
                                    device_runners_[i]->commitMTPShiftedRowFromCurrentTerminalHidden(
                                        token,
                                        already_appended_tokens,
                                        allow_speculative_discard,
                                        position_offset_override);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_sequential_commit_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_mtp_shifted_sequential_commit_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }

        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "commitMTPShiftedRowFromCurrentTerminalHidden",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden: requesting TP context abort after worker timeout");
                    tp_ctx_->requestAbort();
                }
                continue;
            }

            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }

            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden: Device "
                          << r.worker_index << " shifted-row commit failed");
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "commitMTPShiftedRowFromCurrentTerminalHidden",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        return all_success;
    }

    bool RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden(
        const PrefixStateSnapshot &checkpoint,
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_shifted_checkpoint_terminal_hidden_commit_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (!checkpoint.valid || already_appended_tokens < 0)
            return false;

        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            const PrefixStateSnapshot *tail_checkpoint = &checkpoint;
            if (!checkpoint.participant_snapshots.empty())
                tail_checkpoint = &checkpoint.participant_snapshots.back();
            return pp_sidecar->commitMTPShiftedRowFromCheckpointTerminalHidden(
                *tail_checkpoint,
                token,
                already_appended_tokens,
                allow_speculative_discard,
                position_offset_override);
        }
        if (device_runners_.empty())
            return false;

        const bool aggregate_checkpoint =
            !checkpoint.participant_snapshots.empty();
        const bool shared_logical_checkpoint =
            !aggregate_checkpoint && checkpoint.logical_checkpoint;
        if (aggregate_checkpoint &&
            checkpoint.participant_snapshots.size() != device_runners_.size())
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: participant snapshot count mismatch: snapshots="
                      << checkpoint.participant_snapshots.size()
                      << " runners=" << device_runners_.size());
            return false;
        }
        auto child_checkpoint = [&](size_t i) -> const PrefixStateSnapshot *
        {
            if (aggregate_checkpoint)
                return &checkpoint.participant_snapshots[i];
            if (device_runners_.size() == 1 || shared_logical_checkpoint)
                return &checkpoint;
            return nullptr;
        };

        if (device_runners_.size() == 1)
        {
            const PrefixStateSnapshot *child = child_checkpoint(0);
            return child &&
                   device_runners_[0] &&
                   device_runners_[0]->commitMTPShiftedRowFromCheckpointTerminalHidden(
                       *child,
                       token,
                       already_appended_tokens,
                       allow_speculative_discard,
                       position_offset_override);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] checkpoint-terminal-hidden shifted-row commit failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_shifted_checkpoint_terminal_hidden_commit_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, &checkpoint, &child_checkpoint, token, already_appended_tokens,
                 allow_speculative_discard, position_offset_override,
                 kernel_phase, rocm_phase, cuda_phase, kv_phase,
                 executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    if (i >= device_runners_.size() ||
                        !device_runners_[i])
                    {
                        return false;
                    }
                    const PrefixStateSnapshot *child = child_checkpoint(i);
                    if (!child)
                    {
                        LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: Device "
                                  << i
                                  << " has no participant checkpoint for non-logical multi-device commit");
                        return false;
                    }

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_checkpoint_terminal_hidden_commit_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token=" << token
                                  << " already_appended=" << already_appended_tokens);
                    }

                    const bool ok =
                        device_runners_[i]->commitMTPShiftedRowFromCheckpointTerminalHidden(
                            *child,
                            token,
                            already_appended_tokens,
                            allow_speculative_discard,
                            position_offset_override);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_checkpoint_terminal_hidden_commit_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_mtp_shifted_checkpoint_terminal_hidden_commit_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }

        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                continue;
            }
            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: Device "
                          << r.worker_index << " shifted-row commit failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "commitMTPShiftedRowFromCheckpointTerminalHidden",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        return all_success;
    }

    bool RankOrchestrator::flushPendingMTPWork()
    {
        if (!pp_stage_runners_.empty())
        {
            bool ok = true;
            for (const auto &runner : pp_stage_runners_)
            {
                ok = runner && runner->flushPendingMTPWork() && ok;
            }
            return ok;
        }

        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner && runner->flushPendingMTPWork() && ok;
        }
        return ok;
    }

    bool RankOrchestrator::ensureMTPCheckpointTerminalHidden()
    {
        if (!pp_stage_runners_.empty())
        {
            /*
             * This helper materializes the terminal hidden row consumed by the
             * MTP sidecar. In LocalPP, only the pipeline tail owns the final
             * hidden state, final norm, LM head, and MTP sidecar. Earlier stages
             * own local KV/GDN replay state, but their activation tensors are
             * transferred onward after forwardPP(); asking them to row-select a
             * "terminal" hidden row can race stale producer-side device buffers.
             */
            IInferenceRunner *pp_sidecar = finalPPSidecarRunner();
            return pp_sidecar && pp_sidecar->ensureMTPCheckpointTerminalHidden();
        }

        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner && runner->ensureMTPCheckpointTerminalHidden() && ok;
        }
        return ok;
    }

    const float *RankOrchestrator::mtpLogits() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->mtpLogits();
        }

        if (device_runners_.empty())
        {
            return nullptr;
        }

        bool has_local_mtp_logits = false;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            has_local_mtp_logits = has_local_mtp_logits || runner->hasMTPLogitsLocal();
        }

        if (has_local_mtp_logits)
        {
            std::vector<LogitsLocalInfo> local_infos;
            local_infos.reserve(device_runners_.size());
            for (const auto &runner : device_runners_)
            {
                if (!runner->hasMTPLogitsLocal())
                {
                    LOG_WARN("RankOrchestrator::mtpLogits: mixed local and replicated MTP logits are unsupported");
                    return nullptr;
                }

                LogitsLocalInfo info = runner->getMTPLogitsLocalInfo();
                if (!info)
                    return nullptr;
                local_infos.push_back(info);
            }

            const int full_vocab = vocab_size();
            if (full_vocab <= 0)
                return nullptr;

            if (!mtp_logits_gatherer_ ||
                mtp_logits_gatherer_->bufferNumel() < static_cast<size_t>(full_vocab))
            {
                mtp_logits_gatherer_ = std::make_unique<LogitsGatherer>(full_vocab, 1);
            }

            if (!mtp_logits_gatherer_ ||
                !mtp_logits_gatherer_->gatherLocalInfos(local_infos, 1, full_vocab))
            {
                return nullptr;
            }
            return mtp_logits_gatherer_->data();
        }

        const int compare_vocab = device_runners_[0] ? device_runners_[0]->vocab_size() : 0;
        if (compare_vocab <= 0)
        {
            return nullptr;
        }

        const float *primary_logits = nullptr;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            if (runner->vocab_size() != compare_vocab)
                return nullptr;

            const float *child_logits = runner->mtpLogits();
            if (!child_logits)
                return nullptr;

            if (!primary_logits)
            {
                primary_logits = child_logits;
                continue;
            }

            if (!std::equal(primary_logits, primary_logits + compare_vocab, child_logits))
            {
                LOG_WARN("RankOrchestrator::mtpLogits: child MTP logits diverged in replicated TP path");
                return nullptr;
            }
        }
        return primary_logits;
    }

    int RankOrchestrator::sampleGreedyFromMTPLogitsOnDevice()
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromMTPLogitsOnDevice();
        }

        if (device_runners_.empty())
        {
            return -1;
        }
        if (device_runners_.size() == 1)
        {
            return device_runners_[0] ? device_runners_[0]->sampleGreedyFromMTPLogitsOnDevice() : -1;
        }

        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasMTPLogitsLocal())
            {
                LOG_DEBUG("RankOrchestrator::sampleGreedyFromMTPLogitsOnDevice: "
                          "participant missing local MTP logits");
                return -1;
            }

            LogitsLocalInfo info = runner->consumeMTPLogitsLocalInfoForSampling();
            if (!info)
            {
                LOG_DEBUG("RankOrchestrator::sampleGreedyFromMTPLogitsOnDevice: "
                          "participant returned empty local MTP logits info");
                return -1;
            }
            local_infos.push_back(info);
        }

        const int token = DeviceSampler::sampleGreedyFromLocalInfos(local_infos, 0);
        if (token < 0)
        {
            std::ostringstream oss;
            oss << "RankOrchestrator::sampleGreedyFromMTPLogitsOnDevice: "
                   "sharded MTP sampling failed for "
                << local_infos.size() << " participants";
            for (size_t i = 0; i < local_infos.size(); ++i)
            {
                const auto &info = local_infos[i];
                oss << " [idx=" << i
                    << " tensor=" << (info.tensor ? "yes" : "no")
                    << " gpu_ptr=" << info.gpu_ptr
                    << " device="
                    << (info.device.has_value() ? info.device->toString() : std::string("none"))
                    << " stream=" << info.stream
                    << " vocab_local=" << info.vocab_local
                    << " argmax_capacity=" << info.argmax_partial_capacity
                    << "]";
            }
            LOG_DEBUG(oss.str());
        }
        return token;
    }

    bool RankOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot(
        int draft_sample_slot,
        int32_t *out_token)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                draft_sample_slot,
                out_token);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                draft_sample_slot,
                out_token);
        }
        return sampleRankGreedyMTPLogitsToLocalTPDraftSlot(
            draft_sample_slot,
            out_token);
    }

    bool RankOrchestrator::sampleGreedyFromMainLogitsToDeviceTargetSlot(
        int target_sample_slot,
        int32_t *out_token)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                target_sample_slot,
                out_token);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                target_sample_slot,
                out_token);
        }
        return sampleRankGreedyMainLogitsToLocalTPTargetSlot(
            target_sample_slot,
            out_token);
    }

    int RankOrchestrator::sampleGreedyFromAllPositionLogitsOnDevice(int row)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromAllPositionLogitsOnDevice(row);
        }

        if (device_runners_.empty() || row < 0)
        {
            return -1;
        }
        if (device_runners_.size() == 1)
        {
            return device_runners_[0]
                       ? device_runners_[0]->sampleGreedyFromAllPositionLogitsOnDevice(row)
                       : -1;
        }

        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasAllPositionLogitsLocal())
            {
                return -1;
            }

            LogitsLocalInfo info = runner->getAllPositionLogitsLocalInfo();
            if (!info)
            {
                return -1;
            }
            local_infos.push_back(info);
        }

        return DeviceSampler::sampleGreedyFromLocalInfos(local_infos, row);
    }

    bool RankOrchestrator::sampleGreedyFromAllPositionLogitsOnDeviceRows(
        int start_row,
        int row_count,
        int32_t *out_tokens)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                start_row,
                row_count,
                out_tokens);
        }

        if (device_runners_.empty() ||
            start_row < 0 ||
            row_count <= 0 ||
            !out_tokens)
        {
            return false;
        }
        if (device_runners_.size() == 1)
        {
            return device_runners_[0] &&
                   device_runners_[0]->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                       start_row,
                       row_count,
                       out_tokens);
        }

        bool any_local_logits = false;
        bool all_local_logits = true;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return false;
            const bool has_local = runner->hasAllPositionLogitsLocal();
            any_local_logits = any_local_logits || has_local;
            all_local_logits = all_local_logits && has_local;
        }

        if (!any_local_logits)
        {
            /*
             * Some LocalTP tests and replicated-logits topologies expose a full
             * verifier-logits tensor on each child instead of vocab shards. One
             * representative child can sample those rows directly.
             */
            return device_runners_[0]->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                start_row,
                row_count,
                out_tokens);
        }

        if (!all_local_logits)
        {
            return false;
        }

        /*
         * Column-parallel LocalTP all-position verifier rows are logits shards.
         * Sampling is the verifier replay consumer, so collect one consuming
         * LogitsLocalInfo per child and reuse those exact streams for every
         * row in this batch.  Using the non-consuming getter here would sample
         * on child default streams and can race graph-captured verifier replay.
         */
        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            LogitsLocalInfo info =
                runner->consumeAllPositionLogitsLocalInfoForSampling();
            if (!info)
                return false;
            local_infos.push_back(info);
        }

        if (DeviceSampler::sampleGreedyRowsFromLocalInfos(
                local_infos,
                start_row,
                row_count,
                out_tokens))
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_all_position_verifier_token_batch_samples",
                static_cast<double>(row_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"start_row", std::to_string(start_row)},
                 {"implementation", "batched_cross_shard_argmax"}});
            return true;
        }

        for (int i = 0; i < row_count; ++i)
        {
            const int token = DeviceSampler::sampleGreedyFromLocalInfos(
                local_infos,
                start_row + i);
            if (token < 0)
                return false;
            out_tokens[i] = static_cast<int32_t>(token);
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_all_position_verifier_token_batch_samples",
            static_cast<double>(row_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"start_row", std::to_string(start_row)},
             {"implementation", "serial_cross_shard_argmax"}});
        return true;
    }

    bool RankOrchestrator::resolveRankGreedyOutcomeTokensForLocalTP(
        const int32_t *draft_tokens,
        int draft_token_count,
        std::vector<int32_t> *out_tokens) const
    {
        if (!draft_tokens ||
            draft_token_count <= 0 ||
            !out_tokens)
        {
            return false;
        }

        out_tokens->clear();
        out_tokens->reserve(static_cast<size_t>(draft_token_count));

        int resolved_shadow_count = 0;
        for (int i = 0; i < draft_token_count; ++i)
        {
            int32_t token = draft_tokens[i];
            if (token < 0)
            {
                /*
                 * OrchestrationRunner uses negative shadows only after it has
                 * staged the corresponding compact token into rank-owned slots.
                 * Slot zero is the fixed first-token target slot for the current
                 * single-request transaction; draft entries map one-to-one to
                 * draft slots starting at zero.
                 */
                token = (i == 0)
                            ? rankStochasticTargetSampleToken(/*slot=*/0)
                            : rankStochasticDraftSampleToken(/*slot=*/i - 1);
                ++resolved_shadow_count;
            }
            if (token < 0)
            {
                out_tokens->clear();
                return false;
            }
            out_tokens->push_back(token);
        }

        if (resolved_shadow_count > 0)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_greedy_outcome_deferred_token_resolutions",
                static_cast<double>(resolved_shadow_count),
                "decode",
                "rank",
                {{"tokens", std::to_string(draft_token_count)}});
        }
        return true;
    }

    bool RankOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDevice(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeVerifyBatchOutcome *out)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyGreedyAllPositionBatchOutcomeOnDevice(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyGreedyAllPositionBatchOutcomeOnDevice(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out);
        }

        using namespace sampling_math;
        if (!out)
            return false;
        *out = DeviceSpeculativeVerifyBatchOutcome{};

        const int compare_rows = draft_token_count - 1;
        if (device_runners_.size() < 2 ||
            !draft_tokens ||
            draft_token_count <= 0 ||
            draft_token_count > kSpeculativeBatchMaxRows ||
            compare_rows < 0 ||
            compare_rows > kSpeculativeBatchMaxRows ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens))
        {
            return false;
        }

        std::vector<int32_t> resolved_draft_tokens;
        if (!resolveRankGreedyOutcomeTokensForLocalTP(
                draft_tokens,
                draft_token_count,
                &resolved_draft_tokens))
        {
            return false;
        }

        /*
         * LocalTP all-position verifier logits are vocab shards.  Reduce the
         * compact verifier row batch across child-local shards, then run the
         * same SamplingMath summary used by single-device GPU reducers.  This
         * avoids gathering full logits and keeps accepted-count/ready-token
         * semantics identical to serial greedy decode.
         */
        std::array<int32_t, kSpeculativeBatchMaxRows> verifier_tokens_i32{};
        verifier_tokens_i32.fill(-1);
        if (!sampleGreedyFromAllPositionLogitsOnDeviceRows(
                /*start_row=*/0,
                draft_token_count,
                verifier_tokens_i32.data()))
        {
            return false;
        }

        std::array<int, kSpeculativeBatchMaxRows> verifier_tokens{};
        std::array<int, kSpeculativeBatchMaxRows> packed_draft_tokens{};
        verifier_tokens.fill(-1);
        packed_draft_tokens.fill(-1);
        for (int row = 0; row < draft_token_count; ++row)
        {
            verifier_tokens[static_cast<size_t>(row)] =
                static_cast<int>(verifier_tokens_i32[static_cast<size_t>(row)]);
            packed_draft_tokens[static_cast<size_t>(row)] =
                static_cast<int>(resolved_draft_tokens[static_cast<size_t>(row)]);
        }

        std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens{};
        packed_stop_tokens.fill(-1);
        for (int i = 0; i < stop_token_count; ++i)
            packed_stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];

        std::array<int, kSpeculativeBatchMaxOutputTokens> output_tokens_int{};
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        output_tokens_int.fill(-1);
        meta.fill(0);
        summarize_greedy_speculative_verify_batch(
            static_cast<int>(resolved_draft_tokens[0]),
            verifier_tokens.data(),
            packed_draft_tokens.data(),
            compare_rows,
            packed_stop_tokens.data(),
            stop_token_count,
            output_tokens_int.data(),
            meta.data());

        std::array<int32_t, kSpeculativeBatchMaxOutputTokens> output_tokens{};
        output_tokens.fill(-1);
        for (size_t i = 0; i < output_tokens.size(); ++i)
            output_tokens[i] = static_cast<int32_t>(output_tokens_int[i]);

        if (!fillRankSpeculativeVerifyOutcomeFromMeta(output_tokens, meta, out))
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_all_position_greedy_compact_outcomes",
            static_cast<double>(draft_token_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"compare_rows", std::to_string(compare_rows)},
             {"implementation", "cross_shard_argmax"}});
        return true;
    }

    bool RankOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out_handle);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out_handle);
        }

        if (!out_handle)
            return false;
        *out_handle = DeviceSpeculativeOutcomeHandle{};
        rank_compact_outcome_valid_ = false;
        rank_compact_outcome_published_ = false;
        rank_resident_child_logical_state_handles_.clear();

        std::vector<int32_t> resolved_draft_tokens;
        if (!resolveRankGreedyOutcomeTokensForLocalTP(
                draft_tokens,
                draft_token_count,
                &resolved_draft_tokens))
        {
            return false;
        }

        DeviceSpeculativeVerifyBatchOutcome outcome;
        if (!verifyGreedyAllPositionBatchOutcomeOnDevice(
                resolved_draft_tokens.data(),
                draft_token_count,
                stop_tokens,
                stop_token_count,
                &outcome))
        {
            return false;
        }

        /*
         * Stage the already-reduced SamplingMath outcome into rank-owned
         * compact arrays.  The handle points at these arrays so publication and
         * response materialization consume one identical summary.  No child
         * runner is re-entered here, and no verifier row is replayed.
         */
        using namespace sampling_math;
        rank_compact_output_tokens_ = outcome.output_tokens;
        rank_compact_output_meta_.fill(0);
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::Greedy;
        rank_compact_output_meta_[kSpecBatchMetaOk] = outcome.ok ? 1 : 0;
        rank_compact_output_meta_[kSpecBatchMetaOutputCount] =
            outcome.output_token_count;
        rank_compact_output_meta_[kSpecBatchMetaAcceptedSpeculativePrefix] =
            outcome.accepted_speculative_prefix;
        rank_compact_output_meta_
            [kSpecBatchMetaTargetVerifierStateCommitCount] =
                outcome.target_verifier_state_commit_count;
        rank_compact_output_meta_[kSpecBatchMetaReadyToken] =
            outcome.ready_token;
        rank_compact_output_meta_[kSpecBatchMetaRejectedVerifiedToken] =
            outcome.rejected_verified_token;
        rank_compact_output_meta_[kSpecBatchMetaStoppedOnOutput] =
            outcome.stopped_on_output ? 1 : 0;
        rank_compact_output_meta_[kSpecBatchMetaAllSpeculativeAccepted] =
            outcome.all_speculative_accepted ? 1 : 0;
        rank_compact_output_meta_[kSpecBatchMetaConsumedVerifierRows] =
            outcome.consumed_verifier_rows;
        rank_compact_output_meta_[kSpecBatchMetaSampledTerminal] =
            outcome.sampled_terminal ? 1 : 0;

        rank_compact_last_draft_tokens_.assign(
            resolved_draft_tokens.begin(),
            resolved_draft_tokens.end());
        rank_compact_last_stop_tokens_.clear();
        if (stop_token_count > 0)
        {
            rank_compact_last_stop_tokens_.assign(
                stop_tokens,
                stop_tokens + stop_token_count);
        }

        out_handle->output_tokens_device = rank_compact_output_tokens_.data();
        out_handle->meta_device = rank_compact_output_meta_.data();
        out_handle->request_count = 1;
        out_handle->output_token_stride = kSpeculativeBatchMaxOutputTokens;
        out_handle->meta_stride = kSpeculativeBatchMetaCount;
        out_handle->device = primaryDeviceId();
        out_handle->stream = &rank_compact_outcome_stream_token_;
        out_handle->response_ready_event =
            std::shared_ptr<void>(
                &rank_compact_outcome_ready_event_token_,
                [](void *) {});

        rank_compact_outcome_valid_ = out_handle->valid();
        if (rank_compact_outcome_valid_)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_all_position_greedy_resident_compatible_outcomes",
                static_cast<double>(draft_token_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"implementation", "rank_owned_compact_sampling_math"}});
        }
        return rank_compact_outcome_valid_;
    }

    bool RankOrchestrator::buildRankStochasticDistributionFromLocalTP(
        DeviceLogitsSource source,
        int row,
        DeviceDistributionBuffer buffer,
        int slot,
        const SamplingParams &params,
        int vocab_size)
    {
        if (buffer != DeviceDistributionBuffer::Target ||
            device_runners_.size() < 2 ||
            row < 0 ||
            slot < 0 ||
            slot >= kRankStochasticMaxSlots ||
            params.top_k <= 0 ||
            params.top_k > sampling_math::kMaxTopK ||
            vocab_size <= 0)
        {
            return false;
        }

        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return false;

            LogitsLocalInfo info;
            switch (source)
            {
            case DeviceLogitsSource::Main:
                if (!runner->hasLogitsLocal())
                    return false;
                info = runner->consumeLogitsLocalInfoForSampling();
                break;
            case DeviceLogitsSource::MTP:
                if (!runner->hasMTPLogitsLocal())
                    return false;
                info = runner->consumeMTPLogitsLocalInfoForSampling();
                break;
            case DeviceLogitsSource::AllPosition:
                if (!runner->hasAllPositionLogitsLocal())
                    return false;
                info = runner->consumeAllPositionLogitsLocalInfoForSampling();
                break;
            }
            if (!info || info.vocab_local == 0)
                return false;
            local_infos.push_back(info);
        }

        struct Candidate
        {
            float value = -std::numeric_limits<float>::infinity();
            int token = -1;
        };

        std::vector<Candidate> candidates;
        candidates.reserve(
            local_infos.size() * static_cast<size_t>(params.top_k));
        size_t implicit_vocab_offset = 0;
        const bool use_explicit_vocab_offsets =
            std::any_of(
                local_infos.begin(),
                local_infos.end(),
                [](const LogitsLocalInfo &info)
                {
                    return info.vocab_offset != 0;
                });

        for (const LogitsLocalInfo &info : local_infos)
        {
            if (!info.tensor || info.vocab_local == 0)
                return false;

            const auto &shape = info.tensor->shape();
            const size_t rows = shape.size() >= 2 ? shape[0] : 1;
            const size_t cols = info.vocab_local;
            const size_t row_stride =
                info.row_stride > 0 ? info.row_stride : info.vocab_local;
            if (cols == 0 ||
                row_stride < cols ||
                static_cast<size_t>(row) >= rows ||
                cols > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return false;
            }

            const int local_k =
                std::min(params.top_k, static_cast<int>(cols));
            if (local_k <= 0)
                return false;

            std::vector<float> top_values(static_cast<size_t>(local_k));
            std::vector<int> top_indices(static_cast<size_t>(local_k), -1);
            int selected = 0;

            if (info.device.has_value() && info.device->is_gpu())
            {
                if (!info.gpu_ptr || !info.stream)
                    return false;
                IBackend *backend = getBackendFor(*info.device);
                if (!backend)
                    return false;

                const float *row_ptr =
                    static_cast<const float *>(info.gpu_ptr) +
                    static_cast<size_t>(row) * row_stride;
                if (!backend->topKF32(
                        row_ptr,
                        static_cast<int>(cols),
                        local_k,
                        info.device->gpu_ordinal(),
                        top_values.data(),
                        top_indices.data(),
                        info.stream))
                {
                    return false;
                }
                selected = local_k;
            }
            else
            {
                const float *data = info.tensor->fp32_data();
                if (!data)
                    return false;
                const float *row_ptr =
                    data + static_cast<size_t>(row) * row_stride;
                selected = cpu_sampling::select_topk(
                    row_ptr,
                    static_cast<int>(cols),
                    local_k,
                    top_values.data(),
                    top_indices.data());
            }

            const size_t vocab_offset =
                use_explicit_vocab_offsets
                    ? info.vocab_offset
                    : implicit_vocab_offset;
            for (int i = 0; i < selected; ++i)
            {
                const int local_index = top_indices[static_cast<size_t>(i)];
                if (local_index < 0)
                    continue;
                const int token =
                    static_cast<int>(vocab_offset) + local_index;
                if (token < 0 || token >= vocab_size)
                    continue;
                candidates.push_back(
                    Candidate{top_values[static_cast<size_t>(i)], token});
            }
            implicit_vocab_offset += cols;
        }

        if (candidates.empty())
            return false;

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate &a, const Candidate &b)
            {
                if (a.value != b.value)
                    return a.value > b.value;
                return a.token < b.token;
            });
        const int selected =
            std::min(params.top_k, static_cast<int>(candidates.size()));
        if (selected <= 0)
            return false;

        std::vector<float> sorted_logits(static_cast<size_t>(selected));
        std::vector<int> sorted_ids(static_cast<size_t>(selected));
        std::vector<float> scratch(static_cast<size_t>(selected), 0.0f);
        std::vector<int> out_ids(static_cast<size_t>(selected), -1);
        std::vector<float> out_probs(static_cast<size_t>(selected), 0.0f);
        for (int i = 0; i < selected; ++i)
        {
            sorted_logits[static_cast<size_t>(i)] =
                candidates[static_cast<size_t>(i)].value;
            sorted_ids[static_cast<size_t>(i)] =
                candidates[static_cast<size_t>(i)].token;
        }

        sampling_math::build_topk_topp_distribution_from_sorted(
            sorted_logits.data(),
            sorted_ids.data(),
            selected,
            params.top_p,
            params.temperature,
            out_ids.data(),
            out_probs.data(),
            scratch.data());

        const size_t slot_offset =
            static_cast<size_t>(slot) *
            static_cast<size_t>(sampling_math::kMaxTopK);
        for (int i = 0; i < sampling_math::kMaxTopK; ++i)
        {
            rank_stochastic_target_token_ids_[slot_offset +
                                              static_cast<size_t>(i)] = -1;
            rank_stochastic_target_probs_[slot_offset +
                                          static_cast<size_t>(i)] = 0.0f;
        }
        for (int i = 0; i < selected; ++i)
        {
            rank_stochastic_target_token_ids_[slot_offset +
                                              static_cast<size_t>(i)] =
                out_ids[static_cast<size_t>(i)];
            rank_stochastic_target_probs_[slot_offset +
                                          static_cast<size_t>(i)] =
                out_probs[static_cast<size_t>(i)];
        }
        rank_stochastic_target_top_k_[static_cast<size_t>(slot)] = selected;
        rank_stochastic_target_sample_tokens_[static_cast<size_t>(slot)] = -1;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_compact_distribution_builds",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(slot)},
             {"row", std::to_string(row)},
             {"top_k", std::to_string(params.top_k)},
             {"implementation", "cross_shard_topk_candidate_merge"}});
        return true;
    }

    bool RankOrchestrator::buildRankStochasticTargetDistributionsFromLocalTPRows(
        DeviceLogitsSource source,
        int first_row,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        if (source != DeviceLogitsSource::AllPosition ||
            first_row < 0 ||
            first_slot < 0 ||
            row_count <= 0 ||
            first_slot + row_count > kRankStochasticMaxSlots ||
            params.top_k <= 0 ||
            params.top_k > sampling_math::kMaxTopK ||
            vocab_size <= 0)
        {
            return false;
        }

        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasAllPositionLogitsLocal())
                return false;
            LogitsLocalInfo info =
                runner->consumeAllPositionLogitsLocalInfoForSampling();
            if (!info || info.vocab_local == 0)
                return false;
            local_infos.push_back(info);
        }

        auto build_row_from_consumed_infos =
            [&](int row, int slot) -> bool
        {
            struct Candidate
            {
                float value = -std::numeric_limits<float>::infinity();
                int token = -1;
            };

            std::vector<Candidate> candidates;
            candidates.reserve(
                local_infos.size() * static_cast<size_t>(params.top_k));
            size_t implicit_vocab_offset = 0;
            const bool use_explicit_vocab_offsets =
                std::any_of(
                    local_infos.begin(),
                    local_infos.end(),
                    [](const LogitsLocalInfo &info)
                    {
                        return info.vocab_offset != 0;
                    });

            for (const LogitsLocalInfo &info : local_infos)
            {
                const auto &shape = info.tensor->shape();
                const size_t rows = shape.size() >= 2 ? shape[0] : 1;
                const size_t cols = info.vocab_local;
                const size_t row_stride =
                    info.row_stride > 0 ? info.row_stride : info.vocab_local;
                if (cols == 0 ||
                    row_stride < cols ||
                    static_cast<size_t>(row) >= rows ||
                    cols > static_cast<size_t>(std::numeric_limits<int>::max()))
                {
                    return false;
                }

                const int local_k =
                    std::min(params.top_k, static_cast<int>(cols));
                std::vector<float> top_values(static_cast<size_t>(local_k));
                std::vector<int> top_indices(static_cast<size_t>(local_k), -1);
                int selected = 0;

                if (info.device.has_value() && info.device->is_gpu())
                {
                    if (!info.gpu_ptr || !info.stream)
                        return false;
                    IBackend *backend = getBackendFor(*info.device);
                    if (!backend)
                        return false;
                    const float *row_ptr =
                        static_cast<const float *>(info.gpu_ptr) +
                        static_cast<size_t>(row) * row_stride;
                    if (!backend->topKF32(
                            row_ptr,
                            static_cast<int>(cols),
                            local_k,
                            info.device->gpu_ordinal(),
                            top_values.data(),
                            top_indices.data(),
                            info.stream))
                    {
                        return false;
                    }
                    selected = local_k;
                }
                else
                {
                    const float *data = info.tensor->fp32_data();
                    if (!data)
                        return false;
                    const float *row_ptr =
                        data + static_cast<size_t>(row) * row_stride;
                    selected = cpu_sampling::select_topk(
                        row_ptr,
                        static_cast<int>(cols),
                        local_k,
                        top_values.data(),
                        top_indices.data());
                }

                const size_t vocab_offset =
                    use_explicit_vocab_offsets
                        ? info.vocab_offset
                        : implicit_vocab_offset;
                for (int i = 0; i < selected; ++i)
                {
                    const int local_index =
                        top_indices[static_cast<size_t>(i)];
                    if (local_index < 0)
                        continue;
                    const int token =
                        static_cast<int>(vocab_offset) + local_index;
                    if (token >= 0 && token < vocab_size)
                    {
                        candidates.push_back(
                            Candidate{
                                top_values[static_cast<size_t>(i)],
                                token});
                    }
                }
                implicit_vocab_offset += cols;
            }

            if (candidates.empty())
                return false;
            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const Candidate &a, const Candidate &b)
                {
                    if (a.value != b.value)
                        return a.value > b.value;
                    return a.token < b.token;
                });

            const int selected =
                std::min(params.top_k, static_cast<int>(candidates.size()));
            std::vector<float> sorted_logits(static_cast<size_t>(selected));
            std::vector<int> sorted_ids(static_cast<size_t>(selected));
            std::vector<float> scratch(static_cast<size_t>(selected), 0.0f);
            std::vector<int> out_ids(static_cast<size_t>(selected), -1);
            std::vector<float> out_probs(static_cast<size_t>(selected), 0.0f);
            for (int i = 0; i < selected; ++i)
            {
                sorted_logits[static_cast<size_t>(i)] =
                    candidates[static_cast<size_t>(i)].value;
                sorted_ids[static_cast<size_t>(i)] =
                    candidates[static_cast<size_t>(i)].token;
            }
            sampling_math::build_topk_topp_distribution_from_sorted(
                sorted_logits.data(),
                sorted_ids.data(),
                selected,
                params.top_p,
                params.temperature,
                out_ids.data(),
                out_probs.data(),
                scratch.data());

            const size_t slot_offset =
                static_cast<size_t>(slot) *
                static_cast<size_t>(sampling_math::kMaxTopK);
            for (int i = 0; i < sampling_math::kMaxTopK; ++i)
            {
                rank_stochastic_target_token_ids_[slot_offset +
                                                  static_cast<size_t>(i)] = -1;
                rank_stochastic_target_probs_[slot_offset +
                                              static_cast<size_t>(i)] = 0.0f;
            }
            for (int i = 0; i < selected; ++i)
            {
                rank_stochastic_target_token_ids_[slot_offset +
                                                  static_cast<size_t>(i)] =
                    out_ids[static_cast<size_t>(i)];
                rank_stochastic_target_probs_[slot_offset +
                                              static_cast<size_t>(i)] =
                    out_probs[static_cast<size_t>(i)];
            }
            rank_stochastic_target_top_k_[static_cast<size_t>(slot)] =
                selected;
            rank_stochastic_target_sample_tokens_[static_cast<size_t>(slot)] =
                -1;
            return true;
        };

        for (int row = 0; row < row_count; ++row)
        {
            if (!build_row_from_consumed_infos(
                    first_row + row,
                    first_slot + row))
            {
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_compact_distribution_batch_rows",
            static_cast<double>(row_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"first_row", std::to_string(first_row)},
             {"first_slot", std::to_string(first_slot)},
             {"implementation", "cross_shard_topk_candidate_merge"}});
        return true;
    }

    bool RankOrchestrator::sampleRankStochasticDistribution(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold,
        int32_t *out_token)
    {
        if (buffer != DeviceDistributionBuffer::Target ||
            slot < 0 ||
            slot >= kRankStochasticMaxSlots)
        {
            return false;
        }

        const int active_top_k =
            rank_stochastic_target_top_k_[static_cast<size_t>(slot)];
        if (active_top_k <= 0 ||
            active_top_k > sampling_math::kMaxTopK)
        {
            return false;
        }

        const size_t slot_offset =
            static_cast<size_t>(slot) *
            static_cast<size_t>(sampling_math::kMaxTopK);
        float selected_probability = 0.0f;
        const int token =
            sampling_math::sample_distribution_with_threshold_and_probability(
                rank_stochastic_target_token_ids_.data() + slot_offset,
                rank_stochastic_target_probs_.data() + slot_offset,
                active_top_k,
                threshold,
                &selected_probability);
        if (token < 0)
            return false;

        rank_stochastic_target_sample_tokens_[static_cast<size_t>(slot)] =
            static_cast<int32_t>(token);
        if (!stageRankStochasticTargetTokenForLocalTP(
                slot,
                static_cast<int32_t>(token)))
        {
            return false;
        }
        if (out_token)
            *out_token = static_cast<int32_t>(token);

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_compact_distribution_samples",
            1.0,
            "decode",
            "rank",
            {{"slot", std::to_string(slot)},
             {"top_k", std::to_string(active_top_k)},
             {"token", std::to_string(token)},
             {"probability", std::to_string(selected_probability)}});
        return true;
    }

    bool RankOrchestrator::sampleRankGreedyMTPLogitsToLocalTPDraftSlot(
        int draft_sample_slot,
        int32_t *out_token)
    {
        if (device_runners_.size() < 2 ||
            draft_sample_slot < 0 ||
            draft_sample_slot >= kRankStochasticMaxSlots)
        {
            return false;
        }

        /*
         * Each child owns only its local MTP-head shard.  The rank samples the
         * global greedy token by reducing child-local argmax metadata, then
         * stages that one compact token into every child draft slot.  The
         * verifier can therefore consume child device slots exactly as it does
         * for stochastic MTP, without gathering full logits or replaying rows.
         */
        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasMTPLogitsLocal())
                return false;
            LogitsLocalInfo info =
                runner->consumeMTPLogitsLocalInfoForSampling();
            if (!info || info.vocab_local == 0)
                return false;
            local_infos.push_back(info);
        }

        const int token =
            DeviceSampler::sampleGreedyFromLocalInfos(local_infos, /*row=*/0);
        if (token < 0)
            return false;

        const int32_t staged_token = static_cast<int32_t>(token);
        if (!stageStochasticDraftTokensForDeviceVerification(
                &staged_token,
                /*draft_token_count=*/1,
                draft_sample_slot))
        {
            return false;
        }
        if (out_token)
            *out_token = staged_token;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_greedy_mtp_draft_slot_samples",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(draft_sample_slot)},
             {"token", std::to_string(token)},
             {"implementation", "cross_shard_argmax_device_partial"}});
        return true;
    }

    bool RankOrchestrator::sampleRankGreedyMainLogitsToLocalTPTargetSlot(
        int target_sample_slot,
        int32_t *out_token)
    {
        if (device_runners_.size() < 2 ||
            target_sample_slot < 0 ||
            target_sample_slot >= kRankStochasticMaxSlots)
        {
            return false;
        }

        /*
         * LocalTP main logits are vocab-sharded: each child owns a compact row
         * for its LM-head slice.  DeviceSampler performs the economical
         * per-shard argmax and returns only the winning global token.  That small
         * rank decision is then staged into every child target slot so downstream
         * verifier graph input remains device-resident and participant-symmetric.
         */
        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasLogitsLocal())
                return false;
            LogitsLocalInfo info =
                runner->consumeLogitsLocalInfoForSampling();
            if (!info || info.vocab_local == 0)
                return false;
            local_infos.push_back(info);
        }

        const int token =
            DeviceSampler::sampleGreedyFromLocalInfos(local_infos, /*row=*/0);
        if (token < 0)
            return false;

        if (!stageRankStochasticTargetTokenForLocalTP(
                target_sample_slot,
                static_cast<int32_t>(token)))
        {
            return false;
        }

        const size_t slot_offset =
            static_cast<size_t>(target_sample_slot) *
            static_cast<size_t>(sampling_math::kMaxTopK);
        for (int i = 0; i < sampling_math::kMaxTopK; ++i)
        {
            rank_stochastic_target_token_ids_[slot_offset +
                                              static_cast<size_t>(i)] = -1;
            rank_stochastic_target_probs_[slot_offset +
                                          static_cast<size_t>(i)] = 0.0f;
        }
        rank_stochastic_target_token_ids_[slot_offset] = token;
        rank_stochastic_target_probs_[slot_offset] = 1.0f;
        rank_stochastic_target_top_k_[static_cast<size_t>(target_sample_slot)] = 1;
        rank_stochastic_target_sample_tokens_[
            static_cast<size_t>(target_sample_slot)] =
            static_cast<int32_t>(token);
        if (out_token)
            *out_token = static_cast<int32_t>(token);

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_greedy_main_target_slot_samples",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(target_sample_slot)},
             {"token", std::to_string(token)},
             {"implementation", "cross_shard_argmax_device_partial"}});
        return true;
    }

    bool RankOrchestrator::stageRankStochasticTargetTokenForLocalTP(
        int slot,
        int32_t token)
    {
        if (device_runners_.size() < 2 ||
            slot < 0 ||
            slot >= kRankStochasticMaxSlots ||
            token < 0)
        {
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i] ||
                !device_runners_[i]->stageStochasticTargetTokenForDeviceSampling(
                    token,
                    slot))
            {
                LOG_ERROR("[RankOrchestrator] LocalTP child "
                          << i
                          << " could not stage rank stochastic target token slot="
                          << slot);
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_target_token_device_stages",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(slot)},
             {"token", std::to_string(token)}});
        return true;
    }

    bool RankOrchestrator::stageRankStochasticDraftTokensForLocalTP(
        const int32_t *tokens,
        int token_count,
        int first_slot)
    {
        if (device_runners_.size() < 2 ||
            !tokens ||
            token_count <= 0 ||
            first_slot < 0 ||
            first_slot + token_count > kRankStochasticMaxSlots)
        {
            return false;
        }

        for (int i = 0; i < token_count; ++i)
        {
            if (tokens[i] < 0)
                return false;
        }

        for (size_t child = 0; child < device_runners_.size(); ++child)
        {
            if (!device_runners_[child] ||
                !device_runners_[child]->stageStochasticDraftTokensForDeviceVerification(
                    tokens,
                    token_count,
                    first_slot))
            {
                LOG_ERROR("[RankOrchestrator] LocalTP child "
                          << child
                          << " could not stage rank stochastic draft tokens first_slot="
                          << first_slot
                          << " count=" << token_count);
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_draft_token_device_stages",
            static_cast<double>(token_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"first_slot", std::to_string(first_slot)}});
        return true;
    }

    const void *RankOrchestrator::prepareRankVerifierTokenSlotsForLocalTP(
        bool first_token_from_device,
        int32_t first_token,
        int first_target_sample_slot,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (device_runners_.size() < 2 ||
            total_verifier_input_tokens <= 0 ||
            draft_token_count < 0 ||
            draft_token_count + 1 != total_verifier_input_tokens ||
            first_draft_slot < 0 ||
            first_draft_slot + draft_token_count > kRankStochasticMaxSlots)
        {
            return nullptr;
        }
        if (first_token_from_device &&
            (first_target_sample_slot < 0 ||
             first_target_sample_slot >= kRankStochasticMaxSlots))
        {
            return nullptr;
        }
        if (!first_token_from_device && first_token < 0)
            return nullptr;

        rank_mtp_verifier_child_token_inputs_.assign(
            device_runners_.size(),
            nullptr);
        rank_mtp_verifier_child_token_count_ =
            total_verifier_input_tokens;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                return nullptr;

            rank_mtp_verifier_child_token_inputs_[i] =
                first_token_from_device
                    ? device_runners_[i]->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
                          first_target_sample_slot,
                          first_draft_slot,
                          draft_token_count,
                          total_verifier_input_tokens)
                    : device_runners_[i]->prepareMTPVerifierInputTokensOnDevice(
                          first_token,
                          first_draft_slot,
                          draft_token_count,
                          total_verifier_input_tokens);
            if (!rank_mtp_verifier_child_token_inputs_[i])
            {
                rank_mtp_verifier_child_token_inputs_.clear();
                rank_mtp_verifier_child_token_count_ = 0;
                return nullptr;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_token_rows_prepared_from_device_slots",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"tokens", std::to_string(total_verifier_input_tokens)},
             {"draft_tokens", std::to_string(draft_token_count)},
             {"first_token_from_device",
              first_token_from_device ? "true" : "false"}});
        return rank_mtp_verifier_child_token_inputs_.data();
    }

    const void *RankOrchestrator::stageRankVerifierTokenRowForLocalTP(
        const int32_t *tokens,
        int total_verifier_input_tokens,
        int draft_token_count)
    {
        if (device_runners_.size() < 2 ||
            !tokens ||
            total_verifier_input_tokens <= 0 ||
            draft_token_count < 0 ||
            draft_token_count >= total_verifier_input_tokens)
        {
            return nullptr;
        }

        rank_mtp_verifier_child_token_inputs_.assign(
            device_runners_.size(),
            nullptr);
        rank_mtp_verifier_child_token_count_ =
            total_verifier_input_tokens;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                return nullptr;
            rank_mtp_verifier_child_token_inputs_[i] =
                device_runners_[i]->prepareMTPVerifierInputTokensOnDeviceFromHostRow(
                    tokens,
                    total_verifier_input_tokens,
                    draft_token_count);
            if (!rank_mtp_verifier_child_token_inputs_[i])
            {
                rank_mtp_verifier_child_token_inputs_.clear();
                rank_mtp_verifier_child_token_count_ = 0;
                return nullptr;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_token_rows_staged_from_compact_samples",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"tokens", std::to_string(total_verifier_input_tokens)},
             {"draft_tokens", std::to_string(draft_token_count)}});
        return rank_mtp_verifier_child_token_inputs_.data();
    }

    int32_t RankOrchestrator::rankStochasticDraftSampleToken(int slot) const
    {
        if (slot < 0 || slot >= kRankStochasticMaxSlots)
            return -1;
        return rank_stochastic_draft_sample_tokens_[static_cast<size_t>(slot)];
    }

    int32_t RankOrchestrator::rankStochasticTargetSampleToken(int slot) const
    {
        if (slot < 0 || slot >= kRankStochasticMaxSlots)
            return -1;
        return rank_stochastic_target_sample_tokens_[static_cast<size_t>(slot)];
    }

    bool RankOrchestrator::supportsGreedyAllPositionBatchOutcomeOnDevice() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsGreedyAllPositionBatchOutcomeOnDevice();
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->supportsGreedyAllPositionBatchOutcomeOnDevice();
        }

        /*
         * Multi-child LocalTP advertises this only when the full compact
         * lifecycle is available: sharded row argmax reduction in this rank,
         * compact SamplingMath summary storage, and grouped decode-equivalent
         * accepted-state publication on every participant.
         */
        return supportsDeviceResidentMTPSpecStatePublication();
    }

    bool RankOrchestrator::supportsDeviceStochasticMTPVerification() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsDeviceStochasticMTPVerification();
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->supportsDeviceStochasticMTPVerification();
        }

        /*
         * In multi-child LocalTP, no child owns a full-vocab distribution.
         * The rank itself owns stochastic verification by reducing each
         * child-local top-k candidate set into one compact table and then
         * publishing the compact outcome through grouped child publication.
         */
        return device_runners_.size() >= 2 &&
               supportsDeviceResidentMTPSpecStatePublication();
    }

    bool RankOrchestrator::buildStochasticDistributionOnDevice(
        DeviceLogitsSource source,
        int row,
        DeviceDistributionBuffer buffer,
        int slot,
        const SamplingParams &params,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->buildStochasticDistributionOnDevice(
                source,
                row,
                buffer,
                slot,
                params,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->buildStochasticDistributionOnDevice(
                source,
                row,
                buffer,
                slot,
                params,
                vocab_size);
        }
        return buildRankStochasticDistributionFromLocalTP(
            source,
            row,
            buffer,
            slot,
            params,
            vocab_size);
    }

    bool RankOrchestrator::buildStochasticDistributionsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->buildStochasticDistributionsOnDevice(
                source,
                first_row,
                buffer,
                first_slot,
                row_count,
                params,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->buildStochasticDistributionsOnDevice(
                source,
                first_row,
                buffer,
                first_slot,
                row_count,
                params,
                vocab_size);
        }
        if (buffer != DeviceDistributionBuffer::Target)
            return false;
        return buildRankStochasticTargetDistributionsFromLocalTPRows(
            source,
            first_row,
            first_slot,
            row_count,
            params,
            vocab_size);
    }

    bool RankOrchestrator::buildStochasticProcessedLogitRowsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->buildStochasticProcessedLogitRowsOnDevice(
                source,
                first_row,
                buffer,
                first_slot,
                row_count,
                params,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->buildStochasticProcessedLogitRowsOnDevice(
                source,
                first_row,
                buffer,
                first_slot,
                row_count,
                params,
                vocab_size);
        }
        return false;
    }

    int RankOrchestrator::sampleStochasticDraftProposalOnDevice(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleStochasticDraftProposalOnDevice(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleStochasticDraftProposalOnDevice(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold);
        }

        (void)params;
        (void)vocab_size;
        (void)threshold;
        if (source != DeviceLogitsSource::MTP ||
            row < 0 ||
            slot < 0 ||
            slot >= kRankStochasticMaxSlots ||
            device_runners_.size() < 2)
        {
            return -1;
        }

        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasMTPLogitsLocal())
                return -1;
            LogitsLocalInfo info =
                runner->consumeMTPLogitsLocalInfoForSampling();
            if (!info)
                return -1;
            local_infos.push_back(info);
        }

        const int token =
            DeviceSampler::sampleGreedyFromLocalInfos(local_infos, row);
        if (token < 0)
            return -1;
        rank_stochastic_draft_sample_tokens_[static_cast<size_t>(slot)] =
            static_cast<int32_t>(token);

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_draft_greedy_proposals",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(slot)},
             {"implementation", "cross_shard_argmax"}});
        return token;
    }

    bool RankOrchestrator::sampleStochasticDraftProposalOnDeviceDeferred(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleStochasticDraftProposalOnDeviceDeferred(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleStochasticDraftProposalOnDeviceDeferred(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold);
        }
        const int token = sampleStochasticDraftProposalOnDevice(
            source,
            row,
            slot,
            params,
            vocab_size,
            threshold);
        if (token < 0)
            return false;
        const int32_t staged_token = static_cast<int32_t>(token);
        return stageRankStochasticDraftTokensForLocalTP(
            &staged_token,
            /*token_count=*/1,
            slot);
    }

    int RankOrchestrator::sampleStochasticDistributionOnDevice(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleStochasticDistributionOnDevice(
                buffer,
                slot,
                threshold);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleStochasticDistributionOnDevice(
                buffer,
                slot,
                threshold);
        }

        int32_t token = -1;
        if (!sampleRankStochasticDistribution(
                buffer,
                slot,
                threshold,
                &token))
        {
            return -1;
        }
        return token;
    }

    bool RankOrchestrator::sampleStochasticDistributionOnDeviceDeferred(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleStochasticDistributionOnDeviceDeferred(
                buffer,
                slot,
                threshold);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleStochasticDistributionOnDeviceDeferred(
                buffer,
                slot,
                threshold);
        }
        return sampleRankStochasticDistribution(
            buffer,
            slot,
            threshold,
            /*out_token=*/nullptr);
    }

    bool RankOrchestrator::stageStochasticTargetTokenForDeviceSampling(
        int32_t target_token,
        int target_sample_slot)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->stageStochasticTargetTokenForDeviceSampling(
                target_token,
                target_sample_slot);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->stageStochasticTargetTokenForDeviceSampling(
                target_token,
                target_sample_slot);
        }
        if (!stageRankStochasticTargetTokenForLocalTP(
                target_sample_slot,
                target_token))
        {
            return false;
        }
        rank_stochastic_target_sample_tokens_[
            static_cast<size_t>(target_sample_slot)] = target_token;
        return true;
    }

    bool RankOrchestrator::stageStochasticDraftTokensForDeviceVerification(
        const int32_t *draft_tokens,
        int draft_token_count,
        int first_draft_slot)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * Device-side stochastic verification for LocalPP is final-stage
             * owned: the final stage owns verifier logits/distributions and the
             * draft-token slots consumed by the verifier summary kernel.
             */
            return pp_sidecar->stageStochasticDraftTokensForDeviceVerification(
                draft_tokens,
                draft_token_count,
                first_draft_slot);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->stageStochasticDraftTokensForDeviceVerification(
                draft_tokens,
                draft_token_count,
                first_draft_slot);
        }
        if (device_runners_.size() < 2 ||
            !draft_tokens ||
            draft_token_count <= 0 ||
            first_draft_slot < 0 ||
            first_draft_slot + draft_token_count > kRankStochasticMaxSlots)
        {
            return false;
        }

        if (first_draft_slot == 0)
            rank_stochastic_draft_sample_tokens_.fill(-1);
        if (rank_stochastic_staged_draft_tokens_.size() <
            static_cast<size_t>(first_draft_slot + draft_token_count))
        {
            rank_stochastic_staged_draft_tokens_.resize(
                static_cast<size_t>(first_draft_slot + draft_token_count),
                -1);
        }
        for (int i = 0; i < draft_token_count; ++i)
        {
            const int32_t token = draft_tokens[i];
            if (token < 0)
                return false;
            rank_stochastic_draft_sample_tokens_[
                static_cast<size_t>(first_draft_slot + i)] = token;
            rank_stochastic_staged_draft_tokens_[
                static_cast<size_t>(first_draft_slot + i)] = token;
        }
        if (!stageRankStochasticDraftTokensForLocalTP(
                draft_tokens,
                draft_token_count,
                first_draft_slot))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_draft_token_stages",
            static_cast<double>(draft_token_count),
            "decode",
            "rank",
            {{"first_slot", std::to_string(first_draft_slot)},
             {"participants", std::to_string(device_runners_.size())}});
        return true;
    }

    const void *RankOrchestrator::prepareMTPVerifierInputTokensOnDevice(
        int32_t first_token,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * LocalPP verifier replay runs on the logits-owning final stage.
             * The prepared verifier token row is consumed by that same final
             * stage, so it follows final-stage ownership rather than the
             * pipeline-head token input ownership used by normal prefill/decode.
             */
            return pp_sidecar->prepareMTPVerifierInputTokensOnDevice(
                first_token,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->prepareMTPVerifierInputTokensOnDevice(
                first_token,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }

        if (device_runners_.size() < 2 ||
            total_verifier_input_tokens <= 0 ||
            draft_token_count < 0 ||
            draft_token_count + 1 > total_verifier_input_tokens)
            return nullptr;

        for (int i = 0; i < draft_token_count; ++i)
        {
            const int32_t token =
                rankStochasticDraftSampleToken(first_draft_slot + i);
            if (token < 0)
                return nullptr;
        }
        return prepareRankVerifierTokenSlotsForLocalTP(
            /*first_token_from_device=*/false,
            first_token,
            /*first_target_sample_slot=*/-1,
            first_draft_slot,
            draft_token_count,
            total_verifier_input_tokens);
    }

    const void *RankOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromHostRow(
        const int32_t *verifier_tokens,
        int total_verifier_input_tokens,
        int draft_token_count)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->prepareMTPVerifierInputTokensOnDeviceFromHostRow(
                verifier_tokens,
                total_verifier_input_tokens,
                draft_token_count);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->prepareMTPVerifierInputTokensOnDeviceFromHostRow(
                verifier_tokens,
                total_verifier_input_tokens,
                draft_token_count);
        }

        if (device_runners_.size() < 2 ||
            !verifier_tokens ||
            total_verifier_input_tokens <= 0)
        {
            return nullptr;
        }

        return stageRankVerifierTokenRowForLocalTP(
            verifier_tokens,
            total_verifier_input_tokens,
            draft_token_count);
    }

    const void *RankOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
        int first_target_sample_slot,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * The first target sample slot and draft sample slots are produced
             * by the final-stage stochastic verifier hooks, so compose the
             * verifier token row on that same stage to avoid cross-stage device
             * pointer aliasing.
             */
            return pp_sidecar->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
                first_target_sample_slot,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
                first_target_sample_slot,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }

        if (device_runners_.size() < 2 ||
            total_verifier_input_tokens <= 0 ||
            draft_token_count < 0 ||
            draft_token_count + 1 > total_verifier_input_tokens)
            return nullptr;

        const int32_t first_token =
            rankStochasticTargetSampleToken(first_target_sample_slot);
        if (first_token < 0)
            return nullptr;

        for (int i = 0; i < draft_token_count; ++i)
        {
            const int32_t token =
                rankStochasticDraftSampleToken(first_draft_slot + i);
            if (token < 0)
                return nullptr;
        }
        return prepareRankVerifierTokenSlotsForLocalTP(
            /*first_token_from_device=*/true,
            /*first_token=*/-1,
            first_target_sample_slot,
            first_draft_slot,
            draft_token_count,
            total_verifier_input_tokens);
    }

    bool RankOrchestrator::verifyStochasticDistributionsOnDevice(
        int target_slot,
        int draft_slot,
        int draft_token,
        float accept_threshold,
        float residual_threshold,
        DeviceSpeculativeVerifyResult *out)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyStochasticDistributionsOnDevice(
                target_slot,
                draft_slot,
                draft_token,
                accept_threshold,
                residual_threshold,
                out);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsOnDevice(
                target_slot,
                draft_slot,
                draft_token,
                accept_threshold,
                residual_threshold,
                out);
        }
        return false;
    }

    bool RankOrchestrator::verifyStochasticDistributionsBatchOnDevice(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        DeviceSpeculativeVerifyResult *out)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyStochasticDistributionsBatchOnDevice(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                out);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsBatchOnDevice(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                out);
        }
        return false;
    }

    bool RankOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDevice(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int32_t first_token,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        bool use_vllm_probability_rejection)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyStochasticDistributionsBatchOutcomeOnDevice(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_token,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                out,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsBatchOutcomeOnDevice(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_token,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                out,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection);
        }

        DeviceSpeculativeOutcomeHandle handle;
        if (!verifyStochasticDistributionsBatchOutcomeOnDeviceResident(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_token,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                &handle,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection))
        {
            return false;
        }
        return copyDeviceSpeculativeOutcomesToHost(handle, out);
    }

    bool RankOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int first_target_sample_slot,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        bool use_vllm_probability_rejection)
    {
        if (finalPPSidecarRunner())
        {
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_target_sample_slot,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                out,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection);
        }

        DeviceSpeculativeOutcomeHandle handle;
        if (!verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_target_sample_slot,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                &handle,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection))
        {
            return false;
        }
        return copyDeviceSpeculativeOutcomesToHost(handle, out);
    }

    bool RankOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
        const DeviceStochasticBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        using namespace sampling_math;
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                out_handle);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                out_handle);
        }
        if (out_handle)
            *out_handle = DeviceSpeculativeOutcomeHandle{};
        rank_compact_outcome_valid_ = false;
        rank_compact_outcome_published_ = false;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::None;
        rank_resident_child_logical_state_handles_.clear();

        if (!requests ||
            request_count != 1 ||
            !out_handle ||
            device_runners_.size() < 2 ||
            !supportsDeviceStochasticMTPVerification())
        {
            return false;
        }

        const DeviceStochasticBatchOutcomeRequest &request = requests[0];
        if (request.row_count <= 0 ||
            request.row_count > kSpeculativeBatchMaxRows ||
            request.first_target_slot < 0 ||
            request.first_draft_slot < 0 ||
            request.first_target_slot + request.row_count >
                kRankStochasticMaxSlots ||
            request.first_draft_slot + request.row_count >
                kRankStochasticMaxSlots ||
            request.stop_token_count < 0 ||
            request.stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (request.bonus_target_slot >= kRankStochasticMaxSlots))
        {
            return false;
        }

        const bool derive_thresholds =
            request.derive_thresholds_from_seed &&
            request.use_vllm_probability_rejection &&
            request.inverse_sample_seed != 0;
        std::array<float, kSpeculativeBatchMaxRows> accept_thresholds{};
        std::array<float, kSpeculativeBatchMaxRows> residual_thresholds{};
        for (int row = 0; row < request.row_count; ++row)
        {
            if (derive_thresholds)
            {
                const int logical_position =
                    request.inverse_sample_first_logical_position + row;
                accept_thresholds[static_cast<size_t>(row)] =
                    mtp_spec_threshold_from_seed(
                        request.inverse_sample_seed,
                        logical_position,
                        1 /* MTPSpecStochasticDrawPurpose::Accept */);
                residual_thresholds[static_cast<size_t>(row)] =
                    mtp_spec_threshold_from_seed(
                        request.inverse_sample_seed,
                        logical_position,
                        2 /* MTPSpecStochasticDrawPurpose::Residual */);
            }
            else
            {
                accept_thresholds[static_cast<size_t>(row)] =
                    request.accept_thresholds[static_cast<size_t>(row)];
                residual_thresholds[static_cast<size_t>(row)] =
                    request.residual_thresholds[static_cast<size_t>(row)];
            }
        }

        const int32_t first_token =
            request.first_token_from_device
                ? rankStochasticTargetSampleToken(
                      request.first_target_sample_slot)
                : request.first_token;
        if (first_token < 0)
            return false;

        std::vector<int32_t> actual_draft_tokens;
        actual_draft_tokens.reserve(
            static_cast<size_t>(request.row_count + 1));
        actual_draft_tokens.push_back(first_token);

        std::array<int, kSpeculativeBatchMaxRows> row_tokens{};
        std::array<int, kSpeculativeBatchMaxRows> row_accepted{};
        row_tokens.fill(-1);
        row_accepted.fill(0);

        for (int row = 0; row < request.row_count; ++row)
        {
            const int draft_slot = request.first_draft_slot + row;
            const int32_t draft_token =
                request.use_device_draft_tokens
                    ? rankStochasticDraftSampleToken(draft_slot)
                    : request.draft_tokens[static_cast<size_t>(row)];
            if (draft_token < 0)
                return false;
            actual_draft_tokens.push_back(draft_token);

            const int target_slot = request.first_target_slot + row;
            const int target_top_k =
                rank_stochastic_target_top_k_[
                    static_cast<size_t>(target_slot)];
            if (target_top_k <= 0 ||
                target_top_k > kMaxTopK)
            {
                return false;
            }

            const size_t slot_offset =
                static_cast<size_t>(target_slot) *
                static_cast<size_t>(kMaxTopK);
            int out_token = -1;
            int out_accepted = 0;
            float accept_probability = 0.0f;
            float accept_threshold = 0.0f;
            if (request.use_vllm_probability_rejection)
            {
                speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
                    rank_stochastic_target_token_ids_.data() + slot_offset,
                    rank_stochastic_target_probs_.data() + slot_offset,
                    target_top_k,
                    vocab_size(),
                    draft_token,
                    accept_thresholds[static_cast<size_t>(row)],
                    request.inverse_sample_seed,
                    request.inverse_sample_first_logical_position + row,
                    &out_token,
                    &out_accepted,
                    &accept_probability,
                    &accept_threshold);
            }
            else
            {
                speculative_verify_with_thresholds_one_hot_draft(
                    rank_stochastic_target_token_ids_.data() + slot_offset,
                    rank_stochastic_target_probs_.data() + slot_offset,
                    target_top_k,
                    draft_token,
                    accept_thresholds[static_cast<size_t>(row)],
                    residual_thresholds[static_cast<size_t>(row)],
                    &out_token,
                    &out_accepted,
                    &accept_probability,
                    &accept_threshold);
            }
            (void)accept_probability;
            (void)accept_threshold;
            if (out_token < 0)
                return false;
            row_tokens[static_cast<size_t>(row)] = out_token;
            row_accepted[static_cast<size_t>(row)] = out_accepted;
        }

        int bonus_token = -1;
        int has_bonus = 0;
        if (request.bonus_target_slot >= 0)
        {
            int32_t sampled_bonus = -1;
            if (!sampleRankStochasticDistribution(
                    DeviceDistributionBuffer::Target,
                    request.bonus_target_slot,
                    request.bonus_threshold,
                    &sampled_bonus))
            {
                return false;
            }
            bonus_token = sampled_bonus;
            has_bonus = 1;
        }

        std::array<int, kSpeculativeBatchMaxOutputTokens> output_tokens_int{};
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens{};
        output_tokens_int.fill(-1);
        meta.fill(0);
        packed_stop_tokens.fill(-1);
        for (int i = 0; i < request.stop_token_count; ++i)
        {
            packed_stop_tokens[static_cast<size_t>(i)] =
                static_cast<int>(request.stop_tokens[static_cast<size_t>(i)]);
        }
        summarize_speculative_verify_batch(
            first_token,
            row_tokens.data(),
            row_accepted.data(),
            request.row_count,
            request.stop_token_count > 0 ? packed_stop_tokens.data() : nullptr,
            request.stop_token_count,
            bonus_token,
            has_bonus,
            output_tokens_int.data(),
            meta.data());
        if (meta[kSpecBatchMetaOk] == 0)
            return false;

        rank_compact_output_tokens_.fill(-1);
        for (size_t i = 0; i < rank_compact_output_tokens_.size(); ++i)
        {
            rank_compact_output_tokens_[i] =
                static_cast<int32_t>(output_tokens_int[i]);
        }
        rank_compact_output_meta_ = meta;
        rank_compact_last_draft_tokens_ = std::move(actual_draft_tokens);
        rank_compact_last_stop_tokens_.clear();
        for (int i = 0; i < request.stop_token_count; ++i)
        {
            rank_compact_last_stop_tokens_.push_back(
                request.stop_tokens[static_cast<size_t>(i)]);
        }

        out_handle->output_tokens_device = rank_compact_output_tokens_.data();
        out_handle->meta_device = rank_compact_output_meta_.data();
        out_handle->request_count = 1;
        out_handle->output_token_stride = kSpeculativeBatchMaxOutputTokens;
        out_handle->meta_stride = kSpeculativeBatchMetaCount;
        out_handle->device = primaryDeviceId();
        out_handle->stream = &rank_compact_outcome_stream_token_;
        out_handle->response_ready_event =
            std::shared_ptr<void>(
                &rank_compact_outcome_ready_event_token_,
                [](void *) {});

        rank_compact_outcome_kind_ = RankCompactOutcomeKind::Stochastic;
        rank_compact_outcome_valid_ = out_handle->valid();
        if (!rank_compact_outcome_valid_)
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_compact_resident_compatible_outcomes",
            static_cast<double>(request.row_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"implementation", "rank_owned_compact_stochastic_topk"}});
        return true;
    }

    bool RankOrchestrator::setComputeAllPositionLogits(bool enabled)
    {
        auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        bool all_success = true;
        for (auto &runner : participants)
        {
            if (!runner || !runner->setComputeAllPositionLogits(enabled))
            {
                all_success = false;
            }
        }
        if (all_success && !enabled)
        {
            // Child runners clear their row-indexed graph-builder state when
            // all-position logits are disabled. Mirror that aggregate state so
            // TP logits gathering uses the full-row shape on the next request.
            current_all_position_logit_rows_ = 0;
        }
        return all_success;
    }

    bool RankOrchestrator::setComputeRowIndexedAllPositionLogits(bool enabled, int row_count)
    {
        auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        bool all_success = true;
        for (auto &runner : participants)
        {
            if (!runner || !runner->setComputeRowIndexedAllPositionLogits(enabled, row_count))
            {
                all_success = false;
            }
        }
        if (all_success)
        {
            current_all_position_logit_rows_ = enabled ? row_count : 0;
        }
        return all_success;
    }

    bool RankOrchestrator::setMTPSpecVerifierInputPlan(
        const MTPSpecDecodeVerifierInputPlan &plan)
    {
        auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        bool all_success = true;
        for (auto &runner : participants)
        {
            if (!runner || !runner->setMTPSpecVerifierInputPlan(plan))
            {
                all_success = false;
            }
        }
        return all_success;
    }

    void RankOrchestrator::clearMTPSpecVerifierInputPlan()
    {
        auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        for (auto &runner : participants)
        {
            if (runner)
                runner->clearMTPSpecVerifierInputPlan();
        }
    }

    bool RankOrchestrator::supportsLogicalMTPVerifierBaseCheckpoint() const
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        for (const auto &runner : participants)
        {
            if (!runner || !runner->supportsLogicalMTPVerifierBaseCheckpoint())
            {
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::supportsMTPSpecStatePublication() const
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        for (const auto &runner : participants)
        {
            if (!runner || !runner->supportsMTPSpecStatePublication())
            {
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::supportsDeviceResidentMTPSpecStatePublication() const
    {
        /*
         * Single-child ranks are only a wrapper around the real runner, so keep
         * their resident publication capability identical to the child.  The
         * multi-child LocalTP implementation below is intentionally separate:
         * it owns a rank-level compact outcome and publishes through grouped
         * decode-equivalent child APIs rather than delegating to one child.
         */
        if (pp_stage_runners_.empty() &&
            device_runners_.size() == 1 &&
            device_runners_[0])
        {
            return device_runners_[0]->supportsDeviceResidentMTPSpecStatePublication();
        }

        /*
         * A PP sidecar can produce final-stage logits, but a rank-level
         * resident publication must also mutate every earlier PP stage's KV and
         * recurrent state.  Do not advertise this path for PP until there is a
         * PP-wide compact metadata mailbox and grouped publication proof.
         */
        if (!pp_stage_runners_.empty() || device_runners_.size() < 2)
        {
            return false;
        }

        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner != nullptr;
            });
    }

    bool RankOrchestrator::copyDeviceSpeculativeOutcomesToHost(
        const DeviceSpeculativeOutcomeHandle &handle,
        DeviceSpeculativeVerifyBatchOutcome *outcomes)
    {
        if (!outcomes ||
            !rank_compact_outcome_valid_ ||
            !handle.valid() ||
            handle.output_tokens_device != rank_compact_output_tokens_.data() ||
            handle.meta_device != rank_compact_output_meta_.data() ||
            handle.request_count != 1 ||
            handle.output_token_stride !=
                sampling_math::kSpeculativeBatchMaxOutputTokens ||
            handle.meta_stride != sampling_math::kSpeculativeBatchMetaCount)
        {
            return false;
        }

        /*
         * This bridge decodes the compact SamplingMath arrays that the rank
         * already used for publication planning.  It exists only to populate
         * response tokens and host diagnostics after live state has been
         * published; no row replay, model forward, or child mutation is hidden
         * behind this call.
         */
        if (!fillRankSpeculativeVerifyOutcomeFromMeta(
                rank_compact_output_tokens_,
                rank_compact_output_meta_,
                outcomes))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::Stochastic
                ? "rank_compact_stochastic_outcome_host_materializations"
                : "rank_compact_greedy_outcome_host_materializations",
            1.0,
            "decode",
            "rank",
            {{"requests", std::to_string(handle.request_count)},
             {"source", "rank_compact_sampling_math"}});
        return true;
    }

    bool RankOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
        const DeviceSpeculativePublicationRequest &request,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[RankOrchestrator] " << reason);
            return false;
        };

        if (pp_stage_runners_.empty() &&
            device_runners_.size() == 1 &&
            device_runners_[0])
        {
            const bool ok =
                device_runners_[0]->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                    request,
                    error);
            if (ok)
                refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return ok;
        }

        if (!supportsDeviceResidentMTPSpecStatePublication())
        {
            return fail(
                "rank compact MTP publication requires multi-child LocalTP grouped publication support on every participant");
        }
        if (!request.valid())
        {
            return fail(
                "rank compact MTP publication received an invalid device outcome request");
        }
        if (!rank_compact_outcome_valid_ ||
            request.outcome.output_tokens_device !=
                rank_compact_output_tokens_.data() ||
            request.outcome.meta_device != rank_compact_output_meta_.data() ||
            request.outcome.request_count != 1)
        {
            return fail(
                "rank compact MTP publication request does not reference the current rank-owned verifier outcome");
        }
        if (rank_compact_last_draft_tokens_.empty() ||
            static_cast<int>(rank_compact_last_draft_tokens_.size()) !=
                request.max_draft_tokens)
        {
            return fail(
                "rank compact MTP publication has no matching draft-token shape for the compact outcome");
        }

        DeviceSpeculativeVerifyBatchOutcome device_outcome;
        if (!copyDeviceSpeculativeOutcomesToHost(request.outcome, &device_outcome))
        {
            return fail(
                "rank compact MTP publication could not decode compact verifier metadata");
        }

        MTPDecodeCatchupGreedyRequest catchup_request;
        catchup_request.draft_tokens = rank_compact_last_draft_tokens_;
        catchup_request.stop_tokens = rank_compact_last_stop_tokens_;
        catchup_request.base_sidecar_position = request.base_sidecar_position;
        catchup_request.allow_speculative_discard = true;
        catchup_request.verifier_path =
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::Stochastic
                ? "rank_grouped_decode_equivalent_stochastic"
                : "rank_grouped_decode_equivalent_greedy";
        catchup_request.implementation_name =
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::Stochastic
                ? "rank_compact_cross_shard_stochastic_grouped_publication"
                : "rank_compact_cross_shard_argmax_grouped_publication";

        MTPSpecDecodeMetadataShape metadata_shape;
        metadata_shape.max_requests = request.request_count;
        metadata_shape.max_draft_tokens = request.max_draft_tokens;

        const std::vector<int> request_ids{0};
        const std::vector<MTPDecodeCatchupGreedyRequest> requests{
            catchup_request};
        const std::vector<MTPDeviceRejectionBatchOutcome> device_outcomes{
            device_outcome};
        const std::vector<int32_t> base_cached_tokens{
            !request.base_cached_tokens.empty()
                ? request.base_cached_tokens.front()
                : static_cast<int32_t>(request.base_sidecar_position)};

        MTPSpecTransactionBatchPlan transaction_plan =
            buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
                metadata_shape,
                request_ids,
                vocab_size(),
                requests,
                device_outcomes,
                base_cached_tokens);
        if (!transaction_plan.ok)
        {
            return fail(
                std::string("rank compact MTP transaction planning failed: ") +
                transaction_plan.error);
        }
        if (transaction_plan.requiresDecodeEquivalentReplayPublication())
        {
            return fail(
                "rank compact MTP transaction unexpectedly requested decode-equivalent replay publication");
        }

        for (MTPSpecStepPlan &step : transaction_plan.step_plans.steps)
        {
            step.publish_mtp_shifted_kv = request.publish_mtp_shifted_kv;
        }

        std::string publish_error;
        if (!publishGroupedDecodeEquivalentMTPSpecStateBatch(
                transaction_plan.step_plans,
                &publish_error))
        {
            return fail(
                std::string("rank compact MTP grouped publication failed: ") +
                publish_error);
        }

        rank_resident_child_logical_state_handles_.clear();
        rank_resident_child_logical_state_handles_.reserve(
            device_runners_.size());
        int child_request_count = -1;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
            {
                return fail(
                    "rank compact MTP grouped publication lost a LocalTP participant while aggregating resident mailboxes");
            }
            DeviceResidentLogicalSequenceStateHandle child_handle =
                device_runners_[i]->deviceResidentLogicalSequenceState();
            if (!child_handle.valid())
            {
                return fail(
                    "rank compact MTP grouped publication did not produce a resident logical-state mailbox on participant " +
                    std::to_string(i));
            }
            if (child_request_count < 0)
                child_request_count = child_handle.request_count;
            if (child_handle.request_count != child_request_count)
            {
                return fail(
                    "rank compact MTP grouped publication produced mismatched resident mailbox request counts");
            }
            rank_resident_child_logical_state_handles_.push_back(child_handle);
        }
        ++rank_resident_logical_state_epoch_;

        rank_last_device_outcome_step_plans_ = transaction_plan.step_plans;
        rank_compact_outcome_published_ = true;

        PerfStatsCollector::addCounter(
            "mtp",
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::Stochastic
                ? "rank_compact_stochastic_device_outcome_publications"
                : "rank_compact_greedy_device_outcome_publications",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"request_count", std::to_string(request.request_count)},
             {"max_draft_tokens", std::to_string(request.max_draft_tokens)},
             {"implementation", "grouped_decode_equivalent_child_publish"}});
        return true;
    }

    bool RankOrchestrator::adoptDeviceResidentMTPSpecPublishedHostState(
        const MTPSpecStepPlanBatch &plans,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[RankOrchestrator] " << reason);
            return false;
        };

        if (pp_stage_runners_.empty() &&
            device_runners_.size() == 1 &&
            device_runners_[0])
        {
            const bool ok =
                device_runners_[0]->adoptDeviceResidentMTPSpecPublishedHostState(
                    plans,
                    error);
            if (ok)
                refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return ok;
        }

        if (!plans.ok ||
            plans.request_count <= 0 ||
            static_cast<int>(plans.steps.size()) != plans.request_count)
        {
            return fail(
                "rank compact MTP host-state adoption received an invalid step-plan batch");
        }
        if (!rank_compact_outcome_published_)
        {
            return fail(
                "rank compact MTP host-state adoption ran before compact grouped publication");
        }

        /*
         * Publication already mutated every child through
         * publishGroupedDecodeEquivalentMTPSpecStateBatch(), which also
         * refreshed the rank mirrors.  Keep this hook as an explicit no-op
         * adoption boundary so OrchestrationRunner can use the same lifecycle
         * as single-device resident publication without accidentally publishing
         * the same accepted rows twice.
         */
        rank_last_device_outcome_step_plans_ = plans;
        refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();

        PerfStatsCollector::addCounter(
            "mtp",
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::Stochastic
                ? "rank_compact_stochastic_host_state_adoptions"
                : "rank_compact_greedy_host_state_adoptions",
            1.0,
            "decode",
            "rank",
            {{"request_count", std::to_string(plans.request_count)},
             {"implementation", "already_grouped_published"}});
        return true;
    }

    MTPVerifierRowCapability RankOrchestrator::mtpVerifierRowCapability() const
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        MTPVerifierRowCapability capability;
        if (participants.empty())
        {
            return capability;
        }

        bool initialized = false;
        for (const auto &runner : participants)
        {
            if (!runner)
            {
                return {};
            }
            const MTPVerifierRowCapability child =
                runner->mtpVerifierRowCapability();
            if (!initialized)
            {
                capability = child;
                initialized = true;
            }
            else
            {
                capability.intersectWith(child);
            }
        }
        return capability;
    }

    bool RankOrchestrator::publishAcceptedMTPSpecState(
        const MTPSpecStepPlan &plan,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[RankOrchestrator] " << reason);
            return false;
        };

        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_spec_state_publication_total",
            "decode",
            "rank",
            {{"participants",
              std::to_string(!pp_stage_runners_.empty()
                                 ? pp_stage_runners_.size()
                                 : device_runners_.size())}});

        if (!pp_stage_runners_.empty())
        {
            /*
             * Pipeline publication is an all-stage local-state mutation:
             * earlier stages own their KV/GDN rows, while the final stage also
             * owns logits and terminal hidden.  Publish every stage to the same
             * common accepted count before the request continues.
             */
            std::vector<MTPSpecStepPlan> participant_plans(
                pp_stage_runners_.size(),
                plan);
            MTPSpecCommonStepPlan common_plan =
                coordinateMTPSpecCommonAcceptedPrefix(participant_plans);
            if (!common_plan.ok ||
                common_plan.clamped_steps.size() != pp_stage_runners_.size())
            {
                return fail(
                    std::string("PP MTP spec-state common-prefix coordination failed: ") +
                    (common_plan.ok
                         ? std::string("missing clamped stage plans")
                         : common_plan.error));
            }
            if (common_plan.requires_common_fallback_replay)
            {
                return fail("PP MTP spec-state publication cannot publish divergent stages directly");
            }

            bool all_success = true;
            std::vector<std::string> child_errors(pp_stage_runners_.size());
            for (size_t i = 0; i < pp_stage_runners_.size(); ++i)
            {
                if (!pp_stage_runners_[i])
                {
                    child_errors[i] = "stage runner is unavailable";
                    all_success = false;
                    continue;
                }
                if (!pp_stage_runners_[i]->supportsMTPSpecStatePublication())
                {
                    child_errors[i] =
                        "stage does not support verifier-state publication";
                    all_success = false;
                    continue;
                }
                MTPSpecStepPlan stage_plan = common_plan.clamped_steps[i];
                stage_plan.publish_mtp_shifted_kv =
                    i + 1 == pp_stage_runners_.size();
                if (!pp_stage_runners_[i]->publishAcceptedMTPSpecState(
                        stage_plan,
                        &child_errors[i]))
                {
                    all_success = false;
                }
            }

            if (!all_success)
            {
                for (size_t i = 0; i < child_errors.size(); ++i)
                {
                    if (!child_errors[i].empty())
                    {
                        std::ostringstream msg;
                        msg << "PP MTP spec-state publication failed on stage "
                            << i << ": " << child_errors[i];
                        return fail(msg.str());
                    }
                }
                return fail("PP MTP spec-state publication failed on at least one stage");
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mtp_spec_state_publications",
                1.0,
                "decode",
                "rank",
                {{"topology", "local_pp"},
                 {"participants", std::to_string(pp_stage_runners_.size())},
                 {"accepted_count",
                  std::to_string(common_plan.common_accepted_count)}});
            refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return true;
        }
        if (device_runners_.empty())
        {
            return fail("MTP spec-state publication requires at least one rank participant");
        }

        /*
         * Build one participant plan per child even though the current
         * rank-level verifier has a single accepted-count decision.  Keeping
         * this path on the shared common-prefix helper prevents the LocalTP,
         * NodeLocalTP, PP, and ExpertParallel implementations from drifting
         * once participant-local accepted-state availability is introduced.
         */
        std::vector<MTPSpecStepPlan> participant_plans(
            device_runners_.size(),
            plan);
        MTPSpecCommonStepPlan common_plan =
            coordinateMTPSpecCommonAcceptedPrefix(participant_plans);
        if (!common_plan.ok ||
            common_plan.clamped_steps.size() != device_runners_.size())
        {
            return fail(
                std::string("MTP spec-state common-prefix coordination failed: ") +
                (common_plan.ok ? std::string("missing clamped participant plans")
                                : common_plan.error));
        }
        if (common_plan.requires_common_fallback_replay)
        {
            return fail("MTP spec-state publication cannot publish divergent rank participants directly");
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
            {
                std::ostringstream msg;
                msg << "MTP spec-state publication participant " << i
                    << " is unavailable";
                return fail(msg.str());
            }
            if (!device_runners_[i]->supportsMTPSpecStatePublication())
            {
                std::ostringstream msg;
                msg << "MTP spec-state publication participant " << i
                    << " does not support verifier-state publication";
                return fail(msg.str());
            }
        }

        if (device_runners_.size() == 1)
        {
            const bool ok = device_runners_[0]->publishAcceptedMTPSpecState(
                common_plan.clamped_steps.front(),
                error);
            if (ok)
                refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return ok;
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP spec-state publication failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        std::vector<std::string> child_errors(device_runners_.size());
        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_spec_state_publication_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, &common_plan, &child_errors, kernel_phase, rocm_phase,
                 cuda_phase, kv_phase, executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_spec_state_publication_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " accepted_count="
                                  << common_plan.clamped_steps[i].accepted_count);
                    }

                    const bool ok =
                        device_runners_[i] &&
                        device_runners_[i]->publishAcceptedMTPSpecState(
                            common_plan.clamped_steps[i],
                            &child_errors[i]);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_spec_state_publication_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_mtp_spec_state_publication_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }

        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecState: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "publishAcceptedMTPSpecState",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::publishAcceptedMTPSpecState: requesting TP context abort after worker timeout");
                    tp_ctx_->requestAbort();
                }
                continue;
            }

            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }

            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecState: Device "
                          << r.worker_index << " publication failed: "
                          << child_errors[static_cast<size_t>(r.worker_index)]);
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "publishAcceptedMTPSpecState",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecState: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (!all_success)
        {
            for (size_t i = 0; i < child_errors.size(); ++i)
            {
                if (!child_errors[i].empty())
                {
                    std::ostringstream msg;
                    msg << "MTP spec-state publication failed on participant "
                        << i << ": " << child_errors[i];
                    return fail(msg.str());
                }
            }
            return fail("MTP spec-state publication failed on at least one participant");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mtp_spec_state_publications",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"accepted_count", std::to_string(common_plan.common_accepted_count)}});
        refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
        return true;
    }

    bool RankOrchestrator::publishAcceptedMTPSpecStateBatch(
        const MTPSpecStepPlanBatch &plans,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[RankOrchestrator] " << reason);
            return false;
        };

        const size_t participant_count =
            !pp_stage_runners_.empty()
                ? pp_stage_runners_.size()
                : device_runners_.size();

        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_spec_state_batch_publication_total",
            "decode",
            "rank",
            {{"participants", std::to_string(participant_count)},
             {"request_count", std::to_string(plans.request_count)}});

        if (!plans.ok)
        {
            return fail("MTP spec-state batch publication received an invalid plan batch: " +
                        plans.error);
        }
        if (plans.request_count <= 0 ||
            static_cast<int>(plans.steps.size()) != plans.request_count)
        {
            return fail("MTP spec-state batch publication received inconsistent request count");
        }

        auto build_common_batches =
            [&](size_t count,
                bool local_pp,
                std::vector<MTPSpecStepPlanBatch> &out_batches) -> bool
        {
            out_batches.assign(count, plans);

            /*
             * Clamp each logical request independently.  Every participant sees
             * the same request batch today, but keeping the per-request
             * common-prefix step here prevents future per-participant state
             * availability from mutating live state divergently.
             */
            for (size_t step_idx = 0; step_idx < plans.steps.size(); ++step_idx)
            {
                std::vector<MTPSpecStepPlan> participant_steps(
                    count,
                    plans.steps[step_idx]);
                MTPSpecCommonStepPlan common_plan =
                    coordinateMTPSpecCommonAcceptedPrefix(participant_steps);
                if (!common_plan.ok ||
                    common_plan.clamped_steps.size() != count)
                {
                    return fail(
                        std::string("MTP spec-state batch common-prefix coordination failed: ") +
                        (common_plan.ok
                             ? std::string("missing clamped participant plans")
                             : common_plan.error));
                }
                if (common_plan.requires_common_fallback_replay)
                {
                    return fail("MTP spec-state batch publication cannot publish divergent participants directly");
                }

                for (size_t participant = 0; participant < count; ++participant)
                {
                    MTPSpecStepPlan clamped_step =
                        common_plan.clamped_steps[participant];
                    if (local_pp)
                    {
                        /*
                         * Non-final PP stages own main per-layer state only.
                         * The final stage owns the sidecar shifted KV cache.
                         */
                        clamped_step.publish_mtp_shifted_kv =
                            participant + 1 == count;
                    }
                    out_batches[participant].steps[step_idx] = clamped_step;
                }
            }
            return true;
        };

        if (!pp_stage_runners_.empty())
        {
            std::vector<MTPSpecStepPlanBatch> stage_batches;
            if (!build_common_batches(
                    pp_stage_runners_.size(),
                    /*local_pp=*/true,
                    stage_batches))
            {
                return false;
            }

            bool all_success = true;
            std::vector<std::string> child_errors(pp_stage_runners_.size());
            for (size_t i = 0; i < pp_stage_runners_.size(); ++i)
            {
                if (!pp_stage_runners_[i])
                {
                    child_errors[i] = "stage runner is unavailable";
                    all_success = false;
                    continue;
                }
                if (!grouped_decode_equivalent_spec_publication_scope_ &&
                    !pp_stage_runners_[i]->supportsMTPSpecStatePublication())
                {
                    child_errors[i] =
                        "stage does not support verifier-state publication";
                    all_success = false;
                    continue;
                }
                const bool published =
                    grouped_decode_equivalent_spec_publication_scope_
                        ? pp_stage_runners_[i]->publishGroupedDecodeEquivalentMTPSpecStateBatch(
                              stage_batches[i],
                              &child_errors[i])
                        : pp_stage_runners_[i]->publishAcceptedMTPSpecStateBatch(
                              stage_batches[i],
                              &child_errors[i]);
                if (!published)
                {
                    all_success = false;
                }
            }

            if (!all_success)
            {
                for (size_t i = 0; i < child_errors.size(); ++i)
                {
                    if (!child_errors[i].empty())
                    {
                        std::ostringstream msg;
                        msg << "PP MTP spec-state batch publication failed on stage "
                            << i << ": " << child_errors[i];
                        return fail(msg.str());
                    }
                }
                return fail("PP MTP spec-state batch publication failed on at least one stage");
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mtp_spec_state_batch_publications",
                1.0,
                "decode",
                "rank",
                {{"topology", "local_pp"},
                 {"participants", std::to_string(pp_stage_runners_.size())},
                 {"request_count", std::to_string(plans.request_count)}});
            refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return true;
        }

        if (device_runners_.empty())
        {
            return fail("MTP spec-state batch publication requires at least one rank participant");
        }

        std::vector<MTPSpecStepPlanBatch> participant_batches;
        if (!build_common_batches(
                device_runners_.size(),
                /*local_pp=*/false,
                participant_batches))
        {
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
            {
                std::ostringstream msg;
                msg << "MTP spec-state batch publication participant " << i
                    << " is unavailable";
                return fail(msg.str());
            }
            if (!grouped_decode_equivalent_spec_publication_scope_ &&
                !device_runners_[i]->supportsMTPSpecStatePublication())
            {
                std::ostringstream msg;
                msg << "MTP spec-state batch publication participant " << i
                    << " does not support verifier-state publication";
                return fail(msg.str());
            }
        }

        if (device_runners_.size() == 1)
        {
            const bool ok =
                grouped_decode_equivalent_spec_publication_scope_
                    ? device_runners_[0]->publishGroupedDecodeEquivalentMTPSpecStateBatch(
                          participant_batches.front(),
                          error)
                    : device_runners_[0]->publishAcceptedMTPSpecStateBatch(
                          participant_batches.front(),
                          error);
            if (ok)
                refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return ok;
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP spec-state batch publication failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        std::vector<std::string> child_errors(device_runners_.size());
        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_spec_state_batch_publication_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_count", std::to_string(plans.request_count)}});
            tp_worker_pool_->dispatch(
                [this, &participant_batches, &child_errors, kernel_phase,
                 rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_spec_state_batch_publication_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " request_count="
                                  << participant_batches[i].request_count);
                    }

                    const bool ok =
                        device_runners_[i] &&
                        (grouped_decode_equivalent_spec_publication_scope_
                             ? device_runners_[i]->publishGroupedDecodeEquivalentMTPSpecStateBatch(
                                   participant_batches[i],
                                   &child_errors[i])
                             : device_runners_[i]->publishAcceptedMTPSpecStateBatch(
                                   participant_batches[i],
                                   &child_errors[i]));

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_spec_state_batch_publication_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_mtp_spec_state_batch_publication_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_count", std::to_string(plans.request_count)}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }

        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecStateBatch: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "publishAcceptedMTPSpecStateBatch",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::publishAcceptedMTPSpecStateBatch: requesting TP context abort after worker timeout");
                    tp_ctx_->requestAbort();
                }
                continue;
            }

            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }

            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecStateBatch: Device "
                          << r.worker_index << " publication failed: "
                          << child_errors[static_cast<size_t>(r.worker_index)]);
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "publishAcceptedMTPSpecStateBatch",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecStateBatch: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (!all_success)
        {
            for (size_t i = 0; i < child_errors.size(); ++i)
            {
                if (!child_errors[i].empty())
                {
                    std::ostringstream msg;
                    msg << "MTP spec-state batch publication failed on participant "
                        << i << ": " << child_errors[i];
                    return fail(msg.str());
                }
            }
            return fail("MTP spec-state batch publication failed on at least one participant");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mtp_spec_state_batch_publications",
            1.0,
            "decode",
            "rank",
            {{"topology", "local_tp"},
             {"participants", std::to_string(device_runners_.size())},
             {"request_count", std::to_string(plans.request_count)}});
        refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
        return true;
    }

    /**
     * @brief Publish grouped decode-equivalent accepted rows across the rank.
     *
     * The ordinary batch publisher already owns the hard parts: per-request
     * common-prefix clamping, PP shifted-KV ownership, TP worker-pool fan-out,
     * timeout handling, and participant error aggregation.  This method wraps
     * that machinery in a scope that swaps child calls to the grouped API, so
     * the new lane inherits the same correctness coordination without copying a
     * second publication implementation.
     */
    bool RankOrchestrator::publishGroupedDecodeEquivalentMTPSpecStateBatch(
        const MTPSpecStepPlanBatch &plans,
        std::string *error)
    {
        /*
         * The scoped flag is the only difference from a direct batch
         * publication.  It makes publishAcceptedMTPSpecStateBatch() ask each
         * participant's grouped publisher directly, then restores normal
         * direct-publish behavior before returning to the caller.
         */
        struct ScopedGroupedDecodeEquivalentPublication
        {
            bool &slot;
            bool previous;

            explicit ScopedGroupedDecodeEquivalentPublication(bool &slot_in)
                : slot(slot_in), previous(slot_in)
            {
                slot = true;
            }

            ~ScopedGroupedDecodeEquivalentPublication()
            {
                slot = previous;
            }
        } scope(grouped_decode_equivalent_spec_publication_scope_);

        const bool ok = publishAcceptedMTPSpecStateBatch(plans, error);
        if (ok)
        {
            const size_t participant_count =
                !pp_stage_runners_.empty()
                    ? pp_stage_runners_.size()
                    : device_runners_.size();
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_grouped_decode_equivalent_spec_state_batch_publications",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(participant_count)},
                 {"request_count", std::to_string(plans.request_count)}});
        }
        return ok;
    }

    const float *RankOrchestrator::getAllPositionLogits() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->getAllPositionLogits();
        }

        if (device_runners_.empty())
        {
            return nullptr;
        }

        const size_t rows = std::max<size_t>(
            1,
            static_cast<size_t>(
                current_all_position_logit_rows_ > 0
                    ? current_all_position_logit_rows_
                    : std::max(0, current_padded_seq_len_)));

        bool has_local_all_position_logits = false;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            has_local_all_position_logits =
                has_local_all_position_logits || runner->hasAllPositionLogitsLocal();
        }

        if (has_local_all_position_logits)
        {
            std::vector<LogitsLocalInfo> local_infos;
            local_infos.reserve(device_runners_.size());
            for (const auto &runner : device_runners_)
            {
                if (!runner->hasAllPositionLogitsLocal())
                {
                    LOG_WARN("RankOrchestrator::getAllPositionLogits: mixed local and replicated verifier logits are unsupported");
                    return nullptr;
                }

                LogitsLocalInfo info = runner->getAllPositionLogitsLocalInfo();
                if (!info)
                    return nullptr;
                local_infos.push_back(info);
            }

            const int full_vocab = vocab_size();
            if (full_vocab <= 0)
                return nullptr;

            const size_t required_elements = rows * static_cast<size_t>(full_vocab);
            if (!all_position_logits_gatherer_ ||
                all_position_logits_gatherer_->bufferNumel() < required_elements)
            {
                all_position_logits_gatherer_ = std::make_unique<LogitsGatherer>(full_vocab, rows);
            }

            if (!all_position_logits_gatherer_ ||
                !all_position_logits_gatherer_->gatherLocalInfos(local_infos, rows, full_vocab))
            {
                return nullptr;
            }
            return all_position_logits_gatherer_->data();
        }

        const int compare_vocab = device_runners_[0] ? device_runners_[0]->vocab_size() : 0;
        if (compare_vocab <= 0)
        {
            return nullptr;
        }

        /*
         * Replicated full-vocab verifier logits follow the same public
         * contract as ordinary serial TP decode: child 0 is the authoritative
         * host-visible logits tensor.  Requiring bitwise equality across every
         * child here is stricter than serial decode and makes decode-equivalent
         * verifier rows fail before the parity harness can compare them with
         * the serial path that production sampling actually observes.
         */
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            if (runner->vocab_size() != compare_vocab)
                return nullptr;
        }
        return device_runners_[0]->getAllPositionLogits();
    }

    std::string RankOrchestrator::mtpDecodeUnsupportedReason() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            const std::string child_reason =
                pp_sidecar->mtpDecodeUnsupportedReason();
            if (!child_reason.empty())
                return child_reason;
            return {};
        }
        if (device_runners_.size() > 1)
        {
            for (const auto &runner : device_runners_)
            {
                if (!runner)
                    return "MTP decode requires every TP participant to be available";

                const std::string child_reason = runner->mtpDecodeUnsupportedReason();
                if (!child_reason.empty())
                    return child_reason;
            }
        }
        return {};
    }

    bool RankOrchestrator::supportsMTPSidecarLogitsStreamHandoff() const
    {
        /*
         * LocalPP is intentionally false for now: the sidecar logits are
         * produced on the pipeline tail, but verifier token input belongs to
         * the pipeline head.  Keeping this false prevents the vLLM-style
         * stochastic path from assuming one stage's device token slot can be
         * consumed by another stage without an explicit handoff contract.
         */
        if (finalPPSidecarRunner())
        {
            return false;
        }
        if (device_runners_.empty())
            return false;
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->supportsMTPSidecarLogitsStreamHandoff();
            });
    }

    bool RankOrchestrator::supportsMTPDeviceDraftTokenInput() const
    {
        if (finalPPSidecarRunner())
        {
            return false;
        }
        if (device_runners_.empty())
            return false;
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->supportsMTPDeviceDraftTokenInput();
            });
    }

    bool RankOrchestrator::supportsMTPSidecarPreservesMainState() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsMTPSidecarPreservesMainState();
        }
        if (device_runners_.empty())
            return false;

        /*
         * LocalTP runs the MTP sidecar on every participant in lockstep.  The
         * rank-level capability is therefore a domain-wide AND: a single child
         * that cannot preserve main state makes the whole TP sidecar unsafe for
         * verifier-base reuse.
         */
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->supportsMTPSidecarPreservesMainState();
            });
    }

    bool RankOrchestrator::supportsMTPShiftedRowReuseFromSidecar() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsMTPShiftedRowReuseFromSidecar();
        }
        if (device_runners_.empty())
            return false;

        /*
         * Reusing the first shifted MTP KV row is only correct when every TP
         * shard can keep its sidecar row as the accepted row-zero publication
         * boundary.  Treat it as a rank-wide all-participant capability rather
         * than a property of participant 0.
         */
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->supportsMTPShiftedRowReuseFromSidecar();
            });
    }

    bool RankOrchestrator::applyPenaltiesOnDevice(
        const std::vector<LogitPenalty> &penalties,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyPenaltiesOnDevice(penalties, vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->applyPenaltiesOnDevice(
                penalties,
                vocab_size);
        }
        if (penalties.empty())
            return true;

        /*
         * Column-parallel logits are rank-wide state.  Each child runner
         * applies the same sparse global-token penalty map and internally
         * ignores tokens outside its vocab shard.  Returning false here for
         * non-empty maps made temperature-zero LocalTP MTP fail as soon as
         * model defaults enabled repetition penalties.
         */
        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner &&
                 runner->applyPenaltiesOnDevice(penalties, vocab_size) &&
                 ok;
        }
        return ok;
    }

    bool RankOrchestrator::applyPenaltiesToMTPLogitsOnDevice(
        const std::vector<LogitPenalty> &penalties,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyPenaltiesToMTPLogitsOnDevice(
                penalties,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->applyPenaltiesToMTPLogitsOnDevice(
                penalties,
                vocab_size);
        }
        if (penalties.empty())
            return true;

        /*
         * MTP sidecar logits are sharded exactly like the main LM head in
         * LocalTP.  Fan out the global sparse penalties to every participant so
         * the next rank-level sampler sees a coherent penalty-mutated row.
         */
        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner &&
                 runner->applyPenaltiesToMTPLogitsOnDevice(
                     penalties,
                     vocab_size) &&
                 ok;
        }
        return ok;
    }

    bool RankOrchestrator::applyPenaltiesToAllPositionLogitsOnDeviceRow(
        int row,
        const std::vector<LogitPenalty> &penalties,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                row,
                penalties,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                row,
                penalties,
                vocab_size);
        }
        if (penalties.empty())
            return true;

        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner &&
                 runner->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                     row,
                     penalties,
                     vocab_size) &&
                 ok;
        }
        return ok;
    }

    bool RankOrchestrator::supportsRowLocalAllPositionPenaltyApplication() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsRowLocalAllPositionPenaltyApplication();
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->supportsRowLocalAllPositionPenaltyApplication();
        }

        /*
         * Multi-child TP/PP can promote penalty-greedy compact verification
         * only after every participant can mutate its verifier shard with the
         * same branch-local sampler history.  The compact reducer is a separate
         * capability, so callers still need to check that too.
         */
        if (device_runners_.empty())
            return false;
        for (const auto &runner : device_runners_)
        {
            if (!runner ||
                !runner->supportsRowLocalAllPositionPenaltyApplication())
            {
                return false;
            }
        }
        return true;
    }

    void RankOrchestrator::setSkipLogitsGatherDecode(bool skip)
    {
        skip_logits_gather_decode_ = skip;
        applyLogitsGatherSkipFlags();
    }

    void RankOrchestrator::setSkipLogitsGatherPrefill(bool skip)
    {
        skip_logits_gather_prefill_ = skip;
        applyLogitsGatherSkipFlags();
    }

    void RankOrchestrator::setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled)
    {
        for (auto &runner : device_runners_)
        {
            if (runner)
                runner->setMTPAllPositionVerifierSyncDeferralEnabled(enabled);
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
                runner->setMTPAllPositionVerifierSyncDeferralEnabled(enabled);
        }
    }

    void RankOrchestrator::setMTPMainDecodeSyncDeferralEnabled(bool enabled)
    {
        for (auto &runner : device_runners_)
        {
            if (runner)
                runner->setMTPMainDecodeSyncDeferralEnabled(enabled);
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
                runner->setMTPMainDecodeSyncDeferralEnabled(enabled);
        }
    }

    void RankOrchestrator::applyLogitsGatherSkipFlags()
    {
        if (logits_gatherer_)
        {
            logits_gatherer_->setSkipDecode(skip_logits_gather_decode_);
            logits_gatherer_->setSkipPrefill(skip_logits_gather_prefill_);
        }

        for (auto &runner : device_runners_)
        {
            if (!runner)
                continue;
            runner->setSkipLogitsGatherDecode(skip_logits_gather_decode_);
            runner->setSkipLogitsGatherPrefill(skip_logits_gather_prefill_);
        }

        for (auto &runner : pp_stage_runners_)
        {
            if (!runner)
                continue;
            runner->setSkipLogitsGatherDecode(skip_logits_gather_decode_);
            runner->setSkipLogitsGatherPrefill(skip_logits_gather_prefill_);
        }
    }

    void RankOrchestrator::wireLocalTPMoERuntimeHistogramSyncs()
    {
        if (local_tp_moe_histogram_syncs_wired_)
            return;
        local_tp_moe_histogram_syncs_wired_ = true;

        if (!tp_ctx_ || tp_ctx_->degree() <= 1 || device_runners_.size() <= 1)
            return;
        if (usesDeviceSideMoERebalanceController())
        {
            LOG_DEBUG("RankOrchestrator: skipping host MoE runtime histogram bridge; "
                      "device-side graph rebalance owns histogram allgather");
            return;
        }

        std::vector<MoERebalanceController *> controllers;
        controllers.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                continue;
            for (auto *controller : runner->moeRebalanceControllers())
            {
                if (controller &&
                    std::find(controllers.begin(), controllers.end(), controller) == controllers.end())
                {
                    controllers.push_back(controller);
                }
            }
        }

        auto *active = selectActiveMoERebalanceController(controllers);
        if (!active || !active->histogram())
            return;

        DecodeExpertHistogram *active_histogram = active->histogram();
        const std::string domain_id = active->domainId();
        const int num_layers = active->numLayers();
        const int num_experts = active->numExperts();
        int sibling_count = 0;

        for (auto *source_controller : controllers)
        {
            if (!source_controller || source_controller == active)
                continue;
            if (source_controller->domainId() != domain_id)
                continue;
            if (source_controller->numLayers() != num_layers ||
                source_controller->numExperts() != num_experts)
            {
                LOG_WARN("RankOrchestrator: skipping LocalTP MoE runtime histogram bridge for domain "
                         << domain_id << " because controller dimensions differ");
                continue;
            }

            DecodeExpertHistogram *source_histogram = source_controller->histogram();
            if (!source_histogram || source_histogram == active_histogram)
                continue;

            const int source_participant = ++sibling_count;
            active_histogram->registerRuntimeHistogramSync(
                [active_histogram,
                 source_histogram,
                 domain_id,
                 num_layers,
                 num_experts,
                 source_participant]() -> bool
                {
                    if (!source_histogram->syncRuntimeHistograms())
                        return false;

                    uint64_t activations_merged = 0;
                    for (int layer = 0; layer < num_layers; ++layer)
                    {
                        auto counts = source_histogram->layerHistogram(layer);
                        for (uint64_t count : counts)
                            activations_merged += count;
                        active_histogram->mergeLayerCounts(
                            layer,
                            counts.data(),
                            num_experts,
                            /*count_window_tokens=*/false);
                    }

                    source_histogram->resetWindow();

                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "rank_runtime_histogram_sibling_syncs",
                        1.0,
                        "rebalance",
                        {},
                        {{"domain_id", domain_id},
                         {"source_participant", std::to_string(source_participant)}});
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "rank_runtime_histogram_sibling_activations",
                        static_cast<double>(activations_merged),
                        "rebalance",
                        {},
                        {{"domain_id", domain_id},
                         {"source_participant", std::to_string(source_participant)}});
                    return true;
                });

            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "rank_runtime_histogram_sibling_sync_registrations",
                1.0,
                "rebalance",
                primaryDeviceId().toString(),
                {{"domain_id", domain_id},
                 {"source_participant", std::to_string(source_participant)}});
        }

        if (sibling_count > 0)
        {
            LOG_DEBUG("RankOrchestrator: wired " << sibling_count
                                                 << " LocalTP sibling MoE runtime histogram sync(s) for domain "
                                                 << domain_id);
        }
    }

    bool RankOrchestrator::forward_batch(const std::vector<std::vector<int>> &token_batches)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::forward_batch: No device runners available");
            return false;
        }

        LOG_DEBUG("RankOrchestrator::forward_batch: batch_size=" << token_batches.size()
                                                                 << ", devices=" << device_runners_.size());

        // Launch parallel batch forward passes on all devices
        std::vector<std::future<bool>> futures;
        futures.reserve(device_runners_.size());

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            auto &runner = device_runners_[i];
            if (runner)
            {
                futures.push_back(std::async(std::launch::async,
                                             [&runner, &token_batches]()
                                             {
                                                 return runner->forward_batch(token_batches);
                                             }));
            }
        }

        // Wait for all to complete - using same exception capture pattern as forward()
        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;

        for (size_t i = 0; i < futures.size(); ++i)
        {
            try
            {
                bool success = futures[i].get();
                if (!success)
                {
                    LOG_ERROR("RankOrchestrator::forward_batch: Device " << i << " forward_batch failed");
                    all_success = false;
                }
            }
            catch (const std::exception &e)
            {
                all_success = false;
                std::string error_msg = e.what();
                bool is_context_destroyed = (error_msg.find("context is destroyed") != std::string::npos ||
                                             error_msg.find("error 709") != std::string::npos);

                if (!first_exception)
                {
                    first_exception = std::current_exception();
                    first_exception_device = i;
                    LOG_ERROR("RankOrchestrator::forward_batch: Device " << i
                                                                         << " threw PRIMARY exception: " << error_msg);
                }
                else if (is_context_destroyed)
                {
                    LOG_WARN("RankOrchestrator::forward_batch: Device " << i
                                                                        << " threw SECONDARY exception (context destroyed): " << error_msg);
                }
                else
                {
                    LOG_ERROR("RankOrchestrator::forward_batch: Device " << i
                                                                         << " threw exception: " << error_msg);
                }
            }
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forward_batch: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (all_success)
        {
            // Update batch tracking from primary device
            if (!device_runners_.empty() && device_runners_[0])
            {
                current_batch_size_ = device_runners_[0]->batch_size();
                current_padded_seq_len_ = device_runners_[0]->padded_seq_len();
                current_sequence_lengths_ = device_runners_[0]->sequence_lengths();
            }
            stats_dirty_ = true;
        }

        return all_success;
    }

    const float *RankOrchestrator::getLogits(int seq_idx) const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getLogits(seq_idx);
        }
        return nullptr;
    }

    int RankOrchestrator::batch_size() const
    {
        return current_batch_size_;
    }

    int RankOrchestrator::padded_seq_len() const
    {
        return current_padded_seq_len_;
    }

    const std::vector<int> &RankOrchestrator::sequence_lengths() const
    {
        return current_sequence_lengths_;
    }

    int RankOrchestrator::vocab_size() const
    {
        // For tensor-parallel setups, the LM head may be column-sharded across devices.
        // In that case, each device has logits for vocab_size/tp_degree tokens.
        // We should return the FULL vocab size (sum of all devices), not just device 0's.
        //
        // The model_ctx_ always has the true total vocab size from the model metadata.
        if (model_ctx_)
        {
            return static_cast<int>(model_ctx_->vocabSize());
        }

        // Fallback: if no model context, use device 0's vocab (may be wrong for TP)
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->vocab_size();
        }
        return 0;
    }

    void RankOrchestrator::resetInferenceState(const InferenceStateResetRequest &request)
    {
        if (request.boundary != InferenceStateResetRequest::Boundary::Request ||
            !request.resetsAllLiveRequestOwners() ||
            !request.reset_model_runtime ||
            !request.preserve_replay_safe_graphs)
        {
            throw std::invalid_argument(
                "RankOrchestrator::resetInferenceState currently supports only full "
                "request-boundary resets that preserve replay-safe graph captures");
        }
        LOG_DEBUG("RankOrchestrator::resetInferenceState: resetting request-owned state on "
                  << device_runners_.size() << " TP devices and "
                  << pp_stage_runners_.size() << " PP stages"
                  << " reason=" << (request.reason ? request.reason : "request-boundary"));

        // Reset TP device runners.
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->resetInferenceState(request);
            }
        }

        // Reset PP stage runners (each has its own KV/GDN owner).
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->resetInferenceState(request);
            }
        }

        current_position_ = 0;
        current_padded_seq_len_ = 0;
        current_sequence_lengths_.assign(
            static_cast<size_t>(std::max(1, config_.batch_size)),
            0);
        stats_dirty_ = true;
    }

    void RankOrchestrator::clear_cache()
    {
        resetInferenceState(
            InferenceStateResetRequest::requestBoundary("clear_cache"));
    }

    void RankOrchestrator::drainCompletedDecodeBoundaryMaintenanceDiagnostics()
    {
        for (auto &runner : device_runners_)
        {
            if (runner)
                runner->drainCompletedDecodeBoundaryMaintenanceDiagnostics();
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
                runner->drainCompletedDecodeBoundaryMaintenanceDiagnostics();
        }
    }

    int RankOrchestrator::get_position() const
    {
        // PP mode: return position from first PP stage runner
        // All PP stage runners should have synchronized positions
        if (!pp_stage_runners_.empty() && pp_stage_runners_[0])
        {
            return pp_stage_runners_[0]->get_position();
        }

        // TP mode: return position from primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->get_position();
        }

        return current_position_;
    }

    ExecutionPath RankOrchestrator::executionPath() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->executionPath();
        }
        return ExecutionPath::GRAPH;
    }

    const char *RankOrchestrator::architecture() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->architecture();
        }
        return "Unknown";
    }

    uint64_t RankOrchestrator::moePlacementEpoch() const
    {
        uint64_t epoch = 0;
        for (const auto &runner : device_runners_)
        {
            if (runner)
                epoch = std::max(epoch, runner->moePlacementEpoch());
        }
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
                epoch = std::max(epoch, runner->moePlacementEpoch());
        }
        return epoch;
    }

    uint64_t RankOrchestrator::moeRuntimeMovementEpoch() const
    {
        uint64_t epoch = 0;
        for (const auto &runner : device_runners_)
        {
            if (runner)
                epoch = std::max(epoch, runner->moeRuntimeMovementEpoch());
        }
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
                epoch = std::max(epoch, runner->moeRuntimeMovementEpoch());
        }
        return epoch;
    }

    PrefixLookupResult RankOrchestrator::lookupPrefix(const std::vector<int32_t> &tokens)
    {
        PrefixLookupResult aggregate;
        last_device_prefix_hits_.clear();
        last_pp_prefix_hits_.clear();

        std::vector<PrefixParticipantLookup> participants;
        int block_size = 0;
        int participant_id = 0;

        auto query_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                 std::vector<PrefixLookupResult> &hits,
                                 bool fingerprint_must_match)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;

                PrefixLookupResult hit = runner->lookupPrefix(tokens);
                LOG_DEBUG("RankOrchestrator::lookupPrefix participant " << participant_id
                          << " device=" << runner->primaryDeviceId().toString()
                          << " supported=" << hit.supported
                          << " cached_tokens=" << hit.cached_tokens
                          << " blocks=" << hit.blocks.size()
                          << " requires_terminal_logits=" << hit.requires_terminal_logits
                          << " has_terminal_logits=" << hit.has_terminal_logits
                          << " requires_terminal_hidden=" << hit.requires_terminal_hidden
                          << " has_terminal_hidden=" << hit.has_terminal_hidden);
                PrefixParticipantLookup participant = makePrefixParticipantLookup(
                    participant_id++,
                    runner->primaryDeviceId(),
                    hit,
                    {},
                    runner->moePlacementEpoch());
                participant.fingerprint_must_match = fingerprint_must_match;
                participants.push_back(std::move(participant));
                hits.push_back(hit);
                if (block_size <= 0 && hit.block_size > 0)
                    block_size = hit.block_size;
            }
        };

        // Local TP participants own sharded payload slices, so their local
        // fingerprints can legitimately differ. Each child lookup has already
        // validated its own fingerprint; the rank aggregate only clamps to the
        // common restorable token count.
        query_runners(device_runners_, last_device_prefix_hits_, /*fingerprint_must_match=*/false);
        query_runners(pp_stage_runners_, last_pp_prefix_hits_, /*fingerprint_must_match=*/false);

        if (participants.empty())
            return aggregate;

        aggregate = makePrefixLookupResult(coordinatePrefixLookups(std::move(participants)), block_size);
        LOG_DEBUG("RankOrchestrator::lookupPrefix aggregate supported=" << aggregate.supported
                  << " cached_tokens=" << aggregate.cached_tokens
                  << " block_size=" << aggregate.block_size
                  << " requires_terminal_logits=" << aggregate.requires_terminal_logits
                  << " has_terminal_logits=" << aggregate.has_terminal_logits
                  << " requires_terminal_hidden=" << aggregate.requires_terminal_hidden
                  << " has_terminal_hidden=" << aggregate.has_terminal_hidden);
        const int common_tokens = std::max(0, aggregate.cached_tokens);
        if (common_tokens > 0)
        {
            const auto copy_representative_blocks =
                [&](const std::vector<PrefixLookupResult> &hits)
            {
                for (const auto &hit : hits)
                {
                    if (hit.cached_tokens >= common_tokens && !hit.blocks.empty())
                    {
                        aggregate.blocks = hit.clampedTo(common_tokens).blocks;
                        return true;
                    }
                }
                return false;
            };

            if (!copy_representative_blocks(last_device_prefix_hits_))
            {
                copy_representative_blocks(last_pp_prefix_hits_);
            }
        }

        return aggregate;
    }

    bool RankOrchestrator::populatePrefix(const PrefixLookupResult &hit, int seq_idx)
    {
        const int common_tokens = std::max(0, hit.cached_tokens);
        if (common_tokens <= 0)
            return true;

        auto populate_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                    const std::vector<PrefixLookupResult> &hits)
        {
            size_t hit_index = 0;
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                if (hit_index >= hits.size())
                    return false;

                PrefixLookupResult child_hit = hits[hit_index++].clampedTo(common_tokens);
                child_hit.restore_model_runtime_state = hit.restore_model_runtime_state;
                child_hit.restore_hybrid_state_for_suffix_prefill =
                    hit.restore_hybrid_state_for_suffix_prefill;
                if (!runner->populatePrefix(child_hit, seq_idx))
                    return false;
            }
            return hit_index == hits.size();
        };

        const bool populated =
            populate_runners(device_runners_, last_device_prefix_hits_) &&
            populate_runners(pp_stage_runners_, last_pp_prefix_hits_);
        if (!populated)
        {
            clear_cache();
            return false;
        }

        current_position_ = common_tokens;
        current_padded_seq_len_ = 0;
        current_sequence_lengths_.assign(
            static_cast<size_t>(std::max(1, config_.batch_size)),
            common_tokens);
        stats_dirty_ = true;
        return true;
    }

    bool RankOrchestrator::harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count)
    {
        bool saw_runner = false;
        bool ok = true;
        auto harvest_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                saw_runner = true;
                ok = runner->harvestPrefix(tokens, prompt_token_count) && ok;
            }
        };

        harvest_runners(device_runners_);
        harvest_runners(pp_stage_runners_);
        return saw_runner && ok;
    }

    bool RankOrchestrator::restorePrefixTerminalState(const PrefixLookupResult &hit)
    {
        const int common_tokens = std::max(0, hit.cached_tokens);
        if (common_tokens <= 0 || !hit.has_terminal_logits)
            return false;

        auto restore_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                   const std::vector<PrefixLookupResult> &hits)
        {
            size_t hit_index = 0;
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;

                PrefixLookupResult child_hit =
                    hit_index < hits.size() ? hits[hit_index++].clampedTo(common_tokens)
                                            : hit.clampedTo(common_tokens);
                LOG_DEBUG("RankOrchestrator::restorePrefixTerminalState child device="
                          << runner->primaryDeviceId().toString()
                          << " cached_tokens=" << child_hit.cached_tokens
                          << " requires_terminal_logits=" << child_hit.requires_terminal_logits
                          << " has_terminal_logits=" << child_hit.has_terminal_logits);
                if (!runner->restorePrefixTerminalState(child_hit))
                {
                    LOG_DEBUG("RankOrchestrator::restorePrefixTerminalState child restore failed");
                    return false;
                }
            }
            return true;
        };

        const bool restored =
            restore_runners(device_runners_, last_device_prefix_hits_) &&
            restore_runners(pp_stage_runners_, last_pp_prefix_hits_);
        if (!restored)
            return false;

        if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
        {
            if (!logits_gatherer_)
            {
                logits_gatherer_ = std::make_unique<LogitsGatherer>(0, 0);
                applyLogitsGatherSkipFlags();
            }
            logits_gatherer_->copyFromStage(*pp_stage_runners_.back(),
                                            static_cast<size_t>(vocab_size()),
                                            config_.batch_size,
                                            static_cast<int>(config_.max_seq_len));
        }
        current_position_ = common_tokens;
        stats_dirty_ = true;
        return true;
    }

    PrefixStateSnapshot RankOrchestrator::captureLivePrefixState(int seq_idx) const
    {
        PrefixStateSnapshot aggregate;
        bool saw_runner = false;
        bool have_common_tokens = false;
        bool have_common_provenance = false;
        bool same_provenance = true;
        int common_tokens = 0;
        PrefixStateProvenance common_provenance = PrefixStateProvenance::Unknown;

        auto capture_runners = [&](
                                   const std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                   const char *runner_group)
        {
            size_t participant_index = 0;
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;

                saw_runner = true;
                PrefixStateSnapshot child = runner->captureLivePrefixState(seq_idx);
                if (!child.valid)
                {
                    rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                        "capture_live_prefix_state",
                        runner_group,
                        participant_index,
                        runner.get(),
                        "child_snapshot_invalid");
                    return false;
                }

                if (!have_common_tokens)
                {
                    common_tokens = child.cached_tokens;
                    have_common_tokens = true;
                }
                else if (child.cached_tokens != common_tokens)
                {
                    rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                        "capture_live_prefix_state",
                        runner_group,
                        participant_index,
                        runner.get(),
                        "child_token_count_mismatch",
                        common_tokens,
                        child.cached_tokens);
                    return false;
                }

                if (!have_common_provenance)
                {
                    common_provenance = child.provenance;
                    have_common_provenance = true;
                }
                else if (child.provenance != common_provenance)
                {
                    same_provenance = false;
                }

                aggregate.participant_snapshots.push_back(std::move(child));
                ++participant_index;
            }
            return true;
        };

        if (!capture_runners(device_runners_, "device") ||
            !capture_runners(pp_stage_runners_, "pp"))
        {
            return {};
        }
        if (!saw_runner || !have_common_tokens)
        {
            rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                "capture_live_prefix_state",
                "rank",
                0,
                nullptr,
                saw_runner ? "missing_common_token_count" : "no_participants");
            return {};
        }

        aggregate.valid = true;
        aggregate.cached_tokens = common_tokens;
        aggregate.provenance = same_provenance ? common_provenance
                                               : PrefixStateProvenance::PayloadCheckpoint;
        return aggregate;
    }

    PrefixStateSnapshot RankOrchestrator::captureLivePrefixCheckpoint(int seq_idx) const
    {
        PrefixStateSnapshot aggregate;
        bool saw_runner = false;
        bool have_common_tokens = false;
        bool have_common_provenance = false;
        bool same_provenance = true;
        bool all_logical = true;
        int common_tokens = 0;
        PrefixStateProvenance common_provenance = PrefixStateProvenance::Unknown;

        auto capture_runners = [&](
                                   const std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                   const char *runner_group)
        {
            size_t participant_index = 0;
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;

                saw_runner = true;
                PrefixStateSnapshot child = runner->captureLivePrefixCheckpoint(seq_idx);
                if (!child.valid)
                {
                    rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                        "capture_live_prefix_checkpoint",
                        runner_group,
                        participant_index,
                        runner.get(),
                        "child_checkpoint_invalid");
                    return false;
                }

                if (!have_common_tokens)
                {
                    common_tokens = child.cached_tokens;
                    have_common_tokens = true;
                }
                else if (child.cached_tokens != common_tokens)
                {
                    rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                        "capture_live_prefix_checkpoint",
                        runner_group,
                        participant_index,
                        runner.get(),
                        "child_token_count_mismatch",
                        common_tokens,
                        child.cached_tokens);
                    return false;
                }

                all_logical = all_logical && child.logical_checkpoint;
                if (!have_common_provenance)
                {
                    common_provenance = child.provenance;
                    have_common_provenance = true;
                }
                else if (child.provenance != common_provenance)
                {
                    same_provenance = false;
                }
                aggregate.participant_snapshots.push_back(std::move(child));
                ++participant_index;
            }
            return true;
        };

        if (!capture_runners(device_runners_, "device") ||
            !capture_runners(pp_stage_runners_, "pp"))
        {
            return {};
        }
        if (!saw_runner || !have_common_tokens)
        {
            rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                "capture_live_prefix_checkpoint",
                "rank",
                0,
                nullptr,
                saw_runner ? "missing_common_token_count" : "no_participants");
            return {};
        }

        aggregate.valid = true;
        aggregate.logical_checkpoint = all_logical;
        aggregate.cached_tokens = common_tokens;
        aggregate.provenance = same_provenance
                                   ? common_provenance
                                   : (all_logical
                                          ? PrefixStateProvenance::LogicalCheckpoint
                                          : PrefixStateProvenance::PayloadCheckpoint);
        return aggregate;
    }

    bool RankOrchestrator::restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx)
    {
        if (!snapshot.valid || snapshot.participant_snapshots.empty())
            return false;

        size_t participant_index = 0;
        bool ok = true;
        bool saw_runner = false;
        auto restore_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;

                saw_runner = true;
                if (participant_index >= snapshot.participant_snapshots.size())
                {
                    ok = false;
                    continue;
                }
                ok = runner->restoreLivePrefixState(
                         snapshot.participant_snapshots[participant_index++],
                         seq_idx) &&
                     ok;
            }
        };

        restore_runners(device_runners_);
        restore_runners(pp_stage_runners_);
        if (!(saw_runner && ok && participant_index == snapshot.participant_snapshots.size()))
            return false;

        /*
         * A prefix restore is atomic at the rank boundary as well as at each
         * child runner.  The children own the actual KV/GDN payloads, but rank
         * orchestration still owns public position/sequence bookkeeping used by
         * diagnostics, PP mode, and any code path that cannot delegate directly
         * to a single child.  Keep that aggregate state in lockstep with the
         * restored child snapshots so repeated restore/replay cycles cannot
         * observe stale parent metadata.
         */
        current_position_ = snapshot.cached_tokens;
        current_padded_seq_len_ = 0;
        current_sequence_lengths_.assign(
            static_cast<size_t>(std::max(1, config_.batch_size)),
            snapshot.cached_tokens);
        stats_dirty_ = true;
        return true;
    }

    bool RankOrchestrator::truncateLivePrefixState(int cached_tokens, int seq_idx)
    {
        bool saw_runner = false;
        bool ok = true;
        auto truncate_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                saw_runner = true;
                ok = runner->truncateLivePrefixState(cached_tokens, seq_idx) && ok;
            }
        };

        truncate_runners(device_runners_);
        truncate_runners(pp_stage_runners_);
        return saw_runner && ok;
    }

    PrefixRuntimeStateSnapshot RankOrchestrator::prefixStateProbe() const
    {
        PrefixRuntimeStateSnapshot snapshot;
        snapshot.initialized = true;
        snapshot.architecture = architecture();
        snapshot.execution_path = "rank-orchestrator";
        snapshot.current_position = get_position();
        snapshot.positions = {snapshot.current_position};
        snapshot.sequence_lengths = current_sequence_lengths_;

        auto merge_child = [&snapshot](const PrefixRuntimeStateSnapshot &child)
        {
            snapshot.initialized = snapshot.initialized && child.initialized;
            snapshot.has_hidden = snapshot.has_hidden || child.has_hidden;
            snapshot.has_logits = snapshot.has_logits || child.has_logits;
            snapshot.session_epoch = std::max(snapshot.session_epoch, child.session_epoch);
            snapshot.moe_runtime_movement_epoch =
                std::max(snapshot.moe_runtime_movement_epoch,
                         child.moe_runtime_movement_epoch);
            snapshot.prefix_cache_config_enabled =
                snapshot.prefix_cache_config_enabled || child.prefix_cache_config_enabled;
            snapshot.prefix_cache_ready = snapshot.prefix_cache_ready || child.prefix_cache_ready;
            snapshot.prefix_cache_bypassed =
                snapshot.prefix_cache_bypassed || child.prefix_cache_bypassed;
            if (snapshot.prefix_cache_bypass_reason.empty() &&
                !child.prefix_cache_bypass_reason.empty())
            {
                snapshot.prefix_cache_bypass_reason = child.prefix_cache_bypass_reason;
            }
            snapshot.prefix_cache_lookups += child.prefix_cache_lookups;
            snapshot.prefix_cache_hits += child.prefix_cache_hits;
            snapshot.prefix_cache_partial_hits += child.prefix_cache_partial_hits;
            snapshot.prefix_cache_misses += child.prefix_cache_misses;
            snapshot.prefix_cache_matched_blocks += child.prefix_cache_matched_blocks;
            snapshot.prefix_cache_matched_tokens += child.prefix_cache_matched_tokens;
            snapshot.prefix_cache_stores += child.prefix_cache_stores;
            snapshot.prefix_cache_inserts += child.prefix_cache_inserts;
            snapshot.prefix_cache_evictions += child.prefix_cache_evictions;
            snapshot.prefix_cache_promotions += child.prefix_cache_promotions;
            snapshot.prefix_cache_disk_hydrations += child.prefix_cache_disk_hydrations;
            snapshot.prefix_cache_terminal_state_hits += child.prefix_cache_terminal_state_hits;
            snapshot.prefix_cache_ram_bytes += child.prefix_cache_ram_bytes;
            snapshot.prefix_cache_device_bytes += child.prefix_cache_device_bytes;
            snapshot.prefix_cache_disk_bytes += child.prefix_cache_disk_bytes;
            snapshot.prefix_cache_hybrid_state_bytes += child.prefix_cache_hybrid_state_bytes;
            snapshot.prefix_cache_mtp_state_bytes += child.prefix_cache_mtp_state_bytes;
            snapshot.prefix_cache_bypasses += child.prefix_cache_bypasses;
            snapshot.prefix_cache_unsupported_backend_bypasses +=
                child.prefix_cache_unsupported_backend_bypasses;
            snapshot.prefix_cache_fingerprint_bypasses +=
                child.prefix_cache_fingerprint_bypasses;
            snapshot.prefix_cache_terminal_state_bypasses +=
                child.prefix_cache_terminal_state_bypasses;
            snapshot.mtp_config_enabled = snapshot.mtp_config_enabled || child.mtp_config_enabled;
            snapshot.mtp_bypassed = snapshot.mtp_bypassed || child.mtp_bypassed;
            if (snapshot.mtp_bypass_reason.empty() && !child.mtp_bypass_reason.empty())
            {
                snapshot.mtp_bypass_reason = child.mtp_bypass_reason;
            }
            snapshot.mtp_draft_steps += child.mtp_draft_steps;
            snapshot.mtp_accepted_tokens += child.mtp_accepted_tokens;
            snapshot.mtp_rejected_tokens += child.mtp_rejected_tokens;
            snapshot.mtp_rollbacks += child.mtp_rollbacks;
            snapshot.mtp_bypasses += child.mtp_bypasses;
            snapshot.mtp_verifier_runs += child.mtp_verifier_runs;
            snapshot.mtp_verifier_token_count += child.mtp_verifier_token_count;
            snapshot.mtp_depth_policy_windows += child.mtp_depth_policy_windows;
            snapshot.mtp_depth_policy_updates += child.mtp_depth_policy_updates;
            snapshot.mtp_depth_policy_promotions += child.mtp_depth_policy_promotions;
            snapshot.mtp_depth_policy_demotions += child.mtp_depth_policy_demotions;
            snapshot.mtp_depth_policy_observe_recommendations +=
                child.mtp_depth_policy_observe_recommendations;
            snapshot.mtp_current_depth =
                snapshot.mtp_current_depth == 0
                    ? child.mtp_current_depth
                    : std::min(snapshot.mtp_current_depth, child.mtp_current_depth);
            snapshot.mtp_min_depth =
                snapshot.mtp_min_depth == 0
                    ? child.mtp_min_depth
                    : std::min(snapshot.mtp_min_depth, child.mtp_min_depth);
            snapshot.mtp_max_depth =
                snapshot.mtp_max_depth == 0
                    ? child.mtp_max_depth
                    : std::min(snapshot.mtp_max_depth, child.mtp_max_depth);
            snapshot.prefill_chunk_schedules += child.prefill_chunk_schedules;
            snapshot.prefill_chunk_successful_schedules +=
                child.prefill_chunk_successful_schedules;
            snapshot.prefill_chunks += child.prefill_chunks;
            snapshot.prefill_chunk_real_tokens += child.prefill_chunk_real_tokens;
            snapshot.prefill_chunk_padded_tokens += child.prefill_chunk_padded_tokens;
            snapshot.prefill_chunk_failures += child.prefill_chunk_failures;
            if (snapshot.primary_device.is_cpu() && !child.primary_device.is_cpu())
            {
                snapshot.primary_device = child.primary_device;
            }
            if (snapshot.positions.empty() && !child.positions.empty())
            {
                snapshot.positions = child.positions;
            }
            if (snapshot.sequence_lengths.empty() && !child.sequence_lengths.empty())
            {
                snapshot.sequence_lengths = child.sequence_lengths;
            }
            snapshot.kv_caches.insert(snapshot.kv_caches.end(), child.kv_caches.begin(), child.kv_caches.end());
            snapshot.mtp_kv_caches.insert(snapshot.mtp_kv_caches.end(),
                                          child.mtp_kv_caches.begin(),
                                          child.mtp_kv_caches.end());
            snapshot.gdn_layers.insert(snapshot.gdn_layers.end(), child.gdn_layers.begin(), child.gdn_layers.end());
            snapshot.prefill_graphs.insert(snapshot.prefill_graphs.end(),
                                           child.prefill_graphs.begin(),
                                           child.prefill_graphs.end());
        };

        bool saw_child = false;
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                merge_child(runner->prefixStateProbe());
                saw_child = true;
            }
        }
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                merge_child(runner->prefixStateProbe());
                saw_child = true;
            }
        }

        if (!saw_child)
        {
            snapshot.initialized = false;
        }
        if (snapshot.prefix_cache_bypassed)
        {
            snapshot.prefix_cache_ready = false;
        }

        return snapshot;
    }

    // =========================================================================
    // Hidden State API (for Pipeline Parallelism nesting)
    // =========================================================================

    TensorBase *RankOrchestrator::getHiddenState()
    {
        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, all devices have same hidden state after allreduce
            // Delegate to primary device runner
            if (!device_runners_.empty() && device_runners_[0])
            {
                return device_runners_[0]->getHiddenState();
            }
        }
        else
        {
            // PP or TP_PP mode - last stage has the final hidden state
            if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
            {
                return pp_stage_runners_.back()->getHiddenState();
            }
        }
        return nullptr;
    }

    const TensorBase *RankOrchestrator::getHiddenState() const
    {
        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, all devices have same hidden state after allreduce
            // Delegate to primary device runner
            if (!device_runners_.empty() && device_runners_[0])
            {
                return device_runners_[0]->getHiddenState();
            }
        }
        else
        {
            // PP or TP_PP mode - last stage has the final hidden state
            if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
            {
                return pp_stage_runners_.back()->getHiddenState();
            }
        }
        return nullptr;
    }

    void RankOrchestrator::setHiddenState(TensorBase *hidden_state)
    {
        hidden_state_input_ = hidden_state;

        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, set on ALL device runners (they all need the same input)
            for (auto &runner : device_runners_)
            {
                if (runner)
                {
                    runner->setHiddenState(hidden_state);
                }
            }
        }
        else
        {
            // PP or TP_PP mode - set on first stage runner (stage 0 receives external input)
            if (!pp_stage_runners_.empty() && pp_stage_runners_.front())
            {
                pp_stage_runners_.front()->setHiddenState(hidden_state);
            }
        }
    }

    bool RankOrchestrator::hasHiddenStateInput() const
    {
        return hidden_state_input_ != nullptr;
    }

    void RankOrchestrator::clearHiddenStateInput()
    {
        hidden_state_input_ = nullptr;

        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, clear on all device runners
            for (auto &runner : device_runners_)
            {
                if (runner)
                {
                    runner->clearHiddenStateInput();
                }
            }
        }
        else
        {
            // PP or TP_PP mode - clear on first stage runner
            if (!pp_stage_runners_.empty() && pp_stage_runners_.front())
            {
                pp_stage_runners_.front()->clearHiddenStateInput();
            }
        }
    }

    // =========================================================================
    // Snapshot API
    // =========================================================================

    void RankOrchestrator::enableSnapshotCapture(const std::string &output_dir)
    {
        LOG_DEBUG("RankOrchestrator::enableSnapshotCapture: Enabling on all devices");

        // Enable on TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->enableSnapshotCapture(output_dir);
            }
        }

        // Enable on PP stage runners
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->enableSnapshotCapture(output_dir);
            }
        }
    }

    void RankOrchestrator::setSnapshotCaptureFilter(const std::vector<std::string> &keys)
    {
        for (auto &runner : device_runners_)
        {
            if (runner)
                runner->setSnapshotCaptureFilter(keys);
        }

        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
                runner->setSnapshotCaptureFilter(keys);
        }
    }

    void RankOrchestrator::disableSnapshotCapture()
    {
        LOG_DEBUG("RankOrchestrator::disableSnapshotCapture: Disabling on all devices");

        // Disable on TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->disableSnapshotCapture();
            }
        }

        // Disable on PP stage runners
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->disableSnapshotCapture();
            }
        }
    }

    void RankOrchestrator::clearSnapshots()
    {
        LOG_DEBUG("RankOrchestrator::clearSnapshots: Clearing on all devices");

        // Clear on TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->clearSnapshots();
            }
        }

        // Clear on PP stage runners
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->clearSnapshots();
            }
        }
    }

    bool RankOrchestrator::phaseSplitReplicatedDecodeSnapshotsActive() const
    {
        if (current_padded_seq_len_ != 1)
            return false;

        const auto &plan = config_.moe_expert_parallel_plan;
        if (!plan || !plan->isTieredOverlay())
            return false;

        const auto &dense = plan->continuation_domain_spec;
        return dense.dense_tp_enabled && dense.dense_decode_replicated;
    }

    bool RankOrchestrator::isPhaseSplitReplicatedDecodeSnapshotKey(const std::string &key)
    {
        const std::string stage_type = extractStageType(key);

        return stage_type == "ATTENTION_OUTPUT" ||
               stage_type == "FFN_DOWN" ||
               stage_type == "GDN_OUTPUT" ||
               stage_type == "MOE_SHARED_EXPERT_OUTPUT" ||
               stage_type == "MOE_SHARED_GATE_OUTPUT";
    }

    SnapshotShardingMode RankOrchestrator::resolveSnapshotShardingMode(const std::string &key) const
    {
        SnapshotShardingMode mode = getStageShardingMode(key, stage_sharding_map_);

        /*
         * Phase-split MoE overlay uses DenseTP for prefill but switches dense and
         * always-on shared decode work to full replicated weights. The schema still
         * marks dense output projections as ROW_PARALLEL because that is correct
         * for plain TP. During replicated decode, summing per-device snapshots
         * doubles the captured tensor and produces false parity failures.
         */
        if (mode == SnapshotShardingMode::ROW_PARALLEL &&
            phaseSplitReplicatedDecodeSnapshotsActive() &&
            isPhaseSplitReplicatedDecodeSnapshotKey(key))
        {
            return SnapshotShardingMode::REPLICATED;
        }

        return mode;
    }

    const float *RankOrchestrator::getSnapshot(const std::string &key, size_t &out_size) const
    {
        // For GATHERED stages (e.g., LM_HEAD) with multi-device TP, return the gathered combined_logits.
        // This is necessary because each device only has logits_local with vocab_local entries,
        // but tests expect the full vocab_size logits.
        //
        // CRITICAL: For hybrid PP+TP, only the stage that actually HAS the LM head should
        // return combined_logits. Otherwise, a nested TP stage (like PP stage 0) that has
        // logits_local buffers allocated but never computes LM_HEAD would return stale data.
        if (getStageShardingMode(key, stage_sharding_map_) == SnapshotShardingMode::GATHERED &&
            device_runners_.size() > 1 && logits_gatherer_ && logits_gatherer_->isAllocated() && tp_ctx_)
        {
            bool owns_lm_head = true;
            if (config_.nested_pp_stage_config.has_value())
            {
                owns_lm_head = config_.nested_pp_stage_config->has_lm_head;
            }
            else
            {
                auto wm = model_ctx_->weightManager();
                owns_lm_head = wm && wm->hasLMHead();
            }

            if (!owns_lm_head)
            {
                LOG_DEBUG("RankOrchestrator::getSnapshot LM_HEAD: this stage doesn't own LM_HEAD, "
                          << "falling through to PP stage search"
                          << " (nested_pp_config=" << config_.nested_pp_stage_config.has_value()
                          << " has_lm_head=" << (config_.nested_pp_stage_config.has_value() ? config_.nested_pp_stage_config->has_lm_head : false) << ")");
            }
            else
            {
                bool has_column_parallel_lm_head = false;
                for (const auto &runner : device_runners_)
                {
                    if (runner && runner->hasLogitsLocal())
                    {
                        has_column_parallel_lm_head = true;
                        break;
                    }
                }

                if (has_column_parallel_lm_head)
                {
                    out_size = logits_gatherer_->lastGatheredSize();
                    const float *ptr = logits_gatherer_->data();
                    LOG_DEBUG("RankOrchestrator::getSnapshot LM_HEAD returning combined_logits with "
                              << out_size << " elements (column-parallel gathering), ptr=" << (void *)ptr
                              << " first_element=" << (ptr ? ptr[0] : -999999.0f));
                    return ptr;
                }
            }
        }

        // PP mode: search across all PP stage runners
        if (!pp_stage_runners_.empty())
        {
            for (const auto &runner : pp_stage_runners_)
            {
                if (runner)
                {
                    const float *result = runner->getSnapshot(key, out_size);
                    if (result != nullptr)
                    {
                        return result;
                    }
                }
            }
            out_size = 0;
            return nullptr;
        }

        // Default (TP mode): prefer the primary device, then fall back to any
        // participant that actually published the key. Some graph diagnostics
        // are intentionally participant-local; getSnapshotKeys() merges all
        // participants, so returning only primary would silently drop those keys.
        if (!device_runners_.empty() && device_runners_[0])
        {
            const float *primary = device_runners_[0]->getSnapshot(key, out_size);
            if (primary)
                return primary;
        }
        for (size_t i = 1; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                continue;
            const float *participant = device_runners_[i]->getSnapshot(key, out_size);
            if (participant)
                return participant;
        }
        out_size = 0;
        return nullptr;
    }

    std::vector<std::string> RankOrchestrator::getSnapshotKeys() const
    {
        // Merge keys from all devices (use set to deduplicate)
        std::set<std::string> all_keys;

        // Collect from TP device runners
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                auto keys = runner->getSnapshotKeys();
                all_keys.insert(keys.begin(), keys.end());
            }
        }

        // Collect from PP stage runners
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                auto keys = runner->getSnapshotKeys();
                all_keys.insert(keys.begin(), keys.end());
            }
        }

        return std::vector<std::string>(all_keys.begin(), all_keys.end());
    }

    SnapshotInfo RankOrchestrator::getSnapshotWithShape(const std::string &key) const
    {
        // PP mode: search across all PP stage runners
        if (!pp_stage_runners_.empty())
        {
            for (const auto &runner : pp_stage_runners_)
            {
                if (runner)
                {
                    auto snap = runner->getSnapshotWithShape(key);
                    if (snap)
                        return snap;
                }
            }
            return {};
        }

        // Default (TP mode): prefer primary, then search participants for
        // participant-local diagnostics that were included in merged keys.
        if (!device_runners_.empty() && device_runners_[0])
        {
            auto primary = device_runners_[0]->getSnapshotWithShape(key);
            if (primary)
                return primary;
        }
        for (size_t i = 1; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                continue;
            auto participant = device_runners_[i]->getSnapshotWithShape(key);
            if (participant)
                return participant;
        }
        return {};
    }

    TPSnapshot RankOrchestrator::getTPSnapshot(const std::string &key) const
    {
        TPSnapshot result;
        result.key = key;
        result.mode = resolveSnapshotShardingMode(key);
        result.tp_degree = static_cast<int>(device_runners_.size());

        LOG_DEBUG("RankOrchestrator::getTPSnapshot: key=" << key
                                                          << " mode=" << shardingModeToString(result.mode)
                                                          << " tp_degree=" << result.tp_degree);

        // =========================================================================
        // PP Mode: Delegate to appropriate PP stage runner
        // For PP+TP hybrid, each PP stage runner may be an MDO for TP
        // =========================================================================
        if (!pp_stage_runners_.empty())
        {
            // Find which PP stage owns this snapshot key
            for (size_t stage_idx = 0; stage_idx < pp_stage_runners_.size(); ++stage_idx)
            {
                if (!pp_stage_runners_[stage_idx])
                    continue;

                // Check if this stage has the snapshot (with shape metadata)
                auto snap = pp_stage_runners_[stage_idx]->getSnapshotWithShape(key);
                if (!snap)
                    continue;

                // Found the owning stage - check if it's a TP domain (RankOrchestrator)
                auto *inner_mdo = dynamic_cast<const RankOrchestrator *>(
                    pp_stage_runners_[stage_idx].get());

                if (inner_mdo)
                {
                    // This PP stage is a TP domain - delegate to its getTPSnapshot
                    LOG_DEBUG("RankOrchestrator::getTPSnapshot: PP stage " << stage_idx
                                                                           << " is TP domain, delegating getTPSnapshot for key=" << key);
                    return inner_mdo->getTPSnapshot(key);
                }
                else
                {
                    // Single-device PP stage - wrap the snapshot as non-TP
                    LOG_DEBUG("RankOrchestrator::getTPSnapshot: PP stage " << stage_idx
                                                                           << " is single device for key=" << key);

                    result.tp_degree = 1;
                    result.mode = SnapshotShardingMode::REPLICATED;

                    DeviceSnapshotData dev_data;
                    dev_data.device_index = 0;
                    dev_data.rows = snap.rows;
                    dev_data.cols = snap.cols;
                    dev_data.global_start_col = 0;
                    dev_data.global_total_cols = snap.cols;
                    dev_data.data.assign(snap.data, snap.data + snap.size);

                    result.device_data.push_back(std::move(dev_data));
                    result.combined_valid = true;
                    result.combined_data = result.device_data[0].data;
                    result.combined_rows = snap.rows;
                    result.combined_cols = snap.cols;

                    return result;
                }
            }

            // Key not found in any PP stage
            LOG_DEBUG("RankOrchestrator::getTPSnapshot: key=" << key
                                                              << " not found in any PP stage");
            return result;
        }

        // =========================================================================
        // TP Mode (non-PP): Collect from device_runners_
        // =========================================================================

        // Special case: GATHERED stages (e.g., LM_HEAD) with combined logits already gathered
        if (result.mode == SnapshotShardingMode::GATHERED &&
            device_runners_.size() > 1 && logits_gatherer_ && logits_gatherer_->isAllocated() && tp_ctx_)
        {
            bool has_column_parallel_lm_head = false;
            for (const auto &runner : device_runners_)
            {
                if (runner && runner->hasLogitsLocal())
                {
                    has_column_parallel_lm_head = true;
                    break;
                }
            }

            if (has_column_parallel_lm_head)
            {
                size_t gathered_size = logits_gatherer_->lastGatheredSize();
                const float *gathered_ptr = logits_gatherer_->data();
                DeviceSnapshotData gathered;
                gathered.device_id = GlobalDeviceId::gpu(0, 0, config_.devices[0].device_type);
                gathered.device_index = 0;
                gathered.rows = 1;
                gathered.cols = gathered_size;
                gathered.global_start_col = 0;
                gathered.global_total_cols = gathered_size;
                gathered.data.assign(gathered_ptr, gathered_ptr + gathered_size);

                result.device_data.push_back(std::move(gathered));
                result.combined_valid = true;
                result.combined_data = result.device_data[0].data;
                result.combined_rows = result.device_data[0].rows;
                result.combined_cols = result.device_data[0].cols;

                LOG_DEBUG("RankOrchestrator::getTPSnapshot: LM_HEAD using combined_logits "
                          << "size=" << gathered_size);
                return result;
            }
        }

        // Collect snapshots from all device runners
        size_t global_col_offset = 0;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                continue;

            // Use shape-aware snapshot retrieval — the stage itself reported
            // its output rows/cols via getDumpInfo() at capture time, so we
            // don't need model-specific dimension calculations here.
            auto snap = device_runners_[i]->getSnapshotWithShape(key);

            if (!snap)
            {
                LOG_DEBUG("RankOrchestrator::getTPSnapshot: device " << i
                                                                     << " has no data for key=" << key);
                continue;
            }

            // Debug: log data pointer and first 4 values to verify each device has different data
            LOG_DEBUG("RankOrchestrator::getTPSnapshot: device " << i
                                                                 << " key=" << key << " ptr=" << static_cast<const void *>(snap.data)
                                                                 << " size=" << snap.size
                                                                 << " shape=[" << snap.rows << "x" << snap.cols << "]"
                                                                 << " val[0-3]=" << snap.data[0] << "," << snap.data[1] << "," << snap.data[2] << "," << snap.data[3]);

            DeviceSnapshotData dev_data;
            // Use device type from config if available, otherwise default to CUDA
            DeviceType dev_type = DeviceType::CUDA;
            if (i < config_.devices.size())
            {
                dev_type = config_.devices[i].device_type;
            }
            dev_data.device_id = GlobalDeviceId::gpu(0, static_cast<int>(i), dev_type);
            dev_data.device_index = static_cast<int>(i);
            dev_data.data.assign(snap.data, snap.data + snap.size);

            // Use shape metadata from the stage's getDumpInfo() output.
            // This is model-agnostic: stages report their own dimensions
            // (e.g., KV projections report kv_dim cols, FFN reports d_ff cols).
            if (result.mode == SnapshotShardingMode::COLUMN_PARALLEL)
            {
                dev_data.rows = snap.rows;
                dev_data.cols = snap.cols;
                dev_data.global_start_col = global_col_offset;
                global_col_offset += snap.cols;
            }
            else
            {
                // Replicated or row-parallel - each device has full output
                dev_data.rows = snap.rows;
                dev_data.cols = snap.cols;
                dev_data.global_start_col = 0;
                dev_data.global_total_cols = snap.cols;
            }

            LOG_DEBUG("RankOrchestrator::getTPSnapshot: device " << i
                                                                 << " size=" << snap.size
                                                                 << " cols=" << dev_data.cols
                                                                 << " start_col=" << dev_data.global_start_col);

            result.device_data.push_back(std::move(dev_data));
        }

        // Set global_total_cols for column-parallel stages
        if (result.mode == SnapshotShardingMode::COLUMN_PARALLEL && !result.device_data.empty())
        {
            size_t total_cols = global_col_offset;
            for (auto &dev : result.device_data)
            {
                dev.global_total_cols = total_cols;
            }
        }

        return result;
    }

    std::vector<std::pair<std::string, SnapshotShardingMode>>
    RankOrchestrator::getSnapshotKeysWithSharding() const
    {
        auto keys = getSnapshotKeys();
        std::vector<std::pair<std::string, SnapshotShardingMode>> result;
        result.reserve(keys.size());

        for (const auto &key : keys)
        {
            result.emplace_back(key, resolveSnapshotShardingMode(key));
        }

        return result;
    }

    // =========================================================================
    // Profiling API
    // =========================================================================

    const GraphExecutorStats *RankOrchestrator::executorStats() const
    {
        aggregateStats();
        return aggregated_stats_.get();
    }

    void RankOrchestrator::resetExecutorStats()
    {
        LOG_DEBUG("RankOrchestrator::resetExecutorStats: Resetting on all devices");

        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->resetExecutorStats();
            }
        }

        if (aggregated_stats_)
        {
            aggregated_stats_->reset();
        }
        tp_decode_stats_.reset();
        stats_dirty_ = true;
    }

    // =========================================================================
    // Orchestration API
    // =========================================================================

    bool RankOrchestrator::hasPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->hasPlacementPlan();
        }
        return false;
    }

    const PlacementPlan &RankOrchestrator::getPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getPlacementPlan();
        }
        throw std::runtime_error("No placement plan available: no device runners");
    }

    // =========================================================================
    // IRankOrchestrator Interface Implementation
    // =========================================================================

    int RankOrchestrator::device_count() const
    {
        return static_cast<int>(device_runners_.size());
    }

    IInferenceRunner *RankOrchestrator::deviceRunner(int device_idx)
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    const IInferenceRunner *RankOrchestrator::deviceRunner(int device_idx) const
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    ILocalTPContext *RankOrchestrator::localTPContext()
    {
        return tp_ctx_.get();
    }

    const ILocalTPContext *RankOrchestrator::localTPContext() const
    {
        return tp_ctx_.get();
    }

    bool RankOrchestrator::allDevicesReady() const
    {
        if (device_runners_.empty())
        {
            return false;
        }

        for (const auto &runner : device_runners_)
        {
            if (!runner)
            {
                return false;
            }
            // Check if runner is ready (has vocab_size > 0 indicates initialization)
            if (runner->vocab_size() <= 0)
            {
                return false;
            }
        }

        return true;
    }

    void RankOrchestrator::synchronizeDevices()
    {
        LOG_DEBUG("RankOrchestrator::synchronizeDevices: Synchronizing all devices");

        if (tp_ctx_)
        {
            tp_ctx_->synchronize();
        }

        auto synchronize_runner_device =
            [](const std::unique_ptr<IInferenceRunner> &runner)
        {
            if (!runner)
                return;
            const DeviceId device = runner->primaryDeviceId();
            if (!device.is_gpu())
                return;

            IBackend *backend = getBackendFor(device);
            if (!backend)
            {
                LOG_WARN("RankOrchestrator::synchronizeDevices: Backend unavailable for "
                         << device.toString());
                return;
            }
            if (!backend->synchronize(device.gpu_ordinal()))
            {
                LOG_WARN("RankOrchestrator::synchronizeDevices: Device synchronization failed for "
                         << device.toString());
            }
        };

        for (const auto &runner : device_runners_)
        {
            synchronize_runner_device(runner);
        }
        for (const auto &runner : pp_stage_runners_)
        {
            if (auto *rank = dynamic_cast<RankOrchestrator *>(runner.get()))
            {
                rank->synchronizeDevices();
                continue;
            }
            synchronize_runner_device(runner);
        }
    }

    MoERebalanceController *RankOrchestrator::moeRebalanceController() const
    {
        auto controllers = moeRebalanceControllers();
        return selectActiveMoERebalanceController(controllers);
    }

    std::vector<MoERebalanceController *> RankOrchestrator::moeRebalanceControllers() const
    {
        std::vector<MoERebalanceController *> controllers;
        auto append_from = [&](const std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;
                for (auto *controller : runner->moeRebalanceControllers())
                {
                    if (controller &&
                        std::find(controllers.begin(), controllers.end(), controller) == controllers.end())
                    {
                        controllers.push_back(controller);
                    }
                }
            }
        };

        append_from(device_runners_);
        append_from(pp_stage_runners_);
        return controllers;
    }

    MoERebalanceController *RankOrchestrator::moeRebalanceControllerForDomain(
        const std::string &domain_id) const
    {
        for (auto *controller : moeRebalanceControllers())
        {
            if (controller && controller->domainId() == domain_id)
                return controller;
        }
        return nullptr;
    }

    bool RankOrchestrator::usesDeviceSideMoERebalanceController() const
    {
        const auto &runners = pp_stage_runners_.empty()
                                  ? device_runners_
                                  : pp_stage_runners_;
        if (runners.empty())
            return false;

        for (const auto &runner : runners)
        {
            if (!runner || !runner->usesDeviceSideMoERebalanceController())
                return false;
        }
        return true;
    }

    bool RankOrchestrator::applyMoEExpertMasksForAllDevices(
        const MoERebalanceController &controller,
        const ExpertReplicaSet *replica_arrivals,
        const std::vector<int> *previous_ownership_placement)
    {
        auto snapshot = snapshotMoEExpertMasksForAllDevices(
            controller,
            replica_arrivals,
            previous_ownership_placement);
        return applyMoEExpertMasksForAllDevices(
            snapshot.masks_by_participant,
            snapshot.domain_id,
            snapshot.transferMasks());
    }

    RankOrchestrator::MoEExpertMaskSnapshot RankOrchestrator::snapshotMoEExpertMasksForAllDevices(
        const MoERebalanceController &controller,
        const ExpertReplicaSet *replica_arrivals,
        const std::vector<int> *previous_ownership_placement) const
    {
        MoEExpertMaskSnapshot snapshot;
        snapshot.domain_id = controller.domainId();

        const int gpu_cache_experts = debugEnv().moe_rebalance.gpu_cache_experts_per_layer;
        if (gpu_cache_experts > 0)
        {
            snapshot.masks_by_participant = controller.computeGpuCacheExpertMasks(gpu_cache_experts);
            return snapshot;
        }

        snapshot.masks_by_participant.reserve(device_runners_.size());
        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
            snapshot.masks_by_participant.push_back(controller.computeExpertMasksForParticipant(static_cast<int>(device_idx)));

        if (replica_arrivals)
        {
            snapshot.transfer_masks_by_participant =
                rank_orchestrator_detail::buildReplicaArrivalTransferMasks(
                    controller,
                    *replica_arrivals);
        }
        else if (previous_ownership_placement)
        {
            snapshot.transfer_masks_by_participant =
                rank_orchestrator_detail::buildOwnershipArrivalTransferMasks(
                    controller,
                    *previous_ownership_placement);
        }

        return snapshot;
    }

    RankOrchestrator::PreparedMoEExpertMaskUpdate RankOrchestrator::prepareMoEExpertMaskTransfersForAllDevices(
        const std::vector<std::vector<std::vector<bool>>> &masks_by_participant,
        const std::string &domain_id,
        const std::vector<std::vector<std::vector<bool>>> *transfer_masks_by_participant)
    {
        using Clock = std::chrono::steady_clock;
        int direct_transfer_pair_batches = 0;
        int serialized_transfer_pair_batches = 0;
        size_t transfer_mask_entries = 0;
        const std::string stats_device = primaryDeviceId().is_valid()
                                             ? primaryDeviceId().toString()
                                             : std::string{};
        const PerfStatsCollector::Tags phase_tags{{"domain_id", domain_id}};

        std::vector<DeviceGraphOrchestrator *> local_dgos;
        local_dgos.reserve(device_runners_.size());
        for (auto &runner : device_runners_)
            local_dgos.push_back(dynamic_cast<DeviceGraphOrchestrator *>(runner.get()));
        std::vector<bool> gpu_direct_prepare_ok(local_dgos.size(), true);
        for (auto *dgo : local_dgos)
        {
            if (dgo)
                dgo->clearPendingGpuDirectExpertTransfers();
        }

        auto count_mask_entries = [](const std::vector<std::vector<bool>> &masks) -> size_t
        {
            size_t count = 0;
            for (const auto &layer_mask : masks)
                count += static_cast<size_t>(std::count(layer_mask.begin(), layer_mask.end(), true));
            return count;
        };

        auto collect_local_transfers = [&](size_t destination_idx,
                                           const std::vector<std::vector<bool>> &target_masks,
                                           const std::vector<std::vector<bool>> &transfer_masks)
        {
            ReceivedWeightsMap merged;
            (void)target_masks;
            auto direct_remaining_masks = transfer_masks;
            auto serialized_remaining_masks = transfer_masks;
            bool direct_only_gpu_transfer_attempted = false;
            for (size_t source_idx = 0; source_idx < local_dgos.size(); ++source_idx)
            {
                if (source_idx == destination_idx || !local_dgos[source_idx])
                    continue;

                auto is_same_backend_gpu_pair = [&]()
                {
                    if (!local_dgos[destination_idx] || !local_dgos[source_idx])
                        return false;
                    const DeviceId dst_device = local_dgos[destination_idx]->primaryDeviceId();
                    const DeviceId src_device = local_dgos[source_idx]->primaryDeviceId();
                    return rank_orchestrator_detail::sameBackendGpuExpertTransferIsDirectOnly(
                        dst_device,
                        src_device);
                };
                const bool same_backend_gpu_pair = is_same_backend_gpu_pair();
                auto &remaining_masks =
                    same_backend_gpu_pair ? direct_remaining_masks : serialized_remaining_masks;
                const size_t requested_entries = count_mask_entries(remaining_masks);
                if (requested_entries == 0)
                    continue;

                if (same_backend_gpu_pair)
                {
                    ++direct_transfer_pair_batches;
                }
                else
                {
                    ++serialized_transfer_pair_batches;
                }
                transfer_mask_entries += requested_entries;

                if (same_backend_gpu_pair)
                {
                    direct_only_gpu_transfer_attempted = true;
                    if (local_dgos[destination_idx])
                    {
                        auto prepared_direct =
                            local_dgos[destination_idx]->prepareExpertWeightsDirectForMasksFrom(
                                *local_dgos[source_idx],
                                remaining_masks);
                        remaining_masks = std::move(prepared_direct.remaining_masks);
                    }
                    continue;
                }

                auto source_blobs = local_dgos[source_idx]->collectExpertWeightsForMasks(remaining_masks);
                for (auto &layer_entry : source_blobs)
                {
                    auto &dst_layer = merged[layer_entry.first];
                    for (auto &expert_entry : layer_entry.second)
                    {
                        if (dst_layer.find(expert_entry.first) == dst_layer.end())
                        {
                            dst_layer.emplace(expert_entry.first, std::move(expert_entry.second));
                            if (layer_entry.first >= 0 &&
                                layer_entry.first < static_cast<int>(remaining_masks.size()) &&
                                expert_entry.first >= 0 &&
                                expert_entry.first < static_cast<int>(remaining_masks[static_cast<size_t>(layer_entry.first)].size()))
                            {
                                remaining_masks[static_cast<size_t>(layer_entry.first)][static_cast<size_t>(expert_entry.first)] = false;
                            }
                        }
                    }
                }
            }
            if (direct_only_gpu_transfer_attempted &&
                count_mask_entries(direct_remaining_masks) > 0 &&
                destination_idx < gpu_direct_prepare_ok.size())
            {
                gpu_direct_prepare_ok[destination_idx] = false;
                LOG_ERROR("[RankOrchestrator] Same-backend GPU expert transfer prepare left "
                          << count_mask_entries(direct_remaining_masks)
                          << " arrival mask entries unsatisfied for participant "
                          << destination_idx << " domain=" << domain_id);
            }
            return merged;
        };

        std::vector<ReceivedWeightsMap> received_by_device(device_runners_.size());
        const auto transfer_start = Clock::now();
        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx < masks_by_participant.size() && local_dgos[device_idx])
            {
                const auto &transfer_masks =
                    (transfer_masks_by_participant &&
                     device_idx < transfer_masks_by_participant->size())
                        ? (*transfer_masks_by_participant)[device_idx]
                        : masks_by_participant[device_idx];
                received_by_device[device_idx] =
                    collect_local_transfers(
                        device_idx,
                        masks_by_participant[device_idx],
                        transfer_masks);
            }
        }
        const auto transfer_end = Clock::now();

        PerfStatsCollector::recordTimingNs(
            "moe_rebalance",
            "rank_local_transfer_prepare",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      transfer_end - transfer_start)
                                      .count()),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_local_transfer_mask_entries",
            static_cast<double>(transfer_mask_entries),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_local_direct_transfer_pair_batches",
            static_cast<double>(direct_transfer_pair_batches),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_local_serialized_transfer_pair_batches",
            static_cast<double>(serialized_transfer_pair_batches),
            "rebalance",
            stats_device,
            phase_tags);

        PreparedMoEExpertMaskUpdate prepared;
        prepared.domain_id = domain_id;
        prepared.masks_by_participant = masks_by_participant;
        prepared.received_by_device = std::move(received_by_device);
        prepared.gpu_direct_prepare_ok_by_device = std::move(gpu_direct_prepare_ok);
        return prepared;
    }

    bool RankOrchestrator::publishPreparedMoEExpertMasksForAllDevices(
        const PreparedMoEExpertMaskUpdate &prepared)
    {
        using Clock = std::chrono::steady_clock;
        int applied = 0;
        const std::string stats_device = primaryDeviceId().is_valid()
                                             ? primaryDeviceId().toString()
                                             : std::string{};
        const PerfStatsCollector::Tags phase_tags{{"domain_id", prepared.domain_id}};

        std::vector<DeviceGraphOrchestrator *> local_dgos;
        local_dgos.reserve(device_runners_.size());
        for (auto &runner : device_runners_)
            local_dgos.push_back(dynamic_cast<DeviceGraphOrchestrator *>(runner.get()));

        const auto publish_start = Clock::now();
        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx >= prepared.masks_by_participant.size())
                continue;
            auto *dgo = local_dgos[device_idx];
            if (!dgo)
                continue;
            if (device_idx < prepared.gpu_direct_prepare_ok_by_device.size() &&
                !prepared.gpu_direct_prepare_ok_by_device[device_idx])
            {
                LOG_ERROR("[RankOrchestrator] Aborting MoE expert mask publish for domain="
                          << prepared.domain_id
                          << " because same-backend GPU-direct staging was incomplete for participant "
                          << device_idx);
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "rank_local_mask_publish_aborts",
                    1.0,
                    "rebalance",
                    stats_device,
                    {{"domain_id", prepared.domain_id},
                     {"reason", "incomplete_gpu_direct_prepare"}});
                for (auto *pending_dgo : local_dgos)
                {
                    if (pending_dgo)
                        pending_dgo->clearPendingGpuDirectExpertTransfers();
                }
                return false;
            }
        }

        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx >= prepared.masks_by_participant.size())
                continue;

            auto *dgo = local_dgos[device_idx];
            if (!dgo)
                continue;
            if (!dgo->activatePendingGpuDirectExpertTransfers())
            {
                LOG_ERROR("[RankOrchestrator] Aborting MoE expert mask publish for domain="
                          << prepared.domain_id
                          << " because staged GPU-direct activation failed for participant "
                          << device_idx);
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "rank_local_mask_publish_aborts",
                    1.0,
                    "rebalance",
                    stats_device,
                    {{"domain_id", prepared.domain_id},
                     {"reason", "gpu_direct_activation_failed"}});
                for (auto *pending_dgo : local_dgos)
                {
                    if (pending_dgo)
                        pending_dgo->clearPendingGpuDirectExpertTransfers();
                }
                return false;
            }
        }

        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx >= prepared.masks_by_participant.size())
                continue;

            auto *dgo = local_dgos[device_idx];
            if (!dgo)
                continue;

            const ReceivedWeightsMap empty_received;
            const auto &received =
                device_idx < prepared.received_by_device.size()
                    ? prepared.received_by_device[device_idx]
                    : empty_received;
            dgo->applyExpertMasksForDomain(
                prepared.domain_id,
                prepared.masks_by_participant[device_idx],
                received);
            ++applied;
        }

        if (!pp_stage_runners_.empty())
        {
            for (auto &runner : pp_stage_runners_)
            {
                if (auto *rank = dynamic_cast<RankOrchestrator *>(runner.get()))
                {
                    if (!rank->applyMoEExpertMasksForAllDevices(
                        prepared.masks_by_participant,
                        prepared.domain_id))
                    {
                        LOG_ERROR("[RankOrchestrator] Nested LocalTP MoE expert mask publish failed for domain="
                                  << prepared.domain_id);
                        return false;
                    }
                    ++applied;
                    continue;
                }
                if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                {
                    if (!prepared.masks_by_participant.empty())
                    {
                        dgo->applyExpertMasksForDomain(
                            prepared.domain_id,
                            prepared.masks_by_participant.front(),
                            ReceivedWeightsMap{});
                        ++applied;
                    }
                }
            }
        }
        const auto publish_end = Clock::now();
        PerfStatsCollector::recordTimingNs(
            "moe_rebalance",
            "rank_local_mask_publish",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      publish_end - publish_start)
                                      .count()),
            "rebalance",
            stats_device,
            phase_tags);
        LOG_DEBUG("[RankOrchestrator] Applied LocalTP MoE expert masks to "
                  << applied << "/" << device_runners_.size() << " device runners");
        return true;
    }

    void RankOrchestrator::clearPendingGpuDirectExpertTransfersForAllDevices()
    {
        for (auto &runner : device_runners_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                dgo->clearPendingGpuDirectExpertTransfers();
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (auto *rank = dynamic_cast<RankOrchestrator *>(runner.get()))
            {
                rank->clearPendingGpuDirectExpertTransfersForAllDevices();
                continue;
            }
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                dgo->clearPendingGpuDirectExpertTransfers();
        }
    }

    bool RankOrchestrator::applyMoEExpertMasksForAllDevices(
        const std::vector<std::vector<std::vector<bool>>> &masks_by_participant,
        const std::string &domain_id,
        const std::vector<std::vector<std::vector<bool>>> *transfer_masks_by_participant)
    {
        auto prepared = prepareMoEExpertMaskTransfersForAllDevices(
            masks_by_participant,
            domain_id,
            transfer_masks_by_participant);
        return publishPreparedMoEExpertMasksForAllDevices(prepared);
    }

    void RankOrchestrator::setExpertReplicaSetForAllDevices(const ExpertReplicaSet &replicas)
    {
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_replica_set_device_broadcasts",
            static_cast<double>(device_runners_.size()),
            "rebalance",
            primaryDeviceId().toString(),
            {{"domain_id", replicas.domain_id},
             {"replicas", std::to_string(replicas.num_replicated)}});

        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(device_runners_[device_idx].get());
            if (dgo)
                dgo->setExpertReplicaSet(replicas, static_cast<int>(device_idx));
        }
    }

    void RankOrchestrator::setSuppressTimeline(bool suppress)
    {
        for (auto &runner : device_runners_)
            runner->setSuppressTimeline(suppress);
    }

    void RankOrchestrator::setAccumulatePrefill(bool accumulate)
    {
        for (auto &runner : device_runners_)
            runner->setAccumulatePrefill(accumulate);
    }

    void RankOrchestrator::flushStageTimeline()
    {
        // Print per-device GPU stage timelines
        for (auto &runner : device_runners_)
            runner->flushStageTimeline();

        // Print TP orchestrator decode summary if profiling data was collected
        if (tp_decode_stats_.iterations > 0)
        {
            const size_t n = tp_decode_stats_.iterations;
            const double avg_wall = tp_decode_stats_.total_wall_ms / n;
            const double avg_dispatch = tp_decode_stats_.total_dispatch_ms / n;
            const double avg_wait = tp_decode_stats_.total_wait_ms / n;
            const double avg_gather = tp_decode_stats_.total_gather_ms / n;
            // "Other" captures orchestrator overhead not attributed to
            // dispatch/wait/gather: exception checking, position tracking,
            // condition evaluation, etc.
            const double avg_other = avg_wall - avg_dispatch - avg_wait - avg_gather;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row
            {
                std::ostringstream title;
                title << "TP DECODE ORCHESTRATOR PROFILING"
                      << " (avg of " << n << " decode steps, "
                      << device_runners_.size() << " devices)";
                table << title.str() << "" << "" << fort::endr;
                table[0][0].set_cell_span(3);
                table[0][0].set_cell_text_align(fort::text_align::center);
            }

            // Summary
            {
                std::ostringstream info;
                info << std::fixed << std::setprecision(2);
                info << "Wall Avg: " << avg_wall << " ms/tok";
                if (avg_wall > 0)
                    info << "  |  " << std::setprecision(1) << (1000.0 / avg_wall) << " tok/s";
                table << info.str() << "" << "" << fort::endr;
                table[1][0].set_cell_span(3);
            }

            // Header
            table << fort::header << "PHASE" << "AVG (ms)" << "%" << fort::endr;
            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);

            auto fmt_row = [&](const char *name, double ms)
            {
                std::ostringstream ms_str, pct_str;
                ms_str << std::fixed << std::setprecision(3) << ms;
                double pct = avg_wall > 0 ? (ms / avg_wall) * 100.0 : 0.0;
                pct_str << std::fixed << std::setprecision(1) << pct << "%";
                table << name << ms_str.str() << pct_str.str() << fort::endr;
            };

            fmt_row("Dispatch (wake workers)", avg_dispatch);
            fmt_row("Wait (device forward)", avg_wait);
            fmt_row("Gather logits", avg_gather);
            if (avg_other > 0.001)
                fmt_row("Other overhead", avg_other);

            // Separator + total
            table << fort::separator;
            {
                std::ostringstream ms_str;
                ms_str << std::fixed << std::setprecision(3) << avg_wall;
                table << "TOTAL" << ms_str.str() << "100.0%" << fort::endr;
            }

            std::cout << table.to_string() << std::flush;
            tp_decode_stats_.reset();
        }
    }

} // namespace llaminar2
