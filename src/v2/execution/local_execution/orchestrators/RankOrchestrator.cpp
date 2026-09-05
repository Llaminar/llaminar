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

#include "transfer/MappedTransferProgressEpoch.h"
#include "LogitsGatherer.h"
#include "DeviceSampler.h"
#include "DeviceGraphOrchestrator.h"
#include "PipelineGraphExecutionPlan.h"
#include "../../../collective/CollectiveTimeoutPolicy.h"
#include "../../mtp/MTPSpecTransactionDriver.h"
#include "../../mtp/MTPSpecStateContract.h"
#include "../../factory/InferenceRunnerFactory.h"
#include "../../moe/MoEExpertOverlayRuntimePlan.h"
#include "../../moe/MoEOverlayNodeLocalRouteExchange.h"
#include "../../moe/MoEOverlayInferenceTransactionService.h"
#include "../../moe/MoEOverlayInferenceInterferenceProbe.h"
#include "../engine/PrefillBucketUtils.h"
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
#include "../../../planning/CollectiveMemoryEstimator.h"
#include "../graph/SchemaFactoryRegistry.h" // Model-agnostic sharding config access
#include "../../../tensors/TensorClasses.h"
#include "../../../tensors/TensorFactory.h"
#include "../../../backends/BackendManager.h"       // getBackendFor() for partial D2H in gatherLogits
#include "../../../backends/ComputeBackend.h"       // exact driver-backed P2P topology
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
#include <chrono>
#include <future>
#include <iomanip>
#include <limits>
#include <numeric>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#ifdef __linux__
#endif

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Resolve the exact packed LocalTP layout of one GDN checkpoint.
         *
         * Production preparation partitions Q, K, and every modulo-linked V
         * repeat together. The semantic stage selects whether the captured row
         * contains fused Q/K/V vectors, value vectors, or value scalars. The
         * retired replicated-Q/K GPU representation is deliberately rejected.
         *
         * @param device_data Captured participant rows with stable TP indices.
         * @param tp_degree Number of participants required by the LocalTP cell.
         * @param stage_type Schema-facing semantic checkpoint name.
         * @param key_heads Global GDN Q/K head count.
         * @param value_heads Global GDN V head count.
         * @param state_width Elements per GDN head.
         * @param error Optional diagnostic populated when no unique layout exists.
         * @return Ordered Q/K/V group descriptors, or an empty vector on error.
         */
        std::vector<SnapshotColumnGroup> resolvePackedGDNColumnGroups(
            const std::vector<DeviceSnapshotData> &device_data,
            int tp_degree,
            const std::string &stage_type,
            int key_heads,
            int value_heads,
            int state_width,
            std::string *error)
        {
            const auto fail = [error](const std::string &message)
            {
                if (error)
                    *error = message;
                return std::vector<SnapshotColumnGroup>{};
            };

            if (tp_degree <= 0)
                return fail("invalid LocalTP degree");
            if (device_data.size() != static_cast<size_t>(tp_degree))
            {
                return fail("not every LocalTP participant published the packed checkpoint");
            }

            std::vector<size_t> local_row_widths(
                static_cast<size_t>(tp_degree), 0);
            std::vector<bool> participant_seen(
                static_cast<size_t>(tp_degree), false);
            for (const auto &device : device_data)
            {
                if (device.device_index < 0 || device.device_index >= tp_degree)
                    return fail("captured packed checkpoint has an out-of-range TP index");
                const size_t participant =
                    static_cast<size_t>(device.device_index);
                if (participant_seen[participant])
                    return fail("captured packed checkpoint has a duplicate TP index");
                participant_seen[participant] = true;
                local_row_widths[participant] = device.cols;
            }

            ModuloLinkedGDNSnapshotLayout layout;
            if (stage_type == "QKV_PROJECTION" ||
                stage_type == "GDN_CONV1D_OUTPUT")
            {
                layout = ModuloLinkedGDNSnapshotLayout::FusedQKV;
            }
            else if (stage_type == "GDN_Z_PROJECTION" ||
                     stage_type == "GDN_RECURRENCE" ||
                     stage_type == "GDN_DELTA_RULE_OUTPUT" ||
                     stage_type == "GATED_RMSNORM" ||
                     stage_type == "GDN_NORM_GATE_OUTPUT")
            {
                layout = ModuloLinkedGDNSnapshotLayout::ValueVector;
            }
            else if (stage_type == "GDN_ALPHA" || stage_type == "GDN_BETA")
            {
                layout = ModuloLinkedGDNSnapshotLayout::ValueScalar;
            }
            else
            {
                return fail(
                    "semantic stage has no modulo-linked GDN snapshot layout: " +
                    stage_type);
            }

            return resolveModuloLinkedGDNSnapshotColumnGroups(
                local_row_widths,
                key_heads,
                value_heads,
                state_width,
                layout,
                error);
        }

        /**
         * @brief Compute a stable diagnostic hash for one terminal token row.
         *
         * The hash is never used for correctness decisions; exact vector
         * equality remains mandatory. It lets production diagnostics compare
         * long responses without printing hundreds of tokens while still
         * reporting the first byte-level disagreement separately.
         */
        uint64_t terminalTokenHash(std::span<const int32_t> tokens) noexcept
        {
            uint64_t hash = 1469598103934665603ULL;
            for (const int32_t token : tokens)
            {
                const uint32_t bits = static_cast<uint32_t>(token);
                for (unsigned int byte = 0; byte < sizeof(bits); ++byte)
                {
                    hash ^= static_cast<uint8_t>(bits >> (byte * 8U));
                    hash *= 1099511628211ULL;
                }
            }
            return hash;
        }

        /**
         * @brief Describe every byte-level boundary that differs by participant.
         *
         * Mirrored-state capture is an exceptional-path diagnostic performed
         * only after production has already rejected divergent participant
         * results. Keeping comparison and formatting here gives both the
         * immediate draft sampler and the fully captured terminal parent the
         * same ordered interpretation of device state.
         */
        std::string describeMirroredDigestMismatch(
            const std::vector<std::vector<MTPMirroredTensorDigest>> &participant_digests)
        {
            std::ostringstream summary;
            summary << " Mirrored MTP device-state digest comparison:";
            const size_t boundary_count =
                participant_digests.empty()
                    ? 0
                    : participant_digests.front().size();
            for (size_t boundary = 0; boundary < boundary_count; ++boundary)
            {
                const auto &reference =
                    participant_digests.front()[boundary];
                bool differs = !reference.available;
                for (size_t participant = 1;
                     participant < participant_digests.size();
                     ++participant)
                {
                    if (boundary >= participant_digests[participant].size())
                    {
                        differs = true;
                        continue;
                    }
                    const auto &candidate =
                        participant_digests[participant][boundary];
                    differs =
                        differs ||
                        !candidate.available ||
                        candidate.name != reference.name ||
                        candidate.byte_count != reference.byte_count ||
                        candidate.hash != reference.hash;
                }
                if (!differs)
                    continue;

                summary << " [" << reference.name;
                for (size_t participant = 0;
                     participant < participant_digests.size();
                     ++participant)
                {
                    summary << " p" << participant << "=";
                    if (boundary >= participant_digests[participant].size() ||
                        !participant_digests[participant][boundary].available)
                    {
                        summary << "unavailable";
                        continue;
                    }
                    const auto &digest =
                        participant_digests[participant][boundary];
                    summary << "0x" << std::hex << digest.hash << std::dec
                            << "/" << digest.byte_count << "B";
                    if (!digest.value_preview.empty())
                        summary << digest.value_preview;
                }
                summary << "]";
            }
            return summary.str();
        }

        /**
         * @brief Describe the first exact mismatch between mirrored ledgers.
         *
         * Mirrored LocalTP participants must produce byte-identical terminal
         * responses. A generic operator== failure does not reveal whether the
         * divergence originated in sampling, stop handling, or controller
         * accounting, so this helper reports the first token mismatch and all
         * controller fields for the affected request.
         */
        std::string describeTerminalLedgerMismatch(
            const std::vector<DeviceGenerationTerminalRequestResult> &primary,
            const std::vector<DeviceGenerationTerminalRequestResult> &participant)
        {
            std::ostringstream detail;
            detail << " primary_requests=" << primary.size()
                   << " participant_requests=" << participant.size();
            const size_t shared_requests =
                std::min(primary.size(), participant.size());
            for (size_t request = 0; request < shared_requests; ++request)
            {
                const auto &lhs = primary[request];
                const auto &rhs = participant[request];
                if (lhs == rhs)
                    continue;

                const size_t shared_tokens =
                    std::min(lhs.tokens.size(), rhs.tokens.size());
                size_t first_token_mismatch = shared_tokens;
                for (size_t token = 0; token < shared_tokens; ++token)
                {
                    if (lhs.tokens[token] != rhs.tokens[token])
                    {
                        first_token_mismatch = token;
                        break;
                    }
                }

                detail << " request=" << request
                       << " primary_token_count=" << lhs.tokens.size()
                       << " participant_token_count=" << rhs.tokens.size()
                       << " primary_token_hash="
                       << terminalTokenHash(lhs.tokens)
                       << " participant_token_hash="
                       << terminalTokenHash(rhs.tokens)
                       << " first_token_mismatch=";
                if (first_token_mismatch < shared_tokens)
                {
                    detail << first_token_mismatch
                           << " primary_token="
                           << lhs.tokens[first_token_mismatch]
                           << " participant_token="
                           << rhs.tokens[first_token_mismatch];
                }
                else if (lhs.tokens.size() != rhs.tokens.size())
                {
                    detail << shared_tokens << " value=missing";
                }
                else
                {
                    detail << "none";
                }

                detail << " primary_control={remaining="
                       << lhs.remaining_token_count
                       << ",stopped=" << lhs.model_stopped
                       << ",transactions=" << lhs.transaction_count
                       << ",accepted="
                       << lhs.accepted_speculative_token_count
                       << ",rejected=" << lhs.rejected_transaction_count
                       << ",verifier_rows="
                       << lhs.consumed_verifier_row_count
                       << ",state_commits="
                       << lhs.published_state_commit_count << "}"
                       << " participant_control={remaining="
                       << rhs.remaining_token_count
                       << ",stopped=" << rhs.model_stopped
                       << ",transactions=" << rhs.transaction_count
                       << ",accepted="
                       << rhs.accepted_speculative_token_count
                       << ",rejected=" << rhs.rejected_transaction_count
                       << ",verifier_rows="
                       << rhs.consumed_verifier_row_count
                       << ",state_commits="
                       << rhs.published_state_commit_count << "}";
                return detail.str();
            }
            return detail.str();
        }

        /**
         * @brief Test-factory resolver that prevents unit tests from touching GPU backends.
         *
         * `RankOrchestrator::createForTest()` is used with injected mock runners
         * and GPU-shaped `DeviceId`s to exercise orchestration policy.  Unit tests
         * must not page-lock host memory or initialize CUDA/HIP contexts, so rank
         * gatherers created by that factory resolve no backend and copy from the
         * mock tensors already owned by the injected runners.
         */
        IBackend *resolveNoBackendForInjectedUnitTest(DeviceId)
        {
            return nullptr;
        }
    } // namespace

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
            const int participants = arrivals.participantCount();

            const int num_layers = controller.numLayers();
            const int num_experts = controller.numExperts();
            std::vector<std::vector<std::vector<bool>>> masks_by_participant(
                static_cast<size_t>(std::max(0, participants)),
                std::vector<std::vector<bool>>(
                    static_cast<size_t>(std::max(0, num_layers)),
                    std::vector<bool>(static_cast<size_t>(std::max(0, num_experts)), false)));

            if (participants <= 0 || num_layers <= 0 || num_experts <= 0)
                return masks_by_participant;

            for (int participant = 0; participant < participants; ++participant)
            {
                for (int layer = 0; layer < num_layers; ++layer)
                {
                    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
                    {
                        const int owner =
                            arrivals.ownerParticipant(layer, expert_id);
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
            const MoELayeredExpertOwnership &previous_ownership)
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

            const auto &current_ownership = controller.currentOwnership();
            for (const auto &change :
                 current_ownership.changesFrom(previous_ownership))
            {
                if (change.current_participant < 0 ||
                    change.current_participant >= participants)
                {
                    throw std::logic_error(
                        "Layered MoE ownership change targets an invalid participant");
                }
                masks_by_participant
                    [static_cast<size_t>(change.current_participant)]
                    [static_cast<size_t>(change.layer_idx)]
                    [static_cast<size_t>(change.expert_id)] = true;
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
            const std::vector<int32_t> &output_tokens,
            const std::array<int, sampling_math::kSpeculativeBatchMetaCount> &meta,
            DeviceSpeculativeVerifyBatchOutcome *out)
        {
            using namespace sampling_math;
            if (!out ||
                meta[kSpecBatchMetaOk] == 0 ||
                meta[kSpecBatchMetaOutputCount] < 0 ||
                meta[kSpecBatchMetaOutputCount] >
                    static_cast<int>(output_tokens.size()))
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
            out->commit_boundary_clipped =
                meta[kSpecBatchMetaCommitBoundaryClipped] != 0;
            return true;
        }

        const RoutedExpertDomain *findMoEExpertDomain(
            const MoERoutedExpertPlacementPlan &plan,
            const std::string &name)
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                                   [&](const auto &domain)
                                   {
                                       return domain.name == name;
                                   });
            return it == plan.domains.end() ? nullptr : &*it;
        }

        bool routedOverlayUsesExpertIdApportionment(const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan)
        {
            if (!plan || !plan->usesExpertOverlayAuthority())
                return false;

            for (const auto &tier : plan->routed_tiers)
            {
                const auto *domain = findMoEExpertDomain(*plan, tier.domain);
                if (domain &&
                    domain->routed_compute_policy == RoutedExpertComputePolicy::Apportioned)
                    return true;
            }
            return false;
        }

        bool moeOverlayDenseTPEnabled(const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan)
        {
            if (!plan || !plan->usesExpertOverlayAuthority())
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

        if (prepared_weight_admission ==
                PreparedWeightAdmission::ReuseCertifiedCompleteSet &&
            !prepared_weight_store)
        {
            LOG_ERROR(
                "RankOrchestrator::Config: certified prepared-weight reuse "
                "requires the exact model-owned PreparedWeightStore");
            return false;
        }

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
        config.resident_graph_rows = plan.runtime.resident_graph_rows;
        config.batch_size = plan.runtime.batch_size;
        config.activation_precision = plan.runtime.activation_precision;
        config.kv_cache_precision = plan.runtime.kv_cache_precision;
        config.tp_allreduce_precision_override =
            plan.runtime.tp_allreduce_precision_override;
        config.prefix_cache = plan.runtime.prefix_cache;
        config.mtp = plan.runtime.mtp;
        config.routed_expert_compute_policy = plan.runtime.routed_expert_compute_policy;
        config.routed_expert_owner_order =
            plan.runtime.routed_expert_owner_order;
        config.moe_hot_expert_cache = plan.runtime.moe_hot_expert_cache;
        config.moe_routed_prefill = plan.runtime.moe_routed_prefill;
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
        orchestrator->initializePPGraphExecutionPlan();

        /*
         * Injected unit-test runners stand in for already prepared endpoints.
         * The public production constructor never takes this shortcut: its
         * native children cross the real materialization transition before
         * request admission.
         */
        std::string materialization_error;
        if (!orchestrator->pp_graph_execution_plan_->markMaterialized(
                orchestrator->pp_graph_execution_plan_->nativeSegmentCount(),
                &materialization_error))
        {
            throw std::runtime_error(materialization_error);
        }
        return orchestrator;
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    RankOrchestrator::RankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        const Config &config,
        std::unique_ptr<ILocalTPContext> tp_ctx)
        : model_ctx_(std::move(model_ctx)),
          config_(config),
          current_batch_size_(std::max(1, config.batch_size))
    {
        if (!config_.validate())
        {
            throw std::invalid_argument("Invalid RankOrchestrator configuration");
        }
        initializeRankMTPScratch();

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
          config_(config),
          logits_backend_resolver_(resolveNoBackendForInjectedUnitTest),
          external_device_backend_access_enabled_(false),
          current_batch_size_(std::max(1, config.batch_size))
    {
        initializeRankMTPScratch();

        // Initialize stage sharding map from model architecture (if registered)
        const auto arch = model_ctx_->architecture();
        if (SchemaFactoryRegistry::isSupported(arch))
        {
            stage_sharding_map_ = SchemaFactoryRegistry::getStageShardingConfig(arch);
            applyRuntimeSnapshotShardingOverrides();
        }

        LOG_DEBUG("RankOrchestrator: Created via createForTest with "
                  << device_runners_.size() << " injected device runners");

        if (ownsMainForwardLogits() && model_ctx_ && !device_runners_.empty())
        {
            int vocab = vocab_size();
            if (vocab > 0)
            {
                size_t max_tokens = static_cast<size_t>(config_.batch_size) *
                                    static_cast<size_t>(config_.max_seq_len);
                logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    vocab,
                    max_tokens,
                    logits_backend_resolver_);

                if (device_runners_.size() > 1)
                {
                    DeviceId primary_dev = device_runners_[0]->primaryDeviceId();
                    if (external_device_backend_access_enabled_ &&
                        primary_dev.is_gpu())
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

    void RankOrchestrator::initializeRankMTPScratch()
    {
        rank_max_draft_depth_ = resolveMTPMaximumDraftDepth(config_.mtp);
        rank_max_verifier_rows_ = rank_max_draft_depth_ + 1;
        rank_compact_output_token_stride_ = rank_max_verifier_rows_;

        /*
         * Target slots include one bonus row per request. Reusing that larger
         * capacity for the rank's draft-token bookkeeping costs only a handful
         * of integers and keeps every slot validation on one explicit bound.
         */
        rank_stochastic_slot_capacity_ =
            std::max(1, resolveMTPMaxTargetQueryRows(config_.mtp));

        rank_compact_output_tokens_.assign(
            static_cast<size_t>(rank_compact_output_token_stride_),
            -1);
        const size_t distribution_entry_capacity =
            static_cast<size_t>(rank_stochastic_slot_capacity_) *
            static_cast<size_t>(sampling_math::kMaxTopK);
        rank_stochastic_target_token_ids_.assign(
            distribution_entry_capacity,
            -1);
        rank_stochastic_target_probs_.assign(
            distribution_entry_capacity,
            0.0f);
        rank_stochastic_target_top_k_.assign(
            static_cast<size_t>(rank_stochastic_slot_capacity_),
            0);
        rank_stochastic_target_sample_tokens_.assign(
            static_cast<size_t>(rank_stochastic_slot_capacity_),
            -1);
        rank_stochastic_draft_sample_tokens_.assign(
            static_cast<size_t>(rank_stochastic_slot_capacity_),
            -1);
        rank_mirrored_target_distribution_ready_.assign(
            static_cast<size_t>(rank_stochastic_slot_capacity_),
            false);
    }

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
         * stream. Keep every semantic snapshot from projection through the
         * effective attention input in lockstep. In particular, per-head
         * K_NORM is still a full replica even though Q_NORM remains sharded by
         * query head; concatenating those K replicas invents TP-degree times as
         * many logical KV columns as the model owns.
         */
        static constexpr std::array<std::string_view, 10>
            kReplicatedGQAKVSnapshotStageTypes = {
            "K_PROJECTION",
            "V_PROJECTION",
            "K_NORM",
            "K_ROPE",
            "KV_APPEND_SOURCE_K",
            "KV_APPEND_SOURCE_V",
            "KV_CACHE_K",
            "KV_CACHE_V",
            "ATTENTION_EFFECTIVE_K",
            "ATTENTION_EFFECTIVE_V",
        };

        for (const std::string_view stage_type :
             kReplicatedGQAKVSnapshotStageTypes)
        {
            stage_sharding_map_[std::string(stage_type)] =
                SnapshotShardingMode::REPLICATED;
        }

        LOG_DEBUG(
            "RankOrchestrator: runtime GQA snapshot policy treats the complete "
            "K/V checkpoint family as replicated (n_kv_heads="
            << model_ctx_->headCountKV()
            << " < tp_degree=" << tp_ctx_->degree() << ")");
    }

    bool RankOrchestrator::ownsMainForwardLogits() const noexcept
    {
        return !config_.nested_pp_stage_config.has_value() ||
               config_.nested_pp_stage_config->has_lm_head;
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
        LOG_DEBUG(
            "RankOrchestrator: continuation route transport candidate"
            << " overlay_plan="
            << (config_.moe_routed_expert_plan != nullptr)
            << " devices=" << devices.size()
            << " preinstalled="
            << (config_.moe_node_local_route_exchange != nullptr)
            << " requested_transport="
            << moeOverlayNodeLocalRouteTransportName(
                   config_.moe_node_local_route_transport)
            << " dense_policy="
            << (config_.moe_routed_expert_plan
                    ? denseParallelPolicyToString(
                          config_.moe_routed_expert_plan
                              ->continuation_domain_spec
                              .effectiveDensePolicy())
                    : "none")
            << " dense_tp_enabled="
            << (config_.moe_routed_expert_plan &&
                denseParallelPolicyEnablesTP(
                    config_.moe_routed_expert_plan
                        ->continuation_domain_spec
                        .effectiveDensePolicy())));

        /*
         * Routed-expert ownership is independent of dense tensor parallelism.
         * Every multi-device continuation therefore resolves one explicit
         * publication transport even when dense weights are replicated. Match
         * the declaration by device identity rather than vendor, ordinal,
         * socket, rank, or declaration order; auxiliary expert ranks do not
         * match and consequently own no continuation-local transport.
         */
        if (config_.moe_routed_expert_plan && devices.size() > 1u)
        {
            const auto &overlay_plan = *config_.moe_routed_expert_plan;
            const auto continuation_domain = std::find_if(
                overlay_plan.domains.begin(), overlay_plan.domains.end(),
                [&](const RoutedExpertDomain &domain)
                {
                    return domain.name ==
                           overlay_plan.continuation_domain_spec.domain;
                });
            LOG_DEBUG(
                "RankOrchestrator: continuation route domain lookup"
                << " requested="
                << overlay_plan.continuation_domain_spec.domain
                << " found="
                << (continuation_domain != overlay_plan.domains.end())
                << " domain_participants="
                << (continuation_domain != overlay_plan.domains.end()
                        ? continuation_domain->participants.size()
                        : 0u)
                << " cell_participants=" << devices.size());
            if (continuation_domain != overlay_plan.domains.end() &&
                continuation_domain->participants.size() == devices.size())
            {
                std::vector<DeviceId> cell_devices;
                std::vector<std::string> cell_keys;
                std::vector<std::string> continuation_keys;
                cell_devices.reserve(devices.size());
                cell_keys.reserve(devices.size());
                continuation_keys.reserve(devices.size());
                bool all_gpu = true;
                for (const auto &address : devices)
                {
                    const DeviceId local = address.toLocalDeviceId();
                    all_gpu = all_gpu && local.is_gpu();
                    cell_devices.push_back(local);
                    cell_keys.push_back(local.toString());
                }
                for (const auto &address : continuation_domain->participants)
                    continuation_keys.push_back(
                        address.toLocalDeviceId().toString());
                std::sort(cell_keys.begin(), cell_keys.end());
                std::sort(
                    continuation_keys.begin(), continuation_keys.end());

                LOG_DEBUG(
                    "RankOrchestrator: continuation route topology match"
                    << " domain=" << continuation_domain->name
                    << " all_gpu=" << all_gpu
                    << " cell_devices=" << cell_keys.size()
                    << " continuation_devices="
                    << continuation_keys.size()
                    << " identity_match="
                    << (cell_keys == continuation_keys));

                const int root_index = overlay_plan
                                           .continuation_domain_spec
                                           .logical_root_participant;
                if (all_gpu && cell_keys == continuation_keys &&
                    root_index >= 0 &&
                    root_index < static_cast<int>(
                                     continuation_domain->participants.size()))
                {
                    const bool homogeneous = std::all_of(
                        cell_devices.begin() + 1,
                        cell_devices.end(),
                        [&](const DeviceId &candidate)
                        {
                            return candidate.type ==
                                   cell_devices.front().type;
                        });
                    std::optional<PeerAccessCoverage> peer_coverage;
                    if (homogeneous)
                    {
                        peer_coverage =
                            DeviceManager::instance().peerAccessCoverage(
                                cell_devices);
                    }

                    MoEOverlayNodeLocalRouteTransport selected_transport =
                        config_.moe_node_local_route_transport;
                    if (selected_transport ==
                        MoEOverlayNodeLocalRouteTransport::Unresolved)
                    {
                        selected_transport =
                            selectMoEOverlayNodeLocalRouteTransport(
                                cell_devices, peer_coverage);
                    }
                    else if (!homogeneous)
                    {
                        const auto measured =
                            selectMoEOverlayNodeLocalRouteTransport(
                                cell_devices, std::nullopt);
                        if (selected_transport != measured)
                        {
                            throw std::runtime_error(
                                "RankOrchestrator: preselected continuation route transport contradicts heterogeneous device topology");
                        }
                    }
                    else if (peer_coverage)
                    {
                        const auto measured =
                            selectMoEOverlayNodeLocalRouteTransport(
                                cell_devices, peer_coverage);
                        if (selected_transport != measured)
                        {
                            throw std::runtime_error(
                                "RankOrchestrator: preselected continuation route transport contradicts driver-reported P2P topology");
                        }
                    }
                    else if (!config_.moe_node_local_route_exchange)
                    {
                        throw std::runtime_error(
                            "RankOrchestrator: homogeneous continuation route transport has no driver-backed P2P matrix");
                    }
                    config_.moe_node_local_route_transport =
                        selected_transport;

                    const DeviceId root_device =
                        continuation_domain
                            ->participants[static_cast<std::size_t>(root_index)]
                            .toLocalDeviceId();

                    if (selected_transport ==
                        MoEOverlayNodeLocalRouteTransport::MappedSparse)
                    {
                        if (!config_.moe_node_local_route_exchange)
                        {
                            config_.moe_node_local_route_exchange =
                                std::make_shared<
                                    MoEOverlayNodeLocalRouteExchange>(
                                    MoEOverlayNodeLocalRouteExchange::Config{
                                        .devices = cell_devices,
                                        .root_device = root_device,
                                        .identity =
                                            "continuation:" +
                                            continuation_domain->name,
                                    });
                        }
                    }
                    else if (config_.moe_node_local_route_exchange)
                    {
                        throw std::runtime_error(
                            "RankOrchestrator: native continuation route transport cannot retain a mapped sparse exchange");
                    }

                    const std::string peer_access =
                        homogeneous
                            ? (peer_coverage
                                   ? peerAccessCoverageName(*peer_coverage)
                                   : "preselected")
                            : "heterogeneous";
                    const PerfStatsCollector::Tags transport_tags{
                        {"domain", continuation_domain->name},
                        {"transport",
                         moeOverlayNodeLocalRouteTransportName(
                             selected_transport)},
                        {"peer_access", peer_access},
                        {"backend",
                         homogeneous
                             ? (cell_devices.front().is_cuda() ? "cuda"
                                                                : "rocm")
                             : "heterogeneous"},
                        {"participants", std::to_string(devices.size())}};
                    PerfStatsCollector::addCounter(
                        "moe_overlay_transport",
                        "selection",
                        1.0,
                        {},
                        root_device.toString(),
                        transport_tags);
                    LOG_INFO(
                        "RankOrchestrator: selected continuation route "
                        "transport="
                        << moeOverlayNodeLocalRouteTransportName(
                               selected_transport)
                        << " peer_access=" << peer_access << " domain="
                        << continuation_domain->name
                        << " root=" << root_device.toString()
                        << " participants=" << devices.size());
                }
            }
        }

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
                    moeOverlayDenseTPEnabled(config_.moe_routed_expert_plan);

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
                if (routedOverlayUsesExpertIdApportionment(config_.moe_routed_expert_plan))
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

                /*
                 * Runtime snapshot layout was frozen from this same model and
                 * TP topology before device runners were initialized. Do not
                 * maintain a second, partial override list here: it previously
                 * omitted K_NORM and let parity concatenate four GQA replicas.
                 */
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
            const size_t hidden_size = model_ctx_->embeddingLength();
            const auto collective_bom =
                CollectiveMemoryEstimator::localTP(
                    config_.max_seq_len,
                    static_cast<int>(hidden_size),
                    tp_ctx_->backend());

            std::shared_ptr<PhysicalMemoryAuthority> memory_authority;
            if (const auto concrete_model =
                    std::dynamic_pointer_cast<ModelContext>(model_ctx_))
            {
                const auto weight_manager =
                    concrete_model->concreteWeightManager();
                memory_authority = weight_manager
                                       ? weight_manager
                                             ->physicalMemoryAuthority()
                                       : nullptr;
            }

            const bool reserved = tp_ctx_->reserveCollectiveResources(
                collective_bom.backend_payload_capacity_bytes,
                collective_bom.fp16_scratch_elements,
                memory_authority);
            logVramBomLine(
                "collective_temp_reservation",
                "source=RankOrchestrator backend=local_tp max_seq_len=" + std::to_string(config_.max_seq_len) +
                    " hidden_size=" + std::to_string(hidden_size) +
                    " precision=" + activationPrecisionToString(config_.activation_precision) +
                    " status=" + (reserved ? "pass" : "fail") +
                    " " + vramBomBytes(
                        collective_bom.perDeviceBytes()));
            if (reserved)
            {
                LOG_DEBUG("RankOrchestrator: Reserved collective resources: "
                          << "logical_payload_capacity="
                          << collective_bom.backend_payload_capacity_bytes
                          << " bytes, physical_fp16_scratch="
                          << collective_bom.fp16_scratch_bytes << " bytes ("
                          << "max_seq_len=" << config_.max_seq_len
                          << ", hidden_size=" << hidden_size
                          << ", precision=" << activationPrecisionToString(config_.activation_precision) << ")");
            }
            else
            {
                throw std::runtime_error(
                    "RankOrchestrator: failed to reserve complete LocalTP "
                    "collective resources");
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
                    const auto installed_store =
                        concrete_weight_mgr->preparedWeightStoreIfInitialized();
                    if (config_.prepared_weight_store)
                    {
                        if (installed_store &&
                            installed_store != config_.prepared_weight_store)
                        {
                            throw std::runtime_error(
                                "RankOrchestrator cannot replace the model-owned "
                                "PreparedWeightStore");
                        }
                        concrete_weight_mgr->setPreparedWeightStore(config_.prepared_weight_store);
                    }
                    else if (installed_store)
                    {
                        /*
                         * Prepared storage is additive and model-owned. A new
                         * rank graph must never silently install a second store
                         * and orphan handles whose raw sources may already have
                         * been released.
                         */
                        config_.prepared_weight_store = installed_store;
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
                        !(config_.moe_routed_expert_plan &&
                          config_.moe_routed_expert_plan
                              ->usesExpertOverlayAuthority());
                    if (config_.prepared_weight_admission ==
                        PreparedWeightAdmission::ReuseCertifiedCompleteSet)
                    {
                        auto concrete_weight_mgr =
                            std::dynamic_pointer_cast<WeightManager>(weight_mgr);
                        if (!concrete_weight_mgr ||
                            !config_.prepared_weight_store)
                        {
                            throw std::runtime_error(
                                "RankOrchestrator certified prepared-weight "
                                "reuse lacks its concrete model authority");
                        }
                        for (const DeviceId device : device_ids)
                        {
                            if (concrete_weight_mgr
                                    ->preparedRecordCountForDevice(device) == 0u)
                            {
                                throw std::runtime_error(
                                    "RankOrchestrator certified prepared-weight "
                                    "reuse has no records for " +
                                    device.toString());
                            }
                        }

                        /*
                         * Non-GEMM device tensors live in WeightManager's
                         * model-owned device cache rather than PreparedWeightStore.
                         * Revisit that cache so a missing binding fails here,
                         * while deliberately avoiding the broad GEMM pipeline
                         * whose exact handles are already certified resident.
                         */
                        if (!weight_mgr->preloadForDevices(device_ids))
                        {
                            throw std::runtime_error(
                                "RankOrchestrator failed to restore non-GEMM "
                                "device bindings for certified weight reuse");
                        }
                        PerfStatsCollector::addCounter(
                            "weight_loading",
                            "rank_prepared_weight_materialization_reuses",
                            1.0,
                            "load",
                            {},
                            {{"devices", std::to_string(device_ids.size())}});
                    }
                    else
                    {
                        LOG_DEBUG("RankOrchestrator: Finalizing weights for "
                                  << device_ids.size() << " devices"
                                  << " (release_host_data=false, deferred until after graph build"
                                  << ", include_expert_jobs=" << include_expert_jobs << ")");
                        if (!weight_mgr->finalizeForDevices(
                                device_ids,
                                /*release_host_data=*/false,
                                include_expert_jobs))
                        {
                            throw std::runtime_error(
                                "RankOrchestrator required weight finalization failed");
                        }
                    }

                    /**
                     * Routed-overlay engines are deliberately not prepared here.
                     * This parent owns only mutable preload caches; it does not own
                     * the per-runner frozen TP/replication bindings that define the
                     * graph's actual weight identity. Each device runner prepares
                     * its overlay from that immutable set during materialization.
                     */
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
                                         [this, device_idx, device_id]() -> RunnerResult
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
                                                 runner_config.activation_seq_len = config_.resident_graph_rows;
                                                 runner_config.batch_size = config_.batch_size;
                                                 runner_config.activation_precision = config_.activation_precision;
                                                 runner_config.kv_cache_scale_k = config_.kv_cache_scale_k;
                                                 runner_config.kv_cache_scale_v = config_.kv_cache_scale_v;
                                                 runner_config.kv_cache_precision = config_.kv_cache_precision;
                                                 runner_config.tp_allreduce_precision_override =
                                                     config_.tp_allreduce_precision_override;
                                                 runner_config.prefix_cache = config_.prefix_cache;
                                                 runner_config.mtp = config_.mtp;
                                                 runner_config.routed_expert_compute_policy = config_.routed_expert_compute_policy;
                                                 runner_config.routed_expert_owner_order = config_.routed_expert_owner_order;
                                                 runner_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
                                                 runner_config.moe_routed_prefill = config_.moe_routed_prefill;
                                                 runner_config.moe_rebalance = config_.moe_rebalance;
                                                 runner_config.prepared_weight_store = config_.prepared_weight_store;
                                                 runner_config.prepared_weight_admission =
                                                     config_.prepared_weight_admission;
                                                 runner_config.reusable_execution_workspaces =
                                                     config_.reusable_execution_workspaces;
                                                 runner_config.moe_routed_expert_plan = config_.moe_routed_expert_plan;
                                                 runner_config.moe_expert_overlay_residency_authority =
                                                     config_.moe_expert_overlay_residency_authority;
                                                 runner_config.moe_expert_overlay_participant_residency =
                                                     config_.moe_expert_overlay_participant_residency;
                                                 runner_config.moe_expert_overlay_decode_histogram =
                                                     config_.moe_expert_overlay_decode_histogram;
                                                 runner_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
                                                 runner_config.moe_rank_batch_transport_registry =
                                                     config_.moe_rank_batch_transport_registry;
                                                 runner_config.moe_device_controller_fabric =
                                                     config_.moe_device_controller_fabric;
                                                 runner_config.moe_node_local_route_exchange =
                                                     config_.moe_node_local_route_exchange;
                                                 runner_config.moe_node_local_route_transport =
                                                     config_.moe_node_local_route_transport;
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
                    }
                }
            }
            else
            {
                LOG_DEBUG("RankOrchestrator: Retaining host weight data "
                          "(nested TP-in-PP; outer caller will release)");
            }
        }

        // Only a rank whose typed PP role owns the LM head may materialize a
        // host logits aggregate. Non-terminal nested TP ranks publish hidden
        // activations to the outer PP transfer boundary instead.
        if (ownsMainForwardLogits() && model_ctx_ && device_runners_.size() > 0)
        {
            int vocab = vocab_size();
            if (vocab > 0)
            {
                size_t max_tokens = static_cast<size_t>(config_.batch_size) *
                                    static_cast<size_t>(config_.max_seq_len);
                logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    vocab,
                    max_tokens,
                    logits_backend_resolver_);

                // Pin the logits buffer for faster D2H DMA
                if (device_runners_.size() > 1)
                {
                    DeviceId primary_dev = device_runners_[0]->primaryDeviceId();
                    if (external_device_backend_access_enabled_ &&
                        primary_dev.is_gpu())
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
            runner_config.activation_seq_len = config_.resident_graph_rows;
            runner_config.batch_size = config_.batch_size;
            runner_config.activation_precision = config_.activation_precision;
            runner_config.kv_cache_scale_k = config_.kv_cache_scale_k;
            runner_config.kv_cache_scale_v = config_.kv_cache_scale_v;
            runner_config.kv_cache_precision = config_.kv_cache_precision;
            runner_config.tp_allreduce_precision_override =
                config_.tp_allreduce_precision_override;
            runner_config.prefix_cache = config_.prefix_cache;
            runner_config.mtp = config_.mtp;
            runner_config.routed_expert_compute_policy = config_.routed_expert_compute_policy;
            runner_config.routed_expert_owner_order = config_.routed_expert_owner_order;
            runner_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
            runner_config.moe_routed_prefill = config_.moe_routed_prefill;
            runner_config.moe_rebalance = config_.moe_rebalance;
            runner_config.prepared_weight_store = config_.prepared_weight_store;
            runner_config.prepared_weight_admission =
                config_.prepared_weight_admission;
            runner_config.reusable_execution_workspaces =
                config_.reusable_execution_workspaces;
            runner_config.moe_routed_expert_plan = config_.moe_routed_expert_plan;
            runner_config.moe_expert_overlay_residency_authority =
                config_.moe_expert_overlay_residency_authority;
            runner_config.moe_expert_overlay_participant_residency =
                config_.moe_expert_overlay_participant_residency;
            runner_config.moe_expert_overlay_decode_histogram =
                config_.moe_expert_overlay_decode_histogram;
            runner_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
            runner_config.moe_rank_batch_transport_registry =
                config_.moe_rank_batch_transport_registry;
            runner_config.moe_device_controller_fabric =
                config_.moe_device_controller_fabric;
            runner_config.moe_node_local_route_exchange =
                config_.moe_node_local_route_exchange;
            runner_config.moe_node_local_route_transport =
                config_.moe_node_local_route_transport;
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
                nested_config.resident_graph_rows = config_.resident_graph_rows;
                nested_config.batch_size = config_.batch_size;
                nested_config.activation_precision = config_.activation_precision;
                nested_config.kv_cache_scale_k = config_.kv_cache_scale_k;
                nested_config.kv_cache_scale_v = config_.kv_cache_scale_v;
                nested_config.kv_cache_precision = config_.kv_cache_precision;
                nested_config.tp_allreduce_precision_override =
                    config_.tp_allreduce_precision_override;
                nested_config.prefix_cache = config_.prefix_cache;
                nested_config.mtp = config_.mtp;
                nested_config.routed_expert_compute_policy = config_.routed_expert_compute_policy;
                nested_config.routed_expert_owner_order = config_.routed_expert_owner_order;
                nested_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
                nested_config.moe_routed_prefill = config_.moe_routed_prefill;
                nested_config.moe_rebalance = config_.moe_rebalance;
                nested_config.prepared_weight_store = config_.prepared_weight_store;
                nested_config.prepared_weight_admission =
                    config_.prepared_weight_admission;
                nested_config.reusable_execution_workspaces =
                    config_.reusable_execution_workspaces;
                nested_config.moe_routed_expert_plan = config_.moe_routed_expert_plan;
                nested_config.moe_expert_overlay_residency_authority =
                    config_.moe_expert_overlay_residency_authority;
                nested_config.moe_expert_overlay_participant_residency =
                    config_.moe_expert_overlay_participant_residency;
                nested_config.moe_expert_overlay_decode_histogram =
                    config_.moe_expert_overlay_decode_histogram;
                nested_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
                nested_config.moe_rank_batch_transport_registry =
                    config_.moe_rank_batch_transport_registry;
                nested_config.moe_device_controller_fabric =
                    config_.moe_device_controller_fabric;
                nested_config.moe_node_local_route_exchange =
                    config_.moe_node_local_route_exchange;
                nested_config.moe_node_local_route_transport =
                    config_.moe_node_local_route_transport;
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
        initializePPGraphExecutionPlan();
    }

    void RankOrchestrator::initializePPGraphExecutionPlan()
    {
        if (pp_stage_runners_.empty())
        {
            throw std::runtime_error(
                "Cannot initialize a pipeline graph execution plan without stages");
        }

        std::vector<PipelineGraphExecutionSegment> segments;
        segments.reserve(pp_stage_runners_.size());
        for (std::size_t stage = 0; stage < pp_stage_runners_.size(); ++stage)
        {
            const auto &runner = pp_stage_runners_[stage];
            if (!runner)
            {
                throw std::runtime_error(
                    "Pipeline graph execution plan found a missing runner at stage " +
                    std::to_string(stage));
            }

            const ServingGraphPreparationKind kind =
                runner->servingGraphPreparationKind();
            PipelineGraphSegmentExecution execution;
            switch (kind)
            {
            case ServingGraphPreparationKind::EagerHostGraph:
                execution = PipelineGraphSegmentExecution::HostDeclarative;
                break;
            case ServingGraphPreparationKind::NativeDeviceExecutableFamily:
                execution =
                    PipelineGraphSegmentExecution::NativeDeviceExecutable;
                break;
            case ServingGraphPreparationKind::Unresolved:
                throw std::runtime_error(
                    "Pipeline graph execution plan found an unresolved graph owner at stage " +
                    std::to_string(stage));
            }

            segments.push_back(PipelineGraphExecutionSegment{
                .stage_index = stage,
                .primary_device = runner->primaryDeviceId(),
                .execution = execution,
            });
        }

        pp_graph_execution_plan_ =
            std::make_unique<PipelineGraphExecutionPlan>(
                std::move(segments));
    }

    bool RankOrchestrator::validatePPGraphTransactionReady(
        const char *operation) const noexcept
    {
        const char *const operation_name =
            operation && *operation ? operation : "LocalPP transaction";
        if (!pp_graph_execution_plan_)
        {
            LOG_ERROR(operation_name << " has no frozen pipeline graph plan");
            return false;
        }
        if (pp_graph_execution_plan_->segmentCount() !=
            pp_stage_runners_.size())
        {
            LOG_ERROR(
                operation_name
                << " runner cardinality changed after graph-plan construction"
                << " planned="
                << pp_graph_execution_plan_->segmentCount()
                << " live=" << pp_stage_runners_.size());
            return false;
        }

        for (const auto &segment : pp_graph_execution_plan_->segments())
        {
            if (segment.stage_index >= pp_stage_runners_.size() ||
                !pp_stage_runners_[segment.stage_index])
            {
                LOG_ERROR(
                    operation_name << " lost pipeline stage "
                                   << segment.stage_index);
                return false;
            }
            const auto &runner = pp_stage_runners_[segment.stage_index];
            const ServingGraphPreparationKind expected_kind =
                segment.execution ==
                        PipelineGraphSegmentExecution::HostDeclarative
                    ? ServingGraphPreparationKind::EagerHostGraph
                    : ServingGraphPreparationKind::
                          NativeDeviceExecutableFamily;
            if (runner->primaryDeviceId() != segment.primary_device ||
                runner->servingGraphPreparationKind() != expected_kind)
            {
                LOG_ERROR(
                    operation_name
                    << " detected pipeline graph identity drift at stage "
                    << segment.stage_index
                    << " planned_device="
                    << segment.primary_device.toString()
                    << " live_device="
                    << runner->primaryDeviceId().toString());
                return false;
            }
        }

        if (pp_graph_execution_plan_->nativeSegmentCount() != 0u &&
            !pp_graph_execution_plan_->materialized())
        {
            LOG_ERROR(
                operation_name
                << " preceded native pipeline graph-family materialization");
            return false;
        }
        return true;
    }

    void RankOrchestrator::recordCompletedPPGraphTransactions(
        const char *phase,
        std::size_t completed_segments,
        std::size_t transaction_count) const
    {
        if (!pp_graph_execution_plan_ ||
            !pp_graph_execution_plan_->hasHeterogeneousBoundary())
        {
            return;
        }

        std::string certification_error;
        if (transaction_count == 0u ||
            !pp_graph_execution_plan_->certifiesReplay(
                completed_segments, &certification_error))
        {
            LOG_ERROR(
                "Completed LocalPP transaction violated its frozen graph plan: "
                << (transaction_count == 0u
                        ? "transaction count is zero"
                        : certification_error));
            std::terminate();
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "segmented_replay_segments",
            static_cast<double>(completed_segments * transaction_count),
            phase && *phase ? phase : "inference",
            "pipeline_coordinator",
            {{"segments_per_transaction",
              std::to_string(completed_segments)},
             {"transactions", std::to_string(transaction_count)},
             {"heterogeneous_segmented", "true"},
             {"boundary_authority", "rank_pipeline_graph_plan"}});
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
            success = forwardTP(
                tokens, seq_len, MainForwardDispatch::Automatic);
            break;
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            // PP and TP_PP both use sequential stage execution
            // The difference is that TP_PP stages may be nested MDOs (TP domains)
            // but forwardPP() works through IInferenceRunner interface regardless
            success = forwardPP(
                tokens, seq_len, MainForwardDispatch::Automatic);
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
            return forwardTP(
                tokens, seq_len, MainForwardDispatch::Prefill);
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            return forwardPP(
                tokens, seq_len, MainForwardDispatch::Prefill);
        default:
            LOG_ERROR("RankOrchestrator::forwardPrefill: Unknown parallelism mode");
            return false;
        }
    }

    bool RankOrchestrator::forwardRestoredPrefixMTPDecodeBridge(
        const RestoredPrefixMTPDecodeBridgeRequest &request)
    {
        if (!request.valid() || current_position_ != request.restored_prefix_tokens)
        {
            LOG_ERROR(
                "RankOrchestrator restored-prefix MTP bridge does not match "
                "the rank-local published history"
                << " restored=" << request.restored_prefix_tokens
                << " position=" << current_position_);
            return false;
        }
        const int token = request.token_id;
        switch (mode_)
        {
        case ParallelismMode::TP:
            return forwardTP(
                &token,
                /*seq_len=*/1,
                MainForwardDispatch::RestoredPrefixMTPDecodeBridge,
                request.restored_prefix_tokens);
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            return forwardPP(
                &token,
                /*seq_len=*/1,
                MainForwardDispatch::RestoredPrefixMTPDecodeBridge,
                request.restored_prefix_tokens);
        default:
            LOG_ERROR(
                "RankOrchestrator::forwardRestoredPrefixMTPDecodeBridge: "
                "Unknown parallelism mode");
            return false;
        }
    }

    ServingGraphPreparationKind
    RankOrchestrator::servingGraphPreparationKind() const noexcept
    {
        const auto composite_kind = [](
                                        const std::vector<
                                            std::unique_ptr<IInferenceRunner>>
                                            &runners)
        {
            if (runners.empty())
                return ServingGraphPreparationKind::Unresolved;
            bool native_family_required = false;
            for (const auto &runner : runners)
            {
                if (!runner)
                    return ServingGraphPreparationKind::Unresolved;
                const auto child_kind =
                    runner->servingGraphPreparationKind();
                if (child_kind ==
                    ServingGraphPreparationKind::Unresolved)
                {
                    return child_kind;
                }
                native_family_required =
                    native_family_required ||
                    child_kind == ServingGraphPreparationKind::
                                      NativeDeviceExecutableFamily;
            }
            return native_family_required
                       ? ServingGraphPreparationKind::
                             NativeDeviceExecutableFamily
                       : ServingGraphPreparationKind::EagerHostGraph;
        };

        if (mode_ == ParallelismMode::TP &&
            pp_stage_runners_.empty())
        {
            if (device_runners_.empty() || !device_runners_.front())
                return ServingGraphPreparationKind::Unresolved;
            const auto first =
                device_runners_.front()->servingGraphPreparationKind();
            if (first == ServingGraphPreparationKind::Unresolved)
                return first;
            for (const auto &runner : device_runners_)
            {
                if (!runner ||
                    runner->servingGraphPreparationKind() != first)
                {
                    // A mixed LocalTP tier needs a distinct symmetric graph
                    // protocol; it cannot inherit PP's sequential composite.
                    return ServingGraphPreparationKind::Unresolved;
                }
            }
            return first;
        }
        if ((mode_ == ParallelismMode::PP ||
             mode_ == ParallelismMode::TP_PP) &&
            device_runners_.empty())
        {
            /*
             * PP stages are sequential, so a mixed CPU/GPU pipeline reports
             * the strongest child transition. Materialization below skips
             * already-built eager children and seals each native stage in
             * pipeline order. Nested LocalTP stages remain responsible for
             * their own symmetric participant capture.
             */
            return composite_kind(pp_stage_runners_);
        }
        return ServingGraphPreparationKind::Unresolved;
    }

    bool RankOrchestrator::materializeServingGraphFamilyWithoutLaunch(
        const ServingGraphFamilyMaterializationPlan &plan)
    {
        if (!plan.valid())
        {
            LOG_ERROR(
                "RankOrchestrator serving graph setup requires a valid frozen family plan");
            return false;
        }

        if (mode_ == ParallelismMode::PP ||
            mode_ == ParallelismMode::TP_PP)
        {
            if (!device_runners_.empty() || pp_stage_runners_.empty())
            {
                LOG_ERROR(
                    "RankOrchestrator PP serving graph setup found an invalid composite runner shape");
                return false;
            }
            if (!pp_graph_execution_plan_ ||
                pp_graph_execution_plan_->segmentCount() !=
                    pp_stage_runners_.size())
            {
                LOG_ERROR(
                    "RankOrchestrator PP serving graph setup has no exact frozen pipeline plan");
                return false;
            }

            std::size_t materialized_native_segments = 0u;
            for (const auto &segment :
                 pp_graph_execution_plan_->segments())
            {
                const std::size_t stage = segment.stage_index;
                auto &runner = pp_stage_runners_[stage];
                if (!runner)
                {
                    LOG_ERROR(
                        "RankOrchestrator PP serving graph setup found a missing stage at index "
                        << stage);
                    return false;
                }
                const auto child_kind =
                    runner->servingGraphPreparationKind();
                const ServingGraphPreparationKind planned_kind =
                    segment.execution ==
                            PipelineGraphSegmentExecution::HostDeclarative
                        ? ServingGraphPreparationKind::EagerHostGraph
                        : ServingGraphPreparationKind::
                              NativeDeviceExecutableFamily;
                if (child_kind != planned_kind ||
                    runner->primaryDeviceId() != segment.primary_device)
                {
                    LOG_ERROR(
                        "RankOrchestrator PP serving graph setup detected identity drift at stage "
                        << stage << " planned_device="
                        << segment.primary_device.toString()
                        << " live_device="
                        << runner->primaryDeviceId().toString());
                    return false;
                }
                if (segment.execution ==
                    PipelineGraphSegmentExecution::HostDeclarative)
                {
                    continue;
                }

                ServingGraphFamilyMaterializationPlan stage_plan = plan;
                stage_plan.pipeline_hidden_input = nullptr;
                if (stage > 0u)
                {
                    auto *previous = pp_stage_runners_[stage - 1u].get();
                    stage_plan.pipeline_hidden_input =
                        previous ? previous->getHiddenState() : nullptr;
                    if (!stage_plan.pipeline_hidden_input)
                    {
                        LOG_ERROR(
                            "RankOrchestrator PP serving graph setup could not bind the stable activation owner from stage "
                            << (stage - 1u) << " to stage " << stage);
                        return false;
                    }
                }
                if (!runner->materializeServingGraphFamilyWithoutLaunch(
                        stage_plan))
                {
                    LOG_ERROR(
                        "RankOrchestrator PP serving graph setup failed at stage "
                        << stage);
                    return false;
                }
                ++materialized_native_segments;
            }

            std::string materialization_error;
            if (!pp_graph_execution_plan_->markMaterialized(
                    materialized_native_segments,
                    &materialization_error))
            {
                LOG_ERROR(
                    "RankOrchestrator PP serving graph setup did not cover its frozen native inventory: "
                    << materialization_error);
                return false;
            }

            if (pp_graph_execution_plan_->hasHeterogeneousBoundary())
            {
                const std::string total_segments = std::to_string(
                    pp_graph_execution_plan_->segmentCount());
                const std::string native_segments = std::to_string(
                    pp_graph_execution_plan_->nativeSegmentCount());
                const std::string host_segments = std::to_string(
                    pp_graph_execution_plan_->hostSegmentCount());
                const PerfStatsCollector::Tags tags =
                    {{"scope", "pipeline_coordinator"},
                     {"total_segments", total_segments},
                     {"native_segments", native_segments},
                     {"host_segments", host_segments},
                     {"heterogeneous_segmented", "true"},
                     {"boundary_authority", "rank_pipeline_graph_plan"}};
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "segmented_plan_segments",
                    static_cast<double>(
                        pp_graph_execution_plan_->segmentCount()),
                    "setup",
                    "pipeline_coordinator",
                    tags);
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "segmented_graph_capture_segments",
                    static_cast<double>(materialized_native_segments),
                    "setup",
                    "pipeline_coordinator",
                    tags);
            }

            PerfStatsCollector::addCounter(
                "forward_graph",
                "serving_graph_family_pipeline_completions",
                1.0,
                "setup",
                "rank",
                {{"stages", std::to_string(pp_stage_runners_.size())},
                 {"request_state_mutations", "0"},
                 {"executable_launches", "0"}});
            return true;
        }

        if (mode_ != ParallelismMode::TP ||
            !pp_stage_runners_.empty() || device_runners_.empty())
        {
            LOG_ERROR(
                "RankOrchestrator serving graph setup requires one flat, "
                "non-empty LocalTP graph and a valid frozen family plan");
            return false;
        }
        for (std::size_t index = 0; index < device_runners_.size(); ++index)
        {
            if (!device_runners_[index] ||
                !device_runners_[index]->primaryDeviceId().is_gpu())
            {
                LOG_ERROR(
                    "RankOrchestrator serving graph setup found a missing or "
                    "non-GPU LocalTP participant at index "
                    << index);
                return false;
            }
        }

        if (device_runners_.size() == 1)
        {
            return device_runners_.front()
                ->materializeServingGraphFamilyWithoutLaunch(plan);
        }

        /*
         * Native graph capture can enter LocalTP collective nodes.  All
         * participant graphs must therefore build and capture on persistent
         * workers at the same time; a serial setup loop deadlocks at the first
         * collective and a process-wide cache-miss mutex creates the same
         * asymmetry less visibly.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback(
                    [this]()
                    {
                        LOG_WARN(
                            "[TPWorkerPool] Serving graph setup failure "
                            "detected - aborting collective backend");
                        tp_ctx_->requestAbort();
                    });
            }
        }
        if (tp_worker_pool_->numWorkers() != device_runners_.size())
        {
            LOG_ERROR(
                "RankOrchestrator serving graph setup worker topology no "
                "longer matches the immutable LocalTP participant count");
            return false;
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             &plan,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](std::size_t index) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                const DeviceId device =
                    device_runners_[index]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                const auto started = std::chrono::steady_clock::now();
                LOG_INFO(
                    "[ServingGraphMaterialization] LocalTP worker begin"
                    << " participant=" << index
                    << " device=" << device.toString());
                const bool success = device_runners_[index]
                                         ->materializeServingGraphFamilyWithoutLaunch(
                                             plan);
                const auto elapsed = std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - started);
                LOG_INFO(
                    "[ServingGraphMaterialization] LocalTP worker complete"
                    << " participant=" << index
                    << " device=" << device.toString()
                    << " success=" << (success ? "true" : "false")
                    << " elapsed_ms=" << elapsed.count());
                return success;
            });

        bool all_success = true;
        std::exception_ptr first_exception;
        std::size_t first_exception_device = 0;
        const auto results = tp_worker_pool_->collectAll(
            /*timeout_ms=*/0);
        for (const auto &result : results)
        {
            if (!result.completed || !result.success)
                all_success = false;
            if (result.exception && !first_exception)
            {
                first_exception = result.exception;
                first_exception_device = result.worker_index;
            }
        }
        if (first_exception)
        {
            LOG_ERROR(
                "RankOrchestrator serving graph setup rethrowing failure from "
                "LocalTP participant "
                << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        PerfStatsCollector::addCounter(
            "forward_graph",
            "serving_graph_family_rank_completions",
            1.0,
            "setup",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"prefill_buckets",
              std::to_string(plan.prefill_bucket_rows.size())},
             {"fanout", "concurrent_persistent_workers"}});
        return true;
    }

    /**
     * @brief Forward one overlay request generation through the rank-local graph tree.
     *
     * A RankOrchestrator may own LocalTP children or nested PP-stage runners.
     * Each child that can enter a graph-native sparse boundary must see the
     * same root-published generation before any worker thread begins its graph
     * launch. Propagating during request setup keeps this out of the token hot
     * path and prevents a child graph cache from inventing its own epoch.
     */
    bool RankOrchestrator::setMoEOverlayCollectiveRequestGeneration(
        uint64_t generation_id)
    {
        if (generation_id == 0)
        {
            LOG_ERROR(
                "RankOrchestrator::setMoEOverlayCollectiveRequestGeneration "
                "received an invalid zero generation");
            return false;
        }

        const auto &children =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (children.empty())
        {
            LOG_ERROR(
                "RankOrchestrator::setMoEOverlayCollectiveRequestGeneration "
                "has no child graph runners");
            return false;
        }

        for (const auto &child : children)
        {
            if (!child ||
                !child->setMoEOverlayCollectiveRequestGeneration(
                    generation_id))
            {
                LOG_ERROR(
                    "RankOrchestrator could not publish graph-native MoE "
                    "collective generation to every child graph");
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::setMoEOverlayInferenceTransactionCoordinator(
        std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
            coordinator,
        int continuation_participant_index)
    {
        /*
         * A RankOrchestrator is itself one participant only when nested under
         * another composite runner.  The production heterogeneous overlay cell
         * is the outer LocalTP rank, so it owns all child participant indices
         * and requires callers to address that rank as participant zero.
         */
        if (!coordinator || continuation_participant_index != 0 ||
            mode_ != ParallelismMode::TP ||
            device_runners_.empty() || !pp_stage_runners_.empty() ||
            coordinator->continuationParticipantCount() !=
                static_cast<int>(device_runners_.size()))
        {
            LOG_ERROR(
                "RankOrchestrator requires a flat LocalTP graph whose child "
                "count exactly matches the ExpertOverlay coordinator");
            return false;
        }

        if (moe_overlay_inference_transaction_coordinator_ &&
            moe_overlay_inference_transaction_coordinator_ != coordinator)
        {
            LOG_ERROR(
                "RankOrchestrator cannot replace its ExpertOverlay transaction authority");
            return false;
        }
        if (moe_overlay_interference_probe_)
        {
            std::string probe_error;
            if (!coordinator->bindPrefillInterferenceProbe(
                    moe_overlay_interference_probe_, &probe_error))
            {
                LOG_ERROR(
                    "RankOrchestrator could not bind chunk-level ExpertOverlay "
                    "prefill calibration: " << probe_error);
                return false;
            }
        }

        for (std::size_t index = 0; index < device_runners_.size(); ++index)
        {
            auto &child = device_runners_[index];
            if (!child ||
                !child->setMoEOverlayInferenceTransactionCoordinator(
                    coordinator, static_cast<int>(index)))
            {
                LOG_ERROR(
                    "RankOrchestrator could not bind every LocalTP child to "
                    "the rank-wide ExpertOverlay transaction authority");
                return false;
            }
        }
        moe_overlay_inference_transaction_coordinator_ =
            std::move(coordinator);
        moe_overlay_coordinator_owns_prefill_probe_ =
            static_cast<bool>(moe_overlay_interference_probe_);
        return true;
    }

    bool RankOrchestrator::setMoEOverlayInferenceInterferenceProbe(
        std::shared_ptr<MoEOverlayInferenceInterferenceProbe> probe)
    {
        if (!probe || mode_ != ParallelismMode::TP ||
            device_runners_.empty() || !pp_stage_runners_.empty() ||
            (moe_overlay_interference_probe_ &&
             moe_overlay_interference_probe_ != probe))
        {
            LOG_ERROR(
                "RankOrchestrator requires one flat LocalTP owner and one stable "
                "ExpertOverlay interference probe");
            return false;
        }
        moe_overlay_interference_probe_ = std::move(probe);
        if (moe_overlay_inference_transaction_coordinator_)
        {
            std::string probe_error;
            if (!moe_overlay_inference_transaction_coordinator_
                     ->bindPrefillInterferenceProbe(
                         moe_overlay_interference_probe_, &probe_error))
            {
                LOG_ERROR(
                    "RankOrchestrator could not bind chunk-level ExpertOverlay "
                    "prefill calibration: " << probe_error);
                moe_overlay_interference_probe_.reset();
                return false;
            }
            moe_overlay_coordinator_owns_prefill_probe_ = true;
        }
        return true;
    }

    bool RankOrchestrator::forwardGroupedMTPVerifierWithHostTokenIds(
        const std::vector<std::vector<int>> &token_batches)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardGroupedMTPVerifierWithHostTokenIds(
                token_batches);
        }

        for (const auto &runner : device_runners_)
        {
            if (runner && !runner->primaryDeviceId().is_cpu())
            {
                LOG_ERROR(
                    "RankOrchestrator::forwardGroupedMTPVerifierWithHostTokenIds: "
                    "host-token verification is invalid for GPU participants");
                return false;
            }
        }

        return forwardHostTokenBatchAcrossDevices(
            token_batches,
            &IInferenceRunner::forwardGroupedMTPVerifierWithHostTokenIds,
            "forwardGroupedMTPVerifierWithHostTokenIds");
    }

    bool RankOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds(
        const int *token_shadow,
        const void *token_ids_device,
        int seq_len)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardGroupedMTPVerifierWithDeviceTokenIds(
                token_shadow,
                token_ids_device,
                seq_len);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->forwardGroupedMTPVerifierWithDeviceTokenIds(
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
            LOG_ERROR("RankOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds: "
                      "invalid LocalTP device-token input bundle");
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i] ||
                !rank_mtp_verifier_child_token_inputs_[i])
            {
                LOG_ERROR("RankOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds: participant "
                          << i << " has no staged device-token input row");
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
                    device_runners_[i]
                        ->forwardGroupedMTPVerifierWithDeviceTokenIds(
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
                "forwardGroupedMTPVerifierWithDeviceTokenIds",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds: "
                      "Re-throwing primary exception from device "
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

    bool RankOrchestrator::
        advanceMTPMainConditionFromDeviceResidentLogicalState(
            int32_t token_shadow,
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar
                ->advanceMTPMainConditionFromDeviceResidentLogicalState(
                    token_shadow,
                    logical_state,
                    request_index);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->advanceMTPMainConditionFromDeviceResidentLogicalState(
                    token_shadow,
                    logical_state,
                    request_index);
        }

        const DeviceResidentLogicalSequenceStateHandle current_rank_state =
            deviceResidentLogicalSequenceState();
        if (!logical_state.valid() ||
            !current_rank_state.valid() ||
            !logical_state.sameMailboxAs(current_rank_state) ||
            !logical_state.coversRequest(request_index) ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            LOG_ERROR(
                "[RankOrchestrator] Resident main-condition advance received a "
                "stale, foreign, or incomplete LocalTP mailbox");
            return false;
        }

        const bool advanced = dispatchLocalTPMTPMainCondition(
            "advanceMTPMainConditionFromDeviceResidentLogicalState",
            [this, token_shadow, request_index](
                IInferenceRunner &child,
                size_t participant)
            {
                return child
                    .advanceMTPMainConditionFromDeviceResidentLogicalState(
                        token_shadow,
                        rank_resident_child_logical_state_handles_[participant],
                        request_index);
            });
        if (!advanced)
            return false;

        /*
         * Every child has consumed and retired its one-shot condition input.
         * The rank-level bundle is only a typed view over those child handles,
         * so retaining it would resurrect state that no participant owns.
         * Canonical child KV becomes the sole post-forward position authority.
         */
        invalidateRankResidentLogicalStateAggregate(
            "resident_main_condition_advance",
            "condition_input_consumed");
        ++current_position_;
        for (int &length : current_sequence_lengths_)
            ++length;
        current_padded_seq_len_ = 1;
        stats_dirty_ = true;
        return true;
    }

    bool RankOrchestrator::advanceMTPMainConditionFromDeviceTargetSample(
        int32_t token_shadow,
        int target_sample_slot)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->advanceMTPMainConditionFromDeviceTargetSample(
                token_shadow,
                target_sample_slot);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->advanceMTPMainConditionFromDeviceTargetSample(
                    token_shadow,
                    target_sample_slot);
        }
        if (target_sample_slot < 0 ||
            target_sample_slot >= rank_stochastic_slot_capacity_)
        {
            LOG_ERROR(
                "[RankOrchestrator] Device-target main-condition advance "
                "received an invalid target slot");
            return false;
        }

        const bool advanced = dispatchLocalTPMTPMainCondition(
            "advanceMTPMainConditionFromDeviceTargetSample",
            [token_shadow, target_sample_slot](
                IInferenceRunner &child,
                size_t)
            {
                return child.advanceMTPMainConditionFromDeviceTargetSample(
                    token_shadow,
                    target_sample_slot);
            });
        if (!advanced)
            return false;

        /*
         * Target-sample publication is the condition graph's input, not its
         * output. Child completion retires those mailboxes; rank orchestration
         * must likewise expose no aggregate until sampling or verifier
         * publication creates the next device-owned condition transaction.
         */
        invalidateRankResidentLogicalStateAggregate(
            "target_sample_main_condition_advance",
            "condition_input_consumed");
        ++current_position_;
        for (int &length : current_sequence_lengths_)
            ++length;
        current_padded_seq_len_ = 1;
        stats_dirty_ = true;
        return true;
    }

    bool RankOrchestrator::forwardBatchWithDeviceTokenIds(
        const std::vector<std::vector<int>> &token_batches,
        const void *token_ids_device,
        int padded_seq_len)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardBatchWithDeviceTokenIds(
                token_batches,
                token_ids_device,
                padded_seq_len);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardBatchWithDeviceTokenIds(
                token_batches,
                token_ids_device,
                padded_seq_len);
        }
        if (device_runners_.size() < 2 ||
            token_batches.empty() ||
            padded_seq_len <= 0 ||
            token_ids_device != rank_mtp_verifier_child_token_inputs_.data() ||
            rank_mtp_verifier_child_token_inputs_.size() !=
                device_runners_.size() ||
            rank_mtp_verifier_child_token_count_ != padded_seq_len)
        {
            LOG_ERROR("RankOrchestrator::forwardBatchWithDeviceTokenIds: invalid LocalTP verifier token bundle");
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i] ||
                !rank_mtp_verifier_child_token_inputs_[i])
            {
                LOG_ERROR("RankOrchestrator::forwardBatchWithDeviceTokenIds: participant "
                          << i << " has no staged verifier token matrix");
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
                    LOG_WARN("[TPWorkerPool] device-token batch verifier forward failure detected - aborting collective backend");
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
             &token_batches,
             padded_seq_len,
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
                    LOG_DEBUG("[TP_WORKER_CONTRACT] event=device_token_batch_forward_enter"
                              << " worker=" << i
                              << " device=" << device_id.toString()
                              << " padded_seq_len=" << padded_seq_len
                              << " request_batch=" << token_batches.size());
                }

                const bool ok =
                    device_runners_[i]->forwardBatchWithDeviceTokenIds(
                        token_batches,
                        rank_mtp_verifier_child_token_inputs_[i],
                        padded_seq_len);

                if (debugEnv().tp_collective_contract_trace)
                {
                    LOG_DEBUG("[TP_WORKER_CONTRACT] event=device_token_batch_forward_leave"
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
                "forwardBatchWithDeviceTokenIds",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardBatchWithDeviceTokenIds: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        current_position_ += padded_seq_len;
        current_padded_seq_len_ = padded_seq_len;
        current_sequence_lengths_.resize(std::max(
            current_sequence_lengths_.size(),
            token_batches.size()),
            0);
        for (size_t request = 0; request < token_batches.size(); ++request)
        {
            current_sequence_lengths_[request] +=
                static_cast<int>(token_batches[request].size());
        }
        stats_dirty_ = true;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_device_token_batch_forwards",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(token_batches.size())},
             {"padded_seq_len", std::to_string(padded_seq_len)}});
        return true;
    }

    bool RankOrchestrator::supportsPrefillChunkSchedule(int seq_len) const
    {
        const auto *runners =
            mode_ == ParallelismMode::TP
                ? &device_runners_
                : (mode_ == ParallelismMode::PP ||
                           mode_ == ParallelismMode::TP_PP
                       ? &pp_stage_runners_
                       : nullptr);
        if (!runners || runners->empty())
            return false;
        if ((mode_ == ParallelismMode::PP ||
             mode_ == ParallelismMode::TP_PP) &&
            !pp_ctx_)
        {
            return false;
        }

        for (const auto &runner : *runners)
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
                MainForwardDispatch::Automatic,
                /*restored_prefix_tokens=*/0,
                &policy,
                pad_token_id,
                allow_padded_execution);
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            return forwardPPPrefillChunkSchedule(
                tokens,
                seq_len,
                policy,
                pad_token_id,
                allow_padded_execution);
        default:
            LOG_ERROR("RankOrchestrator::forwardPrefillChunkSchedule: Unknown parallelism mode");
            return false;
        }
    }

    bool RankOrchestrator::forwardPPPrefillChunkSchedule(
        const int *tokens,
        int seq_len,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        if (!tokens || seq_len <= 0 ||
            pp_stage_runners_.empty() || !pp_ctx_)
        {
            LOG_ERROR(
                "RankOrchestrator PP prefill schedule requires tokens, "
                "pipeline stages, and a LocalPP transfer context");
            return false;
        }
        if (!validatePPGraphTransactionReady(
                "RankOrchestrator PP prefill schedule"))
        {
            return false;
        }
        if (policy.real_token_start != current_position_ ||
            policy.real_token_count != seq_len)
        {
            LOG_ERROR(
                "RankOrchestrator PP prefill schedule disagrees with the "
                "pipeline request cursor"
                << " schedule_start=" << policy.real_token_start
                << " position=" << current_position_
                << " schedule_rows=" << policy.real_token_count
                << " request_rows=" << seq_len);
            return false;
        }

        const PrefillChunkSchedule root_schedule =
            planPrefillChunkSchedule(policy);
        if (!root_schedule || root_schedule.chunks.empty())
        {
            LOG_ERROR(
                "RankOrchestrator could not plan its pipeline-wide prefill "
                "schedule: "
                << root_schedule.error);
            return false;
        }

        last_logits_forward_phase_ = LogitsForwardPhase::Prefill;
        if (logits_gatherer_)
            logits_gatherer_->invalidate();

        std::uint64_t transferred_rows = 0u;
        const std::size_t num_stages =
            pp_graph_execution_plan_->segmentCount();
        std::size_t completed_transactions = 0u;
        for (const PrefillChunkPlan &chunk : root_schedule.chunks)
        {
            const int relative_offset =
                chunk.token_offset - policy.real_token_start;
            if (relative_offset < 0 || chunk.real_count <= 0 ||
                relative_offset > seq_len ||
                chunk.real_count > seq_len - relative_offset ||
                chunk.bucket_seq_len < chunk.real_count)
            {
                LOG_ERROR(
                    "RankOrchestrator PP prefill plan contains an invalid "
                    "chunk"
                    << " chunk=" << chunk.chunk_index
                    << " offset=" << relative_offset
                    << " real_rows=" << chunk.real_count
                    << " bucket_rows=" << chunk.bucket_seq_len);
                return false;
            }

            /*
             * A preceding transfer may leave TensorBase's current-device
             * metadata naming the destination. The retained stage-0 graph
             * still owns its original allocation, so make that allocation the
             * selected producer again before the next chunk writes it.
             */
            if (num_stages > 1u)
            {
                TensorBase *const stage0_hidden =
                    pp_stage_runners_.front()->getHiddenState();
                if (stage0_hidden)
                {
                    const DeviceId stage0_device =
                        pp_ctx_->deviceForStage(0).toLocalDeviceId();
                    const auto current_device =
                        stage0_hidden->current_device();
                    if (current_device && *current_device != stage0_device)
                        stage0_hidden->allocateOnDevice(stage0_device);
                }
            }

            std::size_t completed_segments = 0u;
            for (const auto &segment :
                 pp_graph_execution_plan_->segments())
            {
                const std::size_t stage = segment.stage_index;
                auto &runner = pp_stage_runners_[stage];
                if (!runner)
                {
                    LOG_ERROR(
                        "RankOrchestrator PP prefill schedule found a missing "
                        "stage at index "
                        << stage);
                    return false;
                }

                if (stage > 0u)
                {
                    TensorBase *const hidden =
                        pp_stage_runners_[stage - 1u]->getHiddenState();
                    if (!hidden)
                    {
                        LOG_ERROR(
                            "RankOrchestrator PP prefill stage "
                            << (stage - 1u)
                            << " produced no activation owner");
                        return false;
                    }

                    /*
                     * GPU children replay a fixed-width captured graph. Copy
                     * the complete physical bucket so every downstream row is
                     * initialized; only real_count is allowed to advance KV
                     * and request state.
                     */
                    const std::size_t active_bytes =
                        pp_activation_contract_
                            ? pp_activation_contract_
                                  ->transfer(static_cast<int>(stage - 1u))
                                  .activeBytes(chunk.bucket_seq_len)
                            : static_cast<std::size_t>(
                                  chunk.bucket_seq_len) *
                                  model_ctx_->embeddingLength() *
                                  sizeof(float);
                    if (!pp_ctx_->transfer(
                            hidden,
                            static_cast<int>(stage - 1u),
                            static_cast<int>(stage),
                            active_bytes))
                    {
                        LOG_ERROR(
                            "RankOrchestrator PP prefill activation transfer "
                            "failed from stage "
                            << (stage - 1u) << " to " << stage
                            << " for " << chunk.bucket_seq_len
                            << " physical rows");
                        return false;
                    }
                    transferred_rows += static_cast<std::uint64_t>(
                        chunk.bucket_seq_len);
                    runner->setHiddenState(hidden);
                }

                /*
                 * The root owns chunk boundaries and maintenance decisions.
                 * Each child receives exactly one fixed-width transaction at
                 * its own canonical cursor, preventing independent child
                 * schedulers from splitting a pipeline chunk differently.
                 */
                PrefillChunkSchedulerPolicy child_policy;
                child_policy.bucket_sizes = {chunk.bucket_seq_len};
                child_policy.fixed_chunk_real_tokens =
                    chunk.bucket_seq_len;
                child_policy.real_token_start = runner->get_position();
                child_policy.real_token_count = chunk.real_count;

                const bool child_ok = runner->forwardPrefillChunkSchedule(
                    tokens + relative_offset,
                    chunk.real_count,
                    child_policy,
                    pad_token_id,
                    allow_padded_execution);
                if (stage > 0u)
                    runner->clearHiddenStateInput();
                if (!child_ok)
                {
                    LOG_ERROR(
                        "RankOrchestrator PP prefill transaction failed at "
                        "stage "
                        << stage << " chunk=" << chunk.chunk_index);
                    return false;
                }
                ++completed_segments;
            }
            if (completed_segments != num_stages)
            {
                LOG_ERROR(
                    "RankOrchestrator PP prefill transaction did not traverse its frozen pipeline plan"
                    << " completed=" << completed_segments
                    << " expected=" << num_stages);
                std::terminate();
            }
            ++completed_transactions;
        }

        if (!skip_logits_gather_prefill_)
        {
            if (!logits_gatherer_)
            {
                logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    0, 0, logits_backend_resolver_);
                applyLogitsGatherSkipFlags();
            }
            logits_gatherer_->copyFromStage(
                *pp_stage_runners_.back(),
                0,
                config_.batch_size,
                config_.max_seq_len);
        }

        recordCompletedPPGraphTransactions(
            "prefill", num_stages, completed_transactions);
        current_position_ += seq_len;
        PerfStatsCollector::addCounter(
            "forward_graph",
            "pipeline_prefill_chunk_transactions",
            static_cast<double>(root_schedule.chunks.size()),
            "prefill",
            "rank",
            {{"stages", std::to_string(num_stages)},
             {"logical_rows", std::to_string(seq_len)},
             {"transferred_physical_rows",
              std::to_string(transferred_rows)},
             {"boundary_authority", "pipeline_root"}});
        return true;
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
        MainForwardDispatch dispatch,
        int restored_prefix_tokens,
        const PrefillChunkSchedulerPolicy *chunk_schedule_policy,
        int chunk_schedule_pad_token_id,
        bool chunk_schedule_allow_padded_execution)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::forwardTP: No device runners available");
            return false;
        }
        const bool restored_prefix_bridge =
            dispatch == MainForwardDispatch::RestoredPrefixMTPDecodeBridge;
        if ((restored_prefix_bridge &&
             (!tokens || seq_len != 1 || restored_prefix_tokens <= 0)) ||
            (!restored_prefix_bridge && restored_prefix_tokens != 0) ||
            (chunk_schedule_policy &&
             dispatch != MainForwardDispatch::Automatic))
        {
            LOG_ERROR(
                "RankOrchestrator::forwardTP received an invalid typed main-forward dispatch"
                << " dispatch=" << static_cast<int>(dispatch)
                << " seq_len=" << seq_len
                << " restored=" << restored_prefix_tokens
                << " chunked=" << (chunk_schedule_policy != nullptr));
            return false;
        }

        const LogitsForwardPhase logits_phase =
            (dispatch == MainForwardDispatch::Prefill || seq_len != 1)
                ? LogitsForwardPhase::Prefill
                : LogitsForwardPhase::Decode;
        last_logits_forward_phase_ = logits_phase;

        /*
         * The MPI root and every follower execute this exact rank-local
         * boundary. Keeping the one-shot probe here gives distributed economy
         * calibration symmetric evidence without putting timing or policy on
         * the sparse device graph. Schedule geometry is planned only for an
         * explicitly chunked prefill, where the production path already owns
         * the same bounded host plan.
         */
        int probe_execution_rows = seq_len;
        int probe_transaction_count = 1;
        std::uint64_t probe_schedule_fingerprint = 0u;
        if (moe_overlay_interference_probe_ && chunk_schedule_policy)
        {
            const auto schedule = planPrefillChunkSchedule(
                *chunk_schedule_policy);
            if (schedule)
            {
                probe_execution_rows = 0;
                probe_transaction_count = static_cast<int>(
                    schedule.chunks.size());
                probe_schedule_fingerprint = 14695981039346656037ull;
                for (const auto &chunk : schedule.chunks)
                {
                    probe_execution_rows += chunk.bucket_seq_len;
                    for (const auto value : {
                             static_cast<std::uint64_t>(chunk.real_count),
                             static_cast<std::uint64_t>(chunk.bucket_seq_len)})
                    {
                        constexpr std::uint64_t kPrime = 1099511628211ull;
                        std::uint64_t mixed = value;
                        for (std::size_t byte = 0u;
                             byte < sizeof(mixed);
                             ++byte)
                        {
                            probe_schedule_fingerprint ^= mixed & 0xffu;
                            probe_schedule_fingerprint *= kPrime;
                            mixed >>= 8u;
                        }
                    }
                }
            }
        }
        if (probe_schedule_fingerprint == 0u)
        {
            probe_schedule_fingerprint = 14695981039346656037ull;
            for (const auto value : {
                     static_cast<std::uint64_t>(
                         logits_phase == LogitsForwardPhase::Decode
                             ? ExpertHistogramSource::DecodeToken
                             : ExpertHistogramSource::PrefillChunk),
                     static_cast<std::uint64_t>(seq_len),
                     static_cast<std::uint64_t>(probe_execution_rows),
                     static_cast<std::uint64_t>(probe_transaction_count),
                     std::uint64_t{0}})
            {
                constexpr std::uint64_t kPrime = 1099511628211ull;
                std::uint64_t mixed = value;
                for (std::size_t byte = 0u; byte < sizeof(mixed); ++byte)
                {
                    probe_schedule_fingerprint ^= mixed & 0xffu;
                    probe_schedule_fingerprint *= kPrime;
                    mixed >>= 8u;
                }
            }
        }
        const MoEOverlayInferenceWorkloadIdentity probe_workload{
            .source = logits_phase == LogitsForwardPhase::Decode
                          ? ExpertHistogramSource::DecodeToken
                          : ExpertHistogramSource::PrefillChunk,
            .real_rows = seq_len,
            .execution_rows = probe_execution_rows,
            .transaction_count = probe_transaction_count,
            .speculative_depth = 0,
            .schedule_fingerprint = probe_schedule_fingerprint,
        };
        if (chunk_schedule_policy &&
            moe_overlay_coordinator_owns_prefill_probe_)
        {
            std::string declaration_error;
            if (!moe_overlay_inference_transaction_coordinator_ ||
                !moe_overlay_inference_transaction_coordinator_
                     ->declarePrefillInterferenceSchedule(
                         probe_workload, &declaration_error))
            {
                LOG_ERROR(
                    "RankOrchestrator could not declare its complete ExpertOverlay prefill calibration schedule: "
                    << declaration_error);
                return false;
            }
        }
        MoEOverlayInferenceInterferenceScope interference_scope(
            chunk_schedule_policy &&
                    moe_overlay_coordinator_owns_prefill_probe_
                ? nullptr
                : moe_overlay_interference_probe_.get(),
            probe_workload);
        if (logits_gatherer_)
        {
            logits_gatherer_->invalidate();
        }

        // TP timing diagnostic: explicit executor instrumentation only.
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
                    bool ok = false;
                    if (chunk_schedule_policy)
                    {
                        ok = runner_iface->forwardPrefillChunkSchedule(
                            tokens,
                            seq_len,
                            *chunk_schedule_policy,
                            chunk_schedule_pad_token_id,
                            chunk_schedule_allow_padded_execution);
                    }
                    else if (restored_prefix_bridge)
                    {
                        ok = runner_iface->forwardRestoredPrefixMTPDecodeBridge(
                            {.token_id = tokens[0],
                             .restored_prefix_tokens =
                                 restored_prefix_tokens});
                    }
                    else if (dispatch == MainForwardDispatch::Prefill)
                    {
                        ok = runner_iface->forwardPrefill(tokens, seq_len);
                    }
                    else
                    {
                        ok = runner_iface->forward(tokens, seq_len);
                    }
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
             dispatch,
             restored_prefix_tokens,
             restored_prefix_bridge,
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
                        bool ok = false;
                        if (chunk_schedule_policy)
                        {
                            ok = runner_iface->forwardPrefillChunkSchedule(
                                tokens,
                                seq_len,
                                *chunk_schedule_policy,
                                chunk_schedule_pad_token_id,
                                chunk_schedule_allow_padded_execution);
                        }
                        else if (restored_prefix_bridge)
                        {
                            ok = runner_iface
                                     ->forwardRestoredPrefixMTPDecodeBridge(
                                         {.token_id = tokens[0],
                                          .restored_prefix_tokens =
                                              restored_prefix_tokens});
                        }
                        else if (dispatch == MainForwardDispatch::Prefill)
                        {
                            ok = runner_iface->forwardPrefill(
                                tokens, seq_len);
                        }
                        else
                        {
                            ok = runner_iface->forward(tokens, seq_len);
                        }
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

            if (!all_success && config_.moe_device_controller_fabric)
            {
                LOG_ERROR(
                    "[RankOrchestrator] Fatal LocalTP ExpertOverlay epoch evidence "
                    << config_.moe_device_controller_fabric
                           ->describeInferenceEpochBarrier());
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
            const bool need_gather =
                ownsMainForwardLogits() &&
                logits_gatherer_ && logits_gatherer_->needsGather(logits_phase);

            if (!ownsMainForwardLogits())
            {
                PerfStatsCollector::addCounter(
                    "execution",
                    "rank_tp_hidden_only_forward_completions",
                    1.0,
                    {},
                    primaryDeviceId().toString(),
                    {{"boundary", "nested_non_head_pp_stage"},
                     {"phase",
                      logits_phase == LogitsForwardPhase::Decode
                          ? "decode"
                          : "prefill"},
                     {"host_logits_gathers", "0"}});
            }

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
            const size_t gather_rows =
                logits_phase == LogitsForwardPhase::Prefill
                    ? 1
                    : static_cast<size_t>(seq_len);
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
                    /*
                     * The worker owns residual host-buffer release, mmap host
                     * unregistration, backing-aware advice, and allocator
                     * trimming as one ordered operation.  Publishing this edge
                     * performs none of that maintenance on the inference
                     * authority thread.
                     */
                    const auto submission = wm->scheduleMmapReclaim();
                    if (debugEnv().vram_trace)
                    {
                        LOG_TRACE("[VRAM_TRACE] rank_mmap_reclaim phase=after_first_prefill submission="
                                  << MmapReclaimLifecycle::toString(submission));
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
    bool RankOrchestrator::forwardPP(
        const int *tokens,
        int seq_len,
        MainForwardDispatch dispatch,
        int restored_prefix_tokens)
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
        if (!validatePPGraphTransactionReady(
                "RankOrchestrator PP forward"))
        {
            return false;
        }

        const bool restored_prefix_bridge =
            dispatch == MainForwardDispatch::RestoredPrefixMTPDecodeBridge;
        if ((restored_prefix_bridge &&
             (!tokens || seq_len != 1 || restored_prefix_tokens <= 0)) ||
            (!restored_prefix_bridge && restored_prefix_tokens != 0))
        {
            LOG_ERROR(
                "RankOrchestrator::forwardPP received an invalid typed main-forward dispatch"
                << " dispatch=" << static_cast<int>(dispatch)
                << " seq_len=" << seq_len
                << " restored=" << restored_prefix_tokens);
            return false;
        }

        const LogitsForwardPhase logits_phase =
            (dispatch == MainForwardDispatch::Prefill || seq_len != 1)
                ? LogitsForwardPhase::Prefill
                : LogitsForwardPhase::Decode;
        last_logits_forward_phase_ = logits_phase;
        if (logits_gatherer_)
        {
            logits_gatherer_->invalidate();
        }

        const size_t num_stages =
            pp_graph_execution_plan_->segmentCount();
        std::size_t completed_segments = 0u;

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
        const auto execute_stage =
            [&](IInferenceRunner &runner, const int *stage_tokens)
        {
            if (restored_prefix_bridge)
            {
                return runner.forwardRestoredPrefixMTPDecodeBridge(
                    {.token_id = tokens[0],
                     .restored_prefix_tokens = restored_prefix_tokens});
            }
            if (dispatch == MainForwardDispatch::Prefill)
                return runner.forwardPrefill(stage_tokens, seq_len);
            return runner.forward(stage_tokens, seq_len);
        };
        if (!execute_stage(*stage0_runner, tokens))
        {
            LOG_ERROR("RankOrchestrator::forwardPP: Stage 0 forward failed");
            return false;
        }
        ++completed_segments;

        // =====================================================================
        // Intermediate stages: Transfer activations and continue execution
        // =====================================================================
        for (std::size_t plan_index = 1u;
             plan_index < pp_graph_execution_plan_->segments().size();
             ++plan_index)
        {
            const size_t stage_idx =
                pp_graph_execution_plan_->segments()[plan_index].stage_index;
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
            if (!execute_stage(*curr_runner, stage_tokens))
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << stage_idx << " forward failed");
                return false;
            }
            ++completed_segments;

            // Clear hidden state input for clean state on next forward
            curr_runner->clearHiddenStateInput();
        }

        // =====================================================================
        // Copy logits from last stage to combined buffer
        // =====================================================================
        const bool gather_host_logits =
            logits_phase == LogitsForwardPhase::Decode
                ? !skip_logits_gather_decode_
                : !skip_logits_gather_prefill_;
        if (gather_host_logits)
        {
            int last_stage = static_cast<int>(num_stages - 1);
            if (last_stage >= 0 && static_cast<size_t>(last_stage) < pp_stage_runners_.size() && pp_stage_runners_[last_stage])
            {
                if (!logits_gatherer_)
                {
                    logits_gatherer_ = std::make_unique<LogitsGatherer>(
                        0,
                        0,
                        logits_backend_resolver_);
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
        recordCompletedPPGraphTransactions(
            logits_phase == LogitsForwardPhase::Decode
                ? "decode"
                : "prefill",
            completed_segments);
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

    bool RankOrchestrator::consumeUnusedReplicatedMainLogitsPublications(
        const char *boundary)
    {
        if (device_runners_.size() < 2)
            return true;

        size_t consumed_participants = 0;
        for (size_t participant = 1;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *runner = device_runners_[participant].get();
            if (!runner)
            {
                LOG_ERROR("[RankOrchestrator] Replicated main-logits publication "
                          "has no participant " << participant
                          << " at boundary="
                          << (boundary ? boundary : "unknown"));
                return false;
            }

            /*
             * CPU execution is synchronous and owns no graph-stream token.
             * A GPU participant must explicitly implement the retirement API;
             * the interface default fails so an unrecognized nested runner
             * cannot silently leak or discard an ordering edge.
             */
            if (!runner->primaryDeviceId().is_gpu())
                continue;
            if (runner->hasLogitsLocal())
            {
                LOG_ERROR("[RankOrchestrator] Cannot retire a sharded main-logits "
                          "publication as an unused replica participant="
                          << participant << " boundary="
                          << (boundary ? boundary : "unknown"));
                return false;
            }
            if (!runner->consumeUnusedReplicatedMainLogitsPublication())
            {
                LOG_ERROR("[RankOrchestrator] Failed to retire unused replicated "
                          "main-logits publication participant="
                          << participant << " boundary="
                          << (boundary ? boundary : "unknown"));
                return false;
            }
            ++consumed_participants;
        }

        if (consumed_participants > 0)
        {
            PerfStatsCollector::addCounter(
                "sampling",
                "rank_unused_replicated_main_logits_publications_consumed",
                static_cast<double>(consumed_participants),
                "decode",
                "rank",
                {{"boundary", boundary ? boundary : "unknown"},
                 {"participants", std::to_string(device_runners_.size())},
                 {"sampler_participant", "0"}});
        }
        return true;
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
        {
            /*
             * CPU LocalTP has no graph-stream handoff to consume: the current
             * LOGITS_LOCAL tensors are ordinary host-resident shard rows.  Use
             * the non-consuming local views directly so ready prefill logits and
             * MTP condition logits are reduced by the same grouped argmax used
             * for verifier rows.  GPU children still route through
             * DeviceSampler::sampleGreedy(), whose consuming view enforces the
             * producer stream contract before launching per-shard argmax.
             */
            std::vector<LogitsLocalInfo> host_local_infos;
            host_local_infos.reserve(device_runners_.size());
            bool all_host_local_infos = true;
            for (const auto &runner : device_runners_)
            {
                LogitsLocalInfo info = runner ? runner->getLogitsLocalInfo() : LogitsLocalInfo{};
                if (!info || info.vocab_local == 0)
                {
                    all_host_local_infos = false;
                    break;
                }
                if (info.device.has_value() && info.device->is_gpu())
                {
                    all_host_local_infos = false;
                    break;
                }
                host_local_infos.push_back(info);
            }
            if (all_host_local_infos)
            {
                return DeviceSampler::sampleGreedyFromLocalInfos(
                    host_local_infos,
                    /*row=*/0);
            }
            return DeviceSampler::sampleGreedy(device_runners_);
        }
        if (!any_local_logits && !device_runners_.empty() && device_runners_[0])
        {
            const int token = device_runners_[0]->sampleGreedyOnDevice();
            if (token < 0 ||
                !consumeUnusedReplicatedMainLogitsPublications(
                    "sampleGreedyOnDevice"))
            {
                return -1;
            }
            return token;
        }
        return -1;
    }

    int RankOrchestrator::sampleOnDevice(const SamplingParams &params)
    {
        return sampleOnDeviceAtLogicalPosition(params, get_position());
    }

    int RankOrchestrator::sampleOnDeviceAtLogicalPosition(
        const SamplingParams &params,
        int logical_position)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleOnDeviceAtLogicalPosition(
                params,
                logical_position);
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
        {
            const float threshold =
                params.seed != 0
                    ? sampling_math::mtp_spec_threshold_from_seed(
                          static_cast<uint64_t>(params.seed),
                          logical_position,
                          /*draw_purpose=*/0)
                    : sampling_math::uniform01(
                          0xD1B54A32D192ED03ull,
                          static_cast<uint64_t>(std::max(0, logical_position)));
            return DeviceSampler::sample(device_runners_, params, threshold);
        }
        if (!any_local_logits && !device_runners_.empty() && device_runners_[0])
        {
            const int token =
                device_runners_[0]->sampleOnDeviceAtLogicalPosition(
                    params,
                    logical_position);
            if (token < 0 ||
                !consumeUnusedReplicatedMainLogitsPublications(
                    "sampleOnDeviceAtLogicalPosition"))
            {
                return -1;
            }
            return token;
        }
        return -1;
    }

    bool RankOrchestrator::publishMainLogitsBatchSamplesToDeviceResidentState(
        int request_count,
        const SamplingParams &params,
        const uint64_t *stochastic_position_seeds)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->publishMainLogitsBatchSamplesToDeviceResidentState(
                request_count,
                params,
                stochastic_position_seeds);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->publishMainLogitsBatchSamplesToDeviceResidentState(
                request_count,
                params,
                stochastic_position_seeds);
        }
        if (request_count <= 0 ||
            device_runners_.size() < 2 ||
            !usesMirroredMTPHeadForVerifier() ||
            !supportsDeviceResidentMTPSpecStatePublication())
        {
            return false;
        }

        /*
         * Every child must execute the sampler. Besides selecting the response
         * token, DeviceGraphOrchestrator publishes that child's initial target
         * position, sequence length, condition token, and accepted-count row
         * into its resident mailbox. Sampling only child zero and broadcasting
         * the token would leave the remaining publication transactions without
         * authoritative device-owned logical state.
         *
         * No token vector is materialized for host comparison. Mirrored
         * participants execute the same immutable sampling policy and seeds;
         * rank-level generation later authenticates their dispatch tickets and
         * terminal ledgers before surfacing the final response.
         */
        std::vector<std::future<bool>> futures;
        futures.reserve(device_runners_.size());
        for (size_t child_index = 0;
             child_index < device_runners_.size();
             ++child_index)
        {
            IInferenceRunner *child = device_runners_[child_index].get();
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->usesMirroredMTPHeadForVerifier() ||
                !child->supportsDeviceResidentMTPSpecStatePublication())
            {
                return false;
            }

            futures.push_back(std::async(
                std::launch::async,
                [child,
                 request_count,
                 &params,
                 stochastic_position_seeds]()
                {
                    return child
                        ->publishMainLogitsBatchSamplesToDeviceResidentState(
                        request_count,
                        params,
                        stochastic_position_seeds);
                }));
        }

        for (size_t child_index = 0; child_index < futures.size(); ++child_index)
        {
            try
            {
                if (!futures[child_index].get())
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP request-batch "
                              "prefill sampling failed on child "
                              << child_index);
                    return false;
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP request-batch "
                          "prefill sampling threw on child "
                          << child_index << ": " << e.what());
                return false;
            }
        }

        std::string mailbox_error;
        if (!adoptMirroredLocalTPResidentLogicalStateMailboxes(
                request_count,
                "request_batch_prefill_sample",
                &mailbox_error))
        {
            LOG_ERROR("[RankOrchestrator] Mirrored LocalTP request-batch "
                      "prefill could not adopt child logical-state mailboxes: "
                      << mailbox_error);
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_request_batch_prefill_publications",
            static_cast<double>(request_count),
            /*phase=*/"decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"sampling", params.is_greedy() ? "greedy" : "stochastic"},
             {"logical_state_owner", "child_device_mailboxes"},
             {"host_token_materializations", "0"},
             {"boundary", "first_grouped_verifier_transaction"}});
        return true;
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

        if (!ownsMainForwardLogits())
        {
            throw std::logic_error(
                "RankOrchestrator::logits: nested non-head PP stage publishes "
                "hidden state and has no logits boundary");
        }

        if (primaryDeviceId().is_gpu())
        {
            if (!last_logits_forward_phase_.has_value())
            {
                throw std::logic_error(
                    "RankOrchestrator::logits: GPU host logits were requested before a forward transaction");
            }

            const bool host_observation_disabled =
                *last_logits_forward_phase_ == LogitsForwardPhase::Decode
                    ? skip_logits_gather_decode_
                    : skip_logits_gather_prefill_;
            if (host_observation_disabled)
            {
                throw std::logic_error(
                    "RankOrchestrator::logits: GPU logits are device-owned for the active forward phase; use the device sampler/result publication API");
            }
        }

        // For PP mode: return combined logits (copied from final stage)
        if (mode_ == ParallelismMode::PP || mode_ == ParallelismMode::TP_PP)
        {
            const float *gathered = logits_gatherer_ ? logits_gatherer_->data() : nullptr;
            if (gathered)
                return gathered;
            if (primaryDeviceId().is_gpu())
            {
                throw std::logic_error(
                    "RankOrchestrator::logits: the GPU pipeline did not publish current host logits");
            }
            return nullptr;
        }

        // For TP mode: return combined logits if available (multi-device)
        if (logits_gatherer_ && logits_gatherer_->data() && device_runners_.size() > 1)
        {
            return logits_gatherer_->data();
        }

        // For single device, return primary device's logits
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->logits();
        }

        if (primaryDeviceId().is_gpu())
        {
            throw std::logic_error(
                "RankOrchestrator::logits: the active GPU forward did not publish current host logits");
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

    bool RankOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
        int draft_sample_slot,
        int position_offset)
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
            return device_runners_[0]
                ->forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
                draft_sample_slot,
                position_offset);
        }
        if (!supportsMTPDeviceDraftTokenInput() ||
            draft_sample_slot < 0 ||
            draft_sample_slot >= rank_stochastic_slot_capacity_ ||
            position_offset < 0)
        {
            LOG_ERROR("[RankOrchestrator] LocalTP device-draft sidecar requires staged child draft slots and live device positions on every participant");
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
             position_offset,
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
                       device_runners_[i]
                           ->forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
                           draft_sample_slot,
                           position_offset);
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
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling: Device "
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
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling: Device "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling: Re-throwing primary exception from device "
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
                 {"slot", std::to_string(draft_sample_slot)},
                 {"position_offset", std::to_string(position_offset)},
                 {"position_owner", "child_live_device_kv_counts"}});
        }
        return all_success;
    }

    bool RankOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
        int target_sample_slot)
    {
        if (finalPPSidecarRunner())
        {
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
                    target_sample_slot);
        }
        if (!supportsMTPDeviceDraftTokenInput() ||
            target_sample_slot < 0 ||
            target_sample_slot >= rank_stochastic_slot_capacity_)
        {
            LOG_ERROR("[RankOrchestrator] LocalTP device-target sidecar requires staged child target slots and live device positions on every participant");
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
                       device_runners_[i]
                           ->forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
                               target_sample_slot);
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
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling: Device "
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
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling: Device "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling: Re-throwing primary exception from device "
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
                 {"slot", std::to_string(target_sample_slot)},
                 {"position_owner", "child_live_device_kv_counts"}});
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

    bool RankOrchestrator::forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        int first_draft_slot,
        int slot_stride)
    {
        if (finalPPSidecarRunner())
        {
            LOG_ERROR("[RankOrchestrator] Resident request-batched MTP is a LocalTP operation and cannot cross a LocalPP sidecar boundary");
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
                    logical_state,
                    request_batch,
                    first_draft_slot,
                    slot_stride);
        }

        return dispatchMirroredLocalTPResidentMTPRequestBatch(
            logical_state,
            request_batch,
            first_draft_slot,
            slot_stride,
            "forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots",
            "resident_logical_state",
            [request_batch,
             first_draft_slot,
             slot_stride](
                IInferenceRunner &child,
                const DeviceResidentLogicalSequenceStateHandle &child_state)
            {
                return child
                    .forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
                        child_state,
                        request_batch,
                        first_draft_slot,
                        slot_stride);
            });
    }

    bool RankOrchestrator::advanceMTPRequestBatchConditionOnDevice(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        const SamplingParams &params,
        const uint64_t *stochastic_position_seeds)
    {
        if (finalPPSidecarRunner())
        {
            LOG_ERROR("[RankOrchestrator] Resident request-batch condition advance cannot cross a LocalPP sidecar boundary");
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->advanceMTPRequestBatchConditionOnDevice(
                logical_state,
                request_batch,
                params,
                stochastic_position_seeds);
        }

        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("[RankOrchestrator] Resident request-batch condition advance failed: "
                      << reason);
            return false;
        };
        const DeviceResidentLogicalSequenceStateHandle current_rank_state =
            deviceResidentLogicalSequenceState();
        if (device_runners_.size() < 2 || !tp_ctx_ ||
            !current_rank_state.valid() ||
            !logical_state.sameMailboxAs(current_rank_state) ||
            logical_state.request_count != request_batch ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            return fail("rank or child logical-state mailbox is stale or incomplete");
        }
        if (tp_ctx_->backend() != CollectiveBackendType::NCCL &&
            tp_ctx_->backend() != CollectiveBackendType::RCCL)
        {
            return fail("LocalTP condition advance requires NCCL/RCCL collectives");
        }
        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            if (!device_runners_[participant] ||
                !rank_resident_child_logical_state_handles_[participant].valid() ||
                rank_resident_child_logical_state_handles_[participant]
                        .request_count != request_batch)
            {
                return fail(
                    "participant " + std::to_string(participant) +
                    " has no matching resident mailbox");
            }
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            tp_worker_pool_->setFailureCallback([this]()
                                                {
                LOG_WARN("[TPWorkerPool] resident request-batch condition advance failed; aborting collective backend");
                if (tp_ctx_)
                    tp_ctx_->requestAbort(); });
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();

        /*
         * Main and sidecar graphs contain LocalTP collectives, so every child
         * must enter the grouped condition transition concurrently. Request
         * rows remain grouped inside each participant; the worker pool is only
         * the device-participant fanout.
         */
        tp_worker_pool_->dispatch(
            [this,
             request_batch,
             &params,
             stochastic_position_seeds,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t participant) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                const DeviceId device_id =
                    device_runners_[participant]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                if (debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
                {
                    LOG_INFO(
                        "[MTPConditionGraphContract] event=participant_enter"
                        << " participant=" << participant
                        << " device=" << device_id.toString()
                        << " requests=" << request_batch);
                }
                const bool success = device_runners_[participant]
                    ->advanceMTPRequestBatchConditionOnDevice(
                        rank_resident_child_logical_state_handles_[participant],
                        request_batch,
                        params,
                        stochastic_position_seeds);
                if (debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
                {
                    LOG_INFO(
                        "[MTPConditionGraphContract] event=participant_leave"
                        << " participant=" << participant
                        << " device=" << device_id.toString()
                        << " requests=" << request_batch
                        << " success=" << (success ? "true" : "false"));
                }
                return success;
            });

        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception;
        size_t first_exception_participant = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (auto &result : results)
        {
            if (!result.completed)
            {
                worker_timeout = true;
                all_success = false;
                continue;
            }
            if (result.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = result.exception;
                    first_exception_participant = result.worker_index;
                }
                continue;
            }
            all_success = all_success && result.success;
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "advanceMTPRequestBatchConditionOnDevice",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Resident request-batch condition advance rethrowing exception from participant "
                      << first_exception_participant);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        std::string mailbox_error;
        if (!adoptMirroredLocalTPResidentLogicalStateMailboxes(
                request_batch,
                "request_batch_condition_advance",
                &mailbox_error))
        {
            return fail("could not adopt advanced child mailboxes: " + mailbox_error);
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_request_batch_condition_advances",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(request_batch)},
             {"sampling", params.is_greedy() ? "greedy" : "stochastic"},
             {"logical_state_owner", "child_device_mailboxes"}});
        return true;
    }

    bool RankOrchestrator::forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        int first_condition_slot,
        int condition_slot_stride,
        int position_offset,
        int first_draft_slot,
        int draft_slot_stride)
    {
        const int64_t last_condition_slot =
            static_cast<int64_t>(first_condition_slot) +
            static_cast<int64_t>(request_batch - 1) *
                static_cast<int64_t>(condition_slot_stride);
        if (request_batch <= 0 ||
            first_condition_slot < 0 ||
            condition_slot_stride <= 0 ||
            last_condition_slot < 0 ||
            last_condition_slot >= rank_stochastic_slot_capacity_ ||
            position_offset <= 0)
        {
            LOG_ERROR("[RankOrchestrator] Chained resident request-batched MTP received invalid source-slot or depth geometry"
                      << " request_batch=" << request_batch
                      << " first_condition_slot=" << first_condition_slot
                      << " condition_slot_stride=" << condition_slot_stride
                      << " position_offset=" << position_offset);
            return false;
        }
        if (finalPPSidecarRunner())
        {
            LOG_ERROR("[RankOrchestrator] Chained resident request-batched MTP is a LocalTP operation and cannot cross a LocalPP sidecar boundary");
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
                    logical_state,
                    request_batch,
                    first_condition_slot,
                    condition_slot_stride,
                    position_offset,
                    first_draft_slot,
                    draft_slot_stride);
        }

        return dispatchMirroredLocalTPResidentMTPRequestBatch(
            logical_state,
            request_batch,
            first_draft_slot,
            draft_slot_stride,
            "forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots",
            "device_draft_slots",
            [request_batch,
             first_condition_slot,
             condition_slot_stride,
             position_offset,
             first_draft_slot,
             draft_slot_stride](
                IInferenceRunner &child,
                const DeviceResidentLogicalSequenceStateHandle &child_state)
            {
                return child
                    .forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
                        child_state,
                        request_batch,
                        first_condition_slot,
                        condition_slot_stride,
                        position_offset,
                        first_draft_slot,
                        draft_slot_stride);
            });
    }

    bool RankOrchestrator::dispatchLocalTPMTPMainCondition(
        const char *operation_name,
        const std::function<bool(IInferenceRunner &, size_t)> &child_operation)
    {
        const std::string operation =
            operation_name && operation_name[0] != '\0'
                ? operation_name
                : "mtp_main_condition";
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("RankOrchestrator::" << operation << ": " << reason);
            return false;
        };

        if (!child_operation)
            return fail("missing participant operation");
        if (device_runners_.size() < 2)
            return fail("requires at least two local participants");
        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            if (!device_runners_[participant] ||
                !device_runners_[participant]->primaryDeviceId().is_gpu())
            {
                return fail(
                    "participant " + std::to_string(participant) +
                    " is missing or is not GPU-backed");
            }
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            tp_worker_pool_->setFailureCallback([this, operation]()
                                                {
                LOG_WARN("[TPWorkerPool] " << operation
                         << " failure detected - aborting collective backend");
                if (tp_ctx_)
                    tp_ctx_->requestAbort(); });
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             child_operation,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t participant) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                const DeviceId device_id =
                    device_runners_[participant]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return child_operation(
                    *device_runners_[participant],
                    participant);
            });

        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (auto &result : results)
        {
            if (!result.completed)
            {
                worker_timeout = true;
                all_success = false;
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                continue;
            }
            if (result.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = result.exception;
                    first_exception_device = result.worker_index;
                }
                continue;
            }
            if (!result.success)
                all_success = false;
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                operation.c_str(),
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR(
                "RankOrchestrator::" << operation
                << ": rethrowing participant exception from device "
                << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_device_resident_main_condition_advances",
                1.0,
                "decode",
                "rank",
                {{"operation", operation},
                 {"participants", std::to_string(device_runners_.size())},
                 {"input_owner", "participant_device_logical_state"}});
        }
        return all_success;
    }

    bool RankOrchestrator::dispatchMirroredLocalTPResidentMTPRequestBatch(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        int first_draft_slot,
        int draft_slot_stride,
        const char *operation_name,
        const char *source_name,
        const std::function<bool(
            IInferenceRunner &,
            const DeviceResidentLogicalSequenceStateHandle &)> &child_operation)
    {
        const std::string operation =
            operation_name && operation_name[0] != '\0'
                ? operation_name
                : "resident_request_batch_mtp";
        const std::string source =
            source_name && source_name[0] != '\0'
                ? source_name
                : "unspecified";
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("RankOrchestrator::" << operation << ": " << reason);
            return false;
        };

        const int64_t last_draft_slot =
            static_cast<int64_t>(first_draft_slot) +
            static_cast<int64_t>(request_batch - 1) *
                static_cast<int64_t>(draft_slot_stride);
        if (request_batch <= 0 ||
            first_draft_slot < 0 ||
            draft_slot_stride <= 0 ||
            last_draft_slot < 0 ||
            last_draft_slot >= rank_stochastic_slot_capacity_)
        {
            return fail(
                "invalid destination-slot geometry: request_batch=" +
                std::to_string(request_batch) +
                " first_draft_slot=" + std::to_string(first_draft_slot) +
                " draft_slot_stride=" + std::to_string(draft_slot_stride));
        }
        if (!child_operation)
            return fail("missing grouped child operation");
        if (device_runners_.size() < 2 || !tp_ctx_)
            return fail("requires at least two LocalTP participants and a live collective context");
        if (tp_ctx_->backend() != CollectiveBackendType::NCCL &&
            tp_ctx_->backend() != CollectiveBackendType::RCCL)
        {
            return fail(
                std::string(
                    "requires NCCL/RCCL device-slot publication, got backend=") +
                collectiveBackendTypeToString(tp_ctx_->backend()));
        }
        if (!usesMirroredMTPHeadForVerifier() ||
            !supportsMTPDeviceDraftTokenInput() ||
            !supportsDeviceStochasticMTPVerification())
        {
            return fail(
                "requires mirrored full-vocabulary MTP heads and device-resident draft/verifier support on every participant");
        }

        /*
         * The rank handle is deliberately opaque. Comparing it with a freshly
         * materialized aggregate checks every pointer, stream, event, epoch, and
         * request-count field without dereferencing child device memory.
         */
        const DeviceResidentLogicalSequenceStateHandle current_rank_state =
            deviceResidentLogicalSequenceState();
        if (!current_rank_state.valid() ||
            !logical_state.sameMailboxAs(current_rank_state) ||
            logical_state.request_count != request_batch)
        {
            return fail(
                "received a stale, foreign, or shape-mismatched rank logical-state mailbox");
        }
        if (rank_resident_child_logical_state_handles_.size() !=
            device_runners_.size())
        {
            return fail("rank aggregate no longer has one child mailbox per participant");
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            const DeviceResidentLogicalSequenceStateHandle &child_state =
                rank_resident_child_logical_state_handles_[participant];
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->usesMirroredMTPHeadForVerifier() ||
                !child->supportsMTPDeviceDraftTokenInput() ||
                !child->supportsDeviceStochasticMTPVerification())
            {
                return fail(
                    "participant " + std::to_string(participant) +
                    " does not expose the mirrored device-resident MTP contract");
            }
            if (!child_state.valid() ||
                child_state.request_count != request_batch ||
                child_state.device != child->primaryDeviceId())
            {
                return fail(
                    "participant " + std::to_string(participant) +
                    " has a stale, foreign, or shape-mismatched child mailbox");
            }
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            tp_worker_pool_->setFailureCallback([this, operation]()
                                                {
                LOG_WARN("[TPWorkerPool] " << operation
                         << " failure detected - aborting collective backend");
                if (tp_ctx_)
                    tp_ctx_->requestAbort(); });
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();

        /*
         * The worker callback is participant-parallel and invokes exactly one
         * grouped child operation. Request rows remain a batch inside the child
         * graph; there is intentionally no rank-side loop over sidecar rows.
         */
        tp_worker_pool_->dispatch(
            [this,
             child_operation,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t participant) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                const DeviceId device_id =
                    device_runners_[participant]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return child_operation(
                    *device_runners_[participant],
                    rank_resident_child_logical_state_handles_[participant]);
            });

        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_participant = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (auto &result : results)
        {
            if (!result.completed)
            {
                LOG_ERROR("RankOrchestrator::" << operation
                          << ": participant " << result.worker_index
                          << " did not complete");
                worker_timeout = true;
                all_success = false;
                continue;
            }
            if (result.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = result.exception;
                    first_exception_participant = result.worker_index;
                }
                continue;
            }
            if (!result.success)
            {
                LOG_ERROR("RankOrchestrator::" << operation
                          << ": participant " << result.worker_index
                          << " rejected the grouped sidecar");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                operation.c_str(),
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::" << operation
                      << ": rethrowing primary exception from participant "
                      << first_exception_participant);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_resident_request_batch_sidecars",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(request_batch)},
             {"source", source},
             {"implementation", "grouped_child_graphs"}});
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_resident_request_batch_draft_slot_publications",
            static_cast<double>(request_batch),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(request_batch)},
             {"source", source},
             {"first_slot", std::to_string(first_draft_slot)},
             {"slot_stride", std::to_string(draft_slot_stride)},
             {"implementation", "participant_local_device_slots"},
             {"collective", "none"}});
        return true;
    }

    void RankOrchestrator::invalidateRankResidentLogicalStateAggregate(
        const char *lifecycle,
        const char *reason,
        int participant) const
    {
        const bool had_aggregate =
            !rank_resident_child_logical_state_handles_.empty();
        rank_resident_child_logical_state_handles_.clear();
        ++rank_resident_logical_state_epoch_;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_resident_logical_state_aggregate_invalidations",
            1.0,
            "decode",
            "rank",
            {{"lifecycle", lifecycle && lifecycle[0] != '\0'
                               ? lifecycle
                               : "unspecified"},
             {"reason", reason && reason[0] != '\0'
                            ? reason
                            : "unspecified"},
             {"participant", std::to_string(participant)},
             {"had_aggregate", had_aggregate ? "true" : "false"},
             {"rank_epoch", std::to_string(rank_resident_logical_state_epoch_)}});
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
            if (!device_runners_[i] || !child.valid())
            {
                invalidateRankResidentLogicalStateAggregate(
                    "aggregate_observation",
                    "cached_child_handle_invalid",
                    static_cast<int>(i));
                return {};
            }

            /*
             * A cached handle is only a rank-local ownership token. Ask the
             * child for its current mailbox before exposing the aggregate so a
             * prefix replacement, reset, workspace rebind, or publication epoch
             * change cannot leave a structurally valid but stale rank handle.
             * This compares pointers and events only; it performs no GPU work
             * and introduces no host synchronization.
             */
            const DeviceResidentLogicalSequenceStateHandle current_child =
                device_runners_[i]->deviceResidentLogicalSequenceState();
            if (!current_child.sameMailboxAs(child))
            {
                invalidateRankResidentLogicalStateAggregate(
                    "aggregate_observation",
                    "child_mailbox_no_longer_current",
                    static_cast<int>(i));
                return {};
            }
            if (request_count < 0)
                request_count = child.request_count;
            if (child.request_count != request_count)
            {
                invalidateRankResidentLogicalStateAggregate(
                    "aggregate_observation",
                    "child_request_count_mismatch",
                    static_cast<int>(i));
                return {};
            }
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
        handle.publication_generation = rank_resident_logical_state_epoch_;
        return handle;
    }

    bool RankOrchestrator::observeDeviceResidentNextConditionTokens(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_count,
        int32_t *out_tokens)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->observeDeviceResidentNextConditionTokens(
                logical_state,
                request_count,
                out_tokens);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->observeDeviceResidentNextConditionTokens(
                logical_state,
                request_count,
                out_tokens);
        }

        const DeviceResidentLogicalSequenceStateHandle current_rank_state =
            deviceResidentLogicalSequenceState();
        if (request_count <= 0 ||
            !out_tokens ||
            !current_rank_state.valid() ||
            !logical_state.sameMailboxAs(current_rank_state) ||
            logical_state.request_count < request_count ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size() ||
            device_runners_.empty() ||
            !device_runners_[0])
        {
            LOG_ERROR(
                "[RankOrchestrator] Resident next-condition-token observation "
                "received a stale, foreign, or incomplete aggregate mailbox");
            return false;
        }

        const DeviceResidentLogicalSequenceStateHandle &primary_state =
            rank_resident_child_logical_state_handles_[0];
        if (!primary_state.valid() ||
            primary_state.request_count < request_count ||
            primary_state.device != device_runners_[0]->primaryDeviceId())
        {
            LOG_ERROR(
                "[RankOrchestrator] Resident next-condition-token observation "
                "has no current primary-child mailbox");
            return false;
        }

        /*
         * Mirrored LocalTP publishes child zero as the authoritative compact
         * token owner. The host sees only that final result row; every execution
         * consumer continues to use its own child-local device mailbox.
         */
        if (!device_runners_[0]->observeDeviceResidentNextConditionTokens(
                primary_state,
                request_count,
                out_tokens))
        {
            LOG_ERROR(
                "[RankOrchestrator] Primary child could not publish resident "
                "next-condition-token result rows");
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_resident_next_condition_token_host_observations",
            1.0,
            "decode",
            "rank",
            {{"requests", std::to_string(request_count)},
             {"owner", "primary_child"},
             {"boundary", "host_result"}});
        return true;
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
        bool allow_speculative_discard)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->commitMTPShiftedRowFromDeviceTargetSample(
                target_sample_slot,
                already_appended_tokens,
                allow_speculative_discard);
        }
        if (device_runners_.empty())
        {
            return false;
        }
        if (device_runners_.size() == 1)
        {
            return device_runners_[0] &&
                   device_runners_[0]->commitMTPShiftedRowFromDeviceTargetSample(
                       target_sample_slot,
                       already_appended_tokens,
                       allow_speculative_discard);
        }

        /*
         * LocalTP resolves the first stochastic/greedy target token once at rank
         * scope and stages that token into the same device slot on every child.
         * The shifted MTP KV row must therefore be rebuilt on every participant
         * from that child-owned slot.  This is not a fallback: each child runs
         * its normal device-resident KV-only sidecar commit against its own MTP
         * cache, preserving the symmetric collective order used by the rest of
         * the grouped verifier transaction.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(device_runners_.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] device-target shifted commit failure detected - aborting collective backend");
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
             already_appended_tokens,
             allow_speculative_discard,
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

                if (i >= device_runners_.size() || !device_runners_[i])
                    return false;
                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                return device_runners_[i]->commitMTPShiftedRowFromDeviceTargetSample(
                           target_sample_slot,
                           already_appended_tokens,
                           allow_speculative_discard);
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
            LOG_ERROR("[RankOrchestrator] Device-target shifted commit rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_device_target_shifted_commits",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"target_slot", std::to_string(target_sample_slot)},
                 {"already_appended", std::to_string(already_appended_tokens)}});
        }
        return all_success;
    }

    bool RankOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index,
        int already_appended_tokens,
        bool allow_speculative_discard)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->commitMTPShiftedRowFromDeviceResidentLogicalState(
                logical_state,
                request_index,
                already_appended_tokens,
                allow_speculative_discard);
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
                               allow_speculative_discard);
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
                invalidateRankResidentLogicalStateAggregate(
                    "shifted_row_commit",
                    "child_mailboxes_consumed",
                    /*participant=*/-1);
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
            allow_speculative_discard);
    }

    bool RankOrchestrator::commitMTPShiftedRowsFromDeviceOutcome(
        const DeviceSpeculativeOutcomeHandle &outcome,
        int request_index,
        int already_appended_tokens,
        int max_state_commit_rows,
        int main_forward_token_count,
        bool allow_speculative_discard)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->commitMTPShiftedRowsFromDeviceOutcome(
                outcome,
                request_index,
                already_appended_tokens,
                max_state_commit_rows,
                main_forward_token_count,
                allow_speculative_discard);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->commitMTPShiftedRowsFromDeviceOutcome(
                outcome,
                request_index,
                already_appended_tokens,
                max_state_commit_rows,
                main_forward_token_count,
                allow_speculative_discard);
        }

        std::string outcome_error;
        if (!validateCurrentMirroredLocalTPOutcome(
                outcome,
                "shifted MTP suffix commit",
                &outcome_error))
        {
            LOG_ERROR("[RankOrchestrator] " << outcome_error);
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
                    LOG_WARN("[TPWorkerPool] device-outcome shifted commit failure detected - aborting collective backend");
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
             max_state_commit_rows,
             main_forward_token_count,
             allow_speculative_discard,
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
                       device_runners_[i]->commitMTPShiftedRowsFromDeviceOutcome(
                           rank_mirrored_child_outcomes_[i],
                           request_index,
                           already_appended_tokens,
                           max_state_commit_rows,
                           main_forward_token_count,
                           allow_speculative_discard);
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
            LOG_ERROR("[RankOrchestrator] Device-outcome shifted commit rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_device_outcome_shifted_commits",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_index", std::to_string(request_index)},
                 {"rows", std::to_string(std::max(0, max_state_commit_rows - already_appended_tokens))}});
        }
        return all_success;
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

    bool RankOrchestrator::commitMTPInitialShiftedRowFromDeviceOutcome(
        const PrefixStateSnapshot &checkpoint,
        const DeviceSpeculativeOutcomeHandle &outcome,
        int request_index,
        int main_forward_token_count,
        bool allow_speculative_discard)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_initial_shifted_device_outcome_commit_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (!checkpoint.valid || request_index < 0 || main_forward_token_count <= 0)
            return false;

        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            const PrefixStateSnapshot *tail_checkpoint = &checkpoint;
            if (!checkpoint.participant_snapshots.empty())
                tail_checkpoint = &checkpoint.participant_snapshots.back();
            return pp_sidecar->commitMTPInitialShiftedRowFromDeviceOutcome(
                *tail_checkpoint,
                outcome,
                request_index,
                main_forward_token_count,
                allow_speculative_discard);
        }
        if (device_runners_.empty())
            return false;
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            const PrefixStateSnapshot *child_checkpoint = &checkpoint;
            if (!checkpoint.participant_snapshots.empty())
                child_checkpoint = &checkpoint.participant_snapshots.front();
            return device_runners_[0]->commitMTPInitialShiftedRowFromDeviceOutcome(
                *child_checkpoint,
                outcome,
                request_index,
                main_forward_token_count,
                allow_speculative_discard);
        }

        const bool aggregate_checkpoint =
            !checkpoint.participant_snapshots.empty();
        const bool shared_logical_checkpoint =
            !aggregate_checkpoint && checkpoint.logical_checkpoint;
        if (aggregate_checkpoint &&
            checkpoint.participant_snapshots.size() != device_runners_.size())
        {
            LOG_ERROR("[RankOrchestrator] Device-outcome initial shifted MTP commit checkpoint count mismatch: snapshots="
                      << checkpoint.participant_snapshots.size()
                      << " runners=" << device_runners_.size());
            return false;
        }
        auto child_checkpoint = [&](size_t i) -> const PrefixStateSnapshot *
        {
            if (aggregate_checkpoint)
                return &checkpoint.participant_snapshots[i];
            if (shared_logical_checkpoint)
                return &checkpoint;
            return nullptr;
        };

        std::string outcome_error;
        if (!validateCurrentMirroredLocalTPOutcome(
                outcome,
                "initial shifted MTP row commit",
                &outcome_error))
        {
            LOG_ERROR("[RankOrchestrator] " << outcome_error);
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
                    LOG_WARN("[TPWorkerPool] device-outcome initial shifted commit failure detected - aborting collective backend");
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
             &child_checkpoint,
             request_index,
             main_forward_token_count,
             allow_speculative_discard,
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

                if (i >= device_runners_.size() ||
                    !device_runners_[i])
                {
                    return false;
                }
                const PrefixStateSnapshot *child = child_checkpoint(i);
                if (!child)
                {
                    LOG_ERROR("[RankOrchestrator] Device-outcome initial shifted MTP commit has no participant checkpoint for child "
                              << i);
                    return false;
                }

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                return device_runners_[i]->commitMTPInitialShiftedRowFromDeviceOutcome(
                    *child,
                    rank_mirrored_child_outcomes_[i],
                    request_index,
                    main_forward_token_count,
                    allow_speculative_discard);
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
            LOG_ERROR("[RankOrchestrator] Device-outcome initial shifted commit rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_device_outcome_initial_shifted_commits",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_index", std::to_string(request_index)}});
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
                    LOG_ERROR("RankOrchestrator::mtpLogits: mixed local and replicated MTP logits are invalid");
                    return nullptr;
                }

                LogitsLocalInfo info =
                    runner->consumeMTPLogitsLocalInfoForHostGather();
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
                mtp_logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    full_vocab,
                    1,
                    logits_backend_resolver_);
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
        if (out_token)
            *out_token = -1;
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
        if (usesMirroredMTPHeadForVerifier())
        {
            /*
             * Mirrored LocalTP MTP heads expose full-vocabulary sidecar logits
             * on every participant.  Sampling each child directly keeps the
             * draft-token slots device-resident and avoids a tiny rank-level
             * allreduce/gather.  When the caller still asks for a host shadow,
             * use it only as a consistency check and response mirror.
             */
            bool have_shadow = false;
            int32_t rank_token = -1;
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                IInferenceRunner *runner = device_runners_[child].get();
                if (!runner)
                    return false;

                int32_t child_token = -1;
                int32_t *child_shadow = out_token ? &child_token : nullptr;
                if (!runner->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                        draft_sample_slot,
                        child_shadow))
                {
                    LOG_ERROR("RankOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot: "
                              "mirrored child " << child
                                                << " failed device draft-slot sampling");
                    return false;
                }

                if (out_token)
                {
                    if (child_token < 0)
                    {
                        LOG_ERROR("RankOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot: "
                                  "mirrored child " << child
                                                    << " returned an invalid draft token");
                        return false;
                    }
                    if (!have_shadow)
                    {
                        rank_token = child_token;
                        have_shadow = true;
                    }
                    else if (child_token != rank_token)
                    {
                        std::vector<std::vector<MTPMirroredTensorDigest>>
                            participant_digests;
                        participant_digests.reserve(device_runners_.size());
                        for (const auto &participant : device_runners_)
                        {
                            auto *device_orchestrator =
                                dynamic_cast<DeviceGraphOrchestrator *>(
                                    participant.get());
                            participant_digests.push_back(
                                device_orchestrator
                                    ? device_orchestrator
                                          ->captureMirroredMTPDigests()
                                    : std::vector<MTPMirroredTensorDigest>{});
                        }

                        LOG_ERROR("RankOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot: "
                                  "mirrored children sampled different draft tokens "
                                  << rank_token << " and " << child_token
                                  << describeMirroredDigestMismatch(
                                         participant_digests));
                        return false;
                    }
                }
            }

            if (out_token)
                *out_token = rank_token;

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_mtp_draft_slot_samples",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(draft_sample_slot)},
                 {"host_shadow", out_token ? "true" : "false"}});
            return true;
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

        /*
         * The main LM head and the MTP verifier head have independent
         * distribution policies.  In particular, LocalTP may shard the main
         * LM head while mirroring the much smaller MTP head.  Never infer the
         * former from usesMirroredMTPHeadForVerifier(): doing so would
         * sample only participant zero's main-logits shard and then incorrectly
         * retire participant one as though it owned a redundant full-vocabulary
         * row.
         */
        bool any_local_main_logits = false;
        bool all_local_main_logits = true;
        for (const auto &runner : device_runners_)
        {
            const bool has_local = runner && runner->hasLogitsLocal();
            any_local_main_logits = any_local_main_logits || has_local;
            all_local_main_logits = all_local_main_logits && has_local;
        }
        if (any_local_main_logits != all_local_main_logits)
        {
            LOG_ERROR("[RankOrchestrator] LocalTP main-target argmax has mixed "
                      "sharded and replicated main-logits declarations");
            return false;
        }
        if (all_local_main_logits)
        {
            return sampleRankGreedyMainLogitsToLocalTPTargetSlot(
                target_sample_slot,
                out_token);
        }

        if (usesMirroredMTPHeadForVerifier())
        {
            if (target_sample_slot < 0 ||
                target_sample_slot >= rank_stochastic_slot_capacity_ ||
                device_runners_.size() < 2)
            {
                return false;
            }
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                const IInferenceRunner *runner = device_runners_[child].get();
                if (!runner ||
                    !runner->usesMirroredMTPHeadForVerifier() ||
                    !runner->supportsDeviceStochasticMTPVerification())
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP main-target argmax requires a full-head device sampler on participant "
                              << child);
                    return false;
                }
            }

            /*
             * The mirrored rows are identical, so one argmax is sufficient.
             * Keep the normal serving path entirely device-owned by requesting
             * no host shadow from child zero, then broadcast its target mailbox
             * slot through the configured LocalTP collective. A host value is
             * materialized only for callers that explicitly request one.
             */
            int32_t primary_shadow = -1;
            if (!device_runners_.front()
                     ->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                         target_sample_slot,
                         out_token ? &primary_shadow : nullptr))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP main-target argmax failed on primary participant");
                return false;
            }
            if (!broadcastPrimaryLocalTPMainTargetSlotToChildren(
                    target_sample_slot))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP main-target argmax could not broadcast the primary device slot");
                return false;
            }
            if (out_token)
                *out_token = primary_shadow;

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_main_target_argmax_samples",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(target_sample_slot)},
                 {"host_shadow", out_token ? "true" : "false"},
                 {"implementation", "primary_child_device_slot_broadcast"}});
            return true;
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

            LogitsLocalInfo info =
                runner->consumeAllPositionLogitsLocalInfoForSampling();
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
            draft_token_count > rank_max_verifier_rows_ ||
            compare_rows < 0 ||
            compare_rows > rank_max_draft_depth_ ||
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
        std::vector<int32_t> verifier_tokens_i32(
            static_cast<size_t>(draft_token_count),
            -1);
        if (!sampleGreedyFromAllPositionLogitsOnDeviceRows(
                /*start_row=*/0,
                draft_token_count,
                verifier_tokens_i32.data()))
        {
            return false;
        }

        std::vector<int> verifier_tokens(
            static_cast<size_t>(draft_token_count),
            -1);
        std::vector<int> packed_draft_tokens(
            static_cast<size_t>(draft_token_count),
            -1);
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

        std::vector<int> output_tokens_int(
            static_cast<size_t>(rank_compact_output_token_stride_),
            -1);
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        meta.fill(0);
        summarize_greedy_speculative_verify_batch(
            static_cast<int>(resolved_draft_tokens[0]),
            verifier_tokens.data(),
            packed_draft_tokens.data(),
            compare_rows,
            packed_stop_tokens.data(),
            stop_token_count,
            output_tokens_int.data(),
            static_cast<int>(output_tokens_int.size()),
            meta.data());

        std::vector<int32_t> output_tokens(
            static_cast<size_t>(rank_compact_output_token_stride_),
            -1);
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

    bool RankOrchestrator::prepareGreedyAllPositionBatchOutcomeGraph(
        int verifier_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->prepareGreedyAllPositionBatchOutcomeGraph(
                verifier_token_count,
                stop_tokens,
                stop_token_count,
                penalty_policy);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->prepareGreedyAllPositionBatchOutcomeGraph(
                    verifier_token_count,
                    stop_tokens,
                    stop_token_count,
                    penalty_policy);
        }
        if (device_runners_.size() < 2 ||
            !usesMirroredMTPHeadForVerifier())
        {
            LOG_ERROR("[RankOrchestrator] Multi-device graph-owned greedy outcomes require a mirrored LocalTP MTP head");
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            const auto &child = device_runners_[i];
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->usesMirroredMTPHeadForVerifier())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP graph-owned greedy outcome participant "
                          << i << " is not eligible");
                return false;
            }
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i]
                     ->prepareGreedyAllPositionBatchOutcomeGraph(
                         verifier_token_count,
                         stop_tokens,
                         stop_token_count,
                         penalty_policy))
            {
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                throw std::runtime_error(
                    "Failed to arm every mirrored LocalTP graph-owned greedy "
                    "outcome participant");
            }
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_graph_owned_greedy_outcome_transactions_armed",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"rows", std::to_string(verifier_token_count)}});
        return true;
    }

    bool RankOrchestrator::configureMTPRequestStopTokens(
        const std::vector<int32_t> &stop_tokens)
    {
        auto configure_runners =
            [&stop_tokens](
                std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                const char *group)
        {
            for (size_t i = 0; i < runners.size(); ++i)
            {
                if (!runners[i])
                {
                    throw std::logic_error(
                        std::string{"Cannot configure MTP request stop tokens "
                                    "on a null "} +
                        group + " participant");
                }
                if (!runners[i]->configureMTPRequestStopTokens(stop_tokens))
                {
                    throw std::runtime_error(
                        std::string{"MTP request stop-token configuration was "
                                    "rejected by "} +
                        group + " participant " + std::to_string(i));
                }
            }
        };

        configure_runners(device_runners_, "device");
        configure_runners(pp_stage_runners_, "pipeline");
        return true;
    }

    bool RankOrchestrator::configureMTPRequestPenaltyPolicy(
        const MTPRequestPenaltyPolicy &policy)
    {
        auto configure_runners =
            [&policy](
                std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                const char *group)
        {
            for (size_t i = 0; i < runners.size(); ++i)
            {
                if (!runners[i])
                {
                    throw std::logic_error(
                        std::string{"Cannot configure MTP request penalty policy "
                                    "on a null "} +
                        group + " participant");
                }
                if (!runners[i]->configureMTPRequestPenaltyPolicy(policy))
                {
                    throw std::runtime_error(
                        std::string{"MTP request penalty-policy configuration "
                                    "was rejected by "} +
                        group + " participant " + std::to_string(i));
                }
            }
        };

        configure_runners(device_runners_, "device");
        configure_runners(pp_stage_runners_, "pipeline");
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
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::None;
        rank_mirrored_child_outcomes_.clear();
        rank_mirrored_primary_outcome_ = DeviceSpeculativeOutcomeHandle{};
        rank_mirrored_child_outcomes_valid_ = false;
        rank_resident_child_logical_state_handles_.clear();

        if (usesMirroredMTPHeadForVerifier())
        {
            return verifyGreedyMirroredLocalTPBatchOutcomeOnDeviceResident(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out_handle);
        }

        if (primaryDeviceId().is_gpu())
        {
            LOG_ERROR("[RankOrchestrator] GPU LocalTP MTP resident greedy verification requires mirrored full-head child verifier outcomes");
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
        if (outcome.output_tokens.size() > rank_compact_output_tokens_.size())
            return false;
        std::fill(
            rank_compact_output_tokens_.begin(),
            rank_compact_output_tokens_.end(),
            -1);
        std::copy(
            outcome.output_tokens.begin(),
            outcome.output_tokens.end(),
            rank_compact_output_tokens_.begin());
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
        rank_compact_output_meta_[kSpecBatchMetaCommitBoundaryClipped] =
            outcome.commit_boundary_clipped ? 1 : 0;

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
        out_handle->logical_verifier_rows_per_request = draft_token_count;
        out_handle->physical_verifier_rows_per_request = draft_token_count;
        out_handle->output_token_stride = rank_compact_output_token_stride_;
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

    bool RankOrchestrator::verifyGreedyMirroredLocalTPBatchOutcomeOnDeviceResident(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        using namespace sampling_math;
        if (!out_handle)
            return false;

        const int compare_rows = draft_token_count - 1;
        if (device_runners_.size() < 2 ||
            !draft_tokens ||
            draft_token_count <= 0 ||
            draft_token_count > rank_max_verifier_rows_ ||
            compare_rows < 0 ||
            compare_rows > rank_max_draft_depth_ ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens))
        {
            return false;
        }

        std::vector<DeviceSpeculativeOutcomeHandle> child_outcomes(
            device_runners_.size());
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            IInferenceRunner *child = device_runners_[i].get();
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->usesMirroredMTPHeadForVerifier() ||
                !child->supportsDeviceResidentMTPSpecStatePublication())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP greedy MTP requires every child to expose a GPU mirrored-head resident publisher; participant "
                          << i << " is not ready");
                return false;
            }

            /*
             * Pass the verifier token shadows through unchanged.  If the first
             * token or drafts are device-resident, the child reducer reads the
             * already-materialized verifier input row from its own arena buffer;
             * resolving the shadows here would reintroduce a host ownership edge.
             */
            if (!child->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                    draft_tokens,
                    draft_token_count,
                    stop_tokens,
                    stop_token_count,
                    &child_outcomes[i]) ||
                !child_outcomes[i].valid())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP greedy MTP child "
                          << i << " failed resident verifier reduction");
                return false;
            }
        }
        const bool every_child_published_in_graph =
            std::all_of(
                child_outcomes.begin(),
                child_outcomes.end(),
                [](const DeviceSpeculativeOutcomeHandle &outcome)
                {
                    return outcome
                        .mirrored_local_tp_locally_complete;
                });
        if (!every_child_published_in_graph)
        {
            if (tp_ctx_)
                tp_ctx_->requestAbort();
            throw std::runtime_error(
                "Mirrored LocalTP greedy outcomes were not produced locally "
                "by every participant");
        }

        rank_mirrored_child_outcomes_ = std::move(child_outcomes);
        rank_mirrored_primary_outcome_ =
            rank_mirrored_child_outcomes_.front();
        *out_handle = rank_mirrored_primary_outcome_;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::MirroredGreedy;
        rank_compact_outcome_valid_ = out_handle->valid();
        rank_mirrored_child_outcomes_valid_ = rank_compact_outcome_valid_;

        if (rank_compact_outcome_valid_)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_greedy_resident_outcomes",
                static_cast<double>(draft_token_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"compare_rows", std::to_string(compare_rows)},
                 {"implementation",
                  "graph_captured_participant_local_outcomes"},
                 {"collective", "none"}});
        }
        return rank_compact_outcome_valid_;
    }

    bool RankOrchestrator::verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
        const DeviceGreedyBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                out_handle);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                out_handle);
        }
        if (out_handle)
            *out_handle = DeviceSpeculativeOutcomeHandle{};
        rank_compact_outcome_valid_ = false;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::None;
        rank_mirrored_child_outcomes_.clear();
        rank_mirrored_primary_outcome_ = DeviceSpeculativeOutcomeHandle{};
        rank_mirrored_child_outcomes_valid_ = false;
        rank_resident_child_logical_state_handles_.clear();

        if (!out_handle ||
            !requests ||
            request_count <= 0 ||
            device_runners_.size() < 2)
        {
            return false;
        }
        if (!usesMirroredMTPHeadForVerifier())
        {
            if (primaryDeviceId().is_gpu())
            {
                LOG_ERROR("[RankOrchestrator] GPU LocalTP MTP resident greedy request batching requires mirrored full-head child verifier outcomes");
            }
            return false;
        }

        std::vector<DeviceSpeculativeOutcomeHandle> child_outcomes(
            device_runners_.size());
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            IInferenceRunner *child = device_runners_[i].get();
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->usesMirroredMTPHeadForVerifier() ||
                !child->supportsDeviceResidentMTPSpecStatePublication())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP greedy request-batch MTP requires every child to expose a GPU mirrored-head resident publisher; participant "
                          << i << " is not ready");
                return false;
            }
            if (!child->verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
                    requests,
                    request_count,
                    &child_outcomes[i]) ||
                !child_outcomes[i].valid())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP greedy request-batch MTP child "
                          << i << " failed resident verifier reduction");
                return false;
            }
        }
        if (!validateMirroredLocalTPChildOutcomesComplete(
                child_outcomes,
                "rank_mirrored_localtp_greedy_request_batch_local_outcomes"))
        {
            return false;
        }

        rank_mirrored_child_outcomes_ = std::move(child_outcomes);
        rank_mirrored_primary_outcome_ =
            rank_mirrored_child_outcomes_.front();
        *out_handle = rank_mirrored_primary_outcome_;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::MirroredGreedy;
        rank_compact_outcome_valid_ = out_handle->valid();
        rank_mirrored_child_outcomes_valid_ = rank_compact_outcome_valid_;

        if (rank_compact_outcome_valid_)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_greedy_request_batch_resident_outcomes",
                static_cast<double>(request_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"implementation", "participant_local_device_outcomes"},
                 {"collective", "none"}});
        }
        return rank_compact_outcome_valid_;
    }

    bool RankOrchestrator::verifyStochasticMirroredLocalTPRequestBatchOutcomesOnDeviceResident(
        const DeviceStochasticBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        if (!out_handle)
            return false;
        *out_handle = DeviceSpeculativeOutcomeHandle{};

        if (!requests ||
            request_count <= 0 ||
            device_runners_.size() < 2)
        {
            return false;
        }

        std::vector<DeviceSpeculativeOutcomeHandle> child_outcomes(
            device_runners_.size());
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            IInferenceRunner *child = device_runners_[i].get();
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->usesMirroredMTPHeadForVerifier() ||
                !child->supportsDeviceStochasticMTPVerification() ||
                !child->supportsDeviceResidentMTPSpecStatePublication())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic MTP requires every child to expose a GPU mirrored-head resident stochastic publisher; participant "
                          << i << " is not ready");
                return false;
            }

            /*
             * The descriptors are value-owned and already name child-local target
             * distribution slots, draft-token slots, first-token slots, and RNG
             * draws. Forward the exact same descriptor list to every mirrored
             * child so each participant has a local compact mailbox; the
             * device-side broadcast below then replaces those mailbox contents
             * with the primary child's authoritative rank outcome.
             */
            if (!child->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                    requests,
                    request_count,
                    &child_outcomes[i]) ||
                !child_outcomes[i].valid())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic MTP child "
                          << i << " failed resident verifier reduction");
                return false;
            }
        }
        if (!validateMirroredLocalTPChildOutcomesComplete(
                child_outcomes,
                "rank_mirrored_localtp_stochastic_local_outcomes"))
        {
            return false;
        }

        rank_mirrored_child_outcomes_ = std::move(child_outcomes);
        rank_mirrored_primary_outcome_ =
            rank_mirrored_child_outcomes_.front();
        *out_handle = rank_mirrored_primary_outcome_;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::MirroredStochastic;
        rank_compact_outcome_valid_ = out_handle->valid();
        rank_mirrored_child_outcomes_valid_ = rank_compact_outcome_valid_;

        if (rank_compact_outcome_valid_)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_resident_outcomes",
                static_cast<double>(request_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"implementation", "participant_local_device_outcomes"},
                 {"collective", "none"}});
        }
        return rank_compact_outcome_valid_;
    }

    bool RankOrchestrator::validateMirroredLocalTPChildOutcomesComplete(
        const std::vector<DeviceSpeculativeOutcomeHandle> &child_outcomes,
        const char *context_name)
    {
        const std::string operation_name =
            context_name && context_name[0] != '\0'
                ? std::string(context_name)
                : std::string("rank_mirrored_localtp_local_outcomes");
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("[RankOrchestrator] " << operation_name << ": " << reason);
            return false;
        };

        if (device_runners_.size() < 2 ||
            child_outcomes.size() != device_runners_.size())
        {
            return fail(
                "participant-local outcome validation requires one child handle per LocalTP participant");
        }

        const DeviceSpeculativeOutcomeHandle &primary = child_outcomes.front();
        if (!primary.valid())
            return fail("primary mirrored child outcome is invalid");

        /*
         * Production never copies these compact rows to compare them. The
         * grouped verifier parity suites prove byte identity across the same
         * deterministic local kernels. Runtime validates only the complete
         * ownership contract and fails hard if any participant did not execute
         * the participant-local path.
         */
        for (size_t i = 0; i < child_outcomes.size(); ++i)
        {
            const DeviceSpeculativeOutcomeHandle &child = child_outcomes[i];
            IInferenceRunner *runner = device_runners_[i].get();
            if (!runner)
            {
                return fail("missing mirrored LocalTP participant " +
                            std::to_string(i));
            }
            if (!child.valid())
            {
                return fail("invalid mirrored child outcome for participant " +
                            std::to_string(i));
            }
            if (child.request_count != primary.request_count ||
                child.logical_verifier_rows_per_request !=
                    primary.logical_verifier_rows_per_request ||
                child.physical_verifier_rows_per_request !=
                    primary.physical_verifier_rows_per_request ||
                child.output_token_stride != primary.output_token_stride ||
                child.meta_stride != primary.meta_stride)
            {
                return fail(
                    "mirrored child outcome shape mismatch for participant " +
                    std::to_string(i));
            }
            if (child.device != runner->primaryDeviceId())
            {
                return fail(
                    "mirrored child outcome device does not match participant " +
                    std::to_string(i));
            }
            if (!child.device.is_gpu())
            {
                return fail(
                    "mirrored participant-local outcomes are GPU-only; participant " +
                    std::to_string(i) + " is " + child.device.toString());
            }
            if (!child.mirrored_local_tp_locally_complete)
                return fail(
                    "participant " +
                    std::to_string(i) +
                    " did not publish a complete local mirrored outcome");
            if (!child.stream || !child.response_ready_event)
            {
                return fail(
                    "participant " +
                    std::to_string(i) +
                    " did not retain its exact outcome producer stream/event");
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_participant_local_outcomes",
            static_cast<double>(child_outcomes.size()),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"request_count", std::to_string(primary.request_count)},
             {"implementation", "independent_identical_device_outcomes"},
             {"collective", "none"},
             {"context", operation_name}});
        return true;
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
            slot >= rank_stochastic_slot_capacity_ ||
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
            case DeviceLogitsSource::MainRequestBatch:
                // Live request-batch condition rows are mirrored and child-owned.
                // They must never enter the legacy rank-side shard reducer.
                return false;
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
                IBackend *backend = resolveLogitsBackend(*info.device);
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
            first_slot + row_count > rank_stochastic_slot_capacity_ ||
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
                    IBackend *backend = resolveLogitsBackend(*info.device);
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
            slot >= rank_stochastic_slot_capacity_)
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
            draft_sample_slot >= rank_stochastic_slot_capacity_)
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
            target_sample_slot >= rank_stochastic_slot_capacity_)
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
            slot >= rank_stochastic_slot_capacity_ ||
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
            first_slot + token_count > rank_stochastic_slot_capacity_)
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

    bool RankOrchestrator::broadcastPrimaryLocalTPMainTargetSlotToChildren(
        int slot)
    {
        if (device_runners_.size() < 2 ||
            !tp_ctx_ ||
            !usesMirroredMTPHeadForVerifier() ||
            slot < 0 ||
            slot >= rank_stochastic_slot_capacity_)
        {
            return false;
        }

        std::vector<DeviceStochasticSampleSlotHandle> child_slots(
            device_runners_.size());
        for (size_t child = 0; child < device_runners_.size(); ++child)
        {
            IInferenceRunner *runner = device_runners_[child].get();
            if (!runner ||
                !runner->usesMirroredMTPHeadForVerifier() ||
                !runner->supportsDeviceStochasticMTPVerification())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP "
                          "main-target slot broadcast requires every child to expose mirrored device stochastic support; participant "
                          << child << " is not ready");
                return false;
            }

            child_slots[child] =
                child == 0
                    ? runner->deviceStochasticTargetSampleProducerSlot(slot)
                    : runner
                          ->deviceStochasticTargetSampleBroadcastDestinationSlot(
                              slot);
            if (!child_slots[child].valid())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP "
                          "main-target slot broadcast could not acquire "
                          << (child == 0
                                  ? "ready primary producer"
                                  : "predecessor-ordered destination")
                          << " slot=" << slot
                          << " for participant " << child);
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
                    LOG_WARN("[TPWorkerPool] mirrored sample slot broadcast failure detected - aborting collective backend");
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
             slot,
             &child_slots,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t child) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto device_id = device_runners_[child]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                LocalTPCollectiveSidebandBuffer sampled_token;
                sampled_token.kind = LocalTPCollectiveSidebandKind::Broadcast;
                sampled_token.send_buffer =
                    child == 0 ? child_slots[0].token_device : nullptr;
                sampled_token.recv_buffer = child_slots[child].token_device;
                sampled_token.element_count = 1;
                sampled_token.dtype = CollectiveDataType::INT32;
                sampled_token.root_device_index = 0;
                sampled_token.name = "mtp_mirrored_main_target_token";

                const bool broadcast_ok = tp_ctx_->collectiveSidebandOnStream(
                    {sampled_token},
                    static_cast<int>(child),
                    child_slots[child].stream,
                    "mtp_rank_mirrored_main_target_token_broadcast");
                if (!broadcast_ok)
                    return false;

                /*
                 * The collective write is now the producer for every participant's
                 * slot, including the root.  Record a fresh event after the
                 * broadcast so sidecars and verifier reducers wait on the
                 * common rank-owned token rather than any pre-broadcast
                 * child-local sampler event.
                 */
                return device_runners_[child]
                    ->recordStochasticTargetSampleSlotReadyFromDevice(
                        slot,
                        child_slots[child].stream,
                        /*verifier_consumer_pending=*/true);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_child = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::broadcastPrimaryLocalTPMainTargetSlotToChildren: participant "
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
                    first_exception_child = r.worker_index;
                }
                continue;
            }
            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::broadcastPrimaryLocalTPMainTargetSlotToChildren: participant "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "broadcastPrimaryLocalTPMainTargetSlotToChildren",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::broadcastPrimaryLocalTPMainTargetSlotToChildren: rethrowing primary exception from participant "
                      << first_exception_child);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_main_target_slot_broadcasts",
                1.0,
                "decode",
                "rank",
                 {{"participants", std::to_string(device_runners_.size())},
                  {"slot", std::to_string(slot)},
                 {"buffer", "target"},
                 {"implementation", "primary_child_device_slot_broadcast"}});
        }
        return all_success;
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
            first_draft_slot + draft_token_count > rank_stochastic_slot_capacity_)
        {
            return nullptr;
        }
        if (first_token_from_device &&
            (first_target_sample_slot < 0 ||
             first_target_sample_slot >= rank_stochastic_slot_capacity_))
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
        if (slot < 0 || slot >= rank_stochastic_slot_capacity_)
            return -1;
        return rank_stochastic_draft_sample_tokens_[static_cast<size_t>(slot)];
    }

    int32_t RankOrchestrator::rankStochasticTargetSampleToken(int slot) const
    {
        if (slot < 0 || slot >= rank_stochastic_slot_capacity_)
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
         * GPU LocalTP does not advertise the rank-compact reducer as a production
         * resident verifier.  Every child must own a mirrored full-vocab verifier
         * head so it can reduce its own compact outcome and publish that exact
         * device handle later.
         */
        return supportsDeviceResidentMTPSpecStatePublication();
    }

    bool RankOrchestrator::usesMirroredMTPHeadForVerifier() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->usesMirroredMTPHeadForVerifier();
        }
        if (device_runners_.empty())
            return false;

        /*
         * The rank is only an aggregator here.  Mirroring is a property of the
         * child graph builders and their verifier heads, so the aggregate answer
         * is true exactly when every participant reports the mirrored full-vocab
         * verifier head.  Do not route through any rank-side compact reducer:
         * resident LocalTP MTP publication feeds each child its own device handle.
         */
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->usesMirroredMTPHeadForVerifier();
            });
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
         * Stochastic GPU LocalTP mirrors the full verifier head so each child runs
         * the native full-vocabulary resident stochastic reducer.  Rank-owned
         * sharded top-k tables are useful diagnostics, but they are not a
         * production publication path.
         */
        return device_runners_.size() >= 2 &&
               supportsDeviceResidentMTPSpecStatePublication() &&
               std::all_of(
                   device_runners_.begin(),
                   device_runners_.end(),
                   [](const std::unique_ptr<IInferenceRunner> &runner)
                   {
                       return runner &&
                              runner->supportsDeviceStochasticMTPVerification();
                   });
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
        if (slot >= 0 && slot < rank_stochastic_slot_capacity_)
        {
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)] = false;
        }
        if ((source == DeviceLogitsSource::Main ||
             source == DeviceLogitsSource::MainRequestBatch) &&
            usesMirroredMTPHeadForVerifier())
        {
            if (buffer != DeviceDistributionBuffer::Target ||
                slot < 0 ||
                slot >= rank_stochastic_slot_capacity_ ||
                device_runners_.size() < 2)
            {
                return false;
            }
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                const IInferenceRunner *runner = device_runners_[child].get();
                if (!runner ||
                    !runner->usesMirroredMTPHeadForVerifier() ||
                    !runner->supportsDeviceStochasticMTPVerification())
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic main-target distribution requires full-head device support on participant "
                              << child);
                    return false;
                }
            }

            /*
             * Main-target sampling needs one distribution and one common token,
             * not one independently sampled token per mirrored participant.
             * Build on child zero now; sampleStochasticDistributionOnDevice*()
             * will publish that child's target slot through NCCL/RCCL.
             */
            if (!device_runners_.front()->buildStochasticDistributionOnDevice(
                    source,
                    row,
                    buffer,
                    slot,
                    params,
                    vocab_size))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic main-target distribution build failed on primary participant");
                return false;
            }
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)] = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_distribution_builds",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"source",
                  source == DeviceLogitsSource::MainRequestBatch
                      ? "main_request_batch"
                      : "main"},
                 {"rows", "1"},
                 {"implementation", "primary_child_full_vocab"}});
            return true;
        }
        if (source == DeviceLogitsSource::AllPosition &&
            usesMirroredMTPHeadForVerifier())
        {
            for (size_t i = 0; i < device_runners_.size(); ++i)
            {
                IInferenceRunner *child = device_runners_[i].get();
                if (!child ||
                    !child->supportsDeviceStochasticMTPVerification() ||
                    !child->buildStochasticDistributionOnDevice(
                        source,
                        row,
                        buffer,
                        slot,
                        params,
                        vocab_size))
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic target distribution build failed on participant "
                              << i);
                    return false;
                }
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_distribution_builds",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"source", "all_position"},
                 {"rows", "1"},
                 {"implementation", "mirrored_child_full_vocab"}});
            return true;
        }
        if (primaryDeviceId().is_gpu() &&
            source == DeviceLogitsSource::AllPosition)
        {
            LOG_ERROR("[RankOrchestrator] GPU LocalTP stochastic verifier target distributions require mirrored full-head child reducers");
            return false;
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
        if (source == DeviceLogitsSource::AllPosition &&
            usesMirroredMTPHeadForVerifier())
        {
            for (size_t i = 0; i < device_runners_.size(); ++i)
            {
                IInferenceRunner *child = device_runners_[i].get();
                if (!child ||
                    !child->supportsDeviceStochasticMTPVerification() ||
                    !child->buildStochasticDistributionsOnDevice(
                        source,
                        first_row,
                        buffer,
                        first_slot,
                        row_count,
                        params,
                        vocab_size))
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic target-row distribution build failed on participant "
                              << i);
                    return false;
                }
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_distribution_builds",
                static_cast<double>(row_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"source", "all_position"},
                 {"rows", std::to_string(row_count)},
                 {"implementation", "mirrored_child_full_vocab"}});
            return true;
        }
        if (primaryDeviceId().is_gpu() &&
            source == DeviceLogitsSource::AllPosition)
        {
            LOG_ERROR("[RankOrchestrator] GPU LocalTP stochastic verifier target-row distributions require mirrored full-head child reducers");
            return false;
        }
        return buildRankStochasticTargetDistributionsFromLocalTPRows(
            source,
            first_row,
            first_slot,
            row_count,
            params,
            vocab_size);
    }

    bool RankOrchestrator::
        buildCapturedStochasticVerifierTargetDistributions(
            int row_count,
            const SamplingParams &params,
            const MTPRequestPenaltyPolicy &penalty_policy,
            int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar
                ->buildCapturedStochasticVerifierTargetDistributions(
                    row_count,
                    params,
                    penalty_policy,
                    vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->buildCapturedStochasticVerifierTargetDistributions(
                    row_count,
                    params,
                    penalty_policy,
                    vocab_size);
        }
        if (!primaryDeviceId().is_gpu() ||
            !usesMirroredMTPHeadForVerifier())
        {
            LOG_ERROR("[RankOrchestrator] Captured stochastic verifier target distributions require mirrored GPU verifier heads");
            return false;
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child ||
                !child->buildCapturedStochasticVerifierTargetDistributions(
                    row_count,
                    params,
                    penalty_policy,
                    vocab_size))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored participant "
                          << participant
                          << " failed captured stochastic verifier target preparation");
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_captured_stochastic_verifier_target_rows",
            static_cast<double>(row_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"top_k", std::to_string(params.top_k)},
             {"implementation", "mirrored_monolithic_graph"}});
        return true;
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

    bool RankOrchestrator::publishCapturedMTPDraftToken(
        int row,
        int slot,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->publishCapturedMTPDraftToken(
                row,
                slot,
                penalty_policy);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->publishCapturedMTPDraftToken(
                row,
                slot,
                penalty_policy);
        }
        if (!primaryDeviceId().is_gpu() ||
            !usesMirroredMTPHeadForVerifier())
        {
            LOG_ERROR("[RankOrchestrator] Captured MTP draft publication requires mirrored full-vocabulary GPU MTP heads");
            return false;
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child ||
                !child->publishCapturedMTPDraftToken(
                    row,
                    slot,
                    penalty_policy))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored participant "
                          << participant
                          << " failed captured MTP draft publication");
                return false;
            }
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_captured_mtp_draft_publications",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"row", std::to_string(row)},
             {"slot", std::to_string(slot)},
             {"implementation", "mirrored_monolithic_graph"}});
        return true;
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

        if (source != DeviceLogitsSource::MTP ||
            row < 0 ||
            slot < 0 ||
            slot >= rank_stochastic_slot_capacity_ ||
            device_runners_.size() < 2)
        {
            return -1;
        }

        if (usesMirroredMTPHeadForVerifier())
        {
            /*
             * This host-returning API is a response/diagnostic boundary, not
             * the production deferred lane. Child zero materializes its local
             * token for the caller while every peer performs the same sampling
             * operation without a host read. Each slot remains produced by its
             * own sampler stream; no H2D restaging or rank collective follows.
             */
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                IInferenceRunner *runner = device_runners_[child].get();
                if (!runner ||
                    !runner->usesMirroredMTPHeadForVerifier() ||
                    !runner->supportsDeviceStochasticMTPVerification())
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic draft proposal requires every child to expose a mirrored full-head device sampler; participant "
                              << child << " is not ready");
                    return -1;
                }
            }

            int sampled_token = -1;
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                const bool ok =
                    child == 0
                        ? ((sampled_token = device_runners_[child]
                                                ->sampleStochasticDraftProposalOnDevice(
                                                    source,
                                                    row,
                                                    slot,
                                                    params,
                                                    vocab_size,
                                                    threshold)) >= 0)
                        : device_runners_[child]
                              ->sampleStochasticDraftProposalOnDeviceDeferred(
                                  source,
                                  row,
                                  slot,
                                  params,
                                  vocab_size,
                                  threshold);
                if (!ok)
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic draft proposal participant "
                              << child
                              << " failed local device sampling for slot="
                              << slot);
                    return -1;
                }
            }

            rank_stochastic_draft_sample_tokens_[static_cast<size_t>(slot)] =
                static_cast<int32_t>(sampled_token);
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_draft_proposals",
                1.0,
                "decode",
                 "rank",
                 {{"participants", std::to_string(device_runners_.size())},
                  {"slot", std::to_string(slot)},
                  {"implementation", "participant_local_identical_samples"},
                  {"collective", "none"},
                  {"host_reads", "primary_response_only"}});
            return sampled_token;
        }

        (void)params;
        (void)vocab_size;
        (void)threshold;

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

        if (source == DeviceLogitsSource::MTP &&
            usesMirroredMTPHeadForVerifier())
        {
            if (row < 0 ||
                slot < 0 ||
                slot >= rank_stochastic_slot_capacity_ ||
                device_runners_.size() < 2)
            {
                return false;
            }
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                IInferenceRunner *runner = device_runners_[child].get();
                if (!runner ||
                    !runner->usesMirroredMTPHeadForVerifier() ||
                    !runner->supportsDeviceStochasticMTPVerification())
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP deferred stochastic draft proposal requires every child to expose a mirrored full-head device sampler; participant "
                              << child << " is not ready");
                    return false;
                }
            }

            /*
             * Each child consumes its exact pending MTP-sidecar stream and
             * records its own sample-ready event. The argmax kernel is
             * deterministic over byte-identical mirrored logits, so this
             * locally complete publication is the production invariant. A
             * host comparison or corrective token broadcast would merely hide
             * a grouped-decode parity defect and is intentionally absent.
             */
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                if (!device_runners_[child]
                         ->sampleStochasticDraftProposalOnDeviceDeferred(
                             source,
                             row,
                             slot,
                             params,
                             vocab_size,
                             threshold))
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP deferred stochastic draft proposal participant "
                              << child
                              << " failed local device proposal sampling for slot="
                              << slot);
                    return false;
                }
            }

            rank_stochastic_draft_sample_tokens_[static_cast<size_t>(slot)] = -1;
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_deferred_draft_proposals",
                1.0,
                "decode",
                 "rank",
                 {{"participants", std::to_string(device_runners_.size())},
                  {"slot", std::to_string(slot)},
                  {"implementation", "participant_local_identical_samples"},
                  {"collective", "none"}});
            return true;
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

        if (buffer == DeviceDistributionBuffer::Target &&
            slot >= 0 &&
            slot < rank_stochastic_slot_capacity_ &&
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)])
        {
            const int token = device_runners_.front()
                                  ->sampleStochasticDistributionOnDevice(
                                      buffer,
                                      slot,
                                      threshold);
            if (token < 0 ||
                !broadcastPrimaryLocalTPMainTargetSlotToChildren(slot))
            {
                return -1;
            }
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)] = false;
            rank_stochastic_target_sample_tokens_[
                static_cast<size_t>(slot)] = static_cast<int32_t>(token);
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_main_target_samples",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(slot)},
                 {"host_shadow", "true"},
                 {"implementation", "primary_child_device_slot_broadcast"}});
            return token;
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
        if (buffer == DeviceDistributionBuffer::Target &&
            slot >= 0 &&
            slot < rank_stochastic_slot_capacity_ &&
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)])
        {
            if (!device_runners_.front()
                     ->sampleStochasticDistributionOnDeviceDeferred(
                         buffer,
                         slot,
                         threshold) ||
                !broadcastPrimaryLocalTPMainTargetSlotToChildren(slot))
            {
                return false;
            }
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)] = false;
            rank_stochastic_target_sample_tokens_[
                static_cast<size_t>(slot)] = -1;
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_main_target_samples",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(slot)},
                 {"host_shadow", "false"},
                 {"implementation", "primary_child_device_slot_broadcast"}});
            return true;
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

    bool RankOrchestrator::publishDeviceResidentConditionTokenToTargetSampleSlot(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index,
        int target_sample_slot)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar
                ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                    logical_state,
                    request_index,
                    target_sample_slot);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                    logical_state,
                    request_index,
                    target_sample_slot);
        }
        if (device_runners_.size() < 2 ||
            !logical_state.coversRequest(request_index) ||
            logical_state.target_positions_device !=
                rank_resident_logical_state_marker_.data() ||
            logical_state.live_state_epoch != rank_resident_logical_state_epoch_ ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            LOG_ERROR("[RankOrchestrator] Resident condition-token target publication received a stale or foreign LocalTP mailbox");
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
                    LOG_WARN("[TPWorkerPool] resident condition-token target publication failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             request_index,
             target_sample_slot,
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

                if (!device_runners_[i])
                    return false;
                const DeviceId device_id =
                    device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return device_runners_[i]
                    ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                        rank_resident_child_logical_state_handles_[i],
                        request_index,
                        target_sample_slot);
            });

        bool all_success = true;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        auto results =
            tp_worker_pool_->collectAll(effectiveTPWorkerJoinTimeoutMs());
        for (auto &result : results)
        {
            if (!result.completed || !result.success)
                all_success = false;
            if (result.exception && !first_exception)
            {
                first_exception = result.exception;
                first_exception_device = result.worker_index;
                all_success = false;
            }
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Resident condition-token target publication rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_resident_condition_token_target_publications",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_index", std::to_string(request_index)},
                 {"target_slot", std::to_string(target_sample_slot)},
                 {"transfer", "d2d"}});
        }
        return all_success;
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
            first_draft_slot + draft_token_count > rank_stochastic_slot_capacity_)
        {
            return false;
        }

        if (first_draft_slot == 0)
        {
            std::fill(
                rank_stochastic_draft_sample_tokens_.begin(),
                rank_stochastic_draft_sample_tokens_.end(),
                -1);
        }
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

        if (usesMirroredMTPHeadForVerifier())
        {
            return prepareRankVerifierTokenSlotsForLocalTP(
                /*first_token_from_device=*/false,
                first_token,
                /*first_target_sample_slot=*/-1,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }

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

    const void *RankOrchestrator::prepareMTPVerifierInputTokenBatchOnDevice(
        const DeviceMTPVerifierInputBatchRequest *requests,
        int request_count,
        int logical_padded_seq_len)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->prepareMTPVerifierInputTokenBatchOnDevice(
                requests,
                request_count,
                logical_padded_seq_len);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->prepareMTPVerifierInputTokenBatchOnDevice(
                requests,
                request_count,
                logical_padded_seq_len);
        }
        if (device_runners_.size() < 2 ||
            !requests ||
            request_count <= 0 ||
            logical_padded_seq_len <= 0)
        {
            return nullptr;
        }

        /*
         * Verifier conditions use participant-local canonical target slots.
         * Initial sampling and accepted-state publication both refresh those
         * slots, so rank orchestration forwards immutable slot indices and
         * never translates aggregate logical-mailbox pointers into child state.
         */
        for (int row_index = 0; row_index < request_count; ++row_index)
        {
            const DeviceMTPVerifierInputBatchRequest &row =
                requests[row_index];
            if (!row.first_token_from_device ||
                row.first_target_sample_slot < 0)
            {
                LOG_ERROR("[RankOrchestrator] Request-batched GPU verifier token row "
                          << row_index
                          << " does not name its canonical device target slot");
                return nullptr;
            }
        }

        rank_mtp_verifier_child_token_inputs_.assign(
            device_runners_.size(),
            nullptr);
        rank_mtp_verifier_child_token_count_ = logical_padded_seq_len;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                return nullptr;

            rank_mtp_verifier_child_token_inputs_[i] =
                device_runners_[i]->prepareMTPVerifierInputTokenBatchOnDevice(
                    requests,
                    request_count,
                    logical_padded_seq_len);
            if (!rank_mtp_verifier_child_token_inputs_[i])
            {
                rank_mtp_verifier_child_token_inputs_.clear();
                rank_mtp_verifier_child_token_count_ = 0;
                return nullptr;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_token_batches_prepared_from_device_slots",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(request_count)},
             {"logical_padded_seq_len",
              std::to_string(logical_padded_seq_len)}});
        return rank_mtp_verifier_child_token_inputs_.data();
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

        if (usesMirroredMTPHeadForVerifier())
        {
            return prepareRankVerifierTokenSlotsForLocalTP(
                /*first_token_from_device=*/true,
                /*first_token=*/-1,
                first_target_sample_slot,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }

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
        return copyDeviceSpeculativeOutcomesToHostForDiagnostics(handle, out);
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
        return copyDeviceSpeculativeOutcomesToHostForDiagnostics(handle, out);
    }

    DeviceGenerationExecutionPolicy
    RankOrchestrator::deviceGenerationExecutionPolicy(
        DeviceGenerationLoopTopology topology) const noexcept
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
            return DeviceGenerationExecutionPolicy::Unsupported;

        DeviceGenerationExecutionPolicy rank_policy =
            DeviceGenerationExecutionPolicy::NativeConditionalGraph;
        for (const auto &participant : participants)
        {
            if (!participant)
                return DeviceGenerationExecutionPolicy::Unsupported;

            const DeviceGenerationExecutionPolicy participant_policy =
                participant->deviceGenerationExecutionPolicy(topology);
            if (participant_policy ==
                DeviceGenerationExecutionPolicy::Unsupported)
            {
                return DeviceGenerationExecutionPolicy::Unsupported;
            }
            if (participant_policy ==
                DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions)
            {
                rank_policy = DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions;
            }
        }
        return rank_policy;
    }

    bool RankOrchestrator::beginDeviceResidentGeneration(
        const DeviceGenerationAdmissionRequest &request)
    {
        if (!request.valid())
        {
            LOG_ERROR("[RankOrchestrator] Invalid device-resident generation admission: requests="
                      << request.request_count << " max_new_tokens="
                      << request.max_new_tokens << " leading_row_disposition="
                      << static_cast<int>(
                             request.initial_leading_row_disposition));
            return false;
        }
        materialized_device_generation_execution_policy_.reset();
        materialized_device_generation_loop_topology_.reset();
        rank_hosted_device_generation_tickets_.clear();
        admitted_device_generation_max_new_tokens_ = 0;

        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Device-resident generation admission has no participants");
            return false;
        }

        /*
         * Every participant that may publish mirrored KV, recurrent, or
         * response state owns an independent resident ledger.  Admitting only
         * the final sampler would leave earlier pipeline/TP participants with
         * an unbounded publication authority and make cross-device equality an
         * accident of host scheduling.
         */
        for (size_t participant = 0; participant < participants.size(); ++participant)
        {
            if (!participants[participant] ||
                !participants[participant]
                     ->beginDeviceResidentGeneration(
                         request))
            {
                LOG_ERROR("[RankOrchestrator] Device-resident generation admission failed on participant "
                          << participant);
                return false;
            }
        }
        admitted_device_generation_max_new_tokens_ = request.max_new_tokens;
        return true;
    }

    bool RankOrchestrator::materializeDeviceResidentGeneration(
        int request_count,
        int draft_depth,
        DeviceGenerationLoopTopology topology,
        DeviceGenerationSamplingMode sampling_mode)
    {
        if (request_count <= 0 || draft_depth <= 0 ||
            !isValidDeviceGenerationSamplingMode(sampling_mode))
        {
            LOG_ERROR("[RankOrchestrator] Invalid device-generation parent preparation geometry"
                      << " requests=" << request_count
                      << " draft_depth=" << draft_depth
                      << " sampling_mode="
                      << deviceGenerationSamplingModeName(sampling_mode));
            return false;
        }

        const bool use_pp_participants = !pp_stage_runners_.empty();
        auto &participants =
            use_pp_participants ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Device-generation parent preparation has no participants");
            return false;
        }
        const DeviceGenerationExecutionPolicy execution_policy =
            deviceGenerationExecutionPolicy(topology);
        if (execution_policy ==
            DeviceGenerationExecutionPolicy::Unsupported)
        {
            LOG_ERROR("[RankOrchestrator] Device-generation materialization has no complete rank execution policy");
            return false;
        }
        if (participants.size() == 1)
        {
            const bool materialized =
                participants.front() &&
                participants.front()->materializeDeviceResidentGeneration(
                    request_count,
                    draft_depth,
                    topology,
                    sampling_mode);
            if (materialized)
            {
                materialized_device_generation_execution_policy_ =
                    execution_policy;
                materialized_device_generation_loop_topology_ = topology;
                if (execution_policy ==
                    DeviceGenerationExecutionPolicy::
                        HostScheduledCapturedTransactions)
                {
                    rank_hosted_device_generation_tickets_.resize(1);
                }
            }
            return materialized;
        }

        /*
         * Parent composition calls backend graph-cloning/instantiation APIs and
         * therefore belongs to each participant's persistent worker context.
         * Complete every composition before launchDeviceResident... submits any
         * graph: a rejected child on participant N must never be discovered
         * after participant zero has entered a collective-bearing WHILE body.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                std::make_unique<TPWorkerPool>(participants.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback(
                    [this]()
                    {
                        LOG_WARN("[TPWorkerPool] Device-generation parent preparation failure detected - aborting collective backend");
                        tp_ctx_->requestAbort();
                    });
            }
        }
        if (tp_worker_pool_->numWorkers() != participants.size())
        {
            LOG_ERROR("[RankOrchestrator] Device-generation parent preparation participant count does not match the persistent TP worker pool");
            return false;
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this, use_pp_participants, request_count, draft_depth, topology,
             sampling_mode,
             kernel_phase, rocm_phase, cuda_phase, kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto &worker_participants =
                    use_pp_participants
                        ? pp_stage_runners_
                        : device_runners_;
                if (i >= worker_participants.size() ||
                    !worker_participants[i])
                {
                    return false;
                }

                const DeviceId device =
                    worker_participants[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                return worker_participants[i]
                    ->materializeDeviceResidentGeneration(
                        request_count,
                        draft_depth,
                        topology,
                        sampling_mode);
            });

        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (auto &result : results)
        {
            if (!result.completed)
            {
                worker_timeout = true;
                all_success = false;
            }
            if (!result.success)
                all_success = false;
            if (result.exception && !first_exception)
            {
                first_exception = result.exception;
                first_exception_device = result.worker_index;
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "materializeDeviceResidentGeneration",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Device-generation parent preparation re-throwing primary exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            materialized_device_generation_execution_policy_ =
                execution_policy;
            materialized_device_generation_loop_topology_ = topology;
            if (execution_policy ==
                DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions)
            {
                rank_hosted_device_generation_tickets_.resize(
                    participants.size());
            }
        }
        return all_success;
    }

    bool RankOrchestrator::observeDeviceGenerationDispatchTicket(
        sampling_math::DeviceGenerationDispatchTicket *out_ticket)
    {
        if (out_ticket)
            *out_ticket = sampling_math::DeviceGenerationDispatchTicket{};
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (!out_ticket ||
            !materialized_device_generation_execution_policy_ ||
            *materialized_device_generation_execution_policy_ !=
                DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions ||
            participants.empty() ||
            rank_hosted_device_generation_tickets_.size() !=
                participants.size())
        {
            LOG_ERROR("[RankOrchestrator] Hosted device-generation ticket observation has no exact materialized rank policy");
            return false;
        }

        for (size_t participant_index = 0;
             participant_index < participants.size();
             ++participant_index)
        {
            if (!participants[participant_index] ||
                !participants[participant_index]
                     ->observeDeviceGenerationDispatchTicket(
                         &rank_hosted_device_generation_tickets_[
                             participant_index]))
            {
                LOG_ERROR("[RankOrchestrator] Hosted device-generation ticket observation failed on participant "
                          << participant_index);
                return false;
            }
        }

        const auto &authoritative =
            rank_hosted_device_generation_tickets_.front();
        for (size_t participant_index = 1;
             participant_index <
                 rank_hosted_device_generation_tickets_.size();
             ++participant_index)
        {
            if (!authoritative.hasSameDispatchDecision(
                    rank_hosted_device_generation_tickets_[
                        participant_index]))
            {
                const auto &divergent =
                    rank_hosted_device_generation_tickets_[
                        participant_index];
                LOG_ERROR("[RankOrchestrator] Device-generation controllers selected divergent hosted graph branches"
                          << " participant=" << participant_index
                          << " authoritative_transaction="
                          << authoritative.transaction_count
                          << " divergent_transaction="
                          << divergent.transaction_count
                          << " authoritative_depth="
                          << authoritative.next_draft_depth
                          << " divergent_depth="
                          << divergent.next_draft_depth
                          << " authoritative_complete="
                          << authoritative.complete
                          << " divergent_complete=" << divergent.complete
                          << " authoritative_maintenance="
                          << authoritative.maintenance_due
                          << " divergent_maintenance="
                          << divergent.maintenance_due);
                std::terminate();
            }
        }

        *out_ticket = authoritative;
        return true;
    }

    bool RankOrchestrator::submitHostScheduledDeviceGenerationAdvance(
        const sampling_math::DeviceGenerationDispatchTicket &ticket)
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (!materialized_device_generation_execution_policy_ ||
            *materialized_device_generation_execution_policy_ !=
                DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions ||
            participants.empty() ||
            rank_hosted_device_generation_tickets_.size() !=
                participants.size() ||
            !ticket.hasSameDispatchDecision(
                rank_hosted_device_generation_tickets_.front()))
        {
            LOG_ERROR("[RankOrchestrator] Hosted device-generation submission rejected a stale rank decision");
            return false;
        }

        const bool use_pp_participants = !pp_stage_runners_.empty();
        auto dispatch_to_all =
            [&](const char *operation,
                const std::function<bool(IInferenceRunner *, size_t)> &fn)
                -> bool
        {
            if (participants.size() == 1)
                return fn(participants.front().get(), 0);
            if (!tp_worker_pool_ ||
                tp_worker_pool_->numWorkers() != participants.size())
            {
                LOG_ERROR("[RankOrchestrator] " << operation
                          << " has no matching persistent TP worker pool");
                return false;
            }

            const auto kernel_phase = KernelProfiler::getCurrentPhase();
            const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            const auto kv_phase = KVCacheProfiler::getCurrentPhase();
            const auto executor_phase = GraphExecutorStats::currentPhase();
            tp_worker_pool_->dispatch(
                [this, use_pp_participants, fn, kernel_phase, rocm_phase,
                 cuda_phase, kv_phase, executor_phase](size_t index) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);
                    auto &worker_participants =
                        use_pp_participants ? pp_stage_runners_
                                            : device_runners_;
                    if (index >= worker_participants.size() ||
                        !worker_participants[index])
                    {
                        return false;
                    }
                    const DeviceId device =
                        worker_participants[index]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                    return fn(worker_participants[index].get(), index);
                });

            const int timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            auto results = tp_worker_pool_->collectAll(timeout_ms);
            for (const auto &result : results)
            {
                if (!result.completed || !result.success || result.exception)
                {
                    if (!result.completed && timeout_ms > 0)
                    {
                        abortAfterTPWorkerTimeout(
                            operation,
                            timeout_ms,
                            tp_worker_pool_->completedCount(),
                            tp_worker_pool_->numWorkers());
                    }
                    return false;
                }
            }
            return true;
        };

        /*
         * Submit the complete authenticated transaction on every persistent
         * participant worker in one rank-level fan-out. Each child walks the
         * same immutable producer-ordered branch and its sparse graph scopes
         * provide the only rendezvous required between participants. Keeping
         * the worker alive for the whole transaction removes one host
         * dispatch/collection barrier per semantic fragment without changing
         * device stream order or allowing either the rank or host to inspect
         * mutable generation state.
         */
        if (!dispatch_to_all(
                "submitHostScheduledDeviceGenerationAdvance",
                [this](IInferenceRunner *participant,
                       size_t participant_index) -> bool
                {
                    return participant &&
                        participant
                            ->submitHostScheduledDeviceGenerationAdvance(
                                rank_hosted_device_generation_tickets_[
                                    participant_index]);
                }))
        {
            LOG_ERROR("[RankOrchestrator] Hosted transaction-bundle submission failed after distributed submission began");
            std::terminate();
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_hosted_transaction_bundle_submissions",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(participants.size())},
             {"worker_fanouts", "1"},
             {"fragment_barriers", "0"}});
        return true;
    }

    bool RankOrchestrator::launchDeviceResidentGeneration()
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Device-generation parent launch has no participants");
            return false;
        }
        if (!materialized_device_generation_execution_policy_ ||
            !materialized_device_generation_loop_topology_)
        {
            LOG_ERROR("[RankOrchestrator] Device-generation launch has no exact materialized policy/topology");
            return false;
        }

        if (*materialized_device_generation_execution_policy_ ==
            DeviceGenerationExecutionPolicy::
                HostScheduledCapturedTransactions)
        {
            if (admitted_device_generation_max_new_tokens_ <= 0)
            {
                LOG_ERROR("[RankOrchestrator] Hosted device-generation loop has no admitted transaction bound");
                return false;
            }
            for (int observation = 0;
                 observation < admitted_device_generation_max_new_tokens_;
                 ++observation)
            {
                sampling_math::DeviceGenerationDispatchTicket ticket;
                if (!observeDeviceGenerationDispatchTicket(&ticket) ||
                    !submitHostScheduledDeviceGenerationAdvance(ticket))
                {
                    LOG_ERROR("[RankOrchestrator] Hosted device-generation transaction loop failed");
                    return false;
                }
                if (ticket.complete != 0)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "rank_device_generation_parent_launches",
                        1.0,
                        "decode",
                        "rank",
                        {{"participants",
                          std::to_string(participants.size())},
                         {"launch_order",
                          "all_participants_before_ticket_wait"},
                         {"execution",
                          "hosted_ticket_selected_captured_transactions"}});
                    return true;
                }
            }
            LOG_ERROR("[RankOrchestrator] Hosted device-generation loop exceeded its admitted response bound without completing");
            return false;
        }
        if (*materialized_device_generation_execution_policy_ !=
            DeviceGenerationExecutionPolicy::NativeConditionalGraph)
        {
            LOG_ERROR("[RankOrchestrator] Device-generation launch encountered an invalid materialized policy");
            return false;
        }

        /*
         * Enqueue every participant before waiting for any terminal result.
         * Captured NCCL/RCCL nodes may rendezvous across these graphs, so
         * synchronizing participant zero before participant one is launched
         * would deadlock the domain. A failure after any earlier launch leaves
         * a partially submitted distributed transaction and is therefore
         * process-fatal rather than recoverable.
         */
        for (size_t participant_index = 0;
             participant_index < participants.size();
             ++participant_index)
        {
            if (!participants[participant_index] ||
                !participants[participant_index]
                     ->launchDeviceResidentGeneration())
            {
                LOG_ERROR("[RankOrchestrator] Device-generation parent launch failed on participant "
                          << participant_index);
                if (participant_index != 0)
                    std::terminate();
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_device_generation_parent_launches",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(participants.size())},
             {"launch_order", "all_participants_before_terminal_wait"}});
        return true;
    }

    bool RankOrchestrator::finishDeviceResidentGeneration(
        DeviceGenerationTerminalResult *out_result)
    {
        if (out_result)
            *out_result = DeviceGenerationTerminalResult{};
        if (!out_result)
        {
            LOG_ERROR("[RankOrchestrator] Terminal device-generation result requires a destination");
            return false;
        }

        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Terminal device-generation result has no participants");
            return false;
        }

        std::vector<DeviceGenerationTerminalResult> participant_results;
        participant_results.reserve(participants.size());
        for (size_t participant_index = 0;
             participant_index < participants.size();
             ++participant_index)
        {
            DeviceGenerationTerminalResult participant_result;
            if (!participants[participant_index] ||
                !participants[participant_index]
                     ->finishDeviceResidentGeneration(
                         &participant_result) ||
                !participant_result.valid())
            {
                LOG_ERROR("[RankOrchestrator] Terminal device-generation materialization failed on participant "
                          << participant_index);
                return false;
            }
            participant_results.push_back(
                std::move(participant_result));
        }

        const auto &authoritative_requests =
            participant_results.front().requests;
        for (size_t participant_index = 1;
             participant_index < participant_results.size();
             ++participant_index)
        {
            if (participant_results[participant_index].requests !=
                authoritative_requests)
            {
                std::vector<std::vector<MTPMirroredTensorDigest>>
                    participant_digests;
                participant_digests.reserve(participants.size());
                for (const auto &participant : participants)
                {
                    auto *device_orchestrator =
                        dynamic_cast<DeviceGraphOrchestrator *>(
                            participant.get());
                    participant_digests.push_back(
                        device_orchestrator
                            ? device_orchestrator
                                  ->captureMirroredMTPDigests()
                            : std::vector<MTPMirroredTensorDigest>{});
                }
                LOG_ERROR("[RankOrchestrator] Mirrored terminal device-generation ledgers disagree"
                          << " primary="
                          << participant_results.front().device.toString()
                          << " participant="
                          << participant_results[participant_index]
                                 .device.toString()
                          << " participant_index=" << participant_index
                          << describeTerminalLedgerMismatch(
                                 authoritative_requests,
                                 participant_results[participant_index]
                                     .requests)
                          << describeMirroredDigestMismatch(
                                 participant_digests));
                return false;
            }
        }

        *out_result = std::move(participant_results.front());
        materialized_device_generation_execution_policy_.reset();
        materialized_device_generation_loop_topology_.reset();
        rank_hosted_device_generation_tickets_.clear();
        admitted_device_generation_max_new_tokens_ = 0;
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_device_generation_terminal_response_bridges",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(participants.size())},
             {"validation", "byte_identical_host_terminal_results"}});
        return true;
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
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::None;
        rank_mirrored_child_outcomes_.clear();
        rank_mirrored_primary_outcome_ = DeviceSpeculativeOutcomeHandle{};
        rank_mirrored_child_outcomes_valid_ = false;
        rank_resident_child_logical_state_handles_.clear();

        if (usesMirroredMTPHeadForVerifier())
        {
            return verifyStochasticMirroredLocalTPRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                out_handle);
        }
        if (primaryDeviceId().is_gpu())
        {
            LOG_ERROR("[RankOrchestrator] GPU LocalTP MTP resident stochastic verification requires mirrored full-head child verifier outcomes");
            return false;
        }

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
            request.row_count > rank_max_draft_depth_ ||
            request.first_target_slot < 0 ||
            request.first_draft_slot < 0 ||
            request.first_target_slot + request.row_count >
                rank_stochastic_slot_capacity_ ||
            request.first_draft_slot + request.row_count >
                rank_stochastic_slot_capacity_ ||
            request.stop_token_count < 0 ||
            request.stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (request.bonus_target_slot >= rank_stochastic_slot_capacity_) ||
            (!request.use_device_draft_tokens &&
             request.draft_tokens.size() <
                 static_cast<size_t>(request.row_count)) ||
            (!request.derive_thresholds_from_seed &&
             (request.accept_thresholds.size() <
                  static_cast<size_t>(request.row_count) ||
              request.residual_thresholds.size() <
                  static_cast<size_t>(request.row_count))))
        {
            return false;
        }

        const bool derive_thresholds =
            request.derive_thresholds_from_seed &&
            request.draw_position_source ==
                DeviceStochasticDrawPositionSource::HostLogicalPosition &&
            request.use_vllm_probability_rejection &&
            request.inverse_sample_seed != 0 &&
            request.inverse_sample_first_logical_position >= 0;
        if (request.derive_thresholds_from_seed != derive_thresholds)
        {
            LOG_ERROR("[RankOrchestrator] CPU rank stochastic verifier cannot consume a device-owned draw-position descriptor");
            return false;
        }
        std::vector<float> accept_thresholds(
            static_cast<size_t>(request.row_count),
            0.0f);
        std::vector<float> residual_thresholds(
            static_cast<size_t>(request.row_count),
            0.0f);
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

        std::vector<int> row_tokens(
            static_cast<size_t>(request.row_count),
            -1);
        std::vector<int> row_accepted(
            static_cast<size_t>(request.row_count),
            0);

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

        std::vector<int> output_tokens_int(
            static_cast<size_t>(rank_compact_output_token_stride_),
            -1);
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens{};
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
            static_cast<int>(output_tokens_int.size()),
            meta.data());
        if (meta[kSpecBatchMetaOk] == 0)
            return false;

        std::fill(
            rank_compact_output_tokens_.begin(),
            rank_compact_output_tokens_.end(),
            -1);
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
        out_handle->logical_verifier_rows_per_request = request.row_count + 1;
        out_handle->physical_verifier_rows_per_request = request.row_count + 1;
        out_handle->output_token_stride = rank_compact_output_token_stride_;
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
         * their resident publication capability identical to the child.
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
        if (!usesMirroredMTPHeadForVerifier())
        {
            return false;
        }

        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->primaryDeviceId().is_gpu() &&
                       runner->supportsDeviceResidentMTPSpecStatePublication();
            });
    }

    bool RankOrchestrator::validateCurrentMirroredLocalTPOutcome(
        const DeviceSpeculativeOutcomeHandle &outcome,
        const char *operation,
        std::string *error) const
    {
        const bool mirrored_kind =
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredGreedy ||
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredStochastic;
        const bool complete_participant_set =
            rank_mirrored_child_outcomes_.size() == device_runners_.size();
        const bool primary_valid = rank_mirrored_primary_outcome_.valid();
        const bool exact_primary_identity =
            outcome.output_tokens_device ==
                rank_mirrored_primary_outcome_.output_tokens_device &&
            outcome.meta_device == rank_mirrored_primary_outcome_.meta_device &&
            outcome.device == rank_mirrored_primary_outcome_.device &&
            outcome.request_count ==
                rank_mirrored_primary_outcome_.request_count &&
            outcome.logical_verifier_rows_per_request ==
                rank_mirrored_primary_outcome_
                    .logical_verifier_rows_per_request &&
            outcome.physical_verifier_rows_per_request ==
                rank_mirrored_primary_outcome_
                    .physical_verifier_rows_per_request &&
            outcome.output_token_stride ==
                rank_mirrored_primary_outcome_.output_token_stride &&
            outcome.meta_stride == rank_mirrored_primary_outcome_.meta_stride &&
            outcome.stream == rank_mirrored_primary_outcome_.stream &&
            outcome.response_ready_event.get() ==
                rank_mirrored_primary_outcome_.response_ready_event.get() &&
            outcome.mtp_transaction.state.get() ==
                rank_mirrored_primary_outcome_.mtp_transaction.state.get() &&
            outcome.device_generation_controller_owned ==
                rank_mirrored_primary_outcome_
                    .device_generation_controller_owned &&
            outcome.mirrored_local_tp_locally_complete ==
                rank_mirrored_primary_outcome_.mirrored_local_tp_locally_complete;

        if (mirrored_kind &&
            rank_compact_outcome_valid_ &&
            rank_mirrored_child_outcomes_valid_ &&
            complete_participant_set &&
            outcome.valid() &&
            primary_valid &&
            exact_primary_identity)
        {
            return true;
        }

        if (error)
        {
            std::ostringstream message;
            message << (operation && operation[0] != '\0'
                            ? operation
                            : "mirrored LocalTP outcome consumer")
                    << " rejected a non-current mirrored verifier outcome: "
                    << "kind=" << static_cast<int>(rank_compact_outcome_kind_)
                    << " mirrored_kind=" << mirrored_kind
                    << " rank_valid=" << rank_compact_outcome_valid_
                    << " child_set_valid=" << rank_mirrored_child_outcomes_valid_
                    << " child_handles=" << rank_mirrored_child_outcomes_.size()
                    << " participants=" << device_runners_.size()
                    << " candidate_valid=" << outcome.valid()
                    << " primary_valid=" << primary_valid
                    << " exact_primary_identity=" << exact_primary_identity
                    << " candidate_tokens="
                    << static_cast<const void *>(outcome.output_tokens_device)
                    << " primary_tokens="
                    << static_cast<const void *>(
                           rank_mirrored_primary_outcome_.output_tokens_device)
                    << " candidate_meta="
                    << static_cast<const void *>(outcome.meta_device)
                    << " primary_meta="
                    << static_cast<const void *>(
                           rank_mirrored_primary_outcome_.meta_device)
                    << " candidate_stream=" << outcome.stream
                    << " primary_stream=" << rank_mirrored_primary_outcome_.stream
                    << " candidate_ready=" << outcome.response_ready_event.get()
                    << " primary_ready="
                    << rank_mirrored_primary_outcome_.response_ready_event.get()
                    << " candidate_local_complete="
                    << outcome.mirrored_local_tp_locally_complete
                    << " primary_local_complete="
                    << rank_mirrored_primary_outcome_
                           .mirrored_local_tp_locally_complete;
            *error = message.str();
        }
        return false;
    }

    bool RankOrchestrator::copyDeviceSpeculativeOutcomesToHostForDiagnostics(
        const DeviceSpeculativeOutcomeHandle &handle,
        DeviceSpeculativeVerifyBatchOutcome *outcomes)
    {
        const bool mirrored_local_tp_domain =
            pp_stage_runners_.empty() &&
            device_runners_.size() > 1 &&
            usesMirroredMTPHeadForVerifier();
        if (mirrored_local_tp_domain)
        {
            std::string outcome_error;
            if (!outcomes ||
                !validateCurrentMirroredLocalTPOutcome(
                    handle,
                    "response materialization",
                    &outcome_error))
            {
                LOG_ERROR("[RankOrchestrator] "
                          << (outcomes
                                  ? outcome_error
                                  : "response materialization received a null outcome destination"));
                return false;
            }

            if (!device_runners_.front() ||
                !device_runners_.front()->copyDeviceSpeculativeOutcomesToHostForDiagnostics(
                    rank_mirrored_child_outcomes_.front(),
                    outcomes))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP response materialization failed in primary child after rank ownership validation: device="
                          << (device_runners_.front()
                                  ? device_runners_.front()->primaryDeviceId().toString()
                                  : "missing")
                          << " request_count=" << handle.request_count
                          << " output_stride=" << handle.output_token_stride
                          << " meta_stride=" << handle.meta_stride
                          << " local_complete="
                          << handle.mirrored_local_tp_locally_complete);
                return false;
            }

            PerfStatsCollector::addCounter(
                "mtp",
                rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredStochastic
                    ? "rank_mirrored_localtp_stochastic_outcome_host_materializations"
                    : "rank_mirrored_localtp_greedy_outcome_host_materializations",
                1.0,
                "decode",
                "rank",
                {{"requests", std::to_string(handle.request_count)},
                 {"source", "primary_child_device_outcome"}});
            return true;
        }

        if (!outcomes ||
            !rank_compact_outcome_valid_ ||
            !handle.valid() ||
            handle.output_tokens_device != rank_compact_output_tokens_.data() ||
            handle.meta_device != rank_compact_output_meta_.data() ||
            handle.request_count != 1 ||
            handle.output_token_stride != rank_compact_output_token_stride_ ||
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
                "GPU LocalTP MTP resident publication requires mirrored full-head child verifier outcomes on every participant");
        }
        if (!request.valid())
        {
            return fail(
                "mirrored LocalTP MTP publication received an invalid device outcome request");
        }
        if (rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredGreedy ||
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredStochastic)
        {
            return publishMirroredLocalTPDeviceResidentMTPSpecStateBatch(
                request,
                error);
        }
        return fail(
            "GPU LocalTP MTP publication refuses rank-owned compact metadata; run mirrored child verifier reduction first");
    }

    bool RankOrchestrator::adoptMirroredLocalTPResidentLogicalStateMailboxes(
        int expected_request_count,
        const char *lifecycle,
        std::string *error)
    {
        const std::string lifecycle_name =
            lifecycle && lifecycle[0] != '\0' ? lifecycle : "unspecified";
        auto invalidate_and_fail = [&](const std::string &reason) -> bool
        {
            invalidateRankResidentLogicalStateAggregate(
                lifecycle_name.c_str(),
                reason.c_str());
            if (error)
                *error = reason;
            return false;
        };

        if (expected_request_count <= 0 || device_runners_.size() < 2)
        {
            return invalidate_and_fail(
                "resident mailbox adoption requires a non-empty multi-device request batch");
        }

        std::vector<DeviceResidentLogicalSequenceStateHandle> next_handles;
        next_handles.reserve(device_runners_.size());
        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child)
            {
                return invalidate_and_fail(
                    "resident mailbox adoption lost participant " +
                    std::to_string(participant));
            }

            DeviceResidentLogicalSequenceStateHandle child_handle =
                child->deviceResidentLogicalSequenceState();
            if (!child_handle.valid())
            {
                return invalidate_and_fail(
                    "resident mailbox adoption found no live child mailbox on participant " +
                    std::to_string(participant));
            }
            if (child_handle.request_count != expected_request_count)
            {
                return invalidate_and_fail(
                    "resident mailbox adoption found request-count mismatch on participant " +
                    std::to_string(participant) + ": expected " +
                    std::to_string(expected_request_count) + ", got " +
                    std::to_string(child_handle.request_count));
            }
            if (child_handle.device != child->primaryDeviceId())
            {
                return invalidate_and_fail(
                    "resident mailbox adoption found cross-device ownership on participant " +
                    std::to_string(participant));
            }
            next_handles.push_back(child_handle);
        }

        rank_resident_child_logical_state_handles_.swap(next_handles);
        ++rank_resident_logical_state_epoch_;
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_resident_mailbox_adoptions",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"request_count", std::to_string(expected_request_count)},
             {"lifecycle", lifecycle_name},
             {"payload_owner", "child_device_mailboxes"}});
        return true;
    }

    bool RankOrchestrator::publishMirroredLocalTPDeviceResidentMTPSpecStateBatch(
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

        std::string outcome_error;
        if (!validateCurrentMirroredLocalTPOutcome(
                request.outcome,
                "mirrored LocalTP state publication",
                &outcome_error))
        {
            return fail(outcome_error);
        }
        if (!request.valid())
        {
            return fail("mirrored LocalTP MTP publication request has invalid capture geometry");
        }

        /*
         * Each child owns the verifier state, compact outcome, and exact stream
         * that produced both. Passing child-local handles preserves that
         * ownership while byte-identical mirrored inputs and deterministic
         * reducers provide the common accepted count and next-condition token.
         */
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            IInferenceRunner *child = device_runners_[i].get();
            if (!child)
            {
                return fail(
                    "mirrored LocalTP MTP resident publication lost a participant");
            }

            DeviceSpeculativePublicationRequest child_request = request;
            child_request.outcome = rank_mirrored_child_outcomes_[i];
            std::string publish_error;
            if (!child->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                    child_request,
                    &publish_error))
            {
                return fail(
                    "mirrored LocalTP MTP resident publication failed on participant " +
                    std::to_string(i) +
                    (publish_error.empty() ? std::string()
                                           : ": " + publish_error));
            }
        }

        std::string mailbox_error;
        if (!adoptMirroredLocalTPResidentLogicalStateMailboxes(
                request.requestCount(),
                "accepted_state_publication",
                &mailbox_error))
        {
            return fail(
                "mirrored LocalTP MTP resident publication could not adopt child logical-state mailboxes: " +
                mailbox_error);
        }

        PerfStatsCollector::addCounter(
            "mtp",
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredStochastic
                ? "rank_mirrored_localtp_stochastic_device_outcome_publications"
                : "rank_mirrored_localtp_greedy_device_outcome_publications",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"request_count", std::to_string(request.requestCount())},
             {"logical_verifier_rows",
              std::to_string(request.logicalVerifierRowsPerRequest())},
             {"physical_verifier_rows",
              std::to_string(request.physicalVerifierRowsPerRequest())},
             {"implementation", "mirrored_child_device_publish"}});
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
         * NodeTP, PP, and routed-expert implementations from drifting
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
                    LOG_ERROR("RankOrchestrator::getAllPositionLogits: mixed local and replicated verifier logits are invalid");
                    return nullptr;
                }

                LogitsLocalInfo info =
                    runner->consumeAllPositionLogitsLocalInfoForHostGather();
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
                all_position_logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    full_vocab,
                    rows,
                    logits_backend_resolver_);
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

    bool RankOrchestrator::applyDeviceOwnedMTPPenaltiesToLogitRows(
        DeviceLogitsSource source,
        int row_count,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyDeviceOwnedMTPPenaltiesToLogitRows(
                source,
                row_count,
                penalty_policy);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->applyDeviceOwnedMTPPenaltiesToLogitRows(
                    source,
                    row_count,
                    penalty_policy);
        }
        if (device_runners_.empty() || row_count <= 0)
            return false;

        if (source == DeviceLogitsSource::Main)
        {
            IInferenceRunner *primary = device_runners_[0].get();
            return primary &&
                   primary->applyDeviceOwnedMTPPenaltiesToLogitRows(
                       source,
                       row_count,
                       penalty_policy);
        }
        if (source != DeviceLogitsSource::AllPosition ||
            !usesMirroredMTPHeadForVerifier())
        {
            LOG_ERROR("[RankOrchestrator] Device-owned stochastic verifier penalties require mirrored full-vocabulary LocalTP heads");
            return false;
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child ||
                !child->applyDeviceOwnedMTPPenaltiesToLogitRows(
                    source,
                    row_count,
                    penalty_policy))
            {
                LOG_ERROR("[RankOrchestrator] Device-owned stochastic verifier penalty transform failed on participant "
                          << participant);
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::applyDeviceOwnedMTPBranchPenaltiesToLogits(
        int prior_draft_count,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyDeviceOwnedMTPBranchPenaltiesToLogits(
                prior_draft_count,
                penalty_policy);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->applyDeviceOwnedMTPBranchPenaltiesToLogits(
                    prior_draft_count,
                    penalty_policy);
        }
        if (device_runners_.empty() || prior_draft_count < 0 ||
            !usesMirroredMTPHeadForVerifier())
        {
            LOG_ERROR("[RankOrchestrator] Device-owned MTP branch penalties require mirrored full-vocabulary LocalTP heads");
            return false;
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child ||
                !child->applyDeviceOwnedMTPBranchPenaltiesToLogits(
                    prior_draft_count,
                    penalty_policy))
            {
                LOG_ERROR("[RankOrchestrator] Device-owned MTP branch penalty transform failed on participant "
                          << participant);
                return false;
            }
        }
        return true;
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

    bool RankOrchestrator::waitForLastInferenceCompletionForBenchmark()
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Benchmark completion boundary has no participants");
            return false;
        }

        /*
         * forwardTP() submits every participant before returning. Waiting here
         * is therefore safe for captured NCCL/RCCL graphs: no rank can be held
         * before its peer graph is launched. Each child waits on its own exact
         * terminal event, so this does not widen into a device synchronization.
         */
        for (size_t participant = 0; participant < participants.size(); ++participant)
        {
            if (!participants[participant] ||
                !participants[participant]
                     ->waitForLastInferenceCompletionForBenchmark())
            {
                LOG_ERROR("[RankOrchestrator] Benchmark terminal-event wait failed for participant "
                          << participant);
                return false;
            }
        }
        return true;
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
        if (moeOverlayAuthorityExecution() ==
            MoEOverlayAuthorityExecutionKind::
                DeviceResident)
        {
            LOG_DEBUG("RankOrchestrator: skipping host MoE runtime histogram bridge; "
                      "device-resident ExpertOverlay authority owns histogram allgather");
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
        return forwardHostTokenBatchAcrossDevices(
            token_batches,
            &IInferenceRunner::forward_batch,
            "forward_batch");
    }

    bool RankOrchestrator::forwardHostTokenBatchAcrossDevices(
        const std::vector<std::vector<int>> &token_batches,
        bool (IInferenceRunner::*entrypoint)(
            const std::vector<std::vector<int>> &),
        const char *operation)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::" << operation
                                            << ": No device runners available");
            return false;
        }
        if (!entrypoint || !operation)
        {
            LOG_ERROR(
                "RankOrchestrator::forwardHostTokenBatchAcrossDevices: missing "
                "typed child entrypoint or operation identity");
            return false;
        }

        LOG_DEBUG("RankOrchestrator::" << operation
                                        << ": batch_size=" << token_batches.size()
                                        << ", devices=" << device_runners_.size());

        // Launch parallel batch forward passes on all devices
        std::vector<std::future<bool>> futures;
        futures.reserve(device_runners_.size());

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            auto &runner = device_runners_[i];
            if (runner)
            {
                IInferenceRunner *const child = runner.get();
                futures.push_back(std::async(std::launch::async,
                                             [child, &token_batches, entrypoint]()
                                             {
                                                 return (child->*entrypoint)(token_batches);
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
                    LOG_ERROR("RankOrchestrator::" << operation
                                                    << ": Device " << i
                                                    << " forward failed");
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
                    LOG_ERROR("RankOrchestrator::" << operation
                                                    << ": Device " << i
                                                    << " threw PRIMARY exception: "
                                                    << error_msg);
                }
                else if (is_context_destroyed)
                {
                    LOG_WARN("RankOrchestrator::" << operation
                                                   << ": Device " << i
                                                   << " threw SECONDARY exception (context destroyed): "
                                                   << error_msg);
                }
                else
                {
                    LOG_ERROR("RankOrchestrator::" << operation
                                                    << ": Device " << i
                                                    << " threw exception: "
                                                    << error_msg);
                }
            }
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::" << operation
                                            << ": Re-throwing primary exception from device "
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
        std::fill(
            rank_stochastic_target_token_ids_.begin(),
            rank_stochastic_target_token_ids_.end(),
            -1);
        std::fill(
            rank_stochastic_target_probs_.begin(),
            rank_stochastic_target_probs_.end(),
            0.0f);
        std::fill(
            rank_stochastic_target_top_k_.begin(),
            rank_stochastic_target_top_k_.end(),
            0);
        std::fill(
            rank_stochastic_target_sample_tokens_.begin(),
            rank_stochastic_target_sample_tokens_.end(),
            -1);
        std::fill(
            rank_stochastic_draft_sample_tokens_.begin(),
            rank_stochastic_draft_sample_tokens_.end(),
            -1);
        std::fill(
            rank_mirrored_target_distribution_ready_.begin(),
            rank_mirrored_target_distribution_ready_.end(),
            false);
        rank_stochastic_staged_draft_tokens_.clear();
        std::fill(
            rank_compact_output_tokens_.begin(),
            rank_compact_output_tokens_.end(),
            -1);
        rank_compact_output_meta_.fill(0);
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::None;
        rank_compact_last_draft_tokens_.clear();
        rank_compact_last_stop_tokens_.clear();
        rank_mtp_verifier_child_token_inputs_.clear();
        rank_mtp_verifier_child_token_count_ = 0;
        rank_compact_outcome_valid_ = false;
        rank_mirrored_child_outcomes_.clear();
        rank_mirrored_primary_outcome_ = DeviceSpeculativeOutcomeHandle{};
        rank_mirrored_child_outcomes_valid_ = false;
        rank_hosted_device_moe_rebalance_tickets_.fill(
            DeviceMoERebalanceDispatchTicket{});
        rank_hosted_device_moe_rebalance_ticket_count_ = 0;
        rank_hosted_device_moe_rebalance_observation_schedule_ = {};
        rank_hosted_device_moe_rounds_until_observation_ = 0u;
        rank_hosted_device_moe_observation_schedule_initialized_ = false;
        invalidateRankResidentLogicalStateAggregate(
            "request_reset",
            request.reason ? request.reason : "request_boundary");
        stats_dirty_ = true;
    }

    void RankOrchestrator::clear_cache()
    {
        resetInferenceState(
            InferenceStateResetRequest::requestBoundary("clear_cache"));
    }

    bool RankOrchestrator::purgePrefixCache()
    {
        /* Every LocalTP/PP child owns a shard of the same logical archive. */
        for (auto &runner : device_runners_)
        {
            if (runner && !runner->purgePrefixCache())
                return false;
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (runner && !runner->purgePrefixCache())
                return false;
        }
        return true;
    }

    DeviceMoERebalanceMaintenanceExecutionPolicy
    RankOrchestrator::deviceMoERebalanceMaintenanceExecutionPolicy()
        const noexcept
    {
        /* Pipeline stages are independent maintenance domains. The outer rank
         * fans their complete boundary calls out concurrently and never tries
         * to compare participant IDs across unrelated LocalTP domains. */
        if (!pp_stage_runners_.empty())
            return DeviceMoERebalanceMaintenanceExecutionPolicy::Inactive;
        if (device_runners_.empty())
            return DeviceMoERebalanceMaintenanceExecutionPolicy::Inactive;

        DeviceMoERebalanceMaintenanceExecutionPolicy policy =
            device_runners_.front()
                ? device_runners_.front()
                      ->deviceMoERebalanceMaintenanceExecutionPolicy()
                : DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported;
        for (size_t index = 1; index < device_runners_.size(); ++index)
        {
            const auto participant_policy =
                device_runners_[index]
                    ? device_runners_[index]
                          ->deviceMoERebalanceMaintenanceExecutionPolicy()
                    : DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported;
            if (participant_policy != policy)
                return DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported;
        }
        return policy;
    }

    DeviceMoERebalanceHostedObservationSchedule
    RankOrchestrator::deviceMoERebalanceHostedObservationSchedule()
        const noexcept
    {
        if (!pp_stage_runners_.empty() || device_runners_.empty() ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            !device_runners_.front())
        {
            return {};
        }

        const auto schedule = device_runners_.front()
                                  ->deviceMoERebalanceHostedObservationSchedule();
        if (!schedule.valid())
            return {};
        for (size_t index = 1; index < device_runners_.size(); ++index)
        {
            if (!device_runners_[index] ||
                device_runners_[index]
                        ->deviceMoERebalanceHostedObservationSchedule() !=
                    schedule)
            {
                return {};
            }
        }
        return schedule;
    }

    std::vector<MoEOverlayDeviceControllerRuntimeBinding>
    RankOrchestrator::moeOverlayDeviceControllerRuntimeBindings() const
    {
        std::vector<MoEOverlayDeviceControllerRuntimeBinding> result;
        const auto append = [&result](const auto &runners)
        {
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;
                auto child =
                    runner->moeOverlayDeviceControllerRuntimeBindings();
                result.insert(
                    result.end(), child.begin(), child.end());
            }
        };
        append(device_runners_);
        append(pp_stage_runners_);
        std::sort(
            result.begin(),
            result.end(),
            [](const auto &left, const auto &right)
            {
                return left.overlay_participant_id <
                       right.overlay_participant_id;
            });
        for (std::size_t index = 0u; index < result.size(); ++index)
        {
            if (!result[index].valid() ||
                (index != 0u &&
                 result[index - 1u].overlay_participant_id ==
                     result[index].overlay_participant_id))
            {
                throw std::logic_error(
                    "RankOrchestrator collected invalid or duplicate ExpertOverlay controller runtime bindings");
            }
        }
        return result;
    }

    bool RankOrchestrator::installMoEOverlayTransferProgressEpoch(
        std::shared_ptr<MappedTransferProgressEpoch> epoch)
    {
        if (!epoch || !epoch->device().is_gpu())
        {
            LOG_ERROR(
                "RankOrchestrator received an invalid ExpertOverlay transfer-progress epoch");
            return false;
        }

        IInferenceRunner *owner = nullptr;
        const auto find_owner = [&](auto &runners) -> bool
        {
            for (auto &runner : runners)
            {
                if (!runner || runner->primaryDeviceId() != epoch->device())
                    continue;
                if (owner)
                {
                    LOG_ERROR(
                        "RankOrchestrator found duplicate device owners for ExpertOverlay transfer progress on "
                        << epoch->device().toString());
                    return false;
                }
                owner = runner.get();
            }
            return true;
        };
        if (!find_owner(device_runners_) || !find_owner(pp_stage_runners_) ||
            !owner)
        {
            LOG_ERROR(
                "RankOrchestrator could not resolve the exact child for ExpertOverlay transfer progress on "
                << epoch->device().toString());
            return false;
        }
        return owner->installMoEOverlayTransferProgressEpoch(std::move(epoch));
    }

    bool RankOrchestrator::
        submitHostScheduledDeviceMoERebalanceKnownNonDueBoundary()
    {
        if (!pp_stage_runners_.empty() || device_runners_.empty() ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            !deviceMoERebalanceHostedObservationSchedule().valid() ||
            rank_hosted_device_moe_rebalance_ticket_count_ != 0u)
        {
            LOG_ERROR("[RankOrchestrator] HIP known-non-due MoE boundary has no exact idle LocalTP schedule");
            return false;
        }

        PerfStatsCollector::ScopedTimer submission_timer(
            "moe_rebalance",
            "rank_device_moe_known_non_due_boundary_submission",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"dispatch", "serial_non_collective"}});

        /* Each participant's complete decode/MTP graph already owns the device
         * cadence edge. This loop validates the immutable embedded-topology
         * contract only; routing it through the persistent TP worker pool would
         * add a host wake/collect round-trip despite having no GPU or collective
         * submission. The due path below still uses concurrent workers because
         * its maintenance graph contains real RCCL collectives. */
        for (size_t index = 0; index < device_runners_.size(); ++index)
        {
            IInferenceRunner *const runner = device_runners_[index].get();
            if (!runner ||
                !runner
                     ->submitHostScheduledDeviceMoERebalanceKnownNonDueBoundary())
            {
                /* Earlier participants may already have advanced their device
                 * clocks. There is no recoverable continuation that can prove
                 * participant symmetry, so fail fatally at the first partial
                 * submission instead of entering the next inference round. */
                LOG_ERROR("[RankOrchestrator] HIP known-non-due MoE boundary failed after serial participant submission began"
                          << " participant=" << index);
                std::terminate();
            }
        }

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_device_moe_rebalance_ticket_observations_elided",
            1.0,
            "decode",
            "rank",
             {{"participants", std::to_string(device_runners_.size())},
             {"submission", "embedded_decode_graph_noop"}});
        return true;
    }

    bool RankOrchestrator::observeDeviceMoERebalanceDispatchTicket(
        DeviceMoERebalanceDispatchTicket *out_ticket)
    {
        if (out_ticket)
            *out_ticket = DeviceMoERebalanceDispatchTicket{};
        if (!out_ticket || !pp_stage_runners_.empty() ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            device_runners_.empty() ||
            device_runners_.size() >
                rank_hosted_device_moe_rebalance_tickets_.size() ||
            rank_hosted_device_moe_rebalance_ticket_count_ != 0u)
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE ticket observation has no exact, idle LocalTP policy");
            return false;
        }

        if (device_runners_.size() == 1u)
        {
            if (!device_runners_.front() ||
                !device_runners_.front()
                     ->observeDeviceMoERebalanceDispatchTicket(
                         &rank_hosted_device_moe_rebalance_tickets_.front()))
            {
                return false;
            }
        }
        else
        {
            if (!tp_worker_pool_)
            {
                tp_worker_pool_ =
                    std::make_unique<TPWorkerPool>(device_runners_.size());
                if (tp_ctx_)
                {
                    tp_worker_pool_->setFailureCallback(
                        [this]()
                        {
                            LOG_WARN("[TPWorkerPool] HIP Device MoE ticket observation failure detected - aborting collective backend");
                            tp_ctx_->requestAbort();
                        });
                }
            }
            if (tp_worker_pool_->numWorkers() != device_runners_.size())
            {
                LOG_ERROR("[RankOrchestrator] HIP Device MoE ticket participant count does not match the persistent worker pool");
                return false;
            }

            const auto kernel_phase = KernelProfiler::getCurrentPhase();
            const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            const auto kv_phase = KVCacheProfiler::getCurrentPhase();
            const auto executor_phase = GraphExecutorStats::currentPhase();
            tp_worker_pool_->dispatch(
                [this, kernel_phase, rocm_phase, cuda_phase, kv_phase,
                 executor_phase](size_t index) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);
                    if (index >= device_runners_.size() ||
                        !device_runners_[index])
                    {
                        return false;
                    }
                    const DeviceId device =
                        device_runners_[index]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                    return device_runners_[index]
                        ->observeDeviceMoERebalanceDispatchTicket(
                            &rank_hosted_device_moe_rebalance_tickets_[index]);
                });

            bool all_success = true;
            std::exception_ptr first_exception;
            size_t first_exception_device = 0;
            const int timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            bool worker_timeout = false;
            auto results = tp_worker_pool_->collectAll(timeout_ms);
            for (auto &result : results)
            {
                if (!result.completed)
                {
                    worker_timeout = true;
                    all_success = false;
                }
                if (!result.success)
                    all_success = false;
                if (result.exception && !first_exception)
                {
                    first_exception = result.exception;
                    first_exception_device = result.worker_index;
                    all_success = false;
                }
            }
            if (worker_timeout && timeout_ms > 0)
            {
                abortAfterTPWorkerTimeout(
                    "observeDeviceMoERebalanceDispatchTicket",
                    timeout_ms,
                    tp_worker_pool_->completedCount(),
                    tp_worker_pool_->numWorkers());
            }
            if (first_exception)
            {
                LOG_ERROR("[RankOrchestrator] HIP Device MoE ticket observation re-throwing participant exception from "
                          << first_exception_device);
                std::rethrow_exception(first_exception);
            }
            if (!all_success)
                return false;
        }

        const auto &authoritative =
            rank_hosted_device_moe_rebalance_tickets_.front();
        for (size_t index = 1; index < device_runners_.size(); ++index)
        {
            const auto &candidate =
                rank_hosted_device_moe_rebalance_tickets_[index];
            if (!authoritative.hasSameDispatchDecision(candidate))
            {
                LOG_ERROR("[RankOrchestrator] HIP Device MoE controllers published divergent maintenance decisions"
                          << " participant=" << index
                          << " authoritative_committed="
                          << authoritative.decode_rounds_committed
                          << " divergent_committed="
                          << candidate.decode_rounds_committed
                          << " authoritative_remaining="
                          << authoritative.decode_rounds_until_maintenance
                          << " divergent_remaining="
                          << candidate.decode_rounds_until_maintenance
                          << " authoritative_due="
                          << authoritative.maintenance_due
                          << " divergent_due=" << candidate.maintenance_due
                          << " authoritative_error="
                          << authoritative.error_code
                          << " divergent_error=" << candidate.error_code);
                std::terminate();
            }
        }

        rank_hosted_device_moe_rebalance_ticket_count_ =
            device_runners_.size();
        *out_ticket = authoritative;
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_device_moe_rebalance_dispatch_tickets_validated",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"maintenance_due",
              authoritative.maintenance_due ? "true" : "false"},
             {"bytes_per_participant",
              std::to_string(sizeof(DeviceMoERebalanceDispatchTicket))}});
        return true;
    }

    bool RankOrchestrator::
        submitHostScheduledDeviceMoERebalanceMaintenance(
            const DeviceMoERebalanceDispatchTicket &ticket)
    {
        if (!pp_stage_runners_.empty() ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            device_runners_.empty() ||
            rank_hosted_device_moe_rebalance_ticket_count_ !=
                device_runners_.size() ||
            !ticket.hasSameDispatchDecision(
                rank_hosted_device_moe_rebalance_tickets_.front()))
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE maintenance submission rejected a stale rank decision");
            return false;
        }

        if (!tp_worker_pool_ && device_runners_.size() > 1u)
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE maintenance submission has no persistent participant pool");
            return false;
        }

        bool all_success = true;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        if (device_runners_.size() == 1u)
        {
            all_success = device_runners_.front() &&
                          device_runners_.front()
                              ->submitHostScheduledDeviceMoERebalanceMaintenance(
                                  rank_hosted_device_moe_rebalance_tickets_
                                      .front());
        }
        else
        {
            const auto kernel_phase = KernelProfiler::getCurrentPhase();
            const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            const auto kv_phase = KVCacheProfiler::getCurrentPhase();
            const auto executor_phase = GraphExecutorStats::currentPhase();
            tp_worker_pool_->dispatch(
                [this, kernel_phase, rocm_phase, cuda_phase, kv_phase,
                 executor_phase](size_t index) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);
                    if (index >= device_runners_.size() ||
                        !device_runners_[index])
                    {
                        return false;
                    }
                    const DeviceId device =
                        device_runners_[index]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                    return device_runners_[index]
                        ->submitHostScheduledDeviceMoERebalanceMaintenance(
                            rank_hosted_device_moe_rebalance_tickets_[index]);
                });

            const int timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            bool worker_timeout = false;
            auto results = tp_worker_pool_->collectAll(timeout_ms);
            for (auto &result : results)
            {
                if (!result.completed)
                {
                    worker_timeout = true;
                    all_success = false;
                }
                if (!result.success)
                    all_success = false;
                if (result.exception && !first_exception)
                {
                    first_exception = result.exception;
                    first_exception_device = result.worker_index;
                    all_success = false;
                }
            }
            if (worker_timeout && timeout_ms > 0)
            {
                abortAfterTPWorkerTimeout(
                    "submitHostScheduledDeviceMoERebalanceMaintenance",
                    timeout_ms,
                    tp_worker_pool_->completedCount(),
                    tp_worker_pool_->numWorkers());
            }
        }

        /* Any participant may have enqueued a collective or acknowledgement
         * before a sibling reported failure. Continuing would leave an
         * unknowable distributed timeline, so post-dispatch failure is fatal. */
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE maintenance submission re-throwing participant exception from "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE maintenance submission failed after distributed dispatch began");
            std::terminate();
        }

        rank_hosted_device_moe_rebalance_ticket_count_ = 0u;
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_device_moe_rebalance_host_scheduled_submissions",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"maintenance_due",
              ticket.maintenance_due ? "true" : "false"},
             {"submission", "all_participants_before_next_observation"}});
        return true;
    }

    bool RankOrchestrator::maybeApplyDecodeBoundaryMaintenance(
        uint64_t committed_tokens)
    {
        if (committed_tokens == 0u)
        {
            LOG_ERROR("[RankOrchestrator] Device MoE maintenance cannot retire an empty decode transaction");
            return false;
        }
        const bool use_pp_participants = !pp_stage_runners_.empty();
        auto &participants =
            use_pp_participants ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
            return true;
        if (!use_pp_participants)
        {
            const auto policy =
                deviceMoERebalanceMaintenanceExecutionPolicy();
            if (policy ==
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance)
            {
                const auto schedule =
                    deviceMoERebalanceHostedObservationSchedule();
                if (!schedule.valid())
                {
                    LOG_ERROR("[RankOrchestrator] Hosted HIP Device MoE maintenance has no shared immutable observation schedule");
                    return false;
                }
                if (!rank_hosted_device_moe_observation_schedule_initialized_)
                {
                    rank_hosted_device_moe_rebalance_observation_schedule_ =
                        schedule;
                    rank_hosted_device_moe_rounds_until_observation_ =
                        schedule.initial_round_interval;
                    rank_hosted_device_moe_observation_schedule_initialized_ =
                        true;
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "rank_device_moe_rebalance_observation_schedule_initializations",
                        1.0,
                        "decode",
                        "rank",
                        {{"initial_round_interval",
                          std::to_string(schedule.initial_round_interval)},
                         {"recurring_round_interval",
                          std::to_string(schedule.recurring_round_interval)},
                         {"maximum_committed_rounds_per_boundary",
                          std::to_string(
                              schedule
                                  .maximum_committed_rounds_per_boundary)}});
                }
                else if (
                    schedule !=
                    rank_hosted_device_moe_rebalance_observation_schedule_)
                {
                    LOG_ERROR("[RankOrchestrator] Hosted HIP Device MoE observation schedule changed inside one request");
                    return false;
                }
                if (rank_hosted_device_moe_rounds_until_observation_ ==
                    0u)
                {
                    LOG_ERROR("[RankOrchestrator] Hosted HIP Device MoE observation countdown reached an invalid zero state");
                    return false;
                }

                const uint64_t rounds_until_observation =
                    rank_hosted_device_moe_rounds_until_observation_;
                if (committed_tokens < rounds_until_observation)
                {
                    rank_hosted_device_moe_rounds_until_observation_ -=
                        static_cast<uint32_t>(committed_tokens);
                    return submitHostScheduledDeviceMoERebalanceKnownNonDueBoundary();
                }

                rank_hosted_device_moe_rounds_until_observation_ = 0u;

                DeviceMoERebalanceDispatchTicket ticket;
                if (!observeDeviceMoERebalanceDispatchTicket(&ticket) ||
                    !submitHostScheduledDeviceMoERebalanceMaintenance(
                        ticket))
                {
                    return false;
                }
                const uint32_t next_remaining_rounds =
                    ticket.maintenance_due != 0u
                        ? schedule.recurring_round_interval
                        : ticket.decode_rounds_until_maintenance;
                rank_hosted_device_moe_rounds_until_observation_ =
                    next_remaining_rounds;
                if (rank_hosted_device_moe_rounds_until_observation_ ==
                    0u)
                {
                    LOG_ERROR("[RankOrchestrator] Authenticated HIP Device MoE ticket could not arm the next conservative observation interval");
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "rank_device_moe_rebalance_observation_intervals_armed",
                    1.0,
                    "decode",
                    "rank",
                    {{"maintenance_due",
                      ticket.maintenance_due != 0u ? "true" : "false"},
                     {"remaining_rounds",
                      std::to_string(next_remaining_rounds)},
                     {"rounds_until_observation",
                      std::to_string(
                          rank_hosted_device_moe_rounds_until_observation_)}});
                return true;
            }
            if (policy ==
                DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported)
            {
                LOG_ERROR("[RankOrchestrator] Device MoE maintenance participants do not share one supported scheduling policy");
                return false;
            }
        }
        if (participants.size() == 1)
        {
            return participants.front() &&
                   participants.front()->maybeApplyDecodeBoundaryMaintenance(
                       committed_tokens);
        }

        /*
         * LocalTP maintenance graphs contain NCCL/RCCL collectives and a
         * domain rendezvous.  Every participant must enter the boundary
         * concurrently; serial child calls would make the first device wait
         * for siblings that the host has not dispatched yet.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = std::make_unique<TPWorkerPool>(participants.size());
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback(
                    [this]()
                    {
                        LOG_WARN("[TPWorkerPool] Device MoE decode-boundary maintenance failure detected - aborting collective backend");
                        tp_ctx_->requestAbort();
                    });
            }
        }
        if (tp_worker_pool_->numWorkers() != participants.size())
        {
            LOG_ERROR("[RankOrchestrator] Device MoE decode-boundary maintenance participant count does not match the persistent TP worker pool");
            return false;
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this, use_pp_participants, committed_tokens, kernel_phase,
             rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto &worker_participants =
                    use_pp_participants ? pp_stage_runners_ : device_runners_;
                if (i >= worker_participants.size() || !worker_participants[i])
                    return false;

                const DeviceId device =
                    worker_participants[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                return worker_participants[i]
                    ->maybeApplyDecodeBoundaryMaintenance(committed_tokens);
            });

        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (auto &result : results)
        {
            if (!result.completed)
            {
                worker_timeout = true;
                all_success = false;
            }
            if (!result.success)
                all_success = false;
            if (result.exception && !first_exception)
            {
                first_exception = result.exception;
                first_exception_device = result.worker_index;
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "maybeApplyDecodeBoundaryMaintenance",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Device MoE decode-boundary maintenance re-throwing primary exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        return all_success;
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

        auto query_runners = [&](
                                 std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                 std::vector<PrefixLookupResult> &hits,
                                 PrefixFingerprintCoordinationPolicy
                                     fingerprint_policy)
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
                    runner->moePlacementEpoch(),
                    fingerprint_policy);
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
        query_runners(
            device_runners_,
            last_device_prefix_hits_,
            PrefixFingerprintCoordinationPolicy::ValidateParticipantLocally);
        query_runners(
            pp_stage_runners_,
            last_pp_prefix_hits_,
            PrefixFingerprintCoordinationPolicy::ValidateParticipantLocally);

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

        /*
         * Prefix population replaces every child runner's live request state.
         * Retire the aggregate before the first child crosses that boundary so
         * no observer can carry a mailbox from the previous request into the
         * restored transaction, even if child population later reports an
         * error.
         */
        invalidateRankResidentLogicalStateAggregate(
            "populate_prefix",
            "prefix_replaces_child_live_state");

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

    bool RankOrchestrator::harvestPrefix(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        int prompt_token_count)
    {
        if (!admission.cache_enabled || !admission.supported)
            return false;

        bool saw_runner = false;
        bool ok = true;
        auto harvest_runners = [&]
        (
            std::vector<std::unique_ptr<IInferenceRunner>> &runners,
            const std::vector<PrefixLookupResult> &admissions)
        {
            size_t admission_index = 0;
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                if (admission_index >= admissions.size())
                    return false;
                saw_runner = true;
                ok = runner->harvestPrefix(
                         admissions[admission_index++],
                         tokens,
                         prompt_token_count) &&
                     ok;
            }
            return admission_index == admissions.size();
        };

        const bool admissions_complete =
            harvest_runners(device_runners_, last_device_prefix_hits_) &&
            harvest_runners(pp_stage_runners_, last_pp_prefix_hits_);
        return saw_runner && admissions_complete && ok;
    }

    bool RankOrchestrator::restorePrefixTerminalState(const PrefixLookupResult &hit)
    {
        const int common_tokens = std::max(0, hit.cached_tokens);
        if (common_tokens <= 0 ||
            (hit.requires_terminal_logits && !hit.has_terminal_logits))
            return false;

        /*
         * A nested LocalTP domain can be one non-head PP stage. Its children
         * own KV/runtime prefix payloads but intentionally own no terminal
         * logits. Requiring logits at this composite boundary made the outer
         * pipeline fail even though its final stage had restored the sole
         * authoritative terminal row. The typed requirement bit, rather than
         * orchestration depth, decides whether absence is an error.
         */

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

        if (device_runners_.size() > 1 &&
            logits_gatherer_ &&
            logits_gatherer_->isAllocated())
        {
            const bool has_gpu_child =
                std::any_of(
                    device_runners_.begin(),
                    device_runners_.end(),
                    [](const std::unique_ptr<IInferenceRunner> &runner)
                    {
                        return runner && runner->primaryDeviceId().is_gpu();
                    });

            if (!has_gpu_child)
            {
                /*
                 * A TP prefix hit restores terminal logits into each child
                 * shard. The rank-level CPU sampling surface is a separate
                 * full-vocabulary aggregate that may still contain the final
                 * row from the previous request. Rebuild that aggregate in the
                 * same restore transaction so no caller can observe restored
                 * KV/terminal state paired with stale logits.
                 *
                 * GPU TP deliberately does not enter this path. Its production
                 * sampler consumes device-owned child logits directly, and a
                 * restore must not manufacture a D2H host mirror merely to keep
                 * an obsolete aggregate warm.
                 */
                if (!logits_gatherer_->gather(
                        device_runners_,
                        /*seq_len=*/1,
                        vocab_size()))
                {
                    LOG_ERROR(
                        "RankOrchestrator::restorePrefixTerminalState could not "
                        "refresh CPU TP terminal logits after child restore");
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "prefix_cache",
                    "cpu_tp_terminal_logits_aggregate_refreshes",
                    1.0,
                    "restore",
                    "rank",
                    {{"participants", std::to_string(device_runners_.size())},
                     {"cached_tokens", std::to_string(common_tokens)},
                     {"ownership", "cpu_rank_aggregate"}});
            }
        }

        if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
        {
            if (!logits_gatherer_)
            {
                logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    0,
                    0,
                    logits_backend_resolver_);
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

    PrefixStateSnapshot RankOrchestrator::captureLivePrefixCheckpoint(
        const PrefixCheckpointCaptureRequest &request) const
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
                PrefixStateSnapshot child =
                    runner->captureLivePrefixCheckpoint(request);
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

        const bool use_authoritative_child_logical_state =
            pp_stage_runners_.empty() && !device_runners_.empty();
        std::optional<int> authoritative_child_position;
        std::vector<int> authoritative_child_positions;
        std::vector<int> authoritative_child_sequence_lengths;
        std::optional<bool> authoritative_verifier_identity_present;
        auto adopt_authoritative_child_logical_state =
            [&](const PrefixRuntimeStateSnapshot &child)
        {
            /*
             * A child probe is already a host-visible observation boundary. On
             * GPU it materializes either the live logical-state mailbox or the
             * canonical cache-owned sequence count after mailbox retirement.
             * Requiring a currently live mailbox again at rank scope would
             * discard that proven child truth immediately after restore and
             * revive the rank's scheduler-only host cursor.
             *
             * Pipeline stages may legitimately own different positions, so
             * only symmetric non-PP participants participate in this equality
             * contract. Empty mock/uninitialized probes carry no authority.
             */
            if (!use_authoritative_child_logical_state ||
                !child.initialized)
                return;
            if (!authoritative_child_position.has_value())
            {
                authoritative_child_position = child.current_position;
                authoritative_child_positions = child.positions;
                authoritative_child_sequence_lengths =
                    child.sequence_lengths;
                return;
            }
            if (*authoritative_child_position != child.current_position ||
                authoritative_child_positions != child.positions ||
                authoritative_child_sequence_lengths != child.sequence_lengths)
            {
                throw std::runtime_error(
                    "Rank prefix-state diagnostics observed divergent "
                    "canonical logical metadata across LocalTP participants");
            }
        };

        auto merge_child =
            [&snapshot,
             &authoritative_verifier_identity_present,
             use_authoritative_child_logical_state](
                const PrefixRuntimeStateSnapshot &child)
        {
            constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
            constexpr uint64_t kFnvPrime = 1099511628211ull;
            auto fold_terminal_value = [](uint64_t digest, uint64_t value)
            {
                for (unsigned byte = 0; byte < sizeof(value); ++byte)
                {
                    digest ^= value & 0xffull;
                    digest *= kFnvPrime;
                    value >>= 8;
                }
                return digest;
            };
            auto merge_terminal_hash =
                [&](bool child_available,
                    size_t child_bytes,
                    uint64_t child_hash,
                    bool *aggregate_available,
                    size_t *aggregate_bytes,
                    uint64_t *aggregate_hash)
            {
                if (!child_available)
                    return;

                uint64_t digest =
                    *aggregate_available
                        ? *aggregate_hash
                        : kFnvOffsetBasis;
                digest = fold_terminal_value(
                    digest,
                    static_cast<uint64_t>(child_bytes));
                digest = fold_terminal_value(digest, child_hash);
                *aggregate_available = true;
                *aggregate_bytes += child_bytes;
                *aggregate_hash = digest;
            };

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
            snapshot.prefix_cache_ram_to_disk_demotions +=
                child.prefix_cache_ram_to_disk_demotions;
            snapshot.prefix_cache_device_hot_promotions +=
                child.prefix_cache_device_hot_promotions;
            snapshot.prefix_cache_device_hot_repromotions +=
                child.prefix_cache_device_hot_repromotions;
            snapshot.prefix_cache_device_hot_evictions +=
                child.prefix_cache_device_hot_evictions;
            snapshot.prefix_cache_disk_evictions +=
                child.prefix_cache_disk_evictions;
            snapshot.prefix_cache_device_hot_direct_hits +=
                child.prefix_cache_device_hot_direct_hits;
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
            snapshot.mtp_last_transaction_draft_depth =
                snapshot.mtp_last_transaction_draft_depth == 0
                    ? child.mtp_last_transaction_draft_depth
                    : std::min(
                          snapshot.mtp_last_transaction_draft_depth,
                          child.mtp_last_transaction_draft_depth);
            snapshot.mtp_last_transaction_emitted_token_count =
                snapshot.mtp_last_transaction_emitted_token_count == 0
                    ? child.mtp_last_transaction_emitted_token_count
                    : std::min(
                          snapshot.mtp_last_transaction_emitted_token_count,
                          child.mtp_last_transaction_emitted_token_count);
            const bool child_has_verifier_identity =
                child.mtp_observed_verifier_transaction_count > 0 ||
                child.mtp_observed_verifier_draft_depth > 0 ||
                !child.mtp_observed_verifier_draft_tokens.empty();
            if (!authoritative_verifier_identity_present.has_value())
            {
                authoritative_verifier_identity_present =
                    child_has_verifier_identity;
            }
            else if (use_authoritative_child_logical_state &&
                     *authoritative_verifier_identity_present !=
                         child_has_verifier_identity)
            {
                throw std::runtime_error(
                    "Rank prefix-state diagnostics observed committed MTP "
                    "verifier identity on only a subset of mirrored "
                    "participants");
            }
            if (child_has_verifier_identity)
            {
                if (child.mtp_observed_verifier_transaction_count <= 0 ||
                    child.mtp_observed_verifier_draft_depth <= 0 ||
                    child.mtp_observed_verifier_draft_tokens.size() !=
                        static_cast<size_t>(
                            child.mtp_observed_verifier_draft_depth))
                {
                    throw std::runtime_error(
                        "Rank prefix-state diagnostics observed an incomplete "
                        "committed MTP verifier identity");
                }
                if (snapshot.mtp_observed_verifier_draft_tokens.empty())
                {
                    snapshot.mtp_observed_verifier_transaction_count =
                        child.mtp_observed_verifier_transaction_count;
                    snapshot.mtp_observed_verifier_draft_depth =
                        child.mtp_observed_verifier_draft_depth;
                    snapshot.mtp_observed_verifier_draft_tokens =
                        child.mtp_observed_verifier_draft_tokens;
                }
                else if (
                    snapshot.mtp_observed_verifier_transaction_count !=
                        child.mtp_observed_verifier_transaction_count ||
                    snapshot.mtp_observed_verifier_draft_depth !=
                        child.mtp_observed_verifier_draft_depth ||
                    snapshot.mtp_observed_verifier_draft_tokens !=
                        child.mtp_observed_verifier_draft_tokens)
                {
                    auto format_tokens = [](const std::vector<int32_t> &tokens)
                    {
                        std::ostringstream out;
                        out << '[';
                        for (size_t index = 0; index < tokens.size(); ++index)
                        {
                            if (index != 0)
                                out << ',';
                            out << tokens[index];
                        }
                        out << ']';
                        return out.str();
                    };
                    throw std::runtime_error(
                        "Rank prefix-state diagnostics observed divergent "
                        "committed MTP verifier draft rows across mirrored "
                        "participants: authoritative=" +
                        format_tokens(
                            snapshot.mtp_observed_verifier_draft_tokens) +
                        " divergent=" +
                        format_tokens(
                            child.mtp_observed_verifier_draft_tokens));
                }
            }
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
            /*
             * LocalTP participants own distinct device buffers even when the
             * model policy replicates their logical rows.  Fold every child in
             * stable participant order rather than selecting participant zero;
             * a one-device restore defect must change the rank-level digest.
             */
            merge_terminal_hash(
                child.terminal_hidden_hash_available,
                child.terminal_hidden_bytes,
                child.terminal_hidden_hash,
                &snapshot.terminal_hidden_hash_available,
                &snapshot.terminal_hidden_bytes,
                &snapshot.terminal_hidden_hash);
            snapshot.terminal_hidden_values.insert(
                snapshot.terminal_hidden_values.end(),
                child.terminal_hidden_values.begin(),
                child.terminal_hidden_values.end());
            merge_terminal_hash(
                child.terminal_logits_hash_available,
                child.terminal_logits_bytes,
                child.terminal_logits_hash,
                &snapshot.terminal_logits_hash_available,
                &snapshot.terminal_logits_bytes,
                &snapshot.terminal_logits_hash);
            snapshot.terminal_logits_values.insert(
                snapshot.terminal_logits_values.end(),
                child.terminal_logits_values.begin(),
                child.terminal_logits_values.end());
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
                const PrefixRuntimeStateSnapshot child =
                    runner->prefixStateProbe();
                adopt_authoritative_child_logical_state(child);
                merge_child(child);
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
        if (authoritative_child_position.has_value())
        {
            /*
             * The rank host cursor is only a scheduler/response shadow. Child
             * probes own canonical sequence truth across both active mailbox
             * and post-restore cache-count lifecycles, so rank diagnostics must
             * never substitute the shadow merely because an outcome mailbox
             * was retired.
             */
            snapshot.current_position = *authoritative_child_position;
            snapshot.positions =
                std::move(authoritative_child_positions);
            snapshot.sequence_lengths =
                std::move(authoritative_child_sequence_lengths);
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
        const int max_decode_like_rows = config_.mtp.enabled
                                             ? std::max(1, resolveMTPMaxTargetQueryRows(config_.mtp))
                                             : 1;
        if (current_padded_seq_len_ <= 0 ||
            current_padded_seq_len_ > max_decode_like_rows)
        {
            return false;
        }

        const auto &plan = config_.moe_routed_expert_plan;
        if (!plan || !plan->usesExpertOverlayAuthority())
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

    bool RankOrchestrator::replicatedMTPSidecarSnapshotsActive() const noexcept
    {
        return tp_ctx_ &&
               tp_ctx_->degree() > 1 &&
               retainsMTPGraphCapacity(config_.mtp) &&
               config_.mtp.sidecar_dense_policy ==
                   MTPSidecarDensePolicy::ReplicatedPerParticipant;
    }

    bool RankOrchestrator::mtpSnapshotRetainsIndependentSharding(
        const std::string &stage_type)
    {
        /*
         * Replicating the learned predictor does not replicate its
         * vocabulary table/head, nor does it replace ExpertOverlay's routed
         * expert publication protocol.  Those authorities keep the ordinary
         * schema layout while all other TP-shaped predictor intermediates are
         * complete per-participant tensors.
         */
        static constexpr std::array<std::string_view, 5>
            kIndependentlyOwnedMTPStages = {
                "EMBEDDING",
                "LM_HEAD",
                "MOE_EXPERT_OUTPUT",
                "MOE_ROUTE_CONTRIBUTIONS",
                "MOE_SHARED_RANK_BANK_PUBLISH",
            };
        return std::find(
                   kIndependentlyOwnedMTPStages.begin(),
                   kIndependentlyOwnedMTPStages.end(),
                   stage_type) != kIndependentlyOwnedMTPStages.end();
    }

    SnapshotShardingMode RankOrchestrator::resolveSnapshotShardingMode(const std::string &key) const
    {
        /*
         * Snapshot scopes describe transaction timing, not tensor layout.
         * Resolve the underlying semantic stage through the ordinary model
         * schema so column- and row-parallel LocalTP intermediates are combined
         * exactly as their unscoped counterparts.  Only the terminal LM head
         * is special: MTP explicitly mirrors that full-vocabulary projection on
         * every participant, even while the transformer body remains TP.
         */
        static constexpr std::array<std::string_view, 2> kConditionScopes = {
            "MTP_REQUEST_BATCH_CONDITION_",
            "MTP_SCALAR_CONDITION_",
        };
        std::string_view condition_scope;
        for (const std::string_view candidate : kConditionScopes)
        {
            if (key.rfind(candidate, 0) == 0)
            {
                condition_scope = candidate;
                break;
            }
        }
        const bool condition_snapshot = !condition_scope.empty();
        const std::string_view semantic_key_view =
            condition_snapshot
                ? std::string_view(key).substr(condition_scope.size())
                : std::string_view(key);
        const std::string semantic_key(semantic_key_view);
        const bool mtp_model_snapshot =
            isMTPDepthQualifiedSnapshot(semantic_key);
        if (extractStageType(semantic_key) == "LM_HEAD" &&
            (condition_snapshot ||
             (mtp_model_snapshot && usesMirroredMTPHeadForVerifier())))
        {
            return SnapshotShardingMode::REPLICATED;
        }

        SnapshotShardingMode mode =
            getStageShardingMode(semantic_key, stage_sharding_map_);

        /*
         * Static schemas describe the main transformer's TP layout.  A
         * replicated MTP predictor deliberately reuses the same semantic stage
         * names while binding complete dense/shared weights and omitting the
         * row-parallel collectives.  Reclassify only depth-qualified model
         * checkpoints whose schema layout denotes a TP partial.  Independent
         * vocabulary and ExpertOverlay publications retain their own authority
         * above; UNKNOWN/GATHERED/ROOT_ONLY modes are never guessed here.
         */
        const std::string stage_type = extractStageType(semantic_key);
        const bool schema_mode_is_tp_partial =
            mode == SnapshotShardingMode::COLUMN_PARALLEL ||
            mode == SnapshotShardingMode::PACKED_COLUMN_PARALLEL ||
            mode == SnapshotShardingMode::ROW_PARALLEL;
        if (mtp_model_snapshot &&
            replicatedMTPSidecarSnapshotsActive() &&
            schema_mode_is_tp_partial &&
            !mtpSnapshotRetainsIndependentSharding(stage_type))
        {
            return SnapshotShardingMode::REPLICATED;
        }

        /*
         * Phase-split MoE overlay uses DenseTP for prefill but switches dense and
         * always-on shared decode work to full replicated weights. The schema still
         * marks dense output projections as ROW_PARALLEL because that is correct
         * for plain TP. During replicated decode, summing per-device snapshots
         * doubles the captured tensor and produces false parity failures.
         */
        if (mode == SnapshotShardingMode::ROW_PARALLEL &&
            phaseSplitReplicatedDecodeSnapshotsActive() &&
            isPhaseSplitReplicatedDecodeSnapshotKey(semantic_key))
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
        else if (result.mode == SnapshotShardingMode::PACKED_COLUMN_PARALLEL &&
                 !result.device_data.empty())
        {
            /*
             * Qwen GDN participants publish dependency-closed local rows.
             * Resolve every semantic repeat from the model's authoritative
             * head geometry and each producing stage's real shape; no retired
             * replicated-Q/K representation is accepted.
             */
            std::string layout_error;
            if (!model_ctx_ || !model_ctx_->loader())
            {
                layout_error = "model loader is unavailable";
            }
            else
            {
                const auto loader = model_ctx_->loader();
                const std::string metadata_prefix =
                    model_ctx_->architecture() + ".ssm.";
                const int key_heads =
                    loader->getInt(metadata_prefix + "group_count", 0);
                const int value_heads =
                    loader->getInt(metadata_prefix + "time_step_rank", 0);
                const int state_width =
                    loader->getInt(metadata_prefix + "state_size", 0);
                result.column_groups = resolvePackedGDNColumnGroups(
                    result.device_data,
                    result.tp_degree,
                    extractStageType(key),
                    key_heads,
                    value_heads,
                    state_width,
                    &layout_error);
            }

            if (result.column_groups.empty())
            {
                LOG_ERROR("RankOrchestrator::getTPSnapshot: packed checkpoint '"
                          << key << "' has no valid GDN TP layout: "
                          << layout_error);
            }
            else
            {
                const size_t total_cols = std::accumulate(
                    result.column_groups.begin(),
                    result.column_groups.end(),
                    size_t{0},
                    [](size_t sum, const SnapshotColumnGroup &group)
                    {
                        return sum + group.global_cols;
                    });
                for (auto &device : result.device_data)
                {
                    device.global_start_col = 0;
                    device.global_total_cols = total_cols;
                }
            }
        }

        return result;
    }

    std::string RankOrchestrator::describeFailedMirroredMTPState() const
    {
        if (!DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_GRAPH_REUSE_DIAGNOSTICS"))
        {
            return {};
        }

        std::vector<std::vector<MTPMirroredTensorDigest>>
            participant_digests;
        participant_digests.reserve(device_runners_.size());
        for (const auto &participant : device_runners_)
        {
            auto *device_orchestrator =
                dynamic_cast<DeviceGraphOrchestrator *>(participant.get());
            participant_digests.push_back(
                device_orchestrator
                    ? device_orchestrator
                          ->captureMirroredMTPDigests()
                    : std::vector<MTPMirroredTensorDigest>{});
        }
        return describeMirroredDigestMismatch(participant_digests);
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

        /*
         * createForTest() may inject runners whose placement metadata names a
         * CUDA or ROCm device.  Those identifiers are policy fixtures, not live
         * hardware ownership.  The injected LocalTP context above is host-only,
         * and the unit-test boundary must end before BackendManager lookup so
         * teardown cannot create a physical GPU context as a side effect.
         */
        if (!external_device_backend_access_enabled_)
            return;

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

    IBackend *RankOrchestrator::resolveLogitsBackend(DeviceId device) const
    {
        if (logits_backend_resolver_)
            return logits_backend_resolver_(device);
        return getBackendFor(device);
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

    MoEOverlayAuthorityExecutionKind
    RankOrchestrator::moeOverlayAuthorityExecution() const
    {
        const auto &runners = pp_stage_runners_.empty()
                                  ? device_runners_
                                  : pp_stage_runners_;
        if (runners.empty())
            return MoEOverlayAuthorityExecutionKind::Unresolved;

        MoEOverlayAuthorityExecutionKind selected =
            MoEOverlayAuthorityExecutionKind::Unresolved;
        for (const auto &runner : runners)
        {
            if (!runner)
            {
                throw std::logic_error(
                    "ExpertOverlay authority query reached a null rank participant");
            }
            const auto participant =
                runner->moeOverlayAuthorityExecution();
            if (selected ==
                MoEOverlayAuthorityExecutionKind::Unresolved)
            {
                selected = participant;
                continue;
            }
            if (participant != selected)
            {
                throw std::logic_error(
                    "RankOrchestrator participants disagree on the frozen ExpertOverlay authority execution backend");
            }
        }
        return selected;
    }

    bool RankOrchestrator::
        deviceResidentMoEOverlayMaintenanceReady() const
    {
        const auto &runners = pp_stage_runners_.empty()
                                  ? device_runners_
                                  : pp_stage_runners_;
        if (runners.empty() ||
            moeOverlayAuthorityExecution() !=
                MoEOverlayAuthorityExecutionKind::
                    DeviceResident)
        {
            return false;
        }
        return std::all_of(
            runners.begin(),
            runners.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner
                           ->deviceResidentMoEOverlayMaintenanceReady();
            });
    }

    bool RankOrchestrator::applyMoEExpertMasksForAllDevices(
        const MoERebalanceController &controller,
        const ExpertReplicaSet *replica_arrivals,
        const MoELayeredExpertOwnership *previous_ownership)
    {
        auto snapshot = snapshotMoEExpertMasksForAllDevices(
            controller,
            replica_arrivals,
            previous_ownership);
        return applyMoEExpertMasksForAllDevices(
            snapshot.masks_by_participant,
            snapshot.domain_id,
            snapshot.transferMasks());
    }

    RankOrchestrator::MoEExpertMaskSnapshot RankOrchestrator::snapshotMoEExpertMasksForAllDevices(
        const MoERebalanceController &controller,
        const ExpertReplicaSet *replica_arrivals,
        const MoELayeredExpertOwnership *previous_ownership) const
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
        else if (previous_ownership)
        {
            snapshot.transfer_masks_by_participant =
                rank_orchestrator_detail::buildOwnershipArrivalTransferMasks(
                    controller,
                    *previous_ownership);
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
