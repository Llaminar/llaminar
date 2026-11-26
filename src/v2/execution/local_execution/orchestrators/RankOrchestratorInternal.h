/**
 * @file RankOrchestratorInternal.h
 * @brief Shared internal helpers, rank_orchestrator_detail declarations, and RetainedSourceAdmission for RankOrchestrator shards.
 */
#pragma once
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
 * Retained prefill heads never borrow the current transaction's logits buffer.
 * Diagnostic assembly obeys each immutable producer publication, including
 * complete values and explicit output-column partitions, instead of assuming
 * that a semantic checkpoint always retains its schema's historical layout.
 * The main host logits output holds the terminal row only. Full-context KV
 * capacity never determines output storage; verifier rows are a separate API.
 * PP child construction authenticates stage-local expert ownership before
 * materializing any child, preserving the authored TP-in-PP decomposition.
 * The outer PP family owns the shared weight lifecycle: nested participants
 * cannot release sources or publish graph completion for unfinished siblings.
 * Resident condition forwards retire the preceding host result before fan-out.
 * Only an explicitly enabled diagnostic observation republishes current logits;
 * pending input publication and device-only sampling do not gather host rows.
 * Composed forward-only domain completion refreshes an enabled tail diagnostic
 * through the original child publication edges before advancing rank metadata.
 * Every main-model entry admits all shared source channels before participant
 * dispatch, including a first MTP condition or verifier after prefix restoration.
 * Input-only publication does not open or replace a storage interval.
 *
 * Ordinary native QFN local TP derives one immutable assignment from the original
 * directory before weight slicing; physical device order remains context-owned.
 * CPU serial storage declarations survive admission and runtime projection.
 * Explicit prefill transport diagnostics freeze into the existing phase-aware
 * graph configuration; driver topology and decode/MTP policy stay authoritative.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */

#include "RankOrchestrator.h"
#include "execution/qwen4exp/PooledImmutableRetainedParticipants.h"
#include "execution/moe/MoEOverlayCPUHostMaintenance.h"
#include "snapshots/SnapshotAuthoredTokenRows.h"
#include "PipelineResidualWidth.h"
#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
#include "models/qwen4exp/Qwen4ExpReplicatedPLEPrefixOrigin.h"

#include "transfer/MappedTransferProgressEpoch.h"
#include "LogitsGatherer.h"
#include "DeviceSampler.h"
#include "DeviceGraphOrchestrator.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "PipelineGraphExecutionPlan.h"
#include "PipelineDeviceGeneration.h"
#include "../../../collective/CollectiveTimeoutPolicy.h"
#include "../../mtp/MTPSpecTransactionDriver.h"
#include "../../mtp/MTPSpecStateContract.h"
#include "../../factory/InferenceRunnerFactory.h"
#include "../../moe/MoEExpertOverlayRuntimePlan.h"
#include "../../moe/MoEOverlayCpuPreparationGate.h"
#include "../../moe/MoEOverlayNodeLocalRouteExchange.h"
#include "../../moe/MoEOverlayInferenceTransactionService.h"
#include "../../moe/MoEOverlayInferenceInterferenceProbe.h"
#include "../engine/PrefillBucketUtils.h"
#include "../../prefix_cache/PrefixCacheCoordinator.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../kernels/cpu/sampling/CPUSamplerPrimitives.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../collective/DeviceCountedAllGather.h"
#include "../../moe/MoEProjectionArenaGeometry.h"
#include "../../../planning/ActivationBufferSizing.h"
#include "../../../planning/ModelMemoryProfile.h"
#include "planning/TensorParallelSourceDeclaration.h"
#include "../../../collective/ILocalPPContext.h"
#include "../../../config/TensorParallelConfig.h"
#include "../../config/KVHeadConsumerGeometry.h"
#include "../../../interfaces/IModelContext.h"
#include "../../../loaders/ModelContext.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../loaders/WeightManager.h"
#include "../../../planning/CollectiveMemoryEstimator.h"
#include "../../moe/MoEOverlayPipelineStageBinding.h"
#include "../../moe/MoEOverlayResidencyAuthority.h"
#include "../../moe/MoEOverlayParticipantResidency.h"
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
         * @brief Authenticate all local graph owners against a frozen PP domain.
         * @param runner The single-device runner or existing nested TP owner.
         * @param segment Immutable projection of its setup membership.
         * @return Whether ordered runner and collective membership still agree.
         *
         * This query allocates nothing and enters no collective. In particular,
         * an unchanged leader cannot conceal a replaced/missing nonleader.
         */
        bool pipelineStageMatches(const IInferenceRunner &runner,
            const PipelineGraphExecutionSegment &segment) noexcept
        {
            const auto expected = segment.execution == PipelineGraphSegmentExecution::HostDeclarative
                ? ServingGraphPreparationKind::EagerHostGraph
                : ServingGraphPreparationKind::NativeDeviceExecutableFamily;
            if (runner.primaryDeviceId() != segment.primaryDevice() ||
                runner.servingGraphPreparationKind() != expected)
                return false;
            const auto *rank = dynamic_cast<const IRankOrchestrator *>(&runner);
            if (!rank) return segment.participants.size() == 1;
            if (rank->device_count() != static_cast<int>(segment.participants.size())) return false;
            const auto *context = rank->localTPContext();
            if (!context || context->degree() != rank->device_count() ||
                context->devices().size() != segment.participants.size()) return false;
            for (size_t index = 0; index < segment.participants.size(); ++index)
            {
                const auto *child = rank->deviceRunner(static_cast<int>(index));
                if (!child || dynamic_cast<const IRankOrchestrator *>(child) ||
                    child->primaryDeviceId() != segment.participants[index] ||
                    child->servingGraphPreparationKind() != expected ||
                    context->devices()[index].toLocalDeviceId() != segment.participants[index])
                    return false;
            }
            return true;
        }

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
        void recordRankPrefixSnapshotFailure(
            const char *operation,
            const char *runner_group,
            size_t participant_index,
            const IInferenceRunner *runner,
            const std::string &reason,
            int expected_tokens = -1,
            int actual_tokens = -1)
;
        bool sameBackendGpuExpertTransferIsDirectOnly(DeviceId destination, DeviceId source)
;
        std::vector<std::vector<std::vector<bool>>> buildReplicaArrivalTransferMasks(
            const MoERebalanceController &controller,
            const ExpertReplicaSet &arrivals)
;
        std::vector<std::vector<std::vector<bool>>> buildOwnershipArrivalTransferMasks(
            const MoERebalanceController &controller,
            const MoELayeredExpertOwnership &previous_ownership)
;
    } // namespace rank_orchestrator_detail

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

    struct RankOrchestrator::RetainedSourceAdmission
    {
        std::unique_ptr<qwen4exp::ImmutableLookupExecutionBoundary::SourceGroupAdmission> original;
        bool failed = false;
        std::exception_ptr retired_failure;
        std::function<void()> cancel;
        /** @brief Cancel failed execution without discarding any original adopted retirement right. */
        struct Attempt {
            RankOrchestrator& rank;
            std::unique_ptr<RetainedSourceAdmission>& owner;
            const char* operation;
            bool keep = false;
            /** @brief Bind the exact host Rank/source scope before its original worker fanout. */
            Attempt(RankOrchestrator& rank,std::unique_ptr<RetainedSourceAdmission>& owner,const char* operation)
                :rank(rank),owner(owner),operation(operation) {}
            /** @brief Worker results have drained before this scope unwinds; missing native proof remains fatal. */
            ~Attempt() noexcept {
                if (!keep && owner) rank.failRetainedSourceAttempt(operation);
            }
        };
    };
} // namespace llaminar2
