/**
 * @file OrchestrationRunnerInternal.h
 * @brief Shared includes and internal helpers for OrchestrationRunner translation units.
 */
#pragma once

#include "execution/local_execution/graph/RetainedHostServiceDeclarations.h"
#include "execution/moe/MoEOverlayRestorationGeometry.h"
#include "execution/moe/MoEOverlayRowCandidateFinalization.h"
#include "execution/diagnostics/QueryKVGeometryObservation.h"
#include "planning/TensorParallelSourceDeclaration.h"
#include "planning/OriginalRankCapacityExchange.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "loaders/ImmutableRowTable.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "OrchestrationRunner.h"
#include "PreparedPipelineGraphMaterialization.h"
#include "execution/global/GlobalOrchestrator.h"
#include "execution/global/GlobalStageTPValidation.h"
#include "transfer/CapturedTransferChannel.h"
#include "collective/GlobalTPContext.h"
#include "collective/NativeCPUCollectiveProgram.h"
#include <variant>
#include "execution/moe/MoEOverlayCPUHostMaintenance.h"
#include "execution/moe/ExpertHistogramServingRegime.h"
#include "planning/ResolvedRankOrchestration.h"
#include "execution/runner/PreparedWeightAuthorityIdentity.h"
#include "IOrchestrationRunnerFactory.h"
#include "ModelContextRetirement.h"
#include "MTPVerifierForwardExecutor.h"
#include "MTPStochasticVerificationOwner.h"
#include "../global/NamedDomainGraphBuilder.h"
#include "../../app/StartupBanner.h"
#include "../../config/OrchestrationConfigParser.h"
#include "../../config/BackendSelector.h"
#include "../../config/TPPPValidator.h"
#include "../mpi_orchestration/ExecutionPlanBuilder.h"
#include "../factory/InferenceRunnerFactory.h"
#include "../mtp/MTPStateTransaction.h"
#include "../mtp/MTPDecodeCatchup.h"
#include "../mtp/MTPDeviceGenerationPolicy.h"
#include "../mtp/HostSparseQueryResponseScope.h"
#include "../mtp/MTPRequestSamplingPolicy.h"
#include "../mtp/MTPRejectionSampler.h"
#include "../mtp/MTPSpecDecodeMetadata.h"
#include "../mtp/MTPSpecDecodeTransaction.h"
#include "../mtp/MTPSpecStateContract.h"
#include "../mtp/MTPSpecTransactionDriver.h"
#include "../mtp/MTPVerifierPolicy.h"
#include "../mtp/MTPWeightManifest.h"
#include "../prefix_cache/PrefixCacheCoordinator.h"
#include "../prefix_cache/PrefixCacheShutdown.h"
#include "../local_execution/engine/PrefillBucketUtils.h"
#include "../local_execution/orchestrators/RankOrchestrator.h"
#include "../moe/MoEExpertOverlayAuthorityPlan.h"
#include "../moe/MoEExpertOwnerMap.h"
#include "../moe/MoEOverlayParticipantResidency.h"
#include "../moe/MoEOverlayParticipantMigration.h"
#include "../moe/MoEOverlayDistributedResidencyTransport.h"
#include "../moe/MoEOverlayMPIEconomyEvidenceExchange.h"
#include "../moe/MoEOverlayMPIResidencyProposalPublisher.h"
#include "../moe/MoEOverlayMPIRemoteProjectionTransport.h"
#include "../moe/MoEOverlayMPIResidencyConsensus.h"
#include "../moe/MoEOverlayCapacityAdmission.h"
#include "execution/moe/MoEOverlayPhysicalLaneDemand.h"
#include "../moe/MoEOverlayActivationChannelPlan.h"
#include "../moe/MoEOverlayRankBatchPreparation.h"
#include "../moe/MoEOverlayEconomyCalibrationController.h"
#include "../moe/MoEOverlayEconomyCertificationController.h"
#include "../moe/MoEOverlayDeviceServiceTelemetryPublisher.h"
#include "../moe/MoEOverlayHostAuthorityDeviceBankPublisher.h"
#include "../moe/MoEOverlayLocalCapacityPlanner.h"
#include "../moe/MoEOverlayCPUServiceMeasurement.h"
#include "../moe/MoEOverlayOriginalGPUCohortContexts.h"
#include "planning/PlanningGPUExpertMeasurement.h"
#include "../moe/MoEOverlayMigrationMeasurementExchange.h"
#include "../moe/MoEOverlayPhysicalResidencyFabric.h"
#include "../moe/MoEOverlayResidencyAuthority.h"
#include "../moe/MoEOverlayResidencySetup.h"
#include "../moe/MoEPipelinePreparedPlan.h"
#include "../moe/MoEOptimizationPipeline.h"
#include "../moe/MoEOverlayPipelineStageBinding.h"
#include "../moe/MoEOverlayResidencyMaintenanceService.h"
#include "../moe/MoEOverlayInferenceInterferenceProbe.h"
#include "../moe/MoEOverlayTierMigrationTransport.h"
#include "../../kernels/common/SamplingMath.h"
#include "utils/MTPDepthTransactionTraceCPU.h"
#include "utils/MTPDepthTransactionTracePerfStats.h"
#include "../parallelism_tree/ParallelismTree.h"
#include "../parallelism_tree/TreeToRunnerCompiler.h"
#include "../../collective/LocalTPContext.h"
#include "../../collective/ILocalPPContext.h"
#include "../../collective/BackendRouter.h"
#include "../../backends/BackendManager.h"
#include "../../backends/GPUDeviceContextPool.h"
#include "../mtp/MTPModelClassPolicy.h"
#include "../mtp/MTPDepthPolicyScopeProjection.h"
#include "../mtp/MTPServingForwardCaptureGeometry.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/ModelContextConfig.h"
#include "../../loaders/ModelLoader.h"
#include "../../loaders/GPUVramPreflight.h"
#include "../../loaders/MmapRegion.h"
#include "../../loaders/PreparedWeightStore.h"
#include "../../loaders/ExpertGemmRegistry.h"
#include "../../collective/CollectiveTimeoutPolicy.h"
#include "../local_execution/graph/SchemaFactoryRegistry.h"
#include "../../backends/ComputeBackend.h"
#include "../../planning/ClusterInventoryGatherer.h"
#include "../../planning/ModelMemoryProfile.h"
#include "../../config/OrchestrationStartupPolicy.h"
#include "../../planning/ActivationBufferSizing.h"
#include "../../planning/RankMemoryPlanInputs.h"
#include "../../planning/RankHardwareOwnership.h"
#include "../../backends/NativeContextPreparation.h"
#include "../../planning/MoEOverlayPlanningInputs.h"
#include "../../planning/MoEOverlayMemoryPlanInputs.h"
#include "../../planning/PlanningObservedResource.h"
#include "../../backends/DeviceAddressAdapter.h"
#include "../../kernels/KernelFactory.h"
#include "../../tensors/TensorFactory.h"
#include "../../transfer/MappedTransferProgressEpoch.h"
#include "../../transfer/TransferEngine.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/CPUFeatures.h"
#include "../../utils/MPITopology.h"
#include "../../utils/NodeDetection.h"
#include "../../utils/NUMATopology.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/PrefillGraphBucketDefaults.h"
#include "../../utils/WeightLoadingProfiler.h"
#include "../local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../execution/moe/MoERoutedExpertPlacementPlan.h"
#include "../../execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "../../execution/moe/MoEExpertOwnerMap.h"
#include "../../execution/moe/MoEOverlayDeviceControllerTopology.h"
#include "../../execution/moe/MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "../../execution/moe/MoEOverlayDeviceControllerGraphService.h"
#include "../../execution/moe/MoEOverlayParticipantGraphRunner.h"
#include "../../execution/moe/MoEOverlayHostServiceContract.h"
#include "../../execution/moe/MoEOverlayInferenceTransactionService.h"
#include "../../execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "../../execution/moe/RoutedExpertOwnerAssignment.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <print>
#include <random>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <type_traits>
#if defined(__GLIBC__)
#endif

namespace llaminar2
{
    namespace
    {
        /** Wire version for one root-authored request-selectable MTP policy. */
        constexpr std::int32_t kMTPRequestPolicyWireVersion = 2;

        /**
         * Fixed-width MPI payload for @ref MTPRequestPolicy.
         *
         * The three floating thresholds are transmitted as exact IEEE-754
         * bit pairs.  Converting them through the existing float broadcast
         * would let rank-local adaptive controllers make different decisions
         * at a threshold boundary.
         */
        using MTPRequestPolicyWire = std::array<std::int32_t, 23>;

        /** @brief Store one double without changing any payload bits. */
        void encodeMTPPolicyDouble(
            double value,
            MTPRequestPolicyWire *wire,
            std::size_t offset)
        {
            const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
            (*wire)[offset] = std::bit_cast<std::int32_t>(
                static_cast<std::uint32_t>(bits));
            (*wire)[offset + 1u] = std::bit_cast<std::int32_t>(
                static_cast<std::uint32_t>(bits >> 32u));
        }

        /** @brief Restore one exact double from its two MPI words. */
        [[nodiscard]] double decodeMTPPolicyDouble(
            const MTPRequestPolicyWire &wire,
            std::size_t offset)
        {
            const std::uint64_t low =
                std::bit_cast<std::uint32_t>(wire[offset]);
            const std::uint64_t high =
                std::bit_cast<std::uint32_t>(wire[offset + 1u]);
            return std::bit_cast<double>(low | (high << 32u));
        }

        /** @brief Encode every request-selectable field into one exact wire value. */
        [[nodiscard]] MTPRequestPolicyWire encodeMTPRequestPolicy(
            const MTPRequestPolicy &policy)
        {
            MTPRequestPolicyWire wire{
                kMTPRequestPolicyWireVersion,
                policy.enabled ? 1 : 0,
                policy.draft_tokens,
                static_cast<std::int32_t>(policy.verify_mode),
                policy.require_terminal_hidden_for_full_hit ? 1 : 0,
                static_cast<std::int32_t>(policy.depth_policy.mode),
                static_cast<std::int32_t>(policy.depth_policy.backend),
                static_cast<std::int32_t>(policy.depth_policy.model_class),
                policy.depth_policy.min_depth,
                policy.depth_policy.max_depth,
                policy.depth_policy.initial_depth,
                policy.depth_policy.window_size,
                policy.depth_policy.min_samples,
                policy.depth_policy.cooldown_steps,
                policy.depth_policy.promote_consecutive_windows,
                policy.depth_policy.use_generated_policy ? 1 : 0,
            };
            encodeMTPPolicyDouble(
                policy.depth_policy.promote_full_accept_rate,
                &wire,
                16u);
            encodeMTPPolicyDouble(
                policy.depth_policy.demote_zero_accept_rate.value_or(0.0),
                &wire,
                18u);
            encodeMTPPolicyDouble(
                policy.depth_policy.demote_acceptance_rate,
                &wire,
                20u);
            // Preserve automatic intent separately from numeric zero. The
            // topology profile is retained locally, never authored by a request.
            wire[22] = policy.depth_policy.demote_zero_accept_rate ? 1 : 0;
            return wire;
        }

        /**
         * @brief Decode one policy payload before capacity admission.
         * @param wire Exact fixed-width payload received from the authority.
         * @param error Receives malformed-wire diagnostics.
         * @return Complete policy, or empty when the wire itself is invalid.
         */
        [[nodiscard]] std::optional<MTPRequestPolicy>
        decodeMTPRequestPolicy(
            const MTPRequestPolicyWire &wire,
            std::string *error)
        {
            if (wire[0] != kMTPRequestPolicyWireVersion)
            {
                if (error)
                    *error = "unsupported MTP request-policy wire version";
                return std::nullopt;
            }
            if ((wire[1] != 0 && wire[1] != 1) ||
                (wire[4] != 0 && wire[4] != 1) ||
                (wire[15] != 0 && wire[15] != 1) ||
                (wire[22] != 0 && wire[22] != 1))
            {
                if (error)
                    *error = "MTP request-policy wire contains a non-boolean flag";
                return std::nullopt;
            }

            MTPRequestPolicy policy;
            policy.enabled = wire[1] != 0;
            policy.draft_tokens = wire[2];
            policy.verify_mode = static_cast<MTPVerifyMode>(wire[3]);
            policy.require_terminal_hidden_for_full_hit = wire[4] != 0;
            policy.depth_policy.mode =
                static_cast<MTPDepthPolicyMode>(wire[5]);
            policy.depth_policy.backend =
                static_cast<MTPDepthPolicyBackend>(wire[6]);
            policy.depth_policy.model_class =
                static_cast<MTPDepthPolicyModelClass>(wire[7]);
            policy.depth_policy.min_depth = wire[8];
            policy.depth_policy.max_depth = wire[9];
            policy.depth_policy.initial_depth = wire[10];
            policy.depth_policy.window_size = wire[11];
            policy.depth_policy.min_samples = wire[12];
            policy.depth_policy.cooldown_steps = wire[13];
            policy.depth_policy.promote_consecutive_windows = wire[14];
            policy.depth_policy.use_generated_policy = wire[15] != 0;
            policy.depth_policy.promote_full_accept_rate =
                decodeMTPPolicyDouble(wire, 16u);
            if (wire[22] != 0)
                policy.depth_policy.demote_zero_accept_rate = decodeMTPPolicyDouble(wire, 18u);
            else if (wire[18] != 0 || wire[19] != 0)
            {
                if (error)
                    *error = "automatic MTP threshold carries an explicit payload";
                return std::nullopt;
            }
            policy.depth_policy.demote_acceptance_rate =
                decodeMTPPolicyDouble(wire, 20u);
            return policy;
        }

        /** @brief Mix one scalar into an allocation-free workload fingerprint. */
        std::uint64_t mixInferenceWorkloadScalar(
            std::uint64_t hash,
            std::uint64_t value) noexcept
        {
            constexpr std::uint64_t kPrime = 1099511628211ull;
            for (std::size_t byte = 0; byte < sizeof(value); ++byte)
            {
                hash ^= value & 0xffu;
                hash *= kPrime;
                value >>= 8u;
            }
            return hash;
        }

        /**
         * @brief Fail-closed transport guard for immutable residency policies.
         *
         * `MoEOverlayResidencyAuthority::beginApply()` returns
         * `StaticNoMovement` before consulting transport. Supplying this guard
         * lets model setup execute and record that proof without allocating any
         * inactive expert slots or background streams. A non-empty wave under
         * a static policy is an invariant violation and fails explicitly.
         */
        class StaticMoEOverlayTransportGuard final
            : public IMoEOverlayResidencyTransport
        {
        public:
            /** @brief Reject the impossible physical stage for a static policy. */
            MoEOverlayResidencyStageStart beginStage(
                const MoEOverlayResidencyTransaction &) override
            {
                return {
                    .status =
                        MoEOverlayResidencyStageStartStatus::Failed,
                    .error =
                        "Static ExpertOverlay attempted to start physical migration",
                };
            }
        };

        /**
         * @brief Publish proof of each process-local installed initial expert bank.
         *
         * The participant registry exposes selections only after immutable
         * prepared gate/up/down banks exist. Emitting the historical placement
         * metric here preserves one CSV contract across the legacy sliced
         * loader and the production ExpertOverlay full-bank preparation path,
         * without claiming that a desired owner map was physically installed.
         *
         * @param registry Canonical process-local physical bank authority.
         * @param owner_order Frozen owner-order policy used to build the epoch.
         * @param main_layer_count Number of ordinary decoder layers. Any
         *        subsequently indexed bank is an MTP predictor layer, not an
         *        out-of-range main graph layer.
         */
        void recordInitialMoEOverlayBankSelections(
            const MoEOverlayParticipantResidencyRegistry &registry,
            RoutedExpertOwnerOrder owner_order,
            int main_layer_count)
        {
            if (!PerfStatsCollector::isDomainEnabled("moe_placement"))
                return;
            if (main_layer_count <= 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay bank evidence requires a positive main-layer count");
            }

            for (const auto &selection :
                 registry.initialBankExpertSelections())
            {
                const bool mtp_predictor_layer =
                    selection.layer_idx >= main_layer_count;
                PerfStatsCollector::addCounter(
                    "moe_placement",
                    "routed_expert_weight_selection",
                    static_cast<double>(selection.expert_ids.size()),
                    "model_setup",
                    selection.device.to_string(),
                    {
                        {"layer", std::to_string(selection.layer_idx)},
                        {"owner_order",
                         routedExpertOwnerOrderToString(owner_order)},
                        {"participant",
                         std::to_string(selection.participant_id)},
                        {"tier", std::to_string(selection.tier_idx)},
                        {"tier_participant_count",
                         std::to_string(selection.tier_participant_count)},
                        {"selection_layout",
                         routed_expert_ownership::
                                 expertIdsFormContiguousSpan(
                                     selection.expert_ids)
                             ? "contiguous"
                             : "noncontiguous"},
                        {"frozen_owner_map_match",
                         selection.matches_frozen_owner_map
                             ? "true"
                             : "false"},
                        {"publication", "initial_epoch_bank"},
                        {"layer_role",
                         mtp_predictor_layer
                             ? "mtp_predictor"
                             : "main_transformer"},
                    });
            }
        }

        /**
         * @brief Describe the CPU backend using the ISA that dispatch will use.
         *
         * An AVX-512 build may intentionally run its AVX2 implementation under
         * `LLAMINAR_ISA_LEVEL=avx2`.  Reporting the compile-time maximum here
         * made benchmark matrices claim AVX-512 while measuring AVX2.
         */
        std::string cpuBackendDescription()
        {
            return std::string("CPU (oneDNN/") +
                   isaLevelName(activeISALevel()) + ")";
        }

        bool requiresOverlayMPIWorld(const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan)
        {
            if (!plan || !plan->usesExpertOverlayAuthority())
                return false;

            for (const auto &domain : plan->domains)
            {
                if (domain.world_ranks.empty() && domain.owner_rank < 0)
                    return true;
                if (domain.owner_rank > 0)
                    return true;
                if (domain.world_ranks.size() > 1)
                    return true;
                if (domain.scope == ExecutionDomainScope::NODE_LOCAL && domain.participants.size() > 1)
                    return true;
            }
            return false;
        }

        /** @brief Decide whether initialization owns a world-rank protocol. */
        bool requiresMultiRankMPI(const OrchestrationConfig &config)
        {
            return requiresNamedDomainGlobalGraph(config) ||
                   config.pp_degree > 1 ||
                   config.tp_scope == TPScope::GLOBAL ||
                   config.tp_scope == TPScope::NODE_LOCAL ||
                   config.tp_scope == TPScope::HYBRID ||
                   requiresOverlayMPIWorld(config.moe_routed_expert_plan);
        }

        /**
         * @brief Name the exact physical-transfer profile reuse class.
         *
         * The routed-weight identity supplies participant devices, integer
         * priorities, placement policy, and capacity-affecting graph geometry.
         * The catalog identity additionally authenticates every projection
         * format, shape, and source-weight class. Length-prefixing preserves a
         * collision-free structural identity without treating it as a model
         * checksum.
         */
        std::string expertOverlayMigrationProfileReuseIdentity(
            const std::string &routed_weight_identity,
            const MoEOverlayEconomyCalibrationLayerCatalog &catalog)
        {
            if (routed_weight_identity.empty() ||
                catalog.identity().empty())
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration-profile reuse requires exact routed-weight and layer-catalog identities");
            }
            std::ostringstream out;
            out << "expert-overlay-migration-profile-reuse-v1/"
                << routed_weight_identity.size() << ':'
                << routed_weight_identity << '/'
                << catalog.identity().size() << ':'
                << catalog.identity();
            return out.str();
        }

        bool samplingParamsEqual(const SamplingParams &a, const SamplingParams &b)
        {
            return a.temperature == b.temperature &&
                   a.top_k == b.top_k &&
                   a.top_p == b.top_p &&
                   a.seed == b.seed &&
                   a.presence_penalty == b.presence_penalty &&
                   a.frequency_penalty == b.frequency_penalty &&
                   a.repetition_penalty == b.repetition_penalty &&
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

        /**
         * @brief Sample CPU logits with the serial/MTP output-position contract.
         * @param sampler CPU-owned distribution, penalty history and unseeded RNG.
         * @param logits Complete host vocabulary row from the live runner.
         * @param vocab_size Number of logits in that row.
         * @param params Immutable sampling policy for the admitted request.
         * @param logical_position Cached-token boundary producing the next output.
         * @return Token selected with the same seeded draw as an MTP target row.
         * @throws std::invalid_argument if no logical output boundary is supplied.
         *
         * Keep greedy and unseeded behavior unchanged. Seeded draws must not
         * advance mt19937: speculation can sample a future bonus before serial
         * decode reaches it. The existing shared MTP distribution primitive
         * preserves the arithmetic order without adding another CDF algorithm.
         */
        int sampleHostLogitsAtLogicalPosition(
            Sampler &sampler, const float *logits, size_t vocab_size,
            const SamplingParams &params, int logical_position)
        {
            if (logical_position < 0)
                throw std::invalid_argument("CPU sampling requires a non-negative logical position");
            if (params.is_greedy() || params.seed == 0)
                return sampler.sample(logits, vocab_size, params);
            return sampleMTPDistributionWithThreshold(
                sampler.compute_distribution(logits, vocab_size, params),
                mtpSpecStochasticThresholdForPosition(
                    params, sampler, logical_position, MTPSpecStochasticDrawPurpose::Sample));
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

        /**
         * @brief Resolve one immutable seed for a request-batched stochastic lane.
         *
         * A non-zero API seed is preserved exactly so the same request sees the
         * same position-keyed draws whether it is served alone or in a batch.
         * Seed zero retains its documented non-deterministic behavior, but the
         * entropy is consumed once at request creation. GPU kernels can then own
         * every mutable draw position without consulting or advancing a host
         * `std::mt19937` stream on each decode transaction.
         */
        uint64_t resolveRequestBatchedStochasticSeed(
            const SamplingParams &params,
            size_t request_index)
        {
            if (params.seed != 0)
                return static_cast<uint64_t>(params.seed);

            std::random_device entropy;
            uint64_t seed =
                (static_cast<uint64_t>(entropy()) << 32) ^
                static_cast<uint64_t>(entropy());
            seed ^= static_cast<uint64_t>(
                        std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count());
            seed ^= sampling_math::splitmix64(
                static_cast<uint64_t>(request_index) + 1ull);
            seed = sampling_math::splitmix64(seed);
            return seed != 0 ? seed : 0xD1B54A32D192ED03ull;
        }

        bool traceChatGeneratedTokensEnabled()
        {
            return debugEnv().runtime_debug.trace_generated_tokens;
        }

        bool prefixCacheTraceEnabled()
        {
            const char *value = DebugEnv::envValue("LLAMINAR_PREFIX_CACHE_TRACE");
            return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
        }

        /**
         * @brief Format existing immutable GDN diagnostic protocol metadata.
         * @param layer Complete copied probe, never a live state owner.
         * @return Storage-format id and every exact CPU phase word, without payload reads.
         */
        std::string gdnDiagnosticStorageSuffix(const PrefixGDNLayerProbe &layer)
        {
            std::ostringstream out;
            out << "/storage=" << static_cast<unsigned>(layer.recurrence_storage)
                << "/epoch_phase_words=[";
            for (size_t head = 0; head < layer.recurrence_epoch_phase_words.size(); ++head)
            {
                if (head > 0) out << ',';
                out << layer.recurrence_epoch_phase_words[head];
            }
            out << ']';
            return out.str();
        }

        std::string summarizePrefixProbeForTrace(const PrefixRuntimeStateSnapshot &probe)
        {
            /*
             * Fold every available persistent-state hash into compact cache
             * and recurrent-state digests. Printing only the first few layers
             * is useful for a human-readable preview, but it can hide a
             * restore defect in a later layer. These aggregate values make a
             * whole-model fresh-prefill versus restored-prefix comparison
             * possible without dumping or retaining any tensor payload.
             */
            constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
            constexpr uint64_t kFnvPrime = 1099511628211ull;
            auto fold_u64 = [](uint64_t &digest, uint64_t value)
            {
                for (unsigned byte = 0; byte < sizeof(value); ++byte)
                {
                    digest ^= value & 0xffull;
                    digest *= kFnvPrime;
                    value >>= 8;
                }
            };

            uint64_t kv_digest = kFnvOffsetBasis;
            uint64_t mtp_kv_digest = kFnvOffsetBasis;
            uint64_t gdn_digest = kFnvOffsetBasis;
            bool have_kv_digest = false;
            bool have_mtp_kv_digest = false;
            bool have_gdn_digest = false;
            auto fold_kv_caches =
                [&](const std::vector<PrefixKVCacheProbe> &caches,
                    uint64_t *digest,
                    bool *have_digest)
            {
                for (const PrefixKVCacheProbe &cache : caches)
                {
                    for (const PrefixKVLayerProbe &layer : cache.layers)
                    {
                        fold_u64(
                            *digest,
                            static_cast<uint64_t>(layer.global_layer));
                        fold_u64(
                            *digest,
                            static_cast<uint64_t>(layer.cached_tokens));
                        fold_u64(
                            *digest,
                            static_cast<uint64_t>(layer.ring_head));
                        if (layer.payload_hash_available)
                        {
                            fold_u64(*digest, layer.k_payload_hash);
                            fold_u64(*digest, layer.v_payload_hash);
                            *have_digest = true;
                        }
                        if (layer.leading_segment_hash_available)
                        {
                            fold_u64(*digest, layer.leading_k_payload_hash);
                            fold_u64(*digest, layer.leading_v_payload_hash);
                            *have_digest = true;
                        }
                        if (layer.trailing_segment_hash_available)
                        {
                            fold_u64(*digest, layer.trailing_k_payload_hash);
                            fold_u64(*digest, layer.trailing_v_payload_hash);
                            *have_digest = true;
                        }
                        for (const PrefixKVSegmentProbe &segment :
                             layer.segments)
                        {
                            if (!segment.hash_available)
                                continue;
                            fold_u64(*digest, segment.k_payload_hash);
                            fold_u64(*digest, segment.v_payload_hash);
                            *have_digest = true;
                        }
                    }
                }
            };
            fold_kv_caches(
                probe.kv_caches,
                &kv_digest,
                &have_kv_digest);
            fold_kv_caches(
                probe.mtp_kv_caches,
                &mtp_kv_digest,
                &have_mtp_kv_digest);
            for (const PrefixGDNLayerProbe &layer : probe.gdn_layers)
            {
                fold_u64(gdn_digest, static_cast<uint64_t>(layer.global_layer));
                fold_u64(gdn_digest, static_cast<uint64_t>(layer.recurrence_layout));
                fold_u64(gdn_digest, static_cast<uint64_t>(layer.recurrence_storage));
                fold_u64(gdn_digest, layer.recurrence_epoch_phase_words.size());
                for (uint32_t phase_word : layer.recurrence_epoch_phase_words)
                    fold_u64(gdn_digest, phase_word);
                fold_u64(gdn_digest, layer.recurrence_hash);
                fold_u64(gdn_digest, layer.conv_hash);
                if (layer.device_state_hash_available)
                {
                    fold_u64(gdn_digest, layer.recurrence_device_hash);
                    fold_u64(gdn_digest, layer.conv_device_hash);
                    have_gdn_digest = true;
                }
                if (layer.local_device_state_hash_available)
                {
                    fold_u64(gdn_digest, layer.recurrence_local_device_hash);
                    fold_u64(gdn_digest, layer.conv_local_device_hash);
                    have_gdn_digest = true;
                }
            }

            std::ostringstream oss;
            oss << "pos=" << probe.current_position
                << " kv_tokens=" << probe.totalCachedTokens()
                << " gdn_layers=" << probe.gdn_layers.size();
            if (have_kv_digest)
                oss << " kv_digest=" << kv_digest;
            if (have_mtp_kv_digest)
                oss << " mtp_kv_tokens=" << probe.totalMTPCachedTokens()
                    << " mtp_kv_digest=" << mtp_kv_digest;
            if (have_gdn_digest)
                oss << " gdn_digest=" << gdn_digest;
            if (probe.terminal_hidden_hash_available)
                oss << " terminal_hidden_bytes="
                    << probe.terminal_hidden_bytes
                    << " terminal_hidden_hash="
                    << probe.terminal_hidden_hash;
            if (probe.terminal_logits_hash_available)
                oss << " terminal_logits_bytes="
                    << probe.terminal_logits_bytes
                    << " terminal_logits_hash="
                    << probe.terminal_logits_hash;
            const size_t kv_limit =
                probe.kv_caches.empty()
                    ? 0
                    : std::min<size_t>(probe.kv_caches.front().layers.size(), 4);
            for (size_t i = 0; i < kv_limit; ++i)
            {
                const auto &layer = probe.kv_caches.front().layers[i];
                oss << " KV" << layer.global_layer
                    << "/n=" << layer.cached_tokens;
                if (layer.leading_segment_hash_available)
                    oss << "/lead=" << layer.leading_k_payload_hash
                        << ":" << layer.leading_v_payload_hash;
                if (layer.trailing_segment_hash_available)
                    oss << "/tail=" << layer.trailing_k_payload_hash
                        << ":" << layer.trailing_v_payload_hash;
            }
            const size_t gdn_limit = std::min<size_t>(probe.gdn_layers.size(), 4);
            for (size_t i = 0; i < gdn_limit; ++i)
            {
                const auto &layer = probe.gdn_layers[i];
                oss << " L" << layer.global_layer
                    << "/layout=" << gdnRecurrenceStateLayoutName(layer.recurrence_layout)
                    << gdnDiagnosticStorageSuffix(layer)
                    << "/rh=" << layer.recurrence_hash << "/ch=" << layer.conv_hash;
                if (layer.device_state_hash_available)
                    oss << "/rdh=" << layer.recurrence_device_hash
                        << "/cdh=" << layer.conv_device_hash;
                if (layer.local_device_state_hash_available)
                    oss << "/rlh=" << layer.recurrence_local_device_hash
                        << "/clh=" << layer.conv_local_device_hash;
            }
            return oss.str();
        }

        const char *perfBool(bool value)
        {
            return value ? "true" : "false";
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
         * @brief Abort a live MPI command world if rank zero fails after publishing a command.
         *
         * A coordinated command is a distributed transaction, not a local
         * function call.  Once rank zero has published its tag, a failure on
         * the root can leave a worker inside a sparse-expert collective or an
         * end-of-command fence while root has already returned to its caller.
         * That state has no legal next command.  This scope therefore treats
         * an uncompleted root command as fatal for the live communicator.
         *
         * Unit tests use scripted MPI contexts with `MPI_COMM_NULL`.  Those
         * contexts intentionally do not abort a process, which lets tests
         * inspect the ordinary returned error while production communicator
         * failures remain fail-fast.
         */
        class ScopedMPIRootCommandFailureAbort
        {
        public:
            /**
             * @brief Construct a root-command completion guard.
             * @param mpi_ctx Context owning the command communicator.
             * @param root_command True only on rank zero of a multi-rank
             *        coordinated command.
             * @param command Stable diagnostic name for the published command.
             * @param failure_detail Root-owned result detail, whose lifetime
             *        must extend through this guard's destruction.
             */
            ScopedMPIRootCommandFailureAbort(
                const IMPIContext *mpi_ctx,
                bool root_command,
                const char *command,
                const std::string *failure_detail = nullptr)
                : mpi_ctx_(mpi_ctx),
                  root_command_(root_command),
                  command_(command ? command : "unknown"),
                  failure_detail_(failure_detail)
            {
            }

            /**
             * @brief Abort the live world when a published command did not complete.
             */
            ~ScopedMPIRootCommandFailureAbort() noexcept
            {
                if (published_ && !completed_)
                    abortLiveCommandWorld();
            }

            ScopedMPIRootCommandFailureAbort(const ScopedMPIRootCommandFailureAbort &) = delete;
            ScopedMPIRootCommandFailureAbort &operator=(const ScopedMPIRootCommandFailureAbort &) = delete;

            /**
             * @brief Mark that rank zero has made the command visible to workers.
             *
             * The guard remains inert until this point so a validation failure
             * before the command tag is broadcast stays a normal local error.
             */
            void markPublished() noexcept
            {
                published_ = root_command_;
            }

            /**
             * @brief Mark successful completion of the distributed command.
             */
            void markCompleted() noexcept
            {
                completed_ = true;
            }

        private:
            /**
             * @brief Abort the command communicator without throwing from a destructor.
             *
             * A null communicator denotes the scripted unit-test transport;
             * there is no live peer to release in that environment.  A live
             * communicator must never return to an ambiguous command stream,
             * so an unexpected MPI_Abort return is followed by std::abort().
             */
            void abortLiveCommandWorld() const noexcept
            {
                if (!mpi_ctx_)
                    return;

                const MPI_Comm communicator = mpi_ctx_->communicator();
                if (communicator == MPI_COMM_NULL)
                    return;

                LOG_ERROR(
                    "[MPI] rank " << mpi_ctx_->rank()
                    << " failed coordinated " << command_
                    << " after publishing its command"
                    << (failure_detail_ && !failure_detail_->empty()
                            ? ": " + *failure_detail_
                            : ": no failure detail was recorded")
                    << "; aborting the MPI communicator because collective "
                       "order is indeterminate");
                const int abort_status = MPI_Abort(communicator, EXIT_FAILURE);
                LOG_ERROR("[MPI] MPI_Abort unexpectedly returned with status "
                          << abort_status << " for coordinated " << command_
                          << "; terminating root directly");
                std::abort();
            }

            const IMPIContext *mpi_ctx_{nullptr};
            bool root_command_{false};
            bool published_{false};
            bool completed_{false};
            const char *command_{"unknown"};
            /// Root-owned result detail that outlives this guard.
            const std::string *failure_detail_{nullptr};
        };

        /**
         * @brief Own one continuation-authoritative ExpertOverlay command terminal.
         *
         * The MPI command tag wakes a remote transaction follower before model
         * execution starts. Every return after that point must therefore publish
         * either Complete or Abort on the fixed control lane. This guard makes
         * that terminal ownership independent of the many validation exits in
         * serial and speculative decode.
         */
        class ScopedMoEOverlayRootCommand
        {
        public:
            /** @brief Bind the optional continuation-side coordinator. */
            explicit ScopedMoEOverlayRootCommand(
                std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
                    coordinator)
                : coordinator_(std::move(coordinator))
            {
            }

            /** @brief Abort an admitted command that did not publish Complete. */
            ~ScopedMoEOverlayRootCommand() noexcept
            {
                if (!coordinator_ || !begun_ || completed_)
                    return;
                std::string error;
                if (!coordinator_->abortCommand(
                        coordinator_->currentPlacementEpoch(),
                        /*error_code=*/1,
                        &error))
                {
                    LOG_ERROR(
                        "[OrchestrationRunner] ExpertOverlay decode command "
                        "could not publish its abort terminal: "
                        << error);
                }
            }

            ScopedMoEOverlayRootCommand(
                const ScopedMoEOverlayRootCommand &) = delete;
            ScopedMoEOverlayRootCommand &operator=(
                const ScopedMoEOverlayRootCommand &) = delete;

            /**
             * @brief Open the exact outer command on every remote follower lane.
             * @param world_rank Actual original overlay communicator rank.
             * @param command Root-owned request, command, and placement-floor identity.
             * @param error Optional stable failure diagnostic.
             * @return True for an inert local scope or a successfully opened lane.
             */
            bool begin(
                int world_rank,
                const MoEOverlayInferenceCommandIdentity &command,
                std::string *error)
            {
                if (!coordinator_)
                    return true;
                if (begun_)
                {
                    if (error)
                        *error = "ExpertOverlay decode command began twice";
                    return false;
                }
                begun_ = coordinator_->beginRootCommand(world_rank, command, error);
                return begun_;
            }

            /**
             * @brief Admit one serial or speculative graph sequence.
             * @param draft_depth Zero for serial, positive for MTP.
             * @param error Optional stable failure diagnostic.
             */
            bool beginGraphSequence(int draft_depth, std::string *error)
            {
                return !coordinator_ ||
                       (begun_ && coordinator_->beginGraphSequence(
                                      draft_depth, error));
            }

            /**
             * @brief Admit a restored-prefix suffix after any completed prefill.
             * @param error Optional stable lifecycle rejection diagnostic.
             * @return True for an inert local scope or an admitted decode bridge.
             */
            bool beginRestoredPrefixDecodeSequence(std::string *error)
            {
                return !coordinator_ ||
                       (begun_ && coordinator_->beginRestoredPrefixDecodeSequence(error));
            }

            /**
             * @brief Retire one completed sequence before another row uses this command.
             * @param error Optional stable failure diagnostic.
             * @return True for an inert local scope or a retired graph sequence.
             *
             * A forced output may first owe a previously emitted condition row.
             * Both rows share the outer command, but each has its own exact
             * participant graph schedule and cannot overlap the other's sequence.
             */
            bool retireGraphSequence(std::string *error)
            {
                return !coordinator_ ||
                       (begun_ && coordinator_->retireCompletedGraphSequence(error));
            }

            /**
             * @brief Publish the successful terminal after all graph returns.
             * @param error Optional stable failure diagnostic.
             */
            bool complete(std::string *error)
            {
                if (!coordinator_)
                {
                    completed_ = true;
                    return true;
                }
                if (!begun_)
                {
                    if (error)
                        *error = "ExpertOverlay decode command was never opened";
                    return false;
                }
                completed_ = coordinator_->completeCommand(
                    coordinator_->currentPlacementEpoch(), error);
                return completed_;
            }

            /** @return Whether this scope owns a remote follower command. */
            [[nodiscard]] bool active() const noexcept
            {
                return coordinator_ != nullptr;
            }

        private:
            std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
                coordinator_;
            bool begun_ = false;
            bool completed_ = false;
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
            /**
             * @brief Begin one request-scoped verifier synchronization deferral.
             *
             * @param runner Runner that owns the deferred verifier stream edge.
             * @param enabled Whether this transaction needs deferred handoff.
             */
            ScopedMTPAllPositionVerifierSyncDeferral(
                IInferenceRunner *runner,
                bool enabled)
                : runner_(runner),
                  enabled_(enabled && runner != nullptr)
            {
                if (enabled_)
                    runner_->setMTPAllPositionVerifierSyncDeferralEnabled(true);
            }

            /** @brief Retire any still-active deferral before leaving scope. */
            ~ScopedMTPAllPositionVerifierSyncDeferral()
            {
                close();
            }

            /**
             * @brief Publish the deferred consumer edge and reopen writer admission.
             *
             * Failure rollback must call this before restoring a prefix
             * checkpoint. Restoration is an exclusive logical-state writer;
             * attempting it while this transaction still owns a reader edge
             * would wait on the transaction performing the rollback.
             */
            void close() noexcept
            {
                if (!enabled_)
                    return;
                runner_->setMTPAllPositionVerifierSyncDeferralEnabled(false);
                enabled_ = false;
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
                close();
            }

            bool installed() const { return installed_; }

            /**
             * @brief Retire the transient verifier-row plan before rollback.
             *
             * Explicit closure lets failure paths restore a checkpoint only
             * after graph metadata no longer names the failed transaction.
             */
            void close() noexcept
            {
                if (!runner_ || !installed_)
                    return;
                runner_->clearMTPSpecVerifierInputPlan();
                installed_ = false;
            }

            ScopedMTPSpecVerifierInputPlan(
                const ScopedMTPSpecVerifierInputPlan &) = delete;
            ScopedMTPSpecVerifierInputPlan &operator=(
                const ScopedMTPSpecVerifierInputPlan &) = delete;

        private:
            IInferenceRunner *runner_ = nullptr;
            bool installed_ = false;
        };

        /**
         * @brief Own one complete row-indexed all-position verifier binding.
         *
         * A grouped stochastic verifier has three coupled pieces of transient
         * graph state: the compact-row metadata plan, row-indexed logits, and
         * all-position logits.  The target-distribution builder consumes those
         * logits after the forward graph returns, and the compact outcome
         * reducer consumes the resulting device distributions after that.
         * Consequently, none of the three bindings may be retired at the end
         * of the forward call itself.
         *
         * This scope makes that producer/consumer lifetime structural.  It
         * installs all three bindings together and keeps them alive until the
         * caller has enqueued the compact outcome consumer.  Destruction is a
         * final safety net: cleanup failure terminates the process because
         * continuing with a partially active verifier transaction could make a
         * later captured replay consume stale row metadata or stale logits.
         */
        class ScopedMTPAllPositionVerifierTransaction
        {
        public:
            ScopedMTPAllPositionVerifierTransaction(
                IInferenceRunner *runner,
                const MTPSpecDecodeVerifierInputPlan &plan)
                : runner_(runner)
            {
                if (!runner_)
                {
                    error_ = "Grouped MTP verifier transaction has no runner";
                    return;
                }
                if (!runner_->setMTPSpecVerifierInputPlan(plan))
                {
                    error_ = "Grouped MTP verifier could not install row metadata plan";
                    return;
                }
                plan_installed_ = true;

                if (!runner_->setComputeRowIndexedAllPositionLogits(
                        true,
                        plan.compact_logit_row_count))
                {
                    error_ = "Grouped MTP verifier could not enable row-indexed logits";
                    cleanupOrTerminate();
                    return;
                }
                row_indexed_enabled_ = true;

                if (!runner_->setComputeAllPositionLogits(true))
                {
                    error_ = "Grouped MTP verifier could not enable all-position logits";
                    cleanupOrTerminate();
                    return;
                }
                all_position_enabled_ = true;
                ready_ = true;
            }

            ~ScopedMTPAllPositionVerifierTransaction() noexcept
            {
                cleanupOrTerminate();
            }

            bool ready() const { return ready_; }
            const std::string &error() const { return error_; }

            /**
             * @brief Retire the verifier bindings after its final device consumer.
             *
             * @param error Receives a precise cleanup failure if retirement
             *              cannot be completed.
             * @return true when every transient binding has been retired.
             */
            bool close(std::string *error = nullptr)
            {
                if (closed_)
                    return true;

                std::string cleanup_error;
                bool ok = true;
                if (all_position_enabled_)
                {
                    if (runner_->setComputeAllPositionLogits(false))
                    {
                        all_position_enabled_ = false;
                    }
                    else
                    {
                        ok = false;
                        cleanup_error =
                            "Grouped MTP verifier could not disable all-position logits";
                    }
                }
                if (row_indexed_enabled_)
                {
                    if (runner_->setComputeRowIndexedAllPositionLogits(false, 0))
                    {
                        row_indexed_enabled_ = false;
                    }
                    else
                    {
                        ok = false;
                        if (!cleanup_error.empty())
                            cleanup_error += "; ";
                        cleanup_error +=
                            "Grouped MTP verifier could not disable row-indexed logits";
                    }
                }
                if (plan_installed_)
                {
                    runner_->clearMTPSpecVerifierInputPlan();
                    plan_installed_ = false;
                }

                closed_ = ok;
                ready_ = false;
                if (!ok && error)
                    *error = std::move(cleanup_error);
                return ok;
            }

            ScopedMTPAllPositionVerifierTransaction(
                const ScopedMTPAllPositionVerifierTransaction &) = delete;
            ScopedMTPAllPositionVerifierTransaction &operator=(
                const ScopedMTPAllPositionVerifierTransaction &) = delete;

        private:
            void cleanupOrTerminate() noexcept
            {
                std::string cleanup_error;
                if (close(&cleanup_error))
                    return;

                LOG_ERROR("[OrchestrationRunner] Fatal grouped MTP verifier "
                          "transaction cleanup failure: "
                          << cleanup_error);
                std::terminate();
            }

            IInferenceRunner *runner_ = nullptr;
            bool plan_installed_ = false;
            bool row_indexed_enabled_ = false;
            bool all_position_enabled_ = false;
            bool ready_ = false;
            bool closed_ = false;
            std::string error_;
        };

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

            return expectedShiftedMTPTokens(snapshot.cached_tokens,snapshot.mtp_shifted_origin);
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
            stamp.shifted_origin=snapshot.mtp_shifted_origin;
            stamp.position = snapshot.cached_tokens;
            stamp.has_terminal_hidden = has_terminal_hidden;
            stamp.has_terminal_logits = has_terminal_logits;
            stamp.has_ready_token = has_ready_token;
            stamp.provenance = snapshot.provenance;
            stamp.label = std::move(label);
            return stamp;
        }

        std::shared_ptr<const MoEExpertOverlayExecutionPlan> resolveOverlayExecutionPlanForRunner(
            const std::shared_ptr<const MoERoutedExpertPlacementPlan> &plan,
            const std::shared_ptr<IMPIContext> &mpi_ctx)
        {
            if (!plan || !plan->usesExpertOverlayAuthority() || !mpi_ctx)
                return nullptr;

            return std::make_shared<MoEExpertOverlayExecutionPlan>(
                resolveMoEExpertOverlayExecutionPlan(
                    plan,
                    MoEExpertOverlayExecutionPlanResolverOptions{
                        .current_world_rank = mpi_ctx->rank(),
                        .world_size = mpi_ctx->world_size(),
                    }));
        }

        /**
         * @brief Resolve the request publisher from the complete admitted graph family.
         * @param flat_plan Placement of a non-pipeline ExpertOverlay runner.
         * @param pipeline Independent immutable placements of every PP stage.
         * @param context Communicator that owns this request lifecycle.
         * @return The common continuation root, or nullopt when no overlay exists.
         * @throws std::logic_error for missing, mixed or conflicting authorities.
         *
         * PP deliberately has no model-wide routed placement. Its stage plans
         * must nevertheless share one request publisher before any sparse graph
         * can execute. Inspect immutable topology at admission/reset, never
         * replace the stage placements or derive identity from graph lifetimes.
         */
        std::optional<int> resolveMoEOverlayRequestAuthorityRank(
            const std::shared_ptr<const MoERoutedExpertPlacementPlan> &flat_plan,
            const std::shared_ptr<const MoEPipelinePreparedPlan> &pipeline,
            const std::shared_ptr<IMPIContext> &context)
        {
            const bool flat_overlay = flat_plan && flat_plan->usesExpertOverlayAuthority();
            if (!flat_overlay && !pipeline)
                return std::nullopt;
            if (!context || (flat_overlay && pipeline))
                throw std::logic_error("MoE request publication requires one unambiguous admitted graph family and communicator");

            std::optional<int> authority_rank;
            const auto require_plan = [&](const auto &placement)
            {
                const auto execution = resolveOverlayExecutionPlanForRunner(placement, context);
                if (!execution || (authority_rank && *authority_rank != execution->continuation_root_rank))
                    throw std::logic_error("MoE request publication requires the same continuation authority in every stage");
                authority_rank = execution->continuation_root_rank;
            };
            if (pipeline)
            {
                if (const auto &global = pipeline->admission()->globalPipeline(); global)
                {
                    if (global->world_size != context->world_size())
                        throw std::logic_error("Pipeline request publication crossed its original global communicator");
                    return global->embeddingPublisherRank();
                }
                if (pipeline->stages().empty())
                    throw std::logic_error("MoE pipeline request publication has no admitted stages");
                for (const auto &stage : pipeline->stages())
                    require_plan(stage.placement);
            }
            else
                require_plan(flat_plan);
            return authority_rank;
        }

        /**
         * @brief Resolve every local device whose prepared weights one rank owns.
         *
         * An ExpertOverlay follower has a synthetic CPU primary because it does
         * not build the continuation graph, yet it can own several CUDA/ROCm
         * expert participants.  Reuse certification must inspect those actual
         * endpoints, not the synthetic primary.  Ordinary LocalTP resolves the
         * same authority set from the rank plan. PP includes every inner TP
         * participant, not merely the primary device of each pipeline stage.
         * Frozen-zero routed-only endpoints retain their runtime/storage life,
         * but cannot be required to own prepared weight records they never load.
         */
        std::vector<DeviceId> rankLocalPreparedWeightAuthorityDevices(
            const RankExecutionPlan &rank_plan,
            const std::shared_ptr<const MoERoutedExpertPlacementPlan> &overlay_plan,
            const std::shared_ptr<IMPIContext> &mpi_ctx)
        {
            std::vector<DeviceId> devices;
            const auto append_unique = [&devices](DeviceId device)
            {
                if (!device.is_valid() ||
                    std::find(devices.begin(), devices.end(), device) !=
                        devices.end())
                {
                    return;
                }
                devices.push_back(device);
            };

            const auto overlay_execution =
                resolveOverlayExecutionPlanForRunner(overlay_plan, mpi_ctx);
            if (overlay_execution)
            {
                return admittedOverlayPreparedWeightDevices(*overlay_plan, *overlay_execution);
            }

            if (rank_plan.usesLocalPP())
            {
                for (const auto &device : rank_plan.local_pp_devices)
                    append_unique(device.toLocalDeviceId());
                for (const auto &stage : rank_plan.local_pp_stage_tp_info)
                    for (const auto &device : stage.devices)
                        append_unique(device.toLocalDeviceId());
                return devices;
            }

            if (rank_plan.usesLocalTP())
            {
                for (const auto &device : rank_plan.local_tp_devices)
                    append_unique(device.toLocalDeviceId());
                return devices;
            }

            append_unique(rank_plan.primary_device.toLocalDeviceId());
            return devices;
        }

        /**
         * @brief Resolve every rank-local GPU with execution-lifetime owners.
         *
         * ExpertOverlay followers may have a synthetic CPU primary while their
         * participant graphs live on arbitrary CUDA/ROCm endpoints. Ordinary
         * LocalTP, LocalPP, and PP-with-inner-TP plans express the same ownership
         * through different plan fields. Reclamation must union those typed
         * sources rather than infer a fixed topology from rank or tier names.
         */
        std::vector<DeviceId> rankLocalGPUExecutionDevices(
            const RankExecutionPlan &rank_plan,
            const std::shared_ptr<const MoERoutedExpertPlacementPlan> &overlay_plan,
            const std::shared_ptr<IMPIContext> &mpi_ctx)
        {
            std::vector<DeviceId> devices;
            const auto append_unique = [&devices](DeviceId device)
            {
                if (!device.is_gpu() ||
                    std::find(devices.begin(), devices.end(), device) !=
                        devices.end())
                {
                    return;
                }
                devices.push_back(device);
            };

            if (const auto overlay_execution =
                    resolveOverlayExecutionPlanForRunner(
                        overlay_plan, mpi_ctx))
            {
                for (const DeviceId device :
                     overlay_execution->currentRankPlan().local_devices)
                {
                    append_unique(device);
                }
            }
            for (const auto &device : rank_plan.local_tp_devices)
                append_unique(device.toLocalDeviceId());
            for (const auto &device : rank_plan.local_pp_devices)
                append_unique(device.toLocalDeviceId());
            for (const auto &stage : rank_plan.local_pp_stage_tp_info)
            {
                for (const auto &device : stage.devices)
                    append_unique(device.toLocalDeviceId());
            }
            append_unique(rank_plan.primary_device.toLocalDeviceId());
            return devices;
        }

        /**
         * @brief Resolve the immutable coordinated-command authority.
         *
         * A heterogeneous overlay's continuation owner and a global pipeline's
         * vocabulary-head leader come from their resolved topology authorities.
         * A participant-local graph leaves the enclosing plan's choice intact.
         * @param plan Resolved expert-overlay topology, when configured.
         * @param mpi_ctx The command communicator used to validate the owner.
         * @param runner Constructed graph, if available at this lifecycle point.
         * @return One validated command/sampler/result owner for all ranks.
         * @throws std::logic_error if graph and overlay ownership conflict.
         */
        int coordinatedRootRankForRunner(
            const std::shared_ptr<const MoERoutedExpertPlacementPlan> &plan,
            const std::shared_ptr<IMPIContext> &mpi_ctx,
            const IInferenceRunner *runner = nullptr)
        {
            const auto execution =
                resolveOverlayExecutionPlanForRunner(plan, mpi_ctx);
            if (runner)
            {
                if (const auto owner = runner->requestAuthorityRank())
                {
                    if (!mpi_ctx || *owner < 0 || *owner >= mpi_ctx->world_size())
                        throw std::logic_error("Model graph declares an invalid request authority rank");
                    if (execution && execution->continuation_root_rank != *owner)
                        throw std::logic_error("Model graph and expert overlay declare different request authorities");
                    return *owner;
                }
            }
            return execution ? execution->continuation_root_rank : 0;
        }


        /**
         * @brief Compare the weight-affecting identity of two rank-local plans.
         *
         * Runtime shape and request policy (KV precision, sequence capacity,
         * fixed/adaptive MTP depth, and sampling) rebuild runner-owned state but
         * consume the same prepared weights. Device, layer, shard, and MTP
         * enablement change the physical prepared set and must match exactly.
         *
         * @return Empty on an exact reusable weight topology; otherwise a
         *         user-facing incompatibility diagnostic.
         */
        std::optional<std::string> retainedPreparedWeightPlanMismatch(
            const RankExecutionPlan &prepared,
            const RankExecutionPlan &current)
        {
            const auto same_local_pp_tp = [](const auto &lhs, const auto &rhs)
            {
                if (lhs.size() != rhs.size())
                    return false;
                for (std::size_t index = 0u; index < lhs.size(); ++index)
                {
                    if (lhs[index].devices != rhs[index].devices ||
                        lhs[index].tp_weights != rhs[index].tp_weights ||
                        lhs[index].tp_backend != rhs[index].tp_backend)
                    {
                        return false;
                    }
                }
                return true;
            };
            if (prepared.primary_device != current.primary_device)
            {
                return "Retained prepared weights belong to " +
                       prepared.primary_device.toString() +
                       " but the current plan targets " +
                       current.primary_device.toString();
            }
            if (prepared.first_layer != current.first_layer ||
                prepared.last_layer != current.last_layer ||
                prepared.has_embedding != current.has_embedding ||
                prepared.has_lm_head != current.has_lm_head)
            {
                return "Retained prepared weights have different layer or "
                       "global-weight ownership";
            }
            if (prepared.weight_shard.shard_index !=
                    current.weight_shard.shard_index ||
                prepared.weight_shard.total_shards !=
                    current.weight_shard.total_shards ||
                prepared.weight_shard.work_fraction !=
                    current.weight_shard.work_fraction)
            {
                return "Retained prepared weights have a different tensor "
                       "shard identity";
            }
            if (prepared.local_tp_devices != current.local_tp_devices ||
                prepared.local_tp_weights != current.local_tp_weights ||
                prepared.local_pp_devices != current.local_pp_devices ||
                prepared.local_pp_layer_boundaries !=
                    current.local_pp_layer_boundaries ||
                !same_local_pp_tp(
                    prepared.local_pp_stage_tp_info,
                    current.local_pp_stage_tp_info) ||
                prepared.global_tp_domain_id !=
                    current.global_tp_domain_id ||
                prepared.global_tp_rank_in_domain !=
                    current.global_tp_rank_in_domain ||
                prepared.global_tp_domain_size !=
                    current.global_tp_domain_size)
            {
                return "Retained prepared weights have a different rank-local "
                       "device topology";
            }
            const auto &prepared_mtp = prepared.runtime.mtp;
            const auto &current_mtp = current.runtime.mtp;
            if (!retainedMTPWeightAuthorityMatches(prepared_mtp, current_mtp))
            {
                return "Retained prepared weights have a different MTP "
                       "setup-capacity, predictor or terminal-head authority";
            }
            return std::nullopt;
        }
    }

    std::uint32_t completeMoEOverlayDeviceCommandCapacity(
        int num_layers,
        int num_experts,
        std::size_t participant_count);
} // namespace llaminar2
