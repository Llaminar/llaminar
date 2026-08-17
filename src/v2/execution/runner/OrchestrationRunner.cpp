/**
 * @file OrchestrationRunner.cpp
 * @brief Coordinates request lifecycle, graph execution, and rank-local inference.
 *
 * This implementation turns validated orchestration configuration into a
 * rank-local production execution plan, owns request lifecycle state, and
 * drives prefill, decode, MTP, prefix-cache, and dynamic MoE maintenance.
 * Rank zero publishes coordinated commands while non-root ranks execute the
 * same graph/collective schedule. Failure handling is deliberately strict:
 * once a worker can no longer participate in that schedule, it aborts the MPI
 * job rather than returning to a command loop that would leave a peer blocked
 * in an unmatched collective.
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
#include "../moe/MoEExpertOverlayAuthorityPlan.h"
#include "../moe/MoEExpertOwnerMap.h"
#include "../moe/MoEOverlayParticipantResidency.h"
#include "../moe/MoEOverlayParticipantMigration.h"
#include "../moe/MoEOverlayDistributedResidencyTransport.h"
#include "../moe/MoEOverlayMPIEconomyEvidenceExchange.h"
#include "../moe/MoEOverlayMPIHistogramPublisher.h"
#include "../moe/MoEOverlayMPIRemoteProjectionTransport.h"
#include "../moe/MoEOverlayMPIResidencyConsensus.h"
#include "../moe/MoEOverlayCapacityAdmission.h"
#include "../moe/MoEOverlayEconomyCalibrationController.h"
#include "../moe/MoEOverlayEconomyCertificationController.h"
#include "../moe/MoEOverlayLocalCapacityPlanner.h"
#include "../moe/MoEOverlayMigrationMeasurementExchange.h"
#include "../moe/MoEOverlayPhysicalResidencyFabric.h"
#include "../moe/MoEOverlayResidencyAuthority.h"
#include "../moe/MoEOverlayResidencyMaintenanceService.h"
#include "../moe/MoEOverlayInferenceInterferenceProbe.h"
#include "../moe/MoEOverlayTierMigrationTransport.h"
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
#include "../../loaders/GPUVramPreflight.h"
#include "../../loaders/MmapRegion.h"
#include "../../loaders/PreparedWeightStore.h"
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
#include "../../utils/CPUFeatures.h"
#include "../../utils/MPITopology.h"
#include "../../utils/NodeDetection.h"
#include "../../utils/NUMATopology.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/WeightLoadingProfiler.h"
#include "../local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../execution/moe/MoERoutedExpertPlacementPlan.h"
#include "../../execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "../../execution/moe/MoEExpertOwnerMap.h"
#include "../../execution/moe/MoEOverlayParticipantGraphRunner.h"
#include "../../execution/moe/MoEOverlayInferenceTransactionService.h"

#include <algorithm>
#include <array>
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
#include <print>
#include <random>
#include <stdexcept>
#include <sstream>
#include <type_traits>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace llaminar2
{
    namespace
    {
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

        /** @brief Build one complete live graph-geometry timing identity. */
        MoEOverlayInferenceWorkloadIdentity inferenceWorkloadIdentity(
            ExpertHistogramSource source,
            int real_rows,
            int execution_rows,
            int transaction_count,
            int speculative_depth,
            std::uint64_t schedule_fingerprint = 0) noexcept
        {
            std::uint64_t fingerprint = schedule_fingerprint;
            if (fingerprint == 0)
            {
                fingerprint = 14695981039346656037ull;
                fingerprint = mixInferenceWorkloadScalar(
                    fingerprint, static_cast<std::uint64_t>(source));
                fingerprint = mixInferenceWorkloadScalar(
                    fingerprint, static_cast<std::uint64_t>(real_rows));
                fingerprint = mixInferenceWorkloadScalar(
                    fingerprint, static_cast<std::uint64_t>(execution_rows));
                fingerprint = mixInferenceWorkloadScalar(
                    fingerprint,
                    static_cast<std::uint64_t>(transaction_count));
                fingerprint = mixInferenceWorkloadScalar(
                    fingerprint,
                    static_cast<std::uint64_t>(speculative_depth));
            }
            return {
                .source = source,
                .real_rows = real_rows,
                .execution_rows = execution_rows,
                .transaction_count = transaction_count,
                .speculative_depth = speculative_depth,
                .schedule_fingerprint = fingerprint,
            };
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

        std::string formatStochasticThreshold(float threshold)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(9) << threshold;
            return oss.str();
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
                    << "/rh=" << layer.recurrence_hash
                    << "/ch=" << layer.conv_hash;
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
             * @param command Root-owned request, command, and placement identity.
             * @param error Optional stable failure diagnostic.
             * @return True for an inert local scope or a successfully opened lane.
             */
            bool begin(
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
                begun_ = coordinator_->beginCommand(command, error);
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

        void synchronizeRunnerDevicesBeforeRelease(
            IInferenceRunner *runner,
            bool physical_backend_access_enabled)
        {
            if (!runner)
                return;

            if (auto *rank = dynamic_cast<IRankOrchestrator *>(runner))
            {
                rank->synchronizeDevices();
                return;
            }

            /*
             * Injected unit-test runners use GPU-shaped DeviceIds to exercise
             * orchestration policy but do not own CUDA/HIP resources. Resolving
             * BackendManager here would turn a CPU-only unit into physical GPU
             * work merely because its mock was destroyed.
             */
            if (!physical_backend_access_enabled)
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
                                          block.device_mtp_storage != nullptr ||
                                          block.device_mtp_allocation != nullptr;
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
         * @brief Resolve the immutable coordinated-command authority.
         *
         * A heterogeneous overlay's dense continuation owner may be any rank
         * selected by cluster inventory. All other orchestration retains the
         * established rank-zero command root.
         */
        int coordinatedRootRankForRunner(
            const std::shared_ptr<const MoERoutedExpertPlacementPlan> &plan,
            const std::shared_ptr<IMPIContext> &mpi_ctx)
        {
            const auto execution =
                resolveOverlayExecutionPlanForRunner(plan, mpi_ctx);
            return execution ? execution->continuation_root_rank : 0;
        }

        const RoutedExpertDomain *overlayDomainForName(
            const MoERoutedExpertPlacementPlan &plan,
            const std::string &domain_name)
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                                   [&](const RoutedExpertDomain &domain)
                                   {
                                       return domain.name == domain_name;
                                   });
            return it == plan.domains.end() ? nullptr : &(*it);
        }

        bool applyOverlayRankRoleToExecutionPlan(
            RankExecutionPlan &rank_plan,
            const MoERoutedExpertPlacementPlan &overlay_plan,
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
            if (base_domain->scope == ExecutionDomainScope::NODE_LOCAL)
            {
                if (overlay_plan.continuation_domain_spec.effectiveDensePolicy() !=
                    DenseParallelPolicy::TensorParallel)
                {
                    error = "MoE overlay NodeTP continuation domain '" +
                            base_domain_name +
                            "' requires dense_policy=tensor_parallel";
                    return false;
                }
                const bool participates_in_dense_domain =
                    std::any_of(
                        rank_plan.my_domains.begin(),
                        rank_plan.my_domains.end(),
                        [&](const TPDomainParticipation &participation)
                        {
                            return participation.domain_name ==
                                   base_domain_name;
                        });
                if (!participates_in_dense_domain ||
                    !rank_plan.usesGlobalTP() ||
                    rank_plan.global_tp_domain_size <= 1)
                {
                    error = "MoE overlay NodeTP continuation rank " +
                            std::to_string(rank_plan.rank) +
                            " did not retain its resolved cross-rank dense TP domain";
                    return false;
                }

                /*
                 * The named-domain planner already installed this rank's one
                 * exact CPU participant, shard index, collective domain, and
                 * NUMA-qualified primary device. Preserve those facts. The
                 * overlay role resolver decides that this rank builds a dense
                 * graph shard; it must not synthesize another local TP view.
                 */
                rank_plan.local_pp_devices.clear();
                rank_plan.local_pp_layer_boundaries.clear();
                rank_plan.local_pp_stage_tp_info.clear();
                return true;
            }

            rank_plan.local_pp_devices.clear();
            rank_plan.local_pp_layer_boundaries.clear();
            rank_plan.local_pp_stage_tp_info.clear();
            rank_plan.primary_device = base_domain->participants.front();
            rank_plan.local_tp_backend = base_domain->backend;
            rank_plan.local_tp_weights = base_domain->weights;
            rank_plan.weight_shard = {};

            if (base_domain->scope == ExecutionDomainScope::RANK_LOCAL)
            {
                rank_plan.tp_scope = TPScope::RANK_LOCAL;
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

        /**
         * @brief Compare the weight-affecting identity of two single-device plans.
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
            const auto has_parallel_weight_authorities = [](const auto &plan)
            {
                return plan.usesLocalTP() || plan.usesLocalPP() ||
                       plan.usesPipelineParallel() || plan.usesGlobalTP();
            };
            if (has_parallel_weight_authorities(prepared) ||
                has_parallel_weight_authorities(current))
            {
                return "Retained prepared-weight reuse currently requires one "
                       "single-device model authority";
            }
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
                    current.local_pp_layer_boundaries)
            {
                return "Retained prepared weights have a different rank-local "
                       "device topology";
            }
            if (prepared.runtime.mtp.enabled != current.runtime.mtp.enabled)
            {
                return "Retained prepared weights disagree on whether trailing "
                       "MTP predictor weights are required";
            }
            return std::nullopt;
        }

        /** @brief Return the one setup/materialization policy for this runner. */
        MoEOverlayCapacityAdmissionPolicy overlayCapacityPolicy(
            const MoERoutedExpertPlacementPlan &plan,
            const OrchestrationConfig &config,
            int world_size)
        {
            const bool materialize =
                plan.usesExpertOverlayAuthority() &&
                config.moe_rebalance.mode ==
                    MoERebalanceRuntimeMode::Dynamic;
            return {
                .materialize_migration_fabric = materialize,
                .shadow_slots_per_endpoint_layer =
                    materialize
                        ? MoEOverlayCapacityAdmissionPolicy::
                              kProductionShadowSlots
                        : 0,
                .staging_capacity_bytes =
                    materialize
                        ? MoEOverlayCapacityAdmissionPolicy::
                              kProductionStagingBytes
                        : 0,
                .distributed_transport =
                    materialize && world_size > 1,
                .overlay_world_size = std::max(1, world_size),
            };
        }

        /** @brief Convert an optional MiB CLI limit without integer wraparound. */
        std::optional<std::size_t> optionalMemoryBytes(
            const std::optional<std::size_t> &memory_mb,
            const char *name)
        {
            constexpr std::size_t kMiB = 1024u * 1024u;
            if (!memory_mb.has_value())
                return std::nullopt;
            if (*memory_mb >
                std::numeric_limits<std::size_t>::max() / kMiB)
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay ") + name +
                    " memory limit overflows size_t");
            }
            return *memory_mb * kMiB;
        }

        /**
         * @brief Return a safe upload-slot source bound from parsed GGUF metadata.
         *
         * Every concrete pipeline job is either one source tensor or a
         * row-aligned slice of one. The largest declared tensor is therefore a
         * model-wide upper bound suitable for capacity admission before jobs
         * are materialized on individual devices.
         */
        std::size_t maximumGGUFTensorPayloadBytes(const GGUFModel &model)
        {
            std::uint64_t maximum = 0;
            for (const auto &tensor : model.tensors)
                maximum = std::max(maximum, tensor.size_bytes);
            if (maximum == 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay GPU capacity cannot price an empty GGUF tensor manifest");
            }
            if (maximum > std::numeric_limits<std::size_t>::max())
            {
                throw std::overflow_error(
                    "ExpertOverlay maximum GGUF tensor payload exceeds size_t");
            }
            return static_cast<std::size_t>(maximum);
        }

        /** @brief Append one unsigned scalar in a host-independent wire order. */
        template <typename UInt>
        void appendCapacityWireScalar(
            std::vector<std::uint8_t> &bytes,
            UInt value)
        {
            static_assert(std::is_unsigned_v<UInt>);
            for (std::size_t offset = 0; offset < sizeof(UInt); ++offset)
            {
                bytes.push_back(static_cast<std::uint8_t>(
                    value >> (offset * 8u)));
            }
        }

        /** @brief Read one bounds-checked unsigned little-endian scalar. */
        template <typename UInt>
        UInt readCapacityWireScalar(
            const std::vector<std::uint8_t> &bytes,
            std::size_t &cursor,
            std::size_t end)
        {
            static_assert(std::is_unsigned_v<UInt>);
            if (cursor > end || sizeof(UInt) > end - cursor)
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity wire record is truncated");
            }
            UInt value = 0;
            for (std::size_t offset = 0; offset < sizeof(UInt); ++offset)
            {
                value |= static_cast<UInt>(bytes[cursor++]) <<
                         (offset * 8u);
            }
            return value;
        }

        /** @brief Serialize rank-local capacity records without ABI padding. */
        std::vector<std::uint8_t> serializeOverlayCapacityBudgets(
            const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &budgets)
        {
            constexpr std::uint32_t kMagic = 0x43415032u; // "CAP2"
            constexpr std::uint32_t kVersion = 1;
            constexpr std::size_t kRecordBytes = 56;
            std::vector<std::uint8_t> bytes;
            if (budgets.size() >
                std::numeric_limits<std::size_t>::max() / kRecordBytes)
            {
                throw std::overflow_error(
                    "ExpertOverlay capacity wire record count overflows size_t");
            }
            bytes.reserve(budgets.size() * kRecordBytes);
            for (const auto &budget : budgets)
            {
                appendCapacityWireScalar(bytes, kMagic);
                appendCapacityWireScalar(bytes, kVersion);
                appendCapacityWireScalar(
                    bytes, static_cast<std::uint32_t>(budget.world_rank));
                appendCapacityWireScalar(
                    bytes, static_cast<std::uint32_t>(budget.device.type));
                appendCapacityWireScalar(
                    bytes, static_cast<std::uint32_t>(budget.device.ordinal));
                appendCapacityWireScalar(bytes, std::uint32_t{0});
                appendCapacityWireScalar(
                    bytes, static_cast<std::uint64_t>(
                               budget.usable_budget_bytes));
                appendCapacityWireScalar(
                    bytes, static_cast<std::uint64_t>(budget.fixed_bytes));
                appendCapacityWireScalar(
                    bytes, static_cast<std::uint64_t>(
                               budget.additional_transfer_staging_bytes));
                appendCapacityWireScalar(
                    bytes, static_cast<std::uint64_t>(
                               budget.safety_reserve_bytes));
            }
            return bytes;
        }

        /** @brief Parse and authenticate one rank segment of capacity records. */
        void parseOverlayCapacityBudgetSegment(
            const std::vector<std::uint8_t> &bytes,
            std::size_t begin,
            std::size_t end,
            int contributing_rank,
            std::vector<MoEOverlayBoundPhysicalMemoryBudget> &output)
        {
            constexpr std::uint32_t kMagic = 0x43415032u;
            constexpr std::uint32_t kVersion = 1;
            constexpr std::size_t kRecordBytes = 56;
            if (begin > end || end > bytes.size() ||
                (end - begin) % kRecordBytes != 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity rank segment has invalid framing");
            }
            std::size_t cursor = begin;
            while (cursor < end)
            {
                const auto magic = readCapacityWireScalar<std::uint32_t>(
                    bytes, cursor, end);
                const auto version = readCapacityWireScalar<std::uint32_t>(
                    bytes, cursor, end);
                const int world_rank = static_cast<int>(
                    readCapacityWireScalar<std::uint32_t>(
                        bytes, cursor, end));
                const auto device_type = static_cast<DeviceType>(
                    readCapacityWireScalar<std::uint32_t>(
                        bytes, cursor, end));
                const int ordinal = static_cast<int>(
                    readCapacityWireScalar<std::uint32_t>(
                        bytes, cursor, end));
                (void)readCapacityWireScalar<std::uint32_t>(
                    bytes, cursor, end);
                const auto usable = readCapacityWireScalar<std::uint64_t>(
                    bytes, cursor, end);
                const auto fixed = readCapacityWireScalar<std::uint64_t>(
                    bytes, cursor, end);
                const auto additional =
                    readCapacityWireScalar<std::uint64_t>(
                        bytes, cursor, end);
                const auto reserve = readCapacityWireScalar<std::uint64_t>(
                    bytes, cursor, end);
                const DeviceId device(device_type, ordinal);
                if (magic != kMagic || version != kVersion ||
                    world_rank != contributing_rank ||
                    !device.is_valid() ||
                    (!device.is_cpu() && !device.is_gpu()) ||
                    usable > std::numeric_limits<std::size_t>::max() ||
                    fixed > std::numeric_limits<std::size_t>::max() ||
                    additional > std::numeric_limits<std::size_t>::max() ||
                    reserve > std::numeric_limits<std::size_t>::max())
                {
                    throw std::invalid_argument(
                        "ExpertOverlay capacity wire record failed identity or range validation");
                }
                output.push_back({
                    .world_rank = world_rank,
                    .device = device,
                    .resource_id =
                        MoEOverlayLocalCapacityPlanner::physicalResourceId(
                            world_rank, device),
                    .usable_budget_bytes =
                        static_cast<std::size_t>(usable),
                    .fixed_bytes = static_cast<std::size_t>(fixed),
                    .additional_transfer_staging_bytes =
                        static_cast<std::size_t>(additional),
                    .safety_reserve_bytes =
                        static_cast<std::size_t>(reserve),
                });
            }
        }

        /** @brief All-gather variable rank-local physical BOMs in rank order. */
        std::vector<MoEOverlayBoundPhysicalMemoryBudget>
        gatherOverlayCapacityBudgets(
            const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &local,
            const std::shared_ptr<IMPIContext> &mpi_context)
        {
            if (!mpi_context || mpi_context->world_size() <= 1)
                return local;

            const auto send = serializeOverlayCapacityBudgets(local);
            if (send.size() > static_cast<std::size_t>(INT_MAX))
            {
                throw std::overflow_error(
                    "ExpertOverlay rank-local capacity wire payload exceeds MPI int count");
            }
            const int world_size = mpi_context->world_size();
            const int send_count = static_cast<int>(send.size());
            std::vector<int> counts(static_cast<std::size_t>(world_size), 0);
            mpi_context->allgather_bytes(
                &send_count, counts.data(), sizeof(send_count));

            std::vector<int> displacements(
                static_cast<std::size_t>(world_size), 0);
            std::size_t total = 0;
            for (int rank = 0; rank < world_size; ++rank)
            {
                if (counts[static_cast<std::size_t>(rank)] < 0 ||
                    total > static_cast<std::size_t>(INT_MAX))
                {
                    throw std::overflow_error(
                        "ExpertOverlay global capacity wire payload exceeds MPI int displacement");
                }
                displacements[static_cast<std::size_t>(rank)] =
                    static_cast<int>(total);
                total += static_cast<std::size_t>(
                    counts[static_cast<std::size_t>(rank)]);
            }
            if (total > static_cast<std::size_t>(INT_MAX))
            {
                throw std::overflow_error(
                    "ExpertOverlay global capacity wire payload exceeds MPI int count");
            }

            std::vector<std::uint8_t> received(total);
            mpi_context->allgatherv_bytes(
                send.empty() ? nullptr : send.data(),
                send_count,
                received.empty() ? nullptr : received.data(),
                counts.data(),
                displacements.data());

            std::vector<MoEOverlayBoundPhysicalMemoryBudget> result;
            for (int rank = 0; rank < world_size; ++rank)
            {
                const std::size_t begin = static_cast<std::size_t>(
                    displacements[static_cast<std::size_t>(rank)]);
                const std::size_t end = begin + static_cast<std::size_t>(
                    counts[static_cast<std::size_t>(rank)]);
                parseOverlayCapacityBudgetSegment(
                    received, begin, end, rank, result);
            }
            return result;
        }

        /** @brief Setup-only readiness consensus before variable MPI exchange. */
        bool allOverlayRanksReady(
            bool local_ready,
            const std::shared_ptr<IMPIContext> &mpi_context)
        {
            if (!mpi_context || mpi_context->world_size() <= 1)
                return local_ready;
            const int ready = local_ready ? 1 : 0;
            std::vector<int> all_ready(
                static_cast<std::size_t>(mpi_context->world_size()), 0);
            mpi_context->allgather_bytes(
                &ready, all_ready.data(), sizeof(ready));
            return std::all_of(
                all_ready.begin(), all_ready.end(),
                [](int value) { return value == 1; });
        }

        /**
         * @brief Prove every overlay rank will execute the same admission loop.
         *
         * Capacity admission performs collectives once per candidate. A rank
         * with a different configured bucket ladder would otherwise leave its
         * peers inside an unmatched collective. First gathering the count gives
         * every rank the same safe exit point; only equal counts permit the
         * second, fixed-width gather of the exact candidate values.
         */
        void requireIdenticalOverlayGraphRowCandidates(
            const std::vector<int> &local_candidates,
            const std::shared_ptr<IMPIContext> &mpi_context)
        {
            if (local_candidates.empty())
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity admission requires at least one graph-row candidate");
            }
            const bool uncaptured_contract =
                local_candidates.size() == 1u &&
                local_candidates.front() == 0;
            if (!uncaptured_contract &&
                (std::any_of(
                     local_candidates.begin(),
                     local_candidates.end(),
                     [](int rows) { return rows <= 0; }) ||
                 !std::is_sorted(
                     local_candidates.begin(),
                     local_candidates.end(),
                     std::greater<int>{})))
            {
                throw std::invalid_argument(
                    "ExpertOverlay graph-row candidates must be positive and descending");
            }
            if (!mpi_context || mpi_context->world_size() <= 1)
                return;
            if (local_candidates.size() >
                static_cast<std::size_t>(INT_MAX) / sizeof(int))
            {
                throw std::overflow_error(
                    "ExpertOverlay graph-row candidate payload exceeds MPI int count");
            }

            const int world_size = mpi_context->world_size();
            const int local_count = static_cast<int>(local_candidates.size());
            std::vector<int> counts(
                static_cast<std::size_t>(world_size), 0);
            mpi_context->allgather_bytes(
                &local_count, counts.data(), sizeof(local_count));
            if (!std::all_of(
                    counts.begin(), counts.end(),
                    [&](int count) { return count == local_count; }))
            {
                throw std::invalid_argument(
                    "ExpertOverlay ranks disagree on the captured-prefill candidate count");
            }

            const auto candidate_count =
                static_cast<std::size_t>(local_count);
            if (candidate_count >
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(world_size))
            {
                throw std::overflow_error(
                    "ExpertOverlay gathered graph-row candidates overflow size_t");
            }
            std::vector<int> gathered(
                candidate_count * static_cast<std::size_t>(world_size));
            mpi_context->allgather_bytes(
                local_candidates.data(),
                gathered.data(),
                candidate_count * sizeof(int));
            for (int rank = 0; rank < world_size; ++rank)
            {
                const auto begin = gathered.begin() +
                                   static_cast<std::ptrdiff_t>(
                                       candidate_count *
                                       static_cast<std::size_t>(rank));
                if (!std::equal(
                        local_candidates.begin(),
                        local_candidates.end(),
                        begin))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay ranks disagree on the captured-prefill candidate ladder");
                }
            }
        }

    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> freezeMoEExpertOverlayPlanForModel(
        IModelContext &model_ctx,
        const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan,
        const MTPRuntimeConfig &mtp)
    {
        if (!plan)
            return nullptr;

        InferenceRunnerConfig runner_config;
        runner_config.moe_routed_expert_plan = plan;
        runner_config.mtp = mtp;
        return resolveMoERoutedExpertPlacementPlanForModel(model_ctx, runner_config);
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
        std::unique_ptr<IExecutionPlanBuilder> plan_builder,
        std::shared_ptr<ModelContext> model_ctx)
        : config_(std::move(config)),
          plan_builder_(std::move(plan_builder)),
          model_ctx_(std::move(model_ctx)),
          sampler_(0)
    {
        if (!plan_builder_)
            plan_builder_ = createExecutionPlanBuilder();
        if (!model_ctx_)
        {
            throw std::invalid_argument(
                "OrchestrationRunner preloaded ModelContext must be non-null");
        }
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder,
        ModelContextReuseContract reuse_contract)
        : config_(std::move(config)),
          plan_builder_(std::move(plan_builder)),
          model_ctx_(std::move(reuse_contract.context)),
          retained_prepared_weight_plan_(
              std::move(reuse_contract.prepared_weight_plan)),
          sampler_(0)
    {
        if (!plan_builder_)
        {
            plan_builder_ = createExecutionPlanBuilder();
        }
        if (!model_ctx_)
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract must contain a ModelContext");
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
          runner_(std::move(runner)),
          physical_runner_backend_access_enabled_(false),
          initialized_(true), sampler_(0)
    {
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan,
        std::unique_ptr<IInferenceRunner> runner,
        std::shared_ptr<IMPIContext> mpi_ctx)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true),
          mpi_ctx_(std::move(mpi_ctx)), runner_(std::move(runner)),
          physical_runner_backend_access_enabled_(false),
          initialized_(true), sampler_(0)
    {
        mpi_coordinated_root_rank_ = coordinatedRootRankForRunner(
            config_.moe_routed_expert_plan,
            mpi_ctx_);
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

            // Step 4b: Validate topology and requested context before either
            // becomes an input to physical capacity admission.
            if (!validateTPPPConfiguration())
            {
                syncInitStep(false, "validateTPPPConfiguration");
                return false;
            }
            if (!syncInitStep(true, "validateTPPPConfiguration"))
            {
                return false;
            }

            if (!validateContextLength())
            {
                syncInitStep(false, "validateContextLength");
                return false;
            }
            if (!syncInitStep(true, "validateContextLength"))
            {
                return false;
            }

            // Step 4c: Freeze model-aware placement and captured-prefill
            // capacity together before constructing any endpoint graph.
            if (!freezeMoEExpertOverlayPlanForLoadedModel())
            {
                syncInitStep(false, "freezeMoEExpertOverlayPlanForLoadedModel");
                return false;
            }
            if (!syncInitStep(true, "freezeMoEExpertOverlayPlanForLoadedModel"))
            {
                return false;
            }

            // Step 4d: Create one owner-map authority before any participant
            // graph can bind its dispatch ticket and runtime placement banks.
            if (!initializeMoEExpertOverlayResidencyAuthority())
            {
                syncInitStep(false, "initializeMoEExpertOverlayResidencyAuthority");
                return false;
            }
            if (!syncInitStep(
                    true,
                    "initializeMoEExpertOverlayResidencyAuthority"))
            {
                return false;
            }

            // Step 5: Validate the frozen memory plan.
            if (!validateMemoryPlan())
            {
                syncInitStep(false, "validateMemoryPlan");
                return false;
            }
            if (!syncInitStep(true, "validateMemoryPlan"))
            {
                return false;
            }

            // A distributed ExpertOverlay is one sparse-collective protocol,
            // not independent rank-local prefill loops. Freeze its common
            // bucket ladder after all local memory plans are known and before
            // any root or expert-only graph binds persistent arena addresses.
            if (!establishMoEOverlayPrefillScheduleContract())
            {
                syncInitStep(false, "establishMoEOverlayPrefillScheduleContract");
                return false;
            }
            if (!syncInitStep(true, "establishMoEOverlayPrefillScheduleContract"))
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

            /*
             * A request ticket has a 30-second protocol deadline once armed;
             * native graph construction does not. Seal every continuation
             * executable now, while remote expert ranks have retained their
             * parents but no follower is admitted. LocalTP children perform
             * this concurrently so capture-wave collectives remain symmetric.
             */
            if (!materializeMoEOverlayContinuationServingGraphFamily())
            {
                syncInitStep(
                    false,
                    "materializeMoEOverlayContinuationServingGraphFamily");
                return false;
            }
            if (!syncInitStep(
                    true,
                    "materializeMoEOverlayContinuationServingGraphFamily"))
            {
                return false;
            }

            /*
             * Graph construction resolves the only authoritative prepared
             * engine lifetimes. Materialize inactive slots and transfer lanes
             * only after those initial banks are complete, but before serving
             * can admit a request or publish routing evidence.
             */
            if (!initializeMoEExpertOverlayResidencyMaintenance())
            {
                syncInitStep(
                    false,
                    "initializeMoEExpertOverlayResidencyMaintenance");
                return false;
            }
            if (!syncInitStep(
                    true,
                    "initializeMoEExpertOverlayResidencyMaintenance"))
            {
                return false;
            }

            if (!initializeMoEOverlayInferenceTransactions())
            {
                syncInitStep(
                    false,
                    "initializeMoEOverlayInferenceTransactions");
                return false;
            }
            if (!syncInitStep(
                    true,
                    "initializeMoEOverlayInferenceTransactions"))
            {
                return false;
            }

            if (!publishMoEOverlayCollectiveRequestGeneration("initialization"))
            {
                syncInitStep(false, "publishMoEOverlayCollectiveRequestGeneration");
                return false;
            }
            if (!syncInitStep(
                    true,
                    "publishMoEOverlayCollectiveRequestGeneration"))
            {
                return false;
            }

            /*
             * GPU inference owns logits on device for the entire request.
             * Sampling and speculative publication consume the resident tensor;
             * only compact response tokens cross to the host. Install this once
             * at the lifecycle boundary rather than toggling gather behavior per
             * request, which could expose stale host data between transactions.
             * Tests that deliberately inspect full logits must explicitly opt
             * into host observation after initialization.
             */
            if (runner_ && runner_->primaryDeviceId().is_gpu())
            {
                runner_->setSkipLogitsGatherPrefill(true);
                runner_->setSkipLogitsGatherDecode(true);
            }
            if (runner_ &&
                !runner_->configureMTPRequestStopTokens(stop_tokens_))
            {
                syncInitStep(false, "configureMTPRequestStopTokens");
                return setError(
                    "Inference runner rejected request stop-token policy");
            }
            if (!syncInitStep(true, "configureMTPRequestStopTokens"))
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

    bool OrchestrationRunner::initializeForDryRun()
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
                    setError(std::string("Dry-run preflight failed on another rank at step: ") + step_name);
                }
                return false;
            }
            return true;
        };

        try
        {
            if (!initializeMPI())
                return false;
            if (!syncInitStep(true, "initializeMPI"))
                return false;

            if (!buildExecutionPlan())
            {
                syncInitStep(false, "buildExecutionPlan");
                return false;
            }
            if (!syncInitStep(true, "buildExecutionPlan"))
                return false;

            if (!setupLocalTPContext())
            {
                syncInitStep(false, "setupLocalTPContext");
                return false;
            }
            if (!syncInitStep(true, "setupLocalTPContext"))
                return false;

            if (!setupLocalPPContext())
            {
                syncInitStep(false, "setupLocalPPContext");
                return false;
            }
            if (!syncInitStep(true, "setupLocalPPContext"))
                return false;

            if (!loadWeights(/*prepopulate_page_cache=*/false))
            {
                syncInitStep(false, "loadWeights");
                return false;
            }
            if (!syncInitStep(true, "loadWeights"))
                return false;

            if (!validateTPPPConfiguration())
            {
                syncInitStep(false, "validateTPPPConfiguration");
                return false;
            }
            if (!syncInitStep(true, "validateTPPPConfiguration"))
                return false;

            if (!validateContextLength())
            {
                syncInitStep(false, "validateContextLength");
                return false;
            }
            if (!syncInitStep(true, "validateContextLength"))
                return false;

            if (!freezeMoEExpertOverlayPlanForLoadedModel())
            {
                syncInitStep(false, "freezeMoEExpertOverlayPlanForLoadedModel");
                return false;
            }
            if (!syncInitStep(true, "freezeMoEExpertOverlayPlanForLoadedModel"))
                return false;

            if (!validateMemoryPlan())
            {
                syncInitStep(false, "validateMemoryPlan");
                return false;
            }
            if (!syncInitStep(true, "validateMemoryPlan"))
                return false;

            printStartupBanner(stdout);

            if (!syncInitStep(true, "dryRunComplete"))
                return false;

            initialized_ = true;

            if (!mpi_ctx_ ||
                mpi_ctx_->rank() == mpi_coordinated_root_rank_)
            {
                LOG_INFO("[Main] --dry-run complete: preflight validation passed; no compute graph was built and no inference was started");
            }

            return true;
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Dry-run preflight failed: ") + e.what());
        }
    }

    void OrchestrationRunner::shutdown()
    {
        if (!initialized_)
        {
            return;
        }

        /*
         * Stop new proposals and drain transfer events while graph engines,
         * source banks, and backend contexts are still alive. Final counters
         * are flushed only after this worker has published its terminal state.
         */
        shutdownMoEExpertOverlayResidencyMaintenance();
        PerfStatsCollector::flushFromEnv();

        if (runner_)
        {
            try
            {
                resetUnderlyingRunnerRequestState("shutdown");
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[OrchestrationRunner] shutdown cache reset failed: " << e.what());
                if (auto *rank = dynamic_cast<RankOrchestrator *>(runner_.get()))
                    rank->clearPendingGpuDirectExpertTransfersForAllDevices();
            }
            synchronizeRunnerDevicesBeforeRelease(
                runner_.get(),
                physical_runner_backend_access_enabled_);
        }

        /*
         * In coordinated serving mode a worker runner is blocked in
         * runMPIWorkerLoop() until the coordinated root publishes SHUTDOWN. Tests create
         * fresh runners in one MPI process for multiple fixtures, so making
         * shutdown itself close that protocol is a lifecycle invariant rather
         * than a test-specific cleanup workaround.
         */
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
        {
            shutdownMPIWorkers();
        }

        // Release resources in reverse order
        moe_overlay_inference_transaction_follower_.reset();
        moe_overlay_inference_transaction_coordinator_.reset();
        runner_.reset();
        moe_expert_overlay_participant_residency_.reset();
        moe_expert_overlay_residency_authority_.reset();
        moe_expert_overlay_decode_histogram_.reset();
        moe_expert_overlay_interference_probe_.reset();
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

    bool OrchestrationRunner::leastLoadedCurrentBatchPrefillUsesStableWindows() const
    {
        const auto &placement = config_.moe_routed_expert_plan;
        if (!placement || !placement->enabled)
            return false;

        return std::any_of(
            placement->domains.begin(),
            placement->domains.end(),
            [](const RoutedExpertDomain &domain)
            {
                return domain.routed_prefill_assignment_policy ==
                       RoutedExpertAssignmentPolicy::LeastLoadedResident;
            });
    }

    int OrchestrationRunner::stableRoutedPrefillAssignmentWindowTokens(
        int prefix_cache_block_size,
        bool prefix_cache_enabled) const
    {
        if (!leastLoadedCurrentBatchPrefillUsesStableWindows())
        {
            return 0;
        }

        const int plan_window =
            plan_.runtime.moe_routed_prefill.assignment_window_tokens;
        const int config_window =
            config_.moe_routed_prefill.assignment_window_tokens;
        if (plan_window < 0 || config_window < 0)
        {
            return -1;
        }
        if (plan_window > 0)
        {
            return plan_window;
        }
        if (config_window > 0)
        {
            return config_window;
        }
        if (prefix_cache_enabled)
        {
            return prefix_cache_block_size;
        }
        return 0;
    }

    bool OrchestrationRunner::forwardSinglePrefillTransaction(
        const int *tokens,
        int token_count,
        const std::string &failure_message)
    {
        if (!runner_ || !tokens || token_count <= 0)
            return setError(failure_message);

        const auto &exec = debugEnv().execution;
        const auto &overlay_schedule =
            plan_.runtime.overlay_prefill_schedule;
        /*
         * Ordinary runners may derive a bucket from their own graph arena.
         * ExpertOverlay cannot: its next sparse dispatch must be accepted by
         * every remote participant.  Initialization has already frozen a
         * root-authoritative schedule at the minimum admitted capacity, so the
         * request hot path only reads that typed contract.
         */
        const auto buckets = overlay_schedule.enabled()
                                 ? overlay_schedule.bucket_rows
                                 : prefillGraphBucketsAtOrBelowCapacity(
                                       exec.prefill_graph_bucket_sizes,
                                       plan_.runtime.resident_graph_rows);
        const bool long_bucketed_prefill =
            exec.gpu_graphs &&
            exec.prefill_graph_buckets &&
            !buckets.empty() &&
            token_count > buckets.back();

        if (long_bucketed_prefill)
        {
            if (!runner_->supportsPrefillChunkSchedule(token_count))
            {
                ++prefill_chunk_stats_.failures;
                return setError(
                    failure_message +
                    " (chunked prefill required by activation graph bucket capacity, "
                    "but runner does not support prefill chunk scheduling)");
            }

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
            std::uint64_t schedule_fingerprint = 14695981039346656037ull;
            std::int64_t execution_rows = 0;
            for (const auto &chunk : chunk_schedule.chunks)
            {
                execution_rows += chunk.bucket_seq_len;
                schedule_fingerprint = mixInferenceWorkloadScalar(
                    schedule_fingerprint,
                    static_cast<std::uint64_t>(chunk.real_count));
                schedule_fingerprint = mixInferenceWorkloadScalar(
                    schedule_fingerprint,
                    static_cast<std::uint64_t>(chunk.bucket_seq_len));
            }
            if (execution_rows <= 0 ||
                execution_rows > std::numeric_limits<int>::max())
            {
                ++prefill_chunk_stats_.failures;
                return setError(
                    failure_message +
                    " (chunked prefill execution-row identity overflowed)");
            }
            MoEOverlayInferenceInterferenceScope interference_scope(
                moe_expert_overlay_interference_probe_.get(),
                inferenceWorkloadIdentity(
                    ExpertHistogramSource::PrefillChunk,
                    token_count,
                    static_cast<int>(execution_rows),
                    static_cast<int>(chunk_schedule.chunks.size()),
                    0,
                    schedule_fingerprint));
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

        int execution_rows = token_count;
        if (exec.gpu_graphs && exec.prefill_graph_buckets && !buckets.empty())
        {
            const auto selected =
                selectPrefillGraphBucket(token_count, buckets);
            if (selected)
                execution_rows = selected.bucket_seq_len;
        }
        MoEOverlayInferenceInterferenceScope interference_scope(
            moe_expert_overlay_interference_probe_.get(),
            inferenceWorkloadIdentity(
                ExpertHistogramSource::PrefillChunk,
                token_count,
                execution_rows,
                1,
                0));
        if (!runner_->forwardPrefill(tokens, token_count))
            return setError(failure_message);
        return true;
    }

    bool OrchestrationRunner::forwardPrefillTokens(
        const int *tokens,
        int token_count,
        const std::string &failure_message,
        int stable_segment_tokens)
    {
        if (!runner_ || !tokens || token_count <= 0)
            return setError(failure_message);
        if (stable_segment_tokens < 0)
        {
            return setError(
                failure_message +
                " (stable prefill segment size cannot be negative)");
        }

        if (stable_segment_tokens > 0 && token_count > stable_segment_tokens)
        {
            for (int offset = 0; offset < token_count; offset += stable_segment_tokens)
            {
                const int segment_tokens =
                    std::min(stable_segment_tokens, token_count - offset);
                if (!forwardSinglePrefillTransaction(
                        tokens + offset,
                        segment_tokens,
                        failure_message))
                {
                    return false;
                }
            }
            return true;
        }

        return forwardSinglePrefillTransaction(tokens, token_count, failure_message);
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

        const bool mpi_coordinated_world =
            mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->world_size() > 1;
        const bool mpi_root_command =
            mpi_coordinated_world &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_;
        ScopedMPIRootCommandFailureAbort prefill_command_failure_abort(
            mpi_ctx_.get(),
            mpi_root_command,
            "PREFILL",
            &last_error_);
        ScopedMoEOverlayRootCommand overlay_prefill_command(
            moe_overlay_inference_transaction_coordinator_);

        if (!runner_ ||
            !runner_->configureMTPRequestStopTokens(stop_tokens_))
        {
            return setError(
                "Inference runner rejected request stop-token policy at "
                "request admission");
        }
        if (!runner_->configureMTPRequestPenaltyPolicy(
                MTPRequestPenaltyPolicy{
                    .presence_penalty =
                        active_sampling_params_.presence_penalty,
                    .frequency_penalty =
                        active_sampling_params_.frequency_penalty,
                }))
        {
            return setError(
                "Inference runner rejected request penalty policy at "
                "request admission");
        }

        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        mtp_stats_ = {};
        device_generation_terminal_ledger_authoritative_ = false;
        device_generation_embedded_moe_maintenance_pending_ack_ = false;
        device_generation_admission_.reset();
        ready_mtp_condition_.reset();
        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        decode_transaction_planning_position_.reset();
        clearBatchedDecodeState();
        last_token_ = prompt_tokens.back();

        // Broadcast to worker ranks so they prefill with the same tokens.
        if (mpi_root_command)
        {
            broadcastCommand(MPICommand::PREFILL);
            int32_t n_tokens = static_cast<int32_t>(prompt_tokens.size());
            mpi_ctx_->broadcast_int32(
                &n_tokens, 1, mpi_coordinated_root_rank_);
            // const_cast is safe: the root is the sender, buffer is not modified.
            mpi_ctx_->broadcast_int32(const_cast<int32_t *>(prompt_tokens.data()),
                                      static_cast<size_t>(n_tokens),
                                      mpi_coordinated_root_rank_);
            prefill_command_failure_abort.markPublished();
        }

        if (overlay_prefill_command.active())
        {
            if (!mpi_root_command ||
                moe_overlay_collective_generation_id_ == 0 ||
                !moe_expert_overlay_residency_authority_)
            {
                return setError(
                    "ExpertOverlay prefill authority has incomplete root, "
                    "request-generation, or residency ownership");
            }
            const auto snapshot =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!snapshot || !snapshot->valid() ||
                moe_overlay_inference_command_sequence_ ==
                    std::numeric_limits<std::uint64_t>::max())
            {
                return setError(
                    "ExpertOverlay prefill authority has no valid placement "
                    "epoch or command identity");
            }
            const MoEOverlayInferenceCommandIdentity command_identity{
                .request_generation =
                    moe_overlay_collective_generation_id_,
                .command_id =
                    ++moe_overlay_inference_command_sequence_,
                .initial_placement_epoch = snapshot->epoch,
            };
            std::string command_error;
            if (!overlay_prefill_command.begin(
                    command_identity, &command_error))
            {
                return setError(
                    "ExpertOverlay remote prefill command admission failed: " +
                    command_error);
            }
        }

        const auto complete_distributed_prefill = [&]() -> bool
        {
            std::string command_error;
            if (!overlay_prefill_command.complete(&command_error))
            {
                return setError(
                    "ExpertOverlay prefill command completion failed: " +
                    command_error);
            }
            prefill_command_failure_abort.markCompleted();
            return true;
        };

        const auto &plan_prefix = plan_.runtime.prefix_cache;
        const auto &config_prefix = config_.prefix_cache;
        const MTPRuntimeConfig &active_mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        /*
         * The remote expert rank retains MTP-draft and grouped-verifier graph
         * families, but it does not own a sidecar head, sampler, KV state, or
         * depth controller.  Its follower consumes those graph families only
         * after the continuation rank publishes an authenticated transaction
         * ticket.  Treating it as an ordinary MTP decoder here would reject a
         * perfectly complete follower merely because supportsChainedMTPDrafts()
         * correctly reports false.
         */
        const bool owns_mtp_continuation_authority =
            active_mtp.enabled &&
            !moe_overlay_inference_transaction_follower_;
        const bool prefix_cache_enabled =
            (plan_prefix.enabled || config_prefix.enabled) &&
            plan_prefix.storage_mode != PrefixCacheStorageMode::Disabled &&
            config_prefix.storage_mode != PrefixCacheStorageMode::Disabled;
        const bool mtp_full_hit_requires_terminal_hidden =
            owns_mtp_continuation_authority &&
            active_mtp.require_terminal_hidden_for_full_hit;
        prefix_request_summary_ = {};
        prefix_request_summary_.enabled = prefix_cache_enabled;
        prefix_request_summary_.requested_tokens = static_cast<int>(prompt_tokens.size());

        if (owns_mtp_continuation_authority && runner_)
        {
            if (!ensureMTPDepthController(active_mtp))
            {
                return false;
            }
            /*
             * prefill() is the public request-admission boundary even when the
             * prefix cache satisfies every prompt token and no model prefill
             * graph runs.  Adaptive depth observations belong to one request:
             * carrying a promotion, rejection memory, or partial hysteresis
             * window into the next generate() call makes identical seeded
             * requests select different graph depths.
             *
             * clearCache() also resets this controller, but callers are not
             * required to clear the persistent prefix cache between requests.
             * Reset here after configuration so both ordinary admission and
             * exact prefix restore begin from the configured initial depth.
             */
            mtp_depth_controller_->reset();
            const int effective_max_draft_tokens = effectiveMTPMaxDraftDepth(active_mtp);
            if (effective_max_draft_tokens < 1)
            {
                return setError(
                    "MTP decode requires --mtp-draft-tokens >= 1");
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
                    runner_->moeRuntimeMovementEpoch());

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
                    local_hit.block_size > 0
                        ? local_hit.block_size
                        : (plan_prefix.block_size > 0
                               ? plan_prefix.block_size
                               : config_prefix.block_size);
                PrefixLookupResult coordinated_hit =
                    makePrefixLookupResult(coordination, coordination_block_size);
                const bool coordinated_cache_available =
                    coordinated_hit.cache_enabled && coordinated_hit.supported;
                const bool coordinated_prefix_hit =
                    coordinated_cache_available && coordinated_hit.cached_tokens > 0;
                int stable_prefix_prefill_segment_tokens =
                    stableRoutedPrefillAssignmentWindowTokens(
                        coordinated_hit.block_size,
                        coordinated_prefix_hit);
                if (leastLoadedCurrentBatchPrefillUsesStableWindows() &&
                    stable_prefix_prefill_segment_tokens < 0)
                {
                    std::ostringstream error;
                    error << "Routed-prefill assignment window cannot be negative"
                          << " plan_prefill_window="
                          << plan_.runtime.moe_routed_prefill.assignment_window_tokens
                          << " config_prefill_window="
                          << config_.moe_routed_prefill.assignment_window_tokens;
                    return setError(error.str());
                }
                if (coordinated_prefix_hit &&
                    leastLoadedCurrentBatchPrefillUsesStableWindows() &&
                    stable_prefix_prefill_segment_tokens <= 0)
                {
                    std::ostringstream error;
                    error << "Prefix cache least-loaded routed prefill requires a positive stable block size"
                          << " local_hit_block_size=" << local_hit.block_size
                          << " plan_block_size=" << plan_prefix.block_size
                          << " config_block_size=" << config_prefix.block_size
                          << " plan_prefill_window="
                          << plan_.runtime.moe_routed_prefill.assignment_window_tokens
                          << " config_prefill_window="
                          << config_.moe_routed_prefill.assignment_window_tokens;
                    return setError(error.str());
                }
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

                prefill_logits_ready_ = false;
                ready_mtp_condition_.reset();
                pending_mtp_condition_token_.reset();
                pending_mtp_condition_params_.reset();
                pending_mtp_condition_resident_state_.reset();
                prelaunched_mtp_first_sidecar_resident_state_.reset();
                prelaunched_mtp_first_sidecar_params_.reset();
                decode_transaction_planning_position_.reset();

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
                    hit.restore_model_runtime_state = matched_tokens > 0;
                    hit.restore_hybrid_state_for_suffix_prefill =
                        matched_tokens > 0 &&
                        matched_tokens < static_cast<int>(prompt_tokens.size());
                    return hit;
                };

                PrefixLookupResult common_hit = make_common_hit();
                prefix_request_summary_.bypassed = !coordinated_hit.supported;
                prefix_request_summary_.bypass_reason = coordinated_hit.bypass_reason;
                if (owns_mtp_continuation_authority &&
                    matched_tokens > 0 &&
                    matched_tokens < static_cast<int>(prompt_tokens.size()) &&
                    !common_hit.has_terminal_hidden)
                {
                    const int block_size =
                        common_hit.block_size > 0 ? common_hit.block_size : plan_prefix.block_size;
                    matched_tokens = std::max(0, matched_tokens - std::max(1, block_size));
                    common_hit = make_common_hit();
                }

                if (matched_tokens == static_cast<int>(prompt_tokens.size()) &&
                    !(common_hit.has_terminal_logits &&
                      (!mtp_full_hit_requires_terminal_hidden ||
                       common_hit.has_terminal_hidden)))
                {
                    /*
                     * A full prefix hit can serve decode directly only when the
                     * terminal block carries the logits and, for MTP, the
                     * verifier's terminal hidden row.  Decide that boundary
                     * before importing any cached state so a shortened restore
                     * is a single prefix transaction rather than a populate,
                     * request reset, and second populate.
                     */
                    const int block_size =
                        common_hit.block_size > 0 ? common_hit.block_size : plan_prefix.block_size;
                    matched_tokens = std::max(0, matched_tokens - std::max(1, block_size));
                    common_hit = make_common_hit();
                    LOG_DEBUG("[OrchestrationRunner] Prefix cache full hit lacks terminal state; "
                              "restoring matched_tokens="
                              << matched_tokens
                              << " and pre-filling the suffix");
                }

                auto full_terminal_hit_has_restorable_runtime = [&]() -> bool
                {
                    if (matched_tokens != static_cast<int>(prompt_tokens.size()) ||
                        !common_hit.has_terminal_logits ||
                        (mtp_full_hit_requires_terminal_hidden &&
                         !common_hit.has_terminal_hidden) ||
                        common_hit.blocks.empty())
                    {
                        return false;
                    }

                    const auto &terminal_block = common_hit.blocks.back();
                    return terminal_block.has_model_runtime_state &&
                           terminal_block.model_runtime_state_storage &&
                           !terminal_block.model_runtime_state_storage->empty();
                };

                if (stable_prefix_prefill_segment_tokens > 0 &&
                    matched_tokens % stable_prefix_prefill_segment_tokens != 0 &&
                    !full_terminal_hit_has_restorable_runtime())
                {
                    const int original_matched_tokens = matched_tokens;
                    matched_tokens =
                        (matched_tokens / stable_prefix_prefill_segment_tokens) *
                        stable_prefix_prefill_segment_tokens;
                    common_hit = make_common_hit();
                    LOG_INFO("[OrchestrationRunner] Prefix cache LLEP prefill aligned "
                             "matched_tokens="
                             << original_matched_tokens
                             << " down to stable boundary " << matched_tokens
                             << " (stable_block="
                             << stable_prefix_prefill_segment_tokens
                             << "); suffix prefill will recompute the remainder");
                }

                if (matched_tokens == 0 && stable_prefix_prefill_segment_tokens > 0)
                {
                    stable_prefix_prefill_segment_tokens =
                        stableRoutedPrefillAssignmentWindowTokens(
                            coordinated_hit.block_size,
                            /*prefix_cache_enabled=*/false);
                }

                if (matched_tokens > 0)
                {
                    if (!runner_->populatePrefix(common_hit))
                    {
                        std::ostringstream error;
                        error << "Prefix cache populate failed at matched_tokens="
                              << matched_tokens
                              << "; refusing to downgrade the hit to a miss";
                        return setError(error.str());
                    }
                }
                else
                {
                    /*
                     * Misses with live request state still cross the ordinary
                     * request boundary before prefill.  A fresh runner at
                     * position zero has no KV/GDN/MTP rows to retire, so
                     * manufacturing an extra reset boundary would make the
                     * prefix-cache miss path semantically different from an
                     * uncached first request.  Hits do not reset here:
                     * populatePrefix() owns the prefix restore boundary and
                     * may preserve graph-owned model runtime state long enough
                     * to replay a portable snapshot.
                     */
                    const int current_position = runner_->get_position();
                    if (current_position < 0)
                    {
                        return setError(
                            "Prefix cache miss observed a negative runner position before request reset");
                    }
                    if (current_position > 0)
                    {
                        resetUnderlyingRunnerRequestState("prefix-cache-initial-reset");
                        if (!advanceMoEOverlayCollectiveRequestGeneration(
                                "prefix-cache-initial-reset"))
                        {
                            return false;
                        }
                    }
                }

                if (matched_tokens > 0 && prefixCacheTraceEnabled())
                {
                    const PrefixRuntimeStateSnapshot probe = runner_->prefixStateProbe();
                    LOG_INFO("[PREFIX_TRACE] post-populate prefix state "
                             << summarizePrefixProbeForTrace(probe));
                }

                int suffix_start = matched_tokens;
                int suffix_len = static_cast<int>(prompt_tokens.size()) - suffix_start;
                bool terminal_state_restored = false;

                if (suffix_len > 0)
                {
                    if (!forwardPrefillTokens(prompt_tokens.data() + suffix_start,
                                              suffix_len,
                                              "Forward pass failed during prefix-cache suffix prefill",
                                              stable_prefix_prefill_segment_tokens))
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
                    std::ostringstream error;
                    error << "Prefix cache terminal restore failed at matched_tokens="
                          << matched_tokens
                          << " terminal_logits="
                          << (common_hit.has_terminal_logits ? "yes" : "no")
                          << " terminal_hidden="
                          << (common_hit.has_terminal_hidden ? "yes" : "no")
                          << " mtp_requires_hidden="
                          << (mtp_full_hit_requires_terminal_hidden ? "yes" : "no");
                    return setError(error.str());
                }

                if (prefixCacheTraceEnabled())
                {
                    /*
                     * This is the exact source boundary captured by
                     * harvestPrefix(). Pair it with the post-populate trace
                     * above to prove that restore reproduced every probed
                     * persistent-state byte before suffix decode starts.
                     */
                    const PrefixRuntimeStateSnapshot probe =
                        runner_->prefixStateProbe();
                    LOG_INFO("[PREFIX_TRACE] pre-harvest prefix state "
                             << summarizePrefixProbeForTrace(probe));
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
                if (!initializeDecodeTransactionPlanningPositionAfterPrefill(
                        static_cast<int>(prompt_tokens.size()),
                        terminal_state_restored
                            ? "prefix_terminal_restore"
                            : (matched_tokens > 0
                                   ? "prefix_suffix_prefill"
                                   : "prefix_cache_miss_prefill")))
                {
                    return false;
                }
                return complete_distributed_prefill();
            }
            catch (const std::exception &e)
            {
                return setError(std::string("Prefill with prefix cache failed: ") + e.what());
            }
        }

        // Run forward pass
        try
        {
            const int stable_llep_prefill_window =
                stableRoutedPrefillAssignmentWindowTokens(
                    /*prefix_cache_block_size=*/0,
                    /*prefix_cache_enabled=*/false);
            if (stable_llep_prefill_window < 0)
            {
                std::ostringstream error;
                error << "Routed-prefill assignment window cannot be negative"
                      << " plan_prefill_window="
                      << plan_.runtime.moe_routed_prefill.assignment_window_tokens
                      << " config_prefill_window="
                      << config_.moe_routed_prefill.assignment_window_tokens;
                return setError(error.str());
            }
            if (!forwardPrefillTokens(prompt_tokens.data(),
                                      static_cast<int>(prompt_tokens.size()),
                                      "Forward pass failed during prefill",
                                      stable_llep_prefill_window))
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

        if (!initializeDecodeTransactionPlanningPositionAfterPrefill(
                static_cast<int>(prompt_tokens.size()),
                "ordinary_prefill"))
        {
            return false;
        }

        return complete_distributed_prefill();
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

        if (plan_.usesLocalPP() ||
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

        if (plan_.usesLocalPP() ||
            plan_.usesGlobalTP() ||
            plan_.usesPipelineParallel())
        {
            return setError(
                "Request-batched prefill is currently implemented for "
                "SingleDevice and LocalTP runners");
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
        for (size_t request_index = 0;
             request_index < token_batches.size();
             ++request_index)
        {
            const std::vector<int32_t> &tokens = token_batches[request_index];
            if (tokens.empty())
                return setError("Request-batched prefill received an empty prompt");

            converted.emplace_back(tokens.begin(), tokens.end());

            BatchedDecodeRequestState state;
            state.last_token = tokens.back();
            state.logical_tokens = static_cast<int>(tokens.size());
            state.stochastic_position_seed =
                resolveRequestBatchedStochasticSeed(
                    active_sampling_params_, request_index);
            state.prefill_logits_ready = true;
            state.sampler = Sampler(active_sampling_params_.seed);
            next_states.push_back(std::move(state));
        }

        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        mtp_stats_ = {};
        device_generation_terminal_ledger_authoritative_ = false;
        device_generation_embedded_moe_maintenance_pending_ack_ = false;
        device_generation_admission_.reset();
        prefix_request_summary_ = {};
        ready_mtp_condition_.reset();
        prefill_logits_ready_ = false;
        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        decode_transaction_planning_position_.reset();
        last_token_ = next_states.front().last_token;

        if (!runner_->forward_batch(converted))
        {
            device_generation_admission_.reset();
            clearBatchedDecodeState();
            return setError("Forward batch failed during request-batched prefill");
        }
        if (runner_->primaryDeviceId().is_gpu() &&
            !device_generation_admission_.openAfterPrefill())
        {
            clearBatchedDecodeState();
            return setError(
                "Request-batched GPU prefill could not open its unique device-generation admission boundary");
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
        {
            /*
             * Mixed ownership is a supported transient transaction: this call
             * flushes each already sampled resident ready token without running
             * model work, then the following call rejoins grouped verification.
             * Advertising false here would make a healthy device-owned state
             * look like a missing implementation to the serving layer.
            */
            return true;
        }
        if (has_ready_prefill_logits)
            return true;

        const MTPRuntimeConfig &mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        const int draft_depth = effectiveMTPMaxDraftDepth(mtp);
        /*
         * Greedy sampling is the deterministic specialization of the
         * speculative verifier contract. A deployment may keep the verifier in
         * SpeculativeSampling mode while selecting temperature=0 for one
         * request; that does not require a different grouped graph or outcome
         * publisher. Key the greedy lane from the active sampling contract,
         * rather than rejecting an otherwise executable resident transaction
         * because of the configured stochastic superset.
         */
        const bool greedy_batch_verify =
            active_sampling_params_.is_greedy();
        const bool stochastic_batch_verify =
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            !active_sampling_params_.is_greedy() &&
            !active_sampling_params_.has_penalties() &&
            runner_->supportsDeviceStochasticMTPVerification();
        /*
         * CPU request batching may still publish the grouped transaction plan
         * directly.  GPU request batching must publish from compact resident
         * outcome rows and must feed verifier rows from device draft-token slots.
         */
        const bool gpu_request_batch =
            runner_->primaryDeviceId().is_gpu();
        const bool direct_publication_ok =
            !gpu_request_batch &&
            runner_->supportsMTPSpecStatePublication();
        const bool resident_publication_ok =
            gpu_request_batch &&
            runner_->supportsDeviceResidentMTPSpecStatePublication() &&
            runner_->supportsMTPDeviceDraftTokenInput();
        const bool chained_ok =
            draft_depth <= 1 || runner_->supportsChainedMTPDrafts();
        return has_verifier_continuation &&
               !has_completed_requests &&
               mtp.enabled &&
               (greedy_batch_verify || stochastic_batch_verify) &&
               draft_depth >= 1 &&
               chained_ok &&
               (direct_publication_ok || resident_publication_ok);
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

        const bool gpu_request_batch = runner_->primaryDeviceId().is_gpu();
        const int padded_seq_len = runner_->padded_seq_len();
        if (padded_seq_len <= 0)
        {
            batch_result.error =
                "decodeStepBatch() requires positive request-batch geometry";
            return batch_result;
        }

        std::vector<int> planning_sequence_lengths(
            static_cast<size_t>(request_batch),
            0);
        if (gpu_request_batch)
        {
            for (int request = 0; request < request_batch; ++request)
            {
                const int logical_tokens =
                    batched_request_states_[static_cast<size_t>(request)]
                        .logical_tokens;
                if (logical_tokens <= 0)
                {
                    batch_result.error =
                        "decodeStepBatch() GPU request batch has invalid immutable prefill geometry";
                    return batch_result;
                }
                planning_sequence_lengths[static_cast<size_t>(request)] =
                    logical_tokens;
            }
        }
        else
        {
            const std::vector<int> &sequence_lengths =
                runner_->sequence_lengths();
            if (static_cast<int>(sequence_lengths.size()) < request_batch)
            {
                batch_result.error =
                    "decodeStepBatch() CPU request batch requires per-request sequence metadata";
                return batch_result;
            }
            std::copy_n(
                sequence_lengths.begin(),
                request_batch,
                planning_sequence_lengths.begin());
        }
        DeviceResidentLogicalSequenceStateHandle resident_batch_state =
            runner_->deviceResidentLogicalSequenceState();
        if (!gpu_request_batch && resident_batch_state.valid())
        {
            bool shadows_valid = true;
            for (int request = 0; request < request_batch; ++request)
            {
                const int logical_tokens =
                    batched_request_states_[static_cast<size_t>(request)]
                        .logical_tokens;
                if (logical_tokens < 0)
                {
                    shadows_valid = false;
                    break;
                }
                planning_sequence_lengths[static_cast<size_t>(request)] =
                    logical_tokens;
            }
            if (!shadows_valid)
            {
                batch_result.error =
                    "decodeStepBatch() has resident logical state but missing "
                    "transaction-derived request positions";
                return batch_result;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_planning_resident_shadow_reads",
                1.0,
                "decode",
                {},
                {{"request_count", std::to_string(request_batch)}});
        }

        batch_result.requests.resize(static_cast<size_t>(request_batch));
        bool initial_resident_prefill_transaction = false;

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
            if (gpu_request_batch)
            {
                batch_result.error =
                    "decodeStepBatch() GPU request batch reached mixed host-stepped state; complete resident generation must terminate in one parent launch";
                return batch_result;
            }
            /*
             * A mixed batch is a normal speculative-decode outcome.  A lane
             * that accepted every draft owns a token sampled from its terminal
             * verifier row, while a rejecting lane owns a correction condition
             * and must enter another verifier transaction.  Returning the ready
             * token at the end of the previous call made that lane produce one
             * more response token than scalar decode.
             *
             * Preserve scalar API boundaries by flushing only already-sampled
             * ready tokens on this call.  Lanes without a ready token remain
             * unchanged and return an empty response for this turn.  After the
             * flush every live lane owns an unforwarded condition token, so the
             * next call rejoins the single grouped, device-resident verifier
             * path without serial replay or host-side model execution.
             */
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
                    continue;
                if (!state.ready_sampled_token.has_value() ||
                    !state.ready_sampled_params.has_value())
                {
                    batch_result.error =
                        "decodeStepBatch() mixed ready lane is missing its "
                        "device-sampled token transaction";
                    return batch_result;
                }
                if (!samplingParamsEqual(
                        *state.ready_sampled_params,
                        active_sampling_params_))
                {
                    batch_result.error =
                        "decodeStepBatch() mixed ready token was sampled with "
                        "different sampling parameters";
                    return batch_result;
                }

                const int32_t token = *state.ready_sampled_token;
                state.prefill_logits_ready = false;
                state.ready_sampled_token.reset();
                state.ready_sampled_params.reset();
                state.last_token = token;
                state.sampler.record_token(token);

                request_result.tokens.push_back(token);
                request_result.is_complete =
                    std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
                    stop_tokens_.end();
                state.is_complete = request_result.is_complete;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_mixed_ready_token_flushes",
                1.0,
                "decode",
                {},
                {{"request_count", std::to_string(request_batch)}});
            return batch_result;
        }

        if (has_ready_prefill_logits && gpu_request_batch)
        {
            if (has_verifier_continuation || has_completed_requests)
            {
                batch_result.error =
                    "decodeStepBatch() GPU prefill admission requires every request row to enter transaction zero together";
                return batch_result;
            }
            for (const BatchedDecodeRequestState &state :
                 batched_request_states_)
            {
                if (!state.prefill_logits_ready ||
                    state.ready_sampled_token.has_value() ||
                    state.ready_sampled_params.has_value())
                {
                    batch_result.error =
                        "decodeStepBatch() GPU prefill boundary contains a forbidden host-sampled token shadow";
                    return batch_result;
                }
            }
            if (!admitRequestBatchDeviceResidentGeneration(request_batch))
            {
                batch_result.error =
                    last_error_.empty()
                        ? "decodeStepBatch() could not admit request-batched device generation"
                        : last_error_;
                return batch_result;
            }

            std::vector<uint64_t> position_seeds;
            if (!active_sampling_params_.is_greedy())
            {
                position_seeds.reserve(static_cast<size_t>(request_batch));
                for (const BatchedDecodeRequestState &state :
                     batched_request_states_)
                {
                    position_seeds.push_back(state.stochastic_position_seed);
                }
            }
            if (!runner_
                     ->publishMainLogitsBatchSamplesToDeviceResidentState(
                         request_batch,
                         active_sampling_params_,
                         position_seeds.empty() ? nullptr
                                                : position_seeds.data()))
            {
                batch_result.error =
                    "decodeStepBatch() could not publish request-batched prefill samples into resident state";
                return batch_result;
            }

            resident_batch_state =
                runner_->deviceResidentLogicalSequenceState();
            if (!resident_batch_state.valid() ||
                resident_batch_state.request_count != request_batch)
            {
                batch_result.error =
                    "decodeStepBatch() request-batched prefill sampler produced no complete resident logical-state mailbox";
                return batch_result;
            }
            for (BatchedDecodeRequestState &state :
                 batched_request_states_)
            {
                state.prefill_logits_ready = false;
            }
            initial_resident_prefill_transaction = true;
            has_ready_prefill_logits = false;
            has_verifier_continuation = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_prefill_to_resident_transaction_zero",
                1.0,
                "decode",
                {},
                {{"request_count", std::to_string(request_batch)},
                 {"host_token_materializations", "0"}});
        }

        if (has_ready_prefill_logits)
        {
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
                else
                {
                    const int logical_length =
                        planning_sequence_lengths[static_cast<size_t>(request)];
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
        /*
         * Keep execution symmetric with supportsDecodeStepBatch(): greedy
         * sampling uses the deterministic branch of either verifier mode.
         * SpeculativeSampling remains relevant only for non-greedy sampling,
         * where device-resident probability rejection is required.
         */
        const bool greedy_batch_verify =
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
        if (draft_depth < 1)
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires --mtp-draft-tokens >= 1";
            return batch_result;
        }
        if (draft_depth > 1 && !runner_->supportsChainedMTPDrafts())
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires batched chained MTP draft support for depth > 1";
            return batch_result;
        }
        /*
         * CPU request batches can publish the grouped transaction plan directly.
         * GPU request batches publish compact resident verifier outcomes and feed
         * verifier inputs from device draft-token slots; the direct host-plan
         * publisher is intentionally not considered a GPU continuation path.
         */
        const bool request_batch_direct_publication_ok =
            !gpu_request_batch &&
            runner_->supportsMTPSpecStatePublication();
        const bool request_batch_resident_publication_ok =
            gpu_request_batch &&
            runner_->supportsDeviceResidentMTPSpecStatePublication() &&
            runner_->supportsMTPDeviceDraftTokenInput();
        if (!request_batch_direct_publication_ok &&
            !request_batch_resident_publication_ok)
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires CPU grouped publication or GPU device-resident "
                "compact publication with device draft-token input";
            return batch_result;
        }

        if (!runner_->ensureMTPCheckpointTerminalHidden())
        {
            batch_result.error =
                "decodeStepBatch() could not materialize MTP checkpoint terminal hidden";
            return batch_result;
        }
        /*
         * Request-batched planning_sequence_lengths is the scheduler-owned
         * logical cursor set.  The current rollback API archives sequence zero;
         * pass that lane's exact cursor explicitly so a GPU child never consults
         * a lagging host mirror while sizing the device-resident checkpoint.
         */
        PrefixStateSnapshot checkpoint =
            runner_->captureLivePrefixCheckpoint(
                PrefixCheckpointCaptureRequest{
                    .sequence_index = 0,
                    .logical_cached_tokens =
                        planning_sequence_lengths.front()});
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

        bool request_batch_condition_advanced = false;
        if (gpu_request_batch && !initial_resident_prefill_transaction)
        {
            std::vector<uint64_t> condition_position_seeds;
            if (stochastic_batch_verify)
            {
                condition_position_seeds.resize(
                    static_cast<size_t>(request_batch),
                    0);
                for (int request = 0; request < request_batch; ++request)
                {
                    condition_position_seeds[static_cast<size_t>(request)] =
                        batched_request_states_[static_cast<size_t>(request)]
                            .stochastic_position_seed;
                }
            }

            /*
             * Match scalar decode's transaction boundary before drafting. The
             * main graph consumes the token returned by the preceding call,
             * samples the first new target token, and republishes that token as
             * the MTP sidecar condition. Mutable token identities and logical
             * positions never cross the host while this transition executes.
             */
            if (!runner_->advanceMTPRequestBatchConditionOnDevice(
                    resident_batch_state,
                    request_batch,
                    active_sampling_params_,
                    condition_position_seeds.empty()
                        ? nullptr
                        : condition_position_seeds.data()))
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() could not advance the resident main "
                    "condition batch before MTP drafting");
            }
            resident_batch_state =
                runner_->deviceResidentLogicalSequenceState();
            if (!resident_batch_state.valid() ||
                resident_batch_state.request_count < request_batch)
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() condition advance did not publish a "
                    "complete resident logical-state mailbox");
            }
            for (int request = 0; request < request_batch; ++request)
            {
                ++planning_sequence_lengths[static_cast<size_t>(request)];
            }
            request_batch_condition_advanced = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_condition_forward_transactions",
                1.0,
                "decode",
                {},
                {{"requests", std::to_string(request_batch)},
                 {"sampling",
                  stochastic_batch_verify ? "stochastic" : "greedy"},
                 {"state_owner", "device_logical_state_mailbox"}});
        }

        std::vector<int32_t> condition_tokens;
        std::vector<int> position_ids;
        std::vector<int32_t> sidecar_drafts;
        std::vector<std::vector<int32_t>> request_drafts(
            static_cast<size_t>(request_batch));
        const bool use_request_batch_device_draft_slots =
            gpu_request_batch &&
            runner_->supportsMTPDeviceDraftTokenInput();
        const bool use_request_batch_resident_condition_tokens =
            gpu_request_batch &&
            resident_batch_state.valid() &&
            resident_batch_state.request_count >= request_batch;
        if (gpu_request_batch &&
            (!use_request_batch_device_draft_slots ||
             !use_request_batch_resident_condition_tokens))
        {
            return fail_after_checkpoint(
                "decodeStepBatch() GPU request-batched MTP requires a live "
                "device-resident logical-state mailbox and device draft slots");
        }
        if (!use_request_batch_device_draft_slots)
        {
            condition_tokens.assign(
                static_cast<size_t>(request_batch),
                kMTPSpecDecodeInvalidToken);
            position_ids.assign(
                static_cast<size_t>(request_batch),
                -1);
            sidecar_drafts.assign(
                static_cast<size_t>(request_batch),
                kMTPSpecDecodeInvalidToken);
            for (int request = 0; request < request_batch; ++request)
            {
                const BatchedDecodeRequestState &state =
                    batched_request_states_[static_cast<size_t>(request)];
                if (state.is_complete)
                {
                    batch_result.requests[static_cast<size_t>(request)].is_complete = true;
                    continue;
                }
                const int logical_length =
                    planning_sequence_lengths[static_cast<size_t>(request)];
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
             * CPU grouped sidecars still consume explicit host rows. Completed
             * rows receive a harmless value solely to preserve the dense grouped
             * shape; the GPU lane has no corresponding host token or position row.
             */
            for (int request = 0; request < request_batch; ++request)
            {
                if (condition_tokens[static_cast<size_t>(request)] ==
                    kMTPSpecDecodeInvalidToken)
                {
                    condition_tokens[static_cast<size_t>(request)] = 0;
                    position_ids[static_cast<size_t>(request)] =
                        std::max(
                            0,
                            planning_sequence_lengths[static_cast<size_t>(request)]);
                }
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
                    ? runner_->forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
                          resident_batch_state,
                          request_batch,
                          /*first_draft_slot=*/0,
                          /*slot_stride=*/draft_depth)
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
        if (!use_request_batch_device_draft_slots)
        {
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
        }

        for (int draft_index = 1; draft_index < draft_depth; ++draft_index)
        {
            if (!use_request_batch_device_draft_slots)
            {
                for (int request = 0; request < request_batch; ++request)
                {
                    condition_tokens[static_cast<size_t>(request)] =
                        request_drafts[static_cast<size_t>(request)].back();
                    position_ids[static_cast<size_t>(request)] =
                        planning_sequence_lengths[static_cast<size_t>(request)] +
                        draft_index;
                    sidecar_drafts[static_cast<size_t>(request)] =
                        kMTPSpecDecodeInvalidToken;
                }
            }

            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "request_batch_chained_sidecar_forward",
                "decode",
                {},
                {{"draft_index", std::to_string(draft_index)}});
            const bool chained_sidecar_ok =
                use_request_batch_device_draft_slots
                    ? runner_->forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
                          resident_batch_state,
                          request_batch,
                          /*first_condition_slot=*/draft_index - 1,
                          /*condition_slot_stride=*/draft_depth,
                          /*position_offset=*/draft_index,
                          /*first_draft_slot=*/draft_index,
                          /*draft_slot_stride=*/draft_depth)
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
            if (!use_request_batch_device_draft_slots)
            {
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
        }

        if (!use_request_batch_device_draft_slots &&
            !runner_->flushPendingMTPWork())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() request-batched MTP sidecar flush failed");
        }
        if (use_request_batch_device_draft_slots)
        {
            /*
             * Every proposal column records a readiness event. Chained sidecars
             * and verifier token materialization queue stream waits on those
             * events, so a CPU-visible stream flush would be both redundant and
             * a serialization bug in the fully device-resident lane.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_sidecar_host_flushes_avoided",
                1.0,
                "decode",
                {},
                {{"request_count", std::to_string(request_batch)},
                 {"draft_depth", std::to_string(draft_depth)}});
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
            if (!use_request_batch_device_draft_slots)
            {
                const int32_t draft =
                    sidecar_drafts[static_cast<size_t>(request)];
                if (draft < 0)
                {
                    return fail_after_checkpoint(
                        "decodeStepBatch() request-batched MTP sidecar produced "
                        "an invalid draft token");
                }
            }

            MTPSpecSchedulableRequest pending;
            pending.request_id = request;
            pending.ready = true;
            pending.mode = stochastic_batch_verify
                               ? MTPSpecRequestBatchMode::STOCHASTIC
                               : MTPSpecRequestBatchMode::GREEDY;
            pending.verifier_input =
                use_request_batch_device_draft_slots
                    ? MTPSpecVerifierInputPlacement::DEVICE_TOKEN_ROW
                    : MTPSpecVerifierInputPlacement::HOST_TOKENS;
            pending.compatibility_key = compatibility_key;
            pending.vocab_size = vocab;
            pending.base_cached_tokens =
                static_cast<int32_t>(
                    planning_sequence_lengths[static_cast<size_t>(request)]);
            pending.requires_shifted_kv_publication = true;
            pending.greedy_request.draft_tokens.clear();
            pending.greedy_request.draft_tokens.reserve(
                static_cast<size_t>(draft_depth) + 1u);
            pending.greedy_request.draft_tokens.push_back(
                use_request_batch_device_draft_slots
                    ? 0
                    : state.last_token);
            if (use_request_batch_device_draft_slots)
            {
                /*
                 * The scheduler needs only the verifier row shape. Proposal
                 * values stay exclusively in the request-major device matrix;
                 * zero is a valid non-semantic shadow that is never consumed by
                 * GPU embedding, comparison, or publication kernels.
                 */
                pending.greedy_request.draft_tokens.insert(
                    pending.greedy_request.draft_tokens.end(),
                    static_cast<size_t>(draft_depth),
                    0);
            }
            else
            {
                pending.greedy_request.draft_tokens.insert(
                    pending.greedy_request.draft_tokens.end(),
                    request_drafts[static_cast<size_t>(request)].begin(),
                    request_drafts[static_cast<size_t>(request)].end());
            }
            pending.greedy_request.stop_tokens = stop_tokens_;
            pending.greedy_request.base_sidecar_position =
                planning_sequence_lengths[static_cast<size_t>(request)];
            pending.greedy_request.verifier_path =
                "request_batched_grouped_decode_equivalent_publication";
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
            gpu_request_batch;
        if (use_device_resident_request_batch_publication &&
            !runner_->supportsDeviceResidentMTPSpecStatePublication())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() GPU request batching requires "
                "device-resident MTP state publication");
        }
        if (use_device_resident_request_batch_publication &&
            !runner_->supportsMTPDeviceDraftTokenInput())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() GPU request batching requires "
                "device draft-token verifier input");
        }

        DeviceSpeculativeOutcomeHandle resident_request_batch_outcome;
        bool resident_request_batch_outcome_ready = false;
        int resident_request_batch_verifier_rows = 0;
        std::vector<std::optional<Sampler>> sampled_terminal_samplers(
            static_cast<size_t>(request_batch));

        auto publish = [&](const MTPSpecTransactionBatchPlan &plan,
                           std::string *error) -> bool
        {
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

                /*
                 * The compact outcome is already materialized for the serving
                 * response at this point, so this opt-in trace adds no device
                 * transfer or synchronization. Keeping every publication
                 * decision on one line makes batch-invariance failures
                 * directly comparable with a scalar request: the trace records
                 * the RNG policy, emitted token bytes, accepted/committed rows,
                 * correction token, and terminal ownership together.
                 */
                if (debugEnv().runtime_debug.mtp_publication_diagnostics)
                {
                    std::ostringstream output_tokens;
                    const int output_count = std::clamp(
                        outcomes[i].output_token_count,
                        0,
                        static_cast<int>(outcomes[i].output_tokens.size()));
                    output_tokens << '[';
                    for (int token_index = 0;
                         token_index < output_count;
                         ++token_index)
                    {
                        if (token_index > 0)
                            output_tokens << ',';
                        output_tokens << outcomes[i].output_tokens[
                            static_cast<size_t>(token_index)];
                    }
                    output_tokens << ']';

                    LOG_INFO(
                        "[MTPPublicationDiagnostics] phase=request_batch_stochastic_compact_outcome"
                        << " request=" << request_id
                        << " rows=" << physical_rows
                        << " serial_sample_equivalent="
                        << (outcome_requests[i].serial_sample_equivalent
                                ? "true"
                                : "false")
                        << " vllm_probability_rejection="
                        << (outcome_requests[i].use_vllm_probability_rejection
                                ? "true"
                                : "false")
                        << " output_count=" << output_count
                        << " output_tokens=" << output_tokens.str()
                        << " accepted_prefix="
                        << outcomes[i].accepted_speculative_prefix
                        << " committed_states="
                        << outcomes[i].target_verifier_state_commit_count
                        << " consumed_rows="
                        << outcomes[i].consumed_verifier_rows
                        << " ready_token=" << outcomes[i].ready_token
                        << " rejected_token="
                        << outcomes[i].rejected_verified_token
                        << " all_accepted="
                        << (outcomes[i].all_speculative_accepted
                                ? "true"
                                : "false")
                        << " sampled_terminal="
                        << (outcomes[i].sampled_terminal ? "true" : "false")
                        << " stopped="
                        << (outcomes[i].stopped_on_output ? "true" : "false"));
                }

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
        std::vector<int32_t> scheduled_base_cached_tokens;
        std::vector<MTPDecodeCatchupGreedyResult> catchup_results;

        if (!stochastic_batch_verify)
        {
            if (use_device_resident_request_batch_publication)
            {
                auto release_and_fail =
                    [&](MTPOwnedDeviceOutcomeBatchTransactionResult &tx,
                        std::string message) -> GenerationBatchResult
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

                auto produce_greedy_resident_outcomes =
                    [&](const MTPSpecRequestBatch &scheduled_batch,
                        DeviceSpeculativeOutcomeHandle *resident_handle,
                        std::string *error) -> bool
                {
                    auto set_producer_error = [&](std::string message) -> bool
                    {
                        if (error)
                            *error = std::move(message);
                        return false;
                    };

                    if (!resident_handle)
                        return set_producer_error("greedy resident outcome handle is null");
                    *resident_handle = {};
                    if (!scheduled_batch.ok)
                        return set_producer_error("scheduled greedy batch is invalid");
                    if (scheduled_batch.request_count <= 0)
                        return set_producer_error("scheduled greedy batch is empty");

                    std::vector<MTPSpecDecodeVerifierDraftRequest>
                        verifier_requests;
                    verifier_requests.reserve(
                        scheduled_batch.greedy_requests.size());
                    for (size_t i = 0;
                         i < scheduled_batch.greedy_requests.size();
                         ++i)
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
                            std::string("greedy request-batch verifier input plan failed: ") +
                            verifier_input_plan.error);
                    }
                    if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                    {
                        return set_producer_error(
                            "greedy request-batch verifier row metadata is malformed");
                    }

                    const int padded_seq_len =
                        scheduled_batch.shape.max_draft_tokens;
                    std::vector<DeviceMTPVerifierInputBatchRequest>
                        token_batch_requests;
                    token_batch_requests.reserve(
                        scheduled_batch.greedy_requests.size());
                    for (size_t i = 0;
                         i < scheduled_batch.greedy_requests.size();
                         ++i)
                    {
                        const int request_id = scheduled_batch.request_ids[i];
                        if (request_id < 0 || request_id >= request_batch)
                        {
                            return set_producer_error(
                                "greedy request-batch verifier returned an out-of-range request id");
                        }
                        const MTPDecodeCatchupGreedyRequest &request =
                            scheduled_batch.greedy_requests[i];
                        const int verifier_token_count =
                            static_cast<int>(request.draft_tokens.size());
                        if (verifier_token_count <= 1 ||
                            verifier_token_count > padded_seq_len)
                        {
                            return set_producer_error(
                                "greedy request-batch verifier request shape is invalid");
                        }

                        DeviceMTPVerifierInputBatchRequest descriptor;
                        descriptor.request_id = request_id;
                        descriptor.first_token = request.draft_tokens.front();
                        if (use_request_batch_resident_condition_tokens)
                        {
                            descriptor.first_token_from_device = true;
                            descriptor.first_target_sample_slot = request_id;
                            if (!resident_batch_state.coversRequest(request_id))
                            {
                                return set_producer_error(
                                    "greedy request-batch verifier has no resident condition-token row for request");
                            }
                        }
                        else
                        {
                            descriptor.first_token_from_device = false;
                            descriptor.first_target_sample_slot = -1;
                        }
                        descriptor.first_draft_slot =
                            request_id * draft_depth;
                        descriptor.draft_token_count =
                            verifier_token_count - 1;
                        descriptor.total_verifier_input_tokens =
                            verifier_token_count;
                        token_batch_requests.push_back(descriptor);
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
                            "greedy request-batch verifier could not enable row-indexed logits");
                    }
                    row_indexed_enabled = true;
                    if (!runner_->setMTPSpecVerifierInputPlan(
                            verifier_input_plan))
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? "greedy request-batch verifier could not install row plan"
                                : "greedy request-batch verifier could not install row plan and cleanup failed");
                    }
                    /*
                     * Installing a verifier plan defines a new graph transaction
                     * and deliberately invalidates any pending device-token
                     * composition. Stage the request-major token matrix only
                     * after that transaction exists, so the verifier forward can
                     * materialize the same rows later consumed by the compact
                     * outcome reducer.
                     */
                    const void *device_token_batch =
                        runner_->prepareMTPVerifierInputTokenBatchOnDevice(
                            token_batch_requests.data(),
                            static_cast<int>(token_batch_requests.size()),
                            padded_seq_len);
                    if (!device_token_batch)
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? "greedy request-batch verifier could not prepare device token matrix"
                                : "greedy request-batch verifier could not prepare device token matrix and cleanup failed");
                    }
                    if (!runner_->setComputeAllPositionLogits(true))
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? "greedy request-batch verifier could not enable all-position logits"
                                : "greedy request-batch verifier could not enable all-position logits and cleanup failed");
                    }
                    all_position_enabled = true;

                    MTPVerifierForwardExecutionOptions forward_options;
                    forward_options.device_token_ids = device_token_batch;
                    forward_options.allow_batched_host_forward = false;
                    {
                        ScopedMTPAllPositionVerifierSyncDeferral
                            verifier_sync_deferral(
                                runner_.get(),
                                runner_->primaryDeviceId().is_gpu());
                        PerfStatsCollector::ScopedTimer verifier_timer(
                            "mtp",
                            "request_batch_greedy_verifier_forward_device_tokens",
                            "decode",
                            {},
                            {{"requests",
                              std::to_string(
                                  scheduled_batch.request_count)},
                             {"draft_depth", std::to_string(draft_depth)}});
                        const MTPVerifierForwardExecutionResult forward =
                            executeMTPSpecVerifierForward(
                                *runner_,
                                verifier_input_plan,
                                forward_options);
                        if (!forward.ok)
                        {
                            const bool cleanup_ok = cleanup_row_modes();
                            return set_producer_error(
                                cleanup_ok
                                    ? std::string("greedy request-batch verifier forward failed: ") +
                                          forward.error
                                    : std::string("greedy request-batch verifier forward failed and cleanup failed: ") +
                                          forward.error);
                        }
                    }

                    if (!cleanup_row_modes())
                    {
                        return set_producer_error(
                            "greedy request-batch verifier could not disable row-indexed logits");
                    }

                    std::vector<DeviceGreedyBatchOutcomeRequest>
                        outcome_requests;
                    outcome_requests.reserve(
                        scheduled_batch.greedy_requests.size());
                    for (size_t i = 0;
                         i < scheduled_batch.greedy_requests.size();
                         ++i)
                    {
                        const MTPDecodeCatchupGreedyRequest &request =
                            scheduled_batch.greedy_requests[i];
                        const int verifier_token_count =
                            static_cast<int>(request.draft_tokens.size());
                        if (static_cast<int>(
                                verifier_input_plan.query_start_locs.size()) <=
                            static_cast<int>(i))
                        {
                            return set_producer_error(
                                "greedy request-batch verifier query row metadata is undersized");
                        }

                        DeviceGreedyBatchOutcomeRequest descriptor;
                        descriptor.request_id =
                            scheduled_batch.request_ids[i];
                        descriptor.first_target_row =
                            verifier_input_plan.query_start_locs[i];
                        descriptor.verifier_token_count =
                            verifier_token_count;
                        descriptor.token_row_stride = padded_seq_len;
                        descriptor.token_row_offset =
                            static_cast<int>(i) * padded_seq_len;
                        descriptor.first_token =
                            request.draft_tokens.front();
                        descriptor.stop_token_count =
                            static_cast<int>(stop_tokens_.size());
                        if (descriptor.stop_token_count >
                            static_cast<int>(
                                sampling_math::kSpeculativeBatchMaxStopTokens))
                        {
                            return set_producer_error(
                                "greedy request-batch stop-token count exceeds device summary capacity");
                        }
                        for (int stop_index = 0;
                             stop_index < descriptor.stop_token_count;
                             ++stop_index)
                        {
                            descriptor.stop_tokens[
                                static_cast<size_t>(stop_index)] =
                                stop_tokens_[static_cast<size_t>(stop_index)];
                        }
                        outcome_requests.push_back(descriptor);
                    }

                    if (!runner_->verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
                            outcome_requests.data(),
                            static_cast<int>(outcome_requests.size()),
                            resident_handle))
                    {
                        return set_producer_error(
                            "greedy request-batch resident device outcome verifier failed");
                    }
                    return resident_handle->valid();
                };

                MTPOwnedDeviceOutcomeBatchTransactionResult tx;
                tx.scheduled_batch = owner.scheduleNextBatch(scheduler);
                if (!tx.scheduled_batch.ok)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched greedy "
                                    "verifier scheduling failed: ") +
                        tx.scheduled_batch.error);
                }

                std::string produce_error;
                tx.produced = produce_greedy_resident_outcomes(
                    tx.scheduled_batch,
                    &resident_request_batch_outcome,
                    &produce_error);
                resident_request_batch_outcome_ready =
                    tx.produced && resident_request_batch_outcome.valid();
                resident_request_batch_verifier_rows =
                    tx.scheduled_batch.shape.max_draft_tokens;
                if (!tx.produced)
                {
                    std::string message =
                        "decodeStepBatch() request-batched greedy resident "
                        "verifier production failed";
                    if (!produce_error.empty())
                    {
                        message += ": ";
                        message += produce_error;
                    }
                    return release_and_fail(tx, std::move(message));
                }
                if (!resident_request_batch_outcome_ready ||
                    resident_request_batch_verifier_rows <= 0)
                {
                    return release_and_fail(
                        tx,
                        "decodeStepBatch() request-batched greedy resident "
                        "verifier produced no valid device outcome");
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = resident_request_batch_outcome;
                publication_request.max_state_commit_rows =
                    resident_request_batch_verifier_rows;
                publication_request.publish_mtp_shifted_kv =
                    tx.scheduled_batch.requires_shifted_kv_publication;

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer publication_timer(
                        "mtp",
                        "request_batch_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"sampling", "greedy"},
                         {"request_count",
                          std::to_string(publication_request.requestCount())},
                         {"logical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .logicalVerifierRowsPerRequest())},
                         {"physical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .physicalVerifierRowsPerRequest())}});
                    tx.published =
                        runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                            publication_request,
                            &publication_error);
                }
                if (!tx.published)
                {
                    std::string message =
                        "decodeStepBatch() request-batched greedy resident "
                        "state publication failed";
                    if (!publication_error.empty())
                    {
                        message += ": ";
                        message += publication_error;
                    }
                    return release_and_fail(tx, std::move(message));
                }

                const DeviceResidentLogicalSequenceStateHandle logical_state =
                    runner_->deviceResidentLogicalSequenceState();
                if (!logical_state.valid() ||
                    logical_state.request_count <
                        tx.scheduled_batch.request_count)
                {
                    return release_and_fail(
                        tx,
                        "decodeStepBatch() request-batched greedy resident "
                        "publication produced no valid logical-state mailbox");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "request_batch_resident_plan_checks",
                    1.0,
                    "decode",
                    {},
                    {{"request_count",
                      std::to_string(tx.scheduled_batch.request_count)},
                     {"sampling", "greedy"}});

                std::string commit_error;
                tx.committed = owner.commitInFlightBatch(&commit_error);
                if (!tx.committed)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched greedy "
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
                      std::to_string(publication_request.requestCount())},
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())},
                     {"sampling", "greedy"}});
                tx.ok = true;
                return completeDeviceResidentBatchGeneration(
                    publication_request,
                    DeviceGenerationSamplingMode::Greedy,
                    draft_depth,
                    draft_depth,
                    std::move(batch_result));
            }
            else
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
                scheduled_base_cached_tokens =
                    tx.scheduled_batch.base_cached_tokens;
                catchup_results = tx.transaction.catchup.results;
            }
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

                const int padded_seq_len =
                    scheduled_batch.shape.max_draft_tokens;
                std::vector<DeviceMTPVerifierInputBatchRequest>
                    token_batch_requests;
                const void *device_token_batch = nullptr;
                if (use_device_resident_request_batch_publication)
                {
                    token_batch_requests.reserve(
                        scheduled_batch.greedy_requests.size());
                    for (size_t i = 0;
                         i < scheduled_batch.greedy_requests.size();
                         ++i)
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
                        if (verifier_token_count <= 1 ||
                            verifier_token_count > padded_seq_len)
                        {
                            return set_producer_error(
                                "stochastic request-batch verifier request shape is invalid");
                        }

                        DeviceMTPVerifierInputBatchRequest descriptor;
                        descriptor.request_id = request_id;
                        descriptor.first_token = request.draft_tokens.front();
                        if (use_request_batch_resident_condition_tokens)
                        {
                            descriptor.first_token_from_device = true;
                            descriptor.first_target_sample_slot = request_id;
                            if (!resident_batch_state.coversRequest(request_id))
                            {
                                return set_producer_error(
                                    "stochastic request-batch verifier has no resident condition-token row for request");
                            }
                        }
                        else
                        {
                            descriptor.first_token_from_device = false;
                            descriptor.first_target_sample_slot = -1;
                        }
                        descriptor.first_draft_slot =
                            request_id * draft_depth;
                        descriptor.draft_token_count =
                            verifier_token_count - 1;
                        descriptor.total_verifier_input_tokens =
                            verifier_token_count;
                        token_batch_requests.push_back(descriptor);
                    }
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
                if (use_device_resident_request_batch_publication)
                {
                    /*
                     * setMTPSpecVerifierInputPlan() starts the authoritative
                     * verifier transaction and clears stale token-composition
                     * work. Publish this transaction first, then stage the
                     * device-owned request rows that its forward will consume.
                     */
                    device_token_batch =
                        runner_->prepareMTPVerifierInputTokenBatchOnDevice(
                            token_batch_requests.data(),
                            static_cast<int>(token_batch_requests.size()),
                            padded_seq_len);
                    if (!device_token_batch)
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? "stochastic request-batch verifier could not prepare device token matrix"
                                : "stochastic request-batch verifier could not prepare device token matrix and cleanup failed");
                    }
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
                        use_device_resident_request_batch_publication
                            ? "request_batch_stochastic_verifier_forward_device_tokens"
                            : "request_batch_stochastic_verifier_forward",
                        "decode",
                        {},
                        {{"requests", std::to_string(scheduled_batch.request_count)},
                         {"draft_depth", std::to_string(draft_depth)}});
                    MTPVerifierForwardExecutionOptions forward_options;
                    forward_options.device_token_ids = device_token_batch;
                    forward_options.allow_batched_host_forward =
                        !use_device_resident_request_batch_publication;
                    const MTPVerifierForwardExecutionResult forward =
                        executeMTPSpecVerifierForward(
                            *runner_,
                            verifier_input_plan,
                            forward_options);
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
                        compare_rows > draft_depth ||
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
                    const int first_draft_slot =
                        use_request_batch_device_draft_slots
                            ? request_id * draft_depth
                            : next_draft_slot;
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
                    const uint64_t request_position_seed =
                        batched_request_states_[static_cast<size_t>(request_id)]
                            .stochastic_position_seed;
                    DeviceStochasticBatchOutcomeRequest descriptor;
                    if (!descriptor.ensureHostRowCapacity(compare_rows))
                    {
                        return set_producer_error(
                            "stochastic request-batch verifier could not size its host diagnostic descriptor");
                    }
                    descriptor.request_id = request_id;
                    descriptor.first_target_slot = first_compact_row;
                    descriptor.first_draft_slot = first_draft_slot;
                    descriptor.row_count = compare_rows;
                    descriptor.first_token = request.draft_tokens.front();
                    if (use_device_resident_request_batch_publication)
                    {
                        descriptor.first_token_from_device = true;
                        descriptor.first_target_sample_slot = -1;
                        descriptor.token_row_offset =
                            static_cast<int>(i) * padded_seq_len;
                        descriptor.token_row_stride = padded_seq_len;
                    }
                    else
                    {
                        descriptor.first_token_from_device = false;
                        descriptor.first_target_sample_slot = -1;
                        descriptor.token_row_offset = -1;
                        descriptor.token_row_stride = 0;
                    }
                    descriptor.bonus_target_slot = first_compact_row + bonus_row;
                    descriptor.use_device_draft_tokens = true;
                    /*
                     * A fixed seed is a pathwise reproducibility contract, not
                     * merely a distributional one. The grouped GPU verifier
                     * therefore samples every target row in parallel with the
                     * same position-keyed draw as serial decode, then compares
                     * those sampled tokens against the resident draft matrix.
                     * vLLM probability rejection remains appropriate for an
                     * unseeded request, where no cross-run token identity is
                     * promised, but it cannot make a seeded RB=N transaction
                     * byte-equivalent to RB=1.
                     */
                    const bool serial_sample_equivalent =
                        active_sampling_params_.seed != 0;
                    descriptor.serial_sample_equivalent =
                        serial_sample_equivalent;
                    descriptor.use_vllm_probability_rejection =
                        !serial_sample_equivalent;
                    const int base_cached_tokens =
                        scheduled_batch.base_cached_tokens[i];
                    if (use_device_resident_request_batch_publication)
                    {
                        /*
                         * GPU stochastic verification owns both the random draw
                         * and its mutable logical-position source. The seed is
                         * immutable request configuration; the position is read
                         * from the publication mailbox by the verifier kernel.
                         * Host arrays remain constructor sentinels and must never
                         * be adopted as launch arguments in this mode.
                         */
                        descriptor.inverse_sample_seed = request_position_seed;
                        descriptor.inverse_sample_first_logical_position = -1;
                        descriptor.derive_thresholds_from_seed = true;
                        descriptor.draw_position_source =
                            DeviceStochasticDrawPositionSource::ResidentLogicalState;
                        descriptor.bonus_threshold = 0.0f;
                    }
                    else
                    {
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
                        }
                        Sampler bonus_sampler = request_sampler;
                        descriptor.bonus_threshold =
                            mtpSpecStochasticThresholdForPosition(
                                active_sampling_params_,
                                bonus_sampler,
                                base_cached_tokens + verifier_token_count,
                                MTPSpecStochasticDrawPurpose::Sample);
                        bonus_samplers[static_cast<size_t>(request_id)] =
                            std::move(bonus_sampler);
                        descriptor.inverse_sample_seed =
                            mtpSpecInverseSampleSeedForThresholds(
                                active_sampling_params_,
                                descriptor.residual_thresholds.data(),
                                static_cast<size_t>(compare_rows));
                        descriptor.inverse_sample_first_logical_position =
                            base_cached_tokens + 1;
                    }
                    for (int row = 0; row < compare_rows; ++row)
                    {
                        descriptor.draft_tokens[static_cast<size_t>(row)] =
                            request.draft_tokens[static_cast<size_t>(row + 1)];
                    }

                    if (use_device_resident_request_batch_publication)
                    {
                        bonus_samplers[static_cast<size_t>(request_id)] =
                            request_sampler;
                    }
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
                publication_request.max_state_commit_rows =
                    resident_request_batch_verifier_rows;
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
                          std::to_string(publication_request.requestCount())},
                         {"logical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .logicalVerifierRowsPerRequest())},
                         {"physical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .physicalVerifierRowsPerRequest())}});
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

                const DeviceResidentLogicalSequenceStateHandle logical_state =
                    runner_->deviceResidentLogicalSequenceState();
                if (!logical_state.valid() ||
                    logical_state.request_count <
                        tx.scheduled_batch.request_count)
                {
                    return release_and_fail(
                        "decodeStepBatch() request-batched stochastic resident "
                        "publication produced no valid logical-state mailbox");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "request_batch_resident_plan_checks",
                    1.0,
                    "decode",
                    {},
                    {{"request_count",
                      std::to_string(tx.scheduled_batch.request_count)},
                     {"state_owner", "device_transaction"}});

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
                      std::to_string(publication_request.requestCount())},
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())}});
                tx.ok = true;
                return completeDeviceResidentBatchGeneration(
                    publication_request,
                    DeviceGenerationSamplingMode::Stochastic,
                    draft_depth,
                    draft_depth,
                    std::move(batch_result));
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
            scheduled_base_cached_tokens =
                tx.scheduled_batch.base_cached_tokens;
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
            if (i >= scheduled_base_cached_tokens.size() ||
                catchup.target_verifier_state_commit_count < 0)
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() request-batched verifier produced "
                    "invalid accepted-state length metadata");
            }
            state.logical_tokens =
                scheduled_base_cached_tokens[i] +
                catchup.target_verifier_state_commit_count;
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_resident_planning_position_commits",
                1.0,
                "decode",
                {},
                {{"request", std::to_string(request)},
                 {"position", std::to_string(state.logical_tokens)}});

            /*
             * A resident GPU batch has already consumed the previously emitted
             * condition token in its grouped main forward. Verifier row zero is
             * therefore the first newly sampled target token and must be
             * returned. CPU's older host-batch lane still includes the prior
             * response token at row zero until it adopts the same grouped
             * condition-forward contract.
             */
            auto new_token_begin =
                catchup.accepted_tokens.begin() +
                (request_batch_condition_advanced ? 0 : 1);
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
            request_result.is_complete =
                catchup.stopped_on_output;
            state.is_complete = request_result.is_complete;
            state.last_token = !request_result.tokens.empty()
                                   ? request_result.tokens.back()
                                   : catchup.accepted_tokens.back();

            if (has_ready_token)
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
        if (effective_max_draft_tokens < 1)
        {
            return "MTP decode requires --mtp-draft-tokens >= 1";
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
            !active_sampling_params_.is_greedy())
        {
            /*
             * CPU Global/NodeLocal TP owns a complete stochastic distribution
             * through graph collectives: the ordinary target verifier gathers its
             * row-indexed LM-head output, and the MTP sidecar now terminates with
             * an explicit allgather of its configured compact row batch. Every
             * rank therefore consumes the same exact full-vocabulary bytes while
             * retaining economical column-sharded head compute.
             *
             * GPU LocalTP takes the lower-latency architecture instead: each child
             * owns a mirrored full-vocabulary head, reduces a compact resident
             * stochastic outcome, and publishes from that same device handle.
             * Arbitrary MPI worlds and GPU GlobalTP still have neither ownership
             * contract and remain hard failures.
             */
            const bool global_or_mpi_tp =
                plan_.usesGlobalTP() ||
                (mpi_ctx_ && mpi_ctx_->world_size() > 1);
            const bool cpu_global_tp_with_gathered_stochastic_logits =
                plan_.usesGlobalTP() &&
                runner_->primaryDeviceId().is_cpu() &&
                runner_->supportsMTPTokenCoordination();
            const bool global_or_mpi_without_stochastic_owner =
                global_or_mpi_tp &&
                !cpu_global_tp_with_gathered_stochastic_logits;
            const bool local_tp_without_mirrored_stochastic =
                plan_.usesLocalTP() &&
                (!runner_->primaryDeviceId().is_gpu() ||
                 !runner_->usesMirroredMTPHeadForVerifier() ||
                 !runner_->supportsDeviceStochasticMTPVerification() ||
                 !runner_->supportsDeviceResidentMTPSpecStatePublication());
            if (global_or_mpi_without_stochastic_owner ||
                local_tp_without_mirrored_stochastic)
            {
                return "MTP speculative sampling verification requires graph-gathered CPU GlobalTP logits or mirrored LocalTP child-resident verifier outcomes";
            }
        }
        if (runner_->primaryDeviceId().is_rocm() && debugEnv().rocm.concurrent_decode)
        {
            return "ROCm MTP decode is incompatible with LLAMINAR_ROCM_CONCURRENT_DECODE";
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
        /*
         * Generic MPI worlds still need an explicit token-coordination
         * contract.  Heterogeneous ExpertOverlay is different: the
         * continuation rank remains the sole token/MTP authority and the
         * remote rank follows immutable retained-graph tickets, so it neither
         * samples nor participates in a token collective.
         */
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1 &&
            !moe_overlay_inference_transaction_coordinator_ &&
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
        return resolveMTPMaximumDraftDepth(mtp);
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
         * Dynamic depth is a request-level scheduling decision.  In NodeTP /
         * GlobalTP every rank must execute the same sidecar/verifier shape in the
         * same order, so the coordinated root's controller decision is the scalar
         * source of truth and broadcast before the step begins.  This mirrors the
         * vLLM-style contract: the speculative batch shape is coordinated once,
         * while tensor data still moves through the graph and collective layers.
         */
        if (mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic &&
            mpi_ctx_ &&
            mpi_ctx_->world_size() > 1 &&
            !moe_overlay_inference_transaction_coordinator_ &&
            !moe_overlay_inference_transaction_follower_)
        {
            int32_t coordinated_depth =
                mpi_ctx_->rank() == mpi_coordinated_root_rank_
                    ? static_cast<int32_t>(depth)
                    : 0;
            mpi_ctx_->broadcast_int32(
                &coordinated_depth, 1, mpi_coordinated_root_rank_);
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

    bool OrchestrationRunner::initializeDecodeTransactionPlanningPositionAfterPrefill(
        int committed_tokens,
        const char *source)
    {
        if (!runner_ || !runner_->primaryDeviceId().is_gpu())
        {
            decode_transaction_planning_position_.reset();
            device_generation_admission_.reset();
            return true;
        }
        if (committed_tokens < 0)
        {
            decode_transaction_planning_position_.reset();
            return setError(
                "GPU MTP prefill produced a negative transaction planning position");
        }

        /*
         * The request scheduler already owns the exact number of prompt tokens
         * committed by prefill. This is control-plane transaction metadata,
         * not a mirror copied back from GPU KV state. Establishing it at the
         * successful prefill boundary gives the first speculative graph the
         * same position that later device publications advance from compact
         * accepted-count metadata.
        */
        decode_transaction_planning_position_ = committed_tokens;
        /*
         * Admission is a property of the successful GPU prefill boundary, not
         * of whichever decode policy happens to consume that boundary. The
         * first decode path closes or consumes it; stochastic MTP additionally
         * publishes the resident generation budget. Keeping MTP policy out of
         * this initializer preserves one transaction-position lifecycle for
         * ordinary, greedy-MTP, and stochastic-MTP decode.
         */
        device_generation_admission_.reset();
        if (!device_generation_admission_.openAfterPrefill())
        {
            return setError(
                "GPU prefill could not open its unique device-generation admission boundary");
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "gpu_decode_transaction_position_initializations",
            1.0,
            "prefill",
            {},
            {{"position", std::to_string(committed_tokens)},
             {"source", source && source[0] != '\0' ? source : "unknown"},
             {"position_owner", "request_scheduler"}});
        return true;
    }

    bool OrchestrationRunner::admitScalarDeviceResidentGeneration(
        bool grouped_device_verify,
        sampling_math::DeviceGenerationLeadingRowDisposition
            initial_leading_row_disposition)
    {
        if (!grouped_device_verify)
            return true;
        if (!device_generation_admission_.awaitsAdmission())
        {
            return setError(
                "GPU grouped MTP generation reached a decode boundary without an admission-ready controller lifecycle");
        }
        if (!runner_ || !decode_transaction_planning_position_.has_value() ||
            *decode_transaction_planning_position_ < 0)
        {
            return setError(
                "GPU grouped MTP generation admission has no initialized prefill boundary");
        }
        int response_budget = decode_step_token_budget_;
        if (response_budget <= 0)
        {
            const int configured_context =
                plan_.runtime.max_seq_len > 0
                    ? plan_.runtime.max_seq_len
                    : config_.max_seq_len;
            response_budget =
                configured_context - *decode_transaction_planning_position_;
        }
        if (response_budget <= 0)
        {
            return setError(
                "GPU grouped MTP generation admission has no positive response capacity");
        }

        const DeviceGenerationAdmissionRequest admission{
            .request_count = 1,
            .max_new_tokens = response_budget,
            .initial_leading_row_disposition =
                initial_leading_row_disposition,
        };
        if (!runner_->beginDeviceResidentGeneration(admission))
        {
            return setError(
                "Failed to admit the device-resident GPU generation response ledger");
        }

        if (!device_generation_admission_.admitController(response_budget))
        {
            return setError(
                "GPU grouped MTP generation could not bind its response budget to the admitted controller");
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_admission_boundaries",
            1.0,
            "decode",
            {},
            {{"request_count", "1"},
             {"response_budget", std::to_string(response_budget)},
             {"initial_leading_committed_output_count",
              std::to_string(
                  sampling_math::
                      device_generation_leading_committed_output_count(
                          initial_leading_row_disposition))},
             {"boundary", "first_scalar_grouped_mtp_decode"}});
        return true;
    }

    bool OrchestrationRunner::admitRequestBatchDeviceResidentGeneration(
        int request_count)
    {
        if (!runner_ || !runner_->primaryDeviceId().is_gpu() ||
            request_count <= 1 ||
            static_cast<int>(batched_request_states_.size()) != request_count)
        {
            return setError(
                "Request-batched device generation admission requires one complete GPU request set");
        }
        if (!device_generation_admission_.awaitsAdmission())
        {
            return setError(
                "Request-batched device generation reached sampling without an open prefill admission boundary");
        }

        int maximum_committed_tokens = 0;
        for (const BatchedDecodeRequestState &state : batched_request_states_)
        {
            if (state.logical_tokens <= 0 || state.is_complete ||
                !state.prefill_logits_ready)
            {
                return setError(
                    "Request-batched device generation admission requires every prefill row to be live and unconsumed");
            }
            maximum_committed_tokens =
                std::max(maximum_committed_tokens, state.logical_tokens);
        }

        int response_budget = decode_step_token_budget_;
        if (response_budget <= 0)
        {
            const int configured_context =
                plan_.runtime.max_seq_len > 0
                    ? plan_.runtime.max_seq_len
                    : config_.max_seq_len;
            response_budget = configured_context - maximum_committed_tokens;
        }
        if (response_budget <= 0)
        {
            return setError(
                "Request-batched device generation admission has no common positive response capacity");
        }

        if (!runner_->beginDeviceResidentGeneration(
                DeviceGenerationAdmissionRequest{
                    .request_count = request_count,
                    .max_new_tokens = response_budget,
                    .initial_leading_row_disposition =
                        sampling_math::
                            DeviceGenerationLeadingRowDisposition::
                                PendingResponse,
                }))
        {
            return setError(
                "Failed to admit the request-batched device generation response ledger");
        }

        if (!device_generation_admission_.admitController(response_budget))
        {
            return setError(
                "Request-batched device generation could not bind its response budget to the admitted controller");
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_admission_boundaries",
            1.0,
            "decode",
            {},
            {{"request_count", std::to_string(request_count)},
             {"response_budget", std::to_string(response_budget)},
             {"boundary", "request_batch_prefill_to_grouped_mtp"}});
        return true;
    }

    bool OrchestrationRunner::advanceDecodeTransactionPlanningPositionAfterForward(
        const char *source)
    {
        if (!runner_ || !runner_->primaryDeviceId().is_gpu())
            return true;
        if (!decode_transaction_planning_position_.has_value() ||
            *decode_transaction_planning_position_ < 0)
        {
            return setError(
                "GPU decode forward has no initialized scheduler-owned transaction position");
        }

        ++(*decode_transaction_planning_position_);
        PerfStatsCollector::addCounter(
            "mtp",
            "gpu_decode_transaction_position_forward_advances",
            1.0,
            "decode",
            {},
            {{"position",
              std::to_string(*decode_transaction_planning_position_)},
             {"source", source && source[0] != '\0' ? source : "unknown"},
             {"position_owner", "orchestration_transaction"}});
        return true;
    }

    bool OrchestrationRunner::publishDecodeTransactionPlanningPositionAfterMTPCommit(
        int transaction_base_tokens,
        int committed_rows,
        const char *source)
    {
        if (!runner_ || !runner_->primaryDeviceId().is_gpu())
        {
            decode_transaction_planning_position_.reset();
            return true;
        }
        if (transaction_base_tokens < 0 || committed_rows < 0)
        {
            return setError(
                "GPU MTP commit produced a negative scheduler transaction coordinate");
        }
        if (!decode_transaction_planning_position_.has_value() ||
            *decode_transaction_planning_position_ < 0)
        {
            return setError(
                "GPU MTP commit has no initialized scheduler-owned transaction position");
        }
        if (*decode_transaction_planning_position_ != transaction_base_tokens)
        {
            std::ostringstream error;
            error << "GPU MTP commit transaction base does not match the "
                     "scheduler-owned position: scheduler="
                  << *decode_transaction_planning_position_
                  << " transaction_base=" << transaction_base_tokens;
            return setError(error.str());
        }

        /*
         * The transaction validator has already proved that these rows are the
         * decode-equivalent continuation of transaction_base_tokens.  This
         * scalar is scheduler metadata used to plan the next captured launch;
         * it is not a host mirror of KV, GDN, terminal-hidden, or outcome
         * storage.  In particular, do not condition this publication on the
         * lifetime of a transient device outcome mailbox.
         */
        decode_transaction_planning_position_ =
            transaction_base_tokens + committed_rows;
        PerfStatsCollector::addCounter(
            "mtp",
            "gpu_decode_transaction_position_publications",
            1.0,
            "decode",
            {},
            {{"path", source && source[0] != '\0' ? source : "unknown"},
             {"position",
              std::to_string(*decode_transaction_planning_position_)},
             {"advanced_tokens", std::to_string(committed_rows)},
             {"position_owner", "orchestration_transaction"}});
        return true;
    }

    std::optional<int> OrchestrationRunner::currentDecodeTransactionPositionForPlanning(
        const char *context,
        std::string *error) const
    {
        if (!runner_)
        {
            if (error)
                *error = "MTP sidecar position planning requires an initialized runner";
            return std::nullopt;
        }

        if (runner_->primaryDeviceId().is_gpu())
        {
            if (decode_transaction_planning_position_.has_value() &&
                *decode_transaction_planning_position_ >= 0)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "gpu_decode_transaction_position_reads",
                    1.0,
                    "decode",
                    {},
                    {{"context", context ? context : "unknown"},
                     {"position",
                      std::to_string(*decode_transaction_planning_position_)},
                     {"position_owner", "orchestration_transaction"}});
                return *decode_transaction_planning_position_;
            }

            std::string message =
                "GPU decode planning has no scheduler-owned transaction position";
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
            "cpu_decode_position_planning_reads",
            1.0,
            "decode",
            {},
            {{"context", context ? context : "unknown"},
             {"position", std::to_string(position)},
             {"position_owner", "cpu_runner"}});
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

    bool OrchestrationRunner::noteEmbeddedDeviceMoEOverlayMaintenance(
        uint64_t transaction_count,
        DeviceGenerationExecutionPolicy execution_policy)
    {
        if (!runner_ ||
            runner_->moeOverlayAuthorityExecution() !=
                MoEOverlayAuthorityExecutionKind::
                    HomogeneousDeviceResident ||
            !runner_->deviceResidentMoEOverlayMaintenanceReady())
        {
            return true;
        }
        if (transaction_count == 0)
        {
            return setError(
                "Device-resident ExpertOverlay generation reported no committed maintenance boundary");
        }
        if (device_generation_embedded_moe_maintenance_pending_ack_)
        {
            return setError(
                "Device-resident ExpertOverlay generation crossed a second outer step before acknowledging embedded maintenance");
        }

        device_generation_embedded_moe_maintenance_pending_ack_ = true;
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "device_generation_embedded_maintenance_boundaries",
            static_cast<double>(transaction_count),
            "maintenance",
            runner_->primaryDeviceId().toString(),
            {{"authority_execution", "homogeneous_device_resident"},
             {"execution_policy",
              deviceGenerationExecutionPolicyName(execution_policy)},
             {"outer_host_launch", "false"},
             {"acknowledgement_pending", "true"}});
        return true;
    }

    GenerationBatchResult
    OrchestrationRunner::completeDeviceResidentBatchGeneration(
        const DeviceSpeculativePublicationRequest &publication_request,
        DeviceGenerationSamplingMode sampling_mode,
        int requested_draft_depth,
        int capture_draft_depth,
        GenerationBatchResult result)
    {
        const auto fail = [&](std::string message) -> GenerationBatchResult
        {
            result.error = std::move(message);
            return result;
        };
        if (!runner_ ||
            !isValidDeviceGenerationSamplingMode(sampling_mode) ||
            !publication_request.valid() ||
            !publication_request.outcome.device_generation_controller_owned ||
            publication_request.requestCount() <= 1)
        {
            return fail(
                "Request-batched device generation requires one valid controller-owned compact outcome per request");
        }

        const int request_count = publication_request.requestCount();
        const int admitted_token_budget =
            device_generation_admission_.tokenBudget();
        if (admitted_token_budget <= 0 ||
            static_cast<int>(batched_request_states_.size()) != request_count)
        {
            return fail(
                "Request-batched device generation has no matching admitted response budget and request set");
        }

        const MTPRuntimeConfig &mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        const DeviceGenerationLoopTopology loop_topology =
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth
                : DeviceGenerationLoopTopology::FixedDepth;
        const DeviceGenerationExecutionPolicy execution_policy =
            runner_->deviceGenerationExecutionPolicy(loop_topology);
        if (execution_policy == DeviceGenerationExecutionPolicy::Unsupported)
        {
            return fail(
                "Request-batched device generation has no complete captured execution policy");
        }
        const int parent_draft_depth =
            publication_request.logicalVerifierRowsPerRequest() - 1;
        if (parent_draft_depth <= 0 || capture_draft_depth <= 0 ||
            parent_draft_depth != capture_draft_depth ||
            !runner_->materializeDeviceResidentGeneration(
                request_count,
                parent_draft_depth,
                loop_topology,
                sampling_mode))
        {
            return fail(
                "Request-batched device generation could not materialize its complete captured graph family");
        }
        if (!runner_->launchDeviceResidentGeneration())
        {
            return fail(
                "Request-batched device generation could not launch its captured parent");
        }

        DeviceGenerationTerminalResult terminal;
        if (!runner_->finishDeviceResidentGeneration(&terminal) ||
            !terminal.valid() ||
            static_cast<int>(terminal.requests.size()) != request_count)
        {
            return fail(
                "Request-batched device generation did not produce one valid terminal ledger per request");
        }

        result.requests.resize(static_cast<size_t>(request_count));
        uint64_t total_output_tokens = 0;
        uint64_t total_transactions = 0;
        uint64_t total_accepted = 0;
        uint64_t total_rejected = 0;
        uint64_t total_consumed_rows = 0;
        uint64_t total_attempted_drafts = 0;
        uint64_t total_verifier_tokens = 0;
        uint64_t total_depth_windows = 0;
        uint64_t total_depth_updates = 0;
        uint64_t total_depth_promotions = 0;
        uint64_t total_depth_demotions = 0;
        int final_depth = 0;

        for (int request_index = 0;
             request_index < request_count;
             ++request_index)
        {
            const DeviceGenerationTerminalRequestResult &request =
                terminal.requests[static_cast<size_t>(request_index)];
            const int response_count =
                static_cast<int>(request.tokens.size());
            if (response_count <= 0 ||
                request.remaining_token_count < 0 ||
                response_count + request.remaining_token_count !=
                    admitted_token_budget ||
                (request.remaining_token_count > 0 && !request.model_stopped) ||
                request.transaction_count <= 0 ||
                request.published_state_commit_count < 0 ||
                request.published_state_commit_count > response_count + 1 ||
                request.attempted_draft_token_count <= 0 ||
                request.verifier_token_count !=
                    request.attempted_draft_token_count +
                        request.transaction_count)
            {
                return fail(
                    "Request-batched terminal ledger disagrees with its admitted response, publication, or depth budget for request " +
                    std::to_string(request_index));
            }

            GenerationResult &request_result =
                result.requests[static_cast<size_t>(request_index)];
            request_result.tokens = request.tokens;
            request_result.is_complete = request.model_stopped;

            total_output_tokens += static_cast<uint64_t>(response_count);
            total_transactions +=
                static_cast<uint64_t>(request.transaction_count);
            total_accepted += static_cast<uint64_t>(
                request.accepted_speculative_token_count);
            total_rejected += static_cast<uint64_t>(
                request.rejected_transaction_count);
            total_consumed_rows += static_cast<uint64_t>(
                request.consumed_verifier_row_count);
            total_attempted_drafts += static_cast<uint64_t>(
                request.attempted_draft_token_count);
            total_verifier_tokens += static_cast<uint64_t>(
                request.verifier_token_count);
            total_depth_windows += static_cast<uint64_t>(
                request.depth_evaluated_window_count);
            total_depth_updates += static_cast<uint64_t>(
                request.depth_update_count);
            total_depth_promotions += static_cast<uint64_t>(
                request.depth_promotion_count);
            total_depth_demotions += static_cast<uint64_t>(
                request.depth_demotion_count);
            final_depth = std::max(final_depth, request.final_draft_depth);
        }

        const bool stochastic =
            sampling_mode == DeviceGenerationSamplingMode::Stochastic;
        mtp_stats_.draft_steps += total_attempted_drafts;
        mtp_stats_.verifier_runs += total_transactions;
        mtp_stats_.verifier_token_count += total_verifier_tokens;
        mtp_stats_.accepted_tokens += total_accepted;
        mtp_stats_.rejected_tokens += total_rejected;
        mtp_stats_.transaction_commits += total_transactions;
        mtp_stats_.depth_policy_windows += total_depth_windows;
        mtp_stats_.depth_policy_updates += total_depth_updates;
        mtp_stats_.depth_policy_promotions += total_depth_promotions;
        mtp_stats_.depth_policy_demotions += total_depth_demotions;
        mtp_stats_.current_depth = final_depth;
        if (stochastic)
        {
            mtp_stats_.rollbacks += total_rejected;
            mtp_stats_.transaction_rollbacks += total_rejected;
            mtp_stats_.stochastic_accept_tests += total_consumed_rows;
            mtp_stats_.stochastic_accepts += total_accepted;
            mtp_stats_.stochastic_residual_samples += total_rejected;
        }

        const PerfStatsCollector::Tags terminal_tags{
            {"path", "request_batch_device_resident_generation_loop"},
            {"execution_policy",
             deviceGenerationExecutionPolicyName(execution_policy)},
            {"sampling", deviceGenerationSamplingModeName(sampling_mode)},
            {"requests", std::to_string(request_count)},
            {"requested_depth", std::to_string(requested_draft_depth)},
            {"capture_depth", std::to_string(capture_draft_depth)},
            {"final_depth", std::to_string(final_depth)},
            {"host_transaction_materializations", "0"}};
        PerfStatsCollector::addCounter(
            "mtp",
            "device_resident_generation_requests",
            static_cast<double>(request_count),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "output_tokens",
            static_cast<double>(total_output_tokens),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            stochastic
                ? "grouped_decode_equivalent_stochastic_verifier_runs"
                : "grouped_decode_equivalent_greedy_verifier_runs",
            static_cast<double>(total_transactions),
            "decode",
            {},
            terminal_tags);

        if (!device_generation_admission_.controllerActive())
        {
            return fail(
                "Request-batched terminal ledger lost its active controller lifecycle before retirement");
        }
        if (!noteEmbeddedDeviceMoEOverlayMaintenance(
                total_transactions, execution_policy))
        {
            return fail(
                last_error_.empty()
                    ? "Request-batched device generation could not publish embedded ExpertOverlay maintenance ownership"
                    : last_error_);
        }
        device_generation_admission_.reset();
        device_generation_terminal_ledger_authoritative_ = true;
        clearBatchedDecodeState();
        return result;
    }

    GenerationResult OrchestrationRunner::completeDeviceResidentGeneration(
        const DeviceSpeculativePublicationRequest &publication_request,
        DeviceGenerationSamplingMode sampling_mode,
        int transaction_base_cached_tokens,
        int requested_draft_depth,
        int capture_draft_depth,
        GenerationResult result)
    {
        const auto fail = [&](const std::string &message) -> GenerationResult
        {
            result.error = message;
            return result;
        };
        if (!isValidDeviceGenerationSamplingMode(sampling_mode) ||
            !publication_request.valid() ||
            !publication_request.outcome.device_generation_controller_owned)
        {
            return fail(
                "Device-resident MTP generation requires one valid controller-owned compact outcome and sampling topology");
        }

        const char *const sampling_name =
            deviceGenerationSamplingModeName(sampling_mode);
        const bool stochastic =
            sampling_mode == DeviceGenerationSamplingMode::Stochastic;
        const MTPRuntimeConfig &mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        const DeviceGenerationLoopTopology loop_topology =
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth
                : DeviceGenerationLoopTopology::FixedDepth;
        const DeviceGenerationExecutionPolicy execution_policy =
            runner_->deviceGenerationExecutionPolicy(loop_topology);
        if (execution_policy ==
            DeviceGenerationExecutionPolicy::Unsupported)
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP has no complete generation-loop execution policy");
        }

        const int parent_draft_depth =
            publication_request.logicalVerifierRowsPerRequest() - 1;
        if (parent_draft_depth <= 0 || capture_draft_depth <= 0 ||
            parent_draft_depth != capture_draft_depth ||
            !runner_->materializeDeviceResidentGeneration(
                publication_request.requestCount(),
                parent_draft_depth,
                loop_topology,
                sampling_mode))
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP could not materialize its complete device-generation graph family");
        }

        if (!runner_->launchDeviceResidentGeneration())
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP could not launch its device-generation graph");
        }

        DeviceGenerationTerminalResult terminal;
        if (!runner_->finishDeviceResidentGeneration(&terminal) ||
            !terminal.valid() || terminal.requests.size() != 1u)
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP could not materialize one valid terminal device ledger");
        }

        const DeviceGenerationTerminalRequestResult &request_result =
            terminal.requests.front();
        const int response_count =
            static_cast<int>(request_result.tokens.size());
        const int admitted_token_budget =
            device_generation_admission_.tokenBudget();
        if (admitted_token_budget <= 0 || response_count <= 0 ||
            request_result.remaining_token_count < 0 ||
            !sampling_math::
                valid_device_generation_leading_row_disposition(
                    request_result.next_leading_row_disposition) ||
            (request_result.model_stopped &&
             request_result.next_leading_row_disposition !=
                 sampling_math::DeviceGenerationLeadingRowDisposition::
                     PendingResponse) ||
            response_count + request_result.remaining_token_count !=
                admitted_token_budget ||
            (request_result.remaining_token_count > 0 &&
             !request_result.model_stopped) ||
            request_result.transaction_count <= 0 ||
            request_result.published_state_commit_count < 0 ||
            request_result.published_state_commit_count > response_count + 1 ||
            (!request_result.model_stopped &&
             request_result.published_state_commit_count <= 0))
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP terminal ledger disagrees with the admitted response/state budget: response_count=" +
                std::to_string(response_count) +
                " remaining_token_count=" +
                std::to_string(request_result.remaining_token_count) +
                " admitted_token_budget=" +
                std::to_string(admitted_token_budget) +
                " model_stopped=" +
                (request_result.model_stopped ? "true" : "false") +
                " next_leading_committed_output_count=" +
                std::to_string(
                    sampling_math::
                        device_generation_leading_committed_output_count(
                            request_result.next_leading_row_disposition)) +
                " transaction_count=" +
                std::to_string(request_result.transaction_count) +
                " published_state_commit_count=" +
                std::to_string(
                    request_result.published_state_commit_count));
        }
        if (!publishDecodeTransactionPlanningPositionAfterMTPCommit(
                transaction_base_cached_tokens,
                request_result.published_state_commit_count,
                "device_resident_generation_loop"))
        {
            return fail(
                last_error_.empty()
                    ? std::string("Device-resident ") + sampling_name +
                          " MTP could not publish its terminal scheduler position"
                    : last_error_);
        }

        std::optional<ReadyMTPCondition> terminal_ready_condition;
        if (!request_result.model_stopped)
        {
            /*
             * Exhausting the caller's response budget is not a model-state
             * boundary. The final compact publication has already committed
             * the exact verifier prefix and sampled the token that belongs at
             * the next logical position. finishDeviceResidentGeneration()
             * leaves that mailbox live by contract; preserve the authenticated
             * handle itself and never infer the continuation from the last
             * host-visible response token.
             */
            DeviceResidentLogicalSequenceStateHandle resident_condition =
                runner_->deviceResidentLogicalSequenceState();
            if (!resident_condition.coversRequest(0) ||
                resident_condition.device != terminal.device ||
                resident_condition.device != runner_->primaryDeviceId())
            {
                return fail(
                    std::string("Device-resident ") + sampling_name +
                    " MTP terminal ledger has no matching live continuation mailbox");
            }
            terminal_ready_condition =
                ReadyMTPCondition::deviceResident(
                    active_sampling_params_,
                    std::move(resident_condition),
                    request_result.next_leading_row_disposition);
            if (!terminal_ready_condition->valid())
            {
                return fail(
                    std::string("Device-resident ") + sampling_name +
                    " MTP terminal continuation is incomplete");
            }
        }

        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        prefill_logits_ready_ = terminal_ready_condition.has_value();
        ready_mtp_condition_ = std::move(terminal_ready_condition);

        for (const int32_t token : request_result.tokens)
        {
            sampler_.record_token(token);
            result.tokens.push_back(token);
        }
        last_token_ = request_result.tokens.back();
        result.is_complete = request_result.model_stopped;

        const uint64_t transactions =
            static_cast<uint64_t>(request_result.transaction_count);
        const uint64_t accepted = static_cast<uint64_t>(
            request_result.accepted_speculative_token_count);
        const uint64_t rejected = static_cast<uint64_t>(
            request_result.rejected_transaction_count);
        const uint64_t consumed_rows = static_cast<uint64_t>(
            request_result.consumed_verifier_row_count);
        if (request_result.attempted_draft_token_count <= 0 ||
            request_result.verifier_token_count <= 0 ||
            request_result.verifier_token_count !=
                request_result.attempted_draft_token_count +
                    request_result.transaction_count ||
            mtp_stats_.draft_steps <
                static_cast<uint64_t>(capture_draft_depth))
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP terminal depth ledger is internally inconsistent");
        }

        /*
         * Dynamic first-use capture physically warms every capacity slot. Its
         * host-side setup counter therefore describes graph construction, not
         * selected work. Replace that setup width with the exact cumulative
         * widths recorded by the sole device controller.
         */
        mtp_stats_.draft_steps -=
            static_cast<uint64_t>(capture_draft_depth);
        mtp_stats_.draft_steps += static_cast<uint64_t>(
            request_result.attempted_draft_token_count);
        mtp_stats_.verifier_runs += transactions;
        mtp_stats_.verifier_token_count += static_cast<uint64_t>(
            request_result.verifier_token_count);
        mtp_stats_.last_transaction_draft_depth =
            request_result.last_transaction_draft_depth;
        mtp_stats_.last_transaction_emitted_token_count =
            request_result.last_transaction_emitted_token_count;
        mtp_stats_.accepted_tokens += accepted;
        mtp_stats_.rejected_tokens += rejected;
        if (stochastic)
        {
            mtp_stats_.rollbacks += rejected;
            mtp_stats_.transaction_rollbacks += rejected;
            mtp_stats_.stochastic_accept_tests += consumed_rows;
            mtp_stats_.stochastic_accepts += accepted;
            mtp_stats_.stochastic_residual_samples += rejected;
        }
        mtp_stats_.transaction_commits += transactions;
        mtp_stats_.depth_policy_windows += static_cast<uint64_t>(
            request_result.depth_evaluated_window_count);
        mtp_stats_.depth_policy_updates += static_cast<uint64_t>(
            request_result.depth_update_count);
        mtp_stats_.depth_policy_promotions += static_cast<uint64_t>(
            request_result.depth_promotion_count);
        mtp_stats_.depth_policy_demotions += static_cast<uint64_t>(
            request_result.depth_demotion_count);
        mtp_stats_.current_depth = request_result.final_draft_depth;
        if (!device_generation_admission_.retireController(
                request_result.model_stopped))
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP could not retire its controller into the next explicit request phase");
        }

        const PerfStatsCollector::Tags terminal_tags{
            {"path", "device_resident_generation_loop"},
            {"execution_policy",
             deviceGenerationExecutionPolicyName(execution_policy)},
            {"sampling", sampling_name},
            {"depth", std::to_string(requested_draft_depth)},
            {"capture_depth", std::to_string(capture_draft_depth)},
            {"final_depth",
             std::to_string(request_result.final_draft_depth)},
            {"depth_evaluated_windows",
             std::to_string(request_result.depth_evaluated_window_count)},
            {"depth_updates",
             std::to_string(request_result.depth_update_count)},
            {"depth_promotions",
             std::to_string(request_result.depth_promotion_count)},
            {"depth_demotions",
             std::to_string(request_result.depth_demotion_count)},
            {"transactions",
             std::to_string(request_result.transaction_count)},
            {"state_commits",
             std::to_string(request_result.published_state_commit_count)}};
        PerfStatsCollector::addCounter(
            "mtp",
            "device_resident_generation_requests",
            1.0,
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "output_tokens",
            static_cast<double>(response_count),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_runs",
            static_cast<double>(transactions),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_tokens",
            static_cast<double>(request_result.verifier_token_count),
            "decode",
            {},
            terminal_tags);

        const std::string grouped_route_counter =
            stochastic
                ? "grouped_decode_equivalent_stochastic_verifier_runs"
                : "grouped_decode_equivalent_greedy_verifier_runs";
        const std::string verifier_path =
            stochastic
                ? "grouped_decode_equivalent_stochastic"
                : "grouped_decode_equivalent_greedy";
        PerfStatsCollector::addCounter(
            "mtp",
            grouped_route_counter,
            static_cast<double>(transactions),
            "decode",
            {},
            {{"execution",
              deviceGenerationExecutionPolicyName(execution_policy)},
             {"sampling", sampling_name},
             {"verifier_forward_tokens",
              std::to_string(request_result.verifier_token_count)},
             {"verifier_rows",
              std::to_string(
                  publication_request.logicalVerifierRowsPerRequest())},
             {"replay_forward_tokens", "0"},
             {"accepted_tokens", std::to_string(accepted)},
             {"state_publication", "device_resident"}});

        if (stochastic)
        {
            PerfStatsCollector::Tags stochastic_tags = terminal_tags;
            stochastic_tags.emplace("device_resident", "true");
            stochastic_tags.emplace("verifier_path", verifier_path);
            stochastic_tags.emplace(
                "implementation",
                "device_resident_generation_terminal_ledger");
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_accept_tests",
                static_cast<double>(consumed_rows),
                "decode",
                {},
                stochastic_tags);
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_accepts",
                static_cast<double>(accepted),
                "decode",
                {},
                stochastic_tags);
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "depth_policy_windows",
            static_cast<double>(request_result.depth_evaluated_window_count),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "accepted_tokens",
            static_cast<double>(accepted),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "rejected_tokens",
            static_cast<double>(rejected),
            "decode",
            {},
            terminal_tags);
        if (!noteEmbeddedDeviceMoEOverlayMaintenance(
                transactions, execution_policy))
        {
            return fail(
                last_error_.empty()
                    ? std::string("Device-resident ") + sampling_name +
                          " MTP could not publish embedded ExpertOverlay maintenance ownership"
                    : last_error_);
        }
        device_generation_terminal_ledger_authoritative_ = true;
        return result;
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

        const bool use_ready_logits = prefill_logits_ready_;
        const std::optional<ReadyMTPCondition> ready_condition =
            ready_mtp_condition_;
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
        ready_mtp_condition_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        PrefixStateSnapshot rollback_checkpoint;
        bool rollback_checkpoint_captured = false;
        PrefixStateSnapshot verifier_base_checkpoint;
        int transaction_base_cached_tokens = -1;

        auto current_checkpoint_capture_request =
            [&](const char *context,
                std::string *error)
            -> std::optional<PrefixCheckpointCaptureRequest>
        {
            const std::optional<int> position =
                currentDecodeTransactionPositionForPlanning(context, error);
            if (!position)
                return std::nullopt;
            return PrefixCheckpointCaptureRequest{
                .sequence_index = 0,
                .logical_cached_tokens = *position};
        };

        auto fail_without_checkpoint = [&](const std::string &message) -> GenerationResult
        {
            PerfStatsCollector::addCounter("mtp", "decode_step_failures", 1.0, "decode",
                                           std::string{}, {{"reason", message}});
            prefill_logits_ready_ = use_ready_logits;
            ready_mtp_condition_ = ready_condition;
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
                std::string position_error;
                const auto capture_request =
                    current_checkpoint_capture_request(
                        "rollback checkpoint capture",
                        &position_error);
                if (!capture_request)
                {
                    last_error_ = std::move(position_error);
                    return false;
                }
                rollback_checkpoint =
                    runner_->captureLivePrefixCheckpoint(*capture_request);
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
            ready_mtp_condition_ = ready_condition;
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
            mpi_ctx_ && mpi_ctx_->world_size() > 1 &&
            !moe_overlay_inference_transaction_coordinator_ &&
            !moe_overlay_inference_transaction_follower_;
        auto fence_mpi_mtp_boundary =
            [&](const char *boundary) -> std::optional<std::string>
        {
            if (!needs_mpi_mtp_boundary_fence)
                return std::nullopt;

            /*
             * Global/NodeLocal TP ranks must enter sidecar and verifier
             * collectives in the same logical order. A local runner flush only
             * drains the participant's own work; it does not stop the root from
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

        if (ready_condition.has_value())
        {
            /*
             * A ready verifier condition is sampled one decode boundary before
             * it is consumed. Treat its authority and sampling contract as one
             * atomic state: a detached condition or a changed policy would mix
             * two sampling regimes in one request.
             */
            if (!use_ready_logits)
            {
                return fail_after_checkpoint(
                    "Ready MTP condition exists without a terminal-logits boundary");
            }
            if (!ready_condition->valid())
            {
                return fail_after_checkpoint(
                    "Ready MTP condition has an invalid authority payload");
            }
            if (!samplingParamsEqual(
                    ready_condition->sampling_params,
                    active_sampling_params_))
            {
                return fail_after_checkpoint(
                    "Ready MTP condition was sampled with different sampling parameters");
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
        const bool use_serial_sample_equivalent_host_stochastic =
            stochastic_host_verify &&
            active_sampling_params_.seed != 0;
        const bool use_sampling_penalties =
            active_sampling_params_.has_penalties() && !stochastic_verify;
        if (stochastic_device_verify &&
            active_sampling_params_.dry_multiplier != 0.0f &&
            active_sampling_params_.dry_penalty_last_n != 0)
        {
            return fail_without_checkpoint(
                "GPU stochastic MTP requires a device-owned DRY sequence-history implementation");
        }
        if (runner_->primaryDeviceId().is_gpu() && use_sampling_penalties &&
            active_sampling_params_.dry_multiplier != 0.0f &&
            active_sampling_params_.dry_penalty_last_n != 0)
        {
            return fail_without_checkpoint(
                "GPU greedy MTP requires a device-owned DRY sequence-history implementation");
        }
        if (runner_->primaryDeviceId().is_gpu() &&
            !runner_->supportsMTPSidecarLogitsStreamHandoff())
        {
            /*
             * This is a foundational GPU MTP ownership contract, so validate it
             * before selecting a verifier policy or inspecting any secondary
             * grouped-publication capability. No GPU MTP transaction may begin
             * if its first sidecar producer cannot hand ownership to the next
             * device consumer through an explicit stream event.
             */
            return fail_without_checkpoint(
                "GPU MTP requires device-resident sidecar stream handoff");
        }
        const MTPDepthPolicyModelClass verifier_model_class =
            inferMTPDepthPolicyModelClass(model_ctx_);
        const int verifier_policy_probe_rows =
            std::max(1, effectiveMTPMaxDraftDepth(mtp));
        const MTPVerifierPolicyDecision verifier_policy =
            chooseMTPVerifierPolicy(
                MTPVerifierPolicyInput{
                    .greedy_sampling = active_sampling_params_.is_greedy(),
                    .stochastic_verify = stochastic_verify,
                    .uses_sampling_penalties = use_sampling_penalties,
                    .supports_row_local_penalty_application =
                        !use_sampling_penalties ||
                        runner_->supportsRowLocalAllPositionPenaltyApplication(),
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
             {"grouped_outcome_required", "true"},
             {"device_publication_supported",
              perfBool(
                  runner_->supportsDeviceResidentMTPSpecStatePublication())}});
        const bool use_grouped_outcome_device_resident_publication_verifier =
            verifier_policy.path ==
                MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome &&
            (!stochastic_verify || stochastic_device_verify) &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsDeviceResidentMTPSpecStatePublication();
        const bool use_grouped_decode_equivalent_outcome_verifier =
            verifier_policy.path ==
            MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome;
        if (use_grouped_decode_equivalent_outcome_verifier &&
            runner_->primaryDeviceId().is_gpu() &&
            stochastic_verify &&
            !stochastic_device_verify)
        {
            return fail_without_checkpoint(
                "GPU stochastic MTP requires device-resident distribution verification");
        }
        if (use_grouped_decode_equivalent_outcome_verifier &&
            runner_->primaryDeviceId().is_gpu() &&
            !use_grouped_outcome_device_resident_publication_verifier)
        {
            /*
             * CUDA/ROCm grouped verifier rows are only a production path when
             * the accepted-state publication also stays device-resident.  If we
             * already know the runner cannot publish the compact outcome on
             * device, fail before enabling verifier-row logits or forwarding
             * any grouped rows; row replay and host step plans are diagnostics,
             * not GPU hot-path substitutes.
             */
            return fail_without_checkpoint(
                "Grouped decode-equivalent MTP verifier has no grouped publication path; GPU grouped verifier requires device-resident accepted-state publication");
        }
        const DeviceGenerationLoopTopology generation_loop_topology =
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth
                : DeviceGenerationLoopTopology::FixedDepth;
        const DeviceGenerationExecutionPolicy generation_execution_policy =
            runner_->deviceGenerationExecutionPolicy(
                generation_loop_topology);
        if (use_grouped_outcome_device_resident_publication_verifier &&
            generation_execution_policy ==
                DeviceGenerationExecutionPolicy::Unsupported)
        {
            return fail_without_checkpoint(
                "GPU grouped MTP has no complete generation-loop execution policy for the requested depth topology");
        }

        /*
         * Native dynamic generation materializes one policy-complete parent
         * from the first transaction. HIP's explicit hosted policy keeps the
         * existing transaction loop and must never attempt native composition.
         * Preserve this admission edge before beginDeviceResidentGeneration()
         * consumes the pending marker.
         */
        const bool materialize_dynamic_generation_loop_this_step =
            device_generation_admission_.awaitsAdmission() &&
            use_grouped_outcome_device_resident_publication_verifier &&
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic;
        const bool use_grouped_outcome_host_publication_verifier =
            verifier_policy.path ==
                MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome &&
            !use_grouped_outcome_device_resident_publication_verifier &&
            !runner_->primaryDeviceId().is_gpu() &&
            (!stochastic_verify || stochastic_host_verify);
        if (use_grouped_outcome_host_publication_verifier)
        {
            /*
             * CPU grouped verification still publishes through host-visible step
             * plans. GPU grouped verification must use the compact
             * device-resident publisher; allowing this middle lane on CUDA/ROCm
             * quietly reintroduced host-owned MTP state after verifier rows had
             * already been proven decode-equivalent.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "grouped_outcome_host_publication_uses",
                1.0,
                "decode",
                {},
                {{"model_class", mtpDepthPolicyModelClassName(verifier_model_class)},
                 {"probe_rows", std::to_string(verifier_policy_probe_rows)},
                 {"reason", verifier_policy.reason}});
        }
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
        /*
         * A rejected verifier row has already been emitted to the caller, but
         * it is not live model state until the next decode step consumes it as
         * the condition token.  Grouped-host publication uses host-visible
         * transaction plans rather than device-resident mailboxes, but it still
         * shares the same pending-condition contract as the other verifier-row
         * publishers.  Treating it as a bypass would sample from stale logits
         * and skip the rejected correction token on the next step.
         */
        const bool use_pending_condition_row =
            pending_condition_candidate &&
            (use_grouped_outcome_host_publication_verifier ||
             use_grouped_outcome_device_resident_publication_verifier);
        const bool ready_condition_has_resident_state =
            use_ready_logits &&
            ready_condition.has_value() &&
            ready_condition->resident_state.has_value() &&
            ready_condition->resident_state->valid();
        std::optional<DeviceResidentLogicalSequenceStateHandle>
            first_token_resident_state;
        if (use_pending_condition_row &&
            pending_condition_has_resident_state)
        {
            first_token_resident_state = pending_condition_resident_state;
        }
        else if (ready_condition_has_resident_state)
        {
            first_token_resident_state = ready_condition->resident_state;
        }
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
        const bool grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint =
            use_grouped_outcome_device_resident_publication_verifier &&
            plan_.usesLocalTP() &&
            !runner_->supportsMTPShiftedRowReuseFromSidecar();
        /*
         * LocalTP MoE grouped-outcome publication cannot reuse the first shifted
         * MTP KV row from the sidecar: routed expert state and shifted sidecar
         * KV must be committed from the accepted verifier row.  That initial
         * shifted-row commit needs the verifier-base terminal hidden payload,
         * so a token-count-only logical checkpoint is not a valid base for this
         * lane even though the accepted-state publication itself is resident.
         */
        const bool use_device_publication_without_rollback_checkpoint =
            use_grouped_outcome_device_resident_publication_verifier &&
            (!stochastic_verify || stochastic_device_verify) &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsDeviceResidentMTPSpecStatePublication() &&
            runner_->supportsMTPSidecarPreservesMainState() &&
            !grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint &&
            !verify_sidecar_preserves_main_state &&
            !verify_commit_replay_check;
        const bool can_synthesize_verifier_base_checkpoint =
            use_grouped_outcome_device_resident_publication_verifier &&
            runner_->supportsMTPSidecarPreservesMainState() &&
            (runner_->supportsLogicalMTPVerifierBaseCheckpoint() ||
             use_device_publication_without_rollback_checkpoint) &&
            !grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint &&
            !verify_sidecar_preserves_main_state &&
            !verify_commit_replay_check;

        if (use_device_publication_without_rollback_checkpoint)
        {
            std::string position_error;
            const std::optional<int> base_position =
                currentDecodeTransactionPositionForPlanning(
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
                 {"verifier_path", "grouped_decode_equivalent_outcome"}});
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

                if (!publishDecodeTransactionPlanningPositionAfterMTPCommit(
                        base.cached_tokens,
                        advanced_tokens,
                        path))
                {
                    return last_error_.empty()
                               ? std::optional<std::string>{
                                     "MTP transaction could not publish its "
                                     "scheduler-owned position"}
                               : std::optional<std::string>{last_error_};
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
                ready_mtp_condition_ = ReadyMTPCondition::hostVisible(
                    *ready_token,
                    active_sampling_params_,
                    ready_condition_resident_state);
                if (!ready_mtp_condition_->valid())
                {
                    return std::string(
                        "MTP transaction produced an invalid host-visible ready condition");
                }
                pending_mtp_condition_token_.reset();
                pending_mtp_condition_params_.reset();
                pending_mtp_condition_resident_state_.reset();
            }
            else
            {
                ready_mtp_condition_.reset();
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

        auto inspect_spec_decode_metadata = [&](
                                                    const char *path,
                                                    const std::string &implementation,
                                                    const MTPSpecDecodeMetadataBatch &metadata,
                                                    bool stopped_on_output,
                                                    const std::string &draft_token_description,
                                                    const std::vector<int32_t> &committed_output_tokens)
            -> std::optional<std::string>
        {
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
                return std::string("MTP spec-decode metadata failed on ") +
                       path + ": " + metadata.error;
            }
            if (metadata.transactions.empty())
                return std::string("MTP spec-decode metadata produced no transaction");

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
                 {"draft_tokens", draft_token_description},
                 {"committed_output_tokens",
                  join_tokens(committed_output_tokens)}});
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
                                                        int accepted_mtp_draft_prefix,
                                                        bool commit_boundary_clipped)
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

            MTPSpecDecodeMetadataBatch metadata;
            if (commit_boundary_clipped)
            {
                MTPSpecDecodeAcceptedOutcome boundary_outcome;
                boundary_outcome.request_id = 0;
                boundary_outcome.vocab_size = vocab;
                boundary_outcome.draft_count =
                    static_cast<int>(draft_tokens_for_tx.size());
                boundary_outcome.committed_output_tokens =
                    committed_output_tokens;
                boundary_outcome.commit_boundary_ready_token = ready_token;
                boundary_outcome.accepted_verifier_input_prefix =
                    accepted_mtp_draft_prefix + 1;
                boundary_outcome.target_verifier_state_commit_count =
                    static_cast<int>(committed_output_tokens.size());
                boundary_outcome.all_drafts_accepted = false;
                boundary_outcome.stopped_on_output = false;
                boundary_outcome.commit_boundary_clipped = true;
                metadata = buildMTPSpecDecodeMetadataBatchFromAcceptedOutcome(
                    metadata_shape,
                    boundary_outcome);
            }
            else
            {
                MTPDecodeCatchupGreedyRequest catchup_request_for_tx;
                catchup_request_for_tx.draft_tokens = draft_tokens_for_tx;
                MTPDecodeCatchupGreedyResult catchup_result_for_tx;
                catchup_result_for_tx.ok = true;
                catchup_result_for_tx.accepted_tokens =
                    committed_output_tokens;
                catchup_result_for_tx.all_speculative_accepted =
                    all_drafts_accepted;
                catchup_result_for_tx.stopped_on_output = stopped_on_output;
                catchup_result_for_tx.accepted_speculative_prefix =
                    accepted_mtp_draft_prefix;
                catchup_result_for_tx.ready_token =
                    ready_token.value_or(kMTPSpecDecodeInvalidToken);
                metadata = buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
                    metadata_shape,
                    /*request_id=*/0,
                    vocab,
                    catchup_request_for_tx,
                    catchup_result_for_tx);
            }
            return inspect_spec_decode_metadata(
                path,
                implementation,
                metadata,
                stopped_on_output,
                join_tokens(draft_tokens_for_tx),
                committed_output_tokens);
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
            return inspect_spec_decode_metadata(
                path,
                implementation,
                metadata,
                outcome.stopped_on_output,
                std::string("device_deferred:") +
                    std::to_string(outcome.draft_count),
                outcome.committed_output_tokens);
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
            /*
             * The condition forward is an ordinary serial-decode state
             * advance: it consumes condition_token in the main graph.  MTP's
             * shifted sidecar cache must therefore publish the row derived from
             * the current terminal hidden before that main forward runs.  If we
             * skip this maintenance step, prefix-cache-restored LocalTP/MoE
             * runs leave shifted KV one row behind the main position, and the
             * next direct-emit or verifier publication observes a cache-head
             * mismatch.
             */
            const bool gpu_condition_forward =
                runner_->primaryDeviceId().is_gpu();
            constexpr int kConditionTargetSampleSlot = 0;
            DeviceResidentLogicalSequenceStateHandle condition_state;
            std::optional<int> cpu_condition_sidecar_position;

            if (gpu_condition_forward)
            {
                /*
                 * The compact verifier publisher owns the condition token and
                 * its pre-forward logical position as one event-published
                 * device transaction. Preserve the token D2D before shifted-KV
                 * maintenance consumes that one-shot mailbox. The persistent
                 * target slot then becomes the sole token source for the main
                 * condition graph; no host token or position is uploaded.
                 */
                condition_state =
                    runner_->deviceResidentLogicalSequenceState();
                if (!condition_state.valid())
                {
                    return fail_after_checkpoint(
                        "GPU MTP condition forward requires a device-resident logical-state mailbox");
                }
                if (!runner_
                         ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                             condition_state,
                             /*request_index=*/0,
                             kConditionTargetSampleSlot))
                {
                    return fail_after_checkpoint(
                        "GPU MTP condition forward could not preserve its resident token in the device target slot");
                }
            }
            else
            {
                std::string condition_sidecar_position_error;
                cpu_condition_sidecar_position =
                    currentDecodeTransactionPositionForPlanning(
                        "condition forward shifted commit",
                        &condition_sidecar_position_error);
                if (!cpu_condition_sidecar_position)
                {
                    return fail_after_checkpoint(
                        condition_sidecar_position_error);
                }
            }

            bool condition_shifted_commit_ok = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "condition_forward_shifted_commit",
                    "decode");
                condition_shifted_commit_ok = gpu_condition_forward
                    ? runner_
                          ->commitMTPShiftedRowFromDeviceResidentLogicalState(
                              condition_state,
                              /*request_index=*/0,
                              /*already_appended_tokens=*/0,
                              /*allow_speculative_discard=*/true)
                    : runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                          condition_token,
                          /*already_appended_tokens=*/0,
                          /*allow_speculative_discard=*/true,
                          *cpu_condition_sidecar_position);
            }
            if (!condition_shifted_commit_ok)
            {
                return fail_after_checkpoint(
                    "MTP condition-forward shifted-cache maintenance failed");
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "condition_forward_shifted_commits",
                1.0,
                "decode");

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
                if (gpu_condition_forward)
                {
                    /*
                     * Shifted-KV publication consumed the original logical-state
                     * mailbox. Recompose the condition token from the durable
                     * target slot with the now-current device KV position, then
                     * launch the captured main graph from that typed mailbox.
                     */
                    ok = runner_
                             ->advanceMTPMainConditionFromDeviceTargetSample(
                                 condition_token,
                                 kConditionTargetSampleSlot);
                }
                else
                {
                    ok = runner_->forward(&condition_token, 1);
                }
                if (!ok)
                {
                    runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                }
            }
            if (!ok)
                return fail_after_checkpoint("Forward pass failed during MTP condition decode");
            if (!advanceDecodeTransactionPlanningPositionAfterForward(
                    "mtp_condition_forward"))
            {
                return fail_after_checkpoint(last_error_);
            }
            if (can_synthesize_verifier_base_checkpoint)
            {
                std::string position_error;
                const std::optional<int> verifier_base_position =
                    currentDecodeTransactionPositionForPlanning(
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
                     {"verifier_path", "grouped_decode_equivalent_outcome"}});
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
                std::string position_error;
                const auto capture_request =
                    current_checkpoint_capture_request(
                        "verifier-base checkpoint capture",
                        &position_error);
                if (!capture_request)
                    return fail_after_checkpoint(position_error);
                verifier_base_checkpoint =
                    runner_->captureLivePrefixCheckpoint(*capture_request);
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
        const int transaction_draft_capacity =
            materialize_dynamic_generation_loop_this_step
                ? effectiveMTPMaxDraftDepth(mtp)
                : requested_speculative_draft_count;
        if (transaction_draft_capacity < requested_speculative_draft_count)
        {
            return fail_after_checkpoint(
                "Dynamic MTP graph-family capture capacity is narrower than its admitted device selector");
        }
        /*
         * Publish the selected transaction geometry before any response-budget
         * clipping.  This is the backend-neutral execution ledger for fixed
         * depth: CPU executes one host-owned grouped transaction at a time,
         * while CUDA and ROCm hand this same selected/capture geometry to the
         * device-resident generation parent.  Completion evidence is recorded
         * separately by verifier counters, so this record cannot certify a
         * transaction that selected a depth and then failed before execution.
         */
        PerfStatsCollector::addCounter(
            "mtp",
            "decode_transaction_depth_selections",
            1.0,
            "decode",
            runner_->primaryDeviceId().toString(),
            {{"depth_policy",
              mtpDepthPolicyModeToString(mtp.depth_policy.mode)},
             {"requested_depth",
              std::to_string(requested_speculative_draft_count)},
             {"capture_depth",
              std::to_string(transaction_draft_capacity)}});
        if (materialize_dynamic_generation_loop_this_step)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "dynamic_device_generation_capacity_capture_transactions",
                1.0,
                "graph_setup",
                {},
                {{"selected_depth",
                  std::to_string(requested_speculative_draft_count)},
                 {"capture_depth",
                  std::to_string(transaction_draft_capacity)},
                 {"authority", "device_generation_controller"}});
        }
        const bool first_condition_was_already_emitted =
            use_pending_condition_row ||
            (use_ready_logits && ready_condition.has_value() &&
             ready_condition->wasAlreadyEmitted());
        const int first_token_output_budget_cost =
            first_condition_was_already_emitted ? 0 : 1;
        const int pre_sample_effective_draft_count =
            decode_step_token_budget_ > 0
                ? std::min(
                      transaction_draft_capacity,
                      std::max(0, decode_step_token_budget_ -
                                      first_token_output_budget_cost))
                : transaction_draft_capacity;
        /*
         * Admit only a transaction that can actually reach grouped verification.
         * A one-token response boundary can collapse the speculative width to
         * zero and return the already device-owned target sample directly. It
         * must not leave behind a resident controller that no graph transaction
         * can complete. This edge remains before every draft sample and verifier
         * launch, so the first compact reducer still consumes an initialized
         * response/controller ledger.
         */
        const bool admit_device_generation_this_step =
            use_grouped_outcome_device_resident_publication_verifier &&
            pre_sample_effective_draft_count > 0;
        if (device_generation_admission_.awaitsAdmission() &&
            admit_device_generation_this_step)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "device_generation_execution_policy_selections",
                1.0,
                "decode",
                runner_->primaryDeviceId().toString(),
                {{"policy",
                  deviceGenerationExecutionPolicyName(
                      generation_execution_policy)},
                 {"topology",
                  generation_loop_topology ==
                          DeviceGenerationLoopTopology::DynamicDepth
                      ? "dynamic_depth"
                      : "fixed_depth"},
                 {"selection_boundary", "pre_first_draft"}});
        }
        if (!admitScalarDeviceResidentGeneration(
                admit_device_generation_this_step,
                first_condition_was_already_emitted
                    ? sampling_math::
                          DeviceGenerationLeadingRowDisposition::AlreadyEmitted
                    : sampling_math::
                          DeviceGenerationLeadingRowDisposition::PendingResponse))
        {
            return fail_after_checkpoint(
                last_error_.empty()
                    ? "GPU grouped MTP generation admission failed"
                    : last_error_);
        }
        constexpr int32_t kDeferredMTPFirstTokenShadow = -3;
        const bool verifier_accepts_device_first_token =
            use_grouped_outcome_device_resident_publication_verifier;
        const bool can_defer_stochastic_first_host_read =
            stochastic_device_verify &&
            verifier_accepts_device_first_token &&
            runner_->primaryDeviceId().is_gpu() &&
            pre_sample_effective_draft_count > 0 &&
            stop_tokens_.size() <=
                static_cast<size_t>(
                    sampling_math::kSpeculativeBatchMaxStopTokens) &&
            runner_->supportsMTPDeviceDraftTokenInput();
        const bool can_defer_greedy_first_host_read =
            !stochastic_verify &&
            use_grouped_outcome_device_resident_publication_verifier &&
            runner_->primaryDeviceId().is_gpu() &&
            pre_sample_effective_draft_count > 0 &&
            stop_tokens_.size() <=
                static_cast<size_t>(
                    sampling_math::kSpeculativeBatchMaxStopTokens) &&
            runner_->supportsMTPDeviceDraftTokenInput();

        int32_t first_token = -1;
        bool first_token_device_target_slot_available = false;
        const bool first_token_is_already_emitted_condition =
            first_condition_was_already_emitted;
        if (use_pending_condition_row)
        {
            first_token = condition_token;
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_pending_condition_rows",
                1.0,
                "decode",
                {},
                {{"token", std::to_string(first_token)}});
        }
        else if (use_ready_logits && ready_condition.has_value() &&
                 ready_condition->host_token.has_value())
        {
            first_token = *ready_condition->host_token;
            PerfStatsCollector::addCounter("mtp", "first_token_ready_cache_hits", 1.0, "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_ready_cache_token",
                1.0,
                "decode",
                {},
                {{"token", std::to_string(first_token)}});
        }
        else if (use_ready_logits && ready_condition.has_value() &&
                 ready_condition->isDeviceResidentOnly())
        {
            /*
             * The terminal generation loop sampled this condition on device
             * after committing the exact previous verifier prefix. Its fused
             * publisher already refreshed canonical target slot zero and the
             * logical-state mailbox. Keep a sentinel only in diagnostic host
             * vectors; every production sidecar/verifier consumer below reads
             * the authenticated device sources.
             */
            first_token = kDeferredMTPFirstTokenShadow;
            first_token_device_target_slot_available = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_resident_continuation_uses",
                1.0,
                "decode",
                {},
                {{"publication_generation",
                  std::to_string(
                      ready_condition->resident_state
                          ->publication_generation)},
                 {"host_token_materializations", "0"}});
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
                    if (active_sampling_params_.has_penalties() &&
                        !runner_->applyDeviceOwnedMTPPenaltiesToLogitRows(
                            DeviceLogitsSource::Main,
                            /*row_count=*/1,
                            MTPRequestPenaltyPolicy{
                                .presence_penalty =
                                    active_sampling_params_.presence_penalty,
                                .frequency_penalty =
                                    active_sampling_params_.frequency_penalty,
                            }))
                    {
                        return fail_after_checkpoint(
                            "MTP stochastic first-token device-history penalty application failed");
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
                             {"threshold", formatStochasticThreshold(first_token_threshold)},
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
                    first_token_device_target_slot_available = true;
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
                        if (use_serial_sample_equivalent_host_stochastic)
                        {
                            const auto distribution =
                                sampler_.compute_distribution(
                                    main_logits,
                                    static_cast<size_t>(vocab),
                                    active_sampling_params_);
                            const int logical_position =
                                transaction_base_cached_tokens;
                            const float threshold =
                                sample_threshold_for_position(
                                    sampler_,
                                    logical_position);
                            first_token = sampleMTPDistributionWithThreshold(
                                distribution,
                                threshold);
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "first_token_stochastic_serial_equivalent_host_samples",
                                1.0,
                                "decode",
                                {},
                                {{"logical_position", std::to_string(logical_position)},
                                 {"threshold", formatStochasticThreshold(threshold)},
                                 {"sampled_token", std::to_string(first_token)},
                                 {"position_owner", "transaction_base_cached_tokens"},
                                 {"source", "cpu_rank_terminal_logits"}});
                        }
                        else
                        {
                            first_token = sampler_.sample(
                                main_logits,
                                static_cast<size_t>(vocab),
                                active_sampling_params_);
                        }
                    }
                    PerfStatsCollector::addCounter("mtp", "first_token_stochastic_samples", 1.0, "decode");
                }
            }
            else
            {
                if (use_sampling_penalties)
                {
                    const bool penalty_ok =
                        runner_->primaryDeviceId().is_gpu()
                            ? runner_->applyDeviceOwnedMTPPenaltiesToLogitRows(
                                  DeviceLogitsSource::Main,
                                  /*row_count=*/1,
                                  MTPRequestPenaltyPolicy{
                                      .presence_penalty =
                                          active_sampling_params_.presence_penalty,
                                      .frequency_penalty =
                                          active_sampling_params_.frequency_penalty,
                                  })
                            : runner_->applyPenaltiesOnDevice(
                                  sampler_.compute_penalty_map(
                                      active_sampling_params_,
                                      vocab),
                                  vocab);
                    if (!penalty_ok)
                    {
                        return fail_after_checkpoint(
                            runner_->primaryDeviceId().is_gpu()
                                ? "MTP first-token device-history penalty application failed"
                                : "MTP first-token CPU penalty application failed");
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
                            "MTP first-token GPU sampling failed; device target-slot deferred sampling failed; host logits sampling is CPU-only");
                    }
                    first_token = kDeferredMTPFirstTokenShadow;
                    first_token_device_target_slot_available = true;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "first_token_greedy_deferred_host_reads",
                        1.0,
                        "decode");
                }
                else if (verifier_accepts_device_first_token &&
                         runner_->primaryDeviceId().is_gpu() &&
                         runner_->supportsMTPDeviceDraftTokenInput())
                {
                    /*
                     * Keep a persistent device owner even when the request's
                     * remaining output budget clamps speculative depth to zero.
                     * The host shadow is still returned because it is the user
                     * visible result and feeds sampler history, but shifted-MTP
                     * publication and the main-model state advance below both
                     * consume this target slot directly.  Restricting target-slot
                     * sampling to positive draft depth made the final one-token
                     * transaction fall back into the obsolete scalar GPU path.
                     */
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "sample_first_token_greedy_device_target_slot_shadow",
                        "decode");
                    if (!runner_->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                            /*target_sample_slot=*/0,
                            &first_token))
                    {
                        return fail_after_checkpoint(
                            "MTP first-token GPU sampling failed; device target-slot shadow sampling failed; host logits sampling is CPU-only");
                    }
                    first_token_device_target_slot_available = true;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "first_token_greedy_device_target_slot_shadow_samples",
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
                    if (runner_->primaryDeviceId().is_gpu())
                    {
                        return fail_after_checkpoint(
                            "MTP first-token GPU sampling failed; host logits sampling is CPU-only");
                    }
                    PerfStatsCollector::addCounter("mtp", "first_token_cpu_host_samples", 1.0, "decode");
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

        /**
         * Compose one grouped-verifier token row from device-owned producers.
         *
         * Target sample slot zero is the canonical verifier condition mailbox.
         * The initial sampler writes it directly; accepted-state publication
         * refreshes it in the same captured kernel that publishes a ready bonus
         * or rejected correction token. The verifier therefore has one stable
         * pointer topology across every transaction and cannot reconstruct row
         * zero from a host response shadow. Draft rows always come from the
         * resident draft sample bank beginning at slot zero.
         */
        auto prepare_grouped_gpu_verifier_input_tokens =
            [&](const MTPSpecDecodeVerifierInputPlan &input_plan,
                const char *consumer,
                std::string *error) -> const void *
        {
            auto fail = [&](std::string message) -> const void *
            {
                if (error)
                    *error = std::move(message);
                return nullptr;
            };

            if (!runner_->primaryDeviceId().is_gpu())
            {
                return fail(
                    "Device-resident grouped-verifier token composition requires a GPU runner");
            }
            if (!input_plan.ok ||
                input_plan.total_verifier_input_tokens <= 1)
            {
                return fail(
                    "Device-resident grouped-verifier token composition requires a valid grouped row plan");
            }

            DeviceMTPVerifierInputBatchRequest request{
                .request_id = 0,
                .first_token = -1,
                .first_token_from_device = true,
                .first_target_sample_slot = 0,
                .first_draft_slot = 0,
                .draft_token_count =
                    input_plan.total_verifier_input_tokens - 1,
                .total_verifier_input_tokens =
                    input_plan.total_verifier_input_tokens,
            };

            if (!first_token_device_target_slot_available &&
                (!first_token_resident_state.has_value() ||
                 !first_token_resident_state->valid() ||
                 !first_token_resident_state->coversRequest(0)))
            {
                return fail(
                    "GPU grouped verifier has no device-owned condition transaction for canonical target slot zero");
            }

            const void *tokens =
                runner_->prepareMTPVerifierInputTokenBatchOnDevice(
                    &request,
                    /*request_count=*/1,
                    input_plan.total_verifier_input_tokens);
            if (!tokens)
            {
                return fail(
                    "GPU grouped verifier rejected canonical device target slot zero");
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "grouped_verifier_first_token_device_sources",
                1.0,
                "decode",
                {},
                {{"source", "canonical_target_slot"},
                 {"consumer", consumer ? consumer : "unknown"},
                 {"total_tokens",
                  std::to_string(
                      input_plan.total_verifier_input_tokens)}});
            return tokens;
        };

        int speculative_draft_count = transaction_draft_capacity;
        bool draft_count_budget_limited = false;
        if (decode_step_token_budget_ > 0)
        {
            const int budgeted_speculative_outputs =
                std::max(0, decode_step_token_budget_ -
                                  first_token_output_budget_cost);
            draft_count_budget_limited =
                budgeted_speculative_outputs <
                requested_speculative_draft_count;
            if (draft_count_budget_limited)
            {
                const bool resident_controller_owns_commit_budget =
                    use_grouped_outcome_device_resident_publication_verifier &&
                    runner_->primaryDeviceId().is_gpu() &&
                    budgeted_speculative_outputs > 0;
                if (resident_controller_owns_commit_budget)
                {
                    /*
                     * The device selector names the complete transaction width.
                     * Trimming sidecars here would make host response budget a
                     * second graph-geometry authority and leave captured token
                     * rows narrower than the resident selector. Execute the
                     * selected transaction and let its device commit budget
                     * limit serial-visible output and state publication.
                     */
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "draft_budget_delegated_to_device_controller",
                        1.0,
                        "decode",
                        {},
                        {{"selected_depth",
                          std::to_string(requested_speculative_draft_count)},
                         {"speculative_output_budget",
                          std::to_string(budgeted_speculative_outputs)},
                         {"token_budget",
                          std::to_string(decode_step_token_budget_)}});
                }
                else
                {
                    speculative_draft_count = std::min(
                        speculative_draft_count,
                        budgeted_speculative_outputs);
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
        }

        if (speculative_draft_count == 0)
        {
            PerfStatsCollector::addCounter("mtp", "budget_limited_direct_emits", 1.0, "decode");
            PerfStatsCollector::addCounter("mtp", "output_tokens", 1.0, "decode");

            if (first_token == kDeferredMTPFirstTokenShadow)
            {
                /*
                 * A one-token response budget has no grouped verifier ledger in
                 * which to return this already-sampled condition token. Observe
                 * exactly the compact mailbox row at the public result boundary;
                 * execution below continues to consume the mailbox and target
                 * slot directly, so this host value never becomes an execution
                 * input or a replacement state authority.
                 */
                if (!first_token_resident_state.has_value() ||
                    !first_token_resident_state->coversRequest(0) ||
                    !runner_->observeDeviceResidentNextConditionTokens(
                        *first_token_resident_state,
                        /*request_count=*/1,
                        &first_token) ||
                    first_token < 0 || first_token >= vocab)
                {
                    return fail_after_checkpoint(
                        "MTP GPU budget-limited direct emit could not materialize its terminal resident token");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "budget_limited_direct_emit_terminal_token_materializations",
                    1.0,
                    "decode",
                    runner_->primaryDeviceId().toString(),
                    {{"boundary", "host_result"},
                     {"bytes", std::to_string(sizeof(first_token))},
                     {"execution_authority", "device_mailbox"}});
            }

            const bool first_token_is_stop =
                std::find(stop_tokens_.begin(), stop_tokens_.end(), first_token) != stop_tokens_.end();
            if (!first_token_is_stop)
            {
                /*
                 * Direct emit publishes exactly one shifted sidecar row before
                 * the main graph advances with first_token.  Prefix-cache and
                 * device-resident verifier-base paths can expose a host
                 * runner position that is already staged past the sidecar KV
                 * head, so the precondition must be anchored to the snapshot
                 * that owns the terminal hidden row we are about to publish.
                 *
                 * snapshotShiftedMTPTokens() returns the shifted sidecar count
                 * represented by that checkpoint.  The sequential shifted-row
                 * committer expects position_offset - 1 rows to be resident
                 * when already_appended_tokens is zero, hence the +1 below.
                 */
                const int base_sidecar_position =
                    snapshotShiftedMTPTokens(verifier_base_checkpoint) + 1;
                bool shifted_commit_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "budget_limited_direct_emit_shifted_commit",
                        "decode");
                    if (runner_->primaryDeviceId().is_gpu())
                    {
                        /*
                         * Ready and rejection-correction tokens are owned by the
                         * resident publication mailbox, not by their host response
                         * shadows. Preserve the token in the persistent target
                         * arena before shifted-cache publication consumes that
                         * one-shot mailbox. This is an event-ordered
                         * D2D handoff; the host scalar is never uploaded.
                         */
                        if (first_token_resident_state.has_value())
                        {
                            if (!runner_
                                     ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                                         *first_token_resident_state,
                                         /*request_index=*/0,
                                         /*target_sample_slot=*/0))
                            {
                                return fail_after_checkpoint(
                                    "MTP GPU budget-limited direct emit could not publish its resident condition token to the device target slot");
                            }
                            first_token_device_target_slot_available = true;
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "budget_limited_direct_emit_resident_target_publications",
                                1.0,
                                "decode",
                                {},
                                {{"source",
                                  use_pending_condition_row
                                      ? "pending_condition"
                                      : "ready_token"},
                                 {"transfer", "d2d"}});
                        }
                        if (!first_token_device_target_slot_available)
                        {
                            return fail_after_checkpoint(
                                "MTP GPU budget-limited direct emit has no device-owned target token");
                        }
                        shifted_commit_ok =
                            first_token_resident_state.has_value()
                                ? runner_
                                      ->commitMTPShiftedRowFromDeviceResidentLogicalState(
                                          *first_token_resident_state,
                                          /*request_index=*/0,
                                          /*already_appended_tokens=*/0,
                                          /*allow_speculative_discard=*/true)
                                : runner_
                                      ->commitMTPShiftedRowFromDeviceTargetSample(
                                          /*target_sample_slot=*/0,
                                          /*already_appended_tokens=*/0,
                                          /*allow_speculative_discard=*/true);
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
                    if (runner_->primaryDeviceId().is_gpu())
                    {
                        /*
                         * Publish the same target slot consumed by shifted-MTP
                         * maintenance together with the canonical pre-forward KV
                         * position, then consume that typed mailbox in every
                         * participant's captured main graph.
                         */
                        advance_ok =
                            runner_
                                ->advanceMTPMainConditionFromDeviceTargetSample(
                                    first_token,
                                    /*target_sample_slot=*/0);
                    }
                    else
                    {
                        advance_ok = runner_->forward(&first_token, 1);
                    }
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
            currentDecodeTransactionPositionForPlanning(
                "speculative sidecar",
                &base_position_error);
        if (!maybe_base_sidecar_position)
            return fail_after_checkpoint(base_position_error);
        const int base_sidecar_position = *maybe_base_sidecar_position;
        bool first_token_is_stop =
            first_token != kDeferredMTPFirstTokenShadow &&
            std::find(stop_tokens_.begin(), stop_tokens_.end(), first_token) != stop_tokens_.end();
        if (first_token_is_stop &&
            !first_token_is_already_emitted_condition)
        {
            /*
             * A host-visible stop token selected from the current target row is
             * already the serial decode answer.  No speculative sidecar row or
             * grouped verifier row may become part of live state, and running
             * them would only add work after the request is known complete.
             * Deferred device-token lanes still flow through the compact
             * outcome reducer because the host cannot inspect the token here.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_stop_direct_completes",
                1.0,
                "decode",
                {},
                {{"requested_draft_tokens",
                  std::to_string(requested_speculative_draft_count)},
                 {"stochastic_verify", stochastic_verify ? "true" : "false"},
                 {"policy_path",
                  use_grouped_outcome_device_resident_publication_verifier
                      ? "grouped_outcome_device_resident_publication"
                      : "grouped_outcome_host_publication"}});
            PerfStatsCollector::addCounter(
                "mtp",
                "output_tokens",
                1.0,
                "decode");
            if (auto commit_error = commit_mtp_transaction_outputs(
                    "first_token_stop_direct",
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
            return result;
        }
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(static_cast<size_t>(speculative_draft_count) + 1);
        draft_tokens.push_back(first_token);
        Sampler draft_sampler = sampler_;
        if ((stochastic_verify || use_sampling_penalties) &&
            !first_token_is_already_emitted_condition &&
            first_token != kDeferredMTPFirstTokenShadow)
        {
            draft_sampler.record_token(first_token);
        }

        std::vector<PrefixStateSnapshot> sidecar_checkpoints;
        sidecar_checkpoints.reserve(1);
        std::vector<std::vector<SamplingDistributionEntry>> host_mtp_draft_distributions(
            static_cast<size_t>(std::max(0, speculative_draft_count)));
        constexpr int32_t kDeferredMTPDraftTokenShadow = -2;
        std::string mtp_token_sampling_error;
        const bool use_greedy_device_draft_slots =
            use_grouped_outcome_device_resident_publication_verifier &&
            !stochastic_verify &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsMTPDeviceDraftTokenInput();

        auto sample_mtp_token = [&](int draft_idx, bool defer_host_read) -> int32_t
        {
            mtp_token_sampling_error.clear();
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
                    if (use_serial_sample_equivalent_host_stochastic)
                    {
                        const int logical_position =
                            transaction_base_cached_tokens + 1 + draft_idx;
                        const float threshold =
                            sample_threshold_for_position(
                                draft_sampler,
                                logical_position);
                        token = sampleMTPDistributionWithThreshold(
                            distribution,
                            threshold);
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "mtp_token_stochastic_position_keyed_host_samples",
                            1.0,
                            "decode",
                            {},
                            {{"draft_idx", std::to_string(draft_idx)},
                             {"logical_position", std::to_string(logical_position)},
                             {"threshold", formatStochasticThreshold(threshold)}});
                    }
                    else
                    {
                        token = draft_sampler.sample_from_distribution(distribution);
                    }
                }
                PerfStatsCollector::addCounter("mtp", "mtp_token_stochastic_samples", 1.0, "decode");
                return token;
            }

            if (use_sampling_penalties)
            {
                const bool penalty_ok =
                    runner_->primaryDeviceId().is_gpu()
                        ? runner_->applyDeviceOwnedMTPBranchPenaltiesToLogits(
                              draft_idx,
                              MTPRequestPenaltyPolicy{
                                  .presence_penalty =
                                      active_sampling_params_.presence_penalty,
                                  .frequency_penalty =
                                      active_sampling_params_.frequency_penalty,
                              })
                        : runner_->applyPenaltiesToMTPLogitsOnDevice(
                              draft_sampler.compute_penalty_map(
                                  active_sampling_params_,
                                  vocab),
                              vocab);
                if (!penalty_ok)
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
                    mtp_token_sampling_error =
                        "MTP draft-token GPU sampling failed; device draft-slot sampling failed; host logits sampling is CPU-only";
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
                mtp_token_sampling_error =
                    "MTP draft-token penalized GPU sampling failed";
                return -1;
            }

            if (runner_->primaryDeviceId().is_gpu())
            {
                mtp_token_sampling_error =
                    "MTP draft-token GPU sampling failed; host logits sampling is CPU-only";
                return -1;
            }

            PerfStatsCollector::addCounter("mtp", "mtp_token_cpu_host_samples", 1.0, "decode");
            const float *mtp_logits = runner_->mtpLogits();
            if (!mtp_logits)
            {
                mtp_token_sampling_error = "No MTP logits available";
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
         * Every GPU MTP lane hands sidecar ownership to its next consumer with
         * a stream event. Penalty kernels are ordinary device consumers and
         * republish the same handoff after mutation; they are not a reason to
         * drain the producer stream on the host.
         */
        const bool use_device_resident_sidecar_stream_handoff =
            runner_->primaryDeviceId().is_gpu() &&
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
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsMTPDeviceDraftTokenInput() &&
            (stochastic_device_verify || use_greedy_device_draft_slots);
        const bool use_resident_pending_condition_sidecar =
            use_pending_condition_row &&
            pending_condition_has_resident_state &&
            use_device_draft_token_sidecar;
        const bool use_resident_ready_condition_sidecar =
            ready_condition_has_resident_state &&
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
              prelaunch_matches_resident_state(
                  ready_condition->resident_state)) ||
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
            verifier_accepts_device_first_token;
        const bool can_defer_greedy_draft_host_reads =
            !stochastic_verify &&
            use_greedy_device_draft_slots &&
            verifier_accepts_device_first_token &&
            runner_->primaryDeviceId().is_gpu();
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
                        if (use_resident_ready_condition_sidecar)
                        {
                            sidecar_ok = runner_
                                ->forwardMTPFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlot(
                                    *ready_condition->resident_state,
                                    /*request_index=*/0,
                                    draft_idx,
                                    sample_host_shadow);
                        }
                        else if (use_resident_pending_condition_sidecar)
                        {
                            sidecar_ok = runner_
                                ->forwardMTPFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlot(
                                    *pending_condition_resident_state,
                                    /*request_index=*/0,
                                    draft_idx,
                                    sample_host_shadow);
                        }
                        else if (first_token_device_target_slot_available &&
                                 use_greedy_device_draft_slots)
                        {
                            sidecar_ok =
                                runner_
                                    ->forwardMTPFromDeviceTargetAtLivePositionAndSampleGreedyToDeviceDraftSlot(
                                    /*target_sample_slot=*/0,
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
                                    *ready_condition->resident_state,
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
                        else if (first_token_device_target_slot_available)
                        {
                            sidecar_ok =
                                use_device_draft_token_sidecar &&
                                runner_
                                    ->forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
                                        /*target_sample_slot=*/0);
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
                            /*
                             * A GPU sidecar may never reconstruct a device
                             * token/position pair from host shadows. CPU keeps
                             * its ordinary host-owned sidecar contract.
                             */
                            sidecar_ok =
                                runner_->primaryDeviceId().is_gpu()
                                    ? false
                                    : runner_->forwardMTPForDeviceSampling(
                                          draft_tokens.back());
                        }
                    }
                    else if (use_resident_ready_condition_sidecar)
                    {
                        sidecar_ok =
                            runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                *ready_condition->resident_state,
                                /*request_index=*/0);
                    }
                    else if (use_resident_pending_condition_sidecar)
                    {
                        sidecar_ok =
                            runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                *pending_condition_resident_state,
                                /*request_index=*/0);
                    }
                    else if (first_token_device_target_slot_available &&
                             use_device_draft_token_sidecar)
                    {
                        sidecar_ok = runner_
                            ->forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
                                /*target_sample_slot=*/0);
                    }
                    else
                    {
                        sidecar_ok =
                            runner_->primaryDeviceId().is_gpu()
                                ? false
                                : runner_->forwardMTP(draft_tokens.back());
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
                        if (use_greedy_device_draft_slots)
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
                                runner_
                                    ->forwardMTPFromDeviceDraftAtLivePositionAndSampleGreedyToDeviceDraftSlot(
                                    draft_idx - 1,
                                    /*position_offset=*/draft_idx,
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
                            runner_
                                ->forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
                                draft_idx - 1,
                                /*position_offset=*/draft_idx);
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
            if (runner_->primaryDeviceId().is_gpu())
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "sidecar_iteration_host_flushes_avoided",
                    1.0,
                    "decode",
                    {},
                    {{"draft_idx", std::to_string(draft_idx)}});
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
                if (use_grouped_decode_equivalent_outcome_verifier &&
                    runner_->supportsMTPSidecarPreservesMainState())
                {
                    /*
                     * Grouped decode-equivalent publication still uses the main
                     * verifier base checkpoint, but graph-native sidecars prove
                     * they do not mutate that base while drafting. Capturing a
                     * post-sidecar payload checkpoint here would only export
                     * hybrid KV/GDN state that grouped publication does not use.
                     */
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "post_sidecar_checkpoint_skipped_sidecar_preserved",
                        1.0,
                        "decode",
                        {},
                        {{"verifier_path",
                          "grouped_decode_equivalent_outcome"}});
                }
                else
                {
                    PerfStatsCollector::ScopedTimer timer("mtp", "capture_post_sidecar_prefix_state", "decode");
                    if (!runner_->ensureMTPCheckpointTerminalHidden())
                    {
                        return fail_after_checkpoint(
                            "MTP decode could not materialize post-sidecar terminal hidden");
                    }
                    std::string position_error;
                    const auto capture_request =
                        current_checkpoint_capture_request(
                            "post-sidecar checkpoint capture",
                            &position_error);
                    if (!capture_request)
                        return fail_after_checkpoint(position_error);
                    sidecar_checkpoints.push_back(
                        runner_->captureLivePrefixCheckpoint(*capture_request));
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
                const bool greedy_verifier_can_consume_device_token =
                    can_defer_greedy_draft_host_reads &&
                    use_device_draft_token_sidecar;
                const bool defer_draft_host_read =
                    (can_defer_stochastic_draft_host_reads &&
                     !next_sidecar_needs_host_token) ||
                    greedy_verifier_can_consume_device_token;
                mtp_token = sample_mtp_token(draft_idx, defer_draft_host_read);
            }
            if (mtp_token < 0)
            {
                if (mtp_token != kDeferredMTPDraftTokenShadow)
                    return fail_after_checkpoint(
                        mtp_token_sampling_error.empty()
                            ? "No MTP logits available"
                            : mtp_token_sampling_error);
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
            runner_->primaryDeviceId().is_gpu() &&
            use_device_resident_sidecar_stream_handoff;
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
                 {"verifier_path", "grouped_decode_equivalent_outcome"}});
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
                const void *verifier_input_tokens_device = nullptr;
                if (runner_->primaryDeviceId().is_gpu())
                {
                    /*
                     * This diagnostic executes the same verifier graph as the
                     * production transaction. GPU verifier metadata therefore
                     * has the same ownership contract: token IDs and their
                     * position row are composed in persistent runner storage,
                     * after waiting for the target/draft sample producers.
                     * Calling the host-token forward entrypoint here leaves the
                     * position row unpublished and no longer represents a legal
                     * GPU verifier invocation.
                     */
                    std::string preparation_error;
                    verifier_input_tokens_device =
                        prepare_grouped_gpu_verifier_input_tokens(
                            verifier_input_plan,
                            "sidecar_preservation_probe",
                            &preparation_error);
                    if (!verifier_input_tokens_device)
                    {
                        LOG_ERROR("[OrchestrationRunner] "
                                  << preparation_error);
                        return std::nullopt;
                    }
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(true, verifier_row_count))
                    return std::nullopt;
                if (!runner_->setComputeAllPositionLogits(true))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return std::nullopt;
                }
                MTPVerifierForwardExecutionOptions forward_options;
                forward_options.device_token_ids =
                    verifier_input_tokens_device;
                const MTPVerifierForwardExecutionResult forward_result =
                    executeMTPSpecVerifierForward(
                        *runner_,
                        verifier_input_plan,
                        forward_options);
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
                            << "/dev_hybrid="
                            << ((block.device_hybrid_storage ||
                                 block.device_hybrid_allocation)
                                    ? "1"
                                    : "0");
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
            PrefixRuntimeStateSnapshot serial_probe_after_state_prefix;
            bool have_serial_probe_after_state_prefix = false;
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
                struct ReplayCacheCounts
                {
                    int main = -1;
                    int shifted = -1;
                };
                auto current_replay_cache_counts = [&]() -> ReplayCacheCounts
                {
                    ReplayCacheCounts counts;
                    const PrefixRuntimeStateSnapshot probe =
                        runner_->prefixStateProbe();
                    for (const PrefixKVCacheProbe &cache : probe.kv_caches)
                    {
                        for (const PrefixKVLayerProbe &layer : cache.layers)
                        {
                            if (layer.seq_idx == 0)
                            {
                                counts.main =
                                    std::max(
                                        counts.main,
                                        layer.cached_tokens);
                            }
                        }
                    }
                    for (const PrefixKVCacheProbe &cache : probe.mtp_kv_caches)
                    {
                        for (const PrefixKVLayerProbe &layer : cache.layers)
                        {
                            if (layer.seq_idx == 0)
                            {
                                counts.shifted =
                                    std::max(
                                        counts.shifted,
                                        layer.cached_tokens);
                            }
                        }
                    }
                    return counts;
                };

                /*
                 * A verifier base can already contain the speculative shifted
                 * row for its first replay input.  Re-appending that row advances
                 * shifted KV one token ahead of grouped publication and changes
                 * the next MTP condition.  Decide from both canonical cache
                 * counts: reuse when shifted equals main, append only when it
                 * trails main by exactly one, and reject every other lifecycle
                 * shape as an oracle failure.
                 */
                const ReplayCacheCounts cache_counts =
                    current_replay_cache_counts();
                const MTPShiftedReplayRowPlan shifted_plan =
                    planMTPShiftedReplayRow(
                        cache_counts.main,
                        cache_counts.shifted);
                if (!shifted_plan)
                {
                    continuation_failure_detail =
                        std::string("shifted MTP replay lifecycle invalid in ") +
                        context +
                        " token_index=" + std::to_string(token_index) +
                        " main_before=" +
                        std::to_string(cache_counts.main) +
                        " shifted_before=" +
                        std::to_string(cache_counts.shifted) +
                        " reason=" + shifted_plan.reason;
                    return false;
                }
                if (shifted_plan.action ==
                    MTPShiftedReplayRowAction::ReuseResidentRow)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "commit_replay_check_shifted_row_reuses",
                        1.0,
                        "decode",
                        {},
                        {{"path", path},
                         {"context", context},
                         {"token_index", std::to_string(token_index)}});
                    return true;
                }

                constexpr int shifted_rows_committed_in_this_replay = 0;
                const int replay_position_offset =
                    shifted_plan.append_position_offset;
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
                    if (runner_->primaryDeviceId().is_gpu())
                    {
                        constexpr int kDiagnosticTargetSampleSlot = 0;
                        ok = runner_
                                 ->stageStochasticTargetTokenForDeviceSampling(
                                     token,
                                     kDiagnosticTargetSampleSlot) &&
                             runner_
                                 ->commitMTPShiftedRowFromDeviceTargetSample(
                                     kDiagnosticTargetSampleSlot,
                                     shifted_rows_committed_in_this_replay,
                                     /*allow_speculative_discard=*/true);
                    }
                    else
                    {
                        ok = runner_
                                 ->commitMTPShiftedRowFromCurrentTerminalHidden(
                                     token,
                                     shifted_rows_committed_in_this_replay,
                                     /*allow_speculative_discard=*/true,
                                     replay_position_offset);
                    }
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
                        " main_before=" +
                        std::to_string(cache_counts.main) +
                        " shifted_before=" +
                        std::to_string(cache_counts.shifted) +
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
                const std::string summary =
                    derived_next_token_from_deferred_condition
                        ? std::string("MTP commit replay check deferred condition forward failed")
                        : std::string("MTP commit replay check missing expected next token");
                return summary + continuation_failure_suffix();
            }
            expected_next_token = *prepared_next_token;

            const PrefixStateSnapshot &base = *verifier_replay_base_checkpoint;
            if (!runner_->restoreLivePrefixState(base))
            {
                return std::string("MTP commit replay check could not restore verifier base state");
            }
            bool sequential_replay_ok = true;
            int32_t replay_next_token = -1;
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

                /*
                 * Validate every output transition, not merely the state after
                 * the complete transaction. A grouped verifier can emit a wrong
                 * intermediate correction token whose later continuation happens
                 * to reconverge. Feeding that wrong token into the replay oracle
                 * and checking only the final sample would bless the corruption.
                 * Sampling after each serial row proves byte-for-byte token-stream
                 * equivalence at the earliest observable boundary.
                 */
                replay_next_token = runner_->sampleGreedyOnDevice();
                if (replay_next_token < 0)
                {
                    return std::string(
                        "MTP commit replay check row sampling failed at replay index=") +
                           std::to_string(i);
                }
                const int32_t expected_row_next =
                    i + 1 < tokens_to_replay.size()
                        ? tokens_to_replay[i + 1]
                        : expected_next_token;
                if (replay_next_token != expected_row_next)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "commit_replay_check_serial_output_token_mismatches",
                        1.0,
                        "decode",
                        {},
                        {{"path", path},
                         {"replay_index", std::to_string(i)},
                         {"replay_token", std::to_string(replay_token)},
                         {"serial_next", std::to_string(replay_next_token)},
                         {"grouped_next",
                          std::to_string(expected_row_next)}});
                    return std::string(
                               "MTP grouped output token mismatch against serial row replay: path=") +
                           path +
                           " replay_index=" + std::to_string(i) +
                           " replay_token=" + std::to_string(replay_token) +
                           " serial_next=" +
                           std::to_string(replay_next_token) +
                           " grouped_next=" +
                           std::to_string(expected_row_next) +
                           " condition_token=" +
                           std::to_string(condition_token) +
                           " committed_tokens=" +
                           join_tokens(tokens_to_replay) +
                           " committed_probe_before={" +
                           summarize_probe(committed_probe_before) + "}" +
                           " replay_probe={" +
                           summarize_probe(runner_->prefixStateProbe()) + "}" +
                           (debug_context.empty()
                                ? std::string{}
                                : " " + debug_context);
                }
                if (!have_serial_probe_after_state_prefix &&
                    static_cast<int>(i + 1) == replay_prefix_count)
                {
                    /*
                     * Forced-reject grouped MTP publishes only the accepted
                     * verifier prefix, then carries the rejected correction as
                     * the next condition token.  Capturing the serial replay
                     * state at that exact prefix boundary lets the failure
                     * message distinguish a bad accepted-row publication from
                     * a bad one-token continuation after an otherwise correct
                     * publication.
                     */
                    serial_probe_after_state_prefix =
                        runner_->prefixStateProbe();
                    have_serial_probe_after_state_prefix = true;
                }
            }
            if (!sequential_replay_ok)
            {
                return std::string("MTP commit replay check sequential replay failed") +
                       continuation_failure_suffix();
            }
            const PrefixRuntimeStateSnapshot serial_probe_after_full_replay =
                runner_->prefixStateProbe();
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
                       " committed_probe_before={" + summarize_probe(committed_probe_before) + "}" +
                       " committed_restore_probe={" + summarize_probe(first_committed_restore_probe) + "}" +
                       (have_serial_probe_after_state_prefix
                            ? " serial_probe_after_state_prefix={" +
                                  summarize_probe(serial_probe_after_state_prefix) +
                                  "} state_prefix_gdn_first_mismatch={" +
                                  first_gdn_mismatch(
                                      committed_probe_before,
                                      serial_probe_after_state_prefix,
                                      "committed",
                                      "serial_prefix") +
                                  "}"
                            : std::string{}) +
                       " serial_probe_after_full_replay={" + summarize_probe(serial_probe_after_full_replay) + "}" +
                       " gdn_first_mismatch={" +
                       first_gdn_mismatch(
                           first_committed_restore_probe,
                           serial_probe_after_full_replay,
                           "committed",
                           "serial") +
                       "}" +
                       " committed_snapshot={" + summarize_snapshot(committed_checkpoint) + "}" +
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
                std::ostringstream mismatch;
                mismatch
                    << "MTP commit replay check committed ready-token derivation mismatch"
                    << ": expected=" << expected_next_token
                    << " actual="
                    << (committed_ready_token
                            ? std::to_string(*committed_ready_token)
                            : std::string("none"))
                    << " current_probe={" << summarize_probe(runner_->prefixStateProbe()) << "}"
                    << " committed_probe_before={" << summarize_probe(committed_probe_before) << "}"
                    << " first_committed_restore_probe={"
                    << summarize_probe(first_committed_restore_probe) << "}"
                    << " second_committed_restore_probe={"
                    << summarize_probe(second_committed_restore_probe) << "}"
                    << " committed_snapshot={" << summarize_snapshot(committed_checkpoint) << "}";
                return mismatch.str();
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

            /*
             * Replay diagnostics intentionally replace the CPU live timeline
             * several times. Restore the exact committed checkpoint once more
             * before the host oracle returns to the real decode transaction.
             * GPU grouped publication never enters this host replay diagnostic.
             */
            if (!runner_->restoreLivePrefixState(committed_checkpoint))
            {
                return std::string(
                    "MTP commit replay check could not perform final committed-state restore");
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

        const MTPVisibleStateCommitPlan visible_state_commit_plan =
            planMTPVisibleStateCommit(
                static_cast<int>(draft_tokens.size()),
                first_token_is_already_emitted_condition ? 1 : 0,
                decode_step_token_budget_);
        if (!visible_state_commit_plan)
        {
            return fail_after_checkpoint(
                std::string("MTP visible-state commit planning failed: ") +
                visible_state_commit_plan.reason);
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "visible_state_commit_plans",
            1.0,
            "decode",
            {},
            {{"verifier_rows",
              std::to_string(visible_state_commit_plan.verifier_input_rows)},
             {"max_state_commit_rows",
              std::to_string(visible_state_commit_plan.max_state_commit_rows)},
             {"output_budget",
              std::to_string(
                  visible_state_commit_plan.remaining_output_budget)},
             {"pending_condition_input",
              first_token_is_already_emitted_condition ? "true" : "false"},
             {"response_boundary_clipped",
              visible_state_commit_plan.response_boundary_clipped
                  ? "true"
                  : "false"}});

        if (use_grouped_outcome_host_publication_verifier)
        {
            constexpr const char *verifier_publication_path =
                "grouped_decode_equivalent_host_publication";
            constexpr const char *verifier_replay_check_path =
                verifier_publication_path;
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
                    {{"verifier_path", verifier_publication_path}});
                return true;
            };
            /*
             * This branch is the CPU ownership domain. CUDA and ROCm are
             * admitted only through the grouped device-resident branch below,
             * where verifier publication and continuation remain on device.
             * Keeping a deferred-stream or compact-device option here would
             * recreate the retired host-outcome bridge behind an unreachable
             * condition.
             */
            if (runner_->primaryDeviceId().is_gpu())
            {
                return fail_after_checkpoint(
                    "Grouped host publication is CPU-only");
            }
            if (first_token == kDeferredMTPFirstTokenShadow ||
                std::find(draft_tokens.begin(),
                          draft_tokens.end(),
                          kDeferredMTPDraftTokenShadow) != draft_tokens.end())
            {
                return fail_after_checkpoint(
                    "Grouped host publication received a device-owned token shadow");
            }
            {
                PerfStatsCollector::ScopedTimer verifier_timer(
                    "mtp",
                    "verifier_forward",
                    "decode",
                    {},
                    {{"implementation", verifier_publication_path},
                     {"verifier_path", verifier_publication_path}});
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
                MTPVerifierForwardExecutionOptions verifier_forward_options;
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
                /*
                 * Rank-level LocalTP row sampling consumes the verifier replay
                 * stream through LogitsLocalInfo.  Do that while the
                 * row-indexed all-position logits contract is still active:
                 * disabling the graph mode first is allowed to rebind or clear
                 * the active verifier-logit view, which turns a valid remote
                 * shard winner into the default token 0.  Compact device
                 * outcome reducers are excluded because their captured
                 * penalty-aware argmax already owns row transformation and
                 * sampling on the verifier stream.
                 */
                if (!stochastic_verify && use_sampling_penalties)
                {
                    const int compare_rows =
                        static_cast<int>(draft_tokens.size()) - 1;
                    const int bonus_row = compare_rows;
                    if (!apply_all_position_row_penalties_for_history(
                            compare_rows,
                            bonus_row,
                            "greedy_vllm_penalty_rows_preapplied"))
                    {
                        runner_->setComputeAllPositionLogits(false);
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        return fail_after_checkpoint(
                            "All-position greedy MTP row penalty application failed before cleanup");
                    }
                }
                if (!stochastic_verify)
                {
                    if (!runner_->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                            0,
                            static_cast<int>(sampled_verifier_rows.size()),
                            sampled_verifier_rows.data()))
                    {
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        return fail_after_checkpoint(
                            "All-position MTP verifier could not sample verifier rows before cleanup");
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "greedy_verifier_rows_sampled_before_all_position_cleanup",
                        static_cast<double>(sampled_verifier_rows.size()),
                        "decode",
                        {},
                        {{"verifier_path", verifier_publication_path}});
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
            catchup_request.verifier_path = verifier_publication_path;
            catchup_request.implementation_name = verifier_publication_path;
            catchup_request.verifier_base_checkpoint = &verifier_base_checkpoint;

            Sampler all_position_stochastic_penalty_sampler = sampler_;
            MTPDecodeCatchupGreedyResult catchup;
            if (stochastic_verify)
            {
                if (!stochastic_host_verify)
                {
                    return fail_after_checkpoint(
                        "Grouped host stochastic MTP requires host distribution verification");
                }
                std::vector<MTPRejectionSampleRowResult> stochastic_rows;
                stochastic_rows.reserve(
                    draft_tokens.size() > 0 ? draft_tokens.size() - 1 : 0);
                bool stochastic_stopped_on_output = false;
                std::optional<int32_t> bonus_ready_token;
                all_position_stochastic_penalty_sampler.record_token(first_token);

                if (std::find(stop_tokens_.begin(),
                              stop_tokens_.end(),
                              first_token) != stop_tokens_.end())
                {
                    stochastic_stopped_on_output = true;
                }

                std::vector<SamplingDistributionEntry> host_target_distribution;
                auto build_all_position_target_distribution =
                    [&](int row) -> bool
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
                        {{"implementation",
                          "grouped_decode_equivalent_host_publication"}});
                    host_target_distribution =
                        all_position_stochastic_penalty_sampler
                            .compute_distribution(
                                row_logits,
                                static_cast<size_t>(vocab),
                                active_sampling_params_);
                    return !host_target_distribution.empty();
                };

                for (int draft_idx = 1;
                     !stochastic_stopped_on_output &&
                     draft_idx < static_cast<int>(draft_tokens.size());
                     ++draft_idx)
                {
                    const int row = draft_idx - 1;
                    if (!build_all_position_target_distribution(row))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP target distribution build failed");
                    }

                    const int32_t draft_token =
                        draft_tokens[static_cast<size_t>(draft_idx)];
                    const int logical_position =
                        transaction_base_cached_tokens + draft_idx;
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
                    if (use_serial_sample_equivalent_host_stochastic)
                    {
                        const float sample_threshold =
                            sample_threshold_for_position(
                                sampler_,
                                logical_position);
                        row_result = sampleMTPSerialEquivalentTargetRow(
                            host_target_distribution,
                            draft_token,
                            sample_threshold);
                    }
                    else
                    {
                        const float accept_threshold =
                            accept_threshold_for_position(
                                sampler_,
                                logical_position);
                        const float residual_threshold =
                            residual_threshold_for_position(
                                sampler_,
                                logical_position);
                        row_result = sampleMTPRejectionRowFromDistributions(
                            host_target_distribution,
                            draft_distribution,
                            draft_token,
                            accept_threshold,
                            residual_threshold);
                    }
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
                         {"device_resident", "false"},
                         {"stochastic_coupling",
                          use_serial_sample_equivalent_host_stochastic
                              ? "serial_sample_equivalent"
                              : "vllm_probability_rejection"},
                         {"verifier_path",
                          "grouped_decode_equivalent_host_publication"}});

                    if (use_serial_sample_equivalent_host_stochastic)
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "stochastic_serial_equivalent_host_verifier_rows",
                            1.0,
                            "decode",
                            {},
                            {{"row", std::to_string(row)},
                             {"logical_position", std::to_string(logical_position)}});
                    }

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
                            {{"verifier_path",
                              "grouped_decode_equivalent_host_publication"}});
                    }
                    else
                    {
                        if (output_token < 0)
                        {
                            return fail_after_checkpoint(
                                "All-position stochastic MTP verifier produced no correction token");
                        }
                        sampled_verifier_rows[static_cast<size_t>(row)] =
                            output_token;
                        ++mtp_stats_.stochastic_residual_samples;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            use_serial_sample_equivalent_host_stochastic
                                ? "stochastic_serial_equivalent_correction_host_samples"
                                : "stochastic_residual_host_samples",
                            1.0,
                            "decode",
                            {},
                            {{"row", std::to_string(row)},
                             {"draft_token", std::to_string(draft_token)},
                             {"correction_token", std::to_string(output_token)},
                             {"stochastic_coupling",
                              use_serial_sample_equivalent_host_stochastic
                                  ? "serial_sample_equivalent"
                                  : "vllm_probability_rejection"},
                             {"verifier_path",
                              "grouped_decode_equivalent_host_publication"}});
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

                if (!stochastic_stopped_on_output &&
                    all_rows_verified &&
                    all_rows_accepted)
                {
                    const int bonus_row =
                        static_cast<int>(draft_tokens.size()) - 1;
                    if (!build_all_position_target_distribution(bonus_row))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP bonus distribution build failed");
                    }
                    const int32_t ready_token =
                        sampleMTPDistributionWithThreshold(
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
                        "stochastic_terminal_host_samples",
                        1.0,
                        "decode",
                            {},
                            {{"verifier_path",
                              "grouped_decode_equivalent_host_publication"}});
                }
                catchup = buildAllPositionMTPDecodeCatchupStochasticResult(
                    catchup_request,
                    stochastic_rows,
                    bonus_ready_token);
            }
            else
            {
                catchup = buildAllPositionMTPDecodeCatchupGreedyResult(
                    catchup_request,
                    sampled_verifier_rows);
            }
            if (!catchup.ok)
                return fail_after_checkpoint(catchup.error);

            MTPSpecDecodeMetadataShape metadata_shape;
            metadata_shape.max_requests = 1;
            metadata_shape.max_draft_tokens =
                static_cast<int>(draft_tokens.size());
            const int32_t verifier_base_cached_tokens =
                static_cast<int32_t>(verifier_base_checkpoint.cached_tokens);
            MTPSpecTransactionBatchPlan transaction_plan;
            {
                PerfStatsCollector::ScopedTimer transaction_plan_timer(
                    "mtp",
                    "all_position_transaction_plan_build",
                    "decode",
                    {},
                    {{"source", "grouped_host_catchup"}});
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
            int shifted_publication_commit_count = 0;
            bool first_shifted_row_available_for_publication =
                first_shifted_row_available_from_sidecar;
            if (!first_shifted_row_available_for_publication &&
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
                     * Non-reusable MoE/TP sidecars cannot keep the speculative
                     * row-zero shifted KV append, but the verifier has not yet
                     * published a hidden row for the accepted first token.  The
                     * only serial-decode source for that shifted row is the
                     * verifier-base terminal hidden captured before sidecar
                     * draft work.  Import just that checkpoint hidden row here;
                     * the main KV/GDN/position publication remains owned by the
                     * verifier-row publisher below.
                     */
                    initial_shifted_commit_ok =
                        runner_->commitMTPShiftedRowFromCheckpointTerminalHidden(
                            verifier_base_checkpoint,
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
                shifted_publication_commit_count += 1;
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_initial_shifted_commits",
                    1.0,
                    "decode",
                    {},
                    {{"source", "verifier_base_checkpoint_terminal_hidden"}});
            }

            mutable_step.reuse_initial_mtp_shifted_kv_row =
                first_shifted_row_available_for_publication;
            const MTPSpecStepPlan &step = mutable_step;
            if (!first_token_is_stop &&
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
                shifted_publication_commit_count += accepted_state_count - 1;
            }

            std::string publication_error;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "grouped_outcome_publish_accepted_state_host",
                    "decode");
                if (!runner_->publishGroupedDecodeEquivalentMTPSpecStateBatch(
                        step_plans,
                        &publication_error))
                {
                    return fail_after_checkpoint(
                        std::string("MTP grouped verifier state publication failed: ") +
                        publication_error);
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_host_state_publications",
                    1.0,
                    "decode",
                    {},
                    {{"request_count", std::to_string(step_plans.request_count)},
                     {"accepted_state_count", std::to_string(accepted_state_count)}});
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
                    {{"verifier_path",
                      "grouped_decode_equivalent_host_publication"},
                     {"start_index", std::to_string(replay_start)}});
            }

            std::vector<int32_t> accepted_tokens =
                std::move(catchup.accepted_tokens);
            std::vector<int32_t> verifier_tokens =
                std::move(catchup.verifier_tokens);
            const bool all_speculative_accepted =
                catchup.all_speculative_accepted;
            constexpr bool commit_boundary_clipped = false;
            const int accepted_speculative_prefix =
                catchup.accepted_speculative_prefix;
            const int32_t rejected_verified_token =
                catchup.rejected_verified_token;
            const bool stopped_on_output = catchup.stopped_on_output;
            if (!step.requiresCorrectionReplay() &&
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
                          "grouped_decode_equivalent_host_publication"},
                         {"accepted_state_count",
                          std::to_string(accepted_state_count)},
                         {"committed_output_count",
                          std::to_string(accepted_tokens.size())}});
                }
            }
            const int32_t raw_ready_token = catchup.ready_token;
            int32_t ready_token = raw_ready_token;
            const bool has_deferred_correction_condition =
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
                    {{"verifier_path",
                      "grouped_decode_equivalent_host_publication"},
                     {"raw_ready_token", std::to_string(raw_ready_token)},
                     {"deferred_condition_tokens",
                      std::to_string(deferred_correction_condition_count)}});
                ready_token = -1;
            }
            const int main_forward_token_count =
                catchup.main_forward_token_count + correction_forward_count;
            result.is_complete = result.is_complete || stopped_on_output;
            const int emitted_token_start_index =
                first_token_is_already_emitted_condition ? 1 : 0;
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

            if (!stopped_on_output &&
                has_deferred_correction_condition)
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
                }
                else if (deferred_correction_condition_count > 1)
                {
                    return fail_after_checkpoint(
                        "All-position MTP pending-condition fast path supports one correction row");
                }
            }
            std::optional<std::string> tx_error =
                validate_spec_decode_transaction(
                    verifier_replay_check_path,
                    verifier_publication_path,
                    draft_tokens,
                    accepted_tokens,
                    stopped_on_output || ready_token < 0
                        ? std::optional<int32_t>{}
                        : std::optional<int32_t>{ready_token},
                    all_speculative_accepted,
                    stopped_on_output,
                    accepted_speculative_prefix,
                    commit_boundary_clipped);
            if (tx_error)
            {
                return fail_after_checkpoint(*tx_error);
            }

            ++mtp_stats_.verifier_runs;
            mtp_stats_.verifier_token_count +=
                static_cast<uint64_t>(main_forward_token_count);
            mtp_stats_.last_transaction_draft_depth =
                speculative_draft_count;
            PerfStatsCollector::addCounter("mtp", "verifier_runs", 1.0, "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_tokens",
                static_cast<double>(main_forward_token_count),
                "decode");
            {
                /*
                 * CPU greedy and stochastic verification share the same
                 * decode-equivalent grouped forward and host publication
                 * machinery.  Keep their route telemetry distinct: the
                 * stochastic matrix treats this counter as proof that target
                 * distributions and speculative acceptance were evaluated,
                 * rather than merely observing that some grouped forward ran.
                 */
                const char *grouped_host_verifier_counter =
                    stochastic_verify
                        ? "grouped_decode_equivalent_stochastic_verifier_runs"
                        : "grouped_decode_equivalent_greedy_verifier_runs";
                PerfStatsCollector::addCounter(
                    "mtp",
                    grouped_host_verifier_counter,
                    1.0,
                    "decode",
                    {},
                    {{"verifier_forward_tokens", std::to_string(main_forward_token_count)},
                     {"verifier_rows", std::to_string(sampled_verifier_rows.size())},
                     {"replay_forward_tokens", std::to_string(correction_forward_count)},
                     {"shifted_commits", std::to_string(shifted_publication_commit_count)},
                     {"verify_mode", stochastic_verify ? "stochastic" : "greedy"},
                     {"state_publication", "grouped_host"}});
            }

            recordMTPDepthObservation(
                requested_speculative_draft_count,
                speculative_draft_count,
                accepted_speculative_prefix,
                draft_count_budget_limited,
                /*rollback=*/false);

            if (!all_speculative_accepted && !commit_boundary_clipped)
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
                 {"commit_boundary_clipped",
                  commit_boundary_clipped ? "true" : "false"},
                 {"verifier_state_matches_output", "true"},
                 {"verifier_path", verifier_publication_path},
                 {"catchup_implementation", verifier_publication_path},
                 {"policy_path", "grouped_outcome_host_publication"},
                 {"decode_equivalent_replay_required", "false"},
                 {"correction_replay_tokens", std::to_string(correction_forward_count)},
                 {"deferred_correction_condition_tokens",
                  std::to_string(deferred_correction_condition_count)},
                 {"output_tokens", std::to_string(newly_emitted_token_count)},
                 {"ready_token", std::to_string(ready_token)},
                 {"raw_ready_token", std::to_string(raw_ready_token)},
                 {"pending_condition_input",
                  first_token_is_already_emitted_condition ? "true" : "false"},
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
                if (auto mismatch = verify_committed_prefix_replay(
                        verifier_replay_check_path,
                        accepted_tokens,
                        ready_token,
                        accepted_state_count,
                        replay_context.str()))
                {
                    return fail_after_checkpoint(*mismatch);
                }
            }
            if (auto commit_error = commit_mtp_transaction_outputs(
                    "grouped_decode_equivalent_host_verifier",
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

        if (use_grouped_decode_equivalent_outcome_verifier)
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
            if (use_grouped_outcome_host_publication_verifier)
            {
                /*
                 * LocalTP can reduce greedy verifier rows across sharded
                 * logits, but it still lacks one compact device-resident
                 * outcome reducer spanning every TP participant.  This middle
                 * lane keeps the grouped verifier forward and publishes the
                 * accepted rows from a host-visible MTPSpecStepPlanBatch
                 * through the narrow grouped publication API.
                 */
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_host_publication_uses",
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
                        "Grouped decode-equivalent MTP verifier requires a post-sidecar checkpoint");
                }

                sidecar_checkpoint = &sidecar_checkpoints.front();
                if (!sidecar_checkpoint->valid)
                {
                    return fail_after_checkpoint(
                        "Grouped decode-equivalent MTP verifier received an invalid post-sidecar checkpoint");
                }
            }
            bool restored_verifier_base = sidecar_preserves_main_state;
            if (sidecar_preserves_main_state)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_decode_equivalent_verifier_base_restore_skipped_sidecar_preserved",
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
                    "grouped_decode_equivalent_verifier_restore_base_checkpoint",
                    "decode");
                restored_verifier_base =
                    runner_->restoreLivePrefixState(verifier_base_checkpoint);
                if (restored_verifier_base)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "grouped_decode_equivalent_verifier_base_restores",
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
                    "Grouped decode-equivalent MTP verifier could not restore verifier base checkpoint after sidecar draft");
            }

            if (!grouped_outcome_device_resident_publication)
            {
                if (stochastic_verify &&
                    runner_->primaryDeviceId().is_gpu() &&
                    runner_->supportsDeviceResidentMTPSpecStatePublication() &&
                    !stochastic_device_verify)
                {
                    return fail_after_checkpoint(
                        "Grouped decode-equivalent stochastic MTP verifier requires device-resident distribution verification");
                }
                return fail_after_checkpoint(
                    "Grouped decode-equivalent MTP verifier has no grouped publication path; serial row replay is diagnostic-only and is not a production fallback");
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
                const bool use_serial_sample_equivalent_stochastic =
                    active_sampling_params_.seed != 0;
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
                const bool can_defer_grouped_verifier_sync =
                    !first_token_is_stop;
                ScopedMTPAllPositionVerifierSyncDeferral verifier_sync_deferral(
                    runner_.get(),
                    can_defer_grouped_verifier_sync);
                ScopedMTPAllPositionVerifierTransaction verifier_transaction(
                    runner_.get(),
                    verifier_input_plan);
                auto fail_active_verifier_transaction =
                    [&](std::string message) -> GenerationResult
                {
                    std::string cleanup_error;
                    if (!verifier_transaction.close(&cleanup_error))
                    {
                        if (!message.empty())
                            message += "; ";
                        message += cleanup_error;
                    }
                    /*
                     * Prefix restoration is an exclusive logical-state writer.
                     * Retire the deferred verifier reader before entering the
                     * shared rollback helper or the writer would wait on this
                     * same stack frame's reader admission.
                     */
                    verifier_sync_deferral.close();
                    return fail_after_checkpoint(message);
                };
                if (!verifier_transaction.ready())
                {
                    return fail_active_verifier_transaction(
                        verifier_transaction.error());
                }
                const void *verifier_input_tokens_device = nullptr;
                {
                    PerfStatsCollector::ScopedTimer verifier_timer(
                        "mtp",
                        "grouped_outcome_stochastic_verifier_forward",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"rows", std::to_string(verifier_row_count)}});
                    std::string preparation_error;
                    verifier_input_tokens_device =
                        prepare_grouped_gpu_verifier_input_tokens(
                            verifier_input_plan,
                            "grouped_stochastic_verifier",
                            &preparation_error);
                    if (!verifier_input_tokens_device)
                    {
                        return fail_active_verifier_transaction(
                            preparation_error);
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
                        return fail_active_verifier_transaction(
                            std::string("Grouped-outcome MTP verifier forward failed: ") +
                            verifier_forward.error);
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

                const int bonus_row = compare_rows;
                const MTPRequestPenaltyPolicy verifier_penalty_policy{
                    .presence_penalty =
                        active_sampling_params_.presence_penalty,
                    .frequency_penalty =
                        active_sampling_params_.frequency_penalty,
                };
                if (!runner_
                         ->buildCapturedStochasticVerifierTargetDistributions(
                             compare_rows + 1,
                             active_sampling_params_,
                             verifier_penalty_policy,
                             vocab))
                {
                    return fail_active_verifier_transaction(
                        "Grouped-outcome stochastic MTP captured penalty/distribution transaction failed");
                }

                std::vector<float> accept_thresholds;
                std::vector<float> residual_thresholds;
                accept_thresholds.reserve(static_cast<size_t>(compare_rows));
                residual_thresholds.reserve(static_cast<size_t>(compare_rows));
                for (int row = 0; row < compare_rows; ++row)
                {
                    const int row_logical_position =
                        transaction_base_cached_tokens + 1 + row;
                    if (!use_serial_sample_equivalent_stochastic)
                    {
                        accept_thresholds.push_back(
                            accept_threshold_for_position(
                                sampler_,
                                row_logical_position));
                        residual_thresholds.push_back(
                            residual_threshold_for_position(
                                sampler_,
                                row_logical_position));
                    }
                }

                Sampler bonus_sampler = sampler_;
                const float bonus_threshold =
                    use_serial_sample_equivalent_stochastic
                        ? 0.0f
                        : sample_threshold_for_position(
                              bonus_sampler,
                              transaction_base_cached_tokens +
                                  static_cast<int>(draft_tokens.size()));
                const uint64_t inverse_sample_seed =
                    use_serial_sample_equivalent_stochastic
                        ? static_cast<uint64_t>(active_sampling_params_.seed)
                        : inverse_sample_seed_for_thresholds(
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
                    if (use_serial_sample_equivalent_stochastic)
                    {
                        DeviceStochasticBatchOutcomeRequest request;
                        request.request_id = 0;
                        request.first_target_slot = 0;
                        request.first_draft_slot = 0;
                        request.row_count = compare_rows;
                        request.first_token = -1;
                        request.first_token_from_device = true;
                        /*
                         * Captured verifier preparation has already copied the
                         * selected target-slot or resident-mailbox token into
                         * entry zero of the verifier token row. From this point
                         * onward that materialized row is the sole owner of the
                         * condition token consumed by the serial-equivalent
                         * summary kernel. Keeping either original producer named
                         * here would advertise two resident sources for one
                         * logical token and make stream/lifetime ownership
                         * ambiguous.
                         */
                        request.first_target_sample_slot = -1;
                        request.token_row_offset = 0;
                        request.token_row_stride =
                            verifier_input_plan.total_verifier_input_tokens;
                        request.bonus_target_slot = bonus_row;
                        request.bonus_threshold = bonus_threshold;
                        request.inverse_sample_seed = inverse_sample_seed;
                        request.inverse_sample_first_logical_position = -1;
                        request.derive_thresholds_from_seed = true;
                        request.draw_position_source =
                            DeviceStochasticDrawPositionSource::VerifierBaseSnapshot;
                        request.serial_sample_equivalent = true;
                        request.use_device_draft_tokens = true;
                        request.stop_token_count =
                            static_cast<int>(stop_tokens_.size());
                        for (size_t i = 0; i < stop_tokens_.size(); ++i)
                        {
                            request.stop_tokens[i] = stop_tokens_[i];
                        }
                        resident_outcome_ok =
                            runner_->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                                &request,
                                /*request_count=*/1,
                                &outcome_handle);
                    }
                    else
                    {
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
                }
                std::string verifier_cleanup_error;
                if (!verifier_transaction.close(&verifier_cleanup_error))
                {
                    verifier_sync_deferral.close();
                    return fail_after_checkpoint(verifier_cleanup_error);
                }
                verifier_sync_deferral.close();
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

                DeviceSpeculativeVerifyBatchOutcome device_outcome;
                MTPDecodeCatchupGreedyResult catchup;
                int shifted_publication_commit_count = 0;
                if (plan_.usesLocalTP())
                {
                    const bool first_shifted_row_available_from_sidecar =
                        sidecar_preserves_main_state &&
                        runner_->supportsMTPShiftedRowReuseFromSidecar() &&
                        !first_token_is_stop;
                    const int max_state_commit_rows =
                        static_cast<int>(draft_tokens.size());
                    bool first_shifted_row_available_for_publication =
                        first_shifted_row_available_from_sidecar;
                    if (!first_shifted_row_available_from_sidecar &&
                        !first_token_is_stop &&
                        max_state_commit_rows > 0)
                    {
                        bool initial_shifted_commit_ok = false;
                        {
                            PerfStatsCollector::ScopedTimer timer(
                                "mtp",
                                "grouped_outcome_stochastic_initial_shifted_device_commit",
                                "decode");
                            initial_shifted_commit_ok =
                                runner_->commitMTPInitialShiftedRowFromDeviceOutcome(
                                    verifier_base_checkpoint,
                                    outcome_handle,
                                    /*request_index=*/0,
                                    /*main_forward_token_count=*/max_state_commit_rows,
                                    /*allow_speculative_discard=*/true);
                        }
                        if (!initial_shifted_commit_ok)
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome stochastic MTP device-resident initial shifted-cache commit failed");
                        }
                        first_shifted_row_available_for_publication = true;
                        shifted_publication_commit_count += 1;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_stochastic_initial_shifted_device_commits",
                            1.0,
                            "decode",
                            {},
                            {{"source", "device_outcome_checkpoint_terminal_hidden"}});
                    }
                    if (first_shifted_row_available_for_publication &&
                        max_state_commit_rows > 1)
                    {
                        bool shifted_catchup_ok = false;
                        {
                            PerfStatsCollector::ScopedTimer timer(
                                "mtp",
                                "grouped_outcome_stochastic_shifted_prefix_device_commit",
                                "decode");
                            /*
                             * The sidecar owns row zero.  Rows after that boundary
                             * are prepared from compact device outcome metadata on
                             * the sidecar stream; non-accepted suffix rows are valid
                             * speculative writes and are discarded by shifted-KV
                             * publication's device-derived target count.
                             */
                            shifted_catchup_ok =
                                runner_->commitMTPShiftedRowsFromDeviceOutcome(
                                    outcome_handle,
                                    /*request_index=*/0,
                                    /*already_appended_tokens=*/1,
                                    max_state_commit_rows,
                                    /*main_forward_token_count=*/max_state_commit_rows,
                                    /*allow_speculative_discard=*/true);
                        }
                        if (!shifted_catchup_ok)
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome stochastic MTP device-resident shifted-cache suffix commit failed");
                        }
                        shifted_publication_commit_count +=
                            max_state_commit_rows - 1;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_stochastic_shifted_prefix_device_commits",
                            static_cast<double>(max_state_commit_rows - 1),
                            "decode",
                            {},
                            {{"rows", std::to_string(max_state_commit_rows - 1)}});
                    }
                }

                if (!runner_->supportsDeviceResidentMTPSpecStatePublication())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP requires device-resident accepted-state publication; replay publication is not an accepted production path");
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = outcome_handle;
                publication_request.max_state_commit_rows =
                    visible_state_commit_plan.max_state_commit_rows;
                publication_request.publish_mtp_shifted_kv = true;
                publication_request.penalty_policy =
                    MTPRequestPenaltyPolicy{
                        .presence_penalty =
                            active_sampling_params_.presence_penalty,
                        .frequency_penalty =
                            active_sampling_params_.frequency_penalty,
                    };
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_device_resident_publication_uses",
                    1.0,
                    "decode",
                    {},
                    {{"policy_path", "grouped_outcome_device_resident_publication"},
                     {"sampling", "stochastic"},
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())},
                     {"physical_verifier_rows",
                      std::to_string(
                          publication_request.physicalVerifierRowsPerRequest())}});

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer direct_publish_timer(
                        "mtp",
                        "grouped_outcome_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"request_count", "1"},
                         {"logical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .logicalVerifierRowsPerRequest())}});
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
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())},
                     {"shifted_commits",
                      std::to_string(shifted_publication_commit_count)}});
                /*
                 * A GPU grouped transaction is admitted before its first draft
                 * launch. The compact reducer must therefore publish a
                 * controller-owned outcome, and dynamic execution must consume
                 * the same first-transaction admission edge used to size the
                 * complete captured graph family. Any other state is a broken
                 * lifecycle, never permission to reconstruct the transaction on
                 * the host.
                 */
                if (!publication_request.outcome
                         .device_generation_controller_owned)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP lost device-generation controller ownership after admission");
                }
                if (mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic &&
                    !materialize_dynamic_generation_loop_this_step)
                {
                    return fail_after_checkpoint(
                        "Dynamic grouped-outcome stochastic MTP did not begin at its admitted prefill boundary");
                }

                GenerationResult resident_result =
                    completeDeviceResidentGeneration(
                        publication_request,
                        DeviceGenerationSamplingMode::Stochastic,
                        transaction_base_cached_tokens,
                        requested_speculative_draft_count,
                        speculative_draft_count,
                        std::move(result));
                if (!resident_result.success())
                    return fail_after_checkpoint(resident_result.error);
                return resident_result;
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
                            effectiveMTPMaxDraftDepth(mtp) + 1))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP draft width exceeds the configured verifier graph capacity");
                }

                const bool first_token_deferred =
                    first_token == kDeferredMTPFirstTokenShadow;
                const bool has_deferred_draft_token =
                    std::find(
                        draft_tokens.begin() + 1,
                        draft_tokens.end(),
                        kDeferredMTPDraftTokenShadow) != draft_tokens.end();
                if (runner_->primaryDeviceId().is_gpu() &&
                    active_sampling_params_.dry_multiplier != 0.0f &&
                    active_sampling_params_.dry_penalty_last_n != 0)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome GPU greedy MTP requires a "
                        "device-owned DRY history implementation");
                }
                if (!runner_->supportsMTPDeviceDraftTokenInput())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy GPU MTP requires device-resident verifier token input");
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
                    auto fail_active_greedy_verifier =
                        [&](std::string message) -> GenerationResult
                    {
                        runner_->setComputeAllPositionLogits(false);
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        verifier_plan_scope.close();
                        /*
                         * Rollback is a logical-state writer. The verifier
                         * stream reader must publish and retire its edge before
                         * the shared failure helper requests writer admission.
                         */
                        verifier_sync_deferral.close();
                        return fail_after_checkpoint(message);
                    };
                    if (!verifier_plan_scope.installed())
                    {
                        return fail_active_greedy_verifier(
                            "Grouped-outcome greedy MTP verifier could not install row metadata plan");
                    }
                    if (!runner_->setComputeRowIndexedAllPositionLogits(
                            true,
                            verifier_row_count))
                    {
                        return fail_active_greedy_verifier(
                            "Grouped-outcome greedy MTP verifier could not enable row-indexed logits");
                    }
                    if (!runner_->setComputeAllPositionLogits(true))
                    {
                        return fail_active_greedy_verifier(
                            "Grouped-outcome greedy MTP verifier could not enable all-position logits");
                    }
                    std::string preparation_error;
                    verifier_input_tokens_device =
                        prepare_grouped_gpu_verifier_input_tokens(
                            verifier_input_plan,
                            "grouped_greedy_verifier",
                            &preparation_error);
                    if (!verifier_input_tokens_device)
                    {
                        return fail_active_greedy_verifier(
                            preparation_error);
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

                    MTPVerifierForwardExecutionOptions verifier_forward_options;
                    verifier_forward_options.device_token_ids =
                        verifier_input_tokens_device;
                    if (runner_->primaryDeviceId().is_gpu() &&
                        !runner_
                             ->prepareGreedyAllPositionBatchOutcomeGraph(
                                 verifier_row_count,
                                 stop_tokens_.data(),
                                 static_cast<int>(
                                     stop_tokens_.size()),
                                 MTPRequestPenaltyPolicy{
                                     .presence_penalty =
                                         active_sampling_params_
                                             .presence_penalty,
                                     .frequency_penalty =
                                         active_sampling_params_
                                             .frequency_penalty,
                                 }))
                    {
                        return fail_active_greedy_verifier(
                            "Grouped-outcome greedy MTP could not arm the "
                            "graph-owned outcome transaction");
                    }
                    const MTPVerifierForwardExecutionResult verifier_forward =
                        executeMTPSpecVerifierForward(
                            *runner_,
                            verifier_input_plan,
                            verifier_forward_options);
                    if (!verifier_forward.ok)
                    {
                        return fail_active_greedy_verifier(
                            std::string("Grouped-outcome greedy MTP verifier forward failed: ") +
                            verifier_forward.error);
                    }
                    /*
                     * Keep row-indexed all-position logits alive until the
                     * compact outcome reducer has sampled every verifier row.
                     * Rank-level LocalTP consumes child LogitsLocalInfo after
                     * this forward, and tearing the views down here can make a
                     * rejected row sample from cleared or rebound buffers.
                     */
                }

                const int compare_rows =
                    static_cast<int>(draft_tokens.size()) - 1;

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
                        runner_->setComputeAllPositionLogits(false);
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        verifier_sync_deferral.close();
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP resident device outcome verifier failed");
                    }
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_greedy_verifier_rows_sampled_before_all_position_cleanup",
                    static_cast<double>(compare_rows + 1),
                    "decode",
                    {},
                    {{"policy_path",
                      "grouped_outcome_device_resident_publication"},
                     {"sampling", "greedy"}});
                if (!runner_->setComputeAllPositionLogits(false))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    verifier_sync_deferral.close();
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP verifier could not disable all-position logits");
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(false, 0))
                {
                    verifier_sync_deferral.close();
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP verifier could not disable row-indexed logits");
                }
                verifier_sync_deferral.close();

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

                DeviceSpeculativeVerifyBatchOutcome device_outcome;
                MTPDecodeCatchupGreedyResult catchup;
                int shifted_publication_commit_count = 0;
                if (plan_.usesLocalTP())
                {
                    const bool first_shifted_row_available_from_sidecar =
                        sidecar_preserves_main_state &&
                        runner_->supportsMTPShiftedRowReuseFromSidecar() &&
                        !first_token_is_stop;
                    const int max_state_commit_rows =
                        static_cast<int>(draft_tokens.size());
                    bool first_shifted_row_available_for_publication =
                        first_shifted_row_available_from_sidecar;
                    if (!first_shifted_row_available_from_sidecar &&
                        !first_token_is_stop &&
                        max_state_commit_rows > 0)
                    {
                        bool initial_shifted_commit_ok = false;
                        {
                            PerfStatsCollector::ScopedTimer timer(
                                "mtp",
                                "grouped_outcome_greedy_initial_shifted_device_commit",
                                "decode");
                            initial_shifted_commit_ok =
                                runner_->commitMTPInitialShiftedRowFromDeviceOutcome(
                                    verifier_base_checkpoint,
                                    outcome_handle,
                                    /*request_index=*/0,
                                    /*main_forward_token_count=*/max_state_commit_rows,
                                    /*allow_speculative_discard=*/true);
                        }
                        if (!initial_shifted_commit_ok)
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome greedy MTP device-resident initial shifted-cache commit failed");
                        }
                        first_shifted_row_available_for_publication = true;
                        shifted_publication_commit_count += 1;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_greedy_initial_shifted_device_commits",
                            1.0,
                            "decode",
                            {},
                            {{"source", "device_outcome_checkpoint_terminal_hidden"}});
                    }
                    if (first_shifted_row_available_for_publication &&
                        max_state_commit_rows > 1)
                    {
                        bool shifted_catchup_ok = false;
                        {
                            PerfStatsCollector::ScopedTimer timer(
                                "mtp",
                                "grouped_outcome_greedy_shifted_prefix_device_commit",
                                "decode");
                            shifted_catchup_ok =
                                runner_->commitMTPShiftedRowsFromDeviceOutcome(
                                    outcome_handle,
                                    /*request_index=*/0,
                                    /*already_appended_tokens=*/1,
                                    max_state_commit_rows,
                                    /*main_forward_token_count=*/max_state_commit_rows,
                                    /*allow_speculative_discard=*/true);
                        }
                        if (!shifted_catchup_ok)
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome greedy MTP device-resident shifted-cache suffix commit failed");
                        }
                        shifted_publication_commit_count +=
                            max_state_commit_rows - 1;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_greedy_shifted_prefix_device_commits",
                            static_cast<double>(max_state_commit_rows - 1),
                            "decode",
                            {},
                            {{"rows", std::to_string(max_state_commit_rows - 1)}});
                    }
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = outcome_handle;
                publication_request.max_state_commit_rows =
                    visible_state_commit_plan.max_state_commit_rows;
                publication_request.publish_mtp_shifted_kv = true;
                publication_request.penalty_policy =
                    MTPRequestPenaltyPolicy{
                        .presence_penalty =
                            active_sampling_params_.presence_penalty,
                        .frequency_penalty =
                            active_sampling_params_.frequency_penalty,
                    };

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer direct_publish_timer(
                        "mtp",
                        "grouped_outcome_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"request_count", "1"},
                         {"logical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .logicalVerifierRowsPerRequest())},
                         {"physical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .physicalVerifierRowsPerRequest())},
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
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())},
                     {"shifted_commits",
                      std::to_string(shifted_publication_commit_count)},
                     {"sampling", "greedy"}});

                /*
                 * Greedy and stochastic grouped publication share the same
                 * controller-owned terminal contract. The reducer has already
                 * committed the first transaction; the selected CUDA/HIP
                 * execution policy owns every remaining graph transaction and
                 * surfaces only the terminal response ledger.
                 */
                if (!publication_request.outcome
                         .device_generation_controller_owned)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP lost device-generation controller ownership after admission");
                }
                if (mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic &&
                    !materialize_dynamic_generation_loop_this_step)
                {
                    return fail_after_checkpoint(
                        "Dynamic grouped-outcome greedy MTP did not begin at its admitted prefill boundary");
                }

                GenerationResult resident_result =
                    completeDeviceResidentGeneration(
                        publication_request,
                        DeviceGenerationSamplingMode::Greedy,
                        transaction_base_cached_tokens,
                        requested_speculative_draft_count,
                        speculative_draft_count,
                        std::move(result));
                if (!resident_result.success())
                    return fail_after_checkpoint(resident_result.error);
                return resident_result;
            }

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
        const bool mpi_coordinated_world =
            mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->world_size() > 1;
        const bool mpi_root_command =
            mpi_coordinated_world &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_;
        ScopedMPIRootCommandFailureAbort decode_command_failure_abort(
            mpi_ctx_.get(),
            mpi_root_command,
            "DECODE_STEP",
            &result.error);
        ScopedMoEOverlayRootCommand overlay_decode_command(
            moe_overlay_inference_transaction_coordinator_);

        // Broadcast to worker ranks so they run decode in lockstep.  The
        // current token budget is part of the decode command: Qwen thinking
        // budget handling calls setDecodeStepTokenBudget(1) only on the root,
        // and MTP draft-depth clamping must be identical on every rank or TP
        // collectives diverge inside the sidecar/verifier graphs.
        if (mpi_root_command)
        {
            broadcastCommand(MPICommand::DECODE_STEP);
            int32_t token_budget = static_cast<int32_t>(decode_step_token_budget_);
            mpi_ctx_->broadcast_int32(
                &token_budget, 1, mpi_coordinated_root_rank_);
            // From this point a worker may enter model collectives before root
            // reaches its sampler.  Every later error must abort the command
            // world instead of returning to a caller that cannot release it.
            decode_command_failure_abort.markPublished();
        }

        if (overlay_decode_command.active())
        {
            if (!mpi_root_command ||
                moe_overlay_collective_generation_id_ == 0 ||
                !moe_expert_overlay_residency_authority_)
            {
                result.error =
                    "ExpertOverlay decode authority has incomplete root, request-generation, or residency ownership";
                return result;
            }
            const auto snapshot =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!snapshot || !snapshot->valid() ||
                moe_overlay_inference_command_sequence_ ==
                    std::numeric_limits<std::uint64_t>::max())
            {
                result.error =
                    "ExpertOverlay decode authority has no valid placement epoch or command identity";
                return result;
            }
            const MoEOverlayInferenceCommandIdentity command_identity{
                .request_generation =
                    moe_overlay_collective_generation_id_,
                .command_id =
                    ++moe_overlay_inference_command_sequence_,
                .initial_placement_epoch = snapshot->epoch,
            };
            std::string command_error;
            if (!overlay_decode_command.begin(
                    command_identity, &command_error))
            {
                result.error =
                    "ExpertOverlay remote decode command admission failed: " +
                    command_error;
                return result;
            }
        }

        const bool mpi_worker_rank =
            mpi_coordinated_world &&
            mpi_ctx_->rank() != mpi_coordinated_root_rank_;
        auto trace_position = [this](const char *context) -> int
        {
            return currentDecodeTransactionPositionForPlanning(context, nullptr)
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
                const int selected_draft_depth =
                    currentMTPDraftDepth(mtp);
                if (selected_draft_depth > 0)
                {
                    std::string sequence_error;
                    if (!overlay_decode_command.beginGraphSequence(
                            selected_draft_depth, &sequence_error))
                    {
                        result.error =
                            "ExpertOverlay MTP graph-sequence admission failed: " +
                            sequence_error;
                        return result;
                    }
                    const std::uint64_t verifier_runs_before =
                        mtp_stats_.verifier_runs;
                    MoEOverlayInferenceInterferenceScope interference_scope(
                        moe_expert_overlay_interference_probe_.get(),
                        inferenceWorkloadIdentity(
                            ExpertHistogramSource::GroupedVerifier,
                            selected_draft_depth + 1,
                            effectiveMTPMaxDraftDepth(mtp) + 1,
                            1,
                            selected_draft_depth));
                    GenerationResult mtp_result = decodeStepMTP();
                    if (mtp_stats_.verifier_runs == verifier_runs_before)
                        interference_scope.discard();
                    if (!mtp_result.success())
                    {
                        /*
                         * The outer command guards abort the live distributed
                         * transaction while this stack unwinds. Emit the root
                         * cause before that terminal publication so MPI_Abort
                         * cannot erase the only actionable diagnostic.
                         */
                        LOG_ERROR(
                            "[OrchestrationRunner] MTP decode failed after "
                            "ExpertOverlay command admission: "
                            << (mtp_result.error.empty()
                                    ? "no failure detail was recorded"
                                    : mtp_result.error));
                    }
                    if (mtp_result.success())
                    {
                        std::string command_error;
                        if (!overlay_decode_command.complete(&command_error))
                        {
                            mtp_result.error =
                                "ExpertOverlay MTP command completion failed: " +
                                command_error;
                            result = std::move(mtp_result);
                            return result;
                        }
                        decode_command_failure_abort.markCompleted();
                    }
                    result = std::move(mtp_result);
                    return result;
                }
                if (ready_mtp_condition_.has_value() &&
                    ready_mtp_condition_->isDeviceResidentOnly())
                {
                    result.error =
                        "Device-resident MTP continuation cannot cross into a depth-zero serial decode transaction";
                    return result;
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
            if (ready_mtp_condition_.has_value())
            {
                if (!ready_mtp_condition_->valid())
                {
                    result.error =
                        "Ready MTP condition has an invalid authority payload";
                    return result;
                }
                if (!samplingParamsEqual(
                        ready_mtp_condition_->sampling_params,
                        active_sampling_params_))
                {
                    result.error =
                        "Ready MTP condition was sampled with different sampling parameters";
                    return result;
                }
                if (ready_mtp_condition_->isDeviceResidentOnly())
                {
                    result.error =
                        "Device-resident MTP continuation reached the serial decode sampler";
                    return result;
                }
                ready_token_for_decode =
                    ready_mtp_condition_->host_token;
                ready_mtp_condition_.reset();
            }
            prefill_logits_ready_ = false;
            LOG_TRACE("[decodeStep] Using prefill logits (skipping forward)");
        }
        else
        {
            if (ready_mtp_condition_.has_value())
            {
                result.error =
                    "Ready MTP condition exists without a terminal-logits boundary";
                return result;
            }
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
            std::optional<MoEOverlayInferenceInterferenceScope>
                interference_scope;
            interference_scope.emplace(
                moe_expert_overlay_interference_probe_.get(),
                inferenceWorkloadIdentity(
                    ExpertHistogramSource::DecodeToken,
                    1,
                    1,
                    1,
                    0));
            // Run single-token forward with last token.
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << (mpi_ctx_ ? mpi_ctx_->rank() : 0)
                         << " forward begin token=" << last_token_);
            }
            std::string sequence_error;
            if (!overlay_decode_command.beginGraphSequence(
                    /*draft_depth=*/0, &sequence_error))
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                result.error =
                    "ExpertOverlay serial graph-sequence admission failed: " +
                    sequence_error;
                return result;
            }
            if (!runner_->forward(&last_token_, 1))
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                result.error = "Forward pass failed during decode";
                return result;
            }
            if (!advanceDecodeTransactionPlanningPositionAfterForward(
                    "decode_condition_forward"))
            {
                result.error = last_error_;
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

        // Tail stage: keep GPU logits on the device sampling path.  CPU-only
        // runners may sample from host logits below, but a GPU sampler failure
        // is a hard error rather than a hidden D2H host sampling path.
        int token = -1;
        bool device_sampling_attempted = false;
        bool device_penalty_application_failed = false;
        auto current_stochastic_sample_logical_position = [&]() -> std::optional<int>
        {
            std::string position_error;
            const std::optional<int> mtp_position =
                currentDecodeTransactionPositionForPlanning(
                    "stochastic main-logits sample",
                    &position_error);
            if (mtp_position.has_value())
                return mtp_position;

            LOG_DEBUG("[OrchestrationRunner/decodeStep] could not determine "
                      "logical stochastic sample position: "
                      << position_error);
            return std::nullopt;
        };

        auto sample_current_logits_on_device = [&]() -> int
        {
            device_sampling_attempted = true;
            if (active_sampling_params_.is_greedy())
                return runner_->sampleGreedyOnDevice();

            const std::optional<int> logical_position =
                current_stochastic_sample_logical_position();
            if (!logical_position.has_value())
                return -1;

            const float threshold = mtpSpecStochasticThresholdForPosition(
                active_sampling_params_,
                sampler_,
                *logical_position,
                MTPSpecStochasticDrawPurpose::Sample);
            const int sampled_token =
                runner_->sampleOnDeviceAtLogicalPosition(
                    active_sampling_params_,
                    *logical_position);
            if (sampled_token >= 0)
            {
                PerfStatsCollector::addCounter(
                    "sampling",
                    "main_stochastic_logical_position_samples",
                    1.0,
                    "decode",
                    {},
                    {{"logical_position", std::to_string(*logical_position)},
                     {"threshold", formatStochasticThreshold(threshold)},
                     {"logical_position_sampler", "true"}});
            }
            return sampled_token;
        };

        const bool mpi_sampling_collective_required =
            mpi_coordinated_world &&
            runner_->requiresMPICoordinatedDecodeSampling(active_sampling_params_);
        const bool worker_sampling_required =
            mpi_worker_rank && mpi_sampling_collective_required;
        /*
         * The coordinated root is always the token authority. `requiresMPICoordinatedDecodeSampling()`
         * only tells an auxiliary rank whether that rank must participate in a
         * sharded sampler collective before root publishes the committed token;
         * it never transfers sampling authority away from root. Treating a
         * false value as "nobody samples" left a GPU continuation rank with
         * token == -1 after a successful prefill and stranded its workers at
         * the post-sampling barrier.
         */
        const bool this_rank_must_sample =
            !mpi_worker_rank || !mpi_coordinated_world ||
            mpi_sampling_collective_required;
        if (mpi_worker_rank && !ready_token_for_decode.has_value())
        {
            /*
             * Worker ranks must participate in any device/distributed sampling
             * collectives, but the root owns the committed token. Do not run
             * root-only CPU sampling here; after the post-sampling fence below,
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
                    token = sample_current_logits_on_device();
                }
                else if (runner_->primaryDeviceId().is_gpu())
                {
                    device_penalty_application_failed = true;
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
                token = sample_current_logits_on_device();
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

            if (this_rank_must_sample)
            {
                if (!penalty_map.empty())
                {
                    // Try GPU-side penalty application + sampling.
                    bool gpu_penalties_applied = runner_->applyPenaltiesOnDevice(penalty_map, vocab);
                    if (gpu_penalties_applied)
                    {
                        token = sample_current_logits_on_device();
                    }
                    else if (runner_->primaryDeviceId().is_gpu())
                    {
                        device_penalty_application_failed = true;
                    }
                }
                else
                {
                    token = sample_current_logits_on_device();
                }
            }
        }
        else if (active_sampling_params_.is_greedy())
        {
            if (this_rank_must_sample)
            {
                token = sample_current_logits_on_device();
            }
        }
        else
        {
            if (this_rank_must_sample)
            {
                token = sample_current_logits_on_device();
                if (token >= 0)
                {
                    LOG_TRACE("[decodeStep] GPU top-k/top-p sampled token=" << token);
                }
            }
        }

        if (token < 0)
        {
            if (runner_->primaryDeviceId().is_gpu())
            {
                if (device_penalty_application_failed)
                {
                    result.error =
                        "GPU decode penalty application failed; host logits sampling is CPU-only";
                }
                else if (device_sampling_attempted)
                {
                    result.error =
                        "GPU decode sampling failed; host logits sampling is CPU-only";
                }
                else
                {
                    result.error =
                        "GPU decode sampling was required but not attempted; host logits sampling is CPU-only";
                }
                if (decode_sampling_sync_deferred)
                    result.error += " after deferred logits sync";
                return result;
            }

            // CPU-only host sampling.  GPU runners must not reach this path.
            const float *logits = runner_->logits();
            if (!logits)
            {
                result.error = "No logits available";
                return result;
            }
            int vocab = vocabSize();
            token = sampler_.sample(logits, static_cast<size_t>(vocab), active_sampling_params_);
        }

        if (mpi_coordinated_world &&
            !moe_overlay_inference_transaction_coordinator_ &&
            !moe_overlay_inference_transaction_follower_)
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
            mpi_ctx_->broadcast_int32(
                &coordinated_token, 1, mpi_coordinated_root_rank_);
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
                currentDecodeTransactionPositionForPlanning(
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

        std::string overlay_completion_error;
        if (!overlay_decode_command.complete(&overlay_completion_error))
        {
            result.error =
                "ExpertOverlay serial command completion failed: " +
                overlay_completion_error;
            return result;
        }
        decode_command_failure_abort.markCompleted();
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
        const bool coordinated_force_command =
            mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->world_size() > 1;
        const bool mpi_root_command =
            coordinated_force_command &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_;
        ScopedMPICoordinatedCommandFence force_command_fence(
            mpi_ctx_.get(),
            coordinated_force_command,
            "forceDecodeToken");
        ScopedMPIRootCommandFailureAbort force_command_failure_abort(
            mpi_ctx_.get(),
            mpi_root_command,
            "FORCE_DECODE_TOKEN",
            &result.error);
        ScopedMoEOverlayRootCommand overlay_force_command(
            moe_overlay_inference_transaction_coordinator_);

        auto trace_position = [this](const char *context) -> int
        {
            return currentDecodeTransactionPositionForPlanning(context, nullptr)
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

        if (mpi_root_command)
        {
            if (traceChatGeneratedTokensEnabled())
                LOG_INFO("[OrchestrationRunner/forceDecodeToken] broadcasting forced token");
            broadcastCommand(MPICommand::FORCE_DECODE_TOKEN);
            int32_t forced = token;
            mpi_ctx_->broadcast_int32(
                &forced, 1, mpi_coordinated_root_rank_);
            force_command_failure_abort.markPublished();
        }

        if (overlay_force_command.active())
        {
            if (!mpi_root_command ||
                moe_overlay_collective_generation_id_ == 0 ||
                !moe_expert_overlay_residency_authority_)
            {
                result.error =
                    "ExpertOverlay forced-token authority has incomplete root, request-generation, or residency ownership";
                return result;
            }
            const auto snapshot =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!snapshot || !snapshot->valid() ||
                moe_overlay_inference_command_sequence_ ==
                    std::numeric_limits<std::uint64_t>::max())
            {
                result.error =
                    "ExpertOverlay forced-token authority has no valid placement epoch or command identity";
                return result;
            }
            const MoEOverlayInferenceCommandIdentity command_identity{
                .request_generation =
                    moe_overlay_collective_generation_id_,
                .command_id =
                    ++moe_overlay_inference_command_sequence_,
                .initial_placement_epoch = snapshot->epoch,
            };
            std::string command_error;
            if (!overlay_force_command.begin(
                    command_identity, &command_error))
            {
                result.error =
                    "ExpertOverlay forced-token command admission failed: " +
                    command_error;
                return result;
            }
        }

        const auto begin_serial_overlay_graph = [&]() -> bool
        {
            std::string sequence_error;
            if (!overlay_force_command.beginGraphSequence(
                    /*draft_depth=*/0, &sequence_error))
            {
                result.error =
                    "ExpertOverlay forced-token graph-sequence admission failed: " +
                    sequence_error;
                return false;
            }
            return true;
        };

        const bool token_is_stop =
            std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
        const bool gpu_mtp_force =
            shouldUseMTPDecode() &&
            runner_->primaryDeviceId().is_gpu();

        if (gpu_mtp_force && token_is_stop)
        {
            /*
             * A terminal control token ends generation and therefore never
             * becomes a model-state row. Keep this as an explicit GPU-MTP
             * lifecycle state instead of allowing it to enter the CPU/non-MTP
             * branch below. The ready logits and their sampled shadows belong
             * to the completed request and are retired without publishing,
             * committing, or forwarding the terminal token on any device.
             */
            prefill_logits_ready_ = false;
            ready_mtp_condition_.reset();
            PerfStatsCollector::addCounter(
                "mtp",
                "forced_stop_token_state_mutations_skipped",
                1.0,
                "decode",
                {},
                {{"state_transition", "terminal_control_only"},
                 {"device_execution", "none"}});
        }
        else if (gpu_mtp_force)
        {
            /*
             * A forced policy token replaces the sample selected from an already
             * materialized terminal-logits boundary. Requiring that boundary
             * makes the state transition unambiguous: the old ready token has
             * never entered main KV/GDN state, and the forced token can become
             * the sole owner of the next row. Every forced token is forwarded
             * immediately below, which recreates the same boundary for the next
             * token in a multi-token stop-thinking sequence.
             */
            if (!prefill_logits_ready_)
            {
                result.error =
                    "GPU MTP forced decode requires a ready terminal-logits boundary";
                return result;
            }

            prefill_logits_ready_ = false;
            ready_mtp_condition_.reset();

            /*
             * Request policy selected `token` on the host, but execution state
             * becomes device-owned here. The backend publishes the scalar with a
             * one-thread kernel into the persistent target slot and records the
             * exact producer stream. No stack-backed H2D copy, allocation,
             * default stream, or synchronization is permitted.
             */
            if (!runner_->stageStochasticTargetTokenForDeviceSampling(
                    token,
                    /*target_sample_slot=*/0))
            {
                result.error =
                    "GPU MTP forced decode could not publish its control token to device state";
                return result;
            }

            /*
             * Shifted MTP KV consumes the device token first, while terminal
             * hidden still names the state immediately before that token. The
             * commit's transaction lease and target-slot readiness event order
             * this mutation without observing a host position mirror.
             */
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "forced_token_shifted_commit_device",
                    "decode");
                if (!runner_->commitMTPShiftedRowFromDeviceTargetSample(
                        /*target_sample_slot=*/0,
                        /*already_appended_tokens=*/0,
                        /*allow_speculative_discard=*/true))
                {
                    result.error =
                        "GPU MTP forced-token shifted-cache maintenance failed";
                    return result;
                }
            }

            runner_->setMTPMainDecodeSyncDeferralEnabled(true);
            if (!begin_serial_overlay_graph())
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                return result;
            }
            if (!runner_->advanceMTPMainConditionFromDeviceTargetSample(
                    token,
                    /*target_sample_slot=*/0))
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                result.error =
                    "GPU MTP forced-token main-state advance failed";
                return result;
            }
            if (!advanceDecodeTransactionPlanningPositionAfterForward(
                    "forced_decode_device_condition_forward"))
            {
                result.error = last_error_;
                return result;
            }

            prefill_logits_ready_ = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "forced_token_device_owned_transactions",
                1.0,
                "decode",
                {},
                {{"publication", "control_scalar_kernel"},
                 {"shifted_state", "device_target_slot"},
                 {"main_forward", "device_token_ids"}});
        }
        else
        {
            /*
             * CPU and non-MTP execution retain their serial token contract. A
             * forced token chosen at request level replaces ready logits when
             * present; otherwise the previous token must first advance ordinary
             * serial decode state.
             */
            if (prefill_logits_ready_)
            {
                prefill_logits_ready_ = false;
                ready_mtp_condition_.reset();
            }
            else if (!token_is_stop)
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[OrchestrationRunner/forceDecodeToken] rank="
                             << rank << " forwarding previous token " << last_token_);
                }
                if (!begin_serial_overlay_graph())
                    return result;
                if (!runner_->forward(&last_token_, 1))
                {
                    result.error =
                        "Forward pass failed while committing forced token";
                    return result;
                }
                if (!advanceDecodeTransactionPlanningPositionAfterForward(
                        "forced_decode_condition_forward"))
                {
                    result.error = last_error_;
                    return result;
                }
            }

            /*
             * CPU MTP maintains the same shifted sidecar stream using its
             * serial-row implementation. GPU MTP never enters this host-token
             * API: its complete device-owned transaction is handled above.
             */
            if (shouldUseMTPDecode() && !token_is_stop)
            {
                std::string position_error;
                const std::optional<int> base_sidecar_position =
                    currentDecodeTransactionPositionForPlanning(
                        "forced decode token",
                        &position_error);
                if (!base_sidecar_position)
                {
                    result.error = position_error;
                    return result;
                }

                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "forced_token_shifted_commit",
                    "decode");
                if (!runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                        token,
                        /*already_appended_tokens=*/0,
                        /*allow_speculative_discard=*/true,
                        *base_sidecar_position))
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

        std::string overlay_completion_error;
        if (!overlay_force_command.complete(&overlay_completion_error))
        {
            result.error =
                "ExpertOverlay forced-token command completion failed: " +
                overlay_completion_error;
            return result;
        }
        force_command_failure_abort.markCompleted();
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

        /*
         * Sampling policy is request admission state. Install it before prefill
         * so GPU request-input or prefix-restore publication can initialize the
         * graph-stable penalty policy on its exact reset-ordered stream.
         */
        active_sampling_params_ = sampling;
        sampler_ = Sampler(sampling.seed);

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

    uint64_t OrchestrationRunner::moeRuntimeMovementEpoch() const
    {
        return runner_ ? runner_->moeRuntimeMovementEpoch() : 0;
    }

    bool OrchestrationRunner::maybeApplyMoERebalance()
    {
        const std::string device =
            runner_ && runner_->primaryDeviceId().is_valid()
                ? runner_->primaryDeviceId().toString()
                : std::string{};

        if (device_generation_embedded_moe_maintenance_pending_ack_)
        {
            if (!runner_ ||
                runner_->moeOverlayAuthorityExecution() !=
                    MoEOverlayAuthorityExecutionKind::
                        HomogeneousDeviceResident ||
                !runner_
                     ->deviceResidentMoEOverlayMaintenanceReady())
            {
                return setError(
                    "Embedded ExpertOverlay maintenance acknowledgement lost its device-resident authority binding");
            }
            device_generation_embedded_moe_maintenance_pending_ack_ = false;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "device_generation_embedded_maintenance_acknowledgements",
                1.0,
                "maintenance",
                device,
                {{"authority_execution", "homogeneous_device_resident"},
                 {"outer_host_launch", "false"},
                 {"replayed", "false"}});
            return true;
        }

        if (usesHomogeneousGpuDeviceResidentMoEOverlayAuthority())
        {
            if (config_.moe_rebalance.mode !=
                MoERebalanceRuntimeMode::Dynamic)
            {
                return true;
            }
            if (!runner_ ||
                runner_->moeOverlayAuthorityExecution() !=
                    MoEOverlayAuthorityExecutionKind::
                        HomogeneousDeviceResident)
            {
                return setError(
                    "Homogeneous device-resident ExpertOverlay lost its graph-family authority binding");
            }
            try
            {
                if (!runner_->maybeApplyDecodeBoundaryMaintenance())
                {
                    return setError(
                        "Homogeneous device-resident ExpertOverlay maintenance enqueue failed");
                }
            }
            catch (const std::exception &error)
            {
                return setError(
                    std::string(
                        "Homogeneous device-resident ExpertOverlay maintenance failed: ") +
                    error.what());
            }

            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "decode_boundary_background_notifications",
                1.0,
                "maintenance",
                device,
                {{"blocking", "false"},
                 {"authority_execution",
                  "homogeneous_device_resident"},
                 {"scheduler", "captured_device_stream"}});
            return true;
        }

        if (moe_expert_overlay_maintenance_service_)
        {
            if (!moe_expert_overlay_maintenance_service_->healthy())
            {
                return setError(
                    "ExpertOverlay background residency maintenance failed: " +
                    moe_expert_overlay_maintenance_service_
                        ->failureMessage());
            }
            /* Wake-only: inference never polls, joins, or waits for movement. */
            moe_expert_overlay_maintenance_service_
                ->notifyMaintenanceProgress();
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "decode_boundary_background_notifications",
                1.0,
                "maintenance",
                device,
                {{"blocking", "false"},
                 {"authority_execution", "host_coordinated"},
                 {"scheduler", "host_background_service"}});
            return true;
        }

        /* Static and observe-only authorities intentionally own no maintenance
         * worker. Their setup-time immobility evidence is sufficient and a
         * committed inference boundary has no placement work to perform. */
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
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
            broadcastCommand(MPICommand::CLEAR_CACHE);

        resetUnderlyingRunnerRequestState("request-clear-cache");
        if (!advanceMoEOverlayCollectiveRequestGeneration("request-clear-cache"))
        {
            throw std::runtime_error(
                "clearCache() could not publish a fresh graph-native MoE "
                "overlay collective generation");
        }
#if defined(__GLIBC__)
        ::malloc_trim(0);
#endif
        prefill_logits_ready_ = false;
        ready_mtp_condition_.reset();
        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        decode_transaction_planning_position_.reset();
        device_generation_admission_.reset();
        device_generation_terminal_ledger_authoritative_ = false;
        device_generation_embedded_moe_maintenance_pending_ack_ = false;
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

    void OrchestrationRunner::drainCompletedDecodeBoundaryMaintenanceDiagnostics()
    {
        if (runner_)
            runner_->drainCompletedDecodeBoundaryMaintenanceDiagnostics();
    }

    PrefixRuntimeStateSnapshot OrchestrationRunner::prefixStateProbe() const
    {
        PrefixRuntimeStateSnapshot snapshot = runner_ ? runner_->prefixStateProbe()
                                                      : PrefixRuntimeStateSnapshot{};
        snapshot.initialized = initialized_;
        snapshot.prefill_logits_ready = prefill_logits_ready_;
        /*
         * The child probe owns the meaning of live position. GPU runners read
         * it from the device-resident logical-state mailbox, while
         * get_position() is only a host planning mirror and is intentionally
         * allowed to lag accepted grouped publication. Overwriting the child
         * result here made a preserved sidecar appear to move main state
         * backwards and, more seriously, taught diagnostics to trust stale
         * host state over the device owner.
         */
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
        snapshot.mtp_last_transaction_draft_depth =
            mtp_stats_.last_transaction_draft_depth;
        snapshot.mtp_last_transaction_emitted_token_count =
            mtp_stats_.last_transaction_emitted_token_count;
        if (runner_)
        {
            /*
             * Keep the same ownership split used by transaction planning.
             * The GPU scalar is scheduler metadata advanced from validated
             * compact publication; it is not a shadow of device KV, GDN, or
             * token state. CPU execution keeps its live position on host.
             */
            snapshot.mtp_next_condition_position =
                runner_->primaryDeviceId().is_gpu()
                    ? decode_transaction_planning_position_.value_or(-1)
                    : runner_->get_position();
        }
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
            device_generation_terminal_ledger_authoritative_
                ? mtp_stats_.current_depth
                : (mtp_depth_controller_
                       ? mtp_depth_controller_->currentDepth()
                       : std::max(0, mtp.draft_tokens));
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
        if (device_generation_terminal_ledger_authoritative_)
        {
            /*
             * Native generation owns every adaptive transition after admission.
             * The terminal ledger currently exports aggregate transition counts,
             * not a host-readable per-window decision enum.  Reporting the
             * dormant host controller's reason beside the device-selected depth
             * would manufacture a mixed-authority diagnostic.
             */
            snapshot.mtp_request.last_depth_policy_reason =
                "device_terminal_ledger";
        }
        else if (mtp_depth_controller_)
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

    DeviceId OrchestrationRunner::primaryDeviceId() const
    {
        return runner_ ? runner_->primaryDeviceId() : DeviceId::cpu();
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
        PerfStatsCollector::addCounter(
            "sampling",
            "host_logits_access",
            1.0,
            {},
            runner_->primaryDeviceId().toString(),
            {{"source", "orchestration_runner_last_logits"}});
        return runner_->logits();
    }

    void OrchestrationRunner::setStopTokens(const std::vector<int32_t> &stop_tokens)
    {
        if (stop_tokens.size() >
            static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        {
            throw std::invalid_argument(
                "Request stop-token policy exceeds the MPI protocol's int32 count range");
        }

        /*
         * Stop recognition changes whether an MTP transaction launches another
         * sidecar. Install the policy locally before publishing it: a rejected
         * root policy must never release workers into a request with a different
         * termination contract. Once accepted, the command and its two payloads
         * form one ordered worker-loop transaction ahead of PREFILL/DECODE_STEP.
         */
        if (runner_ &&
            !runner_->configureMTPRequestStopTokens(stop_tokens))
        {
            throw std::runtime_error(
                "Inference runner rejected request stop-token policy");
        }
        stop_tokens_ = stop_tokens;

        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SET_STOP_TOKENS);
            int32_t stop_token_count = static_cast<int32_t>(stop_tokens_.size());
            mpi_ctx_->broadcast_int32(
                &stop_token_count, 1, mpi_coordinated_root_rank_);
            if (stop_token_count > 0)
            {
                mpi_ctx_->broadcast_int32(
                    stop_tokens_.data(),
                    static_cast<size_t>(stop_token_count),
                    mpi_coordinated_root_rank_);
            }
        }
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
                                    requiresOverlayMPIWorld(config_.moe_routed_expert_plan);

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

        auto bindExpertOverlayAuthorityPlan =
            [this](MoEExpertOverlayPlanInstallOrigin install_origin) -> bool
        {
            if (!config_.moe_routed_expert_plan ||
                !config_.moe_routed_expert_plan
                     ->usesExpertOverlayAuthority())
            {
                return true;
            }
            try
            {
                auto resolved = bindMoEExpertOverlayPlanToClusterInventory(
                    *config_.moe_routed_expert_plan,
                    cluster_inventory_);
                const auto install_errors =
                    installResolvedMoEExpertOverlayPlan(
                        config_,
                        std::move(resolved),
                        install_origin);
                if (!install_errors.empty())
                {
                    std::ostringstream error;
                    error << "Failed to install hardware-resolved MoE overlay topology:";
                    for (const auto &entry : install_errors)
                        error << "\n  - " << entry;
                    return setError(error.str());
                }

                mpi_coordinated_root_rank_ =
                    coordinatedRootRankForRunner(
                        config_.moe_routed_expert_plan,
                        mpi_ctx_);
                if (!mpi_ctx_ || mpi_coordinated_root_rank_ < 0 ||
                    mpi_coordinated_root_rank_ >= mpi_ctx_->world_size())
                {
                    return setError(
                        "Hardware-resolved MoE overlay produced an invalid "
                        "coordinated continuation root rank");
                }
            }
            catch (const std::exception &e)
            {
                return setError(
                    std::string("Failed to bind MoE overlay participants to discovered hardware: ") +
                    e.what());
            }
            return true;
        };

        if (!bindExpertOverlayAuthorityPlan(
                MoEExpertOverlayPlanInstallOrigin::UserDeclaredDomains))
            return false;

        // Get model config (need to load model metadata first)
        // Read actual model metadata from the GGUF file for accurate plan building
        ModelConfig model_config;
        bool model_has_routed_experts = false;
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
            const std::string architecture = metadata_loader.architecture();
            model_has_routed_experts =
                metadata_loader.getInt(
                    architecture + ".expert_count",
                    metadata_loader.getInt("expert_count", 0)) > 0;
            LOG_DEBUG("Model metadata for plan building: n_layers=" << model_config.n_layers
                                                                    << " n_heads=" << model_config.n_heads
                                                                    << " n_kv_heads=" << model_config.n_kv_heads
                                                                    << " hidden_size=" << model_config.hidden_size
                                                                    << " routed_moe=" << (model_has_routed_experts ? "true" : "false"));
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

        // Build a preliminary plan before implicit MoE authority normalization.
        // Its resolved LocalTP participants are the topology facts consumed by
        // the one-tier plan; the normalizer never repeats hardware discovery.
        int my_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        RankExecutionPlan preliminary_plan =
            plan_builder_->buildPlanForRank(
                config_, model_config, cluster_inventory_, my_rank);

        MoEExpertOverlayAuthorityPlanResult authority_plan;
        try
        {
            authority_plan = normalizeMoEExpertOverlayAuthorityPlan({
                .requested_plan = config_.moe_routed_expert_plan,
                .model_has_routed_experts = model_has_routed_experts,
                .world_rank = my_rank,
                .local_tp_participants =
                    preliminary_plan.local_tp_devices,
                .local_tp_weights =
                    preliminary_plan.local_tp_weights,
                .local_tp_backend =
                    preliminary_plan.local_tp_backend,
                .has_cross_rank_tensor_parallel =
                    preliminary_plan.usesGlobalTP(),
                .has_pipeline_parallel =
                    preliminary_plan.usesLocalPP() ||
                    preliminary_plan.usesPipelineParallel(),
                .routed_compute_policy =
                    config_.routed_expert_compute_policy,
                .owner_order =
                    config_.routed_expert_owner_order,
                .residency_maintenance =
                    config_.moe_rebalance.mode,
            });
        }
        catch (const std::exception &error)
        {
            return setError(
                std::string(
                    "Failed to normalize the universal multi-device MoE authority: ") +
                error.what());
        }

        if (authority_plan.synthesized())
        {
            config_.moe_routed_expert_plan = authority_plan.plan;
            if (!bindExpertOverlayAuthorityPlan(
                    MoEExpertOverlayPlanInstallOrigin::
                        SynthesizedSimpleTP))
                return false;

            /*
             * Hardware binding installs the implicit domain into the canonical
             * named-domain inventory. Revalidate and rebuild so rank planning,
             * weight preparation, and graph lowering all consume that exact
             * same topology instead of the preliminary discovery view.
             */
            errors = plan_builder_->validateConfig(
                config_, model_config, cluster_inventory_);
            if (!errors.empty())
            {
                std::string error_msg =
                    "Implicit ExpertOverlay configuration validation failed:";
                for (const auto &entry : errors)
                    error_msg += "\n  - " + entry;
                return setError(error_msg);
            }
            preliminary_plan = plan_builder_->buildPlanForRank(
                config_, model_config, cluster_inventory_, my_rank);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "implicit_single_tier_authority_plan",
                1.0,
                "model_setup",
                "priority_0",
                {{"participants",
                  std::to_string(
                      config_.moe_routed_expert_plan
                          ->domains.front().participants.size())},
                 {"world_rank", std::to_string(my_rank)},
                 {"topology", "single-domain"},
                 {"authority", "expert-overlay-rcu"}});
        }

        plan_ = std::move(preliminary_plan);

        auto overlay_execution_plan = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_);
        if (overlay_execution_plan)
        {
            std::string overlay_plan_error;
            if (!applyOverlayRankRoleToExecutionPlan(
                    plan_,
                    *config_.moe_routed_expert_plan,
                    *overlay_execution_plan,
                    overlay_plan_error))
            {
                return setError(overlay_plan_error);
            }

            if (overlay_execution_plan->buildsRootGraph())
            {
                LOG_DEBUG("[OrchestrationRunner] MoE overlay root plan bound to base domain '"
                          << config_.moe_routed_expert_plan->effectiveBaseModelDomain()
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

        /*
         * The overlay rank-role resolver has already reduced non-root ranks to
         * endpoint-only plans with no local_tp_devices. If devices remain here,
         * they are the continuation root's real dense LocalTP graph and must
         * receive the same collective context as any other production LocalTP
         * model. Overlay sparse-expert endpoints remain independent inside that
         * graph; they do not justify dropping its dense TP authority.
         *
         * LOCAL PP is different: each nested stage owns its own TP context.
         */
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

    bool OrchestrationRunner::loadWeights(bool prepopulate_page_cache_enabled)
    {
        // Get model path from config
        std::string model_path = config_.model_path;

        // Skip weight loading if no model path (for testing)
        if (model_path.empty())
        {
            if (model_ctx_)
            {
                return setError(
                    "A preloaded ModelContext requires an explicit model_path");
            }
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

        /*
         * A process-resident parity campaign may retain the immutable model and
         * PreparedWeightStore while rebuilding every runner-owned graph, arena,
         * stream, controller, and request state.  Prove that the retained
         * authority exactly matches this execution plan before consuming it.
         * A mismatch is fatal: silently reloading would make reuse evidence lie.
         */
        if (model_ctx_)
        {
            if (model_ctx_->path() != model_path)
            {
                return setError(
                    "Preloaded ModelContext path does not match model_path: " +
                    model_ctx_->path() + " != " + model_path);
            }

            const auto weight_manager = model_ctx_->concreteWeightManager();
            if (!weight_manager)
            {
                return setError(
                    "Preloaded ModelContext has no concrete WeightManager");
            }
            if (retained_prepared_weight_plan_)
            {
                if (config_.moe_routed_expert_plan)
                {
                    return setError(
                        "A single ModelContext reuse contract cannot certify routed "
                        "expert-placement authorities");
                }
                if (const auto mismatch = retainedPreparedWeightPlanMismatch(
                        *retained_prepared_weight_plan_, plan_))
                {
                    return setError(*mismatch);
                }
            }
            WeightDistributionStrategy expected_strategy =
                weight_config.strategy;
            if (weight_config.isLayerPartitioned() &&
                expected_strategy == WeightDistributionStrategy::REPLICATED)
            {
                expected_strategy =
                    WeightDistributionStrategy::LAYER_PARTITIONED;
            }
            if (weight_manager->strategy() != expected_strategy)
            {
                return setError(
                    "Preloaded ModelContext distribution strategy does not match "
                    "the execution plan");
            }
            if (weight_manager->hasEmbedding() != weight_config.has_embedding ||
                weight_manager->hasLMHead() != weight_config.has_lm_head)
            {
                return setError(
                    "Preloaded ModelContext global-weight ownership does not match "
                    "the execution plan");
            }
            if (expected_strategy ==
                WeightDistributionStrategy::LAYER_PARTITIONED)
            {
                if (!weight_manager->hasLayerRange())
                {
                    return setError(
                        "Preloaded ModelContext lacks the execution plan's layer range");
                }
                const auto [loaded_first, loaded_last_exclusive] =
                    weight_manager->layerRange();
                const int expected_last_exclusive =
                    weight_config.last_layer < 0
                        ? -1
                        : weight_config.last_layer + 1;
                if (loaded_first != weight_config.first_layer ||
                    loaded_last_exclusive != expected_last_exclusive)
                {
                    return setError(
                        "Preloaded ModelContext layer range does not match the "
                        "execution plan");
                }
            }

            /*
             * Only this successful comparison permits memory preflight to count
             * prepared weights as already resident. Store population and the
             * lifecycle completion gates are checked separately per device.
             */
            retained_prepared_weight_plan_validated_ =
                retained_prepared_weight_plan_.has_value();

            /*
             * A preloaded context can either be a freshly parsed model handed
             * to its first runner or a process-campaign authority whose packed
             * device weights already survived a prior runner.  Publish those
             * cases separately: setup-time economy tests must prove the latter
             * instead of inferring it from a caller-owned cache-hit boolean.
             * PreparedWeightStore is model-owned, so observing it here neither
             * creates a host mirror nor changes its lifetime.
             */
            const auto prepared_store =
                weight_manager->preparedWeightStoreIfInitialized();
            const size_t prepared_entry_count =
                prepared_store
                    ? prepared_store->sizeForDevice(
                          plan_.primary_device.toLocalDeviceId())
                    : 0u;
            PerfStatsCollector::addCounter(
                "weight_loading",
                "preloaded_model_context_uses",
                1.0,
                "load",
                plan_.primary_device.toString(),
                {{"prepared_entries",
                  std::to_string(prepared_entry_count)}});
            PerfStatsCollector::addCounter(
                "weight_loading",
                "preloaded_prepared_weight_store_reuses",
                prepared_entry_count > 0u ? 1.0 : 0.0,
                "load",
                plan_.primary_device.toString(),
                {{"prepared_entries",
                  std::to_string(prepared_entry_count)}});
            PerfStatsCollector::addCounter(
                "weight_loading",
                "preloaded_prepared_weight_entries",
                static_cast<double>(prepared_entry_count),
                "load",
                plan_.primary_device.toString());

            tokenizer_ = createTokenizer(model_ctx_);
            if (!tokenizer_)
            {
                LOG_WARN("Failed to create tokenizer from preloaded model context");
            }
            LOG_INFO("Reusing preloaded ModelContext for production runner: "
                     << model_path);
            return true;
        }

        /*
         * Publish payload lifetime independently from the rank's synthetic
         * primary device. An auxiliary ExpertOverlay rank is represented by a
         * CPU primary in RankExecutionPlan because it does not own the dense
         * continuation graph, but it may own CPU and GPU expert endpoints.
         * Its GGUF mapping is only a source for exact expert selections and
         * must never be mistaken for the dense, inference-resident CPU model.
         */
        const auto overlay_execution_for_load =
            resolveOverlayExecutionPlanForRunner(
                config_.moe_routed_expert_plan,
                moe_expert_overlay_mpi_ctx_
                    ? moe_expert_overlay_mpi_ctx_
                    : mpi_ctx_);
        const bool sparse_overlay_payload =
            overlay_execution_for_load &&
            !overlay_execution_for_load->currentRankPlan().loads_root_weights;
        if (sparse_overlay_payload)
        {
            weight_config.payload_access_pattern =
                ModelPayloadAccessPattern::SparseSelection;
        }
        else if (plan_.primary_device.device_type != DeviceType::CPU)
        {
            weight_config.payload_access_pattern =
                ModelPayloadAccessPattern::DeviceStaging;
        }
        else
        {
            weight_config.payload_access_pattern =
                ModelPayloadAccessPattern::DenseCpuResident;
        }

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
        bool node_has_dense_cpu_payload =
            weight_config.payload_access_pattern ==
            ModelPayloadAccessPattern::DenseCpuResident;
        if (is_multi_rank)
        {
            int local_dense_cpu_payload =
                node_has_dense_cpu_payload ? 1 : 0;
            int any_dense_cpu_payload = local_dense_cpu_payload;
            MPI_Comm cache_comm = mpi_ctx_->intra_node_comm();
            if (cache_comm == MPI_COMM_NULL)
                cache_comm = mpi_ctx_->communicator();
            if (cache_comm != MPI_COMM_NULL)
            {
                MPI_Allreduce(
                    &local_dense_cpu_payload,
                    &any_dense_cpu_payload,
                    1,
                    MPI_INT,
                    MPI_MAX,
                    cache_comm);
            }
            node_has_dense_cpu_payload = any_dense_cpu_payload != 0;
        }
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

        const bool should_prepopulate_page_cache =
            prepopulate_page_cache_enabled &&
            config_.use_mmap &&
            node_has_dense_cpu_payload;

        if (should_prepopulate_page_cache && !is_multi_rank)
        {
            // CPU decode reads weights from the mapping for the lifetime of the
            // model, so a NUMA-local prewarm remains useful. GPU-only startup is
            // deliberately demand-paged: pre-reading the entire GGUF defeats the
            // bounded staging contract on low-RAM/high-VRAM hosts.
            prepopulate_page_cache("Single-rank CPU");
            weight_config.skip_mmap_cache_eviction = true;
        }
        else if (should_prepopulate_page_cache && is_multi_rank)
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
        else if (prepopulate_page_cache_enabled &&
                 config_.use_mmap &&
                 weight_config.payload_access_pattern ==
                     ModelPayloadAccessPattern::DeviceStaging)
        {
            LOG_DEBUG("GPU-only weight loading: skipping whole-file page-cache prepopulation; "
                      "GGUF pages will be consumed and discarded incrementally");
        }
        else if (prepopulate_page_cache_enabled &&
                 config_.use_mmap &&
                 weight_config.payload_access_pattern ==
                     ModelPayloadAccessPattern::SparseSelection)
        {
            LOG_DEBUG(
                "Sparse ExpertOverlay weight loading: skipping whole-file "
                "page-cache prepopulation; only selected expert slices will "
                "fault source pages");
            PerfStatsCollector::addCounter(
                "weight_loading",
                "mmap_sparse_selection_prepopulate_bypass",
                1.0,
                "load",
                {},
                {{"rank", std::to_string(mpi_ctx_ ? mpi_ctx_->rank() : 0)}});
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
        if (!model_ctx_ || !config_.moe_routed_expert_plan)
            return true;

        const bool requested_without_placements =
            config_.moe_routed_expert_plan
                ->usesExpertOverlayAuthority() &&
            config_.moe_routed_expert_plan->placements.empty();

        try
        {
            const MTPRuntimeConfig &mtp =
                plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
            if (config_.moe_routed_expert_plan
                    ->usesExpertOverlayAuthority())
            {
                const auto overlay_world =
                    moe_expert_overlay_mpi_ctx_
                        ? moe_expert_overlay_mpi_ctx_
                        : mpi_ctx_;
                const int world_rank = overlay_world
                                           ? overlay_world->rank()
                                           : plan_.rank;
                const int world_size = overlay_world
                                           ? overlay_world->world_size()
                                           : 1;
                if (world_rank != plan_.rank)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay capacity communicator rank differs from the production rank plan");
                }

                const auto metadata =
                    resolveMoERoutedExpertModelMetadataForModel(
                        *model_ctx_, mtp);
                const auto layer_weight_manifest =
                    buildMoEOverlayLayerWeightManifestFromGGUF(
                        model_ctx_->concreteLoader().getModel(),
                        metadata.num_layers,
                        metadata.num_experts);
                const auto execution_plan =
                    resolveMoEExpertOverlayExecutionPlan(
                        config_.moe_routed_expert_plan,
                        MoEExpertOverlayExecutionPlanResolverOptions{
                            .current_world_rank = world_rank,
                            .world_size = world_size,
                        });
                const auto policy = overlayCapacityPolicy(
                    *config_.moe_routed_expert_plan,
                    config_,
                    world_size);

                const auto profile = ModelMemoryProfile::fromGGUF(
                    model_ctx_->model());
                const auto &rank_inventory =
                    cluster_inventory_.getRank(world_rank);
                const MoEOverlayGPUWeightLoadCapacityInput
                    gpu_weight_load{
                        .policy =
                            configuredGPUWeightLoadMemoryPolicy(),
                        .maximum_source_bytes =
                            maximumGGUFTensorPayloadBytes(
                                model_ctx_->concreteLoader().getModel()),
                    };

                /*
                 * Captured-prefill residency and expert residency spend the
                 * same physical bytes. Evaluate them as one descending
                 * admission search so a large graph bucket cannot consume the
                 * space required for complete fallback-tier coverage. All
                 * ranks execute the same candidate/consensus sequence before
                 * any graph or migration pool receives a stable address.
                 */
                std::vector<int> graph_row_candidates{0};
                if (debugEnv().execution.gpu_graphs &&
                    debugEnv().execution.prefill_graph_buckets)
                {
                    /*
                     * ExpertOverlay executes long prompts as ordered captured
                     * segments.  Its resident graph is therefore bounded by
                     * the configured physical segment, while max_seq_len
                     * continues to size the independent KV/request state.
                     * Admitting a larger dense graph here would make the
                     * workspace-family declaration build a non-executable
                     * full-context MoE graph and force its sparse scratch back
                     * to the very capacity this protocol is designed to avoid.
                     */
                    graph_row_candidates =
                        segmentedPrefillGraphRowCandidates(
                            debugEnv().execution.prefill_graph_bucket_sizes,
                            plan_.runtime.max_seq_len,
                            plan_.runtime.moe_routed_prefill
                                .overlay_segment_rows);
                    if (graph_row_candidates.empty())
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay captured-prefill admission has no positive resident graph-row candidate");
                    }
                }
                requireIdenticalOverlayGraphRowCandidates(
                    graph_row_candidates, overlay_world);

                std::optional<MoEOverlayLocalCapacityPlannerResult>
                    selected_local_capacity;
                std::optional<MoEOverlayResolvedCapacityPlan>
                    selected_capacity;
                std::string last_capacity_error;
                for (const int candidate_rows : graph_row_candidates)
                {
                    std::optional<MoEOverlayLocalCapacityPlannerResult>
                        local_capacity;
                    std::string local_capacity_error;
                    try
                    {
                        local_capacity =
                            MoEOverlayLocalCapacityPlanner::plan({
                                .model_profile = &profile,
                                .rank_plan = &plan_,
                                .overlay_plan =
                                    config_.moe_routed_expert_plan.get(),
                                .rank_inventory = &rank_inventory,
                                .builds_root_graph =
                                    execution_plan.buildsRootGraph(),
                                .require_host_memory_authority =
                                    policy.materialize_migration_fabric,
                                .max_gpu_memory_bytes = optionalMemoryBytes(
                                    config_.max_gpu_memory_mb,
                                    "GPU"),
                                .max_cpu_memory_bytes = optionalMemoryBytes(
                                    config_.max_cpu_memory_mb,
                                    "CPU"),
                                .resident_graph_rows = candidate_rows,
                                .gpu_weight_load = gpu_weight_load,
                            });
                    }
                    catch (const std::exception &error)
                    {
                        local_capacity_error = error.what();
                    }
                    catch (...)
                    {
                        local_capacity_error =
                            "non-standard exception while building the rank-local fixed BOM";
                    }

                    if (!allOverlayRanksReady(
                            local_capacity.has_value(), overlay_world))
                    {
                        last_capacity_error =
                            local_capacity_error.empty()
                                ? "ExpertOverlay fixed capacity planning failed on another overlay rank"
                                : local_capacity_error;
                        LOG_DEBUG(
                            "[MoEOverlayCapacity] Rejected resident graph candidate "
                            << candidate_rows << " rows: "
                            << last_capacity_error);
                        continue;
                    }

                    std::optional<MoEOverlayResolvedCapacityPlan>
                        candidate_capacity;
                    std::string candidate_capacity_error;
                    try
                    {
                        const auto physical_budgets =
                            gatherOverlayCapacityBudgets(
                                local_capacity->physical_budgets,
                                overlay_world);
                        const auto capacity_input =
                            MoEOverlayCapacityAdmission::buildResolverInput(
                                *config_.moe_routed_expert_plan,
                                metadata.num_experts,
                                layer_weight_manifest,
                                physical_budgets,
                                policy);
                        candidate_capacity =
                            MoEOverlayCapacityResolver::resolve(
                                capacity_input);
                    }
                    catch (const std::exception &error)
                    {
                        candidate_capacity_error = error.what();
                    }
                    catch (...)
                    {
                        candidate_capacity_error =
                            "non-standard exception while resolving the complete physical BOM";
                    }

                    if (!allOverlayRanksReady(
                            candidate_capacity.has_value(), overlay_world))
                    {
                        last_capacity_error =
                            candidate_capacity_error.empty()
                                ? "ExpertOverlay complete capacity resolution failed on another overlay rank"
                                : candidate_capacity_error;
                        LOG_DEBUG(
                            "[MoEOverlayCapacity] Rejected resident graph candidate "
                            << candidate_rows << " rows: "
                            << last_capacity_error);
                        continue;
                    }

                    selected_local_capacity = std::move(local_capacity);
                    selected_capacity = std::move(candidate_capacity);
                    if (candidate_rows > 0)
                        plan_.runtime.resident_graph_rows = candidate_rows;
                    break;
                }

                if (!selected_local_capacity || !selected_capacity)
                {
                    throw std::runtime_error(
                        "ExpertOverlay has no resident captured-prefill shape with complete expert coverage under the physical BOM" +
                        (last_capacity_error.empty()
                             ? std::string{}
                             : ": " + last_capacity_error));
                }

                const auto &capacity = *selected_capacity;
                LOG_DEBUG(
                    "[MoEOverlayCapacity] Selected rank-local fixed BOM at "
                    << selected_local_capacity->resident_graph_rows
                    << " resident rows:\n"
                    << selected_local_capacity->fixed_memory_plan.renderTable());
                auto capacity_bound = std::make_shared<
                    MoERoutedExpertPlacementPlan>(
                    MoEOverlayCapacityResolver::installResolvedQuotas(
                        *config_.moe_routed_expert_plan,
                        capacity));
                config_.moe_routed_expert_plan =
                    std::move(capacity_bound);

                for (const auto &resource : capacity.physical_resources)
                {
                    LOG_DEBUG(
                        "[MoEOverlayCapacity] resource="
                        << resource.resource_id
                        << " device=" << resource.device.toString()
                        << " usable=" << resource.usable_budget_bytes
                        << " fixed=" << resource.fixed_bytes
                        << " staging=" << resource.transfer_staging_bytes
                        << " reserve=" << resource.safety_reserve_bytes
                        << " shadow=" << resource.shadow_bytes
                        << " live_experts="
                        << resource.live_expert_bytes
                        << " remaining=" << resource.remaining_bytes);
                    PerfStatsCollector::addCounter(
                        "moe_overlay_capacity",
                        "physical_resource_admitted",
                        1.0,
                        "model_setup",
                        resource.device.toString(),
                        {{"resource_id", resource.resource_id},
                         {"usable_bytes", std::to_string(
                                              resource.usable_budget_bytes)},
                         {"fixed_bytes", std::to_string(
                                             resource.fixed_bytes)},
                         {"staging_bytes", std::to_string(
                                               resource.transfer_staging_bytes)},
                         {"safety_reserve_bytes", std::to_string(
                                                      resource.safety_reserve_bytes)},
                         {"shadow_bytes", std::to_string(
                                              resource.shadow_bytes)},
                         {"live_expert_bytes", std::to_string(
                                                   resource.live_expert_bytes)},
                         {"remaining_bytes", std::to_string(
                                                 resource.remaining_bytes)}});
                }
                PerfStatsCollector::addCounter(
                    "moe_overlay_capacity",
                    "exact_capacity_plan_installed",
                    1.0,
                    "model_setup",
                    "expert_overlay",
                    {{"tiers", std::to_string(capacity.tiers.size())},
                     {"physical_resources", std::to_string(
                                                capacity.physical_resources.size())},
                     {"layers", std::to_string(
                                    capacity.layer_footprints.size())},
                     {"migration_fabric",
                      policy.materialize_migration_fabric
                          ? "true"
                          : "false"},
                     {"resident_graph_rows", std::to_string(
                                                   selected_local_capacity
                                                       ->resident_graph_rows)},
                     {"priority_only", "true"}});
            }

            auto frozen_plan = freezeMoEExpertOverlayPlanForModel(
                *model_ctx_,
                config_.moe_routed_expert_plan,
                mtp);
            if (!frozen_plan)
                return true;

            config_.moe_routed_expert_plan = std::move(frozen_plan);
            if (config_.moe_routed_expert_plan
                    ->usesExpertOverlayAuthority())
            {
                LOG_DEBUG("[OrchestrationRunner] MoE expert overlay plan frozen: placements="
                          << config_.moe_routed_expert_plan->placements.size()
                          << " routed_tiers=" << config_.moe_routed_expert_plan->routed_tiers.size()
                          << " domains=" << config_.moe_routed_expert_plan->domains.size()
                          << (requested_without_placements ? " (planned from model metadata)" : ""));
            }
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Failed to freeze MoE expert overlay plan from model metadata: ") + e.what());
        }

        return true;
    }

    bool OrchestrationRunner::initializeMoEExpertOverlayResidencyAuthority()
    {
        shutdownMoEExpertOverlayResidencyMaintenance();
        moe_expert_overlay_participant_residency_.reset();
        moe_expert_overlay_residency_authority_.reset();
        moe_expert_overlay_decode_histogram_.reset();
        moe_expert_overlay_interference_probe_.reset();

        const auto &plan = config_.moe_routed_expert_plan;
        if (!model_ctx_ || !plan ||
            !plan->usesExpertOverlayAuthority())
            return true;

        try
        {
            const MTPRuntimeConfig &mtp =
                plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
            const auto metadata =
                resolveMoERoutedExpertModelMetadataForModel(
                    *model_ctx_,
                    mtp);
            if (metadata.num_layers <= 0 || metadata.num_experts <= 0)
            {
                return setError(
                    "Tiered ExpertOverlay residency could not resolve positive model geometry");
            }

            const MoEExpertOwnerMap owner_map =
                MoEExpertOwnerMap::build(*plan);
            const bool observes_residency =
                config_.moe_rebalance.mode !=
                MoERebalanceRuntimeMode::Off;
            const bool dynamic_residency =
                config_.moe_rebalance.mode ==
                MoERebalanceRuntimeMode::Dynamic;
            const bool host_dynamic_residency =
                dynamic_residency &&
                !usesHomogeneousGpuDeviceResidentMoEOverlayAuthority();

            if (observes_residency)
            {
                DecodeExpertHistogramConfig histogram_config;
                histogram_config.num_layers = metadata.num_layers;
                histogram_config.num_experts = metadata.num_experts;
                histogram_config.top_k =
                    model_ctx_->concreteLoader().getInt(
                        model_ctx_->architecture() + ".expert_used_count",
                        0);
                histogram_config.window_size =
                    std::max(1, config_.moe_rebalance.window_size);
                histogram_config.token_boundary_layer_idx = -1;
                for (const auto &placement : plan->placements)
                {
                    histogram_config.token_boundary_layer_idx =
                        std::max(
                            histogram_config.token_boundary_layer_idx,
                            placement.layer);
                }
                for (const auto &participant : owner_map.participants())
                    histogram_config.sockets.push_back(participant.device);
                histogram_config.ownership = owner_map.layeredOwnership(
                    metadata.num_layers,
                    metadata.num_experts);

                if (histogram_config.top_k <= 0 ||
                    histogram_config.sockets.empty())
                {
                    return setError(
                        "Dynamic ExpertOverlay residency could not resolve routing or participant geometry");
                }
                moe_expert_overlay_decode_histogram_ =
                    std::make_shared<DecodeExpertHistogram>(
                        std::move(histogram_config));
                if (host_dynamic_residency)
                {
                    moe_expert_overlay_interference_probe_ =
                        std::make_shared<
                            MoEOverlayInferenceInterferenceProbe>();
                }
            }

            std::string perf_device;
            for (const auto &tier : plan->routed_tiers)
            {
                if (!perf_device.empty())
                    perf_device += '/';
                perf_device += tier.name;
            }
            if (perf_device.empty())
                perf_device = "expert_overlay";

            moe_expert_overlay_residency_authority_ =
                std::make_shared<MoEOverlayResidencyAuthority>(
                    MoEOverlayResidencyAuthority::Config{
                        .initial_plan = *plan,
                        .model_metadata = metadata,
                        .maintenance_mode =
                            config_.moe_rebalance.mode,
                        .histogram =
                            moe_expert_overlay_decode_histogram_.get(),
                        .participant_rebalance_policy = {
                            .enabled = host_dynamic_residency,
                            .imbalance_threshold_per_mille =
                                config_.moe_rebalance
                                    .dynamic_imbalance_threshold_per_mille,
                            .minimum_improvement_per_mille =
                                config_.moe_rebalance
                                    .dynamic_min_improvement_per_mille,
                            .maximum_swaps_per_layer =
                                config_.moe_rebalance
                                    .dynamic_max_swaps_per_layer,
                            .maximum_plan_entries_per_wave =
                                config_.moe_rebalance
                                    .dynamic_max_plan_entries_per_wave,
                            .minimum_window_activations =
                                config_.moe_rebalance
                                    .dynamic_min_window_activations,
                        },
                        /* Physical shadow/staging capacity remains the hard
                         * bound even when policy admits multiple independent
                         * endpoint/layer cycles in one publication epoch. */
                        .shadow_slots_per_endpoint_layer = 1,
                        .max_concurrent_cycles =
                            config_.moe_rebalance
                                .migration_max_cycles_per_wave,
                        .perf_device = std::move(perf_device),
                    });

            const auto initial_snapshot =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!initial_snapshot || !initial_snapshot->valid())
            {
                return setError(
                    "Tiered ExpertOverlay residency authority did not publish "
                    "a valid initial epoch");
            }

            const auto overlay_world =
                moe_expert_overlay_mpi_ctx_
                    ? moe_expert_overlay_mpi_ctx_
                    : mpi_ctx_;
            const int current_rank = overlay_world ? overlay_world->rank() : 0;
            const int world_size =
                overlay_world ? overlay_world->world_size() : 1;
            std::vector<int> local_participant_ids;
            local_participant_ids.reserve(owner_map.participants().size());
            for (const auto &participant : owner_map.participants())
            {
                if (!participant.world_rank_known)
                {
                    /*
                     * A one-rank plan has no remote placement ambiguity.  In a
                     * real multi-rank world, however, accepting an unresolved
                     * endpoint would let two ranks prepare or omit the same
                     * bank, so the hardware-binding phase must resolve it.
                     */
                    if (world_size != 1)
                    {
                        return setError(
                            "Tiered ExpertOverlay participant " +
                            std::to_string(participant.participant_id) +
                            " has no resolved world-rank owner");
                    }
                    local_participant_ids.push_back(
                        participant.participant_id);
                    continue;
                }
                if (participant.world_rank == current_rank)
                {
                    local_participant_ids.push_back(
                        participant.participant_id);
                }
            }

            moe_expert_overlay_participant_residency_ =
                std::make_shared<
                    MoEOverlayParticipantResidencyRegistry>(
                    MoEOverlayParticipantResidencyRegistry::Config{
                        .owner_map = initial_snapshot->owner_map,
                        .local_participant_ids =
                            std::move(local_participant_ids),
                        .num_layers = metadata.num_layers,
                        .num_experts = metadata.num_experts,
                        .initial_epoch = initial_snapshot->epoch,
                        .retained_epoch_capacity = 2,
                        .collect_economy_service_measurements =
                            host_dynamic_residency,
                    });

            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "production_authorities_created",
                1.0,
                "model_setup",
                {},
                {
                    {"dynamic", dynamic_residency ? "true" : "false"},
                    {"maintenance_mode",
                     moeRebalanceRuntimeModeToString(
                         config_.moe_rebalance.mode)},
                    {"participants",
                     std::to_string(owner_map.participants().size())},
                    {"local_participants",
                     std::to_string(
                         moe_expert_overlay_participant_residency_
                             ->localParticipantIds()
                             .size())},
                    {"tiers", std::to_string(plan->routed_tiers.size())},
                    {"world_rank",
                     std::to_string(current_rank)},
                    {"world_size", std::to_string(world_size)},
                });
        }
        catch (const std::exception &error)
        {
            return setError(
                std::string(
                    "Failed to initialize production ExpertOverlay residency authority: ") +
                error.what());
        }

        return true;
    }

    bool OrchestrationRunner::
        usesHomogeneousGpuDeviceResidentMoEOverlayAuthority() const
    {
        const auto &plan = config_.moe_routed_expert_plan;
        if (!plan || !plan->usesExpertOverlayAuthority() ||
            plan->authority_execution !=
                MoEOverlayAuthorityExecutionKind::
                    HomogeneousDeviceResident)
        {
            return false;
        }
        if (plan->routed_tiers.size() != 1u)
        {
            throw std::logic_error(
                "A device-resident ExpertOverlay authority must have exactly one routed tier");
        }

        const std::string &domain_name =
            plan->routed_tiers.front().domain;
        const auto domain = std::find_if(
            plan->domains.begin(),
            plan->domains.end(),
            [&](const RoutedExpertDomain &candidate)
            {
                return candidate.name == domain_name;
            });
        if (domain == plan->domains.end() ||
            domain->participants.empty())
        {
            throw std::logic_error(
                "A device-resident ExpertOverlay authority cannot resolve its routed participant domain");
        }
        return std::all_of(
            domain->participants.begin(),
            domain->participants.end(),
            [](const GlobalDeviceAddress &participant)
            {
                return participant.isGPU();
            });
    }

    bool OrchestrationRunner::
        initializeMoEExpertOverlayResidencyMaintenance()
    {
        shutdownMoEExpertOverlayResidencyMaintenance();
        if (!moe_expert_overlay_residency_authority_)
            return true;
        if (!moe_expert_overlay_participant_residency_ ||
            !moe_expert_overlay_participant_residency_->allInitialBanksReady())
        {
            std::ostringstream detail;
            if (moe_expert_overlay_participant_residency_)
            {
                for (const auto &deficit :
                     moe_expert_overlay_participant_residency_
                         ->incompleteInitialBanks())
                {
                    detail << " p" << deficit.participant_id << '@'
                           << deficit.device.to_string();
                    if (deficit.publication_pending)
                    {
                        detail << "{publication-pending}";
                        continue;
                    }
                    detail << "{missing-layers=";
                    for (std::size_t index = 0;
                         index < deficit.missing_layers.size();
                         ++index)
                    {
                        if (index != 0)
                            detail << ',';
                        detail << deficit.missing_layers[index];
                    }
                    detail << '}';
                }
            }
            return setError(
                "ExpertOverlay graph construction did not publish every process-local initial prepared bank:" +
                detail.str());
        }

        const auto initial_snapshot =
            moe_expert_overlay_residency_authority_->snapshot();
        if (!initial_snapshot || !initial_snapshot->valid())
        {
            return setError(
                "ExpertOverlay maintenance requires a valid initial residency snapshot");
        }

        std::string perf_device;
        for (const auto &tier : initial_snapshot->placement_plan->routed_tiers)
        {
            if (!perf_device.empty())
                perf_device += '/';
            perf_device += tier.name;
        }
        if (perf_device.empty())
            perf_device = "expert_overlay";

        if (moe_expert_overlay_residency_authority_->maintenanceMode() !=
            config_.moe_rebalance.mode)
        {
            return setError(
                "ExpertOverlay authority maintenance mode disagrees with the frozen runtime configuration");
        }

        if (!moe_expert_overlay_residency_authority_
                 ->migrationEnabled())
        {
            /*
             * Off and Observe have no physical movement authority. Execute
             * the typed immobility check once so PerfStats proves zero
             * migrations without allocating a fabric or polling thread.
             */
            StaticMoEOverlayTransportGuard guard;
            const auto transaction =
                moe_expert_overlay_residency_authority_
                    ->proposeFromHistogram();
            const auto result =
                moe_expert_overlay_residency_authority_->beginApply(
                    transaction, guard);
            if (result.status !=
                MoEOverlayResidencyApplyStatus::StaticNoMovement)
            {
                return setError(
                    result.error.empty()
                        ? "Movement-disabled ExpertOverlay maintenance did not prove zero movement"
                        : result.error);
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "maintenance_not_started",
                1.0,
                "model_setup",
                perf_device,
                {{"runtime_mode",
                  moeRebalanceRuntimeModeToString(
                      config_.moe_rebalance.mode)},
                 {"reason", "explicit_runtime_policy"},
                 {"movement", "false"}});
            return true;
        }

        if (usesHomogeneousGpuDeviceResidentMoEOverlayAuthority())
        {
            if (!runner_)
            {
                return setError(
                    "Homogeneous device-resident ExpertOverlay authority has no constructed inference runner");
            }
            const auto runner_execution =
                runner_->moeOverlayAuthorityExecution();
            if (runner_execution !=
                MoEOverlayAuthorityExecutionKind::
                    HomogeneousDeviceResident)
            {
                return setError(
                    "ExpertOverlay graph family disagrees with the frozen homogeneous device-resident authority selection");
            }
            if (!runner_
                     ->deviceResidentMoEOverlayMaintenanceReady())
            {
                return setError(
                    "Homogeneous device-resident ExpertOverlay did not materialize its complete captured maintenance family");
            }

            /*
             * The Qwen graph family has already materialized the captured
             * all-layer transaction and its NCCL/RCCL transfer stream. Do not
             * construct a host fabric or maintenance thread here: that would
             * create a second planner and a second durable epoch writer.
             */
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "production_maintenance_composed",
                1.0,
                "model_setup",
                perf_device,
                {{"authority_execution", "homogeneous_device_resident"},
                 {"background", "device_stream"},
                 {"host_maintenance_service", "false"},
                 {"durable_writer_count", "1"}});
            return true;
        }

        const auto local_ids =
            moe_expert_overlay_participant_residency_
                ->localParticipantIds();

        try
        {
            const auto overlay_world =
                moe_expert_overlay_mpi_ctx_
                    ? moe_expert_overlay_mpi_ctx_
                    : mpi_ctx_;
            const int world_rank = overlay_world ? overlay_world->rank() : 0;
            const int world_size =
                overlay_world ? overlay_world->world_size() : 1;
            const bool distributed = world_size > 1;
            const auto capacity_policy = overlayCapacityPolicy(
                *initial_snapshot->placement_plan,
                config_,
                world_size);
            if (!capacity_policy.materialize_migration_fabric)
            {
                throw std::logic_error(
                    "Dynamic ExpertOverlay maintenance disagrees with its setup-time capacity policy");
            }
            const MTPRuntimeConfig &mtp =
                plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
            const ExpertHistogramProductionSourceMask active_sources{
                true,
                true,
                mtp.enabled,
            };
            const auto metadata =
                resolveMoERoutedExpertModelMetadataForModel(
                    *model_ctx_,
                    mtp);
            auto layer_weight_manifest =
                buildMoEOverlayLayerWeightManifestFromGGUF(
                    model_ctx_->concreteLoader().getModel(),
                    metadata.num_layers,
                    metadata.num_experts);
            auto calibration_layer_catalog = std::make_shared<
                MoEOverlayEconomyCalibrationLayerCatalog>(
                layer_weight_manifest);

            const auto execution_plan =
                resolveMoEExpertOverlayExecutionPlan(
                    initial_snapshot->placement_plan,
                    MoEExpertOverlayExecutionPlanResolverOptions{
                        .current_world_rank = world_rank,
                        .world_size = world_size,
                    });
            std::shared_ptr<IMoEOverlayEconomyEvidenceExchange>
                economy_evidence_exchange;

            if (distributed)
            {
                if (!overlay_world)
                {
                    throw std::logic_error(
                        "Distributed ExpertOverlay maintenance lost its overlay MPI context");
                }

                /*
                 * One simple closed cycle visits each physical participant at
                 * most once, but policy may admit several cycles from
                 * different layers in one wave. Size the remote projection
                 * pool from all three factors so raising the public cycle cap
                 * also raises its exact setup-time lane BOM. Every MPI lane
                 * and receive staging buffer is materialized now, never on
                 * maintenance or inference paths.
                 */
                const MoEOverlayRemoteProjectionLaneBudget lane_budget{
                    .maximum_participants_per_cycle =
                        std::max<std::size_t>(
                            1,
                            initial_snapshot->owner_map
                                .participants().size()),
                    .maximum_concurrent_cycles =
                        config_.moe_rebalance
                            .migration_max_cycles_per_wave,
                    .projections_per_expert =
                        kMoEOverlayExpertProjectionCount,
                };
                /*
                 * Duplicate every private communicator in one invariant order
                 * on every rank before fallible rank-local slot construction.
                 * A local VRAM or format error can then unwind idle lanes
                 * without leaving a peer blocked in a later collective dup.
                 */
                moe_expert_overlay_residency_consensus_ =
                    std::make_shared<MoEOverlayMPIResidencyConsensus>(
                        MoEOverlayMPIResidencyConsensus::Config{
                            .mpi_context = overlay_world,
                            .perf_device = perf_device,
                        });
                moe_expert_overlay_remote_projection_transport_ =
                    std::make_shared<
                        MoEOverlayMPIRemoteProjectionTransport>(
                        MoEOverlayMPIRemoteProjectionTransport::Config{
                            .mpi_context = overlay_world,
                            .lane_budget = lane_budget,
                            .staging_capacity_bytes =
                                capacity_policy.staging_capacity_bytes,
                            .perf_device = perf_device,
                        });
                moe_expert_overlay_histogram_publisher_ =
                    std::make_shared<MoEOverlayMPIHistogramPublisher>(
                        MoEOverlayMPIHistogramPublisher::Config{
                            .mpi_context = overlay_world,
                            .coordinator_world_rank =
                                execution_plan.continuation_root_rank,
                            .num_layers = metadata.num_layers,
                            .num_experts = metadata.num_experts,
                            .perf_device = perf_device,
                        });
                economy_evidence_exchange = std::make_shared<
                    MoEOverlayMPIEconomyEvidenceExchange>(
                    MoEOverlayMPIEconomyEvidenceExchange::Config{
                        .mpi_context = overlay_world,
                        .owner_map = initial_snapshot->owner_map,
                        .num_layers = metadata.num_layers,
                        .active_sources = active_sources,
                        .perf_device = perf_device,
                    });
            }

            std::shared_ptr<IMoEOverlayResidencyTransport>
                maintenance_transport;
            std::shared_ptr<MoEOverlayMigrationMeasurementJournal>
                local_measurement_journal;
            std::shared_ptr<MoEOverlayEconomyCertificationController>
                local_economy_certification;
            std::string local_materialization_error;
            try
            {
                moe_expert_overlay_physical_residency_fabric_ =
                    MoEOverlayPhysicalResidencyFabric::create({
                        .registry =
                            moe_expert_overlay_participant_residency_,
                        .initial_snapshot = initial_snapshot,
                        .layer_weight_manifest =
                            std::move(layer_weight_manifest),
                        .remote_projection_transport =
                            moe_expert_overlay_remote_projection_transport_,
                        .shadow_slots_per_endpoint_layer =
                            capacity_policy.shadow_slots_per_endpoint_layer,
                        .staging_capacity_bytes =
                            capacity_policy.staging_capacity_bytes,
                        .gpu_vram_safety_margin_bytes =
                            MoEOverlayCapacityAdmissionPolicy::
                                kProductionGpuSafetyMarginBytes,
                        .collect_economy_measurements = true,
                        .perf_device = perf_device,
                    });
                moe_expert_overlay_wave_factory_ =
                    std::make_unique<
                        MoEOverlayParticipantPreparedWaveFactory>(
                        MoEOverlayParticipantPreparedWaveFactory::Config{
                            .registry =
                                moe_expert_overlay_participant_residency_,
                            .transfer_provider =
                                moe_expert_overlay_physical_residency_fabric_,
                            .perf_device = perf_device,
                        });
                local_measurement_journal = std::make_shared<
                    MoEOverlayMigrationMeasurementJournal>(
                    MoEOverlayMigrationMeasurementJournal::Config{
                        .maximum_migrations_per_wave = 2,
                    });
                maintenance_transport =
                    moe_expert_overlay_migration_transport_ =
                        std::make_shared<MoEOverlayTierMigrationTransport>(
                            MoEOverlayTierMigrationTransport::Config{
                                .factory =
                                    moe_expert_overlay_wave_factory_.get(),
                                .projections_per_expert =
                                    kMoEOverlayExpertProjectionCount,
                                .measurement_sink =
                                    local_measurement_journal,
                                .measurement_scope =
                                    MoEOverlayMigrationMeasurementScope::
                                        EconomyCalibrationOnly,
                                .require_complete_local_measurements =
                                    !distributed,
                                .perf_device = perf_device,
                            });

                if (distributed)
                {
                    moe_expert_overlay_distributed_migration_transport_ =
                        std::make_shared<
                            MoEOverlayDistributedResidencyTransport>(
                            MoEOverlayDistributedResidencyTransport::Config{
                                .local_transport =
                                    moe_expert_overlay_migration_transport_.get(),
                                .consensus =
                                    moe_expert_overlay_residency_consensus_,
                                .perf_device = perf_device,
                            });
                    maintenance_transport =
                        moe_expert_overlay_distributed_migration_transport_;
                }
            }
            catch (const std::exception &error)
            {
                local_materialization_error = error.what();
            }
            catch (...)
            {
                local_materialization_error =
                    "non-standard exception during rank-local overlay materialization";
            }

            if (distributed)
            {
                const int local_ready =
                    local_materialization_error.empty() ? 1 : 0;
                int all_ranks_ready = 0;
                const int reduce_result = MPI_Allreduce(
                    &local_ready,
                    &all_ranks_ready,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    overlay_world->communicator());
                if (reduce_result != MPI_SUCCESS)
                {
                    throw std::runtime_error(
                        "Distributed ExpertOverlay failed its setup-time rank-local materialization consensus");
                }
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "production_materialization_consensus",
                    all_ranks_ready ? 1.0 : 0.0,
                    "model_setup",
                    perf_device,
                    {{"world_rank", std::to_string(world_rank)},
                     {"world_size", std::to_string(world_size)},
                     {"blocking", "true"},
                     {"inference_path", "false"}});
                if (!all_ranks_ready)
                {
                    if (local_materialization_error.empty())
                    {
                        local_materialization_error =
                            "rank-local ExpertOverlay materialization failed on another overlay rank";
                    }
                    throw std::runtime_error(
                        local_materialization_error);
                }
            }
            else if (!local_materialization_error.empty())
            {
                throw std::runtime_error(local_materialization_error);
            }

            {
                auto calibration_planner = std::make_shared<
                    MoEOverlayEconomyCalibrationPlanner>(
                    MoEOverlayEconomyCalibrationPlanner::Config{
                        .live_snapshot = initial_snapshot,
                        .complete_expert_bytes_per_layer =
                            calibration_layer_catalog
                                ->completeExpertBytesPerLayer(),
                        .calibration_layers =
                            calibration_layer_catalog
                                ->representativeLayers(),
                    });
                auto calibration_ledger = std::make_shared<
                    MoEOverlayMigrationMeasurementLedger>(
                    MoEOverlayMigrationMeasurementLedger::Config{
                        .required_coordinates =
                            calibration_planner->requiredCoordinates(),
                        .warmup_samples_per_coordinate = 1,
                        .measured_samples_per_coordinate =
                            MoEOverlayEconomyProfileComposer::
                                kMinimumMigrationSamples,
                        .required_sources = active_sources,
                        .measurement_identity =
                            calibration_layer_catalog->identity(),
                    });
                auto calibration = std::make_shared<
                    MoEOverlayEconomyCalibrationController>(
                    MoEOverlayEconomyCalibrationController::Config{
                        .planner = std::move(calibration_planner),
                        .ledger = std::move(calibration_ledger),
                        .journal = local_measurement_journal,
                        .probe = moe_expert_overlay_interference_probe_,
                        .transport = maintenance_transport,
                        .evidence_exchange = economy_evidence_exchange,
                        .perf_device = perf_device,
                    });
                local_economy_certification = std::make_shared<
                    MoEOverlayEconomyCertificationController>(
                    MoEOverlayEconomyCertificationController::Config{
                        .calibration = std::move(calibration),
                        .registry =
                            moe_expert_overlay_participant_residency_,
                        .layer_catalog =
                            std::move(calibration_layer_catalog),
                        .authority =
                            moe_expert_overlay_residency_authority_,
                        .model_metadata = metadata,
                        .economy_policy =
                            MoEOverlayMigrationEconomyPolicy{
                                .payoff_horizon_tokens =
                                    config_.moe_rebalance
                                        .migration_payoff_horizon_tokens,
                            },
                        .active_sources = active_sources,
                        .evidence_exchange =
                            economy_evidence_exchange,
                        .perf_device = perf_device,
                    });
            }

            moe_expert_overlay_maintenance_service_ =
                std::make_unique<
                    MoEOverlayResidencyMaintenanceService>(
                    MoEOverlayResidencyMaintenanceService::Config{
                        .authority =
                            moe_expert_overlay_residency_authority_,
                        .transport =
                            std::move(maintenance_transport),
                        .economy_certification =
                            std::move(local_economy_certification),
                        .histogram_publisher =
                            moe_expert_overlay_histogram_publisher_,
                        .idle_poll_interval =
                            std::chrono::milliseconds(2),
                        .perf_device = perf_device,
                    });

            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "production_maintenance_composed",
                1.0,
                "model_setup",
                perf_device,
                {{"shadow_slots_per_endpoint_layer",
                  std::to_string(
                      capacity_policy.shadow_slots_per_endpoint_layer)},
                 {"max_concurrent_cycles",
                  std::to_string(
                      config_.moe_rebalance
                          .migration_max_cycles_per_wave)},
                 {"staging_bytes",
                  std::to_string(
                      capacity_policy.staging_capacity_bytes)},
                 {"participants",
                  std::to_string(local_ids.size())},
                 {"global_participants",
                  std::to_string(
                      initial_snapshot->owner_map.participants().size())},
                 {"world_rank", std::to_string(world_rank)},
                 {"world_size", std::to_string(world_size)},
                 {"distributed", distributed ? "true" : "false"},
                 {"histogram_coordinator_rank",
                  std::to_string(execution_plan.continuation_root_rank)},
                 {"background", "true"}});
        }
        catch (const std::exception &error)
        {
            shutdownMoEExpertOverlayResidencyMaintenance();
            return setError(
                std::string(
                    "Failed to compose production ExpertOverlay residency maintenance: ") +
                error.what());
        }
        return true;
    }

    void OrchestrationRunner::
        shutdownMoEExpertOverlayResidencyMaintenance() noexcept
    {
        if (moe_expert_overlay_maintenance_service_)
        {
            try
            {
                moe_expert_overlay_maintenance_service_->stopAndDrain();
                if (!moe_expert_overlay_maintenance_service_->healthy())
                {
                    LOG_ERROR(
                        "[OrchestrationRunner] ExpertOverlay maintenance stopped after failure: "
                        << moe_expert_overlay_maintenance_service_
                               ->failureMessage());
                }
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[OrchestrationRunner] ExpertOverlay maintenance drain failed: "
                    << error.what());
            }
            catch (...)
            {
                LOG_ERROR(
                    "[OrchestrationRunner] ExpertOverlay maintenance drain raised a non-standard exception");
            }
        }

        /* Dependency order is part of the asynchronous ownership contract. */
        moe_expert_overlay_maintenance_service_.reset();
        if (moe_expert_overlay_histogram_publisher_)
        {
            try
            {
                moe_expert_overlay_histogram_publisher_->stopAndDrain();
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[OrchestrationRunner] ExpertOverlay histogram publisher drain failed: "
                    << error.what());
            }
            catch (...)
            {
                LOG_ERROR(
                    "[OrchestrationRunner] ExpertOverlay histogram publisher drain raised a non-standard exception");
            }
        }
        moe_expert_overlay_histogram_publisher_.reset();
        moe_expert_overlay_distributed_migration_transport_.reset();
        moe_expert_overlay_migration_transport_.reset();
        moe_expert_overlay_wave_factory_.reset();
        moe_expert_overlay_physical_residency_fabric_.reset();
        moe_expert_overlay_remote_projection_transport_.reset();
        moe_expert_overlay_residency_consensus_.reset();
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

        struct DevicePlanningInventory
        {
            size_t total_bytes = 0;
            size_t free_bytes = 0;
            int compute_units = 0;
        };
        auto inventoryForDevice = [&](DeviceId device)
        {
            DevicePlanningInventory inventory;
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
                            inventory.total_bytes = gpu.memory_bytes;
                            inventory.free_bytes = gpu.free_memory_bytes;
                            inventory.compute_units = gpu.compute_units;
                            break;
                        }
                    }
                }
                else
                {
                    inventory.total_bytes = rank_inv.cpu_memory_bytes;
                    inventory.free_bytes = rank_inv.cpu_memory_bytes;
                    inventory.compute_units = rank_inv.cpu.compute_units;
                }
            }
            return inventory;
        };

        std::vector<DevicePlanConfig> device_configs;
        auto makeConfigForDevice = [&](DeviceId device, int shard_index, int total_shards)
        {
            const DevicePlanningInventory inventory =
                inventoryForDevice(device);

            DevicePlanConfig cfg;
            cfg.device = device;
            cfg.device_total_bytes = inventory.total_bytes;
            cfg.device_free_bytes = inventory.free_bytes;
            cfg.device_compute_units = inventory.compute_units;
            cfg.shard_index = shard_index;
            cfg.total_shards = total_shards;
            cfg.first_layer = plan_.first_layer;
            cfg.last_layer = plan_.last_layer;
            cfg.batch_size = plan_.runtime.batch_size;
            cfg.max_seq_len = plan_.runtime.max_seq_len;
            cfg.activation_seq_len = resolveActivationBufferSeqLen(cfg.max_seq_len, device);
            cfg.mtp_enabled = plan_.runtime.mtp.enabled;
            cfg.mtp_target_query_rows =
                resolveMTPMaxTargetQueryRows(plan_.runtime.mtp);
            cfg.mtp_terminal_logits_layout =
                resolveMTPTerminalLogitsLayout(
                    total_shards > 1,
                    plan_.runtime.mtp.terminal_head_policy);

            if (retained_prepared_weight_plan_validated_ && device.is_gpu())
            {
                const auto weight_manager = model_ctx_->concreteWeightManager();
                const auto store = weight_manager
                                       ? weight_manager->preparedWeightStoreIfInitialized()
                                       : nullptr;
                const bool lifecycle_complete =
                    weight_manager &&
                    weight_manager->lifecycleGates().device_preparation_complete &&
                    weight_manager->lifecycleGates().graph_materialization_complete;
                if (!lifecycle_complete || !store ||
                    store->sizeForDevice(device) == 0u)
                {
                    throw std::runtime_error(
                        "Certified retained prepared weights are no longer "
                        "complete for " +
                        device.toString());
                }

                /*
                 * GPU inventory reports live free bytes after the retained pool
                 * allocation. MemoryPlanner must preserve that pool in its final
                 * BOM while charging only new graph/runtime storage. CPU inventory
                 * currently reports capacity rather than live free memory, so CPU
                 * remains conservatively full-footprint admitted.
                 */
                cfg.prepared_weight_admission =
                    PreparedWeightAdmission::ReuseCertifiedCompleteSet;
            }

            cfg.kv_precision = activationPrecisionToString(
                resolveKVCacheStoragePrecision(
                    plan_.runtime.kv_cache_precision,
                    device.is_cpu()));

            if (total_shards > 1 && profile.n_kv_heads > 0)
            {
                cfg.local_kv_heads = profile.n_kv_heads / total_shards;
                if (cfg.local_kv_heads < 1)
                    cfg.local_kv_heads = 1;
            }
            return cfg;
        };

        const bool has_expert_overlay =
            config_.moe_routed_expert_plan &&
            config_.moe_routed_expert_plan
                ->usesExpertOverlayAuthority();
        if (has_expert_overlay)
        {
            /*
             * ExpertOverlay is neither TP nor PP weight distribution. Build
             * one capacity record for every participant physically owned by
             * this MPI rank, then combine participants that share one local
             * DeviceId. Every continuation shard additionally owns its exact
             * dense/shared/global tensor shard and dense execution state;
             * auxiliary devices own only their exact expert selections.
             */
            const auto overlay_execution = resolveOverlayExecutionPlanForRunner(
                config_.moe_routed_expert_plan,
                moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_);
            if (!overlay_execution)
            {
                return setError(
                    "Tiered MoE overlay memory planning could not resolve the rank execution role");
            }

            const MoEExpertOwnerMap owner_map =
                MoEExpertOwnerMap::build(*config_.moe_routed_expert_plan);
            const int current_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            struct LocalOverlayDeviceResidency
            {
                DeviceId device = DeviceId::invalid();
                std::vector<int> selected_by_layer;
                /**
                 * @brief Local sparse participants with independently owned packets.
                 *
                 * A DeviceId is not sufficient here: NodeTP can expose
                 * more than one logical CPU participant through one kernel
                 * device identity, and those participants must never alias
                 * their graph-stable compact route tensors.
                 */
                int serial_compact_participant_count = 0;
                bool continuation = false;
                int continuation_shard_index = 0;
                int continuation_total_shards = 1;
                int continuation_first_layer = 0;
                int continuation_last_layer = -1;
            };
            std::vector<LocalOverlayDeviceResidency> local_residencies;

            auto residencyFor = [&](DeviceId device)
                -> LocalOverlayDeviceResidency &
            {
                auto found = std::find_if(
                    local_residencies.begin(),
                    local_residencies.end(),
                    [&](const auto &candidate)
                    {
                        return candidate.device == device;
                    });
                if (found != local_residencies.end())
                    return *found;

                LocalOverlayDeviceResidency residency;
                residency.device = device;
                residency.selected_by_layer.assign(
                    static_cast<size_t>(std::max(0, profile.n_layers)), 0);
                local_residencies.push_back(std::move(residency));
                return local_residencies.back();
            };

            for (const auto &participant : owner_map.participants())
            {
                if (!participant.world_rank_known ||
                    participant.world_rank != current_rank)
                {
                    continue;
                }

                auto &residency = residencyFor(participant.device);
                for (int layer = 0; layer < profile.n_layers; ++layer)
                {
                    const size_t owned = owner_map.expertsForParticipant(
                        layer, participant.participant_id).size();
                    if (owned > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                        residency.selected_by_layer[static_cast<size_t>(layer)] >
                            std::numeric_limits<int>::max() - static_cast<int>(owned))
                    {
                        return setError(
                            "MoE overlay participant expert count exceeds planner integer geometry");
                    }
                    residency.selected_by_layer[static_cast<size_t>(layer)] +=
                        static_cast<int>(owned);
                }
                /*
                 * Participant graph topology is independent of its initial
                 * expert quota. Every configured participant retains one
                 * serial packet so later residency epochs only replace data.
                 */
                if (residency.serial_compact_participant_count ==
                    std::numeric_limits<int>::max())
                {
                    return setError(
                        "MoE overlay serial compact participant count exceeds planner geometry");
                }
                ++residency.serial_compact_participant_count;
            }

            for (const auto &shard :
                 MoEOverlayLocalCapacityPlanner::continuationShards(
                     plan_, overlay_execution->buildsRootGraph()))
            {
                auto &residency = residencyFor(shard.device);
                if (residency.continuation)
                {
                    return setError(
                        "ExpertOverlay final memory validation cannot merge multiple continuation shards on " +
                        shard.device.toString());
                }
                residency.continuation = true;
                residency.continuation_shard_index = shard.shard_index;
                residency.continuation_total_shards = shard.total_shards;
                residency.continuation_first_layer = shard.first_layer;
                residency.continuation_last_layer = shard.last_layer;
            }

            if (local_residencies.empty())
            {
                return setError(
                    "MoE overlay rank owns no local continuation or expert participant devices");
            }

            device_configs.reserve(local_residencies.size());
            for (auto &residency : local_residencies)
            {
                auto cfg = makeConfigForDevice(
                    residency.device,
                    residency.continuation
                        ? residency.continuation_shard_index
                        : 0,
                    residency.continuation
                        ? residency.continuation_total_shards
                        : 1);
                if (residency.continuation)
                {
                    cfg.first_layer =
                        residency.continuation_first_layer;
                    cfg.last_layer =
                        residency.continuation_last_layer;
                }
                /*
                 * Qwen's graph-native overlay declares one serial execution
                 * family per participant.  Decode, prefill buckets, MTP
                 * verifier roles, and transformer layers are ordered by the
                 * production graph, so their compact route packet has one
                 * immutable owner per logical participant rather than one
                 * allocation per layer.  Keep that contract explicit in the
                 * preflight plan: a future concurrently runnable participant
                 * must select a different typed lifetime instead of silently
                 * sharing this storage.
                 */
                cfg.routed_expert_compact_buffer_lifetime =
                    RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;
                cfg.serial_routed_expert_participant_count =
                    residency.serial_compact_participant_count;
                /*
                 * The planner may admit fewer resident graph rows than the
                 * model's requested activation length.  Price and allocate
                 * only the admitted sparse segment envelope; using the
                 * pre-admission activation length here would recreate a
                 * larger compact family than the capacity plan certified.
                 */
                const int admitted_graph_rows =
                    plan_.runtime.resident_graph_rows > 0
                        ? plan_.runtime.resident_graph_rows
                        : cfg.activation_seq_len;
                const int overlay_segment_rows = std::min(
                    std::max(1, admitted_graph_rows),
                    plan_.runtime.moe_routed_prefill
                        .overlay_segment_rows);
                cfg.serial_routed_expert_compact_rows = std::max(
                    overlay_segment_rows,
                    cfg.mtp_enabled
                        ? cfg.mtp_target_query_rows
                        : 1);
                if (residency.continuation)
                {
                    cfg.execution_role =
                        DeviceExecutionMemoryRole::ContinuationGraph;
                    cfg.additional_weight_sets =
                        resolveAdditionalPersistentWeightSets(
                            config_.moe_routed_expert_plan
                                ->continuation_domain_spec
                                .effectiveDensePolicy(),
                            residency.continuation_total_shards);
                    cfg.weight_residency =
                        DeviceWeightResidency::
                            continuationWithSelectedRoutedExperts(
                                profile.expert_count,
                                std::move(residency.selected_by_layer));
                }
                else
                {
                    cfg.execution_role =
                        DeviceExecutionMemoryRole::RoutedExpertParticipant;
                    cfg.weight_residency =
                        DeviceWeightResidency::selectedRoutedExpertsOnly(
                            profile.expert_count,
                            std::move(residency.selected_by_layer));
                }
                device_configs.push_back(std::move(cfg));
            }
        }
        else if (plan_.usesLocalPP())
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

        const bool has_resident_graph_participant = std::any_of(
            device_configs.begin(),
            device_configs.end(),
            [](const DevicePlanConfig &config)
            {
                return config.device.is_gpu() ||
                       config.execution_role ==
                           DeviceExecutionMemoryRole::RoutedExpertParticipant;
            });

        MemoryPlan plan;
        if (has_resident_graph_participant &&
            debugEnv().execution.gpu_graphs &&
            debugEnv().execution.prefill_graph_buckets)
        {
            if (has_expert_overlay)
            {
                if (plan_.runtime.resident_graph_rows <= 0)
                {
                    return setError(
                        "Tiered ExpertOverlay reached final memory validation without a jointly admitted resident graph-row capacity");
                }

                for (auto &cfg : device_configs)
                {
                    if (cfg.device.is_gpu() ||
                        cfg.execution_role ==
                            DeviceExecutionMemoryRole::RoutedExpertParticipant)
                    {
                        cfg.activation_seq_len =
                            plan_.runtime.resident_graph_rows;
                    }
                }
                plan = MemoryPlanner::plan(profile, device_configs);
            }
            else
            {
                const auto configured_buckets = normalizePrefillGraphBuckets(
                    debugEnv().execution.prefill_graph_bucket_sizes);
                auto resident_plan =
                    MemoryPlanner::planLargestFittingResidentGraphRows(
                        profile,
                        device_configs,
                        configured_buckets);
                plan_.runtime.resident_graph_rows =
                    resident_plan.resident_graph_rows;
                plan = std::move(resident_plan.memory_plan);
            }

            if (plan.fits())
            {
                LOG_INFO("[MemoryPlanner] Selected resident graph capacity "
                         << plan_.runtime.resident_graph_rows
                         << " rows; full KV context remains "
                         << plan_.runtime.max_seq_len
                         << " tokens");
            }
        }
        else
        {
            plan = MemoryPlanner::plan(profile, device_configs);
            plan_.runtime.resident_graph_rows =
                device_configs.empty()
                    ? 0
                    : device_configs.front().activation_seq_len;
        }

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

    /**
     * @brief Publish one executable captured-prefill contract for ExpertOverlay.
     *
     * A sparse prefill transaction has to enter the same participant boundary
     * for every chunk. The contract freezes the root-selected bucket ladder
     * and, for a distributed world, reduces independent local capacities to
     * the one shape limit every endpoint can honor. This runs after memory
     * planning and before graph construction, so no captured graph or serial
     * compact arena can be bound to a larger uncoordinated request shape.
     */
    bool OrchestrationRunner::establishMoEOverlayPrefillScheduleContract()
    {
        plan_.runtime.overlay_prefill_schedule = {};

        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        const auto overlay_execution = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            overlay_context);
        if (!overlay_execution || !overlay_context)
        {
            return true;
        }
        const bool distributed = overlay_context->world_size() > 1;
        if (distributed && overlay_context->communicator() == MPI_COMM_NULL)
        {
            return setError(
                "Distributed MoE ExpertOverlay requires a live MPI "
                "communicator to publish its prefill schedule contract");
        }

        const int local_graph_capacity = plan_.runtime.resident_graph_rows;
        const int requested_segment_rows =
            plan_.runtime.moe_routed_prefill.overlay_segment_rows;
        if (local_graph_capacity <= 0 || requested_segment_rows <= 0)
        {
            return setError(
                "MoE ExpertOverlay rank has no positive planner-admitted graph "
                "capacity or configured prefill segment size");
        }
        const int local_capacity =
            std::min(local_graph_capacity, requested_segment_rows);

        int common_capacity = local_capacity;
        if (distributed &&
            MPI_Allreduce(
                &local_capacity,
                &common_capacity,
                1,
                MPI_INT,
                MPI_MIN,
                overlay_context->communicator()) != MPI_SUCCESS)
        {
            return setError(
                "Distributed MoE ExpertOverlay could not reduce its jointly "
                "admitted prefill segment capacity");
        }
        if (common_capacity <= 0)
        {
            return setError(
                "MoE ExpertOverlay resolved a non-positive prefill segment capacity");
        }

        const int continuation_root_rank =
            overlay_execution->continuation_root_rank;
        if (continuation_root_rank < 0 ||
            continuation_root_rank >= overlay_context->world_size())
        {
            return setError(
                "Distributed MoE ExpertOverlay has an invalid continuation "
                "root rank for prefill schedule publication");
        }

        std::vector<int> bucket_rows;
        int bucket_count = 0;
        if (!distributed ||
            overlay_context->rank() == continuation_root_rank)
        {
            bucket_rows = prefillGraphBucketsAtOrBelowCapacity(
                debugEnv().execution.prefill_graph_bucket_sizes,
                common_capacity);
            if (bucket_rows.empty() ||
                bucket_rows.size() >
                    static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return setError(
                    "Distributed MoE ExpertOverlay root could not publish a "
                    "bounded prefill bucket ladder");
            }
            bucket_count = static_cast<int>(bucket_rows.size());
        }

        if (distributed &&
            MPI_Bcast(
                &bucket_count,
                1,
                MPI_INT,
                continuation_root_rank,
                overlay_context->communicator()) != MPI_SUCCESS)
        {
            return setError(
                "Distributed MoE ExpertOverlay failed to publish its prefill "
                "bucket count");
        }
        if (bucket_count <= 0)
        {
            return setError(
                "MoE ExpertOverlay resolved an empty prefill bucket contract");
        }

        if (distributed &&
            overlay_context->rank() != continuation_root_rank)
        {
            bucket_rows.resize(static_cast<size_t>(bucket_count));
        }
        if (distributed &&
            MPI_Bcast(
                bucket_rows.data(),
                bucket_count,
                MPI_INT,
                continuation_root_rank,
                overlay_context->communicator()) != MPI_SUCCESS)
        {
            return setError(
                "Distributed MoE ExpertOverlay failed to publish its prefill "
                "bucket ladder");
        }

        const auto normalized_bucket_rows =
            normalizePrefillGraphBuckets(bucket_rows);
        if (normalized_bucket_rows != bucket_rows ||
            normalized_bucket_rows.empty() ||
            normalized_bucket_rows.back() > common_capacity)
        {
            return setError(
                "Distributed MoE ExpertOverlay received an invalid prefill "
                "bucket ladder");
        }

        plan_.runtime.overlay_prefill_schedule = {
            .graph_row_capacity = common_capacity,
            .bucket_rows = std::move(bucket_rows),
        };
        /*
         * Graph builders consume the resolved segment size through the normal
         * typed runner configuration. Publishing the reduced value here keeps
         * root and endpoint arenas identical even when one rank admitted a
         * smaller dense graph than its peers.
         */
        plan_.runtime.moe_routed_prefill.overlay_segment_rows =
            common_capacity;

        PerfStatsCollector::addCounter(
            "forward_graph",
            "moe_overlay_prefill_schedule_contract_rows",
            static_cast<double>(common_capacity),
            "model_setup",
            {},
            {
                {"authority", "continuation_root"},
                {"continuation_root_rank", std::to_string(continuation_root_rank)},
                {"local_graph_capacity", std::to_string(local_graph_capacity)},
                {"local_segment_capacity", std::to_string(local_capacity)},
                {"requested_segment_rows", std::to_string(requested_segment_rows)},
                {"bucket_count", std::to_string(bucket_count)},
                {"distributed", distributed ? "true" : "false"},
                {"immutable", "true"},
            });
        return true;
    }

    void OrchestrationRunner::printStartupBanner(FILE *stream)
    {
        // The rank owning dense continuation state is the banner authority.
        if (mpi_ctx_ && mpi_ctx_->rank() != mpi_coordinated_root_rank_)
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
            prec_oss << "Activations: "
                     << activationPrecisionToString(
                            plan_.runtime.activation_precision)
                     << " | KV Cache: ";
            if (plan_.runtime.kv_cache_precision ==
                KVCachePrecision::TQ)
            {
                prec_oss << kvCachePrecisionToString(
                    plan_.runtime.kv_cache_precision);
            }
            else
            {
                prec_oss << activationPrecisionToString(
                    resolveKVCacheStoragePrecision(
                        plan_.runtime.kv_cache_precision,
                        device.is_cpu()));
            }
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
                data.backend = cpuBackendDescription();
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
                char buf[64];
                const DeviceId primary_device =
                    DeviceAddressAdapter::toDeviceId(plan_.primary_device);
                const auto &load_config = debugEnv().rocm;
                if (primary_device.is_gpu() && config_.use_mmap &&
                    load_config.repack_budget_mb > 0)
                {
                    snprintf(
                        buf, sizeof(buf),
                        "%d MiB bounded staging cap",
                        load_config.repack_budget_mb);
                }
                else
                {
                    const double weight_gb =
                        static_cast<double>(weight_bytes) /
                        (1024.0 * 1024.0 * 1024.0);
                    snprintf(buf, sizeof(buf), "%.1f GB required", weight_gb);
                }
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

        // Print directly (bypassing LOG_INFO) to preserve ANSI colors.
        // LOG_INFO strips escape codes via its formatting pipeline.
        if (!banner.empty())
        {
            std::print(stream ? stream : stderr, "{}\n", banner);
        }
    }

    bool OrchestrationRunner::buildComputeGraph()
    {
        ScopedWeightLoadTimer timer(WeightLoadPhase::GRAPH_BUILD);

        auto overlay_execution_plan = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_);
        if (overlay_execution_plan)
        {
            if (!overlay_execution_plan->buildsRootGraph())
            {
                auto overlay_ctx =
                    moe_expert_overlay_mpi_ctx_
                        ? moe_expert_overlay_mpi_ctx_
                        : mpi_ctx_;
                const MTPRuntimeConfig &active_mtp =
                    plan_.runtime.mtp.enabled
                        ? plan_.runtime.mtp
                        : config_.mtp;
                runner_ = createMoEOverlayParticipantGraphRunner(
                    MoEOverlayParticipantGraphRunnerConfig{
                        .model_context = model_ctx_,
                        .mpi_context = std::move(overlay_ctx),
                        .placement_plan =
                            config_.moe_routed_expert_plan,
                        .residency_authority =
                            moe_expert_overlay_residency_authority_,
                        .participant_residency =
                            moe_expert_overlay_participant_residency_,
                        .decode_histogram =
                            moe_expert_overlay_decode_histogram_,
                        .max_seq_len = config_.max_seq_len,
                        /* The memory planner admits this exact bounded graph
                         * family.  Passing the full KV context here would
                         * over-allocate every remote compact route arena and
                         * make graph admission disagree with runtime storage. */
                        .max_graph_activation_rows =
                            std::max(
                                plan_.runtime.overlay_prefill_schedule
                                    .graph_row_capacity,
                                active_mtp.enabled
                                    ? resolveMTPMaxTargetQueryRows(active_mtp)
                                    : 1),
                        .prefill_graph_row_shapes =
                            plan_.runtime.overlay_prefill_schedule.bucket_rows,
                        .max_decode_activation_rows =
                            active_mtp.enabled
                                ? resolveMTPMaxTargetQueryRows(active_mtp)
                                : 1,
                        .mtp_enabled = active_mtp.enabled,
                        .max_mtp_draft_depth =
                            active_mtp.enabled
                                ? resolveMTPMaximumDraftDepth(active_mtp)
                                : 0,
                        .max_request_count =
                            std::max(1, plan_.runtime.batch_size),
                    });
                if (!runner_)
                {
                    return setError(
                        "Failed to create the rank-local MoE overlay "
                        "participant graph");
                }
                LOG_DEBUG(
                    "[OrchestrationRunner] Built rank-local expert-only "
                    "MoE overlay graph for rank "
                    << overlay_execution_plan->currentRankPlan().world_rank);
                return true;
            }
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

    bool OrchestrationRunner::
        materializeMoEOverlayContinuationServingGraphFamily()
    {
        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_
                                        : mpi_ctx_;
        const auto execution = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            overlay_context);
        if (!execution || !overlay_context ||
            overlay_context->world_size() <= 1 ||
            !execution->buildsRootGraph())
        {
            return true;
        }
        if (!runner_)
        {
            return setError(
                "Distributed ExpertOverlay continuation graph setup requires "
                "a built root inference runner");
        }

        const auto &schedule = plan_.runtime.overlay_prefill_schedule;
        ServingGraphFamilyMaterializationPlan family_plan{
            .prefill_bucket_rows = schedule.bucket_rows,
            .prefill_pad_token_id =
                debugEnv().execution.prefill_graph_pad_token_id,
        };
        if (!schedule.enabled() || !family_plan.valid() ||
            schedule.graph_row_capacity <= 0 ||
            family_plan.prefill_bucket_rows.back() >
                schedule.graph_row_capacity)
        {
            return setError(
                "Distributed ExpertOverlay continuation graph setup requires "
                "the frozen common prefill schedule produced by memory "
                "admission");
        }

        if (!runner_->materializeServingGraphFamilyWithoutLaunch(family_plan))
        {
            return setError(
                "Distributed ExpertOverlay continuation runner could not "
                "capture and instantiate its admitted serving graph family");
        }
        PerfStatsCollector::addCounter(
            "forward_graph",
            "overlay_continuation_serving_family_completions",
            1.0,
            "setup",
            "rank",
            {{"prefill_buckets",
              std::to_string(family_plan.prefill_bucket_rows.size())},
             {"graph_row_capacity",
              std::to_string(schedule.graph_row_capacity)},
             {"capacity_authority", "overlay_prefill_schedule"},
             {"ticket_authority_installed", "false"}});
        return true;
    }

    bool OrchestrationRunner::initializeMoEOverlayInferenceTransactions()
    {
        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_
                                        : mpi_ctx_;
        const auto execution = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan, overlay_context);
        if (!execution || !overlay_context ||
            overlay_context->world_size() <= 1)
        {
            return true;
        }
        if (!runner_ || !model_ctx_ ||
            !moe_expert_overlay_residency_authority_)
        {
            return setError(
                "Distributed ExpertOverlay inference transactions require "
                "a built runner, model context, and residency authority");
        }
        if (moe_overlay_inference_transaction_coordinator_ ||
            moe_overlay_inference_transaction_follower_)
        {
            return setError(
                "ExpertOverlay inference transaction authority is setup-only");
        }

        const auto snapshot =
            moe_expert_overlay_residency_authority_->snapshot();
        if (!snapshot || !snapshot->valid())
        {
            return setError(
                "Distributed ExpertOverlay inference transactions require a "
                "valid initial residency snapshot");
        }

        const MTPRuntimeConfig &active_mtp =
            plan_.runtime.mtp.enabled ? plan_.runtime.mtp : config_.mtp;
        const int max_mtp_draft_depth =
            active_mtp.enabled
                ? resolveMTPMaximumDraftDepth(active_mtp)
                : 0;
        const int max_decode_rows =
            active_mtp.enabled
                ? resolveMTPMaxTargetQueryRows(active_mtp)
                : 1;
        const int max_graph_rows = std::max(
            plan_.runtime.overlay_prefill_schedule.graph_row_capacity,
            max_decode_rows);
        const int max_request_count =
            std::max(1, plan_.runtime.batch_size);
        const auto graph_family =
            resolveMoEOverlayInferenceGraphFamilyIdentity(
                model_ctx_->concreteLoader(),
                model_ctx_->architecture(),
                model_ctx_->totalBlockCount(),
                active_mtp.enabled,
                /*graph_family_generation=*/1,
                max_graph_rows,
                max_decode_rows,
                max_request_count,
                max_mtp_draft_depth);
        const std::size_t slot_count =
            moeOverlayInferenceTransactionSlotCount(
                max_mtp_draft_depth);
        const int source_rank = execution->continuation_root_rank;
        const int local_rank = overlay_context->rank();
        if (source_rank < 0 ||
            source_rank >= overlay_context->world_size() ||
            slot_count == 0)
        {
            return setError(
                "Distributed ExpertOverlay inference resolved an invalid "
                "source rank or retained transaction capacity");
        }

        try
        {
            if (local_rank == source_rank)
            {
                std::vector<std::shared_ptr<
                    IMoEOverlayInferenceTransactionPublisher>> publishers;
                for (const auto &rank_plan : execution->rank_plans)
                {
                    if (rank_plan.world_rank == source_rank ||
                        !rank_plan.loads_expert_weights)
                    {
                        continue;
                    }
                    if (rank_plan.builds_root_graph)
                    {
                        return setError(
                            "A remote ExpertOverlay follower rank cannot also "
                            "own a dense continuation graph");
                    }

                    const auto topology =
                        makeMoEOverlayInferenceTopologyIdentity(
                            snapshot->owner_map,
                            graph_family,
                            source_rank,
                            rank_plan.world_rank);
                    auto channel = std::make_shared<
                        MoEOverlayMPIInferenceTransactionChannel>(
                        MoEOverlayMPIInferenceTransactionChannel::Config{
                            .mpi_ctx = overlay_context,
                            .source_world_rank = source_rank,
                            .target_world_rank = rank_plan.world_rank,
                            .send_slot_count = slot_count,
                        });
                    publishers.push_back(std::make_shared<
                        MoEOverlayInferenceTransactionPublisher>(
                        MoEOverlayInferenceTransactionPublisher::Config{
                            .channel = std::move(channel),
                            .protocol = {
                                .topology = topology,
                                .slot_count = slot_count,
                                .max_request_count = max_request_count,
                                .max_rows_per_request = max_graph_rows,
                                .max_mtp_draft_depth =
                                    max_mtp_draft_depth,
                            },
                        }));
                }
                if (publishers.empty())
                {
                    return setError(
                        "Distributed ExpertOverlay continuation rank found no "
                        "remote expert graph followers");
                }

                int continuation_participants = 1;
                if (const auto *rank_runner =
                        dynamic_cast<const RankOrchestrator *>(runner_.get()))
                {
                    continuation_participants = rank_runner->device_count();
                }
                const int logical_root_participant =
                    snapshot->placement_plan->continuation_domain_spec
                        .logical_root_participant;
                const auto *const ticket_authority =
                    snapshot->owner_map.participantForId(
                        logical_root_participant);
                if (!ticket_authority ||
                    ticket_authority->domain_name !=
                        execution->continuation_domain ||
                    !ticket_authority->world_rank_known ||
                    ticket_authority->world_rank != source_rank ||
                    ticket_authority->domain_participant_index < 0 ||
                    ticket_authority->domain_participant_index >=
                        continuation_participants)
                {
                    return setError(
                        "Distributed ExpertOverlay could not map the "
                        "planner-declared continuation root to one local "
                        "ticket-authority graph participant");
                }
                auto coordinator = std::make_shared<
                    MoEOverlayInferenceTransactionCoordinator>(
                    MoEOverlayInferenceTransactionCoordinator::Config{
                        .publishers = std::move(publishers),
                        .continuation_participant_count =
                            continuation_participants,
                        .ticket_authority_participant_index =
                            ticket_authority->domain_participant_index,
                        .max_transactions_per_command = slot_count,
                        .max_mtp_draft_depth = max_mtp_draft_depth,
                    });
                if (!runner_->setMoEOverlayInferenceTransactionCoordinator(
                        coordinator,
                        /*continuation_participant_index=*/0))
                {
                    return setError(
                        "Continuation runner rejected the rank-wide "
                        "ExpertOverlay inference transaction authority");
                }
                moe_overlay_inference_transaction_coordinator_ =
                    std::move(coordinator);
            }
            else
            {
                const auto &rank_plan = execution->currentRankPlan();
                if (rank_plan.builds_root_graph)
                {
                    return setError(
                        "Non-authority distributed ExpertOverlay continuation "
                        "graphs require an explicit multi-source transaction design");
                }
                if (!rank_plan.loads_expert_weights)
                {
                    return setError(
                        "Distributed ExpertOverlay worker rank owns neither a "
                        "continuation graph nor a retained expert graph");
                }
                auto *participant_runner =
                    dynamic_cast<MoEOverlayParticipantGraphRunner *>(
                        runner_.get());
                if (!participant_runner)
                {
                    return setError(
                        "Remote ExpertOverlay rank did not build the retained "
                        "participant graph runner required by its follower");
                }

                auto protocol =
                    participant_runner->inferenceTransactionProtocolConfig();
                if (protocol.topology.source_world_rank != source_rank ||
                    protocol.topology.target_world_rank != local_rank ||
                    protocol.slot_count != slot_count ||
                    protocol.max_request_count != max_request_count ||
                    protocol.max_rows_per_request != max_graph_rows ||
                    protocol.max_mtp_draft_depth != max_mtp_draft_depth)
                {
                    return setError(
                        "Remote ExpertOverlay retained graph family disagrees "
                        "with orchestration transaction geometry");
                }
                auto channel = std::make_shared<
                    MoEOverlayMPIInferenceTransactionChannel>(
                    MoEOverlayMPIInferenceTransactionChannel::Config{
                        .mpi_ctx = overlay_context,
                        .source_world_rank = source_rank,
                        .target_world_rank = local_rank,
                        .send_slot_count = slot_count,
                    });
                moe_overlay_inference_transaction_follower_ =
                    std::make_unique<
                        MoEOverlayInferenceTransactionFollower>(
                        MoEOverlayInferenceTransactionFollower::Config{
                            .channel = std::move(channel),
                            .executor = participant_runner,
                            .protocol = std::move(protocol),
                        });
            }
        }
        catch (const std::exception &error)
        {
            return setError(
                std::string(
                    "Failed to initialize ExpertOverlay inference "
                    "transactions: ") +
                error.what());
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_transaction",
            "control_plane_initializations",
            1.0,
            "graph_setup",
            {},
            {{"role", local_rank == source_rank
                          ? "continuation_authority"
                          : "remote_follower"},
             {"source_rank", std::to_string(source_rank)},
             {"local_rank", std::to_string(local_rank)},
             {"slots", std::to_string(slot_count)},
             {"max_mtp_depth", std::to_string(max_mtp_draft_depth)}});
        return true;
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

    const IModelContext *OrchestrationRunner::modelContextForDiagnostics() const
    {
        return model_ctx_.get();
    }

    std::shared_ptr<const MoEOverlayResidencySnapshot>
    OrchestrationRunner::expertOverlayResidencySnapshotForDiagnostics() const
    {
        return moe_expert_overlay_residency_authority_
                   ? moe_expert_overlay_residency_authority_->snapshot()
                   : nullptr;
    }

    std::optional<ModelContextReuseContract>
    OrchestrationRunner::modelContextReuseContract() const
    {
        /*
         * A reuse contract is proof emitted by a successfully initialized
         * production runner, not a generic ModelContext accessor.  The current
         * single-authority contract deliberately excludes local/global parallel
         * and routed-placement plans; those require a typed set of per-device or
         * per-rank model authorities rather than one shared pointer.
         */
        if (!initialized_ || !model_ctx_ || plan_.usesLocalTP() ||
            plan_.usesLocalPP() || plan_.usesPipelineParallel() ||
            plan_.usesGlobalTP() || config_.moe_routed_expert_plan)
        {
            return std::nullopt;
        }

        const auto weight_manager = model_ctx_->concreteWeightManager();
        if (!weight_manager)
            return std::nullopt;
        const auto &gates = weight_manager->lifecycleGates();
        const auto store = weight_manager->preparedWeightStoreIfInitialized();
        const DeviceId device = plan_.primary_device.toLocalDeviceId();
        if (!gates.device_preparation_complete ||
            !gates.graph_materialization_complete || !store ||
            store->sizeForDevice(device) == 0u)
        {
            return std::nullopt;
        }

        /*
         * Sharing extends only model-owned lifetime. All mutable execution state
         * remains below runner_; shutdown retires it before the consumer starts.
         * The copied plan is the certificate the consumer must match.
         */
        return ModelContextReuseContract{
            .context = model_ctx_,
            .prepared_weight_plan = plan_,
        };
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
        mdo_config.moe_routed_expert_plan = config_.moe_routed_expert_plan;
        mdo_config.moe_expert_overlay_residency_authority =
            moe_expert_overlay_residency_authority_;
        mdo_config.moe_expert_overlay_participant_residency =
            moe_expert_overlay_participant_residency_;
        mdo_config.moe_expert_overlay_decode_histogram =
            moe_expert_overlay_decode_histogram_;
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
        mdo_config.moe_routed_expert_plan = config_.moe_routed_expert_plan;
        mdo_config.moe_expert_overlay_residency_authority =
            moe_expert_overlay_residency_authority_;
        mdo_config.moe_expert_overlay_participant_residency =
            moe_expert_overlay_participant_residency_;
        mdo_config.moe_expert_overlay_decode_histogram =
            moe_expert_overlay_decode_histogram_;
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
            LOG_DEBUG("[OrchestrationRunner]   Backend: " << cpuBackendDescription());
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
        runner_config.moe_routed_expert_plan = config_.moe_routed_expert_plan;
        runner_config.moe_expert_overlay_residency_authority =
            moe_expert_overlay_residency_authority_;
        runner_config.moe_expert_overlay_participant_residency =
            moe_expert_overlay_participant_residency_;
        runner_config.moe_expert_overlay_decode_histogram =
            moe_expert_overlay_decode_histogram_;
        runner_config.moe_expert_overlay_mpi_ctx = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;

        LOG_DEBUG("[OrchestrationRunner] Single-device precision config: activation="
                  << activationPrecisionToString(runner_config.activation_precision)
                  << ", kv_cache=" << kvCachePrecisionToString(runner_config.kv_cache_precision));

        // Create runner via factory (returns IInferenceRunner)
        if (model_ctx_)
        {
            std::shared_ptr<IMPIContext> graph_mpi_ctx = mpi_ctx_;
            const auto overlay_execution =
                resolveOverlayExecutionPlanForRunner(
                    config_.moe_routed_expert_plan,
                    moe_expert_overlay_mpi_ctx_
                        ? moe_expert_overlay_mpi_ctx_
                        : mpi_ctx_);
            if (overlay_execution &&
                overlay_execution->buildsRootGraph() &&
                overlay_execution->world_size > 1 &&
                !plan_.usesGlobalTP())
            {
                /*
                 * A non-TP dense continuation graph is participant-local. Its
                 * only cross-rank edges are the explicitly configured sparse
                 * MoE stages, which retain the overlay world in runner_config.
                 * A NodeTP continuation is different: every rank builds
                 * one dense shard and ordinary graph construction must retain
                 * that exact global TP communicator.
                 */
                graph_mpi_ctx = MPIContextFactory::self();
                PerfStatsCollector::addCounter(
                    "moe_overlay",
                    "root_graph_rank_local_mpi_scope",
                    1.0,
                    "graph_build",
                    {},
                    {{"overlay_world_size",
                      std::to_string(overlay_execution->world_size)}});
            }
            runner_ = createInferenceRunner(
                model_ctx_,
                std::move(graph_mpi_ctx),
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
        snapshot_combined_cache_.clear();
        if (runner_)
        {
            runner_->enableSnapshotCapture(output_dir);
        }
    }

    void OrchestrationRunner::setSnapshotCaptureFilter(const std::vector<std::string> &keys)
    {
        if (runner_)
        {
            runner_->setSnapshotCaptureFilter(keys);
        }
    }

    void OrchestrationRunner::disableSnapshotCapture()
    {
        snapshot_combined_cache_.clear();
        if (runner_)
        {
            runner_->disableSnapshotCapture();
        }
    }

    void OrchestrationRunner::clearSnapshots()
    {
        snapshot_combined_cache_.clear();
        if (runner_)
        {
            runner_->clearSnapshots();
        }
    }

    const float *OrchestrationRunner::getSnapshot(const std::string &key, size_t &out_size) const
    {
        if (runner_)
        {
            if (const auto *rank = dynamic_cast<const RankOrchestrator *>(runner_.get()))
            {
                TPSnapshot tp_snapshot = rank->getTPSnapshot(key);
                if (tp_snapshot.mode == SnapshotShardingMode::UNKNOWN &&
                    tp_snapshot.tp_degree > 1 &&
                    !tp_snapshot.device_data.empty())
                {
                    LOG_ERROR("[OrchestrationRunner] Refusing to return TP snapshot '" << key
                                                                                      << "' because its sharding mode is UNKNOWN. "
                                                                                         "Add the snapshot key to the model schema or a runtime "
                                                                                         "sharding override before comparing multi-device output.");
                    out_size = 0;
                    return nullptr;
                }

                size_t combined_size = 0;
                const float *combined = tp_snapshot.getCombinedData(combined_size);
                if (combined && combined_size > 0)
                {
                    auto &cache = snapshot_combined_cache_[key];
                    cache.assign(combined, combined + combined_size);
                    out_size = cache.size();
                    return cache.data();
                }
                if (tp_snapshot.tp_degree > 1 && !tp_snapshot.device_data.empty())
                {
                    LOG_ERROR("[OrchestrationRunner] Failed to combine TP snapshot '" << key
                                                                                    << "' with mode "
                                                                                    << shardingModeToString(tp_snapshot.mode)
                                                                                    << ". Refusing to fall back to a single participant view.");
                    out_size = 0;
                    return nullptr;
                }
            }
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

    void OrchestrationRunner::resetUnderlyingRunnerRequestState(const char *reason)
    {
        if (!runner_)
            return;

        LOG_DEBUG("[OrchestrationRunner] Resetting underlying runner request-owned inference state"
                  << (reason && reason[0] ? std::string(" for ") + reason : std::string{}));
        runner_->resetInferenceState(
            InferenceStateResetRequest::requestBoundary(reason));
    }

    /**
     * @brief Publish the one root-authoritative sparse-collective generation for this request.
     *
     * The rank-local continuation graph and remote expert graph are allowed to
     * have different capture and cache lifetimes. They are not allowed to
     * derive different wire identities. This narrow lifecycle call is made by
     * every overlay rank at synchronized initialization and request-reset
     * boundaries; no hot-path MPI broadcast or stage-local fallback is used.
     */
    bool OrchestrationRunner::publishMoEOverlayCollectiveRequestGeneration(
        const char *reason)
    {
        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        const auto overlay_execution = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            overlay_context);
        if (!overlay_execution || !overlay_context ||
            overlay_context->world_size() <= 1)
        {
            return true;
        }
        if (!runner_)
        {
            return setError(
                "Distributed MoE ExpertOverlay has no runner for collective "
                "generation publication");
        }
        const int authority_rank = overlay_execution->continuation_root_rank;
        if (authority_rank < 0 || authority_rank >= overlay_context->world_size())
        {
            return setError(
                "Distributed MoE ExpertOverlay resolved an invalid continuation "
                "rank for collective-generation publication");
        }

        /*
         * The continuation rank is the sole authority for this value.  The
         * other ranks retain only the exact generation it publishes; they do
         * not advance a local counter and hope that independent request/cache
         * lifecycles happened to remain aligned.
         *
         * IMPIContext exposes a typed int32 broadcast.  Two words preserve the
         * complete uint64_t identity instead of truncating a long-running
         * serving process's request epoch.  Both ranks execute this only at a
         * synchronized initialization or request-reset boundary, never from
         * the captured inference hot path.
         */
        std::array<int32_t, 2> generation_wire{};
        if (overlay_context->rank() == authority_rank)
        {
            if (request_epoch_ == std::numeric_limits<uint64_t>::max())
            {
                return setError(
                    "Distributed MoE ExpertOverlay request-generation counter "
                    "overflowed");
            }

            const uint64_t authority_generation = request_epoch_ + 1;
            generation_wire[0] = static_cast<int32_t>(
                static_cast<uint32_t>(authority_generation));
            generation_wire[1] = static_cast<int32_t>(
                static_cast<uint32_t>(authority_generation >> 32));
        }
        overlay_context->broadcast_int32(
            generation_wire.data(), generation_wire.size(), authority_rank);
        const uint64_t generation_id =
            static_cast<uint64_t>(static_cast<uint32_t>(generation_wire[0])) |
            (static_cast<uint64_t>(static_cast<uint32_t>(generation_wire[1])) << 32);
        if (generation_id == 0)
        {
            return setError(
                "Distributed MoE ExpertOverlay continuation rank published "
                "an invalid zero collective generation");
        }
        // Keep a received mirror for lifecycle diagnostics only.  The source
        // of truth remains the continuation rank and is re-published at every
        // request boundary.
        request_epoch_ = generation_id - 1;
        moe_overlay_collective_generation_id_ = generation_id;
        if (!runner_->setMoEOverlayCollectiveRequestGeneration(generation_id))
        {
            return setError(
                "Distributed MoE ExpertOverlay runner rejected root-authoritative "
                "collective request generation");
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "moe_overlay_collective_request_generation",
            static_cast<double>(generation_id),
            "request_lifecycle",
            {},
            {{"authority", "continuation_root_rank"},
             {"authority_rank", std::to_string(authority_rank)},
             {"reason", reason && reason[0] ? reason : "unspecified"},
             {"generation", std::to_string(generation_id)},
             {"immutable_until_request_reset", "true"}});
        return true;
    }

    /**
     * @brief Advance the distributed request generation after a complete reset boundary.
     */
    bool OrchestrationRunner::advanceMoEOverlayCollectiveRequestGeneration(
        const char *reason)
    {
        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        const auto overlay_execution = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            overlay_context);
        const bool distributed_overlay =
            overlay_execution && overlay_context &&
            overlay_context->world_size() > 1;
        const bool is_authority_rank =
            distributed_overlay &&
            overlay_context->rank() == overlay_execution->continuation_root_rank;

        if ((!distributed_overlay || is_authority_rank) &&
            request_epoch_ >= std::numeric_limits<uint64_t>::max() - 1)
        {
            return setError(
                "Distributed MoE ExpertOverlay request-generation counter "
                "overflowed before request reset");
        }

        if (!distributed_overlay || is_authority_rank)
        {
            // Only the continuation rank advances the source counter. Remote
            // ranks receive the resulting generation in publish... below.
            ++request_epoch_;
        }
        return publishMoEOverlayCollectiveRequestGeneration(reason);
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

    bool OrchestrationRunner::waitForLastForwardCompletionForBenchmark()
    {
        if (!runner_)
        {
            LOG_ERROR("[OrchestrationRunner] Benchmark completion boundary has no inference runner");
            return false;
        }
        return runner_->waitForLastForwardCompletionForBenchmark();
    }

    void OrchestrationRunner::setSkipLogitsGatherDecode(bool skip)
    {
        // Broadcast to worker ranks
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SKIP_LOGITS_DECODE);
            int32_t val = skip ? 1 : 0;
            mpi_ctx_->broadcast_int32(
                &val, 1, mpi_coordinated_root_rank_);
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
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SET_SAMPLING);
            float params_buf[6] = {
                params.temperature,
                params.top_p,
                static_cast<float>(params.top_k),
                static_cast<float>(params.seed),
                params.presence_penalty,
                params.frequency_penalty};
            mpi_ctx_->broadcast(
                params_buf, 6, mpi_coordinated_root_rank_);
        }

        active_sampling_params_ = params;
        // Reset token history and deterministic RNG for a new conversation/request.
        sampler_ = Sampler(params.seed);
        if (runner_ &&
            !runner_->configureMTPRequestPenaltyPolicy(
                MTPRequestPenaltyPolicy{
                    .presence_penalty = params.presence_penalty,
                    .frequency_penalty = params.frequency_penalty,
                }))
        {
            throw std::runtime_error(
                "Inference runner rejected request penalty policy");
        }
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
        mpi_ctx_->broadcast_int32(
            &tag, 1, mpi_coordinated_root_rank_);
    }

    void OrchestrationRunner::shutdownMPIWorkers()
    {
        if (!mpi_ctx_ || mpi_ctx_->world_size() <= 1)
            return;

        if (mpi_ctx_->rank() != mpi_coordinated_root_rank_)
        {
            LLAMINAR_UNREACHABLE(
                "Only the coordinated root may close the MPI worker loop");
        }

        // The first shutdown releases worker ranks from their blocking command
        // receive.  Re-broadcasting after that point has no receiver and would
        // deadlock, so shutdown is deliberately idempotent at the protocol
        // boundary rather than relying on each caller to remember cleanup.
        if (mpi_workers_shutdown_)
            return;

        LOG_DEBUG("[MPI] Coordinated root rank "
                  << mpi_coordinated_root_rank_
                  << " sending SHUTDOWN to worker ranks");
        broadcastCommand(MPICommand::SHUTDOWN);
        mpi_workers_shutdown_ = true;

        /*
         * SHUTDOWN makes every worker drain the same rank-local asynchronous
         * residency service before leaving its command loop. Drain the root
         * concurrently after publishing that command so an already-started
         * distributed vote or migration retains all participants until it
         * reaches a terminal state. Letting runner destructors stop services
         * independently can strand a later rank inside a collective.
         */
        shutdownMoEExpertOverlayResidencyMaintenance();
    }

    [[noreturn]] void OrchestrationRunner::terminateFailedMPIWorkerCommand(
        const char *command,
        const std::string &reason)
    {
        const int rank = mpi_ctx_ ? mpi_ctx_->rank() : -1;
        const std::string diagnostic =
            "[MPIWorkerLoop] rank " + std::to_string(rank) +
            " failed coordinated " + command + " command: " +
            (reason.empty() ? "no failure detail was recorded" : reason) +
            "; aborting because further collective order is indeterminate";
        LOG_ERROR(diagnostic);

        // A real worker cannot safely receive another root command after a
        // graph-stage failure: a peer may already be waiting in a sparse MoE
        // all-gather or a TP collective that this rank will never enter.
        // MPI_Abort terminates every participant in this communicator promptly.
        if (mpi_ctx_)
        {
            const MPI_Comm communicator = mpi_ctx_->communicator();
            if (communicator != MPI_COMM_NULL)
            {
                const int abort_status = MPI_Abort(communicator, EXIT_FAILURE);
                LOG_ERROR("[MPIWorkerLoop] MPI_Abort unexpectedly returned with status "
                          << abort_status << "; terminating this worker directly");
                std::abort();
            }
        }

        // Scripted contexts intentionally have no live communicator. Throwing
        // preserves the same non-returning contract while making the invariant
        // observable in a device-free unit test.
        throw std::runtime_error(diagnostic);
    }

    void OrchestrationRunner::runMPIWorkerLoop()
    {
        if (!mpi_ctx_ || mpi_ctx_->rank() == mpi_coordinated_root_rank_)
        {
            LOG_WARN("[MPIWorkerLoop] Should only be called on worker ranks");
            return;
        }

        LOG_DEBUG("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
                                          << " entering worker loop");

        const auto notifyOverlayMaintenance =
            [this](const char *command)
        {
            /*
             * The coordinated root calls maybeApplyMoERebalance() at the same
             * committed request boundaries. ExpertOverlay maintenance is
             * process-local, so a worker must wake its own service too; the
             * root notification cannot cross this ownership boundary. Keep
             * legacy/device controllers under their existing APPLY command
             * protocol—only the asynchronous overlay service is wake-only.
             */
            if (!moe_expert_overlay_maintenance_service_)
                return;
            if (!maybeApplyMoERebalance())
            {
                terminateFailedMPIWorkerCommand(
                    command,
                    lastError().empty()
                        ? "ExpertOverlay maintenance notification failed"
                        : lastError());
            }
        };

        while (true)
        {
            // Wait for a command from the coordinated root.
            int32_t tag = 0;
            mpi_ctx_->broadcast_int32(
                &tag, 1, mpi_coordinated_root_rank_);
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
                float params_buf[6]; // temperature, top_p, top_k, seed, penalties
                mpi_ctx_->broadcast(
                    params_buf, 6, mpi_coordinated_root_rank_);
                SamplingParams sp;
                sp.temperature = params_buf[0];
                sp.top_p = params_buf[1];
                sp.top_k = static_cast<int>(params_buf[2]);
                sp.seed = static_cast<uint64_t>(params_buf[3]);
                sp.presence_penalty = params_buf[4];
                sp.frequency_penalty = params_buf[5];
                setSamplingParams(sp);
                break;
            }

            case MPICommand::SET_STOP_TOKENS:
            {
                int32_t stop_token_count = 0;
                mpi_ctx_->broadcast_int32(
                    &stop_token_count, 1, mpi_coordinated_root_rank_);
                if (stop_token_count < 0)
                {
                    terminateFailedMPIWorkerCommand(
                        "SET_STOP_TOKENS",
                        "received a negative request stop-token count");
                }

                std::vector<int32_t> stop_tokens(
                    static_cast<size_t>(stop_token_count));
                if (stop_token_count > 0)
                {
                    mpi_ctx_->broadcast_int32(
                        stop_tokens.data(),
                        static_cast<size_t>(stop_token_count),
                        mpi_coordinated_root_rank_);
                }
                setStopTokens(stop_tokens);
                break;
            }

            case MPICommand::PREFILL:
            {
                // Receive token count then tokens
                int32_t n_tokens = 0;
                mpi_ctx_->broadcast_int32(
                    &n_tokens, 1, mpi_coordinated_root_rank_);

                if (n_tokens <= 0)
                {
                    terminateFailedMPIWorkerCommand(
                        "PREFILL",
                        "received a non-positive prompt token count");
                }

                std::vector<int32_t> tokens(n_tokens);
                mpi_ctx_->broadcast_int32(
                    tokens.data(),
                    static_cast<size_t>(n_tokens),
                    mpi_coordinated_root_rank_);

                if (moe_overlay_inference_transaction_follower_)
                {
                    /*
                     * A remote ExpertOverlay rank has no token embedding, KV,
                     * logits, or sampler authority during prefill either. The
                     * root graph publishes one authenticated ticket per
                     * retained prefill segment and then a terminal ticket for
                     * this outer command. Receiving the prompt above preserves
                     * the public command wire shape; those token bytes must not
                     * cause the follower to enter an independent model runner.
                     */
                    const auto follower_result =
                        moe_overlay_inference_transaction_follower_
                            ->runOneCommand();
                    if (!follower_result.ok)
                    {
                        terminateFailedMPIWorkerCommand(
                            "PREFILL",
                            follower_result.error.empty()
                                ? "ExpertOverlay prefill transaction follower failed"
                                : follower_result.error);
                    }
                    PerfStatsCollector::addCounter(
                        "moe_overlay_transaction",
                        "follower_prefill_commands",
                        1.0,
                        "prefill",
                        {},
                        {{"transactions",
                          std::to_string(
                              follower_result.executed_transactions)},
                         {"token_state_mutated", "false"}});
                    notifyOverlayMaintenance(
                        "PREFILL transaction-follower maintenance boundary");
                    break;
                }

                if (!prefill(tokens))
                    terminateFailedMPIWorkerCommand("PREFILL", lastError());
                notifyOverlayMaintenance("PREFILL maintenance boundary");
                break;
            }

            case MPICommand::DECODE_STEP:
            {
                int32_t token_budget = 0;
                mpi_ctx_->broadcast_int32(
                    &token_budget, 1, mpi_coordinated_root_rank_);
                if (moe_overlay_inference_transaction_follower_)
                {
                    /*
                     * A heterogeneous expert rank owns neither continuation KV
                     * nor sampling state. It follows the root's authenticated
                     * graph tickets until Complete instead of masquerading as a
                     * second decoder and entering unrelated token collectives.
                     */
                    const auto follower_result =
                        moe_overlay_inference_transaction_follower_
                            ->runOneCommand();
                    if (!follower_result.ok)
                    {
                        terminateFailedMPIWorkerCommand(
                            "DECODE_STEP",
                            follower_result.error.empty()
                                ? "ExpertOverlay transaction follower failed"
                                : follower_result.error);
                    }
                    PerfStatsCollector::addCounter(
                        "moe_overlay_transaction",
                        "follower_decode_commands",
                        1.0,
                        "decode",
                        {},
                        {{"transactions",
                          std::to_string(
                              follower_result.executed_transactions)},
                         {"token_state_mutated", "false"}});
                    notifyOverlayMaintenance(
                        "DECODE_STEP transaction-follower maintenance boundary");
                    break;
                }
                setDecodeStepTokenBudget(token_budget);
                GenerationResult decode_result = decodeStep();
                // Rank 0 scopes thinking-budget decode through ChatCompletionHandler;
                // reset workers too so forced-token or later control commands never
                // observe a stale request-local budget.
                setDecodeStepTokenBudget(0);
                if (!decode_result.success())
                    terminateFailedMPIWorkerCommand("DECODE_STEP", decode_result.error);
                notifyOverlayMaintenance("DECODE_STEP maintenance boundary");
                break;
            }

            case MPICommand::FORCE_DECODE_TOKEN:
            {
                int32_t forced = 0;
                mpi_ctx_->broadcast_int32(
                    &forced, 1, mpi_coordinated_root_rank_);
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[MPIWorkerLoop] rank " << mpi_ctx_->rank()
                                                     << " received forced token "
                                                     << forced);
                }
                if (moe_overlay_inference_transaction_follower_)
                {
                    /*
                     * The forced token is continuation policy state. A remote
                     * expert rank consumes only the authenticated retained
                     * graph (if this command advances model state) followed by
                     * its terminal ticket. It must still join the ordinary MPI
                     * command fence after the follower returns so the next root
                     * command cannot overtake this control boundary.
                     */
                    ScopedMPICoordinatedCommandFence command_fence(
                        mpi_ctx_.get(),
                        /*active=*/true,
                        "forceDecodeTokenFollower");
                    const auto follower_result =
                        moe_overlay_inference_transaction_follower_
                            ->runOneCommand();
                    if (!follower_result.ok)
                    {
                        terminateFailedMPIWorkerCommand(
                            "FORCE_DECODE_TOKEN",
                            follower_result.error.empty()
                                ? "ExpertOverlay forced-token transaction follower failed"
                                : follower_result.error);
                    }
                    PerfStatsCollector::addCounter(
                        "moe_overlay_transaction",
                        "follower_forced_token_commands",
                        1.0,
                        "decode",
                        {},
                        {{"transactions",
                          std::to_string(
                              follower_result.executed_transactions)},
                         {"token_state_mutated", "false"}});
                    notifyOverlayMaintenance(
                        "FORCE_DECODE_TOKEN transaction-follower maintenance boundary");
                    break;
                }
                GenerationResult forced_result = forceDecodeToken(forced);
                if (!forced_result.success())
                    terminateFailedMPIWorkerCommand(
                        "FORCE_DECODE_TOKEN",
                        forced_result.error);
                notifyOverlayMaintenance(
                    "FORCE_DECODE_TOKEN maintenance boundary");
                break;
            }

            case MPICommand::SKIP_LOGITS_DECODE:
            {
                int32_t skip = 0;
                mpi_ctx_->broadcast_int32(
                    &skip, 1, mpi_coordinated_root_rank_);
                runner_->setSkipLogitsGatherDecode(skip != 0);
                break;
            }

            case MPICommand::SHUTDOWN:
            {
                LOG_DEBUG("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
                                                  << " received SHUTDOWN");
                shutdownMoEExpertOverlayResidencyMaintenance();
                return;
            }

            default:
                terminateFailedMPIWorkerCommand(
                    "unknown",
                    "received unsupported command tag " + std::to_string(tag));
            }
        }
    }

} // namespace llaminar2
