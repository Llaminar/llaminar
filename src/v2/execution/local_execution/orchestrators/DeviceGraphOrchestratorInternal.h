/**
 * @file DeviceGraphOrchestratorInternal.h
 * @brief Shared internal helpers and nested scratch definitions for DeviceGraphOrchestrator shards.
 */
#pragma once
/**
 * @file DeviceGraphOrchestrator.cpp
 * @brief Participant-local production graph orchestration and lifecycle.
 *
 * This file builds, captures, caches, and submits one device participant's
 * declarative model graphs. It owns the typed setup and request transitions
 * for prefill, serial decode, MTP, prefix restore, ExpertOverlay epochs, and
 * diagnostic checkpoint publication. Cross-participant coordination remains
 * outside ordinary compute stages; every GPU operation here is bound to an
 * explicit stream and ordered through captured graph or event dependencies.
 * Pipeline composition is owned by RankOrchestrator/GlobalOrchestrator. This
 * participant owns only its declared layer shard and injected collectives;
 * restored-prefix Off requests retain the ordinary decode role and canonical
 * logits authority, independently of any physically retained MTP capacity.
 * it cannot manufacture another pipeline or lazily discover TP domains.
 * CPU accepted-state publication copies the selected verifier logits row into
 * canonical storage alongside KV/recurrent/terminal-hidden publication; a
 * retained multirow verifier allocation never implicitly owns scalar row zero.
 * Arena registration consumes the same resolved physical-root inventory as
 * admission, including explicit phase-split and retained MTP mailboxes.
 */

#include "execution/qwen4exp/PooledImmutableParticipantExecutionPolicy.h"
#include "../../../utils/OpenMPUtils.h"
#include "execution/qwen4exp/PooledImmutableLocalHotTransportGeometry.h"
#include "execution/local_execution/graph/RetainedHostServiceDeclarations.h"
#include "execution/diagnostics/QueryKVGeometryObservation.h"
#include "DeviceGraphOrchestrator.h"
#include "execution/qwen4exp/QwenSparseQueryProgramObservation.h"
#include "loaders/ModelContext.h"
#include "MTPGraphPolicyScopes.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "../../../backends/BackendManager.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../collective/BackendRouter.h" // GlobalBackendRouter for PP copy
#include "../../../collective/CollectiveTimeoutPolicy.h"
#include "../../../collective/ILocalTPContext.h" // createLocalTPContext()
#include "../../../config/TensorParallelConfig.h"
#include "../../config/KVHeadConsumerGeometry.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../kernels/HybridKVCacheConfig.h"
#include "../../../kernels/IHybridKVCache.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../kernels/cpu/sampling/CPUSamplerPrimitives.h"
#include "../../../loaders/IWeightManager.h"
#include "../../../loaders/IWeightPlacementMap.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../loaders/WeightManager.h"
#include "../../../loaders/WeightPlacementMap.h"
#include "../../../loaders/WeightPlan.h"
#include "../../../models/qwen35moe/Qwen35MoEGraph.h"
#include "../../../planning/ActivationBufferSizing.h"
#include "../../../planning/PhysicalMemoryAuthority.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../tensors/TensorFactory.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/MTPDepthTransactionTracePerfStats.h"
#include "../../../utils/Logger.h"
#include "../../../utils/MPIContext.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/Sampler.h" // LogitPenalty
#include "../../../utils/VramBillOfMaterials.h"
#include "../../compute_stages/stages/AllGatherStage.h"
#include "../../compute_stages/stages/MoEDeviceRebalanceStage.h"
#include "../../compute_stages/stages/MoEExpertComputeStage.h"
#include "../../compute_stages/stages/MoEOverlayTicketPublishStage.h"
#include "../../compute_stages/stages/MoERoutingStage.h"
#include "../../compute_stages/stages/RoPEStage.h"
#include "../../config/HybridPrecisionConfig.h"
#include "../../config/InferenceMode.h"
#include "../../moe/ExpertWeightPayloadProvider.h"
#include "../../moe/ExpertWeightTransfer.h"
#include "../../moe/GpuExpertTransferStagingPool.h"
#include "../../moe/MoEGroupedVerifierHistogramBoundarySet.h"
#include "../../moe/MoEOverlayInferenceTransactionService.h"
#include "../../moe/MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "../../moe/MoEOverlayResidencyAuthority.h"
#include "../../moe/MoEOverlayParticipantResidency.h"
#include "../../moe/MoEOverlayRetainedParentComposer.h"
#include "../../moe/MoERebalanceController.h"
#include "../../moe/RoutedExpertOwnerAssignment.h"
#include "../../mtp/MTPCheckpointPolicy.h"
#include "../../mtp/MTPDeviceGenerationPolicy.h"
#include "../../mtp/MTPServingForwardCaptureGeometry.h"
#include "../../mtp/MTPSpecKVPublisher.h"
#include "../../mtp/MTPSpecStatePublisher.h"
#include "../../mtp/MTPVerifierPolicy.h"
#include "../../prefix_cache/PrefixArchiveCheckpointReadiness.h"
#include "../../prefix_cache/PrefixArchivePublicationScope.h"
#include "../../prefix_cache/PrefixAuxiliaryStateIO.h"
#include "execution/compute_stages/stages/NativeVocabularyAllGatherStage.h"
#include "../../prefix_cache/ResidualRowGeometry.h"
#include "../engine/ForwardGraphEntryOrdering.h"
#include "../engine/PrefillBucketUtils.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include "../graph/DeviceExecutionTimeline.h"
#include "../graph/DeviceGraphCaptureController.h"
#include "../graph/GraphCaptureGuard.h"
#include "../graph/RequestInputPositionAdmission.h"
#include "../graph/SchemaFactoryRegistry.h"
#include "memory/GraphArenaStorageLayout.h"
#include "MTPSidecarStreamBinding.h"
#include "PipelineForwardGraphEdges.h"
#include "PipelineMaintenanceAdmission.h"
#include "MoEMaintenanceObservationContext.h"
#include "execution/moe/DeviceMoERebalancePerfStats.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "execution/prefix_cache/BlockHash.h"
#include "execution/prefix_cache/DeviceHotPrefixStorageBackend.h"
#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixCacheFingerprint.h"
#include "execution/prefix_cache/PrefixCacheStateProbe.h"
#include "execution/prefix_cache/PrefixStateCache.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "kernels/qwen4exp/QwenAdaptiveTerminalCounters.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTerminal.h"
#include "kernels/qwen4exp/QwenFixedHeadTerminalCounters.h"
#include "kernels/qwen4exp/QwenSparseTerminalCounters.h"
#include "models/qwen4exp/Qwen4ExpConfig.h"
#include "transfer/MappedTransferProgressEpoch.h"
#include "transfer/TransferEngine.h"
#include "utils/Sha256.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2
{
    using dgo_detail::ScopedGroupedMTPVerifierGraphPolicy;
    using dgo_detail::ScopedLiveMTPRequestBatchConditionPolicy;


    namespace dgo_internal
    {
        bool appendDeviceMoERebalanceTraceJsonl(
            const std::string &path,
            const std::string &device_key,
            const PerfStatsCollector::Tags &maintenance_tags,
            const std::string &workspace_name,
            const MoEDeviceRebalanceStage::Params &params,
            const DeviceMoERebalanceStatus &status,
            const DeviceMoERebalanceGraphControllerState &controller_state,
            const DeviceMoERebalanceApplyStatus &copy_status,
            bool valid_copy_status,
            const DeviceMoERebalanceApplyStatus &apply_status,
            const std::vector<DeviceMoERebalanceCommandBufferHeader> &command_headers,
            const std::vector<DeviceMoERebalanceWaveState> &wave_states,
            const std::vector<uint32_t> &plan_counts,
            const std::vector<DeviceMoERebalancePlanEntry> &plan_entries,
            uint32_t plan_capacity,
            const DeviceMoERebalanceTransferCostEstimate &transfer_cost,
            uint32_t transfer_copied_arrivals,
            uint32_t transfer_applied_arrivals,
            uint32_t transfer_payload_bucket_slots,
            uint64_t transfer_payload_edge_mask,
            const std::vector<uint64_t> &trace_gathered_histograms,
            uint32_t trace_histogram_layer_count,
            const std::vector<int32_t> &trace_owner_participants,
            const DeviceMoERuntimeTable *trace_runtime_table)
;
    } // namespace dgo_internal
    using dgo_internal::appendDeviceMoERebalanceTraceJsonl;

    namespace
    {
        /**
         * @brief Persistent backend event closing one calibration sample.
         *
         * The object is allocated during graph setup and reused serially. The
         * inference thread only records the event on the exact terminal stream;
         * the maintenance thread only queries it. No stream synchronization,
         * host callback, or per-transaction allocation is introduced.
         */
        class BackendMoEOverlayInferenceCompletionEvent final
            : public IMoEOverlayInferenceCompletionEvent
        {
        public:
            /**
             * @brief Allocate the event once on its owning backend device.
             * @param backend Stable model-lifetime backend authority.
             * @param device Exact GPU event owner.
             * @throws std::invalid_argument when ownership is invalid.
             * @throws std::runtime_error when event allocation fails.
             */
            BackendMoEOverlayInferenceCompletionEvent(
                IBackend *backend,
                DeviceId device)
                : backend_(backend),
                  device_ordinal_(device.gpu_ordinal()),
                  device_name_(device.toString())
            {
                if (!backend_ || !device.is_gpu() || device_ordinal_ < 0)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay calibration fence requires one GPU backend device");
                }
                void *const raw_event =
                    backend_->createEvent(device_ordinal_);
                if (!raw_event)
                {
                    throw std::runtime_error(
                        "ExpertOverlay calibration could not allocate its persistent device event on " +
                        device_name_);
                }
                event_.reset(
                    raw_event,
                    [backend = backend_, ordinal = device_ordinal_](
                        void *event)
                    {
                        if (event)
                            backend->destroyEvent(event, ordinal);
                    });
            }

            /** @copydoc IMoEOverlayInferenceCompletionEvent::record */
            bool record(
                void *producer_stream,
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                EventState observed = state_.load(std::memory_order_acquire);
                while (observed != EventState::InFlight &&
                       !state_.compare_exchange_weak(
                           observed,
                           EventState::InFlight,
                           std::memory_order_acq_rel,
                           std::memory_order_acquire))
                {
                }
                if (!producer_stream || !event_ ||
                    observed == EventState::InFlight)
                {
                    if (observed != EventState::InFlight)
                        state_.store(EventState::Idle, std::memory_order_release);
                    if (error)
                    {
                        *error =
                            "ExpertOverlay calibration event is null, missing its exact stream, or already in flight on " +
                            device_name_;
                    }
                    return false;
                }
                if (!backend_->recordEvent(
                        event_.get(), device_ordinal_, producer_stream))
                {
                    state_.store(EventState::Idle, std::memory_order_release);
                    if (error)
                    {
                        *error =
                            "ExpertOverlay calibration could not record its exact graph-terminal event on " +
                            device_name_;
                    }
                    return false;
                }
                return true;
            }

            /** @copydoc IMoEOverlayInferenceCompletionFence::poll */
            [[nodiscard]] MoEOverlayInferenceCompletionFenceProgress poll(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                const EventState observed =
                    state_.load(std::memory_order_acquire);
                if (observed == EventState::Ready)
                    return MoEOverlayInferenceCompletionFenceProgress::Ready;
                if (observed != EventState::InFlight || !event_)
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay calibration queried an unrecorded event on " +
                            device_name_;
                    }
                    return MoEOverlayInferenceCompletionFenceProgress::Failed;
                }

                bool ready = false;
                if (!backend_->queryEvent(
                        event_.get(), device_ordinal_, &ready))
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay calibration backend event query failed on " +
                            device_name_;
                    }
                    return MoEOverlayInferenceCompletionFenceProgress::Failed;
                }
                if (!ready)
                {
                    return MoEOverlayInferenceCompletionFenceProgress::Pending;
                }
                /*
                 * Keep Ready idempotent: an aggregate may revisit this leaf
                 * while a slower sibling event is still Pending.
                 */
                state_.store(EventState::Ready, std::memory_order_release);
                return MoEOverlayInferenceCompletionFenceProgress::Ready;
            }

        private:
            /** Reusable event lifecycle visible to inference and maintenance. */
            enum class EventState : std::uint8_t
            {
                Idle,
                InFlight,
                Ready,
            };

            IBackend *backend_ = nullptr; ///< Stable backend event authority.
            int device_ordinal_ = -1; ///< Backend-local event owner ordinal.
            std::string device_name_; ///< Setup-cached failure diagnostic.
            std::shared_ptr<void> event_; ///< Persistent backend event owner.
            std::atomic<EventState> state_{EventState::Idle}; ///< Reuse guard.
        };

        /**
         * @brief Report the exact device timeline dependency that could not be joined.
         *
         * Request reset is a fatal ownership boundary: continuing after one of
         * its event waits fails would allow cache memory to be cleared while an
         * older graph still owns it. The ordinary logger retains the message
         * for diagnostics, while the direct stderr write guarantees that the
         * dependency name survives an immediately following std::terminate().
         *
         * @param device Device whose timeline dependency failed.
         * @param boundary Logical operation attempting to join the dependency.
         * @param dependency Stable name of the failed publication or event edge.
         * @param consumer User-facing name of the consuming operation.
         * @return Always false, so callers can return the diagnostic directly.
         */
        bool reportDeviceTimelineJoinFailure(
            DeviceId device,
            const char *boundary,
            const char *dependency,
            const char *consumer)
        {
            const char *safe_boundary =
                boundary && boundary[0] != '\0' ? boundary : "unknown";
            const char *safe_dependency =
                dependency && dependency[0] != '\0' ? dependency : "unknown";
            const char *safe_consumer =
                consumer && consumer[0] != '\0' ? consumer : "unknown";
            const std::string device_name = device.toString();

            LOG_ERROR("[DeviceTimelineJoin] Failed to join required device event"
                      << " boundary=" << safe_boundary
                      << " dependency=" << safe_dependency
                      << " consumer=" << safe_consumer
                      << " device=" << device_name);
            std::fprintf(
                stderr,
                "[FATAL] Device timeline join failed: boundary=%s "
                "dependency=%s consumer=%s device=%s\n",
                safe_boundary,
                safe_dependency,
                safe_consumer,
                device_name.c_str());
            std::fflush(stderr);
            return false;
        }

        constexpr size_t kStochasticDistributionMaxK = 256;
        constexpr size_t kStochasticTopKSmallKCap = 64;
        constexpr size_t kStochasticTopKPartialBlocks = 128;
        constexpr size_t kStochasticTopKSmallKThreads = 64;
        // Keep a few resident rows available even for depth-one configurations
        // so condition and bonus sampling can use disjoint scratch slots. This
        // is an allocation floor, not a verifier-row limit.
        /**
         * @brief Returns true when compact MTP publication snapshots should be logged.
         *
         * The publication path is normally fully device resident and must not
         * synchronize merely to observe tiny metadata vectors.  This diagnostic
         * switch is therefore intentionally opt-in; when enabled, the helper
         * below copies compact INT32 buffers back after enqueueing stream work
         * so failed parity cases can report the exact device-side restore rows,
         * cache targets, and compact reducer flags that drove publication.
         */
        bool mtpPublicationDiagnosticsEnabled()
        {
            return DebugEnv::isTruthyEnv("LLAMINAR_MTP_PUBLICATION_DIAGNOSTICS");
        }

        /**
         * @brief Describe one device-resident byte range at an MTP graph boundary.
         *
         * These records are consumed only by the opt-in graph-reuse diagnostic
         * below.  The production sidecar remains fully device owned; no host
         * shadow is retained by the orchestrator.
         */
        struct MTPGraphReuseDiagnosticSource
        {
            std::string name;
            const void *device_data = nullptr;
            size_t byte_count = 0;
        };

        /**
         * @brief Return the byte width of the first logical tensor row.
         *
         * Sidecar decode operates on one row.  Hashing only that row excludes
         * unused arena capacity and stale tail rows from the diagnostic while
         * still observing the exact bytes consumed or produced by the graph.
         */
        size_t mtpDiagnosticTensorRowBytes(const TensorBase *tensor)
        {
            if (!tensor)
                return 0;
            const size_t rows = std::max<size_t>(1, tensor->rows());
            const size_t bytes = tensor->size_bytes();
            return bytes > 0 && bytes % rows == 0 ? bytes / rows : 0;
        }

        /**
         * @brief Build a deterministic FNV-1a digest for diagnostic bytes.
         */
        uint64_t mtpDiagnosticHashBytes(
            const uint8_t *bytes,
            size_t byte_count)
        {
            constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
            constexpr uint64_t kFnvPrime = 1099511628211ULL;
            uint64_t hash = kFnvOffsetBasis;
            for (size_t i = 0; i < byte_count; ++i)
            {
                hash ^= static_cast<uint64_t>(bytes[i]);
                hash *= kFnvPrime;
            }
            return hash;
        }

        /**
         * @brief Check whether graph-reuse diagnostics apply to one sidecar context.
         *
         * `LLAMINAR_MTP_GRAPH_REUSE_DIAGNOSTICS_CONTEXT` can name one context,
         * or end in `*` to select a context prefix such as
         * `mtp_decode_sidecar*`.  A long parity run can therefore compare the
         * first and chained decode sidecars without copying unrelated prefill
         * or catch-up buffers.
         */
        bool mtpGraphReuseDiagnosticsEnabledForContext(
            const std::string &context)
        {
            if (!DebugEnv::isTruthyEnv(
                    "LLAMINAR_MTP_GRAPH_REUSE_DIAGNOSTICS"))
            {
                return false;
            }
            const char *filter = DebugEnv::envValue(
                "LLAMINAR_MTP_GRAPH_REUSE_DIAGNOSTICS_CONTEXT");
            if (!filter)
                return true;
            const std::string_view filter_view(filter);
            if (!filter_view.empty() && filter_view.back() == '*')
            {
                const std::string_view prefix =
                    filter_view.substr(0, filter_view.size() - 1);
                return std::string_view(context).starts_with(prefix);
            }
            return context == filter_view;
        }

        /**
         * @brief Select one GDN layer for request-reset state diagnostics.
         *
         * The state-bank observation can copy several megabytes, so graph
         * reuse diagnostics do not inspect every recurrent layer implicitly.
         * A developer investigating a localized layer boundary names that
         * layer explicitly with
         * `LLAMINAR_MTP_GRAPH_REUSE_DIAGNOSTICS_GDN_LAYER`.
         */
        std::optional<int> mtpGraphReuseDiagnosticGDNLayer()
        {
            const char *value = DebugEnv::envValue(
                "LLAMINAR_MTP_GRAPH_REUSE_DIAGNOSTICS_GDN_LAYER");
            if (!value || value[0] == '\0')
                return std::nullopt;

            char *end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (!end || end == value || *end != '\0' ||
                parsed < 0 ||
                parsed > static_cast<long>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument(
                    "LLAMINAR_MTP_GRAPH_REUSE_DIAGNOSTICS_GDN_LAYER must "
                    "name one non-negative layer index");
            }
            return static_cast<int>(parsed);
        }

        /**
         * @brief Hash device truth at one MTP sidecar graph boundary.
         *
         * Every D2H copy is enqueued on the exact sidecar producer stream, then
         * that stream is synchronized once after all copies have been queued.
         * This makes the diagnostic a faithful observation of the graph's
         * ordered inputs or outputs rather than an observation from an
         * unrelated default stream.  It is deliberately opt-in because the
         * final synchronization perturbs latency and may help distinguish a
         * missing event edge from a stale stable-pointer binding.
         *
         * @return `true` when every requested byte range was captured.
         */
        bool logMTPGraphReuseBoundaryDiagnostics(
            const char *boundary,
            const std::string &context,
            uint64_t session_epoch,
            bool graph_rebuilt,
            bool graph_replayed,
            IBackend &backend,
            const DeviceId &device,
            void *stream,
            const std::vector<MTPGraphReuseDiagnosticSource> &sources,
            bool force_enabled = false)
        {
            if (!force_enabled &&
                !mtpGraphReuseDiagnosticsEnabledForContext(context))
                return true;
            if (!device.is_gpu() || !stream)
            {
                LOG_ERROR("[MTPGraphReuseDiagnostics] requested without an exact GPU stream"
                          << " context=" << context
                          << " boundary=" << (boundary ? boundary : "<unknown>"));
                return false;
            }

            std::vector<std::vector<uint8_t>> host_bytes(sources.size());
            for (size_t i = 0; i < sources.size(); ++i)
            {
                const auto &source = sources[i];
                if (!source.device_data || source.byte_count == 0)
                {
                    LOG_ERROR("[MTPGraphReuseDiagnostics] missing device bytes"
                              << " context=" << context
                              << " boundary=" << (boundary ? boundary : "<unknown>")
                              << " source=" << (source.name.empty() ? "<unnamed>" : source.name)
                              << " bytes=" << source.byte_count);
                    return false;
                }
                host_bytes[i].resize(source.byte_count);
            }

            for (size_t i = 0; i < sources.size(); ++i)
            {
                const auto &source = sources[i];
                if (!backend.deviceToHostOnStream(
                        host_bytes[i].data(),
                        source.device_data,
                        source.byte_count,
                        device.gpu_ordinal(),
                        stream))
                {
                    LOG_ERROR("[MTPGraphReuseDiagnostics] failed to enqueue device observation"
                              << " context=" << context
                              << " boundary=" << (boundary ? boundary : "<unknown>")
                              << " source=" << (source.name.empty() ? "<unnamed>" : source.name)
                              << " bytes=" << source.byte_count);
                    return false;
                }
            }
            void *completion_event =
                backend.createEvent(device.gpu_ordinal());
            if (!completion_event)
            {
                LOG_ERROR("[MTPGraphReuseDiagnostics] failed to allocate the diagnostic completion event"
                          << " context=" << context
                          << " boundary=" << (boundary ? boundary : "<unknown>"));
                return false;
            }
            const bool published =
                backend.recordEvent(
                    completion_event,
                    device.gpu_ordinal(),
                    stream);
            const bool completed =
                published &&
                backend.waitForEvent(
                    completion_event,
                    device.gpu_ordinal());
            backend.destroyEvent(
                completion_event,
                device.gpu_ordinal());
            if (!completed)
            {
                LOG_ERROR("[MTPGraphReuseDiagnostics] failed to complete the exact diagnostic event"
                          << " context=" << context
                          << " boundary=" << (boundary ? boundary : "<unknown>")
                          << " published=" << published);
                return false;
            }

            std::ostringstream message;
            message << "[MTPGraphReuseDiagnostics]"
                    << " epoch=" << session_epoch
                    << " device=" << device.toString()
                    << " context=" << context
                    << " boundary=" << (boundary ? boundary : "<unknown>")
                    << " rebuilt=" << (graph_rebuilt ? "true" : "false")
                    << " replayed=" << (graph_replayed ? "true" : "false");
            for (size_t i = 0; i < sources.size(); ++i)
            {
                const auto &source = sources[i];
                message << ' '
                        << (source.name.empty() ? "unnamed" : source.name)
                        << "_bytes=" << source.byte_count
                        << ' '
                        << (source.name.empty() ? "unnamed" : source.name)
                        << "_hash=0x" << std::hex
                        << mtpDiagnosticHashBytes(
                               host_bytes[i].data(),
                               host_bytes[i].size())
                        << std::dec;
                if (source.byte_count == sizeof(int32_t))
                {
                    int32_t value = 0;
                    std::memcpy(
                        &value,
                        host_bytes[i].data(),
                        sizeof(value));
                    message << ' '
                            << (source.name.empty() ? "unnamed" : source.name)
                            << "_value=" << value;
                }
            }
            LOG_INFO(message.str());
            return true;
        }

        /**
         * @brief Observe the complete scalar target-sample publication boundary.
         *
         * This diagnostic establishes whether a stale logical position entered
         * the publication kernel, was produced by that kernel, or appeared only
         * after a later mailbox lifetime transition. Every source is copied on
         * the exact target-sample producer stream and one event is host-waited
         * only when the explicit publication diagnostic is enabled.
         *
         * Production execution never calls a transfer, allocates an event, or
         * blocks the host through this helper because the environment gate is
         * false by default.
         */
        void logMTPDeviceTargetLogicalStatePublicationDiagnostics(
            IBackend &backend,
            const DeviceId &device,
            void *producer_stream,
            uint64_t live_state_epoch,
            const int32_t *target_token_device,
            const int32_t *live_position_device,
            const int32_t *base_cached_tokens_device,
            const int32_t *target_position_device,
            const int32_t *next_condition_token_device,
            const int32_t *publication_ok_device)
        {
            if (!mtpPublicationDiagnosticsEnabled())
                return;

            const bool observed = logMTPGraphReuseBoundaryDiagnostics(
                "published",
                "device_target_logical_state_publication",
                live_state_epoch,
                /*graph_rebuilt=*/false,
                /*graph_replayed=*/false,
                backend,
                device,
                producer_stream,
                {
                    {"target_token", target_token_device, sizeof(int32_t)},
                    {"live_position", live_position_device, sizeof(int32_t)},
                    {"base_cached_tokens",
                     base_cached_tokens_device,
                     sizeof(int32_t)},
                    {"target_position",
                     target_position_device,
                     sizeof(int32_t)},
                    {"next_condition_token",
                     next_condition_token_device,
                     sizeof(int32_t)},
                    {"publication_ok",
                     publication_ok_device,
                     sizeof(int32_t)},
                },
                /*force_enabled=*/true);
            if (!observed)
            {
                LOG_ERROR(
                    "[MTPPublicationDiagnostics] failed to observe the "
                    "device-target logical-state publication boundary");
            }
        }

        /**
         * @brief Format a compact INT32 vector for single-line publication logs.
         */
        std::string formatMTPDiagnosticIntVector(
            const std::vector<int32_t> &values,
            size_t max_values = 32)
        {
            std::ostringstream out;
            out << '[';
            const size_t shown = std::min(values.size(), max_values);
            for (size_t i = 0; i < shown; ++i)
            {
                if (i != 0)
                    out << ',';
                out << values[i];
            }
            if (shown < values.size())
                out << ",...";
            out << ']';
            return out.str();
        }

        /**
         * @brief Queue a tiny D2H metadata copy for opt-in MTP diagnostics.
         *
         * The caller publishes one completion event after enqueuing every buffer
         * in a snapshot, then waits for that exact event. Returning false here
         * keeps the diagnostic from hiding pointer-device mistakes: if a future
         * workspace binding hands us a host pointer or a pointer from another
         * GPU, the log will identify the specific compact buffer that could not
         * be read.
         */
        bool enqueueMTPDiagnosticIntCopy(
            IBackend &backend,
            const char *label,
            const int32_t *device_values,
            int value_count,
            const DeviceId &device,
            void *stream,
            std::vector<int32_t> &host_values)
        {
            host_values.clear();
            if (value_count <= 0)
                return true;
            if (!device_values)
            {
                LOG_ERROR("[MTPPublicationDiagnostics] missing device buffer for "
                          << (label ? label : "<unnamed>"));
                return false;
            }
            host_values.assign(static_cast<size_t>(value_count), 0);
            const size_t bytes =
                sizeof(int32_t) * static_cast<size_t>(value_count);
            if (!backend.deviceToHostOnStream(
                    host_values.data(),
                    device_values,
                    bytes,
                    device.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[MTPPublicationDiagnostics] failed to enqueue D2H copy for "
                          << (label ? label : "<unnamed>")
                          << " count=" << value_count
                          << " device=" << device.toString());
                return false;
            }
            return true;
        }

        /**
         * @brief Queue a tiny FP32 D2H copy for opt-in MTP diagnostics.
         *
         * This is the floating-point companion to
         * enqueueMTPDiagnosticIntCopy().  It deliberately enqueues only; the
         * caller publishes one event after every related integer and
         * floating-point observation so one coherent stream point is exposed
         * to the host.
         */
        bool enqueueMTPDiagnosticFloatCopy(
            IBackend &backend,
            const char *label,
            const float *device_values,
            int value_count,
            const DeviceId &device,
            void *stream,
            std::vector<float> &host_values)
        {
            host_values.clear();
            if (value_count <= 0)
                return true;
            if (!device_values)
            {
                LOG_ERROR("[MTPPublicationDiagnostics] missing device buffer for "
                          << (label ? label : "<unnamed>"));
                return false;
            }
            host_values.assign(static_cast<size_t>(value_count), 0.0f);
            const size_t bytes =
                sizeof(float) * static_cast<size_t>(value_count);
            if (!backend.deviceToHostOnStream(
                    host_values.data(),
                    device_values,
                    bytes,
                    device.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[MTPPublicationDiagnostics] failed to enqueue D2H copy for "
                          << (label ? label : "<unnamed>")
                          << " count=" << value_count
                          << " device=" << device.toString());
                return false;
            }
            return true;
        }

        /**
         * @brief Complete an opt-in diagnostic observation at one stream point.
         *
         * Diagnostic D2H copies need host-visible completion before their values
         * can be formatted. They must not achieve that completion by draining an
         * entire stream: the copy batch is instead followed by an event recorded
         * on the exact producer stream, and the host waits only for that event.
         *
         * Event creation, publication, and completion are required contracts.
         * A failure cannot leave diagnostics reading partially copied host data,
         * so every failure is fatal after any allocated event has been released.
         *
         * @param backend Backend that owns the producer stream and event.
         * @param device GPU device on which the producer stream executes.
         * @param producer_stream Exact stream that enqueued the diagnostic copies.
         * @param context Human-readable failure context.
         *
         * @throws std::invalid_argument If a GPU device or exact stream is absent.
         * @throws std::runtime_error If event creation, record, or wait fails.
         */
        void completeMTPDiagnosticObservation(
            IBackend &backend,
            const DeviceId &device,
            void *producer_stream,
            const char *context)
        {
            if (!device.is_gpu() || !producer_stream)
            {
                throw std::invalid_argument(
                    std::string(context ? context : "MTP diagnostic observation") +
                    " requires an exact GPU producer stream");
            }

            void *completion_event = backend.createEvent(device.gpu_ordinal());
            if (!completion_event)
            {
                throw std::runtime_error(
                    std::string(context ? context : "MTP diagnostic observation") +
                    " could not create its completion event");
            }

            const bool published = backend.recordEvent(
                completion_event,
                device.gpu_ordinal(),
                producer_stream);
            if (!published)
            {
                backend.destroyEvent(completion_event, device.gpu_ordinal());
                throw std::runtime_error(
                    std::string(context ? context : "MTP diagnostic observation") +
                    " could not publish its completion event");
            }

            const bool completed =
                backend.waitForEvent(completion_event, device.gpu_ordinal());
            backend.destroyEvent(completion_event, device.gpu_ordinal());
            if (!completed)
            {
                throw std::runtime_error(
                    std::string(context ? context : "MTP diagnostic observation") +
                    " could not complete its published event");
            }
        }

        /**
         * @brief Observe one serial-equivalent grouped stochastic decision.
         *
         * A seeded grouped verifier must produce exactly the same sampled token
         * at every logical position as repeated serial decode.  When that
         * contract fails, the useful evidence is the complete decision tuple:
         * the device-owned base position, compact target distributions,
         * materialized speculative row, per-row sampled targets, and final
         * compact summary.  Capturing only the published next token cannot
         * distinguish a stale position from a top-k construction or summary
         * defect.
         *
         * This helper is unreachable unless publication diagnostics are
         * explicitly enabled.  Every D2H observation is queued on the exact
         * verifier producer stream, followed by one event publication and one
         * host wait.  Production execution therefore remains allocation-free,
         * synchronization-free, and fully device resident.
         */
        void logMTPSerialEquivalentStochasticDecisionDiagnostics(
            IBackend &backend,
            const DeviceId &device,
            void *producer_stream,
            uint64_t seed,
            const int32_t *base_position_device,
            int position_offset,
            const int32_t *target_token_ids_device,
            const float *target_probs_device,
            int distribution_stride,
            int top_k,
            int target_row_count,
            const int32_t *verifier_tokens_device,
            int verifier_token_count,
            const int32_t *sampled_target_tokens_device,
            const int32_t *summary_tokens_device,
            int summary_token_count,
            const int32_t *summary_meta_device,
            int summary_meta_count)
        {
            if (!mtpPublicationDiagnosticsEnabled())
                return;
            if (!device.is_gpu() || !producer_stream ||
                seed == 0 || !base_position_device ||
                !target_token_ids_device || !target_probs_device ||
                distribution_stride < top_k || top_k <= 0 ||
                target_row_count <= 0 ||
                !verifier_tokens_device || verifier_token_count <= 0 ||
                !sampled_target_tokens_device ||
                !summary_tokens_device || summary_token_count <= 0 ||
                !summary_meta_device || summary_meta_count <= 0)
            {
                throw std::invalid_argument(
                    "serial-equivalent stochastic diagnostics require a "
                    "complete GPU decision tuple and exact producer stream");
            }

            std::vector<int32_t> base_position;
            std::vector<std::vector<int32_t>> target_ids(
                static_cast<size_t>(target_row_count));
            std::vector<std::vector<float>> target_probs(
                static_cast<size_t>(target_row_count));
            std::vector<int32_t> verifier_tokens;
            std::vector<int32_t> sampled_target_tokens;
            std::vector<int32_t> summary_tokens;
            std::vector<int32_t> summary_meta;

            bool ok = enqueueMTPDiagnosticIntCopy(
                backend,
                "serial_equivalent_base_position",
                base_position_device,
                1,
                device,
                producer_stream,
                base_position);
            for (int row = 0; row < target_row_count; ++row)
            {
                const size_t row_offset =
                    static_cast<size_t>(row) *
                    static_cast<size_t>(distribution_stride);
                ok = enqueueMTPDiagnosticIntCopy(
                         backend,
                         "serial_equivalent_target_ids",
                         target_token_ids_device + row_offset,
                         top_k,
                         device,
                         producer_stream,
                         target_ids[static_cast<size_t>(row)]) &&
                     ok;
                ok = enqueueMTPDiagnosticFloatCopy(
                         backend,
                         "serial_equivalent_target_probs",
                         target_probs_device + row_offset,
                         top_k,
                         device,
                         producer_stream,
                         target_probs[static_cast<size_t>(row)]) &&
                     ok;
            }
            ok = enqueueMTPDiagnosticIntCopy(
                     backend,
                     "serial_equivalent_verifier_tokens",
                     verifier_tokens_device,
                     verifier_token_count,
                     device,
                     producer_stream,
                     verifier_tokens) &&
                 ok;
            ok = enqueueMTPDiagnosticIntCopy(
                     backend,
                     "serial_equivalent_sampled_targets",
                     sampled_target_tokens_device,
                     target_row_count,
                     device,
                     producer_stream,
                     sampled_target_tokens) &&
                 ok;
            ok = enqueueMTPDiagnosticIntCopy(
                     backend,
                     "serial_equivalent_summary_tokens",
                     summary_tokens_device,
                     summary_token_count,
                     device,
                     producer_stream,
                     summary_tokens) &&
                 ok;
            ok = enqueueMTPDiagnosticIntCopy(
                     backend,
                     "serial_equivalent_summary_meta",
                     summary_meta_device,
                     summary_meta_count,
                     device,
                     producer_stream,
                     summary_meta) &&
                 ok;
            if (!ok)
            {
                throw std::runtime_error(
                    "serial-equivalent stochastic diagnostics could not "
                    "enqueue the complete decision tuple");
            }

            completeMTPDiagnosticObservation(
                backend,
                device,
                producer_stream,
                "serial-equivalent stochastic decision diagnostics");

            std::ostringstream message;
            message << "[MTPStochasticDecisionDiagnostics]"
                    << " device=" << device.toString()
                    << " seed=" << seed
                    << " base=" << base_position.front()
                    << " position_offset=" << position_offset
                    << " verifier_tokens="
                    << formatMTPDiagnosticIntVector(verifier_tokens)
                    << " sampled_targets="
                    << formatMTPDiagnosticIntVector(sampled_target_tokens)
                    << " summary_tokens="
                    << formatMTPDiagnosticIntVector(summary_tokens)
                    << " summary_meta="
                    << formatMTPDiagnosticIntVector(summary_meta);
            for (int row = 0; row < target_row_count; ++row)
            {
                const int logical_position =
                    base_position.front() + position_offset + row;
                const float threshold =
                    sampling_math::mtp_spec_threshold_from_seed(
                        seed,
                        logical_position,
                        0 /* MTPSpecStochasticDrawPurpose::Sample */);
                message << " row" << row
                        << "_position=" << logical_position
                        << " row" << row
                        << "_threshold=" << std::fixed
                        << std::setprecision(9) << threshold
                        << " row" << row << "_topk=[";
                for (int entry = 0; entry < top_k; ++entry)
                {
                    if (entry != 0)
                        message << ',';
                    message
                        << target_ids[static_cast<size_t>(row)]
                                     [static_cast<size_t>(entry)]
                        << ':'
                        << std::setprecision(9)
                        << target_probs[static_cast<size_t>(row)]
                                       [static_cast<size_t>(entry)];
                }
                message << ']';
            }
            LOG_INFO(message.str());
        }

        /**
         * @brief Report the immutable per-layer KV checkpoint captured for MTP.
         *
         * GPU sequence-state checkpoints use a compact, backend-independent
         * layout: one ring head per cache layer followed by one visible token
         * count per cache layer.  Dumping the complete row at the verifier
         * boundary makes it possible to identify the first layer whose metadata
         * diverges without introducing host state into the production lifetime.
         *
         * This helper is strictly opt-in diagnostics. Its D2H copy and exact
         * producer-event wait are unreachable unless
         * `LLAMINAR_MTP_PUBLICATION_DIAGNOSTICS=1`.
         */
        void logMTPKVSequenceStateCheckpointDiagnostics(
            const char *phase,
            IBackend &backend,
            const DeviceId &device,
            void *stream,
            int request_index,
            const DeviceKVSequenceStateCheckpoint &checkpoint)
        {
            if (!mtpPublicationDiagnosticsEnabled() || !device.is_gpu() || !stream)
                return;

            const size_t int_count = checkpoint.bytes / sizeof(int32_t);
            if (!checkpoint.valid() ||
                int_count == 0 ||
                int_count % 2 != 0)
            {
                LOG_ERROR("[MTPPublicationDiagnostics] invalid KV checkpoint"
                          << " phase=" << (phase ? phase : "<unknown>")
                          << " request=" << request_index
                          << " bytes=" << checkpoint.bytes);
                return;
            }

            std::vector<int32_t> checkpoint_values;
            if (!enqueueMTPDiagnosticIntCopy(
                    backend,
                    "main_kv_sequence_state_checkpoint",
                    static_cast<const int32_t *>(checkpoint.data()),
                    static_cast<int>(int_count),
                    device,
                    stream,
                    checkpoint_values))
            {
                return;
            }
            completeMTPDiagnosticObservation(
                backend,
                device,
                stream,
                "MTP KV checkpoint diagnostics");

            const size_t layer_count = int_count / 2;
            const std::vector<int32_t> heads(
                checkpoint_values.begin(),
                checkpoint_values.begin() +
                    static_cast<std::ptrdiff_t>(layer_count));
            const std::vector<int32_t> counts(
                checkpoint_values.begin() +
                    static_cast<std::ptrdiff_t>(layer_count),
                checkpoint_values.end());
            LOG_INFO("[MTPPublicationDiagnostics]"
                     << " phase=" << (phase ? phase : "<unknown>")
                     << " device=" << device.toString()
                     << " request=" << request_index
                     << " layers=" << layer_count
                     << " heads=" << formatMTPDiagnosticIntVector(heads)
                     << " counts=" << formatMTPDiagnosticIntVector(counts));
        }

        /**
         * @brief Render the global bindings present in a frozen weight set.
         *
         * Decode sidecar validation happens at the request/setup boundary, so
         * the error should identify the exact alternate weight view that was
         * installed instead of forcing the caller to infer it from a later graph
         * construction failure. Layer bindings are counted separately because a
         * decode sidecar can contain hundreds of valid per-layer weights while
         * still missing the global embedding, norm, or LM-head bindings that
         * define its lifetime contract.
         *
         * @param weight_set Frozen weight set being validated.
         * @return Compact textual summary of global names and layer-binding count.
         */
        std::string summarizeFrozenWeightSetGlobals(const FrozenModelWeightSet &weight_set)
        {
            std::ostringstream summary;
            size_t global_count = 0;
            size_t layer_count = 0;
            for (const auto &binding : weight_set.bindings())
            {
                if (binding.identity.layer >= 0)
                {
                    ++layer_count;
                    continue;
                }

                if (global_count > 0)
                    summary << ", ";
                summary << binding.identity.canonical_name;
                ++global_count;
            }

            if (global_count == 0)
                summary << "<none>";
            summary << " (global_count=" << global_count
                    << ", layer_count=" << layer_count << ")";
            return summary.str();
        }

        /**
         * @brief Result of preflighting the zero-host-callback maintenance replay.
         *
         * `NotApplicable` is a policy decision made before any graph is
         * submitted. `Launched` means every segment was enqueued. There is no
         * failure value: once this path attempts a launch, backend rejection is
         * a hard execution error and must not be conflated with applicability.
         */
        enum class DirectMoEMaintenanceReplayResult
        {
            NotApplicable,
            Launched
        };

        const char *graphExecutableLaunchPhaseTag(
            DeviceGraphExecutor::GraphExecutableLaunchPhase phase)
        {
            using Phase = DeviceGraphExecutor::GraphExecutableLaunchPhase;
            switch (phase)
            {
            case Phase::InitialTransaction:
                return "initial_transaction";
            case Phase::SteadyReplay:
                return "steady_replay";
            case Phase::DiagnosticRecapture:
                return "diagnostic_recapture";
            case Phase::DiagnosticVerification:
                return "diagnostic_verification";
            }
            return "unknown";
        }

        DirectMoEMaintenanceReplayResult
        tryLaunchCapturedMoERebalanceMaintenanceGraphDirect(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            const std::string &device_key,
            const PerfStatsCollector::Tags &tags,
            const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency)
        {
            if (!ctx ||
                !gpu_ctx ||
                !segment_cache.initialized ||
                segment_cache.needs_capture ||
                !segment_cache.capture_stream ||
                segment_cache.segments.empty())
            {
                return DirectMoEMaintenanceReplayResult::NotApplicable;
            }

            for (const auto &segment : segment_cache.segments)
            {
                if (!segment.capturable ||
                    !segment.capture ||
                    !segment.capture->hasExecutable() ||
                    !segment.cached_arena_writes.empty())
                {
                    return DirectMoEMaintenanceReplayResult::NotApplicable;
                }
            }

            DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);
            const uint64_t replay_step = ++segment_cache.decode_step;
            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_graph_phase",
                1.0,
                "decode",
                device_key,
                {{"context", segment_cache.perf_context},
                 {"phase", DeviceGraphCaptureController::phaseName(
                               DeviceGraphCaptureController::Phase::Replay)}});

            if (launch_dependency &&
                !launch_dependency(
                    DeviceGraphExecutor::GraphExecutableLaunchPhase::SteadyReplay,
                    segment_cache.capture_stream))
            {
                throw std::runtime_error(
                    "Direct MoE rebalance maintenance launch dependency failed "
                    "after the path was selected on " +
                    device_key);
            }

            for (auto &segment : segment_cache.segments)
            {
                auto launch_tags = tags;
                launch_tags["context"] = segment_cache.perf_context;
                launch_tags["stage_count"] = std::to_string(segment.stage_names.size());
                launch_tags["first_stage"] =
                    segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front();
                launch_tags["last_stage"] =
                    segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.back();
                PerfStatsCollector::ScopedTimer launch_timer(
                    "moe_rebalance",
                    "device_maintenance_graph_direct_launch",
                    "decode",
                    device_key,
                    launch_tags);
                if (!segment.capture->launch())
                {
                    throw std::runtime_error(
                        "Direct MoE rebalance maintenance graph launch failed "
                        "after the path was selected on " +
                        device_key);
                }
                segment.last_executed_step = replay_step;
            }

            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_maintenance_graph_direct_replay",
                1.0,
                "decode",
                device_key,
                tags);
            return DirectMoEMaintenanceReplayResult::Launched;
        }

        void recordMoERebalanceTransferStreamPaths(
            const ComputeGraph *graph,
            const std::string &device_key,
            const PerfStatsCollector::Tags &base_tags)
        {
            if (!PerfStatsCollector::isDomainEnabled("moe_rebalance") ||
                !graph)
                return;

            for (const auto &node_name : graph->getExecutionOrder())
            {
                const ComputeNode *node = graph->getNode(node_name);
                if (!node || !node->stage)
                    continue;

                const auto *rebalance_stage =
                    dynamic_cast<const MoEDeviceRebalanceStage *>(node->stage.get());
                if (!rebalance_stage)
                    continue;

                const auto &params = rebalance_stage->getParams();
                const bool uses_transfer_slots =
                    deviceMoERebalanceModeUsesTransferSlots(params.transfer_mode) &&
                    params.local_transfer_slots != nullptr &&
                    params.local_transfer_slot_count > 0;
                if (!uses_transfer_slots)
                    continue;

                auto tags = base_tags;
                tags["stage"] = params.stage_name.empty() ? node_name : params.stage_name;
                tags["phase"] = std::to_string(static_cast<uint32_t>(params.phase));
                tags["transfer_mode"] = std::to_string(static_cast<uint32_t>(params.transfer_mode));
                tags["path"] = "auxiliary_stream";
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_rebalance_transfer_stream_path",
                    1.0,
                    "decode",
                    device_key,
                    std::move(tags));
            }
        }

        std::string normalizedTraceEnvValue(const char *value)
        {
            if (!value)
                return {};
            std::string normalized(value);
            normalized.erase(
                normalized.begin(),
                std::find_if(normalized.begin(), normalized.end(), [](unsigned char ch)
                             { return !std::isspace(ch); }));
            normalized.erase(
                std::find_if(normalized.rbegin(), normalized.rend(), [](unsigned char ch)
                             { return !std::isspace(ch); })
                    .base(),
                normalized.end());
            std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch)
                           { return static_cast<char>(std::tolower(ch)); });
            return normalized;
        }

        bool traceEnvIsFalsey(const char *value)
        {
            const std::string normalized = normalizedTraceEnvValue(value);
            return normalized.empty() || normalized == "0" || normalized == "false" ||
                   normalized == "off" || normalized == "no";
        }

        bool traceEnvIsTruthy(const char *value)
        {
            const std::string normalized = normalizedTraceEnvValue(value);
            if (!normalized.empty() &&
                std::all_of(normalized.begin(), normalized.end(), [](unsigned char ch)
                            { return std::isdigit(ch) != 0; }))
            {
                return std::atoi(normalized.c_str()) != 0;
            }
            return normalized == "1" || normalized == "true" ||
                   normalized == "on" || normalized == "yes";
        }

        std::string deviceMoERebalanceTracePathFromEnv()
        {
            const char *value = DebugEnv::envValue("LLAMINAR_MOE_REBALANCE_TRACE_JSONL");
            if (!value || traceEnvIsFalsey(value))
                return {};
            if (traceEnvIsTruthy(value))
                return "/tmp/llaminar_moe_rebalance_trace.jsonl";
            return value;
        }

        std::string jsonEscapeLocal(const std::string &value)
        {
            std::ostringstream out;
            for (char ch : value)
            {
                switch (ch)
                {
                case '\\':
                    out << "\\\\";
                    break;
                case '"':
                    out << "\\\"";
                    break;
                case '\n':
                    out << "\\n";
                    break;
                case '\r':
                    out << "\\r";
                    break;
                case '\t':
                    out << "\\t";
                    break;
                default:
                    out << ch;
                    break;
                }
            }
            return out.str();
        }

        void appendJsonStringField(
            std::ostringstream &out,
            bool &first,
            const char *name,
            const std::string &value)
        {
            if (!first)
                out << ',';
            first = false;
            out << '"' << name << "\":\"" << jsonEscapeLocal(value) << '"';
        }

        template <typename T>
        void appendJsonNumberField(
            std::ostringstream &out,
            bool &first,
            const char *name,
            T value)
        {
            if (!first)
                out << ',';
            first = false;
            out << '"' << name << "\":" << value;
        }

        void appendJsonTagsObject(
            std::ostringstream &out,
            const PerfStatsCollector::Tags &tags)
        {
            out << '{';
            bool first = true;
            for (const auto &[key, value] : tags)
                appendJsonStringField(out, first, key.c_str(), value);
            out << '}';
        }



        /**
         * @brief Logical perf/capture lane for publishing accepted shifted MTP KV rows.
         *
         * One-row sequential commits, batched hidden-row catch-up, and
         * device-token target commits use separate graph-cache shapes
         * internally, but they all implement the same accepted-state
         * publication contract.  Keeping the diagnostics on one context makes
         * graph capture/replay ownership easy to audit.
         */
        constexpr const char *kMTPDecodeCatchupContext = "mtp_decode_catchup";
        void ensureGraphStableMoERebalanceTransferOrThrow(
            const ILocalTPContext &local_tp,
            const DeviceId &primary_device)
        {
            const auto &moe_env = debugEnv().moe_rebalance;
            if (moe_env.device_rebalance_payload_sideband ||
                moe_env.allow_legacy_collective_rebalance_transfer)
            {
                throw std::runtime_error(
                    "Device-side MoE rebalance payload migration is disabled for " +
                    primary_device.to_string() +
                    " because fixed-size collective payload arenas move empty expert slots. "
                    "Use CompactTransferSlots maintenance for non-empty transfer-slot arrivals.");
            }

            if (debugEnv().moe_rebalance.device_rebalance_maintenance_graph)
            {
                if (local_tp.supportsRawAllgatherOnStreamGraphCapture())
                {
                    LOG_TRACE("[DeviceGraphOrchestrator] Device-side MoE rebalance will collect "
                              "histogram state on an async maintenance graph for "
                              << primary_device.to_string()
                              << " using graph-capturable raw NCCL/RCCL allgather");
                    return;
                }

                throw std::runtime_error(
                    "Device-side MoE rebalance maintenance requires graph-capturable raw NCCL/RCCL allgather on " +
                    primary_device.to_string() +
                    "; host publish/apply fallback is refused");
            }

            throw std::runtime_error(
                "Device-side MoE rebalance requires LLAMINAR_MOE_DEVICE_REBALANCE_MAINTENANCE_GRAPH=1 "
                "so histogram state moves on the async rolling-wave maintenance lane for " +
                primary_device.to_string() +
                "; host publish/apply fallback is refused");
        }

        uint64_t stableLegacyGraphBindingId(
            ModelContextId model_id,
            const std::string &canonical_name,
            WeightRole role,
            int layer_idx,
            DeviceId device)
        {
            uint64_t hash = 1469598103934665603ull;
            auto mixByte = [&](uint8_t byte)
            {
                hash ^= static_cast<uint64_t>(byte);
                hash *= 1099511628211ull;
            };
            auto mixString = [&](const std::string &value)
            {
                for (unsigned char ch : value)
                    mixByte(ch);
                mixByte(0xffu);
            };
            auto mixInt = [&](int64_t value)
            {
                uint64_t bits = static_cast<uint64_t>(value);
                for (int i = 0; i < 8; ++i)
                    mixByte(static_cast<uint8_t>((bits >> (i * 8)) & 0xffu));
            };

            mixInt(static_cast<int64_t>(model_id.value));
            mixString(canonical_name);
            mixInt(static_cast<int>(role));
            mixInt(layer_idx);
            mixInt(static_cast<int>(device.type));
            mixInt(device.ordinal);
            return hash == 0 ? 1 : hash;
        }

        class ScopedStringOverride
        {
        public:
            ScopedStringOverride(std::string &target, std::string value)
                : target_(target), previous_(target)
            {
                target_ = std::move(value);
            }

            ~ScopedStringOverride()
            {
                target_ = std::move(previous_);
            }

            ScopedStringOverride(const ScopedStringOverride &) = delete;
            ScopedStringOverride &operator=(const ScopedStringOverride &) = delete;

        private:
            std::string &target_;
            std::string previous_;
        };

        /**
         * @brief Populate the public speculative outcome object from compact
         *        sampler metadata.
         *
         * CPU, CUDA, and ROCm all use the same SamplingMath metadata layout.
         * Keeping this decode in one place makes the host CPU reducer and the
         * GPU device reducer structurally identical after they have produced
         * `output_tokens` and `meta`.
         */
        bool fillSpeculativeVerifyOutcomeFromMeta(
            const int32_t *output_tokens,
            int output_token_capacity,
            const int *meta,
            DeviceSpeculativeVerifyBatchOutcome *out)
        {
            using namespace sampling_math;
            if (!out || !output_tokens || !meta || output_token_capacity <= 0 ||
                meta[kSpecBatchMetaOk] == 0 ||
                meta[kSpecBatchMetaOutputCount] < 0 ||
                meta[kSpecBatchMetaOutputCount] > output_token_capacity)
            {
                return false;
            }
            for (int i = 0; i < meta[kSpecBatchMetaOutputCount]; ++i)
            {
                if (output_tokens[i] < 0)
                    return false;
            }

            out->ok = true;
            out->output_tokens.assign(
                output_tokens,
                output_tokens + output_token_capacity);
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

        /// @brief Emit a coarse VRAM checkpoint for orchestrator allocation phases.
        void logOrchestratorVramTrace(DeviceId device, const char *label)
        {
            if (!debugEnv().vram_trace || !device.is_gpu())
                return;

            IBackend *backend = getBackendFor(device);
            if (!backend)
                return;

            const int ordinal = device.gpu_ordinal();
            const size_t free_bytes = backend->deviceMemoryFree(ordinal);
            const size_t total_bytes = backend->deviceMemoryTotal(ordinal);
            const size_t used_bytes = total_bytes > free_bytes ? total_bytes - free_bytes : 0;
            LOG_TRACE("[VRAM_TRACE] " << label
                                     << " device=" << device.toString()
                                     << " used_mib=" << (used_bytes / (1024 * 1024))
                                     << " free_mib=" << (free_bytes / (1024 * 1024))
                                     << " total_mib=" << (total_bytes / (1024 * 1024)));
        }

        void logKVCacheBom(const char *label,
                           DeviceId device,
                           IKVCache *cache,
                           int batch_size)
        {
            if (!vramBomEnabled() || !cache)
                return;

            const std::string cache_label = label ? label : "kv";
            const int first_layer = cache->first_layer_index();
            const int layers = std::max(0, cache->n_layers());
            const int sequences = std::max(1, batch_size);
            const int max_seq_len = std::max(0, cache->max_seq_len());
            size_t total_bytes = 0;
            size_t buffer_count = 0;

            for (int layer_offset = 0; layer_offset < layers; ++layer_offset)
            {
                const int layer = first_layer + layer_offset;
                const auto layout = cache->logicalBlockLayout(layer, max_seq_len);
                if (layout.k_bytes == 0 && layout.v_bytes == 0)
                    continue;

                for (int seq = 0; seq < sequences; ++seq)
                {
                    const int cached_tokens = cache->get_cached_tokens(layer, seq);

                    auto log_reserved_tensor = [&](const char *side,
                                                   ActivationPrecision precision,
                                                   size_t bytes)
                    {
                        if (bytes == 0)
                            return;
                        total_bytes += bytes;
                        ++buffer_count;
                        logVramBomLine(
                            "kv_cache_buffer",
                            "label=" + cache_label +
                                " device=" + device.toString() +
                                " layer=" + std::to_string(layer) +
                                " seq=" + std::to_string(seq) +
                                " side=" + side +
                                " precision=" + activationPrecisionToString(precision) +
                                " layout=" + layoutNameShort(layout.layout) +
                                " rows=" + std::to_string(max_seq_len) +
                                " cols=" + std::to_string(layout.local_kv_heads * layout.head_dim) +
                                " reserved_tokens=" + std::to_string(max_seq_len) +
                                " cached_tokens=" + std::to_string(cached_tokens) +
                                " " + vramBomBytes(bytes));
                    };

                    log_reserved_tensor("K", layout.k_precision, layout.k_bytes);
                    log_reserved_tensor("V", layout.v_precision, layout.v_bytes);
                }
            }

            logVramBomLine(
                "kv_cache_summary",
                "label=" + cache_label +
                    " device=" + device.toString() +
                    " layers=" + std::to_string(layers) +
                    " first_layer=" + std::to_string(first_layer) +
                    " batch_size=" + std::to_string(sequences) +
                    " max_seq_len=" + std::to_string(max_seq_len) +
                    " total_kv_heads=" + std::to_string(cache->n_kv_heads()) +
                    " local_kv_heads=" + std::to_string(cache->local_n_kv_heads()) +
                    " local_kv_dim=" + std::to_string(cache->local_kv_dim()) +
                    " sharded=" + std::string(cache->is_sharded() ? "true" : "false") +
                    " buffers=" + std::to_string(buffer_count) +
                    " total_bytes=" + std::to_string(total_bytes) +
                    " total_mib=" + vramBomMiB(total_bytes));
        }

        bool applyPenaltiesToTensorRowOnDevice(
            TensorBase *tensor,
            DeviceId device,
            const LogitPenaltyBatch &penalties,
            int vocab_size,
            int row,
            int token_offset,
            void *stream,
            const char *operation)
        {
            if (penalties.empty())
                return true;
            if (!tensor || !device.is_gpu() || !stream || vocab_size <= 0 || row < 0)
                return false;
            if (!tensor->deviceValid())
                return false;

            IBackend *backend = getBackendFor(device);
            if (!backend)
                return false;

            void *gpu_ptr = tensor->gpu_data_ptr();
            if (!gpu_ptr)
                return false;

            const auto &shape = tensor->shape();
            if (shape.empty())
                return false;
            const size_t rows = shape.size() >= 2 ? shape[0] : 1;
            const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
            if (cols == 0 || static_cast<size_t>(row) >= rows ||
                cols > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return false;
            }

            std::vector<int> local_token_ids;
            std::vector<float> local_penalties;
            local_token_ids.reserve(penalties.size());
            local_penalties.reserve(penalties.size());
            const int local_vocab = static_cast<int>(cols);
            const int local_begin = std::max(0, token_offset);
            const int local_end = local_begin + local_vocab;
            for (const auto &penalty : penalties)
            {
                if (penalty.token_id < local_begin || penalty.token_id >= local_end)
                    continue;
                local_token_ids.push_back(penalty.token_id - local_begin);
                local_penalties.push_back(penalty.penalty);
            }

            if (local_token_ids.empty())
                return true;

            float *row_ptr = static_cast<float *>(gpu_ptr) +
                             static_cast<size_t>(row) * cols;
            const bool ok = backend->applyLogitPenaltiesF32(
                row_ptr,
                local_token_ids.data(),
                local_penalties.data(),
                static_cast<int>(local_token_ids.size()),
                local_vocab,
                device.gpu_ordinal(),
                stream, penalties.repetitionPenalty());
            if (!ok)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] "
                          << (operation ? operation : "applyPenaltiesToTensorRowOnDevice")
                          << " failed on " << device.toString());
            }
            return ok;
        }

        /**
         * @brief Apply sparse logit penalties to a host-owned FP32 logits row.
         *
         * The public runner API says "on device" because the caller is asking
         * the runner to mutate whichever execution device owns the logits.  For
         * CPU runners that device is the host, so this is not a fallback: it is
         * the CPU implementation of the same sparse-penalty contract used by
         * CUDA/ROCm kernels above.  `token_offset` lets tensor-parallel shards
         * consume global token ids without touching tokens outside their local
         * vocabulary slice.
         */
        bool applyPenaltiesToTensorRowOnHost(
            TensorBase *tensor,
            const LogitPenaltyBatch &penalties,
            int vocab_size,
            int row,
            int token_offset,
            const char *operation)
        {
            if (penalties.empty())
                return true;
            if (!tensor || vocab_size <= 0 || row < 0)
                return false;

            const auto &shape = tensor->shape();
            if (shape.empty())
                return false;
            const size_t rows = shape.size() >= 2 ? shape[0] : 1;
            const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
            if (cols == 0 || static_cast<size_t>(row) >= rows ||
                cols > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return false;
            }

            float *data = tensor->mutable_fp32_data();
            if (!data)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] "
                          << (operation ? operation : "applyPenaltiesToTensorRowOnHost")
                          << " requires a mutable FP32 logits tensor");
                return false;
            }

            float *row_ptr = data + static_cast<size_t>(row) * cols;
            const int local_vocab = static_cast<int>(cols);
            const int local_begin = std::max(0, token_offset);
            const int local_end = local_begin + local_vocab;
            for (const auto &penalty : penalties)
            {
                if (penalty.token_id < local_begin || penalty.token_id >= local_end)
                    continue;
                float &value = row_ptr[static_cast<size_t>(penalty.token_id - local_begin)];
                value = sampling_math::penalizedLogit(value, penalties.repetitionPenalty(), penalty.penalty);
            }
            return true;
        }

        bool prefixCacheTraceEnabled()
        {
            const char *value = DebugEnv::envValue("LLAMINAR_PREFIX_CACHE_TRACE");
            return value && value[0] != '\0' && value[0] != '0';
        }

        /**
         * @brief Returns true when focused prefix KV payload handoff hashing is enabled.
         *
         * This diagnostic is intentionally narrower than LLAMINAR_PREFIX_CACHE_TRACE:
         * it hashes large native-precision KV payloads, so callers should enable it
         * only while investigating a specific prefix-cache restore mismatch.  The
         * layer and block filters below keep the signal small enough for long
         * context parity logs.
         */
        bool prefixPayloadHashTraceEnabled()
        {
            return !DebugEnv::isFalseyEnv("LLAMINAR_PREFIX_PAYLOAD_HASH_TRACE");
        }

        /**
         * @brief Parse an optional integer environment filter for prefix payload tracing.
         *
         * @param name Environment variable name.
         * @param fallback Value returned when the variable is unset or empty.
         * @return Parsed integer value or @p fallback.
         */
        int prefixPayloadHashTraceInt(const char *name, int fallback)
        {
            const char *value = DebugEnv::envValue(name);
            if (!value || !*value)
                return fallback;
            return std::atoi(value);
        }

        /**
         * @brief Decide whether a prefix block/layer should emit payload handoff hashes.
         *
         * LLAMINAR_PREFIX_PAYLOAD_HASH_TRACE_BLOCK and
         * LLAMINAR_PREFIX_PAYLOAD_HASH_TRACE_LAYER are optional exact-match filters.
         * A negative or unset filter means "all" for that dimension.
         *
         * @param key Prefix block key being harvested or restored.
         * @param global_layer Global model layer represented by the payload slice.
         * @return True when the slice should be logged.
         */
        bool shouldTracePrefixPayloadHash(const PrefixCacheKey &key, int global_layer)
        {
            if (!prefixPayloadHashTraceEnabled())
                return false;

            const int block_filter =
                prefixPayloadHashTraceInt("LLAMINAR_PREFIX_PAYLOAD_HASH_TRACE_BLOCK", -1);
            const int layer_filter =
                prefixPayloadHashTraceInt("LLAMINAR_PREFIX_PAYLOAD_HASH_TRACE_LAYER", -1);
            return (block_filter < 0 || block_filter == key.block_index) &&
                   (layer_filter < 0 || layer_filter == global_layer);
        }

        /**
         * @brief Emit same-hash-family diagnostics for one native KV payload slice.
         *
         * The parity probe reports @ref hashByteBufferForPrefixProbe, so using the
         * same hash here lets a single failing hash be followed across harvest,
         * restore, and live-cache re-export without byte-dumping large KV blocks.
         *
         * @param phase Stable phase label such as harvest_export or populate_import_live.
         * @param device Device that owns the cache operation.
         * @param key Prefix key containing block identity and token range.
         * @param global_layer Global model layer for the cache slice.
         * @param local_layer Prefix-payload local layer index.
         * @param k_payload Native K payload bytes.
         * @param k_bytes Number of K bytes in @p k_payload.
         * @param v_payload Native V payload bytes.
         * @param v_bytes Number of V bytes in @p v_payload.
         */
        void logPrefixPayloadHashTrace(
            const char *phase,
            const DeviceId &device,
            const PrefixCacheKey &key,
            int global_layer,
            int local_layer,
            const void *k_payload,
            size_t k_bytes,
            const void *v_payload,
            size_t v_bytes)
        {
            LOG_INFO("[PREFIX_PAYLOAD_TRACE] phase=" << (phase ? phase : "unknown")
                                                     << " device=" << device.toString()
                                                     << " block=" << key.block_index
                                                     << " token_start=" << key.token_start
                                                     << " token_count=" << key.token_count
                                                     << " key=" << key.toHex()
                                                     << " global_layer=" << global_layer
                                                     << " local_layer=" << local_layer
                                                     << " k_bytes=" << k_bytes
                                                     << " v_bytes=" << v_bytes
                                                     << " k_hash="
                                                     << hashByteBufferForPrefixProbe(k_payload, k_bytes)
                                                     << " v_hash="
                                                     << hashByteBufferForPrefixProbe(v_payload, v_bytes));
        }

        std::string lowerASCII(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch)
                           { return static_cast<char>(std::tolower(ch)); });
            return value;
        }

        std::string boolTag(bool value)
        {
            return value ? "true" : "false";
        }

        /** @return Stable diagnostic name for one graph replay authority. */
        const char *graphReplayPlanPolicyTag(
            DeviceGraphExecutor::GraphReplayPlanPolicy policy) noexcept
        {
            switch (policy)
            {
            case DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph:
                return "require_full_graph";
            case DeviceGraphExecutor::GraphReplayPlanPolicy::
                AllowHeterogeneousBoundarySegmentation:
                return "allow_heterogeneous_boundary_segmentation";
            case DeviceGraphExecutor::GraphReplayPlanPolicy::
                RequireRetainedParentComposition:
                return "require_retained_parent_composition";
            case DeviceGraphExecutor::GraphReplayPlanPolicy::
                RequireRetainedParentWithConcurrentTicketService:
                return "require_retained_parent_with_concurrent_ticket_service";
            }
            return "invalid";
        }

        const char *deviceMoERebalanceTransferModeTag(DeviceMoERebalanceTransferMode mode)
        {
            switch (mode)
            {
            case DeviceMoERebalanceTransferMode::ResidentOnly:
                return "resident_only";
            case DeviceMoERebalanceTransferMode::CompactTransferSlots:
                return "compact_transfer_slots";
            case DeviceMoERebalanceTransferMode::CollectiveSidebandPayload:
                return "collective_sideband_payload";
            case DeviceMoERebalanceTransferMode::LegacyCollectiveAllGather:
                return "legacy_collective_allgather";
            default:
                return "unknown";
            }
        }

        size_t fp32LogitsRowBytes(const TensorBase *tensor)
        {
            if (!tensor)
                return 0;
            const auto &shape = tensor->shape();
            if (shape.empty())
                return 0;
            const size_t cols = shape.size() >= 2 ? shape.back() : shape.front();
            return cols * sizeof(float);
        }

        bool containsAny(const std::string &haystack, std::initializer_list<const char *> needles)
        {
            for (const char *needle : needles)
            {
                if (haystack.find(needle) != std::string::npos)
                {
                    return true;
                }
            }
            return false;
        }

        void validateMoERebalanceDomain(
            const DeviceGraphOrchestrator &orchestrator,
            const std::string &domain_id,
            const char *operation)
        {
            if (domain_id.empty())
                return;

            const auto *controller = orchestrator.moeRebalanceControllerForDomain(domain_id);
            if (!controller)
            {
                throw std::runtime_error(
                    std::string(operation) + " rejected for MoE rebalance domain '" +
                    domain_id + "': runner has no domain controller");
            }
        }

        struct GreedyLogitCandidate
        {
            float value = 0.0f;
            int32_t token = -1;
            int32_t valid = 0;
            int32_t reserved = 0;
        };

        int vocabOffsetForTPConfig(const GraphConfig &config)
        {
            if (!config.lm_head_column_parallel || !config.tp_config)
                return 0;

            const int rank = config.local_rank >= 0 ? config.local_rank : config.tp_device_idx;
            if (rank < 0 || rank >= config.tp_config->worldSize())
                return 0;

            return config.tp_config->forRank(rank).vocab_start;
        }

        bool isBetterGreedyCandidate(const GreedyLogitCandidate &candidate,
                                     const GreedyLogitCandidate &best)
        {
            if (!candidate.valid)
                return false;
            if (!best.valid)
                return true;
            if (candidate.value > best.value)
                return true;
            return candidate.value == best.value && candidate.token >= 0 &&
                   (best.token < 0 || candidate.token < best.token);
        }

        const char *perfPhaseName()
        {
            switch (GraphExecutorStats::currentPhase())
            {
            case ExecutionPhase::PREFILL:
                return "prefill";
            case ExecutionPhase::DECODE:
                return "decode";
            case ExecutionPhase::COMBINED:
            default:
                return "combined";
            }
        }

        bool isTruthyProfilingEnv(const char *name)
        {
            const char *value = DebugEnv::envValue(name);
            if (!value)
                return false;
            if (std::strcmp(value, "0") == 0 ||
                std::strcmp(value, "false") == 0 ||
                std::strcmp(value, "FALSE") == 0 ||
                std::strcmp(value, "off") == 0 ||
                std::strcmp(value, "OFF") == 0)
            {
                return false;
            }
            return true;
        }

        bool greedyMarginStatsEnabled()
        {
            return PerfStatsCollector::isDomainEnabled("sampling") &&
                   isTruthyProfilingEnv("LLAMINAR_GREEDY_MARGIN_STATS");
        }

        void recordGreedyMarginStats(
            const char *source,
            const DeviceId &device,
            int row,
            float top1,
            float top2)
        {
            if (!greedyMarginStatsEnabled())
                return;

            const float margin = top1 - top2;
            const std::string phase = perfPhaseName();
            const std::string device_name = device.toString();
            PerfStatsCollector::Tags tags{
                {"source", source ? source : "unknown"},
                {"row", std::to_string(row)}};

            PerfStatsCollector::addCounter(
                "sampling", "greedy_samples", 1.0, phase, device_name, tags);
            PerfStatsCollector::addCounter(
                "sampling", "greedy_top2_margin", margin, phase, device_name, tags);

            auto near_tie = [&](const char *name, float threshold)
            {
                if (margin <= threshold)
                {
                    PerfStatsCollector::addCounter(
                        "sampling", name, 1.0, phase, device_name, tags);
                }
            };
            near_tie("greedy_near_tie_le_1e-6", 1.0e-6f);
            near_tie("greedy_near_tie_le_1e-5", 1.0e-5f);
            near_tie("greedy_near_tie_le_1e-4", 1.0e-4f);
            near_tie("greedy_near_tie_le_1e-3", 1.0e-3f);
            near_tie("greedy_near_tie_le_1e-2", 1.0e-2f);
        }

        void recordGreedyMarginUnavailable(
            const char *source,
            const DeviceId &device,
            int row)
        {
            if (!greedyMarginStatsEnabled())
                return;

            PerfStatsCollector::addCounter(
                "sampling",
                "greedy_top2_margin_unavailable",
                1.0,
                perfPhaseName(),
                device.toString(),
                PerfStatsCollector::Tags{
                    {"source", source ? source : "unknown"},
                    {"row", std::to_string(row)}});
        }

        /**
         * @brief Select the first maximal token from the producer's exact row.
         *
         * GPU tensors retain their stream-ordered backend reduction. CPU rows
         * use the shared ISA-dispatched selector; the optional margin observer
         * alone needs a second-best logit. Neither path changes row ownership
         * or coordinates participants: the caller owns that explicit edge.
         * @param tensor Published logits tensor, not an unpopulated sibling.
         * @param row Logical row inside the published tensor.
         * @param token_offset Global vocabulary origin of this participant.
         * @param argmax_partial_vals Prepared GPU reduction-value scratch.
         * @param argmax_partial_idxs Prepared GPU reduction-index scratch.
         * @param argmax_partial_capacity Capacity of both GPU scratch banks.
         * @param stream Exact GPU producer/consumer stream; unused for CPU.
         * @param source Diagnostic name of the publication surface.
         * @return Candidate with global token ID, or an invalid candidate.
         */
        GreedyLogitCandidate sampleGreedyCandidateFromTensor(
            TensorBase *tensor,
            int row,
            int token_offset,
            void *argmax_partial_vals = nullptr,
            void *argmax_partial_idxs = nullptr,
            int argmax_partial_capacity = 0,
            void *stream = nullptr,
            const char *source = "unknown")
        {
            GreedyLogitCandidate candidate;
            if (!tensor || row < 0)
                return candidate;

            const auto &shape = tensor->shape();
            if (shape.empty())
                return candidate;

            const size_t rows = shape.size() >= 2 ? shape[0] : 1;
            const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
            if (cols == 0 || static_cast<size_t>(row) >= rows ||
                cols > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return candidate;
            }

            const size_t row_offset = static_cast<size_t>(row) * cols;
            float max_val = 0.0f;
            int max_idx = -1;

            auto device_opt = tensor->current_device();
            void *gpu_ptr_for_guard = tensor->gpu_data_ptr();
            const bool gpu_resident_logits =
                gpu_ptr_for_guard != nullptr ||
                (device_opt.has_value() && device_opt->is_gpu());
            if (device_opt.has_value() && device_opt->is_gpu() && tensor->deviceValid())
            {
                if (!stream)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] GPU greedy sampling requires an explicit non-null stream");
                    return candidate;
                }
                IBackend *backend = getBackendFor(*device_opt);
                const void *gpu_ptr = gpu_ptr_for_guard;
                if (backend && gpu_ptr)
                {
                    const auto *base = static_cast<const float *>(gpu_ptr);
                    const void *target_row = base + row_offset;
                    if (backend->argmaxF32(target_row,
                                           static_cast<int>(cols),
                                           device_opt->gpu_ordinal(),
                                           &max_val,
                                           &max_idx,
                                           stream,
                                           argmax_partial_vals,
                                           argmax_partial_idxs,
                                           argmax_partial_capacity))
                    {
                        candidate.value = max_val;
                        candidate.token = token_offset + max_idx;
                        candidate.valid = 1;

                        if (greedyMarginStatsEnabled())
                        {
                            float top_values[2] = {};
                            int top_indices[2] = {};
                            if (backend->topKF32(target_row,
                                                 static_cast<int>(cols),
                                                 2,
                                                 device_opt->gpu_ordinal(),
                                                 top_values,
                                                 top_indices,
                                                 stream))
                            {
                                recordGreedyMarginStats(
                                    source,
                                    *device_opt,
                                    row,
                                    top_values[0],
                                    top_values[1]);
                            }
                            else
                            {
                                recordGreedyMarginUnavailable(source, *device_opt, row);
                            }
                        }
                        return candidate;
                    }
                }

            }

            if (gpu_resident_logits)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU greedy sampling failed for "
                          << (source ? source : "unknown")
                          << " row=" << row
                          << " device="
                          << (device_opt.has_value() ? device_opt->toString() : "none")
                          << " device_valid=" << (tensor->deviceValid() ? "true" : "false")
                          << " gpu_ptr=" << gpu_ptr_for_guard
                          << " stream=" << stream
                          << " argmax_capacity=" << argmax_partial_capacity
                          << "; CPU logits fallback is disabled");
                return candidate;
            }

            if (!tensor->hostValid())
            {
                auto download = TransferEngine::instance().download(tensor);
                if (!download.success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to download logits for greedy sampling: "
                              << download.error);
                    return candidate;
                }
            }

            const float *data = tensor->fp32_data();
            if (!data)
                return candidate;

            const float *row_data = data + row_offset;
            if (!greedyMarginStatsEnabled())
            {
                // The selector visits lanes in vocabulary order and only
                // replaces a winner on strict greater-than. SIMD therefore
                // preserves first-token ties (including signed zero) without
                // computing unused runner-up diagnostics on every MTP row.
                if (cpu_sampling::select_topk(
                        row_data, static_cast<int>(cols), 1,
                        &max_val, &max_idx) != 1)
                    return candidate;
                candidate.value = max_val;
                candidate.token = token_offset + max_idx;
                candidate.valid = 1;
                return candidate;
            }

            // Keep the explicitly requested margin observation's original
            // second-best/NaN semantics; it does not label clean timing runs.
            max_idx = 0;
            max_val = row_data[0];
            float second_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 1; i < cols; ++i)
            {
                if (row_data[i] > max_val)
                {
                    second_val = max_val;
                    max_val = row_data[i];
                    max_idx = static_cast<int>(i);
                }
                else if (row_data[i] > second_val)
                {
                    second_val = row_data[i];
                }
            }

            candidate.value = max_val;
            candidate.token = token_offset + max_idx;
            candidate.valid = 1;
            if (cols >= 2)
            {
                auto device_for_stats = device_opt.value_or(DeviceId::cpu());
                recordGreedyMarginStats(source, device_for_stats, row, max_val, second_val);
            }
            return candidate;
        }

        int coordinateGreedyCandidate(
            const GreedyLogitCandidate &local_candidate,
            IGlobalTPContext *ctx)
        {
            if (!ctx || ctx->degree() <= 1)
                return local_candidate.valid ? local_candidate.token : -1;

            std::vector<GreedyLogitCandidate> candidates(static_cast<size_t>(ctx->degree()));
            if (!ctx->allgatherBytes(&local_candidate,
                                     candidates.data(),
                                     sizeof(GreedyLogitCandidate)))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to allgather GlobalTP greedy candidates");
                return -1;
            }

            GreedyLogitCandidate best;
            for (const auto &candidate : candidates)
            {
                if (isBetterGreedyCandidate(candidate, best))
                    best = candidate;
            }
            return best.valid ? best.token : -1;
        }

        bool attachMTPPayloadLayout(PrefixPayloadLayout &layout, const IKVCache &mtp_cache)
        {
            if (layout.block_size <= 0)
            {
                return false;
            }

            PrefixPayloadLayout mtp_layout = buildDensePrefixPayloadLayout(
                mtp_cache,
                layout.device,
                layout.block_size);
            if (mtp_layout.fa_layers <= 0 || mtp_layout.faKVBytes() == 0)
            {
                return false;
            }

            layout.mtp_origin=mtp_cache.stateOwnership().shifted_origin;
            layout.mtp_layers = mtp_layout.fa_layers;
            layout.mtp_local_kv_heads = mtp_layout.local_kv_heads;
            layout.mtp_kv_head_start = mtp_layout.kv_head_start;
            layout.mtp_head_dim = mtp_layout.head_dim;
            layout.mtp_k_precision = mtp_layout.k_precision;
            layout.mtp_v_precision = mtp_layout.v_precision;
            layout.mtp_kv_layout = mtp_layout.kv_layout;
            layout.bytes_per_mtp_layer_k = mtp_layout.bytes_per_fa_layer_k;
            layout.bytes_per_mtp_layer_v = mtp_layout.bytes_per_fa_layer_v;
            layout.mtp_kv_bytes = mtp_layout.faKVBytes();
            layout.includes_mtp_state = layout.mtp_kv_bytes > 0;
            layout.mtp_auxiliary=mtp_layout.main_auxiliary;
            return layout.includes_mtp_state;
        }

        void copyMTPPayloadLayoutFields(const PrefixPayloadLayout &src,
                                        PrefixPayloadLayout &dst)
        {
            dst.mtp_origin=src.mtp_origin;
            dst.mtp_layers = src.mtp_layers;
            dst.mtp_local_kv_heads = src.mtp_local_kv_heads;
            dst.mtp_kv_head_start = src.mtp_kv_head_start;
            dst.mtp_head_dim = src.mtp_head_dim;
            dst.mtp_k_precision = src.mtp_k_precision;
            dst.mtp_v_precision = src.mtp_v_precision;
            dst.mtp_kv_layout = src.mtp_kv_layout;
            dst.bytes_per_mtp_layer_k = src.bytes_per_mtp_layer_k;
            dst.bytes_per_mtp_layer_v = src.bytes_per_mtp_layer_v;
            dst.mtp_kv_bytes = src.mtp_kv_bytes;
            dst.includes_mtp_state = src.includes_mtp_state;
            dst.mtp_auxiliary = src.mtp_auxiliary;
        }

        bool samePrefixPayloadShape(const PrefixPayloadLayout &lhs,
                                    const PrefixPayloadLayout &rhs)
        {
            return lhs.compatiblePayloadShape(rhs) &&
                   lhs.includes_hybrid_state == rhs.includes_hybrid_state &&
                   lhs.includes_terminal_hidden == rhs.includes_terminal_hidden &&
                   lhs.includes_terminal_logits == rhs.includes_terminal_logits &&
                   lhs.totalBytes() == rhs.totalBytes();
        }

        /** @return Exact model-owned cache ordinal for one immutable logical main block. */
        int mtpTokenStartForPrefixBlock(const PrefixCacheKey &key,MTPShiftedCacheOrigin origin)
        { return mtpShiftedBlockRange(key.token_start,key.token_count,origin).first; }
        /** @return Live shifted rows, including QFN's visible token-zero seed. */
        int mtpTokenCountForPrefixBlock(const PrefixCacheKey &key,MTPShiftedCacheOrigin origin)
        { return mtpShiftedBlockRange(key.token_start,key.token_count,origin).count; }

        /**
         * @brief Allocate pure VRAM storage with orchestrator-lifetime ownership.
         *
         * Transient live checkpoints are not tensors and must not acquire an
         * implicit CPU backing allocation. The returned owner invokes only the
         * selected backend's allocation API and remains retained by the runner's
         * checkpoint pool until runner teardown.
         */
        std::shared_ptr<void> allocatePureDeviceStorage(size_t bytes,
                                                        DeviceId device)
        {
            if (bytes == 0 || !device.is_gpu())
            {
                return nullptr;
            }

            IBackend *backend = getBackendFor(device);
            if (!backend)
                return nullptr;
            const int ordinal = device.gpu_ordinal();
            void *allocation = backend->allocate(bytes, ordinal);
            if (!allocation)
                return nullptr;

            return std::shared_ptr<void>(
                allocation,
                [backend, ordinal](void *pointer)
                {
                    if (pointer)
                        backend->free(pointer, ordinal);
                });
        }

        /**
         * @brief Create one reusable backend event during runner initialization.
         */
        std::shared_ptr<void> allocatePersistentDeviceEvent(DeviceId device)
        {
            if (!device.is_gpu())
                return nullptr;
            IBackend *backend = getBackendFor(device);
            if (!backend)
                return nullptr;
            const int ordinal = device.gpu_ordinal();
            void *event = backend->createEvent(ordinal);
            if (!event)
                return nullptr;
            return std::shared_ptr<void>(
                event,
                [backend, ordinal](void *value)
                {
                    if (value)
                        backend->destroyEvent(value, ordinal);
                });
        }

        /**
         * @brief Create one reusable backend timing event during initialization.
         *
         * Timing-capable events are deliberately separate from ordinary timeline
         * events because CUDA and HIP may allocate different backend resources
         * for elapsed-time recording. Callers retain the shared owner for the
         * complete runner lifetime and only borrow additional references while a
         * measurement is pending.
         */
        std::shared_ptr<void> allocatePersistentDeviceTimingEvent(DeviceId device)
        {
            if (!device.is_gpu())
                return nullptr;
            IBackend *backend = getBackendFor(device);
            if (!backend)
                return nullptr;
            const int ordinal = device.gpu_ordinal();
            void *event = backend->createTimingEvent(ordinal);
            if (!event)
                return nullptr;
            return std::shared_ptr<void>(
                event,
                [backend, ordinal](void *value)
                {
                    if (value)
                        backend->destroyEvent(value, ordinal);
                });
        }

        bool exportMTPPrefixPayload(const IKVCache &mtp_cache,
                                    int seq_idx,
                                    const PrefixCacheKey &key,
                                    PrefixBlockHandle *handle,
                                    PrefixBlockHandle *device_hot_handle = nullptr,
                                    void *stream = nullptr,
                                    void *device_k_staging = nullptr,
                                    void *device_v_staging = nullptr,
                                    IBackend *backend = nullptr,
                                    DeviceId device = DeviceId::cpu())
        {
            if (!handle || !handle->layout.includes_mtp_state)
            {
                return true;
            }

            requireMTPShiftedCacheOrigin(mtp_cache.stateOwnership().shifted_origin,handle->layout.mtp_origin);
            if(device_hot_handle)
                requireMTPShiftedCacheOrigin(mtp_cache.stateOwnership().shifted_origin,device_hot_handle->layout.mtp_origin);
            const int token_count = mtpTokenCountForPrefixBlock(key,mtp_cache.stateOwnership().shifted_origin);
            if (token_count == 0)
            {
                return true;
            }
            if (token_count < 0 || !handle->mtpKData() || !handle->mtpVData())
            {
                return false;
            }

            for (int local_layer = 0; local_layer < handle->layout.mtp_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(mtp_cache, local_layer);
                if (global_layer < 0)
                {
                    return false;
                }

                uint8_t *k_dst = handle->mtpKData() +
                                 static_cast<size_t>(local_layer) * handle->layout.bytes_per_mtp_layer_k;
                uint8_t *v_dst = handle->mtpVData() +
                                 static_cast<size_t>(local_layer) * handle->layout.bytes_per_mtp_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = mtpTokenStartForPrefixBlock(key,mtp_cache.stateOwnership().shifted_origin);
                desc.token_count = token_count;
                desc.stream = stream;
                const auto payload_layout =
                    mtp_cache.logicalBlockLayout(global_layer, token_count);
                if (device.is_gpu())
                {
                    if (!backend || !stream ||
                        !device_k_staging || !device_v_staging)
                    {
                        return false;
                    }
                    desc.payload_domain =
                        IKVCache::KVCacheLogicalBlockPayloadDomain::Device;
                    uint8_t *hot_k_dst =
                        device_hot_handle
                            ? device_hot_handle->deviceMTPKData() +
                                  static_cast<size_t>(local_layer) *
                                      handle->layout.bytes_per_mtp_layer_k
                            : nullptr;
                    uint8_t *hot_v_dst =
                        device_hot_handle
                            ? device_hot_handle->deviceMTPVData() +
                                  static_cast<size_t>(local_layer) *
                                      handle->layout.bytes_per_mtp_layer_v
                            : nullptr;
                    if ((device_hot_handle &&
                         (!hot_k_dst || !hot_v_dst)) ||
                        !mtp_cache.exportLogicalBlock(
                            desc,
                            device_k_staging,
                            device_v_staging) ||
                        !backend->deviceToHostOnStream(
                            k_dst,
                            device_k_staging,
                            payload_layout.k_bytes,
                            device.gpu_ordinal(),
                            stream) ||
                        !backend->deviceToHostOnStream(
                            v_dst,
                            device_v_staging,
                            payload_layout.v_bytes,
                            device.gpu_ordinal(),
                            stream) ||
                        (device_hot_handle &&
                         (!backend->deviceCopyAsync(
                              hot_k_dst,
                              device_k_staging,
                              payload_layout.k_bytes,
                              device.gpu_ordinal(),
                              stream) ||
                          !backend->deviceCopyAsync(
                              hot_v_dst,
                              device_v_staging,
                              payload_layout.v_bytes,
                              device.gpu_ordinal(),
                              stream))))
                    {
                        return false;
                    }
                }
                else if (!mtp_cache.exportLogicalBlock(desc, k_dst, v_dst))
                {
                    return false;
                }
            }
            return true;
        }

        bool importMTPPrefixPayload(IKVCache &mtp_cache,
                                    int seq_idx,
                                    const PrefixBlockHandle &handle,
                                    void *stream = nullptr,
                                    void *device_k_staging = nullptr,
                                    void *device_v_staging = nullptr,
                                    IBackend *backend = nullptr,
                                    DeviceId device = DeviceId::cpu())
        {
            if (!handle.layout.includes_mtp_state)
            {
                return true;
            }

            requireMTPShiftedCacheOrigin(mtp_cache.stateOwnership().shifted_origin,handle.layout.mtp_origin);
            const int token_count = mtpTokenCountForPrefixBlock(handle.key,mtp_cache.stateOwnership().shifted_origin);
            if (token_count == 0)
            {
                return true;
            }
            const bool device_hot =
                handle.tier == PrefixStorageTier::DeviceHot;
            if (token_count < 0 ||
                (device_hot
                     ? (!handle.deviceMTPKData() ||
                        !handle.deviceMTPVData())
                     : (!handle.mtpKData() || !handle.mtpVData())))
            {
                return false;
            }

            for (int local_layer = 0; local_layer < handle.layout.mtp_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(mtp_cache, local_layer);
                if (global_layer < 0)
                {
                    return false;
                }

                const uint8_t *k_src =
                    (device_hot ? handle.deviceMTPKData()
                                : handle.mtpKData()) +
                    static_cast<size_t>(local_layer) *
                        handle.layout.bytes_per_mtp_layer_k;
                const uint8_t *v_src =
                    (device_hot ? handle.deviceMTPVData()
                                : handle.mtpVData()) +
                    static_cast<size_t>(local_layer) *
                        handle.layout.bytes_per_mtp_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = mtpTokenStartForPrefixBlock(handle.key,mtp_cache.stateOwnership().shifted_origin);
                desc.token_count = token_count;
                desc.stream = stream;
                const auto payload_layout =
                    mtp_cache.logicalBlockLayout(global_layer, token_count);
                if (device.is_gpu())
                {
                    if (!backend || !stream)
                    {
                        return false;
                    }
                    desc.payload_domain =
                        IKVCache::KVCacheLogicalBlockPayloadDomain::Device;
                    if (device_hot)
                    {
                        if (!mtp_cache.importLogicalBlock(desc, k_src, v_src))
                            return false;
                    }
                    else if (!device_k_staging || !device_v_staging ||
                             !backend->hostToDeviceOnStream(
                                 device_k_staging,
                                 k_src,
                                 payload_layout.k_bytes,
                                 device.gpu_ordinal(),
                                 stream) ||
                             !backend->hostToDeviceOnStream(
                                 device_v_staging,
                                 v_src,
                                 payload_layout.v_bytes,
                                 device.gpu_ordinal(),
                                 stream) ||
                             !mtp_cache.importLogicalBlock(
                                 desc,
                                 device_k_staging,
                                 device_v_staging))
                    {
                        return false;
                    }
                }
                else if (!mtp_cache.importLogicalBlock(desc, k_src, v_src))
                {
                    return false;
                }
            }
            return true;
        }

        void *hybridPayloadHostPtr(PrefixBlockHandle &handle)
        {
            return handle.layout.hybrid_host_state_bytes > 0 ? handle.hybrid_payload : nullptr;
        }

        const void *hybridPayloadHostPtr(const PrefixBlockHandle &handle)
        {
            return handle.layout.hybrid_host_state_bytes > 0 ? handle.hybrid_payload : nullptr;
        }

        void *hybridPayloadDevicePtr(PrefixBlockHandle &handle)
        {
            return handle.deviceHybridData();
        }

        const void *hybridPayloadDevicePtr(const PrefixBlockHandle &handle)
        {
            return handle.deviceHybridData();
        }

        /**
         * @brief Export only the selected explicit durable/local representation.
         * @param cache Canonical recurrent state owner.
         * @param seq_idx Authenticated request sequence index.
         * @param logical_token_count Existing logical checkpoint coordinate.
         * @param handle Mutable result-owned payload with exact host extent.
         * @param synchronize Existing intentional durable/local completion policy.
         * @param stream Exact native stream for selected device state, unused by CPU copies.
         * @param purpose Closed durable/local format authority; never inferred from byte size.
         * @return True after a valid complete operation, false without accepting foreign formats.
         */
        bool exportHybridPrefixPayload(
            const IKVCache &cache,
            int seq_idx,
            int logical_token_count,
            PrefixBlockHandle *handle,
            bool synchronize = true,
            void *stream = nullptr,
            HybridPrefixStatePurpose purpose = HybridPrefixStatePurpose::DurablePrefix)
        {
            if (!handle || !handle->layout.includes_hybrid_state)
            {
                return true;
            }
            const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&cache);
            void *host_payload = hybridPayloadHostPtr(*handle);
            void *device_payload = hybridPayloadDevicePtr(*handle);
            if (!host_payload && handle->layout.hybrid_device_state_bytes > 0 && handle->hybrid_payload)
                host_payload = handle->hybrid_payload;
            const bool host_staged_device_state =
                handle->layout.hybrid_device_state_bytes > 0 &&
                !device_payload &&
                host_payload != nullptr;
            if (!hybrid ||
                (handle->layout.hybrid_host_state_bytes > 0 && !host_payload) ||
                (handle->layout.hybrid_device_state_bytes > 0 &&
                 !device_payload &&
                 !host_staged_device_state))
            {
                return false;
            }

            HybridPrefixStateDescriptor desc;
            desc.purpose = purpose;
            desc.host_encoding = handle->layout.hybrid_host_encoding;
            desc.host_payload_bytes = handle->layout.hybrid_host_state_bytes;
            desc.seq_idx = seq_idx;
            desc.logical_token_count = logical_token_count;
            desc.stream = stream;
            desc.synchronize = synchronize;
            desc.host_recurrence_layout = handle->layout.hybrid_host_recurrence_layout;
            desc.device_recurrence_layout = handle->layout.hybrid_device_recurrence_layout;
            desc.include_host_state = handle->layout.hybrid_host_state_bytes > 0;
            desc.include_device_state = handle->layout.hybrid_device_state_bytes > 0;
            if (!hybrid->exportHybridPrefixState(
                    desc,
                    host_payload,
                    device_payload))
            {
                return false;
            }
            handle->has_hybrid_state = true;
            return true;
        }

        /**
         * @brief Import only the authenticated explicit durable/local representation.
         * @param cache Canonical recurrent state owner.
         * @param seq_idx Authenticated request sequence index.
         * @param handle Immutable payload retained throughout validated import.
         * @param restore_for_suffix_prefill Existing exact restore-to-prefill policy.
         * @param synchronize Existing intentional durable/local completion policy.
         * @param stream Exact native stream for selected device state, unused by CPU copies.
         * @param purpose Closed durable/local format authority; never inferred from byte size.
         * @return True after a valid complete operation, false without accepting foreign formats.
         */
        bool importHybridPrefixPayload(
            IKVCache &cache,
            const PrefixBlockHandle &handle,
            int seq_idx,
            bool restore_for_suffix_prefill,
            bool synchronize = true,
            void *stream = nullptr,
            HybridPrefixStatePurpose purpose = HybridPrefixStatePurpose::DurablePrefix)
        {
            if (!handle.layout.includes_hybrid_state)
            {
                return true;
            }
            auto *hybrid = dynamic_cast<IHybridKVCache *>(&cache);
            const void *host_payload = hybridPayloadHostPtr(handle);
            const void *device_payload = hybridPayloadDevicePtr(handle);
            if (!host_payload && handle.layout.hybrid_device_state_bytes > 0 && handle.hybrid_payload)
                host_payload = handle.hybrid_payload;
            const bool host_staged_device_state =
                handle.layout.hybrid_device_state_bytes > 0 &&
                !device_payload &&
                host_payload != nullptr;
            if (!hybrid || !handle.has_hybrid_state ||
                (handle.layout.hybrid_host_state_bytes > 0 && !host_payload) ||
                (handle.layout.hybrid_device_state_bytes > 0 &&
                 !device_payload &&
                 !host_staged_device_state))
            {
                return false;
            }

            HybridPrefixStateDescriptor desc;
            desc.purpose = purpose;
            desc.host_encoding = handle.layout.hybrid_host_encoding;
            desc.host_payload_bytes = handle.layout.hybrid_host_state_bytes;
            desc.seq_idx = seq_idx;
            desc.logical_token_count = handle.key.token_start + handle.key.token_count;
            desc.stream = stream;
            desc.synchronize = synchronize;
            desc.host_recurrence_layout = handle.layout.hybrid_host_recurrence_layout;
            desc.device_recurrence_layout = handle.layout.hybrid_device_recurrence_layout;
            desc.include_host_state = handle.layout.hybrid_host_state_bytes > 0;
            desc.include_device_state = handle.layout.hybrid_device_state_bytes > 0;
            if (restore_for_suffix_prefill &&
                handle.layout.hybrid_device_state_bytes == 0 &&
                handle.layout.hybrid_host_state_bytes == 0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Partial prefix restore requested "
                          "suffix-prefill hybrid state, but the cached block has no restorable hybrid payload");
                return false;
            }
            if (prefixCacheTraceEnabled())
            {
                LOG_INFO("[PREFIX_TRACE] import hybrid state cached_tokens="
                         << (handle.key.token_start + handle.key.token_count)
                         << " suffix_prefill="
                         << (restore_for_suffix_prefill ? "yes" : "no")
                         << " host_bytes=" << handle.layout.hybrid_host_state_bytes
                         << " device_bytes=" << handle.layout.hybrid_device_state_bytes
                         << " include_device="
                         << (desc.include_device_state ? "yes" : "no"));
            }
            return hybrid->importHybridPrefixState(
                desc,
                host_payload,
                device_payload);
        }

        bool resetHybridPrefixPayloadState(
            IKVCache &cache,
            const IKVCache::StateResetContext &context)
        {
            auto *hybrid = dynamic_cast<IHybridKVCache *>(&cache);
            if (!hybrid)
            {
                return true;
            }
            for (int offset = 0; offset < cache.n_layers(); ++offset)
            {
                // Both classification and mutation address the same global
                // layer; a local ordinal aliases another PP-owned layer.
                const int global_layer = cache.first_layer_index() + offset;
                if (hybrid->isGDNLayer(global_layer) &&
                    !cache.resetLayerState(global_layer, context))
                {
                    return false;
                }
            }
            return true;
        }

        bool liveCheckpointHasHybridState(const IKVCache &cache,
                                          DeviceId device,
                                          int cached_tokens)
        {
            if (cached_tokens <= 0)
            {
                return false;
            }

            const PrefixPayloadLayout layout =
                buildDensePrefixPayloadLayout(cache, device, cached_tokens);
            return layout.includes_hybrid_state && layout.hybrid_state_bytes > 0;
        }

        /**
         * @brief Reject appends that could overwrite an archived logical prefix.
         * @param cache Physical cache whose retained bytes support rollback.
         * @param cached_tokens Scheduler-owned checkpoint cursor.
         * @param speculative_append_headroom Sealed upper bound on future writes.
         * @return Whether the declared writes exceed non-wrapping capacity.
         */
        bool liveCheckpointLacksHeadroom(const IKVCache &cache,
                                         int cached_tokens,
                                         int speculative_append_headroom)
        {
            if (cached_tokens < 0 || speculative_append_headroom < 0)
            {
                return true;
            }

            // Subtraction avoids overflow for malformed near-INT_MAX bounds.
            return cached_tokens > cache.max_seq_len() ||
                   speculative_append_headroom > cache.max_seq_len() - cached_tokens;
        }

        PrefixPayloadLayout hybridOnlyCheckpointLayout(const PrefixPayloadLayout &layout)
        {
            PrefixPayloadLayout hybrid_layout = layout;
            hybrid_layout.main_auxiliary={};hybrid_layout.mtp_auxiliary={};
            hybrid_layout.fa_layers = 0;
            hybrid_layout.local_kv_heads = 0;
            hybrid_layout.kv_head_start = 0;
            hybrid_layout.head_dim = 0;
            hybrid_layout.bytes_per_fa_layer_k = 0;
            hybrid_layout.bytes_per_fa_layer_v = 0;
            hybrid_layout.terminal_hidden_bytes = 0;
            hybrid_layout.terminal_logits_bytes = 0;
            hybrid_layout.includes_terminal_hidden = false;
            hybrid_layout.includes_terminal_logits = false;
            hybrid_layout.mtp_layers = 0;
            hybrid_layout.mtp_local_kv_heads = 0;
            hybrid_layout.mtp_kv_head_start = 0;
            hybrid_layout.mtp_head_dim = 0;
            hybrid_layout.bytes_per_mtp_layer_k = 0;
            hybrid_layout.bytes_per_mtp_layer_v = 0;
            hybrid_layout.mtp_kv_bytes = 0;
            hybrid_layout.includes_mtp_state = false;
            return hybrid_layout;
        }

        /**
         * @brief Preserve the existing complete CPU physical image for local rollback only.
         * @param layout Process-local checkpoint layout, never a durable record schema.
         * @param cache Original physical owner; its full host extent remains unchanged.
         * @throws std::logic_error For a host section without its canonical cache owner.
         */
        void applyLocalHybridCheckpointExtent(PrefixPayloadLayout &layout, const IKVCache &cache)
        {
            if (!layout.device.is_cpu() || layout.hybrid_host_state_bytes == 0) return;
            const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&cache);
            if (!hybrid) throw std::logic_error("Local GDN checkpoint requires its cache owner");
            const auto metadata = hybrid->hybridPrefixStateMetadata();
            layout.hybrid_host_encoding = HybridPrefixHostEncoding::CPUFixedEpochLocalImage;
            layout.hybrid_host_state_capacity_bytes = metadata.host_bytes;
            layout.hybrid_host_state_bytes = metadata.host_bytes;
            layout.hybrid_state_bytes = metadata.host_bytes + layout.hybrid_device_state_bytes;
        }

        /**
         * @brief Build the hybrid-state payload layout used by live rollback checkpoints.
         *
         * Live MTP checkpoints are not portable prefix-cache records; they are
         * process-local rollback handles. GPU checkpoints retain only the
         * kernel-owned local/full device banks. CPU checkpoints retain the
         * host-owned vectors that are CPU's actual live state.
         * @param layout Stable canonical prefix schema owned by this participant.
         * @param device_only True for the existing GPU local/full-only rollback scope.
         * @param cache Original recurrence owner, never a second state mirror.
         * @return Process-local checkpoint geometry with original full physical bytes.
         */
        PrefixPayloadLayout liveHybridCheckpointLayout(const PrefixPayloadLayout &layout,
                                                       bool device_only, const IKVCache &cache)
        {
            PrefixPayloadLayout hybrid_layout = hybridOnlyCheckpointLayout(layout);
            applyLocalHybridCheckpointExtent(hybrid_layout, cache);
            if (device_only && hybrid_layout.hybrid_device_state_bytes > 0)
            {
                hybrid_layout.hybrid_host_state_bytes = 0;
                hybrid_layout.hybrid_state_bytes = hybrid_layout.hybrid_device_state_bytes;
            }
            hybrid_layout.includes_hybrid_state = hybrid_layout.hybrid_state_bytes > 0;
            return hybrid_layout;
        }

        /**
         * @brief Engine-facing lease for one coordinator-owned prefill chunk.
         *
         * The existing participant scope owns the ExpertOverlay protocol state
         * machine.  This adapter keeps that scope alive across the execution
         * engine boundary and binds its launch-edge arm to the exact chunk
         * executable without exposing MoE types to ForwardExecutionEngine.
         */
        class MoEOverlayPrefillChunkGraphSubmissionLease final
            : public IPrefillChunkGraphSubmissionLease
        {
        public:
            /**
             * @brief Enter one continuation participant into a prefill graph group.
             * @param coordinator Rank-wide transaction authority.
             * @param descriptor Immutable chunk role and row geometry.
             * @param participant_index Stable LocalTP participant ordinal.
             */
            MoEOverlayPrefillChunkGraphSubmissionLease(
                std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
                    coordinator,
                MoEOverlayInferenceExecutionDescriptor descriptor,
                int participant_index)
                : graph_scope_(
                      std::move(coordinator),
                      std::move(descriptor),
                      participant_index)
            {
            }

            /** @return Whether participant entry succeeded. */
            [[nodiscard]] bool ready() const noexcept
            {
                return graph_scope_.ready();
            }

            /** @return Participant-entry failure diagnostic, when present. */
            [[nodiscard]] const std::string &error() const noexcept
            {
                return graph_scope_.error();
            }

            /**
             * @brief Stamp wire identity and install the exact launch-edge arm.
             *
             * The callback captures this heap-owned lease.  The execution
             * engine retains the lease until execute() and finish() return, so
             * every callback copy is destroyed before the participant scope.
             *
             * @param input Chunk input passed directly to the forward engine.
             * @param expected_generation Request generation installed at root admission.
             * @param error Receives a binding or arming diagnostic.
             * @return True when the chunk is fully bound.
             */
            bool bind(
                ForwardInput &input,
                std::uint64_t expected_generation,
                std::string *error)
            {
                if (error)
                    error->clear();
                if (!graph_scope_.ready())
                    return reject(graph_scope_.error(), error);
                if (!graph_scope_.active())
                    return true;

                const auto &binding = graph_scope_.binding();
                if (expected_generation == 0 ||
                    expected_generation != binding.request_generation)
                {
                    return reject(
                        "ExpertOverlay prefill chunk generation disagrees "
                        "with the request generation installed at admission",
                        error);
                }

                input.moe_overlay_collective_generation_id =
                    binding.request_generation;
                input.moe_overlay_collective_step_id =
                    binding.descriptor.logical_step_id;
                input.moe_overlay_sequence_id = binding.sequence_id;
                input.moe_overlay_sequence_placement_epoch =
                    binding.descriptor.placement_epoch;
                input.moe_overlay_sequence_graph_ordinal =
                    binding.sequence_graph_ordinal;
                input.moe_overlay_sequence_graph_count =
                    binding.sequence_graph_count;

                if (!graph_scope_.ownsTicketAuthority())
                    return true;

                if (!input.device.is_gpu())
                {
                    return graph_scope_.armForHostExecution(error);
                }

                const DeviceId device = input.device;
                input.moe_overlay_graph_launch_dependency =
                    [this, device](
                        DeviceGraphExecutor::GraphExecutableLaunchPhase,
                        void *execution_stream) -> bool
                {
                    std::string arm_error;
                    if (!graph_scope_.armForExecutableLaunch(
                            execution_stream, &arm_error))
                    {
                        LOG_ERROR(
                            "[DeviceGraphOrchestrator] ExpertOverlay prefill "
                            "chunk could not arm its remote follower at the "
                            "executable launch edge on "
                            << device.toString() << ": " << arm_error);
                        return false;
                    }
                    return true;
                };
                return true;
            }

            /** @copydoc IPrefillChunkGraphSubmissionLease::finish */
            bool finish(
                bool execution_succeeded,
                std::string *error) override
            {
                return graph_scope_.finish(execution_succeeded, error);
            }

        private:
            /** @brief Store one diagnostic and reject the current operation. */
            static bool reject(
                const std::string &message,
                std::string *error)
            {
                if (error)
                {
                    *error = message.empty()
                                 ? "ExpertOverlay prefill chunk binding failed"
                                 : message;
                }
                return false;
            }

            /** Participant-local lifecycle and ticket authority. */
            MoEOverlayInferenceParticipantGraphScope graph_scope_;
        };
    }

    /**
     * @brief Persistent pinned host destination for compact stochastic outcomes.
     *
     * The host-visible compatibility path still needs to read compact outcome
     * metadata, but CUDA/HIP async D2H copies only stay non-blocking when the
     * destination is pinned.  This scratch lives for the orchestrator lifetime
     * so decode steps do not allocate or pageable-stage tiny outcome arrays.
     */
    struct DeviceGraphOrchestrator::PinnedHostScratch
    {
        IBackend *backend = nullptr;       ///< Backend that owns the pinned allocation API.
        int device_id = -1;                ///< Backend-local GPU ordinal used for freePinned().
        int request_capacity = 0;          ///< Maximum request rows represented by this scratch.
        size_t output_token_elements = 0;  ///< INT32 elements in output_tokens.
        size_t meta_elements = 0;          ///< INT32 elements in meta.
        int32_t *output_tokens = nullptr;  ///< Pinned INT32 [request, output-token stride].
        int *meta = nullptr;               ///< Pinned INT32 [optional local report, request metadata].
        PhysicalMemoryAllocationLease local_terminal_claim; ///< Exact whole combined pinned metadata destination; original token scratch is unchanged.

        PinnedHostScratch() = default;
        PinnedHostScratch(const PinnedHostScratch &) = delete;
        PinnedHostScratch &operator=(const PinnedHostScratch &) = delete;

        ~PinnedHostScratch()
        {
            release();
        }

        /**
         * @brief Allocate token and metadata buffers for the requested capacity.
         *
         * @return true when both buffers are pinned and ready for async D2H.
         */
        bool allocate(
            IBackend *owner_backend,
            int owner_device_id,
            int requests,
            int output_token_stride,
            int metadata_stride,
            std::shared_ptr<PhysicalMemoryAuthority> reportAuthority={},
            std::size_t terminalBytes=0)
        {
            release();
            if (!owner_backend || requests <= 0 || output_token_stride <= 0 ||
                metadata_stride <= 0)
                return false;

            backend = owner_backend;
            device_id = owner_device_id;
            request_capacity = requests;
            output_token_elements =
                static_cast<size_t>(requests) *
                static_cast<size_t>(output_token_stride);
            meta_elements =
                static_cast<size_t>(requests) *
                static_cast<size_t>(metadata_stride);

            if(terminalBytes){
                if(!reportAuthority || terminalBytes!=qwen4exp::local_hot::terminalObservationBytes(requests,metadata_stride))
                    throw std::invalid_argument("Pinned local terminal observation changed its exact complete cold declaration");
                local_terminal_claim=reportAuthority->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,terminalBytes);
                meta_elements=terminalBytes/sizeof(int);
            }
            output_tokens = static_cast<int32_t *>(
                backend->allocatePinned(
                    output_token_elements * sizeof(int32_t),
                    device_id));
            meta = static_cast<int *>(
                backend->allocatePinned(
                    meta_elements * sizeof(int),
                    device_id));
            if (!output_tokens || !meta)
            {
                release();
                return false;
            }
            return true;
        }

        /**
         * @brief Return true when the scratch can hold the requested copy shape.
         */
        bool canServe(int requests, int output_stride, int meta_stride) const
        {
            if (requests < 0 || output_stride < 0 || meta_stride < 0)
                return false;
            return requests <= request_capacity &&
                   static_cast<size_t>(requests) * static_cast<size_t>(output_stride) <= output_token_elements &&
                   static_cast<size_t>(requests) * static_cast<size_t>(meta_stride) <= meta_elements;
        }

        /**
         * @brief Release pinned memory through the backend that allocated it.
         */
        void release()
        {
            if (backend)
            {
                if (output_tokens)
                    backend->freePinned(output_tokens, device_id);
                if (meta)
                    backend->freePinned(meta, device_id);
            }
            local_terminal_claim={};
            backend = nullptr;
            device_id = -1;
            request_capacity = 0;
            output_token_elements = 0;
            meta_elements = 0;
            output_tokens = nullptr;
            meta = nullptr;
        }
    };

    /**
     * @brief Lifetime-owned pinned observer for HIP graph-dispatch tickets.
     *
     * The buffer is allocated once with graph workspace setup and is never used
     * as a transfer source.  That one-way type and ownership boundary is
     * intentional: host scheduling may observe a device decision, but it has no
     * API with which to publish or reconstruct inference state.
     */
    struct DeviceGraphOrchestrator::PinnedDispatchTicketScratch
    {
        IBackend *backend = nullptr;
        int device_id = -1;
        int request_capacity = 0;
        sampling_math::DeviceGenerationDispatchTicket *tickets = nullptr;

        PinnedDispatchTicketScratch() = default;
        PinnedDispatchTicketScratch(const PinnedDispatchTicketScratch &) =
            delete;
        PinnedDispatchTicketScratch &operator=(
            const PinnedDispatchTicketScratch &) = delete;

        ~PinnedDispatchTicketScratch()
        {
            release();
        }

        /** @brief Allocate one fixed ticket row per admitted request slot. */
        bool allocate(
            IBackend *owner_backend,
            int owner_device_id,
            int requests)
        {
            release();
            if (!owner_backend || requests <= 0)
                return false;

            backend = owner_backend;
            device_id = owner_device_id;
            request_capacity = requests;
            tickets = static_cast<
                sampling_math::DeviceGenerationDispatchTicket *>(
                backend->allocatePinned(
                    sizeof(sampling_math::DeviceGenerationDispatchTicket) *
                        static_cast<size_t>(requests),
                    device_id));
            if (!tickets)
            {
                release();
                return false;
            }
            return true;
        }

        /** @brief Return whether this allocation can observe the request set. */
        bool canServe(int requests) const noexcept
        {
            return tickets && requests > 0 && requests <= request_capacity;
        }

        /** @brief Release the pinned destination through its owning backend. */
        void release()
        {
            if (backend && tickets)
                backend->freePinned(tickets, device_id);
            backend = nullptr;
            device_id = -1;
            request_capacity = 0;
            tickets = nullptr;
        }
    };

    /**
     * @brief Lifetime-owned one-way observer for HIP MoE cadence tickets.
     *
     * Allocation is performed during graph-family setup.  The pointer is never
     * lent as an H2D source, which keeps the host scheduler structurally unable
     * to author controller state while deciding whether an RCCL graph is due.
     */
    struct DeviceGraphOrchestrator::
        PinnedMoERebalanceDispatchTicketScratch
    {
        IBackend *backend = nullptr;
        int device_id = -1;
        DeviceMoERebalanceDispatchTicket *ticket = nullptr;

        PinnedMoERebalanceDispatchTicketScratch() = default;
        PinnedMoERebalanceDispatchTicketScratch(
            const PinnedMoERebalanceDispatchTicketScratch &) = delete;
        PinnedMoERebalanceDispatchTicketScratch &operator=(
            const PinnedMoERebalanceDispatchTicketScratch &) = delete;

        /** @brief Release the pinned destination through its owning backend. */
        ~PinnedMoERebalanceDispatchTicketScratch()
        {
            release();
        }

        /** @brief Allocate the single fixed ticket destination. */
        bool allocate(IBackend *owner_backend, int owner_device_id)
        {
            release();
            if (!owner_backend || owner_device_id < 0)
                return false;

            backend = owner_backend;
            device_id = owner_device_id;
            ticket = static_cast<DeviceMoERebalanceDispatchTicket *>(
                backend->allocatePinned(
                    sizeof(DeviceMoERebalanceDispatchTicket),
                    device_id));
            if (!ticket)
            {
                release();
                return false;
            }
            return true;
        }

        /** @brief Return whether this observer owns a usable destination. */
        bool valid() const noexcept
        {
            return backend && device_id >= 0 && ticket;
        }

        /** @brief Release all ownership and clear the typed pointer. */
        void release()
        {
            if (backend && ticket)
                backend->freePinned(ticket, device_id);
            backend = nullptr;
            device_id = -1;
            ticket = nullptr;
        }
    };

    /**
     * @brief Persistent pure-device buffers used to serialize prefix archives.
     *
     * The staging allocations deliberately bypass TensorBase storage because a
     * normal FP32 tensor also allocates an equally large pageable host vector.
     * Prefix staging has no host representation: it is an explicit-stream
     * bridge between live device state and pinned RAM archive sections.
     */
    struct DeviceGraphOrchestrator::PrefixArchiveDeviceStaging
    {
        struct Slot
        {
            size_t capacity_bytes = 0;
            std::shared_ptr<void> allocation;
            std::unique_ptr<GpuTensorView> view;

            void *devicePtr() const
            {
                return allocation.get();
            }
        };

        IBackend *backend = nullptr;
        DeviceId device = DeviceId::invalid();
        Slot k;
        Slot v;
        Slot hybrid;
        Slot mtp_k;
        Slot mtp_v;

        PrefixArchiveDeviceStaging(IBackend *owner_backend, DeviceId owner_device)
            : backend(owner_backend),
              device(owner_device)
        {
        }

        /**
         * @brief Grow one slot outside the transfer hot path.
         */
        bool ensure(Slot &slot, size_t required_bytes)
        {
            if (required_bytes == 0)
                return true;
            if (slot.allocation &&
                slot.capacity_bytes >= required_bytes &&
                slot.view)
            {
                return true;
            }
            if (!backend || !device.is_gpu())
                return false;

            const int ordinal = device.gpu_ordinal();
            void *raw = backend->allocate(required_bytes, ordinal);
            if (!raw)
                return false;

            std::shared_ptr<void> owner(
                raw,
                [owner_backend = backend, ordinal](void *pointer)
                {
                    if (pointer)
                        owner_backend->free(pointer, ordinal);
                });
            const size_t fp32_columns =
                (required_bytes + sizeof(float) - 1) / sizeof(float);
            auto view = std::make_unique<GpuTensorView>(
                raw,
                1,
                fp32_columns,
                TensorType::FP32,
                device);

            slot.view = std::move(view);
            slot.allocation = std::move(owner);
            slot.capacity_bytes = required_bytes;
            return true;
        }

        Slot *slot(BufferId id)
        {
            switch (id)
            {
            case BufferId::PREFIX_K_STAGING:
                return &k;
            case BufferId::PREFIX_V_STAGING:
                return &v;
            case BufferId::PREFIX_HYBRID_STATE_STAGING:
                return &hybrid;
            case BufferId::PREFIX_MTP_K_STAGING:
                return &mtp_k;
            case BufferId::PREFIX_MTP_V_STAGING:
                return &mtp_v;
            default:
                return nullptr;
            }
        }
    };

} // namespace llaminar2
