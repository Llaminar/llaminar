/**
 * @file ForwardExecutionEngineInternal.h
 * @brief Shared internal graph-cache, stage-param, and snapshot helpers for ForwardExecutionEngine shards.
 */
#pragma once
/**
 * @file ForwardExecutionEngine.cpp
 * @brief Implementation of ForwardExecutionEngine
 *
 * Contains the forward graph execution logic extracted from
 * DeviceGraphOrchestrator::executeForward().
 * Every invocation owns one participant's graph and output stream. Pipeline
 * composition belongs to the rank/global orchestrator; there is no uncached
 * multi-device execution branch in this engine.
 * Prefill schedules preserve one admitted request bank and device KV cursor
 * across a retained bucket family; shape-specific materializer and shifted-MTP
 * identities must be complete before the first chunk is submitted.
 *
 * Split into:
 * - execute():          Entry point — signature computation, cache dispatch
 * - executeCacheHit():  Reuse cached graph (buffer update, PP copy, execution)
 * - executeCacheMiss(): Build new graph, execute, populate cache
 * - collectTimeline():  GPU stage timing collection and printing
 */

#include "ForwardExecutionEngine.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "execution/local_execution/graph/GraphCaptureStageActivity.h"
#include "execution/local_execution/orchestrators/PipelineForwardGraphEdges.h"
#include "PrefillBucketUtils.h"
#include "../graph/DeviceGraphCaptureController.h"
#include "../../compute_stages/ComputeStageFactory.h"
#include "../../compute_stages/stages/DecodePositionSnapshotStage.h"
#include "../../moe/MoEOverlayRetainedParentComposer.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/OpenMPTeamForkEvidence.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include <chrono>
#include <optional>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <unordered_set>


namespace llaminar2
{
    namespace
    {
        /** @brief Clone the original native budget recording before a follower's complete verifier parent.
         * Existing ExpertOverlay composition remains the body authority. The
         * temporary graph is a cold compiler definition, never an executable
         * or an independently submitted transaction. */
        bool composePipelineMaintenanceAdmission(DeviceGraphExecutor::DecodeCapturePolicy &policy,
            const PipelineForwardGraphEdges *edges, ForwardExecutionRole role,
            IWorkerGPUContext *gpu, void *stream)
        {
            if (!edges || !edges->requiresMaintenanceAdmission(role)) return false;
            if (!gpu || !stream || !policy.allow_cached_graph_replay)
                throw std::logic_error("Pipeline maintenance admission requires its original complete native graph compiler");
            const auto owner = edges->maintenanceAdmission();
            const auto member = edges->maintenanceParticipant();
            auto prior = std::move(policy.retained_parent_composer);
            policy.retained_parent_composer = [owner, member, prior = std::move(prior), gpu, stream](IGPUGraphCapture &destination,
                const ComputeGraph &graph, std::span<const DeviceGraphExecutor::GraphSegmentCache::RetainedCaptureUnitTemplateView> units) {
                std::unique_ptr<IGPUGraphCapture> body;
                std::vector<GPUOrderedTimelineStep> steps{{.name = "original pipeline maintenance budget", .capture = owner->recording(member)}};
                if (prior) {
                    body = gpu->createGraphCapture(stream);
                    if (!body || !prior(*body, graph, units)) return false;
                    steps.push_back({.name = "original complete verifier body", .capture = body.get()});
                } else {
                    for (const auto &unit : units) {
                        if (!unit.capture) return false;
                        steps.push_back({.name = "original verifier capture unit", .capture = unit.capture});
                    }
                }
                return !units.empty() && destination.buildOrderedTimelineTransaction(steps);
            };
            // A native minimum is inside this complete participant-local parent;
            // no eager prefix or host-count phase may substitute for it.
            policy.graph_replay_plan_policy = DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentComposition;
            policy.defer_final_sync = true;
            return true;
        }

        /// @brief Build a prefill graph-cache config from the cached debug environment.
        PrefillGraphConfig makePrefillGraphConfigFromEnv(
            const IForwardExecutionHost &host)
        {
            const auto &env = debugEnv();
            const auto resident_graph_rows = host.residentGraphRows();
            PrefillGraphConfig config;
            config.enabled = env.execution.gpu_graphs;
            config.minimum_padded_bucket_seq_len =
                effectivePrefillGraphMinimumPaddedBucketSeqLen(
                    env.execution.prefill_graph_min_seq,
                    resident_graph_rows);
            config.trace = env.execution.prefill_graph_trace;
            config.buckets_enabled = env.execution.prefill_graph_buckets;
            config.bucket_sizes = env.execution.prefill_graph_bucket_sizes;
            config.max_cached_entries = host.retainedExactPrefillCacheEntries().value_or(
                static_cast<size_t>(env.execution.prefill_graph_max_cached_buckets));
            return config;
        }

        /** @brief Select the common capture-cache stream from the actual graph execution policy.
         * @param envelope Graph-owned native transaction boundary.
         * @param policy Envelope-constrained host replay policy, including any installed retained parent.
         * @return True when prefill executes through the common native transaction cache.
         * Preparation and submission must use this same decision: an indivisible
         * device timeline does not require a heterogeneous-segmentation flag.
         */
        bool prefillUsesCommonGraphCache(GraphNativeCaptureEnvelope envelope,
            const DeviceGraphExecutor::DecodeCapturePolicy &policy)
        {
            return policy.heterogeneous_segmented_enabled ||
                DeviceGraphExecutor::isRetainedParentPlanPolicy(policy.graph_replay_plan_policy) ||
                requiresSingleNativeExecutable(envelope) || requiresHeterogeneousTicketSegmentation(envelope);
        }

        /// @brief Return the real-token count for an input, falling back to its execution length.
        int effectiveRealSeqLen(const ForwardInput &input)
        {
            return input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
        }

        /// @brief Return the fixed bucket length for an input, falling back to its execution length.
        int effectiveBucketSeqLen(const ForwardInput &input)
        {
            return input.bucket_seq_len > 0 ? input.bucket_seq_len : input.seq_len;
        }

        /// @brief Return true when this execution uses a padded fixed bucket.
        bool isPaddedBucketExecution(const ForwardInput &input)
        {
            const int real_seq_len = effectiveRealSeqLen(input);
            const int bucket_seq_len = effectiveBucketSeqLen(input);
            return real_seq_len > 0 && bucket_seq_len > 0 && real_seq_len < bucket_seq_len;
        }

        /**
         * @brief Restore the host's diagnostic namespace after one chunk attempt.
         *
         * Graph snapshot callbacks execute synchronously at their explicit
         * post-launch diagnostic boundary, but the namespace must still be
         * restored on every early return and exception. This small RAII owner
         * keeps that diagnostic concern separate from the graph's live-state
         * lifecycle and makes an incomplete schedule unable to contaminate the
         * next request's artifacts.
         */
        class ScopedPrefillChunkSnapshotDiagnostics
        {
        public:
            /**
             * @brief Enter the host namespace for @p chunk.
             * @param host Graph-execution host that owns optional snapshots.
             * @param chunk Scheduler-owned chunk geometry.
             */
            ScopedPrefillChunkSnapshotDiagnostics(
                IForwardExecutionHost &host,
                const PrefillChunkPlan &chunk) noexcept
                : host_(host), chunk_(chunk)
            {
                host_.beginPrefillChunkSnapshotDiagnostics(chunk_);
            }

            /** @brief Restore the prior diagnostic namespace. */
            ~ScopedPrefillChunkSnapshotDiagnostics()
            {
                host_.endPrefillChunkSnapshotDiagnostics(chunk_);
            }

            ScopedPrefillChunkSnapshotDiagnostics(
                const ScopedPrefillChunkSnapshotDiagnostics &) = delete;
            ScopedPrefillChunkSnapshotDiagnostics &operator=(
                const ScopedPrefillChunkSnapshotDiagnostics &) = delete;

        private:
            IForwardExecutionHost &host_;
            const PrefillChunkPlan &chunk_;
        };

        /**
         * @brief Own one forward graph's admitted live-state mailbox read.
         *
         * `prepareLiveStateForForwardGraphExecution()` can retain a shared
         * mailbox-admission lease inside the host until the engine publishes a
         * completion event. Every return after that prelude must therefore
         * retire the read, including failures while preparing token rows,
         * verifier metadata, dynamic parameters, or the execution rendezvous.
         * Keeping that obligation in a lexical owner prevents prefix rollback
         * from requesting the exclusive mailbox lease while the same thread
         * still owns an abandoned reader lease.
         *
         * The normal post-launch path supplies the actual producer stream so
         * the transaction also verifies that graph execution did not migrate
         * away from the stream admitted by the prelude. Before-launch failure
         * cleanup uses the admitted stream itself; recording an event there is
         * conservative and preserves all prelude work already enqueued on it.
         */
        class ScopedForwardLiveStateRead final
        {
        public:
            /**
             * @brief Arm completion ownership after a successful host prelude.
             * @param host Host that owns the pending mailbox admission.
             * @param input Forward invocation associated with the admission.
             * @param admission_stream Exact stream admitted by the prelude.
             * @param device Device that owns @p admission_stream.
             */
            ScopedForwardLiveStateRead(
                IForwardExecutionHost &host,
                const ForwardInput &input,
                void *admission_stream,
                DeviceId device) noexcept
                : host_(host),
                  input_(input),
                  admission_stream_(admission_stream),
                  device_(device)
            {
            }

            /**
             * @brief Retire a prelude whose caller left before explicit completion.
             *
             * A failed completion would leave an exclusive rollback or reset
             * permanently blocked, so it is a fatal lifecycle violation rather
             * than a recoverable inference error.
             */
            ~ScopedForwardLiveStateRead() noexcept
            {
                if (active_)
                    completeOrTerminate(admission_stream_, "scope_exit");
            }

            ScopedForwardLiveStateRead(
                const ScopedForwardLiveStateRead &) = delete;
            ScopedForwardLiveStateRead &operator=(
                const ScopedForwardLiveStateRead &) = delete;

            /**
             * @brief Publish the graph's mailbox-read completion explicitly.
             * @param producer_stream Exact stream which consumed live state.
             *
             * This must be called immediately after graph execution, whether
             * the executor reported success or failure. Stream identity is
             * checked before the host records its event because substituting a
             * different stream would publish readiness ahead of real readers.
             */
            void complete(void *producer_stream) noexcept
            {
                completeOrTerminate(producer_stream, "post_launch");
            }

        private:
            /**
             * @brief Complete once or terminate on a broken ownership edge.
             * @param producer_stream Stream on which the completion event belongs.
             * @param boundary Diagnostic lifecycle boundary requesting completion.
             */
            void completeOrTerminate(
                void *producer_stream,
                const char *boundary) noexcept
            {
                if (!active_)
                    return;
                if (producer_stream != admission_stream_)
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] Live-state read changed streams between admission and completion"
                        << " admitted_stream=" << admission_stream_
                        << " producer_stream=" << producer_stream
                        << " device=" << device_.toString()
                        << " boundary=" << boundary);
                    std::terminate();
                }

                try
                {
                    if (!host_.completeLiveStateForForwardGraphExecution(
                            input_, producer_stream, device_))
                    {
                        LOG_ERROR(
                            "[ForwardExecutionEngine] Failed to publish forward graph live-state read completion"
                            << " device=" << device_.toString()
                            << " stream=" << producer_stream
                            << " boundary=" << boundary);
                        std::terminate();
                    }
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] Exception while publishing forward graph live-state read completion"
                        << " device=" << device_.toString()
                        << " stream=" << producer_stream
                        << " boundary=" << boundary
                        << " error=" << e.what());
                    std::terminate();
                }
                catch (...)
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] Unknown exception while publishing forward graph live-state read completion"
                        << " device=" << device_.toString()
                        << " stream=" << producer_stream
                        << " boundary=" << boundary);
                    std::terminate();
                }
                active_ = false;
            }

            IForwardExecutionHost &host_;
            const ForwardInput &input_;
            void *admission_stream_ = nullptr;
            DeviceId device_;
            bool active_ = true;
        };

        /**
         * @brief Map a typed forward invocation onto workspace graph semantics.
         *
         * `is_decode` remains relevant for ordinary MainInference because that
         * public role spans prompt prefill and serial decode. Internal MTP roles
         * are already exact: a condition is one decode-equivalent row and a
         * grouped verifier is compact grouped decode regardless of M or position.
         */
        WorkspaceGraphParticipantRole workspaceParticipantRole(
            ForwardExecutionRole execution_role,
            bool is_decode)
        {
            switch (execution_role)
            {
            case ForwardExecutionRole::GroupedMTPVerifier:
                return WorkspaceGraphParticipantRole::GroupedVerifier;
            case ForwardExecutionRole::MTPCondition:
                return WorkspaceGraphParticipantRole::MTPCondition;
            case ForwardExecutionRole::MainInference:
                return is_decode
                           ? WorkspaceGraphParticipantRole::Decode
                           : WorkspaceGraphParticipantRole::Prefill;
            }
            throw std::logic_error(
                "Forward execution has no workspace participant role");
        }

        /**
         * @brief Select the physical family policy for one typed participant.
         *
         * Only ordinary prefill owns the largest-row envelope. Decode and grouped
         * verification use exact compact geometry within the already-published
         * serial family allocation.
         */
        WorkspaceGraphFamilyPolicy workspaceFamilyPolicy(
            WorkspaceGraphParticipantRole role)
        {
            return role == WorkspaceGraphParticipantRole::Prefill
                       ? WorkspaceGraphFamilyPolicy::
                             SerialDeviceFamilyLargestParticipant
                       : WorkspaceGraphFamilyPolicy::
                             SerialDeviceFamilyExactParticipant;
        }

        /**
         * @brief Publish the ordering owner for one successful forward call.
         *
         * This helper deliberately writes the caller-owned ForwardOutput. The
         * execution-scoped record is the unambiguous handoff token for an
         * immediate consumer such as shifted-MTP prefill cache construction.
         * Longer-lived consumers use the orchestrator-owned durable timeline
         * event published before this producer stream can be invalidated.
         */
        void publishForwardOutputProvenance(
            ForwardOutput &output,
            DeviceId device,
            void *stream,
            ForwardExecutionRole execution_role,
            bool is_decode,
            bool all_position_logits,
            ForwardCompletionScope completion_scope,
            int graph_seq_len,
            int graph_batch_size)
        {
            output.execution = {
                .valid = !device.is_gpu() || stream != nullptr,
                .device = device,
                .stream = stream,
                .execution_role = execution_role,
                .is_decode = is_decode,
                .all_position_logits = all_position_logits,
                .completion_scope = completion_scope,
                .graph_seq_len = graph_seq_len,
                .graph_batch_size = graph_batch_size,
            };
        }

        /**
         * @brief Resolve the device that actually produced a forward output.
         *
         * `TensorBase::home_device()` is immutable creation metadata. Activation
         * storage can be created on the host and subsequently become
         * device-authoritative, so using the home device here loses the exact
         * GPU stream/event boundary and permits stale host reads. Prefer the
         * tensor's coherence owner, then its live GPU allocation, and retain
         * the graph execution device when neither supplies stronger evidence.
         *
         * @param output Successful forward result whose hidden/logits storage
         *        carries the latest coherence state.
         * @param execution_device Device on which the graph was executed.
         * @return Runtime producer device suitable for stream provenance.
         */
        DeviceId resolveForwardOutputProducerDevice(
            const ForwardOutput &output,
            DeviceId execution_device)
        {
            TensorBase *tensor = output.hidden ? output.hidden : output.logits;
            if (!tensor)
                return execution_device;

            if (const auto authoritative = tensor->getAuthoritativeDevice();
                authoritative.has_value() && authoritative->is_valid())
            {
                return *authoritative;
            }
            if (const auto current = tensor->current_device();
                current.has_value() && current->is_valid())
            {
                return *current;
            }

            const DeviceId home = tensor->home_device();
            return home.is_gpu() ? home : execution_device;
        }

        /**
         * @brief Install a host-owned worker resolver for one engine call.
         *
         * DeviceGraphExecutor also supports standalone use, where it resolves
         * the process GPU pool directly. ForwardExecutionEngine has a stronger
         * host boundary, so every nested eager/captured executor operation must
         * use the same host-selected worker. Restoring the prior resolver on all
         * returns prevents a short-lived test host from remaining captured by a
         * long-lived executor.
         */
        class ScopedWorkerGPUContextResolver
        {
        public:
            ScopedWorkerGPUContextResolver(
                DeviceGraphExecutor &executor,
                IForwardExecutionHost &host,
                DeviceId device)
                : executor_(executor),
                  previous_resolver_(executor.config().worker_gpu_context_resolver),
                  previous_uses_process_pool_(
                      executor.config().worker_gpu_context_uses_process_pool)
            {
                executor_.setWorkerGPUContextResolver(
                    [&host](DeviceId requested_device)
                    {
                        return host.getWorkerGPUContext(requested_device);
                    },
                    host.workerGPUContextUsesProcessPool(device));
            }

            ~ScopedWorkerGPUContextResolver()
            {
                executor_.setWorkerGPUContextResolver(
                    std::move(previous_resolver_),
                    previous_uses_process_pool_);
            }

            ScopedWorkerGPUContextResolver(const ScopedWorkerGPUContextResolver &) = delete;
            ScopedWorkerGPUContextResolver &operator=(const ScopedWorkerGPUContextResolver &) = delete;

        private:
            DeviceGraphExecutor &executor_;
            std::function<IWorkerGPUContext *(DeviceId)> previous_resolver_;
            bool previous_uses_process_pool_ = false;
        };

        /// @brief Return the absolute logical-token offset for prefill-style inputs.
        ///
        /// `token_offset` is the first-class request-range owner for prepared
        /// prefill chunks. Raw server/orchestrator inputs historically carried
        /// only `position_offset`; treating that as the logical-token offset
        /// keeps restored-prefix suffix prefill aligned with RoPE, KV append,
        /// and replay metadata until every caller is migrated to an explicit
        /// non-optional request range.
        int effectiveTokenOffset(const ForwardInput &input)
        {
            return input.token_offset != 0 ? input.token_offset : input.position_offset;
        }

        /// @brief Build the fixed-bucket ForwardInput view backed by a runtime plan.
        ForwardInput makeBucketedPrefillInput(
            const ForwardInput &base_input,
            const ForwardExecutionEngine::PrefillChunkRuntimePlan &plan)
        {
            ForwardInput chunk_input = base_input;
            /*
             * Entering this typed adapter is itself an explicit prefill
             * admission. A terminal one-row chunk can inherit scalar geometry
             * from its request envelope, but it must never inherit a decode
             * phase selected for some unrelated direct invocation.
             */
            chunk_input.execution_phase = ForwardExecutionPhase::Prefill;
            switch (plan.chunk.token_authority)
            {
            case PrefillChunkTokenAuthority::HostPaddedRows:
                chunk_input.token_ids = plan.chunk.token_ids.data();
                chunk_input.token_ids_device = nullptr;
                break;
            case PrefillChunkTokenAuthority::DeviceResidentRows:
                /*
                 * Request admission has already initialized every physical row
                 * through the selected bucket. Keeping the host pointer null is
                 * part of the device-ownership contract, not an absent-input
                 * fallback: the graph embeds the stable arena pointer below.
                 */
                chunk_input.token_ids = nullptr;
                chunk_input.token_ids_device = base_input.token_ids_device;
                break;
            }
            chunk_input.position_policy = plan.position_policy;
            if (plan.position_policy == ForwardPositionPolicy::ContiguousOffset)
            {
                // The absolute range is represented by position_offset. Keeping
                // both pointers null prevents graph construction or replay from
                // recording a mutable host-row upload for this device-owned path.
                chunk_input.position_ids = nullptr;
                chunk_input.position_ids_device = nullptr;
            }
            else
            {
                if (base_input.position_ids_device)
                {
                    chunk_input.position_ids = nullptr;
                    chunk_input.position_ids_device =
                        base_input.position_ids_device;
                }
                else
                {
                    chunk_input.position_ids = plan.chunk.position_ids.data();
                    chunk_input.position_ids_device = nullptr;
                }
            }
            chunk_input.seq_len = plan.chunk.bucket_seq_len;
            chunk_input.real_seq_len = plan.chunk.real_count;
            chunk_input.bucket_seq_len = plan.chunk.bucket_seq_len;
            chunk_input.token_offset = plan.chunk.token_offset;
            chunk_input.position_offset = plan.chunk.token_offset;
            chunk_input.prefill_chunk_index = plan.chunk_index;
            if (base_input.sparse_query_prefill_proof) {
                chunk_input.sparse_query_prefill_proof = base_input.sparse_query_prefill_proof->chunk(
                    plan.chunk.token_offset, plan.chunk.real_count);
                chunk_input.sparse_query_execution_program = chunk_input.sparse_query_prefill_proof->program(
                    qwen4exp::QwenSparseFirstRankingLogicalRow);
            }
            /*
             * Sparse MoE packets carry only live rows, but the matching root
             * and participant graphs must still name this padded bucket by its
             * shared logical start. Graph-capture history is never part of the
             * transaction key.
             */
            chunk_input.moe_overlay_collective_step_id =
                static_cast<uint64_t>(plan.chunk.token_offset);
            return chunk_input;
        }

        PrefillChunkPlan makeMaintenanceChunkPlan(
            const ForwardExecutionEngine::PrefillChunkRuntimePlan &plan)
        {
            PrefillChunkPlan chunk;
            chunk.token_offset = plan.chunk.token_offset;
            chunk.real_count = plan.chunk.real_count;
            chunk.bucket_seq_len = plan.chunk.bucket_seq_len;
            chunk.chunk_index = plan.chunk_index;
            chunk.rebalance_allowed_after = plan.rebalance_allowed_after;
            chunk.rebalance_required_after = plan.rebalance_required_after;
            return chunk;
        }

        /// @brief Build the fixed-bucket replay metadata consumed by row-select,
        ///        LM-head, and KV-cache append stages.
        IComputeStage::PrefillReplayParams makePrefillReplayParams(const ForwardInput &input)
        {
            return IComputeStage::PrefillReplayParams{
                effectiveRealSeqLen(input),
                effectiveBucketSeqLen(input),
                effectiveTokenOffset(input)};
        }

        /// @brief Collect stages that need real-token metadata for padded prefill execution.
        std::vector<IComputeStage *> collectPrefillReplayParamStages(ComputeGraph &graph)
        {
            std::vector<IComputeStage *> stages;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                ComputeNode *node = graph.getNode(node_name);
                if (node && node->stage && node->stage->hasPrefillReplayParams())
                    stages.push_back(node->stage.get());
            }
            return stages;
        }

        /// @brief Push real-token metadata to fixed-bucket stages before any prefill execution path.
        void updatePrefillReplayParamStages(
            const ForwardInput &input,
            const std::vector<IComputeStage *> &stages)
        {
            const auto replay_params = makePrefillReplayParams(input);
            for (auto *stage : stages)
            {
                if (stage)
                    stage->updatePrefillReplayParams(replay_params);
            }
        }

        /**
         * @brief Collect the stages whose launch inputs change without rebuilding a graph.
         *
         * Decode setup, first execution, and replay must prepare exactly the same
         * stage set. Keeping discovery here prevents setup-only capture from
         * accidentally omitting an input producer merely because it does not run
         * the request-scoped live-state prelude.
         *
         * @param graph Stable compute graph whose stages will be captured or run.
         * @return Execution-order-preserving list of dynamic-input consumers.
         */
        std::vector<IComputeStage *> collectDynamicParamStages(
            ComputeGraph &graph)
        {
            std::vector<IComputeStage *> stages;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                ComputeNode *node = graph.getNode(node_name);
                if (node && node->stage && node->stage->hasDynamicParams())
                    stages.push_back(node->stage.get());
            }
            return stages;
        }

        /**
         * @brief Publish one forward invocation's dynamic inputs to graph stages.
         *
         * The caller binds every GPU stage to its exact non-null execution stream
         * before entering this helper. Embedding uses that stream to preload host
         * token rows outside native capture; RoPE consumes the explicit host or
         * device position authority; the remaining stages stamp their stable
         * device parameter buffers. The same arithmetic-free prelude is therefore
         * valid for setup materialization, first execution, and cached replay.
         *
         * @param input Typed token/position geometry for the upcoming submission.
         * @param stages Dynamic stages collected in graph execution order.
         * @param host_position_ids Host position authority, or null when positions
         *        are device-owned or represented by a contiguous scalar offset.
         * @param boundary Short diagnostic name for the calling lifecycle edge.
         * @return True after every stage accepted the typed position authority.
         */
        bool updateDynamicParamStages(
            const ForwardInput &input,
            const std::vector<IComputeStage *> &stages,
            const int *host_position_ids,
            const char *boundary)
        {
            const int position_row_count = forwardPositionRowCount(input);
            if ((input.position_ids_device || host_position_ids) &&
                position_row_count <= 0)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] "
                    << (boundary ? boundary : "dynamic input preparation")
                    << " received invalid flattened position geometry: batch="
                    << input.batch_size << " seq_len=" << input.seq_len);
                return false;
            }

            for (auto *stage : stages)
            {
                if (!stage)
                    continue;

                if (input.position_ids_device)
                {
                    if (!stage->supportsDeviceResidentDynamicPositionReplay())
                    {
                        LOG_ERROR(
                            "[ForwardExecutionEngine] Stage "
                            << stage->name() << " cannot consume device-resident "
                            << "positions during "
                            << (boundary ? boundary
                                         : "dynamic input preparation"));
                        return false;
                    }
                    stage->updateDynamicDevicePositionIds(
                        input.position_ids_device,
                        position_row_count);
                    /*
                     * RoPE consumes the explicit device rows directly. Calling
                     * its scalar update afterward would clear that pointer. Other
                     * stages use the device-position hook as an ownership marker
                     * and still need their non-position launch values stamped.
                     */
                    if (stage->type() == ComputeStageType::ROPE)
                        continue;
                }

                stage->updateDynamicParams(
                    input.position_offset,
                    input.seq_len);
                if (host_position_ids)
                {
                    stage->updateDynamicPositionIds(
                        host_position_ids,
                        position_row_count);
                }
            }
            return true;
        }

        /** @brief Build the explicit wire identity for one MoE overlay graph execution. */
        IComputeStage::MoEOverlayCollectiveRuntimeParams
        makeMoEOverlayCollectiveRuntimeParams(const ForwardInput &input)
        {
            return IComputeStage::MoEOverlayCollectiveRuntimeParams{
                .generation_id =
                    input.moe_overlay_collective_generation_id,
                .step_id = input.moe_overlay_collective_step_id,
                .execution_semantics =
                    input.execution_role ==
                            ForwardExecutionRole::GroupedMTPVerifier
                        ? IComputeStage::MoEOverlayCollectiveRuntimeParams::
                              ExecutionSemantics::GroupedVerifier
                        : (input.execution_phase ==
                                   ForwardExecutionPhase::Prefill
                               ? IComputeStage::
                                     MoEOverlayCollectiveRuntimeParams::
                                         ExecutionSemantics::Prefill
                               : IComputeStage::
                                     MoEOverlayCollectiveRuntimeParams::
                                         ExecutionSemantics::Decode),
                .mtp_depth = input.moe_overlay_mtp_depth,
                .placement_epoch =
                    input.moe_overlay_sequence_placement_epoch,
            };
        }

        /** @brief Collect only sparse stages that require distributed wire identity. */
        std::vector<IComputeStage *> collectMoEOverlayCollectiveRuntimeStages(
            ComputeGraph &graph)
        {
            std::vector<IComputeStage *> stages;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                ComputeNode *node = graph.getNode(node_name);
                if (node && node->stage &&
                    node->stage->hasMoEOverlayCollectiveRuntimeParams())
                {
                    stages.push_back(node->stage.get());
                }
            }
            return stages;
        }

        /** @brief Stamp sparse graph boundaries before cold execution or cached replay. */
        void updateMoEOverlayCollectiveRuntimeStages(
            const ForwardInput &input,
            const std::vector<IComputeStage *> &stages)
        {
            const auto runtime_params =
                makeMoEOverlayCollectiveRuntimeParams(input);
            for (auto *stage : stages)
            {
                if (stage)
                    stage->updateMoEOverlayCollectiveRuntimeParams(
                        runtime_params);
            }
        }

        /**
         * @brief Emit one production-evidence record for a completed sparse graph execution.
         *
         * The counter is intentionally outside the stage loop: one graph
         * invocation can contain many layer-local dispatch/return boundaries.
         * Stable counters retain phase and geometry totals, while the ordered
         * sequence row folds request generation and logical step into bounded
         * evidence that can be compared with the remote participant. Callers invoke this
         * only after the graph result and its live-state handoff have both
         * succeeded: cache lookup is not execution evidence, because the
         * first use of an atomic GPU graph deliberately transitions from a
         * cache miss into the cache-hit execution path.
         */
        void recordMoEOverlayCollectiveTransaction(
            const ForwardInput &input,
            const std::vector<IComputeStage *> &stages,
            const char *graph_path)
        {
            if (stages.empty())
                return;

            const auto runtime_params =
                makeMoEOverlayCollectiveRuntimeParams(input);
            if (!runtime_params.valid())
                return;

            const char *phase =
                input.execution_phase == ForwardExecutionPhase::Decode
                    ? "decode"
                    : "prefill";
            const int logical_rows =
                input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
            PerfStatsCollector::recordOrderedSequenceStep(
                "forward_graph",
                "moe_overlay_collective_transaction_sequence",
                {runtime_params.generation_id,
                 runtime_params.step_id,
                 static_cast<uint64_t>(logical_rows),
                 static_cast<uint64_t>(input.seq_len)},
                phase,
                input.device.to_string(),
                {{"role", "continuation_graph"},
                 {"identity_source", "orchestration_request_and_chunk"},
                 {"logical_step_semantics", "monotonic_transaction"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                "moe_overlay_collective_transaction",
                1.0,
                phase,
                input.device.to_string(),
                {{"role", "continuation_graph"},
                 {"identity_source", "orchestration_request_and_chunk"},
                 {"logical_step_semantics", "monotonic_transaction"},
                 {"logical_rows", std::to_string(logical_rows)},
                 {"physical_rows", std::to_string(input.seq_len)},
                 {"graph_path", graph_path ? graph_path : "unknown"},
                 {"sparse_stage_count", std::to_string(stages.size())}});
        }

        std::string boolTag(bool value)
        {
            return value ? "true" : "false";
        }

        const char *prefillGraphPhaseName(PrefillGraphPhase phase)
        {
            switch (phase)
            {
            case PrefillGraphPhase::Disabled:
                return "disabled";
            case PrefillGraphPhase::Cold:
                return "cold";
            case PrefillGraphPhase::Initialized:
                return "initialized";
            case PrefillGraphPhase::Warmup:
                return "warmup";
            case PrefillGraphPhase::Capturing:
                return "capturing";
            case PrefillGraphPhase::Ready:
                return "ready";
            }
            return "unknown";
        }

        /**
         * @brief Count native nodes owned by the segmented/retained executable.
         *
         * Heterogeneous and retained-parent prefill deliberately bypass
         * PrefillGraphCache and use the same GraphSegmentCache authority as
         * captured decode. A retained parent owns one composed executable;
         * ordinary segmentation owns one native executable per capturable
         * segment. Reporting either through the monolithic cache would make a
         * live graph appear cold and is therefore forbidden.
         */
        size_t capturedSegmentGraphNodeCount(
            const DeviceGraphExecutor::GraphSegmentCache &cache)
        {
            if (cache.retained_parent_capture)
                return cache.retained_parent_capture->nodeCount();

            size_t nodes = 0;
            for (const auto &segment : cache.segments)
            {
                if (segment.capture)
                    nodes += segment.capture->nodeCount();
            }
            return nodes;
        }

        /**
         * @brief Project the authoritative segmented graph lifecycle into the
         *        backend-neutral prefill probe.
         *
         * The probe is consumed by both server readiness and benchmark gates.
         * It must describe the executable that actually runs, independently of
         * whether that executable is monolithic, segmented at a declared
         * heterogeneous boundary, or composed into one retained parent.
         */
        void applySegmentedPrefillGraphSnapshot(
            const DeviceGraphExecutor::GraphSegmentCache &cache,
            ForwardExecutionEngine::PrefillGraphCacheSnapshot *snapshot)
        {
            if (!snapshot)
                return;

            snapshot->prefill_cache_initialized = true;
            snapshot->cache_size = cache.initialized ? 1u : 0u;
            snapshot->node_count = capturedSegmentGraphNodeCount(cache);
            snapshot->capture_count = cache.successful_capture_count;
            snapshot->replay_count = static_cast<int>(std::min<uint64_t>(
                cache.successful_submission_count,
                static_cast<uint64_t>(std::numeric_limits<int>::max())));

            if (!cache.initialized)
            {
                snapshot->phase = PrefillGraphPhase::Cold;
                return;
            }

            switch (cache.executable_submission_state)
            {
            case DeviceGraphExecutor::GraphSegmentCache::
                ExecutableSubmissionState::MaterializedUnlaunched:
                snapshot->phase = PrefillGraphPhase::Initialized;
                snapshot->initialized_count = 1;
                return;
            case DeviceGraphExecutor::GraphSegmentCache::
                ExecutableSubmissionState::ReplayReady:
                snapshot->phase = PrefillGraphPhase::Ready;
                return;
            case DeviceGraphExecutor::GraphSegmentCache::
                ExecutableSubmissionState::Empty:
                snapshot->phase = PrefillGraphPhase::Cold;
                return;
            }
        }

        PrefillGraphCacheKey makePrefillGraphKey(
            const ForwardInput &input,
            const IForwardExecutionHost &host)
        {
            PrefillGraphCacheKey key;
            key.seq_len = input.seq_len;
            key.device_id = input.device;
            key.domain_id = host.prefillGraphDomainId();
            key.participant_id = host.prefillGraphParticipantId();
            key.placement_epoch = host.moePlacementEpoch();
            key.topology_signature = host.prefillGraphTopologySignature();
            return key;
        }

        PrefillGraphExecutionObservation makePrefillGraphObservation(
            const ForwardInput &input,
            const PrefillGraphCacheKey &key,
            const char *capture_phase,
            const std::string &recapture_reason,
            const std::string &reject_stage_name = std::string(),
            const std::string &reject_stage_type = std::string())
        {
            const int real_count = effectiveRealSeqLen(input);
            const int bucket_len = effectiveBucketSeqLen(input);
            PrefillGraphExecutionObservation observation;
            observation.valid = true;
            observation.chunk_index = input.prefill_chunk_index;
            observation.bucket_seq_len = bucket_len;
            observation.real_token_start = effectiveTokenOffset(input);
            observation.real_token_count = real_count;
            observation.real_token_end = observation.real_token_start + real_count;
            observation.domain_id = key.domain_id;
            observation.participant_id = key.participant_id;
            observation.placement_epoch = key.placement_epoch;
            observation.topology_signature = key.topology_signature;
            observation.capture_phase = capture_phase ? capture_phase : "unknown";
            observation.recapture_reason = recapture_reason.empty() ? "none" : recapture_reason;
            observation.reject_stage_name = reject_stage_name;
            observation.reject_stage_type = reject_stage_type;
            return observation;
        }

        PerfStatsCollector::Tags prefillGraphObservationTags(
            const PrefillGraphExecutionObservation &observation,
            const char *cache_phase)
        {
            // Exact graph identity and bounded live geometry remain dimensions.
            // Request chunk positions are folded into the publication witness.
            auto tags = PerfStatsCollector::Tags{
                {"capture_phase", observation.capture_phase},
                {"cache_phase", cache_phase ? cache_phase : "unknown"},
                {"bucket_seq_len", std::to_string(observation.bucket_seq_len)},
                {"real_token_count", std::to_string(observation.real_token_count)},
                {"domain_id", observation.domain_id},
                {"participant_id", std::to_string(observation.participant_id)},
                {"placement_epoch", std::to_string(observation.placement_epoch)},
                {"topology_signature", std::to_string(observation.topology_signature)},
                {"recapture_reason", observation.recapture_reason}};
            if (!observation.reject_stage_name.empty())
                tags.emplace("reject_stage_name", observation.reject_stage_name);
            if (!observation.reject_stage_type.empty())
                tags.emplace("reject_stage_type", observation.reject_stage_type);
            return tags;
        }

        void publishPrefillGraphObservation(
            ForwardGraphCache &forward_cache,
            const ForwardInput &input,
            const PrefillGraphCacheKey &key,
            PrefillGraphPhase cache_phase,
            const char *capture_phase,
            const std::string &recapture_reason,
            const std::string &reject_stage_name = std::string(),
            const std::string &reject_stage_type = std::string())
        {
            auto observation = makePrefillGraphObservation(
                input,
                key,
                capture_phase,
                recapture_reason,
                reject_stage_name,
                reject_stage_type);
            forward_cache.last_prefill_graph_observation = observation;

            PerfStatsCollector::addCounterWithSequence(
                "forward_graph",
                "prefill_graph_lifecycle",
                1.0,
                {static_cast<uint64_t>(observation.chunk_index),
                 static_cast<uint64_t>(observation.real_token_start),
                 static_cast<uint64_t>(observation.real_token_end)},
                "prefill",
                input.device.toString(),
                prefillGraphObservationTags(observation, prefillGraphPhaseName(cache_phase)));
            PerfStatsCollector::addCounterWithSequence(
                "forward_graph",
                "prefill_graph_phase",
                1.0,
                {static_cast<uint64_t>(observation.chunk_index),
                 static_cast<uint64_t>(observation.real_token_start),
                 static_cast<uint64_t>(observation.real_token_end)},
                "prefill",
                input.device.toString(),
                prefillGraphObservationTags(observation, prefillGraphPhaseName(cache_phase)));
        }

        std::string forwardGraphPerfContext(const ForwardGraphSignature &signature)
        {
            if (!signature.decode)
                return signature.is_bucketed_prefill ? "prefill_bucket" : "prefill";
            if (signature.live_mtp_request_batch_condition)
                return "main_condition_batch";
            return signature.all_position_logits ? "main_verifier" : "main_decode";
        }

        /**
         * @brief Install one signature-derived snapshot reuse proof on both caches.
         *
         * Full-graph and explicitly admitted segmented capture are alternative
         * lowering forms of the same forward signature. They must therefore
         * name the same diagnostic arena namespace before either form begins
         * descriptor discovery or native capture.
         */
        void bindGraphSnapshotArenaReusePolicy(
            ForwardGraphCache &cache,
            const ForwardGraphSignature &signature)
        {
            const auto policy =
                graphSnapshotArenaReusePolicyForSignature(signature);
            cache.snapshot_manifest.bindStorageReusePolicy(policy);
            cache.segment_cache.snapshot_manifest.bindStorageReusePolicy(
                policy);
        }

        PerfStatsCollector::Tags forwardCacheLookupTags(
            const ForwardGraphSignature &signature,
            const char *result)
        {
            return {
                {"context", forwardGraphPerfContext(signature)},
                {"result", result},
                {"seq_len", std::to_string(signature.seq_len)},
                {"decode_has_history", boolTag(signature.decode_has_history)},
                {"all_position_logits", boolTag(signature.all_position_logits)},
                {"live_mtp_request_batch_condition",
                 boolTag(signature.live_mtp_request_batch_condition)},
                {"all_position_logit_rows", std::to_string(signature.all_position_logit_rows)},
                {"uses_device_token_ids", boolTag(signature.usesDeviceTokenIds())},
                {"uses_device_position_ids", boolTag(signature.usesDevicePositionIds())},
                {"uses_device_sequence_lengths", boolTag(signature.usesDeviceSequenceLengths())},
                {"moe_placement_epoch", std::to_string(signature.moe_placement_epoch)}};
        }

        /// @brief Run the full prefill graph preflight for a fixed-bucket input.
        PrefillGraphRejectReason preflightPrefillGraph(
            const PrefillGraphCache &cache,
            const ComputeGraph &graph,
            const PrefillGraphCacheKey &key,
            const std::unordered_set<std::string> &collective_nodes,
            const ForwardInput &input,
            bool snapshots_active,
            bool moe_rebalancing_active,
            PrefillGraphPreflightMode mode = PrefillGraphPreflightMode::Default,
            bool collectives_graph_capturable = false,
            bool heterogeneous_segmentation_admitted = false,
            bool moe_rebalancing_graph_stable = false,
            bool host_policy_disabled = false,
            std::string *reject_stage_name = nullptr,
            std::string *reject_stage_type = nullptr)
        {
            if (reject_stage_name)
                reject_stage_name->clear();
            if (reject_stage_type)
                reject_stage_type->clear();
            if (host_policy_disabled)
                return PrefillGraphRejectReason::HostPolicyDisabled;
            return cache.preflight(
                graph,
                key,
                &collective_nodes,
                snapshots_active,
                moe_rebalancing_active,
                effectiveRealSeqLen(input),
                effectiveBucketSeqLen(input),
                mode,
                collectives_graph_capturable,
                heterogeneous_segmentation_admitted,
                moe_rebalancing_graph_stable
                    ? PrefillMoEGraphStability::Stable
                    : PrefillMoEGraphStability::Unstable,
                reject_stage_name,
                reject_stage_type);
        }

        /// @brief Deterministic tie-breaker for exact and bucketed prefill LRU victims.
        bool prefillSignatureLessForEviction(
            const ForwardGraphSignature &lhs,
            const ForwardGraphSignature &rhs)
        {
            if (lhs.bucket_seq_len != rhs.bucket_seq_len)
                return lhs.bucket_seq_len < rhs.bucket_seq_len;
            if (lhs.device != rhs.device)
                return lhs.device < rhs.device;
            if (lhs.seq_len != rhs.seq_len)
                return lhs.seq_len < rhs.seq_len;
            if (lhs.all_position_logits != rhs.all_position_logits)
                return lhs.all_position_logits < rhs.all_position_logits;
            if (lhs.live_mtp_request_batch_condition !=
                rhs.live_mtp_request_batch_condition)
            {
                return lhs.live_mtp_request_batch_condition <
                       rhs.live_mtp_request_batch_condition;
            }
            if (lhs.all_position_logit_rows != rhs.all_position_logit_rows)
                return lhs.all_position_logit_rows < rhs.all_position_logit_rows;
            return lhs.batch_size < rhs.batch_size;
        }
    }

    namespace {
    /**
     * @brief Lower the declared source association through the existing cold host boundary.
     * @param host Original graph execution host; retained callbacks never outlive this cold call.
     * @param input Original prefill/decode input naming the existing boundary geometry.
     * @param device Actual participant, without changing TP arithmetic membership order.
     * @param decode Select the existing decode or prefill boundary operation.
     * @param cache Original final-address graph/cache; ready replay creates no new callback.
     * @return Original branch factory with an optional cold-only metadata callback.
     */
    GraphCaptureAuxiliaryBranchFactory sourceDefinitionBoundaryFactory(IForwardExecutionHost& host,
        const ForwardInput& input,DeviceId device,bool decode,const ForwardGraphCache& cache){
        auto factory=host.forwardGraphAuxiliaryBranchFactory(device);
        // No new callback allocation or graph scan belongs to steady replay.
        if(!cache.graph || !cache.graph->retainedParticipantDefinitionDomain() || cache.segment_cache.initialized)return factory;
        factory.source_definition_boundary=[&host,&input,device,decode](const std::string& name,void* stream,
            LocalTPCaptureSourceDefinition& handoff){
            return decode?host.waitAtDecodeGraphCaptureBoundary(input,device,name,stream,handoff):
                host.waitAtPrefillGraphCaptureBoundary(input,device,name,stream,handoff);
        };
        return factory;
    }
    }
} // namespace llaminar2
