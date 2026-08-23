/**
 * @file ForwardExecutionEngine.cpp
 * @brief Implementation of ForwardExecutionEngine
 *
 * Contains the forward graph execution logic extracted from
 * DeviceGraphOrchestrator::executeForward().
 *
 * Split into:
 * - execute():          Entry point — signature computation, cache dispatch
 * - executeCacheHit():  Reuse cached graph (buffer update, PP copy, execution)
 * - executeCacheMiss(): Build new graph, execute, populate cache
 * - collectTimeline():  GPU stage timing collection and printing
 */

#include "ForwardExecutionEngine.h"
#include "PrefillBucketUtils.h"
#include "../graph/DeviceGraphCaptureController.h"
#include "../../compute_stages/ComputeStageFactory.h"
#include "../../moe/MoEOverlayRetainedParentComposer.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <unordered_set>

namespace llaminar2
{
    namespace
    {
        /// @brief Build a prefill graph-cache config from the cached debug environment.
        PrefillGraphConfig makePrefillGraphConfigFromEnv(
            int resident_graph_rows)
        {
            const auto &env = debugEnv();
            PrefillGraphConfig config;
            config.enabled = env.execution.gpu_graphs;
            config.minimum_padded_bucket_seq_len =
                effectivePrefillGraphMinimumPaddedBucketSeqLen(
                    env.execution.prefill_graph_min_seq,
                    resident_graph_rows);
            config.trace = env.execution.prefill_graph_trace;
            config.buckets_enabled = env.execution.prefill_graph_buckets;
            config.bucket_sizes = env.execution.prefill_graph_bucket_sizes;
            config.max_cached_entries = static_cast<size_t>(env.execution.prefill_graph_max_cached_buckets);
            return config;
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
         * invocation can contain many layer-local dispatch/return boundaries,
         * but they must all carry the same request generation and logical
         * chunk start.  Tests can therefore prove the continuation and remote
         * expert graph used the same protocol sequence without turning
         * layer-count into an observability artefact.  Callers invoke this
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
            PerfStatsCollector::addCounter(
                "forward_graph",
                "moe_overlay_collective_transaction",
                1.0,
                phase,
                input.device.to_string(),
                {{"role", "continuation_graph"},
                 {"identity_source", "orchestration_request_and_chunk"},
                 {"generation", std::to_string(runtime_params.generation_id)},
                 {"logical_step", std::to_string(runtime_params.step_id)},
                 {"logical_step_semantics", "monotonic_transaction"},
                 {"prefill_chunk_index",
                  std::to_string(input.prefill_chunk_index)},
                 {"token_offset", std::to_string(input.token_offset)},
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
            auto tags = PerfStatsCollector::Tags{
                {"capture_phase", observation.capture_phase},
                {"cache_phase", cache_phase ? cache_phase : "unknown"},
                {"chunk_index", std::to_string(observation.chunk_index)},
                {"bucket_seq_len", std::to_string(observation.bucket_seq_len)},
                {"real_token_start", std::to_string(observation.real_token_start)},
                {"real_token_count", std::to_string(observation.real_token_count)},
                {"real_token_end", std::to_string(observation.real_token_end)},
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

            PerfStatsCollector::addCounter(
                "forward_graph",
                "prefill_graph_lifecycle",
                1.0,
                "prefill",
                input.device.toString(),
                prefillGraphObservationTags(observation, prefillGraphPhaseName(cache_phase)));
            PerfStatsCollector::addCounter(
                "forward_graph",
                "prefill_graph_phase",
                1.0,
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
                {"uses_device_token_ids", boolTag(signature.uses_device_token_ids)},
                {"uses_device_position_ids", boolTag(signature.uses_device_position_ids)},
                {"uses_device_sequence_lengths", boolTag(signature.uses_device_sequence_lengths)},
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

    // =========================================================================
    // Construction
    // =========================================================================

    ForwardExecutionEngine::ForwardExecutionEngine(Config config, DeviceGraphExecutor &executor)
        : config_(std::move(config)), executor_(executor)
    {
    }

    // =========================================================================
    // Cache Management
    // =========================================================================

    ForwardExecutionEngine::PrefillChunkRuntimePlan ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        const ForwardInput &input,
        const std::vector<int> &bucket_sizes,
        int pad_token_id,
        bool allow_padded_execution)
    {
        PrefillChunkRuntimePlan plan;

        if (!input.token_ids && !input.token_ids_device)
        {
            plan.error = "bucketed prefill requires one authoritative token source";
            return plan;
        }
        if (input.token_ids && input.token_ids_device)
        {
            plan.error =
                "bucketed prefill rejects simultaneous host and device token authorities";
            return plan;
        }
        if (input.token_ids_device && !input.device.is_gpu())
        {
            plan.error =
                "bucketed prefill device token input requires a GPU device";
            return plan;
        }
        if (input.batch_size != 1)
        {
            plan.error = "bucketed prefill chunk runner currently supports batch_size=1";
            return plan;
        }

        const int real_seq_len = input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
        plan.selection = selectPrefillGraphBucket(real_seq_len, bucket_sizes);
        if (!plan.selection)
        {
            plan.error = plan.selection.error;
            return plan;
        }

        const int token_offset = effectiveTokenOffset(input);

        plan.padding_required = !plan.selection.exact;
        plan.position_policy =
            input.position_ids_device
                ? ForwardPositionPolicy::ExplicitRows
                : input.device.is_gpu()
                      ? ForwardPositionPolicy::ContiguousOffset
                      : ForwardPositionPolicy::ExplicitRows;
        plan.chunk.token_offset = token_offset;
        plan.chunk.real_count = real_seq_len;
        plan.chunk.bucket_seq_len = plan.selection.bucket_seq_len;
        plan.chunk.token_authority = input.token_ids_device
                                         ? PrefillChunkTokenAuthority::DeviceResidentRows
                                         : PrefillChunkTokenAuthority::HostPaddedRows;
        if (plan.chunk.token_authority ==
            PrefillChunkTokenAuthority::HostPaddedRows)
        {
            plan.chunk.token_ids = padPrefillTokensToBucket(
                input.token_ids,
                real_seq_len,
                plan.selection.bucket_seq_len,
                pad_token_id);
        }
        if (plan.position_policy == ForwardPositionPolicy::ExplicitRows &&
            !input.position_ids_device)
        {
            plan.chunk.position_ids = buildPrefillChunkPositionIds(
                real_seq_len,
                plan.selection.bucket_seq_len,
                token_offset,
                input.batch_size);
        }

        if ((plan.chunk.token_authority ==
                 PrefillChunkTokenAuthority::HostPaddedRows &&
             plan.chunk.token_ids.empty()) ||
            (plan.position_policy == ForwardPositionPolicy::ExplicitRows &&
             !input.position_ids_device &&
             plan.chunk.position_ids.empty()))
        {
            plan.error = "failed to prepare bucketed prefill chunk buffers";
            return plan;
        }

        if (plan.padding_required && !allow_padded_execution)
        {
            plan.error = "Bucketed prefill with padding requires caller opt-in to padded execution: real_seq_len=" +
                         std::to_string(real_seq_len) +
                         " bucket_seq_len=" + std::to_string(plan.selection.bucket_seq_len);
            return plan;
        }

        plan.chunk.ok = true;
        plan.ok = true;
        return plan;
    }

    ForwardExecutionEngine::PrefillChunkRuntimeSchedule ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        const ForwardInput &input,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        PrefillChunkRuntimeSchedule runtime_schedule;

        if ((!input.token_ids && !input.token_ids_device) ||
            (input.token_ids && input.token_ids_device))
        {
            runtime_schedule.error =
                "bucketed prefill schedule requires exactly one token authority";
            return runtime_schedule;
        }
        if (input.token_ids_device && !input.device.is_gpu())
        {
            runtime_schedule.error =
                "bucketed prefill device token input requires a GPU device";
            return runtime_schedule;
        }
        if (input.batch_size != 1)
        {
            runtime_schedule.error = "bucketed prefill schedule currently supports batch_size=1";
            return runtime_schedule;
        }

        runtime_schedule.schedule = planPrefillChunkSchedule(policy);
        if (!runtime_schedule.schedule)
        {
            runtime_schedule.error = runtime_schedule.schedule.error;
            return runtime_schedule;
        }
        if (input.token_ids_device &&
            runtime_schedule.schedule.chunks.size() > 1 &&
            !input.device_prefill_chunk.has_value())
        {
            runtime_schedule.error =
                "multi-chunk device prefill requires a captured device chunk materializer";
            return runtime_schedule;
        }

        const int input_real_tokens = effectiveRealSeqLen(input);
        const int input_start = effectiveTokenOffset(input);
        if (input_real_tokens <= 0)
        {
            runtime_schedule.error = "input real token count must be positive";
            return runtime_schedule;
        }
        if (input_start < 0)
        {
            runtime_schedule.error = "input token offset must be non-negative";
            return runtime_schedule;
        }
        const int input_end = input_start + input_real_tokens;

        runtime_schedule.chunks.reserve(runtime_schedule.schedule.chunks.size());
        for (const auto &chunk : runtime_schedule.schedule.chunks)
        {
            if (chunk.token_offset < input_start ||
                chunk.token_offset + chunk.real_count > input_end)
            {
                runtime_schedule.error = "scheduled chunk range is outside input token range";
                runtime_schedule.chunks.clear();
                return runtime_schedule;
            }

            ForwardInput chunk_input = input;
            const int relative_offset = chunk.token_offset - input_start;
            chunk_input.token_ids =
                input.token_ids ? input.token_ids + relative_offset : nullptr;
            chunk_input.token_ids_device = input.token_ids_device;
            chunk_input.seq_len = chunk.real_count;
            chunk_input.real_seq_len = chunk.real_count;
            chunk_input.bucket_seq_len = 0;
            chunk_input.token_offset = chunk.token_offset;
            chunk_input.position_offset = chunk.token_offset;

            /*
             * The scheduler is the sole authority for physical graph width.
             * Prepare against that one selected bucket instead of repeating
             * selection across the policy's complete bucket set.  Repeating
             * the policy decision here allowed a short final chunk to choose a
             * smaller graph even when fixed_chunk_real_tokens deliberately
             * established one immutable capture width for the transaction.
             */
            auto runtime_plan = prepareSinglePrefillChunkRuntimePlan(
                chunk_input,
                std::vector<int>{chunk.bucket_seq_len},
                pad_token_id,
                allow_padded_execution);
            runtime_plan.chunk_index = chunk.chunk_index;
            runtime_plan.rebalance_allowed_after = chunk.rebalance_allowed_after;
            runtime_plan.rebalance_required_after = chunk.rebalance_required_after;

            if (!runtime_plan)
            {
                runtime_schedule.error = runtime_plan.error;
                runtime_schedule.chunks.push_back(std::move(runtime_plan));
                return runtime_schedule;
            }
            if (runtime_plan.chunk.bucket_seq_len != chunk.bucket_seq_len)
            {
                runtime_schedule.error = "runtime chunk bucket does not match scheduler bucket";
                runtime_schedule.chunks.clear();
                return runtime_schedule;
            }

            runtime_schedule.chunks.push_back(std::move(runtime_plan));
        }

        runtime_schedule.ok = true;
        return runtime_schedule;
    }

    bool ForwardExecutionEngine::runPrefillChunk(
        const ForwardInput &base_input,
        const PrefillChunkRuntimePlan &plan,
        ForwardOutput &output,
        IForwardExecutionHost &host)
    {
        if (!plan)
        {
            LOG_ERROR("[ForwardExecutionEngine] Invalid prefill chunk plan: " << plan.error);
            return false;
        }
        if (!plan.chunk)
        {
            LOG_ERROR("[ForwardExecutionEngine] Invalid prefill chunk input: " << plan.chunk.error);
            return false;
        }
        if (plan.padding_required &&
            (!config_.cache_config.enabled ||
             !base_input.device.is_gpu() ||
             !debugEnv().execution.gpu_graphs ||
             !debugEnv().execution.prefill_graph_buckets))
        {
            LOG_ERROR("[ForwardExecutionEngine] Padded prefill chunk execution requires GPU graph bucket preflight: real_count="
                      << plan.chunk.real_count << " bucket_seq_len=" << plan.chunk.bucket_seq_len
                      << " device=" << base_input.device.toString());
            return false;
        }

        // The runtime input owns padded token/position buffers, whereas the
        // host-facing hooks deliberately consume the stable scheduler contract.
        // Materialize that contract once so snapshots and chunk maintenance
        // observe identical real-row and bucket geometry.
        const PrefillChunkPlan scheduled_chunk = makeMaintenanceChunkPlan(plan);
        ScopedPrefillChunkSnapshotDiagnostics snapshot_scope(host, scheduled_chunk);

        // Preserve the caller's execution context and swap in the prepared
        // chunk buffers plus fixed-bucket metadata for the delegated launch.
        ForwardInput chunk_input = makeBucketedPrefillInput(base_input, plan);

        std::unique_ptr<IPrefillChunkGraphSubmissionLease> submission_lease;
        std::string submission_error;
        if (!host.beginPrefillChunkGraphSubmission(
                chunk_input,
                scheduled_chunk,
                &submission_lease,
                &submission_error))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Prefill chunk graph submission "
                "admission failed for chunk "
                << scheduled_chunk.chunk_index << ": "
                << (submission_error.empty()
                        ? "external graph authority rejected the chunk"
                        : submission_error));
            return false;
        }

        const bool execution_succeeded = execute(chunk_input, output, host);
        if (submission_lease &&
            !submission_lease->finish(
                execution_succeeded, &submission_error))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Prefill chunk graph submission "
                "could not publish its terminal state for chunk "
                << scheduled_chunk.chunk_index << ": "
                << (submission_error.empty()
                        ? "external graph authority rejected the terminal state"
                        : submission_error));
            return false;
        }
        if (!execution_succeeded)
            return false;

        const PrefillChunkMaintenanceState maintenance_state =
            host.prefillChunkMaintenanceState(scheduled_chunk);
        const PrefillChunkMaintenanceDecision maintenance_decision =
            evaluatePrefillChunkMaintenance(scheduled_chunk, maintenance_state);

        const bool maintenance_was_requested =
            maintenance_state.rebalance_requested ||
            scheduled_chunk.rebalance_required_after;
        if (maintenance_was_requested && !maintenance_decision)
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk maintenance blocked after chunk "
                      << scheduled_chunk.chunk_index << ": "
                      << maintenance_decision.reason);
            return false;
        }
        if (maintenance_decision.can_run &&
            !host.onPrefillChunkMaintenance(scheduled_chunk, maintenance_decision))
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk maintenance failed after chunk "
                      << scheduled_chunk.chunk_index << ": "
                      << maintenance_decision.reason);
            return false;
        }

        return true;
    }

    bool ForwardExecutionEngine::runPrefillChunkSchedule(
        const ForwardInput &base_input,
        const PrefillChunkRuntimeSchedule &schedule,
        ForwardOutput &output,
        IForwardExecutionHost &host)
    {
        if (!schedule)
        {
            LOG_ERROR("[ForwardExecutionEngine] Invalid prefill chunk schedule: "
                      << schedule.error);
            return false;
        }
        if (schedule.chunks.empty())
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk schedule is empty");
            return false;
        }

        for (const auto &chunk_plan : schedule.chunks)
        {
            if (!runPrefillChunk(base_input, chunk_plan, output, host))
            {
                host.cancelPrefillChunkSnapshotDiagnostics();
                LOG_ERROR("[ForwardExecutionEngine] Prefill chunk schedule failed at chunk "
                          << chunk_plan.chunk_index);
                return false;
            }
        }

        if (!host.finalizePrefillChunkSnapshotDiagnostics(schedule.schedule))
        {
            host.cancelPrefillChunkSnapshotDiagnostics();
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk schedule produced "
                      "incomplete snapshot diagnostics");
            return false;
        }

        return true;
    }

    void ForwardExecutionEngine::invalidateAll()
    {
        all_position_verifier_recapture_pending_ = false;
        for (auto &[_, cache] : cache_)
        {
            cache.invalidate();
        }
    }

    void ForwardExecutionEngine::discardAllCachedGraphs()
    {
        invalidateAll();
        cache_.clear();
    }

    size_t ForwardExecutionEngine::invalidateMoEPlacementSensitiveGraphsForStablePlacement()
    {
        size_t invalidated = 0;
        for (auto &[signature, cache] : cache_)
        {
            if (!cache.valid)
                continue;

            const bool graph_stable_single_token_decode =
                signature.decode && !signature.all_position_logits &&
                signature.seq_len == 1 && signature.batch_size <= 1;
            const bool graph_stable_fixed_prefill = !signature.decode;
            if (graph_stable_single_token_decode || graph_stable_fixed_prefill)
            {
                cache.markGPUStreamBindingsDirty();
                continue;
            }

            cache.invalidate();
            ++invalidated;
        }
        return invalidated;
    }

    void ForwardExecutionEngine::resetCapturedReplayState()
    {
        all_position_verifier_recapture_pending_ = false;
        for (auto &entry : cache_)
        {
            entry.second.resetReplayState();
        }
    }

    ForwardExecutionEngine::ReplayStateResetSummary
    ForwardExecutionEngine::resetSessionReplayState(
        bool preserve_replay_safe_graphs)
    {
        ReplayStateResetSummary summary;
        all_position_verifier_recapture_pending_ = false;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            const ForwardReplayStateAction action =
                preserve_replay_safe_graphs
                    ? chooseForwardReplayStateAction(
                          ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                          signature)
                    : ForwardReplayStateAction::ResetReplayState;

            if (action == ForwardReplayStateAction::ResetReplayState)
            {
                cache.resetSessionState();
                ++summary.reset_replay_state;
                if (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                    cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
                    ++summary.ordinary_decode_reset;
                continue;
            }

            cache.resetSessionStatePreservingGraphReplay();
            ++summary.preserved_for_stream_rebind;
            if (cache_class == ForwardReplayStateCacheClass::AllPositionVerifier)
                ++summary.all_position_verifier_preserved;
            else if (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                     cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
                ++summary.other_preserved;
            else
                ++summary.other_preserved;
        }
        return summary;
    }

    ForwardExecutionEngine::ReplayStateResetSummary
    ForwardExecutionEngine::resetCapturedReplayStateForCorrectionReplay(
        uint64_t live_state_epoch,
        bool preserve_single_token_decode_replay)
    {
        ReplayStateResetSummary summary;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            ForwardReplayStateAction action =
                chooseForwardReplayStateAction(
                    ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary,
                    signature);
            if (!preserve_single_token_decode_replay &&
                (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                 cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode))
            {
                action = ForwardReplayStateAction::ResetReplayState;
            }
            if (action == ForwardReplayStateAction::ResetReplayState)
            {
                cache.resetReplayState();
                ++summary.reset_replay_state;
                if (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                    cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
                    ++summary.ordinary_decode_reset;
            }
            else
            {
                cache.markGPUStreamBindingsDirty();
                cache.markReplayStateSafeForLiveEpoch(live_state_epoch);
                ++summary.preserved_for_stream_rebind;
                if (cache_class == ForwardReplayStateCacheClass::AllPositionVerifier)
                    ++summary.all_position_verifier_preserved;
                else
                    ++summary.other_preserved;
            }
        }
        return summary;
    }

    ForwardExecutionEngine::ReplayStateResetSummary
    ForwardExecutionEngine::rebindCapturedReplayStateAfterPrefixRestore(
        uint64_t live_state_epoch)
    {
        ReplayStateResetSummary summary;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            const ForwardReplayStateAction action =
                chooseForwardReplayStateAction(
                    ForwardReplayStateMutationKind::PrefixCheckpointRestore,
                    signature);
            if (action == ForwardReplayStateAction::ResetReplayState)
            {
                cache.resetReplayState();
                ++summary.reset_replay_state;
                if (cache_class ==
                        ForwardReplayStateCacheClass::OrdinaryDecode ||
                    cache_class ==
                        ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
                {
                    ++summary.ordinary_decode_reset;
                }
                continue;
            }

            /*
             * Prefix restore also retires request-local token, position, and
             * sequence-length metadata from the discarded timeline.  Reset that
             * metadata through the captured-replay-preserving stage API: this
             * keeps the graph executable and stable buffers, but guarantees that
             * the next invocation republishes every dynamic row before replay.
             * The restore producer event still owns all device writes, and the
             * epoch stamp prevents the cache from being mistaken for an
             * unversioned capture.
             */
            cache.resetSessionStatePreservingGraphReplay();
            cache.markReplayStateSafeForLiveEpoch(live_state_epoch);
            ++summary.preserved_for_stream_rebind;
            if (cache_class ==
                ForwardReplayStateCacheClass::AllPositionVerifier)
            {
                ++summary.all_position_verifier_preserved;
            }
            else
            {
                ++summary.other_preserved;
            }
        }
        return summary;
    }

    ForwardExecutionEngine::ReplayStateResetSummary
    ForwardExecutionEngine::resetAllPositionVerifierReplayState()
    {
        ReplayStateResetSummary summary;
        all_position_verifier_recapture_pending_ = true;
        summary.all_position_verifier_recapture_requested = true;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            if (cache_class == ForwardReplayStateCacheClass::AllPositionVerifier)
            {
                cache.resetReplayState();
                ++summary.reset_replay_state;
                continue;
            }

            cache.markGPUStreamBindingsDirty();
            ++summary.preserved_for_stream_rebind;
            if (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
            {
                ++summary.other_preserved;
            }
        }
        return summary;
    }

    void ForwardExecutionEngine::touchPrefillForwardCache(
        const ForwardGraphSignature &signature,
        ForwardGraphCache &cache)
    {
        if (signature.decode || !cache.valid)
            return;
        cache.prefill_last_access_tick = ++prefill_forward_access_counter_;
    }

    size_t ForwardExecutionEngine::prefillForwardCacheSize() const
    {
        size_t count = 0;
        for (const auto &[signature, cache] : cache_)
        {
            if (!signature.decode && cache.valid)
                ++count;
        }
        return count;
    }

    void ForwardExecutionEngine::enforcePrefillForwardCapacity(
        const ForwardGraphSignature *active_signature)
    {
        const int configured_cap = debugEnv().execution.prefill_graph_max_cached_buckets;
        if (configured_cap <= 0)
            return;

        const size_t cap = static_cast<size_t>(configured_cap);
        while (prefillForwardCacheSize() > cap)
        {
            auto victim = cache_.end();
            uint64_t oldest_tick = std::numeric_limits<uint64_t>::max();

            // Exact CPU prefill and bucketed GPU prefill both retain complete
            // graph topology and stage-owned persistent buffers. Bound them as
            // one cache class; decode has its independent stable-shape lifetime.
            for (auto it = cache_.begin(); it != cache_.end(); ++it)
            {
                const auto &signature = it->first;
                const auto &cache = it->second;
                if (signature.decode || !cache.valid)
                    continue;
                if (active_signature && signature == *active_signature)
                    continue;

                const uint64_t tick = cache.prefill_last_access_tick;
                if (victim == cache_.end() ||
                    tick < oldest_tick ||
                    (tick == oldest_tick && prefillSignatureLessForEviction(signature, victim->first)))
                {
                    oldest_tick = tick;
                    victim = it;
                }
            }

            if (victim == cache_.end())
                return;

            const auto evicted_signature = victim->first;
            victim->second.invalidate();
            cache_.erase(victim);
            ++prefill_forward_eviction_count_;

            LOG_INFO("[ForwardExecutionEngine] Evicted prefill forward topology rows="
                     << (evicted_signature.is_bucketed_prefill
                             ? evicted_signature.bucket_seq_len
                             : evicted_signature.seq_len)
                     << " bucketed=" << evicted_signature.is_bucketed_prefill
                     << " device=" << evicted_signature.device.toString()
                     << " due to cache cap=" << configured_cap);
        }
    }

    // =========================================================================
    // execute() — Entry Point
    // =========================================================================

    bool ForwardExecutionEngine::execute(
        const ForwardInput &input,
        ForwardOutput &output,
        IForwardExecutionHost &host)
    {
        ScopedWorkerGPUContextResolver worker_resolver_scope(
            executor_,
            host,
            input.device);
        last_executed_forward_graph_ = {};
        const bool setup_materialization =
            input.graph_submission_intent ==
            ForwardGraphSubmissionIntent::
                MaterializeExecutableWithoutLaunch;
        if (setup_materialization &&
            (!input.device.is_gpu() ||
             input.moe_overlay_graph_launch_dependency))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup-only forward graph "
                "materialization requires a GPU input with no request-scoped "
                "launch dependency");
            return false;
        }

        /*
         * Workspace-family discovery builds the exact production topology with
         * permanent arena/KV addresses before a workspace generation exists.
         * That typed declaration is never executable: only the runtime binding
         * whose capture identity includes the finalized generation may cross
         * this boundary.
         */
        const bool shifted_main_prefill =
            input.execution_role == ForwardExecutionRole::MainInference &&
            input.execution_phase == ForwardExecutionPhase::Prefill &&
            input.state_transaction == ForwardStateTransaction::Ordinary;
        const bool shifted_restored_prefix_decode =
            input.execution_role == ForwardExecutionRole::MTPCondition &&
            input.execution_phase == ForwardExecutionPhase::Decode &&
            input.state_transaction ==
                ForwardStateTransaction::RestoredPrefixMTPDecodeBridge &&
            input.batch_size == 1 && input.seq_len == 1;
        if (input.shifted_mtp_prefill &&
            (!input.shifted_mtp_prefill->executableForRequestCount(
                 input.batch_size) ||
             (!shifted_main_prefill &&
              !shifted_restored_prefix_decode)))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Refusing a declaration-only or "
                "incomplete shifted-MTP prefill binding");
            return false;
        }
        if (input.mtp_main_terminal_hidden &&
            (!input.mtp_main_terminal_hidden->executableForRequestCount(
                 input.batch_size) ||
             input.shifted_mtp_prefill.has_value() ||
             input.state_transaction != ForwardStateTransaction::Ordinary ||
             input.execution_phase != ForwardExecutionPhase::Decode ||
             (input.execution_role != ForwardExecutionRole::MainInference &&
              input.execution_role != ForwardExecutionRole::MTPCondition)))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Refusing an incomplete or "
                "mis-scoped MTP main terminal-hidden binding");
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        const int first_position = input.position_ids ? input.position_ids[0] : input.position_offset;
        /*
         * The public orchestrator resolves legacy shape/history heuristics once
         * and publishes the result through ForwardInput::execution_phase. This
         * lower execution boundary must consume that typed decision verbatim.
         * Reclassifying M=1 here used to turn a one-token request prefill into a
         * decode-shaped continuation graph after the heterogeneous overlay had
         * already admitted a padded prefill transaction, leaving the two sides
         * with incompatible sparse-collective timelines.
         */
        const bool all_position_logits = host.computeAllPositionLogitsEnabled();
        const bool live_mtp_request_batch_condition =
            host.liveMTPRequestBatchConditionEnabled();
        const bool grouped_mtp_verifier =
            input.execution_role ==
            ForwardExecutionRole::GroupedMTPVerifier;
        if (grouped_mtp_verifier)
        {
            /*
             * A new verifier attempt owns the publication handle. Clear any
             * earlier row-snapshot producer before execution so a failed
             * verifier cannot leave stale snapshots publishable. All-position
             * logits alone are insufficient: condition and diagnostic graphs
             * may also expose multiple rows but do not own verifier state.
             */
            clearLastAllPositionVerifierForwardGraph();
        }
        const bool mtp_condition =
            input.execution_role == ForwardExecutionRole::MTPCondition;
        const bool role_requires_decode =
            grouped_mtp_verifier || mtp_condition;
        const bool is_decode =
            input.execution_phase == ForwardExecutionPhase::Decode;
        if (live_mtp_request_batch_condition && !mtp_condition)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Live MTP condition policy was "
                "submitted without the typed MTPCondition execution role");
            return false;
        }
        if (role_requires_decode && !is_decode)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Decode-owned execution role was "
                "submitted with prefill phase (role="
                << static_cast<int>(input.execution_role)
                << ", live_request_batch_condition="
                << live_mtp_request_batch_condition << ")");
            return false;
        }
        const bool decode_has_history = is_decode && first_position > 0;
        const bool has_unified_pp = config_.has_unified_pp;
        const bool is_standard_path = !has_unified_pp && !config_.pp_stage_config.has_value();
        const bool is_partial_pp_path = !has_unified_pp && config_.pp_stage_config.has_value();

        // =====================================================================
        // Decode Graph Cache: Reuse cached graph for decode mode (seq_len=1)
        // =====================================================================
        // During decode, the graph structure is identical between steps —
        // only token_ids, position_ids, and position_offset change.
        // Instead of rebuilding hundreds of stage objects every forward() call,
        // we cache the graph after the first decode step and reuse it.
        //
        // Benefits:
        // - Eliminates stage object construction/destruction (~100s of allocs)
        // - Preserves kernel caches in stages (JIT attention, RoPE inv_freq)
        // - Avoids workspace re-binding (bindWorkspace → inv_freq reset)
        // - Skips graph traversal in ensureDeviceWorkspaceAllocated()
        // =====================================================================

        // PP non-embedding stages receive hidden state instead of tokens,
        // so token_ids is legitimately nullptr. They still have stable inputs
        // (position_ids for RoPE, hidden state via setHiddenState).
        const bool is_pp_non_embedding_stage =
            config_.pp_stage_config.has_value() && !config_.pp_stage_config->has_embedding;
        const bool has_position_input =
            input.position_ids != nullptr ||
            (input.position_ids_device != nullptr && input.device.is_gpu()) ||
            (input.position_policy == ForwardPositionPolicy::ContiguousOffset &&
             input.batch_size == 1 &&
             input.position_ids == nullptr &&
             input.position_ids_device == nullptr);
        const bool has_stable_forward_inputs =
            (((input.token_ids != nullptr) || (input.token_ids_device != nullptr)) &&
             has_position_input) ||
            (is_pp_non_embedding_stage && has_position_input);

        const auto &env = debugEnv();
        const int resident_graph_rows = host.residentGraphRows();
        const auto resident_prefill_buckets =
            prefillGraphBucketsAtOrBelowCapacity(
                env.execution.prefill_graph_bucket_sizes,
                resident_graph_rows);
        const int raw_prefill_bucket_floor =
            effectivePrefillGraphMinimumPaddedBucketSeqLen(
                env.execution.prefill_graph_min_seq,
                resident_graph_rows);
        const std::vector<int> raw_prefill_buckets =
            rawPrefillGraphBucketsForResidentCapacity(
                env.execution.prefill_graph_bucket_sizes,
                resident_graph_rows,
                env.execution.prefill_graph_min_seq);
        /*
         * Forward topology reuse and native GPU graph capture are separate
         * policies. Exact-shape CPU prefill keeps its ComputeGraph, prepared
         * kernels, and stage-owned persistent workspace warm, then executes it
         * through the ordinary CPU executor. Homogeneous GPU prefill additionally
         * enters the bucketed capture/replay state machine below.
         */
        const bool prefill_topology_cache_eligible =
            config_.cache_config.enabled &&
            !is_decode &&
            !has_unified_pp &&
            has_stable_forward_inputs &&
            (is_standard_path || is_partial_pp_path);
        const bool cpu_exact_prefill_cache_eligible =
            prefill_topology_cache_eligible && input.device.is_cpu();
        const bool prefill_cache_eligible =
            prefill_topology_cache_eligible &&
            input.device.is_gpu() &&
            env.execution.gpu_graphs;

        ForwardInput effective_input = input;
        std::optional<PrefillChunkRuntimePlan> raw_bucket_plan;
        bool bucketed_prefill = false;
        int bucketed_prefill_seq_len = input.seq_len;

        if (!is_decode && input.bucket_seq_len > 0 && isPaddedBucketExecution(input) &&
            (!prefill_cache_eligible || !env.execution.prefill_graph_buckets))
        {
            LOG_ERROR("[ForwardExecutionEngine] Padded bucketed prefill input requires GPU graph bucket preflight: seq_len="
                      << input.seq_len << " real_seq_len=" << effectiveRealSeqLen(input)
                      << " bucket_seq_len=" << effectiveBucketSeqLen(input)
                      << " device=" << input.device.toString());
            return false;
        }

        const bool bucketed_prefill_eligible =
            prefill_cache_eligible &&
            env.execution.prefill_graph_buckets &&
            input.batch_size == 1;

        if (prefill_cache_eligible && env.execution.prefill_graph_buckets && input.batch_size > 1)
        {
            LOG_DEBUG("[ForwardExecutionEngine] Skipping bucketed prefill for batched request prefill: batch_size="
                      << input.batch_size << " seq_len=" << input.seq_len
                      << " device=" << input.device.toString());
        }

        if (bucketed_prefill_eligible)
        {
            const bool already_bucketed_input = input.bucket_seq_len > 0;
            if (already_bucketed_input)
            {
                const int real_seq_len = effectiveRealSeqLen(input);
                if (input.seq_len != input.bucket_seq_len || real_seq_len > input.bucket_seq_len)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Prepared bucketed prefill input has inconsistent shape: seq_len="
                              << input.seq_len << " real_seq_len=" << real_seq_len
                              << " bucket_seq_len=" << input.bucket_seq_len);
                    return false;
                }
                if (resident_graph_rows > 0 &&
                    input.bucket_seq_len > resident_graph_rows)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Prepared prefill bucket "
                              << input.bucket_seq_len
                              << " exceeds memory-planned resident graph rows "
                              << resident_graph_rows);
                    return false;
                }
                bucketed_prefill = true;
                bucketed_prefill_seq_len = input.bucket_seq_len;
            }
            else
            {
                if (raw_prefill_buckets.empty())
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] No resident prefill graph bucket meets the configured raw-prompt floor: floor="
                        << raw_prefill_bucket_floor
                        << " resident_graph_rows=" << resident_graph_rows
                        << "; correct the bucket/capacity contract instead of executing eagerly");
                    return false;
                }
                ForwardInput planning_input = input;
                planning_input.token_offset = effectiveTokenOffset(input);
                raw_bucket_plan = prepareSinglePrefillChunkRuntimePlan(
                    planning_input,
                    raw_prefill_buckets,
                    env.execution.prefill_graph_pad_token_id,
                    /*allow_padded_execution=*/true);
                if (!raw_bucket_plan || !raw_bucket_plan->chunk)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Bucketed prefill graph request rejected: "
                              << (raw_bucket_plan ? raw_bucket_plan->error : std::string("failed to prepare bucketed input"))
                              << " (seq_len=" << input.seq_len << ")");
                    return false;
                }
                effective_input = makeBucketedPrefillInput(input, *raw_bucket_plan);
                bucketed_prefill = true;
                bucketed_prefill_seq_len = raw_bucket_plan->chunk.bucket_seq_len;
            }
        }

        const bool forward_cache_eligible =
            (config_.cache_config.enabled &&
             is_decode &&
             !has_unified_pp &&
             has_stable_forward_inputs &&
             (is_standard_path || is_partial_pp_path)) ||
            cpu_exact_prefill_cache_eligible ||
            prefill_cache_eligible;

        ForwardGraphSignature forward_signature;
        ForwardGraphCache *active_forward_cache = nullptr;

        if (forward_cache_eligible)
        {
            int pp_first_layer = -1;
            int pp_last_layer = -1;
            bool pp_has_embedding = false;
            bool pp_has_lm_head = false;
            if (config_.pp_stage_config.has_value())
            {
                const auto &pp = config_.pp_stage_config.value();
                pp_first_layer = pp.first_layer;
                pp_last_layer = pp.last_layer;
                pp_has_embedding = pp.has_embedding;
                pp_has_lm_head = pp.has_lm_head;
            }

            forward_signature = ForwardGraphSignature{
                .seq_len = bucketed_prefill
                               ? bucketed_prefill_seq_len
                               : effective_input.seq_len,
                .batch_size = effective_input.batch_size,
                .device = effective_input.device,
                .execution_role = effective_input.execution_role,
                .state_transaction = effective_input.state_transaction,
                .decode = is_decode,
                .decode_has_history = decode_has_history,
                .all_position_logits = all_position_logits,
                .live_mtp_request_batch_condition =
                    live_mtp_request_batch_condition,
                .all_position_logit_rows =
                    all_position_logits
                        ? std::max(0, host.allPositionLogitRows())
                        : 0,
                .mtp_verifier_outcome_graph_mode =
                    host.mtpVerifierOutcomeGraphMode(),
                .uses_device_token_ids = input.token_ids_device != nullptr,
                .uses_device_position_ids = input.position_ids_device != nullptr,
                .position_policy = effective_input.position_policy,
                .uses_device_sequence_lengths =
                    input.sequence_lengths_device != nullptr,
                .device_prefill_chunk_capture_identity =
                    effective_input.device_prefill_chunk
                        ? effective_input.device_prefill_chunk->capture_identity
                        : uint64_t{0},
                .shifted_mtp_prefill_capture_identity =
                    effective_input.shifted_mtp_prefill
                        ? effective_input.shifted_mtp_prefill->capture_identity
                        : uint64_t{0},
                .mtp_main_terminal_hidden_capture_identity =
                    effective_input.mtp_main_terminal_hidden
                        ? effective_input.mtp_main_terminal_hidden
                              ->capture_identity
                        : uint64_t{0},
                .standard_path = is_standard_path,
                .pp_stage_enabled = config_.pp_stage_config.has_value(),
                .pp_first_layer = pp_first_layer,
                .pp_last_layer = pp_last_layer,
                .pp_has_embedding = pp_has_embedding,
                .pp_has_lm_head = pp_has_lm_head,
                .is_bucketed_prefill = bucketed_prefill,
                .bucket_seq_len =
                    bucketed_prefill ? bucketed_prefill_seq_len : 0,
                .rehydrate_prefix_runtime_on_device =
                    effective_input.rehydrate_prefix_runtime_on_device,
                .moe_placement_epoch = host.moePlacementEpoch()};

            auto cache_it = cache_.find(forward_signature);
            if (cache_it != cache_.end())
            {
                active_forward_cache = &cache_it->second;
                active_forward_cache->segment_cache.replay_workload =
                    replayWorkloadGeometryForSignature(forward_signature);
            }
        }

        const bool use_cached_forward = forward_cache_eligible &&
                                        active_forward_cache &&
                                        active_forward_cache->valid;

        if (forward_cache_eligible)
        {
            LOG_TRACE(
                "[ForwardGraphCacheIdentity] result="
                << (use_cached_forward ? "hit" : "miss")
                << " submission="
                << (setup_materialization ? "setup_materialize" : "runtime")
                << " seq_len=" << forward_signature.seq_len
                << " batch=" << forward_signature.batch_size
                << " device=" << forward_signature.device.toString()
                << " role="
                << static_cast<int>(forward_signature.execution_role)
                << " decode=" << boolTag(forward_signature.decode)
                << " history="
                << boolTag(forward_signature.decode_has_history)
                << " all_logits="
                << boolTag(forward_signature.all_position_logits)
                << " device_tokens="
                << boolTag(forward_signature.uses_device_token_ids)
                << " device_positions="
                << boolTag(forward_signature.uses_device_position_ids)
                << " position_policy="
                << static_cast<int>(forward_signature.position_policy)
                << " device_lengths="
                << boolTag(forward_signature.uses_device_sequence_lengths)
                << " chunk_identity="
                << forward_signature.device_prefill_chunk_capture_identity
                << " shifted_mtp_identity="
                << forward_signature.shifted_mtp_prefill_capture_identity
                << " mtp_terminal_hidden_identity="
                << forward_signature
                       .mtp_main_terminal_hidden_capture_identity
                << " bucketed="
                << boolTag(forward_signature.is_bucketed_prefill)
                << " bucket_rows=" << forward_signature.bucket_seq_len
                << " prefix_rehydrate="
                << boolTag(
                       forward_signature.rehydrate_prefix_runtime_on_device)
                << " placement_epoch="
                << forward_signature.moe_placement_epoch);
        }

        if (live_mtp_request_batch_condition &&
            debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            LOG_INFO(
                "[MTPConditionGraphContract] event=cache_decision"
                << " device=" << effective_input.device.toString()
                << " result=" << (use_cached_forward ? "hit" : "miss")
                << " seq_len=" << forward_signature.seq_len
                << " batch_size=" << forward_signature.batch_size
                << " device_tokens="
                << boolTag(forward_signature.uses_device_token_ids)
                << " device_positions="
                << boolTag(forward_signature.uses_device_position_ids)
                << " device_lengths="
                << boolTag(
                       forward_signature.uses_device_sequence_lengths)
                << " placement_epoch="
                << forward_signature.moe_placement_epoch);
        }

        if (use_cached_forward)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "forward_cache_lookup",
                1.0,
                forward_signature.decode ? "decode" : "prefill",
                forward_signature.device.toString(),
                forwardCacheLookupTags(forward_signature, "hit"));
            touchPrefillForwardCache(forward_signature, *active_forward_cache);
            const bool success = executeCacheHit(effective_input, output, *active_forward_cache, host,
                                                 is_decode, start);
            if (setup_materialization)
            {
                if (success && !forward_signature.decode)
                    enforcePrefillForwardCapacity(&forward_signature);
                return success;
            }
            if (live_mtp_request_batch_condition &&
                debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
            {
                LOG_INFO(
                    "[MTPConditionGraphContract] event=execution_complete"
                    << " device=" << effective_input.device.toString()
                    << " cache=hit"
                    << " success=" << boolTag(success));
            }
            if (success)
            {
                recordLastExecutedForwardGraph(
                    forward_signature,
                    /*cache_hit=*/true,
                    output.execution.stream);
            }
            if (success && !forward_signature.decode)
                enforcePrefillForwardCapacity(&forward_signature);
            if (success)
                host.commitSuccessfulForwardOutput(output);
            return success;
        }

        // Cache MISS path
        ForwardGraphCache *build_cache = nullptr;
        bool should_cache_after_build = false;
        if (forward_cache_eligible)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "forward_cache_lookup",
                1.0,
                forward_signature.decode ? "decode" : "prefill",
                forward_signature.device.toString(),
                forwardCacheLookupTags(forward_signature, "miss"));
            auto [it, _inserted] = cache_.try_emplace(forward_signature);
            build_cache = &it->second;
            build_cache->segment_cache.replay_workload =
                replayWorkloadGeometryForSignature(forward_signature);
            should_cache_after_build = !build_cache->valid;
        }

        ForwardInput build_input = effective_input;
        if (forward_cache_eligible && forward_signature.is_bucketed_prefill && build_input.bucket_seq_len <= 0)
        {
            // Mark exact raw bucket requests as bucket-shaped so LM-head
            // row-select is present if the cached bucket is later replayed with
            // a shorter real length.
            build_input.real_seq_len = effectiveRealSeqLen(effective_input);
            build_input.bucket_seq_len = forward_signature.bucket_seq_len;
        }

        const bool success = executeCacheMiss(build_input, output, forward_signature, build_cache,
                                              should_cache_after_build, host, is_decode,
                                              has_unified_pp, start);
        if (!success && forward_cache_eligible)
        {
            /*
             * First-use materialization is transactional. Leaving an invalid
             * entry behind would preserve graph-owned streams, events, and
             * pointers from a failed capture and make the next invocation look
             * like a reusable cache object. Erase the entire half-built entry;
             * a later request must construct a fresh ownership lifetime.
             */
            cache_.erase(forward_signature);
        }
        if (live_mtp_request_batch_condition &&
            debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            LOG_INFO(
                "[MTPConditionGraphContract] event=execution_complete"
                << " device=" << effective_input.device.toString()
                << " cache=miss"
                << " success=" << boolTag(success));
        }
        if (success && build_cache && build_cache->valid)
            recordLastExecutedForwardGraph(
                forward_signature,
                /*cache_hit=*/false,
                output.execution.stream);
        if (setup_materialization)
        {
            if (success && !forward_signature.decode)
                enforcePrefillForwardCapacity(&forward_signature);
            return success;
        }
        if (success && !output.execution.valid)
        {
            LOG_ERROR("[ForwardExecutionEngine] Successful forward did not publish execution provenance for "
                      << effective_input.device.toString());
            return false;
        }
        if (success)
            host.commitSuccessfulForwardOutput(output);
        return success;
    }

    void ForwardExecutionEngine::recordLastExecutedForwardGraph(
        const ForwardGraphSignature &signature,
        bool cache_hit,
        void *producer_stream)
    {
        last_executed_forward_graph_.valid = true;
        last_executed_forward_graph_.signature = signature;
        last_executed_forward_graph_.cache_hit = cache_hit;
        last_executed_forward_graph_.producer_stream = producer_stream;
        if (signature.execution_role ==
                ForwardExecutionRole::GroupedMTPVerifier &&
            signature.decode &&
            signature.all_position_logits)
        {
            last_all_position_verifier_graph_.valid = true;
            last_all_position_verifier_graph_.signature = signature;
            last_all_position_verifier_graph_.cache_hit = cache_hit;
            last_all_position_verifier_graph_.producer_stream =
                producer_stream;
        }
    }

    std::optional<ForwardExecutionEngine::LastExecutedForwardGraphView>
    ForwardExecutionEngine::lastExecutedForwardGraph()
    {
        return viewForLastExecutedForwardGraphState(last_executed_forward_graph_);
    }

    std::optional<ForwardExecutionEngine::LastExecutedForwardGraphView>
    ForwardExecutionEngine::lastAllPositionVerifierForwardGraph()
    {
        return viewForLastExecutedForwardGraphState(last_all_position_verifier_graph_);
    }

    std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
    ForwardExecutionEngine::lastExecutedDeviceLoopGraphTemplate(
        std::string *error) const
    {
        return deviceLoopGraphTemplateForState(
            last_executed_forward_graph_, error);
    }

    std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
    ForwardExecutionEngine::lastAllPositionVerifierDeviceLoopGraphTemplate(
        std::string *error) const
    {
        return deviceLoopGraphTemplateForState(
            last_all_position_verifier_graph_, error);
    }

    std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
    ForwardExecutionEngine::deviceLoopGraphTemplate(
        const ForwardGraphSignature &signature,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<DeviceLoopGraphTemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();

        const auto cache_it = cache_.find(signature);
        if (cache_it == cache_.end())
        {
            return reject(
                "no cached forward graph matches the exact signature");
        }
        if (!cache_it->second.valid || !cache_it->second.graph)
        {
            return reject(
                "the exact forward graph cache entry is not valid");
        }

        const ForwardGraphCache &cache = cache_it->second;
        if (!signature.device.is_gpu() || !signature.decode)
            return reject("device-loop composition requires a GPU decode graph");
        if (!signature.uses_device_token_ids ||
            !signature.uses_device_position_ids)
        {
            return reject(
                "device-loop composition requires device-owned token and position rows");
        }
        if (!cache.phase3_active)
            return reject("forward graph is not replay-ready");

        std::string template_error;
        const auto template_view = cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            &template_error);
        if (!template_view)
            return reject(template_error);

        DeviceLoopGraphTemplateView view;
        view.capture = template_view->capture;
        view.signature = signature;
        view.device = signature.device;
        view.stream = template_view->stream;
        view.stage_count = template_view->stage_count;
        view.captured_node_count = template_view->captured_node_count;
        return view;
    }

    std::optional<ForwardExecutionEngine::RetainedDecodeGraphView>
    ForwardExecutionEngine::retainedDecodeGraph(
        const ForwardGraphSignature &signature,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<RetainedDecodeGraphView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();

        const auto cache_it = cache_.find(signature);
        if (cache_it == cache_.end())
            return reject("no cached forward graph matches the exact signature");

        const ForwardGraphCache &cache = cache_it->second;
        if (!cache.valid || !cache.graph)
            return reject("the exact forward graph cache entry is not valid");
        if (!signature.device.is_gpu() || !signature.decode)
            return reject("retained replay requires a GPU decode graph");
        if (!signature.uses_device_token_ids ||
            !signature.uses_device_position_ids)
        {
            return reject(
                "retained replay requires device-owned token and position rows");
        }
        if (!cache.phase3_active || !cache.segment_cache.initialized ||
            cache.segment_cache.needs_capture)
        {
            return reject(
                "forward graph has not reached steady retained replay state");
        }
        if (!cache.segment_cache.capture_stream || !cache.gpu_ctx ||
            cache.segment_cache.segments.empty())
        {
            return reject(
                "retained forward replay has incomplete stream, context, or segment ownership");
        }

        RetainedDecodeGraphView view;
        view.signature = signature;
        view.device = signature.device;
        view.stream = cache.segment_cache.capture_stream;
        const auto &parent_plan =
            cache.segment_cache.retained_composed_parent_replay;
        if (parent_plan.valid())
        {
            if (cache.segment_cache.graph_replay_plan_policy !=
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        RequireRetainedParentComposition ||
                !cache.segment_cache.retained_parent_capture ||
                !cache.segment_cache.retained_parent_capture->hasExecutable())
            {
                return reject(
                    "retained parent identity is incomplete or disagrees with replay policy");
            }
            view.replay_unit_count = 1u;
            view.source_capture_unit_count = parent_plan.child_unit_count;
            view.composed_parent = true;
            view.segmented = false;
        }
        else
        {
            view.replay_unit_count = cache.segment_cache.segments.size();
            view.source_capture_unit_count = view.replay_unit_count;
            view.segmented = view.replay_unit_count > 1;
        }
        return view;
    }

    std::optional<std::vector<
        DeviceGraphExecutor::GraphSegmentCache::
            RetainedCaptureUnitTemplateView>>
    ForwardExecutionEngine::retainedDecodeCaptureUnitTemplates(
        const ForwardGraphSignature &signature,
        std::string *error) const
    {
        using UnitView = DeviceGraphExecutor::GraphSegmentCache::
            RetainedCaptureUnitTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<std::vector<UnitView>>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();

        std::string retained_error;
        if (!retainedDecodeGraph(signature, &retained_error))
            return reject(retained_error);

        const auto cache_it = cache_.find(signature);
        if (cache_it == cache_.end() || !cache_it->second.graph)
        {
            return reject(
                "retained decode capture-unit cache identity disappeared");
        }
        return cache_it->second.segment_cache.retainedCaptureUnitTemplates(
            *cache_it->second.graph, error);
    }

    bool ForwardExecutionEngine::replayRetainedDecodeGraph(
        const ForwardGraphSignature &signature,
        IDeviceContext *ctx,
        const IComputeStage::MoEOverlayCollectiveRuntimeParams &sparse_params,
        const DeviceGraphExecutor::GraphLaunchDependencyHook
            &launch_dependency,
        const GraphCaptureAuxiliaryBranchFactory
            &auxiliary_branch_factory,
        void *transaction_stream,
        void **out_producer_stream,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            return false;
        };
        if (error)
            error->clear();
        if (out_producer_stream)
            *out_producer_stream = nullptr;

        std::string retained_error;
        const auto retained = retainedDecodeGraph(signature, &retained_error);
        if (!retained)
            return fail(retained_error);
        if (!ctx || ctx->deviceId() != signature.device)
            return fail("retained replay received the wrong device context");
        if (!transaction_stream)
        {
            return fail(
                "retained replay requires a non-null external transaction stream");
        }

        auto cache_it = cache_.find(signature);
        if (cache_it == cache_.end())
            return fail("retained replay cache identity disappeared");
        ForwardGraphCache &cache = cache_it->second;
        if (!cache.gpu_ctx ||
            !cache.segment_cache.orderCaptureStreamAfter(
                cache.gpu_ctx, transaction_stream))
        {
            return fail(
                "retained replay could not consume its external transaction stream");
        }

        if (!cache.moe_overlay_collective_runtime_stages.empty() &&
            !sparse_params.valid())
        {
            return fail(
                "retained sparse replay requires a root-authoritative wire identity");
        }
        for (IComputeStage *stage :
             cache.moe_overlay_collective_runtime_stages)
        {
            if (!stage)
                return fail("retained sparse replay contains a null runtime stage");
            stage->updateMoEOverlayCollectiveRuntimeParams(sparse_params);
        }

        const auto plan_policy =
            cache.segment_cache.graph_replay_plan_policy;
        DeviceGraphExecutor::RetainedParentCompositionHook parent_composer;
        if (retained->composed_parent)
        {
            if (plan_policy !=
                DeviceGraphExecutor::GraphReplayPlanPolicy::
                    RequireRetainedParentComposition)
            {
                return fail(
                    "retained parent view disagrees with cache replay policy");
            }
            try
            {
                auto composer = makeMoEOverlayRetainedParentComposer(*cache.graph);
                if (!composer)
                {
                    return fail(
                        "retained parent graph no longer declares typed packet frontiers");
                }
                parent_composer = std::move(*composer);
            }
            catch (const std::exception &ex)
            {
                return fail(
                    std::string(
                        "retained parent topology validation failed: ") +
                    ex.what());
            }
        }

        /*
         * Phase-3 graph units do not call markCompleted(), while manual sparse
         * boundaries retain ordinary ComputeGraph stage flags. Resetting the
         * declarative graph here clears only host lifecycle flags; the retained
         * executables, streams, workspaces, and device state remain untouched.
         */
        cache.graph->reset();
        if (!executor_.executeWithCachedGraphReplay(
                *cache.graph,
                ctx,
                cache.segment_cache,
                retained->stream,
                cache.gpu_ctx,
                &cache.collective_nodes,
                /*collectives_graph_capturable=*/
                    !cache.collective_nodes.empty(),
                /*force_recapture=*/false,
                /*defer_final_sync=*/true,
                {},
                plan_policy,
                launch_dependency,
                std::span<const BufferId>{},
                std::move(parent_composer),
                DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                    CaptureInstantiateAndLaunch,
                auxiliary_branch_factory))
        {
            return fail("retained decode graph replay submission failed");
        }
        if (!cache.segment_cache.orderStreamAfterCapture(
                cache.gpu_ctx, transaction_stream))
        {
            return fail(
                "retained replay could not publish its exact terminal to the external transaction stream");
        }

        if (out_producer_stream)
            *out_producer_stream = retained->stream;
        PerfStatsCollector::addCounter(
            "forward_graph",
            "hosted_retained_decode_replays",
            1.0,
            "decode",
            signature.device.toString(),
            {{"replay_units", std::to_string(retained->replay_unit_count)},
             {"source_capture_units",
              std::to_string(retained->source_capture_unit_count)},
             {"segmented", retained->segmented ? "true" : "false"},
             {"composed_parent",
              retained->composed_parent ? "true" : "false"},
             {"role", std::to_string(static_cast<int>(
                          signature.execution_role))}});
        return true;
    }

    void ForwardExecutionEngine::clearLastAllPositionVerifierForwardGraph()
    {
        last_all_position_verifier_graph_ = {};
    }

    std::optional<ForwardExecutionEngine::LastExecutedForwardGraphView>
    ForwardExecutionEngine::viewForLastExecutedForwardGraphState(
        const LastExecutedForwardGraphState &state)
    {
        if (!state.valid)
            return std::nullopt;

        auto it = cache_.find(state.signature);
        if (it == cache_.end() || !it->second.valid || !it->second.graph)
            return std::nullopt;

        LastExecutedForwardGraphView view;
        view.graph = it->second.graph.get();
        view.signature = state.signature;
        view.device = view.signature.device;
        view.stream = state.producer_stream;
        view.cache_hit = state.cache_hit;
        view.is_decode = view.signature.decode;
        view.all_position_logits = view.signature.all_position_logits;
        return view;
    }

    std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
    ForwardExecutionEngine::deviceLoopGraphTemplateForState(
        const LastExecutedForwardGraphState &state,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<DeviceLoopGraphTemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state.valid)
            return reject("no successful forward graph is retained");

        auto view = deviceLoopGraphTemplate(state.signature, error);
        if (!view && error &&
            *error == "no cached forward graph matches the exact signature")
        {
            *error = "retained forward graph no longer owns a valid cache entry";
        }
        return view;
    }

    std::vector<ForwardExecutionEngine::ReplayCacheObservation>
    ForwardExecutionEngine::replayCacheObservations(uint64_t live_state_epoch) const
    {
        std::vector<ReplayCacheObservation> observations;
        observations.reserve(cache_.size());
        for (const auto &[signature, cache] : cache_)
        {
            ReplayCacheObservation observation;
            observation.signature = signature;
            observation.valid = cache.valid;
            observation.segment_initialized = cache.segment_cache.initialized;
            observation.segment_needs_capture = cache.segment_cache.needs_capture;
            observation.phase3_active = cache.phase3_active;
            observation.gpu_stream_bindings_applied = cache.gpu_stream_applied;
            observation.has_capture_stream =
                cache.segment_cache.capture_stream != nullptr;
            observation.segment_decode_step = cache.segment_cache.decode_step;
            observation.graph_replay_live_state_epoch =
                cache.graph_replay_live_state_epoch;
            observation.requires_live_state_epoch_recapture =
                cache.requiresLiveStateEpochRecapture(
                    isLiveStateVersionedReplayCache(signature),
                    /*graph_replay_allowed=*/true,
                    live_state_epoch);
            observation.all_position_verifier_recapture_pending =
                all_position_verifier_recapture_pending_;
            observations.push_back(observation);
        }
        return observations;
    }

    std::vector<ForwardExecutionEngine::PrefillGraphCacheSnapshot>
    ForwardExecutionEngine::prefillGraphCacheSnapshots() const
    {
        std::vector<PrefillGraphCacheSnapshot> snapshots;
        snapshots.reserve(cache_.size());
        for (const auto &[signature, cache] : cache_)
        {
            if (signature.decode)
                continue;

            PrefillGraphCacheSnapshot snapshot;
            snapshot.forward_cache_valid = cache.valid;
            snapshot.eviction_count = prefill_forward_eviction_count_;
            snapshot.bucket_seq_len = signature.is_bucketed_prefill
                                          ? signature.bucket_seq_len
                                          : signature.seq_len;
            snapshot.placement_epoch = signature.moe_placement_epoch;
            snapshot.capture_phase = "unknown";
            snapshot.recapture_reason = "none";

            PrefillGraphCacheKey key;
            key.seq_len = snapshot.bucket_seq_len;
            key.device_id = signature.device;
            key.placement_epoch = signature.moe_placement_epoch;

            if (cache.last_prefill_graph_observation.valid)
            {
                const auto &observation = cache.last_prefill_graph_observation;
                snapshot.observation_valid = true;
                snapshot.chunk_index = observation.chunk_index;
                snapshot.bucket_seq_len = observation.bucket_seq_len;
                snapshot.real_token_start = observation.real_token_start;
                snapshot.real_token_count = observation.real_token_count;
                snapshot.real_token_end = observation.real_token_end;
                snapshot.domain_id = observation.domain_id;
                snapshot.participant_id = observation.participant_id;
                snapshot.placement_epoch = observation.placement_epoch;
                snapshot.topology_signature = observation.topology_signature;
                snapshot.capture_phase = observation.capture_phase;
                snapshot.recapture_reason = observation.recapture_reason;
                snapshot.reject_stage_name = observation.reject_stage_name;
                snapshot.reject_stage_type = observation.reject_stage_type;

                key.seq_len = observation.bucket_seq_len;
                key.domain_id = observation.domain_id;
                key.participant_id = observation.participant_id;
                key.placement_epoch = observation.placement_epoch;
                key.topology_signature = observation.topology_signature;
            }

            if (cache.segment_cache.initialized ||
                cache.segment_cache.successful_capture_count != 0u)
            {
                applySegmentedPrefillGraphSnapshot(
                    cache.segment_cache, &snapshot);
            }
            else if (cache.prefill_graph_cache)
            {
                const PrefillGraphCache &prefill_cache = *cache.prefill_graph_cache;
                snapshot.prefill_cache_initialized = true;
                snapshot.phase = prefill_cache.phase(key);
                snapshot.cache_size = prefill_cache.size();
                snapshot.node_count = prefill_cache.nodeCount(key);
                snapshot.replay_count = prefill_cache.replayCount(key);
                snapshot.warmup_count = prefill_cache.warmupCount(key);
                snapshot.initialized_count = prefill_cache.initializedCount(key);
                snapshot.capture_count = prefill_cache.captureCount(key);
                snapshot.eviction_count += prefill_cache.evictionCount();
            }

            snapshots.push_back(std::move(snapshot));
        }
        return snapshots;
    }

    bool ForwardExecutionEngine::materializeCachedExecutableWithoutLaunch(
        const ForwardInput &input,
        ForwardGraphCache &forward_cache,
        IForwardExecutionHost &host,
        bool is_decode)
    {
        if (input.graph_submission_intent !=
                ForwardGraphSubmissionIntent::
                    MaterializeExecutableWithoutLaunch ||
            !input.device.is_gpu() || !forward_cache.graph ||
            input.moe_overlay_graph_launch_dependency)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup graph materialization requires "
                "one cached GPU graph and forbids request-scoped launch "
                "dependencies");
            return false;
        }
        if (is_decode && forward_cache.segment_cache.initialized &&
            !forward_cache.segment_cache.needs_capture &&
            forward_cache.segment_cache.executable_submission_state !=
                DeviceGraphExecutor::GraphSegmentCache::
                    ExecutableSubmissionState::Empty)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "setup_materialized_graph_reuses",
                1.0,
                "setup",
                input.device.toString(),
                {{"context", is_decode ? "main_decode" : "prefill"}});
            return true;
        }

        IDeviceContext *const ctx = host.getDeviceContext(input.device);
        IWorkerGPUContext *const gpu_ctx =
            ctx ? host.getWorkerGPUContext(ctx->deviceId()) : nullptr;
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup graph materialization could "
                "not establish its exact device context on "
                << input.device.toString());
            return false;
        }
        forward_cache.gpu_ctx = gpu_ctx;
        forward_cache.gpu_stream = gpu_ctx->defaultStream();
        if (!forward_cache.gpu_stream)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup graph materialization resolved "
                "a null worker stream on "
                << input.device.toString());
            return false;
        }

        constexpr auto kMaterialize =
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                MaterializeWithoutLaunch;
        if (!is_decode)
        {
            bool used_graph_replay = false;
            return executePrefillWithGraphCache(
                input,
                forward_cache,
                ctx,
                host,
                &used_graph_replay,
                nullptr,
                kMaterialize);
        }

        if (!forward_cache.segment_cache.ensureCaptureStream(
                gpu_ctx, ctx->deviceId()) ||
            !forward_cache.segment_cache.ensureCaptureInputEvent(gpu_ctx) ||
            !forward_cache.segment_cache.ensureCaptureOutputEvent(gpu_ctx) ||
            !forward_cache.segment_cache.capture_stream)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup decode graph materialization "
                "could not establish its exact capture stream on "
                << input.device.toString());
            return false;
        }

        /*
         * Setup capture has no admitted request and launches no model work.
         * Graph construction may nevertheless publish immutable device
         * descriptors. Join that producer on the exact decode capture stream so
         * the same event-owned state is visible to every graph in the family.
         */
        if (!host.prepareGraphBuildStateForMaterialization(
                input,
                forward_cache.segment_cache.capture_stream,
                ctx->deviceId()))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup decode graph materialization "
                "could not consume graph-build device-state readiness on "
                << input.device.toString());
            return false;
        }

        DeviceGraphExecutor::DecodeCapturePolicy capture_policy =
            host.buildDecodeCapturePolicy(
                !forward_cache.collective_nodes.empty(), ctx);
        if (!capture_policy.allow_cached_graph_replay)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup decode materialization requires "
                "the production cached-graph policy on "
                << input.device.toString());
            return false;
        }
        try
        {
            capture_policy.auxiliary_branch_factory =
                host.forwardGraphAuxiliaryBranchFactory(
                    input, ctx->deviceId());
            const GraphNativeCaptureEnvelope native_capture_envelope =
                forward_cache.graph->nativeCaptureEnvelope();
            std::string envelope_error;
            if (!DeviceGraphCaptureController::
                    constrainReplayPolicyToNativeEnvelope(
                        native_capture_envelope,
                        capture_policy,
                        &envelope_error))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Setup graph materialization "
                    "rejected its native capture envelope: "
                    << envelope_error);
                return false;
            }
            if (auto sparse_parent_composer =
                    makeMoEOverlayRetainedParentComposer(
                        *forward_cache.graph);
                native_capture_envelope ==
                        GraphNativeCaptureEnvelope::Ordinary &&
                    sparse_parent_composer)
            {
                capture_policy.graph_replay_plan_policy =
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        RequireRetainedParentComposition;
                capture_policy.retained_parent_composer =
                    std::move(*sparse_parent_composer);
                capture_policy.defer_final_sync = true;
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup decode materialization could "
                "not derive its retained ExpertOverlay topology: "
                << error.what());
            return false;
        }

        const DeviceId capture_device = ctx->deviceId();
        capture_policy.capture_boundary =
            [&host, &input, capture_device](
                const std::string &boundary_name,
                void *capture_stream)
        {
            return host.waitAtDecodeGraphCaptureBoundary(
                input,
                capture_device,
                boundary_name,
                capture_stream);
        };
        capture_policy.launch_dependency = {};
        capture_policy.force_recapture = false;
        forward_cache.segment_cache.perf_context = "main_decode";

        bool used_graph_replay = false;
        return executor_.executeDecodeWithCapturePolicy(
            *forward_cache.graph,
            ctx,
            &forward_cache.segment_cache,
            forward_cache.gpu_stream,
            forward_cache.gpu_ctx,
            &forward_cache.collective_nodes,
            capture_policy,
            &used_graph_replay,
            kMaterialize);
    }

    // =========================================================================
    // executeCacheHit() — Reuse Cached Forward Graph
    // =========================================================================

    bool ForwardExecutionEngine::executeCacheHit(
        const ForwardInput &input,
        ForwardOutput &output,
        ForwardGraphCache &forward_cache,
        IForwardExecutionHost &host,
        bool is_decode,
        std::chrono::high_resolution_clock::time_point start)
    {
        // ===== CACHE HIT: Reuse cached decode graph =====

        // Setup sub-phase timing (only when profiling is active)
        /*
         * The coarse forward-pass decomposition is intentionally available
         * without per-kernel or per-stage GPU events.  It adds only host clock
         * reads and thread-local accumulation, which lets a focused PerfStats
         * campaign attribute orchestration latency without changing the GPU
         * execution DAG it is trying to measure.
         */
        const bool profiling_setup =
            KernelProfiler::isEnabled() ||
            PerfStatsCollector::isDomainEnabled("forward_pass");
        using Clock = std::chrono::high_resolution_clock;
        Clock::time_point setup_workspace_t0, setup_workspace_t1;
        Clock::time_point setup_token_copy_t0, setup_token_copy_t1;
        Clock::time_point setup_stream_t0, setup_stream_t1;
        Clock::time_point setup_dynamic_params_t0, setup_dynamic_params_t1;
        Clock::time_point setup_graph_reset_t0, setup_graph_reset_t1;

        if (profiling_setup)
            setup_workspace_t0 = Clock::now();

        if (input.device.is_gpu() && forward_cache.graph)
        {
            // Fast-path: if workspace generation hasn't changed since our last
            // validation, skip the expensive O(N_stages) graph traversal inside
            // ensureDeviceWorkspaceAllocated. During steady-state decode (seq_len=1),
            // nothing triggers workspace reallocation, so this saves ~3.5ms/iter
            // by avoiding 1595 dynamic_casts + string comparisons every step.
            const uint64_t current_workspace_generation = host.workspaceGeneration(input.device);
            const bool workspace_validated = (forward_cache.workspace_generation != 0 &&
                                              current_workspace_generation == forward_cache.workspace_generation);

            if (!workspace_validated)
            {
                const WorkspaceGraphParticipantRole workspace_role =
                    workspaceParticipantRole(
                        input.execution_role,
                        is_decode);
                if (!host.ensureDeviceWorkspaceAllocated(
                        *forward_cache.graph,
                        input.seq_len,
                        workspaceFamilyPolicy(workspace_role),
                        workspace_role))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to refresh cached graph workspace for "
                              << input.device.toString() << " seq_len=" << input.seq_len);
                    return false;
                }

                const uint64_t new_generation = host.workspaceGeneration(input.device);
                if (new_generation != forward_cache.workspace_generation)
                {
                    if (forward_cache.workspace_generation != 0)
                    {
                        LOG_DEBUG("[ForwardExecutionEngine] Workspace generation changed on "
                                  << input.device.toString()
                                  << " from " << forward_cache.workspace_generation
                                  << " to " << new_generation
                                  << "; dropping captured replay state for cached graph");
                        forward_cache.resetReplayStateAfterWorkspaceRebind();
                    }
                    forward_cache.workspace_generation = new_generation;
                }
            }
        }

        if (profiling_setup)
        {
            setup_workspace_t1 = Clock::now();
            setup_token_copy_t0 = setup_workspace_t1;
        }

        // Update stable buffers — stages hold pointers to these, so the
        // pointed-to values change but the pointers remain valid
        const int total_tokens = input.batch_size * input.seq_len;
        if (input.token_ids)
        {
            if (static_cast<int>(forward_cache.token_ids.size()) == total_tokens)
            {
                std::memcpy(forward_cache.token_ids.data(), input.token_ids,
                            static_cast<size_t>(total_tokens) * sizeof(int));
            }
            else
            {
                forward_cache.token_ids.assign(input.token_ids,
                                               input.token_ids + total_tokens);
            }
        }
        if (input.position_ids)
        {
            if (static_cast<int>(forward_cache.position_ids.size()) == total_tokens)
            {
                std::memcpy(forward_cache.position_ids.data(), input.position_ids,
                            static_cast<size_t>(total_tokens) * sizeof(int));
            }
            else
            {
                forward_cache.position_ids.assign(input.position_ids,
                                                  input.position_ids + total_tokens);
            }
        }

        // PP hidden state copy: for non-embedding PP stages, copy the
        // external hidden state (from previous PP stage) to the working
        // buffer before executing the cached graph.
        if (forward_cache.pp_needs_copy &&
            forward_cache.pp_external_hidden_state &&
            forward_cache.pp_working_buffer)
        {
            // PP hidden-state handoff. copyActivation picks the cheapest transport
            // for the (source authoritative device -> pp_device) pair:
            //   - same physical GPU   -> intra-VRAM device-to-device memcpy
            //   - same-vendor diff GPU -> peer copy (NCCL/RCCL or peer DMA)
            //   - cross-vendor / host  -> single host-staged bounce
            // and leaves the destination DEVICE_AUTHORITATIVE on pp_device.
            //
            // In the steady-state heterogeneous CUDA->ROCm case the producing
            // stage's transferActivation() has already moved the external hidden
            // state onto pp_device (rocm:0), so this resolves to an intra-GPU D2D
            // copy. This replaces the old data()/memcpy/ensureOnDevice host
            // round-trip that cost ~30ms/token.
            const bool pp_time = is_decode && debugEnv().tp_timing;
            auto pp_t0 = pp_time ? Clock::now() : Clock::time_point{};

            auto pp_result = TransferEngine::instance().copyActivation(
                forward_cache.pp_external_hidden_state, forward_cache.pp_working_buffer,
                forward_cache.pp_device, forward_cache.pp_copy_bytes);

            // Fail loud: a failed hidden-state handoff corrupts the entire
            // downstream pipeline stage, so there is no safe fallback.
            if (!pp_result.success)
            {
                LOG_ERROR("[PP_COPY] copyActivation failed on "
                          << forward_cache.pp_device << ": " << pp_result.error);
                throw std::runtime_error("PP hidden-state copy failed: " + pp_result.error);
            }

            if (pp_time)
            {
                auto pp_t1 = Clock::now();
                double total_us =
                    std::chrono::duration<double, std::micro>(pp_t1 - pp_t0).count();
                LOG_DEBUG("[PP_COPY] dev=" << forward_cache.pp_device << " method="
                          << to_string(pp_result.method_used) << " total=" << total_us
                          << "us bytes=" << forward_cache.pp_copy_bytes);
            }
        }

        if (profiling_setup)
        {
            setup_token_copy_t1 = Clock::now();
            setup_stream_t0 = setup_token_copy_t1;
        }

        // For GPU graph replay: set an explicit capture stream on all stages
        // before dynamic params are updated. The stream is owned by the segment
        // cache and survives ordinary replay-state resets, so stages never need
        // to fall back to the device context's default stream between a failed
        // replay/capture and the next warmup attempt.
        auto apply_cached_graph_stream = [&](void *stream)
        {
            if (!stream)
                return;
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage)
                    node->stage->setGPUStream(stream);
            }
            forward_cache.gpu_stream_applied = true;
            forward_cache.applied_stream = stream;
        };

        void *replay_stream = forward_cache.segment_cache.capture_stream;
        void *dynamic_param_stream = nullptr;
        IDeviceContext *stream_ctx = nullptr;
        IWorkerGPUContext *stream_gpu_ctx = nullptr;
        DeviceId stream_device = input.device;
        bool graph_gpu_stage_found = false;
        const auto &stream_order = forward_cache.graph->getExecutionOrder();
        for (const auto &node_name : stream_order)
        {
            ComputeNode *node = forward_cache.graph->getNode(node_name);
            if (!node || !node->stage)
                continue;
            const DeviceId stage_device = node->stage->device();
            if (stage_device.is_gpu())
            {
                stream_device = stage_device;
                graph_gpu_stage_found = true;
                break;
            }
            if (node->device.is_gpu())
            {
                stream_device = node->device;
                graph_gpu_stage_found = true;
                break;
            }
        }

        /*
         * GraphCacheConfig::decode_seq_len is a heuristic boundary for ordinary
         * continuation traffic. It is not an MTP capacity declaration. Typed MTP
         * roles have already passed their configured verifier-row validation and
         * are decode-equivalent for every supported M, including M values above
         * the legacy four-row threshold. Applying that heuristic here silently
         * sent grouped verifier M=5..15 through eager execution.
         */
        const bool typed_mtp_decode_role =
            input.execution_role ==
                ForwardExecutionRole::GroupedMTPVerifier ||
            input.execution_role ==
                ForwardExecutionRole::MTPCondition;
        const bool decode_capture_allowed =
            is_decode &&
            input.batch_size <= 1 &&
            (typed_mtp_decode_role ||
             input.seq_len <=
                 std::max(1, config_.cache_config.decode_seq_len));

        if (stream_device.is_gpu())
        {
            stream_ctx = host.getDeviceContext(stream_device);
            if (stream_ctx && stream_ctx->deviceId().is_gpu())
            {
                try
                {
                    stream_gpu_ctx = host.getWorkerGPUContext(stream_ctx->deviceId());
                    if (!stream_gpu_ctx)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for "
                                  << stream_ctx->deviceId().toString()
                                  << " before cached dynamic params");
                        return false;
                    }
                    forward_cache.gpu_ctx = stream_gpu_ctx;
                    if (!forward_cache.gpu_stream)
                        forward_cache.gpu_stream = stream_gpu_ctx->defaultStream();
                    dynamic_param_stream = forward_cache.gpu_stream;
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to resolve the mandatory worker GPU context for "
                              << stream_ctx->deviceId().toString()
                              << " before cached dynamic params: " << e.what());
                    return false;
                }
            }
        }

        bool heterogeneous_prefill_segmented = false;
        if ((decode_capture_allowed || !is_decode) &&
            stream_ctx && stream_ctx->deviceId().is_gpu() && stream_gpu_ctx)
        {
            const auto early_capture_policy = host.buildDecodeCapturePolicy(
                !forward_cache.collective_nodes.empty(),
                stream_ctx);

            const bool use_segment_cache_stream =
                decode_capture_allowed ||
                (!is_decode &&
                 early_capture_policy.heterogeneous_segmented_enabled);
            if (early_capture_policy.allow_cached_graph_replay &&
                use_segment_cache_stream)
            {
                if (forward_cache.segment_cache.ensureCaptureStream(
                        stream_gpu_ctx,
                        stream_ctx->deviceId()) &&
                    forward_cache.segment_cache.ensureCaptureInputEvent(
                        stream_gpu_ctx) &&
                    forward_cache.segment_cache.ensureCaptureOutputEvent(
                        stream_gpu_ctx))
                {
                    replay_stream = forward_cache.segment_cache.capture_stream;
                    dynamic_param_stream = replay_stream;
                    heterogeneous_prefill_segmented =
                        !is_decode &&
                        early_capture_policy.heterogeneous_segmented_enabled;
                }
                else
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to create the mandatory explicit decode graph stream for "
                              << stream_ctx->deviceId().toString());
                    return false;
                }
            }
        }

        if (!is_decode && stream_gpu_ctx && !heterogeneous_prefill_segmented)
        {
            if (forward_cache.prefill_capture_stream.ensure(stream_gpu_ctx))
            {
                dynamic_param_stream = forward_cache.prefill_capture_stream.stream;
            }
            else
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to create explicit prefill graph stream for "
                          << stream_ctx->deviceId().toString());
            }
        }

        // Dynamic-param stages can perform device-side uploads before replay
        // (for example ROCm attention's device-param row). Bind an explicit
        // stream before updateDynamicParams(); DeviceGraphExecutor's normal
        // per-stage stream binding happens later and is too late for this setup
        // phase. `defaultStream()` here is Llaminar's owned non-null stream, not
        // the HIP/CUDA legacy null stream.
        if (dynamic_param_stream && forward_cache.applied_stream != dynamic_param_stream)
        {
            apply_cached_graph_stream(dynamic_param_stream);
        }
        else if (stream_device.is_gpu())
        {
            LOG_DEBUG("[ForwardExecutionEngine] Cached graph dynamic-param stream prelude: input_device="
                      << input.device.toString()
                      << " stream_device=" << stream_device.toString()
                      << " graph_gpu_stage_found=" << graph_gpu_stage_found
                      << " stream_ctx=" << (stream_ctx != nullptr)
                      << " stream_gpu_ctx=" << (stream_gpu_ctx != nullptr)
                      << " dynamic_param_stream=" << dynamic_param_stream
                      << " applied_stream=" << forward_cache.applied_stream);
        }

        // Discovery is topology-owned and shared by setup capture and every
        // request execution. Only the small resulting list is walked per launch.
        if (!forward_cache.dynamic_param_stages_cached)
        {
            forward_cache.dynamic_param_stages =
                collectDynamicParamStages(*forward_cache.graph);
            forward_cache.dynamic_param_stages_cached = true;
        }

        if (input.graph_submission_intent ==
            ForwardGraphSubmissionIntent::
                MaterializeExecutableWithoutLaunch)
        {
            /*
             * Setup capture intentionally skips mutable request admission, but
             * native capture still requires every stable input slot to be ready.
             * In particular, host-token decode must enqueue its sentinel H2D copy
             * before beginCapture(); the embedding kernel correctly rejects an
             * H2D operation discovered from inside capture.
             */
            const int *setup_position_ids =
                selectForwardReplayHostPositionIds(forward_cache, input);
            if (!updateDynamicParamStages(
                    input,
                    forward_cache.dynamic_param_stages,
                    setup_position_ids,
                    "setup graph materialization"))
            {
                return false;
            }
            return materializeCachedExecutableWithoutLaunch(
                input,
                forward_cache,
                host,
                is_decode);
        }

        if (profiling_setup)
        {
            setup_stream_t1 = Clock::now();
            setup_dynamic_params_t0 = setup_stream_t1;
        }

        if (!host.prepareLiveStateForForwardGraphExecution(
                input,
                dynamic_param_stream,
                stream_device))
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to prepare live state for cached forward graph execution");
            return false;
        }
        ScopedForwardLiveStateRead live_state_read(
            host,
            input,
            dynamic_param_stream,
            stream_device);

        if (!host.prepareDeviceTokenInputsForForwardGraphExecution(
                input,
                dynamic_param_stream,
                stream_device))
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to prepare cached forward graph device-token inputs");
            return false;
        }

        if (host.computeAllPositionLogitsEnabled() &&
            host.allPositionLogitRows() > 0)
        {
            if (!host.prepareAllPositionVerifierGraphMetadata(
                    input,
                    dynamic_param_stream,
                    stream_device))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to prepare cached all-position verifier graph metadata");
                return false;
            }
        }

        if (!forward_cache.prefill_replay_param_stages_cached)
        {
            forward_cache.prefill_replay_param_stages.clear();
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage && node->stage->hasPrefillReplayParams())
                    forward_cache.prefill_replay_param_stages.push_back(node->stage.get());
            }
            forward_cache.prefill_replay_param_stages_cached = true;
        }
        if (!is_decode)
        {
            updatePrefillReplayParamStages(input, forward_cache.prefill_replay_param_stages);
        }
        if (!forward_cache.moe_overlay_collective_runtime_stages_cached)
        {
            forward_cache.moe_overlay_collective_runtime_stages =
                collectMoEOverlayCollectiveRuntimeStages(*forward_cache.graph);
            forward_cache.moe_overlay_collective_runtime_stages_cached = true;
        }
        updateMoEOverlayCollectiveRuntimeStages(
            input,
            forward_cache.moe_overlay_collective_runtime_stages);
        const int *replay_position_ids =
            selectForwardReplayHostPositionIds(forward_cache, input);
        if (!updateDynamicParamStages(
                input,
                forward_cache.dynamic_param_stages,
                replay_position_ids,
                "cached graph replay"))
        {
            return false;
        }

        if (profiling_setup)
        {
            setup_dynamic_params_t1 = Clock::now();
            setup_graph_reset_t0 = setup_dynamic_params_t1;
        }

        // Skip graph reset when Phase 3 replay is active — Phase 3 doesn't
        // call markCompleted(), so all flags are already false from last reset.
        if (!forward_cache.phase3_active)
        {
            forward_cache.graph->reset();
        }

        if (profiling_setup)
        {
            setup_graph_reset_t1 = Clock::now();
        }

        if (!host.waitBeforeForwardGraphExecution(
                input,
                input.device,
                /*cache_miss=*/false))
        {
            LOG_ERROR("[ForwardExecutionEngine] Forward graph pre-execution rendezvous failed for "
                      << input.device.toString());
            return false;
        }

        output = forward_cache.output;

        // Execute with single device context (standard path, no PP)
        IDeviceContext *ctx = host.getDeviceContext(input.device);
        if (!ctx)
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to get device context");
            return false;
        }

        bool success;
        const bool has_collective_nodes = !forward_cache.collective_nodes.empty();

        // Stable decode and verifier shapes use one complete captured graph on
        // homogeneous GPUs. A mixed-device collective domain may explicitly
        // admit collective-only segmentation through the topology policy.
        // Prefill uses its separate bucketed graph-cache state machine below.
        bool used_graph_replay = false;
        void *prefill_producer_stream = nullptr;
        bool requested_deferred_all_position_sync = false;
        bool requested_deferred_main_decode_sync = false;
        bool executed_deferred_all_position_sync = false;
        bool executed_deferred_main_decode_sync = false;

        auto exec_t0 = std::chrono::high_resolution_clock::now();
        if (profiling_setup)
        {
            ForwardPassProfiler::resetReplayTimings();
        }

        if (!is_decode)
        {
            if (input.device.is_gpu())
            {
                // GPU prefill uses the native capture/replay state machine.
                // Snapshot capture is a post-graph drain so parity diagnostics
                // never force eager stage execution.
                success = executePrefillWithGraphCache(
                    input,
                    forward_cache,
                    ctx,
                    host,
                    &used_graph_replay,
                    &prefill_producer_stream);
            }
            else
            {
                /*
                 * CPU exact prefill reuses the complete graph topology without
                 * pretending that CPU execution is a GPU graph replay. The
                 * graph owns its prepared kernels and persistent stage state;
                 * reset/update above refreshes request data before each eager
                 * traversal. Keep snapshots bound to this exact graph just as
                 * on first materialization.
                 */
                success = executor_.executeWithSnapshotManifest(
                    *forward_cache.graph,
                    ctx,
                    forward_cache.snapshot_manifest);
            }
        }
        else
        {
            DeviceGraphExecutor::DecodeCapturePolicy capture_policy;
            if (decode_capture_allowed)
            {
                capture_policy = host.buildDecodeCapturePolicy(
                    has_collective_nodes,
                    ctx);
            }
            if (input.moe_overlay_graph_launch_dependency)
            {
                if (capture_policy.launch_dependency)
                {
                    const auto existing =
                        std::move(capture_policy.launch_dependency);
                    const auto overlay =
                        input.moe_overlay_graph_launch_dependency;
                    capture_policy.launch_dependency =
                        [existing, overlay](
                            DeviceGraphExecutor::GraphExecutableLaunchPhase phase,
                            void *stream)
                    {
                        return existing(phase, stream) &&
                               overlay(phase, stream);
                    };
                }
                else
                {
                    capture_policy.launch_dependency =
                        input.moe_overlay_graph_launch_dependency;
                }
                if (!capture_policy.allow_cached_graph_replay)
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] GPU ExpertOverlay decode "
                        "requires retained graph execution so its remote "
                        "ticket can be armed at the exact executable launch "
                        "edge");
                    return false;
                }
            }
            const GraphNativeCaptureEnvelope native_capture_envelope =
                forward_cache.graph->nativeCaptureEnvelope();
            const bool uses_device_timeline_transaction =
                requiresSingleNativeExecutable(native_capture_envelope);
            const bool uses_heterogeneous_ticket_transaction =
                requiresHeterogeneousTicketSegmentation(
                    native_capture_envelope);
            bool uses_retained_sparse_parent = false;
            try
            {
                capture_policy.auxiliary_branch_factory =
                    host.forwardGraphAuxiliaryBranchFactory(
                        input, ctx->deviceId());
                std::string envelope_error;
                if (!DeviceGraphCaptureController::
                        constrainReplayPolicyToNativeEnvelope(
                            native_capture_envelope,
                            capture_policy,
                            &envelope_error))
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] Decode rejected its graph-native capture envelope: "
                        << envelope_error);
                    return false;
                }
                if (auto sparse_parent_composer =
                        makeMoEOverlayRetainedParentComposer(
                            *forward_cache.graph);
                    native_capture_envelope ==
                            GraphNativeCaptureEnvelope::Ordinary &&
                        sparse_parent_composer)
                {
                    if (!decode_capture_allowed ||
                        !capture_policy.allow_cached_graph_replay)
                    {
                        LOG_ERROR(
                            "[ForwardExecutionEngine] Typed ExpertOverlay packet graph requires mandatory retained-parent capture/replay");
                        return false;
                    }
                    capture_policy.graph_replay_plan_policy =
                        DeviceGraphExecutor::GraphReplayPlanPolicy::
                            RequireRetainedParentComposition;
                    capture_policy.retained_parent_composer =
                        std::move(*sparse_parent_composer);
                    uses_retained_sparse_parent = true;
                }
            }
            catch (const std::exception &ex)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Could not derive retained ExpertOverlay endpoint authority: "
                    << ex.what());
                return false;
            }
            if (typed_mtp_decode_role &&
                input.device.is_gpu() &&
                !capture_policy.allow_cached_graph_replay)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] GPU MTP role requires cached "
                    "full-graph replay for every validated row count"
                    << " role="
                    << static_cast<int>(input.execution_role)
                    << " seq_len=" << input.seq_len
                    << " device=" << input.device.toString());
                return false;
            }
            if (capture_policy.heterogeneous_segmented_enabled)
            {
                LOG_DEBUG(
                    "[ForwardExecutionEngine] Heterogeneous collective-only "
                    "segmented GPU-graph replay admitted by topology policy");
            }

            const bool all_position_verifier =
                host.computeAllPositionLogitsEnabled();
            const bool pending_all_position_verifier_recapture =
                all_position_verifier &&
                input.seq_len > 1 &&
                all_position_verifier_recapture_pending_;
            if (all_position_verifier &&
                DebugEnv::isTruthyEnv("LLAMINAR_MTP_FORCE_VERIFIER_GRAPH_RECAPTURE"))
            {
                /*
                 * Diagnostic-only recapture splitter: Phase 10 tuning needs to
                 * distinguish stale all-position verifier graph inputs from
                 * stale MTP sidecar graph inputs.  This switch deliberately
                 * affects only verifier graphs so the benchmark matrix can
                 * identify the stale replay owner without disabling GPU graphs
                 * globally.
                 */
                capture_policy.force_recapture = true;
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "main_verifier_force_recapture_diagnostic",
                    1.0,
                    "decode",
                    input.device.toString());
            }
            if (pending_all_position_verifier_recapture)
            {
                /*
                 * Shifted-MTP-KV catch-up changes verifier input state without
                 * broadly invalidating ordinary decode or sidecar replay.  The
                 * next all-position verifier must therefore refresh its GPU
                 * executable before publication consumes verifier rows.  Keep
                 * this as a one-shot structural recapture instead of relying on
                 * the diagnostic LLAMINAR_MTP_FORCE_VERIFIER_GRAPH_RECAPTURE
                 * environment switch.
                 */
                capture_policy.force_recapture = true;
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "main_verifier_pending_recapture",
                    1.0,
                    "decode",
                    input.device.toString());
            }
            if (capture_policy.allow_cached_graph_replay)
            {
                const DeviceId capture_device = ctx->deviceId();
                capture_policy.capture_boundary =
                    [&host, &input, capture_device](
                        const std::string &boundary_name,
                        void *capture_stream) -> bool
                {
                    return host.waitAtDecodeGraphCaptureBoundary(
                        input,
                        capture_device,
                        boundary_name,
                        capture_stream);
                };
            }
            const bool wants_all_position_sync_defer =
                all_position_verifier &&
                host.shouldDeferAllPositionVerifierFinalSync();
            const bool wants_main_decode_sync_defer =
                !all_position_verifier &&
                (host.shouldDeferMainDecodeFinalSync() ||
                 uses_retained_sparse_parent ||
                 uses_device_timeline_transaction ||
                 uses_heterogeneous_ticket_transaction);
            const bool wants_captured_collective_sync_defer =
                !all_position_verifier &&
                has_collective_nodes &&
                capture_policy.collectives_graph_capturable &&
                !debugEnv().gpu_stage_timing &&
                !debugEnv().gpu_stage_timing_detail;

            if (capture_policy.allow_cached_graph_replay &&
                wants_all_position_sync_defer)
            {
                capture_policy.defer_final_sync = true;
            }
            else if (capture_policy.allow_cached_graph_replay &&
                     (wants_main_decode_sync_defer ||
                      wants_captured_collective_sync_defer))
            {
                capture_policy.defer_final_sync = true;
            }
            if (uses_retained_sparse_parent ||
                uses_device_timeline_transaction ||
                uses_heterogeneous_ticket_transaction)
            {
                // The endpoint executable publishes an asynchronous device
                // timeline transaction; its consumer owns completion.
                capture_policy.defer_final_sync = true;
            }
            requested_deferred_all_position_sync =
                wants_all_position_sync_defer;
            requested_deferred_main_decode_sync =
                wants_main_decode_sync_defer ||
                wants_captured_collective_sync_defer;
            executed_deferred_all_position_sync =
                wants_all_position_sync_defer && capture_policy.defer_final_sync;
            executed_deferred_main_decode_sync =
                requested_deferred_main_decode_sync && capture_policy.defer_final_sync;

            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_capture_policy",
                1.0,
                "decode",
                input.device.toString(),
                {{"context", all_position_verifier ? "main_verifier" : "main_decode"},
                 {"allow_graph_replay", boolTag(capture_policy.allow_cached_graph_replay)},
                 {"defer_final_sync", boolTag(capture_policy.defer_final_sync)},
                 {"has_collectives", boolTag(has_collective_nodes)},
                 {"heterogeneous_segmented", boolTag(capture_policy.heterogeneous_segmented_enabled)},
                 {"retained_sparse_parent", boolTag(uses_retained_sparse_parent)},
                 {"device_timeline_transaction", boolTag(uses_device_timeline_transaction)},
                 {"heterogeneous_ticket_transaction",
                  boolTag(uses_heterogeneous_ticket_transaction)},
                 {"collectives_graph_capturable", boolTag(capture_policy.collectives_graph_capturable)},
                 {"replay_plan_policy",
                  capture_policy.graph_replay_plan_policy ==
                          DeviceGraphExecutor::GraphReplayPlanPolicy::
                              RequireFullGraph
                      ? "require_full_graph"
                      : (capture_policy.graph_replay_plan_policy ==
                                 DeviceGraphExecutor::GraphReplayPlanPolicy::
                                     RequireRetainedParentComposition
                             ? "require_retained_parent_composition"
                             : "allow_heterogeneous_boundary_segmentation")}});

            if (capture_policy.allow_cached_graph_replay && !forward_cache.gpu_stream)
            {
                DeviceId dev_id = ctx->deviceId();
                if (dev_id.is_gpu())
                {
                    IWorkerGPUContext *gpu_ctx = host.getWorkerGPUContext(dev_id);
                    if (!gpu_ctx)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for captured decode on "
                                  << dev_id.toString());
                        return false;
                    }
                    forward_cache.gpu_stream = gpu_ctx->defaultStream();
                    forward_cache.gpu_ctx = gpu_ctx;
                }
            }

            const bool main_decode_replay = !all_position_verifier;
            /*
             * Multi-token ordinary decode captures are live-state-versioned:
             * their graph mutates live KV/recurrent state inline.  All-position
             * verifier captures are deliberately not epoch-versioned here.
             * Verifier row state is published through stage-owned capture slots
             * and row metadata is refreshed before each launch, so preserving the
             * verifier executable is the canonical vLLM-style fast path.
             */
            const bool live_state_versioned_replay =
                main_decode_replay && input.seq_len > 1;
            const uint64_t live_state_epoch = host.liveReplayStateEpoch();
            if (forward_cache.requiresLiveStateEpochRecapture(
                    live_state_versioned_replay,
                    capture_policy.allow_cached_graph_replay,
                    live_state_epoch))
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "decode_graph_state_epoch_recapture",
                    1.0,
                    "decode",
                    input.device.toString(),
                    {{"context", all_position_verifier ? "main_verifier" : "main_decode"},
                     {"old_epoch", std::to_string(forward_cache.graph_replay_live_state_epoch)},
                     {"new_epoch", std::to_string(live_state_epoch)}});
                forward_cache.resetReplayState();
            }

            forward_cache.segment_cache.perf_context =
                all_position_verifier ? "main_verifier" : "main_decode";
            success = executor_.executeDecodeWithCapturePolicy(
                *forward_cache.graph,
                ctx,
                &forward_cache.segment_cache,
                forward_cache.gpu_stream,
                forward_cache.gpu_ctx,
                &forward_cache.collective_nodes,
                capture_policy,
                &used_graph_replay);
        }

        /*
         * Seal any live-state read admitted by the prelude before another
         * producer can reuse those persistent rows. This call is deliberately
         * outside the success branch: an executor failure can still follow
         * asynchronously enqueued GPU work, and that work retains ownership
         * until its exact stream reaches this event publication.
         */
        void *const exact_output_producer_stream = success
            ? (is_decode
                   ? forward_cache.decodeOutputProducerStream(
                         used_graph_replay
                             ? ForwardGraphCache::DecodeLaunchPath::CapturedReplay
                             : ForwardGraphCache::DecodeLaunchPath::Direct)
                   : prefill_producer_stream)
            : dynamic_param_stream;
        void *const live_state_completion_stream =
            exact_output_producer_stream;
        live_state_read.complete(live_state_completion_stream);

        if (success && !is_decode && !input.device.is_gpu() &&
            effectiveRealSeqLen(input) > 0)
        {
            const int terminal_row = effectiveRealSeqLen(input) - 1;
            if (!executor_.publishCapturedTerminalStateAfterGraphExecution(
                    *forward_cache.graph,
                    terminal_row,
                    nullptr,
                    "cpu_exact_prefill_cache_hit",
                    input.sequence_lengths_device,
                    input.batch_size,
                    input.seq_len,
                    input.sequence_lengths))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to publish terminal CPU prefill state after exact topology replay");
                return false;
            }
        }

        auto exec_t1 = std::chrono::high_resolution_clock::now();

        if (success && used_graph_replay &&
            forward_cache.segment_cache.initialized &&
            !forward_cache.segment_cache.needs_capture)
        {
            // Phase 3 replay doesn't call markCompleted(), so we can
            // skip graph.reset() on subsequent steps.
            forward_cache.phase3_active = true;
            if (is_decode)
            {
                const bool all_position_verifier =
                    host.computeAllPositionLogitsEnabled();
                const bool main_decode_replay = !all_position_verifier;
                const bool live_state_versioned_replay =
                    main_decode_replay && input.seq_len > 1;
                if (live_state_versioned_replay)
                {
                    forward_cache.graph_replay_live_state_epoch =
                        host.liveReplayStateEpoch();
                }
                if (all_position_verifier &&
                    all_position_verifier_recapture_pending_)
                {
                    /*
                     * Consume the request only once the cache is replay-ready.
                     * Warmup and capture phases are allowed to run after the
                     * request, but they are not sufficient evidence that later
                     * replay will use a fresh executable.
                     */
                    all_position_verifier_recapture_pending_ = false;
                    PerfStatsCollector::addCounter(
                        "forward_graph",
                        "main_verifier_pending_recapture_consumed",
                        1.0,
                        "decode",
                        input.device.toString());
                }
            }
        }
        else
        {
            forward_cache.phase3_active = false;
        }

        const bool all_position_verifier_sync_deferred =
            success &&
            used_graph_replay &&
            executed_deferred_all_position_sync &&
            forward_cache.segment_cache.capture_stream != nullptr;
        const bool main_decode_sync_deferred =
            success &&
            used_graph_replay &&
            executed_deferred_main_decode_sync &&
            forward_cache.segment_cache.capture_stream != nullptr;
        if (success)
        {
            /*
             * Publish invocation provenance while `used_graph_replay` is still
             * in scope. A decode replay must name the segment cache's typed
             * capture stream, never the generic stream most recently applied
             * to stage bindings. DeviceGraphOrchestrator immediately records
             * its durable output event on this exact producer stream.
             */
            publishForwardOutputProvenance(
                output,
                input.device,
                exact_output_producer_stream,
                input.execution_role,
                is_decode,
                host.computeAllPositionLogitsEnabled(),
                forwardCompletionScopeForInput(input),
                input.seq_len,
                input.batch_size);
            if (input.device.is_gpu() && !output.execution.valid)
            {
                LOG_ERROR("[ForwardExecutionEngine] Successful cached GPU forward has no exact output producer stream"
                          << " device=" << input.device.toString()
                          << " decode=" << is_decode
                          << " replay=" << used_graph_replay);
                return false;
            }

            /*
             * Grouped verifier state is always consumed asynchronously by a
             * sampler, accepted-state publisher, or diagnostic observer. Publish
             * its completion edge for every successful GPU invocation, including
             * warmup and first capture. Whether the caller also requests a
             * host-facing logits boundary is independent of this device-state
             * ownership contract.
             */
            if (is_decode &&
                input.device.is_gpu() &&
                host.computeAllPositionLogitsEnabled())
            {
                host.setPendingAllPositionVerifierStream(
                    output.execution.stream);
            }
            if (main_decode_sync_deferred)
            {
                host.setPendingMainDecodeStream(
                    output.execution.stream);
            }
            else if (requested_deferred_main_decode_sync)
            {
                host.setPendingMainDecodeStream(nullptr);
            }
        }

        // Publish logits at the forward ownership boundary. Device consumers
        // inherit the producer stream; host consumers wait only on the exact
        // tensor or replay completion event.
        if (success)
        {
            if (!all_position_verifier_sync_deferred &&
                !main_decode_sync_deferred)
            {
                /*
                 * The cached graph's ForwardOutput is immutable graph metadata:
                 * it names the exact local-shard, gathered, or replicated logits
                 * tensor selected when the graph was built. Publish that tensor
                 * after every warmup/capture/replay launch. A device consumer can
                 * instead inherit the producer stream through the deferred path
                 * above, so neither route needs a host or stream synchronization.
                 */
                if (!host.publishForwardResultAtBoundary(
                        output,
                        ctx))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to publish cached forward result at the device boundary");
                    return false;
                }
            }

            const std::string stage_context =
                is_decode
                    ? (forward_cache.segment_cache.perf_context.empty()
                           ? "main_decode"
                           : forward_cache.segment_cache.perf_context)
                    : "prefill";
            collectTimeline(host, ctx, is_decode, input, start, stage_context);
        }

        if (success)
        {
            recordMoEOverlayCollectiveTransaction(
                input,
                forward_cache.moe_overlay_collective_runtime_stages,
                "cache_hit");
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        /*
         * Keep one low-overhead end-to-end cache-hit timing in the same CSV as
         * the heterogeneous transport evidence.  The timer already brackets
         * every cache hit for the DEBUG summary below; exporting it therefore
         * adds no clock reads and lets Release campaigns distinguish capture
         * setup from steady graph replay without enabling log-heavy DEBUG.
         */
        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            PerfStatsCollector::recordTimingNs(
                "forward_graph",
                "forward_cache_hit_total",
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - start)
                        .count()),
                is_decode ? "decode" : "prefill",
                input.device.toString(),
                {{"phase3_active", forward_cache.phase3_active ? "true" : "false"},
                 {"sequence_rows", std::to_string(input.seq_len)},
                 {"used_graph_replay", used_graph_replay ? "true" : "false"}});
        }

        // Decode step timing breakdown (enabled via TP_TIMING)
        if (is_decode && debugEnv().tp_timing)
        {
            double setup_us = std::chrono::duration<double, std::micro>(exec_t0 - start).count();
            double exec_us = std::chrono::duration<double, std::micro>(exec_t1 - exec_t0).count();
            double sync_us = std::chrono::duration<double, std::micro>(end - exec_t1).count();
            std::ostringstream subphase;
            if (profiling_setup)
            {
                subphase << " [ws=" << std::chrono::duration<double, std::micro>(setup_workspace_t1 - setup_workspace_t0).count()
                         << "us tok=" << std::chrono::duration<double, std::micro>(setup_token_copy_t1 - setup_token_copy_t0).count()
                         << "us strm=" << std::chrono::duration<double, std::micro>(setup_stream_t1 - setup_stream_t0).count()
                         << "us dyn=" << std::chrono::duration<double, std::micro>(setup_dynamic_params_t1 - setup_dynamic_params_t0).count()
                         << "us rst=" << std::chrono::duration<double, std::micro>(setup_graph_reset_t1 - setup_graph_reset_t0).count() << "us]";
            }
            LOG_DEBUG("[DEVICE_DECODE] dev=" << input.device
                                             << " setup=" << std::fixed << std::setprecision(1) << setup_us << "us"
                                             << " exec=" << exec_us << "us"
                                             << " sync=" << sync_us << "us"
                                             << " total=" << (ms * 1000.0) << "us"
                                             << " phase3=" << forward_cache.phase3_active
                                             << subphase.str());
        }

        // Forward-pass host phases are exported through the explicit PerfStats timing gate.
        if (profiling_setup)
        {
            ForwardPassProfiler::PhaseTimings timings;
            timings.setup_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(exec_t0 - start).count());
            timings.execute_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(exec_t1 - exec_t0).count());
            timings.sync_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - exec_t1).count());

            // Setup sub-phase timings
            timings.setup_workspace_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_workspace_t1 - setup_workspace_t0).count());
            timings.setup_token_copy_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_token_copy_t1 - setup_token_copy_t0).count());
            timings.setup_stream_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_stream_t1 - setup_stream_t0).count());
            timings.setup_dynamic_params_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_dynamic_params_t1 - setup_dynamic_params_t0).count());
            timings.setup_graph_reset_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_graph_reset_t1 - setup_graph_reset_t0).count());

            // Graph replay sub-phase timings are populated by decode segmented
            // replay and prefill graph-cache launch paths via thread-local
            // ReplayPhaseTimings. Eager paths simply consume zeros.
            const auto &replay_timings = ForwardPassProfiler::consumeReplayTimings();
            timings.graph_launch_ns = replay_timings.graph_launch_ns;
            timings.post_launch_ns = replay_timings.post_launch_ns;
            timings.stream_sync_ns = replay_timings.stream_sync_ns;
            timings.manual_dispatch_ns =
                replay_timings.manual_dispatch_ns;
            timings.manual_sparse_protocol_ns =
                replay_timings.manual_sparse_protocol_ns;
            timings.manual_other_ns = replay_timings.manual_other_ns;

            if (is_decode)
                forward_pass_profiler_.recordDecodeIteration(
                    timings, input.device.toString());
            else
                forward_pass_profiler_.recordPrefillIteration(
                    timings, input.device.toString());
        }

        LOG_DEBUG("[ForwardExecutionEngine] Forward (cached) completed in "
                  << ms << "ms, success=" << success);

        return success;
    }

    // =========================================================================
    // executePrefillWithGraphCache() — Prefill Graph Capture/Replay State Machine
    // =========================================================================

    bool ForwardExecutionEngine::executePrefillWithGraphCache(
        const ForwardInput &input,
        ForwardGraphCache &forward_cache,
        IDeviceContext *ctx,
        IForwardExecutionHost &host,
        bool *used_graph_replay,
        void **out_producer_stream,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy
            initial_submission)
    {
        if (used_graph_replay)
            *used_graph_replay = false;
        if (out_producer_stream)
            *out_producer_stream = nullptr;
        const bool materialize_without_launch =
            initial_submission ==
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                MaterializeWithoutLaunch;
        // Initialize prefill graph cache on first use
        if (!forward_cache.prefill_graph_cache)
        {
            forward_cache.prefill_graph_cache =
                std::make_unique<PrefillGraphCache>(
                    makePrefillGraphConfigFromEnv(host.residentGraphRows()));
        }

        auto &cache = *forward_cache.prefill_graph_cache;
        const uint64_t snapshot_configuration_epoch =
            executor_.snapshotConfigurationEpoch();
        if (forward_cache.snapshot_configuration_epoch !=
            snapshot_configuration_epoch)
        {
            /*
             * Selected GPU snapshot stages become captured D2D nodes. Retire
             * prefill executables recorded under another callback/filter
             * topology before choosing the current key's phase. The stable
             * ComputeGraph and prepared weights remain reusable; the cache
             * simply performs its normal warmup/capture lifecycle again.
             */
            cache.invalidateAll();
            forward_cache.snapshot_manifest.clear();
            forward_cache.snapshot_configuration_epoch =
                snapshot_configuration_epoch;
            PerfStatsCollector::addCounter(
                "forward_graph",
                "prefill_snapshot_configuration_rewarm",
                1.0,
                "prefill",
                input.device.toString(),
                {{"snapshot_epoch",
                  std::to_string(snapshot_configuration_epoch)}});
        }

        PrefillGraphCacheKey key = makePrefillGraphKey(input, host);

        auto phase = cache.phase(key);
        const int real_seq_len = effectiveRealSeqLen(input);
        const int bucket_seq_len = effectiveBucketSeqLen(input);
        const bool padded_bucket = isPaddedBucketExecution(input);
        const bool snapshots_active = (executor_.config().snapshot_callback != nullptr);
        DeviceGraphExecutor::GraphSnapshotLogicalRows snapshot_logical_rows;
        if (padded_bucket && snapshots_active)
        {
            if (input.batch_size <= 0 || bucket_seq_len <= 0 ||
                real_seq_len <= 0 || real_seq_len > bucket_seq_len)
            {
                LOG_ERROR("[ForwardExecutionEngine] Padded prefill snapshot publication has invalid row geometry: batch_size="
                          << input.batch_size
                          << " real_seq_len=" << real_seq_len
                          << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }

            snapshot_logical_rows.physical_rows_per_sequence =
                static_cast<size_t>(bucket_seq_len);
            snapshot_logical_rows.logical_rows_per_sequence.reserve(
                static_cast<size_t>(input.batch_size));
            for (int batch = 0; batch < input.batch_size; ++batch)
            {
                const int logical_rows =
                    input.sequence_lengths &&
                            static_cast<size_t>(batch) <
                                input.sequence_lengths->size()
                        ? (*input.sequence_lengths)[static_cast<size_t>(batch)]
                        : real_seq_len;
                if (logical_rows <= 0 || logical_rows > bucket_seq_len)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Padded prefill snapshot publication has an invalid logical row count for batch "
                              << batch << ": logical_rows=" << logical_rows
                              << " bucket_seq_len=" << bucket_seq_len);
                    return false;
                }
                snapshot_logical_rows.logical_rows_per_sequence.push_back(
                    static_cast<size_t>(logical_rows));
            }
        }
        const auto *snapshot_logical_rows_ptr =
            snapshot_logical_rows.active() ? &snapshot_logical_rows : nullptr;
        if (snapshots_active)
        {
            /*
             * This log is deliberately confined to snapshot-enabled runs.
             * Fixed-width graph execution is allowed to retain padded device
             * storage, whereas the diagnostic callback must publish only the
             * request-owned prefix.  Reporting both descriptors here makes a
             * malformed handoff distinguishable from a later publisher bug
             * without adding work to ordinary inference.
             */
            LOG_DEBUG(
                "[ForwardExecutionEngine] Prefill snapshot row contract"
                << " chunk=" << input.prefill_chunk_index
                << " real_seq_len=" << real_seq_len
                << " bucket_seq_len=" << bucket_seq_len
                << " padded=" << padded_bucket
                << " batch_size=" << input.batch_size
                << " projection_active="
                << (snapshot_logical_rows_ptr != nullptr)
                << " logical_rows="
                << (snapshot_logical_rows_ptr &&
                            !snapshot_logical_rows.logical_rows_per_sequence.empty()
                        ? std::to_string(
                              snapshot_logical_rows
                                  .logical_rows_per_sequence.front())
                        : std::string("none")));
        }
        const bool moe_rebalancing_active = host.isMoeRebalancingActive();
        const bool moe_rebalancing_graph_stable =
            host.isMoeRebalancingGraphStableForPrefillCapture();
        const bool host_prefill_graph_disabled = host.prefillGraphCaptureDisabledByHost();
        auto prefill_capture_policy =
            host.buildDecodeCapturePolicy(
                !forward_cache.collective_nodes.empty(),
                ctx);
        if (input.moe_overlay_graph_launch_dependency)
        {
            if (prefill_capture_policy.launch_dependency)
            {
                const auto existing =
                    std::move(prefill_capture_policy.launch_dependency);
                const auto overlay =
                    input.moe_overlay_graph_launch_dependency;
                prefill_capture_policy.launch_dependency =
                    [existing, overlay](
                        DeviceGraphExecutor::GraphExecutableLaunchPhase phase,
                        void *stream)
                {
                    return existing(phase, stream) &&
                           overlay(phase, stream);
                };
            }
            else
            {
                prefill_capture_policy.launch_dependency =
                    input.moe_overlay_graph_launch_dependency;
            }
            if (!prefill_capture_policy.allow_cached_graph_replay)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] GPU ExpertOverlay prefill "
                    "requires retained graph execution so its remote ticket "
                    "can be armed at the exact executable launch edge");
                return false;
            }
        }
        const GraphNativeCaptureEnvelope native_capture_envelope =
            forward_cache.graph->nativeCaptureEnvelope();
        const bool uses_device_timeline_transaction =
            requiresSingleNativeExecutable(native_capture_envelope);
        const bool uses_heterogeneous_ticket_transaction =
            requiresHeterogeneousTicketSegmentation(
                native_capture_envelope);
        bool uses_retained_sparse_parent = false;
        try
        {
            prefill_capture_policy.auxiliary_branch_factory =
                host.forwardGraphAuxiliaryBranchFactory(
                    input, ctx->deviceId());
            std::string envelope_error;
            if (!DeviceGraphCaptureController::
                    constrainReplayPolicyToNativeEnvelope(
                        native_capture_envelope,
                        prefill_capture_policy,
                        &envelope_error))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Prefill rejected its graph-native capture envelope: "
                    << envelope_error);
                return false;
            }
            if (auto sparse_parent_composer =
                    makeMoEOverlayRetainedParentComposer(
                        *forward_cache.graph);
                native_capture_envelope ==
                        GraphNativeCaptureEnvelope::Ordinary &&
                    sparse_parent_composer)
            {
                if (!prefill_capture_policy.allow_cached_graph_replay)
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] Typed ExpertOverlay prefill packet graph requires mandatory retained-parent capture/replay");
                    return false;
                }
                prefill_capture_policy.graph_replay_plan_policy =
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        RequireRetainedParentComposition;
                prefill_capture_policy.retained_parent_composer =
                    std::move(*sparse_parent_composer);
                /*
                 * The retained parent is a single asynchronous device
                 * transaction. Terminal-state and snapshot publication are
                 * enqueued on its exact producer stream below; no host fence is
                 * needed between the sparse parent and those consumers.
                 */
                prefill_capture_policy.defer_final_sync = true;
                uses_retained_sparse_parent = true;
            }
        }
        catch (const std::exception &ex)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Could not derive retained ExpertOverlay prefill endpoint authority: "
                << ex.what());
            return false;
        }
        const bool prefill_collectives_graph_capturable =
            !forward_cache.collective_nodes.empty() &&
            prefill_capture_policy.collectives_graph_capturable;
        const bool prefill_heterogeneous_segmentation_admitted =
            prefill_capture_policy.heterogeneous_segmented_enabled ||
            uses_retained_sparse_parent ||
            uses_device_timeline_transaction ||
            uses_heterogeneous_ticket_transaction;
        bool padded_preflight_checked = false;
        PrefillGraphRejectReason padded_preflight_reason = PrefillGraphRejectReason::None;
        std::string padded_reject_stage_name;
        std::string padded_reject_stage_type;

        auto launchPrefillGraph = [&](IWorkerGPUContext *gpu_ctx,
                                      void *stream,
                                      const char *capture_phase,
                                      PrefillGraphPhase cache_phase,
                                      const char *launch_kind,
                                      const std::string &recapture_reason) -> bool
        {
            void *gpu_start_event = nullptr;
            void *gpu_stop_event = nullptr;
            const bool should_time_gpu =
                PerfStatsCollector::gpuStageEventTimingEnabled() &&
                gpu_ctx != nullptr &&
                stream != nullptr;
            bool gpu_event_timing_active = false;
            if (should_time_gpu)
            {
                gpu_start_event = gpu_ctx->createEvent();
                gpu_stop_event = gpu_ctx->createEvent();
                if (gpu_start_event && gpu_stop_event)
                {
                    gpu_ctx->recordEvent(gpu_start_event, stream);
                    gpu_event_timing_active = true;
                }
            }

            const auto launch_t0 = std::chrono::high_resolution_clock::now();
            const bool ok = cache.launch(key);
            const auto launch_t1 = std::chrono::high_resolution_clock::now();
            const auto ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(launch_t1 - launch_t0).count());

            auto destroy_gpu_events = [&]()
            {
                if (gpu_ctx)
                {
                    if (gpu_start_event)
                        gpu_ctx->destroyEvent(gpu_start_event);
                    if (gpu_stop_event)
                        gpu_ctx->destroyEvent(gpu_stop_event);
                }
                gpu_start_event = nullptr;
                gpu_stop_event = nullptr;
            };

            if (KernelProfiler::isEnabled())
            {
                ForwardPassProfiler::addReplayLaunchNs(ns);
            }

            if (PerfStatsCollector::isEnabled())
            {
                auto observation = makePrefillGraphObservation(
                    input,
                    key,
                    capture_phase,
                    recapture_reason);
                auto tags = prefillGraphObservationTags(
                    observation,
                    prefillGraphPhaseName(cache_phase));
                tags.emplace("launch_kind", launch_kind ? launch_kind : "unknown");
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    "prefill_graph_launch",
                    ns,
                    "prefill",
                    input.device.toString(),
                    std::move(tags));
            }

            if (gpu_event_timing_active && ok)
            {
                gpu_ctx->recordEvent(gpu_stop_event, stream);
                gpu_ctx->synchronizeEvent(gpu_stop_event);
                const float elapsed_ms = gpu_ctx->eventElapsedTime(gpu_start_event, gpu_stop_event);
                if (elapsed_ms >= 0.0f)
                {
                    auto observation = makePrefillGraphObservation(
                        input,
                        key,
                        capture_phase,
                        recapture_reason);
                    auto tags = prefillGraphObservationTags(
                        observation,
                        prefillGraphPhaseName(cache_phase));
                    tags.emplace("attribution", "gpu_event");
                    tags.emplace("source", "prefill_graph_cache");
                    tags.emplace("graph_capture_scope", "prefill_graph_replay");
                    tags.emplace("timing_scope", "total_replay_gpu_event");
                    tags.emplace("sync_scope", "profiling_event_synchronized");
                    tags.emplace("launch_kind", launch_kind ? launch_kind : "unknown");
                    PerfStatsCollector::recordTimingNs(
                        "stage_gpu",
                        "prefill_graph.replay",
                        static_cast<uint64_t>(static_cast<double>(elapsed_ms) * 1.0e6),
                        "prefill",
                        input.device.toString(),
                        std::move(tags));
                }
            }
            destroy_gpu_events();

            return ok;
        };

        if (padded_bucket)
        {
            padded_preflight_reason = preflightPrefillGraph(
                cache,
                *forward_cache.graph,
                key,
                forward_cache.collective_nodes,
                input,
                snapshots_active,
                moe_rebalancing_active,
                PrefillGraphPreflightMode::ColdPaddedSupport,
                prefill_collectives_graph_capturable,
                prefill_heterogeneous_segmentation_admitted,
                moe_rebalancing_graph_stable,
                host_prefill_graph_disabled,
                &padded_reject_stage_name,
                &padded_reject_stage_type);
            padded_preflight_checked = true;

            if (padded_preflight_reason != PrefillGraphRejectReason::None)
            {
                LOG_ERROR("[ForwardExecutionEngine] Padded prefill graph rejected by preflight: "
                          << toString(padded_preflight_reason)
                          << (padded_reject_stage_name.empty() ? "" : " stage=")
                          << padded_reject_stage_name
                          << (padded_reject_stage_type.empty() ? "" : " type=")
                          << padded_reject_stage_type
                          << " real_seq_len=" << real_seq_len
                          << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }
        }

        auto gpuContextForPrefill = [&]() -> IWorkerGPUContext *
        {
            if (!ctx || !ctx->deviceId().is_gpu())
                return nullptr;
            return host.getWorkerGPUContext(ctx->deviceId());
        };

        auto bindPrefillStreamToStages = [&](void *stream)
        {
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage)
                    node->stage->setGPUStream(stream);
            }
        };

        auto preparePrefillGraphLaunchMetadata = [&](void *stream,
                                                     GraphLaunchPreparationPhase phase,
                                                     const char *phase_name) -> bool
        {
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (!node || !node->stage ||
                    !requiresGraphLaunchPreparation(
                        node->stage->graphLaunchPreparationPolicy(), phase))
                    continue;

                /*
                 * Monolithic prefill graphs have the same mutable-metadata
                 * contract as segmented decode graphs: tiny device-side
                 * scalars such as row-select indices must be uploaded on the
                 * explicit capture/replay stream before the graph is recorded
                 * or launched. The stage hook owns that upload; recording an
                 * H2D inside capture is a correctness bug because future graph
                 * replays would bake the first request's metadata.
                 */
                if (!node->stage->prepareGraphLaunch(ctx, stream))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Prefill graph "
                              << (phase_name ? phase_name : "launch")
                              << " metadata preparation failed for stage '"
                              << node_name << "'");
                    return false;
                }
            }
            return true;
        };

        auto publishPrefillCapturedTerminalState = [&](void *stream,
                                                       const char *context) -> bool
        {
            const int terminal_row = real_seq_len - 1;
            if (!executor_.publishCapturedTerminalStateAfterGraphExecution(
                    *forward_cache.graph,
                    terminal_row,
                    stream,
                    context,
                    input.sequence_lengths_device,
                    input.batch_size,
                    input.seq_len,
                    input.sequence_lengths))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph terminal state publication failed"
                          << (context ? std::string(" during ") + context : std::string{})
                          << " real_seq_len=" << real_seq_len
                          << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }
            return true;
        };

        auto ensurePrefillCaptureStream = [&]() -> std::pair<IWorkerGPUContext *, void *>
        {
            IWorkerGPUContext *gpu_ctx = gpuContextForPrefill();
            if (!gpu_ctx)
                return {nullptr, nullptr};
            if (!forward_cache.prefill_capture_stream.ensure(gpu_ctx))
                return {gpu_ctx, nullptr};
            return {gpu_ctx, forward_cache.prefill_capture_stream.stream};
        };

        if (prefill_heterogeneous_segmentation_admitted)
        {
            /*
             * A heterogeneous prefill graph is never routed through the
             * monolithic PrefillGraphCache. Its explicit CPU/cross-device
             * boundaries are stable members of the replay topology, so the
             * generic segment cache captures every GPU region and executes each
             * declared manual boundary in the same order on transaction zero
             * and every replay. Only exact row geometry is admitted here.
             */
            std::string reject_stage_name;
            std::string reject_stage_type;
            const auto segmented_preflight = preflightPrefillGraph(
                cache,
                *forward_cache.graph,
                key,
                forward_cache.collective_nodes,
                input,
                snapshots_active,
                moe_rebalancing_active,
                PrefillGraphPreflightMode::Default,
                prefill_collectives_graph_capturable,
                /*heterogeneous_segmentation_admitted=*/true,
                moe_rebalancing_graph_stable,
                host_prefill_graph_disabled,
                &reject_stage_name,
                &reject_stage_type);
            if (segmented_preflight != PrefillGraphRejectReason::None)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Heterogeneous segmented prefill "
                    "rejected by typed preflight: "
                    << toString(segmented_preflight)
                    << (reject_stage_name.empty() ? "" : " stage=")
                    << reject_stage_name
                    << (reject_stage_type.empty() ? "" : " type=")
                    << reject_stage_type
                    << " real_seq_len=" << real_seq_len
                    << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }

            IWorkerGPUContext *gpu_ctx = gpuContextForPrefill();
            if (!gpu_ctx ||
                !forward_cache.segment_cache.ensureCaptureStream(
                    gpu_ctx,
                    ctx->deviceId()) ||
                !forward_cache.segment_cache.ensureCaptureInputEvent(gpu_ctx) ||
                !forward_cache.segment_cache.ensureCaptureOutputEvent(gpu_ctx) ||
                !forward_cache.segment_cache.capture_stream)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Heterogeneous segmented prefill "
                    "could not establish its explicit capture stream");
                return false;
            }

            void *const stream = forward_cache.segment_cache.capture_stream;
            bindPrefillStreamToStages(stream);
            forward_cache.segment_cache.perf_context =
                input.bucket_seq_len > 0 ? "prefill_bucket" : "prefill";
            const bool was_initialized =
                forward_cache.segment_cache.initialized &&
                !forward_cache.segment_cache.needs_capture;

            if (materialize_without_launch && !was_initialized &&
                !host.prepareGraphBuildStateForMaterialization(
                    input,
                    stream,
                    ctx->deviceId()))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Setup segmented prefill "
                    "materialization could not consume graph-build device-state "
                    "readiness on "
                    << input.device.toString());
                return false;
            }

            auto capture_boundary =
                [&host, &input](
                    const std::string &boundary_name,
                    void *capture_stream) -> bool
            {
                return host.waitAtPrefillGraphCaptureBoundary(
                    input,
                    input.device,
                    boundary_name,
                    capture_stream);
            };

            const bool replay_ok = executor_.executeWithCachedGraphReplay(
                *forward_cache.graph,
                ctx,
                forward_cache.segment_cache,
                stream,
                gpu_ctx,
                &forward_cache.collective_nodes,
                prefill_capture_policy.collectives_graph_capturable,
                /*force_recapture=*/false,
                prefill_capture_policy.defer_final_sync,
                std::move(capture_boundary),
                prefill_capture_policy.graph_replay_plan_policy,
                prefill_capture_policy.launch_dependency,
                std::span<const BufferId>{},
                prefill_capture_policy.retained_parent_composer,
                initial_submission,
                prefill_capture_policy.auxiliary_branch_factory);
            if (!replay_ok)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Mandatory heterogeneous segmented "
                    "prefill capture/replay failed");
                return false;
            }

            if (materialize_without_launch)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "prefill_graph_materialized_without_launch",
                    1.0,
                    "setup",
                    input.device.toString(),
                    {{"bucket_seq_len", std::to_string(bucket_seq_len)},
                     {"replay_plan",
                      uses_device_timeline_transaction
                          ? "device_timeline_full_graph"
                          : uses_retained_sparse_parent
                          ? "retained_sparse_parent"
                          : "heterogeneous_segments"}});
                publishPrefillGraphObservation(
                    forward_cache,
                    input,
                    key,
                    PrefillGraphPhase::Ready,
                    "materialized_without_launch",
                    uses_retained_sparse_parent
                        ? "retained_sparse_parent"
                        : "heterogeneous_boundary_segmentation");
                return true;
            }

            if (!publishPrefillCapturedTerminalState(
                    stream,
                    was_initialized
                        ? "heterogeneous_prefill_segmented_replay"
                        : "heterogeneous_prefill_segmented_capture"))
            {
                return false;
            }
            if (!executor_.publishSnapshotsAfterGraphExecution(
                    *forward_cache.graph,
                    stream,
                    was_initialized
                        ? "heterogeneous_prefill_segmented_replay"
                        : "heterogeneous_prefill_segmented_capture",
                    &forward_cache.segment_cache.snapshot_manifest,
                    snapshot_logical_rows_ptr))
            {
                return false;
            }

            if (used_graph_replay)
                *used_graph_replay = true;
            if (out_producer_stream)
                *out_producer_stream = stream;
            PerfStatsCollector::addCounter(
                "forward_graph",
                "prefill_heterogeneous_segmented_graph",
                1.0,
                "prefill",
                input.device.toString(),
                {{"transaction", was_initialized ? "replay" : "capture"},
                 {"real_seq_len", std::to_string(real_seq_len)},
                     {"bucket_seq_len", std::to_string(bucket_seq_len)},
                     {"replay_plan",
                      uses_retained_sparse_parent
                          ? "retained_sparse_parent"
                          : "heterogeneous_segments"}});
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                was_initialized ? "replay" : "capture",
                uses_retained_sparse_parent
                    ? "retained_sparse_parent"
                    : "heterogeneous_boundary_segmentation");
            return true;
        }

        /*
         * Setup owns a direct Cold -> Initialized transition. It has already
         * built the exact graph topology and permanent storage family, so an
         * eager synthetic request would add payload side effects without
         * establishing any additional capture invariant.
         */
        if (materialize_without_launch && phase == PrefillGraphPhase::Cold)
        {
            cache.markInitializedForSetupMaterialization(key);
            phase = PrefillGraphPhase::Initialized;
        }

        if (materialize_without_launch &&
            phase == PrefillGraphPhase::Ready && cache.hasGraph(key))
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "setup_materialized_graph_reuses",
                1.0,
                "setup",
                input.device.toString(),
                {{"context", "monolithic_prefill"},
                 {"bucket_seq_len", std::to_string(bucket_seq_len)},
                 {"transaction_zero_pending",
                  cache.materializedTransactionZeroPending(key) ? "true"
                                                                : "false"}});
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                "materialized_without_launch",
                "monolithic_prefill_reuse");
            return true;
        }

        if (phase == PrefillGraphPhase::Ready && !host_prefill_graph_disabled)
        {
            // === REPLAY PATH ===
            // Dynamic params already updated by executeCacheHit caller.
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (!gpu_ctx || !stream)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph replay missing explicit capture stream");
                return false;
            }
            bindPrefillStreamToStages(stream);

            const bool transaction_zero_after_setup =
                cache.materializedTransactionZeroPending(key);
            if (transaction_zero_after_setup &&
                !executor_.prepareGraphStorageForCapture(
                    *forward_cache.graph,
                    ctx,
                    stream,
                    "prefill_graph_transaction_zero_frontier",
                    GraphCaptureDependencyLedger::ExternalInputAuthority::
                        RequireReadyBytes))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Setup-materialized prefill graph "
                    "could not validate its live transaction-zero input frontier");
                return false;
            }

            if (!preparePrefillGraphLaunchMetadata(
                    stream,
                    GraphLaunchPreparationPhase::Replay,
                    "replay"))
                return false;

            const char *const launch_kind =
                transaction_zero_after_setup
                    ? "transaction_zero_after_setup"
                    : "replay";
            const std::string replay_reason =
                transaction_zero_after_setup
                    ? "setup_materialized_transaction_zero"
                    : "none";
            if (!launchPrefillGraph(
                    gpu_ctx,
                    stream,
                    "replay",
                    PrefillGraphPhase::Ready,
                    launch_kind,
                    replay_reason))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph replay FAILED for seq_len=" << input.seq_len);
                return false;
            }

            if (!publishPrefillCapturedTerminalState(
                    stream,
                    "prefill_graph_replay"))
            {
                return false;
            }

            if (!executor_.publishSnapshotsAfterGraphExecution(
                    *forward_cache.graph,
                    stream,
                    "prefill_graph_replay",
                    &forward_cache.snapshot_manifest,
                    snapshot_logical_rows_ptr))
            {
                return false;
            }

            if (cache.config().trace)
                LOG_TRACE("[ForwardExecutionEngine] Prefill graph REPLAY seq_len=" << input.seq_len
                                                                                    << " replay_count=" << cache.replayCount(key));
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                "replay",
                replay_reason);
            if (used_graph_replay)
                *used_graph_replay = true;
            if (out_producer_stream)
                *out_producer_stream = stream;
            return true;
        }

        const bool can_attempt_capture =
            phase == PrefillGraphPhase::Warmup ||
            phase == PrefillGraphPhase::Initialized;
        PrefillGraphRejectReason capture_ready_reason = PrefillGraphRejectReason::None;
        std::string capture_ready_reject_stage_name;
        std::string capture_ready_reject_stage_type;
        if (can_attempt_capture)
        {
            capture_ready_reason = preflightPrefillGraph(
                cache,
                *forward_cache.graph,
                key,
                forward_cache.collective_nodes,
                input,
                snapshots_active,
                moe_rebalancing_active,
                materialize_without_launch
                    ? PrefillGraphPreflightMode::Default
                    : PrefillGraphPreflightMode::CaptureReady,
                prefill_collectives_graph_capturable,
                prefill_heterogeneous_segmentation_admitted,
                moe_rebalancing_graph_stable,
                host_prefill_graph_disabled,
                &capture_ready_reject_stage_name,
                &capture_ready_reject_stage_type);
        }

        if (can_attempt_capture && capture_ready_reason == PrefillGraphRejectReason::None)
        {
            // === CAPTURE PATH ===
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (!gpu_ctx || !stream)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture missing explicit capture stream");
                return false;
            }

            /*
             * Apply the dedicated prefill stream to every stage before capture.
             * The warmup, metadata preparation, snapshot preparation, and
             * captured launch all use this same explicit stream. CUDA/HIP
             * stream order therefore supplies the complete producer-to-capture
             * dependency; a host stream drain here would only serialize the
             * Warmup -> Capture transition.
            */
            bindPrefillStreamToStages(stream);
            if (materialize_without_launch &&
                !host.prepareGraphBuildStateForMaterialization(
                    input,
                    stream,
                    ctx->deviceId()))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Setup monolithic prefill "
                    "materialization could not consume graph-build device-state "
                    "readiness on "
                    << input.device.toString());
                return false;
            }
            if (!executor_.allocateGraphStorageForCapture(
                    *forward_cache.graph,
                    ctx,
                    "prefill_graph_capture_prebind"))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph storage prebind failed for seq_len="
                          << input.seq_len);
                return false;
            }
            if (!preparePrefillGraphLaunchMetadata(
                    stream,
                    GraphLaunchPreparationPhase::Capture,
                    "capture"))
                return false;

            if (materialize_without_launch)
            {
                capture_ready_reason = preflightPrefillGraph(
                    cache,
                    *forward_cache.graph,
                    key,
                    forward_cache.collective_nodes,
                    input,
                    snapshots_active,
                    moe_rebalancing_active,
                    PrefillGraphPreflightMode::CaptureReady,
                    prefill_collectives_graph_capturable,
                    prefill_heterogeneous_segmentation_admitted,
                    moe_rebalancing_graph_stable,
                    host_prefill_graph_disabled,
                    &capture_ready_reject_stage_name,
                    &capture_ready_reject_stage_type);
                if (capture_ready_reason != PrefillGraphRejectReason::None)
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] Setup monolithic prefill "
                        "capture readiness rejected after launch preparation: "
                        << toString(capture_ready_reason)
                        << (capture_ready_reject_stage_name.empty()
                                ? ""
                                : " stage=")
                        << capture_ready_reject_stage_name
                        << (capture_ready_reject_stage_type.empty()
                                ? ""
                                : " type=")
                        << capture_ready_reject_stage_type);
                    return false;
                }
            }
            if (!executor_.prepareSnapshotsForGraphCapture(
                    *forward_cache.graph,
                    ctx,
                    stream,
                    "prefill_graph_capture",
                    &forward_cache.snapshot_manifest))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph snapshot preparation failed for seq_len="
                          << input.seq_len);
                return false;
            }

            /*
             * Warmup may publish arena inputs from eager streams other than this
             * graph's dedicated stream. Join every such producer here, while
             * external event waits are legal, so the capture body contains only
             * graph-owned ordering. This is a nonblocking device-side fence; it
             * performs no stream or device synchronization.
             */
            if (!executor_.prepareGraphStorageForCapture(
                    *forward_cache.graph,
                    ctx,
                    stream,
                    "prefill_graph_capture",
                    materialize_without_launch
                        ? GraphCaptureDependencyLedger::ExternalInputAuthority::
                              BindDeclaredAddressesOnly
                        : GraphCaptureDependencyLedger::ExternalInputAuthority::
                              RequireReadyBytes))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph storage preparation failed for seq_len="
                          << input.seq_len);
                return false;
            }

            const auto &prefill_capture_order =
                forward_cache.graph->getExecutionOrder();
            auto dependency_ledger = executor_.planGraphCaptureDependencies(
                *forward_cache.graph,
                std::span<const std::string>(
                    prefill_capture_order.data(), prefill_capture_order.size()),
                input.device,
                stream,
                "prefill_graph_capture",
                std::span<const BufferId>{},
                materialize_without_launch
                    ? GraphCaptureDependencyLedger::ExternalInputAuthority::
                          BindDeclaredAddressesOnly
                    : GraphCaptureDependencyLedger::ExternalInputAuthority::
                          RequireReadyBytes);
            if (!dependency_ledger)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Prefill graph dependency planning "
                    "failed for seq_len="
                    << input.seq_len);
                return false;
            }

            const std::string capture_begin_boundary =
                "before_begin:seq=" + std::to_string(input.seq_len) +
                ":real=" + std::to_string(real_seq_len) +
                ":bucket=" + std::to_string(bucket_seq_len) +
                ":offset=" + std::to_string(input.position_offset);
            if (!host.waitAtPrefillGraphCaptureBoundary(
                    input,
                    input.device,
                    capture_begin_boundary,
                    stream))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture begin rendezvous failed for seq_len="
                          << input.seq_len << " on " << input.device.toString());
                return false;
            }

            /*
             * The cache owns begin/body/end as one scoped transaction. The
             * callback cannot strand the native stream in capture mode, and a
             * failed graph body cannot enter eager recovery.
             */
            if (!cache.captureAndInstantiate(
                    key,
                    gpu_ctx,
                    stream,
                    [&]() -> bool
                    {
                        if (materialize_without_launch)
                        {
                            return executor_.recordSetupGraphMaterialization(
                                *forward_cache.graph,
                                ctx,
                                &forward_cache.collective_nodes,
                                &forward_cache.snapshot_manifest);
                        }
                        return executor_.executeFastDecode(
                            *forward_cache.graph,
                            ctx,
                            &forward_cache.collective_nodes,
                            &forward_cache.snapshot_manifest);
                    },
                    dependency_ledger.get()))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Mandatory prefill graph capture transaction failed for seq_len="
                    << input.seq_len);
                return false;
            }

            if (materialize_without_launch)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "prefill_graph_materialized_without_launch",
                    1.0,
                    "setup",
                    input.device.toString(),
                    {{"bucket_seq_len", std::to_string(bucket_seq_len)},
                     {"replay_plan", "monolithic_prefill"},
                     {"transaction_zero_pending", "true"}});
                publishPrefillGraphObservation(
                    forward_cache,
                    input,
                    key,
                    PrefillGraphPhase::Ready,
                    "materialized_without_launch",
                    "monolithic_prefill");
                return true;
            }

            const std::string capture_launch_boundary =
                "before_launch_after_capture:seq=" + std::to_string(input.seq_len) +
                ":real=" + std::to_string(real_seq_len) +
                ":bucket=" + std::to_string(bucket_seq_len) +
                ":offset=" + std::to_string(input.position_offset);
            if (!host.waitAtPrefillGraphCaptureBoundary(
                    input,
                    input.device,
                    capture_launch_boundary,
                    stream))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture launch rendezvous failed for seq_len="
                          << input.seq_len << " on " << input.device.toString());
                return false;
            }

            // Kernels recorded during HIP/CUDA stream capture are not executed
            // until the executable graph is launched. Launch once immediately so
            // the capture request produces logits and advances device state.
            const std::string recapture_reason =
                phase == PrefillGraphPhase::Initialized
                    ? "lazy_initialized_after_request_reset"
                    : "armed_warmup";
            if (!launchPrefillGraph(gpu_ctx, stream, "capture", PrefillGraphPhase::Ready, "launch_after_capture",
                                    recapture_reason))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph launch-after-capture failed for seq_len=" << input.seq_len);
                return false;
            }

            if (!publishPrefillCapturedTerminalState(
                    stream,
                    "prefill_graph_capture_launch"))
            {
                return false;
            }

            if (!executor_.publishSnapshotsAfterGraphExecution(
                    *forward_cache.graph,
                    stream,
                    "prefill_graph_capture_launch",
                    &forward_cache.snapshot_manifest,
                    snapshot_logical_rows_ptr))
            {
                return false;
            }

            LOG_INFO("[ForwardExecutionEngine] Prefill graph CAPTURED seq_len=" << input.seq_len
                                                                                << " nodes=" << cache.nodeCount(key));
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                "capture",
                recapture_reason);
            if (out_producer_stream)
                *out_producer_stream = stream;
            return true;
        }

        if (can_attempt_capture && capture_ready_reason != PrefillGraphRejectReason::None)
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture readiness rejected after "
                      << prefillGraphPhaseName(phase)
                      << ": " << toString(capture_ready_reason)
                      << (capture_ready_reject_stage_name.empty() ? "" : " stage=")
                      << capture_ready_reject_stage_name
                      << (capture_ready_reject_stage_type.empty() ? "" : " type=")
                      << capture_ready_reject_stage_type
                      << " seq_len=" << input.seq_len
                      << "; refusing eager prefill fallback");
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                phase,
                "rejected",
                toString(capture_ready_reason),
                capture_ready_reject_stage_name,
                capture_ready_reject_stage_type);
            return false;
        }

        // === WARMUP/COLD PATH ===
        bool cold_capture_candidate = false;
        PrefillGraphRejectReason cold_reject_reason = PrefillGraphRejectReason::None;
        std::string cold_reject_stage_name;
        std::string cold_reject_stage_type;
        const bool cold_phase =
            phase == PrefillGraphPhase::Cold ||
            phase == PrefillGraphPhase::Initialized ||
            phase == PrefillGraphPhase::Warmup;
        if (cold_phase)
        {
            if (padded_preflight_checked)
            {
                cold_reject_reason = padded_preflight_reason;
                cold_reject_stage_name = padded_reject_stage_name;
                cold_reject_stage_type = padded_reject_stage_type;
            }
            else
            {
                cold_reject_reason = preflightPrefillGraph(
                    cache,
                    *forward_cache.graph,
                    key,
                    forward_cache.collective_nodes,
                    input,
                    snapshots_active,
                    moe_rebalancing_active,
                    padded_bucket
                        ? PrefillGraphPreflightMode::ColdPaddedSupport
                        : PrefillGraphPreflightMode::Default,
                    prefill_collectives_graph_capturable,
                    prefill_heterogeneous_segmentation_admitted,
                    moe_rebalancing_graph_stable,
                    host_prefill_graph_disabled,
                    &cold_reject_stage_name,
                    &cold_reject_stage_type);
            }
            cold_capture_candidate = (cold_reject_reason == PrefillGraphRejectReason::None);
        }

        bool cold_stream_ready = false;
        if (cold_capture_candidate)
        {
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (gpu_ctx && stream)
            {
                // Warm up lazy allocations on the same explicit stream that the
                // next request will capture on. This mirrors decode segmented
                // capture and avoids capture-unsafe first-use work.
                bindPrefillStreamToStages(stream);
                cold_stream_ready = true;
            }
            else
            {
                cold_reject_reason = PrefillGraphRejectReason::NoGPUContext;
                cold_capture_candidate = false;
            }
        }

        if (cold_phase && (!cold_capture_candidate || !cold_stream_ready))
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture rejected: "
                      << toString(cold_reject_reason)
                      << (cold_reject_stage_name.empty() ? "" : " stage=")
                      << cold_reject_stage_name
                      << (cold_reject_stage_type.empty() ? "" : " type=")
                      << cold_reject_stage_type
                      << " seq_len=" << input.seq_len
                      << "; refusing eager prefill fallback");
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Cold,
                "rejected",
                toString(cold_reject_reason),
                cold_reject_stage_name,
                cold_reject_stage_type);
            return false;
        }

        // Execute normally to warm up lazy allocations.
        bool exec_success = executor_.executeFastDecode(
            *forward_cache.graph,
            ctx,
            &forward_cache.collective_nodes,
            &forward_cache.snapshot_manifest);

        if (!exec_success)
            return false;

        if (!publishPrefillCapturedTerminalState(
                nullptr,
                "prefill_graph_warmup"))
        {
            return false;
        }

        if (!executor_.publishSnapshotsAfterGraphExecution(
                *forward_cache.graph,
                nullptr,
                "prefill_graph_warmup",
                &forward_cache.snapshot_manifest,
                snapshot_logical_rows_ptr))
        {
            return false;
        }

        if (out_producer_stream)
            *out_producer_stream =
                forward_cache.prefill_capture_stream.stream;

        // After successful warmup, check if graph capture is eligible
        if (cold_phase)
        {
            cache.markWarmedUp(key);
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Warmup,
                "warmup",
                "none");
            if (cache.config().trace)
                LOG_INFO("[ForwardExecutionEngine] Prefill graph ARMED for capture: seq_len=" << input.seq_len);
        }

        return true;
    }

    // =========================================================================
    // executeCacheMiss() — Build and Execute New Forward Graph
    // =========================================================================

    bool ForwardExecutionEngine::executeCacheMiss(
        const ForwardInput &input_in,
        ForwardOutput &output,
        const ForwardGraphSignature &signature,
        ForwardGraphCache *build_cache,
        bool should_cache,
        IForwardExecutionHost &host,
        bool is_decode,
        bool has_unified_pp,
        std::chrono::high_resolution_clock::time_point start)
    {
        // ===== CACHE MISS: Build new graph =====

        // Unified PP path currently executes multi-device graphs and does not use
        // this forward cache; clear entries to avoid stale memory growth.
        if (has_unified_pp && !cache_.empty())
        {
            invalidateAll();
            cache_.clear();
            LOG_DEBUG("[ForwardExecutionEngine] Cleared forward graph cache for unified PP execution path");
        }

        // For cache misses on standard path: redirect token_ids and
        // position_ids to stable buffers so that cached stages' pointers survive.
        ForwardInput effective_input = input_in;

        if (should_cache && build_cache)
        {
            const int total_tokens = effective_input.batch_size * effective_input.seq_len;
            if (effective_input.token_ids)
            {
                build_cache->token_ids.assign(
                    effective_input.token_ids,
                    effective_input.token_ids + total_tokens);
                effective_input.token_ids = build_cache->token_ids.data();
            }
            if (effective_input.position_ids)
            {
                build_cache->position_ids.assign(
                    effective_input.position_ids,
                    effective_input.position_ids + total_tokens);
                effective_input.position_ids = build_cache->position_ids.data();
            }
        }

        /*
         * A non-embedding pipeline stage consumes an activation produced by the
         * preceding stage. Perform that handoff before graph construction so
         * model graph builders remain declarative and never materialize GPU
         * activations through data()/mutable_data(). TransferEngine selects D2D,
         * peer transport, or the explicitly designed heterogeneous host-staged
         * boundary and publishes destination authority as one operation.
         */
        const auto initial_pp_copy = host.resolvePPCopyInfo(effective_input);
        if (initial_pp_copy.needs_copy)
        {
            auto pp_result = TransferEngine::instance().copyActivation(
                initial_pp_copy.external_hidden,
                initial_pp_copy.working_buffer,
                initial_pp_copy.device,
                initial_pp_copy.copy_bytes);
            if (!pp_result.success)
            {
                throw std::runtime_error(
                    "Initial PP hidden-state copy failed: " + pp_result.error);
            }
        }

        // Build forward graph via host callback
        GraphBuildResult build_result = host.buildForwardGraph(effective_input);

        if (!build_result)
        {
            LOG_ERROR("[ForwardExecutionEngine] Graph build failed: " << build_result.error());
            return false;
        }

        output = build_result.output();
        ComputeGraph graph = build_result.takeGraph();

        if (effective_input.device_prefill_chunk)
        {
            const DevicePrefillChunkGraphBinding &binding =
                *effective_input.device_prefill_chunk;
            const bool binding_matches_forward_input =
                binding.valid() && effective_input.device.is_gpu() &&
                effective_input.batch_size == 1 &&
                effective_input.seq_len == binding.bucket_seq_len &&
                effective_input.token_ids == nullptr &&
                effective_input.token_ids_device ==
                    binding.chunk_token_ids_device &&
                effective_input.position_ids == nullptr &&
                effective_input.position_ids_device ==
                    binding.chunk_position_ids_device &&
                effective_input.position_policy ==
                    ForwardPositionPolicy::ExplicitRows &&
                effective_input.sequence_lengths_device ==
                    binding.chunk_real_rows_device;
            if (!binding_matches_forward_input)
            {
                LOG_ERROR("[ForwardExecutionEngine] Device prefill chunk binding does not exactly own the forward graph inputs");
                return false;
            }

            const std::vector<std::string> model_roots = graph.getRootNodes();
            if (model_roots.empty())
            {
                LOG_ERROR("[ForwardExecutionEngine] Cannot prepend a device chunk materializer to an empty model graph");
                return false;
            }

            constexpr const char *kChunkMaterializationNode =
                "prefill_chunk_materialization";
            graph.addNode(
                kChunkMaterializationNode,
                ComputeStageFactory::createPrefillChunkMaterialization({
                    .device_id = effective_input.device,
                    .backend = binding.backend,
                    .request_token_ids_device =
                        binding.request_token_ids_device,
                    .request_position_ids_device =
                        binding.request_position_ids_device,
                    .request_total_rows_device =
                        binding.request_total_rows_device,
                    .cached_tokens_device = binding.cached_tokens_device,
                    .chunk_token_ids_device = binding.chunk_token_ids_device,
                    .chunk_position_ids_device =
                        binding.chunk_position_ids_device,
                    .chunk_real_rows_device = binding.chunk_real_rows_device,
                    .chunk_row_stride_device =
                        binding.chunk_row_stride_device,
                    .request_row_capacity = binding.request_row_capacity,
                    .bucket_seq_len = binding.bucket_seq_len,
                    .pad_token_id = binding.pad_token_id,
                    .capture_identity = binding.capture_identity,
                    .stage_name = kChunkMaterializationNode,
                }),
                effective_input.device);
            for (const std::string &root : model_roots)
                graph.addDependency(root, kChunkMaterializationNode);
        }
        auto collective_nodes = graph.collectiveNodeNames();

        if (signature.live_mtp_request_batch_condition &&
            debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            std::ostringstream manifest;
            bool first = true;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                if (!collective_nodes.contains(node_name))
                    continue;
                if (!first)
                    manifest << ',';
                first = false;
                manifest << node_name;
            }
            LOG_INFO(
                "[MTPConditionGraphContract] event=graph_manifest"
                << " device=" << effective_input.device.toString()
                << " stages=" << graph.size()
                << " collective_count=" << collective_nodes.size()
                << " collectives=[" << manifest.str() << ']');
        }

        LOG_DEBUG("[ForwardExecutionEngine] Forward graph built with " << graph.size() << " stages");

        if (graph.size() == 0)
        {
            LOG_ERROR("[ForwardExecutionEngine] Empty forward graph");
            return false;
        }

        const bool bucketed_prefill_miss =
            should_cache && build_cache && signature.is_bucketed_prefill && !is_decode;
        bool bucketed_prefill_capture_candidate = false;
        bool bucketed_prefill_heterogeneous_segmented_candidate = false;
        PrefillGraphRejectReason bucketed_prefill_reject_reason = PrefillGraphRejectReason::None;
        std::string bucketed_prefill_reject_stage_name;
        std::string bucketed_prefill_reject_stage_type;

        if (bucketed_prefill_miss)
        {
            PrefillGraphCache preflight_cache(
                makePrefillGraphConfigFromEnv(host.residentGraphRows()));
            PrefillGraphCacheKey key = makePrefillGraphKey(effective_input, host);
            /*
             * Reject graph-intrinsic violations before asking the host for a
             * launch context.  The optimistic policy flags make this first
             * pass ignore only the two capabilities that genuinely depend on
             * that context; recurrent padded stages, invalid geometry, and
             * host/rebalancing policy remain fatal here.  Besides preserving
             * fail-fast behavior, this keeps rejected graphs from touching a
             * backend whose stream/context lifecycle they can never use.
             */
            bucketed_prefill_reject_reason = preflightPrefillGraph(
                preflight_cache,
                graph,
                key,
                collective_nodes,
                effective_input,
                executor_.config().snapshot_callback != nullptr,
                host.isMoeRebalancingActive(),
                PrefillGraphPreflightMode::Default,
                /*collectives_graph_capturable=*/true,
                /*heterogeneous_segmentation_admitted=*/true,
                host.isMoeRebalancingGraphStableForPrefillCapture(),
                host.prefillGraphCaptureDisabledByHost(),
                &bucketed_prefill_reject_stage_name,
                &bucketed_prefill_reject_stage_type);

            if (bucketed_prefill_reject_reason ==
                PrefillGraphRejectReason::None)
            {
                IDeviceContext *prefill_preflight_ctx =
                    host.getDeviceContext(effective_input.device);
                const auto prefill_capture_policy =
                    host.buildDecodeCapturePolicy(
                        !collective_nodes.empty(),
                        prefill_preflight_ctx);
                const bool prefill_collectives_graph_capturable =
                    !collective_nodes.empty() &&
                    prefill_capture_policy.collectives_graph_capturable;
                const bool prefill_heterogeneous_segmentation_admitted =
                    prefill_capture_policy.heterogeneous_segmented_enabled;

                bucketed_prefill_reject_reason = preflightPrefillGraph(
                    preflight_cache,
                    graph,
                    key,
                    collective_nodes,
                    effective_input,
                    executor_.config().snapshot_callback != nullptr,
                    host.isMoeRebalancingActive(),
                    PrefillGraphPreflightMode::Default,
                    prefill_collectives_graph_capturable,
                    prefill_heterogeneous_segmentation_admitted,
                    host.isMoeRebalancingGraphStableForPrefillCapture(),
                    host.prefillGraphCaptureDisabledByHost(),
                    &bucketed_prefill_reject_stage_name,
                    &bucketed_prefill_reject_stage_type);
                bucketed_prefill_heterogeneous_segmented_candidate =
                    bucketed_prefill_reject_reason ==
                        PrefillGraphRejectReason::None &&
                    prefill_heterogeneous_segmentation_admitted;
            }
            bucketed_prefill_capture_candidate =
                (bucketed_prefill_reject_reason == PrefillGraphRejectReason::None);

            if (!bucketed_prefill_capture_candidate)
            {
                if (build_cache && !build_cache->prefill_graph_cache)
                {
                    build_cache->prefill_graph_cache =
                        std::make_unique<PrefillGraphCache>(
                            makePrefillGraphConfigFromEnv(
                                host.residentGraphRows()));
                }
                if (build_cache && build_cache->prefill_graph_cache)
                {
                    publishPrefillGraphObservation(
                        *build_cache,
                        effective_input,
                        key,
                        PrefillGraphPhase::Cold,
                        "rejected",
                        toString(bucketed_prefill_reject_reason),
                        bucketed_prefill_reject_stage_name,
                        bucketed_prefill_reject_stage_type);
                }

                std::string rejected_stage_readiness;
                if (!bucketed_prefill_reject_stage_name.empty())
                {
                    const auto *rejected_node =
                        graph.getNode(bucketed_prefill_reject_stage_name);
                    if (rejected_node && rejected_node->stage)
                    {
                        rejected_stage_readiness =
                            rejected_node->stage
                                ->graphCaptureReadinessDebugString();
                    }
                }

                LOG_ERROR("[ForwardExecutionEngine] Bucketed prefill graph rejected by preflight before execution: "
                          << toString(bucketed_prefill_reject_reason)
                          << (bucketed_prefill_reject_stage_name.empty() ? "" : " stage=")
                          << bucketed_prefill_reject_stage_name
                          << (bucketed_prefill_reject_stage_type.empty() ? "" : " type=")
                          << bucketed_prefill_reject_stage_type
                          << " real_seq_len=" << effectiveRealSeqLen(effective_input)
                          << " bucket_seq_len=" << effectiveBucketSeqLen(effective_input)
                          << (rejected_stage_readiness.empty()
                                  ? ""
                                  : " readiness={")
                          << rejected_stage_readiness
                          << (rejected_stage_readiness.empty() ? "" : "}")
                          << "; refusing eager prefill fallback");
                return false;
            }
        }

        // Cache hits already refresh this small stage list before replay.
        // Cache misses must do the same before the first graph execution; otherwise
        // padded bucket stages can append/select the bucket tail during warmup.
        std::vector<IComputeStage *> prefill_replay_param_stages;
        if (!is_decode)
        {
            prefill_replay_param_stages = collectPrefillReplayParamStages(graph);
            updatePrefillReplayParamStages(effective_input, prefill_replay_param_stages);
        }

        // Sparse manual boundaries must receive the same root-published
        // identity before the very first graph execution and all later cache
        // replays. A local stage-object execution count is not stable across
        // independent rank graph construction or capture lifecycle changes.
        std::vector<IComputeStage *> moe_overlay_collective_runtime_stages =
            collectMoEOverlayCollectiveRuntimeStages(graph);
        updateMoEOverlayCollectiveRuntimeStages(
            effective_input,
            moe_overlay_collective_runtime_stages);

        // Ensure declared CPU/GPU workspace is allocated for this graph. Cache
        // hits repeat this step because another bucket can grow the shared
        // per-device workspace and leave cached stages bound to old pointers.
        const WorkspaceGraphParticipantRole workspace_role =
            workspaceParticipantRole(
                effective_input.execution_role,
                is_decode);
        if (!host.ensureDeviceWorkspaceAllocated(
                graph,
                effective_input.seq_len,
                workspaceFamilyPolicy(workspace_role),
                workspace_role))
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to allocate workspace for forward graph on "
                      << effective_input.device.toString() << " seq_len=" << effective_input.seq_len);
            return false;
        }
        const uint64_t workspace_generation = host.workspaceGeneration(effective_input.device);

        // Notify host that graph is ready — allows releasing transient resources
        // (e.g., mmap pages) before execution allocates large activation buffers.
        if (!first_graph_ready_fired_)
        {
            first_graph_ready_fired_ = true;
            host.onFirstGraphReady();
        }

        /*
         * A cacheable GPU decode graph and an exact heterogeneous segmented
         * prefill graph each have one lifecycle, including their first
         * invocation. Install the fully built graph into its stable cache
         * before submitting any kernels, then enter executeCacheHit(), whose
         * capture transaction executes the logical payload once and returns
         * with complete native graph executable units.
         *
         * Keeping a separate eager cache-miss execution here used to expose a
         * valid ComputeGraph with no replay-ready GraphSegmentCache. The MTP
         * parent graph could observe that half-state immediately after the
         * grouped verifier produced its rows. Delegating first use through the
         * ordinary cached path makes that state structurally impossible and
         * also guarantees that cache misses and hits share identical stream,
         * dynamic-parameter, collective, snapshot, and publication ordering.
         */
        const bool atomic_gpu_graph_first_use =
            is_decode ||
            effective_input.graph_submission_intent ==
                ForwardGraphSubmissionIntent::
                    MaterializeExecutableWithoutLaunch ||
            bucketed_prefill_heterogeneous_segmented_candidate;
        if (should_cache && build_cache && atomic_gpu_graph_first_use &&
            effective_input.device.is_gpu())
        {
            build_cache->graph =
                std::make_unique<ComputeGraph>(std::move(graph));
            build_cache->output = output;
            build_cache->workspace_generation = workspace_generation;
            build_cache->snapshot_configuration_epoch =
                executor_.snapshotConfigurationEpoch();
            build_cache->collective_nodes = std::move(collective_nodes);
            build_cache->prefill_replay_param_stages =
                std::move(prefill_replay_param_stages);
            build_cache->prefill_replay_param_stages_cached = !is_decode;
            build_cache->moe_overlay_collective_runtime_stages =
                std::move(moe_overlay_collective_runtime_stages);
            build_cache->moe_overlay_collective_runtime_stages_cached = true;
            build_cache->valid = true;

            /*
             * The initial PP handoff already ran before graph construction.
             * Publish its immutable cache bindings now, but arm repetition only
             * after first-use execution so that invocation does not copy the
             * same activation twice.
             */
            if (initial_pp_copy.needs_copy)
            {
                build_cache->pp_external_hidden_state =
                    initial_pp_copy.external_hidden;
                build_cache->pp_working_buffer =
                    initial_pp_copy.working_buffer;
                build_cache->pp_copy_bytes = initial_pp_copy.copy_bytes;
                build_cache->pp_device = initial_pp_copy.device;
            }

            const bool first_use_success = executeCacheHit(
                effective_input,
                output,
                *build_cache,
                host,
                is_decode,
                start);
            if (!first_use_success)
            {
                build_cache->invalidate();
                return false;
            }

            build_cache->pp_needs_copy = initial_pp_copy.needs_copy;
            LOG_DEBUG(
                "[ForwardExecutionEngine] Atomically materialized first-use "
                "GPU graph [kind="
                << (is_decode
                        ? "decode"
                        : effective_input.graph_submission_intent ==
                                  ForwardGraphSubmissionIntent::
                                      MaterializeExecutableWithoutLaunch
                              ? "setup_prefill"
                              : "heterogeneous_prefill")
                << ", seq_len="
                << signature.seq_len
                << ", batch_size=" << signature.batch_size
                << ", device=" << signature.device.to_string()
                << ", all_position_logits="
                << signature.all_position_logits << "]");
            return true;
        }

        if (effective_input.graph_submission_intent ==
            ForwardGraphSubmissionIntent::
                MaterializeExecutableWithoutLaunch)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup-owned GPU graph family was not "
                "admitted to the stable forward cache; refusing to execute its "
                "model payload eagerly");
            return false;
        }

        bool success = false;

        /*
         * Retain the concrete producer stream independently of graph caching.
         * Short GPU prefills can be intentionally ineligible for caching, but
         * their hidden-state consumers need the same exact ordering contract as
         * Ready graph replays.
         */
        void *execution_stream_used = nullptr;

        // Execution path depends on configuration:
        // - Unified PP: multi-device execution with all PP stage devices
        // - Single-device: standard single-context execution
        if (has_unified_pp)
        {
            auto contexts = host.getPipelineDeviceContexts();
            if (contexts.empty())
            {
                LOG_ERROR("[ForwardExecutionEngine] No pipeline device contexts available");
                return false;
            }

            LOG_DEBUG("[ForwardExecutionEngine] Executing unified PP graph with "
                      << contexts.size() << " device contexts...");

            success = executor_.executeMultiDevice(graph, contexts);
        }
        else
        {
            LOG_DEBUG("[ForwardExecutionEngine] Getting device context for " << effective_input.device << "...");
            IDeviceContext *ctx = host.getDeviceContext(effective_input.device);
            if (!ctx)
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to get device context");
                return false;
            }
            LOG_DEBUG("[ForwardExecutionEngine] Got device context, starting execution...");

            void *execution_stream = nullptr;
            IWorkerGPUContext *execution_gpu_ctx = nullptr;
            if (ctx->deviceId().is_gpu())
            {
                try
                {
                    execution_gpu_ctx = host.getWorkerGPUContext(ctx->deviceId());
                    if (!execution_gpu_ctx)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for "
                                  << ctx->deviceId().toString());
                        return false;
                    }
                    execution_stream = execution_gpu_ctx->defaultStream();
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Could not resolve forward graph stream for "
                              << ctx->deviceId().toString() << ": " << e.what());
                    return false;
                }

                if (bucketed_prefill_capture_candidate)
                {
                    /*
                     * The first bucketed-prefill cache miss is the warmup pass
                     * for the later monolithic graph capture.  Run it on the
                     * same explicit stream that the Warmup->Capture transition
                     * will record on; otherwise lazy first-use work may be
                     * ordered on one stream while capture later observes another.
                     */
                    if (!build_cache->prefill_capture_stream.ensure(execution_gpu_ctx) ||
                        !build_cache->prefill_capture_stream.stream)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Bucketed prefill cache miss could not create an explicit capture stream for "
                                  << ctx->deviceId().toString());
                        return false;
                    }
                    execution_stream = build_cache->prefill_capture_stream.stream;
                }
                execution_stream_used = execution_stream;
            }

            /*
             * Cache-hit replay runs a stream/dynamic-parameter prelude before
             * launching the cached graph. Cache misses need the same contract
             * for their first execution: graph capture may start inside
             * DeviceGraphExecutor, so stages such as EmbeddingStage must upload
             * token IDs to their workspace before executor-side capture begins.
             *
             * The stream is Llaminar's owned worker stream, never the legacy
             * CUDA/HIP null stream. Stages that do not override
             * updateDynamicParams() simply ignore this setup.
             */
            std::vector<IComputeStage *> cache_miss_dynamic_param_stages;
            if (ctx->deviceId().is_gpu())
            {
                if (!execution_stream)
                {
                    LOG_ERROR("[ForwardExecutionEngine] GPU cache-miss graph execution requires an explicit stream for "
                              << ctx->deviceId().toString());
                    return false;
                }

                cache_miss_dynamic_param_stages =
                    collectDynamicParamStages(graph);
                for (const auto &node_name : graph.getExecutionOrder())
                {
                    ComputeNode *node = graph.getNode(node_name);
                    if (!node || !node->stage)
                        continue;
                    node->stage->setGPUStream(execution_stream);
                }

                if (!is_decode)
                {
                    /*
                     * Padded prefill replay params were pushed once before
                     * workspace allocation so host-only stages can see the
                     * metadata early.  GPU GDN/short-conv stages also upload a
                     * workspace-backed effective-length scalar, so repeat the
                     * update after the workspace has been bound and after every
                     * stage owns this explicit stream.  Otherwise a warmup run
                     * can leave the scalar at the previous real length and the
                     * following capture mutates live GDN state through one or
                     * more padding rows.
                     */
                    updatePrefillReplayParamStages(
                        effective_input,
                        prefill_replay_param_stages);
                }
            }

            if (!host.prepareLiveStateForForwardGraphExecution(
                    effective_input,
                    execution_stream,
                    ctx->deviceId()))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to prepare live state for forward graph execution");
                return false;
            }
            ScopedForwardLiveStateRead live_state_read(
                host,
                effective_input,
                execution_stream,
                ctx->deviceId());

            if (!host.prepareDeviceTokenInputsForForwardGraphExecution(
                    effective_input,
                    execution_stream,
                    ctx->deviceId()))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to prepare forward graph device-token inputs");
                return false;
            }

            if (host.computeAllPositionLogitsEnabled() &&
                host.allPositionLogitRows() > 0)
            {
                if (!host.prepareAllPositionVerifierGraphMetadata(
                        effective_input,
                        execution_stream,
                        ctx->deviceId()))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to prepare all-position verifier graph metadata");
                    return false;
                }
            }

            if (!updateDynamicParamStages(
                    effective_input,
                    cache_miss_dynamic_param_stages,
                    effective_input.position_ids,
                    "cache-miss graph execution"))
            {
                return false;
            }

            if (!host.waitBeforeForwardGraphExecution(
                    effective_input,
                    ctx->deviceId(),
                    /*cache_miss=*/true))
            {
                LOG_ERROR("[ForwardExecutionEngine] Forward graph pre-execution rendezvous failed for "
                          << ctx->deviceId().toString());
                return false;
            }

            success =
                build_cache
                    ? executor_.executeWithSnapshotManifest(
                          graph,
                          ctx,
                          build_cache->snapshot_manifest)
                    : executor_.execute(graph, ctx);

            /*
             * Match the pre-execution live-state admission on every outcome.
             * Even a failed executor may have queued kernels, so skipping this
             * event edge would let the next mailbox writer race partial work.
             */
            live_state_read.complete(execution_stream);
        }

        DeviceId producer_device = effective_input.device;
        bool deferred_all_position_verifier_sync = false;
        bool deferred_main_decode_sync = false;
        if (success)
        {
            /*
             * A unified pipeline graph may finish on a device other than the
             * input token owner. Resolve that output device's worker stream
             * explicitly so provenance still names the stream which executes
             * its final hidden-state stages.
             */
            producer_device =
                resolveForwardOutputProducerDevice(
                    output,
                    effective_input.device);
            if (has_unified_pp && producer_device.is_gpu())
            {
                IWorkerGPUContext *producer_gpu_ctx =
                    host.getWorkerGPUContext(producer_device);
                if (!producer_gpu_ctx || !producer_gpu_ctx->defaultStream())
                {
                    LOG_ERROR("[ForwardExecutionEngine] Unified GPU forward completed without an explicit output producer stream on "
                              << producer_device.toString());
                    return false;
                }
                execution_stream_used = producer_gpu_ctx->defaultStream();
            }

            publishForwardOutputProvenance(
                output,
                producer_device,
                execution_stream_used,
                effective_input.execution_role,
                is_decode,
                host.computeAllPositionLogitsEnabled(),
                forwardCompletionScopeForInput(effective_input),
                effective_input.seq_len,
                effective_input.batch_size);

            if (is_decode && producer_device.is_gpu())
            {
                const bool all_position_verifier =
                    host.computeAllPositionLogitsEnabled();
                const bool defer_all_position_verifier =
                    all_position_verifier &&
                    host.shouldDeferAllPositionVerifierFinalSync();
                const bool defer_main_decode =
                    !all_position_verifier &&
                    host.shouldDeferMainDecodeFinalSync();

                if ((all_position_verifier || defer_main_decode) &&
                    !execution_stream_used)
                {
                    LOG_ERROR("[ForwardExecutionEngine] GPU cache-miss decode requires an explicit producer stream for its device-state handoff on "
                              << producer_device.toString());
                    return false;
                }

                if (all_position_verifier)
                {
                    host.setPendingAllPositionVerifierStream(
                        execution_stream_used);
                }
                if (defer_all_position_verifier)
                {
                    deferred_all_position_verifier_sync = true;
                }
                if (defer_main_decode)
                {
                    host.setPendingMainDecodeStream(execution_stream_used);
                    deferred_main_decode_sync = true;
                }
            }
        }

        /*
         * Host materialization remains the default public forward boundary.
         * MTP decode is different: its immediate consumer is a GPU sampler or
         * verifier-publication stage. Cache misses publish the exact producer
         * stream through the same typed handoff used by cached replay, so the
         * first execution cannot silently fall back to a host stream drain.
         */
        if (success)
        {
            if (!is_decode && effectiveRealSeqLen(effective_input) > 0)
            {
                const int terminal_row = effectiveRealSeqLen(effective_input) - 1;
                if (!executor_.publishCapturedTerminalStateAfterGraphExecution(
                        graph,
                        terminal_row,
                        nullptr,
                        should_cache
                            ? "prefill_cache_miss"
                            : "prefill_eager",
                        effective_input.sequence_lengths_device,
                        effective_input.batch_size,
                        effective_input.seq_len,
                        effective_input.sequence_lengths))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to publish captured terminal prefill state after cache miss");
                    return false;
                }
            }

            const bool device_consumer_owns_logits =
                deferred_all_position_verifier_sync ||
                deferred_main_decode_sync;
            IDeviceContext *sync_ctx =
                host.getDeviceContext(producer_device);
            if (sync_ctx && !device_consumer_owns_logits)
            {
                if (!host.publishForwardResultAtBoundary(
                        output,
                        sync_ctx))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to publish cache-miss forward result at the device boundary");
                    return false;
                }
            }

            const std::string stage_context =
                should_cache
                    ? forwardGraphPerfContext(signature)
                    : (is_decode ? "main_decode" : "prefill");
            collectTimeline(
                host,
                host.getDeviceContext(effective_input.device),
                is_decode, effective_input, start, stage_context);
        }

        if (success)
        {
            recordMoEOverlayCollectiveTransaction(
                effective_input,
                moe_overlay_collective_runtime_stages,
                "cache_miss");
        }

        // Cache the graph for future matching forward signatures
        if (should_cache && build_cache && success)
        {
            build_cache->graph = std::make_unique<ComputeGraph>(std::move(graph));
            build_cache->output = output;
            build_cache->workspace_generation = workspace_generation;
            build_cache->snapshot_configuration_epoch =
                executor_.snapshotConfigurationEpoch();

            // Pre-compute collective node set for fast decode intercept and
            // padded-prefill safety checks on later same-bucket cache hits.
            build_cache->collective_nodes = std::move(collective_nodes);

            build_cache->prefill_replay_param_stages = std::move(prefill_replay_param_stages);
            build_cache->prefill_replay_param_stages_cached = !is_decode;
            build_cache->moe_overlay_collective_runtime_stages =
                std::move(moe_overlay_collective_runtime_stages);
            build_cache->moe_overlay_collective_runtime_stages_cached = true;

            build_cache->valid = true;
            touchPrefillForwardCache(signature, *build_cache);

            // Store PP hidden state copy info for cache HIT replay
            if (initial_pp_copy.needs_copy)
            {
                build_cache->pp_external_hidden_state =
                    initial_pp_copy.external_hidden;
                build_cache->pp_working_buffer =
                    initial_pp_copy.working_buffer;
                build_cache->pp_copy_bytes = initial_pp_copy.copy_bytes;
                build_cache->pp_device = initial_pp_copy.device;
                build_cache->pp_needs_copy = true;

                LOG_DEBUG("[ForwardExecutionEngine] Stored PP copy info: "
                          << initial_pp_copy.copy_bytes << " bytes on "
                          << initial_pp_copy.device.toString());
            }

            LOG_DEBUG("[ForwardExecutionEngine] Cached forward graph for signature "
                      << "[seq_len=" << signature.seq_len
                      << ", batch_size=" << signature.batch_size
                      << ", device=" << signature.device.to_string()
                      << ", decode=" << signature.decode
                      << ", decode_has_history=" << signature.decode_has_history
                      << "] (" << build_cache->graph->size() << " stages)");

            if (signature.is_bucketed_prefill)
            {
                if (!build_cache->prefill_graph_cache)
                {
                    build_cache->prefill_graph_cache =
                        std::make_unique<PrefillGraphCache>(
                            makePrefillGraphConfigFromEnv(
                                host.residentGraphRows()));
                }

                PrefillGraphCacheKey key = makePrefillGraphKey(effective_input, host);
                if (bucketed_prefill_capture_candidate)
                {
                    build_cache->prefill_graph_cache->markWarmedUp(key);
                    publishPrefillGraphObservation(
                        *build_cache,
                        effective_input,
                        key,
                        PrefillGraphPhase::Warmup,
                        "warmup",
                        "none");
                    if (build_cache->prefill_graph_cache->config().trace)
                    {
                        LOG_INFO("[ForwardExecutionEngine] Prefill graph ARMED during cache miss warmup: seq_len="
                                 << effective_input.seq_len);
                    }
                }
                else
                {
                    publishPrefillGraphObservation(
                        *build_cache,
                        effective_input,
                        key,
                        PrefillGraphPhase::Cold,
                        "rejected",
                        toString(bucketed_prefill_reject_reason),
                        bucketed_prefill_reject_stage_name,
                        bucketed_prefill_reject_stage_type);
                }
            }
            if (!signature.decode)
                enforcePrefillForwardCapacity(&signature);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        LOG_DEBUG("[ForwardExecutionEngine] Forward completed in " << ms << "ms, success=" << success);

        return success;
    }

    // =========================================================================
    // collectTimeline() — GPU Stage Timing
    // =========================================================================

    void ForwardExecutionEngine::collectTimeline(
        IForwardExecutionHost &host,
        IDeviceContext *ctx,
        bool is_decode,
        const ForwardInput &input,
        std::chrono::high_resolution_clock::time_point start,
        std::string stage_context)
    {
        if (!PerfStatsCollector::gpuStageEventTimingEnabled() ||
            !ctx || !ctx->deviceId().is_gpu())
        {
            return;
        }

        auto &timeline = executor_.stageTimeline();
        if (!timeline.isInitialized())
        {
            return;
        }

        // When suppressed (warmup), discard any stale events but don't
        // collect or accumulate. This prevents stale events from the Cold
        // phase persisting after graph capture invalidates the stream state.
        if (suppress_timeline_)
        {
            timeline.resetTimings();
            return;
        }

        IWorkerGPUContext *gpu_ctx = host.getWorkerGPUContext(ctx->deviceId());
        if (!gpu_ctx)
        {
            LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for timeline collection on "
                      << ctx->deviceId().toString());
            return;
        }
        timeline.collect(gpu_ctx);

        double wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count();
        std::string dev_str = ctx->deviceId().toString();
        const char *dev_name = dev_str.c_str();
        const char *phase_name = is_decode ? "decode" : "prefill";
        PerfStatsCollector::Tags stage_tags;
        if (!stage_context.empty())
            stage_tags.emplace("context", std::move(stage_context));
        timeline.recordPerfStats(phase_name, dev_name, "stage_gpu", std::move(stage_tags));

        if (is_decode)
        {
            // Accumulate decode iterations — print once via flushStageTimeline()
            timeline.accumulateIteration(wall_ms);
        }
        else if (accumulate_prefill_)
        {
            // Accumulate prefill iterations — print once via flushStageTimeline() (benchmark mode)
            int tokens = input.batch_size * input.seq_len;
            timeline.accumulatePrefillIteration(wall_ms, tokens);
        }
        else
        {
            // Flush any pending decode data before printing prefill
            timeline.printAccumulatedSummary("DECODE", dev_name);

            int tokens = input.batch_size * input.seq_len;
            timeline.printSummary("PREFILL", tokens, wall_ms, dev_name);
            if (debugEnv().gpu_stage_timing_detail)
                timeline.printDetailedTimeline("PREFILL", dev_name);
        }
        timeline.resetTimings();
    }

    void ForwardExecutionEngine::forEachCachedStage(
        ComputeStageType type,
        const std::function<void(IComputeStage *)> &visitor) const
    {
        for (const auto &[sig, cache] : cache_)
        {
            if (!cache.valid || !cache.graph)
                continue;
            for (const auto &node_name : cache.graph->getExecutionOrder())
            {
                auto *node = cache.graph->getNode(node_name);
                if (node && node->stage && node->stage->type() == type)
                    visitor(node->stage.get());
            }
        }
    }

    void ForwardExecutionEngine::forEachCachedStage(
        const std::function<void(IComputeStage *)> &visitor) const
    {
        for (const auto &[sig, cache] : cache_)
        {
            (void)sig;
            if (!cache.valid || !cache.graph)
                continue;
            for (const auto &node_name : cache.graph->getExecutionOrder())
            {
                auto *node = cache.graph->getNode(node_name);
                if (node && node->stage)
                    visitor(node->stage.get());
            }
        }
    }

    std::optional<ForwardExecutionEngine::PrefillGraphCacheSnapshot> ForwardExecutionEngine::prefillGraphCacheSnapshot(
        const ForwardGraphSignature &signature,
        const PrefillGraphCacheKey &key) const
    {
        auto it = cache_.find(signature);
        if (it == cache_.end())
            return std::nullopt;

        PrefillGraphCacheSnapshot snapshot;
        const ForwardGraphCache &forward_cache = it->second;
        snapshot.forward_cache_valid = forward_cache.valid;
        snapshot.eviction_count = prefill_forward_eviction_count_;
        snapshot.bucket_seq_len = key.seq_len;
        snapshot.domain_id = key.domain_id;
        snapshot.participant_id = key.participant_id;
        snapshot.placement_epoch = key.placement_epoch;
        snapshot.topology_signature = key.topology_signature;
        snapshot.capture_phase = "unknown";
        snapshot.recapture_reason = "none";
        snapshot.reject_stage_name.clear();
        snapshot.reject_stage_type.clear();

        if (forward_cache.last_prefill_graph_observation.valid)
        {
            const auto &observation = forward_cache.last_prefill_graph_observation;
            snapshot.observation_valid = true;
            snapshot.chunk_index = observation.chunk_index;
            snapshot.bucket_seq_len = observation.bucket_seq_len;
            snapshot.real_token_start = observation.real_token_start;
            snapshot.real_token_count = observation.real_token_count;
            snapshot.real_token_end = observation.real_token_end;
            snapshot.domain_id = observation.domain_id;
            snapshot.participant_id = observation.participant_id;
            snapshot.placement_epoch = observation.placement_epoch;
            snapshot.topology_signature = observation.topology_signature;
            snapshot.capture_phase = observation.capture_phase;
            snapshot.recapture_reason = observation.recapture_reason;
            snapshot.reject_stage_name = observation.reject_stage_name;
            snapshot.reject_stage_type = observation.reject_stage_type;
        }

        if (forward_cache.segment_cache.initialized ||
            forward_cache.segment_cache.successful_capture_count != 0u)
        {
            applySegmentedPrefillGraphSnapshot(
                forward_cache.segment_cache, &snapshot);
            return snapshot;
        }

        if (!forward_cache.prefill_graph_cache)
            return snapshot;

        const PrefillGraphCache &prefill_cache = *forward_cache.prefill_graph_cache;
        snapshot.prefill_cache_initialized = true;
        snapshot.phase = prefill_cache.phase(key);
        snapshot.cache_size = prefill_cache.size();
        snapshot.node_count = prefill_cache.nodeCount(key);
        snapshot.replay_count = prefill_cache.replayCount(key);
        snapshot.warmup_count = prefill_cache.warmupCount(key);
        snapshot.initialized_count = prefill_cache.initializedCount(key);
        snapshot.capture_count = prefill_cache.captureCount(key);
        snapshot.eviction_count += prefill_cache.evictionCount();
        return snapshot;
    }

} // namespace llaminar2
