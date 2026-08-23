/**
 * @file DeviceGraphExecutor.h
 * @brief Compute graph execution engine
 * @author David Sanftenberg
 * @date December 2025
 *
 * DeviceGraphExecutor coordinates the execution of ComputeStages within a compute graph,
 * managing:
 * - Device-aware stage scheduling
 * - Asynchronous execution with dependency tracking
 * - Multi-device work distribution via WorkDistributor
 *
 * Note: This was previously named LayerExecutor but renamed to better reflect
 * its purpose - it executes graphs, not layers specifically.
 *
 * Architecture:
 *   Pipeline / ModelExecutor
 *      |
 *      v
 *   DeviceGraphExecutor  <-- This component
 *      |
 *      v
 *   ComputeStage[] (device-specific implementations)
 *      |
 *      v
 *   DeviceContext (execution environment)
 */

#pragma once

#include "ComputeGraph.h"
#include "GraphCaptureGuard.h"
#include "IGraphCaptureAuxiliaryBranch.h"
#include "IGraphExecutor.h"
#include "../device/DeviceContext.h"
#include "StageTimeline.h"
#include "../../../utils/DebugEnv.h" // For LLAMINAR_ASSERTIONS_ACTIVE
#include "../../../interfaces/ICollectiveContext.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../memory/BufferArena.h" // Phase 2: contract-based coherence
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <optional>
#include <span>
#include <string_view>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class FP32Tensor;

    /**
     * @brief Policy controlling what happens during stage execution.
     *
     * All execution paths (full, fast-decode, graph-capture manual segments)
     * use a single `runStages()` loop parameterised by this policy. This
     * eliminates the class of bugs caused by divergent code paths where one
     * path does coherence / validation / profiling and another silently skips it.
     *
     * Factory methods provide canonical configurations:
     *   - full()       — first-time execution (prefill, cache miss)
     *   - fastDecode() — cached decode graphs where buffers are already on-device
     *   - debug()      — full + timeline (for diagnostics)
     */
    struct StageRunPolicy
    {
        /**
         * @brief Authority semantics while C++ stage methods record graph nodes.
         */
        enum class GraphRecordingAuthority : uint8_t
        {
            RuntimeExecution = 0, ///< Inputs are live and successful writes publish authority.
            SetupAddressesOnly,  ///< Record immutable addresses; no live bytes or writes exist yet.
        };

        bool coherence = true;            ///< Arena contract-based input/output coherence
        bool weight_coherence = true;     ///< Upload weights to device
        bool mark_dirty = true;           ///< Publish outputs after real execution; setup-only recording disables it.
        bool validation = true;           ///< NaN/Inf output validation (Debug/Integration only)
        bool profiling = true;            ///< Per-stage timing breakdown
        bool collective_intercept = true; ///< Use CollectiveContext for allreduce/allgather
        bool timeline = false;            ///< GPU event-based per-stage profiling
        bool stage_dump = true;           ///< Stage dump framework (input/output snapshots)
        bool snapshot_callback = true;    ///< Invoke snapshot callback after execution
        bool pointer_validation = false;  ///< GPU pointer device validation
        bool preserve_gpu_streams = false; ///< Keep caller-assigned streams instead of rebinding normal passes to the worker stream
        GraphRecordingAuthority graph_recording_authority =
            GraphRecordingAuthority::RuntimeExecution; ///< Setup capture may bind pointers without publishing payload authority.

        /// Full execution — coherence, validation, profiling, everything.
        static StageRunPolicy full()
        {
            StageRunPolicy p{};
            p.timeline = true; // Enable GPU event timeline for prefill profiling
            return p;
        }

        /// Fast decode — minimal overhead for cached decode graphs.
        /// Buffers are already on-device, weights already uploaded.
        /// Legacy per-stage profiling remains independently opt-in; production
        /// graph replay timing is collected by PerfStats events around replay.
        static StageRunPolicy fastDecode()
        {
            StageRunPolicy p;
            p.coherence = false;
            p.weight_coherence = false;
            p.mark_dirty = true;
            p.validation = false;
            p.profiling = debugEnv().profile.enabled;
            p.stage_dump = false;
            p.snapshot_callback = false;
            p.pointer_validation = false;
            p.timeline = true;
            return p;
        }

        /// Capture phase — normal coherence and dirty marking, no host diagnostics.
        ///
        /// Phase-2 graph capture executes stages while HIP/CUDA stream capture may
        /// be active on sibling TP participants.  Diagnostics that materialize
        /// outputs on the host (validation, dumps, snapshot callbacks) are not
        /// part of inference semantics and can make HIP report illegal legacy
        /// stream dependencies during concurrent multi-device capture.
        static StageRunPolicy capturePhase()
        {
            StageRunPolicy p;
            p.coherence = true;
            p.weight_coherence = true;
            p.mark_dirty = true;
            p.validation = false;
            p.profiling = debugEnv().profile.enabled;
            p.collective_intercept = true;
            p.timeline = false;
            p.stage_dump = false;
            p.snapshot_callback = false;
            p.pointer_validation = false;
            p.preserve_gpu_streams = true;
            return p;
        }

        /**
         * @brief Record a setup-owned graph without importing request bytes.
         *
         * The executor has already allocated every arena binding, joined all
         * immutable weight producers, and prepared launch metadata. Stage calls
         * therefore only enqueue native nodes. They must neither read host
         * coherence state nor claim that captured outputs have executed.
         */
        static StageRunPolicy setupGraphMaterialization()
        {
            StageRunPolicy p = capturePhase();
            p.coherence = false;
            p.weight_coherence = false;
            p.mark_dirty = false;
            p.profiling = false;
            p.graph_recording_authority =
                GraphRecordingAuthority::SetupAddressesOnly;
            return p;
        }

        /// Debug — everything on, including timeline.
        static StageRunPolicy debug()
        {
            StageRunPolicy p;
            p.timeline = true;
            p.pointer_validation = true;
            return p;
        }
    };

    /**
     * @brief Orchestrates execution of compute stages within a graph
     *
     * DeviceGraphExecutor manages the execution of compute graphs by handling
     * topological ordering, device contexts, and execution modes.
     *
     * Usage:
     * @code
     * DeviceGraphExecutor executor(config);
     *
     * // Build a graph
     * ComputeGraph graph;
     * graph.addNode("norm", createRMSNormStage(...));
     * graph.addNode("proj", createGEMMStage(...));
     * graph.addDependency("proj", "norm");
     *
     * // Execute
     * executor.execute(graph, ctx);
     * @endcode
     */
    class DeviceGraphExecutor : public IGraphExecutor
    {
    public:
        /**
         * @brief One immutable, graph-owned GPU snapshot output descriptor.
         *
         * The graph owner's launch-preparation pass finalizes both this
         * descriptor and its stable destination storage before native capture
         * begins. The captured producer records the point-in-time D2D copy;
         * replay publication consumes the same immutable slot. Identically
         * named stages in another graph geometry therefore cannot overwrite it.
         */
        struct GraphSnapshotOutputCopy
        {
            std::string name;
            std::string dtype;
            size_t rows = 0;
            size_t cols = 0;
            size_t element_size = sizeof(float);
            size_t byte_size = 0;
            size_t storage_bytes = 0;
            DeviceId device = DeviceId::invalid();
            const void *source_ptr = nullptr;
            /**
             * True after launch preparation froze the complete source and
             * destination descriptor. The producer has not executed at that
             * point; its point-in-time D2D copy is recorded later inside the
             * native graph. Capture rejects any descriptor drift.
             */
            bool descriptor_finalized = false;
            std::unique_ptr<FP32Tensor> storage;
        };

        /** @brief Snapshot slots owned by one stage in one forward graph. */
        struct GraphSnapshotStageCopies
        {
            std::vector<GraphSnapshotOutputCopy> outputs;
        };

        /**
         * @brief Complete GPU snapshot manifest owned by one forward graph.
         *
         * Stage names are unique only inside a ComputeGraph; they are not
         * process-wide identities. ForwardGraphCache stores this object beside
         * the executable whose captured D2D nodes write these slots.
         */
        struct GraphSnapshotManifest
        {
            std::unordered_map<std::string, GraphSnapshotStageCopies> stage_copies;
            std::unordered_set<std::string> outputless_stages;
            /**
             * Stages rejected by the descriptor-aware filter during preflight.
             *
             * Replay must use this frozen decision instead of re-entering a live
             * stage or repeating name-only inference after graph launch.
             */
            std::unordered_set<std::string> filtered_stages;

            void clear()
            {
                stage_copies.clear();
                outputless_stages.clear();
                filtered_stages.clear();
            }
        };

        /**
         * @brief Logical token rows exposed from a fixed-width graph snapshot.
         *
         * Captured prefill graphs execute a stable bucket width even when a
         * request owns fewer real tokens. Snapshot D2D nodes must retain that
         * fixed physical shape as part of graph identity, while diagnostic
         * consumers must see only the request-owned rows. This descriptor
         * identifies the real prefix of every sequence in the captured batch;
         * publication compacts those prefixes on the host after the graph has
         * completed and before invoking the diagnostic callback.
         */
        struct GraphSnapshotLogicalRows
        {
            size_t physical_rows_per_sequence = 0;
            std::vector<size_t> logical_rows_per_sequence;

            /** @brief Return true when this descriptor requests row projection. */
            [[nodiscard]] bool active() const noexcept
            {
                return physical_rows_per_sequence > 0 &&
                       !logical_rows_per_sequence.empty();
            }
        };

        /**
         * @brief Construct with configuration
         * @param config Executor configuration
         */
        explicit DeviceGraphExecutor(const GraphExecutorConfig &config = GraphExecutorConfig());
        ~DeviceGraphExecutor() override;

        // Non-copyable, movable
        DeviceGraphExecutor(const DeviceGraphExecutor &) = delete;
        DeviceGraphExecutor &operator=(const DeviceGraphExecutor &) = delete;
        DeviceGraphExecutor(DeviceGraphExecutor &&) = default;
        DeviceGraphExecutor &operator=(DeviceGraphExecutor &&) = default;

        // =========================================================================
        // Configuration (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Get current configuration
         */
        const GraphExecutorConfig &config() const override { return config_; }

        /**
         * @brief Set execution mode
         */
        void setExecutionMode(ExecutionMode mode) override { config_.mode = mode; }

        /**
         * @brief Get current execution mode
         */
        ExecutionMode executionMode() const override { return config_.mode; }

        /**
         * @brief Enable/disable profiling
         */
        void setProfilingEnabled(bool enabled) override { config_.enable_profiling = enabled; }

        /**
         * @brief Enable/disable output validation
         */
        void setValidationEnabled(bool enabled) override { config_.enable_validation = enabled; }

        /**
         * @brief Set snapshot callback for debugging
         */
        void setSnapshotCallback(StageSnapshotCallback callback) override
        {
            config_.snapshot_callback = std::move(callback);
            advanceSnapshotConfigurationEpoch();
        }
        void setSnapshotStageFilter(StageSnapshotFilter filter)
        {
            config_.snapshot_stage_filter = std::move(filter);
            advanceSnapshotConfigurationEpoch();
        }
        /**
         * @brief Current graph identity epoch for diagnostic snapshot nodes.
         *
         * Snapshot callback/filter changes alter captured graph topology because
         * selected GPU stages gain or lose D2D copy nodes. Cached executables
         * compare this epoch before launch and re-enter their normal warmup and
         * capture lifecycle when it changes.
         */
        uint64_t snapshotConfigurationEpoch() const noexcept
        {
            return snapshot_configuration_epoch_;
        }

        void setStageFailureCallback(StageFailureCallback callback) { config_.stage_failure_callback = std::move(callback); }
        void setCancellationCallback(ExecutionCancellationCallback callback) { config_.cancellation_requested = std::move(callback); }
        /** @brief Override worker GPU context resolution for this executor. */
        void setWorkerGPUContextResolver(
            std::function<IWorkerGPUContext *(DeviceId)> resolver,
            bool uses_process_pool = false)
        {
            config_.worker_gpu_context_resolver = std::move(resolver);
            config_.worker_gpu_context_uses_process_pool = uses_process_pool;
        }

        /**
         * @brief Set the current layer context for stage dumping
         */
        void setCurrentLayerIdx(int layer_idx) override { config_.current_layer_idx = layer_idx; }

        /**
         * @brief Set the current iteration context for stage dumping
         */
        void setCurrentIteration(int iteration) override { config_.current_iteration = iteration; }

        /**
         * @brief Set the MPI rank for stage dumping
         */
        void setMPIRank(int rank) override { config_.mpi_rank = rank; }

        // =========================================================================
        // Execution (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Execute a compute graph
         * @param graph The compute graph to execute
         * @param ctx Device context for execution
         * @return true on success
         */
        bool execute(ComputeGraph &graph, IDeviceContext *ctx) override;

        /**
         * @brief Frozen host schedule for an immutable stage-owned multi-device graph.
         *
         * This plan is intended for explicit heterogeneous boundaries whose
         * stages own their transfer/coherence contracts (for example, one CPU
         * MPI dispatch, several participant-stream submissions, their completion
         * wave, and one CPU return). Setup resolves every graph node and exact
         * context once. Steady execution verifies the scalar topology generation
         * before dereferencing any borrowed pointer and never reconstructs an
         * unordered context lookup or completion-state traversal.
         */
        struct RetainedMultiDeviceExecutionPlan
        {
            /** @brief One pre-resolved stage invocation in topological order. */
            struct Entry
            {
                ComputeNode *node = nullptr;          ///< Graph-owned immutable node.
                IDeviceContext *context = nullptr;    ///< Exact setup-owned context.
                DeviceId device = DeviceId::invalid(); ///< Device identity sealed at setup.
                /**
                 * @brief Exact persistent worker that owns this GPU's state.
                 *
                 * CPU entries leave this null. GPU entries are submitted to
                 * this worker so independent devices can launch concurrently
                 * without accessing streams, events, or library handles from
                 * the rank controller thread.
                 */
                IWorkerGPUContext *worker_context = nullptr;
                bool validate_prepared_each_execution = true; ///< Conservative stage validation policy.
            };

            /**
             * @brief One dependency-complete host scheduling wave.
             *
             * All entries in a wave have had every predecessor wave finish.
             * A wave is eligible for concurrent submission only when every
             * member is a GPU entry on a distinct device. CPU/manual protocol
             * boundaries remain ordered on the rank controller thread.
             */
            struct Wave
            {
                std::vector<size_t> entry_indices; ///< Indices into @ref entries.
                bool concurrent_distinct_gpu_devices = false; ///< Setup-proven concurrency contract.
            };

            const ComputeGraph *graph = nullptr; ///< Exact graph that owns every entry.
            uint64_t topology_generation = 0;    ///< Graph topology represented by entries.
            uint64_t snapshot_configuration_epoch = 0; ///< Executor diagnostics identity.
            std::vector<Entry> entries;          ///< Direct execution schedule.
            std::vector<Wave> waves;              ///< Dependency-complete execution waves.
            size_t max_wave_width = 0;            ///< Scratch bound for one replay.
            size_t concurrent_gpu_wave_count = 0; ///< Setup evidence for the production path.

            /** @return true when this plan carries one complete graph schedule. */
            [[nodiscard]] bool valid() const noexcept
            {
                return graph != nullptr && topology_generation != 0 &&
                       !entries.empty() && !waves.empty() &&
                       max_wave_width != 0;
            }

            /** @brief Forget all graph/context borrows without touching owners. */
            void clear() noexcept
            {
                graph = nullptr;
                topology_generation = 0;
                snapshot_configuration_epoch = 0;
                entries.clear();
                waves.clear();
                max_wave_width = 0;
                concurrent_gpu_wave_count = 0;
            }
        };

        /**
         * @brief Execute a compute graph with multi-device support
         * @param graph The compute graph to execute
         * @param contexts Map of DeviceId -> DeviceContext
         * @return true on success
         */
        bool executeMultiDevice(
            ComputeGraph &graph,
            const std::unordered_map<DeviceId, IDeviceContext *> &contexts) override;

        /**
         * @brief Validate and seal an immutable stage-owned multi-device schedule.
         *
         * Every stage must declare CoherencePolicy::NONE and an empty arena
         * contract, because the retained executor deliberately bypasses generic
         * arena preparation/publication. Prepared weights are proven here; only
         * stages retaining the conservative PerExecution lifetime are checked
         * again in the hot path.
         *
         * @param graph Finalized graph whose topology will remain immutable.
         * @param contexts Exact context map owned beyond @p plan.
         * @param plan Destination retained schedule.
         * @param error Optional first violated invariant.
         * @return true when every entry and lifetime contract is sealed.
         */
        bool prepareRetainedMultiDeviceExecutionPlan(
            ComputeGraph &graph,
            const std::unordered_map<DeviceId, IDeviceContext *> &contexts,
            RetainedMultiDeviceExecutionPlan &plan,
            std::string *error = nullptr);

        /**
         * @brief Execute one setup-sealed heterogeneous host schedule.
         *
         * No graph reset, name lookup, context lookup, or completion-flag
         * mutation occurs. Setup-sealed dependency waves retain the graph's
         * exact ordering while independent distinct-device GPU stages submit
         * through their persistent worker contexts. CPU/manual boundaries and
         * same-device entries remain serial.
         *
         * @param plan Valid retained schedule prepared by this executor.
         * @param error Optional first execution failure.
         * @return true after every stage completes its declared boundary.
         */
        bool executeRetainedMultiDevice(
            const RetainedMultiDeviceExecutionPlan &plan,
            std::string *error = nullptr);

        // =========================================================================
        // Statistics (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Get execution statistics
         */
        const GraphExecutorStats &stats() const override { return stats_; }

        /**
         * @brief Reset statistics
         */
        void resetStats() override { stats_.reset(); }

        // =========================================================================
        // Collective Context
        // =========================================================================

        /**
         * @brief Set the collective context for GPU-native collectives
         *
         * When a collective context is set, ALLREDUCE and ALLGATHER stages
         * will be intercepted and executed via the CollectiveContext
         * infrastructure (RCCL, NCCL, HOST) instead of the stage's
         * internal MPI fallback.
         *
         * @param ctx Collective context (not owned, must outlive executor)
         */
        void setCollectiveContext(ICollectiveContext *ctx) { collective_ctx_ = ctx; }

        /**
         * @brief Get the current collective context
         * @return Pointer to collective context (nullptr if not set)
         */
        ICollectiveContext *collectiveContext() const { return collective_ctx_; }

        // =========================================================================
        // Buffer Management
        // =========================================================================

        /**
         * @brief Set the BufferArena for contract-based coherence
         *
         * When both the arena is set and a stage returns a non-empty bufferContract(),
         * the executor uses the arena for coherence instead of getDumpInfo().
         *
         * @param arena BufferArena (not owned, must outlive executor)
         */
        void setArena(BufferArena *arena) { arena_ = arena; }

        /**
         * @brief Get the current BufferArena
         * @return Pointer to arena (nullptr if not set)
         */
        BufferArena *arena() const { return arena_; }

        /**
         * @brief Execute a cached decode graph with minimal overhead
         *
         * Stripped-down execution path for cached decode graphs where:
         * - All tensors are already on the target GPU device
         * - Output GPU buffers are already allocated
         * - No stage dumping, validation, or profiling is needed
         * - Stage objects are reused (not rebuilt)
         *
         * Skips: getDumpInfo and ordinary arena preparation,
         *        shouldDump, printStageOutputs, profiling, assertions
         *
         * The collective_nodes set enables O(1) lookup for TP stages instead
         * of calling stage->type() (virtual dispatch) on every node.
         *
         * @param graph The cached compute graph (stages already configured)
         * @param ctx Device context for execution
         * @param collective_nodes Pre-computed set of node names that are collective stages (optional, for TP>1)
         * @return true on success
         */
        /**
         * @brief Get the stage timeline for external collection/printing
         *
         * The timeline is populated during executeFastDecode() when
         * LLAMINAR_GPU_STAGE_TIMING=1. The caller is responsible for calling
         * collect() and printSummary() after the forward pass completes.
         */
        StageTimeline &stageTimeline() { return stage_timeline_; }
        const StageTimeline &stageTimeline() const { return stage_timeline_; }

        bool executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx,
                               const std::unordered_set<std::string> *collective_nodes = nullptr,
                               GraphSnapshotManifest *snapshot_manifest = nullptr);

        /**
         * @brief Record a complete setup-owned graph body without executing it.
         *
         * This entry point is legal only while the exact backend stream is under
         * native graph capture. It invokes the same canonical stage runner as
         * production execution, but with @ref
         * StageRunPolicy::setupGraphMaterialization: arena addresses and prepared
         * kernel descriptors may be embedded in graph nodes, while request
         * coherence, dirty publication, snapshot callbacks, completion flags, and
         * profiling authority remain untouched. Calling it outside native capture
         * is a hard error because stage methods would otherwise execute real model
         * work during setup.
         *
         * @param graph Stable participant-local graph being materialized.
         * @param ctx Exact GPU context that owns the active capture stream.
         * @param collective_nodes Precomputed collective-stage identities.
         * @param snapshot_manifest Graph-owned D2D snapshot slots to record.
         * @return true after every stage node was recorded successfully.
         */
        bool recordSetupGraphMaterialization(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            const std::unordered_set<std::string> *collective_nodes = nullptr,
            GraphSnapshotManifest *snapshot_manifest = nullptr);

        /**
         * @brief Execute an eager graph against its explicit snapshot owner.
         *
         * Cached graph construction uses this boundary so warmup descriptors
         * remain attached to the exact graph that will later capture them.
         */
        bool executeWithSnapshotManifest(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            GraphSnapshotManifest &snapshot_manifest);

        /**
         * @brief Publish snapshot callbacks from an already-executed graph.
         *
         * GPU graph capture/replay intentionally disables per-stage callbacks
         * while kernels are being recorded or launched. Parity diagnostics still
         * need the resulting stage tensors. GPU stages are published exclusively
         * from the immutable graph-snapshot manifest whose D2D copy nodes ran at
         * each stage boundary; the executor never re-queries mutable stage dump
         * metadata after launch. CPU stages retain their ordinary synchronous
         * dump callback because they do not participate in GPU graph capture.
         *
         * @param graph Graph whose stages have just executed.
         * @param producer_stream_override Explicit producer stream for the graph
         *                                 execution. When non-null it is treated
         *                                 as authoritative, because captured graph
         *                                 launches may run on a stream different
         *                                 from stale per-stage stream pointers.
         *                                 Null is allowed only for stages whose
         *                                 outputs are already host-readable or
         *                                 whose stage stream is current.
         * @param context Human-readable context for diagnostics.
         * @param logical_rows Optional request-owned row projection for a
         *                     fixed-width captured graph. Outputs whose row
         *                     count does not match the captured token matrix
         *                     retain their immutable manifest shape.
         * @return true if all requested snapshots were published.
         */
        bool publishSnapshotsAfterGraphExecution(
            ComputeGraph &graph,
            void *producer_stream_override = nullptr,
            const char *context = nullptr,
            GraphSnapshotManifest *snapshot_manifest = nullptr,
            const GraphSnapshotLogicalRows *logical_rows = nullptr);

        /**
         * @brief Publish the terminal captured mutable state row from a graph.
         *
         * GDN/short-conv stages can execute against speculative state slots when
         * snapshot/verifier capture is enabled. The graph may then have correct
         * tensor outputs while the backend-owned live recurrent state still
         * points at the pre-graph value. This helper restores the final real row
         * into stage-owned live state on the graph stream before callers observe,
         * snapshot, or prefix-cache that live state.
         *
         * This is intentionally graph-generic: prefix cache, parity snapshots,
         * and future runtime-state transactions should all depend on the same
         * publication primitive instead of each feature inventing its own GDN
         * state side path.
         *
         * @param graph Graph whose mutable-state stages have just completed.
         * @param terminal_row Flat terminal row for a scalar request. Ignored
         *        for a logical request batch, where terminal rows are derived
         *        from the metadata owner appropriate to each stage backend.
         * @param producer_stream_override Stream that produced graph outputs,
         *        or null to use each stage's bound stream.
         * @param context Optional diagnostic label.
         * @param device_request_seq_lens Stable device metadata allocation.
         *        This pointer may remain non-null when request capacity exceeds
         *        the active scalar request count; `request_count`, not pointer
         *        presence, selects grouped publication.
         * @param request_count Active logical request count for this execution.
         * @param request_row_width Padded rows per request for grouped state.
         * @param host_request_seq_lens Host-owned real lengths for CPU request
         *        batches. GPU stages never consume this vector; they derive
         *        terminal rows from @p device_request_seq_lens without a host
         *        synchronization. At least one length owner is required for a
         *        logical request batch, and each stage requires the owner that
         *        matches its backend.
         * @return true when every captured state owner is committed.
         */
        bool publishCapturedTerminalStateAfterGraphExecution(
            ComputeGraph &graph,
            int terminal_row,
            void *producer_stream_override = nullptr,
            const char *context = nullptr,
            const int *device_request_seq_lens = nullptr,
            int request_count = 1,
            int request_row_width = 0,
            const std::vector<int> *host_request_seq_lens = nullptr);

        /**
         * @brief Pre-register graph-stable snapshot buffers before GPU graph capture.
         *
         * Captured parity diagnostics record device-to-device snapshot copies as
         * graph nodes immediately after producing stages execute. The destination
         * buffers and per-stage output descriptors must therefore exist before
         * beginCapture(); discovering either while capture is active is a hard
         * graph-capture contract violation.
         *
         * This method performs no payload copies and does not execute graph
         * stages. Callers must pass the explicit graph stream that will be used
         * for capture.
         */
        bool prepareSnapshotsForGraphCapture(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            void *producer_stream,
            const char *context = nullptr,
            GraphSnapshotManifest *snapshot_manifest = nullptr);

        /**
         * @brief Allocate stable arena addresses before launch preparation.
         *
         * Cold graph construction cannot assume an eager execution has already
         * allocated every arena tensor. Some stages build persistent descriptor
         * tables during `prepareGraphLaunch()` and therefore need the final
         * addresses of their arena inputs and scratch outputs at that earlier
         * lifecycle point. This allocation-only pass neither transfers payload
         * bytes nor changes tensor authority; exact-stream input coherence still
         * belongs to @ref prepareGraphStorageForCapture after launch preparation.
         *
         * @param graph Graph whose arena bindings need stable GPU addresses.
         * @param ctx GPU context that owns the future capture transaction.
         * @param context Optional diagnostic label.
         * @return true after every declared arena binding has stable storage.
         */
        bool allocateGraphStorageForCapture(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            const char *context = nullptr);

        /**
         * @brief Prepare the complete arena storage frontier before graph capture.
         *
         * Arena residency and producer ordering are separate contracts. Warmup
         * can leave valid activation storage whose completion event belongs to a
         * different eager stream. CUDA/HIP graph capture cannot import that
         * external event after `beginCapture()`, so this method walks the graph
         * in topological order and joins only reads whose producer is outside the
         * captured graph. Reads produced by an earlier stage are internal native
         * graph edges and must never be mistaken for stale external inputs.
         * Raw tensor weights consumed directly by a stage are external reads too;
         * their producer events are joined here even though they are not arena
         * bindings. Prepared-weight entries instead name backend-owned storage
         * validated by launch preparation. Every declared output also receives
         * its stable device allocation before capture begins; allocation does not
         * publish authority or invent output bytes. Backing tensors are
         * deduplicated before event joins or storage allocation are performed on
         * @p capture_stream.
         *
         * `TransferEngine::requireDeviceInput()` records each exact
         * `{completion event, capture stream}` pair. Stage-level prepared-input
         * guards can then prove the dependency was established and must hard-fail
         * instead of attempting a late capture-time event wait.
         *
         * @param graph Graph whose complete native body will be captured.
         * @param ctx GPU context owning the graph and stream.
         * @param capture_stream Exact non-null stream passed to beginCapture().
         * @param context Optional diagnostic label.
         * @param external_input_authority Whether the frontier must own live
         *        request bytes or setup may bind its already allocated addresses.
         * @return true after all external dependencies and output storage are ready.
         */
        bool prepareGraphStorageForCapture(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            void *capture_stream,
            const char *context = nullptr,
            GraphCaptureDependencyLedger::ExternalInputAuthority
                external_input_authority =
                    GraphCaptureDependencyLedger::ExternalInputAuthority::
                        RequireReadyBytes);

        /**
         * @brief Freeze the typed stage dependency sequence for native capture.
         *
         * This planner resolves BufferId contracts to canonical tensor storage
         * owners before beginCapture().  Each internal read records the precise
         * earlier producer-stage index, while values entering the capture unit
         * remain strict external inputs.  The returned ledger owns only host-side
         * identities and booleans; it performs no allocation while recording.
         *
         * @param graph Graph containing every named stage.
         * @param stage_names Exact topological capture unit, in recording order.
         * @param capture_device GPU owning this native graph transaction.
         * @param capture_stream Exact non-null stream passed to beginCapture().
         * @param context Stable diagnostic label.
         * @param retained_parent_input_ids Arena values produced by an earlier
         *        child of the same mandatory retained parent. Ordinary and
         *        manually segmented captures must pass an empty span.
         * @param external_input_authority Whether typed frontier inputs must
         *        own live bytes or setup may bind only their stable addresses.
         * @return Frozen ledger, or nullptr after a precise planning diagnostic.
         */
        std::unique_ptr<GraphCaptureDependencyLedger>
        planGraphCaptureDependencies(
            ComputeGraph &graph,
            std::span<const std::string> stage_names,
            DeviceId capture_device,
            void *capture_stream,
            const char *context,
            std::span<const BufferId> retained_parent_input_ids = {},
            GraphCaptureDependencyLedger::ExternalInputAuthority
                external_input_authority =
                    GraphCaptureDependencyLedger::ExternalInputAuthority::
                        RequireReadyBytes);

        /**
         * @brief Record one complete participant-local graph as a retained child.
         *
         * This operation performs the same storage prebinding, launch
         * preparation, dependency-ledger validation, and structural backend
         * capture used by normal production graph execution. It intentionally
         * stops after `endCapture()`: the returned native graph is not
         * instantiated or launched because a parent transaction will import it
         * as a captured fragment.
         *
         * Retained composition is stricter than immediate replay. The capture
         * owner must be pristine, every graph node must target @p ctx exactly,
         * collectives and manual host boundaries are rejected, and no stage may
         * require per-replay launch preparation. The caller must retain the
         * graph, arena, tensors, prepared weights, capture owner, and exact
         * stream for at least as long as every parent graph importing the child.
         *
         * @param graph Complete same-device declarative child graph.
         * @param ctx Exact GPU device context selected by topology planning.
         * @param capture Pristine capture owner bound to a non-null stream.
         * @param context Stable diagnostic identity for this retained fragment.
         * @return true only when a non-empty native child graph was recorded.
         */
        bool captureRetainedGraphFragment(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            IGPUGraphCapture *capture,
            const char *context = nullptr);

        /**
         * @brief Execute a cached decode graph with GPU graph capture/replay
         *
         * On first call: captures all kernel launches into a GPU graph, instantiates
         * and launches it. On subsequent calls: re-captures, updates the executable
         * in-place (hipGraphExecUpdate/cudaGraphExecUpdate), and launches.
         *
         * Capture, instantiation, update, and launch failures are hard failures.
         * This API never changes execution modes after work begins. Callers
         * selecting eager execution must do so explicitly before invoking it.
         *
         * @param graph The cached compute graph
         * @param ctx Device context for execution
         * @param capture GPU graph capture object (created once, reused across calls)
         * @param externally_visible_outputs Tensor outputs whose device writes
         *        become observable outside this graph call. The executor records
         *        their completion events on capture.executionStream() after the
         *        graph launch. An empty list is an invalid publication contract.
         * @param collective_nodes Pre-computed collective node names (for TP>1)
         * @return true on success
         */
        bool executeWithGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                     IGPUGraphCapture *capture,
                                     std::span<ITensor *const> externally_visible_outputs,
                                     const std::unordered_set<std::string> *collective_nodes = nullptr);

        // =========================================================================
        // Cached GPU Graph Replay
        // =========================================================================

        /**
         * @brief One contiguous replay unit in a GPU execution graph.
         *
         * A production homogeneous CUDA/ROCm graph should normally contain one
         * capturable unit spanning compute, state mutation, and NCCL/RCCL
         * collectives. Multiple units have two explicit meanings: host-sequenced
         * units around a genuinely uncapturable heterogeneous boundary, or
         * graph-only children cloned into one topology-composed device parent.
         * They must never appear merely because a specialized collective was
         * omitted from orchestration classification.
         */
        struct GraphSegment
        {
            /**
             * @brief One domain capture wave joined without recording device work.
             *
             * A passive participant calls the same begin/end lifecycle
             * rendezvous as the sibling that records the active graph.  It does
             * not create or launch an empty CUDA/HIP graph.
             */
            struct PassiveCaptureWave
            {
                size_t ordinal = 0;      ///< Domain-ordered capture-wave number.
                std::string identity;    ///< Explicit identity shared with the active sibling segment.
                std::vector<std::string> declarative_noop_stage_names; ///< Statically passive nodes represented by this join.
            };

            struct ArenaWriteBinding
            {
                BufferId id;
                DeviceId device;
            };

            std::vector<std::string> stage_names;          ///< Ordered stage names in this segment
            bool capturable = true;                        ///< Whether this segment can be graph-captured
            std::unique_ptr<IGPUGraphCapture> capture;     ///< GPU graph (only for capturable segments)
            uint64_t last_executed_step = 0;               ///< Last decode-step where this segment executed
            size_t capture_wave_ordinal = 0;                ///< Ordered active capture wave; meaningful only when capturable.
            std::string capture_wave_identity;              ///< Optional explicit cross-participant identity.
            std::vector<PassiveCaptureWave> passive_capture_waves_after; ///< Ordered no-work waves joined after this segment completes.

            /**
             * Arena reads whose producer is an earlier native child of this
             * exact retained parent. The child records a stable pointer read;
             * the parent composer installs the device-side ordering edge. This
             * list is empty for full-graph and host-segmented replay policies.
             */
            std::vector<BufferId> retained_parent_input_ids;

            // Output coherence is marked through stable BufferArena ids after
            // replay. Avoid retaining raw TensorBase* caches here: prefix
            // restore, rollback, and request clears can legally mutate tensor
            // ownership while preserving graph topology.
            std::vector<ArenaWriteBinding> cached_arena_writes;
            bool arena_writes_cached = false;
        };

        using GraphCaptureBoundaryHook =
            std::function<bool(const std::string &, void *capture_stream)>;

        /**
         * @brief Lifecycle position of one captured-graph executable launch.
         *
         * External producer events must be joined after native capture has
         * closed for transaction zero, but before the executable is launched.
         * Steady replay can join the same producers immediately before launch
         * because no capture transaction is active.  Keeping the distinction
         * typed prevents a caller from accidentally importing a cross-lifetime
         * event into the graph body.
         */
        enum class GraphExecutableLaunchPhase : uint8_t
        {
            InitialTransaction,   ///< First launch after capture and instantiation.
            SteadyReplay,         ///< Ordinary launch of an existing executable.
            DiagnosticRecapture,  ///< First launch after a diagnostic recapture.
            DiagnosticVerification ///< Captured launch used by graph verification.
        };

        /**
         * @brief Queue an external producer dependency on the exact graph stream.
         *
         * The callback may enqueue event waits on @p execution_stream; it must
         * not synchronize the host/device, allocate resources, transfer payload
         * data, or launch a substitute implementation.  The controller invokes
         * it exactly once immediately before each executable launch.  This hook
         * is legal only for one complete capturable graph and is rejected for
         * segmented or manual replay plans.
         */
        using GraphLaunchDependencyHook =
            std::function<bool(GraphExecutableLaunchPhase phase,
                               void *execution_stream)>;

        /**
         * @brief Topology contract for native graph materialization and replay.
         *
         * @ref RequireFullGraph is the ordinary homogeneous contract: every
         * operation is recorded into one native executable. @ref
         * AllowHeterogeneousBoundarySegmentation admits explicit manual device
         * boundaries and therefore retains host sequencing between graph units.
         *
         * @ref RequireRetainedParentComposition is materially different from
         * segmented replay. Every source unit must be a native, unlaunched child
         * graph. A topology-specific composer inserts device-owned ordering nodes
         * between those children and produces the sole executable that may launch.
         * This is the contract used by node-local heterogeneous sparse collectives;
         * it never authorizes a host walk over the child units.
         */
        enum class GraphReplayPlanPolicy
        {
            RequireFullGraph,
            AllowHeterogeneousBoundarySegmentation,
            RequireRetainedParentComposition,
        };

        /**
         * @brief First-use submission contract for a newly materialized graph.
         *
         * Ordinary inference records, instantiates, and submits transaction zero
         * atomically. Setup-owned graph families instead use @ref
         * MaterializeWithoutLaunch: every native unit (or its topology-composed
         * parent) is fully captured and instantiated before request admission,
         * while declared manual units remain unexecuted. The next ordinary call
         * performs the pending initial transaction in exact graph order and all
         * later calls are steady replay.
         */
        enum class GraphInitialSubmissionPolicy : uint8_t
        {
            CaptureInstantiateAndLaunch = 0, ///< Materialize and submit transaction zero.
            MaterializeWithoutLaunch, ///< Seal an executable during setup without submitting work.
        };

        /**
         * @brief Persistent cache of GPU graph replay units.
         *
         * Built on the first decode step and reused across subsequent steps.
         * The ordinary production plan has one fully captured replay unit.
         * Manual units are retained only for explicitly supported non-capturable
         * topologies and are structurally visible through replay-mode telemetry.
         */
        struct GraphSegmentCache
        {
            /**
             * @brief Submission state of the cache-owned executable.
             *
             * This state is deliberately distinct from @ref initialized, which
             * describes native materialization. A setup-sealed graph is
             * initialized but still owes its first transaction launch. Keeping
             * that distinction typed prevents an authenticated ticket or ordinary
             * first request from being mislabeled as steady replay or submitted
             * during setup.
             */
            enum class ExecutableSubmissionState : uint8_t
            {
                Empty = 0, ///< No complete executable is owned by the cache.
                MaterializedUnlaunched, ///< Instantiated executable awaits transaction zero.
                ReplayReady, ///< Transaction zero has been submitted successfully.
            };

            /**
             * @brief Select the host controller used for steady native replay.
             *
             * GeneralController preserves the complete diagnostic, segmented,
             * timing, and recapture machinery. RetainedFullGraph is an explicit
             * production contract for a fixed-topology, single-executable graph:
             * after the first capture the cache seals direct stage pointers and
             * launch identity, then validates those scalar identities before one
             * asynchronous launch. It never admits segmentation or silently
             * substitutes eager execution.
             */
            enum class SteadyReplayHostPolicy : uint8_t
            {
                GeneralController = 0, ///< Rebuild the general replay controller state each step.
                RetainedFullGraph,     ///< Use a sealed, prevalidated one-executable launch plan.
            };

            /**
             * @brief Frozen host-side launch plan for one steady full-graph replay.
             *
             * The ordinary replay controller is deliberately general: it supports
             * segmentation, diagnostics, recapture, timing events, and mutable
             * launch preparation.  Tiny retained endpoint graphs need none of
             * that machinery after capture.  This plan freezes the exact graph,
             * topology generation, ordered stage pointers, and their captured
             * variant signatures so steady replay can validate scalar identity
             * and submit one executable without rebuilding callback/traversal
             * state on every transformer layer.
             *
             * Stage pointers remain owned by the associated ComputeGraph.  A
             * topology mutation changes @ref ComputeGraph::topologyGeneration and
             * therefore makes this plan unusable before any pointer is dereferenced.
             */
            struct RetainedFullGraphReplayPlan
            {
                const ComputeGraph *graph = nullptr; ///< Exact graph whose stages were captured.
                uint64_t topology_generation = 0;    ///< Graph generation represented by the executable.
                uint64_t snapshot_configuration_epoch = 0; ///< Diagnostic topology represented at sealing.
                uint64_t capture_variant_signature = 0; ///< Graph-wide captured launch variant.
                std::vector<IComputeStage *> stages; ///< Stable execution-order stage owners.
                std::vector<uint64_t> stage_variant_signatures; ///< Captured variant for each stage above.

                /** @return true only when all fixed-size identity vectors agree. */
                [[nodiscard]] bool valid() const noexcept
                {
                    return graph != nullptr && topology_generation != 0 &&
                           !stages.empty() &&
                           stages.size() == stage_variant_signatures.size();
                }

                /** @brief Forget borrowed graph pointers before cache reset. */
                void clear() noexcept
                {
                    graph = nullptr;
                    topology_generation = 0;
                    snapshot_configuration_epoch = 0;
                    capture_variant_signature = 0;
                    stages.clear();
                    stage_variant_signatures.clear();
                }
            };

            /**
             * @brief Frozen identity for one topology-composed parent executable.
             *
             * The source children remain graph-only templates and are never
             * submitted. `stages` freezes the exact declarative graph whose
             * packet frontiers the topology composer inspected. `arena_writes`
             * is the deduplicated publication plan for the complete parent, so
             * steady replay performs one launch and one compact flags-only
             * authority update rather than walking or launching child units.
             */
            struct RetainedComposedParentReplayPlan
            {
                const ComputeGraph *graph = nullptr; ///< Exact source graph represented by the parent.
                uint64_t topology_generation = 0; ///< Source topology embedded in child graphs.
                uint64_t snapshot_configuration_epoch = 0; ///< Captured diagnostic topology.
                uint64_t capture_variant_signature = 0; ///< Graph-wide launch variant.
                size_t child_unit_count = 0; ///< Graph-only native templates cloned into the parent.
                std::vector<IComputeStage *> stages; ///< Stable source stages in execution order.
                std::vector<uint64_t> stage_variant_signatures; ///< Immutable launch variants.
                std::vector<GraphSegment::ArenaWriteBinding> arena_writes; ///< Complete deduplicated publication set.

                /** @return true when identity, child count, and stage vectors agree. */
                [[nodiscard]] bool valid() const noexcept
                {
                    return graph != nullptr && topology_generation != 0 &&
                           child_unit_count != 0 && !stages.empty() &&
                           stages.size() == stage_variant_signatures.size();
                }

                /** @brief Forget borrowed graph/stage identities during reset. */
                void clear() noexcept
                {
                    graph = nullptr;
                    topology_generation = 0;
                    snapshot_configuration_epoch = 0;
                    capture_variant_signature = 0;
                    child_unit_count = 0;
                    stages.clear();
                    stage_variant_signatures.clear();
                    arena_writes.clear();
                }
            };

            /**
             * @brief Immutable view of one complete replay-ready graph template.
             *
             * The cache retains ownership. A parent graph may clone the native
             * executable through @ref capture, but must not launch, reset, or
             * mutate it through this borrowed view.
             */
            struct DeviceLoopGraphTemplateView
            {
                const IGPUGraphCapture *capture = nullptr;
                void *stream = nullptr;
                size_t stage_count = 0;
                size_t captured_node_count = 0;
            };

            /**
             * @brief Borrowed view of one active unit in a retained capture plan.
             *
             * Heterogeneous sparse parents cannot clone a whole LocalTP graph as
             * one child because mapped dispatch/return timeline nodes must be
             * inserted between selected capture units.  The ordinary capture
             * controller still owns capture-wave rendezvous and records each
             * participant-local unit.  This view exposes that already-proven
             * native unit without exposing mutable cache state or authorizing a
             * host replay.
             *
             * `stage_names` and `capture_wave_identity` borrow storage from the
             * cache and remain valid only while the cache is unchanged.  The
             * native capture likewise remains owned by the cache.
             */
            struct RetainedCaptureUnitTemplateView
            {
                const IGPUGraphCapture *capture = nullptr; ///< Exact native child graph.
                void *stream = nullptr; ///< Exact stream used during coordinated capture.
                std::span<const std::string> stage_names; ///< Contiguous graph stages represented by the child.
                size_t capture_wave_ordinal = 0u; ///< Domain capture-wave ordinal.
                std::string_view capture_wave_identity; ///< Optional explicit cross-participant identity.
                size_t captured_node_count = 0u; ///< Native nodes retained by the child.
            };

            /**
             * @brief Lifecycle in which retained child-template inspection occurs.
             *
             * ReplayReady inspects ordinary instantiated child executables after
             * transaction zero. ParentComposition inspects graph-only children
             * during the atomic first-use transaction, before any child can be
             * instantiated or launched. The latter state is intentionally
             * inaccessible through the ordinary replay-ready exporter.
             */
            enum class RetainedCaptureUnitInspection : uint8_t
            {
                ReplayReady = 0,
                ParentComposition,
            };

            /**
             * @brief Stable execution geometry represented by this graph cache.
             *
             * Deferred GPU timing events can complete several host iterations
             * after their graph launch.  Reading mutable runner state when an
             * event is reclaimed would consequently attach the wrong M, batch,
             * or verifier mode to an otherwise accurate GPU interval.  The
             * descriptor instead travels with the cached executable whose event
             * ring produced the sample.
             *
             * The fields are deliberately scalar and backend-neutral.  They add
             * no allocation, transfer, synchronization, or graph node to replay;
             * PerfStats converts them to strings only when structured metrics
             * are already enabled.  `m` is the flattened physical row count and
             * is therefore the exact grouped-verifier work size for the common
             * batch-size-one MTP transaction.
             */
            struct ReplayWorkloadGeometry
            {
                int seq_len = 0;                 ///< Rows per logical request.
                int batch_size = 0;              ///< Logical request count.
                int m = 0;                       ///< Flattened physical graph rows.
                int all_position_rows = 0;       ///< Compact verifier/logit rows, or zero.
                uint8_t verifier_outcome_mode = 0; ///< Typed MTP outcome enum value.
                uint8_t position_policy = 0;     ///< Typed ForwardPositionPolicy value.
                uint64_t moe_placement_epoch = 0; ///< Expert-placement identity in the cache key.
                bool decode = false;             ///< Decode-equivalent graph family.
                bool all_position_logits = false; ///< Graph projects verifier rows.
                bool live_mtp_request_batch_condition = false; ///< One live condition row per request.

                [[nodiscard]] bool valid() const noexcept
                {
                    return seq_len > 0 && batch_size > 0 && m > 0;
                }
            };

            /**
             * @brief One preallocated asynchronous GPU replay timing interval.
             *
             * Timing events are backend resources, so they are created during
             * graph warmup and retained for the cache lifetime. Replay only
             * records the two existing events and changes this small host-side
             * state record; it never allocates, destroys, or synchronizes an
             * event. A completed interval is reclaimed with a nonblocking event
             * query before the slot is reused.
             */
            struct ReplayGpuTimingSlot
            {
                enum class Scope
                {
                    TotalReplay,
                    ReplayUnit
                };

                void *start_event = nullptr;
                void *stop_event = nullptr;
                Scope scope = Scope::TotalReplay;
                size_t replay_unit_index = 0;
                bool deferred_final_fence = false;
                bool started = false;
                bool pending = false;
            };

            /**
             * @brief Controls whether reset() keeps the explicit capture stream alive.
             *
             * Preserve is used for recapture/retry state resets where cached stages
             * may still hold the stream pointer. Destroy is reserved for teardown or
             * cache invalidation where no stage will execute again before rebinding.
             */
            enum class StreamResetPolicy
            {
                Preserve,
                Destroy
            };

            /**
             * @brief Select how a terminal host observer awaits one GPU event.
             *
             * Resource teardown uses the backend's blocking event wait because
             * latency no longer lies on an inference critical path. Explicit
             * heterogeneous transaction boundaries may instead actively query
             * the same event, avoiding scheduler wake-up latency before the
             * host publishes a small result to a remote participant. Both
             * policies fence one exact event; neither drains a stream or device.
             */
            enum class HostFenceWaitPolicy : uint8_t
            {
                Blocking = 0, ///< Let the backend block the calling host thread.
                ActiveProgress, ///< Poll with the canonical bounded deadline.
            };

            /**
             * @brief Ownership of the exact stream used for capture and replay.
             *
             * Ordinary forward graphs create a private stream and therefore
             * own its teardown.  A participant-local heterogeneous endpoint
             * may instead capture directly on its context-owned worker stream;
             * the cache orders and fences that stream but must never destroy it.
             */
            enum class CaptureStreamOwnership : uint8_t
            {
                None = 0, ///< No capture stream has been established.
                Owned,    ///< The cache created and must destroy the stream.
                Borrowed, ///< A longer-lived context owns the bound stream.
            };

            std::vector<GraphSegment> segments;       ///< Ordered segments
            /**
             * Snapshot descriptors and destinations captured by these exact
             * replay units. Cached decode callers cannot omit this owner because
             * it is an intrinsic part of the segment cache lifecycle.
             */
            GraphSnapshotManifest snapshot_manifest;
            bool initialized = false;                 ///< Whether segments have been built
            bool needs_capture = false;               ///< Transient true only within atomic first-use materialization
            ExecutableSubmissionState executable_submission_state =
                ExecutableSubmissionState::Empty;     ///< Whether the sealed executable has ever been submitted.
            uint64_t decode_step = 0;                 ///< Monotonic segmented-execution step counter
            uint64_t successful_capture_count = 0;   ///< Lifetime native materializations completed by this cache.
            uint64_t successful_submission_count = 0; ///< Lifetime inference transactions submitted through captured executables.
            uint64_t capture_variant_signature = 0;   ///< Stage-reported launch-topology variant for this cache
            uint64_t variant_recapture_count = 0;     ///< Resets caused by launch-topology variant changes
            uint64_t snapshot_configuration_epoch = 0; ///< Executor snapshot topology represented by this cache
            std::string perf_context;                 ///< Optional structured stats tag for the replay caller
            ReplayWorkloadGeometry replay_workload;   ///< Exact cache-key geometry for deferred replay metrics
            void *capture_stream = nullptr;           ///< Exact non-null stream for capture/replay
            CaptureStreamOwnership capture_stream_ownership =
                CaptureStreamOwnership::None;         ///< Typed stream lifetime contract
            /** Capture-stream output -> external-consumer handoff/fence. */
            void *sync_event = nullptr;
            /** External producer -> capture-stream handoff event. */
            void *capture_input_event = nullptr;
            void *terminal_event = nullptr;           ///< Dedicated event for one exact deferred host terminal
            uint64_t terminal_fence_published_generation = 0; ///< Latest exact terminal recorded after submission
            uint64_t terminal_fence_observed_generation = 0; ///< Latest exact terminal acquired by its host owner
            IWorkerGPUContext *gpu_ctx_ref = nullptr; ///< GPU context for stream lifecycle (not owned)
            DeviceId capture_device = DeviceId::invalid(); ///< Device used to resolve the stream owner at teardown
            bool capture_context_from_pool = false; ///< True when the stream was created by the pool context
            std::vector<ReplayGpuTimingSlot> replay_gpu_timing_slots; ///< Fixed event ring allocated before capture
            std::string replay_gpu_timing_device_name; ///< Stable PerfStats device label for completed slots
            uint64_t replay_gpu_timing_busy_samples = 0; ///< Replays intentionally not sampled while every slot is in flight
            GraphReplayPlanPolicy graph_replay_plan_policy =
                GraphReplayPlanPolicy::RequireFullGraph; ///< Exact materialization/replay authority represented by this cache.
            SteadyReplayHostPolicy steady_replay_host_policy =
                SteadyReplayHostPolicy::GeneralController; ///< Explicit steady host-launch authority.
            RetainedFullGraphReplayPlan retained_full_graph_replay; ///< Optional prevalidated steady replay plan.
            std::unique_ptr<IGPUGraphCapture> retained_parent_capture; ///< Sole executable for a topology-composed transaction.
            RetainedComposedParentReplayPlan retained_composed_parent_replay; ///< Frozen identity/publication plan for @ref retained_parent_capture.
            const void *auxiliary_branch_authority = nullptr; ///< Stable owner identity embedded in every captured branch node.
            std::unique_ptr<IGraphCaptureAuxiliaryBranch> auxiliary_branch; ///< Cache-private parallel branch and event lifetime.

            GraphSegmentCache() = default;
            ~GraphSegmentCache()
            {
                reset(StreamResetPolicy::Destroy);
            }

            // Move-only (non-copyable due to stream/event ownership)
            GraphSegmentCache(GraphSegmentCache &&other) noexcept
                : segments(std::move(other.segments)),
                  snapshot_manifest(std::move(other.snapshot_manifest)),
                  initialized(other.initialized),
                  needs_capture(other.needs_capture),
                  executable_submission_state(
                      other.executable_submission_state),
                  decode_step(other.decode_step),
                  successful_capture_count(other.successful_capture_count),
                  successful_submission_count(other.successful_submission_count),
                  capture_variant_signature(other.capture_variant_signature),
                  variant_recapture_count(other.variant_recapture_count),
                  snapshot_configuration_epoch(other.snapshot_configuration_epoch),
                  perf_context(std::move(other.perf_context)),
                  replay_workload(other.replay_workload),
                  capture_stream(other.capture_stream),
                  capture_stream_ownership(other.capture_stream_ownership),
                  sync_event(other.sync_event),
                  capture_input_event(other.capture_input_event),
                  terminal_event(other.terminal_event),
                  terminal_fence_published_generation(
                      other.terminal_fence_published_generation),
                  terminal_fence_observed_generation(
                      other.terminal_fence_observed_generation),
                  gpu_ctx_ref(other.gpu_ctx_ref),
                  capture_device(other.capture_device),
                  capture_context_from_pool(other.capture_context_from_pool),
                  replay_gpu_timing_slots(std::move(other.replay_gpu_timing_slots)),
                  replay_gpu_timing_device_name(std::move(other.replay_gpu_timing_device_name)),
                  replay_gpu_timing_busy_samples(other.replay_gpu_timing_busy_samples),
                  graph_replay_plan_policy(other.graph_replay_plan_policy),
                  steady_replay_host_policy(other.steady_replay_host_policy),
                  retained_full_graph_replay(
                      std::move(other.retained_full_graph_replay)),
                  retained_parent_capture(
                      std::move(other.retained_parent_capture)),
                  retained_composed_parent_replay(
                      std::move(other.retained_composed_parent_replay)),
                  auxiliary_branch_authority(
                      other.auxiliary_branch_authority),
                  auxiliary_branch(std::move(other.auxiliary_branch))
            {
                other.capture_stream = nullptr;
                other.capture_stream_ownership = CaptureStreamOwnership::None;
                other.sync_event = nullptr;
                other.capture_input_event = nullptr;
                other.terminal_event = nullptr;
                other.terminal_fence_published_generation = 0;
                other.terminal_fence_observed_generation = 0;
                other.gpu_ctx_ref = nullptr;
                other.capture_device = DeviceId::invalid();
                other.capture_context_from_pool = false;
                other.capture_variant_signature = 0;
                other.successful_capture_count = 0;
                other.successful_submission_count = 0;
                other.executable_submission_state =
                    ExecutableSubmissionState::Empty;
                other.variant_recapture_count = 0;
                other.snapshot_configuration_epoch = 0;
                other.replay_gpu_timing_slots.clear();
                other.replay_gpu_timing_device_name.clear();
                other.replay_gpu_timing_busy_samples = 0;
                other.graph_replay_plan_policy =
                    GraphReplayPlanPolicy::RequireFullGraph;
                other.steady_replay_host_policy =
                    SteadyReplayHostPolicy::GeneralController;
                other.retained_full_graph_replay.clear();
                other.retained_composed_parent_replay.clear();
                other.auxiliary_branch_authority = nullptr;
            }
            GraphSegmentCache &operator=(GraphSegmentCache &&other) noexcept
            {
                if (this != &other)
                {
                    reset(StreamResetPolicy::Destroy);
                    segments = std::move(other.segments);
                    snapshot_manifest = std::move(other.snapshot_manifest);
                    initialized = other.initialized;
                    needs_capture = other.needs_capture;
                    executable_submission_state =
                        other.executable_submission_state;
                    decode_step = other.decode_step;
                    successful_capture_count =
                        other.successful_capture_count;
                    successful_submission_count =
                        other.successful_submission_count;
                    capture_variant_signature = other.capture_variant_signature;
                    variant_recapture_count = other.variant_recapture_count;
                    snapshot_configuration_epoch = other.snapshot_configuration_epoch;
                    perf_context = std::move(other.perf_context);
                    replay_workload = other.replay_workload;
                    capture_stream = other.capture_stream;
                    capture_stream_ownership = other.capture_stream_ownership;
                    sync_event = other.sync_event;
                    capture_input_event = other.capture_input_event;
                    terminal_event = other.terminal_event;
                    terminal_fence_published_generation =
                        other.terminal_fence_published_generation;
                    terminal_fence_observed_generation =
                        other.terminal_fence_observed_generation;
                    gpu_ctx_ref = other.gpu_ctx_ref;
                    capture_device = other.capture_device;
                    capture_context_from_pool = other.capture_context_from_pool;
                    replay_gpu_timing_slots = std::move(other.replay_gpu_timing_slots);
                    replay_gpu_timing_device_name = std::move(other.replay_gpu_timing_device_name);
                    replay_gpu_timing_busy_samples = other.replay_gpu_timing_busy_samples;
                    graph_replay_plan_policy = other.graph_replay_plan_policy;
                    steady_replay_host_policy =
                        other.steady_replay_host_policy;
                    retained_full_graph_replay =
                        std::move(other.retained_full_graph_replay);
                    retained_parent_capture =
                        std::move(other.retained_parent_capture);
                    retained_composed_parent_replay =
                        std::move(other.retained_composed_parent_replay);
                    auxiliary_branch_authority =
                        other.auxiliary_branch_authority;
                    auxiliary_branch = std::move(other.auxiliary_branch);
                    other.capture_stream = nullptr;
                    other.capture_stream_ownership = CaptureStreamOwnership::None;
                    other.sync_event = nullptr;
                    other.capture_input_event = nullptr;
                    other.terminal_event = nullptr;
                    other.terminal_fence_published_generation = 0;
                    other.terminal_fence_observed_generation = 0;
                    other.gpu_ctx_ref = nullptr;
                    other.capture_device = DeviceId::invalid();
                    other.capture_context_from_pool = false;
                    other.capture_variant_signature = 0;
                    other.successful_capture_count = 0;
                    other.successful_submission_count = 0;
                    other.executable_submission_state =
                        ExecutableSubmissionState::Empty;
                    other.variant_recapture_count = 0;
                    other.snapshot_configuration_epoch = 0;
                    other.replay_gpu_timing_slots.clear();
                    other.replay_gpu_timing_device_name.clear();
                    other.replay_gpu_timing_busy_samples = 0;
                    other.graph_replay_plan_policy =
                        GraphReplayPlanPolicy::RequireFullGraph;
                    other.steady_replay_host_policy =
                        SteadyReplayHostPolicy::GeneralController;
                    other.retained_full_graph_replay.clear();
                    other.retained_composed_parent_replay.clear();
                    other.auxiliary_branch_authority = nullptr;
                }
                return *this;
            }
            GraphSegmentCache(const GraphSegmentCache &) = delete;
            GraphSegmentCache &operator=(const GraphSegmentCache &) = delete;

            /**
             * @brief Record one successfully materialized native executable.
             *
             * A capture may be sealed during setup without submitting inference.
             * Keeping materialization and submission counts separate lets the
             * public graph probe distinguish that valid state from transaction
             * zero and from every steady replay.
             *
             * @param submitted_transaction_zero True only when the newly
             *        materialized executable was also launched for inference.
             */
            void recordSuccessfulCapture(
                bool submitted_transaction_zero) noexcept
            {
                ++successful_capture_count;
                if (submitted_transaction_zero)
                    ++successful_submission_count;
            }

            /**
             * @brief Record one successful launch of an existing executable.
             *
             * Call this only after every launch/publication edge for the
             * transaction has succeeded. Failed attempts must never advance the
             * diagnostic proof consumed by benchmark and server readiness gates.
             */
            void recordSuccessfulReplay() noexcept
            {
                ++successful_submission_count;
            }

            /**
             * @brief Adopt snapshot topology identity before the first capture.
             *
             * The execution prelude may establish an exact capture stream and
             * enqueue request-state publication before the executor compares
             * snapshot topology identity.  A brand-new cache has no executable,
             * capture unit, or snapshot allocation to retire, so routing that
             * first identity assignment through @ref reset would publish a host
             * event fence behind those request operations and synchronously wait
             * for it.  Device-owned overlay transactions can deliberately keep
             * that stream pending until transaction zero is launched, making the
             * unnecessary reset a lifecycle deadlock.
             *
             * This operation succeeds only for the complete pristine executable
             * state.  Any evidence of an earlier capture/materialization attempt
             * rejects adoption so the caller must use the ordinary fenced reset
             * before destroying backend resources.
             *
             * @param epoch Non-zero executor-owned snapshot topology generation.
             * @return true when the identity was adopted without touching the
             *         stream; false when retirement is required.
             */
            [[nodiscard]] bool adoptSnapshotConfigurationEpochIfPristine(
                uint64_t epoch) noexcept
            {
                const bool pristine =
                    epoch != 0 && !initialized && !needs_capture &&
                    executable_submission_state ==
                        ExecutableSubmissionState::Empty &&
                    segments.empty() &&
                    snapshot_manifest.stage_copies.empty() &&
                    snapshot_manifest.outputless_stages.empty() &&
                    snapshot_manifest.filtered_stages.empty() &&
                    decode_step == 0 && capture_variant_signature == 0 &&
                    replay_gpu_timing_slots.empty() &&
                    !retained_parent_capture &&
                    !auxiliary_branch &&
                    auxiliary_branch_authority == nullptr &&
                    !retained_full_graph_replay.valid() &&
                    !retained_composed_parent_replay.valid();
                if (!pristine)
                    return false;

                snapshot_configuration_epoch = epoch;
                return true;
            }

            /**
             * @brief Export a strict monolithic template for device-loop composition.
             *
             * Segmentation, incomplete capture, ambiguous stream ownership, or
             * incomplete stage coverage are hard failures. This method never
             * launches eagerly and never recaptures as a substitute.
             *
             * @param graph Complete source graph whose stage lifecycle
             *        contracts must be self-contained at replay time.
             * @param error Optional first violated contract.
             * @return Borrowed replay template when every invariant holds.
             */
            [[nodiscard]] std::optional<DeviceLoopGraphTemplateView>
            deviceLoopGraphTemplate(
                const ComputeGraph &graph,
                std::string *error = nullptr) const;

            /**
             * @brief Export every active unit of a strict retained capture plan.
             *
             * Unlike @ref deviceLoopGraphTemplate, this method admits multiple
             * captured units because a topology-owned parent will insert typed
             * device synchronization between them.  It still rejects every
             * manual/CPU unit, missing executable, ambiguous stream, incomplete
             * graph-stage coverage, and stage requiring host work before replay.
             * Passive capture-wave no-op nodes are validated as represented graph
             * topology but do not produce empty native child graphs.
             *
             * This is a setup-only inspection. It never launches, captures,
             * synchronizes, mutates the graph cache, or selects an eager path.
             *
             * @param graph Exact declarative graph represented by this cache.
             * @param error Optional first violated retained-plan invariant.
             * @return Borrowed active-unit views in graph/capture order.
             */
            [[nodiscard]] std::optional<
                std::vector<RetainedCaptureUnitTemplateView>>
            retainedCaptureUnitTemplates(
                const ComputeGraph &graph,
                std::string *error = nullptr) const;

            /**
             * @brief Export graph-only child templates inside parent transaction zero.
             *
             * This is the capture-phase counterpart to @ref
             * retainedCaptureUnitTemplates. It succeeds only while this cache is
             * materializing @ref RequireRetainedParentComposition, after every
             * child graph has been recorded and before any child has an executable.
             * The returned captures may only be cloned into the cache-owned parent;
             * launching or instantiating them would violate the cache lifecycle.
             *
             * @param graph Exact declarative source graph.
             * @param error Optional first violated lifecycle/topology contract.
             * @return Borrowed graph-only child views in execution order.
             */
            [[nodiscard]] std::optional<
                std::vector<RetainedCaptureUnitTemplateView>>
            retainedCaptureUnitTemplatesForParentComposition(
                const ComputeGraph &graph,
                std::string *error = nullptr) const;

            /**
             * @brief Immutable identity of one exactly published stream terminal.
             *
             * The generation is cache-local and strictly serial. It prevents a
             * host observer from accidentally waiting on an event recorded for
             * an older or newer graph submission.
             */
            struct CaptureStreamTerminalTicket
            {
                uint64_t generation = 0;

                /** @return Whether this ticket names a real publication. */
                [[nodiscard]] constexpr bool valid() const noexcept
                {
                    return generation != 0;
                }
            };

            void reset(StreamResetPolicy stream_policy = StreamResetPolicy::Destroy)
            {
                if (terminal_fence_published_generation >
                    terminal_fence_observed_generation)
                {
                    waitForPublishedCaptureStreamTerminal(
                        CaptureStreamTerminalTicket{
                            terminal_fence_published_generation});
                }
                else
                {
                    waitForCaptureStreamFence();
                }
                // The parent owns clones of every child graph. Destroy it first
                // so no backend child-node lifetime can outlive its source cache.
                retained_parent_capture.reset();
                segments.clear();
                auxiliary_branch.reset();
                auxiliary_branch_authority = nullptr;
                initialized = false;
                needs_capture = false;
                executable_submission_state =
                    ExecutableSubmissionState::Empty;
                decode_step = 0;
                capture_variant_signature = 0;
                terminal_fence_published_generation = 0;
                terminal_fence_observed_generation = 0;
                graph_replay_plan_policy =
                    GraphReplayPlanPolicy::RequireFullGraph;
                retained_full_graph_replay.clear();
                retained_composed_parent_replay.clear();
                destroyReplayGpuTimingEvents();
                destroySyncEvent();
                if (stream_policy == StreamResetPolicy::Destroy)
                {
                    snapshot_manifest.clear();
                    snapshot_configuration_epoch = 0;
                    destroyTerminalEvent();
                    destroyCaptureStream();
                }
            }

            /// Create a local blocking stream for graph capture via the GPU context.
            /// @param ctx GPU context that creates the stream (stored for cleanup)
            bool ensureCaptureStream(
                IWorkerGPUContext *ctx,
                DeviceId device = DeviceId::invalid(),
                bool context_from_process_pool = false);

            /**
             * @brief Bind a longer-lived exact stream for capture and replay.
             *
             * This is used when the graph is one operation in an existing
             * participant stream DAG.  Capturing on the producer stream keeps
             * H2D preparation, graph launch, and output publication ordered
             * without inserting a pair of cross-stream events per invocation.
             * The cache retains no ownership of @p stream and will only fence
             * it before releasing graph/event resources.
             *
             * Rebinding an initialized cache, changing the stream/device, or
             * replacing a cache-owned stream is rejected so capture identity
             * cannot change behind a live executable.
             *
             * @param ctx Live GPU context that owns @p stream.
             * @param stream Exact non-null context-owned stream.
             * @param device GPU device represented by the context.
             * @param context_from_process_pool Whether teardown should
             *        re-resolve @p ctx from the process-wide context pool.
             * @return true when the borrowed binding is installed or already
             *         matches this exact cache; false for an invalid rebind.
             */
            bool bindBorrowedCaptureStream(
                IWorkerGPUContext *ctx,
                void *stream,
                DeviceId device,
                bool context_from_process_pool = false);

            /**
             * @brief Allocate the dedicated exact-terminal event during setup.
             *
             * The event is separate from both directional handoff events
             * because ordinary producer/consumer publication may legally
             * overwrite them. Production callers must invoke this before request
             * admission; @ref publishCaptureStreamTerminal never allocates.
             *
             * @param ctx Worker context owning the bound capture stream.
             * @return True when the event already exists or was created.
             */
            [[nodiscard]] bool prepareCaptureStreamTerminal(
                IWorkerGPUContext *ctx);

            /**
             * @brief Record the exact terminal immediately after graph launch.
             *
             * Invoke this in the same worker-owned submission closure as the
             * retained graph launch. No unrelated stream work can then slip
             * between the executable and its terminal. A second publication is
             * rejected until the first ticket is observed because one event
             * cannot represent two concurrently live generations.
             *
             * @return Non-zero serial ticket naming the recorded terminal.
             */
            [[nodiscard]] CaptureStreamTerminalTicket
            publishCaptureStreamTerminal();

            /**
             * @brief Observe a terminal that was already recorded at launch.
             *
             * This method never records an event. It therefore cannot attach
             * the transaction boundary to newer maintenance or inference work
             * queued after the retained executable.
             *
             * @param ticket Exact ticket returned by the publication call.
             * @param wait_policy Blocking teardown wait or bounded active poll.
             * @param active_timeout_diagnostic Optional read-only timeout state.
             */
            void waitForPublishedCaptureStreamTerminal(
                CaptureStreamTerminalTicket ticket,
                HostFenceWaitPolicy wait_policy =
                    HostFenceWaitPolicy::Blocking,
                std::function<std::string()> active_timeout_diagnostic = {});

            /**
             * @brief Wait on one event representing all prior capture-stream work.
             *
             * This is a host ownership fence, not an ordering primitive between
             * GPU streams. Normal producer/consumer ordering must use
             * orderCaptureStreamAfter() or another device-side event wait. The
             * fence is reserved for terminal host observation and resource
             * teardown, where the host must know that prior stream work has
             * completed before changing stream or graph lifetime. Native graph
             * capture entry must remain stream-ordered and never call it.
             *
             * @param wait_policy Blocking teardown wait or bounded active
             *        progress for a latency-critical heterogeneous boundary.
             * @param active_timeout_diagnostic Optional acquire-safe snapshot
             *        appended only if bounded active progress expires. It must
             *        not mutate graph, stream, or event ownership.
             */
            void waitForCaptureStreamFence(
                HostFenceWaitPolicy wait_policy =
                    HostFenceWaitPolicy::Blocking,
                std::function<std::string()> active_timeout_diagnostic = {});

            /// Destroy the capture stream if it exists
            void destroyCaptureStream();

            /**
             * @brief Prepare the producer-to-capture handoff event.
             *
             * The event retains this direction for its whole lifetime. It is
             * allocated only when an external producer stream exists, so a
             * cache that already executes on its producer stream carries no
             * unused input-edge resource.
             *
             * @param ctx Exact context that owns the producer and capture streams.
             * @return True when the dedicated input event is ready for use.
             */
            bool ensureCaptureInputEvent(IWorkerGPUContext *ctx);

            /**
             * @brief Prepare the capture-to-consumer handoff/fence event.
             *
             * The output-direction event may also serve a terminal host fence
             * during teardown, but it is never re-recorded as an input edge.
             * Keeping the two preparations independent preserves directional
             * ownership without eagerly allocating an event that a topology
             * does not use.
             *
             * @param ctx Exact context that owns the capture and consumer streams.
             * @return True when the dedicated output event is ready for use.
             */
            bool ensureCaptureOutputEvent(IWorkerGPUContext *ctx);

            /**
             * @brief Order the capture stream after work queued on another stream.
             *
             * The method records the input-direction event on `producer_stream` and
             * queues a wait on `capture_stream`. Both operations stay on the
             * device; no host wait or device-wide synchronization is permitted.
             *
             * @param ctx GPU context owning both streams and the event.
             * @param producer_stream Explicit stream whose prior work must finish.
             * @return true when the complete GPU-side dependency was queued.
             */
            bool orderCaptureStreamAfter(
                IWorkerGPUContext *ctx,
                void *producer_stream);

            /**
             * @brief Order an external consumer after the capture stream.
             *
             * This is the reverse half of @ref orderCaptureStreamAfter. It
             * records the output-direction handoff event on the exact stream that
             * launched the captured graph, then queues a wait on the caller's
             * explicit consumer stream. The operation is entirely device-side;
             * it never synchronizes the host or changes graph ownership.
             *
             * @param ctx GPU context that owns both streams and the handoff event.
             * @param consumer_stream Explicit stream that will consume graph output.
             * @return true after the stream wait has been enqueued.
             */
            bool orderStreamAfterCapture(
                IWorkerGPUContext *ctx,
                void *consumer_stream);

            /** Destroy both directional handoff events if they exist. */
            void destroySyncEvent();

            /** Destroy the dedicated exact-terminal event, if prepared. */
            void destroyTerminalEvent();

            /**
             * @brief Destroy every cache-owned replay timing event after its stream fence.
             *
             * reset() calls this only after waitForCaptureStreamFence(), so no
             * recorded event can still be in flight. Steady-state replay must
             * never call this method; event ownership remains fixed from warmup
             * until reset or teardown.
             */
            void destroyReplayGpuTimingEvents();

            /**
             * @brief Resolve the mandatory owner of a live stream or event.
             *
             * Calling this method means a backend resource already exists.
             * Failure is therefore process-fatal: silently dropping the
             * resource would make queued-work and graph lifetime unknowable.
             *
             * @param operation Human-readable lifecycle operation for diagnostics.
             * @return Non-null context that owns the live backend resource.
             */
            IWorkerGPUContext *requireLifecycleContext(const char *operation);
        };

        /**
         * @brief Build one cache-owned parent from ordered graph-only children.
         *
         * The executor creates @p destination on the cache's exact stream and
         * retains its ownership. The callback may only clone @p units and add
         * device-native ordering/control nodes. It must not instantiate, launch,
         * synchronize, allocate replay-time storage, or mutate the source cache.
         * Returning true requires a non-empty, uninstantiated destination graph.
         */
        using RetainedParentCompositionHook = std::function<bool(
            IGPUGraphCapture &destination,
            const ComputeGraph &graph,
            std::span<const GraphSegmentCache::RetainedCaptureUnitTemplateView>
                units)>;

        /**
         * @brief Execute a cached GPU graph replay plan.
         *
         * The planner consumes each stage's graph-capture and collective
         * contracts. Fully capturable homogeneous CUDA/ROCm graphs, including
         * NCCL/RCCL LocalTP collectives, become one captured replay unit.
         * Segmentation is reserved for explicitly admitted heterogeneous
         * collective boundaries; it is not an automatic stage fallback.
         *
         * On first call the method builds and captures the replay plan. On
         * subsequent calls it either launches one full graph, launches the one
         * topology-composed parent, or walks explicitly admitted heterogeneous
         * manual/captured units according to @p plan_policy.
         *
         * @param graph The cached compute graph
         * @param ctx Device context for execution
         * @param segment_cache Persistent segment cache (built once, reused)
         * @param gpu_stream Opaque GPU stream pointer for kernel dispatch
         * @param gpu_ctx GPU context for creating new graph captures
         * @param event_published_outputs Arena outputs that cross an external
         *        host/device boundary after this replay. The executor records
         *        one exact capture-stream event for each named owner; all other
         *        graph-internal writes retain flags-only publication.
         * @param retained_parent_composer Required only by @ref
         *        RequireRetainedParentComposition. It lowers graph-only children
         *        into the one executable owned and launched by this cache.
         * @param initial_submission Selects atomic transaction-zero submission or
         *        setup-only materialization. Setup records and instantiates every
         *        native unit, including an optional retained parent, but neither
         *        launches an executable nor executes manual boundaries, invokes
         *        the launch dependency, or publishes arena writes.
         * @param auxiliary_branch_factory Optional cache-private parallel branch
         *        factory. It is legal only for one complete native graph and
         *        becomes part of immutable graph-cache identity.
         * @return true on success
         */
        bool executeWithCachedGraphReplay(ComputeGraph &graph, IDeviceContext *ctx,
                                              GraphSegmentCache &segment_cache,
                                              void *gpu_stream,
                                              IWorkerGPUContext *gpu_ctx,
                                              const std::unordered_set<std::string> *collective_nodes = nullptr,
                                              bool collectives_graph_capturable = false,
                                              bool force_recapture = false,
                                              bool defer_final_sync = false,
                                              GraphCaptureBoundaryHook capture_boundary = {},
                                              GraphReplayPlanPolicy plan_policy =
                                                  GraphReplayPlanPolicy::RequireFullGraph,
                                              GraphLaunchDependencyHook launch_dependency = {},
                                              std::span<const BufferId>
                                                  event_published_outputs = {},
                                              RetainedParentCompositionHook
                                                  retained_parent_composer = {},
                                              GraphInitialSubmissionPolicy
                                                  initial_submission =
                                                      GraphInitialSubmissionPolicy::
                                                          CaptureInstantiateAndLaunch,
                                              GraphCaptureAuxiliaryBranchFactory
                                                  auxiliary_branch_factory = {});

        /**
         * @brief Policy object for decode capture/replay execution mode selection
         */
        struct DecodeCapturePolicy
        {
            bool allow_fast_decode = true;
            bool allow_cached_graph_replay = false;
            bool heterogeneous_segmented_enabled = false;
            bool collectives_graph_capturable = false; ///< True when LocalTP NCCL/RCCL collectives are captured in the replay graph
            GraphReplayPlanPolicy graph_replay_plan_policy =
                GraphReplayPlanPolicy::RequireFullGraph; ///< Full graph unless a proven heterogeneous collective domain explicitly admits segmentation.
            bool force_recapture = false;              ///< Re-record graph segments on replay for callers with dynamic params not yet replay-safe
            bool defer_final_sync = false;             ///< Caller will synchronize the replay stream through a following operation.
            /**
             * Optional domain-level capture lifecycle rendezvous.
             *
             * Capture invokes this hook before beginCapture() and immediately
             * after endCapture(). LocalTP participants install one shared cyclic
             * barrier so every device enters and exits native capture as a
             * domain. The exit rendezvous remains required for a single full
             * graph because sibling graph instances are recorded independently.
             */
            GraphCaptureBoundaryHook capture_boundary;
            /**
             * External event join performed immediately before every captured
             * executable launch.  Transaction zero runs it only after capture
             * and instantiation, so the dependency cannot become a root event
             * in an exported child graph.
             */
            GraphLaunchDependencyHook launch_dependency;
            /**
             * Topology-specific device parent lowerer. Mandatory exactly when
             * graph_replay_plan_policy is RequireRetainedParentComposition.
             */
            RetainedParentCompositionHook retained_parent_composer;
            /**
             * Optional bounded work captured as a parallel root-to-terminal
             * branch of this exact production executable. The graph cache owns
             * the constructed branch and validates its authority on replay.
             */
            GraphCaptureAuxiliaryBranchFactory auxiliary_branch_factory;
        };

        /**
         * @brief Execute decode graph according to a single policy object
         *
         * Centralizes the configuration-time selection between cached GPU graph
         * replay, fast decode, and the general executor. Once cached replay is
         * selected, any capture or launch failure is returned as a hard
         * execution failure; this method never changes paths after an attempt.
         */
        bool executeDecodeWithCapturePolicy(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            GraphSegmentCache *segment_cache,
            void *gpu_stream,
            IWorkerGPUContext *gpu_ctx,
            const std::unordered_set<std::string> *collective_nodes,
            const DecodeCapturePolicy &policy,
            bool *used_graph_replay = nullptr,
            GraphInitialSubmissionPolicy initial_submission =
                GraphInitialSubmissionPolicy::CaptureInstantiateAndLaunch);

    private:
        /** @brief Lifecycle intended for one native graph recording pass. */
        enum class GraphCaptureRecordPurpose : uint8_t
        {
            ImmediateReplay,     ///< Instantiate and launch immediately afterward.
            RetainedComposition, ///< Import later into one parent transaction.
        };

        /**
         * @brief Record the shared native graph body without publishing it.
         *
         * This is the sole direct-capture implementation used by immediate
         * replay and retained child composition. Keeping the capture interval in
         * one method prevents the two lifecycles from drifting on arena
         * prebinding, launch metadata, dependency-ledger ordering, or failure
         * cleanup.
         */
        bool recordGraphCaptureBody(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            IGPUGraphCapture *capture,
            GraphCaptureRecordPurpose purpose,
            const char *context);

        GraphExecutorConfig config_;
        GraphExecutorStats stats_;
        ICollectiveContext *collective_ctx_ = nullptr; ///< Optional collective context (not owned)
        BufferArena *arena_ = nullptr;                 ///< Optional arena for contract coherence (not owned)
        StageTimeline stage_timeline_;                 ///< GPU event-based per-stage timeline profiler
        bool stage_timeline_info_populated_ = false;   ///< True after first setStageInfo pass (names never change)
        bool weights_session_cohered_ = false;         ///< True after first forward completes weight coherence for all nodes
        uint64_t snapshot_configuration_epoch_ = 1;   ///< Monotonic snapshot graph-topology identity

        /**
         * @brief Allocate one stage's arena bindings without changing authority.
         *
         * Reads are included because launch descriptors may embed an external
         * input pointer before its payload is uploaded. Writes requiring normal
         * executor preparation are included for the same reason. The caller's
         * identity set deduplicates aliases across the complete capture unit.
         *
         * @param node Stage whose declarative arena bindings are inspected.
         * @param capture_device GPU that will own the capture transaction.
         * @param allocated Tensor identities already allocated in this unit.
         * @param context Optional diagnostic label.
         * @return true when every relevant binding has stable device storage.
         */
        bool allocateStageArenaStorageForCapture(
            ComputeNode &node,
            DeviceId capture_device,
            std::unordered_set<ITensor *> &allocated,
            const char *context);

        /**
         * @brief Prepare one stage's arena frontier before native capture.
         *
         * StageBufferContract is the sole authority: external reads must own
         * valid device bytes and every overwrite-only output must own stable
         * device storage. CoherencePolicy cannot suppress either requirement.
         * The deduplication sets span one capture unit so aliases and repeated
         * layer scratch bindings do not repeat event joins or allocation checks.
         *
         * @param node Stage whose declarative contract is being prepared.
         * @param external_reads Reads produced outside the current capture unit.
         * @param capture_device Device that owns the native capture transaction.
         * @param capture_stream Exact non-null stream used by beginCapture().
         * @param prepared_inputs Tensor identities already joined in this unit.
         * @param prepared_outputs Tensor identities already allocated in this unit.
         * @param input_policy Whether external request bytes must already be
         *        valid or setup is binding addresses without launching them.
         * @param context Optional diagnostic label for the owning capture path.
         * @return true when the stage frontier is capture-ready.
         */
        bool prepareStageArenaFrontierForCapture(
            ComputeNode &node,
            const std::vector<BufferBinding> &external_reads,
            DeviceId capture_device,
            void *capture_stream,
            std::unordered_set<ITensor *> &prepared_inputs,
            std::unordered_set<ITensor *> &prepared_outputs,
            GraphCaptureDependencyLedger::ExternalInputAuthority
                external_input_authority,
            const char *context);

        void advanceSnapshotConfigurationEpoch() noexcept
        {
            ++snapshot_configuration_epoch_;
            if (snapshot_configuration_epoch_ == 0)
                snapshot_configuration_epoch_ = 1;
        }

        // =====================================================================
        // Unified stage runner (replaces divergent paths)
        // =====================================================================

        /**
         * @brief Execute all stages in a graph according to the given policy.
         *
         * This is the ONE execution loop used by every entry point:
         *   - execute()/executeSequential()  → full policy
         *   - executeFastDecode()             → fastDecode policy
         *   - graph capture non-captured segs → full policy
         *
         * @param graph            Compute graph with stages to run
         * @param ctx              Device context for execution
         * @param policy           Controls coherence, validation, profiling etc.
         * @param collective_nodes Pre-identified collective stage names (optional)
         * @return true if all stages executed successfully
         */
        bool runStages(ComputeGraph &graph,
                       IDeviceContext *ctx,
                       const StageRunPolicy &policy,
                       const std::unordered_set<std::string> *collective_nodes = nullptr,
                       GraphSnapshotManifest *snapshot_manifest = nullptr);

        bool cancellationRequested(const std::string &node_name) const;
        void notifyStageFailure(const std::string &node_name, const std::string &reason) const;

        /**
         * @brief Validate every device-owned prepared weight declared by a stage.
         *
         * Prepared weight identity is the tuple
         * `{model, binding, kind, device}` held by PreparedWeightRef. Validation
         * happens before raw tensor coherence so a host-only source tensor can
         * never be mistaken for the allocation consumed by a GPU kernel.
         */
        bool validatePreparedWeightBindings(
            const ComputeNode &node,
            const StageBufferContract &contract,
            DeviceId target_device) const;

        /**
         * @brief Execute a single stage according to the given policy.
         *
         * Implements every per-stage phase (coherence, execution, marking,
         * validation, profiling, dumps) gated by the policy flags.
         *
         * @param node          The compute node to execute
         * @param ctx           Device context
         * @param policy        Controls which phases are active
         * @param is_collective Pre-computed flag: true if stage is collective
         * @return true if stage executed successfully
         */
        bool runStage(ComputeNode &node,
                      IDeviceContext *ctx,
                      const StageRunPolicy &policy,
                      bool is_collective,
                      GraphSnapshotManifest *snapshot_manifest = nullptr);

        /**
         * @brief Execute the stage body after capture-order RAII has been armed.
         *
         * Keeping this implementation private ensures every caller enters through
         * runStage(), which advances GraphCaptureDependencyLedger when native
         * capture is active.  No graph path may call this implementation directly.
         */
        bool runStageImpl(ComputeNode &node,
                          IDeviceContext *ctx,
                          const StageRunPolicy &policy,
                          bool is_collective,
                          GraphSnapshotManifest *snapshot_manifest);

        bool prepareOrRecordGraphSnapshotCopies(ComputeNode &node,
                                                DeviceId target_device,
                                                void *producer_stream,
                                                bool record_device_copy,
                                                GraphSnapshotManifest &snapshot_manifest);
        /**
         * @brief Evaluate the configured filter against one concrete publication.
         *
         * @param node_name Graph-local producer name.
         * @param dump_info Stable output descriptor exposed by that producer.
         * @return True when this exact publication must be captured.
         */
        bool shouldCaptureSnapshotStage(const std::string &node_name,
                                        const StageDumpInfo &dump_info) const;

        /**
         * @brief Prepare immutable graph-stable snapshot descriptors/storage.
         *
         * GPU graph capture cannot discover or allocate snapshot outputs while
         * capture is active. The stage must therefore expose its stable output
         * descriptor after launch preparation and before model arithmetic.
         * This method allocates the graph-owned destination and freezes the
         * exact source descriptor without copying payload bytes. The captured
         * point-in-time copy is recorded only after the producer executes.
         */
        bool prepareGraphSnapshotCopies(ComputeNode &node,
                                        DeviceId target_device,
                                        void *producer_stream,
                                        GraphSnapshotManifest &snapshot_manifest);

        /**
         * @brief Enqueue point-in-time device copies for graph-captured snapshots.
         *
         * Fast decode, segmented graph capture, and prefill graph capture disable
         * host callbacks while device work is being recorded/launched. Draining
         * live stage outputs after the full graph finishes is incorrect when the
         * arena reuses an activation slot later in the graph. This helper records
         * a device-to-device copy immediately after the producing stage into
         * a unique device-visible mapped-host slot. Graph replay therefore
         * preserves the stage-boundary value without re-entering eager
         * execution or reserving a second activation graph in device memory.
         */
        bool captureGraphSnapshotCopies(ComputeNode &node,
                                        DeviceId target_device,
                                        void *producer_stream,
                                        GraphSnapshotManifest &snapshot_manifest);

        /**
         * @brief Publish one GPU stage from its immutable graph snapshot slots.
         *
         * Graph replay updates a mapped slot through its device-visible pointer
         * without executing stage C++. This method records a real post-launch
         * completion event, waits only for that publication point, builds a
         * callback descriptor solely from the frozen slot manifest, and invokes
         * the configured snapshot callback over the host-visible mapping.
         * It deliberately never calls getDumpInfo() on the live stage.
         */
        bool publishGraphSnapshotCopies(const std::string &stage_name,
                                        void *producer_stream,
                                        GraphSnapshotManifest &snapshot_manifest,
                                        const GraphSnapshotLogicalRows *logical_rows);

        // =====================================================================
        // Legacy internal helpers (now delegate to runStages/runStage)
        // =====================================================================
        bool executeSequential(ComputeGraph &graph, IDeviceContext *ctx);
        // executeParallel removed — PARALLEL mode falls through to executeSequential
        bool executeNode(ComputeNode &node, IDeviceContext *ctx);

        // Collective stage intercept helpers (GPU-native collectives)
        bool executeCollectiveAllreduce(ComputeNode &node, IDeviceContext *ctx);
        bool executeCollectiveAllgather(ComputeNode &node, IDeviceContext *ctx);

        /**
         * @brief Execute strided AllGather using NCCL + CUDA deinterleave
         *
         * Optimized for column-parallel LM head. Returns false if NCCL not
         * available or device is not CUDA, allowing fallback to MPI path.
         *
         * @param node The compute node containing AllGatherStage
         * @param ctx Device context (unused, CollectiveContext handles device)
         * @return true if executed successfully, false to fall back to stage execution
         */
        bool executeCollectiveStridedAllgather(ComputeNode &node, IDeviceContext *ctx);

        // Buffer validation (Debug/Integration builds only)
#if LLAMINAR_ASSERTIONS_ACTIVE
        // Stage verification delegated to free functions in StageVerifier.h
#endif

        // Workspace management
        std::vector<float> temp_buffer_;
        size_t temp_buffer_size_ = 0;
        /**
         * Snapshot owner for uncached eager diagnostics only.
         *
         * Every cached forward graph passes its ForwardGraphCache-owned manifest
         * explicitly. This transient state is never a cached executable owner.
         */
        GraphSnapshotManifest transient_snapshot_manifest_;

        float *getTemporaryBuffer(size_t elements);
    };

    // Backwards compatibility alias
    using LayerExecutor = DeviceGraphExecutor;

} // namespace llaminar2
