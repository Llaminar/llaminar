/**
 * @file ForwardExecutionEngine.h
 * @brief Execution engine for forward graph dispatch and caching
 *
 * Extracted from DeviceGraphOrchestrator (Phase 3 of DGO refactor).
 *
 * This engine owns the forward graph cache and handles:
 * - Cache signature computation and lookup
 * - Cache HIT path: buffer updates, PP copy, dynamic params, GPU graph replay
 * - Cache MISS path: graph build via host callback, execution, cache population
 * - GPU stage timeline collection and printing
 *
 * The engine delegates model-specific operations (graph building, device context
 * management, logits sync) to an IForwardExecutionHost interface implemented by
 * the orchestrator.
 */

#pragma once

#include "ForwardGraphTypes.h"
#include "PrefillBucketUtils.h"
#include "../graph/DeviceGraphExecutor.h"
#include "../device/DeviceContext.h"
#include "../device/WorkspaceAllocator.h"
#include "../../factory/InferenceRunnerFactory.h" // For FactoryPPStageConfig
#include "../../../utils/ForwardPassProfiler.h"

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class IWorkerGPUContext;

    /**
     * @brief Request-scoped authority over one scheduled prefill graph submission.
     *
     * Heterogeneous ExpertOverlay execution has an explicit host-visible ticket
     * boundary even though model state and sparse payloads remain device-owned.
     * The authority that opens that ticket must therefore remain alive from
     * chunk admission through the exact executable launch and local submission
     * result.  This lease gives ForwardExecutionEngine one typed lifetime for
     * that boundary without teaching the engine about any particular MoE
     * transport or topology.
     *
     * A host without an external graph authority returns no lease.  A host that
     * returns a lease must finish it exactly once after execute() returns; its
     * destructor is responsible for failing an abandoned transaction during an
     * exception or early exit.
     */
    class IPrefillChunkGraphSubmissionLease
    {
    public:
        virtual ~IPrefillChunkGraphSubmissionLease() = default;

        /**
         * @brief Publish the local submission result and close this lease.
         *
         * @param execution_succeeded Whether the production graph was submitted
         *        successfully on its owned execution stream.
         * @param error Receives a precise lifecycle diagnostic on failure.
         * @return True when the external authority accepted the terminal state.
         */
        virtual bool finish(
            bool execution_succeeded,
            std::string *error) = 0;
    };

    /**
     * @brief Host interface for ForwardExecutionEngine callbacks
     *
     * The engine delegates model-specific and device-management operations
     * to the host (typically DeviceGraphOrchestrator). This interface defines
     * the minimal set of operations the engine cannot perform itself.
     *
     * Design rationale: virtual interface for clean abstraction boundary
     * and testability, even with a single production implementor.
     */
    class IForwardExecutionHost
    {
    public:
        virtual ~IForwardExecutionHost() = default;

        // ----- Graph Building (cache MISS path) -----

        /**
         * @brief Build a forward graph for the given input.
         *
         * The host dispatches internally to the appropriate builder
         * (standard / partial PP / unified PP).
         */
        virtual GraphBuildResult buildForwardGraph(const ForwardInput &input) = 0;

        // ----- Device Context -----

        /** Get or create a device context for the given device. */
        virtual IDeviceContext *getDeviceContext(DeviceId device) = 0;

        /**
         * @brief Resolve the worker-owned GPU runtime context for a device.
         *
         * ForwardExecutionEngine must not reach through a global backend pool:
         * the host owns device lifecycle and is the only component that can
         * distinguish a production physical context from a hardware-free test
         * double. GPU execution requires a non-null result. CPU devices return
         * null because they have no stream, event, or graph-capture runtime.
         *
         * @param device Logical GPU device whose worker owns execution.
         * @return Host-owned worker context, or null for a non-GPU device.
         */
        virtual IWorkerGPUContext *getWorkerGPUContext(DeviceId device) = 0;

        /**
         * @brief Report whether worker cleanup may re-resolve through the process pool.
         *
         * Production DGO and physical GPU integration hosts return true. Unit
         * hosts return false so cached stream/event cleanup continues to use
         * the process-lifetime in-memory worker without probing CUDA or ROCm.
         */
        virtual bool workerGPUContextUsesProcessPool(DeviceId device) const = 0;

        /** Get device contexts for all devices in a unified PP pipeline. */
        virtual std::unordered_map<DeviceId, IDeviceContext *> getPipelineDeviceContexts() = 0;

        // ----- Workspace -----

        /** Ensure GPU workspace is allocated for GEMM kernels in the graph. */
        virtual bool ensureDeviceWorkspaceAllocated(
            const ComputeGraph &graph,
            int workspace_seq_len = 0) = 0;

        /**
         * @brief Ensure workspace under an explicit graph-family lifetime policy.
         *
         * Production GPU hosts override this typed form. The compatibility
         * implementation preserves lightweight test hosts while making the
         * execution engine's serial-family intent visible at the API boundary.
         */
        virtual bool ensureDeviceWorkspaceAllocated(
            const ComputeGraph &graph,
            int workspace_seq_len,
            WorkspaceGraphFamilyPolicy graph_family_policy)
        {
            (void)graph_family_policy;
            return ensureDeviceWorkspaceAllocated(graph, workspace_seq_len);
        }

        /**
         * @brief Ensure workspace for one explicitly typed graph-family participant.
         *
         * Graph geometry is not a semantic discriminator. A short prompt and a
         * grouped verifier can have the same M while selecting different kernels,
         * recurrent-state banks, and LM-head row policies. Production hosts must
         * therefore receive the role selected from ForwardInput rather than
         * reconstructing it from sequence length or decode-history heuristics.
         *
         * Lightweight test hosts may retain the three-argument implementation;
         * this compatibility body deliberately discards only the additional role,
         * not the graph-family lifetime policy.
         *
         * @param graph Materialized graph whose consumers require binding.
         * @param workspace_seq_len Participant rows per request.
         * @param graph_family_policy Physical lifetime policy for the family.
         * @param participant_role Mathematical role of this exact graph.
         * @return true when all consumers are bound to stable workspace addresses.
         */
        virtual bool ensureDeviceWorkspaceAllocated(
            const ComputeGraph &graph,
            int workspace_seq_len,
            WorkspaceGraphFamilyPolicy graph_family_policy,
            WorkspaceGraphParticipantRole participant_role)
        {
            (void)participant_role;
            return ensureDeviceWorkspaceAllocated(
                graph,
                workspace_seq_len,
                graph_family_policy);
        }

        /**
         * @brief Return the workspace generation for a device, if the host tracks it.
         *
         * A generation change means raw workspace addresses may have changed.
         * Cached graphs can still be reused, but captured GPU graph replay state
         * must be discarded and stages must be rebound before execution.
         */
        virtual uint64_t workspaceGeneration(DeviceId device) const
        {
            (void)device;
            return 0;
        }

        /**
         * @brief Return the host's memory-planned resident graph-row capacity.
         *
         * A zero result means the lightweight host has no explicit bound.
         * Production GPU hosts return a positive value selected before graph
         * construction.
         */
        virtual int residentGraphRows() const
        {
            return 0;
        }

        /**
         * @brief Called once after the first graph build completes and workspace
         *        is allocated, but before execution starts.
         *
         * Use this to release transient resources that are only needed during
         * graph construction (e.g., mmap pages via madvise(MADV_DONTNEED)).
         */
        virtual void onFirstGraphReady() {}

        /**
         * @brief Optional rendezvous immediately before graph execution.
         *
         * LocalTP first forwards can materialize graph/workspace state at
         * different speeds. Hosts may use this hook to let every participant
         * finish production graph preparation before any one participant enters
         * the first collective.
         */
        virtual bool waitBeforeForwardGraphExecution(
            const ForwardInput &input,
            DeviceId execution_device,
            bool cache_miss)
        {
            (void)input;
            (void)execution_device;
            (void)cache_miss;
            return true;
        }

        /**
         * @brief Consume graph-build device-state readiness before setup capture.
         *
         * Setup-owned graph families capture and instantiate an executable without
         * launching it. They therefore do not enter the ordinary live-state
         * prelude, but graph-builder uploads still need one exact-stream consumer
         * before another family member may reuse the publication event. The host
         * queues that event wait on @p capture_stream and retires only the
         * graph-build publication; request admission, reset, and mutable inference
         * state remain untouched.
         *
         * Lightweight hosts have no asynchronous graph-build publication and may
         * retain the default no-op implementation.
         *
         * @param input Setup-only forward invocation being materialized.
         * @param capture_stream Exact non-null stream that will record the graph.
         * @param execution_device GPU that owns @p capture_stream.
         * @return true when no publication was pending or its wait was enqueued.
         */
        virtual bool prepareGraphBuildStateForMaterialization(
            const ForwardInput &input,
            void *capture_stream,
            DeviceId execution_device)
        {
            (void)input;
            (void)capture_stream;
            (void)execution_device;
            return true;
        }

        // ----- Forward-Result Publication -----

        /**
         * @brief Publish the graph-declared result tensor on its producer stream.
         *
         * The graph builder is the sole authority for the concrete result:
         * terminal stages publish `ForwardOutput::logits`, while an explicitly
         * configured non-head pipeline stage publishes
         * `ForwardOutput::hidden`. GPU implementations record a completion event
         * on the exact stream in `ForwardOutput::execution`. They must not
         * reselect storage from mutable runtime mode flags, synchronize the
         * stream/device, or materialize the result on the host. An explicit host
         * result accessor performs the eventual D2H transfer when requested.
         *
         * @param output Graph-declared tensors and exact execution provenance.
         * @param ctx Context for the device that owns the published result.
         * @return true when device ownership and completion are published.
         */
        virtual bool publishForwardResultAtBoundary(
            const ForwardOutput &output,
            IDeviceContext *ctx) = 0;

        /**
         * @brief Commit one successful engine invocation to its semantic owner.
         *
         * `ForwardExecutionEngine::execute()` invokes this exactly once before
         * returning success. The host must publish the exact output tensor and
         * producer event, and may close any typed transaction armed for the
         * invocation. Making this an engine-owned callback prevents alternate
         * callers such as chunk-scheduled prefill from bypassing publication.
         *
         * @param output Successful output carrying terminal tensor identity and
         *        exact producer provenance.
         */
        virtual void commitSuccessfulForwardOutput(
            const ForwardOutput &output) = 0;

        // ----- Decode Capture Policy -----

        /** Build the GPU graph capture/replay policy for decode steps. */
        virtual DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool has_collective_nodes,
            IDeviceContext *ctx) const = 0;

        /**
         * @brief Describe bounded maintenance captured with this forward graph.
         *
         * The engine asks only for real model forward executables, never for
         * tiny publication, sampler, or maintenance graphs. An empty descriptor
         * means no branch is configured. A non-empty descriptor is cached as
         * part of the executable's immutable identity and must continue naming
         * the same authority on every replay, including hosted MTP replay.
         * Authority belongs to the runner/device lifetime, not to a transient
         * ForwardInput or a caller-reconstructed speculative geometry.
         *
         * @param execution_device GPU that will own the native executable.
         * @return Empty or complete cache-private branch factory.
         */
        virtual GraphCaptureAuxiliaryBranchFactory
        forwardGraphAuxiliaryBranchFactory(
            DeviceId execution_device)
        {
            (void)execution_device;
            return {};
        }

        // ----- PP Copy Info -----

        /**
         * @brief Information for copying external hidden state to working buffer.
         *
         * For PP non-embedding stages, the previous stage's output (external hidden)
         * must be copied to the local working buffer before graph execution. During
         * cache MISS this happens inline in graph build; during cache HIT the engine
         * must redo this copy using the stored PPCopyInfo.
         */
        struct PPCopyInfo
        {
            TensorBase *external_hidden = nullptr; ///< Source (stage N-1 output)
            TensorBase *working_buffer = nullptr;  ///< Destination (local residual/hidden)
            size_t copy_bytes = 0;
            DeviceId device;
            bool needs_copy = false;
        };

        /** Resolve PP copy info for the given input (after a cache-miss build). */
        virtual PPCopyInfo resolvePPCopyInfo(const ForwardInput &input) const = 0;

        /** Check if MoE dynamic rebalancing is active for this forward domain. */
        virtual bool isMoeRebalancingActive() const { return false; }

        /**
         * @brief True when active MoE placement changes are graph-stable.
         *
         * Homogeneous GPU LocalTP expert-ID apportionment can publish ownership and replica changes by
         * mutating persistent runtime tables and transfer slots in place. Captured
         * prefill graphs may then replay against padded buckets because the graph
         * records stable table pointers, not placement topology. Hosts that cannot
         * prove this must keep the default false value so padded graph capture
         * fails fast instead of silently executing an unsafe eager path.
         */
        virtual bool isMoeRebalancingGraphStableForPrefillCapture() const { return false; }

        /** Check if the host execution mode should run prefill eagerly instead of graph-capturing it. */
        virtual bool prefillGraphCaptureDisabledByHost() const { return false; }

        /**
         * @brief Wait at a named prefill GPU graph-capture lifecycle boundary.
         *
         * Multi-device LocalTP capture must be coordinated at request/chunk
         * boundaries: one participant starting capture while another is still
         * draining the previous eager chunk can poison HIP/CUDA capture state and
         * make the next grouped collective fail. Hosts that own a LocalTP domain
         * should route this hook to that domain's capture-boundary rendezvous.
         * Single-device hosts can keep the default no-op implementation.
         *
         * @param input Forward input for the prefill chunk being captured.
         * @param execution_device Device entering the boundary.
         * @param boundary_name Stable boundary identifier for diagnostics.
         * @param capture_stream Exact stream entering the lifecycle transition.
         * @return true when the boundary is safe to cross.
         */
        virtual bool waitAtPrefillGraphCaptureBoundary(
            const ForwardInput &input,
            DeviceId execution_device,
            const std::string &boundary_name,
            void *capture_stream)
        {
            (void)input;
            (void)execution_device;
            (void)boundary_name;
            if (!capture_stream)
                throw std::invalid_argument(
                    "IForwardExecutionHost::waitAtPrefillGraphCaptureBoundary requires a non-null GPU stream");
            return true;
        }

        /**
         * @brief Wait at a named decode GPU graph-capture lifecycle boundary.
         *
         * Segmented decode capture has the same multi-device lifecycle hazard as
         * prefill capture: no participant may enter HIP/CUDA beginCapture()
         * while a sibling is still preparing or draining its capture stream.
         * Hosts that own a LocalTP domain should route this to that domain's
         * graph-capture boundary rendezvous.
         */
        virtual bool waitAtDecodeGraphCaptureBoundary(
            const ForwardInput &input,
            DeviceId execution_device,
            const std::string &boundary_name,
            void *capture_stream)
        {
            (void)input;
            (void)execution_device;
            (void)boundary_name;
            if (!capture_stream)
                throw std::invalid_argument(
                    "IForwardExecutionHost::waitAtDecodeGraphCaptureBoundary requires a non-null GPU stream");
            return true;
        }

        /** Whether the current forward graph must materialize logits for every input row. */
        virtual bool computeAllPositionLogitsEnabled() const { return false; }

        /**
         * @brief Whether this forward is the live grouped MTP condition batch.
         *
         * The shape is `batch_size > 1, seq_len == 1`, which would otherwise be
         * classified as prompt prefill. The explicit signal routes it through
         * decode graph capture while preserving live recurrent-state ownership.
         */
        virtual bool liveMTPRequestBatchConditionEnabled() const { return false; }

        /** Compact verifier row count when all-position logits are row-indexed; 0 means full input rows. */
        virtual int allPositionLogitRows() const { return 0; }

        /**
         * @brief Terminal MTP outcome topology owned by the current graph.
         */
        virtual MTPVerifierOutcomeGraphMode
        mtpVerifierOutcomeGraphMode() const
        {
            return MTPVerifierOutcomeGraphMode::Disabled;
        }

        /**
         * @brief True while a vLLM-style MTP verifier row plan is installed.
         *
         * Batched verifier forwards have `batch_size > 1` and `seq_len > 1`,
         * so their shape alone looks like prefill. This explicit host signal
         * lets the engine route only those planned verifier forwards through
         * the decode graph cache and keeps ordinary batched prompt execution on
         * the prefill path.
         */
        virtual bool mtpSpecVerifierInputPlanActive() const { return false; }

        /**
         * @brief Prepare per-step metadata consumed by an all-position verifier graph.
         *
         * The forward engine calls this after workspace binding and after it has
         * chosen the stream that the graph will use for eager execution, warmup,
         * capture, or replay. Hosts can upload small graph-facing metadata here
         * without the engine knowing feature-specific details such as MTP row
         * selection. GPU hosts must treat `execution_stream` as mandatory when
         * `execution_device` is a GPU.
         */
        virtual bool prepareAllPositionVerifierGraphMetadata(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device)
        {
            (void)input;
            (void)execution_stream;
            (void)execution_device;
            return true;
        }

        /**
         * @brief Order a forward graph after any pending live-state publication.
         *
         * MTP accepted-state publication can update KV, recurrent, and short-conv
         * state asynchronously on the verifier stream.  Before the next forward
         * reads that live state, the host gets the exact stream chosen for eager
         * execution, graph warmup, capture, or replay and may queue a GPU-side
         * wait.  This keeps publication atomic without forcing a CPU stream sync.
         */
        virtual bool prepareLiveStateForForwardGraphExecution(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device)
        {
            (void)input;
            (void)execution_stream;
            (void)execution_device;
            return true;
        }

        /**
         * @brief Publish completion of the forward graph's live-state reads.
         *
         * `prepareLiveStateForForwardGraphExecution()` may admit a GPU graph as
         * an asynchronous reader of persistent live-state rows. The execution
         * engine calls this matching hook immediately after the graph launch,
         * on both success and failure, so the host can append the graph's exact
         * producer stream to any reusable-row access fence. Implementations
         * must not synchronize the stream or materialize device state on the
         * host; this is an event-only producer/consumer handoff.
         *
         * @param input Forward invocation whose prelude admitted the read.
         * @param execution_stream Exact stream on which the graph work ended.
         * @param execution_device Device that owns @p execution_stream.
         * @return True when no read was armed or its completion was published.
         */
        virtual bool completeLiveStateForForwardGraphExecution(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device)
        {
            (void)input;
            (void)execution_stream;
            (void)execution_device;
            return true;
        }

        /**
         * @brief Publish caller-staged device token rows on the graph stream.
         *
         * Device-token composition is a forward-input concern, not verifier
         * metadata.  The engine invokes this hook for every GPU forward after
         * joining live-state publication and before any dynamic graph parameter
         * reads the row.  Implementations must enqueue copies on
         * `execution_stream` and publish the resulting arena write; they must
         * not synchronize the stream or defer this work to an all-position-only
         * hook.
         */
        virtual bool prepareDeviceTokenInputsForForwardGraphExecution(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device)
        {
            (void)input;
            (void)execution_stream;
            (void)execution_device;
            return true;
        }

        /**
         * @brief Monotonic live-state epoch for decode graph replay safety.
         *
         * Speculative publication can jump KV/GDN/short-conv state by multiple
         * tokens. Ordinary one-token decode captures are only safe to replay
         * when they were captured after the current live-state epoch was
         * established, unless a future state-sync hook explicitly stamps them.
         */
        virtual uint64_t liveReplayStateEpoch() const { return 0; }

        /**
         * @brief Whether the next all-position verifier replay may defer its final sync.
         *
         * The host is responsible for allowing this only when the immediate
         * caller will consume all-position logits through a device operation on
         * the same replay stream. The default keeps the normal synchronized
         * forward boundary.
         */
        virtual bool shouldDeferAllPositionVerifierFinalSync() const { return false; }

        /**
         * @brief Receive the stream whose all-position verifier logits are pending.
         *
         * Called after a successful deferred replay. The host should enqueue the
         * next logits-consuming operation on this stream, or clear the pending
         * stream by passing nullptr when deferral is not active.
         */
        virtual void setPendingAllPositionVerifierStream(void *stream)
        {
            (void)stream;
        }

        /**
         * @brief Whether the next one-row main decode may defer its final sync.
         *
         * MTP condition-forward decode immediately consumes main logits with a
         * GPU sampler or GPU distribution-builder.  Deferring the replay sync
         * lets that consumer enqueue on the same stream instead of forcing the
         * CPU to wait between model forward and sampling.  Hosts should return
         * true only for one-shot decode paths where a device consumer is
         * guaranteed to follow.
         */
        virtual bool shouldDeferMainDecodeFinalSync() const { return false; }

        /**
         * @brief Receive the stream whose main-decode logits are pending.
         *
         * Called after a successful deferred main-decode replay.  The host must
         * enqueue the next logits-consuming device operation on this stream, or
         * clear the pending stream when nullptr is passed.
         */
        virtual void setPendingMainDecodeStream(void *stream)
        {
            (void)stream;
        }

        /** Domain placement epoch for MoE-sensitive prefill graph-cache keys. */
        virtual uint64_t moePlacementEpoch() const { return 0; }

        /** Domain id for MoE-sensitive prefill graph-cache observations. */
        virtual std::string prefillGraphDomainId() const { return "single"; }

        /** Domain-local participant id for MoE-sensitive prefill graph-cache observations. */
        virtual int prefillGraphParticipantId() const { return 0; }

        /** Stable topology fingerprint for MoE-sensitive prefill graph-cache observations. */
        virtual uint64_t prefillGraphTopologySignature() const { return 0; }

        /**
         * @brief Return safety state for optional chunk-boundary maintenance.
         *
         * Hosts with MoE rebalance domains can report histogram merge,
         * sparse-boundary, capture/replay, and participant-alignment state.
         * The default means no maintenance is requested for this chunk.
         */
        virtual PrefillChunkMaintenanceState prefillChunkMaintenanceState(
            const PrefillChunkPlan &chunk) const
        {
            PrefillChunkMaintenanceState state;
            state.chunk_index = chunk.chunk_index;
            state.histograms_merged = true;
            return state;
        }

        /**
         * @brief Run chunk-boundary maintenance after the gate allows it.
         *
         * This is where later MoE implementations merge telemetry-driven
         * rebalancing, prepared-weight transfer, runtime-table bank flips, and
         * graph-cache invalidation. The default is a no-op for non-MoE hosts.
         */
        virtual bool onPrefillChunkMaintenance(
            const PrefillChunkPlan &chunk,
            const PrefillChunkMaintenanceDecision &decision)
        {
            (void)chunk;
            (void)decision;
            return true;
        }

        /**
         * @brief Admit and bind one scheduled prefill graph to an external authority.
         *
         * The engine calls this after constructing the chunk-specific
         * ForwardInput and immediately before execute().  Implementations may
         * stamp transaction identity into @p chunk_input and install its exact
         * graph-launch dependency.  Any returned lease remains alive until the
         * delegated execution result has been published through finish().
         *
         * The default represents a graph with no external transaction authority.
         * It deliberately returns an empty lease rather than inventing a host
         * synchronization boundary.
         *
         * @param chunk_input Mutable chunk view that will be passed to execute().
         * @param chunk Immutable scheduler geometry for this submission.
         * @param lease Receives the request-scoped authority, when one is active.
         * @param error Receives an admission or binding diagnostic on failure.
         * @return True when the chunk may be submitted.
         */
        virtual bool beginPrefillChunkGraphSubmission(
            ForwardInput &chunk_input,
            const PrefillChunkPlan &chunk,
            std::unique_ptr<IPrefillChunkGraphSubmissionLease> *lease,
            std::string *error)
        {
            (void)chunk_input;
            (void)chunk;
            if (lease)
                lease->reset();
            if (error)
                error->clear();
            return true;
        }

        /**
         * @brief Begin the diagnostic namespace for one ordered prefill chunk.
         *
         * Production graph execution does not depend on this hook. Hosts that
         * retain parity snapshots use it to preserve every real-row checkpoint
         * under an immutable chunk ordinal while the graph executor publishes
         * its post-launch diagnostic copies. The hook runs on the diagnostic
         * host boundary and may retain host artifact metadata, but it must never
         * alter live inference state, captured graph topology, or stream order.
         *
         * @param chunk Scheduler-owned real/bucket geometry about to execute.
         */
        virtual void beginPrefillChunkSnapshotDiagnostics(
            const PrefillChunkPlan &chunk)
        {
            (void)chunk;
        }

        /**
         * @brief End the diagnostic namespace begun for one prefill chunk.
         *
         * The engine invokes this through an RAII scope, including failed and
         * exceptional graph attempts, so a later request cannot inherit an
         * obsolete chunk namespace.
         *
         * @param chunk Scheduler-owned chunk whose diagnostic scope is ending.
         */
        virtual void endPrefillChunkSnapshotDiagnostics(
            const PrefillChunkPlan &chunk)
        {
            (void)chunk;
        }

        /**
         * @brief Publish complete prompt-wide snapshots after all chunks succeed.
         *
         * A host may replace the historical bare snapshot keys, which otherwise
         * contain only the final chunk, with an ordered concatenation of the
         * context-qualified live rows. Returning false rejects the request's
         * diagnostic transaction rather than silently comparing an incomplete
         * checkpoint against a full reference prompt.
         *
         * @param schedule Successfully executed scheduler plan.
         * @return True when diagnostics are complete or the host does not
         *         retain snapshots.
         */
        virtual bool finalizePrefillChunkSnapshotDiagnostics(
            const PrefillChunkSchedule &schedule)
        {
            (void)schedule;
            return true;
        }

        /**
         * @brief Discard pending segmented-prefill diagnostic bookkeeping.
         *
         * This is paired with a failed schedule and keeps a partial sequence
         * from being mistaken for a later request's first chunk. Captured graph
         * resources and live tensors remain untouched.
         */
        virtual void cancelPrefillChunkSnapshotDiagnostics() noexcept {}
    };

    /**
     * @brief Forward graph execution engine with caching
     *
     * Manages the full lifecycle of forward graph execution:
     * 1. Compute cache signature from input + config
     * 2. On cache HIT: update buffers, copy PP hidden state, dispatch execution
     * 3. On cache MISS: build graph via host, execute, populate cache
     * 4. Collect GPU stage timeline telemetry
     *
     * Lifetime: owned by DeviceGraphOrchestrator, one per orchestrator instance.
     */
    class ForwardExecutionEngine
    {
    public:
        /**
         * @brief Static configuration for the engine (set once at construction).
         */
        struct Config
        {
            GraphCacheConfig cache_config;
            std::optional<FactoryPPStageConfig> pp_stage_config;
            bool has_unified_pp = false; ///< pipeline_config_ && pipeline_config_->hasPP()
        };

        ForwardExecutionEngine(Config config, DeviceGraphExecutor &executor);

        /**
         * @brief Runtime-facing host descriptor for one bucketed prefill chunk.
         *
         * The descriptor is intentionally pure: it selects a bucket, prepares
         * padded token/position buffers, and records whether padding would be
         * required. Launching remains gated by execute() and the prefill graph
         * cache preflight so unsafe padded graphs fail before execution.
         *
         * `ok` means the plan is allowed to execute under the current gate.
         * `chunk.ok` mirrors `ok` only for accepted plans; rejected padded plans
         * may still contain prepared buffers for diagnostics/tests.
         */
        struct PrefillChunkRuntimePlan
        {
            bool ok = false;                  ///< True when this chunk may execute under current gates.
            bool padding_required = false;    ///< True when real_count < bucket_seq_len.
            ForwardPositionPolicy position_policy =
                ForwardPositionPolicy::ExplicitRows; ///< Host rows on CPU; scalar device offset on GPU.
            int chunk_index = 0;              ///< Stable chunk ordinal within a prepared schedule.
            bool rebalance_allowed_after = false; ///< True when a maintenance hook may run after this chunk.
            bool rebalance_required_after = false; ///< True when a maintenance hook must run after this chunk.
            PrefillBucketSelection selection; ///< Bucket selection result.
            PrefillChunkExecutionInput chunk; ///< Prepared host buffers and real/bucket metadata.
            std::string error;                ///< Failure reason when ok is false.

            /// @brief Convenience conversion for success checks.
            explicit operator bool() const { return ok; }
        };

        /**
         * @brief Prepared runtime plans for a multi-chunk prefill range.
         */
        struct PrefillChunkRuntimeSchedule
        {
            bool ok = false;
            PrefillChunkSchedule schedule;
            std::vector<PrefillChunkRuntimePlan> chunks;
            std::string error;

            explicit operator bool() const { return ok; }
        };

        /**
         * @brief Prepare one bucketed prefill chunk from a ForwardInput.
         *
         * @param input ForwardInput with exactly one authoritative token source.
         *        CPU/host execution supplies `token_ids`, which this planner owns
         *        and pads. GPU request admission supplies `token_ids_device`, a
         *        stable arena bank whose inactive tail is already initialized.
         *        `token_offset` supplies the absolute prompt offset for position
         *        IDs and is not added to either token pointer.
         * @param allow_padded_execution Leave false for callers that have not
         *        opted into fixed-bucket execution. Setting it true prepares
         *        padded buffers for runPrefillChunk(); graph safety is still
         *        enforced by PrefillGraphCache preflight before execution/capture.
         */
        static PrefillChunkRuntimePlan prepareSinglePrefillChunkRuntimePlan(
            const ForwardInput &input,
            const std::vector<int> &bucket_sizes,
            int pad_token_id,
            bool allow_padded_execution = false);

        /**
         * @brief Prepare all bucketed runtime chunks for an explicit schedule policy.
         *
         * `input.token_ids` must point to the first real token at
         * `input.token_offset`. The policy's real-token range must be contained
         * within `[input.token_offset, input.token_offset + input.real_seq_len)`;
         * `input.seq_len` is used when `input.real_seq_len` is not set.
         */
        static PrefillChunkRuntimeSchedule preparePrefillChunkRuntimeSchedule(
            const ForwardInput &input,
            const PrefillChunkSchedulerPolicy &policy,
            int pad_token_id,
            bool allow_padded_execution = false);

        /**
         * @brief Execute a forward pass (cache-aware).
         *
         * @param input  Prepared input (position_ids resolved, external_hidden applied)
         * @param output Receives logits/hidden pointers on success
         * @param host   Host interface for model-specific callbacks
         * @return true on success
         *
         * @pre ExplicitRows inputs provide `position_ids` or a GPU-resident
         *      `position_ids_device`; ContiguousOffset inputs have batch_size=1
         *      and leave both pointers null.
         * @pre external_hidden_state already set in input (if applicable)
         */
        bool execute(const ForwardInput &input,
                     ForwardOutput &output,
                     IForwardExecutionHost &host);

        struct LastExecutedForwardGraphView
        {
            ComputeGraph *graph = nullptr;
            ForwardGraphSignature signature;
            DeviceId device = DeviceId::invalid();
            void *stream = nullptr;
            bool cache_hit = false;
            bool is_decode = false;
            bool all_position_logits = false;

            explicit operator bool() const { return graph != nullptr; }
        };

        /**
         * @brief Immutable capture template admitted for device-controlled composition.
         *
         * This view is deliberately stricter than LastExecutedForwardGraphView.
         * It is returned only for a replay-ready, monolithic GPU graph whose
         * token and position geometry are device-owned and whose replay has no
         * host callback obligations. The capture remains owned by this engine;
         * callers may clone it into a parent graph but must never mutate or launch
         * it through this view.
         */
        struct DeviceLoopGraphTemplateView
        {
            const IGPUGraphCapture *capture = nullptr;
            ForwardGraphSignature signature;
            DeviceId device = DeviceId::invalid();
            void *stream = nullptr;
            size_t stage_count = 0;
            size_t captured_node_count = 0;

            explicit operator bool() const noexcept
            {
                return capture != nullptr && device.is_gpu() && stream != nullptr &&
                       stage_count > 0 && captured_node_count > 0;
            }
        };

        /**
         * @brief Immutable identity of one exact retained decode replay plan.
         *
         * Unlike @ref DeviceLoopGraphTemplateView, this view deliberately
         * admits an explicitly segmented heterogeneous plan.  It does not
         * expose the segment cache or an executable pointer: callers may use
         * the identity only with @ref replayRetainedDecodeGraph, which keeps
         * capture ownership and execution policy inside this engine.
         */
        struct RetainedDecodeGraphView
        {
            ForwardGraphSignature signature;
            DeviceId device = DeviceId::invalid();
            void *stream = nullptr;
            size_t replay_unit_count = 0; ///< Executables submitted per invocation; one for a composed parent.
            size_t source_capture_unit_count = 0; ///< Native child templates represented by the plan.
            bool segmented = false; ///< True only for host-sequenced heterogeneous replay.
            bool composed_parent = false; ///< True when one device parent owns all source units.

            explicit operator bool() const noexcept
            {
                return device.is_gpu() && stream != nullptr &&
                       replay_unit_count > 0;
            }
        };

        struct ReplayCacheObservation
        {
            ForwardGraphSignature signature;
            bool valid = false;
            bool segment_initialized = false;
            bool segment_needs_capture = false;
            bool phase3_active = false;
            bool gpu_stream_bindings_applied = false;
            bool has_capture_stream = false;
            uint64_t segment_decode_step = 0;
            uint64_t graph_replay_live_state_epoch = 0;
            bool requires_live_state_epoch_recapture = false;
            bool all_position_verifier_recapture_pending = false;
        };

        /**
         * @brief Counts the replay-state action chosen for a live-state mutation.
         *
         * MTP publication is a precise boundary: live-state-versioned captures
         * must be recaptured before reuse, while single-row condition decode can
         * keep its executable and only rebind explicit streams. Returning this
         * summary makes that split testable and exportable.
         */
        struct ReplayStateResetSummary
        {
            size_t reset_replay_state = 0;
            size_t preserved_for_stream_rebind = 0;
            size_t ordinary_decode_reset = 0;
            size_t all_position_verifier_preserved = 0;
            size_t other_preserved = 0;
            bool all_position_verifier_recapture_requested = false;
        };

        /**
         * @brief Return the cached forward graph used by the most recent
         *        successful cached execution.
         *
         * MTP accepted-state publication needs the exact verifier graph that
         * just produced `draft_count + 1` target rows. This view is intentionally
         * narrow and returns empty for uncached or failed forwards so callers do
         * not accidentally publish from stale graph-cache entries.
         */
        std::optional<LastExecutedForwardGraphView> lastExecutedForwardGraph();

        /**
         * @brief Return the most recent all-position verifier graph.
         *
         * Accepted-state publication needs the graph whose GDN/short-conv stages
         * captured verifier row snapshots. A shifted-MTP sidecar commit may run
         * another graph before publication, so this handle is intentionally
         * separate from lastExecutedForwardGraph().
         */
        std::optional<LastExecutedForwardGraphView> lastAllPositionVerifierForwardGraph();

        /**
         * @brief Export the last successful forward as a device-loop graph template.
         *
         * @param error Optional diagnostic describing the first violated hard
         *        contract. Failure never selects eager execution or host replay.
         */
        std::optional<DeviceLoopGraphTemplateView>
        lastExecutedDeviceLoopGraphTemplate(std::string *error = nullptr) const;

        /**
         * @brief Export one replay-ready capture by its complete immutable signature.
         *
         * Parent graph construction may compose several forward geometries at
         * once, so mutable "last executed" state is not a sufficient identity.
         * This lookup requires an exact ForwardGraphSignature cache hit and never
         * substitutes a nearby bucket, a newer execution, or an eager path.
         *
         * @param signature Complete forward-cache identity embedded by capture.
         * @param error Optional diagnostic describing the first violated hard
         *        contract. Failure never mutates cache state.
         * @return Immutable capture view when the exact entry is replay-ready.
         */
        std::optional<DeviceLoopGraphTemplateView>
        deviceLoopGraphTemplate(
            const ForwardGraphSignature &signature,
            std::string *error = nullptr) const;

        /**
         * @brief Inspect one exact retained decode graph without requiring monolithic capture.
         *
         * The cache entry must already be in steady replay state and must own
         * device-resident token and position inputs.  Both a single native
         * capture and a typed heterogeneous segmented plan are valid; warmup,
         * recapture, eager execution, and stale cache lookup are rejected.
         *
         * @param signature Complete immutable forward-cache identity.
         * @param error Optional first violated retained-replay invariant.
         * @return Read-only identity for an exact replay-ready plan.
         */
        std::optional<RetainedDecodeGraphView> retainedDecodeGraph(
            const ForwardGraphSignature &signature,
            std::string *error = nullptr) const;

        /**
         * @brief Export the active native units of one exact retained decode graph.
         *
         * This is the topology-aware companion to @ref deviceLoopGraphTemplate.
         * A heterogeneous sparse parent may need to insert mapped timeline nodes
         * between LocalTP-coordinated capture units, so a monolithic child is not
         * always the correct composition boundary. Every returned unit has already
         * passed the strict cache-level retained-plan contract; manual replay units
         * and per-replay host preparation remain hard failures.
         *
         * The returned views borrow graph-cache storage and are setup-only. They
         * neither launch the graph nor authorize the ordinary segmented host
         * controller to remain in the inference path.
         *
         * @param signature Complete immutable forward-cache identity.
         * @param error Optional first violated retained-plan invariant.
         * @return Ordered active native capture units for parent composition.
         */
        std::optional<std::vector<
            DeviceGraphExecutor::GraphSegmentCache::
                RetainedCaptureUnitTemplateView>>
        retainedDecodeCaptureUnitTemplates(
            const ForwardGraphSignature &signature,
            std::string *error = nullptr) const;

        /**
         * @brief Replay one exact retained decode plan with a new sparse wire identity.
         *
         * This is the hosted heterogeneous counterpart to cloning a monolithic
         * child into a native CUDA conditional parent.  It launches the already
         * captured production plan on its owned stream, including declared
         * sparse-collective manual boundaries, and never warms, captures,
         * allocates, synchronizes, or substitutes eager execution.
         *
         * @param signature Exact cache identity returned by retainedDecodeGraph().
         * @param ctx Device context used by the retained plan.
         * @param sparse_params Root-authoritative generation/operation identity.
         * @param launch_dependency Exact pre-launch edge used to arm an
         *        external heterogeneous follower after native graph setup.
         * @param host The same runner authority used during materialization.
         *        The engine obtains its background branch policy directly and
         *        authenticates it against the retained executable. Callers
         *        cannot omit that policy or substitute an empty descriptor.
         * @param transaction_stream Exact external scheduler stream ordered
         *        before and after this retained replay through the cache-owned
         *        device event. It must be non-null and may not alias another
         *        device.
         * @param out_producer_stream Receives the exact replay producer stream.
         * @param error Optional first violated lifecycle invariant.
         * @return True after the retained plan has been enqueued successfully.
         */
        bool replayRetainedDecodeGraph(
            const ForwardGraphSignature &signature,
            IDeviceContext *ctx,
            const IComputeStage::MoEOverlayCollectiveRuntimeParams
                &sparse_params,
            const DeviceGraphExecutor::GraphLaunchDependencyHook
                &launch_dependency,
            IForwardExecutionHost &host,
            void *transaction_stream,
            void **out_producer_stream,
            std::string *error = nullptr);

        /**
         * @brief Export the retained all-position verifier capture for composition.
         *
         * The same monolithic/device-owned/callback-free requirements apply as
         * lastExecutedDeviceLoopGraphTemplate().
         */
        std::optional<DeviceLoopGraphTemplateView>
        lastAllPositionVerifierDeviceLoopGraphTemplate(
            std::string *error = nullptr) const;

        /**
         * @brief Drop the retained verifier graph publication handle.
         *
         * Callers clear this after accepting, rejecting, or abandoning the
         * publication transaction so a later request cannot publish from stale
         * row snapshots.
         */
        void clearLastAllPositionVerifierForwardGraph();

        /**
         * @brief Snapshot replay-cache version/capture state for diagnostics/tests.
         *
         * The returned state is intentionally read-only. It lets unit and
         * integration guards prove that live-state mutations advance epochs and
         * force stale multi-row decode/verifier captures to recapture.
         */
        std::vector<ReplayCacheObservation> replayCacheObservations(
            uint64_t live_state_epoch) const;

        /**
         * @brief Immutable diagnostic snapshot for one cached prefill graph bucket.
         *
         * This is intentionally read-only: callers can observe the warmup,
         * capture, replay, and eviction counters without reaching into the
         * ForwardGraphCache internals or mutating graph lifetime state.
         */
        struct PrefillGraphCacheSnapshot
        {
            bool forward_cache_valid = false;                  ///< True when the matching forward graph entry is valid.
            bool prefill_cache_initialized = false;            ///< True after the prefill cache has been created for the entry.
            PrefillGraphPhase phase = PrefillGraphPhase::Cold; ///< Current bucket lifecycle phase.
            size_t cache_size = 0;                             ///< Number of bucket entries held by the prefill cache.
            size_t node_count = 0;                             ///< Captured graph node count for Ready entries.
            int replay_count = 0;                              ///< Successful graph launches for this bucket.
            uint64_t warmup_count = 0;                         ///< Lifetime warmups for this bucket.
            uint64_t initialized_count = 0;                    ///< Lifetime request resets preserving lazy init for this bucket.
            uint64_t capture_count = 0;                        ///< Lifetime successful captures for this bucket.
            uint64_t eviction_count = 0;                       ///< Total prefill bucket evictions observed by this engine.
            bool observation_valid = false;                    ///< True when runtime chunk/capture metadata has been observed.
            int chunk_index = 0;                               ///< Stable chunk ordinal from the latest prefill execution.
            int bucket_seq_len = 0;                            ///< Fixed graph bucket length from the latest prefill execution.
            int real_token_start = 0;                          ///< Inclusive real-token start offset from the latest prefill execution.
            int real_token_count = 0;                          ///< Real tokens represented by the latest prefill execution.
            int real_token_end = 0;                            ///< Exclusive real-token end offset from the latest prefill execution.
            std::string domain_id;                             ///< Prefix/MoE domain id associated with this graph observation.
            int participant_id = 0;                            ///< Domain-local participant id associated with this graph observation.
            uint64_t placement_epoch = 0;                      ///< MoE placement epoch associated with this graph observation.
            uint64_t topology_signature = 0;                   ///< Topology signature associated with this graph observation.
            std::string capture_phase;                         ///< cold, warmup, capture, replay, or rejected.
            std::string recapture_reason;                      ///< none or a structured reason for recapture/rejection.
            std::string reject_stage_name;                     ///< First stage that blocked graph capture, if any.
            std::string reject_stage_type;                     ///< Type of the first stage that blocked graph capture, if any.
        };

        /**
         * @brief Snapshot all cached prefill graph buckets for diagnostics.
         *
         * The vector is intentionally read-only and includes the latest observed
         * capture/replay metadata, so benchmark and server probes can assert
         * that bucketed prefill is using graph replay without parsing logs.
         */
        std::vector<PrefillGraphCacheSnapshot> prefillGraphCacheSnapshots() const;

        /**
         * @brief Execute one prepared bucketed prefill chunk.
         *
         * This Phase 6 boundary consumes the prepared chunk buffers from a
         * PrefillChunkRuntimePlan and delegates to execute() with fixed-bucket
         * shape metadata. Padded chunks are allowed only as prepared plans; the
         * forward graph path must pass PrefillGraphCache preflight before any
         * padded graph is executed, captured, or replayed.
         *
         * @param base_input ForwardInput carrying device, KV cache, PP, and
         *        batching context to preserve for the chunk execution.
         * @param plan Prepared runtime plan for the chunk. Must be exact.
         * @param output Receives logits/hidden pointers on success.
         * @param host Host interface for model-specific callbacks.
         * @return true when execute() succeeds; false if the plan is invalid,
         *         graph preflight rejects padded execution, or the delegated
         *         execution fails.
         */
        bool runPrefillChunk(
            const ForwardInput &base_input,
            const PrefillChunkRuntimePlan &plan,
            ForwardOutput &output,
            IForwardExecutionHost &host);

        /**
         * @brief Execute all chunks in a prepared prefill runtime schedule.
         *
         * Chunks run in schedule order. Each successful chunk passes through
         * the same chunk-boundary maintenance gate as runPrefillChunk(); the
         * first failed execution or maintenance hook stops the schedule.
         */
        bool runPrefillChunkSchedule(
            const ForwardInput &base_input,
            const PrefillChunkRuntimeSchedule &schedule,
            ForwardOutput &output,
            IForwardExecutionHost &host);

        // ----- Cache Management -----

        /** Invalidate all cached graphs and release resources. */
        void invalidateAll();

        /** Discard all cached forward graphs after a topology/workspace lifetime change. */
        void discardAllCachedGraphs();

        /**
         * @brief Invalidate cached graphs whose MoE placement is encoded in
         *        captured stage topology rather than runtime descriptor tables.
         *
         * Dynamic GPU MoE rebalance mutates single-token decode runtime tables
         * in place. Prefill, all-position verifier, and multi-token decode
         * graphs are not yet proven placement-stable and must not replay after
         * a mask publication under a stable MoE epoch.
         *
         * @return Number of valid cached graphs invalidated.
         */
        size_t invalidateMoEPlacementSensitiveGraphsForStablePlacement();

        /**
         * @brief Drop captured GPU replay state while keeping cached ComputeGraphs.
         *
         * Live checkpoint restore rewinds KV/GDN/MTP state within an active request.
         * Cached graph objects and stable buffers are still usable, but replaying a
         * previously captured HIP/CUDA graph across that restore boundary can encode
         * stale runtime state. The next execution will warm up/capture again.
         */
        void resetCapturedReplayState();

        /**
         * @brief Drop only replay state that can be consumed by MTP correction replay.
         *
         * Rejected speculative steps publish an accepted verifier prefix, then
         * immediately replay the corrected token through the ordinary main
         * decode path. Any capture that encodes multi-row live-state progression
         * must be fresh. Dense single-token decode can preserve its executable
         * because every dynamic input is rebound before launch and its recurrent
         * kernels read backend-owned live buffers at stable addresses. CUDA and
         * ROCm GDN/short-conv integration regressions prove that contract by
         * publishing every accepted row at M=2/3/4 and requiring byte-identical
         * captured-graph continuation versus serial M=1 decode. Callers may set
         * @p preserve_single_token_decode_replay to false only for execution
         * domains that have not established the same ownership contract.
         * Preserved caches still have their stage stream bindings dirtied so a
         * KernelFactory dynamic-state reset cannot leave backend kernels with
         * null streams before the next updateDynamicParams() call.
         */
        ReplayStateResetSummary resetCapturedReplayStateForCorrectionReplay(
            uint64_t live_state_epoch = 0,
            bool preserve_single_token_decode_replay = true);

        /**
         * @brief Rebind replay-safe captures after restoring a live checkpoint.
         *
         * A checkpoint restore replaces the contents behind canonical KV,
         * recurrent-state, token, and position buffers without changing their
         * addresses.  Single-token decode, all-position verifier, and fixed
         * prefill captures therefore remain valid after their producer streams
         * are rebound to the restore event.  Multi-row ordinary decode remains
         * live-state-versioned and has only its replay state reset.
         *
         * @param live_state_epoch Epoch established by the prefix-restore
         *        mutation. Preserved entries are stamped with this epoch.
         * @return Counts describing which cache classes were preserved or reset.
         */
        ReplayStateResetSummary rebindCapturedReplayStateAfterPrefixRestore(
            uint64_t live_state_epoch);

        /**
         * @brief Drop captured all-position verifier replay after verifier-input mutation.
         *
         * Shifted MTP KV catch-up mutates an auxiliary cache read only by
         * multi-row verifier graphs.  Ordinary decode and sidecar captures do
         * not need to be discarded for that boundary, but ROCm verifier graph
         * replay has proven unsafe when only the live-state epoch advances.
         *
         * This method also records a one-shot recapture request for the next
         * all-position verifier graph.  That matters when the shifted-cache
         * mutation happens before the verifier cache exists, or while it is
         * between capture/replay phases: the next verifier execution
         * still has to settle on a freshly captured executable before the
         * request is consumed.
         */
        ReplayStateResetSummary resetAllPositionVerifierReplayState();

        /**
         * @brief Reset request/replay state without discarding cached forward graphs.
         *
         * Used at request/session boundaries after the orchestrator clears KV and
         * model recurrent state. Keeping the ComputeGraph avoids weight
         * re-coherence. When @p preserve_replay_safe_graphs is true,
         * single-token decode and all-position verifier segment captures survive
         * the request reset because their device inputs are rebound/refreshed
         * before every launch. Exact and bucketed prefill cache state also uses
         * the preserving reset path so parity and serving can keep the captured
         * prefill fast path warm; request-local monolithic prefill graph-cache
         * executables are demoted separately by PrefillGraphCache. Multi-token
         * ordinary decode replay state is still discarded.
         */
        ReplayStateResetSummary resetSessionReplayState(
            bool preserve_replay_safe_graphs = false);

        /** Check if cache is empty. */
        [[nodiscard]] bool cacheEmpty() const { return cache_.empty(); }

        /**
         * @brief Visit all stages of a given type across all cached graphs.
         * @param type   Stage type to filter for.
         * @param visitor Callback receiving (IComputeStage*) for each match.
         */
        void forEachCachedStage(ComputeStageType type,
                                const std::function<void(IComputeStage *)> &visitor) const;

        /**
         * @brief Visit every stage across all valid cached forward graphs.
         *
         * This is used for lifecycle boundaries that apply to stage-owned
         * metadata regardless of stage type, such as invalidating handles into
         * backend kernel-dynamic state after KernelFactory resets its dynamic
         * tables. The cached ComputeGraphs remain valid; only the visited
         * stage's own dynamic handles are expected to change.
         *
         * @param visitor Callback receiving each cached IComputeStage pointer.
         */
        void forEachCachedStage(const std::function<void(IComputeStage *)> &visitor) const;

        /**
         * @brief Return diagnostic prefill graph cache state for a cached forward signature.
         *
         * Returns `std::nullopt` when the forward graph signature has not been
         * cached. A present snapshot may still report `prefill_cache_initialized=false`
         * because the first prefill request builds and caches the forward graph;
         * prefill graph warmup starts on the following cache hit.
         */
        std::optional<PrefillGraphCacheSnapshot> prefillGraphCacheSnapshot(
            const ForwardGraphSignature &signature,
            const PrefillGraphCacheKey &key) const;

        // ----- Mutable Execution Flags -----

        void setSuppressTimeline(bool v) { suppress_timeline_ = v; }
        void setAccumulatePrefill(bool v) { accumulate_prefill_ = v; }

        /** Access the forward pass profiler for flushing at benchmark end. */
        ForwardPassProfiler &forwardPassProfiler() { return forward_pass_profiler_; }

    private:
        struct LastExecutedForwardGraphState
        {
            bool valid = false;
            ForwardGraphSignature signature;
            bool cache_hit = false;
            void *producer_stream = nullptr;
        };

        /**
         * @brief Retain the exact graph and producer identity of one launch.
         * @param signature Complete cache identity of the launched graph.
         * @param cache_hit Whether an existing graph topology was reused.
         * @param producer_stream Exact output producer stream; null only for CPU.
         */
        void recordLastExecutedForwardGraph(
            const ForwardGraphSignature &signature,
            bool cache_hit,
            void *producer_stream);
        std::optional<LastExecutedForwardGraphView> viewForLastExecutedForwardGraphState(
            const LastExecutedForwardGraphState &state);
        std::optional<DeviceLoopGraphTemplateView>
        deviceLoopGraphTemplateForState(
            const LastExecutedForwardGraphState &state,
            std::string *error) const;

        /**
         * @brief Submit a cached graph under its complete immutable identity.
         * @param input Request inputs or explicit setup-materialization intent.
         * @param output Destination for the executed forward's published views.
         * @param cache The exact graph and stable storage selected by signature.
         * @param host Participant-local execution and publication services.
         * @param signature Canonical cache identity, including verifier/condition role.
         * @param start Start of this forward's host timing interval.
         * @return True after the requested materialization or execution succeeds.
         */
        bool executeCacheHit(
            const ForwardInput &input,
            ForwardOutput &output,
            ForwardGraphCache &cache,
            IForwardExecutionHost &host,
            const ForwardGraphSignature &signature,
            std::chrono::high_resolution_clock::time_point start);

        /**
         * @brief Capture and instantiate one cached GPU graph without execution.
         *
         * The cache entry, workspace generation, permanent input addresses, and
         * explicit capture stream must already be installed. This setup boundary
         * records native units and local-TP capture waves only; it never prepares
         * live request state, executes manual segments, launches an executable,
         * or publishes output provenance.
         * @param input Explicit setup inputs; no request-owned launch dependency.
         * @param forward_cache Graph and permanent buffers belonging to signature.
         * @param host Participant-local capture and ordering services.
         * @param signature Exact graph role used for capture and replay attribution.
         * @return True when the exact executable is retained without launching it.
         */
        bool materializeCachedExecutableWithoutLaunch(
            const ForwardInput &input,
            ForwardGraphCache &forward_cache,
            IForwardExecutionHost &host,
            const ForwardGraphSignature &signature);

        // ----- Cache MISS execution path -----
        bool executeCacheMiss(
            const ForwardInput &input_in,
            ForwardOutput &output,
            const ForwardGraphSignature &signature,
            ForwardGraphCache *build_cache,
            bool should_cache,
            IForwardExecutionHost &host,
            bool is_decode,
            bool has_unified_pp,
            std::chrono::high_resolution_clock::time_point start);

        /**
         * @brief Execute one prefill through its production graph lifecycle.
         *
         * Monolithic bucket capture and explicitly admitted heterogeneous
         * segmentation own different caches and streams. The method therefore
         * returns the concrete launch stream instead of asking callers to
         * reconstruct provenance from phase booleans. Setup-only
         * materialization succeeds with a null producer because it launches no
         * inference transaction.
         *
         * @param input Exact prefill invocation and graph identity.
         * @param forward_cache Stable topology and native executable storage.
         * @param ctx Device context owning the graph participant.
         * @param host Model/runtime lifecycle authority.
         * @param used_graph_replay Receives whether retained graph replay ran.
         * @param out_producer_stream Receives the exact non-null launch stream
         *        after GPU inference; remains null for setup materialization.
         * @param initial_submission Launch or setup-materialization policy.
         * @return True after successful materialization or graph submission.
         */
        bool executePrefillWithGraphCache(
            const ForwardInput &input,
            ForwardGraphCache &forward_cache,
            IDeviceContext *ctx,
            IForwardExecutionHost &host,
            bool *used_graph_replay = nullptr,
            void **out_producer_stream = nullptr,
            DeviceGraphExecutor::GraphInitialSubmissionPolicy
                initial_submission =
                    DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                        CaptureInstantiateAndLaunch);

        /** @brief Mark any exact or bucketed prefill topology entry as recently used. */
        void touchPrefillForwardCache(
            const ForwardGraphSignature &signature,
            ForwardGraphCache &cache);

        /** @brief Enforce the engine-level cap for all reusable prefill topologies. */
        void enforcePrefillForwardCapacity(
            const ForwardGraphSignature *active_signature = nullptr);

        /** @brief Count valid top-level exact and bucketed prefill cache entries. */
        size_t prefillForwardCacheSize() const;

        // ----- GPU Stage Timeline collection -----
        void collectTimeline(
            IForwardExecutionHost &host,
            IDeviceContext *ctx,
            bool is_decode,
            const ForwardInput &input,
            std::chrono::high_resolution_clock::time_point start,
            std::string stage_context = {});

        // ----- Configuration -----
        Config config_;
        DeviceGraphExecutor &executor_;

        // ----- Cache -----
        std::unordered_map<ForwardGraphSignature, ForwardGraphCache, ForwardGraphSignatureHash> cache_;
        uint64_t prefill_forward_access_counter_ = 0;
        uint64_t prefill_forward_eviction_count_ = 0;
        LastExecutedForwardGraphState last_executed_forward_graph_;
        LastExecutedForwardGraphState last_all_position_verifier_graph_;

        /*
         * Shifted MTP KV mutation is verifier-input mutation, not ordinary
         * live-prefix mutation.  A cache scan can drop existing verifier
         * captures, but it cannot protect a verifier graph that is created
         * after the mutation.  Keep a pending intent until an all-position
         * verifier has reached a fresh replay-ready capture.
         */
        bool all_position_verifier_recapture_pending_ = false;

        // ----- Mutable execution flags -----
        bool suppress_timeline_ = false;
        bool accumulate_prefill_ = false;
        bool first_graph_ready_fired_ = false;

        // ----- Forward pass wall-clock profiler -----
        ForwardPassProfiler forward_pass_profiler_;
    };

} // namespace llaminar2
