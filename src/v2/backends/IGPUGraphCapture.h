/**
 * @file IGPUGraphCapture.h
 * @brief Backend-neutral ownership and composition API for native GPU graphs.
 *
 * The interface exposes graph lifecycle operations together with the small set
 * of native conditional compositions used by fully device-owned generation.
 * Conditional predicates and selectors always name persistent device rows;
 * implementations may not materialize those values on the host, insert host
 * callbacks, synchronize a stream, or allocate replay-time storage.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{

    class IGPUGraphCapture;

    /**
     * @brief Device-owned execution policy for one transaction fragment.
     *
     * CUDA lowers conditional fragments into one native parent graph. HIP
     * retains the same typed branch description and submits a conditional tail
     * only when its authenticated device-published ticket says it is due. The
     * policy never authorizes host inspection of the predicate word itself,
     * mutable inference state, or an eager kernel path.
     */
    enum class DeviceControlledLoopFragmentExecution
    {
        Always,              ///< Execute on every admitted loop iteration.
        IfDeviceWordNonZero, ///< Execute only when the bound device word is non-zero.
        /** Execute when the validated transaction selector is at least the fragment threshold. */
        IfDeviceSelectorAtLeast,
    };

    /// Result of an in-place graph executable update
    enum class GraphUpdateResult
    {
        Success,            ///< Graph executable updated in-place
        NeedsReinstantiate, ///< Topology changed, needs full re-instantiation
        Failed              ///< Update failed; the owning execution path must fail
    };

    /**
     * @brief One recursively discovered kernel node in a captured GPU graph.
     *
     * The record contains only immutable capture-time metadata. Inspecting a
     * graph never launches it, reads device memory, synchronizes a stream, or
     * changes executable state. `graph_path` identifies the nested child-graph
     * route used to reach the node; it is diagnostic identity and is not an
     * execution-order guarantee.
     *
     * A backend may encounter a kernel registered through a library mechanism
     * that does not expose a symbolic name through its runtime API. Such a node
     * remains present with `name_resolved == false` and its non-zero function
     * identity, so an inventory can never silently omit work.
     */
    struct GPUGraphKernelNodeInfo
    {
        std::string name;                       ///< Runtime kernel name or `unresolved_kernel`.
        std::string graph_path;                 ///< Recursive root/child path for diagnostics.
        uintptr_t function_identity = 0;        ///< Backend runtime function identity.
        uint32_t grid_x = 0;                    ///< Capture-time grid width.
        uint32_t grid_y = 0;                    ///< Capture-time grid height.
        uint32_t grid_z = 0;                    ///< Capture-time grid depth.
        uint32_t block_x = 0;                   ///< Capture-time block width.
        uint32_t block_y = 0;                   ///< Capture-time block height.
        uint32_t block_z = 0;                   ///< Capture-time block depth.
        size_t dynamic_shared_memory_bytes = 0;   ///< Per-block dynamic shared memory.
        size_t static_shared_memory_bytes = 0;    ///< Compiler-owned shared memory per block.
        size_t local_memory_bytes_per_thread = 0; ///< Compiler-owned local memory per thread.
        uint32_t registers_per_thread = 0;         ///< Compiler-assigned registers per thread.
        uint32_t max_threads_per_block = 0;        ///< Backend launch ceiling for this kernel.
        uint32_t max_active_blocks_per_sm = 0;     ///< Occupancy ceiling at captured geometry.
        size_t nesting_depth = 0;                 ///< Number of enclosing child graphs.
        bool name_resolved = false;               ///< Whether `name` came from the runtime.

        /**
         * @brief Verify that the record describes a launchable kernel node.
         *
         * @return true when function identity and every launch dimension are
         *         present. A symbolic name is intentionally not required because
         *         opaque library kernels must still be represented.
         */
        [[nodiscard]] bool valid() const noexcept
        {
            return function_identity != 0 && grid_x > 0 && grid_y > 0 &&
                   grid_z > 0 && block_x > 0 && block_y > 0 && block_z > 0 &&
                   registers_per_thread > 0 && max_threads_per_block > 0 &&
                   max_active_blocks_per_sm > 0;
        }
    };

    /**
     * @brief Device-resident predicate for a graph-owned request loop.
     *
     * The predicate addresses a row-major controller table that remains on the
     * accelerator for the complete graph launch.  A loop iteration is admitted
     * only while every request is healthy and at least one request remains
     * incomplete.  A controller failure therefore stops device execution and is
     * surfaced by the caller's terminal validation; it cannot spin forever or
     * silently switch to host scheduling.
     *
     * The field indices deliberately remain explicit.  The graph abstraction
     * does not own the generation-controller ABI, while its caller can bind the
     * exact health and completion fields from that ABI without manufacturing a
     * second device flag or a host-visible mirror.
     */
    struct DeviceControlledLoopPredicate
    {
        const int *control_rows_device = nullptr; ///< First word of row zero on device.
        int control_stride = 0;                   ///< Controller words between request rows.
        int request_count = 0;                    ///< Number of active request rows.
        int healthy_index = -1;                   ///< Non-zero means the row remains valid.
        int complete_index = -1;                  ///< Non-zero means the row is terminal.

        [[nodiscard]] bool valid() const noexcept
        {
            return control_rows_device != nullptr &&
                   control_stride > 0 &&
                   request_count > 0 &&
                   healthy_index >= 0 &&
                   healthy_index < control_stride &&
                   complete_index >= 0 &&
                   complete_index < control_stride;
        }
    };

    /**
     * @brief Device-resident selector and fatal validation policy for one loop transaction.
     *
     * Every healthy, incomplete request must publish the same selector. Values
     * outside `[minimum_selector, maximum_selector]`, or disagreement between
     * active rows, invalidate all active rows before any transaction fragment
     * executes. The validated value gates a monotonic linear prefix; it never
     * selects among duplicated complete transaction bodies.
     */
    struct DeviceControlledLoopSelector
    {
        int *control_rows_device = nullptr; ///< Mutable first word of row zero.
        int control_stride = 0;             ///< Controller words between rows.
        int request_count = 0;              ///< Uniformly switched request rows.
        int healthy_index = -1;             ///< Non-zero means the row is valid.
        int complete_index = -1;            ///< Non-zero means the row is terminal.
        int selector_index = -1;            ///< Branch index published by the device.
        int error_index = -1;                ///< Fatal diagnostic word to publish.
        int minimum_selector = 0;            ///< First implemented branch index.
        int maximum_selector = -1;           ///< Last implemented branch index.
        int invalid_selector_error = 0;      ///< Non-zero stable terminal error code.

        /**
         * @brief Validate scalar layout independently of the branch table.
         */
        [[nodiscard]] bool valid() const noexcept
        {
            return control_rows_device != nullptr && control_stride > 0 &&
                   request_count > 0 && healthy_index >= 0 &&
                   healthy_index < control_stride && complete_index >= 0 &&
                   complete_index < control_stride && selector_index >= 0 &&
                   selector_index < control_stride && error_index >= 0 &&
                   error_index < control_stride && minimum_selector >= 0 &&
                   maximum_selector >= minimum_selector &&
                   invalid_selector_error != 0;
        }
    };

    /**
     * @brief One named native-graph fragment in a device-controlled transaction.
     *
     * The semantic name is part of the composition contract rather than an
     * incidental log string. Native conditional APIs impose stricter recursive
     * node rules than ordinary graph replay, so a rejected fragment must identify
     * the exact producer role that exported an incompatible graph. The name and
     * capture are borrowed only for the duration of parent construction.
     */
    struct DeviceControlledLoopFragment
    {
        const char *name = nullptr;                ///< Stable non-empty producer role.
        const IGPUGraphCapture *capture = nullptr; ///< Borrowed captured graph owner.
        DeviceControlledLoopFragmentExecution execution =
            DeviceControlledLoopFragmentExecution::Always;
        /**
         * Persistent device scalar used by @ref IfDeviceWordNonZero.
         *
         * The graph reads this address through a one-thread condition kernel.
         * It must remain stable for the complete executable lifetime.
         */
        const uint32_t *condition_word_device = nullptr;
        /**
         * Inclusive selector threshold used by @ref IfDeviceSelectorAtLeast.
         *
         * The enclosing selector contract owns the device address and legal
         * interval. A negative value means that this fragment has no selector
         * gate, making the three execution policies disjoint by construction.
         */
        int minimum_selector = -1;

        /** @brief Verify that the fragment carries complete diagnostic identity. */
        [[nodiscard]] bool valid() const noexcept
        {
            bool condition_binding_valid = false;
            switch (execution)
            {
            case DeviceControlledLoopFragmentExecution::Always:
                condition_binding_valid = condition_word_device == nullptr &&
                                          minimum_selector < 0;
                break;
            case DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero:
                condition_binding_valid = condition_word_device != nullptr &&
                                          minimum_selector < 0;
                break;
            case DeviceControlledLoopFragmentExecution::
                IfDeviceSelectorAtLeast:
                condition_binding_valid = condition_word_device == nullptr &&
                                          minimum_selector >= 0;
                break;
            }
            return name != nullptr && name[0] != '\0' && capture != nullptr &&
                   condition_binding_valid;
        }

        /**
         * @brief Compare every field embedded in a composed executable.
         *
         * The semantic name is diagnostic metadata. Capture identity,
         * execution policy, and device predicate address determine replay
         * compatibility and therefore participate in graph-cache identity.
         */
        [[nodiscard]] bool hasSameExecutionIdentity(
            const DeviceControlledLoopFragment &other) const noexcept
        {
            return capture == other.capture && execution == other.execution &&
                   condition_word_device == other.condition_word_device &&
                   minimum_selector == other.minimum_selector;
        }
    };

    /**
     * @brief Authenticated hosted replay view of one device-controlled iteration.
     *
     * Native conditional graphs preserve the source fragment order themselves.
     * A backend without conditional graph nodes publishes the two immutable
     * decisions needed to reproduce that selection: whether another loop
     * iteration is admitted and whether its conditional tail is due.  This
     * helper applies those decisions while retaining the original fragment
     * order; it must never move a conditional maintenance tail ahead of the
     * transaction that produces and releases its inference state.
     */
    struct DeviceControlledLoopTicketSelection
    {
        bool iteration_admitted = false;       ///< Whether the native loop body would run.
        bool conditional_word_nonzero = false; ///< Authenticated value for the conditional tail.
        int selector = -1; ///< Authenticated selector for a depth-gated fragment.

        /**
         * @brief Decide whether one execution policy participates in this iteration.
         * @tparam Fragment A fragment exposing execution and threshold fields.
         * @param fragment Typed policy stored in the retained transaction.
         * @return true when the fragment must be submitted.
         */
        template <typename Fragment>
        [[nodiscard]] constexpr bool selects(
            const Fragment &fragment) const noexcept
        {
            if (!iteration_admitted)
                return false;
            switch (fragment.execution)
            {
            case DeviceControlledLoopFragmentExecution::Always:
                return true;
            case DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero:
                return conditional_word_nonzero;
            case DeviceControlledLoopFragmentExecution::
                IfDeviceSelectorAtLeast:
                return selector >= fragment.minimum_selector;
            }
            return false;
        }

        /**
         * @brief Count selected fragments without allocating a filtered branch.
         * @tparam Fragment A fragment type exposing an `execution` member.
         * @param ordered_fragments Producer-ordered retained branch.
         * @return Number of fragments selected for the authenticated iteration.
         */
        template <typename Fragment>
        [[nodiscard]] constexpr size_t countSelected(
            std::span<const Fragment> ordered_fragments) const noexcept
        {
            size_t count = 0;
            for (const Fragment &fragment : ordered_fragments)
            {
                if (selects(fragment))
                    ++count;
            }
            return count;
        }

        /**
         * @brief Resolve a selected ordinal while preserving producer order.
         * @tparam Fragment A fragment type exposing an `execution` member.
         * @param ordered_fragments Producer-ordered retained branch.
         * @param selected_ordinal Zero-based ordinal in the filtered view.
         * @return Borrowed selected fragment, or nullptr when out of range.
         */
        template <typename Fragment>
        [[nodiscard]] constexpr const Fragment *selectOrdinal(
            std::span<const Fragment> ordered_fragments,
            size_t selected_ordinal) const noexcept
        {
            size_t current_ordinal = 0;
            for (const Fragment &fragment : ordered_fragments)
            {
                if (!selects(fragment))
                    continue;
                if (current_ordinal == selected_ordinal)
                    return &fragment;
                ++current_ordinal;
            }
            return nullptr;
        }
    };

    /**
     * @brief Closed set of steps in one retained mapped-timeline transaction.
     *
     * A captured fragment owns ordinary kernels/library nodes. Wait and publish
     * steps become backend-native graph nodes with device-owned ordering,
     * avoiding scalar stream APIs whose capture behavior differs across GPUs.
     */
    enum class GPUOrderedTimelineStepKind : std::uint8_t
    {
        CapturedFragment, ///< Ordered retained graph captured on the same device.
        WaitValue64, ///< Unsigned-GEQ wait on an aligned GPU-visible word.
        PublishValue64, ///< Fenced 64-bit write after preceding graph work.
    };

    /**
     * @brief Optional setup-time instrumentation for an ordered GPU timeline.
     *
     * The disabled policy preserves the production graph exactly. Per-step
     * timing inserts backend-native event-record nodes around each already
     * ordered step; it never adds a host callback, replay-time allocation, or
     * synchronization. Consumers reclaim a completed prior replay through
     * @ref IGPUGraphCapture::consumeOrderedTimelineTiming.
     */
    enum class GPUOrderedTimelineInstrumentation : std::uint8_t
    {
        Disabled, ///< Build the production graph without diagnostic nodes.
        PerStepEvents, ///< Record one GPU event interval per ordered step.
    };

    /**
     * @brief Lifecycle state of one instrumented ordered-timeline replay.
     */
    enum class GPUOrderedTimelineTimingState : std::uint8_t
    {
        Disabled, ///< This graph was built without timeline instrumentation.
        AwaitingLaunch, ///< Instrumented graph has not completed a launch yet.
        Pending, ///< The latest launch is still executing on the device.
        Complete, ///< @ref GPUOrderedTimelineTimingSnapshot::samples are valid.
        Failed, ///< A backend event query or elapsed-time read failed.
    };

    /**
     * @brief One device-event measurement for an ordered timeline step.
     */
    struct GPUOrderedTimelineTimingSample
    {
        std::string name; ///< Stable semantic identity copied at composition.
        GPUOrderedTimelineStepKind kind =
            GPUOrderedTimelineStepKind::CapturedFragment;
        double elapsed_ms = 0.0; ///< Device elapsed time, including device waits.
    };

    /**
     * @brief Non-blocking snapshot of the latest ordered-timeline replay.
     *
     * Samples are populated only for @ref GPUOrderedTimelineTimingState::Complete.
     * `error` is populated only for @ref GPUOrderedTimelineTimingState::Failed.
     */
    struct GPUOrderedTimelineTimingSnapshot
    {
        GPUOrderedTimelineTimingState state =
            GPUOrderedTimelineTimingState::Disabled;
        std::vector<GPUOrderedTimelineTimingSample> samples;
        std::string error;
    };

    /**
     * @brief Backend-internal lowered step for an ordered native transaction.
     *
     * Public inference code obtains these bindings through TransferEngine,
     * which validates mapped-region ownership and resolves the exact device
     * alias. Backends borrow every pointer only while constructing the parent;
     * embedded fragment graphs and signal addresses must remain alive for the
     * executable's complete lifetime.
     */
    struct GPUOrderedTimelineStep
    {
        const char *name = nullptr; ///< Stable non-empty semantic identity.
        GPUOrderedTimelineStepKind kind =
            GPUOrderedTimelineStepKind::CapturedFragment;
        const IGPUGraphCapture *capture = nullptr; ///< Fragment for CapturedFragment.
        void *signal = nullptr; ///< Aligned device alias for timeline operations.
        std::uint64_t value = 0u; ///< Positive capture-stable timeline value.

        /** @return Whether exactly the fields required by @ref kind are bound. */
        [[nodiscard]] bool valid() const noexcept
        {
            if (!name || name[0] == '\0')
                return false;
            switch (kind)
            {
            case GPUOrderedTimelineStepKind::CapturedFragment:
                return capture != nullptr && signal == nullptr && value == 0u;
            case GPUOrderedTimelineStepKind::WaitValue64:
            case GPUOrderedTimelineStepKind::PublishValue64:
                return capture == nullptr && signal != nullptr && value != 0u &&
                       (reinterpret_cast<std::uintptr_t>(signal) &
                        (alignof(std::uint64_t) - 1u)) == 0u;
            }
            return false;
        }
    };

    /// Abstract interface for GPU graph capture and replay.
    /// Abstracts over HIP Graphs (ROCm) and CUDA Graphs (NVIDIA).
    ///
    /// Lifecycle:
    ///   1. beginCapture() — stream enters capture mode
    ///   2. (launch kernels into the captured stream)
    ///   3. endCapture() — stop capture, produce graph object
    ///   4. instantiate() — compile graph into executable
    ///   5. launch() — replay the entire captured workload
    ///
    /// For subsequent decode steps on a backend that supports executable updates:
    ///   1. beginCapture() → launch kernels → endCapture()
    ///   2. tryUpdate() — in-place update of the executable
    ///   3. launch()
    ///
    /// If tryUpdate() returns NeedsReinstantiate:
    ///   1. instantiate() to create a new executable from the latest capture
    ///   2. launch()
    ///
    /// On a backend that does not support executable updates:
    ///   1. beginCapture() → launch kernels → endCapture()
    ///   2. instantiate() to replace the executable from the latest capture
    ///   3. launch()
    class IGPUGraphCapture
    {
    public:
        virtual ~IGPUGraphCapture() = default;

        /// Begin stream capture. All subsequent kernel launches on the associated
        /// stream will be recorded into a graph rather than executed.
        /// @return true on success
        virtual bool beginCapture() = 0;

        /// End stream capture and produce a graph object from the recorded operations.
        /// @return true on success
        virtual bool endCapture() = 0;

        /// Compile the captured graph into an executable.
        /// Must be called after endCapture() before the first launch().
        /// @return true on success
        virtual bool instantiate() = 0;

        /// Launch (replay) the instantiated graph executable on the associated stream.
        /// @return true on success
        virtual bool launch() = 0;

        /**
         * @brief Launch this executable on an explicit scheduler-owned stream.
         *
         * A captured graph does not retain the stream on which it was captured
         * as an execution dependency. Host-scheduled HIP transaction graphs
         * therefore replay retained child executables on one persistent stream,
         * making producer-to-consumer order a visible stream property without
         * host synchronization or per-fragment events. Implementations must
         * reject null/default streams and preserve the graph owner's device.
         *
         * The operation is logically const: replay changes device execution
         * state but never graph topology, executable identity, or ownership.
         * The default hard failure keeps test doubles and non-GPU backends
         * source-compatible while making support explicit in real backends.
         *
         * @param stream Exact non-null stream that will own this replay.
         * @return true only when the executable was submitted successfully.
         */
        [[nodiscard]] virtual bool launchOnStream(void *stream) const
        {
            (void)stream;
            return false;
        }

        /**
         * @brief Report whether this graph owner supports a device-controlled WHILE node.
         *
         * This is a static backend/runtime property.  A caller selecting a
         * fully device-resident generation architecture must require it (or a
         * backend-native equivalent) before admitting the request; this method
         * is not permission to retry through a host loop.
         */
        [[nodiscard]] virtual bool supportsDeviceControlledWhileLoop() const noexcept
        {
            return false;
        }

        /**
         * @brief Report support for a selector-gated linear transaction in WHILE.
         *
         * This is stronger than @ref supportsDeviceControlledWhileLoop: the
         * backend must validate a device selector before entering the
         * transaction and lower monotonic selector thresholds without cloning
         * one complete body per legal selector.
         */
        [[nodiscard]] virtual bool
        supportsDeviceControlledSelectorWhileLoop() const noexcept
        {
            return false;
        }

        /**
         * @brief Report support for a one-shot device-controlled transaction.
         *
         * This capability is narrower than a device-controlled loop. The
         * backend must compose an ordered fragment list exactly once and must
         * lower @ref DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero
         * without observing the predicate on the host. It is used for sparse
         * maintenance work whose cheap publisher runs at every boundary while
         * the expensive transaction runs only when device state says it is due.
         */
        [[nodiscard]] virtual bool
        supportsDeviceControlledTransaction() const noexcept
        {
            return false;
        }

        /**
         * @brief Replace this graph with one ordered device-controlled transaction.
         *
         * Every unconditional fragment executes once. Conditional fragments
         * evaluate their persistent device word after all preceding fragments
         * complete and execute once only when that word is non-zero. The
         * implementation must preserve producer-to-consumer ordering in one
         * native graph and may not read a predicate on the host, add a host
         * callback, allocate replay-time storage, or synchronize a stream.
         *
         * On success the graph is built but not instantiated. The caller must
         * retain all source captures through this call and then invoke
         * instantiate() before replay.
         *
         * @param ordered_fragments Non-empty captured fragments in execution order.
         * @return true when this object owns a complete one-shot transaction.
         */
        virtual bool buildDeviceControlledTransaction(
            std::span<const DeviceControlledLoopFragment> ordered_fragments)
        {
            (void)ordered_fragments;
            return false;
        }

        /**
         * @brief Build one retained native graph from fragments and timeline edges.
         *
         * Every step depends directly on the complete preceding frontier.
         * Implementations lower waits/publications to native device graph
         * nodes, never to a host callback, host polling thread, scalar stream
         * capture, or replay-time graph mutation. The result is built but
         * uninstantiated.
         *
         * @param ordered_steps Non-empty exact producer-to-consumer sequence.
         * @return true when this capture owns the complete parent graph.
         */
        virtual bool buildOrderedTimelineTransaction(
            std::span<const GPUOrderedTimelineStep> ordered_steps,
            GPUOrderedTimelineInstrumentation instrumentation =
                GPUOrderedTimelineInstrumentation::Disabled)
        {
            (void)ordered_steps;
            (void)instrumentation;
            return false;
        }

        /**
         * @brief Consume completed per-step timings without waiting on the GPU.
         *
         * Backends return `Pending` while the terminal event is incomplete.
         * A `Complete` snapshot consumes exactly one replay sample and returns
         * the graph to `AwaitingLaunch`. Uninstrumented implementations retain
         * this default `Disabled` result.
         */
        [[nodiscard]] virtual GPUOrderedTimelineTimingSnapshot
        consumeOrderedTimelineTiming()
        {
            return {};
        }

        /**
         * @brief Replace this graph with a device-controlled repetition of captured fragments.
         *
         * The implementation clones every element of @p ordered_body_fragments
         * into one conditional WHILE body, adds an explicit dependency from each
         * fragment to its successor, lowers any IfDeviceWordNonZero fragment
         * to a native device conditional, and appends the device predicate
         * update after the final fragment. Backend-native composers may lower internal
         * multi-stream event handoffs to equivalent direct dependency edges when
         * conditional bodies do not admit event nodes. A root wait or terminal
         * record crosses the fragment boundary and must remain a hard error. The
         * first iteration is admitted by request admission; subsequent iterations
         * are controlled exclusively by @p predicate. No D2H copy, host callback,
         * allocation, or stream synchronization is permitted in the generated
         * graph.
         *
         * On success the graph is built but not instantiated.  The caller must
         * call instantiate() exactly as it would after endCapture().
         *
         * @param ordered_body_fragments Named, captured, non-empty transaction
         *        fragments in producer-to-consumer order. Every capture pointer
         *        must remain valid through this call and identify the same
         *        backend/device context as this graph owner. Source captures may
         *        own different streams: child-graph cloning discards launch-stream
         *        identity, and parent dependencies establish transaction ordering.
         * @param predicate Persistent device controller binding.
         * @return true when this object owns a complete loop graph.
         */
        virtual bool buildDeviceControlledWhileLoop(
            std::span<const DeviceControlledLoopFragment> ordered_body_fragments,
            const DeviceControlledLoopPredicate &predicate)
        {
            (void)ordered_body_fragments;
            (void)predicate;
            return false;
        }

        /**
         * @brief Convenience overload for a transaction captured as one graph.
         *
         * This overload deliberately delegates to the ordered-fragment contract,
         * so backends have exactly one composition implementation and monolithic
         * transactions cannot acquire subtly different loop semantics.
         */
        bool buildDeviceControlledWhileLoop(
            const IGPUGraphCapture &body,
            const DeviceControlledLoopPredicate &predicate)
        {
            const DeviceControlledLoopFragment fragments[] = {{
                .name = "complete transaction",
                .capture = &body,
            }};
            return buildDeviceControlledWhileLoop(fragments, predicate);
        }

        /**
         * @brief Replace this graph with a selector-gated linear transaction in WHILE.
         *
         * The implementation first evaluates @p predicate to decide whether the
         * loop may begin. Each iteration validates @p selector_policy on device,
         * executes one producer-ordered transaction, then reevaluates @p predicate.
         * Invalid or divergent selectors execute no fragment and make the request
         * terminally unhealthy. Selector-gated fragments must form one monotonic
         * region: their thresholds increase with transaction width, after which
         * the shared verifier/publication tail appears exactly once. Ordinary
         * device-word conditional fragments remain available in that common tail.
         *
         * Fragments required at the minimum selector are unconditional. Every
         * selector-gated threshold must be greater than the configured minimum
         * and no greater than the configured maximum. This canonical shape keeps
         * executable storage O(maximum selector) instead of O(sum of selectors).
         *
         * @param ordered_body_fragments One complete maximum-width transaction.
         * @param predicate Device-owned loop continuation policy.
         * @param selector_policy Device-owned selector and fatal validation policy.
         * @return true when this object owns a built, uninstantiated parent graph.
         */
        virtual bool buildDeviceControlledSelectorWhileLoop(
            std::span<const DeviceControlledLoopFragment> ordered_body_fragments,
            const DeviceControlledLoopPredicate &predicate,
            const DeviceControlledLoopSelector &selector_policy)
        {
            (void)ordered_body_fragments;
            (void)predicate;
            (void)selector_policy;
            return false;
        }

        /**
         * @brief Return the exact stream owned by this capture lifecycle.
         *
         * Capture, replay, and post-launch completion publication must all use
         * this stream. Exposing the non-owning identity through the backend
         * interface prevents callers from accepting a second, potentially
         * different stream and silently publishing an event against work that
         * was launched elsewhere.
         *
         * @return Opaque non-null stream identity for a usable GPU capture.
         */
        [[nodiscard]] virtual void *executionStream() const noexcept = 0;

        /// Attempt to update the existing executable in-place with a newly captured graph.
        /// Call this after endCapture() on subsequent iterations where the graph topology
        /// is unchanged but kernel parameters differ.
        ///
        /// This operation is valid only when supportsExecutableUpdate() returns true.
        /// Calling it on a backend that does not advertise the operation is a programming
        /// error and returns GraphUpdateResult::Failed. Callers must instantiate the newly
        /// captured graph directly for such backends; they must not probe support by
        /// invoking this method and interpreting a runtime error.
        /// @return GraphUpdateResult indicating success, need for reinstantiation, or failure
        virtual GraphUpdateResult tryUpdate() = 0;

        /**
         * @brief Report whether this backend can update an instantiated executable in place.
         *
         * Graph recapture always produces a new graph object. Backends that return true may
         * attempt to transplant the new graph's parameters and topology into the existing
         * executable through tryUpdate(). Backends that return false require instantiate()
         * to replace the executable directly. This is a static backend capability, not a
         * transient runtime condition, so orchestration code can select one explicit
         * lifecycle without failed API calls, sticky-error clearing, or retry behavior.
         *
         * @return true when tryUpdate() is a supported active lifecycle operation.
         */
        [[nodiscard]] virtual bool supportsExecutableUpdate() const noexcept = 0;

        /// @return true if an instantiated executable exists and is ready for launch()
        virtual bool hasExecutable() const = 0;

        /**
         * @brief Return setup-observed device bytes owned by this executable.
         *
         * The graph owner records this value directly around native
         * instantiation. PerfStats and log output are diagnostics, never the
         * accounting authority. A graph without an executable reports zero;
         * a live executable may also report zero when an existing driver pool
         * satisfies the allocation.
         */
        [[nodiscard]] virtual std::size_t residentMemoryBytes() const noexcept = 0;

        /// @return Number of nodes in the last captured graph (0 if no capture done)
        virtual size_t nodeCount() const = 0;

        /**
         * @brief Recursively enumerate every kernel in the captured graph.
         *
         * This setup/diagnostic operation walks native graph metadata only. It
         * must not instantiate or launch the graph, allocate device memory,
         * synchronize execution, or transfer device data. Implementations clear
         * @p kernel_nodes before writing and fail when no captured graph exists or
         * when a native node cannot be inspected. Name-resolution failure alone
         * does not drop a node; see @ref GPUGraphKernelNodeInfo.
         *
         * @param kernel_nodes Receives one record for every recursively reachable
         *        kernel node, including kernels inside child graphs.
         * @param error Optional precise diagnostic, cleared on success.
         * @return true when the complete graph was traversed without omission.
         */
        virtual bool inspectKernelNodes(
            std::vector<GPUGraphKernelNodeInfo> &kernel_nodes,
            std::string *error = nullptr) const
        {
            kernel_nodes.clear();
            if (error)
                *error = "GPU graph backend does not implement kernel-node inspection";
            return false;
        }

        /// Destroy all captured graph and executable resources.
        /// Safe to call multiple times or on an empty object.
        virtual void reset() = 0;

        /// @return Human-readable backend name ("HIP" or "CUDA")
        virtual const char *backendName() const = 0;

    protected:
        IGPUGraphCapture() = default;
        // Non-copyable
        IGPUGraphCapture(const IGPUGraphCapture &) = delete;
        IGPUGraphCapture &operator=(const IGPUGraphCapture &) = delete;
    };

} // namespace llaminar2
