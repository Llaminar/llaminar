/**
 * @file CUDAGraphCapture.h
 * @brief CUDA ownership, composition, replay, and metadata inspection for native graphs.
 *
 * The capture owns CUDA graph and executable handles but borrows one exact,
 * non-default execution stream. It also exposes read-only recursive kernel
 * inventory so orchestration diagnostics can attribute captured production work
 * without replaying a second path or involving device memory.
 */

#pragma once

#ifdef HAVE_CUDA

#include "../IGPUGraphCapture.h"
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Splice one mapped 64-bit wait into an active CUDA stream capture.
     *
     * CUDA conditional nodes must remain children of the top-level captured
     * graph, so a complete ExpertOverlay transaction cannot be assembled later
     * from child graphs. This helper inserts the native batch-memory wait at
     * the stream's exact current dependency frontier and makes it the frontier
     * for subsequently captured work.
     *
     * @param stream Exact non-default stream currently being captured.
     * @param signal Aligned CUDA-visible mapping of the node-local signal word.
     * @param value Positive capture-stable unsigned-GEQ threshold.
     * @return True only when the node was appended to the active parent graph.
     */
    bool appendCUDAActiveCaptureTimelineWait64(
        cudaStream_t stream,
        void *signal,
        std::uint64_t value) noexcept;

    /**
     * @brief Splice one system-release publication into an active CUDA capture.
     *
     * The publication kernel is ordered after the current stream frontier,
     * performs a system fence, stores the mapped timeline value, and becomes
     * the sole dependency of later captured work. This preserves packet-byte
     * visibility across CUDA, ROCm, and CPU endpoints without a host callback.
     *
     * @param stream Exact non-default stream currently being captured.
     * @param signal Aligned CUDA-visible mapping of the node-local signal word.
     * @param value Positive capture-stable value to publish.
     * @return True only when the node was appended to the active parent graph.
     */
    bool appendCUDAActiveCaptureTimelinePublish64(
        cudaStream_t stream,
        void *signal,
        std::uint64_t value) noexcept;

#if CUDART_VERSION >= 12030
    /**
     * @brief Body identity for one native CUDA conditional transaction.
     *
     * CUDA executes body zero when the device-owned conditional handle is
     * non-zero. An `IfElse` transaction additionally executes body one when it
     * is zero. Naming those positions prevents call sites from depending on an
     * unexplained integer convention.
     */
    enum class CUDAActiveCaptureConditionalBranch : std::uint8_t
    {
        IfNonZero = 0, ///< Execute when the predicate publishes a non-zero value.
        ElseZero = 1,  ///< Execute when the predicate publishes zero.
    };

    /**
     * @brief Physical body shape of one native CUDA conditional node.
     *
     * `IfOnly` is useful when an economical root kernel can retire immediately
     * for the non-selected regime and only the expensive path needs a child
     * graph. `IfElse` reserves two mutually exclusive child graphs. Making the
     * shape explicit prevents callers from constructing an accidental empty
     * ELSE body or paying its dispatch cost without declaring that policy.
     */
    enum class CUDAActiveCaptureConditionalKind : std::uint8_t
    {
        IfOnly, ///< One non-zero body; zero executes no child graph.
        IfElse, ///< Non-zero and zero each own a non-empty body.
    };

    /**
     * @brief Fluently append one device-selected conditional transaction to an active stream capture.
     *
     * The builder is the single authority for splicing a manually constructed
     * CUDA conditional node into an otherwise ordinary stream capture. It
     * snapshots the stream's current dependency frontier, lends its condition
     * handle to exactly one scoped device publisher, owns the declared ordered
     * branch tails, and replaces the stream frontier with every completed
     * transaction tail. The publisher is normally an existing device-state
     * producer, so conditional control does not add a second tiny kernel to the
     * graph. Subsequent captured work therefore waits for the selected body and
     * any explicitly concurrent guarded root without a host callback, state
     * download, stream synchronization, or alternate replay.
     *
     * Construction and graph-node insertion happen only while publishing a new
     * executable. The object owns no CUDA graph resource: every handle belongs
     * to the active parent capture and remains valid until that parent graph is
     * destroyed. A failed operation poisons this builder and must make the
     * enclosing capture fail; there is deliberately no rollback or eager
     * substitute.
     */
    class CUDAActiveCaptureConditional final
    {
    public:
        /**
         * @brief Bind to the exact dependency frontier of an active CUDA stream capture.
         * @param stream Explicit non-default stream currently being captured.
         * @param kind Declared one-body or two-body conditional topology.
         */
        CUDAActiveCaptureConditional(
            cudaStream_t stream,
            CUDAActiveCaptureConditionalKind kind);

        CUDAActiveCaptureConditional(const CUDAActiveCaptureConditional &) = delete;
        CUDAActiveCaptureConditional &operator=(
            const CUDAActiveCaptureConditional &) = delete;
        CUDAActiveCaptureConditional(CUDAActiveCaptureConditional &&) = delete;
        CUDAActiveCaptureConditional &operator=(
            CUDAActiveCaptureConditional &&) = delete;

        /** @return True while the builder can accept its predicate publisher. */
        [[nodiscard]] bool ready() const noexcept;

        /**
         * @brief Publish the condition through exactly one captured device producer.
         *
         * The callback receives the CUDA-owned handle and must enqueue one or
         * more operations on the builder's exact stream that publish the logical
         * decision. Handles are reset to zero by CUDA at every replay; a producer
         * may therefore omit `cudaGraphSetConditional` for that default and must
         * call it exactly once when publishing a non-zero decision. Returning
         * false poisons the transaction. On success, the builder authenticates
         * that capture advanced beyond its incoming frontier. The caller may then
         * enqueue format preparation before calling @ref beginBranches, which
         * anchors every declared body after the complete current dependency
         * frontier.
         *
         * Keeping the handle scoped to this callback makes it impossible for a
         * caller to retain it and publish the branch outside the graph-owned
         * producer/consumer lifetime.
         *
         * @tparam Publisher Callable accepting `cudaGraphConditionalHandle` and
         *         returning a bool-like submission result.
         * @param publisher Device producer submission callable.
         * @return True when the predicate producer has been structurally bound.
         */
        template <typename Publisher>
        bool publishPredicate(Publisher &&publisher)
        {
            if (!ready())
                return fail(
                    "CUDA active-capture conditional predicate was published out of lifecycle order");
            if (!static_cast<bool>(
                    std::forward<Publisher>(publisher)(condition_)))
            {
                return fail(
                    "CUDA active-capture conditional predicate producer rejected submission");
            }
            return bindPublishedPredicate();
        }

        /**
         * @brief Materialize all declared IF bodies after input preparation.
         *
         * The exact stream may enqueue conversion or layout kernels after the
         * predicate producer. Calling this method only when branch inputs are
         * complete makes those kernels structural dependencies of every body.
         *
         * @return True when every declared body is ready to receive work.
         */
        bool beginBranches();

        /**
         * @brief Fork one guarded parent kernel beside an IF-only child graph.
         *
         * Both the supplied root kernel and the conditional node depend on the
         * exact current stream frontier. The root kernel must use the same
         * device-owned predicate inputs and retire before mutating output when
         * the IF body owns the transaction. Commit joins both siblings before
         * subsequent captured work. This shape hides conditional scheduling
         * underneath useful short-path work without introducing a race, host
         * decision, extra stream, or segmented graph.
         *
         * @param params Complete immutable launch parameters for the guarded
         *        parent-graph kernel.
         * @param semantic_name Stable producer role used in fatal diagnostics.
         * @return True when the guarded root and IF body are ready to be filled.
         */
        bool beginBranchesWithConcurrentRootKernel(
            const cudaKernelNodeParams &params,
            const char *semantic_name);

        /** @return True after predicate publication and before branch sealing. */
        [[nodiscard]] bool branchesReady() const noexcept
        {
            return state_ == State::BuildingBranches;
        }

        /**
         * @brief Append one kernel after the current tail of a selected branch.
         * @param branch Device-condition branch receiving the kernel.
         * @param params Complete immutable kernel launch parameters.
         * @param semantic_name Stable producer role used in fatal diagnostics.
         * @return True when the branch now owns the kernel node.
         */
        bool appendKernel(
            CUDAActiveCaptureConditionalBranch branch,
            const cudaKernelNodeParams &params,
            const char *semantic_name);

        /**
         * @brief Seal non-empty branches and publish the conditional as the stream tail.
         * @return True only when subsequent captured operations are ordered after
         *         the complete conditional transaction.
         */
        bool commit();

        /** @return First precise construction failure, or an empty string. */
        [[nodiscard]] const std::string &error() const noexcept { return error_; }

    private:
        enum class State : std::uint8_t
        {
            Failed,
            AwaitingPredicate,
            AwaitingBranches,
            BuildingBranches,
            Committed,
        };

        /** @brief Convert a typed branch into the CUDA-owned body-array index. */
        [[nodiscard]] static constexpr std::size_t branchIndex(
            CUDAActiveCaptureConditionalBranch branch) noexcept
        {
            return static_cast<std::size_t>(branch);
        }

        /** @brief Record one fatal CUDA operation and make later calls inert. */
        bool fail(const char *operation, cudaError_t status);

        /** @brief Record one fatal lifecycle/argument violation. */
        bool fail(const std::string &message);

        /** @brief Authenticate the publisher frontier and materialize IF bodies. */
        bool bindPublishedPredicate();

        /** @brief Copy and authenticate the exact current capture frontier. */
        bool queryCurrentFrontier(
            std::vector<cudaGraphNode_t> &dependencies,
            std::vector<cudaGraphEdgeData> &edge_data,
            const char *operation);

        /** @brief Materialize declared bodies after an authenticated frontier. */
        bool materializeBranches(
            const std::vector<cudaGraphNode_t> &dependencies,
            const std::vector<cudaGraphEdgeData> &edge_data);

        cudaStream_t stream_ = nullptr; ///< Borrowed exact active-capture stream.
        cudaGraph_t parent_graph_ = nullptr; ///< Borrowed graph owned by the capture.
        unsigned long long capture_id_ = 0; ///< Immutable stream-capture identity.
        cudaGraphConditionalHandle condition_ = 0; ///< Parent-owned device condition.
        cudaGraphNode_t conditional_node_ = nullptr; ///< Complete conditional tail.
        cudaGraphNode_t concurrent_root_node_ = nullptr; ///< Guarded sibling root.
        std::array<cudaGraph_t, 2> branch_graphs_{}; ///< CUDA-owned IF bodies.
        std::array<cudaGraphNode_t, 2> branch_tails_{}; ///< Ordered branch tails.
        std::array<std::size_t, 2> branch_node_counts_{}; ///< Non-empty proof.
        std::vector<cudaGraphNode_t> incoming_dependencies_; ///< Captured frontier.
        std::vector<cudaGraphEdgeData> incoming_edge_data_; ///< Frontier semantics.
        std::string error_; ///< First fatal construction diagnostic.
        CUDAActiveCaptureConditionalKind kind_ =
            CUDAActiveCaptureConditionalKind::IfElse; ///< Declared body topology.
        State state_ = State::Failed; ///< Explicit fluent lifecycle.
    };
#endif

    /// CUDA Graph capture/replay implementation for NVIDIA GPUs.
    ///
    /// Wraps cudaGraph_t / cudaGraphExec_t lifecycle. The stream is NOT owned;
    /// it must outlive this object. The CUDA ordinal is part of the capture's
    /// immutable ownership identity so capture, composition, replay, update,
    /// and destruction never depend on ambient thread-local CUDA state.
    class CUDAGraphCapture : public IGPUGraphCapture
    {
    public:
        /// @param stream The explicit CUDA stream to capture on. Must remain
        ///               valid for the lifetime of this object.
        /// @param device_ordinal CUDA device that owns @p stream and every graph
        ///                       resource composed into this capture.
        CUDAGraphCapture(cudaStream_t stream, int device_ordinal);
        ~CUDAGraphCapture() override;

        // Move-only (graph handles are not copyable)
        CUDAGraphCapture(CUDAGraphCapture &&other) noexcept;
        CUDAGraphCapture &operator=(CUDAGraphCapture &&other) noexcept;

        bool beginCapture() override;
        bool endCapture() override;
        bool instantiate() override;
        bool launch() override;
        [[nodiscard]] bool launchOnStream(void *stream) const override;
        [[nodiscard]] bool supportsDeviceControlledWhileLoop() const noexcept override
        {
#if CUDART_VERSION >= 12030
            return true;
#else
            return false;
#endif
        }
        [[nodiscard]] bool
        supportsDeviceControlledSelectorWhileLoop() const noexcept override
        {
#if CUDART_VERSION >= 12030
            return true;
#else
            return false;
#endif
        }
        [[nodiscard]] bool
        supportsDeviceControlledTransaction() const noexcept override
        {
#if CUDART_VERSION >= 12030
            return true;
#else
            return false;
#endif
        }
        bool buildDeviceControlledTransaction(
            std::span<const DeviceControlledLoopFragment> ordered_fragments) override;
        bool buildOrderedTimelineTransaction(
            std::span<const GPUOrderedTimelineStep> ordered_steps,
            GPUOrderedTimelineInstrumentation instrumentation =
                GPUOrderedTimelineInstrumentation::Disabled) override;
        [[nodiscard]] GPUOrderedTimelineTimingSnapshot
        consumeOrderedTimelineTiming() override;
        using IGPUGraphCapture::buildDeviceControlledWhileLoop;
        bool buildDeviceControlledWhileLoop(
            std::span<const DeviceControlledLoopFragment> ordered_body_fragments,
            const DeviceControlledLoopPredicate &predicate) override;
        bool buildDeviceControlledSelectorWhileLoop(
            std::span<const DeviceControlledLoopFragment> ordered_body_fragments,
            const DeviceControlledLoopPredicate &predicate,
            const DeviceControlledLoopSelector &selector_policy) override;
        [[nodiscard]] void *executionStream() const noexcept override
        {
            return static_cast<void *>(stream_);
        }
        GraphUpdateResult tryUpdate() override;
        [[nodiscard]] bool supportsExecutableUpdate() const noexcept override { return true; }
        bool hasExecutable() const override;
        [[nodiscard]] std::size_t residentMemoryBytes() const noexcept override
        {
            return resident_memory_bytes_;
        }
        size_t nodeCount() const override;
        bool inspectKernelNodes(
            std::vector<GPUGraphKernelNodeInfo> &kernel_nodes,
            std::string *error = nullptr) const override;
        void reset() override;
        const char *backendName() const override { return "CUDA"; }

        /// @return The underlying CUDA graph (may be nullptr)
        cudaGraph_t graph() const { return graph_; }
        /// @return The underlying CUDA graph executable (may be nullptr)
        cudaGraphExec_t executable() const { return exec_; }
        /// @return The immutable CUDA device owning this capture.
        int deviceOrdinal() const noexcept { return device_ordinal_; }

    private:
        /**
         * @brief Select the immutable owner before using a CUDA graph resource.
         *
         * CUDA runtime device selection is thread-local. LocalTP submits graph
         * operations for several devices from one host thread, and same-stream
         * event edges are correctly elided. Consequently no neighbouring API
         * call can be relied upon to select this graph's device. Every graph
         * operation establishes its own context explicitly.
         */
        bool activateOwner(const char *operation) const noexcept;

        /** @brief One persistent CUDA event pair embedded around a timeline step. */
        struct OrderedTimelineTimingEvents
        {
            std::string name; ///< Stable step identity owned by this graph.
            GPUOrderedTimelineStepKind kind =
                GPUOrderedTimelineStepKind::CapturedFragment;
            cudaEvent_t start = nullptr; ///< Recorded before the timed step.
            cudaEvent_t stop = nullptr; ///< Recorded after the complete step frontier.
        };

        /** @brief Destroy every setup-owned timing event without touching graph work. */
        void destroyOrderedTimelineTimingEvents() noexcept;

        cudaStream_t stream_ = nullptr;       ///< Non-owned stream
        int device_ordinal_ = -1;             ///< Immutable owner of stream/graph
        cudaGraph_t graph_ = nullptr;         ///< Captured graph (owned)
        cudaGraphExec_t exec_ = nullptr;      ///< Instantiated executable (owned)
        size_t node_count_ = 0;               ///< Cached node count from last capture
        std::size_t resident_memory_bytes_ = 0u; ///< Setup-observed opaque driver VRAM.
        std::vector<OrderedTimelineTimingEvents>
            ordered_timeline_timing_events_; ///< Empty on the uninstrumented path.
        mutable bool ordered_timeline_timing_pending_ = false; ///< Latest replay awaits non-blocking collection.
    };

} // namespace llaminar2

#endif // HAVE_CUDA
