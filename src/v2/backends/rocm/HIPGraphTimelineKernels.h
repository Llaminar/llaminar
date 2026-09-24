/**
 * @file HIPGraphTimelineKernels.h
 * @brief HIP device primitives used by retained graph timeline transactions.
 *
 * Current HIP batch-memory graph nodes do not provide dependable dependency
 * ordering for mapped cross-runtime pages. This boundary supplies ordinary
 * kernel nodes for one-wave system-acquire waits and system-release
 * publications, keeping the transaction immutable and device-owned.
 */

#pragma once

#ifdef HAVE_ROCM

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace llaminar2::hip_graph_timeline
{
    /**
     * @brief Splice a system-acquire wait into an active HIP stream capture.
     *
     * The helper resolves the stream's current graph frontier, appends the
     * proven one-wave wait kernel, and replaces the capture frontier with that
     * node. It is used by complete endpoint captures so mapped activation
     * ordering remains inside the same graph as ordinary model kernels.
     *
     * @param stream Exact non-default stream currently being captured.
     * @param signal Aligned HIP-visible mapping of the node-local signal word.
     * @param value Positive capture-stable unsigned-GEQ threshold.
     * @return True only when the active parent graph owns the appended node.
     */
    bool appendActiveCaptureSystemWaitValue64(
        hipStream_t stream,
        const void *signal,
        std::uint64_t value) noexcept;

    /**
     * @brief Splice a system-release publication into an active HIP capture.
     *
     * @param stream Exact non-default stream currently being captured.
     * @param signal Aligned HIP-visible mapping of the node-local signal word.
     * @param value Positive capture-stable value to publish.
     * @return True only when the active parent graph owns the appended node.
     */
    bool appendActiveCaptureSystemReleaseValue64(
        hipStream_t stream,
        void *signal,
        std::uint64_t value) noexcept;

    /**
     * @brief Append a device-owned system-acquire timeline wait to a HIP graph.
     *
     * Current HIP batch-memory graph waits can allow dependent graph work to
     * issue before a mapped peer value is visible. This one-wave kernel waits
     * on the exact mapped word and completes only after a system-scope acquire
     * observes `signal >= value`. Ordinary kernel dependency semantics then
     * gate every imported fragment root without host dispatch or polling.
     *
     * @param node Receives the newly owned graph-node identity.
     * @param graph Parent graph receiving the wait.
     * @param dependencies Exact predecessor nodes, or null when count is zero.
     * @param dependency_count Number of entries in @p dependencies.
     * @param signal Aligned GPU-visible mapped timeline word.
     * @param value Positive capture-stable threshold to await.
     * @return HIP runtime status from validation or graph-node construction.
     */
    hipError_t addSystemWaitValue64Node(
        hipGraphNode_t *node,
        hipGraph_t graph,
        const hipGraphNode_t *dependencies,
        std::size_t dependency_count,
        const void *signal,
        std::uint64_t value) noexcept;

    /**
     * @brief Append one system-scope 64-bit publication to a HIP graph.
     *
     * The one-thread kernel is ordered after every supplied dependency and
     * publishes only after preceding graph work is globally visible. Kernel
     * arguments are copied into the graph at construction, so @p signal must
     * remain mapped while the executable exists but the host argument storage
     * need not remain alive.
     *
     * @param node Receives the newly owned graph-node identity.
     * @param graph Parent graph receiving the publication.
     * @param dependencies Exact predecessor nodes, or null when count is zero.
     * @param dependency_count Number of entries in @p dependencies.
     * @param signal Aligned GPU-visible mapped timeline word.
     * @param value Positive capture-stable value to publish.
     * @return HIP runtime status from validation or graph-node construction.
     */
    hipError_t addSystemReleaseValue64Node(
        hipGraphNode_t *node,
        hipGraph_t graph,
        const hipGraphNode_t *dependencies,
        std::size_t dependency_count,
        void *signal,
        std::uint64_t value) noexcept;
} // namespace llaminar2::hip_graph_timeline

#endif // HAVE_ROCM
