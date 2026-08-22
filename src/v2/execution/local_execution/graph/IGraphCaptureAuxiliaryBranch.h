/**
 * @file IGraphCaptureAuxiliaryBranch.h
 * @brief Typed ownership contract for work captured parallel to a GPU graph.
 *
 * A production inference executable may own a bounded auxiliary branch whose
 * work must begin at the graph's root and join before its terminal frontier.
 * The branch is recorded on an explicit auxiliary stream, while the primary
 * graph stream supplies the fork and join edges.  This interface keeps that
 * lifecycle backend-neutral and makes the branch a persistent part of graph
 * cache identity instead of an opportunistic host launch.
 */

#pragma once

#include "backends/DeviceId.h"

#include <functional>
#include <memory>
#include <string_view>

namespace llaminar2
{
    /**
     * @brief One cache-owned branch recorded into a complete native GPU graph.
     *
     * Implementations allocate all events, device storage, and auxiliary
     * streams before @ref recordFork is called.  `recordFork()` and
     * `recordJoin()` execute only while the primary stream is inside native
     * capture.  They may enqueue bounded capture-safe device work and event
     * edges, but may never allocate, synchronize, invoke a host callback, or
     * transfer a payload through a host API.
     *
     * Each instance belongs to exactly one graph cache.  This one-owner rule
     * prevents two independently replayable graphs from sharing mutable event
     * generations even when they use the same underlying maintenance authority.
     */
    class IGraphCaptureAuxiliaryBranch
    {
    public:
        virtual ~IGraphCaptureAuxiliaryBranch() = default;

        IGraphCaptureAuxiliaryBranch(
            const IGraphCaptureAuxiliaryBranch &) = delete;
        IGraphCaptureAuxiliaryBranch &operator=(
            const IGraphCaptureAuxiliaryBranch &) = delete;

        /** @return Stable authority identity embedded in graph-cache identity. */
        [[nodiscard]] virtual const void *authorityIdentity() const noexcept = 0;

        /** @return Exact GPU on which both primary and auxiliary work execute. */
        [[nodiscard]] virtual DeviceId device() const noexcept = 0;

        /** @return Stable non-empty diagnostic name for this branch. */
        [[nodiscard]] virtual std::string_view name() const noexcept = 0;

        /**
         * @brief Record the root fork and bounded auxiliary device work.
         * @param primary_capture_stream Exact non-null stream currently being captured.
         * @return true after the complete fork-side DAG was recorded.
         */
        [[nodiscard]] virtual bool recordFork(
            void *primary_capture_stream) noexcept = 0;

        /**
         * @brief Join the auxiliary terminal back to the primary graph frontier.
         * @param primary_capture_stream Same exact stream passed to @ref recordFork.
         * @return true after the terminal event edge was recorded.
         */
        [[nodiscard]] virtual bool recordJoin(
            void *primary_capture_stream) noexcept = 0;

    protected:
        IGraphCaptureAuxiliaryBranch() = default;
    };

    /**
     * @brief Cold factory and immutable authority identity for one graph branch.
     *
     * A forward graph supplies this descriptor on every execution.  The graph
     * cache invokes @ref create only during first materialization, retains the
     * resulting branch for the executable lifetime, and compares
     * @ref authority_identity on later calls.  An empty factory means that the
     * graph has no auxiliary branch; it is never interpreted as permission to
     * launch equivalent work from the host.
     */
    struct GraphCaptureAuxiliaryBranchFactory
    {
        const void *authority_identity = nullptr; ///< Stable model-lifetime owner.
        DeviceId device = DeviceId::invalid();    ///< Exact branch GPU.
        std::function<std::unique_ptr<IGraphCaptureAuxiliaryBranch>()>
            create; ///< Cold, allocation-permitted construction callback.

        /** @return true only when all factory identity fields are complete. */
        [[nodiscard]] bool valid() const noexcept
        {
            return authority_identity != nullptr && device.is_gpu() &&
                   static_cast<bool>(create);
        }

        /** @return true when no partial branch configuration was supplied. */
        [[nodiscard]] bool empty() const noexcept
        {
            return authority_identity == nullptr && !device.is_valid() &&
                   !create;
        }
    };
} // namespace llaminar2
