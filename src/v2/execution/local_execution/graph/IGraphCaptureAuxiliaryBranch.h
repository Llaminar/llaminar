/**
 * @file IGraphCaptureAuxiliaryBranch.h
 * @brief Typed ownership contract for work captured parallel to a GPU graph.
 *
 * A production inference executable may own a bounded auxiliary branch whose
 * work must begin at the graph's root and join before its terminal frontier.
 * Three small captured fragments supply Open, parallel work and Close. The
 * final native graph owns their fork/join edges before instantiation. This keeps that
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
    class IGPUGraphCapture;
    /**
     * @brief One cache-owned branch recorded into a complete native GPU graph.
     *
     * Implementations prepare storage and graph-only fragments during setup.
     * @ref attach decorates the final sealed graph exactly once, before it is
     * instantiated. This is the same lifecycle for direct native capture and
     * a topology-composed parent: never attach to its graph-only children.
     * The branch spans every inference wait and retires after a bounded amount
     * of work following Close, not after a complete maintenance command.
     *
     * Each instance belongs to exactly one graph cache.  This one-owner rule
     * prevents independently replayable graphs from sharing interval state,
     * even when they use the same underlying maintenance authority.
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
         * @brief Attach one complete branch to its final native execution owner.
         * @param graph Sealed, uninstantiated graph on this branch's exact GPU.
         * @return true after native fork/join assembly, without executing work.
         *
         * Failure is fatal to materialization. A second attachment or an
         * executable destination must be rejected, never silently replayed or
         * recaptured. No host callback belongs to the resulting device graph.
         */
        [[nodiscard]] virtual bool attach(IGPUGraphCapture &graph) noexcept = 0;

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
