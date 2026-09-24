/**
 * @file KVCacheTestWorkspace.h
 * @brief RAII binding for integration tests of workspace-owned GPU KV caches.
 *
 * Production GPU KV caches receive conversion storage from the graph workspace
 * planner.  Integration tests that instantiate a cache directly must reproduce
 * that ownership contract explicitly; allowing the cache to allocate private
 * scratch would hide hot-path allocations and exercise a non-production path.
 */
#pragma once

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"

#include <memory>
#include <stdexcept>

namespace llaminar2::test
{
    /**
     * @brief Own and bind the complete workspace declared by a cache.
     *
     * The cache is borrowed and must outlive this binding.  Destruction unbinds
     * the workspace before releasing its device allocation, matching graph
     * teardown order and preventing a dangling workspace owner.
     */
    class KVCacheTestWorkspaceBinding
    {
    public:
        /**
         * @brief Bind the consumer's complete declared workspace before capture.
         * @param consumer Cache or associated attention consumer, borrowed.
         * @param device Device owning the workspace allocation.
         * @param query_rows Immutable attention geometry; caches ignore this hint.
         * @throws std::runtime_error If allocation or binding is incomplete.
         */
        KVCacheTestWorkspaceBinding(
            IWorkspaceConsumer &consumer,
            DeviceId device,
            int query_rows = 1)
            : consumer_(&consumer)
        {
            const WorkspaceRequirements requirements =
                consumer.getWorkspaceRequirements(
                    query_rows,
                    /*n=*/0,
                    /*k=*/0);
            workspace_ = std::make_unique<DeviceWorkspaceManager>(
                device,
                requirements.total_bytes_with_alignment() + 4096);
            if (!workspace_->allocate(requirements))
            {
                throw std::runtime_error(
                    "Failed to allocate the GPU KV-cache test workspace");
            }
            consumer.bindWorkspace(workspace_.get());
            if (!consumer.hasWorkspace())
            {
                throw std::runtime_error(
                    "GPU KV cache rejected its declared test workspace");
            }
        }

        /** @brief Unbind borrowed pointers before destroying their storage. */
        ~KVCacheTestWorkspaceBinding()
        {
            if (consumer_)
                consumer_->unbindWorkspace();
        }

        KVCacheTestWorkspaceBinding(const KVCacheTestWorkspaceBinding &) = delete;
        KVCacheTestWorkspaceBinding &operator=(
            const KVCacheTestWorkspaceBinding &) = delete;
        KVCacheTestWorkspaceBinding(KVCacheTestWorkspaceBinding &&) = delete;
        KVCacheTestWorkspaceBinding &operator=(
            KVCacheTestWorkspaceBinding &&) = delete;

    private:
        IWorkspaceConsumer *consumer_ = nullptr;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
    };
} // namespace llaminar2::test
