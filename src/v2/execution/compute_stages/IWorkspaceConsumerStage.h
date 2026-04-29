/**
 * @file IWorkspaceConsumerStage.h
 * @brief Mixin interface for stages that delegate workspace to underlying kernels
 * @author David Sanftenberg
 * @date January 2026
 *
 * GEMM stages don't consume workspace directly - their underlying kernels do.
 * This interface provides the delegation pattern where a stage acts as a
 * facade for the kernel's workspace requirements.
 *
 * ## Design Pattern: Workspace Delegation
 *
 * The workspace system was designed for stages to be the workspace consumers
 * (DeviceGraphBufferManager binds workspace to stages). However, the actual memory
 * is consumed by kernels inside stages. This mixin bridges the gap:
 *
 * @code
 * // DeviceGraphBufferManager binds to stage
 * for (auto* stage : stages) {
 *     auto* consumer = dynamic_cast<IWorkspaceConsumer*>(stage);
 *     consumer->bindWorkspace(workspace);  // Stage delegates to kernel
 * }
 *
 * // During execute, kernel uses pre-bound workspace
 * stage->execute();  // Kernel uses hasWorkspace() and getBuffer()
 * @endcode
 *
 * ## Usage
 *
 * GEMM stages inherit from this mixin and implement getKernelAsWorkspaceConsumer():
 *
 * @code
 * class GEMMStage : public IComputeStage, public IWorkspaceConsumerStage
 * {
 *     IWorkspaceConsumer* getKernelAsWorkspaceConsumer() override {
 *         // Resolve through prepared-handle + device-scoped GEMM engine
 *         auto* prepared = KernelFactory::getOrCreatePreparedGemmWeights(weights_, target_device_);
 *         auto* engine = KernelFactory::getOrCreateGemmEngine(prepared);
 *         auto* gemm = engine->resolveKernel(prepared);
 *         return dynamic_cast<IWorkspaceConsumer*>(gemm);
 *     }
 * };
 * @endcode
 *
 * Note: No explicit caching needed at the stage level. KernelFactory owns both
 * prepared-weight handles (tensor+device scoped) and device-scoped GEMM engines.
 *
 * @see IWorkspaceConsumer for the base interface
 * @see DeviceGraphBufferManager::allocateDeviceWorkspace() for the binding logic
 */

#pragma once

#include "../../interfaces/IWorkspaceConsumer.h"
#include "../local_execution/device/DeviceWorkspaceManager.h"
#include "../../utils/Logger.h"
#include <stdexcept>

namespace llaminar2
{

    /**
     * @brief Mixin interface for stages that delegate workspace to underlying kernels
     *
     * Implements IWorkspaceConsumer by forwarding all calls to a kernel obtained
     * via getKernelAsWorkspaceConsumer().
     *
     * ## Kernel Resolution
     *
     * KernelFactory owns prepared GEMM handles and device-scoped GEMM engines,
     * so stages don't need their own caching. getKernelAsWorkspaceConsumer() can
     * resolve the active kernel from those shared registries.
     *
     * ## Thread Safety
     *
     * Not thread-safe. bindWorkspace() should be called during single-threaded
     * graph setup, not during parallel execution.
     */
    class IWorkspaceConsumerStage : public IWorkspaceConsumer
    {
    public:
        virtual ~IWorkspaceConsumerStage() = default;

        /**
         * @brief Get the underlying kernel that needs workspace
         *
         * Subclasses implement this to return their cached kernel pointer,
         * cast to IWorkspaceConsumer. Returns nullptr if the kernel is not
         * yet initialized or doesn't implement IWorkspaceConsumer.
         *
         * @return Pointer to kernel implementing IWorkspaceConsumer, or nullptr
         */
        virtual IWorkspaceConsumer *getKernelAsWorkspaceConsumer() = 0;

        // =========================================================================
        // IWorkspaceConsumer Implementation (Delegation to Kernel)
        // =========================================================================

        /**
         * @brief Get workspace requirements from the underlying kernel
         *
         * Delegates to the kernel's getWorkspaceRequirements().
         * Returns empty requirements if kernel is not available.
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m, int n = 0, int k = 0) const override
        {
            auto *consumer = const_cast<IWorkspaceConsumerStage *>(this)->getKernelAsWorkspaceConsumer();
            if (consumer)
            {
                return consumer->getWorkspaceRequirements(m, n, k);
            }
            LOG_TRACE("[IWorkspaceConsumerStage] No kernel available for workspace requirements");
            return WorkspaceRequirements{}; // Empty requirements
        }

        /**
         * @brief Bind workspace to the underlying kernel
         *
         * Delegates to the kernel's bindWorkspace(). Stores the workspace pointer
         * locally for hasWorkspace() and getWorkspace() queries.
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            bound_workspace_ = workspace;
            auto *consumer = getKernelAsWorkspaceConsumer();
            if (consumer)
            {
                consumer->bindWorkspace(workspace);
                LOG_DEBUG("[IWorkspaceConsumerStage] Bound workspace to kernel");
            }
            else
            {
                LOG_TRACE("[IWorkspaceConsumerStage] No kernel available, storing workspace locally");
            }
        }

        /**
         * @brief Unbind workspace from the underlying kernel
         */
        void unbindWorkspace() override
        {
            auto *consumer = getKernelAsWorkspaceConsumer();
            if (consumer)
            {
                consumer->unbindWorkspace();
            }
            bound_workspace_ = nullptr;
        }

        /**
         * @brief Check if workspace is bound
         *
         * Returns true if both:
         * 1. A workspace has been bound to this stage
         * 2. The underlying kernel reports hasWorkspace() == true
         */
        bool hasWorkspace() const override
        {
            if (!bound_workspace_)
                return false;
            auto *consumer = const_cast<IWorkspaceConsumerStage *>(this)->getKernelAsWorkspaceConsumer();
            return consumer && consumer->hasWorkspace();
        }

        /**
         * @brief Get the bound workspace manager
         */
        DeviceWorkspaceManager *getWorkspace() const override
        {
            return bound_workspace_;
        }

    protected:
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
    };

} // namespace llaminar2
