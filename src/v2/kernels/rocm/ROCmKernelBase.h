#pragma once

#include "kernels/IWorkspaceConsumer.h"
#include "utils/Logging.h"

namespace llaminar2
{

    /**
     * @brief Base class for ROCm kernels with default IWorkspaceConsumer support.
     *
     * Design Pattern: Dual-Mode Operation
     * - LEGACY MODE (hasWorkspace()=false): Kernel manages its own allocations (SLOW on ROCm!)
     * - MANAGED MODE (hasWorkspace()=true): Kernel uses pre-allocated workspace buffers
     *
     * IMPORTANT: ROCm's hipMalloc/hipFree are extremely expensive (multiple seconds).
     * All ROCm kernels MUST support workspace binding to eliminate hot-path allocations.
     *
     * Subclasses requiring workspace should:
     * 1. Override getWorkspaceRequirements() to declare buffer needs
     * 2. In execution methods, check hasWorkspace() and use getBuffer() if bound
     * 3. Fall back to internal allocation only when workspace is not bound (tests)
     */
    class ROCmKernelBase : public IWorkspaceConsumer
    {
    public:
        virtual ~ROCmKernelBase() = default;

        // IWorkspaceConsumer Interface - Default "No Workspace" Implementation

        WorkspaceRequirements getWorkspaceRequirements(
            [[maybe_unused]] int m, [[maybe_unused]] int n = 0, [[maybe_unused]] int k = 0) const override
        {
            return WorkspaceRequirements{}; // No workspace needed by default
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            workspace_ = workspace;
            if (workspace_)
            {
                LOG_DEBUG("[ROCmKernelBase] Workspace bound with " << workspace_->bufferCount() << " buffers");
            }
        }

        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

    protected:
        ROCmKernelBase() = default;

        // Helper to get a typed buffer pointer from workspace
        template <typename T>
        T *getWorkspaceBuffer(const std::string &name) const
        {
            if (!workspace_)
                return nullptr;
            return static_cast<T *>(workspace_->getBuffer(name));
        }

        DeviceWorkspaceManager *workspace_ = nullptr;
    };

} // namespace llaminar2
