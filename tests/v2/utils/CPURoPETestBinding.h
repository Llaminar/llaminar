/**
 * @file CPURoPETestBinding.h
 * @brief Admitted angle workspace binding for direct CPU RoPE kernel tests.
 *
 * RoPEStage binds its graph's admitted `cpu_rope_angles` workspace to its
 * stage-owned kernel. Direct kernel tests have no stage, so this fixture admits
 * the same declaration through PhysicalMemoryAuthority and binds it to one
 * kernel for that kernel's lifetime. Kernels without angle planes (Q16_1) are
 * left unbound. Construct the binding next to its kernel, never per call.
 */
#pragma once

#include "kernels/cpu/ops/CPURoPEKernelT.h"
#include "kernels/rope/RoPEWorkspaceContract.h"
#include "CPUProjectionTestWorkspace.h"

#include <memory>
#include <type_traits>

namespace llaminar2::test
{
    /** @brief One kernel's admitted angle workspace, sized for a test envelope. */
    class CPURoPETestBinding final
    {
    public:
        /// Largest row count any direct RoPE kernel test rotates in one call.
        static constexpr int kDefaultRows = 4096;
        /// Widest physical head any direct RoPE kernel test uses.
        static constexpr int kDefaultHeadDim = 512;

        /**
         * @brief Admit and bind the angle planes for one kernel.
         * @tparam Kernel A CPURoPEKernelT specialization.
         * @param kernel Kernel that borrows the planes until this binding retires.
         * @param rows Maximum rows of any call made through this kernel.
         * @param head_dim Maximum physical head width of those calls.
         */
        template <class Kernel>
        explicit CPURoPETestBinding(Kernel &kernel, int rows = kDefaultRows, int head_dim = kDefaultHeadDim)
        {
            if constexpr (std::is_base_of_v<CPURoPEAngleWorkspaceConsumer, Kernel>)
            {
                workspace_ = std::make_unique<CPUProjectionTestWorkspace>(
                    cpu_rope_workspace::requirements({.graph_rows = rows, .head_dim = head_dim}));
                kernel.bindWorkspace(workspace_->get());
            }
        }

        /** @return The bound workspace, or null for a kernel without angle planes. */
        DeviceWorkspaceManager *get() const noexcept { return workspace_ ? workspace_->get() : nullptr; }

    private:
        std::unique_ptr<CPUProjectionTestWorkspace> workspace_; ///< Admitted planes; outlives every call.
    };

    /**
     * @brief Admit a stage's own workspace declaration and bind it to the stage.
     *
     * Graph construction does this through the WorkspaceAllocator; direct
     * stage tests use the same declaration, so a stage-owned CPU RoPE kernel
     * executes against exactly the planes it declared.
     */
    class StageWorkspaceTestBinding final
    {
    public:
        /**
         * @brief Admit the stage's requirements for `rows` and bind them.
         * @tparam Stage A workspace-consuming compute stage.
         * @param stage Stage whose kernel borrows the workspace.
         * @param rows Maximum rows of any execute() made through the stage.
         */
        template <class Stage>
        StageWorkspaceTestBinding(Stage &stage, int rows)
            : workspace_(std::make_unique<CPUProjectionTestWorkspace>(stage.getWorkspaceRequirements(rows)))
        {
            stage.bindWorkspace(workspace_->get());
        }

    private:
        std::unique_ptr<CPUProjectionTestWorkspace> workspace_; ///< Admitted stage workspace.
    };
}
