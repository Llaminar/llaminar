/**
 * @file CPUKernelBase.h
 * @brief Base class for all CPU kernel implementations
 * @author David Sanftenberg
 * @date 2025-11-22
 *
 * ## Workspace Management
 *
 * CPUKernelBase implements IWorkspaceConsumer with default no-op behavior.
 * Subkernels that need workspace buffers override the relevant methods:
 *
 * - `getWorkspaceRequirements()`: Declare buffer needs
 * - `bindWorkspace()`: Receive pre-allocated buffers
 * - `hasWorkspace()` / `getWorkspace()`: Query workspace state
 *
 * Kernels that don't need workspace (e.g., in-place operations like RMSNorm,
 * SwiGLU, RoPE) inherit the default behavior and require no changes.
 *
 * Example for a kernel that needs workspace:
 * ```cpp
 * class QuantisedGemmKernel : public ITensorGemm, public CPUKernelBase {
 * public:
 *     WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const override {
 *         WorkspaceRequirements req;
 *         req.buffers.push_back({"q8_activation", m * ((k + 31) / 32) * 36, 64});
 *         return req;
 *     }
 *     void bindWorkspace(DeviceWorkspaceManager* ws) override {
 *         workspace_ = ws;
 *     }
 *     // ... use workspace_ in multiply() methods
 * };
 * ```
 */

#pragma once

#include "../../interfaces/IWorkspaceConsumer.h"
#include "../../execution/local_execution/device/WorkspaceDescriptor.h"

namespace llaminar2
{
    // Forward declaration
    class DeviceWorkspaceManager;

    /**
     * @brief Base class for all CPU kernel implementations
     *
     * Provides common functionality and interface for CPU-based kernels.
     * All CPU kernel classes should inherit from this base.
     *
     * Implements IWorkspaceConsumer with default "no workspace needed" behavior.
     * Subkernels that require workspace buffers override the relevant methods.
     */
    class CPUKernelBase : public IWorkspaceConsumer
    {
    public:
        virtual ~CPUKernelBase() = default;

        // =========================================================================
        // IWorkspaceConsumer - Default "No Workspace" Implementation
        // =========================================================================

        /**
         * @brief Get workspace requirements (default: none)
         *
         * Override in subkernels that need workspace buffers.
         *
         * @param m Number of rows (batch size / sequence length)
         * @param n Number of output features
         * @param k Number of input features
         * @return Empty WorkspaceRequirements by default
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m, int n = 0, int k = 0) const override
        {
            (void)m;
            (void)n;
            (void)k;
            return WorkspaceRequirements{}; // No workspace needed by default
        }

        /**
         * @brief Bind workspace manager (default: store reference)
         *
         * Override if subkernel needs custom binding logic.
         *
         * @param workspace Pointer to workspace manager (may be nullptr to unbind)
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            workspace_ = workspace;
        }

        /**
         * @brief Check if workspace is bound
         *
         * @return true if bindWorkspace() was called with non-null pointer
         */
        bool hasWorkspace() const override
        {
            return workspace_ != nullptr;
        }

        /**
         * @brief Get bound workspace manager
         *
         * @return Pointer to bound workspace, or nullptr if not bound
         */
        DeviceWorkspaceManager *getWorkspace() const override
        {
            return workspace_;
        }

    protected:
        CPUKernelBase() = default;

        /// Bound workspace manager (nullptr = legacy mode with internal allocations)
        DeviceWorkspaceManager *workspace_ = nullptr;
    };

} // namespace llaminar2
