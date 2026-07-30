#pragma once

#include <stdexcept>

#include "backends/ExplicitGPUStream.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "utils/Logger.h"

namespace llaminar2
{

    inline bool validateROCmWorkspaceBinding(
        DeviceWorkspaceManager *workspace,
        int expected_rocm_device,
        const char *kernel_name,
        bool require_allocated = true)
    {
        if (!workspace)
        {
            LOG_ERROR("[" << kernel_name << "] Workspace not bound. Call bindWorkspace() first.");
            return false;
        }

        if (require_allocated && !workspace->isAllocated())
        {
            LOG_ERROR("[" << kernel_name << "] Workspace is bound but not allocated");
            return false;
        }

        if (expected_rocm_device < 0)
        {
            LOG_ERROR("[" << kernel_name << "] Invalid expected ROCm device index: "
                          << expected_rocm_device);
            return false;
        }

        const DeviceId workspace_device = workspace->device();
        if (!workspace_device.is_rocm() || workspace_device.rocm_ordinal() != expected_rocm_device)
        {
            LOG_ERROR("[" << kernel_name << "] Workspace device mismatch: workspace="
                          << workspace_device.to_string() << " expected=ROCm:" << expected_rocm_device
                          << " workspace_ptr=" << static_cast<void *>(workspace));
            return false;
        }

        return true;
    }

    /**
     * @brief Base class for ROCm kernels with default IWorkspaceConsumer support.
     *
     * Design Pattern: Dual-Mode Operation
     * - LEGACY MODE (hasWorkspace()=false): Kernel manages its own allocations (SLOW on ROCm!)
     * - MANAGED MODE (hasWorkspace()=true): Kernel uses pre-allocated workspace buffers
     *
     * Device Context Support (Phase 4):
     * - When device context is set, kernels can use shared hipBLAS handles and streams
     * - When no context, kernels manage their own handles (legacy behavior)
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
                LOG_TRACE("[ROCmKernelBase] Workspace bound with " << workspace_->bufferCount() << " buffers");
            }
        }

        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

        // =========================================================================
        // Device Context Support (Phase 4)
        // =========================================================================

        /**
         * @brief Set the device context for this kernel
         *
         * When a device context is set, the kernel can use shared library handles
         * (hipBLAS, etc.) and streams from the context instead of creating
         * its own. This reduces resource usage and improves multi-kernel coordination.
         *
         * @param ctx Device context (owned by GPUDeviceContextPool, not this kernel)
         */
        void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }

        /**
         * @brief Get the currently bound device context
         * @return Device context, or nullptr if not bound
         */
        IWorkerGPUContext *deviceContext() const { return device_ctx_; }

        /**
         * @brief Check if a device context is bound
         * @return true if setDeviceContext() was called with a non-null context
         */
        bool hasDeviceContext() const { return device_ctx_ != nullptr; }

        // =========================================================================
        // GPU Stream Management (Canonical Location)
        // =========================================================================

        /**
         * @brief Set the GPU stream for kernel dispatch
         *
         * All GPU kernel work MUST be dispatched on an explicit stream.
         * The default HIP stream (stream 0) causes race conditions with
         * the executor's event-based coherence tracking.
         *
         * @param stream Validated non-null HIP stream binding.
         */
        void bindGPUStream(ExplicitGPUStream stream) { gpu_stream_ = stream.get(); }

        /**
         * @brief Explicitly release the borrowed execution-stream binding.
         */
        void clearGPUStreamBinding() noexcept { gpu_stream_ = nullptr; }

        /**
         * @brief Report whether a caller explicitly selected the launch stream.
         *
         * Consumers use this query to verify that the current transaction
         * explicitly named its launch stream.
         *
         * @return true only when bindGPUStream() received a non-null stream.
         */
        [[nodiscard]] bool hasExplicitGPUStream() const noexcept
        {
            return gpu_stream_ != nullptr;
        }

        /**
         * @brief Get the non-null GPU stream for kernel dispatch.
         *
         * A kernel without an explicit binding is not launchable. Device
         * contexts own streams but do not silently lend one to a kernel: the
         * executor must name the exact producer stream for every invocation.
         *
         * @return hipStream_t cast to a non-null opaque pointer.
         * @throws std::runtime_error when no owned stream exists.
         */
        void *getStream() const
        {
            if (gpu_stream_)
                return gpu_stream_;
            throw std::runtime_error(
                "[ROCmKernelBase] GPU execution requires an explicit non-null stream binding");
        }

        /**
         * @brief Get the GPU stream, throwing if none is set
         *
         * Use this at the top of GPU kernel execution methods to enforce
         * that a stream has been explicitly bound. Running on the default
         * stream (nullptr) causes race conditions with event-based coherence.
         *
         * @param kernel_name Name for error messages
         * @return Non-null stream pointer
         * @throws std::runtime_error if no stream is available
         */
        void *requireStream(const char *kernel_name = "ROCmKernel") const
        {
            try
            {
                return getStream();
            }
            catch (const std::runtime_error &)
            {
                throw std::runtime_error(
                    std::string("[") + kernel_name +
                    "] No owned GPU stream is available. Bind an explicit "
                    "stream or a worker context before execution; stream zero "
                    "cannot participate in event-backed coherence.");
            }
        }

        /**
         * @brief Get the hipBLAS handle from the device context
         *
         * Returns the hipBLAS handle from the device context if one is bound,
         * otherwise returns nullptr. Kernels should fall back to their own
         * handle when this returns nullptr.
         *
         * @return hipblasHandle_t cast to void*, or nullptr
         */
        void *getBlasHandle() const
        {
            return device_ctx_ ? device_ctx_->blasHandle() : nullptr;
        }

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
        IWorkerGPUContext *device_ctx_ = nullptr;
        void *gpu_stream_ = nullptr; ///< Explicit GPU stream for kernel dispatch (nullptr = NOT SET)
    };

} // namespace llaminar2
