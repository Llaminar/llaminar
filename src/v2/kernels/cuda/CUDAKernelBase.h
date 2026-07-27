#pragma once

#include <stdexcept>

#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "utils/Logger.h"

namespace llaminar2
{

    /**
     * @brief Base class for CUDA kernels with default IWorkspaceConsumer support.
     *
     * Design Pattern: Dual-Mode Operation
     * - LEGACY MODE (hasWorkspace()=false): Kernel manages its own allocations
     * - MANAGED MODE (hasWorkspace()=true): Kernel uses pre-allocated workspace buffers
     *
     * Device Context Support (Phase 4):
     * - When device context is set, kernels can use shared cuBLAS handles and streams
     * - When no context, kernels manage their own handles (legacy behavior)
     *
     * Subclasses requiring workspace should:
     * 1. Override getWorkspaceRequirements() to declare buffer needs
     * 2. In execution methods, check hasWorkspace() and use getBuffer() if bound
     */
    class CUDAKernelBase : public IWorkspaceConsumer
    {
    public:
        virtual ~CUDAKernelBase() = default;

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
                LOG_TRACE("[CUDAKernelBase] Workspace bound with " << workspace_->bufferCount() << " buffers");
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
         * (cuBLAS, cuDNN, etc.) and streams from the context instead of creating
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
         * The default CUDA stream (stream 0) causes race conditions with
         * the executor's event-based coherence tracking.
         *
         * @param stream Opaque CUDA stream pointer (cudaStream_t cast to void*)
         */
        void setGPUStream(void *stream) { gpu_stream_ = stream; }

        /**
         * @brief Report whether a caller explicitly selected the launch stream.
         *
         * `getStream()` may return a worker-context default. Consumers that
         * establish transfer or graph ordering must distinguish that fallback
         * from a stream explicitly owned by the current transaction.
         *
         * @return true only when `setGPUStream()` received a non-null stream.
         */
        [[nodiscard]] bool hasExplicitGPUStream() const noexcept
        {
            return gpu_stream_ != nullptr;
        }

        /**
         * @brief Get the non-null GPU stream for kernel dispatch.
         *
         * Returns the explicitly bound stream, or the worker context's real
         * non-default stream when the kernel belongs to that context. A kernel
         * with neither source of stream ownership is not launchable: throwing
         * here makes the violation happen while C++ evaluates launch arguments,
         * before CUDA can enqueue work on stream zero.
         *
         * @return cudaStream_t cast to a non-null opaque pointer.
         * @throws std::runtime_error when no owned stream exists.
         */
        void *getStream() const
        {
            if (gpu_stream_)
                return gpu_stream_;
            if (device_ctx_)
            {
                void *stream = device_ctx_->defaultStream();
                if (stream)
                    return stream;
            }
            throw std::runtime_error(
                "[CUDAKernelBase] GPU execution requires an explicit or "
                "worker-context-owned non-null stream");
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
        void *requireStream(const char *kernel_name = "CUDAKernel") const
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
         * @brief Get the cuBLAS handle from the device context
         *
         * Returns the cuBLAS handle from the device context if one is bound,
         * otherwise returns nullptr. Kernels should fall back to their own
         * handle when this returns nullptr.
         *
         * @return cublasHandle_t cast to void*, or nullptr
         */
        void *getBlasHandle() const
        {
            return device_ctx_ ? device_ctx_->blasHandle() : nullptr;
        }

    protected:
        CUDAKernelBase() = default;

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
