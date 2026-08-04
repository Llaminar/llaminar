/**
 * @file CUDAGraphCapture.h
 * @brief CUDA ownership, composition, replay, and metadata inspection for native graphs.
 *
 * The capture owns CUDA graph and executable handles but borrows one exact,
 * non-default execution stream. It also exposes read-only recursive kernel
 * inventory so orchestration diagnostics can attribute captured production work
 * without replaying a second path or involving device memory.
 */

#pragma once

#ifdef HAVE_CUDA

#include "../IGPUGraphCapture.h"
#include <cuda_runtime.h>

namespace llaminar2
{

    /// CUDA Graph capture/replay implementation for NVIDIA GPUs.
    ///
    /// Wraps cudaGraph_t / cudaGraphExec_t lifecycle. The stream is NOT owned;
    /// it must outlive this object. The CUDA ordinal is part of the capture's
    /// immutable ownership identity so capture, composition, replay, update,
    /// and destruction never depend on ambient thread-local CUDA state.
    class CUDAGraphCapture : public IGPUGraphCapture
    {
    public:
        /// @param stream The explicit CUDA stream to capture on. Must remain
        ///               valid for the lifetime of this object.
        /// @param device_ordinal CUDA device that owns @p stream and every graph
        ///                       resource composed into this capture.
        CUDAGraphCapture(cudaStream_t stream, int device_ordinal);
        ~CUDAGraphCapture() override;

        // Move-only (graph handles are not copyable)
        CUDAGraphCapture(CUDAGraphCapture &&other) noexcept;
        CUDAGraphCapture &operator=(CUDAGraphCapture &&other) noexcept;

        bool beginCapture() override;
        bool endCapture() override;
        bool instantiate() override;
        bool launch() override;
        [[nodiscard]] bool supportsDeviceControlledWhileLoop() const noexcept override
        {
#if CUDART_VERSION >= 12030
            return true;
#else
            return false;
#endif
        }
        [[nodiscard]] bool supportsDeviceControlledSwitchWhileLoop() const noexcept override
        {
#if CUDART_VERSION >= 13000
            return true;
#else
            return false;
#endif
        }
        using IGPUGraphCapture::buildDeviceControlledWhileLoop;
        bool buildDeviceControlledWhileLoop(
            std::span<const DeviceControlledLoopFragment> ordered_body_fragments,
            const DeviceControlledLoopPredicate &predicate) override;
        bool buildDeviceControlledSwitchWhileLoop(
            std::span<const DeviceControlledLoopBranch> branches,
            const DeviceControlledLoopPredicate &predicate,
            const DeviceControlledLoopSwitch &switch_policy) override;
        [[nodiscard]] void *executionStream() const noexcept override
        {
            return static_cast<void *>(stream_);
        }
        GraphUpdateResult tryUpdate() override;
        [[nodiscard]] bool supportsExecutableUpdate() const noexcept override { return true; }
        bool hasExecutable() const override;
        size_t nodeCount() const override;
        bool inspectKernelNodes(
            std::vector<GPUGraphKernelNodeInfo> &kernel_nodes,
            std::string *error = nullptr) const override;
        void reset() override;
        const char *backendName() const override { return "CUDA"; }

        /// @return The underlying CUDA graph (may be nullptr)
        cudaGraph_t graph() const { return graph_; }
        /// @return The underlying CUDA graph executable (may be nullptr)
        cudaGraphExec_t executable() const { return exec_; }
        /// @return The immutable CUDA device owning this capture.
        int deviceOrdinal() const noexcept { return device_ordinal_; }

    private:
        /**
         * @brief Select the immutable owner before using a CUDA graph resource.
         *
         * CUDA runtime device selection is thread-local. LocalTP submits graph
         * operations for several devices from one host thread, and same-stream
         * event edges are correctly elided. Consequently no neighbouring API
         * call can be relied upon to select this graph's device. Every graph
         * operation establishes its own context explicitly.
         */
        bool activateOwner(const char *operation) const noexcept;

        cudaStream_t stream_ = nullptr;       ///< Non-owned stream
        int device_ordinal_ = -1;             ///< Immutable owner of stream/graph
        cudaGraph_t graph_ = nullptr;         ///< Captured graph (owned)
        cudaGraphExec_t exec_ = nullptr;      ///< Instantiated executable (owned)
        size_t node_count_ = 0;               ///< Cached node count from last capture
    };

} // namespace llaminar2

#endif // HAVE_CUDA
