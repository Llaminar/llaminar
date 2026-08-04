/**
 * @file HIPGraphCapture.h
 * @brief HIP ownership, replay, and metadata inspection for native GPU graphs.
 *
 * The capture borrows one exact HIP stream and owns its graph/executable
 * handles. Read-only recursive kernel inspection mirrors the CUDA contract so
 * graph-level performance diagnostics remain backend-symmetric.
 */

#pragma once

#ifdef HAVE_ROCM

#include "../IGPUGraphCapture.h"
#include <hip/hip_runtime.h>

namespace llaminar2
{

    /// HIP Graph capture/replay implementation for AMD ROCm GPUs.
    ///
    /// Wraps hipGraph_t / hipGraphExec_t lifecycle. The stream is NOT owned;
    /// it must outlive this object.
    class HIPGraphCapture : public IGPUGraphCapture
    {
    public:
        /// @param stream The HIP stream to capture on. Must remain valid for the
        ///               lifetime of this object. Typically from AMDDeviceContext::defaultStream().
        explicit HIPGraphCapture(hipStream_t stream);
        ~HIPGraphCapture() override;

        // Move-only (graph handles are not copyable)
        HIPGraphCapture(HIPGraphCapture &&other) noexcept;
        HIPGraphCapture &operator=(HIPGraphCapture &&other) noexcept;

        bool beginCapture() override;
        bool endCapture() override;
        bool instantiate() override;
        bool launch() override;
        [[nodiscard]] void *executionStream() const noexcept override
        {
            return static_cast<void *>(stream_);
        }
        GraphUpdateResult tryUpdate() override;
        [[nodiscard]] bool supportsExecutableUpdate() const noexcept override { return false; }
        bool hasExecutable() const override;
        size_t nodeCount() const override;
        bool inspectKernelNodes(
            std::vector<GPUGraphKernelNodeInfo> &kernel_nodes,
            std::string *error = nullptr) const override;
        void reset() override;
        const char *backendName() const override { return "HIP"; }

        /// @return The underlying HIP graph (may be nullptr)
        hipGraph_t graph() const { return graph_; }
        /// @return The underlying HIP graph executable (may be nullptr)
        hipGraphExec_t executable() const { return exec_; }

    private:
        hipStream_t stream_ = nullptr;        ///< Non-owned stream
        hipGraph_t graph_ = nullptr;          ///< Captured graph (owned)
        hipGraphExec_t exec_ = nullptr;       ///< Instantiated executable (owned)
        size_t node_count_ = 0;               ///< Cached node count from last capture
    };

} // namespace llaminar2

#endif // HAVE_ROCM
