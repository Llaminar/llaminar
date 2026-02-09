#pragma once

#ifdef HAVE_CUDA

#include "../IGPUGraphCapture.h"
#include <cuda_runtime.h>

namespace llaminar2
{

    /// CUDA Graph capture/replay implementation for NVIDIA GPUs.
    ///
    /// Wraps cudaGraph_t / cudaGraphExec_t lifecycle. The stream is NOT owned;
    /// it must outlive this object.
    class CUDAGraphCapture : public IGPUGraphCapture
    {
    public:
        /// @param stream The CUDA stream to capture on. Must remain valid for the
        ///               lifetime of this object. Typically from NvidiaDeviceContext::defaultStream().
        explicit CUDAGraphCapture(cudaStream_t stream);
        ~CUDAGraphCapture() override;

        // Move-only (graph handles are not copyable)
        CUDAGraphCapture(CUDAGraphCapture &&other) noexcept;
        CUDAGraphCapture &operator=(CUDAGraphCapture &&other) noexcept;

        bool beginCapture() override;
        bool endCapture() override;
        bool instantiate() override;
        bool launch() override;
        GraphUpdateResult tryUpdate() override;
        bool hasExecutable() const override;
        size_t nodeCount() const override;
        void reset() override;
        const char *backendName() const override { return "CUDA"; }

        /// @return The underlying CUDA graph (may be nullptr)
        cudaGraph_t graph() const { return graph_; }
        /// @return The underlying CUDA graph executable (may be nullptr)
        cudaGraphExec_t executable() const { return exec_; }

    private:
        cudaStream_t stream_ = nullptr;       ///< Non-owned stream
        cudaGraph_t graph_ = nullptr;         ///< Captured graph (owned)
        cudaGraphExec_t exec_ = nullptr;      ///< Instantiated executable (owned)
        size_t node_count_ = 0;               ///< Cached node count from last capture
        int consecutive_update_failures_ = 0; ///< Track failures for fallback heuristic
    };

} // namespace llaminar2

#endif // HAVE_CUDA
