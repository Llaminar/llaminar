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
        ///               lifetime of this object.
        /// @param device_ordinal Immutable ROCm device owning the stream/graph.
        HIPGraphCapture(hipStream_t stream, int device_ordinal);
        ~HIPGraphCapture() override;

        // Move-only (graph handles are not copyable)
        HIPGraphCapture(HIPGraphCapture &&other) noexcept;
        HIPGraphCapture &operator=(HIPGraphCapture &&other) noexcept;

        bool beginCapture() override;
        bool endCapture() override;
        bool instantiate() override;
        bool launch() override;
        [[nodiscard]] bool launchOnStream(void *stream) const override;
        bool buildOrderedTimelineTransaction(
            std::span<const GPUOrderedTimelineStep> ordered_steps,
            GPUOrderedTimelineInstrumentation instrumentation =
                GPUOrderedTimelineInstrumentation::Disabled) override;
        [[nodiscard]] GPUOrderedTimelineTimingSnapshot
        consumeOrderedTimelineTiming() override;
        [[nodiscard]] void *executionStream() const noexcept override
        {
            return static_cast<void *>(stream_);
        }
        GraphUpdateResult tryUpdate() override;
        [[nodiscard]] bool supportsExecutableUpdate() const noexcept override { return false; }
        bool hasExecutable() const override;
        [[nodiscard]] std::size_t residentMemoryBytes() const noexcept override
        {
            return resident_memory_bytes_;
        }
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
        /// @return Immutable ROCm ordinal owning this graph.
        int deviceOrdinal() const noexcept { return device_ordinal_; }

    private:
        /** @brief Select the immutable HIP owner before a runtime operation. */
        bool activateOwner(const char *operation) const noexcept;

        /** @brief One persistent event pair embedded around a timeline step. */
        struct OrderedTimelineTimingEvents
        {
            std::string name; ///< Stable step identity owned by this graph.
            GPUOrderedTimelineStepKind kind =
                GPUOrderedTimelineStepKind::CapturedFragment;
            hipEvent_t start = nullptr; ///< Recorded before the timed step.
            hipEvent_t stop = nullptr; ///< Recorded after the complete step frontier.
        };

        /** @brief Destroy every setup-owned timing event without touching graph work. */
        void destroyOrderedTimelineTimingEvents() noexcept;

        hipStream_t stream_ = nullptr;        ///< Non-owned stream
        int device_ordinal_ = -1;             ///< Immutable HIP resource owner.
        hipGraph_t graph_ = nullptr;          ///< Captured graph (owned)
        hipGraphExec_t exec_ = nullptr;       ///< Instantiated executable (owned)
        size_t node_count_ = 0;               ///< Cached node count from last capture
        std::size_t resident_memory_bytes_ = 0u; ///< Setup-observed opaque driver VRAM.
        std::vector<OrderedTimelineTimingEvents>
            ordered_timeline_timing_events_; ///< Empty on the uninstrumented path.
        mutable bool ordered_timeline_timing_pending_ = false; ///< Latest replay awaits non-blocking collection.
    };

} // namespace llaminar2

#endif // HAVE_ROCM
