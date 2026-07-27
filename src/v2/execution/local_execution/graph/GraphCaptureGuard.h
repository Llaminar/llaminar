#pragma once

#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../utils/Logger.h"

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Thread-local flag indicating that the current thread is inside
     *        a HIP/CUDA graph capture recording window.
     *
     * During graph capture (between beginCapture/endCapture), many HIP/CUDA
     * operations are illegal:
     *   - hipDeviceSynchronize / cudaDeviceSynchronize
     *   - hipStreamSynchronize / cudaStreamSynchronize (on capture stream)
     *   - hipMemcpy (synchronous variants)
     *   - hipEventSynchronize
     *
     * Code that might call these operations (e.g., TensorBase::ensureOnDevice)
     * can check isGraphCaptureActive() and take a fast path that avoids sync.
     *
     * Usage:
     *   // In capture controller (between beginCapture/endCapture):
     *   {
     *       GraphCaptureGuard guard;  // sets flag true
     *       for (auto& stage : stages)
     *           stage->execute(ctx);
     *   }  // guard destructor sets flag false
     *
     *   // In tensor code:
     *   if (isGraphCaptureActive() && device_valid_)
     *       return true;  // skip sync, data already on device from warmup
     */

    /// Thread-local flag: true when inside a graph capture recording window.
    inline thread_local bool tls_graph_capture_active = false;

    /// Query whether the current thread is recording into a GPU graph.
    inline bool isGraphCaptureActive() { return tls_graph_capture_active; }

    /**
     * @brief RAII guard that sets the graph-capture-active flag for the
     *        duration of a capture recording window.
     */
    class GraphCaptureGuard
    {
    public:
        GraphCaptureGuard()
            : prev_(tls_graph_capture_active)
        {
            tls_graph_capture_active = true;
        }

        ~GraphCaptureGuard()
        {
            tls_graph_capture_active = prev_;
        }

        // Non-copyable, non-movable
        GraphCaptureGuard(const GraphCaptureGuard &) = delete;
        GraphCaptureGuard &operator=(const GraphCaptureGuard &) = delete;

    private:
        bool prev_; ///< Previous graph-capture flag (for nested guard support).
    };

    /**
     * @brief Own one exact HIP/CUDA backend stream-capture transaction.
     *
     * A capture interval is indivisible: after `beginCapture()` succeeds,
     * exactly one `endCapture()` must execute before the stream can be queried,
     * synchronized, destroyed, or reused. Keeping the worker-context flag,
     * thread-local guard, and backend graph owner in separate call-site code
     * allowed early returns to strand a native stream in capture mode.
     *
     * This owner makes that lifecycle structural. `finish()` closes the normal
     * interval, while the destructor closes one abandoned by an exception or
     * early return. Backend `endCapture()` failure is process-fatal because the
     * native stream's ownership state is then unknowable; eager recovery or a
     * stream synchronization would itself be illegal.
     */
    class ScopedBackendGraphCapture final
    {
    public:
        /**
         * @brief Bind a software and backend capture owner.
         *
         * @param gpu_context Worker context whose capture-active state protects
         *        backend operations on this device.
         * @param capture Backend graph object bound to the exact native stream.
         * @param operation Stable diagnostic name for fatal lifecycle failures.
         */
        ScopedBackendGraphCapture(
            IWorkerGPUContext &gpu_context,
            IGPUGraphCapture &capture,
            std::string operation)
            : gpu_context_(&gpu_context),
              capture_(capture),
              operation_(std::move(operation))
        {
        }

        /**
         * @brief Bind a backend capture when no worker-context flag is available.
         *
         * Legacy single-graph callers still receive the same structural
         * begin/end pairing and thread-local protection. New execution paths
         * should pass the worker context through the primary constructor.
         */
        ScopedBackendGraphCapture(
            IGPUGraphCapture &capture,
            std::string operation)
            : capture_(capture),
              operation_(std::move(operation))
        {
        }

        ~ScopedBackendGraphCapture()
        {
            if (!active_)
                return;

            clearSoftwareCaptureState();
            active_ = false;
            if (!capture_.endCapture())
            {
                LOG_ERROR(
                    "[ScopedBackendGraphCapture] Fatal failure closing "
                    "abandoned backend graph capture for "
                    << operation_);
                std::terminate();
            }
        }

        ScopedBackendGraphCapture(
            const ScopedBackendGraphCapture &) = delete;
        ScopedBackendGraphCapture &operator=(
            const ScopedBackendGraphCapture &) = delete;

        /**
         * @brief Enter backend stream capture and publish software state.
         *
         * @return true after the backend accepted capture; false before any
         *         capture interval became active.
         */
        bool begin()
        {
            if (active_)
                throw std::logic_error(
                    "ScopedBackendGraphCapture cannot begin twice");

            if (gpu_context_)
            {
                gpu_context_->setGraphCaptureActive(true);
            }
            if (!capture_.beginCapture())
            {
                if (gpu_context_)
                    gpu_context_->setGraphCaptureActive(false);
                return false;
            }

            capture_guard_.emplace();
            active_ = true;
            return true;
        }

        /**
         * @brief Close the exact backend capture interval.
         *
         * Backend failure leaves the native stream in an unknowable state, so
         * this method terminates instead of exposing a recoverable result.
         */
        void finish()
        {
            if (!active_)
                throw std::logic_error(
                    "ScopedBackendGraphCapture cannot finish an inactive capture");

            clearSoftwareCaptureState();
            active_ = false;
            if (!capture_.endCapture())
            {
                LOG_ERROR(
                    "[ScopedBackendGraphCapture] Fatal backend endCapture "
                    "failure for "
                    << operation_);
                std::terminate();
            }
        }

    private:
        void clearSoftwareCaptureState() noexcept
        {
            capture_guard_.reset();
            if (gpu_context_)
                gpu_context_->setGraphCaptureActive(false);
        }

        IWorkerGPUContext *gpu_context_ = nullptr;
        IGPUGraphCapture &capture_;
        std::string operation_;
        std::optional<GraphCaptureGuard> capture_guard_;
        bool active_ = false;
    };

} // namespace llaminar2
