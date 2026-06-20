#pragma once

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

    /**
     * @brief True when captured stage execution should also apply logical
     *        host-side bookkeeping.
     *
     * Stream capture records GPU work but does not execute it until launch.
     * Segmented decode capture skips replay callbacks on the immediate
     * launch-after-capture, so stateful stages still need to update host
     * metadata while recording for later stages in the same captured segment.
     * Prefill capture enables this for stateful stages whose later captured
     * stages need same-pass logical metadata, while still skipping duplicate
     * replay callbacks after the immediate launch-after-capture.
     */
    inline thread_local bool tls_graph_capture_host_bookkeeping = false;

    /// Query whether the current thread is recording into a GPU graph.
    inline bool isGraphCaptureActive() { return tls_graph_capture_active; }

    /// Query whether captured stage execution should update logical host state.
    inline bool isGraphCaptureHostBookkeepingActive() { return tls_graph_capture_host_bookkeeping; }

    /**
     * @brief RAII guard that sets the graph-capture-active flag for the
     *        duration of a capture recording window.
     */
    class GraphCaptureGuard
    {
    public:
        explicit GraphCaptureGuard(bool host_bookkeeping = false)
            : prev_(tls_graph_capture_active), prev_host_bookkeeping_(tls_graph_capture_host_bookkeeping)
        {
            tls_graph_capture_active = true;
            tls_graph_capture_host_bookkeeping = prev_host_bookkeeping_ || host_bookkeeping;
        }

        ~GraphCaptureGuard()
        {
            tls_graph_capture_host_bookkeeping = prev_host_bookkeeping_;
            tls_graph_capture_active = prev_;
        }

        // Non-copyable, non-movable
        GraphCaptureGuard(const GraphCaptureGuard &) = delete;
        GraphCaptureGuard &operator=(const GraphCaptureGuard &) = delete;

    private:
        bool prev_;                  ///< Previous graph-capture flag (for nested guard support)
        bool prev_host_bookkeeping_; ///< Previous logical bookkeeping flag
    };

} // namespace llaminar2
