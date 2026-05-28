/**
 * @file ROCmKernelProfiler.hip
 * @brief ROCm/HIP event-based kernel timing implementation
 * @author David Sanftenberg
 *
 * Implements HIP event-based timing for accurate GPU kernel profiling.
 */

#include "ROCmKernelProfiler.h"
#include <hip/hip_runtime.h>
#include <cstdio>

namespace llaminar2
{

    // ========================================================================
    // ScopedROCmKernelTimer Implementation
    // ========================================================================

    ScopedROCmKernelTimer::ScopedROCmKernelTimer(ROCmKernelType type, hipStream_t stream)
        : type_(type), enabled_(ROCmKernelProfiler::isEnabled()),
          start_event_(nullptr), stop_event_(nullptr), stream_(stream)
    {
        if (enabled_)
        {
            // Skip event timing if the stream is currently in graph capture mode.
            // hipEventRecord/hipEventSynchronize are not permitted on capturing
            // streams and would poison the capture context, causing subsequent
            // kernel launches to fail.
            hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
            if (stream_ && hipStreamIsCapturing(stream_, &capture_status) == hipSuccess &&
                capture_status != hipStreamCaptureStatusNone)
            {
                enabled_ = false;
                return;
            }

            hipError_t err;
            err = hipEventCreate(&start_event_);
            if (err != hipSuccess)
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to create start event: %s\n",
                        hipGetErrorString(err));
                enabled_ = false;
                return;
            }

            err = hipEventCreate(&stop_event_);
            if (err != hipSuccess)
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to create stop event: %s\n",
                        hipGetErrorString(err));
                hipEventDestroy(start_event_);
                enabled_ = false;
                return;
            }

            // Record start event on the stream
            err = hipEventRecord(start_event_, stream_);
            if (err != hipSuccess)
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to record start event: %s\n",
                        hipGetErrorString(err));
                hipEventDestroy(start_event_);
                hipEventDestroy(stop_event_);
                enabled_ = false;
            }
        }
    }

    ScopedROCmKernelTimer::~ScopedROCmKernelTimer()
    {
        if (enabled_)
        {
            // Record stop event
            hipError_t err = hipEventRecord(stop_event_, stream_);
            if (err != hipSuccess)
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to record stop event: %s\n",
                        hipGetErrorString(err));
                hipEventDestroy(start_event_);
                hipEventDestroy(stop_event_);
                return;
            }

            // Synchronize on stop event to get accurate timing
            err = hipEventSynchronize(stop_event_);
            if (err != hipSuccess)
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to synchronize stop event: %s\n",
                        hipGetErrorString(err));
                hipEventDestroy(start_event_);
                hipEventDestroy(stop_event_);
                return;
            }

            // Calculate elapsed time in milliseconds
            float elapsed_ms = 0.0f;
            err = hipEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
            if (err == hipSuccess)
            {
                // Convert to microseconds and record
                ROCmKernelProfiler::record(type_, static_cast<double>(elapsed_ms) * 1000.0);
            }
            else
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to get elapsed time: %s\n",
                        hipGetErrorString(err));
            }

            // Clean up events
            hipEventDestroy(start_event_);
            hipEventDestroy(stop_event_);
        }
    }

    // ========================================================================
    // ManualROCmKernelTimer Implementation
    // ========================================================================

    ManualROCmKernelTimer::ManualROCmKernelTimer()
        : enabled_(ROCmKernelProfiler::isEnabled()), started_(false),
          start_event_(nullptr), stop_event_(nullptr), stream_(nullptr)
    {
        if (enabled_)
        {
            hipError_t err;
            err = hipEventCreate(&start_event_);
            if (err != hipSuccess)
            {
                enabled_ = false;
                return;
            }

            err = hipEventCreate(&stop_event_);
            if (err != hipSuccess)
            {
                hipEventDestroy(start_event_);
                enabled_ = false;
                return;
            }
        }
    }

    ManualROCmKernelTimer::~ManualROCmKernelTimer()
    {
        if (enabled_)
        {
            if (start_event_)
                hipEventDestroy(start_event_);
            if (stop_event_)
                hipEventDestroy(stop_event_);
        }
    }

    void ManualROCmKernelTimer::begin(hipStream_t stream)
    {
        if (enabled_)
        {
            // Skip if the stream is in graph capture mode — event timing is
            // incompatible with stream capture.
            hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
            if (stream && hipStreamIsCapturing(stream, &capture_status) == hipSuccess &&
                capture_status != hipStreamCaptureStatusNone)
            {
                return;
            }

            stream_ = stream;
            hipError_t err = hipEventRecord(start_event_, stream_);
            if (err == hipSuccess)
            {
                started_ = true;
            }
            else
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to record start event: %s\n",
                        hipGetErrorString(err));
            }
        }
    }

    void ManualROCmKernelTimer::end(ROCmKernelType type)
    {
        if (enabled_ && started_)
        {
            hipError_t err = hipEventRecord(stop_event_, stream_);
            if (err != hipSuccess)
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to record stop event: %s\n",
                        hipGetErrorString(err));
                return;
            }

            err = hipEventSynchronize(stop_event_);
            if (err != hipSuccess)
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to synchronize stop event: %s\n",
                        hipGetErrorString(err));
                return;
            }

            float elapsed_ms = 0.0f;
            err = hipEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
            if (err == hipSuccess)
            {
                ROCmKernelProfiler::record(type, static_cast<double>(elapsed_ms) * 1000.0);
            }
            else
            {
                fprintf(stderr, "[ROCmKernelProfiler] Failed to get elapsed time: %s\n",
                        hipGetErrorString(err));
            }

            started_ = false;
        }
    }

} // namespace llaminar2

